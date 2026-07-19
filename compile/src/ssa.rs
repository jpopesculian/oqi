//! Static Single Assignment (SSA) form over [`crate::cfg::Cfg`].
//!
//! Given a [`Cfg`], `build_cfg` produces an [`SsaCfg`] where every classical
//! variable read references a specific [`SsaValue`] (a `(SymbolId, version)`
//! pair) and every write produces a fresh version. Control-flow merges that
//! join two distinct definitions of the same variable get a [`Phi`] node at
//! the top of the joined block.
//!
//! Qubits, gate references, calibration ports, and other non-data symbols
//! are not versioned (they pass through as their original [`SymbolId`]).
//! Indexed writes use whole-array semantics: every `a[i] = x` produces a
//! fresh version of `a` as a whole.
//!
//! Nested CFGs inside `BlockStmtKind::Box`, `BlockStmtKind::Cal`, and
//! `BlockExprKind::DurationOf` are SSA-converted independently. Outer-scope
//! reads inside a nested CFG resolve to version 0 (the implicit "entry
//! value").
//!
//! The construction is *maximal* SSA: phis are placed on the iterated
//! dominance frontier of each variable's def sites with no liveness
//! pruning, so a phi may exist for a variable that is dead at the join.
//! Consumers (e.g. phi elimination in the bytecode backend) must
//! tolerate — or strip — dead phis.

use std::collections::{HashMap, HashSet};

use oqi_lex::Span;

use crate::cfg::{
    BasicBlockId, BlockCalibrationBody, BlockExpr, BlockExprKind, BlockStmt, BlockStmtKind, Cfg,
    CfgOwner, ProgramCfgs, Terminator,
};
use crate::classical::Primitive;
use crate::sir::{
    Alias, Annotation, ArrayLiteral, Assignment, Binary, Call, Cast, Delay, GateCall, GateModifier,
    Index, IndexItem, IndexKind, IndexOp, LValue, Measure, MeasureExpr, MeasureExprKind,
    QubitOperand, RValue, RangeExpr, SwitchLabels, Unary,
};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::Type;

// ── SSA value identity ───────────────────────────────────────────────

/// A versioned reference to a symbol. Version 0 represents the value
/// flowing into the CFG (parameter, externally-visible state, etc.);
/// versions ≥ 1 are assigned by [`build_cfg`] to each fresh def.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaValue {
    pub symbol: SymbolId,
    pub version: u32,
}

#[derive(Clone)]
pub struct Phi {
    pub dest: SsaValue,
    pub ty: Type,
    /// One entry per predecessor of the containing block. The pair is
    /// `(predecessor_block, value_at_predecessor_exit)`. Order matches
    /// the dom-tree-DFS order in which predecessors were visited and
    /// is not stable across builds — consumers should match by
    /// `predecessor_block` rather than positionally.
    pub sources: Vec<(BasicBlockId, SsaValue)>,
    pub span: Span,
}

// ── CFG / block ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SsaCfg {
    pub blocks: Vec<SsaBlock>,
    pub entry: BasicBlockId,
    pub exit: BasicBlockId,
    pub owner: CfgOwner,
    /// Reaching definition of each live symbol at the `exit` block —
    /// i.e. the SSA value holding each variable's final value when the
    /// body falls off the end. Empty when `exit` is unreachable (the
    /// body ends with an explicit `return`/`end`). Used by the bytecode
    /// emitter to map named program outputs to registers.
    pub exit_defs: HashMap<SymbolId, SsaValue>,
}

#[derive(Clone)]
pub struct SsaBlock {
    pub id: BasicBlockId,
    pub phis: Vec<Phi>,
    pub stmts: Vec<SsaStmt>,
    pub terminator: SsaTerminator,
    pub span: Span,
}

// ── Expressions ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SsaExpr {
    pub kind: SsaExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone)]
pub enum SsaExprKind {
    Literal(Primitive),
    Var(SsaValue),
    HardwareQubit(usize),
    Binary(Binary<SsaExpr>),
    Unary(Unary<SsaExpr>),
    Cast(Cast<SsaExpr>),
    Index(Index<SsaExpr>),
    Call(Call<SsaExpr>),
    DurationOf(SsaCfg),
    ArrayLiteral(ArrayLiteral<SsaExpr>),
}

// ── Statements ───────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SsaStmt {
    pub kind: SsaStmtKind,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Clone)]
pub enum SsaStmtKind {
    Alias(Alias<SsaExpr>),
    GateCall(GateCall<SsaExpr>),
    Measure(SsaMeasure),
    Reset(QubitOperand<SsaExpr>),
    Barrier(Vec<QubitOperand<SsaExpr>>),
    Delay(Delay<SsaExpr>),
    Box(SsaBoxStmt),
    Assignment(SsaAssignment),
    Pragma(String),
    Cal(SsaCalibrationBody),
    ExprStmt(SsaExpr),
    Nop(Vec<QubitOperand<SsaExpr>>),
}

/// An SSA destination. Scalar assignments produce a single fresh
/// version. Indexed writes carry both the version read (`old`) and the
/// version produced (`new`), reflecting whole-array kill+def.
#[derive(Clone)]
pub enum SsaLValue {
    Var(SsaValue),
    Indexed {
        old: SsaValue,
        new: SsaValue,
        indices: Vec<IndexOp<SsaExpr>>,
    },
}

#[derive(Clone)]
pub struct SsaAssignment {
    pub target: SsaLValue,
    pub value: RValue<SsaExpr>,
}

#[derive(Clone)]
pub struct SsaMeasure {
    pub measure: MeasureExpr<SsaExpr>,
    pub target: Option<SsaLValue>,
}

#[derive(Clone)]
pub struct SsaBoxStmt {
    pub duration: Option<SsaExpr>,
    pub body: SsaCfg,
}

#[derive(Clone)]
pub enum SsaCalibrationBody {
    Opaque(String),
    OpenPulse(SsaCfg),
}

// ── Terminator ───────────────────────────────────────────────────────

#[derive(Clone)]
pub enum SsaTerminator {
    Goto(BasicBlockId),
    Branch {
        cond: SsaExpr,
        then_bb: BasicBlockId,
        else_bb: BasicBlockId,
    },
    Switch {
        target: SsaExpr,
        cases: Vec<(SwitchLabels<SsaExpr>, BasicBlockId)>,
        default: Option<BasicBlockId>,
    },
    Return(Option<RValue<SsaExpr>>),
    End,
    Unreachable,
}

// ── Program-level ────────────────────────────────────────────────────

pub struct ProgramSsa {
    pub top_level: SsaCfg,
    pub subroutines: Vec<SsaCfg>,
    pub gates: Vec<SsaCfg>,
    /// Parallel to `ProgramCfgs::calibrations`. `None` mirrors opaque defcals.
    pub calibrations: Vec<Option<SsaCfg>>,
}

// ── Entry points ─────────────────────────────────────────────────────

pub fn build_program(cfgs: &ProgramCfgs, symbols: &SymbolTable) -> ProgramSsa {
    let top_level = build_cfg(&cfgs.top_level, symbols);
    let subroutines = cfgs
        .subroutines
        .iter()
        .map(|c| build_cfg(c, symbols))
        .collect();
    let gates = cfgs.gates.iter().map(|c| build_cfg(c, symbols)).collect();
    let calibrations = cfgs
        .calibrations
        .iter()
        .map(|c| c.as_ref().map(|c| build_cfg(c, symbols)))
        .collect();
    ProgramSsa {
        top_level,
        subroutines,
        gates,
        calibrations,
    }
}

pub fn build_cfg(cfg: &Cfg, symbols: &SymbolTable) -> SsaCfg {
    let preds = predecessors(cfg);
    let idom = dominators(cfg, &preds);
    let df = dominance_frontiers(cfg, &idom, &preds);
    let defs = collect_defs(cfg, symbols);
    let phi_placement = place_phis(&defs, &df);
    rename(cfg, &idom, &phi_placement, symbols)
}

// ── Helpers: SSA candidacy ───────────────────────────────────────────

fn is_ssa_candidate(symbols: &SymbolTable, sym: SymbolId) -> bool {
    matches!(
        symbols.get(sym).kind,
        SymbolKind::Variable
            | SymbolKind::Input
            | SymbolKind::Output
            | SymbolKind::SubroutineParam
            | SymbolKind::GateParam
            | SymbolKind::LoopVar
            | SymbolKind::Temp
    )
}

// ── Predecessors ─────────────────────────────────────────────────────

fn predecessors(cfg: &Cfg) -> Vec<Vec<BasicBlockId>> {
    let mut preds = vec![Vec::new(); cfg.blocks.len()];
    for bb in &cfg.blocks {
        for s in bb.terminator.successors() {
            preds[s.0].push(bb.id);
        }
    }
    preds
}

// ── Reverse postorder ────────────────────────────────────────────────

/// Iterative DFS producing reverse-postorder of blocks reachable from
/// `cfg.entry`. Used by the dominator computation.
fn reverse_postorder(cfg: &Cfg) -> Vec<BasicBlockId> {
    let n = cfg.blocks.len();
    let mut visited = vec![false; n];
    let mut postorder = Vec::new();

    // Iterative postorder: stack carries (block, successors,
    // child_index_to_visit_next). Successors are collected once per
    // block — re-iterating them per child would be quadratic for
    // switches.
    let succs = |bb: BasicBlockId| -> Vec<BasicBlockId> {
        cfg.blocks[bb.0].terminator.successors().collect()
    };
    let mut stack: Vec<(BasicBlockId, Vec<BasicBlockId>, usize)> = Vec::new();
    visited[cfg.entry.0] = true;
    stack.push((cfg.entry, succs(cfg.entry), 0));

    loop {
        let child = {
            let Some((_, children, next_child_ix)) = stack.last_mut() else {
                break;
            };
            let child = children.get(*next_child_ix).copied();
            if child.is_some() {
                *next_child_ix += 1;
            }
            child
        };
        match child {
            Some(child) => {
                if !visited[child.0] {
                    visited[child.0] = true;
                    stack.push((child, succs(child), 0));
                }
            }
            None => {
                let (bb, ..) = stack.pop().unwrap();
                postorder.push(bb);
            }
        }
    }

    postorder.reverse();
    postorder
}

// ── Dominators (Cooper-Harvey-Kennedy) ───────────────────────────────

const UNDEFINED: u32 = u32::MAX;

/// Compute the immediate-dominator map: `idom[bb] = idom_block`. Blocks
/// unreachable from `cfg.entry` get `idom = bb` (self-dominating
/// sentinel) and should be filtered out of subsequent phases.
fn dominators(cfg: &Cfg, preds: &[Vec<BasicBlockId>]) -> Vec<BasicBlockId> {
    let n = cfg.blocks.len();
    let rpo = reverse_postorder(cfg);

    // rpo_index[bb] = position of bb in rpo; UNDEFINED if unreachable.
    let mut rpo_index = vec![UNDEFINED; n];
    for (i, bb) in rpo.iter().enumerate() {
        rpo_index[bb.0] = i as u32;
    }

    // idom[bb] = bb.0 as u32, or UNDEFINED if not yet computed.
    let mut idom = vec![UNDEFINED; n];
    idom[cfg.entry.0] = cfg.entry.0 as u32;

    let mut changed = true;
    while changed {
        changed = false;
        // Visit blocks in reverse-postorder (skip entry, which we've fixed).
        for &bb in rpo.iter().skip(1) {
            // Find first processed predecessor.
            let mut new_idom: Option<u32> = None;
            for p in &preds[bb.0] {
                if rpo_index[p.0] == UNDEFINED {
                    continue; // unreachable predecessor — ignore
                }
                if idom[p.0] != UNDEFINED {
                    new_idom = Some(p.0 as u32);
                    break;
                }
            }
            let Some(mut new_idom) = new_idom else {
                continue;
            };
            // Intersect with the rest.
            for p in &preds[bb.0] {
                if p.0 as u32 == new_idom {
                    continue;
                }
                if rpo_index[p.0] == UNDEFINED || idom[p.0] == UNDEFINED {
                    continue;
                }
                new_idom = intersect(p.0 as u32, new_idom, &idom, &rpo_index);
            }
            if idom[bb.0] != new_idom {
                idom[bb.0] = new_idom;
                changed = true;
            }
        }
    }

    // Materialize: for any unreachable block, point idom at itself so
    // downstream code doesn't dereference UNDEFINED.
    (0..n)
        .map(|i| {
            if idom[i] == UNDEFINED {
                BasicBlockId(i)
            } else {
                BasicBlockId(idom[i] as usize)
            }
        })
        .collect()
}

fn intersect(mut b1: u32, mut b2: u32, idom: &[u32], rpo_index: &[u32]) -> u32 {
    while b1 != b2 {
        while rpo_index[b1 as usize] > rpo_index[b2 as usize] {
            b1 = idom[b1 as usize];
        }
        while rpo_index[b2 as usize] > rpo_index[b1 as usize] {
            b2 = idom[b2 as usize];
        }
    }
    b1
}

// ── Dominance frontiers ──────────────────────────────────────────────

fn dominance_frontiers(
    cfg: &Cfg,
    idom: &[BasicBlockId],
    preds: &[Vec<BasicBlockId>],
) -> Vec<HashSet<BasicBlockId>> {
    let n = cfg.blocks.len();
    let mut df: Vec<HashSet<BasicBlockId>> = vec![HashSet::new(); n];
    for bb_idx in 0..n {
        let bb = BasicBlockId(bb_idx);
        if preds[bb_idx].len() < 2 {
            continue;
        }
        for &p in &preds[bb_idx] {
            let mut runner = p;
            while runner != idom[bb_idx] && runner != bb {
                df[runner.0].insert(bb);
                let next = idom[runner.0];
                if next == runner {
                    // Self-dom sentinel for unreachable; bail.
                    break;
                }
                runner = next;
            }
        }
    }
    df
}

// ── Defs collection ──────────────────────────────────────────────────

/// For each SSA-candidate symbol, the set of blocks containing a def.
fn collect_defs(cfg: &Cfg, symbols: &SymbolTable) -> HashMap<SymbolId, HashSet<BasicBlockId>> {
    let mut defs: HashMap<SymbolId, HashSet<BasicBlockId>> = HashMap::new();
    for bb in &cfg.blocks {
        for stmt in &bb.stmts {
            if let Some(sym) = stmt_def(stmt)
                && is_ssa_candidate(symbols, sym)
            {
                defs.entry(sym).or_default().insert(bb.id);
            }
        }
    }
    defs
}

/// The symbol (if any) defined by a `BlockStmt`. Whole-array semantics
/// for indexed writes: `a[i] = ...` is a def of `a`. Reads don't count.
/// Every variant either defines exactly one symbol or none, so the
/// return type is `Option`.
fn stmt_def(stmt: &BlockStmt) -> Option<SymbolId> {
    match &stmt.kind {
        BlockStmtKind::Assignment(Assignment { target, .. }) => Some(lvalue_def(target)),
        BlockStmtKind::Measure(Measure {
            target: Some(lv), ..
        }) => Some(lvalue_def(lv)),
        _ => None,
    }
}

fn lvalue_def(lv: &LValue<BlockExpr>) -> SymbolId {
    match lv {
        LValue::Var(s) => *s,
        LValue::Indexed { symbol, .. } => *symbol,
    }
}

// ── Phi placement (iterated dominance frontier) ──────────────────────

fn place_phis(
    defs: &HashMap<SymbolId, HashSet<BasicBlockId>>,
    df: &[HashSet<BasicBlockId>],
) -> Vec<HashSet<SymbolId>> {
    let mut placement: Vec<HashSet<SymbolId>> = (0..df.len()).map(|_| HashSet::new()).collect();
    for (sym, def_blocks) in defs {
        let mut worklist: Vec<BasicBlockId> = def_blocks.iter().copied().collect();
        let mut placed: HashSet<BasicBlockId> = HashSet::new();
        while let Some(bb) = worklist.pop() {
            for &c in &df[bb.0] {
                if placed.insert(c) {
                    placement[c.0].insert(*sym);
                    // The block where we placed a phi now has a "def" of
                    // sym too, so its DF needs to propagate further.
                    if !def_blocks.contains(&c) {
                        worklist.push(c);
                    }
                }
            }
        }
    }
    placement
}

// ── Rename pass ──────────────────────────────────────────────────────

struct Renamer<'a> {
    symbols: &'a SymbolTable,
    /// For each symbol: `(next_version_to_allocate, live_version_stack)`.
    /// `next_version` starts at 1 (0 is the implicit "entry value");
    /// the stack holds versions currently live in the dom-tree DFS.
    versions: HashMap<SymbolId, (u32, Vec<u32>)>,
}

impl<'a> Renamer<'a> {
    fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            versions: HashMap::new(),
        }
    }

    fn alloc(&mut self, sym: SymbolId) -> SsaValue {
        let entry = self.versions.entry(sym).or_insert((1, Vec::new()));
        let version = entry.0;
        entry.0 += 1;
        entry.1.push(version);
        SsaValue {
            symbol: sym,
            version,
        }
    }

    /// Snapshot the reaching definition of every symbol with a live
    /// version on its stack. Symbols whose only value is the implicit
    /// entry value (version 0) are omitted.
    fn snapshot_live(&self) -> HashMap<SymbolId, SsaValue> {
        self.versions
            .iter()
            .filter_map(|(&symbol, (_, stack))| {
                stack
                    .last()
                    .map(|&version| (symbol, SsaValue { symbol, version }))
            })
            .collect()
    }

    /// Current SSA name for a read of `sym`. If `sym` has no defs in
    /// scope (or is a non-candidate), returns version 0.
    fn read(&self, sym: SymbolId) -> SsaValue {
        let version = self
            .versions
            .get(&sym)
            .and_then(|(_, stack)| stack.last())
            .copied()
            .unwrap_or(0);
        SsaValue {
            symbol: sym,
            version,
        }
    }

    /// Push a previously allocated version onto the live stack without
    /// advancing the version counter. Used by the rename DFS to replay
    /// phi destinations that were allocated up front in the pre-pass.
    /// For *fresh* defs use [`Renamer::alloc`].
    fn push(&mut self, sym: SymbolId, version: u32) {
        self.versions
            .entry(sym)
            .or_insert((1, Vec::new()))
            .1
            .push(version);
    }

    fn pop(&mut self, sym: SymbolId) {
        let popped = self
            .versions
            .get_mut(&sym)
            .and_then(|(_, stack)| stack.pop());
        debug_assert!(
            popped.is_some(),
            "pop on empty version stack for symbol {sym:?} — push/pop accounting is wrong"
        );
    }

    /// Debug-only: assert every live version stack is empty. Used at
    /// the end of the dom-tree DFS as a sanity check.
    #[cfg(debug_assertions)]
    fn assert_all_stacks_empty(&self) {
        for (sym, (_, stack)) in &self.versions {
            debug_assert!(
                stack.is_empty(),
                "rename DFS left {} versions on the stack for {sym:?}",
                stack.len()
            );
        }
    }
}

fn rename(
    cfg: &Cfg,
    idom: &[BasicBlockId],
    phi_placement: &[HashSet<SymbolId>],
    symbols: &SymbolTable,
) -> SsaCfg {
    let n = cfg.blocks.len();
    let mut renamer = Renamer::new(symbols);

    // ── Phi allocation pass ─────────────────────────────────────────
    //
    // Allocate each phi's dest version up front so that, during the
    // dom-tree DFS, predecessors visiting a join block already see the
    // versions that the phi dests own.
    //
    // `phis_by_block[bb]` holds the phi nodes for `bb`, in a stable
    // SymbolId-sorted order.
    let mut phis_by_block: Vec<Vec<Phi>> = (0..n).map(|_| Vec::new()).collect();
    for (bb_idx, syms) in phi_placement.iter().enumerate() {
        let mut syms_sorted: Vec<SymbolId> = syms.iter().copied().collect();
        syms_sorted.sort_by_key(|s| s.0);
        for sym in syms_sorted {
            let dest = renamer.alloc(sym);
            phis_by_block[bb_idx].push(Phi {
                dest,
                ty: symbols.get(sym).ty.clone(),
                sources: Vec::new(),
                span: cfg.blocks[bb_idx].span,
            });
        }
    }
    // The phi alloc pass left versions pushed; pop them so the DFS
    // re-establishes them per-block in the correct scope.
    for phis in &phis_by_block {
        for phi in phis {
            renamer.pop(phi.dest.symbol);
        }
    }

    // ── Build dominator children ────────────────────────────────────
    let mut dom_children: Vec<Vec<BasicBlockId>> = vec![Vec::new(); n];
    for (i, parent) in idom.iter().enumerate() {
        let bb = BasicBlockId(i);
        if *parent != bb {
            dom_children[parent.0].push(bb);
        }
    }
    for kids in &mut dom_children {
        kids.sort_by_key(|b| b.0);
    }

    let mut ctx = RenameCtx {
        cfg,
        dom_children,
        phis_by_block,
        out_blocks: (0..n).map(|_| None).collect(),
        exit_defs: HashMap::new(),
    };
    ctx.rename_block(cfg.entry, &mut renamer);

    // Invariant: the dom-tree DFS pushes and pops in matched pairs, so
    // by the time the recursion unwinds every live version stack
    // should be empty. Any leftover entries indicate a bug in
    // push/pop accounting inside `rename_block`.
    #[cfg(debug_assertions)]
    renamer.assert_all_stacks_empty();

    // ── Materialize SsaCfg ──────────────────────────────────────────
    //
    // Reachable blocks have a `Some(SsaBlock)` slot from the DFS but
    // their `phis` field is still empty — splice in the now-complete
    // phi vectors. Unreachable blocks (still `None`) get a placeholder
    // lowered through the *same* `renamer` (which is back to empty
    // stacks after the DFS unwound) so any def allocations advance the
    // shared `next_version` counter, preserving SSA-name uniqueness.
    // Defs are popped after each block so its versions don't leak into
    // later unreachable blocks.
    let RenameCtx {
        mut phis_by_block,
        mut out_blocks,
        exit_defs,
        ..
    } = ctx;
    let blocks: Vec<SsaBlock> = (0..n)
        .map(|i| match out_blocks[i].take() {
            Some(mut b) => {
                b.phis = std::mem::take(&mut phis_by_block[i]);
                b
            }
            None => {
                let bb = &cfg.blocks[i];
                let mut pushed: Vec<SymbolId> = Vec::new();
                let stmts: Vec<SsaStmt> = bb
                    .stmts
                    .iter()
                    .map(|s| {
                        let (stmt, def) = lower_block_stmt_with_defs(s, &mut renamer);
                        if let Some(sym) = def {
                            pushed.push(sym);
                        }
                        stmt
                    })
                    .collect();
                let terminator = lower_terminator(&bb.terminator, &renamer);
                for sym in pushed.into_iter().rev() {
                    renamer.pop(sym);
                }
                SsaBlock {
                    id: bb.id,
                    phis: std::mem::take(&mut phis_by_block[i]),
                    stmts,
                    terminator,
                    span: bb.span,
                }
            }
        })
        .collect();

    SsaCfg {
        blocks,
        entry: cfg.entry,
        exit: cfg.exit,
        owner: cfg.owner.clone(),
        exit_defs,
    }
}

/// Mutable state threaded through the dom-tree DFS of [`rename`].
struct RenameCtx<'a> {
    cfg: &'a Cfg,
    dom_children: Vec<Vec<BasicBlockId>>,
    phis_by_block: Vec<Vec<Phi>>,
    out_blocks: Vec<Option<SsaBlock>>,
    /// Reaching defs snapshotted when the DFS visits `cfg.exit`.
    exit_defs: HashMap<SymbolId, SsaValue>,
}

impl RenameCtx<'_> {
    fn rename_block(&mut self, bb: BasicBlockId, renamer: &mut Renamer<'_>) {
        // One entry per push into the version stack; popped in reverse on
        // return so each symbol's stack returns to its pre-block state.
        let mut pushed: Vec<SymbolId> = Vec::new();

        for phi in &self.phis_by_block[bb.0] {
            renamer.push(phi.dest.symbol, phi.dest.version);
            pushed.push(phi.dest.symbol);
        }

        // At the (synthetic, statement-free) exit block, the live
        // version of each symbol after its phis is exactly that symbol's
        // final value when the body falls off the end.
        if bb == self.cfg.exit {
            self.exit_defs = renamer.snapshot_live();
        }

        let src_block = &self.cfg.blocks[bb.0];
        let mut out_stmts: Vec<SsaStmt> = Vec::with_capacity(src_block.stmts.len());
        for stmt in &src_block.stmts {
            let (lowered, def) = lower_block_stmt_with_defs(stmt, renamer);
            if let Some(sym) = def {
                pushed.push(sym);
            }
            out_stmts.push(lowered);
        }

        let term = lower_terminator(&src_block.terminator, renamer);

        // Fill phi sources at each successor with the live versions at
        // block exit. Done in place so a self-edge (succ == bb) works.
        for succ in src_block.terminator.successors() {
            for phi in self.phis_by_block[succ.0].iter_mut() {
                let value = renamer.read(phi.dest.symbol);
                phi.sources.push((bb, value));
            }
        }

        // Materialize the output block *without* its phis — we'll
        // splice the phis in after the full DFS completes, since
        // predecessors visiting later (e.g. the back-edge of a
        // `while`) still need to push into `phis_by_block[bb.0]` after
        // we've finished here.
        self.out_blocks[bb.0] = Some(SsaBlock {
            id: src_block.id,
            phis: Vec::new(),
            stmts: out_stmts,
            terminator: term,
            span: src_block.span,
        });

        // Recurse. `mem::take` releases the borrow on self.dom_children
        // without cloning; the slot is never read again.
        let children = std::mem::take(&mut self.dom_children[bb.0]);
        for child in &children {
            self.rename_block(*child, renamer);
        }

        // Pop in reverse to restore the pre-block stack state.
        for sym in pushed.into_iter().rev() {
            renamer.pop(sym);
        }
    }
}

// ── Lowering helpers ─────────────────────────────────────────────────

/// Lower a `BlockStmt`, allocating a fresh version on the (at most one)
/// def and returning that symbol so the caller can record it for
/// stack-unwind accounting.
fn lower_block_stmt_with_defs(
    stmt: &BlockStmt,
    r: &mut Renamer<'_>,
) -> (SsaStmt, Option<SymbolId>) {
    let mut def: Option<SymbolId> = None;
    let kind = match &stmt.kind {
        BlockStmtKind::Alias(a) => SsaStmtKind::Alias(Alias {
            symbol: a.symbol,
            value: a.value.iter().map(|e| lower_expr(e, r)).collect(),
        }),
        BlockStmtKind::GateCall(g) => SsaStmtKind::GateCall(GateCall {
            gate: g.gate,
            modifiers: g
                .modifiers
                .iter()
                .map(|m| lower_gate_modifier(m, r))
                .collect(),
            args: g.args.iter().map(|e| lower_expr(e, r)).collect(),
            qubits: g.qubits.iter().map(|q| lower_qubit_operand(q, r)).collect(),
            duration: None,
        }),
        BlockStmtKind::Measure(m) => {
            let measure = lower_measure_expr(&m.measure, r);
            let target = m.target.as_ref().map(|lv| {
                let (lowered, sym) = lower_lvalue_for_def(lv, r);
                if is_ssa_candidate(r.symbols, sym) {
                    def = Some(sym);
                }
                lowered
            });
            SsaStmtKind::Measure(SsaMeasure { measure, target })
        }
        BlockStmtKind::Reset(q) => SsaStmtKind::Reset(lower_qubit_operand(q, r)),
        BlockStmtKind::Barrier(qs) => {
            SsaStmtKind::Barrier(qs.iter().map(|q| lower_qubit_operand(q, r)).collect())
        }
        BlockStmtKind::Delay(d) => SsaStmtKind::Delay(Delay {
            duration: lower_expr(&d.duration, r),
            operands: d
                .operands
                .iter()
                .map(|q| lower_qubit_operand(q, r))
                .collect(),
        }),
        BlockStmtKind::Box(b) => SsaStmtKind::Box(SsaBoxStmt {
            duration: b.duration.as_ref().map(|e| lower_expr(e, r)),
            // Inner CFG is SSA-converted independently.
            body: build_cfg(&b.body, r.symbols),
        }),
        BlockStmtKind::Assignment(a) => {
            // RHS reads BEFORE the LHS allocates a fresh version.
            let value = lower_rvalue(&a.value, r);
            let (target, sym) = lower_lvalue_for_def(&a.target, r);
            if is_ssa_candidate(r.symbols, sym) {
                def = Some(sym);
            }
            SsaStmtKind::Assignment(SsaAssignment { target, value })
        }
        BlockStmtKind::Pragma(s) => SsaStmtKind::Pragma(s.clone()),
        BlockStmtKind::Cal(c) => match c {
            BlockCalibrationBody::Opaque(s) => {
                SsaStmtKind::Cal(SsaCalibrationBody::Opaque(s.clone()))
            }
            BlockCalibrationBody::OpenPulse(inner) => {
                SsaStmtKind::Cal(SsaCalibrationBody::OpenPulse(build_cfg(inner, r.symbols)))
            }
        },
        BlockStmtKind::ExprStmt(e) => SsaStmtKind::ExprStmt(lower_expr(e, r)),
        BlockStmtKind::Nop(qs) => {
            SsaStmtKind::Nop(qs.iter().map(|q| lower_qubit_operand(q, r)).collect())
        }
    };
    (
        SsaStmt {
            kind,
            annotations: stmt.annotations.clone(),
            span: stmt.span,
        },
        def,
    )
}

fn lower_lvalue_for_def(lv: &LValue<BlockExpr>, r: &mut Renamer<'_>) -> (SsaLValue, SymbolId) {
    match lv {
        LValue::Var(s) => {
            let dest = if is_ssa_candidate(r.symbols, *s) {
                r.alloc(*s)
            } else {
                SsaValue {
                    symbol: *s,
                    version: 0,
                }
            };
            (SsaLValue::Var(dest), *s)
        }
        LValue::Indexed { symbol, indices } => {
            let old = r.read(*symbol);
            let lowered_indices: Vec<IndexOp<SsaExpr>> =
                indices.iter().map(|i| lower_index_op(i, r)).collect();
            let new = if is_ssa_candidate(r.symbols, *symbol) {
                r.alloc(*symbol)
            } else {
                SsaValue {
                    symbol: *symbol,
                    version: 0,
                }
            };
            (
                SsaLValue::Indexed {
                    old,
                    new,
                    indices: lowered_indices,
                },
                *symbol,
            )
        }
    }
}

fn lower_terminator(term: &Terminator, r: &Renamer<'_>) -> SsaTerminator {
    match term {
        Terminator::Goto(t) => SsaTerminator::Goto(*t),
        Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } => SsaTerminator::Branch {
            cond: lower_expr(cond, r),
            then_bb: *then_bb,
            else_bb: *else_bb,
        },
        Terminator::Switch {
            target,
            cases,
            default,
        } => SsaTerminator::Switch {
            target: lower_expr(target, r),
            cases: cases
                .iter()
                .map(|(labels, bb)| (lower_switch_labels(labels, r), *bb))
                .collect(),
            default: *default,
        },
        Terminator::Return(rv) => SsaTerminator::Return(rv.as_ref().map(|v| lower_rvalue(v, r))),
        Terminator::End => SsaTerminator::End,
        Terminator::Unreachable => SsaTerminator::Unreachable,
    }
}

fn lower_switch_labels(labels: &SwitchLabels<BlockExpr>, r: &Renamer<'_>) -> SwitchLabels<SsaExpr> {
    match labels {
        SwitchLabels::Values(vs) => {
            SwitchLabels::Values(vs.iter().map(|e| lower_expr(e, r)).collect())
        }
        SwitchLabels::Default => SwitchLabels::Default,
    }
}

// ── Expression / operand lowering (pure: never mutates `Renamer`) ────

fn lower_expr(e: &BlockExpr, r: &Renamer<'_>) -> SsaExpr {
    let kind = match &e.kind {
        BlockExprKind::Literal(p) => SsaExprKind::Literal(p.clone()),
        BlockExprKind::Var(s) => {
            let value = if is_ssa_candidate(r.symbols, *s) {
                r.read(*s)
            } else {
                SsaValue {
                    symbol: *s,
                    version: 0,
                }
            };
            SsaExprKind::Var(value)
        }
        BlockExprKind::HardwareQubit(n) => SsaExprKind::HardwareQubit(*n),
        BlockExprKind::Binary(b) => SsaExprKind::Binary(Binary {
            op: b.op,
            left: Box::new(lower_expr(&b.left, r)),
            right: Box::new(lower_expr(&b.right, r)),
        }),
        BlockExprKind::Unary(u) => SsaExprKind::Unary(Unary {
            op: u.op,
            operand: Box::new(lower_expr(&u.operand, r)),
        }),
        BlockExprKind::Cast(c) => SsaExprKind::Cast(Cast {
            target_ty: c.target_ty.clone(),
            operand: Box::new(lower_expr(&c.operand, r)),
        }),
        BlockExprKind::Index(i) => SsaExprKind::Index(Index {
            base: Box::new(lower_expr(&i.base, r)),
            index: lower_index_op(&i.index, r),
        }),
        BlockExprKind::Call(c) => SsaExprKind::Call(Call {
            callee: c.callee.clone(),
            args: c.args.iter().map(|a| lower_expr(a, r)).collect(),
        }),
        BlockExprKind::DurationOf(inner) => SsaExprKind::DurationOf(build_cfg(inner, r.symbols)),
        BlockExprKind::ArrayLiteral(a) => SsaExprKind::ArrayLiteral(ArrayLiteral {
            items: a.items.iter().map(|x| lower_expr(x, r)).collect(),
            span: a.span,
        }),
    };
    SsaExpr {
        kind,
        ty: e.ty.clone(),
        span: e.span,
    }
}

fn lower_qubit_operand(q: &QubitOperand<BlockExpr>, r: &Renamer<'_>) -> QubitOperand<SsaExpr> {
    match q {
        QubitOperand::Indexed { symbol, indices } => QubitOperand::Indexed {
            symbol: *symbol,
            indices: indices.iter().map(|i| lower_index_op(i, r)).collect(),
        },
        QubitOperand::Hardware(n) => QubitOperand::Hardware(*n),
    }
}

fn lower_index_op(io: &IndexOp<BlockExpr>, r: &Renamer<'_>) -> IndexOp<SsaExpr> {
    IndexOp {
        kind: match &io.kind {
            IndexKind::Set(es) => IndexKind::Set(es.iter().map(|e| lower_expr(e, r)).collect()),
            IndexKind::Items(items) => IndexKind::Items(
                items
                    .iter()
                    .map(|it| match it {
                        IndexItem::Single(e) => IndexItem::Single(Box::new(lower_expr(e, r))),
                        IndexItem::Range(rng) => IndexItem::Range(RangeExpr {
                            start: rng.start.as_ref().map(|e| Box::new(lower_expr(e, r))),
                            step: rng.step.as_ref().map(|e| Box::new(lower_expr(e, r))),
                            end: rng.end.as_ref().map(|e| Box::new(lower_expr(e, r))),
                        }),
                    })
                    .collect(),
            ),
        },
        span: io.span,
    }
}

fn lower_gate_modifier(m: &GateModifier<BlockExpr>, r: &Renamer<'_>) -> GateModifier<SsaExpr> {
    match m {
        GateModifier::Inv => GateModifier::Inv,
        GateModifier::Pow(e) => GateModifier::Pow(Box::new(lower_expr(e, r))),
        GateModifier::Ctrl(n) => GateModifier::Ctrl(*n),
        GateModifier::NegCtrl(n) => GateModifier::NegCtrl(*n),
    }
}

fn lower_measure_expr(m: &MeasureExpr<BlockExpr>, r: &Renamer<'_>) -> MeasureExpr<SsaExpr> {
    MeasureExpr {
        kind: match &m.kind {
            MeasureExprKind::Measure { operand } => MeasureExprKind::Measure {
                operand: lower_qubit_operand(operand, r),
            },
            MeasureExprKind::QuantumCall {
                callee,
                args,
                qubits,
            } => MeasureExprKind::QuantumCall {
                callee: *callee,
                args: args.iter().map(|e| lower_expr(e, r)).collect(),
                qubits: qubits.iter().map(|q| lower_qubit_operand(q, r)).collect(),
            },
        },
        ty: m.ty.clone(),
        span: m.span,
    }
}

fn lower_rvalue(rv: &RValue<BlockExpr>, r: &Renamer<'_>) -> RValue<SsaExpr> {
    match rv {
        RValue::Expr(e) => RValue::Expr(Box::new(lower_expr(e, r))),
        RValue::Measure(m) => RValue::Measure(Box::new(lower_measure_expr(m, r))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg as cfg_mod;
    use crate::lower::compile_source;
    use crate::resolve::DefaultIncludeResolver;

    fn compile(src: &str) -> crate::sir::Program {
        compile_source(src, DefaultIncludeResolver, None).expect("compile should succeed")
    }

    fn build(src: &str) -> (crate::sir::Program, SsaCfg) {
        let p = compile(src);
        let cfgs = cfg_mod::build_program(&p).expect("CFG build should succeed");
        let ssa = build_cfg(&cfgs.top_level, &p.symbols);
        (p, ssa)
    }

    fn count_total_phis(ssa: &SsaCfg) -> usize {
        ssa.blocks.iter().map(|b| b.phis.len()).sum()
    }

    fn lookup_symbol(program: &crate::sir::Program, name: &str) -> SymbolId {
        program.symbols.lookup(name).expect("symbol should exist")
    }

    #[test]
    fn straight_line_no_phis() {
        let (_, ssa) = build(
            r#"
                int x = 1;
                int y = x + 1;
            "#,
        );
        assert_eq!(count_total_phis(&ssa), 0);
    }

    #[test]
    fn if_else_merge_inserts_phi() {
        let (program, ssa) = build(
            r#"
                int c = 0;
                int x = 0;
                if (c == 0) {
                    x = 1;
                } else {
                    x = 2;
                }
                int y = x;
            "#,
        );
        let x_sym = lookup_symbol(&program, "x");
        let phi_for_x: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == x_sym)
            .collect();
        assert_eq!(
            phi_for_x.len(),
            1,
            "expected exactly one phi for `x` at the merge block"
        );
        // The phi must have exactly two sources: one from the then-branch
        // and one from the else-branch.
        assert_eq!(
            phi_for_x[0].sources.len(),
            2,
            "if/else merge phi must have exactly 2 sources"
        );
        // Both sources should refer to versions of x that come from the
        // pre-merge writes (versions ≥ 1).
        for (_pred, value) in &phi_for_x[0].sources {
            assert_eq!(value.symbol, x_sym);
            assert!(
                value.version >= 1,
                "x phi source should be a versioned def, got version {}",
                value.version
            );
        }
        // The two sources should be distinct versions (the two branches
        // wrote different values).
        assert_ne!(
            phi_for_x[0].sources[0].1.version, phi_for_x[0].sources[1].1.version,
            "then and else versions of x must differ"
        );
    }

    #[test]
    fn while_header_phi_has_two_sources() {
        // Regression: a previous bug dropped the back-edge phi source
        // when the body block visited after the header materialized.
        let (program, ssa) = build(
            r#"
                int i = 0;
                while (i < 10) {
                    i += 1;
                }
            "#,
        );
        let i_sym = lookup_symbol(&program, "i");
        let phi_for_i: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == i_sym)
            .collect();
        assert_eq!(
            phi_for_i.len(),
            1,
            "expected exactly one phi for `i` at the while header"
        );
        assert_eq!(
            phi_for_i[0].sources.len(),
            2,
            "while header phi must have both pre-header init and back-edge advance as sources"
        );
        // The two source versions must be distinct (init def vs. advance def).
        let v0 = phi_for_i[0].sources[0].1.version;
        let v1 = phi_for_i[0].sources[1].1.version;
        assert_ne!(v0, v1, "pre-header and back-edge versions must be distinct");
    }

    #[test]
    fn for_range_header_phi_has_two_sources() {
        let (program, ssa) = build(
            r#"
                int sum = 0;
                for int i in [0:10] {
                    sum += i;
                }
            "#,
        );
        let i_sym = lookup_symbol(&program, "i");
        let sum_sym = lookup_symbol(&program, "sum");
        let i_phis: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == i_sym)
            .collect();
        let sum_phis: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == sum_sym)
            .collect();
        assert_eq!(i_phis.len(), 1, "expected one phi for loop var `i`");
        assert_eq!(
            i_phis[0].sources.len(),
            2,
            "for-loop var phi must merge init + advance"
        );
        assert_eq!(sum_phis.len(), 1, "expected one phi for accumulator `sum`");
        assert_eq!(
            sum_phis[0].sources.len(),
            2,
            "for-loop accumulator phi must merge pre-loop + body"
        );
    }

    #[test]
    fn switch_merge_has_phi_per_case() {
        let (program, ssa) = build(
            r#"
                int c = 1;
                int x = 0;
                switch (c) {
                    case 1 { x = 10; }
                    case 2 { x = 20; }
                    default { x = 99; }
                }
                int y = x;
            "#,
        );
        let x_sym = lookup_symbol(&program, "x");
        let phi_for_x: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == x_sym)
            .collect();
        assert_eq!(
            phi_for_x.len(),
            1,
            "expected exactly one phi for `x` at the switch merge"
        );
        // Three cases + default = 3 distinct paths into the merge block.
        assert_eq!(
            phi_for_x[0].sources.len(),
            3,
            "switch merge phi must have one source per case (2 values + default)"
        );
    }

    #[test]
    fn break_inside_loop_does_not_break_phi() {
        // Verifies the rename pass tolerates `break` cleanly: the after
        // block still receives a phi if the loop has any def downstream
        // that survives both the natural fallthrough and the break edge.
        let (program, ssa) = build(
            r#"
                int i = 0;
                int last = 0;
                while (i < 10) {
                    last = i;
                    if (i == 5) {
                        break;
                    }
                    i += 1;
                }
            "#,
        );
        // `last` should have a phi at the merge of break + natural exit.
        // (The exact arity depends on CFG shape; assert at least 1 phi
        // exists and all sources reference versioned defs of `last`.)
        let last_sym = lookup_symbol(&program, "last");
        let last_phis: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == last_sym)
            .collect();
        assert!(
            !last_phis.is_empty(),
            "expected ≥1 phi for `last` somewhere in the CFG"
        );
        for phi in &last_phis {
            for (_pred, val) in &phi.sources {
                assert_eq!(val.symbol, last_sym);
            }
        }
    }

    #[test]
    fn continue_routes_back_to_header() {
        // `continue` jumps to the loop's continue target; the header phi
        // for `i` should still see all defs reaching the header.
        let (program, ssa) = build(
            r#"
                int i = 0;
                while (i < 10) {
                    if (i == 2) {
                        i += 100;
                        continue;
                    }
                    i += 1;
                }
            "#,
        );
        let i_sym = lookup_symbol(&program, "i");
        let i_phis: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == i_sym)
            .collect();
        assert_eq!(i_phis.len(), 1, "expected one header phi for `i`");
        // Pre-header def + one or more back-edges (continue + body fallthrough).
        assert!(
            i_phis[0].sources.len() >= 2,
            "while header phi for `i` should have ≥2 sources, got {}",
            i_phis[0].sources.len()
        );
    }

    #[test]
    fn build_program_runs_over_fixture() {
        // End-to-end smoke test: build_program over a real fixture must
        // succeed and produce SSA for every CFG it owns.
        let src = include_str!("../../fixtures/qasm/adder.qasm");
        let program = compile_source(src, DefaultIncludeResolver, None)
            .expect("adder.qasm should compile to SIR");
        let cfgs = cfg_mod::build_program(&program).expect("CFG build");
        let ssa = build_program(&cfgs, &program.symbols);

        // Top-level CFG should have at least one phi (the program uses
        // for-loops, so a loop-var phi is expected somewhere).
        let total_phis: usize = ssa.top_level.blocks.iter().map(|b| b.phis.len()).sum();
        assert!(
            total_phis > 0,
            "adder.qasm top-level should have ≥1 phi (for-loop variable)"
        );

        // Every subroutine, gate, and OpenPulse calibration should have
        // produced an SsaCfg with at least an entry and exit block.
        for sub in &ssa.subroutines {
            assert!(
                sub.blocks.len() >= 2,
                "subroutine CFG should have ≥2 blocks"
            );
        }
        for gate in &ssa.gates {
            assert!(gate.blocks.len() >= 2, "gate CFG should have ≥2 blocks");
        }
    }

    #[test]
    fn defs_have_increasing_versions() {
        let (_, ssa) = build(
            r#"
                int x = 1;
                x = 2;
                x = 3;
            "#,
        );
        // No control flow → no phis, but three defs of x with distinct
        // versions ≥ 1.
        let mut versions = Vec::new();
        for block in &ssa.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Assignment(SsaAssignment {
                    target: SsaLValue::Var(v),
                    ..
                }) = &stmt.kind
                {
                    versions.push(v.version);
                }
            }
        }
        assert_eq!(versions.len(), 3);
        let mut sorted = versions.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "versions must be distinct");
        assert!(sorted.iter().all(|&v| v >= 1));
    }

    #[test]
    fn qubits_not_versioned() {
        let (program, ssa) = build(
            r#"
                include "stdgates.inc";
                qubit q;
                bit b;
                reset q;
                h q;
                b = measure q;
            "#,
        );
        // No phis on q (qubit) but possibly on b (bit / Variable).
        let q_sym = lookup_symbol(&program, "q");
        let q_phis: Vec<&Phi> = ssa
            .blocks
            .iter()
            .flat_map(|b| b.phis.iter())
            .filter(|p| p.dest.symbol == q_sym)
            .collect();
        assert!(q_phis.is_empty(), "qubits should not get SSA phis");

        // All reads of `q` in the SSA should still carry a value with
        // symbol == q_sym and version 0 (unversioned).
        let mut found_q_read = false;
        for block in &ssa.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Reset(QubitOperand::Indexed { symbol, .. }) = &stmt.kind
                    && *symbol == q_sym
                {
                    found_q_read = true;
                }
            }
        }
        assert!(found_q_read, "expected at least one reset of q");
    }

    #[test]
    fn indexed_assignment_versions_whole_array() {
        let (program, ssa) = build(
            r#"
                array[int[32], 4] a = {0, 0, 0, 0};
                a[0] = 1;
                a[1] = 2;
            "#,
        );
        let a_sym = lookup_symbol(&program, "a");
        // Find both indexed assignments: each should produce an SsaLValue::Indexed
        // with old.version < new.version.
        let mut indexed_targets: Vec<(SsaValue, SsaValue)> = Vec::new();
        for block in &ssa.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Assignment(SsaAssignment {
                    target: SsaLValue::Indexed { old, new, .. },
                    ..
                }) = &stmt.kind
                    && old.symbol == a_sym
                {
                    indexed_targets.push((*old, *new));
                }
            }
        }
        assert!(
            indexed_targets.len() >= 2,
            "expected ≥2 indexed assignments to `a`"
        );
        // Sort by new version and check monotonicity.
        indexed_targets.sort_by_key(|(_, n)| n.version);
        let v0 = indexed_targets[0];
        let v1 = indexed_targets[1];
        assert!(
            v0.1.version < v1.1.version,
            "versions must monotonically increase"
        );
        // The second indexed assignment's `old` should equal the first's `new`
        // (it reads what the previous write produced).
        assert_eq!(v1.0, v0.1, "second a[..] write must read first's def");
    }

    #[test]
    fn nested_box_independent_ssa() {
        let (_, ssa) = build(
            r#"
                include "stdgates.inc";
                qubit q;
                box {
                    int y = 2;
                    y = y + 1;
                }
            "#,
        );
        // The outer SSA should still version x, and the box body should
        // contain its own nested SsaCfg (via SsaBoxStmt) with versioned
        // defs of y.
        let mut box_count = 0;
        for block in &ssa.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Box(b) = &stmt.kind {
                    box_count += 1;
                    // Body is itself an SsaCfg — check it has at least
                    // one Assignment with a versioned dest.
                    let mut found_versioned = false;
                    for inner_block in &b.body.blocks {
                        for inner_stmt in &inner_block.stmts {
                            if let SsaStmtKind::Assignment(SsaAssignment {
                                target: SsaLValue::Var(v),
                                ..
                            }) = &inner_stmt.kind
                                && v.version >= 1
                            {
                                found_versioned = true;
                            }
                        }
                    }
                    assert!(found_versioned, "box body should have SSA-versioned defs");
                }
            }
        }
        assert_eq!(box_count, 1, "expected exactly one Box block stmt");
    }

    #[test]
    fn subroutine_param_starts_at_version_zero() {
        let (program, _ssa) = build(
            r#"
                def add_one(int x) -> int {
                    return x + 1;
                }
                int y = add_one(5);
            "#,
        );
        // Build subroutine SSA separately.
        let cfgs = cfg_mod::build_program(&program).expect("CFG build");
        let sub_ssa = build_cfg(&cfgs.subroutines[0], &program.symbols);
        // Find a read of `x` inside the subroutine — should be version 0.
        let x_sym = lookup_symbol(&program, "x");
        let mut found_x_read = false;
        for block in &sub_ssa.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Assignment(_) = &stmt.kind {
                    // skip
                }
            }
            // The return terminator reads x.
            if let SsaTerminator::Return(Some(RValue::Expr(e))) = &block.terminator
                && let SsaExprKind::Binary(b) = &e.kind
                && let SsaExprKind::Var(v) = &b.left.kind
                && v.symbol == x_sym
            {
                assert_eq!(v.version, 0, "subroutine param x should be version 0");
                found_x_read = true;
            }
        }
        assert!(
            found_x_read,
            "expected to find a read of x in the subroutine body"
        );
    }

    #[test]
    fn dead_block_defs_do_not_leak_into_later_dead_blocks() {
        // Each loop body has dead code after `break`, landing in its
        // own unreachable block. The def of `x` in the first dead
        // block must not leak into the renamer state seen by the
        // second dead block: the read of `x` there has no dominating
        // def, so it must resolve to version 0.
        let (program, ssa) = build(
            r#"
                int x = 0;
                while (x < 10) {
                    break;
                    x = 1;
                }
                while (x < 10) {
                    break;
                    x = x + 2;
                }
            "#,
        );
        let x_sym = lookup_symbol(&program, "x");
        let mut found_dead_read = false;
        for block in &ssa.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Assignment(a) = &stmt.kind
                    && let RValue::Expr(e) = &a.value
                    && let SsaExprKind::Binary(b) = &e.kind
                    && let SsaExprKind::Var(v) = &b.left.kind
                    && v.symbol == x_sym
                {
                    assert_eq!(
                        v.version, 0,
                        "read of x in an unreachable block should be version 0"
                    );
                    found_dead_read = true;
                }
            }
        }
        assert!(
            found_dead_read,
            "expected to find the dead `x = x + 2` read"
        );
    }
}
