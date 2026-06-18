//! Global quantum memory layout.
//!
//! OpenQASM qubits are all globally scoped with compile-time-constant
//! sizes (docs/types.rst), so every named quantum register can be
//! statically allocated into one flat global quantum memory. This
//! module builds that layout from a [`sir::Program`] and resolves
//! register references — including `let` aliases built from slices,
//! discrete index sets, and `++` concatenation — to global qubit
//! indices via [`oqi_quantum::QuantumMemory`].
//!
//! The bytecode emitter consumes the layout to rewrite symbolic qubit
//! operands into global references; the `oqi_quantum` types never
//! cross the serialization boundary.
//!
//! Qubit parameters of gates and subroutines cannot be resolved
//! statically (they bind at call time); the layout maps them to
//! positional slots instead. Physical qubits (`$N`) live in a separate
//! hardware namespace and are not part of the layout.

use std::collections::{HashMap, HashSet};

use oqi_quantum::{QuantumMemory, QuantumRegister};

use crate::classical::Primitive;
use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{self, IndexItem, IndexKind, IndexOp, ParamPassing, RangeExpr};
use crate::ssa::{SsaExpr, SsaExprKind};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::Type;

/// Static layout of all named quantum registers in global memory.
pub struct QubitLayout {
    memory: QuantumMemory,
    /// Global `SymbolKind::Qubit` declarations, allocated in
    /// declaration (`SymbolId`) order.
    registers: HashMap<SymbolId, QuantumRegister>,
    /// Resolved qubit `let` aliases; populated during bytecode
    /// emission, in block order.
    aliases: HashMap<SymbolId, QuantumRegister>,
    /// Gate qubit params and qubit-typed subroutine params, mapped to
    /// their positional slot (gate: position in the declared qubit
    /// list; subroutine: position in the full parameter list).
    param_slots: HashMap<SymbolId, u32>,
    /// Classical parameters of each gate/subroutine, keyed by the
    /// gate/subroutine symbol, in declaration order. Used by the
    /// bytecode emitter to record the calling convention.
    classical_params: HashMap<SymbolId, Vec<SymbolId>>,
}

/// Allocate every global qubit declaration into a fresh global memory
/// and record the parameter slots of gate/subroutine qubit params.
pub fn build_layout(program: &sir::Program) -> QubitLayout {
    let mut memory = QuantumMemory::new();
    let mut registers = HashMap::new();
    for sym in program.symbols.iter() {
        if sym.kind != SymbolKind::Qubit || sym.scope.is_some() {
            continue;
        }
        let size = match sym.ty {
            Type::Qubit => 1,
            Type::QubitReg(n) => n,
            _ => continue,
        };
        registers.insert(sym.id, memory.alloc(size));
    }

    let mut param_slots = HashMap::new();
    let mut classical_params = HashMap::new();
    for gate in &program.gates {
        for (slot, q) in gate.qubits.iter().enumerate() {
            param_slots.insert(*q, slot as u32);
        }
        // Every declared gate parameter is classical (gates take only
        // qubits and classical angle-like params).
        classical_params.insert(gate.symbol, gate.params.clone());
    }
    for sub in &program.subroutines {
        let mut classical = Vec::new();
        for (slot, p) in sub.params.iter().enumerate() {
            if matches!(p.passing, ParamPassing::QubitRef) {
                param_slots.insert(p.symbol, slot as u32);
            } else {
                classical.push(p.symbol);
            }
        }
        classical_params.insert(sub.symbol, classical);
    }

    QubitLayout {
        memory,
        registers,
        aliases: HashMap::new(),
        param_slots,
        classical_params,
    }
}

impl QubitLayout {
    /// Total size of global quantum memory.
    pub fn num_qubits(&self) -> usize {
        self.memory.size()
    }

    /// The register for a declared qubit symbol or a resolved alias.
    pub fn register_of(&self, sym: SymbolId) -> Option<&QuantumRegister> {
        self.registers.get(&sym).or_else(|| self.aliases.get(&sym))
    }

    /// Positional slot of a gate/subroutine qubit parameter.
    pub fn param_slot(&self, sym: SymbolId) -> Option<u32> {
        self.param_slots.get(&sym).copied()
    }

    /// Classical parameters of a gate/subroutine, in declaration order.
    pub fn classical_params(&self, sym: SymbolId) -> &[SymbolId] {
        self.classical_params
            .get(&sym)
            .map_or(&[], |v| v.as_slice())
    }

    pub fn define_alias(&mut self, sym: SymbolId, reg: QuantumRegister) {
        self.aliases.insert(sym, reg);
    }

    /// Global memory index of the qubit at logical position `local`.
    pub fn global_index(&self, reg: &QuantumRegister, local: usize) -> Option<usize> {
        self.memory.get(reg).get(local)
    }

    /// Global index ranges of `reg`, with adjacent ranges merged.
    pub fn global_ranges(&self, reg: &QuantumRegister) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = Vec::new();
        for r in self.memory.get(reg).ranges() {
            if r.is_empty() {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.1 == r.start as u32 => last.1 = r.end as u32,
                _ => out.push((r.start as u32, r.end as u32)),
            }
        }
        out
    }

    /// Resolve a `let` alias value (one expression per `++` operand)
    /// into a register. Returns `Ok(None)` when the first operand is
    /// not a quantum reference at all — the alias is classical and the
    /// caller keeps its existing behavior.
    pub fn resolve_alias_value(
        &self,
        exprs: &[SsaExpr],
        symbols: &SymbolTable,
    ) -> Result<Option<QuantumRegister>> {
        let mut iter = exprs.iter();
        let Some(first) = iter.next() else {
            return Ok(None);
        };
        let Some(mut combined) = self.resolve_alias_operand(first, symbols)? else {
            return Ok(None);
        };
        for expr in iter {
            let Some(reg) = self.resolve_alias_operand(expr, symbols)? else {
                return Err(CompileError::new(ErrorKind::Unsupported(
                    "cannot mix quantum and classical operands in an alias".into(),
                ))
                .with_span(expr.span));
            };
            combined = combined.concat(reg);
        }

        // "A register cannot be concatenated with any part of itself"
        // (docs/types.rst) — reject overlapping global indices.
        let mut seen = HashSet::new();
        for g in self.memory.get(&combined).iter() {
            if !seen.insert(g) {
                return Err(CompileError::new(ErrorKind::InvalidContext(
                    "a quantum register cannot be concatenated with any part of itself".into(),
                ))
                .with_span(first.span));
            }
        }
        Ok(Some(combined))
    }

    /// Resolve a single alias operand to a register, or `Ok(None)` if
    /// it is not a quantum reference.
    fn resolve_alias_operand(
        &self,
        expr: &SsaExpr,
        symbols: &SymbolTable,
    ) -> Result<Option<QuantumRegister>> {
        if matches!(expr.kind, SsaExprKind::HardwareQubit(_)) {
            return Err(CompileError::new(ErrorKind::Unsupported(
                "physical qubits cannot be aliased".into(),
            ))
            .with_span(expr.span));
        }
        let Some((sym, ops)) = peel_index_chain(expr) else {
            return Ok(None);
        };
        if self.param_slots.contains_key(&sym) {
            return Err(CompileError::new(ErrorKind::Unsupported(
                "aliases of qubit parameters are not supported".into(),
            ))
            .with_span(expr.span));
        }
        let Some(reg) = self.register_of(sym) else {
            return if matches!(
                symbols.get(sym).kind,
                SymbolKind::Qubit | SymbolKind::GateQubit
            ) {
                Err(CompileError::new(ErrorKind::Unsupported(format!(
                    "cannot resolve qubit reference `{}`",
                    symbols.get(sym).name
                )))
                .with_span(expr.span))
            } else {
                Ok(None)
            };
        };
        match ops.as_slice() {
            [] => Ok(Some(reg.clone())),
            [io] => {
                let idxs = resolve_static_index(io, reg.len()).map_err(|e| e.with_span(io.span))?;
                Ok(Some(select(reg, &idxs)))
            }
            _ => Err(CompileError::new(ErrorKind::Unsupported(
                "multi-dimensional index on a quantum register".into(),
            ))
            .with_span(expr.span)),
        }
    }
}

/// Peel an `Index` chain down to its base variable, collecting index
/// ops in application order.
pub fn peel_index_chain(e: &SsaExpr) -> Option<(SymbolId, Vec<&IndexOp<SsaExpr>>)> {
    match &e.kind {
        SsaExprKind::Var(v) => Some((v.symbol, Vec::new())),
        SsaExprKind::Index(ix) => {
            let (sym, mut ops) = peel_index_chain(&ix.base)?;
            ops.push(&ix.index);
            Some((sym, ops))
        }
        _ => None,
    }
}

/// `Some(i)` iff `e` is an integer literal (const folding during SIR
/// lowering guarantees constant indices arrive as literals).
pub fn literal_index(e: &SsaExpr) -> Option<i128> {
    match &e.kind {
        SsaExprKind::Literal(Primitive::Int(i)) => Some(*i),
        SsaExprKind::Literal(Primitive::Uint(u)) => i128::try_from(*u).ok(),
        _ => None,
    }
}

/// The single index expression of `io`, if it is a plain `q[expr]`.
pub fn single_index_expr(io: &IndexOp<SsaExpr>) -> Option<&SsaExpr> {
    match &io.kind {
        IndexKind::Items(items) => match items.as_slice() {
            [IndexItem::Single(e)] => Some(e),
            _ => None,
        },
        IndexKind::Set(_) => None,
    }
}

/// Statically evaluate an index op against a register of length `len`,
/// returning the selected local indices in order. Implements the index
/// set semantics from docs/types.rst: single (possibly negative)
/// integers, discrete sets `{a,b,c}`, and inclusive ranges `a:b` /
/// `a:c:b` = {a, a+c, ..., a+mc} with m maximized. Negative members
/// select from the end. All constituent expressions must be integer
/// literals.
pub fn resolve_static_index(io: &IndexOp<SsaExpr>, len: usize) -> Result<Vec<usize>> {
    let idxs = match &io.kind {
        IndexKind::Set(es) => es
            .iter()
            .map(|e| normalize(const_index(e)?, len))
            .collect::<Result<Vec<_>>>()?,
        IndexKind::Items(items) => match items.as_slice() {
            [IndexItem::Single(e)] => vec![normalize(const_index(e)?, len)?],
            [IndexItem::Range(r)] => expand_range(r, len)?,
            _ => {
                return Err(CompileError::new(ErrorKind::Unsupported(
                    "multi-dimensional index on a quantum register".into(),
                ))
                .with_span(io.span));
            }
        },
    };
    if idxs.is_empty() {
        return Err(CompileError::new(ErrorKind::InvalidContext(
            "a quantum register cannot be indexed by an empty index set".into(),
        ))
        .with_span(io.span));
    }
    Ok(idxs)
}

/// Build a register selecting `indices` (in order) from `reg`, using a
/// single slice per run of consecutive indices.
pub fn select(reg: &QuantumRegister, indices: &[usize]) -> QuantumRegister {
    let mut out = QuantumRegister::new();
    let mut i = 0;
    while i < indices.len() {
        let start = indices[i];
        let mut end = start + 1;
        i += 1;
        while i < indices.len() && indices[i] == end {
            end += 1;
            i += 1;
        }
        out = out.concat(reg.slice(start..end));
    }
    out
}

fn const_index(e: &SsaExpr) -> Result<i128> {
    literal_index(e)
        .ok_or_else(|| CompileError::new(ErrorKind::NonConstantExpression).with_span(e.span))
}

/// Map a possibly-negative index into `0..len`.
fn normalize(idx: i128, len: usize) -> Result<usize> {
    let n = len as i128;
    let adjusted = if idx < 0 { idx + n } else { idx };
    if (0..n).contains(&adjusted) {
        Ok(adjusted as usize)
    } else {
        Err(CompileError::new(ErrorKind::QubitIndexOutOfRange {
            index: idx,
            len,
        }))
    }
}

/// Expand an inclusive range per the spec: the member set is generated
/// arithmetically from the raw (possibly negative) bounds, then each
/// member is normalized as an index. A direction mismatch yields the
/// empty set (rejected by the caller).
fn expand_range(r: &RangeExpr<SsaExpr>, len: usize) -> Result<Vec<usize>> {
    let step = match &r.step {
        Some(e) => const_index(e)?,
        None => 1,
    };
    if step == 0 {
        return Err(CompileError::new(ErrorKind::InvalidContext(
            "range step must be non-zero".into(),
        )));
    }
    let n = len as i128;
    let start = match &r.start {
        Some(e) => const_index(e)?,
        None => {
            if step > 0 {
                0
            } else {
                n - 1
            }
        }
    };
    let end = match &r.end {
        Some(e) => const_index(e)?,
        None => {
            if step > 0 {
                n - 1
            } else {
                0
            }
        }
    };

    let mut out = Vec::new();
    let mut cur = start;
    while (step > 0 && cur <= end) || (step < 0 && cur >= end) {
        out.push(normalize(cur, len)?);
        cur += step;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg as cfg_mod;
    use crate::lower::compile_source;
    use crate::resolve::DefaultIncludeResolver;
    use crate::ssa::{self, SsaStmtKind};

    fn compile(src: &str) -> sir::Program {
        compile_source(src, DefaultIncludeResolver, None).expect("compile")
    }

    /// Build the layout and resolve every top-level alias in block
    /// order, mirroring what the bytecode emitter does.
    fn try_layout(src: &str) -> Result<(sir::Program, QubitLayout)> {
        let p = compile(src);
        let cfgs = cfg_mod::build_program(&p).expect("cfg");
        let ssa_prog = ssa::build_program(&cfgs, &p.symbols);
        let mut layout = build_layout(&p);
        for block in &ssa_prog.top_level.blocks {
            for stmt in &block.stmts {
                if let SsaStmtKind::Alias(a) = &stmt.kind
                    && let Some(reg) = layout.resolve_alias_value(&a.value, &p.symbols)?
                {
                    layout.define_alias(a.symbol, reg);
                }
            }
        }
        Ok((p, layout))
    }

    fn layout(src: &str) -> (sir::Program, QubitLayout) {
        match try_layout(src) {
            Ok(out) => out,
            Err(e) => panic!("layout failed: {e}"),
        }
    }

    fn layout_err(src: &str) -> CompileError {
        match try_layout(src) {
            Err(e) => e,
            Ok(_) => panic!("expected a layout error for `{src}`"),
        }
    }

    fn ranges_of(p: &sir::Program, l: &QubitLayout, name: &str) -> Vec<(u32, u32)> {
        let sym = p.symbols.lookup(name).expect("symbol");
        let reg = l.register_of(sym).expect("register");
        l.global_ranges(reg)
    }

    #[test]
    fn registers_allocate_in_declaration_order() {
        let (p, l) = layout("qubit c; qubit[4] a; qubit[4] b;");
        assert_eq!(l.num_qubits(), 9);
        assert_eq!(ranges_of(&p, &l, "c"), vec![(0, 1)]);
        assert_eq!(ranges_of(&p, &l, "a"), vec![(1, 5)]);
        assert_eq!(ranges_of(&p, &l, "b"), vec![(5, 9)]);
        let b = p.symbols.lookup("b").unwrap();
        assert_eq!(l.global_index(l.register_of(b).unwrap(), 0), Some(5));
    }

    #[test]
    fn alias_slice_is_inclusive() {
        // docs/types.rst: `let myreg = q[1:4];` — myreg[0] is q[1].
        let (p, l) = layout("qubit[5] q; let myreg = q[1:4];");
        assert_eq!(ranges_of(&p, &l, "myreg"), vec![(1, 5)]);
    }

    #[test]
    fn alias_concat_merges_adjacent_ranges() {
        let (p, l) = layout("qubit[2] one; qubit[10] two; let concatenated = one ++ two;");
        assert_eq!(ranges_of(&p, &l, "concatenated"), vec![(0, 12)]);
    }

    #[test]
    fn alias_discrete_set() {
        // `two` starts at global 2.
        let (p, l) = layout("qubit[2] one; qubit[10] two; let sel = two[{0, 3, 5}];");
        assert_eq!(ranges_of(&p, &l, "sel"), vec![(2, 3), (5, 6), (7, 8)]);
    }

    #[test]
    fn alias_negative_index_and_range() {
        let (p, l) =
            layout("qubit[2] one; qubit[10] two; let last = two[-1]; let last_three = two[-4:-1];");
        assert_eq!(ranges_of(&p, &l, "last"), vec![(11, 12)]);
        // two[-4:-1] = local {6,7,8,9} = global 8..12.
        assert_eq!(ranges_of(&p, &l, "last_three"), vec![(8, 12)]);
    }

    #[test]
    fn alias_step_range() {
        let (p, l) = layout("qubit[2] one; qubit[10] two; let every_second = two[0:2:8];");
        // local {0,2,4,6,8} = global {2,4,6,8,10}.
        assert_eq!(
            ranges_of(&p, &l, "every_second"),
            vec![(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        );
    }

    #[test]
    fn alias_of_alias() {
        let (p, l) = layout(
            r#"
                qubit[2] one;
                qubit[10] two;
                let concatenated = one ++ two;
                let sliced = concatenated[0:6];
                let last_three = two[-4:-1];
                let both = sliced ++ last_three;
            "#,
        );
        assert_eq!(ranges_of(&p, &l, "sliced"), vec![(0, 7)]);
        assert_eq!(ranges_of(&p, &l, "both"), vec![(0, 7), (8, 12)]);
    }

    #[test]
    fn alias_out_of_range_index_is_rejected() {
        let err = layout_err("qubit[4] q; let bad = q[4];");
        assert!(matches!(
            err.kind,
            ErrorKind::QubitIndexOutOfRange { index: 4, len: 4 }
        ));
    }

    #[test]
    fn alias_self_concatenation_is_rejected() {
        let err = layout_err("qubit[4] q; let bad = q ++ q[0:1];");
        assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
    }

    #[test]
    fn alias_runtime_index_is_rejected() {
        let err = layout_err("input uint[32] i; qubit[4] q; let bad = q[{i}];");
        assert!(matches!(err.kind, ErrorKind::NonConstantExpression));
    }

    #[test]
    fn classical_alias_is_not_a_qubit_alias() {
        let (p, l) = layout("bit[4] c; let view = c[0:1];");
        let sym = p.symbols.lookup("view").expect("symbol");
        assert!(l.register_of(sym).is_none());
    }

    #[test]
    fn param_slots_for_gates_and_subroutines() {
        let (p, l) = layout(
            r#"
                qubit[2] b;
                gate g a, c { }
                def f(int n, qubit[2] d) { reset d[0]; }
            "#,
        );
        let gate = &p.gates[0];
        assert_eq!(l.param_slot(gate.qubits[0]), Some(0));
        assert_eq!(l.param_slot(gate.qubits[1]), Some(1));
        let sub = &p.subroutines[0];
        assert_eq!(l.param_slot(sub.params[0].symbol), None);
        assert_eq!(l.param_slot(sub.params[1].symbol), Some(1));
    }

    #[test]
    fn empty_index_set_is_rejected() {
        // Direction mismatch produces the empty set, which the spec
        // forbids as a register index.
        let err = layout_err("qubit[4] q; let bad = q[3:1];");
        assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
    }
}
