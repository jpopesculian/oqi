//! Control flow graph (CFG) construction over SIR.
//!
//! Builds one [`Cfg`] per function-shaped body in a [`sir::Program`]:
//! the top-level body, every subroutine, every gate, and every structured
//! OpenPulse calibration body. Opaque defcal bodies are skipped.
//!
//! The CFG is owned: each [`BasicBlock`] holds cloned statements lifted out of
//! SIR. The original program is left untouched so SIR analyses can keep using
//! it after CFG construction.

use std::cmp::Ordering;

use oqi_lex::Span;

use crate::classical::{Primitive, PrimitiveTy, ValueTy};
use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{
    self, Alias, Annotation, ArrayLiteral, Assignment, BinOp, Binary, BoxStmt, Call,
    CalibrationBody, Cast, Delay, Expr, ExprKind, For, ForIterable, GateCall, GateModifier, If,
    Index, IndexItem, IndexKind, IndexOp, LValue, Measure, MeasureExpr, MeasureExprKind,
    QubitOperand, RValue, Stmt, StmtKind, SwitchCase, SwitchLabels, UnOp, Unary, While,
};
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasicBlockId(pub usize);

#[derive(Clone)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub stmts: Vec<BlockStmt>,
    pub terminator: Terminator,
    pub span: Span,
}

#[derive(Clone)]
pub struct BlockStmt {
    pub kind: BlockStmtKind,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

/// The non-control-flow subset of [`sir::StmtKind`]. Statements that can appear
/// inside a [`BasicBlock`]: control-flow constructs become [`Terminator`]s and
/// are statically excluded here. `Box` and `Cal` bodies are themselves nested
/// [`Cfg`]s so the no-control-flow invariant holds recursively. All embedded
/// expressions use [`BlockExpr`] (no `DurationOf(Vec<Stmt>)` smuggling control
/// flow through expression position).
#[derive(Clone)]
pub enum BlockStmtKind {
    Alias(Alias<BlockExpr>),
    GateCall(GateCall<BlockExpr>),
    Measure(Measure<BlockExpr>),
    Reset(QubitOperand<BlockExpr>),
    Barrier(Vec<QubitOperand<BlockExpr>>),
    Delay(Delay<BlockExpr>),
    Box(BlockBoxStmt),
    Assignment(Assignment<BlockExpr>),
    Pragma(String),
    Cal(BlockCalibrationBody),
    ExprStmt(BlockExpr),
    Nop(Vec<QubitOperand<BlockExpr>>),
}

#[derive(Clone)]
pub struct BlockBoxStmt {
    pub duration: Option<BlockExpr>,
    pub body: Cfg,
}

#[derive(Clone)]
pub enum BlockCalibrationBody {
    Opaque(String),
    OpenPulse(Cfg),
}

/// The non-control-flow subset of [`sir::Expr`]. Mirrors `Expr` except
/// `DurationOf` holds a nested [`Cfg`] instead of `Vec<Stmt>`, so no
/// control flow can hide inside an expression position.
#[derive(Clone)]
pub struct BlockExpr {
    pub kind: BlockExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone)]
pub enum BlockExprKind {
    Literal(Primitive),
    Var(SymbolId),
    HardwareQubit(usize),
    Binary(Binary<BlockExpr>),
    Unary(Unary<BlockExpr>),
    Cast(Cast<BlockExpr>),
    Index(Index<BlockExpr>),
    Call(Call<BlockExpr>),
    DurationOf(Cfg),
    ArrayLiteral(ArrayLiteral<BlockExpr>),
}

#[derive(Clone)]
pub enum Terminator {
    /// Unconditional jump.
    Goto(BasicBlockId),
    /// Two-way branch on a boolean condition.
    Branch {
        cond: BlockExpr,
        then_bb: BasicBlockId,
        else_bb: BasicBlockId,
    },
    /// Multi-way dispatch on `target`. Cases are checked in order; if none match
    /// and `default` is `None`, control falls through to whatever block comes
    /// after the switch (the merge block) — the builder always populates
    /// `default` with that merge target.
    Switch {
        target: BlockExpr,
        cases: Vec<(SwitchLabels<BlockExpr>, BasicBlockId)>,
        default: Option<BasicBlockId>,
    },
    /// Explicit `return` from a subroutine (or implicit fall-off-the-end).
    Return(Option<RValue<BlockExpr>>),
    /// OpenQASM `end;` — program halt.
    End,
    /// Block has no successors and is statically unreachable. Used as a
    /// placeholder while constructing the graph and for blocks following a
    /// terminator within the same source body.
    Unreachable,
}

impl Terminator {
    /// Iterate the block ids this terminator can transfer control to.
    /// Allocation-free.
    pub fn successors(&self) -> Successors<'_> {
        let inner = match self {
            Terminator::Goto(t) => SuccessorsInner::One(Some(*t)),
            Terminator::Branch {
                then_bb, else_bb, ..
            } => SuccessorsInner::Two(Some(*then_bb), Some(*else_bb)),
            Terminator::Switch {
                cases, default, ..
            } => SuccessorsInner::Switch {
                cases: cases.iter(),
                default: *default,
            },
            Terminator::Return(_) | Terminator::End | Terminator::Unreachable => {
                SuccessorsInner::Empty
            }
        };
        Successors(inner)
    }
}

/// Iterator returned by [`Terminator::successors`]. Opaque wrapper:
/// callers should only obtain a `Successors` via the terminator method.
pub struct Successors<'a>(SuccessorsInner<'a>);

enum SuccessorsInner<'a> {
    Empty,
    One(Option<BasicBlockId>),
    Two(Option<BasicBlockId>, Option<BasicBlockId>),
    Switch {
        cases: std::slice::Iter<'a, (SwitchLabels<BlockExpr>, BasicBlockId)>,
        default: Option<BasicBlockId>,
    },
}

impl Iterator for Successors<'_> {
    type Item = BasicBlockId;

    fn next(&mut self) -> Option<BasicBlockId> {
        match &mut self.0 {
            SuccessorsInner::Empty => None,
            SuccessorsInner::One(slot) => slot.take(),
            SuccessorsInner::Two(a, b) => a.take().or_else(|| b.take()),
            SuccessorsInner::Switch { cases, default } => cases
                .next()
                .map(|(_, bb)| *bb)
                .or_else(|| default.take()),
        }
    }
}

#[derive(Clone)]
pub enum CfgOwner {
    TopLevel,
    Subroutine(SymbolId),
    Gate(SymbolId),
    Calibration(usize),
    /// Body of a `box { ... }` block statement.
    Box,
    /// Body of an inline `cal { ... }` block statement (top-level `defcal`s
    /// use [`CfgOwner::Calibration`]).
    InlineCal,
    /// Body of a `durationof({...})` expression.
    DurationOf,
}

#[derive(Clone)]
pub struct Cfg {
    /// May contain blocks unreachable from `entry`: dead statements
    /// after a `break`/`continue`/`return`/`end` land in a fresh block
    /// with no incoming edges. Passes that iterate `blocks` directly
    /// (rather than walking from `entry`) will see that dead code.
    pub blocks: Vec<BasicBlock>,
    /// Sole entry block. No predecessors; all control flow enters here.
    pub entry: BasicBlockId,
    /// Synthetic "fall-off-the-end" exit block. Always present, always
    /// terminated by `Return(None)`. Reachable when the body finishes
    /// without an explicit `return`/`end`; otherwise dangling (no
    /// predecessors). Analyses that walk from `entry` simply won't
    /// visit it in the dangling case.
    pub exit: BasicBlockId,
    pub owner: CfgOwner,
}

impl Cfg {
    /// Pair this CFG with a symbol table for [`std::fmt::Display`].
    /// Use as `format!("{}", cfg.display(&symbols))`.
    pub fn display<'a>(&'a self, symbols: &'a SymbolTable) -> CfgDisplay<'a> {
        CfgDisplay { cfg: self, symbols }
    }
}

pub struct CfgDisplay<'a> {
    pub(crate) cfg: &'a Cfg,
    pub(crate) symbols: &'a SymbolTable,
}

pub struct ProgramCfgs {
    pub top_level: Cfg,
    pub subroutines: Vec<Cfg>,
    pub gates: Vec<Cfg>,
    /// Parallel to `Program::calibrations`. `None` for opaque defcals.
    pub calibrations: Vec<Option<Cfg>>,
}

pub fn build_program(program: &sir::Program) -> Result<ProgramCfgs> {
    let top_level = build_body(
        program.body.clone(),
        CfgOwner::TopLevel,
        false,
        &program.symbols,
    )?;
    let subroutines = program
        .subroutines
        .iter()
        .map(|s| {
            build_body(
                s.body.clone(),
                CfgOwner::Subroutine(s.symbol),
                true,
                &program.symbols,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let gates = program
        .gates
        .iter()
        .map(|g| {
            build_body(
                g.body.body.clone(),
                CfgOwner::Gate(g.symbol),
                false,
                &program.symbols,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let calibrations = program
        .calibrations
        .iter()
        .enumerate()
        .map(|(i, c)| match &c.body {
            CalibrationBody::OpenPulse(stmts) => Ok(Some(build_body(
                stmts.clone(),
                CfgOwner::Calibration(i),
                false,
                &program.symbols,
            )?)),
            CalibrationBody::Opaque(_) => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ProgramCfgs {
        top_level,
        subroutines,
        gates,
        calibrations,
    })
}

/// Construct a [`Cfg`] from a body of SIR statements.
///
/// Always emits a synthetic [`Cfg::exit`] block with a
/// [`Terminator::Return(None)`] terminator. If the body finishes with
/// `current` still open (no explicit `return`/`end` reached), that
/// trailing block is wired to `exit` via `Goto`. If the body ends in an
/// explicit terminator, `exit` is unreachable — it still exists as a
/// canonical "fall-off-the-end" anchor for downstream analyses, but no
/// predecessor edges point at it.
fn build_body(
    stmts: Vec<Stmt>,
    owner: CfgOwner,
    allow_return: bool,
    symbols: &SymbolTable,
) -> Result<Cfg> {
    let mut b = CfgBuilder::new(allow_return, symbols);
    let entry = b.new_block(Span::default());
    b.current = Some(entry);
    b.lower_stmts(stmts)?;
    let exit = b.new_block(Span::default());
    if let Some(cur) = b.current.take() {
        b.set_terminator(cur, Terminator::Goto(exit));
    }
    b.set_terminator(exit, Terminator::Return(None));
    Ok(Cfg {
        blocks: b.blocks,
        entry,
        exit,
        owner,
    })
}

struct LoopFrame {
    /// Where `continue` jumps. For `while`, the header. For `for`, the latch.
    continue_target: BasicBlockId,
    /// Where `break` jumps.
    break_target: BasicBlockId,
}

enum FrameKind {
    Loop(LoopFrame),
    Switch { break_target: BasicBlockId },
}

struct CfgBuilder<'a> {
    blocks: Vec<BasicBlock>,
    /// `None` after a terminator, until a new statement opens a fresh block.
    current: Option<BasicBlockId>,
    frames: Vec<FrameKind>,
    allow_return: bool,
    symbols: &'a SymbolTable,
}

impl<'a> CfgBuilder<'a> {
    fn new(allow_return: bool, symbols: &'a SymbolTable) -> Self {
        Self {
            blocks: Vec::new(),
            current: None,
            frames: Vec::new(),
            allow_return,
            symbols,
        }
    }

    fn new_block(&mut self, span: Span) -> BasicBlockId {
        let id = BasicBlockId(self.blocks.len());
        self.blocks.push(BasicBlock {
            id,
            stmts: Vec::new(),
            terminator: Terminator::Unreachable,
            span,
        });
        id
    }

    fn set_terminator(&mut self, block: BasicBlockId, term: Terminator) {
        self.blocks[block.0].terminator = term;
    }

    fn append(&mut self, stmt: BlockStmt) {
        let cur = match self.current {
            Some(c) => c,
            None => {
                // Dead code path: open a fresh block so subsequent stmts have
                // somewhere to live. It has no incoming edges.
                let id = self.new_block(stmt.span);
                self.current = Some(id);
                id
            }
        };
        self.blocks[cur.0].stmts.push(stmt);
    }

    fn terminate(&mut self, term: Terminator) {
        if let Some(cur) = self.current.take() {
            self.set_terminator(cur, term);
        }
    }

    fn innermost_loop(&self) -> Option<&LoopFrame> {
        self.frames.iter().rev().find_map(|f| match f {
            FrameKind::Loop(l) => Some(l),
            FrameKind::Switch { .. } => None,
        })
    }

    fn innermost_break(&self) -> Option<BasicBlockId> {
        self.frames.iter().rev().find_map(|f| match f {
            FrameKind::Loop(l) => Some(l.break_target),
            FrameKind::Switch { break_target } => Some(*break_target),
        })
    }

    fn lower_stmts(&mut self, stmts: Vec<Stmt>) -> Result<()> {
        for stmt in stmts {
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    fn lower_stmt(&mut self, stmt: Stmt) -> Result<()> {
        let Stmt {
            kind,
            annotations,
            span,
        } = stmt;
        match kind {
            StmtKind::If(If {
                condition,
                then_body,
                else_body,
            }) => {
                let cond = self.lower_expr(condition)?;
                let then_bb = self.new_block(span);
                let else_bb = self.new_block(span);
                let after_bb = self.new_block(span);
                self.terminate(Terminator::Branch {
                    cond,
                    then_bb,
                    else_bb,
                });
                self.current = Some(then_bb);
                self.lower_stmts(then_body)?;
                self.terminate(Terminator::Goto(after_bb));
                self.current = Some(else_bb);
                if let Some(else_body) = else_body {
                    self.lower_stmts(else_body)?;
                }
                self.terminate(Terminator::Goto(after_bb));
                self.current = Some(after_bb);
                Ok(())
            }
            StmtKind::While(While { condition, body }) => {
                let cond = self.lower_expr(condition)?;
                let header_bb = self.new_block(span);
                let body_bb = self.new_block(span);
                let after_bb = self.new_block(span);
                self.terminate(Terminator::Goto(header_bb));
                self.set_terminator(
                    header_bb,
                    Terminator::Branch {
                        cond,
                        then_bb: body_bb,
                        else_bb: after_bb,
                    },
                );
                self.frames.push(FrameKind::Loop(LoopFrame {
                    continue_target: header_bb,
                    break_target: after_bb,
                }));
                self.current = Some(body_bb);
                self.lower_stmts(body)?;
                self.terminate(Terminator::Goto(header_bb));
                self.frames.pop();
                self.current = Some(after_bb);
                Ok(())
            }
            StmtKind::For(For {
                var,
                iterable,
                body,
            }) => self.lower_for(var, iterable, body, span),
            StmtKind::Switch(sir::Switch { target, cases }) => {
                self.lower_switch(target, cases, span)
            }
            StmtKind::Break => {
                let target = self.innermost_break().ok_or_else(|| {
                    CompileError::new(ErrorKind::InvalidContext(
                        "`break` outside of loop or switch".into(),
                    ))
                    .with_span(span)
                })?;
                self.terminate(Terminator::Goto(target));
                Ok(())
            }
            StmtKind::Continue => {
                let target = self
                    .innermost_loop()
                    .map(|l| l.continue_target)
                    .ok_or_else(|| {
                        CompileError::new(ErrorKind::InvalidContext(
                            "`continue` outside of loop".into(),
                        ))
                        .with_span(span)
                    })?;
                self.terminate(Terminator::Goto(target));
                Ok(())
            }
            StmtKind::Return(value) => {
                if !self.allow_return {
                    return Err(CompileError::new(ErrorKind::InvalidContext(
                        "`return` outside of subroutine".into(),
                    ))
                    .with_span(span));
                }
                let lowered = match value {
                    Some(rv) => Some(self.lower_rvalue(rv)?),
                    None => None,
                };
                self.terminate(Terminator::Return(lowered));
                Ok(())
            }
            StmtKind::End => {
                self.terminate(Terminator::End);
                Ok(())
            }
            other => {
                let kind = self.lower_non_cf_stmt(other)?;
                self.append(BlockStmt {
                    kind,
                    annotations,
                    span,
                });
                Ok(())
            }
        }
    }

    /// Lower a non-control-flow [`StmtKind`] to its [`BlockStmtKind`]
    /// counterpart. Caller must have already destructured all
    /// control-flow variants — this function panics on those.
    fn lower_non_cf_stmt(&mut self, kind: StmtKind) -> Result<BlockStmtKind> {
        Ok(match kind {
            StmtKind::Alias(Alias { symbol, value }) => {
                let value = value
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                BlockStmtKind::Alias(Alias { symbol, value })
            }
            StmtKind::GateCall(GateCall {
                gate,
                modifiers,
                args,
                qubits,
            }) => {
                let modifiers = modifiers
                    .into_iter()
                    .map(|m| self.lower_gate_modifier(m))
                    .collect::<Result<Vec<_>>>()?;
                let args = args
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                let qubits = qubits
                    .into_iter()
                    .map(|q| self.lower_qubit_operand(q))
                    .collect::<Result<Vec<_>>>()?;
                BlockStmtKind::GateCall(GateCall {
                    gate,
                    modifiers,
                    args,
                    qubits,
                })
            }
            StmtKind::Measure(m) => BlockStmtKind::Measure(self.lower_measure(m)?),
            StmtKind::Reset(q) => BlockStmtKind::Reset(self.lower_qubit_operand(q)?),
            StmtKind::Barrier(qs) => {
                let qs = qs
                    .into_iter()
                    .map(|q| self.lower_qubit_operand(q))
                    .collect::<Result<Vec<_>>>()?;
                BlockStmtKind::Barrier(qs)
            }
            StmtKind::Delay(Delay { duration, operands }) => {
                let duration = self.lower_expr(duration)?;
                let operands = operands
                    .into_iter()
                    .map(|q| self.lower_qubit_operand(q))
                    .collect::<Result<Vec<_>>>()?;
                BlockStmtKind::Delay(Delay { duration, operands })
            }
            StmtKind::Assignment(Assignment { target, value }) => {
                let target = self.lower_lvalue(target)?;
                let value = self.lower_rvalue(value)?;
                BlockStmtKind::Assignment(Assignment { target, value })
            }
            StmtKind::Pragma(p) => BlockStmtKind::Pragma(p),
            StmtKind::ExprStmt(e) => BlockStmtKind::ExprStmt(self.lower_expr(e)?),
            StmtKind::Nop(qs) => {
                let qs = qs
                    .into_iter()
                    .map(|q| self.lower_qubit_operand(q))
                    .collect::<Result<Vec<_>>>()?;
                BlockStmtKind::Nop(qs)
            }
            StmtKind::Box(BoxStmt { duration, body }) => {
                let duration = match duration {
                    Some(e) => Some(self.lower_expr(e)?),
                    None => None,
                };
                let inner = build_body(body, CfgOwner::Box, self.allow_return, self.symbols)?;
                BlockStmtKind::Box(BlockBoxStmt {
                    duration,
                    body: inner,
                })
            }
            StmtKind::Cal(CalibrationBody::Opaque(s)) => {
                BlockStmtKind::Cal(BlockCalibrationBody::Opaque(s))
            }
            StmtKind::Cal(CalibrationBody::OpenPulse(stmts)) => {
                let inner = build_body(stmts, CfgOwner::InlineCal, false, self.symbols)?;
                BlockStmtKind::Cal(BlockCalibrationBody::OpenPulse(inner))
            }
            StmtKind::If(_)
            | StmtKind::For(_)
            | StmtKind::While(_)
            | StmtKind::Switch(_)
            | StmtKind::Break
            | StmtKind::Continue
            | StmtKind::Return(_)
            | StmtKind::End => {
                unreachable!("control-flow variants are handled by lower_stmt before reaching this point")
            }
        })
    }

    fn lower_for(
        &mut self,
        var: SymbolId,
        iterable: ForIterable,
        body: Vec<Stmt>,
        span: Span,
    ) -> Result<()> {
        let (start, step, end) = match iterable {
            ForIterable::Range { start, step, end } => (start, step, end),
            ForIterable::Set(_) | ForIterable::Expr(_) => {
                return Err(CompileError::new(ErrorKind::Unsupported(
                    "CFG construction for non-range for-loops not yet implemented".into(),
                ))
                .with_span(span));
            }
        };
        let end = end.ok_or_else(|| {
            CompileError::new(ErrorKind::Unsupported(
                "CFG construction requires an explicit end in range for-loops".into(),
            ))
            .with_span(span)
        })?;

        let var_ty = self.symbols.get(var).ty.clone();

        // Synthesized init assignment `var = start` (defaults to 0 when
        // the range omits start). Has the for-statement's span and no
        // annotations — it doesn't correspond to any user-written
        // statement, so it won't round-trip back to source.
        let init_expr = match start {
            Some(s) => self.lower_expr(*s)?,
            None => block_literal_int(&var_ty, 0, span),
        };
        self.append(BlockStmt {
            kind: BlockStmtKind::Assignment(Assignment {
                target: LValue::Var(var),
                value: RValue::Expr(Box::new(init_expr)),
            }),
            annotations: vec![],
            span,
        });

        let header_bb = self.new_block(span);
        let body_bb = self.new_block(span);
        let latch_bb = self.new_block(span);
        let after_bb = self.new_block(span);

        self.terminate(Terminator::Goto(header_bb));

        // Step expression (defaults to 1), lowered before the header
        // condition because the comparison direction depends on its
        // sign: ranges include `end`, counting up for a positive step
        // and down for a negative one.
        let step_expr = match step {
            Some(s) => self.lower_expr(*s)?,
            None => block_literal_int(&var_ty, 1, span),
        };
        // Caveat: for a descending range over an unsigned loop var with
        // `end == 0`, `var >= end` is a tautology after `var` wraps —
        // the evaluator must compute the advance in signed arithmetic
        // for the exit test to fire.
        let cmp_op = match const_step_sign(&step_expr) {
            Some(Ordering::Greater) => BinOp::Lte,
            Some(Ordering::Less) => BinOp::Gte,
            Some(Ordering::Equal) => {
                return Err(CompileError::new(ErrorKind::InvalidContext(
                    "for-loop range step must be non-zero".into(),
                ))
                .with_span(span));
            }
            None => {
                return Err(CompileError::new(ErrorKind::Unsupported(
                    "CFG construction requires a compile-time constant step in range for-loops"
                        .into(),
                ))
                .with_span(span));
            }
        };

        let end_block = self.lower_expr(*end)?;
        let cond = BlockExpr {
            kind: BlockExprKind::Binary(Binary {
                op: cmp_op,
                left: Box::new(block_var_expr(var, var_ty.clone(), span)),
                right: Box::new(end_block),
            }),
            ty: Type::Classical(ValueTy::bool()),
            span,
        };
        self.set_terminator(
            header_bb,
            Terminator::Branch {
                cond,
                then_bb: body_bb,
                else_bb: after_bb,
            },
        );

        self.frames.push(FrameKind::Loop(LoopFrame {
            continue_target: latch_bb,
            break_target: after_bb,
        }));

        self.current = Some(body_bb);
        self.lower_stmts(body)?;
        self.terminate(Terminator::Goto(latch_bb));

        self.frames.pop();

        self.current = Some(latch_bb);
        // Synthesized advance assignment `var = var + step`. Like the
        // init above, this is a synthetic statement with the
        // for-statement's span and no annotations.
        let advance = BlockExpr {
            kind: BlockExprKind::Binary(Binary {
                op: BinOp::Add,
                left: Box::new(block_var_expr(var, var_ty.clone(), span)),
                right: Box::new(step_expr),
            }),
            ty: var_ty,
            span,
        };
        self.append(BlockStmt {
            kind: BlockStmtKind::Assignment(Assignment {
                target: LValue::Var(var),
                value: RValue::Expr(Box::new(advance)),
            }),
            annotations: vec![],
            span,
        });
        self.terminate(Terminator::Goto(header_bb));

        self.current = Some(after_bb);
        Ok(())
    }

    fn lower_switch(
        &mut self,
        target: Expr,
        cases: Vec<SwitchCase>,
        span: Span,
    ) -> Result<()> {
        let target = self.lower_expr(target)?;
        let after_bb = self.new_block(span);

        // Allocate one block per case up front so the dispatcher can reference
        // them all before any case body is lowered.
        let mut case_blocks: Vec<(SwitchLabels<BlockExpr>, BasicBlockId)> = Vec::new();
        let mut default: Option<BasicBlockId> = None;
        let mut allocated: Vec<BasicBlockId> = Vec::with_capacity(cases.len());
        for case in &cases {
            let bb_span = case.body.first().map(|s| s.span).unwrap_or(span);
            let bb = self.new_block(bb_span);
            allocated.push(bb);
            match &case.labels {
                SwitchLabels::Values(vals) => {
                    let lowered_vals = vals
                        .iter()
                        .cloned()
                        .map(|v| self.lower_expr(v))
                        .collect::<Result<Vec<_>>>()?;
                    case_blocks.push((SwitchLabels::Values(lowered_vals), bb));
                }
                SwitchLabels::Default => default = Some(bb),
            }
        }

        self.terminate(Terminator::Switch {
            target,
            cases: case_blocks,
            default: Some(default.unwrap_or(after_bb)),
        });

        self.frames.push(FrameKind::Switch {
            break_target: after_bb,
        });

        for (case, bb) in cases.into_iter().zip(allocated) {
            self.current = Some(bb);
            self.lower_stmts(case.body)?;
            self.terminate(Terminator::Goto(after_bb));
        }

        self.frames.pop();
        self.current = Some(after_bb);
        Ok(())
    }

    // ── Expression lowering: sir::Expr → cfg::BlockExpr ────────────────

    fn lower_expr(&mut self, e: Expr) -> Result<BlockExpr> {
        let Expr { kind, ty, span } = e;
        Ok(BlockExpr {
            kind: self.lower_expr_kind(kind)?,
            ty,
            span,
        })
    }

    fn lower_expr_kind(&mut self, kind: ExprKind) -> Result<BlockExprKind> {
        Ok(match kind {
            ExprKind::Literal(p) => BlockExprKind::Literal(p),
            ExprKind::Var(s) => BlockExprKind::Var(s),
            ExprKind::HardwareQubit(n) => BlockExprKind::HardwareQubit(n),
            ExprKind::Binary(Binary { op, left, right }) => {
                let left = Box::new(self.lower_expr(*left)?);
                let right = Box::new(self.lower_expr(*right)?);
                BlockExprKind::Binary(Binary { op, left, right })
            }
            ExprKind::Unary(Unary { op, operand }) => {
                let operand = Box::new(self.lower_expr(*operand)?);
                BlockExprKind::Unary(Unary { op, operand })
            }
            ExprKind::Cast(Cast { target_ty, operand }) => {
                let operand = Box::new(self.lower_expr(*operand)?);
                BlockExprKind::Cast(Cast { target_ty, operand })
            }
            ExprKind::Index(Index { base, index }) => {
                let base = Box::new(self.lower_expr(*base)?);
                let index = self.lower_index_op(index)?;
                BlockExprKind::Index(Index { base, index })
            }
            ExprKind::Call(Call { callee, args }) => {
                let args = args
                    .into_iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<Vec<_>>>()?;
                BlockExprKind::Call(Call { callee, args })
            }
            ExprKind::DurationOf(stmts) => {
                let inner = build_body(stmts, CfgOwner::DurationOf, false, self.symbols)?;
                BlockExprKind::DurationOf(inner)
            }
            ExprKind::ArrayLiteral(ArrayLiteral { items, span: lit_span }) => {
                let items = items
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                BlockExprKind::ArrayLiteral(ArrayLiteral {
                    items,
                    span: lit_span,
                })
            }
        })
    }

    fn lower_qubit_operand(
        &mut self,
        q: QubitOperand<Expr>,
    ) -> Result<QubitOperand<BlockExpr>> {
        Ok(match q {
            QubitOperand::Indexed { symbol, indices } => {
                let indices = indices
                    .into_iter()
                    .map(|i| self.lower_index_op(i))
                    .collect::<Result<Vec<_>>>()?;
                QubitOperand::Indexed { symbol, indices }
            }
            QubitOperand::Hardware(n) => QubitOperand::Hardware(n),
        })
    }

    fn lower_lvalue(&mut self, lv: LValue<Expr>) -> Result<LValue<BlockExpr>> {
        Ok(match lv {
            LValue::Var(s) => LValue::Var(s),
            LValue::Indexed { symbol, indices } => {
                let indices = indices
                    .into_iter()
                    .map(|i| self.lower_index_op(i))
                    .collect::<Result<Vec<_>>>()?;
                LValue::Indexed { symbol, indices }
            }
        })
    }

    fn lower_rvalue(&mut self, rv: RValue<Expr>) -> Result<RValue<BlockExpr>> {
        Ok(match rv {
            RValue::Expr(e) => RValue::Expr(Box::new(self.lower_expr(*e)?)),
            RValue::Measure(m) => RValue::Measure(self.lower_measure_expr(m)?),
        })
    }

    fn lower_measure(&mut self, m: Measure<Expr>) -> Result<Measure<BlockExpr>> {
        let Measure { measure, target } = m;
        let measure = self.lower_measure_expr(measure)?;
        let target = match target {
            Some(lv) => Some(self.lower_lvalue(lv)?),
            None => None,
        };
        Ok(Measure { measure, target })
    }

    fn lower_measure_expr(
        &mut self,
        m: MeasureExpr<Expr>,
    ) -> Result<MeasureExpr<BlockExpr>> {
        let MeasureExpr { kind, ty, span } = m;
        let kind = match kind {
            MeasureExprKind::Measure { operand } => MeasureExprKind::Measure {
                operand: self.lower_qubit_operand(operand)?,
            },
            MeasureExprKind::QuantumCall {
                callee,
                args,
                qubits,
            } => {
                let args = args
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                let qubits = qubits
                    .into_iter()
                    .map(|q| self.lower_qubit_operand(q))
                    .collect::<Result<Vec<_>>>()?;
                MeasureExprKind::QuantumCall {
                    callee,
                    args,
                    qubits,
                }
            }
        };
        Ok(MeasureExpr { kind, ty, span })
    }

    fn lower_index_op(&mut self, io: IndexOp<Expr>) -> Result<IndexOp<BlockExpr>> {
        let IndexOp { kind, span } = io;
        let kind = match kind {
            IndexKind::Set(exprs) => {
                let exprs = exprs
                    .into_iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                IndexKind::Set(exprs)
            }
            IndexKind::Items(items) => {
                let items = items
                    .into_iter()
                    .map(|i| self.lower_index_item(i))
                    .collect::<Result<Vec<_>>>()?;
                IndexKind::Items(items)
            }
        };
        Ok(IndexOp { kind, span })
    }

    fn lower_index_item(
        &mut self,
        item: IndexItem<Expr>,
    ) -> Result<IndexItem<BlockExpr>> {
        Ok(match item {
            IndexItem::Single(e) => IndexItem::Single(Box::new(self.lower_expr(*e)?)),
            IndexItem::Range(r) => {
                let sir::RangeExpr { start, step, end } = r;
                let start = match start {
                    Some(e) => Some(Box::new(self.lower_expr(*e)?)),
                    None => None,
                };
                let step = match step {
                    Some(e) => Some(Box::new(self.lower_expr(*e)?)),
                    None => None,
                };
                let end = match end {
                    Some(e) => Some(Box::new(self.lower_expr(*e)?)),
                    None => None,
                };
                IndexItem::Range(sir::RangeExpr { start, step, end })
            }
        })
    }

    fn lower_gate_modifier(
        &mut self,
        m: GateModifier<Expr>,
    ) -> Result<GateModifier<BlockExpr>> {
        Ok(match m {
            GateModifier::Inv => GateModifier::Inv,
            GateModifier::Pow(e) => GateModifier::Pow(Box::new(self.lower_expr(*e)?)),
            GateModifier::Ctrl(n) => GateModifier::Ctrl(n),
            GateModifier::NegCtrl(n) => GateModifier::NegCtrl(n),
        })
    }
}

fn block_var_expr(var: SymbolId, ty: Type, span: Span) -> BlockExpr {
    BlockExpr {
        kind: BlockExprKind::Var(var),
        ty,
        span,
    }
}

/// Statically-known sign of a for-range step expression. Constant
/// steps are folded to a `Literal` during SIR lowering; the `Neg` arm
/// covers shapes the folder didn't reach.
fn const_step_sign(e: &BlockExpr) -> Option<Ordering> {
    match &e.kind {
        BlockExprKind::Literal(Primitive::Int(i)) => Some(i.cmp(&0)),
        BlockExprKind::Literal(Primitive::Uint(u)) => Some(u.cmp(&0)),
        BlockExprKind::Unary(u) if matches!(u.op, UnOp::Neg) => {
            const_step_sign(&u.operand).map(Ordering::reverse)
        }
        _ => None,
    }
}

fn block_literal_int(ty: &Type, value: i128, span: Span) -> BlockExpr {
    let prim = match ty.scalar_ty() {
        Some(PrimitiveTy::Uint(_)) => Primitive::uint(value as u128),
        _ => Primitive::int(value),
    };
    BlockExpr {
        kind: BlockExprKind::Literal(prim),
        ty: ty.clone(),
        span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bool_literal_expr() -> Expr {
        Expr {
            kind: ExprKind::Literal(Primitive::int(0)),
            ty: Type::Classical(ValueTy::bool()),
            span: Span::default(),
        }
    }

    fn pragma_stmt(s: &str) -> Stmt {
        Stmt {
            kind: StmtKind::Pragma(s.into()),
            annotations: vec![],
            span: Span::default(),
        }
    }

    fn if_stmt(then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt {
            kind: StmtKind::If(If {
                condition: bool_literal_expr(),
                then_body,
                else_body: Some(else_body),
            }),
            annotations: vec![],
            span: Span::default(),
        }
    }

    /// Locate the single `BlockStmtKind` in the outer CFG that matches a
    /// predicate. Panics if not found or if more than one matches.
    fn find_unique_block_stmt<F>(cfg: &Cfg, mut f: F) -> &BlockStmtKind
    where
        F: FnMut(&BlockStmtKind) -> bool,
    {
        let mut found = None;
        for block in &cfg.blocks {
            for stmt in &block.stmts {
                if f(&stmt.kind) {
                    assert!(found.is_none(), "expected exactly one matching stmt");
                    found = Some(&stmt.kind);
                }
            }
        }
        found.expect("no matching block stmt")
    }

    #[test]
    fn block_stmt_box_contains_inner_cfg() {
        // box { if (...) { pragma a; } else { pragma b; } }
        let symbols = SymbolTable::new();
        let inner_if = if_stmt(vec![pragma_stmt("a")], vec![pragma_stmt("b")]);
        let box_stmt = Stmt {
            kind: StmtKind::Box(BoxStmt {
                duration: None,
                body: vec![inner_if],
            }),
            annotations: vec![],
            span: Span::default(),
        };
        let cfg = build_body(vec![box_stmt], CfgOwner::TopLevel, false, &symbols)
            .expect("build_body");

        let inner = find_unique_block_stmt(&cfg, |k| matches!(k, BlockStmtKind::Box(_)));
        let BlockStmtKind::Box(BlockBoxStmt { body, .. }) = inner else {
            unreachable!()
        };
        // The if was lifted into terminators, so the inner CFG has more than
        // just an entry-and-exit pair.
        assert!(
            body.blocks.len() > 2,
            "expected inner CFG to have multiple blocks (got {})",
            body.blocks.len()
        );
        assert!(matches!(body.owner, CfgOwner::Box));
    }

    #[test]
    fn block_stmt_inline_cal_contains_inner_cfg() {
        // cal { if (...) { pragma a; } else { pragma b; } }
        let symbols = SymbolTable::new();
        let inner_if = if_stmt(vec![pragma_stmt("a")], vec![pragma_stmt("b")]);
        let cal_stmt = Stmt {
            kind: StmtKind::Cal(CalibrationBody::OpenPulse(vec![inner_if])),
            annotations: vec![],
            span: Span::default(),
        };
        let cfg = build_body(vec![cal_stmt], CfgOwner::TopLevel, false, &symbols)
            .expect("build_body");

        let inner = find_unique_block_stmt(&cfg, |k| matches!(k, BlockStmtKind::Cal(_)));
        let BlockStmtKind::Cal(BlockCalibrationBody::OpenPulse(body)) = inner else {
            panic!("expected OpenPulse cal body");
        };
        assert!(
            body.blocks.len() > 2,
            "expected inner CFG to have multiple blocks (got {})",
            body.blocks.len()
        );
        assert!(matches!(body.owner, CfgOwner::InlineCal));
    }

    #[test]
    fn block_stmt_opaque_cal_passes_through() {
        let symbols = SymbolTable::new();
        let cal_stmt = Stmt {
            kind: StmtKind::Cal(CalibrationBody::Opaque("raw".into())),
            annotations: vec![],
            span: Span::default(),
        };
        let cfg = build_body(vec![cal_stmt], CfgOwner::TopLevel, false, &symbols)
            .expect("build_body");

        let inner = find_unique_block_stmt(&cfg, |k| matches!(k, BlockStmtKind::Cal(_)));
        assert!(matches!(
            inner,
            BlockStmtKind::Cal(BlockCalibrationBody::Opaque(s)) if s == "raw"
        ));
    }

    #[test]
    fn block_expr_durationof_contains_inner_cfg() {
        // x = durationof({ if (...) { pragma a; } else { pragma b; } });
        // Build a Pragma statement carrying a Var = durationof(...) Expr in
        // a way that ends up in a BlockStmt. Easiest: use an ExprStmt whose
        // expr is the DurationOf.
        let symbols = SymbolTable::new();
        let inner_if = if_stmt(vec![pragma_stmt("a")], vec![pragma_stmt("b")]);
        let duration_expr = Expr {
            kind: ExprKind::DurationOf(vec![inner_if]),
            ty: Type::Void,
            span: Span::default(),
        };
        let expr_stmt = Stmt {
            kind: StmtKind::ExprStmt(duration_expr),
            annotations: vec![],
            span: Span::default(),
        };
        let cfg = build_body(vec![expr_stmt], CfgOwner::TopLevel, false, &symbols)
            .expect("build_body");

        let inner =
            find_unique_block_stmt(&cfg, |k| matches!(k, BlockStmtKind::ExprStmt(_)));
        let BlockStmtKind::ExprStmt(BlockExpr {
            kind: BlockExprKind::DurationOf(inner_cfg),
            ..
        }) = inner
        else {
            panic!("expected ExprStmt(DurationOf(_))");
        };
        assert!(
            inner_cfg.blocks.len() > 2,
            "expected inner CFG to have multiple blocks (got {})",
            inner_cfg.blocks.len()
        );
        assert!(matches!(inner_cfg.owner, CfgOwner::DurationOf));
    }

    #[test]
    fn terminator_branch_uses_block_expr() {
        // Build a simple `if (lit) {} else {}` and confirm the Branch
        // terminator's cond is a BlockExpr (statically guaranteed by the
        // type system — this test mostly documents the invariant and
        // exercises the lower_expr path for If conditions).
        let symbols = SymbolTable::new();
        let if_only = if_stmt(vec![], vec![]);
        let cfg = build_body(vec![if_only], CfgOwner::TopLevel, false, &symbols)
            .expect("build_body");

        let mut found_branch = false;
        for block in &cfg.blocks {
            if let Terminator::Branch { cond, .. } = &block.terminator {
                // Type-checks only because `cond` is `BlockExpr`.
                let _: &BlockExpr = cond;
                found_branch = true;
            }
        }
        assert!(found_branch, "expected at least one Branch terminator");
    }
}
