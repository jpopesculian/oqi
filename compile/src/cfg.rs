//! Control flow graph (CFG) construction over SIR.
//!
//! Builds one [`Cfg`] per function-shaped body in a [`sir::Program`]:
//! the top-level body, every subroutine, every gate, and every structured
//! OpenPulse calibration body. Opaque defcal bodies are skipped.
//!
//! A [`BasicBlock`] holds straight-line SIR statements — the kinds with no
//! control flow of their own — and ends in a [`Terminator`]. `if`, `while`,
//! `for`, `switch`, `break`, `continue`, `return`, `end` and `box` are
//! lowered to blocks and terminators; a `box` is a local scope of its
//! enclosing body (docs/scope.rst), so its statements become blocks of the
//! same graph.
//!
//! Two kinds of nested statement lists stay inside the statements and
//! expressions that own them, each built into a `Cfg` of its own (with
//! [`build_body`]) when bytecode is emitted: the scope of a `durationof`
//! and the body of an inline `cal` block.

use oqi_lex::Span;

use crate::bytecode::ProcOwner;
use crate::classical::{
    ArrayRefShape, ArrayShape, ArrayTy, IntWidth, Primitive, PrimitiveTy, ValueTy,
};
use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{
    self, ArrayLiteral, Assignment, BinOp, Binary, CalibrationBody, Call, CallTarget, Cast, Expr,
    ExprKind, For, ForIterable, Index, IndexItem, IndexKind, IndexOp, Intrinsic, LValue, RValue,
    RangeExpr, Stmt, StmtKind, SwitchCase, SwitchLabels,
};
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasicBlockId(pub usize);

#[derive(Clone)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    /// Straight-line statements only: every control-flow kind of
    /// [`StmtKind`] has become a [`Terminator`].
    pub stmts: Vec<Stmt>,
    pub terminator: Terminator,
    pub span: Span,
}

#[derive(Clone)]
pub enum Terminator {
    /// Unconditional jump.
    Goto(BasicBlockId),
    /// Two-way branch on a boolean condition.
    Branch {
        cond: Expr,
        then_bb: BasicBlockId,
        else_bb: BasicBlockId,
    },
    /// Multi-way dispatch on `target`. Cases are checked in order; if none
    /// match, control goes to `default` (the source `default` case, or the
    /// merge block after the switch when there is none).
    Switch {
        target: Expr,
        cases: Vec<(Vec<Expr>, BasicBlockId)>,
        default: BasicBlockId,
    },
    /// Entry to a `box[duration] { ... }`. The contents run from `body` and
    /// fall through to `after`; a timing pass counts `duration` and goes
    /// straight to `after` instead (docs/delays.rst: the box is pinned to
    /// that duration).
    Box {
        duration: Expr,
        body: BasicBlockId,
        after: BasicBlockId,
    },
    /// Explicit `return` from a subroutine (or implicit fall-off-the-end).
    Return(Option<RValue>),
    /// OpenQASM `end;` — program halt.
    End,
    /// Block has no successors and is statically unreachable. Used as a
    /// placeholder while constructing the graph and for blocks following a
    /// terminator within the same source body.
    Unreachable,
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
    pub owner: ProcOwner,
}

pub struct ProgramCfgs {
    pub top_level: Cfg,
    pub subroutines: Vec<Cfg>,
    pub gates: Vec<Cfg>,
    /// Parallel to `Program::calibrations`. `None` for opaque defcals.
    pub calibrations: Vec<Option<Cfg>>,
}

/// Build the graph of every body in `program`. `symbols` starts as the
/// program's symbol table and gains the hidden variables `for` loops
/// iterate with.
pub fn build_program(program: &sir::Program, symbols: &mut SymbolTable) -> Result<ProgramCfgs> {
    let top_level = build_body(&program.body, ProcOwner::TopLevel, false, symbols)?;
    let subroutines = program
        .subroutines
        .iter()
        .map(|s| build_body(&s.body, ProcOwner::Subroutine(s.symbol), true, symbols))
        .collect::<Result<_>>()?;
    let gates = program
        .gates
        .iter()
        .map(|g| build_body(&g.body, ProcOwner::Gate(g.symbol), false, symbols))
        .collect::<Result<_>>()?;
    let calibrations = program
        .calibrations
        .iter()
        .enumerate()
        .map(|(i, c)| match &c.body {
            CalibrationBody::OpenPulse(stmts) => {
                let owner = ProcOwner::Calibration(i as u32);
                build_body(stmts, owner, c.return_ty.is_some(), symbols).map(Some)
            }
            CalibrationBody::Opaque(_) => Ok(None),
        })
        .collect::<Result<_>>()?;
    Ok(ProgramCfgs {
        top_level,
        subroutines,
        gates,
        calibrations,
    })
}

/// Construct a [`Cfg`] from a body of SIR statements.
///
/// The last block is a synthetic "fall-off-the-end" exit, terminated by
/// [`Terminator::Return(None)`]. A body that finishes without an explicit
/// `return`/`end` jumps to it; otherwise it has no predecessors.
pub fn build_body(
    stmts: &[Stmt],
    owner: ProcOwner,
    allow_return: bool,
    symbols: &mut SymbolTable,
) -> Result<Cfg> {
    let mut b = CfgBuilder {
        blocks: Vec::new(),
        current: None,
        frames: Vec::new(),
        allow_return,
        symbols,
    };
    let entry = b.new_block(Span::default());
    b.current = Some(entry);
    b.lower_stmts(stmts)?;
    let exit = b.new_block(Span::default());
    b.terminate(Terminator::Goto(exit));
    b.blocks[exit.0].terminator = Terminator::Return(None);
    Ok(Cfg {
        blocks: b.blocks,
        entry,
        owner,
    })
}

/// Where `break` and `continue` go from inside a loop or switch.
struct Frame {
    break_target: BasicBlockId,
    /// `None` for a switch: `continue` belongs to the loop around it.
    continue_target: Option<BasicBlockId>,
}

struct CfgBuilder<'a> {
    blocks: Vec<BasicBlock>,
    /// `None` after a terminator, until a new statement opens a fresh block.
    current: Option<BasicBlockId>,
    frames: Vec<Frame>,
    allow_return: bool,
    symbols: &'a mut SymbolTable,
}

impl CfgBuilder<'_> {
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

    fn append(&mut self, stmt: Stmt) {
        // After a terminator this is dead code: it goes in a fresh block
        // with no incoming edges.
        let current = match self.current {
            Some(current) => current,
            None => self.new_block(stmt.span),
        };
        self.current = Some(current);
        self.blocks[current.0].stmts.push(stmt);
    }

    /// `symbol = value;`, synthesized (no annotations).
    fn assign(&mut self, symbol: SymbolId, value: Expr, span: Span) {
        self.append(Stmt {
            kind: StmtKind::Assignment(Assignment {
                target: LValue::Var(symbol),
                value: RValue::Expr(Box::new(value)),
            }),
            annotations: Vec::new(),
            span,
        });
    }

    /// End the current block, if one is open.
    fn terminate(&mut self, term: Terminator) {
        if let Some(current) = self.current.take() {
            self.blocks[current.0].terminator = term;
        }
    }

    /// End the current block with a jump to a new block and continue there.
    fn goto_new_block(&mut self, span: Span) -> BasicBlockId {
        let block = self.new_block(span);
        self.terminate(Terminator::Goto(block));
        self.current = Some(block);
        block
    }

    fn lower_stmts(&mut self, stmts: &[Stmt]) -> Result<()> {
        stmts.iter().try_for_each(|stmt| self.lower_stmt(stmt))
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        let span = stmt.span;
        let invalid =
            |message: &str| CompileError::at(ErrorKind::InvalidContext(message.into()), span);
        match &stmt.kind {
            StmtKind::If(sir::If {
                condition,
                then_body,
                else_body,
            }) => {
                let then_bb = self.new_block(span);
                let else_bb = self.new_block(span);
                let after_bb = self.new_block(span);
                self.terminate(Terminator::Branch {
                    cond: condition.clone(),
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
            }
            StmtKind::While(sir::While { condition, body }) => {
                let header_bb = self.goto_new_block(span);
                let body_bb = self.new_block(span);
                let after_bb = self.new_block(span);
                self.terminate(Terminator::Branch {
                    cond: condition.clone(),
                    then_bb: body_bb,
                    else_bb: after_bb,
                });
                self.lower_loop_body(body, body_bb, header_bb, after_bb)?;
                self.terminate(Terminator::Goto(header_bb));
                self.current = Some(after_bb);
            }
            StmtKind::For(for_loop) => self.lower_for(for_loop, span)?,
            StmtKind::Switch(sir::Switch { target, cases }) => {
                self.lower_switch(target, cases, span)?
            }
            StmtKind::Box(sir::BoxStmt { duration, body }) => {
                // A box is a scope of the enclosing body. Only a designated
                // duration needs a terminator, for timing passes to skip the
                // contents by.
                let Some(duration) = duration else {
                    return self.lower_stmts(body);
                };
                let body_bb = self.new_block(span);
                let after_bb = self.new_block(span);
                self.terminate(Terminator::Box {
                    duration: duration.clone(),
                    body: body_bb,
                    after: after_bb,
                });
                self.current = Some(body_bb);
                self.lower_stmts(body)?;
                self.terminate(Terminator::Goto(after_bb));
                self.current = Some(after_bb);
            }
            StmtKind::Break => {
                let frame = self.frames.last();
                let target = frame.ok_or_else(|| invalid("`break` outside of loop or switch"))?;
                self.terminate(Terminator::Goto(target.break_target));
            }
            StmtKind::Continue => {
                let mut targets = self.frames.iter().rev().filter_map(|f| f.continue_target);
                let target = targets
                    .next()
                    .ok_or_else(|| invalid("`continue` outside of loop"))?;
                self.terminate(Terminator::Goto(target));
            }
            StmtKind::Return(value) => {
                if !self.allow_return {
                    return Err(invalid("`return` outside of subroutine"));
                }
                self.terminate(Terminator::Return(value.clone()));
            }
            StmtKind::End => self.terminate(Terminator::End),
            StmtKind::Alias(_)
            | StmtKind::GateCall(_)
            | StmtKind::Measure(_)
            | StmtKind::Reset(_)
            | StmtKind::Barrier(_)
            | StmtKind::Delay(_)
            | StmtKind::Assignment(_)
            | StmtKind::Pragma(_)
            | StmtKind::Cal(_)
            | StmtKind::ExprStmt(_)
            | StmtKind::Nop(_) => self.append(stmt.clone()),
        }
        Ok(())
    }

    /// Lower a loop body into `body_bb`, with `continue` going to
    /// `continue_bb` and `break` to `break_bb`. Leaves the body's last
    /// block open.
    fn lower_loop_body(
        &mut self,
        body: &[Stmt],
        body_bb: BasicBlockId,
        continue_bb: BasicBlockId,
        break_bb: BasicBlockId,
    ) -> Result<()> {
        self.frames.push(Frame {
            break_target: break_bb,
            continue_target: Some(continue_bb),
        });
        self.current = Some(body_bb);
        let result = self.lower_stmts(body);
        self.frames.pop();
        result
    }

    /// A `for` loop steps a hidden counter, from which the loop variable is
    /// assigned at the top of each iteration — so assigning the loop
    /// variable in the body "does not affect the next value that the loop
    /// variable will take" (docs/classical.rst). What it iterates over is
    /// evaluated once, on entry.
    ///
    /// The counter is an `int[128]`: wide enough that stepping past the last
    /// value of any narrower loop variable type cannot wrap around (255 + 1
    /// in a `uint[8]` is 0 again) and re-enter the range.
    fn lower_for(&mut self, for_loop: &For, span: Span) -> Result<()> {
        let For {
            var,
            iterable,
            body,
        } = for_loop;
        let var_ty = self.symbols.get(*var).ty.clone();
        let counter = self.symbols.new_temp(counter_ty(), span, None);
        let counter_expr = || var_expr(counter, counter_ty(), span);

        // The first counter value, the step, the last value (inclusive),
        // and how the loop variable derives from the counter.
        let (first, step, last, element) = match iterable {
            ForIterable::Range(RangeExpr { start, step, end }) => {
                let end = end.as_deref().ok_or_else(|| {
                    let message = "a `for` loop range needs an end";
                    CompileError::at(ErrorKind::InvalidContext(message.into()), span)
                })?;
                let first = match start {
                    Some(start) => widen(start.as_ref().clone()),
                    None => int_literal(0, span),
                };
                let step = match step {
                    Some(step) => self.once(widen(step.as_ref().clone()), span),
                    None => int_literal(1, span),
                };
                if literal_int(&step) == Some(0) {
                    let message = "for-loop range step must be non-zero";
                    return Err(CompileError::at(
                        ErrorKind::InvalidContext(message.into()),
                        span,
                    ));
                }
                let last = self.once(widen(end.clone()), span);
                (first, step, last, cast(counter_expr(), var_ty.clone()))
            }
            ForIterable::Set(items) => {
                // The members are evaluated into an array, then indexed.
                let element_ty = var_ty.scalar_ty().ok_or_else(|| {
                    let message = "the loop variable of a set must be a scalar";
                    CompileError::at(ErrorKind::InvalidContext(message.into()), span)
                })?;
                if items.is_empty() {
                    return Ok(());
                }
                let shape = ArrayShape::new(vec![items.len()])
                    .map_err(|e| crate::types::invalid_type(e, span))?;
                let values = Expr {
                    kind: ExprKind::ArrayLiteral(ArrayLiteral {
                        items: items.clone(),
                        span,
                    }),
                    ty: Type::from(ArrayTy::new(element_ty, shape)),
                    span,
                };
                let values = self.once(values, span);
                let last = int_literal(items.len() as i128 - 1, span);
                let element = element_at(values, counter_expr(), var_ty.clone());
                (int_literal(0, span), int_literal(1, span), last, element)
            }
            ForIterable::Expr(values) => {
                let values = self.once(values.as_ref().clone(), span);
                let last = self.last_index(&values, span)?;
                let item_ty = item_type(&values.ty).unwrap_or_else(|| var_ty.clone());
                let element = cast(element_at(values, counter_expr(), item_ty), var_ty.clone());
                (int_literal(0, span), int_literal(1, span), last, element)
            }
        };

        self.assign(counter, first, span);
        let header_bb = self.goto_new_block(span);
        let body_bb = self.new_block(span);
        let latch_bb = self.new_block(span);
        let after_bb = self.new_block(span);
        self.terminate(Terminator::Branch {
            cond: in_range(counter_expr(), &step, last, span),
            then_bb: body_bb,
            else_bb: after_bb,
        });

        self.current = Some(body_bb);
        self.assign(*var, element, span);
        self.lower_loop_body(body, body_bb, latch_bb, after_bb)?;
        self.terminate(Terminator::Goto(latch_bb));

        self.current = Some(latch_bb);
        let next = binary(BinOp::Add, counter_expr(), step, counter_ty(), span);
        self.assign(counter, next, span);
        self.terminate(Terminator::Goto(header_bb));

        self.current = Some(after_bb);
        Ok(())
    }

    /// `value` as an expression that can be read again without evaluating
    /// it again: a literal as it is, anything else through a hidden variable
    /// assigned here.
    fn once(&mut self, value: Expr, span: Span) -> Expr {
        if matches!(value.kind, ExprKind::Literal(_)) {
            return value;
        }
        let temp = self.symbols.new_temp(value.ty.clone(), span, None);
        let read = var_expr(temp, value.ty.clone(), span);
        self.assign(temp, value, span);
        read
    }

    /// The last index of the register or one-dimensional array `values`.
    fn last_index(&mut self, values: &Expr, span: Span) -> Result<Expr> {
        let len = match values.ty.value_ty() {
            Some(ValueTy::Scalar(PrimitiveTy::BitReg(n))) => Some(n as usize),
            Some(ValueTy::Scalar(_)) => None,
            Some(ValueTy::Array(ty)) => one_dimension(ty.shape().dims()),
            Some(ValueTy::ArrayRef(ty)) => match ty.shape() {
                ArrayRefShape::Fixed(shape) => one_dimension(shape.dims()),
                // An unspecified-length reference knows its size at run time.
                ArrayRefShape::Dim(rank) if rank.get() == 1 => {
                    let uint = Type::Classical(ValueTy::uint(IntWidth::B64));
                    let len = Expr {
                        kind: ExprKind::Call(Call {
                            callee: CallTarget::Intrinsic(Intrinsic::Sizeof),
                            args: vec![values.clone()],
                        }),
                        ty: uint,
                        span,
                    };
                    let last = binary(
                        BinOp::Sub,
                        widen(len),
                        int_literal(1, span),
                        counter_ty(),
                        span,
                    );
                    return Ok(self.once(last, span));
                }
                ArrayRefShape::Dim(_) => None,
            },
            None => None,
        };
        let len = len.ok_or_else(|| {
            let message = format!(
                "a `for` loop iterates over a range, a set, a bit register or a \
                 one-dimensional array, not `{}`",
                values.ty
            );
            CompileError::at(ErrorKind::InvalidContext(message), values.span)
        })?;
        Ok(int_literal(len as i128 - 1, span))
    }

    fn lower_switch(&mut self, target: &Expr, cases: &[SwitchCase], span: Span) -> Result<()> {
        let after_bb = self.new_block(span);

        // Allocate one block per case up front so the dispatcher can reference
        // them all before any case body is lowered.
        let mut case_blocks = Vec::new();
        let mut default = after_bb;
        let mut bodies = Vec::with_capacity(cases.len());
        for case in cases {
            let bb_span = case.body.first().map(|s| s.span).unwrap_or(span);
            let bb = self.new_block(bb_span);
            bodies.push((bb, &case.body));
            match &case.labels {
                SwitchLabels::Values(values) => case_blocks.push((values.clone(), bb)),
                SwitchLabels::Default => default = bb,
            }
        }

        self.terminate(Terminator::Switch {
            target: target.clone(),
            cases: case_blocks,
            default,
        });

        self.frames.push(Frame {
            break_target: after_bb,
            continue_target: None,
        });
        for (bb, body) in bodies {
            self.current = Some(bb);
            self.lower_stmts(body)?;
            self.terminate(Terminator::Goto(after_bb));
        }
        self.frames.pop();
        self.current = Some(after_bb);
        Ok(())
    }
}

/// The type of a loop counter.
fn counter_ty() -> Type {
    Type::Classical(ValueTy::int(IntWidth::B128))
}

fn var_expr(symbol: SymbolId, ty: Type, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Var(symbol),
        ty,
        span,
    }
}

fn int_literal(value: i128, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Literal(Primitive::int(value)),
        ty: counter_ty(),
        span,
    }
}

/// The value of an integer literal.
fn literal_int(e: &Expr) -> Option<i128> {
    match &e.kind {
        ExprKind::Literal(Primitive::Int(i)) => Some(*i),
        ExprKind::Literal(Primitive::Uint(u)) => i128::try_from(*u).ok(),
        _ => None,
    }
}

fn binary(op: BinOp, left: Expr, right: Expr, ty: Type, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Binary(Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }),
        ty,
        span,
    }
}

/// `e` converted to `ty`, if it is not of that type already.
fn cast(e: Expr, ty: Type) -> Expr {
    if e.ty == ty {
        return e;
    }
    Expr {
        span: e.span,
        kind: ExprKind::Cast(Cast {
            target_ty: ty.clone(),
            operand: Box::new(e),
        }),
        ty,
    }
}

/// An integer expression as a loop-counter value.
fn widen(e: Expr) -> Expr {
    match literal_int(&e) {
        Some(value) => int_literal(value, e.span),
        None => cast(e, counter_ty()),
    }
}

/// Whether `counter` is still within a range ending at `last` (inclusive)
/// that is walked in steps of `step`.
fn in_range(counter: Expr, step: &Expr, last: Expr, span: Span) -> Expr {
    let bool_ty = || Type::Classical(ValueTy::bool());
    let compare = |op, left: Expr, right: Expr| binary(op, left, right, bool_ty(), span);
    match literal_int(step) {
        Some(step) if step > 0 => compare(BinOp::Lte, counter, last),
        Some(_) => compare(BinOp::Gte, counter, last),
        // The direction is only known at run time; a zero step walks nothing.
        None => {
            let zero = || int_literal(0, span);
            let up = compare(
                BinOp::LogAnd,
                compare(BinOp::Gt, step.clone(), zero()),
                compare(BinOp::Lte, counter.clone(), last.clone()),
            );
            let down = compare(
                BinOp::LogAnd,
                compare(BinOp::Lt, step.clone(), zero()),
                compare(BinOp::Gte, counter, last),
            );
            compare(BinOp::LogOr, up, down)
        }
    }
}

/// The length of a one-dimensional shape.
fn one_dimension(shape: &[usize]) -> Option<usize> {
    match shape {
        [len] => Some(*len),
        _ => None,
    }
}

/// The type of one item of the register or one-dimensional array type `ty`.
pub(crate) fn item_type(ty: &Type) -> Option<Type> {
    let index = [oqi_classical::Index::Item(0)];
    ty.value_ty()?.get(&index).ok().map(Type::from)
}

/// `values[index]`, of type `ty`.
fn element_at(values: Expr, index: Expr, ty: Type) -> Expr {
    let span = values.span;
    Expr {
        kind: ExprKind::Index(Index {
            base: Box::new(values),
            index: IndexOp {
                kind: IndexKind::Items(vec![IndexItem::Single(Box::new(index))]),
                span,
            },
        }),
        ty,
        span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stmt(kind: StmtKind) -> Stmt {
        Stmt {
            kind,
            annotations: vec![],
            span: Span::default(),
        }
    }

    fn pragma(s: &str) -> Stmt {
        stmt(StmtKind::Pragma(s.into()))
    }

    fn bool_literal() -> Expr {
        Expr {
            kind: ExprKind::Literal(Primitive::bit(false)),
            ty: Type::Classical(ValueTy::bool()),
            span: Span::default(),
        }
    }

    fn build(stmts: &[Stmt], allow_return: bool) -> Result<Cfg> {
        build_body(
            stmts,
            ProcOwner::TopLevel,
            allow_return,
            &mut SymbolTable::new(),
        )
    }

    fn pragmas(cfg: &Cfg) -> Vec<&str> {
        let stmts = cfg.blocks.iter().flat_map(|b| &b.stmts);
        stmts
            .filter_map(|s| match &s.kind {
                StmtKind::Pragma(p) => Some(p.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_plain_box_is_part_of_the_enclosing_block() {
        let boxed = stmt(StmtKind::Box(sir::BoxStmt {
            duration: None,
            body: vec![pragma("inside")],
        }));
        let cfg = build(&[pragma("before"), boxed, pragma("after")], false).unwrap();
        let entry = &cfg.blocks[cfg.entry.0];
        assert_eq!(entry.stmts.len(), 3);
        assert!(matches!(
            entry.terminator,
            Terminator::Goto(BasicBlockId(1))
        ));
    }

    #[test]
    fn a_timed_box_branches_around_its_contents() {
        let boxed = stmt(StmtKind::Box(sir::BoxStmt {
            duration: Some(bool_literal()),
            body: vec![pragma("inside")],
        }));
        let cfg = build(&[boxed, pragma("after")], false).unwrap();
        let Terminator::Box { body, after, .. } = &cfg.blocks[cfg.entry.0].terminator else {
            panic!("expected a box terminator");
        };
        assert!(matches!(cfg.blocks[body.0].terminator, Terminator::Goto(t) if t == *after));
        assert!(matches!(
            cfg.blocks[body.0].stmts[0].kind,
            StmtKind::Pragma(_)
        ));
        assert_eq!(pragmas(&cfg), ["inside", "after"]);
    }

    #[test]
    fn return_leaves_a_box_in_a_subroutine() {
        let boxed = stmt(StmtKind::Box(sir::BoxStmt {
            duration: None,
            body: vec![stmt(StmtKind::Return(None))],
        }));
        let cfg = build(std::slice::from_ref(&boxed), true).unwrap();
        assert!(matches!(
            cfg.blocks[cfg.entry.0].terminator,
            Terminator::Return(None)
        ));
        assert!(build(&[boxed], false).is_err());
    }

    #[test]
    fn nested_bodies_stay_in_their_statements() {
        // The scope of a `durationof` and an inline `cal` body are not part
        // of the graph: they are built separately at emission.
        let branch = stmt(StmtKind::If(sir::If {
            condition: bool_literal(),
            then_body: vec![pragma("a")],
            else_body: None,
        }));
        let duration = Expr {
            kind: ExprKind::DurationOf(vec![branch.clone()]),
            ty: Type::Classical(ValueTy::duration()),
            span: Span::default(),
        };
        let cal = stmt(StmtKind::Cal(CalibrationBody::OpenPulse(vec![branch])));
        let cfg = build(&[stmt(StmtKind::ExprStmt(duration)), cal], false).unwrap();
        assert_eq!(cfg.blocks.len(), 2, "entry and exit only");
        assert_eq!(cfg.blocks[cfg.entry.0].stmts.len(), 2);
    }
}
