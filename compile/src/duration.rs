//! `durationof` intrinsic resolution.
//!
//! The [`Timings`] trait is implemented by a backend to supply durations for
//! measurements, resets and gate calls. [`resolve_durationof`] walks a compiled
//! [`crate::sir::Program`] and replaces every `ExprKind::DurationOf` with a
//! constant [`crate::classical::Duration`] literal, stacking per-qubit
//! durations and taking the maximum across qubits at sync points.

use std::collections::HashMap;

use oqi_lex::Span;

use crate::classical::{
    Duration, DurationUnit, PrimitiveTy, Value, ValueTy, value_as_usize,
};
use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{
    BinOp, CallTarget, Expr, ExprKind, GateCallTarget, GateDecl, GateModifier, IndexItem,
    IndexKind, IndexOp, Intrinsic, MeasureExpr, MeasureExprKind, Program, QubitOperand, Stmt,
    StmtKind, UnOp,
};
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::{CompileOptions, Type};

// ── Public types ────────────────────────────────────────────────────────

/// A duration that may use the backend-dependent `dt` unit.
///
/// `Timings` callbacks return this so the backend can express durations in
/// either SI units (`ns`, `us`, `ms`, `s`) or in `dt` units — the latter are
/// resolved via [`CompileOptions::dt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimingDuration {
    /// Duration in standard SI units.
    Si(Duration),
    /// Duration in backend-dependent `dt` units.
    Dt(f64),
}

impl TimingDuration {
    pub const fn ns(v: f64) -> Self {
        Self::Si(Duration::new(v, DurationUnit::Ns))
    }
    pub const fn us(v: f64) -> Self {
        Self::Si(Duration::new(v, DurationUnit::Us))
    }
    pub const fn ms(v: f64) -> Self {
        Self::Si(Duration::new(v, DurationUnit::Ms))
    }
    pub const fn s(v: f64) -> Self {
        Self::Si(Duration::new(v, DurationUnit::S))
    }
    pub const fn dt(v: f64) -> Self {
        Self::Dt(v)
    }
    pub const fn zero() -> Self {
        Self::Si(Duration::new(0.0, DurationUnit::Ns))
    }

    /// Convert to a concrete SI [`Duration`] using the supplied `dt` unit.
    pub fn resolve(self, dt: &Duration) -> Duration {
        match self {
            Self::Si(d) => d,
            Self::Dt(v) => Duration::new(v * dt.value, dt.unit),
        }
    }
}

impl From<Duration> for TimingDuration {
    fn from(value: Duration) -> Self {
        Self::Si(value)
    }
}

/// Return of [`Timings::gate_call`].
pub enum GateCallTiming {
    /// The gate call has a known duration.
    Duration(TimingDuration),
    /// Enter the gate body and recursively compute its duration.
    Enter,
}

/// A gate modifier with its designator already evaluated.
#[derive(Debug, Clone)]
pub enum ResolvedGateModifier {
    Inv,
    Pow(Value),
    Ctrl(usize),
    NegCtrl(usize),
}

/// A qubit operand with symbols resolved to names (not ids).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QubitRef {
    /// Named qubit symbol, with optional index (e.g., `q[0]` is
    /// `Symbol { name: "q", index: Some(0) }`, and a single `qubit q;` is
    /// `Symbol { name: "q", index: None }`).
    Symbol { name: String, index: Option<usize> },
    /// Hardware qubit (e.g., `$3`).
    Hardware(usize),
}

/// Arguments passed to [`Timings::measurement`].
pub struct MeasureArgs {
    pub qubits: Vec<QubitRef>,
}

/// Arguments passed to [`Timings::reset`].
pub struct ResetArgs {
    pub qubits: Vec<QubitRef>,
}

/// Arguments passed to [`Timings::gate_call`].
pub struct GateCallArgs {
    /// Gate name (`"h"`, `"cx"`, or `"gphase"`).
    pub name: String,
    pub modifiers: Vec<ResolvedGateModifier>,
    pub args: Vec<Value>,
    pub qubits: Vec<QubitRef>,
}

/// Backend-supplied durations for the operations that can appear in a
/// `durationof` scope.
pub trait Timings {
    fn measurement(&self, args: &MeasureArgs) -> TimingDuration;
    fn reset(&self, args: &ResetArgs) -> TimingDuration;
    fn gate_call(&self, args: &GateCallArgs) -> GateCallTiming;
}

// ── Public entry points ─────────────────────────────────────────────────

/// Resolve every `durationof` expression in `program` to a constant
/// duration literal, using `timings` to supply per-op durations.
pub fn resolve_durationof<T: Timings>(
    program: &mut Program,
    timings: &T,
    options: &CompileOptions,
) -> Result<()> {
    // Pass 1: top-level body. `ctx` takes a shared borrow of `program.gates`
    // which is fine alongside the mutable borrow of the disjoint `program.body`
    // field.
    {
        let mut ctx = ResolveCtx {
            timings,
            symbols: &program.symbols,
            gates: &program.gates,
            options,
        };
        for stmt in &mut program.body {
            ctx.visit_stmt(stmt)?;
        }
    }

    // Pass 2: gate bodies. Detach each body in turn so we can mutate it while
    // the ctx holds a shared borrow of the rest of `program.gates`. Gates are
    // non-recursive in OpenQASM, so a gate's own body being empty during its
    // own processing is not a problem.
    let n_gates = program.gates.len();
    for i in 0..n_gates {
        let mut body = std::mem::take(&mut program.gates[i].body.body);
        let result: Result<()> = (|| {
            let mut ctx = ResolveCtx {
                timings,
                symbols: &program.symbols,
                gates: &program.gates,
                options,
            };
            for stmt in &mut body {
                ctx.visit_stmt(stmt)?;
            }
            Ok(())
        })();
        program.gates[i].body.body = body;
        result?;
    }

    // Pass 3: subroutine bodies.
    let n_sub = program.subroutines.len();
    for i in 0..n_sub {
        let mut body = std::mem::take(&mut program.subroutines[i].body);
        let result: Result<()> = (|| {
            let mut ctx = ResolveCtx {
                timings,
                symbols: &program.symbols,
                gates: &program.gates,
                options,
            };
            for stmt in &mut body {
                ctx.visit_stmt(stmt)?;
            }
            Ok(())
        })();
        program.subroutines[i].body = body;
        result?;
    }

    Ok(())
}

// ── Visitor: rewrites ExprKind::DurationOf in place ─────────────────────

struct ResolveCtx<'a, T: Timings> {
    timings: &'a T,
    symbols: &'a SymbolTable,
    gates: &'a [GateDecl],
    options: &'a CompileOptions,
}

impl<'a, T: Timings> ResolveCtx<'a, T> {
    fn visit_stmts(&mut self, stmts: &mut [Stmt]) -> Result<()> {
        for stmt in stmts {
            self.visit_stmt(stmt)?;
        }
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &mut Stmt) -> Result<()> {
        match &mut stmt.kind {
            StmtKind::ClassicalDecl { init, .. } => {
                if let Some(init) = init {
                    self.visit_decl_init(init)?;
                }
            }
            StmtKind::ConstDecl { init, .. } => self.visit_expr(init)?,
            StmtKind::Alias { value, .. } => {
                for e in value {
                    self.visit_expr(e)?;
                }
            }
            StmtKind::GateCall { args, qubits, .. } => {
                for a in args {
                    self.visit_expr(a)?;
                }
                for q in qubits {
                    self.visit_qubit_operand(q)?;
                }
            }
            StmtKind::Measure { measure, .. } => self.visit_measure_expr(measure)?,
            StmtKind::Reset { operand } => self.visit_qubit_operand(operand)?,
            StmtKind::Barrier { operands } | StmtKind::Nop { operands } => {
                for o in operands {
                    self.visit_qubit_operand(o)?;
                }
            }
            StmtKind::Delay { duration, operands } => {
                self.visit_expr(duration)?;
                for o in operands {
                    self.visit_qubit_operand(o)?;
                }
            }
            StmtKind::Box { duration, body } => {
                if let Some(d) = duration {
                    self.visit_expr(d)?;
                }
                self.visit_stmts(body)?;
            }
            StmtKind::Assignment { value, .. } => self.visit_assign_value(value)?,
            StmtKind::If { condition, then_body, else_body } => {
                self.visit_expr(condition)?;
                self.visit_stmts(then_body)?;
                if let Some(eb) = else_body {
                    self.visit_stmts(eb)?;
                }
            }
            StmtKind::For { iterable, body, .. } => {
                self.visit_for_iterable(iterable)?;
                self.visit_stmts(body)?;
            }
            StmtKind::While { condition, body } => {
                self.visit_expr(condition)?;
                self.visit_stmts(body)?;
            }
            StmtKind::Switch { target, cases } => {
                self.visit_expr(target)?;
                for c in cases {
                    if let crate::sir::SwitchLabels::Values(v) = &mut c.labels {
                        for e in v {
                            self.visit_expr(e)?;
                        }
                    }
                    self.visit_stmts(&mut c.body)?;
                }
            }
            StmtKind::Return(Some(rv)) => match rv {
                crate::sir::ReturnValue::Expr(e) => self.visit_expr(e)?,
                crate::sir::ReturnValue::Measure(m) => self.visit_measure_expr(m)?,
            },
            StmtKind::ExprStmt(e) => self.visit_expr(e)?,
            _ => {}
        }
        Ok(())
    }

    fn visit_assign_value(&mut self, value: &mut crate::sir::AssignValue) -> Result<()> {
        match value {
            crate::sir::AssignValue::Expr(e) => self.visit_expr(e),
            crate::sir::AssignValue::Measure(m) => self.visit_measure_expr(m),
        }
    }

    fn visit_decl_init(&mut self, init: &mut crate::sir::DeclInit) -> Result<()> {
        match init {
            crate::sir::DeclInit::Expr(e) => self.visit_expr(e),
            crate::sir::DeclInit::Measure(m) => self.visit_measure_expr(m),
            crate::sir::DeclInit::ArrayLiteral(al) => self.visit_array_literal(al),
        }
    }

    fn visit_array_literal(&mut self, al: &mut crate::sir::ArrayLiteral) -> Result<()> {
        for item in &mut al.items {
            match item {
                crate::sir::ArrayLiteralItem::Expr(e) => self.visit_expr(e)?,
                crate::sir::ArrayLiteralItem::Nested(n) => self.visit_array_literal(n)?,
            }
        }
        Ok(())
    }

    fn visit_measure_expr(&mut self, m: &mut MeasureExpr) -> Result<()> {
        match &mut m.kind {
            MeasureExprKind::Measure { operand } => self.visit_qubit_operand(operand),
            MeasureExprKind::QuantumCall { args, qubits, .. } => {
                for a in args {
                    self.visit_expr(a)?;
                }
                for q in qubits {
                    self.visit_qubit_operand(q)?;
                }
                Ok(())
            }
        }
    }

    fn visit_qubit_operand(&mut self, op: &mut QubitOperand) -> Result<()> {
        if let QubitOperand::Indexed { indices, .. } = op {
            for idx in indices {
                self.visit_index_op(idx)?;
            }
        }
        Ok(())
    }

    fn visit_index_op(&mut self, op: &mut IndexOp) -> Result<()> {
        match &mut op.kind {
            IndexKind::Set(exprs) => {
                for e in exprs {
                    self.visit_expr(e)?;
                }
            }
            IndexKind::Items(items) => {
                for item in items {
                    match item {
                        IndexItem::Single(e) => self.visit_expr(e)?,
                        IndexItem::Range(r) => {
                            if let Some(e) = &mut r.start {
                                self.visit_expr(e)?;
                            }
                            if let Some(e) = &mut r.step {
                                self.visit_expr(e)?;
                            }
                            if let Some(e) = &mut r.end {
                                self.visit_expr(e)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn visit_for_iterable(&mut self, it: &mut crate::sir::ForIterable) -> Result<()> {
        match it {
            crate::sir::ForIterable::Range { start, step, end } => {
                if let Some(e) = start {
                    self.visit_expr(e)?;
                }
                if let Some(e) = step {
                    self.visit_expr(e)?;
                }
                if let Some(e) = end {
                    self.visit_expr(e)?;
                }
                Ok(())
            }
            crate::sir::ForIterable::Set(v) => {
                for e in v {
                    self.visit_expr(e)?;
                }
                Ok(())
            }
            crate::sir::ForIterable::Expr(e) => self.visit_expr(e),
        }
    }

    fn visit_expr(&mut self, expr: &mut Expr) -> Result<()> {
        // Rewrite durationof in place, then recurse into children.
        if let ExprKind::DurationOf(stmts) = &expr.kind {
            let duration = self.compute_scope_duration(stmts, expr.span)?;
            expr.kind = ExprKind::Literal(Value::from(duration));
            expr.ty = Type::Classical(ValueTy::duration());
            return Ok(());
        }
        match &mut expr.kind {
            ExprKind::Binary { left, right, .. } => {
                self.visit_expr(left)?;
                self.visit_expr(right)?;
            }
            ExprKind::Unary { operand, .. } => self.visit_expr(operand)?,
            ExprKind::Cast { operand, .. } => self.visit_expr(operand)?,
            ExprKind::Index { base, index } => {
                self.visit_expr(base)?;
                self.visit_index_op(index)?;
            }
            ExprKind::Call { args, .. } => {
                for a in args {
                    self.visit_expr(a)?;
                }
            }
            ExprKind::Literal(_) | ExprKind::Var(_) | ExprKind::HardwareQubit(_) => {}
            ExprKind::DurationOf(_) => unreachable!("handled above"),
        }
        Ok(())
    }

    // ── Core: compute the duration of a scope ───────────────────────────

    fn compute_scope_duration(&self, stmts: &[Stmt], span: Span) -> Result<Duration> {
        let mut tracker = Tracker::default();
        let frame = Frame::default();
        for stmt in stmts {
            self.process_stmt(stmt, &mut tracker, &frame, span)?;
        }
        Ok(tracker.total())
    }

    fn process_stmts(
        &self,
        stmts: &[Stmt],
        tracker: &mut Tracker,
        frame: &Frame,
        span: Span,
    ) -> Result<()> {
        for stmt in stmts {
            self.process_stmt(stmt, tracker, frame, span)?;
        }
        Ok(())
    }

    fn process_stmt(
        &self,
        stmt: &Stmt,
        tracker: &mut Tracker,
        frame: &Frame,
        outer_span: Span,
    ) -> Result<()> {
        let span = stmt.span;
        match &stmt.kind {
            StmtKind::GateCall {
                gate,
                modifiers,
                args,
                qubits,
            } => self.process_gate_call(gate, modifiers, args, qubits, tracker, frame, span),
            StmtKind::Measure { measure, .. } => self.process_measure(measure, tracker, frame),
            StmtKind::ClassicalDecl { init: Some(crate::sir::DeclInit::Measure(m)), .. } => {
                self.process_measure(m, tracker, frame)
            }
            StmtKind::Reset { operand } => {
                let qr = resolve_qubit_operand(operand, self.symbols, frame, span)?;
                let args = ResetArgs { qubits: qr.clone() };
                let dur = self.timings.reset(&args).resolve(&self.options.dt);
                tracker.advance(&qr, dur);
                Ok(())
            }
            StmtKind::Delay { duration, operands } => {
                let dur_val = self.eval_const_expr(duration, frame)?;
                let dur = value_to_duration(&dur_val, duration.span)?;
                let qubits = resolve_qubit_operands(operands, self.symbols, frame, span)?;
                if qubits.is_empty() {
                    return Err(err(ErrorKind::InvalidContext(
                        "delay requires at least one qubit operand".into(),
                    ), span));
                }
                tracker.advance(&qubits, dur);
                Ok(())
            }
            StmtKind::Barrier { operands } | StmtKind::Nop { operands } => {
                let qubits = resolve_qubit_operands(operands, self.symbols, frame, span)?;
                if !qubits.is_empty() {
                    tracker.sync(&qubits);
                }
                Ok(())
            }
            StmtKind::Box { duration, body } => {
                if let Some(dur_expr) = duration {
                    let v = self.eval_const_expr(dur_expr, frame)?;
                    let dur = value_to_duration(&v, dur_expr.span)?;
                    // A box with an explicit duration pins the enclosed scope
                    // to that duration across its qubits.
                    let inner = self.compute_scope_duration_with(body, frame, span)?;
                    let (qubits, _) = inner;
                    if !qubits.is_empty() {
                        tracker.advance(&qubits, dur);
                    }
                } else {
                    let (qubits, dur) = self.compute_scope_duration_with(body, frame, span)?;
                    if !qubits.is_empty() {
                        tracker.advance(&qubits, dur);
                    }
                }
                Ok(())
            }
            _ => Err(err(
                ErrorKind::InvalidContext(format!(
                    "statement not supported inside `durationof`"
                )),
                if span == Span::default() { outer_span } else { span },
            )),
        }
    }

    fn compute_scope_duration_with(
        &self,
        stmts: &[Stmt],
        frame: &Frame,
        span: Span,
    ) -> Result<(Vec<QubitRef>, Duration)> {
        let mut inner = Tracker::default();
        self.process_stmts(stmts, &mut inner, frame, span)?;
        let qubits: Vec<QubitRef> = inner.times.keys().cloned().collect();
        let total = inner.total();
        Ok((qubits, total))
    }

    fn process_gate_call(
        &self,
        gate: &GateCallTarget,
        modifiers: &[GateModifier],
        args: &[Expr],
        qubits: &[QubitOperand],
        tracker: &mut Tracker,
        frame: &Frame,
        span: Span,
    ) -> Result<()> {
        let name = match gate {
            GateCallTarget::Symbol(sid) => self.symbols.get(*sid).name.clone(),
            GateCallTarget::GPhase => "gphase".to_string(),
        };
        let qubit_refs = resolve_qubit_operands(qubits, self.symbols, frame, span)?;
        let resolved_args: Vec<Value> = args
            .iter()
            .map(|a| self.eval_const_expr(a, frame))
            .collect::<Result<_>>()?;
        let resolved_mods = modifiers
            .iter()
            .map(|m| self.resolve_modifier(m, frame))
            .collect::<Result<_>>()?;

        let call_args = GateCallArgs {
            name,
            modifiers: resolved_mods,
            args: resolved_args.clone(),
            qubits: qubit_refs.clone(),
        };
        match self.timings.gate_call(&call_args) {
            GateCallTiming::Duration(d) => {
                if !qubit_refs.is_empty() {
                    tracker.advance(&qubit_refs, d.resolve(&self.options.dt));
                }
                Ok(())
            }
            GateCallTiming::Enter => {
                let sid = match gate {
                    GateCallTarget::Symbol(sid) => *sid,
                    GateCallTarget::GPhase => {
                        return Err(err(
                            ErrorKind::InvalidContext(
                                "cannot enter a gphase call to compute duration".into(),
                            ),
                            span,
                        ));
                    }
                };
                let decl = self
                    .gates
                    .iter()
                    .find(|g| g.symbol == sid)
                    .ok_or_else(|| {
                        err(
                            ErrorKind::InvalidContext(format!(
                                "gate `{}` not found for duration recursion",
                                call_args.name
                            )),
                            span,
                        )
                    })?;
                if decl.params.len() != resolved_args.len() {
                    return Err(err(
                        ErrorKind::InvalidContext(format!(
                            "gate `{}` expects {} arg(s), got {}",
                            call_args.name,
                            decl.params.len(),
                            resolved_args.len()
                        )),
                        span,
                    ));
                }
                if decl.qubits.len() != qubit_refs.len() {
                    return Err(err(
                        ErrorKind::InvalidContext(format!(
                            "gate `{}` expects {} qubit(s), got {}",
                            call_args.name,
                            decl.qubits.len(),
                            qubit_refs.len()
                        )),
                        span,
                    ));
                }
                let mut inner_frame = Frame::default();
                for (sym, val) in decl.params.iter().zip(resolved_args.iter()) {
                    inner_frame.params.insert(*sym, val.clone());
                }
                for (sym, qr) in decl.qubits.iter().zip(qubit_refs.iter()) {
                    inner_frame.qubits.insert(*sym, qr.clone());
                }
                let (inner_qubits, inner_dur) =
                    self.compute_scope_duration_with(&decl.body.body, &inner_frame, span)?;
                let _ = inner_qubits;
                if !qubit_refs.is_empty() {
                    tracker.advance(&qubit_refs, inner_dur);
                }
                Ok(())
            }
        }
    }

    fn process_measure(
        &self,
        measure: &MeasureExpr,
        tracker: &mut Tracker,
        frame: &Frame,
    ) -> Result<()> {
        match &measure.kind {
            MeasureExprKind::Measure { operand } => {
                let qr = resolve_qubit_operand(operand, self.symbols, frame, measure.span)?;
                let args = MeasureArgs { qubits: qr.clone() };
                let dur = self.timings.measurement(&args).resolve(&self.options.dt);
                tracker.advance(&qr, dur);
                Ok(())
            }
            MeasureExprKind::QuantumCall {
                callee,
                args,
                qubits,
            } => {
                // A quantum call used as a measure expression — treat as a
                // gate call, but the Timings callback decides the semantics.
                let mods: Vec<GateModifier> = vec![];
                self.process_gate_call(
                    &GateCallTarget::Symbol(*callee),
                    &mods,
                    args,
                    qubits,
                    tracker,
                    frame,
                    measure.span,
                )
            }
        }
    }

    fn resolve_modifier(
        &self,
        m: &GateModifier,
        frame: &Frame,
    ) -> Result<ResolvedGateModifier> {
        Ok(match m {
            GateModifier::Inv => ResolvedGateModifier::Inv,
            GateModifier::Pow(expr) => {
                ResolvedGateModifier::Pow(self.eval_const_expr(expr, frame)?)
            }
            GateModifier::Ctrl(n) => ResolvedGateModifier::Ctrl(*n),
            GateModifier::NegCtrl(n) => ResolvedGateModifier::NegCtrl(*n),
        })
    }

    // ── Constant evaluation over SIR (with param overlay) ───────────────

    fn eval_const_expr(&self, expr: &Expr, frame: &Frame) -> Result<Value> {
        match &expr.kind {
            ExprKind::Literal(v) => Ok(v.clone()),
            ExprKind::Var(sid) => {
                if let Some(v) = frame.params.get(sid) {
                    return Ok(v.clone());
                }
                self.symbols
                    .get(*sid)
                    .const_value
                    .clone()
                    .ok_or_else(|| err(ErrorKind::NonConstantExpression, expr.span))
            }
            ExprKind::Binary { op, left, right } => {
                let lv = self.eval_const_expr(left, frame)?;
                let rv = self.eval_const_expr(right, frame)?;
                apply_binop(*op, lv, rv, expr.span)
            }
            ExprKind::Unary { op, operand } => {
                let v = self.eval_const_expr(operand, frame)?;
                apply_unop(*op, v, expr.span)
            }
            ExprKind::Cast { target_ty, operand } => {
                let v = self.eval_const_expr(operand, frame)?;
                let ty = target_ty
                    .value_ty()
                    .ok_or_else(|| err(ErrorKind::NonConstantExpression, expr.span))?;
                v.cast(ty)
                    .map_err(|_| err(ErrorKind::NonConstantExpression, expr.span))
            }
            ExprKind::Call { callee, args } => match callee {
                CallTarget::Intrinsic(i) => self.eval_intrinsic(i, args, frame, expr.span),
                CallTarget::Symbol(_) => {
                    Err(err(ErrorKind::NonConstantExpression, expr.span))
                }
            },
            ExprKind::DurationOf(stmts) => {
                let d = self.compute_scope_duration(stmts, expr.span)?;
                Ok(Value::from(d))
            }
            ExprKind::Index { .. } | ExprKind::HardwareQubit(_) => {
                Err(err(ErrorKind::NonConstantExpression, expr.span))
            }
        }
    }

    fn eval_intrinsic(
        &self,
        i: &Intrinsic,
        args: &[Expr],
        frame: &Frame,
        span: Span,
    ) -> Result<Value> {
        let vals: Vec<Value> = args
            .iter()
            .map(|a| self.eval_const_expr(a, frame))
            .collect::<Result<_>>()?;
        let map_err = |_| err(ErrorKind::NonConstantExpression, span);
        match i {
            Intrinsic::Sin => one_arg(vals, span)?.sin_().map_err(map_err),
            Intrinsic::Cos => one_arg(vals, span)?.cos_().map_err(map_err),
            Intrinsic::Tan => one_arg(vals, span)?.tan_().map_err(map_err),
            Intrinsic::Arcsin => one_arg(vals, span)?.arcsin_().map_err(map_err),
            Intrinsic::Arccos => one_arg(vals, span)?.arccos_().map_err(map_err),
            Intrinsic::Arctan => one_arg(vals, span)?.arctan_().map_err(map_err),
            Intrinsic::Exp => one_arg(vals, span)?.exp_().map_err(map_err),
            Intrinsic::Log => one_arg(vals, span)?.log_().map_err(map_err),
            Intrinsic::Sqrt => one_arg(vals, span)?.sqrt_().map_err(map_err),
            Intrinsic::Ceiling => one_arg(vals, span)?.ceiling_().map_err(map_err),
            Intrinsic::Floor => one_arg(vals, span)?.floor_().map_err(map_err),
            Intrinsic::Popcount => one_arg(vals, span)?.popcount_().map_err(map_err),
            Intrinsic::Real => one_arg(vals, span)?.real_().map_err(map_err),
            Intrinsic::Imag => one_arg(vals, span)?.imag_().map_err(map_err),
            Intrinsic::Mod => {
                let [l, r] = two_args(vals, span)?;
                l.rem_(r).map_err(map_err)
            }
            Intrinsic::Rotl => {
                let [l, r] = two_args(vals, span)?;
                l.rotl_(r).map_err(map_err)
            }
            Intrinsic::Rotr => {
                let [l, r] = two_args(vals, span)?;
                l.rotr_(r).map_err(map_err)
            }
            Intrinsic::Sizeof => match vals.len() {
                1 => {
                    let [v] = one_arg_array(vals, span)?;
                    v.sizeof_().map_err(map_err)
                }
                2 => {
                    let [v, d] = two_args(vals, span)?;
                    v.sizeof_dim_(d).map_err(map_err)
                }
                _ => Err(err(ErrorKind::NonConstantExpression, span)),
            },
        }
    }
}

// ── Tracker: per-qubit cumulative duration ──────────────────────────────

#[derive(Default)]
struct Tracker {
    times: HashMap<QubitRef, Duration>,
}

impl Tracker {
    fn get(&self, q: &QubitRef) -> Duration {
        self.times
            .get(q)
            .copied()
            .unwrap_or_else(|| Duration::new(0.0, DurationUnit::Ns))
    }

    fn sync(&mut self, qubits: &[QubitRef]) -> Duration {
        let mut max = Duration::new(0.0, DurationUnit::Ns);
        for q in qubits {
            let d = self.get(q);
            if d > max {
                max = d;
            }
        }
        for q in qubits {
            self.times.insert(q.clone(), max);
        }
        max
    }

    fn advance(&mut self, qubits: &[QubitRef], dur: Duration) {
        if qubits.is_empty() {
            return;
        }
        let start = self.sync(qubits);
        let end = start + dur;
        for q in qubits {
            self.times.insert(q.clone(), end);
        }
    }

    fn total(&self) -> Duration {
        let mut max = Duration::new(0.0, DurationUnit::Ns);
        for d in self.times.values() {
            if *d > max {
                max = *d;
            }
        }
        max
    }
}

// ── Frame: substitution for recursive gate-body evaluation ──────────────

#[derive(Default)]
struct Frame {
    /// Formal gate qubit → concrete qubit.
    qubits: HashMap<SymbolId, QubitRef>,
    /// Formal gate/subroutine param → resolved value.
    params: HashMap<SymbolId, Value>,
}

// ── Qubit resolution ────────────────────────────────────────────────────

fn resolve_qubit_operands(
    ops: &[QubitOperand],
    symbols: &SymbolTable,
    frame: &Frame,
    span: Span,
) -> Result<Vec<QubitRef>> {
    let mut out = Vec::new();
    for op in ops {
        let mut qs = resolve_qubit_operand(op, symbols, frame, span)?;
        out.append(&mut qs);
    }
    Ok(out)
}

fn resolve_qubit_operand(
    op: &QubitOperand,
    symbols: &SymbolTable,
    frame: &Frame,
    span: Span,
) -> Result<Vec<QubitRef>> {
    match op {
        QubitOperand::Hardware(n) => Ok(vec![QubitRef::Hardware(*n)]),
        QubitOperand::Indexed { symbol, indices } => {
            if let Some(subst) = frame.qubits.get(symbol) {
                if !indices.is_empty() {
                    return Err(err(
                        ErrorKind::InvalidContext(
                            "cannot index a formal qubit parameter inside `durationof`".into(),
                        ),
                        span,
                    ));
                }
                return Ok(vec![subst.clone()]);
            }
            let sym = symbols.get(*symbol);
            let name = sym.name.clone();
            if indices.is_empty() {
                match &sym.ty {
                    Type::Qubit => Ok(vec![QubitRef::Symbol { name, index: None }]),
                    Type::QubitReg(n) => {
                        let n = *n;
                        Ok((0..n)
                            .map(|i| QubitRef::Symbol {
                                name: name.clone(),
                                index: Some(i),
                            })
                            .collect())
                    }
                    _ => Err(err(
                        ErrorKind::InvalidContext(format!(
                            "`{name}` is not a qubit"
                        )),
                        span,
                    )),
                }
            } else {
                if indices.len() != 1 {
                    return Err(err(
                        ErrorKind::InvalidContext(
                            "multi-dimensional qubit indexing is not supported in `durationof`"
                                .into(),
                        ),
                        span,
                    ));
                }
                let idx = single_index(&indices[0])?;
                Ok(vec![QubitRef::Symbol {
                    name,
                    index: Some(idx),
                }])
            }
        }
    }
}

fn single_index(op: &IndexOp) -> Result<usize> {
    let items = match &op.kind {
        IndexKind::Items(items) => items,
        IndexKind::Set(_) => {
            return Err(err(
                ErrorKind::InvalidContext(
                    "set indices are not supported in `durationof` qubit operands".into(),
                ),
                op.span,
            ));
        }
    };
    if items.len() != 1 {
        return Err(err(
            ErrorKind::InvalidContext(
                "multi-dimensional qubit indexing is not supported in `durationof`".into(),
            ),
            op.span,
        ));
    }
    match &items[0] {
        IndexItem::Single(e) => match &e.kind {
            ExprKind::Literal(v) => value_as_usize(v).ok_or_else(|| {
                err(
                    ErrorKind::InvalidContext("qubit index must be a non-negative integer".into()),
                    op.span,
                )
            }),
            _ => Err(err(
                ErrorKind::InvalidContext(
                    "qubit index must be a constant integer literal in `durationof`".into(),
                ),
                op.span,
            )),
        },
        IndexItem::Range(_) => Err(err(
            ErrorKind::InvalidContext(
                "range indices are not supported in `durationof` qubit operands".into(),
            ),
            op.span,
        )),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn value_to_duration(v: &Value, span: Span) -> Result<Duration> {
    let scalar = match v {
        Value::Scalar(s) => s,
        _ => {
            return Err(err(
                ErrorKind::TypeMismatch {
                    expected: Box::new(Type::Classical(ValueTy::duration())),
                    got: Box::new(Type::Classical(v.ty())),
                },
                span,
            ));
        }
    };
    scalar
        .cast(PrimitiveTy::Duration)
        .ok()
        .and_then(|s| s.value().as_duration())
        .ok_or_else(|| {
            err(
                ErrorKind::TypeMismatch {
                    expected: Box::new(Type::Classical(ValueTy::duration())),
                    got: Box::new(Type::Classical(v.ty())),
                },
                span,
            )
        })
}

fn apply_binop(op: BinOp, lv: Value, rv: Value, span: Span) -> Result<Value> {
    let map_err = |_| err(ErrorKind::NonConstantExpression, span);
    match op {
        BinOp::Add => lv.add_(rv).map_err(map_err),
        BinOp::Sub => lv.sub_(rv).map_err(map_err),
        BinOp::Mul => lv.mul_(rv).map_err(map_err),
        BinOp::Div => lv.div_(rv).map_err(map_err),
        BinOp::Mod => lv.rem_(rv).map_err(map_err),
        BinOp::Pow => lv.pow_(rv).map_err(map_err),
        BinOp::BitAnd => lv.and_(rv).map_err(map_err),
        BinOp::BitOr => lv.or_(rv).map_err(map_err),
        BinOp::BitXor => lv.xor_(rv).map_err(map_err),
        BinOp::Shl => lv.shl_(rv).map_err(map_err),
        BinOp::Shr => lv.shr_(rv).map_err(map_err),
        BinOp::LogAnd => lv.land_(rv).map_err(map_err),
        BinOp::LogOr => lv.lor_(rv).map_err(map_err),
        BinOp::Eq => lv.eq_(rv).map_err(map_err),
        BinOp::Neq => lv.neq_(rv).map_err(map_err),
        BinOp::Lt => lv.lt_(rv).map_err(map_err),
        BinOp::Gt => lv.gt_(rv).map_err(map_err),
        BinOp::Lte => lv.lte_(rv).map_err(map_err),
        BinOp::Gte => lv.gte_(rv).map_err(map_err),
    }
}

fn apply_unop(op: UnOp, v: Value, span: Span) -> Result<Value> {
    let map_err = |_| err(ErrorKind::NonConstantExpression, span);
    match op {
        UnOp::Neg => v.neg_().map_err(map_err),
        UnOp::BitNot => v.not_().map_err(map_err),
        UnOp::LogNot => v.lnot_().map_err(map_err),
    }
}

fn one_arg(vs: Vec<Value>, span: Span) -> Result<Value> {
    let mut it = vs.into_iter();
    let v = it
        .next()
        .ok_or_else(|| err(ErrorKind::NonConstantExpression, span))?;
    if it.next().is_some() {
        return Err(err(ErrorKind::NonConstantExpression, span));
    }
    Ok(v)
}

fn one_arg_array(vs: Vec<Value>, span: Span) -> Result<[Value; 1]> {
    let v = one_arg(vs, span)?;
    Ok([v])
}

fn two_args(vs: Vec<Value>, span: Span) -> Result<[Value; 2]> {
    let mut it = vs.into_iter();
    let a = it
        .next()
        .ok_or_else(|| err(ErrorKind::NonConstantExpression, span))?;
    let b = it
        .next()
        .ok_or_else(|| err(ErrorKind::NonConstantExpression, span))?;
    if it.next().is_some() {
        return Err(err(ErrorKind::NonConstantExpression, span));
    }
    Ok([a, b])
}

fn err(kind: ErrorKind, span: Span) -> CompileError {
    CompileError::new(kind).with_span(span)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::compile_source;
    use crate::resolve::StdFileResolver;

    fn compile(source: &str) -> Program {
        compile_source(source, StdFileResolver, None).expect("compile ok")
    }

    /// A simple backend that returns a fixed duration for every gate/reset/
    /// measurement, keyed by name.
    struct FixedTimings {
        gate_ns: HashMap<String, f64>,
        measure_ns: f64,
        reset_ns: f64,
        enter: Vec<String>,
    }

    impl FixedTimings {
        fn new() -> Self {
            Self {
                gate_ns: HashMap::new(),
                measure_ns: 0.0,
                reset_ns: 0.0,
                enter: Vec::new(),
            }
        }
        fn gate(mut self, name: &str, ns: f64) -> Self {
            self.gate_ns.insert(name.into(), ns);
            self
        }
        fn measure(mut self, ns: f64) -> Self {
            self.measure_ns = ns;
            self
        }
        fn reset(mut self, ns: f64) -> Self {
            self.reset_ns = ns;
            self
        }
        fn enter(mut self, name: &str) -> Self {
            self.enter.push(name.into());
            self
        }
    }

    impl Timings for FixedTimings {
        fn measurement(&self, _args: &MeasureArgs) -> TimingDuration {
            TimingDuration::ns(self.measure_ns)
        }
        fn reset(&self, _args: &ResetArgs) -> TimingDuration {
            TimingDuration::ns(self.reset_ns)
        }
        fn gate_call(&self, args: &GateCallArgs) -> GateCallTiming {
            if self.enter.iter().any(|n| n == &args.name) {
                return GateCallTiming::Enter;
            }
            let ns = self.gate_ns.get(&args.name).copied().unwrap_or(0.0);
            GateCallTiming::Duration(TimingDuration::ns(ns))
        }
    }

    fn get_duration_literal(expr: &Expr) -> Duration {
        match &expr.kind {
            ExprKind::Literal(Value::Scalar(s)) => s.value().as_duration().unwrap(),
            _ => panic!("expected literal duration, got {:?}", expr.ty),
        }
    }

    #[test]
    fn single_gate_on_hardware_qubit() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({x $0;});
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new().gate("x", 50.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let dur = first_init_duration(&p);
        assert_eq!(dur.value, 50.0);
    }

    #[test]
    fn serial_gates_stack_on_same_qubit() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                x $0;
                y $0;
                z $0;
            });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new()
            .gate("x", 10.0)
            .gate("y", 20.0)
            .gate("z", 30.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let d = first_init_duration(&p);
        assert_eq!(d.value, 60.0);
    }

    #[test]
    fn parallel_gates_take_max() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                x $0;
                y $1;
            });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new().gate("x", 10.0).gate("y", 30.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let d = first_init_duration(&p);
        assert_eq!(d.value, 30.0);
    }

    #[test]
    fn multi_qubit_gate_syncs_then_advances() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                x $0;       // $0 at 10
                y $1;       // $1 at 20
                cx $0, $1;  // sync at 20, then +30 → both at 50
            });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new()
            .gate("x", 10.0)
            .gate("y", 20.0)
            .gate("cx", 30.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let d = first_init_duration(&p);
        assert_eq!(d.value, 50.0);
    }

    #[test]
    fn dt_is_resolved_via_options() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({x $0;});
        "#;
        let mut p = compile(src);
        struct DtTimings;
        impl Timings for DtTimings {
            fn measurement(&self, _a: &MeasureArgs) -> TimingDuration {
                TimingDuration::zero()
            }
            fn reset(&self, _a: &ResetArgs) -> TimingDuration {
                TimingDuration::zero()
            }
            fn gate_call(&self, _a: &GateCallArgs) -> GateCallTiming {
                GateCallTiming::Duration(TimingDuration::dt(4.0))
            }
        }
        let options = CompileOptions {
            dt: Duration::new(0.5, DurationUnit::Ns),
            ..Default::default()
        };
        resolve_durationof(&mut p, &DtTimings, &options).unwrap();
        let d = first_init_duration(&p);
        // 4 dt × 0.5 ns/dt = 2 ns
        assert_eq!(d.to_unit(DurationUnit::Ns).value, 2.0);
    }

    #[test]
    fn enter_recurses_into_gate_body() {
        let src = r#"
            include "stdgates.inc";
            gate my_pair a { x a; y a; }
            duration d = durationof({ my_pair $0; });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new()
            .gate("x", 10.0)
            .gate("y", 20.0)
            .enter("my_pair");
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let d = first_init_duration(&p);
        assert_eq!(d.value, 30.0);
    }

    #[test]
    fn delay_contributes_to_duration() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                x $0;
                delay[100ns] $0;
            });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new().gate("x", 10.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let d = first_init_duration(&p);
        assert_eq!(d.to_unit(DurationUnit::Ns).value, 110.0);
    }

    #[test]
    fn measure_and_reset_durations() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                reset $0;
                h $0;
                bit c = measure $0;
            });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new().gate("h", 20.0).measure(200.0).reset(50.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        let d = first_init_duration(&p);
        assert_eq!(d.value, 270.0);
    }

    #[test]
    fn unsupported_statement_errors() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                int x = 0;
                h $0;
            });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new().gate("h", 10.0);
        let err = resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
    }

    #[test]
    fn const_decl_value_is_derivable() {
        // The value should be available in the compiled const, but const_value
        // is set at lower time before durationof resolution. We verify that
        // the init expression's literal duration matches expectations — users
        // who want to propagate this into `const_value` can run a subsequent
        // fold pass.
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({x $0;}) * 2;
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new().gate("x", 15.0);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        // The init expression for `d` is `Binary { Mul, DurationOf → 15ns, 2 }`.
        // After resolution the DurationOf child is a literal; the outer * is
        // not re-folded by this pass.
        let decl = p
            .body
            .iter()
            .find(|s| matches!(&s.kind, StmtKind::ClassicalDecl { init: Some(_), .. }))
            .unwrap();
        if let StmtKind::ClassicalDecl {
            init: Some(crate::sir::DeclInit::Expr(e)),
            ..
        } = &decl.kind
        {
            if let ExprKind::Binary { left, .. } = &e.kind {
                let inner = get_duration_literal(left);
                assert_eq!(inner.value, 15.0);
            } else {
                panic!("expected binary expression after resolution");
            }
        } else {
            panic!("expected classical decl with init");
        }
    }

    fn first_init_duration(p: &Program) -> Duration {
        let decl = p
            .body
            .iter()
            .find(|s| matches!(&s.kind, StmtKind::ClassicalDecl { init: Some(_), .. }))
            .expect("decl with init");
        match &decl.kind {
            StmtKind::ClassicalDecl {
                init: Some(crate::sir::DeclInit::Expr(e)),
                ..
            } => get_duration_literal(e),
            _ => panic!("unexpected kind"),
        }
    }
}
