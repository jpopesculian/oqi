//! `durationof` intrinsic resolution.
//!
//! The [`Timings`] trait is implemented by a backend to supply durations for
//! measurements, resets and gate calls. [`resolve_durationof`] walks a compiled
//! [`crate::sir::Program`] and replaces every `ExprKind::DurationOf` with a
//! constant [`crate::classical::Duration`] literal, stacking per-qubit
//! durations and taking the maximum across qubits at sync points.
//!
//! This is the compile-time timing path the spec mandates (docs/delays.rst:
//! "all operations on durations happen at compile time"). It is strict:
//! control flow and classical statements inside a `durationof` scope are
//! errors, because they cannot be timed statically. When no [`Timings`] is
//! supplied (the pass is simply not run), `durationof` expressions survive to
//! the bytecode and the VM's runtime timing pass evaluates them instead —
//! that path additionally supports control flow, but knows no per-gate
//! durations (uncalibrated gates are zero-width). On that runtime path an
//! unresolved `stretch` reads as its default-initialized value (0 ns) via
//! register zero-seeding, so stretchy delays execute zero-width — the
//! minimal solution for an unconstrained stretch. This pass instead errors
//! on `stretch` ("resolution is not implemented") until a real constraint
//! solver exists.

use std::collections::{HashMap, HashSet};

use oqi_lex::Span;

use crate::classical::{
    Duration, DurationUnit, Primitive, PrimitiveTy, Value, ValueTy, value_as_usize,
};
use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{
    BinOp, CalibrationArg, CalibrationBody, CalibrationDecl, CalibrationOperand, CalibrationTarget,
    CallTarget, Expr, ExprKind, GateDecl, GateModifier, IndexItem, IndexKind, IndexOp, Intrinsic,
    LValue, MeasureExpr, MeasureExprKind, Program, QubitOperand, RValue, Stmt, StmtKind,
    SwitchLabels, UnOp,
};
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::{CompileOptions, Type};

mod stretch;

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
///
/// The methods are fallible so a provider that *derives* durations (e.g.
/// [`TableTimings`] walking a defcal body) can report why a duration
/// cannot be determined statically. Spanless errors get the call site's
/// span attached by the pass.
pub trait Timings {
    fn measurement(&self, args: &MeasureArgs) -> Result<TimingDuration>;
    fn reset(&self, args: &ResetArgs) -> Result<TimingDuration>;
    fn gate_call(&self, args: &GateCallArgs) -> Result<GateCallTiming>;
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

    // Pass 4: stretch constraint resolution (only when stretch symbols
    // exist; a stretch-free program is untouched).
    if program
        .symbols
        .iter()
        .any(|s| matches!(s.ty, Type::Stretch))
    {
        stretch::resolve_stretch(program, timings, options)?;

        // Pass 5: durationof scopes retained for stretch resolution are
        // now concrete; fold them with the Pass-1 machinery (idempotent
        // for everything already rewritten).
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

    // LValue index expressions (assignment/measure targets) are not visited:
    // a `durationof` inside an assignment-target index is pathological and
    // out of scope for this pass.
    fn visit_stmt(&mut self, stmt: &mut Stmt) -> Result<()> {
        match &mut stmt.kind {
            StmtKind::Alias(a) => {
                for e in &mut a.value {
                    self.visit_expr(e)?;
                }
            }
            StmtKind::GateCall(gc) => {
                for m in &mut gc.modifiers {
                    if let GateModifier::Pow(e) = m {
                        self.visit_expr(e)?;
                    }
                }
                for a in &mut gc.args {
                    self.visit_expr(a)?;
                }
                for q in &mut gc.qubits {
                    self.visit_qubit_operand(q)?;
                }
                if let Some(d) = &mut gc.duration {
                    self.visit_expr(d)?;
                }
            }
            StmtKind::Measure(m) => self.visit_measure_expr(&mut m.measure)?,
            StmtKind::Reset(operand) => self.visit_qubit_operand(operand)?,
            StmtKind::Barrier(operands) | StmtKind::Nop(operands) => {
                for o in operands {
                    self.visit_qubit_operand(o)?;
                }
            }
            StmtKind::Delay(d) => {
                self.visit_expr(&mut d.duration)?;
                for o in &mut d.operands {
                    self.visit_qubit_operand(o)?;
                }
            }
            StmtKind::Box(b) => {
                if let Some(d) = &mut b.duration {
                    self.visit_expr(d)?;
                }
                self.visit_stmts(&mut b.body)?;
            }
            StmtKind::Assignment(a) => self.visit_rvalue(&mut a.value)?,
            StmtKind::If(i) => {
                self.visit_expr(&mut i.condition)?;
                self.visit_stmts(&mut i.then_body)?;
                if let Some(eb) = &mut i.else_body {
                    self.visit_stmts(eb)?;
                }
            }
            StmtKind::For(f) => {
                self.visit_for_iterable(&mut f.iterable)?;
                self.visit_stmts(&mut f.body)?;
            }
            StmtKind::While(w) => {
                self.visit_expr(&mut w.condition)?;
                self.visit_stmts(&mut w.body)?;
            }
            StmtKind::Switch(s) => {
                self.visit_expr(&mut s.target)?;
                for c in &mut s.cases {
                    if let SwitchLabels::Values(v) = &mut c.labels {
                        for e in v {
                            self.visit_expr(e)?;
                        }
                    }
                    self.visit_stmts(&mut c.body)?;
                }
            }
            StmtKind::Return(Some(rv)) => self.visit_rvalue(rv)?,
            StmtKind::ExprStmt(e) => self.visit_expr(e)?,
            StmtKind::Return(None)
            | StmtKind::Break
            | StmtKind::Continue
            | StmtKind::End
            | StmtKind::Pragma(_)
            | StmtKind::Cal(_) => {}
        }
        Ok(())
    }

    fn visit_rvalue(&mut self, value: &mut RValue<Expr>) -> Result<()> {
        match value {
            RValue::Expr(e) => self.visit_expr(e),
            RValue::Measure(m) => self.visit_measure_expr(m),
        }
    }

    fn visit_measure_expr(&mut self, m: &mut MeasureExpr<Expr>) -> Result<()> {
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

    fn visit_qubit_operand(&mut self, op: &mut QubitOperand<Expr>) -> Result<()> {
        if let QubitOperand::Indexed { indices, .. } = op {
            for idx in indices {
                self.visit_index_op(idx)?;
            }
        }
        Ok(())
    }

    fn visit_index_op(&mut self, op: &mut IndexOp<Expr>) -> Result<()> {
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
            // A scope referencing stretch is retained: Pass 4 measures it
            // affinely and rewrites its stretch variables, then Pass 5
            // re-runs this rewrite on the by-then concrete scope.
            if stretch::stmts_have_stretch_var(stmts, self.symbols) {
                return Ok(());
            }
            let duration = self.compute_scope_duration(stmts, expr.span)?;
            expr.kind = ExprKind::Literal(Primitive::from(duration));
            expr.ty = Type::Classical(ValueTy::duration());
            return Ok(());
        }
        match &mut expr.kind {
            ExprKind::Binary(b) => {
                self.visit_expr(&mut b.left)?;
                self.visit_expr(&mut b.right)?;
            }
            ExprKind::Unary(u) => self.visit_expr(&mut u.operand)?,
            ExprKind::Cast(c) => self.visit_expr(&mut c.operand)?,
            ExprKind::Index(ix) => {
                self.visit_expr(&mut ix.base)?;
                self.visit_index_op(&mut ix.index)?;
            }
            ExprKind::Call(c) => {
                for a in &mut c.args {
                    self.visit_expr(a)?;
                }
            }
            ExprKind::ArrayLiteral(al) => {
                for e in &mut al.items {
                    self.visit_expr(e)?;
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
            StmtKind::GateCall(gc) => self.process_gate_call(
                gc.gate,
                &gc.modifiers,
                &gc.args,
                &gc.qubits,
                gc.duration.as_ref(),
                tracker,
                frame,
                span,
            ),
            // Covers both a bare `measure q;` and `bit c = measure q;` (a
            // decl-with-measure-init lowers to `Measure { target: Some(..) }`).
            StmtKind::Measure(m) => self.process_measure(&m.measure, tracker, frame),
            StmtKind::Reset(operand) => {
                let (qr, dur) = self.resolve_reset_duration(operand, frame, span)?;
                tracker.advance(&qr, dur);
                Ok(())
            }
            StmtKind::Delay(d) => {
                let dur_val = self.eval_const_expr(&d.duration, frame)?;
                let dur = value_to_duration(&dur_val, d.duration.span)?;
                let qubits = resolve_qubit_operands(&d.operands, self.symbols, frame, span)?;
                if qubits.is_empty() {
                    return Err(err(
                        ErrorKind::InvalidContext(
                            "delay requires at least one qubit operand".into(),
                        ),
                        span,
                    ));
                }
                tracker.advance(&qubits, dur);
                Ok(())
            }
            StmtKind::Barrier(operands) | StmtKind::Nop(operands) => {
                let qubits = resolve_qubit_operands(operands, self.symbols, frame, span)?;
                if !qubits.is_empty() {
                    tracker.sync(&qubits);
                }
                Ok(())
            }
            StmtKind::Box(b) => {
                if let Some(dur_expr) = &b.duration {
                    let v = self.eval_const_expr(dur_expr, frame)?;
                    let dur = value_to_duration(&v, dur_expr.span)?;
                    // A box with an explicit duration pins the enclosed scope
                    // to that duration across its qubits.
                    let inner = self.compute_scope_duration_with(&b.body, frame, span)?;
                    let (qubits, _) = inner;
                    if !qubits.is_empty() {
                        tracker.advance(&qubits, dur);
                    }
                } else {
                    let (qubits, dur) = self.compute_scope_duration_with(&b.body, frame, span)?;
                    if !qubits.is_empty() {
                        tracker.advance(&qubits, dur);
                    }
                }
                Ok(())
            }
            _ => Err(err(
                ErrorKind::InvalidContext(
                    "statement not supported inside `durationof` when compile-time timings \
                     are supplied (control flow and classical statements cannot be timed \
                     statically)"
                        .into(),
                ),
                if span == Span::default() {
                    outer_span
                } else {
                    span
                },
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

    #[allow(clippy::too_many_arguments)]
    fn process_gate_call(
        &self,
        gate: SymbolId,
        modifiers: &[GateModifier<Expr>],
        args: &[Expr],
        qubits: &[QubitOperand<Expr>],
        designator: Option<&Expr>,
        tracker: &mut Tracker,
        frame: &Frame,
        span: Span,
    ) -> Result<()> {
        let (qubit_refs, dur) = self
            .resolve_gate_call_duration(gate, modifiers, args, qubits, designator, frame, span)?;
        tracker.advance(&qubit_refs, dur);
        Ok(())
    }

    /// A gate call's operand wires and resolved duration, without touching
    /// any clock.
    #[allow(clippy::too_many_arguments)]
    fn resolve_gate_call_duration(
        &self,
        gate: SymbolId,
        modifiers: &[GateModifier<Expr>],
        args: &[Expr],
        qubits: &[QubitOperand<Expr>],
        designator: Option<&Expr>,
        frame: &Frame,
        span: Span,
    ) -> Result<(Vec<QubitRef>, Duration)> {
        // A `gate[duration]` designator IS the call's duration (spec
        // delays.rst:212-222): it bypasses the timing table, defcals and
        // gate-body recursion, and the classical args need no resolution.
        if let Some(d) = designator {
            let qubit_refs = resolve_qubit_operands(qubits, self.symbols, frame, span)?;
            let v = self.eval_const_expr(d, frame)?;
            let dur = value_to_duration(&v, d.span)?;
            if dur.to_unit(DurationUnit::Ns).value < 0.0 {
                return Err(err(
                    ErrorKind::InvalidContext(format!(
                        "gate duration designator is negative ({dur})"
                    )),
                    d.span,
                ));
            }
            return Ok((qubit_refs, dur));
        }
        // `gphase` arrives as an ordinary seeded symbol named "gphase".
        let name = self.symbols.get(gate).name.clone();
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
        match self
            .timings
            .gate_call(&call_args)
            .map_err(|e| at_span(e, span))?
        {
            GateCallTiming::Duration(d) => Ok((qubit_refs, d.resolve(&self.options.dt))),
            GateCallTiming::Enter => {
                // Built-ins (`U`, `gphase`) have no GateDecl and fail the
                // lookup, producing the not-found error below.
                let decl = self
                    .gates
                    .iter()
                    .find(|g| g.symbol == gate)
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
                Ok((qubit_refs, inner_dur))
            }
        }
    }

    fn process_measure(
        &self,
        measure: &MeasureExpr<Expr>,
        tracker: &mut Tracker,
        frame: &Frame,
    ) -> Result<()> {
        let (qr, dur) = self.resolve_measure_duration(measure, frame)?;
        tracker.advance(&qr, dur);
        Ok(())
    }

    /// A measure expression's operand wires and resolved duration, without
    /// touching any clock.
    fn resolve_measure_duration(
        &self,
        measure: &MeasureExpr<Expr>,
        frame: &Frame,
    ) -> Result<(Vec<QubitRef>, Duration)> {
        match &measure.kind {
            MeasureExprKind::Measure { operand } => {
                let qr = resolve_qubit_operand(operand, self.symbols, frame, measure.span)?;
                let args = MeasureArgs { qubits: qr.clone() };
                let dur = self
                    .timings
                    .measurement(&args)
                    .map_err(|e| at_span(e, measure.span))?
                    .resolve(&self.options.dt);
                Ok((qr, dur))
            }
            MeasureExprKind::QuantumCall {
                callee,
                args,
                qubits,
            } => {
                // A quantum call used as a measure expression — treat as a
                // gate call, but the Timings callback decides the semantics.
                self.resolve_gate_call_duration(
                    *callee,
                    &[],
                    args,
                    qubits,
                    None,
                    frame,
                    measure.span,
                )
            }
        }
    }

    /// A reset's operand wires and resolved duration, without touching any
    /// clock.
    fn resolve_reset_duration(
        &self,
        operand: &QubitOperand<Expr>,
        frame: &Frame,
        span: Span,
    ) -> Result<(Vec<QubitRef>, Duration)> {
        let qr = resolve_qubit_operand(operand, self.symbols, frame, span)?;
        let args = ResetArgs { qubits: qr.clone() };
        let dur = self
            .timings
            .reset(&args)
            .map_err(|e| at_span(e, span))?
            .resolve(&self.options.dt);
        Ok((qr, dur))
    }

    fn resolve_modifier(
        &self,
        m: &GateModifier<Expr>,
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
            ExprKind::Literal(p) => Ok(Value::from(p.clone())),
            ExprKind::Var(sid) => {
                if let Some(v) = frame.params.get(sid) {
                    return Ok(v.clone());
                }
                let sym = self.symbols.get(*sid);
                if matches!(sym.ty, Type::Stretch) {
                    return Err(stretch_err(expr.span));
                }
                sym.const_value
                    .clone()
                    .ok_or_else(|| err(ErrorKind::NonConstantExpression, expr.span))
            }
            ExprKind::Binary(b) => {
                let lv = self.eval_const_expr(&b.left, frame)?;
                let rv = self.eval_const_expr(&b.right, frame)?;
                apply_binop(b.op, lv, rv, expr.span)
            }
            ExprKind::Unary(u) => {
                let v = self.eval_const_expr(&u.operand, frame)?;
                apply_unop(u.op, v, expr.span)
            }
            ExprKind::Cast(c) => {
                let v = self.eval_const_expr(&c.operand, frame)?;
                let ty = c
                    .target_ty
                    .value_ty()
                    .ok_or_else(|| err(ErrorKind::NonConstantExpression, expr.span))?;
                v.cast(ty)
                    .map_err(|_| err(ErrorKind::NonConstantExpression, expr.span))
            }
            ExprKind::Call(c) => match &c.callee {
                CallTarget::Intrinsic(i) => self.eval_intrinsic(i, &c.args, frame, expr.span),
                CallTarget::Symbol(_) => Err(err(ErrorKind::NonConstantExpression, expr.span)),
            },
            ExprKind::DurationOf(stmts) => {
                let d = self.compute_scope_duration(stmts, expr.span)?;
                Ok(Value::from(d))
            }
            ExprKind::Index(_) | ExprKind::HardwareQubit(_) | ExprKind::ArrayLiteral(_) => {
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

#[derive(Default, Clone)]
struct Frame {
    /// Formal gate qubit → concrete qubit.
    qubits: HashMap<SymbolId, QubitRef>,
    /// Formal gate/subroutine param → resolved value.
    params: HashMap<SymbolId, Value>,
}

// ── Qubit resolution ────────────────────────────────────────────────────

fn resolve_qubit_operands(
    ops: &[QubitOperand<Expr>],
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
    op: &QubitOperand<Expr>,
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
                        ErrorKind::InvalidContext(format!("`{name}` is not a qubit")),
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
                let idx = single_index(&indices[0], symbols, frame)?;
                Ok(vec![QubitRef::Symbol {
                    name,
                    index: Some(idx),
                }])
            }
        }
    }
}

fn single_index(op: &IndexOp<Expr>, symbols: &SymbolTable, frame: &Frame) -> Result<usize> {
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
        IndexItem::Single(e) => {
            let v = match &e.kind {
                ExprKind::Literal(p) => Some(Value::from(p.clone())),
                // Loop variables (via the frame) and consts index too.
                ExprKind::Var(sid) => frame
                    .params
                    .get(sid)
                    .cloned()
                    .or_else(|| symbols.get(*sid).const_value.clone()),
                _ => None,
            };
            v.as_ref().and_then(value_as_usize).ok_or_else(|| {
                err(
                    ErrorKind::InvalidContext(
                        "qubit index must be a compile-time non-negative integer".into(),
                    ),
                    op.span,
                )
            })
        }
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
    let mismatch = || {
        err(
            ErrorKind::TypeMismatch {
                expected: Box::new(Type::Classical(ValueTy::duration())),
                got: Box::new(Type::Classical(v.ty())),
            },
            span,
        )
    };
    let scalar = match v {
        Value::Scalar(s) => s,
        _ => return Err(mismatch()),
    };
    scalar
        .clone()
        .cast(PrimitiveTy::Duration)
        .ok()
        .and_then(|s| s.value().as_duration())
        .ok_or_else(mismatch)
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

/// Spec-wise `stretch` resolves at compile time too, but no solver exists
/// yet — give it an honest error rather than a generic non-constant one
/// (clean seam for the future pass).
fn stretch_err(span: Span) -> CompileError {
    err(
        ErrorKind::InvalidContext(
            "stretch resolution is not implemented; `stretch` values cannot \
             be used where timings are resolved at compile time"
                .into(),
        ),
        span,
    )
}

/// Attach `span` to `e` unless it already carries one (provider errors may
/// point inside a defcal body; keep the more precise span).
fn at_span(e: CompileError, span: Span) -> CompileError {
    if e.span == Span::default() {
        e.with_span(span)
    } else {
        e
    }
}

// ── TableTimings: a name → duration table ───────────────────────────────

/// A [`Timings`] impl backed by a name → duration table.
///
/// Lookup policy for `gate_call`: a named entry always wins; otherwise a
/// matching defcal (via [`TableTimings::with_defcals`]) has its pulse body's
/// duration derived; otherwise a gate registered as enterable (via
/// [`TableTimings::with_program_gates`]) is entered and its duration derived
/// from its body; otherwise (built-ins like `U`/`gphase`, or unregistered
/// gates) the call costs 0 ns — matching the runtime convention that
/// uncalibrated gates take zero time. `measure` and `reset` default to a
/// matching measure/reset defcal, then 0 ns.
#[derive(Default, Clone)]
pub struct TableTimings {
    gates: HashMap<String, TimingDuration>,
    measure: Option<TimingDuration>,
    reset: Option<TimingDuration>,
    enterable: HashSet<String>,
    defcals: Option<DefcalTable>,
}

// Manual: `DefcalTable` holds SIR nodes, which have no `Debug`.
impl std::fmt::Debug for TableTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableTimings")
            .field("gates", &self.gates)
            .field("measure", &self.measure)
            .field("reset", &self.reset)
            .field("enterable", &self.enterable)
            .field("defcals", &self.defcals.is_some())
            .finish()
    }
}

impl TableTimings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the duration of the gate `name`.
    pub fn gate(mut self, name: impl Into<String>, d: TimingDuration) -> Self {
        self.gates.insert(name.into(), d);
        self
    }

    /// Set the duration of measurements.
    pub fn measure(mut self, d: TimingDuration) -> Self {
        self.measure = Some(d);
        self
    }

    /// Set the duration of resets.
    pub fn reset(mut self, d: TimingDuration) -> Self {
        self.reset = Some(d);
        self
    }

    /// Register every gate declared with a body in `program` as enterable:
    /// a call to one (absent a named entry) recurses into its body and
    /// derives the duration from the gates it decomposes into.
    pub fn with_program_gates(mut self, program: &Program) -> Self {
        for g in &program.gates {
            self.enterable
                .insert(program.symbols.get(g.symbol).name.clone());
        }
        self
    }

    /// Derive durations from the program's `defcal` bodies: a gate call
    /// (or measure/reset) matching a calibration — most exact
    /// hardware-operand match wins, mirroring the VM's dispatch — takes
    /// the busiest-frame duration of its pulse body. Explicit table
    /// entries still win. `capture` counts one `dt` per sample; a matched
    /// defcal whose durations aren't compile-time constants is an error,
    /// not silently zero. The calibrations are cloned out of `program`
    /// (the pass later needs `&mut Program`).
    pub fn with_defcals(mut self, program: &Program, dt: &Duration) -> Self {
        let mut table = DefcalTable {
            gates: HashMap::new(),
            measures: Vec::new(),
            resets: Vec::new(),
            names: HashMap::new(),
            consts: HashMap::new(),
            stretches: HashSet::new(),
            dt: *dt,
        };
        for s in program.symbols.iter() {
            table.names.insert(s.id, s.name.clone());
            if let Some(v) = &s.const_value {
                table.consts.insert(s.id, v.clone());
            }
            if matches!(s.ty, Type::Stretch) {
                table.stretches.insert(s.id);
            }
        }
        for cal in &program.calibrations {
            match &cal.target {
                CalibrationTarget::Named(sid) => table
                    .gates
                    .entry(program.symbols.get(*sid).name.clone())
                    .or_default()
                    .push(cal.clone()),
                CalibrationTarget::Measure => table.measures.push(cal.clone()),
                CalibrationTarget::Reset => table.resets.push(cal.clone()),
                CalibrationTarget::Delay => {}
            }
        }
        self.defcals = Some(table);
        self
    }

    /// Build from `(name, duration-literal)` string pairs — the shared
    /// driver entry point for `--timing NAME=DURATION`-style options. The
    /// names `measure` and `reset` set those operations' durations; any
    /// other name is a gate. Values use OpenQASM timing-literal syntax
    /// (`"50ns"`, `"4dt"`); `dt` values are resolved against `dt`. Unknown
    /// gate names are permitted (a timing table is a device property, not
    /// program-specific); negative durations are rejected.
    pub fn from_str_entries<'a>(
        entries: impl IntoIterator<Item = (&'a str, &'a str)>,
        dt: &Duration,
    ) -> Result<Self> {
        let mut table = TableTimings::new();
        for (name, raw) in entries {
            let d = crate::types::parse_timing_literal(raw, dt).map_err(|_| {
                CompileError::new(ErrorKind::InvalidLiteral(format!(
                    "timing for `{name}`: `{raw}` is not a duration literal"
                )))
            })?;
            if d.value.is_nan() || d.value < 0.0 {
                return Err(CompileError::new(ErrorKind::InvalidLiteral(format!(
                    "timing for `{name}`: duration must be non-negative"
                ))));
            }
            let td = TimingDuration::Si(d);
            match name {
                "measure" => table.measure = Some(td),
                "reset" => table.reset = Some(td),
                _ => {
                    table.gates.insert(name.to_string(), td);
                }
            }
        }
        Ok(table)
    }
}

impl Timings for TableTimings {
    fn measurement(&self, args: &MeasureArgs) -> Result<TimingDuration> {
        if let Some(d) = self.measure {
            return Ok(d);
        }
        if let Some(dc) = &self.defcals
            && let Some(cal) = DefcalTable::find(&dc.measures, &args.qubits, 0)
        {
            return dc.body_duration(cal, &[]).map(TimingDuration::from);
        }
        Ok(TimingDuration::zero())
    }
    fn reset(&self, args: &ResetArgs) -> Result<TimingDuration> {
        if let Some(d) = self.reset {
            return Ok(d);
        }
        if let Some(dc) = &self.defcals
            && let Some(cal) = DefcalTable::find(&dc.resets, &args.qubits, 0)
        {
            return dc.body_duration(cal, &[]).map(TimingDuration::from);
        }
        Ok(TimingDuration::zero())
    }
    fn gate_call(&self, args: &GateCallArgs) -> Result<GateCallTiming> {
        if let Some(d) = self.gates.get(&args.name) {
            return Ok(GateCallTiming::Duration(*d));
        }
        if let Some(dc) = &self.defcals
            && let Some(cals) = dc.gates.get(&args.name)
            && let Some(cal) = DefcalTable::find(cals, &args.qubits, args.args.len())
        {
            return dc
                .body_duration(cal, &args.args)
                .map(|d| GateCallTiming::Duration(d.into()));
        }
        if self.enterable.contains(&args.name) {
            return Ok(GateCallTiming::Enter);
        }
        Ok(GateCallTiming::Duration(TimingDuration::zero()))
    }
}

// ── Defcal-derived timings ──────────────────────────────────────────────

/// Calibration data cloned out of a [`Program`] so [`TableTimings`] can
/// derive durations from defcal pulse bodies.
#[derive(Clone)]
struct DefcalTable {
    /// Gate defcals grouped by gate name, in declaration order.
    gates: HashMap<String, Vec<CalibrationDecl>>,
    measures: Vec<CalibrationDecl>,
    resets: Vec<CalibrationDecl>,
    /// Symbol names, for resolving intrinsic callees and frame variables.
    names: HashMap<SymbolId, String>,
    /// Compile-time constant values of symbols.
    consts: HashMap<SymbolId, Value>,
    /// Symbols declared `stretch`, for the honest not-implemented error.
    stretches: HashSet<SymbolId>,
    /// The `dt` duration captured at build time (one `capture` sample
    /// lasts one `dt`).
    dt: Duration,
}

impl DefcalTable {
    /// Pick the best-matching calibration for a call on `qubits` with
    /// `n_args` classical arguments. Mirrors the VM's dispatch rule: arity
    /// must match, `Hardware` operands must equal (`Ident` matches
    /// anything), most exact matches win, ties go to the first declared.
    fn find<'c>(
        cals: &'c [CalibrationDecl],
        qubits: &[QubitRef],
        n_args: usize,
    ) -> Option<&'c CalibrationDecl> {
        let mut best: Option<(&CalibrationDecl, usize)> = None;
        for cal in cals {
            if cal.operands.len() != qubits.len() || cal.args.len() != n_args {
                continue;
            }
            let mut exact = 0usize;
            let ok = cal.operands.iter().zip(qubits).all(|(op, q)| match op {
                CalibrationOperand::Hardware(n) => {
                    exact += 1;
                    matches!(q, QubitRef::Hardware(m) if m == n)
                }
                CalibrationOperand::Ident(_) => true,
            });
            if ok && best.is_none_or(|(_, e)| exact > e) {
                best = Some((cal, exact));
            }
        }
        best.map(|(c, _)| c)
    }

    /// Duration of a defcal's pulse body: per-frame clocks advanced by
    /// plays, captures and delays; the result is the busiest frame
    /// (mirroring the VM's runtime `TimingState`).
    fn body_duration(&self, cal: &CalibrationDecl, call_args: &[Value]) -> Result<Duration> {
        let CalibrationBody::OpenPulse(stmts) = &cal.body else {
            return Err(err(
                ErrorKind::InvalidContext("opaque defcal bodies cannot be timed".into()),
                cal.span,
            ));
        };
        let mut params = HashMap::new();
        for (arg, v) in cal.args.iter().zip(call_args) {
            if let CalibrationArg::Param(sid) = arg {
                params.insert(*sid, v.clone());
            }
        }
        let mut walk = CalWalk {
            table: self,
            params,
            frames: HashMap::new(),
            waveforms: HashMap::new(),
        };
        for stmt in stmts {
            walk.stmt(stmt)?;
        }
        Ok(walk.total())
    }
}

/// One walk over a defcal body, accumulating per-frame clocks.
struct CalWalk<'a> {
    table: &'a DefcalTable,
    /// Defcal params bound to the call's actual argument values.
    params: HashMap<SymbolId, Value>,
    /// Frame variable → its clock.
    frames: HashMap<SymbolId, Duration>,
    /// Local waveform variable → its duration.
    waveforms: HashMap<SymbolId, Duration>,
}

impl CalWalk<'_> {
    fn total(&self) -> Duration {
        let mut max = Duration::new(0.0, DurationUnit::Ns);
        for d in self.frames.values() {
            if *d > max {
                max = *d;
            }
        }
        max
    }

    fn advance(&mut self, frame: SymbolId, dur: Duration) {
        let clock = self
            .frames
            .entry(frame)
            .or_insert(Duration::new(0.0, DurationUnit::Ns));
        *clock = *clock + dur;
    }

    fn name(&self, sid: SymbolId) -> &str {
        self.table.names.get(&sid).map(String::as_str).unwrap_or("")
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<()> {
        let span = stmt.span;
        match &stmt.kind {
            StmtKind::Assignment(a) => {
                // `waveform wf = gaussian(...)` binds a local waveform.
                if let RValue::Expr(e) = &a.value
                    && let ExprKind::Call(c) = &e.kind
                    && let CallTarget::Symbol(callee) = &c.callee
                    && self.name(*callee) == "gaussian"
                {
                    let dur = self.waveform_duration(e)?;
                    if let LValue::Var(sid) = &a.target {
                        self.waveforms.insert(*sid, dur);
                    }
                    return Ok(());
                }
                // Captures nested in the assigned expression still advance
                // their frame; other classical work is zero-width.
                if let RValue::Expr(e) = &a.value {
                    self.scan_expr(e)?;
                }
                Ok(())
            }
            StmtKind::ExprStmt(e) => self.scan_expr(e),
            StmtKind::Return(Some(RValue::Expr(e))) => self.scan_expr(e),
            StmtKind::Return(_) => Ok(()),
            // A defcal body may call other gates (recursive defcal
            // dispatch); their pulses land on this walk's frame clocks.
            StmtKind::GateCall(gc) => self.gate_call(gc, span),
            StmtKind::Delay(d) => {
                let v = eval_cal_expr(&d.duration, &self.params, self.table)?;
                let dur = value_to_duration(&v, d.duration.span)?;
                let mut targets: Vec<SymbolId> = Vec::new();
                for op in &d.operands {
                    match op {
                        QubitOperand::Indexed { symbol, indices } if indices.is_empty() => {
                            targets.push(*symbol);
                        }
                        _ => {
                            return Err(err(
                                ErrorKind::InvalidContext(
                                    "delay operands in a timed defcal body must be frames".into(),
                                ),
                                span,
                            ));
                        }
                    }
                }
                if targets.is_empty() {
                    targets = self.frames.keys().copied().collect();
                }
                for f in targets {
                    self.advance(f, dur);
                }
                Ok(())
            }
            StmtKind::Barrier(ops) => {
                let mut targets: Vec<SymbolId> = ops
                    .iter()
                    .filter_map(|op| match op {
                        QubitOperand::Indexed { symbol, indices } if indices.is_empty() => {
                            Some(*symbol)
                        }
                        _ => None,
                    })
                    .collect();
                if targets.is_empty() {
                    targets = self.frames.keys().copied().collect();
                }
                let mut max = Duration::new(0.0, DurationUnit::Ns);
                for f in &targets {
                    let d = self
                        .frames
                        .get(f)
                        .copied()
                        .unwrap_or(Duration::new(0.0, DurationUnit::Ns));
                    if d > max {
                        max = d;
                    }
                }
                for f in targets {
                    self.frames.insert(f, max);
                }
                Ok(())
            }
            StmtKind::Nop(_) => Ok(()),
            _ => Err(err(
                ErrorKind::InvalidContext(
                    "statement not supported in a defcal body timed at compile time".into(),
                ),
                span,
            )),
        }
    }

    /// Advance clocks for every pulse operation found in `e`; everything
    /// classical is zero-width.
    fn scan_expr(&mut self, e: &Expr) -> Result<()> {
        if let ExprKind::Call(c) = &e.kind {
            if let CallTarget::Symbol(callee) = &c.callee {
                match self.name(*callee) {
                    "play" => return self.play(&c.args, e.span),
                    "capture" => return self.capture(&c.args, e.span),
                    // A waveform constructed but not played is zero-width.
                    "gaussian" => return Ok(()),
                    _ => {}
                }
            }
            // shift_phase / newframe / threshold / ordinary calls: captures
            // may hide in their arguments.
            for a in &c.args {
                self.scan_expr(a)?;
            }
            return Ok(());
        }
        match &e.kind {
            ExprKind::Binary(b) => {
                self.scan_expr(&b.left)?;
                self.scan_expr(&b.right)
            }
            ExprKind::Unary(u) => self.scan_expr(&u.operand),
            ExprKind::Cast(c) => self.scan_expr(&c.operand),
            _ => Ok(()),
        }
    }

    /// Recursive defcal dispatch for a gate call inside a defcal body:
    /// find the called gate's own defcal (same matching rule) and walk its
    /// body against this walk's frame clocks.
    fn gate_call(&mut self, gc: &crate::sir::GateCall<Expr>, span: Span) -> Result<()> {
        if gc.duration.is_some() {
            return Err(err(
                ErrorKind::InvalidContext(
                    "gate duration designators are not supported in a timed defcal body".into(),
                ),
                span,
            ));
        }
        let table = self.table;
        let name = self.name(gc.gate).to_string();
        let mut qubits = Vec::new();
        for op in &gc.qubits {
            match op {
                QubitOperand::Hardware(n) => qubits.push(QubitRef::Hardware(*n)),
                QubitOperand::Indexed { symbol, indices } if indices.is_empty() => {
                    qubits.push(QubitRef::Symbol {
                        name: self.name(*symbol).to_string(),
                        index: None,
                    });
                }
                _ => {
                    return Err(err(
                        ErrorKind::InvalidContext(
                            "indexed qubit operands are not supported in a timed defcal body"
                                .into(),
                        ),
                        span,
                    ));
                }
            }
        }
        let call_args: Vec<Value> = gc
            .args
            .iter()
            .map(|a| eval_cal_expr(a, &self.params, table))
            .collect::<Result<_>>()?;
        let cal = table
            .gates
            .get(&name)
            .and_then(|cals| DefcalTable::find(cals, &qubits, call_args.len()))
            .ok_or_else(|| {
                err(
                    ErrorKind::InvalidContext(format!(
                        "no defcal matches `{name}` inside a timed defcal body"
                    )),
                    span,
                )
            })?;
        let CalibrationBody::OpenPulse(stmts) = &cal.body else {
            return Err(err(
                ErrorKind::InvalidContext("opaque defcal bodies cannot be timed".into()),
                cal.span,
            ));
        };
        let mut sub_params = HashMap::new();
        for (arg, v) in cal.args.iter().zip(&call_args) {
            if let CalibrationArg::Param(sid) = arg {
                sub_params.insert(*sid, v.clone());
            }
        }
        let saved_params = std::mem::replace(&mut self.params, sub_params);
        let saved_waveforms = std::mem::take(&mut self.waveforms);
        let result = stmts.iter().try_for_each(|s| self.stmt(s));
        self.params = saved_params;
        self.waveforms = saved_waveforms;
        result
    }

    fn play(&mut self, args: &[Expr], span: Span) -> Result<()> {
        let [frame, wf] = args else {
            return Err(err(
                ErrorKind::InvalidContext("play expects (frame, waveform)".into()),
                span,
            ));
        };
        let ExprKind::Var(frame_sid) = &frame.kind else {
            return Err(err(
                ErrorKind::InvalidContext("play frame must be a frame variable".into()),
                frame.span,
            ));
        };
        let dur = match &wf.kind {
            ExprKind::Var(wf_sid) => self.waveforms.get(wf_sid).copied().ok_or_else(|| {
                err(
                    ErrorKind::InvalidContext(
                        "played waveform's duration is not statically known".into(),
                    ),
                    wf.span,
                )
            })?,
            _ => self.waveform_duration(wf)?,
        };
        self.advance(*frame_sid, dur);
        Ok(())
    }

    fn capture(&mut self, args: &[Expr], span: Span) -> Result<()> {
        let [frame, samples] = args else {
            return Err(err(
                ErrorKind::InvalidContext("capture expects (frame, samples)".into()),
                span,
            ));
        };
        let ExprKind::Var(frame_sid) = &frame.kind else {
            return Err(err(
                ErrorKind::InvalidContext("capture frame must be a frame variable".into()),
                frame.span,
            ));
        };
        let v = eval_cal_expr(samples, &self.params, self.table)?;
        let n = value_as_usize(&v).ok_or_else(|| {
            err(
                ErrorKind::InvalidContext(
                    "capture sample count must be a constant non-negative integer".into(),
                ),
                samples.span,
            )
        })?;
        let dt = self.table.dt;
        self.advance(*frame_sid, Duration::new(n as f64 * dt.value, dt.unit));
        Ok(())
    }

    /// Duration of a waveform constructor call (`gaussian(amp, dur, sigma)`).
    fn waveform_duration(&self, e: &Expr) -> Result<Duration> {
        let ExprKind::Call(c) = &e.kind else {
            return Err(err(
                ErrorKind::InvalidContext("waveform value is not a constructor call".into()),
                e.span,
            ));
        };
        let CallTarget::Symbol(callee) = &c.callee else {
            return Err(err(
                ErrorKind::InvalidContext("waveform value is not a constructor call".into()),
                e.span,
            ));
        };
        match self.name(*callee) {
            "gaussian" => {
                let dur_expr = c.args.get(1).ok_or_else(|| {
                    err(
                        ErrorKind::InvalidContext("gaussian expects (amp, duration, sigma)".into()),
                        e.span,
                    )
                })?;
                let v = eval_cal_expr(dur_expr, &self.params, self.table)?;
                value_to_duration(&v, dur_expr.span)
            }
            other => Err(err(
                ErrorKind::InvalidContext(format!(
                    "cannot statically time waveform constructor `{other}`"
                )),
                e.span,
            )),
        }
    }
}

/// Constant evaluation inside a defcal body: literals, defcal params,
/// global consts, and arithmetic (no intrinsics).
fn eval_cal_expr(
    expr: &Expr,
    params: &HashMap<SymbolId, Value>,
    table: &DefcalTable,
) -> Result<Value> {
    match &expr.kind {
        ExprKind::Literal(p) => Ok(Value::from(p.clone())),
        ExprKind::Var(sid) => {
            if table.stretches.contains(sid) {
                return Err(stretch_err(expr.span));
            }
            params
                .get(sid)
                .or_else(|| table.consts.get(sid))
                .cloned()
                .ok_or_else(|| err(ErrorKind::NonConstantExpression, expr.span))
        }
        ExprKind::Binary(b) => {
            let lv = eval_cal_expr(&b.left, params, table)?;
            let rv = eval_cal_expr(&b.right, params, table)?;
            apply_binop(b.op, lv, rv, expr.span)
        }
        ExprKind::Unary(u) => {
            let v = eval_cal_expr(&u.operand, params, table)?;
            apply_unop(u.op, v, expr.span)
        }
        ExprKind::Cast(c) => {
            let v = eval_cal_expr(&c.operand, params, table)?;
            let ty = c
                .target_ty
                .value_ty()
                .ok_or_else(|| err(ErrorKind::NonConstantExpression, expr.span))?;
            v.cast(ty)
                .map_err(|_| err(ErrorKind::NonConstantExpression, expr.span))
        }
        _ => Err(err(ErrorKind::NonConstantExpression, expr.span)),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::compile_source;
    use crate::resolve::DefaultIncludeResolver;

    fn compile(source: &str) -> Program {
        compile_source(source, DefaultIncludeResolver, None).expect("compile ok")
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
        fn measurement(&self, _args: &MeasureArgs) -> Result<TimingDuration> {
            Ok(TimingDuration::ns(self.measure_ns))
        }
        fn reset(&self, _args: &ResetArgs) -> Result<TimingDuration> {
            Ok(TimingDuration::ns(self.reset_ns))
        }
        fn gate_call(&self, args: &GateCallArgs) -> Result<GateCallTiming> {
            if self.enter.iter().any(|n| n == &args.name) {
                return Ok(GateCallTiming::Enter);
            }
            let ns = self.gate_ns.get(&args.name).copied().unwrap_or(0.0);
            Ok(GateCallTiming::Duration(TimingDuration::ns(ns)))
        }
    }

    fn get_duration_literal(expr: &Expr) -> Duration {
        match &expr.kind {
            ExprKind::Literal(Primitive::Duration(d)) => *d,
            _ => panic!("expected literal duration, got {:?}", expr.ty),
        }
    }

    /// The init expression of the first decl-with-init (which lowers to an
    /// assignment), with any cast to the declared type unwrapped.
    fn first_init_expr(p: &Program) -> &Expr {
        let stmt = p
            .body
            .iter()
            .find(|s| matches!(&s.kind, StmtKind::Assignment(_)))
            .expect("assignment from decl init");
        let StmtKind::Assignment(a) = &stmt.kind else {
            unreachable!()
        };
        let RValue::Expr(e) = &a.value else {
            panic!("expected expr rvalue")
        };
        match &e.kind {
            ExprKind::Cast(c) => &c.operand,
            _ => e,
        }
    }

    fn first_init_duration(p: &Program) -> Duration {
        get_duration_literal(first_init_expr(p))
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
            fn measurement(&self, _a: &MeasureArgs) -> Result<TimingDuration> {
                Ok(TimingDuration::zero())
            }
            fn reset(&self, _a: &ResetArgs) -> Result<TimingDuration> {
                Ok(TimingDuration::zero())
            }
            fn gate_call(&self, _a: &GateCallArgs) -> Result<GateCallTiming> {
                Ok(GateCallTiming::Duration(TimingDuration::dt(4.0)))
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
    fn designator_short_circuits_enter() {
        // A `gate[duration]` designator IS the call's duration: the
        // enterable body (10 + 20) is never consulted.
        let src = r#"
            include "stdgates.inc";
            gate my_pair a { x a; y a; }
            duration d = durationof({ my_pair[70ns] $0; });
        "#;
        let mut p = compile(src);
        let t = FixedTimings::new()
            .gate("x", 10.0)
            .gate("y", 20.0)
            .enter("my_pair");
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Ns).value,
            70.0
        );
    }

    #[test]
    fn builtin_u_designator() {
        let src = r#"
            duration d = durationof({ U(0, 0, 0)[50ns] $0; });
        "#;
        let mut p = compile(src);
        resolve_durationof(&mut p, &TableTimings::new(), &CompileOptions::default()).unwrap();
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Ns).value,
            50.0
        );
    }

    #[test]
    fn negative_designator_errors() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({ x[-5ns] $0; });
        "#;
        let mut p = compile(src);
        let e = resolve_durationof(&mut p, &TableTimings::new(), &CompileOptions::default())
            .unwrap_err();
        match e.kind {
            ErrorKind::InvalidContext(msg) => assert!(msg.contains("negative"), "{msg}"),
            other => panic!("expected InvalidContext, got {other:?}"),
        }
    }

    #[test]
    fn designator_in_defcal_body_errors() {
        let src = format!(
            "include \"stdgates.inc\";
            {CAL_PRELUDE}
            defcal a2 q {{
                x[10ns] q;
            }}
            duration d = durationof({{ a2 $0; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        let e = resolve_durationof(&mut p, &t, &opts).unwrap_err();
        match e.kind {
            ErrorKind::InvalidContext(msg) => {
                assert!(msg.contains("defcal"), "{msg}");
            }
            other => panic!("expected InvalidContext, got {other:?}"),
        }
    }

    #[test]
    fn measure_call_designator_parse_errors() {
        // `f(1)[100ns] q -> c;` — designators have no meaning on measure
        // calls; the parser rejects them.
        assert!(
            compile_source(
                "qubit q;\nbit c;\nf(1)[100ns] q -> c;\n",
                DefaultIncludeResolver,
                None,
            )
            .is_err()
        );
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
        let t = FixedTimings::new()
            .gate("h", 20.0)
            .measure(200.0)
            .reset(50.0);
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
        let e = first_init_expr(&p);
        if let ExprKind::Binary(b) = &e.kind {
            let inner = get_duration_literal(&b.left);
            assert_eq!(inner.value, 15.0);
        } else {
            panic!("expected binary expression after resolution");
        }
    }

    // ── TableTimings ────────────────────────────────────────────────────

    #[test]
    fn table_timings_named_entry_wins() {
        let src = r#"
            include "stdgates.inc";
            gate my_pair a { x a; y a; }
            duration d = durationof({ my_pair $0; });
        "#;
        let mut p = compile(src);
        // `my_pair` is enterable (body would derive 0ns with no entries),
        // but its named entry takes precedence.
        let t = TableTimings::new()
            .gate("my_pair", TimingDuration::ns(70.0))
            .with_program_gates(&p);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(first_init_duration(&p).value, 70.0);
    }

    #[test]
    fn table_timings_enters_defined_gates() {
        let src = r#"
            include "stdgates.inc";
            gate my_pair a { x a; y a; }
            duration d = durationof({ my_pair $0; });
        "#;
        let mut p = compile(src);
        let t = TableTimings::new()
            .gate("x", TimingDuration::ns(10.0))
            .gate("y", TimingDuration::ns(20.0))
            .with_program_gates(&p);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(first_init_duration(&p).value, 30.0);
    }

    #[test]
    fn table_timings_derives_through_u() {
        // Only the built-in `U` has an entry; the user gate derives its
        // duration by recursing down to it.
        let src = r#"
            gate my_x a { U(3.14, 0.0, 3.14) a; }
            duration d = durationof({ my_x $0; });
        "#;
        let mut p = compile(src);
        let t = TableTimings::new()
            .gate("U", TimingDuration::ns(40.0))
            .with_program_gates(&p);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(first_init_duration(&p).value, 40.0);
    }

    #[test]
    fn table_timings_unknown_gate_is_zero() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({ x $0; });
        "#;
        let mut p = compile(src);
        // Empty table, no enterable gates: `x` costs 0ns, no error.
        let t = TableTimings::new();
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(first_init_duration(&p).value, 0.0);
    }

    #[test]
    fn table_timings_measure_reset_defaults() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                reset $0;
                bit c = measure $0;
            });
        "#;
        // Defaults: both 0ns.
        let mut p = compile(src);
        resolve_durationof(&mut p, &TableTimings::new(), &CompileOptions::default()).unwrap();
        assert_eq!(first_init_duration(&p).value, 0.0);

        // Reserved names via the string entry point.
        let mut p = compile(src);
        let t = TableTimings::from_str_entries(
            [("measure", "200ns"), ("reset", "50ns")],
            &CompileOptions::default().dt,
        )
        .unwrap();
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Ns).value,
            250.0
        );
    }

    #[test]
    fn table_timings_from_str_entries_dt() {
        let dt = Duration::new(0.5, DurationUnit::Ns);
        let t = TableTimings::from_str_entries([("x", "4dt")], &dt).unwrap();
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({ x $0; });
        "#;
        let mut p = compile(src);
        resolve_durationof(&mut p, &t, &CompileOptions::default()).unwrap();
        assert_eq!(first_init_duration(&p).to_unit(DurationUnit::Ns).value, 2.0);

        let err = TableTimings::from_str_entries([("x", "-5ns")], &dt).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::InvalidLiteral(_)));
        let err = TableTimings::from_str_entries([("x", "abc")], &dt).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::InvalidLiteral(_)));
    }

    // ── Defcal-derived timings ──────────────────────────────────────────

    const CAL_PRELUDE: &str = r#"
        defcalgrammar "openpulse";
        cal {
            extern port d0;
            extern port d1;
            frame drive0 = newframe(d0, 5.0e9, 0.0);
            frame cr1 = newframe(d1, 5.2e9, 0.0);
        }
    "#;

    #[test]
    fn defcal_timings_derives_play_duration() {
        let src = format!(
            "{CAL_PRELUDE}
            defcal x q {{
                waveform wf = gaussian(0.1, 100dt, 30dt);
                play(drive0, wf);
            }}
            duration d = durationof({{ x $1; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default(); // dt = 1us, so 100dt = 100us
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        resolve_durationof(&mut p, &t, &opts).unwrap();
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Us).value,
            100.0
        );
    }

    #[test]
    fn defcal_timings_max_across_frames() {
        // The shared defcal fixture: `cx $0, $1` recursively dispatches to
        // zx90_ix/x defcals; cr1 plays twice at 160us — the busiest frame.
        // Twin of the VM test `durationof_takes_max_across_frames` (320us).
        let fixture = include_str!("../../fixtures/qasm/defcal.qasm");
        let src = format!("{fixture}\nduration d = durationof({{ cx $0, $1; }});\n");
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        resolve_durationof(&mut p, &t, &opts).unwrap();
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Us).value,
            320.0
        );
    }

    #[test]
    fn defcal_timings_capture_uses_dt() {
        let src = format!(
            "{CAL_PRELUDE}
            defcal measure q -> bit {{
                return threshold(capture(drive0, 200), 100);
            }}
            duration d = durationof({{ bit c = measure $0; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions {
            dt: Duration::new(0.5, DurationUnit::Ns),
            ..Default::default()
        };
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        resolve_durationof(&mut p, &t, &opts).unwrap();
        // 200 samples × 0.5 ns/dt = 100 ns.
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Ns).value,
            100.0
        );
    }

    #[test]
    fn defcal_table_entry_overrides_defcal() {
        let src = format!(
            "{CAL_PRELUDE}
            defcal x q {{
                play(drive0, gaussian(0.1, 100us, 25us));
            }}
            duration d = durationof({{ x $0; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new()
            .gate("x", TimingDuration::ns(5.0))
            .with_defcals(&p, &opts.dt);
        resolve_durationof(&mut p, &t, &opts).unwrap();
        assert_eq!(first_init_duration(&p).to_unit(DurationUnit::Ns).value, 5.0);
    }

    #[test]
    fn defcal_hardware_match_beats_wildcard() {
        let src = format!(
            "{CAL_PRELUDE}
            defcal x q {{
                play(drive0, gaussian(0.1, 100us, 25us));
            }}
            defcal x $0 {{
                play(drive0, gaussian(0.1, 200us, 50us));
            }}
            duration a = durationof({{ x $0; }});
            duration b = durationof({{ x $1; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        resolve_durationof(&mut p, &t, &opts).unwrap();
        let durations: Vec<f64> = p
            .body
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::Assignment(a) => match &a.value {
                    RValue::Expr(e) => {
                        let e = match &e.kind {
                            ExprKind::Cast(c) => &c.operand,
                            _ => e,
                        };
                        Some(get_duration_literal(e).to_unit(DurationUnit::Us).value)
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(durations, vec![200.0, 100.0]);
    }

    #[test]
    fn defcal_param_substitution() {
        // The defcal's duration depends on a classical parameter bound at
        // the call site.
        let src = format!(
            "{CAL_PRELUDE}
            defcal rx(duration len) q {{
                play(drive0, gaussian(0.1, len, 25us));
            }}
            duration d = durationof({{ rx(150us) $0; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        resolve_durationof(&mut p, &t, &opts).unwrap();
        assert_eq!(
            first_init_duration(&p).to_unit(DurationUnit::Us).value,
            150.0
        );
    }

    #[test]
    fn stretch_in_timed_scope_names_stretch() {
        // Stretchy delays inside `durationof` are resolved these days;
        // positions with no stretch semantics (a gate *argument*) still get
        // the honest named error.
        let src = r#"
            include "stdgates.inc";
            stretch sd;
            duration d = durationof({ rx(sd) $0; });
        "#;
        let mut p = compile(src);
        let e = resolve_durationof(&mut p, &TableTimings::new(), &CompileOptions::default())
            .unwrap_err();
        match e.kind {
            ErrorKind::InvalidContext(msg) => {
                assert!(msg.contains("stretch"), "{msg}");
            }
            other => panic!("expected InvalidContext naming stretch, got {other:?}"),
        }
    }

    #[test]
    fn defcal_stretch_names_stretch() {
        let src = format!(
            "{CAL_PRELUDE}
            stretch sd;
            defcal x q {{
                play(drive0, gaussian(0.1, sd, 25us));
            }}
            duration d = durationof({{ x $0; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        let e = resolve_durationof(&mut p, &t, &opts).unwrap_err();
        match e.kind {
            ErrorKind::InvalidContext(msg) => {
                assert!(msg.contains("stretch"), "{msg}");
            }
            other => panic!("expected InvalidContext naming stretch, got {other:?}"),
        }
    }

    #[test]
    fn defcal_nonconst_duration_errors() {
        let src = format!(
            "{CAL_PRELUDE}
            input duration len;
            defcal x q {{
                play(drive0, gaussian(0.1, len, 25us));
            }}
            duration d = durationof({{ x $0; }});"
        );
        let mut p = compile(&src);
        let opts = CompileOptions::default();
        let t = TableTimings::new().with_defcals(&p, &opts.dt);
        let err = resolve_durationof(&mut p, &t, &opts).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::NonConstantExpression));
    }

    #[test]
    fn table_timings_control_flow_errors() {
        let src = r#"
            include "stdgates.inc";
            duration d = durationof({
                for int i in [0:2] { x $0; }
            });
        "#;
        let mut p = compile(src);
        let err = resolve_durationof(&mut p, &TableTimings::new(), &CompileOptions::default())
            .unwrap_err();
        assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
    }
}
