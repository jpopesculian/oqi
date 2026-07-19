//! v1 `stretch` constraint resolver.
//!
//! The spec (docs/delays.rst) models `stretch` after TeX's boxes and glue:
//! flat, non-negative durations resolved at compile time so that stretchy
//! instructions fill their enclosing spans exactly. This resolver walks the
//! straight-line program body building per-qubit clocks as affine forms
//! (`fixed + Σ coeff·stretch`), emits an equality constraint wherever a
//! stretchy wire meets a synchronization point (differing-clock multi-wire
//! ops, barriers, box boundaries, end of program), solves greedily in
//! program order, and rewrites every stretch variable reference to its
//! resolved constant. Spans are minimal-feasible (`max` of the fixed parts,
//! clamped at zero), so an unconstrained stretch resolves to 0 — matching
//! the VM's default-initialized runtime behavior. Underdetermined
//! constraints split slack equally per unit coefficient. Everything outside
//! straight-line top-level code errs honestly rather than mistiming.

use std::collections::{HashMap, HashSet};

use oqi_lex::Span;

use crate::classical::{
    Duration, DurationUnit, FloatWidth, Primitive, PrimitiveTy, Value, ValueTy,
};
use crate::error::{ErrorKind, Result};
use crate::sir::{
    Expr, ExprKind, IndexItem, IndexKind, IndexOp, LValue, MeasureExprKind, Program, QubitOperand,
    RValue, Stmt, StmtKind, SwitchLabels,
};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::{CompileOptions, Type};

use super::{CalibrationBody, Frame, QubitRef, ResolveCtx, Timings, err, resolve_qubit_operands};

const EPS_NS: f64 = 1e-9;

fn zero() -> Duration {
    Duration::new(0.0, DurationUnit::Ns)
}

fn ns(d: Duration) -> f64 {
    d.to_unit(DurationUnit::Ns).value
}

fn is_duration_typed(ty: &Type) -> bool {
    matches!(ty.value_ty(), Some(ValueTy::Scalar(PrimitiveTy::Duration)))
}

// ── Entry point ─────────────────────────────────────────────────────────

/// Resolve every `stretch` in `program` to a constant duration. Called as
/// Pass 4 of [`super::resolve_durationof`] (gate durations come from
/// `timings`); the caller guarantees at least one stretch-typed symbol
/// exists.
pub(super) fn resolve_stretch<T: Timings>(
    program: &mut Program,
    timings: &T,
    options: &CompileOptions,
) -> Result<()> {
    boundary_scan(program)?;
    // An unused `stretch` declaration changes nothing.
    if !stmts_have_stretch_var(&program.body, &program.symbols) {
        return Ok(());
    }
    if stmts_have_subroutine_call(&program.body, &program.subroutines) {
        return Err(err(
            ErrorKind::InvalidContext(
                "subroutine calls are not supported where stretch is resolved".into(),
            ),
            Span::default(),
        ));
    }

    let values = {
        let ctx = ResolveCtx {
            timings,
            symbols: &program.symbols,
            gates: &program.gates,
            options,
        };
        let mut resolver = StretchResolver {
            ctx,
            env: HashMap::new(),
            stretch_assigned: HashSet::new(),
            solver: Solver {
                solved: HashMap::new(),
            },
            delay_checks: Vec::new(),
        };
        let mut clocks = AffineClocks::default();
        resolver.walk_stmts(&program.body, &mut clocks)?;
        // End of program is the outermost synchronization point.
        let wires: Vec<QubitRef> = clocks.times.iter().map(|(q, _)| q.clone()).collect();
        if !wires.is_empty() {
            resolver.sync_point(&mut clocks, &wires, Span::default(), "end of program")?;
        }
        resolver.finalize(&program.symbols)?
    };

    rewrite_stretch_vars(&mut program.body, &values);
    Ok(())
}

// ── Affine durations ────────────────────────────────────────────────────

/// `fixed + Σ coeff·stretch`. Terms hold only nonzero coefficients.
#[derive(Clone, Debug, PartialEq)]
struct AffineDur {
    fixed: Duration,
    terms: HashMap<SymbolId, f64>,
}

impl AffineDur {
    fn concrete(d: Duration) -> Self {
        Self {
            fixed: d,
            terms: HashMap::new(),
        }
    }

    fn var(sid: SymbolId) -> Self {
        Self {
            fixed: zero(),
            terms: HashMap::from([(sid, 1.0)]),
        }
    }

    fn add(&self, o: &Self) -> Self {
        let mut terms = self.terms.clone();
        for (sid, c) in &o.terms {
            let e = terms.entry(*sid).or_insert(0.0);
            *e += c;
            if e.abs() < EPS_NS {
                terms.remove(sid);
            }
        }
        Self {
            fixed: self.fixed + o.fixed,
            terms,
        }
    }

    fn sub(&self, o: &Self) -> Self {
        self.add(&o.scale(-1.0))
    }

    fn scale(&self, k: f64) -> Self {
        if k == 0.0 {
            return Self::concrete(zero());
        }
        Self {
            fixed: self.fixed * k,
            terms: self.terms.iter().map(|(s, c)| (*s, c * k)).collect(),
        }
    }

    /// Fold solved stretches into the fixed part.
    fn substituted(&self, solved: &HashMap<SymbolId, Duration>) -> Self {
        let mut fixed = self.fixed;
        let mut terms = HashMap::new();
        for (sid, c) in &self.terms {
            match solved.get(sid) {
                Some(v) => fixed = fixed + *v * *c,
                None => {
                    terms.insert(*sid, *c);
                }
            }
        }
        Self { fixed, terms }
    }
}

// ── Greedy solver ───────────────────────────────────────────────────────

struct Solver {
    solved: HashMap<SymbolId, Duration>,
}

impl Solver {
    /// Enforce `lhs == rhs` after substituting solved stretches. Remaining
    /// unknowns split the residual equally per unit coefficient.
    fn require_eq(
        &mut self,
        lhs: &AffineDur,
        rhs: &AffineDur,
        symbols: &SymbolTable,
        span: Span,
        what: &str,
    ) -> Result<()> {
        let diff = lhs.sub(rhs).substituted(&self.solved);
        let names = |terms: &HashMap<SymbolId, f64>| {
            let mut ids: Vec<SymbolId> = terms.keys().copied().collect();
            ids.sort_by_key(|s| s.0);
            ids.iter()
                .map(|s| symbols.get(*s).name.clone())
                .collect::<Vec<_>>()
                .join("`, `")
        };
        if diff.terms.is_empty() {
            let residual = ns(diff.fixed);
            if residual.abs() > EPS_NS {
                let involved = names(&lhs.sub(rhs).terms);
                return Err(err(
                    ErrorKind::InvalidContext(format!(
                        "over-constrained stretch system: `{involved}` cannot satisfy {what} \
                         (residual {residual}ns)"
                    )),
                    span,
                ));
            }
            return Ok(());
        }
        let coeff_sum: f64 = diff.terms.values().sum();
        if coeff_sum.abs() < EPS_NS {
            return Err(err(
                ErrorKind::InvalidContext(format!(
                    "stretch weights cancel in the constraint for `{}` at {what}",
                    names(&diff.terms)
                )),
                span,
            ));
        }
        let v = -ns(diff.fixed) / coeff_sum;
        if v < -EPS_NS {
            return Err(err(
                ErrorKind::InvalidContext(format!(
                    "stretch `{}` resolves to a negative duration ({v}ns) at {what} \
                     (durations must be non-negative after stretch resolution)",
                    names(&diff.terms)
                )),
                span,
            ));
        }
        let v = Duration::new(v.max(0.0), DurationUnit::Ns);
        for sid in diff.terms.keys() {
            self.solved.insert(*sid, v);
        }
        Ok(())
    }
}

// ── Timeline walk ───────────────────────────────────────────────────────

/// Per-wire affine clocks, insertion-ordered for determinism.
#[derive(Default)]
struct AffineClocks {
    times: Vec<(QubitRef, AffineDur)>,
}

impl AffineClocks {
    fn idx(&mut self, qr: &QubitRef) -> usize {
        if let Some(i) = self.times.iter().position(|(q, _)| q == qr) {
            return i;
        }
        self.times.push((qr.clone(), AffineDur::concrete(zero())));
        self.times.len() - 1
    }
}

struct StretchResolver<'a, T: Timings> {
    ctx: ResolveCtx<'a, T>,
    /// Duration/stretch-typed variables at the current program point.
    env: HashMap<SymbolId, AffineDur>,
    /// Stretch-typed variables already assigned (single-assignment).
    stretch_assigned: HashSet<SymbolId>,
    solver: Solver,
    /// Resolved-value non-negativity checks deferred to after solving.
    delay_checks: Vec<(AffineDur, Span)>,
}

impl<T: Timings> StretchResolver<'_, T> {
    fn walk_stmts(&mut self, stmts: &[Stmt], clocks: &mut AffineClocks) -> Result<()> {
        for stmt in stmts {
            self.walk_stmt(stmt, clocks)?;
        }
        Ok(())
    }

    fn walk_stmt(&mut self, stmt: &Stmt, clocks: &mut AffineClocks) -> Result<()> {
        let span = stmt.span;
        let frame = Frame::default();
        match &stmt.kind {
            StmtKind::GateCall(gc) => {
                // A `gate[duration]` designator is the call's duration and
                // may be stretchy (spec delays.rst:212-222, the rotary
                // idiom); it bypasses the Timings lookup entirely.
                if let Some(d) = &gc.duration {
                    let dur = self.eval_affine_expr(d)?;
                    self.delay_checks.push((dur.clone(), d.span));
                    let wires = resolve_qubit_operands(&gc.qubits, self.ctx.symbols, &frame, span)?;
                    return self.advance(clocks, &wires, &dur, span);
                }
                let (wires, dur) = self.ctx.resolve_gate_call_duration(
                    gc.gate,
                    &gc.modifiers,
                    &gc.args,
                    &gc.qubits,
                    None,
                    &frame,
                    span,
                )?;
                self.advance(clocks, &wires, &AffineDur::concrete(dur), span)
            }
            StmtKind::Measure(m) => {
                let (wires, dur) = self.ctx.resolve_measure_duration(&m.measure, &frame)?;
                self.advance(clocks, &wires, &AffineDur::concrete(dur), span)
            }
            StmtKind::Reset(operand) => {
                let (wires, dur) = self.ctx.resolve_reset_duration(operand, &frame, span)?;
                self.advance(clocks, &wires, &AffineDur::concrete(dur), span)
            }
            StmtKind::Delay(d) => {
                let dur = self.eval_affine_expr(&d.duration)?;
                self.delay_checks.push((dur.clone(), d.duration.span));
                if d.operands.is_empty() {
                    if !dur.terms.is_empty() {
                        return Err(err(
                            ErrorKind::InvalidContext(
                                "a stretchy delay requires explicit qubit operands".into(),
                            ),
                            span,
                        ));
                    }
                    // Bare concrete delay: no runtime effect (matches VM).
                    return Ok(());
                }
                let wires = resolve_qubit_operands(&d.operands, self.ctx.symbols, &frame, span)?;
                self.advance(clocks, &wires, &dur, span)
            }
            StmtKind::Barrier(ops) | StmtKind::Nop(ops) => {
                let wires = if ops.is_empty() {
                    clocks.times.iter().map(|(q, _)| q.clone()).collect()
                } else {
                    resolve_qubit_operands(ops, self.ctx.symbols, &frame, span)?
                };
                if !wires.is_empty() {
                    self.sync_point(clocks, &wires, span, "barrier synchronization")?;
                }
                Ok(())
            }
            StmtKind::Box(b) => self.walk_box(b, clocks, span),
            StmtKind::Assignment(a) => {
                if let RValue::Measure(m) = &a.value {
                    let (wires, dur) = self.ctx.resolve_measure_duration(m, &frame)?;
                    return self.advance(clocks, &wires, &AffineDur::concrete(dur), span);
                }
                let RValue::Expr(e) = &a.value else {
                    return Ok(());
                };
                if let LValue::Var(sid) = &a.target {
                    let sym = self.ctx.symbols.get(*sid);
                    if is_duration_typed(&sym.ty) {
                        if matches!(sym.ty, Type::Stretch) && !self.stretch_assigned.insert(*sid) {
                            return Err(err(
                                ErrorKind::InvalidContext(format!(
                                    "stretch `{}` is assigned more than once; stretch values \
                                     are single-assignment",
                                    sym.name
                                )),
                                span,
                            ));
                        }
                        let v = self.eval_affine_expr(e)?;
                        self.env.insert(*sid, v);
                        return Ok(());
                    }
                }
                // Other classical assignments are zero-width.
                Ok(())
            }
            // Time-free statements.
            StmtKind::ExprStmt(_)
            | StmtKind::Pragma(_)
            | StmtKind::Break
            | StmtKind::Continue
            | StmtKind::Return(_) => Ok(()),
            StmtKind::If(_) | StmtKind::For(_) | StmtKind::While(_) | StmtKind::Switch(_) => {
                Err(err(
                    ErrorKind::InvalidContext(
                        "stretch resolution requires straight-line code (control flow is \
                         not supported)"
                            .into(),
                    ),
                    span,
                ))
            }
            StmtKind::End => Err(err(
                ErrorKind::InvalidContext(
                    "`end` is not supported where stretch is resolved".into(),
                ),
                span,
            )),
            StmtKind::Cal(_) => Err(err(
                ErrorKind::InvalidContext(
                    "calibration blocks cannot be timed where stretch is resolved".into(),
                ),
                span,
            )),
            StmtKind::Alias(_) => Err(err(
                ErrorKind::InvalidContext(
                    "qubit aliases are not supported where stretch is resolved".into(),
                ),
                span,
            )),
        }
    }

    /// A box is walked in its own relative clock set; its resolved duration
    /// then advances the enclosing timeline as one rigid block.
    fn walk_box(
        &mut self,
        b: &crate::sir::BoxStmt,
        clocks: &mut AffineClocks,
        span: Span,
    ) -> Result<()> {
        let mut inner = AffineClocks::default();
        self.walk_stmts(&b.body, &mut inner)?;

        // Minimal feasible interior span.
        let mut s_min = zero();
        for (_, clock) in &inner.times {
            let f = clock.substituted(&self.solver.solved).fixed;
            if f > s_min {
                s_min = f;
            }
        }

        let d = match &b.duration {
            Some(e) => self.eval_affine_expr(e)?,
            None => AffineDur::concrete(s_min),
        };
        // `box[st]` with concrete contents: the designator itself is pinned
        // to the minimal interior span.
        if !d.substituted(&self.solver.solved).terms.is_empty() {
            self.solver.require_eq(
                &d,
                &AffineDur::concrete(s_min),
                self.ctx.symbols,
                span,
                "the box duration",
            )?;
        }
        let d_conc = d.substituted(&self.solver.solved);
        debug_assert!(d_conc.terms.is_empty());
        if ns(d_conc.fixed) < ns(s_min) - EPS_NS {
            return Err(err(
                ErrorKind::InvalidContext(format!(
                    "box contents exceed the box duration ({}ns of content in a {}ns box)",
                    ns(s_min),
                    ns(d_conc.fixed)
                )),
                span,
            ));
        }

        // Stretchy interior wires must fill the box exactly.
        let stretchy: Vec<(QubitRef, AffineDur)> = inner
            .times
            .iter()
            .filter(|(_, c)| !c.terms.is_empty())
            .cloned()
            .collect();
        for (_, clock) in &stretchy {
            self.solver
                .require_eq(clock, &d_conc, self.ctx.symbols, span, "the box duration")?;
        }

        // Enclosing timeline: entry sync over the box's wires, then the box
        // advances them as one rigid block.
        let wires: Vec<QubitRef> = inner.times.iter().map(|(q, _)| q.clone()).collect();
        if !wires.is_empty() {
            let entry = self.sync_point(clocks, &wires, span, "box entry")?;
            for w in &wires {
                let i = clocks.idx(w);
                clocks.times[i].1 = AffineDur::concrete(entry + d_conc.fixed);
            }
        }
        Ok(())
    }

    /// Advance `wires` by `dur`. Only a multi-wire op over differing clocks
    /// is a synchronization point; otherwise clocks advance unconstrained.
    fn advance(
        &mut self,
        clocks: &mut AffineClocks,
        wires: &[QubitRef],
        dur: &AffineDur,
        span: Span,
    ) -> Result<()> {
        if wires.is_empty() {
            return Ok(());
        }
        let idxs: Vec<usize> = wires.iter().map(|w| clocks.idx(w)).collect();
        let all_equal = idxs
            .iter()
            .all(|&i| clocks.times[i].1 == clocks.times[idxs[0]].1);
        if !all_equal {
            self.sync_point(clocks, wires, span, "multi-qubit synchronization")?;
        }
        for &i in &idxs {
            clocks.times[i].1 = clocks.times[i].1.add(dur);
        }
        Ok(())
    }

    /// Synchronize `wires`: the span is the minimal feasible time (max of
    /// the substituted fixed parts, clamped at zero); wires whose clocks
    /// carry stretch must fill it exactly, stretch-free wires idle
    /// implicitly. All listed clocks become the span.
    fn sync_point(
        &mut self,
        clocks: &mut AffineClocks,
        wires: &[QubitRef],
        span: Span,
        what: &str,
    ) -> Result<Duration> {
        let idxs: Vec<usize> = wires.iter().map(|w| clocks.idx(w)).collect();
        let mut sp = zero();
        for &i in &idxs {
            let f = clocks.times[i].1.substituted(&self.solver.solved).fixed;
            if f > sp {
                sp = f;
            }
        }
        let target = AffineDur::concrete(sp);
        for &i in &idxs {
            let clock = clocks.times[i].1.clone();
            if !clock.terms.is_empty() {
                self.solver
                    .require_eq(&clock, &target, self.ctx.symbols, span, what)?;
            }
        }
        for &i in &idxs {
            clocks.times[i].1 = target.clone();
        }
        Ok(sp)
    }

    // ── Affine expression evaluation ────────────────────────────────────

    fn eval_affine_expr(&self, expr: &Expr) -> Result<AffineDur> {
        // Concrete fast path: consts, intrinsics, folded arithmetic, and
        // dt-resolved literals all behave exactly as in the main pass.
        if let Ok(v) = self.ctx.eval_const_expr(expr, &Frame::default()) {
            return Ok(AffineDur::concrete(super::value_to_duration(
                &v, expr.span,
            )?));
        }
        match &expr.kind {
            ExprKind::Var(sid) => {
                if let Some(v) = self.env.get(sid) {
                    return Ok(v.clone());
                }
                let sym = self.ctx.symbols.get(*sid);
                if matches!(sym.ty, Type::Stretch) {
                    return Ok(AffineDur::var(*sid));
                }
                Err(err(
                    ErrorKind::InvalidContext(format!(
                        "`{}` cannot be statically evaluated in a stretch expression",
                        sym.name
                    )),
                    expr.span,
                ))
            }
            ExprKind::Binary(b) => {
                use crate::sir::BinOp;
                match b.op {
                    BinOp::Add => Ok(self
                        .eval_affine_expr(&b.left)?
                        .add(&self.eval_affine_expr(&b.right)?)),
                    BinOp::Sub => Ok(self
                        .eval_affine_expr(&b.left)?
                        .sub(&self.eval_affine_expr(&b.right)?)),
                    BinOp::Mul => {
                        if let Some(k) = self.const_scalar_f64(&b.left) {
                            Ok(self.eval_affine_expr(&b.right)?.scale(k))
                        } else if let Some(k) = self.const_scalar_f64(&b.right) {
                            Ok(self.eval_affine_expr(&b.left)?.scale(k))
                        } else {
                            Err(err(
                                ErrorKind::InvalidContext(
                                    "stretch expressions must be linear".into(),
                                ),
                                expr.span,
                            ))
                        }
                    }
                    BinOp::Div => {
                        if let Some(k) = self.const_scalar_f64(&b.right) {
                            if k == 0.0 {
                                return Err(err(
                                    ErrorKind::InvalidContext(
                                        "division by zero in a stretch expression".into(),
                                    ),
                                    expr.span,
                                ));
                            }
                            Ok(self.eval_affine_expr(&b.left)?.scale(1.0 / k))
                        } else {
                            Err(err(
                                ErrorKind::InvalidContext(
                                    "stretch expressions must be linear".into(),
                                ),
                                expr.span,
                            ))
                        }
                    }
                    _ => Err(err(
                        ErrorKind::InvalidContext("stretch expressions must be linear".into()),
                        expr.span,
                    )),
                }
            }
            ExprKind::Unary(u) => {
                use crate::sir::UnOp;
                match u.op {
                    UnOp::Neg => Ok(self.eval_affine_expr(&u.operand)?.scale(-1.0)),
                    _ => Err(err(
                        ErrorKind::InvalidContext("stretch expressions must be linear".into()),
                        expr.span,
                    )),
                }
            }
            // Decl inits are wrapped in a cast to the declared type.
            ExprKind::Cast(c) => self.eval_affine_expr(&c.operand),
            _ => Err(err(
                ErrorKind::InvalidContext(
                    "expression cannot be statically evaluated for stretch resolution".into(),
                ),
                expr.span,
            )),
        }
    }

    /// A constant scalar factor (int/uint/float), or None if not constant.
    fn const_scalar_f64(&self, expr: &Expr) -> Option<f64> {
        let v = self.ctx.eval_const_expr(expr, &Frame::default()).ok()?;
        let Value::Scalar(s) = v else { return None };
        s.cast(PrimitiveTy::Float(FloatWidth::F64))
            .ok()?
            .value()
            .as_float(FloatWidth::F64)
    }

    /// After the walk: default unsolved stretches to zero, evaluate derived
    /// stretches, and validate every recorded instruction duration.
    fn finalize(mut self, symbols: &SymbolTable) -> Result<HashMap<SymbolId, Duration>> {
        let stretch_syms: Vec<SymbolId> = symbols
            .iter()
            .filter(|s| matches!(s.ty, Type::Stretch))
            .map(|s| s.id)
            .collect();
        for sid in &stretch_syms {
            if !self.solver.solved.contains_key(sid) && !self.env.contains_key(sid) {
                self.solver.solved.insert(*sid, zero());
            }
        }
        let mut values = HashMap::new();
        for sid in &stretch_syms {
            let v = match self.env.get(sid) {
                Some(aff) => {
                    let sub = aff.substituted(&self.solver.solved);
                    if !sub.terms.is_empty() {
                        // Derived from stretches that themselves defaulted.
                        sub.substituted(&self.solver.solved).fixed
                    } else {
                        sub.fixed
                    }
                }
                None => self.solver.solved[sid],
            };
            values.insert(*sid, v);
        }
        for (aff, span) in &self.delay_checks {
            let v = aff.substituted(&self.solver.solved);
            let sub = v.substituted(&values);
            if ns(sub.fixed) < -EPS_NS {
                return Err(err(
                    ErrorKind::InvalidContext(format!(
                        "duration resolves to a negative value ({}ns) after stretch \
                         resolution (durations must be non-negative)",
                        ns(sub.fixed)
                    )),
                    *span,
                ));
            }
        }
        Ok(values)
    }
}

// ── Boundary scans ──────────────────────────────────────────────────────

fn boundary_scan(program: &Program) -> Result<()> {
    for s in program.symbols.iter() {
        if matches!(s.ty, Type::Stretch) && s.kind != SymbolKind::Variable {
            return Err(err(
                ErrorKind::InvalidContext(format!(
                    "stretch is only supported as a top-level variable (`{}` is declared \
                     as {:?})",
                    s.name, s.kind
                )),
                s.span,
            ));
        }
    }
    for sub in &program.subroutines {
        if matches!(sub.return_ty, Some(Type::Stretch)) {
            return Err(err(
                ErrorKind::InvalidContext("stretch cannot be a subroutine return type".into()),
                sub.span,
            ));
        }
    }
    for ext in &program.externs {
        if matches!(ext.return_ty, Some(Type::Stretch)) {
            return Err(err(
                ErrorKind::InvalidContext("stretch cannot be an extern return type".into()),
                ext.span,
            ));
        }
    }
    for g in &program.gates {
        if stmts_have_stretch_var(&g.body.body, &program.symbols) {
            return Err(err(
                ErrorKind::InvalidContext("stretch cannot be used inside gate bodies".into()),
                g.span,
            ));
        }
    }
    for sub in &program.subroutines {
        if stmts_have_stretch_var(&sub.body, &program.symbols) {
            return Err(err(
                ErrorKind::InvalidContext("stretch cannot be used inside subroutine bodies".into()),
                sub.span,
            ));
        }
    }
    Ok(())
}

// ── Mechanical scans ────────────────────────────────────────────────────

fn stmts_have_stretch_var(stmts: &[Stmt], symbols: &SymbolTable) -> bool {
    any_expr_in_stmts(
        stmts,
        &mut |e| matches!(&e.kind, ExprKind::Var(sid) if matches!(symbols.get(*sid).ty, Type::Stretch)),
    )
}

fn stmts_have_subroutine_call(stmts: &[Stmt], subs: &[crate::sir::SubroutineDecl]) -> bool {
    let sub_ids: HashSet<SymbolId> = subs.iter().map(|s| s.symbol).collect();
    any_expr_in_stmts(stmts, &mut |e| {
        matches!(&e.kind, ExprKind::Call(c)
            if matches!(&c.callee, crate::sir::CallTarget::Symbol(sid) if sub_ids.contains(sid)))
    })
}

fn any_expr_in_stmts(stmts: &[Stmt], pred: &mut dyn FnMut(&Expr) -> bool) -> bool {
    stmts.iter().any(|s| stmt_any_expr(s, pred))
}

fn stmt_any_expr(stmt: &Stmt, pred: &mut dyn FnMut(&Expr) -> bool) -> bool {
    let ops_any = |ops: &[QubitOperand<Expr>], pred: &mut dyn FnMut(&Expr) -> bool| {
        ops.iter().any(|o| match o {
            QubitOperand::Indexed { indices, .. } => {
                indices.iter().any(|i| index_any_expr(i, pred))
            }
            QubitOperand::Hardware(_) => false,
        })
    };
    match &stmt.kind {
        StmtKind::Alias(a) => a.value.iter().any(|e| expr_any(e, pred)),
        StmtKind::GateCall(gc) => {
            gc.modifiers.iter().any(|m| match m {
                crate::sir::GateModifier::Pow(e) => expr_any(e, pred),
                _ => false,
            }) || gc.args.iter().any(|e| expr_any(e, pred))
                || ops_any(&gc.qubits, pred)
                || gc.duration.as_ref().is_some_and(|e| expr_any(e, pred))
        }
        StmtKind::Measure(m) => measure_any_expr(&m.measure, pred),
        StmtKind::Reset(op) => ops_any(std::slice::from_ref(op), pred),
        StmtKind::Barrier(ops) | StmtKind::Nop(ops) => ops_any(ops, pred),
        StmtKind::Delay(d) => expr_any(&d.duration, pred) || ops_any(&d.operands, pred),
        StmtKind::Box(b) => {
            b.duration.as_ref().is_some_and(|e| expr_any(e, pred))
                || any_expr_in_stmts(&b.body, pred)
        }
        StmtKind::Assignment(a) => match &a.value {
            RValue::Expr(e) => expr_any(e, pred),
            RValue::Measure(m) => measure_any_expr(m, pred),
        },
        StmtKind::If(i) => {
            expr_any(&i.condition, pred)
                || any_expr_in_stmts(&i.then_body, pred)
                || i.else_body
                    .as_ref()
                    .is_some_and(|b| any_expr_in_stmts(b, pred))
        }
        StmtKind::For(f) => {
            let it = match &f.iterable {
                crate::sir::ForIterable::Range { start, step, end } => [start, step, end]
                    .into_iter()
                    .flatten()
                    .any(|e| expr_any(e, pred)),
                crate::sir::ForIterable::Set(v) => v.iter().any(|e| expr_any(e, pred)),
                crate::sir::ForIterable::Expr(e) => expr_any(e, pred),
            };
            it || any_expr_in_stmts(&f.body, pred)
        }
        StmtKind::While(w) => expr_any(&w.condition, pred) || any_expr_in_stmts(&w.body, pred),
        StmtKind::Switch(s) => {
            expr_any(&s.target, pred)
                || s.cases.iter().any(|c| {
                    (match &c.labels {
                        SwitchLabels::Values(v) => v.iter().any(|e| expr_any(e, pred)),
                        SwitchLabels::Default => false,
                    }) || any_expr_in_stmts(&c.body, pred)
                })
        }
        StmtKind::Return(Some(rv)) => match rv {
            RValue::Expr(e) => expr_any(e, pred),
            RValue::Measure(m) => measure_any_expr(m, pred),
        },
        StmtKind::ExprStmt(e) => expr_any(e, pred),
        StmtKind::Cal(CalibrationBody::OpenPulse(body)) => any_expr_in_stmts(body, pred),
        StmtKind::Return(None)
        | StmtKind::Break
        | StmtKind::Continue
        | StmtKind::End
        | StmtKind::Pragma(_)
        | StmtKind::Cal(CalibrationBody::Opaque(_)) => false,
    }
}

fn measure_any_expr(
    m: &crate::sir::MeasureExpr<Expr>,
    pred: &mut dyn FnMut(&Expr) -> bool,
) -> bool {
    match &m.kind {
        MeasureExprKind::Measure { operand } => match operand {
            QubitOperand::Indexed { indices, .. } => {
                indices.iter().any(|i| index_any_expr(i, pred))
            }
            QubitOperand::Hardware(_) => false,
        },
        MeasureExprKind::QuantumCall { args, qubits, .. } => {
            args.iter().any(|e| expr_any(e, pred))
                || qubits.iter().any(|o| match o {
                    QubitOperand::Indexed { indices, .. } => {
                        indices.iter().any(|i| index_any_expr(i, pred))
                    }
                    QubitOperand::Hardware(_) => false,
                })
        }
    }
}

fn index_any_expr(op: &IndexOp<Expr>, pred: &mut dyn FnMut(&Expr) -> bool) -> bool {
    match &op.kind {
        IndexKind::Set(exprs) => exprs.iter().any(|e| expr_any(e, pred)),
        IndexKind::Items(items) => items.iter().any(|it| match it {
            IndexItem::Single(e) => expr_any(e, pred),
            IndexItem::Range(r) => [&r.start, &r.step, &r.end]
                .into_iter()
                .flatten()
                .any(|e| expr_any(e, pred)),
        }),
    }
}

fn expr_any(e: &Expr, pred: &mut dyn FnMut(&Expr) -> bool) -> bool {
    if pred(e) {
        return true;
    }
    match &e.kind {
        ExprKind::Binary(b) => expr_any(&b.left, pred) || expr_any(&b.right, pred),
        ExprKind::Unary(u) => expr_any(&u.operand, pred),
        ExprKind::Cast(c) => expr_any(&c.operand, pred),
        ExprKind::Index(ix) => expr_any(&ix.base, pred) || index_any_expr(&ix.index, pred),
        ExprKind::Call(c) => c.args.iter().any(|a| expr_any(a, pred)),
        ExprKind::ArrayLiteral(al) => al.items.iter().any(|a| expr_any(a, pred)),
        ExprKind::DurationOf(stmts) => any_expr_in_stmts(stmts, pred),
        ExprKind::Literal(_) | ExprKind::Var(_) | ExprKind::HardwareQubit(_) => false,
    }
}

// ── Rewrite ─────────────────────────────────────────────────────────────

/// Replace every reference to a stretch variable with its resolved constant.
/// Duration variables derived from stretches keep their assignments — the
/// rewritten RHS is concrete arithmetic the VM evaluates identically.
fn rewrite_stretch_vars(stmts: &mut [Stmt], values: &HashMap<SymbolId, Duration>) {
    for stmt in stmts {
        rewrite_stmt(stmt, values);
    }
}

fn rewrite_stmt(stmt: &mut Stmt, values: &HashMap<SymbolId, Duration>) {
    let ops = |ops: &mut [QubitOperand<Expr>], values: &HashMap<SymbolId, Duration>| {
        for o in ops {
            if let QubitOperand::Indexed { indices, .. } = o {
                for i in indices {
                    rewrite_index(i, values);
                }
            }
        }
    };
    match &mut stmt.kind {
        StmtKind::Alias(a) => {
            for e in &mut a.value {
                rewrite_expr(e, values);
            }
        }
        StmtKind::GateCall(gc) => {
            for m in &mut gc.modifiers {
                if let crate::sir::GateModifier::Pow(e) = m {
                    rewrite_expr(e, values);
                }
            }
            for a in &mut gc.args {
                rewrite_expr(a, values);
            }
            ops(&mut gc.qubits, values);
            if let Some(d) = &mut gc.duration {
                rewrite_expr(d, values);
            }
        }
        StmtKind::Measure(m) => rewrite_measure(&mut m.measure, values),
        StmtKind::Reset(op) => ops(std::slice::from_mut(op), values),
        StmtKind::Barrier(o) | StmtKind::Nop(o) => ops(o, values),
        StmtKind::Delay(d) => {
            rewrite_expr(&mut d.duration, values);
            ops(&mut d.operands, values);
        }
        StmtKind::Box(b) => {
            if let Some(e) = &mut b.duration {
                rewrite_expr(e, values);
            }
            rewrite_stretch_vars(&mut b.body, values);
        }
        StmtKind::Assignment(a) => match &mut a.value {
            RValue::Expr(e) => rewrite_expr(e, values),
            RValue::Measure(m) => rewrite_measure(m, values),
        },
        StmtKind::Return(Some(RValue::Expr(e))) => rewrite_expr(e, values),
        StmtKind::Return(Some(RValue::Measure(m))) => rewrite_measure(m, values),
        StmtKind::ExprStmt(e) => rewrite_expr(e, values),
        // Straight-line-only: control flow errored during the walk; the
        // remaining kinds carry no rewritable expressions.
        _ => {}
    }
}

fn rewrite_measure(m: &mut crate::sir::MeasureExpr<Expr>, values: &HashMap<SymbolId, Duration>) {
    if let MeasureExprKind::QuantumCall { args, .. } = &mut m.kind {
        for a in args {
            rewrite_expr(a, values);
        }
    }
}

fn rewrite_index(op: &mut IndexOp<Expr>, values: &HashMap<SymbolId, Duration>) {
    match &mut op.kind {
        IndexKind::Set(exprs) => {
            for e in exprs {
                rewrite_expr(e, values);
            }
        }
        IndexKind::Items(items) => {
            for it in items {
                match it {
                    IndexItem::Single(e) => rewrite_expr(e, values),
                    IndexItem::Range(r) => {
                        for e in [&mut r.start, &mut r.step, &mut r.end]
                            .into_iter()
                            .flatten()
                        {
                            rewrite_expr(e, values);
                        }
                    }
                }
            }
        }
    }
}

fn rewrite_expr(expr: &mut Expr, values: &HashMap<SymbolId, Duration>) {
    if let ExprKind::Var(sid) = &expr.kind
        && let Some(d) = values.get(sid)
    {
        expr.kind = ExprKind::Literal(Primitive::from(*d));
        expr.ty = Type::Classical(ValueTy::duration());
        return;
    }
    match &mut expr.kind {
        ExprKind::Binary(b) => {
            rewrite_expr(&mut b.left, values);
            rewrite_expr(&mut b.right, values);
        }
        ExprKind::Unary(u) => rewrite_expr(&mut u.operand, values),
        ExprKind::Cast(c) => rewrite_expr(&mut c.operand, values),
        ExprKind::Index(ix) => {
            rewrite_expr(&mut ix.base, values);
            rewrite_index(&mut ix.index, values);
        }
        ExprKind::Call(c) => {
            for a in &mut c.args {
                rewrite_expr(a, values);
            }
        }
        ExprKind::ArrayLiteral(al) => {
            for e in &mut al.items {
                rewrite_expr(e, values);
            }
        }
        ExprKind::DurationOf(_)
        | ExprKind::Literal(_)
        | ExprKind::Var(_)
        | ExprKind::HardwareQubit(_) => {}
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{TableTimings, TimingDuration, resolve_durationof};
    use super::*;
    use crate::lower::compile_source;
    use crate::resolve::DefaultIncludeResolver;
    use crate::sir::BinOp;

    fn compile(source: &str) -> Program {
        compile_source(source, DefaultIncludeResolver, None).expect("compile ok")
    }

    fn table() -> TableTimings {
        TableTimings::new()
            .gate("cx", TimingDuration::ns(300.0))
            .gate("swap", TimingDuration::ns(200.0))
            .gate("U", TimingDuration::ns(60.0))
            .gate("x", TimingDuration::ns(20.0))
            .gate("y", TimingDuration::ns(20.0))
    }

    fn resolve(p: &mut Program, t: &TableTimings) -> Result<()> {
        resolve_durationof(p, t, &CompileOptions::default())
    }

    /// Evaluate a rewritten (stretch-free) constant expression to ns.
    /// Panics on an unresolved `Var` — which is itself an assertion that
    /// the rewrite reached everything.
    fn eval_ns(e: &Expr) -> f64 {
        match &e.kind {
            ExprKind::Literal(Primitive::Duration(d)) => d.to_unit(DurationUnit::Ns).value,
            ExprKind::Literal(Primitive::Int(i)) => *i as f64,
            ExprKind::Literal(Primitive::Uint(u)) => *u as f64,
            ExprKind::Literal(Primitive::Float(f)) => *f,
            ExprKind::Binary(b) => {
                let (l, r) = (eval_ns(&b.left), eval_ns(&b.right));
                match b.op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    other => panic!("unexpected op {other:?}"),
                }
            }
            ExprKind::Unary(u) => match u.op {
                crate::sir::UnOp::Neg => -eval_ns(&u.operand),
                other => panic!("unexpected unop {other:?}"),
            },
            ExprKind::Cast(c) => eval_ns(&c.operand),
            ExprKind::Var(_) => panic!("unresolved variable survived the rewrite"),
            other => panic!("unexpected expr {:?}", std::mem::discriminant(other)),
        }
    }

    /// All delay-duration expressions in program order, boxes included.
    fn delays(stmts: &[Stmt]) -> Vec<&Expr> {
        let mut out = Vec::new();
        for s in stmts {
            match &s.kind {
                StmtKind::Delay(d) => out.push(&d.duration),
                StmtKind::Box(b) => out.extend(delays(&b.body)),
                _ => {}
            }
        }
        out
    }

    fn delay_ns(p: &Program) -> Vec<f64> {
        delays(&p.body).into_iter().map(eval_ns).collect()
    }

    fn assignment_rhs_ns(p: &Program, name: &str) -> f64 {
        for s in &p.body {
            if let StmtKind::Assignment(a) = &s.kind
                && let LValue::Var(sid) = &a.target
                && p.symbols.get(*sid).name == name
                && let RValue::Expr(e) = &a.value
            {
                return eval_ns(e);
            }
        }
        panic!("no assignment to `{name}`");
    }

    fn first_box_duration_ns(p: &Program) -> f64 {
        for s in &p.body {
            if let StmtKind::Box(b) = &s.kind
                && let Some(e) = &b.duration
            {
                return eval_ns(e);
            }
        }
        panic!("no box with a designator");
    }

    fn error_msg(e: crate::error::CompileError) -> String {
        match e.kind {
            ErrorKind::InvalidContext(m) => m,
            other => panic!("expected InvalidContext, got {other:?}"),
        }
    }

    const APPROX: f64 = 1e-6;
    fn assert_ns(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < APPROX,
            "expected {expected}ns, got {actual}ns"
        );
    }

    #[test]
    fn alignment_left_pad() {
        // fixtures/qasm/alignment.qasm: barriers force
        // 3g + dur(U) == dur(cx)  =>  g = (300 - 60) / 3 = 80.
        let src = r#"
            include "stdgates.inc";
            stretch g;
            qubit[3] q;
            barrier q;
            cx q[0], q[1];
            delay[g] q[2];
            U(pi/4, 0, pi/2) q[2];
            delay[2*g] q[2];
            barrier q;
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        let d = delay_ns(&p);
        assert_ns(d[0], 80.0);
        assert_ns(d[1], 160.0);
    }

    #[test]
    fn left_justify_spec() {
        // Spec delays.rst:81-98 (third gate swapped for a distinct name):
        // each wire's stretch fills to the longest wire (300ns).
        let src = r#"
            include "stdgates.inc";
            qubit[5] q;
            barrier q;
            cx q[0], q[1];
            U(pi/4, 0, pi/2) q[2];
            swap q[3], q[4];
            stretch a;
            stretch b;
            stretch c;
            delay[a] q[0], q[1];
            delay[b] q[2];
            delay[c] q[3], q[4];
            barrier q;
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        let d = delay_ns(&p);
        assert_ns(d[0], 0.0);
        assert_ns(d[1], 240.0);
        assert_ns(d[2], 100.0);
    }

    #[test]
    fn dd_spec_box() {
        // The spec's DD example (delays.rst:247-266) in the fixture's boxed
        // form, with the spec-correct 0.5 coefficients. The box span is set
        // by the two sequential cx (600ns); wire $0 telescopes to exactly
        // 5a, so a = 120ns and the delays resolve to 110/100/110.
        let src = r#"
            include "stdgates.inc";
            stretch a;
            duration start_stretch = -0.5 * durationof({x $0;}) + a;
            duration middle_stretch = -0.5 * durationof({x $0;}) - 0.5 * durationof({y $0;}) + a;
            duration end_stretch = -0.5 * durationof({y $0;}) + a;
            box {
              delay[start_stretch] $0;
              x $0;
              delay[middle_stretch] $0;
              y $0;
              delay[middle_stretch] $0;
              x $0;
              delay[middle_stretch] $0;
              y $0;
              delay[end_stretch] $0;

              cx $2, $3;
              cx $1, $2;
              x $3;
            }
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        assert_ns(assignment_rhs_ns(&p, "start_stretch"), 110.0);
        assert_ns(assignment_rhs_ns(&p, "middle_stretch"), 100.0);
        assert_ns(assignment_rhs_ns(&p, "end_stretch"), 110.0);
    }

    #[test]
    fn box_concrete_designator() {
        // Spec delays.rst:312-329: the stretch absorbs the box's slack.
        let src = r#"
            include "stdgates.inc";
            stretch str1;
            qubit q;
            box[150ns] {
                delay[str1] q;
                x q;
            }
        "#;
        let mut p = compile(src);
        let t = TableTimings::new().gate("x", TimingDuration::ns(50.0));
        resolve(&mut p, &t).unwrap();
        assert_ns(delay_ns(&p)[0], 100.0);
    }

    #[test]
    fn box_stretch_designator_nop() {
        // Spec delays.rst:332-352: box[st] is pinned by its contents; the
        // nop wire is synchronized without operations.
        let src = r#"
            include "stdgates.inc";
            stretch st;
            box[st] {
                cx $0, $1;
                nop $2;
            }
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        assert_ns(first_box_duration_ns(&p), 300.0);
    }

    #[test]
    fn over_constrained_errors() {
        let src = r#"
            include "stdgates.inc";
            stretch st;
            box[100ns] { delay[st] $0; }
            box[200ns] { delay[st] $1; }
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("over-constrained"), "{msg}");
        assert!(msg.contains('s'), "{msg}");
    }

    #[test]
    fn negative_resolution_errors() {
        // A negative coefficient can force a negative stretch: the box
        // demands 150ns but `d = 100ns - s` only shrinks as s grows.
        let src = r#"
            include "stdgates.inc";
            stretch st;
            duration d = 100ns - 1 * st;
            qubit q;
            box[150ns] { delay[d] q; }
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("negative"), "{msg}");
        assert!(msg.contains('s'), "{msg}");
    }

    #[test]
    fn box_contents_exceed_errors() {
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit[2] q;
            box[100ns] {
                x q[0];
                delay[st] q[1];
            }
        "#;
        let mut p = compile(src);
        let t = TableTimings::new().gate("x", TimingDuration::ns(130.0));
        let msg = error_msg(resolve(&mut p, &t).unwrap_err());
        assert!(msg.contains("exceed"), "{msg}");
    }

    #[test]
    fn derived_stretch_equal_split() {
        // Underdetermined slack splits equally per unit coefficient:
        // a + 2c == 300  =>  a = c = 100, d = 300.
        let src = r#"
            include "stdgates.inc";
            stretch a;
            stretch c;
            stretch d = a + 2 * c;
            qubit[3] q;
            cx q[0], q[1];
            delay[d] q[2];
            barrier q;
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        assert_ns(assignment_rhs_ns(&p, "d"), 300.0);
        assert_ns(delay_ns(&p)[0], 300.0);
    }

    /// All gate-call designator expressions in program order, boxes included.
    fn gate_designators(stmts: &[Stmt]) -> Vec<&Expr> {
        let mut out = Vec::new();
        for s in stmts {
            match &s.kind {
                StmtKind::GateCall(gc) => out.extend(gc.duration.as_ref()),
                StmtKind::Box(b) => out.extend(gate_designators(&b.body)),
                _ => {}
            }
        }
        out
    }

    #[test]
    fn stretch_gate_designator_spectator() {
        // The spec's rotary idiom (delays.rst:212-222): a stretchy gate on a
        // spectator wire fills the span of the cx on the other wires.
        let src = r#"
            include "stdgates.inc";
            stretch a;
            qubit[3] q;
            barrier q;
            cx q[0], q[1];
            x[a] q[2];
            barrier q;
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        let d = gate_designators(&p.body);
        assert_eq!(d.len(), 1);
        assert_ns(eval_ns(d[0]), 300.0);
    }

    #[test]
    fn designator_overrides_table() {
        // The designator wins over the timing table (x would be 20ns).
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit q;
            box[250ns] {
                x[200ns] q;
                delay[st] q;
            }
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        assert_ns(delay_ns(&p)[0], 50.0);
    }

    #[test]
    fn designator_only_stretch_use_detected() {
        // A stretch referenced ONLY through a gate designator must still be
        // detected and solved.
        let src = r#"
            include "stdgates.inc";
            stretch a;
            qubit q;
            box[100ns] {
                x[a] q;
            }
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        let d = gate_designators(&p.body);
        assert_ns(eval_ns(d[0]), 100.0);
    }

    #[test]
    fn stretchy_designator_negative_after_solve() {
        // `st` is pinned to 100ns by the box; the designator then resolves
        // to -100ns and the post-solve check rejects it.
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit q;
            box[100ns] { delay[st] q; }
            x[100ns - 2 * st] q;
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("negative"), "{msg}");
    }

    #[test]
    fn unconstrained_resolves_zero() {
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit q;
            delay[st] q;
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        assert_ns(delay_ns(&p)[0], 0.0);
    }

    #[test]
    fn pinned_then_reused() {
        // Greedy program-order solving: `s` is pinned by the first box and
        // concrete for the later delay.
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit[2] q;
            box[100ns] { delay[st] q[0]; }
            delay[st] q[1];
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        let d = delay_ns(&p);
        assert_ns(d[0], 100.0);
        assert_ns(d[1], 100.0);
    }

    #[test]
    fn multiqubit_delay_pins_greedily() {
        // Documents the greedy rule: the multi-qubit delay is a sync point
        // that pins `a` to the minimal feasible value at that instant.
        let src = r#"
            include "stdgates.inc";
            stretch a;
            qubit[2] q;
            delay[a] q[0];
            delay[100ns] q[0], q[1];
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
        let d = delay_ns(&p);
        assert_ns(d[0], 0.0);
        assert_ns(d[1], 100.0);
    }

    #[test]
    fn control_flow_errors() {
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit q;
            delay[st] q;
            int v = 0;
            if (v == 1) { x q; }
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("straight-line"), "{msg}");
    }

    #[test]
    fn bare_stretchy_delay_errors() {
        let src = r#"
            include "stdgates.inc";
            stretch g;
            qubit q;
            delay[g];
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("explicit qubit operands"), "{msg}");
    }

    #[test]
    fn unused_stretch_decl_untouched() {
        // An unused stretch declaration must not trip the straight-line
        // rule for the rest of the program.
        let src = r#"
            include "stdgates.inc";
            stretch st;
            qubit q;
            int v = 0;
            if (v == 1) { x q; }
            bit c = measure q;
        "#;
        let mut p = compile(src);
        resolve(&mut p, &table()).unwrap();
    }

    #[test]
    fn stretch_in_def_body_errors() {
        let src = r#"
            include "stdgates.inc";
            def f(qubit a) {
                stretch s2;
                delay[s2] a;
            }
            qubit q;
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("subroutine bodies"), "{msg}");
    }

    #[test]
    fn stretch_reassignment_errors() {
        let src = r#"
            include "stdgates.inc";
            stretch a;
            stretch d = a;
            qubit q;
            d = a;
            delay[d] q;
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("more than once"), "{msg}");
    }

    #[test]
    fn stretch_param_errors() {
        let src = r#"
            include "stdgates.inc";
            def f(stretch s, qubit a) { x a; }
            qubit q;
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("top-level variable"), "{msg}");
    }

    #[test]
    fn input_stretch_errors() {
        let src = r#"
            include "stdgates.inc";
            input stretch st;
            qubit q;
        "#;
        let mut p = compile(src);
        let msg = error_msg(resolve(&mut p, &table()).unwrap_err());
        assert!(msg.contains("top-level variable"), "{msg}");
    }
}
