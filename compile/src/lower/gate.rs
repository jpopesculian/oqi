//! Gate-body lowering.
//!
//! A gate body is a restricted statement list: only `GateCall` and
//! `Barrier` are allowed, `for` loops are fully unrolled (with the loop
//! variable bound as a `const` so index/angle expressions fold to
//! concrete values), and all other control flow is rejected. This keeps
//! the general statement lowering in [`super`] free of the gate-only
//! rules.

use oqi_parse::ast;

use crate::classical::{PrimitiveTy, Value, iw};
use crate::error::{CompileError, ErrorKind, Result};
use crate::scope::ScopeKind;
use crate::sir;
use crate::symbol::SymbolKind;
use crate::types::{Type, eval_const_expr, resolve_scalar_type};

use super::Lowerer;

impl Lowerer {
    pub(super) fn lower_gate_body(&mut self, scope: &ast::Scope<'_>) -> Result<sir::GateBody> {
        let mut stmts = Vec::new();
        for item in &scope.body {
            self.lower_gate_item(item, &mut stmts)?;
        }
        Ok(sir::GateBody { body: stmts })
    }

    /// Lower one gate-body item, unrolling `for` loops and rejecting
    /// branching. Appends only `GateCall`/`Barrier` statements to `out`.
    fn lower_gate_item(
        &mut self,
        item: &ast::StmtOrScope<'_>,
        out: &mut Vec<sir::Stmt>,
    ) -> Result<()> {
        match item {
            ast::StmtOrScope::Scope(scope) => {
                self.with_scope(ScopeKind::Anonymous, scope.span, |this| {
                    for it in &scope.body {
                        this.lower_gate_item(it, out)?;
                    }
                    Ok(())
                })
            }
            ast::StmtOrScope::Stmt(stmt) => self.lower_gate_stmt(stmt, out),
        }
    }

    fn lower_gate_stmt(&mut self, stmt: &ast::Stmt<'_>, out: &mut Vec<sir::Stmt>) -> Result<()> {
        match &stmt.kind {
            ast::StmtKind::For {
                ty,
                var,
                iterable,
                body,
            } => self.unroll_gate_for(ty, var, iterable, body, out),
            ast::StmtKind::If { .. }
            | ast::StmtKind::While { .. }
            | ast::StmtKind::Switch { .. } => Err(CompileError::new(ErrorKind::InvalidGateBody(
                "branching is not allowed in a gate body".into(),
            ))
            .with_span(stmt.span)),
            _ => {
                let lowered = self.lower_stmt(stmt)?;
                for s in &lowered {
                    validate_gate_stmt(s)?;
                }
                out.extend(lowered);
                Ok(())
            }
        }
    }

    /// Fully unroll a `for` loop in a gate body. The loop variable is bound
    /// as a `const` of its declared value, so index/angle expressions in the
    /// body (`q[i+1]`, `ry(theta * i)`) resolve to concrete values.
    fn unroll_gate_for(
        &mut self,
        ty: &ast::ScalarType<'_>,
        var: &ast::Ident<'_>,
        iterable: &ast::ForIterable<'_>,
        body: &ast::StmtOrScope<'_>,
        out: &mut Vec<sir::Stmt>,
    ) -> Result<()> {
        let var_ty = resolve_scalar_type(ty, self.resolver.symbols(), self.resolver.options())?;
        let values = self.gate_for_values(iterable)?;
        let body_span = match body {
            ast::StmtOrScope::Stmt(s) => s.span,
            ast::StmtOrScope::Scope(sc) => sc.span,
        };
        for v in values {
            let cv = gate_loop_value(&var_ty, v);
            self.with_scope(ScopeKind::For, body_span, |this| {
                let var_sym =
                    this.resolver
                        .declare(var.name, SymbolKind::Const, var_ty.clone(), var.span)?;
                this.resolver.symbols_mut().set_const_value(var_sym, cv);
                this.lower_gate_item(body, out)
            })?;
        }
        Ok(())
    }

    /// Evaluate the iterable of a gate-body `for` loop to a concrete sequence
    /// of integers. Range semantics mirror the CFG builder: `start` defaults
    /// to 0, `step` to 1, and `end` is inclusive.
    fn gate_for_values(&self, it: &ast::ForIterable<'_>) -> Result<Vec<i128>> {
        match it {
            ast::ForIterable::Range(range, span) => {
                let start = match &range.start {
                    Some(e) => self.gate_const_int(e)?,
                    None => 0,
                };
                let step = match &range.step {
                    Some(e) => self.gate_const_int(e)?,
                    None => 1,
                };
                let end = match &range.end {
                    Some(e) => self.gate_const_int(e)?,
                    None => {
                        return Err(CompileError::new(ErrorKind::InvalidGateBody(
                            "range for-loop in a gate body requires an explicit end".into(),
                        ))
                        .with_span(*span));
                    }
                };
                if step == 0 {
                    return Err(CompileError::new(ErrorKind::InvalidGateBody(
                        "for-loop range step must be non-zero".into(),
                    ))
                    .with_span(*span));
                }
                let mut values = Vec::new();
                let mut i = start;
                if step > 0 {
                    while i <= end {
                        values.push(i);
                        i += step;
                    }
                } else {
                    while i >= end {
                        values.push(i);
                        i += step;
                    }
                }
                Ok(values)
            }
            ast::ForIterable::Set(exprs, _) => {
                exprs.iter().map(|e| self.gate_const_int(e)).collect()
            }
            ast::ForIterable::Expr(e) => Err(CompileError::new(ErrorKind::InvalidGateBody(
                "iterating a general expression in a gate body is not supported".into(),
            ))
            .with_span(e.span())),
        }
    }

    /// Evaluate `e` to a compile-time constant integer, for use as a
    /// gate-body loop bound. Errors if `e` is not a constant integer.
    fn gate_const_int(&self, e: &ast::Expr<'_>) -> Result<i128> {
        let err = || {
            CompileError::new(ErrorKind::InvalidGateBody(
                "loop bounds in a gate body must be compile-time constant integers".into(),
            ))
            .with_span(e.span())
        };
        let Value::Scalar(s) = eval_const_expr(e, self.resolver.symbols(), self.resolver.options())
            .map_err(|_| err())?
        else {
            return Err(err());
        };
        s.cast(PrimitiveTy::Int(iw(128)))
            .ok()
            .and_then(|s| s.value().as_int(iw(128)))
            .ok_or_else(err)
    }
}

/// The loop variable's `const` value for one unrolled iteration: the
/// integer `v` cast to the loop variable's declared type.
fn gate_loop_value(var_ty: &Type, v: i128) -> Value {
    let base = Value::int(v, iw(128));
    match var_ty.value_ty() {
        Some(vt) => base.clone().cast(vt).unwrap_or(base),
        None => base,
    }
}

/// Reject any lowered statement that isn't legal in a gate body.
fn validate_gate_stmt(stmt: &sir::Stmt) -> Result<()> {
    match &stmt.kind {
        sir::StmtKind::GateCall(_) | sir::StmtKind::Barrier(_) => Ok(()),
        _ => Err(CompileError::new(ErrorKind::InvalidGateBody(
            "statement not allowed in gate body".into(),
        ))
        .with_span(stmt.span)),
    }
}
