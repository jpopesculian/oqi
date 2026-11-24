use std::path::{Path, PathBuf};

use oqi_classical::{Index as ClassicalIndex, ops};
use oqi_parse::ast;

use crate::cfg;
use crate::duration::{TableTimings, resolve_durationof};
use crate::error::{CompileError, ErrorKind, Result, ResultExt};
use crate::openpulse;
use crate::qubits::{self, Resolved};
use crate::resolve::{IncludeResolver, IncludeSource, Resolver};
use crate::sir;
use crate::symbol::{SymbolId, SymbolKind};
use crate::types::{
    CompileOptions, Type, UnsizedType, bin_op, eval_const_expr, eval_designator, invalid_type,
    parse_bitstring_literal, parse_float_literal, parse_imag_literal, parse_int_literal,
    parse_timing_literal, resolve_array_ref_type, resolve_old_style_type, resolve_qubit_type,
    resolve_scalar_type, resolve_type, un_op,
};
use crate::{
    classical::{
        ArrayRefShape, ArrayShape, ArrayTy, Primitive, PrimitiveTy, Scalar, Value, ValueTy, iw,
    },
    types::SystemWidth,
};

mod gate;

// ── Public API ──────────────────────────────────────────────────────────

/// Source → SIR: parse, resolve includes, type-check and lower.
///
/// With `timings`, `durationof` and `stretch` are resolved at compile time
/// against that table plus the program's own defcal and gate bodies;
/// without, they are left for the VM's runtime pass.
pub(crate) fn compile_to_sir(
    source: &str,
    include_resolver: &dyn IncludeResolver,
    options: &CompileOptions,
    timings: Option<&TableTimings>,
) -> Result<sir::Program> {
    let ast = oqi_parse::parse(source)
        .map_err(|e| CompileError::from(e).or_path(options.source_name.clone()))?;
    let mut lowerer = Lowerer::new(include_resolver, options.clone());
    lowerer.lower_program(&ast)?;
    let mut program = lowerer.finish(&ast);
    if let Some(table) = timings {
        let table = table
            .clone()
            .with_defcals(&program, &options.dt)
            .with_program_gates(&program);
        resolve_durationof(&mut program, &table, options)?;
    }
    Ok(program)
}

// ── Lowerer ─────────────────────────────────────────────────────────────

struct Lowerer<'r> {
    resolver: Resolver<'r>,
    gates: Vec<sir::GateDecl>,
    subroutines: Vec<sir::SubroutineDecl>,
    externs: Vec<sir::ExternDecl>,
    calibrations: Vec<sir::CalibrationDecl>,
    calibration_grammar: Option<String>,
    body: Vec<sir::Stmt>,
}

impl<'r> Lowerer<'r> {
    fn new(include_resolver: &'r dyn IncludeResolver, options: CompileOptions) -> Self {
        Self {
            resolver: Resolver::new(include_resolver, options),
            gates: Vec::new(),
            subroutines: Vec::new(),
            externs: Vec::new(),
            calibrations: Vec::new(),
            calibration_grammar: None,
            body: Vec::new(),
        }
    }

    fn finish(self, program: &ast::Program<'_>) -> sir::Program {
        sir::Program {
            version: program.version.as_ref().map(|v| v.specifier.to_string()),
            calibration_grammar: self.calibration_grammar,
            symbols: self.resolver.into_symbols(),
            gates: self.gates,
            subroutines: self.subroutines,
            externs: self.externs,
            calibrations: self.calibrations,
            body: self.body,
        }
    }

    fn lower_program(&mut self, program: &ast::Program<'_>) -> Result<()> {
        // `OPENQASM M.m;` (docs/comments.rst): this compiler speaks major
        // version 3, any minor version.
        if let Some(version) = &program.version {
            let major = version.specifier.split('.').next().unwrap_or_default();
            if major != "3" {
                let message = format!(
                    "OpenQASM version {} (only version 3 is supported)",
                    version.specifier
                );
                return Err(CompileError::at(
                    ErrorKind::Unsupported(message),
                    version.span,
                ));
            }
        }
        self.body = self.lower_items(&program.body)?;
        Ok(())
    }

    fn with_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.resolver.push_scope();
        let result = f(self);
        self.resolver.pop_scope();
        result
    }

    /// Lower the parameters and body of a gate, subroutine or defcal
    /// (docs/scope.rst, "Subroutine and gate scope"): a scope of its own,
    /// which of the global scope sees only the constants and definitions.
    fn with_definition<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.resolver.set_in_definition(true);
        let result = self.with_scope(f);
        self.resolver.set_in_definition(false);
        result
    }

    /// Lower a `cal` block or a `defcal` (docs/pulses.rst, "Inline
    /// calibration blocks"): it sees the values `cal` blocks declare, which
    /// the rest of the program does not.
    fn with_cal_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let outer = self.resolver.enter_cal();
        let result = f(self);
        self.resolver.leave_cal(outer);
        result
    }

    fn with_include<T>(
        &mut self,
        path: PathBuf,
        f: impl FnOnce(&mut Self, &Path) -> Result<T>,
    ) -> Result<T> {
        self.resolver.push_include(path.clone())?;
        let result = f(self, &path);
        self.resolver.pop_include();
        result
    }

    // ── Statement lowering ──────────────────────────────────────────────

    fn lower_stmt_or_scope(&mut self, item: &ast::StmtOrScope<'_>) -> Result<Vec<sir::Stmt>> {
        let current_path = self.resolver.current_source_path().map(Path::to_path_buf);
        let result = match item {
            ast::StmtOrScope::Stmt(stmt) => self.lower_stmt(stmt),
            ast::StmtOrScope::Scope(scope) => self.with_scope(|this| this.lower_items(&scope.body)),
        };
        result.or_path(current_path)
    }

    fn lower_items(&mut self, items: &[ast::StmtOrScope<'_>]) -> Result<Vec<sir::Stmt>> {
        let mut stmts = Vec::new();
        for item in items {
            stmts.extend(self.lower_stmt_or_scope(item)?);
        }
        Ok(stmts)
    }

    /// Lower the body of a control-flow statement, in the current scope.
    fn lower_body(&mut self, body: &ast::StmtOrScope<'_>) -> Result<Vec<sir::Stmt>> {
        match body {
            ast::StmtOrScope::Stmt(stmt) => self.lower_stmt(stmt),
            ast::StmtOrScope::Scope(scope) => self.lower_items(&scope.body),
        }
    }

    /// Reject a declaration the spec only allows in the global scope
    /// (docs/scope.rst).
    fn require_global(&self, what: &'static str, span: oqi_lex::Span) -> Result<()> {
        match self.resolver.current_scope() {
            None => Ok(()),
            Some(_) => Err(CompileError::at(ErrorKind::NotGlobalScope(what), span)),
        }
    }

    /// The type of a classical declaration. "All arrays must be declared
    /// within the global scope of the program" (docs/types.rst).
    fn resolve_decl_type(&self, ty: &ast::TypeExpr<'_>, span: oqi_lex::Span) -> Result<Type> {
        let ty = resolve_type(ty, &self.resolver)?;
        if ty.array_ty().is_some() {
            self.require_global("array declarations", span)?;
        }
        Ok(ty)
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt<'_>) -> Result<Vec<sir::Stmt>> {
        let annotations = self.lower_annotations(&stmt.annotations);
        let span = stmt.span;
        // A statement lowers to one SIR statement, with two exceptions: a
        // declaration without an initializer lowers to none (the symbol
        // table carries it), and an `include` to those of the file.
        let none = Ok(Vec::new());
        // A statement that has to run before the one lowered to.
        let mut before = None;

        let kind = match &stmt.kind {
            ast::StmtKind::Include(path) => {
                self.require_global("`include` statements", span)?;
                return self.lower_include(path.value, span);
            }

            ast::StmtKind::ClassicalDecl { ty, name, init } => {
                let ty = self.resolve_decl_type(ty, span)?;
                let symbol =
                    self.resolver
                        .declare(name.name, SymbolKind::Variable, ty, name.span)?;
                match self.lower_init(symbol, init.as_ref(), span)? {
                    Some(kind) => kind,
                    None => return none,
                }
            }

            ast::StmtKind::ConstDecl { ty, name, init } => {
                let ty = self.resolve_decl_type(ty, span)?;
                let e = match init {
                    ast::ExprOrMeasure::Measure(_) => {
                        let message = "const initializer cannot be a measurement";
                        return Err(CompileError::at(
                            ErrorKind::InvalidContext(message.into()),
                            span,
                        ));
                    }
                    ast::ExprOrMeasure::Expr(e) => e,
                };
                let const_val = match e {
                    ast::Expr::ArrayLiteral(_) => None,
                    e => eval_const_expr(e, &self.resolver).ok(),
                };
                let symbol = self
                    .resolver
                    .declare(name.name, SymbolKind::Const, ty, name.span)?;
                if let Some(cv) = const_val {
                    self.resolver.symbols_mut().set_const_value(symbol, cv);
                }
                return none;
            }

            ast::StmtKind::QuantumDecl { ty, name } => {
                self.require_global("qubit declarations", span)?;
                let ty = resolve_qubit_type(ty, &self.resolver)?;
                self.resolver
                    .declare(name.name, SymbolKind::Qubit, ty, name.span)?;
                return none;
            }

            ast::StmtKind::OldStyleDecl {
                keyword,
                name,
                designator,
            } => {
                let kind = match keyword {
                    ast::OldStyleKind::Creg => SymbolKind::Variable,
                    ast::OldStyleKind::Qreg => {
                        self.require_global("qubit declarations", span)?;
                        SymbolKind::Qubit
                    }
                };
                let ty = resolve_old_style_type(keyword, designator.as_deref(), &self.resolver)?;
                self.resolver.declare(name.name, kind, ty, name.span)?;
                return none;
            }

            ast::StmtKind::IoDecl { dir, ty, name } => {
                // The spec does not say where these may appear; they are the
                // program's interface, so they are kept to the global scope.
                self.require_global("`input` and `output` declarations", span)?;
                let ty = resolve_type(ty, &self.resolver)?;
                let kind = match dir {
                    ast::IoDir::Input => SymbolKind::Input,
                    ast::IoDir::Output => SymbolKind::Output,
                };
                self.resolver.declare(name.name, kind, ty, name.span)?;
                return none;
            }

            ast::StmtKind::Gate {
                name,
                params,
                qubits,
                body,
            } => {
                self.require_global("`gate` definitions", span)?;
                let symbol =
                    self.resolver
                        .declare(name.name, SymbolKind::Gate, Type::Void, name.span)?;
                let (params, qubits, body) = self.with_definition(|this| {
                    let angle =
                        Type::Classical(ValueTy::angle(this.resolver.options().system_width.iw()));
                    let params: Vec<_> = params
                        .iter()
                        .map(|p| {
                            let kind = SymbolKind::GateParam;
                            this.resolver.declare(p.name, kind, angle.clone(), p.span)
                        })
                        .collect::<Result<_>>()?;
                    let qubits: Vec<_> = qubits
                        .iter()
                        .map(|q| {
                            let kind = SymbolKind::GateQubit;
                            this.resolver.declare(q.name, kind, Type::Qubit, q.span)
                        })
                        .collect::<Result<_>>()?;
                    Ok((params, qubits, this.lower_gate_body(body)?))
                })?;
                self.gates.push(sir::GateDecl {
                    symbol,
                    params,
                    qubits,
                    body,
                    span,
                });
                return none;
            }

            ast::StmtKind::Def {
                name,
                params,
                return_ty,
                body,
            } => {
                self.require_global("subroutine definitions", span)?;
                let symbol = self.resolver.declare(
                    name.name,
                    SymbolKind::Subroutine,
                    Type::Void,
                    name.span,
                )?;
                let (params, return_ty, body) = self.with_definition(|this| {
                    let params = params
                        .iter()
                        .map(|p| this.lower_arg_def(p))
                        .collect::<Result<_>>()?;
                    let return_ty = return_ty
                        .as_ref()
                        .map(|ty| resolve_scalar_type(ty, &this.resolver))
                        .transpose()?;
                    if let Some(ty) = &return_ty {
                        this.resolver.symbols_mut().get_mut(symbol).ty = ty.clone();
                    }
                    Ok((params, return_ty, this.lower_items(&body.body)?))
                })?;
                self.subroutines.push(sir::SubroutineDecl {
                    symbol,
                    params,
                    return_ty,
                    body,
                    span,
                });
                return none;
            }

            ast::StmtKind::Extern {
                name,
                params,
                return_ty,
            } => {
                let symbol =
                    self.resolver
                        .declare(name.name, SymbolKind::Extern, Type::Void, name.span)?;
                let param_types = self.lower_extern_args(params)?;
                let return_ty = return_ty
                    .as_ref()
                    .map(|ty| resolve_scalar_type(ty, &self.resolver))
                    .transpose()?;
                if let Some(ty) = &return_ty {
                    self.resolver.symbols_mut().get_mut(symbol).ty = ty.clone();
                }
                self.externs.push(sir::ExternDecl {
                    symbol,
                    param_types,
                    return_ty,
                    span,
                });
                return none;
            }

            ast::StmtKind::Defcal {
                target,
                args,
                operands,
                return_ty,
                body,
            } => {
                self.require_global("`defcal` definitions", span)?;
                let target = match target {
                    ast::DefcalTarget::Measure(_) => sir::CalibrationTarget::Measure,
                    ast::DefcalTarget::Reset(_) => sir::CalibrationTarget::Reset,
                    ast::DefcalTarget::Delay(_) => sir::CalibrationTarget::Delay,
                    ast::DefcalTarget::Ident(id) => {
                        sir::CalibrationTarget::Named(self.defcal_gate(id)?)
                    }
                };
                let lower = |this: &mut Self| {
                    let args: Vec<_> = args
                        .iter()
                        .map(|a| match a {
                            ast::DefcalArgDef::Expr(e) => {
                                Ok(sir::CalibrationArg::Expr(Box::new(this.lower_expr(e)?)))
                            }
                            ast::DefcalArgDef::ArgDef(ad) => {
                                Ok(sir::CalibrationArg::Param(this.lower_arg_def(ad)?))
                            }
                        })
                        .collect::<Result<_>>()?;
                    let operands: Vec<_> = operands
                        .iter()
                        .map(|o| match o {
                            ast::DefcalOperand::HardwareQubit(s, span) => Ok(
                                sir::CalibrationOperand::Hardware(parse_hardware_qubit(s, *span)?),
                            ),
                            ast::DefcalOperand::Ident(id) => {
                                let kind = SymbolKind::GateQubit;
                                let symbol =
                                    this.resolver.declare(id.name, kind, Type::Qubit, id.span)?;
                                Ok(sir::CalibrationOperand::Ident(symbol))
                            }
                        })
                        .collect::<Result<_>>()?;
                    let return_ty = return_ty
                        .as_ref()
                        .map(|ty| resolve_scalar_type(ty, &this.resolver))
                        .transpose()?;
                    Ok((args, operands, return_ty, this.lower_cal_body(body)?))
                };
                let (args, operands, return_ty, body) =
                    self.with_cal_scope(|this| this.with_definition(lower))?;
                self.calibrations.push(sir::CalibrationDecl {
                    target,
                    args,
                    operands,
                    return_ty,
                    body,
                    span,
                });
                return none;
            }

            ast::StmtKind::Cal(body) => {
                sir::StmtKind::Cal(self.with_cal_scope(|this| this.lower_cal_body(body))?)
            }

            ast::StmtKind::CalibrationGrammar(grammar) => {
                let name = grammar.value;
                if name == "openpulse" {
                    self.seed_openpulse_intrinsics()?;
                }
                self.calibration_grammar = Some(name.to_string());
                return none;
            }

            ast::StmtKind::ExternFrame { name } => {
                let ty = self.frame_type();
                self.resolver
                    .declare(name.name, SymbolKind::ExternFrame, ty, name.span)?;
                return none;
            }

            ast::StmtKind::ExternPort { name } => {
                let ty = Self::port_type();
                self.resolver
                    .declare(name.name, SymbolKind::ExternPort, ty, name.span)?;
                return none;
            }

            ast::StmtKind::GateCall {
                modifiers,
                name,
                args,
                designator,
                operands,
            } => {
                let gate = self.resolver.resolve(name.name, name.span)?;
                let modifiers = modifiers
                    .iter()
                    .map(|m| self.lower_gate_modifier(m))
                    .collect::<Result<_>>()?;
                let args = args
                    .iter()
                    .flatten()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                let qubits = self.lower_gate_operands(operands)?;
                let duration = designator
                    .as_deref()
                    .map(|e| self.lower_expr(e))
                    .transpose()?;
                sir::StmtKind::GateCall(sir::GateCall {
                    gate,
                    modifiers,
                    args,
                    qubits,
                    duration,
                })
            }

            ast::StmtKind::MeasureArrow { measure, target } => {
                let measure = self.lower_measure_expr(measure)?;
                let target = target
                    .as_ref()
                    .map(|id| self.lower_indexed_ident_to_lvalue(id))
                    .transpose()?;
                sir::StmtKind::Measure(sir::Measure { measure, target })
            }

            ast::StmtKind::Reset(operand) => {
                sir::StmtKind::Reset(self.lower_gate_operand(operand)?)
            }

            ast::StmtKind::Barrier(operands) => {
                sir::StmtKind::Barrier(self.lower_gate_operands(operands)?)
            }

            ast::StmtKind::Assignment { target, op, value } => {
                let lv = self.lower_indexed_ident_to_lvalue(target)?;
                let value = match (compound_to_bin_op(op), value) {
                    // `target = measure q;`
                    (None, ast::ExprOrMeasure::Measure(m)) => {
                        sir::RValue::Measure(Box::new(self.lower_measure_expr(m)?))
                    }
                    (None, ast::ExprOrMeasure::Expr(e)) => {
                        let rhs = self.lower_rhs(e)?;
                        let ty = self.lvalue_type(&lv)?;
                        sir::RValue::Expr(Box::new(coerce_assign(rhs, &ty, span)?))
                    }
                    // `a op= b` is `a = a op b`. A measurement is taken into
                    // a temporary first: `$N = measure q; a = a op $N`.
                    (Some(bin_op), value) => {
                        let right = match value {
                            ast::ExprOrMeasure::Expr(e) => self.lower_rhs(e)?,
                            ast::ExprOrMeasure::Measure(m) => {
                                let measure = self.lower_measure_expr(m)?;
                                let ty = measure.ty.clone();
                                let scope = self.resolver.current_scope();
                                let temp =
                                    self.resolver
                                        .symbols_mut()
                                        .new_temp(ty.clone(), span, scope);
                                before = Some(sir::StmtKind::Measure(sir::Measure {
                                    measure,
                                    target: Some(sir::LValue::Var(temp)),
                                }));
                                sir::Expr {
                                    kind: sir::ExprKind::Var(temp),
                                    ty,
                                    span,
                                }
                            }
                        };
                        let left = self.lower_indexed_ident_to_expr(target)?;
                        sir::RValue::Expr(Box::new(sir::Expr {
                            ty: left.ty.clone(),
                            kind: sir::ExprKind::Binary(sir::Binary {
                                op: bin_op,
                                left: Box::new(left),
                                right: Box::new(right),
                            }),
                            span,
                        }))
                    }
                };
                sir::StmtKind::Assignment(sir::Assignment { target: lv, value })
            }

            ast::StmtKind::If {
                condition,
                then_body,
                else_body,
            } => {
                let condition = self.lower_expr(condition)?;
                // The two branches are different scopes.
                let then_body = self.with_scope(|this| this.lower_body(then_body))?;
                let else_body = else_body
                    .as_ref()
                    .map(|body| self.with_scope(|this| this.lower_body(body)))
                    .transpose()?;
                sir::StmtKind::If(sir::If {
                    condition,
                    then_body,
                    else_body,
                })
            }

            ast::StmtKind::For {
                ty,
                var,
                iterable,
                body,
            } => {
                let var_ty = resolve_scalar_type(ty, &self.resolver)?;
                let iterable = self.lower_for_iterable(iterable, &var_ty)?;
                // The loop variable is scoped to the body.
                let (var, body) = self.with_scope(|this| {
                    let kind = SymbolKind::LoopVar;
                    let var = this.resolver.declare(var.name, kind, var_ty, var.span)?;
                    Ok((var, this.lower_body(body)?))
                })?;
                sir::StmtKind::For(sir::For {
                    var,
                    iterable,
                    body,
                })
            }

            ast::StmtKind::While { condition, body } => {
                let condition = self.lower_expr(condition)?;
                let body = self.with_scope(|this| this.lower_body(body))?;
                sir::StmtKind::While(sir::While { condition, body })
            }

            ast::StmtKind::Switch { target, cases } => {
                let target = self.lower_expr(target)?;
                let cases = cases
                    .iter()
                    .map(|c| self.lower_switch_case(c))
                    .collect::<Result<_>>()?;
                sir::StmtKind::Switch(sir::Switch { target, cases })
            }

            ast::StmtKind::Break => sir::StmtKind::Break,
            ast::StmtKind::Continue => sir::StmtKind::Continue,
            ast::StmtKind::End => sir::StmtKind::End,

            ast::StmtKind::Return(value) => sir::StmtKind::Return(match value {
                Some(ast::ExprOrMeasure::Expr(e)) => {
                    Some(sir::RValue::Expr(Box::new(self.lower_expr(e)?)))
                }
                Some(ast::ExprOrMeasure::Measure(m)) => {
                    Some(sir::RValue::Measure(Box::new(self.lower_measure_expr(m)?)))
                }
                None => None,
            }),

            ast::StmtKind::Alias { name, value } => {
                let value: Vec<sir::Expr> = value
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                let ty = self.alias_type(&value)?;
                let symbol = self
                    .resolver
                    .declare(name.name, SymbolKind::Alias, ty, name.span)?;
                sir::StmtKind::Alias(sir::Alias { symbol, value })
            }

            ast::StmtKind::Delay {
                designator,
                operands,
            } => sir::StmtKind::Delay(sir::Delay {
                duration: self.lower_expr(designator)?,
                operands: self.lower_gate_operands(operands)?,
            }),

            ast::StmtKind::Box { designator, body } => {
                let duration = designator
                    .as_ref()
                    .map(|e| self.lower_expr(e))
                    .transpose()?;
                let body = self.with_scope(|this| this.lower_items(&body.body))?;
                sir::StmtKind::Box(sir::BoxStmt { duration, body })
            }

            ast::StmtKind::Nop(operands) => sir::StmtKind::Nop(self.lower_gate_operands(operands)?),

            ast::StmtKind::Pragma(content) => sir::StmtKind::Pragma(content.to_string()),

            ast::StmtKind::Expr(expr) => {
                let e = self.lower_expr(expr)?;
                if matches!(e.kind, sir::ExprKind::Literal(_)) {
                    return none;
                }
                sir::StmtKind::ExprStmt(e)
            }
        };

        let before = before.map(|kind| sir::Stmt {
            kind,
            annotations: Vec::new(),
            span,
        });
        let stmt = sir::Stmt {
            kind,
            annotations,
            span,
        };
        Ok(before.into_iter().chain([stmt]).collect())
    }

    /// The gate a `defcal` calibrates. A name not yet in use declares one;
    /// any other use of the name is a redefinition (docs/scope.rst).
    fn defcal_gate(&mut self, name: &ast::Ident<'_>) -> Result<SymbolId> {
        let Ok(symbol) = self.resolver.resolve(name.name, name.span) else {
            return self
                .resolver
                .declare(name.name, SymbolKind::Gate, Type::Void, name.span);
        };
        let existing = self.resolver.symbols().get(symbol);
        if existing.kind != SymbolKind::Gate {
            let kind = ErrorKind::DuplicateDefinition {
                name: name.name.to_string(),
                prior: existing.span,
            };
            return Err(CompileError::at(kind, name.span));
        }
        Ok(symbol)
    }

    /// The right-hand side of an assignment, where an array literal may
    /// appear.
    fn lower_rhs(&mut self, e: &ast::Expr<'_>) -> Result<sir::Expr> {
        match e {
            ast::Expr::ArrayLiteral(al) => self.lower_array_literal_expr(al),
            e => self.lower_expr(e),
        }
    }

    fn lower_gate_operands(
        &mut self,
        operands: &[ast::GateOperand<'_>],
    ) -> Result<Vec<sir::QubitOperand>> {
        operands
            .iter()
            .map(|o| self.lower_gate_operand(o))
            .collect()
    }

    // ── Include handling ────────────────────────────────────────────────

    fn lower_include(&mut self, path: &str, span: oqi_lex::Span) -> Result<Vec<sir::Stmt>> {
        let source = self.resolver.classify_include(path, span)?;
        match source {
            IncludeSource::Lib(_) => {
                let content = self.resolver.resolve_source(&source, span)?.into_owned();
                // A library is not a file a diagnostic could point into, so a
                // syntax error in one is reported at the `include`.
                let ast = oqi_parse::parse(&content).map_err(|e| {
                    let message = format!("in include '{path}': {}", e.message());
                    CompileError::at(ErrorKind::Syntax(message), span)
                })?;
                self.lower_items(&ast.body)
            }
            IncludeSource::Path(file_path) => self.with_include(file_path, |this, file_path| {
                let src = IncludeSource::Path(file_path.to_path_buf());
                let content = this.resolver.resolve_source(&src, span)?.into_owned();
                let ast = oqi_parse::parse(&content)
                    .map_err(|e| CompileError::from(e).or_path(Some(file_path.to_path_buf())))?;
                this.lower_items(&ast.body)
            }),
        }
    }

    // ── Expression lowering ─────────────────────────────────────────────

    fn lower_expr(&mut self, expr: &ast::Expr<'_>) -> Result<sir::Expr> {
        let span = expr.span();
        if is_foldable_kind(expr)
            && let Ok(Value::Scalar(scalar)) = eval_const_expr(expr, &self.resolver)
        {
            let ty = Type::from(scalar.ty());
            return Ok(sir::Expr {
                kind: sir::ExprKind::Literal(scalar.into_value()),
                ty,
                span,
            });
        }
        match expr {
            ast::Expr::Ident(id) => {
                let sym = self.resolver.resolve(id.name, id.span)?;
                let ty = self.resolver.symbols().get(sym).ty.clone();
                Ok(sir::Expr {
                    kind: sir::ExprKind::Var(sym),
                    ty,
                    span,
                })
            }

            ast::Expr::HardwareQubit(s, _) => {
                let n = parse_hardware_qubit(s, span)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::HardwareQubit(n),
                    ty: Type::PhysicalQubit,
                    span,
                })
            }

            ast::Expr::IntLiteral(s, enc, _) => {
                let int = parse_int_literal(s, *enc).with_span(span)?;
                // Unsized: the literal adopts the width of its surrounding
                // sized context, falling back to the system width if it never
                // meets one (see `Type::value_ty`).
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(Primitive::int(int)),
                    ty: Type::Unsized(UnsizedType::Int),
                    span,
                })
            }

            ast::Expr::FloatLiteral(s, _) => Ok(sir::Expr {
                kind: sir::ExprKind::Literal(Primitive::float(
                    parse_float_literal(s).with_span(span)?,
                )),
                ty: Type::Unsized(UnsizedType::Float),
                span,
            }),

            ast::Expr::ImaginaryLiteral(s, _) => {
                let fw = self.resolver.options().system_width.fw();
                let (re, im) = parse_imag_literal(s)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(Primitive::complex(re, im)),
                    ty: Type::Classical(ValueTy::complex(fw)),
                    span,
                })
            }

            ast::Expr::BoolLiteral(b, _) => Ok(sir::Expr {
                kind: sir::ExprKind::Literal(Primitive::bit(*b)),
                ty: Type::Classical(ValueTy::bool()),
                span,
            }),

            ast::Expr::BitstringLiteral(s, _) => {
                let (bits, len) = parse_bitstring_literal(s)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(Primitive::bitreg_u128(bits)),
                    ty: Type::Classical(ValueTy::bitreg(len as u32)),
                    span,
                })
            }

            ast::Expr::TimingLiteral(s, _) => {
                let tv = parse_timing_literal(s, &self.resolver.options().dt)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(Primitive::from(tv)),
                    ty: Type::Classical(ValueTy::duration()),
                    span,
                })
            }

            ast::Expr::Paren(inner, _) => self.lower_expr(inner),

            ast::Expr::Binary {
                left, op, right, ..
            } => {
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let sir_op = bin_op(*op);
                // A same-family unsized operand collapses to the other (sized)
                // operand's type, so the op runs at a single concrete width
                // (e.g. `v + 1` with `v: uint[8]` stays uint[8]). Shifts are
                // excluded: their RHS is an independent shift count.
                let (l, r) = if matches!(sir_op, sir::BinOp::Shl | sir::BinOp::Shr) {
                    (l, r)
                } else {
                    collapse_binary_operands(l, r)
                };
                let ty = binary_result_type(&sir_op, &l.ty, &r.ty, span)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Binary(sir::Binary {
                        op: sir_op,
                        left: Box::new(l),
                        right: Box::new(r),
                    }),
                    ty,
                    span,
                })
            }

            ast::Expr::Unary { op, operand, .. } => {
                let inner = self.lower_expr(operand)?;
                let sir_op = un_op(*op);
                let ty = unary_result_type(&sir_op, &inner.ty);
                Ok(sir::Expr {
                    kind: sir::ExprKind::Unary(sir::Unary {
                        op: sir_op,
                        operand: Box::new(inner),
                    }),
                    ty,
                    span,
                })
            }

            ast::Expr::Index { expr, index, .. } => {
                let base = self.lower_expr(expr)?;
                let idx = self.lower_index_op(index)?;
                let ty = index_result_type(&base.ty, &idx)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Index(sir::Index {
                        base: Box::new(base),
                        index: idx,
                    }),
                    ty,
                    span,
                })
            }

            ast::Expr::Call { name, args, .. } => {
                let callee = self.resolver.resolve_call(name.name, name.span)?;
                let mut sir_args: Vec<sir::Expr> = args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<_>>()?;
                if let sir::CallTarget::Symbol(sym) = &callee
                    && let Some(param_tys) = self.param_types_of(*sym)
                    && param_tys.len() == sir_args.len()
                {
                    for (arg, pty) in sir_args.iter_mut().zip(&param_tys) {
                        let taken = std::mem::replace(
                            arg,
                            sir::Expr {
                                kind: sir::ExprKind::Literal(Primitive::bit(false)),
                                ty: Type::Void,
                                span,
                            },
                        );
                        *arg = collapse_to(taken, pty);
                    }
                }
                let ty = self.call_result_type(&callee, &sir_args).with_span(span)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Call(sir::Call {
                        callee,
                        args: sir_args,
                    }),
                    ty,
                    span,
                })
            }

            ast::Expr::Cast { ty, operand, .. } => {
                let target_ty = resolve_type(ty, &self.resolver)?;
                let mut inner = self.lower_expr(operand)?;
                // An unsized literal operand collapses to the cast's target
                // width first, so `bit[8](5)` becomes an equal-width cast.
                if inner.ty.unsized_ty().is_some() {
                    inner = collapse_to(inner, &target_ty);
                }
                validate_cast(&inner.ty, &target_ty, span)?;
                let result_ty = target_ty.clone();
                Ok(sir::Expr {
                    kind: sir::ExprKind::Cast(sir::Cast {
                        target_ty,
                        operand: Box::new(inner),
                    }),
                    ty: result_ty,
                    span,
                })
            }

            ast::Expr::DurationOf { scope, .. } => {
                let stmts = self.with_scope(|this| this.lower_items(&scope.body))?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::DurationOf(stmts),
                    ty: Type::Classical(ValueTy::duration()),
                    span,
                })
            }

            ast::Expr::ArrayLiteral(_) => Err(CompileError::at(ErrorKind::InvalidContext(
                "array literal only valid as the right-hand side of a declaration or assignment"
                    .into(),
            ), span)),
        }
    }

    // ── Helper lowering methods ─────────────────────────────────────────

    fn lower_gate_operand(&mut self, op: &ast::GateOperand<'_>) -> Result<sir::QubitOperand> {
        match op {
            ast::GateOperand::Indexed(id) => {
                let sym = self.resolver.resolve(id.name.name, id.name.span)?;
                let indices = id
                    .indices
                    .iter()
                    .map(|i| self.lower_index_op(i))
                    .collect::<Result<_>>()?;
                Ok(sir::QubitOperand::Indexed {
                    symbol: sym,
                    indices,
                })
            }
            ast::GateOperand::HardwareQubit(s, span) => {
                Ok(sir::QubitOperand::Hardware(parse_hardware_qubit(s, *span)?))
            }
        }
    }

    fn lower_indexed_ident_to_lvalue(&mut self, id: &ast::IndexedIdent<'_>) -> Result<sir::LValue> {
        let sym = self.resolver.resolve(id.name.name, id.name.span)?;
        if id.indices.is_empty() {
            Ok(sir::LValue::Var(sym))
        } else {
            let indices = id
                .indices
                .iter()
                .map(|i| self.lower_index_op(i))
                .collect::<Result<_>>()?;
            Ok(sir::LValue::Indexed {
                symbol: sym,
                indices,
            })
        }
    }

    fn lower_indexed_ident_to_expr(&mut self, id: &ast::IndexedIdent<'_>) -> Result<sir::Expr> {
        let sym = self.resolver.resolve(id.name.name, id.name.span)?;
        let base_ty = self.resolver.symbols().get(sym).ty.clone();
        let mut expr = sir::Expr {
            kind: sir::ExprKind::Var(sym),
            ty: base_ty,
            span: id.name.span,
        };
        for idx in &id.indices {
            let sir_idx = self.lower_index_op(idx)?;
            let ty = index_result_type(&expr.ty, &sir_idx)?;
            let span = id.name.span.join(idx.span);
            expr = sir::Expr {
                kind: sir::ExprKind::Index(sir::Index {
                    base: Box::new(expr),
                    index: sir_idx,
                }),
                ty,
                span,
            };
        }
        Ok(expr)
    }

    fn lower_index_op(&mut self, op: &ast::IndexOp<'_>) -> Result<sir::IndexOp> {
        let kind = match &op.kind {
            ast::IndexKind::Set(exprs) => {
                let items = exprs
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                sir::IndexKind::Set(items)
            }
            ast::IndexKind::Items(items) => {
                let sir_items = items
                    .iter()
                    .map(|item| match item {
                        ast::IndexItem::Single(e) => {
                            Ok(sir::IndexItem::Single(Box::new(self.lower_expr(e)?)))
                        }
                        ast::IndexItem::Range(r) => Ok(sir::IndexItem::Range(sir::RangeExpr {
                            start: r
                                .start
                                .as_ref()
                                .map(|e| self.lower_expr(e).map(Box::new))
                                .transpose()?,
                            step: r
                                .step
                                .as_ref()
                                .map(|e| self.lower_expr(e).map(Box::new))
                                .transpose()?,
                            end: r
                                .end
                                .as_ref()
                                .map(|e| self.lower_expr(e).map(Box::new))
                                .transpose()?,
                        })),
                    })
                    .collect::<Result<_>>()?;
                sir::IndexKind::Items(sir_items)
            }
        };
        Ok(sir::IndexOp {
            kind,
            span: op.span,
        })
    }

    fn lower_gate_modifier(&mut self, m: &ast::GateModifier<'_>) -> Result<sir::GateModifier> {
        match m {
            ast::GateModifier::Inv(_) => Ok(sir::GateModifier::Inv),
            ast::GateModifier::Pow(expr, _) => {
                Ok(sir::GateModifier::Pow(Box::new(self.lower_expr(expr)?)))
            }
            ast::GateModifier::Ctrl(designator, _) => {
                let n = match designator {
                    Some(e) => eval_designator(e, &self.resolver)?,
                    None => 1,
                };
                Ok(sir::GateModifier::Ctrl(n))
            }
            ast::GateModifier::NegCtrl(designator, _) => {
                let n = match designator {
                    Some(e) => eval_designator(e, &self.resolver)?,
                    None => 1,
                };
                Ok(sir::GateModifier::NegCtrl(n))
            }
        }
    }

    fn lower_measure_expr(&mut self, m: &ast::MeasureExpr<'_>) -> Result<sir::MeasureExpr> {
        match m {
            ast::MeasureExpr::Measure { operand, span } => {
                let op = self.lower_gate_operand(operand)?;
                let ty = measure_result_type(&self.qubit_operand_type(&op)?);
                Ok(sir::MeasureExpr {
                    kind: sir::MeasureExprKind::Measure { operand: op },
                    ty,
                    span: *span,
                })
            }
            ast::MeasureExpr::QuantumCall {
                name,
                args,
                operands,
                span,
            } => {
                let sym = self.resolver.resolve(name.name, name.span)?;
                let ty = self.resolver.symbols().get(sym).ty.clone();
                let sir_args = args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<_>>()?;
                let sir_qubits = operands
                    .iter()
                    .map(|o| self.lower_gate_operand(o))
                    .collect::<Result<_>>()?;
                Ok(sir::MeasureExpr {
                    kind: sir::MeasureExprKind::QuantumCall {
                        callee: sym,
                        args: sir_args,
                        qubits: sir_qubits,
                    },
                    ty,
                    span: *span,
                })
            }
        }
    }

    fn param_types_of(&self, sym: SymbolId) -> Option<Vec<Type>> {
        if let Some(d) = self.subroutines.iter().find(|d| d.symbol == sym) {
            Some(
                d.params
                    .iter()
                    .map(|p| self.resolver.symbols().get(*p).ty.clone())
                    .collect(),
            )
        } else {
            self.externs
                .iter()
                .find(|d| d.symbol == sym)
                .map(|d| d.param_types.clone())
        }
    }

    /// The type of a `let` alias of the concatenation of `parts`
    /// (docs/types.rst, "Aliasing" and "Register concatenation and
    /// slicing"). Qubits and bits of declared registers can be aliased: an
    /// alias of bits is typed as the register of them, an alias of qubits
    /// is left to the qubit layout (`Type::Void`).
    fn alias_type(&self, parts: &[sir::Expr]) -> Result<Type> {
        let invalid = |message: &str, span| {
            CompileError::at(ErrorKind::InvalidOperands(message.into()), span)
        };
        // How many bits each operand that names bits contributes.
        let mut widths = Vec::new();
        for part in parts {
            if matches!(part.kind, sir::ExprKind::HardwareQubit(_)) {
                let message = "physical qubits are not declared and so cannot be aliased";
                return Err(invalid(message, part.span));
            }
            let root = qubits::peel_index_chain(part)
                .map(|(symbol, _)| self.resolver.symbols().get(symbol));
            let width = match root.map(|root| (&root.ty, root.kind)) {
                Some((Type::Qubit | Type::QubitReg(_), _) | (Type::Void, SymbolKind::Alias)) => {
                    continue;
                }
                // A constant is no register.
                Some((_, SymbolKind::Const)) | None => None,
                Some((root_ty, _)) => match (root_ty.scalar_ty(), part.ty.scalar_ty()) {
                    (Some(PrimitiveTy::Bit | PrimitiveTy::BitReg(_)), Some(PrimitiveTy::Bit)) => {
                        Some(1)
                    }
                    (
                        Some(PrimitiveTy::Bit | PrimitiveTy::BitReg(_)),
                        Some(PrimitiveTy::BitReg(width)),
                    ) => Some(width),
                    _ => None,
                },
            };
            let message = "only qubits, qubit registers and bit registers can be aliased";
            widths.push(width.ok_or_else(|| invalid(message, part.span))?);
        }
        let one_bit = matches!(parts, [part] if part.ty.scalar_ty() == Some(PrimitiveTy::Bit));
        match widths.len() {
            0 => Ok(Type::Void),
            n if n != parts.len() => {
                let message = "an alias concatenates registers of the same type, \
                               classical or quantum";
                let span = parts[0].span.join(parts[parts.len() - 1].span);
                Err(invalid(message, span))
            }
            _ if one_bit => Ok(Type::Classical(ValueTy::bit())),
            _ => Ok(Type::Classical(ValueTy::bitreg(widths.iter().sum()))),
        }
    }

    /// The type of `symbol` after the subscripts `indices`.
    fn indexed_type(&self, symbol: SymbolId, indices: &[sir::IndexOp]) -> Result<Type> {
        let base = self.resolver.symbols().get(symbol).ty.clone();
        indices
            .iter()
            .try_fold(base, |ty, index| index_result_type(&ty, index))
    }

    fn lvalue_type(&self, lv: &sir::LValue) -> Result<Type> {
        match lv {
            sir::LValue::Var(symbol) => self.indexed_type(*symbol, &[]),
            sir::LValue::Indexed { symbol, indices } => self.indexed_type(*symbol, indices),
        }
    }

    fn qubit_operand_type(&self, op: &sir::QubitOperand) -> Result<Type> {
        match op {
            sir::QubitOperand::Hardware(_) => Ok(Type::PhysicalQubit),
            sir::QubitOperand::Indexed { symbol, indices } => self.indexed_type(*symbol, indices),
        }
    }

    /// Lower the initializer of a classical declaration to the statement
    /// that performs it. A missing initializer needs none: the symbol entry
    /// alone carries the declaration.
    fn lower_init(
        &mut self,
        symbol: SymbolId,
        init: Option<&ast::ExprOrMeasure<'_>>,
        span: oqi_lex::Span,
    ) -> Result<Option<sir::StmtKind>> {
        let target = sir::LValue::Var(symbol);
        Ok(Some(match init {
            None => return Ok(None),
            Some(ast::ExprOrMeasure::Expr(e)) => {
                let target_ty = self.resolver.symbols().get(symbol).ty.clone();
                let value = match e {
                    ast::Expr::ArrayLiteral(al) => self.lower_array_literal_expr(al)?,
                    e => coerce_assign(self.lower_expr(e)?, &target_ty, span)?,
                };
                let value = sir::RValue::Expr(Box::new(value));
                sir::StmtKind::Assignment(sir::Assignment { target, value })
            }
            Some(ast::ExprOrMeasure::Measure(m)) => sir::StmtKind::Measure(sir::Measure {
                measure: self.lower_measure_expr(m)?,
                target: Some(target),
            }),
        }))
    }

    fn lower_array_literal_expr(&mut self, al: &ast::ArrayLiteral<'_>) -> Result<sir::Expr> {
        if al.items.is_empty() {
            return Err(CompileError::at(
                ErrorKind::InvalidLiteral("array literal must have at least one element".into()),
                al.span,
            ));
        }
        let items = al
            .items
            .iter()
            .map(|item| match item {
                ast::Expr::ArrayLiteral(inner) => self.lower_array_literal_expr(inner),
                e => self.lower_expr(e),
            })
            .collect::<Result<Vec<_>>>()?;
        let ty = synth_array_literal_ty(&items).map_err(|e| invalid_type(e, al.span))?;
        Ok(sir::Expr {
            kind: sir::ExprKind::ArrayLiteral(sir::ArrayLiteral {
                items,
                span: al.span,
            }),
            ty,
            span: al.span,
        })
    }

    fn lower_annotations(&mut self, anns: &[ast::Annotation<'_>]) -> Vec<sir::Annotation> {
        anns.iter()
            .map(|a| sir::Annotation {
                keyword: a.keyword.to_string(),
                content: a.content.map(|s| s.to_string()),
                span: a.span,
            })
            .collect()
    }

    /// Lower what a `for` loop with a variable of type `var_ty` iterates
    /// over (docs/classical.rst): a range of integers, a set of values, or
    /// the items of a bit register or one-dimensional array.
    fn lower_for_iterable(
        &mut self,
        it: &ast::ForIterable<'_>,
        var_ty: &Type,
    ) -> Result<sir::ForIterable> {
        match it {
            ast::ForIterable::Set(exprs, _) => exprs
                .iter()
                .map(|e| coerce_assign(self.lower_expr(e)?, var_ty, e.span()))
                .collect::<Result<_>>()
                .map(sir::ForIterable::Set),
            ast::ForIterable::Range(range, span) => {
                let integer = Type::Classical(ValueTy::int(iw(128)));
                validate_cast(&integer, var_ty, *span)?;
                // "All three values must be of integer or unsigned-integer
                // types." The bounds adopt the loop variable's width; the
                // step is a distance, and keeps its sign.
                let mut bound = |e: &ast::Expr<'_>, collapse: bool| -> Result<Box<sir::Expr>> {
                    let e = self.lower_expr(e)?;
                    let e = if collapse { collapse_to(e, var_ty) } else { e };
                    match e.ty.scalar_ty() {
                        Some(PrimitiveTy::Int(_) | PrimitiveTy::Uint(_)) => Ok(Box::new(e)),
                        _ => {
                            let message = format!("a range is made of integers, not `{}`", e.ty);
                            Err(CompileError::at(
                                ErrorKind::InvalidOperands(message),
                                e.span,
                            ))
                        }
                    }
                };
                let integers = matches!(
                    var_ty.scalar_ty(),
                    Some(PrimitiveTy::Int(_) | PrimitiveTy::Uint(_))
                );
                Ok(sir::ForIterable::Range(sir::RangeExpr {
                    start: range
                        .start
                        .as_ref()
                        .map(|e| bound(e, integers))
                        .transpose()?,
                    step: range.step.as_ref().map(|e| bound(e, false)).transpose()?,
                    end: range.end.as_ref().map(|e| bound(e, integers)).transpose()?,
                }))
            }
            ast::ForIterable::Expr(e) => {
                let values = self.lower_expr(e)?;
                let Some(item) = cfg::item_type(&values.ty) else {
                    let message = format!(
                        "a `for` loop iterates over a range, a set, a bit register or a \
                         one-dimensional array, not `{}`",
                        values.ty
                    );
                    return Err(CompileError::at(
                        ErrorKind::InvalidOperands(message),
                        values.span,
                    ));
                };
                check_assignable(&item, var_ty, values.span)?;
                Ok(sir::ForIterable::Expr(Box::new(values)))
            }
        }
    }

    fn lower_switch_case(&mut self, case: &ast::SwitchCase<'_>) -> Result<sir::SwitchCase> {
        let (labels, scope) = match case {
            ast::SwitchCase::Case(labels, scope) => {
                let labels = labels
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                (sir::SwitchLabels::Values(labels), scope)
            }
            ast::SwitchCase::Default(scope) => (sir::SwitchLabels::Default, scope),
        };
        let body = self.with_scope(|this| this.lower_items(&scope.body))?;
        Ok(sir::SwitchCase { labels, body })
    }

    // ── Subroutine param lowering ───────────────────────────────────────

    /// Declare a subroutine (or `defcal`) parameter.
    fn lower_arg_def(&mut self, arg: &ast::ArgDef<'_>) -> Result<SymbolId> {
        let (ty, name) = match arg {
            ast::ArgDef::Scalar(ty, name) => (resolve_scalar_type(ty, &self.resolver)?, name),
            ast::ArgDef::Qubit(ty, name) => (resolve_qubit_type(ty, &self.resolver)?, name),
            ast::ArgDef::Creg(name, designator) => {
                let kind = &ast::OldStyleKind::Creg;
                let ty = resolve_old_style_type(kind, designator.as_ref(), &self.resolver)?;
                (ty, name)
            }
            ast::ArgDef::Qreg(name, designator) => {
                let kind = &ast::OldStyleKind::Qreg;
                let ty = resolve_old_style_type(kind, designator.as_ref(), &self.resolver)?;
                (ty, name)
            }
            ast::ArgDef::ArrayRef(arr_ref, name) => {
                (resolve_array_ref_type(arr_ref, &self.resolver)?, name)
            }
        };
        self.resolver
            .declare(name.name, SymbolKind::SubroutineParam, ty, name.span)
    }

    fn lower_extern_args(&mut self, params: &[ast::ExternArg<'_>]) -> Result<Vec<Type>> {
        params
            .iter()
            .map(|p| match p {
                ast::ExternArg::Scalar(ty) => resolve_scalar_type(ty, &self.resolver),
                ast::ExternArg::ArrayRef(arr_ref) => {
                    resolve_array_ref_type(arr_ref, &self.resolver)
                }
                ast::ExternArg::Creg(designator) => resolve_old_style_type(
                    &ast::OldStyleKind::Creg,
                    designator.as_ref(),
                    &self.resolver,
                ),
            })
            .collect()
    }

    fn call_result_type(&self, callee: &sir::CallTarget, args: &[sir::Expr]) -> Result<Type> {
        match callee {
            sir::CallTarget::Intrinsic(i) => {
                intrinsic_result_type(i, args, self.resolver.options().system_width)
            }
            sir::CallTarget::Symbol(sym) => Ok(self.resolver.symbols().get(*sym).ty.clone()),
        }
    }
}

// ── Free functions ──────────────────────────────────────────────────────

/// The index of a hardware qubit `$n`. The program's qubit count, which
/// covers `$n`, is a `u32`, so `n` must be below `u32::MAX`.
fn parse_hardware_qubit(s: &str, span: oqi_lex::Span) -> Result<usize> {
    let invalid = |message: String| CompileError::at(ErrorKind::InvalidLiteral(message), span);
    let digits = s
        .strip_prefix('$')
        .ok_or_else(|| invalid(format!("invalid hardware qubit `{s}`")))?;
    digits
        .parse::<u32>()
        .ok()
        .filter(|&n| n < u32::MAX)
        .map(|n| n as usize)
        .ok_or_else(|| {
            invalid(format!(
                "hardware qubit `{s}` is out of range (the highest is ${})",
                u32::MAX - 1
            ))
        })
}

impl Lowerer<'_> {
    fn lower_cal_body(&mut self, body: &ast::CalBody<'_>) -> Result<sir::CalibrationBody> {
        match body {
            ast::CalBody::Raw(text) => {
                Ok(sir::CalibrationBody::Opaque(text.unwrap_or("").to_string()))
            }
            ast::CalBody::OpenPulse(items) => {
                Ok(sir::CalibrationBody::OpenPulse(self.lower_items(items)?))
            }
        }
    }

    fn port_type() -> Type {
        Type::Openpulse(openpulse::PrimitiveTy::Port)
    }

    fn frame_type(&self) -> Type {
        Type::Openpulse(openpulse::PrimitiveTy::Frame(
            self.resolver.options().system_width.fw(),
            self.resolver.options().system_width.iw(),
        ))
    }

    fn waveform_type(&self) -> Type {
        Type::Openpulse(openpulse::PrimitiveTy::Waveform(
            self.resolver.options().system_width.fw(),
        ))
    }

    fn seed_openpulse_intrinsics(&mut self) -> Result<()> {
        use crate::classical::ValueTy;
        let angle_bw = self.resolver.options().system_width.iw();
        let uint_bw = self.resolver.options().system_width.iw();
        let float_fw = self.resolver.options().system_width.fw();
        let complex_ty = Type::Classical(ValueTy::complex(float_fw));
        let angle_ty = Type::Classical(ValueTy::angle(angle_bw));
        let uint_ty = Type::Classical(ValueTy::uint(uint_bw));
        let float_ty = Type::Classical(ValueTy::float(float_fw));
        let duration_ty = Type::Classical(ValueTy::duration());
        let bit_ty = Type::Classical(ValueTy::bit());
        let port_ty = Self::port_type();
        let frame_ty = self.frame_type();
        let waveform_ty = self.waveform_type();
        let span = oqi_lex::Span::default();

        let intrinsics: [(&str, Vec<Type>, Option<Type>); 6] = [
            (
                "newframe",
                vec![port_ty, float_ty.clone(), angle_ty.clone()],
                Some(frame_ty.clone()),
            ),
            (
                "gaussian",
                vec![float_ty.clone(), duration_ty.clone(), duration_ty.clone()],
                Some(waveform_ty.clone()),
            ),
            ("play", vec![frame_ty.clone(), waveform_ty], None),
            (
                "capture",
                vec![frame_ty.clone(), uint_ty.clone()],
                Some(complex_ty.clone()),
            ),
            ("shift_phase", vec![frame_ty, angle_ty.clone()], None),
            ("threshold", vec![complex_ty, uint_ty], Some(bit_ty)),
        ];

        for (name, params, ret_ty) in intrinsics {
            if self.resolver.resolve(name, span).is_ok() {
                continue;
            }
            let sym = self
                .resolver
                .declare(name, SymbolKind::Extern, Type::Void, span)?;
            if let Some(ref ty) = ret_ty {
                self.resolver.symbols_mut().get_mut(sym).ty = ty.clone();
            }
            self.externs.push(sir::ExternDecl {
                symbol: sym,
                param_types: params,
                return_ty: ret_ty,
                span,
            });
        }
        Ok(())
    }
}

fn compound_to_bin_op(op: &ast::AssignOp) -> Option<sir::BinOp> {
    match op {
        ast::AssignOp::Assign => None,
        ast::AssignOp::AddAssign => Some(sir::BinOp::Add),
        ast::AssignOp::SubAssign => Some(sir::BinOp::Sub),
        ast::AssignOp::MulAssign => Some(sir::BinOp::Mul),
        ast::AssignOp::DivAssign => Some(sir::BinOp::Div),
        ast::AssignOp::ModAssign => Some(sir::BinOp::Mod),
        ast::AssignOp::PowAssign => Some(sir::BinOp::Pow),
        ast::AssignOp::BitAndAssign => Some(sir::BinOp::BitAnd),
        ast::AssignOp::BitOrAssign => Some(sir::BinOp::BitOr),
        ast::AssignOp::BitXorAssign => Some(sir::BinOp::BitXor),
        ast::AssignOp::LeftShiftAssign => Some(sir::BinOp::Shl),
        ast::AssignOp::RightShiftAssign => Some(sir::BinOp::Shr),
    }
}

// ── Type inference ──────────────────────────────────────────────────────

fn binary_result_type(
    op: &sir::BinOp,
    left: &Type,
    right: &Type,
    span: oqi_lex::Span,
) -> Result<Type> {
    let invalid = |message: String| CompileError::at(ErrorKind::InvalidOperands(message), span);
    let (Some(lhs), Some(rhs)) = (left.value_ty(), right.value_ty()) else {
        let operation = format!("{left} {} {right}", op.name());
        return Err(invalid(format!("`{operation}` is not defined")));
    };
    op.return_ty(lhs, rhs)
        .map(Type::from)
        .map_err(|e| invalid(e.to_string()))
}

fn unary_result_type(op: &sir::UnOp, operand: &Type) -> Type {
    operand
        .value_ty()
        .and_then(|value_ty| op.return_ty(value_ty).ok())
        .map(Type::from)
        .unwrap_or_else(|| match op {
            sir::UnOp::LogNot => Type::Classical(ValueTy::bool()),
            _ => operand.clone(),
        })
}

/// Collapse a same-family unsized operand of a binary op to the other,
/// concrete operand's type. Cross-family pairs (e.g. an unsized float with a
/// sized int) and unsized∘unsized are left alone — the carrier width and
/// normal promotion handle them.
fn collapse_binary_operands(l: sir::Expr, r: sir::Expr) -> (sir::Expr, sir::Expr) {
    match (l.ty.unsized_ty().is_some(), r.ty.unsized_ty().is_some()) {
        (true, false) if same_family(&l.ty, &r.ty) => {
            let target = r.ty.clone();
            (collapse_to(l, &target), r)
        }
        (false, true) if same_family(&l.ty, &r.ty) => {
            let target = l.ty.clone();
            (l, collapse_to(r, &target))
        }
        _ => (l, r),
    }
}

/// Whether two types are in the same broad numeric family (int-like, float-like,
/// or angle), comparing via their (carrier) scalar types.
fn same_family(a: &Type, b: &Type) -> bool {
    use crate::classical::PrimitiveTy::*;
    let family = |t: &Type| {
        t.scalar_ty().map(|p| match p {
            Bit | Bool | Int(_) | Uint(_) | BitReg(_) => 0u8,
            Float(_) | Complex(_) => 1,
            Angle(_) => 2,
            Duration => 3,
        })
    };
    matches!((family(a), family(b)), (Some(x), Some(y)) if x == y)
}

/// Collapse a literal expression to a target type's scalar width. An *unsized*
/// literal adopts the target width directly (value conversion, no equal-width
/// check — an unsized literal can take any width), which is how it "collapses
/// to the sized type it meets". A *concrete* literal is re-cast exactly as the
/// old `coerce_literal` did. Non-literals and non-scalar targets pass through.
fn collapse_to(expr: sir::Expr, target: &Type) -> sir::Expr {
    let sir::Expr { kind, ty, span } = expr;
    let prim = match kind {
        sir::ExprKind::Literal(p) => p,
        other => {
            return sir::Expr {
                kind: other,
                ty,
                span,
            };
        }
    };
    let rebuild = |prim, ty| sir::Expr {
        kind: sir::ExprKind::Literal(prim),
        ty,
        span,
    };
    let Some(to) = target.scalar_ty() else {
        return rebuild(prim, ty);
    };
    if ty.unsized_ty().is_some() {
        return match prim.clone().as_ty(to) {
            Ok(v) => rebuild(v, Type::from(to)),
            // No value conversion exists (e.g. int -> angle): leave the literal
            // at its carrier type and let the surrounding context handle it.
            Err(_) => rebuild(prim, ty),
        };
    }
    let Some(from) = ty.scalar_ty() else {
        return rebuild(prim, ty);
    };
    if from == to {
        return rebuild(prim, ty);
    }
    match Scalar::new(prim.clone(), from).and_then(|s| s.cast(to)) {
        Ok(casted) => rebuild(casted.into_value(), Type::from(to)),
        Err(_) => rebuild(prim, ty),
    }
}

/// Coerce an assignment/decl-init RHS to the target type, per the spec's
/// assignment rules (see [`check_assignable`]), with a cast where the
/// representation changes.
fn coerce_assign(expr: sir::Expr, target: &Type, span: oqi_lex::Span) -> Result<sir::Expr> {
    let expr = collapse_to(expr, target);
    check_assignable(&expr.ty, target, span)?;
    match (expr.ty.scalar_ty(), target.scalar_ty()) {
        // `from == to` covers types of one representation (e.g. `stretch`
        // into `duration`).
        (Some(from), Some(to)) if from != to => Ok(sir::Expr {
            ty: target.clone(),
            span: expr.span,
            kind: sir::ExprKind::Cast(sir::Cast {
                target_ty: target.clone(),
                operand: Box::new(expr),
            }),
        }),
        _ => Ok(expr),
    }
}

/// Whether a value of type `from` can be assigned to a `to` (docs/types.rst,
/// implicit-promotion-rules): standard classical types convert implicitly
/// (C99-style), as do the noted special-type exceptions (float → angle,
/// angle width changes, bit ↔ bool), and an unsized literal takes any type
/// it can be cast to; any other mismatch demands an explicit cast. Arrays
/// (and array and scalar) must agree in shape.
fn check_assignable(from: &Type, to: &Type, span: oqi_lex::Span) -> Result<()> {
    let mismatch = || {
        CompileError::at(
            ErrorKind::TypeMismatch {
                expected: to.to_string(),
                got: from.to_string(),
            },
            span,
        )
    };
    match (from.value_ty(), to.value_ty()) {
        (Some(ValueTy::Scalar(from_ty)), Some(ValueTy::Scalar(to_ty))) => {
            if from_ty == to_ty {
                return Ok(());
            }
            if !(from.unsized_ty().is_some() || is_implicit_assign(&from_ty, &to_ty)) {
                return Err(mismatch());
            }
            validate_cast(from, to, span)
        }
        (Some(from_ty), Some(to_ty)) => match (array_shape(from_ty), array_shape(to_ty)) {
            // A shape only known at run time is checked there.
            (Some(Some(a)), Some(Some(b))) if a != b => Err(mismatch()),
            (Some(_), Some(_)) => Ok(()),
            _ => Err(mismatch()),
        },
        // Not values (aliases, qubits): nothing to convert.
        _ => Ok(()),
    }
}

/// The shape of an array type: `None` for a scalar, `Some(None)` for an
/// array reference whose sizes are only known at run time.
fn array_shape(ty: ValueTy) -> Option<Option<Vec<usize>>> {
    match ty {
        ValueTy::Scalar(_) => None,
        ValueTy::Array(ty) => Some(Some(ty.shape().dims().to_vec())),
        ValueTy::ArrayRef(ty) => Some(match ty.shape() {
            ArrayRefShape::Fixed(shape) => Some(shape.dims().to_vec()),
            ArrayRefShape::Dim(_) => None,
        }),
    }
}

fn is_implicit_assign(
    from: &crate::classical::PrimitiveTy,
    to: &crate::classical::PrimitiveTy,
) -> bool {
    use crate::classical::PrimitiveTy::{Angle, Bit, Bool, Complex, Float, Int, Uint};
    let standard = |p: &crate::classical::PrimitiveTy| {
        matches!(p, Bool | Int(_) | Uint(_) | Float(_) | Complex(_))
    };
    match (from, to) {
        _ if standard(from) && standard(to) => true,
        (Angle(_), Angle(_)) => true,
        (Float(_), Angle(_)) => true,
        (Bool, Bit) | (Bit, Bool) => true,
        _ => false,
    }
}

fn is_foldable_kind(expr: &ast::Expr<'_>) -> bool {
    matches!(
        expr,
        ast::Expr::Ident(_)
            | ast::Expr::Binary { .. }
            | ast::Expr::Unary { .. }
            | ast::Expr::Call { .. }
            | ast::Expr::Cast { .. }
            | ast::Expr::Paren(_, _)
    )
}

fn measure_result_type(qubit_ty: &Type) -> Type {
    match qubit_ty {
        Type::QubitReg(n) => Type::Classical(ValueTy::bitreg(*n as u32)),
        _ => Type::Classical(ValueTy::bit()),
    }
}

/// The type of `base[index]` (docs/types.rst, "Index sets and slicing"):
/// a single position selects an item, a range or set a register or array
/// of as many items as it selects.
pub(crate) fn index_result_type(base_ty: &Type, index: &sir::IndexOp) -> Result<Type> {
    if let Type::QubitReg(len) = base_ty {
        let selected = qubits::resolve_index(index, *len)?;
        return Ok(match selected {
            _ if qubits::single_index_expr(index).is_some() => Type::Qubit,
            Resolved::Static(positions) => Type::QubitReg(positions.len()),
            // The size is only known at run time.
            Resolved::Dynamic => base_ty.clone(),
        });
    }
    // Not a value (an alias, say): resolved when the reference is.
    let Some(value_ty) = base_ty.value_ty() else {
        return Ok(base_ty.clone());
    };
    let indices = typing_indices(value_ty, index)?;
    let indexed = value_ty.get(&indices).map(Type::from);
    indexed.map_err(|e| CompileError::at(ErrorKind::InvalidOperands(e.to_string()), index.span))
}

/// The classical indices that type the subscript `index` of a value of type
/// `ty`: a literal position as itself (so it is range-checked), any other
/// as position 0, and a range or set as the positions it selects.
///
/// The size of a slice is part of its type, so the bounds of a range have
/// to be constants — except on an array reference of unspecified length,
/// whose sizes are not part of its type either.
fn typing_indices(ty: ValueTy, index: &sir::IndexOp) -> Result<Vec<ClassicalIndex>> {
    let invalid =
        |message: String| CompileError::at(ErrorKind::InvalidOperands(message), index.span);
    let empty = || invalid("a register cannot be indexed by an empty index set".into());
    let position = |e: &sir::Expr| qubits::literal_index(e).map_or(0, |i| i as isize);
    // The sizes the subscript positions index into, where known: the
    // dimensions of an array, then (or for a scalar) its bits.
    let (mut sizes, element): (Vec<Option<usize>>, PrimitiveTy) = match ty {
        ValueTy::Scalar(ty) => (Vec::new(), ty),
        ValueTy::Array(ty) => (
            ty.shape().dims().iter().copied().map(Some).collect(),
            ty.element_ty(),
        ),
        ValueTy::ArrayRef(ty) => match ty.shape() {
            ArrayRefShape::Fixed(shape) => (
                shape.dims().iter().copied().map(Some).collect(),
                ty.element_ty(),
            ),
            ArrayRefShape::Dim(rank) => (vec![None; rank.get()], ty.element_ty()),
        },
    };
    let is_array = !sizes.is_empty();
    sizes.push(element.bit_count().map(|n| n as usize));

    match &index.kind {
        sir::IndexKind::Set(_) if is_array => {
            // docs/types.rst: arrays "cannot be indexed by a comma-separated
            // list of integers contained in braces".
            Err(invalid(
                "an array cannot be indexed by a set; use a range".into(),
            ))
        }
        sir::IndexKind::Set(members) if members.is_empty() => Err(empty()),
        sir::IndexKind::Set(members) => Ok(vec![ClassicalIndex::Select(
            members.iter().map(position).collect(),
        )]),
        sir::IndexKind::Items(items) => {
            let mut indices = Vec::with_capacity(items.len());
            for (k, item) in items.iter().enumerate() {
                let range = match item {
                    sir::IndexItem::Single(e) => {
                        indices.push(ClassicalIndex::Item(position(e)));
                        continue;
                    }
                    sir::IndexItem::Range(range) => range,
                };
                let positions = match sizes.get(k).copied().flatten() {
                    None => Vec::new(),
                    Some(len) => match qubits::expand_range(range, len) {
                        Ok(Some(positions)) if positions.is_empty() => return Err(empty()),
                        Ok(Some(positions)) => positions.into_iter().map(|p| p as isize).collect(),
                        Ok(None) => {
                            let kind = ErrorKind::NonConstantExpression;
                            return Err(CompileError::at(kind, index.span));
                        }
                        Err(e) => {
                            return Err(match e.kind {
                                ErrorKind::QubitIndexOutOfRange { index, .. } => {
                                    invalid(format!("index {index} is out of range for `{ty}`"))
                                }
                                _ => e.with_span(index.span),
                            });
                        }
                    },
                };
                indices.push(ClassicalIndex::Select(positions));
            }
            Ok(indices)
        }
    }
}

fn synth_array_literal_ty(items: &[sir::Expr]) -> oqi_classical::Result<Type> {
    let Some(first) = items.first() else {
        return Ok(Type::Void);
    };
    let len = items.len();
    Ok(match &first.ty {
        Type::Classical(ValueTy::Array(inner)) => {
            let mut shape = vec![len];
            shape.extend_from_slice(inner.shape().dims());
            Type::from(ArrayTy::new(inner.element_ty(), ArrayShape::new(shape)?))
        }
        Type::Classical(ValueTy::Scalar(prim)) => {
            Type::from(ArrayTy::new(*prim, ArrayShape::new(vec![len])?))
        }
        _ => first.ty.clone(),
    })
}

fn intrinsic_result_type(
    intrinsic: &sir::Intrinsic,
    args: &[sir::Expr],
    system_width: SystemWidth,
) -> Result<Type> {
    let arg_ty = |arg: &sir::Expr| intrinsic_arg_type(intrinsic, &arg.ty);
    match intrinsic {
        sir::Intrinsic::Unary(op) => {
            let [arg] = args else {
                return Err(intrinsic_arity_error(intrinsic, 1, args.len()));
            };
            op.return_ty(arg_ty(arg)?)
                .map(Type::from)
                .map_err(classical_intrinsic_error)
        }
        sir::Intrinsic::Binary(op) => {
            let [lhs, rhs] = args else {
                return Err(intrinsic_arity_error(intrinsic, 2, args.len()));
            };
            op.return_ty(arg_ty(lhs)?, arg_ty(rhs)?)
                .map(Type::from)
                .map_err(classical_intrinsic_error)
        }
        sir::Intrinsic::Sizeof => {
            let (value, dim) = match args {
                [value] => (value, None),
                [value, dim] => (value, Some(dim)),
                _ => {
                    return Err(CompileError::new(ErrorKind::InvalidOperands(format!(
                        "`{intrinsic}` expects 1 or 2 argument(s), got {}",
                        args.len()
                    ))));
                }
            };
            let value_ty = arg_ty(value)?;
            let dim_ty = dim.map(arg_ty).transpose()?;
            ops::sizeof_ty(value_ty, dim_ty).map_err(classical_intrinsic_error)?;
            if let Some(dim) = dim
                && let sir::ExprKind::Literal(value) = &dim.kind
            {
                let Some(dim) = value.as_int(iw(128)).and_then(|i| usize::try_from(i).ok()) else {
                    return Err(CompileError::new(ErrorKind::InvalidOperands(format!(
                        "`{intrinsic}` requires a non-negative integer dimension, got `{}`",
                        dim.ty
                    ))));
                };
                // For unspecified-length array refs the concrete sizes
                // aren't known until run time, but the rank is — validate
                // the literal dimension against the rank instead.
                if dim > value_ty.sizeof_rank().unwrap_or(0) {
                    return Err(CompileError::new(ErrorKind::InvalidOperands(format!(
                        "`{intrinsic}` does not support dimension {dim} for argument type `{value_ty}`"
                    ))));
                }
            }
            Ok(Type::Classical(ValueTy::uint(system_width.iw())))
        }
    }
}

fn intrinsic_arg_type(intrinsic: &sir::Intrinsic, ty: &Type) -> Result<ValueTy> {
    ty.value_ty().ok_or_else(|| {
        CompileError::new(ErrorKind::InvalidOperands(format!(
            "`{intrinsic}` does not support argument type `{ty}`",
        )))
    })
}

fn classical_intrinsic_error(err: oqi_classical::Error) -> CompileError {
    CompileError::new(ErrorKind::InvalidOperands(err.to_string()))
}

fn intrinsic_arity_error(
    intrinsic: &sir::Intrinsic,
    expected: usize,
    actual: usize,
) -> CompileError {
    CompileError::new(ErrorKind::InvalidOperands(format!(
        "`{intrinsic}` expects {expected} argument(s), got {actual}",
    )))
}

fn validate_cast(from: &Type, to: &Type, span: oqi_lex::Span) -> Result<()> {
    if from == to {
        return Ok(());
    }
    match (from.value_ty(), to.value_ty()) {
        (Some(from_ty), Some(to_ty)) => from_ty.cast(to_ty).map(|_| ()).map_err(|_| {
            CompileError::at(
                ErrorKind::TypeMismatch {
                    expected: to.to_string(),
                    got: from.to_string(),
                },
                span,
            )
        }),
        _ => Err(CompileError::at(
            ErrorKind::TypeMismatch {
                expected: to.to_string(),
                got: from.to_string(),
            },
            span,
        )),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::FloatWidth;
    use crate::classical::PrimitiveTy;
    use crate::resolve::DefaultIncludeResolver;
    use crate::symbol::SymbolKind;
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[derive(Debug, Default)]
    struct TestIncludeResolver {
        files: HashMap<PathBuf, String>,
    }

    impl TestIncludeResolver {
        fn new(files: impl IntoIterator<Item = (PathBuf, String)>) -> Self {
            Self {
                files: files.into_iter().collect(),
            }
        }
    }

    impl IncludeResolver for TestIncludeResolver {
        fn resolve_path(
            &self,
            path: &Path,
        ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
            self.files
                .get(path)
                .map(|s| Cow::Borrowed(s.as_str()))
                .ok_or_else(|| format!("not found: {}", path.display()).into())
        }
    }

    fn compile_inline(source: &str) -> Result<sir::Program> {
        compile_to_sir(
            source,
            &DefaultIncludeResolver,
            &CompileOptions::default(),
            None,
        )
    }

    fn typed_expr(ty: Type) -> sir::Expr {
        sir::Expr {
            kind: sir::ExprKind::Literal(Primitive::bit(false)),
            ty,
            span: oqi_lex::Span::default(),
        }
    }

    #[test]
    fn unsized_int_literal_collapses_to_target_width() {
        // A bare integer literal lowers to an unsized type and adopts the
        // target's width on collapse — including narrowing to a bit register.
        let lit = sir::Expr {
            kind: sir::ExprKind::Literal(Primitive::int(5)),
            ty: Type::Unsized(UnsizedType::Int),
            span: oqi_lex::Span::default(),
        };
        let to_u8 = collapse_to(lit.clone(), &Type::from(PrimitiveTy::Uint(iw(8))));
        assert_eq!(to_u8.ty, Type::from(PrimitiveTy::Uint(iw(8))));

        let to_bits = collapse_to(lit, &Type::from(PrimitiveTy::BitReg(4)));
        assert_eq!(to_bits.ty, Type::from(PrimitiveTy::BitReg(4)));
        match to_bits.kind {
            sir::ExprKind::Literal(Primitive::BitReg(r)) => assert_eq!(r.as_u128(), 5),
            _ => panic!("expected a bit register literal"),
        }
    }

    #[test]
    fn for_loop_bound_collapses_to_loop_var_width() {
        // The synthesized counter is fed from range bounds that must adopt the
        // loop variable's declared width, not the literal's carrier width.
        let program = compile_inline("for uint[32] i in [0:3] { }").expect("compile");
        let for_stmt = program
            .body
            .iter()
            .find_map(|s| match &s.kind {
                sir::StmtKind::For(f) => Some(f),
                _ => None,
            })
            .expect("a for statement");
        let sir::ForIterable::Range(sir::RangeExpr { start, end, .. }) = &for_stmt.iterable else {
            panic!("expected a range iterable");
        };
        let want = Type::from(PrimitiveTy::Uint(iw(32)));
        assert_eq!(start.as_ref().expect("start").ty, want);
        assert_eq!(end.as_ref().expect("end").ty, want);
    }

    #[test]
    fn compile_error_in_root_stmt_has_source_path() {
        let path = Path::new("/project/main.qasm");
        let err = match compile_to_sir(
            "missing_symbol;",
            &DefaultIncludeResolver,
            &CompileOptions {
                source_name: Some(path.to_path_buf()),
                ..Default::default()
            },
            None,
        ) {
            Ok(_) => panic!("expected lowering to fail"),
            Err(err) => err,
        };
        assert_eq!(err.path.as_deref(), Some(path));
    }

    #[test]
    fn compile_error_in_nested_include_uses_included_path() {
        let path = Path::new("/project/main.qasm");
        let include_resolver = TestIncludeResolver::new([
            (
                PathBuf::from("/project/file/1/path"),
                "include \"../2/path\";".to_string(),
            ),
            (
                PathBuf::from("/project/file/2/path"),
                "missing_symbol;".to_string(),
            ),
        ]);

        let err = match compile_to_sir(
            "include \"file/1/path\";",
            &include_resolver,
            &CompileOptions {
                source_name: Some(path.to_path_buf()),
                ..Default::default()
            },
            None,
        ) {
            Ok(_) => panic!("expected nested include lowering to fail"),
            Err(err) => err,
        };

        assert_eq!(err.path.as_deref(), Some(Path::new("/project/file/2/path")));
        assert!(matches!(err.kind, ErrorKind::UndefinedName(ref name) if name == "missing_symbol"));
    }

    #[test]
    fn compile_teleport() {
        let source = include_str!("../../../fixtures/qasm/teleport.qasm");
        let program = compile_inline(source).expect("teleport should compile");

        assert_eq!(program.version.as_deref(), Some("3"));

        // 1 user gate (post), stdgates are hoisted
        let user_gates: Vec<_> = program
            .gates
            .iter()
            .filter(|g| program.symbols.get(g.symbol).name == "post")
            .collect();
        assert_eq!(user_gates.len(), 1);
        assert!(user_gates[0].body.is_empty());

        // stdgates should also be present
        assert!(program.gates.len() > 1);

        // Check symbols exist
        assert!(program.symbols.lookup("q").is_some());
        assert!(program.symbols.lookup("c0").is_some());
        assert!(program.symbols.lookup("c1").is_some());
        assert!(program.symbols.lookup("c2").is_some());
    }

    #[test]
    fn compile_adder() {
        let source = include_str!("../../../fixtures/qasm/adder.qasm");
        let program = compile_inline(source).expect("adder should compile");

        // Custom gates: majority and unmaj
        let gate_names: Vec<_> = program
            .gates
            .iter()
            .map(|g| program.symbols.get(g.symbol).name.as_str())
            .collect();
        assert!(gate_names.contains(&"majority"));
        assert!(gate_names.contains(&"unmaj"));

        // For loops should be present in the body
        let has_for = program
            .body
            .iter()
            .any(|s| matches!(s.kind, sir::StmtKind::For(_)));
        assert!(has_for);

        // Symbol table has key variables
        assert!(program.symbols.lookup("cin").is_some());
        assert!(program.symbols.lookup("a").is_some());
        assert!(program.symbols.lookup("b").is_some());
        assert!(program.symbols.lookup("cout").is_some());
        assert!(program.symbols.lookup("ans").is_some());
    }

    #[test]
    fn compile_stdgates() {
        // Compile stdgates content standalone (U is pre-seeded by Resolver)
        let source = include_str!("../stdgates.inc");
        let program = compile_inline(source).expect("stdgates should compile");

        // Should have 30+ gate declarations
        assert!(
            program.gates.len() >= 30,
            "expected 30+ gates, got {}",
            program.gates.len()
        );

        // Check some specific gates exist
        let gate_names: Vec<_> = program
            .gates
            .iter()
            .map(|g| program.symbols.get(g.symbol).name.as_str())
            .collect();
        assert!(gate_names.contains(&"h"));
        assert!(gate_names.contains(&"cx"));
        assert!(gate_names.contains(&"ccx"));

        // gphase calls should be present in gate bodies
        let has_gphase = program.gates.iter().any(|g| {
            g.body.iter().any(|s| {
                matches!(
                    &s.kind,
                    sir::StmtKind::GateCall(sir::GateCall { gate, .. })
                        if program.symbols.get(*gate).name == "gphase"
                )
            })
        });
        assert!(has_gphase);
    }

    #[test]
    fn compile_empty_program() {
        let program = compile_inline("").expect("empty source should compile");
        assert!(program.body.is_empty());
        assert!(program.version.is_none());
    }

    #[test]
    fn compile_version_preserved() {
        let program = compile_inline("OPENQASM 3.0;").expect("version-only should compile");
        assert_eq!(program.version.as_deref(), Some("3.0"));
    }

    #[test]
    fn literal_parsing() {
        let source = r#"
            int[32] a = 42;
            int[32] b = 0xFF;
            int[32] c = 0b1010;
            int[32] d = 0o77;
            float[64] e = 3.14;
            bool f = true;
            bit[4] g = "0110";
        "#;
        let program = compile_inline(source).expect("literals should compile");
        // All declarations should be present
        assert!(program.symbols.lookup("a").is_some());
        assert!(program.symbols.lookup("b").is_some());
        assert!(program.symbols.lookup("c").is_some());
        assert!(program.symbols.lookup("d").is_some());
        assert!(program.symbols.lookup("e").is_some());
        assert!(program.symbols.lookup("f").is_some());
        assert!(program.symbols.lookup("g").is_some());
    }

    #[test]
    fn scope_flattening() {
        let source = r#"
            OPENQASM 3;
            int[32] x = 1;
            {
                int[32] y = 2;
            }
        "#;
        let program = compile_inline(source).expect("scope should compile");
        // Both decls should appear in the flattened body
        assert_eq!(program.body.len(), 2);
    }

    #[test]
    fn scopes_are_attached_to_symbols() {
        let source = r#"
            OPENQASM 3;
            int[32] g = 1;
            def f(int[32] p) -> int[32] {
                int[32] s = 2;
                for uint i in [0:3] {
                    if (p > 0) {
                        int[32] t = 3;
                    }
                }
                return s;
            }
        "#;
        let program = compile_inline(source).expect("source should compile");

        // Globals (`g`, `f`) carry no scope
        let g = program.symbols.lookup("g").unwrap();
        let f = program.symbols.lookup("f").unwrap();
        assert_eq!(program.symbols.get(g).scope, None);
        assert_eq!(program.symbols.get(f).scope, None);

        // Subroutine param `p` lives in the subroutine's scope
        let p = program
            .symbols
            .iter()
            .find(|s| s.name == "p" && s.kind == SymbolKind::SubroutineParam)
            .unwrap();
        let p_scope = p.scope.expect("p must be scoped");

        // `s` is also in the subroutine's scope (same one as `p`)
        let s = program.symbols.iter().find(|sym| sym.name == "s").unwrap();
        assert_eq!(s.scope, Some(p_scope));

        // Loop variable `i` is in its own scope inside the subroutine
        let i = program
            .symbols
            .iter()
            .find(|sym| sym.name == "i" && sym.kind == SymbolKind::LoopVar)
            .unwrap();
        let for_scope = i.scope.expect("i must be scoped");
        assert_ne!(for_scope, p_scope);

        // `t` is in the `if` body's scope, distinct again
        let t = program.symbols.iter().find(|sym| sym.name == "t").unwrap();
        let if_scope = t.scope.expect("t must be scoped");
        assert_ne!(if_scope, for_scope);
        assert_ne!(if_scope, p_scope);
    }

    #[test]
    fn old_style_decls() {
        let source = r#"
            creg c[4];
            qreg q[2];
        "#;
        let program = compile_inline(source).expect("old style should compile");
        let c_sym = program.symbols.lookup("c").expect("c should exist");
        let q_sym = program.symbols.lookup("q").expect("q should exist");
        assert_eq!(program.symbols.get(c_sym).kind, SymbolKind::Variable);
        assert_eq!(program.symbols.get(q_sym).kind, SymbolKind::Qubit);
        assert_eq!(
            program.symbols.get(c_sym).ty,
            Type::Classical(ValueTy::bitreg(4))
        );
        assert_eq!(program.symbols.get(q_sym).ty, Type::QubitReg(2));
    }

    #[test]
    fn include_stdgates_resolves_gates() {
        let source = r#"
            include "stdgates.inc";
            qubit q;
            h q;
        "#;
        let program = compile_inline(source).expect("should compile with stdgates");
        // h should resolve to a gate call
        let has_gate_call = program
            .body
            .iter()
            .any(|s| matches!(&s.kind, sir::StmtKind::GateCall(_)));
        assert!(has_gate_call);
    }

    #[test]
    fn undeclared_name_errors() {
        let source = r#"
            qubit q;
            h q;
        "#;
        match compile_inline(source) {
            Err(e) => assert!(matches!(e.kind, ErrorKind::UndefinedName(ref n) if n == "h")),
            Ok(_) => panic!("expected error for undeclared name"),
        }
    }

    // ── Phase 5: Type inference tests ───────────────────────────────────

    fn find_expr_stmt(program: &sir::Program) -> &sir::Expr {
        for stmt in &program.body {
            if let sir::StmtKind::ExprStmt(e) = &stmt.kind {
                return e;
            }
        }
        panic!("no ExprStmt found in program body");
    }

    #[test]
    fn type_int_plus_float() {
        let source = "int[32] x = 1; float[64] y = 2.0; x + y;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(
            e.ty,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            "x + y should be Float"
        );
    }

    #[test]
    fn type_bool_and() {
        let source = "true && false;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(
            e.ty,
            Type::Classical(ValueTy::bool()),
            "true && false should be Bool"
        );
    }

    #[test]
    fn type_comparison_is_bool() {
        let source = "int[32] x = 1; x == 2;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(
            e.ty,
            Type::Classical(ValueTy::bool()),
            "x == 2 should be Bool"
        );
    }

    #[test]
    fn type_var_inherits_declared_type() {
        let source = "float[64] x = 1.0; x;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::float(FloatWidth::F64)));
    }

    #[test]
    fn type_unary_neg_preserves_type() {
        let source = "float[64] x = 1.0; -x;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::float(FloatWidth::F64)));
    }

    #[test]
    fn type_logical_not_is_bool() {
        let source = "bool x = true; !x;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::bool()));
    }

    #[test]
    fn type_cast_result() {
        let source = "int[32] x = 5; float[64](x);";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::float(FloatWidth::F64)));
    }

    #[test]
    fn type_cast_angle_to_float_invalid() {
        // docs/types.rst, "Allowed casts": angle -> float is "No".
        let Err(err) = compile_inline("angle[32] a; float[64](a);") else {
            panic!("angle -> float is not an allowed cast");
        };
        assert!(matches!(err.kind, ErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn type_cast_float_to_angle_valid() {
        let source = "float[64] f = 1.0; angle[32](f);";
        let program = compile_inline(source).expect("float → angle should be valid");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::angle(iw(32))));
    }

    #[test]
    fn type_index_into_qubit_reg() {
        let source = "qubit[4] q; q[0];";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Qubit);
    }

    #[test]
    fn type_index_into_int() {
        let source = "uint[4] x = 5; x[0];";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::bit()));
    }

    #[test]
    fn type_angle_arithmetic() {
        // angle + angle → angle
        let source = "angle[32] a; angle[32] b; a + b;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::angle(iw(32))));
    }

    #[test]
    fn type_angle_div_angle() {
        // angle / angle → uint
        let source = "angle[32] a; angle[32] b; a / b;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::uint(iw(32))));
    }

    #[test]
    fn type_angle_mul_uint() {
        // angle * uint → angle
        let source = "angle[32] a; uint[32] n = 2; a * n;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Classical(ValueTy::angle(iw(32))));
    }

    #[test]
    fn type_add_signed_unsigned_same_width_uses_classical_promotion() {
        let result = binary_result_type(
            &sir::BinOp::Add,
            &Type::Classical(ValueTy::int(iw(32))),
            &Type::Classical(ValueTy::uint(iw(32))),
            oqi_lex::Span::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::uint(iw(32))));
    }

    #[test]
    fn type_add_float_complex_uses_classical_promotion() {
        let result = binary_result_type(
            &sir::BinOp::Add,
            &Type::Classical(ValueTy::float(FloatWidth::F64)),
            &Type::Classical(ValueTy::complex(FloatWidth::F32)),
            oqi_lex::Span::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::complex(FloatWidth::F64)));
    }

    #[test]
    fn type_add_bool_int_is_rejected() {
        let result = binary_result_type(
            &sir::BinOp::Add,
            &Type::Classical(ValueTy::bool()),
            &Type::Classical(ValueTy::int(iw(32))),
            oqi_lex::Span::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejected_binary_operator_names_the_operation() {
        // Both operands have the same type, so "expected X, got X" would
        // say nothing.
        let source = "complex a; bool b = a < a;";
        let err = compile_inline(source)
            .err()
            .expect("complex numbers are not ordered");
        assert_eq!(
            err.to_string(),
            "`complex[float[64]] < complex[float[64]]` is not defined"
        );
        assert_eq!(&source[err.span.unwrap()], "a < a");

        let err = compile_inline("qubit q; q + q;")
            .err()
            .expect("qubits are not numbers");
        assert_eq!(err.to_string(), "`qubit + qubit` is not defined");
    }

    #[test]
    fn type_intrinsic_ceiling_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Unary(sir::UnOp::Ceiling),
            &[typed_expr(Type::Classical(ValueTy::float(FloatWidth::F32)))],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::float(FloatWidth::F32)));
    }

    #[test]
    fn type_intrinsic_sin_angle_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Unary(sir::UnOp::Sin),
            &[typed_expr(Type::Classical(ValueTy::angle(iw(32))))],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::float(FloatWidth::F64)));
    }

    #[test]
    fn type_intrinsic_mod_uses_classical_promotion() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Binary(sir::BinOp::Mod),
            &[
                typed_expr(Type::Classical(ValueTy::uint(iw(8)))),
                typed_expr(Type::Classical(ValueTy::uint(iw(16)))),
            ],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::uint(iw(16))));
    }

    #[test]
    fn type_intrinsic_popcount_uses_input_width() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Unary(sir::UnOp::Popcount),
            &[typed_expr(Type::Classical(ValueTy::bitreg(8)))],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::uint(iw(8))));
    }

    #[test]
    fn type_intrinsic_real_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Unary(sir::UnOp::Real),
            &[typed_expr(Type::Classical(ValueTy::complex(
                FloatWidth::F32,
            )))],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::float(FloatWidth::F32)));
    }

    #[test]
    fn type_intrinsic_imag_uses_classical_promotion() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Unary(sir::UnOp::Imag),
            &[typed_expr(Type::Classical(ValueTy::int(iw(8))))],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::float(FloatWidth::F64)));
    }

    #[test]
    fn type_intrinsic_sizeof_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Sizeof,
            &[typed_expr(Type::Classical(ValueTy::array(
                crate::classical::PrimitiveTy::Uint(crate::classical::iw(8)),
                crate::classical::ashape(vec![2, 3]),
            )))],
            Default::default(),
        )
        .unwrap();
        assert_eq!(result, Type::Classical(ValueTy::uint(iw(64))));
    }

    #[test]
    fn type_intrinsic_sizeof_rejects_invalid_literal_dimension() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Sizeof,
            &[
                typed_expr(Type::Classical(ValueTy::array(
                    crate::classical::PrimitiveTy::Uint(crate::classical::iw(8)),
                    crate::classical::ashape(vec![2, 3]),
                ))),
                sir::Expr {
                    kind: sir::ExprKind::Literal(Primitive::int(3)),
                    ty: Type::Classical(ValueTy::int(iw(64))),
                    span: oqi_lex::Span::default(),
                },
            ],
            Default::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn type_intrinsic_sizeof_rejects_invalid_dimension_type() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Sizeof,
            &[
                typed_expr(Type::Classical(ValueTy::array(
                    crate::classical::PrimitiveTy::Uint(crate::classical::iw(8)),
                    crate::classical::ashape(vec![2, 3]),
                ))),
                typed_expr(Type::Classical(ValueTy::angle(iw(8)))),
            ],
            Default::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn type_intrinsic_invalid_argument_is_rejected() {
        let err = match compile_inline("sin(true);") {
            Ok(_) => panic!("expected intrinsic type error"),
            Err(err) => err,
        };
        assert!(matches!(err.kind, ErrorKind::InvalidOperands(_)));
    }

    #[test]
    fn type_index_into_array_returns_sub_array() {
        let index = sir::IndexOp {
            kind: sir::IndexKind::Items(vec![sir::IndexItem::Single(Box::new(sir::Expr {
                kind: sir::ExprKind::Literal(Primitive::int(0)),
                ty: Type::Classical(ValueTy::int(iw(64))),
                span: oqi_lex::Span::default(),
            }))]),
            span: oqi_lex::Span::default(),
        };
        let result = index_result_type(
            &Type::Classical(ValueTy::array(
                crate::classical::PrimitiveTy::Uint(crate::classical::iw(8)),
                crate::classical::ashape(vec![2, 3]),
            )),
            &index,
        );
        assert_eq!(
            result.unwrap(),
            Type::Classical(ValueTy::array(
                crate::classical::PrimitiveTy::Uint(crate::classical::iw(8)),
                crate::classical::ashape(vec![3]),
            ))
        );
    }

    #[test]
    fn type_teleport_condition_is_bool() {
        let source = include_str!("../../../fixtures/qasm/teleport.qasm");
        let program = compile_inline(source).expect("teleport should compile");
        // Find an if-statement and check its condition type
        let if_stmt = program
            .body
            .iter()
            .find(|s| matches!(s.kind, sir::StmtKind::If(_)));
        assert!(if_stmt.is_some());
        if let sir::StmtKind::If(sir::If { condition, .. }) = &if_stmt.unwrap().kind {
            assert_eq!(
                condition.ty,
                Type::Classical(ValueTy::bool()),
                "if condition should be Bool"
            );
        }
    }

    #[test]
    fn type_adder_cast_is_bool() {
        // adder.qasm uses bool(a_in[i]) - verify cast produces Bool
        let source = include_str!("../../../fixtures/qasm/adder.qasm");
        let program = compile_inline(source).expect("adder should compile");
        // Find a for loop with if(bool(...))
        let for_stmt = program
            .body
            .iter()
            .find(|s| matches!(s.kind, sir::StmtKind::For(_)));
        assert!(for_stmt.is_some());
        if let sir::StmtKind::For(sir::For { body, .. }) = &for_stmt.unwrap().kind {
            let if_stmt = body.iter().find(|s| matches!(s.kind, sir::StmtKind::If(_)));
            assert!(if_stmt.is_some());
            if let sir::StmtKind::If(sir::If { condition, .. }) = &if_stmt.unwrap().kind {
                assert_eq!(
                    condition.ty,
                    Type::Classical(ValueTy::bool()),
                    "bool() cast should yield Bool"
                );
            }
        }
    }

    #[test]
    fn empty_array_literal_is_rejected() {
        let err = match compile_to_sir(
            "array[int[32], 1] a = {};",
            &DefaultIncludeResolver,
            &CompileOptions::default(),
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("empty array literal should be rejected"),
        };
        assert!(matches!(err.kind, ErrorKind::InvalidLiteral(_)));
    }

    #[test]
    fn overflowing_hardware_qubit_is_rejected() {
        let err = match compile_to_sir(
            "reset $99999999999999999999999999;",
            &DefaultIncludeResolver,
            &CompileOptions::default(),
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("overflowing hardware qubit index should be rejected"),
        };
        assert!(matches!(err.kind, ErrorKind::InvalidLiteral(_)));
    }

    // ── Gate-body loop unrolling / branch rejection ─────────────────────

    fn gate_body<'a>(program: &'a sir::Program, name: &str) -> &'a [sir::Stmt] {
        let g = program
            .gates
            .iter()
            .find(|g| program.symbols.get(g.symbol).name == name)
            .expect("gate should exist");
        &g.body
    }

    #[test]
    fn gate_for_loop_is_unrolled() {
        let program = compile_inline("gate g q { for uint i in [0:2] { U(0, 0, 0) q; } }").unwrap();
        let body = gate_body(&program, "g");
        assert_eq!(body.len(), 3);
        assert!(
            body.iter()
                .all(|s| matches!(s.kind, sir::StmtKind::GateCall(_)))
        );
    }

    #[test]
    fn gate_nested_for_loops_are_unrolled() {
        let program = compile_inline(
            "gate g q { for uint i in [0:1] { for uint j in [0:2] { U(0, 0, 0) q; } } }",
        )
        .unwrap();
        // 2 * 3 = 6 unrolled gate calls, no control flow.
        assert_eq!(gate_body(&program, "g").len(), 6);
    }

    #[test]
    fn gate_empty_range_unrolls_to_identity() {
        let program = compile_inline("gate g q { for int i in [0:-1] { U(0, 0, 0) q; } }").unwrap();
        assert!(gate_body(&program, "g").is_empty());
    }

    #[test]
    fn gate_set_iterable_is_unrolled() {
        let program =
            compile_inline("gate g q { for uint i in {1, 3, 5} { U(0, 0, 0) q; } }").unwrap();
        assert_eq!(gate_body(&program, "g").len(), 3);
    }

    fn expect_invalid_gate_body(source: &str) {
        match compile_inline(source) {
            Err(e) => assert!(
                matches!(e.kind, ErrorKind::InvalidGateBody(_)),
                "expected InvalidGateBody, got {:?}",
                e.kind
            ),
            Ok(_) => panic!("expected `{source}` to be rejected"),
        }
    }

    #[test]
    fn if_in_gate_body_is_rejected() {
        expect_invalid_gate_body("gate g q { if (true) { U(0, 0, 0) q; } }");
    }

    #[test]
    fn while_in_gate_body_is_rejected() {
        expect_invalid_gate_body("gate g q { while (true) { U(0, 0, 0) q; } }");
    }

    #[test]
    fn switch_in_gate_body_is_rejected() {
        expect_invalid_gate_body("gate g q { switch (1) { case 1 { U(0, 0, 0) q; } } }");
    }

    #[test]
    fn nonconst_loop_bound_in_gate_body_is_rejected() {
        expect_invalid_gate_body("gate g(theta) q { for uint i in [0:theta] { U(0, 0, 0) q; } }");
    }

    #[test]
    fn unrolled_gate_emits_single_block() {
        let program = compile_inline("gate g q { for uint i in [0:2] { U(0, 0, 0) q; } }").unwrap();
        let module = crate::bytecode::emit(&program).unwrap();
        let proc = module
            .procedures
            .iter()
            .find(|p| {
                matches!(&p.owner, crate::bytecode::ProcOwner::Gate(s)
                    if program.symbols.get(*s).name == "g")
            })
            .expect("gate procedure should exist");
        // No loops or branches survive: every block ends in straight-line
        // control flow (Goto/Return), never a conditional Branch/Switch.
        assert!(proc.blocks.iter().all(|b| matches!(
            b.terminator,
            crate::bytecode::Terminator::Goto(_)
                | crate::bytecode::Terminator::Return(_)
                | crate::bytecode::Terminator::End
        )));
        // The three unrolled U calls land in a single straight-line block.
        let gate_calls: usize = proc
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .filter(|i| matches!(i.op, crate::bytecode::Op::GateCall { .. }))
            .count();
        assert_eq!(gate_calls, 3);
    }
}
