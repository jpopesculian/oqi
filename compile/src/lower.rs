use std::path::{Path, PathBuf};

use oqi_classical::{
    Index as ClassicalIndex,
    ops::{
        Add as ClassicalAdd, Arccos as ClassicalArccos, Arcsin as ClassicalArcsin,
        Arctan as ClassicalArctan, BinOp as ClassicalBinOp, BitAnd as ClassicalBitAnd,
        BitNot as ClassicalBitNot, BitOr as ClassicalBitOr, BitXor as ClassicalBitXor,
        Ceiling as ClassicalCeiling, Cos as ClassicalCos, Div as ClassicalDiv, Eq as ClassicalEq,
        Exp as ClassicalExp, Floor as ClassicalFloor, Gt as ClassicalGt, Gte as ClassicalGte,
        Imag as ClassicalImag, Log as ClassicalLog, LogAnd as ClassicalLogAnd,
        LogNot as ClassicalLogNot, LogOr as ClassicalLogOr, Lt as ClassicalLt, Lte as ClassicalLte,
        Mul as ClassicalMul, Neg as ClassicalNeg, Neq as ClassicalNeq,
        Popcount as ClassicalPopcount, Pow as ClassicalPow, Real as ClassicalReal,
        Rem as ClassicalRem, Rotl as ClassicalRotl, Rotr as ClassicalRotr, Shl as ClassicalShl,
        Shr as ClassicalShr, Sin as ClassicalSin, Sqrt as ClassicalSqrt, Sub as ClassicalSub,
        Tan as ClassicalTan, UnOp as ClassicalUnOp,
    },
};
use oqi_parse::ast;

use crate::classical::{
    ValueTy, bitreg_value, bool_value, complex_value, duration_value, float_value, int_value,
    value_as_usize,
};
use crate::error::{CompileError, ErrorKind, Result, ResultExt};
use crate::resolve::{FileResolver, IncludeSource, Resolver, StdFileResolver};
use crate::sir;
use crate::symbol::SymbolKind;
use crate::types::{
    CompileOptions, Type, eval_const_expr, eval_designator, float_width_from_system_width,
    parse_bitstring_literal, parse_float_literal, parse_imag_literal, parse_int_literal,
    parse_timing_literal, resolve_array_ref_type, resolve_old_style_type, resolve_qubit_type,
    resolve_scalar_type, resolve_type,
};

// ── Public API ──────────────────────────────────────────────────────────

pub fn compile_ast(
    program: &ast::Program<'_>,
    file_resolver: impl FileResolver + 'static,
    options: CompileOptions,
) -> Result<sir::Program> {
    let mut lowerer = Lowerer::new(file_resolver, options);
    lowerer.lower_program(program)?;
    Ok(lowerer.finish(program))
}

pub fn compile_source(
    source: &str,
    file_resolver: impl FileResolver + 'static,
    source_name: Option<&Path>,
) -> Result<sir::Program> {
    let ast = oqi_parse::parse(source).map_err(|e| {
        CompileError::new(ErrorKind::Unsupported(format!("parse error: {e:?}")))
            .with_path(source_name.map(Path::to_path_buf))
    })?;
    let options = CompileOptions {
        source_name: source_name.map(|p| p.to_path_buf()),
        ..Default::default()
    };
    compile_ast(&ast, file_resolver, options)
}

pub fn compile_file(path: &Path) -> Result<sir::Program> {
    let file_resolver = StdFileResolver;
    let source = file_resolver.resolve_source(path, Default::default())?;
    compile_source(&source, file_resolver, Some(path))
}

// ── Lowerer ─────────────────────────────────────────────────────────────

struct Lowerer {
    resolver: Resolver,
    gates: Vec<sir::GateDecl>,
    subroutines: Vec<sir::SubroutineDecl>,
    externs: Vec<sir::ExternDecl>,
    calibrations: Vec<sir::CalibrationDecl>,
    calibration_grammar: Option<String>,
    body: Vec<sir::Stmt>,
}

impl Lowerer {
    fn new(file_resolver: impl FileResolver + 'static, options: CompileOptions) -> Self {
        Self {
            resolver: Resolver::new(file_resolver, options),
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
        for item in &program.body {
            let stmts = self.lower_stmt_or_scope(item)?;
            self.body.extend(stmts);
        }
        Ok(())
    }

    fn with_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.resolver.push_scope();
        let result = f(self);
        self.resolver.pop_scope();
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
            ast::StmtOrScope::Scope(scope) => self.with_scope(|this| {
                let mut stmts = Vec::new();
                for item in &scope.body {
                    stmts.extend(this.lower_stmt_or_scope(item)?);
                }
                Ok(stmts)
            }),
        };

        result.with_path(current_path)
    }

    fn lower_body(&mut self, item: &ast::StmtOrScope<'_>) -> Result<Vec<sir::Stmt>> {
        self.with_scope(|this| match item {
            ast::StmtOrScope::Stmt(stmt) => this.lower_stmt(stmt),
            ast::StmtOrScope::Scope(scope) => {
                let mut stmts = Vec::new();
                for item in &scope.body {
                    stmts.extend(this.lower_stmt_or_scope(item)?);
                }
                Ok(stmts)
            }
        })
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt<'_>) -> Result<Vec<sir::Stmt>> {
        let annotations = self.lower_annotations(&stmt.annotations);
        let span = stmt.span;

        let stmts = match &stmt.kind {
            ast::StmtKind::Include(path) => {
                return self.lower_include(path, span);
            }

            ast::StmtKind::ClassicalDecl { ty, name, init } => {
                let resolved_ty =
                    resolve_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let symbol = self.resolver.declare(
                    name.name,
                    SymbolKind::Variable,
                    resolved_ty,
                    name.span,
                )?;
                let init = match init {
                    Some(d) => Some(self.lower_decl_init(d)?),
                    None => None,
                };
                vec![sir::Stmt {
                    kind: sir::StmtKind::ClassicalDecl { symbol, init },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::ConstDecl { ty, name, init } => {
                let resolved_ty =
                    resolve_scalar_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let init_expr = match init {
                    ast::DeclExpr::Expr(e) => self.lower_expr(e)?,
                    _ => {
                        return Err(CompileError::new(ErrorKind::InvalidContext(
                            "const initializer must be an expression".into(),
                        ))
                        .with_span(span));
                    }
                };
                let const_val = eval_const_expr(
                    match init {
                        ast::DeclExpr::Expr(e) => e,
                        _ => unreachable!(),
                    },
                    self.resolver.symbols(),
                    self.resolver.options(),
                )
                .ok();
                let symbol =
                    self.resolver
                        .declare(name.name, SymbolKind::Const, resolved_ty, name.span)?;
                if let Some(cv) = const_val {
                    self.resolver.symbols_mut().set_const_value(symbol, cv);
                }
                vec![sir::Stmt {
                    kind: sir::StmtKind::ConstDecl {
                        symbol,
                        init: init_expr,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::QuantumDecl { ty, name } => {
                let resolved_ty =
                    resolve_qubit_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let symbol =
                    self.resolver
                        .declare(name.name, SymbolKind::Qubit, resolved_ty, name.span)?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::QubitDecl { symbol },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::OldStyleDecl {
                keyword,
                name,
                designator,
            } => {
                let resolved_ty = resolve_old_style_type(
                    keyword,
                    designator.as_deref(),
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                let (kind, stmt_kind) = match keyword {
                    ast::OldStyleKind::Creg => {
                        let sym_kind = SymbolKind::Variable;
                        let symbol =
                            self.resolver
                                .declare(name.name, sym_kind, resolved_ty, name.span)?;
                        (
                            sym_kind,
                            sir::StmtKind::ClassicalDecl { symbol, init: None },
                        )
                    }
                    ast::OldStyleKind::Qreg => {
                        let sym_kind = SymbolKind::Qubit;
                        let symbol =
                            self.resolver
                                .declare(name.name, sym_kind, resolved_ty, name.span)?;
                        (sym_kind, sir::StmtKind::QubitDecl { symbol })
                    }
                };
                let _ = kind;
                vec![sir::Stmt {
                    kind: stmt_kind,
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::IoDecl { dir, ty, name } => {
                let resolved_ty =
                    resolve_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let sym_kind = match dir {
                    ast::IoDir::Input => SymbolKind::Input,
                    ast::IoDir::Output => SymbolKind::Output,
                };
                let symbol = self
                    .resolver
                    .declare(name.name, sym_kind, resolved_ty, name.span)?;
                let dir = match dir {
                    ast::IoDir::Input => sir::IoDir::Input,
                    ast::IoDir::Output => sir::IoDir::Output,
                };
                vec![sir::Stmt {
                    kind: sir::StmtKind::IoDecl { symbol, dir },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Gate {
                name,
                params,
                qubits,
                body,
            } => {
                let gate_sym =
                    self.resolver
                        .declare(name.name, SymbolKind::Gate, Type::Void, name.span)?;
                let (param_ids, qubit_ids, gate_body) = self.with_scope(|this| {
                    let angle_width = this.resolver.options().system_angle_width;
                    let param_ids: Vec<_> = params
                        .iter()
                        .map(|p| {
                            this.resolver.declare(
                                p.name,
                                SymbolKind::GateParam,
                                Type::angle(angle_width),
                                p.span,
                            )
                        })
                        .collect::<Result<_>>()?;
                    let qubit_ids: Vec<_> = qubits
                        .iter()
                        .map(|q| {
                            this.resolver.declare(
                                q.name,
                                SymbolKind::GateQubit,
                                Type::Qubit,
                                q.span,
                            )
                        })
                        .collect::<Result<_>>()?;
                    let gate_body = this.lower_gate_body(body)?;
                    Ok((param_ids, qubit_ids, gate_body))
                })?;
                self.gates.push(sir::GateDecl {
                    symbol: gate_sym,
                    params: param_ids,
                    qubits: qubit_ids,
                    body: gate_body,
                    span,
                });
                vec![]
            }

            ast::StmtKind::Def {
                name,
                params,
                return_ty,
                body,
            } => {
                let sub_sym = self.resolver.declare(
                    name.name,
                    SymbolKind::Subroutine,
                    Type::Void,
                    name.span,
                )?;
                let (sir_params, ret_ty, body_stmts) = self.with_scope(|this| {
                    let sir_params = this.lower_arg_defs(params)?;
                    let ret_ty = match return_ty {
                        Some(s) => Some(resolve_scalar_type(
                            s,
                            this.resolver.symbols(),
                            this.resolver.options(),
                        )?),
                        None => None,
                    };
                    if let Some(ref ty) = ret_ty {
                        this.resolver.symbols_mut().get_mut(sub_sym).ty = ty.clone();
                    }
                    let mut body_stmts = Vec::new();
                    for item in &body.body {
                        body_stmts.extend(this.lower_stmt_or_scope(item)?);
                    }
                    Ok((sir_params, ret_ty, body_stmts))
                })?;
                self.subroutines.push(sir::SubroutineDecl {
                    symbol: sub_sym,
                    params: sir_params,
                    return_ty: ret_ty,
                    body: body_stmts,
                    span,
                });
                vec![]
            }

            ast::StmtKind::Extern {
                name,
                params,
                return_ty,
            } => {
                let ext_sym =
                    self.resolver
                        .declare(name.name, SymbolKind::Extern, Type::Void, name.span)?;
                let param_types = self.lower_extern_args(params)?;
                let ret_ty = match return_ty {
                    Some(s) => Some(resolve_scalar_type(
                        s,
                        self.resolver.symbols(),
                        self.resolver.options(),
                    )?),
                    None => None,
                };
                if let Some(ref ty) = ret_ty {
                    self.resolver.symbols_mut().get_mut(ext_sym).ty = ty.clone();
                }
                self.externs.push(sir::ExternDecl {
                    symbol: ext_sym,
                    param_types,
                    return_ty: ret_ty,
                    span,
                });
                vec![]
            }

            ast::StmtKind::Defcal {
                target,
                args,
                operands,
                return_ty,
                body,
            } => {
                let sir_target = match target {
                    ast::DefcalTarget::Measure(_) => sir::CalibrationTarget::Measure,
                    ast::DefcalTarget::Reset(_) => sir::CalibrationTarget::Reset,
                    ast::DefcalTarget::Delay(_) => sir::CalibrationTarget::Delay,
                    ast::DefcalTarget::Ident(id) => {
                        sir::CalibrationTarget::Named(id.name.to_string())
                    }
                };
                let sir_args = args
                    .iter()
                    .map(|a| match a {
                        ast::DefcalArgDef::Expr(e) => {
                            Ok(sir::CalibrationArg::Expr(Box::new(self.lower_expr(e)?)))
                        }
                        ast::DefcalArgDef::ArgDef(ad) => {
                            let (sym, _) = self.lower_single_arg_def(ad)?;
                            Ok(sir::CalibrationArg::Param(sym))
                        }
                    })
                    .collect::<Result<_>>()?;
                let sir_operands = operands
                    .iter()
                    .map(|o| match o {
                        ast::DefcalOperand::HardwareQubit(s, _) => {
                            Ok(sir::CalibrationOperand::Hardware(parse_hardware_qubit(s)))
                        }
                        ast::DefcalOperand::Ident(id) => {
                            Ok(sir::CalibrationOperand::Ident(id.name.to_string()))
                        }
                    })
                    .collect::<Result<_>>()?;
                let ret_ty = match return_ty {
                    Some(s) => Some(resolve_scalar_type(
                        s,
                        self.resolver.symbols(),
                        self.resolver.options(),
                    )?),
                    None => None,
                };
                let sir_body = sir::CalibrationBody::Opaque(body.unwrap_or("").to_string());
                self.calibrations.push(sir::CalibrationDecl {
                    target: sir_target,
                    args: sir_args,
                    operands: sir_operands,
                    return_ty: ret_ty,
                    body: sir_body,
                    span,
                });
                vec![]
            }

            ast::StmtKind::Cal(body) => {
                vec![sir::Stmt {
                    kind: sir::StmtKind::Cal {
                        body: sir::CalibrationBody::Opaque(body.unwrap_or("").to_string()),
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::CalibrationGrammar(grammar) => {
                self.calibration_grammar = Some(grammar.to_string());
                vec![]
            }

            ast::StmtKind::GateCall {
                modifiers,
                name,
                args,
                designator: _,
                operands,
            } => {
                let gate = match name {
                    ast::GateCallName::Ident(id) => {
                        let sym = self.resolver.resolve(id.name, id.span)?;
                        sir::GateCallTarget::Symbol(sym)
                    }
                    ast::GateCallName::Gphase(_) => sir::GateCallTarget::GPhase,
                };
                let sir_mods = modifiers
                    .iter()
                    .map(|m| self.lower_gate_modifier(m))
                    .collect::<Result<_>>()?;
                let sir_args = match args {
                    Some(a) => a
                        .iter()
                        .map(|e| self.lower_expr(e))
                        .collect::<Result<_>>()?,
                    None => vec![],
                };
                let sir_qubits = operands
                    .iter()
                    .map(|o| self.lower_gate_operand(o))
                    .collect::<Result<_>>()?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::GateCall {
                        gate,
                        modifiers: sir_mods,
                        args: sir_args,
                        qubits: sir_qubits,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::MeasureArrow { measure, target } => {
                let sir_measure = self.lower_measure_expr(measure)?;
                let sir_target = match target {
                    Some(id) => Some(self.lower_indexed_ident_to_lvalue(id)?),
                    None => None,
                };
                vec![sir::Stmt {
                    kind: sir::StmtKind::Measure {
                        measure: sir_measure,
                        target: sir_target,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Reset(operand) => {
                let op = self.lower_gate_operand(operand)?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Reset { operand: op },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Barrier(operands) => {
                let ops = operands
                    .iter()
                    .map(|o| self.lower_gate_operand(o))
                    .collect::<Result<_>>()?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Barrier { operands: ops },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Assignment { target, op, value } => {
                let lv = self.lower_indexed_ident_to_lvalue(target)?;
                let sir_op = map_assign_op(op);
                let sir_value = match value {
                    ast::ExprOrMeasure::Expr(e) => {
                        sir::AssignValue::Expr(Box::new(self.lower_expr(e)?))
                    }
                    ast::ExprOrMeasure::Measure(m) => {
                        sir::AssignValue::Measure(self.lower_measure_expr(m)?)
                    }
                };
                vec![sir::Stmt {
                    kind: sir::StmtKind::Assignment {
                        target: lv,
                        op: sir_op,
                        value: sir_value,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::If {
                condition,
                then_body,
                else_body,
            } => {
                let cond = self.lower_expr(condition)?;
                let then_stmts = self.lower_body(then_body)?;
                let else_stmts = match else_body {
                    Some(b) => Some(self.lower_body(b)?),
                    None => None,
                };
                vec![sir::Stmt {
                    kind: sir::StmtKind::If {
                        condition: cond,
                        then_body: then_stmts,
                        else_body: else_stmts,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::For {
                ty,
                var,
                iterable,
                body,
            } => {
                let var_ty =
                    resolve_scalar_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let (var_sym, sir_iterable, body_stmts) = self.with_scope(|this| {
                    let var_sym =
                        this.resolver
                            .declare(var.name, SymbolKind::LoopVar, var_ty, var.span)?;
                    let sir_iterable = this.lower_for_iterable(iterable)?;
                    let body_stmts = match body.as_ref() {
                        ast::StmtOrScope::Stmt(s) => this.lower_stmt(s)?,
                        ast::StmtOrScope::Scope(sc) => {
                            let mut stmts = Vec::new();
                            for item in &sc.body {
                                stmts.extend(this.lower_stmt_or_scope(item)?);
                            }
                            stmts
                        }
                    };
                    Ok((var_sym, sir_iterable, body_stmts))
                })?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::For {
                        var: var_sym,
                        iterable: sir_iterable,
                        body: body_stmts,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::While { condition, body } => {
                let cond = self.lower_expr(condition)?;
                let body_stmts = self.lower_body(body)?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::While {
                        condition: cond,
                        body: body_stmts,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Switch { target, cases } => {
                let tgt = self.lower_expr(target)?;
                let sir_cases = cases
                    .iter()
                    .map(|c| self.lower_switch_case(c))
                    .collect::<Result<_>>()?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Switch {
                        target: tgt,
                        cases: sir_cases,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Break => vec![sir::Stmt {
                kind: sir::StmtKind::Break,
                annotations,
                span,
            }],

            ast::StmtKind::Continue => vec![sir::Stmt {
                kind: sir::StmtKind::Continue,
                annotations,
                span,
            }],

            ast::StmtKind::End => vec![sir::Stmt {
                kind: sir::StmtKind::End,
                annotations,
                span,
            }],

            ast::StmtKind::Return(value) => {
                let ret_val = match value {
                    Some(ast::ExprOrMeasure::Expr(e)) => {
                        Some(sir::ReturnValue::Expr(Box::new(self.lower_expr(e)?)))
                    }
                    Some(ast::ExprOrMeasure::Measure(m)) => {
                        Some(sir::ReturnValue::Measure(self.lower_measure_expr(m)?))
                    }
                    None => None,
                };
                vec![sir::Stmt {
                    kind: sir::StmtKind::Return(ret_val),
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Alias { name, value } => {
                let exprs = value
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                let symbol =
                    self.resolver
                        .declare(name.name, SymbolKind::Alias, Type::Void, name.span)?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Alias {
                        symbol,
                        value: exprs,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Delay {
                designator,
                operands,
            } => {
                let dur = self.lower_expr(designator)?;
                let ops = operands
                    .iter()
                    .map(|o| self.lower_gate_operand(o))
                    .collect::<Result<_>>()?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Delay {
                        duration: dur,
                        operands: ops,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Box { designator, body } => {
                let dur = match designator {
                    Some(e) => Some(self.lower_expr(e)?),
                    None => None,
                };
                let body_stmts = self.with_scope(|this| {
                    let mut body_stmts = Vec::new();
                    for item in &body.body {
                        body_stmts.extend(this.lower_stmt_or_scope(item)?);
                    }
                    Ok(body_stmts)
                })?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Box {
                        duration: dur,
                        body: body_stmts,
                    },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Nop(operands) => {
                let ops = operands
                    .iter()
                    .map(|o| self.lower_gate_operand(o))
                    .collect::<Result<_>>()?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::Nop { operands: ops },
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Pragma(content) => {
                vec![sir::Stmt {
                    kind: sir::StmtKind::Pragma(content.to_string()),
                    annotations,
                    span,
                }]
            }

            ast::StmtKind::Expr(expr) => {
                let e = self.lower_expr(expr)?;
                vec![sir::Stmt {
                    kind: sir::StmtKind::ExprStmt(e),
                    annotations,
                    span,
                }]
            }
        };

        Ok(stmts)
    }

    // ── Include handling ────────────────────────────────────────────────

    fn lower_include(&mut self, path: &str, span: oqi_lex::Span) -> Result<Vec<sir::Stmt>> {
        let path = path.trim_matches('"');
        let source = self.resolver.resolve_include_path(path, span)?;
        match source {
            IncludeSource::Embedded(content) => self.lower_include_source(content, path, span),
            IncludeSource::File(file_path) => self.with_include(file_path, |this, file_path| {
                let content = this.resolver.resolve_source(file_path, span)?;
                this.lower_include_source(&content, path, span)
            }),
        }
    }

    fn lower_include_source(
        &mut self,
        content: &str,
        path: &str,
        span: oqi_lex::Span,
    ) -> Result<Vec<sir::Stmt>> {
        let current_path = self.resolver.current_source_path().map(Path::to_path_buf);
        let ast = oqi_parse::parse(content).map_err(|e| {
            CompileError::new(ErrorKind::Unsupported(format!(
                "parse error in include '{path}': {e:?}"
            )))
            .with_span(span)
            .with_path(current_path)
        })?;
        let mut stmts = Vec::new();
        for item in &ast.body {
            stmts.extend(self.lower_stmt_or_scope(item)?);
        }
        Ok(stmts)
    }

    // ── Expression lowering ─────────────────────────────────────────────

    fn lower_expr(&mut self, expr: &ast::Expr<'_>) -> Result<sir::Expr> {
        let span = expr.span();
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
                let n = parse_hardware_qubit(s);
                Ok(sir::Expr {
                    kind: sir::ExprKind::HardwareQubit(n),
                    ty: Type::PhysicalQubit,
                    span,
                })
            }

            ast::Expr::IntLiteral(s, enc, _) => {
                let int = parse_int_literal(s, *enc).with_span(span)?;
                let ty = Type::int(self.system_width(), true);
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(int_value(int)),
                    ty,
                    span,
                })
            }

            ast::Expr::FloatLiteral(s, _) => {
                let fw = float_width_from_system_width(self.system_width());
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(float_value(
                        parse_float_literal(s).with_span(span)?,
                        fw,
                    )),
                    ty: Type::float(fw),
                    span,
                })
            }

            ast::Expr::ImagLiteral(s, _) => {
                let fw = float_width_from_system_width(self.system_width());
                let (re, im) = parse_imag_literal(s)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(complex_value(re, im, fw)),
                    ty: Type::complex(fw),
                    span,
                })
            }

            ast::Expr::BoolLiteral(b, _) => Ok(sir::Expr {
                kind: sir::ExprKind::Literal(bool_value(*b)),
                ty: Type::bool(),
                span,
            }),

            ast::Expr::BitstringLiteral(s, _) => {
                let (bits, len) = parse_bitstring_literal(s)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(bitreg_value(bits, len)),
                    ty: Type::bitreg(len),
                    span,
                })
            }

            ast::Expr::TimingLiteral(s, _) => {
                let tv = parse_timing_literal(s, &self.resolver.options().dt)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Literal(duration_value(tv)),
                    ty: Type::duration(),
                    span,
                })
            }

            ast::Expr::Paren(inner, _) => self.lower_expr(inner),

            ast::Expr::BinOp {
                left, op, right, ..
            } => {
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let sir_op = map_bin_op(op);
                let ty = binary_result_type(&sir_op, &l.ty, &r.ty, span)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Binary {
                        op: sir_op,
                        left: Box::new(l),
                        right: Box::new(r),
                    },
                    ty,
                    span,
                })
            }

            ast::Expr::UnaryOp { op, operand, .. } => {
                let inner = self.lower_expr(operand)?;
                let sir_op = map_un_op(op);
                let ty = unary_result_type(&sir_op, &inner.ty);
                Ok(sir::Expr {
                    kind: sir::ExprKind::Unary {
                        op: sir_op,
                        operand: Box::new(inner),
                    },
                    ty,
                    span,
                })
            }

            ast::Expr::Index { expr, index, .. } => {
                let base = self.lower_expr(expr)?;
                let idx = self.lower_index_op(index)?;
                let ty = index_result_type(&base.ty, &idx);
                Ok(sir::Expr {
                    kind: sir::ExprKind::Index {
                        base: Box::new(base),
                        index: idx,
                    },
                    ty,
                    span,
                })
            }

            ast::Expr::Call { name, args, .. } => {
                let callee = self.resolver.resolve_call(name.name, name.span)?;
                let sir_args: Vec<sir::Expr> = args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<_>>()?;
                let ty = self.call_result_type(&callee, &sir_args).with_span(span)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::Call {
                        callee,
                        args: sir_args,
                    },
                    ty,
                    span,
                })
            }

            ast::Expr::Cast { ty, operand, .. } => {
                let target_ty = resolve_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let inner = self.lower_expr(operand)?;
                validate_cast(&inner.ty, &target_ty, span)?;
                let result_ty = target_ty.clone();
                Ok(sir::Expr {
                    kind: sir::ExprKind::Cast {
                        target_ty,
                        operand: Box::new(inner),
                    },
                    ty: result_ty,
                    span,
                })
            }

            ast::Expr::DurationOf { scope, .. } => {
                let stmts = self.with_scope(|this| {
                    let mut stmts = Vec::new();
                    for item in &scope.body {
                        stmts.extend(this.lower_stmt_or_scope(item)?);
                    }
                    Ok(stmts)
                })?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::DurationOf(stmts),
                    ty: Type::duration(),
                    span,
                })
            }
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
            ast::GateOperand::HardwareQubit(s, _) => {
                Ok(sir::QubitOperand::Hardware(parse_hardware_qubit(s)))
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
                    Some(e) => {
                        eval_designator(e, self.resolver.symbols(), self.resolver.options())?
                    }
                    None => 1,
                };
                Ok(sir::GateModifier::Ctrl(n))
            }
            ast::GateModifier::NegCtrl(designator, _) => {
                let n = match designator {
                    Some(e) => {
                        eval_designator(e, self.resolver.symbols(), self.resolver.options())?
                    }
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
                Ok(sir::MeasureExpr {
                    kind: sir::MeasureExprKind::Measure { operand: op },
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
                    span: *span,
                })
            }
        }
    }

    fn lower_decl_init(&mut self, d: &ast::DeclExpr<'_>) -> Result<sir::DeclInit> {
        match d {
            ast::DeclExpr::Expr(e) => Ok(sir::DeclInit::Expr(Box::new(self.lower_expr(e)?))),
            ast::DeclExpr::Measure(m) => Ok(sir::DeclInit::Measure(self.lower_measure_expr(m)?)),
            ast::DeclExpr::ArrayLiteral(al) => {
                Ok(sir::DeclInit::ArrayLiteral(self.lower_array_literal(al)?))
            }
        }
    }

    fn lower_array_literal(&mut self, al: &ast::ArrayLiteral<'_>) -> Result<sir::ArrayLiteral> {
        let items = al
            .items
            .iter()
            .map(|item| match item {
                ast::ArrayLiteralItem::Expr(e) => {
                    Ok(sir::ArrayLiteralItem::Expr(Box::new(self.lower_expr(e)?)))
                }
                ast::ArrayLiteralItem::Nested(inner) => Ok(sir::ArrayLiteralItem::Nested(
                    self.lower_array_literal(inner)?,
                )),
            })
            .collect::<Result<_>>()?;
        Ok(sir::ArrayLiteral {
            items,
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

    fn lower_for_iterable(&mut self, it: &ast::ForIterable<'_>) -> Result<sir::ForIterable> {
        match it {
            ast::ForIterable::Set(exprs, _) => {
                let items = exprs
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                Ok(sir::ForIterable::Set(items))
            }
            ast::ForIterable::Range(range, _) => Ok(sir::ForIterable::Range {
                start: range
                    .start
                    .as_ref()
                    .map(|e| self.lower_expr(e).map(Box::new))
                    .transpose()?,
                step: range
                    .step
                    .as_ref()
                    .map(|e| self.lower_expr(e).map(Box::new))
                    .transpose()?,
                end: range
                    .end
                    .as_ref()
                    .map(|e| self.lower_expr(e).map(Box::new))
                    .transpose()?,
            }),
            ast::ForIterable::Expr(e) => Ok(sir::ForIterable::Expr(Box::new(self.lower_expr(e)?))),
        }
    }

    fn lower_switch_case(&mut self, case: &ast::SwitchCase<'_>) -> Result<sir::SwitchCase> {
        match case {
            ast::SwitchCase::Case(labels, scope) => {
                let label_exprs = labels
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                let body = self.with_scope(|this| {
                    let mut body = Vec::new();
                    for item in &scope.body {
                        body.extend(this.lower_stmt_or_scope(item)?);
                    }
                    Ok(body)
                })?;
                Ok(sir::SwitchCase {
                    labels: sir::SwitchLabels::Values(label_exprs),
                    body,
                })
            }
            ast::SwitchCase::Default(scope) => {
                let body = self.with_scope(|this| {
                    let mut body = Vec::new();
                    for item in &scope.body {
                        body.extend(this.lower_stmt_or_scope(item)?);
                    }
                    Ok(body)
                })?;
                Ok(sir::SwitchCase {
                    labels: sir::SwitchLabels::Default,
                    body,
                })
            }
        }
    }

    // ── Gate body validation ────────────────────────────────────────────

    fn lower_gate_body(&mut self, scope: &ast::Scope<'_>) -> Result<sir::GateBody> {
        let mut stmts = Vec::new();
        for item in &scope.body {
            let lowered = self.lower_stmt_or_scope(item)?;
            for s in &lowered {
                validate_gate_stmt(s)?;
            }
            stmts.extend(lowered);
        }
        Ok(sir::GateBody { body: stmts })
    }

    // ── Subroutine param lowering ───────────────────────────────────────

    fn lower_arg_defs(&mut self, params: &[ast::ArgDef<'_>]) -> Result<Vec<sir::SubroutineParam>> {
        params
            .iter()
            .map(|p| {
                let (sym, passing) = self.lower_single_arg_def(p)?;
                Ok(sir::SubroutineParam {
                    symbol: sym,
                    passing,
                })
            })
            .collect()
    }

    fn lower_single_arg_def(
        &mut self,
        arg: &ast::ArgDef<'_>,
    ) -> Result<(crate::symbol::SymbolId, sir::ParamPassing)> {
        match arg {
            ast::ArgDef::Scalar(ty, name) => {
                let resolved =
                    resolve_scalar_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    resolved,
                    name.span,
                )?;
                Ok((sym, sir::ParamPassing::ByValue))
            }
            ast::ArgDef::Qubit(ty, name) => {
                let resolved =
                    resolve_qubit_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    resolved,
                    name.span,
                )?;
                Ok((sym, sir::ParamPassing::QubitRef))
            }
            ast::ArgDef::Creg(name, designator) => {
                let ty = resolve_old_style_type(
                    &ast::OldStyleKind::Creg,
                    designator.as_ref(),
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                let sym =
                    self.resolver
                        .declare(name.name, SymbolKind::SubroutineParam, ty, name.span)?;
                Ok((sym, sir::ParamPassing::ByValue))
            }
            ast::ArgDef::Qreg(name, designator) => {
                let ty = resolve_old_style_type(
                    &ast::OldStyleKind::Qreg,
                    designator.as_ref(),
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                let sym =
                    self.resolver
                        .declare(name.name, SymbolKind::SubroutineParam, ty, name.span)?;
                Ok((sym, sir::ParamPassing::QubitRef))
            }
            ast::ArgDef::ArrayRef(arr_ref, name) => {
                let ty = resolve_array_ref_type(
                    arr_ref,
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                let passing = match arr_ref.mutability {
                    ast::ArrayRefMut::Readonly => sir::ParamPassing::ReadonlyRef,
                    ast::ArrayRefMut::Mutable => sir::ParamPassing::MutableRef,
                };
                let sym =
                    self.resolver
                        .declare(name.name, SymbolKind::SubroutineParam, ty, name.span)?;
                Ok((sym, passing))
            }
        }
    }

    fn lower_extern_args(&mut self, params: &[ast::ExternArg<'_>]) -> Result<Vec<Type>> {
        params
            .iter()
            .map(|p| match p {
                ast::ExternArg::Scalar(ty) => {
                    resolve_scalar_type(ty, self.resolver.symbols(), self.resolver.options())
                }
                ast::ExternArg::ArrayRef(arr_ref) => resolve_array_ref_type(
                    arr_ref,
                    self.resolver.symbols(),
                    self.resolver.options(),
                ),
                ast::ExternArg::Creg(designator) => resolve_old_style_type(
                    &ast::OldStyleKind::Creg,
                    designator.as_ref(),
                    self.resolver.symbols(),
                    self.resolver.options(),
                ),
            })
            .collect()
    }

    fn system_width(&self) -> usize {
        self.resolver.options().system_angle_width
    }

    fn call_result_type(&self, callee: &sir::CallTarget, args: &[sir::Expr]) -> Result<Type> {
        match callee {
            sir::CallTarget::Intrinsic(i) => intrinsic_result_type(i, args, self.system_width()),
            sir::CallTarget::Symbol(sym) => Ok(self.resolver.symbols().get(*sym).ty.clone()),
        }
    }
}

// ── Free functions ──────────────────────────────────────────────────────

fn validate_gate_stmt(stmt: &sir::Stmt) -> Result<()> {
    match &stmt.kind {
        sir::StmtKind::GateCall { .. } | sir::StmtKind::Barrier { .. } => Ok(()),
        sir::StmtKind::If {
            then_body,
            else_body,
            ..
        } => {
            for s in then_body {
                validate_gate_stmt(s)?;
            }
            if let Some(eb) = else_body {
                for s in eb {
                    validate_gate_stmt(s)?;
                }
            }
            Ok(())
        }
        sir::StmtKind::For { body, .. } => {
            for s in body {
                validate_gate_stmt(s)?;
            }
            Ok(())
        }
        sir::StmtKind::While { body, .. } => {
            for s in body {
                validate_gate_stmt(s)?;
            }
            Ok(())
        }
        _ => Err(CompileError::new(ErrorKind::InvalidGateBody(
            "statement not allowed in gate body".into(),
        ))
        .with_span(stmt.span)),
    }
}

fn parse_hardware_qubit(s: &str) -> usize {
    s.strip_prefix('$')
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

fn map_bin_op(op: &ast::BinOp) -> sir::BinOp {
    match op {
        ast::BinOp::Add => sir::BinOp::Add,
        ast::BinOp::Sub => sir::BinOp::Sub,
        ast::BinOp::Mul => sir::BinOp::Mul,
        ast::BinOp::Div => sir::BinOp::Div,
        ast::BinOp::Mod => sir::BinOp::Mod,
        ast::BinOp::Pow => sir::BinOp::Pow,
        ast::BinOp::BitAnd => sir::BinOp::BitAnd,
        ast::BinOp::BitOr => sir::BinOp::BitOr,
        ast::BinOp::BitXor => sir::BinOp::BitXor,
        ast::BinOp::Shl => sir::BinOp::Shl,
        ast::BinOp::Shr => sir::BinOp::Shr,
        ast::BinOp::LogAnd => sir::BinOp::LogAnd,
        ast::BinOp::LogOr => sir::BinOp::LogOr,
        ast::BinOp::Eq => sir::BinOp::Eq,
        ast::BinOp::Neq => sir::BinOp::Neq,
        ast::BinOp::Lt => sir::BinOp::Lt,
        ast::BinOp::Gt => sir::BinOp::Gt,
        ast::BinOp::Lte => sir::BinOp::Lte,
        ast::BinOp::Gte => sir::BinOp::Gte,
    }
}

fn map_un_op(op: &ast::UnOp) -> sir::UnOp {
    match op {
        ast::UnOp::Neg => sir::UnOp::Neg,
        ast::UnOp::BitNot => sir::UnOp::BitNot,
        ast::UnOp::LogNot => sir::UnOp::LogNot,
    }
}

fn map_assign_op(op: &ast::AssignOp) -> sir::AssignOp {
    match op {
        ast::AssignOp::Assign => sir::AssignOp::Assign,
        ast::AssignOp::AddAssign => sir::AssignOp::AddAssign,
        ast::AssignOp::SubAssign => sir::AssignOp::SubAssign,
        ast::AssignOp::MulAssign => sir::AssignOp::MulAssign,
        ast::AssignOp::DivAssign => sir::AssignOp::DivAssign,
        ast::AssignOp::ModAssign => sir::AssignOp::ModAssign,
        ast::AssignOp::PowAssign => sir::AssignOp::PowAssign,
        ast::AssignOp::BitAndAssign => sir::AssignOp::BitAndAssign,
        ast::AssignOp::BitOrAssign => sir::AssignOp::BitOrAssign,
        ast::AssignOp::BitXorAssign => sir::AssignOp::BitXorAssign,
        ast::AssignOp::LeftShiftAssign => sir::AssignOp::ShlAssign,
        ast::AssignOp::RightShiftAssign => sir::AssignOp::ShrAssign,
    }
}

// ── Type inference ──────────────────────────────────────────────────────

fn classical_binary_return_ty(
    op: &sir::BinOp,
    lhs: ValueTy,
    rhs: ValueTy,
) -> oqi_classical::Result<ValueTy> {
    match op {
        sir::BinOp::Add => ClassicalAdd::return_ty(lhs, rhs),
        sir::BinOp::Sub => ClassicalSub::return_ty(lhs, rhs),
        sir::BinOp::Mul => ClassicalMul::return_ty(lhs, rhs),
        sir::BinOp::Div => ClassicalDiv::return_ty(lhs, rhs),
        sir::BinOp::Mod => ClassicalRem::return_ty(lhs, rhs),
        sir::BinOp::Pow => ClassicalPow::return_ty(lhs, rhs),
        sir::BinOp::BitAnd => ClassicalBitAnd::return_ty(lhs, rhs),
        sir::BinOp::BitOr => ClassicalBitOr::return_ty(lhs, rhs),
        sir::BinOp::BitXor => ClassicalBitXor::return_ty(lhs, rhs),
        sir::BinOp::Shl => ClassicalShl::return_ty(lhs, rhs),
        sir::BinOp::Shr => ClassicalShr::return_ty(lhs, rhs),
        sir::BinOp::LogAnd => ClassicalLogAnd::return_ty(lhs, rhs),
        sir::BinOp::LogOr => ClassicalLogOr::return_ty(lhs, rhs),
        sir::BinOp::Eq => ClassicalEq::return_ty(lhs, rhs),
        sir::BinOp::Neq => ClassicalNeq::return_ty(lhs, rhs),
        sir::BinOp::Lt => ClassicalLt::return_ty(lhs, rhs),
        sir::BinOp::Gt => ClassicalGt::return_ty(lhs, rhs),
        sir::BinOp::Lte => ClassicalLte::return_ty(lhs, rhs),
        sir::BinOp::Gte => ClassicalGte::return_ty(lhs, rhs),
    }
}

fn classical_unary_return_ty(op: &sir::UnOp, arg: ValueTy) -> oqi_classical::Result<ValueTy> {
    match op {
        sir::UnOp::Neg => ClassicalNeg::return_ty(arg),
        sir::UnOp::BitNot => ClassicalBitNot::return_ty(arg),
        sir::UnOp::LogNot => ClassicalLogNot::return_ty(arg),
    }
}

fn type_mismatch(expected: &Type, got: &Type, span: oqi_lex::Span) -> CompileError {
    CompileError::new(ErrorKind::TypeMismatch {
        expected: Box::new(expected.clone()),
        got: Box::new(got.clone()),
    })
    .with_span(span)
}

fn single_item_indices(index: &sir::IndexOp) -> Option<Vec<ClassicalIndex>> {
    match &index.kind {
        sir::IndexKind::Items(items) => items
            .iter()
            .map(|item| match item {
                sir::IndexItem::Single(_) => Some(ClassicalIndex::Item(0)),
                sir::IndexItem::Range(_) => None,
            })
            .collect(),
        sir::IndexKind::Set(_) => None,
    }
}

fn binary_result_type(
    op: &sir::BinOp,
    left: &Type,
    right: &Type,
    span: oqi_lex::Span,
) -> Result<Type> {
    let (Some(lhs), Some(rhs)) = (left.value_ty(), right.value_ty()) else {
        return Err(type_mismatch(left, right, span));
    };
    classical_binary_return_ty(op, lhs, rhs)
        .map(Type::from)
        .map_err(|_| type_mismatch(left, right, span))
}

fn unary_result_type(op: &sir::UnOp, operand: &Type) -> Type {
    operand
        .value_ty()
        .and_then(|value_ty| classical_unary_return_ty(op, value_ty).ok())
        .map(Type::from)
        .unwrap_or_else(|| match op {
            sir::UnOp::LogNot => Type::bool(),
            sir::UnOp::Neg | sir::UnOp::BitNot => operand.clone(),
        })
}

fn index_result_type(base_ty: &Type, index: &sir::IndexOp) -> Type {
    if let Some(indices) = single_item_indices(index) {
        if matches!(base_ty, Type::QubitReg(_)) && indices.len() == 1 {
            return Type::Qubit;
        }
        if let Some(value_ty) = base_ty.value_ty()
            && let Ok(result_ty) = value_ty.get(&indices)
        {
            return Type::from(result_ty);
        }
    }
    // Dynamic range/set index — conservatively return the base type
    base_ty.clone()
}

fn intrinsic_result_type(
    intrinsic: &sir::Intrinsic,
    args: &[sir::Expr],
    system_width: usize,
) -> Result<Type> {
    use sir::Intrinsic::*;
    match intrinsic {
        Sin | Cos | Tan | Arcsin | Arccos | Arctan | Exp | Log | Sqrt | Ceiling | Floor
        | Popcount | Real | Imag => {
            let [arg] = args else {
                return Err(intrinsic_arity_error(intrinsic, 1, args.len()));
            };
            let arg_ty = intrinsic_arg_type(intrinsic, &arg.ty)?;
            classical_intrinsic_unary_return_ty(intrinsic, arg_ty)
                .map(Type::from)
                .map_err(classical_intrinsic_error)
        }
        Mod | Rotl | Rotr => {
            let [lhs, rhs] = args else {
                return Err(intrinsic_arity_error(intrinsic, 2, args.len()));
            };
            let lhs_ty = intrinsic_arg_type(intrinsic, &lhs.ty)?;
            let rhs_ty = intrinsic_arg_type(intrinsic, &rhs.ty)?;
            classical_intrinsic_binary_return_ty(intrinsic, lhs_ty, rhs_ty)
                .map(Type::from)
                .map_err(classical_intrinsic_error)
        }
        Sizeof => {
            let (value, dim) = match args {
                [value] => (value, None),
                [value, dim] => (value, Some(dim)),
                _ => {
                    return Err(CompileError::new(ErrorKind::Unsupported(format!(
                        "intrinsic `{intrinsic}` expects 1 or 2 argument(s), got {}",
                        args.len()
                    ))));
                }
            };
            let value_ty = intrinsic_arg_type(intrinsic, &value.ty)?;
            if let Some(dim) = dim {
                if let sir::ExprKind::Literal(value) = &dim.kind {
                    let Some(dim) = value_as_usize(value) else {
                        return Err(CompileError::new(ErrorKind::Unsupported(format!(
                            "intrinsic `{intrinsic}` requires a non-negative integer dimension, got `{}`",
                            dim.ty
                        ))));
                    };
                    if value_ty.size(dim).is_none() {
                        return Err(CompileError::new(ErrorKind::Unsupported(format!(
                            "intrinsic `{intrinsic}` does not support dimension {dim} for argument type `{}`",
                            value.ty()
                        ))));
                    }
                }
            } else if value_ty.size(0).is_none() {
                return Err(CompileError::new(ErrorKind::Unsupported(format!(
                    "intrinsic `{intrinsic}` does not support argument type `{}`",
                    value.ty
                ))));
            }
            Ok(Type::int(system_width, false))
        }
    }
}

fn classical_intrinsic_unary_return_ty(
    intrinsic: &sir::Intrinsic,
    arg: ValueTy,
) -> oqi_classical::Result<ValueTy> {
    use sir::Intrinsic::*;
    match intrinsic {
        Sin => ClassicalSin::return_ty(arg),
        Cos => ClassicalCos::return_ty(arg),
        Tan => ClassicalTan::return_ty(arg),
        Arcsin => ClassicalArcsin::return_ty(arg),
        Arccos => ClassicalArccos::return_ty(arg),
        Arctan => ClassicalArctan::return_ty(arg),
        Exp => ClassicalExp::return_ty(arg),
        Log => ClassicalLog::return_ty(arg),
        Sqrt => ClassicalSqrt::return_ty(arg),
        Ceiling => ClassicalCeiling::return_ty(arg),
        Floor => ClassicalFloor::return_ty(arg),
        Popcount => ClassicalPopcount::return_ty(arg),
        Real => ClassicalReal::return_ty(arg),
        Imag => ClassicalImag::return_ty(arg),
        intrinsic => unreachable!("unsupported unary intrinsic: {intrinsic:?}"),
    }
}

fn classical_intrinsic_binary_return_ty(
    intrinsic: &sir::Intrinsic,
    lhs: ValueTy,
    rhs: ValueTy,
) -> oqi_classical::Result<ValueTy> {
    use sir::Intrinsic::*;
    match intrinsic {
        Mod => ClassicalRem::return_ty(lhs, rhs),
        Rotl => ClassicalRotl::return_ty(lhs, rhs),
        Rotr => ClassicalRotr::return_ty(lhs, rhs),
        intrinsic => unreachable!("unsupported binary intrinsic: {intrinsic:?}"),
    }
}

fn intrinsic_arg_type(intrinsic: &sir::Intrinsic, ty: &Type) -> Result<ValueTy> {
    ty.value_ty().ok_or_else(|| {
        CompileError::new(ErrorKind::Unsupported(format!(
            "intrinsic `{intrinsic}` does not support argument type `{ty}`",
        )))
    })
}

fn classical_intrinsic_error(err: oqi_classical::Error) -> CompileError {
    CompileError::new(ErrorKind::Unsupported(format!("{err:?}")))
}

fn intrinsic_arity_error(
    intrinsic: &sir::Intrinsic,
    expected: usize,
    actual: usize,
) -> CompileError {
    CompileError::new(ErrorKind::Unsupported(format!(
        "intrinsic `{intrinsic}` expects {expected} argument(s), got {actual}",
    )))
}

fn validate_cast(from: &Type, to: &Type, span: oqi_lex::Span) -> Result<()> {
    if from == to {
        return Ok(());
    }
    match (from.value_ty(), to.value_ty()) {
        (Some(from_ty), Some(to_ty)) => from_ty.cast(to_ty).map(|_| ()).map_err(|_| {
            CompileError::new(ErrorKind::TypeMismatch {
                expected: Box::new(to.clone()),
                got: Box::new(from.clone()),
            })
            .with_span(span)
        }),
        _ => Err(CompileError::new(ErrorKind::TypeMismatch {
            expected: Box::new(to.clone()),
            got: Box::new(from.clone()),
        })
        .with_span(span)),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::StdFileResolver;
    use crate::symbol::SymbolKind;
    use crate::types::FloatWidth;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[derive(Debug, Default)]
    struct TestFileResolver {
        files: HashMap<PathBuf, String>,
    }

    impl TestFileResolver {
        fn new(files: impl IntoIterator<Item = (PathBuf, String)>) -> Self {
            Self {
                files: files.into_iter().collect(),
            }
        }
    }

    impl FileResolver for TestFileResolver {
        fn resolve_source(&self, path: &Path, span: oqi_lex::Span) -> Result<String> {
            self.files.get(path).cloned().ok_or_else(|| {
                CompileError::new(ErrorKind::IncludeNotFound(path.display().to_string()))
                    .with_span(span)
            })
        }
    }

    fn compile_inline(source: &str) -> Result<sir::Program> {
        compile_source(source, StdFileResolver, None)
    }

    fn typed_expr(ty: Type) -> sir::Expr {
        sir::Expr {
            kind: sir::ExprKind::Literal(bool_value(false)),
            ty,
            span: oqi_lex::span(0, 0),
        }
    }

    #[test]
    fn compile_error_in_root_stmt_has_source_path() {
        let path = Path::new("/project/main.qasm");
        let err = match compile_source("missing_symbol;", StdFileResolver, Some(path)) {
            Ok(_) => panic!("expected lowering to fail"),
            Err(err) => err,
        };
        assert_eq!(err.path.as_deref(), Some(path));
    }

    #[test]
    fn compile_error_in_nested_include_uses_included_path() {
        let path = Path::new("/project/main.qasm");
        let file_resolver = TestFileResolver::new([
            (
                PathBuf::from("/project/file/1/path"),
                "include \"../2/path\";".to_string(),
            ),
            (
                PathBuf::from("/project/file/2/path"),
                "missing_symbol;".to_string(),
            ),
        ]);

        let err = match compile_source("include \"file/1/path\";", file_resolver, Some(path)) {
            Ok(_) => panic!("expected nested include lowering to fail"),
            Err(err) => err,
        };

        assert_eq!(err.path.as_deref(), Some(Path::new("/project/file/2/path")));
        assert!(matches!(err.kind, ErrorKind::UndefinedName(ref name) if name == "missing_symbol"));
    }

    #[test]
    fn compile_teleport() {
        let source = include_str!("../../fixtures/qasm/teleport.qasm");
        let program = compile_inline(source).expect("teleport should compile");

        assert_eq!(program.version.as_deref(), Some("3"));

        // 1 user gate (post), stdgates are hoisted
        let user_gates: Vec<_> = program
            .gates
            .iter()
            .filter(|g| program.symbols.get(g.symbol).name == "post")
            .collect();
        assert_eq!(user_gates.len(), 1);
        assert!(user_gates[0].body.body.is_empty());

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
        let source = include_str!("../../fixtures/qasm/adder.qasm");
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
            .any(|s| matches!(s.kind, sir::StmtKind::For { .. }));
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
        let source = include_str!("./stdgates.inc");
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
            g.body.body.iter().any(|s| {
                matches!(
                    &s.kind,
                    sir::StmtKind::GateCall {
                        gate: sir::GateCallTarget::GPhase,
                        ..
                    }
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
        assert_eq!(program.symbols.get(c_sym).ty, Type::bitreg(4));
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
        let has_gate_call = program.body.iter().any(|s| {
            matches!(
                &s.kind,
                sir::StmtKind::GateCall {
                    gate: sir::GateCallTarget::Symbol(_),
                    ..
                }
            )
        });
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
        let source = "1 + 2.0;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        let sw = usize::BITS as usize;
        assert_eq!(
            e.ty,
            Type::float(float_width_from_system_width(sw)),
            "1 + 2.0 should be Float"
        );
    }

    #[test]
    fn type_bool_and() {
        let source = "true && false;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::bool(), "true && false should be Bool");
    }

    #[test]
    fn type_comparison_is_bool() {
        let source = "int[32] x = 1; x == 2;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::bool(), "x == 2 should be Bool");
    }

    #[test]
    fn type_int_literal() {
        let source = "42;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        let sw = usize::BITS as usize;
        assert_eq!(
            e.ty,
            Type::int(sw, true),
            "42 should be Int(system_width, signed)"
        );
    }

    #[test]
    fn type_float_literal() {
        let source = "3.14;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        let sw = usize::BITS as usize;
        assert_eq!(
            e.ty,
            Type::float(float_width_from_system_width(sw)),
            "3.14 should be Float"
        );
    }

    #[test]
    fn type_bool_literal() {
        let source = "true;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::bool());
    }

    #[test]
    fn type_bitstring_literal() {
        let source = r#""0110";"#;
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::bitreg(4));
    }

    #[test]
    fn type_var_inherits_declared_type() {
        let source = "float[64] x = 1.0; x;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::float(FloatWidth::F64));
    }

    #[test]
    fn type_unary_neg_preserves_type() {
        let source = "float[64] x = 1.0; -x;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::float(FloatWidth::F64));
    }

    #[test]
    fn type_logical_not_is_bool() {
        let source = "bool x = true; !x;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::bool());
    }

    #[test]
    fn type_cast_result() {
        let source = "int[32] x = 5; float[64](x);";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::float(FloatWidth::F64));
    }

    #[test]
    fn type_cast_angle_to_float_valid() {
        let source = "angle[32] a; float[64](a);";
        let program = compile_inline(source).expect("angle -> float should be valid");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::float(FloatWidth::F64));
    }

    #[test]
    fn type_cast_float_to_angle_valid() {
        let source = "float[64] f = 1.0; angle[32](f);";
        let program = compile_inline(source).expect("float → angle should be valid");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::angle(32));
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
        assert_eq!(e.ty, Type::bit());
    }

    #[test]
    fn type_angle_arithmetic() {
        // angle + angle → angle
        let source = "angle[32] a; angle[32] b; a + b;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::angle(32));
    }

    #[test]
    fn type_angle_div_angle() {
        // angle / angle → uint
        let source = "angle[32] a; angle[32] b; a / b;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::int(32, false));
    }

    #[test]
    fn type_angle_mul_uint() {
        // angle * uint → angle
        let source = "angle[32] a; uint[32] n = 2; a * n;";
        let program = compile_inline(source).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::angle(32));
    }

    #[test]
    fn type_add_signed_unsigned_same_width_uses_classical_promotion() {
        let result = binary_result_type(
            &sir::BinOp::Add,
            &Type::int(32, true),
            &Type::int(32, false),
            oqi_lex::span(0, 0),
        )
        .unwrap();
        assert_eq!(result, Type::int(32, false));
    }

    #[test]
    fn type_add_float_complex_uses_classical_promotion() {
        let result = binary_result_type(
            &sir::BinOp::Add,
            &Type::float(FloatWidth::F64),
            &Type::complex(FloatWidth::F32),
            oqi_lex::span(0, 0),
        )
        .unwrap();
        assert_eq!(result, Type::complex(FloatWidth::F64));
    }

    #[test]
    fn type_add_bool_int_is_rejected() {
        let result = binary_result_type(
            &sir::BinOp::Add,
            &Type::bool(),
            &Type::int(32, true),
            oqi_lex::span(0, 0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn type_intrinsic_ceiling_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Ceiling,
            &[typed_expr(Type::float(FloatWidth::F32))],
            usize::BITS as usize,
        )
        .unwrap();
        assert_eq!(result, Type::float(FloatWidth::F32));
    }

    #[test]
    fn type_intrinsic_sin_angle_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Sin,
            &[typed_expr(Type::angle(32))],
            usize::BITS as usize,
        )
        .unwrap();
        assert_eq!(result, Type::float(FloatWidth::F64));
    }

    #[test]
    fn type_intrinsic_mod_uses_classical_promotion() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Mod,
            &[
                typed_expr(Type::int(8, false)),
                typed_expr(Type::int(16, false)),
            ],
            usize::BITS as usize,
        )
        .unwrap();
        assert_eq!(result, Type::int(16, false));
    }

    #[test]
    fn type_intrinsic_popcount_uses_input_width() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Popcount,
            &[typed_expr(Type::bitreg(8))],
            usize::BITS as usize,
        )
        .unwrap();
        assert_eq!(result, Type::int(8, false));
    }

    #[test]
    fn type_intrinsic_real_uses_classical_return_ty() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Real,
            &[typed_expr(Type::complex(FloatWidth::F32))],
            usize::BITS as usize,
        )
        .unwrap();
        assert_eq!(result, Type::float(FloatWidth::F32));
    }

    #[test]
    fn type_intrinsic_imag_uses_classical_promotion() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Imag,
            &[typed_expr(Type::int(8, true))],
            usize::BITS as usize,
        )
        .unwrap();
        assert_eq!(result, Type::float(FloatWidth::F64));
    }

    #[test]
    fn type_intrinsic_sizeof_rejects_invalid_literal_dimension() {
        let result = intrinsic_result_type(
            &sir::Intrinsic::Sizeof,
            &[
                typed_expr(Type::array(
                    crate::classical::PrimitiveTy::Uint(crate::classical::bit_width(8)),
                    vec![2, 3],
                )),
                sir::Expr {
                    kind: sir::ExprKind::Literal(int_value(3)),
                    ty: Type::int(64, true),
                    span: oqi_lex::span(0, 0),
                },
            ],
            usize::BITS as usize,
        );
        assert!(result.is_err());
    }

    #[test]
    fn type_intrinsic_invalid_argument_is_rejected() {
        let err = match compile_inline("sin(true);") {
            Ok(_) => panic!("expected intrinsic type error"),
            Err(err) => err,
        };
        assert!(matches!(err.kind, ErrorKind::Unsupported(_)));
    }

    #[test]
    fn type_index_into_array_returns_array_ref() {
        let index = sir::IndexOp {
            kind: sir::IndexKind::Items(vec![sir::IndexItem::Single(Box::new(sir::Expr {
                kind: sir::ExprKind::Literal(int_value(0)),
                ty: Type::int(64, true),
                span: oqi_lex::span(0, 0),
            }))]),
            span: oqi_lex::span(0, 0),
        };
        let result = index_result_type(
            &Type::array(
                crate::classical::PrimitiveTy::Uint(crate::classical::bit_width(8)),
                vec![2, 3],
            ),
            &index,
        );
        assert_eq!(
            result,
            Type::array_ref_fixed(
                crate::classical::PrimitiveTy::Uint(crate::classical::bit_width(8)),
                vec![3],
                crate::classical::RefAccess::Mutable,
            )
        );
    }

    #[test]
    fn type_teleport_condition_is_bool() {
        let source = include_str!("../../fixtures/qasm/teleport.qasm");
        let program = compile_inline(source).expect("teleport should compile");
        // Find an if-statement and check its condition type
        let if_stmt = program
            .body
            .iter()
            .find(|s| matches!(s.kind, sir::StmtKind::If { .. }));
        assert!(if_stmt.is_some());
        if let sir::StmtKind::If { condition, .. } = &if_stmt.unwrap().kind {
            assert_eq!(condition.ty, Type::bool(), "if condition should be Bool");
        }
    }

    #[test]
    fn type_adder_cast_is_bool() {
        // adder.qasm uses bool(a_in[i]) - verify cast produces Bool
        let source = include_str!("../../fixtures/qasm/adder.qasm");
        let program = compile_inline(source).expect("adder should compile");
        // Find a for loop with if(bool(...))
        let for_stmt = program
            .body
            .iter()
            .find(|s| matches!(s.kind, sir::StmtKind::For { .. }));
        assert!(for_stmt.is_some());
        if let sir::StmtKind::For { body, .. } = &for_stmt.unwrap().kind {
            let if_stmt = body
                .iter()
                .find(|s| matches!(s.kind, sir::StmtKind::If { .. }));
            assert!(if_stmt.is_some());
            if let sir::StmtKind::If { condition, .. } = &if_stmt.unwrap().kind {
                assert_eq!(condition.ty, Type::bool(), "bool() cast should yield Bool");
            }
        }
    }
}
