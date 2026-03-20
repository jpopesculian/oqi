use std::path::Path;

use bitvec::vec::BitVec;
use oqi_parse::ast;

use crate::error::{CompileError, ErrorKind, Result};
use crate::resolve::{IncludeSource, Resolver};
use crate::sir;
use crate::symbol::SymbolKind;
use crate::types::{
    eval_const_expr, eval_designator, float_width_from_system_width, parse_int_literal,
    resolve_array_ref_type, resolve_old_style_type, resolve_qubit_type, resolve_scalar_type,
    resolve_type, CompileOptions, FloatWidth, Type,
};
use crate::value::{FloatValue, TimeUnit, TimingNumber, TimingValue};

// ── Public API ──────────────────────────────────────────────────────────

pub fn compile_ast(program: &ast::Program<'_>, options: CompileOptions) -> Result<sir::Program> {
    let mut lowerer = Lowerer::new(options);
    lowerer.lower_program(program)?;
    Ok(lowerer.finish(program))
}

pub fn compile_source(source: &str, source_name: Option<&Path>) -> Result<sir::Program> {
    let ast = oqi_parse::parse(source).map_err(|e| CompileError {
        kind: ErrorKind::Unsupported(format!("parse error: {e:?}")),
        span: 0..0,
    })?;
    let options = CompileOptions {
        source_name: source_name.map(|p| p.to_path_buf()),
        ..Default::default()
    };
    compile_ast(&ast, options)
}

pub fn compile_file(path: &Path) -> Result<sir::Program> {
    let source = std::fs::read_to_string(path).map_err(|_| CompileError {
        kind: ErrorKind::IncludeNotFound(path.display().to_string()),
        span: 0..0,
    })?;
    compile_source(&source, Some(path))
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
    fn new(options: CompileOptions) -> Self {
        Self {
            resolver: Resolver::new(options),
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

    // ── Statement lowering ──────────────────────────────────────────────

    fn lower_stmt_or_scope(&mut self, item: &ast::StmtOrScope<'_>) -> Result<Vec<sir::Stmt>> {
        match item {
            ast::StmtOrScope::Stmt(stmt) => {
                let s = self.lower_stmt(stmt)?;
                Ok(s)
            }
            ast::StmtOrScope::Scope(scope) => {
                self.resolver.push_scope();
                let mut stmts = Vec::new();
                for item in &scope.body {
                    stmts.extend(self.lower_stmt_or_scope(item)?);
                }
                self.resolver.pop_scope();
                Ok(stmts)
            }
        }
    }

    fn lower_body(&mut self, item: &ast::StmtOrScope<'_>) -> Result<Vec<sir::Stmt>> {
        self.resolver.push_scope();
        let stmts = match item {
            ast::StmtOrScope::Stmt(stmt) => self.lower_stmt(stmt)?,
            ast::StmtOrScope::Scope(scope) => {
                let mut stmts = Vec::new();
                for item in &scope.body {
                    stmts.extend(self.lower_stmt_or_scope(item)?);
                }
                stmts
            }
        };
        self.resolver.pop_scope();
        Ok(stmts)
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt<'_>) -> Result<Vec<sir::Stmt>> {
        let annotations = self.lower_annotations(&stmt.annotations);
        let span = stmt.span.clone();

        let stmts = match &stmt.kind {
            ast::StmtKind::Include(path) => {
                return self.lower_include(path, &span);
            }

            ast::StmtKind::ClassicalDecl { ty, name, init } => {
                let resolved_ty =
                    resolve_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let symbol = self.resolver.declare(
                    name.name,
                    SymbolKind::Variable,
                    resolved_ty,
                    name.span.clone(),
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
                let resolved_ty = resolve_scalar_type(
                    ty,
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                let init_expr = match init {
                    ast::DeclExpr::Expr(e) => self.lower_expr(e)?,
                    _ => {
                        return Err(CompileError {
                            kind: ErrorKind::InvalidContext(
                                "const initializer must be an expression".into(),
                            ),
                            span,
                        });
                    }
                };
                let const_val = eval_const_expr(
                    match init {
                        ast::DeclExpr::Expr(e) => e,
                        _ => unreachable!(),
                    },
                    self.resolver.symbols(),
                )?;
                let symbol = self.resolver.declare(
                    name.name,
                    SymbolKind::Const,
                    resolved_ty,
                    name.span.clone(),
                )?;
                self.resolver
                    .symbols_mut()
                    .set_const_value(symbol, const_val);
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
                let resolved_ty = resolve_qubit_type(ty, self.resolver.symbols())?;
                let symbol = self.resolver.declare(
                    name.name,
                    SymbolKind::Qubit,
                    resolved_ty,
                    name.span.clone(),
                )?;
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
                )?;
                let (kind, stmt_kind) = match keyword {
                    ast::OldStyleKind::Creg => {
                        let sym_kind = SymbolKind::Variable;
                        let symbol = self.resolver.declare(
                            name.name,
                            sym_kind,
                            resolved_ty,
                            name.span.clone(),
                        )?;
                        (sym_kind, sir::StmtKind::ClassicalDecl { symbol, init: None })
                    }
                    ast::OldStyleKind::Qreg => {
                        let sym_kind = SymbolKind::Qubit;
                        let symbol = self.resolver.declare(
                            name.name,
                            sym_kind,
                            resolved_ty,
                            name.span.clone(),
                        )?;
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
                let symbol = self.resolver.declare(
                    name.name,
                    sym_kind,
                    resolved_ty,
                    name.span.clone(),
                )?;
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
                let gate_sym = self.resolver.declare(
                    name.name,
                    SymbolKind::Gate,
                    Type::Void,
                    name.span.clone(),
                )?;
                self.resolver.push_scope();
                let angle_width = self.resolver.options().system_angle_width;
                let param_ids: Vec<_> = params
                    .iter()
                    .map(|p| {
                        self.resolver.declare(
                            p.name,
                            SymbolKind::GateParam,
                            Type::Angle(angle_width),
                            p.span.clone(),
                        )
                    })
                    .collect::<Result<_>>()?;
                let qubit_ids: Vec<_> = qubits
                    .iter()
                    .map(|q| {
                        self.resolver.declare(
                            q.name,
                            SymbolKind::GateQubit,
                            Type::Qubit,
                            q.span.clone(),
                        )
                    })
                    .collect::<Result<_>>()?;
                let gate_body = self.lower_gate_body(body)?;
                self.resolver.pop_scope();
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
                    name.span.clone(),
                )?;
                self.resolver.push_scope();
                let sir_params = self.lower_arg_defs(params)?;
                let ret_ty = match return_ty {
                    Some(s) => Some(resolve_scalar_type(
                        s,
                        self.resolver.symbols(),
                        self.resolver.options(),
                    )?),
                    None => None,
                };
                if let Some(ref ty) = ret_ty {
                    self.resolver.symbols_mut().get_mut(sub_sym).ty = ty.clone();
                }
                let mut body_stmts = Vec::new();
                for item in &body.body {
                    body_stmts.extend(self.lower_stmt_or_scope(item)?);
                }
                self.resolver.pop_scope();
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
                let ext_sym = self.resolver.declare(
                    name.name,
                    SymbolKind::Extern,
                    Type::Void,
                    name.span.clone(),
                )?;
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
                        ast::DefcalArgDef::Expr(e) => Ok(sir::CalibrationArg::Expr(self.lower_expr(e)?)),
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
                        let sym = self.resolver.resolve(id.name, id.span.clone())?;
                        sir::GateCallTarget::Symbol(sym)
                    }
                    ast::GateCallName::Gphase(_) => sir::GateCallTarget::GPhase,
                };
                let sir_mods = modifiers
                    .iter()
                    .map(|m| self.lower_gate_modifier(m))
                    .collect::<Result<_>>()?;
                let sir_args = match args {
                    Some(a) => a.iter().map(|e| self.lower_expr(e)).collect::<Result<_>>()?,
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
                    ast::ExprOrMeasure::Expr(e) => sir::AssignValue::Expr(self.lower_expr(e)?),
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
                let var_ty = resolve_scalar_type(
                    ty,
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                self.resolver.push_scope();
                let var_sym = self.resolver.declare(
                    var.name,
                    SymbolKind::LoopVar,
                    var_ty,
                    var.span.clone(),
                )?;
                let sir_iterable = self.lower_for_iterable(iterable)?;
                let body_stmts = match body.as_ref() {
                    ast::StmtOrScope::Stmt(s) => self.lower_stmt(s)?,
                    ast::StmtOrScope::Scope(sc) => {
                        let mut stmts = Vec::new();
                        for item in &sc.body {
                            stmts.extend(self.lower_stmt_or_scope(item)?);
                        }
                        stmts
                    }
                };
                self.resolver.pop_scope();
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
                        Some(sir::ReturnValue::Expr(self.lower_expr(e)?))
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
                let symbol = self.resolver.declare(
                    name.name,
                    SymbolKind::Alias,
                    Type::Void,
                    name.span.clone(),
                )?;
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
                self.resolver.push_scope();
                let mut body_stmts = Vec::new();
                for item in &body.body {
                    body_stmts.extend(self.lower_stmt_or_scope(item)?);
                }
                self.resolver.pop_scope();
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

    fn lower_include(&mut self, path: &str, span: &oqi_lex::Span) -> Result<Vec<sir::Stmt>> {
        let path = path.trim_matches('"');
        let source = self.resolver.resolve_include_path(path)?;
        match source {
            IncludeSource::Embedded(content) => self.lower_include_source(content, path, span),
            IncludeSource::File(file_path) => {
                self.resolver.push_include(file_path.clone())?;
                let content = std::fs::read_to_string(&file_path).map_err(|_| CompileError {
                    kind: ErrorKind::IncludeNotFound(path.to_string()),
                    span: span.clone(),
                })?;
                let result = self.lower_include_source(&content, path, span);
                self.resolver.pop_include();
                result
            }
        }
    }

    fn lower_include_source(
        &mut self,
        content: &str,
        path: &str,
        span: &oqi_lex::Span,
    ) -> Result<Vec<sir::Stmt>> {
        let ast = oqi_parse::parse(content).map_err(|e| CompileError {
            kind: ErrorKind::Unsupported(format!("parse error in include '{path}': {e:?}")),
            span: span.clone(),
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
                let sym = self.resolver.resolve(id.name, id.span.clone())?;
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
                let awi = parse_int_literal(s, *enc).ok_or_else(|| CompileError {
                    kind: ErrorKind::Unsupported(format!("invalid integer literal: {s}")),
                    span: span.clone(),
                })?;
                let ty = Type::Int {
                    width: self.system_width(),
                    signed: true,
                };
                Ok(sir::Expr {
                    kind: sir::ExprKind::IntLit(awi),
                    ty,
                    span,
                })
            }

            ast::Expr::FloatLiteral(s, _) => {
                let v: f64 = s.replace('_', "").parse().map_err(|_| CompileError {
                    kind: ErrorKind::Unsupported(format!("invalid float literal: {s}")),
                    span: span.clone(),
                })?;
                let fw = float_width_from_system_width(self.system_width());
                Ok(sir::Expr {
                    kind: sir::ExprKind::FloatLit(FloatValue::F64(v)),
                    ty: Type::Float(fw),
                    span,
                })
            }

            ast::Expr::ImagLiteral(s, _) => {
                let num_part = s.strip_suffix("im").unwrap_or(s);
                let v: f64 = num_part.replace('_', "").parse().map_err(|_| CompileError {
                    kind: ErrorKind::Unsupported(format!("invalid imaginary literal: {s}")),
                    span: span.clone(),
                })?;
                let fw = float_width_from_system_width(self.system_width());
                Ok(sir::Expr {
                    kind: sir::ExprKind::ImagLit(FloatValue::F64(v)),
                    ty: Type::Complex(fw),
                    span,
                })
            }

            ast::Expr::BoolLiteral(b, _) => Ok(sir::Expr {
                kind: sir::ExprKind::BoolLit(*b),
                ty: Type::Bool,
                span,
            }),

            ast::Expr::BitstringLiteral(s, _) => {
                let bv = parse_bitstring(s, &span)?;
                let len = bv.len() as u32;
                Ok(sir::Expr {
                    kind: sir::ExprKind::BitstringLit(bv),
                    ty: Type::BitReg(len),
                    span,
                })
            }

            ast::Expr::TimingLiteral(s, _) => {
                let tv = parse_timing_literal(s, &span)?;
                Ok(sir::Expr {
                    kind: sir::ExprKind::TimingLit(tv),
                    ty: Type::Duration,
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
                let ty = binary_result_type(&sir_op, &l.ty, &r.ty, &span)?;
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
                let callee = self.resolver.resolve_call(name.name, name.span.clone())?;
                let sir_args: Vec<sir::Expr> = args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<_>>()?;
                let ty = self.call_result_type(&callee, &sir_args);
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
                let target_ty =
                    resolve_type(ty, self.resolver.symbols(), self.resolver.options())?;
                let inner = self.lower_expr(operand)?;
                validate_cast(&inner.ty, &target_ty, &span)?;
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
                self.resolver.push_scope();
                let mut stmts = Vec::new();
                for item in &scope.body {
                    stmts.extend(self.lower_stmt_or_scope(item)?);
                }
                self.resolver.pop_scope();
                Ok(sir::Expr {
                    kind: sir::ExprKind::DurationOf(stmts),
                    ty: Type::Duration,
                    span,
                })
            }
        }
    }

    // ── Helper lowering methods ─────────────────────────────────────────

    fn lower_gate_operand(&mut self, op: &ast::GateOperand<'_>) -> Result<sir::QubitOperand> {
        match op {
            ast::GateOperand::Indexed(id) => {
                let sym = self.resolver.resolve(id.name.name, id.name.span.clone())?;
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
        let sym = self.resolver.resolve(id.name.name, id.name.span.clone())?;
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
                            Ok(sir::IndexItem::Single(self.lower_expr(e)?))
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
            span: op.span.clone(),
        })
    }

    fn lower_gate_modifier(&mut self, m: &ast::GateModifier<'_>) -> Result<sir::GateModifier> {
        match m {
            ast::GateModifier::Inv(_) => Ok(sir::GateModifier::Inv),
            ast::GateModifier::Pow(expr, _) => {
                Ok(sir::GateModifier::Pow(self.lower_expr(expr)?))
            }
            ast::GateModifier::Ctrl(designator, _) => {
                let n = match designator {
                    Some(e) => eval_designator(e, self.resolver.symbols())?,
                    None => 1,
                };
                Ok(sir::GateModifier::Ctrl(n))
            }
            ast::GateModifier::NegCtrl(designator, _) => {
                let n = match designator {
                    Some(e) => eval_designator(e, self.resolver.symbols())?,
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
                    span: span.clone(),
                })
            }
            ast::MeasureExpr::QuantumCall {
                name,
                args,
                operands,
                span,
            } => {
                let sym = self.resolver.resolve(name.name, name.span.clone())?;
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
                    span: span.clone(),
                })
            }
        }
    }

    fn lower_decl_init(&mut self, d: &ast::DeclExpr<'_>) -> Result<sir::DeclInit> {
        match d {
            ast::DeclExpr::Expr(e) => Ok(sir::DeclInit::Expr(self.lower_expr(e)?)),
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
                    Ok(sir::ArrayLiteralItem::Expr(self.lower_expr(e)?))
                }
                ast::ArrayLiteralItem::Nested(inner) => Ok(sir::ArrayLiteralItem::Nested(
                    self.lower_array_literal(inner)?,
                )),
            })
            .collect::<Result<_>>()?;
        Ok(sir::ArrayLiteral {
            items,
            span: al.span.clone(),
        })
    }

    fn lower_annotations(&mut self, anns: &[ast::Annotation<'_>]) -> Vec<sir::Annotation> {
        anns.iter()
            .map(|a| sir::Annotation {
                keyword: a.keyword.to_string(),
                content: a.content.map(|s| s.to_string()),
                span: a.span.clone(),
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
            ast::ForIterable::Expr(e) => Ok(sir::ForIterable::Expr(self.lower_expr(e)?)),
        }
    }

    fn lower_switch_case(&mut self, case: &ast::SwitchCase<'_>) -> Result<sir::SwitchCase> {
        match case {
            ast::SwitchCase::Case(labels, scope) => {
                let label_exprs = labels
                    .iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<_>>()?;
                self.resolver.push_scope();
                let mut body = Vec::new();
                for item in &scope.body {
                    body.extend(self.lower_stmt_or_scope(item)?);
                }
                self.resolver.pop_scope();
                Ok(sir::SwitchCase {
                    labels: sir::SwitchLabels::Values(label_exprs),
                    body,
                })
            }
            ast::SwitchCase::Default(scope) => {
                self.resolver.push_scope();
                let mut body = Vec::new();
                for item in &scope.body {
                    body.extend(self.lower_stmt_or_scope(item)?);
                }
                self.resolver.pop_scope();
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
        params.iter().map(|p| {
            let (sym, passing) = self.lower_single_arg_def(p)?;
            Ok(sir::SubroutineParam { symbol: sym, passing })
        }).collect()
    }

    fn lower_single_arg_def(
        &mut self,
        arg: &ast::ArgDef<'_>,
    ) -> Result<(crate::symbol::SymbolId, sir::ParamPassing)> {
        match arg {
            ast::ArgDef::Scalar(ty, name) => {
                let resolved = resolve_scalar_type(
                    ty,
                    self.resolver.symbols(),
                    self.resolver.options(),
                )?;
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    resolved,
                    name.span.clone(),
                )?;
                Ok((sym, sir::ParamPassing::ByValue))
            }
            ast::ArgDef::Qubit(ty, name) => {
                let resolved = resolve_qubit_type(ty, self.resolver.symbols())?;
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    resolved,
                    name.span.clone(),
                )?;
                Ok((sym, sir::ParamPassing::QubitRef))
            }
            ast::ArgDef::Creg(name, designator) => {
                let ty = resolve_old_style_type(
                    &ast::OldStyleKind::Creg,
                    designator.as_ref(),
                    self.resolver.symbols(),
                )?;
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    ty,
                    name.span.clone(),
                )?;
                Ok((sym, sir::ParamPassing::ByValue))
            }
            ast::ArgDef::Qreg(name, designator) => {
                let ty = resolve_old_style_type(
                    &ast::OldStyleKind::Qreg,
                    designator.as_ref(),
                    self.resolver.symbols(),
                )?;
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    ty,
                    name.span.clone(),
                )?;
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
                let sym = self.resolver.declare(
                    name.name,
                    SymbolKind::SubroutineParam,
                    ty,
                    name.span.clone(),
                )?;
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
                ast::ExternArg::ArrayRef(arr_ref) => {
                    resolve_array_ref_type(arr_ref, self.resolver.symbols(), self.resolver.options())
                }
                ast::ExternArg::Creg(designator) => resolve_old_style_type(
                    &ast::OldStyleKind::Creg,
                    designator.as_ref(),
                    self.resolver.symbols(),
                ),
            })
            .collect()
    }

    fn system_width(&self) -> u32 {
        self.resolver.options().system_angle_width
    }

    fn call_result_type(&self, callee: &sir::CallTarget, args: &[sir::Expr]) -> Type {
        match callee {
            sir::CallTarget::Intrinsic(i) => {
                intrinsic_result_type(i, args, self.system_width())
            }
            sir::CallTarget::Symbol(sym) => {
                self.resolver.symbols().get(*sym).ty.clone()
            }
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
        _ => Err(CompileError {
            kind: ErrorKind::InvalidGateBody(format!(
                "statement not allowed in gate body"
            )),
            span: stmt.span.clone(),
        }),
    }
}

fn parse_hardware_qubit(s: &str) -> u32 {
    s.strip_prefix('$')
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

fn parse_bitstring(s: &str, span: &oqi_lex::Span) -> Result<BitVec> {
    // Strip surrounding quotes if present
    let inner = s.trim_matches('"');
    let mut bv = BitVec::new();
    for ch in inner.chars() {
        match ch {
            '0' => bv.push(false),
            '1' => bv.push(true),
            _ => {
                return Err(CompileError {
                    kind: ErrorKind::Unsupported(format!("invalid bitstring character: {ch}")),
                    span: span.clone(),
                });
            }
        }
    }
    Ok(bv)
}

fn parse_timing_literal(s: &str, span: &oqi_lex::Span) -> Result<TimingValue> {
    // Find the boundary between numeric part and unit suffix
    let unit_start = s
        .find(|c: char| c.is_alphabetic())
        .ok_or_else(|| CompileError {
            kind: ErrorKind::Unsupported(format!("invalid timing literal: {s}")),
            span: span.clone(),
        })?;
    let (num_str, unit_str) = s.split_at(unit_start);
    let num_str = num_str.replace('_', "");

    let unit = match unit_str {
        "dt" => TimeUnit::Dt,
        "ns" => TimeUnit::Ns,
        "us" | "µs" => TimeUnit::Us,
        "ms" => TimeUnit::Ms,
        "s" => TimeUnit::S,
        _ => {
            return Err(CompileError {
                kind: ErrorKind::Unsupported(format!("unknown time unit: {unit_str}")),
                span: span.clone(),
            });
        }
    };

    let value = if num_str.contains('.') {
        let v: f64 = num_str.parse().map_err(|_| CompileError {
            kind: ErrorKind::Unsupported(format!("invalid timing number: {num_str}")),
            span: span.clone(),
        })?;
        TimingNumber::Float(FloatValue::F64(v))
    } else {
        let v: i64 = num_str.parse().map_err(|_| CompileError {
            kind: ErrorKind::Unsupported(format!("invalid timing number: {num_str}")),
            span: span.clone(),
        })?;
        TimingNumber::Integer(v)
    };

    Ok(TimingValue { value, unit })
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

fn binary_result_type(
    op: &sir::BinOp,
    left: &Type,
    right: &Type,
    span: &oqi_lex::Span,
) -> Result<Type> {
    use sir::BinOp::*;
    match op {
        Eq | Neq | Lt | Gt | Lte | Gte => Ok(Type::Bool),
        LogAnd | LogOr => Ok(Type::Bool),
        Shl | Shr => Ok(left.clone()),
        BitAnd | BitOr | BitXor => Ok(left.clone()),
        Add | Sub | Mul | Div | Mod | Pow => {
            // Angle special rules
            if is_angle(left) && is_angle(right) {
                return match op {
                    Add | Sub => Ok(left.clone()),
                    Div => {
                        let w = angle_width(left);
                        Ok(Type::Int {
                            width: w,
                            signed: false,
                        })
                    }
                    _ => Err(CompileError {
                        kind: ErrorKind::TypeMismatch {
                            expected: left.clone(),
                            got: right.clone(),
                        },
                        span: span.clone(),
                    }),
                };
            }
            if is_angle(left) && is_integer(right) {
                return match op {
                    Mul | Div => Ok(left.clone()),
                    _ => Err(CompileError {
                        kind: ErrorKind::TypeMismatch {
                            expected: left.clone(),
                            got: right.clone(),
                        },
                        span: span.clone(),
                    }),
                };
            }
            if is_integer(left) && is_angle(right) {
                return match op {
                    Mul => Ok(right.clone()),
                    _ => Err(CompileError {
                        kind: ErrorKind::TypeMismatch {
                            expected: left.clone(),
                            got: right.clone(),
                        },
                        span: span.clone(),
                    }),
                };
            }
            // Duration arithmetic
            if matches!(left, Type::Duration) && matches!(right, Type::Duration) {
                return match op {
                    Add | Sub => Ok(Type::Duration),
                    _ => Err(CompileError {
                        kind: ErrorKind::TypeMismatch {
                            expected: left.clone(),
                            got: right.clone(),
                        },
                        span: span.clone(),
                    }),
                };
            }
            // Standard numeric promotion
            promote_numeric(left, right).ok_or_else(|| CompileError {
                kind: ErrorKind::TypeMismatch {
                    expected: left.clone(),
                    got: right.clone(),
                },
                span: span.clone(),
            })
        }
    }
}

fn unary_result_type(op: &sir::UnOp, operand: &Type) -> Type {
    match op {
        sir::UnOp::LogNot => Type::Bool,
        sir::UnOp::Neg | sir::UnOp::BitNot => operand.clone(),
    }
}

fn index_result_type(base_ty: &Type, index: &sir::IndexOp) -> Type {
    let is_single = match &index.kind {
        sir::IndexKind::Items(items) => {
            items.len() == 1 && matches!(&items[0], sir::IndexItem::Single(_))
        }
        _ => false,
    };
    if is_single {
        match base_ty {
            Type::QubitReg(_) => Type::Qubit,
            Type::BitReg(_) | Type::Int { .. } => Type::Bit,
            Type::Array { element, dims } if dims.len() <= 1 => *element.clone(),
            Type::Array { element, dims } => Type::Array {
                element: element.clone(),
                dims: dims[1..].to_vec(),
            },
            _ => base_ty.clone(),
        }
    } else {
        // Range/set index — conservatively return the base type
        base_ty.clone()
    }
}

fn intrinsic_result_type(
    intrinsic: &sir::Intrinsic,
    args: &[sir::Expr],
    system_width: u32,
) -> Type {
    use sir::Intrinsic::*;
    match intrinsic {
        Sin | Cos | Tan | Arcsin | Arccos | Arctan | Exp | Log | Sqrt => {
            args.first().map(|a| a.ty.clone()).unwrap_or(Type::Void)
        }
        Ceiling | Floor => Type::Int {
            width: system_width,
            signed: true,
        },
        Mod => args.first().map(|a| a.ty.clone()).unwrap_or(Type::Void),
        Popcount => Type::Int {
            width: system_width,
            signed: false,
        },
        Rotl | Rotr => args.first().map(|a| a.ty.clone()).unwrap_or(Type::Void),
        Real | Imag => match args.first().map(|a| &a.ty) {
            Some(Type::Complex(w)) => Type::Float(*w),
            _ => Type::Float(float_width_from_system_width(system_width)),
        },
        Sizeof => Type::Int {
            width: system_width,
            signed: false,
        },
    }
}

fn validate_cast(from: &Type, to: &Type, span: &oqi_lex::Span) -> Result<()> {
    if from == to {
        return Ok(());
    }
    let valid = match (cast_category(from), cast_category(to)) {
        (CastCat::Bool, CastCat::Bool | CastCat::Int | CastCat::Float | CastCat::Bit) => true,
        (CastCat::Int, CastCat::Bool | CastCat::Int | CastCat::Float | CastCat::Bit) => true,
        (CastCat::Float, CastCat::Bool | CastCat::Int | CastCat::Float | CastCat::Angle) => true,
        (CastCat::Angle, CastCat::Bool | CastCat::Angle | CastCat::Bit) => true,
        (CastCat::Bit, CastCat::Bool | CastCat::Int | CastCat::Bit | CastCat::Angle) => true,
        (cat_from, cat_to) if cat_from == cat_to => true,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(CompileError {
            kind: ErrorKind::TypeMismatch {
                expected: to.clone(),
                got: from.clone(),
            },
            span: span.clone(),
        })
    }
}

#[derive(PartialEq, Eq)]
enum CastCat {
    Bool,
    Int,
    Float,
    Complex,
    Angle,
    Bit,
    Duration,
    Other,
}

fn cast_category(ty: &Type) -> CastCat {
    match ty {
        Type::Bool => CastCat::Bool,
        Type::Int { .. } => CastCat::Int,
        Type::Float(_) => CastCat::Float,
        Type::Complex(_) => CastCat::Complex,
        Type::Angle(_) => CastCat::Angle,
        Type::Bit | Type::BitReg(_) => CastCat::Bit,
        Type::Duration | Type::Stretch => CastCat::Duration,
        _ => CastCat::Other,
    }
}

fn promote_numeric(a: &Type, b: &Type) -> Option<Type> {
    if a == b {
        return Some(a.clone());
    }
    let rank = |t: &Type| -> Option<u8> {
        match t {
            Type::Bool | Type::Bit => Some(0),
            Type::Int { .. } => Some(1),
            Type::Float(_) => Some(2),
            Type::Complex(_) => Some(3),
            _ => None,
        }
    };
    let ra = rank(a)?;
    let rb = rank(b)?;
    if ra == rb {
        match (a, b) {
            (Type::Bool | Type::Bit, Type::Bool | Type::Bit) => Some(Type::Bool),
            (
                Type::Int {
                    width: w1,
                    signed: s1,
                },
                Type::Int {
                    width: w2,
                    signed: s2,
                },
            ) => {
                if w1 == w2 {
                    // C99: same width, unsigned wins
                    Some(Type::Int {
                        width: *w1,
                        signed: *s1 && *s2,
                    })
                } else if w1 > w2 {
                    Some(Type::Int {
                        width: *w1,
                        signed: *s1,
                    })
                } else {
                    Some(Type::Int {
                        width: *w2,
                        signed: *s2,
                    })
                }
            }
            (Type::Float(w1), Type::Float(w2)) => Some(Type::Float(wider_float(*w1, *w2))),
            (Type::Complex(w1), Type::Complex(w2)) => {
                Some(Type::Complex(wider_float(*w1, *w2)))
            }
            _ => None,
        }
    } else {
        // Higher rank wins
        let hi = if ra > rb { a } else { b };
        Some(hi.clone())
    }
}

fn wider_float(a: FloatWidth, b: FloatWidth) -> FloatWidth {
    if matches!(a, FloatWidth::F64) || matches!(b, FloatWidth::F64) {
        FloatWidth::F64
    } else {
        FloatWidth::F32
    }
}

fn is_angle(ty: &Type) -> bool {
    matches!(ty, Type::Angle(_))
}

fn is_integer(ty: &Type) -> bool {
    matches!(ty, Type::Int { .. } | Type::Bool | Type::Bit)
}

fn angle_width(ty: &Type) -> u32 {
    match ty {
        Type::Angle(w) => *w,
        _ => 0,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolKind;

    #[test]
    fn compile_teleport() {
        let source = include_str!("../../fixtures/qasm/teleport.qasm");
        let program = compile_source(source, None).expect("teleport should compile");

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
        let program = compile_source(source, None).expect("adder should compile");

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
        let program = compile_source(source, None).expect("stdgates should compile");

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
        let program = compile_source("", None).expect("empty source should compile");
        assert!(program.body.is_empty());
        assert!(program.version.is_none());
    }

    #[test]
    fn compile_version_preserved() {
        let program =
            compile_source("OPENQASM 3.0;", None).expect("version-only should compile");
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
        let program = compile_source(source, None).expect("literals should compile");
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
        let program = compile_source(source, None).expect("scope should compile");
        // Both decls should appear in the flattened body
        assert_eq!(program.body.len(), 2);
    }

    #[test]
    fn old_style_decls() {
        let source = r#"
            creg c[4];
            qreg q[2];
        "#;
        let program = compile_source(source, None).expect("old style should compile");
        let c_sym = program.symbols.lookup("c").expect("c should exist");
        let q_sym = program.symbols.lookup("q").expect("q should exist");
        assert_eq!(program.symbols.get(c_sym).kind, SymbolKind::Variable);
        assert_eq!(program.symbols.get(q_sym).kind, SymbolKind::Qubit);
        assert_eq!(program.symbols.get(c_sym).ty, Type::BitReg(4));
        assert_eq!(program.symbols.get(q_sym).ty, Type::QubitReg(2));
    }

    #[test]
    fn include_stdgates_resolves_gates() {
        let source = r#"
            include "stdgates.inc";
            qubit q;
            h q;
        "#;
        let program = compile_source(source, None).expect("should compile with stdgates");
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
        match compile_source(source, None) {
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
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        let sw = usize::BITS;
        assert_eq!(
            e.ty,
            Type::Float(float_width_from_system_width(sw)),
            "1 + 2.0 should be Float"
        );
    }

    #[test]
    fn type_bool_and() {
        let source = "true && false;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Bool, "true && false should be Bool");
    }

    #[test]
    fn type_comparison_is_bool() {
        let source = "int[32] x = 1; x == 2;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Bool, "x == 2 should be Bool");
    }

    #[test]
    fn type_int_literal() {
        let source = "42;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        let sw = usize::BITS;
        assert_eq!(
            e.ty,
            Type::Int {
                width: sw,
                signed: true
            },
            "42 should be Int(system_width, signed)"
        );
    }

    #[test]
    fn type_float_literal() {
        let source = "3.14;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        let sw = usize::BITS;
        assert_eq!(
            e.ty,
            Type::Float(float_width_from_system_width(sw)),
            "3.14 should be Float"
        );
    }

    #[test]
    fn type_bool_literal() {
        let source = "true;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Bool);
    }

    #[test]
    fn type_bitstring_literal() {
        let source = r#""0110";"#;
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::BitReg(4));
    }

    #[test]
    fn type_var_inherits_declared_type() {
        let source = "float[64] x = 1.0; x;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Float(FloatWidth::F64));
    }

    #[test]
    fn type_unary_neg_preserves_type() {
        let source = "float[64] x = 1.0; -x;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Float(FloatWidth::F64));
    }

    #[test]
    fn type_logical_not_is_bool() {
        let source = "bool x = true; !x;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Bool);
    }

    #[test]
    fn type_cast_result() {
        let source = "int[32] x = 5; float[64](x);";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Float(FloatWidth::F64));
    }

    #[test]
    fn type_cast_angle_to_float_invalid() {
        let source = "angle[32] a; float[64](a);";
        match compile_source(source, None) {
            Err(e) => assert!(matches!(e.kind, ErrorKind::TypeMismatch { .. })),
            Ok(_) => panic!("angle → float cast should be rejected"),
        }
    }

    #[test]
    fn type_cast_float_to_angle_valid() {
        let source = "float[64] f = 1.0; angle[32](f);";
        let program = compile_source(source, None).expect("float → angle should be valid");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Angle(32));
    }

    #[test]
    fn type_index_into_qubit_reg() {
        let source = "qubit[4] q; q[0];";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Qubit);
    }

    #[test]
    fn type_index_into_int() {
        let source = "uint[4] x = 5; x[0];";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Bit);
    }

    #[test]
    fn type_angle_arithmetic() {
        // angle + angle → angle
        let source = "angle[32] a; angle[32] b; a + b;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Angle(32));
    }

    #[test]
    fn type_angle_div_angle() {
        // angle / angle → uint
        let source = "angle[32] a; angle[32] b; a / b;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(
            e.ty,
            Type::Int {
                width: 32,
                signed: false
            }
        );
    }

    #[test]
    fn type_angle_mul_uint() {
        // angle * uint → angle
        let source = "angle[32] a; uint[32] n = 2; a * n;";
        let program = compile_source(source, None).expect("should compile");
        let e = find_expr_stmt(&program);
        assert_eq!(e.ty, Type::Angle(32));
    }

    #[test]
    fn type_promote_int_int_same_width() {
        // Same width, one signed one unsigned → unsigned wins (C99)
        let result = promote_numeric(
            &Type::Int {
                width: 32,
                signed: true,
            },
            &Type::Int {
                width: 32,
                signed: false,
            },
        );
        assert_eq!(
            result,
            Some(Type::Int {
                width: 32,
                signed: false
            })
        );
    }

    #[test]
    fn type_promote_int_float() {
        let result = promote_numeric(
            &Type::Int {
                width: 32,
                signed: true,
            },
            &Type::Float(FloatWidth::F64),
        );
        assert_eq!(result, Some(Type::Float(FloatWidth::F64)));
    }

    #[test]
    fn type_promote_float_complex() {
        let result = promote_numeric(&Type::Float(FloatWidth::F64), &Type::Complex(FloatWidth::F32));
        assert_eq!(result, Some(Type::Complex(FloatWidth::F32)));
    }

    #[test]
    fn type_promote_bool_int() {
        let result = promote_numeric(
            &Type::Bool,
            &Type::Int {
                width: 32,
                signed: true,
            },
        );
        assert_eq!(
            result,
            Some(Type::Int {
                width: 32,
                signed: true
            })
        );
    }

    #[test]
    fn type_teleport_condition_is_bool() {
        let source = include_str!("../../fixtures/qasm/teleport.qasm");
        let program = compile_source(source, None).expect("teleport should compile");
        // Find an if-statement and check its condition type
        let if_stmt = program
            .body
            .iter()
            .find(|s| matches!(s.kind, sir::StmtKind::If { .. }));
        assert!(if_stmt.is_some());
        if let sir::StmtKind::If { condition, .. } = &if_stmt.unwrap().kind {
            assert_eq!(condition.ty, Type::Bool, "if condition should be Bool");
        }
    }

    #[test]
    fn type_adder_cast_is_bool() {
        // adder.qasm uses bool(a_in[i]) - verify cast produces Bool
        let source = include_str!("../../fixtures/qasm/adder.qasm");
        let program = compile_source(source, None).expect("adder should compile");
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
                assert_eq!(condition.ty, Type::Bool, "bool() cast should yield Bool");
            }
        }
    }
}
