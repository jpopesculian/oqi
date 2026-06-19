//! Step D + E: SSA → bytecode.
//!
//! Walks each [`SsaCfg`] in `ProgramSsa`, optionally running phi
//! elimination first, then converts every [`SsaStmt`](crate::ssa::SsaStmt) into one or more
//! [`BcInstr`]s. Nested CFGs (Box / inline-Cal / DurationOf) are
//! recursively lowered to their own procedures and referenced by
//! [`ProcId`].
//!
//! Qubit references are lowered through the [`QubitLayout`]: named
//! registers and resolved `let` aliases become global quantum memory
//! references ([`BcOperand::Qubit`] / [`BcOperand::QubitRegion`] /
//! [`BcOperand::QubitIndexed`]); gate and subroutine qubit parameters
//! become positional [`BcOperand::QubitParam`]s bound at call time.

use std::collections::HashMap;

use oqi_classical::{Value, ValueTy};
use oqi_lex::Span;

use crate::classical::PrimitiveTy;
use crate::error::{CompileError, ErrorKind, Result};
use crate::qubits::{self, QubitLayout};
use crate::sir::{
    Alias, Binary, Call, CallTarget, Cast, Delay, GateCall, GateModifier, Index, IndexItem,
    IndexKind, IndexOp, MeasureExpr, MeasureExprKind, QubitOperand, RValue,
};
use crate::ssa::{
    ProgramSsa, SsaAssignment, SsaBlock, SsaBoxStmt, SsaCalibrationBody, SsaCfg, SsaExpr,
    SsaExprKind, SsaLValue, SsaMeasure, SsaStmtKind, SsaTerminator, SsaValue,
};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};

use super::phi_elim::deconstruct_phis;
use super::regalloc::{RegMap, allocate_registers};
use super::types::{
    BcBlock, BcCallTarget, BcGateModifier, BcInstr, BcModule, BcOp, BcOperand, BcProcedure,
    BcSwitchLabels, BcTerminator, BcVersion, BlockId, ConstId, ProcId, ProcOwner, QubitRegion,
    QubitRegionId, QubitTable, Reg, StringId,
};

/// Entry point: lower an SSA program to bytecode against the global
/// qubit layout (see [`crate::qubits::build_layout`]).
pub fn emit(
    ssa: &ProgramSsa,
    symbols: &SymbolTable,
    layout: QubitLayout,
) -> Result<BcModule, CompileError> {
    let mut ctx = EmitCtx::new(symbols, layout);
    let entry = ctx.emit_root(&ssa.top_level, ProcOwner::TopLevel)?;
    for sub in &ssa.subroutines {
        let owner = sub_owner(&sub.owner);
        ctx.emit_root(sub, owner)?;
    }
    for gate in &ssa.gates {
        let owner = gate_owner(&gate.owner);
        ctx.emit_root(gate, owner)?;
    }
    for (i, cal) in ssa.calibrations.iter().enumerate() {
        if let Some(c) = cal {
            ctx.emit_root(c, ProcOwner::Calibration(i as u32))?;
        }
    }

    Ok(BcModule {
        version: BcVersion::CURRENT,
        symbols: symbols.clone(),
        constants: ctx.constants,
        strings: ctx.strings,
        qubits: QubitTable {
            num_qubits: ctx.layout.num_qubits() as u32,
            regions: ctx.qubit_regions,
        },
        procedures: ctx.procedures,
        entry,
        inputs: ctx.inputs,
        outputs: ctx.outputs,
    })
}

fn sub_owner(owner: &crate::cfg::CfgOwner) -> ProcOwner {
    match owner {
        crate::cfg::CfgOwner::Subroutine(s) => ProcOwner::Subroutine(*s),
        _ => unreachable!("subroutine CFG with non-subroutine owner"),
    }
}

fn gate_owner(owner: &crate::cfg::CfgOwner) -> ProcOwner {
    match owner {
        crate::cfg::CfgOwner::Gate(s) => ProcOwner::Gate(*s),
        _ => unreachable!("gate CFG with non-gate owner"),
    }
}

/// Mutable emission state shared across procedures.
struct EmitCtx<'a> {
    symbols: &'a SymbolTable,
    /// Global qubit layout; gains alias definitions during emission.
    layout: QubitLayout,
    qubit_regions: Vec<QubitRegion>,
    /// Resolved ranges of each entry in `qubit_regions`, for dedup.
    region_index: HashMap<Vec<(u32, u32)>, QubitRegionId>,
    procedures: Vec<BcProcedure>,
    constants: Vec<Value>,
    /// Postcard encoding of each entry in `constants`, for dedup.
    const_index: HashMap<Vec<u8>, ConstId>,
    strings: Vec<String>,
    /// The program's input contract, resolved while emitting the
    /// top-level body.
    inputs: Vec<(SymbolId, Reg)>,
    /// Named program outputs, resolved while emitting the top-level body.
    outputs: Vec<(SymbolId, Reg)>,
}

impl<'a> EmitCtx<'a> {
    fn new(symbols: &'a SymbolTable, layout: QubitLayout) -> Self {
        Self {
            symbols,
            layout,
            qubit_regions: Vec::new(),
            region_index: HashMap::new(),
            procedures: Vec::new(),
            constants: Vec::new(),
            const_index: HashMap::new(),
            strings: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Emit a top-level CFG (or a recursive nested one); return its
    /// `ProcId`. Always reserves the slot first so nested emissions
    /// see a stable id.
    fn emit_root(&mut self, cfg: &SsaCfg, owner: ProcOwner) -> Result<ProcId> {
        let id = ProcId(self.procedures.len() as u32);
        // Reserve a placeholder slot. We'll overwrite it after the
        // body is emitted (so nested CFGs can recurse and grab
        // smaller ids if they're emitted first).
        self.procedures.push(BcProcedure {
            owner: owner.clone(),
            register_types: Vec::new(),
            params: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
        });
        let proc = self.emit_proc(cfg, owner)?;
        self.procedures[id.0 as usize] = proc;
        Ok(id)
    }

    fn emit_proc(&mut self, cfg: &SsaCfg, owner: ProcOwner) -> Result<BcProcedure> {
        // Phi-eliminate a clone so the source SSA isn't mutated.
        let cfg = deconstruct_phis(cfg.clone());
        let mut reg_map = allocate_registers(&cfg, self.symbols);

        // Record the calling convention: the register holding each
        // classical parameter (read at version 0). Allocating here is
        // idempotent — a referenced param reuses its existing register,
        // an unreferenced one gets a fresh slot.
        let params = self.param_registers(&owner, &mut reg_map);

        // The top-level body's `input` decls and exit reaching defs
        // become the program's input contract and named outputs.
        if matches!(owner, ProcOwner::TopLevel) {
            self.inputs = self.collect_inputs(&mut reg_map);
            self.outputs = self.collect_outputs(&cfg, &reg_map);
        }

        let blocks: Vec<BcBlock> = cfg
            .blocks
            .iter()
            .map(|b| self.emit_block(b, &mut reg_map))
            .collect::<Result<_>>()?;

        Ok(BcProcedure {
            owner,
            register_types: reg_map.types,
            params,
            blocks,
            entry: BlockId(cfg.entry.0 as u32),
        })
    }

    /// The program's input contract: a `(symbol, reg)` for every
    /// `input`-declared variable. The register is the symbol's version-0
    /// (entry) value, force-allocated here so even an unread input has a
    /// seedable slot — mirroring how `param_registers` allocates
    /// unreferenced parameters. Sorted by symbol id (declaration order).
    fn collect_inputs(&self, reg_map: &mut RegMap) -> Vec<(SymbolId, Reg)> {
        let mut inputs: Vec<(SymbolId, Reg)> = self
            .symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Input)
            .filter_map(|s| {
                let ty = s.ty.value_ty()?;
                let reg = reg_map.alloc(
                    SsaValue {
                        symbol: s.id,
                        version: 0,
                    },
                    ty,
                );
                Some((s.id, reg))
            })
            .collect();
        inputs.sort_by_key(|(sym, _)| sym.0);
        inputs
    }

    /// Resolve the top-level body's exit reaching defs to named program
    /// outputs. Per OpenQASM 3: if any `output` is declared, only those
    /// symbols are outputs; otherwise every named classical variable is.
    /// Only symbols actually assigned at top level (version ≥ 1, with an
    /// allocated register) are included. Sorted by symbol id for
    /// deterministic bytecode.
    fn collect_outputs(&self, cfg: &SsaCfg, reg_map: &RegMap) -> Vec<(SymbolId, Reg)> {
        let has_outputs = self.symbols.iter().any(|s| s.kind == SymbolKind::Output);
        let wanted = if has_outputs {
            SymbolKind::Output
        } else {
            SymbolKind::Variable
        };
        let mut outputs: Vec<(SymbolId, Reg)> = cfg
            .exit_defs
            .iter()
            .filter(|(sym, ssa)| ssa.version >= 1 && self.symbols.get(**sym).kind == wanted)
            .filter_map(|(sym, ssa)| reg_map.by_ssa.get(ssa).map(|reg| (*sym, *reg)))
            .collect();
        outputs.sort_by_key(|(sym, _)| sym.0);
        outputs
    }

    /// Registers holding the classical parameters of a gate/subroutine,
    /// in declaration order. Empty for owners that take no classical
    /// parameters.
    fn param_registers(&self, owner: &ProcOwner, reg_map: &mut RegMap) -> Vec<Reg> {
        let sym = match owner {
            ProcOwner::Gate(s) | ProcOwner::Subroutine(s) => *s,
            _ => return Vec::new(),
        };
        let params = self.layout.classical_params(sym).to_vec();
        params
            .into_iter()
            .map(|symbol| {
                let v = SsaValue { symbol, version: 0 };
                let ty = self
                    .symbols
                    .get(symbol)
                    .ty
                    .value_ty()
                    .expect("classical parameter must have a value type");
                reg_map.alloc(v, ty)
            })
            .collect()
    }

    fn emit_block(&mut self, block: &SsaBlock, reg_map: &mut RegMap) -> Result<BcBlock> {
        // phi_elim has already cleared phis; if any remain (e.g. a
        // future caller skipped phi elimination), record them as
        // placeholder Moves — but assert empty to keep the bytecode
        // honest.
        debug_assert!(
            block.phis.is_empty(),
            "phi_elim must clear all phis before emit_block"
        );

        let mut instrs: Vec<BcInstr> = Vec::new();
        for stmt in &block.stmts {
            self.emit_stmt(stmt, &mut instrs, reg_map)?;
        }
        let terminator =
            self.emit_terminator(&block.terminator, &mut instrs, reg_map, block.span)?;

        Ok(BcBlock {
            id: BlockId(block.id.0 as u32),
            instrs,
            terminator,
            span: block.span,
        })
    }

    fn emit_stmt(
        &mut self,
        stmt: &crate::ssa::SsaStmt,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> Result<()> {
        let span = stmt.span;
        match &stmt.kind {
            SsaStmtKind::Alias(Alias { symbol, value }) => {
                // Qubit aliases resolve into the layout and emit
                // nothing; classical aliases keep their metadata op.
                match self.layout.resolve_alias_value(value, self.symbols)? {
                    Some(reg) => self.layout.define_alias(*symbol, reg),
                    None => {
                        let value = value
                            .iter()
                            .map(|e| self.lower_expr_to_operand(e, instrs, reg_map))
                            .collect::<Result<_>>()?;
                        instrs.push(BcInstr {
                            op: BcOp::Alias {
                                symbol: *symbol,
                                value,
                            },
                            span,
                        });
                    }
                }
            }
            SsaStmtKind::GateCall(GateCall {
                gate,
                modifiers,
                args,
                qubits,
            }) => {
                // A bare gate-call whose name resolves to a `def` (e.g.
                // `hadamard_layer ancilla;`) is a subroutine call, not a gate.
                // Redirect it to a `Call`, binding the classical args followed
                // by the qubit operands (the order bare syntax always yields).
                if self.symbols.get(*gate).kind == SymbolKind::Subroutine {
                    if !modifiers.is_empty() {
                        return Err(CompileError::new(ErrorKind::InvalidContext(
                            "gate modifiers cannot be applied to a subroutine call".into(),
                        ))
                        .with_span(span));
                    }
                    let mut call_args: Vec<BcOperand> = args
                        .iter()
                        .map(|a| self.lower_expr_to_operand(a, instrs, reg_map))
                        .collect::<Result<_>>()?;
                    for q in qubits {
                        call_args.push(self.lower_qubit_operand(q, instrs, reg_map, span)?);
                    }
                    instrs.push(BcInstr {
                        op: BcOp::Call {
                            dest: None,
                            callee: BcCallTarget::Symbol(*gate),
                            args: call_args,
                        },
                        span,
                    });
                    return Ok(());
                }
                let modifiers: Vec<BcGateModifier> = modifiers
                    .iter()
                    .map(|m| self.lower_gate_modifier(m, instrs, reg_map))
                    .collect::<Result<_>>()?;
                let args: Vec<BcOperand> = args
                    .iter()
                    .map(|a| self.lower_expr_to_operand(a, instrs, reg_map))
                    .collect::<Result<_>>()?;
                let qubits: Vec<BcOperand> = qubits
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map, span))
                    .collect::<Result<_>>()?;
                instrs.push(BcInstr {
                    op: BcOp::GateCall {
                        gate: *gate,
                        modifiers,
                        args,
                        qubits,
                    },
                    span,
                });
            }
            SsaStmtKind::Measure(SsaMeasure { measure, target }) => match target {
                // `measure q -> lv;` assigns like `lv = measure q;` —
                // reuse the assignment path so an indexed target gets
                // its StoreElement.
                Some(t) => {
                    let value = RValue::Measure(Box::new(measure.clone()));
                    self.emit_assignment(t, &value, instrs, reg_map, span)?;
                }
                None => {
                    let qubit = self.measure_to_operand(measure, instrs, reg_map)?;
                    instrs.push(BcInstr {
                        op: BcOp::Measure { dest: None, qubit },
                        span,
                    });
                }
            },
            SsaStmtKind::Reset(q) => {
                let qubit = self.lower_qubit_operand(q, instrs, reg_map, span)?;
                instrs.push(BcInstr {
                    op: BcOp::Reset { qubit },
                    span,
                });
            }
            SsaStmtKind::Barrier(qs) => {
                let qubits = qs
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map, span))
                    .collect::<Result<_>>()?;
                instrs.push(BcInstr {
                    op: BcOp::Barrier { qubits },
                    span,
                });
            }
            SsaStmtKind::Delay(Delay { duration, operands }) => {
                let duration = self.lower_expr_to_operand(duration, instrs, reg_map)?;
                let qubits = operands
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map, span))
                    .collect::<Result<_>>()?;
                instrs.push(BcInstr {
                    op: BcOp::Delay { duration, qubits },
                    span,
                });
            }
            SsaStmtKind::Box(SsaBoxStmt { duration, body }) => {
                let duration = duration
                    .as_ref()
                    .map(|e| self.lower_expr_to_operand(e, instrs, reg_map))
                    .transpose()?;
                let body_id = self.emit_root(body, ProcOwner::Box)?;
                instrs.push(BcInstr {
                    op: BcOp::Box {
                        duration,
                        body: body_id,
                    },
                    span,
                });
            }
            SsaStmtKind::Assignment(SsaAssignment { target, value }) => {
                self.emit_assignment(target, value, instrs, reg_map, span)?;
            }
            SsaStmtKind::Pragma(s) => {
                let content = self.intern_string(s.clone());
                instrs.push(BcInstr {
                    op: BcOp::Pragma { content },
                    span,
                });
            }
            SsaStmtKind::Cal(body) => match body {
                SsaCalibrationBody::Opaque(s) => {
                    let content = self.intern_string(s.clone());
                    instrs.push(BcInstr {
                        op: BcOp::CalOpaque { content },
                        span,
                    });
                }
                SsaCalibrationBody::OpenPulse(cfg) => {
                    let body_id = self.emit_root(cfg, ProcOwner::InlineCal)?;
                    instrs.push(BcInstr {
                        op: BcOp::CalOpenPulse { body: body_id },
                        span,
                    });
                }
            },
            SsaStmtKind::ExprStmt(e) => {
                let _ = self.lower_expr_to_operand(e, instrs, reg_map)?;
                // Pure-side-effect statements (calls etc.) get emitted
                // inside lower_expr_to_operand. Top-level value is
                // discarded.
            }
            SsaStmtKind::Nop(qs) => {
                let qubits = qs
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map, span))
                    .collect::<Result<_>>()?;
                instrs.push(BcInstr {
                    op: BcOp::Nop { qubits },
                    span,
                });
            }
        }
        Ok(())
    }

    fn emit_assignment(
        &mut self,
        target: &SsaLValue,
        value: &RValue<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) -> Result<()> {
        match target {
            SsaLValue::Var(v) => {
                let dest = self.reg_for(*v, reg_map);
                self.emit_rvalue_to_dest(value, dest, instrs, reg_map, span)?;
            }
            SsaLValue::Indexed { old, new, indices } => {
                // Indexed store: read `old`, compute `new = old[index] = value`.
                let base = BcOperand::Reg(self.reg_for(*old, reg_map));
                let new_reg = self.reg_for(*new, reg_map);
                // A slice or multi-element target (`reg[a:b] = ...`) writes a
                // multi-bit value across several positions. Resolve the
                // positions statically and emit a slice store; a single
                // element keeps the scalar StoreElement path below.
                if let Some(io) = indices.first().filter(|io| is_multi_index(io)) {
                    if let Some(len) = self.symbol_bit_len(old.symbol) {
                        let positions = qubits::resolve_static_index(io, len)?;
                        let value = self.rvalue_to_operand(value, instrs, reg_map)?;
                        instrs.push(BcInstr {
                            op: BcOp::StoreSlice {
                                new: new_reg,
                                base,
                                indices: positions.iter().map(|&i| i as u32).collect(),
                                value,
                            },
                            span,
                        });
                        return Ok(());
                    }
                }
                // OpenQASM supports multi-dim indexed assignment via
                // multiple index ops; the bytecode currently flattens
                // each as a separate StoreElement chained through a
                // single new register. v1: only the first dim.
                let index = if let Some(io) = indices.first() {
                    self.lower_index_op_to_operand(io, instrs, reg_map)?
                } else {
                    BcOperand::Const(self.intern_const(Value::int(0, oqi_classical::iw(64))))
                };
                let value = self.rvalue_to_operand(value, instrs, reg_map)?;
                instrs.push(BcInstr {
                    op: BcOp::StoreElement {
                        new: new_reg,
                        base,
                        index,
                        value,
                    },
                    span,
                });
            }
        }
        Ok(())
    }

    fn emit_rvalue_to_dest(
        &mut self,
        rv: &RValue<SsaExpr>,
        dest: Reg,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) -> Result<()> {
        match rv {
            RValue::Expr(e) => self.emit_expr_to_dest(e, dest, instrs, reg_map, span)?,
            RValue::Measure(m) => {
                let qubit = self.measure_to_operand(m, instrs, reg_map)?;
                instrs.push(BcInstr {
                    op: BcOp::Measure {
                        dest: Some(dest),
                        qubit,
                    },
                    span,
                });
            }
        }
        Ok(())
    }

    fn emit_expr_to_dest(
        &mut self,
        e: &SsaExpr,
        dest: Reg,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) -> Result<()> {
        match &e.kind {
            SsaExprKind::Binary(Binary { op, left, right }) => {
                let lhs = self.lower_expr_to_operand(left, instrs, reg_map)?;
                let rhs = self.lower_expr_to_operand(right, instrs, reg_map)?;
                let bc = match op {
                    crate::sir::BinOp::Add => BcOp::Add { dest, lhs, rhs },
                    crate::sir::BinOp::Sub => BcOp::Sub { dest, lhs, rhs },
                    crate::sir::BinOp::Mul => BcOp::Mul { dest, lhs, rhs },
                    crate::sir::BinOp::Div => BcOp::Div { dest, lhs, rhs },
                    crate::sir::BinOp::Mod => BcOp::Mod { dest, lhs, rhs },
                    crate::sir::BinOp::Pow => BcOp::Pow { dest, lhs, rhs },
                    crate::sir::BinOp::BitAnd => BcOp::BitAnd { dest, lhs, rhs },
                    crate::sir::BinOp::BitOr => BcOp::BitOr { dest, lhs, rhs },
                    crate::sir::BinOp::BitXor => BcOp::BitXor { dest, lhs, rhs },
                    crate::sir::BinOp::Shl => BcOp::Shl { dest, lhs, rhs },
                    crate::sir::BinOp::Shr => BcOp::Shr { dest, lhs, rhs },
                    crate::sir::BinOp::LogAnd => BcOp::LogAnd { dest, lhs, rhs },
                    crate::sir::BinOp::LogOr => BcOp::LogOr { dest, lhs, rhs },
                    crate::sir::BinOp::Eq => BcOp::Eq { dest, lhs, rhs },
                    crate::sir::BinOp::Neq => BcOp::Neq { dest, lhs, rhs },
                    crate::sir::BinOp::Lt => BcOp::Lt { dest, lhs, rhs },
                    crate::sir::BinOp::Gt => BcOp::Gt { dest, lhs, rhs },
                    crate::sir::BinOp::Lte => BcOp::Le { dest, lhs, rhs },
                    crate::sir::BinOp::Gte => BcOp::Ge { dest, lhs, rhs },
                };
                instrs.push(BcInstr { op: bc, span });
            }
            SsaExprKind::Unary(crate::sir::Unary { op, operand }) => {
                let src = self.lower_expr_to_operand(operand, instrs, reg_map)?;
                let bc = match op {
                    crate::sir::UnOp::Neg => BcOp::Neg { dest, src },
                    crate::sir::UnOp::BitNot => BcOp::BitNot { dest, src },
                    crate::sir::UnOp::LogNot => BcOp::LogNot { dest, src },
                };
                instrs.push(BcInstr { op: bc, span });
            }
            SsaExprKind::Cast(Cast { target_ty, operand }) => {
                let src = self.lower_expr_to_operand(operand, instrs, reg_map)?;
                let target_ty = target_ty
                    .value_ty()
                    .unwrap_or(ValueTy::Scalar(PrimitiveTy::Bool));
                instrs.push(BcInstr {
                    op: BcOp::Cast {
                        dest,
                        target_ty,
                        src,
                    },
                    span,
                });
            }
            SsaExprKind::Index(Index { base, index }) => {
                let base = self.lower_expr_to_operand(base, instrs, reg_map)?;
                let index = self.lower_index_op_to_operand(index, instrs, reg_map)?;
                instrs.push(BcInstr {
                    op: BcOp::LoadElement { dest, base, index },
                    span,
                });
            }
            SsaExprKind::Call(Call { callee, args }) => {
                let args = args
                    .iter()
                    .map(|a| self.lower_expr_to_operand(a, instrs, reg_map))
                    .collect::<Result<_>>()?;
                let callee = match callee {
                    CallTarget::Symbol(s) => BcCallTarget::Symbol(*s),
                    CallTarget::Intrinsic(i) => BcCallTarget::Intrinsic(i.clone()),
                };
                instrs.push(BcInstr {
                    op: BcOp::Call {
                        dest: Some(dest),
                        callee,
                        args,
                    },
                    span,
                });
            }
            SsaExprKind::ArrayLiteral(arr) => {
                let items = arr
                    .items
                    .iter()
                    .map(|x| self.lower_expr_to_operand(x, instrs, reg_map))
                    .collect::<Result<_>>()?;
                instrs.push(BcInstr {
                    op: BcOp::NewArray { dest, items },
                    span,
                });
            }
            SsaExprKind::DurationOf(cfg) => {
                let body_id = self.emit_root(cfg, ProcOwner::DurationOf)?;
                instrs.push(BcInstr {
                    op: BcOp::DurationOf {
                        dest,
                        body: body_id,
                    },
                    span,
                });
            }
            // Trivial cases: just produce an operand and Move it.
            SsaExprKind::Literal(_) | SsaExprKind::Var(_) | SsaExprKind::HardwareQubit(_) => {
                let src = self.lower_expr_to_operand(e, instrs, reg_map)?;
                instrs.push(BcInstr {
                    op: BcOp::Move { dest, src },
                    span,
                });
            }
        }
        Ok(())
    }

    /// Lower an SsaExpr into a single BcOperand, emitting intermediate
    /// instructions for any non-trivial sub-expression. Qubit
    /// references (named registers, aliases, parameters — bare or
    /// indexed) are resolved through the layout so that call arguments
    /// carry global qubit references.
    fn lower_expr_to_operand(
        &mut self,
        e: &SsaExpr,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> Result<BcOperand> {
        if let Some((sym, ops)) = self.peel_qubit_ref(e) {
            let ops: Vec<IndexOp<SsaExpr>> = ops.into_iter().cloned().collect();
            return self.resolve_qubit_ref(sym, &ops, instrs, reg_map, e.span);
        }
        Ok(match &e.kind {
            SsaExprKind::Literal(p) => {
                let v = primitive_to_value(p.clone(), &e.ty);
                BcOperand::Const(self.intern_const(v))
            }
            SsaExprKind::Var(v) => {
                // Compile-time constants (`pi`, `tau`, `euler`, user
                // `const`s) carry their value in the symbol table; inline
                // it as a pooled constant rather than a register read that
                // nothing ever assigns.
                match self.symbols.get(v.symbol).const_value.clone() {
                    Some(value) => BcOperand::Const(self.intern_const(value)),
                    None => BcOperand::Reg(self.reg_for(*v, reg_map)),
                }
            }
            SsaExprKind::HardwareQubit(n) => BcOperand::HardwareQubit(*n as u32),
            // Anything else: spill into a synthetic temp register.
            _ => {
                let ty =
                    e.ty.value_ty()
                        .unwrap_or(ValueTy::Scalar(PrimitiveTy::Bool));
                let temp = self.alloc_temp_reg(reg_map, ty);
                self.emit_expr_to_dest(e, temp, instrs, reg_map, e.span)?;
                BcOperand::Reg(temp)
            }
        })
    }

    fn lower_qubit_operand(
        &mut self,
        q: &QubitOperand<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) -> Result<BcOperand> {
        match q {
            QubitOperand::Indexed { symbol, indices } => {
                self.resolve_qubit_ref(*symbol, indices, instrs, reg_map, span)
            }
            QubitOperand::Hardware(n) => Ok(BcOperand::HardwareQubit(*n as u32)),
        }
    }

    /// Resolve a (possibly indexed) named qubit reference to a global
    /// memory operand, or to a positional parameter reference inside
    /// gate/subroutine bodies.
    fn resolve_qubit_ref(
        &mut self,
        symbol: SymbolId,
        indices: &[IndexOp<SsaExpr>],
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) -> Result<BcOperand> {
        // Gate / subroutine qubit parameter: bound at call time.
        if let Some(slot) = self.layout.param_slot(symbol) {
            return match indices {
                [] => Ok(BcOperand::QubitParam { slot, index: None }),
                [io] => match qubits::single_index_expr(io) {
                    Some(e) => {
                        let index = Box::new(self.lower_expr_to_operand(e, instrs, reg_map)?);
                        Ok(BcOperand::QubitParam {
                            slot,
                            index: Some(index),
                        })
                    }
                    None => Err(CompileError::new(ErrorKind::Unsupported(
                        "slices of qubit parameters are not supported".into(),
                    ))
                    .with_span(io.span)),
                },
                _ => Err(CompileError::new(ErrorKind::Unsupported(
                    "multi-dimensional index on a quantum register".into(),
                ))
                .with_span(span)),
            };
        }

        // Declared register or resolved alias: global memory.
        if let Some(reg) = self.layout.register_of(symbol) {
            let reg = reg.clone();
            return match indices {
                [] => {
                    if reg.len() == 1 {
                        Ok(BcOperand::Qubit(self.must_global(&reg, 0)))
                    } else {
                        Ok(BcOperand::QubitRegion(self.region_for(&reg, Some(symbol))))
                    }
                }
                [io] => match qubits::resolve_static_index(io, reg.len()) {
                    Ok(idxs) => {
                        if let [only] = idxs.as_slice() {
                            Ok(BcOperand::Qubit(self.must_global(&reg, *only)))
                        } else {
                            let sub = qubits::select(&reg, &idxs);
                            Ok(BcOperand::QubitRegion(self.region_for(&sub, Some(symbol))))
                        }
                    }
                    // Runtime single index: the VM maps the logical
                    // index through the region's ranges.
                    Err(e) if matches!(e.kind, ErrorKind::NonConstantExpression) => {
                        match qubits::single_index_expr(io) {
                            Some(e) => {
                                let region = self.region_for(&reg, Some(symbol));
                                let index =
                                    Box::new(self.lower_expr_to_operand(e, instrs, reg_map)?);
                                Ok(BcOperand::QubitIndexed { region, index })
                            }
                            None => Err(CompileError::new(ErrorKind::Unsupported(
                                "runtime-valued slice of a quantum register".into(),
                            ))
                            .with_span(io.span)),
                        }
                    }
                    Err(e) => Err(e),
                },
                _ => Err(CompileError::new(ErrorKind::Unsupported(
                    "multi-dimensional index on a quantum register".into(),
                ))
                .with_span(span)),
            };
        }

        Err(CompileError::new(ErrorKind::Unsupported(format!(
            "cannot resolve qubit reference `{}`",
            self.symbols.get(symbol).name
        )))
        .with_span(span))
    }

    /// Whether `e` is a (possibly indexed) reference to a named qubit
    /// register, qubit alias, or qubit parameter; returns the base
    /// symbol and the index ops in application order.
    fn peel_qubit_ref<'e>(&self, e: &'e SsaExpr) -> Option<(SymbolId, Vec<&'e IndexOp<SsaExpr>>)> {
        let (sym, ops) = qubits::peel_index_chain(e)?;
        let qubit_like = self.layout.param_slot(sym).is_some()
            || self.layout.register_of(sym).is_some()
            || matches!(
                self.symbols.get(sym).kind,
                SymbolKind::Qubit | SymbolKind::GateQubit
            );
        qubit_like.then_some((sym, ops))
    }

    fn must_global(&self, reg: &oqi_quantum::QuantumRegister, local: usize) -> u32 {
        self.layout
            .global_index(reg, local)
            .expect("local index validated against register length") as u32
    }

    /// Intern a region for `reg`, deduplicating on its global ranges.
    fn region_for(
        &mut self,
        reg: &oqi_quantum::QuantumRegister,
        origin: Option<SymbolId>,
    ) -> QubitRegionId {
        let ranges = self.layout.global_ranges(reg);
        if let Some(&id) = self.region_index.get(&ranges) {
            return id;
        }
        let id = QubitRegionId(self.qubit_regions.len() as u32);
        self.qubit_regions.push(QubitRegion {
            ranges: ranges.clone(),
            origin,
        });
        self.region_index.insert(ranges, id);
        id
    }

    fn lower_index_op_to_operand(
        &mut self,
        io: &IndexOp<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> Result<BcOperand> {
        // v1: only single-item indices. Slices/sets collapse to their
        // first element. (A richer model would lift slicing into its
        // own opcodes.)
        Ok(match &io.kind {
            IndexKind::Set(es) => {
                if let Some(e) = es.first() {
                    self.lower_expr_to_operand(e, instrs, reg_map)?
                } else {
                    BcOperand::Const(self.intern_const(Value::int(0, oqi_classical::iw(64))))
                }
            }
            IndexKind::Items(items) => {
                if let Some(item) = items.first() {
                    match item {
                        IndexItem::Single(e) => self.lower_expr_to_operand(e, instrs, reg_map)?,
                        IndexItem::Range(r) => {
                            if let Some(start) = &r.start {
                                self.lower_expr_to_operand(start, instrs, reg_map)?
                            } else {
                                BcOperand::Const(
                                    self.intern_const(Value::int(0, oqi_classical::iw(64))),
                                )
                            }
                        }
                    }
                } else {
                    BcOperand::Const(self.intern_const(Value::int(0, oqi_classical::iw(64))))
                }
            }
        })
    }

    /// Number of indexable bits of a declared bit-register symbol, used to
    /// resolve negative and open-ended slice bounds. `None` for symbols that
    /// aren't fixed-width bit registers.
    fn symbol_bit_len(&self, sym: SymbolId) -> Option<usize> {
        match self.symbols.get(sym).ty.value_ty()? {
            ValueTy::Scalar(p) => p.bit_count().map(|w| w as usize),
            _ => None,
        }
    }

    fn lower_gate_modifier(
        &mut self,
        m: &GateModifier<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> Result<BcGateModifier> {
        Ok(match m {
            GateModifier::Inv => BcGateModifier::Inv,
            GateModifier::Pow(e) => {
                BcGateModifier::Pow(self.lower_expr_to_operand(e, instrs, reg_map)?)
            }
            GateModifier::Ctrl(n) => BcGateModifier::Ctrl(*n as u32),
            GateModifier::NegCtrl(n) => BcGateModifier::NegCtrl(*n as u32),
        })
    }

    /// Lower the qubit being measured to an operand.
    fn measure_to_operand(
        &mut self,
        m: &MeasureExpr<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> Result<BcOperand> {
        match &m.kind {
            MeasureExprKind::Measure { operand } => {
                self.lower_qubit_operand(operand, instrs, reg_map, m.span)
            }
            MeasureExprKind::QuantumCall { qubits, .. } => {
                // QuantumCall in measure position: lower its first
                // qubit as the measure target. (Multi-qubit
                // measurement semantics aren't yet modeled in
                // bytecode.)
                qubits
                    .first()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map, m.span))
                    .unwrap_or(Ok(BcOperand::HardwareQubit(0)))
            }
        }
    }

    fn rvalue_to_operand(
        &mut self,
        rv: &RValue<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> Result<BcOperand> {
        match rv {
            RValue::Expr(e) => self.lower_expr_to_operand(e, instrs, reg_map),
            RValue::Measure(m) => {
                let qubit = self.measure_to_operand(m, instrs, reg_map)?;
                let ty = m.ty.value_ty().unwrap_or(ValueTy::Scalar(PrimitiveTy::Bit));
                let temp = self.alloc_temp_reg(reg_map, ty);
                instrs.push(BcInstr {
                    op: BcOp::Measure {
                        dest: Some(temp),
                        qubit,
                    },
                    span: m.span,
                });
                Ok(BcOperand::Reg(temp))
            }
        }
    }

    fn emit_terminator(
        &mut self,
        term: &SsaTerminator,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        _span: Span,
    ) -> Result<BcTerminator> {
        Ok(match term {
            SsaTerminator::Goto(b) => BcTerminator::Goto(BlockId(b.0 as u32)),
            SsaTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let cond = self.lower_expr_to_operand(cond, instrs, reg_map)?;
                BcTerminator::Branch {
                    cond,
                    then_bb: BlockId(then_bb.0 as u32),
                    else_bb: BlockId(else_bb.0 as u32),
                }
            }
            SsaTerminator::Switch {
                target,
                cases,
                default,
            } => {
                let target = self.lower_expr_to_operand(target, instrs, reg_map)?;
                let cases = cases
                    .iter()
                    .map(|(labels, bb)| {
                        let lab = match labels {
                            crate::sir::SwitchLabels::Default => BcSwitchLabels::Default,
                            crate::sir::SwitchLabels::Values(vs) => BcSwitchLabels::Values(
                                vs.iter()
                                    .map(|v| self.lower_expr_to_operand(v, instrs, reg_map))
                                    .collect::<Result<_>>()?,
                            ),
                        };
                        Ok((lab, BlockId(bb.0 as u32)))
                    })
                    .collect::<Result<_>>()?;
                BcTerminator::Switch {
                    target,
                    cases,
                    default: default.map(|b| BlockId(b.0 as u32)),
                }
            }
            SsaTerminator::Return(rv) => BcTerminator::Return(
                rv.as_ref()
                    .map(|r| self.rvalue_to_operand(r, instrs, reg_map))
                    .transpose()?,
            ),
            SsaTerminator::End => BcTerminator::End,
            SsaTerminator::Unreachable => BcTerminator::Unreachable,
        })
    }

    // ── Helpers ────────────────────────────────────────────────────

    fn reg_for(&self, v: SsaValue, reg_map: &mut RegMap) -> Reg {
        if let Some(r) = reg_map.by_ssa.get(&v) {
            return *r;
        }
        // Symbol lookup; fall back to bool type if non-classical.
        let ty = self
            .symbols
            .get(v.symbol)
            .ty
            .value_ty()
            .unwrap_or(ValueTy::Scalar(PrimitiveTy::Bool));
        reg_map.alloc(v, ty)
    }

    fn alloc_temp_reg(&self, reg_map: &mut RegMap, ty: ValueTy) -> Reg {
        // Synthetic reg with no SsaValue mapping — just push a slot.
        let r = Reg(reg_map.types.len() as u32);
        reg_map.types.push(ty);
        r
    }

    fn intern_const(&mut self, v: Value) -> ConstId {
        // Dedup keyed on the postcard encoding: bit-exact, so values
        // that differ only in float payload (e.g. NaNs) stay distinct.
        let key = postcard::to_allocvec(&v).expect("constant Value should encode");
        if let Some(&id) = self.const_index.get(&key) {
            return id;
        }
        let id = ConstId(self.constants.len() as u32);
        self.constants.push(v);
        self.const_index.insert(key, id);
        id
    }

    fn intern_string(&mut self, s: String) -> StringId {
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s);
        id
    }
}

/// True when an lvalue index selects more than one position — a range
/// (`a:b`) or a multi-element discrete set — so the store target is a slice
/// rather than a single element.
fn is_multi_index(io: &IndexOp<SsaExpr>) -> bool {
    match &io.kind {
        IndexKind::Set(es) => es.len() > 1,
        IndexKind::Items(items) => !matches!(items.as_slice(), [IndexItem::Single(_)]),
    }
}

fn primitive_to_value(p: crate::classical::Primitive, ty: &crate::types::Type) -> Value {
    // Wrap a Primitive into the canonical Value by recovering a
    // PrimitiveTy from the surrounding `Type`.
    let vty = ty
        .value_ty()
        .unwrap_or(ValueTy::Scalar(crate::classical::PrimitiveTy::Bit));
    match vty {
        ValueTy::Scalar(pty) => Value::Scalar(crate::classical::Scalar::new_unchecked(p, pty)),
        // Non-scalar literals shouldn't appear at this level; fall
        // back to scalar bit.
        _ => Value::Scalar(crate::classical::Scalar::new_unchecked(
            p,
            crate::classical::PrimitiveTy::Bit,
        )),
    }
}
