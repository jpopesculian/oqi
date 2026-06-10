//! Step D + E: SSA → bytecode.
//!
//! Walks each [`SsaCfg`] in `ProgramSsa`, optionally running phi
//! elimination first, then converts every [`SsaStmt`] into one or more
//! [`BcInstr`]s. Nested CFGs (Box / inline-Cal / DurationOf) are
//! recursively lowered to their own procedures and referenced by
//! [`ProcId`].

use std::collections::HashMap;

use oqi_classical::{Value, ValueTy};
use oqi_lex::Span;

use crate::classical::PrimitiveTy;
use crate::sir::{
    Alias, Binary, Call, CallTarget, Cast, Delay, GateCall, GateModifier, Index, IndexItem,
    IndexKind, IndexOp, MeasureExpr, MeasureExprKind, QubitOperand, RValue,
};
use crate::ssa::{
    ProgramSsa, SsaAssignment, SsaBlock, SsaBoxStmt, SsaCalibrationBody, SsaCfg, SsaExpr,
    SsaExprKind, SsaLValue, SsaMeasure, SsaStmtKind, SsaTerminator, SsaValue,
};
use crate::symbol::SymbolTable;

use super::phi_elim::deconstruct_phis;
use super::regalloc::{RegMap, allocate_registers};
use super::types::{
    BcBlock, BcCallTarget, BcGateModifier, BcInstr, BcModule, BcOp, BcOperand, BcProcedure,
    BcSwitchLabels, BcTerminator, BcVersion, BlockId, ConstId, ProcId, ProcOwner, Reg, StringId,
};

/// Entry point: lower an SSA program to bytecode.
pub fn emit(ssa: &ProgramSsa, symbols: &SymbolTable) -> BcModule {
    let mut ctx = EmitCtx::new(symbols);
    let entry = ctx.emit_root(&ssa.top_level, ProcOwner::TopLevel);
    for sub in &ssa.subroutines {
        let owner = sub_owner(&sub.owner);
        ctx.emit_root(sub, owner);
    }
    for gate in &ssa.gates {
        let owner = gate_owner(&gate.owner);
        ctx.emit_root(gate, owner);
    }
    for (i, cal) in ssa.calibrations.iter().enumerate() {
        if let Some(c) = cal {
            ctx.emit_root(c, ProcOwner::Calibration(i as u32));
        }
    }

    BcModule {
        version: BcVersion::CURRENT,
        symbols: symbols.clone(),
        constants: ctx.constants,
        strings: ctx.strings,
        procedures: ctx.procedures,
        entry,
    }
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
    procedures: Vec<BcProcedure>,
    constants: Vec<Value>,
    /// Postcard encoding of each entry in `constants`, for dedup.
    const_index: HashMap<Vec<u8>, ConstId>,
    strings: Vec<String>,
}

impl<'a> EmitCtx<'a> {
    fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            procedures: Vec::new(),
            constants: Vec::new(),
            const_index: HashMap::new(),
            strings: Vec::new(),
        }
    }

    /// Emit a top-level CFG (or a recursive nested one); return its
    /// `ProcId`. Always reserves the slot first so nested emissions
    /// see a stable id.
    fn emit_root(&mut self, cfg: &SsaCfg, owner: ProcOwner) -> ProcId {
        let id = ProcId(self.procedures.len() as u32);
        // Reserve a placeholder slot. We'll overwrite it after the
        // body is emitted (so nested CFGs can recurse and grab
        // smaller ids if they're emitted first).
        self.procedures.push(BcProcedure {
            owner: owner.clone(),
            register_types: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
        });
        let proc = self.emit_proc(cfg, owner);
        self.procedures[id.0 as usize] = proc;
        id
    }

    fn emit_proc(&mut self, cfg: &SsaCfg, owner: ProcOwner) -> BcProcedure {
        // Phi-eliminate a clone so the source SSA isn't mutated.
        let cfg = deconstruct_phis(cfg.clone());
        let mut reg_map = allocate_registers(&cfg, self.symbols);

        let blocks: Vec<BcBlock> = cfg
            .blocks
            .iter()
            .map(|b| self.emit_block(b, &mut reg_map))
            .collect();

        BcProcedure {
            owner,
            register_types: reg_map.types,
            blocks,
            entry: BlockId(cfg.entry.0 as u32),
        }
    }

    fn emit_block(&mut self, block: &SsaBlock, reg_map: &mut RegMap) -> BcBlock {
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
            self.emit_stmt(stmt, &mut instrs, reg_map);
        }
        let terminator = self.emit_terminator(&block.terminator, &mut instrs, reg_map, block.span);

        BcBlock {
            id: BlockId(block.id.0 as u32),
            instrs,
            terminator,
            span: block.span,
        }
    }

    fn emit_stmt(
        &mut self,
        stmt: &crate::ssa::SsaStmt,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) {
        let span = stmt.span;
        match &stmt.kind {
            SsaStmtKind::Alias(Alias { symbol, value }) => {
                let value = value
                    .iter()
                    .map(|e| self.lower_expr_to_operand(e, instrs, reg_map))
                    .collect();
                instrs.push(BcInstr {
                    op: BcOp::Alias {
                        symbol: *symbol,
                        value,
                    },
                    span,
                });
            }
            SsaStmtKind::GateCall(GateCall {
                gate,
                modifiers,
                args,
                qubits,
            }) => {
                let modifiers: Vec<BcGateModifier> = modifiers
                    .iter()
                    .map(|m| self.lower_gate_modifier(m, instrs, reg_map))
                    .collect();
                let args: Vec<BcOperand> = args
                    .iter()
                    .map(|a| self.lower_expr_to_operand(a, instrs, reg_map))
                    .collect();
                let qubits: Vec<BcOperand> = qubits
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map))
                    .collect();
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
                    self.emit_assignment(t, &value, instrs, reg_map, span);
                }
                None => {
                    let qubit = self.measure_to_operand(measure, instrs, reg_map);
                    instrs.push(BcInstr {
                        op: BcOp::Measure { dest: None, qubit },
                        span,
                    });
                }
            },
            SsaStmtKind::Reset(q) => {
                let qubit = self.lower_qubit_operand(q, instrs, reg_map);
                instrs.push(BcInstr {
                    op: BcOp::Reset { qubit },
                    span,
                });
            }
            SsaStmtKind::Barrier(qs) => {
                let qubits = qs
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map))
                    .collect();
                instrs.push(BcInstr {
                    op: BcOp::Barrier { qubits },
                    span,
                });
            }
            SsaStmtKind::Delay(Delay { duration, operands }) => {
                let duration = self.lower_expr_to_operand(duration, instrs, reg_map);
                let qubits = operands
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map))
                    .collect();
                instrs.push(BcInstr {
                    op: BcOp::Delay { duration, qubits },
                    span,
                });
            }
            SsaStmtKind::Box(SsaBoxStmt { duration, body }) => {
                let duration = duration
                    .as_ref()
                    .map(|e| self.lower_expr_to_operand(e, instrs, reg_map));
                let body_id = self.emit_root(body, ProcOwner::Box);
                instrs.push(BcInstr {
                    op: BcOp::Box {
                        duration,
                        body: body_id,
                    },
                    span,
                });
            }
            SsaStmtKind::Assignment(SsaAssignment { target, value }) => {
                self.emit_assignment(target, value, instrs, reg_map, span);
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
                    let body_id = self.emit_root(cfg, ProcOwner::InlineCal);
                    instrs.push(BcInstr {
                        op: BcOp::CalOpenPulse { body: body_id },
                        span,
                    });
                }
            },
            SsaStmtKind::ExprStmt(e) => {
                let _ = self.lower_expr_to_operand(e, instrs, reg_map);
                // Pure-side-effect statements (calls etc.) get emitted
                // inside lower_expr_to_operand. Top-level value is
                // discarded.
            }
            SsaStmtKind::Nop(qs) => {
                let qubits = qs
                    .iter()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map))
                    .collect();
                instrs.push(BcInstr {
                    op: BcOp::Nop { qubits },
                    span,
                });
            }
        }
    }

    fn emit_assignment(
        &mut self,
        target: &SsaLValue,
        value: &RValue<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) {
        match target {
            SsaLValue::Var(v) => {
                let dest = self.reg_for(*v, reg_map);
                self.emit_rvalue_to_dest(value, dest, instrs, reg_map, span);
            }
            SsaLValue::Indexed { old, new, indices } => {
                // Indexed store: read `old`, compute `new = old[index] = value`.
                let base = BcOperand::Reg(self.reg_for(*old, reg_map));
                let new_reg = self.reg_for(*new, reg_map);
                // OpenQASM supports multi-dim indexed assignment via
                // multiple index ops; the bytecode currently flattens
                // each as a separate StoreElement chained through a
                // single new register. v1: only the first dim.
                let index = if let Some(io) = indices.first() {
                    self.lower_index_op_to_operand(io, instrs, reg_map)
                } else {
                    BcOperand::Const(self.intern_const(Value::int(0, oqi_classical::iw(64))))
                };
                let value = self.rvalue_to_operand(value, instrs, reg_map);
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
    }

    fn emit_rvalue_to_dest(
        &mut self,
        rv: &RValue<SsaExpr>,
        dest: Reg,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) {
        match rv {
            RValue::Expr(e) => self.emit_expr_to_dest(e, dest, instrs, reg_map, span),
            RValue::Measure(m) => {
                let qubit = self.measure_to_operand(m, instrs, reg_map);
                instrs.push(BcInstr {
                    op: BcOp::Measure {
                        dest: Some(dest),
                        qubit,
                    },
                    span,
                });
            }
        }
    }

    fn emit_expr_to_dest(
        &mut self,
        e: &SsaExpr,
        dest: Reg,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        span: Span,
    ) {
        match &e.kind {
            SsaExprKind::Binary(Binary { op, left, right }) => {
                let lhs = self.lower_expr_to_operand(left, instrs, reg_map);
                let rhs = self.lower_expr_to_operand(right, instrs, reg_map);
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
                let src = self.lower_expr_to_operand(operand, instrs, reg_map);
                let bc = match op {
                    crate::sir::UnOp::Neg => BcOp::Neg { dest, src },
                    crate::sir::UnOp::BitNot => BcOp::BitNot { dest, src },
                    crate::sir::UnOp::LogNot => BcOp::LogNot { dest, src },
                };
                instrs.push(BcInstr { op: bc, span });
            }
            SsaExprKind::Cast(Cast { target_ty, operand }) => {
                let src = self.lower_expr_to_operand(operand, instrs, reg_map);
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
                let base = self.lower_expr_to_operand(base, instrs, reg_map);
                let index = self.lower_index_op_to_operand(index, instrs, reg_map);
                instrs.push(BcInstr {
                    op: BcOp::LoadElement { dest, base, index },
                    span,
                });
            }
            SsaExprKind::Call(Call { callee, args }) => {
                let args = args
                    .iter()
                    .map(|a| self.lower_expr_to_operand(a, instrs, reg_map))
                    .collect();
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
                    .collect();
                instrs.push(BcInstr {
                    op: BcOp::NewArray { dest, items },
                    span,
                });
            }
            SsaExprKind::DurationOf(cfg) => {
                let body_id = self.emit_root(cfg, ProcOwner::DurationOf);
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
                let src = self.lower_expr_to_operand(e, instrs, reg_map);
                instrs.push(BcInstr {
                    op: BcOp::Move { dest, src },
                    span,
                });
            }
        }
    }

    /// Lower an SsaExpr into a single BcOperand, emitting intermediate
    /// instructions for any non-trivial sub-expression.
    fn lower_expr_to_operand(
        &mut self,
        e: &SsaExpr,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> BcOperand {
        match &e.kind {
            SsaExprKind::Literal(p) => {
                let v = primitive_to_value(p.clone(), &e.ty);
                BcOperand::Const(self.intern_const(v))
            }
            SsaExprKind::Var(v) => BcOperand::Reg(self.reg_for(*v, reg_map)),
            SsaExprKind::HardwareQubit(n) => BcOperand::HardwareQubit(*n as u32),
            // Anything else: spill into a synthetic temp register.
            _ => {
                let ty =
                    e.ty.value_ty()
                        .unwrap_or(ValueTy::Scalar(PrimitiveTy::Bool));
                let temp = self.alloc_temp_reg(reg_map, ty);
                self.emit_expr_to_dest(e, temp, instrs, reg_map, e.span);
                BcOperand::Reg(temp)
            }
        }
    }

    fn lower_qubit_operand(
        &mut self,
        q: &QubitOperand<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> BcOperand {
        match q {
            QubitOperand::Indexed { symbol, indices } => {
                let index = indices
                    .first()
                    .map(|io| Box::new(self.lower_index_op_to_operand(io, instrs, reg_map)));
                BcOperand::QubitReg {
                    symbol: *symbol,
                    index,
                }
            }
            QubitOperand::Hardware(n) => BcOperand::HardwareQubit(*n as u32),
        }
    }

    fn lower_index_op_to_operand(
        &mut self,
        io: &IndexOp<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> BcOperand {
        // v1: only single-item indices. Slices/sets collapse to their
        // first element. (A richer model would lift slicing into its
        // own opcodes.)
        match &io.kind {
            IndexKind::Set(es) => {
                if let Some(e) = es.first() {
                    self.lower_expr_to_operand(e, instrs, reg_map)
                } else {
                    BcOperand::Const(self.intern_const(Value::int(0, oqi_classical::iw(64))))
                }
            }
            IndexKind::Items(items) => {
                if let Some(item) = items.first() {
                    match item {
                        IndexItem::Single(e) => self.lower_expr_to_operand(e, instrs, reg_map),
                        IndexItem::Range(r) => {
                            if let Some(start) = &r.start {
                                self.lower_expr_to_operand(start, instrs, reg_map)
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
        }
    }

    fn lower_gate_modifier(
        &mut self,
        m: &GateModifier<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> BcGateModifier {
        match m {
            GateModifier::Inv => BcGateModifier::Inv,
            GateModifier::Pow(e) => {
                BcGateModifier::Pow(self.lower_expr_to_operand(e, instrs, reg_map))
            }
            GateModifier::Ctrl(n) => BcGateModifier::Ctrl(*n as u32),
            GateModifier::NegCtrl(n) => BcGateModifier::NegCtrl(*n as u32),
        }
    }

    /// Lower the qubit being measured to an operand.
    fn measure_to_operand(
        &mut self,
        m: &MeasureExpr<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> BcOperand {
        match &m.kind {
            MeasureExprKind::Measure { operand } => {
                self.lower_qubit_operand(operand, instrs, reg_map)
            }
            MeasureExprKind::QuantumCall { qubits, .. } => {
                // QuantumCall in measure position: lower its first
                // qubit as the measure target. (Multi-qubit
                // measurement semantics aren't yet modeled in
                // bytecode.)
                qubits
                    .first()
                    .map(|q| self.lower_qubit_operand(q, instrs, reg_map))
                    .unwrap_or(BcOperand::HardwareQubit(0))
            }
        }
    }

    fn rvalue_to_operand(
        &mut self,
        rv: &RValue<SsaExpr>,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
    ) -> BcOperand {
        match rv {
            RValue::Expr(e) => self.lower_expr_to_operand(e, instrs, reg_map),
            RValue::Measure(m) => {
                let qubit = self.measure_to_operand(m, instrs, reg_map);
                let ty = m.ty.value_ty().unwrap_or(ValueTy::Scalar(PrimitiveTy::Bit));
                let temp = self.alloc_temp_reg(reg_map, ty);
                instrs.push(BcInstr {
                    op: BcOp::Measure {
                        dest: Some(temp),
                        qubit,
                    },
                    span: m.span,
                });
                BcOperand::Reg(temp)
            }
        }
    }

    fn emit_terminator(
        &mut self,
        term: &SsaTerminator,
        instrs: &mut Vec<BcInstr>,
        reg_map: &mut RegMap,
        _span: Span,
    ) -> BcTerminator {
        match term {
            SsaTerminator::Goto(b) => BcTerminator::Goto(BlockId(b.0 as u32)),
            SsaTerminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let cond = self.lower_expr_to_operand(cond, instrs, reg_map);
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
                let target = self.lower_expr_to_operand(target, instrs, reg_map);
                let cases = cases
                    .iter()
                    .map(|(labels, bb)| {
                        let lab = match labels {
                            crate::sir::SwitchLabels::Default => BcSwitchLabels::Default,
                            crate::sir::SwitchLabels::Values(vs) => BcSwitchLabels::Values(
                                vs.iter()
                                    .map(|v| self.lower_expr_to_operand(v, instrs, reg_map))
                                    .collect(),
                            ),
                        };
                        (lab, BlockId(bb.0 as u32))
                    })
                    .collect();
                BcTerminator::Switch {
                    target,
                    cases,
                    default: default.map(|b| BlockId(b.0 as u32)),
                }
            }
            SsaTerminator::Return(rv) => BcTerminator::Return(
                rv.as_ref()
                    .map(|r| self.rvalue_to_operand(r, instrs, reg_map)),
            ),
            SsaTerminator::End => BcTerminator::End,
            SsaTerminator::Unreachable => BcTerminator::Unreachable,
        }
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
