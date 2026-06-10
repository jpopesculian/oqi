//! Dense register allocation: assign every reachable [`SsaValue`] a
//! `Reg(0)`..`Reg(n)` slot.
//!
//! The strategy is intentionally simple: walk the CFG in block order,
//! collect every distinct `SsaValue` that's either defined or read,
//! and assign them ascending register IDs. Each register's type comes
//! from the symbol table (qubits never end up here — they're
//! addressed as `BcOperand::QubitReg` / `HardwareQubit`).

use std::collections::HashMap;

use oqi_classical::ValueTy;

use crate::ssa::{
    SsaAssignment, SsaCfg, SsaExpr, SsaExprKind, SsaLValue, SsaMeasure, SsaStmtKind, SsaTerminator,
    SsaValue,
};
use crate::sir::{
    Alias, Binary, Call, Cast, Delay, GateCall, GateModifier, Index, IndexItem, IndexKind,
    IndexOp, MeasureExpr, MeasureExprKind, QubitOperand, RValue, RangeExpr, SwitchLabels, Unary,
};
use crate::symbol::SymbolTable;

use super::types::Reg;

pub struct RegMap {
    pub by_ssa: HashMap<SsaValue, Reg>,
    pub types: Vec<ValueTy>,
}

impl RegMap {
    fn empty() -> Self {
        Self {
            by_ssa: HashMap::new(),
            types: Vec::new(),
        }
    }

    pub(crate) fn alloc(&mut self, v: SsaValue, ty: ValueTy) -> Reg {
        *self.by_ssa.entry(v).or_insert_with(|| {
            let r = Reg(self.types.len() as u32);
            self.types.push(ty);
            r
        })
    }
}

/// Dense-number every classical `SsaValue` referenced by the CFG.
/// Qubits, gate references, etc. are not registers and are skipped.
pub fn allocate_registers(cfg: &SsaCfg, symbols: &SymbolTable) -> RegMap {
    let mut map = RegMap::empty();
    for block in &cfg.blocks {
        for phi in &block.phis {
            try_alloc(&mut map, symbols, phi.dest);
            for (_, src) in &phi.sources {
                try_alloc(&mut map, symbols, *src);
            }
        }
        for stmt in &block.stmts {
            visit_stmt(&stmt.kind, &mut map, symbols);
        }
        visit_terminator(&block.terminator, &mut map, symbols);
    }
    map
}

fn try_alloc(map: &mut RegMap, symbols: &SymbolTable, v: SsaValue) {
    if let Some(ty) = symbols.get(v.symbol).ty.value_ty() {
        map.alloc(v, ty);
    }
    // Symbols without a `ValueTy` (qubits etc.) don't go into registers.
}

fn visit_stmt(kind: &SsaStmtKind, map: &mut RegMap, symbols: &SymbolTable) {
    match kind {
        SsaStmtKind::Alias(Alias { value, .. }) => {
            for e in value {
                visit_expr(e, map, symbols);
            }
        }
        SsaStmtKind::GateCall(GateCall {
            modifiers,
            args,
            qubits,
            ..
        }) => {
            for m in modifiers {
                visit_gate_modifier(m, map, symbols);
            }
            for a in args {
                visit_expr(a, map, symbols);
            }
            for q in qubits {
                visit_qubit_operand(q, map, symbols);
            }
        }
        SsaStmtKind::Measure(SsaMeasure { measure, target }) => {
            visit_measure_expr(measure, map, symbols);
            if let Some(t) = target {
                visit_lvalue(t, map, symbols);
            }
        }
        SsaStmtKind::Reset(q) => visit_qubit_operand(q, map, symbols),
        SsaStmtKind::Barrier(qs) | SsaStmtKind::Nop(qs) => {
            for q in qs {
                visit_qubit_operand(q, map, symbols);
            }
        }
        SsaStmtKind::Delay(Delay { duration, operands }) => {
            visit_expr(duration, map, symbols);
            for q in operands {
                visit_qubit_operand(q, map, symbols);
            }
        }
        SsaStmtKind::Box(b) => {
            if let Some(d) = &b.duration {
                visit_expr(d, map, symbols);
            }
            // Inner cfg is its own procedure with its own register
            // numbering. Skip.
        }
        SsaStmtKind::Assignment(SsaAssignment { target, value }) => {
            visit_lvalue(target, map, symbols);
            visit_rvalue(value, map, symbols);
        }
        SsaStmtKind::Pragma(_) | SsaStmtKind::Cal(_) => {}
        SsaStmtKind::ExprStmt(e) => visit_expr(e, map, symbols),
    }
}

fn visit_terminator(term: &SsaTerminator, map: &mut RegMap, symbols: &SymbolTable) {
    match term {
        SsaTerminator::Goto(_) | SsaTerminator::End | SsaTerminator::Unreachable => {}
        SsaTerminator::Branch { cond, .. } => visit_expr(cond, map, symbols),
        SsaTerminator::Switch { target, cases, .. } => {
            visit_expr(target, map, symbols);
            for (labels, _) in cases {
                if let SwitchLabels::Values(vs) = labels {
                    for v in vs {
                        visit_expr(v, map, symbols);
                    }
                }
            }
        }
        SsaTerminator::Return(rv) => {
            if let Some(rv) = rv {
                visit_rvalue(rv, map, symbols);
            }
        }
    }
}

fn visit_expr(e: &SsaExpr, map: &mut RegMap, symbols: &SymbolTable) {
    match &e.kind {
        SsaExprKind::Literal(_) | SsaExprKind::HardwareQubit(_) => {}
        SsaExprKind::Var(v) => try_alloc(map, symbols, *v),
        SsaExprKind::Binary(Binary { left, right, .. }) => {
            visit_expr(left, map, symbols);
            visit_expr(right, map, symbols);
        }
        SsaExprKind::Unary(Unary { operand, .. }) => visit_expr(operand, map, symbols),
        SsaExprKind::Cast(Cast { operand, .. }) => visit_expr(operand, map, symbols),
        SsaExprKind::Index(Index { base, index }) => {
            visit_expr(base, map, symbols);
            visit_index_op(index, map, symbols);
        }
        SsaExprKind::Call(Call { args, .. }) => {
            for a in args {
                visit_expr(a, map, symbols);
            }
        }
        SsaExprKind::DurationOf(_) => {} // inner cfg, separate procedure
        SsaExprKind::ArrayLiteral(arr) => {
            for x in &arr.items {
                visit_expr(x, map, symbols);
            }
        }
    }
}

fn visit_qubit_operand(q: &QubitOperand<SsaExpr>, map: &mut RegMap, symbols: &SymbolTable) {
    match q {
        QubitOperand::Indexed { indices, .. } => {
            for io in indices {
                visit_index_op(io, map, symbols);
            }
        }
        QubitOperand::Hardware(_) => {}
    }
}

fn visit_lvalue(lv: &SsaLValue, map: &mut RegMap, symbols: &SymbolTable) {
    match lv {
        SsaLValue::Var(v) => try_alloc(map, symbols, *v),
        SsaLValue::Indexed {
            old,
            new,
            indices,
        } => {
            try_alloc(map, symbols, *old);
            try_alloc(map, symbols, *new);
            for io in indices {
                visit_index_op(io, map, symbols);
            }
        }
    }
}

fn visit_rvalue(rv: &RValue<SsaExpr>, map: &mut RegMap, symbols: &SymbolTable) {
    match rv {
        RValue::Expr(e) => visit_expr(e, map, symbols),
        RValue::Measure(m) => visit_measure_expr(m, map, symbols),
    }
}

fn visit_measure_expr(m: &MeasureExpr<SsaExpr>, map: &mut RegMap, symbols: &SymbolTable) {
    match &m.kind {
        MeasureExprKind::Measure { operand } => visit_qubit_operand(operand, map, symbols),
        MeasureExprKind::QuantumCall { args, qubits, .. } => {
            for a in args {
                visit_expr(a, map, symbols);
            }
            for q in qubits {
                visit_qubit_operand(q, map, symbols);
            }
        }
    }
}

fn visit_index_op(io: &IndexOp<SsaExpr>, map: &mut RegMap, symbols: &SymbolTable) {
    match &io.kind {
        IndexKind::Set(es) => {
            for e in es {
                visit_expr(e, map, symbols);
            }
        }
        IndexKind::Items(items) => {
            for item in items {
                match item {
                    IndexItem::Single(e) => visit_expr(e, map, symbols),
                    IndexItem::Range(RangeExpr { start, step, end }) => {
                        for e in [start, step, end].into_iter().flatten() {
                            visit_expr(e, map, symbols);
                        }
                    }
                }
            }
        }
    }
}

fn visit_gate_modifier(m: &GateModifier<SsaExpr>, map: &mut RegMap, symbols: &SymbolTable) {
    if let GateModifier::Pow(e) = m {
        visit_expr(e, map, symbols);
    }
}
