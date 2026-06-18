//! The bytecode interpreter.
//!
//! Executes a [`BcModule`] by walking each procedure's basic blocks and
//! following terminators. Classical work is delegated to `oqi-classical`
//! (`checked_op`, `Value::cast`, array indexing); quantum work to a
//! pluggable [`QuantumBackend`]; `extern` calls to an [`ExternProvider`].

use std::collections::HashMap;

use oqi_classical::ops::{BinOp, UnOp};
use oqi_classical::{
    Array, ArrayTy, DurationUnit, FloatWidth, Index, Primitive, PrimitiveTy, Scalar, Value,
    ValueTy, iw, ops,
};
use oqi_compile::bytecode::{
    BcCallTarget, BcGateModifier, BcModule, BcOp, BcOperand, BcSwitchLabels, BcTerminator, BlockId,
    ProcId, ProcOwner, Reg,
};
use oqi_compile::sir::Intrinsic;
use oqi_compile::symbol::{SymbolId, SymbolKind};

use crate::backend::{GateModifiers, QuantumBackend};
use crate::error::{Result, VmError};
use crate::extern_fns::ExternProvider;

/// The outcome of a run.
#[derive(Debug)]
pub struct RunResult {
    /// Measurement outcomes in execution order: (global qubit index, bit).
    pub measurements: Vec<(u32, bool)>,
    /// Final register file of the top-level procedure, for inspection.
    pub registers: Vec<Option<Value>>,
}

/// One procedure activation.
struct Frame {
    proc: ProcId,
    /// Register file, indexed by [`Reg`].
    regs: Vec<Option<Value>>,
    /// Bound qubit parameters, indexed by slot; each slot holds the
    /// resolved global qubit indices for that parameter.
    qubit_args: Vec<Vec<u32>>,
    /// Modifier context (controls / power) inherited by gate calls in
    /// this body.
    mods: GateModifiers,
}

/// A virtual machine over a bytecode module with a chosen quantum
/// backend and extern provider.
pub struct Vm<'m, B: QuantumBackend, E: ExternProvider> {
    module: &'m BcModule,
    backend: B,
    externs: E,
    gate_procs: HashMap<SymbolId, ProcId>,
    sub_procs: HashMap<SymbolId, ProcId>,
    measurements: Vec<(u32, bool)>,
}

impl<'m, B: QuantumBackend, E: ExternProvider> Vm<'m, B, E> {
    pub fn new(module: &'m BcModule, backend: B, externs: E) -> Self {
        let mut gate_procs = HashMap::new();
        let mut sub_procs = HashMap::new();
        for (i, proc) in module.procedures.iter().enumerate() {
            match proc.owner {
                ProcOwner::Gate(s) => {
                    gate_procs.insert(s, ProcId(i as u32));
                }
                ProcOwner::Subroutine(s) => {
                    sub_procs.insert(s, ProcId(i as u32));
                }
                _ => {}
            }
        }
        Vm {
            module,
            backend,
            externs,
            gate_procs,
            sub_procs,
            measurements: Vec::new(),
        }
    }

    /// Access the backend (e.g. to read a simulator's state vector after
    /// a run).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Execute the module's entry procedure.
    pub fn run(&mut self) -> Result<RunResult> {
        let entry = self.module.entry;
        let mut frame = Frame {
            proc: entry,
            regs: self.fresh_regs(entry),
            qubit_args: Vec::new(),
            mods: GateModifiers::none(),
        };
        self.exec_proc(&mut frame)?;
        Ok(RunResult {
            measurements: std::mem::take(&mut self.measurements),
            registers: frame.regs,
        })
    }

    fn exec_proc(&mut self, frame: &mut Frame) -> Result<Option<Value>> {
        let module = self.module;
        let proc = &module.procedures[frame.proc.0 as usize];
        let mut current = proc.entry;
        loop {
            let block = proc
                .blocks
                .iter()
                .find(|b| b.id == current)
                .ok_or_else(|| VmError::Unsupported(format!("missing block bb{}", current.0)))?;
            for instr in &block.instrs {
                self.exec_op(&instr.op, frame)?;
            }
            match &block.terminator {
                BcTerminator::Goto(b) => current = *b,
                BcTerminator::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let v = self.eval(frame, cond)?;
                    current = if value_bit(&v)? { *then_bb } else { *else_bb };
                }
                BcTerminator::Switch {
                    target,
                    cases,
                    default,
                } => {
                    let t = value_i128(&self.eval(frame, target)?)?;
                    current = self.switch_target(frame, t, cases, *default)?;
                }
                BcTerminator::Return(opt) => {
                    return Ok(match opt {
                        Some(o) => Some(self.eval(frame, o)?),
                        None => None,
                    });
                }
                BcTerminator::End => return Ok(None),
                BcTerminator::Unreachable => return Err(VmError::Unreachable),
            }
        }
    }

    fn switch_target(
        &self,
        frame: &Frame,
        target: i128,
        cases: &[(BcSwitchLabels, BlockId)],
        default: Option<BlockId>,
    ) -> Result<BlockId> {
        for (labels, bb) in cases {
            if let BcSwitchLabels::Values(vs) = labels {
                for v in vs {
                    if value_i128(&self.eval(frame, v)?)? == target {
                        return Ok(*bb);
                    }
                }
            }
        }
        default
            .ok_or_else(|| VmError::Unsupported("switch with no matching case or default".into()))
    }

    fn exec_op(&mut self, op: &BcOp, frame: &mut Frame) -> Result<()> {
        match op {
            // ── Binary classical ops ──────────────────────────────────
            BcOp::Add { dest, lhs, rhs } => self.bin::<ops::Add>(frame, *dest, lhs, rhs),
            BcOp::Sub { dest, lhs, rhs } => self.bin::<ops::Sub>(frame, *dest, lhs, rhs),
            BcOp::Mul { dest, lhs, rhs } => self.bin::<ops::Mul>(frame, *dest, lhs, rhs),
            BcOp::Div { dest, lhs, rhs } => self.bin::<ops::Div>(frame, *dest, lhs, rhs),
            BcOp::Mod { dest, lhs, rhs } => self.bin::<ops::Rem>(frame, *dest, lhs, rhs),
            BcOp::Pow { dest, lhs, rhs } => self.bin::<ops::Pow>(frame, *dest, lhs, rhs),
            BcOp::BitAnd { dest, lhs, rhs } => self.bin::<ops::BitAnd>(frame, *dest, lhs, rhs),
            BcOp::BitOr { dest, lhs, rhs } => self.bin::<ops::BitOr>(frame, *dest, lhs, rhs),
            BcOp::BitXor { dest, lhs, rhs } => self.bin::<ops::BitXor>(frame, *dest, lhs, rhs),
            BcOp::Shl { dest, lhs, rhs } => self.bin::<ops::Shl>(frame, *dest, lhs, rhs),
            BcOp::Shr { dest, lhs, rhs } => self.bin::<ops::Shr>(frame, *dest, lhs, rhs),
            BcOp::LogAnd { dest, lhs, rhs } => self.bin::<ops::LogAnd>(frame, *dest, lhs, rhs),
            BcOp::LogOr { dest, lhs, rhs } => self.bin::<ops::LogOr>(frame, *dest, lhs, rhs),
            BcOp::Eq { dest, lhs, rhs } => self.bin::<ops::Eq>(frame, *dest, lhs, rhs),
            BcOp::Neq { dest, lhs, rhs } => self.bin::<ops::Neq>(frame, *dest, lhs, rhs),
            BcOp::Lt { dest, lhs, rhs } => self.bin::<ops::Lt>(frame, *dest, lhs, rhs),
            BcOp::Gt { dest, lhs, rhs } => self.bin::<ops::Gt>(frame, *dest, lhs, rhs),
            BcOp::Le { dest, lhs, rhs } => self.bin::<ops::Lte>(frame, *dest, lhs, rhs),
            BcOp::Ge { dest, lhs, rhs } => self.bin::<ops::Gte>(frame, *dest, lhs, rhs),

            // ── Unary classical ops ───────────────────────────────────
            BcOp::Neg { dest, src } => self.un::<ops::Neg>(frame, *dest, src),
            BcOp::BitNot { dest, src } => self.un::<ops::BitNot>(frame, *dest, src),
            BcOp::LogNot { dest, src } => self.un::<ops::LogNot>(frame, *dest, src),
            BcOp::Cast {
                dest,
                target_ty,
                src,
            } => {
                let v = self.eval(frame, src)?.cast(*target_ty)?;
                self.set(frame, *dest, v);
                Ok(())
            }

            // ── Moves & memory ────────────────────────────────────────
            BcOp::Move { dest, src } => {
                let v = self.eval(frame, src)?;
                self.set(frame, *dest, v);
                Ok(())
            }
            BcOp::LoadElement { dest, base, index } => {
                let base = self.eval(frame, base)?;
                let i = value_isize(&self.eval(frame, index)?)?;
                let v = base.get(&[Index::Item(i)])?;
                self.set(frame, *dest, v);
                Ok(())
            }
            BcOp::StoreElement {
                new,
                base,
                index,
                value,
            } => {
                let mut arr = self.eval(frame, base)?;
                let i = value_isize(&self.eval(frame, index)?)?;
                let value = self.eval(frame, value)?;
                arr.set(&[Index::Item(i)], value)?;
                self.set(frame, *new, arr);
                Ok(())
            }
            BcOp::NewArray { dest, items } => {
                let aty = self.array_ty(frame, *dest)?;
                let prims: Vec<Primitive> = items
                    .iter()
                    .map(|op| match self.eval(frame, op)? {
                        Value::Scalar(s) => Ok(s.into_value()),
                        _ => Err(VmError::Type("array element must be scalar".into())),
                    })
                    .collect::<Result<_>>()?;
                let arr = Array::new(prims, aty)?;
                self.set(frame, *dest, Value::Array(arr));
                Ok(())
            }

            // ── Call ──────────────────────────────────────────────────
            BcOp::Call { dest, callee, args } => self.exec_call(frame, *dest, callee, args),

            // ── Quantum ops ───────────────────────────────────────────
            BcOp::GateCall {
                gate,
                modifiers,
                args,
                qubits,
            } => self.exec_gate_call(*gate, modifiers, args, qubits, frame),
            BcOp::Measure { dest, qubit } => {
                let qs = self.qubits(frame, qubit)?;
                let mut bits: u128 = 0;
                for (i, q) in qs.iter().enumerate() {
                    let b = self.backend.measure(*q);
                    self.measurements.push((*q, b));
                    if b {
                        bits |= 1 << i;
                    }
                }
                if let Some(d) = dest {
                    let v = if qs.len() == 1 {
                        Value::bit(bits & 1 != 0)
                    } else {
                        Value::bitreg_u128(bits, qs.len() as u32)
                    };
                    self.set(frame, *d, v);
                }
                Ok(())
            }
            BcOp::Reset { qubit } => {
                for q in self.qubits(frame, qubit)? {
                    self.backend.reset(q);
                }
                Ok(())
            }
            BcOp::Barrier { qubits } => {
                let qs = self.qubit_list(frame, qubits)?;
                self.backend.barrier(&qs);
                Ok(())
            }
            BcOp::Delay { duration, qubits } => {
                let d = self.eval(frame, duration)?;
                let dur = match d {
                    Value::Scalar(s) => s.value().as_duration(),
                    _ => None,
                }
                .ok_or_else(|| VmError::Type("delay duration must be a duration".into()))?;
                let qs = self.qubit_list(frame, qubits)?;
                self.backend.delay(&qs, dur);
                Ok(())
            }
            BcOp::Nop { .. } => Ok(()),

            // ── Structured / misc (MVP: timing & pulse are no-ops) ────
            BcOp::Box { body, .. } => {
                let mut child = Frame {
                    proc: *body,
                    regs: self.fresh_regs(*body),
                    qubit_args: Vec::new(),
                    mods: GateModifiers::none(),
                };
                self.exec_proc(&mut child)?;
                Ok(())
            }
            BcOp::Pragma { .. } | BcOp::Alias { .. } | BcOp::CalOpaque { .. } => Ok(()),
            BcOp::CalOpenPulse { .. } => Ok(()),
            BcOp::DurationOf { .. } => {
                Err(VmError::Unsupported("durationof timing analysis".into()))
            }
        }
    }

    // ── Calls ─────────────────────────────────────────────────────────

    fn exec_call(
        &mut self,
        frame: &mut Frame,
        dest: Option<Reg>,
        callee: &BcCallTarget,
        args: &[BcOperand],
    ) -> Result<()> {
        match callee {
            BcCallTarget::Intrinsic(i) => {
                let vals: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval(frame, a))
                    .collect::<Result<_>>()?;
                let v = eval_intrinsic(i, &vals)?;
                if let Some(d) = dest {
                    self.set(frame, d, v);
                }
                Ok(())
            }
            BcCallTarget::Symbol(s) => {
                let sym = self.module.symbols.get(*s);
                if matches!(
                    sym.kind,
                    SymbolKind::Extern | SymbolKind::ExternPort | SymbolKind::ExternFrame
                ) {
                    let name = sym.name.clone();
                    let vals: Vec<Value> = args
                        .iter()
                        .map(|a| self.eval(frame, a))
                        .collect::<Result<_>>()?;
                    let ret = self.externs.call(&name, &vals)?;
                    if let Some(d) = dest {
                        let v = ret.ok_or_else(|| {
                            VmError::Type(format!("extern `{name}` returned no value"))
                        })?;
                        self.set(frame, d, v);
                    }
                    Ok(())
                } else {
                    let proc_id = *self.sub_procs.get(s).ok_or_else(|| {
                        VmError::Unsupported(format!("call to non-subroutine symbol s{}", s.0))
                    })?;
                    let mut child = self.bind_subroutine(frame, proc_id, args)?;
                    let ret = self.exec_proc(&mut child)?;
                    if let Some(d) = dest {
                        let v = ret.ok_or_else(|| {
                            VmError::Type("subroutine returned no value for a value call".into())
                        })?;
                        self.set(frame, d, v);
                    }
                    Ok(())
                }
            }
        }
    }

    /// Build a callee frame for a subroutine, binding positional args.
    /// Classical args go to the procedure's parameter registers (in
    /// declaration order); qubit args bind to their positional slot.
    fn bind_subroutine(&self, frame: &Frame, proc_id: ProcId, args: &[BcOperand]) -> Result<Frame> {
        let proc = &self.module.procedures[proc_id.0 as usize];
        let mut regs = self.fresh_regs(proc_id);
        let mut qubit_args: Vec<Vec<u32>> = vec![Vec::new(); args.len()];
        let mut classical = 0usize;
        for (slot, arg) in args.iter().enumerate() {
            if is_qubit_operand(arg) {
                qubit_args[slot] = self.qubits(frame, arg)?;
            } else {
                let v = self.eval(frame, arg)?;
                let reg = proc.params.get(classical).ok_or_else(|| {
                    VmError::Unsupported("more classical args than parameters".into())
                })?;
                regs[reg.0 as usize] = Some(v);
                classical += 1;
            }
        }
        Ok(Frame {
            proc: proc_id,
            regs,
            qubit_args,
            mods: GateModifiers::none(),
        })
    }

    // ── Gates ─────────────────────────────────────────────────────────

    fn exec_gate_call(
        &mut self,
        gate: SymbolId,
        modifiers: &[BcGateModifier],
        args: &[BcOperand],
        qubits: &[BcOperand],
        frame: &mut Frame,
    ) -> Result<()> {
        // Fold modifiers (inherited context + this call's) into one
        // effective set, consuming leading qubit operands as controls.
        let mut power = frame.mods.power;
        let mut ctrl_kinds: Vec<bool> = Vec::new(); // true = negctrl
        for m in modifiers {
            match m {
                BcGateModifier::Inv => power = -power,
                BcGateModifier::Pow(e) => power *= value_f64(&self.eval(frame, e)?)?,
                BcGateModifier::Ctrl(k) => {
                    ctrl_kinds.extend(std::iter::repeat_n(false, *k as usize))
                }
                BcGateModifier::NegCtrl(k) => {
                    ctrl_kinds.extend(std::iter::repeat_n(true, *k as usize))
                }
            }
        }
        let n_ctrl = ctrl_kinds.len();
        let mut controls = frame.mods.controls.clone();
        let mut neg_controls = frame.mods.neg_controls.clone();
        for (i, neg) in ctrl_kinds.iter().enumerate() {
            for q in self.qubits(frame, &qubits[i])? {
                if *neg {
                    neg_controls.push(q);
                } else {
                    controls.push(q);
                }
            }
        }
        let eff = GateModifiers {
            controls,
            neg_controls,
            power,
        };
        let arg_ops = &qubits[n_ctrl..];

        if let Some(&proc_id) = self.gate_procs.get(&gate) {
            // User/std gate: bind params + qubits, recurse, broadcasting
            // over register operands.
            let arg_vals: Vec<Value> = args
                .iter()
                .map(|a| self.eval(frame, a))
                .collect::<Result<_>>()?;
            let resolved: Vec<Vec<u32>> = arg_ops
                .iter()
                .map(|o| self.qubits(frame, o))
                .collect::<Result<_>>()?;
            let bcast = broadcast_len(&resolved)?;
            for j in 0..bcast {
                let mut regs = self.fresh_regs(proc_id);
                let proc = &self.module.procedures[proc_id.0 as usize];
                for (k, v) in arg_vals.iter().enumerate() {
                    let reg = proc.params.get(k).ok_or_else(|| {
                        VmError::Unsupported("gate has fewer params than arguments".into())
                    })?;
                    regs[reg.0 as usize] = Some(v.clone());
                }
                let qubit_args: Vec<Vec<u32>> = resolved
                    .iter()
                    .map(|qs| {
                        // Singletons repeat; registers (validated equal to
                        // `bcast`) are indexed by the broadcast position.
                        let idx = if qs.len() == 1 { 0 } else { j };
                        vec![qs[idx]]
                    })
                    .collect();
                let mut child = Frame {
                    proc: proc_id,
                    regs,
                    qubit_args,
                    mods: eff.clone(),
                };
                self.exec_proc(&mut child)?;
            }
            Ok(())
        } else {
            let name = self.module.symbols.get(gate).name.as_str();
            match name {
                "U" => {
                    let theta = value_f64(&self.eval(frame, &args[0])?)?;
                    let phi = value_f64(&self.eval(frame, &args[1])?)?;
                    let lambda = value_f64(&self.eval(frame, &args[2])?)?;
                    for op in arg_ops {
                        for q in self.qubits(frame, op)? {
                            self.backend.u(q, theta, phi, lambda, &eff);
                        }
                    }
                    Ok(())
                }
                "gphase" => {
                    let gamma = value_f64(&self.eval(frame, &args[0])?)?;
                    self.backend.gphase(gamma, &eff);
                    Ok(())
                }
                other => Err(VmError::UndefinedGate(other.to_string())),
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────

    fn bin<O: BinOp>(
        &mut self,
        frame: &mut Frame,
        dest: Reg,
        lhs: &BcOperand,
        rhs: &BcOperand,
    ) -> Result<()> {
        let l = self.eval(frame, lhs)?;
        let r = self.eval(frame, rhs)?;
        let v = O::checked_op(l, r)?;
        self.set(frame, dest, v);
        Ok(())
    }

    fn un<O: UnOp>(&mut self, frame: &mut Frame, dest: Reg, src: &BcOperand) -> Result<()> {
        let v = self.eval(frame, src)?;
        let v = O::checked_op(v)?;
        self.set(frame, dest, v);
        Ok(())
    }

    fn set(&self, frame: &mut Frame, reg: Reg, value: Value) {
        frame.regs[reg.0 as usize] = Some(value);
    }

    /// Resolve a classical operand to a value.
    fn eval(&self, frame: &Frame, op: &BcOperand) -> Result<Value> {
        match op {
            BcOperand::Reg(r) => frame.regs[r.0 as usize]
                .clone()
                .ok_or(VmError::UnsetRegister(r.0)),
            BcOperand::Const(c) => Ok(self.module.constants[c.0 as usize].clone()),
            _ => Err(VmError::Type("expected a classical operand".into())),
        }
    }

    fn array_ty(&self, frame: &Frame, dest: Reg) -> Result<ArrayTy> {
        match self.module.procedures[frame.proc.0 as usize].register_types[dest.0 as usize] {
            ValueTy::Array(aty) => Ok(aty),
            _ => Err(VmError::Type("NewArray destination is not an array".into())),
        }
    }

    /// Resolve a qubit operand to its global qubit indices.
    fn qubits(&self, frame: &Frame, op: &BcOperand) -> Result<Vec<u32>> {
        match op {
            BcOperand::Qubit(i) => Ok(vec![*i]),
            BcOperand::HardwareQubit(i) => Ok(vec![*i]),
            BcOperand::QubitRegion(id) => Ok(self.region_indices(id.0)),
            BcOperand::QubitIndexed { region, index } => {
                let idx = value_usize(&self.eval(frame, index)?)?;
                self.region_indices(region.0)
                    .get(idx)
                    .copied()
                    .map(|q| vec![q])
                    .ok_or_else(|| VmError::Type(format!("qubit index {idx} out of range")))
            }
            BcOperand::QubitParam { slot, index } => {
                let bound = frame
                    .qubit_args
                    .get(*slot as usize)
                    .ok_or_else(|| VmError::Unsupported("unbound qubit parameter".into()))?;
                match index {
                    None => Ok(bound.clone()),
                    Some(ix) => {
                        let i = value_usize(&self.eval(frame, ix)?)?;
                        bound
                            .get(i)
                            .copied()
                            .map(|q| vec![q])
                            .ok_or_else(|| VmError::Type(format!("qubit index {i} out of range")))
                    }
                }
            }
            _ => Err(VmError::Type("expected a qubit operand".into())),
        }
    }

    /// Resolve and flatten a list of qubit operands.
    fn qubit_list(&self, frame: &Frame, ops: &[BcOperand]) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for op in ops {
            out.extend(self.qubits(frame, op)?);
        }
        Ok(out)
    }

    fn region_indices(&self, region: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for (s, e) in &self.module.qubits.regions[region as usize].ranges {
            out.extend(*s..*e);
        }
        out
    }

    /// A fresh register file for `proc`, with every register set to its
    /// type's zero. OpenQASM default-initializes classical variables to
    /// zero, and this is also the SSA "entry value" of any variable;
    /// callers then override parameter registers with bound arguments.
    fn fresh_regs(&self, proc: ProcId) -> Vec<Option<Value>> {
        self.module.procedures[proc.0 as usize]
            .register_types
            .iter()
            .map(|ty| zero_value(*ty))
            .collect()
    }
}

// ── Free helpers ──────────────────────────────────────────────────────

/// The zero value of a register type (OpenQASM default initialization).
/// Returns `None` for types with no natural zero we materialize (array
/// references); such registers are bound explicitly before use.
fn zero_value(ty: ValueTy) -> Option<Value> {
    match ty {
        ValueTy::Scalar(pty) => Some(Value::Scalar(Scalar::new_unchecked(
            zero_primitive(pty),
            pty,
        ))),
        ValueTy::Array(aty) => {
            let n: usize = aty.shape().get().iter().product();
            let prims = vec![zero_primitive(aty.ty()); n];
            Some(Value::Array(Array::new_unchecked(prims, aty)))
        }
        ValueTy::ArrayRef(_) => None,
    }
}

fn zero_primitive(ty: PrimitiveTy) -> Primitive {
    match ty {
        PrimitiveTy::Bit | PrimitiveTy::Bool => Primitive::bit(false),
        PrimitiveTy::Int(_) => Primitive::int(0),
        PrimitiveTy::Uint(_) => Primitive::uint(0),
        PrimitiveTy::Float(_) => Primitive::float(0.0),
        PrimitiveTy::Complex(_) => Primitive::complex(0.0, 0.0),
        PrimitiveTy::Angle(_) => Primitive::angle(0.0),
        PrimitiveTy::Duration => Primitive::duration(0.0, DurationUnit::Ns),
        PrimitiveTy::BitReg(_) => Primitive::bitreg_u128(0),
    }
}

/// The broadcast length for a gate's resolved qubit operands. All
/// register (non-singleton) operands must share one length; single
/// qubits are repeated. Returns 1 when every operand is a single qubit.
fn broadcast_len(operands: &[Vec<u32>]) -> Result<usize> {
    let mut n = 1;
    for qs in operands {
        if qs.len() == 1 {
            continue;
        }
        if n == 1 {
            n = qs.len();
        } else if n != qs.len() {
            return Err(VmError::BroadcastMismatch(
                operands.iter().map(|q| q.len()).collect(),
            ));
        }
    }
    Ok(n)
}

fn is_qubit_operand(op: &BcOperand) -> bool {
    matches!(
        op,
        BcOperand::Qubit(_)
            | BcOperand::HardwareQubit(_)
            | BcOperand::QubitRegion(_)
            | BcOperand::QubitIndexed { .. }
            | BcOperand::QubitParam { .. }
    )
}

fn scalar(v: &Value) -> Result<&Primitive> {
    match v {
        Value::Scalar(s) => Ok(s.value()),
        _ => Err(VmError::Type("expected a scalar value".into())),
    }
}

fn value_bit(v: &Value) -> Result<bool> {
    Ok(scalar(v)?.as_bit())
}

fn value_f64(v: &Value) -> Result<f64> {
    scalar(v)?
        .as_float(FloatWidth::F64)
        .ok_or_else(|| VmError::Type("expected a float-like value".into()))
}

fn value_i128(v: &Value) -> Result<i128> {
    scalar(v)?
        .as_int(iw(128))
        .ok_or_else(|| VmError::Type("expected an integer value".into()))
}

fn value_usize(v: &Value) -> Result<usize> {
    scalar(v)?
        .as_uint(iw(64))
        .map(|u| u as usize)
        .ok_or_else(|| VmError::Type("expected an unsigned integer value".into()))
}

fn value_isize(v: &Value) -> Result<isize> {
    Ok(value_i128(v)? as isize)
}

/// Evaluate an intrinsic call against the classical op library.
fn eval_intrinsic(i: &Intrinsic, args: &[Value]) -> Result<Value> {
    let arg0 = || -> Result<Value> {
        args.first()
            .cloned()
            .ok_or_else(|| VmError::Type("intrinsic missing argument".into()))
    };
    let arg = |n: usize| -> Result<Value> {
        args.get(n)
            .cloned()
            .ok_or_else(|| VmError::Type("intrinsic missing argument".into()))
    };
    Ok(match i {
        Intrinsic::Sin => ops::Sin::checked_op(arg0()?)?,
        Intrinsic::Cos => ops::Cos::checked_op(arg0()?)?,
        Intrinsic::Tan => ops::Tan::checked_op(arg0()?)?,
        Intrinsic::Arcsin => ops::Arcsin::checked_op(arg0()?)?,
        Intrinsic::Arccos => ops::Arccos::checked_op(arg0()?)?,
        Intrinsic::Arctan => ops::Arctan::checked_op(arg0()?)?,
        Intrinsic::Exp => ops::Exp::checked_op(arg0()?)?,
        Intrinsic::Log => ops::Log::checked_op(arg0()?)?,
        Intrinsic::Sqrt => ops::Sqrt::checked_op(arg0()?)?,
        Intrinsic::Ceiling => ops::Ceiling::checked_op(arg0()?)?,
        Intrinsic::Floor => ops::Floor::checked_op(arg0()?)?,
        Intrinsic::Popcount => ops::Popcount::checked_op(arg0()?)?,
        Intrinsic::Real => ops::Real::checked_op(arg0()?)?,
        Intrinsic::Imag => ops::Imag::checked_op(arg0()?)?,
        Intrinsic::Mod => ops::Rem::checked_op(arg0()?, arg(1)?)?,
        Intrinsic::Rotl => ops::Rotl::checked_op(arg0()?, arg(1)?)?,
        Intrinsic::Rotr => ops::Rotr::checked_op(arg0()?, arg(1)?)?,
        Intrinsic::Sizeof => {
            if args.len() >= 2 {
                ops::SizeofDim::checked_op(arg0()?, arg(1)?)?
            } else {
                ops::Sizeof::checked_op(arg0()?)?
            }
        }
    })
}
