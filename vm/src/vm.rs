//! The bytecode interpreter.
//!
//! Executes a [`BcModule`] by walking each procedure's basic blocks and
//! following terminators. Classical work is delegated to `oqi-classical`
//! (`checked_op`, `Value::cast`, array indexing); quantum work to a
//! pluggable [`QuantumBackend`]; `extern` calls to an [`ExternProvider`].

use std::collections::HashMap;

use num_complex::Complex64;
use oqi_classical::ops::{BinOp, UnOp};
use oqi_classical::{
    Array, ArrayTy, Duration, DurationUnit, FloatWidth, Index, Primitive, PrimitiveTy, Scalar,
    Value, ValueTy, iw, ops,
};
use oqi_compile::bytecode::{
    BcAliasSegment, BcCalArg, BcCalBody, BcCalOperand, BcCalTarget, BcCallTarget, BcGateModifier,
    BcModule, BcOp, BcOperand, BcSwitchLabels, BcTerminator, BlockId, ProcId, ProcOwner,
    QubitSource, Reg,
};
use oqi_compile::sir::Intrinsic;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_lex::Span;

use crate::backend::{GateModifiers, QuantumBackend};
use crate::cal::{FrameHandle, OpaqueCalHandler, OpenPulseHandler, PortHandle, WaveformHandle};
use crate::error::{Result, VmError, VmErrorKind};
use crate::extern_fns::ExternProvider;

/// The outcome of a run.
#[derive(Debug)]
pub struct RunResult {
    /// Measurement outcomes in execution order: (global qubit index, bit).
    pub measurements: Vec<(u32, bool)>,
    /// Final register file of the top-level procedure, for inspection.
    pub registers: Vec<Option<Value>>,
    /// Named program outputs and their final values. Each `SymbolId`
    /// resolves to a name via [`BcModule::symbols`]. See
    /// [`BcModule::outputs`] for the selection rule.
    pub outputs: Vec<(SymbolId, Value)>,
}

/// One procedure activation.
struct Frame {
    proc: ProcId,
    /// Register file, indexed by [`Reg`].
    regs: Vec<Option<Value>>,
    /// Frame-local qubit slots, indexed by slot id (size
    /// `BcProcedure::num_qubit_slots`). Slots `[0, n_qubit_params)` hold
    /// the bound qubit parameters; the rest hold runtime aliases,
    /// populated by [`BcOp::AliasBind`](oqi_compile::bytecode::BcOp) as
    /// the body runs. Each slot holds resolved global qubit indices.
    slots: Vec<Vec<u32>>,
    /// Modifier context (controls / power) inherited by gate calls in
    /// this body.
    mods: GateModifiers,
}

/// A leaf primitive recorded while flattening a gate body. Used to apply
/// `inv`/`pow` modifiers correctly to composite gates (see
/// [`Vm::exec_gate_call`]): the body is executed once with recording on,
/// producing a straight-line trace of these, which is then reversed and
/// inverted (for `inv`) or repeated (for integer `pow`) before emission.
enum Leaf {
    U {
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        controls: Vec<u32>,
        neg_controls: Vec<u32>,
        power: f64,
    },
    Gphase {
        gamma: f64,
        controls: Vec<u32>,
        neg_controls: Vec<u32>,
        power: f64,
    },
}

impl Leaf {
    /// Whether this leaf acts non-trivially on a qubit (a `U`, or a
    /// *controlled* global phase, which is a relative phase). Uncontrolled
    /// `gphase` is a global scalar and so is not "real" here. Used to
    /// decide whether a fractional power can be folded per-leaf exactly.
    fn is_real(&self) -> bool {
        match self {
            Leaf::U { .. } => true,
            Leaf::Gphase {
                controls,
                neg_controls,
                ..
            } => !controls.is_empty() || !neg_controls.is_empty(),
        }
    }
}

/// State of an active `durationof` timing pass: quantum side effects
/// are suppressed and elapsed time accumulates here instead (see the
/// `BcOp::DurationOf` arm of `exec_op`). Frames and the qubit timeline
/// are modeled as independent parallel timelines; the pass's result is
/// their maximum.
struct TimingState {
    /// The qubit timeline, advanced by `delay[d]` and `box[d]`.
    base: Duration,
    /// Per-frame clocks (keyed by frame handle), advanced by
    /// `play`/`capture` using handler-reported durations.
    frames: HashMap<u64, Duration>,
}

impl TimingState {
    fn new() -> Self {
        TimingState {
            base: Duration::new(0.0, DurationUnit::Ns),
            frames: HashMap::new(),
        }
    }

    /// Advance `clock` by `d`. A zero clock adopts `d` wholesale so
    /// results keep the program's own units (duration addition
    /// converts into the left operand's unit).
    fn advance(clock: &mut Duration, d: Duration) {
        *clock = if clock.value == 0.0 { d } else { *clock + d };
    }

    fn advance_base(&mut self, d: Duration) {
        Self::advance(&mut self.base, d);
    }

    fn advance_frame(&mut self, frame: u64, d: Duration) {
        let clock = self
            .frames
            .entry(frame)
            .or_insert(Duration::new(0.0, DurationUnit::Ns));
        Self::advance(clock, d);
    }

    /// The pass's elapsed time: the maximum across all timelines.
    fn elapsed(&self) -> Duration {
        let mut max = self.base;
        for d in self.frames.values() {
            if *d > max {
                max = *d;
            }
        }
        max
    }

    /// Synchronize every timeline to the current maximum (a barrier).
    fn sync(&mut self) {
        let e = self.elapsed();
        self.base = e;
        for d in self.frames.values_mut() {
            *d = e;
        }
    }
}

/// A virtual machine over a bytecode module with a chosen quantum
/// backend and extern provider.
pub struct Vm<'m, B: QuantumBackend, E: ExternProvider> {
    module: &'m BcModule,
    backend: B,
    externs: E,
    gate_procs: HashMap<SymbolId, ProcId>,
    sub_procs: HashMap<SymbolId, ProcId>,
    /// Dispatchable defcal candidates — indexes into
    /// [`BcModule::calibrations`] — gate defcals grouped by target gate
    /// symbol, measure/reset defcals as flat lists. Only OpenPulse
    /// bodies whose args are all plain `Param`s are dispatchable
    /// (measure additionally requires a return type).
    cal_gates: HashMap<SymbolId, Vec<u32>>,
    cal_measures: Vec<u32>,
    cal_resets: Vec<u32>,
    /// Per-procedure block-id → position-in-`blocks` lookup (`u32::MAX` =
    /// absent), built once in [`Vm::new`] to avoid a linear scan per jump.
    block_pos: Vec<Vec<u32>>,
    measurements: Vec<(u32, bool)>,
    /// When `Some`, leaf `U`/`gphase` calls are appended here instead of
    /// being applied to the backend (gate-body flattening for modifiers).
    recording: Option<Vec<Leaf>>,
    /// When `Some`, a `durationof` timing pass is active: quantum side
    /// effects are suppressed and elapsed time accumulates here.
    timing: Option<TimingState>,
    /// Installed OpenPulse handler. `Some` activates calibration
    /// execution: inline `cal` blocks run, and gate/measure/reset
    /// operations on hardware qubits that match a `defcal` execute the
    /// defcal body (see [`crate::cal`]).
    pulse: Option<Box<dyn OpenPulseHandler + 'm>>,
    /// Installed handler for non-OpenPulse `cal` block text.
    opaque_cal: Option<Box<dyn OpaqueCalHandler + 'm>>,
    /// Cal-scope globals (frames/ports/waveforms shared across all
    /// cal/defcal bodies), keyed by symbol. Written by `CalStore`, read
    /// by `CalLoad`; extern ports/frames are minted through the pulse
    /// handler on first read. Cleared at the start of each run.
    pulse_globals: HashMap<SymbolId, Value>,
    /// Span of the instruction (or block) currently executing, used to
    /// locate a runtime error in the source. Defaults to the empty sentinel.
    current_span: Span,
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
        let mut cal_gates: HashMap<SymbolId, Vec<u32>> = HashMap::new();
        let mut cal_measures = Vec::new();
        let mut cal_resets = Vec::new();
        for (i, cal) in module.calibrations.iter().enumerate() {
            let dispatchable = matches!(cal.body, BcCalBody::OpenPulse(_))
                && cal.args.iter().all(|a| matches!(a, BcCalArg::Param(_)));
            if !dispatchable {
                continue;
            }
            match cal.target {
                BcCalTarget::Gate(s) => cal_gates.entry(s).or_default().push(i as u32),
                BcCalTarget::Measure if cal.has_return => cal_measures.push(i as u32),
                // A measure defcal without a return type has nothing to
                // yield to `measure`; delay defcals are never dispatched.
                BcCalTarget::Measure | BcCalTarget::Delay => {}
                BcCalTarget::Reset => cal_resets.push(i as u32),
            }
        }
        // Per-procedure block-id → position lookup, so the block-walking
        // loop indexes directly instead of scanning `blocks` on every
        // jump. `u32::MAX` marks an absent id.
        let block_pos: Vec<Vec<u32>> = module
            .procedures
            .iter()
            .map(|proc| {
                let max_id = proc.blocks.iter().map(|b| b.id.0).max().unwrap_or(0);
                let mut pos = vec![u32::MAX; max_id as usize + 1];
                for (i, b) in proc.blocks.iter().enumerate() {
                    pos[b.id.0 as usize] = i as u32;
                }
                pos
            })
            .collect();
        Vm {
            module,
            backend,
            externs,
            gate_procs,
            sub_procs,
            cal_gates,
            cal_measures,
            cal_resets,
            block_pos,
            measurements: Vec::new(),
            recording: None,
            timing: None,
            pulse: None,
            opaque_cal: None,
            pulse_globals: HashMap::new(),
            current_span: Span::default(),
        }
    }

    /// Install an [`OpenPulseHandler`], activating calibration
    /// execution: inline `cal` blocks run, and gate/measure/reset
    /// operations on hardware qubits that match a `defcal` execute the
    /// defcal body instead of the gate's unitary definition.
    pub fn with_pulse_handler(mut self, handler: impl OpenPulseHandler + 'm) -> Self {
        self.pulse = Some(Box::new(handler));
        self
    }

    /// Install an [`OpaqueCalHandler`] to receive the text of inline
    /// `cal` blocks written in a non-OpenPulse `defcalgrammar`.
    pub fn with_opaque_cal_handler(mut self, handler: impl OpaqueCalHandler + 'm) -> Self {
        self.opaque_cal = Some(Box::new(handler));
        self
    }

    /// Access the backend (e.g. to read a simulator's state vector after
    /// a run).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Execute the module's entry procedure with no host-supplied
    /// inputs. If the program declares any `input`, prefer
    /// [`Vm::run_with_inputs`] — running with an empty map errors on the
    /// first missing input.
    pub async fn run(&mut self) -> std::result::Result<RunResult, VmError> {
        self.run_with_inputs(HashMap::new()).await
    }

    /// Execute the entry procedure, seeding each declared `input` from
    /// `inputs` (keyed by symbol id; resolve names via
    /// [`oqi_compile::bytecode::BcModule::symbols`]). Every declared
    /// input must be present and castable to its declared type; a value
    /// for a symbol that isn't a declared input is rejected.
    pub async fn run_with_inputs(
        &mut self,
        inputs: HashMap<SymbolId, Value>,
    ) -> std::result::Result<RunResult, VmError> {
        self.current_span = Span::default();
        self.run_inner(inputs)
            .await
            .map_err(|kind| VmError::new(kind).with_span(self.current_span))
    }

    /// The execution body, raising spanless [`VmErrorKind`]s. The span of the
    /// instruction in flight is tracked in `self.current_span` and attached by
    /// [`run_with_inputs`](Self::run_with_inputs).
    async fn run_inner(&mut self, mut inputs: HashMap<SymbolId, Value>) -> Result<RunResult> {
        self.pulse_globals.clear();
        let entry = self.module.entry;
        let mut regs = self.fresh_regs(entry);

        // Seed declared inputs; reject missing ones and type mismatches.
        let reg_types = &self.module.procedures[entry.0 as usize].register_types;
        for (sym, reg) in &self.module.inputs {
            let value = inputs.remove(sym).ok_or(VmErrorKind::MissingInput(*sym))?;
            let want = reg_types[reg.0 as usize];
            regs[reg.0 as usize] = Some(value.cast(want)?);
        }
        // Any leftover entries name symbols that aren't declared inputs.
        if let Some(sym) = inputs.keys().next() {
            return Err(VmErrorKind::UnknownInput(*sym));
        }

        let mut frame = Frame {
            proc: entry,
            regs,
            slots: self.fresh_slots(entry),
            mods: GateModifiers::none(),
        };
        self.exec_proc(&mut frame).await?;
        let outputs = self
            .module
            .outputs
            .iter()
            .filter_map(|(sym, reg)| {
                frame
                    .regs
                    .get(reg.0 as usize)
                    .and_then(|v| v.clone())
                    .map(|v| (*sym, v))
            })
            .collect();
        Ok(RunResult {
            measurements: std::mem::take(&mut self.measurements),
            registers: frame.regs,
            outputs,
        })
    }

    async fn exec_proc(&mut self, frame: &mut Frame) -> Result<Option<Value>> {
        let module = self.module;
        let proc = &module.procedures[frame.proc.0 as usize];
        let mut current = proc.entry;
        loop {
            let pos = self.block_pos[frame.proc.0 as usize]
                .get(current.0 as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if pos == u32::MAX {
                return Err(VmErrorKind::Unsupported(format!(
                    "missing block bb{}",
                    current.0
                )));
            }
            let block = &proc.blocks[pos as usize];
            for instr in &block.instrs {
                self.current_span = instr.span;
                self.exec_op(&instr.op, frame).await?;
            }
            self.current_span = block.span;
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
                BcTerminator::Unreachable => return Err(VmErrorKind::Unreachable),
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
        default.ok_or_else(|| {
            VmErrorKind::Unsupported("switch with no matching case or default".into())
        })
    }

    async fn exec_op(&mut self, op: &BcOp, frame: &mut Frame) -> Result<()> {
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
            BcOp::LoadElement {
                dest,
                base,
                indices,
            } => {
                let base = self.eval(frame, base)?;
                let idx = self.eval_index_items(frame, indices)?;
                let v = base.get(&idx)?;
                self.set(frame, *dest, v);
                Ok(())
            }
            BcOp::StoreElement {
                new,
                base,
                indices,
                value,
            } => {
                let mut arr = self.eval(frame, base)?;
                let idx = self.eval_index_items(frame, indices)?;
                let value = self.eval(frame, value)?;
                arr.set(&idx, value)?;
                self.set(frame, *new, arr);
                Ok(())
            }
            BcOp::StoreSlice {
                new,
                base,
                indices,
                value,
            } => {
                let mut arr = self.eval(frame, base)?;
                let value = self.eval(frame, value)?;
                let sel = Index::Select(indices.iter().map(|&i| i as isize).collect());
                arr.set(&[sel], value)?;
                self.set(frame, *new, arr);
                Ok(())
            }
            BcOp::NewArray { dest, items } => {
                let aty = self.array_ty(frame, *dest)?;
                let prims: Vec<Primitive> = items
                    .iter()
                    .map(|op| match self.eval(frame, op)? {
                        Value::Scalar(s) => Ok(s.into_value()),
                        _ => Err(VmErrorKind::Type("array element must be scalar".into())),
                    })
                    .collect::<Result<_>>()?;
                let arr = Array::new(prims, aty)?;
                self.set(frame, *dest, Value::Array(arr));
                Ok(())
            }

            // ── Call ──────────────────────────────────────────────────
            BcOp::Call { dest, callee, args } => self.exec_call(frame, *dest, callee, args).await,

            // ── Quantum ops ───────────────────────────────────────────
            BcOp::GateCall {
                gate,
                modifiers,
                args,
                qubits,
            } => {
                let call_span = self.current_span;
                let r = self
                    .exec_gate_call(*gate, modifiers, args, qubits, frame)
                    .await;
                // Flattening the gate body (recursion / `pow` replay) moves
                // `current_span` into the decomposition — down to synthetic
                // spans in the embedded `stdgates.inc`. Restore this call
                // site's span so an error from the body is attributed to the
                // instruction the user wrote, keeping it in range of the
                // rendered source.
                self.current_span = call_span;
                // Backend primitives are infallible; surface any deferred
                // error (e.g. a sum-over-Cliffords budget overflow) here.
                if let Some(kind) = self.backend.take_error() {
                    return Err(kind);
                }
                r
            }
            BcOp::Measure { dest, qubit } => {
                // Defcal dispatch: `measure $n` runs a matching measure
                // defcal, whose returned value is the measured bit.
                if self.pulse.is_some()
                    && let BcOperand::HardwareQubit(n) = qubit
                    && let Some(idx) = self.match_calibration(&self.cal_measures, &[*n], 0)
                {
                    let ret = self.exec_defcal(idx, &[], &[*n], frame).await?;
                    // In a timing pass the body only advanced frame
                    // clocks; nothing was measured or recorded.
                    if self.timing.is_some() {
                        if let Some(d) = dest {
                            self.set(frame, *d, Value::bit(false));
                        }
                        return Ok(());
                    }
                    let bit = value_bit(&ret.ok_or_else(|| {
                        VmErrorKind::Pulse("measure defcal returned no value".into())
                    })?)?;
                    self.measurements.push((*n, bit));
                    if let Some(d) = dest {
                        self.set(frame, *d, Value::bit(bit));
                    }
                    return Ok(());
                }
                let qs = self.qubits(frame, qubit)?;
                // Timing pass: nothing to measure — a deterministic 0
                // stands in (calibration control flow must have equal
                // branch durations, so the value must not matter).
                if self.timing.is_some() {
                    if let Some(d) = dest {
                        let v = if qs.len() == 1 {
                            Value::bit(false)
                        } else {
                            Value::bitreg_u128(0, qs.len() as u32)
                        };
                        self.set(frame, *d, v);
                    }
                    return Ok(());
                }
                let mut bits: u128 = 0;
                for (i, q) in qs.iter().enumerate() {
                    let b = self.backend.measure(*q).await;
                    self.measurements.push((*q, b));
                    if b {
                        bits |= 1 << i;
                    }
                }
                if let Some(kind) = self.backend.take_error() {
                    return Err(kind);
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
                // Defcal dispatch: `reset $n` runs a matching reset defcal.
                if self.pulse.is_some()
                    && let BcOperand::HardwareQubit(n) = qubit
                    && let Some(idx) = self.match_calibration(&self.cal_resets, &[*n], 0)
                {
                    self.exec_defcal(idx, &[], &[*n], frame).await?;
                    return Ok(());
                }
                // Timing pass: resets are suppressed (zero-duration).
                if self.timing.is_some() {
                    return Ok(());
                }
                for q in self.qubits(frame, qubit)? {
                    self.backend.reset(q).await;
                }
                if let Some(kind) = self.backend.take_error() {
                    return Err(kind);
                }
                Ok(())
            }
            BcOp::Barrier { qubits } => {
                // A barrier in a timing pass synchronizes all timelines.
                if let Some(t) = self.timing.as_mut() {
                    t.sync();
                    return Ok(());
                }
                let qs = self.qubit_list(frame, qubits)?;
                self.backend.barrier(&qs).await;
                Ok(())
            }
            BcOp::Delay { duration, qubits } => {
                let d = self.eval(frame, duration)?;
                let dur = match d {
                    Value::Scalar(s) => s.value().as_duration(),
                    _ => None,
                }
                .ok_or_else(|| VmErrorKind::Type("delay duration must be a duration".into()))?;
                // Timing pass: the delay advances the qubit timeline.
                if let Some(t) = self.timing.as_mut() {
                    t.advance_base(dur);
                    return Ok(());
                }
                let qs = self.qubit_list(frame, qubits)?;
                self.backend.delay(&qs, dur).await;
                Ok(())
            }
            BcOp::Nop { .. } => Ok(()),

            // ── Structured / misc ──────────────────────────────────────
            BcOp::Box { duration, body } => {
                // A `box[d]` pins its subcircuit to duration `d`: a
                // timing pass counts `d` and skips the body (recursing
                // would count its delays twice).
                if self.timing.is_some()
                    && let Some(dur) = duration
                {
                    let d = value_duration(&self.eval(frame, dur)?)?;
                    if let Some(t) = self.timing.as_mut() {
                        t.advance_base(d);
                    }
                    return Ok(());
                }
                let mut child = Frame {
                    proc: *body,
                    regs: self.fresh_regs(*body),
                    slots: self.fresh_slots(*body),
                    mods: GateModifiers::none(),
                };
                Box::pin(self.exec_proc(&mut child)).await?;
                Ok(())
            }
            BcOp::AliasBind { slot, segments } => {
                let mut resolved: Vec<u32> = Vec::new();
                for seg in segments {
                    match seg {
                        BcAliasSegment::Operand(op) => resolved.extend(self.qubits(frame, op)?),
                        BcAliasSegment::Slice {
                            source,
                            start,
                            step,
                            end,
                        } => {
                            let base = self.qubits(frame, source)?;
                            let sliced = self.slice_indices(
                                frame,
                                &base,
                                start.as_deref(),
                                step.as_deref(),
                                end.as_deref(),
                            )?;
                            resolved.extend(sliced);
                        }
                    }
                }
                *frame
                    .slots
                    .get_mut(*slot as usize)
                    .ok_or_else(|| VmErrorKind::Unsupported("alias slot out of range".into()))? =
                    resolved;
                Ok(())
            }
            BcOp::Pragma { .. } | BcOp::Alias { .. } => Ok(()),
            BcOp::CalOpaque { content } => {
                if let Some(handler) = self.opaque_cal.as_mut() {
                    let text = &self.module.strings[content.0 as usize];
                    handler.cal(self.module.calibration_grammar.as_deref(), text)?;
                }
                Ok(())
            }
            BcOp::CalOpenPulse { body } => {
                // Inline cal blocks execute only when a pulse handler is
                // installed; otherwise calibrations stay dormant.
                if self.pulse.is_some() {
                    let mut child = Frame {
                        proc: *body,
                        regs: self.fresh_regs(*body),
                        slots: self.fresh_slots(*body),
                        mods: GateModifiers::none(),
                    };
                    Box::pin(self.exec_proc(&mut child)).await?;
                }
                Ok(())
            }
            BcOp::CalLoad { dest, symbol } => {
                let v = self.cal_load(*symbol)?;
                self.set(frame, *dest, v);
                Ok(())
            }
            BcOp::CalStore { symbol, src } => {
                let v = self.eval(frame, src)?;
                self.pulse_globals.insert(*symbol, v);
                Ok(())
            }
            BcOp::DurationOf { dest, body } => {
                // Run the body as a timing pass: quantum side effects
                // are suppressed and elapsed time accumulates in a
                // fresh [`TimingState`]. Save/restore nests correctly.
                let saved = self.timing.replace(TimingState::new());
                let mut child = Frame {
                    proc: *body,
                    regs: self.fresh_regs(*body),
                    slots: self.fresh_slots(*body),
                    mods: GateModifiers::none(),
                };
                let res = Box::pin(self.exec_proc(&mut child)).await;
                let elapsed = self.timing.take().map(|t| t.elapsed());
                self.timing = saved;
                res?;
                let elapsed = elapsed.expect("timing state active during durationof");
                self.set(frame, *dest, Value::duration(elapsed.value, elapsed.unit));
                Ok(())
            }
        }
    }

    // ── Calls ─────────────────────────────────────────────────────────

    async fn exec_call(
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
                    // OpenPulse intrinsics called from cal/defcal bodies
                    // route to the typed pulse handler rather than the
                    // extern provider.
                    if self.pulse.is_some() && is_pulse_intrinsic(&name) && self.in_cal_proc(frame)
                    {
                        let ret = self.call_pulse_intrinsic(&name, &vals)?;
                        if let Some(d) = dest {
                            let v = ret.ok_or_else(|| {
                                VmErrorKind::Type(format!("intrinsic `{name}` returned no value"))
                            })?;
                            self.set(frame, d, v);
                        }
                        return Ok(());
                    }
                    let ret = self.externs.call(&name, &vals).await?;
                    if let Some(d) = dest {
                        let v = ret.ok_or_else(|| {
                            VmErrorKind::Type(format!("extern `{name}` returned no value"))
                        })?;
                        self.set(frame, d, v);
                    }
                    Ok(())
                } else {
                    let proc_id = *self.sub_procs.get(s).ok_or_else(|| {
                        VmErrorKind::Unsupported(format!("call to non-subroutine symbol s{}", s.0))
                    })?;
                    let mut child = self.bind_subroutine(frame, proc_id, args)?;
                    let ret = Box::pin(self.exec_proc(&mut child)).await?;
                    if let Some(d) = dest {
                        let v = ret.ok_or_else(|| {
                            VmErrorKind::Type(
                                "subroutine returned no value for a value call".into(),
                            )
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
        let mut slots = self.fresh_slots(proc_id);
        // Qubit args fill dense qubit slots `[0, n)`; classical args go to
        // the procedure's parameter registers, both in declaration order.
        let mut qslot = 0usize;
        let mut classical = 0usize;
        for arg in args {
            if is_qubit_operand(arg) {
                slots[qslot] = self.qubits(frame, arg)?;
                qslot += 1;
            } else {
                let v = self.eval(frame, arg)?;
                let reg = proc.params.get(classical).ok_or_else(|| {
                    VmErrorKind::Unsupported("more classical args than parameters".into())
                })?;
                regs[reg.0 as usize] = Some(v);
                classical += 1;
            }
        }
        Ok(Frame {
            proc: proc_id,
            regs,
            slots,
            mods: GateModifiers::none(),
        })
    }

    // ── Calibrations ──────────────────────────────────────────────────

    /// Whether `frame` is executing a calibration (`defcal`) or inline
    /// `cal` body.
    fn in_cal_proc(&self, frame: &Frame) -> bool {
        matches!(
            self.module.procedures[frame.proc.0 as usize].owner,
            ProcOwner::Calibration(_) | ProcOwner::InlineCal
        )
    }

    /// Read a cal-scope global. Extern ports/frames are minted through
    /// the pulse handler on first read; anything else must have been
    /// stored by a `cal` block that already ran.
    fn cal_load(&mut self, symbol: SymbolId) -> Result<Value> {
        if let Some(v) = self.pulse_globals.get(&symbol) {
            return Ok(v.clone());
        }
        let sym = self.module.symbols.get(symbol);
        let pulse = self.pulse.as_mut().ok_or_else(|| {
            VmErrorKind::Pulse(format!(
                "cal code executed without a pulse handler (reading `{}`)",
                sym.name
            ))
        })?;
        let handle = match sym.kind {
            SymbolKind::ExternPort => pulse.port(&sym.name)?.0,
            SymbolKind::ExternFrame => pulse.extern_frame(&sym.name)?.0,
            _ => {
                return Err(VmErrorKind::Pulse(format!(
                    "`{}` read before initialization (its `cal` block has not run)",
                    sym.name
                )));
            }
        };
        let v = handle_value(handle);
        self.pulse_globals.insert(symbol, v.clone());
        Ok(v)
    }

    /// Pick the best-matching calibration among `candidates` for a call
    /// on hardware qubits `hw` with `n_args` classical arguments. Most
    /// exact (`Hardware`) operand matches win — the spec's specificity
    /// rule (docs/pulses.rst) — and ties go to the first declared (the
    /// spec doesn't address equally-specific collisions).
    fn match_calibration(&self, candidates: &[u32], hw: &[u32], n_args: usize) -> Option<u32> {
        let mut best: Option<(u32, usize)> = None;
        for &idx in candidates {
            let cal = &self.module.calibrations[idx as usize];
            if cal.operands.len() != hw.len() || cal.args.len() != n_args {
                continue;
            }
            let mut exact = 0usize;
            let matches = cal.operands.iter().zip(hw).all(|(op, q)| match op {
                BcCalOperand::Hardware(n) => {
                    exact += 1;
                    n == q
                }
                BcCalOperand::Any => true,
            });
            if matches && best.is_none_or(|(_, e)| exact > e) {
                best = Some((idx, exact));
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// Execute defcal `idx` for a call with classical `args` on hardware
    /// qubits `hw`: like [`Self::bind_subroutine`], classical args bind
    /// to the body's parameter registers and each qubit to its
    /// positional slot `[0, n)`. Returns the body's return value
    /// (measure defcals yield the measured bit).
    async fn exec_defcal(
        &mut self,
        idx: u32,
        args: &[BcOperand],
        hw: &[u32],
        frame: &Frame,
    ) -> Result<Option<Value>> {
        let BcCalBody::OpenPulse(proc_id) = self.module.calibrations[idx as usize].body else {
            return Err(VmErrorKind::Pulse(
                "opaque defcal bodies cannot execute".into(),
            ));
        };
        let proc = &self.module.procedures[proc_id.0 as usize];
        let mut regs = self.fresh_regs(proc_id);
        let mut slots = self.fresh_slots(proc_id);
        for (i, arg) in args.iter().enumerate() {
            let v = self.eval(frame, arg)?;
            let reg = proc.params.get(i).ok_or_else(|| {
                VmErrorKind::Unsupported("more classical args than defcal parameters".into())
            })?;
            regs[reg.0 as usize] = Some(v);
        }
        for (slot, q) in hw.iter().enumerate() {
            slots[slot] = vec![*q];
        }
        let mut child = Frame {
            proc: proc_id,
            regs,
            slots,
            mods: GateModifiers::none(),
        };
        Box::pin(self.exec_proc(&mut child)).await
    }

    /// Marshal one OpenPulse intrinsic call to the typed pulse handler.
    /// Handles travel as `uint[64]` values; angles arrive in radians.
    fn call_pulse_intrinsic(&mut self, name: &str, args: &[Value]) -> Result<Option<Value>> {
        let pulse = self
            .pulse
            .as_mut()
            .expect("caller checks a pulse handler is installed");
        let arg = |n: usize| -> Result<&Value> {
            args.get(n).ok_or_else(|| {
                VmErrorKind::Type(format!("intrinsic `{name}` missing argument {n}"))
            })
        };
        // A durationof timing pass suppresses pulse emissions: plays
        // and captures advance their frame's clock by handler-reported
        // durations instead. Constructors (`newframe`, `gaussian`)
        // fall through and run for real — their handles must exist so
        // their durations can be queried.
        if let Some(timing) = self.timing.as_mut() {
            match name {
                "play" => {
                    let frame = value_u64(arg(0)?)?;
                    let wf = WaveformHandle(value_u64(arg(1)?)?);
                    timing.advance_frame(frame, pulse.waveform_duration(wf)?);
                    return Ok(None);
                }
                "capture" => {
                    let frame = value_u64(arg(0)?)?;
                    let samples = value_u64(arg(1)?)?;
                    let d = pulse.capture_duration(FrameHandle(frame), samples)?;
                    timing.advance_frame(frame, d);
                    return Ok(Some(Value::complex(0.0, 0.0, FloatWidth::F64)));
                }
                // Instantaneous; suppressed (threshold would see the
                // fake IQ, so the handler isn't consulted).
                "shift_phase" => return Ok(None),
                "threshold" => return Ok(Some(Value::bit(false))),
                _ => {}
            }
        }
        Ok(match name {
            "newframe" => {
                let frame = pulse.new_frame(
                    PortHandle(value_u64(arg(0)?)?),
                    value_f64(arg(1)?)?,
                    value_f64(arg(2)?)?,
                )?;
                Some(handle_value(frame.0))
            }
            "gaussian" => {
                let wf = pulse.gaussian(
                    value_f64(arg(0)?)?,
                    value_duration(arg(1)?)?,
                    value_duration(arg(2)?)?,
                )?;
                Some(handle_value(wf.0))
            }
            "play" => {
                pulse.play(
                    FrameHandle(value_u64(arg(0)?)?),
                    WaveformHandle(value_u64(arg(1)?)?),
                )?;
                None
            }
            "capture" => {
                let iq = pulse.capture(FrameHandle(value_u64(arg(0)?)?), value_u64(arg(1)?)?)?;
                Some(Value::complex(iq.re, iq.im, FloatWidth::F64))
            }
            "shift_phase" => {
                pulse.shift_phase(FrameHandle(value_u64(arg(0)?)?), value_f64(arg(1)?)?)?;
                None
            }
            "threshold" => {
                let bit = pulse.threshold(value_complex(arg(0)?)?, value_u64(arg(1)?)?)?;
                Some(Value::bit(bit))
            }
            other => {
                return Err(VmErrorKind::Pulse(format!(
                    "`{other}` is not an OpenPulse intrinsic"
                )));
            }
        })
    }

    // ── Gates ─────────────────────────────────────────────────────────

    async fn exec_gate_call(
        &mut self,
        gate: SymbolId,
        modifiers: &[BcGateModifier],
        args: &[BcOperand],
        qubits: &[BcOperand],
        frame: &mut Frame,
    ) -> Result<()> {
        // Defcal dispatch: with a pulse handler installed, an unmodified
        // call on hardware qubits that matches a defcal executes the
        // defcal body in place of the gate's unitary definition
        // (docs/pulses.rst). Modified calls (`ctrl`/`pow`/`inv`, or
        // inherited controls) and calls during gate-body flattening
        // always take the unitary path below.
        if self.pulse.is_some()
            && modifiers.is_empty()
            && frame.mods.controls.is_empty()
            && frame.mods.neg_controls.is_empty()
            && self.recording.is_none()
            && let Some(hw) = hardware_operands(qubits)
        {
            let cal = self
                .cal_gates
                .get(&gate)
                .and_then(|cands| self.match_calibration(cands, &hw, args.len()));
            if let Some(idx) = cal {
                self.exec_defcal(idx, args, &hw, frame).await?;
                return Ok(());
            }
        }

        // Fold this call's `inv`/`pow` modifiers into one local power
        // scalar, and accumulate controls (inherited context + this
        // call's), consuming leading qubit operands as controls. Power is
        // *not* inherited: an enclosing `inv`/`pow` is resolved where it
        // appears, by flattening that gate's body (see the proc branch
        // below), so `frame.mods.power` is always 1.
        let mut power = 1.0;
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
                        VmErrorKind::Unsupported("gate has fewer params than arguments".into())
                    })?;
                    regs[reg.0 as usize] = Some(v.clone());
                }
                // Bind the gate's qubit params into slots `[0, n)`.
                // Singletons repeat; registers (validated equal to `bcast`)
                // are indexed by the broadcast position.
                let mut slots = self.fresh_slots(proc_id);
                for (s, qs) in resolved.iter().enumerate() {
                    let idx = if qs.len() == 1 { 0 } else { j };
                    slots[s] = vec![qs[idx]];
                }
                if eff.power == 1.0 {
                    // No `inv`/`pow`: recurse, propagating controls. `ctrl`
                    // distributes over a sequence, so this is exact.
                    let mut child = Frame {
                        proc: proc_id,
                        regs,
                        slots,
                        mods: eff.clone(),
                    };
                    Box::pin(self.exec_proc(&mut child)).await?;
                } else {
                    // `inv`/`pow` on a (possibly composite) body: flatten
                    // the body to a leaf trace, then reverse+invert or
                    // repeat it before emitting. Recording runs with a
                    // clean context; the outer controls are merged back in
                    // at emit time.
                    let mut child = Frame {
                        proc: proc_id,
                        regs,
                        slots,
                        mods: GateModifiers::none(),
                    };
                    let prev = self.recording.take();
                    self.recording = Some(Vec::new());
                    let res = Box::pin(self.exec_proc(&mut child)).await;
                    let trace = self.recording.take().unwrap_or_default();
                    self.recording = prev;
                    res?;
                    self.apply_transformed(trace, eff.power, &eff).await?;
                }
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
                            self.emit_u(q, theta, phi, lambda, &eff).await;
                        }
                    }
                    Ok(())
                }
                "gphase" => {
                    let gamma = value_f64(&self.eval(frame, &args[0])?)?;
                    self.emit_gphase(gamma, &eff).await;
                    Ok(())
                }
                other => Err(VmErrorKind::UndefinedGate(other.to_string())),
            }
        }
    }

    /// Apply (or, when recording, record) a leaf `U`.
    async fn emit_u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, m: &GateModifiers) {
        // A durationof timing pass applies no gates (uncalibrated
        // gates are zero-duration).
        if self.timing.is_some() {
            return;
        }
        if let Some(buf) = self.recording.as_mut() {
            buf.push(Leaf::U {
                target,
                theta,
                phi,
                lambda,
                controls: m.controls.clone(),
                neg_controls: m.neg_controls.clone(),
                power: m.power,
            });
        } else {
            self.backend.u(target, theta, phi, lambda, m).await;
        }
    }

    /// Apply (or, when recording, record) a leaf `gphase`.
    async fn emit_gphase(&mut self, gamma: f64, m: &GateModifiers) {
        // See `emit_u`: timing passes apply no gates.
        if self.timing.is_some() {
            return;
        }
        if let Some(buf) = self.recording.as_mut() {
            buf.push(Leaf::Gphase {
                gamma,
                controls: m.controls.clone(),
                neg_controls: m.neg_controls.clone(),
                power: m.power,
            });
        } else {
            self.backend.gphase(gamma, m).await;
        }
    }

    /// Emit one recorded leaf, scaling its power by `factor` (1 for a
    /// forward repeat, -1 to invert, or the exponent for a fractional
    /// power) and merging the outer controls `eff` onto it.
    async fn emit_leaf(&mut self, leaf: &Leaf, factor: f64, eff: &GateModifiers) {
        match leaf {
            Leaf::U {
                target,
                theta,
                phi,
                lambda,
                controls,
                neg_controls,
                power,
            } => {
                let m = GateModifiers {
                    controls: merge(controls, &eff.controls),
                    neg_controls: merge(neg_controls, &eff.neg_controls),
                    power: power * factor,
                };
                self.emit_u(*target, *theta, *phi, *lambda, &m).await;
            }
            Leaf::Gphase {
                gamma,
                controls,
                neg_controls,
                power,
            } => {
                let m = GateModifiers {
                    controls: merge(controls, &eff.controls),
                    neg_controls: merge(neg_controls, &eff.neg_controls),
                    power: power * factor,
                };
                self.emit_gphase(*gamma, &m).await;
            }
        }
    }

    /// Emit a flattened gate-body `trace` raised to the power `p`, merging
    /// the outer controls `eff` onto every leaf.
    ///
    /// - integer `p`: repeat the trace `|p|` times (reversed + inverted
    ///   when `p < 0`). Exact, since each leaf is exactly invertible.
    /// - fractional `p`: only exact when at most one leaf acts on a qubit
    ///   (a single `U`, or a relative phase, plus commuting global
    ///   phases), where the power folds per-leaf. A fractional power of a
    ///   genuinely composite body would need a dense matrix power and is
    ///   rejected.
    async fn apply_transformed(
        &mut self,
        trace: Vec<Leaf>,
        p: f64,
        eff: &GateModifiers,
    ) -> Result<()> {
        let rounded = p.round();
        if (rounded - p).abs() < 1e-9 {
            let k = rounded as i64;
            let reps = k.unsigned_abs();
            if k >= 0 {
                for _ in 0..reps {
                    for leaf in &trace {
                        self.emit_leaf(leaf, 1.0, eff).await;
                    }
                }
            } else {
                for _ in 0..reps {
                    for leaf in trace.iter().rev() {
                        self.emit_leaf(leaf, -1.0, eff).await;
                    }
                }
            }
            Ok(())
        } else {
            if trace.iter().filter(|l| l.is_real()).count() > 1 {
                return Err(VmErrorKind::Unsupported(
                    "fractional power of a multi-qubit composite gate".into(),
                ));
            }
            for leaf in &trace {
                self.emit_leaf(leaf, p, eff).await;
            }
            Ok(())
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

    /// Evaluate per-dimension index operands into `Index::Item`s for an
    /// element load/store (`a[i, j, …]`).
    fn eval_index_items(&self, frame: &Frame, indices: &[BcOperand]) -> Result<Vec<Index>> {
        indices
            .iter()
            .map(|op| Ok(Index::Item(value_isize(&self.eval(frame, op)?)?)))
            .collect()
    }

    /// Resolve a classical operand to a value.
    fn eval(&self, frame: &Frame, op: &BcOperand) -> Result<Value> {
        match op {
            BcOperand::Reg(r) => frame.regs[r.0 as usize]
                .clone()
                .ok_or(VmErrorKind::UnsetRegister(r.0)),
            BcOperand::Const(c) => Ok(self.module.constants[c.0 as usize].clone()),
            _ => Err(VmErrorKind::Type("expected a classical operand".into())),
        }
    }

    fn array_ty(&self, frame: &Frame, dest: Reg) -> Result<ArrayTy> {
        match self.module.procedures[frame.proc.0 as usize].register_types[dest.0 as usize] {
            ValueTy::Array(aty) => Ok(aty),
            _ => Err(VmErrorKind::Type(
                "NewArray destination is not an array".into(),
            )),
        }
    }

    /// Resolve a qubit operand to its global qubit indices, validating that
    /// each is within the program's allocated qubit memory.
    fn qubits(&self, frame: &Frame, op: &BcOperand) -> Result<Vec<u32>> {
        let resolved = match op {
            BcOperand::Qubit(i) => vec![*i],
            BcOperand::HardwareQubit(i) => vec![*i],
            BcOperand::Whole(source) => self.source_list(frame, source)?,
            BcOperand::Select { source, positions } => {
                let bound = self.source_list(frame, source)?;
                positions
                    .iter()
                    .map(|&p| {
                        bound.get(p as usize).copied().ok_or_else(|| {
                            VmErrorKind::Type(format!("qubit index {p} out of range"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            BcOperand::Index { source, index } => {
                let bound = self.source_list(frame, source)?;
                let i = value_usize(&self.eval(frame, index)?)?;
                bound
                    .get(i)
                    .copied()
                    .map(|q| vec![q])
                    .ok_or_else(|| VmErrorKind::Type(format!("qubit index {i} out of range")))?
            }
            _ => return Err(VmErrorKind::Type("expected a qubit operand".into())),
        };

        let num_qubits = self.module.qubits.num_qubits;
        for &q in &resolved {
            if q >= num_qubits {
                return Err(VmErrorKind::QubitOutOfRange {
                    qubit: q as usize,
                    num_qubits: num_qubits as usize,
                });
            }
        }
        Ok(resolved)
    }

    /// The global qubit indices of a [`QubitSource`]: a region of global
    /// memory, or a frame-local slot (bound parameter or runtime alias).
    fn source_list(&self, frame: &Frame, source: &QubitSource) -> Result<Vec<u32>> {
        match source {
            QubitSource::Region(id) => Ok(self.region_indices(id.0)),
            QubitSource::Slot(s) => frame
                .slots
                .get(*s as usize)
                .cloned()
                .ok_or_else(|| VmErrorKind::Unsupported("unbound qubit slot".into())),
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

    /// Apply OpenQASM range semantics `base[start : step : end]` at run
    /// time (docs/types.rst): inclusive bounds, defaults start 0 / step 1
    /// / end len-1, negative indices counting from the end. Used to bind
    /// runtime-valued alias slices ([`BcAliasSegment::Slice`]).
    fn slice_indices(
        &self,
        frame: &Frame,
        base: &[u32],
        start: Option<&BcOperand>,
        step: Option<&BcOperand>,
        end: Option<&BcOperand>,
    ) -> Result<Vec<u32>> {
        let len = base.len() as i128;
        let step = match step {
            Some(op) => value_i128(&self.eval(frame, op)?)?,
            None => 1,
        };
        if step == 0 {
            return Err(VmErrorKind::Type("range step must be non-zero".into()));
        }
        let start = match start {
            Some(op) => value_i128(&self.eval(frame, op)?)?,
            None if step > 0 => 0,
            None => len - 1,
        };
        let end = match end {
            Some(op) => value_i128(&self.eval(frame, op)?)?,
            None if step > 0 => len - 1,
            None => 0,
        };

        let mut out = Vec::new();
        let mut cur = start;
        while (step > 0 && cur <= end) || (step < 0 && cur >= end) {
            let adj = if cur < 0 { cur + len } else { cur };
            if !(0..len).contains(&adj) {
                return Err(VmErrorKind::Type(format!("qubit index {cur} out of range")));
            }
            out.push(base[adj as usize]);
            cur += step;
        }
        Ok(out)
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

    /// An empty qubit-slot vector sized to `proc`'s slot count. Parameter
    /// slots are filled by the caller; alias slots are filled by
    /// [`BcOp::AliasBind`] as the body runs.
    fn fresh_slots(&self, proc: ProcId) -> Vec<Vec<u32>> {
        vec![Vec::new(); self.module.procedures[proc.0 as usize].num_qubit_slots as usize]
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
            return Err(VmErrorKind::BroadcastMismatch(
                operands.iter().map(|q| q.len()).collect(),
            ));
        }
    }
    Ok(n)
}

/// Concatenate two control lists (a leaf's own controls and the outer
/// modifier context's).
fn merge(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}

fn is_qubit_operand(op: &BcOperand) -> bool {
    matches!(
        op,
        BcOperand::Qubit(_)
            | BcOperand::HardwareQubit(_)
            | BcOperand::Whole(_)
            | BcOperand::Select { .. }
            | BcOperand::Index { .. }
    )
}

/// The hardware indices of a qubit-operand list, if every operand is a
/// literal `$n` reference — the only form defcals dispatch on
/// (docs/pulses.rst binds calibrations to physical qubits).
fn hardware_operands(qubits: &[BcOperand]) -> Option<Vec<u32>> {
    qubits
        .iter()
        .map(|q| match q {
            BcOperand::HardwareQubit(n) => Some(*n),
            _ => None,
        })
        .collect()
}

fn scalar(v: &Value) -> Result<&Primitive> {
    match v {
        Value::Scalar(s) => Ok(s.value()),
        _ => Err(VmErrorKind::Type("expected a scalar value".into())),
    }
}

fn value_bit(v: &Value) -> Result<bool> {
    Ok(scalar(v)?.as_bit())
}

fn value_f64(v: &Value) -> Result<f64> {
    scalar(v)?
        .as_float(FloatWidth::F64)
        .ok_or_else(|| VmErrorKind::Type("expected a float-like value".into()))
}

fn value_i128(v: &Value) -> Result<i128> {
    scalar(v)?
        .as_int(iw(128))
        .ok_or_else(|| VmErrorKind::Type("expected an integer value".into()))
}

fn value_usize(v: &Value) -> Result<usize> {
    scalar(v)?
        .as_uint(iw(64))
        .map(|u| u as usize)
        .ok_or_else(|| VmErrorKind::Type("expected an unsigned integer value".into()))
}

fn value_isize(v: &Value) -> Result<isize> {
    Ok(value_i128(v)? as isize)
}

fn value_u64(v: &Value) -> Result<u64> {
    scalar(v)?
        .as_uint(iw(64))
        .map(|u| u as u64)
        .ok_or_else(|| VmErrorKind::Type("expected an unsigned integer value".into()))
}

fn value_duration(v: &Value) -> Result<Duration> {
    scalar(v)?
        .as_duration()
        .ok_or_else(|| VmErrorKind::Type("expected a duration value".into()))
}

fn value_complex(v: &Value) -> Result<Complex64> {
    scalar(v)?
        .as_complex(FloatWidth::F64)
        .ok_or_else(|| VmErrorKind::Type("expected a complex value".into()))
}

/// Wrap an opaque pulse handle for storage in a register.
fn handle_value(h: u64) -> Value {
    Value::uint(h as u128, iw(64))
}

/// The OpenPulse intrinsic names seeded by `defcalgrammar "openpulse"`.
fn is_pulse_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "newframe" | "gaussian" | "play" | "capture" | "shift_phase" | "threshold"
    )
}

/// Evaluate an intrinsic call against the classical op library.
fn eval_intrinsic(i: &Intrinsic, args: &[Value]) -> Result<Value> {
    let arg0 = || -> Result<Value> {
        args.first()
            .cloned()
            .ok_or_else(|| VmErrorKind::Type("intrinsic missing argument".into()))
    };
    let arg = |n: usize| -> Result<Value> {
        args.get(n)
            .cloned()
            .ok_or_else(|| VmErrorKind::Type("intrinsic missing argument".into()))
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
