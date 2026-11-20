//! The bytecode interpreter.
//!
//! Executes a [`BcModule`] by walking each procedure's basic blocks and
//! following terminators. Classical work is delegated to `oqi-classical`
//! (operators, `Value::cast`, array indexing); quantum work to a
//! pluggable [`QuantumBackend`]; `extern` calls to an [`ExternProvider`].

use std::collections::HashMap;

use num_complex::Complex64;
use oqi_classical::{
    Array, ArrayRef, ArrayTy, BitReg, Duration, DurationUnit, FloatWidth, Index, Primitive,
    PrimitiveTy, RefAccess, Scalar, Value, ValueTy, iw,
};
use oqi_compile::bytecode::{
    BcAliasSegment, BcCalArg, BcCalBody, BcCalOperand, BcCalTarget, BcCallTarget, BcGateModifier,
    BcModule, BcOp, BcOperand, BcTerminator, BlockId, ProcId, ProcOwner, QubitSource, Reg,
    is_sample_safe,
};
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_lex::Span;

use crate::backend::{BuiltinGate, GateModifiers, QuantumBackend};
use crate::cal::{FrameHandle, OpaqueCalHandler, OpenPulseHandler, PortHandle, WaveformHandle};
use crate::error::{KindResult, VmError, VmErrorKind, VmResult};
use crate::extern_fns::ExternProvider;
use crate::host::{HostValue, coerce_inputs};
use crate::rng::Rng;

/// The outcome of a run.
#[derive(Debug)]
pub struct RunResult {
    /// Measurement outcomes in execution order: (global qubit index, bit).
    pub measurements: Vec<(u32, bool)>,
    /// Named program outputs and their final values. Each `SymbolId`
    /// resolves to a name via [`BcModule::symbol`]. See
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
    /// When `Some`, built-in gate calls are appended here instead of being
    /// applied to the backend. Used to apply `inv`/`pow` to composite gates
    /// (see [`Vm::exec_gate_call`]): the body runs once with recording on,
    /// producing a straight-line trace that is then reversed and inverted
    /// (for `inv`) or repeated (for integer `pow`) before emission.
    recording: Option<Vec<BuiltinGate>>,
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
    /// Shot-sampling "capture" mode: snapshot the amplitudes at the first
    /// measurement (into `captured_state`) before it collapses. Set by
    /// `run_capture`.
    capture: bool,
    captured_state: Option<Vec<Complex64>>,
    /// Shot-sampling "inject" mode: replay the program with pre-sampled
    /// measurement outcomes. Quantum ops (gate/reset/barrier/delay) become
    /// no-ops and `measure` returns the injected bit per qubit, so the
    /// classical program runs quantum-free. Set by `run_inject`.
    inject: Option<Vec<bool>>,
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
            capture: false,
            captured_state: None,
            inject: None,
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
    /// [`Vm::run_with_named_inputs`] — running with none errors on the
    /// first missing input.
    pub async fn run(&mut self) -> VmResult<RunResult> {
        self.run_with_inputs(HashMap::new()).await
    }

    /// Execute the entry procedure, seeding each declared `input` from
    /// `inputs` (keyed by symbol id). Every declared input must be present
    /// and castable to its declared type; a value for a symbol that isn't a
    /// declared input is rejected.
    pub async fn run_with_inputs(
        &mut self,
        inputs: HashMap<SymbolId, Value>,
    ) -> VmResult<RunResult> {
        self.current_span = Span::default();
        self.run_inner(inputs)
            .await
            .map_err(|kind| VmError::new(kind).with_span(self.current_span))
    }

    /// [`run_with_inputs`](Self::run_with_inputs) with the inputs given by
    /// name as host values, coerced to their declared types (see
    /// [`coerce_inputs`]).
    pub async fn run_with_named_inputs(
        &mut self,
        inputs: &[(String, HostValue)],
    ) -> VmResult<RunResult> {
        let inputs = coerce_inputs(self.module, inputs)?;
        self.run_with_inputs(inputs).await
    }

    /// Run the program `shots` times on a fresh backend and return every
    /// shot's result, in order.
    ///
    /// When no quantum operation depends on a measurement outcome
    /// ([`is_sample_safe`]) and the backend holds a dense state, the circuit
    /// runs once: its pre-measurement state is snapshotted and each shot is
    /// drawn from that state's Born-rule distribution with an RNG seeded by
    /// `seed`, replaying only the classical program. Otherwise the whole
    /// circuit is re-run per shot, re-zeroing the backend in between while
    /// the backend's own RNG stream advances — there the backend's seed, not
    /// `seed`, decides the outcomes (front doors give both the same seed).
    pub async fn sample(
        &mut self,
        inputs: HashMap<SymbolId, Value>,
        shots: u32,
        seed: u64,
    ) -> VmResult<Vec<RunResult>> {
        let n = self.module.qubits.num_qubits;
        let mut results = Vec::with_capacity(shots as usize);
        let mut fresh = true;
        if self.backend.has_dense_state() && is_sample_safe(self.module) {
            fresh = false;
            if let Some(psi) = self.run_capture(inputs.clone()).await? {
                // Cumulative |ψ|² table over basis states.
                let mut cumulative = Vec::with_capacity(psi.len());
                let mut total = 0.0f64;
                for amplitude in &psi {
                    total += amplitude.norm_sqr();
                    cumulative.push(total);
                }
                let total = total.max(f64::MIN_POSITIVE);
                let mut rng = Rng::new(seed);
                for _ in 0..shots {
                    let u = rng.next_f64() * total;
                    let idx = cumulative.partition_point(|&c| c < u).min(psi.len() - 1);
                    let basis = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
                    results.push(self.run_inject(inputs.clone(), basis).await?);
                }
                return Ok(results);
            }
        }
        for _ in 0..shots {
            if !fresh {
                self.backend.reset_state()?;
            }
            fresh = false;
            results.push(self.run_with_inputs(inputs.clone()).await?);
        }
        Ok(results)
    }

    /// Run once to snapshot the pre-measurement state: the amplitudes at the
    /// first measurement (before it collapses), or `None` if the program never
    /// measures or the backend has no addressable state vector. Only sound
    /// when the program is sample-safe (no quantum op after a measurement).
    async fn run_capture(
        &mut self,
        inputs: HashMap<SymbolId, Value>,
    ) -> VmResult<Option<Vec<Complex64>>> {
        self.capture = true;
        self.captured_state = None;
        let r = self.run_with_inputs(inputs).await;
        self.capture = false;
        r.map(|_| self.captured_state.take())
    }

    /// Replay the program classically with pre-sampled measurement outcomes
    /// (`basis[q]` = the measured value of global qubit `q`): quantum ops are
    /// no-ops, `measure` returns the injected bits, and the classical program
    /// (including branches on measured values) runs to produce the outputs of
    /// one shot. Cheap — no backend work.
    async fn run_inject(
        &mut self,
        inputs: HashMap<SymbolId, Value>,
        basis: Vec<bool>,
    ) -> VmResult<RunResult> {
        self.inject = Some(basis);
        let r = self.run_with_inputs(inputs).await;
        self.inject = None;
        r
    }

    /// The execution body, raising spanless [`VmErrorKind`]s. The span of the
    /// instruction in flight is tracked in `self.current_span` and attached by
    /// [`run_with_inputs`](Self::run_with_inputs).
    async fn run_inner(&mut self, mut inputs: HashMap<SymbolId, Value>) -> KindResult<RunResult> {
        self.pulse_globals.clear();
        self.measurements.clear();
        let entry = self.module.entry;
        let mut frame = self.fresh_frame(entry);

        // Seed declared inputs; reject missing ones and type mismatches.
        let reg_types = &self.module.procedures[entry.0 as usize].register_types;
        for (sym, reg) in &self.module.inputs {
            let value = inputs
                .remove(sym)
                .ok_or_else(|| VmErrorKind::MissingInput(self.module.symbol(*sym).name.clone()))?;
            let want = reg_types[reg.0 as usize];
            frame.regs[reg.0 as usize] = Some(value.cast(want)?);
        }
        // Any leftover entries name symbols that aren't declared inputs
        // (report the first declared, so the error is deterministic).
        if let Some(sym) = inputs.keys().min_by_key(|sym| sym.0) {
            return Err(VmErrorKind::UnknownInput(
                self.module.symbol(*sym).name.clone(),
            ));
        }

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
            outputs,
        })
    }

    async fn exec_proc(&mut self, frame: &mut Frame) -> KindResult<Option<Value>> {
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
        cases: &[(Vec<BcOperand>, BlockId)],
        default: BlockId,
    ) -> KindResult<BlockId> {
        for (labels, bb) in cases {
            for v in labels {
                if value_i128(&self.eval(frame, v)?)? == target {
                    return Ok(*bb);
                }
            }
        }
        Ok(default)
    }

    async fn exec_op(&mut self, op: &BcOp, frame: &mut Frame) -> KindResult<()> {
        match op {
            // ── Classical operators ───────────────────────────────────
            BcOp::Binary { op, dest, lhs, rhs } => {
                let (lhs, rhs) = (self.eval(frame, lhs)?, self.eval(frame, rhs)?);
                self.set(frame, *dest, op.apply(lhs, rhs)?);
                Ok(())
            }
            BcOp::Unary { op, dest, src } => {
                let value = op.apply(self.eval(frame, src)?)?;
                self.set(frame, *dest, value);
                Ok(())
            }
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
                    .collect::<KindResult<_>>()?;
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
                // Inject (shot replay): gates are no-ops — the state is
                // already captured and outcomes are pre-sampled.
                if self.inject.is_some() {
                    return Ok(());
                }
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
                        self.set(frame, *d, measured_value(&vec![false; qs.len()]));
                    }
                    return Ok(());
                }
                let outcomes: Vec<bool> = if let Some(basis) = &self.inject {
                    // Inject (shot replay): the pre-sampled outcomes for
                    // these qubits; the backend is untouched.
                    qs.iter().map(|&q| basis[q as usize]).collect()
                } else {
                    // Capture (fast-path snapshot): stash the amplitudes at
                    // the first measurement, before it collapses.
                    if self.capture && self.captured_state.is_none() {
                        self.captured_state = self.backend.amplitudes().await;
                    }
                    self.backend.measure_all(&qs).await
                };
                self.measurements
                    .extend(qs.iter().copied().zip(outcomes.iter().copied()));
                if let Some(d) = dest {
                    self.set(frame, *d, measured_value(&outcomes));
                }
                Ok(())
            }
            BcOp::Reset { qubit } => {
                // Inject (shot replay): quantum no-op.
                if self.inject.is_some() {
                    return Ok(());
                }
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
                Ok(())
            }
            BcOp::Barrier { qubits } => {
                // Inject (shot replay): quantum no-op.
                if self.inject.is_some() {
                    return Ok(());
                }
                // A barrier in a timing pass synchronizes all timelines.
                if let Some(t) = self.timing.as_mut() {
                    t.sync();
                    return Ok(());
                }
                let qs = self.qubit_list(frame, qubits)?;
                self.backend.barrier(&qs);
                Ok(())
            }
            BcOp::Delay { duration, qubits } => {
                // Inject (shot replay): quantum no-op.
                if self.inject.is_some() {
                    return Ok(());
                }
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
                self.backend.delay(&qs, dur);
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
                let mut child = self.fresh_frame(*body);
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
                    let mut child = self.fresh_frame(*body);
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
                let mut child = self.fresh_frame(*body);
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
    ) -> KindResult<()> {
        let module = self.module;
        let (name, ret) = match callee {
            BcCallTarget::Intrinsic(i) => {
                let vals = self.eval_all(frame, args)?;
                let v = i.apply(vals).ok_or_else(|| {
                    VmErrorKind::Type(format!("wrong number of arguments for `{i}`"))
                })??;
                (i.name(), Some(v))
            }
            BcCallTarget::Symbol(s) => {
                let sym = module.symbol(*s);
                let name = sym.name.as_str();
                let ret = if matches!(
                    sym.kind,
                    SymbolKind::Extern | SymbolKind::ExternPort | SymbolKind::ExternFrame
                ) {
                    let vals = self.eval_all(frame, args)?;
                    // OpenPulse intrinsics called from cal/defcal bodies
                    // route to the typed pulse handler rather than the
                    // extern provider.
                    if self.pulse.is_some() && is_pulse_intrinsic(name) && self.in_cal_proc(frame) {
                        self.call_pulse_intrinsic(name, &vals)?
                    } else {
                        self.externs.call(name, &vals).await?
                    }
                } else {
                    let proc_id = *self.sub_procs.get(s).ok_or_else(|| {
                        VmErrorKind::Unsupported(format!("call to non-subroutine symbol s{}", s.0))
                    })?;
                    let (mut child, by_ref) = self.bind_subroutine(frame, proc_id, args)?;
                    let ret = Box::pin(self.exec_proc(&mut child)).await?;
                    // Hand each array lent to a `mutable` parameter back to
                    // the register it came from.
                    for (reg, array) in by_ref {
                        self.set(frame, reg, Value::Array(array.to_owned()?));
                    }
                    ret
                };
                (name, ret)
            }
        };
        if let Some(d) = dest {
            let v = ret.ok_or_else(|| VmErrorKind::Type(format!("`{name}` returned no value")))?;
            self.set(frame, d, v);
        }
        Ok(())
    }

    /// Build a callee frame for a subroutine, binding positional args.
    /// Classical args go to the procedure's parameter registers (in
    /// declaration order); qubit args bind to their positional slot.
    ///
    /// Array parameters are references (docs/subroutines.rst). A `readonly`
    /// one can't be told from a copy, so it is bound as one. An array passed
    /// to a `mutable` parameter is shared between the caller's register and
    /// the callee for the duration of the call; those registers are
    /// returned with their shared arrays, for the caller to take back
    /// ownership once the callee is done.
    fn bind_subroutine(
        &self,
        frame: &mut Frame,
        proc_id: ProcId,
        args: &[BcOperand],
    ) -> KindResult<(Frame, Vec<(Reg, ArrayRef)>)> {
        let proc = &self.module.procedures[proc_id.0 as usize];
        let mut child = self.fresh_frame(proc_id);
        let mut by_ref = Vec::new();
        // Qubit args fill dense qubit slots `[0, n)`; classical args go to
        // the procedure's parameter registers, both in declaration order.
        let mut qslot = 0usize;
        let mut params = proc.params.iter();
        for arg in args {
            if is_qubit_operand(arg) {
                child.slots[qslot] = self.qubits(frame, arg)?;
                qslot += 1;
                continue;
            }
            let param = *params.next().ok_or_else(|| {
                VmErrorKind::Unsupported("more classical args than parameters".into())
            })?;
            if let BcOperand::Reg(reg) = arg
                && let ValueTy::ArrayRef(ty) = proc.register_types[param.0 as usize]
                && ty.access() == RefAccess::Mutable
            {
                let held = &mut frame.regs[reg.0 as usize];
                if let Some(Value::Array(array)) = held.take_if(|v| matches!(v, Value::Array(_))) {
                    let shared = array.into_ref_mut();
                    *held = Some(Value::ArrayRef(shared.clone()));
                    by_ref.push((*reg, shared));
                }
            }
            child.regs[param.0 as usize] = Some(self.eval(frame, arg)?);
        }
        Ok((child, by_ref))
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
    fn cal_load(&mut self, symbol: SymbolId) -> KindResult<Value> {
        if let Some(v) = self.pulse_globals.get(&symbol) {
            return Ok(v.clone());
        }
        let sym = self.module.symbol(symbol);
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
    ) -> KindResult<Option<Value>> {
        let BcCalBody::OpenPulse(proc_id) = self.module.calibrations[idx as usize].body else {
            return Err(VmErrorKind::Pulse(
                "opaque defcal bodies cannot execute".into(),
            ));
        };
        let proc = &self.module.procedures[proc_id.0 as usize];
        let mut child = self.fresh_frame(proc_id);
        for (i, arg) in args.iter().enumerate() {
            let v = self.eval(frame, arg)?;
            let reg = proc.params.get(i).ok_or_else(|| {
                VmErrorKind::Unsupported("more classical args than defcal parameters".into())
            })?;
            child.regs[reg.0 as usize] = Some(v);
        }
        for (slot, q) in hw.iter().enumerate() {
            child.slots[slot] = vec![*q];
        }
        Box::pin(self.exec_proc(&mut child)).await
    }

    /// Marshal one OpenPulse intrinsic call to the typed pulse handler.
    /// Handles travel as `uint[64]` values; angles arrive in radians.
    fn call_pulse_intrinsic(&mut self, name: &str, args: &[Value]) -> KindResult<Option<Value>> {
        let pulse = self
            .pulse
            .as_mut()
            .expect("caller checks a pulse handler is installed");
        let arg = |n: usize| -> KindResult<&Value> {
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
    ) -> KindResult<()> {
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
            let arg_vals = self.eval_all(frame, args)?;
            let resolved: Vec<Vec<u32>> = arg_ops
                .iter()
                .map(|o| self.qubits(frame, o))
                .collect::<KindResult<_>>()?;
            let bcast = broadcast_len(&resolved)?;
            for j in 0..bcast {
                let mut child = self.fresh_frame(proc_id);
                let proc = &self.module.procedures[proc_id.0 as usize];
                for (k, v) in arg_vals.iter().enumerate() {
                    let reg = proc.params.get(k).ok_or_else(|| {
                        VmErrorKind::Unsupported("gate has fewer params than arguments".into())
                    })?;
                    child.regs[reg.0 as usize] = Some(v.clone());
                }
                // Bind the gate's qubit params into slots `[0, n)`.
                // Singletons repeat; registers (validated equal to `bcast`)
                // are indexed by the broadcast position.
                for (s, qs) in resolved.iter().enumerate() {
                    let idx = if qs.len() == 1 { 0 } else { j };
                    child.slots[s] = vec![qs[idx]];
                }
                if eff.power == 1.0 {
                    // No `inv`/`pow`: recurse, propagating controls. `ctrl`
                    // distributes over a sequence, so this is exact.
                    child.mods = eff.clone();
                    Box::pin(self.exec_proc(&mut child)).await?;
                } else {
                    // `inv`/`pow` on a (possibly composite) body: flatten
                    // the body to a trace of built-in calls, then
                    // reverse+invert or repeat it before emitting.
                    // Recording runs with a clean context; the outer
                    // controls are merged back in at emit time.
                    let prev = self.recording.replace(Vec::new());
                    let res = Box::pin(self.exec_proc(&mut child)).await;
                    let trace = std::mem::replace(&mut self.recording, prev).unwrap_or_default();
                    res?;
                    self.apply_transformed(&trace, eff.power, &eff)?;
                }
            }
            Ok(())
        } else {
            let name = self.module.symbol(gate).name.as_str();
            match name {
                "U" => {
                    let theta = value_f64(&self.eval(frame, &args[0])?)?;
                    let phi = value_f64(&self.eval(frame, &args[1])?)?;
                    let lambda = value_f64(&self.eval(frame, &args[2])?)?;
                    for op in arg_ops {
                        for target in self.qubits(frame, op)? {
                            self.emit(BuiltinGate::U {
                                target,
                                theta,
                                phi,
                                lambda,
                                mods: eff.clone(),
                            })?;
                        }
                    }
                    Ok(())
                }
                "gphase" => {
                    let gamma = value_f64(&self.eval(frame, &args[0])?)?;
                    self.emit(BuiltinGate::Gphase { gamma, mods: eff })
                }
                other => Err(VmErrorKind::UndefinedGate(other.to_string())),
            }
        }
    }

    /// Apply a built-in gate call to the backend — or, while a gate body is
    /// being flattened, record it instead.
    fn emit(&mut self, call: BuiltinGate) -> KindResult<()> {
        // A durationof timing pass applies no gates (uncalibrated gates
        // are zero-duration).
        if self.timing.is_some() {
            return Ok(());
        }
        match self.recording.as_mut() {
            Some(trace) => {
                trace.push(call);
                Ok(())
            }
            None => call.apply(&mut self.backend),
        }
    }

    /// Emit a flattened gate-body `trace` raised to the power `p`, nesting
    /// every call inside the outer controls `eff`.
    ///
    /// - integer `p`: repeat the trace `|p|` times (reversed + inverted
    ///   when `p < 0`). Exact, since each call is exactly invertible.
    /// - fractional `p`: only exact when at most one call acts on a qubit
    ///   (a single `U`, or a relative phase, plus commuting global
    ///   phases), where the power folds into each call. A fractional power
    ///   of a genuinely composite body would need a dense matrix power and
    ///   is rejected.
    fn apply_transformed(
        &mut self,
        trace: &[BuiltinGate],
        p: f64,
        eff: &GateModifiers,
    ) -> KindResult<()> {
        let rounded = p.round();
        if (rounded - p).abs() < 1e-9 {
            for _ in 0..(rounded as i64).unsigned_abs() {
                if rounded >= 0.0 {
                    for call in trace {
                        self.emit(call.within(eff, 1.0))?;
                    }
                } else {
                    for call in trace.iter().rev() {
                        self.emit(call.within(eff, -1.0))?;
                    }
                }
            }
            Ok(())
        } else {
            if trace.iter().filter(|c| c.acts_on_a_qubit()).count() > 1 {
                return Err(VmErrorKind::Unsupported(
                    "fractional power of a multi-qubit composite gate".into(),
                ));
            }
            trace
                .iter()
                .try_for_each(|call| self.emit(call.within(eff, p)))
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────

    fn set(&self, frame: &mut Frame, reg: Reg, value: Value) {
        frame.regs[reg.0 as usize] = Some(value);
    }

    /// Resolve a list of classical operands to values.
    fn eval_all(&self, frame: &Frame, ops: &[BcOperand]) -> KindResult<Vec<Value>> {
        ops.iter().map(|op| self.eval(frame, op)).collect()
    }

    /// Evaluate per-dimension index operands into `Index::Item`s for an
    /// element load/store (`a[i, j, …]`).
    fn eval_index_items(&self, frame: &Frame, indices: &[BcOperand]) -> KindResult<Vec<Index>> {
        indices
            .iter()
            .map(|op| Ok(Index::Item(value_isize(&self.eval(frame, op)?)?)))
            .collect()
    }

    /// Resolve a classical operand to a value.
    fn eval(&self, frame: &Frame, op: &BcOperand) -> KindResult<Value> {
        match op {
            BcOperand::Reg(r) => frame.regs[r.0 as usize]
                .clone()
                .ok_or(VmErrorKind::UnsetRegister(r.0)),
            BcOperand::Const(c) => Ok(self.module.constants[c.0 as usize].clone()),
            _ => Err(VmErrorKind::Type("expected a classical operand".into())),
        }
    }

    fn array_ty(&self, frame: &Frame, dest: Reg) -> KindResult<ArrayTy> {
        match self.module.procedures[frame.proc.0 as usize].register_types[dest.0 as usize] {
            ValueTy::Array(aty) => Ok(aty),
            _ => Err(VmErrorKind::Type(
                "NewArray destination is not an array".into(),
            )),
        }
    }

    /// Resolve a qubit operand to its global qubit indices, validating that
    /// each is within the program's allocated qubit memory.
    fn qubits(&self, frame: &Frame, op: &BcOperand) -> KindResult<Vec<u32>> {
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
                    .collect::<KindResult<Vec<_>>>()?
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
    fn source_list(&self, frame: &Frame, source: &QubitSource) -> KindResult<Vec<u32>> {
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
    fn qubit_list(&self, frame: &Frame, ops: &[BcOperand]) -> KindResult<Vec<u32>> {
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
    ) -> KindResult<Vec<u32>> {
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

    /// A fresh activation of `proc` with no modifier context: zeroed
    /// registers (callers then bind parameter registers) and empty qubit
    /// slots (callers fill the parameter slots).
    fn fresh_frame(&self, proc: ProcId) -> Frame {
        Frame {
            proc,
            regs: self.fresh_regs(proc),
            slots: self.fresh_slots(proc),
            mods: GateModifiers::none(),
        }
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
fn broadcast_len(operands: &[Vec<u32>]) -> KindResult<usize> {
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

/// Pack measurement outcomes into the measured value: a `bit` for one
/// qubit, otherwise a `bit[n]` register holding outcome `i` at bit `i`.
fn measured_value(outcomes: &[bool]) -> Value {
    if let [bit] = outcomes {
        return Value::bit(*bit);
    }
    let width = outcomes.len() as u32;
    let mut reg = BitReg::zeros(width);
    reg.set_bits(outcomes.iter().copied().enumerate());
    Value::bitreg(reg, width)
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

fn scalar(v: &Value) -> KindResult<&Primitive> {
    match v {
        Value::Scalar(s) => Ok(s.value()),
        _ => Err(VmErrorKind::Type("expected a scalar value".into())),
    }
}

fn value_bit(v: &Value) -> KindResult<bool> {
    Ok(scalar(v)?.as_bit())
}

fn value_f64(v: &Value) -> KindResult<f64> {
    scalar(v)?
        .as_float(FloatWidth::F64)
        .ok_or_else(|| VmErrorKind::Type("expected a float-like value".into()))
}

fn value_i128(v: &Value) -> KindResult<i128> {
    scalar(v)?
        .as_int(iw(128))
        .ok_or_else(|| VmErrorKind::Type("expected an integer value".into()))
}

fn value_usize(v: &Value) -> KindResult<usize> {
    scalar(v)?
        .as_uint(iw(64))
        .map(|u| u as usize)
        .ok_or_else(|| VmErrorKind::Type("expected an unsigned integer value".into()))
}

fn value_isize(v: &Value) -> KindResult<isize> {
    Ok(value_i128(v)? as isize)
}

fn value_u64(v: &Value) -> KindResult<u64> {
    scalar(v)?
        .as_uint(iw(64))
        .map(|u| u as u64)
        .ok_or_else(|| VmErrorKind::Type("expected an unsigned integer value".into()))
}

fn value_duration(v: &Value) -> KindResult<Duration> {
    scalar(v)?
        .as_duration()
        .ok_or_else(|| VmErrorKind::Type("expected a duration value".into()))
}

fn value_complex(v: &Value) -> KindResult<Complex64> {
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
