//! The pluggable quantum execution layer.
//!
//! Every gate in an OpenQASM program is, by the time the compiler is
//! done with it, expressed in terms of the two built-in primitives
//! `U(θ, φ, λ)` and `gphase(γ)` (backends ship their own `stdgates.inc`
//! that decomposes the standard library to these, or maps gates onto
//! `extern` calls). A backend therefore only needs to implement a
//! (possibly controlled) `U`, an uncontrolled `gphase`, measurement and
//! reset, and — optionally — timing.
//!
//! Implementations: [`StateVectorSim`](crate::StateVectorSim),
//! [`SimdSim`](crate::SimdSim), [`AutoSim`](crate::AutoSim),
//! [`MpsSim`](crate::MpsSim) and, with the `gpu` feature, `GpuSim`. A
//! hardware provider would implement the same trait.

use std::f64::consts::PI;

use async_trait::async_trait;
use num_complex::Complex;
use oqi_classical::Duration;

use crate::error::KindResult;
use crate::rng::Rng;

/// Modifiers on a single backend primitive call: the controls in scope
/// plus a `power` to raise this one primitive to. Control indices are
/// global qubit indices.
///
/// `power` carries `inv`/`pow` for the *individual* `U`/`gphase` reaching
/// the backend, which is always exact (`U^k` via matrix power, `gphase(γ)^k
/// = gphase(kγ)`). Composite-gate modifiers — where `inv` must reverse the
/// body and `pow` must apply to the product rather than each factor — are
/// resolved by the VM (it flattens the body and reverses/repeats the trace)
/// before any leaf reaches the backend, so a backend never sees more than a
/// per-primitive power.
#[derive(Debug, Clone)]
pub struct GateModifiers {
    pub controls: Vec<u32>,
    pub neg_controls: Vec<u32>,
    pub power: f64,
}

impl GateModifiers {
    /// No modifiers: the identity context (`power = 1`).
    pub fn none() -> Self {
        GateModifiers::ctrl(&[])
    }

    /// Positive controls on `controls`, nothing else.
    pub fn ctrl(controls: &[u32]) -> Self {
        GateModifiers {
            controls: controls.to_vec(),
            neg_controls: Vec::new(),
            power: 1.0,
        }
    }

    /// These modifiers nested inside `outer`: the control lists are
    /// concatenated and the power is scaled by `factor`.
    pub(crate) fn within(&self, outer: &GateModifiers, factor: f64) -> Self {
        GateModifiers {
            controls: [&self.controls[..], &outer.controls].concat(),
            neg_controls: [&self.neg_controls[..], &outer.neg_controls].concat(),
            power: self.power * factor,
        }
    }

    fn is_controlled(&self) -> bool {
        !self.controls.is_empty() || !self.neg_controls.is_empty()
    }
}

impl Default for GateModifiers {
    fn default() -> Self {
        GateModifiers::none()
    }
}

/// One built-in gate call with its modifiers — what the VM records while
/// flattening a gate body, and what the auto-router logs for replay.
#[derive(Debug, Clone)]
pub(crate) enum BuiltinGate {
    U {
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        mods: GateModifiers,
    },
    Gphase {
        gamma: f64,
        mods: GateModifiers,
    },
}

impl BuiltinGate {
    /// Apply this call to `backend`.
    pub(crate) fn apply<B: QuantumBackend + ?Sized>(&self, backend: &mut B) -> KindResult<()> {
        match self {
            BuiltinGate::U {
                target,
                theta,
                phi,
                lambda,
                mods,
            } => backend.u(*target, *theta, *phi, *lambda, mods),
            BuiltinGate::Gphase { gamma, mods } => apply_gphase(backend, *gamma, mods),
        }
    }

    /// Whether this acts non-trivially on a qubit: a `U`, or a *controlled*
    /// global phase (a relative phase). An uncontrolled `gphase` is a global
    /// scalar.
    pub(crate) fn acts_on_a_qubit(&self) -> bool {
        match self {
            BuiltinGate::U { .. } => true,
            BuiltinGate::Gphase { mods, .. } => mods.is_controlled(),
        }
    }

    /// This call nested inside `outer` (see [`GateModifiers::within`]).
    pub(crate) fn within(&self, outer: &GateModifiers, factor: f64) -> Self {
        match *self {
            BuiltinGate::U {
                target,
                theta,
                phi,
                lambda,
                ref mods,
            } => BuiltinGate::U {
                target,
                theta,
                phi,
                lambda,
                mods: mods.within(outer, factor),
            },
            BuiltinGate::Gphase { gamma, ref mods } => BuiltinGate::Gphase {
                gamma,
                mods: mods.within(outer, factor),
            },
        }
    }
}

/// Apply `gphase(γ)` under `mods`, which backends only implement
/// uncontrolled. A controlled global phase is a relative phase — `e^{iγ}`
/// on the states where every control is 1 and every negative control is 0 —
/// so it is lowered here, phase-exactly, to `U`:
///
/// - with a positive control `c`, the phase gate `U(0, 0, γ) = diag(1,
///   e^{iγ})` on `c` under the remaining controls;
/// - with only negative controls, the same on one of them, `n`, conjugated
///   by X so the phase lands on its |0⟩. The flips are `U(π, 0, π) = i·X`
///   and its exact inverse `U(π, π, 0) = −i·X`, so no stray phase is left.
pub(crate) fn apply_gphase<B: QuantumBackend + ?Sized>(
    backend: &mut B,
    gamma: f64,
    mods: &GateModifiers,
) -> KindResult<()> {
    // `gphase` scales linearly with power: gphase(γ)^k = gphase(kγ).
    let gamma = gamma * mods.power;
    if !mods.is_controlled() {
        return backend.gphase(gamma);
    }
    let mut rest = GateModifiers {
        power: 1.0,
        ..mods.clone()
    };
    if let Some(c) = rest.controls.pop() {
        return backend.u(c, 0.0, 0.0, gamma, &rest);
    }
    let n = rest.neg_controls.pop().expect("a controlled gphase");
    let none = GateModifiers::none();
    backend.u(n, PI, 0.0, PI, &none)?;
    backend.u(n, 0.0, 0.0, gamma, &rest)?;
    backend.u(n, PI, PI, 0.0, &none)
}

/// A quantum execution backend. The VM resolves all gates down to these
/// calls against global qubit indices.
///
/// Applying a gate is synchronous: every backend can queue or apply one
/// without waiting, and it is the VM's hot path. The operations that read
/// the quantum state back — measurement, reset and
/// [`amplitudes`](Self::amplitudes) — are `async` so a GPU/remote backend
/// can `.await` its device without blocking, which is essential in the
/// browser, where the main thread can't block. CPU backends implement them
/// as ready futures. `?Send` keeps the trait object-safe (via
/// `async_trait`'s boxed futures) and works with `!Send` web GPU types:
/// `block_on` drives the future on the calling thread, so the trait's
/// futures need not be `Send` (native callers use `pollster`; wasm uses the
/// browser event loop).
#[async_trait(?Send)]
pub trait QuantumBackend {
    /// Apply `U(θ, φ, λ)` to `target` under the given modifiers.
    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        modifiers: &GateModifiers,
    ) -> KindResult<()>;

    /// Multiply the state by the global phase `e^{iγ}`. Only ever
    /// uncontrolled: a controlled global phase reaches the backend as
    /// [`u`](Self::u) calls.
    fn gphase(&mut self, gamma: f64) -> KindResult<()>;

    /// Measure `qubit` in the Z basis, collapsing the state, and return
    /// the outcome.
    async fn measure(&mut self, qubit: u32) -> bool;

    /// Measure several qubits in the Z basis (in order), collapsing after
    /// each, returning one outcome per qubit. Equivalent to calling
    /// [`measure`](Self::measure) for each in turn (projective measurements on
    /// distinct qubits commute); the default does exactly that. Backends with
    /// a costly per-measurement path (e.g. a GPU readback) override this to
    /// snapshot the state once and sample every qubit from that copy.
    async fn measure_all(&mut self, qubits: &[u32]) -> Vec<bool> {
        let mut out = Vec::with_capacity(qubits.len());
        for &q in qubits {
            out.push(self.measure(q).await);
        }
        out
    }

    /// Reset `qubit` to |0⟩.
    async fn reset(&mut self, qubit: u32);

    /// Reset the whole register to |0…0⟩ so a fresh shot can run on the same
    /// backend, leaving the measurement RNG stream untouched so shots stay
    /// independent and reproducible from the initial seed.
    fn reset_state(&mut self) -> KindResult<()>;

    /// A scheduling barrier across `qubits`. No effect on state.
    fn barrier(&mut self, _qubits: &[u32]) {}

    /// An idle delay on `qubits`. No effect on state in the bundled
    /// simulators.
    fn delay(&mut self, _qubits: &[u32], _duration: Duration) {}

    /// Snapshot the full amplitude vector as `f64` (global phase
    /// resolved), or `None` for backends without an addressable state
    /// vector (e.g. hardware). Used for `--state` printing and tests.
    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        None
    }

    /// Whether [`sample_shots`](Self::sample_shots) can draw shots from the
    /// current state without re-running the circuit. The default holds for
    /// backends whose [`amplitudes`](Self::amplitudes) is a cheap copy of a
    /// state vector they already hold; it is `false` where materializing 2ⁿ
    /// amplitudes can cost far more than running the circuit (a stabilizer
    /// tableau, a sum over Cliffords) — shot sampling then re-runs the
    /// circuit per shot.
    fn can_sample_shots(&self) -> bool {
        true
    }

    /// Draw `shots` Z-basis samples of the whole register from the current
    /// state without collapsing it (`out[i][q]` is qubit `q` in shot `i`),
    /// with an RNG seeded by `seed`; `None` if the backend
    /// [can't](Self::can_sample_shots) or has no readable state. The
    /// default samples the [`amplitudes`](Self::amplitudes) snapshot through
    /// a cumulative |ψ|² table.
    async fn sample_shots(&mut self, shots: u32, seed: u64) -> Option<Vec<Vec<bool>>> {
        if !self.can_sample_shots() {
            return None;
        }
        let psi = self.amplitudes().await?;
        let mut cumulative = Vec::with_capacity(psi.len());
        let mut total = 0.0f64;
        for amplitude in &psi {
            total += amplitude.norm_sqr();
            cumulative.push(total);
        }
        let total = total.max(f64::MIN_POSITIVE);
        let n = psi.len().trailing_zeros();
        let mut rng = Rng::new(seed);
        let shots = (0..shots)
            .map(|_| {
                let u = rng.next_f64() * total;
                let idx = cumulative.partition_point(|&c| c < u).min(psi.len() - 1);
                (0..n).map(|q| (idx >> q) & 1 == 1).collect()
            })
            .collect();
        Some(shots)
    }
}

/// Forward a boxed backend so the VM can be driven by a runtime-selected
/// backend (`Box<dyn QuantumBackend>`) without monomorphizing per choice.
#[async_trait(?Send)]
impl QuantumBackend for Box<dyn QuantumBackend> {
    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        modifiers: &GateModifiers,
    ) -> KindResult<()> {
        (**self).u(target, theta, phi, lambda, modifiers)
    }

    fn gphase(&mut self, gamma: f64) -> KindResult<()> {
        (**self).gphase(gamma)
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        (**self).measure(qubit).await
    }

    async fn measure_all(&mut self, qubits: &[u32]) -> Vec<bool> {
        (**self).measure_all(qubits).await
    }

    async fn reset(&mut self, qubit: u32) {
        (**self).reset(qubit).await;
    }

    fn reset_state(&mut self) -> KindResult<()> {
        (**self).reset_state()
    }

    fn barrier(&mut self, qubits: &[u32]) {
        (**self).barrier(qubits);
    }

    fn delay(&mut self, qubits: &[u32], duration: Duration) {
        (**self).delay(qubits, duration);
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        (**self).amplitudes().await
    }

    fn can_sample_shots(&self) -> bool {
        (**self).can_sample_shots()
    }

    async fn sample_shots(&mut self, shots: u32, seed: u64) -> Option<Vec<Vec<bool>>> {
        (**self).sample_shots(shots, seed).await
    }
}
