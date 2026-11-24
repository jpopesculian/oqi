//! The pluggable quantum execution layer.
//!
//! Every gate in an OpenQASM program is, by the time the VM runs it,
//! expressed in terms of the two built-in primitives `U(θ, φ, λ)` and
//! `gphase(γ)`: the standard library (`stdgates.inc`) defines its gates in
//! those terms. A backend therefore only needs to implement a (possibly
//! controlled) `U`, an uncontrolled `gphase`, measurement and reset.

use std::f64::consts::PI;

use async_trait::async_trait;
use num_complex::Complex;
use oqi_classical::Duration;

use crate::error::{ErrorKind, Result};
use crate::rng::Rng;

/// The 2×2 matrix of the built-in `U(θ, φ, λ)` raised to `power`, rows
/// first (`m[row][column]`). `U` carries the global phase `e^{i(φ+λ)/2}`
/// of docs/gates.rst, so `U(π, 0, π)` is `i·X`; a fractional power takes
/// the principal branch.
///
/// A [`QuantumBackend`] that applies matrices honours
/// [`GateModifiers::power`] with it.
pub fn u_matrix(theta: f64, phi: f64, lambda: f64, power: f64) -> [[Complex<f64>; 2]; 2] {
    crate::clifford::effective_matrix(theta, phi, lambda, power)
}

/// The most memory a CPU backend commits to one quantum state: 16 GiB, about
/// 30 qubits of `f64` amplitudes. A larger request is refused up front with
/// [`ErrorKind::TooManyQubits`](crate::ErrorKind::TooManyQubits)
/// rather than risk an overcommitting allocator killing the process later.
pub(crate) const MAX_STATE_BYTES: u64 = 1 << 34;

/// Modifiers on a single backend primitive call: the controls in scope
/// plus a `power` to raise this one primitive to. Control indices are
/// global qubit indices; a control and the target are always distinct
/// qubits.
///
/// `power` carries `inv`/`pow` for the *individual* `U`/`gphase` reaching
/// the backend, which is always exact (`U^k` via matrix power, `gphase(γ)^k
/// = gphase(kγ)`). Composite-gate modifiers — where `inv` must reverse the
/// body and `pow` must apply to the product rather than each factor — are
/// resolved by the VM (it flattens the body and reverses/repeats the trace)
/// before any leaf reaches the backend, so a backend never sees more than a
/// per-primitive power.
#[derive(Debug, Clone, PartialEq)]
pub struct GateModifiers {
    /// Qubits that must be |1⟩ for the gate to act.
    pub controls: Vec<u32>,
    /// Qubits that must be |0⟩ for the gate to act.
    pub neg_controls: Vec<u32>,
    /// The power to raise the primitive to: `1` normally, `-1` under `inv`,
    /// any real under `pow` (see [`u_matrix`]).
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
    /// Fails if a qubit is both a control and the target, or a control
    /// twice: no gate acts on one qubit through two of its operands.
    pub(crate) fn check_distinct_qubits(&self) -> Result<(), ErrorKind> {
        let (target, mods) = match self {
            BuiltinGate::U { target, mods, .. } => (Some(*target), mods),
            BuiltinGate::Gphase { mods, .. } => (None, mods),
        };
        let mut seen = std::collections::HashSet::new();
        let qubits = target.into_iter().chain(mods.controls.iter().copied());
        for qubit in qubits.chain(mods.neg_controls.iter().copied()) {
            if !seen.insert(qubit) {
                return Err(ErrorKind::Type(format!(
                    "qubit {qubit} is used by more than one operand of a gate"
                )));
            }
        }
        Ok(())
    }

    /// Apply this call to `backend`.
    pub(crate) fn apply<B: QuantumBackend + ?Sized>(
        &self,
        backend: &mut B,
    ) -> Result<(), ErrorKind> {
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
) -> Result<(), ErrorKind> {
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
/// calls against global qubit indices `0..num_qubits`.
///
/// Applying a gate is synchronous: every backend can queue or apply one
/// without waiting, and it is the VM's hot path. The operations that read
/// the quantum state back — measurement, reset and
/// [`amplitudes`](Self::amplitudes) — are `async` so a GPU or remote
/// backend can await its device without blocking, which in the browser,
/// where the main thread can't block, is essential. CPU backends implement
/// them as ready futures.
///
/// The async methods use the [`async_trait`](crate::async_trait) macro
/// (re-exported here) with `?Send` — futures need not be `Send`, since the
/// VM is driven on the calling thread — so an implementation is annotated
/// `#[oqi_vm::async_trait(?Send)]`:
///
/// ```
/// use oqi_vm::{ErrorKind, GateModifiers, QuantumBackend, async_trait};
///
/// /// A backend for programs that never leave |0…0⟩: it accepts only
/// /// identity gates.
/// struct Zeros(u32);
///
/// #[async_trait(?Send)]
/// impl QuantumBackend for Zeros {
///     fn num_qubits(&self) -> u32 {
///         self.0
///     }
///     fn u(
///         &mut self,
///         _target: u32,
///         theta: f64,
///         phi: f64,
///         lambda: f64,
///         _modifiers: &GateModifiers,
///     ) -> Result<(), ErrorKind> {
///         if theta == 0.0 && phi + lambda == 0.0 {
///             Ok(())
///         } else {
///             Err(ErrorKind::Backend("only identity gates".into()))
///         }
///     }
///     fn gphase(&mut self, _gamma: f64) -> Result<(), ErrorKind> {
///         Ok(())
///     }
///     async fn measure(&mut self, _qubit: u32) -> Result<bool, ErrorKind> {
///         Ok(false)
///     }
///     async fn reset(&mut self, _qubit: u32) -> Result<(), ErrorKind> {
///         Ok(())
///     }
///     fn reset_state(&mut self) -> Result<(), ErrorKind> {
///         Ok(())
///     }
/// }
///
/// let program = oqi_compile::compile("qubit q; bit c = measure q;")?;
/// let mut vm = oqi_vm::Vm::new(&program, Zeros(1), oqi_vm::NoExterns);
/// let result = pollster::block_on(vm.run())?;
/// assert_eq!(result.output("c").unwrap().to_string(), "0");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[async_trait(?Send)]
pub trait QuantumBackend {
    /// The number of qubits the backend holds, indexed `0..num_qubits`. The
    /// VM refuses to run a program that needs more.
    fn num_qubits(&self) -> u32;

    /// Apply `U(θ, φ, λ)` to `target` under the given modifiers.
    ///
    /// # Errors
    ///
    /// Any [`ErrorKind`] (usually [`ErrorKind::Backend`]); it stops the run.
    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        modifiers: &GateModifiers,
    ) -> Result<(), ErrorKind>;

    /// Multiply the state by the global phase `e^{iγ}`. Only ever
    /// uncontrolled: a controlled global phase reaches the backend as
    /// [`u`](Self::u) calls.
    ///
    /// # Errors
    ///
    /// As for [`u`](Self::u).
    fn gphase(&mut self, gamma: f64) -> Result<(), ErrorKind>;

    /// Measure `qubit` in the Z basis, collapsing the state, and return
    /// the outcome.
    async fn measure(&mut self, qubit: u32) -> Result<bool, ErrorKind>;

    /// Measure several qubits in the Z basis (in order), collapsing after
    /// each, returning one outcome per qubit. Equivalent to calling
    /// [`measure`](Self::measure) for each in turn (projective measurements on
    /// distinct qubits commute); the default does exactly that. Backends with
    /// a costly per-measurement path (e.g. a remote device) override this to
    /// snapshot the state once and sample every qubit from that copy.
    async fn measure_all(&mut self, qubits: &[u32]) -> Result<Vec<bool>, ErrorKind> {
        let mut out = Vec::with_capacity(qubits.len());
        for &q in qubits {
            out.push(self.measure(q).await?);
        }
        Ok(out)
    }

    /// Reset `qubit` to |0⟩.
    async fn reset(&mut self, qubit: u32) -> Result<(), ErrorKind>;

    /// Reset the whole register to |0…0⟩ so another run can use the same
    /// backend, leaving the measurement RNG stream untouched so runs stay
    /// independent and reproducible from the initial seed. The VM calls it
    /// before every run but the first.
    ///
    /// # Errors
    ///
    /// As for [`u`](Self::u).
    fn reset_state(&mut self) -> Result<(), ErrorKind>;

    /// A scheduling barrier across `qubits`. No effect on state.
    fn barrier(&mut self, _qubits: &[u32]) {}

    /// An idle delay on `qubits`. No effect on state in the bundled
    /// simulators.
    fn delay(&mut self, _qubits: &[u32], _duration: Duration) {}

    /// Snapshot the full amplitude vector as `f64` (global phase
    /// resolved; amplitude `i` is basis state `i`, qubit `q` being bit `q`),
    /// or `None` for backends without one at hand (hardware, or a state too
    /// large to write out).
    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        None
    }

    /// Whether [`sample_shots`](Self::sample_shots) is a cheap way to draw
    /// many shots from the current state, compared with re-running the
    /// circuit for each. When it is (the default), [`Vm::sample`] runs a
    /// circuit whose quantum part doesn't depend on measurement outcomes
    /// once and draws every shot from its final state. Return `false` where
    /// drawing would cost more than re-running, e.g. when it needs all 2ⁿ
    /// amplitudes of a state held more compactly (a stabilizer tableau, a
    /// sum over Cliffords).
    ///
    /// [`Vm::sample`]: crate::Vm::sample
    fn can_sample_shots(&self) -> bool {
        true
    }

    /// Draw `shots` Z-basis samples of the whole register from the current
    /// state without collapsing it (`out[i][q]` is qubit `q` in shot `i`),
    /// with an RNG seeded by `seed`; `None` if the backend
    /// [can't](Self::can_sample_shots) or has no readable state. The
    /// default samples the [`amplitudes`](Self::amplitudes) snapshot through
    /// a cumulative |ψ|² table.
    async fn sample_shots(
        &mut self,
        shots: u32,
        seed: u64,
    ) -> Result<Option<Vec<Vec<bool>>>, ErrorKind> {
        if !self.can_sample_shots() {
            return Ok(None);
        }
        let Some(psi) = self.amplitudes().await else {
            return Ok(None);
        };
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
        Ok(Some(shots))
    }
}

/// Forward every method to the backend `$target` points to.
macro_rules! forward_backend {
    ($ty:ty) => {
        #[async_trait(?Send)]
        impl<B: QuantumBackend + ?Sized> QuantumBackend for $ty {
            fn num_qubits(&self) -> u32 {
                (**self).num_qubits()
            }

            fn u(
                &mut self,
                target: u32,
                theta: f64,
                phi: f64,
                lambda: f64,
                modifiers: &GateModifiers,
            ) -> Result<(), ErrorKind> {
                (**self).u(target, theta, phi, lambda, modifiers)
            }

            fn gphase(&mut self, gamma: f64) -> Result<(), ErrorKind> {
                (**self).gphase(gamma)
            }

            async fn measure(&mut self, qubit: u32) -> Result<bool, ErrorKind> {
                (**self).measure(qubit).await
            }

            async fn measure_all(&mut self, qubits: &[u32]) -> Result<Vec<bool>, ErrorKind> {
                (**self).measure_all(qubits).await
            }

            async fn reset(&mut self, qubit: u32) -> Result<(), ErrorKind> {
                (**self).reset(qubit).await
            }

            fn reset_state(&mut self) -> Result<(), ErrorKind> {
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

            async fn sample_shots(
                &mut self,
                shots: u32,
                seed: u64,
            ) -> Result<Option<Vec<Vec<bool>>>, ErrorKind> {
                (**self).sample_shots(shots, seed).await
            }
        }
    };
}

// A boxed backend lets the VM run a backend chosen at run time
// (`Box<dyn QuantumBackend>`, as `BackendSpec::build` makes); a borrowed one
// leaves the backend with the caller after the VM is done.
forward_backend!(Box<B>);
forward_backend!(&mut B);
