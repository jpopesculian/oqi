//! A CPU state-vector simulator backend.

use async_trait::async_trait;
use num_complex::Complex;
use num_traits::Float;
use oqi_quantum::{Gate, StateVector, Unitary};

use crate::backend::{GateModifiers, QuantumBackend};
use crate::error::{VmError, VmErrorKind};

/// A small, seedable xorshift64* PRNG. Avoids a `rand` dependency and
/// keeps runs reproducible. Shared by the CPU backends.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // xorshift requires a non-zero state.
        Rng {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// State-vector simulator over `Complex<F>` amplitudes (`F` = `f64` by
/// default; instantiate `StateVectorSim<f32>` for single precision).
///
/// Gate angles always arrive as `f64` and are cast to `F` internally;
/// measurement probabilities are accumulated in `f64` regardless of `F`
/// for numerical stability.
pub struct StateVectorSim<F = f64> {
    state: StateVector<F>,
    rng: Rng,
    /// Apply gates with the rayon-parallel kernel (no effect when the
    /// `parallel` feature is disabled).
    parallel: bool,
}

impl<F: Float + Send + Sync> StateVectorSim<F> {
    /// A fresh simulator with an explicit seed, or a
    /// [`VmErrorKind::TooManyQubits`] error if the `2^num_qubits`-amplitude
    /// state vector can't be allocated (it uses a fallible allocation, so an
    /// oversized circuit fails gracefully instead of aborting).
    pub fn try_zeroed(num_qubits: u32, seed: u64) -> std::result::Result<Self, VmError> {
        let state = StateVector::try_zero(num_qubits as usize).ok_or_else(|| {
            VmError::new(VmErrorKind::TooManyQubits {
                requested: num_qubits,
            })
        })?;
        Ok(StateVectorSim {
            state,
            rng: Rng::new(seed),
            parallel: false,
        })
    }

    /// Enable (or disable) the rayon-parallel gate kernel.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// The current amplitudes (global phase unresolved).
    pub fn state(&self) -> &[Complex<F>] {
        self.state.state()
    }

    /// Apply a gate, dispatching to the parallel kernel when enabled.
    fn apply_gate(&mut self, gate: &Gate<F>, target: usize) {
        if self.parallel {
            #[cfg(feature = "parallel")]
            {
                self.state.par_apply(gate, target);
                return;
            }
        }
        self.state.apply(gate, target);
    }

    /// Cast an `f64` gate angle to the amplitude precision `F`.
    fn cast(x: f64) -> F {
        F::from(x).expect("gate angle representable in amplitude precision")
    }

    /// Build a [`Gate`] from a unitary plus the controls/power in `m`.
    fn gate(&self, u: Unitary<F>, m: &GateModifiers) -> Gate<F> {
        let mut g = Gate::new(u);
        for &c in &m.controls {
            g = g.ctrl(c as usize);
        }
        for &c in &m.neg_controls {
            g = g.neg_ctrl(c as usize);
        }
        if m.power != 1.0 {
            g = g.pow(Self::cast(m.power));
        }
        g
    }

    fn apply_x(&mut self, target: u32) {
        // U(π, 0, π) is exactly Pauli-X (no spurious phase).
        let x = Unitary::new(
            Self::cast(std::f64::consts::PI),
            F::zero(),
            Self::cast(std::f64::consts::PI),
        );
        self.apply_gate(&Gate::new(x), target as usize);
    }

    /// Collapse `qubit` onto a *forced* `outcome` and renormalize — the
    /// projection half of [`measure`](Self::measure) with no RNG. Used to
    /// replay a recorded measurement deterministically (e.g. when the
    /// auto-routing backend converts a stabilizer state to a state vector).
    pub fn project(&mut self, qubit: u32, outcome: bool) {
        let bit = 1usize << qubit;
        let norm: f64 = self
            .state
            .state()
            .iter()
            .enumerate()
            .filter(|(i, _)| (i & bit != 0) == outcome)
            .map(|(_, a)| a.norm_sqr().to_f64().unwrap())
            .sum();
        let scale = if norm > 0.0 {
            Self::cast(1.0 / norm.sqrt())
        } else {
            F::zero()
        };
        let zero = Complex::new(F::zero(), F::zero());
        for (i, a) in self.state.state_mut().iter_mut().enumerate() {
            if (i & bit != 0) == outcome {
                *a = *a * scale;
            } else {
                *a = zero;
            }
        }
    }
}

impl StateVectorSim<f64> {
    /// A fresh `f64` simulator with `num_qubits` qubits in |0…0⟩, default seed.
    ///
    /// Panics if the state vector can't be allocated; use
    /// [`try_new`](Self::try_new) to handle that gracefully.
    pub fn new(num_qubits: u32) -> Self {
        Self::with_seed(num_qubits, 0x2545F4914F6CDD1D)
    }

    /// A fresh simulator with an explicit RNG seed (for reproducibility).
    /// Panics if the state vector can't be allocated.
    pub fn with_seed(num_qubits: u32, seed: u64) -> Self {
        Self::try_with_seed(num_qubits, seed).unwrap_or_else(|e| panic!("{e}"))
    }

    /// A fresh simulator with the default seed, or a
    /// [`VmErrorKind::TooManyQubits`] error if its state vector can't be
    /// allocated.
    pub fn try_new(num_qubits: u32) -> std::result::Result<Self, VmError> {
        Self::try_with_seed(num_qubits, 0x2545F4914F6CDD1D)
    }

    /// A fresh `f64` simulator with an explicit seed, or a
    /// [`VmErrorKind::TooManyQubits`] error if it can't be allocated.
    pub fn try_with_seed(num_qubits: u32, seed: u64) -> std::result::Result<Self, VmError> {
        Self::try_zeroed(num_qubits, seed)
    }
}

#[async_trait(?Send)]
impl<F: Float + Send + Sync> QuantumBackend for StateVectorSim<F> {
    async fn u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, m: &GateModifiers) {
        let u = Unitary::new(Self::cast(theta), Self::cast(phi), Self::cast(lambda));
        let g = self.gate(u, m);
        self.apply_gate(&g, target as usize);
    }

    async fn gphase(&mut self, gamma: f64, m: &GateModifiers) {
        // gphase scales linearly with power: gphase(γ)^k = gphase(kγ).
        let g = Self::cast(gamma * m.power);

        if m.controls.is_empty() && m.neg_controls.is_empty() {
            self.state.gphase(g);
            return;
        }

        // A controlled global phase is a relative phase. Since `gphase`
        // has no qubit of its own, the innermost control plays the role
        // of a phase gate's target: ctrlⁿ @ gphase(g) == ctrlⁿ⁻¹ @
        // U(0, 0, g) on that qubit (diag(1, e^{ig})). A negctrl target
        // wants the phase on |0⟩, so it is wrapped in X gates.
        let mut controls = m.controls.clone();
        let mut neg_controls = m.neg_controls.clone();
        let (target, neg_target) = match controls.pop() {
            Some(c) => (c, false),
            None => (neg_controls.pop().expect("at least one control"), true),
        };

        let mut gate = Gate::new(Unitary::new(F::zero(), F::zero(), g));
        for c in controls {
            gate = gate.ctrl(c as usize);
        }
        for c in neg_controls {
            gate = gate.neg_ctrl(c as usize);
        }

        if neg_target {
            self.apply_x(target);
            self.apply_gate(&gate, target as usize);
            self.apply_x(target);
        } else {
            self.apply_gate(&gate, target as usize);
        }
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        let bit = 1usize << qubit;
        let amps = self.state.state();

        // P(outcome = 1) = Σ |amp_i|² over states with this bit set.
        // Accumulated in f64 even for f32 amplitudes, for stability.
        let p_one: f64 = amps
            .iter()
            .enumerate()
            .filter(|(i, _)| i & bit != 0)
            .map(|(_, a)| a.norm_sqr().to_f64().unwrap())
            .sum();

        let outcome = self.rng.next_f64() < p_one;

        // Collapse: zero the non-matching half and renormalize.
        let norm = if outcome { p_one } else { 1.0 - p_one };
        let scale = if norm > 0.0 {
            Self::cast(1.0 / norm.sqrt())
        } else {
            // Degenerate (numerically zero) outcome: leave as-is.
            F::zero()
        };
        let zero = Complex::new(F::zero(), F::zero());
        for (i, a) in self.state.state_mut().iter_mut().enumerate() {
            if (i & bit != 0) == outcome {
                *a = *a * scale;
            } else {
                *a = zero;
            }
        }
        outcome
    }

    async fn reset(&mut self, qubit: u32) {
        if self.measure(qubit).await {
            self.apply_x(qubit);
        }
    }

    async fn reset_state(&mut self, _num_qubits: u32) {
        // Direct re-zero on the existing buffer; leaves the RNG stream intact
        // so shots stay independent and reproducible from the initial seed.
        self.state.zero_in_place();
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        // Resolve the tracked global phase so the snapshot is physically
        // faithful and matches other backends' conventions.
        let phase = Complex::from_polar(1.0, self.state.global_phase().to_f64().unwrap());
        Some(
            self.state
                .state()
                .iter()
                .map(|a| Complex::new(a.re.to_f64().unwrap(), a.im.to_f64().unwrap()) * phase)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `reset_state` returns the register to |0…0⟩ so a reused sim can start a
    /// fresh shot (the shots feature depends on this).
    #[tokio::test]
    async fn reset_state_rezeros() {
        let none = GateModifiers::none();
        let mut sim = StateVectorSim::<f64>::try_zeroed(2, 7).unwrap();
        // X on qubit 0 (U(π,0,π)) → |01>; q0 measures 1 deterministically.
        sim.u(0, std::f64::consts::PI, 0.0, std::f64::consts::PI, &none)
            .await;
        assert!(sim.measure(0).await, "X|0⟩ measures 1");
        // A fresh shot: re-zero, then both qubits measure 0.
        sim.reset_state(2).await;
        assert!(!sim.measure(0).await, "reset_state returns q0 to |0⟩");
        assert!(!sim.measure(1).await, "reset_state returns q1 to |0⟩");
    }
}
