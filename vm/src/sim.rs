//! A CPU state-vector simulator backend.

use num_complex::Complex;
use oqi_quantum::{Gate, StateVector, Unitary};

use crate::backend::{GateModifiers, QuantumBackend};

/// A small, seedable xorshift64* PRNG. Avoids a `rand` dependency and
/// keeps runs reproducible.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// State-vector simulator over `f64` amplitudes.
pub struct StateVectorSim {
    state: StateVector<f64>,
    rng: Rng,
}

impl StateVectorSim {
    /// A fresh simulator with `num_qubits` qubits in |0…0⟩, default seed.
    pub fn new(num_qubits: u32) -> Self {
        Self::with_seed(num_qubits, 0x2545F4914F6CDD1D)
    }

    /// A fresh simulator with an explicit RNG seed (for reproducibility).
    pub fn with_seed(num_qubits: u32, seed: u64) -> Self {
        StateVectorSim {
            state: StateVector::zero(num_qubits as usize),
            rng: Rng::new(seed),
        }
    }

    /// The current amplitudes (global phase unresolved).
    pub fn state(&self) -> &[Complex<f64>] {
        self.state.state()
    }

    /// Build a [`Gate`] from a unitary plus the controls/power in `m`.
    fn gate(&self, u: Unitary<f64>, m: &GateModifiers) -> Gate<f64> {
        let mut g = Gate::new(u);
        for &c in &m.controls {
            g = g.ctrl(c as usize);
        }
        for &c in &m.neg_controls {
            g = g.neg_ctrl(c as usize);
        }
        if m.power != 1.0 {
            g = g.pow(m.power);
        }
        g
    }

    fn apply_x(&mut self, target: u32) {
        // U(π, 0, π) is exactly Pauli-X (no spurious phase).
        let x = Unitary::new(std::f64::consts::PI, 0.0, std::f64::consts::PI);
        self.state.apply(&Gate::new(x), target as usize);
    }
}

impl QuantumBackend for StateVectorSim {
    fn u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, m: &GateModifiers) {
        let g = self.gate(Unitary::new(theta, phi, lambda), m);
        self.state.apply(&g, target as usize);
    }

    fn gphase(&mut self, gamma: f64, m: &GateModifiers) {
        // gphase scales linearly with power: gphase(γ)^k = gphase(kγ).
        let g = gamma * m.power;

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

        let mut gate = Gate::new(Unitary::new(0.0, 0.0, g));
        for c in controls {
            gate = gate.ctrl(c as usize);
        }
        for c in neg_controls {
            gate = gate.neg_ctrl(c as usize);
        }

        if neg_target {
            self.apply_x(target);
            self.state.apply(&gate, target as usize);
            self.apply_x(target);
        } else {
            self.state.apply(&gate, target as usize);
        }
    }

    fn measure(&mut self, qubit: u32) -> bool {
        let bit = 1usize << qubit;
        let amps = self.state.state();

        // P(outcome = 1) = Σ |amp_i|² over states with this bit set.
        let p_one: f64 = amps
            .iter()
            .enumerate()
            .filter(|(i, _)| i & bit != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum();

        let outcome = self.rng.next_f64() < p_one;

        // Collapse: zero the non-matching half and renormalize.
        let norm = if outcome { p_one } else { 1.0 - p_one };
        let scale = if norm > 0.0 {
            1.0 / norm.sqrt()
        } else {
            // Degenerate (numerically zero) outcome: leave as-is.
            0.0
        };
        for (i, a) in self.state.state_mut().iter_mut().enumerate() {
            if (i & bit != 0) == outcome {
                *a *= scale;
            } else {
                *a = Complex::new(0.0, 0.0);
            }
        }
        outcome
    }

    fn reset(&mut self, qubit: u32) {
        if self.measure(qubit) {
            self.apply_x(qubit);
        }
    }
}
