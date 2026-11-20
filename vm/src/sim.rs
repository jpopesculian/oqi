//! A CPU state-vector simulator backend.

use async_trait::async_trait;
use num_complex::Complex;
use num_traits::Float;
use oqi_quantum::{Gate, StateVector, Unitary};

use crate::backend::{GateModifiers, QuantumBackend};
use crate::error::{KindResult, VmError, VmErrorKind, VmResult};
use crate::measure;
use crate::rng::Rng;

/// State-vector simulator over `Complex<F>` amplitudes (`F` = `f64` by
/// default; instantiate `StateVectorSim<f32>` for single precision).
///
/// Gate angles always arrive as `f64` and are cast to `F` internally.
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
    pub fn try_zeroed(num_qubits: u32, seed: u64) -> VmResult<Self> {
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

    /// Enable (or disable) the rayon-parallel gate kernel. Without the
    /// `parallel` build feature there is no such kernel and gates run
    /// serially either way; [`BackendSpec`](crate::BackendSpec) reports that
    /// as an unavailable backend instead.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Continue an existing measurement RNG stream instead of the seed's.
    pub(crate) fn with_rng(mut self, rng: Rng) -> Self {
        self.rng = rng;
        self
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

    /// Collapse `qubit` onto a *forced* `outcome` and renormalize — a
    /// measurement with no RNG. Used to replay a recorded measurement
    /// deterministically (e.g. when the auto-routing backend converts a
    /// stabilizer state to a state vector).
    pub fn project(&mut self, qubit: u32, outcome: bool) {
        measure::project(self.state.state_mut(), qubit, outcome);
    }
}

#[async_trait(?Send)]
impl<F: Float + Send + Sync> QuantumBackend for StateVectorSim<F> {
    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
    ) -> KindResult<()> {
        let mut g = Gate::new(Unitary::new(
            Self::cast(theta),
            Self::cast(phi),
            Self::cast(lambda),
        ));
        for &c in &m.controls {
            g = g.ctrl(c as usize);
        }
        for &c in &m.neg_controls {
            g = g.neg_ctrl(c as usize);
        }
        if m.power != 1.0 {
            g = g.pow(Self::cast(m.power));
        }
        self.apply_gate(&g, target as usize);
        Ok(())
    }

    fn gphase(&mut self, gamma: f64) -> KindResult<()> {
        self.state.gphase(Self::cast(gamma));
        Ok(())
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        measure::measure(self.state.state_mut(), qubit, &mut self.rng)
    }

    async fn reset(&mut self, qubit: u32) {
        if self.measure(qubit).await {
            // U(π, 0, π) is i·X: Pauli-X up to a global phase of i.
            let x = Unitary::new(
                Self::cast(std::f64::consts::PI),
                F::zero(),
                Self::cast(std::f64::consts::PI),
            );
            self.apply_gate(&Gate::new(x), qubit as usize);
        }
    }

    fn reset_state(&mut self) -> KindResult<()> {
        // Direct re-zero on the existing buffer.
        self.state.zero_in_place();
        Ok(())
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
    use std::f64::consts::{FRAC_PI_2, PI};

    /// `reset_state` returns the register to |0…0⟩ so a reused sim can start a
    /// fresh shot (the shots feature depends on this).
    #[test]
    fn reset_state_rezeros() {
        pollster::block_on(async {
            let none = GateModifiers::none();
            let mut sim = StateVectorSim::<f64>::try_zeroed(2, 7).unwrap();
            // X on qubit 0 (U(π,0,π)) → |01>; q0 measures 1 deterministically.
            sim.u(0, PI, 0.0, PI, &none).unwrap();
            assert!(sim.measure(0).await, "X|0⟩ measures 1");
            // A fresh shot: re-zero, then both qubits measure 0.
            sim.reset_state().unwrap();
            assert!(!sim.measure(0).await, "reset_state returns q0 to |0⟩");
            assert!(!sim.measure(1).await, "reset_state returns q1 to |0⟩");
        })
    }

    /// `measure_all` returns the register's bits in order.
    #[test]
    fn measure_all_reads_register() {
        pollster::block_on(async {
            let none = GateModifiers::none();
            let mut sim = StateVectorSim::<f64>::try_zeroed(3, 1).unwrap();
            // X on q0 and q2 → |q2 q1 q0> = |101>.
            sim.u(0, PI, 0.0, PI, &none).unwrap();
            sim.u(2, PI, 0.0, PI, &none).unwrap();
            assert_eq!(sim.measure_all(&[0, 1, 2]).await, vec![true, false, true]);
        })
    }

    /// `measure_all` samples the joint distribution: a Bell pair's two qubits
    /// must agree (a naive independent sample from one snapshot would not).
    #[test]
    fn measure_all_bell_is_correlated() {
        pollster::block_on(async {
            for seed in 0..8u64 {
                let mut sim = StateVectorSim::<f64>::try_zeroed(2, seed).unwrap();
                sim.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
                    .unwrap(); // H q0
                sim.u(1, PI, 0.0, PI, &GateModifiers::ctrl(&[0])).unwrap(); // CX 0->1
                let out = sim.measure_all(&[0, 1]).await;
                assert_eq!(out[0], out[1], "Bell measurement must agree (seed {seed})");
            }
        })
    }
}
