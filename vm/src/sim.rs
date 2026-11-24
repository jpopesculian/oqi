//! A CPU state-vector simulator backend.

use crate::state::{Gate, StateVector, Unitary};
use async_trait::async_trait;
use num_complex::Complex;

use crate::backend::{GateModifiers, QuantumBackend};
use crate::error::{Error, ErrorKind, Result};
use crate::measure;
use crate::rng::Rng;
use crate::sim_simd::Amplitude;

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

impl<F: Amplitude> StateVectorSim<F> {
    /// A simulator of `num_qubits` qubits in |0…0⟩, measuring with the
    /// [`DEFAULT_SEED`](crate::DEFAULT_SEED).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::TooManyQubits`] if the `2^num_qubits` amplitudes would
    /// take more than 16 GiB or can't be allocated (the allocation is
    /// fallible, so an oversized program fails instead of aborting).
    pub fn new(num_qubits: u32) -> Result<Self> {
        let state = StateVector::try_zero(num_qubits as usize).ok_or_else(|| {
            Error::new(ErrorKind::TooManyQubits {
                requested: num_qubits,
            })
        })?;
        Ok(StateVectorSim {
            state,
            rng: Rng::new(crate::DEFAULT_SEED),
            parallel: false,
        })
    }

    /// Seed the measurement RNG: a given seed reproduces a run.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Apply gates with the multi-threaded (rayon) kernel.
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Continue an existing measurement RNG stream instead of the seed's.
    pub(crate) fn set_rng(&mut self, rng: Rng) {
        self.rng = rng;
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
    pub(crate) fn project(&mut self, qubit: u32, outcome: bool) {
        measure::project(self.state.state_mut(), qubit, outcome);
    }
}

impl<F: Amplitude> std::fmt::Debug for StateVectorSim<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateVectorSim")
            .field("num_qubits", &self.state.size())
            .field("precision", &std::any::type_name::<F>())
            .field("parallel", &self.parallel)
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl<F: Amplitude> QuantumBackend for StateVectorSim<F> {
    fn num_qubits(&self) -> u32 {
        self.state.size() as u32
    }

    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
    ) -> Result<(), ErrorKind> {
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

    fn gphase(&mut self, gamma: f64) -> Result<(), ErrorKind> {
        self.state.gphase(Self::cast(gamma));
        Ok(())
    }

    async fn measure(&mut self, qubit: u32) -> Result<bool, ErrorKind> {
        Ok(measure::measure(
            self.state.state_mut(),
            qubit,
            &mut self.rng,
        ))
    }

    async fn reset(&mut self, qubit: u32) -> Result<(), ErrorKind> {
        if self.measure(qubit).await? {
            // U(π, 0, π) is i·X: Pauli-X up to a global phase of i.
            let x = Unitary::new(
                Self::cast(std::f64::consts::PI),
                F::zero(),
                Self::cast(std::f64::consts::PI),
            );
            self.apply_gate(&Gate::new(x), qubit as usize);
        }
        Ok(())
    }

    fn reset_state(&mut self) -> Result<(), ErrorKind> {
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
            let mut sim = StateVectorSim::<f64>::new(2)
                .map(|s| s.with_seed(7))
                .unwrap();
            // X on qubit 0 (U(π,0,π)) → |01>; q0 measures 1 deterministically.
            sim.u(0, PI, 0.0, PI, &none).unwrap();
            assert!(sim.measure(0).await.unwrap(), "X|0⟩ measures 1");
            // A fresh shot: re-zero, then both qubits measure 0.
            sim.reset_state().unwrap();
            assert!(
                !sim.measure(0).await.unwrap(),
                "reset_state returns q0 to |0⟩"
            );
            assert!(
                !sim.measure(1).await.unwrap(),
                "reset_state returns q1 to |0⟩"
            );
        })
    }

    /// `measure_all` returns the register's bits in order.
    #[test]
    fn measure_all_reads_register() {
        pollster::block_on(async {
            let none = GateModifiers::none();
            let mut sim = StateVectorSim::<f64>::new(3)
                .map(|s| s.with_seed(1))
                .unwrap();
            // X on q0 and q2 → |q2 q1 q0> = |101>.
            sim.u(0, PI, 0.0, PI, &none).unwrap();
            sim.u(2, PI, 0.0, PI, &none).unwrap();
            assert_eq!(
                sim.measure_all(&[0, 1, 2]).await.unwrap(),
                vec![true, false, true]
            );
        })
    }

    /// `measure_all` samples the joint distribution: a Bell pair's two qubits
    /// must agree (a naive independent sample from one snapshot would not).
    #[test]
    fn measure_all_bell_is_correlated() {
        pollster::block_on(async {
            for seed in 0..8u64 {
                let mut sim = StateVectorSim::<f64>::new(2)
                    .map(|s| s.with_seed(seed))
                    .unwrap();
                sim.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
                    .unwrap(); // H q0
                sim.u(1, PI, 0.0, PI, &GateModifiers::ctrl(&[0])).unwrap(); // CX 0->1
                let out = sim.measure_all(&[0, 1]).await.unwrap();
                assert_eq!(out[0], out[1], "Bell measurement must agree (seed {seed})");
            }
        })
    }
}
