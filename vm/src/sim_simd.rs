//! A SIMD-accelerated CPU state-vector simulator over a split
//! (struct-of-arrays) layout: real and imaginary parts live in separate
//! `Vec<F>`s so the per-gate complex arithmetic vectorizes without lane
//! shuffles.
//!
//! A single-qubit gate touches amplitude pairs that split into independent
//! contiguous blocks of length `2·target_bit`: within a block the
//! clear-target half pairs element-wise with the set-target half. The
//! vectorized kernel runs over those contiguous slices. Controlled gates
//! fall back to a scalar SoA loop that tests the control mask per element
//! (the control bit pattern is not uniform within a block).

use crate::state::{Gate, Unitary};
use async_trait::async_trait;
use num_complex::Complex;
use wide::{f32x8, f64x4};

use crate::backend::{GateModifiers, MAX_STATE_BYTES, QuantumBackend};
use crate::error::{Error, ErrorKind, Result};
use crate::measure::renorm;
use crate::rng::Rng;

/// Copy the first `N` elements of `s` into a fixed-size array.
fn arr<T: Copy, const N: usize>(s: &[T]) -> [T; N] {
    <[T; N]>::try_from(&s[..N]).unwrap()
}

/// The floating-point type of a state-vector simulator's amplitudes:
/// [`f64`] or [`f32`]. Sealed: the simulators' kernels exist for these two.
pub trait Amplitude: sealed::Lane {}

impl Amplitude for f64 {}
impl Amplitude for f32 {}

mod sealed {
    use num_complex::Complex;
    use num_traits::Float;

    /// A float type with a vectorized single-qubit-gate kernel.
    pub trait Lane: Float + Send + Sync + 'static {
        /// Apply the 2×2 complex matrix `m` to one block in place, updating
        /// the clear-target half (`a`) and the set-target half (`b`):
        /// `a' = m00·a + m01·b`, `b' = m10·a + m11·b`. The four slices have
        /// equal length; processed SIMD-wide with a scalar remainder.
        fn apply_block(
            m: &[[Complex<Self>; 2]; 2],
            a_re: &mut [Self],
            a_im: &mut [Self],
            b_re: &mut [Self],
            b_im: &mut [Self],
        );
    }
}

use sealed::Lane;

/// Scalar complex update for the four amplitudes of one pair.
macro_rules! scalar_pair {
    ($m:expr, $ar:expr, $ai:expr, $br:expr, $bi:expr) => {{
        let (ar, ai, br, bi) = ($ar, $ai, $br, $bi);
        (
            $m[0][0].re * ar - $m[0][0].im * ai + $m[0][1].re * br - $m[0][1].im * bi,
            $m[0][0].re * ai + $m[0][0].im * ar + $m[0][1].re * bi + $m[0][1].im * br,
            $m[1][0].re * ar - $m[1][0].im * ai + $m[1][1].re * br - $m[1][1].im * bi,
            $m[1][0].re * ai + $m[1][0].im * ar + $m[1][1].re * bi + $m[1][1].im * br,
        )
    }};
}

macro_rules! impl_simd_lane {
    ($f:ty, $v:ty, $lanes:expr) => {
        impl Lane for $f {
            fn apply_block(
                m: &[[Complex<$f>; 2]; 2],
                a_re: &mut [$f],
                a_im: &mut [$f],
                b_re: &mut [$f],
                b_im: &mut [$f],
            ) {
                let m00r = <$v>::splat(m[0][0].re);
                let m00i = <$v>::splat(m[0][0].im);
                let m01r = <$v>::splat(m[0][1].re);
                let m01i = <$v>::splat(m[0][1].im);
                let m10r = <$v>::splat(m[1][0].re);
                let m10i = <$v>::splat(m[1][0].im);
                let m11r = <$v>::splat(m[1][1].re);
                let m11i = <$v>::splat(m[1][1].im);

                let len = a_re.len();
                let mut k = 0;
                while k + $lanes <= len {
                    let ar = <$v>::new(arr::<$f, $lanes>(&a_re[k..]));
                    let ai = <$v>::new(arr::<$f, $lanes>(&a_im[k..]));
                    let br = <$v>::new(arr::<$f, $lanes>(&b_re[k..]));
                    let bi = <$v>::new(arr::<$f, $lanes>(&b_im[k..]));

                    let new_ar = m00r * ar - m00i * ai + m01r * br - m01i * bi;
                    let new_ai = m00r * ai + m00i * ar + m01r * bi + m01i * br;
                    let new_br = m10r * ar - m10i * ai + m11r * br - m11i * bi;
                    let new_bi = m10r * ai + m10i * ar + m11r * bi + m11i * br;

                    a_re[k..k + $lanes].copy_from_slice(&new_ar.to_array());
                    a_im[k..k + $lanes].copy_from_slice(&new_ai.to_array());
                    b_re[k..k + $lanes].copy_from_slice(&new_br.to_array());
                    b_im[k..k + $lanes].copy_from_slice(&new_bi.to_array());
                    k += $lanes;
                }
                while k < len {
                    let (nar, nai, nbr, nbi) = scalar_pair!(m, a_re[k], a_im[k], b_re[k], b_im[k]);
                    a_re[k] = nar;
                    a_im[k] = nai;
                    b_re[k] = nbr;
                    b_im[k] = nbi;
                    k += 1;
                }
            }
        }
    };
}

impl_simd_lane!(f64, f64x4, 4);
impl_simd_lane!(f32, f32x8, 8);

/// SIMD state-vector simulator (`F` = `f64` by default; use
/// `SimdSim<f32>` for single precision). Amplitudes are stored split into
/// real/imag arrays; the global phase is tracked separately and resolved
/// only when amplitudes are read out.
pub struct SimdSim<F: Amplitude = f64> {
    re: Vec<F>,
    im: Vec<F>,
    global_phase: f64,
    rng: Rng,
    /// Run the SIMD kernel across blocks in parallel via rayon.
    #[cfg(feature = "parallel")]
    parallel: bool,
}

impl<F: Amplitude> SimdSim<F> {
    /// A simulator of `num_qubits` qubits in |0…0⟩, measuring with the
    /// [`DEFAULT_SEED`](crate::DEFAULT_SEED).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::TooManyQubits`] if the `2^num_qubits` amplitudes would
    /// take more than 16 GiB or can't be allocated.
    pub fn new(num_qubits: u32) -> Result<Self> {
        let too_many = || {
            Error::new(ErrorKind::TooManyQubits {
                requested: num_qubits,
            })
        };
        let len = 1usize.checked_shl(num_qubits).ok_or_else(too_many)?;
        // Match StateVector::try_zero: an overcommitting allocator can grant the
        // reservations below for a multi-TiB pair of arrays and only OOM-kill
        // the process when `resize` faults the pages in. Refuse anything past
        // the same ceiling up front. The re/im arrays together hold
        // 2·len·size_of::<F>() bytes. `u64` math so the ceiling stays valid on
        // 32-bit targets like wasm32.
        let bytes = (len as u64)
            .checked_mul(2 * std::mem::size_of::<F>() as u64)
            .ok_or_else(too_many)?;
        if bytes > MAX_STATE_BYTES {
            return Err(too_many());
        }
        let mut re: Vec<F> = Vec::new();
        let mut im: Vec<F> = Vec::new();
        re.try_reserve_exact(len).map_err(|_| too_many())?;
        im.try_reserve_exact(len).map_err(|_| too_many())?;
        re.resize(len, F::zero());
        im.resize(len, F::zero());
        re[0] = F::one();
        Ok(SimdSim {
            re,
            im,
            global_phase: 0.0,
            rng: Rng::new(crate::DEFAULT_SEED),
            #[cfg(feature = "parallel")]
            parallel: false,
        })
    }

    /// Seed the measurement RNG: a given seed reproduces a run.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Run the SIMD kernel across blocks of the state on several threads
    /// (rayon).
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Cast an `f64` gate angle to the amplitude precision `F`.
    fn cast(x: f64) -> F {
        F::from(x).expect("gate angle representable in amplitude precision")
    }

    /// Apply a 2×2 matrix to `target`, conditioned on the control masks.
    fn apply_mat(
        &mut self,
        m: &[[Complex<F>; 2]; 2],
        target: usize,
        controls: &[u32],
        neg: &[u32],
    ) {
        let t = 1usize << target;
        let n = self.re.len();

        if controls.is_empty() && neg.is_empty() {
            let block = 2 * t;

            #[cfg(feature = "parallel")]
            if self.parallel {
                use rayon::prelude::*;
                self.re
                    .par_chunks_mut(block)
                    .zip(self.im.par_chunks_mut(block))
                    .for_each(|(re_blk, im_blk)| {
                        let (re_lo, re_hi) = re_blk.split_at_mut(t);
                        let (im_lo, im_hi) = im_blk.split_at_mut(t);
                        F::apply_block(m, re_lo, im_lo, re_hi, im_hi);
                    });
                return;
            }

            let mut base = 0;
            while base < n {
                let (re_lo, re_hi) = self.re[base..base + block].split_at_mut(t);
                let (im_lo, im_hi) = self.im[base..base + block].split_at_mut(t);
                F::apply_block(m, re_lo, im_lo, re_hi, im_hi);
                base += block;
            }
            return;
        }

        for i in 0..n {
            if i & t != 0 {
                continue;
            }
            if controls.iter().any(|&c| i & (1usize << c) == 0) {
                continue;
            }
            if neg.iter().any(|&c| i & (1usize << c) != 0) {
                continue;
            }
            let j = i | t;
            let (nar, nai, nbr, nbi) =
                scalar_pair!(m, self.re[i], self.im[i], self.re[j], self.im[j]);
            self.re[i] = nar;
            self.im[i] = nai;
            self.re[j] = nbr;
            self.im[j] = nbi;
        }
    }
}

impl<F: Amplitude> std::fmt::Debug for SimdSim<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("SimdSim");
        debug
            .field("num_qubits", &self.re.len().trailing_zeros())
            .field("precision", &std::any::type_name::<F>());
        #[cfg(feature = "parallel")]
        debug.field("parallel", &self.parallel);
        debug.finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl<F: Amplitude> QuantumBackend for SimdSim<F> {
    fn num_qubits(&self) -> u32 {
        self.re.len().trailing_zeros()
    }

    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
    ) -> Result<(), ErrorKind> {
        let unitary = Unitary::new(Self::cast(theta), Self::cast(phi), Self::cast(lambda));
        let mut g = Gate::new(unitary);
        if m.power != 1.0 {
            g = g.pow(Self::cast(m.power));
        }
        // The matrix is the (powered) unitary; the controls are applied as a
        // subspace mask by `apply_mat`, not folded into the matrix.
        let mat = g.matrix();
        self.apply_mat(&mat, target as usize, &m.controls, &m.neg_controls);
        Ok(())
    }

    fn gphase(&mut self, gamma: f64) -> Result<(), ErrorKind> {
        self.global_phase += gamma;
        Ok(())
    }

    async fn measure(&mut self, qubit: u32) -> Result<bool, ErrorKind> {
        let bit = 1usize << qubit;

        // P(outcome = 1) accumulated in f64 even for f32 amplitudes.
        let p_one: f64 = (0..self.re.len())
            .filter(|i| i & bit != 0)
            .map(|i| {
                let r = self.re[i].to_f64().unwrap();
                let m = self.im[i].to_f64().unwrap();
                r * r + m * m
            })
            .sum();

        let outcome = self.rng.next_f64() < p_one;
        let scale = Self::cast(renorm(p_one, outcome));
        for i in 0..self.re.len() {
            if (i & bit != 0) == outcome {
                self.re[i] = self.re[i] * scale;
                self.im[i] = self.im[i] * scale;
            } else {
                self.re[i] = F::zero();
                self.im[i] = F::zero();
            }
        }
        Ok(outcome)
    }

    async fn reset(&mut self, qubit: u32) -> Result<(), ErrorKind> {
        if self.measure(qubit).await? {
            // U(π, 0, π) is i·X: Pauli-X up to a global phase of i.
            let pi = Self::cast(std::f64::consts::PI);
            let x = Unitary::new(pi, F::zero(), pi).matrix();
            self.apply_mat(&x, qubit as usize, &[], &[]);
        }
        Ok(())
    }

    fn reset_state(&mut self) -> Result<(), ErrorKind> {
        self.re.fill(F::zero());
        self.im.fill(F::zero());
        self.re[0] = F::one();
        self.global_phase = 0.0;
        Ok(())
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        let phase = Complex::from_polar(1.0, self.global_phase);
        Some(
            self.re
                .iter()
                .zip(&self.im)
                .map(|(r, i)| Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap()) * phase)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::apply_gphase;
    use crate::sim::StateVectorSim;
    use std::f64::consts::{FRAC_PI_2, PI};

    const TOL: f64 = 1e-9;

    /// Assert two amplitude vectors match within tolerance.
    fn assert_close(a: &[Complex<f64>], b: &[Complex<f64>]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).norm() < TOL, "mismatch: {x} vs {y}");
        }
    }

    /// H on each of `n` qubits, then a rotation layer — exercises every
    /// target bit, including those below the SIMD lane width.
    fn single_qubit_gates(b: &mut dyn QuantumBackend, n: u32, rotation: (f64, f64, f64)) {
        let none = GateModifiers::none();
        for q in 0..n {
            b.u(q, FRAC_PI_2, 0.0, PI, &none).unwrap();
        }
        let (theta, phi, lambda) = rotation;
        for q in 0..n {
            b.u(q, theta, phi, lambda, &none).unwrap();
        }
    }

    fn controlled_gates(b: &mut dyn QuantumBackend) {
        let plain = GateModifiers::none();
        for q in 0..4 {
            b.u(q, FRAC_PI_2, 0.0, PI, &plain).unwrap();
        }
        // CX(0->3), CX(2->1)
        let c0 = GateModifiers::ctrl(&[0]);
        b.u(3, PI, 0.0, PI, &c0).unwrap();
        b.u(1, PI, 0.0, PI, &GateModifiers::ctrl(&[2])).unwrap();
        // controlled gphase (relative phase) and a global phase
        apply_gphase(b, 0.7, &c0).unwrap();
        b.gphase(0.3).unwrap();
    }

    #[test]
    fn matches_scalar_for_single_qubit_gates() {
        pollster::block_on(async {
            let mut scalar = StateVectorSim::<f64>::new(5)
                .map(|s| s.with_seed(0xABCD))
                .unwrap();
            let mut simd = SimdSim::<f64>::new(5).map(|s| s.with_seed(0xABCD)).unwrap();
            single_qubit_gates(&mut scalar, 5, (0.37, 0.11, 0.59));
            single_qubit_gates(&mut simd, 5, (0.37, 0.11, 0.59));
            assert_close(
                &scalar.amplitudes().await.unwrap(),
                &simd.amplitudes().await.unwrap(),
            );
        })
    }

    #[test]
    fn matches_scalar_for_controlled_and_gphase() {
        pollster::block_on(async {
            let mut scalar = StateVectorSim::<f64>::new(4)
                .map(|s| s.with_seed(0xABCD))
                .unwrap();
            let mut simd = SimdSim::<f64>::new(4).map(|s| s.with_seed(0xABCD)).unwrap();
            controlled_gates(&mut scalar);
            controlled_gates(&mut simd);
            assert_close(
                &scalar.amplitudes().await.unwrap(),
                &simd.amplitudes().await.unwrap(),
            );
        })
    }

    /// `reset_state` returns to exactly |0…0⟩, tracked global phase included.
    #[test]
    fn reset_state_clears_the_global_phase() {
        pollster::block_on(async {
            let mut simd = SimdSim::<f64>::new(2).map(|s| s.with_seed(1)).unwrap();
            single_qubit_gates(&mut simd, 2, (0.37, 0.11, 0.59));
            simd.gphase(0.3).unwrap();
            simd.reset_state().unwrap();
            let amps = simd.amplitudes().await.unwrap();
            let mut zero = vec![Complex::new(0.0, 0.0); 4];
            zero[0] = Complex::new(1.0, 0.0);
            assert_close(&amps, &zero);
        })
    }

    /// The rayon block path must produce exactly the same state as the
    /// serial path (it partitions the same independent blocks).
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial() {
        pollster::block_on(async {
            let mut serial = SimdSim::<f64>::new(8).map(|s| s.with_seed(0xABCD)).unwrap();
            let mut parallel = SimdSim::<f64>::new(8)
                .map(|s| s.with_seed(0xABCD))
                .unwrap()
                .with_parallel(true);
            single_qubit_gates(&mut serial, 8, (0.21, 0.13, 0.42));
            single_qubit_gates(&mut parallel, 8, (0.21, 0.13, 0.42));
            for (x, y) in serial
                .amplitudes()
                .await
                .unwrap()
                .iter()
                .zip(&parallel.amplitudes().await.unwrap())
            {
                assert_eq!(x, y, "parallel and serial SIMD paths diverged");
            }
        })
    }
}
