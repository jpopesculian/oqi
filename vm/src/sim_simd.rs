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

use async_trait::async_trait;
use num_complex::Complex;
use num_traits::Float;
use oqi_quantum::{Gate, Unitary};
use wide::{f32x8, f64x4};

use crate::backend::{GateModifiers, QuantumBackend};
use crate::error::{VmError, VmErrorKind};
use crate::sim::Rng;

/// Copy the first `N` elements of `s` into a fixed-size array.
fn arr<T: Copy, const N: usize>(s: &[T]) -> [T; N] {
    <[T; N]>::try_from(&s[..N]).unwrap()
}

/// A float type with a vectorized single-qubit-gate kernel.
pub trait SimdLane: Float + Send + Sync + 'static {
    /// Apply the 2×2 complex matrix `m` to one block in place, updating the
    /// clear-target half (`a`) and the set-target half (`b`):
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
        impl SimdLane for $f {
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
pub struct SimdSim<F: SimdLane = f64> {
    re: Vec<F>,
    im: Vec<F>,
    global_phase: f64,
    rng: Rng,
    /// Run the SIMD kernel across blocks in parallel via rayon (no effect
    /// when the `parallel` feature is disabled).
    parallel: bool,
}

impl<F: SimdLane> SimdSim<F> {
    /// A fresh simulator with an explicit seed, or a
    /// [`VmErrorKind::TooManyQubits`] error if the `2^num_qubits`-amplitude
    /// arrays can't be allocated.
    pub fn try_zeroed(num_qubits: u32, seed: u64) -> std::result::Result<Self, VmError> {
        let too_many = || {
            VmError::new(VmErrorKind::TooManyQubits {
                requested: num_qubits,
            })
        };
        let len = 1usize.checked_shl(num_qubits).ok_or_else(too_many)?;
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
            rng: Rng::new(seed),
            parallel: false,
        })
    }

    /// Enable (or disable) running the SIMD kernel across blocks in parallel.
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

    fn apply_x(&mut self, target: usize) {
        // U(π, 0, π) is exactly Pauli-X (no spurious phase).
        let x = Unitary::new(
            Self::cast(std::f64::consts::PI),
            F::zero(),
            Self::cast(std::f64::consts::PI),
        )
        .matrix();
        self.apply_mat(&x, target, &[], &[]);
    }
}

#[async_trait(?Send)]
impl<F: SimdLane> QuantumBackend for SimdSim<F> {
    async fn u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, m: &GateModifiers) {
        let unitary = Unitary::new(Self::cast(theta), Self::cast(phi), Self::cast(lambda));
        let mut g = Gate::new(unitary);
        if m.power != 1.0 {
            g = g.pow(Self::cast(m.power));
        }
        // The matrix is the (powered) unitary; the controls are applied as a
        // subspace mask by `apply_mat`, not folded into the matrix.
        let mat = g.matrix();
        self.apply_mat(&mat, target as usize, &m.controls, &m.neg_controls);
    }

    async fn gphase(&mut self, gamma: f64, m: &GateModifiers) {
        let g = gamma * m.power;

        if m.controls.is_empty() && m.neg_controls.is_empty() {
            self.global_phase += g;
            return;
        }

        // A controlled global phase is a relative phase on the innermost
        // control: ctrlⁿ @ gphase(g) == ctrlⁿ⁻¹ @ U(0,0,g) = diag(1, e^{ig}).
        // A negctrl target wants the phase on |0⟩, so it is wrapped in X.
        let mut controls = m.controls.clone();
        let mut neg_controls = m.neg_controls.clone();
        let (target, neg_target) = match controls.pop() {
            Some(c) => (c, false),
            None => (neg_controls.pop().expect("at least one control"), true),
        };

        let mat = Unitary::new(F::zero(), F::zero(), Self::cast(g)).matrix();
        let target = target as usize;
        if neg_target {
            self.apply_x(target);
            self.apply_mat(&mat, target, &controls, &neg_controls);
            self.apply_x(target);
        } else {
            self.apply_mat(&mat, target, &controls, &neg_controls);
        }
    }

    async fn measure(&mut self, qubit: u32) -> bool {
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

        let norm = if outcome { p_one } else { 1.0 - p_one };
        let scale = if norm > 0.0 {
            Self::cast(1.0 / norm.sqrt())
        } else {
            F::zero()
        };
        for i in 0..self.re.len() {
            if (i & bit != 0) == outcome {
                self.re[i] = self.re[i] * scale;
                self.im[i] = self.im[i] * scale;
            } else {
                self.re[i] = F::zero();
                self.im[i] = F::zero();
            }
        }
        outcome
    }

    async fn reset(&mut self, qubit: u32) {
        if self.measure(qubit).await {
            self.apply_x(qubit as usize);
        }
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
    use crate::sim::StateVectorSim;

    const TOL: f64 = 1e-9;

    /// Assert two amplitude vectors match within tolerance.
    fn assert_close(a: &[Complex<f64>], b: &[Complex<f64>]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).norm() < TOL, "mismatch: {x} vs {y}");
        }
    }

    /// H on every qubit, then a rotation layer — exercises every target bit,
    /// including those below the SIMD lane width.
    async fn single_qubit_gates(b: &mut dyn QuantumBackend) {
        let h = GateModifiers::none();
        for q in 0..5 {
            b.u(
                q,
                std::f64::consts::FRAC_PI_2,
                0.0,
                std::f64::consts::PI,
                &h,
            )
            .await;
        }
        for q in 0..5 {
            b.u(q, 0.37, 0.11, 0.59, &h).await;
        }
    }

    async fn controlled_gates(b: &mut dyn QuantumBackend) {
        let plain = GateModifiers::none();
        for q in 0..4 {
            b.u(
                q,
                std::f64::consts::FRAC_PI_2,
                0.0,
                std::f64::consts::PI,
                &plain,
            )
            .await;
        }
        // CX(0->3), CX(2->1)
        let c0 = GateModifiers {
            controls: vec![0],
            neg_controls: vec![],
            power: 1.0,
        };
        b.u(3, std::f64::consts::PI, 0.0, std::f64::consts::PI, &c0)
            .await;
        let c2 = GateModifiers {
            controls: vec![2],
            neg_controls: vec![],
            power: 1.0,
        };
        b.u(1, std::f64::consts::PI, 0.0, std::f64::consts::PI, &c2)
            .await;
        // controlled gphase (relative phase) and a global phase
        b.gphase(0.7, &c0).await;
        b.gphase(0.3, &plain).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn matches_scalar_for_single_qubit_gates() {
        let mut scalar = StateVectorSim::<f64>::try_zeroed(5, 0xABCD).unwrap();
        let mut simd = SimdSim::<f64>::try_zeroed(5, 0xABCD).unwrap();
        single_qubit_gates(&mut scalar).await;
        single_qubit_gates(&mut simd).await;
        assert_close(
            &scalar.amplitudes().await.unwrap(),
            &simd.amplitudes().await.unwrap(),
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn matches_scalar_for_controlled_and_gphase() {
        let mut scalar = StateVectorSim::<f64>::try_zeroed(4, 0xABCD).unwrap();
        let mut simd = SimdSim::<f64>::try_zeroed(4, 0xABCD).unwrap();
        controlled_gates(&mut scalar).await;
        controlled_gates(&mut simd).await;
        assert_close(
            &scalar.amplitudes().await.unwrap(),
            &simd.amplitudes().await.unwrap(),
        );
    }

    /// The rayon block path must produce exactly the same state as the
    /// serial path (it partitions the same independent blocks).
    #[cfg(feature = "parallel")]
    #[tokio::test(flavor = "multi_thread")]
    async fn parallel_matches_serial() {
        async fn gates(b: &mut dyn QuantumBackend) {
            let h = GateModifiers::none();
            for q in 0..8 {
                b.u(
                    q,
                    std::f64::consts::FRAC_PI_2,
                    0.0,
                    std::f64::consts::PI,
                    &h,
                )
                .await;
            }
            for q in 0..8 {
                b.u(q, 0.21, 0.13, 0.42, &h).await;
            }
        }
        let mut serial = SimdSim::<f64>::try_zeroed(8, 0xABCD).unwrap();
        let mut parallel = SimdSim::<f64>::try_zeroed(8, 0xABCD)
            .unwrap()
            .with_parallel(true);
        gates(&mut serial).await;
        gates(&mut parallel).await;
        for (x, y) in serial
            .amplitudes()
            .await
            .unwrap()
            .iter()
            .zip(&parallel.amplitudes().await.unwrap())
        {
            assert_eq!(x, y, "parallel and serial SIMD paths diverged");
        }
    }
}
