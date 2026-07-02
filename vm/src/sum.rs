//! A sum-over-Cliffords state: `Σ_i |φ_i⟩` with each term a phase-exact
//! CH-form stabilizer state ([`ChForm`]) whose ω carries its coefficient.
//!
//! Clifford gates apply to every term without growing the sum; each
//! non-Clifford gate is applied as an *exact* linear combination of
//! Clifford operations, branching every term (worst case ×2 per
//! non-Clifford rotation, ×3 per controlled-diagonal factor). The cost is
//! therefore polynomial in the qubit count and exponential only in the
//! number of non-Clifford gates — the extended-stabilizer regime of
//! Bravyi et al., Quantum 3, 181 (2019).
//!
//! The decompositions used (each exact):
//!
//! - `diag(1, e^{iδ}) = c_I·I + c_S·S` with `c_S = (e^{iδ}−1)/(i−1)`,
//!   after peeling `S^k` so δ ∈ [0, π/2).
//! - `Rz(θ) = e^{−iθ/2}·diag(1, e^{iθ})`; `Ry(θ) = (SH)·Rz(θ)·(SH)†`;
//!   generic `U = e^{iα}·Rz(β)·Ry(γ)·Rz(δ)` (ZYZ, Clifford factors free).
//! - `C^k(diag(d₀,d₁)) = I + (d₀−1)·Π_{C,t=0} + (d₁−1)·Π_{C,t=1}` with
//!   `Π` a projector applied term-wise via [`ChForm::project_z`].
//! - `C^k(X) = H_t · C^k(diag(1,−1)) · H_t`; anti-diagonal `T = diag(b,c)·X`.
//! - Negative controls are X-conjugated into positive ones (exact, free).

use num_complex::Complex;

use crate::backend::GateModifiers;
use crate::ch::{ChForm, Projection};
use crate::clifford::{
    CliffordSink, M2, apply_clifford_gphase, apply_clifford_u, effective_matrix,
};
use crate::sim::Rng;

const TOL: f64 = 1e-6;
/// Terms with |ω| below this are numerically dead and pruned after
/// measurement renormalization (the state has unit norm there, so an
/// absolute threshold is safe).
const PRUNE: f64 = 1e-12;

pub(crate) struct SumState {
    n: usize,
    terms: Vec<ChForm>,
}

/// The ZYZ angles of `mat = e^{iα}·Rz(β)·Ry(γ)·Rz(δ)`.
fn zyz(mat: &M2) -> (f64, f64, f64, f64) {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    let alpha = det.arg() / 2.0;
    let inv = Complex::from_polar(1.0, -alpha);
    let v00 = mat[0][0] * inv;
    let v10 = mat[1][0] * inv;
    let v11 = mat[1][1] * inv;
    if v10.norm() < TOL {
        (alpha, 2.0 * v11.arg(), 0.0, 0.0)
    } else if v00.norm() < TOL {
        (alpha, 2.0 * v10.arg(), std::f64::consts::PI, 0.0)
    } else {
        let gamma = 2.0 * v10.norm().atan2(v00.norm());
        (alpha, v11.arg() + v10.arg(), gamma, v11.arg() - v10.arg())
    }
}

/// Split a phase angle into `(k, residual)` with `angle ≡ k·π/2 + residual
/// (mod 2π)`, `k ∈ 0..4` and `residual ∈ [0, π/2)`; a residual within
/// tolerance of the grid snaps to zero (the gate is Clifford there).
fn peel_quarter_turns(angle: f64) -> (u8, f64) {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let tau = angle.rem_euclid(4.0 * half_pi);
    let k = (tau / half_pi).floor() as u8;
    let residual = tau - k as f64 * half_pi;
    if residual < TOL {
        (k % 4, 0.0)
    } else if half_pi - residual < TOL {
        ((k + 1) % 4, 0.0)
    } else {
        (k % 4, residual)
    }
}

/// Worst-case rank multiplier of a Z-rotation by `angle` (1 if Clifford).
fn rz_multiplier(angle: f64) -> usize {
    if peel_quarter_turns(angle).1 == 0.0 {
        1
    } else {
        2
    }
}

/// Worst-case rank multiplier of `C^k(diag(d0, d1))`.
fn cdiag_multiplier(d0: Complex<f64>, d1: Complex<f64>) -> usize {
    if (d0 - d1).norm() < TOL {
        // λ·I: a single projector over the controls alone.
        1 + usize::from((d0 - 1.0).norm() >= TOL)
    } else {
        1 + usize::from((d0 - 1.0).norm() >= TOL) + usize::from((d1 - 1.0).norm() >= TOL)
    }
}

fn is_diag(mat: &M2) -> bool {
    mat[0][1].norm() < TOL && mat[1][0].norm() < TOL
}

fn is_antidiag(mat: &M2) -> bool {
    mat[0][0].norm() < TOL && mat[1][1].norm() < TOL
}

impl SumState {
    /// |0…0⟩ as a single CH-form term.
    pub(crate) fn zero(n: usize) -> Self {
        SumState {
            n,
            terms: vec![ChForm::zero(n)],
        }
    }

    /// The number of stabilizer terms in the sum.
    pub(crate) fn rank(&self) -> usize {
        self.terms.len()
    }

    // ── Branching primitives ────────────────────────────────────────────

    /// Left-multiply every term by a plain gate.
    fn each(&mut self, f: impl Fn(&mut ChForm)) {
        for t in &mut self.terms {
            f(t);
        }
    }

    /// Clones of all terms with `Π` (controls at |1⟩, plus an optional
    /// target bit) applied and scaled by `coeff`; annihilated clones are
    /// dropped.
    fn projected_clones(
        &self,
        controls: &[u32],
        target: Option<(usize, bool)>,
        coeff: Complex<f64>,
    ) -> Vec<ChForm> {
        self.terms
            .iter()
            .filter_map(|t| {
                let mut c = t.clone();
                for &ctl in controls {
                    if c.project_z(ctl as usize, true) == Projection::Zero {
                        return None;
                    }
                }
                if let Some((tq, b)) = target
                    && c.project_z(tq, b) == Projection::Zero
                {
                    return None;
                }
                c.scale(coeff);
                Some(c)
            })
            .collect()
    }

    /// `C^k(diag(d0, d1)) = I + (d0−1)Π_{C,t=0} + (d1−1)Π_{C,t=1}`, exact.
    fn controlled_diag(
        &mut self,
        controls: &[u32],
        target: usize,
        d0: Complex<f64>,
        d1: Complex<f64>,
    ) {
        let one = Complex::new(1.0, 0.0);
        if (d0 - d1).norm() < TOL {
            // λ·I on the target: the target qubit is irrelevant.
            if (d0 - one).norm() >= TOL {
                let mut clones = self.projected_clones(controls, None, d0 - one);
                self.terms.append(&mut clones);
            }
            return;
        }
        let mut fresh = Vec::new();
        if (d0 - one).norm() >= TOL {
            fresh.extend(self.projected_clones(controls, Some((target, false)), d0 - one));
        }
        if (d1 - one).norm() >= TOL {
            fresh.extend(self.projected_clones(controls, Some((target, true)), d1 - one));
        }
        self.terms.append(&mut fresh);
    }

    /// `diag(1, e^{iδ}) = c_I·I + c_S·S` on qubit `q` (δ ∈ (0, π/2)),
    /// doubling the rank.
    fn branch_phase(&mut self, q: usize, delta: f64) {
        let i = Complex::new(0.0, 1.0);
        let c_s = (Complex::from_polar(1.0, delta) - 1.0) / (i - 1.0);
        let c_i = Complex::new(1.0, 0.0) - c_s;
        let mut clones: Vec<ChForm> = self
            .terms
            .iter()
            .map(|t| {
                let mut c = t.clone();
                c.scale(c_s);
                c.left_s(q);
                c
            })
            .collect();
        for t in &mut self.terms {
            t.scale(c_i);
        }
        self.terms.append(&mut clones);
    }

    /// Uncontrolled `diag(d0, d1)`, exact: a global d0, peeled S^k, and an
    /// {I, S} branch for the residual.
    fn apply_diag(&mut self, q: usize, d0: Complex<f64>, d1: Complex<f64>) {
        self.each(|t| t.scale(d0));
        let (k, residual) = peel_quarter_turns((d1 / d0).arg());
        for _ in 0..k {
            self.each(|t| t.left_s(q));
        }
        if residual != 0.0 {
            self.branch_phase(q, residual);
        }
    }

    /// Uncontrolled `Rz(θ) = e^{−iθ/2}·diag(1, e^{iθ})`, exact.
    fn apply_rz(&mut self, q: usize, theta: f64) {
        let d0 = Complex::from_polar(1.0, -theta / 2.0);
        let d1 = Complex::from_polar(1.0, theta / 2.0);
        self.apply_diag(q, d0, d1);
    }

    /// Non-Clifford `u` decomposition (controls already all positive).
    fn apply_noncliff(&mut self, target: usize, mat: &M2, controls: &[u32]) {
        if controls.is_empty() {
            if is_diag(mat) {
                self.apply_diag(target, mat[0][0], mat[1][1]);
            } else if is_antidiag(mat) {
                // mat = diag(b, c)·X, X applied first.
                self.each(|t| t.left_x(target));
                self.apply_diag(target, mat[0][1], mat[1][0]);
            } else {
                // e^{iα}·Rz(β)·Ry(γ)·Rz(δ), right-to-left; Ry via its
                // exact (SH)·Rz·(SH)† sandwich.
                let (alpha, beta, gamma, delta) = zyz(mat);
                self.apply_rz(target, delta);
                self.each(|t| {
                    t.left_sdg(target);
                    t.left_h(target);
                });
                self.apply_rz(target, gamma);
                self.each(|t| {
                    t.left_h(target);
                    t.left_s(target);
                });
                self.apply_rz(target, beta);
                let ph = Complex::from_polar(1.0, alpha);
                self.each(|t| t.scale(ph));
            }
        } else if is_diag(mat) {
            self.controlled_diag(controls, target, mat[0][0], mat[1][1]);
        } else if is_antidiag(mat) {
            // C^k(diag(b,c)·X) = C^k(diag(b,c)) · H_t·C^k(diag(1,−1))·H_t.
            self.each(|t| t.left_h(target));
            self.controlled_diag(
                controls,
                target,
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
            );
            self.each(|t| t.left_h(target));
            self.controlled_diag(controls, target, mat[0][1], mat[1][0]);
        } else {
            let (alpha, beta, gamma, delta) = zyz(mat);
            let rz_d = |th: f64| {
                (
                    Complex::from_polar(1.0, -th / 2.0),
                    Complex::from_polar(1.0, th / 2.0),
                )
            };
            let (d0, d1) = rz_d(delta);
            self.controlled_diag(controls, target, d0, d1);
            self.each(|t| {
                t.left_sdg(target);
                t.left_h(target);
            });
            let (d0, d1) = rz_d(gamma);
            self.controlled_diag(controls, target, d0, d1);
            self.each(|t| {
                t.left_h(target);
                t.left_s(target);
            });
            // Fold e^{iα} into the leftmost Rz's controlled diagonal.
            let a = Complex::from_polar(1.0, alpha);
            let (d0, d1) = rz_d(beta);
            self.controlled_diag(controls, target, a * d0, a * d1);
        }
    }

    /// Worst-case rank multiplier of the decomposition `apply_noncliff`
    /// would use (kept in lock-step with it; conservative because
    /// annihilated projector branches aren't predicted).
    fn noncliff_multiplier(mat: &M2, num_controls: usize) -> usize {
        if num_controls == 0 {
            if is_diag(mat) || is_antidiag(mat) {
                2
            } else {
                let (_, beta, gamma, delta) = zyz(mat);
                rz_multiplier(beta) * rz_multiplier(gamma) * rz_multiplier(delta)
            }
        } else if is_diag(mat) {
            cdiag_multiplier(mat[0][0], mat[1][1])
        } else if is_antidiag(mat) {
            2 * cdiag_multiplier(mat[0][1], mat[1][0])
        } else {
            let (alpha, beta, gamma, delta) = zyz(mat);
            let a = Complex::from_polar(1.0, alpha);
            let d = |th: f64| {
                (
                    Complex::from_polar(1.0, -th / 2.0),
                    Complex::from_polar(1.0, th / 2.0),
                )
            };
            let (b0, b1) = d(beta);
            let (g0, g1) = d(gamma);
            let (e0, e1) = d(delta);
            cdiag_multiplier(a * b0, a * b1) * cdiag_multiplier(g0, g1) * cdiag_multiplier(e0, e1)
        }
    }

    // ── The backend-facing operations ───────────────────────────────────

    /// Apply a `u` call. Clifford → rank unchanged; non-Clifford → exact
    /// branching. `Err(needed)` if the result would exceed `max_rank`
    /// (the state is left untouched so the caller can escalate).
    pub(crate) fn apply_u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
        max_rank: usize,
    ) -> Result<(), usize> {
        if !m.neg_controls.is_empty() {
            // X-conjugate negative controls into positive ones (exact and
            // rank-neutral; undone on Err so the state stays untouched).
            let mut pos = m.clone();
            let negs = std::mem::take(&mut pos.neg_controls);
            pos.controls.extend(negs);
            for &c in &m.neg_controls {
                self.each(|t| t.left_x(c as usize));
            }
            let r = self.apply_u(target, theta, phi, lambda, &pos, max_rank);
            for &c in &m.neg_controls {
                self.each(|t| t.left_x(c as usize));
            }
            return r;
        }
        if apply_clifford_u(self, target, theta, phi, lambda, m) {
            return Ok(());
        }
        let mat = effective_matrix(theta, phi, lambda, m.power);
        let needed = self
            .rank()
            .saturating_mul(Self::noncliff_multiplier(&mat, m.controls.len()));
        if needed > max_rank {
            return Err(needed);
        }
        self.apply_noncliff(target as usize, &mat, &m.controls);
        Ok(())
    }

    /// Apply a `gphase` call (`λ = e^{iγ·power}`); controlled, it is
    /// `I + (λ−1)·Π_C` over the controls.
    pub(crate) fn apply_gphase(
        &mut self,
        gamma: f64,
        m: &GateModifiers,
        max_rank: usize,
    ) -> Result<(), usize> {
        if !m.neg_controls.is_empty() {
            let mut pos = m.clone();
            let negs = std::mem::take(&mut pos.neg_controls);
            pos.controls.extend(negs);
            for &c in &m.neg_controls {
                self.each(|t| t.left_x(c as usize));
            }
            let r = self.apply_gphase(gamma, &pos, max_rank);
            for &c in &m.neg_controls {
                self.each(|t| t.left_x(c as usize));
            }
            return r;
        }
        if apply_clifford_gphase(self, gamma * m.power, m) {
            return Ok(());
        }
        let lambda = Complex::from_polar(1.0, gamma * m.power);
        if (lambda - 1.0).norm() < TOL {
            return Ok(());
        }
        let needed = self.rank().saturating_mul(2);
        if needed > max_rank {
            return Err(needed);
        }
        let mut clones = self.projected_clones(&m.controls, None, lambda - 1.0);
        self.terms.append(&mut clones);
        Ok(())
    }

    /// `‖Σ terms‖²` via the Hermitian Gram sum (diagonal is just |ω|²).
    fn norm_sqr(terms: &[ChForm]) -> f64 {
        let mut total = 0.0;
        for (i, ti) in terms.iter().enumerate() {
            total += ti.omega().norm_sqr();
            for tj in &terms[i + 1..] {
                total += 2.0 * ti.inner(tj).re;
            }
        }
        total
    }

    /// Project every term onto outcome `b` of qubit `q`, dropping
    /// annihilated terms, renormalizing, and pruning dead terms.
    fn collapse(&mut self, q: usize, b: bool, norm_sqr: f64) {
        let mut kept: Vec<ChForm> = std::mem::take(&mut self.terms)
            .into_iter()
            .filter_map(|mut t| (t.project_z(q, b) != Projection::Zero).then_some(t))
            .collect();
        let scale = Complex::new(1.0 / norm_sqr.max(PRUNE).sqrt(), 0.0);
        for t in &mut kept {
            t.scale(scale);
        }
        kept.retain(|t| t.omega().norm() > PRUNE);
        debug_assert!(!kept.is_empty(), "projection pruned every term");
        self.terms = kept;
    }

    /// Measure qubit `q` in the Z basis: P(1) from the Gram sum over the
    /// projected terms (O(r²·n³)), sample, collapse, renormalize, prune.
    pub(crate) fn measure(&mut self, q: usize, rng: &mut Rng) -> bool {
        let ones = self.projected_clones(&[], Some((q, true)), Complex::new(1.0, 0.0));
        let p_one = Self::norm_sqr(&ones).clamp(0.0, 1.0);
        // Degenerate probabilities are forced (mirrors the tableau's
        // determinism and avoids dividing by √ε).
        let outcome = if p_one < PRUNE {
            false
        } else if p_one > 1.0 - PRUNE {
            true
        } else {
            rng.next_f64() < p_one
        };
        let norm = if outcome { p_one } else { 1.0 - p_one };
        self.collapse(q, outcome, norm);
        outcome
    }

    /// Force qubit `q` onto `outcome` (the projection half of `measure`,
    /// no RNG) — used to replay recorded measurements during handoff.
    pub(crate) fn project(&mut self, q: usize, outcome: bool) {
        let kept = self.projected_clones(&[], Some((q, outcome)), Complex::new(1.0, 0.0));
        let norm = Self::norm_sqr(&kept).clamp(0.0, 1.0);
        self.collapse(q, outcome, norm);
    }

    /// Pauli X on `q`, term-wise (also used to replay recorded resets).
    pub(crate) fn x(&mut self, q: usize) {
        self.each(|t| t.left_x(q));
    }

    /// Reset qubit `q` to |0⟩: measure, then flip if it read 1. Returns
    /// the measured outcome (recorded for deterministic replay). The flip
    /// is `U(π,0,π) = i·X`, matching the dense backends' reset exactly
    /// (the global `i` is observable only to exact amplitude comparisons).
    pub(crate) fn reset(&mut self, q: usize, rng: &mut Rng) -> bool {
        let outcome = self.measure(q, rng);
        if outcome {
            self.x(q);
            self.each(|t| t.scale(Complex::new(0.0, 1.0)));
        }
        outcome
    }

    /// Exact amplitudes `Σ_i ⟨x|φ_i⟩` for all 2ⁿ basis states, or `None`
    /// if that vector can't be allocated.
    pub(crate) fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        let len = 1usize.checked_shl(self.n as u32)?;
        let mut out: Vec<Complex<f64>> = Vec::new();
        out.try_reserve_exact(len).ok()?;
        let mut bits = vec![false; self.n];
        for i in 0..len {
            for (q, b) in bits.iter_mut().enumerate() {
                *b = i >> q & 1 == 1;
            }
            out.push(self.terms.iter().map(|t| t.amplitude(&bits)).sum());
        }
        Some(out)
    }
}

/// Clifford gates forward to every term; `phase` scales every ω, keeping
/// the linear combination exact.
impl CliffordSink for SumState {
    fn h(&mut self, q: usize) {
        self.each(|t| t.left_h(q));
    }
    fn s(&mut self, q: usize) {
        self.each(|t| t.left_s(q));
    }
    fn cnot(&mut self, c: usize, t: usize) {
        self.each(|term| term.left_cx(c, t));
    }
    fn cz(&mut self, c: usize, t: usize) {
        self.each(|term| term.left_cz(c, t));
    }
    fn cy(&mut self, c: usize, t: usize) {
        self.each(|term| term.left_cy(c, t));
    }
    fn phase(&mut self, w: Complex<f64>) {
        self.each(|t| t.scale(w));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::QuantumBackend;
    use crate::sim::StateVectorSim;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const BIG: usize = 1 << 20;

    /// Tiny LCG for reproducible random circuits.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 1
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.next() % n as u64) as u32
        }
        fn angle(&mut self) -> f64 {
            (self.below(10_000) as f64 / 10_000.0) * 2.0 * PI
        }
    }

    fn none() -> GateModifiers {
        GateModifiers::none()
    }

    fn ctrl(cs: &[u32]) -> GateModifiers {
        GateModifiers {
            controls: cs.to_vec(),
            neg_controls: vec![],
            power: 1.0,
        }
    }

    /// One lowered backend call, applied identically to both simulators.
    #[derive(Clone, Debug)]
    enum Call {
        U(u32, f64, f64, f64, GateModifiers),
        Gphase(f64, GateModifiers),
    }

    async fn run_both(n: u32, calls: &[Call]) -> (SumState, StateVectorSim<f64>) {
        let mut sum = SumState::zero(n as usize);
        let mut sv = StateVectorSim::<f64>::try_zeroed(n, 0).unwrap();
        for c in calls {
            match c {
                Call::U(t, th, ph, la, m) => {
                    sum.apply_u(*t, *th, *ph, *la, m, BIG).unwrap();
                    sv.u(*t, *th, *ph, *la, m).await;
                }
                Call::Gphase(g, m) => {
                    sum.apply_gphase(*g, m, BIG).unwrap();
                    sv.gphase(*g, m).await;
                }
            }
        }
        (sum, sv)
    }

    async fn assert_matches_dense(sum: &SumState, sv: &StateVectorSim<f64>, what: &str) {
        let a = sum.amplitudes().unwrap();
        let b = sv.amplitudes().await.unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).norm() < 1e-9,
                "{what}: amplitude {i} mismatch: {x} vs {y}",
            );
        }
    }

    fn h(q: u32) -> Call {
        Call::U(q, FRAC_PI_2, 0.0, PI, none())
    }
    fn t(q: u32) -> Call {
        Call::U(q, 0.0, 0.0, FRAC_PI_4, none())
    }
    fn tdg(q: u32) -> Call {
        Call::U(q, 0.0, 0.0, -FRAC_PI_4, none())
    }

    /// `H T⁴ H = H Z H = X`: sixteen branches must interfere back down to
    /// a deterministic |1⟩.
    #[tokio::test(flavor = "multi_thread")]
    async fn t_gates_interfere_exactly() {
        let calls = vec![h(0), t(0), t(0), t(0), t(0), h(0)];
        let (sum, sv) = run_both(1, &calls).await;
        assert!(sum.rank() <= 16);
        assert_matches_dense(&sum, &sv, "HT⁴H").await;
        let amps = sum.amplitudes().unwrap();
        assert!(amps[0].norm() < 1e-9, "|0⟩ amplitude should cancel");
        assert!((amps[1].norm() - 1.0).abs() < 1e-9);
    }

    /// `T·T†` must cancel exactly (branch coefficients interfere to I).
    #[tokio::test(flavor = "multi_thread")]
    async fn t_tdg_cancels() {
        let calls = vec![h(0), t(0), tdg(0)];
        let (sum, sv) = run_both(1, &calls).await;
        assert_matches_dense(&sum, &sv, "T·T†").await;
    }

    /// `pow` reaches the backend as a per-primitive power: `t` really
    /// arrives as `U(0,0,π)^0.25`.
    #[tokio::test(flavor = "multi_thread")]
    async fn powered_u_matches_dense() {
        let quarter = GateModifiers {
            controls: vec![],
            neg_controls: vec![],
            power: 0.25,
        };
        let calls = vec![h(0), Call::U(0, 0.0, 0.0, PI, quarter)];
        let (sum, sv) = run_both(1, &calls).await;
        assert_matches_dense(&sum, &sv, "pow(0.25)@Z").await;
    }

    /// Random mixed circuits — Cliffords, T's, arbitrary rotations and
    /// generic U's — must match the dense simulator exactly.
    #[tokio::test(flavor = "multi_thread")]
    async fn random_mixed_circuits_match_dense() {
        let n = 3u32;
        for seed in 0..12u64 {
            let mut r = Lcg(seed + 1);
            let mut calls = Vec::new();
            let mut noncliff = 0;
            for _ in 0..25 {
                let q = r.below(n);
                match r.below(8) {
                    0 => calls.push(h(q)),
                    1 => calls.push(Call::U(q, 0.0, 0.0, FRAC_PI_2, none())),
                    2 | 3 => {
                        let c = (q + 1 + r.below(n - 1)) % n;
                        let g = if r.below(2) == 0 { PI } else { 0.0 };
                        // cx (θ=π) or cz (θ=0 ⇒ z), lowered like stdgates.
                        calls.push(Call::U(q, g, 0.0, PI, ctrl(&[c])));
                        if g == PI {
                            calls.push(Call::Gphase(-FRAC_PI_2, ctrl(&[c])));
                        }
                    }
                    4 if noncliff < 4 => {
                        noncliff += 1;
                        calls.push(t(q));
                    }
                    5 if noncliff < 4 => {
                        // rz(θ) as lowered: gphase(−θ/2); U(0,0,θ).
                        noncliff += 1;
                        let th = r.angle();
                        calls.push(Call::Gphase(-th / 2.0, none()));
                        calls.push(Call::U(q, 0.0, 0.0, th, none()));
                    }
                    6 if noncliff < 4 => {
                        // ry(θ) = U(θ, 0, 0) up to the built-in phase.
                        noncliff += 1;
                        calls.push(Call::U(q, r.angle(), 0.0, 0.0, none()));
                    }
                    7 if noncliff < 4 => {
                        noncliff += 1;
                        calls.push(Call::U(q, r.angle(), r.angle(), r.angle(), none()));
                    }
                    _ => calls.push(h(q)),
                }
            }
            let (sum, sv) = run_both(n, &calls).await;
            assert_matches_dense(&sum, &sv, &format!("seed {seed}")).await;
        }
    }

    /// Controlled non-Cliffords: `cp` (two-control gphase), `crz`, `ccx`
    /// (its real two-op lowering), and negative controls.
    #[tokio::test(flavor = "multi_thread")]
    async fn controlled_noncliffords_match_dense() {
        let n = 3u32;
        // cp(0.7) on (0,1) after spreading amplitude.
        let calls = vec![h(0), h(1), h(2), Call::Gphase(0.7, ctrl(&[0, 1]))];
        let (sum, sv) = run_both(n, &calls).await;
        assert_matches_dense(&sum, &sv, "cp").await;

        // crz(1.1) on (0,1): gphase(−0.55) ctrl [0]; U(0,0,1.1) ctrl [0].
        let calls = vec![
            h(0),
            h(1),
            Call::Gphase(-0.55, ctrl(&[0])),
            Call::U(1, 0.0, 0.0, 1.1, ctrl(&[0])),
        ];
        let (sum, sv) = run_both(n, &calls).await;
        assert_matches_dense(&sum, &sv, "crz").await;

        // ccx on (0,1 → 2): U(π,0,π) + gphase(−π/2), both under [0,1].
        let calls = vec![
            h(0),
            h(1),
            Call::U(2, PI, 0.0, PI, ctrl(&[0, 1])),
            Call::Gphase(-FRAC_PI_2, ctrl(&[0, 1])),
        ];
        let (sum, sv) = run_both(n, &calls).await;
        assert_matches_dense(&sum, &sv, "ccx").await;

        // A controlled generic rotation (ZYZ path) and a negative control.
        let neg = GateModifiers {
            controls: vec![0],
            neg_controls: vec![1],
            power: 1.0,
        };
        let calls = vec![
            h(0),
            h(1),
            Call::U(2, 0.9, 0.4, 1.7, ctrl(&[0])),
            Call::U(2, 0.3, 0.0, 0.0, neg),
        ];
        let (sum, sv) = run_both(n, &calls).await;
        assert_matches_dense(&sum, &sv, "ctrl-generic + negctrl").await;
    }

    /// Exceeding the budget returns `Err(needed)` and leaves the state
    /// untouched; retrying with room succeeds.
    #[tokio::test(flavor = "multi_thread")]
    async fn budget_overflow_is_clean() {
        let mut sum = SumState::zero(1);
        sum.apply_u(0, FRAC_PI_2, 0.0, PI, &none(), BIG).unwrap();
        let before = sum.amplitudes().unwrap();
        let e = sum.apply_u(0, 0.0, 0.0, FRAC_PI_4, &none(), 1);
        assert_eq!(e, Err(2));
        let after = sum.amplitudes().unwrap();
        for (x, y) in before.iter().zip(&after) {
            assert!((x - y).norm() < 1e-12, "state changed on Err");
        }
        sum.apply_u(0, 0.0, 0.0, FRAC_PI_4, &none(), 2).unwrap();
        assert_eq!(sum.rank(), 2);

        // Negative-control wrap must also restore the state on Err.
        let neg = GateModifiers {
            controls: vec![],
            neg_controls: vec![1],
            power: 1.0,
        };
        let mut sum = SumState::zero(2);
        sum.apply_u(0, FRAC_PI_2, 0.0, PI, &none(), BIG).unwrap();
        let before = sum.amplitudes().unwrap();
        let e = sum.apply_u(0, 0.0, 0.0, FRAC_PI_4, &neg, 1);
        assert!(e.is_err());
        let after = sum.amplitudes().unwrap();
        for (x, y) in before.iter().zip(&after) {
            assert!((x - y).norm() < 1e-12, "state changed on neg-ctrl Err");
        }
    }

    /// Mid-circuit measurement: outcomes forced onto the dense sim via
    /// `project` must leave both simulators in the same state.
    #[tokio::test(flavor = "multi_thread")]
    async fn measurement_matches_dense() {
        let n = 3u32;
        for seed in 0..10u64 {
            let mut r = Lcg(seed * 77 + 5);
            let mut sum = SumState::zero(n as usize);
            let mut sv = StateVectorSim::<f64>::try_zeroed(n, 0).unwrap();
            let mut rng = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            for step in 0..20 {
                let q = r.below(n);
                match r.below(6) {
                    0 | 1 => {
                        let Call::U(t2, th, ph, la, m) = h(q) else {
                            unreachable!()
                        };
                        sum.apply_u(t2, th, ph, la, &m, BIG).unwrap();
                        sv.u(t2, th, ph, la, &m).await;
                    }
                    2 => {
                        let c = (q + 1 + r.below(n - 1)) % n;
                        sum.apply_u(q, PI, 0.0, PI, &ctrl(&[c]), BIG).unwrap();
                        sv.u(q, PI, 0.0, PI, &ctrl(&[c])).await;
                        sum.apply_gphase(-FRAC_PI_2, &ctrl(&[c]), BIG).unwrap();
                        sv.gphase(-FRAC_PI_2, &ctrl(&[c])).await;
                    }
                    3 if step % 3 == 0 => {
                        sum.apply_u(q, 0.0, 0.0, FRAC_PI_4, &none(), BIG).unwrap();
                        sv.u(q, 0.0, 0.0, FRAC_PI_4, &none()).await;
                    }
                    _ => {
                        // Measure on the sum side; force the dense sim to
                        // the same outcome.
                        let pre = sv.amplitudes().await.unwrap();
                        let outcome = sum.measure(q as usize, &mut rng);
                        let bit = 1usize << q;
                        let p: f64 = pre
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (i & bit != 0) == outcome)
                            .map(|(_, a)| a.norm_sqr())
                            .sum();
                        assert!(p > 1e-9, "seed {seed}: outcome has ~0 probability");
                        sv.project(q, outcome);
                    }
                }
            }
            assert_matches_dense(&sum, &sv, &format!("seed {seed}")).await;

            // Reset drives the qubit back to |0⟩ with unit norm.
            sum.reset(0, &mut rng);
            let amps = sum.amplitudes().unwrap();
            let ones: f64 = amps
                .iter()
                .enumerate()
                .filter(|(i, _)| i & 1 == 1)
                .map(|(_, a)| a.norm_sqr())
                .sum();
            let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            assert!(ones < 1e-9, "reset left |1⟩ weight");
            assert!((norm - 1.0).abs() < 1e-9, "reset broke normalization");
        }
    }
}
