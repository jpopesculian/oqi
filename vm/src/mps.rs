//! A matrix-product-state (MPS) simulator backend.
//!
//! The state is a chain of one rank-3 tensor `A[l, s, r]` per qubit (site
//! `q` is global qubit `q`, `s` its physical index, `l`/`r` the bonds to
//! its neighbours). Memory and gate cost grow with the bond dimension χ —
//! the entanglement across each cut — rather than with 2ⁿ, so wide circuits
//! with modest entanglement stay cheap. After every multi-qubit gate the
//! bonds are recompressed by SVD: singular values carrying a negligible
//! share of the weight are dropped, and at most `max_bond` are kept. Past
//! that cap the state is truncated and renormalized, so results become
//! approximate.
//!
//! The chain is kept in mixed-canonical form around an orthogonality
//! centre: sites left of it are left-orthonormal, sites right of it are
//! right-orthonormal, and the centre site carries the norm. That makes a
//! single-qubit marginal a local computation and SVD truncation optimal.
//!
//! An uncontrolled `U` acts on one site and needs no SVD. A controlled `U`
//! — any number of positive/negative controls, at any distance — is
//! `I + ⊗ₖ Oₖ` over the sites it spans (`Oₖ` a control projector, `M − I`
//! on the target, `I` in between): a bond-dimension-2 operator chain that
//! is contracted into the state and then recompressed.
//!
//! Shots are drawn from the chain itself, without amplitudes: with the
//! centre on the first site, each qubit is sampled in turn from its
//! probability given the bits already drawn, at O(n·χ²) per shot.

use std::f64::consts::PI;

use async_trait::async_trait;
use faer::{Mat, MatRef};
use num_complex::Complex;

use crate::backend::{GateModifiers, MAX_STATE_BYTES, QuantumBackend};
use crate::clifford::{M2, effective_matrix, identity};
use crate::error::{KindResult, VmError, VmErrorKind, VmResult};
use crate::measure::renorm;
use crate::rng::Rng;

type C64 = Complex<f64>;

/// A singular value is dropped when the weight (squared norm) it and every
/// smaller one carry is at most this share of the total — far below what
/// any measurement statistic can resolve, so untruncated runs stay exact.
const CUTOFF: f64 = 1e-14;

/// One site tensor `A[l, s, r]`, stored row-major at `(l·2 + s)·dr + r`.
/// Read row-major, the same buffer is both the left-grouped `(dl·2) × dr`
/// matrix and the right-grouped `dl × (2·dr)` one.
#[derive(Clone)]
struct Site {
    dl: usize,
    dr: usize,
    data: Vec<C64>,
}

impl Site {
    /// |0⟩ with trivial bonds.
    fn zero() -> Self {
        Site {
            dl: 1,
            dr: 1,
            data: vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        }
    }

    /// The `(dl·2) × dr` matrix with rows `(l, s)`.
    fn left(&self) -> Mat<C64> {
        Mat::from_fn(self.dl * 2, self.dr, |i, j| self.data[i * self.dr + j])
    }

    /// The `dl × (2·dr)` matrix with columns `(s, r)`.
    fn right(&self) -> Mat<C64> {
        let cols = 2 * self.dr;
        Mat::from_fn(self.dl, cols, |i, j| self.data[i * cols + j])
    }

    fn from_left(m: MatRef<'_, C64>) -> Self {
        Self::from_rows(m, m.nrows() / 2, m.ncols())
    }

    fn from_right(m: MatRef<'_, C64>) -> Self {
        Self::from_rows(m, m.nrows(), m.ncols() / 2)
    }

    fn from_rows(m: MatRef<'_, C64>, dl: usize, dr: usize) -> Self {
        let data = (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
            .collect();
        Site { dl, dr, data }
    }

    /// Apply the 2×2 `m` to the physical index.
    fn apply(&mut self, m: &M2) {
        let dr = self.dr;
        for l in 0..self.dl {
            let base = l * 2 * dr;
            for r in 0..dr {
                let (a0, a1) = (self.data[base + r], self.data[base + dr + r]);
                self.data[base + r] = m[0][0] * a0 + m[0][1] * a1;
                self.data[base + dr + r] = m[1][0] * a0 + m[1][1] * a1;
            }
        }
    }

    /// `Σ |A[l, s, r]|²` over `l`, `r` for each `s`.
    fn weights(&self) -> [f64; 2] {
        let mut w = [0.0; 2];
        for (i, a) in self.data.iter().enumerate() {
            w[(i / self.dr) % 2] += a.norm_sqr();
        }
        w
    }
}

/// Matrix-product-state simulator: memory and time scale with the
/// entanglement (bond dimension) instead of 2ⁿ, truncating to at most
/// `max_bond` singular values per bond.
///
/// Like the dense simulators it keeps its state under 16 GiB: a gate that
/// would grow the tensors past that fails with [`VmErrorKind::Backend`]
/// (lower the bond cap to trade accuracy for memory), and
/// [`amplitudes`](QuantumBackend::amplitudes) returns `None` for more than
/// 30 qubits.
pub struct MpsSim {
    sites: Vec<Site>,
    /// Total entries over every site's tensor, kept up to date by
    /// [`set_site`](Self::set_site).
    elements: usize,
    /// The orthogonality centre.
    center: usize,
    /// The global phase from `gphase`, kept off the tensors.
    phase: C64,
    max_bond: usize,
    /// Total share of the weight dropped by truncation so far.
    discarded: f64,
    rng: Rng,
}

impl MpsSim {
    /// The default cap on the bond dimension.
    pub const DEFAULT_MAX_BOND: usize = 256;

    /// A fresh simulator with `num_qubits` qubits in |0…0⟩, an RNG seed for
    /// measurement sampling, and a cap on the bond dimension.
    ///
    /// Fails with [`VmErrorKind::Backend`] if `max_bond` is 0, and with
    /// [`VmErrorKind::TooManyQubits`] if even the unentangled chain would
    /// not fit in memory.
    pub fn new(num_qubits: u32, seed: u64, max_bond: usize) -> VmResult<Self> {
        if max_bond == 0 {
            return Err(VmError::new(VmErrorKind::Backend(
                "the MPS bond cap must be at least 1".into(),
            )));
        }
        let per_site = (std::mem::size_of::<Site>() + 2 * std::mem::size_of::<C64>()) as u64;
        if u64::from(num_qubits).saturating_mul(per_site) > MAX_STATE_BYTES {
            return Err(VmError::new(VmErrorKind::TooManyQubits {
                requested: num_qubits,
            }));
        }
        Ok(MpsSim {
            sites: vec![Site::zero(); num_qubits as usize],
            elements: 2 * num_qubits as usize,
            center: 0,
            phase: Complex::new(1.0, 0.0),
            max_bond,
            discarded: 0.0,
            rng: Rng::new(seed),
        })
    }

    /// Replace site `k`, keeping the entry count current.
    fn set_site(&mut self, k: usize, site: Site) {
        self.elements = self.elements - self.sites[k].data.len() + site.data.len();
        self.sites[k] = site;
    }

    /// Fail if `extra` more tensor entries would take the state past the
    /// memory ceiling.
    fn reserve(&self, extra: usize) -> KindResult<()> {
        let entries = self.elements.saturating_add(extra) as u64;
        if entries.saturating_mul(std::mem::size_of::<C64>() as u64) > MAX_STATE_BYTES {
            return Err(VmErrorKind::Backend(format!(
                "the matrix-product state would need more than {} GiB at this \
                 entanglement; lower the bond cap (now {})",
                MAX_STATE_BYTES >> 30,
                self.max_bond
            )));
        }
        Ok(())
    }

    /// Move the orthogonality centre to site `k` by QR sweeps.
    fn move_center(&mut self, k: usize) {
        while self.center < k {
            let c = self.center;
            // A = Q·R: Q stays, R moves right.
            let qr = self.sites[c].left().qr();
            self.set_site(c, Site::from_left(qr.compute_thin_Q().as_ref()));
            let next = qr.thin_R() * self.sites[c + 1].right();
            self.set_site(c + 1, Site::from_right(next.as_ref()));
            self.center += 1;
        }
        while self.center > k {
            let c = self.center;
            // A = L·Q, from Aᴴ = Q'·R': Q = Q'ᴴ stays, L = R'ᴴ moves left.
            let qr = self.sites[c].right().adjoint().qr();
            let q = qr.compute_thin_Q().adjoint().to_owned();
            self.set_site(c, Site::from_right(q.as_ref()));
            let prev = self.sites[c - 1].left() * qr.thin_R().adjoint();
            self.set_site(c - 1, Site::from_left(prev.as_ref()));
            self.center -= 1;
        }
    }

    /// Restore canonical form over sites `a..=b` after an operator was
    /// applied there (everything outside still canonical towards it):
    /// orthonormalize left to right, then truncate right to left by SVD.
    /// The centre ends at `a`. Fails if a singular value decomposition does
    /// not converge, as it may on non-finite amplitudes (a NaN angle).
    fn compress(&mut self, a: usize, b: usize) -> KindResult<()> {
        self.center = a;
        self.move_center(b);
        for c in (a + 1..=b).rev() {
            let svd = self.sites[c].right().thin_svd().map_err(|_| {
                VmErrorKind::Backend(
                    "a singular value decomposition did not converge \
                     (non-finite amplitudes?)"
                        .into(),
                )
            })?;
            let s: Vec<f64> = svd.S().column_vector().iter().map(|x| x.re).collect();
            let keep = self.truncation(&s);
            // Keep the norm: scale the kept singular values back up to the
            // full weight.
            let total: f64 = s.iter().map(|x| x * x).sum();
            let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
            let scale = if kept > 0.0 {
                (total / kept).sqrt()
            } else {
                0.0
            };
            if total > 0.0 {
                self.discarded += 1.0 - kept / total;
            }
            let v = svd.V().get(.., ..keep);
            self.set_site(c, Site::from_right(v.adjoint().to_owned().as_ref()));
            let us = Mat::from_fn(svd.U().nrows(), keep, |i, j| {
                svd.U()[(i, j)] * (s[j] * scale)
            });
            let prev = self.sites[c - 1].left() * us;
            self.set_site(c - 1, Site::from_left(prev.as_ref()));
        }
        self.center = a;
        Ok(())
    }

    /// How many of the (descending) singular values `s` to keep.
    fn truncation(&self, s: &[f64]) -> usize {
        let total: f64 = s.iter().map(|x| x * x).sum();
        let mut keep = s.len();
        let mut tail = 0.0;
        while keep > 1 {
            let w = s[keep - 1] * s[keep - 1];
            if tail + w > CUTOFF * total {
                break;
            }
            tail += w;
            keep -= 1;
        }
        keep.min(self.max_bond)
    }

    /// Draw one Z-basis sample of the register, qubit by qubit, each from
    /// its probability given the bits already drawn. Needs the centre on
    /// site 0: the rest of the chain is then right-orthonormal, so the
    /// weight of a prefix of outcomes is the squared norm of the row vector
    /// `v` it leaves on the next bond.
    fn sample_one(&self, rng: &mut Rng) -> Vec<bool> {
        let mut v = vec![Complex::new(1.0, 0.0)];
        let mut bits = Vec::with_capacity(self.sites.len());
        for site in &self.sites {
            // w[s][r] = Σ_l v[l]·A[l, s, r]
            let mut w = [0, 1].map(|_| vec![Complex::new(0.0, 0.0); site.dr]);
            for (l, &vl) in v.iter().enumerate() {
                for (s, ws) in w.iter_mut().enumerate() {
                    let row = (l * 2 + s) * site.dr;
                    for (r, x) in ws.iter_mut().enumerate() {
                        *x += vl * site.data[row + r];
                    }
                }
            }
            let [w0, w1] = w
                .each_ref()
                .map(|ws| ws.iter().map(|x| x.norm_sqr()).sum::<f64>());
            let bit = rng.next_f64() < w1 / (w0 + w1);
            let norm = if bit { w1 } else { w0 }.sqrt();
            let [v0, v1] = w;
            v = if bit { v1 } else { v0 };
            for x in &mut v {
                *x /= norm;
            }
            bits.push(bit);
        }
        bits
    }

    /// Apply `I + ⊗ₖ Oₖ` over sites `a..=b`, where `ops` gives `Oₖ` for the
    /// sites it names and every other site in the span takes `I`. Channel 0
    /// of the doubled bonds carries `I`, channel 1 carries `Oₖ`.
    fn apply_sum(&mut self, ops: &[(usize, M2)]) -> KindResult<()> {
        let a = ops.iter().map(|&(k, _)| k).min().expect("an operator");
        let b = ops.iter().map(|&(k, _)| k).max().expect("an operator");
        self.move_center(a);
        // Each inner bond of the span doubles until `compress` cuts it back.
        let doubled: usize = (a..=b)
            .map(|k| {
                let site = &self.sites[k];
                let lch = if k == a { 1 } else { 2 };
                let rch = if k == b { 1 } else { 2 };
                site.data.len() * (lch * rch - 1)
            })
            .sum();
        self.reserve(doubled)?;
        let id = identity();
        for k in a..=b {
            let site = &self.sites[k];
            let op = ops.iter().find(|&&(q, _)| q == k).map_or(&id, |(_, m)| m);
            let (dl, dr) = (site.dl, site.dr);
            let lch = if k == a { 1 } else { 2 };
            let rch = if k == b { 1 } else { 2 };
            let ndr = dr * rch;
            let mut data = vec![Complex::new(0.0, 0.0); dl * lch * 2 * ndr];
            for ch in 0..2 {
                let mut branch = site.clone();
                if ch == 1 {
                    branch.apply(op);
                }
                // At the span's ends both channels sum into the one bond.
                let (lc, rc) = (ch % lch, ch % rch);
                for l in 0..dl {
                    for s in 0..2 {
                        for r in 0..dr {
                            let row = (l * lch + lc) * 2 + s;
                            data[row * ndr + r * rch + rc] += branch.data[(l * 2 + s) * dr + r];
                        }
                    }
                }
            }
            let site = Site {
                dl: dl * lch,
                dr: ndr,
                data,
            };
            self.set_site(k, site);
        }
        self.compress(a, b)
    }
}

#[async_trait(?Send)]
impl QuantumBackend for MpsSim {
    fn num_qubits(&self) -> u32 {
        self.sites.len() as u32
    }

    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
    ) -> KindResult<()> {
        let mut mat = effective_matrix(theta, phi, lambda, m.power);
        let target = target as usize;
        if m.controls.is_empty() && m.neg_controls.is_empty() {
            self.sites[target].apply(&mat);
            return Ok(());
        }
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        // `M − I` on the target, a projector on each control.
        mat[0][0] -= one;
        mat[1][1] -= one;
        let mut ops = vec![(target, mat)];
        ops.extend(
            m.controls
                .iter()
                .map(|&c| (c as usize, [[zero, zero], [zero, one]])),
        );
        ops.extend(
            m.neg_controls
                .iter()
                .map(|&c| (c as usize, [[one, zero], [zero, zero]])),
        );
        self.apply_sum(&ops)
    }

    fn gphase(&mut self, gamma: f64) -> KindResult<()> {
        self.phase *= Complex::from_polar(1.0, gamma);
        Ok(())
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        let q = qubit as usize;
        self.move_center(q);
        let site = &mut self.sites[q];
        let [w0, w1] = site.weights();
        let p_one = w1 / (w0 + w1);
        let outcome = self.rng.next_f64() < p_one;
        let scale = renorm(p_one, outcome);
        let dr = site.dr;
        for (i, a) in site.data.iter_mut().enumerate() {
            *a = if ((i / dr) % 2 == 1) == outcome {
                *a * scale
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        outcome
    }

    async fn reset(&mut self, qubit: u32) {
        if self.measure(qubit).await {
            // U(π, 0, π) is i·X, as in the state-vector simulator.
            self.sites[qubit as usize].apply(&effective_matrix(PI, 0.0, PI, 1.0));
        }
    }

    fn reset_state(&mut self) -> KindResult<()> {
        self.sites.fill(Site::zero());
        self.elements = 2 * self.sites.len();
        self.center = 0;
        self.phase = Complex::new(1.0, 0.0);
        self.discarded = 0.0;
        Ok(())
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        // The dense vector obeys the dense simulators' memory ceiling.
        let bytes = (std::mem::size_of::<C64>() as u64).checked_shl(self.num_qubits())?;
        if self.num_qubits() >= 64 || bytes > MAX_STATE_BYTES {
            return None;
        }
        // Contract left to right: `psi[idx·d + r]` holds the amplitude of
        // the qubits so far in basis state `idx` (qubit q is bit q), left
        // open on bond index `r` of width `d`.
        let mut psi = vec![self.phase];
        for (k, site) in self.sites.iter().enumerate() {
            let len = 1usize.checked_shl(k as u32 + 1)?.checked_mul(site.dr)?;
            let mut next = Vec::new();
            next.try_reserve_exact(len).ok()?;
            next.resize(len, Complex::new(0.0, 0.0));
            for idx in 0..1usize << k {
                for l in 0..site.dl {
                    let v = psi[idx * site.dl + l];
                    if v == Complex::new(0.0, 0.0) {
                        continue;
                    }
                    for s in 0..2 {
                        let out = (idx | s << k) * site.dr;
                        let row = (l * 2 + s) * site.dr;
                        for r in 0..site.dr {
                            next[out + r] += v * site.data[row + r];
                        }
                    }
                }
            }
            psi = next;
        }
        Some(psi)
    }

    fn can_sample_shots(&self) -> bool {
        true
    }

    async fn sample_shots(&mut self, shots: u32, seed: u64) -> Option<Vec<Vec<bool>>> {
        self.move_center(0);
        let mut rng = Rng::new(seed);
        Some((0..shots).map(|_| self.sample_one(&mut rng)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::StateVectorSim;
    use std::f64::consts::{FRAC_PI_2, PI};

    const TOL: f64 = 1e-9;

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
            (self.next() % 1_000_000) as f64 / 1_000_000.0 * 2.0 * PI - PI
        }
    }

    enum Op {
        U {
            target: u32,
            theta: f64,
            phi: f64,
            lambda: f64,
            mods: GateModifiers,
        },
        Gphase(f64),
        Measure(u32),
    }

    /// A random circuit of arbitrary `U`s — with up to three positive or
    /// negative controls anywhere on the register and fractional powers —
    /// global phases, and mid-circuit measurements.
    fn random_circuit(n: u32, len: usize, seed: u64) -> Vec<Op> {
        let mut r = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
        (0..len)
            .map(|_| match r.below(10) {
                0 => Op::Gphase(r.angle()),
                1 => Op::Measure(r.below(n)),
                _ => {
                    let target = r.below(n);
                    let mut mods = GateModifiers::none();
                    for _ in 0..r.below(4).min(n - 1) {
                        let mut c = r.below(n);
                        while c == target
                            || mods.controls.contains(&c)
                            || mods.neg_controls.contains(&c)
                        {
                            c = (c + 1) % n;
                        }
                        if r.below(3) == 0 {
                            mods.neg_controls.push(c);
                        } else {
                            mods.controls.push(c);
                        }
                    }
                    mods.power = [1.0, 1.0, 1.0, 0.5, -1.0, 0.3][r.below(6) as usize];
                    Op::U {
                        target,
                        theta: r.angle(),
                        phi: r.angle(),
                        lambda: r.angle(),
                        mods,
                    }
                }
            })
            .collect()
    }

    async fn assert_same_state(mps: &MpsSim, sv: &StateVectorSim<f64>, what: &str) {
        let a = mps.amplitudes().await.unwrap();
        let b = sv.amplitudes().await.unwrap();
        assert_eq!(a.len(), b.len(), "{what}: state length");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).norm() < TOL, "{what}: amp {i}: {x} vs {y}");
        }
    }

    /// Exact (untruncated) MPS evolution must reproduce the state vector,
    /// phase included, through controlled gates spanning distant qubits
    /// and mid-circuit measurements.
    #[test]
    fn matches_statevector_on_random_circuits() {
        pollster::block_on(async {
            let n = 6u32;
            for seed in 0..24u64 {
                let mut mps = MpsSim::new(n, seed + 1, usize::MAX).unwrap();
                let mut sv = StateVectorSim::<f64>::try_zeroed(n, 0).unwrap();
                for op in random_circuit(n, 60, seed) {
                    match op {
                        Op::U {
                            target,
                            theta,
                            phi,
                            lambda,
                            mods,
                        } => {
                            mps.u(target, theta, phi, lambda, &mods).unwrap();
                            sv.u(target, theta, phi, lambda, &mods).unwrap();
                        }
                        Op::Gphase(g) => {
                            mps.gphase(g).unwrap();
                            sv.gphase(g).unwrap();
                        }
                        Op::Measure(q) => {
                            let outcome = mps.measure(q).await;
                            sv.project(q, outcome);
                        }
                    }
                }
                assert_same_state(&mps, &sv, &format!("seed {seed}")).await;
                assert!(
                    mps.discarded < 1e-12,
                    "seed {seed}: truncated {}",
                    mps.discarded
                );
            }
        })
    }

    /// A Bell pair always measures equal bits, and both outcomes occur.
    #[test]
    fn bell_pair_is_perfectly_correlated() {
        pollster::block_on(async {
            let (mut saw_zero, mut saw_one) = (false, false);
            for seed in 0..40u64 {
                let mut mps =
                    MpsSim::new(2, (seed + 1).wrapping_mul(0x9E3779B97F4A7C15), 8).unwrap();
                mps.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
                    .unwrap(); // H
                mps.u(1, PI, 0.0, PI, &GateModifiers::ctrl(&[0])).unwrap(); // CX
                let out = mps.measure_all(&[0, 1]).await;
                assert_eq!(out[0], out[1], "Bell measurement must agree (seed {seed})");
                saw_zero |= !out[0];
                saw_one |= out[0];
            }
            assert!(saw_zero && saw_one, "measurement looks non-random");
        })
    }

    /// `reset` flips a |1⟩ back with the same `U(π, 0, π)` (= i·X) as the
    /// state-vector simulator, so the phases agree.
    #[test]
    fn reset_matches_statevector() {
        pollster::block_on(async {
            let none = GateModifiers::none();
            let mut mps = MpsSim::new(3, 5, 8).unwrap();
            let mut sv = StateVectorSim::<f64>::try_zeroed(3, 5).unwrap();
            for b in [&mut mps as &mut dyn QuantumBackend, &mut sv] {
                b.u(0, FRAC_PI_2, 0.0, PI, &none).unwrap(); // H q0
                b.u(2, PI, 0.0, PI, &GateModifiers::ctrl(&[0])).unwrap(); // CX q0 → q2
                b.u(1, PI, 0.0, PI, &none).unwrap(); // X q1: a deterministic 1
                b.reset(1).await;
            }
            assert_same_state(&mps, &sv, "after reset").await;
        })
    }

    /// `reset_state` returns the register to |0…0⟩.
    #[test]
    fn reset_state_rezeros() {
        pollster::block_on(async {
            let mut mps = MpsSim::new(3, 7, 8).unwrap();
            mps.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
                .unwrap();
            mps.u(2, PI, 0.0, PI, &GateModifiers::ctrl(&[0])).unwrap();
            mps.gphase(0.4).unwrap();
            mps.reset_state().unwrap();
            let amps = mps.amplitudes().await.unwrap();
            assert!((amps[0] - Complex::new(1.0, 0.0)).norm() < TOL);
            assert!(amps[1..].iter().all(|a| a.norm() < TOL));
        })
    }

    /// With a bond cap of 1 a Bell pair is truncated to a product state:
    /// half the weight is discarded and the rest renormalized.
    #[test]
    fn truncation_renormalizes() {
        pollster::block_on(async {
            let mut mps = MpsSim::new(2, 1, 1).unwrap();
            mps.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
                .unwrap();
            mps.u(1, PI, 0.0, PI, &GateModifiers::ctrl(&[0])).unwrap();
            let amps = mps.amplitudes().await.unwrap();
            let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            assert!((norm - 1.0).abs() < TOL, "norm {norm}");
            let nonzero = amps.iter().filter(|a| a.norm() > TOL).count();
            assert_eq!(nonzero, 1, "a product state of basis states: {amps:?}");
            assert!((mps.discarded - 0.5).abs() < TOL, "{}", mps.discarded);
        })
    }

    /// Apply the gates of a random circuit (skipping its measurements).
    fn apply_unitaries(b: &mut dyn QuantumBackend, circuit: &[Op]) {
        for op in circuit {
            match op {
                Op::U {
                    target,
                    theta,
                    phi,
                    lambda,
                    mods,
                } => b.u(*target, *theta, *phi, *lambda, mods).unwrap(),
                Op::Gphase(g) => b.gphase(*g).unwrap(),
                Op::Measure(_) => {}
            }
        }
    }

    /// Shots drawn from the chain follow |ψ|² of the same circuit on the
    /// state vector, bit `q` of the basis index being qubit `q`.
    #[test]
    fn sampled_shots_follow_the_born_rule() {
        pollster::block_on(async {
            let n = 5u32;
            let circuit = random_circuit(n, 40, 3);
            let mut mps = MpsSim::new(n, 1, usize::MAX).unwrap();
            apply_unitaries(&mut mps, &circuit);
            let mut sv = StateVectorSim::<f64>::try_zeroed(n, 1).unwrap();
            apply_unitaries(&mut sv, &circuit);

            let shots = 50_000;
            let mut counts = vec![0u32; 1 << n];
            for shot in mps.sample_shots(shots, 11).await.unwrap() {
                assert_eq!(shot.len(), n as usize);
                let idx = (0..n as usize).fold(0, |acc, q| acc | (shot[q] as usize) << q);
                counts[idx] += 1;
            }
            let amps = sv.amplitudes().await.unwrap();
            let tv: f64 = counts
                .iter()
                .zip(&amps)
                .map(|(&c, a)| (c as f64 / shots as f64 - a.norm_sqr()).abs())
                .sum::<f64>()
                / 2.0;
            assert!(tv < 0.02, "total variation distance {tv}");
        })
    }

    /// Sampling reads the state without collapsing it, wherever the
    /// orthogonality centre was.
    #[test]
    fn sampling_leaves_the_state_untouched() {
        pollster::block_on(async {
            let mut mps = MpsSim::new(4, 2, usize::MAX).unwrap();
            apply_unitaries(&mut mps, &random_circuit(4, 30, 5));
            mps.measure(3).await; // centre on the last site
            let before = mps.amplitudes().await.unwrap();
            mps.sample_shots(100, 1).await.unwrap();
            let after = mps.amplitudes().await.unwrap();
            for (i, (x, y)) in before.iter().zip(&after).enumerate() {
                assert!((x - y).norm() < TOL, "amp {i}: {x} vs {y}");
            }
        })
    }

    /// Shots come straight from the chain: a 100-qubit GHZ, far past any
    /// amplitude vector, samples only all-0 and all-1.
    #[test]
    fn sampling_needs_no_amplitudes() {
        pollster::block_on(async {
            let n = 100u32;
            let mut mps = MpsSim::new(n, 1, 8).unwrap();
            mps.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
                .unwrap();
            for q in 0..n - 1 {
                mps.u(q + 1, PI, 0.0, PI, &GateModifiers::ctrl(&[q]))
                    .unwrap();
            }
            let drawn = mps.sample_shots(200, 4).await.expect("sampled");
            assert!(drawn.iter().all(|s| s.iter().all(|&b| b == s[0])));
            let ones = drawn.iter().filter(|s| s[0]).count();
            assert!((50..150).contains(&ones), "{ones} of 200 all-1 shots");
        })
    }

    /// A register with no qubits has the one amplitude `e^{iγ}`.
    #[test]
    fn zero_qubits_hold_the_global_phase() {
        pollster::block_on(async {
            let mut mps = MpsSim::new(0, 1, 8).unwrap();
            mps.gphase(0.3).unwrap();
            let amps = mps.amplitudes().await.unwrap();
            assert_eq!(amps.len(), 1);
            assert!((amps[0] - Complex::from_polar(1.0, 0.3)).norm() < TOL);
        })
    }

    /// The chain refuses what would not fit in memory, before allocating it.
    #[test]
    fn memory_is_bounded() {
        assert!(matches!(
            MpsSim::new(u32::MAX, 1, 8).err().map(|e| e.kind),
            Some(VmErrorKind::TooManyQubits { .. })
        ));
        assert!(matches!(
            MpsSim::new(2, 1, 0).err().map(|e| e.kind),
            Some(VmErrorKind::Backend(_))
        ));
        // A state already at the ceiling cannot grow a bond.
        let mut mps = MpsSim::new(2, 1, 8).unwrap();
        mps.u(0, FRAC_PI_2, 0.0, PI, &GateModifiers::none())
            .unwrap();
        mps.elements = (MAX_STATE_BYTES / std::mem::size_of::<C64>() as u64) as usize;
        let err = mps.u(1, PI, 0.0, PI, &GateModifiers::ctrl(&[0]));
        assert!(matches!(err, Err(VmErrorKind::Backend(_))), "{err:?}");
    }
}
