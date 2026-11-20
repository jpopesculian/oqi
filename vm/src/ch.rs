//! A single stabilizer state in CH form (Bravyi, Browne, Calpin, Campbell,
//! Gosset, Howard, "Simulation of quantum circuits by low-rank stabilizer
//! decompositions", Quantum 3, 181 (2019), §4):
//!
//! ```text
//!   |φ⟩ = ω · U_C · U_H |s⟩
//! ```
//!
//! with `U_C` a *control-type* Clifford (a product of S/CZ/CX gates, which
//! all fix |0…0⟩), `U_H = ⊗_j H^{v_j}` a Hadamard layer, `s` a computational
//! basis state, and `ω` a complex scalar carrying both the term's exact
//! phase and its weight (|ω|² shrinks under projection). Unlike the CHP
//! tableau this representation is *phase-exact*, so a linear combination of
//! `ChForm`s interferes correctly.
//!
//! `U_C` is tracked by its conjugation action on the Pauli generators:
//!
//! ```text
//!   U_C† X_p U_C = i^{γ_p} · ∏_j X_j^{F_pj} Z_j^{M_pj}
//!   U_C† Z_p U_C = ∏_j Z_j^{G_pj}
//! ```
//!
//! (Z-type images are Z-type with no phase because S/CZ/CX map Z's to Z's.)
//! Symplecticity forces `F·Gᵀ = I`, so `F⁻¹ = Gᵀ` and `G⁻¹ = Fᵀ` come free.
//! The update rules below are derived directly from these invariants; the
//! random-circuit differential tests compare full amplitude vectors —
//! including global phase — against the dense simulator.

// Dense parallel-array Pauli arithmetic: the loop index simultaneously
// addresses several bit-vectors (F, G, M, s, v), so indexed loops mirror the
// algebra far more legibly than iterator adapters. Correctness is pinned by the
// random-circuit amplitude-differential tests.
#![allow(clippy::needless_range_loop)]

use num_complex::Complex;

use crate::clifford::CliffordSink;

/// How a Z-basis projection acted on the state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Projection {
    /// The outcome was certain; the state is unchanged.
    Deterministic,
    /// The opposite outcome was certain; the state was annihilated (ω = 0).
    Zero,
    /// The outcome had probability ½; the state was projected and ω picked
    /// up a factor 1/√2 (left *unnormalized*).
    Half,
}

#[derive(Clone)]
pub(crate) struct ChForm {
    n: usize,
    /// X-part of the image of `X_p` (row `p`).
    f: Vec<Vec<bool>>,
    /// Z-part of the image of `Z_p` (row `p`).
    g: Vec<Vec<bool>>,
    /// Z-part of the image of `X_p` (row `p`).
    m: Vec<Vec<bool>>,
    /// Phase (power of i, mod 4) of the image of `X_p`.
    gamma: Vec<u8>,
    /// The Hadamard layer: `v_j` ⇔ an `H` on qubit `j`.
    v: Vec<bool>,
    /// The basis state behind the Hadamard layer.
    s: Vec<bool>,
    /// Global scalar (phase *and* weight).
    omega: Complex<f64>,
}

/// `i^k` exactly.
fn ipow(k: u8) -> Complex<f64> {
    match k % 4 {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

impl ChForm {
    /// |0…0⟩: `U_C = U_H = 1` (F = G = I, M = 0, γ = 0), ω = 1.
    pub(crate) fn zero(n: usize) -> Self {
        let mut f = vec![vec![false; n]; n];
        let mut g = vec![vec![false; n]; n];
        for j in 0..n {
            f[j][j] = true;
            g[j][j] = true;
        }
        ChForm {
            n,
            f,
            g,
            m: vec![vec![false; n]; n],
            gamma: vec![0; n],
            v: vec![false; n],
            s: vec![false; n],
            omega: Complex::new(1.0, 0.0),
        }
    }

    pub(crate) fn omega(&self) -> Complex<f64> {
        self.omega
    }

    /// ω *= w (phase or weight update).
    pub(crate) fn scale(&mut self, w: Complex<f64>) {
        self.omega *= w;
    }

    // ── U_C conjugation helpers ─────────────────────────────────────────

    /// `U_C† (X^x Z^z) U_C` as `(phase power of i, x-part, z-part)`,
    /// multiplying the per-qubit images left-to-right (X factors first,
    /// then Z factors, matching the canonical ordering of the input).
    fn conj_pauli(&self, x: &[bool], z: &[bool]) -> (u8, Vec<bool>, Vec<bool>) {
        let n = self.n;
        let mut ph = 0u8;
        let mut xv = vec![false; n];
        let mut zv = vec![false; n];
        for j in 0..n {
            if x[j] {
                // Multiply on the right by i^{γ_j} X^{F_j} Z^{M_j}: the
                // current Z-part must commute past the incoming X-part.
                let mut cross = 0u32;
                for k in 0..n {
                    if zv[k] && self.f[j][k] {
                        cross += 1;
                    }
                }
                ph = (ph + self.gamma[j] + 2 * (cross % 2) as u8) % 4;
                for k in 0..n {
                    xv[k] ^= self.f[j][k];
                    zv[k] ^= self.m[j][k];
                }
            }
        }
        for j in 0..n {
            if z[j] {
                // Z-type images append to the Z-part with no crossing.
                for k in 0..n {
                    zv[k] ^= self.g[j][k];
                }
            }
        }
        (ph, xv, zv)
    }

    /// Push `i^{ph} X^x Z^z` through `U_H` (X↔Z on Hadamard qubits, with a
    /// `(−1)` per qubit carrying both an X and a Z), then act on |s⟩:
    /// returns the total phase power of i and the flipped basis state.
    fn pauli_behind_h(&self, ph: u8, x: &[bool], z: &[bool]) -> (u8, Vec<bool>) {
        let n = self.n;
        let mut ph = ph;
        let mut t = self.s.clone();
        for j in 0..n {
            let (bx, bz) = if self.v[j] {
                (z[j], x[j])
            } else {
                (x[j], z[j])
            };
            if self.v[j] && x[j] && z[j] {
                ph = (ph + 2) % 4; // X^a Z^b → Z^a X^b = (−1)^{ab} X^b Z^a
            }
            if bz && self.s[j] {
                ph = (ph + 2) % 4; // Z_j |1⟩ = −|1⟩
            }
            t[j] ^= bx;
        }
        (ph, t)
    }

    // ── Right multiplication U_C ← U_C · V (desuperposition internals) ──

    /// U_C ← U_C · S_q. Conjugation: `S† X_q S = i³ X_q Z_q`, Z's fixed.
    fn right_s(&mut self, q: usize) {
        for p in 0..self.n {
            if self.f[p][q] {
                self.gamma[p] = (self.gamma[p] + 3) % 4;
                self.m[p][q] ^= true;
            }
        }
    }

    /// U_C ← U_C · CX (control `a`, target `b`). Conjugation:
    /// `X_a → X_a X_b`, `Z_b → Z_a Z_b`.
    fn right_cx(&mut self, a: usize, b: usize) {
        for p in 0..self.n {
            let fa = self.f[p][a];
            self.f[p][b] ^= fa;
            let mb = self.m[p][b];
            self.m[p][a] ^= mb;
            let gb = self.g[p][b];
            self.g[p][a] ^= gb;
        }
    }

    /// U_C ← U_C · CZ on `a`,`b`. Conjugation: `X_a → X_a Z_b`,
    /// `X_b → X_b Z_a`, with a `(−1)` when the image carries both X's.
    fn right_cz(&mut self, a: usize, b: usize) {
        for p in 0..self.n {
            if self.f[p][a] && self.f[p][b] {
                self.gamma[p] = (self.gamma[p] + 2) % 4;
            }
            let fa = self.f[p][a];
            let fb = self.f[p][b];
            self.m[p][b] ^= fa;
            self.m[p][a] ^= fb;
        }
    }

    // ── Left multiplication (physical gates) ────────────────────────────

    /// Left `S_q`: image of `X_q` gains `i³ Z`-image of `q` (`S† X S = i³XZ`).
    pub(crate) fn left_s(&mut self, q: usize) {
        self.gamma[q] = (self.gamma[q] + 3) % 4;
        let gq = self.g[q].clone();
        for (mk, &gk) in self.m[q].iter_mut().zip(&gq) {
            *mk ^= gk;
        }
    }

    /// Left `S†_q` (`S X S† = i X Z`).
    pub(crate) fn left_sdg(&mut self, q: usize) {
        self.gamma[q] = (self.gamma[q] + 1) % 4;
        let gq = self.g[q].clone();
        for (mk, &gk) in self.m[q].iter_mut().zip(&gq) {
            *mk ^= gk;
        }
    }

    /// Left CX (control `q`, target `r`): `X_q → X_q X_r`, `Z_r → Z_q Z_r`.
    pub(crate) fn left_cx(&mut self, q: usize, r: usize) {
        let mut cross = 0u32;
        for k in 0..self.n {
            if self.m[q][k] && self.f[r][k] {
                cross += 1;
            }
        }
        self.gamma[q] = (self.gamma[q] + self.gamma[r] + 2 * (cross % 2) as u8) % 4;
        let (fr, mr) = (self.f[r].clone(), self.m[r].clone());
        for k in 0..self.n {
            self.f[q][k] ^= fr[k];
            self.m[q][k] ^= mr[k];
        }
        let gq = self.g[q].clone();
        for (gk, &gqk) in self.g[r].iter_mut().zip(&gq) {
            *gk ^= gqk;
        }
    }

    /// Left CZ on `q`,`r`: `X_q → X_q Z_r`, `X_r → X_r Z_q`.
    pub(crate) fn left_cz(&mut self, q: usize, r: usize) {
        let gr = self.g[r].clone();
        for (mk, &gk) in self.m[q].iter_mut().zip(&gr) {
            *mk ^= gk;
        }
        let gq = self.g[q].clone();
        for (mk, &gk) in self.m[r].iter_mut().zip(&gq) {
            *mk ^= gk;
        }
    }

    /// Left CY (control `q`, target `r`): `(I⊗S)·CX·(I⊗S†)` exactly.
    pub(crate) fn left_cy(&mut self, q: usize, r: usize) {
        self.left_sdg(r);
        self.left_cx(q, r);
        self.left_s(r);
    }

    /// Left Pauli `X_q`: the image `i^{γ_q} X^{F_q} Z^{M_q}` pushes through
    /// `U_H` onto |s⟩ (phase into ω, X-part flips `s`).
    pub(crate) fn left_x(&mut self, q: usize) {
        let (f, m) = (self.f[q].clone(), self.m[q].clone());
        let (ph, t) = self.pauli_behind_h(self.gamma[q], &f, &m);
        self.omega *= ipow(ph);
        self.s = t;
    }

    /// Left Pauli `Z_q` (image `Z^{G_q}`). Completes the `left_*` Clifford
    /// family alongside [`Self::left_x`]; currently exercised only in tests
    /// (Z propagation never gates the sum tier), hence `allow(dead_code)`.
    #[allow(dead_code)]
    pub(crate) fn left_z(&mut self, q: usize) {
        let g = self.g[q].clone();
        let zeros = vec![false; self.n];
        let (ph, t) = self.pauli_behind_h(0, &zeros, &g);
        self.omega *= ipow(ph);
        self.s = t;
    }

    /// Left Hadamard on `q` — the O(n²) case. `H_q = (X_q + Z_q)/√2`
    /// splits the state into two basis-state branches behind `U_H`; equal
    /// branches fold into ω, distinct ones run the desuperposition.
    pub(crate) fn left_h(&mut self, q: usize) {
        // X branch: i^{γ_q} X^{F_q} Z^{M_q} pushed onto |s⟩.
        let (fq, mq) = (self.f[q].clone(), self.m[q].clone());
        let (pa, ta) = self.pauli_behind_h(self.gamma[q], &fq, &mq);
        // Z branch: Z^{G_q} pushed onto |s⟩.
        let gq = self.g[q].clone();
        let zeros = vec![false; self.n];
        let (pb, tb) = self.pauli_behind_h(0, &zeros, &gq);

        if ta == tb {
            // (i^{pa} + i^{pb})/√2 is a unit scalar (the branches can never
            // cancel or reinforce: H is unitary).
            let w = (ipow(pa) + ipow(pb)) * std::f64::consts::FRAC_1_SQRT_2;
            debug_assert!((w.norm() - 1.0).abs() < 1e-9, "H branch fold not unit");
            self.omega *= w;
            self.s = ta;
        } else {
            self.omega *= ipow(pa);
            let d = (pb + 4 - pa) % 4;
            self.desuperpose(ta, tb, d);
        }
    }

    /// Rewrite `U_H (|t⟩ + i^d |u⟩)/√2` (t ≠ u) back into CH form:
    /// right-multiplied control gates onto `U_C` clear all differing bits
    /// but a pivot, whose 1-qubit superposition folds into `v`/`s`/ω.
    fn desuperpose(&mut self, t: Vec<bool>, u: Vec<bool>, d: u8) {
        let n = self.n;
        let mut t = t;
        let mut d = d;
        let diff: Vec<usize> = (0..n).filter(|&j| t[j] != u[j]).collect();
        debug_assert!(!diff.is_empty());

        // Pivot: prefer a qubit outside the Hadamard layer — it can clear
        // differing bits of either kind (CX for v=0, CZ for v=1). A v=1
        // pivot can only clear v=1 bits, but that's all that remains when
        // no v=0 difference exists.
        let q = *diff.iter().find(|&&j| !self.v[j]).unwrap_or(&diff[0]);
        for &j in diff.iter().filter(|&&j| j != q) {
            // Right-multiplied gate whose action behind U_H is x_j ^= x_q
            // (basis-state only — no phases), merging bit j's difference
            // into the pivot's.
            match (self.v[q], self.v[j]) {
                (false, false) => self.right_cx(q, j),
                (false, true) => self.right_cz(q, j),
                (true, true) => self.right_cx(j, q),
                (true, false) => unreachable!("pivot prefers v=0"),
            }
            t[j] ^= t[q];
        }

        // Single difference at the pivot: |a⟩ + i^d |1−a⟩ (behind H_q if
        // v_q). Normalize to a = 0: |1⟩ + i^d|0⟩ = i^d (|0⟩ + i^{−d}|1⟩).
        if t[q] {
            self.omega *= ipow(d);
            d = (4 - d) % 4;
        }
        if !self.v[q] {
            // |0⟩ + i^d|1⟩ = √2 · S^{d odd} H |s_q⟩ with s_q = (d ≥ 2).
            if d % 2 == 1 {
                self.right_s(q);
            }
            self.v[q] = true;
            t[q] = d >= 2;
        } else {
            // H(|0⟩ + i^d|1⟩)/√2: even d collapses the Hadamard; odd d
            // re-expresses as e^{±iπ/4}·S H |s_q⟩.
            match d {
                0 => {
                    self.v[q] = false;
                    t[q] = false;
                }
                2 => {
                    self.v[q] = false;
                    t[q] = true;
                }
                1 => {
                    self.right_s(q);
                    t[q] = true;
                    let f = std::f64::consts::FRAC_1_SQRT_2;
                    self.omega *= Complex::new(f, f); // e^{iπ/4}
                }
                _ => {
                    self.right_s(q);
                    t[q] = false;
                    let f = std::f64::consts::FRAC_1_SQRT_2;
                    self.omega *= Complex::new(f, -f); // e^{−iπ/4}
                }
            }
        }
        self.s = t;
    }

    // ── Readouts ────────────────────────────────────────────────────────

    /// ⟨x|φ⟩, exactly (including ω), in O(n²).
    pub(crate) fn amplitude(&self, x: &[bool]) -> Complex<f64> {
        // ⟨x| U_C = ⟨0| (U_C† X^x U_C) since U_C fixes |0…0⟩.
        let zeros = vec![false; self.n];
        let (ph, xv, zv) = self.conj_pauli(x, &zeros);
        let (ph, t) = self.pauli_behind_h(ph, &xv, &zv);
        // ⟨0|U_H|t⟩ vanishes unless t is 0 outside the Hadamard layer.
        let mut hcount = 0u32;
        for j in 0..self.n {
            if self.v[j] {
                hcount += 1;
            } else if t[j] {
                return Complex::new(0.0, 0.0);
            }
        }
        self.omega * ipow(ph) * std::f64::consts::FRAC_1_SQRT_2.powi(hcount as i32)
    }

    /// Project qubit `q` onto Z-outcome `b`: |φ⟩ ← Π_b|φ⟩ *unnormalized*
    /// (a `Half` projection leaves ω scaled by 1/√2; `Zero` sets ω = 0).
    pub(crate) fn project_z(&mut self, q: usize, b: bool) -> Projection {
        // Z_q|φ⟩ = ε ω U_C U_H |s ⊕ t⟩ with the image Z^{G_q} behind U_H.
        let gq = self.g[q].clone();
        let zeros = vec![false; self.n];
        let (ph, t) = self.pauli_behind_h(0, &zeros, &gq);
        debug_assert!(ph % 2 == 0, "Z image has non-real phase");
        if t == self.s {
            // Diagonal: eigenvalue ε = i^{ph}; outcome is determined.
            let matches = (ph == 0) != b;
            if matches {
                Projection::Deterministic
            } else {
                self.omega = Complex::new(0.0, 0.0);
                Projection::Zero
            }
        } else {
            // Π_b|φ⟩ = (ω/√2) U_C U_H (|s⟩ + i^d|s⊕t⟩)/√2 with
            // i^d = (−1)^b ε.
            let d = (ph + if b { 2 } else { 0 }) % 4;
            self.omega *= std::f64::consts::FRAC_1_SQRT_2;
            let s = self.s.clone();
            self.desuperpose(s, t, d);
            Projection::Half
        }
    }

    /// ⟨self|other⟩ — both ω's included — in O(n³):
    /// `⟨φ|ψ⟩ = ω̄_φ ⟨s_φ| U_Hφ (U_Cφ† U_Cψ) U_Hψ |s_ψ⟩`, computed by
    /// building the composite control-type Clifford's tables, applying
    /// φ's Hadamard layer as left gates, and reading one amplitude.
    pub(crate) fn inner(&self, other: &ChForm) -> Complex<f64> {
        debug_assert_eq!(self.n, other.n);
        let n = self.n;

        // Inverse action B(P) = U_Cφ P U_Cφ†, from F⁻¹ = Gᵀ, G⁻¹ = Fᵀ.
        // B(Z_p) = Z^{col_p(F)}; B(X_p) = i^{−δ} X^{col_p(G)} Z^{col_p(G)·M·Fᵀ}
        // with δ fixed numerically so that A(B(X_p)) = X_p.
        let mut chi = ChForm {
            n,
            f: vec![vec![false; n]; n],
            g: vec![vec![false; n]; n],
            m: vec![vec![false; n]; n],
            gamma: vec![0; n],
            v: other.v.clone(),
            s: other.s.clone(),
            omega: other.omega,
        };
        for p in 0..n {
            // x-part of B(X_p): column p of G.
            let bx: Vec<bool> = (0..n).map(|j| self.g[j][p]).collect();
            // z-part: (bx · M) · Fᵀ.
            let mut bm = vec![false; n];
            for j in 0..n {
                if bx[j] {
                    for k in 0..n {
                        bm[k] ^= self.m[j][k];
                    }
                }
            }
            let bz: Vec<bool> = (0..n)
                .map(|k| (0..n).fold(false, |acc, j| acc ^ (bm[j] && self.f[k][j])))
                .collect();
            // Phase: A(X^bx Z^bz) must be i^δ X_p.
            let (delta, ax, az) = self.conj_pauli(&bx, &bz);
            debug_assert!(
                ax.iter().enumerate().all(|(j, &b)| b == (j == p)) && az.iter().all(|&b| !b),
                "inverse X-image does not conjugate back to X_p",
            );
            let gb = (4 - delta) % 4;
            // Compose with ψ's action: A_V(X_p) = A_ψ(B(X_p)).
            let (ph2, xv2, zv2) = other.conj_pauli(&bx, &bz);
            chi.gamma[p] = (gb + ph2) % 4;
            chi.f[p] = xv2;
            chi.m[p] = zv2;
            // Z rows: A_V(Z_p) = A_ψ(Z^{col_p(F)}).
            for j in 0..n {
                if self.f[j][p] {
                    for k in 0..n {
                        chi.g[p][k] ^= other.g[j][k];
                    }
                }
            }
        }

        // ⟨s_φ| U_Hφ |χ⟩: apply φ's Hadamards, then one amplitude.
        for j in 0..n {
            if self.v[j] {
                chi.left_h(j);
            }
        }
        // χ's amplitude already includes ω_ψ; ω̄_φ joins it here.
        self.omega.conj() * chi.amplitude(&self.s)
    }
}

/// `ChForm` is a phase-exact Clifford sink: `phase` scales ω.
impl CliffordSink for ChForm {
    fn h(&mut self, q: usize) {
        self.left_h(q);
    }
    fn s(&mut self, q: usize) {
        self.left_s(q);
    }
    fn cnot(&mut self, c: usize, t: usize) {
        self.left_cx(c, t);
    }
    fn cz(&mut self, c: usize, t: usize) {
        self.left_cz(c, t);
    }
    fn cy(&mut self, c: usize, t: usize) {
        self.left_cy(c, t);
    }
    fn phase(&mut self, w: Complex<f64>) {
        self.omega *= w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{GateModifiers, QuantumBackend, apply_gphase};
    use crate::sim::StateVectorSim;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

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
    }

    /// A textbook-phase Clifford gate (exact matrices, no `U` phase
    /// convention involved on the CH side).
    #[derive(Clone, Copy, Debug)]
    enum G {
        H(usize),
        S(usize),
        Sdg(usize),
        X(usize),
        Z(usize),
        Cx(usize, usize),
        Cz(usize, usize),
        Cy(usize, usize),
    }

    fn random_word(n: usize, len: usize, seed: u64) -> Vec<G> {
        let mut r = Lcg(seed.wrapping_add(1));
        let q = |r: &mut Lcg| r.below(n as u32) as usize;
        let pair = |r: &mut Lcg| {
            let c = r.below(n as u32) as usize;
            let mut t = r.below(n as u32) as usize;
            if t == c {
                t = (t + 1) % n;
            }
            (c, t)
        };
        (0..len)
            .map(|_| match r.below(8) {
                0 => G::H(q(&mut r)),
                1 => G::S(q(&mut r)),
                2 => G::Sdg(q(&mut r)),
                3 => G::X(q(&mut r)),
                4 => G::Z(q(&mut r)),
                5 => {
                    let (c, t) = pair(&mut r);
                    G::Cx(c, t)
                }
                6 => {
                    let (c, t) = pair(&mut r);
                    G::Cz(c, t)
                }
                _ => {
                    let (c, t) = pair(&mut r);
                    G::Cy(c, t)
                }
            })
            .collect()
    }

    fn apply_ch(ch: &mut ChForm, g: G) {
        match g {
            G::H(q) => ch.left_h(q),
            G::S(q) => ch.left_s(q),
            G::Sdg(q) => ch.left_sdg(q),
            G::X(q) => ch.left_x(q),
            G::Z(q) => ch.left_z(q),
            G::Cx(c, t) => ch.left_cx(c, t),
            G::Cz(c, t) => ch.left_cz(c, t),
            G::Cy(c, t) => ch.left_cy(c, t),
        }
    }

    /// Apply the *textbook* gate to the dense sim: the built-in `U` carries
    /// an `e^{iθ/2}` phase, cancelled exactly by a trailing `gphase` where
    /// needed (mirroring stdgates.inc).
    fn apply_sv(sv: &mut StateVectorSim<f64>, g: G) {
        let none = GateModifiers::none();
        let ctrl = |c: usize| GateModifiers::ctrl(&[c as u32]);
        match g {
            G::H(q) => {
                sv.u(q as u32, FRAC_PI_2, 0.0, PI, &none).unwrap();
                sv.gphase(-FRAC_PI_4).unwrap();
            }
            G::S(q) => sv.u(q as u32, 0.0, 0.0, FRAC_PI_2, &none).unwrap(),
            G::Sdg(q) => sv.u(q as u32, 0.0, 0.0, -FRAC_PI_2, &none).unwrap(),
            G::X(q) => {
                sv.u(q as u32, PI, 0.0, PI, &none).unwrap();
                sv.gphase(-FRAC_PI_2).unwrap();
            }
            G::Z(q) => sv.u(q as u32, 0.0, 0.0, PI, &none).unwrap(),
            G::Cx(c, t) => {
                sv.u(t as u32, PI, 0.0, PI, &ctrl(c)).unwrap();
                apply_gphase(sv, -FRAC_PI_2, &ctrl(c)).unwrap();
            }
            G::Cz(c, t) => sv.u(t as u32, 0.0, 0.0, PI, &ctrl(c)).unwrap(),
            G::Cy(c, t) => {
                sv.u(t as u32, PI, FRAC_PI_2, FRAC_PI_2, &ctrl(c)).unwrap();
                apply_gphase(sv, -FRAC_PI_2, &ctrl(c)).unwrap();
            }
        }
    }

    fn ch_amplitudes(ch: &ChForm, n: usize) -> Vec<Complex<f64>> {
        (0..1usize << n)
            .map(|i| {
                let bits: Vec<bool> = (0..n).map(|q| i >> q & 1 == 1).collect();
                ch.amplitude(&bits)
            })
            .collect()
    }

    fn assert_close(a: &[Complex<f64>], b: &[Complex<f64>], what: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).norm() < 1e-9,
                "{what}: amplitude {i} mismatch: {x} vs {y}",
            );
        }
    }

    #[test]
    fn zero_state_is_ground() {
        let ch = ChForm::zero(3);
        let amps = ch_amplitudes(&ch, 3);
        assert!((amps[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
        for a in &amps[1..] {
            assert!(a.norm() < 1e-12);
        }
    }

    #[test]
    fn single_gates_have_exact_phases() {
        // S on |+⟩: H then S ⇒ (|0⟩ + i|1⟩)/√2 — phase must be exactly i.
        let mut ch = ChForm::zero(1);
        ch.left_h(0);
        ch.left_s(0);
        let amps = ch_amplitudes(&ch, 1);
        let f = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0] - Complex::new(f, 0.0)).norm() < 1e-12);
        assert!((amps[1] - Complex::new(0.0, f)).norm() < 1e-12);

        // X via HZH must land exactly on |1⟩ with amplitude +1.
        let mut ch = ChForm::zero(1);
        ch.left_h(0);
        ch.left_z(0);
        ch.left_h(0);
        let amps = ch_amplitudes(&ch, 1);
        assert!(amps[0].norm() < 1e-12);
        assert!((amps[1] - Complex::new(1.0, 0.0)).norm() < 1e-12);

        // Y = i·X·Z as left ops: Z then X then ω·i ⇒ |1⟩ with amplitude i.
        let mut ch = ChForm::zero(1);
        ch.left_z(0);
        ch.left_x(0);
        ch.scale(Complex::new(0.0, 1.0));
        let amps = ch_amplitudes(&ch, 1);
        assert!((amps[1] - Complex::new(0.0, 1.0)).norm() < 1e-12);
    }

    /// The make-or-break test: random Clifford words must match the dense
    /// simulator's full amplitude vector exactly — global phase included.
    #[test]
    fn random_words_match_dense_exactly() {
        pollster::block_on(async {
            let n = 4usize;
            for seed in 0..25u64 {
                let word = random_word(n, 40, seed);
                let mut ch = ChForm::zero(n);
                let mut sv = StateVectorSim::<f64>::try_zeroed(n as u32, 0).unwrap();
                for &g in &word {
                    apply_ch(&mut ch, g);
                    apply_sv(&mut sv, g);
                }
                let a = ch_amplitudes(&ch, n);
                let b = sv.amplitudes().await.unwrap();
                assert_close(&a, &b, &format!("seed {seed}"));
            }
        })
    }

    /// Compiler-lowered `U`/`gphase` calls through the shared detection
    /// path ([`crate::clifford::apply_clifford_u`]) must also be exact —
    /// this exercises `word_phase` compensation on top of the CH form.
    #[test]
    fn lowered_u_calls_match_dense_exactly() {
        pollster::block_on(async {
            use crate::clifford::apply_clifford_u;
            let n = 4usize;
            for seed in 100..115u64 {
                let word = random_word(n, 30, seed);
                let mut ch = ChForm::zero(n);
                let mut sv = StateVectorSim::<f64>::try_zeroed(n as u32, 0).unwrap();
                let none = GateModifiers::none();
                let ctrl = |c: usize| GateModifiers::ctrl(&[c as u32]);
                for &g in &word {
                    // Route the same lowered (θ,φ,λ)/gphase calls to both.
                    let calls: Vec<(usize, f64, f64, f64, GateModifiers)> = match g {
                        G::H(q) => vec![(q, FRAC_PI_2, 0.0, PI, none.clone())],
                        G::S(q) => vec![(q, 0.0, 0.0, FRAC_PI_2, none.clone())],
                        G::Sdg(q) => vec![(q, 0.0, 0.0, -FRAC_PI_2, none.clone())],
                        G::X(q) => vec![(q, PI, 0.0, PI, none.clone())],
                        G::Z(q) => vec![(q, 0.0, 0.0, PI, none.clone())],
                        G::Cx(c, t) => vec![(t, PI, 0.0, PI, ctrl(c))],
                        G::Cz(c, t) => vec![(t, 0.0, 0.0, PI, ctrl(c))],
                        G::Cy(c, t) => vec![(t, PI, FRAC_PI_2, FRAC_PI_2, ctrl(c))],
                    };
                    for (t, th, ph, la, m) in calls {
                        assert!(apply_clifford_u(&mut ch, t as u32, th, ph, la, &m));
                        sv.u(t as u32, th, ph, la, &m).unwrap();
                    }
                    // An occasional uncontrolled gphase to exercise ω.
                    if let G::S(q) = g {
                        let alpha = 0.25 + q as f64;
                        ch.phase(Complex::from_polar(1.0, alpha));
                        sv.gphase(alpha).unwrap();
                    }
                }
                let a = ch_amplitudes(&ch, n);
                let b = sv.amplitudes().await.unwrap();
                assert_close(&a, &b, &format!("seed {seed}"));
            }
        })
    }

    /// ⟨φ|φ⟩ = 1 and ⟨φ|ψ⟩ matches the dense dot product.
    #[test]
    fn inner_product_matches_dense() {
        pollster::block_on(async {
            let n = 4usize;
            for seed in 0..15u64 {
                let (w1, w2) = (random_word(n, 30, seed), random_word(n, 30, seed + 500));
                let mut ch1 = ChForm::zero(n);
                let mut ch2 = ChForm::zero(n);
                let mut sv1 = StateVectorSim::<f64>::try_zeroed(n as u32, 0).unwrap();
                let mut sv2 = StateVectorSim::<f64>::try_zeroed(n as u32, 0).unwrap();
                for &g in &w1 {
                    apply_ch(&mut ch1, g);
                    apply_sv(&mut sv1, g);
                }
                for &g in &w2 {
                    apply_ch(&mut ch2, g);
                    apply_sv(&mut sv2, g);
                }
                let dense: Complex<f64> = sv1
                    .amplitudes()
                    .await
                    .unwrap()
                    .iter()
                    .zip(sv2.amplitudes().await.unwrap())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                let got = ch1.inner(&ch2);
                assert!(
                    (got - dense).norm() < 1e-9,
                    "seed {seed}: inner {got} vs dense {dense}",
                );
                let self_ip = ch1.inner(&ch1);
                assert!(
                    (self_ip - Complex::new(1.0, 0.0)).norm() < 1e-9,
                    "seed {seed}: ⟨φ|φ⟩ = {self_ip}",
                );
            }
        })
    }

    /// `project_z` matches the dense projector on every qubit/outcome and
    /// reports the right determinism class.
    #[test]
    fn projection_matches_dense() {
        pollster::block_on(async {
            let n = 4usize;
            let mut classes = [0u32; 3];
            for seed in 0..15u64 {
                let word = random_word(n, 30, seed);
                for q in 0..n {
                    for b in [false, true] {
                        let mut ch = ChForm::zero(n);
                        let mut sv = StateVectorSim::<f64>::try_zeroed(n as u32, 0).unwrap();
                        for &g in &word {
                            apply_ch(&mut ch, g);
                            apply_sv(&mut sv, g);
                        }
                        let pre = sv.amplitudes().await.unwrap();
                        let bit = 1usize << q;
                        // Unnormalized dense projection.
                        let dense: Vec<Complex<f64>> = pre
                            .iter()
                            .enumerate()
                            .map(|(i, a)| {
                                if (i & bit != 0) == b {
                                    *a
                                } else {
                                    Complex::new(0.0, 0.0)
                                }
                            })
                            .collect();
                        let p: f64 = dense.iter().map(|a| a.norm_sqr()).sum();
                        let class = ch.project_z(q, b);
                        match class {
                            Projection::Deterministic => {
                                assert!((p - 1.0).abs() < 1e-9);
                                classes[0] += 1;
                            }
                            Projection::Zero => {
                                assert!(p < 1e-9);
                                classes[1] += 1;
                            }
                            Projection::Half => {
                                assert!((p - 0.5).abs() < 1e-9);
                                classes[2] += 1;
                            }
                        }
                        let got = ch_amplitudes(&ch, n);
                        assert_close(&got, &dense, &format!("seed {seed} q{q} b{b}"));
                    }
                }
            }
            assert!(
                classes.iter().all(|&c| c > 0),
                "all projection classes exercised: {classes:?}",
            );
        })
    }
}
