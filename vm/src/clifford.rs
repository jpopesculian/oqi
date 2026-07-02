//! Clifford detection and decomposition, shared by the stabilizer tableau
//! and the sum-over-Cliffords backends.
//!
//! Gates arrive as `U(θ,φ,λ)`/`gphase` matrices, so Clifford-ness is
//! detected from the gate's matrix: a single-qubit gate is Clifford iff it
//! conjugates both Paulis to signed Paulis. A recognized Clifford is
//! realized as an `{H, S}` word (via a BFS table of the 24 single-qubit
//! Cliffords mod phase) and applied to a [`CliffordSink`]. The word matches
//! the gate only up to a global phase; phase-exact sinks (the CH-form sum)
//! recover it via [`word_phase`], the tableau ignores it.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::OnceLock;

use num_complex::Complex;
use oqi_quantum::{Gate, Unitary};

use crate::backend::GateModifiers;

// ── 2×2 complex matrix helpers (for Clifford detection) ─────────────────

pub(crate) type M2 = [[Complex<f64>; 2]; 2];

pub(crate) fn cpx(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}
pub(crate) fn pauli_x() -> M2 {
    [
        [cpx(0.0, 0.0), cpx(1.0, 0.0)],
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
    ]
}
pub(crate) fn pauli_y() -> M2 {
    [
        [cpx(0.0, 0.0), cpx(0.0, -1.0)],
        [cpx(0.0, 1.0), cpx(0.0, 0.0)],
    ]
}
pub(crate) fn pauli_z() -> M2 {
    [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), cpx(-1.0, 0.0)],
    ]
}

pub(crate) fn matmul(a: &M2, b: &M2) -> M2 {
    let mut o = [[cpx(0.0, 0.0); 2]; 2];
    for (i, orow) in o.iter_mut().enumerate() {
        for (j, oij) in orow.iter_mut().enumerate() {
            *oij = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    o
}

pub(crate) fn dagger(a: &M2) -> M2 {
    [
        [a[0][0].conj(), a[1][0].conj()],
        [a[0][1].conj(), a[1][1].conj()],
    ]
}

/// `m · p · m†`.
pub(crate) fn conjugate(m: &M2, p: &M2) -> M2 {
    matmul(&matmul(m, p), &dagger(m))
}

pub(crate) fn approx_eq(a: &M2, b: &M2) -> bool {
    const TOL: f64 = 1e-6;
    (0..2).all(|i| (0..2).all(|j| (a[i][j] - b[i][j]).norm() < TOL))
}

pub(crate) fn scaled(p: &M2, s: f64) -> M2 {
    [[p[0][0] * s, p[0][1] * s], [p[1][0] * s, p[1][1] * s]]
}

/// A signed single-qubit Pauli: axis (0=X, 1=Y, 2=Z) and a negative flag.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SignedPauli {
    axis: u8,
    neg: bool,
}

impl SignedPauli {
    fn key(self) -> u8 {
        self.axis * 2 + self.neg as u8
    }
}

/// Classify a 2×2 matrix as a signed Pauli (up to numerical tolerance), or
/// `None` if it isn't one (and so the gate is non-Clifford).
pub(crate) fn classify(m: &M2) -> Option<SignedPauli> {
    for (axis, base) in [(0u8, pauli_x()), (1, pauli_y()), (2, pauli_z())] {
        if approx_eq(m, &base) {
            return Some(SignedPauli { axis, neg: false });
        }
        if approx_eq(m, &scaled(&base, -1.0)) {
            return Some(SignedPauli { axis, neg: true });
        }
    }
    None
}

// ── Single-qubit Clifford decomposition into {H, S} ─────────────────────

#[derive(Clone, Copy)]
pub(crate) enum Prim {
    H,
    S,
}

/// Conjugate a signed Pauli by a primitive (combinatorial — no floats).
/// `H`: X↔Z (sign kept), Y→−Y. `S`: X→Y (kept), Y→−X, Z→Z.
fn conj_prim(prim: Prim, p: SignedPauli) -> SignedPauli {
    let SignedPauli { axis, neg } = p;
    match (prim, axis) {
        (Prim::H, 0) => SignedPauli { axis: 2, neg },
        (Prim::H, 2) => SignedPauli { axis: 0, neg },
        (Prim::H, 1) => SignedPauli { axis: 1, neg: !neg },
        (Prim::S, 0) => SignedPauli { axis: 1, neg },
        (Prim::S, 1) => SignedPauli { axis: 0, neg: !neg },
        (Prim::S, 2) => SignedPauli { axis: 2, neg },
        _ => unreachable!(),
    }
}

/// BFS table mapping a Clifford's `(image of X, image of Z)` to a primitive
/// word realizing it. Built once; the single-qubit Clifford group (mod phase)
/// has 24 elements, all reachable from `⟨H, S⟩`.
fn clifford_table() -> &'static HashMap<u16, Vec<Prim>> {
    static TABLE: OnceLock<HashMap<u16, Vec<Prim>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let start = (
            SignedPauli {
                axis: 0,
                neg: false,
            }, // X → +X
            SignedPauli {
                axis: 2,
                neg: false,
            }, // Z → +Z
        );
        let key = |ix: SignedPauli, iz: SignedPauli| (ix.key() as u16) * 6 + iz.key() as u16;
        let mut table: HashMap<u16, Vec<Prim>> = HashMap::new();
        table.insert(key(start.0, start.1), Vec::new());
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some((ix, iz)) = queue.pop_front() {
            let word = table[&key(ix, iz)].clone();
            for prim in [Prim::H, Prim::S] {
                let nx = conj_prim(prim, ix);
                let nz = conj_prim(prim, iz);
                if let std::collections::hash_map::Entry::Vacant(e) = table.entry(key(nx, nz)) {
                    let mut w = word.clone();
                    w.push(prim);
                    e.insert(w);
                    queue.push_back((nx, nz));
                }
            }
        }
        table
    })
}

fn clifford_word(ix: SignedPauli, iz: SignedPauli) -> Option<&'static [Prim]> {
    let key = (ix.key() as u16) * 6 + iz.key() as u16;
    clifford_table().get(&key).map(|w| w.as_slice())
}

/// The effective single-qubit matrix `U(θ,φ,λ)^power`.
pub(crate) fn effective_matrix(theta: f64, phi: f64, lambda: f64, power: f64) -> M2 {
    let base = Gate::new(Unitary::<f64>::new(theta, phi, lambda));
    if power != 1.0 {
        base.pow(power).matrix()
    } else {
        base.matrix()
    }
}

pub(crate) fn identity() -> M2 {
    [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), cpx(1.0, 0.0)],
    ]
}

/// The phase gate `diag(1, e^{iα})` (`= U(0,0,α)`), used to express a
/// relative phase on a control qubit.
pub(crate) fn phase_matrix(alpha: f64) -> M2 {
    [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), Complex::from_polar(1.0, alpha)],
    ]
}

/// The primitive `{H, S}` word realizing a single-qubit Clifford `mat`
/// (global phase ignored, since it cancels under conjugation), or `None` if
/// `mat` isn't Clifford.
pub(crate) fn single_qubit_word(mat: &M2) -> Option<Vec<Prim>> {
    let ix = classify(&conjugate(mat, &pauli_x()))?;
    let iz = classify(&conjugate(mat, &pauli_z()))?;
    clifford_word(ix, iz).map(<[Prim]>::to_vec)
}

/// Decompose `mat` as `λ · base` with `base ∈ {I, X, Y, Z}` (so `mat` is a
/// Pauli up to a global phase `λ`), or `None` otherwise. `axis` is `None` for
/// the identity base.
pub(crate) fn phased_pauli(mat: &M2) -> Option<(Option<u8>, f64)> {
    const TOL: f64 = 1e-6;
    let bases: [(Option<u8>, M2); 4] = [
        (None, identity()),
        (Some(0), pauli_x()),
        (Some(1), pauli_y()),
        (Some(2), pauli_z()),
    ];
    for (axis, base) in bases {
        // Paulis and I are involutions, so mat = λ·base ⇔ mat·base = λ·I.
        let prod = matmul(mat, &base);
        let lambda = prod[0][0];
        let lam_i = [[lambda, cpx(0.0, 0.0)], [cpx(0.0, 0.0), lambda]];
        if (lambda.norm() - 1.0).abs() < TOL && approx_eq(&prod, &lam_i) {
            return Some((axis, lambda.arg()));
        }
    }
    None
}

/// The exact 2×2 matrix of an `{H, S}` word, with the textbook phases
/// `H = (X+Z)/√2`, `S = diag(1, i)`. Words apply first-element-first to the
/// state, so the product is right-to-left.
pub(crate) fn word_matrix(word: &[Prim]) -> M2 {
    let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
    let h = [
        [cpx(sqrt_half, 0.0), cpx(sqrt_half, 0.0)],
        [cpx(sqrt_half, 0.0), cpx(-sqrt_half, 0.0)],
    ];
    let s = [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), cpx(0.0, 1.0)],
    ];
    let mut m = identity();
    for prim in word {
        let p = match prim {
            Prim::H => &h,
            Prim::S => &s,
        };
        m = matmul(p, &m);
    }
    m
}

/// The scalar `μ` with `mat = μ · word_matrix(word)`. The BFS table matches
/// a Clifford only mod global phase; phase-exact sinks (the CH-form sum)
/// need `μ` to stay faithful. Debug-asserts proportionality.
pub(crate) fn word_phase(mat: &M2, word: &[Prim]) -> Complex<f64> {
    let wm = word_matrix(word);
    // Any entry with non-negligible magnitude gives the ratio; every entry
    // of a Clifford word matrix is 0 or of magnitude ≥ 1/2.
    let (i, j) = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .find(|&(i, j)| wm[i][j].norm() > 0.1)
        .expect("word matrix is unitary, so has a large entry");
    let mu = mat[i][j] / wm[i][j];
    debug_assert!(
        approx_eq(
            mat,
            &[
                [wm[0][0] * mu, wm[0][1] * mu],
                [wm[1][0] * mu, wm[1][1] * mu]
            ]
        ),
        "matrix is not proportional to its Clifford word",
    );
    mu
}

// ── Applying detected Cliffords to a sink ────────────────────────────────

/// A target for Clifford application. The stabilizer tableau (phase-blind)
/// and the CH-form sum (phase-exact) both implement it, so one detection
/// path drives both.
pub(crate) trait CliffordSink {
    fn h(&mut self, q: usize);
    fn s(&mut self, q: usize);
    fn cnot(&mut self, c: usize, t: usize);
    fn cz(&mut self, c: usize, t: usize);
    fn cy(&mut self, c: usize, t: usize);
    /// Multiply the state's global scalar by `w`. No-op for sinks that
    /// don't track phase (the tableau).
    fn phase(&mut self, _w: Complex<f64>) {}
}

fn apply_word(sink: &mut impl CliffordSink, q: usize, word: &[Prim]) {
    for prim in word {
        match prim {
            Prim::H => sink.h(q),
            Prim::S => sink.s(q),
        }
    }
}

/// Apply a single-qubit `mat` to the sink, or `false` if non-Clifford.
pub(crate) fn apply_single_qubit(sink: &mut impl CliffordSink, q: usize, mat: &M2) -> bool {
    match single_qubit_word(mat) {
        Some(word) => {
            sink.phase(word_phase(mat, &word));
            apply_word(sink, q, &word);
            true
        }
        None => false,
    }
}

/// Apply a controlled single-qubit `mat` (one control `c`, target `t`), or
/// `false` if non-Clifford. `ctrl@(λ·P)` = `(controlled-P)` followed by the
/// relative phase `arg(λ)` on the control (a single-qubit phase gate), which
/// is Clifford iff that phase is a multiple of π/2.
pub(crate) fn apply_controlled(sink: &mut impl CliffordSink, c: usize, t: usize, mat: &M2) -> bool {
    let (axis, alpha) = match phased_pauli(mat) {
        Some(v) => v,
        None => return false,
    };
    // The relative phase on the control must itself be Clifford.
    let phase_mat = phase_matrix(alpha);
    let phase_word = match single_qubit_word(&phase_mat) {
        Some(w) => w,
        None => return false,
    };
    match axis {
        None => {} // controlled-(scalar): only the control phase
        Some(0) => sink.cnot(c, t),
        Some(1) => sink.cy(c, t),
        Some(2) => sink.cz(c, t),
        _ => unreachable!(),
    }
    sink.phase(word_phase(&phase_mat, &phase_word));
    apply_word(sink, c, &phase_word);
    true
}

/// Try to apply a `u()` call to the sink as a Clifford. Returns `true` on
/// success, `false` if non-Clifford (caller must fall back).
pub(crate) fn apply_clifford_u(
    sink: &mut impl CliffordSink,
    target: u32,
    theta: f64,
    phi: f64,
    lambda: f64,
    m: &GateModifiers,
) -> bool {
    if !m.neg_controls.is_empty() {
        return false; // negative controls: route to the fallback path
    }
    let mat = effective_matrix(theta, phi, lambda, m.power);
    let t = target as usize;
    match m.controls.as_slice() {
        [] => apply_single_qubit(sink, t, &mat),
        [c] => apply_controlled(sink, *c as usize, t, &mat),
        _ => false, // 2+ controls (e.g. Toffoli): non-Clifford
    }
}

/// Try to apply a `gphase` to the sink as a Clifford. An uncontrolled global
/// phase only scales the global scalar (no effect on a phase-blind sink); a
/// single-control global phase is a phase `g` on that control (Clifford iff
/// `g` is a multiple of π/2).
pub(crate) fn apply_clifford_gphase(
    sink: &mut impl CliffordSink,
    g: f64,
    m: &GateModifiers,
) -> bool {
    if m.controls.is_empty() && m.neg_controls.is_empty() {
        sink.phase(Complex::from_polar(1.0, g));
        return true;
    }
    if m.neg_controls.is_empty() && m.controls.len() == 1 {
        return apply_single_qubit(sink, m.controls[0] as usize, &phase_matrix(g));
    }
    false
}
