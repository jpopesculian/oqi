//! An auto-routing backend: runs Clifford circuits on a stabilizer
//! [`Tableau`](crate::stabilizer::Tableau) (O(n²)/gate) and transparently
//! falls back to the dense [`StateVectorSim`] the moment a non-Clifford gate
//! appears, so any circuit still runs correctly.
//!
//! Gates arrive as `U(θ,φ,λ)`/`gphase` (the compiler lowers `h`/`s`/`cx`
//! before the backend), so Clifford-ness is detected from the gate's matrix:
//! a single-qubit gate is Clifford iff it conjugates both Paulis to signed
//! Paulis. While in stabilizer mode every applied op is logged; on the first
//! non-Clifford op the log is replayed onto a fresh state vector (recorded
//! measurement outcomes forced via [`StateVectorSim::project`]) and execution
//! continues there.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::OnceLock;

use async_trait::async_trait;
use num_complex::Complex;
use oqi_quantum::{Gate, Unitary};

use crate::backend::{GateModifiers, QuantumBackend};
use crate::sim::{Rng, StateVectorSim};
use crate::stabilizer::Tableau;

const SEED: u64 = 0x2545F4914F6CDD1D;

// ── 2×2 complex matrix helpers (for Clifford detection) ─────────────────

type M2 = [[Complex<f64>; 2]; 2];

fn cpx(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}
fn pauli_x() -> M2 {
    [
        [cpx(0.0, 0.0), cpx(1.0, 0.0)],
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
    ]
}
fn pauli_y() -> M2 {
    [
        [cpx(0.0, 0.0), cpx(0.0, -1.0)],
        [cpx(0.0, 1.0), cpx(0.0, 0.0)],
    ]
}
fn pauli_z() -> M2 {
    [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), cpx(-1.0, 0.0)],
    ]
}

fn matmul(a: &M2, b: &M2) -> M2 {
    let mut o = [[cpx(0.0, 0.0); 2]; 2];
    for (i, orow) in o.iter_mut().enumerate() {
        for (j, oij) in orow.iter_mut().enumerate() {
            *oij = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    o
}

fn dagger(a: &M2) -> M2 {
    [
        [a[0][0].conj(), a[1][0].conj()],
        [a[0][1].conj(), a[1][1].conj()],
    ]
}

/// `m · p · m†`.
fn conjugate(m: &M2, p: &M2) -> M2 {
    matmul(&matmul(m, p), &dagger(m))
}

fn approx_eq(a: &M2, b: &M2) -> bool {
    const TOL: f64 = 1e-6;
    (0..2).all(|i| (0..2).all(|j| (a[i][j] - b[i][j]).norm() < TOL))
}

fn scaled(p: &M2, s: f64) -> M2 {
    [[p[0][0] * s, p[0][1] * s], [p[1][0] * s, p[1][1] * s]]
}

/// A signed single-qubit Pauli: axis (0=X, 1=Y, 2=Z) and a negative flag.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SignedPauli {
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
fn classify(m: &M2) -> Option<SignedPauli> {
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
enum Prim {
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
fn effective_matrix(theta: f64, phi: f64, lambda: f64, power: f64) -> M2 {
    let base = Gate::new(Unitary::<f64>::new(theta, phi, lambda));
    if power != 1.0 {
        base.pow(power).matrix()
    } else {
        base.matrix()
    }
}

fn identity() -> M2 {
    [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), cpx(1.0, 0.0)],
    ]
}

/// The phase gate `diag(1, e^{iα})` (`= U(0,0,α)`), used to express a
/// relative phase on a control qubit.
fn phase_matrix(alpha: f64) -> M2 {
    [
        [cpx(1.0, 0.0), cpx(0.0, 0.0)],
        [cpx(0.0, 0.0), Complex::from_polar(1.0, alpha)],
    ]
}

/// The primitive `{H, S}` word realizing a single-qubit Clifford `mat`
/// (global phase ignored, since it cancels under conjugation), or `None` if
/// `mat` isn't Clifford.
fn single_qubit_word(mat: &M2) -> Option<Vec<Prim>> {
    let ix = classify(&conjugate(mat, &pauli_x()))?;
    let iz = classify(&conjugate(mat, &pauli_z()))?;
    clifford_word(ix, iz).map(<[Prim]>::to_vec)
}

/// Decompose `mat` as `λ · base` with `base ∈ {I, X, Y, Z}` (so `mat` is a
/// Pauli up to a global phase `λ`), or `None` otherwise. `axis` is `None` for
/// the identity base.
fn phased_pauli(mat: &M2) -> Option<(Option<u8>, f64)> {
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

fn apply_word(tab: &mut Tableau, q: usize, word: &[Prim]) {
    for prim in word {
        match prim {
            Prim::H => tab.h(q),
            Prim::S => tab.s(q),
        }
    }
}

/// Apply a single-qubit `mat` to the tableau, or `false` if non-Clifford.
fn apply_single_qubit(tab: &mut Tableau, q: usize, mat: &M2) -> bool {
    match single_qubit_word(mat) {
        Some(word) => {
            apply_word(tab, q, &word);
            true
        }
        None => false,
    }
}

/// Apply a controlled single-qubit `mat` (one control `c`, target `t`), or
/// `false` if non-Clifford. `ctrl@(λ·P)` = `(controlled-P)` followed by the
/// relative phase `arg(λ)` on the control (a single-qubit phase gate), which
/// is Clifford iff that phase is a multiple of π/2.
fn apply_controlled(tab: &mut Tableau, c: usize, t: usize, mat: &M2) -> bool {
    let (axis, alpha) = match phased_pauli(mat) {
        Some(v) => v,
        None => return false,
    };
    // The relative phase on the control must itself be Clifford.
    let phase_word = match single_qubit_word(&phase_matrix(alpha)) {
        Some(w) => w,
        None => return false,
    };
    match axis {
        None => {} // controlled-(scalar): only the control phase
        Some(0) => tab.cnot(c, t),
        Some(1) => tab.cy(c, t),
        Some(2) => tab.cz(c, t),
        _ => unreachable!(),
    }
    apply_word(tab, c, &phase_word);
    true
}

/// Try to apply a `u()` call to the tableau as a Clifford. Returns `true` on
/// success, `false` if non-Clifford (caller must fall back).
fn apply_clifford_u(
    tab: &mut Tableau,
    target: u32,
    theta: f64,
    phi: f64,
    lambda: f64,
    m: &GateModifiers,
) -> bool {
    if !m.neg_controls.is_empty() {
        return false; // negative controls: route to the dense sim
    }
    let mat = effective_matrix(theta, phi, lambda, m.power);
    let t = target as usize;
    match m.controls.as_slice() {
        [] => apply_single_qubit(tab, t, &mat),
        [c] => apply_controlled(tab, *c as usize, t, &mat),
        _ => false, // 2+ controls (e.g. Toffoli): non-Clifford
    }
}

/// Try to apply a `gphase` to the tableau as a Clifford. Uncontrolled global
/// phase has no effect on the stabilizer state (the caller logs it for a
/// faithful state-vector readout); a single-control global phase is a phase
/// `g` on that control (Clifford iff `g` is a multiple of π/2).
fn apply_clifford_gphase(tab: &mut Tableau, g: f64, m: &GateModifiers) -> bool {
    if m.controls.is_empty() && m.neg_controls.is_empty() {
        return true; // no tableau effect
    }
    if m.neg_controls.is_empty() && m.controls.len() == 1 {
        return apply_single_qubit(tab, m.controls[0] as usize, &phase_matrix(g));
    }
    false
}

// ── The auto-routing backend ────────────────────────────────────────────

/// A recorded operation, replayed onto a state vector on fallback.
enum Op {
    U {
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        controls: Vec<u32>,
        neg_controls: Vec<u32>,
        power: f64,
    },
    Gphase {
        gamma: f64,
        controls: Vec<u32>,
        neg_controls: Vec<u32>,
        power: f64,
    },
    Measure {
        qubit: u32,
        outcome: bool,
    },
    Reset {
        qubit: u32,
        outcome: bool,
    },
}

enum Mode {
    Stabilizer {
        tab: Tableau,
        log: Vec<Op>,
        rng: Rng,
    },
    StateVector(StateVectorSim<f64>),
}

/// State-vector simulator that starts in stabilizer mode and falls back to a
/// dense state vector on the first non-Clifford gate.
pub struct AutoSim {
    n: u32,
    mode: Mode,
}

impl AutoSim {
    /// A fresh simulator with `num_qubits` qubits in |0…0⟩ (default seed).
    pub fn new(num_qubits: u32) -> Self {
        Self::with_seed(num_qubits, SEED)
    }

    /// A fresh simulator with an explicit RNG seed (for reproducible
    /// measurement sampling in stabilizer mode).
    pub fn with_seed(num_qubits: u32, seed: u64) -> Self {
        AutoSim {
            n: num_qubits,
            mode: Mode::Stabilizer {
                tab: Tableau::zero(num_qubits as usize),
                log: Vec::new(),
                rng: Rng::new(seed),
            },
        }
    }

    /// Replay a recorded Clifford prefix onto a fresh state vector.
    async fn replay(n: u32, log: &[Op]) -> Option<StateVectorSim<f64>> {
        let mut sv = StateVectorSim::<f64>::try_zeroed(n, SEED).ok()?;
        let none = GateModifiers::none();
        for op in log {
            match op {
                Op::U {
                    target,
                    theta,
                    phi,
                    lambda,
                    controls,
                    neg_controls,
                    power,
                } => {
                    let m = GateModifiers {
                        controls: controls.clone(),
                        neg_controls: neg_controls.clone(),
                        power: *power,
                    };
                    sv.u(*target, *theta, *phi, *lambda, &m).await;
                }
                Op::Gphase {
                    gamma,
                    controls,
                    neg_controls,
                    power,
                } => {
                    let m = GateModifiers {
                        controls: controls.clone(),
                        neg_controls: neg_controls.clone(),
                        power: *power,
                    };
                    sv.gphase(*gamma, &m).await;
                }
                Op::Measure { qubit, outcome } => sv.project(*qubit, *outcome),
                Op::Reset { qubit, outcome } => {
                    sv.project(*qubit, *outcome);
                    if *outcome {
                        sv.u(
                            *qubit,
                            std::f64::consts::PI,
                            0.0,
                            std::f64::consts::PI,
                            &none,
                        )
                        .await;
                    }
                }
            }
        }
        Some(sv)
    }

    /// Switch to dense state-vector mode by replaying the logged prefix.
    async fn fall_back(&mut self) {
        let log = match &mut self.mode {
            Mode::Stabilizer { log, .. } => std::mem::take(log),
            Mode::StateVector(_) => return,
        };
        let sv = Self::replay(self.n, &log)
            .await
            .expect("state vector allocation on Clifford fallback");
        self.mode = Mode::StateVector(sv);
    }
}

#[async_trait(?Send)]
impl QuantumBackend for AutoSim {
    async fn u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, m: &GateModifiers) {
        if let Mode::Stabilizer { tab, log, .. } = &mut self.mode {
            if apply_clifford_u(tab, target, theta, phi, lambda, m) {
                log.push(Op::U {
                    target,
                    theta,
                    phi,
                    lambda,
                    controls: m.controls.clone(),
                    neg_controls: m.neg_controls.clone(),
                    power: m.power,
                });
                return;
            }
            // Non-Clifford: fall back, then apply on the state vector.
            self.fall_back().await;
        }
        if let Mode::StateVector(sv) = &mut self.mode {
            sv.u(target, theta, phi, lambda, m).await;
        }
    }

    async fn gphase(&mut self, gamma: f64, m: &GateModifiers) {
        if let Mode::Stabilizer { tab, log, .. } = &mut self.mode {
            // Uncontrolled phase: no tableau effect but logged for a faithful
            // state-vector readout. A single-control phase is a Clifford phase
            // on the control when it's a multiple of π/2 (e.g. the `gphase`
            // correction trailing `cx` in stdgates).
            if apply_clifford_gphase(tab, gamma * m.power, m) {
                log.push(Op::Gphase {
                    gamma,
                    controls: m.controls.clone(),
                    neg_controls: m.neg_controls.clone(),
                    power: m.power,
                });
                return;
            }
            self.fall_back().await;
        }
        if let Mode::StateVector(sv) = &mut self.mode {
            sv.gphase(gamma, m).await;
        }
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        match &mut self.mode {
            Mode::Stabilizer { tab, log, rng } => {
                let outcome = tab.measure(qubit as usize, rng);
                log.push(Op::Measure { qubit, outcome });
                outcome
            }
            Mode::StateVector(sv) => sv.measure(qubit).await,
        }
    }

    async fn reset(&mut self, qubit: u32) {
        match &mut self.mode {
            Mode::Stabilizer { tab, log, rng } => {
                let outcome = tab.measure(qubit as usize, rng);
                if outcome {
                    tab.x(qubit as usize);
                }
                log.push(Op::Reset { qubit, outcome });
            }
            Mode::StateVector(sv) => sv.reset(qubit).await,
        }
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        match &self.mode {
            Mode::StateVector(sv) => sv.amplitudes().await,
            Mode::Stabilizer { log, .. } => {
                // Materialize the stabilizer state as a vector (only sensible
                // for modest n — e.g. `--state` / tests).
                let sv = Self::replay(self.n, log).await?;
                sv.amplitudes().await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::StateVectorSim;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    /// A Clifford gate, expressed exactly as the `U`/controlled-`U` calls the
    /// compiler would lower it to (so the detection path is exercised too).
    #[derive(Clone, Copy)]
    enum G {
        H(u32),
        S(u32),
        Sdg(u32),
        X(u32),
        Y(u32),
        Z(u32),
        Cx(u32, u32),
        Cz(u32, u32),
    }

    fn ctrl(c: u32) -> GateModifiers {
        GateModifiers {
            controls: vec![c],
            neg_controls: vec![],
            power: 1.0,
        }
    }

    async fn apply(b: &mut dyn QuantumBackend, g: G) {
        let n = GateModifiers::none();
        match g {
            G::H(q) => b.u(q, FRAC_PI_2, 0.0, PI, &n).await,
            G::S(q) => b.u(q, 0.0, 0.0, FRAC_PI_2, &n).await,
            G::Sdg(q) => b.u(q, 0.0, 0.0, -FRAC_PI_2, &n).await,
            G::X(q) => b.u(q, PI, 0.0, PI, &n).await,
            G::Y(q) => b.u(q, PI, FRAC_PI_2, FRAC_PI_2, &n).await,
            G::Z(q) => b.u(q, 0.0, 0.0, PI, &n).await,
            G::Cx(c, t) => b.u(t, PI, 0.0, PI, &ctrl(c)).await,
            G::Cz(c, t) => b.u(t, 0.0, 0.0, PI, &ctrl(c)).await,
        }
    }

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

    fn random_clifford(n: u32, len: usize, seed: u64) -> Vec<G> {
        let mut r = Lcg(seed.wrapping_add(1));
        (0..len)
            .map(|_| match r.below(8) {
                0 => G::H(r.below(n)),
                1 => G::S(r.below(n)),
                2 => G::Sdg(r.below(n)),
                3 => G::X(r.below(n)),
                4 => G::Y(r.below(n)),
                5 => G::Z(r.below(n)),
                6 | 7 => {
                    let c = r.below(n);
                    let mut t = r.below(n);
                    if t == c {
                        t = (t + 1) % n;
                    }
                    if r.below(2) == 0 {
                        G::Cx(c, t)
                    } else {
                        G::Cz(c, t)
                    }
                }
                _ => unreachable!(),
            })
            .collect()
    }

    /// Differential: every all-qubit measurement the stabilizer tableau
    /// produces must land in the support of the true state (computed by the
    /// state-vector sim). A wrong tableau would yield zero-amplitude outcomes.
    #[tokio::test(flavor = "multi_thread")]
    async fn tableau_measurements_lie_in_statevector_support() {
        let n = 5u32;
        for seed in 0..12u64 {
            let circ = random_clifford(n, 40, seed);

            let mut sv = StateVectorSim::<f64>::try_zeroed(n, 0).unwrap();
            for &g in &circ {
                apply(&mut sv, g).await;
            }
            let amps = sv.amplitudes().await.unwrap();

            for shot in 0..16u64 {
                let mut auto = AutoSim::with_seed(n, seed * 1000 + shot + 1);
                for &g in &circ {
                    apply(&mut auto, g).await;
                }
                let mut idx = 0usize;
                for q in 0..n {
                    if auto.measure(q).await {
                        idx |= 1 << q;
                    }
                }
                assert!(
                    amps[idx].norm_sqr() > 1e-9,
                    "seed {seed} shot {shot}: tableau outcome {idx:b} has ~0 amplitude",
                );
            }
        }
    }

    /// Entangled correlations: a Bell pair always measures equal bits.
    #[tokio::test(flavor = "multi_thread")]
    async fn bell_pair_is_perfectly_correlated() {
        let mut saw_zero = false;
        let mut saw_one = false;
        for seed in 0..40u64 {
            // Spread seeds so the first RNG draw varies (xorshift mixes weakly
            // across tiny sequential seeds).
            let s = (seed + 1).wrapping_mul(0x9E3779B97F4A7C15);
            let mut a = AutoSim::with_seed(2, s);
            apply(&mut a, G::H(0)).await;
            apply(&mut a, G::Cx(0, 1)).await;
            let m0 = a.measure(0).await;
            let m1 = a.measure(1).await;
            assert_eq!(m0, m1, "bell pair uncorrelated");
            saw_zero |= !m0;
            saw_one |= m0;
        }
        assert!(saw_zero && saw_one, "measurement looks non-random");
    }

    /// `H Z H = X`, so |0⟩ → |1⟩ deterministically (tests H/Z phase handling).
    #[tokio::test(flavor = "multi_thread")]
    async fn hzh_flips_deterministically() {
        for seed in 0..8u64 {
            let mut a = AutoSim::with_seed(1, seed + 1);
            apply(&mut a, G::H(0)).await;
            apply(&mut a, G::Z(0)).await;
            apply(&mut a, G::H(0)).await;
            assert!(a.measure(0).await, "HZH|0> should always measure 1");
        }
    }

    /// Fallback: a Clifford prefix, then a non-Clifford `T`, then more gates.
    /// `AutoSim` (tableau → replay → state vector) must match the pure
    /// state-vector sim exactly.
    /// A Clifford prefix, a non-Clifford `T`, then more (incl. non-Clifford).
    async fn mixed_circuit(b: &mut dyn QuantumBackend) {
        let none = GateModifiers::none();
        apply(b, G::H(0)).await;
        apply(b, G::Cx(0, 1)).await;
        apply(b, G::S(2)).await;
        apply(b, G::Cz(1, 3)).await;
        // Non-Clifford T on qubit 0 → triggers fallback for AutoSim.
        b.u(0, 0.0, 0.0, FRAC_PI_4, &none).await;
        apply(b, G::H(2)).await;
        apply(b, G::Cx(2, 0)).await;
        b.u(3, 0.37, 0.11, 0.59, &none).await; // arbitrary non-Clifford
    }

    /// Fallback: `AutoSim` (tableau → replay → state vector) must match the
    /// pure state-vector sim exactly on a mixed Clifford/non-Clifford circuit.
    #[tokio::test(flavor = "multi_thread")]
    async fn fallback_matches_statevector() {
        let n = 4u32;
        let mut auto = AutoSim::new(n);
        mixed_circuit(&mut auto).await;
        let mut sv = StateVectorSim::<f64>::try_zeroed(n, 0).unwrap();
        mixed_circuit(&mut sv).await;

        let a = auto.amplitudes().await.unwrap();
        let b = sv.amplitudes().await.unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).norm() < 1e-9, "fallback mismatch: {x} vs {y}");
        }
    }
}
