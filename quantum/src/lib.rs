use std::ops::Range;

use num_complex::Complex;
use num_traits::{Float, One, Zero};

/// Allocator for the single flat global quantum memory. OpenQASM qubits
/// are all statically allocated, in declaration order, and never freed or
/// relocated, so allocation is just a bump of a running cursor: each
/// [`alloc`](Self::alloc) hands out the next contiguous span of global
/// indices.
#[derive(Debug, Clone, Default)]
pub struct QuantumMemory {
    size: usize,
}

impl QuantumMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of qubits allocated so far.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Allocate `size` fresh qubits and return them as a contiguous
    /// register over global indices.
    #[allow(clippy::single_range_in_vec_init)]
    pub fn alloc(&mut self, size: usize) -> QuantumRegister {
        let start = self.size;
        self.size += size;
        QuantumRegister {
            ranges: vec![start..start + size],
        }
    }
}

/// A logical quantum register: an ordered list of half-open ranges of
/// **global** qubit indices. A plain declaration is one contiguous range;
/// slicing, discrete selection, and `++` concatenation build registers
/// with several ranges. Because indices are already global, a register is
/// directly usable by the simulator with no further resolution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuantumRegister {
    ranges: Vec<Range<usize>>,
}

impl QuantumRegister {
    pub fn new() -> Self {
        Self::default()
    }

    /// The global index ranges making up this register, in logical order.
    pub fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    pub fn len(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.iter().all(|r| r.is_empty())
    }

    /// The register covering logical positions `range` of this one.
    pub fn slice(&self, range: Range<usize>) -> QuantumRegister {
        assert!(range.start <= range.end, "slice start exceeds end");
        assert!(range.end <= self.len(), "slice end exceeds register length");

        let mut ranges = Vec::new();
        let mut skip = range.start;
        let mut remaining = range.end - range.start;

        for part in &self.ranges {
            if remaining == 0 {
                break;
            }
            let part_len = part.len();
            if skip >= part_len {
                skip -= part_len;
                continue;
            }
            let take = remaining.min(part_len - skip);
            let start = part.start + skip;
            ranges.push(start..(start + take));
            remaining -= take;
            skip = 0;
        }

        QuantumRegister { ranges }
    }

    pub fn concat(mut self, other: Self) -> Self {
        self.ranges.extend(other.ranges);
        self
    }

    /// Global index of the qubit at logical position `idx`.
    pub fn global_index_of(&self, mut idx: usize) -> Option<usize> {
        for range in &self.ranges {
            let n = range.len();
            if idx < n {
                return Some(range.start + idx);
            }
            idx -= n;
        }
        None
    }

    /// Iterate the register's global qubit indices in logical order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().cloned().flatten()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unitary<F> {
    pub theta: F,
    pub phi: F,
    pub lambda: F,
}

impl<F> Unitary<F> {
    pub const fn new(theta: F, phi: F, lambda: F) -> Self {
        Self { theta, phi, lambda }
    }
}

impl<F: Float> Unitary<F> {
    pub fn matrix(&self) -> [[Complex<F>; 2]; 2] {
        let two = F::one() + F::one();
        let c = (self.theta / two).cos();
        let s = (self.theta / two).sin();
        // OpenQASM's built-in `U` carries a global phase of e^{iθ/2}
        // relative to the bare Bloch/ZYZ form (docs/gates.rst:304). It is
        // unobservable for an uncontrolled gate, but becomes a *relative*
        // phase once `U` is controlled — e.g. `cx = ctrl @ x` where
        // `x = U(π,0,π); gphase(-π/2)`. Omitting it makes every controlled
        // standard-library gate wrong by a phase, so it must live here.
        let gp = Complex::from_polar(F::one(), self.theta / two);
        let m00 = gp * Complex::new(c, F::zero());
        let m01 = gp * -Complex::from_polar(s, self.lambda);
        let m10 = gp * Complex::from_polar(s, self.phi);
        let m11 = gp * Complex::from_polar(c, self.phi + self.lambda);
        [[m00, m01], [m10, m11]]
    }
}

#[derive(Debug, Clone)]
pub struct Gate<F> {
    pub unitary: Unitary<F>,
    pub controls: Vec<usize>,
    pub neg_controls: Vec<usize>,
    pub power: F,
}

impl<F: Float> Gate<F> {
    pub fn new(unitary: Unitary<F>) -> Self {
        Self {
            unitary,
            controls: Vec::new(),
            neg_controls: Vec::new(),
            power: F::one(),
        }
    }

    pub fn ctrl(mut self, q: usize) -> Self {
        self.controls.push(q);
        self
    }

    pub fn neg_ctrl(mut self, q: usize) -> Self {
        self.neg_controls.push(q);
        self
    }

    pub fn inv(mut self) -> Self {
        self.power = -self.power;
        self
    }

    pub fn pow(mut self, k: F) -> Self {
        self.power = self.power * k;
        self
    }

    pub fn matrix(&self) -> [[Complex<F>; 2]; 2] {
        let m = self.unitary.matrix();
        if self.power == F::one() {
            return m;
        }
        matrix_pow(m, self.power)
    }
}

impl<F: Float> From<Unitary<F>> for Gate<F> {
    fn from(u: Unitary<F>) -> Self {
        Self::new(u)
    }
}

fn matrix_pow<F: Float>(m: [[Complex<F>; 2]; 2], k: F) -> [[Complex<F>; 2]; 2] {
    let two = F::one() + F::one();
    let four = two + two;
    let tr = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let disc = tr * tr - det.scale(four);
    let tol = F::epsilon().sqrt();
    let two_c = Complex::new(two, F::zero());
    if disc.norm() < tol {
        let lambda = tr / two_c;
        let lk = lambda.powf(k);
        let zero = Complex::zero();
        return [[lk, zero], [zero, lk]];
    }
    let sqrt_disc = disc.sqrt();
    let l1 = (tr + sqrt_disc) / two_c;
    let l2 = (tr - sqrt_disc) / two_c;
    let l1k = l1.powf(k);
    let l2k = l2.powf(k);
    let denom = l1 - l2;
    let mut out = [[Complex::zero(); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let iden = if i == j {
                Complex::one()
            } else {
                Complex::zero()
            };
            let m_minus_l2 = m[i][j] - l2 * iden;
            let m_minus_l1 = m[i][j] - l1 * iden;
            out[i][j] = (l1k * m_minus_l2 - l2k * m_minus_l1) / denom;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct StateVector<F> {
    state: Vec<Complex<F>>,
    size: usize,
    global_phase: F,
}

/// Ceiling on a dense state vector's byte size. On an overcommitting allocator
/// a multi-TiB `try_reserve_exact` can succeed and only OOM-kill the process
/// when the pages are faulted in, so [`StateVector::try_zero`] refuses anything
/// larger up front. 2^34 B = 16 GiB ≈ 30 qubits (f64) — far above any real CPU
/// state-vector run, far below where overcommit bites. `u64` so the constant
/// (and the byte math) stays valid on 32-bit targets like wasm32.
const MAX_STATE_VECTOR_BYTES: u64 = 1 << 34;

impl<F: Float> StateVector<F> {
    /// Allocate the |0…0⟩ state for `size` qubits, or `None` if its
    /// `2^size`-amplitude vector cannot be allocated — the length overflows
    /// `usize`, its byte size exceeds [`MAX_STATE_VECTOR_BYTES`], or the
    /// allocator can't satisfy the request. This lets callers fail gracefully
    /// on oversized circuits instead of aborting the process on an infallible
    /// allocation.
    pub fn try_zero(size: usize) -> Option<Self> {
        let len = 1usize.checked_shl(size as u32)?;
        // On an overcommitting allocator the `try_reserve_exact` below can be
        // granted for a multi-TiB vector and only OOM-kill the process when
        // `resize` faults in the pages. Refuse allocations past the ceiling so
        // callers fail gracefully instead.
        let bytes = (len as u64).checked_mul(std::mem::size_of::<Complex<F>>() as u64)?;
        if bytes > MAX_STATE_VECTOR_BYTES {
            return None;
        }
        let mut state: Vec<Complex<F>> = Vec::new();
        state.try_reserve_exact(len).ok()?;
        state.resize(len, Complex::zero());
        state[0] = Complex::one();
        Some(Self {
            state,
            size,
            global_phase: F::zero(),
        })
    }

    /// Like [`try_zero`](Self::try_zero) but panics if the state vector
    /// cannot be allocated.
    pub fn zero(size: usize) -> Self {
        Self::try_zero(size).expect("state vector allocation")
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Reset in place to |0…0⟩ without reallocating — all amplitudes zeroed,
    /// `[0] = 1`, global phase cleared. Lets a simulator start a fresh shot on
    /// its existing buffer.
    pub fn zero_in_place(&mut self) {
        self.state.iter_mut().for_each(|a| *a = Complex::zero());
        self.state[0] = Complex::one();
        self.global_phase = F::zero();
    }

    pub fn state(&self) -> &[Complex<F>] {
        &self.state
    }

    /// Mutable access to the raw amplitudes. Callers are responsible for
    /// keeping the state normalized (e.g. measurement collapse).
    pub fn state_mut(&mut self) -> &mut [Complex<F>] {
        &mut self.state
    }

    pub fn global_phase(&self) -> F {
        self.global_phase
    }

    pub fn gphase(&mut self, gamma: F) {
        self.global_phase = self.global_phase + gamma;
    }

    pub fn resolve(&mut self) {
        if self.global_phase == F::zero() {
            return;
        }
        let phase = Complex::from_polar(F::one(), self.global_phase);
        for a in self.state.iter_mut() {
            *a = *a * phase;
        }
        self.global_phase = F::zero();
    }

    pub fn apply_unitary(&mut self, u: &Unitary<F>, target: usize) {
        self.apply(&Gate::new(*u), target);
    }

    pub fn apply(&mut self, gate: &Gate<F>, target: usize) {
        assert!(target < self.size, "target qubit out of range");
        assert!(
            !gate.controls.contains(&target) && !gate.neg_controls.contains(&target),
            "target qubit overlaps a control",
        );
        for &c in &gate.controls {
            assert!(c < self.size, "control qubit out of range");
        }
        for &c in &gate.neg_controls {
            assert!(c < self.size, "neg_control qubit out of range");
        }

        let m = gate.matrix();
        let target_bit = 1usize << target;
        let n = self.state.len();
        for i in 0..n {
            if i & target_bit != 0 {
                continue;
            }
            if gate.controls.iter().any(|&c| i & (1usize << c) == 0) {
                continue;
            }
            if gate.neg_controls.iter().any(|&c| i & (1usize << c) != 0) {
                continue;
            }
            let j = i | target_bit;
            let a = self.state[i];
            let b = self.state[j];
            self.state[i] = m[0][0] * a + m[0][1] * b;
            self.state[j] = m[1][0] * a + m[1][1] * b;
        }
    }
}

#[cfg(feature = "parallel")]
impl<F: Float + Send + Sync> StateVector<F> {
    /// Rayon-parallel equivalent of [`apply`](Self::apply).
    ///
    /// The amplitudes split into independent contiguous blocks of length
    /// `2·target_bit`: within a block, the first half (target bit clear)
    /// pairs element-wise with the second half (target bit set). Blocks
    /// have no cross-dependencies, so they run in parallel. Control masks
    /// are tested against the global index of each clear-target element.
    pub fn par_apply(&mut self, gate: &Gate<F>, target: usize) {
        use rayon::prelude::*;

        assert!(target < self.size, "target qubit out of range");
        assert!(
            !gate.controls.contains(&target) && !gate.neg_controls.contains(&target),
            "target qubit overlaps a control",
        );
        for &c in &gate.controls {
            assert!(c < self.size, "control qubit out of range");
        }
        for &c in &gate.neg_controls {
            assert!(c < self.size, "neg_control qubit out of range");
        }

        let m = gate.matrix();
        let target_bit = 1usize << target;
        let controls = &gate.controls;
        let neg_controls = &gate.neg_controls;

        self.state
            .par_chunks_mut(2 * target_bit)
            .enumerate()
            .for_each(|(block, chunk)| {
                let base = block * 2 * target_bit;
                for k in 0..target_bit {
                    let i = base + k;
                    if controls.iter().any(|&c| i & (1usize << c) == 0) {
                        continue;
                    }
                    if neg_controls.iter().any(|&c| i & (1usize << c) != 0) {
                        continue;
                    }
                    let a = chunk[k];
                    let b = chunk[target_bit + k];
                    chunk[k] = m[0][0] * a + m[0][1] * b;
                    chunk[target_bit + k] = m[1][0] * a + m[1][1] * b;
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn alloc_returns_register_with_single_range() {
        let mut mem = QuantumMemory::new();
        let reg = mem.alloc(3);
        assert_eq!(reg.ranges(), &[0..3]);
        assert_eq!(reg.len(), 3);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn allocations_advance_the_global_cursor() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(2);
        assert_eq!(a.ranges(), &[0..3]);
        assert_eq!(b.ranges(), &[3..5]);
        assert_eq!(mem.size(), 5);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn registers_carry_global_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(2);
        let c = mem.alloc(4);

        assert_eq!(a.ranges(), &[0..3]);
        assert_eq!(b.ranges(), &[3..5]);
        assert_eq!(c.ranges(), &[5..9]);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn slice_within_single_range() {
        let mut mem = QuantumMemory::new();
        let reg = mem.alloc(5);
        let s = reg.slice(1..4);
        assert_eq!(s.ranges(), &[1..4]);
    }

    #[test]
    fn slice_spans_multiple_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let joined = a.concat(b);
        let s = joined.slice(2..6);
        assert_eq!(s.ranges(), &[2..3, 3..6]);
    }

    #[test]
    fn concat_appends_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(2);
        let b = mem.alloc(3);
        let joined = a.concat(b);
        assert_eq!(joined.ranges(), &[0..2, 2..5]);
        assert_eq!(joined.len(), 5);
    }

    #[test]
    fn slice_of_concat_yields_expected_global_indices() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let joined = a.concat(b);
        let indices: Vec<usize> = joined.slice(2..6).iter().collect();
        assert_eq!(indices, vec![2, 3, 4, 5]);
    }

    #[test]
    fn register_iter_yields_global_indices() {
        let mut mem = QuantumMemory::new();
        let _ = mem.alloc(3);
        let b = mem.alloc(2);
        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![3, 4]);
    }

    #[test]
    fn global_index_of_returns_global_index() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let reg = a.concat(b);
        assert_eq!(reg.global_index_of(0), Some(0));
        assert_eq!(reg.global_index_of(3), Some(3));
        assert_eq!(reg.global_index_of(6), Some(6));
        assert_eq!(reg.global_index_of(7), None);
    }

    const TOL: f64 = 1e-10;

    fn close(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a - b).norm() < TOL
    }

    fn states_close(a: &[Complex<f64>], b: &[Complex<f64>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    /// Compare `actual` to an `ideal` state up to a single global phase.
    /// The built-in `U` matches each standard gate only up to e^{iθ/2}
    /// (docs/gates.rst), so uncontrolled-gate states are equal to the
    /// textbook result modulo an unobservable global phase.
    fn states_close_up_to_phase(actual: &[Complex<f64>], ideal: &[Complex<f64>]) -> bool {
        if actual.len() != ideal.len() {
            return false;
        }
        let phase = ideal
            .iter()
            .zip(actual)
            .find(|(b, _)| b.norm() > 1e-9)
            .map(|(b, a)| a / b)
            .unwrap_or(Complex::new(1.0, 0.0));
        ideal.iter().zip(actual).all(|(b, a)| close(*a, phase * b))
    }

    fn pauli_x() -> Unitary<f64> {
        Unitary::new(std::f64::consts::PI, 0.0, std::f64::consts::PI)
    }

    fn hadamard() -> Unitary<f64> {
        Unitary::new(std::f64::consts::FRAC_PI_2, 0.0, std::f64::consts::PI)
    }

    #[test]
    fn state_vector_zero_is_ground_state() {
        let sv: StateVector<f64> = StateVector::zero(2);
        assert_eq!(sv.size(), 2);
        assert_eq!(sv.state().len(), 4);
        assert!(close(sv.state()[0], Complex::new(1.0, 0.0)));
        for amp in &sv.state()[1..] {
            assert!(close(*amp, Complex::zero()));
        }
        assert_eq!(sv.global_phase(), 0.0);
    }

    #[test]
    fn unitary_pauli_x_matrix() {
        // `U(π, 0, π)` is Pauli-X times the built-in `U`'s e^{iθ/2} global
        // phase, i.e. e^{iπ/2}·X = i·X (docs/gates.rst). The standard
        // library recovers a phaseless `x` with a trailing `gphase(-π/2)`.
        let m = pauli_x().matrix();
        assert!(close(m[0][0], Complex::zero()));
        assert!(close(m[0][1], Complex::new(0.0, 1.0)));
        assert!(close(m[1][0], Complex::new(0.0, 1.0)));
        assert!(close(m[1][1], Complex::zero()));
    }

    #[test]
    fn apply_x_flips_ground_state() {
        let mut sv: StateVector<f64> = StateVector::zero(2);
        sv.apply_unitary(&pauli_x(), 0);
        let expected = [
            Complex::zero(),
            Complex::new(1.0, 0.0),
            Complex::zero(),
            Complex::zero(),
        ];
        assert!(states_close_up_to_phase(sv.state(), &expected));
    }

    #[test]
    fn hadamard_squared_is_identity() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        sv.apply_unitary(&hadamard(), 0);
        sv.apply_unitary(&hadamard(), 0);
        let expected = [Complex::new(1.0, 0.0), Complex::zero()];
        assert!(states_close_up_to_phase(sv.state(), &expected));
    }

    #[test]
    fn inv_then_apply_cancels_x() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        sv.apply_unitary(&pauli_x(), 0);
        sv.apply(&Gate::new(pauli_x()).inv(), 0);
        let expected = [Complex::new(1.0, 0.0), Complex::zero()];
        assert!(states_close(sv.state(), &expected));
    }

    #[test]
    fn pow_half_squared_equals_x() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        sv.apply(&Gate::new(pauli_x()).pow(0.5), 0);
        sv.apply(&Gate::new(pauli_x()).pow(0.5), 0);
        let expected = [Complex::zero(), Complex::new(1.0, 0.0)];
        assert!(states_close_up_to_phase(sv.state(), &expected));
    }

    #[test]
    fn ctrl_x_is_cnot() {
        let mut sv: StateVector<f64> = StateVector::zero(2);
        sv.apply_unitary(&pauli_x(), 0);
        sv.apply(&Gate::new(pauli_x()).ctrl(0), 1);
        let mut expected = [Complex::zero(); 4];
        expected[3] = Complex::new(1.0, 0.0);
        assert!(states_close_up_to_phase(sv.state(), &expected));
    }

    #[test]
    fn ctrl_x_no_op_when_control_zero() {
        let mut sv: StateVector<f64> = StateVector::zero(2);
        sv.apply(&Gate::new(pauli_x()).ctrl(0), 1);
        let mut expected = [Complex::zero(); 4];
        expected[0] = Complex::new(1.0, 0.0);
        assert!(states_close(sv.state(), &expected));
    }

    #[test]
    fn neg_ctrl_x_fires_when_control_zero() {
        let mut sv: StateVector<f64> = StateVector::zero(2);
        sv.apply(&Gate::new(pauli_x()).neg_ctrl(0), 1);
        let mut expected = [Complex::zero(); 4];
        expected[2] = Complex::new(1.0, 0.0);
        assert!(states_close_up_to_phase(sv.state(), &expected));
    }

    #[test]
    fn gphase_accumulates_without_touching_state() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        let snapshot: Vec<Complex<f64>> = sv.state().to_vec();
        sv.gphase(0.5);
        sv.gphase(0.25);
        assert!((sv.global_phase() - 0.75).abs() < TOL);
        assert!(states_close(sv.state(), &snapshot));
    }

    #[test]
    fn resolve_bakes_phase_into_amplitudes() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        sv.gphase(std::f64::consts::FRAC_PI_2);
        sv.resolve();
        assert_eq!(sv.global_phase(), 0.0);
        let expected = [Complex::new(0.0, 1.0), Complex::zero()];
        assert!(states_close(sv.state(), &expected));
    }

    #[test]
    fn apply_does_not_disturb_accumulated_phase() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        sv.gphase(0.3);
        sv.apply_unitary(&pauli_x(), 0);
        assert!((sv.global_phase() - 0.3).abs() < TOL);
    }

    /// Build a non-trivial 4-qubit state by spreading amplitudes with
    /// Hadamards, for use as a parallel/serial comparison fixture.
    #[cfg(feature = "parallel")]
    fn spread_state() -> StateVector<f64> {
        let mut sv: StateVector<f64> = StateVector::zero(4);
        for q in 0..4 {
            sv.apply_unitary(&hadamard(), q);
        }
        sv
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_apply_matches_apply_for_each_target() {
        // A controlled gate on every (control, target) pair must give the
        // same state whether applied serially or in parallel.
        for target in 0..4 {
            for control in 0..4 {
                if control == target {
                    continue;
                }
                let gate = Gate::new(hadamard()).ctrl(control);
                let mut serial = spread_state();
                let mut parallel = spread_state();
                serial.apply(&gate, target);
                parallel.par_apply(&gate, target);
                assert!(
                    states_close(serial.state(), parallel.state()),
                    "mismatch for control {control}, target {target}",
                );
            }
        }
    }
}
