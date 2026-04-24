use std::ops::Range;

use num_complex::Complex;
use num_traits::{Float, One, Zero};

pub type Id = usize;

#[derive(Debug, Clone, Default)]
pub struct QuantumMemory {
    allocations: Vec<(Id, usize)>,
}

impl QuantumMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocations(&self) -> &[(Id, usize)] {
        &self.allocations
    }

    pub fn size(&self) -> usize {
        self.allocations.iter().map(|(_, size)| size).sum()
    }

    pub fn alloc(&mut self, size: usize) -> QuantumRegister {
        let id = self.allocations.len();
        self.allocations.push((id, size));
        QuantumRegister {
            parts: vec![(id, 0..size)],
        }
    }

    pub fn get(&self, register: &QuantumRegister) -> QuantumMemorySlice {
        let ranges = register
            .parts
            .iter()
            .map(|(id, range)| {
                let offset = self.offset_of(*id);
                (offset + range.start)..(offset + range.end)
            })
            .collect();
        QuantumMemorySlice { ranges }
    }

    fn offset_of(&self, id: Id) -> usize {
        self.allocations
            .iter()
            .take_while(|(i, _)| *i != id)
            .map(|(_, size)| size)
            .sum()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuantumRegister {
    parts: Vec<(Id, Range<usize>)>,
}

impl QuantumRegister {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parts(&self) -> &[(Id, Range<usize>)] {
        &self.parts
    }

    pub fn len(&self) -> usize {
        self.parts.iter().map(|(_, r)| r.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|(_, r)| r.is_empty())
    }

    pub fn slice(&self, range: Range<usize>) -> QuantumRegister {
        assert!(range.start <= range.end, "slice start exceeds end");
        assert!(range.end <= self.len(), "slice end exceeds register length");

        let mut parts = Vec::new();
        let mut skip = range.start;
        let mut remaining = range.end - range.start;

        for (id, part) in &self.parts {
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
            parts.push((*id, start..(start + take)));
            remaining -= take;
            skip = 0;
        }

        QuantumRegister { parts }
    }

    pub fn concat(mut self, other: Self) -> Self {
        self.parts.extend(other.parts);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuantumMemorySlice {
    ranges: Vec<Range<usize>>,
}

impl QuantumMemorySlice {
    pub fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    pub fn len(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.iter().all(|r| r.is_empty())
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().cloned().flatten()
    }

    pub fn get(&self, mut idx: usize) -> Option<usize> {
        for range in &self.ranges {
            let n = range.len();
            if idx < n {
                return Some(range.start + idx);
            }
            idx -= n;
        }
        None
    }
}

impl IntoIterator for QuantumMemorySlice {
    type Item = usize;
    type IntoIter = std::iter::Flatten<std::vec::IntoIter<Range<usize>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.ranges.into_iter().flatten()
    }
}

impl<'a> IntoIterator for &'a QuantumMemorySlice {
    type Item = usize;
    type IntoIter = std::iter::Flatten<std::iter::Cloned<std::slice::Iter<'a, Range<usize>>>>;

    fn into_iter(self) -> Self::IntoIter {
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
        let m00 = Complex::new(c, F::zero());
        let m01 = -Complex::from_polar(s, self.lambda);
        let m10 = Complex::from_polar(s, self.phi);
        let m11 = Complex::from_polar(c, self.phi + self.lambda);
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
            let iden = if i == j { Complex::one() } else { Complex::zero() };
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

impl<F: Float> StateVector<F> {
    pub fn zero(size: usize) -> Self {
        let len = 1usize << size;
        let mut state = vec![Complex::zero(); len];
        state[0] = Complex::one();
        Self {
            state,
            size,
            global_phase: F::zero(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn state(&self) -> &[Complex<F>] {
        &self.state
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_register_with_single_part() {
        let mut mem = QuantumMemory::new();
        let reg = mem.alloc(3);
        assert_eq!(reg.parts(), &[(0, 0..3)]);
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn allocations_are_tracked() {
        let mut mem = QuantumMemory::new();
        let _ = mem.alloc(3);
        let _ = mem.alloc(2);
        assert_eq!(mem.allocations(), &[(0, 3), (1, 2)]);
        assert_eq!(mem.size(), 5);
    }

    #[test]
    fn get_maps_to_global_ranges() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(2);
        let c = mem.alloc(4);

        assert_eq!(mem.get(&a).ranges(), &[0..3]);
        assert_eq!(mem.get(&b).ranges(), &[3..5]);
        assert_eq!(mem.get(&c).ranges(), &[5..9]);
    }

    #[test]
    fn slice_within_single_part() {
        let mut mem = QuantumMemory::new();
        let reg = mem.alloc(5);
        let s = reg.slice(1..4);
        assert_eq!(s.parts(), &[(0, 1..4)]);
    }

    #[test]
    fn slice_spans_multiple_parts() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let joined = a.concat(b);
        let s = joined.slice(2..6);
        assert_eq!(s.parts(), &[(0, 2..3), (1, 0..3)]);
    }

    #[test]
    fn concat_appends_parts() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(2);
        let b = mem.alloc(3);
        let joined = a.concat(b);
        assert_eq!(joined.parts(), &[(0, 0..2), (1, 0..3)]);
        assert_eq!(joined.len(), 5);
    }

    #[test]
    fn slice_of_concat_maps_to_expected_global_indices() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let joined = a.concat(b);
        let slice = mem.get(&joined.slice(2..6));
        let indices: Vec<usize> = slice.iter().collect();
        assert_eq!(indices, vec![2, 3, 4, 5]);
    }

    #[test]
    fn memory_slice_iter_yields_global_indices() {
        let mut mem = QuantumMemory::new();
        let _ = mem.alloc(3);
        let b = mem.alloc(2);
        let slice = mem.get(&b);
        let indices: Vec<usize> = slice.iter().collect();
        assert_eq!(indices, vec![3, 4]);
    }

    #[test]
    fn memory_slice_get_returns_global_index() {
        let mut mem = QuantumMemory::new();
        let a = mem.alloc(3);
        let b = mem.alloc(4);
        let slice = mem.get(&a.concat(b));
        assert_eq!(slice.get(0), Some(0));
        assert_eq!(slice.get(3), Some(3));
        assert_eq!(slice.get(6), Some(6));
        assert_eq!(slice.get(7), None);
    }

    const TOL: f64 = 1e-10;

    fn close(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a - b).norm() < TOL
    }

    fn states_close(a: &[Complex<f64>], b: &[Complex<f64>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
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
        let m = pauli_x().matrix();
        assert!(close(m[0][0], Complex::zero()));
        assert!(close(m[0][1], Complex::new(1.0, 0.0)));
        assert!(close(m[1][0], Complex::new(1.0, 0.0)));
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
        assert!(states_close(sv.state(), &expected));
    }

    #[test]
    fn hadamard_squared_is_identity() {
        let mut sv: StateVector<f64> = StateVector::zero(1);
        sv.apply_unitary(&hadamard(), 0);
        sv.apply_unitary(&hadamard(), 0);
        let expected = [Complex::new(1.0, 0.0), Complex::zero()];
        assert!(states_close(sv.state(), &expected));
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
        assert!(states_close(sv.state(), &expected));
    }

    #[test]
    fn ctrl_x_is_cnot() {
        let mut sv: StateVector<f64> = StateVector::zero(2);
        sv.apply_unitary(&pauli_x(), 0);
        sv.apply(&Gate::new(pauli_x()).ctrl(0), 1);
        let mut expected = [Complex::zero(); 4];
        expected[3] = Complex::new(1.0, 0.0);
        assert!(states_close(sv.state(), &expected));
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
        assert!(states_close(sv.state(), &expected));
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
}
