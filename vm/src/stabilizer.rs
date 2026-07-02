//! A stabilizer tableau (Gottesman–Knill / CHP) for simulating Clifford
//! circuits in O(n²) time per gate and O(n²) memory — independent of the 2ⁿ
//! state-vector size.
//!
//! Layout: `2n + 1` rows. Rows `0..n` are *destabilizer* generators, rows
//! `n..2n` are *stabilizer* generators, and row `2n` is a scratch row used by
//! measurement. Each row is a Pauli `iˢ · ∏_j X_j^{x_j} Z_j^{z_j}`, with the
//! phase tracked as an explicit power of `i` (`s ∈ {0,1,2,3}`) rather than a
//! bare sign — this keeps Pauli multiplication total (the transient odd-phase
//! row produced during a random measurement is overwritten immediately).

use crate::clifford::CliffordSink;
use crate::sim::Rng;

#[derive(Clone)]
struct Row {
    x: Vec<bool>,
    z: Vec<bool>,
    /// Phase as a power of `i`, mod 4.
    s: u8,
}

impl Row {
    fn zero(n: usize) -> Self {
        Row {
            x: vec![false; n],
            z: vec![false; n],
            s: 0,
        }
    }
}

/// A stabilizer tableau over `n` qubits.
pub struct Tableau {
    n: usize,
    rows: Vec<Row>,
}

impl Tableau {
    /// The |0…0⟩ tableau: destabilizers `X_i`, stabilizers `Z_i`.
    pub fn zero(n: usize) -> Self {
        let mut rows = vec![Row::zero(n); 2 * n + 1];
        for i in 0..n {
            rows[i].x[i] = true;
            rows[n + i].z[i] = true;
        }
        Tableau { n, rows }
    }

    /// Hadamard on qubit `a`: `X↔Z`, with an `i²` per `Y` (`x=z=1`).
    pub fn h(&mut self, a: usize) {
        let n = self.n;
        for row in self.rows.iter_mut().take(2 * n) {
            if row.x[a] && row.z[a] {
                row.s = (row.s + 2) % 4;
            }
            std::mem::swap(&mut row.x[a], &mut row.z[a]);
        }
    }

    /// Phase gate `S` on qubit `a`: `X→Y` (an `i` per `X`), `Z→Z`.
    pub fn s(&mut self, a: usize) {
        let n = self.n;
        for row in self.rows.iter_mut().take(2 * n) {
            if row.x[a] {
                row.s = (row.s + 1) % 4;
                row.z[a] ^= true;
            }
        }
    }

    /// `S†` on qubit `a` (`S³`).
    pub fn sdg(&mut self, a: usize) {
        self.s(a);
        self.s(a);
        self.s(a);
    }

    /// Pauli-X on qubit `a`: flips the sign (`i²`) of `Z`-bearing rows.
    pub fn x(&mut self, a: usize) {
        let n = self.n;
        for row in self.rows.iter_mut().take(2 * n) {
            if row.z[a] {
                row.s = (row.s + 2) % 4;
            }
        }
    }

    /// CNOT with control `a`, target `b` (no phase in this convention).
    pub fn cnot(&mut self, a: usize, b: usize) {
        let n = self.n;
        for row in self.rows.iter_mut().take(2 * n) {
            let xa = row.x[a];
            let zb = row.z[b];
            row.x[b] ^= xa;
            row.z[a] ^= zb;
        }
    }

    /// CZ on qubits `c`, `t` (symmetric): `H(t)·CNOT(c,t)·H(t)`.
    pub fn cz(&mut self, c: usize, t: usize) {
        self.h(t);
        self.cnot(c, t);
        self.h(t);
    }

    /// CY with control `c`, target `t`: `S†(t)·CNOT(c,t)·S(t)`.
    pub fn cy(&mut self, c: usize, t: usize) {
        self.sdg(t);
        self.cnot(c, t);
        self.s(t);
    }

    /// Measure qubit `a` in the Z basis, collapsing the tableau, and return
    /// the outcome. Random when the outcome is undetermined (a stabilizer
    /// anticommutes with `Z_a`), otherwise the determined value.
    pub fn measure(&mut self, a: usize, rng: &mut Rng) -> bool {
        let n = self.n;
        let p = (n..2 * n).find(|&i| self.rows[i].x[a]);
        match p {
            Some(p) => {
                for i in 0..2 * n {
                    if i != p && self.rows[i].x[a] {
                        self.rowsum(i, p);
                    }
                }
                // The destabilizer p-n becomes the old stabilizer p; the new
                // stabilizer p is ±Z_a with a random sign (the outcome). Any
                // transient odd phase left in row p-n by the loop is discarded
                // by this overwrite.
                self.rows[p - n] = self.rows[p].clone();
                let outcome = rng.next_f64() < 0.5;
                let mut newrow = Row::zero(n);
                newrow.z[a] = true;
                newrow.s = if outcome { 2 } else { 0 };
                self.rows[p] = newrow;
                outcome
            }
            None => {
                self.rows[2 * n] = Row::zero(n);
                for i in 0..n {
                    if self.rows[i].x[a] {
                        self.rowsum(2 * n, i + n);
                    }
                }
                debug_assert!(
                    self.rows[2 * n].s.is_multiple_of(2),
                    "determined outcome has odd phase"
                );
                self.rows[2 * n].s == 2
            }
        }
    }

    /// Left-multiply row `h` by row `i`: `row_h := row_i · row_h`. The phase
    /// picks up `i²` for each qubit where `Z_i` must pass `X_h`.
    fn rowsum(&mut self, h: usize, i: usize) {
        let mut cross = 0u32;
        for j in 0..self.n {
            if self.rows[i].z[j] && self.rows[h].x[j] {
                cross += 1;
            }
        }
        self.rows[h].s = (self.rows[h].s + self.rows[i].s + 2 * (cross % 2) as u8) % 4;
        for j in 0..self.n {
            let (xi, zi) = (self.rows[i].x[j], self.rows[i].z[j]);
            self.rows[h].x[j] ^= xi;
            self.rows[h].z[j] ^= zi;
        }
    }
}

/// The tableau is a phase-blind Clifford sink: global phase is not part of
/// the stabilizer formalism, so [`CliffordSink::phase`] keeps its no-op
/// default.
impl CliffordSink for Tableau {
    fn h(&mut self, q: usize) {
        Tableau::h(self, q);
    }
    fn s(&mut self, q: usize) {
        Tableau::s(self, q);
    }
    fn cnot(&mut self, c: usize, t: usize) {
        Tableau::cnot(self, c, t);
    }
    fn cz(&mut self, c: usize, t: usize) {
        Tableau::cz(self, c, t);
    }
    fn cy(&mut self, c: usize, t: usize) {
        Tableau::cy(self, c, t);
    }
}
