//! Z-basis measurement on dense amplitudes, shared by the state-vector
//! backends. Probabilities are accumulated in `f64` whatever the amplitude
//! precision, for numerical stability.

use num_complex::Complex;
use num_traits::Float;

use crate::rng::Rng;

/// The factor that renormalizes the amplitudes surviving a collapse onto
/// `outcome`, given the probability `p_one` of measuring 1. Zero for a
/// numerically impossible outcome, which leaves nothing to renormalize.
pub(crate) fn renorm(p_one: f64, outcome: bool) -> f64 {
    let norm = if outcome { p_one } else { 1.0 - p_one };
    if norm > 0.0 { 1.0 / norm.sqrt() } else { 0.0 }
}

/// P(`qubit` = 1): Σ |amp_i|² over the basis states with that bit set.
fn p_one<F: Float>(amps: &[Complex<F>], qubit: u32) -> f64 {
    let bit = 1usize << qubit;
    amps.iter()
        .enumerate()
        .filter(|(i, _)| i & bit != 0)
        .map(|(_, a)| a.norm_sqr().to_f64().unwrap())
        .sum()
}

/// Zero the half of the state that disagrees with `outcome` and renormalize
/// the rest.
fn collapse<F: Float>(amps: &mut [Complex<F>], qubit: u32, outcome: bool, p_one: f64) {
    let bit = 1usize << qubit;
    let scale = F::from(renorm(p_one, outcome)).unwrap();
    for (i, a) in amps.iter_mut().enumerate() {
        *a = if (i & bit != 0) == outcome {
            *a * scale
        } else {
            Complex::new(F::zero(), F::zero())
        };
    }
}

/// Measure `qubit`, collapsing `amps`, and return the outcome.
pub(crate) fn measure<F: Float>(amps: &mut [Complex<F>], qubit: u32, rng: &mut Rng) -> bool {
    let p_one = p_one(amps, qubit);
    let outcome = rng.next_f64() < p_one;
    collapse(amps, qubit, outcome, p_one);
    outcome
}

/// Collapse `qubit` onto a forced `outcome` — a measurement with no RNG,
/// used to replay a recorded one.
pub(crate) fn project<F: Float>(amps: &mut [Complex<F>], qubit: u32, outcome: bool) {
    let p_one = p_one(amps, qubit);
    collapse(amps, qubit, outcome, p_one);
}
