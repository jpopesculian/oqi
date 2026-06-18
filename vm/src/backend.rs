//! The pluggable quantum execution layer.
//!
//! Every gate in an OpenQASM program is, by the time the compiler is
//! done with it, expressed in terms of the two built-in primitives
//! `U(θ, φ, λ)` and `gphase(γ)` (backends ship their own `stdgates.inc`
//! that decomposes the standard library to these, or maps gates onto
//! `extern` calls). A backend therefore only needs to implement those
//! two primitives plus measurement, reset, and — optionally — timing.
//!
//! Implementations: [`crate::sim::StateVectorSim`] (CPU simulator). A
//! GPU simulator or hardware provider would implement the same trait.

use oqi_classical::Duration;

/// Modifiers accumulated from `ctrl @` / `negctrl @` / `inv @` /
/// `pow(k) @` on a gate call (and inherited from enclosing gate
/// bodies). Control indices are global qubit indices.
///
/// `inv` is folded into `power` as a sign flip, and `pow(k)` as a
/// product, so a single `power` captures both: this is exact for the
/// built-in primitives (`U^k` via matrix power, `gphase(γ)^k =
/// gphase(kγ)`).
#[derive(Debug, Clone)]
pub struct GateModifiers {
    pub controls: Vec<u32>,
    pub neg_controls: Vec<u32>,
    pub power: f64,
}

impl GateModifiers {
    /// No modifiers: the identity context (`power = 1`).
    pub fn none() -> Self {
        GateModifiers {
            controls: Vec::new(),
            neg_controls: Vec::new(),
            power: 1.0,
        }
    }
}

impl Default for GateModifiers {
    fn default() -> Self {
        GateModifiers::none()
    }
}

/// A quantum execution backend. The VM resolves all gates down to these
/// calls against global qubit indices.
pub trait QuantumBackend {
    /// Apply `U(θ, φ, λ)` to `target` under the given modifiers.
    fn u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, modifiers: &GateModifiers);

    /// Apply the global phase `γ` under the given modifiers. A
    /// controlled global phase is a relative phase (the `p`/phase gate).
    fn gphase(&mut self, gamma: f64, modifiers: &GateModifiers);

    /// Measure `qubit` in the Z basis, collapsing the state, and return
    /// the outcome.
    fn measure(&mut self, qubit: u32) -> bool;

    /// Reset `qubit` to |0⟩.
    fn reset(&mut self, qubit: u32);

    /// A scheduling barrier across `qubits`. No effect on state.
    fn barrier(&mut self, _qubits: &[u32]) {}

    /// An idle delay on `qubits`. No effect on state in the MVP.
    fn delay(&mut self, _qubits: &[u32], _duration: Duration) {}
}
