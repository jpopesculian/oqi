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

use async_trait::async_trait;
use num_complex::Complex;
use oqi_classical::Duration;

/// Modifiers on a single backend primitive call: the controls in scope
/// plus a `power` to raise this one primitive to. Control indices are
/// global qubit indices.
///
/// `power` carries `inv`/`pow` for the *individual* `U`/`gphase` reaching
/// the backend, which is always exact (`U^k` via matrix power, `gphase(γ)^k
/// = gphase(kγ)`). Composite-gate modifiers — where `inv` must reverse the
/// body and `pow` must apply to the product rather than each factor — are
/// resolved by the VM (it flattens the body and reverses/repeats the trace)
/// before any leaf reaches the backend, so a backend never sees more than a
/// per-primitive power.
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
///
/// Methods are `async` so a GPU/remote backend can `.await` its device
/// without blocking — essential in the browser, where the main thread can't
/// block. CPU backends implement them as ready futures. `?Send` keeps the
/// trait object-safe (via `async_trait`'s boxed futures) and works with
/// `!Send` web GPU types: a multi-threaded runtime's `block_on` still drives
/// the future on the calling thread, so the trait's futures need not be
/// `Send` (native callers use tokio; wasm uses the browser event loop).
#[async_trait(?Send)]
pub trait QuantumBackend {
    /// Apply `U(θ, φ, λ)` to `target` under the given modifiers.
    async fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        modifiers: &GateModifiers,
    );

    /// Apply the global phase `γ` under the given modifiers. A
    /// controlled global phase is a relative phase (the `p`/phase gate).
    async fn gphase(&mut self, gamma: f64, modifiers: &GateModifiers);

    /// Measure `qubit` in the Z basis, collapsing the state, and return
    /// the outcome.
    async fn measure(&mut self, qubit: u32) -> bool;

    /// Reset `qubit` to |0⟩.
    async fn reset(&mut self, qubit: u32);

    /// A scheduling barrier across `qubits`. No effect on state.
    async fn barrier(&mut self, _qubits: &[u32]) {}

    /// An idle delay on `qubits`. No effect on state in the MVP.
    async fn delay(&mut self, _qubits: &[u32], _duration: Duration) {}

    /// Snapshot the full amplitude vector as `f64` (global phase
    /// unresolved), or `None` for backends without an addressable state
    /// vector (e.g. hardware). Used for `--state` printing and tests.
    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        None
    }
}

/// Forward a boxed backend so the VM can be driven by a runtime-selected
/// backend (`Box<dyn QuantumBackend>`) without monomorphizing per choice.
#[async_trait(?Send)]
impl QuantumBackend for Box<dyn QuantumBackend> {
    async fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        modifiers: &GateModifiers,
    ) {
        (**self).u(target, theta, phi, lambda, modifiers).await;
    }

    async fn gphase(&mut self, gamma: f64, modifiers: &GateModifiers) {
        (**self).gphase(gamma, modifiers).await;
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        (**self).measure(qubit).await
    }

    async fn reset(&mut self, qubit: u32) {
        (**self).reset(qubit).await;
    }

    async fn barrier(&mut self, qubits: &[u32]) {
        (**self).barrier(qubits).await;
    }

    async fn delay(&mut self, qubits: &[u32], duration: Duration) {
        (**self).delay(qubits, duration).await;
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        (**self).amplitudes().await
    }
}
