//! Pluggable calibration handling.
//!
//! Two extension points, mirroring [`ExternProvider`](crate::ExternProvider):
//!
//! - [`OpenPulseHandler`] — typed callbacks for the OpenPulse intrinsics
//!   (`newframe`, `gaussian`, `play`, `capture`, `shift_phase`,
//!   `threshold`). Installing one activates calibration execution:
//!   inline `cal` blocks run, and gate/measure/reset operations on
//!   hardware qubits that match a `defcal` execute the defcal body in
//!   place of the gate's unitary definition (docs/pulses.rst), routing
//!   pulse operations to the handler.
//! - [`OpaqueCalHandler`] — receives the raw text of inline `cal`
//!   blocks written in a non-OpenPulse `defcalgrammar`.
//!
//! Without a handler installed the VM ignores calibrations entirely:
//! inline `cal` blocks are no-ops and defcals stay dormant.
//!
//! Pulse values (ports, frames, waveforms) are opaque `u64` handles
//! minted by the handler; the VM stores them in ordinary `uint[64]`
//! registers and never inspects them.
//!
//! # `durationof` timing passes
//!
//! A `durationof({...})` expression runs its block in a timing pass:
//! no gates, pulses, measurements, or resets are emitted; instead
//! `delay[d]`/`box[d]` advance a base clock, and — with a pulse
//! handler installed — dispatched defcal bodies advance per-frame
//! clocks by [`OpenPulseHandler::waveform_duration`] /
//! [`OpenPulseHandler::capture_duration`]. Constructor intrinsics
//! (`newframe`, `gaussian`, port minting) still call the handler so
//! handles are real and their durations queryable. The result is the
//! maximum across all timelines (frames and the qubit timeline are
//! modeled as independent; a `barrier` synchronizes them). Gates with
//! no dispatched calibration contribute zero. A `measure` in a timing
//! pass yields a deterministic `0` and records nothing — the spec
//! requires control-flow branches in calibrations to have equal
//! durations, so the value must not matter (docs/pulses.rst).
//!
//! Dispatch is deliberately conservative. Known limitations:
//! - Calls carrying `ctrl`/`negctrl`/`pow`/`inv` modifiers (or
//!   inherited controls) never dispatch to a defcal — they take the
//!   unitary path.
//! - Expression-specialized defcals (`defcal rx(pi/2) $0`) and `delay`
//!   defcals never dispatch.
//! - Classical (non-pulse) values declared in `cal` blocks are not
//!   shared across cal/defcal bodies; reading one from another body
//!   errors with `UnsetRegister`. Likewise a `durationof` body cannot
//!   read classical variables from its enclosing scope.

use num_complex::Complex64;
use oqi_classical::Duration;

use crate::error::{Result, VmErrorKind};

/// Opaque handle to a handler-owned port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortHandle(pub u64);

/// Opaque handle to a handler-owned frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameHandle(pub u64);

/// Opaque handle to a handler-owned waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaveformHandle(pub u64);

/// Host implementation of the OpenPulse intrinsics. Angles arrive in
/// radians; durations keep their source unit (e.g. `dt`).
pub trait OpenPulseHandler {
    /// Resolve an `extern port` declaration by name.
    fn port(&mut self, name: &str) -> Result<PortHandle>;

    /// Resolve an `extern frame` declaration by name. Defaults to an
    /// error, for handlers that don't pre-provision frames.
    fn extern_frame(&mut self, name: &str) -> Result<FrameHandle> {
        Err(VmErrorKind::Pulse(format!(
            "extern frame `{name}` is not provided"
        )))
    }

    /// `newframe(port, frequency, phase)`.
    fn new_frame(&mut self, port: PortHandle, frequency: f64, phase: f64) -> Result<FrameHandle>;

    /// `gaussian(amp, duration, sigma)`.
    fn gaussian(&mut self, amp: f64, duration: Duration, sigma: Duration)
    -> Result<WaveformHandle>;

    /// `play(frame, waveform)`.
    fn play(&mut self, frame: FrameHandle, waveform: WaveformHandle) -> Result<()>;

    /// `capture(frame, samples)`: acquire a raw IQ value.
    fn capture(&mut self, frame: FrameHandle, samples: u64) -> Result<Complex64>;

    /// `shift_phase(frame, phase)`.
    fn shift_phase(&mut self, frame: FrameHandle, phase: f64) -> Result<()>;

    /// `threshold(iq, discriminator)`: classify an IQ value into a bit.
    fn threshold(&mut self, iq: Complex64, discriminator: u64) -> Result<bool>;

    /// Duration of a waveform, queried during `durationof` timing
    /// passes (a `play` advances its frame's clock by this much
    /// instead of emitting the pulse). Defaults to an error —
    /// implement it to let `durationof` resolve calibrated gates.
    fn waveform_duration(&mut self, waveform: WaveformHandle) -> Result<Duration> {
        let _ = waveform;
        Err(VmErrorKind::Pulse(
            "handler does not report waveform durations".into(),
        ))
    }

    /// Duration of a `capture`, queried during `durationof` timing
    /// passes. Defaults to an error.
    fn capture_duration(&mut self, frame: FrameHandle, samples: u64) -> Result<Duration> {
        let _ = (frame, samples);
        Err(VmErrorKind::Pulse(
            "handler does not report capture durations".into(),
        ))
    }
}

/// Host handler for calibration text in a non-OpenPulse grammar.
pub trait OpaqueCalHandler {
    /// Handle the raw text of an inline `cal { ... }` block. `grammar`
    /// is the program's `defcalgrammar` (quotes stripped), if declared.
    fn cal(&mut self, grammar: Option<&str>, text: &str) -> Result<()>;
}
