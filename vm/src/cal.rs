//! Calibration handling: the [`OpenPulseHandler`] and [`OpaqueCalHandler`]
//! extension points (documented on the traits).

use num_complex::Complex64;
use oqi_classical::Duration;

use crate::error::{ErrorKind, Result};

/// Opaque handle to a handler-owned port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortHandle(pub u64);

/// Opaque handle to a handler-owned frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameHandle(pub u64);

/// Opaque handle to a handler-owned waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaveformHandle(pub u64);

/// Host implementation of the OpenPulse intrinsics (`newframe`, `gaussian`,
/// `play`, `capture`, `shift_phase`, `threshold`). Angles arrive in
/// radians; durations keep their source unit (e.g. `dt`).
///
/// Installing one ([`Vm::with_pulse_handler`](crate::Vm::with_pulse_handler))
/// activates calibration execution: inline `cal` blocks run, and gate,
/// `measure` and `reset` operations on physical qubits that match a `defcal`
/// run the defcal body in place of the gate's unitary definition
/// (docs/pulses.rst), routing pulse operations here. Without a handler the
/// VM ignores calibrations entirely: `cal` blocks are no-ops and defcals stay
/// dormant.
///
/// Pulse values (ports, frames, waveforms) are opaque `u64` handles minted
/// by the handler; the VM stores them and never inspects them.
///
/// # `durationof`
///
/// A `durationof({...})` the compiler did not resolve runs its block in a
/// timing pass: no gates, pulses, measurements or resets take effect;
/// `delay[d]` and `box[d]` advance a base clock, and dispatched defcal bodies
/// advance per-frame clocks by [`waveform_duration`](Self::waveform_duration)
/// and [`capture_duration`](Self::capture_duration). Constructors
/// (`newframe`, `gaussian`, ports) still call the handler, so handles are
/// real and their durations can be queried. The result is the maximum across
/// all timelines (frames and the qubit timeline are independent; a `barrier`
/// synchronizes them). Gates without a dispatched calibration take no time.
/// A `measure` in a timing pass yields `0` and records nothing: the spec
/// requires calibration branches to have equal durations, so the value must
/// not matter.
///
/// # Errors
///
/// Every method may return an [`ErrorKind`] (usually [`ErrorKind::Pulse`]),
/// which stops the run; the defaults of the optional methods do.
///
/// # Limitations
///
/// - Calls with `ctrl`, `negctrl`, `pow` or `inv` modifiers (or inherited
///   controls) never dispatch to a defcal; they take the unitary path.
/// - Defcals specialised on an expression (`defcal rx(pi/2) $0`) and
///   `delay` defcals never dispatch.
pub trait OpenPulseHandler {
    /// Resolve an `extern port` declaration by name.
    fn port(&mut self, name: &str) -> Result<PortHandle, ErrorKind>;

    /// Resolve an `extern frame` declaration by name. Defaults to an
    /// error, for handlers that don't pre-provision frames.
    fn extern_frame(&mut self, name: &str) -> Result<FrameHandle, ErrorKind> {
        Err(ErrorKind::Pulse(format!(
            "extern frame `{name}` is not provided"
        )))
    }

    /// `newframe(port, frequency, phase)`.
    fn new_frame(
        &mut self,
        port: PortHandle,
        frequency: f64,
        phase: f64,
    ) -> Result<FrameHandle, ErrorKind>;

    /// `gaussian(amp, duration, sigma)`.
    fn gaussian(
        &mut self,
        amp: f64,
        duration: Duration,
        sigma: Duration,
    ) -> Result<WaveformHandle, ErrorKind>;

    /// `play(frame, waveform)`.
    fn play(&mut self, frame: FrameHandle, waveform: WaveformHandle) -> Result<(), ErrorKind>;

    /// `capture(frame, samples)`: acquire a raw IQ value.
    fn capture(&mut self, frame: FrameHandle, samples: u64) -> Result<Complex64, ErrorKind>;

    /// `shift_phase(frame, phase)`.
    fn shift_phase(&mut self, frame: FrameHandle, phase: f64) -> Result<(), ErrorKind>;

    /// `threshold(iq, discriminator)`: classify an IQ value into a bit.
    fn threshold(&mut self, iq: Complex64, discriminator: u64) -> Result<bool, ErrorKind>;

    /// Duration of a waveform, queried during `durationof` timing
    /// passes (a `play` advances its frame's clock by this much
    /// instead of emitting the pulse). Defaults to an error —
    /// implement it to let `durationof` resolve calibrated gates.
    fn waveform_duration(&mut self, waveform: WaveformHandle) -> Result<Duration, ErrorKind> {
        let _ = waveform;
        Err(ErrorKind::Pulse(
            "handler does not report waveform durations".into(),
        ))
    }

    /// Duration of a `capture`, queried during `durationof` timing
    /// passes. Defaults to an error.
    fn capture_duration(
        &mut self,
        frame: FrameHandle,
        samples: u64,
    ) -> Result<Duration, ErrorKind> {
        let _ = (frame, samples);
        Err(ErrorKind::Pulse(
            "handler does not report capture durations".into(),
        ))
    }
}

/// Host handler for calibration text in a non-OpenPulse grammar: it
/// receives the raw text of each inline `cal` block when the program's
/// `defcalgrammar` is not `"openpulse"`. Without one those blocks are
/// no-ops.
///
/// # Errors
///
/// An [`ErrorKind`] from [`cal`](Self::cal) stops the run.
pub trait OpaqueCalHandler {
    /// Handle the raw text of an inline `cal { ... }` block. `grammar`
    /// is the program's `defcalgrammar` (quotes stripped), if declared.
    fn cal(&mut self, grammar: Option<&str>, text: &str) -> Result<(), ErrorKind>;
}
