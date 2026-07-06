//! Calibration handling: OpenPulse handler dispatch, defcal execution,
//! and opaque cal delivery.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use num_complex::Complex64;
use oqi_classical::{Duration, DurationUnit};
use oqi_compile::bytecode::{BcModule, emit};
use oqi_compile::lower::compile_source;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::{cfg, qubits, ssa};
use oqi_vm::{
    FrameHandle, NoExterns, OpaqueCalHandler, OpenPulseHandler, PortHandle, Result as VmResult,
    StateVectorSim, Vm, VmErrorKind, WaveformHandle,
};

/// Drive an async VM call to completion on a shared multi-threaded tokio
/// runtime (mirrors `run.rs`, which each test crate must carry itself).
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| tokio::runtime::Runtime::new().expect("tokio runtime"))
        .block_on(fut)
}

/// Compile source straight through to a bytecode module.
fn build(src: &str) -> BcModule {
    let program = compile_source(src, DefaultIncludeResolver, None).expect("compile");
    let cfgs = cfg::build_program(&program).expect("cfg");
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    emit(&ssa, &program, layout).expect("emit")
}

fn sim_for(module: &BcModule) -> StateVectorSim {
    StateVectorSim::with_seed(module.qubits.num_qubits, 0xABCD)
}

const DEFCAL_QASM: &str = include_str!("../../fixtures/qasm/defcal.qasm");

/// Everything the mock handler saw, in call order. Durations are
/// compared as `(value, unit)` (`Duration` itself has no `PartialEq`).
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Port(String),
    NewFrame {
        port: u64,
        frequency: f64,
        phase: f64,
    },
    Gaussian {
        amp: f64,
        duration: (f64, DurationUnit),
        sigma: (f64, DurationUnit),
    },
    Play {
        frame: u64,
        waveform: u64,
    },
    Capture {
        frame: u64,
        samples: u64,
    },
    ShiftPhase {
        frame: u64,
        phase: f64,
    },
    Threshold {
        iq: Complex64,
        discriminator: u64,
    },
}

/// Records every call and mints sequential handles starting at 1.
/// `capture` returns a fixed IQ point; `threshold` returns `true`.
/// Remembers each gaussian's duration so `waveform_duration` (used by
/// `durationof` timing passes) can report it.
struct MockPulse {
    events: Rc<RefCell<Vec<Event>>>,
    next_handle: u64,
    waveforms: HashMap<u64, Duration>,
}

impl MockPulse {
    fn new() -> (Self, Rc<RefCell<Vec<Event>>>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mock = MockPulse {
            events: events.clone(),
            next_handle: 1,
            waveforms: HashMap::new(),
        };
        (mock, events)
    }

    fn mint(&mut self) -> u64 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }
}

const MOCK_IQ: Complex64 = Complex64::new(0.25, -0.5);

impl OpenPulseHandler for MockPulse {
    fn port(&mut self, name: &str) -> VmResult<PortHandle> {
        self.events.borrow_mut().push(Event::Port(name.to_string()));
        Ok(PortHandle(self.mint()))
    }

    fn new_frame(&mut self, port: PortHandle, frequency: f64, phase: f64) -> VmResult<FrameHandle> {
        self.events.borrow_mut().push(Event::NewFrame {
            port: port.0,
            frequency,
            phase,
        });
        Ok(FrameHandle(self.mint()))
    }

    fn gaussian(
        &mut self,
        amp: f64,
        duration: Duration,
        sigma: Duration,
    ) -> VmResult<WaveformHandle> {
        self.events.borrow_mut().push(Event::Gaussian {
            amp,
            duration: (duration.value, duration.unit),
            sigma: (sigma.value, sigma.unit),
        });
        let h = self.mint();
        self.waveforms.insert(h, duration);
        Ok(WaveformHandle(h))
    }

    fn play(&mut self, frame: FrameHandle, waveform: WaveformHandle) -> VmResult<()> {
        self.events.borrow_mut().push(Event::Play {
            frame: frame.0,
            waveform: waveform.0,
        });
        Ok(())
    }

    fn capture(&mut self, frame: FrameHandle, samples: u64) -> VmResult<Complex64> {
        self.events.borrow_mut().push(Event::Capture {
            frame: frame.0,
            samples,
        });
        Ok(MOCK_IQ)
    }

    fn shift_phase(&mut self, frame: FrameHandle, phase: f64) -> VmResult<()> {
        self.events.borrow_mut().push(Event::ShiftPhase {
            frame: frame.0,
            phase,
        });
        Ok(())
    }

    fn threshold(&mut self, iq: Complex64, discriminator: u64) -> VmResult<bool> {
        self.events
            .borrow_mut()
            .push(Event::Threshold { iq, discriminator });
        Ok(true)
    }

    fn waveform_duration(&mut self, waveform: WaveformHandle) -> VmResult<Duration> {
        self.waveforms
            .get(&waveform.0)
            .copied()
            .ok_or_else(|| VmErrorKind::Pulse("unknown waveform".into()))
    }
}

/// The `160dt`/`40dt` literals under the default `dt` of 1µs.
const DT160: (f64, DurationUnit) = (160.0, DurationUnit::Us);
const DT40: (f64, DurationUnit) = (40.0, DurationUnit::Us);

fn gaussian_wf() -> Event {
    Event::Gaussian {
        amp: 0.1,
        duration: DT160,
        sigma: DT40,
    }
}

/// The full defcal fixture driven by gate/measure calls: every pulse
/// operation lands on the handler in program order, the cx defcal
/// recursively dispatches the defcals it calls, and the measure defcal's
/// thresholded bit becomes the measurement result.
#[test]
fn defcal_fixture_dispatches_to_pulse_handler() {
    let src = format!("{DEFCAL_QASM}\nx $0;\nrz(1.5) $1;\ncx $0, $1;\nbit c = measure $0;\n");
    let module = build(&src);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    let result = block_on(vm.run()).expect("run");

    // The measure defcal's threshold(…) = true is the measured bit.
    assert_eq!(result.measurements, vec![(0, true)]);
    let c = result
        .outputs
        .iter()
        .find(|(sym, _)| module.symbols.get(*sym).name == "c")
        .map(|(_, v)| v.to_string())
        .expect("output c");
    assert_eq!(c, "1");

    // Handles: ports d0=1, d1=2; frames drive0=3, drive1=4, cr1=5,
    // meas0=6; waveforms mint 7.. per gaussian call.
    let expected = vec![
        // Inline cal block: entry CalLoads (symbol order), then newframes.
        Event::Port("d0".into()),
        Event::Port("d1".into()),
        Event::NewFrame {
            port: 1,
            frequency: 5.0e9,
            phase: 0.0,
        }, // drive0=3
        Event::NewFrame {
            port: 2,
            frequency: 5.1e9,
            phase: 0.0,
        }, // drive1=4
        Event::NewFrame {
            port: 1,
            frequency: 5.2e9,
            phase: 0.0,
        }, // cr1=5
        Event::NewFrame {
            port: 1,
            frequency: 6.0e9,
            phase: 0.0,
        }, // meas0=6
        // x $0
        gaussian_wf(), // 7
        Event::Play {
            frame: 3,
            waveform: 7,
        },
        // rz(1.5) $1 — generic defcal; theta arrives in radians, negated.
        Event::ShiftPhase {
            frame: 3,
            phase: -1.5,
        },
        // cx $0, $1 — recursive dispatch of zx90_ix / x defcals.
        gaussian_wf(), // 8: zx90_ix $0, $1
        Event::Play {
            frame: 5,
            waveform: 8,
        },
        gaussian_wf(), // 9: x $0
        Event::Play {
            frame: 3,
            waveform: 9,
        },
        Event::ShiftPhase {
            frame: 5,
            phase: 0.5,
        },
        gaussian_wf(), // 10: zx90_ix $0, $1
        Event::Play {
            frame: 5,
            waveform: 10,
        },
        gaussian_wf(), // 11: x $0
        Event::Play {
            frame: 3,
            waveform: 11,
        },
        gaussian_wf(), // 12: x $1
        Event::Play {
            frame: 4,
            waveform: 12,
        },
        // measure $0
        gaussian_wf(), // 13
        Event::Play {
            frame: 6,
            waveform: 13,
        },
        Event::Capture {
            frame: 6,
            samples: 2048,
        },
        Event::Threshold {
            iq: MOCK_IQ,
            discriminator: 1234,
        },
    ];
    assert_eq!(*events.borrow(), expected);
}

/// Without a handler, defcal.qasm runs exactly as before this feature:
/// the inline cal block and defcals are dormant, nothing is measured.
#[test]
fn defcal_fixture_is_inert_without_handler() {
    let module = build(DEFCAL_QASM);
    let sim = sim_for(&module);
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = block_on(vm.run()).expect("run");
    assert!(result.measurements.is_empty());
}

/// A generic `defcal x q` alongside stdgates' x, no handler installed:
/// the unitary gate applies and the qubit measures 1.
const T1_STYLE: &str = r#"
    include "stdgates.inc";
    defcalgrammar "openpulse";
    cal {
        extern port d0;
        frame drive0 = newframe(d0, 5.0e9, 0.0);
    }
    defcal x q {
        waveform wf = gaussian(0.1, 100dt, 30dt);
        play(drive0, wf);
    }
    x $0;
    bit c = measure $0;
"#;

#[test]
fn defcal_dormant_without_handler_gate_body_applies() {
    let module = build(T1_STYLE);
    let sim = sim_for(&module);
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = block_on(vm.run()).expect("run");
    assert_eq!(result.measurements, vec![(0, true)]);
}

/// With a handler, the defcal replaces the stdgates body
/// (docs/pulses.rst): pulses are emitted, the backend state never
/// flips, and the (non-defcal'd) measure reads 0.
#[test]
fn defcal_beats_gate_body_with_handler() {
    let module = build(T1_STYLE);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    let result = block_on(vm.run()).expect("run");
    assert_eq!(result.measurements, vec![(0, false)]);
    let events = events.borrow();
    assert!(events.contains(&Event::Port("d0".into())));
    assert!(
        events.iter().any(|e| matches!(e, Event::Play { .. })),
        "generic defcal x should have played a pulse: {events:?}"
    );
}

/// An exact-operand defcal outranks a generic one for its qubit; the
/// generic one still serves other qubits. (The spec's specificity rule;
/// declaration order deliberately puts the generic first.)
#[test]
fn exact_operand_defcal_outranks_generic() {
    let src = r#"
        defcalgrammar "openpulse";
        cal {
            extern port d0;
            frame drive0 = newframe(d0, 5.0e9, 0.0);
        }
        defcal x q { shift_phase(drive0, 0.75); }
        defcal x $0 { shift_phase(drive0, 0.25); }
        x $0;
        x $1;
    "#;
    let module = build(src);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    block_on(vm.run()).expect("run");
    let phases: Vec<f64> = events
        .borrow()
        .iter()
        .filter_map(|e| match e {
            Event::ShiftPhase { phase, .. } => Some(*phase),
            _ => None,
        })
        .collect();
    assert_eq!(phases, vec![0.25, 0.75]);
}

/// `reset $0` dispatches to a reset defcal.
#[test]
fn reset_defcal_dispatches() {
    let src = r#"
        defcalgrammar "openpulse";
        cal {
            extern port d0;
            frame drive0 = newframe(d0, 5.0e9, 0.0);
        }
        defcal reset $0 {
            waveform wf = gaussian(0.1, 160dt, 40dt);
            play(drive0, wf);
        }
        reset $0;
    "#;
    let module = build(src);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    block_on(vm.run()).expect("run");
    let events = events.borrow();
    assert!(
        events.iter().any(|e| matches!(e, Event::Play { .. })),
        "reset defcal should have played a pulse: {events:?}"
    );
}

/// A defcal referencing an `extern frame` the handler doesn't provide
/// surfaces the default `extern_frame` error, naming the frame.
#[test]
fn unprovided_extern_frame_errors() {
    let src = r#"
        defcalgrammar "openpulse";
        cal {
            extern frame ef;
        }
        defcal x $0 {
            waveform wf = gaussian(0.1, 160dt, 40dt);
            play(ef, wf);
        }
        x $0;
    "#;
    let module = build(src);
    let sim = sim_for(&module);
    let (mock, _events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    let err = block_on(vm.run()).expect_err("unprovided extern frame should error");
    assert!(
        err.to_string().contains("ef"),
        "error should name the frame: {err}"
    );
}

/// Recorded opaque-cal deliveries: `(grammar, text)`.
type CalCalls = Rc<RefCell<Vec<(Option<String>, String)>>>;

struct MockCal(CalCalls);

impl OpaqueCalHandler for MockCal {
    fn cal(&mut self, grammar: Option<&str>, text: &str) -> VmResult<()> {
        self.0
            .borrow_mut()
            .push((grammar.map(str::to_string), text.to_string()));
        Ok(())
    }
}

const OPAQUE_SRC: &str = r#"
    defcalgrammar "mypulses";
    cal { raw pulse text }
"#;

/// Inline cal text in a foreign grammar reaches the opaque handler with
/// the (quote-stripped) grammar name.
#[test]
fn opaque_cal_reaches_handler() {
    let module = build(OPAQUE_SRC);
    let sim = sim_for(&module);
    let calls = Rc::new(RefCell::new(Vec::new()));
    let mut vm = Vm::new(&module, sim, NoExterns).with_opaque_cal_handler(MockCal(calls.clone()));
    block_on(vm.run()).expect("run");
    let calls = calls.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0.as_deref(), Some("mypulses"));
    assert!(
        calls[0].1.contains("raw pulse text"),
        "text: {:?}",
        calls[0].1
    );
}

/// Without an opaque handler the cal block stays a no-op.
#[test]
fn opaque_cal_is_inert_without_handler() {
    let module = build(OPAQUE_SRC);
    let sim = sim_for(&module);
    let mut vm = Vm::new(&module, sim, NoExterns);
    block_on(vm.run()).expect("run");
}

// ── durationof over calibrated gates ─────────────────────────────────

/// Extract a named duration output as its display string.
fn output_str(module: &BcModule, result: &oqi_vm::RunResult, name: &str) -> String {
    result
        .outputs
        .iter()
        .find(|(sym, _)| module.symbols.get(*sym).name == name)
        .map(|(_, v)| v.to_string())
        .unwrap_or_else(|| panic!("output {name}"))
}

/// `durationof` over a defcal'd gate reports the calibrated pulse
/// duration (handler-supplied), and the timing pass plays nothing —
/// only constructors run.
#[test]
fn durationof_uses_calibrated_pulse_durations() {
    let src = r#"
        defcalgrammar "openpulse";
        cal {
            extern port d0;
            frame drive0 = newframe(d0, 5.0e9, 0.0);
        }
        defcal x q {
            waveform wf = gaussian(0.1, 100dt, 30dt);
            play(drive0, wf);
        }
        duration d = durationof({ x $1; });
    "#;
    let module = build(src);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    let result = block_on(vm.run()).expect("run");
    // 100dt @ the default dt of 1us.
    assert_eq!(output_str(&module, &result, "d"), "100us");
    let events = events.borrow();
    assert!(
        !events.iter().any(|e| matches!(e, Event::Play { .. })),
        "timing pass must not play: {events:?}"
    );
}

/// Frames are parallel timelines: `durationof({cx $0, $1;})` over the
/// defcal fixture is the busiest frame's total (cr1 and drive0 each
/// play twice at 160us), exercising recursive dispatch under timing.
#[test]
fn durationof_takes_max_across_frames() {
    let src = format!("{DEFCAL_QASM}\nduration d = durationof({{ cx $0, $1; }});\n");
    let module = build(&src);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    let result = block_on(vm.run()).expect("run");
    assert_eq!(output_str(&module, &result, "d"), "320us");
    let events = events.borrow();
    assert!(!events.iter().any(|e| matches!(e, Event::Play { .. })));
}

/// A handler that doesn't implement `waveform_duration` surfaces the
/// default error when `durationof` needs it.
struct MinimalPulse;

impl OpenPulseHandler for MinimalPulse {
    fn port(&mut self, _name: &str) -> VmResult<PortHandle> {
        Ok(PortHandle(1))
    }
    fn new_frame(&mut self, _port: PortHandle, _freq: f64, _phase: f64) -> VmResult<FrameHandle> {
        Ok(FrameHandle(2))
    }
    fn gaussian(&mut self, _amp: f64, _d: Duration, _s: Duration) -> VmResult<WaveformHandle> {
        Ok(WaveformHandle(3))
    }
    fn play(&mut self, _f: FrameHandle, _w: WaveformHandle) -> VmResult<()> {
        Ok(())
    }
    fn capture(&mut self, _f: FrameHandle, _s: u64) -> VmResult<Complex64> {
        Ok(Complex64::new(0.0, 0.0))
    }
    fn shift_phase(&mut self, _f: FrameHandle, _p: f64) -> VmResult<()> {
        Ok(())
    }
    fn threshold(&mut self, _iq: Complex64, _d: u64) -> VmResult<bool> {
        Ok(false)
    }
}

#[test]
fn durationof_without_waveform_durations_errors() {
    let src = r#"
        defcalgrammar "openpulse";
        cal {
            extern port d0;
            frame drive0 = newframe(d0, 5.0e9, 0.0);
        }
        defcal x $0 {
            waveform wf = gaussian(0.1, 160dt, 40dt);
            play(drive0, wf);
        }
        duration d = durationof({ x $0; });
    "#;
    let module = build(src);
    let sim = sim_for(&module);
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(MinimalPulse);
    let err = block_on(vm.run()).expect_err("default waveform_duration should error");
    assert!(
        err.to_string().contains("waveform durations"),
        "error should mention waveform durations: {err}"
    );
}

/// Timing a gate leaves it fully functional: the later real call is
/// the only one that plays.
#[test]
fn timed_gate_still_executes_when_called() {
    let src = r#"
        defcalgrammar "openpulse";
        cal {
            extern port d0;
            frame drive0 = newframe(d0, 5.0e9, 0.0);
        }
        defcal x $0 {
            waveform wf = gaussian(0.1, 160dt, 40dt);
            play(drive0, wf);
        }
        duration d = durationof({ x $0; });
        x $0;
    "#;
    let module = build(src);
    let sim = sim_for(&module);
    let (mock, events) = MockPulse::new();
    let mut vm = Vm::new(&module, sim, NoExterns).with_pulse_handler(mock);
    let result = block_on(vm.run()).expect("run");
    assert_eq!(output_str(&module, &result, "d"), "160us");
    let plays = events
        .borrow()
        .iter()
        .filter(|e| matches!(e, Event::Play { .. }))
        .count();
    assert_eq!(plays, 1, "only the real call plays");
}
