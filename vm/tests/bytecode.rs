//! The bytecode format end to end: programs read back from bytes run as
//! the compiled ones do, and bytes that are not a valid program are
//! rejected or run without panicking.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use async_trait::async_trait;
use num_complex::Complex;
use oqi_classical::{Duration, PrimitiveTy, ValueTy};
use oqi_compile::Program;
use oqi_compile::bytecode::{Callee, Op, ProcOwner, Terminator};
use oqi_vm::{
    DEFAULT_SEED, GateModifiers, KindResult, MAX_CALL_DEPTH, NoExterns, QuantumBackend,
    StateVectorSim, Vm, VmErrorKind,
};

/// Every fixture, except the two that take minutes to run in a debug build.
const FIXTURES: &[(&str, &str)] = &[
    ("adder", include_str!("../../fixtures/qasm/adder.qasm")),
    (
        "alignment",
        include_str!("../../fixtures/qasm/alignment.qasm"),
    ),
    ("arrays", include_str!("../../fixtures/qasm/arrays.qasm")),
    ("cphase", include_str!("../../fixtures/qasm/cphase.qasm")),
    ("dd", include_str!("../../fixtures/qasm/dd.qasm")),
    ("defcal", include_str!("../../fixtures/qasm/defcal.qasm")),
    (
        "gateteleport",
        include_str!("../../fixtures/qasm/gateteleport.qasm"),
    ),
    (
        "inverseqft1",
        include_str!("../../fixtures/qasm/inverseqft1.qasm"),
    ),
    (
        "inverseqft2",
        include_str!("../../fixtures/qasm/inverseqft2.qasm"),
    ),
    ("ipe", include_str!("../../fixtures/qasm/ipe.qasm")),
    ("qec", include_str!("../../fixtures/qasm/qec.qasm")),
    ("qft", include_str!("../../fixtures/qasm/qft.qasm")),
    ("qpt", include_str!("../../fixtures/qasm/qpt.qasm")),
    ("rb", include_str!("../../fixtures/qasm/rb.qasm")),
    ("rus", include_str!("../../fixtures/qasm/rus.qasm")),
    ("scqec", include_str!("../../fixtures/qasm/scqec.qasm")),
    ("t1", include_str!("../../fixtures/qasm/t1.qasm")),
    (
        "teleport",
        include_str!("../../fixtures/qasm/teleport.qasm"),
    ),
    ("vqe", include_str!("../../fixtures/qasm/vqe.qasm")),
];

const SLOW_FIXTURES: &[(&str, &str)] = &[
    ("msd", include_str!("../../fixtures/qasm/msd.qasm")),
    (
        "varteleport",
        include_str!("../../fixtures/qasm/varteleport.qasm"),
    ),
];

fn round_trip(program: &Program) -> Program {
    let bytes = program.to_bytes().expect("encode");
    Program::from_bytes(&bytes).unwrap_or_else(|e| panic!("decode: {e}"))
}

/// A seeded run, as text: measurements and outputs, or the error.
fn run(program: &Program) -> String {
    let sim = StateVectorSim::<f64>::try_zeroed(program.num_qubits(), DEFAULT_SEED).unwrap();
    let mut vm = Vm::new(program, sim, NoExterns);
    match pollster::block_on(vm.run()) {
        Ok(result) => {
            let outputs: Vec<String> = result
                .outputs
                .iter()
                .map(|(symbol, value)| format!("{} = {value}", program.symbol(*symbol).name))
                .collect();
            format!("{:?} {outputs:?}", result.measurements)
        }
        Err(e) => format!("error: {e}"),
    }
}

#[test]
fn every_fixture_reads_back_as_the_program_it_was() {
    for (name, source) in FIXTURES.iter().chain(SLOW_FIXTURES) {
        let program = oqi_compile::compile(source).unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = round_trip(&program);
        assert_eq!(
            decoded.to_string(),
            program.to_string(),
            "{name}: the disassembly changed"
        );
    }
}

#[test]
fn every_fixture_runs_the_same_from_bytes() {
    for (name, source) in FIXTURES {
        let program = oqi_compile::compile(source).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(run(&round_trip(&program)), run(&program), "{name}");
    }
}

#[test]
fn recursion_deeper_than_the_limit_is_an_error() {
    let program = |depth: u32| {
        oqi_compile::compile(&format!(
            "def f(int n) -> int {{ if (n == 0) {{ return 0; }} return f(n - 1) + 1; }}
             int r = f({depth});"
        ))
        .unwrap()
    };
    // The top-level body is one level, each call of `f` another.
    let deepest = program(MAX_CALL_DEPTH - 2);
    assert_eq!(
        run(&deepest),
        format!("[] [\"r = {}\"]", MAX_CALL_DEPTH - 2)
    );

    let too_deep = program(MAX_CALL_DEPTH);
    let sim = StateVectorSim::<f64>::try_zeroed(0, 1).unwrap();
    let err = pollster::block_on(Vm::new(&too_deep, sim, NoExterns).run()).unwrap_err();
    assert!(
        matches!(err.kind, VmErrorKind::CallDepth { limit } if limit == MAX_CALL_DEPTH),
        "{err}"
    );
}

// ── Corrupted bytes ─────────────────────────────────────────────────────

/// A program without loops that uses most of the instruction set: calls
/// with classical and qubit arguments, gate modifiers, arrays, bit slices,
/// a switch, a box, a runtime alias and a `durationof` scope.
const VARIED: &str = r#"
    include "stdgates.inc";
    qubit[3] q;
    bit[3] c;
    array[int[8], 2, 2] grid = {{1, 2}, {3, 4}};
    def flip(int k, qubit a) -> bit {
        if (k > 1) { x a; }
        return measure a;
    }
    gate twist(theta) a, b { cx a, b; rz(theta) b; }
    h q[0];
    ctrl @ x q[0], q[1];
    negctrl @ inv @ pow(2) @ s q[1], q[2];
    twist(pi / 4) q[1], q[2];
    int i = 1;
    let pair = q[i:i + 1];
    reset pair;
    grid[1, 0] = grid[0, 1] + 5;
    c[0] = flip(grid[1, 0], q[2]);
    c[1:2] = measure q[0:1];
    switch (i) {
        case 0, 1 { i = 2; }
        default { i = 3; }
    }
    box { x q[0]; }
    duration d = durationof({ x q[1]; });
    bit[2] low = c[0:1];
"#;

/// A backend that fails once it has applied `left` gates: a corrupted
/// program may ask for `pow(1e18)`.
struct Budget {
    sim: StateVectorSim<f64>,
    left: u32,
}

impl Budget {
    fn spend(&mut self) -> KindResult<()> {
        self.left = self
            .left
            .checked_sub(1)
            .ok_or_else(|| VmErrorKind::Backend("gate budget spent".into()))?;
        Ok(())
    }
}

#[async_trait(?Send)]
impl QuantumBackend for Budget {
    fn num_qubits(&self) -> u32 {
        self.sim.num_qubits()
    }
    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        modifiers: &GateModifiers,
    ) -> KindResult<()> {
        self.spend()?;
        self.sim.u(target, theta, phi, lambda, modifiers)
    }
    fn gphase(&mut self, gamma: f64) -> KindResult<()> {
        self.spend()?;
        self.sim.gphase(gamma)
    }
    async fn measure(&mut self, qubit: u32) -> bool {
        self.sim.measure(qubit).await
    }
    async fn reset(&mut self, qubit: u32) {
        self.sim.reset(qubit).await
    }
    fn reset_state(&mut self) -> KindResult<()> {
        self.sim.reset_state()
    }
    fn delay(&mut self, qubits: &[u32], duration: Duration) {
        self.sim.delay(qubits, duration)
    }
    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        self.sim.amplitudes().await
    }
}

/// Whether running `program` is sure to finish quickly: no loop in any
/// procedure, no recursion, a small state and small registers. Decoding
/// keeps a corrupted program in bounds, not short: a program may loop
/// forever or allocate a large array, like a compiled one can.
fn finishes_quickly(program: &Program) -> bool {
    let small = |ty: &ValueTy| match ty {
        ValueTy::Scalar(PrimitiveTy::BitReg(width)) => *width <= 1024,
        ValueTy::Array(array) => array.shape().total() <= 4096,
        _ => true,
    };
    if program.num_qubits() > 12
        || program
            .procedures
            .iter()
            .flat_map(|p| &p.register_types)
            .any(|ty| !small(ty))
    {
        return false;
    }

    // The procedures each one enters: calls, gate bodies, nested bodies.
    let owners: HashMap<_, usize> = program
        .procedures
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p.owner {
            ProcOwner::Gate(s) | ProcOwner::Subroutine(s) => Some((s, i)),
            _ => None,
        })
        .collect();
    let mut calls: Vec<Vec<usize>> = vec![Vec::new(); program.procedures.len()];
    for (i, procedure) in program.procedures.iter().enumerate() {
        let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
        for block in &procedure.blocks {
            for instr in &block.instrs {
                let callee = match &instr.op {
                    Op::Call {
                        callee: Callee::Symbol(s),
                        ..
                    }
                    | Op::GateCall { gate: s, .. } => owners.get(s).copied(),
                    Op::CalOpenPulse { body } | Op::DurationOf { body, .. } => {
                        Some(body.0 as usize)
                    }
                    _ => None,
                };
                calls[i].extend(callee);
            }
            let targets = next.entry(block.id.0).or_default();
            match &block.terminator {
                Terminator::Goto(b) => targets.push(b.0),
                Terminator::Branch {
                    then_bb, else_bb, ..
                } => targets.extend([then_bb.0, else_bb.0]),
                Terminator::Switch { cases, default, .. } => {
                    targets.extend(cases.iter().map(|(_, b)| b.0));
                    targets.push(default.0);
                }
                Terminator::Box { body, after, .. } => targets.extend([body.0, after.0]),
                Terminator::Return(_) | Terminator::End | Terminator::Unreachable => {}
            }
        }
        if has_cycle(&next) {
            return false;
        }
    }
    let calls: HashMap<u32, Vec<u32>> = calls
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i as u32, c.into_iter().map(|c| c as u32).collect()))
        .collect();
    !has_cycle(&calls)
}

fn has_cycle(edges: &HashMap<u32, Vec<u32>>) -> bool {
    fn visit(
        node: u32,
        edges: &HashMap<u32, Vec<u32>>,
        open: &mut HashSet<u32>,
        done: &mut HashSet<u32>,
    ) -> bool {
        if done.contains(&node) {
            return false;
        }
        if !open.insert(node) {
            return true;
        }
        let cyclic = edges
            .get(&node)
            .into_iter()
            .flatten()
            .any(|&next| visit(next, edges, open, done));
        open.remove(&node);
        done.insert(node);
        cyclic
    }
    let (mut open, mut done) = (HashSet::new(), HashSet::new());
    edges
        .keys()
        .any(|&node| visit(node, edges, &mut open, &mut done))
}

/// Decode `bytes` and, when the program decodes and finishes quickly, run
/// it. Returns whether it ran; panics surface to the caller.
fn decode_and_run(bytes: &[u8]) -> bool {
    let Ok(program) = Program::from_bytes(bytes) else {
        return false;
    };
    // Whatever decodes also displays and encodes.
    let _ = program.to_string();
    let _ = program.to_bytes();
    if !finishes_quickly(&program) {
        return false;
    }
    let Ok(sim) = StateVectorSim::<f64>::try_zeroed(program.num_qubits(), 7) else {
        return false;
    };
    let backend = Budget { sim, left: 10_000 };
    let mut vm = Vm::new(&program, backend, NoExterns);
    let _ = pollster::block_on(vm.run());
    true
}

/// Every one-byte change and every truncation of two valid programs either
/// fails to decode or decodes to a program that runs without panicking.
#[test]
fn corrupted_bytecode_is_rejected_or_runs_without_panicking() {
    let teleport = FIXTURES
        .iter()
        .find(|(name, _)| *name == "teleport")
        .unwrap()
        .1;
    let mut panics = Vec::new();
    let (mut variants, mut ran) = (0, 0);
    for (name, source) in [("teleport", teleport), ("varied", VARIED)] {
        let bytes = oqi_compile::compile(source)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .to_bytes()
            .unwrap();
        let mut corrupt = |what: String, bytes: Vec<u8>| {
            variants += 1;
            match catch_unwind(AssertUnwindSafe(|| decode_and_run(&bytes))) {
                Ok(true) => ran += 1,
                Ok(false) => {}
                Err(panic) => {
                    let message = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    panics.push(format!("{name}, {what}: {message}"));
                }
            }
        };
        for at in 0..bytes.len() {
            for (how, change) in [
                ("inverted", (|b: u8| !b) as fn(u8) -> u8),
                ("incremented", |b: u8| b.wrapping_add(1)),
            ] {
                let mut changed = bytes.clone();
                changed[at] = change(changed[at]);
                corrupt(format!("byte {at} {how}"), changed);
            }
            corrupt(format!("cut at {at}"), bytes[..at].to_vec());
        }
    }
    eprintln!("{ran} of {variants} corrupted programs decoded and ran");
    assert!(
        panics.is_empty(),
        "{} panics:\n{}",
        panics.len(),
        panics.join("\n")
    );
    // The test means something only if many corrupted programs decode.
    assert!(ran > variants / 20, "only {ran} of {variants} variants ran");
}

/// A gate cannot act on one qubit through two of its operands; every
/// backend reports it instead of panicking or computing something else.
#[test]
fn a_qubit_used_twice_in_one_gate_is_an_error() {
    for source in [
        "include \"stdgates.inc\"; qubit[2] q; ctrl @ x q[0], q[0];",
        "include \"stdgates.inc\"; qubit[2] q; cx q[1], q[1];",
        "include \"stdgates.inc\"; qubit[2] q; gate g a, b { cx a, b; } g q[0], q[0];",
        "qubit[2] q; ctrl @ ctrl @ gphase(0.5) q[1], q[1];",
    ] {
        let program = oqi_compile::compile(source).unwrap();
        for name in ["scalar", "simd", "auto", "mps"] {
            let spec: oqi_vm::BackendSpec = name.parse().unwrap();
            let (backend, _) = pollster::block_on(spec.build(program.num_qubits())).unwrap();
            let err = pollster::block_on(Vm::new(&program, backend, NoExterns).run())
                .err()
                .unwrap_or_else(|| panic!("{name}: {source} ran"));
            assert!(
                err.to_string().contains("more than one operand"),
                "{name}: {err}"
            );
        }
    }
}
