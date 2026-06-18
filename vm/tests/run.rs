//! End-to-end tests: compile OpenQASM source to bytecode, then run it.

use oqi_classical::{Value, iw};
use oqi_compile::bytecode::{BcModule, emit};
use oqi_compile::lower::compile_source;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::{cfg, qubits, ssa};
use oqi_vm::{FnRegistry, NoExterns, StateVectorSim, Vm, VmError};

/// Compile source straight through to a bytecode module.
fn build(src: &str) -> BcModule {
    let program = compile_source(src, DefaultIncludeResolver, None).expect("compile");
    let cfgs = cfg::build_program(&program).expect("cfg");
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    emit(&ssa, &program.symbols, layout).expect("emit")
}

/// Run with the CPU simulator and no externs; return the measurement log.
fn run_measurements(src: &str) -> Vec<(u32, bool)> {
    let module = build(src);
    let sim = StateVectorSim::with_seed(module.qubits.num_qubits, 0xABCD);
    let mut vm = Vm::new(&module, sim, NoExterns);
    vm.run().expect("run").measurements
}

const TOL: f64 = 1e-9;

#[test]
fn classical_arithmetic_drives_a_branch() {
    // x = 1, y = x + 2 = 3; the (y == 3) branch flips the qubit.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            int v = 1;
            int w = v + 2;
            if (w == 3) { x q; }
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true)]);
}

#[test]
fn untaken_branch_leaves_qubit_in_ground_state() {
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            int v = 0;
            if (v == 1) { x q; }
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, false)]);
}

#[test]
fn bell_state_amplitudes() {
    let module = build(
        r#"
            include "stdgates.inc";
            qubit[2] q;
            h q[0];
            cx q[0], q[1];
        "#,
    );
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    vm.run().expect("run");
    let amps = vm.backend().state();
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (amps[0].norm() - inv_sqrt2).abs() < TOL,
        "|00>: {:?}",
        amps[0]
    );
    assert!(amps[1].norm() < TOL, "|01>: {:?}", amps[1]);
    assert!(amps[2].norm() < TOL, "|10>: {:?}", amps[2]);
    assert!(
        (amps[3].norm() - inv_sqrt2).abs() < TOL,
        "|11>: {:?}",
        amps[3]
    );
}

#[test]
fn h_z_h_is_x_via_parametric_gate_path() {
    // HZH = X. Exercises z -> p(π) -> ctrl @ gphase(π): the classical
    // calling convention (param register binding) and controlled-gphase.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            h q;
            z q;
            h q;
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true)]);
}

#[test]
fn rx_pi_flips_qubit() {
    // rx(π) is a bit flip (up to phase). Exercises a parametric gate
    // with a runtime/const angle argument bound to a param register.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            rx(pi) q;
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true)]);
}

#[test]
fn reset_returns_to_ground_state() {
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            x q;
            reset q;
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, false)]);
}

#[test]
fn extern_function_result_is_used() {
    let module = build(
        r#"
            include "stdgates.inc";
            extern inc(int[32]) -> int[32];
            qubit q;
            int[32] v = inc(5);
            if (v == 6) { x q; }
            bit c = measure q;
        "#,
    );
    let mut registry = FnRegistry::new();
    registry.register("inc", |args: &[Value]| {
        let n = match &args[0] {
            Value::Scalar(s) => s.value().as_int(iw(32)).unwrap(),
            _ => panic!("expected scalar"),
        };
        Ok(Some(Value::int(n + 1, iw(32))))
    });
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, registry);
    let result = vm.run().expect("run");
    assert_eq!(result.measurements, vec![(0, true)]);
}

#[test]
fn subroutine_call_with_argument_and_return() {
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            def add_one(int[32] a) -> int[32] {
                return a + 1;
            }
            qubit q;
            int[32] v = add_one(5);
            if (v == 6) { x q; }
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true)]);
}

#[test]
fn intrinsic_call_is_evaluated() {
    // popcount(7) == 3 routes through BcCallTarget::Intrinsic.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            bit[4] b = "0111";
            uint[32] n = popcount(b);
            if (n == 3) { x q; }
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true)]);
}

#[test]
fn equal_length_register_broadcast() {
    // `x q` over a 3-qubit register flips all three; `cx a, b` over two
    // equal-length registers zips pairwise. All end up |1>.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[3] q;
            x q;
            bit[3] c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true), (1, true), (2, true)]);
}

#[test]
fn mismatched_register_broadcast_is_rejected() {
    // cx over a length-2 and a length-3 register must error, not silently
    // reuse an element.
    let module = build(
        r#"
            include "stdgates.inc";
            qubit[2] a;
            qubit[3] b;
            cx a, b;
        "#,
    );
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    match vm.run() {
        Err(VmError::BroadcastMismatch(lengths)) => assert_eq!(lengths, vec![2, 3]),
        other => panic!("expected BroadcastMismatch, got {other:?}"),
    }
}

#[test]
fn teleport_fixture_runs_end_to_end() {
    let src = include_str!("../../fixtures/qasm/teleport.qasm");
    let m = run_measurements(src);
    // teleport.qasm performs three measurements (c0, c1, c2).
    assert_eq!(m.len(), 3, "expected 3 measurements, got {m:?}");
}
