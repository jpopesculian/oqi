//! End-to-end tests: compile OpenQASM source to bytecode, then run it.

use std::collections::HashMap;

use oqi_classical::{Value, iw};
use oqi_compile::bytecode::{BcModule, emit};
use oqi_compile::lower::compile_source;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::symbol::SymbolId;
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

/// Final state-vector amplitudes as `(re, im)` pairs (no measurement, so
/// the run is deterministic). Amplitudes ignore global phase.
fn final_state(src: &str) -> Vec<(f64, f64)> {
    let module = build(src);
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    vm.run().expect("run");
    vm.backend().state().iter().map(|c| (c.re, c.im)).collect()
}

fn assert_states_eq(a: &[(f64, f64)], b: &[(f64, f64)]) {
    assert_eq!(a.len(), b.len(), "state length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x.0 - y.0).abs() < TOL && (x.1 - y.1).abs() < TOL,
            "amp {i}: {x:?} vs {y:?}"
        );
    }
}

// A non-self-inverse composite with non-commuting factors, used by the
// modifier tests below. Operator order is `rx(1.1)·cx·rz(0.7)·cx`.
const COMPOSITE: &str = r#"
    include "stdgates.inc";
    gate g a, b { cx a, b; rz(0.7) b; cx a, b; rx(1.1) a; }
    qubit[2] q;
    h q[0];
    rx(0.9) q[1];
    cx q[0], q[1];
"#;

#[test]
fn inv_of_composite_is_a_true_inverse() {
    // `g; inv @ g` must return to the prepared state. A scalar-power fold
    // (the old behavior) inverts the factors in the wrong order and fails
    // for a non-commuting body like `g`.
    let prepared = final_state(COMPOSITE);
    let round_trip = final_state(&format!(
        "{COMPOSITE}\ng q[0], q[1];\ninv @ g q[0], q[1];\n"
    ));
    assert_states_eq(&round_trip, &prepared);
}

#[test]
fn inv_of_composite_matches_hand_reversed_body() {
    // `inv @ g` ≡ the body run in reverse with each op inverted (cx is
    // self-inverse, so `inv @ cx == cx`).
    let via_modifier = final_state(&format!("{COMPOSITE}\ninv @ g q[0], q[1];\n"));
    let hand_written = final_state(&format!(
        "{COMPOSITE}\ninv @ rx(1.1) q[0]; cx q[0], q[1]; inv @ rz(0.7) q[1]; cx q[0], q[1];\n"
    ));
    assert_states_eq(&via_modifier, &hand_written);
}

#[test]
fn integer_pow_of_composite_repeats_the_body() {
    // pow(2) @ g ≡ g; g, and pow(-1) @ g ≡ inv @ g.
    let squared = final_state(&format!("{COMPOSITE}\npow(2) @ g q[0], q[1];\n"));
    let twice = final_state(&format!("{COMPOSITE}\ng q[0], q[1];\ng q[0], q[1];\n"));
    assert_states_eq(&squared, &twice);

    let pow_neg1 = final_state(&format!("{COMPOSITE}\npow(-1) @ g q[0], q[1];\n"));
    let inv = final_state(&format!("{COMPOSITE}\ninv @ g q[0], q[1];\n"));
    assert_states_eq(&pow_neg1, &inv);
}

#[test]
fn odd_integer_pow_of_swap_is_swap() {
    // swap = cx;cx;cx is self-inverse, so pow(3) @ swap ≡ swap.
    let cubed = final_state(
        r#"
            include "stdgates.inc";
            qubit[2] q;
            h q[0];
            pow(3) @ swap q[0], q[1];
        "#,
    );
    let once = final_state(
        r#"
            include "stdgates.inc";
            qubit[2] q;
            h q[0];
            swap q[0], q[1];
        "#,
    );
    assert_states_eq(&cubed, &once);
}

#[test]
fn fractional_pow_of_single_qubit_composite_still_works() {
    // pow(0.5) @ z is `s` (one real leaf), and applying it twice equals z.
    // This must keep working through the new trace path.
    let via_pow = final_state(
        r#"
            include "stdgates.inc";
            qubit q;
            h q;
            pow(0.5) @ z q;
            pow(0.5) @ z q;
        "#,
    );
    let via_z = final_state(
        r#"
            include "stdgates.inc";
            qubit q;
            h q;
            z q;
        "#,
    );
    assert_states_eq(&via_pow, &via_z);
}

#[test]
fn fractional_pow_of_multi_qubit_composite_is_rejected() {
    // sqrt(SWAP) is a genuine 2-qubit gate; it would need a dense matrix
    // power, which is out of scope. It must error, not silently misbehave.
    let module = build(
        r#"
            include "stdgates.inc";
            qubit[2] q;
            pow(0.5) @ swap q[0], q[1];
        "#,
    );
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    match vm.run() {
        Err(VmError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn teleport_fixture_runs_end_to_end() {
    let src = include_str!("../../fixtures/qasm/teleport.qasm");
    let m = run_measurements(src);
    // teleport.qasm performs three measurements (c0, c1, c2).
    assert_eq!(m.len(), 3, "expected 3 measurements, got {m:?}");
}

/// Run and return named outputs as `(name, displayed value)`, sorted by name.
fn run_outputs(src: &str) -> Vec<(String, String)> {
    let module = build(src);
    let sim = StateVectorSim::with_seed(module.qubits.num_qubits, 0xABCD);
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = vm.run().expect("run");
    let mut out: Vec<(String, String)> = result
        .outputs
        .iter()
        .map(|(sym, v)| (module.symbols.get(*sym).name.clone(), v.to_string()))
        .collect();
    out.sort();
    out
}

#[test]
fn no_output_decl_returns_all_named_classical_vars() {
    let outs = run_outputs(
        r#"
            include "stdgates.inc";
            qubit q;
            int v = 1;
            int w = v + 2;
            if (w == 3) { x q; }
            bit c = measure q;
        "#,
    );
    assert_eq!(
        outs,
        vec![
            ("c".to_string(), "1".to_string()),
            ("v".to_string(), "1".to_string()),
            ("w".to_string(), "3".to_string()),
        ]
    );
}

#[test]
fn output_decl_returns_only_outputs_taken_branch() {
    // Output reassigned in a taken branch: final value must be from that
    // branch (exercises the reaching-def-at-exit path through a phi).
    let outs = run_outputs(
        r#"
            output int x;
            int c = 1;
            x = 1;
            if (c == 1) { x = 2; }
        "#,
    );
    assert_eq!(outs, vec![("x".to_string(), "2".to_string())]);
}

#[test]
fn output_decl_returns_only_outputs_untaken_branch() {
    let outs = run_outputs(
        r#"
            output int x;
            int c = 0;
            x = 1;
            if (c == 1) { x = 2; }
        "#,
    );
    assert_eq!(outs, vec![("x".to_string(), "1".to_string())]);
}

/// Symbol id of a declared symbol by name.
fn sym_id(module: &BcModule, name: &str) -> SymbolId {
    module.symbols.iter().find(|s| s.name == name).unwrap().id
}

const INPUT_BRANCH_SRC: &str = r#"
    include "stdgates.inc";
    input int n;
    qubit q;
    if (n == 1) { x q; }
    bit c = measure q;
"#;

#[test]
fn input_value_drives_a_branch() {
    let module = build(INPUT_BRANCH_SRC);
    let n = sym_id(&module, "n");
    for (val, expect) in [(1i128, true), (0, false)] {
        let sim = StateVectorSim::with_seed(module.qubits.num_qubits, 0xABCD);
        let mut vm = Vm::new(&module, sim, NoExterns);
        let inputs = HashMap::from([(n, Value::int(val, iw(32)))]);
        let r = vm.run_with_inputs(inputs).expect("run");
        assert_eq!(r.measurements, vec![(0, expect)], "n = {val}");
    }
}

#[test]
fn missing_input_is_rejected() {
    let module = build(INPUT_BRANCH_SRC);
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    match vm.run_with_inputs(HashMap::new()) {
        Err(VmError::MissingInput(_)) => {}
        other => panic!("expected MissingInput, got {other:?}"),
    }
}

#[test]
fn value_for_non_input_symbol_is_rejected() {
    let module = build(INPUT_BRANCH_SRC);
    let n = sym_id(&module, "n");
    let q = sym_id(&module, "q");
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    let inputs = HashMap::from([(n, Value::int(1, iw(32))), (q, Value::int(0, iw(32)))]);
    match vm.run_with_inputs(inputs) {
        Err(VmError::UnknownInput(_)) => {}
        other => panic!("expected UnknownInput, got {other:?}"),
    }
}
