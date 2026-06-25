//! End-to-end tests: compile OpenQASM source to bytecode, then run it.

use std::collections::HashMap;

use oqi_classical::{Value, iw};
use oqi_compile::bytecode::{BcModule, emit};
use oqi_compile::lower::compile_source;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::symbol::SymbolId;
use oqi_compile::{cfg, qubits, ssa};
use oqi_vm::{FnRegistry, NoExterns, StateVectorSim, Vm, VmError, VmErrorKind};

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
fn sizeof_on_unspecified_length_array_ref_is_runtime() {
    // `sizeof(a, dim)` on a `#dim` (unspecified-length) array reference is not
    // a compile-time constant; the dimension is read at run time from the
    // concrete array the subroutine receives.
    let outs = run_outputs(
        r#"
            def dims(readonly array[int[32], #dim=3] a) -> uint[32] {
                return sizeof(a, 1);
            }
            array[int[32], 2, 3, 4] data;
            output uint[32] d;
            d = dims(data);
        "#,
    );
    assert_eq!(outs, vec![("d".to_string(), "3".to_string())]);
}

#[test]
fn multi_dim_array_literal_initializes() {
    // A nested (multi-dimensional) array literal flattens to one `NewArray`
    // against the destination's full shape, producing the right values.
    let outs = run_outputs("array[int[32], 2, 3] m = { {1, 2, 3}, {4, 5, 6} };");
    assert_eq!(
        outs,
        vec![("m".to_string(), "{{1, 2, 3}, {4, 5, 6}}".to_string())]
    );
}

#[test]
fn multi_dim_element_read() {
    // `m[i, j]` reads a single element across both dimensions, not just the
    // first index.
    let outs = run_outputs(
        r#"
            array[int[32], 2, 3] m = { {1, 2, 3}, {4, 5, 6} };
            output int[32] a;
            output int[32] b;
            a = m[1, 0];
            b = m[0, 2];
        "#,
    );
    assert_eq!(
        outs,
        vec![
            ("a".to_string(), "4".to_string()),
            ("b".to_string(), "3".to_string())
        ]
    );
}

#[test]
fn multi_dim_element_write() {
    // `m[i, j] = v` updates exactly one element across both dimensions.
    let outs = run_outputs(
        r#"
            array[int[32], 2, 3] m = { {1, 2, 3}, {4, 5, 6} };
            m[1, 2] = 99;
            m[0, 0] = 7;
        "#,
    );
    assert_eq!(
        outs,
        vec![("m".to_string(), "{{7, 2, 3}, {4, 5, 99}}".to_string())]
    );
}

#[test]
fn sub_array_read_displays_the_row() {
    // `r = m[0]` reads a row of a 2-D array. The result is an array view that
    // must display as the row it refers to, not the whole base array.
    let outs = run_outputs(
        r#"
            array[int[32], 2, 2] m = { {1, 2}, {3, 4} };
            array[int[32], 2] r = m[0];
        "#,
    );
    assert_eq!(
        outs,
        vec![
            ("m".to_string(), "{{1, 2}, {3, 4}}".to_string()),
            ("r".to_string(), "{1, 2}".to_string())
        ]
    );
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
        Err(VmError { kind: VmErrorKind::BroadcastMismatch(lengths), .. }) => assert_eq!(lengths, vec![2, 3]),
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
        Err(VmError { kind: VmErrorKind::Unsupported(_), .. }) => {}
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

#[test]
fn cphase_fixture_runs_end_to_end() {
    // cphase.qasm defines CX and a controlled-phase gate from primitives and
    // applies cphase to |00>; a phase gate leaves the computational basis
    // unchanged, so both qubits measure 0.
    let src = include_str!("../../fixtures/qasm/cphase.qasm");
    let m = run_measurements(src);
    assert_eq!(m, vec![(0, false), (1, false)]);
}

#[test]
fn register_measurement_into_a_slice() {
    // `measure q[0:3] -> ans[0:3]` measures a 4-qubit slice and stores the
    // 4-bit result into a slice of a wider classical register. The store
    // target is a slice, not a single element, so this exercises
    // slice-assignment of a multi-bit value.
    let outs = run_outputs(
        r#"
            include "stdgates.inc";
            qubit[4] q;
            x q[0];
            x q[2];
            bit[5] ans;
            measure q[0:3] -> ans[0:3];
        "#,
    );
    // q = 0b0101 over [0,1,2,3]; ans[4] stays 0. Bit registers print
    // MSB-first, so ans reads "00101".
    assert_eq!(outs, vec![("ans".to_string(), "\"00101\"".to_string())]);
}

#[test]
fn angle_register_bit_indexing_and_shift() {
    // An `angle[n]` is bit-indexable like `bit[n]`: assign into a bit via a
    // measure target, shift left, assign again. The accumulated numerator
    // 0b0011 = 3/16 turn = 3π/8.
    let outs = run_outputs(
        r#"
            include "stdgates.inc";
            qubit q;
            angle[4] c;
            x q;
            measure q -> c[0];          // numerator 0b0001
            c <<= 1;                    // 0b0010
            reset q;
            x q;
            measure q -> c[0];          // 0b0011 = 3/16 turn
        "#,
    );
    assert_eq!(outs, vec![("c".to_string(), "(3*π/8)".to_string())]);
}

#[test]
fn bare_call_of_subroutine_executes_as_call() {
    // `flip q;` is a `def` invoked with bare gate-call syntax (no parens).
    // It must run as a subroutine call, flipping both qubits.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            def flip(qubit[2] qs) { x qs[0]; x qs[1]; }
            qubit[2] q;
            flip q;
            bit[2] c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true), (1, true)]);
}

#[test]
fn bare_call_of_subroutine_with_classical_then_qubit_args() {
    // A bare-call passes classical args (in parens) before qubit operands;
    // they must bind to the subroutine's params in declaration order.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            def maybe_flip(int[32] k, qubit[2] qs) {
                if (k == 1) { x qs[0]; x qs[1]; }
            }
            qubit[2] q;
            maybe_flip(1) q;
            bit[2] c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true), (1, true)]);
}

#[test]
fn bit_cast_of_loop_var_sum_runs() {
    // `bit[32](row + col)` where row/col are `uint[32]` loop vars. The range
    // bounds are integer literals that must collapse to the loop var's width,
    // so `row + col` stays uint[32] and the equal-width bit-cast succeeds.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            for uint[32] row in [0:1] {
                for uint[32] col in [0:1] {
                    bit[32] s = bit[32](row + col);
                    if (s[0] == 1) { x q; }
                }
            }
            bit c = measure q;
        "#,
    );
    // row+col is odd in 2 of 4 iterations → x applied twice → back to |0>.
    assert_eq!(m, vec![(0, false)]);
}

#[test]
fn bit_cast_of_sized_var_plus_literal_runs() {
    // Covers the sized-var + unsized-literal binary-operand collapse:
    // `v + 1` must collapse the literal to v's width so the cast is equal-width.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit q;
            uint[8] v = 4;
            bit[8] b = bit[8](v + 1);   // 5 -> "00000101"
            if (b[0] == 1) { x q; }
            bit c = measure q;
        "#,
    );
    assert_eq!(m, vec![(0, true)]);
}

#[test]
fn ipe_fixture_runs_end_to_end() {
    let src = include_str!("../../fixtures/qasm/ipe.qasm");
    let m = run_measurements(src);
    // The estimation loop runs n = 10 iterations, one measurement each.
    assert_eq!(m.len(), 10, "expected 10 measurements, got {m:?}");
}

#[test]
fn runtime_indexed_set_alias_in_loop() {
    // `let bp = q[{2*i, 2*i+1}]` is a runtime-bound alias: each loop
    // iteration aliases a different pair, set/CX it, and measure both.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[4] q;
            bit[2] c;
            for uint i in [0:1] {
                let bp = q[{2*i, 2*i + 1}];
                x bp[0];
                cx bp[0], bp[1];
                c[0] = measure bp[0];
                c[1] = measure bp[1];
            }
        "#,
    );
    // Pairs (0,1) then (2,3), each driven to |11>.
    assert_eq!(m, vec![(0, true), (1, true), (2, true), (3, true)]);
}

#[test]
fn runtime_range_alias() {
    // `let a = q[i:j]` with runtime bounds aliases q[1], q[2], q[3].
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[6] q;
            uint i = 1;
            uint j = 3;
            let a = q[i:j];
            x a[0];
            x a[2];
            bit c0 = measure a[0];
            bit c1 = measure a[1];
            bit c2 = measure a[2];
        "#,
    );
    // a[0]=q[1] set, a[1]=q[2] ground, a[2]=q[3] set.
    assert_eq!(m, vec![(1, true), (2, false), (3, true)]);
}

#[test]
fn inline_runtime_indexed_set_operand() {
    // A runtime-valued index set used directly as a gate operand, with no
    // intervening `let` alias: `x q[{2*i, 2*i+1}]` broadcasts X over each pair.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[4] q;
            bit[4] c;
            for uint i in [0:1] {
                x q[{2*i, 2*i + 1}];
            }
            c[0] = measure q[0];
            c[1] = measure q[1];
            c[2] = measure q[2];
            c[3] = measure q[3];
        "#,
    );
    assert_eq!(m, vec![(0, true), (1, true), (2, true), (3, true)]);
}

#[test]
fn inline_runtime_range_operand() {
    // A runtime-valued range used directly as a gate operand, no `let` alias.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[6] q;
            uint i = 1;
            uint j = 3;
            x q[i:j];
            bit c0 = measure q[1];
            bit c1 = measure q[2];
            bit c2 = measure q[3];
        "#,
    );
    // q[1..=3] all flipped to |1>.
    assert_eq!(m, vec![(1, true), (2, true), (3, true)]);
}

#[test]
fn inline_set_index_of_runtime_alias() {
    // A set index applied directly to a runtime-bound alias (`a` is dynamic
    // because `q[i:j]` has runtime bounds) — exercises the alias-base path of
    // the transient-alias lowering.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[6] q;
            uint i = 1;
            uint j = 4;
            let a = q[i:j];
            x a[{0, 2}];
            bit c0 = measure q[1];
            bit c1 = measure q[2];
            bit c2 = measure q[3];
        "#,
    );
    // a = [q1,q2,q3,q4]; a[{0,2}] = q1,q3 flipped; q2 stays |0>.
    assert_eq!(m, vec![(1, true), (2, false), (3, true)]);
}

#[test]
fn teleportation_preserves_a_phase_state() {
    // Regression guard for the built-in `U` global-phase convention: a
    // controlled gate (`cx = ctrl @ x`) must carry no spurious relative
    // phase. Teleporting |+> and applying H must yield |0> deterministically
    // — a Z error on the teleported qubit (the old bug) would give |1>.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            qubit[3] q;
            bit out;
            reset q[0]; h q[0];                              // input |+>
            reset q[1]; reset q[2]; h q[1]; cx q[1], q[2];   // Bell pair
            cx q[0], q[1];
            h q[0];
            bit[2] pf;
            pf[0] = measure q[0];
            pf[1] = measure q[1];
            if (pf[0]==1) z q[2];
            if (pf[1]==1) x q[2];
            h q[2];
            out = measure q[2];
        "#,
    );
    // The final measurement (on q[2]) must be 0 regardless of the random
    // intermediate outcomes.
    assert_eq!(m.last().map(|&(q, b)| (q, b)), Some((2, false)));
}

#[test]
fn varteleport_fixture_runs_end_to_end() {
    // Exercises a `teleport` subroutine called over inline runtime-indexed
    // qubit args (`q[2*i - 1]`, `q[{2*i, 2*i + 1}]`) inside a loop.
    let src = include_str!("../../fixtures/qasm/varteleport.qasm");
    let m = run_measurements(src);
    // hop 0 (2 measurements) + 9 loop hops (2 each) + 1 final = 21. The final
    // bit is probabilistic (reflects the prepared input state), so only the
    // count is asserted.
    assert_eq!(m.len(), 21, "expected 21 measurements, got {m:?}");
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
        Err(VmError { kind: VmErrorKind::MissingInput(_), .. }) => {}
        other => panic!("expected MissingInput, got {other:?}"),
    }
}

#[test]
fn physical_qubit_program_sizes_its_register() {
    // A program that operates on a physical qubit `$0` and declares no
    // `qubit` registers must size the simulator to cover it, not fail
    // with an out-of-range error.
    let module = build("OPENQASM 3.0;\nU(0, 0, 0) $0;\n");
    assert_eq!(module.qubits.num_qubits, 1);
    let sim = StateVectorSim::new(module.qubits.num_qubits);
    let mut vm = Vm::new(&module, sim, NoExterns);
    vm.run().expect("physical-qubit program should run");
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
        Err(VmError { kind: VmErrorKind::UnknownInput(_), .. }) => {}
        other => panic!("expected UnknownInput, got {other:?}"),
    }
}

#[test]
fn physical_qubits_size_the_register() {
    // A program that uses only hardware qubits (`$n`) and declares no
    // `qubit` registers must still allocate enough simulator memory to
    // cover the highest physical index it touches.
    let m = run_measurements(
        r#"
            include "stdgates.inc";
            x $0;
            x $2;
            bit c0 = measure $0;
            bit c1 = measure $1;
            bit c2 = measure $2;
        "#,
    );
    assert_eq!(m, vec![(0, true), (1, false), (2, true)]);
}

