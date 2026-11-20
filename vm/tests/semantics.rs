//! End-to-end checks of scoping, indexing, loops and calls against the spec
//! (docs/scope.rst, docs/types.rst, docs/classical.rst, docs/subroutines.rst):
//! compile a program, run it, and read the named outputs back.

use oqi_compile::bytecode::BcModule;
use oqi_compile::{CompileError, CompileOptions, DefaultIncludeResolver};
use oqi_vm::{NoExterns, StateVectorSim, Vm};

fn compile(src: &str) -> Result<BcModule, CompileError> {
    oqi_compile::compile(
        src,
        DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
}

fn output(src: &str, name: &str) -> String {
    let module = compile(src).unwrap_or_else(|e| panic!("compile: {e}\n{src}"));
    let sim =
        StateVectorSim::<f64>::try_zeroed(module.qubits.num_qubits, oqi_vm::DEFAULT_SEED).unwrap();
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = pollster::block_on(vm.run()).unwrap_or_else(|e| panic!("run: {e}\n{src}"));
    result
        .outputs
        .iter()
        .find(|(sym, _)| module.symbol(*sym).name == name)
        .unwrap_or_else(|| panic!("no output `{name}`"))
        .1
        .to_string()
}

fn compile_error(src: &str) -> CompileError {
    match compile(src) {
        Ok(_) => panic!("expected a compile error for: {src}"),
        Err(e) => e,
    }
}

// ── Index sets and slicing (docs/types.rst) ─────────────────────────────

/// `a[0]` is the rightmost character of the literal.
const REG: &str = r#"bit[8] a = "10110010";"#;

#[test]
fn register_slice_is_a_register_of_the_selected_bits() {
    // Positions 1, 2, 3 hold 1, 0, 0.
    assert_eq!(output(&format!("{REG} bit[3] b = a[1:3];"), "b"), "\"001\"");
    // A descending range: positions 7, 5, 3, 1 hold 1, 1, 0, 1.
    assert_eq!(
        output(&format!("{REG} bit[4] b = a[7:-2:1];"), "b"),
        "\"1011\""
    );
    // Negative bounds count from the end: positions 5, 6, 7.
    assert_eq!(
        output(&format!("{REG} bit[3] b = a[-3:-1];"), "b"),
        "\"101\""
    );
}

#[test]
fn register_index_set_selects_the_listed_bits() {
    // Positions 0 and 7 hold 0 and 1.
    assert_eq!(
        output(&format!("{REG} bit[2] b = a[{{0, 7}}];"), "b"),
        "\"10\""
    );
    // Members may be run-time values.
    assert_eq!(
        output(
            &format!("{REG} int i = 1; int j = 4; bit[2] b = a[{{i, j}}];"),
            "b"
        ),
        "\"11\""
    );
}

#[test]
fn slice_width_is_checked_against_its_target() {
    let err = compile_error(&format!("{REG} bit[8] b = a[1:3];"));
    assert!(err.to_string().contains("bit[3]"), "{err}");
}

#[test]
fn integer_bit_slicing_follows_the_spec_examples() {
    let src = r#"
        int[32] myInt = 15;
        bit[16] evenBits = myInt[0:2:31];
        bit[16] upperBits = myInt[-16:-1];
        myInt[4:7] = "1010";
    "#;
    assert_eq!(output(src, "evenBits"), "\"0000000000000011\"");
    assert_eq!(output(src, "upperBits"), "\"0000000000000000\"");
    assert_eq!(output(src, "myInt"), "175");
}

#[test]
fn array_slices_read_and_write() {
    let src = r#"
        array[int[8], 2] first = {0, 1};
        array[int[8], 3] second = {2, 3, 4};
        array[int[8], 2] secondSlice = second[1:2];
        second[1:2] = first[0:1];
    "#;
    assert_eq!(output(src, "secondSlice"), "{3, 4}");
    assert_eq!(output(src, "second"), "{2, 0, 1}");
}

#[test]
fn arrays_cannot_be_indexed_by_a_set() {
    // "cannot be indexed by a comma-separated list of integers in braces".
    compile_error("array[int[8], 3] a = {1, 2, 3}; array[int[8], 2] b = a[{0, 1}];");
}

#[test]
fn measuring_a_register_slice_yields_that_many_bits() {
    let src = r#"
        include "stdgates.inc";
        qubit[5] q;
        x q[1];
        bit[3] c = measure q[0:2];
    "#;
    assert_eq!(output(src, "c"), "\"010\"");
}

// ── Constants and scope (docs/scope.rst) ────────────────────────────────

#[test]
fn a_shadowing_constant_does_not_outlive_its_scope() {
    let src = r#"
        const int N = 4;
        def f() { const int N = 2; }
        bit[N] b;
        b[3] = 1;
    "#;
    assert_eq!(output(src, "b"), "\"1000\"");

    let src = r#"
        const int N = 4;
        for int i in [0:1] { const int N = 2; }
        qubit[N] q;
        bit c = measure q[3];
    "#;
    assert_eq!(output(src, "c"), "0");
}

#[test]
fn qubit_index_out_of_range_points_at_the_index() {
    let err = compile_error("qubit[2] q; reset q[3];");
    assert_ne!(err.span, Default::default(), "{err}");
}

#[test]
fn definitions_are_only_valid_in_the_global_scope() {
    for src in [
        "if (true) { gate g a { } }",
        "{ def f() { } }",
        "while (false) { defcal g $0 { } }",
        "def f() { qubit q; }",
        "if (true) { qubit[2] q; }",
        "def f() { array[int[8], 2] a; }",
        "for int i in [0:1] { array[int[8], 2] a; }",
        "if (true) { input int x; }",
        "def f() { output int x; }",
    ] {
        let err = compile_error(src);
        assert!(err.to_string().contains("global scope"), "{src}: {err}");
    }
    // Aliases may be declared in local scopes.
    assert!(compile("qubit[2] q; if (true) { let a = q[0]; reset a; }").is_ok());
}

// ── Box (docs/scope.rst: a local block scope) ───────────────────────────

#[test]
fn a_box_shares_the_enclosing_scope() {
    let src = r#"
        int a = 5;
        int w = 0;
        box { w = a + 1; }
    "#;
    assert_eq!(output(src, "w"), "6");
}

#[test]
fn control_flow_leaves_a_box() {
    let src = r#"
        def f() -> int { box { return 7; } return 1; }
        int r = f();
        int n = 0;
        for int i in [0:9] { box { if (i == 3) { break; } } n += 1; }
    "#;
    assert_eq!(output(src, "r"), "7");
    assert_eq!(output(src, "n"), "3");
}

#[test]
fn durationof_sees_the_enclosing_scope() {
    let src = r#"
        qubit q;
        int a = 5;
        duration d = durationof({ if (a == 5) { delay[10ns] q; } else { delay[20ns] q; } });
    "#;
    assert_eq!(output(src, "d"), "10ns");
}

// ── For loops (docs/classical.rst) ──────────────────────────────────────

#[test]
fn ranges_count_down_for_any_integer_type() {
    for ty in ["int", "uint[8]", "int[8]", "uint"] {
        let src =
            format!("int s = 0; int n = 0; for {ty} i in [3:-1:0] {{ s += int(i); n += 1; }}");
        assert_eq!(output(&src, "s"), "6", "{ty}");
        assert_eq!(output(&src, "n"), "4", "{ty}");
    }
}

#[test]
fn ranges_count_up_to_the_limit_of_their_type() {
    for (ty, range, count) in [
        ("uint[8]", "[250:255]", "6"),
        ("int[8]", "[120:127]", "8"),
        ("uint[8]", "[0:2:6]", "4"),
        ("int[8]", "[-128:64:127]", "4"),
        ("uint[8]", "[5:4]", "0"),
    ] {
        let src = format!("int n = 0; for {ty} i in {range} {{ n += 1; }}");
        assert_eq!(output(&src, "n"), count, "{ty} {range}");
    }
}

#[test]
fn a_range_step_may_be_a_run_time_value() {
    let src = r#"
        int up = 2;
        int down = -2;
        int a = 0;
        int b = 0;
        for int i in [0:up:6] { a += i; }
        for int i in [6:down:1] { b += i; }
    "#;
    assert_eq!(output(src, "a"), "12");
    assert_eq!(output(src, "b"), "12");
}

#[test]
fn assigning_the_loop_variable_does_not_change_the_iteration() {
    // "Assigning a value to the loop variable within an iteration over the
    // body does not affect the next value that the loop variable will take."
    let src = "int n = 0; for int i in [0:3] { i = 10; n += 1; }";
    assert_eq!(output(src, "n"), "4");
}

#[test]
fn loops_over_sets_arrays_and_registers() {
    let src = r#"
        int[32] b = 0;
        for int[32] i in {1, 5, 10} { b += i; }

        array[float[64], 4] my_floats = {1.5, -3.5, 0.5, 9.5};
        float[64] total = 0.0;
        for float[64] f in my_floats { total += f; }

        bit[5] register = "10110";
        int ones = 0;
        int order = 0;
        for bit x in register { if (x) { ones += 1; } order = 2 * order + int(x); }

        int sliced = 0;
        for bit x in register[1:2] { if (x) { sliced += 1; } }
    "#;
    assert_eq!(output(src, "b"), "16");
    assert_eq!(output(src, "total"), "8");
    assert_eq!(output(src, "ones"), "3");
    // Sequential index order: bits 0, 1, 1, 0, 1.
    assert_eq!(output(src, "order"), "13");
    assert_eq!(output(src, "sliced"), "2");
}

#[test]
fn loop_iterables_are_type_checked() {
    compile_error("float[64] f = 1.0; for int i in f { }");
    compile_error("array[int[8], 2, 2] m; for int[8] i in m { }");
}

#[test]
fn the_adder_fixture_adds() {
    // 1 + 15 = 16. Its uncompute loop counts down over a `uint`.
    let src = include_str!("../../fixtures/qasm/adder.qasm");
    assert_eq!(output(src, "ans"), "\"10000\"");
}

#[test]
fn only_the_program_s_own_variables_are_outputs() {
    let src = r#"
        defcalgrammar "openpulse";
        qubit q;
        int kept = 1;
        cal { float in_cal = 1.5; }
        duration d = durationof({ int timed = 2; delay[10ns] q; });
    "#;
    let module = compile(src).unwrap();
    let mut names: Vec<_> = module
        .outputs
        .iter()
        .map(|(symbol, _)| module.symbol(*symbol).name.as_str())
        .collect();
    names.sort();
    assert_eq!(names, ["d", "kept"]);
}

// ── Subroutines (docs/subroutines.rst) ──────────────────────────────────

#[test]
fn a_subroutine_called_on_qubits_returns_its_value() {
    let src = r#"
        include "stdgates.inc";
        def xmeasure(qubit q) -> bit { x q; return measure q; }
        qubit[2] q;
        bit r = xmeasure q[1];
        bit again;
        again = xmeasure q[0];
    "#;
    assert_eq!(output(src, "r"), "1");
    assert_eq!(output(src, "again"), "1");
}

#[test]
fn a_gate_called_for_a_value_is_an_error() {
    compile_error(r#"include "stdgates.inc"; qubit q; bit r = h q;"#);
}

#[test]
fn mutable_array_parts_are_passed_by_reference() {
    let src = r#"
        def bump(mutable array[int[8], #dim = 1] a) { a[0] = a[0] + 10; }
        array[int[8], 2, 2] m = {{1, 2}, {3, 4}};
        bump(m[1]);
        int[8] row = m[1, 0];
        array[int[8], 4] xs = {1, 2, 3, 4};
        bump(xs[2:3]);
        int[8] part = xs[2];
        bump(xs);
        int[8] whole = xs[0];
        array[int[8], 2] copy = m[1];
        m[1, 1] = 9;
        int[8] kept = copy[1];
    "#;
    assert_eq!(output(src, "row"), "13");
    assert_eq!(output(src, "part"), "13");
    assert_eq!(output(src, "whole"), "11");
    // A sub-array read is a copy, even after the array was lent out.
    assert_eq!(output(src, "kept"), "4");
}

#[test]
fn a_mutable_reference_passes_through_nested_calls() {
    let src = r#"
        def inner(mutable array[int[8], #dim = 1] a) { a[0] = 42; }
        def outer(mutable array[int[8], #dim = 2] a) { inner(a[1]); }
        array[int[8], 2, 2] m = {{1, 2}, {3, 4}};
        outer(m);
        int[8] r = m[1, 0];
    "#;
    assert_eq!(output(src, "r"), "42");
}

// ── Operators on bits, registers and angles (docs/classical.rst) ────────

#[test]
fn bits_and_angles_support_the_bitwise_operators() {
    let src = r#"
        bit p = 1;
        bit q = 0;
        bit and_ = p & q;
        bit or_ = p | q;
        bit xor_ = p ^ q;
        angle[4] a = 9 * (pi / 8);
        angle[4] m = pi / 4;
        bit[4] both = bit[4](a & m);
        bit[4] either = bit[4](a | m);
        bit[4] rotated = bit[4](rotl(a, 1));
    "#;
    assert_eq!(output(src, "and_"), "0");
    assert_eq!(output(src, "or_"), "1");
    assert_eq!(output(src, "xor_"), "1");
    // a is "1001", m is "0010".
    assert_eq!(output(src, "both"), "\"0000\"");
    assert_eq!(output(src, "either"), "\"1011\"");
    assert_eq!(output(src, "rotated"), "\"0011\"");
}

#[test]
fn bits_and_registers_are_ordered() {
    let src = r#"
        bit[8] a = "10001111";
        bit[8] b = "01110000";
        bool gt = a > b;
        bool lte = a <= b;
        bit p = 1;
        bit q = 0;
        bool bits = q < p;
    "#;
    assert_eq!(output(src, "gt"), "true");
    assert_eq!(output(src, "lte"), "false");
    assert_eq!(output(src, "bits"), "true");
}
