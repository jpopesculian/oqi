//! End-to-end checks of scoping, indexing, loops and calls against the spec
//! (docs/scope.rst, docs/types.rst, docs/classical.rst, docs/subroutines.rst):
//! compile a program, run it, and read the named outputs back.

use oqi_compile::CompileError;
use oqi_compile::bytecode::Program;
use oqi_vm::{NoExterns, StateVectorSim, Vm};

fn compile(src: &str) -> Result<Program, CompileError> {
    oqi_compile::compile(src)
}

fn output(src: &str, name: &str) -> String {
    let module = compile(src).unwrap_or_else(|e| panic!("compile: {e}\n{src}"));
    let sim = StateVectorSim::<f64>::new(module.qubits.num_qubits)
        .map(|s| s.with_seed(oqi_vm::DEFAULT_SEED))
        .unwrap();
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = pollster::block_on(vm.run()).unwrap_or_else(|e| panic!("run: {e}\n{src}"));
    result
        .output(name)
        .unwrap_or_else(|| panic!("no output `{name}`"))
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

const TWO_D: &str = "array[int[8], 3, 2] twoD = {{1, 2}, {3, 4}, {5, 6}};";

#[test]
fn a_subscript_indexes_one_dimension_per_position() {
    // docs/types.rst: `twoD[1:2, 0] = anotherTwoD[0:1, 1]; // allowed`.
    let src = format!(
        "{TWO_D}
        array[int[8], 3, 2] anotherTwoD = {{{{10, 20}}, {{30, 40}}, {{50, 60}}}};
        array[int[8], 2] column = twoD[1:2, 0];
        array[int[8], 2, 1] cells = twoD[0:1, 1:1];
        array[int[8], 2, 2] rows = twoD[0:2:2];
        int[8] corner = twoD[-1, -1];
        twoD[1:2, 0] = anotherTwoD[0:1, 1];"
    );
    assert_eq!(output(&src, "column"), "{3, 5}");
    assert_eq!(output(&src, "cells"), "{{2}, {4}}");
    assert_eq!(output(&src, "rows"), "{{1, 2}, {5, 6}}");
    assert_eq!(output(&src, "corner"), "6");
    assert_eq!(output(&src, "twoD"), "{{1, 2}, {20, 4}, {40, 6}}");
}

#[test]
fn array_assignment_follows_the_spec_examples() {
    let decls = format!(
        "int[8] scalar = 7;
        array[int[8], 2] oneD = {{8, 9}};
        {TWO_D}
        array[int[8], 3, 2] anotherTwoD = {{{{10, 20}}, {{30, 40}}, {{50, 60}}}};
        array[int[8], 4, 3, 2] threeD;
        array[int[8], 2, 3, 4] anotherThreeD;"
    );
    let src = format!(
        "{decls}
        threeD[0, 0, 0] = scalar;
        threeD[1, 0] = oneD;
        threeD[2] = twoD;
        twoD[1:2] = anotherTwoD[0:1];
        array[int[8], 2] a = threeD[0, 0];
        array[int[8], 2] b = threeD[1, 0];
        array[int[8], 3, 2] c = threeD[2];
        array[int[8], 2] d = threeD[2, 1:2, 1];"
    );
    assert_eq!(output(&src, "a"), "{7, 0}");
    assert_eq!(output(&src, "b"), "{8, 9}");
    assert_eq!(output(&src, "c"), "{{1, 2}, {3, 4}, {5, 6}}");
    assert_eq!(output(&src, "d"), "{4, 6}");
    assert_eq!(output(&src, "twoD"), "{{1, 2}, {10, 20}, {30, 40}}");

    // "error - shape mismatch"
    compile_error(&format!("{decls} threeD[0] = oneD;"));
    compile_error(&format!("{decls} threeD[0, 0] = scalar;"));
    compile_error(&format!("{decls} threeD = anotherThreeD;"));
    compile_error(&format!("{decls} twoD[1:2, 0] = anotherTwoD[0:1];"));
}

#[test]
fn a_second_subscript_indexes_what_the_first_selected() {
    // "nearly all instances of multiple subscripts `[][]` will be bit-level
    // accesses of array elements" — the rest index the selection.
    let src = format!(
        "array[int[32], 5] intArr = {{0, 1, 2, 3, 4}};
        intArr[0][0] = 1;
        bit[5] b = intArr[4][0:4];
        {TWO_D}
        array[int[8], 2] oneD = {{8, 9}};
        array[int[8], 2] row = twoD[1:2][0];
        int[8] cell = twoD[1:2, 1][0];
        twoD[1:2][1] = oneD;
        twoD[0:1][1][0] = 7;
        bit[8] reg = \"00000000\";
        reg[2:5][1] = 1;"
    );
    assert_eq!(output(&src, "intArr"), "{1, 1, 2, 3, 4}");
    assert_eq!(output(&src, "b"), "\"00100\"");
    assert_eq!(output(&src, "row"), "{3, 4}");
    assert_eq!(output(&src, "cell"), "4");
    assert_eq!(output(&src, "twoD"), "{{1, 2}, {7, 4}, {8, 9}}");
    assert_eq!(output(&src, "reg"), "\"00001000\"");
}

#[test]
fn per_dimension_subscripts_work_through_array_references() {
    let src = format!(
        "def shift(mutable array[int[8], #dim = 2] a) {{ a[0:1, 1] = a[1:2, 0]; }}
        def mark(mutable array[int[8], #dim = 2] a) {{ a[0, 0] = a[0, 0] + 100; }}
        def pick(readonly array[int[8], 3, 2] a) -> int[8] {{ return a[1:2, 1][1]; }}
        {TWO_D}
        int[8] picked = pick(twoD);
        shift(twoD);
        array[int[8], 2, 3, 2] cube = {{{{{{1, 2}}, {{3, 4}}, {{5, 6}}}}, {{{{1, 2}}, {{3, 4}}, {{5, 6}}}}}};
        shift(cube[1]);
        array[int[8], 3, 2] untouched = cube[0];
        array[int[8], 3, 2] shifted = cube[1];
        array[int[8], 4, 2] tall = {{{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}}};
        shift(tall[1:3]);
        mark(tall[0:2][1:2]);"
    );
    assert_eq!(output(&src, "picked"), "6");
    assert_eq!(output(&src, "twoD"), "{{1, 3}, {3, 5}, {5, 6}}");
    assert_eq!(output(&src, "untouched"), "{{1, 2}, {3, 4}, {5, 6}}");
    assert_eq!(output(&src, "shifted"), "{{1, 3}, {3, 5}, {5, 6}}");
    // Rows 1 to 3 shift; then row 1 of rows 0 to 2 is row 0 of its last two.
    assert_eq!(output(&src, "tall"), "{{1, 2}, {103, 5}, {5, 7}, {7, 8}}");
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

// ── Aliases of bit registers (docs/types.rst, docs/classical.rst) ────────

#[test]
fn a_classical_alias_refers_to_the_bits_it_names() {
    // Positions 1, 2, 3 of `a` hold 1, 0, 0; positions 7 and 6 hold 1, 0.
    let src = format!(
        "{REG}
        let mid = a[1:3];
        let low = a[0];
        let picked = a[{{7, 6}}];
        bit[3] before = mid;
        bit first = low;
        bit[2] chosen = picked;
        bit one = mid[0];
        bit[2] two = mid[1:2];
        a[2] = 1;
        bit[3] after = mid;"
    );
    assert_eq!(output(&src, "before"), "\"001\"");
    assert_eq!(output(&src, "first"), "0");
    assert_eq!(output(&src, "chosen"), "\"01\"");
    assert_eq!(output(&src, "one"), "1");
    assert_eq!(output(&src, "two"), "\"00\"");
    // An alias is a reference: it sees the register as it now is.
    assert_eq!(output(&src, "after"), "\"011\"");
}

#[test]
fn writing_through_a_classical_alias_reaches_the_register() {
    let src = r#"
        bit[8] r = "00000000";
        let mid = r[2:5];
        mid = "1001";
        bit[8] whole = r;
        mid[1] = 1;
        bit[8] item = r;
        mid[2:3] = "01";
        bit[8] slice = r;
        let top = r[7];
        top = 1;
    "#;
    assert_eq!(output(src, "whole"), "\"00100100\"");
    assert_eq!(output(src, "item"), "\"00101100\"");
    assert_eq!(output(src, "slice"), "\"00011100\"");
    assert_eq!(output(src, "r"), "\"10011100\"");
}

#[test]
fn classical_aliases_concatenate_registers() {
    // "The concatenated register is a reference to the bits … of the
    // original registers."
    let src = r#"
        bit[2] hi = "10";
        bit[3] lo = "011";
        let both = lo ++ hi;
        bit[5] joined = both;
        both[4] = 0;
        both[0:1] = "00";
        let inner = both[2:3];
        inner = "11";
        let spread = lo[0] ++ hi[{1, 0}] ++ lo[2];
        bit[4] spread_now = spread;
    "#;
    assert_eq!(output(src, "joined"), "\"10011\"");
    assert_eq!(output(src, "lo"), "\"100\"");
    assert_eq!(output(src, "hi"), "\"01\"");
    // Parts of one register may be concatenated, as long as they are
    // different parts: lo[0], hi[1], hi[0], lo[2] hold 0, 0, 1, 1.
    assert_eq!(output(src, "spread_now"), "\"1100\"");
}

#[test]
fn classical_aliases_loop_and_take_measurements() {
    let src = r#"
        include "stdgates.inc";
        qubit[3] q;
        x q[0];
        x q[2];
        bit[5] c;
        let window = c[1:3];
        window = measure q;
        int ones = 0;
        for bit b in window { ones += int(b); }
        let last = c[4];
        last = measure q[0];
        window[1] = measure q[2];
    "#;
    assert_eq!(output(src, "ones"), "2");
    assert_eq!(output(src, "c"), "\"11110\"");
}

#[test]
fn a_classical_alias_is_bound_where_it_is_declared() {
    let src = r#"
        bit[4] r = "0000";
        int i = 1;
        let chosen = r[i];
        let pair = r[{i, 2}];
        i = 3;
        chosen = 1;
        bit[4] bound = r;
        pair = "10";
        bit[4] set = r;
        let a = r[0];
        if (true) {
            let a = r[3];
            a = 1;
        }
        a = 1;
    "#;
    // `chosen` is r[1] whatever becomes of `i`; `pair` is r[1], r[2].
    assert_eq!(output(src, "bound"), "\"0010\"");
    assert_eq!(output(src, "set"), "\"0100\"");
    // The alias in the block shadows the outer one, and ends with it.
    assert_eq!(output(src, "r"), "\"1101\"");
}

#[test]
fn classical_aliases_work_in_loops_and_subroutines() {
    let src = r#"
        def middle(bit[4] r) -> bit[2] { let m = r[1:2]; m[0] = 1; return m; }
        bit[4] r = "0100";
        bit[2] got = middle(r);
        for int i in [0:2] { let b = r[i]; b = 1; }
        bit[4] looped = r;
        let all = r;
        all ^= "1111";
    "#;
    assert_eq!(output(src, "got"), "\"11\"");
    // Each iteration binds `b` anew.
    assert_eq!(output(src, "looped"), "\"0111\"");
    assert_eq!(output(src, "r"), "\"1000\"");
    // An alias belongs to the body that declares it.
    let err = compile_error("bit[4] r; let a = r[0:1]; def f() -> bit { return a[0]; }");
    assert!(err.to_string().contains("not visible"), "{err}");
}

#[test]
fn only_registers_can_be_aliased() {
    for (src, what) in [
        ("array[int[8], 2] xs = {1, 2}; let a = xs;", "alias"),
        ("int[8] x = 3; let a = x;", "alias"),
        ("int[8] x = 3; let a = x[0:3];", "alias"),
        ("bit[2] c; let a = c & c;", "alias"),
        ("let a = $0;", "physical qubits"),
        ("qubit[2] q; bit[2] c; let a = q ++ c;", "classical"),
        ("bit[4] r; let a = r ++ r[2:3];", "part of itself"),
        (
            "bit[4] r; let a = r[0:1]; let b = r[3] ++ a[{1, 0}] ++ r[1:2];",
            "part of itself",
        ),
    ] {
        let err = compile_error(src);
        assert!(err.to_string().contains(what), "{src}: {err}");
        let span = err.span.expect("a location");
        assert!(span.end > span.start, "{src}: no location");
    }
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
    assert!(err.span.is_some_and(|s| s != Default::default()), "{err}");
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

/// "Globally scoped variables without the `const` modifier are not visible
/// inside the definition", and neither are the program's qubits.
#[test]
fn definitions_see_only_the_constants_of_the_global_scope() {
    for (src, name) in [
        ("int g = 5; def f() -> int { return g; }", "g"),
        ("input float t; def f() -> float { return t; }", "t"),
        ("qubit[2] q; def f() { reset q[0]; }", "q"),
        ("qubit[2] q; let a = q[0:1]; def f() { reset a; }", "a"),
        ("float w = 0.5; gate g a { U(w, 0, 0) a; }", "w"),
        ("qubit q; gate g a { U(0, 0, 0) q; }", "q"),
        ("int n = 2; def f() { bit[n] b; }", "n"),
        (
            r#"defcalgrammar "openpulse"; int n = 2; defcal g $0 { int m = n; }"#,
            "n",
        ),
    ] {
        let err = compile_error(src);
        let message = format!("`{name}` is not visible");
        assert!(err.to_string().contains(&message), "{src}: {err}");
        assert!(err.span.is_some_and(|s| s != Default::default()), "{src}");
    }

    // Constants, gates and subroutines of the global scope are visible, a
    // parameter or local may shadow a variable, and the variable is itself
    // again after the definition.
    let src = r#"
        const int c = 3;
        int a = 10;
        def twice(int x) -> int { return 2 * x; }
        def f(int a) -> int { int b = c; return twice(a) + b; }
        int r = f(4) + a;
    "#;
    assert_eq!(output(src, "r"), "21");
    let src = r#"
        const float half = pi / 2;
        gate g a { U(2 * half, 0, pi) a; }
        qubit q;
        g q;
        bit r = measure q;
    "#;
    assert_eq!(output(src, "r"), "1");
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
