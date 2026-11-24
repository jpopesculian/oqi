//! End-to-end checks of classical typing and evaluation against the spec
//! (docs/types.rst, docs/classical.rst): compile a program, run it, and read
//! the named outputs back.

use oqi_compile::CompileError;
use oqi_compile::bytecode::Program;
use oqi_vm::{NoExterns, StateVectorSim, Vm};

fn compile(src: &str) -> Result<Program, CompileError> {
    oqi_compile::compile(src)
}

/// Run and return named outputs as `(name, displayed value)`, sorted by name.
fn run_outputs(src: &str) -> Vec<(String, String)> {
    let module = compile(src).expect("compile");
    let sim = StateVectorSim::<f64>::new(module.qubits.num_qubits)
        .map(|s| s.with_seed(oqi_vm::DEFAULT_SEED))
        .unwrap();
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = pollster::block_on(vm.run()).expect("run");
    let mut out: Vec<(String, String)> = result
        .outputs()
        .map(|(name, v)| (name.to_string(), v.to_string()))
        .collect();
    out.sort();
    out
}

fn output(src: &str, name: &str) -> String {
    run_outputs(src)
        .into_iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no output `{name}`"))
        .1
}

fn compile_error(src: &str) -> String {
    match compile(src) {
        Ok(_) => panic!("expected a compile error for: {src}"),
        Err(e) => e.kind.to_string(),
    }
}

// ── Widths and dimensions ───────────────────────────────────────────────

#[test]
fn oversized_int_width_is_a_diagnostic_not_a_panic() {
    for src in [
        "int[200] x = 5;",
        "uint[129] x;",
        "angle[0] a;",
        "uint[0] x;",
    ] {
        let msg = compile_error(src);
        assert!(msg.contains("width"), "{src}: {msg}");
    }
    assert!(compile("int[128] x = 5;").is_ok());
}

#[test]
fn arrays_have_at_most_seven_dimensions() {
    // docs/types.rst: "a maximum of 7 total dimensions".
    assert!(compile("array[int[8], 1, 1, 1, 1, 1, 1, 1] a;").is_ok());
    let msg = compile_error("array[int[8], 1, 1, 1, 1, 1, 1, 1, 1] a;");
    assert!(msg.contains("dimensions"), "{msg}");
    let msg = compile_error("array[int[8], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] a;");
    assert!(msg.contains("dimensions"), "{msg}");
    let msg = compile_error("def f(readonly array[int[8], #dim = 8] a) {}");
    assert!(msg.contains("dimensions"), "{msg}");
}

// ── Shifts ──────────────────────────────────────────────────────────────

#[test]
fn shifting_by_the_width_or_more_shifts_every_bit_off() {
    // docs/classical.rst: "The shift operators shift bits off the end."
    let outs = run_outputs(
        r#"
            uint[8] one = 1;
            uint[8] n = 200;
            uint[8] left = one << n;
            uint[8] right = one >> n;
            int[8] neg = -1;
            int[8] sign = neg >> n;
            bit[4] reg = "1111";
            bit[4] cleared = reg << n;
            uint[64] wide = 1;
            uint[64] big = 64;
            uint[64] wide_left = wide << big;
        "#,
    );
    let get = |name: &str| {
        outs.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .unwrap()
    };
    assert_eq!(get("left"), "0");
    assert_eq!(get("right"), "0");
    assert_eq!(get("sign"), "-1", "arithmetic shift keeps the sign");
    assert_eq!(get("cleared"), "\"0000\"");
    assert_eq!(get("wide_left"), "0");
}

// ── Built-in functions ──────────────────────────────────────────────────

#[test]
fn float_functions_accept_integers_by_implicit_promotion() {
    // docs/types.rst: "The normal implicit casting rules apply to the
    // inputs of these functions" — `exp(i1)` with an `int[8]` is the spec's
    // own example, and the rule is the same for every float overload.
    let outs = run_outputs(
        r#"
            int[32] zero = 0;
            uint[8] one = 1;
            float[64] s = sin(zero);
            float[64] c = cos(zero);
            float[64] t = tan(zero);
            float[64] as = arcsin(zero);
            float[64] ac = arccos(one);
            float[64] at = arctan(zero);
            float[64] e = exp(zero);
            float[64] l = log(one);
            float[64] r = sqrt(one);
            float[64] fl = floor(one);
            float[64] ce = ceiling(one);
        "#,
    );
    let get = |name: &str| {
        outs.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .unwrap()
    };
    for (name, want) in [
        ("s", "0"),
        ("c", "1"),
        ("t", "0"),
        ("as", "0"),
        ("ac", "0"),
        ("at", "0"),
        ("e", "1"),
        ("l", "0"),
        ("r", "1"),
        ("fl", "1"),
        ("ce", "1"),
    ] {
        assert_eq!(get(name), want, "{name}");
    }
}

#[test]
fn popcount_accepts_uint() {
    // docs/classical.rst: `uint[6] b = 37; popcount(b); // Produces 3.`
    assert_eq!(output("uint[6] b = 37; uint[6] n = popcount(b);", "n"), "3");
}

// ── Operators the spec does not define are compile errors ───────────────

#[test]
fn ordering_complex_numbers_is_a_compile_error() {
    // docs/classical.rst: complex numbers support + - * / ** only.
    for op in ["<", ">", "<=", ">="] {
        let src = format!("complex[float[64]] a = 1.0 + 2.0im; bool b = a {op} a;");
        assert!(compile(&src).is_err(), "{src}");
    }
    assert!(compile("complex[float[64]] a = 1.0 + 2.0im; bool b = a == a;").is_ok());
}

#[test]
fn power_of_durations_and_angles_is_a_compile_error() {
    assert!(compile("duration a = 10ns; duration b = a ** a;").is_err());
    assert!(compile("angle[8] a = pi; angle[8] b = a ** a;").is_err());
    assert_eq!(output("int[32] a = 2; int[32] b = a ** 3;", "b"), "8");
}

// ── Casts (docs/types.rst, "Allowed casts") ─────────────────────────────

#[test]
fn casts_the_spec_allows() {
    let outs = run_outputs(
        r#"
            int[32] i = 3;
            int[32] z = 0;
            angle[8] a = pi;
            bit[4] reg = "0100";
            bool from_int = bool(i);
            bool from_zero = bool(z);
            bool from_angle = bool(a);
            bool from_reg = bool(reg);
            bit bit_from_int = bit(i);
            bool t = true;
            bit[4] reg_from_bool = bit[4](t);
        "#,
    );
    let get = |name: &str| {
        outs.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .unwrap()
    };
    assert_eq!(get("from_int"), "true");
    assert_eq!(get("from_zero"), "false");
    assert_eq!(get("from_angle"), "true");
    assert_eq!(get("from_reg"), "true");
    assert_eq!(get("bit_from_int"), "1");
    assert_eq!(get("reg_from_bool"), "\"0001\"");
}

#[test]
fn casts_the_spec_forbids() {
    for src in [
        // angle -> float / int / uint: No.
        "angle[8] a = pi; float[64] f = float[64](a);",
        "angle[8] a = pi; int[8] i = int[8](a);",
        "angle[8] a = pi; uint[8] u = uint[8](a);",
        // bit -> float and float -> bit: No (bool is the float-castable one).
        "bit b = 1; float[64] f = float[64](b);",
        "float[64] f = 1.5; bit b = bit(f);",
        // bool/int/uint -> angle: No.
        "int[8] i = 1; angle[8] a = angle[8](i);",
        // duration never casts.
        "duration d = 1ns; float[64] f = float[64](d);",
    ] {
        assert!(compile(src).is_err(), "{src}");
    }
    // The neighbouring allowed casts still work.
    assert_eq!(
        output("bool b = true; float[64] f = float[64](b);", "f"),
        "1"
    );
    assert_eq!(output("float[64] f = 1.5; bool b = bool(f);", "b"), "true");
}

#[test]
fn angles_still_mix_with_floats_in_arithmetic() {
    // Not a cast: the implicit promotion `angle + float -> float` that gate
    // arguments such as `theta + pi / 2` rely on keeps working.
    let f: f64 = output(
        "angle[8] a = pi; float[64] f = 0.0; f = sin(a) + (a + 1.0) * 0.0;",
        "f",
    )
    .parse()
    .unwrap();
    assert!(f.abs() < 1e-9, "{f}");
}

// ── Unsigned arithmetic wraps ───────────────────────────────────────────

#[test]
fn unsigned_arithmetic_wraps_modulo_its_width() {
    // docs/classical.rst (on angles): "This is the same behavior as unsigned
    // integers in many languages (including OpenQASM 3)"; the standard types
    // otherwise follow C99, where unsigned arithmetic is modulo 2^n.
    let outs = run_outputs(
        r#"
            uint[8] x = 255;
            uint[8] one = 1;
            uint[8] sum = x + one;
            uint[8] diff = one - x;
            uint[8] prod = x * x;
            uint[8] two = 2;
            uint[8] nine = 9;
            uint[8] power = two ** nine;
        "#,
    );
    let get = |name: &str| {
        outs.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .unwrap()
    };
    assert_eq!(get("sum"), "0");
    assert_eq!(get("diff"), "2");
    assert_eq!(get("prod"), "1");
    assert_eq!(get("power"), "0");
}

// ── Error messages ──────────────────────────────────────────────────────

#[test]
fn unsupported_operations_read_as_source_not_debug_dumps() {
    let msg = compile_error("bool b = true; float[64] f = sin(b);");
    assert!(msg.contains("sin(bool)"), "{msg}");
    assert!(!msg.contains("UnsupportedOp"), "{msg}");
}

// ── Range loops at the edge of the loop variable's type ─────────────────

/// `run_outputs` on a watchdog: a loop that fails to terminate fails the
/// test instead of hanging it.
fn run_outputs_or_time_out(src: &'static str) -> Vec<(String, String)> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(run_outputs(src)));
    rx.recv_timeout(std::time::Duration::from_secs(30))
        .unwrap_or_else(|_| panic!("did not terminate: {src}"))
}

#[test]
fn range_loops_terminate_at_the_limits_of_the_loop_variable() {
    // The exit test must not depend on `i + step` staying representable:
    // unsigned arithmetic wraps, so 255 + 1 is 0 again.
    for (src, want) in [
        (
            "uint[32] s = 0; for uint[8] i in [250:255] { s += 1; }",
            "6",
        ),
        (
            "uint[32] s = 0; for uint[8] i in [0:2:255] { s += 1; }",
            "128",
        ),
        ("uint[32] s = 0; for uint[4] i in [0:15] { s += i; }", "120"),
        ("uint[32] s = 0; for int[8] i in [120:127] { s += 1; }", "8"),
        (
            "uint[32] s = 0; for int[8] i in [-120:-1:-128] { s += 1; }",
            "9",
        ),
        ("uint[32] s = 0; for int[8] i in [0:3:10] { s += 1; }", "4"),
        ("uint[32] s = 0; for int[8] i in [5:4] { s += 1; }", "0"),
    ] {
        let outs = run_outputs_or_time_out(src);
        let s = &outs.iter().find(|(n, _)| n == "s").unwrap().1;
        assert_eq!(s, want, "{src}");
    }
}
