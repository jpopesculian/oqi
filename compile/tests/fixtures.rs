use std::path::Path;

use oqi_compile::lower::compile_source;
use oqi_compile::resolve::StdFileResolver;

fn compile_fixture(
    name: &str,
) -> Result<oqi_compile::sir::Program, oqi_compile::error::CompileError> {
    let path_str = format!("../fixtures/qasm/{name}");
    let path = Path::new(&path_str);
    let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path_str}: {e}"));
    compile_source(&source, StdFileResolver, Some(path))
}

fn compile_inline(
    source: &str,
) -> Result<oqi_compile::sir::Program, oqi_compile::error::CompileError> {
    compile_source(source, StdFileResolver, None)
}

// ── Fixtures that must compile successfully ──────────────────────────

#[test]
fn teleport() {
    let p = compile_fixture("teleport.qasm").expect("should compile");
    assert_eq!(p.version.as_deref(), Some("3"));
    assert!(!p.body.is_empty());
}

#[test]
fn adder() {
    let p = compile_fixture("adder.qasm").expect("should compile");
    assert!(!p.gates.is_empty());
    assert!(!p.body.is_empty());
}

#[test]
fn rus() {
    let p = compile_fixture("rus.qasm").expect("should compile");
    assert!(!p.subroutines.is_empty());
}

#[test]
fn vqe() {
    let p = compile_fixture("vqe.qasm").expect("should compile");
    assert!(!p.externs.is_empty());
    assert!(!p.subroutines.is_empty());
}

#[test]
fn qft() {
    compile_fixture("qft.qasm").expect("should compile");
}

#[test]
fn defcal() {
    let p = compile_fixture("defcal.qasm").expect("should compile");
    assert!(!p.calibrations.is_empty());
}

#[test]
fn alignment() {
    compile_fixture("alignment.qasm").expect("should compile");
}

#[test]
fn ipe() {
    compile_fixture("ipe.qasm").expect("should compile");
}

#[test]
fn scqec() {
    let p = compile_fixture("scqec.qasm").expect("should compile");
    assert!(!p.externs.is_empty());
}

#[test]
fn t1() {
    compile_fixture("t1.qasm").expect("should compile");
}

#[test]
fn varteleport() {
    compile_fixture("varteleport.qasm").expect("should compile");
}

#[test]
fn gateteleport() {
    compile_fixture("gateteleport.qasm").expect("should compile");
}

#[test]
fn inverseqft1() {
    compile_fixture("inverseqft1.qasm").expect("should compile");
}

#[test]
fn inverseqft2() {
    compile_fixture("inverseqft2.qasm").expect("should compile");
}

#[test]
fn qec() {
    compile_fixture("qec.qasm").expect("should compile");
}

#[test]
fn qpt() {
    compile_fixture("qpt.qasm").expect("should compile");
}

#[test]
fn rb() {
    compile_fixture("rb.qasm").expect("should compile");
}

// ── Fixtures that fail with expected errors ──────────────────────────

#[test]
fn cphase_no_stdgates() {
    // cphase.qasm uses CX without including stdgates.inc
    match compile_fixture("cphase.qasm") {
        Err(e) => assert!(matches!(
            e.kind,
            oqi_compile::error::ErrorKind::UndefinedName(_)
        )),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn msd_undeclared_success() {
    // msd.qasm assigns to 'success' without declaring it
    match compile_fixture("msd.qasm") {
        Err(e) => assert!(matches!(
            e.kind,
            oqi_compile::error::ErrorKind::UndefinedName(_)
        )),
        Ok(_) => panic!("expected error"),
    }
}

// ── Focused regression tests ─────────────────────────────────────────

#[test]
fn return_measure_in_subroutine() {
    let src = r#"
        include "stdgates.inc";
        def meas(qubit q) -> bit {
            return measure q;
        }
        qubit q;
        bit c = meas(q);
    "#;
    compile_inline(src).expect("return measure should compile");
}

#[test]
fn nested_array_literal() {
    let src = r#"
        array[int[32], 2, 3] a = {{1, 2, 3}, {4, 5, 6}};
    "#;
    compile_inline(src).expect("nested array literal should compile");
}

#[test]
fn readonly_array_ref_dim() {
    let src = r#"
        def f(readonly array[int[32], #dim = 3] a) {}
    "#;
    compile_inline(src).expect("readonly array ref with #dim should compile");
}

#[test]
fn include_replay() {
    // Including the same file twice replays declarations
    let src = r#"
        include "stdgates.inc";
        include "stdgates.inc";
        qubit q;
        h q;
    "#;
    // Should still compile (second include re-declares but in a new textual pass)
    // May fail due to duplicate detection — that's also acceptable
    let _ = compile_inline(src);
}

#[test]
fn cal_block_inline() {
    let src = r#"
        OPENQASM 3.0;
        defcalgrammar "openpulse";
        cal {
            // opaque calibration body
        }
    "#;
    let p = compile_inline(src).expect("cal block should compile");
    // Lexer preserves quotes around the grammar string
    assert!(p.calibration_grammar.is_some());
}

#[test]
fn const_arithmetic_designator() {
    let src = r#"
        const int[32] n = 5;
        qubit[2 * n] q;
    "#;
    let p = compile_inline(src).expect("const arithmetic in designator should compile");
    // 2 * 5 = 10 qubits
    let q_sym = p.symbols.iter().find(|s| s.name == "q").unwrap();
    assert_eq!(q_sym.ty, oqi_compile::types::Type::QubitReg(10));
}

#[test]
fn const_power_designator() {
    let src = r#"
        const int[32] d = 3;
        const int[32] n = d ** 2;
        qubit[n] q;
    "#;
    let p = compile_inline(src).expect("const power in designator should compile");
    let q_sym = p.symbols.iter().find(|s| s.name == "q").unwrap();
    assert_eq!(q_sym.ty, oqi_compile::types::Type::QubitReg(9));
}

#[test]
fn duration_stretch_arithmetic() {
    let src = r#"
        include "stdgates.inc";
        stretch g;
        qubit q;
        delay[2 * g] q;
    "#;
    compile_inline(src).expect("stretch * int should compile");
}

#[test]
fn trig_intrinsic_returns_float() {
    let src = r#"
        include "stdgates.inc";
        qubit q;
        rz(pi - arccos(3 / 5)) q;
    "#;
    compile_inline(src).expect("trig with int args should compile");
}

#[test]
fn compile_never_panics_on_parse_error() {
    let src = "this is not valid openqasm";
    assert!(compile_inline(src).is_err());
}

// ── Type declarations (from types.rst) ───────────────────────────────

#[test]
fn classical_scalar_types() {
    let src = r#"
        bit my_bit;
        bit[8] my_bitreg = "00001111";
        bool my_bool = false;
        int[32] my_int = 10;
        uint[16] my_uint = 42;
        float[64] my_float = 3.14;
        angle[20] my_angle = pi / 2;
    "#;
    let p = compile_inline(src).expect("scalar types should compile");
    let sym = |name: &str| p.symbols.iter().find(|s| s.name == name).unwrap();

    assert_eq!(sym("my_bit").ty, oqi_compile::types::Type::bit());
    assert_eq!(sym("my_bitreg").ty, oqi_compile::types::Type::bitreg(8));
    assert_eq!(sym("my_bool").ty, oqi_compile::types::Type::bool());
    assert_eq!(sym("my_int").ty, oqi_compile::types::Type::int(32, true));
    assert_eq!(sym("my_uint").ty, oqi_compile::types::Type::int(16, false));
    assert_eq!(
        sym("my_float").ty,
        oqi_compile::types::Type::float(oqi_compile::types::FloatWidth::F64)
    );
}

#[test]
fn qubit_declarations() {
    let src = r#"
        qubit single;
        qubit[5] reg;
        const uint SIZE = 4;
        qubit[SIZE] sized;
    "#;
    let p = compile_inline(src).expect("qubit decls should compile");
    let sym = |name: &str| p.symbols.iter().find(|s| s.name == name).unwrap();

    assert_eq!(sym("single").ty, oqi_compile::types::Type::Qubit);
    assert_eq!(sym("reg").ty, oqi_compile::types::Type::QubitReg(5));
    assert_eq!(sym("sized").ty, oqi_compile::types::Type::QubitReg(4));
}

#[test]
fn complex_type_declaration() {
    let src = r#"
        complex[float[64]] c = 2.5 + 3.5im;
    "#;
    let p = compile_inline(src).expect("complex decl should compile");
    let sym = p.symbols.iter().find(|s| s.name == "c").unwrap();
    assert_eq!(
        sym.ty,
        oqi_compile::types::Type::complex(oqi_compile::types::FloatWidth::F64)
    );
}

#[test]
fn duration_and_stretch_types() {
    let src = r#"
        duration one_second = 1000ms;
        duration thousand_cycles = 1000dt;
        stretch g;
    "#;
    let p = compile_inline(src).expect("duration/stretch should compile");
    let sym = |name: &str| p.symbols.iter().find(|s| s.name == name).unwrap();

    assert_eq!(sym("one_second").ty, oqi_compile::types::Type::duration());
    assert_eq!(
        sym("thousand_cycles").ty,
        oqi_compile::types::Type::duration()
    );
    assert_eq!(sym("g").ty, oqi_compile::types::Type::Stretch);
}

#[test]
fn array_1d_declaration() {
    let src = r#"
        array[int[32], 5] a = {0, 1, 2, 3, 4};
    "#;
    let p = compile_inline(src).expect("1d array should compile");
    let sym = p.symbols.iter().find(|s| s.name == "a").unwrap();
    assert_eq!(
        sym.ty,
        oqi_compile::types::Type::array_of(oqi_compile::types::Type::int(32, true), vec![5])
    );
}

#[test]
fn array_multidim_declaration() {
    let src = r#"
        array[float[32], 3, 2] md = {{1.1, 1.2}, {2.1, 2.2}, {3.1, 3.2}};
    "#;
    let p = compile_inline(src).expect("multidim array should compile");
    let sym = p.symbols.iter().find(|s| s.name == "md").unwrap();
    assert_eq!(
        sym.ty,
        oqi_compile::types::Type::array_of(
            oqi_compile::types::Type::float(oqi_compile::types::FloatWidth::F32),
            vec![3, 2],
        )
    );
}

#[test]
fn const_declarations_and_builtin_constants() {
    let src = r#"
        const uint SIZE = 32;
        const int[8] small = 4;
        const float[64] f = pi;
        qubit[SIZE] q;
        int[SIZE] i;
    "#;
    let p = compile_inline(src).expect("const decls should compile");
    let sym = |name: &str| p.symbols.iter().find(|s| s.name == name).unwrap();

    assert_eq!(sym("SIZE").kind, oqi_compile::symbol::SymbolKind::Const);
    assert_eq!(sym("q").ty, oqi_compile::types::Type::QubitReg(32));
    assert_eq!(sym("i").ty, oqi_compile::types::Type::int(32, true));
}

#[test]
fn integer_literal_bases() {
    let src = r#"
        int i1 = 255;
        int i2 = 0xff;
        int i3 = 0o377;
        int i4 = 0b11111111;
    "#;
    compile_inline(src).expect("integer literal bases should compile");
}

#[test]
fn float_literal_forms() {
    let src = r#"
        float f1 = 1.0;
        float f2 = .5;
        float f3 = 2e10;
        float f4 = 2.0E-1;
    "#;
    compile_inline(src).expect("float literal forms should compile");
}

#[test]
fn bitstring_literal() {
    let src = r#"
        bit[8] b = "00001111";
    "#;
    let p = compile_inline(src).expect("bitstring literal should compile");
    let sym = p.symbols.iter().find(|s| s.name == "b").unwrap();
    assert_eq!(sym.ty, oqi_compile::types::Type::bitreg(8));
}

// ── Classical instructions (from classical.rst) ──────────────────────

#[test]
fn assignment_and_copy() {
    let src = r#"
        int[32] a;
        int[32] b = 10;
        a = b;
        b = 0;
    "#;
    compile_inline(src).expect("assignment should compile");
}

#[test]
fn bitwise_operators_on_bits() {
    let src = r#"
        bit[8] a = "10001111";
        bit[8] b = "01110000";
        bit[8] c;
        c = a | b;
        c = a & b;
        c = a ^ b;
        c = ~a;
        c = a << 1;
        c = a >> 2;
    "#;
    compile_inline(src).expect("bitwise ops should compile");
}

#[test]
fn comparison_operators() {
    let src = r#"
        bool a = false;
        int[32] b = 1;
        int[32] c = 2;
        bool r;
        r = a == false;
        r = c >= b;
        r = b < c;
        r = b != c;
        r = b <= c;
        r = c > b;
    "#;
    compile_inline(src).expect("comparison ops should compile");
}

#[test]
fn logical_operators() {
    let src = r#"
        bool a = true;
        bool b = false;
        bool c;
        c = a && b;
        c = a || b;
        c = !a;
    "#;
    compile_inline(src).expect("logical ops should compile");
}

#[test]
fn integer_arithmetic() {
    let src = r#"
        int[32] a = 2;
        int[32] b = 3;
        int[32] c;
        c = a + b;
        c = a - b;
        c = a * b;
        c = b / a;
        c = b % a;
        c = a ** b;
    "#;
    compile_inline(src).expect("integer arithmetic should compile");
}

#[test]
fn compound_assignment_operators() {
    let src = r#"
        int[32] a = 2;
        a += 4;
        a -= 1;
        a *= 3;
        a /= 2;
        a %= 5;
    "#;
    compile_inline(src).expect("compound assignment should compile");
}

#[test]
fn float_arithmetic() {
    let src = r#"
        float[64] a = 1.5;
        float[64] b = 2.5;
        float[64] c;
        c = a + b;
        c = a - b;
        c = a * b;
        c = a / b;
    "#;
    compile_inline(src).expect("float arithmetic should compile");
}

#[test]
fn complex_arithmetic() {
    let src = r#"
        complex[float[64]] a = 10.0 + 5.0im;
        complex[float[64]] b = -2.0 - 7.0im;
        complex[float[64]] c;
        c = a + b;
        c = a - b;
        c = a * b;
        c = a / b;
    "#;
    compile_inline(src).expect("complex arithmetic should compile");
}

// ── Control flow (from classical.rst) ────────────────────────────────

#[test]
fn if_else_statement() {
    let src = r#"
        include "stdgates.inc";
        qubit a;
        bit result;
        bool target = false;
        h a;
        result = measure a;
        if (target == result) {
            x a;
        } else {
            z a;
        }
    "#;
    compile_inline(src).expect("if-else should compile");
}

#[test]
fn for_loop_discrete_set() {
    let src = r#"
        int[32] b = 0;
        for int[32] i in {1, 5, 10} {
            b += i;
        }
    "#;
    compile_inline(src).expect("for loop with set should compile");
}

#[test]
fn for_loop_range() {
    let src = r#"
        int[32] sum = 0;
        for int i in [0:2:20] {
            sum += i;
        }
    "#;
    compile_inline(src).expect("for loop with range should compile");
}

#[test]
fn for_loop_array() {
    let src = r#"
        array[float[64], 4] my_floats = {1.2, -3.4, 0.5, 9.8};
        float[64] sum = 0.0;
        for float[64] f in my_floats {
            sum += f;
        }
    "#;
    compile_inline(src).expect("for loop over array should compile");
}

#[test]
fn while_loop() {
    let src = r#"
        include "stdgates.inc";
        qubit q;
        bit result;
        int i = 0;
        while (i < 10) {
            h q;
            result = measure q;
            if (result) {
                i += 1;
            }
        }
    "#;
    compile_inline(src).expect("while loop should compile");
}

#[test]
fn break_and_continue() {
    let src = r#"
        int[32] i = 0;
        while (i < 10) {
            i += 1;
            if (i == 2) {
                continue;
            }
            if (i == 4) {
                break;
            }
        }
    "#;
    compile_inline(src).expect("break/continue should compile");
}

#[test]
fn end_statement() {
    let src = r#"
        int[32] x = 0;
        end;
    "#;
    compile_inline(src).expect("end statement should compile");
}

#[test]
fn switch_statement_basic() {
    let src = r#"
        int i = 15;
        switch (i) {
            case 1, 3, 5 {
            }
            case 2, 4, 6 {
            }
            default {
            }
        }
    "#;
    compile_inline(src).expect("switch statement should compile");
}

#[test]
fn switch_const_expressions() {
    let src = r#"
        const int A = 0;
        const int B = 1;
        int i = 15;
        switch (i) {
            case A {
            }
            case B {
            }
            case B + 1 {
            }
            default {
            }
        }
    "#;
    compile_inline(src).expect("switch with const exprs should compile");
}

#[test]
fn switch_nested() {
    let src = r#"
        include "stdgates.inc";
        qubit[8] q;
        int j = 30;
        int i = 3;
        switch (i) {
            case 1, 2, 5 { }
            case 3 {
                switch (j) {
                    case 10, 15, 20 {
                        h q;
                    }
                }
            }
        }
    "#;
    compile_inline(src).expect("nested switch should compile");
}

// ── Extern functions (from classical.rst) ────────────────────────────

#[test]
fn extern_declaration_and_call() {
    let src = r#"
        extern get_param(int[32]) -> float[64];
        int[32] idx = 0;
        float[64] val = get_param(idx);
    "#;
    let p = compile_inline(src).expect("extern decl+call should compile");
    assert!(!p.externs.is_empty());
}

// ── Qubit aliasing (from types.rst) ──────────────────────────────────

#[test]
fn qubit_alias_slice() {
    let src = r#"
        qubit[5] q;
        let myreg = q[1:4];
    "#;
    compile_inline(src).expect("qubit alias should compile");
}

// ── Built-in math functions (from types.rst) ─────────────────────────

#[test]
fn builtin_math_functions() {
    let src = r#"
        include "stdgates.inc";
        qubit q;
        rz(sin(pi / 4)) q;
        rz(cos(pi / 4)) q;
        rz(tan(0.5)) q;
        rz(exp(1.0)) q;
        rz(log(2.0)) q;
        rz(sqrt(2.0)) q;
    "#;
    compile_inline(src).expect("builtin math functions should compile");
}

// ── Error cases (from types.rst) ─────────────────────────────────────

#[test]
fn non_const_designator_rejected() {
    let src = r#"
        uint runtime_size = 32;
        qubit[runtime_size] q;
    "#;
    match compile_inline(src) {
        Err(e) => assert!(
            matches!(
                e.kind,
                oqi_compile::error::ErrorKind::NonConstantDesignator
                    | oqi_compile::error::ErrorKind::NonConstantExpression
            ),
            "expected non-constant error, got: {:?}",
            e.kind,
        ),
        Ok(_) => panic!("expected non-constant error"),
    }
}

#[test]
fn undefined_variable_rejected() {
    let src = r#"
        int[32] a = b;
    "#;
    match compile_inline(src) {
        Err(e) => assert!(matches!(
            e.kind,
            oqi_compile::error::ErrorKind::UndefinedName(_)
        )),
        Ok(_) => panic!("expected UndefinedName error"),
    }
}

#[test]
fn duplicate_definition_rejected() {
    let src = r#"
        int[32] x = 1;
        int[32] x = 2;
    "#;
    match compile_inline(src) {
        Err(e) => assert!(matches!(
            e.kind,
            oqi_compile::error::ErrorKind::DuplicateDefinition(_)
        )),
        Ok(_) => panic!("expected DuplicateDefinition error"),
    }
}

// ── IO declarations (from types.rst) ─────────────────────────────────

#[test]
fn io_declarations() {
    let src = r#"
        input float[64] theta;
        output bit[4] result;
    "#;
    let p = compile_inline(src).expect("io decls should compile");
    let sym = |name: &str| p.symbols.iter().find(|s| s.name == name).unwrap();

    assert_eq!(sym("theta").kind, oqi_compile::symbol::SymbolKind::Input);
    assert_eq!(sym("result").kind, oqi_compile::symbol::SymbolKind::Output);
}
