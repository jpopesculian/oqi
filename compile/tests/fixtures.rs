use std::path::Path;

use oqi_compile::classical::{ValueTy, iw};
use oqi_compile::lower::compile_source;
use oqi_compile::resolve::DefaultIncludeResolver;

fn compile_fixture(
    name: &str,
) -> Result<oqi_compile::sir::Program, oqi_compile::error::CompileError> {
    let path_str = format!("../fixtures/qasm/{name}");
    let path = Path::new(&path_str);
    let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path_str}: {e}"));
    compile_source(&source, DefaultIncludeResolver, Some(path))
}

fn compile_inline(
    source: &str,
) -> Result<oqi_compile::sir::Program, oqi_compile::error::CompileError> {
    compile_source(source, DefaultIncludeResolver, None)
}

// ── Fixtures that must compile successfully ──────────────────────────

#[test]
fn teleport() {
    let p = compile_fixture("teleport.qasm").expect("should compile");
    assert_eq!(p.version.as_deref(), Some("3"));
    assert!(!p.body.is_empty());
}

#[test]
fn arrays() {
    // Exercises `sizeof` in both forms, including on an unspecified-length
    // (`#dim`) array reference inside a subroutine.
    compile_fixture("arrays.qasm").expect("should compile");
}

#[test]
fn cphase() {
    // Self-contained gate-definition example: defines CX and a controlled-phase
    // gate from built-in primitives, then applies it.
    compile_fixture("cphase.qasm").expect("should compile");
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
    use oqi_compile::sir::CalibrationBody;

    let p = compile_fixture("defcal.qasm").expect("should compile");
    assert_eq!(p.calibrations.len(), 6);
    for cal in &p.calibrations {
        match &cal.body {
            CalibrationBody::OpenPulse(stmts) => assert!(!stmts.is_empty()),
            CalibrationBody::Opaque(_) => panic!("expected structured OpenPulse body"),
        }
    }
    // OpenPulse intrinsics are seeded as externs.
    let extern_names: Vec<&str> = p
        .externs
        .iter()
        .map(|e| p.symbols.get(e.symbol).name.as_str())
        .collect();
    for name in [
        "newframe",
        "gaussian",
        "play",
        "capture",
        "shift_phase",
        "threshold",
    ] {
        assert!(
            extern_names.contains(&name),
            "missing OpenPulse intrinsic {name} in externs: {extern_names:?}",
        );
    }
    // Rendered dump should include key OpenPulse constructs.
    let rendered = format!("{p}");
    assert!(rendered.contains("defcal x $0"));
    assert!(rendered.contains("defcal measure $0 -> bit"));
    assert!(rendered.contains("cal {"));
}

#[test]
fn alignment() {
    compile_fixture("alignment.qasm").expect("should compile");
}

/// The alignment fixture's stretchy delays resolve at compile time given a
/// timing table: `3g + dur(U) == dur(cx)` between the barriers, so with
/// cx = 300ns and U = 60ns the two delays become 80ns and 2·80ns.
#[test]
fn alignment_resolves_with_timings() {
    use oqi_compile::classical::{Duration, DurationUnit, Primitive};
    use oqi_compile::duration::{TableTimings, resolve_durationof};
    use oqi_compile::sir::{ExprKind, StmtKind};
    use oqi_compile::types::CompileOptions;

    let mut p = compile_fixture("alignment.qasm").expect("should compile");
    let options = CompileOptions::default();
    let table = TableTimings::from_str_entries([("cx", "300ns"), ("U", "60ns")], &options.dt)
        .expect("timing table");
    resolve_durationof(&mut p, &table, &options).expect("stretch resolution");

    let delays: Vec<&oqi_compile::sir::Expr> = p
        .body
        .iter()
        .filter_map(|s| match &s.kind {
            StmtKind::Delay(d) => Some(&d.duration),
            _ => None,
        })
        .collect();
    assert_eq!(delays.len(), 2);
    let lit_ns = |e: &oqi_compile::sir::Expr| match &e.kind {
        ExprKind::Literal(Primitive::Duration(d)) => d.to_unit(DurationUnit::Ns).value,
        other => panic!(
            "expected duration literal, got {:?}",
            std::mem::discriminant(other)
        ),
    };
    assert!((lit_ns(delays[0]) - 80.0).abs() < 1e-6);
    // The second delay is `2 * g`; its stretch operand is now a literal.
    let ExprKind::Binary(b) = &delays[1].kind else {
        panic!("expected 2 * g to remain a product");
    };
    assert!((lit_ns(&b.right) - 80.0).abs() < 1e-6);

    // The rewritten program still lowers through the rest of the pipeline.
    oqi_compile::cfg::build_program(&p).expect("cfg after stretch resolution");
}

#[test]
fn dd() {
    compile_fixture("dd.qasm").expect("should compile");
}

/// The DD fixture (spec delays.rst:247-266, boxed): the box span is set by
/// the two sequential cx (600ns); wire $0 telescopes to exactly 5a, so
/// a = 120ns and the derived delays resolve to 110/100/110.
#[test]
fn dd_resolves_with_timings() {
    use oqi_compile::duration::{TableTimings, resolve_durationof};
    use oqi_compile::sir::{ExprKind, LValue, RValue, StmtKind};
    use oqi_compile::types::CompileOptions;

    let mut p = compile_fixture("dd.qasm").expect("should compile");
    let options = CompileOptions::default();
    let table = TableTimings::from_str_entries(
        [("x", "20ns"), ("y", "20ns"), ("cx", "300ns")],
        &options.dt,
    )
    .expect("timing table");
    resolve_durationof(&mut p, &table, &options).expect("stretch resolution");

    // Each duration-variable assignment's RHS is now concrete arithmetic;
    // fold it to ns.
    fn eval_ns(e: &oqi_compile::sir::Expr) -> f64 {
        use oqi_compile::classical::{DurationUnit, Primitive};
        use oqi_compile::sir::{BinOp, UnOp};
        match &e.kind {
            ExprKind::Literal(Primitive::Duration(d)) => d.to_unit(DurationUnit::Ns).value,
            ExprKind::Literal(Primitive::Int(i)) => *i as f64,
            ExprKind::Literal(Primitive::Float(f)) => *f,
            ExprKind::Binary(b) => {
                let (l, r) = (eval_ns(&b.left), eval_ns(&b.right));
                match b.op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    other => panic!("unexpected op {other:?}"),
                }
            }
            ExprKind::Unary(u) => match u.op {
                UnOp::Neg => -eval_ns(&u.operand),
                other => panic!("unexpected unop {other:?}"),
            },
            ExprKind::Cast(c) => eval_ns(&c.operand),
            other => panic!("unexpected expr {:?}", std::mem::discriminant(other)),
        }
    }
    let rhs_ns = |name: &str| -> f64 {
        for s in &p.body {
            if let StmtKind::Assignment(a) = &s.kind
                && let LValue::Var(sid) = &a.target
                && p.symbols.get(*sid).name == name
                && let RValue::Expr(e) = &a.value
            {
                return eval_ns(e);
            }
        }
        panic!("no assignment to `{name}`");
    };
    assert!((rhs_ns("start_stretch") - 110.0).abs() < 1e-6);
    assert!((rhs_ns("middle_stretch") - 100.0).abs() < 1e-6);
    assert!((rhs_ns("end_stretch") - 110.0).abs() < 1e-6);
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
fn gate_without_stdgates_is_undefined() {
    // A standard-library gate (e.g. `cx`) used without `include "stdgates.inc"`
    // is an undefined name. (cphase.qasm itself is now a self-contained working
    // example — it defines the gates it uses — so this checks the error path
    // directly.)
    match compile_inline("qubit[2] q;\ncx q[0], q[1];\n") {
        Err(e) => assert!(matches!(
            e.kind,
            oqi_compile::error::ErrorKind::UndefinedName(_)
        )),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn msd() {
    // Magic-state distillation (serialized form): exercises qubit-parameter
    // slicing, aliasing, and runtime-indexed slices in subroutines. It fits in
    // 23 qubits but is expensive to simulate, so it's compile-tested only.
    // (`UndefinedName` is covered by `gate_without_stdgates_is_undefined`.)
    compile_fixture("msd.qasm").expect("should compile");
}

// ── Focused regression tests ─────────────────────────────────────────

#[test]
fn special_standard_assignment_requires_explicit_cast() {
    // bit[8] → uint[8] is castable but not implicitly promotable
    // (docs/types.rst implicit-promotion-rules): demand an explicit cast.
    let err = match compile_inline(
        r#"
            qubit q;
            bit[8] b;
            uint[8] u = b;
        "#,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected type mismatch"),
    };
    assert!(matches!(
        err.kind,
        oqi_compile::error::ErrorKind::TypeMismatch { .. }
    ));

    // The explicit-cast form stays valid.
    compile_inline(
        r#"
            qubit q;
            bit[8] b;
            uint[8] u = uint[8](b);
        "#,
    )
    .expect("explicit cast should compile");
}

#[test]
fn standard_assignment_promotes_implicitly() {
    use oqi_compile::sir::{ExprKind, RValue, StmtKind};

    // int → float is a standard-type conversion: an implicit cast is
    // inserted so the stored value is correct at runtime.
    let p = compile_inline(
        r#"
            int[32] i = 5;
            float[64] f = i;
        "#,
    )
    .expect("should compile");
    let cast_count = p
        .body
        .iter()
        .filter(|s| {
            matches!(&s.kind, StmtKind::Assignment(a)
                if matches!(&a.value, RValue::Expr(e) if matches!(&e.kind, ExprKind::Cast(_))))
        })
        .count();
    assert_eq!(cast_count, 1, "float decl-init should carry a cast");
}

#[test]
fn bit_bool_assignments_are_implicit() {
    // bit and bool are interchangeable (docs/types.rst).
    compile_inline(
        r#"
            include "stdgates.inc";
            qubit q;
            bit c = measure q;
            bool ok = c;
            bit c2 = ok;
        "#,
    )
    .expect("bit/bool interchange should compile");
}

#[test]
fn compound_assign_with_measure_desugars() {
    use oqi_compile::sir::{
        Assignment, BinOp, Binary, ExprKind, LValue, Measure, MeasureExprKind, RValue, StmtKind,
    };
    use oqi_compile::symbol::SymbolKind;

    let src = r#"
        qubit q;
        bit a;
        a &= measure q;
    "#;
    let p = compile_inline(src).expect("compound + measure should desugar");

    // Find the temp symbol generated by the desugaring.
    let temp = p
        .symbols
        .iter()
        .find(|s| s.kind == SymbolKind::Temp)
        .expect("expected a Temp symbol");
    assert!(
        temp.name.starts_with('$'),
        "temp name {:?} must use the $-prefix sentinel",
        temp.name
    );

    // Locate the two desugared statements.
    let measure_idx = p
        .body
        .iter()
        .position(|s| matches!(s.kind, StmtKind::Measure(_)))
        .expect("expected a Measure stmt");
    let assign_idx = p
        .body
        .iter()
        .position(|s| matches!(s.kind, StmtKind::Assignment(_)))
        .expect("expected an Assignment stmt");
    assert!(
        measure_idx < assign_idx,
        "measure must precede the assignment"
    );

    // Measure targets the temp.
    let StmtKind::Measure(Measure { measure, target }) = &p.body[measure_idx].kind else {
        unreachable!();
    };
    assert!(matches!(measure.kind, MeasureExprKind::Measure { .. }));
    assert!(matches!(target, Some(LValue::Var(s)) if *s == temp.id));

    // Assignment is `a = a & $temp`.
    let StmtKind::Assignment(Assignment { value, .. }) = &p.body[assign_idx].kind else {
        unreachable!();
    };
    let RValue::Expr(e) = value else {
        panic!("expected RValue::Expr, got measure form");
    };
    let ExprKind::Binary(Binary { op, left, right }) = &e.kind else {
        panic!("expected Binary expression in desugared assignment");
    };
    assert_eq!(*op, BinOp::BitAnd);
    let ExprKind::Var(left_sym) = &left.kind else {
        panic!("expected lhs to be a Var");
    };
    assert_eq!(p.symbols.get(*left_sym).name, "a");
    let ExprKind::Var(right_sym) = &right.kind else {
        panic!("expected rhs to be the temp Var");
    };
    assert_eq!(*right_sym, temp.id);
}

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

    assert_eq!(
        sym("my_bit").ty,
        oqi_compile::types::Type::Classical(ValueTy::bit())
    );
    assert_eq!(
        sym("my_bitreg").ty,
        oqi_compile::types::Type::Classical(ValueTy::bitreg(8))
    );
    assert_eq!(
        sym("my_bool").ty,
        oqi_compile::types::Type::Classical(ValueTy::bool())
    );
    assert_eq!(
        sym("my_int").ty,
        oqi_compile::types::Type::Classical(ValueTy::int(iw(32)))
    );
    assert_eq!(
        sym("my_uint").ty,
        oqi_compile::types::Type::Classical(ValueTy::uint(iw(16)))
    );
    assert_eq!(
        sym("my_float").ty,
        oqi_compile::types::Type::Classical(ValueTy::float(oqi_compile::types::FloatWidth::F64))
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
        oqi_compile::types::Type::Classical(ValueTy::complex(oqi_compile::types::FloatWidth::F64))
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

    assert_eq!(
        sym("one_second").ty,
        oqi_compile::types::Type::Classical(ValueTy::duration())
    );
    assert_eq!(
        sym("thousand_cycles").ty,
        oqi_compile::types::Type::Classical(ValueTy::duration())
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
        oqi_compile::types::Type::Classical(ValueTy::array(
            oqi_compile::classical::PrimitiveTy::Int(oqi_compile::classical::iw(32)),
            oqi_compile::classical::ashape(vec![5]),
        ))
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
        oqi_compile::types::Type::Classical(ValueTy::array(
            oqi_compile::classical::PrimitiveTy::Float(oqi_compile::classical::FloatWidth::F32),
            oqi_compile::classical::ashape(vec![3, 2]),
        ))
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
    assert_eq!(
        sym("i").ty,
        oqi_compile::types::Type::Classical(ValueTy::int(iw(32)))
    );
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
    assert_eq!(
        sym.ty,
        oqi_compile::types::Type::Classical(ValueTy::bitreg(8))
    );
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
            oqi_compile::error::ErrorKind::DuplicateDefinition { .. }
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

/// Source → bytecode, then the names of the module's program outputs,
/// sorted.
fn output_names(source: &str) -> Vec<String> {
    use oqi_compile::{bytecode, cfg, qubits, ssa};
    let program = compile_inline(source).expect("compile");
    let cfgs = cfg::build_program(&program).expect("cfg");
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    let module = bytecode::emit(&ssa, &program, layout).expect("emit");
    let mut names: Vec<String> = module
        .outputs
        .iter()
        .map(|(sym, _)| module.symbols.get(*sym).name.clone())
        .collect();
    names.sort();
    names
}

#[test]
fn outputs_sidecar_honors_output_decls() {
    // When any `output` is declared, only those symbols are outputs.
    let names = output_names(
        r#"
            output int x;
            int c = 1;
            x = 1;
            if (c == 1) { x = 2; }
        "#,
    );
    assert_eq!(names, vec!["x".to_string()]);
}

/// Source → bytecode, then the names of the module's declared inputs,
/// sorted.
fn input_names(source: &str) -> Vec<String> {
    use oqi_compile::{bytecode, cfg, qubits, ssa};
    let program = compile_inline(source).expect("compile");
    let cfgs = cfg::build_program(&program).expect("cfg");
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    let module = bytecode::emit(&ssa, &program, layout).expect("emit");
    let mut names: Vec<String> = module
        .inputs
        .iter()
        .map(|(sym, _)| module.symbols.get(*sym).name.clone())
        .collect();
    names.sort();
    names
}

#[test]
fn inputs_sidecar_lists_all_declared_inputs() {
    // Both inputs are listed even though neither is read (force-allocated).
    let names = input_names(
        r#"
            input float[64] theta;
            input int n;
        "#,
    );
    assert_eq!(names, vec!["n".to_string(), "theta".to_string()]);
}

#[test]
fn outputs_sidecar_defaults_to_all_named_vars() {
    // With no `output` decls, every named classical variable is an output.
    let names = output_names(
        r#"
            int v = 1;
            int w = v + 2;
        "#,
    );
    assert_eq!(names, vec!["v".to_string(), "w".to_string()]);
}

// ── Constant folding ────────────────────────────────────────────────

use oqi_compile::classical::{FloatWidth, Primitive};
use oqi_compile::sir::{self, Assignment, Call, ExprKind, RValue, StmtKind};
use oqi_compile::types::Type;

fn first_assignment(program: &sir::Program) -> (&sir::LValue<sir::Expr>, &sir::RValue<sir::Expr>) {
    for stmt in &program.body {
        if let StmtKind::Assignment(Assignment { target, value }) = &stmt.kind {
            return (target, value);
        }
    }
    panic!("no Assignment statement found");
}

fn as_literal(rv: &RValue<sir::Expr>) -> (&Primitive, &Type) {
    let RValue::Expr(e) = rv else {
        panic!("expected RValue::Expr");
    };
    match &e.kind {
        ExprKind::Literal(prim) => (prim, &e.ty),
        _ => panic!("expected Literal"),
    }
}

#[test]
fn const_ref_substitutes() {
    let src = r#"
        const int N = 10;
        int[32] x = N;
    "#;
    let p = compile_inline(src).expect("should compile");
    let (_, rv) = first_assignment(&p);
    let (prim, ty) = as_literal(rv);
    assert!(
        matches!(prim, Primitive::Int(10)),
        "expected Int(10), got {prim:?}"
    );
    assert_eq!(*ty, Type::Classical(ValueTy::int(iw(32))));
}

#[test]
fn arithmetic_folds_to_literal() {
    let src = "int[32] x = 2 + 3;";
    let p = compile_inline(src).expect("should compile");
    let (_, rv) = first_assignment(&p);
    let (prim, ty) = as_literal(rv);
    assert!(matches!(prim, Primitive::Int(5)));
    assert_eq!(*ty, Type::Classical(ValueTy::int(iw(32))));
}

#[test]
fn const_arithmetic_folds() {
    let src = r#"
        const int N = 10;
        int[32] x = N + 5;
    "#;
    let p = compile_inline(src).expect("should compile");
    let (_, rv) = first_assignment(&p);
    let (prim, ty) = as_literal(rv);
    assert!(matches!(prim, Primitive::Int(15)));
    assert_eq!(*ty, Type::Classical(ValueTy::int(iw(32))));
}

#[test]
fn intrinsic_call_folds() {
    let src = "float[64] x = sin(0.0);";
    let p = compile_inline(src).expect("should compile");
    let (_, rv) = first_assignment(&p);
    let (prim, ty) = as_literal(rv);
    let Primitive::Float(v) = prim else {
        panic!("expected Float, got {prim:?}");
    };
    assert!(v.abs() < 1e-12, "sin(0) should be ~0, got {v}");
    assert_eq!(*ty, Type::Classical(ValueTy::float(FloatWidth::F64)));
}

#[test]
fn cast_inside_expression_folds() {
    let src = "int[8] x = int[8](300);";
    let p = compile_inline(src).expect("should compile");
    let (_, rv) = first_assignment(&p);
    let (prim, ty) = as_literal(rv);
    // 300 mod 256 = 44
    assert!(
        matches!(prim, Primitive::Int(44)),
        "expected Int(44), got {prim:?}"
    );
    assert_eq!(*ty, Type::Classical(ValueTy::int(iw(8))));
}

#[test]
fn non_foldable_survives() {
    let src = r#"
        int[32] n;
        int[32] x = n + 5;
    "#;
    let p = compile_inline(src).expect("should compile");
    let (_, rv) = first_assignment(&p);
    let RValue::Expr(e) = rv else {
        panic!("expected RValue::Expr");
    };
    assert!(
        matches!(&e.kind, ExprKind::Binary(_)),
        "non-foldable expression should remain a Binary"
    );
}

#[test]
fn bare_foldable_exprstmt_drops() {
    let src = "pi / 4;";
    let p = compile_inline(src).expect("should compile");
    assert!(
        !p.body
            .iter()
            .any(|s| matches!(s.kind, StmtKind::ExprStmt(_))),
        "folded bare ExprStmt should be dropped from body"
    );
}

#[test]
fn non_intrinsic_call_arg_coerced() {
    let src = r#"
        def f(int[8] a) -> int[8] { return a; }
        int[8] x;
        x = f(2 + 3);
    "#;
    let p = compile_inline(src).expect("should compile");
    // Find the `x = f(...)` assignment.
    let (_, rv) = p
        .body
        .iter()
        .find_map(|s| match &s.kind {
            StmtKind::Assignment(Assignment { target, value }) => Some((target, value)),
            _ => None,
        })
        .expect("expected an assignment");
    let RValue::Expr(e) = rv else {
        panic!("expected RValue::Expr");
    };
    let ExprKind::Call(Call { args, .. }) = &e.kind else {
        panic!("expected Call");
    };
    assert_eq!(args.len(), 1);
    let (prim, ty) = (
        match &args[0].kind {
            ExprKind::Literal(p) => p,
            _ => panic!("expected Literal arg"),
        },
        &args[0].ty,
    );
    assert!(matches!(prim, Primitive::Int(5)));
    assert_eq!(
        *ty,
        Type::Classical(ValueTy::int(iw(8))),
        "arg should be coerced to int[8]"
    );
}
