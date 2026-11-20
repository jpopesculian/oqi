//! The crate's public entry points: `compile`, `compile_source`, and the
//! option helpers the front doors share.

use oqi_compile::classical::{DurationUnit, ValueTy, iw};
use oqi_compile::symbol::SymbolKind;
use oqi_compile::{
    CompileOptions, DefaultIncludeResolver, ErrorKind, LibOnlyResolver, SystemWidth, compile,
    compile_source, parse_dt,
};

const TIMED: &str = r#"
    include "stdgates.inc";
    qubit q;
    duration d = durationof({ x q; delay[30ns] q; });
"#;

#[test]
fn compile_goes_from_source_to_bytecode() {
    let module = compile(
        "input int[32] n; output int[32] m; m = n + 1;",
        DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
    .expect("compile");
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);

    // Inputs resolve by name to their symbol and value type.
    let (sym, ty) = module.input("n").expect("declared input");
    assert_eq!(module.symbol(sym).name, "n");
    assert_eq!(module.symbol(sym).kind, SymbolKind::Input);
    assert_eq!(ty, ValueTy::int(iw(32)));
    assert!(module.input("m").is_none(), "an output is not an input");
    assert!(module.input("nope").is_none());
}

#[test]
fn externs_are_listed_with_their_return_types() {
    let module = compile(
        "extern inc(int[32]) -> int[32]; extern log_it(int[32]);",
        DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
    .expect("compile");
    let externs: Vec<_> = module.externs().collect();
    assert_eq!(
        externs,
        [("inc", Some(ValueTy::int(iw(32)))), ("log_it", None)]
    );
}

#[test]
fn timings_resolve_durationof_at_compile_time() {
    let has_durationof = |timings| {
        let program = compile_source(
            TIMED,
            DefaultIncludeResolver,
            &CompileOptions::default(),
            timings,
        )
        .expect("compile");
        format!("{program}").contains("durationof")
    };
    // Left for the VM without timings; folded to a constant with them (an
    // empty table still enables the pass).
    assert!(has_durationof(None));
    assert!(!has_durationof(Some(&[("x", "20ns")])));
    assert!(!has_durationof(Some(&[])));

    let err = compile(
        TIMED,
        DefaultIncludeResolver,
        &CompileOptions::default(),
        Some(&[("x", "abc")]),
    )
    .err()
    .expect("bad timing literal");
    assert!(matches!(err.kind, ErrorKind::InvalidLiteral(_)));
}

#[test]
fn lib_only_resolver_serves_stdgates_but_no_files() {
    // Relative includes resolve against the root source's name.
    let options = CompileOptions {
        source_name: Some("<source>".into()),
        ..Default::default()
    };
    compile(
        r#"include "stdgates.inc";"#,
        LibOnlyResolver,
        &options,
        None,
    )
    .expect("stdgates");
    let err = compile(
        r#"include "./other.qasm";"#,
        LibOnlyResolver,
        &options,
        None,
    )
    .err()
    .expect("file include");
    assert!(
        err.to_string().contains("file includes are not supported"),
        "{err}"
    );
}

#[test]
fn parse_dt_accepts_positive_si_durations_only() {
    let dt = parse_dt("0.5ns").expect("0.5ns");
    assert_eq!((dt.value, dt.unit), (0.5, DurationUnit::Ns));
    for bad in ["4dt", "abc", "0ns", "-1ns"] {
        let err = parse_dt(bad).expect_err(bad);
        assert!(matches!(err.kind, ErrorKind::InvalidLiteral(_)), "{bad}");
    }
}

#[test]
fn plain_types_are_64_bits_wide_on_every_host() {
    // Not the compiler host's pointer width: the same program must mean the
    // same thing natively and on wasm32.
    assert_eq!(SystemWidth::default(), SystemWidth::B64);
    let module = compile(
        "input int n;",
        DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
    .expect("compile");
    assert_eq!(module.input("n").unwrap().1, ValueTy::int(iw(64)));
}
