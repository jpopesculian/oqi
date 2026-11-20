//! The crate's public entry points: `compile`, `compile_source`, and the
//! option helpers the front doors share.

use std::borrow::Cow;
use std::path::Path;

use oqi_compile::classical::{DurationUnit, ValueTy, iw};
use oqi_compile::symbol::SymbolKind;
use oqi_compile::{
    CompileOptions, DefaultIncludeResolver, ErrorKind, IncludeResolver, LibOnlyResolver,
    SystemWidth, compile, compile_source, parse_dt,
};
use oqi_diagnostics::Diagnostic;

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

#[test]
fn syntax_errors_keep_their_code_and_location() {
    let source = "int x = ;";
    let err = compile(
        source,
        DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
    .err()
    .expect("a compile error");
    assert!(matches!(err.kind, ErrorKind::Syntax(_)), "{:?}", err.kind);
    assert_eq!(err.code().to_string(), "S0001");
    assert_eq!(err.to_string(), "expected expression");
    assert_eq!(&source[err.span], ";");
}

/// Serves the same broken source for every file, and another for every
/// library.
struct BrokenIncludes;

const BROKEN_FILE: &str = "int x = ;\n";

impl IncludeResolver for BrokenIncludes {
    fn resolve_path(&self, _: &Path) -> Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Ok(Cow::Borrowed(BROKEN_FILE))
    }

    fn resolve_lib(&self, _: &str) -> Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Ok(Cow::Borrowed("gate broken q { x q }\n"))
    }
}

#[test]
fn a_syntax_error_in_an_included_file_points_into_that_file() {
    let options = CompileOptions {
        source_name: Some("/project/main.qasm".into()),
        ..Default::default()
    };
    let err = compile(r#"include "./other.qasm";"#, BrokenIncludes, &options, None)
        .err()
        .expect("a compile error");
    assert_eq!(err.code().to_string(), "S0001");
    assert_eq!(err.to_string(), "expected expression");
    assert_eq!(err.path.as_deref(), Some(Path::new("/project/other.qasm")));
    assert_eq!(&BROKEN_FILE[err.span], ";");
}

#[test]
fn a_syntax_error_in_a_library_is_reported_at_the_include() {
    // A library is not a file a diagnostic could point into.
    let source = r#"qubit q; include "broken.inc";"#;
    let err = compile(source, BrokenIncludes, &CompileOptions::default(), None)
        .err()
        .expect("a compile error");
    assert_eq!(err.code().to_string(), "S0001");
    assert_eq!(err.to_string(), "in include 'broken.inc': expected ';'");
    assert_eq!(&source[err.span], r#"include "broken.inc";"#);
}
