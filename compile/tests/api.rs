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

fn compile_default(src: &str) -> oqi_compile::Result<oqi_compile::BcModule> {
    compile(
        src,
        DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
}

#[test]
fn hardware_qubit_indices_must_fit_the_qubit_count() {
    for src in [
        "reset $4294967295;",
        "reset $4294967296;",
        "reset $99999999999999999999;",
    ] {
        let err = compile_default(src)
            .err()
            .unwrap_or_else(|| panic!("{src} compiled"));
        assert!(err.to_string().contains("hardware qubit"), "{src}: {err}");
        assert_ne!(err.span, Default::default(), "{src}");
    }
    assert_eq!(compile_default("reset $7;").unwrap().qubits.num_qubits, 8);
}

#[test]
fn either_quote_delimits_an_include_or_grammar() {
    for src in [
        "include 'stdgates.inc'; qubit q; h q;",
        r#"include "stdgates.inc"; qubit q; h q;"#,
    ] {
        compile_default(src).unwrap_or_else(|e| panic!("{src}: {e}"));
    }
    let module = compile_default("defcalgrammar 'openpulse';").unwrap();
    assert_eq!(module.calibration_grammar.as_deref(), Some("openpulse"));
}

#[test]
fn the_micro_sign_and_the_greek_mu_both_spell_microseconds() {
    // U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU (lexer.g4 accepts
    // both).
    for src in ["duration d = 100\u{b5}s;", "duration d = 100\u{3bc}s;"] {
        compile_default(src).unwrap_or_else(|e| panic!("{src}: {e}"));
    }
}

#[test]
fn only_openqasm_3_is_accepted() {
    for ok in ["OPENQASM 3;", "OPENQASM 3.0;", "OPENQASM 3.1;"] {
        compile_default(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in ["OPENQASM 2.0;", "OPENQASM 4;", "OPENQASM 30;"] {
        let err = compile_default(bad)
            .err()
            .unwrap_or_else(|| panic!("{bad} compiled"));
        assert!(err.to_string().contains("version"), "{bad}: {err}");
    }
}

/// Compile `src` on a thread with a `stack`-byte stack and return the error
/// message, if any.
fn compile_on_stack(src: String, stack: usize) -> Option<String> {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || compile_default(&src).err().map(|e| e.to_string()))
        .unwrap()
        .join()
        .expect("no stack overflow")
}

fn nest(n: usize, open: &str, middle: &str, close: &str) -> String {
    format!("{}{middle}{}", open.repeat(n), close.repeat(n))
}

#[test]
fn deep_nesting_is_a_syntax_error_not_a_stack_overflow() {
    // Far past the limit, on a test thread's 2 MiB stack.
    for src in [
        format!("int v = {};", nest(100_000, "(", "1", ")")),
        format!("int v = {}1;", "-".repeat(100_000)),
        nest(100_000, "{", "", "}"),
        nest(100_000, "if (true) ", "end;", ""),
    ] {
        let err = compile_on_stack(src, 2 << 20).expect("a syntax error");
        assert!(err.contains("nested more than 64 levels"), "{err}");
    }
    // Just under it, programs compile on a main thread's 8 MiB stack, even
    // in a debug build (release builds need far less; the wasm build runs
    // them in 1 MiB).
    let prelude = "include \"stdgates.inc\"; qubit q; int v;";
    for src in [
        format!("{prelude} {}", nest(30, "if (true) { ", "x q;", " }")),
        format!("{prelude} {}x q;", "if (true) ".repeat(60)),
        format!("{prelude} v = {};", nest(60, "(", "1", ")")),
    ] {
        assert_eq!(compile_on_stack(src, 8 << 20), None);
    }
}

#[test]
fn an_unterminated_string_is_a_syntax_error() {
    for src in [
        "include \"stdgates.inc\n;",
        "include 'stdgates.inc;",
        "defcalgrammar \"openpulse",
    ] {
        let err = compile_default(src)
            .err()
            .unwrap_or_else(|| panic!("{src:?} compiled"));
        assert_eq!(err.code().to_string(), "S0001", "{src:?}: {err}");
    }
}
