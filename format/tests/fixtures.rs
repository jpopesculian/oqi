use std::path::Path;

use oqi_format::{Config, format};
use oqi_parse::parse;

fn normalize_ast(source: &str) -> String {
    let program = parse(source).unwrap_or_else(|error| {
        panic!(
            "parse error: {} at {:?}\nsource:\n{}",
            error.message, error.span, source
        )
    });
    strip_spans(&format!("{program:#?}"))
}

fn strip_spans(debug: &str) -> String {
    let mut output = String::with_capacity(debug.len());
    let mut chars = debug.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            output.push(ch);
            continue;
        }

        if ch.is_ascii_digit() {
            let mut digits = String::from(ch);
            let mut probe = chars.clone();

            while let Some(next) = probe.peek().copied() {
                if !next.is_ascii_digit() {
                    break;
                }
                digits.push(next);
                probe.next();
            }

            let mut span_probe = probe.clone();
            if span_probe.next() == Some('.')
                && span_probe.next() == Some('.')
                && matches!(span_probe.peek(), Some(next) if next.is_ascii_digit())
            {
                while matches!(span_probe.peek(), Some(next) if next.is_ascii_digit()) {
                    span_probe.next();
                }
                output.push_str("<span>");
                chars = span_probe;
                continue;
            }

            output.push_str(&digits);
            chars = probe;
            continue;
        }

        output.push(ch);
    }

    output
}

fn check_fixture(name: &str) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let qasm_path = base.join("fixtures/qasm").join(format!("{name}.qasm"));
    let source = std::fs::read_to_string(&qasm_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", qasm_path.display()));

    let formatted_default = format(&source, Config::default()).unwrap_or_else(|error| {
        panic!(
            "{name}: format error: {} at {:?}",
            error.message, error.span
        )
    });
    let formatted_compact = format(&source, Config::compact()).unwrap_or_else(|error| {
        panic!(
            "{name}: format error: {} at {:?}",
            error.message, error.span
        )
    });

    let expected = normalize_ast(&source);
    let default = normalize_ast(&formatted_default);
    let compact = normalize_ast(&formatted_default);

    assert_eq!(
        expected, default,
        "{name}: AST changed after formatting\nformatted:\n{formatted_default}"
    );
    assert_eq!(
        expected, compact,
        "{name}: AST changed after formatting\nformatted:\n{formatted_compact}"
    );
}

macro_rules! fixture_test {
    ($name:ident) => {
        #[test]
        fn $name() {
            check_fixture(stringify!($name));
        }
    };
}

fixture_test!(adder);
fixture_test!(alignment);
fixture_test!(arrays);
fixture_test!(cphase);
fixture_test!(dd);
fixture_test!(defcal);
fixture_test!(gateteleport);
fixture_test!(inverseqft1);
fixture_test!(inverseqft2);
fixture_test!(ipe);
fixture_test!(msd);
fixture_test!(qec);
fixture_test!(qft);
fixture_test!(qpt);
fixture_test!(rb);
fixture_test!(rus);
fixture_test!(scqec);
fixture_test!(t1);
fixture_test!(teleport);
fixture_test!(varteleport);
fixture_test!(vqe);
