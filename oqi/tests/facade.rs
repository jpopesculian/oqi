//! The facade end to end: everything a user does goes through `oqi::` and
//! `oqi::Error`.

use std::path::Path;

use oqi::{Diagnostic, Error, FnRegistry, HostValue, Runner, Value};

const BELL: &str = r#"
    include "stdgates.inc";
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
    bit[2] c = measure q;
"#;

/// Compile, run and sample, with every error converted by `?`.
fn bell_histogram(backend: &str) -> oqi::Result<Vec<(String, u32)>> {
    let program = oqi::compile(BELL)?;
    let samples = Runner::new(&program)
        .backend(backend.parse()?)
        .seed(11)
        .sample_blocking(200)?;
    assert_eq!(samples.shots(), 200);
    Ok(samples.histogram("c").unwrap().counts().to_vec())
}

#[test]
fn compiles_runs_and_samples() {
    for backend in ["scalar", "simd", "auto", "sum", "mps"] {
        let counts = bell_histogram(backend).unwrap();
        let labels: Vec<&str> = counts.iter().map(|(label, _)| label.as_str()).collect();
        assert_eq!(labels, ["00", "11"], "{backend}");
        assert_eq!(counts.iter().map(|(_, n)| n).sum::<u32>(), 200);
    }
}

#[test]
fn runs_with_named_inputs_and_externs() -> oqi::Result<()> {
    let program = oqi::compile(
        r#"
        include "stdgates.inc";
        extern bump(int) -> int;
        input int n;
        int m = bump(n);
        qubit q;
        if (m == 4) { x q; }
        bit c = measure q;
        "#,
    )?;
    let mut externs = FnRegistry::new();
    externs.register("bump", |args: &[Value]| match HostValue::from(&args[0]) {
        HostValue::Int(n) => Ok(Some(HostValue::Int(n + 1).coerce_to(args[0].ty())?)),
        other => panic!("not an int: {other:?}"),
    });
    let result = Runner::new(&program)
        .input("n", 3)
        .externs(externs)
        .run_blocking()?;
    assert_eq!(result.output("m").unwrap().to_string(), "4");
    assert_eq!(result.output("c").unwrap().to_string(), "1");
    Ok(())
}

#[test]
fn every_step_s_error_converts_and_renders() {
    // Syntax, through parse and format.
    let source = "int x = ;";
    let error: Error = oqi::parse(source).unwrap_err().into();
    assert!(matches!(error, Error::Syntax(_)));
    assert_eq!(error.code().to_string(), "S0001");
    let error: Error = oqi::format(source, oqi::FormatConfig::default())
        .unwrap_err()
        .into();
    assert!(matches!(error, Error::Syntax(_)));

    // Compile, rendered against its source.
    let source = "qubit q;\nfoo q;\n";
    let error: Error = oqi::compile(source).unwrap_err().into();
    assert!(matches!(error, Error::Compile(_)));
    let report = oqi::render_to_string(&error, Path::new("main.qasm"), source, &|_| None);
    assert!(report.contains("[C0001]"), "{report}");
    assert!(
        report.contains("main.qasm:2:1: undefined name 'foo'"),
        "{report}"
    );
    assert_eq!(error.to_string(), "undefined name 'foo'");

    // Runtime: a missing input.
    let program = oqi::compile("input int n; int m = n;").unwrap();
    let error: Error = Runner::new(&program).run_blocking().unwrap_err().into();
    assert!(matches!(error, Error::Runtime(_)));
    assert_eq!(error.code().to_string(), "R0002");

    // An unknown backend name, as a runtime error.
    let error: Error = "warp-drive".parse::<oqi::BackendSpec>().unwrap_err().into();
    assert_eq!(error.code().to_string(), "R0016");
    assert!(error.to_string().contains("warp-drive"), "{error}");

    // Bytecode that isn't, rendered message-only.
    let error: Error = oqi::Program::from_bytes(b"OQIB\x09\x00")
        .unwrap_err()
        .into();
    assert!(matches!(error, Error::Decode(_)));
    assert_eq!(error.code().to_string(), "C0018");
    let report = oqi::render_to_string(&error, Path::new("prog.oqib"), "", &|_| None);
    assert!(report.contains("format version 9"), "{report}");
}

#[test]
fn bytecode_round_trips() -> oqi::Result<()> {
    let program = oqi::compile(BELL)?;
    let loaded = oqi::Program::from_bytes(&program.to_bytes()?)?;
    let run = |program: &oqi::Program| -> oqi::Result<String> {
        let result = Runner::new(program).seed(3).run_blocking()?;
        Ok(result.output("c").unwrap().to_string())
    };
    assert_eq!(run(&loaded)?, run(&program)?);
    assert_eq!(loaded.to_string(), program.to_string());
    Ok(())
}

#[test]
fn a_backend_built_by_type_runs_on_the_interpreter() -> oqi::Result<()> {
    let program = oqi::compile(BELL)?;
    let backend = oqi::StateVectorSim::<f64>::new(program.num_qubits())?.with_seed(5);
    let mut vm = oqi::vm::Vm::new(&program, backend, oqi::NoExterns);
    let result = oqi::block_on(vm.run())?;
    let c = result.output("c").unwrap().to_string();
    assert!(c == "\"00\"" || c == "\"11\"", "{c}");
    Ok(())
}

#[test]
fn formats_source() -> oqi::Result<()> {
    let compact = oqi::FormatConfig::default().compact(true);
    assert_eq!(
        oqi::format("qubit q;\n\nh  q ;", compact)?,
        "qubit q; h q;\n"
    );
    Ok(())
}
