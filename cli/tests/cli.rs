//! End-to-end tests of the `oqi` binary.

use std::path::PathBuf;
use std::process::Command;

/// Write `source` to a scratch file under the target directory.
fn file(name: &str, source: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, source).unwrap();
    path
}

/// Run `oqi` with `args`; return (success, stdout, stderr).
fn oqi(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_oqi"))
        .args(args)
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn oqi");
    (
        out.status.success(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

/// Like [`oqi`], for runs that must succeed; returns stdout.
fn ok(args: &[&str]) -> String {
    let (success, stdout, stderr) = oqi(args);
    assert!(success, "oqi {args:?} failed: {stderr}");
    stdout
}

const BELL: &str = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
output bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"#;

#[test]
fn compile_runs_the_whole_pipeline() {
    let bell = file("compile_bell.qasm", BELL);
    let bell = bell.to_str().unwrap();
    assert_eq!(ok(&["compile", bell]), "");
    assert!(ok(&["compile", bell, "--dump"]).starts_with(".module openqasm 3"));
    assert!(ok(&["compile", bell, "--dump-sir"]).starts_with("program v3"));

    // An error that only bytecode emission finds fails `compile`, not just
    // `run`.
    let late = file("compile_late.qasm", "qubit[2] q;\nreset q[5];\n");
    let (success, _, stderr) = oqi(&["compile", late.to_str().unwrap()]);
    assert!(!success);
    assert!(stderr.contains("C0015"), "{stderr}");
}

#[test]
fn syntax_errors_are_reported_alike_by_every_subcommand() {
    let broken = file("syntax.qasm", "// µ\nint x = ;\n");
    let broken = broken.to_str().unwrap();
    for subcommand in ["compile", "run", "fmt"] {
        let (success, _, stderr) = oqi(&[subcommand, broken]);
        assert!(!success);
        assert!(stderr.contains("S0001"), "{subcommand}: {stderr}");
        assert!(
            stderr.contains("expected expression"),
            "{subcommand}: {stderr}"
        );
        // The labelled snippet survives the multi-byte character before it.
        assert!(stderr.contains("syntax error"), "{subcommand}: {stderr}");
    }
}

#[test]
fn fmt_keeps_comments_and_the_final_newline() {
    let source = "qubit q; // one\n// two\n";
    let path = file("fmt_comments.qasm", source);
    ok(&["fmt", path.to_str().unwrap()]);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), source);
    assert_eq!(ok(&["fmt", "--stdout", path.to_str().unwrap()]), source);
}

#[test]
fn fmt_prints_several_files_in_argument_order() {
    let names: Vec<String> = (0..12)
        .map(|i| {
            let path = file(&format!("fmt_order_{i}.qasm"), &format!("qubit q{i};\n"));
            path.to_str().unwrap().to_string()
        })
        .collect();
    let mut args = vec!["fmt", "--stdout"];
    args.extend(names.iter().map(String::as_str));
    let expected: String = (0..12).map(|i| format!("qubit q{i};\n")).collect();
    for _ in 0..5 {
        assert_eq!(ok(&args), expected);
    }
}

#[test]
fn zero_shots_is_an_error() {
    let bell = file("zero_shots.qasm", BELL);
    let (success, _, stderr) = oqi(&["run", bell.to_str().unwrap(), "--shots", "0"]);
    assert!(!success);
    assert!(stderr.contains("--shots"), "{stderr}");
}

#[test]
fn an_error_in_an_included_file_is_shown_in_that_file() {
    let lib = file("diag_lib.qasm", "gate g a {\n  undefined_gate a;\n}\n");
    let main = file(
        "diag_main.qasm",
        "include \"./diag_lib.qasm\";\nqubit q;\ng q;\n",
    );
    let (success, _, stderr) = oqi(&["compile", main.to_str().unwrap()]);
    assert!(!success);
    assert!(
        stderr.contains(&format!("{}:2:3", lib.display())),
        "{stderr}"
    );
    assert!(stderr.contains("undefined_gate a;"), "{stderr}");
}

#[test]
fn seed_reproduces_a_run() {
    let bell = file("seed_bell.qasm", BELL);
    let bell = bell.to_str().unwrap();
    let outcomes: Vec<String> = (0..16u32)
        .map(|seed| {
            let seed = (seed + 1).to_string();
            let first = ok(&["run", bell, "--seed", &seed]);
            assert_eq!(first, ok(&["run", bell, "--seed", &seed]));
            first
        })
        .collect();
    assert!(
        outcomes.iter().any(|o| o != &outcomes[0]),
        "16 seeds should not all agree on a fair coin"
    );
}

#[test]
fn shots_print_a_histogram_per_output() {
    let bell = file("shots_bell.qasm", BELL);
    let bell = bell.to_str().unwrap();
    for backend in ["scalar", "rayon-simd", "auto"] {
        let args = [
            "run",
            bell,
            "--shots",
            "300",
            "--seed",
            "9",
            "--backend",
            backend,
        ];
        let stdout = ok(&args);
        let mut lines = stdout.lines();
        assert_eq!(lines.next(), Some("c:"));
        let counts: Vec<(&str, u32)> = lines
            .map(|line| {
                let (label, count) = line.trim().split_once(": ").unwrap();
                (label, count.parse().unwrap())
            })
            .collect();
        assert_eq!(
            counts.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
            ["00", "11"],
            "{backend}"
        );
        assert_eq!(counts.iter().map(|(_, n)| n).sum::<u32>(), 300);
        assert_eq!(stdout, ok(&args), "{backend}: same seed, same histogram");
    }
}

#[test]
fn inputs_cover_angles_bit_registers_and_arrays() {
    let src = file(
        "inputs.qasm",
        "OPENQASM 3.0;\n\
         input angle[32] theta;\n\
         input bit[4] mask;\n\
         input array[int[8], 3] xs;\n\
         output bit[4] copy;\n\
         output int total;\n\
         output bool half_turn;\n\
         copy = mask;\n\
         total = xs[0] + xs[1] + xs[2];\n\
         half_turn = theta == pi;\n",
    );
    let src = src.to_str().unwrap();
    let pi = std::f64::consts::PI.to_string();
    let theta = format!("theta={pi}");
    let stdout = ok(&[
        "run",
        src,
        "--input",
        &theta,
        "--input",
        "mask=0110",
        "--input",
        "xs=[1, 2, 3]",
    ]);
    assert_eq!(stdout, "copy = \"0110\"\ntotal = 6\nhalf_turn = true\n");

    let (success, _, stderr) = oqi(&[
        "run", src, "--input", &theta, "--input", "mask=01", "--input", "xs=1,2,3",
    ]);
    assert!(!success);
    assert!(
        stderr.contains("input `mask` must be a 4-character string"),
        "{stderr}"
    );

    let (success, _, stderr) = oqi(&["run", src, "--input", "nope=1"]);
    assert!(!success);
    assert!(stderr.contains("no input named `nope`"), "{stderr}");
}

#[test]
fn timings_file_is_combined_with_flags() {
    let src = file(
        "timed.qasm",
        "include \"stdgates.inc\";\n\
         qubit q;\n\
         output duration d;\n\
         d = durationof({ x q; h q; delay[4dt] q; });\n",
    );
    let profile = file(
        "profile.json",
        r#"{ "dt": "0.5ns", "x": "20ns", "h": "1ns" }"#,
    );
    let (src, profile) = (src.to_str().unwrap(), profile.to_str().unwrap());
    // File: x = 20ns, h = 1ns, dt = 0.5ns → 20 + 1 + 2.
    assert_eq!(ok(&["run", src, "--timings", profile]), "d = 23ns\n");
    // Flags override the file's entries and its dt.
    let args = [
        "run",
        src,
        "--timings",
        profile,
        "--timing",
        "h=10ns",
        "--dt",
        "1ns",
    ];
    assert_eq!(ok(&args), "d = 34ns\n");

    let (success, _, stderr) = oqi(&["run", src, "--dt", "2dt"]);
    assert!(!success);
    assert!(stderr.contains("--dt"), "{stderr}");
}

#[test]
fn runtime_errors_point_at_the_flag_that_fixes_them() {
    let src = file(
        "needs_input.qasm",
        "OPENQASM 3.0;\ninput int n;\nint m = n;\n",
    );
    let (success, _, stderr) = oqi(&["run", src.to_str().unwrap()]);
    assert!(!success);
    assert!(
        stderr.contains("no value supplied for input `n`"),
        "{stderr}"
    );
    assert!(stderr.contains("--input NAME=VALUE"), "{stderr}");
}

#[test]
fn bytecode_written_by_compile_runs_like_the_source() {
    let bell = file("bytecode_bell.qasm", BELL);
    let bell = bell.to_str().unwrap();
    // Any name: bytecode is recognized by its header.
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bytecode_bell.bin");
    let out = out.to_str().unwrap();
    assert_eq!(ok(&["compile", bell, "-o", out]), "");
    assert!(std::fs::read(out).unwrap().starts_with(b"OQIB"));

    let run = ["run", "--seed", "3", "--shots", "50"];
    assert_eq!(
        ok(&[run.as_slice(), &[out]].concat()),
        ok(&[run.as_slice(), &[bell]].concat())
    );
    assert_eq!(
        ok(&["compile", out, "--dump"]),
        ok(&["compile", bell, "--dump"])
    );

    // Its durations are resolved, and it keeps no structured IR.
    for args in [
        vec!["run", out, "--dt", "1ns"],
        vec!["compile", out, "--timing", "x=50ns"],
        vec!["compile", out, "--dump-sir"],
    ] {
        let (success, _, stderr) = oqi(&args);
        assert!(!success, "{args:?}");
        assert!(stderr.contains("bytecode"), "{args:?}: {stderr}");
    }
}

#[test]
fn errors_running_bytecode_are_reported_by_message() {
    let src = file(
        "bytecode_input.qasm",
        "input int n;\noutput int m;\nm = n + 1;\n",
    );
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bytecode_input.oqib");
    let out = out.to_str().unwrap();
    ok(&["compile", src.to_str().unwrap(), "-o", out]);
    assert_eq!(ok(&["run", out, "--input", "n=41"]), "m = 42\n");

    let (success, _, stderr) = oqi(&["run", out]);
    assert!(!success);
    assert!(
        stderr.contains("R0002") && stderr.contains("`n`"),
        "{stderr}"
    );

    // Damaged bytecode is an error, not a crash.
    let mut bytes = std::fs::read(out).unwrap();
    bytes.truncate(bytes.len() - 3);
    let damaged = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bytecode_damaged.oqib");
    std::fs::write(&damaged, bytes).unwrap();
    let (success, _, stderr) = oqi(&["run", damaged.to_str().unwrap()]);
    assert!(!success);
    assert!(stderr.contains("malformed bytecode"), "{stderr}");
}
