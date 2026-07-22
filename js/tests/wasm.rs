#![cfg(target_arch = "wasm32")]

use serde::Deserialize;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

const BELL: &str = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"#;

#[derive(Debug, Deserialize)]
struct Measurement {
    qubit: u32,
    value: bool,
}

#[derive(Debug, Deserialize)]
struct OutputEntry {
    name: String,
    value: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct RunOutput {
    measurements: Vec<Measurement>,
    outputs: Vec<OutputEntry>,
    #[serde(default)]
    statevector: Option<Vec<f64>>,
}

fn options(json: &str) -> JsValue {
    use serde::Serialize;
    let value: serde_json::Value = serde_json::from_str(json).unwrap();
    // json_compatible() serializes maps as plain JS objects, not `Map`s.
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap()
}

async fn run(source: &str, opts: JsValue) -> Result<RunOutput, JsValue> {
    let result = oqi_js::run(source.to_string(), opts).await?;
    Ok(serde_wasm_bindgen::from_value(result).unwrap())
}

#[derive(Debug, Deserialize)]
struct HistoBar {
    label: String,
    count: u32,
}

#[derive(Debug, Deserialize)]
struct Histogram {
    name: String,
    total: u32,
    bars: Vec<HistoBar>,
}

#[derive(Debug, Deserialize)]
struct SampleOutput {
    shots: u32,
    backend: String,
    precision: String,
    histograms: Vec<Histogram>,
}

async fn sample(source: &str, opts: JsValue) -> Result<SampleOutput, JsValue> {
    let result = oqi_js::sample(source.to_string(), opts).await?;
    Ok(serde_wasm_bindgen::from_value(result).unwrap())
}

fn error_message(err: JsValue) -> String {
    js_sys::Error::from(err).message().into()
}

#[wasm_bindgen_test]
fn compile_bell() {
    let result = oqi_js::compile(BELL).unwrap();
    assert!(!result.bytecode.is_empty());
    assert!(result.disassembly.contains(".module openqasm 3"));
    assert!(result.disassembly.contains(".proc"));
}

#[wasm_bindgen_test]
async fn run_bell() {
    let opts = r#"{ "seed": 1234, "statevector": true }"#;
    let out = run(BELL, options(opts)).await.unwrap();

    // Bell correlation: both measurements agree, on distinct qubits.
    assert_eq!(out.measurements.len(), 2);
    assert_ne!(out.measurements[0].qubit, out.measurements[1].qubit);
    assert_eq!(out.measurements[0].value, out.measurements[1].value);
    assert!(out.outputs.iter().any(|o| o.name == "c"));
    // 2 qubits -> 4 amplitudes, interleaved re/im.
    assert_eq!(out.statevector.as_ref().unwrap().len(), 8);

    // Fixed seed makes the run deterministic.
    let again = run(BELL, options(opts)).await.unwrap();
    assert_eq!(out.measurements[0].value, again.measurements[0].value);
}

#[wasm_bindgen_test]
fn include_path_rejected() {
    let err = oqi_js::compile("OPENQASM 3.0;\ninclude \"./foo.qasm\";\n")
        .err()
        .expect("expected compile error");
    assert!(error_message(err.into()).contains("file includes are not supported"));
}

#[wasm_bindgen_test]
fn syntax_error_carries_span() {
    // A missing `;` is a parse error; its source span must reach JS as a
    // non-empty `{ start, end }` property so the playground can highlight it.
    let err = oqi_js::compile("OPENQASM 3.0\n")
        .err()
        .expect("expected compile error");
    let span = js_sys::Reflect::get(&err, &JsValue::from_str("span")).unwrap();
    assert!(!span.is_undefined(), "syntax error should carry a span");
    let field = |k: &str| {
        js_sys::Reflect::get(&span, &JsValue::from_str(k))
            .unwrap()
            .as_f64()
            .unwrap()
    };
    let (start, end) = (field("start"), field("end"));
    assert!(end > start, "span should be non-empty, got {start}..{end}");
}

#[wasm_bindgen_test]
async fn bad_input_rejected() {
    let err = run(BELL, options(r#"{ "inputs": { "nope": 1 } }"#))
        .await
        .unwrap_err();
    assert!(error_message(err).contains("no input named `nope`"));
}

const ARRAY_SUM: &str = "OPENQASM 3.0;\n\
    input array[int, 3] xs;\n\
    output int total;\n\
    total = xs[0] + xs[1] + xs[2];\n";

#[wasm_bindgen_test]
async fn array_input_accepted() {
    let out = run(ARRAY_SUM, options(r#"{ "inputs": { "xs": [1, 2, 3] } }"#))
        .await
        .unwrap();
    assert_eq!(output(&out, "total").as_f64().unwrap(), 6.0);
}

#[wasm_bindgen_test]
async fn array_input_wrong_length_rejected() {
    let err = run(ARRAY_SUM, options(r#"{ "inputs": { "xs": [1, 2] } }"#))
        .await
        .unwrap_err();
    // `Array::new` rejects a length that doesn't match the declared shape.
    assert!(error_message(err).contains("input `xs`"));
}

const INC: &str = r#"
OPENQASM 3.0;
qubit q;
extern inc(int[32]) -> int[32];
int[32] x = 41;
int[32] y = inc(x);
"#;

const LOG_IT: &str = r#"
OPENQASM 3.0;
qubit q;
extern log_it(int[32]);
log_it(7);
"#;

/// `{ seed: 7 }` options with `externs` set from name → function pairs.
fn extern_options(externs: &[(&str, js_sys::Function)]) -> JsValue {
    let opts = options(r#"{ "seed": 7 }"#);
    let obj = js_sys::Object::new();
    for (name, f) in externs {
        js_sys::Reflect::set(&obj, &JsValue::from_str(name), f).unwrap();
    }
    js_sys::Reflect::set(&opts, &JsValue::from_str("externs"), &obj).unwrap();
    opts
}

fn output<'a>(out: &'a RunOutput, name: &str) -> &'a serde_json::Value {
    &out.outputs.iter().find(|o| o.name == name).unwrap().value
}

#[wasm_bindgen_test]
async fn extern_sync_callback() {
    let inc = js_sys::Function::new_with_args("x", "return x + 1;");
    let out = run(INC, extern_options(&[("inc", inc)])).await.unwrap();
    assert_eq!(output(&out, "y").as_f64().unwrap(), 42.0);
}

#[wasm_bindgen_test]
async fn extern_async_callback() {
    let inc = js_sys::Function::new_with_args("x", "return Promise.resolve(x + 1);");
    let out = run(INC, extern_options(&[("inc", inc)])).await.unwrap();
    assert_eq!(output(&out, "y").as_f64().unwrap(), 42.0);
}

#[wasm_bindgen_test]
async fn extern_void_return_ignored() {
    let log_it = js_sys::Function::new_with_args("x", "return 123;");
    run(LOG_IT, extern_options(&[("log_it", log_it)]))
        .await
        .unwrap();
}

#[wasm_bindgen_test]
async fn extern_throwing_callback() {
    let log_it = js_sys::Function::new_with_args("x", "throw new Error('boom');");
    let err = run(LOG_IT, extern_options(&[("log_it", log_it)]))
        .await
        .unwrap_err();
    let msg = error_message(err);
    assert!(msg.contains("extern function `log_it` failed"), "{msg}");
    assert!(msg.contains("boom"), "{msg}");
}

#[wasm_bindgen_test]
async fn extern_promise_rejection() {
    let log_it = js_sys::Function::new_with_args("x", "return Promise.reject(new Error('nope'));");
    let err = run(LOG_IT, extern_options(&[("log_it", log_it)]))
        .await
        .unwrap_err();
    let msg = error_message(err);
    assert!(msg.contains("extern function `log_it` failed"), "{msg}");
    assert!(msg.contains("nope"), "{msg}");
}

#[wasm_bindgen_test]
async fn extern_missing_is_rejected() {
    let err = run(INC, extern_options(&[])).await.unwrap_err();
    assert!(error_message(err).contains("extern function `inc` is not provided"));
}

#[wasm_bindgen_test]
async fn extern_angle_return() {
    let src = r#"
OPENQASM 3.0;
qubit q;
extern get_theta() -> angle[16];
angle[16] a = get_theta();
"#;
    let get_theta = js_sys::Function::new_no_args("return Math.PI / 2;");
    let out = run(src, extern_options(&[("get_theta", get_theta)]))
        .await
        .unwrap();
    assert_eq!(output(&out, "a").as_str().unwrap(), "(π/2)");
}

#[wasm_bindgen_test]
async fn extern_bitreg_return() {
    let src = r#"
OPENQASM 3.0;
qubit q;
extern pick() -> bit[4];
bit[4] r = pick();
"#;
    let pick = js_sys::Function::new_no_args("return '0110';");
    let out = run(src, extern_options(&[("pick", pick)])).await.unwrap();
    assert_eq!(output(&out, "r").as_str().unwrap(), "\"0110\"");
}

#[wasm_bindgen_test]
async fn extern_bitreg_arg_round_trip() {
    let src = r#"
OPENQASM 3.0;
qubit q;
extern parity(bit[4]) -> bit;
bit[4] r = "0011";
bit p = parity(r);
"#;
    // The callback asserts the arg arrives as an unquoted MSB-first string
    // (an order-sensitive, non-palindromic value).
    let parity = js_sys::Function::new_with_args(
        "s",
        "if (s !== '0011') throw new Error('got ' + s); return 0;",
    );
    let out = run(src, extern_options(&[("parity", parity)]))
        .await
        .unwrap();
    assert_eq!(output(&out, "p").as_bool().unwrap(), false);
}

#[wasm_bindgen_test]
async fn extern_bad_return_value() {
    let inc = js_sys::Function::new_with_args("x", "return 1.5;");
    let err = run(INC, extern_options(&[("inc", inc)])).await.unwrap_err();
    let msg = error_message(err);
    assert!(msg.contains("extern function `inc` failed"), "{msg}");
}

#[wasm_bindgen_test]
async fn extern_non_function_rejected() {
    let opts = options(r#"{ "externs": { "inc": 5 } }"#);
    let err = run(INC, opts).await.unwrap_err();
    assert!(error_message(err).contains("externs.inc is not a function"));
}

#[wasm_bindgen_test]
async fn extern_unused_is_allowed() {
    let unused = js_sys::Function::new_no_args("return 0;");
    run(BELL, extern_options(&[("unused", unused)]))
        .await
        .unwrap();
}

const TIMED: &str = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
duration d = durationof({x q; delay[30ns] q;});
bit c = measure q;
"#;

#[wasm_bindgen_test]
async fn run_with_timings_resolves_durationof() {
    // With a timing table, `durationof` resolves at compile time.
    let out = run(TIMED, options(r#"{ "timings": { "x": "20ns" } }"#))
        .await
        .unwrap();
    assert_eq!(output(&out, "d").as_str().unwrap(), "50ns");

    // dt-valued timings resolve against the `dt` option.
    let out = run(
        TIMED,
        options(r#"{ "timings": { "x": "40dt" }, "dt": "0.5ns" }"#),
    )
    .await
    .unwrap();
    assert_eq!(output(&out, "d").as_str().unwrap(), "50ns");

    // Without timings, the VM's runtime path still answers (x is 0ns).
    let out = run(TIMED, options(r#"{ "seed": 1 }"#)).await.unwrap();
    assert_eq!(output(&out, "d").as_str().unwrap(), "30ns");
}

#[wasm_bindgen_test]
async fn bad_timing_rejected() {
    let err = run(TIMED, options(r#"{ "timings": { "x": "abc" } }"#))
        .await
        .unwrap_err();
    assert!(error_message(err).contains("is not a duration literal"));
}

#[wasm_bindgen_test]
async fn sample_bell_histogram() {
    let out = sample(BELL, options(r#"{ "shots": 200, "seed": 1234 }"#))
        .await
        .unwrap();
    assert_eq!(out.shots, 200);
    assert_eq!(out.backend, "cpu");
    assert_eq!(out.precision, "f64");
    // One histogram for the `c` register; counts sum to the shot total, and a
    // Bell state only ever yields the correlated "00"/"11" outcomes.
    let c = out.histograms.iter().find(|h| h.name == "c").unwrap();
    assert_eq!(c.total, 200);
    assert_eq!(c.bars.iter().map(|b| b.count).sum::<u32>(), 200);
    for bar in &c.bars {
        assert!(bar.label == "00" || bar.label == "11", "got {}", bar.label);
    }
    assert!(c.bars.len() == 2, "both correlated outcomes should appear");
}

#[wasm_bindgen_test]
async fn sample_cpu_f32() {
    let out = sample(
        BELL,
        options(r#"{ "shots": 100, "backend": "cpu", "precision": "f32" }"#),
    )
    .await
    .unwrap();
    assert_eq!(out.backend, "cpu");
    assert_eq!(out.precision, "f32");
    let c = out.histograms.iter().find(|h| h.name == "c").unwrap();
    assert_eq!(c.total, 100);
    for bar in &c.bars {
        assert!(bar.label == "00" || bar.label == "11");
    }
}

#[wasm_bindgen_test]
async fn sample_feedback_uses_fallback() {
    // A gate conditioned on a measurement is not sample-safe, so this takes the
    // per-shot fallback. `x q[1]` fires iff `a == 1`, so `b == a` every shot.
    let src = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
bit a = measure q[0];
if (a) { x q[1]; }
bit b = measure q[1];
"#;
    let out = sample(src, options(r#"{ "shots": 128, "seed": 5 }"#))
        .await
        .unwrap();
    assert_eq!(out.shots, 128);
    let a = out.histograms.iter().find(|h| h.name == "a").unwrap();
    let b = out.histograms.iter().find(|h| h.name == "b").unwrap();
    assert_eq!(a.total, 128);
    assert_eq!(b.total, 128);
    assert_eq!(a.bars.len(), 2, "H on q0 yields both outcomes");
    let count0 = |h: &Histogram| {
        h.bars
            .iter()
            .find(|x| x.label == "0")
            .map_or(0, |x| x.count)
    };
    assert_eq!(count0(a), count0(b), "b must mirror a under feedback");
}

#[wasm_bindgen_test]
async fn sample_is_reproducible() {
    let opts = r#"{ "shots": 64, "seed": 99 }"#;
    let a = sample(BELL, options(opts)).await.unwrap();
    let b = sample(BELL, options(opts)).await.unwrap();
    // Same seed → identical histogram (labels + counts, in order).
    let ha = &a.histograms[0];
    let hb = &b.histograms[0];
    assert_eq!(ha.bars.len(), hb.bars.len());
    for (x, y) in ha.bars.iter().zip(&hb.bars) {
        assert_eq!(x.label, y.label);
        assert_eq!(x.count, y.count);
    }
}
