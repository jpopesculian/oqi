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
async fn bad_input_rejected() {
    let err = run(BELL, options(r#"{ "inputs": { "nope": 1 } }"#))
        .await
        .unwrap_err();
    assert!(error_message(err).contains("no input named `nope`"));
}
