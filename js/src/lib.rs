//! JavaScript (wasm-bindgen) bindings for the oqi compiler and VM.
//!
//! Exposes a minimal API: [`compile`] (source → bytecode + disassembly) and
//! [`run`] (source → simulated results). Includes are limited to the embedded
//! `stdgates.inc`; file includes are rejected. Errors surface as thrown JS
//! `Error`s whose message is the rendered diagnostic.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use oqi_compile::bytecode::{self, BcModule};
use oqi_compile::classical::{Primitive, PrimitiveTy, Value, ValueTy};
use oqi_compile::resolve::IncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::{cfg, qubits, ssa};
use oqi_vm::{NoExterns, QuantumBackend, StateVectorSim, Vm};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Name used for the root source in rendered diagnostics.
const SOURCE_NAME: &str = "<source>";

/// Default RNG seed (matches the CLI's `DEFAULT_SEED` / `StateVectorSim::new`,
/// so CLI and JS runs of the same program and seed agree).
const DEFAULT_SEED: u64 = 0x2545F4914F6CDD1D;

/// Largest integer a JS `number` represents exactly (2^53 - 1).
const MAX_SAFE_INTEGER: u128 = (1 << 53) - 1;

/// Serves only embedded libraries (`stdgates.inc`); file includes error.
struct LibOnlyResolver;

impl IncludeResolver for LibOnlyResolver {
    fn resolve_path(
        &self,
        path: &Path,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Err(format!(
            "cannot include \"{}\": file includes are not supported in the JS build",
            path.display()
        )
        .into())
    }
}

/// Render any workspace diagnostic against the root source as a `JsError`.
fn diag_err(diag: &dyn oqi_diagnostics::Diagnostic, source: &str) -> JsError {
    JsError::new(&oqi_diagnostics::render_to_string(
        diag,
        Path::new(SOURCE_NAME),
        source,
    ))
}

/// Source → bytecode module; same pipeline the CLI drives in `run`.
fn build_module(source: &str) -> Result<BcModule, JsError> {
    let program =
        oqi_compile::lower::compile_source(source, LibOnlyResolver, Some(Path::new(SOURCE_NAME)))
            .map_err(|e| diag_err(&e, source))?;
    let cfgs = cfg::build_program(&program).map_err(|e| diag_err(&e, source))?;
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    bytecode::emit(&ssa, &program, layout).map_err(|e| diag_err(&e, source))
}

#[wasm_bindgen(getter_with_clone)]
pub struct CompileResult {
    /// Postcard-encoded bytecode module.
    pub bytecode: Vec<u8>,
    /// Textual disassembly of the module.
    pub disassembly: String,
}

/// Compile an OpenQASM 3 program to bytecode.
#[wasm_bindgen]
pub fn compile(source: &str) -> Result<CompileResult, JsError> {
    let module = build_module(source)?;
    let bytes = bytecode::to_bytes(&module)
        .map_err(|e| JsError::new(&format!("bytecode encoding failed: {e:?}")))?;
    Ok(CompileResult {
        bytecode: bytes,
        disassembly: module.to_string(),
    })
}

#[derive(Deserialize)]
#[serde(untagged)]
enum InputValue {
    Bool(bool),
    Number(f64),
    Text(String),
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RunOptions {
    /// Values for the program's `input` declarations, keyed by name.
    #[serde(default)]
    inputs: HashMap<String, InputValue>,
    /// Simulator RNG seed (number or bigint).
    #[serde(default)]
    seed: Option<u64>,
    /// Include the final state vector in the result.
    #[serde(default)]
    statevector: bool,
}

#[derive(Serialize)]
#[serde(untagged)]
enum OutputValue {
    Bool(bool),
    Number(f64),
    Text(String),
}

#[derive(Serialize)]
struct Measurement {
    qubit: u32,
    value: bool,
}

#[derive(Serialize)]
struct OutputEntry {
    name: String,
    value: OutputValue,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunOutput {
    measurements: Vec<Measurement>,
    outputs: Vec<OutputEntry>,
    /// Interleaved amplitudes `[re0, im0, re1, im1, ...]`, present iff requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    statevector: Option<Vec<f64>>,
}

/// Compile and run an OpenQASM 3 program on the state-vector simulator.
///
/// `options` is an optional object: `{ inputs?, seed?, statevector? }`.
/// Returns `{ measurements, outputs, statevector? }`.
#[wasm_bindgen]
pub async fn run(source: String, options: JsValue) -> Result<JsValue, JsError> {
    let module = build_module(&source)?;
    let opts: RunOptions = if options.is_undefined() || options.is_null() {
        RunOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("invalid options: {e}")))?
    };
    let inputs = coerce_inputs(&module, opts.inputs)?;

    let sim = StateVectorSim::<f64>::try_zeroed(
        module.qubits.num_qubits,
        opts.seed.unwrap_or(DEFAULT_SEED),
    )
    .map_err(|e| diag_err(&e, &source))?;
    let mut vm = Vm::new(&module, sim, NoExterns);
    let result = vm
        .run_with_inputs(inputs)
        .await
        .map_err(|e| diag_err(&e, &source))?;

    let statevector = if opts.statevector {
        vm.backend()
            .amplitudes()
            .await
            .map(|amps| amps.iter().flat_map(|c| [c.re, c.im]).collect())
    } else {
        None
    };

    let out = RunOutput {
        measurements: result
            .measurements
            .iter()
            .map(|&(qubit, value)| Measurement { qubit, value })
            .collect(),
        outputs: result
            .outputs
            .iter()
            .map(|(sym, v)| OutputEntry {
                name: module.symbols.get(*sym).name.clone(),
                value: output_value(v),
            })
            .collect(),
        statevector,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsError::new(&format!("result serialization failed: {e}")))
}

/// Coerce JS input values to the declared types of the program's `input`
/// symbols. Mirrors `parse_inputs` in `cli/src/main.rs`; keep the two in sync.
fn coerce_inputs(
    module: &BcModule,
    raw: HashMap<String, InputValue>,
) -> Result<HashMap<SymbolId, Value>, JsError> {
    let mut map = HashMap::new();
    for (name, input) in raw {
        let sym = module
            .symbols
            .iter()
            .find(|s| s.kind == SymbolKind::Input && s.name == name)
            .ok_or_else(|| JsError::new(&format!("no input named `{name}`")))?;
        let ty = sym
            .ty
            .value_ty()
            .ok_or_else(|| JsError::new(&format!("input `{name}` has no value type")))?;
        let value = match ty {
            ValueTy::Scalar(PrimitiveTy::Int(w)) => Value::int(input.to_i128(&name)?, w),
            ValueTy::Scalar(PrimitiveTy::Uint(w)) => Value::uint(input.to_u128(&name)?, w),
            ValueTy::Scalar(PrimitiveTy::Float(w)) => Value::float(input.to_f64(&name)?, w),
            ValueTy::Scalar(PrimitiveTy::Bit) | ValueTy::Scalar(PrimitiveTy::Bool) => {
                Value::bit(input.to_bool(&name)?)
            }
            _ => {
                return Err(JsError::new(&format!(
                    "input `{name}` has a type unsupported by the JS API"
                )));
            }
        };
        map.insert(sym.id, value);
    }
    Ok(map)
}

impl InputValue {
    fn to_i128(&self, name: &str) -> Result<i128, JsError> {
        match self {
            InputValue::Number(n) if n.fract() == 0.0 && n.abs() <= MAX_SAFE_INTEGER as f64 => {
                Ok(*n as i128)
            }
            InputValue::Text(s) => s
                .parse()
                .map_err(|_| JsError::new(&format!("input `{name}`: `{s}` is not an integer"))),
            _ => Err(JsError::new(&format!(
                "input `{name}` must be a safe integer or a decimal string"
            ))),
        }
    }

    fn to_u128(&self, name: &str) -> Result<u128, JsError> {
        match self {
            InputValue::Number(n)
                if n.fract() == 0.0 && *n >= 0.0 && *n as u128 <= MAX_SAFE_INTEGER =>
            {
                Ok(*n as u128)
            }
            InputValue::Text(s) => s.parse().map_err(|_| {
                JsError::new(&format!("input `{name}`: `{s}` is not an unsigned integer"))
            }),
            _ => Err(JsError::new(&format!(
                "input `{name}` must be a non-negative safe integer or a decimal string"
            ))),
        }
    }

    fn to_f64(&self, name: &str) -> Result<f64, JsError> {
        match self {
            InputValue::Number(n) => Ok(*n),
            InputValue::Text(s) => s
                .parse()
                .map_err(|_| JsError::new(&format!("input `{name}`: `{s}` is not a float"))),
            InputValue::Bool(_) => Err(JsError::new(&format!(
                "input `{name}` must be a number or a decimal string"
            ))),
        }
    }

    fn to_bool(&self, name: &str) -> Result<bool, JsError> {
        match self {
            InputValue::Bool(b) => Ok(*b),
            InputValue::Number(n) if *n == 0.0 => Ok(false),
            InputValue::Number(n) if *n == 1.0 => Ok(true),
            InputValue::Text(s) if s == "0" || s == "false" => Ok(false),
            InputValue::Text(s) if s == "1" || s == "true" => Ok(true),
            _ => Err(JsError::new(&format!(
                "input `{name}` is not a bit (use 0/1/true/false)"
            ))),
        }
    }
}

/// Values that fit a JS primitive losslessly come through natively; everything
/// else (bit registers, complex, duration, angle, arrays) is its OpenQASM text
/// form.
fn output_value(v: &Value) -> OutputValue {
    if let Value::Scalar(s) = v {
        match s.value() {
            Primitive::Bit(b) => return OutputValue::Bool(*b),
            Primitive::Int(i) if i.unsigned_abs() <= MAX_SAFE_INTEGER => {
                return OutputValue::Number(*i as f64);
            }
            Primitive::Uint(u) if *u <= MAX_SAFE_INTEGER => {
                return OutputValue::Number(*u as f64);
            }
            Primitive::Float(f) => return OutputValue::Number(*f),
            _ => {}
        }
    }
    OutputValue::Text(v.to_string())
}
