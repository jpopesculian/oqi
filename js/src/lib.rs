//! JavaScript (wasm-bindgen) bindings for the oqi compiler and VM.
//!
//! Exposes a minimal API: [`compile`] (source → bytecode + disassembly),
//! [`run`] (source → simulated results) and [`sample`] (source → per-output
//! histograms over many shots). `run` and `sample` can call host-supplied
//! `extern` implementations (the `externs` option), synchronous or
//! Promise-returning. Includes are limited to the embedded `stdgates.inc`;
//! file includes are rejected. Errors surface as thrown JS `Error`s whose
//! message is the rendered diagnostic.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use oqi_compile::classical::{Value, ValueTy};
use oqi_compile::{BcModule, CompileOptions, LibOnlyResolver};
use oqi_vm::{
    BackendSpec, DEFAULT_SEED, ExternProvider, HostValue, KindResult, QuantumBackend, SumPolicy,
    Vm, VmErrorKind,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Name used for the root source in rendered diagnostics.
const SOURCE_NAME: &str = "<source>";

/// Largest integer a JS `number` represents exactly (2^53 - 1).
const MAX_SAFE_INTEGER: u128 = (1 << 53) - 1;

/// Render any workspace diagnostic against the root source as a thrown JS
/// `Error`. The rendered diagnostic is the `message`; when the primary label
/// has a real source span, it's attached as a `{ start, end }` byte-offset
/// `span` property so callers (e.g. the playground) can highlight it.
fn diag_err(diag: &dyn oqi_diagnostics::Diagnostic, source: &str) -> JsValue {
    let err = js_sys::Error::new(&oqi_diagnostics::render_to_string(
        diag,
        Path::new(SOURCE_NAME),
        source,
    ));
    if let Some(span) = oqi_diagnostics::primary_span(diag) {
        let (mut start, mut end) = (span.start, span.end);
        if start <= source.len() && end <= source.len() {
            // Widen a zero-width span (e.g. a parser "expected ';'") to one
            // character so the editor has a visible range. Prefer the char
            // *before* the gap — for a missing `;` the position sits on the
            // following newline, and the token before it is what's relevant.
            if end == start {
                if let Some(c) = source[..start].chars().next_back() {
                    start -= c.len_utf8();
                } else if let Some(c) = source[start..].chars().next() {
                    end = start + c.len_utf8();
                }
            }
            if end > start {
                let obj = js_sys::Object::new();
                let _ = js_sys::Reflect::set(
                    &obj,
                    &JsValue::from_str("start"),
                    &JsValue::from_f64(start as f64),
                );
                let _ = js_sys::Reflect::set(
                    &obj,
                    &JsValue::from_str("end"),
                    &JsValue::from_f64(end as f64),
                );
                let _ = js_sys::Reflect::set(&err, &JsValue::from_str("span"), &obj);
            }
        }
    }
    err.into()
}

/// Source → bytecode module. With `timings` supplied (even an empty map),
/// `durationof` is resolved at compile time; otherwise it stays for the VM's
/// runtime pass.
fn build_module(
    source: &str,
    timings: Option<&HashMap<String, String>>,
    dt: Option<&str>,
) -> Result<BcModule, JsValue> {
    let mut options = CompileOptions {
        source_name: Some(Path::new(SOURCE_NAME).to_path_buf()),
        ..Default::default()
    };
    if let Some(spec) = dt {
        options.dt = oqi_compile::parse_dt(spec).map_err(|e| diag_err(&e, source))?;
    }
    let timings: Option<Vec<(&str, &str)>> =
        timings.map(|t| t.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());
    oqi_compile::compile(source, LibOnlyResolver, &options, timings.as_deref())
        .map_err(|e| diag_err(&e, source))
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
pub fn compile(source: &str) -> Result<CompileResult, JsValue> {
    let module = build_module(source, None, None)?;
    let bytes = oqi_compile::bytecode::to_bytes(&module)
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
    Array(Vec<InputValue>),
}

impl From<InputValue> for HostValue {
    fn from(value: InputValue) -> Self {
        match value {
            InputValue::Bool(b) => HostValue::Bool(b),
            InputValue::Number(n) => HostValue::Float(n),
            InputValue::Text(s) => HostValue::Text(s),
            InputValue::Array(items) => {
                HostValue::List(items.into_iter().map(HostValue::from).collect())
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RunOptions {
    /// Values for the program's `input` declarations, keyed by name.
    #[serde(default)]
    inputs: BTreeMap<String, InputValue>,
    /// Simulator RNG seed (number or bigint).
    #[serde(default)]
    seed: Option<u64>,
    /// Include the final state vector in the result.
    #[serde(default)]
    statevector: bool,
    /// Extern implementations: an object mapping extern name → function.
    /// Captured raw — JS functions can't pass through serde.
    #[serde(default, with = "serde_wasm_bindgen::preserve")]
    externs: JsValue,
    /// Per-operation durations for compile-time `durationof` resolution,
    /// keyed by name (`measure`/`reset` reserved); values are timing
    /// literals ("50ns", "4dt"). Presence (even `{}`) enables the pass,
    /// which also derives durations from the program's defcal bodies.
    #[serde(default)]
    timings: Option<HashMap<String, String>>,
    /// Duration of one `dt` unit, as a timing literal (e.g. "0.5ns").
    #[serde(default)]
    dt: Option<String>,
    /// Which simulator to run on, by the same names as the CLI's
    /// `--backend`: `"scalar"` (default), `"simd"`, `"gpu"`, `"prefer-gpu"`
    /// (WebGPU when available, else `"scalar"`), `"auto"`, `"sum"` or
    /// `"mps"`.
    #[serde(default)]
    backend: Option<String>,
    /// Amplitude precision of the CPU state-vector simulators: `"f64"`
    /// (default) or `"f32"`. The GPU simulator is always `f32`.
    #[serde(default)]
    precision: Option<String>,
    /// Maximum stabilizer terms for the `"auto"`/`"sum"` backends' exact
    /// sum-over-Cliffords tier.
    #[serde(default)]
    max_rank: Option<usize>,
    /// Maximum bond dimension of the `"mps"` backend, past which its state is
    /// truncated.
    #[serde(default)]
    max_bond: Option<usize>,
    /// Number of shots for [`sample`] (ignored by [`run`]). Clamped to
    /// `1..=100_000`.
    #[serde(default = "default_shots")]
    shots: u32,
}

fn default_shots() -> u32 {
    1024
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
struct SampleOutput {
    shots: u32,
    /// The backend that actually ran (see [`RunOutput::backend`]).
    backend: &'static str,
    /// The amplitude precision that ran: `"f32"` or `"f64"`.
    precision: &'static str,
    /// One histogram per named output variable, in program order.
    histograms: Vec<Histogram>,
}

#[derive(Serialize)]
struct Histogram {
    name: String,
    total: u32,
    /// Distinct observed values and their counts, sorted by value.
    bars: Vec<HistoBar>,
}

#[derive(Serialize)]
struct HistoBar {
    label: String,
    count: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunOutput {
    measurements: Vec<Measurement>,
    outputs: Vec<OutputEntry>,
    /// The backend that actually ran, by name (differs from the requested
    /// one when `"prefer-gpu"` resolves to `"gpu"` or `"scalar"`).
    backend: &'static str,
    /// The amplitude precision that ran: `"f32"` or `"f64"`.
    precision: &'static str,
    /// Interleaved amplitudes `[re0, im0, re1, im1, ...]`, present iff requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    statevector: Option<Vec<f64>>,
}

/// Everything [`run`] and [`sample`] share: the parsed options, the compiled
/// module, and the pieces a [`Vm`] is assembled from.
struct Session {
    module: BcModule,
    inputs: Vec<(String, HostValue)>,
    externs: JsExterns,
    backend: Box<dyn QuantumBackend>,
    /// The backend configuration that was actually built.
    built: BackendSpec,
    seed: u64,
    statevector: bool,
    shots: u32,
}

impl Session {
    async fn new(source: &str, options: JsValue) -> Result<Session, JsValue> {
        let invalid = |e: String| JsError::new(&format!("invalid options: {e}"));
        // An absent options object still gets every field's default.
        let options = if options.is_undefined() || options.is_null() {
            js_sys::Object::new().into()
        } else {
            options
        };
        let opts: RunOptions =
            serde_wasm_bindgen::from_value(options).map_err(|e| invalid(e.to_string()))?;

        let module = build_module(source, opts.timings.as_ref(), opts.dt.as_deref())?;
        let externs = JsExterns::new(&module, &opts.externs)?;

        let seed = opts.seed.unwrap_or(DEFAULT_SEED);
        let mut spec: BackendSpec = opts
            .backend
            .as_deref()
            .unwrap_or("scalar")
            .parse()
            .map_err(invalid)?;
        if let Some(precision) = &opts.precision {
            spec.precision = precision.parse().map_err(invalid)?;
        }
        if let Some(max_rank) = opts.max_rank {
            spec.sum_policy = SumPolicy {
                max_rank,
                ..spec.sum_policy
            };
        }
        if let Some(max_bond) = opts.max_bond {
            spec.max_bond = max_bond;
        }
        spec.seed = seed;
        let (backend, built) = spec
            .build(module.qubits.num_qubits)
            .await
            .map_err(|e| diag_err(&e, source))?;

        Ok(Session {
            module,
            inputs: opts
                .inputs
                .into_iter()
                .map(|(name, value)| (name, value.into()))
                .collect(),
            externs,
            backend,
            built,
            seed,
            statevector: opts.statevector,
            shots: opts.shots,
        })
    }
}

fn to_js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| JsError::new(&format!("result serialization failed: {e}")).into())
}

/// Compile and run an OpenQASM 3 program on a simulator.
///
/// `options` is an optional object:
/// `{ inputs?, seed?, statevector?, externs?, timings?, dt?, backend?, precision?, maxRank?, maxBond? }`.
/// `externs` maps each `extern` function's name to a JS implementation, which
/// may return its result directly or as a `Promise` (awaited). Return values
/// are coerced to the declared return type (radians for `angle[n]`, MSB-first
/// "0101" strings for `bit[n]`). An extern with no implementation, or one
/// that throws, aborts the run with a diagnostic error. `timings` maps
/// operation names to duration literals and enables compile-time
/// `durationof` resolution (defcal bodies included); `dt` sets the device
/// time unit. Returns `{ measurements, outputs, backend, precision, statevector? }`.
#[wasm_bindgen]
pub async fn run(source: String, options: JsValue) -> Result<JsValue, JsValue> {
    let session = Session::new(&source, options).await?;
    let module = &session.module;
    let mut vm = Vm::new(module, session.backend, session.externs);
    let result = vm
        .run_with_named_inputs(&session.inputs)
        .await
        .map_err(|e| diag_err(&e, &source))?;

    let statevector = if session.statevector {
        vm.backend()
            .amplitudes()
            .await
            .map(|amps| amps.iter().flat_map(|c| [c.re, c.im]).collect())
    } else {
        None
    };

    to_js(&RunOutput {
        measurements: result
            .measurements
            .iter()
            .map(|&(qubit, value)| Measurement { qubit, value })
            .collect(),
        outputs: result
            .outputs
            .iter()
            .map(|(sym, v)| OutputEntry {
                name: module.symbol(*sym).name.clone(),
                value: js_value(HostValue::from(v), v),
            })
            .collect(),
        backend: session.built.name(),
        precision: session.built.precision.as_str(),
        statevector,
    })
}

/// Compile and sample an OpenQASM 3 program over many shots, returning a
/// histogram of observed values per named output variable.
///
/// `options` accepts the same object as [`run`] plus `shots` (default 1024,
/// clamped to `1..=100_000`). A given `seed` reproduces the whole histogram.
/// Returns `{ shots, backend, precision, histograms: [{ name, total, bars: [{ label, count }] }] }`.
#[wasm_bindgen]
pub async fn sample(source: String, options: JsValue) -> Result<JsValue, JsValue> {
    let session = Session::new(&source, options).await?;
    let shots = session.shots.clamp(1, 100_000);
    let module = &session.module;
    let mut vm = Vm::new(module, session.backend, session.externs);
    let inputs =
        oqi_vm::coerce_inputs(module, &session.inputs).map_err(|e| diag_err(&e, &source))?;
    let results = vm
        .sample(inputs, shots, session.seed)
        .await
        .map_err(|e| diag_err(&e, &source))?;

    to_js(&SampleOutput {
        shots,
        backend: session.built.name(),
        precision: session.built.precision.as_str(),
        histograms: oqi_vm::histograms(module, &results)
            .into_iter()
            .map(|h| Histogram {
                name: h.name,
                total: h.counts.iter().map(|(_, count)| count).sum(),
                bars: h
                    .counts
                    .into_iter()
                    .map(|(label, count)| HistoBar { label, count })
                    .collect(),
            })
            .collect(),
    })
}

/// Extern implementations supplied from JS (`run`'s `externs` option):
/// name → function, plus each declared extern's return type (`None` = void)
/// — the VM stores extern results uncast, so returns must be built with
/// their declared type.
struct JsExterns {
    fns: HashMap<String, js_sys::Function>,
    ret_tys: HashMap<String, Option<ValueTy>>,
}

impl JsExterns {
    /// `externs` is the raw option value (undefined/null ⇒ none). Errors if
    /// it isn't an object or any property isn't a function.
    fn new(module: &BcModule, externs: &JsValue) -> Result<Self, JsError> {
        let ret_tys = module
            .externs()
            .map(|(name, ty)| (name.to_string(), ty))
            .collect();
        let mut fns = HashMap::new();
        if !externs.is_undefined() && !externs.is_null() {
            let obj = externs.dyn_ref::<js_sys::Object>().ok_or_else(|| {
                JsError::new(
                    "invalid options: `externs` must be an object mapping extern names to functions",
                )
            })?;
            for entry in js_sys::Object::entries(obj).iter() {
                let entry: js_sys::Array = entry.unchecked_into();
                let name = entry.get(0).as_string().unwrap_or_default();
                let f = entry.get(1).dyn_into::<js_sys::Function>().map_err(|_| {
                    JsError::new(&format!(
                        "invalid options: externs.{name} is not a function"
                    ))
                })?;
                fns.insert(name, f);
            }
        }
        Ok(JsExterns { fns, ret_tys })
    }
}

impl ExternProvider for JsExterns {
    async fn call(&mut self, name: &str, args: &[Value]) -> KindResult<Option<Value>> {
        // Not provided, or not a declared classical extern (ports/frames land
        // here when no pulse handler is installed) — same error as `NoExterns`.
        let (Some(f), Some(ret_ty)) = (self.fns.get(name), self.ret_tys.get(name)) else {
            return Err(VmErrorKind::UnknownExtern(name.to_string()));
        };
        let failed = |message: String| VmErrorKind::extern_failed(name, message);
        let js_args = js_sys::Array::new();
        for a in args {
            js_args.push(&match js_value(HostValue::extern_arg(a), a) {
                OutputValue::Bool(b) => JsValue::from_bool(b),
                OutputValue::Number(n) => JsValue::from_f64(n),
                OutputValue::Text(s) => JsValue::from_str(&s),
            });
        }
        let ret = f
            .apply(&JsValue::UNDEFINED, &js_args)
            .map_err(|e| failed(js_error_message(&e)))?;
        // Settle Promise returns before the void check so rejections from
        // void externs still surface.
        let settled = match ret.dyn_into::<js_sys::Promise>() {
            Ok(p) => JsFuture::from(p)
                .await
                .map_err(|e| failed(js_error_message(&e)))?,
            Err(v) => v,
        };
        let Some(ty) = ret_ty else {
            return Ok(None); // void extern: JS return value ignored
        };
        let raw: InputValue = serde_wasm_bindgen::from_value(settled).map_err(|_| {
            failed("returned a value that is not a boolean, number, or string".to_string())
        })?;
        oqi_vm::coerce("return value", &raw.into(), *ty)
            .map(Some)
            .map_err(failed)
    }
}

/// Best-effort message from a thrown/rejected JS value.
fn js_error_message(v: &JsValue) -> String {
    if let Some(e) = v.dyn_ref::<js_sys::Error>() {
        return String::from(e.message());
    }
    v.as_string().unwrap_or_else(|| format!("{v:?}"))
}

/// A host value as JS sees it: booleans and floats natively, integers
/// natively while a JS `number` represents them exactly, and everything else
/// as text (for values with no host form, `value`'s OpenQASM text).
fn js_value(host: HostValue, value: &Value) -> OutputValue {
    match host {
        HostValue::Bool(b) => OutputValue::Bool(b),
        HostValue::Int(i) if i.unsigned_abs() <= MAX_SAFE_INTEGER => OutputValue::Number(i as f64),
        HostValue::Uint(u) if u <= MAX_SAFE_INTEGER => OutputValue::Number(u as f64),
        HostValue::Float(f) => OutputValue::Number(f),
        HostValue::Text(s) => OutputValue::Text(s),
        _ => OutputValue::Text(value.to_string()),
    }
}
