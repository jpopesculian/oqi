//! JavaScript (wasm-bindgen) bindings for the oqi compiler and VM.
//!
//! Exposes a minimal API: [`compile`] (source → bytecode + disassembly) and
//! [`run`] (source → simulated results). `run` can call host-supplied
//! `extern` implementations (the `externs` option), synchronous or
//! Promise-returning. Includes are limited to the embedded `stdgates.inc`;
//! file includes are rejected. Errors surface as thrown JS `Error`s whose
//! message is the rendered diagnostic.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use oqi_compile::bytecode::{self, BcModule};
use oqi_compile::classical::{
    Array, BitReg, Duration, FloatWidth, Primitive, PrimitiveTy, Scalar, Value, ValueTy,
};
use oqi_compile::duration::TableTimings;
use oqi_compile::resolve::IncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::types::CompileOptions;
use oqi_compile::{cfg, duration, qubits, ssa};
use oqi_vm::{ExternProvider, QuantumBackend, StateVectorSim, Vm, VmErrorKind};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

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
    if let Some(label) = diag.labels().into_iter().find(|l| l.primary) {
        let mut start = label.span.start;
        let mut end = label.span.end;
        // A `0..0` span is the "no location" sentinel; skip it.
        if (start != 0 || end != 0) && start <= source.len() && end <= source.len() {
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

/// Source → bytecode module; same pipeline the CLI drives in `run`.
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
        options.dt = parse_dt(spec)?;
    }
    let mut program =
        oqi_compile::lower::compile_source_with_options(source, LibOnlyResolver, options.clone())
            .map_err(|e| diag_err(&e, source))?;
    // With timings supplied (even an empty map), `durationof` is resolved
    // at compile time; otherwise it stays for the VM's runtime pass.
    if let Some(timings) = timings {
        let entries = timings.iter().map(|(k, v)| (k.as_str(), v.as_str()));
        let table = TableTimings::from_str_entries(entries, &options.dt)
            .map_err(|e| diag_err(&e, source))?
            .with_defcals(&program, &options.dt)
            .with_program_gates(&program);
        duration::resolve_durationof(&mut program, &table, &options)
            .map_err(|e| diag_err(&e, source))?;
    }
    let cfgs = cfg::build_program(&program).map_err(|e| diag_err(&e, source))?;
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    bytecode::emit(&ssa, &program, layout).map_err(|e| diag_err(&e, source))
}

/// Parse the `dt` option ("0.5ns"): timing-literal syntax, except the `dt`
/// unit itself is rejected (self-referential) and the value must be
/// positive.
fn parse_dt(spec: &str) -> Result<Duration, JsError> {
    if spec.trim_end().ends_with("dt") {
        return Err(JsError::new(
            "invalid options: `dt` cannot itself be given in `dt` units",
        ));
    }
    let d = oqi_compile::types::parse_timing_literal(spec, &CompileOptions::default().dt).map_err(
        |_| {
            JsError::new(&format!(
                "invalid options: `{spec}` is not a duration literal (e.g. \"0.5ns\")"
            ))
        },
    )?;
    if d.value <= 0.0 {
        return Err(JsError::new(
            "invalid options: `dt` must be a positive duration",
        ));
    }
    Ok(d)
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
    Array(Vec<InputValue>),
}

/// Which simulator backend to run on.
#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    /// The CPU state-vector simulator (default; deterministic). Its amplitude
    /// precision is set by [`RunOptions::precision`].
    #[default]
    Cpu,
    /// The `f32` WebGPU (`wgpu`) simulator; errors if unavailable.
    Gpu,
    /// Prefer WebGPU, fall back to the CPU simulator when it isn't available.
    Auto,
}

/// Amplitude precision for the CPU simulator (the GPU backend is always `f32`).
#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Precision {
    /// Double precision (default).
    #[default]
    F64,
    /// Single precision.
    F32,
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
    /// Which simulator backend to run on: `"cpu"` (default), `"gpu"`, or
    /// `"auto"` (WebGPU when available, else CPU).
    #[serde(default)]
    backend: Backend,
    /// CPU amplitude precision: `"f64"` (default) or `"f32"`. Ignored by the
    /// GPU backend, which is always `f32`.
    #[serde(default)]
    precision: Precision,
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
    /// The backend that actually ran: `"cpu"` or `"gpu"`.
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
    /// The backend that actually ran: `"cpu"` or `"gpu"` (differs from the
    /// requested backend when `"auto"` falls back to the CPU simulator).
    backend: &'static str,
    /// The amplitude precision that ran: `"f32"` or `"f64"`.
    precision: &'static str,
    /// Interleaved amplitudes `[re0, im0, re1, im1, ...]`, present iff requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    statevector: Option<Vec<f64>>,
}

/// Compile and run an OpenQASM 3 program on the state-vector simulator.
///
/// `options` is an optional object:
/// `{ inputs?, seed?, statevector?, externs?, timings?, dt? }`.
/// `externs` maps each `extern` function's name to a JS implementation, which
/// may return its result directly or as a `Promise` (awaited). Return values
/// are coerced to the declared return type (radians for `angle[n]`, MSB-first
/// "0101" strings for `bit[n]`). An extern with no implementation, or one
/// that throws, aborts the run with a diagnostic error. `timings` maps
/// operation names to duration literals and enables compile-time
/// `durationof` resolution (defcal bodies included); `dt` sets the device
/// time unit. Returns `{ measurements, outputs, statevector? }`.
#[wasm_bindgen]
pub async fn run(source: String, options: JsValue) -> Result<JsValue, JsValue> {
    let opts: RunOptions = if options.is_undefined() || options.is_null() {
        RunOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("invalid options: {e}")))?
    };
    let module = build_module(&source, opts.timings.as_ref(), opts.dt.as_deref())?;
    let inputs = coerce_inputs(&module, opts.inputs)?;
    let externs = JsExterns::new(&module, &opts.externs)?;

    let seed = opts.seed.unwrap_or(DEFAULT_SEED);
    let Chosen {
        backend,
        name: used,
        precision,
    } = select_backend(
        opts.backend,
        opts.precision,
        module.qubits.num_qubits,
        seed,
        &source,
    )
    .await?;
    let mut vm = Vm::new(&module, backend, externs);
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
        backend: used,
        precision,
        statevector,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsError::new(&format!("result serialization failed: {e}")).into())
}

/// A backend chosen for a run, boxed for the VM, plus the names of what
/// actually ran (`Auto` may fall back to CPU; `precision` reflects the real
/// amplitude type — always `"f32"` for GPU).
struct Chosen {
    backend: Box<dyn QuantumBackend>,
    name: &'static str,
    precision: &'static str,
}

/// Build the requested [`Backend`] at the requested CPU [`Precision`].
async fn select_backend(
    choice: Backend,
    precision: Precision,
    num_qubits: u32,
    seed: u64,
    source: &str,
) -> Result<Chosen, JsValue> {
    fn cpu(
        precision: Precision,
        num_qubits: u32,
        seed: u64,
        source: &str,
    ) -> Result<Chosen, JsValue> {
        let (backend, precision): (Box<dyn QuantumBackend>, &'static str) = match precision {
            Precision::F64 => (
                Box::new(
                    StateVectorSim::<f64>::try_zeroed(num_qubits, seed)
                        .map_err(|e| diag_err(&e, source))?,
                ),
                "f64",
            ),
            Precision::F32 => (
                Box::new(
                    StateVectorSim::<f32>::try_zeroed(num_qubits, seed)
                        .map_err(|e| diag_err(&e, source))?,
                ),
                "f32",
            ),
        };
        Ok(Chosen {
            backend,
            name: "cpu",
            precision,
        })
    }
    match choice {
        Backend::Cpu => cpu(precision, num_qubits, seed, source),
        #[cfg(feature = "gpu")]
        Backend::Gpu => {
            let gpu = oqi_vm::GpuSim::new(num_qubits)
                .await
                .map_err(|e| JsError::new(&format!("GPU backend unavailable: {e}")))?
                .with_seed(seed);
            Ok(Chosen {
                backend: Box::new(gpu),
                name: "gpu",
                precision: "f32",
            })
        }
        #[cfg(feature = "gpu")]
        Backend::Auto => match oqi_vm::GpuSim::new(num_qubits).await {
            Ok(gpu) => Ok(Chosen {
                backend: Box::new(gpu.with_seed(seed)),
                name: "gpu",
                precision: "f32",
            }),
            Err(_) => cpu(precision, num_qubits, seed, source),
        },
        #[cfg(not(feature = "gpu"))]
        Backend::Gpu => Err(JsError::new(
            "this build has no GPU backend (compile oqi-js with `--features gpu`)",
        )
        .into()),
        #[cfg(not(feature = "gpu"))]
        Backend::Auto => cpu(precision, num_qubits, seed, source),
    }
}

/// Compile and sample an OpenQASM 3 program over many shots, returning a
/// histogram of observed values per named output variable.
///
/// `options` accepts the same object as [`run`] plus `shots` (default 1024,
/// clamped to `1..=100_000`). Shots reuse one backend (respecting the
/// `backend` option, WebGPU included), re-zeroing between runs while a single
/// RNG stream advances — so a given `seed` reproduces the whole histogram.
/// Returns `{ shots, backend, histograms: [{ name, total, bars: [{ label, count }] }] }`.
#[wasm_bindgen]
pub async fn sample(source: String, options: JsValue) -> Result<JsValue, JsValue> {
    let opts: RunOptions = if options.is_undefined() || options.is_null() {
        RunOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("invalid options: {e}")))?
    };
    let shots = opts.shots.clamp(1, 100_000);
    let module = build_module(&source, opts.timings.as_ref(), opts.dt.as_deref())?;
    let inputs = coerce_inputs(&module, opts.inputs)?;
    let externs = JsExterns::new(&module, &opts.externs)?;

    let n = module.qubits.num_qubits;
    let seed = opts.seed.unwrap_or(DEFAULT_SEED);
    let Chosen {
        backend,
        name: used,
        precision,
    } = select_backend(opts.backend, opts.precision, n, seed, &source).await?;
    let mut vm = Vm::new(&module, backend, externs);

    // Accumulate per-output-symbol value counts; `order` preserves the
    // program's output order (fixed across shots) for stable display.
    let mut order: Vec<SymbolId> = Vec::new();
    let mut counts: HashMap<SymbolId, HashMap<String, u32>> = HashMap::new();

    // Fast path: for a program with no measurement feedback, snapshot the final
    // state once and draw every shot from its Born-rule distribution — one
    // backend readback total instead of one per shot.
    let mut sampled = false;
    if bytecode::is_sample_safe(&module)
        && let Some(psi) = vm
            .run_capture(inputs.clone())
            .await
            .map_err(|e| diag_err(&e, &source))?
    {
        // Cumulative |ψ|² table over basis states.
        let mut cum = Vec::with_capacity(psi.len());
        let mut acc = 0.0f64;
        for a in &psi {
            acc += a.norm_sqr();
            cum.push(acc);
        }
        let total = acc.max(f64::MIN_POSITIVE);
        let mut rng = SplitMix64::new(seed);
        for shot in 0..shots {
            let u = rng.next_f64() * total;
            let idx = cum.partition_point(|&c| c < u).min(psi.len() - 1);
            let basis: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
            let result = vm
                .run_inject(inputs.clone(), basis)
                .await
                .map_err(|e| diag_err(&e, &source))?;
            record_shot(&mut order, &mut counts, &result.outputs, shot == 0);
        }
        sampled = true;
    }

    // Fallback: re-run the whole circuit per shot (mid-circuit measurement /
    // feedback), reusing one backend re-zeroed between shots.
    if !sampled {
        for shot in 0..shots {
            if shot > 0 {
                vm.backend_mut().reset_state(n).await;
            }
            let result = vm
                .run_with_inputs(inputs.clone())
                .await
                .map_err(|e| diag_err(&e, &source))?;
            record_shot(&mut order, &mut counts, &result.outputs, shot == 0);
        }
    }

    let histograms = order
        .iter()
        .map(|sym| {
            let map = &counts[sym];
            let mut bars: Vec<HistoBar> = map
                .iter()
                .map(|(label, count)| HistoBar {
                    label: label.clone(),
                    count: *count,
                })
                .collect();
            sort_bars(&mut bars);
            Histogram {
                name: module.symbols.get(*sym).name.clone(),
                total: bars.iter().map(|b| b.count).sum(),
                bars,
            }
        })
        .collect();

    let out = SampleOutput {
        shots,
        backend: used,
        precision,
        histograms,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsError::new(&format!("result serialization failed: {e}")).into())
}

/// Tally one shot's outputs into the running histograms; on the first shot,
/// record each output symbol's order for stable display.
fn record_shot(
    order: &mut Vec<SymbolId>,
    counts: &mut HashMap<SymbolId, HashMap<String, u32>>,
    outputs: &[(SymbolId, Value)],
    first: bool,
) {
    for (sym, v) in outputs {
        if first {
            order.push(*sym);
        }
        *counts
            .entry(*sym)
            .or_default()
            .entry(value_label(v))
            .or_insert(0) += 1;
    }
}

/// A small splitmix64 PRNG for host-side basis-state sampling on the fast path,
/// seeded from the run's `seed` so a program+seed reproduces its histogram.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A double in `[0, 1)` (53-bit mantissa).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sort histogram bars by value: numerically for plain decimal integers
/// (bits, ints), lexicographically otherwise — so fixed-width bit strings
/// like `"010"` keep their binary order instead of being read as decimals.
fn sort_bars(bars: &mut [HistoBar]) {
    let numeric = bars.iter().all(|b| {
        let s = b.label.strip_prefix('-').unwrap_or(&b.label);
        s.parse::<i128>().is_ok() && (s.len() == 1 || !s.starts_with('0'))
    });
    if numeric {
        bars.sort_by_key(|b| b.label.parse::<i128>().unwrap());
    } else {
        bars.sort_by(|a, b| a.label.cmp(&b.label));
    }
}

/// Coerce JS input values to the declared types of the program's `input`
/// symbols. A superset of `parse_inputs` in `cli/src/main.rs` (which handles
/// int/uint/float/bit only); keep the shared types in sync.
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
        let value =
            coerce_value(&format!("input `{name}`"), &input, ty).map_err(|m| JsError::new(&m))?;
        map.insert(sym.id, value);
    }
    Ok(map)
}

/// Coerce a raw JS value to the declared type `ty`. Shared by
/// [`coerce_inputs`] and extern return values; `what` names the value in
/// error messages. Angles are radians (wrapped mod 2π and quantized to the
/// declared width); bit registers are MSB-first "0101" strings of the
/// declared width. Array types accept a JSON array (nesting allowed and
/// flattened row-major) whose elements coerce to the element type.
fn coerce_value(what: &str, raw: &InputValue, ty: ValueTy) -> Result<Value, String> {
    Ok(match ty {
        ValueTy::Scalar(PrimitiveTy::Int(w)) => Value::int(raw.to_i128(what)?, w),
        ValueTy::Scalar(PrimitiveTy::Uint(w)) => Value::uint(raw.to_u128(what)?, w),
        ValueTy::Scalar(PrimitiveTy::Float(w)) => Value::float(raw.to_f64(what)?, w),
        ValueTy::Scalar(PrimitiveTy::Bit) | ValueTy::Scalar(PrimitiveTy::Bool) => {
            Value::bit(raw.to_bool(what)?)
        }
        ValueTy::Scalar(PrimitiveTy::Angle(w)) => {
            Scalar::new(Primitive::float(raw.to_f64(what)?), PrimitiveTy::Angle(w))
                .map_err(|e| format!("{what}: not representable as `{ty}`: {e:?}"))?
                .into()
        }
        ValueTy::Scalar(PrimitiveTy::BitReg(w)) => Value::bitreg(raw.to_bitreg(what, w)?, w),
        ValueTy::Array(aty) => {
            if !matches!(raw, InputValue::Array(_)) {
                return Err(format!("{what} must be a JSON array of `{}` values", aty.ty()));
            }
            let mut leaves: Vec<&InputValue> = Vec::new();
            collect_leaves(raw, &mut leaves);
            let elem_ty = aty.ty();
            let elems = leaves
                .into_iter()
                .enumerate()
                .map(|(i, leaf)| {
                    match coerce_value(&format!("{what}[{i}]"), leaf, ValueTy::Scalar(elem_ty))? {
                        Value::Scalar(s) => Ok(s.into_value()),
                        _ => unreachable!("a scalar type coerces to a scalar value"),
                    }
                })
                .collect::<Result<Vec<Primitive>, String>>()?;
            // `Array::new` validates the element count against the shape.
            Value::Array(Array::new(elems, aty).map_err(|e| format!("{what}: {e:?}"))?)
        }
        _ => return Err(format!("{what} has type `{ty}`, unsupported by the JS API")),
    })
}

/// Flatten a nested JS array into its scalar leaves, row-major, so both
/// `[1,2,3]` and `[[1,2],[3,4]]` feed a multi-dimensional `array[...]`.
fn collect_leaves<'a>(raw: &'a InputValue, out: &mut Vec<&'a InputValue>) {
    match raw {
        InputValue::Array(items) => {
            for item in items {
                collect_leaves(item, out);
            }
        }
        scalar => out.push(scalar),
    }
}

impl InputValue {
    fn to_i128(&self, what: &str) -> Result<i128, String> {
        match self {
            InputValue::Number(n) if n.fract() == 0.0 && n.abs() <= MAX_SAFE_INTEGER as f64 => {
                Ok(*n as i128)
            }
            InputValue::Text(s) => s
                .parse()
                .map_err(|_| format!("{what}: `{s}` is not an integer")),
            _ => Err(format!("{what} must be a safe integer or a decimal string")),
        }
    }

    fn to_u128(&self, what: &str) -> Result<u128, String> {
        match self {
            InputValue::Number(n)
                if n.fract() == 0.0 && *n >= 0.0 && *n as u128 <= MAX_SAFE_INTEGER =>
            {
                Ok(*n as u128)
            }
            InputValue::Text(s) => s
                .parse()
                .map_err(|_| format!("{what}: `{s}` is not an unsigned integer")),
            _ => Err(format!(
                "{what} must be a non-negative safe integer or a decimal string"
            )),
        }
    }

    fn to_f64(&self, what: &str) -> Result<f64, String> {
        match self {
            InputValue::Number(n) => Ok(*n),
            InputValue::Text(s) => s
                .parse()
                .map_err(|_| format!("{what}: `{s}` is not a float")),
            InputValue::Bool(_) | InputValue::Array(_) => {
                Err(format!("{what} must be a number or a decimal string"))
            }
        }
    }

    fn to_bool(&self, what: &str) -> Result<bool, String> {
        match self {
            InputValue::Bool(b) => Ok(*b),
            InputValue::Number(n) if *n == 0.0 => Ok(false),
            InputValue::Number(n) if *n == 1.0 => Ok(true),
            InputValue::Text(s) if s == "0" || s == "false" => Ok(false),
            InputValue::Text(s) if s == "1" || s == "true" => Ok(true),
            _ => Err(format!("{what} is not a bit (use 0/1/true/false)")),
        }
    }

    /// "0101"-style string, MSB first, exactly `width` chars of 0/1.
    fn to_bitreg(&self, what: &str, width: u32) -> Result<BitReg, String> {
        let err = || format!("{what} must be a {width}-character string of 0s and 1s");
        let InputValue::Text(s) = self else {
            return Err(err());
        };
        if s.len() != width as usize || !s.bytes().all(|b| matches!(b, b'0' | b'1')) {
            return Err(err());
        }
        let mut reg = BitReg::zeros(width);
        for (i, b) in s.bytes().enumerate() {
            reg.set_bit(width as usize - 1 - i, b == b'1');
        }
        Ok(reg)
    }
}

/// Extern implementations supplied from JS (`run`'s `externs` option):
/// name → function, plus each declared extern's return type (`None` = void),
/// harvested from the module's symbols — the VM stores extern results
/// uncast, so returns must be built with their declared type.
struct JsExterns {
    fns: HashMap<String, js_sys::Function>,
    ret_tys: HashMap<String, Option<ValueTy>>,
}

impl JsExterns {
    /// `externs` is the raw option value (undefined/null ⇒ none). Errors if
    /// it isn't an object or any property isn't a function.
    fn new(module: &BcModule, externs: &JsValue) -> Result<Self, JsError> {
        let ret_tys = module
            .symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Extern)
            .map(|s| (s.name.clone(), s.ty.value_ty()))
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

#[async_trait(?Send)]
impl ExternProvider for JsExterns {
    async fn call(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> std::result::Result<Option<Value>, VmErrorKind> {
        // Not provided, or not a declared classical extern (ports/frames land
        // here when no pulse handler is installed) — same error as `NoExterns`.
        let (Some(f), Some(ret_ty)) = (self.fns.get(name), self.ret_tys.get(name)) else {
            return Err(VmErrorKind::UnknownExtern(name.to_string()));
        };
        let js_args = js_sys::Array::new();
        for a in args {
            js_args.push(&extern_arg(a));
        }
        let ret = f
            .apply(&JsValue::UNDEFINED, &js_args)
            .map_err(|e| extern_err(name, js_error_message(&e)))?;
        // Settle Promise returns before the void check so rejections from
        // void externs still surface.
        let settled = match ret.dyn_into::<js_sys::Promise>() {
            Ok(p) => JsFuture::from(p)
                .await
                .map_err(|e| extern_err(name, js_error_message(&e)))?,
            Err(v) => v,
        };
        let Some(ty) = ret_ty else {
            return Ok(None); // void extern: JS return value ignored
        };
        let raw: InputValue = serde_wasm_bindgen::from_value(settled).map_err(|_| {
            extern_err(
                name,
                "returned a value that is not a boolean, number, or string",
            )
        })?;
        coerce_value("return value", &raw, *ty)
            .map(Some)
            .map_err(|m| extern_err(name, m))
    }
}

fn extern_err(name: &str, message: impl Into<String>) -> VmErrorKind {
    VmErrorKind::Extern {
        name: name.to_string(),
        message: message.into(),
    }
}

/// Best-effort message from a thrown/rejected JS value.
fn js_error_message(v: &JsValue) -> String {
    if let Some(e) = v.dyn_ref::<js_sys::Error>() {
        return String::from(e.message());
    }
    v.as_string().unwrap_or_else(|| format!("{v:?}"))
}

/// Convert an extern-call argument for a JS callback. Like [`output_value`],
/// except `bit[n]` becomes an unquoted MSB-first "0101" string and `angle` a
/// radians number, so values round-trip with the extern return formats.
fn extern_arg(v: &Value) -> JsValue {
    if let Value::Scalar(s) = v {
        match s.value() {
            Primitive::Bit(b) => return JsValue::from_bool(*b),
            Primitive::Int(i) if i.unsigned_abs() <= MAX_SAFE_INTEGER => {
                return JsValue::from_f64(*i as f64);
            }
            Primitive::Uint(u) if *u <= MAX_SAFE_INTEGER => {
                return JsValue::from_f64(*u as f64);
            }
            Primitive::Float(f) => return JsValue::from_f64(*f),
            Primitive::Angle(_) => {
                if let Ok(f) = s.clone().cast(PrimitiveTy::Float(FloatWidth::F64))
                    && let Primitive::Float(radians) = f.value()
                {
                    return JsValue::from_f64(*radians);
                }
            }
            Primitive::BitReg(reg) => {
                if let PrimitiveTy::BitReg(w) = s.ty() {
                    return JsValue::from_str(&bitreg_string(reg, w));
                }
            }
            _ => {}
        }
    }
    JsValue::from_str(&v.to_string())
}

/// Canonical string label for a value, used as a histogram bucket key:
/// `bit`→`"0"`/`"1"`, `int`/`uint`/`float`→decimal, `bit[n]`→unquoted MSB-first
/// bit string, everything else its OpenQASM text form with surrounding quotes
/// stripped. Unlike [`output_value`], labels are always clean unquoted strings.
fn value_label(v: &Value) -> String {
    if let Value::Scalar(s) = v {
        match s.value() {
            Primitive::Bit(b) => return (if *b { "1" } else { "0" }).to_string(),
            Primitive::Int(i) if i.unsigned_abs() <= MAX_SAFE_INTEGER => return i.to_string(),
            Primitive::Uint(u) if *u <= MAX_SAFE_INTEGER => return u.to_string(),
            Primitive::Float(f) => return f.to_string(),
            Primitive::BitReg(reg) => {
                if let PrimitiveTy::BitReg(w) = s.ty() {
                    return bitreg_string(reg, w);
                }
            }
            _ => {}
        }
    }
    v.to_string().trim_matches('"').to_string()
}

/// Unquoted MSB-first bit string of `reg`'s low `width` bits.
fn bitreg_string(reg: &BitReg, width: u32) -> String {
    (0..width as usize)
        .rev()
        .map(|i| if reg.get_bit(i) { '1' } else { '0' })
        .collect()
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
