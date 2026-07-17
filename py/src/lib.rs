//! Python (PyO3) bindings for the oqi compiler and VM.
//!
//! Exposes a minimal API: [`compile`] (source → bytecode + disassembly) and
//! [`run`] (source → simulated results). `run` can call host-supplied
//! `extern` implementations (the `externs` option), as synchronous callables.
//! Includes are limited to the embedded `stdgates.inc`; file includes are
//! rejected. Errors raise [`OqiError`] whose message is the rendered
//! diagnostic.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use num_complex::Complex;
use oqi_compile::bytecode::{self, BcModule};
use oqi_compile::classical::{BitReg, FloatWidth, Primitive, PrimitiveTy, Scalar, Value, ValueTy};
use oqi_compile::resolve::IncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::{cfg, qubits, ssa};
use oqi_vm::{ExternProvider, QuantumBackend, StateVectorSim, Vm, VmErrorKind};
use pyo3::IntoPyObjectExt;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyTuple};

/// Name used for the root source in rendered diagnostics.
const SOURCE_NAME: &str = "<source>";

/// Default RNG seed (matches the CLI's `DEFAULT_SEED` / `StateVectorSim::new`,
/// so CLI, JS, and Python runs of the same program and seed agree).
const DEFAULT_SEED: u64 = 0x2545F4914F6CDD1D;

/// Largest integer a `float` represents exactly (2^53 - 1).
const MAX_SAFE_INTEGER: f64 = ((1u64 << 53) - 1) as f64;

create_exception!(
    oqi,
    OqiError,
    PyException,
    "Compile or runtime error; the message is the rendered diagnostic."
);

/// Serves only embedded libraries (`stdgates.inc`); file includes error.
struct LibOnlyResolver;

impl IncludeResolver for LibOnlyResolver {
    fn resolve_path(
        &self,
        path: &Path,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Err(format!(
            "cannot include \"{}\": file includes are not supported in the Python build",
            path.display()
        )
        .into())
    }
}

/// Render any workspace diagnostic against the root source as an `OqiError`.
fn diag_err(diag: &dyn oqi_diagnostics::Diagnostic, source: &str) -> PyErr {
    OqiError::new_err(oqi_diagnostics::render_to_string(
        diag,
        Path::new(SOURCE_NAME),
        source,
    ))
}

/// Source → bytecode module; same pipeline the CLI drives in `run`.
fn build_module(source: &str) -> PyResult<BcModule> {
    let program =
        oqi_compile::lower::compile_source(source, LibOnlyResolver, Some(Path::new(SOURCE_NAME)))
            .map_err(|e| diag_err(&e, source))?;
    let cfgs = cfg::build_program(&program).map_err(|e| diag_err(&e, source))?;
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    bytecode::emit(&ssa, &program, layout).map_err(|e| diag_err(&e, source))
}

/// Result of [`compile`]: postcard-encoded bytecode plus its disassembly.
#[pyclass(frozen, get_all, module = "oqi")]
struct CompileResult {
    /// Postcard-encoded bytecode module.
    bytecode: Vec<u8>,
    /// Textual disassembly of the module.
    disassembly: String,
}

#[pymethods]
impl CompileResult {
    fn __repr__(&self) -> String {
        format!("CompileResult(bytecode=<{} bytes>)", self.bytecode.len())
    }
}

/// Compile an OpenQASM 3 program to bytecode.
#[pyfunction]
fn compile(source: &str) -> PyResult<CompileResult> {
    let module = build_module(source)?;
    let bytes = bytecode::to_bytes(&module)
        .map_err(|e| OqiError::new_err(format!("bytecode encoding failed: {e:?}")))?;
    Ok(CompileResult {
        bytecode: bytes,
        disassembly: module.to_string(),
    })
}

/// Result of [`run`].
#[pyclass(frozen, get_all, module = "oqi")]
struct RunResult {
    /// `(global qubit index, measured bit)` pairs in program order.
    measurements: Vec<(u32, bool)>,
    /// Program `output` values, keyed by name.
    outputs: Py<PyDict>,
    /// Final state vector, present iff requested.
    statevector: Option<Vec<Complex<f64>>>,
}

#[pymethods]
impl RunResult {
    fn __repr__(&self) -> String {
        format!(
            "RunResult(measurements={:?}, outputs=<{} entries>)",
            self.measurements,
            Python::attach(|py| self.outputs.bind(py).len()),
        )
    }
}

/// Compile and run an OpenQASM 3 program on the state-vector simulator.
///
/// `externs` maps each `extern` function's name to a synchronous Python
/// callable. Return values are coerced to the declared return type (radians
/// for `angle[n]`, MSB-first "0101" strings for `bit[n]`). An extern with no
/// implementation, or one that raises, aborts the run with a diagnostic
/// error; a callable returning an awaitable is rejected.
#[pyfunction]
#[pyo3(signature = (source, *, inputs=None, seed=None, statevector=false, externs=None))]
fn run(
    py: Python<'_>,
    source: &str,
    inputs: Option<&Bound<'_, PyDict>>,
    seed: Option<u64>,
    statevector: bool,
    externs: Option<&Bound<'_, PyDict>>,
) -> PyResult<RunResult> {
    let module = build_module(source)?;
    let input_map = coerce_inputs(&module, inputs)?;
    let externs = PyExterns::new(&module, externs)?;
    let sim =
        StateVectorSim::<f64>::try_zeroed(module.qubits.num_qubits, seed.unwrap_or(DEFAULT_SEED))
            .map_err(|e| diag_err(&e, source))?;
    let mut vm = Vm::new(&module, sim, externs);

    // The VM futures resolve immediately (async-trait over a synchronous
    // simulator), so blocking here never parks the thread. Runs hold the GIL:
    // `oqi_vm::RunResult` is not `Send`, which rules out `Python::detach`.
    let result =
        pollster::block_on(vm.run_with_inputs(input_map)).map_err(|e| diag_err(&e, source))?;
    let amps = if statevector {
        pollster::block_on(vm.backend().amplitudes())
    } else {
        None
    };

    let outputs = PyDict::new(py);
    for (sym, v) in &result.outputs {
        outputs.set_item(module.symbols.get(*sym).name.as_str(), output_value(py, v)?)?;
    }
    Ok(RunResult {
        measurements: result.measurements,
        outputs: outputs.into(),
        statevector: amps,
    })
}

/// Coerce Python input values to the declared types of the program's `input`
/// symbols. A superset of `parse_inputs` in `cli/src/main.rs` (which handles
/// int/uint/float/bit only); mirrors `coerce_inputs` in `js/src/lib.rs`;
/// keep the three in sync.
fn coerce_inputs(
    module: &BcModule,
    raw: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<SymbolId, Value>> {
    let mut map = HashMap::new();
    let Some(raw) = raw else { return Ok(map) };
    for (key, val) in raw.iter() {
        let name: String = key
            .extract()
            .map_err(|_| OqiError::new_err("input names must be strings"))?;
        let sym = module
            .symbols
            .iter()
            .find(|s| s.kind == SymbolKind::Input && s.name == name)
            .ok_or_else(|| OqiError::new_err(format!("no input named `{name}`")))?;
        let ty = sym
            .ty
            .value_ty()
            .ok_or_else(|| OqiError::new_err(format!("input `{name}` has no value type")))?;
        let value =
            coerce_scalar(&format!("input `{name}`"), &val, ty).map_err(OqiError::new_err)?;
        map.insert(sym.id, value);
    }
    Ok(map)
}

/// Coerce a raw Python value to the declared scalar type `ty`. Shared by
/// [`coerce_inputs`] and extern return values; `what` names the value in
/// error messages. Angles are radians (wrapped mod 2π and quantized to the
/// declared width); bit registers are MSB-first "0101" strings of the
/// declared width.
fn coerce_scalar(what: &str, v: &Bound<'_, PyAny>, ty: ValueTy) -> Result<Value, String> {
    Ok(match ty {
        ValueTy::Scalar(PrimitiveTy::Int(w)) => Value::int(to_i128(what, v)?, w),
        ValueTy::Scalar(PrimitiveTy::Uint(w)) => Value::uint(to_u128(what, v)?, w),
        ValueTy::Scalar(PrimitiveTy::Float(w)) => Value::float(to_f64(what, v)?, w),
        ValueTy::Scalar(PrimitiveTy::Bit) | ValueTy::Scalar(PrimitiveTy::Bool) => {
            Value::bit(to_bool(what, v)?)
        }
        ValueTy::Scalar(PrimitiveTy::Angle(w)) => {
            Scalar::new(Primitive::float(to_f64(what, v)?), PrimitiveTy::Angle(w))
                .map_err(|e| format!("{what}: not representable as `{ty}`: {e:?}"))?
                .into()
        }
        ValueTy::Scalar(PrimitiveTy::BitReg(w)) => Value::bitreg(to_bitreg(what, v, w)?, w),
        _ => {
            return Err(format!(
                "{what} has type `{ty}`, unsupported by the Python API"
            ));
        }
    })
}

fn to_i128(what: &str, v: &Bound<'_, PyAny>) -> Result<i128, String> {
    if !v.is_instance_of::<PyBool>()
        && let Ok(i) = v.extract::<i128>()
    {
        return Ok(i);
    }
    if let Ok(s) = v.extract::<String>() {
        return s
            .parse()
            .map_err(|_| format!("{what}: `{s}` is not an integer"));
    }
    if let Ok(f) = v.extract::<f64>()
        && f.fract() == 0.0
        && f.abs() <= MAX_SAFE_INTEGER
    {
        return Ok(f as i128);
    }
    Err(format!("{what} must be an int or a decimal string"))
}

fn to_u128(what: &str, v: &Bound<'_, PyAny>) -> Result<u128, String> {
    if !v.is_instance_of::<PyBool>()
        && let Ok(u) = v.extract::<u128>()
    {
        return Ok(u);
    }
    if let Ok(s) = v.extract::<String>() {
        return s
            .parse()
            .map_err(|_| format!("{what}: `{s}` is not an unsigned integer"));
    }
    if let Ok(f) = v.extract::<f64>()
        && f.fract() == 0.0
        && (0.0..=MAX_SAFE_INTEGER).contains(&f)
    {
        return Ok(f as u128);
    }
    Err(format!(
        "{what} must be a non-negative int or a decimal string"
    ))
}

fn to_f64(what: &str, v: &Bound<'_, PyAny>) -> Result<f64, String> {
    if !v.is_instance_of::<PyBool>()
        && let Ok(f) = v.extract::<f64>()
    {
        return Ok(f);
    }
    if let Ok(s) = v.extract::<String>() {
        return s
            .parse()
            .map_err(|_| format!("{what}: `{s}` is not a float"));
    }
    Err(format!("{what} must be a number or a decimal string"))
}

fn to_bool(what: &str, v: &Bound<'_, PyAny>) -> Result<bool, String> {
    if v.is_instance_of::<PyBool>() {
        return v.extract::<bool>().map_err(|e| e.to_string());
    }
    if let Ok(i) = v.extract::<i64>() {
        match i {
            0 => return Ok(false),
            1 => return Ok(true),
            _ => {}
        }
    }
    if let Ok(s) = v.extract::<String>() {
        match s.as_str() {
            "0" | "false" => return Ok(false),
            "1" | "true" => return Ok(true),
            _ => {}
        }
    }
    Err(format!("{what} is not a bit (use 0/1/True/False)"))
}

/// "0101"-style string, MSB first, exactly `width` chars of 0/1.
fn to_bitreg(what: &str, v: &Bound<'_, PyAny>, width: u32) -> Result<BitReg, String> {
    let err = || format!("{what} must be a {width}-character string of 0s and 1s");
    let Ok(s) = v.extract::<String>() else {
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

/// Extern implementations supplied from Python (`run`'s `externs` option):
/// name → callable, plus each declared extern's return type (`None` = void),
/// harvested from the module's symbols — the VM stores extern results
/// uncast, so returns must be built with their declared type.
struct PyExterns {
    fns: HashMap<String, Py<PyAny>>,
    ret_tys: HashMap<String, Option<ValueTy>>,
}

impl PyExterns {
    /// `externs` is the `externs=` option (`None` ⇒ none). Errors if any
    /// key isn't a string or any value isn't callable.
    fn new(module: &BcModule, externs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let ret_tys = module
            .symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Extern)
            .map(|s| (s.name.clone(), s.ty.value_ty()))
            .collect();
        let mut fns = HashMap::new();
        if let Some(externs) = externs {
            for (key, val) in externs.iter() {
                let name: String = key
                    .extract()
                    .map_err(|_| OqiError::new_err("extern names must be strings"))?;
                if !val.is_callable() {
                    return Err(OqiError::new_err(format!(
                        "externs value for `{name}` is not callable"
                    )));
                }
                fns.insert(name, val.unbind());
            }
        }
        Ok(PyExterns { fns, ret_tys })
    }
}

#[async_trait(?Send)]
impl ExternProvider for PyExterns {
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
        Python::attach(|py| {
            let py_args = args
                .iter()
                .map(|a| extern_arg(py, a))
                .collect::<PyResult<Vec<_>>>()
                .and_then(|a| PyTuple::new(py, a))
                .map_err(|e| extern_err(name, e.to_string()))?;
            let ret = f
                .bind(py)
                .call1(py_args)
                .map_err(|e| extern_err(name, e.to_string()))?;
            // Reject awaitables before the void check so a coroutine is never
            // silently dropped unawaited.
            if ret.hasattr("__await__").unwrap_or(false) {
                return Err(extern_err(
                    name,
                    "returned an awaitable; Python externs must be synchronous",
                ));
            }
            let Some(ty) = ret_ty else {
                return Ok(None); // void extern: return value ignored
            };
            coerce_scalar("return value", &ret, *ty)
                .map(Some)
                .map_err(|m| extern_err(name, m))
        })
    }
}

fn extern_err(name: &str, message: impl Into<String>) -> VmErrorKind {
    VmErrorKind::Extern {
        name: name.to_string(),
        message: message.into(),
    }
}

/// Convert an extern-call argument for a Python callable. Like
/// [`output_value`], except `bit[n]` becomes an unquoted MSB-first "0101"
/// string and `angle` a radians float, so values round-trip with the extern
/// return formats.
fn extern_arg<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    if let Value::Scalar(s) = v {
        match s.value() {
            Primitive::Angle(_) => {
                if let Ok(f) = s.clone().cast(PrimitiveTy::Float(FloatWidth::F64))
                    && let Primitive::Float(radians) = f.value()
                {
                    return radians.into_bound_py_any(py);
                }
            }
            Primitive::BitReg(reg) => {
                if let PrimitiveTy::BitReg(w) = s.ty() {
                    return bitreg_string(reg, w).into_bound_py_any(py);
                }
            }
            _ => {}
        }
    }
    output_value(py, v)
}

/// Unquoted MSB-first bit string of `reg`'s low `width` bits.
fn bitreg_string(reg: &BitReg, width: u32) -> String {
    (0..width as usize)
        .rev()
        .map(|i| if reg.get_bit(i) { '1' } else { '0' })
        .collect()
}

/// bit → bool, int/uint (any width) → exact Python int, float → float,
/// complex → complex; everything else (bit registers, durations, angles,
/// arrays) is its OpenQASM text form.
fn output_value<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    if let Value::Scalar(s) = v {
        match s.value() {
            Primitive::Bit(b) => return b.into_bound_py_any(py),
            Primitive::Int(i) => return i.into_bound_py_any(py),
            Primitive::Uint(u) => return u.into_bound_py_any(py),
            Primitive::Float(f) => return f.into_bound_py_any(py),
            Primitive::Complex(c) => return c.into_bound_py_any(py),
            _ => {}
        }
    }
    v.to_string().into_bound_py_any(py)
}

#[pymodule]
fn oqi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_class::<CompileResult>()?;
    m.add_class::<RunResult>()?;
    m.add("OqiError", m.py().get_type::<OqiError>())?;
    Ok(())
}
