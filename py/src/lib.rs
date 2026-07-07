//! Python (PyO3) bindings for the oqi compiler and VM.
//!
//! Exposes a minimal API: [`compile`] (source → bytecode + disassembly) and
//! [`run`] (source → simulated results). Includes are limited to the embedded
//! `stdgates.inc`; file includes are rejected. Errors raise [`OqiError`] whose
//! message is the rendered diagnostic.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;
use oqi_compile::bytecode::{self, BcModule};
use oqi_compile::classical::{Primitive, PrimitiveTy, Value, ValueTy};
use oqi_compile::resolve::IncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::{cfg, qubits, ssa};
use oqi_vm::{NoExterns, QuantumBackend, StateVectorSim, Vm};
use pyo3::IntoPyObjectExt;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

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
#[pyfunction]
#[pyo3(signature = (source, *, inputs=None, seed=None, statevector=false))]
fn run(
    py: Python<'_>,
    source: &str,
    inputs: Option<&Bound<'_, PyDict>>,
    seed: Option<u64>,
    statevector: bool,
) -> PyResult<RunResult> {
    let module = build_module(source)?;
    let input_map = coerce_inputs(&module, inputs)?;
    let sim =
        StateVectorSim::<f64>::try_zeroed(module.qubits.num_qubits, seed.unwrap_or(DEFAULT_SEED))
            .map_err(|e| diag_err(&e, source))?;
    let mut vm = Vm::new(&module, sim, NoExterns);

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
/// symbols. Mirrors `parse_inputs` in `cli/src/main.rs` and `coerce_inputs`
/// in `js/src/lib.rs`; keep the three in sync.
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
        let value = match ty {
            ValueTy::Scalar(PrimitiveTy::Int(w)) => Value::int(to_i128(&name, &val)?, w),
            ValueTy::Scalar(PrimitiveTy::Uint(w)) => Value::uint(to_u128(&name, &val)?, w),
            ValueTy::Scalar(PrimitiveTy::Float(w)) => Value::float(to_f64(&name, &val)?, w),
            ValueTy::Scalar(PrimitiveTy::Bit) | ValueTy::Scalar(PrimitiveTy::Bool) => {
                Value::bit(to_bool(&name, &val)?)
            }
            _ => {
                return Err(OqiError::new_err(format!(
                    "input `{name}` has a type unsupported by the Python API"
                )));
            }
        };
        map.insert(sym.id, value);
    }
    Ok(map)
}

fn to_i128(name: &str, v: &Bound<'_, PyAny>) -> PyResult<i128> {
    if !v.is_instance_of::<PyBool>()
        && let Ok(i) = v.extract::<i128>()
    {
        return Ok(i);
    }
    if let Ok(s) = v.extract::<String>() {
        return s
            .parse()
            .map_err(|_| OqiError::new_err(format!("input `{name}`: `{s}` is not an integer")));
    }
    if let Ok(f) = v.extract::<f64>()
        && f.fract() == 0.0
        && f.abs() <= MAX_SAFE_INTEGER
    {
        return Ok(f as i128);
    }
    Err(OqiError::new_err(format!(
        "input `{name}` must be an int or a decimal string"
    )))
}

fn to_u128(name: &str, v: &Bound<'_, PyAny>) -> PyResult<u128> {
    if !v.is_instance_of::<PyBool>()
        && let Ok(u) = v.extract::<u128>()
    {
        return Ok(u);
    }
    if let Ok(s) = v.extract::<String>() {
        return s.parse().map_err(|_| {
            OqiError::new_err(format!("input `{name}`: `{s}` is not an unsigned integer"))
        });
    }
    if let Ok(f) = v.extract::<f64>()
        && f.fract() == 0.0
        && (0.0..=MAX_SAFE_INTEGER).contains(&f)
    {
        return Ok(f as u128);
    }
    Err(OqiError::new_err(format!(
        "input `{name}` must be a non-negative int or a decimal string"
    )))
}

fn to_f64(name: &str, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if !v.is_instance_of::<PyBool>()
        && let Ok(f) = v.extract::<f64>()
    {
        return Ok(f);
    }
    if let Ok(s) = v.extract::<String>() {
        return s
            .parse()
            .map_err(|_| OqiError::new_err(format!("input `{name}`: `{s}` is not a float")));
    }
    Err(OqiError::new_err(format!(
        "input `{name}` must be a number or a decimal string"
    )))
}

fn to_bool(name: &str, v: &Bound<'_, PyAny>) -> PyResult<bool> {
    if v.is_instance_of::<PyBool>() {
        return v.extract();
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
    Err(OqiError::new_err(format!(
        "input `{name}` is not a bit (use 0/1/True/False)"
    )))
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
