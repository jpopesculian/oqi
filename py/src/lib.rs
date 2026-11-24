//! Python (PyO3) bindings for the oqi compiler and VM.
//!
//! Exposes a minimal API: [`compile`] (source → bytecode + disassembly),
//! [`run`] (source → simulated results) and [`sample`] (source → per-output
//! histograms over many shots). `run` and `sample` can call host-supplied
//! `extern` implementations (the `externs` option), as synchronous callables.
//! Includes are limited to the embedded `stdgates.inc`; file includes are
//! rejected. Errors raise [`OqiError`] whose message is the rendered
//! diagnostic.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;
use oqi_compile::classical::{Value, ValueTy};
use oqi_compile::{BcModule, CompileOptions, IncludeResolver, LibOnlyResolver};
use oqi_vm::{
    BackendSpec, DEFAULT_SEED, ExternProvider, HostValue, KindResult, QuantumBackend, SumPolicy,
    Vm, VmErrorKind,
};
use pyo3::IntoPyObjectExt;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

/// Name used for the root source in rendered diagnostics.
const SOURCE_NAME: &str = "<source>";

create_exception!(
    oqi,
    OqiError,
    PyException,
    "Compile or runtime error; the message is the rendered diagnostic."
);

/// The text of a file a diagnostic points into: whatever the include
/// resolver serves (here, no files — only libraries such as `stdgates.inc`).
fn included_file(path: &Path) -> Option<String> {
    LibOnlyResolver
        .resolve_path(path)
        .ok()
        .map(|text| text.into_owned())
}

/// Render any workspace diagnostic against the root source as an `OqiError`.
fn diag_err(diag: &dyn oqi_diagnostics::Diagnostic, source: &str) -> PyErr {
    OqiError::new_err(oqi_diagnostics::render_to_string(
        diag,
        Path::new(SOURCE_NAME),
        source,
        &included_file,
    ))
}

/// Source → bytecode module. With `timings` supplied (even an empty dict),
/// `durationof` is resolved at compile time; otherwise it stays for the VM's
/// runtime pass.
fn build_module(
    source: &str,
    timings: Option<&Bound<'_, PyDict>>,
    dt: Option<&str>,
) -> PyResult<BcModule> {
    let mut options = CompileOptions {
        source_name: Some(Path::new(SOURCE_NAME).to_path_buf()),
        ..Default::default()
    };
    if let Some(spec) = dt {
        options.dt = oqi_compile::parse_dt(spec).map_err(|e| diag_err(&e, source))?;
    }
    let timings: Option<Vec<(String, String)>> = timings
        .map(|timings| {
            timings
                .iter()
                .map(|(key, val)| {
                    let name: String = key
                        .extract()
                        .map_err(|_| OqiError::new_err("timing names must be strings"))?;
                    let dur: String = val.extract().map_err(|_| {
                        OqiError::new_err(format!(
                            "timing for `{name}` must be a duration string (e.g. \"50ns\")"
                        ))
                    })?;
                    Ok((name, dur))
                })
                .collect::<PyResult<_>>()
        })
        .transpose()?;
    let timings: Option<Vec<(&str, &str)>> = timings
        .as_ref()
        .map(|t| t.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());
    oqi_compile::compile(source, LibOnlyResolver, &options, timings.as_deref())
        .map_err(|e| diag_err(&e, source))
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
    let module = build_module(source, None, None)?;
    let bytes = oqi_compile::bytecode::to_bytes(&module)
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
    /// The backend that ran, by name.
    backend: &'static str,
    /// The amplitude precision that ran: `"f64"` or `"f32"`.
    precision: &'static str,
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

/// Result of [`sample`].
#[pyclass(frozen, get_all, module = "oqi")]
struct SampleResult {
    /// The number of shots that ran.
    shots: u32,
    /// Per output variable (in program order), how often each value was
    /// observed: `{name: {label: count}}`, labels sorted by value. `bit` and
    /// `bool` are labelled `"0"`/`"1"`, `bit[n]` as an MSB-first bit string.
    histograms: Py<PyDict>,
    /// The backend that ran, by name.
    backend: &'static str,
    /// The amplitude precision that ran: `"f64"` or `"f32"`.
    precision: &'static str,
}

#[pymethods]
impl SampleResult {
    fn __repr__(&self) -> String {
        format!(
            "SampleResult(shots={}, histograms=<{} entries>)",
            self.shots,
            Python::attach(|py| self.histograms.bind(py).len()),
        )
    }
}

/// Everything [`run`] and [`sample`] share: the compiled module and the
/// pieces a [`Vm`] is assembled from.
struct Session {
    module: BcModule,
    inputs: Vec<(String, HostValue)>,
    externs: PyExterns,
    backend: Box<dyn QuantumBackend>,
    /// The backend configuration that was actually built.
    built: BackendSpec,
    seed: u64,
}

#[allow(clippy::too_many_arguments)]
fn session(
    source: &str,
    inputs: Option<&Bound<'_, PyDict>>,
    seed: Option<u64>,
    externs: Option<&Bound<'_, PyDict>>,
    timings: Option<&Bound<'_, PyDict>>,
    dt: Option<&str>,
    backend: &str,
    precision: &str,
    max_rank: Option<usize>,
    max_bond: Option<usize>,
) -> PyResult<Session> {
    let module = build_module(source, timings, dt)?;
    let externs = PyExterns::new(&module, externs)?;

    let mut named = Vec::new();
    for (key, val) in inputs.iter().flat_map(|inputs| inputs.iter()) {
        let name: String = key
            .extract()
            .map_err(|_| OqiError::new_err("input names must be strings"))?;
        let value = host_value(&val).ok_or_else(|| {
            OqiError::new_err(format!(
                "input `{name}` has an unsupported Python type `{}`",
                type_name(&val)
            ))
        })?;
        named.push((name, value));
    }

    let seed = seed.unwrap_or(DEFAULT_SEED);
    let mut spec: BackendSpec = backend.parse().map_err(OqiError::new_err)?;
    spec.precision = precision.parse().map_err(OqiError::new_err)?;
    spec.seed = seed;
    if let Some(max_rank) = max_rank {
        spec.sum_policy = SumPolicy {
            max_rank,
            ..spec.sum_policy
        };
    }
    if let Some(max_bond) = max_bond {
        spec.max_bond = max_bond;
    }
    // Backend construction resolves immediately for the CPU simulators and
    // polls the device itself for the GPU one, so blocking never parks.
    let (backend, built) = pollster::block_on(spec.build(module.qubits.num_qubits))
        .map_err(|e| diag_err(&e, source))?;

    Ok(Session {
        module,
        inputs: named,
        externs,
        backend,
        built,
        seed,
    })
}

/// Compile and run an OpenQASM 3 program on a simulator.
///
/// `externs` maps each `extern` function's name to a synchronous Python
/// callable. Return values are coerced to the declared return type (radians
/// for `angle[n]`, MSB-first "0101" strings for `bit[n]`). An extern with no
/// implementation, or one that raises, aborts the run with a diagnostic
/// error; a callable returning an awaitable is rejected.
///
/// `timings` maps operation names (`measure`/`reset` reserved) to duration
/// strings ("50ns", "4dt") and enables compile-time `durationof` resolution,
/// which also derives durations from the program's defcal bodies (an empty
/// dict enables defcal derivation alone). `dt` sets the device time unit
/// (e.g. "0.5ns").
///
/// `backend` selects the simulator by the same names as the CLI's
/// `--backend` (`"scalar"`, `"rayon"`, `"simd"`, `"rayon-simd"`, `"gpu"`,
/// `"prefer-gpu"`, `"auto"`, `"sum"`, `"mps"`); `precision` (`"f64"` or
/// `"f32"`) is the amplitude precision of the CPU state-vector simulators,
/// `max_rank` the stabilizer-term budget of `"auto"`/`"sum"` and `max_bond`
/// the bond-dimension cap of `"mps"`, past which its state is truncated.
#[pyfunction]
#[pyo3(signature = (source, *, inputs=None, seed=None, statevector=false, externs=None, timings=None, dt=None, backend="scalar", precision="f64", max_rank=None, max_bond=None))]
#[allow(clippy::too_many_arguments)]
fn run(
    py: Python<'_>,
    source: &str,
    inputs: Option<&Bound<'_, PyDict>>,
    seed: Option<u64>,
    statevector: bool,
    externs: Option<&Bound<'_, PyDict>>,
    timings: Option<&Bound<'_, PyDict>>,
    dt: Option<&str>,
    backend: &str,
    precision: &str,
    max_rank: Option<usize>,
    max_bond: Option<usize>,
) -> PyResult<RunResult> {
    let session = session(
        source, inputs, seed, externs, timings, dt, backend, precision, max_rank, max_bond,
    )?;
    let module = &session.module;
    let mut vm = Vm::new(module, session.backend, session.externs);

    // The VM futures resolve immediately (the CPU simulators and Python
    // externs never suspend), so blocking here never parks the thread. Runs hold the GIL:
    // `oqi_vm::RunResult` is not `Send`, which rules out `Python::detach`.
    let result = pollster::block_on(vm.run_with_named_inputs(&session.inputs))
        .map_err(|e| diag_err(&e, source))?;
    let amps = if statevector {
        pollster::block_on(vm.backend().amplitudes())
    } else {
        None
    };

    let outputs = PyDict::new(py);
    for (sym, v) in &result.outputs {
        outputs.set_item(
            module.symbol(*sym).name.as_str(),
            py_value(py, HostValue::from(v))?,
        )?;
    }
    Ok(RunResult {
        measurements: result.measurements,
        outputs: outputs.into(),
        statevector: amps,
        backend: session.built.name(),
        precision: session.built.precision.as_str(),
    })
}

/// Compile and sample an OpenQASM 3 program over `shots` runs, returning a
/// histogram of observed values per named output variable. Takes the same
/// options as [`run`]; a given `seed` reproduces the whole histogram.
#[pyfunction]
#[pyo3(signature = (source, *, shots=1024, inputs=None, seed=None, externs=None, timings=None, dt=None, backend="scalar", precision="f64", max_rank=None, max_bond=None))]
#[allow(clippy::too_many_arguments)]
fn sample(
    py: Python<'_>,
    source: &str,
    shots: u32,
    inputs: Option<&Bound<'_, PyDict>>,
    seed: Option<u64>,
    externs: Option<&Bound<'_, PyDict>>,
    timings: Option<&Bound<'_, PyDict>>,
    dt: Option<&str>,
    backend: &str,
    precision: &str,
    max_rank: Option<usize>,
    max_bond: Option<usize>,
) -> PyResult<SampleResult> {
    if shots == 0 {
        return Err(OqiError::new_err("`shots` must be at least 1"));
    }
    let session = session(
        source, inputs, seed, externs, timings, dt, backend, precision, max_rank, max_bond,
    )?;
    let module = &session.module;
    let mut vm = Vm::new(module, session.backend, session.externs);
    let inputs =
        oqi_vm::coerce_inputs(module, &session.inputs).map_err(|e| diag_err(&e, source))?;
    let results = pollster::block_on(vm.sample(inputs, shots, session.seed))
        .map_err(|e| diag_err(&e, source))?;

    let histograms = PyDict::new(py);
    for histogram in oqi_vm::histograms(module, &results) {
        let counts = PyDict::new(py);
        for (label, count) in histogram.counts {
            counts.set_item(label, count)?;
        }
        histograms.set_item(histogram.name, counts)?;
    }
    Ok(SampleResult {
        shots,
        histograms: histograms.into(),
        backend: session.built.name(),
        precision: session.built.precision.as_str(),
    })
}

fn type_name(v: &Bound<'_, PyAny>) -> String {
    v.get_type()
        .name()
        .map_or_else(|_| "<unknown>".to_string(), |n| n.to_string())
}

/// A Python value as a host value: `bool`, `int`, `float`, `complex`, `str`,
/// and (nested) `list`/`tuple`s of those. `None` for anything else.
fn host_value(v: &Bound<'_, PyAny>) -> Option<HostValue> {
    // `bool` first: it is a subclass of `int`.
    if v.is_instance_of::<PyBool>() {
        v.extract().ok().map(HostValue::Bool)
    } else if v.is_instance_of::<PyInt>() {
        (v.extract().map(HostValue::Int))
            .or_else(|_| v.extract().map(HostValue::Uint))
            // Beyond 128 bits: let coercion report it against the type.
            .or_else(|_| v.str().map(|s| HostValue::Text(s.to_string())))
            .ok()
    } else if v.is_instance_of::<PyFloat>() {
        v.extract().ok().map(HostValue::Float)
    } else if v.is_instance_of::<PyComplex>() {
        v.extract().ok().map(HostValue::Complex)
    } else if v.is_instance_of::<PyString>() {
        v.extract().ok().map(HostValue::Text)
    } else if let Ok(list) = v.cast::<PyList>() {
        list.iter()
            .map(|item| host_value(&item))
            .collect::<Option<_>>()
            .map(HostValue::List)
    } else if let Ok(tuple) = v.cast::<PyTuple>() {
        tuple
            .iter()
            .map(|item| host_value(&item))
            .collect::<Option<_>>()
            .map(HostValue::List)
    } else {
        None
    }
}

/// A host value as Python sees it: `bool`, exact `int` (any width), `float`,
/// `complex`, `str`, or `list`.
fn py_value<'py>(py: Python<'py>, value: HostValue) -> PyResult<Bound<'py, PyAny>> {
    match value {
        HostValue::Bool(b) => b.into_bound_py_any(py),
        HostValue::Int(i) => i.into_bound_py_any(py),
        HostValue::Uint(u) => u.into_bound_py_any(py),
        HostValue::Float(f) => f.into_bound_py_any(py),
        HostValue::Complex(c) => c.into_bound_py_any(py),
        HostValue::Text(s) => s.into_bound_py_any(py),
        HostValue::List(items) => items
            .into_iter()
            .map(|item| py_value(py, item))
            .collect::<PyResult<Vec<_>>>()?
            .into_bound_py_any(py),
    }
}

/// Extern implementations supplied from Python (the `externs` option):
/// name → callable, plus each declared extern's return type (`None` = void)
/// — the VM stores extern results uncast, so returns must be built with
/// their declared type.
struct PyExterns {
    fns: HashMap<String, Py<PyAny>>,
    ret_tys: HashMap<String, Option<ValueTy>>,
}

impl PyExterns {
    /// `externs` is the `externs=` option (`None` ⇒ none). Errors if any
    /// key isn't a string or any value isn't callable.
    fn new(module: &BcModule, externs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let ret_tys = module
            .externs()
            .map(|(name, ty)| (name.to_string(), ty))
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

impl ExternProvider for PyExterns {
    async fn call(&mut self, name: &str, args: &[Value]) -> KindResult<Option<Value>> {
        // Not provided, or not a declared classical extern (ports/frames land
        // here when no pulse handler is installed) — same error as `NoExterns`.
        let (Some(f), Some(ret_ty)) = (self.fns.get(name), self.ret_tys.get(name)) else {
            return Err(VmErrorKind::UnknownExtern(name.to_string()));
        };
        let failed = |message: String| VmErrorKind::extern_failed(name, message);
        Python::attach(|py| {
            let py_args = args
                .iter()
                .map(|a| py_value(py, HostValue::extern_arg(a)))
                .collect::<PyResult<Vec<_>>>()
                .and_then(|a| PyTuple::new(py, a))
                .map_err(|e| failed(e.to_string()))?;
            let ret = f
                .bind(py)
                .call1(py_args)
                .map_err(|e| failed(e.to_string()))?;
            // Reject awaitables before the void check so a coroutine is never
            // silently dropped unawaited.
            if ret.hasattr("__await__").unwrap_or(false) {
                return Err(failed(
                    "returned an awaitable; Python externs must be synchronous".to_string(),
                ));
            }
            let Some(ty) = ret_ty else {
                return Ok(None); // void extern: return value ignored
            };
            let raw = host_value(&ret).ok_or_else(|| {
                failed(format!(
                    "returned an unsupported Python type `{}`",
                    type_name(&ret)
                ))
            })?;
            oqi_vm::coerce("return value", &raw, *ty)
                .map(Some)
                .map_err(failed)
        })
    }
}

#[pymodule]
fn oqi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_class::<CompileResult>()?;
    m.add_class::<RunResult>()?;
    m.add_class::<SampleResult>()?;
    m.add("OqiError", m.py().get_type::<OqiError>())?;
    Ok(())
}
