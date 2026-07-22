use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand, ValueEnum};
use oqi_compile::classical::{Array, Duration, Primitive, PrimitiveTy, Value, ValueTy};
use oqi_compile::duration::TableTimings;
use oqi_compile::error::CompileError;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::types::CompileOptions;
use oqi_compile::{bytecode, cfg, duration, qubits, ssa};
use oqi_format::Config;
use oqi_vm::{AutoSim, NoExterns, QuantumBackend, SimdSim, StateVectorSim, SumPolicy, Vm};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Amplitude precision for the simulator backend.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Precision {
    /// Double precision (default).
    F64,
    /// Single precision (half the memory, often faster).
    F32,
}

/// Which quantum execution backend to run on.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind {
    /// Single-threaded scalar CPU state-vector simulator.
    Scalar,
    /// Multi-threaded (rayon) CPU state-vector simulator.
    Rayon,
    /// SIMD-vectorized (single-threaded) CPU state-vector simulator.
    Simd,
    /// SIMD-vectorized, multi-threaded (rayon) CPU state-vector simulator.
    RayonSimd,
    /// WebGPU (wgpu) state-vector simulator (single precision; needs the
    /// `gpu` build feature and a working GPU adapter).
    Gpu,
    /// Auto-routing: stabilizer (Clifford) tableau, then an exact
    /// sum-over-Cliffords for non-Clifford gates, escalating to the
    /// state-vector sim only when the term budget is exhausted.
    Auto,
    /// Like `auto` but pinned to the sum-over-Cliffords tier: exceeding
    /// the term budget is an error instead of a dense fallback.
    Sum,
}

#[derive(Parser)]
#[command(name = "oqi")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Compile-time timing options shared by `compile` and `run`.
#[derive(clap::Args)]
struct TimingArgs {
    /// Supply a fixed duration for a named operation and resolve
    /// `durationof` at compile time (repeatable). Names are gate names;
    /// `measure` and `reset` are reserved. Durations use timing-literal
    /// syntax ("50ns", "4dt"). Unknown names are permitted (a timing
    /// table is a device property, not program-specific)
    #[arg(long = "timing", value_name = "NAME=DURATION")]
    timing: Vec<String>,

    /// Duration of one `dt` unit (e.g. "0.5ns"); affects `dt` literals
    /// and dt-valued timings
    #[arg(long, value_name = "DURATION")]
    dt: Option<String>,

    /// Resolve `durationof` at compile time using the program's defcal
    /// bodies (and any --timing entries; --timing implies this)
    #[arg(long)]
    resolve_durations: bool,

    /// Load a device timing profile: a flat JSON object mapping names to
    /// duration literals ("50ns", "4dt"), with reserved keys `measure`,
    /// `reset`, and `dt`. Repeatable --timing flags override file entries;
    /// --dt overrides the file's "dt". Presence enables compile-time
    /// resolution
    #[arg(long = "timings", value_name = "FILE")]
    timings_file: Option<PathBuf>,
}

impl TimingArgs {
    /// The parsed device-profile file (empty when none was given).
    fn file_entries(&self) -> Result<std::collections::BTreeMap<String, String>, ExitCode> {
        let Some(file) = &self.timings_file else {
            return Ok(Default::default());
        };
        let text = fs::read_to_string(file).map_err(|e| {
            eprintln!("--timings {}: {e}", file.display());
            ExitCode::FAILURE
        })?;
        serde_json::from_str(&text).map_err(|e| {
            eprintln!(
                "--timings {}: expected a flat JSON object of name → duration literal: {e}",
                file.display()
            );
            ExitCode::FAILURE
        })
    }

    /// Compile options honoring the profile's "dt" and `--dt`, or a
    /// rendered failure.
    fn options(&self, path: &Path) -> Result<CompileOptions, ExitCode> {
        let mut options = CompileOptions {
            source_name: Some(path.to_path_buf()),
            ..Default::default()
        };
        if let Some(spec) = self.file_entries()?.get("dt") {
            options.dt = parse_dt(spec).map_err(|e| {
                eprintln!("--timings dt: {e}");
                ExitCode::FAILURE
            })?;
        }
        if let Some(spec) = &self.dt {
            options.dt = parse_dt(spec).map_err(|e| {
                eprintln!("--dt: {e}");
                ExitCode::FAILURE
            })?;
        }
        Ok(options)
    }

    /// When requested, resolve `durationof`/stretch at compile time
    /// (spec semantics); otherwise the VM's runtime pass handles them.
    fn apply(
        &self,
        program: &mut oqi_compile::sir::Program,
        options: &CompileOptions,
        path: &Path,
        source: &str,
    ) -> Result<(), ExitCode> {
        if !(self.resolve_durations || !self.timing.is_empty() || self.timings_file.is_some()) {
            return Ok(());
        }
        // File entries first; --timing flags later so they override.
        let file = self.file_entries()?;
        let mut entries: Vec<(&str, &str)> = file
            .iter()
            .filter(|(name, _)| name.as_str() != "dt")
            .map(|(name, dur)| (name.as_str(), dur.as_str()))
            .collect();
        for spec in &self.timing {
            let Some((name, dur)) = spec.split_once('=') else {
                eprintln!("invalid --timing `{spec}` (expected NAME=DURATION)");
                return Err(ExitCode::FAILURE);
            };
            entries.push((name, dur));
        }
        let table = TableTimings::from_str_entries(entries, &options.dt)
            .map_err(|e| report_compile_error(path, source, e))?
            .with_defcals(program, &options.dt)
            .with_program_gates(program);
        duration::resolve_durationof(program, &table, options)
            .map_err(|e| report_compile_error(path, source, e))?;
        Ok(())
    }
}

#[derive(Subcommand)]
enum Command {
    /// Compile an OpenQASM file
    Compile {
        /// The OpenQASM file to compile
        path: PathBuf,

        /// Dump the IR to stdout
        #[arg(long)]
        dump: bool,

        #[command(flatten)]
        timing: TimingArgs,
    },

    /// Compile and run an OpenQASM file on the CPU simulator
    Run {
        /// The OpenQASM file to run
        path: PathBuf,

        /// Print the final state vector after the run
        #[arg(long)]
        state: bool,

        /// Quantum execution backend
        #[arg(long, value_enum, default_value_t = BackendKind::Scalar)]
        backend: BackendKind,

        /// Amplitude precision
        #[arg(long, value_enum, default_value_t = Precision::F64)]
        precision: Precision,

        /// Maximum stabilizer terms for the auto/sum backends' exact
        /// sum-over-Cliffords tier (each non-Clifford gate multiplies the
        /// term count by 2-3)
        #[arg(long, default_value_t = 1024)]
        max_rank: usize,

        /// Supply a value for a declared input (repeatable). Arrays take a
        /// comma-separated list, optionally bracketed: `NAME=1,2,3`.
        #[arg(long = "input", value_name = "NAME=VALUE")]
        input: Vec<String>,

        #[command(flatten)]
        timing: TimingArgs,
    },

    /// Format OpenQASM files
    Fmt {
        /// Use compact formatting
        #[arg(long)]
        compact: bool,

        /// Print to stdout instead of writing back to file
        #[arg(long)]
        stdout: bool,

        /// Files to format
        paths: Vec<PathBuf>,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Compile { path, dump, timing } => compile(&path, dump, &timing),
        Command::Run {
            path,
            state,
            backend,
            precision,
            max_rank,
            input,
            timing,
        } => run(&path, state, backend, precision, max_rank, &input, &timing).await,
        Command::Fmt {
            compact,
            stdout,
            paths,
        } => fmt(compact, stdout, &paths),
    }
}

fn compile(path: &Path, dump: bool, timing: &TimingArgs) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let options = match timing.options(path) {
        Ok(o) => o,
        Err(code) => return code,
    };
    let mut program = match oqi_compile::lower::compile_source_with_options(
        &source,
        DefaultIncludeResolver,
        options.clone(),
    ) {
        Ok(p) => p,
        Err(e) => {
            oqi_diagnostics::emit(&e, path, &source);
            return ExitCode::FAILURE;
        }
    };
    if let Err(code) = timing.apply(&mut program, &options, path, &source) {
        return code;
    }

    if dump {
        print!("{program}");
    }

    ExitCode::SUCCESS
}

async fn run(
    path: &Path,
    show_state: bool,
    backend: BackendKind,
    precision: Precision,
    max_rank: usize,
    input_specs: &[String],
    timing: &TimingArgs,
) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    // Full pipeline: source → SIR → CFG → SSA → bytecode.
    let options = match timing.options(path) {
        Ok(o) => o,
        Err(code) => return code,
    };
    let mut program = match oqi_compile::lower::compile_source_with_options(
        &source,
        DefaultIncludeResolver,
        options.clone(),
    ) {
        Ok(p) => p,
        Err(e) => return report_compile_error(path, &source, e),
    };
    if let Err(code) = timing.apply(&mut program, &options, path, &source) {
        return code;
    }
    let cfgs = match cfg::build_program(&program) {
        Ok(c) => c,
        Err(e) => return report_compile_error(path, &source, e),
    };
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    let module = match bytecode::emit(&ssa, &program, layout) {
        Ok(m) => m,
        Err(e) => return report_compile_error(path, &source, e),
    };

    let inputs = match parse_inputs(&module, input_specs) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let sim: Box<dyn QuantumBackend> = match backend {
        BackendKind::Gpu => match build_gpu_backend(precision, module.qubits.num_qubits).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("gpu backend: {e}");
                return ExitCode::FAILURE;
            }
        },
        _ => match build_backend(backend, precision, max_rank, module.qubits.num_qubits) {
            Ok(s) => s,
            Err(e) => {
                oqi_diagnostics::emit(&e, path, &source);
                return ExitCode::FAILURE;
            }
        },
    };
    let mut vm = Vm::new(&module, sim, NoExterns);
    match vm.run_with_inputs(inputs).await {
        Ok(result) => {
            for (qubit, bit) in &result.measurements {
                println!("q[{qubit}] = {}", if *bit { 1 } else { 0 });
            }
            for (sym, value) in &result.outputs {
                println!("{} = {value}", module.symbols.get(*sym).name);
            }
            if show_state {
                let width = module.qubits.num_qubits as usize;
                if let Some(amps) = vm.backend().amplitudes().await {
                    for (i, amp) in amps.iter().enumerate() {
                        if amp.norm() > 1e-12 {
                            println!("|{i:0width$b}> = {amp}");
                        }
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            oqi_diagnostics::emit(&e, path, &source);
            ExitCode::FAILURE
        }
    }
}

/// Default RNG seed (matches `StateVectorSim::new`).
const DEFAULT_SEED: u64 = 0x2545F4914F6CDD1D;

/// Parse a `--dt` duration ("0.5ns"): timing-literal syntax, except the
/// `dt` unit itself is rejected (self-referential) and the value must be
/// positive.
fn parse_dt(spec: &str) -> Result<Duration, String> {
    if spec.trim_end().ends_with("dt") {
        return Err("cannot itself be given in `dt` units".into());
    }
    let d = oqi_compile::types::parse_timing_literal(spec, &CompileOptions::default().dt)
        .map_err(|_| format!("`{spec}` is not a duration literal (e.g. \"0.5ns\")"))?;
    if d.value <= 0.0 {
        return Err("must be a positive duration".into());
    }
    Ok(d)
}

/// Construct the runtime-selected backend as a boxed trait object.
fn build_backend(
    backend: BackendKind,
    precision: Precision,
    max_rank: usize,
    num_qubits: u32,
) -> Result<Box<dyn QuantumBackend>, oqi_vm::VmError> {
    // Auto-routing is exact (stabilizer tableau + sum-over-Cliffords);
    // precision is irrelevant until it escalates to dense, which uses f64.
    if matches!(backend, BackendKind::Auto | BackendKind::Sum) {
        let policy = SumPolicy {
            max_rank,
            dense_escape: matches!(backend, BackendKind::Auto),
        };
        return Ok(Box::new(AutoSim::with_policy(
            num_qubits,
            DEFAULT_SEED,
            policy,
        )));
    }
    let simd = matches!(backend, BackendKind::Simd | BackendKind::RayonSimd);
    let par = matches!(backend, BackendKind::Rayon | BackendKind::RayonSimd);
    let sim: Box<dyn QuantumBackend> = match (simd, precision) {
        (true, Precision::F64) => {
            Box::new(SimdSim::<f64>::try_zeroed(num_qubits, DEFAULT_SEED)?.with_parallel(par))
        }
        (true, Precision::F32) => {
            Box::new(SimdSim::<f32>::try_zeroed(num_qubits, DEFAULT_SEED)?.with_parallel(par))
        }
        (false, Precision::F64) => Box::new(
            StateVectorSim::<f64>::try_zeroed(num_qubits, DEFAULT_SEED)?.with_parallel(par),
        ),
        (false, Precision::F32) => Box::new(
            StateVectorSim::<f32>::try_zeroed(num_qubits, DEFAULT_SEED)?.with_parallel(par),
        ),
    };
    Ok(sim)
}

/// Construct the WebGPU backend (single precision only). Returns an error
/// string if the build lacks gpu support or no adapter/device is available.
#[cfg(feature = "gpu")]
async fn build_gpu_backend(
    precision: Precision,
    num_qubits: u32,
) -> Result<Box<dyn QuantumBackend>, String> {
    if matches!(precision, Precision::F64) {
        eprintln!("note: the gpu backend is single precision (f32); --precision f64 ignored");
    }
    Ok(Box::new(oqi_vm::GpuSim::new(num_qubits).await?))
}

#[cfg(not(feature = "gpu"))]
async fn build_gpu_backend(
    _precision: Precision,
    _num_qubits: u32,
) -> Result<Box<dyn QuantumBackend>, String> {
    Err("this build has no gpu support; rebuild with `--features gpu`".to_string())
}

/// Parse `--input NAME=VALUE` specs into a symbol-keyed value map,
/// coercing each value to its declared scalar type. Errors on an unknown
/// name, a non-input symbol, an unparsable value, or a non-scalar type.
fn parse_inputs(
    module: &bytecode::BcModule,
    specs: &[String],
) -> Result<HashMap<SymbolId, Value>, String> {
    let mut map = HashMap::new();
    for spec in specs {
        let (name, raw) = spec
            .split_once('=')
            .ok_or_else(|| format!("invalid --input `{spec}` (expected NAME=VALUE)"))?;
        let sym = module
            .symbols
            .iter()
            .find(|s| s.kind == SymbolKind::Input && s.name == name)
            .ok_or_else(|| format!("no input named `{name}`"))?;
        let ty = sym
            .ty
            .value_ty()
            .ok_or_else(|| format!("input `{name}` has no value type"))?;
        let value = match ty {
            // `array[T, N]` accepts a comma-separated list, optionally bracketed
            // (`1,2,3` or `[1,2,3]`), flattened row-major for multi-dim arrays.
            ValueTy::Array(aty) => {
                let inner = raw
                    .trim()
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .unwrap_or(raw)
                    .trim();
                let parts: Vec<&str> = if inner.is_empty() {
                    Vec::new()
                } else {
                    inner.split(',').collect()
                };
                let elem_ty = ValueTy::Scalar(aty.ty());
                let elems = parts
                    .iter()
                    .map(|part| match parse_scalar(name, part.trim(), elem_ty)? {
                        Value::Scalar(s) => Ok(s.into_value()),
                        _ => unreachable!("a scalar type parses to a scalar value"),
                    })
                    .collect::<Result<Vec<Primitive>, String>>()?;
                Value::Array(
                    Array::new(elems, aty).map_err(|e| format!("input `{name}`: {e:?}"))?,
                )
            }
            scalar_ty => parse_scalar(name, raw, scalar_ty)?,
        };
        map.insert(sym.id, value);
    }
    Ok(map)
}

/// Parse a raw `--input` string into a scalar [`Value`] of type `ty`. Also
/// used per-element when parsing `array[…]` inputs. Handles int/uint/float/
/// bit/bool; other scalar types are unsupported on the CLI.
fn parse_scalar(name: &str, raw: &str, ty: ValueTy) -> Result<Value, String> {
    Ok(match ty {
        ValueTy::Scalar(PrimitiveTy::Int(w)) => Value::int(
            raw.parse()
                .map_err(|_| format!("input `{name}`: `{raw}` is not an integer"))?,
            w,
        ),
        ValueTy::Scalar(PrimitiveTy::Uint(w)) => Value::uint(
            raw.parse()
                .map_err(|_| format!("input `{name}`: `{raw}` is not an unsigned integer"))?,
            w,
        ),
        ValueTy::Scalar(PrimitiveTy::Float(w)) => Value::float(
            raw.parse()
                .map_err(|_| format!("input `{name}`: `{raw}` is not a float"))?,
            w,
        ),
        ValueTy::Scalar(PrimitiveTy::Bit) | ValueTy::Scalar(PrimitiveTy::Bool) => {
            let b = match raw {
                "0" | "false" => false,
                "1" | "true" => true,
                _ => {
                    return Err(format!(
                        "input `{name}`: `{raw}` is not a bit (use 0/1/true/false)"
                    ));
                }
            };
            Value::bit(b)
        }
        _ => return Err(format!("input `{name}` has a type unsupported on the CLI")),
    })
}

/// Render a compiler diagnostic with source context and return failure.
fn report_compile_error(path: &Path, source: &str, e: CompileError) -> ExitCode {
    oqi_diagnostics::emit(&e, path, source);
    ExitCode::FAILURE
}

fn fmt(compact: bool, stdout: bool, paths: &[PathBuf]) -> ExitCode {
    let config = if compact {
        Config { compact: true }
    } else {
        Config::default()
    };

    let had_errors = AtomicBool::new(false);

    paths.par_iter().for_each(|path| {
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                had_errors.store(true, Ordering::Relaxed);
                return;
            }
        };

        let formatted = match oqi_format::format(&source, config) {
            Ok(s) => s,
            Err(e) => {
                oqi_diagnostics::emit(&e, path, &source);
                had_errors.store(true, Ordering::Relaxed);
                return;
            }
        };

        if stdout {
            print!("{formatted}");
        } else if let Err(e) = fs::write(path, &formatted) {
            eprintln!("{}: {e}", path.display());
            had_errors.store(true, Ordering::Relaxed);
        }
    });

    if had_errors.load(Ordering::Relaxed) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
