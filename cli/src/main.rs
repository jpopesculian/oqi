use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand, ValueEnum};
use oqi_compile::classical::{PrimitiveTy, Value, ValueTy};
use oqi_compile::error::CompileError;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::{bytecode, cfg, qubits, ssa};
use oqi_format::Config;
use oqi_vm::{NoExterns, QuantumBackend, StateVectorSim, Vm};
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
}

#[derive(Parser)]
#[command(name = "oqi")]
struct Cli {
    #[command(subcommand)]
    command: Command,
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

        /// Supply a value for a declared input (repeatable)
        #[arg(long = "input", value_name = "NAME=VALUE")]
        input: Vec<String>,
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

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Compile { path, dump } => compile(&path, dump),
        Command::Run {
            path,
            state,
            backend,
            precision,
            input,
        } => run(&path, state, backend, precision, &input),
        Command::Fmt {
            compact,
            stdout,
            paths,
        } => fmt(compact, stdout, &paths),
    }
}

fn compile(path: &Path, dump: bool) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let program =
        match oqi_compile::lower::compile_source(&source, DefaultIncludeResolver, Some(path)) {
            Ok(p) => p,
            Err(e) => {
                oqi_diagnostics::emit(&e, path, &source);
                return ExitCode::FAILURE;
            }
        };

    if dump {
        print!("{program}");
    }

    ExitCode::SUCCESS
}

fn run(
    path: &Path,
    show_state: bool,
    backend: BackendKind,
    precision: Precision,
    input_specs: &[String],
) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    // Full pipeline: source → SIR → CFG → SSA → bytecode.
    let program =
        match oqi_compile::lower::compile_source(&source, DefaultIncludeResolver, Some(path)) {
            Ok(p) => p,
            Err(e) => return report_compile_error(path, &source, e),
        };
    let cfgs = match cfg::build_program(&program) {
        Ok(c) => c,
        Err(e) => return report_compile_error(path, &source, e),
    };
    let ssa = ssa::build_program(&cfgs, &program.symbols);
    let layout = qubits::build_layout(&program);
    let module = match bytecode::emit(&ssa, &program.symbols, layout) {
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

    let sim = match build_backend(backend, precision, module.qubits.num_qubits) {
        Ok(s) => s,
        Err(e) => {
            oqi_diagnostics::emit(&e, path, &source);
            return ExitCode::FAILURE;
        }
    };
    let mut vm = Vm::new(&module, sim, NoExterns);
    match vm.run_with_inputs(inputs) {
        Ok(result) => {
            for (qubit, bit) in &result.measurements {
                println!("q[{qubit}] = {}", if *bit { 1 } else { 0 });
            }
            for (sym, value) in &result.outputs {
                println!("{} = {value}", module.symbols.get(*sym).name);
            }
            if show_state {
                let width = module.qubits.num_qubits as usize;
                if let Some(amps) = vm.backend().amplitudes() {
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

/// Construct the runtime-selected backend as a boxed trait object.
fn build_backend(
    backend: BackendKind,
    precision: Precision,
    num_qubits: u32,
) -> Result<Box<dyn QuantumBackend>, oqi_vm::VmError> {
    let par = matches!(backend, BackendKind::Rayon);
    let sim: Box<dyn QuantumBackend> = match precision {
        Precision::F64 => Box::new(
            StateVectorSim::<f64>::try_zeroed(num_qubits, DEFAULT_SEED)?.with_parallel(par),
        ),
        Precision::F32 => Box::new(
            StateVectorSim::<f32>::try_zeroed(num_qubits, DEFAULT_SEED)?.with_parallel(par),
        ),
    };
    Ok(sim)
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
        };
        map.insert(sym.id, value);
    }
    Ok(map)
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
