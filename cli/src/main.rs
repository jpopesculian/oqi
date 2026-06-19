use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use ariadne::{Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};
use oqi_compile::classical::{PrimitiveTy, Value, ValueTy};
use oqi_compile::error::CompileError;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::symbol::{SymbolId, SymbolKind};
use oqi_compile::{bytecode, cfg, qubits, ssa};
use oqi_format::Config;
use oqi_vm::{NoExterns, StateVectorSim, Vm, VmError};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

const DIAGNOSTIC_TAB_SIZE: usize = 4;

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
        Command::Run { path, state, input } => run(&path, state, &input),
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
                let diagnostic_path = e.path.as_deref().unwrap_or(path);
                let diagnostic_source =
                    match load_diagnostic_source(path, &source, e.path.as_deref()) {
                        Ok(source) => source,
                        Err(read_err) => {
                            eprintln!(
                                "{}: failed to read source for diagnostic: {read_err}",
                                diagnostic_path.display()
                            );
                            eprintln!("{}: {}", diagnostic_path.display(), e);
                            return ExitCode::FAILURE;
                        }
                    };
                let (line, column) = e
                    .span
                    .doc_position(diagnostic_source.as_ref(), DIAGNOSTIC_TAB_SIZE);
                let headline =
                    format_diagnostic_message(diagnostic_path, line, column, &e.to_string());
                emit_error_report(
                    diagnostic_path,
                    diagnostic_source.as_ref(),
                    Range::from(e.span),
                    &headline,
                    &e.to_string(),
                );
                return ExitCode::FAILURE;
            }
        };

    if dump {
        print!("{program}");
    }

    ExitCode::SUCCESS
}

fn run(path: &Path, show_state: bool, input_specs: &[String]) -> ExitCode {
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

    let sim = StateVectorSim::new(module.qubits.num_qubits);
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
                for (i, amp) in vm.backend().state().iter().enumerate() {
                    if amp.norm() > 1e-12 {
                        println!("|{i:0width$b}> = {amp}");
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(VmError::MissingInput(sym)) => {
            let name = &module.symbols.get(sym).name;
            eprintln!(
                "{}: missing input `{name}` (supply it with --input {name}=…)",
                path.display()
            );
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("{}: runtime error: {e}", path.display());
            ExitCode::FAILURE
        }
    }
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
    let diagnostic_path = e.path.as_deref().unwrap_or(path);
    let diagnostic_source = match load_diagnostic_source(path, source, e.path.as_deref()) {
        Ok(source) => source,
        Err(read_err) => {
            eprintln!(
                "{}: failed to read source for diagnostic: {read_err}",
                diagnostic_path.display()
            );
            eprintln!("{}: {}", diagnostic_path.display(), e);
            return ExitCode::FAILURE;
        }
    };
    let (line, column) = e
        .span
        .doc_position(diagnostic_source.as_ref(), DIAGNOSTIC_TAB_SIZE);
    let headline = format_diagnostic_message(diagnostic_path, line, column, &e.to_string());
    emit_error_report(
        diagnostic_path,
        diagnostic_source.as_ref(),
        Range::from(e.span),
        &headline,
        &e.to_string(),
    );
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
                let (line, column) = e.span.doc_position(&source, DIAGNOSTIC_TAB_SIZE);
                let headline = format_diagnostic_message(path, line, column, &e.message);
                emit_error_report(path, &source, Range::from(e.span), &headline, &e.message);
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

fn load_diagnostic_source<'a>(
    root_path: &'a Path,
    root_source: &'a str,
    diagnostic_path: Option<&Path>,
) -> std::io::Result<Cow<'a, str>> {
    match diagnostic_path {
        Some(path) if path != root_path => Ok(Cow::Owned(fs::read_to_string(path)?)),
        _ => Ok(Cow::Borrowed(root_source)),
    }
}

fn format_diagnostic_message(path: &Path, line: usize, column: usize, message: &str) -> String {
    format!("{}:{line}:{column}: {message}", path.display())
}

fn emit_error_report(
    path: &Path,
    source: &str,
    span: Range<usize>,
    headline: &str,
    label_message: &str,
) {
    let filename = path.display().to_string();
    Report::build(ReportKind::Error, (&filename, span.clone()))
        .with_message(headline)
        .with_label(Label::new((&filename, span)).with_message(label_message))
        .finish()
        .eprint((&filename, Source::from(source)))
        .ok();
}
