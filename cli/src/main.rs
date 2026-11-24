use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::{Parser, Subcommand};
use oqi_compile::classical::ValueTy;
use oqi_compile::{Compiler, DefaultIncludeResolver, IncludeResolver, Program, TableTimings};
use oqi_format::Config;
use oqi_vm::{BackendSpec, DEFAULT_SEED, HostValue, Precision, Runner, SumPolicy};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

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

/// What [`TimingArgs`] resolves to: the compile options, plus the timing
/// entries when compile-time `durationof` resolution was requested.
impl TimingArgs {
    /// Whether any timing option was given.
    fn any(&self) -> bool {
        !self.timing.is_empty()
            || self.dt.is_some()
            || self.resolve_durations
            || self.timings_file.is_some()
    }

    /// A compiler honoring the profile's "dt" and `--dt`, and the timing
    /// table (file entries first, so `--timing` flags override), or a
    /// reported failure.
    fn compiler(&self, path: &Path) -> Result<Compiler<'static>, ExitCode> {
        let fail = |msg: String| {
            eprintln!("{msg}");
            ExitCode::FAILURE
        };
        let mut file: BTreeMap<String, String> = match &self.timings_file {
            None => BTreeMap::new(),
            Some(file) => {
                let text = fs::read_to_string(file)
                    .map_err(|e| fail(format!("--timings {}: {e}", file.display())))?;
                serde_json::from_str(&text).map_err(|e| {
                    fail(format!(
                        "--timings {}: expected a flat JSON object of name → duration literal: {e}",
                        file.display()
                    ))
                })?
            }
        };

        let mut compiler = Compiler::new().source_name(path);
        if let Some(spec) = file.remove("dt") {
            let dt =
                oqi_compile::parse_dt(&spec).map_err(|e| fail(format!("--timings dt: {e}")))?;
            compiler = compiler.dt(dt);
        }
        if let Some(spec) = &self.dt {
            let dt = oqi_compile::parse_dt(spec).map_err(|e| fail(format!("--dt: {e}")))?;
            compiler = compiler.dt(dt);
        }

        if !(self.resolve_durations || !self.timing.is_empty() || self.timings_file.is_some()) {
            return Ok(compiler);
        }
        let mut entries: Vec<(String, String)> = file.into_iter().collect();
        for spec in &self.timing {
            let (name, dur) = spec.split_once('=').ok_or_else(|| {
                fail(format!(
                    "invalid --timing `{spec}` (expected NAME=DURATION)"
                ))
            })?;
            entries.push((name.to_string(), dur.to_string()));
        }
        let entries = entries.iter().map(|(n, d)| (n.as_str(), d.as_str()));
        let table = TableTimings::from_str_entries(entries, &compiler.compile_options().dt)
            .map_err(|e| {
                oqi_diagnostics::emit(&e, path, "", &included_file);
                ExitCode::FAILURE
            })?;
        Ok(compiler.timings(table))
    }
}

#[derive(Subcommand)]
enum Command {
    /// Compile an OpenQASM file, checking it, and write or print the
    /// bytecode
    Compile {
        /// The OpenQASM file to compile, or a bytecode file to disassemble
        /// with --dump (bytecode is recognized by its header, whatever the
        /// file is called)
        path: PathBuf,

        /// Write the bytecode to this file; `oqi run` runs it
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Dump the bytecode disassembly to stdout
        #[arg(long)]
        dump: bool,

        /// Dump the structured IR (before bytecode emission) to stdout
        #[arg(long)]
        dump_sir: bool,

        #[command(flatten)]
        timing: TimingArgs,
    },

    /// Compile and run an OpenQASM file, or run a bytecode file, on a
    /// simulator
    Run(RunArgs),

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

#[derive(clap::Args)]
struct RunArgs {
    /// The OpenQASM file to run, or a bytecode file written by `oqi compile
    /// -o` (recognized by its header, whatever the file is called)
    path: PathBuf,

    /// Print the final state vector after the run
    #[arg(long, conflicts_with = "shots")]
    state: bool,

    /// Quantum execution backend. `scalar`/`simd` are single-threaded CPU
    /// state-vector simulators and `rayon`/`rayon-simd` their
    /// multi-threaded variants; `gpu` is the WebGPU simulator (single
    /// precision; needs the `gpu` build feature and an adapter) and
    /// `prefer-gpu` falls back to `scalar` without one; `auto` routes
    /// through a stabilizer tableau, then an exact sum over Cliffords,
    /// then a dense state vector; `sum` is `auto` without the dense
    /// fallback; `mps` is a matrix-product-state simulator, approximate once
    /// entanglement exceeds --max-bond
    #[arg(
        long,
        default_value = "scalar",
        value_parser = PossibleValuesParser::new(BackendSpec::NAMES)
            .try_map(|name| name.parse::<BackendSpec>())
    )]
    backend: BackendSpec,

    /// Amplitude precision of the CPU state-vector simulators [default: f64]
    #[arg(
        long,
        value_parser = PossibleValuesParser::new(["f64", "f32"])
            .try_map(|name| name.parse::<Precision>())
    )]
    precision: Option<Precision>,

    /// Maximum stabilizer terms for the auto/sum backends' exact
    /// sum-over-Cliffords tier (each non-Clifford gate multiplies the
    /// term count by 2-3)
    #[arg(long, default_value_t = SumPolicy::default().max_rank)]
    max_rank: usize,

    /// Maximum bond dimension of the mps backend: past it, the state is
    /// truncated to its largest singular values and renormalized
    #[arg(long, default_value_t = BackendSpec::default().max_bond)]
    max_bond: usize,

    /// Simulator RNG seed; a given seed reproduces a run
    #[arg(long)]
    seed: Option<u64>,

    /// Run the program this many times and print, per output variable, how
    /// often each value was observed
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    shots: Option<u32>,

    /// Supply a value for a declared input (repeatable). Angles are
    /// radians, `bit[n]` an MSB-first bit string, and arrays a
    /// comma-separated list, optionally bracketed: `NAME=1,2,3`.
    #[arg(long = "input", value_name = "NAME=VALUE")]
    input: Vec<String>,

    #[command(flatten)]
    timing: TimingArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Compile {
            path,
            output,
            dump,
            dump_sir,
            timing,
        } => compile(&path, output.as_deref(), dump, dump_sir, &timing),
        Command::Run(args) => pollster::block_on(run(&args)),
        Command::Fmt {
            compact,
            stdout,
            paths,
        } => fmt(compact, stdout, &paths),
    }
}

/// What a file given on the command line holds.
enum Input {
    /// OpenQASM source text.
    Source(String),
    /// A program compiled to bytecode.
    Bytecode(Box<Program>),
}

/// Read `path` as bytecode or source, or report why it can't be read.
fn read_input(path: &Path) -> Result<Input, ExitCode> {
    let fail = |why: String| {
        eprintln!("{}: {why}", path.display());
        ExitCode::FAILURE
    };
    let bytes = fs::read(path).map_err(|e| fail(e.to_string()))?;
    if Program::is_bytecode(&bytes) {
        return Program::from_bytes(&bytes)
            .map(|program| Input::Bytecode(Box::new(program)))
            .map_err(|e| fail(e.to_string()));
    }
    String::from_utf8(bytes)
        .map(Input::Source)
        .map_err(|_| fail("neither OpenQASM source (not UTF-8) nor oqi bytecode".into()))
}

/// Reject timing options for bytecode: its durations were resolved when it
/// was compiled.
fn no_timings_for_bytecode(path: &Path, timing: &TimingArgs) -> Result<(), ExitCode> {
    if timing.any() {
        eprintln!(
            "{}: timing options apply when compiling source; this file is bytecode, \
             compiled already",
            path.display()
        );
        return Err(ExitCode::FAILURE);
    }
    Ok(())
}

/// The timing entries as the borrowed pairs the compiler takes.
/// The text of a file a diagnostic points into, read the way `include`
/// read it.
fn included_file(path: &Path) -> Option<String> {
    DefaultIncludeResolver
        .resolve_path(path)
        .ok()
        .map(|text| text.into_owned())
}

fn compile(
    path: &Path,
    output: Option<&Path>,
    dump: bool,
    dump_sir: bool,
    timing: &TimingArgs,
) -> ExitCode {
    let module = match read_input(path) {
        Err(code) => return code,
        Ok(Input::Bytecode(module)) => {
            if let Err(code) = no_timings_for_bytecode(path, timing) {
                return code;
            }
            if dump_sir {
                eprintln!(
                    "{}: bytecode keeps no structured IR; --dump-sir needs the source",
                    path.display()
                );
                return ExitCode::FAILURE;
            }
            *module
        }
        Ok(Input::Source(source)) => {
            let compiler = match timing.compiler(path) {
                Ok(compiler) => compiler,
                Err(code) => return code,
            };
            let fail = |e: oqi_compile::CompileError| {
                oqi_diagnostics::emit(&e, path, &source, &included_file);
                ExitCode::FAILURE
            };
            let program = match compiler.compile_to_sir(&source) {
                Ok(p) => p,
                Err(e) => return fail(e),
            };
            if dump_sir {
                print!("{program}");
            }
            match oqi_compile::bytecode::emit(&program) {
                Ok(m) => m,
                Err(e) => return fail(e),
            }
        }
    };
    if dump {
        print!("{module}");
    }
    if let Some(output) = output {
        let written = module
            .to_bytes()
            .map_err(|e| e.to_string())
            .and_then(|bytes| fs::write(output, bytes).map_err(|e| e.to_string()));
        if let Err(why) = written {
            eprintln!("{}: {why}", output.display());
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

async fn run(args: &RunArgs) -> ExitCode {
    let path = args.path.as_path();
    // Bytecode keeps the spans of its source but not the text: its errors
    // are reported by message alone.
    let (source, compiled) = match read_input(path) {
        Err(code) => return code,
        Ok(Input::Bytecode(module)) => match no_timings_for_bytecode(path, &args.timing) {
            Ok(()) => (String::new(), Some(*module)),
            Err(code) => return code,
        },
        Ok(Input::Source(source)) => (source, None),
    };
    let fail = |e: &dyn oqi_diagnostics::Diagnostic| {
        oqi_diagnostics::emit(e, path, &source, &included_file);
        ExitCode::FAILURE
    };
    // A runtime error, plus how to act on it from this command line.
    let fail_run = |e: &oqi_vm::Error| {
        let code = fail(e);
        match e.kind {
            oqi_vm::ErrorKind::MissingInput(_) => {
                eprintln!("hint: supply it with `--input NAME=VALUE`")
            }
            oqi_vm::ErrorKind::RankOverflow { .. } => {
                eprintln!("hint: raise the term budget with `--max-rank`")
            }
            _ => {}
        }
        code
    };

    let module = match compiled {
        Some(module) => module,
        None => {
            let compiled = args
                .timing
                .compiler(path)
                .and_then(|compiler| compiler.compile(&source).map_err(|e| fail(&e)));
            match compiled {
                Ok(module) => module,
                Err(code) => return code,
            }
        }
    };
    let inputs = match parse_inputs(&module, &args.input) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let mut spec = args.backend;
    spec.precision = args.precision.unwrap_or_default();
    spec.sum_policy.max_rank = args.max_rank;
    spec.max_bond = args.max_bond;
    let runner = Runner::new(&module)
        .backend(spec)
        .seed(args.seed.unwrap_or(DEFAULT_SEED))
        .inputs(inputs);
    // Say when the backend ran at another precision than the one asked for.
    let note_precision = |built: &BackendSpec| {
        if let Some(asked) = args.precision
            && asked != built.precision
        {
            eprintln!(
                "note: the {} backend runs at {}; --precision {asked} ignored",
                built.name(),
                built.precision
            );
        }
    };

    if let Some(shots) = args.shots {
        return match runner.sample(shots).await {
            Ok(samples) => {
                note_precision(samples.backend());
                for histogram in samples.histograms() {
                    println!("{}:", histogram.name());
                    for (label, count) in histogram.counts() {
                        println!("  {label}: {count}");
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => fail_run(&e),
        };
    }

    match runner.statevector(args.state).run().await {
        Ok(result) => {
            if let Some(built) = result.backend() {
                note_precision(built);
            }
            for (qubit, bit) in result.measurements() {
                println!("q[{qubit}] = {}", if *bit { 1 } else { 0 });
            }
            for (name, value) in result.outputs() {
                println!("{name} = {value}");
            }
            let width = module.num_qubits() as usize;
            for (i, amp) in result.statevector().unwrap_or_default().iter().enumerate() {
                if amp.norm() > 1e-12 {
                    println!("|{i:0width$b}> = {amp}");
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail_run(&e),
    }
}

/// Split `--input NAME=VALUE` specs into named host values for the VM to
/// coerce to each input's declared type. An array input takes a
/// comma-separated list, optionally bracketed (`1,2,3` or `[1,2,3]`), filled
/// row-major for multi-dimensional arrays.
fn parse_inputs(module: &Program, specs: &[String]) -> Result<Vec<(String, HostValue)>, String> {
    specs
        .iter()
        .map(|spec| {
            let (name, raw) = spec
                .split_once('=')
                .ok_or_else(|| format!("invalid --input `{spec}` (expected NAME=VALUE)"))?;
            let value = match module.input(name) {
                Some((_, ValueTy::Array(_))) => {
                    let inner = raw.trim();
                    let inner = inner
                        .strip_prefix('[')
                        .and_then(|s| s.strip_suffix(']'))
                        .unwrap_or(inner)
                        .trim();
                    let items = inner.split(',').filter(|_| !inner.is_empty());
                    HostValue::List(
                        items
                            .map(|item| HostValue::Text(item.trim().to_string()))
                            .collect(),
                    )
                }
                // Scalars, and unknown names (which the VM rejects).
                _ => HostValue::Text(raw.to_string()),
            };
            Ok((name.to_string(), value))
        })
        .collect()
}

fn fmt(compact: bool, stdout: bool, paths: &[PathBuf]) -> ExitCode {
    let config = if compact {
        Config { compact: true }
    } else {
        Config::default()
    };

    // Format in parallel; report and print in the order the files were given.
    let results: Vec<_> = paths
        .par_iter()
        .map(|path| -> std::io::Result<_> {
            let source = fs::read_to_string(path)?;
            let formatted = oqi_format::format(&source, config);
            Ok((source, formatted))
        })
        .collect();

    let mut had_errors = false;
    for (path, result) in paths.iter().zip(results) {
        let (source, formatted) = match result {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                had_errors = true;
                continue;
            }
        };
        let formatted = match formatted {
            Ok(s) => s,
            Err(e) => {
                oqi_diagnostics::emit(&e, path, &source, &|_| None);
                had_errors = true;
                continue;
            }
        };
        if stdout {
            print!("{formatted}");
        } else if let Err(e) = fs::write(path, &formatted) {
            eprintln!("{}: {e}", path.display());
            had_errors = true;
        }
    }

    if had_errors {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
