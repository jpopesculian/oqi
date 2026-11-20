use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::{Parser, Subcommand};
use oqi_compile::classical::ValueTy;
use oqi_compile::{BcModule, CompileOptions, DefaultIncludeResolver};
use oqi_format::Config;
use oqi_vm::{
    BackendSpec, DEFAULT_SEED, HostValue, NoExterns, Precision, QuantumBackend, SumPolicy, Vm,
};
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
struct Timing {
    options: CompileOptions,
    entries: Option<Vec<(String, String)>>,
}

impl TimingArgs {
    /// Compile options honoring the profile's "dt" and `--dt`, and the
    /// timing table (file entries first, so `--timing` flags override), or
    /// a reported failure.
    fn resolve(&self, path: &Path) -> Result<Timing, ExitCode> {
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

        let mut options = CompileOptions {
            source_name: Some(path.to_path_buf()),
            ..Default::default()
        };
        if let Some(spec) = file.remove("dt") {
            options.dt =
                oqi_compile::parse_dt(&spec).map_err(|e| fail(format!("--timings dt: {e}")))?;
        }
        if let Some(spec) = &self.dt {
            options.dt = oqi_compile::parse_dt(spec).map_err(|e| fail(format!("--dt: {e}")))?;
        }

        if !(self.resolve_durations || !self.timing.is_empty() || self.timings_file.is_some()) {
            return Ok(Timing {
                options,
                entries: None,
            });
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
        Ok(Timing {
            options,
            entries: Some(entries),
        })
    }
}

#[derive(Subcommand)]
enum Command {
    /// Compile an OpenQASM file
    Compile {
        /// The OpenQASM file to compile
        path: PathBuf,

        /// Dump the bytecode disassembly to stdout
        #[arg(long)]
        dump: bool,

        /// Dump the structured IR (before bytecode emission) to stdout
        #[arg(long)]
        dump_sir: bool,

        #[command(flatten)]
        timing: TimingArgs,
    },

    /// Compile and run an OpenQASM file on a simulator
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
    /// The OpenQASM file to run
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
    /// fallback
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
            dump,
            dump_sir,
            timing,
        } => compile(&path, dump, dump_sir, &timing),
        Command::Run(args) => pollster::block_on(run(&args)),
        Command::Fmt {
            compact,
            stdout,
            paths,
        } => fmt(compact, stdout, &paths),
    }
}

/// Read `path`, or report why it can't be read.
fn read_source(path: &Path) -> Result<String, ExitCode> {
    fs::read_to_string(path).map_err(|e| {
        eprintln!("{}: {e}", path.display());
        ExitCode::FAILURE
    })
}

/// The timing entries as the borrowed pairs the compiler takes.
fn borrowed(entries: &Option<Vec<(String, String)>>) -> Option<Vec<(&str, &str)>> {
    entries
        .as_ref()
        .map(|e| e.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect())
}

fn compile(path: &Path, dump: bool, dump_sir: bool, timing: &TimingArgs) -> ExitCode {
    let (source, timing) = match read_source(path).and_then(|s| Ok((s, timing.resolve(path)?))) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let fail = |e: oqi_compile::CompileError| {
        oqi_diagnostics::emit(&e, path, &source);
        ExitCode::FAILURE
    };

    let program = match oqi_compile::compile_source(
        &source,
        DefaultIncludeResolver,
        &timing.options,
        borrowed(&timing.entries).as_deref(),
    ) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    if dump_sir {
        print!("{program}");
    }
    let module = match oqi_compile::bytecode::emit(&program) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    if dump {
        print!("{module}");
    }
    ExitCode::SUCCESS
}

async fn run(args: &RunArgs) -> ExitCode {
    let path = args.path.as_path();
    let (source, timing) = match read_source(path).and_then(|s| Ok((s, args.timing.resolve(path)?)))
    {
        Ok(v) => v,
        Err(code) => return code,
    };
    let fail = |e: &dyn oqi_diagnostics::Diagnostic| {
        oqi_diagnostics::emit(e, path, &source);
        ExitCode::FAILURE
    };

    let module = match oqi_compile::compile(
        &source,
        DefaultIncludeResolver,
        &timing.options,
        borrowed(&timing.entries).as_deref(),
    ) {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    let inputs = match parse_inputs(&module, &args.input) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let seed = args.seed.unwrap_or(DEFAULT_SEED);
    let spec = BackendSpec {
        precision: args.precision.unwrap_or_default(),
        seed,
        sum_policy: SumPolicy {
            max_rank: args.max_rank,
            ..args.backend.sum_policy
        },
        ..args.backend
    };
    let (sim, built) = match spec.build(module.qubits.num_qubits).await {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    if let Some(asked) = args.precision
        && asked != built.precision
    {
        eprintln!(
            "note: the {} backend runs at {}; --precision {asked} ignored",
            built.name(),
            built.precision
        );
    }
    let mut vm = Vm::new(&module, sim, NoExterns);

    if let Some(shots) = args.shots {
        let results = match oqi_vm::coerce_inputs(&module, &inputs) {
            Ok(inputs) => vm.sample(inputs, shots, seed).await,
            Err(e) => Err(e),
        };
        return match results {
            Ok(results) => {
                for histogram in oqi_vm::histograms(&module, &results) {
                    println!("{}:", histogram.name);
                    for (label, count) in &histogram.counts {
                        println!("  {label}: {count}");
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => fail(&e),
        };
    }

    match vm.run_with_named_inputs(&inputs).await {
        Ok(result) => {
            for (qubit, bit) in &result.measurements {
                println!("q[{qubit}] = {}", if *bit { 1 } else { 0 });
            }
            for (sym, value) in &result.outputs {
                println!("{} = {value}", module.symbol(*sym).name);
            }
            if args.state {
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
        Err(e) => fail(&e),
    }
}

/// Split `--input NAME=VALUE` specs into named host values for the VM to
/// coerce to each input's declared type. An array input takes a
/// comma-separated list, optionally bracketed (`1,2,3` or `[1,2,3]`), filled
/// row-major for multi-dimensional arrays.
fn parse_inputs(module: &BcModule, specs: &[String]) -> Result<Vec<(String, HostValue)>, String> {
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
