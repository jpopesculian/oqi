use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use ariadne::{Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};
use oqi_format::Config;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[derive(Parser)]
#[command(name = "oqi")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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
        Command::Fmt {
            compact,
            stdout,
            paths,
        } => fmt(compact, stdout, &paths),
    }
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
                let filename = path.display().to_string();
                Report::build(ReportKind::Error, (&filename, e.span.clone()))
                    .with_message(&e.message)
                    .with_label(Label::new((&filename, e.span)).with_message(&e.message))
                    .finish()
                    .eprint((&filename, Source::from(&source)))
                    .ok();
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
