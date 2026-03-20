use std::borrow::Cow;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use ariadne::{Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand};
use oqi_compile::resolve::StdFileResolver;
use oqi_format::Config;
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

    let program = match oqi_compile::lower::compile_source(&source, StdFileResolver, Some(path)) {
        Ok(p) => p,
        Err(e) => {
            let diagnostic_path = e.path.as_deref().unwrap_or(path);
            let diagnostic_source = match load_diagnostic_source(path, &source, e.path.as_deref()) {
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
            return ExitCode::FAILURE;
        }
    };

    if dump {
        print!("{program}");
    }

    ExitCode::SUCCESS
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
