//! The OpenQASM 3 compiler: source text → a [`Program`] the `oqi-vm`
//! crate runs.
//!
//! [`compile`] runs the whole pipeline (`parse → resolve includes → lower →
//! cfg → bytecode`) with the defaults; [`Compiler`] sets the include
//! resolver, the compile options and a timing table. Errors are
//! [`CompileError`]s, which render as source-annotated diagnostics through
//! `oqi-diagnostics`.
//!
//! ```
//! let program = oqi_compile::compile(
//!     r#"include "stdgates.inc";
//!        input int n;
//!        qubit[2] q;
//!        h q[0];
//!        cx q[0], q[1];
//!        bit[2] c = measure q;"#,
//! )?;
//! assert_eq!(program.num_qubits(), 2);
//! assert!(program.input("n").is_some());
//!
//! // A compiled program can be saved and loaded again.
//! let bytes = program.to_bytes()?;
//! let loaded = oqi_compile::Program::from_bytes(&bytes)?;
//! assert_eq!(loaded.to_string(), program.to_string());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! This crate is a building block of [`oqi`](https://docs.rs/oqi), the
//! OpenQASM 3 compiler and simulator, which re-exports its supported API;
//! most users want that crate.
//!
//! The intermediate representations ([`sir`], the bytecode instruction set
//! in [`bytecode`]) are public for the VM and for debugging output, but are
//! not a stable API.
#![warn(missing_docs)]

pub mod bytecode;
mod cfg;
#[cfg(test)]
mod cfg_tests;
mod diagnostic;
mod display;
#[doc(hidden)]
pub mod duration;
mod error;
mod lower;
#[doc(hidden)]
pub mod openpulse;
mod qubits;
mod register;
mod resolve;
#[doc(hidden)]
pub mod scope;
#[doc(hidden)]
pub mod sir;
#[doc(hidden)]
pub mod symbol;
#[doc(hidden)]
pub mod types;

use std::path::PathBuf;

pub use oqi_classical as classical;
pub use oqi_lex::Span;

pub use bytecode::Program;
pub use duration::{TableTimings, TimingDuration};
pub use error::{CompileError, ErrorKind, Result};
pub use resolve::{DefaultIncludeResolver, IncludeResolver, LibOnlyResolver};
pub use symbol::{SymbolId, SymbolKind};
pub use types::{CompileOptions, SystemWidth, parse_dt};

use classical::Duration;

/// Compile `source` with the defaults: includes are read from disk (and
/// `stdgates.inc` from the embedded standard library), plain `int`/`float`
/// are 64 bits, one `dt` is 1 µs, and `durationof` is left to the VM.
///
/// # Errors
///
/// A [`CompileError`] for the first syntax, name, type or other error in
/// the program.
pub fn compile(source: &str) -> Result<Program> {
    Compiler::new().compile(source)
}

/// Configures and runs a compilation.
///
/// ```
/// use oqi_compile::{Compiler, LibOnlyResolver, TableTimings};
///
/// let program = Compiler::new()
///     .include_resolver(&LibOnlyResolver)
///     .timings(TableTimings::new())
///     .compile("include \"stdgates.inc\"; qubit q; h q;")?;
/// assert_eq!(program.num_qubits(), 1);
/// # Ok::<(), oqi_compile::CompileError>(())
/// ```
pub struct Compiler<'a> {
    include_resolver: &'a dyn IncludeResolver,
    options: CompileOptions,
    timings: Option<TableTimings>,
}

impl Default for Compiler<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Compiler<'a> {
    /// A compiler with the defaults of [`compile`].
    pub fn new() -> Self {
        Self {
            include_resolver: &DefaultIncludeResolver,
            options: CompileOptions::default(),
            timings: None,
        }
    }

    /// Where `include` statements get their text.
    pub fn include_resolver(mut self, resolver: &'a dyn IncludeResolver) -> Self {
        self.include_resolver = resolver;
        self
    }

    /// Replace every compile option at once.
    pub fn options(mut self, options: CompileOptions) -> Self {
        self.options = options;
        self
    }

    /// The path of the source being compiled: relative includes resolve
    /// against it, and diagnostics name it.
    pub fn source_name(mut self, path: impl Into<PathBuf>) -> Self {
        self.options.source_name = Some(path.into());
        self
    }

    /// The width of plain `int`, `uint`, `float`, `complex` and `angle`.
    pub fn system_width(mut self, width: SystemWidth) -> Self {
        self.options.system_width = width;
        self
    }

    /// The duration of one `dt`, the device time unit (see [`parse_dt`]).
    pub fn dt(mut self, dt: Duration) -> Self {
        self.options.dt = dt;
        self
    }

    /// Resolve `durationof` and `stretch` at compile time from `timings`,
    /// together with the program's own `defcal` and gate bodies. Without a
    /// table they are left to the VM, which times calibrated pulses and
    /// takes other gates as instantaneous.
    pub fn timings(mut self, timings: TableTimings) -> Self {
        self.timings = Some(timings);
        self
    }

    /// The options this compiler uses.
    pub fn compile_options(&self) -> &CompileOptions {
        &self.options
    }

    /// Compile `source` to a [`Program`].
    ///
    /// # Errors
    ///
    /// A [`CompileError`] for the first error in the program.
    pub fn compile(&self, source: &str) -> Result<Program> {
        bytecode::emit(&self.compile_to_sir(source)?)
    }

    /// Compile `source` only as far as the structured IR, a debugging aid
    /// (its `Display` is what `oqi compile --dump-sir` prints);
    /// [`bytecode::emit`] lowers it the rest of the way.
    ///
    /// # Errors
    ///
    /// A [`CompileError`] for the first error in the program.
    pub fn compile_to_sir(&self, source: &str) -> Result<sir::Program> {
        lower::compile_to_sir(
            source,
            self.include_resolver,
            &self.options,
            self.timings.as_ref(),
        )
    }
}
