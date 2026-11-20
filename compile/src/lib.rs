//! The OpenQASM 3 compiler: source → SIR → bytecode.
//!
//! [`compile`] is the whole pipeline (`parse → resolve → lower → cfg → ssa →
//! bytecode`). [`compile_source`] stops at the structured IR
//! ([`sir::Program`]), which [`bytecode::emit`] lowers the rest of the way.

pub mod bytecode;
mod cfg;
#[cfg(test)]
mod cfg_tests;
pub mod classical;
mod diagnostic;
mod display;
pub mod duration;
mod error;
mod lower;
mod openpulse;
mod qubits;
mod resolve;
mod scope;
pub mod sir;
mod ssa;
pub mod symbol;
pub mod types;

pub use bytecode::BcModule;
pub use duration::TableTimings;
pub use error::{CompileError, ErrorKind, Result};
pub use lower::compile_source;
pub use resolve::{DefaultIncludeResolver, IncludeResolver, LibOnlyResolver};
pub use types::{CompileOptions, SystemWidth, parse_dt};

/// Source → bytecode. See [`compile_source`] for `timings`.
pub fn compile(
    source: &str,
    include_resolver: impl IncludeResolver + 'static,
    options: &CompileOptions,
    timings: Option<&[(&str, &str)]>,
) -> Result<BcModule> {
    bytecode::emit(&compile_source(source, include_resolver, options, timings)?)
}
