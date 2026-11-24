#![doc = include_str!("../README.md")]
//!
//! # The API
//!
//! - [`compile`] a source string (or configure a [`Compiler`]) into a
//!   [`Program`]; [`Program::to_bytes`] / [`Program::from_bytes`] store it
//!   as bytecode.
//! - Run it with a [`Runner`]: pick a backend by name ([`BackendSpec`]),
//!   give [`HostValue`] inputs by name, implement `extern` functions
//!   ([`ExternProvider`], [`FnRegistry`]), then
//!   [`run`](Runner::run) once or [`sample`](Runner::sample) many shots.
//!   Running is `async` (a GPU backend awaits its device); with the
//!   `blocking` feature, `run_blocking` and `sample_blocking` wait for you.
//! - Every step's error converts into [`Error`], which renders as a
//!   source-annotated [`Diagnostic`] ([`render_to_string`], [`emit`]).
//! - [`parse`] gives the syntax tree; [`format()`](fn@format) pretty-prints
//!   source, configured by [`FormatConfig`] (that is the formatter's whole
//!   API, so it has no module of its own).
//!
//! The building blocks are available as modules for anything beyond that:
//! [`syntax`] (the parser and AST), [`compiler`], [`vm`] (the interpreter,
//! [`vm::Vm`], and backends), [`classical`] (classical values and types) and
//! [`diagnostics`]. Their items not re-exported here are less stable.
//!
//! # More examples
//!
//! ## Choose a backend by name, or build one yourself
//!
//! ```
//! # fn main() -> oqi::Result<()> {
//! use std::num::NonZeroUsize;
//! use oqi::{MpsSim, NoExterns, Runner};
//!
//! let program = oqi::compile(
//!     r#"include "stdgates.inc"; qubit[3] q; h q[0]; cx q[0], q[1]; cx q[1], q[2]; bit[3] c = measure q;"#,
//! )?;
//!
//! // By name: see `BackendSpec::NAMES`. Fields refine it.
//! let mut spec: oqi::BackendSpec = "mps".parse()?;
//! spec.max_bond = 16;
//! let result = Runner::new(&program).backend(spec).run_blocking()?;
//! assert!(["000", "111"].contains(&result.output("c").unwrap().to_string().trim_matches('"')));
//!
//! // By type: construct the backend and drive the interpreter directly.
//! let backend = MpsSim::new(program.num_qubits())?
//!     .with_max_bond(NonZeroUsize::new(16).unwrap())
//!     .with_seed(7);
//! let mut vm = oqi::vm::Vm::new(&program, backend, NoExterns);
//! let result = oqi::block_on(vm.run())?;
//! assert_eq!(vm.backend().truncation_error(), 0.0);
//! # let _ = result;
//! # Ok(())
//! # }
//! ```
//!
//! ## Implement an `extern` function
//!
//! ```
//! # fn main() -> oqi::Result<()> {
//! use oqi::classical::IntWidth;
//! use oqi::{FnRegistry, HostValue, Runner, Value};
//!
//! let program = oqi::compile("extern double(int) -> int; int y = double(21);")?;
//!
//! let mut externs = FnRegistry::new();
//! externs.register("double", |args: &[Value]| match HostValue::from(&args[0]) {
//!     HostValue::Int(n) => Ok(Some(Value::int(2 * n, IntWidth::B64))),
//!     other => panic!("not an int: {other:?}"),
//! });
//! let result = Runner::new(&program).externs(externs).run_blocking()?;
//! assert_eq!(result.output("y").unwrap().to_string(), "42");
//! # Ok(())
//! # }
//! ```
//!
//! ## Report an error against its source
//!
//! ```
//! use std::path::Path;
//!
//! let source = "qubit q;\nfoo q;\n";
//! let error: oqi::Error = oqi::compile(source).unwrap_err().into();
//! // `included` supplies the text of other files an error may point into.
//! let report = oqi::render_to_string(&error, Path::new("main.qasm"), source, &|_| None);
//! assert!(report.contains("C0001"));
//! assert!(report.contains("main.qasm:2:1: undefined name 'foo'"));
//! ```
//!
//! ## Format source
//!
//! ```
//! # fn main() -> oqi::Result<()> {
//! let tidy = oqi::format("qubit q;h  q ;", oqi::FormatConfig::default())?;
//! assert_eq!(tidy, "qubit q;\nh q;\n");
//! # Ok(())
//! # }
//! ```
//!
//! ## Store a program as bytecode
//!
//! ```
//! # fn main() -> oqi::Result<()> {
//! let program = oqi::compile("bit c = 1;")?;
//! let bytes = program.to_bytes()?;
//! let loaded = oqi::Program::from_bytes(&bytes)?;
//! let result = oqi::Runner::new(&loaded).run_blocking()?;
//! assert_eq!(result.output("c").unwrap().to_string(), "1");
//! # Ok(())
//! # }
//! ```
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

mod error;

pub use error::{Error, Result};

/// The parser and the syntax tree (the `oqi-parse` crate).
pub use oqi_parse as syntax;

/// The compiler (the `oqi-compile` crate).
pub use oqi_compile as compiler;

/// The interpreter and the simulators (the `oqi-vm` crate).
pub use oqi_vm as vm;

/// Classical values and types (the `oqi-classical` crate).
pub use oqi_compile::classical;

/// Diagnostic codes and rendering (the `oqi-diagnostics` crate).
pub use oqi_diagnostics as diagnostics;

pub use oqi_compile::{Compiler, Program, Span, compile};
pub use oqi_diagnostics::{Diagnostic, emit, render_to_string};
pub use oqi_format::{Config as FormatConfig, format};
pub use oqi_parse::parse;
pub use oqi_vm::{
    AutoSim, BackendKind, BackendSpec, Complex64, ExternProvider, FnRegistry, FrameHandle,
    GateModifiers, Histogram, HostValue, NoExterns, OpaqueCalHandler, OpenPulseHandler, PortHandle,
    Precision, QuantumBackend, RunResult, Runner, Samples, SimdSim, StateVectorSim, SumPolicy,
    Value, ValueTy, WaveformHandle, async_trait, u_matrix,
};

#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
pub use oqi_vm::GpuSim;
#[cfg(feature = "mps")]
#[cfg_attr(docsrs, doc(cfg(feature = "mps")))]
pub use oqi_vm::MpsSim;

/// Block the calling thread on a future: for driving [`vm::Vm`] (or any
/// other `async` API here) without an async runtime. Native only — in the
/// browser, `.await` instead.
#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub use pollster::block_on;
