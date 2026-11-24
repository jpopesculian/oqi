//! Runs compiled OpenQASM 3 programs ([`Program`]s from `oqi-compile`) on
//! simulated or real quantum backends.
//!
//! [`Runner`] is the usual way in: pick a backend by name with
//! [`BackendSpec`], give inputs by name, then [`run`](Runner::run) once or
//! [`sample`](Runner::sample) many shots.
//!
//! ```
//! use oqi_vm::Runner;
//!
//! let program = oqi_compile::compile(
//!     r#"include "stdgates.inc";
//!        qubit[2] q;
//!        h q[0];
//!        cx q[0], q[1];
//!        bit[2] c = measure q;"#,
//! )?;
//! let samples = pollster::block_on(Runner::new(&program).seed(7).sample(1000))?;
//! let counts = samples.histogram("c").unwrap().counts();
//! assert!(counts.iter().all(|(label, _)| label == "00" || label == "11"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! This crate is a building block of [`oqi`](https://docs.rs/oqi), the
//! OpenQASM 3 compiler and simulator, which re-exports its supported API
//! and adds blocking helpers by default; most users want that crate.
//!
//! Running is `async` so a GPU or remote backend can await its device (in
//! the browser, nothing may block); with the CPU simulators every future is
//! ready at once, so native callers block on it (e.g. with `pollster`).
//!
//! # Extension points
//!
//! - [`QuantumBackend`]: the quantum execution layer. The bundled ones are
//!   the state-vector simulators [`StateVectorSim`] and [`SimdSim`], the
//!   stabilizer/sum-over-Cliffords router [`AutoSim`], the
//!   matrix-product-state simulator `MpsSim` (feature `mps`, on by default)
//!   and the WebGPU simulator `GpuSim` (feature `gpu`). A backend
//!   implements only the built-in `U` and `gphase`, measurement and reset:
//!   the compiler lowers every gate to those.
//! - [`ExternProvider`]: host implementations of `extern` functions
//!   ([`FnRegistry`] for closures).
//! - [`OpenPulseHandler`] and [`OpaqueCalHandler`]: calibration execution.
//!
//! [`Vm`] is the interpreter itself, for when a [`Runner`] isn't enough: a
//! backend instance of your own, or calibration handlers.
//!
//! Host languages exchange values with the VM as [`HostValue`]s.
//!
//! # Features
//!
//! - `parallel` (default): multi-threaded state-vector kernels.
//! - `mps` (default): the matrix-product-state simulator.
//! - `gpu`: the WebGPU simulator.
//! - `blocking`: `Runner::run_blocking` and `Runner::sample_blocking`,
//!   for native callers that don't use an async runtime.
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod auto;
mod backend;
mod cal;
mod ch;
mod clifford;
mod diagnostic;
mod error;
mod extern_fns;
#[cfg(feature = "gpu")]
mod gpu;
mod host;
mod measure;
#[cfg(feature = "mps")]
mod mps;
mod rng;
mod runner;
mod sim;
mod sim_simd;
mod spec;
mod stabilizer;
mod state;
mod sum;
mod vm;

pub use async_trait::async_trait;
pub use num_complex::{Complex, Complex64};
pub use oqi_compile::classical::{self, Duration, Value, ValueTy};
pub use oqi_compile::{Program, Span, SymbolId};

pub use auto::{AutoSim, SumPolicy};
pub use backend::{GateModifiers, QuantumBackend, u_matrix};
pub use cal::{FrameHandle, OpaqueCalHandler, OpenPulseHandler, PortHandle, WaveformHandle};
pub use error::{Error, ErrorKind, Result};
pub use extern_fns::{ExternProvider, FnRegistry, NoExterns};
#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
pub use gpu::GpuSim;
pub use host::HostValue;
#[cfg(feature = "mps")]
#[cfg_attr(docsrs, doc(cfg(feature = "mps")))]
pub use mps::MpsSim;
pub use runner::{Histogram, Runner, Samples};
pub use sim::StateVectorSim;
pub use sim_simd::{Amplitude, SimdSim};
pub use spec::{BackendKind, BackendSpec, DEFAULT_SEED, ParseNameError, Precision};
pub use vm::{MAX_CALL_DEPTH, RunResult, Vm};
