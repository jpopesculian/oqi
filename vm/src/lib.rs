//! A virtual machine for OQI bytecode ([`oqi_compile::bytecode`]).
//!
//! [`Vm`] interprets a [`BcModule`](oqi_compile::bytecode::BcModule)
//! against two pluggable extension points:
//!
//! - a [`QuantumBackend`] — the quantum execution layer (a CPU
//!   [`StateVectorSim`], or a future GPU simulator / hardware provider).
//!   It only implements the built-in primitives `U`/`gphase` plus
//!   measure/reset; backends ship their own `stdgates.inc` that
//!   decomposes the standard library to these (or to `extern` calls).
//!   [`BackendSpec`] selects and builds one of the bundled simulators by
//!   name.
//! - an [`ExternProvider`] — host implementations of `extern` functions.
//!
//! Front doors (CLI, JS, Python) exchange values with the VM as
//! [`HostValue`]s: see [`coerce_inputs`], [`coerce`] and [`histograms`].

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
mod sim;
mod sim_simd;
mod spec;
mod stabilizer;
mod state;
mod sum;
mod vm;

pub use auto::{AutoSim, SumPolicy};
pub use backend::{GateModifiers, QuantumBackend};
pub use cal::{FrameHandle, OpaqueCalHandler, OpenPulseHandler, PortHandle, WaveformHandle};
pub use error::{KindResult, VmError, VmErrorKind, VmResult};
pub use extern_fns::{ExternProvider, FnRegistry, NoExterns};
#[cfg(feature = "gpu")]
pub use gpu::GpuSim;
pub use host::{Histogram, HostValue, coerce, coerce_inputs, histograms};
#[cfg(feature = "mps")]
pub use mps::MpsSim;
pub use sim::StateVectorSim;
pub use sim_simd::{SimdLane, SimdSim};
pub use spec::{BackendKind, BackendSpec, DEFAULT_SEED, Precision};
pub use vm::{RunResult, Vm};
