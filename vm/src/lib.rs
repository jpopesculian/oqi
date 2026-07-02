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
//! - an [`ExternProvider`] — host implementations of `extern` functions.

pub mod auto;
pub mod backend;
pub mod cal;
pub(crate) mod ch;
pub(crate) mod clifford;
pub mod diagnostic;
pub mod error;
pub mod extern_fns;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod sim;
pub mod sim_simd;
pub(crate) mod stabilizer;
pub(crate) mod sum;
pub mod vm;

pub use auto::{AutoSim, SumPolicy};
pub use backend::{GateModifiers, QuantumBackend};
pub use cal::{FrameHandle, OpaqueCalHandler, OpenPulseHandler, PortHandle, WaveformHandle};
pub use error::{Result, VmError, VmErrorKind};
pub use extern_fns::{ExternProvider, FnRegistry, NoExterns};
#[cfg(feature = "gpu")]
pub use gpu::GpuSim;
pub use sim::StateVectorSim;
pub use sim_simd::SimdSim;
pub use vm::{RunResult, Vm};
