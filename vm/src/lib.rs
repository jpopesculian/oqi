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

pub mod backend;
pub mod error;
pub mod extern_fns;
pub mod sim;
pub mod vm;

pub use backend::{GateModifiers, QuantumBackend};
pub use error::{Result, VmError};
pub use extern_fns::{ExternProvider, FnRegistry, NoExterns};
pub use sim::StateVectorSim;
pub use vm::{RunResult, Vm};
