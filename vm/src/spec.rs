//! Runtime backend selection: a [`BackendSpec`] names a simulator
//! configuration and builds it as a boxed [`QuantumBackend`], so every
//! front door (CLI, JS, Python) offers the same backends under the same
//! names.

use std::fmt;
use std::str::FromStr;

use crate::auto::{AutoSim, SumPolicy};
use crate::backend::QuantumBackend;
use crate::error::{VmError, VmErrorKind, VmResult};
use crate::sim::StateVectorSim;
use crate::sim_simd::SimdSim;

/// RNG seed used when the host doesn't supply one, so runs of the same
/// program agree across front doors.
pub const DEFAULT_SEED: u64 = 0x2545F4914F6CDD1D;

/// Which simulator to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Scalar CPU state-vector simulator.
    Scalar,
    /// SIMD-vectorized CPU state-vector simulator.
    Simd,
    /// WebGPU (`wgpu`) state-vector simulator; single precision only. Needs
    /// the `gpu` build feature and a working adapter.
    Gpu,
    /// [`Gpu`](BackendKind::Gpu) when it's available, else
    /// [`Scalar`](BackendKind::Scalar).
    PreferGpu,
    /// Auto-routing: stabilizer tableau, then an exact sum over Cliffords,
    /// then (if [`SumPolicy::dense_escape`]) a dense state vector.
    Auto,
}

/// Amplitude precision of the dense simulators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    F64,
    F32,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
        }
    }
}

impl FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f64" => Ok(Precision::F64),
            "f32" => Ok(Precision::F32),
            _ => Err(format!("unknown precision `{s}` (expected f64 or f32)")),
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A simulator configuration. Parse one from a backend name (see
/// [`BackendSpec::NAMES`]), adjust the remaining fields, then
/// [`build`](BackendSpec::build) it.
#[derive(Clone, Copy, Debug)]
pub struct BackendSpec {
    pub kind: BackendKind,
    /// Amplitude precision of the scalar/SIMD simulators. The GPU simulator
    /// is always `f32` and the auto-router always `f64`.
    pub precision: Precision,
    /// Measurement RNG seed.
    pub seed: u64,
    /// Use the multi-threaded (rayon) gate kernel of the scalar/SIMD
    /// simulators. Needs the `parallel` build feature.
    pub parallel: bool,
    /// Budget of the auto-router's sum-over-Cliffords tier.
    pub sum_policy: SumPolicy,
}

impl Default for BackendSpec {
    fn default() -> Self {
        BackendSpec {
            kind: BackendKind::Scalar,
            precision: Precision::default(),
            seed: DEFAULT_SEED,
            parallel: false,
            sum_policy: SumPolicy::default(),
        }
    }
}

impl BackendSpec {
    /// The backend names every front door accepts.
    pub const NAMES: [&'static str; 8] = [
        "scalar",
        "rayon",
        "simd",
        "rayon-simd",
        "gpu",
        "prefer-gpu",
        "auto",
        "sum",
    ];

    /// The name this spec parses from.
    pub fn name(&self) -> &'static str {
        match (self.kind, self.parallel) {
            (BackendKind::Scalar, false) => "scalar",
            (BackendKind::Scalar, true) => "rayon",
            (BackendKind::Simd, false) => "simd",
            (BackendKind::Simd, true) => "rayon-simd",
            (BackendKind::Gpu, _) => "gpu",
            (BackendKind::PreferGpu, _) => "prefer-gpu",
            (BackendKind::Auto, _) if self.sum_policy.dense_escape => "auto",
            (BackendKind::Auto, _) => "sum",
        }
    }

    /// Construct the backend for a `num_qubits`-qubit program. Also returns
    /// the spec that was actually built: `kind` is resolved for
    /// [`BackendKind::PreferGpu`] and `precision` is the real amplitude
    /// type.
    pub async fn build(
        mut self,
        num_qubits: u32,
    ) -> VmResult<(Box<dyn QuantumBackend>, BackendSpec)> {
        if self.kind == BackendKind::PreferGpu {
            self.kind = match gpu(num_qubits, self.seed).await {
                Ok(backend) => {
                    self.kind = BackendKind::Gpu;
                    self.precision = Precision::F32;
                    return Ok((backend, self));
                }
                Err(_) => BackendKind::Scalar,
            };
        }
        if self.parallel && !cfg!(feature = "parallel") {
            return Err(unavailable(
                "this build has no multi-threading support (the `parallel` feature is off)",
            ));
        }
        let backend: Box<dyn QuantumBackend> = match (self.kind, self.precision) {
            (BackendKind::Scalar, Precision::F64) => Box::new(
                StateVectorSim::<f64>::try_zeroed(num_qubits, self.seed)?
                    .with_parallel(self.parallel),
            ),
            (BackendKind::Scalar, Precision::F32) => Box::new(
                StateVectorSim::<f32>::try_zeroed(num_qubits, self.seed)?
                    .with_parallel(self.parallel),
            ),
            (BackendKind::Simd, Precision::F64) => Box::new(
                SimdSim::<f64>::try_zeroed(num_qubits, self.seed)?.with_parallel(self.parallel),
            ),
            (BackendKind::Simd, Precision::F32) => Box::new(
                SimdSim::<f32>::try_zeroed(num_qubits, self.seed)?.with_parallel(self.parallel),
            ),
            (BackendKind::Gpu, _) => {
                self.precision = Precision::F32;
                gpu(num_qubits, self.seed).await?
            }
            (BackendKind::Auto, _) => {
                // Exact until it escalates to dense, which is `f64`.
                self.precision = Precision::F64;
                Box::new(AutoSim::with_policy(num_qubits, self.seed, self.sum_policy))
            }
            (BackendKind::PreferGpu, _) => unreachable!("resolved above"),
        };
        Ok((backend, self))
    }
}

impl FromStr for BackendSpec {
    type Err = String;

    /// Parse a backend name into a spec with default precision, seed and
    /// term budget.
    fn from_str(s: &str) -> Result<Self, String> {
        let (kind, parallel, dense_escape) = match s {
            "scalar" => (BackendKind::Scalar, false, true),
            "rayon" => (BackendKind::Scalar, true, true),
            "simd" => (BackendKind::Simd, false, true),
            "rayon-simd" => (BackendKind::Simd, true, true),
            "gpu" => (BackendKind::Gpu, false, true),
            "prefer-gpu" => (BackendKind::PreferGpu, false, true),
            "auto" => (BackendKind::Auto, false, true),
            "sum" => (BackendKind::Auto, false, false),
            _ => {
                return Err(format!(
                    "unknown backend `{s}` (expected one of: {})",
                    Self::NAMES.join(", ")
                ));
            }
        };
        Ok(BackendSpec {
            kind,
            parallel,
            sum_policy: SumPolicy {
                dense_escape,
                ..SumPolicy::default()
            },
            ..BackendSpec::default()
        })
    }
}

fn unavailable(why: &str) -> VmError {
    VmError::new(VmErrorKind::BackendUnavailable(why.to_string()))
}

#[cfg(feature = "gpu")]
async fn gpu(num_qubits: u32, seed: u64) -> VmResult<Box<dyn QuantumBackend>> {
    Ok(Box::new(
        crate::gpu::GpuSim::try_zeroed(num_qubits, seed).await?,
    ))
}

#[cfg(not(feature = "gpu"))]
async fn gpu(_num_qubits: u32, _seed: u64) -> VmResult<Box<dyn QuantumBackend>> {
    Err(unavailable(
        "this build has no GPU support (the `gpu` feature is off)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_round_trips() {
        for name in BackendSpec::NAMES {
            let spec: BackendSpec = name.parse().expect(name);
            assert_eq!(spec.name(), name);
        }
        assert!("cpu".parse::<BackendSpec>().is_err());
    }

    #[test]
    fn build_reports_what_ran() {
        pollster::block_on(async {
            let spec: BackendSpec = "prefer-gpu".parse().unwrap();
            let (_, built) = spec.build(1).await.unwrap();
            // Whichever was available, the request is resolved.
            assert_ne!(built.kind, BackendKind::PreferGpu);
            if built.kind == BackendKind::Gpu {
                assert_eq!(built.precision, Precision::F32);
            }

            let mut spec: BackendSpec = "sum".parse().unwrap();
            spec.precision = Precision::F32;
            let (_, built) = spec.build(1).await.unwrap();
            assert_eq!((built.name(), built.precision), ("sum", Precision::F64));
        })
    }

    #[test]
    fn oversized_state_vector_is_an_error() {
        pollster::block_on(async {
            let spec: BackendSpec = "simd".parse().unwrap();
            let err = spec.build(200).await.err().expect("2^200 amplitudes");
            assert!(matches!(
                err.kind,
                VmErrorKind::TooManyQubits { requested: 200 }
            ));
        })
    }
}
