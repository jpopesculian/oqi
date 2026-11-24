//! Runtime backend selection: a [`BackendSpec`] names a simulator
//! configuration and builds it as a boxed [`QuantumBackend`], so every
//! front door (CLI, JS, Python) offers the same backends under the same
//! names.

use std::fmt;
use std::num::NonZeroUsize;
use std::str::FromStr;

use oqi_compile::Program;

use crate::auto::{AutoSim, SumPolicy};
use crate::backend::QuantumBackend;
use crate::error::{Error, ErrorKind, Result};
use crate::sim::StateVectorSim;
use crate::sim_simd::SimdSim;

/// RNG seed used when the host doesn't supply one, so runs of the same
/// program agree across front doors.
pub const DEFAULT_SEED: u64 = 0x2545F4914F6CDD1D;

/// Default bond-dimension cap of the MPS simulator (`MpsSim::DEFAULT_MAX_BOND`).
pub(crate) const DEFAULT_MAX_BOND: usize = 256;

/// Which simulator to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendKind {
    /// Scalar CPU state-vector simulator ([`StateVectorSim`]).
    Scalar,
    /// SIMD-vectorized CPU state-vector simulator ([`SimdSim`]).
    Simd,
    /// WebGPU (`wgpu`) state-vector simulator; single precision only. Needs
    /// the `gpu` build feature and a working adapter.
    Gpu,
    /// [`Gpu`](BackendKind::Gpu) when it's available, else
    /// [`Scalar`](BackendKind::Scalar).
    PreferGpu,
    /// Auto-routing ([`AutoSim`]): stabilizer tableau, then an exact sum
    /// over Cliffords, then (if [`SumPolicy::dense_escape`]) a dense state
    /// vector.
    Auto,
    /// Matrix-product-state simulator, truncating each bond to at most
    /// [`BackendSpec::max_bond`] singular values. Needs the `mps` build
    /// feature (on by default).
    Mps,
}

impl BackendKind {
    /// Every kind, in the order of [`BackendKind::name`]'s names.
    const ALL: [BackendKind; 6] = [
        BackendKind::Scalar,
        BackendKind::Simd,
        BackendKind::Gpu,
        BackendKind::PreferGpu,
        BackendKind::Auto,
        BackendKind::Mps,
    ];

    /// The kind's name: `scalar`, `simd`, `gpu`, `prefer-gpu`, `auto` or
    /// `mps`.
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Scalar => "scalar",
            BackendKind::Simd => "simd",
            BackendKind::Gpu => "gpu",
            BackendKind::PreferGpu => "prefer-gpu",
            BackendKind::Auto => "auto",
            BackendKind::Mps => "mps",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for BackendKind {
    type Err = ParseNameError;

    fn from_str(s: &str) -> std::result::Result<Self, ParseNameError> {
        BackendKind::ALL
            .into_iter()
            .find(|kind| kind.name() == s)
            .ok_or_else(|| {
                ParseNameError::new("backend kind", s, &BackendKind::ALL.map(BackendKind::name))
            })
    }
}

/// Amplitude precision of the dense simulators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// 64-bit floating-point amplitudes.
    #[default]
    F64,
    /// 32-bit floating-point amplitudes.
    F32,
}

impl Precision {
    /// `"f64"` or `"f32"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
        }
    }
}

impl FromStr for Precision {
    type Err = ParseNameError;

    fn from_str(s: &str) -> std::result::Result<Self, ParseNameError> {
        match s {
            "f64" => Ok(Precision::F64),
            "f32" => Ok(Precision::F32),
            _ => Err(ParseNameError::new("precision", s, &["f64", "f32"])),
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A name that isn't one of the accepted backend, backend-kind or precision
/// names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseNameError {
    what: &'static str,
    found: String,
    expected: Vec<&'static str>,
}

impl ParseNameError {
    fn new(what: &'static str, found: &str, expected: &[&'static str]) -> Self {
        ParseNameError {
            what,
            found: found.to_string(),
            expected: expected.to_vec(),
        }
    }
}

impl fmt::Display for ParseNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown {} `{}` (expected one of: {})",
            self.what,
            self.found,
            self.expected.join(", ")
        )
    }
}

impl std::error::Error for ParseNameError {}

/// A simulator configuration. Parse one from a backend name (see
/// [`BackendSpec::NAMES`]), adjust the remaining fields, then
/// [`build`](BackendSpec::build) it — or hand it to a
/// [`Runner`](crate::Runner), which does.
///
/// ```
/// use oqi_vm::{BackendKind, BackendSpec};
///
/// let mut spec: BackendSpec = "mps".parse()?;
/// spec.max_bond = 64;
/// assert_eq!(spec.kind, BackendKind::Mps);
/// # Ok::<(), oqi_vm::ParseNameError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BackendSpec {
    /// Which simulator.
    pub kind: BackendKind,
    /// Amplitude precision of the scalar/SIMD simulators. The GPU simulator
    /// is always `f32` and the auto-router and MPS simulator always `f64`.
    pub precision: Precision,
    /// Use the multi-threaded (rayon) gate kernel of the scalar/SIMD
    /// simulators. Needs the `parallel` build feature.
    pub parallel: bool,
    /// Budget of the auto-router's sum-over-Cliffords tier.
    pub sum_policy: SumPolicy,
    /// Bond-dimension cap of the MPS simulator; at least 1.
    pub max_bond: usize,
}

impl Default for BackendSpec {
    /// The scalar `f64` state-vector simulator.
    fn default() -> Self {
        BackendSpec {
            kind: BackendKind::Scalar,
            precision: Precision::default(),
            parallel: false,
            sum_policy: SumPolicy::default(),
            max_bond: DEFAULT_MAX_BOND,
        }
    }
}

impl BackendSpec {
    /// The backend names every front door accepts.
    pub const NAMES: [&'static str; 9] = [
        "scalar",
        "rayon",
        "simd",
        "rayon-simd",
        "gpu",
        "prefer-gpu",
        "auto",
        "sum",
        "mps",
    ];

    /// The name this spec parses from.
    pub fn name(&self) -> &'static str {
        match (self.kind, self.parallel) {
            (BackendKind::Scalar, true) => "rayon",
            (BackendKind::Simd, true) => "rayon-simd",
            (BackendKind::Auto, _) if !self.sum_policy.dense_escape => "sum",
            (kind, _) => kind.name(),
        }
    }

    /// Construct the backend for a `num_qubits`-qubit program, measuring with
    /// an RNG seeded by `seed`. Also returns the spec that was actually
    /// built: `kind` is resolved for [`BackendKind::PreferGpu`] and
    /// `precision` is the real amplitude type.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::BackendUnavailable`] if this build or machine can't run
    /// the backend, [`ErrorKind::TooManyQubits`] if it can't hold
    /// `num_qubits` qubits, and [`ErrorKind::Backend`] for an invalid
    /// configuration (a `max_bond` of 0).
    pub async fn build(
        mut self,
        num_qubits: u32,
        seed: u64,
    ) -> Result<(Box<dyn QuantumBackend>, BackendSpec)> {
        if self.kind == BackendKind::PreferGpu {
            self.kind = match gpu(num_qubits, seed).await {
                Ok(backend) => {
                    self.kind = BackendKind::Gpu;
                    self.precision = Precision::F32;
                    return Ok((backend, self));
                }
                Err(_) => BackendKind::Scalar,
            };
        }
        let backend: Box<dyn QuantumBackend> = match (self.kind, self.precision) {
            (BackendKind::Scalar, Precision::F64) => Box::new(
                parallel(StateVectorSim::<f64>::new(num_qubits)?, self.parallel)?.with_seed(seed),
            ),
            (BackendKind::Scalar, Precision::F32) => Box::new(
                parallel(StateVectorSim::<f32>::new(num_qubits)?, self.parallel)?.with_seed(seed),
            ),
            (BackendKind::Simd, Precision::F64) => {
                Box::new(parallel(SimdSim::<f64>::new(num_qubits)?, self.parallel)?.with_seed(seed))
            }
            (BackendKind::Simd, Precision::F32) => {
                Box::new(parallel(SimdSim::<f32>::new(num_qubits)?, self.parallel)?.with_seed(seed))
            }
            (BackendKind::Gpu, _) => {
                self.precision = Precision::F32;
                gpu(num_qubits, seed).await?
            }
            (BackendKind::Auto, _) => {
                // Exact until it escalates to dense, which is `f64`.
                self.precision = Precision::F64;
                Box::new(
                    AutoSim::new(num_qubits)?
                        .with_seed(seed)
                        .with_policy(self.sum_policy),
                )
            }
            (BackendKind::Mps, _) => {
                self.precision = Precision::F64;
                let max_bond = NonZeroUsize::new(self.max_bond).ok_or_else(|| {
                    Error::new(ErrorKind::Backend(
                        "the MPS bond cap must be at least 1".into(),
                    ))
                })?;
                mps(num_qubits, seed, max_bond)?
            }
            (BackendKind::PreferGpu, _) => unreachable!("resolved above"),
        };
        Ok((backend, self))
    }

    /// [`build`](Self::build) the backend for `program`'s qubits.
    ///
    /// # Errors
    ///
    /// As for [`build`](Self::build).
    pub async fn build_for(
        self,
        program: &Program,
        seed: u64,
    ) -> Result<(Box<dyn QuantumBackend>, BackendSpec)> {
        self.build(program.num_qubits(), seed).await
    }
}

impl FromStr for BackendSpec {
    type Err = ParseNameError;

    /// Parse a backend name (one of [`BackendSpec::NAMES`]) into a spec with
    /// default precision, term budget and bond cap.
    fn from_str(s: &str) -> std::result::Result<Self, ParseNameError> {
        let (kind, parallel, dense_escape) = match s {
            "rayon" => (BackendKind::Scalar, true, true),
            "rayon-simd" => (BackendKind::Simd, true, true),
            "sum" => (BackendKind::Auto, false, false),
            _ => match s.parse::<BackendKind>() {
                Ok(kind) => (kind, false, true),
                Err(_) => return Err(ParseNameError::new("backend", s, &Self::NAMES)),
            },
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

impl fmt::Display for BackendSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(not(all(feature = "parallel", feature = "mps", feature = "gpu")))]
fn unavailable(why: &str) -> Error {
    Error::new(ErrorKind::BackendUnavailable(why.to_string()))
}

/// `sim` with its multi-threaded kernel if `parallel`, which needs the
/// `parallel` build feature.
#[cfg(feature = "parallel")]
fn parallel<S: Parallel>(sim: S, parallel: bool) -> Result<S> {
    Ok(sim.parallel(parallel))
}

#[cfg(not(feature = "parallel"))]
fn parallel<S>(sim: S, parallel: bool) -> Result<S> {
    if parallel {
        return Err(unavailable(
            "this build has no multi-threading support (the `parallel` feature is off)",
        ));
    }
    Ok(sim)
}

/// The simulators with a multi-threaded kernel.
#[cfg(feature = "parallel")]
trait Parallel {
    fn parallel(self, parallel: bool) -> Self;
}

#[cfg(feature = "parallel")]
impl<F: crate::Amplitude> Parallel for StateVectorSim<F> {
    fn parallel(self, parallel: bool) -> Self {
        self.with_parallel(parallel)
    }
}

#[cfg(feature = "parallel")]
impl<F: crate::Amplitude> Parallel for SimdSim<F> {
    fn parallel(self, parallel: bool) -> Self {
        self.with_parallel(parallel)
    }
}

#[cfg(feature = "mps")]
fn mps(num_qubits: u32, seed: u64, max_bond: NonZeroUsize) -> Result<Box<dyn QuantumBackend>> {
    Ok(Box::new(
        crate::mps::MpsSim::new(num_qubits)?
            .with_seed(seed)
            .with_max_bond(max_bond),
    ))
}

#[cfg(not(feature = "mps"))]
fn mps(_num_qubits: u32, _seed: u64, _max_bond: NonZeroUsize) -> Result<Box<dyn QuantumBackend>> {
    Err(unavailable(
        "this build has no MPS support (the `mps` feature is off)",
    ))
}

#[cfg(feature = "gpu")]
async fn gpu(num_qubits: u32, seed: u64) -> Result<Box<dyn QuantumBackend>> {
    Ok(Box::new(
        crate::gpu::GpuSim::new(num_qubits).await?.with_seed(seed),
    ))
}

#[cfg(not(feature = "gpu"))]
async fn gpu(_num_qubits: u32, _seed: u64) -> Result<Box<dyn QuantumBackend>> {
    Err(unavailable(
        "this build has no GPU support (the `gpu` feature is off)",
    ))
}

/// Tests that create a GPU device take turns doing so: llvmpipe, Mesa's
/// software Vulkan (the adapter on a machine without a GPU), crashes when two
/// threads create devices at once, and the test harness runs tests on several.
#[cfg(test)]
pub(crate) fn gpu_turn() -> std::sync::MutexGuard<'static, ()> {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TURN.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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
            let turn = gpu_turn();
            let (_, built) = pollster::block_on(spec.build(1, DEFAULT_SEED)).unwrap();
            drop(turn);
            // Whichever was available, the request is resolved.
            assert_ne!(built.kind, BackendKind::PreferGpu);
            if built.kind == BackendKind::Gpu {
                assert_eq!(built.precision, Precision::F32);
            }

            let mut spec: BackendSpec = "sum".parse().unwrap();
            spec.precision = Precision::F32;
            let (_, built) = spec.build(1, DEFAULT_SEED).await.unwrap();
            assert_eq!((built.name(), built.precision), ("sum", Precision::F64));
        })
    }

    #[cfg(feature = "mps")]
    #[test]
    fn mps_is_f64_and_samples_shots() {
        pollster::block_on(async {
            let mut spec: BackendSpec = "mps".parse().unwrap();
            spec.precision = Precision::F32;
            // Far past any state vector, but a product state for an MPS.
            let (backend, built) = spec.build(300, DEFAULT_SEED).await.unwrap();
            assert_eq!((built.name(), built.precision), ("mps", Precision::F64));
            assert!(backend.can_sample_shots());
        })
    }

    #[test]
    fn oversized_state_vector_is_an_error() {
        pollster::block_on(async {
            let spec: BackendSpec = "simd".parse().unwrap();
            let err = spec
                .build(200, DEFAULT_SEED)
                .await
                .err()
                .expect("2^200 amplitudes");
            assert!(matches!(
                err.kind,
                ErrorKind::TooManyQubits { requested: 200 }
            ));
        })
    }
}
