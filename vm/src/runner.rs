//! Running a program end to end: pick a backend, supply inputs and externs,
//! run once or sample many shots.

use std::collections::HashMap;
use std::fmt;

use oqi_classical::{Primitive, Value};
use oqi_compile::Program;

use crate::error::Result;
use crate::extern_fns::{ExternProvider, NoExterns};
use crate::host::{HostValue, coerce_inputs};
use crate::spec::{BackendSpec, DEFAULT_SEED};
use crate::vm::{RunResult, Vm};

/// Runs a compiled [`Program`] on a backend chosen by [`BackendSpec`]: the
/// usual way to execute one.
///
/// One [`seed`](Self::seed) makes a run, or a whole set of samples,
/// reproducible. Inputs are given by name as host values and coerced to
/// their declared types.
///
/// ```
/// use oqi_vm::Runner;
///
/// let program = oqi_compile::compile(
///     r#"include "stdgates.inc";
///        input int n;
///        qubit q;
///        if (n > 2) { x q; }
///        bit c = measure q;"#,
/// )?;
/// let result = pollster::block_on(Runner::new(&program).input("n", 3).run())?;
/// assert_eq!(result.output("c").unwrap().to_string(), "1");
///
/// let samples = pollster::block_on(
///     Runner::new(&program).backend("auto".parse()?).input("n", 0).sample(100),
/// )?;
/// assert_eq!(samples.histogram("c").unwrap().counts(), [("0".to_string(), 100)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Runner<'p, E = NoExterns> {
    program: &'p Program,
    backend: BackendSpec,
    seed: u64,
    inputs: Vec<(String, HostValue)>,
    externs: E,
    statevector: bool,
}

impl<'p> Runner<'p> {
    /// A runner for `program` on the default backend (the scalar `f64`
    /// state-vector simulator), seeded with [`DEFAULT_SEED`], with no inputs
    /// and no externs.
    pub fn new(program: &'p Program) -> Self {
        Runner {
            program,
            backend: BackendSpec::default(),
            seed: DEFAULT_SEED,
            inputs: Vec::new(),
            externs: NoExterns,
            statevector: false,
        }
    }
}

impl<'p, E: ExternProvider> Runner<'p, E> {
    /// Run on the backend `spec` describes.
    pub fn backend(mut self, spec: BackendSpec) -> Self {
        self.backend = spec;
        self
    }

    /// Seed the backend's measurements and the draws of
    /// [`sample`](Self::sample): a given seed reproduces the result.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Give the `input` called `name` the value `value`, coerced to its
    /// declared type (see [`HostValue::coerce_to`]).
    pub fn input(mut self, name: impl Into<String>, value: impl Into<HostValue>) -> Self {
        self.inputs.push((name.into(), value.into()));
        self
    }

    /// [`input`](Self::input) for each `(name, value)`.
    pub fn inputs<N: Into<String>, V: Into<HostValue>>(
        mut self,
        inputs: impl IntoIterator<Item = (N, V)>,
    ) -> Self {
        let inputs = inputs
            .into_iter()
            .map(|(name, value)| (name.into(), value.into()));
        self.inputs.extend(inputs);
        self
    }

    /// Implement the program's `extern` functions with `externs`.
    pub fn externs<X: ExternProvider>(self, externs: X) -> Runner<'p, X> {
        Runner {
            program: self.program,
            backend: self.backend,
            seed: self.seed,
            inputs: self.inputs,
            externs,
            statevector: self.statevector,
        }
    }

    /// Keep the final state vector in [`run`](Self::run)'s result (see
    /// [`RunResult::statevector`]), when the backend has one.
    pub fn statevector(mut self, statevector: bool) -> Self {
        self.statevector = statevector;
        self
    }

    /// Build the backend and a VM over it, and coerce the inputs.
    async fn start(
        self,
    ) -> Result<(
        Vm<'p, Box<dyn crate::QuantumBackend>, E>,
        BackendSpec,
        HashMap<oqi_compile::SymbolId, Value>,
    )> {
        let (backend, built) = self.backend.build_for(self.program, self.seed).await?;
        let inputs = coerce_inputs(self.program, &self.inputs)?;
        let vm = Vm::new(self.program, backend, self.externs).with_seed(self.seed);
        Ok((vm, built, inputs))
    }

    /// Run the program once.
    ///
    /// # Errors
    ///
    /// The backend can't be built (see [`BackendSpec::build`]), an input is
    /// unknown, missing or doesn't fit its type, or the run fails.
    pub async fn run(self) -> Result<RunResult> {
        let statevector = self.statevector;
        let (mut vm, built, inputs) = self.start().await?;
        let result = vm.run_with_inputs(inputs).await?;
        let amplitudes = match statevector {
            true => vm.backend().amplitudes().await,
            false => None,
        };
        Ok(result.with_statevector(amplitudes).with_backend(built))
    }

    /// Run the program `shots` times and tally each output's values (see
    /// [`Vm::sample`] for when the circuit runs only once).
    ///
    /// # Errors
    ///
    /// As for [`run`](Self::run).
    pub async fn sample(self, shots: u32) -> Result<Samples> {
        let program = self.program;
        let (mut vm, built, inputs) = self.start().await?;
        let mut tally = Tally::new(program);
        vm.sample(inputs, shots, |result| tally.add(&result))
            .await?;
        Ok(Samples {
            shots,
            histograms: tally.finish(),
            backend: built,
        })
    }

    /// [`run`](Self::run), blocking the calling thread until it finishes.
    ///
    /// For native callers. In the browser the GPU backend has to await its
    /// device, which a blocked thread can't do: use [`run`](Self::run)
    /// there.
    ///
    /// # Errors
    ///
    /// As for [`run`](Self::run).
    #[cfg(feature = "blocking")]
    #[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
    pub fn run_blocking(self) -> Result<RunResult> {
        pollster::block_on(self.run())
    }

    /// [`sample`](Self::sample), blocking the calling thread until it
    /// finishes. As for [`run_blocking`](Self::run_blocking), for native
    /// callers.
    ///
    /// # Errors
    ///
    /// As for [`run`](Self::run).
    #[cfg(feature = "blocking")]
    #[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
    pub fn sample_blocking(self, shots: u32) -> Result<Samples> {
        pollster::block_on(self.sample(shots))
    }
}

impl<E> fmt::Debug for Runner<'_, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runner")
            .field("backend", &self.backend)
            .field("seed", &self.seed)
            .field("inputs", &self.inputs)
            .field("statevector", &self.statevector)
            .finish_non_exhaustive()
    }
}

/// The outcome of [`Runner::sample`]: how often each output took each value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Samples {
    shots: u32,
    histograms: Vec<Histogram>,
    backend: BackendSpec,
}

impl Samples {
    /// How many shots ran.
    pub fn shots(&self) -> u32 {
        self.shots
    }

    /// One histogram per output, in program order.
    pub fn histograms(&self) -> &[Histogram] {
        &self.histograms
    }

    /// The histogram of the output `name`.
    pub fn histogram(&self, name: &str) -> Option<&Histogram> {
        self.histograms.iter().find(|h| h.name == name)
    }

    /// The backend configuration that ran (see [`RunResult::backend`]).
    pub fn backend(&self) -> &BackendSpec {
        &self.backend
    }
}

/// How often each distinct value of one output was observed across a set of
/// shots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    name: String,
    counts: Vec<(String, u32)>,
}

impl Histogram {
    /// The output's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `(label, count)` per distinct value. A label is the value's canonical
    /// text: `bit`/`bool` as `0`/`1`, `bit[n]` as an unquoted MSB-first bit
    /// string, everything else its OpenQASM text form. Sorted by value:
    /// numerically when every label is a plain decimal integer, else
    /// lexicographically — so fixed-width bit strings keep their binary
    /// order.
    pub fn counts(&self) -> &[(String, u32)] {
        &self.counts
    }

    /// How many shots produced this output.
    pub fn total(&self) -> u32 {
        self.counts.iter().map(|(_, count)| count).sum()
    }
}

/// Histograms in the making, one per output of a program.
struct Tally {
    /// Output names in program order, with their counts by label.
    outputs: Vec<(String, HashMap<String, u32>)>,
}

impl Tally {
    fn new(program: &Program) -> Self {
        let outputs = program
            .outputs
            .iter()
            .map(|(sym, _)| (program.symbol(*sym).name.clone(), HashMap::new()))
            .collect();
        Tally { outputs }
    }

    fn add(&mut self, result: &RunResult) {
        for (name, value) in result.outputs() {
            if let Some((_, counts)) = self.outputs.iter_mut().find(|(n, _)| n == name) {
                *counts.entry(label(value)).or_insert(0) += 1;
            }
        }
    }

    /// The histograms of the outputs some shot produced.
    fn finish(self) -> Vec<Histogram> {
        self.outputs
            .into_iter()
            .filter(|(_, counts)| !counts.is_empty())
            .map(|(name, counts)| {
                let mut counts: Vec<(String, u32)> = counts.into_iter().collect();
                let numeric = counts.iter().all(|(label, _)| {
                    let digits = label.strip_prefix('-').unwrap_or(label);
                    digits.parse::<i128>().is_ok()
                        && (digits.len() == 1 || !digits.starts_with('0'))
                });
                if numeric {
                    counts.sort_by_key(|(label, _)| label.parse::<i128>().unwrap());
                } else {
                    counts.sort();
                }
                Histogram { name, counts }
            })
            .collect()
    }
}

/// A value's histogram label (see [`Histogram::counts`]).
fn label(value: &Value) -> String {
    if let Value::Scalar(s) = value
        && let Primitive::Bit(b) = s.value()
    {
        return (if *b { "1" } else { "0" }).to_string();
    }
    value.to_string().trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_bits_and_unquoted_bit_strings() {
        let reg = HostValue::from("0110").coerce_to(oqi_classical::ValueTy::bitreg(4));
        assert_eq!(label(&reg.unwrap()), "0110");
        assert_eq!(label(&Value::bit(true)), "1");
    }
}
