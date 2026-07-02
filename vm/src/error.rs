use std::fmt;

use oqi_compile::symbol::SymbolId;
use oqi_lex::Span;

/// The kind of error raised while executing a bytecode module, independent
/// of where in the source it occurred. Wrapped in [`VmError`], which adds the
/// source span, when surfaced by [`Vm::run`](crate::Vm::run).
#[derive(Debug)]
pub enum VmErrorKind {
    /// A register was read before being assigned a value.
    UnsetRegister(u32),
    /// A declared `input` was not given a value before running.
    MissingInput(SymbolId),
    /// A value was supplied for a symbol that isn't a declared `input`.
    UnknownInput(SymbolId),
    /// A `Call` targeted an `extern` the provider doesn't implement.
    UnknownExtern(String),
    /// An `extern` call's host implementation failed: it threw/rejected,
    /// or returned a value that can't be coerced to the declared return
    /// type.
    Extern { name: String, message: String },
    /// A `GateCall` referenced a gate with no executable definition
    /// (not a built-in and no lifted gate body).
    UndefinedGate(String),
    /// A construct the MVP doesn't execute yet.
    Unsupported(String),
    /// A value had the wrong shape/type for the operation.
    Type(String),
    /// A broadcast gate call mixed register operands of differing lengths
    /// (all non-singleton operands must share one length).
    BroadcastMismatch(Vec<usize>),
    /// An error surfaced by the classical value layer.
    Classical(oqi_classical::Error),
    /// A resolved qubit index fell outside the backend's register.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// The program's state vector can't be allocated (e.g. the CPU
    /// state-vector simulator's exponential memory cost outran available
    /// memory or the addressable limit).
    TooManyQubits { requested: u32 },
    /// A calibration/pulse operation failed (raised by the installed
    /// [`OpenPulseHandler`](crate::OpenPulseHandler) /
    /// [`OpaqueCalHandler`](crate::OpaqueCalHandler), or by the VM when
    /// pulse state is inconsistent).
    Pulse(String),
    /// The auto backend's sum-over-Cliffords term budget was exhausted on
    /// a non-Clifford gate and the dense fallback is unavailable.
    RankOverflow {
        rank: usize,
        max_rank: usize,
        qubits: u32,
    },
    /// Execution reached a block marked unreachable.
    Unreachable,
}

impl fmt::Display for VmErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmErrorKind::UnsetRegister(r) => write!(f, "register r{r} read before assignment"),
            VmErrorKind::MissingInput(s) => {
                write!(f, "no value supplied for input symbol {}", s.0)
            }
            VmErrorKind::UnknownInput(s) => {
                write!(f, "value supplied for symbol {} which is not an input", s.0)
            }
            VmErrorKind::UnknownExtern(name) => {
                write!(f, "extern function `{name}` is not provided")
            }
            VmErrorKind::Extern { name, message } => {
                write!(f, "extern function `{name}` failed: {message}")
            }
            VmErrorKind::UndefinedGate(name) => {
                write!(f, "gate `{name}` has no executable definition")
            }
            VmErrorKind::Unsupported(what) => write!(f, "unsupported: {what}"),
            VmErrorKind::Type(msg) => write!(f, "type error: {msg}"),
            VmErrorKind::BroadcastMismatch(lengths) => write!(
                f,
                "gate broadcast over registers of differing lengths: {lengths:?}"
            ),
            VmErrorKind::Classical(e) => write!(f, "classical error: {e:?}"),
            VmErrorKind::QubitOutOfRange { qubit, num_qubits } => write!(
                f,
                "qubit index {qubit} is out of range (the program allocates {num_qubits} qubit(s))"
            ),
            VmErrorKind::TooManyQubits { requested } => write!(
                f,
                "program requires {requested} qubits; its state vector \
                 (2^{requested} complex amplitudes) cannot be allocated"
            ),
            VmErrorKind::Pulse(msg) => write!(f, "pulse error: {msg}"),
            VmErrorKind::RankOverflow {
                rank,
                max_rank,
                qubits,
            } => write!(
                f,
                "circuit exceeds the sum-over-Cliffords budget ({rank} > {max_rank} \
                 stabilizer terms) and its 2^{qubits}-amplitude dense state cannot \
                 be allocated; raise the term budget or reduce non-Clifford gates"
            ),
            VmErrorKind::Unreachable => write!(f, "reached an unreachable block"),
        }
    }
}

/// A runtime error paired with the source span of the instruction that raised
/// it, when known.
#[derive(Debug)]
pub struct VmError {
    pub kind: VmErrorKind,
    pub span: Option<Span>,
}

impl VmError {
    /// A spanless error.
    pub fn new(kind: VmErrorKind) -> Self {
        Self { kind, span: None }
    }

    /// Attach `span` unless one is already set or `span` is the empty
    /// `0..0` sentinel — so the innermost (first attached) span wins.
    pub fn with_span(mut self, span: Span) -> Self {
        if self.span.is_none() && (span.start != 0 || span.end != 0) {
            self.span = Some(span);
        }
        self
    }
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for VmError {}

impl From<VmErrorKind> for VmError {
    fn from(kind: VmErrorKind) -> Self {
        VmError::new(kind)
    }
}

impl From<oqi_classical::Error> for VmErrorKind {
    fn from(e: oqi_classical::Error) -> Self {
        VmErrorKind::Classical(e)
    }
}

/// Internal result type. Errors are spanless ([`VmErrorKind`]) until
/// [`Vm::run`](crate::Vm::run) surfaces them as a spanned [`VmError`].
pub type Result<T> = std::result::Result<T, VmErrorKind>;
