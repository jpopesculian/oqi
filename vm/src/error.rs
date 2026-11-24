use std::fmt;

use oqi_lex::Span;

/// What went wrong while running a program, independent of where in the
/// source it happened.
///
/// The extension points ([`QuantumBackend`](crate::QuantumBackend),
/// [`ExternProvider`](crate::ExternProvider),
/// [`OpenPulseHandler`](crate::OpenPulseHandler),
/// [`OpaqueCalHandler`](crate::OpaqueCalHandler)) report failures as an
/// `ErrorKind`; the VM wraps it in an [`Error`] carrying the source span of
/// the instruction in flight.
#[derive(Debug)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A register was read before being assigned a value.
    UnsetRegister(u32),
    /// A declared `input` (named here) was not given a value before running.
    MissingInput(String),
    /// A value was supplied for a name that isn't a declared `input`.
    UnknownInput(String),
    /// A `Call` targeted an `extern` the provider doesn't implement.
    UnknownExtern(String),
    /// An `extern` call's host implementation failed: it threw/rejected,
    /// or returned a value that can't be coerced to the declared return
    /// type.
    Extern { name: String, message: String },
    /// A `GateCall` referenced a gate with no executable definition
    /// (not a built-in and no lifted gate body).
    UndefinedGate(String),
    /// A construct the VM does not execute.
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
    /// A backend can't hold this many qubits: the state it needs (a state
    /// vector, a tableau, a matrix-product state, a GPU buffer) is larger
    /// than the memory it may use.
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
    /// The requested backend can't be constructed here (no GPU adapter, or
    /// support for it was not compiled into this build).
    BackendUnavailable(String),
    /// The backend can't run this program: it rejected its configuration,
    /// is smaller than the program, or failed while applying an operation
    /// or measuring.
    Backend(String),
    /// Calls nested more than `limit` deep (unbounded recursion).
    CallDepth { limit: u32 },
    /// Execution reached a block marked unreachable.
    Unreachable,
}

impl ErrorKind {
    /// An [`Extern`](ErrorKind::Extern) failure of the extern `name`.
    pub fn extern_failed(name: &str, message: impl Into<String>) -> Self {
        ErrorKind::Extern {
            name: name.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnsetRegister(r) => write!(f, "register r{r} read before assignment"),
            ErrorKind::MissingInput(name) => {
                write!(f, "no value supplied for input `{name}`")
            }
            ErrorKind::UnknownInput(name) => write!(f, "no input named `{name}`"),
            ErrorKind::UnknownExtern(name) => {
                write!(f, "extern function `{name}` is not provided")
            }
            ErrorKind::Extern { name, message } => {
                write!(f, "extern function `{name}` failed: {message}")
            }
            ErrorKind::UndefinedGate(name) => {
                write!(f, "gate `{name}` has no executable definition")
            }
            ErrorKind::Unsupported(what) => write!(f, "unsupported: {what}"),
            ErrorKind::Type(msg) => write!(f, "type error: {msg}"),
            ErrorKind::BroadcastMismatch(lengths) => write!(
                f,
                "gate broadcast over registers of differing lengths: {lengths:?}"
            ),
            ErrorKind::Classical(e) => write!(f, "classical error: {e}"),
            ErrorKind::QubitOutOfRange { qubit, num_qubits } => write!(
                f,
                "qubit index {qubit} is out of range (the program allocates {num_qubits} qubit(s))"
            ),
            ErrorKind::TooManyQubits { requested } => write!(
                f,
                "program requires {requested} qubits; its state vector \
                 (2^{requested} complex amplitudes) cannot be allocated"
            ),
            ErrorKind::Pulse(msg) => write!(f, "pulse error: {msg}"),
            ErrorKind::RankOverflow {
                rank,
                max_rank,
                qubits,
            } => write!(
                f,
                "circuit exceeds the sum-over-Cliffords budget ({rank} > {max_rank} \
                 stabilizer terms) and its 2^{qubits}-amplitude dense state cannot \
                 be allocated; raise the term budget or reduce non-Clifford gates"
            ),
            ErrorKind::BackendUnavailable(why) => write!(f, "backend unavailable: {why}"),
            ErrorKind::Backend(why) => write!(f, "backend error: {why}"),
            ErrorKind::CallDepth { limit } => {
                write!(f, "calls nest more than {limit} deep")
            }
            ErrorKind::Unreachable => write!(f, "reached an unreachable block"),
        }
    }
}

impl std::error::Error for ErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ErrorKind::Classical(e) => Some(e),
            _ => None,
        }
    }
}

/// A runtime error: what went wrong, and the source span of the instruction
/// that raised it, when known.
#[derive(Debug)]
pub struct Error {
    /// What went wrong.
    pub kind: ErrorKind,
    /// Where in the program's source, when known. Programs loaded from
    /// bytecode keep their source spans but not the text.
    pub span: Option<Span>,
}

impl Error {
    /// An error with no location.
    pub fn new(kind: ErrorKind) -> Self {
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Error::new(kind)
    }
}

impl From<oqi_classical::Error> for ErrorKind {
    fn from(e: oqi_classical::Error) -> Self {
        ErrorKind::Classical(e)
    }
}

impl From<oqi_classical::Error> for Error {
    fn from(e: oqi_classical::Error) -> Self {
        Error::new(ErrorKind::Classical(e))
    }
}

/// The result of running a program: an [`Error`] carries the source span of
/// the instruction that raised it.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The result of an interpreter step or an extension-point call: a spanless
/// [`ErrorKind`], to which the VM attaches the span of the instruction in
/// flight.
pub(crate) type KindResult<T> = std::result::Result<T, ErrorKind>;
