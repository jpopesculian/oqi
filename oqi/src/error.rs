//! One error type for everything the facade does.

use std::fmt;
use std::path::Path;

use crate::compiler::CompileError;
use crate::compiler::bytecode::{DecodeError, EncodeError};
use crate::diagnostics::{Code, DiagLabel, Diagnostic};
use crate::{syntax, vm};

/// Any error from parsing, formatting, compiling, loading or running a
/// program.
///
/// Each step's own error converts into it, so `?` works across them. It
/// [displays](fmt::Display) as the underlying error does and renders as the
/// same [`Diagnostic`].
///
/// ```
/// fn bell_counts() -> oqi::Result<usize> {
///     let program = oqi::compile(
///         r#"include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0], q[1]; bit[2] c = measure q;"#,
///     )?; // a compile error
///     let samples = oqi::Runner::new(&program)
///         .backend("auto".parse()?) // an unknown backend name
///         .sample_blocking(100)?; // a runtime error
///     Ok(samples.histogram("c").map_or(0, |h| h.counts().len()))
/// }
/// assert_eq!(bell_counts()?, 2);
/// # Ok::<(), oqi::Error>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// The source text doesn't parse ([`parse`](crate::parse),
    /// [`format`](crate::format)).
    Syntax(syntax::Error),
    /// The program doesn't compile ([`compile`](crate::compile),
    /// [`Compiler`](crate::Compiler)). Syntax errors met while compiling are
    /// reported here, with code `S0001`.
    Compile(CompileError),
    /// Running failed ([`Runner`](crate::Runner), [`Vm`](vm::Vm), building a
    /// backend). An unknown backend or precision name is reported here as
    /// [`BackendUnavailable`](vm::ErrorKind::BackendUnavailable).
    Runtime(vm::Error),
    /// Bytes are not loadable bytecode
    /// ([`Program::from_bytes`](crate::Program::from_bytes)).
    Decode(DecodeError),
    /// A program couldn't be encoded as bytecode
    /// ([`Program::to_bytes`](crate::Program::to_bytes)).
    Encode(EncodeError),
}

/// `Result<T, oqi::Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// The underlying error, as the diagnostic it renders as.
    fn inner(&self) -> &(dyn Diagnostic + 'static) {
        match self {
            Error::Syntax(e) => e,
            Error::Compile(e) => e,
            Error::Runtime(e) => e,
            Error::Decode(e) => e,
            Error::Encode(e) => e,
        }
    }

    fn inner_error(&self) -> &(dyn std::error::Error + 'static) {
        match self {
            Error::Syntax(e) => e,
            Error::Compile(e) => e,
            Error::Runtime(e) => e,
            Error::Decode(e) => e,
            Error::Encode(e) => e,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.inner_error(), f)
    }
}

impl std::error::Error for Error {
    /// The underlying error's source: `Error` itself is transparent.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner_error().source()
    }
}

impl Diagnostic for Error {
    fn code(&self) -> Code {
        self.inner().code()
    }

    fn message(&self) -> String {
        self.inner().message()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        self.inner().labels()
    }

    fn notes(&self) -> Vec<String> {
        self.inner().notes()
    }

    fn path(&self) -> Option<&Path> {
        self.inner().path()
    }
}

impl From<syntax::Error> for Error {
    fn from(error: syntax::Error) -> Self {
        Error::Syntax(error)
    }
}

impl From<CompileError> for Error {
    fn from(error: CompileError) -> Self {
        Error::Compile(error)
    }
}

impl From<vm::Error> for Error {
    fn from(error: vm::Error) -> Self {
        Error::Runtime(error)
    }
}

impl From<vm::ErrorKind> for Error {
    fn from(kind: vm::ErrorKind) -> Self {
        Error::Runtime(kind.into())
    }
}

/// An unknown backend or precision name, as a runtime
/// [`BackendUnavailable`](vm::ErrorKind::BackendUnavailable) error.
impl From<vm::ParseNameError> for Error {
    fn from(error: vm::ParseNameError) -> Self {
        vm::ErrorKind::BackendUnavailable(error.to_string()).into()
    }
}

impl From<DecodeError> for Error {
    fn from(error: DecodeError) -> Self {
        Error::Decode(error)
    }
}

impl From<EncodeError> for Error {
    fn from(error: EncodeError) -> Self {
        Error::Encode(error)
    }
}
