use std::path::PathBuf;

use oqi_lex::Span;

/// What went wrong. Each kind has a stable diagnostic code (`C####`, or
/// `S0001` for a syntax error), listed with its message in
/// `diagnostic.rs`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A lexical or syntax error, carrying the parser's message. `S0001`.
    Syntax(String),
    /// A name that is not declared in any visible scope. `C0001`.
    UndefinedName(String),
    /// A name declared twice in the same scope; `prior` is the first
    /// declaration. `C0002`.
    DuplicateDefinition {
        /// The name declared twice.
        name: String,
        /// Where it was first declared.
        prior: Span,
    },
    /// A value whose type is not the one its position requires. `C0003`.
    TypeMismatch {
        /// The type the position requires.
        expected: String,
        /// The type the value has.
        got: String,
    },
    /// An operator or built-in function applied to operands it is not
    /// defined for; the message names the operation. `C0012`.
    InvalidOperands(String),
    /// A type designator (`int[n]`, `qubit[n]`) that is not a compile-time
    /// constant. `C0004`.
    NonConstantDesignator,
    /// An expression that must be a compile-time constant but is not.
    /// `C0005`.
    NonConstantExpression,
    /// A type whose width or rank is out of range (`int[200]`, an array of
    /// eight dimensions). `C0006`.
    InvalidType(String),
    /// An `include` whose file or library could not be read. `C0007`.
    IncludeNotFound(String),
    /// Files that include each other; the chain, outermost first. `C0008`.
    IncludeCycle(Vec<String>),
    /// A relative `include` in a source that has no path to resolve it
    /// against. `C0009`.
    MissingSourceContext,
    /// A construct used where the language does not allow it; the message
    /// says which and why. `C0010`.
    InvalidContext(String),
    /// A declaration (named by the payload) that the spec only allows in
    /// the global scope, made in a local one. `C0016`.
    NotGlobalScope(&'static str),
    /// A variable, qubit or alias of the global scope named inside a gate,
    /// subroutine or defcal, which sees only the global scope's constants.
    /// `C0017`.
    NotVisible {
        /// The name used.
        name: String,
        /// Where it is declared.
        declared: Span,
    },
    /// A statement a gate body may not contain. `C0011`.
    InvalidGateBody(String),
    /// Valid OpenQASM that this compiler does not support. `C0013`.
    Unsupported(String),
    /// A literal that cannot be read (a malformed number, bit string or
    /// duration). `C0014`.
    InvalidLiteral(String),
    /// A qubit index outside its register. `C0015`.
    QubitIndexOutOfRange {
        /// The index used.
        index: i128,
        /// The register's length.
        len: usize,
    },
}

/// A compile error: what went wrong, and where.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// What went wrong.
    pub kind: ErrorKind,
    /// The source range it is about, if it is about one.
    pub span: Option<Span>,
    /// The file `span` refers to, when it is not the file being compiled
    /// (an error inside an `include`d file).
    pub path: Option<PathBuf>,
}

impl CompileError {
    /// An error with no location.
    #[inline]
    pub const fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            span: None,
            path: None,
        }
    }

    /// An error of `kind` located at `span`.
    #[inline]
    pub const fn at(kind: ErrorKind, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
            path: None,
        }
    }

    /// This error, located at `span`.
    #[inline]
    pub fn with_span(self, span: Span) -> Self {
        Self {
            span: Some(span),
            ..self
        }
    }

    /// This error, in the file at `path` unless it already names one: an
    /// error that arose in an included file keeps that file when it
    /// propagates out through the `include`.
    #[inline]
    pub fn or_path(self, path: Option<PathBuf>) -> Self {
        Self {
            path: self.path.or(path),
            ..self
        }
    }
}

impl From<ErrorKind> for CompileError {
    #[inline]
    fn from(kind: ErrorKind) -> Self {
        CompileError::new(kind)
    }
}

impl From<oqi_lex::Error> for CompileError {
    fn from(error: oqi_lex::Error) -> Self {
        CompileError::at(ErrorKind::Syntax(error.message), error.span)
    }
}

/// A `Result` whose error defaults to [`CompileError`].
pub type Result<T, E = CompileError> = std::result::Result<T, E>;

/// Locating an error after the fact, on the `Result` that carries it.
pub(crate) trait ResultExt {
    fn with_span(self, span: Span) -> Self;
    fn or_path(self, path: Option<PathBuf>) -> Self;
}

impl<T> ResultExt for Result<T> {
    #[inline]
    fn with_span(self, span: Span) -> Self {
        self.map_err(|e| e.with_span(span))
    }
    #[inline]
    fn or_path(self, path: Option<PathBuf>) -> Self {
        self.map_err(|e| e.or_path(path))
    }
}
