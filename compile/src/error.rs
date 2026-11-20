use std::path::PathBuf;

use oqi_lex::Span;

use crate::types::Type;

#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// A lexical or syntax error, carrying the parser's message.
    Syntax(String),
    UndefinedName(String),
    DuplicateDefinition {
        name: String,
        prior: Span,
    },
    TypeMismatch {
        expected: Box<Type>,
        got: Box<Type>,
    },
    /// An operator applied to operands it is not defined for; the message
    /// names the operation.
    InvalidOperands(String),
    NonConstantDesignator,
    NonConstantExpression,
    /// A type whose width or rank is out of range (`int[200]`, an array of
    /// eight dimensions).
    InvalidType(String),
    IncludeNotFound(String),
    IncludeCycle(Vec<String>),
    MissingSourceContext,
    InvalidContext(String),
    /// A declaration (named by the payload) that the spec only allows in
    /// the global scope, made in a local one.
    NotGlobalScope(&'static str),
    /// A variable, qubit or alias of the global scope named inside a gate,
    /// subroutine or defcal, which sees only the global scope's constants.
    NotVisible {
        name: String,
        declared: Span,
    },
    InvalidGateBody(String),
    Unsupported(String),
    InvalidLiteral(String),
    QubitIndexOutOfRange {
        index: i128,
        len: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CompileError {
    pub kind: ErrorKind,
    pub span: Span,
    pub path: Option<PathBuf>,
}

impl CompileError {
    #[inline]
    pub const fn new(kind: ErrorKind) -> Self {
        Self::at(kind, Span::new(0, 0))
    }

    /// An error of `kind` located at `span`.
    #[inline]
    pub const fn at(kind: ErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            path: None,
        }
    }

    #[inline]
    pub fn with_span(self, span: Span) -> Self {
        Self { span, ..self }
    }

    #[inline]
    pub fn with_path(self, path: Option<PathBuf>) -> Self {
        if self.path.is_some() || path.is_none() {
            self
        } else {
            Self { path, ..self }
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

pub type Result<T, E = CompileError> = std::result::Result<T, E>;

/// Locating an error after the fact, on the `Result` that carries it.
pub trait ResultExt {
    fn with_span(self, span: Span) -> Self;
    fn with_path(self, path: Option<PathBuf>) -> Self;
}

impl<T> ResultExt for Result<T> {
    #[inline]
    fn with_span(self, span: Span) -> Self {
        self.map_err(|e| e.with_span(span))
    }
    #[inline]
    fn with_path(self, path: Option<PathBuf>) -> Self {
        self.map_err(|e| e.with_path(path))
    }
}
