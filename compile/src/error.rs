use std::path::PathBuf;

use oqi_lex::Span;

use crate::types::Type;

#[derive(Debug, Clone)]
pub enum ErrorKind {
    UndefinedName(String),
    DuplicateDefinition(String),
    TypeMismatch { expected: Box<Type>, got: Box<Type> },
    NonConstantDesignator,
    NonConstantExpression,
    InvalidWidth(usize),
    IncludeNotFound(String),
    IncludeCycle(Vec<String>),
    MissingSourceContext,
    InvalidContext(String),
    InvalidGateBody(String),
    InvalidSwitch(String),
    Unsupported(String),
    InvalidLiteral(String),
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
        Self {
            kind,
            span: Span::new(0, 0),
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

pub type Result<T, E = CompileError> = std::result::Result<T, E>;

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

impl ResultExt for CompileError {
    #[inline]
    fn with_span(self, span: Span) -> Self {
        self.with_span(span)
    }
    #[inline]
    fn with_path(self, path: Option<PathBuf>) -> Self {
        self.with_path(path)
    }
}
