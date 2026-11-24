use std::fmt;

use crate::span::Span;

/// A lexical or syntax error: a message anchored at a source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    span: Span,
    message: String,
}

impl Error {
    /// An error with `message` at `span`.
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }

    /// Where in the source the error is.
    pub fn span(&self) -> Span {
        self.span
    }

    /// What is wrong, e.g. "expected ';'".
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// A result with a lexical or syntax [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
