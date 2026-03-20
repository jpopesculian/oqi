use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    pub span: Span,
    pub message: String,
}

pub type Result<T> = core::result::Result<T, Error>;
