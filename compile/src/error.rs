use oqi_lex::Span;

use crate::types::Type;

#[derive(Debug)]
pub struct CompileError {
    pub kind: ErrorKind,
    pub span: Span,
}

#[derive(Debug)]
pub enum ErrorKind {
    UndefinedName(String),
    DuplicateDefinition(String),
    TypeMismatch { expected: Type, got: Type },
    NonConstantDesignator,
    NonConstantExpression,
    InvalidWidth(u32),
    IncludeNotFound(String),
    IncludeCycle(Vec<String>),
    MissingSourceContext,
    InvalidContext(String),
    InvalidGateBody(String),
    InvalidSwitch(String),
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, CompileError>;
