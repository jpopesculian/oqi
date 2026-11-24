//! How a [`CompileError`] is presented: each [`ErrorKind`]'s stable code
//! (`C####`, or `S0001` for syntax errors), its headline message, the label
//! under the offending span, and — where useful — secondary labels and help
//! notes. Bytecode errors are diagnostics too: [`DecodeError`] is C0018 and
//! [`EncodeError`] C0019.

use std::fmt;
use std::path::Path;

use oqi_diagnostics::{Code, DiagLabel, Diagnostic};

use crate::bytecode::{DecodeError, EncodeError};
use crate::error::{CompileError, ErrorKind};

impl ErrorKind {
    /// The kind's stable code and the label under the offending span.
    fn code_and_label(&self) -> (Code, &'static str) {
        let (num, label) = match self {
            ErrorKind::Syntax(_) => return (Code::syntax(1), "syntax error"),
            ErrorKind::UndefinedName(_) => (1, "not found in this scope"),
            ErrorKind::DuplicateDefinition { .. } => (2, "redefined here"),
            ErrorKind::TypeMismatch { .. } => (3, "type mismatch"),
            ErrorKind::InvalidOperands(_) => (12, "not defined for these types"),
            ErrorKind::NonConstantDesignator => (4, "not a constant"),
            ErrorKind::NonConstantExpression => (5, "not a constant"),
            ErrorKind::InvalidType(_) => (6, "invalid type"),
            ErrorKind::IncludeNotFound(_) => (7, "include not found"),
            ErrorKind::IncludeCycle(_) => (8, "include cycle"),
            ErrorKind::MissingSourceContext => (9, "needs a source-file context"),
            ErrorKind::InvalidContext(_) => (10, "not allowed here"),
            ErrorKind::InvalidGateBody(_) => (11, "not allowed in a gate body"),
            ErrorKind::Unsupported(_) => (13, "unsupported"),
            ErrorKind::InvalidLiteral(_) => (14, "invalid literal"),
            ErrorKind::QubitIndexOutOfRange { .. } => (15, "index out of range"),
            ErrorKind::NotGlobalScope(_) => (16, "not in the global scope"),
            ErrorKind::NotVisible { .. } => (17, "not visible in this definition"),
        };
        (Code::compile(num), label)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Syntax(msg) => write!(f, "{msg}"),
            ErrorKind::UndefinedName(name) => write!(f, "undefined name '{name}'"),
            ErrorKind::DuplicateDefinition { name, .. } => {
                write!(f, "duplicate definition of '{name}'")
            }
            ErrorKind::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            ErrorKind::InvalidOperands(msg) => write!(f, "{msg}"),
            ErrorKind::NonConstantDesignator => {
                write!(f, "designator must be a constant expression")
            }
            ErrorKind::NonConstantExpression => write!(f, "expression must be constant"),
            ErrorKind::InvalidType(msg) => write!(f, "invalid type: {msg}"),
            ErrorKind::IncludeNotFound(path) => write!(f, "include file not found: {path}"),
            ErrorKind::IncludeCycle(chain) => {
                write!(f, "include cycle: {}", chain.join(" -> "))
            }
            ErrorKind::MissingSourceContext => {
                write!(
                    f,
                    "cannot resolve relative include without source file context"
                )
            }
            ErrorKind::InvalidContext(msg) => write!(f, "{msg}"),
            ErrorKind::NotGlobalScope(what) => {
                write!(f, "{what} are only allowed in the global scope")
            }
            ErrorKind::NotVisible { name, .. } => {
                write!(f, "`{name}` is not visible inside a definition")
            }
            ErrorKind::InvalidGateBody(msg) => write!(f, "invalid gate body: {msg}"),
            ErrorKind::InvalidLiteral(msg) => write!(f, "invalid literal: {msg}"),
            ErrorKind::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            ErrorKind::QubitIndexOutOfRange { index, len } => {
                write!(
                    f,
                    "qubit index {index} out of range for register of length {len}"
                )
            }
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for CompileError {}

impl Diagnostic for CompileError {
    fn code(&self) -> Code {
        self.kind.code_and_label().0
    }

    fn message(&self) -> String {
        self.kind.to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        // No location, no labels: the renderer prints just the headline.
        let Some(span) = self.span else {
            return Vec::new();
        };
        let mut labels = vec![DiagLabel::primary(span, self.kind.code_and_label().1)];
        match &self.kind {
            ErrorKind::DuplicateDefinition { prior, .. } => {
                labels.push(DiagLabel::secondary(*prior, "first defined here"));
            }
            ErrorKind::NotVisible { declared, .. } => {
                labels.push(DiagLabel::secondary(*declared, "declared here"));
            }
            _ => {}
        }
        labels
    }

    fn notes(&self) -> Vec<String> {
        match &self.kind {
            ErrorKind::NonConstantDesignator | ErrorKind::NonConstantExpression => {
                vec!["this position requires a value known at compile time".into()]
            }
            ErrorKind::IncludeCycle(chain) => {
                vec![format!("cycle: {}", chain.join(" -> "))]
            }
            ErrorKind::NotVisible { .. } => vec![
                "a gate, subroutine or defcal sees only the constants, gates and \
                 subroutines of the global scope; pass the value as a parameter"
                    .into(),
            ],
            _ => Vec::new(),
        }
    }

    fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

/// Bytes that are not loadable bytecode: code C0018, message only (there
/// is no source to point into).
impl Diagnostic for DecodeError {
    fn code(&self) -> Code {
        Code::compile(18)
    }

    fn message(&self) -> String {
        self.to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        Vec::new()
    }
}

/// A program that could not be encoded as bytecode: code C0019, message
/// only.
impl Diagnostic for EncodeError {
    fn code(&self) -> Code {
        Code::compile(19)
    }

    fn message(&self) -> String {
        self.to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        Vec::new()
    }
}
