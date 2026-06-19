//! [`Diagnostic`] implementation for [`CompileError`], mapping each
//! [`ErrorKind`] to a stable `C####` code, a short primary label, and — where
//! useful — secondary labels and help notes.

use std::path::Path;

use oqi_diagnostics::{Code, DiagLabel, Diagnostic};

use crate::error::{CompileError, ErrorKind};

impl Diagnostic for CompileError {
    fn code(&self) -> Code {
        let num = match self.kind {
            ErrorKind::UndefinedName(_) => 1,
            ErrorKind::DuplicateDefinition { .. } => 2,
            ErrorKind::TypeMismatch { .. } => 3,
            ErrorKind::NonConstantDesignator => 4,
            ErrorKind::NonConstantExpression => 5,
            ErrorKind::InvalidWidth(_) => 6,
            ErrorKind::IncludeNotFound(_) => 7,
            ErrorKind::IncludeCycle(_) => 8,
            ErrorKind::MissingSourceContext => 9,
            ErrorKind::InvalidContext(_) => 10,
            ErrorKind::InvalidGateBody(_) => 11,
            ErrorKind::InvalidSwitch(_) => 12,
            ErrorKind::Unsupported(_) => 13,
            ErrorKind::InvalidLiteral(_) => 14,
            ErrorKind::QubitIndexOutOfRange { .. } => 15,
        };
        Code::compile(num)
    }

    fn message(&self) -> String {
        self.kind.to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        let pointer = match &self.kind {
            ErrorKind::UndefinedName(_) => "not found in this scope",
            ErrorKind::DuplicateDefinition { .. } => "redefined here",
            ErrorKind::TypeMismatch { .. } => "type mismatch",
            ErrorKind::NonConstantDesignator | ErrorKind::NonConstantExpression => {
                "not a constant"
            }
            ErrorKind::InvalidWidth(_) => "invalid width",
            ErrorKind::IncludeNotFound(_) => "include not found",
            ErrorKind::IncludeCycle(_) => "include cycle",
            ErrorKind::MissingSourceContext => "needs a source-file context",
            ErrorKind::InvalidContext(_) => "not allowed here",
            ErrorKind::InvalidGateBody(_) => "not allowed in a gate body",
            ErrorKind::InvalidSwitch(_) => "invalid switch",
            ErrorKind::Unsupported(_) => "unsupported",
            ErrorKind::InvalidLiteral(_) => "invalid literal",
            ErrorKind::QubitIndexOutOfRange { .. } => "index out of range",
        };

        let mut labels = vec![DiagLabel::primary(self.span, pointer)];
        if let ErrorKind::DuplicateDefinition { prior, .. } = &self.kind {
            labels.push(DiagLabel::secondary(*prior, "first defined here"));
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
            _ => Vec::new(),
        }
    }

    fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}
