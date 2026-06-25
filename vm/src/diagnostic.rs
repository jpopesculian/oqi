//! [`Diagnostic`] implementation for [`VmError`], mapping each
//! [`VmErrorKind`] to a stable `R####` code, a short primary label at the
//! failing instruction's span, and — where useful — a help note.

use oqi_diagnostics::{Code, DiagLabel, Diagnostic};

use crate::error::{VmError, VmErrorKind};

impl Diagnostic for VmError {
    fn code(&self) -> Code {
        let num = match self.kind {
            VmErrorKind::UnsetRegister(_) => 1,
            VmErrorKind::MissingInput(_) => 2,
            VmErrorKind::UnknownInput(_) => 3,
            VmErrorKind::UnknownExtern(_) => 4,
            VmErrorKind::UndefinedGate(_) => 5,
            VmErrorKind::Unsupported(_) => 6,
            VmErrorKind::Type(_) => 7,
            VmErrorKind::BroadcastMismatch(_) => 8,
            VmErrorKind::Classical(_) => 9,
            VmErrorKind::QubitOutOfRange { .. } => 10,
            VmErrorKind::Unreachable => 11,
            VmErrorKind::TooManyQubits { .. } => 12,
        };
        Code::runtime(num)
    }

    fn message(&self) -> String {
        self.kind.to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        let pointer = match &self.kind {
            VmErrorKind::UnsetRegister(_) => "read before assignment",
            VmErrorKind::MissingInput(_) => "missing input",
            VmErrorKind::UnknownInput(_) => "not a declared input",
            VmErrorKind::UnknownExtern(_) => "extern not provided",
            VmErrorKind::UndefinedGate(_) => "no executable definition",
            VmErrorKind::Unsupported(_) => "unsupported here",
            VmErrorKind::Type(_) => "type error",
            VmErrorKind::BroadcastMismatch(_) => "length mismatch",
            VmErrorKind::Classical(_) => "classical error",
            VmErrorKind::QubitOutOfRange { .. } => "qubit out of range",
            VmErrorKind::Unreachable => "unreachable",
            VmErrorKind::TooManyQubits { .. } => "too many qubits",
        };
        vec![DiagLabel::primary(self.span.unwrap_or_default(), pointer)]
    }

    fn notes(&self) -> Vec<String> {
        match &self.kind {
            VmErrorKind::MissingInput(_) => {
                vec!["supply it with `--input NAME=VALUE`".into()]
            }
            VmErrorKind::UnknownExtern(name) => {
                vec![format!("no implementation is registered for extern `{name}`")]
            }
            _ => Vec::new(),
        }
    }
}
