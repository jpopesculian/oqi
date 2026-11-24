//! [`Diagnostic`] implementation for [`Error`], mapping each
//! [`ErrorKind`] to a stable `R####` code, a short primary label at the
//! failing instruction's span, and — where useful — a help note.

use oqi_diagnostics::{Code, DiagLabel, Diagnostic};

use crate::error::{Error, ErrorKind};

impl Diagnostic for Error {
    fn code(&self) -> Code {
        let num = match self.kind {
            ErrorKind::UnsetRegister(_) => 1,
            ErrorKind::MissingInput(_) => 2,
            ErrorKind::UnknownInput(_) => 3,
            ErrorKind::UnknownExtern(_) => 4,
            ErrorKind::UndefinedGate(_) => 5,
            ErrorKind::Unsupported(_) => 6,
            ErrorKind::Type(_) => 7,
            ErrorKind::BroadcastMismatch(_) => 8,
            ErrorKind::Classical(_) => 9,
            ErrorKind::QubitOutOfRange { .. } => 10,
            ErrorKind::Unreachable => 11,
            ErrorKind::TooManyQubits { .. } => 12,
            ErrorKind::Pulse(_) => 13,
            ErrorKind::Extern { .. } => 14,
            ErrorKind::RankOverflow { .. } => 15,
            ErrorKind::BackendUnavailable(_) => 16,
            ErrorKind::Backend(_) => 17,
            ErrorKind::CallDepth { .. } => 18,
        };
        Code::runtime(num)
    }

    fn message(&self) -> String {
        self.kind.to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        let pointer = match &self.kind {
            ErrorKind::UnsetRegister(_) => "read before assignment",
            ErrorKind::MissingInput(_) => "missing input",
            ErrorKind::UnknownInput(_) => "not a declared input",
            ErrorKind::UnknownExtern(_) => "extern not provided",
            ErrorKind::UndefinedGate(_) => "no executable definition",
            ErrorKind::Unsupported(_) => "unsupported here",
            ErrorKind::Type(_) => "type error",
            ErrorKind::BroadcastMismatch(_) => "length mismatch",
            ErrorKind::Classical(_) => "classical error",
            ErrorKind::QubitOutOfRange { .. } => "qubit out of range",
            ErrorKind::Unreachable => "unreachable",
            ErrorKind::TooManyQubits { .. } => "too many qubits",
            ErrorKind::Pulse(_) => "pulse error",
            ErrorKind::Extern { .. } => "extern call failed",
            ErrorKind::RankOverflow { .. } => "sum-over-Cliffords budget exhausted",
            ErrorKind::BackendUnavailable(_) => "backend unavailable",
            ErrorKind::Backend(_) => "backend error",
            ErrorKind::CallDepth { .. } => "nested too deeply",
        };
        vec![DiagLabel::primary(self.span.unwrap_or_default(), pointer)]
    }

    fn notes(&self) -> Vec<String> {
        match &self.kind {
            ErrorKind::MissingInput(_) => {
                vec!["every declared `input` needs a value when the program is run".into()]
            }
            ErrorKind::UnknownExtern(name) => {
                vec![format!(
                    "no implementation is registered for extern `{name}`"
                )]
            }
            ErrorKind::RankOverflow { max_rank, .. } => {
                vec![format!(
                    "each non-Clifford gate multiplies the stabilizer term \
                     count; the budget is {max_rank}"
                )]
            }
            _ => Vec::new(),
        }
    }
}
