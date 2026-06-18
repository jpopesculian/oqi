use std::fmt;

/// Errors raised while executing a bytecode module.
#[derive(Debug)]
pub enum VmError {
    /// A register was read before being assigned a value.
    UnsetRegister(u32),
    /// A `Call` targeted an `extern` the provider doesn't implement.
    UnknownExtern(String),
    /// A `GateCall` referenced a gate with no executable definition
    /// (not a built-in and no lifted gate body).
    UndefinedGate(String),
    /// A construct the MVP doesn't execute yet.
    Unsupported(String),
    /// A value had the wrong shape/type for the operation.
    Type(String),
    /// A broadcast gate call mixed register operands of differing lengths
    /// (all non-singleton operands must share one length).
    BroadcastMismatch(Vec<usize>),
    /// An error surfaced by the classical value layer.
    Classical(oqi_classical::Error),
    /// Execution reached a block marked unreachable.
    Unreachable,
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::UnsetRegister(r) => write!(f, "register r{r} read before assignment"),
            VmError::UnknownExtern(name) => write!(f, "extern function `{name}` is not provided"),
            VmError::UndefinedGate(name) => write!(f, "gate `{name}` has no executable definition"),
            VmError::Unsupported(what) => write!(f, "unsupported: {what}"),
            VmError::Type(msg) => write!(f, "type error: {msg}"),
            VmError::BroadcastMismatch(lengths) => write!(
                f,
                "gate broadcast over registers of differing lengths: {lengths:?}"
            ),
            VmError::Classical(e) => write!(f, "classical error: {e:?}"),
            VmError::Unreachable => write!(f, "reached an unreachable block"),
        }
    }
}

impl std::error::Error for VmError {}

impl From<oqi_classical::Error> for VmError {
    fn from(e: oqi_classical::Error) -> Self {
        VmError::Classical(e)
    }
}

pub type Result<T> = std::result::Result<T, VmError>;
