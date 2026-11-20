use serde::{Deserialize, Serialize};

/// Identifies a non-global scope. The global scope is implicit, represented
/// by `Symbol::scope == None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopeId(pub usize);
