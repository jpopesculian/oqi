/// Identifies a non-global scope. The global scope is implicit, represented
/// by `Symbol::scope == None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub usize);

impl ScopeId {
    /// The scope `cal` blocks and `defcal` bodies share (docs/pulses.rst,
    /// "Inline calibration blocks"): what a `cal` block declares at its top
    /// level is visible to every `cal` block and `defcal` body, and to
    /// nothing else.
    pub const CAL: ScopeId = ScopeId(0);
}
