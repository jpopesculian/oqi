use oqi_lex::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub usize);

/// The kind of a non-global scope. The global scope itself is implicit —
/// represented by `Symbol::scope == None` and `Scope::parent == None` —
/// because it has no introducing construct, no span, and no kind to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Subroutine,
    Gate,
    Defcal,
    For,
    While,
    IfThen,
    IfElse,
    Box,
    SwitchCase,
    DurationOf,
    Anonymous,
}

#[derive(Debug, Clone)]
pub struct Scope {
    pub id: ScopeId,
    pub kind: ScopeKind,
    /// Parent scope. `None` means the parent is the (implicit) global scope.
    pub parent: Option<ScopeId>,
    pub span: Span,
    /// Nesting level among non-global scopes. The outermost non-global scope
    /// has depth 0; each further nesting adds 1.
    pub depth: usize,
}

pub struct ScopeTable {
    scopes: Vec<Scope>,
}

impl ScopeTable {
    pub fn new() -> Self {
        Self { scopes: Vec::new() }
    }

    /// Create a new scope. `parent = None` means the scope is directly nested
    /// in the global scope.
    pub fn create(&mut self, kind: ScopeKind, parent: Option<ScopeId>, span: Span) -> ScopeId {
        let depth = parent.map(|p| self.scopes[p.0].depth + 1).unwrap_or(0);
        let id = ScopeId(self.scopes.len());
        self.scopes.push(Scope {
            id,
            kind,
            parent,
            span,
            depth,
        });
        id
    }

    pub fn get(&self, id: ScopeId) -> &Scope {
        &self.scopes[id.0]
    }

    pub fn len(&self) -> usize {
        self.scopes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Scope> {
        self.scopes.iter()
    }
}

impl Default for ScopeTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: usize, end: usize) -> Span {
        oqi_lex::span(start, end)
    }

    #[test]
    fn top_level_scope_has_no_parent_and_depth_zero() {
        let mut t = ScopeTable::new();
        let s = t.create(ScopeKind::For, None, span(0, 1));
        let scope = t.get(s);
        assert_eq!(scope.parent, None);
        assert_eq!(scope.depth, 0);
        assert_eq!(scope.kind, ScopeKind::For);
    }

    #[test]
    fn nested_scopes_track_parent_and_depth() {
        let mut t = ScopeTable::new();
        let outer = t.create(ScopeKind::For, None, span(1, 2));
        let inner = t.create(ScopeKind::IfThen, Some(outer), span(3, 4));
        assert_eq!(t.get(outer).parent, None);
        assert_eq!(t.get(outer).depth, 0);
        assert_eq!(t.get(inner).parent, Some(outer));
        assert_eq!(t.get(inner).depth, 1);
    }

    #[test]
    fn sibling_scopes_share_parent() {
        let mut t = ScopeTable::new();
        let parent = t.create(ScopeKind::Subroutine, None, span(0, 1));
        let then_s = t.create(ScopeKind::IfThen, Some(parent), span(1, 2));
        let else_s = t.create(ScopeKind::IfElse, Some(parent), span(3, 4));
        assert_eq!(t.get(then_s).parent, Some(parent));
        assert_eq!(t.get(else_s).parent, Some(parent));
        assert_ne!(then_s, else_s);
    }
}
