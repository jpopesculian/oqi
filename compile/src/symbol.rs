use oqi_lex::Span;

use crate::types::Type;
use crate::value::ConstValue;

/// Unique identifier for a symbol in the symbol table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

/// Central symbol table holding all declared names.
pub struct SymbolTable {
    symbols: Vec<Symbol>,
}

pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub kind: SymbolKind,
    pub ty: Type,
    pub span: Span,
    pub const_value: Option<ConstValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// `const int[32] N = 10;`
    Const,
    /// `int[8] x;`, `bit[4] c;`
    Variable,
    /// `input float[64] theta;`
    Input,
    /// `output bit[4] result;`
    Output,
    /// `qubit[4] q;` or `qubit q;`
    Qubit,
    /// `let alias = q[0:1] ++ q[3:4];`
    Alias,
    /// Classical parameter of a gate definition: `gate rx(θ) q { ... }`
    GateParam,
    /// Qubit parameter of a gate definition: `gate cx a, b { ... }`
    GateQubit,
    /// Parameter of a subroutine definition: `def f(int[32] x, qubit q) { ... }`
    SubroutineParam,
    /// Loop variable: `for uint i in [0:3] { ... }`
    LoopVar,
    /// Gate name: `gate h a { ... }`
    Gate,
    /// Subroutine name: `def f(...) { ... }`
    Subroutine,
    /// Extern function name: `extern get_param(...) -> ...`
    Extern,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
        }
    }

    /// Insert a new symbol and return its ID.
    pub fn insert(
        &mut self,
        name: String,
        kind: SymbolKind,
        ty: Type,
        span: Span,
    ) -> SymbolId {
        let id = SymbolId(self.symbols.len() as u32);
        self.symbols.push(Symbol {
            id,
            name,
            kind,
            ty,
            span,
            const_value: None,
        });
        id
    }

    /// Set the constant value for a symbol.
    pub fn set_const_value(&mut self, id: SymbolId, value: ConstValue) {
        self.symbols[id.0 as usize].const_value = Some(value);
    }

    /// Get a symbol by ID.
    pub fn get(&self, id: SymbolId) -> &Symbol {
        &self.symbols[id.0 as usize]
    }

    /// Get a mutable reference to a symbol by ID.
    pub fn get_mut(&mut self, id: SymbolId) -> &mut Symbol {
        &mut self.symbols[id.0 as usize]
    }

    /// Look up a symbol by name (linear scan; the resolver will maintain
    /// scope-aware lookup in later phases).
    pub fn lookup(&self, name: &str) -> Option<SymbolId> {
        self.symbols
            .iter()
            .rev()
            .find(|s| s.name == name)
            .map(|s| s.id)
    }

    /// Number of symbols in the table.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Iterate over all symbols.
    pub fn iter(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.iter()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut table = SymbolTable::new();
        let id = table.insert("x".to_string(), SymbolKind::Variable, Type::Bool, 0..1);

        let sym = table.get(id);
        assert_eq!(sym.id, id);
        assert_eq!(sym.name, "x");
        assert_eq!(sym.kind, SymbolKind::Variable);
        assert_eq!(sym.ty, Type::Bool);
        assert_eq!(sym.span, 0..1);
        assert!(sym.const_value.is_none());
    }

    #[test]
    fn test_set_const_value() {
        let mut table = SymbolTable::new();
        let id = table.insert(
            "N".to_string(),
            SymbolKind::Const,
            Type::Int {
                width: 32,
                signed: false,
            },
            0..5,
        );

        let val = ConstValue::Bool(true);
        table.set_const_value(id, val);

        let sym = table.get(id);
        assert!(sym.const_value.is_some());
        match &sym.const_value {
            Some(ConstValue::Bool(true)) => {}
            _ => panic!("expected Bool(true)"),
        }
    }

    #[test]
    fn test_multiple_inserts() {
        let mut table = SymbolTable::new();
        let id0 = table.insert("a".to_string(), SymbolKind::Variable, Type::Bool, 0..1);
        let id1 = table.insert("b".to_string(), SymbolKind::Const, Type::Duration, 2..3);
        let id2 = table.insert(
            "q".to_string(),
            SymbolKind::Qubit,
            Type::QubitReg(4),
            4..10,
        );

        assert_eq!(id0, SymbolId(0));
        assert_eq!(id1, SymbolId(1));
        assert_eq!(id2, SymbolId(2));
        assert_eq!(table.len(), 3);

        assert_eq!(table.get(id0).name, "a");
        assert_eq!(table.get(id1).name, "b");
        assert_eq!(table.get(id2).name, "q");
    }

    #[test]
    fn test_lookup_by_name() {
        let mut table = SymbolTable::new();
        let id = table.insert("theta".to_string(), SymbolKind::Input, Type::Float(crate::types::FloatWidth::F64), 0..5);

        assert_eq!(table.lookup("theta"), Some(id));
        assert_eq!(table.lookup("missing"), None);
    }

    #[test]
    fn test_lookup_returns_latest() {
        let mut table = SymbolTable::new();
        let _id0 = table.insert("x".to_string(), SymbolKind::Variable, Type::Bool, 0..1);
        let id1 = table.insert(
            "x".to_string(),
            SymbolKind::Variable,
            Type::Int {
                width: 32,
                signed: true,
            },
            2..3,
        );

        // lookup returns the latest entry (shadowing)
        assert_eq!(table.lookup("x"), Some(id1));
    }

    #[test]
    fn test_get_mut() {
        let mut table = SymbolTable::new();
        let id = table.insert("y".to_string(), SymbolKind::Variable, Type::Bit, 0..1);

        let sym = table.get_mut(id);
        sym.ty = Type::BitReg(8);

        assert_eq!(table.get(id).ty, Type::BitReg(8));
    }

    #[test]
    fn test_iter() {
        let mut table = SymbolTable::new();
        table.insert("a".to_string(), SymbolKind::Variable, Type::Bool, 0..1);
        table.insert("b".to_string(), SymbolKind::Const, Type::Duration, 2..3);

        let names: Vec<&str> = table.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_empty_table() {
        let table = SymbolTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.lookup("anything"), None);
    }
}
