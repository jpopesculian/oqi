use std::collections::HashMap;
use std::path::PathBuf;

use oqi_lex::Span;

use awint::{bw, Awi};

use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{CallTarget, Intrinsic};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::{CompileOptions, FloatWidth, Type};
use crate::value::{ConstValue, FloatValue};

#[derive(Debug)]
pub enum IncludeSource {
    Embedded(&'static str),
    File(PathBuf),
}

pub struct Resolver {
    symbols: SymbolTable,
    scopes: Vec<HashMap<String, SymbolId>>,
    include_stack: Vec<PathBuf>,
    options: CompileOptions,
}

impl Resolver {
    pub fn new(options: CompileOptions) -> Self {
        let mut symbols = SymbolTable::new();
        let mut global = HashMap::new();

        let angle_width = options.system_angle_width;

        // Seed built-in constants
        // tau = 0 (full turn wraps to zero)
        let tau_id = symbols.insert(
            "tau".into(),
            SymbolKind::Const,
            Type::Angle(angle_width),
            0..0,
        );
        symbols.set_const_value(tau_id, ConstValue::Angle(Awi::zero(bw(angle_width as usize))));
        global.insert("tau".into(), tau_id);
        global.insert("τ".into(), tau_id);

        // pi = 1 << (width - 1) (half turn)
        let mut pi_val = Awi::zero(bw(angle_width as usize));
        pi_val.set(angle_width as usize - 1, true).unwrap();
        let pi_id = symbols.insert(
            "pi".into(),
            SymbolKind::Const,
            Type::Angle(angle_width),
            0..0,
        );
        symbols.set_const_value(pi_id, ConstValue::Angle(pi_val));
        global.insert("pi".into(), pi_id);
        global.insert("π".into(), pi_id);

        // euler's number has no exact angle representation
        let euler_id = symbols.insert(
            "euler".into(),
            SymbolKind::Const,
            Type::Float(FloatWidth::F64),
            0..0,
        );
        symbols.set_const_value(
            euler_id,
            ConstValue::Float(FloatValue::F64(std::f64::consts::E)),
        );
        global.insert("euler".into(), euler_id);
        global.insert("ℇ".into(), euler_id);

        // Seed built-in gate U
        let u_id = symbols.insert("U".into(), SymbolKind::Gate, Type::Void, 0..0);
        global.insert("U".into(), u_id);

        Self {
            symbols,
            scopes: vec![global],
            include_stack: Vec::new(),
            options,
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        assert!(self.scopes.len() > 1, "cannot pop global scope");
        self.scopes.pop();
    }

    pub fn scope_depth(&self) -> usize {
        self.scopes.len() - 1
    }

    pub fn is_global_scope(&self) -> bool {
        self.scopes.len() == 1
    }

    pub fn declare(
        &mut self,
        name: &str,
        kind: SymbolKind,
        ty: Type,
        span: Span,
    ) -> Result<SymbolId> {
        let current = self.scopes.last().unwrap();
        if current.contains_key(name) {
            return Err(CompileError {
                kind: ErrorKind::DuplicateDefinition(name.to_string()),
                span,
            });
        }
        let id = self.symbols.insert(name.to_string(), kind, ty, span);
        self.scopes.last_mut().unwrap().insert(name.to_string(), id);
        Ok(id)
    }

    pub fn resolve(&self, name: &str, span: Span) -> Result<SymbolId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Ok(id);
            }
        }
        Err(CompileError {
            kind: ErrorKind::UndefinedName(name.to_string()),
            span,
        })
    }

    pub fn resolve_call(&self, name: &str, span: Span) -> Result<CallTarget> {
        if let Some(intrinsic) = lookup_intrinsic(name) {
            return Ok(CallTarget::Intrinsic(intrinsic));
        }
        let id = self.resolve(name, span)?;
        Ok(CallTarget::Symbol(id))
    }

    pub fn resolve_include_path(&self, path: &str) -> Result<IncludeSource> {
        if path == "stdgates.inc" {
            return Ok(IncludeSource::Embedded(include_str!("./stdgates.inc")));
        }
        let base = self.options.source_name.as_ref().ok_or(CompileError {
            kind: ErrorKind::MissingSourceContext,
            span: 0..0,
        })?;
        let dir = base.parent().unwrap_or(base.as_path());
        Ok(IncludeSource::File(dir.join(path)))
    }

    pub fn push_include(&mut self, path: PathBuf) -> Result<()> {
        if self.include_stack.contains(&path) {
            let chain: Vec<String> = self
                .include_stack
                .iter()
                .chain(std::iter::once(&path))
                .map(|p| p.display().to_string())
                .collect();
            return Err(CompileError {
                kind: ErrorKind::IncludeCycle(chain),
                span: 0..0,
            });
        }
        self.include_stack.push(path);
        Ok(())
    }

    pub fn pop_include(&mut self) {
        assert!(!self.include_stack.is_empty(), "include stack is empty");
        self.include_stack.pop();
    }

    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    pub fn symbols_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbols
    }

    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    pub fn into_symbols(self) -> SymbolTable {
        self.symbols
    }
}

fn lookup_intrinsic(name: &str) -> Option<Intrinsic> {
    match name {
        "sin" => Some(Intrinsic::Sin),
        "cos" => Some(Intrinsic::Cos),
        "tan" => Some(Intrinsic::Tan),
        "arcsin" => Some(Intrinsic::Arcsin),
        "arccos" => Some(Intrinsic::Arccos),
        "arctan" => Some(Intrinsic::Arctan),
        "exp" => Some(Intrinsic::Exp),
        "log" => Some(Intrinsic::Log),
        "sqrt" => Some(Intrinsic::Sqrt),
        "ceiling" => Some(Intrinsic::Ceiling),
        "floor" => Some(Intrinsic::Floor),
        "mod" => Some(Intrinsic::Mod),
        "popcount" => Some(Intrinsic::Popcount),
        "sizeof" => Some(Intrinsic::Sizeof),
        "rotl" => Some(Intrinsic::Rotl),
        "rotr" => Some(Intrinsic::Rotr),
        "real" => Some(Intrinsic::Real),
        "imag" => Some(Intrinsic::Imag),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn default_resolver() -> Resolver {
        Resolver::new(CompileOptions::default())
    }

    #[test]
    fn seed_builtins_available() {
        let r = default_resolver();
        assert!(r.resolve("pi", 0..0).is_ok());
        assert!(r.resolve("tau", 0..0).is_ok());
        assert!(r.resolve("euler", 0..0).is_ok());
        assert!(r.resolve("U", 0..0).is_ok());

        // Unicode aliases resolve to same symbol
        assert_eq!(
            r.resolve("π", 0..0).unwrap(),
            r.resolve("pi", 0..0).unwrap()
        );
        assert_eq!(
            r.resolve("τ", 0..0).unwrap(),
            r.resolve("tau", 0..0).unwrap()
        );
        assert_eq!(
            r.resolve("ℇ", 0..0).unwrap(),
            r.resolve("euler", 0..0).unwrap()
        );

        // pi is an angle with bit (width-1) set
        let pi_id = r.resolve("pi", 0..0).unwrap();
        let pi_sym = r.symbols().get(pi_id);
        assert_eq!(pi_sym.kind, SymbolKind::Const);
        assert_eq!(pi_sym.ty, Type::Angle(usize::BITS));
        match &pi_sym.const_value {
            Some(ConstValue::Angle(awi)) => {
                assert_eq!(awi.bw(), usize::BITS as usize);
                assert!(awi.get(usize::BITS as usize - 1).unwrap());
            }
            other => panic!("expected Angle, got {other:?}"),
        }

        // tau is an angle with value 0
        let tau_id = r.resolve("tau", 0..0).unwrap();
        let tau_sym = r.symbols().get(tau_id);
        assert_eq!(tau_sym.kind, SymbolKind::Const);
        assert_eq!(tau_sym.ty, Type::Angle(usize::BITS));
        match &tau_sym.const_value {
            Some(ConstValue::Angle(awi)) => assert!(awi.is_zero()),
            other => panic!("expected Angle(0), got {other:?}"),
        }
    }

    #[test]
    fn declare_and_resolve() {
        let mut r = default_resolver();
        let id = r
            .declare("x", SymbolKind::Variable, Type::Bool, 0..1)
            .unwrap();
        let resolved = r.resolve("x", 0..1).unwrap();
        assert_eq!(id, resolved);
    }

    #[test]
    fn undeclared_name_errors() {
        let r = default_resolver();
        let err = r.resolve("nonexistent", 5..10).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UndefinedName(ref n) if n == "nonexistent"));
    }

    #[test]
    fn duplicate_in_same_scope_errors() {
        let mut r = default_resolver();
        r.declare("x", SymbolKind::Variable, Type::Bool, 0..1)
            .unwrap();
        let err = r
            .declare("x", SymbolKind::Variable, Type::Bool, 2..3)
            .unwrap_err();
        assert!(matches!(err.kind, ErrorKind::DuplicateDefinition(ref n) if n == "x"));
    }

    #[test]
    fn shadowing_across_scopes() {
        let mut r = default_resolver();
        let outer = r
            .declare("x", SymbolKind::Variable, Type::Bool, 0..1)
            .unwrap();

        r.push_scope();
        let inner = r
            .declare(
                "x",
                SymbolKind::Variable,
                Type::Int {
                    width: 32,
                    signed: true,
                },
                2..3,
            )
            .unwrap();
        assert_eq!(r.resolve("x", 0..0).unwrap(), inner);

        r.pop_scope();
        assert_eq!(r.resolve("x", 0..0).unwrap(), outer);
    }

    #[test]
    fn scope_depth() {
        let mut r = default_resolver();
        assert_eq!(r.scope_depth(), 0);
        assert!(r.is_global_scope());

        r.push_scope();
        assert_eq!(r.scope_depth(), 1);
        assert!(!r.is_global_scope());

        r.pop_scope();
        assert_eq!(r.scope_depth(), 0);
        assert!(r.is_global_scope());
    }

    #[test]
    fn resolve_call_intrinsic() {
        let r = default_resolver();
        let intrinsics = [
            "sin", "cos", "tan", "arcsin", "arccos", "arctan", "exp", "log", "sqrt", "ceiling",
            "floor", "mod", "popcount", "sizeof", "rotl", "rotr", "real", "imag",
        ];
        for name in intrinsics {
            let target = r.resolve_call(name, 0..0).unwrap();
            assert!(
                matches!(target, CallTarget::Intrinsic(_)),
                "expected intrinsic for {name}"
            );
        }
    }

    #[test]
    fn resolve_call_lexical() {
        let mut r = default_resolver();
        let id = r
            .declare("my_sub", SymbolKind::Subroutine, Type::Void, 0..5)
            .unwrap();
        let target = r.resolve_call("my_sub", 0..0).unwrap();
        assert!(matches!(target, CallTarget::Symbol(sid) if sid == id));
    }

    #[test]
    fn resolve_call_unknown() {
        let r = default_resolver();
        let err = r.resolve_call("unknown_fn", 0..5).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UndefinedName(_)));
    }

    #[test]
    fn resolve_call_intrinsic_priority() {
        let mut r = default_resolver();
        // Declare a symbol named "sin" — intrinsic should still win
        r.declare("sin", SymbolKind::Subroutine, Type::Void, 0..3)
            .unwrap();
        let target = r.resolve_call("sin", 0..0).unwrap();
        assert!(matches!(target, CallTarget::Intrinsic(Intrinsic::Sin)));
    }

    #[test]
    fn include_stdgates_embedded() {
        let r = default_resolver();
        let source = r.resolve_include_path("stdgates.inc").unwrap();
        match source {
            IncludeSource::Embedded(content) => assert!(content.contains("gate h")),
            IncludeSource::File(_) => panic!("expected Embedded"),
        }
    }

    #[test]
    fn include_relative_path() {
        let opts = CompileOptions {
            source_name: Some(PathBuf::from("/project/src/main.qasm")),
            ..Default::default()
        };
        let r = Resolver::new(opts);
        let source = r.resolve_include_path("utils.qasm").unwrap();
        match source {
            IncludeSource::File(p) => assert_eq!(p, Path::new("/project/src/utils.qasm")),
            IncludeSource::Embedded(_) => panic!("expected File"),
        }
    }

    #[test]
    fn include_missing_source_context() {
        let r = default_resolver();
        let err = r.resolve_include_path("other.qasm").unwrap_err();
        assert!(matches!(err.kind, ErrorKind::MissingSourceContext));
    }

    #[test]
    fn include_cycle_detection() {
        let mut r = default_resolver();
        let a = PathBuf::from("a.qasm");
        let b = PathBuf::from("b.qasm");

        r.push_include(a.clone()).unwrap();
        r.push_include(b.clone()).unwrap();

        // a is already in the stack → cycle
        let err = r.push_include(a.clone()).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::IncludeCycle(_)));

        // After popping b, then a, pushing a again should succeed
        r.pop_include();
        r.pop_include();
        assert!(r.push_include(a).is_ok());
    }

    #[test]
    fn include_no_dedup() {
        let mut r = default_resolver();
        let path = PathBuf::from("lib.qasm");

        r.push_include(path.clone()).unwrap();
        r.pop_include();
        // Re-including after pop is allowed (textual inclusion semantics)
        assert!(r.push_include(path).is_ok());
    }

    #[test]
    #[should_panic(expected = "cannot pop global scope")]
    fn pop_scope_panics_on_global() {
        let mut r = default_resolver();
        r.pop_scope();
    }
}
