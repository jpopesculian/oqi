use std::borrow::Cow;
use std::collections::HashMap;
use std::f64::consts;
use std::path::{Component, Path, PathBuf};

use oqi_lex::Span;

use crate::classical::{Value, ValueTy};
use crate::error::{CompileError, ErrorKind, Result};
use crate::sir::{CallTarget, Intrinsic};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::{CompileOptions, FloatWidth, Type};

#[derive(Debug, Clone)]
pub enum IncludeSource {
    Lib(String),
    Path(PathBuf),
}

pub trait IncludeResolver {
    fn resolve_path(
        &self,
        path: &Path,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Ok(Cow::Owned(std::fs::read_to_string(path)?))
    }

    fn resolve_lib(
        &self,
        lib: &str,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        match lib {
            "stdgates.inc" => Ok(Cow::Borrowed(include_str!("./stdgates.inc"))),
            other => Err(format!("unknown library: {other}").into()),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultIncludeResolver;

impl IncludeResolver for DefaultIncludeResolver {}

pub struct Resolver {
    symbols: SymbolTable,
    scopes: Vec<HashMap<String, SymbolId>>,
    include_stack: Vec<PathBuf>,
    options: CompileOptions,
    include_resolver: Box<dyn IncludeResolver>,
}

impl Resolver {
    pub fn new(include_resolver: impl IncludeResolver + 'static, options: CompileOptions) -> Self {
        let mut symbols = SymbolTable::new();
        let mut global = HashMap::new();

        // Seed built-in constants
        // tau = 0 (full turn wraps to zero)
        let tau_id = symbols.insert(
            "tau".into(),
            SymbolKind::Const,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            Default::default(),
        );
        symbols.set_const_value(tau_id, Value::float(consts::TAU, FloatWidth::F64));
        global.insert("tau".into(), tau_id);
        global.insert("τ".into(), tau_id);

        // pi = 1 << (width - 1) (half turn)
        let pi_id = symbols.insert(
            "pi".into(),
            SymbolKind::Const,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            Default::default(),
        );
        symbols.set_const_value(pi_id, Value::float(consts::PI, FloatWidth::F64));
        global.insert("pi".into(), pi_id);
        global.insert("π".into(), pi_id);

        // euler's number has no exact angle representation
        let euler_id = symbols.insert(
            "euler".into(),
            SymbolKind::Const,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            Default::default(),
        );
        symbols.set_const_value(euler_id, Value::float(consts::E, FloatWidth::F64));
        global.insert("euler".into(), euler_id);
        global.insert("ℇ".into(), euler_id);

        // Seed built-in gate U
        let u_id = symbols.insert("U".into(), SymbolKind::Gate, Type::Void, Default::default());
        global.insert("U".into(), u_id);

        // Seed built-in gate gphase
        let gphase_id = symbols.insert(
            "gphase".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
        );
        global.insert("gphase".into(), gphase_id);

        Self {
            symbols,
            scopes: vec![global],
            include_stack: Vec::new(),
            options,
            include_resolver: Box::new(include_resolver),
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
            return Err(
                CompileError::new(ErrorKind::DuplicateDefinition(name.to_string())).with_span(span),
            );
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
        Err(CompileError::new(ErrorKind::UndefinedName(name.to_string())).with_span(span))
    }

    pub fn resolve_call(&self, name: &str, span: Span) -> Result<CallTarget> {
        if let Some(intrinsic) = lookup_intrinsic(name) {
            return Ok(CallTarget::Intrinsic(intrinsic));
        }
        let id = self.resolve(name, span)?;
        Ok(CallTarget::Symbol(id))
    }

    pub fn current_source_path(&self) -> Option<&Path> {
        self.include_stack
            .last()
            .map(PathBuf::as_path)
            .or(self.options.source_name.as_deref())
    }

    pub fn classify_include(&self, path: &str, span: Span) -> Result<IncludeSource> {
        if is_library_name(path) {
            return Ok(IncludeSource::Lib(path.to_string()));
        }
        let base = self
            .current_source_path()
            .ok_or(CompileError::new(ErrorKind::MissingSourceContext).with_span(span))?;
        let dir = base.parent().unwrap_or(base);
        Ok(IncludeSource::Path(normalize_path(dir.join(path))))
    }

    pub fn resolve_source(&self, source: &IncludeSource, span: Span) -> Result<Cow<'_, str>> {
        match source {
            IncludeSource::Lib(name) => self.include_resolver.resolve_lib(name).map_err(|e| {
                CompileError::new(ErrorKind::IncludeNotFound(format!("{name}: {e}")))
                    .with_span(span)
            }),
            IncludeSource::Path(path) => self.include_resolver.resolve_path(path).map_err(|e| {
                CompileError::new(ErrorKind::IncludeNotFound(format!(
                    "{}: {e}",
                    path.display()
                )))
                .with_span(span)
            }),
        }
    }

    pub fn push_include(&mut self, path: PathBuf) -> Result<()> {
        let path = normalize_path(path);
        if self.include_stack.contains(&path) {
            let chain: Vec<String> = self
                .include_stack
                .iter()
                .chain(std::iter::once(&path))
                .map(|p| p.display().to_string())
                .collect();
            return Err(CompileError::new(ErrorKind::IncludeCycle(chain)));
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

fn is_library_name(path: &str) -> bool {
    !path.is_empty() && !path.contains('/') && !path.contains('\\')
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match normalized.components().next_back() {
                Some(Component::Normal(_)) => {
                    normalized.pop();
                }
                Some(Component::ParentDir) | None => normalized.push(component.as_os_str()),
                Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                Some(Component::CurDir) => unreachable!(),
            },
            other => normalized.push(other.as_os_str()),
        }
    }

    normalized
}

pub(crate) fn lookup_intrinsic(name: &str) -> Option<Intrinsic> {
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
    use crate::classical::bw;
    use crate::classical::{PrimitiveTy, Value};
    use std::path::Path;

    fn span(start: usize, end: usize) -> Span {
        oqi_lex::span(start, end)
    }

    fn default_resolver() -> Resolver {
        Resolver::new(DefaultIncludeResolver, CompileOptions::default())
    }

    #[test]
    fn seed_builtins_available() {
        let r = default_resolver();
        assert!(r.resolve("pi", Default::default()).is_ok());
        assert!(r.resolve("tau", Default::default()).is_ok());
        assert!(r.resolve("euler", Default::default()).is_ok());
        assert!(r.resolve("U", Default::default()).is_ok());

        // Unicode aliases resolve to same symbol
        assert_eq!(
            r.resolve("π", Default::default()).unwrap(),
            r.resolve("pi", Default::default()).unwrap()
        );
        assert_eq!(
            r.resolve("τ", Default::default()).unwrap(),
            r.resolve("tau", Default::default()).unwrap()
        );
        assert_eq!(
            r.resolve("ℇ", Default::default()).unwrap(),
            r.resolve("euler", Default::default()).unwrap()
        );

        // pi is an angle with bit (width-1) set
        let pi_id = r.resolve("pi", Default::default()).unwrap();
        let pi_sym = r.symbols().get(pi_id);
        assert_eq!(pi_sym.kind, SymbolKind::Const);
        assert_eq!(pi_sym.ty, Type::Classical(ValueTy::float(FloatWidth::F64)));
        match &pi_sym.const_value {
            Some(Value::Scalar(scalar)) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Float(FloatWidth::F64));
                assert_eq!(scalar.value().as_float(FloatWidth::F64), Some(consts::PI));
            }
            other => panic!("expected Float::PI, got {other:?}"),
        }

        // tau is an angle with value 0
        let tau_id = r.resolve("tau", Default::default()).unwrap();
        let tau_sym = r.symbols().get(tau_id);
        assert_eq!(tau_sym.kind, SymbolKind::Const);
        assert_eq!(tau_sym.ty, Type::Classical(ValueTy::float(FloatWidth::F64)));
        match &tau_sym.const_value {
            Some(Value::Scalar(scalar)) => {
                assert_eq!(scalar.ty(), PrimitiveTy::Float(FloatWidth::F64));
                assert_eq!(scalar.value().as_float(FloatWidth::F64), Some(consts::TAU));
            }
            other => panic!("expected Float::TAU, got {other:?}"),
        }
    }

    #[test]
    fn declare_and_resolve() {
        let mut r = default_resolver();
        let id = r
            .declare(
                "x",
                SymbolKind::Variable,
                Type::Classical(ValueTy::bool()),
                span(0, 1),
            )
            .unwrap();
        let resolved = r.resolve("x", span(0, 1)).unwrap();
        assert_eq!(id, resolved);
    }

    #[test]
    fn undeclared_name_errors() {
        let r = default_resolver();
        let err = r.resolve("nonexistent", span(5, 10)).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UndefinedName(ref n) if n == "nonexistent"));
    }

    #[test]
    fn duplicate_in_same_scope_errors() {
        let mut r = default_resolver();
        r.declare(
            "x",
            SymbolKind::Variable,
            Type::Classical(ValueTy::bool()),
            span(0, 1),
        )
        .unwrap();
        let err = r
            .declare(
                "x",
                SymbolKind::Variable,
                Type::Classical(ValueTy::bool()),
                span(2, 3),
            )
            .unwrap_err();
        assert!(matches!(err.kind, ErrorKind::DuplicateDefinition(ref n) if n == "x"));
    }

    #[test]
    fn shadowing_across_scopes() {
        let mut r = default_resolver();
        let outer = r
            .declare(
                "x",
                SymbolKind::Variable,
                Type::Classical(ValueTy::bool()),
                span(0, 1),
            )
            .unwrap();

        r.push_scope();
        let inner = r
            .declare(
                "x",
                SymbolKind::Variable,
                Type::Classical(ValueTy::int(bw(32))),
                span(2, 3),
            )
            .unwrap();
        assert_eq!(r.resolve("x", Default::default()).unwrap(), inner);

        r.pop_scope();
        assert_eq!(r.resolve("x", Default::default()).unwrap(), outer);
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
            let target = r.resolve_call(name, Default::default()).unwrap();
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
            .declare("my_sub", SymbolKind::Subroutine, Type::Void, span(0, 5))
            .unwrap();
        let target = r.resolve_call("my_sub", Default::default()).unwrap();
        assert!(matches!(target, CallTarget::Symbol(sid) if sid == id));
    }

    #[test]
    fn resolve_call_unknown() {
        let r = default_resolver();
        let err = r.resolve_call("unknown_fn", span(0, 5)).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UndefinedName(_)));
    }

    #[test]
    fn resolve_call_intrinsic_priority() {
        let mut r = default_resolver();
        // Declare a symbol named "sin" — intrinsic should still win
        r.declare("sin", SymbolKind::Subroutine, Type::Void, span(0, 3))
            .unwrap();
        let target = r.resolve_call("sin", Default::default()).unwrap();
        assert!(matches!(target, CallTarget::Intrinsic(Intrinsic::Sin)));
    }

    #[test]
    fn classify_stdgates_as_lib() {
        let r = default_resolver();
        let source = r
            .classify_include("stdgates.inc", Default::default())
            .unwrap();
        match source {
            IncludeSource::Lib(name) => assert_eq!(name, "stdgates.inc"),
            IncludeSource::Path(_) => panic!("expected Lib"),
        }
    }

    #[test]
    fn resolve_lib_stdgates_content() {
        let r = default_resolver();
        let src = IncludeSource::Lib("stdgates.inc".to_string());
        let content = r.resolve_source(&src, Default::default()).unwrap();
        assert!(content.contains("gate h"));
    }

    #[test]
    fn classify_relative_path() {
        let opts = CompileOptions {
            source_name: Some(PathBuf::from("/project/src/main.qasm")),
            ..Default::default()
        };
        let r = Resolver::new(DefaultIncludeResolver, opts);
        let source = r
            .classify_include("./utils.qasm", Default::default())
            .unwrap();
        match source {
            IncludeSource::Path(p) => assert_eq!(p, Path::new("/project/src/utils.qasm")),
            IncludeSource::Lib(_) => panic!("expected Path"),
        }
    }

    #[test]
    fn classify_relative_to_current_include_scope() {
        let opts = CompileOptions {
            source_name: Some(PathBuf::from("/project/root/main.qasm")),
            ..Default::default()
        };
        let mut r = Resolver::new(DefaultIncludeResolver, opts);
        r.push_include(PathBuf::from("/project/file/1/path"))
            .unwrap();
        let source = r.classify_include("../2/path", Default::default()).unwrap();
        match source {
            IncludeSource::Path(p) => assert_eq!(p, Path::new("/project/file/2/path")),
            IncludeSource::Lib(_) => panic!("expected Path"),
        }
    }

    #[test]
    fn classify_missing_source_context() {
        let r = default_resolver();
        let span = span(7, 18);
        let err = r.classify_include("./other.qasm", span).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::MissingSourceContext));
        assert_eq!(err.span, span);
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
