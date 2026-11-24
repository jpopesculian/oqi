use std::borrow::Cow;
use std::collections::HashMap;
use std::f64::consts;
use std::path::{Component, Path, PathBuf};

use oqi_lex::Span;

use crate::classical::FloatWidth;
use crate::classical::{Value, ValueTy};
use crate::error::{CompileError, ErrorKind, Result};
use crate::scope::ScopeId;
use crate::sir::{CallTarget, Intrinsic};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::{CompileOptions, Type};

#[derive(Debug, Clone)]
pub enum IncludeSource {
    Lib(String),
    Path(PathBuf),
}

/// Where `include` statements get their text.
///
/// `include "stdgates.inc";` names a library ([`resolve_lib`](Self::resolve_lib));
/// any other name is a path, relative to the including file
/// ([`resolve_path`](Self::resolve_path)). The defaults read paths from the
/// filesystem and serve the embedded OpenQASM standard library, so a
/// resolver only overrides what it needs: serving files from memory, say, or
/// refusing file includes ([`LibOnlyResolver`]).
pub trait IncludeResolver {
    /// The text of the file at `path`. Defaults to reading it from disk.
    fn resolve_path(
        &self,
        path: &Path,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Ok(Cow::Owned(std::fs::read_to_string(path)?))
    }

    /// The text of the library `lib`. Defaults to serving the OpenQASM 3
    /// standard library, `stdgates.inc`.
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

impl<T: IncludeResolver + ?Sized> IncludeResolver for &T {
    fn resolve_path(
        &self,
        path: &Path,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        (**self).resolve_path(path)
    }

    fn resolve_lib(
        &self,
        lib: &str,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        (**self).resolve_lib(lib)
    }
}

/// Reads included files from disk and serves the standard library: the
/// [`IncludeResolver`] defaults, and what [`Compiler`](crate::Compiler) uses
/// unless told otherwise.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultIncludeResolver;

impl IncludeResolver for DefaultIncludeResolver {}

/// Serves only embedded libraries (`stdgates.inc`); file includes are an
/// error. For hosts with no filesystem to include from (wasm, embedded
/// language bindings).
#[derive(Debug, Default, Clone, Copy)]
pub struct LibOnlyResolver;

impl IncludeResolver for LibOnlyResolver {
    fn resolve_path(
        &self,
        path: &Path,
    ) -> std::result::Result<Cow<'_, str>, Box<dyn std::error::Error>> {
        Err(format!(
            "cannot include \"{}\": file includes are not supported in this build",
            path.display()
        )
        .into())
    }
}

pub struct Resolver<'r> {
    symbols: SymbolTable,
    /// Number of non-global scopes opened so far; the next [`ScopeId`].
    scope_count: usize,
    /// Name → SymbolId chain. `lookup_stack[0]` is always the global lookup
    /// map; further entries correspond one-to-one with `scope_stack`. Hence
    /// `lookup_stack.len() == scope_stack.len() + 1`.
    lookup_stack: Vec<HashMap<String, SymbolId>>,
    /// IDs of currently active *non-global* scopes, innermost last. Empty
    /// while resolving at the global scope.
    scope_stack: Vec<ScopeId>,
    /// The names of the cal scope ([`ScopeId::CAL`]).
    cal_names: HashMap<String, SymbolId>,
    /// While resolving a `cal` block or `defcal`: the length of
    /// `lookup_stack` where it starts. The scopes from there on are its
    /// own, and see the cal scope before the scopes around it.
    cal_start: Option<usize>,
    /// While resolving a gate, subroutine or defcal (docs/scope.rst): the
    /// variables, qubits and aliases of the global scope are not visible.
    in_definition: bool,
    include_stack: Vec<PathBuf>,
    options: CompileOptions,
    include_resolver: &'r dyn IncludeResolver,
}

impl<'r> Resolver<'r> {
    pub fn new(include_resolver: &'r dyn IncludeResolver, options: CompileOptions) -> Self {
        let mut symbols = SymbolTable::new();
        let mut global = HashMap::new();

        // Built-in symbols are conceptually global — `scope = None`.
        // Seed built-in constants
        let tau_id = symbols.insert(
            "tau".into(),
            SymbolKind::Const,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            Default::default(),
            None,
        );
        symbols.set_const_value(tau_id, Value::float(consts::TAU, FloatWidth::F64));
        global.insert("tau".into(), tau_id);
        global.insert("τ".into(), tau_id);

        let pi_id = symbols.insert(
            "pi".into(),
            SymbolKind::Const,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            Default::default(),
            None,
        );
        symbols.set_const_value(pi_id, Value::float(consts::PI, FloatWidth::F64));
        global.insert("pi".into(), pi_id);
        global.insert("π".into(), pi_id);

        let euler_id = symbols.insert(
            "euler".into(),
            SymbolKind::Const,
            Type::Classical(ValueTy::float(FloatWidth::F64)),
            Default::default(),
            None,
        );
        symbols.set_const_value(euler_id, Value::float(consts::E, FloatWidth::F64));
        global.insert("euler".into(), euler_id);
        global.insert("ℇ".into(), euler_id);

        // Seed built-in gate U
        let u_id = symbols.insert(
            "U".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        global.insert("U".into(), u_id);

        // Seed built-in gate gphase
        let gphase_id = symbols.insert(
            "gphase".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        global.insert("gphase".into(), gphase_id);

        Self {
            symbols,
            // `ScopeId(0)` is the cal scope.
            scope_count: ScopeId::CAL.0 + 1,
            lookup_stack: vec![global],
            scope_stack: Vec::new(),
            cal_names: HashMap::new(),
            cal_start: None,
            in_definition: false,
            include_stack: Vec::new(),
            options,
            include_resolver,
        }
    }

    pub fn push_scope(&mut self) -> ScopeId {
        let id = ScopeId(self.scope_count);
        self.scope_count += 1;
        self.scope_stack.push(id);
        self.lookup_stack.push(HashMap::new());
        id
    }

    pub fn pop_scope(&mut self) {
        assert!(!self.scope_stack.is_empty(), "cannot pop global scope");
        self.scope_stack.pop();
        self.lookup_stack.pop();
    }

    /// Start resolving a `cal` block or `defcal`: the cal scope comes into
    /// view, and — until a scope is opened, as a `defcal` does for its
    /// body — declarations go into it. Returns what [`leave_cal`] restores.
    ///
    /// [`leave_cal`]: Self::leave_cal
    pub fn enter_cal(&mut self) -> Option<usize> {
        self.cal_start.replace(self.lookup_stack.len())
    }

    pub fn leave_cal(&mut self, outer: Option<usize>) {
        self.cal_start = outer;
    }

    /// Start (`true`) or finish resolving the parameters and body of a gate,
    /// subroutine or defcal. Definitions are global, so they do not nest.
    pub fn set_in_definition(&mut self, inside: bool) {
        self.in_definition = inside;
    }

    /// `Some(id)` for non-global scopes, `None` at global scope. Use this
    /// when declaring a symbol — it's exactly the value to store in
    /// `Symbol::scope`.
    pub fn current_scope(&self) -> Option<ScopeId> {
        self.scope_stack.last().copied()
    }

    pub fn declare(
        &mut self,
        name: &str,
        kind: SymbolKind,
        ty: Type,
        span: Span,
    ) -> Result<SymbolId> {
        // The top level of a `cal` block declares into the cal scope. The
        // program's qubits, inputs and outputs are no values of the
        // block's: they are declared as usual.
        let cal = self.cal_start == Some(self.lookup_stack.len())
            && !matches!(
                kind,
                SymbolKind::Qubit | SymbolKind::Input | SymbolKind::Output
            );
        let scope = match cal {
            true => Some(ScopeId::CAL),
            false => self.current_scope(),
        };
        let names = match cal {
            true => &mut self.cal_names,
            false => self.lookup_stack.last_mut().unwrap(),
        };
        if let Some(&prior_id) = names.get(name) {
            let prior = self.symbols.get(prior_id).span;
            return Err(CompileError::at(
                ErrorKind::DuplicateDefinition {
                    name: name.to_string(),
                    prior,
                },
                span,
            ));
        }
        let id = self.symbols.insert(name.to_string(), kind, ty, span, scope);
        names.insert(name.to_string(), id);
        Ok(id)
    }

    pub fn resolve(&self, name: &str, span: Span) -> Result<SymbolId> {
        let innermost = |scopes: &[HashMap<String, SymbolId>]| {
            scopes
                .iter()
                .rev()
                .find_map(|scope| scope.get(name).copied())
        };
        let (around, own) = self
            .lookup_stack
            .split_at(self.cal_start.unwrap_or(self.lookup_stack.len()));
        let cal = || {
            self.cal_start
                .and_then(|_| self.cal_names.get(name).copied())
        };
        let id = innermost(own)
            .or_else(cal)
            .or_else(|| innermost(around))
            .ok_or_else(|| CompileError::at(ErrorKind::UndefinedName(name.to_string()), span))?;

        // "Globally scoped variables without the `const` modifier are not
        // visible inside the definition", nor are the program's qubits.
        let symbol = self.symbols.get(id);
        let run_time = matches!(
            symbol.kind,
            SymbolKind::Variable
                | SymbolKind::Input
                | SymbolKind::Output
                | SymbolKind::Qubit
                | SymbolKind::Alias
        );
        if self.in_definition && symbol.scope.is_none() && run_time {
            let kind = ErrorKind::NotVisible {
                name: name.to_string(),
                declared: symbol.span,
            };
            return Err(CompileError::at(kind, span));
        }
        Ok(id)
    }

    pub fn resolve_call(&self, name: &str, span: Span) -> Result<CallTarget> {
        if let Some(intrinsic) = Intrinsic::lookup(name) {
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
            .ok_or(CompileError::at(ErrorKind::MissingSourceContext, span))?;
        let dir = base.parent().unwrap_or(base);
        Ok(IncludeSource::Path(normalize_path(dir.join(path))))
    }

    pub fn resolve_source(&self, source: &IncludeSource, span: Span) -> Result<Cow<'_, str>> {
        match source {
            IncludeSource::Lib(name) => self.include_resolver.resolve_lib(name).map_err(|e| {
                CompileError::at(ErrorKind::IncludeNotFound(format!("{name}: {e}")), span)
            }),
            IncludeSource::Path(path) => self.include_resolver.resolve_path(path).map_err(|e| {
                CompileError::at(
                    ErrorKind::IncludeNotFound(format!("{}: {e}", path.display())),
                    span,
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::iw;
    use crate::classical::{PrimitiveTy, Value};
    use std::path::Path;

    fn span(start: usize, end: usize) -> Span {
        oqi_lex::Span::new(start, end)
    }

    fn default_resolver() -> Resolver<'static> {
        Resolver::new(&DefaultIncludeResolver, CompileOptions::default())
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

        // pi is a float constant
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

        // tau is a float constant
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
        let ErrorKind::DuplicateDefinition { name, prior } = &err.kind else {
            panic!("expected DuplicateDefinition, got {:?}", err.kind);
        };
        assert_eq!(name, "x");
        // The secondary label points back at the first declaration.
        assert_eq!(*prior, span(0, 1));
        // Primary ("redefined here") plus secondary ("first defined here").
        assert_eq!(oqi_diagnostics::Diagnostic::labels(&err).len(), 2);
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
                Type::Classical(ValueTy::int(iw(32))),
                span(2, 3),
            )
            .unwrap();
        assert_eq!(r.resolve("x", Default::default()).unwrap(), inner);

        r.pop_scope();
        assert_eq!(r.resolve("x", Default::default()).unwrap(), outer);
    }

    #[test]
    fn the_cal_scope_is_seen_from_cal_blocks_and_defcals_only() {
        let mut r = default_resolver();
        let bool_var = |r: &mut Resolver, name: &str| {
            let ty = Type::Classical(ValueTy::bool());
            r.declare(name, SymbolKind::Variable, ty, span(0, 1))
        };
        let outside = bool_var(&mut r, "x").unwrap();

        // A `cal` block declares into the cal scope, at its top level.
        let outer = r.enter_cal();
        let shared = bool_var(&mut r, "shared").unwrap();
        let shadow = bool_var(&mut r, "x").unwrap();
        assert_eq!(r.symbols().get(shared).scope, Some(ScopeId::CAL));
        assert_eq!(r.resolve("x", span(0, 1)).unwrap(), shadow);
        let block = r.push_scope();
        let nested = bool_var(&mut r, "nested").unwrap();
        assert_eq!(r.symbols().get(nested).scope, Some(block));
        r.pop_scope();
        r.leave_cal(outer);

        assert!(r.resolve("shared", span(0, 1)).is_err());
        assert_eq!(r.resolve("x", span(0, 1)).unwrap(), outside);

        // A `defcal` sees the cal scope and declares into its own.
        let outer = r.enter_cal();
        let body = r.push_scope();
        assert_eq!(r.resolve("shared", span(0, 1)).unwrap(), shared);
        let local = bool_var(&mut r, "local").unwrap();
        assert_eq!(r.symbols().get(local).scope, Some(body));
        r.pop_scope();
        r.leave_cal(outer);

        let outer = r.enter_cal();
        assert!(r.resolve("local", span(0, 1)).is_err());
        assert!(bool_var(&mut r, "shared").is_err());
        r.leave_cal(outer);
    }

    #[test]
    fn declared_symbol_carries_current_scope() {
        let mut r = default_resolver();
        // Global declaration → scope = None
        let g = r
            .declare(
                "g",
                SymbolKind::Variable,
                Type::Classical(ValueTy::bool()),
                span(0, 1),
            )
            .unwrap();
        assert_eq!(r.symbols().get(g).scope, None);

        // Inside a scope → scope = Some(for_id)
        let for_id = r.push_scope();
        let l = r
            .declare(
                "l",
                SymbolKind::Variable,
                Type::Classical(ValueTy::bool()),
                span(4, 5),
            )
            .unwrap();
        assert_eq!(r.symbols().get(l).scope, Some(for_id));

        // Nested scope → its own id
        let if_id = r.push_scope();
        let n = r
            .declare(
                "n",
                SymbolKind::Variable,
                Type::Classical(ValueTy::bool()),
                span(8, 9),
            )
            .unwrap();
        assert_eq!(r.symbols().get(n).scope, Some(if_id));
        assert_ne!(if_id, for_id);

        r.pop_scope();
        r.pop_scope();
    }

    #[test]
    fn builtins_have_no_scope() {
        let r = default_resolver();
        for name in ["pi", "tau", "euler", "U", "gphase"] {
            let id = r.resolve(name, Default::default()).unwrap();
            assert_eq!(
                r.symbols().get(id).scope,
                None,
                "built-in {name} should be global"
            );
        }
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
        assert!(matches!(
            target,
            CallTarget::Intrinsic(Intrinsic::Unary(crate::sir::UnOp::Sin))
        ));
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
        let r = Resolver::new(&DefaultIncludeResolver, opts);
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
        let mut r = Resolver::new(&DefaultIncludeResolver, opts);
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
        assert_eq!(err.span, Some(span));
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
