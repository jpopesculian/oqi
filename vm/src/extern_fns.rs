//! Host-provided `extern` functions.
//!
//! OpenQASM `extern` functions take classical values and return zero or
//! one classical value (docs/classical.rst). The VM marshals arguments
//! into `Value`s, looks the function up by name, awaits the result and casts
//! it to the declared return type.

use std::collections::HashMap;
use std::fmt;

use oqi_classical::{Value, ValueTy};

use crate::error::{ErrorKind, Result};

/// Supplies implementations for the program's `extern` functions.
///
/// `call` is `async` so a provider can await host I/O (e.g. a JS callback
/// returning a `Promise` in the wasm bindings); synchronous providers
/// return ready futures. Like [`QuantumBackend`](crate::QuantumBackend)'s,
/// the future need not be `Send`: the VM is driven on the calling thread.
#[allow(async_fn_in_trait)]
pub trait ExternProvider {
    /// Call the extern `name` with `args`. `returns` is its declared return
    /// type (`None` for a void extern); the VM casts the returned value to
    /// it, so a provider may return any value castable to it — a provider
    /// converting host values uses it to pick the representation (see
    /// [`HostValue::coerce_to`](crate::HostValue::coerce_to)).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::UnknownExtern`] if the provider doesn't implement
    /// `name`, and [`ErrorKind::Extern`] (see
    /// [`ErrorKind::extern_failed`]) if the implementation fails.
    async fn call(
        &mut self,
        name: &str,
        args: &[Value],
        returns: Option<ValueTy>,
    ) -> Result<Option<Value>, ErrorKind>;
}

/// An [`ExternProvider`] that rejects every call. Use when a program
/// declares no externs (or you want extern calls to be a hard error).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoExterns;

impl ExternProvider for NoExterns {
    async fn call(
        &mut self,
        name: &str,
        _args: &[Value],
        _returns: Option<ValueTy>,
    ) -> Result<Option<Value>, ErrorKind> {
        Err(ErrorKind::UnknownExtern(name.to_string()))
    }
}

type ExternFn = Box<dyn FnMut(&[Value]) -> Result<Option<Value>, ErrorKind>>;

/// A registry of host closures, keyed by extern name.
///
/// ```
/// use oqi_vm::{FnRegistry, Value};
/// use oqi_vm::classical::iw;
///
/// let mut externs = FnRegistry::new();
/// externs.register("seven", |_args: &[Value]| Ok(Some(Value::int(7, iw(32)))));
/// ```
#[derive(Default)]
pub struct FnRegistry {
    fns: HashMap<String, ExternFn>,
}

impl FnRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        FnRegistry::default()
    }

    /// Register `f` as the implementation of the extern `name`. Its result is
    /// cast to the extern's declared return type.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        f: impl FnMut(&[Value]) -> Result<Option<Value>, ErrorKind> + 'static,
    ) -> &mut Self {
        self.fns.insert(name.into(), Box::new(f));
        self
    }
}

impl fmt::Debug for FnRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names: Vec<&str> = self.fns.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("FnRegistry")
            .field("externs", &names)
            .finish()
    }
}

impl ExternProvider for FnRegistry {
    async fn call(
        &mut self,
        name: &str,
        args: &[Value],
        _returns: Option<ValueTy>,
    ) -> Result<Option<Value>, ErrorKind> {
        match self.fns.get_mut(name) {
            Some(f) => f(args),
            None => Err(ErrorKind::UnknownExtern(name.to_string())),
        }
    }
}
