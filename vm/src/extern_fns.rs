//! Host-provided `extern` functions.
//!
//! OpenQASM `extern` functions take classical values and return zero or
//! one classical value (docs/classical.rst). The VM marshals arguments
//! into `Value`s, looks the function up by name, and awaits the result.

use std::collections::HashMap;

use oqi_classical::Value;

use crate::error::{KindResult, VmErrorKind};

/// Supplies implementations for the program's `extern` functions.
///
/// `call` is `async` so a provider can await host I/O (e.g. a JS callback
/// returning a `Promise` in the wasm bindings); synchronous providers
/// return ready futures. Like [`QuantumBackend`](crate::QuantumBackend)'s,
/// the future need not be `Send`: the VM is driven on the calling thread.
#[allow(async_fn_in_trait)]
pub trait ExternProvider {
    /// Call `name` with `args`; return its result (or `None` for a
    /// void extern).
    async fn call(&mut self, name: &str, args: &[Value]) -> KindResult<Option<Value>>;
}

/// An [`ExternProvider`] that rejects every call. Use when a program
/// declares no externs (or you want extern calls to be a hard error).
pub struct NoExterns;

impl ExternProvider for NoExterns {
    async fn call(&mut self, name: &str, _args: &[Value]) -> KindResult<Option<Value>> {
        Err(VmErrorKind::UnknownExtern(name.to_string()))
    }
}

type ExternFn = Box<dyn FnMut(&[Value]) -> KindResult<Option<Value>>>;

/// A registry of host closures, keyed by extern name.
#[derive(Default)]
pub struct FnRegistry {
    fns: HashMap<String, ExternFn>,
}

impl FnRegistry {
    pub fn new() -> Self {
        FnRegistry::default()
    }

    /// Register `f` as the implementation of the extern `name`.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        f: impl FnMut(&[Value]) -> KindResult<Option<Value>> + 'static,
    ) -> &mut Self {
        self.fns.insert(name.into(), Box::new(f));
        self
    }
}

impl ExternProvider for FnRegistry {
    async fn call(&mut self, name: &str, args: &[Value]) -> KindResult<Option<Value>> {
        match self.fns.get_mut(name) {
            Some(f) => f(args),
            None => Err(VmErrorKind::UnknownExtern(name.to_string())),
        }
    }
}
