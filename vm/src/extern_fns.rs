//! Host-provided `extern` functions.
//!
//! OpenQASM `extern` functions take classical values and return zero or
//! one classical value (docs/classical.rst). The VM marshals arguments
//! into `Value`s, looks the function up by name, and awaits the result.

use std::collections::HashMap;

use async_trait::async_trait;
use oqi_classical::Value;

use crate::error::{Result, VmErrorKind};

/// Supplies implementations for the program's `extern` functions.
///
/// `call` is `async` so a provider can await host I/O (e.g. a JS callback
/// returning a `Promise` in the wasm bindings); synchronous providers
/// return ready futures. `?Send` for the same reason as
/// [`QuantumBackend`](crate::QuantumBackend).
#[async_trait(?Send)]
pub trait ExternProvider {
    /// Call `name` with `args`; return its result (or `None` for a
    /// void extern).
    async fn call(&mut self, name: &str, args: &[Value]) -> Result<Option<Value>>;
}

/// An [`ExternProvider`] that rejects every call. Use when a program
/// declares no externs (or you want extern calls to be a hard error).
pub struct NoExterns;

#[async_trait(?Send)]
impl ExternProvider for NoExterns {
    async fn call(&mut self, name: &str, _args: &[Value]) -> Result<Option<Value>> {
        Err(VmErrorKind::UnknownExtern(name.to_string()))
    }
}

type ExternFn = Box<dyn FnMut(&[Value]) -> Result<Option<Value>>>;

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
        f: impl FnMut(&[Value]) -> Result<Option<Value>> + 'static,
    ) -> &mut Self {
        self.fns.insert(name.into(), Box::new(f));
        self
    }
}

#[async_trait(?Send)]
impl ExternProvider for FnRegistry {
    async fn call(&mut self, name: &str, args: &[Value]) -> Result<Option<Value>> {
        match self.fns.get_mut(name) {
            Some(f) => f(args),
            None => Err(VmErrorKind::UnknownExtern(name.to_string())),
        }
    }
}
