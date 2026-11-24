# oqi-compile

The [OpenQASM 3](https://openqasm.com) compiler of
[oqi](https://crates.io/crates/oqi): it resolves includes, type-checks and
lowers a program to bytecode that [`oqi-vm`](https://crates.io/crates/oqi-vm)
runs. Compiled programs can be saved and loaded again.

Most users want the [`oqi`](https://crates.io/crates/oqi) crate, which
re-exports the supported API of this one.

```rust
fn main() -> Result<(), oqi_compile::CompileError> {
    let program = oqi_compile::compile("include \"stdgates.inc\"; qubit q; h q;")?;
    assert_eq!(program.num_qubits(), 1);
    Ok(())
}
```

This crate ships the OpenQASM standard library (`stdgates.inc`) from the
[OpenQASM project](https://github.com/openqasm/openqasm), under the Apache
License, Version 2.0; see NOTICE.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
