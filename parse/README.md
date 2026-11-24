# oqi-parse

A parser for [OpenQASM 3](https://openqasm.com), including the OpenPulse
calibration grammar. It produces a syntax tree that borrows from the source,
with spans on every node, and conforms to the reference ANTLR grammar.

This crate is a building block of [`oqi`](https://crates.io/crates/oqi), an
OpenQASM 3 compiler and simulator, which re-exports it as `oqi::syntax`.

```rust
fn main() -> Result<(), oqi_parse::Error> {
    let program = oqi_parse::parse("OPENQASM 3.0; qubit[2] q; h q[0];")?;
    assert_eq!(program.body.len(), 2);
    Ok(())
}
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
