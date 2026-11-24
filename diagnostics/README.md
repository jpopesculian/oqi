# oqi-diagnostics

Stable error codes and source-annotated rendering for the
[oqi](https://crates.io/crates/oqi) OpenQASM 3 toolchain. Syntax, compile and
runtime errors implement the `Diagnostic` trait (a code such as `C0001`, a
message, labelled source spans and notes); `emit` prints one to stderr and
`render_to_string` returns it as text.

This crate is a building block of [`oqi`](https://crates.io/crates/oqi), which
re-exports `Diagnostic`, `emit` and `render_to_string`.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
