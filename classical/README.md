# oqi-classical

The classical values, types and operators of [OpenQASM 3](https://openqasm.com):
integers of up to 128 bits, floats, angles, complex numbers, durations, bits
and bit registers of any width, and arrays of them, with the language's casts,
operators and built-in functions.

This crate is a building block of [`oqi`](https://crates.io/crates/oqi), an
OpenQASM 3 compiler and simulator, which re-exports it as `oqi::classical`:
program inputs, outputs and `extern` arguments are these values.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
