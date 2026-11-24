# Changelog

All notable changes to oqi are recorded here. The crates are versioned
together; this project follows [Semantic Versioning](https://semver.org).

## 0.1.0

The first release.

- **Compiler** (`oqi-compile`): OpenQASM 3.0, including OpenPulse calibrations,
  checked against the specification and the reference grammar, compiled to a
  register bytecode. Compiled programs save to and load from a versioned
  binary format (`OQIB`, format version 1) that is validated on loading.
  `durationof` and `stretch` resolve at compile time from a timing table, or
  at run time.
- **Simulators** (`oqi-vm`): scalar, SIMD and multi-threaded state vectors in
  `f64` or `f32`; a WebGPU state vector; a stabilizer tableau that moves to an
  exact sum over Cliffords and, when needed, a dense state vector; and a
  matrix-product-state simulator. Shots are sampled from the final state when
  no gate depends on a measurement.
- **Extension points**: custom quantum backends, host `extern` functions and
  OpenPulse handlers.
- **`oqi`**: one crate for Rust users, with blocking helpers and a unified
  error type.
- **`oqi` command line** (`oqi-cli`): `compile`, `run` (with shots, seeds,
  inputs, timing tables and every backend) and `fmt`.
- **Bindings**: WebAssembly/JavaScript (`js/`, with TypeScript types) and Python
  (`py/`, with type stubs), with the same options and backend names as the
  command line, and a browser playground.
- **Diagnostics**: every error has a stable code (`S0001`, `C####`, `R####`)
  and renders against its source.
- **Formatter** (`oqi-format`): keeps comments, idempotent, with a compact
  mode.
