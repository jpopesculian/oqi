# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`oqi` is an OpenQASM 3.0 compiler and quantum simulator, written as a Rust
workspace (edition 2024). It lexes, parses, type-checks, and lowers OpenQASM
(including the OpenPulse grammar) to a bytecode IR, then interprets that bytecode
on a choice of simulators (dense state vectors on CPU or WebGPU, a stabilizer /
sum-over-Cliffords router, matrix product states). The single binary is `oqi`
(built from `cli/`, which uses the `oqi` facade crate); `js/` and `py/` are the
WebAssembly and Python bindings.

## Commands

The dev machine has no swap: a runaway allocation freezes it. Run tests,
benches and the binary under a memory cap, e.g.
`systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 -- cargo test`.

```bash
cargo build                       # build the whole workspace
cargo test                        # run all tests
cargo test -p oqi-compile         # test a single crate
cargo test -p oqi-compile teleport  # run a single test by name
cargo fmt
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --exclude oqi-js --exclude oqi-py
wasm-pack test --node js          # the wasm bindings
(cd py && maturin develop && python -m pytest tests)  # the Python bindings, in a venv

# Run the CLI
cargo run -p oqi-cli -- compile fixtures/qasm/teleport.qasm --dump      # bytecode
cargo run -p oqi-cli -- compile fixtures/qasm/teleport.qasm --dump-sir  # structured IR
cargo run -p oqi-cli -- compile fixtures/qasm/teleport.qasm -o t.oqib   # bytecode file
cargo run -p oqi-cli -- run t.oqib                                      # run bytecode
cargo run -p oqi-cli -- run fixtures/qasm/adder.qasm --state
cargo run -p oqi-cli -- run prog.qasm --input n=3 --input theta=0.5
cargo run -p oqi-cli -- run prog.qasm --shots 1000 --seed 7 --backend auto
cargo run -p oqi-cli -- fmt --stdout fixtures/qasm/qft.qasm
```

## Compilation pipeline

The end-to-end flow (see `compile/src/bytecode/mod.rs` and `cli/src/main.rs`):

```
source → parse → resolve (includes) → lower → cfg → bytecode → vm
```

- **`lex/`** (`oqi-lex`) — lexer producing `Token`s with spans. `Token` derives
  `logos::Logos` for the default mode; `Lexer` adds the grammar's other modes
  (version, strings, pragmas, calibration blocks) by hand.
- **`parse/`** (`oqi-parse`) — hand-written parser; AST in `parse/src/ast.rs`.
  `parse` and `parse_with_comments` are the whole public API. Every list goes
  through `comma_list`, which implements the grammar's trailing-comma rule.
- **`compile/`** (`oqi-compile`) — the heart. `oqi_compile::compile(source)`
  is the main entry point (source → `Program`, the compiled bytecode
  program); `Compiler` is its builder (include resolver, `dt`, system width,
  a `TableTimings` for compile-time `durationof`), and
  `Compiler::compile_to_sir` stops at the SIR, which `bytecode::emit` lowers
  the rest of the way. Errors are `CompileError { kind, span: Option<Span>,
  path }`. The pipeline stages (`cfg`, `qubits`, `lower`, `resolve`) are
  crate-private; `sir`, `types`, `symbol`, `scope`, `duration` and
  `openpulse` are `#[doc(hidden)]` public for tests and debugging output, and
  the bytecode instruction set (`bytecode::{Op, Operand, Terminator, …}`) is
  `#[doc(hidden)]` — it is the VM's interface, not a stable API. Inside
  `bytecode/`, the emitter and disassembler refer to its types as `bc::*`
  to keep them apart from the SIR/CFG types of the same names. Key modules:
  - `lower.rs` — AST → SIR, name resolution, type checking.
  - `resolve.rs` — `include` resolution; `stdgates.inc` is embedded via
    `include_str!` and served as a library include.
  - `sir.rs` — Structured IR (the `Program` with gates, subroutines, externs,
    calibrations, body). One statement/expression tree serves every stage;
    passes walk it with `Expr::children`, `Stmt::exprs` and `Stmt::bodies`
    (and their `_mut` twins) and only spell out the nodes they treat
    specially.
  - `cfg.rs` → `bytecode/` — control-flow graph (blocks of straight-line SIR
    statements ending in terminators), then the flat register bytecode
    (`bytecode::emit`): one register per classical variable per frame, no
    SSA. `box` is a scope of its enclosing body; an inline `cal` body and a
    `durationof` scope are procedures that run on the enclosing frame.
    `Program::to_bytes`/`from_bytes` (`bytecode/binary.rs`) read and write
    the file format: `OQIB`, a `u16` `FORMAT_VERSION` (bump it whenever the
    encoding of a `Program` changes), then postcard. Every decoded program
    goes through `bytecode/validate.rs`, so untrusted bytes can't send the
    VM out of bounds; the emitter checks its own output against it in debug
    builds. `bytecode/disasm.rs` is the textual disassembler.
  - `types.rs`, `symbol.rs`, `scope.rs` — type system, symbol table, scoping.
  - `openpulse.rs` — OpenPulse types (port, frame, waveform).
- **`classical/`** (`oqi-classical`) — classical value/type system (ints up to
  128 bits, floats, complex, arrays, bit registers of any width) and the
  operators under `classical/src/ops/`. An operator is a value (`ops::BinOp`,
  `ops::UnOp`): `return_ty` type-checks it and `apply` evaluates it, both from
  the same per-operator signature, so the SIR, the bytecode (`Op::Binary` /
  `Op::Unary`) and the VM all carry the one enum and never dispatch on it
  themselves. `PrimitiveTy::cast` is the spec's allowed-casts table; operator
  promotions (e.g. `angle + float`) are separate and not casts. Used by both
  compile and vm. Its `serde` feature (and `oqi-lex`'s, for `Span`) exists for
  the bytecode format; compile turns both on.
- **`vm/`** (`oqi-vm`) — interprets bytecode. `Runner` (`vm/src/runner.rs`) is
  the usual entry point: `Runner::new(&program).backend(spec).seed(s)
  .input(name, value).run()` / `.sample(shots)`; one seed drives both the
  backend and shot sampling, and results carry output names. `Vm` is the
  interpreter underneath, for custom backend instances and calibration
  handlers. Extension points: `QuantumBackend` (the simulators `StateVectorSim`,
  `SimdSim`, `AutoSim`, `MpsSim`, `GpuSim`, or a hardware provider) and
  `ExternProvider` (host implementations of `extern` functions; the VM casts
  their results to the declared return type). The compiler lowers every gate to
  the built-in `U`/`gphase`, so a backend implements only those plus
  measure/reset. On a backend, applying a gate is synchronous and fallible
  (`u`, and an uncontrolled `gphase` — the VM lowers a controlled one to `U`
  calls in `vm/src/backend.rs`); measurement, reset, shot sampling and
  amplitude readback are `async` and fallible. Every backend is built as
  `X::new(num_qubits)?.with_seed(..)` plus its own `with_*` options. Errors are
  `oqi_vm::{Error, ErrorKind, Result}`: extension points return
  `Result<T, ErrorKind>`, the VM adds the span. What the front doors share:
  `BackendSpec` (backend selection by name), `HostValue::coerce_to` (host
  values ↔ typed values, `vm/src/host.rs`) and `Runner`.
  Procedure bodies nest at most `MAX_CALL_DEPTH` deep; interpreting one
  level takes ~5 KiB of native stack in release and ~50 KiB in debug, so
  every 16th level runs on a fresh stack segment (`stacker`) instead.
  `vm/src/state.rs` holds the dense state vector and the `U`-with-modifiers
  matrix the dense backends share; compile's flat qubit memory and registers
  are in `compile/src/register.rs`. Features: `parallel` (rayon kernels) and
  `mps` (the `MpsSim` backend, pulls in `faer`) are on by default; `gpu`
  (wgpu) is opt-in. A backend whose feature is off still parses by name and
  fails to build with "backend unavailable". `GpuSim` (`vm/src/gpu.rs`)
  splits the state into chunk buffers no larger than the device's binding
  limit and keeps measurement (a reduction kernel, then a collapse kernel),
  reset and shot sampling on the device; GPU tests take turns creating a
  device (`spec::gpu_turn`, llvmpipe crashes otherwise) and force small
  chunks with `GpuSim::with_chunk_limit`.
- **`oqi/`** (`oqi`) — the facade crate and the supported Rust API: re-exports
  the everyday items of compile, vm, parse, format and diagnostics at its
  root (`compile`, `Runner`, `Program`, backends, `format`, …), the crates
  themselves as modules (`syntax`, `compiler`, `vm`, `classical`,
  `diagnostics`), and one `oqi::Error` every step's error converts into.
  Its default `blocking` feature adds `Runner::run_blocking` /
  `sample_blocking` (vm's `blocking` feature) and `oqi::block_on`. The crate
  doc's examples (and the README it includes) are doctests.
- **`cli/`**, **`js/`** (wasm-bindgen), **`py/`** (PyO3) — the three front
  doors. Keep them thin: native value ↔ `HostValue` mapping and error mapping
  only; anything two of them need belongs in `vm` or `compile`. The CLI
  depends on the `oqi` facade only; js and py use the individual crates (the
  async API, without default features).
- **`format/`** (`oqi-format`) — source formatter (the `fmt` subcommand):
  `format(source, Config)`. Output ends with one newline and keeps comments
  (compact mode drops them); formatting is idempotent.
- **`diagnostics/`** (`oqi-diagnostics`) — `ariadne`-based rendering. Errors
  implement the `Diagnostic` trait and carry stable codes (`C####` compile,
  `R####` runtime, `S0001` syntax). Spans are byte offsets.

## Diagnostics & error codes

`CompileError` (`compile/src/error.rs`) has an `ErrorKind` enum; each variant maps
to a stable numeric code in `compile/src/diagnostic.rs` (`C0001` = undefined name,
etc.; a syntax error keeps the parser's `S0001`). When adding an `ErrorKind`, add
its code and label, its message, and any help text in `diagnostic.rs`. Build
located errors with `CompileError::at(kind, span)`. `oqi_diagnostics::emit`
renders an error against source. `oqi_diagnostics::render_to_string` is the
testable variant.

## Testing & fixtures

- `fixtures/qasm/*.qasm` — real OpenQASM programs (teleport, adder, qft, vqe, …)
  used as end-to-end compile/run test inputs (`compile/tests/fixtures.rs`,
  `vm/tests/run.rs`).
- `fixtures/lexer/*.json` and `fixtures/parser/*.json` — golden token/CST dumps
  generated from the reference **ANTLR** grammar (`lexer.g4`, `parser.g4`). The
  lex/parse test suites (`lex/tests/fixtures.rs`, `parse/tests/fixtures.rs`)
  compare our output against these to stay grammar-conformant.
- Regenerate golden fixtures with the Python scripts (need ANTLR runtime; they use
  PEP-723 inline deps, run with `uv run` or equivalent):
  `scripts/generate_lexer_fixtures.py`, `scripts/generate_parser_fixtures.py`.
  `scripts/antlr_generated/` holds the generated reference lexer/parser.
- `classical/src/ops/tests/` holds the operator tests, one module per operator;
  `agreement.rs` there checks that type-checking and evaluation accept exactly
  the same operands.

## Releasing

The published crates share `[workspace.package]` (one version, license
`MIT OR Apache-2.0`, `rust-version`) and depend on each other through
`[workspace.dependencies]` (path + version). Publish order: `oqi-lex`,
`oqi-classical`, `oqi-parse`, `oqi-diagnostics`, `oqi-format`, `oqi-compile`,
`oqi-vm`, `oqi`, `oqi-cli`; `oqi-js` (npm) and `oqi-py` (PyPI) are
`publish = false`. Every published library has `#![warn(missing_docs)]`, a
README and symlinked license files; integration tests that read the repo's
`fixtures/` are excluded from the packages. `rust-version` is the highest
`rust-version` among the locked dependencies (all features); clippy's
`incompatible_msrv` checks std usage against it. CI (`.github/workflows/ci.yml`)
runs fmt, clippy, tests, wasm and Python tests, docs, an MSRV check and
`cargo package --workspace`.

## Reference material

`docs/*.rst` is the OpenQASM 3.0 language spec (the authoritative description of
semantics this compiler targets). `lexer.g4` / `parser.g4` are the reference
ANTLR grammars.

**Always check the spec before changing language-facing behaviour** (lexing,
parsing, types, semantics, lowering, the standard library, pulses/OpenPulse) and
make sure your change stays consistent with it. `docs/SUMMARY.md` is a
one-line-per-file table of contents for finding the relevant section fast; read
the underlying `.rst` for the authoritative detail.

## Version control

This repo uses **`jujutsu` (`jj`)**, not raw git. Do not squash, rebase, or push
unless explicitly asked.
