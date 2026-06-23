# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`oqi` is an OpenQASM 3.0 compiler and CPU simulator, written as a Rust workspace
(edition 2024). It lexes, parses, type-checks, and lowers OpenQASM (including the
OpenPulse grammar) to a bytecode IR, then interprets that bytecode against a
state-vector simulator. The single binary is `oqi` (built from `cli/`).

## Commands

```bash
cargo build                       # build the whole workspace
cargo test                        # run all tests
cargo test -p oqi-compile         # test a single crate
cargo test -p oqi-compile teleport  # run a single test by name
cargo fmt
cargo clippy --all-targets

# Run the CLI
cargo run -p oqi-cli -- compile fixtures/qasm/teleport.qasm --dump
cargo run -p oqi-cli -- run fixtures/qasm/adder.qasm --state
cargo run -p oqi-cli -- run prog.qasm --input n=3 --input theta=0.5
cargo run -p oqi-cli -- fmt --stdout fixtures/qasm/qft.qasm
```

## Compilation pipeline

The end-to-end flow (see `compile/src/bytecode/mod.rs` and `cli/src/main.rs`):

```
source → parse → resolve (includes) → lower → cfg → ssa → bytecode → vm
```

- **`lex/`** (`oqi-lex`) — `logos`-based lexer producing `Token`s with spans.
- **`parse/`** (`oqi-parse`) — hand-written parser; AST in `parse/src/ast.rs`.
- **`compile/`** (`oqi-compile`) — the heart. `lower::compile_source` is the main
  entry point (source → `sir::Program`). Key modules:
  - `lower.rs` — AST → SIR, name resolution, type checking.
  - `resolve.rs` — `include` resolution; `stdgates.inc` is embedded via
    `include_str!` and served as a library include.
  - `sir.rs` — Structured IR (the `Program` with gates, subroutines, externs,
    calibrations, body).
  - `cfg.rs` → `ssa.rs` → `bytecode/` — control-flow graph, SSA construction,
    then the flat phi-free bytecode (`bytecode::emit`). Bytecode is
    postcard-encoded binary with a textual disassembler (`bytecode/disasm.rs`).
  - `types.rs`, `symbol.rs`, `scope.rs` — type system, symbol table, scoping.
- **`classical/`** (`oqi-classical`) — classical value/type system (ints of
  arbitrary width via `awint`, floats, complex, arrays, bit registers) and all
  scalar operations under `classical/src/ops/`. Used by both compile and vm.
- **`openpulse/`** (`oqi-openpulse`) — OpenPulse types (waveforms, frames, ports).
- **`quantum/`** (`oqi-quantum`) — quantum memory model and complex-amplitude math.
- **`vm/`** (`oqi-vm`) — interprets bytecode. Two pluggable extension points:
  `QuantumBackend` (the `StateVectorSim`, or future GPU/hardware) and
  `ExternProvider` (host implementations of `extern` functions). Backends supply
  their own `stdgates.inc` decomposing the standard library down to the built-in
  `U`/`gphase`/measure/reset primitives.
- **`format/`** (`oqi-format`) — source formatter (the `fmt` subcommand).
- **`diagnostics/`** (`oqi-diagnostics`) — `ariadne`-based rendering. Errors
  implement the `Diagnostic` trait and carry stable `C####` codes.

## Diagnostics & error codes

`CompileError` (`compile/src/error.rs`) has an `ErrorKind` enum; each variant maps
to a stable numeric code in `compile/src/diagnostic.rs` (`C0001` = undefined name,
etc.). When adding an `ErrorKind`, add its code, primary label, and any help text
in `diagnostic.rs`. `oqi_diagnostics::emit` renders an error against source.
`oqi_diagnostics::render_to_string` is the testable variant.

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
- Many ops in `classical/src/ops/` carry their own unit tests inline.

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
