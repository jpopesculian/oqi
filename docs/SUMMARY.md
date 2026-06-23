# OpenQASM 3.0 spec — summary & table of contents

A scannable index of the language spec in this directory. The `.rst` files are the
**authoritative** description of the semantics `oqi` targets — this file only helps
you find the right section fast. Always read the underlying `.rst` for detail, and
keep language-facing changes consistent with it.

Files are listed in the order of the `index.rst` toctree.

## [comments.rst](comments.rst) — Comments
Line (`//`) and block (`/* */`) comments; the leading `OPENQASM 3.0;` version
string; `include` of other files.
**Key topics:** Comments · Version string · Included files

## [types.rst](types.rst) — Types and casting
The classical and quantum type system: the foundational reference for the type
checker.
**Key topics:** Identifiers · Variables · `qubit` / physical qubits (`$0`) ·
classical scalars (`bit`/`int`/`uint`/`float`/`angle`/`bool`/`complex`) · `const`
and compile-time constants · built-in constants & functions · literals · `array`
· timing types (`duration`/`stretch`) · aliasing (`let`) · index sets & slicing ·
register/array concatenation · casting rules (allowed casts table)

## [gates.rst](gates.rst) — Gates
Defining and applying unitary gates and gate modifiers.
**Key topics:** applying gates · broadcasting over registers · parameterized
gates · `gate` definitions · hierarchical definitions · modifiers
(`ctrl` / `negctrl` / `inv` / `pow`) · built-in gates (`U`, `gphase`) · relation
to hardware-native gates

## [insts.rst](insts.rst) — Built-in quantum instructions
Non-unitary built-in operations on qubits.
**Key topics:** initialization / `reset` · `measure` · explicit no-op (`barrier`)

## [classical.rst](classical.rst) — Classical instructions
Classical computation, control flow, and program structure.
**Key topics:** classical bits & registers · boolean/comparison ops · integer,
angle, float, complex arithmetic · evaluation order · `if`/`else` · `for` ·
`while` · `break`/`continue` · `end` (early termination) · `switch` ·
`extern` function calls

## [subroutines.rst](subroutines.rst) — Subroutines
**Key topics:** `def` declarations · parameters & return values · passing arrays
into subroutines

## [scope.rst](scope.rst) — Scoping of variables
**Key topics:** global scope · subroutine and gate scope · block scope

## [directives.rst](directives.rst) — Directives
**Key topics:** pragma & annotation namespacing · `pragma` · annotations
(`@...`) · `input`/`output` declarations

## [standard_library.rst](standard_library.rst) — Standard library
The `stdgates.inc` gate set the standard library provides.
**Key topics:** versioning · notes for implementations · gate API documentation

## [delays.rst](delays.rst) — Circuit timing
Durations, timing, and timing-aware instructions.
**Key topics:** `duration`/`stretch` types · operations on durations · `delay`
and duration-based instructions · `box` (boxed expressions) · `barrier`

## [pulses.rst](pulses.rst) — Pulse-level descriptions of gates
Calibration of gates and measurements at the pulse level.
**Key topics:** `defcal` · inline `cal` calibration blocks · restrictions on
`defcal` bodies · calibrations in practice

## [openpulse.rst](openpulse.rst) — OpenPulse grammar
The OpenPulse extension (in active development) for pulse-level control.
**Key topics:** ports · frames (initialization & manipulation) · waveforms ·
`play` · `capture` · timing (initial time, delay, barrier) · phase tracking ·
collisions · worked examples (Rabi, cross-resonance, geometric gate, neutral
atoms, multiplexed readout)
