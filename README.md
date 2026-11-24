# oqi

An [OpenQASM 3](https://openqasm.com) compiler and quantum simulator, written
in Rust. oqi parses and type-checks OpenQASM 3 programs (including OpenPulse
calibrations), compiles them to a compact bytecode, and runs them on a choice
of simulators:

- dense state vectors: scalar, SIMD and multi-threaded CPU kernels, and WebGPU;
- a stabilizer tableau that moves to an exact sum over Cliffords when a
  circuit leaves the Clifford group, for wide circuits with few non-Clifford
  gates;
- a matrix-product-state simulator, for wide circuits with modest
  entanglement.

It runs from the command line, from Rust, in the browser and Node.js
(WebAssembly), and from Python. Try it in the
[playground](https://jpopesculian.github.io/oqi/).

## Command line

```sh
cargo install oqi-cli          # installs the `oqi` binary
```

```sh
oqi run bell.qasm                          # one run: measurements and outputs
oqi run bell.qasm --shots 1000 --seed 7    # a histogram per output variable
oqi run prog.qasm --input n=3 --input theta=0.5 --backend auto
oqi run prog.qasm --state                  # print the final state vector
oqi compile prog.qasm -o prog.oqib         # check and save bytecode
oqi run prog.oqib                          # run saved bytecode
oqi compile prog.qasm --dump               # print the bytecode
oqi fmt prog.qasm                          # format in place
```

`oqi run --help` lists the backends (`scalar`, `rayon`, `simd`, `rayon-simd`,
`gpu`, `prefer-gpu`, `auto`, `sum`, `mps`) and the timing options. The WebGPU
backend needs `cargo install oqi-cli --features gpu`.

## Rust

```toml
[dependencies]
oqi = "0.1"
```

```rust
fn main() -> oqi::Result<()> {
    let program = oqi::compile(
        r#"
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        bit[2] c = measure q;
        "#,
    )?;
    let samples = oqi::Runner::new(&program).seed(7).sample_blocking(1000)?;
    for (bits, count) in samples.histogram("c").unwrap().counts() {
        println!("{bits}: {count}");
    }
    Ok(())
}
```

See the [`oqi` crate documentation](https://docs.rs/oqi) for inputs, externs,
backends, error rendering, formatting and bytecode.

## JavaScript and Python

The [`js`](js/) directory builds an npm package with
[wasm-pack](https://wasm-bindgen.github.io/wasm-pack/), and the [`py`](py/)
directory a Python package with [maturin](https://www.maturin.rs). Both offer
`compile`, `run` and `sample` with the same options and backend names as the
command line; see their READMEs.

```python
import oqi

samples = oqi.sample(open("bell.qasm").read(), shots=1000, seed=7)
print(samples.histograms)      # {'c': {'00': 497, '11': 503}}
```

## Crates

| Crate | |
|---|---|
| [`oqi`](oqi/) | The supported Rust API: compile, run, sample, format, render errors. |
| [`oqi-cli`](cli/) | The `oqi` command-line tool. |
| [`oqi-compile`](compile/) | The compiler: source to bytecode. |
| [`oqi-vm`](vm/) | The bytecode VM and the simulators. |
| [`oqi-classical`](classical/) | Classical values, types and operators. |
| [`oqi-parse`](parse/) | The parser and syntax tree. |
| [`oqi-lex`](lex/) | The lexer. |
| [`oqi-format`](format/) | The source formatter. |
| [`oqi-diagnostics`](diagnostics/) | Error codes and diagnostic rendering. |

Everything but `oqi` and `oqi-cli` is a building block; their APIs may change
more freely.

## Language support

oqi follows the OpenQASM 3.0 specification (kept in [`docs/`](docs/)) and the
reference ANTLR grammar ([`lexer.g4`](lexer.g4), [`parser.g4`](parser.g4)):
gates and gate modifiers (`ctrl`, `negctrl`, `inv`, `pow`), the standard
library, subroutines and `extern` functions, `input`/`output`, control flow
(`if`, `for`, `while`, `switch`, `break`, `continue`, `end`), every classical
type (`bit`, `int`, `uint`, `float`, `angle`, `bool`, `complex`, `duration`,
`stretch`) and arrays, aliases, index sets and slices, timing (`delay`, `box`,
`durationof`, `stretch`), OpenPulse calibrations (`cal`, `defcal`), pragmas and
annotations, and the OpenQASM 2 style `qreg`, `creg` and `measure a -> b`.

Known limits:

- Integers are at most 128 bits wide; arrays have at most 7 dimensions and
  2²⁶ elements.
- Array concatenation with `++` is only accepted in `let` aliases: the
  reference grammar allows it nowhere else, although the specification's prose
  shows it in expressions.
- A classical `let` alias is usable only in the body that declares it.
- OpenPulse support covers `newframe`, `gaussian`, `play`, `capture`,
  `shift_phase` and `threshold`, through a host-supplied handler. Calls with
  gate modifiers, and `delay` and expression-specialised defcals, never
  dispatch to a calibration.
- A runtime error raised in included code, or while running saved bytecode,
  is reported without its source text.
- The WebGPU simulator is single precision and holds as many qubits as the
  device's memory allows, at most 32. The
  matrix-product-state simulator is approximate once a circuit's entanglement
  exceeds its bond cap.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. The OpenQASM standard library, grammar, specification and
example programs come from the [OpenQASM project](https://github.com/openqasm/openqasm)
under the Apache License, Version 2.0; see [NOTICE](NOTICE).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
