# oqi-js

[oqi](https://github.com/jpopesculian/oqi) for JavaScript: compile, run and
sample [OpenQASM 3](https://openqasm.com) programs in the browser or Node.js,
compiled to WebAssembly. Try it in the
[playground](https://jpopesculian.github.io/oqi/).

## Building

The package is built from this directory's Rust crate with
[wasm-pack](https://wasm-bindgen.github.io/wasm-pack/):

```sh
# from the repository root
wasm-pack build js --target web               # for the browser (ES module)
wasm-pack build js --target nodejs            # for Node.js
wasm-pack build js --target web --features gpu  # with the WebGPU simulator
```

The package lands in `js/pkg`, with TypeScript declarations.

## Usage

```js
import init, { compile, run, sample } from "oqi-js";

await init(); // the `web` target loads the WebAssembly module first

const bell = `
  include "stdgates.inc";
  qubit[2] q;
  h q[0];
  cx q[0], q[1];
  bit[2] c = measure q;
`;

// One run: measurements, output values, optionally the state vector.
const result = await run(bell, { seed: 7, statevector: true });
console.log(result.outputs); // [{ name: "c", value: "00" }] or "11"

// Many shots: one histogram per output variable.
const samples = await sample(bell, { shots: 1000, backend: "auto" });
console.log(samples.histograms[0].bars); // [{ label: "00", count: ... }, ...]

// Inputs and host implementations of `extern` functions.
const out = await run(
  `input int n; extern twice(int) -> int; output int m; m = twice(n);`,
  { inputs: { n: 21 }, externs: { twice: (x) => 2 * x } },
);

// Compile once, run the bytecode many times.
const { bytecode, disassembly } = compile(bell);
await run(bytecode);
```

Program errors throw an `Error` whose message is the rendered diagnostic, with
the diagnostic's `code` (e.g. `"S0001"`) and, when it points into the source,
its `span` (`{ start, end }` byte offsets).

Backends, by the names of the `oqi` CLI's `--backend`: `scalar` (default),
`simd`, `gpu` and `prefer-gpu` (with the `gpu` feature and WebGPU), `auto`,
`sum` and `mps`. Includes are limited to the embedded `stdgates.inc`.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
