# oqi for Python

Compile, run and sample [OpenQASM 3](https://openqasm.com) programs from Python
with [oqi](https://github.com/jpopesculian/oqi): a compiler and a set of
simulators (state vector, stabilizer and sum-over-Cliffords, matrix product
state) written in Rust.

```python
import oqi

bell = """
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
bit[2] c = measure q;
"""

# One run: measurements, output values, optionally the state vector.
result = oqi.run(bell, seed=7, statevector=True)
print(result.outputs)          # {'c': '"00"'} or {'c': '"11"'} (bit[n] as OpenQASM text)

# Many shots: how often each output value was observed.
samples = oqi.sample(bell, shots=1000, backend="auto")
print(samples.histograms)      # {'c': {'00': 497, '11': 503}}

# Inputs and Python implementations of `extern` functions.
out = oqi.run(
    "input int n; extern twice(int) -> int; output int m; m = twice(n);",
    inputs={"n": 21},
    externs={"twice": lambda x: 2 * x},
)
print(out.outputs["m"])        # 42

# Compile once, run the bytecode many times.
compiled = oqi.compile(bell)
oqi.run(compiled.bytecode)
```

Errors raise `oqi.OqiError`: its message is the rendered diagnostic, `code` the
diagnostic's stable code (e.g. `"S0001"`) and `span` the `(start, end)`
character offsets of the offending source text.

Backends, by the names of the `oqi` CLI's `--backend`: `scalar` (default),
`rayon`, `simd`, `rayon-simd`, `auto`, `sum` and `mps`. Includes are limited to
the embedded `stdgates.inc`. The package ships type stubs.

## Building

```sh
pip install maturin
maturin develop    # from this directory, into the active virtualenv
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
