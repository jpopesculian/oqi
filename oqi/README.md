# oqi

An [OpenQASM 3](https://openqasm.com) compiler and quantum simulator in Rust.
`oqi` parses, type-checks and compiles OpenQASM 3 programs (including
OpenPulse calibrations) to bytecode, and runs them on a choice of
simulators: dense state vectors (scalar, SIMD, multi-threaded, WebGPU), a
stabilizer and sum-over-Cliffords router, and a matrix-product-state
simulator.

This crate is the supported entry point: it brings the compiler, the
simulators, error rendering and the formatter together behind one
dependency.

```toml
[dependencies]
oqi = "0.1"
```

## Run a program

```rust
fn main() -> oqi::Result<()> {
    let program = oqi::compile(
        r#"
        include "stdgates.inc";
        input int n;
        qubit q;
        if (n > 2) { x q; }
        bit c = measure q;
        "#,
    )?;
    let result = oqi::Runner::new(&program).input("n", 3).run_blocking()?;
    assert_eq!(result.output("c").unwrap().to_string(), "1");
    Ok(())
}
```

## Sample many shots

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
    let samples = oqi::Runner::new(&program)
        .backend("auto".parse()?)
        .seed(7)
        .sample_blocking(1000)?;
    for (bits, count) in samples.histogram("c").unwrap().counts() {
        println!("{bits}: {count}");
    }
    Ok(())
}
```

## Also

- The `oqi` command-line tool (crate `oqi-cli`) compiles, runs, samples and
  formats programs: `cargo install oqi-cli`.
- A browser playground runs programs client-side:
  <https://jpopesculian.github.io/oqi/>.
- Python and JavaScript bindings live in the repository:
  <https://github.com/jpopesculian/oqi>.

## Features

- `blocking` (default): `Runner::run_blocking`, `Runner::sample_blocking` and
  `oqi::block_on`, for callers without an async runtime.
- `parallel` (default): multi-threaded state-vector kernels.
- `mps` (default): the matrix-product-state simulator.
- `gpu`: the WebGPU simulator.
