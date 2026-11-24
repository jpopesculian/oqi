# oqi-vm

The bytecode VM and quantum simulators of [oqi](https://crates.io/crates/oqi),
an [OpenQASM 3](https://openqasm.com) compiler and simulator. Backends:

- `StateVectorSim` and `SimdSim`: dense state vectors, optionally
  multi-threaded;
- `AutoSim`: a stabilizer tableau that moves to an exact sum over Cliffords,
  and to a state vector only when needed;
- `MpsSim` (feature `mps`, default): matrix product states;
- `GpuSim` (feature `gpu`): WebGPU.

`QuantumBackend`, `ExternProvider` and `OpenPulseHandler` let a host plug in
its own backend, `extern` functions and pulse execution.

Most users want the [`oqi`](https://crates.io/crates/oqi) crate, which
re-exports the supported API of this one and adds blocking helpers.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
