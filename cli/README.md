# oqi-cli

The `oqi` command-line tool: compile, run, sample and format
[OpenQASM 3](https://openqasm.com) programs with
[oqi](https://crates.io/crates/oqi).

```sh
cargo install oqi-cli                 # add --features gpu for the WebGPU backend
oqi run bell.qasm --shots 1000 --seed 7
oqi run prog.qasm --input n=3 --backend auto
oqi compile prog.qasm -o prog.oqib && oqi run prog.oqib
oqi fmt prog.qasm
```

`oqi help <command>` describes every option.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
