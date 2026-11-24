# oqi-format

A source formatter for [OpenQASM 3](https://openqasm.com) programs. It keeps
comments and blank lines between statements, ends the output with one
newline, and is idempotent. A compact mode prints the program on as few lines
as possible.

This crate is a building block of [`oqi`](https://crates.io/crates/oqi), an
OpenQASM 3 compiler and simulator, and backs the `oqi fmt` command.

```rust
use oqi_format::{format, Config};

fn main() -> Result<(), oqi_format::Error> {
    assert_eq!(format("qubit q;h  q ;", Config::default())?, "qubit q;\nh q;\n");
    Ok(())
}
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
