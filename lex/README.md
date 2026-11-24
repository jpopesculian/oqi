# oqi-lex

The lexer for [OpenQASM 3](https://openqasm.com), including the OpenPulse
calibration grammar: source text to tokens with byte-offset spans, following
the reference ANTLR grammar and its lexer modes. Comments are tokens too.

This crate is a building block of [`oqi`](https://crates.io/crates/oqi), an
OpenQASM 3 compiler and simulator. For a syntax tree, use
[`oqi-parse`](https://crates.io/crates/oqi-parse).

```rust
use oqi_lex::Lexer;

fn main() -> Result<(), oqi_lex::Error> {
    for token in Lexer::new("qubit q;") {
        let (token, span) = token?;
        println!("{token:?} at {}..{}", span.start, span.end);
    }
    Ok(())
}
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
