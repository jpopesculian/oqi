use logos::Logos;

/// A token of the OpenQASM 3 lexer grammar.
///
/// The `Logos` derive drives the default lexer mode. Variants without a
/// pattern are produced by [`Lexer`](crate::Lexer) itself: the mode-specific
/// tokens by its hand-written modes, and `Waveform` / `Port` / `Frame` by
/// promoting identifiers when lexing OpenPulse.
#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(skip r"[ \t]+")]
#[logos(skip r"[\r\n]+")]
#[logos(subpattern dec = r"[0-9](_?[0-9])*")]
#[logos(subpattern exp = r"[eE][+-]?(?&dec)")]
#[logos(subpattern float = r"(?&dec)(?&exp)|\.(?&dec)(?&exp)?|(?&dec)\.(?&dec)?(?&exp)?")]
#[logos(subpattern ident = r"[\p{L}\p{Nl}_][\p{L}\p{Nl}0-9_]*")]
pub enum Token<'a> {
    // Keywords
    /// The `OPENQASM` keyword.
    #[token("OPENQASM")]
    OpenQasm,
    /// The `include` keyword.
    #[token("include")]
    Include,
    /// The `defcalgrammar` keyword.
    #[token("defcalgrammar")]
    DefCalGrammar,
    /// The `def` keyword.
    #[token("def")]
    Def,
    /// The `cal` keyword.
    #[token("cal")]
    Cal,
    /// The `defcal` keyword.
    #[token("defcal")]
    DefCal,
    /// The `gate` keyword.
    #[token("gate")]
    Gate,
    /// The `extern` keyword.
    #[token("extern")]
    Extern,
    /// The `box` keyword.
    #[token("box")]
    Box,
    /// The `let` keyword.
    #[token("let")]
    Let,
    /// The `break` keyword.
    #[token("break")]
    Break,
    /// The `continue` keyword.
    #[token("continue")]
    Continue,
    /// The `if` keyword.
    #[token("if")]
    If,
    /// The `else` keyword.
    #[token("else")]
    Else,
    /// The `end` keyword.
    #[token("end")]
    End,
    /// The `return` keyword.
    #[token("return")]
    Return,
    /// The `for` keyword.
    #[token("for")]
    For,
    /// The `while` keyword.
    #[token("while")]
    While,
    /// The `in` keyword.
    #[token("in")]
    In,
    /// The `switch` keyword.
    #[token("switch")]
    Switch,
    /// The `case` keyword.
    #[token("case")]
    Case,
    /// The `default` keyword.
    #[token("default")]
    Default,
    /// The `nop` keyword.
    #[token("nop")]
    Nop,
    /// The `pragma` keyword (also `#pragma`); the rest of the line follows as [`Token::RemainingLineContent`].
    #[regex(r"#?pragma", priority = 3)]
    Pragma,
    /// An annotation keyword, including its `@`: `@bind`, `@my.namespace`. The rest of the line follows as [`Token::RemainingLineContent`].
    #[regex(r"@(?&ident)(\.(?&ident))*", |lex| lex.slice())]
    AnnotationKeyword(&'a str),

    // Type keywords
    /// The `input` keyword.
    #[token("input")]
    Input,
    /// The `output` keyword.
    #[token("output")]
    Output,
    /// The `const` keyword.
    #[token("const")]
    Const,
    /// The `readonly` keyword.
    #[token("readonly")]
    Readonly,
    /// The `mutable` keyword.
    #[token("mutable")]
    Mutable,
    /// The `qreg` keyword.
    #[token("qreg")]
    Qreg,
    /// The `qubit` keyword.
    #[token("qubit")]
    Qubit,
    /// The `creg` keyword.
    #[token("creg")]
    Creg,
    /// The `bool` keyword.
    #[token("bool")]
    Bool,
    /// The `bit` keyword.
    #[token("bit")]
    Bit,
    /// The `int` keyword.
    #[token("int")]
    Int,
    /// The `uint` keyword.
    #[token("uint")]
    Uint,
    /// The `float` keyword.
    #[token("float")]
    Float,
    /// The `angle` keyword.
    #[token("angle")]
    Angle,
    /// The `complex` keyword.
    #[token("complex")]
    Complex,
    /// The `array` keyword.
    #[token("array")]
    Array,
    /// The `void` keyword.
    #[token("void")]
    Void,
    /// The `duration` keyword.
    #[token("duration")]
    Duration,
    /// The `stretch` keyword.
    #[token("stretch")]
    Stretch,

    // OpenPulse-only type keywords (identifiers outside OpenPulse)
    /// The OpenPulse `waveform` type (an identifier outside calibration code).
    Waveform,
    /// The OpenPulse `port` type (an identifier outside calibration code).
    Port,
    /// The OpenPulse `frame` type (an identifier outside calibration code).
    Frame,

    // Builtin identifiers and operations
    /// The `gphase` keyword.
    #[token("gphase")]
    Gphase,
    /// The `inv` keyword.
    #[token("inv")]
    Inv,
    /// The `pow` keyword.
    #[token("pow")]
    Pow,
    /// The `ctrl` keyword.
    #[token("ctrl")]
    Ctrl,
    /// The `negctrl` keyword.
    #[token("negctrl")]
    Negctrl,
    /// The `#dim` keyword of array reference types.
    #[token("#dim")]
    Dim,
    /// The `durationof` keyword.
    #[token("durationof")]
    Durationof,
    /// The `delay` keyword.
    #[token("delay")]
    Delay,
    /// The `reset` keyword.
    #[token("reset")]
    Reset,
    /// The `measure` keyword.
    #[token("measure")]
    Measure,
    /// The `barrier` keyword.
    #[token("barrier")]
    Barrier,
    /// The `im` suffix (as a separate token, after whitespace).
    #[token("im")]
    Imag,

    // Literals
    /// The `true` keyword.
    #[token("true")]
    True,
    /// The `false` keyword.
    #[token("false")]
    False,
    /// A binary integer literal, as written: `0b1010`, `0B1_0`.
    #[regex(r"0[bB][01](_?[01])*", |lex| lex.slice())]
    BinaryIntegerLiteral(&'a str),
    /// An octal integer literal, as written: `0o17`.
    #[regex(r"0o[0-7](_?[0-7])*", |lex| lex.slice())]
    OctalIntegerLiteral(&'a str),
    /// A decimal integer literal, as written (with any `_` separators): `42`, `1_000`.
    #[regex(r"(?&dec)", |lex| lex.slice())]
    DecimalIntegerLiteral(&'a str),
    /// A hexadecimal integer literal, as written: `0xFF`.
    #[regex(r"0[xX][0-9a-fA-F](_?[0-9a-fA-F])*", |lex| lex.slice())]
    HexIntegerLiteral(&'a str),
    /// A floating-point literal, as written: `1.5`, `.5e-3`, `2e10`.
    #[regex(r"(?&float)", |lex| lex.slice())]
    FloatLiteral(&'a str),
    /// An imaginary literal, as written, including the `im` suffix: `1.5im`, `2 im`.
    #[regex(r"((?&float)|(?&dec))[ \t]*im", |lex| lex.slice())]
    ImaginaryLiteral(&'a str),
    // `µs` is spelled with U+00B5 MICRO SIGN in lexer.g4 and with U+03BC
    // GREEK SMALL LETTER MU in docs/types.rst; both are accepted.
    /// A duration literal, as written, including its unit (`dt`, `ns`, `us`, `µs`, `ms` or `s`): `100ns`, `4dt`.
    #[regex(r"((?&float)|(?&dec))[ \t]*(dt|ns|us|µs|μs|ms|s)", |lex| lex.slice())]
    TimingLiteral(&'a str),
    /// A bitstring literal, as written, including the quotes: `"0101"`.
    #[regex(r#""[01](_?[01])*""#, |lex| lex.slice())]
    BitstringLiteral(&'a str),
    /// An identifier.
    #[regex(r"(?&ident)", |lex| lex.slice())]
    Identifier(&'a str),
    /// A physical qubit, as written, including the `$`: `$0`.
    #[regex(r"\$[0-9]+", |lex| lex.slice())]
    HardwareQubit(&'a str),

    // Symbols
    /// `[`
    #[token("[")]
    LBracket,
    /// `]`
    #[token("]")]
    RBracket,
    /// `{`
    #[token("{")]
    LBrace,
    /// `}`
    #[token("}")]
    RBrace,
    /// `(`
    #[token("(")]
    LParen,
    /// `)`
    #[token(")")]
    RParen,
    /// `:`
    #[token(":")]
    Colon,
    /// `;`
    #[token(";")]
    Semicolon,
    /// `.`
    #[token(".")]
    Dot,
    /// `,`
    #[token(",")]
    Comma,
    /// `=`
    #[token("=")]
    Equals,
    /// `->`
    #[token("->")]
    Arrow,
    /// `+`
    #[token("+")]
    Plus,
    /// `++`
    #[token("++")]
    DoublePlus,
    /// `-`
    #[token("-")]
    Minus,
    /// `*`
    #[token("*")]
    Asterisk,
    /// `**`
    #[token("**")]
    DoubleAsterisk,
    /// `/`
    #[token("/")]
    Slash,
    /// `%`
    #[token("%")]
    Percent,
    /// `|`
    #[token("|")]
    Pipe,
    /// `||`
    #[token("||")]
    DoublePipe,
    /// `&`
    #[token("&")]
    Ampersand,
    /// `&&`
    #[token("&&")]
    DoubleAmpersand,
    /// `^`
    #[token("^")]
    Caret,
    /// `@`
    #[token("@")]
    At,
    /// `~`
    #[token("~")]
    Tilde,
    /// `!`
    #[token("!")]
    ExclamationPoint,

    // Multi-value operators
    /// `==`
    #[token("==")]
    DoubleEquals,
    /// `!=`
    #[token("!=")]
    ExclamationEquals,
    /// `+=`
    #[token("+=")]
    PlusEquals,
    /// `-=`
    #[token("-=")]
    MinusEquals,
    /// `*=`
    #[token("*=")]
    AsteriskEquals,
    /// `/=`
    #[token("/=")]
    SlashEquals,
    /// `&=`
    #[token("&=")]
    AmpersandEquals,
    /// `|=`
    #[token("|=")]
    PipeEquals,
    /// `^=`
    #[token("^=")]
    CaretEquals,
    /// `<<=`
    #[token("<<=")]
    LeftShiftEquals,
    /// `>>=`
    #[token(">>=")]
    RightShiftEquals,
    /// `%=`
    #[token("%=")]
    PercentEquals,
    /// `**=`
    #[token("**=")]
    DoubleAsteriskEquals,
    /// `>`
    #[token(">")]
    GreaterThan,
    /// `<`
    #[token("<")]
    LessThan,
    /// `>=`
    #[token(">=")]
    GreaterThanEquals,
    /// `<=`
    #[token("<=")]
    LessThanEquals,
    /// `>>`
    #[token(">>")]
    DoubleGreater,
    /// `<<`
    #[token("<<")]
    DoubleLess,

    // Mode-specific tokens
    /// The version number after `OPENQASM`: `3`, `3.0`.
    VersionSpecifier(&'a str),
    /// A quoted string (after `include` or `defcalgrammar`), as written, including the quotes.
    StringLiteral(&'a str),
    /// The rest of the line after `pragma` or an annotation keyword, as written.
    RemainingLineContent(&'a str),
    /// The body of a `cal` or `defcal` block, as written, between its braces.
    CalibrationBlock(&'a str),

    // Comments
    /// A `// …` comment, as written.
    #[regex(r"//[^\r\n]*", |lex| lex.slice(), allow_greedy = true)]
    LineComment(&'a str),
    /// A `/* … */` comment, as written.
    #[regex(r"/\*[^*]*(\*+[^/*][^*]*)*\*+/", |lex| lex.slice())]
    BlockComment(&'a str),
}
