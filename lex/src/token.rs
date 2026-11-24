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
    #[token("OPENQASM")]
    OpenQasm,
    #[token("include")]
    Include,
    #[token("defcalgrammar")]
    DefCalGrammar,
    #[token("def")]
    Def,
    #[token("cal")]
    Cal,
    #[token("defcal")]
    DefCal,
    #[token("gate")]
    Gate,
    #[token("extern")]
    Extern,
    #[token("box")]
    Box,
    #[token("let")]
    Let,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("end")]
    End,
    #[token("return")]
    Return,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("in")]
    In,
    #[token("switch")]
    Switch,
    #[token("case")]
    Case,
    #[token("default")]
    Default,
    #[token("nop")]
    Nop,
    #[regex(r"#?pragma", priority = 3)]
    Pragma,
    #[regex(r"@(?&ident)(\.(?&ident))*", |lex| lex.slice())]
    AnnotationKeyword(&'a str),

    // Type keywords
    #[token("input")]
    Input,
    #[token("output")]
    Output,
    #[token("const")]
    Const,
    #[token("readonly")]
    Readonly,
    #[token("mutable")]
    Mutable,
    #[token("qreg")]
    Qreg,
    #[token("qubit")]
    Qubit,
    #[token("creg")]
    Creg,
    #[token("bool")]
    Bool,
    #[token("bit")]
    Bit,
    #[token("int")]
    Int,
    #[token("uint")]
    Uint,
    #[token("float")]
    Float,
    #[token("angle")]
    Angle,
    #[token("complex")]
    Complex,
    #[token("array")]
    Array,
    #[token("void")]
    Void,
    #[token("duration")]
    Duration,
    #[token("stretch")]
    Stretch,

    // OpenPulse-only type keywords (identifiers outside OpenPulse)
    Waveform,
    Port,
    Frame,

    // Builtin identifiers and operations
    #[token("gphase")]
    Gphase,
    #[token("inv")]
    Inv,
    #[token("pow")]
    Pow,
    #[token("ctrl")]
    Ctrl,
    #[token("negctrl")]
    Negctrl,
    #[token("#dim")]
    Dim,
    #[token("durationof")]
    Durationof,
    #[token("delay")]
    Delay,
    #[token("reset")]
    Reset,
    #[token("measure")]
    Measure,
    #[token("barrier")]
    Barrier,
    #[token("im")]
    Imag,

    // Literals
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[regex(r"0[bB][01](_?[01])*", |lex| lex.slice())]
    BinaryIntegerLiteral(&'a str),
    #[regex(r"0o[0-7](_?[0-7])*", |lex| lex.slice())]
    OctalIntegerLiteral(&'a str),
    #[regex(r"(?&dec)", |lex| lex.slice())]
    DecimalIntegerLiteral(&'a str),
    #[regex(r"0[xX][0-9a-fA-F](_?[0-9a-fA-F])*", |lex| lex.slice())]
    HexIntegerLiteral(&'a str),
    #[regex(r"(?&float)", |lex| lex.slice())]
    FloatLiteral(&'a str),
    #[regex(r"((?&float)|(?&dec))[ \t]*im", |lex| lex.slice())]
    ImaginaryLiteral(&'a str),
    // `µs` is spelled with U+00B5 MICRO SIGN in lexer.g4 and with U+03BC
    // GREEK SMALL LETTER MU in docs/types.rst; both are accepted.
    #[regex(r"((?&float)|(?&dec))[ \t]*(dt|ns|us|µs|μs|ms|s)", |lex| lex.slice())]
    TimingLiteral(&'a str),
    #[regex(r#""[01](_?[01])*""#, |lex| lex.slice())]
    BitstringLiteral(&'a str),
    #[regex(r"(?&ident)", |lex| lex.slice())]
    Identifier(&'a str),
    #[regex(r"\$[0-9]+", |lex| lex.slice())]
    HardwareQubit(&'a str),

    // Symbols
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token("=")]
    Equals,
    #[token("->")]
    Arrow,
    #[token("+")]
    Plus,
    #[token("++")]
    DoublePlus,
    #[token("-")]
    Minus,
    #[token("*")]
    Asterisk,
    #[token("**")]
    DoubleAsterisk,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("|")]
    Pipe,
    #[token("||")]
    DoublePipe,
    #[token("&")]
    Ampersand,
    #[token("&&")]
    DoubleAmpersand,
    #[token("^")]
    Caret,
    #[token("@")]
    At,
    #[token("~")]
    Tilde,
    #[token("!")]
    ExclamationPoint,

    // Multi-value operators
    #[token("==")]
    DoubleEquals,
    #[token("!=")]
    ExclamationEquals,
    #[token("+=")]
    PlusEquals,
    #[token("-=")]
    MinusEquals,
    #[token("*=")]
    AsteriskEquals,
    #[token("/=")]
    SlashEquals,
    #[token("&=")]
    AmpersandEquals,
    #[token("|=")]
    PipeEquals,
    #[token("^=")]
    CaretEquals,
    #[token("<<=")]
    LeftShiftEquals,
    #[token(">>=")]
    RightShiftEquals,
    #[token("%=")]
    PercentEquals,
    #[token("**=")]
    DoubleAsteriskEquals,
    #[token(">")]
    GreaterThan,
    #[token("<")]
    LessThan,
    #[token(">=")]
    GreaterThanEquals,
    #[token("<=")]
    LessThanEquals,
    #[token(">>")]
    DoubleGreater,
    #[token("<<")]
    DoubleLess,

    // Mode-specific tokens
    VersionSpecifier(&'a str),
    StringLiteral(&'a str),
    RemainingLineContent(&'a str),
    CalibrationBlock(&'a str),

    // Comments
    #[regex(r"//[^\r\n]*", |lex| lex.slice(), allow_greedy = true)]
    LineComment(&'a str),
    #[regex(r"/\*[^*]*(\*+[^/*][^*]*)*\*+/", |lex| lex.slice())]
    BlockComment(&'a str),
}
