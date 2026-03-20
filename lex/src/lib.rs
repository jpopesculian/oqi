mod error;
mod token;

use logos::Logos;

pub use error::{Error, Result};
pub use token::{Span, Token};

// ---------------------------------------------------------------------------
// Internal logos-derived enum for default mode tokenization
// ---------------------------------------------------------------------------

// Shared regex fragments (for documentation; repeated inline in attributes):
//   DEC  = [0-9](_?[0-9])*
//   EXP  = [eE][+-]?[0-9](_?[0-9])*
//   FLOAT = DEC EXP | \.DEC EXP? | DEC\.DEC? EXP?
//   NUM   = FLOAT | DEC

#[derive(Logos, Debug, PartialEq)]
#[logos(skip r"[ \t]+")]
#[logos(skip r"[\r\n]+")]
enum RawToken {
    // ---- Keywords ----
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
    Box_,
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
    Default_,
    #[token("nop")]
    Nop,

    #[regex(r"#?pragma", priority = 3)]
    Pragma,

    #[regex(r"@[\p{L}\p{Nl}_][\p{L}\p{Nl}0-9_]*(\.[\p{L}\p{Nl}_][\p{L}\p{Nl}0-9_]*)*")]
    AnnotationKeyword,

    // ---- Type keywords ----
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

    // ---- Builtin identifiers ----
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

    #[token("true")]
    True,
    #[token("false")]
    False,

    // ---- Symbols ----
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

    // ---- Multi-value operators ----
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

    #[token("im")]
    Imag,

    // ---- Composite literals (number + suffix) ----
    // ImaginaryLiteral: (DEC|FLOAT) [ \t]* 'im'
    #[regex(
        r"([0-9](_?[0-9])*[eE][+-]?[0-9](_?[0-9])*|\.[0-9](_?[0-9])*([eE][+-]?[0-9](_?[0-9])*)?|[0-9](_?[0-9])*\.([0-9](_?[0-9])*)?([eE][+-]?[0-9](_?[0-9])*)?|[0-9](_?[0-9])*)[ \t]*im"
    )]
    ImaginaryLiteral,

    // TimingLiteral: (DEC|FLOAT) [ \t]* TimeUnit
    #[regex(
        r"([0-9](_?[0-9])*[eE][+-]?[0-9](_?[0-9])*|\.[0-9](_?[0-9])*([eE][+-]?[0-9](_?[0-9])*)?|[0-9](_?[0-9])*\.([0-9](_?[0-9])*)?([eE][+-]?[0-9](_?[0-9])*)?|[0-9](_?[0-9])*)[ \t]*(dt|ns|us|µs|ms|s)"
    )]
    TimingLiteral,

    // ---- Number literals ----
    #[regex(r"0[bB][01](_?[01])*")]
    BinaryIntegerLiteral,
    #[regex(r"0o[0-7](_?[0-7])*")]
    OctalIntegerLiteral,
    #[regex(r"0[xX][0-9a-fA-F](_?[0-9a-fA-F])*")]
    HexIntegerLiteral,

    // FloatLiteral (three alternatives, longest match beats DecimalIntegerLiteral)
    #[regex(
        r"[0-9](_?[0-9])*[eE][+-]?[0-9](_?[0-9])*|\.[0-9](_?[0-9])*([eE][+-]?[0-9](_?[0-9])*)?|[0-9](_?[0-9])*\.([0-9](_?[0-9])*)?([eE][+-]?[0-9](_?[0-9])*)?"
    )]
    FloatLiteral,

    #[regex(r"[0-9](_?[0-9])*")]
    DecimalIntegerLiteral,

    #[regex(r#""[01](_?[01])*""#)]
    BitstringLiteral,

    // ---- Identifiers ----
    #[regex(r"[\p{L}\p{Nl}_][\p{L}\p{Nl}0-9_]*")]
    Identifier,

    #[regex(r"\$[0-9]+")]
    HardwareQubit,

    // ---- Comments ----
    #[regex(r"//[^\r\n]*", allow_greedy = true)]
    LineComment,
    #[regex(r"/\*[^*]*(\*+[^/*][^*]*)*\*+/")]
    BlockComment,
}

// ---------------------------------------------------------------------------
// Mode management
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Default,
    VersionIdentifier,
    ArbitraryString,
    EatToLineEnd,
    CalPrelude,
    DefcalPrelude,
    CalBlock,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    mode_stack: Vec<Mode>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            mode_stack: vec![Mode::Default],
        }
    }

    fn current_mode(&self) -> Mode {
        self.mode_stack.last().copied().unwrap_or(Mode::Default)
    }

    fn push_mode(&mut self, mode: Mode) {
        self.mode_stack.push(mode);
    }

    fn pop_mode(&mut self) {
        self.mode_stack.pop();
    }

    fn set_mode(&mut self, mode: Mode) {
        if let Some(last) = self.mode_stack.last_mut() {
            *last = mode;
        }
    }

    // ------------------------------------------------------------------
    // Default / DefcalPrelude mode — delegates to logos
    // ------------------------------------------------------------------

    fn lex_default(&mut self) -> Option<Result<(Token<'a>, Span)>> {
        let remaining = &self.source[self.pos..];
        if remaining.is_empty() {
            return None;
        }

        let mut raw = RawToken::lexer(remaining);
        let result = raw.next()?;
        let span = raw.span();
        let slice = &remaining[span.clone()];
        let abs = (self.pos + span.start)..(self.pos + span.end);
        self.pos += span.end;

        match result {
            Ok(raw_token) => {
                let token = self.map_raw(raw_token, slice);
                Some(Ok((token, abs)))
            }
            Err(()) => Some(Err(Error {
                span: abs,
                message: "unexpected character".into(),
            })),
        }
    }

    fn map_raw(&mut self, raw: RawToken, slice: &'a str) -> Token<'a> {
        let in_defcal = self.current_mode() == Mode::DefcalPrelude;

        match raw {
            // Mode-switching keywords
            RawToken::OpenQasm => {
                self.push_mode(Mode::VersionIdentifier);
                Token::OpenQasm
            }
            RawToken::Include => {
                self.push_mode(Mode::ArbitraryString);
                Token::Include
            }
            RawToken::DefCalGrammar => {
                self.push_mode(Mode::ArbitraryString);
                Token::DefCalGrammar
            }
            RawToken::Cal => {
                self.set_mode(Mode::CalPrelude);
                Token::Cal
            }
            RawToken::DefCal => {
                self.set_mode(Mode::DefcalPrelude);
                Token::DefCal
            }
            RawToken::Pragma => {
                self.push_mode(Mode::EatToLineEnd);
                Token::Pragma
            }
            RawToken::AnnotationKeyword => {
                self.push_mode(Mode::EatToLineEnd);
                Token::AnnotationKeyword(slice)
            }
            RawToken::LBrace => {
                if in_defcal {
                    self.set_mode(Mode::CalBlock);
                }
                Token::LBrace
            }

            // Simple keywords
            RawToken::Def => Token::Def,
            RawToken::Gate => Token::Gate,
            RawToken::Extern => Token::Extern,
            RawToken::Box_ => Token::Box,
            RawToken::Let => Token::Let,
            RawToken::Break => Token::Break,
            RawToken::Continue => Token::Continue,
            RawToken::If => Token::If,
            RawToken::Else => Token::Else,
            RawToken::End => Token::End,
            RawToken::Return => Token::Return,
            RawToken::For => Token::For,
            RawToken::While => Token::While,
            RawToken::In => Token::In,
            RawToken::Switch => Token::Switch,
            RawToken::Case => Token::Case,
            RawToken::Default_ => Token::Default,
            RawToken::Nop => Token::Nop,

            // Types
            RawToken::Input => Token::Input,
            RawToken::Output => Token::Output,
            RawToken::Const => Token::Const,
            RawToken::Readonly => Token::Readonly,
            RawToken::Mutable => Token::Mutable,
            RawToken::Qreg => Token::Qreg,
            RawToken::Qubit => Token::Qubit,
            RawToken::Creg => Token::Creg,
            RawToken::Bool => Token::Bool,
            RawToken::Bit => Token::Bit,
            RawToken::Int => Token::Int,
            RawToken::Uint => Token::Uint,
            RawToken::Float => Token::Float,
            RawToken::Angle => Token::Angle,
            RawToken::Complex => Token::Complex,
            RawToken::Array => Token::Array,
            RawToken::Void => Token::Void,
            RawToken::Duration => Token::Duration,
            RawToken::Stretch => Token::Stretch,

            // Builtins
            RawToken::Gphase => Token::Gphase,
            RawToken::Inv => Token::Inv,
            RawToken::Pow => Token::Pow,
            RawToken::Ctrl => Token::Ctrl,
            RawToken::Negctrl => Token::Negctrl,
            RawToken::Dim => Token::Dim,
            RawToken::Durationof => Token::Durationof,
            RawToken::Delay => Token::Delay,
            RawToken::Reset => Token::Reset,
            RawToken::Measure => Token::Measure,
            RawToken::Barrier => Token::Barrier,

            // Booleans
            RawToken::True => Token::True,
            RawToken::False => Token::False,

            // Symbols
            RawToken::LBracket => Token::LBracket,
            RawToken::RBracket => Token::RBracket,
            RawToken::RBrace => Token::RBrace,
            RawToken::LParen => Token::LParen,
            RawToken::RParen => Token::RParen,
            RawToken::Colon => Token::Colon,
            RawToken::Semicolon => Token::Semicolon,
            RawToken::Dot => Token::Dot,
            RawToken::Comma => Token::Comma,
            RawToken::Equals => Token::Equals,
            RawToken::Arrow => Token::Arrow,
            RawToken::Plus => Token::Plus,
            RawToken::DoublePlus => Token::DoublePlus,
            RawToken::Minus => Token::Minus,
            RawToken::Asterisk => Token::Asterisk,
            RawToken::DoubleAsterisk => Token::DoubleAsterisk,
            RawToken::Slash => Token::Slash,
            RawToken::Percent => Token::Percent,
            RawToken::Pipe => Token::Pipe,
            RawToken::DoublePipe => Token::DoublePipe,
            RawToken::Ampersand => Token::Ampersand,
            RawToken::DoubleAmpersand => Token::DoubleAmpersand,
            RawToken::Caret => Token::Caret,
            RawToken::At => Token::At,
            RawToken::Tilde => Token::Tilde,
            RawToken::ExclamationPoint => Token::ExclamationPoint,

            // Multi-value operators
            RawToken::DoubleEquals => Token::DoubleEquals,
            RawToken::ExclamationEquals => Token::ExclamationEquals,
            RawToken::PlusEquals => Token::PlusEquals,
            RawToken::MinusEquals => Token::MinusEquals,
            RawToken::AsteriskEquals => Token::AsteriskEquals,
            RawToken::SlashEquals => Token::SlashEquals,
            RawToken::AmpersandEquals => Token::AmpersandEquals,
            RawToken::PipeEquals => Token::PipeEquals,
            RawToken::CaretEquals => Token::CaretEquals,
            RawToken::LeftShiftEquals => Token::LeftShiftEquals,
            RawToken::RightShiftEquals => Token::RightShiftEquals,
            RawToken::PercentEquals => Token::PercentEquals,
            RawToken::DoubleAsteriskEquals => Token::DoubleAsteriskEquals,
            RawToken::GreaterThan => Token::GreaterThan,
            RawToken::LessThan => Token::LessThan,
            RawToken::GreaterThanEquals => Token::GreaterThanEquals,
            RawToken::LessThanEquals => Token::LessThanEquals,
            RawToken::DoubleGreater => Token::DoubleGreater,
            RawToken::DoubleLess => Token::DoubleLess,

            // Literals
            RawToken::Imag => Token::Imag,
            RawToken::ImaginaryLiteral => Token::ImaginaryLiteral(slice),
            RawToken::TimingLiteral => Token::TimingLiteral(slice),
            RawToken::BinaryIntegerLiteral => Token::BinaryIntegerLiteral(slice),
            RawToken::OctalIntegerLiteral => Token::OctalIntegerLiteral(slice),
            RawToken::HexIntegerLiteral => Token::HexIntegerLiteral(slice),
            RawToken::FloatLiteral => Token::FloatLiteral(slice),
            RawToken::DecimalIntegerLiteral => Token::DecimalIntegerLiteral(slice),
            RawToken::BitstringLiteral => Token::BitstringLiteral(slice),
            RawToken::Identifier => Token::Identifier(slice),
            RawToken::HardwareQubit => Token::HardwareQubit(slice),

            // Comments
            RawToken::LineComment => Token::LineComment(slice),
            RawToken::BlockComment => Token::BlockComment(slice),
        }
    }

    // ------------------------------------------------------------------
    // VERSION_IDENTIFIER mode — after OPENQASM
    // ------------------------------------------------------------------

    fn lex_version_identifier(&mut self) -> Option<Result<(Token<'a>, Span)>> {
        self.skip_whitespace();
        if self.pos >= self.source.len() {
            self.pop_mode();
            return None;
        }

        let remaining = &self.source[self.pos..];
        let bytes = remaining.as_bytes();

        // Match [0-9]+ ('.' [0-9]+)?
        let mut end = 0;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == 0 {
            let span = self.pos..self.pos + 1;
            self.pos += 1;
            self.pop_mode();
            return Some(Err(Error {
                span,
                message: "expected version specifier".into(),
            }));
        }
        if end < bytes.len() && bytes[end] == b'.' {
            end += 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
        }

        let span = self.pos..self.pos + end;
        let slice = &self.source[span.clone()];
        self.pos += end;
        self.pop_mode();
        Some(Ok((Token::VersionSpecifier(slice), span)))
    }

    // ------------------------------------------------------------------
    // ARBITRARY_STRING mode — after include / defcalgrammar
    // ------------------------------------------------------------------

    fn lex_arbitrary_string(&mut self) -> Option<Result<(Token<'a>, Span)>> {
        self.skip_whitespace();
        if self.pos >= self.source.len() {
            self.pop_mode();
            return None;
        }

        let remaining = &self.source[self.pos..];
        let bytes = remaining.as_bytes();
        let quote = bytes[0];

        if quote != b'"' && quote != b'\'' {
            let span = self.pos..self.pos + 1;
            self.pos += 1;
            self.pop_mode();
            return Some(Err(Error {
                span,
                message: "expected quoted string".into(),
            }));
        }

        let mut end = 1;
        while end < bytes.len() {
            let b = bytes[end];
            if b == quote {
                end += 1; // include closing quote
                break;
            }
            if b == b'\r' || b == b'\n' || b == b'\t' {
                break; // invalid in string
            }
            end += 1;
        }

        let span = self.pos..self.pos + end;
        let slice = &self.source[span.clone()];
        self.pos += end;
        self.pop_mode();
        Some(Ok((Token::StringLiteral(slice), span)))
    }

    // ------------------------------------------------------------------
    // EAT_TO_LINE_END mode — after pragma / annotation
    // ------------------------------------------------------------------

    /// Returns `None` when no token is emitted (bare newline); the outer loop
    /// retries in the restored mode.
    fn lex_eat_to_line_end(&mut self) -> Option<Result<(Token<'a>, Span)>> {
        // Skip leading spaces/tabs only (not newlines)
        while self.pos < self.source.len() {
            match self.source.as_bytes()[self.pos] {
                b' ' | b'\t' => self.pos += 1,
                _ => break,
            }
        }

        if self.pos >= self.source.len() {
            self.pop_mode();
            return None;
        }

        // Newline → pop mode, emit nothing
        let b = self.source.as_bytes()[self.pos];
        if b == b'\r' || b == b'\n' {
            if b == b'\r'
                && self.pos + 1 < self.source.len()
                && self.source.as_bytes()[self.pos + 1] == b'\n'
            {
                self.pos += 2;
            } else {
                self.pos += 1;
            }
            self.pop_mode();
            return None; // caller loops
        }

        // Collect remaining line content
        let start = self.pos;
        while self.pos < self.source.len() {
            let c = self.source.as_bytes()[self.pos];
            if c == b'\r' || c == b'\n' {
                break;
            }
            self.pos += 1;
        }

        let span = start..self.pos;
        let slice = &self.source[span.clone()];
        // Don't pop yet — next call sees newline and pops
        Some(Ok((Token::RemainingLineContent(slice), span)))
    }

    // ------------------------------------------------------------------
    // CAL_PRELUDE mode — skip ws/comments, then '{' → CalBlock
    // ------------------------------------------------------------------

    fn lex_cal_prelude(&mut self) -> Option<Result<(Token<'a>, Span)>> {
        self.skip_whitespace();
        if self.pos >= self.source.len() {
            self.set_mode(Mode::Default);
            return None;
        }

        let remaining = &self.source[self.pos..];

        // Line comment
        if remaining.starts_with("//") {
            let end = remaining.find(['\r', '\n']).unwrap_or(remaining.len());
            let span = self.pos..self.pos + end;
            let slice = &self.source[span.clone()];
            self.pos += end;
            return Some(Ok((Token::LineComment(slice), span)));
        }

        // Block comment
        if let Some(stripped) = remaining.strip_prefix("/*") {
            if let Some(i) = stripped.find("*/") {
                let end = i + 4; // include the closing */
                let span = self.pos..self.pos + end;
                let slice = &self.source[span.clone()];
                self.pos += end;
                return Some(Ok((Token::BlockComment(slice), span)));
            } else {
                let span = self.pos..self.source.len();
                self.pos = self.source.len();
                return Some(Err(Error {
                    span,
                    message: "unterminated block comment".into(),
                }));
            }
        }

        // Opening brace
        if remaining.starts_with('{') {
            let span = self.pos..self.pos + 1;
            self.pos += 1;
            self.set_mode(Mode::CalBlock);
            return Some(Ok((Token::LBrace, span)));
        }

        // Unexpected
        let span = self.pos..self.pos + 1;
        self.pos += 1;
        Some(Err(Error {
            span,
            message: "unexpected character".into(),
        }))
    }

    // ------------------------------------------------------------------
    // CAL_BLOCK mode — balanced-brace content, then '}'
    // ------------------------------------------------------------------

    fn lex_cal_block(&mut self) -> Option<Result<(Token<'a>, Span)>> {
        if self.pos >= self.source.len() {
            self.set_mode(Mode::Default);
            return None;
        }

        // Closing brace at top level
        if self.source.as_bytes()[self.pos] == b'}' {
            let span = self.pos..self.pos + 1;
            self.pos += 1;
            self.set_mode(Mode::Default);
            return Some(Ok((Token::RBrace, span)));
        }

        // Consume content with balanced braces
        let start = self.pos;
        let bytes = self.source.as_bytes();
        let mut depth: usize = 0;
        while self.pos < bytes.len() {
            match bytes[self.pos] {
                b'{' => {
                    depth += 1;
                    self.pos += 1;
                }
                b'}' if depth > 0 => {
                    depth -= 1;
                    self.pos += 1;
                }
                b'}' => break, // top-level closing brace
                _ => self.pos += 1,
            }
        }

        if self.pos == start {
            return None;
        }

        let span = start..self.pos;
        let slice = &self.source[span.clone()];
        Some(Ok((Token::CalibrationBlock(slice), span)))
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn skip_whitespace(&mut self) {
        while self.pos < self.source.len() {
            match self.source.as_bytes()[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<(Token<'a>, Span)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.pos >= self.source.len() {
                return None;
            }
            match self.current_mode() {
                Mode::Default | Mode::DefcalPrelude => return self.lex_default(),
                Mode::VersionIdentifier => return self.lex_version_identifier(),
                Mode::ArbitraryString => return self.lex_arbitrary_string(),
                Mode::EatToLineEnd => match self.lex_eat_to_line_end() {
                    Some(r) => return Some(r),
                    None => continue,
                },
                Mode::CalPrelude => return self.lex_cal_prelude(),
                Mode::CalBlock => return self.lex_cal_block(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<Token<'_>> {
        Lexer::new(src)
            .filter_map(|r| r.ok())
            .map(|(tok, _)| tok)
            .collect()
    }

    #[test]
    fn keywords() {
        assert_eq!(lex("if else for"), vec![Token::If, Token::Else, Token::For]);
    }

    #[test]
    fn identifier() {
        assert_eq!(lex("foo_bar"), vec![Token::Identifier("foo_bar")]);
    }

    #[test]
    fn integers() {
        assert_eq!(
            lex("42 0b10 0o77 0xFF"),
            vec![
                Token::DecimalIntegerLiteral("42"),
                Token::BinaryIntegerLiteral("0b10"),
                Token::OctalIntegerLiteral("0o77"),
                Token::HexIntegerLiteral("0xFF"),
            ]
        );
    }

    #[test]
    fn float_literals() {
        assert_eq!(
            lex("1.5 .5 1e3 1.0e-2"),
            vec![
                Token::FloatLiteral("1.5"),
                Token::FloatLiteral(".5"),
                Token::FloatLiteral("1e3"),
                Token::FloatLiteral("1.0e-2"),
            ]
        );
    }

    #[test]
    fn imaginary_literal() {
        assert_eq!(lex("1.5im"), vec![Token::ImaginaryLiteral("1.5im")]);
        assert_eq!(lex("3im"), vec![Token::ImaginaryLiteral("3im")]);
        // With whitespace between number and suffix
        assert_eq!(lex("3 im"), vec![Token::ImaginaryLiteral("3 im")]);
    }

    #[test]
    fn timing_literal() {
        assert_eq!(lex("100ns"), vec![Token::TimingLiteral("100ns")]);
        assert_eq!(lex("2.5 ms"), vec![Token::TimingLiteral("2.5 ms")]);
        assert_eq!(lex("10dt"), vec![Token::TimingLiteral("10dt")]);
    }

    #[test]
    fn operators() {
        assert_eq!(
            lex("+ ++ ** **= >>= == !="),
            vec![
                Token::Plus,
                Token::DoublePlus,
                Token::DoubleAsterisk,
                Token::DoubleAsteriskEquals,
                Token::RightShiftEquals,
                Token::DoubleEquals,
                Token::ExclamationEquals,
            ]
        );
    }

    #[test]
    fn openqasm_version() {
        assert_eq!(
            lex("OPENQASM 3.0;"),
            vec![
                Token::OpenQasm,
                Token::VersionSpecifier("3.0"),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn include_string() {
        assert_eq!(
            lex(r#"include "stdgates.inc";"#),
            vec![
                Token::Include,
                Token::StringLiteral(r#""stdgates.inc""#),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn pragma_and_annotation() {
        assert_eq!(
            lex("#pragma my_option\n@my.ann some content\nif"),
            vec![
                Token::Pragma,
                Token::RemainingLineContent("my_option"),
                Token::AnnotationKeyword("@my.ann"),
                Token::RemainingLineContent("some content"),
                Token::If,
            ]
        );
    }

    #[test]
    fn comments() {
        assert_eq!(
            lex("x // line\n y"),
            vec![
                Token::Identifier("x"),
                Token::LineComment("// line"),
                Token::Identifier("y"),
            ]
        );
        assert_eq!(
            lex("a /* block */ b"),
            vec![
                Token::Identifier("a"),
                Token::BlockComment("/* block */"),
                Token::Identifier("b"),
            ]
        );
    }

    #[test]
    fn qubit_declaration() {
        assert_eq!(
            lex("qubit[4] q;"),
            vec![
                Token::Qubit,
                Token::LBracket,
                Token::DecimalIntegerLiteral("4"),
                Token::RBracket,
                Token::Identifier("q"),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn hardware_qubit() {
        assert_eq!(lex("$0"), vec![Token::HardwareQubit("$0")]);
    }

    #[test]
    fn bitstring_literal() {
        assert_eq!(lex(r#""1010""#), vec![Token::BitstringLiteral(r#""1010""#)]);
    }

    #[test]
    fn boolean_literal() {
        assert_eq!(
            lex("true false"),
            vec![Token::True, Token::False]
        );
    }

    #[test]
    fn cal_block() {
        assert_eq!(
            lex("cal { some opaque stuff } x"),
            vec![
                Token::Cal,
                Token::LBrace,
                Token::CalibrationBlock(" some opaque stuff "),
                Token::RBrace,
                Token::Identifier("x"),
            ]
        );
    }
}
