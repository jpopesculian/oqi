mod error;
mod span;
mod token;

use logos::Logos;

pub use error::{Error, Result};
pub use span::Span;
pub use token::Token;

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

type Lexed<'a> = Option<Result<(Token<'a>, Span)>>;

#[derive(Clone)]
pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    mode_stack: Vec<Mode>,
    openpulse: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            mode_stack: vec![Mode::Default],
            openpulse: false,
        }
    }

    /// Lex an OpenPulse calibration body.
    ///
    /// `waveform`/`port`/`frame` are emitted as keyword tokens (rather than
    /// identifiers), and `cal`/`defcal` are demoted to identifiers (per the
    /// openpulse grammar's "turn off mode switching" comment) so the lexer
    /// never enters the `CalPrelude`/`DefcalPrelude`/`CalBlock` modes.
    pub fn openpulse(mut self) -> Self {
        self.openpulse = true;
        self
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

    fn rest(&self) -> &'a str {
        &self.source[self.pos..]
    }

    fn skip(&mut self, skipped: impl Fn(char) -> bool) {
        self.pos = self.source.len() - self.rest().trim_start_matches(skipped).len();
    }

    fn skip_whitespace(&mut self) {
        self.skip(|c| matches!(c, ' ' | '\t' | '\r' | '\n'));
    }

    /// Consume the next `len` bytes as one token.
    fn take(&mut self, len: usize, token: fn(&'a str) -> Token<'a>) -> (Token<'a>, Span) {
        let span = Span::new(self.pos, self.pos + len);
        self.pos = span.end;
        (token(&self.source[span]), span)
    }

    /// Consume the next character and report it as an error.
    fn reject_char(&mut self, message: &str) -> Error {
        let len = self.rest().chars().next().map_or(0, char::len_utf8);
        let span = Span::new(self.pos, self.pos + len);
        self.pos = span.end;
        Error::new(span, message)
    }

    // ------------------------------------------------------------------
    // Default / DefcalPrelude mode — delegates to logos
    // ------------------------------------------------------------------

    fn lex_default(&mut self) -> Lexed<'a> {
        let mut lexer = Token::lexer(self.rest());
        let result = lexer.next()?;
        let span = Span::from(lexer.span()).offset(self.pos);
        self.pos = span.end;
        Some(match result {
            Ok(token) => Ok((self.enter_mode(token), span)),
            Err(()) => Err(Error::new(span, "unexpected character")),
        })
    }

    /// Apply the mode switch a default-mode token triggers, and the OpenPulse
    /// keyword adjustments.
    fn enter_mode(&mut self, token: Token<'a>) -> Token<'a> {
        match token {
            Token::OpenQasm => self.push_mode(Mode::VersionIdentifier),
            Token::Include | Token::DefCalGrammar => self.push_mode(Mode::ArbitraryString),
            Token::Pragma | Token::AnnotationKeyword(_) => self.push_mode(Mode::EatToLineEnd),
            Token::LBrace if self.current_mode() == Mode::DefcalPrelude => {
                self.set_mode(Mode::CalBlock)
            }
            Token::Cal if self.openpulse => return Token::Identifier("cal"),
            Token::DefCal if self.openpulse => return Token::Identifier("defcal"),
            Token::Cal => self.set_mode(Mode::CalPrelude),
            Token::DefCal => self.set_mode(Mode::DefcalPrelude),
            Token::Identifier("waveform") if self.openpulse => return Token::Waveform,
            Token::Identifier("port") if self.openpulse => return Token::Port,
            Token::Identifier("frame") if self.openpulse => return Token::Frame,
            _ => {}
        }
        token
    }

    // ------------------------------------------------------------------
    // VERSION_IDENTIFIER mode — after OPENQASM
    // ------------------------------------------------------------------

    fn lex_version_identifier(&mut self) -> Lexed<'a> {
        self.pop_mode();
        self.skip_whitespace();
        let rest = self.rest();
        if rest.is_empty() {
            return None;
        }

        // [0-9]+ ('.' [0-9]*)?
        let digits = |s: &str| s.len() - s.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        let mut len = digits(rest);
        if len == 0 {
            return Some(Err(self.reject_char("expected version specifier")));
        }
        if rest[len..].starts_with('.') {
            len += 1 + digits(&rest[len + 1..]);
        }
        Some(Ok(self.take(len, Token::VersionSpecifier)))
    }

    // ------------------------------------------------------------------
    // ARBITRARY_STRING mode — after include / defcalgrammar
    // ------------------------------------------------------------------

    fn lex_arbitrary_string(&mut self) -> Lexed<'a> {
        self.pop_mode();
        self.skip_whitespace();
        let rest = self.rest();
        let quote = rest.chars().next()?;
        if quote != '"' && quote != '\'' {
            return Some(Err(self.reject_char("expected quoted string")));
        }

        // Up to and including the closing quote; a line break or tab ends an
        // unterminated string early.
        let len = match rest[1..].find([quote, '\r', '\n', '\t']) {
            Some(i) if rest[1 + i..].starts_with(quote) => i + 2,
            Some(i) => i + 1,
            None => rest.len(),
        };
        Some(Ok(self.take(len, Token::StringLiteral)))
    }

    // ------------------------------------------------------------------
    // EAT_TO_LINE_END mode — after pragma / annotation
    // ------------------------------------------------------------------

    /// Returns `None` when no token is emitted (the line is exhausted); the
    /// outer loop retries in the restored mode.
    fn lex_eat_to_line_end(&mut self) -> Lexed<'a> {
        self.skip(|c| matches!(c, ' ' | '\t'));
        let rest = self.rest();
        let len = rest.find(['\r', '\n']).unwrap_or(rest.len());
        if len == 0 {
            // End of the line (or of the input): consume the line break.
            self.pos += if rest.starts_with("\r\n") {
                2
            } else {
                rest.len().min(1)
            };
            self.pop_mode();
            return None;
        }
        // Don't pop yet — the next call sees the line break and pops.
        Some(Ok(self.take(len, Token::RemainingLineContent)))
    }

    // ------------------------------------------------------------------
    // CAL_PRELUDE mode — skip ws/comments, then '{' → CalBlock
    // ------------------------------------------------------------------

    fn lex_cal_prelude(&mut self) -> Lexed<'a> {
        self.skip_whitespace();
        let rest = self.rest();
        if rest.is_empty() {
            self.set_mode(Mode::Default);
            return None;
        }

        Some(if rest.starts_with("//") {
            let len = rest.find(['\r', '\n']).unwrap_or(rest.len());
            Ok(self.take(len, Token::LineComment))
        } else if let Some(body) = rest.strip_prefix("/*") {
            match body.find("*/") {
                Some(i) => Ok(self.take(i + 4, Token::BlockComment)),
                None => {
                    let span = Span::new(self.pos, self.source.len());
                    self.pos = span.end;
                    Err(Error::new(span, "unterminated block comment"))
                }
            }
        } else if rest.starts_with('{') {
            self.set_mode(Mode::CalBlock);
            Ok(self.take(1, |_| Token::LBrace))
        } else {
            Err(self.reject_char("unexpected character"))
        })
    }

    // ------------------------------------------------------------------
    // CAL_BLOCK mode — balanced-brace content, then '}'
    // ------------------------------------------------------------------

    fn lex_cal_block(&mut self) -> Lexed<'a> {
        // Content up to the brace that closes the block.
        let mut depth = 0usize;
        let rest = self.rest();
        let len = rest
            .bytes()
            .position(|b| match b {
                b'{' => {
                    depth += 1;
                    false
                }
                b'}' if depth > 0 => {
                    depth -= 1;
                    false
                }
                b'}' => true,
                _ => false,
            })
            .unwrap_or(rest.len());

        if len == 0 {
            self.set_mode(Mode::Default);
            return Some(Ok(self.take(1, |_| Token::RBrace)));
        }
        Some(Ok(self.take(len, Token::CalibrationBlock)))
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
        assert_eq!(lex("true false"), vec![Token::True, Token::False]);
    }

    #[test]
    fn openpulse_keywords() {
        let openpulse: Vec<_> = Lexer::new("waveform port frame cal defcal")
            .openpulse()
            .map(|r| r.unwrap().0)
            .collect();
        assert_eq!(
            openpulse,
            vec![
                Token::Waveform,
                Token::Port,
                Token::Frame,
                Token::Identifier("cal"),
                Token::Identifier("defcal"),
            ]
        );
        assert_eq!(
            lex("waveform port frame"),
            vec![
                Token::Identifier("waveform"),
                Token::Identifier("port"),
                Token::Identifier("frame"),
            ]
        );
    }

    #[test]
    fn rejected_character_is_consumed_whole() {
        // The error covers the whole multi-byte character, so lexing can go on.
        for src in ["include é;", "OPENQASM é;", "cal é { }"] {
            let results: Vec<_> = Lexer::new(src).collect();
            let err = results.iter().find_map(|r| r.as_ref().err()).unwrap();
            assert_eq!(&src[err.span], "é", "{src}");
        }
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
