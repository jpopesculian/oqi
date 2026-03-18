use oqi_lexer::{Lexer, Token};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct AntlrToken {
    r#type: String,
    text: String,
    start: usize,
    stop: usize,
}

fn antlr_type_name<'a>(token: &Token<'a>) -> &'static str {
    match token {
        Token::OpenQasm => "OPENQASM",
        Token::Include => "INCLUDE",
        Token::DefCalGrammar => "DEFCALGRAMMAR",
        Token::Def => "DEF",
        Token::Cal => "CAL",
        Token::DefCal => "DEFCAL",
        Token::Gate => "GATE",
        Token::Extern => "EXTERN",
        Token::Box => "BOX",
        Token::Let => "LET",
        Token::Break => "BREAK",
        Token::Continue => "CONTINUE",
        Token::If => "IF",
        Token::Else => "ELSE",
        Token::End => "END",
        Token::Return => "RETURN",
        Token::For => "FOR",
        Token::While => "WHILE",
        Token::In => "IN",
        Token::Switch => "SWITCH",
        Token::Case => "CASE",
        Token::Default => "DEFAULT",
        Token::Nop => "NOP",
        Token::Pragma => "PRAGMA",
        Token::AnnotationKeyword(_) => "AnnotationKeyword",
        Token::Input => "INPUT",
        Token::Output => "OUTPUT",
        Token::Const => "CONST",
        Token::Readonly => "READONLY",
        Token::Mutable => "MUTABLE",
        Token::Qreg => "QREG",
        Token::Qubit => "QUBIT",
        Token::Creg => "CREG",
        Token::Bool => "BOOL",
        Token::Bit => "BIT",
        Token::Int => "INT",
        Token::Uint => "UINT",
        Token::Float => "FLOAT",
        Token::Angle => "ANGLE",
        Token::Complex => "COMPLEX",
        Token::Array => "ARRAY",
        Token::Void => "VOID",
        Token::Duration => "DURATION",
        Token::Stretch => "STRETCH",
        Token::Gphase => "GPHASE",
        Token::Inv => "INV",
        Token::Pow => "POW",
        Token::Ctrl => "CTRL",
        Token::Negctrl => "NEGCTRL",
        Token::Dim => "DIM",
        Token::Durationof => "DURATIONOF",
        Token::Delay => "DELAY",
        Token::Reset => "RESET",
        Token::Measure => "MEASURE",
        Token::Barrier => "BARRIER",
        Token::Imag => "IMAG",
        Token::BooleanLiteral(_) => "BooleanLiteral",
        Token::BinaryIntegerLiteral(_) => "BinaryIntegerLiteral",
        Token::OctalIntegerLiteral(_) => "OctalIntegerLiteral",
        Token::DecimalIntegerLiteral(_) => "DecimalIntegerLiteral",
        Token::HexIntegerLiteral(_) => "HexIntegerLiteral",
        Token::FloatLiteral(_) => "FloatLiteral",
        Token::ImaginaryLiteral(_) => "ImaginaryLiteral",
        Token::TimingLiteral(_) => "TimingLiteral",
        Token::BitstringLiteral(_) => "BitstringLiteral",
        Token::Identifier(_) => "Identifier",
        Token::HardwareQubit(_) => "HardwareQubit",
        Token::LBracket => "LBRACKET",
        Token::RBracket => "RBRACKET",
        Token::LBrace => "LBRACE",
        Token::RBrace => "RBRACE",
        Token::LParen => "LPAREN",
        Token::RParen => "RPAREN",
        Token::Colon => "COLON",
        Token::Semicolon => "SEMICOLON",
        Token::Dot => "DOT",
        Token::Comma => "COMMA",
        Token::Equals => "EQUALS",
        Token::Arrow => "ARROW",
        Token::Plus => "PLUS",
        Token::DoublePlus => "DOUBLE_PLUS",
        Token::Minus => "MINUS",
        Token::Asterisk => "ASTERISK",
        Token::DoubleAsterisk => "DOUBLE_ASTERISK",
        Token::Slash => "SLASH",
        Token::Percent => "PERCENT",
        Token::Pipe => "PIPE",
        Token::DoublePipe => "DOUBLE_PIPE",
        Token::Ampersand => "AMPERSAND",
        Token::DoubleAmpersand => "DOUBLE_AMPERSAND",
        Token::Caret => "CARET",
        Token::At => "AT",
        Token::Tilde => "TILDE",
        Token::ExclamationPoint => "EXCLAMATION_POINT",
        Token::EqualityOperator(_) => "EqualityOperator",
        Token::CompoundAssignmentOperator(_) => "CompoundAssignmentOperator",
        Token::ComparisonOperator(_) => "ComparisonOperator",
        Token::BitshiftOperator(_) => "BitshiftOperator",
        Token::VersionSpecifier(_) => "VersionSpecifier",
        Token::StringLiteral(_) => "StringLiteral",
        Token::RemainingLineContent(_) => "RemainingLineContent",
        Token::CalibrationBlock(_) => "CalibrationBlock",
        Token::LineComment(_) => "LineComment",
        Token::BlockComment(_) => "BlockComment",
    }
}

/// Convert a byte offset to a character (code point) offset.
/// ANTLR uses character indices; our lexer uses byte indices.
fn byte_to_char(source: &str, byte_offset: usize) -> usize {
    source[..byte_offset].chars().count()
}

fn check_fixture(name: &str) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let qasm_path = base.join("fixtures/qasm").join(format!("{name}.qasm"));
    let json_path = base.join("fixtures/lexer").join(format!("{name}.json"));

    let source = std::fs::read_to_string(&qasm_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", qasm_path.display()));
    let fixture: Vec<AntlrToken> = serde_json::from_str(
        &std::fs::read_to_string(&json_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_path.display())),
    )
    .unwrap();

    let our_tokens: Vec<_> = Lexer::new(&source)
        .filter_map(|r| r.ok())
        .collect();

    assert_eq!(
        our_tokens.len(),
        fixture.len(),
        "{name}: token count mismatch (ours={}, antlr={})",
        our_tokens.len(),
        fixture.len(),
    );

    for (i, (ours, expected)) in our_tokens.iter().zip(fixture.iter()).enumerate() {
        let (token, span) = ours;
        let our_type = antlr_type_name(token);
        let our_text = &source[span.clone()];

        assert_eq!(
            our_type, expected.r#type,
            "{name} token #{i}: type mismatch\n  ours:   {our_type} {our_text:?}\n  antlr:  {} {:?}",
            expected.r#type, expected.text,
        );

        assert_eq!(
            our_text, expected.text,
            "{name} token #{i} ({our_type}): text mismatch",
        );

        let char_start = byte_to_char(&source, span.start);
        let char_stop = byte_to_char(&source, span.end) - 1; // inclusive

        assert_eq!(
            char_start, expected.start,
            "{name} token #{i} ({our_type}): start offset mismatch (ours={char_start}, antlr={})",
            expected.start,
        );

        assert_eq!(
            char_stop,
            expected.stop,
            "{name} token #{i} ({our_type}): stop offset mismatch (ours={char_stop}, antlr={})",
            expected.stop,
        );
    }
}

macro_rules! fixture_test {
    ($name:ident) => {
        #[test]
        fn $name() {
            check_fixture(stringify!($name));
        }
    };
}

fixture_test!(adder);
fixture_test!(alignment);
fixture_test!(arrays);
fixture_test!(cphase);
fixture_test!(dd);
fixture_test!(defcal);
fixture_test!(gateteleport);
fixture_test!(inverseqft1);
fixture_test!(inverseqft2);
fixture_test!(ipe);
fixture_test!(msd);
fixture_test!(qec);
fixture_test!(qft);
fixture_test!(qpt);
fixture_test!(rb);
fixture_test!(rus);
fixture_test!(scqec);
fixture_test!(t1);
fixture_test!(teleport);
fixture_test!(varteleport);
fixture_test!(vqe);
