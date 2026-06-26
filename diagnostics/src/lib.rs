//! Shared diagnostic rendering for the OpenQASM toolchain.
//!
//! Every error type in the workspace implements [`Diagnostic`], which exposes a
//! stable [`Code`], a headline message, source [`DiagLabel`]s (one primary plus
//! optional secondary labels), and free-form notes. [`emit`] renders a
//! diagnostic to stderr through [`ariadne`]; [`render_to_string`] produces the
//! same report without colour for tests and tooling.
//!
//! # Code table
//!
//! Codes are a stable contract. The leading letter is the [`CodeKind`]; the
//! number identifies the specific error within that category.
//!
//! - `C####` — compile errors (`oqi_compile::ErrorKind`)
//! - `R####` — runtime errors (`oqi_vm::VmError`)
//! - `S####` — syntax / parse errors (`oqi_parse::Error`)

use std::borrow::Cow;
use std::fmt;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::str::FromStr;

use ariadne::{Config, Label, Report, ReportKind, Source};
use oqi_lex::Span;

/// Tab width used when resolving a byte offset to a `line:column` position.
const DIAGNOSTIC_TAB_SIZE: usize = 4;

/// The category of a diagnostic, rendered as the code's leading letter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeKind {
    Compile,
    Runtime,
    Syntax,
}

impl CodeKind {
    /// The single-letter prefix used in the rendered code.
    pub const fn letter(self) -> char {
        match self {
            CodeKind::Compile => 'C',
            CodeKind::Runtime => 'R',
            CodeKind::Syntax => 'S',
        }
    }
}

/// A stable error code such as `C0002`. Round-trips via [`Display`](fmt::Display)
/// and [`FromStr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Code {
    pub kind: CodeKind,
    pub num: u16,
}

impl Code {
    pub const fn compile(num: u16) -> Self {
        Self {
            kind: CodeKind::Compile,
            num,
        }
    }

    pub const fn runtime(num: u16) -> Self {
        Self {
            kind: CodeKind::Runtime,
            num,
        }
    }

    pub const fn syntax(num: u16) -> Self {
        Self {
            kind: CodeKind::Syntax,
            num,
        }
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:04}", self.kind.letter(), self.num)
    }
}

/// Returned when [`Code::from_str`] is given an unrecognised string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeParseError;

impl fmt::Display for CodeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid diagnostic code")
    }
}

impl std::error::Error for CodeParseError {}

impl FromStr for Code {
    type Err = CodeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut chars = s.chars();
        let kind = match chars.next() {
            Some('C') => CodeKind::Compile,
            Some('R') => CodeKind::Runtime,
            Some('S') => CodeKind::Syntax,
            _ => return Err(CodeParseError),
        };
        let num = chars.as_str().parse().map_err(|_| CodeParseError)?;
        Ok(Code { kind, num })
    }
}

/// A single annotated span within a diagnostic.
pub struct DiagLabel {
    pub span: Span,
    pub message: String,
    pub primary: bool,
}

impl DiagLabel {
    /// The main label — its location drives the report's `line:column`.
    pub fn primary(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            primary: true,
        }
    }

    /// A supporting label, e.g. "first defined here".
    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            primary: false,
        }
    }
}

/// An error that can be rendered as a source-annotated diagnostic.
pub trait Diagnostic {
    /// The stable code for this diagnostic.
    fn code(&self) -> Code;

    /// The headline message (no location prefix).
    fn message(&self) -> String;

    /// Source labels, primary first followed by any secondary labels.
    fn labels(&self) -> Vec<DiagLabel>;

    /// Free-form help / note lines shown beneath the snippet.
    fn notes(&self) -> Vec<String> {
        Vec::new()
    }

    /// The file the primary label refers to, when it differs from the file
    /// being processed (e.g. an error inside an `include`d file).
    fn path(&self) -> Option<&Path> {
        None
    }
}

/// Render `diag` to stderr with colour.
pub fn emit(diag: &dyn Diagnostic, root_path: &Path, root_source: &str) {
    let mut stderr = std::io::stderr().lock();
    render(diag, root_path, root_source, true, &mut stderr);
}

/// Render `diag` to a string without colour — for tests and tooling.
pub fn render_to_string(diag: &dyn Diagnostic, root_path: &Path, root_source: &str) -> String {
    let mut buf = Vec::new();
    render(diag, root_path, root_source, false, &mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// A span carries no location when it is the default `0..0` sentinel.
fn has_location(span: Span) -> bool {
    span.start != 0 || span.end != 0
}

fn render(
    diag: &dyn Diagnostic,
    root_path: &Path,
    root_source: &str,
    color: bool,
    w: &mut impl Write,
) {
    let diagnostic_path = diag.path().unwrap_or(root_path);
    let source: Cow<str> = if diagnostic_path != root_path {
        match std::fs::read_to_string(diagnostic_path) {
            Ok(s) => Cow::Owned(s),
            Err(_) => Cow::Borrowed(root_source),
        }
    } else {
        Cow::Borrowed(root_source)
    };

    let labels = diag.labels();
    let primary = labels.iter().find(|l| l.primary).or_else(|| labels.first());

    // Without a usable span there is nothing to point at — fall back to a
    // single headline line so the diagnostic still reaches the user.
    let Some(primary) = primary.filter(|l| has_location(l.span)) else {
        let _ = writeln!(
            w,
            "{}: error[{}]: {}",
            diagnostic_path.display(),
            diag.code(),
            diag.message()
        );
        return;
    };

    let (line, column) = primary
        .span
        .doc_position(source.as_ref(), DIAGNOSTIC_TAB_SIZE);
    let filename = diagnostic_path.display().to_string();
    let headline = format!("{filename}:{line}:{column}: {}", diag.message());

    let mut report = Report::build(ReportKind::Error, (&filename, Range::from(primary.span)))
        .with_config(Config::default().with_color(color))
        .with_code(diag.code())
        .with_message(headline);

    for label in &labels {
        report = report.with_label(
            Label::new((&filename, Range::from(label.span))).with_message(&label.message),
        );
    }

    let notes = diag.notes();
    if !notes.is_empty() {
        report = report.with_note(notes.join("\n"));
    }

    let _ = report
        .finish()
        .write((&filename, Source::from(source.as_ref())), w);
}

/// Syntax / parse errors (`oqi_lex::Error`, re-exported as `oqi_parse::Error`)
/// are a single flat `{ span, message }` shape, so they share one code.
impl Diagnostic for oqi_lex::Error {
    fn code(&self) -> Code {
        Code::syntax(1)
    }

    fn message(&self) -> String {
        self.message.clone()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        vec![DiagLabel::primary(self.span, "syntax error")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_round_trips() {
        for code in [Code::compile(2), Code::runtime(13), Code::syntax(1)] {
            let rendered = code.to_string();
            assert_eq!(rendered.parse::<Code>().unwrap(), code);
        }
        assert_eq!(Code::compile(2).to_string(), "C0002");
        assert_eq!(Code::runtime(13).to_string(), "R0013");
        assert_eq!(Code::syntax(1).to_string(), "S0001");
        assert!("X0001".parse::<Code>().is_err());
        assert!("Cnope".parse::<Code>().is_err());
    }

    struct Fake;

    impl Diagnostic for Fake {
        fn code(&self) -> Code {
            Code::compile(2)
        }
        fn message(&self) -> String {
            "duplicate definition of 'x'".into()
        }
        fn labels(&self) -> Vec<DiagLabel> {
            vec![
                DiagLabel::primary(Span::new(10, 11), "duplicate definition"),
                DiagLabel::secondary(Span::new(0, 1), "first defined here"),
            ]
        }
        fn notes(&self) -> Vec<String> {
            vec!["a name may only be defined once per scope".into()]
        }
    }

    #[test]
    fn renders_code_labels_and_note() {
        let source = "x;\n      x;\n";
        let out = render_to_string(&Fake, Path::new("test.qasm"), source);
        assert!(out.contains("C0002"), "missing code:\n{out}");
        assert!(
            out.contains("duplicate definition"),
            "missing primary label:\n{out}"
        );
        assert!(
            out.contains("first defined here"),
            "missing secondary label:\n{out}"
        );
        assert!(
            out.contains("a name may only be defined once per scope"),
            "missing note:\n{out}"
        );
    }

    #[test]
    fn falls_back_without_location() {
        struct NoSpan;
        impl Diagnostic for NoSpan {
            fn code(&self) -> Code {
                Code::runtime(7)
            }
            fn message(&self) -> String {
                "something went wrong".into()
            }
            fn labels(&self) -> Vec<DiagLabel> {
                vec![DiagLabel::primary(Span::default(), "here")]
            }
        }
        let out = render_to_string(&NoSpan, Path::new("test.qasm"), "");
        assert!(out.contains("R0007"), "{out}");
        assert!(out.contains("something went wrong"), "{out}");
    }
}
