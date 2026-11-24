//! Diagnostic rendering for the OpenQASM toolchain.
//!
//! The errors a user sees — syntax errors (`oqi_lex::Error`), compile errors
//! (`oqi_compile::CompileError`) and runtime errors (`oqi_vm::Error`) —
//! implement [`Diagnostic`]: a stable [`Code`], a headline message, source
//! [`DiagLabel`]s (one primary plus optional secondary labels) and free-form
//! notes. [`emit`] renders a diagnostic to stderr; [`render_to_string`]
//! produces the same report without colour, for tests and tooling.
//!
//! ```
//! use std::path::Path;
//! use oqi_diagnostics::{render_to_string, Span};
//!
//! let source = "int x = ;\n";
//! let error = oqi_lex::Error::new(Span::new(8, 9), "expected expression");
//! let report = render_to_string(&error, Path::new("main.qasm"), source, &|_| None);
//! assert!(report.contains("main.qasm:1:9: expected expression"));
//! assert!(report.contains("S0001"));
//! ```
//!
//! # Code table
//!
//! Codes are a stable contract. The leading letter is the [`CodeKind`]; the
//! number identifies the specific error within that category.
//!
//! - `C####` — compile errors (`oqi_compile::ErrorKind`)
//! - `R####` — runtime errors (`oqi_vm::VmError`)
//! - `S####` — syntax errors (`oqi_lex::Error`, also `oqi_parse::Error`)

use std::borrow::Cow;
use std::fmt;
use std::io::{IsTerminal, Write};
use std::ops::Range;
use std::path::Path;
use std::str::FromStr;

use ariadne::{Config, IndexType, Label, Report, ReportKind, Source};
pub use oqi_lex::Span;

/// Tab width used when resolving a byte offset to a `line:column` position.
const DIAGNOSTIC_TAB_SIZE: usize = 4;

/// The category of a diagnostic, rendered as the code's leading letter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum CodeKind {
    /// `C`: the program is well-formed but invalid (names, types, scoping).
    Compile,
    /// `R`: running the program failed.
    Runtime,
    /// `S`: the program does not parse.
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

/// A stable error code such as `C0002`: a [`CodeKind`] letter and four
/// digits. Round-trips via [`Display`](fmt::Display) and [`FromStr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Code {
    /// The category, rendered as the leading letter.
    pub kind: CodeKind,
    /// The number within the category, rendered as four digits.
    pub num: u16,
}

impl Code {
    /// A compile error code, `C{num:04}`.
    pub const fn compile(num: u16) -> Self {
        Self {
            kind: CodeKind::Compile,
            num,
        }
    }

    /// A runtime error code, `R{num:04}`.
    pub const fn runtime(num: u16) -> Self {
        Self {
            kind: CodeKind::Runtime,
            num,
        }
    }

    /// A syntax error code, `S{num:04}`.
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

/// Returned when [`Code::from_str`] is given anything but a code letter and
/// exactly four digits.
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
        let digits = chars.as_str();
        if digits.len() != 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(CodeParseError);
        }
        let num = digits.parse().map_err(|_| CodeParseError)?;
        Ok(Code { kind, num })
    }
}

/// A single annotated span within a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagLabel {
    /// The source range the label points at. The default `0..0` span means
    /// "no location": such a label is not drawn.
    pub span: Span,
    /// The text shown under the span.
    pub message: String,
    /// Whether this is the label the report's `line:column` refers to.
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
    /// being processed (e.g. an error inside an `include`d file). The
    /// renderer asks its `included` lookup for that file's text.
    fn path(&self) -> Option<&Path> {
        None
    }
}

/// Render `diag` to stderr: in colour when stderr is a terminal and
/// `NO_COLOR` is not set.
///
/// The diagnostic is drawn against `root_source` (the file at `root_path`),
/// or, when it points into another file ([`Diagnostic::path`]), against the
/// text `included` returns for that file. Without that text only a headline
/// is printed.
pub fn emit(
    diag: &dyn Diagnostic,
    root_path: &Path,
    root_source: &str,
    included: &dyn Fn(&Path) -> Option<String>,
) {
    let color = std::io::stderr().is_terminal()
        && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty());
    let mut stderr = std::io::stderr().lock();
    render(diag, root_path, root_source, included, color, &mut stderr);
}

/// Render `diag` to a string without colour, like [`emit`] — for tests and
/// tooling.
pub fn render_to_string(
    diag: &dyn Diagnostic,
    root_path: &Path,
    root_source: &str,
    included: &dyn Fn(&Path) -> Option<String>,
) -> String {
    let mut buf = Vec::new();
    render(diag, root_path, root_source, included, false, &mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// The span `diag` primarily points at, or `None` when it carries no location
/// (the default `0..0` span).
pub fn primary_span(diag: &dyn Diagnostic) -> Option<Span> {
    let labels = diag.labels();
    let primary = labels.iter().find(|l| l.primary).or(labels.first())?;
    (primary.span != Span::default()).then_some(primary.span)
}

fn render(
    diag: &dyn Diagnostic,
    root_path: &Path,
    root_source: &str,
    included: &dyn Fn(&Path) -> Option<String>,
    color: bool,
    w: &mut impl Write,
) {
    let diagnostic_path = diag.path().unwrap_or(root_path);
    // The text the diagnostic points into: empty when it is another file's
    // and that file is unavailable, so no span fits and only the headline
    // is printed.
    let source: Cow<str> = if diagnostic_path != root_path {
        included(diagnostic_path).map_or(Cow::Borrowed(""), Cow::Owned)
    } else {
        Cow::Borrowed(root_source)
    };

    // A span fits the source when it lies inside it, on character
    // boundaries. One that does not belongs to text we could not load (an
    // include that is not a file, say), and cannot be shown.
    let fits = |span: Span| source.get(span.start..span.end).is_some();

    // Without a usable span there is nothing to point at — fall back to a
    // single headline line so the diagnostic still reaches the user.
    let Some(primary) = primary_span(diag).filter(|&span| fits(span)) else {
        let _ = writeln!(
            w,
            "{}: error[{}]: {}",
            diagnostic_path.display(),
            diag.code(),
            diag.message()
        );
        return;
    };

    let (line, column) = line_column(source.as_ref(), primary.start);
    let filename = diagnostic_path.display().to_string();
    let headline = format!("{filename}:{line}:{column}: {}", diag.message());

    let mut report = Report::build(ReportKind::Error, (&filename, Range::from(primary)))
        .with_config(
            Config::default()
                .with_color(color)
                .with_index_type(IndexType::Byte),
        )
        .with_code(diag.code())
        .with_message(headline);

    for label in diag.labels().iter().filter(|label| fits(label.span)) {
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

/// The 1-based line and column of byte `offset` in `source`, counting a tab
/// as [`DIAGNOSTIC_TAB_SIZE`] columns.
fn line_column(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut column = 1;
    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        match c {
            '\n' => {
                line += 1;
                column = 1;
            }
            '\t' => column += DIAGNOSTIC_TAB_SIZE,
            _ => column += 1,
        }
    }
    (line, column)
}

/// Syntax / parse errors (`oqi_lex::Error`, re-exported as `oqi_parse::Error`)
/// are a single flat `{ span, message }` shape, so they share one code.
impl Diagnostic for oqi_lex::Error {
    fn code(&self) -> Code {
        Code::syntax(1)
    }

    fn message(&self) -> String {
        self.message().to_string()
    }

    fn labels(&self) -> Vec<DiagLabel> {
        vec![DiagLabel::primary(self.span(), "syntax error")]
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
        for bad in ["X0001", "Cnope", "C1", "C+001", "C00001", "C-001", "", "C"] {
            assert!(bad.parse::<Code>().is_err(), "{bad:?} parsed");
        }
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
        let out = render_to_string(&Fake, Path::new("test.qasm"), source, &|_| None);
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
    fn spans_are_byte_offsets() {
        // The span is in bytes: characters before it that take several bytes
        // must not shift (or lose) the snippet.
        let source = "// µµµµ ééé\nint x = ;\n";
        let at = source.find(';').unwrap();
        let err = oqi_lex::Error::new(Span::new(at, at + 1), "expected expression");
        let out = render_to_string(&err, Path::new("test.qasm"), source, &|_| None);
        assert!(out.contains("test.qasm:2:9"), "{out}");
        assert!(out.contains("int x = ;"), "missing snippet:\n{out}");
        assert!(out.contains("syntax error"), "missing label:\n{out}");
    }

    #[test]
    fn a_span_outside_the_source_still_renders() {
        // E.g. an error inside an included file that cannot be re-read.
        for span in [Span::new(40, 45), Span::new(3, 400), Span::new(1, 2)] {
            let err = oqi_lex::Error::new(span, "somewhere else");
            let out = render_to_string(&err, Path::new("test.qasm"), "é;\n", &|_| None);
            assert!(out.contains("S0001"), "{out}");
            assert!(out.contains("somewhere else"), "{out}");
        }
    }

    #[test]
    fn an_empty_span_at_the_end_of_the_source_renders() {
        // Where the parser reports what it expected after the last token.
        for source in ["int x =", "int x =\n", "é"] {
            let end = source.len();
            let err = oqi_lex::Error::new(Span::new(end, end), "expected expression");
            let out = render_to_string(&err, Path::new("test.qasm"), source, &|_| None);
            assert!(out.contains("syntax error"), "missing label:\n{out}");
        }
    }

    #[test]
    fn primary_span_skips_the_no_location_sentinel() {
        assert_eq!(primary_span(&Fake), Some(Span::new(10, 11)));
        let nowhere = oqi_lex::Error::new(Span::default(), "nowhere");
        assert_eq!(primary_span(&nowhere), None);
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
        let out = render_to_string(&NoSpan, Path::new("test.qasm"), "", &|_| None);
        assert!(out.contains("R0007"), "{out}");
        assert!(out.contains("something went wrong"), "{out}");
    }

    /// An error inside an included file, at `5..8` of that file.
    struct InInclude;

    impl Diagnostic for InInclude {
        fn code(&self) -> Code {
            Code::compile(1)
        }
        fn message(&self) -> String {
            "undefined name 'zzz'".into()
        }
        fn labels(&self) -> Vec<DiagLabel> {
            vec![DiagLabel::primary(
                Span::new(5, 8),
                "not found in this scope",
            )]
        }
        fn path(&self) -> Option<&Path> {
            Some(Path::new("lib.inc"))
        }
    }

    #[test]
    fn an_error_in_an_included_file_renders_against_that_file() {
        let root = "include \"lib.inc\";\n";
        let lib = |path: &Path| (path == Path::new("lib.inc")).then(|| "int\n zzz;\n".to_string());
        let out = render_to_string(&InInclude, Path::new("main.qasm"), root, &lib);
        assert!(out.contains("lib.inc:2:2"), "{out}");
        assert!(
            out.contains(" zzz;"),
            "missing the included snippet:\n{out}"
        );
        assert!(
            !out.contains("include"),
            "rendered against the root:\n{out}"
        );
    }

    #[test]
    fn an_included_file_that_cannot_be_read_gets_a_headline() {
        // The span happens to fit the root source too; it must not be drawn
        // there.
        let root = "include \"lib.inc\";\n";
        let out = render_to_string(&InInclude, Path::new("main.qasm"), root, &|_| None);
        assert!(
            out.contains("lib.inc: error[C0001]: undefined name 'zzz'"),
            "{out}"
        );
        assert!(
            !out.contains("include"),
            "rendered against the root:\n{out}"
        );
    }
}
