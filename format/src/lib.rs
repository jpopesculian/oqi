//! Source formatter for OpenQASM 3.
//!
//! [`format`] parses the source and prints the syntax tree back out with
//! canonical spacing. Blank lines between statements and (outside compact
//! mode) comments are carried over from the source.

use oqi_parse::{Error, ast, parse_with_comments};

/// Format each item into the printer, in order.
macro_rules! put {
    ($p:expr, $($item:expr),+ $(,)?) => {{
        $( Format::format($item, $p); )+
    }};
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Config {
    /// Put as much as possible on one line, and drop comments.
    pub compact: bool,
}

impl Config {
    pub fn compact() -> Self {
        Self { compact: true }
    }
}

/// Format `source`. Non-empty output ends with exactly one newline.
pub fn format(source: &str, config: Config) -> Result<String, Error> {
    let (program, comments) = parse_with_comments(source)?;
    let mut printer = Printer::new(source, comments, config);
    program.format(&mut printer);
    Ok(printer.finish())
}

// ---------------------------------------------------------------------------
// Printer
// ---------------------------------------------------------------------------

const INDENT: &str = "    ";

/// The output being built, and the comments still to be placed in it.
struct Printer<'a> {
    out: String,
    compact: bool,
    indent: usize,
    comments: Vec<ast::Comment<'a>>,
    next_comment: usize,
    /// Byte offsets of every `\n` in the source.
    newlines: Vec<usize>,
    /// The offsets among `newlines` that end a blank (whitespace-only) line.
    blank_lines: Vec<usize>,
}

impl<'a> Printer<'a> {
    fn new(source: &str, comments: Vec<ast::Comment<'a>>, config: Config) -> Self {
        let mut newlines = Vec::new();
        let mut blank_lines = Vec::new();
        let mut line_is_blank = true;
        for (i, b) in source.bytes().enumerate() {
            if b == b'\n' {
                newlines.push(i);
                if line_is_blank {
                    blank_lines.push(i);
                }
                line_is_blank = true;
            } else if !b.is_ascii_whitespace() {
                line_is_blank = false;
            }
        }
        Self {
            out: String::new(),
            compact: config.compact,
            indent: 0,
            comments,
            next_comment: 0,
            newlines,
            blank_lines,
        }
    }

    fn finish(mut self) -> String {
        self.out.truncate(self.out.trim_end_matches('\n').len());
        if !self.out.is_empty() {
            self.out.push('\n');
        }
        self.out
    }

    // ---- Text ----

    fn at_line_start(&self) -> bool {
        self.out.is_empty() || self.out.ends_with('\n')
    }

    /// Write `text`, indenting it if it starts a line.
    fn write(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.at_line_start() {
            for _ in 0..self.indent {
                self.out.push_str(INDENT);
            }
        }
        self.out.push_str(text);
    }

    fn space(&mut self) {
        self.write(" ");
    }

    fn newline(&mut self) {
        self.out.push('\n');
    }

    /// End the current line, unless nothing is on it yet.
    fn fresh_line(&mut self) {
        if !self.at_line_start() {
            self.newline();
        }
    }

    // ---- Source layout ----

    fn has_newline_between(&self, start: usize, end: usize) -> bool {
        let i = self.newlines.partition_point(|&pos| pos < start);
        self.newlines.get(i).is_some_and(|&pos| pos < end)
    }

    /// Whether the source has a blank line between `start` and `end` that the
    /// output should keep.
    fn keeps_blank_line_between(&self, start: usize, end: usize) -> bool {
        let i = self.blank_lines.partition_point(|&pos| pos < start);
        !self.compact && self.blank_lines.get(i).is_some_and(|&pos| pos < end)
    }

    // ---- Comments ----

    /// The next comment to place, if it starts before `pos`.
    fn comment_before(&self, pos: usize) -> Option<ast::Comment<'a>> {
        let comment = *self.comments.get(self.next_comment)?;
        (comment.span.start < pos && !self.compact).then_some(comment)
    }

    /// Emit the comments that start before `pos`, ending the line after each.
    fn comments_before(&mut self, pos: usize) {
        let mut previous_end = None;
        while let Some(comment) = self.comment_before(pos) {
            self.next_comment += 1;
            if previous_end
                .is_some_and(|end| self.keeps_blank_line_between(end, comment.span.start))
            {
                self.newline();
            }
            if !self.at_line_start() && !self.out.ends_with(' ') {
                self.space();
            }
            self.write(comment.text);
            self.newline();
            previous_end = Some(comment.span.end);
        }
    }

    /// Emit the comments that follow `after` on its source line.
    fn trailing_comments(&mut self, after: usize) {
        while let Some(comment) = self.comment_before(usize::MAX)
            && !self.has_newline_between(after, comment.span.start)
        {
            self.next_comment += 1;
            self.space();
            self.write(comment.text);
        }
    }

    // ---- Structure ----

    fn list<T: Format>(&mut self, items: &[T], separator: &str) {
        for (index, item) in items.iter().enumerate() {
            if index > 0 {
                self.write(separator);
            }
            item.format(self);
        }
    }

    /// `(a, b, c)`
    fn parens<T: Format>(&mut self, items: &[T]) {
        self.write("(");
        self.list(items, ", ");
        self.write(")");
    }

    /// ` a, b, c`, or nothing for no operands.
    fn operands<T: Format>(&mut self, operands: &[T]) {
        if !operands.is_empty() {
            self.space();
            self.list(operands, ", ");
        }
    }

    /// `[designator]`, if there is one.
    fn designator(&mut self, designator: Option<&ast::Expr<'_>>) {
        if let Some(designator) = designator {
            put!(self, "[", designator, "]");
        }
    }

    /// ` -> ty`, if there is one.
    fn return_ty(&mut self, return_ty: Option<&ast::ScalarType<'_>>) {
        if let Some(return_ty) = return_ty {
            put!(self, " -> ", return_ty);
        }
    }

    /// A sequence of statements, with the comments that trail each.
    fn items(&mut self, items: &[ast::StmtOrScope<'_>]) {
        let mut previous: Option<&ast::StmtOrScope<'_>> = None;
        for item in items {
            if let Some(previous) = previous {
                if self.compact && !ends_line(previous) {
                    self.space();
                } else {
                    self.newline();
                    if self.keeps_blank_line_between(previous.span().end, item.span().start) {
                        self.newline();
                    }
                }
            }
            item.format(self);
            self.trailing_comments(item.span().end);
            previous = Some(item);
        }
    }

    /// `{ items }`, where the closing brace is at `close` in the source.
    fn block(&mut self, items: &[ast::StmtOrScope<'_>], close: usize) {
        self.write("{");
        let has_comments = self.comment_before(close).is_some();
        if items.is_empty() && !has_comments {
            return self.write("}");
        }

        if self.compact {
            self.space();
            self.items(items);
            if items.last().is_some_and(ends_line) {
                self.newline();
            } else {
                self.space();
            }
            return self.write("}");
        }

        self.indent += 1;
        self.newline();
        self.items(items);
        if self.comment_before(close).is_some() {
            self.fresh_line();
        }
        self.comments_before(close);
        self.indent -= 1;
        self.fresh_line();
        self.write("}");
    }
}

/// A pragma runs to the end of its line, so nothing may follow it there.
fn ends_line(item: &ast::StmtOrScope<'_>) -> bool {
    matches!(item, ast::StmtOrScope::Stmt(stmt) if matches!(stmt.kind, ast::StmtKind::Pragma(_)))
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

trait Format {
    fn format(&self, p: &mut Printer<'_>);
}

impl Format for str {
    fn format(&self, p: &mut Printer<'_>) {
        p.write(self);
    }
}

impl<T: Format + ?Sized> Format for &T {
    fn format(&self, p: &mut Printer<'_>) {
        (**self).format(p);
    }
}

impl<T: Format + ?Sized> Format for Box<T> {
    fn format(&self, p: &mut Printer<'_>) {
        (**self).format(p);
    }
}

impl Format for ast::Program<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        let mut end = 0;
        if let Some(version) = &self.version {
            p.comments_before(version.span.start);
            put!(p, "OPENQASM ", version.specifier, ";");
            p.trailing_comments(version.span.end);
            end = version.span.end;
            if let Some(first) = self.body.first() {
                p.newline();
                if p.keeps_blank_line_between(end, first.span().start) {
                    p.newline();
                }
            }
        }

        p.items(&self.body);
        if let Some(last) = self.body.last() {
            end = last.span().end;
        }

        // Comments after the last statement.
        if let Some(comment) = p.comment_before(usize::MAX) {
            p.fresh_line();
            if p.keeps_blank_line_between(end, comment.span.start) {
                p.newline();
            }
            p.comments_before(usize::MAX);
        }
    }
}

impl Format for ast::StmtOrScope<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::StmtOrScope::Stmt(stmt) => stmt.format(p),
            ast::StmtOrScope::Scope(scope) => scope.format(p),
        }
    }
}

impl Format for ast::Scope<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        p.block(&self.body, self.span.end);
    }
}

impl Format for ast::Stmt<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);

        for annotation in &self.annotations {
            p.comments_before(annotation.span.start);
            p.write(annotation.keyword);
            if let Some(content) = annotation.content {
                put!(p, " ", content);
            }
            p.newline();
        }

        use ast::StmtKind::*;
        match &self.kind {
            Pragma(content) => put!(p, "pragma ", content),
            CalibrationGrammar(path) => put!(p, "defcalgrammar ", path, ";"),
            Include(path) => put!(p, "include ", path, ";"),
            Break => p.write("break;"),
            Continue => p.write("continue;"),
            End => p.write("end;"),
            For {
                ty,
                var,
                iterable,
                body,
            } => put!(p, "for ", ty, " ", var, " in ", iterable, " ", body),
            If {
                condition,
                then_body,
                else_body,
            } => {
                put!(p, "if (", condition, ") ", then_body);
                if let Some(else_body) = else_body {
                    put!(p, " else ", else_body);
                }
            }
            Return(None) => p.write("return;"),
            Return(Some(value)) => put!(p, "return ", value, ";"),
            While { condition, body } => put!(p, "while (", condition, ") ", body),
            Switch { target, cases } => {
                put!(p, "switch (", target, ") {");
                if cases.is_empty() {
                    // `{}`
                } else if p.compact {
                    p.space();
                    p.list(cases, " ");
                    p.space();
                } else {
                    p.indent += 1;
                    for case in cases {
                        p.newline();
                        case.format(p);
                    }
                    p.indent -= 1;
                    p.newline();
                }
                p.write("}");
            }
            Barrier(operands) => {
                p.write("barrier");
                p.operands(operands);
                p.write(";");
            }
            Box { designator, body } => {
                p.write("box");
                p.designator(designator.as_ref());
                put!(p, " ", body);
            }
            Delay {
                designator,
                operands,
            } => {
                p.write("delay");
                p.designator(Some(designator));
                p.operands(operands);
                p.write(";");
            }
            Nop(operands) => {
                p.write("nop");
                p.operands(operands);
                p.write(";");
            }
            GateCall {
                modifiers,
                name,
                args,
                designator,
                operands,
            } => {
                p.list(modifiers, "");
                name.format(p);
                if let Some(args) = args {
                    p.parens(args);
                }
                p.designator(designator.as_deref());
                p.operands(operands);
                p.write(";");
            }
            MeasureArrow { measure, target } => {
                measure.format(p);
                if let Some(target) = target {
                    put!(p, " -> ", target);
                }
                p.write(";");
            }
            Reset(operand) => put!(p, "reset ", operand, ";"),
            Alias { name, value } => {
                put!(p, "let ", name, " = ");
                p.list(value, " ++ ");
                p.write(";");
            }
            ClassicalDecl { ty, name, init } => {
                put!(p, ty, " ", name);
                if let Some(init) = init {
                    put!(p, " = ", init);
                }
                p.write(";");
            }
            ConstDecl { ty, name, init } => put!(p, "const ", ty, " ", name, " = ", init, ";"),
            IoDecl { dir, ty, name } => put!(p, dir.as_str(), " ", ty, " ", name, ";"),
            OldStyleDecl {
                keyword,
                name,
                designator,
            } => {
                put!(p, keyword.as_str(), " ", name);
                p.designator(designator.as_deref());
                p.write(";");
            }
            QuantumDecl { ty, name } => put!(p, ty, " ", name, ";"),
            Def {
                name,
                params,
                return_ty,
                body,
            } => {
                put!(p, "def ", name);
                p.parens(params);
                p.return_ty(return_ty.as_ref());
                put!(p, " ", body);
            }
            Extern {
                name,
                params,
                return_ty,
            } => {
                put!(p, "extern ", name);
                p.parens(params);
                p.return_ty(return_ty.as_ref());
                p.write(";");
            }
            ExternFrame { name } => put!(p, "extern frame ", name, ";"),
            ExternPort { name } => put!(p, "extern port ", name, ";"),
            Gate {
                name,
                params,
                qubits,
                body,
            } => {
                put!(p, "gate ", name);
                if !params.is_empty() {
                    p.parens(params);
                }
                p.operands(qubits);
                put!(p, " ", body);
            }
            Assignment { target, op, value } => {
                put!(p, target, " ", op.as_str(), " ", value, ";")
            }
            Expr(expr) => put!(p, expr, ";"),
            Cal(body) => {
                p.write("cal ");
                cal_body(body, self.span.end, p);
            }
            Defcal {
                target,
                args,
                operands,
                return_ty,
                body,
            } => {
                put!(p, "defcal ", target);
                if !args.is_empty() {
                    p.parens(args);
                }
                p.operands(operands);
                p.return_ty(return_ty.as_ref());
                p.space();
                cal_body(body, self.span.end, p);
            }
        }
    }
}

/// The braced body of a `cal` or `defcal` that ends at `end` in the source.
fn cal_body(body: &ast::CalBody<'_>, end: usize, p: &mut Printer<'_>) {
    match body {
        ast::CalBody::Raw(raw) => put!(p, "{", raw.unwrap_or_default(), "}"),
        ast::CalBody::OpenPulse(stmts) => p.block(stmts, end),
    }
}

impl Format for ast::Expr<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span().start);

        match self {
            ast::Expr::Ident(ident) => ident.format(p),
            ast::Expr::HardwareQubit(text, _)
            | ast::Expr::IntLiteral(text, _, _)
            | ast::Expr::FloatLiteral(text, _)
            | ast::Expr::ImagLiteral(text, _)
            | ast::Expr::BitstringLiteral(text, _)
            | ast::Expr::TimingLiteral(text, _) => p.write(text),
            ast::Expr::BoolLiteral(value, _) => p.write(if *value { "true" } else { "false" }),
            ast::Expr::Paren(inner, _) => put!(p, "(", inner, ")"),
            ast::Expr::BinOp {
                left, op, right, ..
            } => {
                // As in the parser: the operand on the side an operator does
                // not associate to must bind tighter than the operator.
                let precedence = op.precedence();
                let (left_min, right_min) = if op.is_right_associative() {
                    (precedence + 1, precedence)
                } else {
                    (precedence, precedence + 1)
                };
                operand(left, left_min, p);
                put!(p, " ", op.as_str(), " ");
                operand(right, right_min, p);
            }
            ast::Expr::UnaryOp { op, operand: e, .. } => {
                p.write(op.as_str());
                operand(e, ast::UnOp::PRECEDENCE, p);
            }
            ast::Expr::Index { expr, index, .. } => {
                operand(expr, ast::IndexOp::PRECEDENCE, p);
                index.format(p);
            }
            ast::Expr::Call { name, args, .. } => {
                name.format(p);
                p.parens(args);
            }
            ast::Expr::Cast { ty, operand, .. } => put!(p, ty, "(", operand, ")"),
            ast::Expr::DurationOf { scope, .. } => put!(p, "durationof(", scope, ")"),
            ast::Expr::ArrayLiteral(array) => array.format(p),
        }
    }
}

/// Format an operand, parenthesized if it binds looser than `min_precedence`.
fn operand(expr: &ast::Expr<'_>, min_precedence: u8, p: &mut Printer<'_>) {
    if expr.precedence() < min_precedence {
        p.comments_before(expr.span().start);
        put!(p, "(", expr, ")");
    } else {
        expr.format(p);
    }
}

impl Format for ast::Ident<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        p.write(self.name);
    }
}

impl Format for ast::IndexOp<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        p.write("[");
        match &self.kind {
            ast::IndexKind::Set(exprs) => braces(exprs, p),
            ast::IndexKind::Items(items) => p.list(items, ", "),
        }
        p.write("]");
    }
}

/// `{a, b, c}`
fn braces(exprs: &[ast::Expr<'_>], p: &mut Printer<'_>) {
    p.write("{");
    p.list(exprs, ", ");
    p.write("}");
}

impl Format for ast::IndexItem<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::IndexItem::Single(expr) => expr.format(p),
            ast::IndexItem::Range(range) => range.format(p),
        }
    }
}

impl Format for ast::RangeExpr<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        if let Some(start) = &self.start {
            start.format(p);
        }
        p.write(":");
        if let Some(step) = &self.step {
            put!(p, step, ":");
        }
        if let Some(end) = &self.end {
            end.format(p);
        }
    }
}

impl Format for ast::IndexedIdent<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        self.name.format(p);
        p.list(&self.indices, "");
    }
}

impl Format for ast::ForIterable<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::ForIterable::Set(exprs, span) => {
                p.comments_before(span.start);
                braces(exprs, p);
            }
            ast::ForIterable::Range(range, span) => {
                p.comments_before(span.start);
                put!(p, "[", range, "]");
            }
            ast::ForIterable::Expr(expr) => expr.format(p),
        }
    }
}

impl Format for ast::SwitchCase<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::SwitchCase::Case(exprs, scope) => {
                p.write("case ");
                p.list(exprs, ", ");
                put!(p, " ", scope);
            }
            ast::SwitchCase::Default(scope) => put!(p, "default ", scope),
        }
    }
}

impl Format for ast::GateModifier<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        let (keyword, arg, span) = match self {
            ast::GateModifier::Inv(span) => ("inv", None, span),
            ast::GateModifier::Pow(exponent, span) => ("pow", Some(exponent), span),
            ast::GateModifier::Ctrl(count, span) => ("ctrl", count.as_ref(), span),
            ast::GateModifier::NegCtrl(count, span) => ("negctrl", count.as_ref(), span),
        };
        p.comments_before(span.start);
        p.write(keyword);
        if let Some(arg) = arg {
            put!(p, "(", arg, ")");
        }
        p.write(" @ ");
    }
}

impl Format for ast::GateOperand<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::GateOperand::Indexed(indexed) => indexed.format(p),
            ast::GateOperand::HardwareQubit(name, span) => {
                p.comments_before(span.start);
                p.write(name);
            }
        }
    }
}

impl Format for ast::MeasureExpr<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::MeasureExpr::Measure { operand, span } => {
                p.comments_before(span.start);
                put!(p, "measure ", operand);
            }
            ast::MeasureExpr::QuantumCall {
                name,
                args,
                operands,
                ..
            } => {
                name.format(p);
                if !args.is_empty() {
                    p.parens(args);
                }
                p.operands(operands);
            }
        }
    }
}

impl Format for ast::ExprOrMeasure<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::ExprOrMeasure::Expr(expr) => expr.format(p),
            ast::ExprOrMeasure::Measure(measure) => measure.format(p),
        }
    }
}

impl Format for ast::DefcalTarget<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        let (keyword, span) = match self {
            ast::DefcalTarget::Measure(span) => ("measure", span),
            ast::DefcalTarget::Reset(span) => ("reset", span),
            ast::DefcalTarget::Delay(span) => ("delay", span),
            ast::DefcalTarget::Ident(ident) => return ident.format(p),
        };
        p.comments_before(span.start);
        p.write(keyword);
    }
}

impl Format for ast::DefcalArgDef<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::DefcalArgDef::Expr(expr) => expr.format(p),
            ast::DefcalArgDef::ArgDef(arg) => arg.format(p),
        }
    }
}

impl Format for ast::DefcalOperand<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::DefcalOperand::HardwareQubit(name, span) => {
                p.comments_before(span.start);
                p.write(name);
            }
            ast::DefcalOperand::Ident(ident) => ident.format(p),
        }
    }
}

impl Format for ast::ArgDef<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::ArgDef::Scalar(ty, name) => put!(p, ty, " ", name),
            ast::ArgDef::Qubit(ty, name) => put!(p, ty, " ", name),
            ast::ArgDef::ArrayRef(ty, name) => put!(p, ty, " ", name),
            ast::ArgDef::Creg(name, designator) => {
                put!(p, "creg ", name);
                p.designator(designator.as_ref());
            }
            ast::ArgDef::Qreg(name, designator) => {
                put!(p, "qreg ", name);
                p.designator(designator.as_ref());
            }
        }
    }
}

impl Format for ast::ExternArg<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::ExternArg::Scalar(ty) => ty.format(p),
            ast::ExternArg::ArrayRef(ty) => ty.format(p),
            ast::ExternArg::Creg(designator) => {
                p.write("creg");
                p.designator(designator.as_ref());
            }
        }
    }
}

impl Format for ast::ArrayLiteral<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        braces(&self.items, p);
    }
}

impl Format for ast::ScalarType<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);

        use ast::ScalarKind::*;
        match &self.kind {
            Sized(kind, designator) => {
                p.write(kind.as_str());
                p.designator(designator.as_deref());
            }
            Complex(None) => p.write("complex"),
            Complex(Some(inner)) => put!(p, "complex[", inner, "]"),
            Bool => p.write("bool"),
            Duration => p.write("duration"),
            Stretch => p.write("stretch"),
            Waveform => p.write("waveform"),
            Port => p.write("port"),
            Frame => p.write("frame"),
        }
    }
}

impl Format for ast::QubitType<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        p.write("qubit");
        p.designator(self.designator.as_deref());
    }
}

impl Format for ast::ArrayType<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        put!(p, "array[", &self.element_type, ", ");
        p.list(&self.dimensions, ", ");
        p.write("]");
    }
}

impl Format for ast::ArrayRefType<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        p.comments_before(self.span.start);
        put!(
            p,
            self.mutability.as_str(),
            " array[",
            &self.element_type,
            ", "
        );
        match &self.dimensions {
            ast::ArrayRefDims::ExprList(exprs) => p.list(exprs, ", "),
            ast::ArrayRefDims::Dim(expr) => put!(p, "#dim = ", expr),
        }
        p.write("]");
    }
}

impl Format for ast::TypeExpr<'_> {
    fn format(&self, p: &mut Printer<'_>) {
        match self {
            ast::TypeExpr::Scalar(ty) => ty.format(p),
            ast::TypeExpr::Array(ty) => ty.format(p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oqi_parse::parse;

    #[test]
    fn pretty_formats_basic_program() {
        let source = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
if (true) { x q[0]; }"#;
        let formatted = format(source, Config::default()).unwrap();
        assert_eq!(
            formatted,
            concat!(
                "OPENQASM 3.0;\n",
                "include \"stdgates.inc\";\n",
                "qubit[2] q;\n",
                "if (true) {\n",
                "    x q[0];\n",
                "}\n"
            )
        );
    }

    #[test]
    fn compact_inlines_simple_scope() {
        let source = "if (a) { x q; y q; }";
        let formatted = format(source, Config::compact()).unwrap();
        assert_eq!(formatted, "if (a) { x q; y q; }\n");
    }

    #[test]
    fn output_ends_with_exactly_one_newline() {
        for source in ["x q;", "x q;\n", "x q;\n\n\n", "x q; // c", "// c\n\n"] {
            let formatted = format(source, Config::default()).unwrap();
            assert!(formatted.ends_with('\n'), "{formatted:?}");
            assert!(!formatted.ends_with("\n\n"), "{formatted:?}");
        }
        assert_eq!(format("", Config::default()).unwrap(), "");
        assert_eq!(format(" \n ", Config::default()).unwrap(), "");
    }

    #[test]
    fn keeps_comments_after_the_last_statement() {
        let source = "x q; // same line\n// own line\n\n/* footer */\n";
        assert_eq!(format(source, Config::default()).unwrap(), source);
        assert_eq!(
            format("// only a comment", Config::default()).unwrap(),
            "// only a comment\n"
        );
    }

    #[test]
    fn keeps_comments_in_openpulse_bodies() {
        let source = concat!(
            "defcalgrammar \"openpulse\";\n",
            "cal {\n",
            "    // first\n",
            "    extern port p0; /* same line */\n",
            "    // last\n",
            "}\n",
            "defcal x $0 {\n",
            "    // only a comment\n",
            "}\n",
        );
        assert_eq!(format(source, Config::default()).unwrap(), source);
    }

    #[test]
    fn comments_are_indented_with_the_code() {
        let source = "if (true) {\n// a\n  x q;\n\n  /* b */\n  y q;\n  // c\n}\n";
        assert_eq!(
            format(source, Config::default()).unwrap(),
            concat!(
                "if (true) {\n",
                "    // a\n",
                "    x q;\n",
                "\n",
                "    /* b */\n",
                "    y q;\n",
                "    // c\n",
                "}\n",
            )
        );
    }

    #[test]
    fn compact_keeps_a_pragma_on_its_own_line() {
        let source = "{ pragma foo\n }";
        let formatted = format(source, Config::compact()).unwrap();
        assert_eq!(formatted, "{ pragma foo\n}\n");
        assert!(parse(&formatted).is_ok());
    }

    #[test]
    fn parenthesizes_only_where_precedence_requires() {
        for (source, expected) in [
            ("a = (b + c) * d;", "a = (b + c) * d;\n"),
            ("a = b - (c - d);", "a = b - (c - d);\n"),
            ("a = (b ** c) ** d;", "a = (b ** c) ** d;\n"),
            ("a = b ** -c;", "a = b ** (-c);\n"),
            ("a = -b ** c;", "a = -b ** c;\n"),
            ("a = (b + c)[0];", "a = (b + c)[0];\n"),
        ] {
            assert_eq!(format(source, Config::default()).unwrap(), expected);
        }
    }

    #[test]
    fn formatted_output_round_trips() {
        let source = r#"OPENQASM 3.0;
def rx_gate(angle[32] theta, qubit q) -> angle[32] {
    return theta;
}
measure $0 -> c[0];"#;
        let formatted = format(source, Config::default()).unwrap();
        assert!(parse(&formatted).is_ok(), "{formatted}");
    }
}
