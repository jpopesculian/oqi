mod utils;

use core::fmt;

use oqi_lex::{Lexer, Span, Token};
use oqi_parse::{Error as ParseError, Parser, ast};

pub struct Display<'a, T> {
    ast: T,
    context: Context<'a>,
    config: Config,
}

impl<'a, T> Display<'a, T> {
    pub fn new(ast: T, context: Context<'a>, config: Config) -> Self {
        Self {
            ast,
            context,
            config,
        }
    }
}

impl<'a> Display<'a, ast::Program<'a>> {
    pub fn from_source(
        source: &'a str,
        config: Config,
    ) -> Result<Display<'a, ast::Program<'a>>, ParseError> {
        let lex = Lexer::new(source);
        let tokens = lex.collect::<Result<Vec<_>, _>>()?;
        let context = Context::new(
            tokens
                .iter()
                .cloned()
                .filter_map(Comment::from_lex)
                .collect(),
            utils::find_newlines(source),
        );
        let ast = Parser::new(tokens).parse_program()?;
        Ok(Display::new(ast, context, config))
    }
}

impl<'a, T> fmt::Display for Display<'a, T>
where
    T: Format,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.ast
            .format(fmt, &mut self.context.clone(), &self.config)
    }
}

pub fn format(source: &str, config: Config) -> Result<String, ParseError> {
    Ok(Display::from_source(source, config)?.to_string())
}

pub trait Format {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Config {
    pub compact: bool,
}

impl Config {
    pub fn compact() -> Self {
        Self { compact: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentKind {
    Line,
    Block,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comment<'a> {
    pub kind: CommentKind,
    pub text: &'a str,
    pub span: Span,
}

impl<'a> Comment<'a> {
    pub fn from_lex((tok, span): (Token<'a>, Span)) -> Option<Self> {
        match tok {
            Token::LineComment(text) => Some(Comment {
                kind: CommentKind::Line,
                text,
                span,
            }),
            Token::BlockComment(text) => Some(Comment {
                kind: CommentKind::Block,
                text,
                span,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Context<'a> {
    indent: usize,
    line: usize,
    column: usize,
    comments: Vec<Comment<'a>>,
    newlines: Vec<usize>,
    blank_lines: Vec<usize>,
    next_comment: usize,
}

impl<'a> Default for Context<'a> {
    fn default() -> Self {
        let nl = utils::Newlines {
            all: Vec::new(),
            blank: Vec::new(),
        };
        Self::new(Vec::new(), nl)
    }
}

impl<'a> Context<'a> {
    pub fn new(comments: Vec<Comment<'a>>, newlines: utils::Newlines) -> Self {
        Self {
            indent: 0,
            line: 1,
            column: 1,
            comments,
            newlines: newlines.all,
            blank_lines: newlines.blank,
            next_comment: 0,
        }
    }

    pub fn from_source(source: &'a str) -> Result<Self, ParseError> {
        Ok(Context::new(
            Lexer::new(source)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .filter_map(Comment::from_lex)
                .collect(),
            utils::find_newlines(source),
        ))
    }

    pub fn indent(&self) -> usize {
        self.indent
    }

    pub fn line(&self) -> usize {
        self.line
    }

    pub fn column(&self) -> usize {
        self.column
    }

    pub fn comments(&self) -> &[Comment<'a>] {
        &self.comments
    }

    pub fn comments_in<'b>(&'b self, span: &'b Span) -> impl Iterator<Item = &'b Comment<'a>> {
        self.comments
            .iter()
            .filter(move |comment| span.start <= comment.span.start && comment.span.end <= span.end)
    }

    pub fn comments_before(&self, pos: usize) -> impl Iterator<Item = &Comment<'a>> {
        self.comments
            .iter()
            .filter(move |comment| comment.span.end <= pos)
    }

    pub fn comments_after(&self, pos: usize) -> impl Iterator<Item = &Comment<'a>> {
        self.comments
            .iter()
            .filter(move |comment| pos <= comment.span.start)
    }

    fn has_pending_comments_before(&self, pos: usize) -> bool {
        self.comments
            .get(self.next_comment)
            .is_some_and(|comment| comment.span.start < pos)
    }

    pub fn finish(&mut self, fmt: &mut fmt::Formatter<'_>, config: &Config) -> fmt::Result {
        self.emit_comments_before(fmt, usize::MAX, config)?;
        Ok(())
    }

    fn emit_comments_before(
        &mut self,
        fmt: &mut fmt::Formatter<'_>,
        pos: usize,
        config: &Config,
    ) -> Result<bool, fmt::Error> {
        let mut emitted = false;
        let mut prev_end: Option<usize> = None;

        while let Some(comment) = self.comments.get(self.next_comment).cloned() {
            if pos <= comment.span.start {
                break;
            }
            self.next_comment += 1;
            if config.compact {
                continue;
            }
            if let Some(end) = prev_end
                && self.has_blank_line_between(end, comment.span.start) {
                    self.newline(fmt)?;
                }
            self.write_comment(fmt, &comment, config)?;
            prev_end = Some(comment.span.end);
            emitted = true;
        }

        Ok(emitted)
    }

    fn write_comment(
        &mut self,
        fmt: &mut fmt::Formatter<'_>,
        comment: &Comment<'_>,
        config: &Config,
    ) -> fmt::Result {
        if self.column > 1 {
            self.write_str(fmt, " ")?;
        }

        self.write_str(fmt, comment.text)?;

        match comment.kind {
            CommentKind::Line => self.newline(fmt),
            CommentKind::Block if config.compact => self.write_str(fmt, " "),
            CommentKind::Block => self.newline(fmt),
        }
    }

    fn write_str(&mut self, fmt: &mut fmt::Formatter<'_>, text: &str) -> fmt::Result {
        fmt.write_str(text)?;

        for ch in text.chars() {
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }

        Ok(())
    }

    fn space(&mut self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_str(fmt, " ")
    }

    fn newline(&mut self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_str(fmt, "\n")?;
        self.write_indent(fmt)
    }

    fn write_indent(&mut self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.indent {
            self.write_str(fmt, "    ")?;
        }
        Ok(())
    }

    fn push_indent(&mut self) {
        self.indent += 1;
    }

    fn pop_indent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    fn has_newline_between(&self, start: usize, end: usize) -> bool {
        let i = self.newlines.partition_point(|&pos| pos < start);
        self.newlines.get(i).is_some_and(|&pos| pos < end)
    }

    fn has_blank_line_between(&self, start: usize, end: usize) -> bool {
        let i = self.blank_lines.partition_point(|&pos| pos < start);
        self.blank_lines.get(i).is_some_and(|&pos| pos < end)
    }

    fn emit_trailing_comments(
        &mut self,
        fmt: &mut fmt::Formatter<'_>,
        after: usize,
        config: &Config,
    ) -> fmt::Result {
        while let Some(comment) = self.comments.get(self.next_comment) {
            if self.has_newline_between(after, comment.span.start) {
                break;
            }
            let comment = comment.clone();
            self.next_comment += 1;
            if config.compact {
                continue;
            }
            self.space(fmt)?;
            self.write_str(fmt, comment.text)?;
        }
        Ok(())
    }

    fn scope_separator<T: Format + CompactHint>(
        &mut self,
        fmt: &mut fmt::Formatter<'_>,
        previous: Option<&T>,
        next: &T,
        config: &Config,
    ) -> fmt::Result {
        let Some(previous) = previous else {
            return Ok(());
        };

        if !config.compact || previous.requires_trailing_newline() {
            self.newline(fmt)?;
            if !config.compact
                && self.has_blank_line_between(previous.span_end(), next.compact_anchor())
            {
                self.newline(fmt)?;
            }
            Ok(())
        } else {
            self.space(fmt)
        }
    }
}

trait CompactHint {
    fn compact_anchor(&self) -> usize;
    fn span_end(&self) -> usize;

    fn requires_trailing_newline(&self) -> bool {
        false
    }
}

impl<'a> CompactHint for ast::StmtOrScope<'a> {
    fn compact_anchor(&self) -> usize {
        span_of_stmt_or_scope(self).start
    }

    fn span_end(&self) -> usize {
        span_of_stmt_or_scope(self).end
    }

    fn requires_trailing_newline(&self) -> bool {
        matches!(
            self,
            ast::StmtOrScope::Stmt(ast::Stmt {
                kind: ast::StmtKind::Pragma(_),
                ..
            })
        )
    }
}

impl<'a> Format for ast::Program<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;

        let mut previous: Option<&ast::StmtOrScope<'_>> = None;

        if let Some(version) = &self.version {
            version.format(fmt, ctx, config)?;
            ctx.emit_trailing_comments(fmt, version.span.end, config)?;
            if !self.body.is_empty() {
                ctx.newline(fmt)?;
                if !config.compact
                    && ctx.has_blank_line_between(
                        version.span.end,
                        span_of_stmt_or_scope(&self.body[0]).start,
                    )
                {
                    ctx.newline(fmt)?;
                }
            }
        }

        for item in &self.body {
            if previous.is_some() {
                ctx.scope_separator(fmt, previous, item, config)?;
            }
            item.format(fmt, ctx, config)?;
            ctx.emit_trailing_comments(fmt, span_of_stmt_or_scope(item).end, config)?;
            previous = Some(item);
        }

        Ok(())
    }
}

impl<'a> Format for ast::Version<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, "OPENQASM ")?;
        ctx.write_str(fmt, self.specifier)?;
        ctx.write_str(fmt, ";")
    }
}

impl<'a> Format for ast::StmtOrScope<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::StmtOrScope::Stmt(stmt) => stmt.format(fmt, ctx, config),
            ast::StmtOrScope::Scope(scope) => scope.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::Scope<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, "{")?;

        let has_body = !self.body.is_empty();
        let has_comments = !config.compact && ctx.has_pending_comments_before(self.span.end);

        if !has_body && !has_comments {
            return ctx.write_str(fmt, "}");
        }

        if config.compact {
            let mut previous: Option<&ast::StmtOrScope<'_>> = None;
            for item in &self.body {
                if previous.is_none() {
                    ctx.space(fmt)?;
                }
                ctx.scope_separator(fmt, previous, item, config)?;
                item.format(fmt, ctx, config)?;
                ctx.emit_trailing_comments(fmt, span_of_stmt_or_scope(item).end, config)?;
                previous = Some(item);
            }
            let emitted_comments = ctx.emit_comments_before(fmt, self.span.end, config)?;
            if !emitted_comments && ctx.column > 2 {
                ctx.space(fmt)?;
            }
            return ctx.write_str(fmt, "}");
        }

        ctx.push_indent();
        ctx.newline(fmt)?;

        let mut previous: Option<&ast::StmtOrScope<'_>> = None;
        for item in &self.body {
            ctx.scope_separator(fmt, previous, item, config)?;
            item.format(fmt, ctx, config)?;
            ctx.emit_trailing_comments(fmt, span_of_stmt_or_scope(item).end, config)?;
            previous = Some(item);
        }

        if previous.is_some() && ctx.has_pending_comments_before(self.span.end) {
            ctx.newline(fmt)?;
        }
        ctx.emit_comments_before(fmt, self.span.end, config)?;
        ctx.pop_indent();
        if ctx.column > 1 {
            ctx.newline(fmt)?;
        }
        ctx.write_str(fmt, "}")
    }
}

impl<'a> Format for ast::Annotation<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, self.keyword)?;
        if let Some(content) = self.content {
            ctx.space(fmt)?;
            ctx.write_str(fmt, content)?;
        }
        Ok(())
    }
}

impl<'a> Format for ast::Stmt<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;

        for annotation in &self.annotations {
            annotation.format(fmt, ctx, config)?;
            ctx.newline(fmt)?;
        }

        self.kind.format(fmt, ctx, config)
    }
}

impl<'a> Format for ast::StmtKind<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::StmtKind::Pragma(content) => {
                ctx.write_str(fmt, "pragma ")?;
                ctx.write_str(fmt, content)
            }
            ast::StmtKind::CalibrationGrammar(path) => {
                ctx.write_str(fmt, "defcalgrammar ")?;
                ctx.write_str(fmt, path)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Include(path) => {
                ctx.write_str(fmt, "include ")?;
                ctx.write_str(fmt, path)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Break => ctx.write_str(fmt, "break;"),
            ast::StmtKind::Continue => ctx.write_str(fmt, "continue;"),
            ast::StmtKind::End => ctx.write_str(fmt, "end;"),
            ast::StmtKind::For {
                ty,
                var,
                iterable,
                body,
            } => {
                ctx.write_str(fmt, "for ")?;
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                var.format(fmt, ctx, config)?;
                ctx.write_str(fmt, " in ")?;
                iterable.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                body.format(fmt, ctx, config)
            }
            ast::StmtKind::If {
                condition,
                then_body,
                else_body,
            } => {
                ctx.write_str(fmt, "if (")?;
                condition.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ") ")?;
                then_body.format(fmt, ctx, config)?;
                if let Some(else_body) = else_body {
                    ctx.write_str(fmt, " else ")?;
                    else_body.format(fmt, ctx, config)?;
                }
                Ok(())
            }
            ast::StmtKind::Return(None) => ctx.write_str(fmt, "return;"),
            ast::StmtKind::Return(Some(value)) => {
                ctx.write_str(fmt, "return ")?;
                value.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::While { condition, body } => {
                ctx.write_str(fmt, "while (")?;
                condition.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ") ")?;
                body.format(fmt, ctx, config)
            }
            ast::StmtKind::Switch { target, cases } => {
                ctx.write_str(fmt, "switch (")?;
                target.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ") {")?;

                if cases.is_empty() {
                    return ctx.write_str(fmt, "}");
                }

                if config.compact {
                    ctx.space(fmt)?;
                    let mut first = true;
                    for case in cases {
                        if !first {
                            ctx.space(fmt)?;
                        }
                        case.format(fmt, ctx, config)?;
                        first = false;
                    }
                    ctx.space(fmt)?;
                    return ctx.write_str(fmt, "}");
                }

                ctx.push_indent();
                ctx.newline(fmt)?;
                for (index, case) in cases.iter().enumerate() {
                    if index > 0 {
                        ctx.newline(fmt)?;
                    }
                    case.format(fmt, ctx, config)?;
                }
                ctx.pop_indent();
                ctx.newline(fmt)?;
                ctx.write_str(fmt, "}")
            }
            ast::StmtKind::Barrier(operands) => {
                ctx.write_str(fmt, "barrier")?;
                if !operands.is_empty() {
                    ctx.space(fmt)?;
                    format_slice(operands, fmt, ctx, config, ", ")?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Box { designator, body } => {
                ctx.write_str(fmt, "box")?;
                if let Some(designator) = designator {
                    format_designator(designator, fmt, ctx, config)?;
                }
                ctx.space(fmt)?;
                body.format(fmt, ctx, config)
            }
            ast::StmtKind::Delay {
                designator,
                operands,
            } => {
                ctx.write_str(fmt, "delay")?;
                format_designator(designator, fmt, ctx, config)?;
                if !operands.is_empty() {
                    ctx.space(fmt)?;
                    format_slice(operands, fmt, ctx, config, ", ")?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Nop(operands) => {
                ctx.write_str(fmt, "nop")?;
                if !operands.is_empty() {
                    ctx.space(fmt)?;
                    format_slice(operands, fmt, ctx, config, ", ")?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::GateCall {
                modifiers,
                name,
                args,
                designator,
                operands,
            } => {
                for modifier in modifiers {
                    modifier.format(fmt, ctx, config)?;
                }
                name.format(fmt, ctx, config)?;
                if let Some(args) = args {
                    ctx.write_str(fmt, "(")?;
                    format_slice(args, fmt, ctx, config, ", ")?;
                    ctx.write_str(fmt, ")")?;
                }
                if let Some(designator) = designator {
                    format_designator(designator, fmt, ctx, config)?;
                }
                if !operands.is_empty() {
                    ctx.space(fmt)?;
                    format_slice(operands, fmt, ctx, config, ", ")?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::MeasureArrow { measure, target } => {
                measure.format(fmt, ctx, config)?;
                if let Some(target) = target {
                    ctx.write_str(fmt, " -> ")?;
                    target.format(fmt, ctx, config)?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Reset(operand) => {
                ctx.write_str(fmt, "reset ")?;
                operand.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Alias { name, value } => {
                ctx.write_str(fmt, "let ")?;
                name.format(fmt, ctx, config)?;
                ctx.write_str(fmt, " = ")?;
                format_slice(value, fmt, ctx, config, " ++ ")?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::ClassicalDecl { ty, name, init } => {
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)?;
                if let Some(init) = init {
                    ctx.write_str(fmt, " = ")?;
                    init.format(fmt, ctx, config)?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::ConstDecl { ty, name, init } => {
                ctx.write_str(fmt, "const ")?;
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)?;
                ctx.write_str(fmt, " = ")?;
                init.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::IoDecl { dir, ty, name } => {
                dir.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::OldStyleDecl {
                keyword,
                name,
                designator,
            } => {
                keyword.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)?;
                if let Some(designator) = designator {
                    format_designator(designator, fmt, ctx, config)?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::QuantumDecl { ty, name } => {
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Def {
                name,
                params,
                return_ty,
                body,
            } => {
                ctx.write_str(fmt, "def ")?;
                name.format(fmt, ctx, config)?;
                ctx.write_str(fmt, "(")?;
                format_slice(params, fmt, ctx, config, ", ")?;
                ctx.write_str(fmt, ")")?;
                if let Some(return_ty) = return_ty {
                    ctx.write_str(fmt, " -> ")?;
                    return_ty.format(fmt, ctx, config)?;
                }
                ctx.space(fmt)?;
                body.format(fmt, ctx, config)
            }
            ast::StmtKind::Extern {
                name,
                params,
                return_ty,
            } => {
                ctx.write_str(fmt, "extern ")?;
                name.format(fmt, ctx, config)?;
                ctx.write_str(fmt, "(")?;
                format_slice(params, fmt, ctx, config, ", ")?;
                ctx.write_str(fmt, ")")?;
                if let Some(return_ty) = return_ty {
                    ctx.write_str(fmt, " -> ")?;
                    return_ty.format(fmt, ctx, config)?;
                }
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Gate {
                name,
                params,
                qubits,
                body,
            } => {
                ctx.write_str(fmt, "gate ")?;
                name.format(fmt, ctx, config)?;
                if !params.is_empty() {
                    ctx.write_str(fmt, "(")?;
                    format_slice(params, fmt, ctx, config, ", ")?;
                    ctx.write_str(fmt, ")")?;
                }
                ctx.space(fmt)?;
                format_slice(qubits, fmt, ctx, config, ", ")?;
                ctx.space(fmt)?;
                body.format(fmt, ctx, config)
            }
            ast::StmtKind::Assignment { target, op, value } => {
                target.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                op.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                value.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Expr(expr) => {
                expr.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ";")
            }
            ast::StmtKind::Cal(body) => {
                ctx.write_str(fmt, "cal {")?;
                if let Some(body) = body {
                    ctx.write_str(fmt, body)?;
                }
                ctx.write_str(fmt, "}")
            }
            ast::StmtKind::Defcal {
                target,
                args,
                operands,
                return_ty,
                body,
            } => {
                ctx.write_str(fmt, "defcal ")?;
                target.format(fmt, ctx, config)?;
                if !args.is_empty() {
                    ctx.write_str(fmt, "(")?;
                    format_slice(args, fmt, ctx, config, ", ")?;
                    ctx.write_str(fmt, ")")?;
                }
                if !operands.is_empty() {
                    ctx.space(fmt)?;
                    format_slice(operands, fmt, ctx, config, ", ")?;
                }
                if let Some(return_ty) = return_ty {
                    ctx.write_str(fmt, " -> ")?;
                    return_ty.format(fmt, ctx, config)?;
                }
                ctx.write_str(fmt, " {")?;
                if let Some(body) = body {
                    ctx.write_str(fmt, body)?;
                }
                ctx.write_str(fmt, "}")
            }
        }
    }
}

impl Format for ast::BinOp {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        _config: &Config,
    ) -> fmt::Result {
        ctx.write_str(fmt, bin_op_text(*self))
    }
}

impl Format for ast::UnOp {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        _config: &Config,
    ) -> fmt::Result {
        ctx.write_str(fmt, un_op_text(*self))
    }
}

impl<'a> Format for ast::Expr<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        format_expr(self, fmt, ctx, config, None)
    }
}

impl<'a> Format for ast::Ident<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, self.name)
    }
}

impl<'a> Format for ast::IndexOp<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, "[")?;
        self.kind.format(fmt, ctx, config)?;
        ctx.write_str(fmt, "]")
    }
}

impl<'a> Format for ast::IndexKind<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::IndexKind::Set(exprs) => {
                ctx.write_str(fmt, "{")?;
                format_slice(exprs, fmt, ctx, config, ", ")?;
                ctx.write_str(fmt, "}")
            }
            ast::IndexKind::Items(items) => format_slice(items, fmt, ctx, config, ", "),
        }
    }
}

impl<'a> Format for ast::IndexItem<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::IndexItem::Single(expr) => expr.format(fmt, ctx, config),
            ast::IndexItem::Range(range) => range.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::RangeExpr<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        if let Some(start) = &self.start {
            start.format(fmt, ctx, config)?;
        }
        ctx.write_str(fmt, ":")?;
        if let Some(end) = &self.end {
            end.format(fmt, ctx, config)?;
        }
        if let Some(step) = &self.step {
            ctx.write_str(fmt, ":")?;
            step.format(fmt, ctx, config)?;
        }
        Ok(())
    }
}

impl<'a> Format for ast::IndexedIdent<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        self.name.format(fmt, ctx, config)?;
        for index in &self.indices {
            index.format(fmt, ctx, config)?;
        }
        Ok(())
    }
}

impl<'a> Format for ast::ForIterable<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::ForIterable::Set(exprs, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "{")?;
                format_slice(exprs, fmt, ctx, config, ", ")?;
                ctx.write_str(fmt, "}")
            }
            ast::ForIterable::Range(range, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "[")?;
                range.format(fmt, ctx, config)?;
                ctx.write_str(fmt, "]")
            }
            ast::ForIterable::Expr(expr) => expr.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::SwitchCase<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::SwitchCase::Case(exprs, scope) => {
                ctx.write_str(fmt, "case ")?;
                format_slice(exprs, fmt, ctx, config, ", ")?;
                ctx.space(fmt)?;
                scope.format(fmt, ctx, config)
            }
            ast::SwitchCase::Default(scope) => {
                ctx.write_str(fmt, "default ")?;
                scope.format(fmt, ctx, config)
            }
        }
    }
}

impl Format for ast::IoDir {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        _config: &Config,
    ) -> fmt::Result {
        ctx.write_str(
            fmt,
            match self {
                ast::IoDir::Input => "input",
                ast::IoDir::Output => "output",
            },
        )
    }
}

impl Format for ast::OldStyleKind {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        _config: &Config,
    ) -> fmt::Result {
        ctx.write_str(
            fmt,
            match self {
                ast::OldStyleKind::Creg => "creg",
                ast::OldStyleKind::Qreg => "qreg",
            },
        )
    }
}

impl Format for ast::AssignOp {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        _config: &Config,
    ) -> fmt::Result {
        let s = match self {
            ast::AssignOp::Assign => "=",
            ast::AssignOp::AddAssign => "+=",
            ast::AssignOp::SubAssign => "-=",
            ast::AssignOp::MulAssign => "*=",
            ast::AssignOp::DivAssign => "/=",
            ast::AssignOp::BitAndAssign => "&=",
            ast::AssignOp::BitOrAssign => "|=",
            ast::AssignOp::BitXorAssign => "^=",
            ast::AssignOp::LeftShiftAssign => "<<=",
            ast::AssignOp::RightShiftAssign => ">>=",
            ast::AssignOp::ModAssign => "%=",
            ast::AssignOp::PowAssign => "**=",
        };
        ctx.write_str(fmt, s)
    }
}

impl<'a> Format for ast::GateCallName<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::GateCallName::Ident(ident) => ident.format(fmt, ctx, config),
            ast::GateCallName::Gphase(span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "gphase")
            }
        }
    }
}

impl<'a> Format for ast::GateModifier<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::GateModifier::Inv(span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "inv @ ")
            }
            ast::GateModifier::Pow(expr, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "pow(")?;
                expr.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ") @ ")
            }
            ast::GateModifier::Ctrl(None, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "ctrl @ ")
            }
            ast::GateModifier::Ctrl(Some(expr), span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "ctrl(")?;
                expr.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ") @ ")
            }
            ast::GateModifier::NegCtrl(None, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "negctrl @ ")
            }
            ast::GateModifier::NegCtrl(Some(expr), span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "negctrl(")?;
                expr.format(fmt, ctx, config)?;
                ctx.write_str(fmt, ") @ ")
            }
        }
    }
}

impl<'a> Format for ast::GateOperand<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::GateOperand::Indexed(indexed) => indexed.format(fmt, ctx, config),
            ast::GateOperand::HardwareQubit(name, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, name)
            }
        }
    }
}

impl<'a> Format for ast::MeasureExpr<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::MeasureExpr::Measure { operand, span } => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "measure ")?;
                operand.format(fmt, ctx, config)
            }
            ast::MeasureExpr::QuantumCall {
                name,
                args,
                operands,
                span,
            } => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                name.format(fmt, ctx, config)?;
                if !args.is_empty() {
                    ctx.write_str(fmt, "(")?;
                    format_slice(args, fmt, ctx, config, ", ")?;
                    ctx.write_str(fmt, ")")?;
                }
                ctx.space(fmt)?;
                format_slice(operands, fmt, ctx, config, ", ")
            }
        }
    }
}

impl<'a> Format for ast::ExprOrMeasure<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::ExprOrMeasure::Expr(expr) => expr.format(fmt, ctx, config),
            ast::ExprOrMeasure::Measure(measure) => measure.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::DeclExpr<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::DeclExpr::Expr(expr) => expr.format(fmt, ctx, config),
            ast::DeclExpr::Measure(measure) => measure.format(fmt, ctx, config),
            ast::DeclExpr::ArrayLiteral(array) => array.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::DefcalTarget<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::DefcalTarget::Measure(span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "measure")
            }
            ast::DefcalTarget::Reset(span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "reset")
            }
            ast::DefcalTarget::Delay(span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, "delay")
            }
            ast::DefcalTarget::Ident(ident) => ident.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::DefcalArgDef<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::DefcalArgDef::Expr(expr) => expr.format(fmt, ctx, config),
            ast::DefcalArgDef::ArgDef(arg) => arg.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::DefcalOperand<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::DefcalOperand::HardwareQubit(name, span) => {
                ctx.emit_comments_before(fmt, span.start, config)?;
                ctx.write_str(fmt, name)
            }
            ast::DefcalOperand::Ident(ident) => ident.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::ArgDef<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::ArgDef::Scalar(ty, name) => {
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)
            }
            ast::ArgDef::Qubit(ty, name) => {
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)
            }
            ast::ArgDef::Creg(name, designator) => {
                ctx.write_str(fmt, "creg ")?;
                name.format(fmt, ctx, config)?;
                if let Some(designator) = designator {
                    format_designator(designator, fmt, ctx, config)?;
                }
                Ok(())
            }
            ast::ArgDef::Qreg(name, designator) => {
                ctx.write_str(fmt, "qreg ")?;
                name.format(fmt, ctx, config)?;
                if let Some(designator) = designator {
                    format_designator(designator, fmt, ctx, config)?;
                }
                Ok(())
            }
            ast::ArgDef::ArrayRef(ty, name) => {
                ty.format(fmt, ctx, config)?;
                ctx.space(fmt)?;
                name.format(fmt, ctx, config)
            }
        }
    }
}

impl<'a> Format for ast::ExternArg<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::ExternArg::Scalar(ty) => ty.format(fmt, ctx, config),
            ast::ExternArg::ArrayRef(ty) => ty.format(fmt, ctx, config),
            ast::ExternArg::Creg(None) => ctx.write_str(fmt, "creg"),
            ast::ExternArg::Creg(Some(designator)) => {
                ctx.write_str(fmt, "creg")?;
                format_designator(designator, fmt, ctx, config)
            }
        }
    }
}

impl<'a> Format for ast::ArrayLiteral<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, "{")?;
        format_slice(&self.items, fmt, ctx, config, ", ")?;
        ctx.write_str(fmt, "}")
    }
}

impl<'a> Format for ast::ArrayLiteralItem<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::ArrayLiteralItem::Expr(expr) => expr.format(fmt, ctx, config),
            ast::ArrayLiteralItem::Nested(array) => array.format(fmt, ctx, config),
        }
    }
}

impl<'a> Format for ast::ScalarType<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span().start, config)?;

        match self {
            ast::ScalarType::Bit(designator, _) => {
                ctx.write_str(fmt, "bit")?;
                format_optional_designator(designator.as_deref(), fmt, ctx, config)
            }
            ast::ScalarType::Int(designator, _) => {
                ctx.write_str(fmt, "int")?;
                format_optional_designator(designator.as_deref(), fmt, ctx, config)
            }
            ast::ScalarType::Uint(designator, _) => {
                ctx.write_str(fmt, "uint")?;
                format_optional_designator(designator.as_deref(), fmt, ctx, config)
            }
            ast::ScalarType::Float(designator, _) => {
                ctx.write_str(fmt, "float")?;
                format_optional_designator(designator.as_deref(), fmt, ctx, config)
            }
            ast::ScalarType::Angle(designator, _) => {
                ctx.write_str(fmt, "angle")?;
                format_optional_designator(designator.as_deref(), fmt, ctx, config)
            }
            ast::ScalarType::Bool(_) => ctx.write_str(fmt, "bool"),
            ast::ScalarType::Duration(_) => ctx.write_str(fmt, "duration"),
            ast::ScalarType::Stretch(_) => ctx.write_str(fmt, "stretch"),
            ast::ScalarType::Complex(None, _) => ctx.write_str(fmt, "complex"),
            ast::ScalarType::Complex(Some(inner), _) => {
                ctx.write_str(fmt, "complex[")?;
                inner.format(fmt, ctx, config)?;
                ctx.write_str(fmt, "]")
            }
        }
    }
}

impl<'a> Format for ast::QubitType<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, "qubit")?;
        format_optional_designator(self.designator.as_deref(), fmt, ctx, config)
    }
}

impl<'a> Format for ast::ArrayType<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        ctx.write_str(fmt, "array[")?;
        self.element_type.format(fmt, ctx, config)?;
        ctx.write_str(fmt, ", ")?;
        format_slice(&self.dimensions, fmt, ctx, config, ", ")?;
        ctx.write_str(fmt, "]")
    }
}

impl Format for ast::ArrayRefMut {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        _config: &Config,
    ) -> fmt::Result {
        ctx.write_str(
            fmt,
            match self {
                ast::ArrayRefMut::Readonly => "readonly",
                ast::ArrayRefMut::Mutable => "mutable",
            },
        )
    }
}

impl<'a> Format for ast::ArrayRefType<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        ctx.emit_comments_before(fmt, self.span.start, config)?;
        self.mutability.format(fmt, ctx, config)?;
        ctx.write_str(fmt, " array[")?;
        self.element_type.format(fmt, ctx, config)?;
        ctx.write_str(fmt, ", ")?;
        self.dimensions.format(fmt, ctx, config)?;
        ctx.write_str(fmt, "]")
    }
}

impl<'a> Format for ast::ArrayRefDims<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::ArrayRefDims::ExprList(exprs) => format_slice(exprs, fmt, ctx, config, ", "),
            ast::ArrayRefDims::Dim(expr) => {
                ctx.write_str(fmt, "#dim = ")?;
                expr.format(fmt, ctx, config)
            }
        }
    }
}

impl<'a> Format for ast::TypeExpr<'a> {
    fn format(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        ctx: &mut Context<'_>,
        config: &Config,
    ) -> fmt::Result {
        match self {
            ast::TypeExpr::Scalar(ty) => ty.format(fmt, ctx, config),
            ast::TypeExpr::Array(ty) => ty.format(fmt, ctx, config),
        }
    }
}

fn format_expr(
    expr: &ast::Expr<'_>,
    fmt: &mut fmt::Formatter<'_>,
    ctx: &mut Context<'_>,
    config: &Config,
    parent: Option<ExprParent>,
) -> fmt::Result {
    ctx.emit_comments_before(fmt, expr.span().start, config)?;

    let needs_parens = parent.is_some_and(|parent| needs_expr_parens(expr, parent));
    if needs_parens {
        ctx.write_str(fmt, "(")?;
    }

    match expr {
        ast::Expr::Ident(ident) => ident.format(fmt, ctx, config)?,
        ast::Expr::HardwareQubit(name, _) => ctx.write_str(fmt, name)?,
        ast::Expr::IntLiteral(value, _, _) => ctx.write_str(fmt, value)?,
        ast::Expr::FloatLiteral(value, _) => ctx.write_str(fmt, value)?,
        ast::Expr::ImagLiteral(value, _) => ctx.write_str(fmt, value)?,
        ast::Expr::BoolLiteral(value, _) => {
            ctx.write_str(fmt, if *value { "true" } else { "false" })?
        }
        ast::Expr::BitstringLiteral(value, _) => ctx.write_str(fmt, value)?,
        ast::Expr::TimingLiteral(value, _) => ctx.write_str(fmt, value)?,
        ast::Expr::Paren(inner, _) => {
            ctx.write_str(fmt, "(")?;
            format_expr(inner, fmt, ctx, config, None)?;
            ctx.write_str(fmt, ")")?;
        }
        ast::Expr::BinOp {
            left, op, right, ..
        } => {
            let precedence = bin_op_precedence(*op);
            format_expr(
                left,
                fmt,
                ctx,
                config,
                Some(ExprParent::Binary {
                    precedence,
                    side: ExprSide::Left,
                }),
            )?;
            ctx.space(fmt)?;
            op.format(fmt, ctx, config)?;
            ctx.space(fmt)?;
            format_expr(
                right,
                fmt,
                ctx,
                config,
                Some(ExprParent::Binary {
                    precedence,
                    side: ExprSide::Right,
                }),
            )?;
        }
        ast::Expr::UnaryOp { op, operand, .. } => {
            op.format(fmt, ctx, config)?;
            format_expr(
                operand,
                fmt,
                ctx,
                config,
                Some(ExprParent::Unary { precedence: 21 }),
            )?;
        }
        ast::Expr::Index { expr, index, .. } => {
            format_expr(
                expr,
                fmt,
                ctx,
                config,
                Some(ExprParent::Postfix { precedence: 26 }),
            )?;
            index.format(fmt, ctx, config)?;
        }
        ast::Expr::Call { name, args, .. } => {
            name.format(fmt, ctx, config)?;
            ctx.write_str(fmt, "(")?;
            format_slice(args, fmt, ctx, config, ", ")?;
            ctx.write_str(fmt, ")")?;
        }
        ast::Expr::Cast { ty, operand, .. } => {
            ty.format(fmt, ctx, config)?;
            ctx.write_str(fmt, "(")?;
            operand.format(fmt, ctx, config)?;
            ctx.write_str(fmt, ")")?;
        }
        ast::Expr::DurationOf { scope, .. } => {
            ctx.write_str(fmt, "durationof(")?;
            scope.format(fmt, ctx, config)?;
            ctx.write_str(fmt, ")")?;
        }
    }

    if needs_parens {
        ctx.write_str(fmt, ")")?;
    }

    Ok(())
}

fn format_slice<T: Format>(
    items: &[T],
    fmt: &mut fmt::Formatter<'_>,
    ctx: &mut Context<'_>,
    config: &Config,
    separator: &str,
) -> fmt::Result {
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            ctx.write_str(fmt, separator)?;
        }
        item.format(fmt, ctx, config)?;
    }
    Ok(())
}

fn format_designator(
    designator: &ast::Expr<'_>,
    fmt: &mut fmt::Formatter<'_>,
    ctx: &mut Context<'_>,
    config: &Config,
) -> fmt::Result {
    ctx.write_str(fmt, "[")?;
    designator.format(fmt, ctx, config)?;
    ctx.write_str(fmt, "]")
}

fn format_optional_designator(
    designator: Option<&ast::Expr<'_>>,
    fmt: &mut fmt::Formatter<'_>,
    ctx: &mut Context<'_>,
    config: &Config,
) -> fmt::Result {
    if let Some(designator) = designator {
        format_designator(designator, fmt, ctx, config)?;
    }
    Ok(())
}

fn span_of_stmt_or_scope<'a>(node: &'a ast::StmtOrScope<'a>) -> &'a Span {
    match node {
        ast::StmtOrScope::Stmt(stmt) => &stmt.span,
        ast::StmtOrScope::Scope(scope) => &scope.span,
    }
}

fn expr_precedence(expr: &ast::Expr<'_>) -> u8 {
    match expr {
        ast::Expr::BinOp { op, .. } => bin_op_precedence(*op),
        ast::Expr::UnaryOp { .. } => 21,
        ast::Expr::Index { .. } => 26,
        ast::Expr::Paren(..)
        | ast::Expr::Call { .. }
        | ast::Expr::Cast { .. }
        | ast::Expr::DurationOf { .. }
        | ast::Expr::Ident(_)
        | ast::Expr::HardwareQubit(..)
        | ast::Expr::IntLiteral(..)
        | ast::Expr::FloatLiteral(..)
        | ast::Expr::ImagLiteral(..)
        | ast::Expr::BoolLiteral(..)
        | ast::Expr::BitstringLiteral(..)
        | ast::Expr::TimingLiteral(..) => 27,
    }
}

fn needs_expr_parens(expr: &ast::Expr<'_>, parent: ExprParent) -> bool {
    let child_precedence = expr_precedence(expr);

    match parent {
        ExprParent::Unary { precedence } | ExprParent::Postfix { precedence } => {
            child_precedence < precedence
        }
        ExprParent::Binary { precedence, side } => {
            child_precedence < precedence
                || (child_precedence == precedence
                    && matches!(
                        side,
                        ExprSide::Right if !matches!(expr, ast::Expr::BinOp { op: ast::BinOp::Pow, .. })
                    ))
                || (child_precedence == precedence
                    && matches!(side, ExprSide::Left)
                    && matches!(
                        expr,
                        ast::Expr::BinOp {
                            op: ast::BinOp::Pow,
                            ..
                        }
                    ))
        }
    }
}

fn bin_op_precedence(op: ast::BinOp) -> u8 {
    match op {
        ast::BinOp::LogOr => 2,
        ast::BinOp::LogAnd => 4,
        ast::BinOp::BitOr => 6,
        ast::BinOp::BitXor => 8,
        ast::BinOp::BitAnd => 10,
        ast::BinOp::Eq | ast::BinOp::Neq => 12,
        ast::BinOp::Lt | ast::BinOp::Gt | ast::BinOp::Lte | ast::BinOp::Gte => 14,
        ast::BinOp::Shl | ast::BinOp::Shr => 16,
        ast::BinOp::Add | ast::BinOp::Sub => 18,
        ast::BinOp::Mul | ast::BinOp::Div | ast::BinOp::Mod => 20,
        ast::BinOp::Pow => 22,
    }
}

fn bin_op_text(op: ast::BinOp) -> &'static str {
    match op {
        ast::BinOp::Add => "+",
        ast::BinOp::Sub => "-",
        ast::BinOp::Mul => "*",
        ast::BinOp::Div => "/",
        ast::BinOp::Mod => "%",
        ast::BinOp::Pow => "**",
        ast::BinOp::BitAnd => "&",
        ast::BinOp::BitOr => "|",
        ast::BinOp::BitXor => "^",
        ast::BinOp::Shl => "<<",
        ast::BinOp::Shr => ">>",
        ast::BinOp::LogAnd => "&&",
        ast::BinOp::LogOr => "||",
        ast::BinOp::Eq => "==",
        ast::BinOp::Neq => "!=",
        ast::BinOp::Lt => "<",
        ast::BinOp::Gt => ">",
        ast::BinOp::Lte => "<=",
        ast::BinOp::Gte => ">=",
    }
}

fn un_op_text(op: ast::UnOp) -> &'static str {
    match op {
        ast::UnOp::Neg => "-",
        ast::UnOp::BitNot => "~",
        ast::UnOp::LogNot => "!",
    }
}

#[derive(Clone, Copy)]
enum ExprParent {
    Binary { precedence: u8, side: ExprSide },
    Unary { precedence: u8 },
    Postfix { precedence: u8 },
}

#[derive(Clone, Copy)]
enum ExprSide {
    Left,
    Right,
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
                "}"
            )
        );
    }

    #[test]
    fn compact_inlines_simple_scope() {
        let source = "if (a) { x q; y q; }";
        let formatted = format(source, Config::compact()).unwrap();
        assert_eq!(formatted, "if (a) { x q; y q; }");
    }

    #[test]
    fn collects_and_queries_comments() {
        let source = "/* head */\ninclude \"stdgates.inc\"; // tail\n";
        let context = Context::from_source(source).unwrap();

        assert_eq!(context.comments().len(), 2);
        assert_eq!(context.comments()[0].kind, CommentKind::Block);
        assert_eq!(context.comments()[1].kind, CommentKind::Line);

        let include_span = 11..35;
        let comments_in = context.comments_in(&include_span).count();
        let comments_before = context.comments_before(include_span.start).count();
        let comments_after = context.comments_after(include_span.end).count();

        assert_eq!(comments_in, 0);
        assert_eq!(comments_before, 1);
        assert_eq!(comments_after, 1);
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
