pub mod ast;

use ast::*;
pub use oqi_lex::{Error, Result};
use oqi_lex::{Lexer, Token};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse(source: &str) -> Result<Program<'_>> {
    Ok(parse_with_comments(source)?.0)
}

/// Parse `source`, also returning its comments in source order — including
/// those inside OpenPulse calibration bodies.
pub fn parse_with_comments(source: &str) -> Result<(Program<'_>, Vec<Comment<'_>>)> {
    let mut parser = Parser::new(Lexer::new(source), 0)?;
    let program = parser.parse_program()?;
    parser.comments.sort_by_key(|comment| comment.span.start);
    Ok((program, parser.comments))
}

// ---------------------------------------------------------------------------
// Token classification
// ---------------------------------------------------------------------------

type DecomposedGateHead<'a> = (Ident<'a>, Option<Vec<Expr<'a>>>, Option<Box<Expr<'a>>>);

fn bin_op(token: Token) -> Option<BinOp> {
    Some(match token {
        Token::Plus => BinOp::Add,
        Token::Minus => BinOp::Sub,
        Token::Asterisk => BinOp::Mul,
        Token::Slash => BinOp::Div,
        Token::Percent => BinOp::Mod,
        Token::DoubleAsterisk => BinOp::Pow,
        Token::Ampersand => BinOp::BitAnd,
        Token::Pipe => BinOp::BitOr,
        Token::Caret => BinOp::BitXor,
        Token::DoubleAmpersand => BinOp::LogAnd,
        Token::DoublePipe => BinOp::LogOr,
        Token::DoubleEquals => BinOp::Eq,
        Token::ExclamationEquals => BinOp::Neq,
        Token::LessThan => BinOp::Lt,
        Token::GreaterThan => BinOp::Gt,
        Token::LessThanEquals => BinOp::Lte,
        Token::GreaterThanEquals => BinOp::Gte,
        Token::DoubleLess => BinOp::Shl,
        Token::DoubleGreater => BinOp::Shr,
        _ => return None,
    })
}

fn un_op(token: Token) -> Option<UnOp> {
    Some(match token {
        Token::Minus => UnOp::Neg,
        Token::Tilde => UnOp::BitNot,
        Token::ExclamationPoint => UnOp::LogNot,
        _ => return None,
    })
}

fn assign_op(token: Token) -> Option<AssignOp> {
    Some(match token {
        Token::Equals => AssignOp::Assign,
        Token::PlusEquals => AssignOp::AddAssign,
        Token::MinusEquals => AssignOp::SubAssign,
        Token::AsteriskEquals => AssignOp::MulAssign,
        Token::SlashEquals => AssignOp::DivAssign,
        Token::AmpersandEquals => AssignOp::BitAndAssign,
        Token::PipeEquals => AssignOp::BitOrAssign,
        Token::CaretEquals => AssignOp::BitXorAssign,
        Token::LeftShiftEquals => AssignOp::LeftShiftAssign,
        Token::RightShiftEquals => AssignOp::RightShiftAssign,
        Token::PercentEquals => AssignOp::ModAssign,
        Token::DoubleAsteriskEquals => AssignOp::PowAssign,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: Vec<(Token<'a>, Span)>,
    comments: Vec<Comment<'a>>,
    pos: usize,
    /// End of the last token, where an error at the end of input points.
    end: usize,
    /// Set when the program has declared `defcalgrammar "openpulse";`.
    /// Causes `cal`/`defcal` bodies to be re-lexed and re-parsed against the
    /// OpenPulse grammar instead of being kept as raw `&str` slices.
    openpulse: bool,
}

impl<'a> Parser<'a> {
    /// Lex everything up front, setting the comments aside. `offset` is the
    /// position of the lexed text within the document, so that spans (and
    /// error spans) refer to the document.
    fn new(lexer: Lexer<'a>, offset: usize) -> Result<Self> {
        let mut tokens = Vec::new();
        let mut comments = Vec::new();
        for item in lexer {
            let (token, span) = item.map_err(|e| Error::new(e.span.offset(offset), e.message))?;
            let span = span.offset(offset);
            match token {
                Token::LineComment(text) | Token::BlockComment(text) => {
                    comments.push(Comment { text, span })
                }
                _ => tokens.push((token, span)),
            }
        }
        let end = tokens.last().map_or(0, |(_, span)| span.end);
        Ok(Self {
            tokens,
            comments,
            pos: 0,
            end,
            openpulse: false,
        })
    }

    // ------------------------------------------------------------------
    // Token helpers
    // ------------------------------------------------------------------

    fn peek(&self) -> Option<Token<'a>> {
        self.tokens.get(self.pos).map(|&(token, _)| token)
    }

    fn peek_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map_or(Span::new(self.end, self.end), |&(_, span)| span)
    }

    fn advance(&mut self) -> (Token<'a>, Span) {
        let item = self.tokens[self.pos];
        self.pos += 1;
        item
    }

    fn prev_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].1
        } else {
            Default::default()
        }
    }

    fn error(&self, msg: impl Into<String>) -> Error {
        Error::new(self.peek_span(), msg)
    }

    /// Whether the next token is `token` (a token without a payload).
    fn at(&self, token: Token<'a>) -> bool {
        self.peek() == Some(token)
    }

    /// Whether the next token is one of `tokens`, or the input has ended.
    fn at_any(&self, tokens: &[Token<'a>]) -> bool {
        self.peek().is_none_or(|token| tokens.contains(&token))
    }

    /// Consume `token` if it is next.
    fn eat(&mut self, token: Token<'a>) -> Option<Span> {
        self.at(token).then(|| self.advance().1)
    }

    /// Consume `token`, or fail with "expected `what`".
    fn expect(&mut self, token: Token<'a>, what: &str) -> Result<Span> {
        self.eat(token)
            .ok_or_else(|| self.error(format!("expected {what}")))
    }

    fn semi(&mut self) -> Result<Span> {
        self.expect(Token::Semicolon, "';'")
    }

    fn expect_ident(&mut self) -> Result<Ident<'a>> {
        match self.peek() {
            Some(Token::Identifier(name)) => Ok(Ident {
                name,
                span: self.advance().1,
            }),
            _ => Err(self.error("expected identifier")),
        }
    }

    fn expect_string(&mut self) -> Result<&'a str> {
        match self.peek() {
            Some(Token::StringLiteral(string)) => {
                self.advance();
                Ok(string)
            }
            _ => Err(self.error("expected string literal")),
        }
    }

    fn peek_is_type_keyword(&self) -> bool {
        matches!(
            self.peek(),
            Some(
                Token::Bit
                    | Token::Int
                    | Token::Uint
                    | Token::Float
                    | Token::Angle
                    | Token::Bool
                    | Token::Duration
                    | Token::Stretch
                    | Token::Complex
                    | Token::Array
            )
        )
    }

    fn peek_is_gate_operand(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Identifier(_) | Token::HardwareQubit(_))
        )
    }

    // ------------------------------------------------------------------
    // Lists
    // ------------------------------------------------------------------

    /// `item (COMMA item)* COMMA?` — every list in the grammar has this shape.
    /// `close` are the tokens that may follow the list, which is how a
    /// trailing comma is told apart from a separator.
    fn comma_list<T>(
        &mut self,
        close: &[Token<'a>],
        mut item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut items = vec![item(self)?];
        while self.eat(Token::Comma).is_some() && !self.at_any(close) {
            items.push(item(self)?);
        }
        Ok(items)
    }

    /// An optional [`comma_list`](Self::comma_list): empty when one of
    /// `close` is next.
    fn opt_comma_list<T>(
        &mut self,
        close: &[Token<'a>],
        item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        if self.at_any(close) {
            Ok(vec![])
        } else {
            self.comma_list(close, item)
        }
    }

    /// `( list? )`, returning the items and the span of the closing paren.
    fn paren_list<T>(
        &mut self,
        item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<(Vec<T>, Span)> {
        self.expect(Token::LParen, "'('")?;
        let items = self.opt_comma_list(&[Token::RParen], item)?;
        let end = self.expect(Token::RParen, "')'")?;
        Ok((items, end))
    }

    /// A [`paren_list`](Self::paren_list) that may be left out entirely.
    fn opt_paren_list<T>(
        &mut self,
        item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Option<Vec<T>>> {
        if self.at(Token::LParen) {
            Ok(Some(self.paren_list(item)?.0))
        } else {
            Ok(None)
        }
    }

    /// `expressionList` followed by `close`.
    fn expr_list(&mut self, close: Token<'a>) -> Result<Vec<Expr<'a>>> {
        self.comma_list(&[close], Self::expr)
    }

    // ------------------------------------------------------------------
    // Program structure
    // ------------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Program<'a>> {
        let version = if self.at(Token::OpenQasm) {
            Some(self.parse_version()?)
        } else {
            None
        };
        let mut body = Vec::new();
        while self.peek().is_some() {
            body.push(self.parse_stmt_or_scope()?);
        }
        Ok(Program {
            version,
            body,
            span: Span::new(0, self.end),
        })
    }

    fn parse_version(&mut self) -> Result<Version<'a>> {
        let start = self.advance().1; // eat OPENQASM
        let Some(Token::VersionSpecifier(specifier)) = self.peek() else {
            return Err(self.error("expected version specifier"));
        };
        self.advance();
        let end = self.semi()?;
        Ok(Version {
            specifier,
            span: start.to(end),
        })
    }

    fn parse_stmt_or_scope(&mut self) -> Result<StmtOrScope<'a>> {
        if self.at(Token::LBrace) {
            Ok(StmtOrScope::Scope(self.parse_scope()?))
        } else {
            Ok(StmtOrScope::Stmt(Box::new(self.parse_statement()?)))
        }
    }

    fn parse_scope(&mut self) -> Result<Scope<'a>> {
        let start = self.expect(Token::LBrace, "'{'")?;
        let mut body = Vec::new();
        while !self.at_any(&[Token::RBrace]) {
            body.push(self.parse_stmt_or_scope()?);
        }
        let end = self.expect(Token::RBrace, "'}'")?;
        Ok(Scope {
            body,
            span: start.to(end),
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt<'a>> {
        let start = self.peek_span();
        // A pragma takes no annotations.
        let (annotations, kind) = if self.at(Token::Pragma) {
            (vec![], self.parse_pragma()?)
        } else {
            (self.parse_annotations(), self.parse_stmt_kind()?)
        };
        Ok(Stmt {
            annotations,
            kind,
            span: start.to(self.prev_span()),
        })
    }

    fn parse_annotations(&mut self) -> Vec<Annotation<'a>> {
        let mut annotations = Vec::new();
        while let Some(Token::AnnotationKeyword(keyword)) = self.peek() {
            let start = self.advance().1;
            let content = self.remaining_line();
            annotations.push(Annotation {
                keyword,
                content,
                span: start.to(self.prev_span()),
            });
        }
        annotations
    }

    /// The rest of a pragma or annotation line, if it has any content.
    fn remaining_line(&mut self) -> Option<&'a str> {
        let Some(Token::RemainingLineContent(content)) = self.peek() else {
            return None;
        };
        self.advance();
        Some(content)
    }

    // ------------------------------------------------------------------
    // Statement dispatch
    // ------------------------------------------------------------------

    fn parse_stmt_kind(&mut self) -> Result<StmtKind<'a>> {
        let Some(tok) = self.peek() else {
            return Err(self.error("expected statement"));
        };
        match tok {
            Token::Include => {
                self.advance();
                let path = self.expect_string()?;
                self.semi()?;
                Ok(StmtKind::Include(path))
            }
            Token::DefCalGrammar => self.parse_calibration_grammar(),
            Token::Break | Token::Continue | Token::End => {
                self.advance();
                self.semi()?;
                Ok(match tok {
                    Token::Break => StmtKind::Break,
                    Token::Continue => StmtKind::Continue,
                    _ => StmtKind::End,
                })
            }
            Token::For => self.parse_for(),
            Token::If => self.parse_if(),
            Token::Return => self.parse_return(),
            Token::While => self.parse_while(),
            Token::Switch => self.parse_switch(),
            Token::Barrier => {
                self.advance();
                Ok(StmtKind::Barrier(self.opt_gate_operands()?))
            }
            Token::Box => self.parse_box(),
            Token::Delay => {
                self.advance();
                Ok(StmtKind::Delay {
                    designator: self.parse_designator()?,
                    operands: self.opt_gate_operands()?,
                })
            }
            Token::Nop => {
                self.advance();
                Ok(StmtKind::Nop(self.opt_gate_operands()?))
            }
            Token::Reset => {
                self.advance();
                let operand = self.parse_gate_operand()?;
                self.semi()?;
                Ok(StmtKind::Reset(operand))
            }
            Token::Let => self.parse_alias(),
            Token::Const => self.parse_const_decl(),
            Token::Input | Token::Output => self.parse_io_decl(),
            Token::Creg | Token::Qreg => self.parse_old_style_decl(),
            Token::Qubit => {
                let ty = self.parse_qubit_type()?;
                let name = self.expect_ident()?;
                self.semi()?;
                Ok(StmtKind::QuantumDecl { ty, name })
            }
            Token::Def => self.parse_def(),
            Token::Extern => self.parse_extern(),
            Token::Gate => self.parse_gate(),
            Token::Cal => {
                self.advance();
                Ok(StmtKind::Cal(self.parse_cal_body()?))
            }
            Token::DefCal => self.parse_defcal(),
            Token::Measure => self.parse_measure_arrow(),
            Token::Gphase => self.parse_gphase_call(vec![]),
            // Type keywords → classical declaration
            Token::Bit
            | Token::Int
            | Token::Uint
            | Token::Float
            | Token::Angle
            | Token::Bool
            | Token::Duration
            | Token::Stretch
            | Token::Complex
            | Token::Array
            | Token::Waveform
            | Token::Port
            | Token::Frame => self.parse_classical_decl(),
            // Gate modifiers → gate call
            Token::Inv | Token::Pow | Token::Ctrl | Token::Negctrl => {
                self.parse_modified_gate_call()
            }
            // Identifier → assignment, gate call, expression, or quantum call
            Token::Identifier(_) => self.parse_ambiguous_stmt(),
            // Fallback: expression statement
            _ => self.parse_expression_stmt(),
        }
    }

    // ------------------------------------------------------------------
    // Simple statements
    // ------------------------------------------------------------------

    fn parse_pragma(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat pragma
        match self.remaining_line() {
            Some(content) => Ok(StmtKind::Pragma(content)),
            None => Err(self.error("expected pragma content")),
        }
    }

    fn parse_calibration_grammar(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat defcalgrammar
        let path = self.expect_string()?;
        self.semi()?;
        // The string literal includes the surrounding quotes.
        if matches!(path, "\"openpulse\"" | "'openpulse'") {
            self.openpulse = true;
        }
        Ok(StmtKind::CalibrationGrammar(path))
    }

    // ------------------------------------------------------------------
    // Control flow
    // ------------------------------------------------------------------

    fn parse_for(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat for
        let ty = self.parse_scalar_type()?;
        let var = self.expect_ident()?;
        self.expect(Token::In, "'in'")?;

        let iterable = match self.peek() {
            Some(Token::LBrace) => {
                let (exprs, span) = self.parse_set_expression()?;
                ForIterable::Set(exprs, span)
            }
            Some(Token::LBracket) => {
                let start = self.advance().1;
                let range = self.parse_range_expression()?;
                let end = self.expect(Token::RBracket, "']'")?;
                ForIterable::Range(range, start.to(end))
            }
            _ => ForIterable::Expr(self.expr()?),
        };

        let body = self.parse_stmt_or_scope()?;
        Ok(StmtKind::For {
            ty,
            var,
            iterable,
            body: Box::new(body),
        })
    }

    fn parse_if(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat if
        let (condition, _) = self.parenthesized()?;
        let then_body = self.parse_stmt_or_scope()?;
        let else_body = if self.eat(Token::Else).is_some() {
            Some(Box::new(self.parse_stmt_or_scope()?))
        } else {
            None
        };
        Ok(StmtKind::If {
            condition,
            then_body: Box::new(then_body),
            else_body,
        })
    }

    fn parse_return(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat return
        if self.eat(Token::Semicolon).is_some() {
            return Ok(StmtKind::Return(None));
        }
        let value = self.parse_expr_or_measure()?;
        self.semi()?;
        Ok(StmtKind::Return(Some(value)))
    }

    fn parse_while(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat while
        let (condition, _) = self.parenthesized()?;
        let body = self.parse_stmt_or_scope()?;
        Ok(StmtKind::While {
            condition,
            body: Box::new(body),
        })
    }

    fn parse_switch(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat switch
        let (target, _) = self.parenthesized()?;
        self.expect(Token::LBrace, "'{'")?;
        let mut cases = Vec::new();
        while !self.at_any(&[Token::RBrace]) {
            cases.push(self.parse_switch_case()?);
        }
        self.expect(Token::RBrace, "'}'")?;
        Ok(StmtKind::Switch { target, cases })
    }

    fn parse_switch_case(&mut self) -> Result<SwitchCase<'a>> {
        if self.eat(Token::Default).is_some() {
            Ok(SwitchCase::Default(self.parse_scope()?))
        } else if self.eat(Token::Case).is_some() {
            let exprs = self.expr_list(Token::LBrace)?;
            Ok(SwitchCase::Case(exprs, self.parse_scope()?))
        } else {
            Err(self.error("expected 'case' or 'default'"))
        }
    }

    // ------------------------------------------------------------------
    // Quantum statements
    // ------------------------------------------------------------------

    /// `gateOperandList? SEMICOLON`
    fn opt_gate_operands(&mut self) -> Result<Vec<GateOperand<'a>>> {
        let operands = self.opt_comma_list(&[Token::Semicolon], Self::parse_gate_operand)?;
        self.semi()?;
        Ok(operands)
    }

    fn parse_box(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat box
        let designator = self.opt_designator()?;
        let body = self.parse_scope()?;
        Ok(StmtKind::Box { designator, body })
    }

    /// `MEASURE gateOperand`
    fn parse_measure(&mut self) -> Result<MeasureExpr<'a>> {
        let start = self.advance().1; // eat measure
        let operand = self.parse_gate_operand()?;
        let span = start.to(operand.span());
        Ok(MeasureExpr::Measure { operand, span })
    }

    fn parse_measure_arrow(&mut self) -> Result<StmtKind<'a>> {
        let measure = self.parse_measure()?;
        let target = if self.eat(Token::Arrow).is_some() {
            Some(self.parse_indexed_identifier()?)
        } else {
            None
        };
        self.semi()?;
        Ok(StmtKind::MeasureArrow { measure, target })
    }

    // ------------------------------------------------------------------
    // Gate calls (including ambiguous identifier statements)
    // ------------------------------------------------------------------

    fn parse_ambiguous_stmt(&mut self) -> Result<StmtKind<'a>> {
        let expr = self.expr()?;

        if self.eat(Token::Semicolon).is_some() {
            return Ok(StmtKind::Expr(expr));
        }

        if let Some(op) = self.peek().and_then(assign_op) {
            let target = Self::expr_to_indexed_ident(expr)?;
            self.advance();
            let value = self.parse_expr_or_measure()?;
            self.semi()?;
            return Ok(StmtKind::Assignment { target, op, value });
        }

        if !self.peek_is_gate_operand() {
            return Err(self.error("expected ';', '=', or gate operand after expression"));
        }

        // Gate call or quantum call expression
        let (name, args, designator) = Self::decompose_gate_head(expr)?;
        let operands =
            self.comma_list(&[Token::Semicolon, Token::Arrow], Self::parse_gate_operand)?;
        if !self.at(Token::Arrow) {
            self.semi()?;
            return Ok(StmtKind::GateCall {
                modifiers: vec![],
                name,
                args,
                designator,
                operands,
            });
        }
        if designator.is_some() {
            return Err(self.error("duration designator is not supported on measure calls"));
        }
        self.advance(); // eat ->
        let target = self.parse_indexed_identifier()?;
        self.semi()?;
        let span = name.span.to(target.span);
        Ok(StmtKind::MeasureArrow {
            measure: MeasureExpr::QuantumCall {
                name,
                args: args.unwrap_or_default(),
                operands,
                span,
            },
            target: Some(target),
        })
    }

    fn parse_modified_gate_call(&mut self) -> Result<StmtKind<'a>> {
        let mut modifiers = Vec::new();
        while let Some(modifier) = self.parse_gate_modifier()? {
            modifiers.push(modifier);
        }

        if self.at(Token::Gphase) {
            return self.parse_gphase_call(modifiers);
        }

        let (name, args, designator) = Self::decompose_gate_head(self.expr()?)?;
        let operands = self.comma_list(&[Token::Semicolon], Self::parse_gate_operand)?;
        self.semi()?;
        Ok(StmtKind::GateCall {
            modifiers,
            name,
            args,
            designator,
            operands,
        })
    }

    fn parse_gphase_call(&mut self, modifiers: Vec<GateModifier<'a>>) -> Result<StmtKind<'a>> {
        let name = Ident {
            name: "gphase",
            span: self.advance().1,
        };
        let args = self.opt_paren_list(Self::expr)?;
        let designator = self.opt_designator()?.map(Box::new);
        let operands = self.opt_gate_operands()?;
        Ok(StmtKind::GateCall {
            modifiers,
            name,
            args,
            designator,
            operands,
        })
    }

    /// A `gateModifier`, if one is next.
    fn parse_gate_modifier(&mut self) -> Result<Option<GateModifier<'a>>> {
        let Some(tok @ (Token::Inv | Token::Pow | Token::Ctrl | Token::Negctrl)) = self.peek()
        else {
            return Ok(None);
        };
        let start = self.advance().1;
        // `pow` requires an argument; the control count is optional.
        let arg = if tok == Token::Pow || (tok != Token::Inv && self.at(Token::LParen)) {
            Some(self.parenthesized()?.0)
        } else {
            None
        };
        let span = start.to(self.expect(Token::At, "'@'")?);
        Ok(Some(match (tok, arg) {
            (Token::Pow, Some(exponent)) => GateModifier::Pow(exponent, span),
            (Token::Ctrl, count) => GateModifier::Ctrl(count, span),
            (Token::Negctrl, count) => GateModifier::NegCtrl(count, span),
            _ => GateModifier::Inv(span),
        }))
    }

    // ------------------------------------------------------------------
    // Declaration statements
    // ------------------------------------------------------------------

    fn parse_alias(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat let
        let name = self.expect_ident()?;
        self.expect(Token::Equals, "'='")?;
        // aliasExpression: expression (DOUBLE_PLUS expression)*
        let mut value = vec![self.expr()?];
        while self.eat(Token::DoublePlus).is_some() {
            value.push(self.expr()?);
        }
        self.semi()?;
        Ok(StmtKind::Alias { name, value })
    }

    fn parse_classical_decl(&mut self) -> Result<StmtKind<'a>> {
        // Ambiguity: `float[64] x = ...` is a declaration, but `float[64](x)` is a cast expression.
        // Save position so we can backtrack if no identifier follows the type.
        let saved_pos = self.pos;
        let ty = self.parse_type_expr()?;
        if !matches!(self.peek(), Some(Token::Identifier(_))) {
            // Not a declaration — backtrack and parse as expression statement
            self.pos = saved_pos;
            return self.parse_expression_stmt();
        }
        let name = self.expect_ident()?;
        let init = if self.eat(Token::Equals).is_some() {
            Some(self.parse_expr_or_measure()?)
        } else {
            None
        };
        self.semi()?;
        Ok(StmtKind::ClassicalDecl { ty, name, init })
    }

    fn parse_const_decl(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat const
        let ty = self.parse_type_expr()?;
        let name = self.expect_ident()?;
        self.expect(Token::Equals, "'='")?;
        let init = self.parse_expr_or_measure()?;
        self.semi()?;
        Ok(StmtKind::ConstDecl { ty, name, init })
    }

    fn parse_io_decl(&mut self) -> Result<StmtKind<'a>> {
        let dir = if self.advance().0 == Token::Input {
            IoDir::Input
        } else {
            IoDir::Output
        };
        let ty = self.parse_type_expr()?;
        let name = self.expect_ident()?;
        self.semi()?;
        Ok(StmtKind::IoDecl { dir, ty, name })
    }

    fn parse_old_style_decl(&mut self) -> Result<StmtKind<'a>> {
        let keyword = if self.advance().0 == Token::Creg {
            OldStyleKind::Creg
        } else {
            OldStyleKind::Qreg
        };
        let name = self.expect_ident()?;
        let designator = self.opt_designator()?.map(Box::new);
        self.semi()?;
        Ok(StmtKind::OldStyleDecl {
            keyword,
            name,
            designator,
        })
    }

    // ------------------------------------------------------------------
    // Higher-order definitions
    // ------------------------------------------------------------------

    /// `returnSignature?`
    fn opt_return_signature(&mut self) -> Result<Option<ScalarType<'a>>> {
        if self.eat(Token::Arrow).is_some() {
            Ok(Some(self.parse_scalar_type()?))
        } else {
            Ok(None)
        }
    }

    fn parse_def(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat def
        let name = self.expect_ident()?;
        let (params, _) = self.paren_list(Self::parse_arg_def)?;
        let return_ty = self.opt_return_signature()?;
        let body = self.parse_scope()?;
        Ok(StmtKind::Def {
            name,
            params,
            return_ty,
            body,
        })
    }

    fn parse_extern(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat extern

        // OpenPulse-only forms: `extern frame Identifier ;` and
        // `extern port Identifier ;`. The Frame/Port tokens are only emitted
        // by the OpenPulse lexer, so checking unconditionally is safe.
        if let Some(tok @ (Token::Frame | Token::Port)) = self.peek() {
            self.advance();
            let name = self.expect_ident()?;
            self.semi()?;
            return Ok(if tok == Token::Frame {
                StmtKind::ExternFrame { name }
            } else {
                StmtKind::ExternPort { name }
            });
        }

        let name = self.expect_ident()?;
        let (params, _) = self.paren_list(Self::parse_extern_arg)?;
        let return_ty = self.opt_return_signature()?;
        self.semi()?;
        Ok(StmtKind::Extern {
            name,
            params,
            return_ty,
        })
    }

    fn parse_gate(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat gate
        let name = self.expect_ident()?;
        let params = self.opt_paren_list(Self::expect_ident)?.unwrap_or_default();
        let qubits = self.comma_list(&[Token::LBrace], Self::expect_ident)?;
        let body = self.parse_scope()?;
        Ok(StmtKind::Gate {
            name,
            params,
            qubits,
            body,
        })
    }

    // ------------------------------------------------------------------
    // Cal / Defcal
    // ------------------------------------------------------------------

    /// `LBRACE CalibrationBlock? RBRACE` — parsed as OpenPulse or kept raw,
    /// per the program's `defcalgrammar` selection.
    fn parse_cal_body(&mut self) -> Result<CalBody<'a>> {
        self.expect(Token::LBrace, "'{'")?;
        let body = match self.peek() {
            Some(Token::CalibrationBlock(body)) => {
                let offset = self.advance().1.start;
                if self.openpulse {
                    CalBody::OpenPulse(self.parse_openpulse_block(body, offset)?)
                } else {
                    CalBody::Raw(Some(body))
                }
            }
            _ => CalBody::Raw(None),
        };
        self.expect(Token::RBrace, "'}'")?;
        Ok(body)
    }

    /// Re-lex `body` (a calibration block's contents, at `offset` in the
    /// document) with the OpenPulse lexer and parse it as a sequence of
    /// OpenPulse statements.
    fn parse_openpulse_block(
        &mut self,
        body: &'a str,
        offset: usize,
    ) -> Result<Vec<StmtOrScope<'a>>> {
        let mut sub = Parser::new(Lexer::new(body).openpulse(), offset)?;
        let mut stmts = Vec::new();
        while sub.peek().is_some() {
            let item = sub.parse_stmt_or_scope()?;
            // Directly inside a calibration block only the grammar's
            // `openpulseStatement` subset is allowed. The lexer has already
            // demoted `cal` and `defcal` to identifiers, so those cannot
            // appear; the rest of the exclusions are rejected here.
            let excluded = match &item {
                StmtOrScope::Scope(_) => None,
                StmtOrScope::Stmt(stmt) => match stmt.kind {
                    StmtKind::Gate { .. } => Some("`gate` declarations are"),
                    StmtKind::Nop(_) => Some("`nop` is"),
                    StmtKind::CalibrationGrammar(_) => Some("`defcalgrammar` is"),
                    StmtKind::OldStyleDecl { .. } => Some("`creg` and `qreg` declarations are"),
                    // (A measurement can still be assigned: `c = measure $0;`.)
                    StmtKind::MeasureArrow { .. } => Some("measurement statements are"),
                    _ => None,
                }
                .map(|what| (what, stmt.span)),
            };
            if let Some((what, span)) = excluded {
                let message = format!("{what} not allowed inside an OpenPulse calibration block");
                return Err(Error::new(span, message));
            }
            stmts.push(item);
        }
        self.comments.append(&mut sub.comments);
        Ok(stmts)
    }

    fn parse_defcal(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat defcal

        let target = match self.peek() {
            Some(Token::Measure) => DefcalTarget::Measure(self.advance().1),
            Some(Token::Reset) => DefcalTarget::Reset(self.advance().1),
            Some(Token::Delay) => DefcalTarget::Delay(self.advance().1),
            _ => DefcalTarget::Ident(self.expect_ident()?),
        };
        let args = self
            .opt_paren_list(Self::parse_defcal_arg_def)?
            .unwrap_or_default();
        let operands =
            self.opt_comma_list(&[Token::Arrow, Token::LBrace], Self::parse_defcal_operand)?;
        let return_ty = self.opt_return_signature()?;
        let body = self.parse_cal_body()?;

        Ok(StmtKind::Defcal {
            target,
            args,
            operands,
            return_ty,
            body,
        })
    }

    // ------------------------------------------------------------------
    // Expression statement
    // ------------------------------------------------------------------

    fn parse_expression_stmt(&mut self) -> Result<StmtKind<'a>> {
        let expr = self.expr()?;
        self.semi()?;
        Ok(StmtKind::Expr(expr))
    }

    // ------------------------------------------------------------------
    // Expression and measure helpers
    // ------------------------------------------------------------------

    fn parse_expr_or_measure(&mut self) -> Result<ExprOrMeasure<'a>> {
        if self.at(Token::Measure) {
            return Ok(ExprOrMeasure::Measure(self.parse_measure()?));
        }

        let expr = self.expr()?;
        if !self.peek_is_gate_operand() {
            return Ok(ExprOrMeasure::Expr(expr));
        }

        // Quantum call: an expression followed by gate operands
        let (name, args, designator) = Self::decompose_gate_head(expr)?;
        if designator.is_some() {
            return Err(self.error("duration designator is not supported on measure calls"));
        }
        let operands = self.comma_list(&[Token::Semicolon], Self::parse_gate_operand)?;
        let span = name
            .span
            .to(operands.last().map_or(name.span, GateOperand::span));
        Ok(ExprOrMeasure::Measure(MeasureExpr::QuantumCall {
            name,
            args: args.unwrap_or_default(),
            operands,
            span,
        }))
    }

    fn parse_array_literal(&mut self) -> Result<ArrayLiteral<'a>> {
        let start = self.expect(Token::LBrace, "'{'")?;
        let items = self.opt_comma_list(&[Token::RBrace], Self::expr)?;
        let end = self.expect(Token::RBrace, "'}'")?;
        Ok(ArrayLiteral {
            items,
            span: start.to(end),
        })
    }

    /// `( expression )`, returning the expression and the span of the whole.
    fn parenthesized(&mut self) -> Result<(Expr<'a>, Span)> {
        let start = self.expect(Token::LParen, "'('")?;
        let expr = self.expr()?;
        let end = self.expect(Token::RParen, "')'")?;
        Ok((expr, start.to(end)))
    }

    // ------------------------------------------------------------------
    // Pratt expression parser
    // ------------------------------------------------------------------

    fn expr(&mut self) -> Result<Expr<'a>> {
        self.parse_expr(0)
    }

    /// Parse an expression whose operators all bind at least as tightly as
    /// `min_precedence`.
    fn parse_expr(&mut self, min_precedence: u8) -> Result<Expr<'a>> {
        let mut lhs = self.parse_primary_expr()?;

        loop {
            // Postfix: index operator
            if self.at(Token::LBracket) {
                if IndexOp::PRECEDENCE < min_precedence {
                    break;
                }
                let index = self.parse_index_operator()?;
                let span = lhs.span().to(index.span);
                lhs = Expr::Index {
                    expr: Box::new(lhs),
                    index,
                    span,
                };
                continue;
            }

            // Infix binary operators
            let Some(op) = self.peek().and_then(bin_op) else {
                break;
            };
            let precedence = op.precedence();
            if precedence < min_precedence {
                break;
            }
            self.advance();
            // The right operand of a left-associative operator must bind
            // tighter than the operator itself.
            let rhs = self.parse_expr(if op.is_right_associative() {
                precedence - 1
            } else {
                precedence + 1
            })?;
            let span = lhs.span().to(rhs.span());
            lhs = Expr::BinOp {
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
                span,
            };
        }

        Ok(lhs)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr<'a>> {
        let Some(tok) = self.peek() else {
            return Err(self.error("expected expression"));
        };
        let span = self.peek_span();

        // Prefix unary operators
        if let Some(op) = un_op(tok) {
            self.advance();
            let operand = self.parse_expr(UnOp::PRECEDENCE)?;
            return Ok(Expr::UnaryOp {
                op,
                span: span.to(operand.span()),
                operand: Box::new(operand),
            });
        }

        // Type cast: (scalarType | arrayType) LPAREN expression RPAREN
        if self.peek_is_type_keyword() {
            let ty = self.parse_type_expr()?;
            let (operand, parens) = self.parenthesized()?;
            return Ok(Expr::Cast {
                ty: Box::new(ty),
                operand: Box::new(operand),
                span: span.to(parens),
            });
        }

        let literal = match tok {
            Token::LParen => {
                let (inner, span) = self.parenthesized()?;
                return Ok(Expr::Paren(Box::new(inner), span));
            }
            Token::LBrace => return Ok(Expr::ArrayLiteral(self.parse_array_literal()?)),
            Token::Durationof => {
                self.advance();
                self.expect(Token::LParen, "'('")?;
                let scope = self.parse_scope()?;
                let end = self.expect(Token::RParen, "')'")?;
                return Ok(Expr::DurationOf {
                    scope,
                    span: span.to(end),
                });
            }
            // Identifier or call: Identifier LPAREN expressionList? RPAREN
            Token::Identifier(name) => {
                self.advance();
                let name = Ident { name, span };
                if !self.at(Token::LParen) {
                    return Ok(Expr::Ident(name));
                }
                let (args, end) = self.paren_list(Self::expr)?;
                return Ok(Expr::Call {
                    name,
                    args,
                    span: span.to(end),
                });
            }
            Token::DecimalIntegerLiteral(s) => Expr::IntLiteral(s, IntEncoding::Decimal, span),
            Token::BinaryIntegerLiteral(s) => Expr::IntLiteral(s, IntEncoding::Binary, span),
            Token::OctalIntegerLiteral(s) => Expr::IntLiteral(s, IntEncoding::Octal, span),
            Token::HexIntegerLiteral(s) => Expr::IntLiteral(s, IntEncoding::Hex, span),
            Token::FloatLiteral(s) => Expr::FloatLiteral(s, span),
            Token::ImaginaryLiteral(s) => Expr::ImagLiteral(s, span),
            Token::True => Expr::BoolLiteral(true, span),
            Token::False => Expr::BoolLiteral(false, span),
            Token::BitstringLiteral(s) => Expr::BitstringLiteral(s, span),
            Token::TimingLiteral(s) => Expr::TimingLiteral(s, span),
            Token::HardwareQubit(s) => Expr::HardwareQubit(s, span),
            _ => return Err(self.error("expected expression")),
        };
        self.advance();
        Ok(literal)
    }

    // ------------------------------------------------------------------
    // Index and range parsing
    // ------------------------------------------------------------------

    fn parse_index_operator(&mut self) -> Result<IndexOp<'a>> {
        let start = self.expect(Token::LBracket, "'['")?;
        let kind = if self.at(Token::LBrace) {
            IndexKind::Set(self.parse_set_expression()?.0)
        } else {
            IndexKind::Items(self.comma_list(&[Token::RBracket], Self::parse_index_item)?)
        };
        let end = self.expect(Token::RBracket, "']'")?;
        Ok(IndexOp {
            kind,
            span: start.to(end),
        })
    }

    fn parse_index_item(&mut self) -> Result<IndexItem<'a>> {
        // If starts with ':', it's a range with no start
        if self.eat(Token::Colon).is_some() {
            return Ok(IndexItem::Range(self.parse_range_after_colon(None)?));
        }

        let expr = self.expr()?;

        // If followed by ':', it's a range
        if self.eat(Token::Colon).is_some() {
            return Ok(IndexItem::Range(self.parse_range_after_colon(Some(expr))?));
        }

        Ok(IndexItem::Single(expr))
    }

    fn parse_range_expression(&mut self) -> Result<RangeExpr<'a>> {
        let start = if self.at(Token::Colon) {
            None
        } else {
            Some(self.expr()?)
        };
        self.expect(Token::Colon, "':'")?;
        self.parse_range_after_colon(start)
    }

    /// Parse the rest of a range expression after consuming the first ':'.
    /// `start` is the expression before the colon (if any). Ranges are
    /// `start : end` or `start : step : end` — the component after the
    /// first colon is the step only when a second colon follows.
    fn parse_range_after_colon(&mut self, start: Option<Expr<'a>>) -> Result<RangeExpr<'a>> {
        let second = self.opt_range_bound(&[Token::Colon, Token::RBracket, Token::Comma])?;
        let (step, end) = if self.eat(Token::Colon).is_some() {
            (
                second,
                self.opt_range_bound(&[Token::RBracket, Token::Comma])?,
            )
        } else {
            (None, second)
        };

        Ok(RangeExpr {
            start: start.map(Box::new),
            end,
            step,
        })
    }

    /// A range component, absent when one of `terminators` is next.
    fn opt_range_bound(&mut self, terminators: &[Token<'a>]) -> Result<Option<Box<Expr<'a>>>> {
        if self.at_any(terminators) {
            Ok(None)
        } else {
            Ok(Some(Box::new(self.expr()?)))
        }
    }

    fn parse_set_expression(&mut self) -> Result<(Vec<Expr<'a>>, Span)> {
        let start = self.expect(Token::LBrace, "'{'")?;
        let exprs = self.expr_list(Token::RBrace)?;
        let end = self.expect(Token::RBrace, "'}'")?;
        Ok((exprs, start.to(end)))
    }

    // ------------------------------------------------------------------
    // Type parsing
    // ------------------------------------------------------------------

    fn parse_scalar_type(&mut self) -> Result<ScalarType<'a>> {
        let start = self.peek_span();
        let sized = |kind| ScalarKind::Sized(kind, None);
        let mut kind = match self.peek() {
            Some(Token::Bit) => sized(SizedKind::Bit),
            Some(Token::Int) => sized(SizedKind::Int),
            Some(Token::Uint) => sized(SizedKind::Uint),
            Some(Token::Float) => sized(SizedKind::Float),
            Some(Token::Angle) => sized(SizedKind::Angle),
            Some(Token::Complex) => ScalarKind::Complex(None),
            Some(Token::Bool) => ScalarKind::Bool,
            Some(Token::Duration) => ScalarKind::Duration,
            Some(Token::Stretch) => ScalarKind::Stretch,
            Some(Token::Waveform) => ScalarKind::Waveform,
            Some(Token::Port) => ScalarKind::Port,
            Some(Token::Frame) => ScalarKind::Frame,
            _ => return Err(self.error("expected scalar type")),
        };
        self.advance();
        match &mut kind {
            ScalarKind::Sized(_, designator) => {
                *designator = self.opt_designator()?.map(Box::new);
            }
            ScalarKind::Complex(inner) if self.eat(Token::LBracket).is_some() => {
                *inner = Some(Box::new(self.parse_scalar_type()?));
                self.expect(Token::RBracket, "']'")?;
            }
            _ => {}
        }
        Ok(ScalarType {
            kind,
            span: start.to(self.prev_span()),
        })
    }

    fn parse_qubit_type(&mut self) -> Result<QubitType<'a>> {
        let start = self.expect(Token::Qubit, "'qubit'")?;
        let designator = self.opt_designator()?.map(Box::new);
        Ok(QubitType {
            designator,
            span: start.to(self.prev_span()),
        })
    }

    /// `ARRAY LBRACKET scalarType COMMA`, the head shared by array types and
    /// array reference types.
    fn parse_array_head(&mut self) -> Result<ScalarType<'a>> {
        self.expect(Token::Array, "'array'")?;
        self.expect(Token::LBracket, "'['")?;
        let element_type = self.parse_scalar_type()?;
        self.expect(Token::Comma, "','")?;
        Ok(element_type)
    }

    fn parse_array_type(&mut self) -> Result<ArrayType<'a>> {
        let start = self.peek_span();
        let element_type = self.parse_array_head()?;
        let dimensions = self.expr_list(Token::RBracket)?;
        let end = self.expect(Token::RBracket, "']'")?;
        Ok(ArrayType {
            element_type,
            dimensions,
            span: start.to(end),
        })
    }

    fn parse_array_ref_type(&mut self) -> Result<ArrayRefType<'a>> {
        let start = self.peek_span();
        let mutability = if self.eat(Token::Readonly).is_some() {
            ArrayRefMut::Readonly
        } else if self.eat(Token::Mutable).is_some() {
            ArrayRefMut::Mutable
        } else {
            return Err(self.error("expected 'readonly' or 'mutable'"));
        };
        let element_type = self.parse_array_head()?;
        let dimensions = if self.eat(Token::Dim).is_some() {
            self.expect(Token::Equals, "'='")?;
            ArrayRefDims::Dim(self.expr()?)
        } else {
            ArrayRefDims::ExprList(self.expr_list(Token::RBracket)?)
        };
        let end = self.expect(Token::RBracket, "']'")?;
        Ok(ArrayRefType {
            mutability,
            element_type,
            dimensions,
            span: start.to(end),
        })
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr<'a>> {
        if self.at(Token::Array) {
            Ok(TypeExpr::Array(self.parse_array_type()?))
        } else {
            Ok(TypeExpr::Scalar(self.parse_scalar_type()?))
        }
    }

    fn parse_designator(&mut self) -> Result<Expr<'a>> {
        self.expect(Token::LBracket, "'['")?;
        let expr = self.expr()?;
        self.expect(Token::RBracket, "']'")?;
        Ok(expr)
    }

    fn opt_designator(&mut self) -> Result<Option<Expr<'a>>> {
        if self.at(Token::LBracket) {
            Ok(Some(self.parse_designator()?))
        } else {
            Ok(None)
        }
    }

    // ------------------------------------------------------------------
    // Operands and argument definitions
    // ------------------------------------------------------------------

    fn parse_indexed_identifier(&mut self) -> Result<IndexedIdent<'a>> {
        let name = self.expect_ident()?;
        let mut indices = Vec::new();
        while self.at(Token::LBracket) {
            indices.push(self.parse_index_operator()?);
        }
        Ok(IndexedIdent {
            span: name.span.to(self.prev_span()),
            name,
            indices,
        })
    }

    fn parse_gate_operand(&mut self) -> Result<GateOperand<'a>> {
        match self.peek() {
            Some(Token::HardwareQubit(name)) => {
                Ok(GateOperand::HardwareQubit(name, self.advance().1))
            }
            _ => Ok(GateOperand::Indexed(self.parse_indexed_identifier()?)),
        }
    }

    fn parse_arg_def(&mut self) -> Result<ArgDef<'a>> {
        match self.peek() {
            Some(tok @ (Token::Creg | Token::Qreg)) => {
                self.advance();
                let name = self.expect_ident()?;
                let designator = self.opt_designator()?;
                Ok(if tok == Token::Creg {
                    ArgDef::Creg(name, designator)
                } else {
                    ArgDef::Qreg(name, designator)
                })
            }
            Some(Token::Readonly | Token::Mutable) => {
                let ty = self.parse_array_ref_type()?;
                Ok(ArgDef::ArrayRef(ty, self.expect_ident()?))
            }
            Some(Token::Qubit) => {
                let ty = self.parse_qubit_type()?;
                Ok(ArgDef::Qubit(ty, self.expect_ident()?))
            }
            _ => {
                let ty = self.parse_scalar_type()?;
                Ok(ArgDef::Scalar(ty, self.expect_ident()?))
            }
        }
    }

    fn parse_extern_arg(&mut self) -> Result<ExternArg<'a>> {
        match self.peek() {
            Some(Token::Creg) => {
                self.advance();
                Ok(ExternArg::Creg(self.opt_designator()?))
            }
            Some(Token::Readonly | Token::Mutable) => {
                Ok(ExternArg::ArrayRef(self.parse_array_ref_type()?))
            }
            _ => Ok(ExternArg::Scalar(self.parse_scalar_type()?)),
        }
    }

    fn parse_defcal_arg_def(&mut self) -> Result<DefcalArgDef<'a>> {
        if matches!(
            self.peek(),
            Some(
                Token::Bit
                    | Token::Int
                    | Token::Uint
                    | Token::Float
                    | Token::Angle
                    | Token::Bool
                    | Token::Duration
                    | Token::Stretch
                    | Token::Complex
                    | Token::Qubit
                    | Token::Creg
                    | Token::Qreg
                    | Token::Readonly
                    | Token::Mutable
            )
        ) {
            Ok(DefcalArgDef::ArgDef(self.parse_arg_def()?))
        } else {
            Ok(DefcalArgDef::Expr(self.expr()?))
        }
    }

    fn parse_defcal_operand(&mut self) -> Result<DefcalOperand<'a>> {
        match self.peek() {
            Some(Token::HardwareQubit(name)) => {
                Ok(DefcalOperand::HardwareQubit(name, self.advance().1))
            }
            _ => Ok(DefcalOperand::Ident(self.expect_ident()?)),
        }
    }

    // ------------------------------------------------------------------
    // Conversion helpers
    // ------------------------------------------------------------------

    fn expr_to_indexed_ident(expr: Expr<'a>) -> Result<IndexedIdent<'a>> {
        match expr {
            Expr::Ident(name) => Ok(IndexedIdent {
                span: name.span,
                name,
                indices: vec![],
            }),
            Expr::Index { expr, index, span } => {
                let mut indexed = Self::expr_to_indexed_ident(*expr)?;
                indexed.indices.push(index);
                indexed.span = span;
                Ok(indexed)
            }
            _ => Err(Error::new(
                expr.span(),
                "expected identifier for assignment target",
            )),
        }
    }

    /// Split the expression that heads a gate call into the gate's name, its
    /// arguments and its duration designator.
    fn decompose_gate_head(expr: Expr<'a>) -> Result<DecomposedGateHead<'a>> {
        match expr {
            Expr::Ident(name) => Ok((name, None, None)),
            Expr::Call { name, args, .. } => Ok((name, Some(args), None)),
            Expr::Index { expr, index, .. } => {
                let (name, args, designator) = Self::decompose_gate_head(*expr)?;
                if designator.is_some() {
                    return Err(Error::new(index.span, "invalid gate call syntax"));
                }
                // The designator is an index holding a single expression.
                let IndexKind::Items(mut items) = index.kind else {
                    return Err(Error::new(
                        index.span,
                        "designator must be a single expression",
                    ));
                };
                match (items.pop(), items.is_empty()) {
                    (Some(IndexItem::Single(designator)), true) => {
                        Ok((name, args, Some(Box::new(designator))))
                    }
                    _ => Err(Error::new(
                        index.span,
                        "designator must be a single expression",
                    )),
                }
            }
            _ => Err(Error::new(expr.span(), "expected identifier for gate call")),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Program<'_> {
        parse(src).unwrap_or_else(|e| panic!("parse error: {} at {:?}", e.message, e.span))
    }

    fn parse_stmt(src: &str) -> Stmt<'_> {
        let prog = parse_ok(src);
        assert_eq!(prog.body.len(), 1, "expected exactly one statement");
        match prog.body.into_iter().next().unwrap() {
            StmtOrScope::Stmt(s) => *s,
            StmtOrScope::Scope(_) => panic!("expected statement, got scope"),
        }
    }

    #[test]
    fn version() {
        let prog = parse_ok("OPENQASM 3.0;");
        let v = prog.version.unwrap();
        assert_eq!(v.specifier, "3.0");
    }

    #[test]
    fn include() {
        let stmt = parse_stmt(r#"include "stdgates.inc";"#);
        assert!(matches!(stmt.kind, StmtKind::Include("\"stdgates.inc\"")));
    }

    #[test]
    fn qubit_declaration() {
        let stmt = parse_stmt("qubit[4] q;");
        match stmt.kind {
            StmtKind::QuantumDecl { ty, name } => {
                assert!(ty.designator.is_some());
                assert_eq!(name.name, "q");
            }
            _ => panic!("expected quantum decl"),
        }
    }

    #[test]
    fn classical_declaration() {
        let stmt = parse_stmt("int[32] x = 5;");
        match stmt.kind {
            StmtKind::ClassicalDecl { ty, name, init } => {
                assert!(matches!(
                    ty,
                    TypeExpr::Scalar(ScalarType {
                        kind: ScalarKind::Sized(SizedKind::Int, Some(_)),
                        ..
                    })
                ));
                assert_eq!(name.name, "x");
                assert!(init.is_some());
            }
            _ => panic!("expected classical decl"),
        }
    }

    #[test]
    fn gate_definition() {
        let stmt = parse_stmt("gate cx a, b { }");
        match stmt.kind {
            StmtKind::Gate {
                name,
                params,
                qubits,
                ..
            } => {
                assert_eq!(name.name, "cx");
                assert!(params.is_empty());
                assert_eq!(qubits.len(), 2);
            }
            _ => panic!("expected gate definition"),
        }
    }

    #[test]
    fn gate_with_params() {
        let stmt = parse_stmt("gate rz(theta) q { }");
        match stmt.kind {
            StmtKind::Gate {
                name,
                params,
                qubits,
                ..
            } => {
                assert_eq!(name.name, "rz");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "theta");
                assert_eq!(qubits.len(), 1);
            }
            _ => panic!("expected gate definition"),
        }
    }

    #[test]
    fn gate_call() {
        let stmt = parse_stmt("cx q[0], q[1];");
        match stmt.kind {
            StmtKind::GateCall { name, operands, .. } => {
                assert_eq!(name.name, "cx");
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("expected gate call"),
        }
    }

    #[test]
    fn gate_call_with_args() {
        let stmt = parse_stmt("U(pi/4, 0, pi/2) q[0];");
        match stmt.kind {
            StmtKind::GateCall {
                name,
                args,
                operands,
                ..
            } => {
                assert_eq!(name.name, "U");
                assert_eq!(args.unwrap().len(), 3);
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("expected gate call"),
        }
    }

    #[test]
    fn if_else() {
        let stmt = parse_stmt("if (x == 0) y = 1; else y = 2;");
        assert!(matches!(
            stmt.kind,
            StmtKind::If {
                else_body: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn for_loop() {
        let stmt = parse_stmt("for int i in [0:10] x = i;");
        match stmt.kind {
            StmtKind::For { var, iterable, .. } => {
                assert_eq!(var.name, "i");
                assert!(matches!(iterable, ForIterable::Range(_, _)));
            }
            _ => panic!("expected for loop"),
        }
    }

    #[test]
    fn expression_precedence() {
        // a + b * c should parse as a + (b * c)
        let stmt = parse_stmt("a + b * c;");
        match stmt.kind {
            StmtKind::Expr(Expr::BinOp {
                op: BinOp::Add,
                right,
                ..
            }) => {
                assert!(matches!(*right, Expr::BinOp { op: BinOp::Mul, .. }));
            }
            _ => panic!("expected binary op"),
        }
    }

    #[test]
    fn power_right_assoc() {
        // a ** b ** c should parse as a ** (b ** c)
        let stmt = parse_stmt("a ** b ** c;");
        match stmt.kind {
            StmtKind::Expr(Expr::BinOp {
                op: BinOp::Pow,
                right,
                ..
            }) => {
                assert!(matches!(*right, Expr::BinOp { op: BinOp::Pow, .. }));
            }
            _ => panic!("expected power expression"),
        }
    }

    #[test]
    fn unary_minus_vs_power() {
        // -a ** b should parse as -(a ** b)
        let stmt = parse_stmt("-a ** b;");
        match stmt.kind {
            StmtKind::Expr(Expr::UnaryOp {
                op: UnOp::Neg,
                operand,
                ..
            }) => {
                assert!(matches!(*operand, Expr::BinOp { op: BinOp::Pow, .. }));
            }
            _ => panic!("expected unary neg wrapping power"),
        }
    }

    #[test]
    fn measure_arrow() {
        let stmt = parse_stmt("measure q[0] -> c[0];");
        match stmt.kind {
            StmtKind::MeasureArrow { target, .. } => {
                assert!(target.is_some());
            }
            _ => panic!("expected measure arrow"),
        }
    }

    #[test]
    fn assignment() {
        let stmt = parse_stmt("x[0] = 42;");
        match stmt.kind {
            StmtKind::Assignment { target, op, .. } => {
                assert_eq!(target.name.name, "x");
                assert_eq!(target.indices.len(), 1);
                assert!(matches!(op, AssignOp::Assign));
            }
            _ => panic!("expected assignment"),
        }
    }

    #[test]
    fn compound_assignment() {
        let stmt = parse_stmt("x += 1;");
        match stmt.kind {
            StmtKind::Assignment { op, .. } => {
                assert_eq!(op, AssignOp::AddAssign);
            }
            _ => panic!("expected compound assignment"),
        }
    }

    #[test]
    fn barrier() {
        let stmt = parse_stmt("barrier q;");
        assert!(matches!(stmt.kind, StmtKind::Barrier(_)));
    }

    #[test]
    fn reset() {
        let stmt = parse_stmt("reset q[0];");
        assert!(matches!(stmt.kind, StmtKind::Reset(_)));
    }

    #[test]
    fn while_loop() {
        let stmt = parse_stmt("while (x > 0) x -= 1;");
        assert!(matches!(stmt.kind, StmtKind::While { .. }));
    }

    #[test]
    fn break_continue() {
        let prog = parse_ok("break; continue;");
        assert_eq!(prog.body.len(), 2);
    }

    #[test]
    fn cal_block() {
        let stmt = parse_stmt("cal { some opaque stuff }");
        match stmt.kind {
            StmtKind::Cal(ast::CalBody::Raw(body)) => assert!(body.is_some()),
            _ => panic!("expected cal"),
        }
    }

    #[test]
    fn openpulse_cal_block_is_parsed() {
        let src = r#"defcalgrammar "openpulse";
            cal {
                extern port p0;
                frame d0 = newframe(p0, 5.0e9, 0.0);
            }"#;
        let prog = parse_ok(src);
        let cal = prog
            .body
            .iter()
            .find_map(|item| match item {
                ast::StmtOrScope::Stmt(s) => match &s.kind {
                    StmtKind::Cal(body) => Some(body),
                    _ => None,
                },
                _ => None,
            })
            .expect("cal statement");
        let stmts = match cal {
            ast::CalBody::OpenPulse(stmts) => stmts,
            ast::CalBody::Raw(_) => panic!("expected parsed openpulse body"),
        };
        assert_eq!(stmts.len(), 2);
        assert!(matches!(
            &stmts[0],
            ast::StmtOrScope::Stmt(s) if matches!(s.kind, StmtKind::ExternPort { .. })
        ));
    }

    #[test]
    fn openpulse_defcal_body_is_parsed() {
        let src = r#"defcalgrammar "openpulse";
            defcal x $0 {
                waveform wf = gaussian(0.1, 160dt, 40dt);
            }"#;
        let prog = parse_ok(src);
        let body = prog
            .body
            .iter()
            .find_map(|item| match item {
                ast::StmtOrScope::Stmt(s) => match &s.kind {
                    StmtKind::Defcal { body, .. } => Some(body),
                    _ => None,
                },
                _ => None,
            })
            .expect("defcal statement");
        let stmts = match body {
            ast::CalBody::OpenPulse(stmts) => stmts,
            ast::CalBody::Raw(_) => panic!("expected parsed openpulse body"),
        };
        assert_eq!(stmts.len(), 1);
        let decl = match &stmts[0] {
            ast::StmtOrScope::Stmt(s) => &s.kind,
            _ => panic!("expected stmt"),
        };
        match decl {
            StmtKind::ClassicalDecl { ty, .. } => {
                assert!(matches!(
                    ty,
                    ast::TypeExpr::Scalar(ast::ScalarType {
                        kind: ast::ScalarKind::Waveform,
                        ..
                    })
                ));
            }
            _ => panic!("expected classical decl"),
        }
    }

    #[test]
    fn openpulse_rejects_nested_gate() {
        let src = r#"defcalgrammar "openpulse";
            cal {
                gate g q { }
            }"#;
        let err = parse(src).expect_err("parse should fail");
        assert!(err.message.contains("gate"), "got: {}", err.message);
    }

    #[test]
    fn openpulse_extern_frame_port() {
        let src = r#"defcalgrammar "openpulse";
            cal {
                extern frame f;
                extern port p;
            }"#;
        let prog = parse_ok(src);
        let cal_body = prog
            .body
            .iter()
            .find_map(|item| match item {
                ast::StmtOrScope::Stmt(s) => match &s.kind {
                    StmtKind::Cal(body) => Some(body),
                    _ => None,
                },
                _ => None,
            })
            .expect("cal");
        let stmts = match cal_body {
            ast::CalBody::OpenPulse(stmts) => stmts,
            _ => panic!("expected openpulse body"),
        };
        assert!(matches!(
            &stmts[0],
            ast::StmtOrScope::Stmt(s) if matches!(s.kind, StmtKind::ExternFrame { .. })
        ));
        assert!(matches!(
            &stmts[1],
            ast::StmtOrScope::Stmt(s) if matches!(s.kind, StmtKind::ExternPort { .. })
        ));
    }

    #[test]
    fn non_openpulse_cal_body_stays_raw() {
        let src = r#"defcalgrammar "other";
            cal { anything goes here }"#;
        let prog = parse_ok(src);
        let cal = prog
            .body
            .iter()
            .find_map(|item| match item {
                ast::StmtOrScope::Stmt(s) => match &s.kind {
                    StmtKind::Cal(body) => Some(body),
                    _ => None,
                },
                _ => None,
            })
            .expect("cal");
        assert!(matches!(cal, ast::CalBody::Raw(Some(_))));
    }

    #[test]
    fn cphase_program() {
        // Simplified cphase from fixtures
        let src = r#"gate cphase(θ) a, b
{
  U(0, 0, θ / 2) a;
  CX a, b;
  U(0, 0, -θ / 2) b;
  CX a, b;
  U(0, 0, θ / 2) b;
}
cphase(π / 2) q[0], q[1];
"#;
        let prog = parse_ok(src);
        assert_eq!(prog.body.len(), 2);
        match &prog.body[0] {
            StmtOrScope::Stmt(s) => assert!(matches!(s.kind, StmtKind::Gate { .. })),
            _ => panic!("expected gate"),
        }
        match &prog.body[1] {
            StmtOrScope::Stmt(s) => assert!(matches!(s.kind, StmtKind::GateCall { .. })),
            _ => panic!("expected gate call"),
        }
    }

    #[test]
    fn delay_statement() {
        let stmt = parse_stmt("delay[g] q[2];");
        match stmt.kind {
            StmtKind::Delay { operands, .. } => {
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("expected delay"),
        }
    }

    #[test]
    fn gphase_call() {
        let stmt = parse_stmt("gphase(pi);");
        match stmt.kind {
            StmtKind::GateCall {
                name,
                args,
                operands,
                ..
            } => {
                assert_eq!(name.name, "gphase");
                assert_eq!(args.unwrap().len(), 1);
                assert!(operands.is_empty());
            }
            _ => panic!("expected gphase call"),
        }
    }

    #[test]
    fn switch_statement() {
        let src = "switch (x) { case 0 { y = 1; } default { y = 2; } }";
        let stmt = parse_stmt(src);
        match stmt.kind {
            StmtKind::Switch { cases, .. } => {
                assert_eq!(cases.len(), 2);
            }
            _ => panic!("expected switch"),
        }
    }

    #[test]
    fn def_statement() {
        let src = "def foo(int[32] x, qubit q) -> bit { return measure q; }";
        let stmt = parse_stmt(src);
        match stmt.kind {
            StmtKind::Def {
                name,
                params,
                return_ty,
                ..
            } => {
                assert_eq!(name.name, "foo");
                assert_eq!(params.len(), 2);
                assert!(return_ty.is_some());
            }
            _ => panic!("expected def"),
        }
    }

    #[test]
    fn cast_expression() {
        let stmt = parse_stmt("float[64](x);");
        match stmt.kind {
            StmtKind::Expr(Expr::Cast { .. }) => {}
            _ => panic!("expected cast expression"),
        }
    }

    #[test]
    fn index_expression() {
        let stmt = parse_stmt("a[0];");
        match stmt.kind {
            StmtKind::Expr(Expr::Index { .. }) => {}
            _ => panic!("expected index expression"),
        }
    }

    #[test]
    fn gate_modifier_inv() {
        let stmt = parse_stmt("inv @ cx q[0], q[1];");
        match stmt.kind {
            StmtKind::GateCall { modifiers, .. } => {
                assert_eq!(modifiers.len(), 1);
                assert!(matches!(modifiers[0], GateModifier::Inv(_)));
            }
            _ => panic!("expected gate call with modifier"),
        }
    }

    #[test]
    fn old_style_decl() {
        let stmt = parse_stmt("creg c[4];");
        match stmt.kind {
            StmtKind::OldStyleDecl {
                keyword,
                name,
                designator,
            } => {
                assert!(matches!(keyword, OldStyleKind::Creg));
                assert_eq!(name.name, "c");
                assert!(designator.is_some());
            }
            _ => panic!("expected old-style decl"),
        }
    }

    #[test]
    fn function_call_expression() {
        let stmt = parse_stmt("sin(x);");
        match stmt.kind {
            StmtKind::Expr(Expr::Call { name, args, .. }) => {
                assert_eq!(name.name, "sin");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected function call"),
        }
    }

    #[test]
    fn nested_expression() {
        let stmt = parse_stmt("(a + b) * c;");
        match stmt.kind {
            StmtKind::Expr(Expr::BinOp {
                op: BinOp::Mul,
                left,
                ..
            }) => {
                assert!(matches!(*left, Expr::Paren(..)));
            }
            _ => panic!("expected multiply with paren"),
        }
    }

    #[test]
    fn multiple_statements() {
        let prog = parse_ok("qubit q; bit c; measure q -> c;");
        assert_eq!(prog.body.len(), 3);
    }

    #[test]
    fn annotation() {
        let src = "@my.ann some_content\nbarrier q;";
        let stmt = parse_stmt(src);
        assert_eq!(stmt.annotations.len(), 1);
        assert_eq!(stmt.annotations[0].keyword, "@my.ann");
    }

    #[test]
    fn extern_statement() {
        let stmt = parse_stmt("extern foo(int, float) -> int;");
        match stmt.kind {
            StmtKind::Extern {
                name,
                params,
                return_ty,
            } => {
                assert_eq!(name.name, "foo");
                assert_eq!(params.len(), 2);
                assert!(return_ty.is_some());
            }
            _ => panic!("expected extern"),
        }
    }

    #[test]
    fn io_declarations() {
        let prog = parse_ok("input int[32] x; output bit y;");
        assert_eq!(prog.body.len(), 2);
    }

    #[test]
    fn trailing_commas_are_allowed_in_every_list() {
        // Every list rule of the reference grammar ends in `COMMA?`.
        for src in [
            "barrier q[0], q[1],;",
            "h q[0], q[1],;",
            "ctrl @ x q[0], q[1],;",
            "f q, r, -> c;",
            "c = f q, r,;",
            "gate g a, b, { }",
            "gate g(t, u,) a { }",
            "defcal x $0, $1, { }",
            "defcal x(a, b,) $0 { }",
            "switch (x) { case 0, 1, { } }",
            "def f(int a, int b,) { }",
            "extern g(int, float,);",
            "f(1, 2,);",
            "array[int, 2, 3,] a = {1, 2,};",
            "a[0, 1,];",
            "let b = q[{0, 1,}];",
            "for int i in {1, 2,} { }",
        ] {
            parse_ok(src);
        }
    }

    #[test]
    fn a_list_needs_an_item() {
        for src in [
            "gate g { }",
            "gate g(,) a { }",
            "switch (x) { case { } }",
            "h ,;",
            "f(,);",
        ] {
            assert!(parse(src).is_err(), "{src}");
        }
    }

    #[test]
    fn comments_are_returned_in_order_with_openpulse_bodies() {
        let src = concat!(
            "// a\n",
            "defcalgrammar \"openpulse\";\n",
            "cal { // b\n extern port p; /* c */ }\n",
            "// d\n",
        );
        let (_, comments) = parse_with_comments(src).unwrap();
        let texts: Vec<_> = comments.iter().map(|c| c.text).collect();
        assert_eq!(texts, ["// a", "// b", "/* c */", "// d"]);
        for comment in &comments {
            assert_eq!(&src[comment.span], comment.text);
        }
    }

    #[test]
    fn openpulse_exclusions_apply_to_the_block_itself() {
        // The reference OpenPulse grammar restricts `openpulseStatement`, the
        // statements directly inside a calibration block; nested bodies use
        // the ordinary statement rule.
        let grammar = "defcalgrammar \"openpulse\";";
        for excluded in [
            "nop $0;",
            "gate g q { }",
            "defcalgrammar \"openpulse\";",
            "creg c[2];",
            "qreg q[2];",
            "measure $0;",
            "measure $0 -> c;",
            "@note\n nop $0;",
        ] {
            let err = parse(&format!("{grammar} cal {{ {excluded} }}")).unwrap_err();
            assert!(
                err.message.contains("not allowed inside"),
                "{excluded}: {}",
                err.message
            );
        }
        parse_ok(&format!("{grammar} cal {{ box {{ nop $0; }} }}"));
        parse_ok(&format!("{grammar} cal {{ bit c; c = measure $0; }}"));
    }

    #[test]
    fn sized_type_span_includes_the_designator() {
        let src = "int[32] x; qubit[4] q; complex[float[64]] z;";
        let prog = parse_ok(src);
        let spans: Vec<_> = prog
            .body
            .iter()
            .map(|item| match item {
                StmtOrScope::Stmt(s) => match &s.kind {
                    StmtKind::ClassicalDecl {
                        ty: TypeExpr::Scalar(ty),
                        ..
                    } => ty.span,
                    StmtKind::QuantumDecl { ty, .. } => ty.span,
                    other => panic!("unexpected {other:?}"),
                },
                StmtOrScope::Scope(_) => panic!("unexpected scope"),
            })
            .collect();
        let texts: Vec<_> = spans.into_iter().map(|span| &src[span]).collect();
        assert_eq!(texts, ["int[32]", "qubit[4]", "complex[float[64]]"]);
    }

    #[test]
    fn gate_head_errors_point_at_the_designator() {
        let src = "g[1:2] q;";
        let err = parse(src).unwrap_err();
        assert_eq!(err.message, "designator must be a single expression");
        assert_eq!(&src[err.span], "[1:2]");
    }

    #[test]
    fn range_three_components_is_start_step_end() {
        let stmt = parse_stmt("for uint i in [2: -1: 0] { }");
        let StmtKind::For { iterable, .. } = stmt.kind else {
            panic!("expected for");
        };
        let ForIterable::Range(r, _) = iterable else {
            panic!("expected range iterable");
        };
        assert!(matches!(
            r.start.as_deref(),
            Some(Expr::IntLiteral("2", ..))
        ));
        assert!(
            matches!(r.step.as_deref(), Some(Expr::UnaryOp { .. })),
            "middle component is the step"
        );
        assert!(
            matches!(r.end.as_deref(), Some(Expr::IntLiteral("0", ..))),
            "trailing component is the end"
        );
    }

    #[test]
    fn range_two_components_has_no_step() {
        let stmt = parse_stmt("for uint i in [0: 3] { }");
        let StmtKind::For { iterable, .. } = stmt.kind else {
            panic!("expected for");
        };
        let ForIterable::Range(r, _) = iterable else {
            panic!("expected range iterable");
        };
        assert!(matches!(
            r.start.as_deref(),
            Some(Expr::IntLiteral("0", ..))
        ));
        assert!(r.step.is_none());
        assert!(matches!(r.end.as_deref(), Some(Expr::IntLiteral("3", ..))));
    }
}
