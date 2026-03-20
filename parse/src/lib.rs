pub mod ast;

use ast::*;
pub use oqi_lex::{Error, Result};
use oqi_lex::{Lexer, Token};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse(source: &str) -> Result<Program<'_>> {
    Parser::new(Lexer::new(source).collect::<Result<Vec<_>>>()?).parse_program()
}

// ---------------------------------------------------------------------------
// Type Utilities
// ---------------------------------------------------------------------------
type DecomposedGateHead<'a> = (Ident<'a>, Option<Vec<Expr<'a>>>, Option<Box<Expr<'a>>>);

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct Parser<'a> {
    tokens: Vec<(Token<'a>, Span)>,
    max_pos: usize,
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: impl IntoIterator<Item = (Token<'a>, Span)>) -> Self {
        let tokens = tokens
            .into_iter()
            .filter(|(tok, _)| !matches!(tok, Token::LineComment(_) | Token::BlockComment(_)))
            .collect::<Vec<_>>();
        let max_pos = tokens.last().map(|(_, s)| s.end).unwrap_or(0);
        Self {
            tokens,
            max_pos,
            pos: 0,
        }
    }

    // ------------------------------------------------------------------
    // Utility methods
    // ------------------------------------------------------------------

    fn peek(&self) -> Option<&Token<'a>> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn peek_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|(_, s)| s.clone())
            .unwrap_or(self.max_pos..self.max_pos)
    }

    fn advance(&mut self) -> (Token<'a>, Span) {
        let item = self.tokens[self.pos].clone();
        self.pos += 1;
        item
    }

    fn prev_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].1.clone()
        } else {
            0..0
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn error(&self, msg: impl Into<String>) -> Error {
        Error {
            span: self.peek_span(),
            message: msg.into(),
        }
    }

    fn expect_semi(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::Semicolon)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected ';'"))
        }
    }

    fn expect_lparen(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::LParen)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected '('"))
        }
    }

    fn expect_rparen(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::RParen)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected ')'"))
        }
    }

    fn expect_lbracket(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::LBracket)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected '['"))
        }
    }

    fn expect_rbracket(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::RBracket)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected ']'"))
        }
    }

    fn expect_lbrace(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::LBrace)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected '{'"))
        }
    }

    fn expect_rbrace(&mut self) -> Result<Span> {
        if matches!(self.peek(), Some(Token::RBrace)) {
            Ok(self.advance().1)
        } else {
            Err(self.error("expected '}'"))
        }
    }

    fn expect_ident(&mut self) -> Result<Ident<'a>> {
        if matches!(self.peek(), Some(Token::Identifier(_))) {
            let (tok, span) = self.advance();
            let Token::Identifier(name) = tok else {
                unreachable!()
            };
            Ok(Ident { name, span })
        } else {
            Err(self.error("expected identifier"))
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
    // Program structure
    // ------------------------------------------------------------------

    pub fn parse_program(&mut self) -> Result<Program<'a>> {
        let version = if matches!(self.peek(), Some(Token::OpenQasm)) {
            Some(self.parse_version()?)
        } else {
            None
        };
        let mut body = Vec::new();
        while !self.at_end() {
            body.push(self.parse_stmt_or_scope()?);
        }
        Ok(Program {
            version,
            body,
            span: 0..self.max_pos,
        })
    }

    fn parse_version(&mut self) -> Result<Version<'a>> {
        let (_, start) = self.advance(); // eat OPENQASM
        let spec_span = self.peek_span();
        if matches!(self.peek(), Some(Token::VersionSpecifier(_))) {
            let (tok, _) = self.advance();
            let Token::VersionSpecifier(specifier) = tok else {
                unreachable!()
            };
            let end = self.expect_semi()?;
            Ok(Version {
                specifier,
                span: start.start..end.end,
            })
        } else {
            Err(Error {
                span: spec_span,
                message: "expected version specifier".into(),
            })
        }
    }

    fn parse_stmt_or_scope(&mut self) -> Result<StmtOrScope<'a>> {
        if matches!(self.peek(), Some(Token::LBrace)) {
            Ok(StmtOrScope::Scope(self.parse_scope()?))
        } else {
            Ok(StmtOrScope::Stmt(self.parse_statement()?))
        }
    }

    fn parse_scope(&mut self) -> Result<Scope<'a>> {
        let start = self.expect_lbrace()?;
        let mut body = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace) | None) {
            body.push(self.parse_stmt_or_scope()?);
        }
        let end = self.expect_rbrace()?;
        Ok(Scope {
            body,
            span: start.start..end.end,
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt<'a>> {
        let start = self.peek_span();

        // Pragma doesn't have annotations
        if matches!(self.peek(), Some(Token::Pragma)) {
            let kind = self.parse_pragma()?;
            let end = self.prev_span();
            return Ok(Stmt {
                annotations: vec![],
                kind,
                span: start.start..end.end,
            });
        }

        let annotations = self.parse_annotations();
        let kind = self.parse_stmt_kind()?;
        let end = self.prev_span();
        Ok(Stmt {
            annotations,
            kind,
            span: start.start..end.end,
        })
    }

    fn parse_annotations(&mut self) -> Vec<Annotation<'a>> {
        let mut annotations = Vec::new();
        while matches!(self.peek(), Some(Token::AnnotationKeyword(_))) {
            let (tok, kw_span) = self.advance();
            let Token::AnnotationKeyword(keyword) = tok else {
                unreachable!()
            };
            let content = if matches!(self.peek(), Some(Token::RemainingLineContent(_))) {
                let (tok, _) = self.advance();
                let Token::RemainingLineContent(c) = tok else {
                    unreachable!()
                };
                Some(c)
            } else {
                None
            };
            let end = self.prev_span();
            annotations.push(Annotation {
                keyword,
                content,
                span: kw_span.start..end.end,
            });
        }
        annotations
    }

    // ------------------------------------------------------------------
    // Statement dispatch
    // ------------------------------------------------------------------

    fn parse_stmt_kind(&mut self) -> Result<StmtKind<'a>> {
        let Some(tok) = self.peek().cloned() else {
            return Err(self.error("expected statement"));
        };
        match tok {
            Token::Include => self.parse_include(),
            Token::DefCalGrammar => self.parse_calibration_grammar(),
            Token::Break => {
                self.advance();
                self.expect_semi()?;
                Ok(StmtKind::Break)
            }
            Token::Continue => {
                self.advance();
                self.expect_semi()?;
                Ok(StmtKind::Continue)
            }
            Token::End => {
                self.advance();
                self.expect_semi()?;
                Ok(StmtKind::End)
            }
            Token::For => self.parse_for(),
            Token::If => self.parse_if(),
            Token::Return => self.parse_return(),
            Token::While => self.parse_while(),
            Token::Switch => self.parse_switch(),
            Token::Barrier => self.parse_barrier(),
            Token::Box => self.parse_box(),
            Token::Delay => self.parse_delay(),
            Token::Nop => self.parse_nop(),
            Token::Reset => self.parse_reset(),
            Token::Let => self.parse_alias(),
            Token::Const => self.parse_const_decl(),
            Token::Input | Token::Output => self.parse_io_decl(),
            Token::Creg | Token::Qreg => self.parse_old_style_decl(),
            Token::Qubit => self.parse_quantum_decl(),
            Token::Def => self.parse_def(),
            Token::Extern => self.parse_extern(),
            Token::Gate => self.parse_gate(),
            Token::Cal => self.parse_cal(),
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
            | Token::Array => self.parse_classical_decl(),
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
        if matches!(self.peek(), Some(Token::RemainingLineContent(_))) {
            let (tok, _) = self.advance();
            let Token::RemainingLineContent(content) = tok else {
                unreachable!()
            };
            Ok(StmtKind::Pragma(content))
        } else {
            Err(self.error("expected pragma content"))
        }
    }

    fn parse_include(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat include
        if matches!(self.peek(), Some(Token::StringLiteral(_))) {
            let (tok, _) = self.advance();
            let Token::StringLiteral(path) = tok else {
                unreachable!()
            };
            self.expect_semi()?;
            Ok(StmtKind::Include(path))
        } else {
            Err(self.error("expected string literal"))
        }
    }

    fn parse_calibration_grammar(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat defcalgrammar
        if matches!(self.peek(), Some(Token::StringLiteral(_))) {
            let (tok, _) = self.advance();
            let Token::StringLiteral(path) = tok else {
                unreachable!()
            };
            self.expect_semi()?;
            Ok(StmtKind::CalibrationGrammar(path))
        } else {
            Err(self.error("expected string literal"))
        }
    }

    // ------------------------------------------------------------------
    // Control flow
    // ------------------------------------------------------------------

    fn parse_for(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat for
        let ty = self.parse_scalar_type()?;
        let var = self.expect_ident()?;
        // eat IN
        if !matches!(self.peek(), Some(Token::In)) {
            return Err(self.error("expected 'in'"));
        }
        self.advance();

        let iterable = match self.peek() {
            Some(Token::LBrace) => {
                let (exprs, span) = self.parse_set_expression()?;
                ForIterable::Set(exprs, span)
            }
            Some(Token::LBracket) => {
                let start = self.advance().1;
                let range = self.parse_range_expression()?;
                let end = self.expect_rbracket()?;
                ForIterable::Range(range, start.start..end.end)
            }
            _ => ForIterable::Expr(self.parse_expr(0)?),
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
        self.expect_lparen()?;
        let condition = self.parse_expr(0)?;
        self.expect_rparen()?;
        let then_body = self.parse_stmt_or_scope()?;
        let else_body = if matches!(self.peek(), Some(Token::Else)) {
            self.advance();
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
        if matches!(self.peek(), Some(Token::Semicolon)) {
            self.advance();
            return Ok(StmtKind::Return(None));
        }
        let value = self.parse_expr_or_measure()?;
        self.expect_semi()?;
        Ok(StmtKind::Return(Some(value)))
    }

    fn parse_while(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat while
        self.expect_lparen()?;
        let condition = self.parse_expr(0)?;
        self.expect_rparen()?;
        let body = self.parse_stmt_or_scope()?;
        Ok(StmtKind::While {
            condition,
            body: Box::new(body),
        })
    }

    fn parse_switch(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat switch
        self.expect_lparen()?;
        let target = self.parse_expr(0)?;
        self.expect_rparen()?;
        self.expect_lbrace()?;
        let mut cases = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace) | None) {
            cases.push(self.parse_switch_case()?);
        }
        self.expect_rbrace()?;
        Ok(StmtKind::Switch { target, cases })
    }

    fn parse_switch_case(&mut self) -> Result<SwitchCase<'a>> {
        if matches!(self.peek(), Some(Token::Default)) {
            self.advance();
            let scope = self.parse_scope()?;
            Ok(SwitchCase::Default(scope))
        } else if matches!(self.peek(), Some(Token::Case)) {
            self.advance();
            let exprs = self.parse_expression_list()?;
            let scope = self.parse_scope()?;
            Ok(SwitchCase::Case(exprs, scope))
        } else {
            Err(self.error("expected 'case' or 'default'"))
        }
    }

    // ------------------------------------------------------------------
    // Quantum statements
    // ------------------------------------------------------------------

    fn parse_barrier(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat barrier
        let operands = if self.peek_is_gate_operand() {
            self.parse_gate_operand_list()?
        } else {
            vec![]
        };
        self.expect_semi()?;
        Ok(StmtKind::Barrier(operands))
    }

    fn parse_box(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat box
        let designator = if matches!(self.peek(), Some(Token::LBracket)) {
            Some(self.parse_designator()?)
        } else {
            None
        };
        let body = self.parse_scope()?;
        Ok(StmtKind::Box { designator, body })
    }

    fn parse_delay(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat delay
        let designator = self.parse_designator()?;
        let operands = if self.peek_is_gate_operand() {
            self.parse_gate_operand_list()?
        } else {
            vec![]
        };
        self.expect_semi()?;
        Ok(StmtKind::Delay {
            designator,
            operands,
        })
    }

    fn parse_nop(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat nop
        let operands = if self.peek_is_gate_operand() {
            self.parse_gate_operand_list()?
        } else {
            vec![]
        };
        self.expect_semi()?;
        Ok(StmtKind::Nop(operands))
    }

    fn parse_reset(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat reset
        let operand = self.parse_gate_operand()?;
        self.expect_semi()?;
        Ok(StmtKind::Reset(operand))
    }

    fn parse_measure_arrow(&mut self) -> Result<StmtKind<'a>> {
        let start = self.advance().1; // eat measure
        let operand = self.parse_gate_operand()?;
        let mspan = start.start..operand.span().end;
        let measure = MeasureExpr::Measure {
            operand,
            span: mspan,
        };
        let target = if matches!(self.peek(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_indexed_identifier()?)
        } else {
            None
        };
        self.expect_semi()?;
        Ok(StmtKind::MeasureArrow { measure, target })
    }

    // ------------------------------------------------------------------
    // Gate calls (including ambiguous identifier statements)
    // ------------------------------------------------------------------

    fn parse_ambiguous_stmt(&mut self) -> Result<StmtKind<'a>> {
        let expr = self.parse_expr(0)?;

        match self.peek() {
            Some(Token::Semicolon) => {
                self.advance();
                Ok(StmtKind::Expr(expr))
            }
            Some(
                Token::Equals
                | Token::PlusEquals
                | Token::MinusEquals
                | Token::AsteriskEquals
                | Token::SlashEquals
                | Token::AmpersandEquals
                | Token::PipeEquals
                | Token::CaretEquals
                | Token::LeftShiftEquals
                | Token::RightShiftEquals
                | Token::PercentEquals
                | Token::DoubleAsteriskEquals,
            ) => {
                let target = self.expr_to_indexed_ident(expr)?;
                let (tok, _) = self.advance();
                let op = match tok {
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
                    _ => unreachable!(),
                };
                let value = self.parse_expr_or_measure()?;
                self.expect_semi()?;
                Ok(StmtKind::Assignment { target, op, value })
            }
            Some(Token::Identifier(_) | Token::HardwareQubit(_)) => {
                // Gate call or quantum call expression
                let (name, args, designator) = self.decompose_gate_head(expr)?;
                let operands = self.parse_gate_operand_list()?;
                if matches!(self.peek(), Some(Token::Arrow)) {
                    self.advance();
                    let target = self.parse_indexed_identifier()?;
                    self.expect_semi()?;
                    let span = name.span.start..target.span.end;
                    Ok(StmtKind::MeasureArrow {
                        measure: MeasureExpr::QuantumCall {
                            name,
                            args: args.unwrap_or_default(),
                            operands,
                            span: span.clone(),
                        },
                        target: Some(target),
                    })
                } else {
                    self.expect_semi()?;
                    Ok(StmtKind::GateCall {
                        modifiers: vec![],
                        name: GateCallName::Ident(name),
                        args,
                        designator,
                        operands,
                    })
                }
            }
            _ => Err(self.error("expected ';', '=', or gate operand after expression")),
        }
    }

    fn parse_modified_gate_call(&mut self) -> Result<StmtKind<'a>> {
        let mut modifiers = Vec::new();
        while matches!(
            self.peek(),
            Some(Token::Inv | Token::Pow | Token::Ctrl | Token::Negctrl)
        ) {
            modifiers.push(self.parse_gate_modifier()?);
        }

        if matches!(self.peek(), Some(Token::Gphase)) {
            return self.parse_gphase_call(modifiers);
        }

        // Must be Identifier
        let expr = self.parse_expr(0)?;
        let (name, args, designator) = self.decompose_gate_head(expr)?;
        let operands = self.parse_gate_operand_list()?;
        self.expect_semi()?;
        Ok(StmtKind::GateCall {
            modifiers,
            name: GateCallName::Ident(name),
            args,
            designator,
            operands,
        })
    }

    fn parse_gphase_call(&mut self, modifiers: Vec<GateModifier<'a>>) -> Result<StmtKind<'a>> {
        let (_, gspan) = self.advance(); // eat gphase
        let args = if matches!(self.peek(), Some(Token::LParen)) {
            self.advance();
            let list = if matches!(self.peek(), Some(Token::RParen)) {
                vec![]
            } else {
                self.parse_expression_list()?
            };
            self.expect_rparen()?;
            Some(list)
        } else {
            None
        };
        let designator = if matches!(self.peek(), Some(Token::LBracket)) {
            Some(Box::new(self.parse_designator()?))
        } else {
            None
        };
        let operands = if self.peek_is_gate_operand() {
            self.parse_gate_operand_list()?
        } else {
            vec![]
        };
        self.expect_semi()?;
        Ok(StmtKind::GateCall {
            modifiers,
            name: GateCallName::Gphase(gspan),
            args,
            designator,
            operands,
        })
    }

    fn parse_gate_modifier(&mut self) -> Result<GateModifier<'a>> {
        let start = self.peek_span();
        let tok = self.peek().cloned();
        match tok {
            Some(Token::Inv) => {
                self.advance();
                if !matches!(self.peek(), Some(Token::At)) {
                    return Err(self.error("expected '@'"));
                }
                let end = self.advance().1;
                Ok(GateModifier::Inv(start.start..end.end))
            }
            Some(Token::Pow) => {
                self.advance();
                self.expect_lparen()?;
                let expr = self.parse_expr(0)?;
                self.expect_rparen()?;
                if !matches!(self.peek(), Some(Token::At)) {
                    return Err(self.error("expected '@'"));
                }
                let end = self.advance().1;
                Ok(GateModifier::Pow(expr, start.start..end.end))
            }
            Some(Token::Ctrl) => {
                self.advance();
                let expr = if matches!(self.peek(), Some(Token::LParen)) {
                    self.advance();
                    let e = self.parse_expr(0)?;
                    self.expect_rparen()?;
                    Some(e)
                } else {
                    None
                };
                if !matches!(self.peek(), Some(Token::At)) {
                    return Err(self.error("expected '@'"));
                }
                let end = self.advance().1;
                Ok(GateModifier::Ctrl(expr, start.start..end.end))
            }
            Some(Token::Negctrl) => {
                self.advance();
                let expr = if matches!(self.peek(), Some(Token::LParen)) {
                    self.advance();
                    let e = self.parse_expr(0)?;
                    self.expect_rparen()?;
                    Some(e)
                } else {
                    None
                };
                if !matches!(self.peek(), Some(Token::At)) {
                    return Err(self.error("expected '@'"));
                }
                let end = self.advance().1;
                Ok(GateModifier::NegCtrl(expr, start.start..end.end))
            }
            _ => Err(self.error("expected gate modifier")),
        }
    }

    // ------------------------------------------------------------------
    // Declaration statements
    // ------------------------------------------------------------------

    fn parse_alias(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat let
        let name = self.expect_ident()?;
        if !matches!(self.peek(), Some(Token::Equals)) {
            return Err(self.error("expected '='"));
        }
        self.advance();
        // aliasExpression: expression (DOUBLE_PLUS expression)*
        let mut value = vec![self.parse_expr(0)?];
        while matches!(self.peek(), Some(Token::DoublePlus)) {
            self.advance();
            value.push(self.parse_expr(0)?);
        }
        self.expect_semi()?;
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
        let init = if matches!(self.peek(), Some(Token::Equals)) {
            self.advance();
            Some(self.parse_decl_expr()?)
        } else {
            None
        };
        self.expect_semi()?;
        Ok(StmtKind::ClassicalDecl { ty, name, init })
    }

    fn parse_const_decl(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat const
        let ty = self.parse_scalar_type()?;
        let name = self.expect_ident()?;
        if !matches!(self.peek(), Some(Token::Equals)) {
            return Err(self.error("expected '='"));
        }
        self.advance();
        let init = self.parse_decl_expr()?;
        self.expect_semi()?;
        Ok(StmtKind::ConstDecl { ty, name, init })
    }

    fn parse_io_decl(&mut self) -> Result<StmtKind<'a>> {
        let dir = if matches!(self.peek(), Some(Token::Input)) {
            self.advance();
            IoDir::Input
        } else {
            self.advance(); // output
            IoDir::Output
        };
        let ty = self.parse_type_expr()?;
        let name = self.expect_ident()?;
        self.expect_semi()?;
        Ok(StmtKind::IoDecl { dir, ty, name })
    }

    fn parse_old_style_decl(&mut self) -> Result<StmtKind<'a>> {
        let keyword = if matches!(self.peek(), Some(Token::Creg)) {
            self.advance();
            OldStyleKind::Creg
        } else {
            self.advance(); // qreg
            OldStyleKind::Qreg
        };
        let name = self.expect_ident()?;
        let designator = if matches!(self.peek(), Some(Token::LBracket)) {
            Some(Box::new(self.parse_designator()?))
        } else {
            None
        };
        self.expect_semi()?;
        Ok(StmtKind::OldStyleDecl {
            keyword,
            name,
            designator,
        })
    }

    fn parse_quantum_decl(&mut self) -> Result<StmtKind<'a>> {
        let ty = self.parse_qubit_type()?;
        let name = self.expect_ident()?;
        self.expect_semi()?;
        Ok(StmtKind::QuantumDecl { ty, name })
    }

    // ------------------------------------------------------------------
    // Higher-order definitions
    // ------------------------------------------------------------------

    fn parse_def(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat def
        let name = self.expect_ident()?;
        self.expect_lparen()?;
        let params = if matches!(self.peek(), Some(Token::RParen)) {
            vec![]
        } else {
            self.parse_arg_def_list()?
        };
        self.expect_rparen()?;
        let return_ty = if matches!(self.peek(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_scalar_type()?)
        } else {
            None
        };
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
        let name = self.expect_ident()?;
        self.expect_lparen()?;
        let params = if matches!(self.peek(), Some(Token::RParen)) {
            vec![]
        } else {
            self.parse_extern_arg_list()?
        };
        self.expect_rparen()?;
        let return_ty = if matches!(self.peek(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_scalar_type()?)
        } else {
            None
        };
        self.expect_semi()?;
        Ok(StmtKind::Extern {
            name,
            params,
            return_ty,
        })
    }

    fn parse_gate(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat gate
        let name = self.expect_ident()?;
        let params = if matches!(self.peek(), Some(Token::LParen)) {
            self.advance();
            let list = if matches!(self.peek(), Some(Token::RParen)) {
                vec![]
            } else {
                self.parse_identifier_list()?
            };
            self.expect_rparen()?;
            list
        } else {
            vec![]
        };
        let qubits = self.parse_identifier_list()?;
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

    fn parse_cal(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat cal
        self.expect_lbrace()?;
        let body = if matches!(self.peek(), Some(Token::CalibrationBlock(_))) {
            let (tok, _) = self.advance();
            let Token::CalibrationBlock(s) = tok else {
                unreachable!()
            };
            Some(s)
        } else {
            None
        };
        self.expect_rbrace()?;
        Ok(StmtKind::Cal(body))
    }

    fn parse_defcal(&mut self) -> Result<StmtKind<'a>> {
        self.advance(); // eat defcal

        // defcalTarget
        let target = match self.peek().cloned() {
            Some(Token::Measure) => {
                let s = self.advance().1;
                DefcalTarget::Measure(s)
            }
            Some(Token::Reset) => {
                let s = self.advance().1;
                DefcalTarget::Reset(s)
            }
            Some(Token::Delay) => {
                let s = self.advance().1;
                DefcalTarget::Delay(s)
            }
            _ => DefcalTarget::Ident(self.expect_ident()?),
        };

        // Optional (args)
        let args = if matches!(self.peek(), Some(Token::LParen)) {
            self.advance();
            let list = if matches!(self.peek(), Some(Token::RParen)) {
                vec![]
            } else {
                self.parse_defcal_arg_def_list()?
            };
            self.expect_rparen()?;
            list
        } else {
            vec![]
        };

        // defcalOperandList
        let operands = self.parse_defcal_operand_list()?;

        // Optional return signature
        let return_ty = if matches!(self.peek(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_scalar_type()?)
        } else {
            None
        };

        // Body
        self.expect_lbrace()?;
        let body = if matches!(self.peek(), Some(Token::CalibrationBlock(_))) {
            let (tok, _) = self.advance();
            let Token::CalibrationBlock(s) = tok else {
                unreachable!()
            };
            Some(s)
        } else {
            None
        };
        self.expect_rbrace()?;

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
        let expr = self.parse_expr(0)?;
        self.expect_semi()?;
        Ok(StmtKind::Expr(expr))
    }

    // ------------------------------------------------------------------
    // Expression and measure helpers
    // ------------------------------------------------------------------

    fn parse_expr_or_measure(&mut self) -> Result<ExprOrMeasure<'a>> {
        if matches!(self.peek(), Some(Token::Measure)) {
            let start = self.peek_span();
            self.advance();
            let operand = self.parse_gate_operand()?;
            let span = start.start..operand.span().end;
            return Ok(ExprOrMeasure::Measure(MeasureExpr::Measure {
                operand,
                span,
            }));
        }

        let expr = self.parse_expr(0)?;

        // Check for quantum call: expression followed by gate operands
        if self.peek_is_gate_operand() {
            let (name, args, _) = self.decompose_gate_head(expr)?;
            let operands = self.parse_gate_operand_list()?;
            let end = operands.last().unwrap().span().end;
            let span = name.span.start..end;
            return Ok(ExprOrMeasure::Measure(MeasureExpr::QuantumCall {
                name,
                args: args.unwrap_or_default(),
                operands,
                span,
            }));
        }

        Ok(ExprOrMeasure::Expr(expr))
    }

    fn parse_decl_expr(&mut self) -> Result<DeclExpr<'a>> {
        if matches!(self.peek(), Some(Token::LBrace)) {
            return Ok(DeclExpr::ArrayLiteral(self.parse_array_literal()?));
        }
        if matches!(self.peek(), Some(Token::Measure)) {
            let start = self.peek_span();
            self.advance();
            let operand = self.parse_gate_operand()?;
            let span = start.start..operand.span().end;
            return Ok(DeclExpr::Measure(MeasureExpr::Measure { operand, span }));
        }

        let expr = self.parse_expr(0)?;

        if self.peek_is_gate_operand() {
            let (name, args, _) = self.decompose_gate_head(expr)?;
            let operands = self.parse_gate_operand_list()?;
            let end = operands.last().unwrap().span().end;
            let span = name.span.start..end;
            return Ok(DeclExpr::Measure(MeasureExpr::QuantumCall {
                name,
                args: args.unwrap_or_default(),
                operands,
                span,
            }));
        }

        Ok(DeclExpr::Expr(expr))
    }

    fn parse_array_literal(&mut self) -> Result<ArrayLiteral<'a>> {
        let start = self.expect_lbrace()?;
        let mut items = Vec::new();
        if !matches!(self.peek(), Some(Token::RBrace)) {
            loop {
                let item = if matches!(self.peek(), Some(Token::LBrace)) {
                    ArrayLiteralItem::Nested(self.parse_array_literal()?)
                } else {
                    ArrayLiteralItem::Expr(self.parse_expr(0)?)
                };
                items.push(item);
                if !matches!(self.peek(), Some(Token::Comma)) {
                    break;
                }
                self.advance(); // eat comma
                if matches!(self.peek(), Some(Token::RBrace)) {
                    break; // trailing comma
                }
            }
        }
        let end = self.expect_rbrace()?;
        Ok(ArrayLiteral {
            items,
            span: start.start..end.end,
        })
    }

    // ------------------------------------------------------------------
    // Pratt expression parser
    // ------------------------------------------------------------------

    pub fn parse_expr(&mut self, min_bp: u8) -> Result<Expr<'a>> {
        let mut lhs = self.parse_primary_expr()?;

        loop {
            // Postfix: index operator
            if matches!(self.peek(), Some(Token::LBracket)) {
                let postfix_bp: u8 = 26;
                if postfix_bp < min_bp {
                    break;
                }
                let index = self.parse_index_operator()?;
                let span = lhs.span().start..index.span.end;
                lhs = Expr::Index {
                    expr: Box::new(lhs),
                    index,
                    span,
                };
                continue;
            }

            // Infix binary operators
            let Some((l_bp, r_bp)) = self.peek().and_then(Self::infix_bp) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }

            let (op_token, _) = self.advance();
            let op = Self::token_to_binop(&op_token);
            let rhs = self.parse_expr(r_bp)?;
            let span = lhs.span().start..rhs.span().end;
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
        // Prefix unary operators
        if let Some(bp) = self.peek().and_then(Self::prefix_bp) {
            let (op_token, op_span) = self.advance();
            let op = match op_token {
                Token::Minus => UnOp::Neg,
                Token::Tilde => UnOp::BitNot,
                Token::ExclamationPoint => UnOp::LogNot,
                _ => unreachable!(),
            };
            let operand = self.parse_expr(bp)?;
            let span = op_span.start..operand.span().end;
            return Ok(Expr::UnaryOp {
                op,
                operand: Box::new(operand),
                span,
            });
        }

        // Parenthesized expression
        if matches!(self.peek(), Some(Token::LParen)) {
            let start = self.advance().1;
            let inner = self.parse_expr(0)?;
            let end = self.expect_rparen()?;
            return Ok(Expr::Paren(Box::new(inner), start.start..end.end));
        }

        // DurationOf
        if matches!(self.peek(), Some(Token::Durationof)) {
            let start = self.advance().1;
            self.expect_lparen()?;
            let scope = self.parse_scope()?;
            let end = self.expect_rparen()?;
            return Ok(Expr::DurationOf {
                scope,
                span: start.start..end.end,
            });
        }

        // Type cast: (scalarType | arrayType) LPAREN expression RPAREN
        if self.peek_is_type_keyword() {
            let start = self.peek_span();
            let ty = self.parse_type_expr()?;
            self.expect_lparen()?;
            let operand = self.parse_expr(0)?;
            let end = self.expect_rparen()?;
            return Ok(Expr::Cast {
                ty: Box::new(ty),
                operand: Box::new(operand),
                span: start.start..end.end,
            });
        }

        // Identifier or function call
        if matches!(self.peek(), Some(Token::Identifier(_))) {
            let (tok, span) = self.advance();
            let Token::Identifier(name) = tok else {
                unreachable!()
            };
            let ident = Ident {
                name,
                span: span.clone(),
            };

            // Check for call: Identifier LPAREN expressionList? RPAREN
            if matches!(self.peek(), Some(Token::LParen)) {
                self.advance(); // eat (
                let args = if matches!(self.peek(), Some(Token::RParen)) {
                    vec![]
                } else {
                    self.parse_expression_list()?
                };
                let end = self.expect_rparen()?;
                return Ok(Expr::Call {
                    name: ident,
                    args,
                    span: span.start..end.end,
                });
            }

            return Ok(Expr::Ident(ident));
        }

        // Literals
        if let Some(tok) = self.peek().cloned() {
            match tok {
                Token::DecimalIntegerLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::IntLiteral(s, IntEncoding::Decimal, span));
                }
                Token::BinaryIntegerLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::IntLiteral(s, IntEncoding::Binary, span));
                }
                Token::OctalIntegerLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::IntLiteral(s, IntEncoding::Octal, span));
                }
                Token::HexIntegerLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::IntLiteral(s, IntEncoding::Hex, span));
                }
                Token::FloatLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::FloatLiteral(s, span));
                }
                Token::ImaginaryLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::ImagLiteral(s, span));
                }
                Token::True => {
                    let (_, span) = self.advance();
                    return Ok(Expr::BoolLiteral(true, span));
                }
                Token::False => {
                    let (_, span) = self.advance();
                    return Ok(Expr::BoolLiteral(false, span));
                }
                Token::BitstringLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::BitstringLiteral(s, span));
                }
                Token::TimingLiteral(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::TimingLiteral(s, span));
                }
                Token::HardwareQubit(s) => {
                    let (_, span) = self.advance();
                    return Ok(Expr::HardwareQubit(s, span));
                }
                _ => {}
            }
        }

        Err(self.error("expected expression"))
    }

    fn infix_bp(token: &Token) -> Option<(u8, u8)> {
        match token {
            Token::DoublePipe => Some((2, 3)),
            Token::DoubleAmpersand => Some((4, 5)),
            Token::Pipe => Some((6, 7)),
            Token::Caret => Some((8, 9)),
            Token::Ampersand => Some((10, 11)),
            Token::DoubleEquals | Token::ExclamationEquals => Some((12, 13)),
            Token::GreaterThan
            | Token::LessThan
            | Token::GreaterThanEquals
            | Token::LessThanEquals => Some((14, 15)),
            Token::DoubleGreater | Token::DoubleLess => Some((16, 17)),
            Token::Plus | Token::Minus => Some((18, 19)),
            Token::Asterisk | Token::Slash | Token::Percent => Some((20, 21)),
            Token::DoubleAsterisk => Some((22, 21)), // right-associative
            _ => None,
        }
    }

    fn prefix_bp(token: &Token) -> Option<u8> {
        match token {
            Token::Minus | Token::Tilde | Token::ExclamationPoint => Some(21),
            _ => None,
        }
    }

    fn token_to_binop(token: &Token) -> BinOp {
        match token {
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
            _ => unreachable!(),
        }
    }

    // ------------------------------------------------------------------
    // Index and range parsing
    // ------------------------------------------------------------------

    fn parse_index_operator(&mut self) -> Result<IndexOp<'a>> {
        let start = self.expect_lbracket()?;

        let kind = if matches!(self.peek(), Some(Token::LBrace)) {
            let (exprs, _) = self.parse_set_expression()?;
            IndexKind::Set(exprs)
        } else {
            let mut items = vec![self.parse_index_item()?];
            while matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
                if matches!(self.peek(), Some(Token::RBracket)) {
                    break;
                }
                items.push(self.parse_index_item()?);
            }
            IndexKind::Items(items)
        };

        let end = self.expect_rbracket()?;
        Ok(IndexOp {
            kind,
            span: start.start..end.end,
        })
    }

    fn parse_index_item(&mut self) -> Result<IndexItem<'a>> {
        // If starts with ':', it's a range with no start
        if matches!(self.peek(), Some(Token::Colon)) {
            self.advance();
            return Ok(IndexItem::Range(self.parse_range_after_colon(None)?));
        }

        let expr = self.parse_expr(0)?;

        // If followed by ':', it's a range
        if matches!(self.peek(), Some(Token::Colon)) {
            self.advance();
            return Ok(IndexItem::Range(self.parse_range_after_colon(Some(expr))?));
        }

        Ok(IndexItem::Single(expr))
    }

    fn parse_range_expression(&mut self) -> Result<RangeExpr<'a>> {
        let start = if !matches!(self.peek(), Some(Token::Colon)) {
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        // Expect first colon
        if !matches!(self.peek(), Some(Token::Colon)) {
            return Err(self.error("expected ':'"));
        }
        self.advance();
        self.parse_range_after_colon(start)
    }

    /// Parse the rest of a range expression after consuming the first ':'.
    /// `start` is the expression before the colon (if any).
    fn parse_range_after_colon(&mut self, start: Option<Expr<'a>>) -> Result<RangeExpr<'a>> {
        let end = if !matches!(
            self.peek(),
            Some(Token::Colon | Token::RBracket | Token::Comma) | None
        ) {
            Some(Box::new(self.parse_expr(0)?))
        } else {
            None
        };

        let step = if matches!(self.peek(), Some(Token::Colon)) {
            self.advance();
            Some(Box::new(self.parse_expr(0)?))
        } else {
            None
        };

        Ok(RangeExpr {
            start: start.map(Box::new),
            end,
            step,
        })
    }

    fn parse_set_expression(&mut self) -> Result<(Vec<Expr<'a>>, Span)> {
        let start = self.expect_lbrace()?;
        let mut exprs = vec![self.parse_expr(0)?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            if matches!(self.peek(), Some(Token::RBrace)) {
                break;
            }
            exprs.push(self.parse_expr(0)?);
        }
        let end = self.expect_rbrace()?;
        Ok((exprs, start.start..end.end))
    }

    // ------------------------------------------------------------------
    // Type parsing
    // ------------------------------------------------------------------

    fn parse_scalar_type(&mut self) -> Result<ScalarType<'a>> {
        let start = self.peek_span();
        let tok = self.peek().cloned();
        match tok {
            Some(Token::Bit) => {
                self.advance();
                let desig = self.try_parse_designator()?;
                let end = desig.as_ref().map(|d| d.span().end).unwrap_or(start.end);
                Ok(ScalarType::Bit(desig.map(Box::new), start.start..end))
            }
            Some(Token::Int) => {
                self.advance();
                let desig = self.try_parse_designator()?;
                let end = desig.as_ref().map(|d| d.span().end).unwrap_or(start.end);
                Ok(ScalarType::Int(desig.map(Box::new), start.start..end))
            }
            Some(Token::Uint) => {
                self.advance();
                let desig = self.try_parse_designator()?;
                let end = desig.as_ref().map(|d| d.span().end).unwrap_or(start.end);
                Ok(ScalarType::Uint(desig.map(Box::new), start.start..end))
            }
            Some(Token::Float) => {
                self.advance();
                let desig = self.try_parse_designator()?;
                let end = desig.as_ref().map(|d| d.span().end).unwrap_or(start.end);
                Ok(ScalarType::Float(desig.map(Box::new), start.start..end))
            }
            Some(Token::Angle) => {
                self.advance();
                let desig = self.try_parse_designator()?;
                let end = desig.as_ref().map(|d| d.span().end).unwrap_or(start.end);
                Ok(ScalarType::Angle(desig.map(Box::new), start.start..end))
            }
            Some(Token::Bool) => {
                self.advance();
                Ok(ScalarType::Bool(start))
            }
            Some(Token::Duration) => {
                self.advance();
                Ok(ScalarType::Duration(start))
            }
            Some(Token::Stretch) => {
                self.advance();
                Ok(ScalarType::Stretch(start))
            }
            Some(Token::Complex) => {
                self.advance();
                if matches!(self.peek(), Some(Token::LBracket)) {
                    self.advance();
                    let inner = self.parse_scalar_type()?;
                    let end = self.expect_rbracket()?;
                    Ok(ScalarType::Complex(
                        Some(Box::new(inner)),
                        start.start..end.end,
                    ))
                } else {
                    Ok(ScalarType::Complex(None, start))
                }
            }
            _ => Err(self.error("expected scalar type")),
        }
    }

    fn parse_qubit_type(&mut self) -> Result<QubitType<'a>> {
        let start = self.peek_span();
        if !matches!(self.peek(), Some(Token::Qubit)) {
            return Err(self.error("expected 'qubit'"));
        }
        self.advance();
        let desig = self.try_parse_designator()?;
        let end = desig.as_ref().map(|d| d.span().end).unwrap_or(start.end);
        Ok(QubitType {
            designator: desig.map(Box::new),
            span: start.start..end,
        })
    }

    fn parse_array_type(&mut self) -> Result<ArrayType<'a>> {
        let start = self.peek_span();
        if !matches!(self.peek(), Some(Token::Array)) {
            return Err(self.error("expected 'array'"));
        }
        self.advance();
        self.expect_lbracket()?;
        let element_type = self.parse_scalar_type()?;
        if !matches!(self.peek(), Some(Token::Comma)) {
            return Err(self.error("expected ','"));
        }
        self.advance();
        let dimensions = self.parse_expression_list()?;
        let end = self.expect_rbracket()?;
        Ok(ArrayType {
            element_type,
            dimensions,
            span: start.start..end.end,
        })
    }

    fn parse_array_ref_type(&mut self) -> Result<ArrayRefType<'a>> {
        let start = self.peek_span();
        let mutability = if matches!(self.peek(), Some(Token::Readonly)) {
            self.advance();
            ArrayRefMut::Readonly
        } else if matches!(self.peek(), Some(Token::Mutable)) {
            self.advance();
            ArrayRefMut::Mutable
        } else {
            return Err(self.error("expected 'readonly' or 'mutable'"));
        };
        if !matches!(self.peek(), Some(Token::Array)) {
            return Err(self.error("expected 'array'"));
        }
        self.advance();
        self.expect_lbracket()?;
        let element_type = self.parse_scalar_type()?;
        if !matches!(self.peek(), Some(Token::Comma)) {
            return Err(self.error("expected ','"));
        }
        self.advance();

        let dimensions = if matches!(self.peek(), Some(Token::Dim)) {
            self.advance();
            if !matches!(self.peek(), Some(Token::Equals)) {
                return Err(self.error("expected '='"));
            }
            self.advance();
            ArrayRefDims::Dim(self.parse_expr(0)?)
        } else {
            ArrayRefDims::ExprList(self.parse_expression_list()?)
        };

        let end = self.expect_rbracket()?;
        Ok(ArrayRefType {
            mutability,
            element_type,
            dimensions,
            span: start.start..end.end,
        })
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr<'a>> {
        if matches!(self.peek(), Some(Token::Array)) {
            Ok(TypeExpr::Array(self.parse_array_type()?))
        } else {
            Ok(TypeExpr::Scalar(self.parse_scalar_type()?))
        }
    }

    fn parse_designator(&mut self) -> Result<Expr<'a>> {
        self.expect_lbracket()?;
        let expr = self.parse_expr(0)?;
        self.expect_rbracket()?;
        Ok(expr)
    }

    fn try_parse_designator(&mut self) -> Result<Option<Expr<'a>>> {
        if matches!(self.peek(), Some(Token::LBracket)) {
            Ok(Some(self.parse_designator()?))
        } else {
            Ok(None)
        }
    }

    // ------------------------------------------------------------------
    // List and operand helpers
    // ------------------------------------------------------------------

    fn parse_expression_list(&mut self) -> Result<Vec<Expr<'a>>> {
        let mut exprs = vec![self.parse_expr(0)?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            if matches!(
                self.peek(),
                Some(Token::RParen | Token::RBracket | Token::RBrace)
            ) {
                break; // trailing comma
            }
            exprs.push(self.parse_expr(0)?);
        }
        Ok(exprs)
    }

    fn parse_identifier_list(&mut self) -> Result<Vec<Ident<'a>>> {
        let mut idents = vec![self.expect_ident()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            if matches!(self.peek(), Some(Token::RParen | Token::RBrace)) {
                break;
            }
            idents.push(self.expect_ident()?);
        }
        Ok(idents)
    }

    fn parse_indexed_identifier(&mut self) -> Result<IndexedIdent<'a>> {
        let name = self.expect_ident()?;
        let start = name.span.start;
        let mut indices = Vec::new();
        while matches!(self.peek(), Some(Token::LBracket)) {
            indices.push(self.parse_index_operator()?);
        }
        let end = indices.last().map(|i| i.span.end).unwrap_or(name.span.end);
        Ok(IndexedIdent {
            name,
            indices,
            span: start..end,
        })
    }

    fn parse_gate_operand(&mut self) -> Result<GateOperand<'a>> {
        if matches!(self.peek(), Some(Token::HardwareQubit(_))) {
            let (tok, span) = self.advance();
            let Token::HardwareQubit(s) = tok else {
                unreachable!()
            };
            Ok(GateOperand::HardwareQubit(s, span))
        } else {
            Ok(GateOperand::Indexed(self.parse_indexed_identifier()?))
        }
    }

    fn parse_gate_operand_list(&mut self) -> Result<Vec<GateOperand<'a>>> {
        let mut operands = vec![self.parse_gate_operand()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            operands.push(self.parse_gate_operand()?);
        }
        Ok(operands)
    }

    fn parse_arg_def(&mut self) -> Result<ArgDef<'a>> {
        match self.peek().cloned() {
            Some(Token::Creg) => {
                self.advance();
                let name = self.expect_ident()?;
                let desig = self.try_parse_designator()?;
                Ok(ArgDef::Creg(name, desig))
            }
            Some(Token::Qreg) => {
                self.advance();
                let name = self.expect_ident()?;
                let desig = self.try_parse_designator()?;
                Ok(ArgDef::Qreg(name, desig))
            }
            Some(Token::Readonly | Token::Mutable) => {
                let arr_ref = self.parse_array_ref_type()?;
                let name = self.expect_ident()?;
                Ok(ArgDef::ArrayRef(arr_ref, name))
            }
            Some(Token::Qubit) => {
                let ty = self.parse_qubit_type()?;
                let name = self.expect_ident()?;
                Ok(ArgDef::Qubit(ty, name))
            }
            _ => {
                let ty = self.parse_scalar_type()?;
                let name = self.expect_ident()?;
                Ok(ArgDef::Scalar(ty, name))
            }
        }
    }

    fn parse_arg_def_list(&mut self) -> Result<Vec<ArgDef<'a>>> {
        let mut items = vec![self.parse_arg_def()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            if matches!(self.peek(), Some(Token::RParen)) {
                break;
            }
            items.push(self.parse_arg_def()?);
        }
        Ok(items)
    }

    fn parse_extern_arg(&mut self) -> Result<ExternArg<'a>> {
        match self.peek().cloned() {
            Some(Token::Creg) => {
                self.advance();
                let desig = self.try_parse_designator()?;
                Ok(ExternArg::Creg(desig))
            }
            Some(Token::Readonly | Token::Mutable) => {
                Ok(ExternArg::ArrayRef(self.parse_array_ref_type()?))
            }
            _ => Ok(ExternArg::Scalar(self.parse_scalar_type()?)),
        }
    }

    fn parse_extern_arg_list(&mut self) -> Result<Vec<ExternArg<'a>>> {
        let mut items = vec![self.parse_extern_arg()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            if matches!(self.peek(), Some(Token::RParen)) {
                break;
            }
            items.push(self.parse_extern_arg()?);
        }
        Ok(items)
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
            Ok(DefcalArgDef::Expr(self.parse_expr(0)?))
        }
    }

    fn parse_defcal_arg_def_list(&mut self) -> Result<Vec<DefcalArgDef<'a>>> {
        let mut items = vec![self.parse_defcal_arg_def()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.advance();
            if matches!(self.peek(), Some(Token::RParen)) {
                break;
            }
            items.push(self.parse_defcal_arg_def()?);
        }
        Ok(items)
    }

    fn parse_defcal_operand_list(&mut self) -> Result<Vec<DefcalOperand<'a>>> {
        let mut operands = Vec::new();
        if matches!(
            self.peek(),
            Some(Token::Identifier(_) | Token::HardwareQubit(_))
        ) {
            operands.push(self.parse_defcal_operand()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
                operands.push(self.parse_defcal_operand()?);
            }
        }
        Ok(operands)
    }

    fn parse_defcal_operand(&mut self) -> Result<DefcalOperand<'a>> {
        if matches!(self.peek(), Some(Token::HardwareQubit(_))) {
            let (tok, span) = self.advance();
            let Token::HardwareQubit(s) = tok else {
                unreachable!()
            };
            Ok(DefcalOperand::HardwareQubit(s, span))
        } else {
            Ok(DefcalOperand::Ident(self.expect_ident()?))
        }
    }

    // ------------------------------------------------------------------
    // Conversion helpers
    // ------------------------------------------------------------------

    fn expr_to_indexed_ident(&self, expr: Expr<'a>) -> Result<IndexedIdent<'a>> {
        match expr {
            Expr::Ident(id) => {
                let span = id.span.clone();
                Ok(IndexedIdent {
                    name: id,
                    indices: vec![],
                    span,
                })
            }
            Expr::Index {
                expr, index, span, ..
            } => {
                let mut indexed = self.expr_to_indexed_ident(*expr)?;
                indexed.indices.push(index);
                indexed.span = span;
                Ok(indexed)
            }
            _ => Err(Error {
                span: expr.span(),
                message: "expected identifier for assignment target".into(),
            }),
        }
    }

    fn decompose_gate_head(&self, expr: Expr<'a>) -> Result<DecomposedGateHead<'a>> {
        match expr {
            Expr::Ident(id) => Ok((id, None, None)),
            Expr::Call { name, args, .. } => Ok((name, Some(args), None)),
            Expr::Index { expr, index, .. } => {
                let (name, args, prev_desig) = self.decompose_gate_head(*expr)?;
                if prev_desig.is_some() {
                    return Err(self.error("invalid gate call syntax"));
                }
                // Extract single expression from index as designator
                let desig = match index.kind {
                    IndexKind::Items(mut items) if items.len() == 1 => match items.remove(0) {
                        IndexItem::Single(e) => e,
                        _ => {
                            return Err(self.error("designator must be a single expression"));
                        }
                    },
                    _ => {
                        return Err(self.error("designator must be a single expression"));
                    }
                };
                Ok((name, args, Some(Box::new(desig))))
            }
            _ => Err(Error {
                span: expr.span(),
                message: "expected identifier for gate call".into(),
            }),
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
            StmtOrScope::Stmt(s) => s,
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
                assert!(matches!(ty, TypeExpr::Scalar(ScalarType::Int(Some(_), _))));
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
                assert!(matches!(name, GateCallName::Ident(id) if id.name == "cx"));
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
                assert!(matches!(name, GateCallName::Ident(id) if id.name == "U"));
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
            StmtKind::Cal(body) => assert!(body.is_some()),
            _ => panic!("expected cal"),
        }
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
                assert!(matches!(name, GateCallName::Gphase(_)));
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
}
