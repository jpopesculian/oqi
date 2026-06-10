use oqi_lex::Span;
use serde::{Deserialize, Serialize};

use crate::classical::Primitive;
use crate::scope::ScopeTable;
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::Type;

// ── Program structure (2.1) ──────────────────────────────────────────

pub struct Program {
    pub version: Option<String>,
    pub calibration_grammar: Option<String>,
    pub symbols: SymbolTable,
    pub scopes: ScopeTable,
    pub gates: Vec<GateDecl>,
    pub subroutines: Vec<SubroutineDecl>,
    pub externs: Vec<ExternDecl>,
    pub calibrations: Vec<CalibrationDecl>,
    pub body: Vec<Stmt>,
}

// ── Declarations (2.2) ───────────────────────────────────────────────

pub struct GateDecl {
    pub symbol: SymbolId,
    pub params: Vec<SymbolId>,
    pub qubits: Vec<SymbolId>,
    pub body: GateBody,
    pub span: Span,
}

pub struct GateBody {
    pub body: Vec<Stmt>,
}

pub struct SubroutineDecl {
    pub symbol: SymbolId,
    pub params: Vec<SubroutineParam>,
    pub return_ty: Option<Type>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

pub struct SubroutineParam {
    pub symbol: SymbolId,
    pub passing: ParamPassing,
}

pub enum ParamPassing {
    ByValue,
    QubitRef,
    ReadonlyRef,
    MutableRef,
}

pub struct ExternDecl {
    pub symbol: SymbolId,
    pub param_types: Vec<Type>,
    pub return_ty: Option<Type>,
    pub span: Span,
}

pub struct CalibrationDecl {
    pub target: CalibrationTarget,
    pub args: Vec<CalibrationArg>,
    pub operands: Vec<CalibrationOperand>,
    pub return_ty: Option<Type>,
    pub body: CalibrationBody,
    pub span: Span,
}

pub enum CalibrationTarget {
    Measure,
    Reset,
    Delay,
    Named(String),
}

pub enum CalibrationArg {
    Expr(Box<Expr>),
    Param(SymbolId),
}

pub enum CalibrationOperand {
    Hardware(usize),
    Ident(String),
}

#[derive(Clone)]
pub enum CalibrationBody {
    Opaque(String),
    OpenPulse(Vec<Stmt>),
}

// ── Statements (2.3) ─────────────────────────────────────────────────

#[derive(Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

#[derive(Clone)]
pub struct Annotation {
    pub keyword: String,
    pub content: Option<String>,
    pub span: Span,
}

#[derive(Clone)]
pub enum StmtKind {
    // --- Aliases ---
    Alias(Alias<Expr>),

    // --- Quantum operations ---
    GateCall(GateCall<Expr>),
    Measure(Measure<Expr>),
    Reset(QubitOperand<Expr>),
    Barrier(Vec<QubitOperand<Expr>>),
    Delay(Delay<Expr>),
    Box(BoxStmt),

    // --- Classical operations ---
    Assignment(Assignment<Expr>),

    // --- Control flow ---
    If(If),
    For(For),
    While(While),
    Switch(Switch),
    Break,
    Continue,
    Return(Option<RValue<Expr>>),
    End,

    // --- Misc ---
    Pragma(String),
    Cal(CalibrationBody),
    ExprStmt(Expr),
    Nop(Vec<QubitOperand<Expr>>),
}

// ── Generic statement-level payload structs ──────────────────────────
// Generic over `E` (the expression type) so the same struct can serve
// both SIR (`E = Expr`) and CFG (`E = BlockExpr`).

#[derive(Clone)]
pub struct Alias<E> {
    pub symbol: SymbolId,
    pub value: Vec<E>,
}

#[derive(Clone)]
pub struct GateCall<E> {
    pub gate: SymbolId,
    pub modifiers: Vec<GateModifier<E>>,
    pub args: Vec<E>,
    pub qubits: Vec<QubitOperand<E>>,
}

#[derive(Clone)]
pub struct Measure<E> {
    pub measure: MeasureExpr<E>,
    pub target: Option<LValue<E>>,
}

#[derive(Clone)]
pub struct Delay<E> {
    pub duration: E,
    pub operands: Vec<QubitOperand<E>>,
}

#[derive(Clone)]
pub struct BoxStmt {
    pub duration: Option<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub struct Assignment<E> {
    pub target: LValue<E>,
    pub value: RValue<E>,
}

// ── Control-flow statement structs (SIR-only) ────────────────────────

#[derive(Clone)]
pub struct If {
    pub condition: Expr,
    pub then_body: Vec<Stmt>,
    pub else_body: Option<Vec<Stmt>>,
}

#[derive(Clone)]
pub struct For {
    pub var: SymbolId,
    pub iterable: ForIterable,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub struct While {
    pub condition: Expr,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub struct Switch {
    pub target: Expr,
    pub cases: Vec<SwitchCase>,
}

// ── Expressions (2.4) ────────────────────────────────────────────────

#[derive(Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone)]
pub enum ExprKind {
    // --- Literals ---
    Literal(Primitive),

    // --- References ---
    Var(SymbolId),
    HardwareQubit(usize),

    // --- Operations ---
    Binary(Binary<Expr>),
    Unary(Unary<Expr>),
    Cast(Cast<Expr>),
    Index(Index<Expr>),
    Call(Call<Expr>),
    DurationOf(Vec<Stmt>),
    ArrayLiteral(ArrayLiteral<Expr>),
}

// ── Generic expression payload structs ───────────────────────────────

#[derive(Clone)]
pub struct Binary<E> {
    pub op: BinOp,
    pub left: Box<E>,
    pub right: Box<E>,
}

#[derive(Clone)]
pub struct Unary<E> {
    pub op: UnOp,
    pub operand: Box<E>,
}

#[derive(Clone)]
pub struct Cast<E> {
    pub target_ty: Type,
    pub operand: Box<E>,
}

#[derive(Clone)]
pub struct Index<E> {
    pub base: Box<E>,
    pub index: IndexOp<E>,
}

#[derive(Clone)]
pub struct Call<E> {
    pub callee: CallTarget,
    pub args: Vec<E>,
}

#[derive(Clone)]
pub struct ArrayLiteral<E> {
    pub items: Vec<E>,
    pub span: Span,
}

// ── Supporting types (2.5) ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    LogAnd,
    LogOr,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    BitNot,
    LogNot,
}

#[derive(Clone)]
pub enum GateModifier<E> {
    Inv,
    Pow(Box<E>),
    Ctrl(usize),
    NegCtrl(usize),
}

#[derive(Clone)]
pub enum QubitOperand<E> {
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp<E>>,
    },
    Hardware(usize),
}

#[derive(Clone)]
pub enum LValue<E> {
    Var(SymbolId),
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp<E>>,
    },
}

#[derive(Clone)]
pub struct IndexOp<E> {
    pub kind: IndexKind<E>,
    pub span: Span,
}

#[derive(Clone)]
pub enum IndexKind<E> {
    Set(Vec<E>),
    Items(Vec<IndexItem<E>>),
}

#[derive(Clone)]
pub enum IndexItem<E> {
    Single(Box<E>),
    Range(RangeExpr<E>),
}

#[derive(Clone)]
pub struct RangeExpr<E> {
    pub start: Option<Box<E>>,
    pub step: Option<Box<E>>,
    pub end: Option<Box<E>>,
}

#[derive(Clone)]
pub enum ForIterable {
    Range {
        start: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    Set(Vec<Expr>),
    Expr(Box<Expr>),
}

#[derive(Clone)]
pub struct SwitchCase {
    pub labels: SwitchLabels<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub enum SwitchLabels<E> {
    Values(Vec<E>),
    Default,
}

#[derive(Clone)]
pub struct MeasureExpr<E> {
    pub kind: MeasureExprKind<E>,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone)]
pub enum MeasureExprKind<E> {
    Measure {
        operand: QubitOperand<E>,
    },
    QuantumCall {
        callee: SymbolId,
        args: Vec<E>,
        qubits: Vec<QubitOperand<E>>,
    },
}

#[derive(Debug, Clone)]
pub enum CallTarget {
    Symbol(SymbolId),
    Intrinsic(Intrinsic),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intrinsic {
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Exp,
    Log,
    Sqrt,
    Ceiling,
    Floor,
    Mod,
    Popcount,
    Rotl,
    Rotr,
    Real,
    Imag,
    Sizeof,
}

#[derive(Clone)]
pub enum RValue<E> {
    Expr(Box<E>),
    Measure(Box<MeasureExpr<E>>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{Primitive, ValueTy, iw};
    use crate::symbol::SymbolKind;
    use crate::types::FloatWidth;

    fn stmt(kind: StmtKind, span: Span) -> Stmt {
        Stmt {
            kind,
            annotations: vec![],
            span,
        }
    }

    fn float_expr(val: f64, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Literal(Primitive::float(val)),
            ty: Type::Classical(ValueTy::float(FloatWidth::F64)),
            span,
        }
    }

    fn var_expr(sym: SymbolId, ty: Type, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Var(sym),
            ty,
            span,
        }
    }

    fn uint_expr(val: u128, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Literal(Primitive::uint(val)),
            ty: Type::Classical(ValueTy::uint(iw(64))),
            span,
        }
    }

    fn indexed_qubit(sym: SymbolId, idx: u128, span: Span) -> QubitOperand<Expr> {
        QubitOperand::Indexed {
            symbol: sym,
            indices: vec![IndexOp {
                kind: IndexKind::Items(vec![IndexItem::Single(Box::new(uint_expr(idx, span)))]),
                span,
            }],
        }
    }

    fn bare_qubit(sym: SymbolId) -> QubitOperand<Expr> {
        QubitOperand::Indexed {
            symbol: sym,
            indices: vec![],
        }
    }

    fn gate_call(
        gate: SymbolId,
        args: Vec<Expr>,
        qubits: Vec<QubitOperand<Expr>>,
        span: Span,
    ) -> Stmt {
        stmt(
            StmtKind::GateCall(GateCall {
                gate,
                modifiers: vec![],
                args,
                qubits,
            }),
            span,
        )
    }

    fn measure_assign(target: SymbolId, qubit: QubitOperand<Expr>, span: Span) -> Stmt {
        stmt(
            StmtKind::Measure(Measure {
                measure: MeasureExpr {
                    kind: MeasureExprKind::Measure { operand: qubit },
                    ty: Type::Classical(ValueTy::bit()),
                    span,
                },
                target: Some(LValue::Var(target)),
            }),
            span,
        )
    }

    /// Construct the SIR for teleport.qasm by hand — verifies that the
    /// IR type definitions are expressive enough to represent a real program.
    #[test]
    fn test_teleport_manual_construction() {
        let mut symbols = SymbolTable::new();
        let mut scopes = ScopeTable::new();

        // stdgates symbols (include "stdgates.inc")
        let h_gate = symbols.insert(
            "h".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        let cx_gate = symbols.insert(
            "cx".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        let u_gate = symbols.insert(
            "U".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        let z_gate = symbols.insert(
            "z".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        let x_gate = symbols.insert(
            "x".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );

        // qubit[3] q;
        let q = symbols.insert(
            "q".into(),
            SymbolKind::Qubit,
            Type::QubitReg(3),
            Default::default(),
            None,
        );
        // bit c0; bit c1; bit c2;
        let c0 = symbols.insert(
            "c0".into(),
            SymbolKind::Variable,
            Type::Classical(ValueTy::bit()),
            Default::default(),
            None,
        );
        let c1 = symbols.insert(
            "c1".into(),
            SymbolKind::Variable,
            Type::Classical(ValueTy::bit()),
            Default::default(),
            None,
        );
        let c2 = symbols.insert(
            "c2".into(),
            SymbolKind::Variable,
            Type::Classical(ValueTy::bit()),
            Default::default(),
            None,
        );
        // gate post q { }
        let post_gate = symbols.insert(
            "post".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
            None,
        );
        let gate_scope = scopes.create(crate::scope::ScopeKind::Gate, None, Default::default());
        let post_q = symbols.insert(
            "q".into(),
            SymbolKind::GateQubit,
            Type::Qubit,
            Default::default(),
            Some(gate_scope),
        );

        let s: Span = Default::default(); // placeholder span

        // Declarations (q, c0, c1, c2) carry no runtime effect — symbol table only.
        let _ = (c0, c1, c2);
        let body = vec![
            // reset q;
            stmt(StmtKind::Reset(bare_qubit(q)), s),
            // U(0.3, 0.2, 0.1) q[0];
            gate_call(
                u_gate,
                vec![float_expr(0.3, s), float_expr(0.2, s), float_expr(0.1, s)],
                vec![indexed_qubit(q, 0, s)],
                s,
            ),
            // h q[1];
            gate_call(h_gate, vec![], vec![indexed_qubit(q, 1, s)], s),
            // cx q[1], q[2];
            gate_call(
                cx_gate,
                vec![],
                vec![indexed_qubit(q, 1, s), indexed_qubit(q, 2, s)],
                s,
            ),
            // barrier q;
            stmt(StmtKind::Barrier(vec![bare_qubit(q)]), s),
            // cx q[0], q[1];
            gate_call(
                cx_gate,
                vec![],
                vec![indexed_qubit(q, 0, s), indexed_qubit(q, 1, s)],
                s,
            ),
            // h q[0];
            gate_call(h_gate, vec![], vec![indexed_qubit(q, 0, s)], s),
            // c0 = measure q[0];
            measure_assign(c0, indexed_qubit(q, 0, s), s),
            // c1 = measure q[1];
            measure_assign(c1, indexed_qubit(q, 1, s), s),
            // if (c0 == 1) z q[2];
            stmt(
                StmtKind::If(If {
                    condition: Expr {
                        kind: ExprKind::Binary(Binary {
                            op: BinOp::Eq,
                            left: Box::new(var_expr(c0, Type::Classical(ValueTy::bit()), s)),
                            right: Box::new(uint_expr(1, s)),
                        }),
                        ty: Type::Classical(ValueTy::bool()),
                        span: s,
                    },
                    then_body: vec![gate_call(z_gate, vec![], vec![indexed_qubit(q, 2, s)], s)],
                    else_body: None,
                }),
                s,
            ),
            // if (c1 == 1) { x q[2]; }
            stmt(
                StmtKind::If(If {
                    condition: Expr {
                        kind: ExprKind::Binary(Binary {
                            op: BinOp::Eq,
                            left: Box::new(var_expr(c1, Type::Classical(ValueTy::bit()), s)),
                            right: Box::new(uint_expr(1, s)),
                        }),
                        ty: Type::Classical(ValueTy::bool()),
                        span: s,
                    },
                    then_body: vec![gate_call(x_gate, vec![], vec![indexed_qubit(q, 2, s)], s)],
                    else_body: None,
                }),
                s,
            ),
            // post q[2];
            gate_call(post_gate, vec![], vec![indexed_qubit(q, 2, s)], s),
            // c2 = measure q[2];
            measure_assign(c2, indexed_qubit(q, 2, s), s),
        ];

        let gates = vec![GateDecl {
            symbol: post_gate,
            params: vec![],
            qubits: vec![post_q],
            body: GateBody { body: vec![] },
            span: s,
        }];

        let program = Program {
            version: Some("3".into()),
            calibration_grammar: None,
            symbols,
            scopes,
            gates,
            subroutines: vec![],
            externs: vec![],
            calibrations: vec![],
            body,
        };

        // Verify structure
        assert_eq!(program.version.as_deref(), Some("3"));
        assert_eq!(program.gates.len(), 1);
        assert_eq!(program.body.len(), 13);
        assert_eq!(program.symbols.len(), 11); // 5 stdgates + q + c0 + c1 + c2 + post + post_q

        // Verify gate decl
        assert_eq!(program.symbols.get(program.gates[0].symbol).name, "post");
        assert!(program.gates[0].body.body.is_empty());

        // Verify first body stmt is reset (declarations are symbol-table only)
        assert!(matches!(program.body[0].kind, StmtKind::Reset(_)));

        // Verify last stmt is a measure into c2
        assert!(matches!(
            program.body[12].kind,
            StmtKind::Measure(Measure {
                target: Some(_),
                ..
            })
        ));

        // Verify an if-statement is present
        assert!(matches!(
            program.body[9].kind,
            StmtKind::If(If {
                else_body: None,
                ..
            })
        ));
    }
}
