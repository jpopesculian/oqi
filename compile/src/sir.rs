use oqi_lex::Span;

use crate::classical::Value;
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::Type;

// ── Program structure (2.1) ──────────────────────────────────────────

pub struct Program {
    pub version: Option<String>,
    pub calibration_grammar: Option<String>,
    pub symbols: SymbolTable,
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

pub enum CalibrationBody {
    Opaque(String),
}

// ── Statements (2.3) ─────────────────────────────────────────────────

pub struct Stmt {
    pub kind: StmtKind,
    pub annotations: Vec<Annotation>,
    pub span: Span,
}

pub struct Annotation {
    pub keyword: String,
    pub content: Option<String>,
    pub span: Span,
}

pub enum StmtKind {
    // --- Declarations (inline, have runtime effects) ---
    ClassicalDecl {
        symbol: SymbolId,
        init: Option<DeclInit>,
    },
    ConstDecl {
        symbol: SymbolId,
        init: Expr,
    },
    QubitDecl {
        symbol: SymbolId,
    },
    IoDecl {
        symbol: SymbolId,
        dir: IoDir,
    },
    Alias {
        symbol: SymbolId,
        value: Vec<Expr>,
    },

    // --- Quantum operations ---
    GateCall {
        gate: GateCallTarget,
        modifiers: Vec<GateModifier>,
        args: Vec<Expr>,
        qubits: Vec<QubitOperand>,
    },
    Measure {
        measure: MeasureExpr,
        target: Option<LValue>,
    },
    Reset {
        operand: QubitOperand,
    },
    Barrier {
        operands: Vec<QubitOperand>,
    },
    Delay {
        duration: Expr,
        operands: Vec<QubitOperand>,
    },
    Box {
        duration: Option<Expr>,
        body: Vec<Stmt>,
    },

    // --- Classical operations ---
    Assignment {
        target: LValue,
        op: AssignOp,
        value: AssignValue,
    },

    // --- Control flow ---
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    For {
        var: SymbolId,
        iterable: ForIterable,
        body: Vec<Stmt>,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    Switch {
        target: Expr,
        cases: Vec<SwitchCase>,
    },
    Break,
    Continue,
    Return(Option<ReturnValue>),
    End,

    // --- Misc ---
    Pragma(String),
    Cal {
        body: CalibrationBody,
    },
    ExprStmt(Expr),
    Nop {
        operands: Vec<QubitOperand>,
    },
}

// ── Expressions (2.4) ────────────────────────────────────────────────

pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type,
    pub span: Span,
}

pub enum ExprKind {
    // --- Literals ---
    Literal(Value),

    // --- References ---
    Var(SymbolId),
    HardwareQubit(usize),

    // --- Operations ---
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnOp,
        operand: Box<Expr>,
    },
    Cast {
        target_ty: Type,
        operand: Box<Expr>,
    },
    Index {
        base: Box<Expr>,
        index: IndexOp,
    },
    Call {
        callee: CallTarget,
        args: Vec<Expr>,
    },
    DurationOf(Vec<Stmt>),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    PowAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

pub enum GateModifier {
    Inv,
    Pow(Box<Expr>),
    Ctrl(usize),
    NegCtrl(usize),
}

pub enum GateCallTarget {
    Symbol(SymbolId),
    GPhase,
}

pub enum QubitOperand {
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp>,
    },
    Hardware(usize),
}

pub enum LValue {
    Var(SymbolId),
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp>,
    },
}

pub struct IndexOp {
    pub kind: IndexKind,
    pub span: Span,
}

pub enum IndexKind {
    Set(Vec<Expr>),
    Items(Vec<IndexItem>),
}

pub enum IndexItem {
    Single(Box<Expr>),
    Range(RangeExpr),
}

pub struct RangeExpr {
    pub start: Option<Box<Expr>>,
    pub step: Option<Box<Expr>>,
    pub end: Option<Box<Expr>>,
}

pub enum ForIterable {
    Range {
        start: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    Set(Vec<Expr>),
    Expr(Box<Expr>),
}

pub struct SwitchCase {
    pub labels: SwitchLabels,
    pub body: Vec<Stmt>,
}

pub enum SwitchLabels {
    Values(Vec<Expr>),
    Default,
}

pub struct MeasureExpr {
    pub kind: MeasureExprKind,
    pub span: Span,
}

pub enum MeasureExprKind {
    Measure {
        operand: QubitOperand,
    },
    QuantumCall {
        callee: SymbolId,
        args: Vec<Expr>,
        qubits: Vec<QubitOperand>,
    },
}

pub enum IoDir {
    Input,
    Output,
}

#[derive(Debug)]
pub enum CallTarget {
    Symbol(SymbolId),
    Intrinsic(Intrinsic),
}

#[derive(Debug)]
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

pub enum AssignValue {
    Expr(Box<Expr>),
    Measure(MeasureExpr),
}

pub enum ReturnValue {
    Expr(Box<Expr>),
    Measure(MeasureExpr),
}

pub enum DeclInit {
    Expr(Box<Expr>),
    Measure(MeasureExpr),
    ArrayLiteral(ArrayLiteral),
}

pub struct ArrayLiteral {
    pub items: Vec<ArrayLiteralItem>,
    pub span: Span,
}

pub enum ArrayLiteralItem {
    Expr(Box<Expr>),
    Nested(ArrayLiteral),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{float_value, uint_value};
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
            kind: ExprKind::Literal(float_value(val, FloatWidth::F64)),
            ty: Type::float(FloatWidth::F64),
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
            kind: ExprKind::Literal(uint_value(val, 64)),
            ty: Type::int(64, false),
            span,
        }
    }

    fn indexed_qubit(sym: SymbolId, idx: u128, span: Span) -> QubitOperand {
        QubitOperand::Indexed {
            symbol: sym,
            indices: vec![IndexOp {
                kind: IndexKind::Items(vec![IndexItem::Single(Box::new(uint_expr(idx, span)))]),
                span,
            }],
        }
    }

    fn bare_qubit(sym: SymbolId) -> QubitOperand {
        QubitOperand::Indexed {
            symbol: sym,
            indices: vec![],
        }
    }

    fn gate_call(gate: SymbolId, args: Vec<Expr>, qubits: Vec<QubitOperand>, span: Span) -> Stmt {
        stmt(
            StmtKind::GateCall {
                gate: GateCallTarget::Symbol(gate),
                modifiers: vec![],
                args,
                qubits,
            },
            span,
        )
    }

    fn measure_assign(target: SymbolId, qubit: QubitOperand, span: Span) -> Stmt {
        stmt(
            StmtKind::Assignment {
                target: LValue::Var(target),
                op: AssignOp::Assign,
                value: AssignValue::Measure(MeasureExpr {
                    kind: MeasureExprKind::Measure { operand: qubit },
                    span,
                }),
            },
            span,
        )
    }

    /// Construct the SIR for teleport.qasm by hand — verifies that the
    /// IR type definitions are expressive enough to represent a real program.
    #[test]
    fn test_teleport_manual_construction() {
        let mut symbols = SymbolTable::new();

        // stdgates symbols (include "stdgates.inc")
        let h_gate = symbols.insert("h".into(), SymbolKind::Gate, Type::Void, Default::default());
        let cx_gate = symbols.insert(
            "cx".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
        );
        let u_gate = symbols.insert("U".into(), SymbolKind::Gate, Type::Void, Default::default());
        let z_gate = symbols.insert("z".into(), SymbolKind::Gate, Type::Void, Default::default());
        let x_gate = symbols.insert("x".into(), SymbolKind::Gate, Type::Void, Default::default());

        // qubit[3] q;
        let q = symbols.insert(
            "q".into(),
            SymbolKind::Qubit,
            Type::QubitReg(3),
            Default::default(),
        );
        // bit c0; bit c1; bit c2;
        let c0 = symbols.insert(
            "c0".into(),
            SymbolKind::Variable,
            Type::bit(),
            Default::default(),
        );
        let c1 = symbols.insert(
            "c1".into(),
            SymbolKind::Variable,
            Type::bit(),
            Default::default(),
        );
        let c2 = symbols.insert(
            "c2".into(),
            SymbolKind::Variable,
            Type::bit(),
            Default::default(),
        );
        // gate post q { }
        let post_gate = symbols.insert(
            "post".into(),
            SymbolKind::Gate,
            Type::Void,
            Default::default(),
        );
        let post_q = symbols.insert(
            "q".into(),
            SymbolKind::GateQubit,
            Type::Qubit,
            Default::default(),
        );

        let s: Span = Default::default(); // placeholder span

        let body = vec![
            // qubit[3] q;
            stmt(StmtKind::QubitDecl { symbol: q }, s),
            // bit c0;
            stmt(
                StmtKind::ClassicalDecl {
                    symbol: c0,
                    init: None,
                },
                s,
            ),
            // bit c1;
            stmt(
                StmtKind::ClassicalDecl {
                    symbol: c1,
                    init: None,
                },
                s,
            ),
            // bit c2;
            stmt(
                StmtKind::ClassicalDecl {
                    symbol: c2,
                    init: None,
                },
                s,
            ),
            // reset q;
            stmt(
                StmtKind::Reset {
                    operand: bare_qubit(q),
                },
                s,
            ),
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
            stmt(
                StmtKind::Barrier {
                    operands: vec![bare_qubit(q)],
                },
                s,
            ),
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
                StmtKind::If {
                    condition: Expr {
                        kind: ExprKind::Binary {
                            op: BinOp::Eq,
                            left: Box::new(var_expr(c0, Type::bit(), s)),
                            right: Box::new(uint_expr(1, s)),
                        },
                        ty: Type::bool(),
                        span: s,
                    },
                    then_body: vec![gate_call(z_gate, vec![], vec![indexed_qubit(q, 2, s)], s)],
                    else_body: None,
                },
                s,
            ),
            // if (c1 == 1) { x q[2]; }
            stmt(
                StmtKind::If {
                    condition: Expr {
                        kind: ExprKind::Binary {
                            op: BinOp::Eq,
                            left: Box::new(var_expr(c1, Type::bit(), s)),
                            right: Box::new(uint_expr(1, s)),
                        },
                        ty: Type::bool(),
                        span: s,
                    },
                    then_body: vec![gate_call(x_gate, vec![], vec![indexed_qubit(q, 2, s)], s)],
                    else_body: None,
                },
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
            gates,
            subroutines: vec![],
            externs: vec![],
            calibrations: vec![],
            body,
        };

        // Verify structure
        assert_eq!(program.version.as_deref(), Some("3"));
        assert_eq!(program.gates.len(), 1);
        assert_eq!(program.body.len(), 17);
        assert_eq!(program.symbols.len(), 11); // 5 stdgates + q + c0 + c1 + c2 + post + post_q

        // Verify gate decl
        assert_eq!(program.symbols.get(program.gates[0].symbol).name, "post");
        assert!(program.gates[0].body.body.is_empty());

        // Verify first body stmt is qubit decl
        assert!(matches!(program.body[0].kind, StmtKind::QubitDecl { .. }));

        // Verify last stmt is measure-assign
        assert!(matches!(
            program.body[16].kind,
            StmtKind::Assignment {
                value: AssignValue::Measure(_),
                ..
            }
        ));

        // Verify an if-statement is present
        assert!(matches!(
            program.body[13].kind,
            StmtKind::If {
                else_body: None,
                ..
            }
        ));
    }
}
