use oqi_classical::ops;
use oqi_lex::Span;
use serde::{Deserialize, Serialize};

pub use oqi_classical::ops::{BinOp, UnOp};

use crate::classical::{Primitive, Value};
use crate::symbol::{SymbolId, SymbolTable};
use crate::types::Type;

// ── Program structure (2.1) ──────────────────────────────────────────

pub struct Program {
    pub version: Option<String>,
    /// The `defcalgrammar` name, without its quotes.
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
    pub body: Vec<Stmt>,
    pub span: Span,
}

pub struct SubroutineDecl {
    pub symbol: SymbolId,
    /// Parameters in declaration order. How each is passed follows from its
    /// type: qubits and arrays by reference, everything else by value.
    pub params: Vec<SymbolId>,
    pub return_ty: Option<Type>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

pub struct ExternDecl {
    pub symbol: SymbolId,
    pub param_types: Vec<Type>,
    pub return_ty: Option<Type>,
    pub span: Span,
}

#[derive(Clone)]
pub struct CalibrationDecl {
    pub target: CalibrationTarget,
    pub args: Vec<CalibrationArg>,
    pub operands: Vec<CalibrationOperand>,
    pub return_ty: Option<Type>,
    pub body: CalibrationBody,
    pub span: Span,
}

#[derive(Clone)]
pub enum CalibrationTarget {
    Measure,
    Reset,
    Delay,
    Named(SymbolId),
}

#[derive(Clone)]
pub enum CalibrationArg {
    Expr(Box<Expr>),
    Param(SymbolId),
}

#[derive(Clone)]
pub enum CalibrationOperand {
    Hardware(usize),
    Ident(SymbolId),
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
    Alias(Alias),

    // --- Quantum operations ---
    GateCall(GateCall),
    Measure(Measure),
    Reset(QubitOperand),
    Barrier(Vec<QubitOperand>),
    Delay(Delay),
    Box(BoxStmt),

    // --- Classical operations ---
    Assignment(Assignment),

    // --- Control flow ---
    If(If),
    For(For),
    While(While),
    Switch(Switch),
    Break,
    Continue,
    Return(Option<RValue>),
    End,

    // --- Misc ---
    Pragma(String),
    Cal(CalibrationBody),
    ExprStmt(Expr),
    Nop(Vec<QubitOperand>),
}

#[derive(Clone)]
pub struct Alias {
    pub symbol: SymbolId,
    pub value: Vec<Expr>,
}

#[derive(Clone)]
pub struct GateCall {
    pub gate: SymbolId,
    pub modifiers: Vec<GateModifier>,
    pub args: Vec<Expr>,
    pub qubits: Vec<QubitOperand>,
    /// `gate[duration]` designator (docs/delays.rst: stretchy/timed
    /// instructions). Compile-time timing only: never reaches bytecode or
    /// the VM.
    pub duration: Option<Expr>,
}

#[derive(Clone)]
pub struct Measure {
    pub measure: MeasureExpr,
    pub target: Option<LValue>,
}

#[derive(Clone)]
pub struct Delay {
    pub duration: Expr,
    pub operands: Vec<QubitOperand>,
}

#[derive(Clone)]
pub struct BoxStmt {
    pub duration: Option<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub struct Assignment {
    pub target: LValue,
    pub value: RValue,
}

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
    Binary(Binary),
    Unary(Unary),
    Cast(Cast),
    Index(Index),
    Call(Call),
    /// The duration of a scope. Its statements are a nested body, not
    /// sub-expressions: [`Expr::children`] does not yield them.
    DurationOf(Vec<Stmt>),
    ArrayLiteral(ArrayLiteral),
}

#[derive(Clone)]
pub struct Binary {
    pub op: BinOp,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Clone)]
pub struct Unary {
    pub op: UnOp,
    pub operand: Box<Expr>,
}

#[derive(Clone)]
pub struct Cast {
    pub target_ty: Type,
    pub operand: Box<Expr>,
}

#[derive(Clone)]
pub struct Index {
    pub base: Box<Expr>,
    pub index: IndexOp,
}

#[derive(Clone)]
pub struct Call {
    pub callee: CallTarget,
    pub args: Vec<Expr>,
}

#[derive(Clone)]
pub struct ArrayLiteral {
    pub items: Vec<Expr>,
    pub span: Span,
}

// ── Supporting types (2.5) ───────────────────────────────────────────

#[derive(Clone)]
pub enum GateModifier {
    Inv,
    Pow(Box<Expr>),
    Ctrl(usize),
    NegCtrl(usize),
}

#[derive(Clone)]
pub enum QubitOperand {
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp>,
    },
    Hardware(usize),
}

#[derive(Clone)]
pub enum LValue {
    Var(SymbolId),
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp>,
    },
}

impl LValue {
    /// The variable assigned to, in whole or in part.
    pub fn symbol(&self) -> SymbolId {
        match self {
            LValue::Var(symbol) | LValue::Indexed { symbol, .. } => *symbol,
        }
    }
}

/// One subscript: `[i]`, `[i, j]`, `[a:b]` or `[{a, b}]`.
#[derive(Clone)]
pub struct IndexOp {
    pub kind: IndexKind,
    pub span: Span,
}

#[derive(Clone)]
pub enum IndexKind {
    Set(Vec<Expr>),
    Items(Vec<IndexItem>),
}

#[derive(Clone)]
pub enum IndexItem {
    Single(Box<Expr>),
    Range(RangeExpr),
}

#[derive(Clone)]
pub struct RangeExpr {
    pub start: Option<Box<Expr>>,
    pub step: Option<Box<Expr>>,
    pub end: Option<Box<Expr>>,
}

#[derive(Clone)]
pub enum ForIterable {
    Range(RangeExpr),
    Set(Vec<Expr>),
    Expr(Box<Expr>),
}

#[derive(Clone)]
pub struct SwitchCase {
    pub labels: SwitchLabels,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub enum SwitchLabels {
    Values(Vec<Expr>),
    Default,
}

#[derive(Clone)]
pub struct MeasureExpr {
    pub kind: MeasureExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone)]
pub enum MeasureExprKind {
    Measure {
        operand: QubitOperand,
    },
    /// A subroutine called on qubit operands for its value:
    /// `bit b = xmeasure q;`.
    QuantumCall {
        callee: SymbolId,
        args: Vec<Expr>,
        qubits: Vec<QubitOperand>,
    },
}

#[derive(Debug, Clone)]
pub enum CallTarget {
    Symbol(SymbolId),
    Intrinsic(Intrinsic),
}

/// A built-in function, as the classical operation it performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intrinsic {
    Unary(UnOp),
    Binary(BinOp),
    Sizeof,
}

impl Intrinsic {
    /// Every built-in function (docs/types.rst, "Built-in constant
    /// expression functions", plus `real`, `imag` and `sizeof`).
    const ALL: [Intrinsic; 18] = [
        Intrinsic::Unary(UnOp::Sin),
        Intrinsic::Unary(UnOp::Cos),
        Intrinsic::Unary(UnOp::Tan),
        Intrinsic::Unary(UnOp::Arcsin),
        Intrinsic::Unary(UnOp::Arccos),
        Intrinsic::Unary(UnOp::Arctan),
        Intrinsic::Unary(UnOp::Exp),
        Intrinsic::Unary(UnOp::Log),
        Intrinsic::Unary(UnOp::Sqrt),
        Intrinsic::Unary(UnOp::Ceiling),
        Intrinsic::Unary(UnOp::Floor),
        Intrinsic::Unary(UnOp::Popcount),
        Intrinsic::Unary(UnOp::Real),
        Intrinsic::Unary(UnOp::Imag),
        Intrinsic::Binary(BinOp::Mod),
        Intrinsic::Binary(BinOp::Rotl),
        Intrinsic::Binary(BinOp::Rotr),
        Intrinsic::Sizeof,
    ];

    /// The name the built-in is called by.
    pub const fn name(self) -> &'static str {
        match self {
            Intrinsic::Unary(op) => op.name(),
            // `mod(a, b)` is the function spelling of `a % b`.
            Intrinsic::Binary(BinOp::Mod) => "mod",
            Intrinsic::Binary(op) => op.name(),
            Intrinsic::Sizeof => "sizeof",
        }
    }

    /// The built-in called `name`, if there is one.
    pub fn lookup(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|i| i.name() == name)
    }

    /// Evaluate the built-in on its arguments; `None` if it does not take
    /// that many.
    pub fn apply(self, args: Vec<Value>) -> Option<oqi_classical::Result<Value>> {
        let mut args = args.into_iter();
        Some(match (self, args.next(), args.next(), args.next()) {
            (Intrinsic::Unary(op), Some(arg), None, None) => op.apply(arg),
            (Intrinsic::Binary(op), Some(lhs), Some(rhs), None) => op.apply(lhs, rhs),
            (Intrinsic::Sizeof, Some(value), dim, None) => ops::sizeof(value.ty(), dim),
            _ => return None,
        })
    }
}

#[derive(Clone)]
pub enum RValue {
    Expr(Box<Expr>),
    Measure(Box<MeasureExpr>),
}

// ── Traversal ────────────────────────────────────────────────────────

/// The tree-walking helpers, in a shared and a `mut` flavour: the
/// expressions an expression or statement holds directly, and the nested
/// statement lists of a statement. A pass recurses through these and only
/// spells out the node kinds it treats specially.
macro_rules! walkers {
    ($children:ident, $exprs:ident, $bodies:ident, $collect:ident $(, $m:ident)?) => {
        impl Expr {
            /// The sub-expressions this expression is built from directly, in
            /// evaluation order.
            pub fn $children(& $($m)? self) -> Vec<& $($m)? Expr> {
                let mut out = Vec::new();
                match & $($m)? self.kind {
                    ExprKind::Literal(_)
                    | ExprKind::Var(_)
                    | ExprKind::HardwareQubit(_)
                    | ExprKind::DurationOf(_) => {}
                    ExprKind::Binary(b) => {
                        out.push(& $($m)? *b.left);
                        out.push(& $($m)? *b.right);
                    }
                    ExprKind::Unary(u) => out.push(& $($m)? *u.operand),
                    ExprKind::Cast(c) => out.push(& $($m)? *c.operand),
                    ExprKind::Index(ix) => {
                        out.push(& $($m)? *ix.base);
                        ix.index.$collect(&mut out);
                    }
                    ExprKind::Call(c) => out.extend(& $($m)? c.args),
                    ExprKind::ArrayLiteral(a) => out.extend(& $($m)? a.items),
                }
                out
            }
        }

        impl Stmt {
            /// The expressions this statement holds directly: operands,
            /// conditions and index expressions, but nothing inside its
            /// nested statement lists.
            pub fn $exprs(& $($m)? self) -> Vec<& $($m)? Expr> {
                let mut out = Vec::new();
                match & $($m)? self.kind {
                    StmtKind::Alias(a) => out.extend(& $($m)? a.value),
                    StmtKind::GateCall(gc) => {
                        for m in & $($m)? gc.modifiers {
                            if let GateModifier::Pow(e) = m {
                                out.push(& $($m)? **e);
                            }
                        }
                        out.extend(& $($m)? gc.args);
                        for q in & $($m)? gc.qubits {
                            q.$collect(&mut out);
                        }
                        out.extend(& $($m)? gc.duration);
                    }
                    StmtKind::Measure(m) => {
                        m.measure.$collect(&mut out);
                        if let Some(target) = & $($m)? m.target {
                            target.$collect(&mut out);
                        }
                    }
                    StmtKind::Reset(q) => q.$collect(&mut out),
                    StmtKind::Barrier(qs) | StmtKind::Nop(qs) => {
                        for q in qs {
                            q.$collect(&mut out);
                        }
                    }
                    StmtKind::Delay(d) => {
                        out.push(& $($m)? d.duration);
                        for q in & $($m)? d.operands {
                            q.$collect(&mut out);
                        }
                    }
                    StmtKind::Box(b) => out.extend(& $($m)? b.duration),
                    StmtKind::Assignment(a) => {
                        a.target.$collect(&mut out);
                        a.value.$collect(&mut out);
                    }
                    StmtKind::If(i) => out.push(& $($m)? i.condition),
                    StmtKind::For(f) => match & $($m)? f.iterable {
                        ForIterable::Range(r) => r.$collect(&mut out),
                        ForIterable::Set(items) => out.extend(items),
                        ForIterable::Expr(e) => out.push(e),
                    },
                    StmtKind::While(w) => out.push(& $($m)? w.condition),
                    StmtKind::Switch(s) => {
                        out.push(& $($m)? s.target);
                        for case in & $($m)? s.cases {
                            if let SwitchLabels::Values(values) = & $($m)? case.labels {
                                out.extend(values);
                            }
                        }
                    }
                    StmtKind::Return(Some(value)) => value.$collect(&mut out),
                    StmtKind::ExprStmt(e) => out.push(e),
                    StmtKind::Return(None)
                    | StmtKind::Break
                    | StmtKind::Continue
                    | StmtKind::End
                    | StmtKind::Pragma(_)
                    | StmtKind::Cal(_) => {}
                }
                out
            }

            /// The statement lists nested in this statement: box, branch,
            /// loop and case bodies, and a structured `cal` body.
            pub fn $bodies(& $($m)? self) -> Vec<& $($m)? Vec<Stmt>> {
                match & $($m)? self.kind {
                    StmtKind::Box(b) => vec![& $($m)? b.body],
                    StmtKind::If(i) => {
                        let mut out = vec![& $($m)? i.then_body];
                        out.extend(& $($m)? i.else_body);
                        out
                    }
                    StmtKind::For(f) => vec![& $($m)? f.body],
                    StmtKind::While(w) => vec![& $($m)? w.body],
                    StmtKind::Switch(s) => {
                        (& $($m)? s.cases).into_iter().map(|c| & $($m)? c.body).collect()
                    }
                    StmtKind::Cal(CalibrationBody::OpenPulse(body)) => vec![body],
                    _ => Vec::new(),
                }
            }
        }

        impl RangeExpr {
            fn $collect<'a>(&'a $($m)? self, out: &mut Vec<&'a $($m)? Expr>) {
                let bounds = [& $($m)? self.start, & $($m)? self.step, & $($m)? self.end];
                for bound in bounds.into_iter().flatten() {
                    out.push(bound);
                }
            }
        }

        impl IndexOp {
            fn $collect<'a>(&'a $($m)? self, out: &mut Vec<&'a $($m)? Expr>) {
                match & $($m)? self.kind {
                    IndexKind::Set(items) => out.extend(items),
                    IndexKind::Items(items) => {
                        for item in items {
                            match item {
                                IndexItem::Single(e) => out.push(e),
                                IndexItem::Range(r) => r.$collect(out),
                            }
                        }
                    }
                }
            }
        }

        impl QubitOperand {
            fn $collect<'a>(&'a $($m)? self, out: &mut Vec<&'a $($m)? Expr>) {
                if let QubitOperand::Indexed { indices, .. } = self {
                    for index in indices {
                        index.$collect(out);
                    }
                }
            }
        }

        impl LValue {
            fn $collect<'a>(&'a $($m)? self, out: &mut Vec<&'a $($m)? Expr>) {
                if let LValue::Indexed { indices, .. } = self {
                    for index in indices {
                        index.$collect(out);
                    }
                }
            }
        }

        impl MeasureExpr {
            fn $collect<'a>(&'a $($m)? self, out: &mut Vec<&'a $($m)? Expr>) {
                match & $($m)? self.kind {
                    MeasureExprKind::Measure { operand } => operand.$collect(out),
                    MeasureExprKind::QuantumCall { args, qubits, .. } => {
                        out.extend(args);
                        for q in qubits {
                            q.$collect(out);
                        }
                    }
                }
            }
        }

        impl RValue {
            fn $collect<'a>(&'a $($m)? self, out: &mut Vec<&'a $($m)? Expr>) {
                match self {
                    RValue::Expr(e) => out.push(e),
                    RValue::Measure(m) => m.$collect(out),
                }
            }
        }
    };
}

walkers!(children, exprs, bodies, collect_exprs);
walkers!(children_mut, exprs_mut, bodies_mut, collect_exprs_mut, mut);
