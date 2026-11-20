use oqi_lex::Span;
use serde::{Deserialize, Serialize};

use crate::classical::Primitive;
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
    pub body: Vec<Stmt>,
    pub span: Span,
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
    /// `gate[duration]` designator (docs/delays.rst: stretchy/timed
    /// instructions). Compile-time timing only: dropped at CFG conversion,
    /// never reaches bytecode or the VM.
    pub duration: Option<E>,
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
    pub labels: SwitchLabels,
    pub body: Vec<Stmt>,
}

#[derive(Clone)]
pub enum SwitchLabels {
    Values(Vec<Expr>),
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
