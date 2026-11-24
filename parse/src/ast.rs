pub use oqi_lex::Span;

// ---- Comments ----

/// A line (`// …`) or block (`/* … */`) comment, delimiters included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comment<'a> {
    pub text: &'a str,
    pub span: Span,
}

// ---- Identifiers ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident<'a> {
    pub name: &'a str,
    pub span: Span,
}

// ---- Program structure ----

#[derive(Debug, Clone, PartialEq)]
pub struct Program<'a> {
    pub version: Option<Version<'a>>,
    pub body: Vec<StmtOrScope<'a>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Version<'a> {
    pub specifier: &'a str,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtOrScope<'a> {
    Stmt(Box<Stmt<'a>>),
    Scope(Scope<'a>),
}

impl StmtOrScope<'_> {
    pub fn span(&self) -> Span {
        match self {
            StmtOrScope::Stmt(stmt) => stmt.span,
            StmtOrScope::Scope(scope) => scope.span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scope<'a> {
    pub body: Vec<StmtOrScope<'a>>,
    pub span: Span,
}

/// An annotation such as `@bind $0` on the statement that follows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation<'a> {
    /// The keyword, including the `@`.
    pub keyword: &'a str,
    /// The rest of the line after the keyword, if any.
    pub content: Option<&'a str>,
    pub span: Span,
}

// ---- Statements ----

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt<'a> {
    pub annotations: Vec<Annotation<'a>>,
    pub kind: StmtKind<'a>,
    pub span: Span,
}

/// A quoted string, as in `include "stdgates.inc";`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringLiteral<'a> {
    /// The text between the quotes.
    pub value: &'a str,
    /// The quote it was written with: `"` or `'`.
    pub quote: char,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind<'a> {
    Pragma(&'a str),
    CalibrationGrammar(StringLiteral<'a>),
    Include(StringLiteral<'a>),
    Break,
    Continue,
    End,
    For {
        ty: ScalarType<'a>,
        var: Ident<'a>,
        iterable: ForIterable<'a>,
        body: Box<StmtOrScope<'a>>,
    },
    If {
        condition: Expr<'a>,
        then_body: Box<StmtOrScope<'a>>,
        else_body: Option<Box<StmtOrScope<'a>>>,
    },
    Return(Option<ExprOrMeasure<'a>>),
    While {
        condition: Expr<'a>,
        body: Box<StmtOrScope<'a>>,
    },
    Switch {
        target: Expr<'a>,
        cases: Vec<SwitchCase<'a>>,
    },
    Barrier(Vec<GateOperand<'a>>),
    Box {
        designator: Option<Expr<'a>>,
        body: Scope<'a>,
    },
    Delay {
        designator: Expr<'a>,
        operands: Vec<GateOperand<'a>>,
    },
    Nop(Vec<GateOperand<'a>>),
    GateCall {
        modifiers: Vec<GateModifier<'a>>,
        name: Ident<'a>,
        args: Option<Vec<Expr<'a>>>,
        designator: Option<Box<Expr<'a>>>,
        operands: Vec<GateOperand<'a>>,
    },
    MeasureArrow {
        measure: MeasureExpr<'a>,
        target: Option<IndexedIdent<'a>>,
    },
    Reset(GateOperand<'a>),
    Alias {
        name: Ident<'a>,
        value: Vec<Expr<'a>>,
    },
    ClassicalDecl {
        ty: TypeExpr<'a>,
        name: Ident<'a>,
        init: Option<ExprOrMeasure<'a>>,
    },
    ConstDecl {
        ty: TypeExpr<'a>,
        name: Ident<'a>,
        init: ExprOrMeasure<'a>,
    },
    IoDecl {
        dir: IoDir,
        ty: TypeExpr<'a>,
        name: Ident<'a>,
    },
    OldStyleDecl {
        keyword: OldStyleKind,
        name: Ident<'a>,
        designator: Option<Box<Expr<'a>>>,
    },
    QuantumDecl {
        ty: QubitType<'a>,
        name: Ident<'a>,
    },
    Def {
        name: Ident<'a>,
        params: Vec<ArgDef<'a>>,
        return_ty: Option<ScalarType<'a>>,
        body: Scope<'a>,
    },
    Extern {
        name: Ident<'a>,
        params: Vec<ExternArg<'a>>,
        return_ty: Option<ScalarType<'a>>,
    },
    Gate {
        name: Ident<'a>,
        params: Vec<Ident<'a>>,
        qubits: Vec<Ident<'a>>,
        body: Scope<'a>,
    },
    Assignment {
        target: IndexedIdent<'a>,
        op: AssignOp,
        value: ExprOrMeasure<'a>,
    },
    Expr(Expr<'a>),
    Cal(CalBody<'a>),
    Defcal {
        target: DefcalTarget<'a>,
        args: Vec<DefcalArgDef<'a>>,
        operands: Vec<DefcalOperand<'a>>,
        return_ty: Option<ScalarType<'a>>,
        body: CalBody<'a>,
    },
    /// `extern frame Identifier ;` — OpenPulse-only.
    ExternFrame {
        name: Ident<'a>,
    },
    /// `extern port Identifier ;` — OpenPulse-only.
    ExternPort {
        name: Ident<'a>,
    },
}

/// Body of a `cal` or `defcal` block.
///
/// `Raw` is the default — the body is captured verbatim because we don't know
/// the calibration grammar. `OpenPulse` is populated only when the program has
/// declared `defcalgrammar "openpulse";`, in which case the body is re-parsed
/// against the OpenPulse grammar and stored as structured statements.
#[derive(Debug, Clone, PartialEq)]
pub enum CalBody<'a> {
    Raw(Option<&'a str>),
    OpenPulse(Vec<StmtOrScope<'a>>),
}

// ---- Expressions ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl BinOp {
    /// How tightly the operator binds; a higher precedence binds tighter.
    pub const fn precedence(self) -> u8 {
        match self {
            BinOp::LogOr => 2,
            BinOp::LogAnd => 4,
            BinOp::BitOr => 6,
            BinOp::BitXor => 8,
            BinOp::BitAnd => 10,
            BinOp::Eq | BinOp::Neq => 12,
            BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => 14,
            BinOp::Shl | BinOp::Shr => 16,
            BinOp::Add | BinOp::Sub => 18,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 20,
            BinOp::Pow => 22,
        }
    }

    pub const fn is_right_associative(self) -> bool {
        matches!(self, BinOp::Pow)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Pow => "**",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::LogAnd => "&&",
            BinOp::LogOr => "||",
            BinOp::Eq => "==",
            BinOp::Neq => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Lte => "<=",
            BinOp::Gte => ">=",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    BitNot,
    LogNot,
}

impl UnOp {
    /// Unary operators bind tighter than `*` and looser than `**`.
    pub const PRECEDENCE: u8 = 21;

    pub const fn as_str(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::BitNot => "~",
            UnOp::LogNot => "!",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntEncoding {
    Decimal,
    Binary,
    Octal,
    Hex,
}

/// An expression. Literal payloads are the source text, unprocessed.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Ident(Ident<'a>),
    /// A physical qubit; the text keeps the `$`, as in `$0`.
    HardwareQubit(&'a str, Span),
    /// An integer literal; the text keeps its base prefix (`0x`, `0o`,
    /// `0b`) and any `_` separators.
    IntLiteral(&'a str, IntEncoding, Span),
    /// A floating-point literal, as written (`1.5e-3`, `.5`, `1_000.0`).
    FloatLiteral(&'a str, Span),
    /// An imaginary literal; the text keeps the `im` suffix, as in `2.5im`.
    ImaginaryLiteral(&'a str, Span),
    BoolLiteral(bool, Span),
    /// A bitstring literal; the text keeps its quotes and any `_`
    /// separators, as in `"0101_1100"`.
    BitstringLiteral(&'a str, Span),
    /// A duration literal; the text keeps its unit, as in `100ns` or `4dt`.
    TimingLiteral(&'a str, Span),
    /// A parenthesized expression, kept so a formatter can preserve it.
    Paren(Box<Expr<'a>>, Span),
    Binary {
        left: Box<Expr<'a>>,
        op: BinOp,
        right: Box<Expr<'a>>,
        span: Span,
    },
    Unary {
        op: UnOp,
        operand: Box<Expr<'a>>,
        span: Span,
    },
    Index {
        expr: Box<Expr<'a>>,
        index: IndexOp<'a>,
        span: Span,
    },
    Call {
        name: Ident<'a>,
        args: Vec<Expr<'a>>,
        span: Span,
    },
    Cast {
        ty: Box<TypeExpr<'a>>,
        operand: Box<Expr<'a>>,
        span: Span,
    },
    DurationOf {
        scope: Scope<'a>,
        span: Span,
    },
    ArrayLiteral(ArrayLiteral<'a>),
}

impl<'a> Expr<'a> {
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident(id) => id.span,
            Expr::HardwareQubit(_, s)
            | Expr::IntLiteral(_, _, s)
            | Expr::FloatLiteral(_, s)
            | Expr::ImaginaryLiteral(_, s)
            | Expr::BoolLiteral(_, s)
            | Expr::BitstringLiteral(_, s)
            | Expr::TimingLiteral(_, s)
            | Expr::Paren(_, s)
            | Expr::Binary { span: s, .. }
            | Expr::Unary { span: s, .. }
            | Expr::Index { span: s, .. }
            | Expr::Call { span: s, .. }
            | Expr::Cast { span: s, .. }
            | Expr::DurationOf { span: s, .. } => *s,
            Expr::ArrayLiteral(al) => al.span,
        }
    }

    /// Precedence of the expression's outermost operator. Anything that is
    /// not an operator application binds tightest.
    pub fn precedence(&self) -> u8 {
        match self {
            Expr::Binary { op, .. } => op.precedence(),
            Expr::Unary { .. } => UnOp::PRECEDENCE,
            Expr::Index { .. } => IndexOp::PRECEDENCE,
            _ => u8::MAX,
        }
    }
}

// ---- Index types ----

#[derive(Debug, Clone, PartialEq)]
pub struct IndexOp<'a> {
    pub kind: IndexKind<'a>,
    pub span: Span,
}

impl IndexOp<'_> {
    /// The postfix index operator binds tighter than every other operator.
    pub const PRECEDENCE: u8 = 26;
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexKind<'a> {
    Set(Vec<Expr<'a>>),
    Items(Vec<IndexItem<'a>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexItem<'a> {
    Single(Expr<'a>),
    Range(RangeExpr<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeExpr<'a> {
    pub start: Option<Box<Expr<'a>>>,
    pub end: Option<Box<Expr<'a>>>,
    pub step: Option<Box<Expr<'a>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedIdent<'a> {
    pub name: Ident<'a>,
    pub indices: Vec<IndexOp<'a>>,
    pub span: Span,
}

// ---- Statement-specific types ----

#[derive(Debug, Clone, PartialEq)]
pub enum ForIterable<'a> {
    Set(Vec<Expr<'a>>, Span),
    Range(RangeExpr<'a>, Span),
    Expr(Expr<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SwitchCase<'a> {
    Case(Vec<Expr<'a>>, Scope<'a>),
    Default(Scope<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoDir {
    Input,
    Output,
}

impl IoDir {
    pub const fn as_str(self) -> &'static str {
        match self {
            IoDir::Input => "input",
            IoDir::Output => "output",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OldStyleKind {
    Creg,
    Qreg,
}

impl OldStyleKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            OldStyleKind::Creg => "creg",
            OldStyleKind::Qreg => "qreg",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    LeftShiftAssign,
    RightShiftAssign,
    ModAssign,
    PowAssign,
}

impl AssignOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            AssignOp::Assign => "=",
            AssignOp::AddAssign => "+=",
            AssignOp::SubAssign => "-=",
            AssignOp::MulAssign => "*=",
            AssignOp::DivAssign => "/=",
            AssignOp::BitAndAssign => "&=",
            AssignOp::BitOrAssign => "|=",
            AssignOp::BitXorAssign => "^=",
            AssignOp::LeftShiftAssign => "<<=",
            AssignOp::RightShiftAssign => ">>=",
            AssignOp::ModAssign => "%=",
            AssignOp::PowAssign => "**=",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GateModifier<'a> {
    Inv(Span),
    Pow(Expr<'a>, Span),
    Ctrl(Option<Expr<'a>>, Span),
    NegCtrl(Option<Expr<'a>>, Span),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GateOperand<'a> {
    Indexed(IndexedIdent<'a>),
    HardwareQubit(&'a str, Span),
}

impl<'a> GateOperand<'a> {
    pub fn span(&self) -> Span {
        match self {
            GateOperand::Indexed(id) => id.span,
            GateOperand::HardwareQubit(_, s) => *s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeasureExpr<'a> {
    Measure {
        operand: GateOperand<'a>,
        span: Span,
    },
    QuantumCall {
        name: Ident<'a>,
        args: Vec<Expr<'a>>,
        operands: Vec<GateOperand<'a>>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprOrMeasure<'a> {
    Expr(Expr<'a>),
    Measure(MeasureExpr<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DefcalTarget<'a> {
    Measure(Span),
    Reset(Span),
    Delay(Span),
    Ident(Ident<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DefcalArgDef<'a> {
    Expr(Expr<'a>),
    ArgDef(ArgDef<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DefcalOperand<'a> {
    HardwareQubit(&'a str, Span),
    Ident(Ident<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArgDef<'a> {
    Scalar(ScalarType<'a>, Ident<'a>),
    Qubit(QubitType<'a>, Ident<'a>),
    Creg(Ident<'a>, Option<Expr<'a>>),
    Qreg(Ident<'a>, Option<Expr<'a>>),
    ArrayRef(ArrayRefType<'a>, Ident<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternArg<'a> {
    Scalar(ScalarType<'a>),
    ArrayRef(ArrayRefType<'a>),
    Creg(Option<Expr<'a>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLiteral<'a> {
    pub items: Vec<Expr<'a>>,
    pub span: Span,
}

// ---- Type types ----

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarType<'a> {
    pub kind: ScalarKind<'a>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarKind<'a> {
    /// A type that takes an optional `[designator]`.
    Sized(SizedKind, Option<Box<Expr<'a>>>),
    /// `complex`, optionally `complex[float[n]]`.
    Complex(Option<Box<ScalarType<'a>>>),
    Bool,
    Duration,
    Stretch,
    Waveform,
    Port,
    Frame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizedKind {
    Bit,
    Int,
    Uint,
    Float,
    Angle,
}

impl SizedKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            SizedKind::Bit => "bit",
            SizedKind::Int => "int",
            SizedKind::Uint => "uint",
            SizedKind::Float => "float",
            SizedKind::Angle => "angle",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QubitType<'a> {
    pub designator: Option<Box<Expr<'a>>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayType<'a> {
    pub element_type: ScalarType<'a>,
    pub dimensions: Vec<Expr<'a>>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayRefMut {
    Readonly,
    Mutable,
}

impl ArrayRefMut {
    pub const fn as_str(self) -> &'static str {
        match self {
            ArrayRefMut::Readonly => "readonly",
            ArrayRefMut::Mutable => "mutable",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayRefType<'a> {
    pub mutability: ArrayRefMut,
    pub element_type: ScalarType<'a>,
    pub dimensions: ArrayRefDims<'a>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArrayRefDims<'a> {
    ExprList(Vec<Expr<'a>>),
    Dim(Expr<'a>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr<'a> {
    Scalar(ScalarType<'a>),
    Array(ArrayType<'a>),
}
