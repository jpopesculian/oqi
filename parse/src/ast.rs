pub use oqi_lex::Span;

// ---- Identifiers ----

#[derive(Debug, Clone)]
pub struct Ident<'a> {
    pub name: &'a str,
    pub span: Span,
}

// ---- Program structure ----

#[derive(Debug)]
pub struct Program<'a> {
    pub version: Option<Version<'a>>,
    pub body: Vec<StmtOrScope<'a>>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Version<'a> {
    pub specifier: &'a str,
    pub span: Span,
}

#[derive(Debug)]
pub enum StmtOrScope<'a> {
    Stmt(Stmt<'a>),
    Scope(Scope<'a>),
}

#[derive(Debug)]
pub struct Scope<'a> {
    pub body: Vec<StmtOrScope<'a>>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Annotation<'a> {
    pub keyword: &'a str,
    pub content: Option<&'a str>,
    pub span: Span,
}

// ---- Statements ----

#[derive(Debug)]
pub struct Stmt<'a> {
    pub annotations: Vec<Annotation<'a>>,
    pub kind: StmtKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum StmtKind<'a> {
    Pragma(&'a str),
    CalibrationGrammar(&'a str),
    Include(&'a str),
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
        name: GateCallName<'a>,
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
        init: Option<DeclExpr<'a>>,
    },
    ConstDecl {
        ty: ScalarType<'a>,
        name: Ident<'a>,
        init: DeclExpr<'a>,
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
    ExternFrame { name: Ident<'a> },
    /// `extern port Identifier ;` — OpenPulse-only.
    ExternPort { name: Ident<'a> },
}

/// Body of a `cal` or `defcal` block.
///
/// `Raw` is the default — the body is captured verbatim because we don't know
/// the calibration grammar. `OpenPulse` is populated only when the program has
/// declared `defcalgrammar "openpulse";`, in which case the body is re-parsed
/// against the OpenPulse grammar and stored as structured statements.
#[derive(Debug)]
pub enum CalBody<'a> {
    Raw(Option<&'a str>),
    OpenPulse(Vec<StmtOrScope<'a>>),
}

// ---- Expressions ----

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    BitNot,
    LogNot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntEncoding {
    Decimal,
    Binary,
    Octal,
    Hex,
}

#[derive(Debug)]
pub enum Expr<'a> {
    Ident(Ident<'a>),
    HardwareQubit(&'a str, Span),
    IntLiteral(&'a str, IntEncoding, Span),
    FloatLiteral(&'a str, Span),
    ImagLiteral(&'a str, Span),
    BoolLiteral(bool, Span),
    BitstringLiteral(&'a str, Span),
    TimingLiteral(&'a str, Span),
    Paren(Box<Expr<'a>>, Span),
    BinOp {
        left: Box<Expr<'a>>,
        op: BinOp,
        right: Box<Expr<'a>>,
        span: Span,
    },
    UnaryOp {
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
}

impl<'a> Expr<'a> {
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident(id) => id.span,
            Expr::HardwareQubit(_, s)
            | Expr::IntLiteral(_, _, s)
            | Expr::FloatLiteral(_, s)
            | Expr::ImagLiteral(_, s)
            | Expr::BoolLiteral(_, s)
            | Expr::BitstringLiteral(_, s)
            | Expr::TimingLiteral(_, s)
            | Expr::Paren(_, s)
            | Expr::BinOp { span: s, .. }
            | Expr::UnaryOp { span: s, .. }
            | Expr::Index { span: s, .. }
            | Expr::Call { span: s, .. }
            | Expr::Cast { span: s, .. }
            | Expr::DurationOf { span: s, .. } => *s,
        }
    }
}

// ---- Index types ----

#[derive(Debug)]
pub struct IndexOp<'a> {
    pub kind: IndexKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum IndexKind<'a> {
    Set(Vec<Expr<'a>>),
    Items(Vec<IndexItem<'a>>),
}

#[derive(Debug)]
pub enum IndexItem<'a> {
    Single(Expr<'a>),
    Range(RangeExpr<'a>),
}

#[derive(Debug)]
pub struct RangeExpr<'a> {
    pub start: Option<Box<Expr<'a>>>,
    pub end: Option<Box<Expr<'a>>>,
    pub step: Option<Box<Expr<'a>>>,
}

#[derive(Debug)]
pub struct IndexedIdent<'a> {
    pub name: Ident<'a>,
    pub indices: Vec<IndexOp<'a>>,
    pub span: Span,
}

// ---- Statement-specific types ----

#[derive(Debug)]
pub enum ForIterable<'a> {
    Set(Vec<Expr<'a>>, Span),
    Range(RangeExpr<'a>, Span),
    Expr(Expr<'a>),
}

#[derive(Debug)]
pub enum SwitchCase<'a> {
    Case(Vec<Expr<'a>>, Scope<'a>),
    Default(Scope<'a>),
}

#[derive(Debug, Clone, Copy)]
pub enum IoDir {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy)]
pub enum OldStyleKind {
    Creg,
    Qreg,
}

#[derive(Debug, PartialEq, Eq)]
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

#[derive(Debug)]
pub enum GateCallName<'a> {
    Ident(Ident<'a>),
    Gphase(Span),
}

#[derive(Debug)]
pub enum GateModifier<'a> {
    Inv(Span),
    Pow(Expr<'a>, Span),
    Ctrl(Option<Expr<'a>>, Span),
    NegCtrl(Option<Expr<'a>>, Span),
}

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
pub enum ExprOrMeasure<'a> {
    Expr(Expr<'a>),
    Measure(MeasureExpr<'a>),
}

#[derive(Debug)]
pub enum DeclExpr<'a> {
    Expr(Expr<'a>),
    Measure(MeasureExpr<'a>),
    ArrayLiteral(ArrayLiteral<'a>),
}

#[derive(Debug)]
pub enum DefcalTarget<'a> {
    Measure(Span),
    Reset(Span),
    Delay(Span),
    Ident(Ident<'a>),
}

#[derive(Debug)]
pub enum DefcalArgDef<'a> {
    Expr(Expr<'a>),
    ArgDef(ArgDef<'a>),
}

#[derive(Debug)]
pub enum DefcalOperand<'a> {
    HardwareQubit(&'a str, Span),
    Ident(Ident<'a>),
}

#[derive(Debug)]
pub enum ArgDef<'a> {
    Scalar(ScalarType<'a>, Ident<'a>),
    Qubit(QubitType<'a>, Ident<'a>),
    Creg(Ident<'a>, Option<Expr<'a>>),
    Qreg(Ident<'a>, Option<Expr<'a>>),
    ArrayRef(ArrayRefType<'a>, Ident<'a>),
}

#[derive(Debug)]
pub enum ExternArg<'a> {
    Scalar(ScalarType<'a>),
    ArrayRef(ArrayRefType<'a>),
    Creg(Option<Expr<'a>>),
}

#[derive(Debug)]
pub struct ArrayLiteral<'a> {
    pub items: Vec<ArrayLiteralItem<'a>>,
    pub span: Span,
}

#[derive(Debug)]
pub enum ArrayLiteralItem<'a> {
    Expr(Expr<'a>),
    Nested(ArrayLiteral<'a>),
}

// ---- Type types ----

#[derive(Debug)]
pub enum ScalarType<'a> {
    Bit(Option<Box<Expr<'a>>>, Span),
    Int(Option<Box<Expr<'a>>>, Span),
    Uint(Option<Box<Expr<'a>>>, Span),
    Float(Option<Box<Expr<'a>>>, Span),
    Angle(Option<Box<Expr<'a>>>, Span),
    Bool(Span),
    Duration(Span),
    Stretch(Span),
    Complex(Option<Box<ScalarType<'a>>>, Span),
    Waveform(Span),
    Port(Span),
    Frame(Span),
}

impl<'a> ScalarType<'a> {
    pub fn span(&self) -> Span {
        match self {
            ScalarType::Bit(_, s)
            | ScalarType::Int(_, s)
            | ScalarType::Uint(_, s)
            | ScalarType::Float(_, s)
            | ScalarType::Angle(_, s)
            | ScalarType::Bool(s)
            | ScalarType::Duration(s)
            | ScalarType::Stretch(s)
            | ScalarType::Complex(_, s)
            | ScalarType::Waveform(s)
            | ScalarType::Port(s)
            | ScalarType::Frame(s) => *s,
        }
    }
}

#[derive(Debug)]
pub struct QubitType<'a> {
    pub designator: Option<Box<Expr<'a>>>,
    pub span: Span,
}

#[derive(Debug)]
pub struct ArrayType<'a> {
    pub element_type: ScalarType<'a>,
    pub dimensions: Vec<Expr<'a>>,
    pub span: Span,
}

#[derive(Debug)]
pub enum ArrayRefMut {
    Readonly,
    Mutable,
}

#[derive(Debug)]
pub struct ArrayRefType<'a> {
    pub mutability: ArrayRefMut,
    pub element_type: ScalarType<'a>,
    pub dimensions: ArrayRefDims<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum ArrayRefDims<'a> {
    ExprList(Vec<Expr<'a>>),
    Dim(Expr<'a>),
}

#[derive(Debug)]
pub enum TypeExpr<'a> {
    Scalar(ScalarType<'a>),
    Array(ArrayType<'a>),
}
