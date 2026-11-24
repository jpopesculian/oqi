pub use oqi_lex::Span;

// ---- Comments ----

/// A line (`// …`) or block (`/* … */`) comment, delimiters included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comment<'a> {
    /// The comment's text, including `//` or `/*` and `*/`.
    pub text: &'a str,
    /// Where it is in the source.
    pub span: Span,
}

// ---- Identifiers ----

/// A name, as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident<'a> {
    /// The name.
    pub name: &'a str,
    /// Where it is in the source.
    pub span: Span,
}

// ---- Program structure ----

/// A whole program: the optional version header and the top-level
/// statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Program<'a> {
    /// The `OPENQASM` header, if present.
    pub version: Option<Version<'a>>,
    /// The top-level statements and scopes, in order.
    pub body: Vec<StmtOrScope<'a>>,
    /// The whole source.
    pub span: Span,
}

/// The `OPENQASM 3.0;` header.
#[derive(Debug, Clone, PartialEq)]
pub struct Version<'a> {
    /// The version number, as written: `3`, `3.0`.
    pub specifier: &'a str,
    /// Where the header is in the source.
    pub span: Span,
}

/// An item of a statement list: a statement or a `{ … }` block.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtOrScope<'a> {
    /// A statement.
    Stmt(Box<Stmt<'a>>),
    /// A block in braces.
    Scope(Scope<'a>),
}

impl StmtOrScope<'_> {
    /// Where the statement or block is in the source.
    pub fn span(&self) -> Span {
        match self {
            StmtOrScope::Stmt(stmt) => stmt.span,
            StmtOrScope::Scope(scope) => scope.span,
        }
    }
}

/// A block in braces: `{ … }`.
#[derive(Debug, Clone, PartialEq)]
pub struct Scope<'a> {
    /// The statements and nested blocks, in order.
    pub body: Vec<StmtOrScope<'a>>,
    /// Where the block is in the source, braces included.
    pub span: Span,
}

/// An annotation such as `@bind $0` on the statement that follows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation<'a> {
    /// The keyword, including the `@`.
    pub keyword: &'a str,
    /// The rest of the line after the keyword, if any.
    pub content: Option<&'a str>,
    /// Where the annotation is in the source.
    pub span: Span,
}

// ---- Statements ----

/// A statement and the annotations before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt<'a> {
    /// The annotations on the statement, in order.
    pub annotations: Vec<Annotation<'a>>,
    /// What the statement is.
    pub kind: StmtKind<'a>,
    /// Where the statement is in the source, annotations excluded.
    pub span: Span,
}

/// A quoted string, as in `include "stdgates.inc";`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringLiteral<'a> {
    /// The text between the quotes.
    pub value: &'a str,
    /// The quote it was written with: `"` or `'`.
    pub quote: char,
    /// Where the string is in the source, quotes included.
    pub span: Span,
}

/// The kinds of statement.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind<'a> {
    /// `pragma …`: the rest of the line, as written.
    Pragma(&'a str),
    /// `defcalgrammar "openpulse";`
    CalibrationGrammar(StringLiteral<'a>),
    /// `include "file.inc";`
    Include(StringLiteral<'a>),
    /// `break;`
    Break,
    /// `continue;`
    Continue,
    /// `end;`
    End,
    /// `for type var in iterable body`
    For {
        /// The loop variable's type.
        ty: ScalarType<'a>,
        /// The loop variable.
        var: Ident<'a>,
        /// What the loop runs over.
        iterable: ForIterable<'a>,
        /// The loop body.
        body: Box<StmtOrScope<'a>>,
    },
    /// `if (condition) then_body else else_body`
    If {
        /// The condition.
        condition: Expr<'a>,
        /// What runs when the condition holds.
        then_body: Box<StmtOrScope<'a>>,
        /// What runs otherwise, if there is an `else`.
        else_body: Option<Box<StmtOrScope<'a>>>,
    },
    /// `return;` or `return value;`
    Return(Option<ExprOrMeasure<'a>>),
    /// `while (condition) body`
    While {
        /// The condition.
        condition: Expr<'a>,
        /// The loop body.
        body: Box<StmtOrScope<'a>>,
    },
    /// `switch (target) { case a, b { … } default { … } }`
    Switch {
        /// The value switched on.
        target: Expr<'a>,
        /// The cases, in order.
        cases: Vec<SwitchCase<'a>>,
    },
    /// `barrier operands;` (all qubits if there are none).
    Barrier(Vec<GateOperand<'a>>),
    /// `box[duration] { … }`
    Box {
        /// The duration in brackets, if given.
        designator: Option<Expr<'a>>,
        /// The boxed statements.
        body: Scope<'a>,
    },
    /// `delay[duration] operands;`
    Delay {
        /// The duration.
        designator: Expr<'a>,
        /// The qubits (all qubits if there are none).
        operands: Vec<GateOperand<'a>>,
    },
    /// `nop operands;`
    Nop(Vec<GateOperand<'a>>),
    /// A gate call: `modifiers @ name(args)[duration] operands;`
    GateCall {
        /// The modifiers before the name (`inv`, `pow(k)`, `ctrl`, `negctrl`).
        modifiers: Vec<GateModifier<'a>>,
        /// The gate.
        name: Ident<'a>,
        /// The parameters in parentheses, if any were written.
        args: Option<Vec<Expr<'a>>>,
        /// The duration in brackets, if given.
        designator: Option<Box<Expr<'a>>>,
        /// The qubits.
        operands: Vec<GateOperand<'a>>,
    },
    /// `measure q;` or `measure q -> c;`
    MeasureArrow {
        /// The measurement.
        measure: MeasureExpr<'a>,
        /// Where the outcome goes, after `->`.
        target: Option<IndexedIdent<'a>>,
    },
    /// `reset q;`
    Reset(GateOperand<'a>),
    /// `let name = a ++ b;`
    Alias {
        /// The alias.
        name: Ident<'a>,
        /// The concatenated parts, in order.
        value: Vec<Expr<'a>>,
    },
    /// `type name;` or `type name = init;`
    ClassicalDecl {
        /// The declared type.
        ty: TypeExpr<'a>,
        /// The variable.
        name: Ident<'a>,
        /// The initial value, if any.
        init: Option<ExprOrMeasure<'a>>,
    },
    /// `const type name = init;`
    ConstDecl {
        /// The declared type.
        ty: TypeExpr<'a>,
        /// The constant.
        name: Ident<'a>,
        /// Its value.
        init: ExprOrMeasure<'a>,
    },
    /// `input type name;` or `output type name;`
    IoDecl {
        /// `input` or `output`.
        dir: IoDir,
        /// The declared type.
        ty: TypeExpr<'a>,
        /// The variable.
        name: Ident<'a>,
    },
    /// `creg name[size];` or `qreg name[size];` (OpenQASM 2 style).
    OldStyleDecl {
        /// `creg` or `qreg`.
        keyword: OldStyleKind,
        /// The register.
        name: Ident<'a>,
        /// The size in brackets, if given.
        designator: Option<Box<Expr<'a>>>,
    },
    /// `qubit name;` or `qubit[size] name;`
    QuantumDecl {
        /// The qubit type, with its size.
        ty: QubitType<'a>,
        /// The register.
        name: Ident<'a>,
    },
    /// `def name(params) -> return_ty { … }`
    Def {
        /// The subroutine.
        name: Ident<'a>,
        /// Its parameters.
        params: Vec<ArgDef<'a>>,
        /// Its return type, if it returns a value.
        return_ty: Option<ScalarType<'a>>,
        /// Its body.
        body: Scope<'a>,
    },
    /// `extern name(params) -> return_ty;`
    Extern {
        /// The function.
        name: Ident<'a>,
        /// Its parameter types.
        params: Vec<ExternArg<'a>>,
        /// Its return type, if it returns a value.
        return_ty: Option<ScalarType<'a>>,
    },
    /// `gate name(params) qubits { … }`
    Gate {
        /// The gate.
        name: Ident<'a>,
        /// Its angle parameters.
        params: Vec<Ident<'a>>,
        /// Its qubit parameters.
        qubits: Vec<Ident<'a>>,
        /// Its body.
        body: Scope<'a>,
    },
    /// `target op value;`, e.g. `a = 1;` or `c[0] += 2;`
    Assignment {
        /// What is assigned to.
        target: IndexedIdent<'a>,
        /// `=` or a compound assignment operator.
        op: AssignOp,
        /// The value.
        value: ExprOrMeasure<'a>,
    },
    /// An expression statement, such as a subroutine call.
    Expr(Expr<'a>),
    /// `cal { … }`
    Cal(CalBody<'a>),
    /// `defcal target(args) operands -> return_ty { … }`
    Defcal {
        /// The gate or instruction calibrated.
        target: DefcalTarget<'a>,
        /// Its parameters: declarations or fixed values.
        args: Vec<DefcalArgDef<'a>>,
        /// The qubits it applies to.
        operands: Vec<DefcalOperand<'a>>,
        /// Its return type, if it returns a value.
        return_ty: Option<ScalarType<'a>>,
        /// Its body.
        body: CalBody<'a>,
    },
    /// `extern frame Identifier ;` — OpenPulse-only.
    ExternFrame {
        /// The frame.
        name: Ident<'a>,
    },
    /// `extern port Identifier ;` — OpenPulse-only.
    ExternPort {
        /// The port.
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
    /// The text between the braces, as written (`None` for an empty body).
    Raw(Option<&'a str>),
    /// The body parsed as OpenPulse statements.
    OpenPulse(Vec<StmtOrScope<'a>>),
}

// ---- Expressions ----

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Mod,
    /// `**`
    Pow,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `&&`
    LogAnd,
    /// `||`
    LogOr,
    /// `==`
    Eq,
    /// `!=`
    Neq,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Lte,
    /// `>=`
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

    /// Whether `a op b op c` groups as `a op (b op c)`: true only for `**`.
    pub const fn is_right_associative(self) -> bool {
        matches!(self, BinOp::Pow)
    }

    /// The operator as written.
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

/// A prefix operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// `-`
    Neg,
    /// `~`
    BitNot,
    /// `!`
    LogNot,
}

impl UnOp {
    /// Unary operators bind tighter than `*` and looser than `**`.
    pub const PRECEDENCE: u8 = 21;

    /// The operator as written.
    pub const fn as_str(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::BitNot => "~",
            UnOp::LogNot => "!",
        }
    }
}

/// The base an integer literal is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntEncoding {
    /// `42`
    Decimal,
    /// `0b101010`
    Binary,
    /// `0o52`
    Octal,
    /// `0x2A`
    Hex,
}

/// An expression. Literal payloads are the source text, unprocessed.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    /// A name.
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
    /// `true` or `false`.
    BoolLiteral(bool, Span),
    /// A bitstring literal; the text keeps its quotes and any `_`
    /// separators, as in `"0101_1100"`.
    BitstringLiteral(&'a str, Span),
    /// A duration literal; the text keeps its unit, as in `100ns` or `4dt`.
    TimingLiteral(&'a str, Span),
    /// A parenthesized expression, kept so a formatter can preserve it.
    Paren(Box<Expr<'a>>, Span),
    /// `left op right`
    Binary {
        /// The left operand.
        left: Box<Expr<'a>>,
        /// The operator.
        op: BinOp,
        /// The right operand.
        right: Box<Expr<'a>>,
        /// Where the expression is in the source.
        span: Span,
    },
    /// `op operand`
    Unary {
        /// The operator.
        op: UnOp,
        /// The operand.
        operand: Box<Expr<'a>>,
        /// Where the expression is in the source.
        span: Span,
    },
    /// `expr[index]`
    Index {
        /// The indexed value.
        expr: Box<Expr<'a>>,
        /// The subscript.
        index: IndexOp<'a>,
        /// Where the expression is in the source.
        span: Span,
    },
    /// `name(args)`: a subroutine, extern or built-in function call.
    Call {
        /// The function.
        name: Ident<'a>,
        /// Its arguments.
        args: Vec<Expr<'a>>,
        /// Where the call is in the source.
        span: Span,
    },
    /// `type(operand)`
    Cast {
        /// The target type.
        ty: Box<TypeExpr<'a>>,
        /// The value converted.
        operand: Box<Expr<'a>>,
        /// Where the cast is in the source.
        span: Span,
    },
    /// `durationof({ … })`
    DurationOf {
        /// The statements timed.
        scope: Scope<'a>,
        /// Where the expression is in the source.
        span: Span,
    },
    /// `{a, b, c}`, an array initializer.
    ArrayLiteral(ArrayLiteral<'a>),
}

impl<'a> Expr<'a> {
    /// Where the expression is in the source.
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

/// One subscript in brackets: `[i]`, `[i, j]`, `[a:b]` or `[{a, b}]`.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexOp<'a> {
    /// What the brackets hold.
    pub kind: IndexKind<'a>,
    /// Where the subscript is in the source, brackets included.
    pub span: Span,
}

impl IndexOp<'_> {
    /// The postfix index operator binds tighter than every other operator.
    pub const PRECEDENCE: u8 = 26;
}

/// What a subscript holds.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexKind<'a> {
    /// An index set: `[{0, 2, 5}]`.
    Set(Vec<Expr<'a>>),
    /// One index or range per dimension: `[i]`, `[i, 1:3]`.
    Items(Vec<IndexItem<'a>>),
}

/// One dimension's index in a subscript.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexItem<'a> {
    /// A single position.
    Single(Expr<'a>),
    /// A range of positions.
    Range(RangeExpr<'a>),
}

/// A range `start:end` or `start:step:end`; any part may be omitted.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeExpr<'a> {
    /// The first position, if written.
    pub start: Option<Box<Expr<'a>>>,
    /// The last position (inclusive), if written.
    pub end: Option<Box<Expr<'a>>>,
    /// The step, if written.
    pub step: Option<Box<Expr<'a>>>,
}

/// A name with any number of subscripts, as an assignment or measurement
/// target: `c`, `c[0]`, `a[1][2:3]`.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedIdent<'a> {
    /// The variable.
    pub name: Ident<'a>,
    /// The subscripts, in order.
    pub indices: Vec<IndexOp<'a>>,
    /// Where it is in the source.
    pub span: Span,
}

// ---- Statement-specific types ----

/// What a `for` loop runs over.
#[derive(Debug, Clone, PartialEq)]
pub enum ForIterable<'a> {
    /// A set of values: `{1, 5, 7}`.
    Set(Vec<Expr<'a>>, Span),
    /// A range in brackets: `[0:4]`.
    Range(RangeExpr<'a>, Span),
    /// A register, an array or a slice of one.
    Expr(Expr<'a>),
}

/// A branch of a `switch`.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchCase<'a> {
    /// `case a, b { … }`
    Case(Vec<Expr<'a>>, Scope<'a>),
    /// `default { … }`
    Default(Scope<'a>),
}

/// Whether an I/O declaration is an `input` or an `output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoDir {
    /// `input`
    Input,
    /// `output`
    Output,
}

impl IoDir {
    /// The keyword.
    pub const fn as_str(self) -> &'static str {
        match self {
            IoDir::Input => "input",
            IoDir::Output => "output",
        }
    }
}

/// The OpenQASM 2 style register keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OldStyleKind {
    /// `creg`, a classical bit register.
    Creg,
    /// `qreg`, a qubit register.
    Qreg,
}

impl OldStyleKind {
    /// The keyword.
    pub const fn as_str(self) -> &'static str {
        match self {
            OldStyleKind::Creg => "creg",
            OldStyleKind::Qreg => "qreg",
        }
    }
}

/// An assignment operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignOp {
    /// `=`
    Assign,
    /// `+=`
    AddAssign,
    /// `-=`
    SubAssign,
    /// `*=`
    MulAssign,
    /// `/=`
    DivAssign,
    /// `&=`
    BitAndAssign,
    /// `|=`
    BitOrAssign,
    /// `^=`
    BitXorAssign,
    /// `<<=`
    LeftShiftAssign,
    /// `>>=`
    RightShiftAssign,
    /// `%=`
    ModAssign,
    /// `**=`
    PowAssign,
}

impl AssignOp {
    /// The operator as written.
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

/// A gate modifier, written before the gate name with `@`.
#[derive(Debug, Clone, PartialEq)]
pub enum GateModifier<'a> {
    /// `inv @`
    Inv(Span),
    /// `pow(k) @`
    Pow(Expr<'a>, Span),
    /// `ctrl @` or `ctrl(n) @`
    Ctrl(Option<Expr<'a>>, Span),
    /// `negctrl @` or `negctrl(n) @`
    NegCtrl(Option<Expr<'a>>, Span),
}

/// A qubit operand of a gate or quantum instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum GateOperand<'a> {
    /// A register or qubit variable, possibly subscripted.
    Indexed(IndexedIdent<'a>),
    /// A physical qubit; the text keeps the `$`.
    HardwareQubit(&'a str, Span),
}

impl<'a> GateOperand<'a> {
    /// Where the operand is in the source.
    pub fn span(&self) -> Span {
        match self {
            GateOperand::Indexed(id) => id.span,
            GateOperand::HardwareQubit(_, s) => *s,
        }
    }
}

/// Something that measures: `measure q`, or a call taking qubits.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasureExpr<'a> {
    /// `measure operand`
    Measure {
        /// The qubit or register measured.
        operand: GateOperand<'a>,
        /// Where the expression is in the source.
        span: Span,
    },
    /// `name(args) operands`, a subroutine or gate applied to qubits for
    /// its value.
    QuantumCall {
        /// The subroutine or gate.
        name: Ident<'a>,
        /// The classical arguments in parentheses.
        args: Vec<Expr<'a>>,
        /// The qubits.
        operands: Vec<GateOperand<'a>>,
        /// Where the call is in the source.
        span: Span,
    },
}

/// The value of an assignment, a declaration or a `return`: an expression or
/// a measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprOrMeasure<'a> {
    /// A classical expression.
    Expr(Expr<'a>),
    /// A measurement.
    Measure(MeasureExpr<'a>),
}

/// What a `defcal` calibrates.
#[derive(Debug, Clone, PartialEq)]
pub enum DefcalTarget<'a> {
    /// `defcal measure …`
    Measure(Span),
    /// `defcal reset …`
    Reset(Span),
    /// `defcal delay …`
    Delay(Span),
    /// `defcal gate_name …`
    Ident(Ident<'a>),
}

/// A `defcal` parameter: a declaration, or a fixed value the calibration
/// matches.
#[derive(Debug, Clone, PartialEq)]
pub enum DefcalArgDef<'a> {
    /// A fixed value, such as `pi/2` in `defcal rx(pi/2) $0`.
    Expr(Expr<'a>),
    /// A typed parameter.
    ArgDef(ArgDef<'a>),
}

/// A `defcal` qubit operand.
#[derive(Debug, Clone, PartialEq)]
pub enum DefcalOperand<'a> {
    /// A physical qubit; the text keeps the `$`.
    HardwareQubit(&'a str, Span),
    /// A qubit parameter.
    Ident(Ident<'a>),
}

/// A subroutine parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgDef<'a> {
    /// `type name`
    Scalar(ScalarType<'a>, Ident<'a>),
    /// `qubit name` or `qubit[n] name`
    Qubit(QubitType<'a>, Ident<'a>),
    /// `creg name` or `creg name[n]`
    Creg(Ident<'a>, Option<Expr<'a>>),
    /// `qreg name` or `qreg name[n]`
    Qreg(Ident<'a>, Option<Expr<'a>>),
    /// `readonly array[…] name` or `mutable array[…] name`
    ArrayRef(ArrayRefType<'a>, Ident<'a>),
}

/// An `extern` function's parameter type.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternArg<'a> {
    /// A scalar type.
    Scalar(ScalarType<'a>),
    /// An array reference type.
    ArrayRef(ArrayRefType<'a>),
    /// `creg` or `creg[n]`
    Creg(Option<Expr<'a>>),
}

/// `{a, b, c}`, an array initializer; items may themselves be array literals.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayLiteral<'a> {
    /// The items, in order.
    pub items: Vec<Expr<'a>>,
    /// Where the literal is in the source, braces included.
    pub span: Span,
}

// ---- Type types ----

/// A scalar type as written: `int[32]`, `bool`, `complex[float[64]]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarType<'a> {
    /// Which type.
    pub kind: ScalarKind<'a>,
    /// Where the type is in the source.
    pub span: Span,
}

/// The scalar types.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarKind<'a> {
    /// A type that takes an optional `[designator]`.
    Sized(SizedKind, Option<Box<Expr<'a>>>),
    /// `complex`, optionally `complex[float[n]]`.
    Complex(Option<Box<ScalarType<'a>>>),
    /// `bool`
    Bool,
    /// `duration`
    Duration,
    /// `stretch`
    Stretch,
    /// `waveform` (OpenPulse)
    Waveform,
    /// `port` (OpenPulse)
    Port,
    /// `frame` (OpenPulse)
    Frame,
}

/// The scalar types that take a size: `bit[n]`, `int[n]`, `uint[n]`,
/// `float[n]`, `angle[n]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizedKind {
    /// `bit`
    Bit,
    /// `int`
    Int,
    /// `uint`
    Uint,
    /// `float`
    Float,
    /// `angle`
    Angle,
}

impl SizedKind {
    /// The keyword.
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

/// `qubit` or `qubit[n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct QubitType<'a> {
    /// The register size, if given.
    pub designator: Option<Box<Expr<'a>>>,
    /// Where the type is in the source.
    pub span: Span,
}

/// `array[element_type, d1, d2, …]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayType<'a> {
    /// The element type.
    pub element_type: ScalarType<'a>,
    /// The size of each dimension.
    pub dimensions: Vec<Expr<'a>>,
    /// Where the type is in the source.
    pub span: Span,
}

/// Whether an array reference parameter may be written through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayRefMut {
    /// `readonly`
    Readonly,
    /// `mutable`
    Mutable,
}

impl ArrayRefMut {
    /// The keyword.
    pub const fn as_str(self) -> &'static str {
        match self {
            ArrayRefMut::Readonly => "readonly",
            ArrayRefMut::Mutable => "mutable",
        }
    }
}

/// An array reference type, for subroutine parameters:
/// `readonly array[int[8], 4]` or `mutable array[float, #dim = 2]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayRefType<'a> {
    /// `readonly` or `mutable`.
    pub mutability: ArrayRefMut,
    /// The element type.
    pub element_type: ScalarType<'a>,
    /// The dimensions: their sizes, or only their number.
    pub dimensions: ArrayRefDims<'a>,
    /// Where the type is in the source.
    pub span: Span,
}

/// The dimensions of an array reference type.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayRefDims<'a> {
    /// The size of each dimension.
    ExprList(Vec<Expr<'a>>),
    /// `#dim = n`: only the number of dimensions.
    Dim(Expr<'a>),
}

/// A type in a declaration or cast: a scalar type or an array type.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr<'a> {
    /// A scalar type.
    Scalar(ScalarType<'a>),
    /// An array type.
    Array(ArrayType<'a>),
}
