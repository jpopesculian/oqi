# Semantic IR Plan

## Reference

The OpenQASM 3 language specification is available locally at `plan/language/`. Key files:

- `types.rst` — complete type system (scalars, arrays, angles, casting rules)
- `gates.rst` — gate definitions, modifiers (ctrl, inv, pow), broadcasting
- `insts.rst` — built-in quantum instructions (measure, reset, barrier)
- `classical.rst` — classical operations, control flow (if, for, while, switch)
- `subroutines.rst` — def functions, parameter passing
- `scope.rst` — scoping rules, lifetime, shadowing
- `directives.rst` — pragmas, annotations, I/O declarations
- `standard_library.rst` — standard gate library (h, cx, ccx, etc.)
- `delays.rst` — timing, durations, stretch, box
- `pulses.rst` / `openpulse.rst` — pulse-level features (defcal)
- `comments.rst` — comments and version strings

## Overview

Build a `compile` crate (`oqi-compile`) that lowers the parse AST (`oqi-parse`) into a **Semantic IR** — an owned, name-resolved, type-checked intermediate representation that preserves the high-level structure and meaning of the OpenQASM 3 program.

This is the first of two planned IRs:

1. **Semantic IR** (this plan) — preserves structured control flow, slices/ranges, gate modifiers, register operations, calibration headers, and opaque calibration bodies. Lexical names are resolved to symbol IDs, intrinsic calls are classified explicitly, type designators and array shapes are resolved where the language requires compile-time constants, literals are parsed, and includes are expanded with source-aware resolution.
2. **Lowered circuit/control IR** (future) — slices expanded, control flow lowered to CFG (blocks + terminators), gate modifiers applied, broadcasting expanded. Pleasant for analysis and backend lowering.

### What the Semantic IR does (AST → SIR)

| Transformation          | Example                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| Name resolution         | `Ident { name: "q" }` → `SymbolId(3)`                                          |
| Literal parsing         | `IntLiteral("0xFF")` → awint-backed integer value                              |
| Type resolution         | `int[8]` with designator expr → `Type::Int { width: 8, signed: true }`         |
| Include expansion       | `include "stdgates.inc"` → included source lowered in the current global scope |
| Old-style normalization | `creg c[4]` → `bit[4] c`, `qreg q[4]` → `qubit[4] q`                           |
| Compound op resolution  | `Compound("+=")` → `AssignOp::AddAssign`                                       |
| Ownership               | `&'a str` references → owned `String` / parsed values                          |

### What the Semantic IR preserves (NOT lowered)

- Structured control flow: `if`/`for`/`while`/`switch` as statement nodes
- Slices, ranges, and set indices as-is
- Gate modifiers (`ctrl`, `inv`, `pow`) as-is
- Register broadcasting (e.g., `h q;` on a register) as-is
- Expression trees (not flattened to SSA)
- Annotations

### Architecture

```
compile/
├── src/
│   ├── lib.rs          -- public API: compile_ast(...), compile_source(...), compile_file(...)
│   ├── sir.rs          -- Semantic IR type definitions
│   ├── types.rs        -- Type enum + type utilities
│   ├── value.rs        -- ConstValue, FloatValue, ComplexValue, TimingValue; shared with interpreter
│   ├── symbol.rs       -- SymbolTable, SymbolId, Symbol, ConstValue
│   ├── intrinsic.rs    -- builtin call targets and const-eval helpers
│   ├── lower.rs        -- AST → SIR lowering (main pass)
│   ├── resolve.rs      -- name resolution, scope tracking, include expansion
│   └── error.rs        -- compile errors with Span
```

Dependencies: `oqi-parse`, `oqi-lex` (for `Span`), `bitvec` (for `BitVec` in bitstring literals), `awint` (for arbitrary-width integer values), `num-complex` (for width-aware complex constant/interpreter values), and `ariadne` for CLI diagnostics.

Constant evaluation uses a richer `ConstValue` model shared with the future
expression interpreter. The goal is to preserve language-level values such as
bits, angles, timings, bitstrings, arbitrary-width integers, and complex numbers
instead of immediately collapsing everything to plain host scalars.

---

## Phase 1: Core IR Types

### 1.1 `symbol.rs` — Symbol table

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

pub struct SymbolTable {
    symbols: Vec<Symbol>,
}

pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub kind: SymbolKind,
    pub ty: Type,
    pub span: Span,
    pub const_value: Option<ConstValue>,
}

pub enum SymbolKind {
    /// `const int[32] N = 10;`
    Const,
    /// `int[8] x;`, `bit[4] c;`
    Variable,
    /// `input float[64] theta;`
    Input,
    /// `output bit[4] result;`
    Output,
    /// `qubit[4] q;` or `qubit q;`
    Qubit,
    /// `let alias = q[0:1] ++ q[3:4];` — a view into existing registers
    Alias,
    /// Classical parameter of a gate definition: `gate rx(θ) q { ... }`
    /// Gate params are implicitly `angle` type.
    GateParam,
    /// Qubit parameter of a gate definition: `gate cx a, b { ... }`
    /// Gate qubits are implicitly `qubit` type.
    GateQubit,
    /// Parameter of a subroutine definition: `def f(int[32] x, qubit q) { ... }`
    SubroutineParam,
    /// Loop variable: `for uint i in [0:3] { ... }`
    LoopVar,
    /// Gate name: `gate h a { ... }`
    Gate,
    /// Subroutine name: `def f(...) { ... }`
    Subroutine,
    /// Extern function name: `extern get_param(...) -> ...`
    Extern,
}
```

`SymbolTable` methods:

- `insert(name, kind, ty, span) -> SymbolId`
- `set_const_value(id, value)`
- `get(id) -> &Symbol`
- `get_mut(id) -> &mut Symbol`

### 1.2 `types.rs` — Type representation

Fully resolved types — no expression designators, no lifetimes.

```rust
pub enum Type {
    Void,
    Bool,
    Bit,                                     // `bit`
    BitReg(u32),                             // `bit[N]`
    Int { width: u32, signed: bool },        // int or int[N], uint or uint[N]
    Float(FloatWidth),                       // float or float[32]/float[64]
    /// Resolved angle precision. Plain `angle` and gate parameters use the
    /// compiler-provided system width during lowering.
    Angle(u32),
    Complex(FloatWidth),                     // complex or complex[float[32]]/complex[float[64]]
    Duration,
    Stretch,
    Array {
        element: Box<Type>,
        dims: Vec<u32>,
    },
    ArrayRef {
        element: Box<Type>,
        dims: ArrayRefDims,
        access: ArrayAccess,
    },
    Qubit,                                   // `qubit`
    QubitReg(u32),                           // `qubit[N]`
    /// Physical qubit reference: `$0`, `$1`
    PhysicalQubit,
}

pub enum FloatWidth {
    F32,
    F64,
}

pub enum ArrayRefDims {
    Fixed(Vec<u32>),
    Rank(u32),                               // `#dim = N`
}

pub enum ArrayAccess {
    Readonly,
    Mutable,
}
```

A `resolve_type(ast_type, symbols, options) -> Result<Type>` function that:

- Evaluates designator expressions to constant `u32` values via `eval_const_expr` (see below).
- Maps `ScalarType::Int(Some(expr))` → `Type::Int { width: eval(expr), signed: true }`.
- Maps `ScalarType::Int(None)` → `Type::Int { width: options.system_angle_width, signed: true }`.
- Maps `ScalarType::Uint(Some(expr))` → `Type::Int { width: eval(expr), signed: false }`.
- Maps `ScalarType::Uint(None)` → `Type::Int { width: options.system_angle_width, signed: false }`.
- Maps `ScalarType::Float(None)` → `Type::Float(float_width_from_system_width(options.system_angle_width))`.
- Maps `ScalarType::Complex(None)` → `Type::Complex(float_width_from_system_width(options.system_angle_width))`.
- Maps `ScalarType::Bit(None)` → `Type::Bit` and `ScalarType::Bit(Some(expr))` → `Type::BitReg(eval(expr))`.
- Maps `ScalarType::Angle(Some(expr))` → `Type::Angle(eval(expr))`.
- Maps `ScalarType::Angle(None)` → `Type::Angle(options.system_angle_width)`.
- Maps `TypeExpr::Array` → `Type::Array { ... }`.
- Maps `OldStyleKind::Creg` + designator → `Type::BitReg(n)`.
- Maps `OldStyleKind::Qreg` + designator → `Type::QubitReg(n)`.

Here `options.system_angle_width` is the compiler's shared implicit-width
setting despite the historical field name: it provides the default width for
plain `int`, `uint`, `float`, `complex`, and `angle`.

`float_width_from_system_width` accepts the system widths supported by the
language's floating and complex types today (`32` and `64`).

Array-reference types used by subroutines and externs are lowered by a separate
`resolve_arg_type` helper:

- `readonly array[int[8], 3]` → `Type::ArrayRef { access: Readonly, dims: Fixed([3]), ... }`
- `mutable array[int[8], #dim = 2]` → `Type::ArrayRef { access: Mutable, dims: Rank(2), ... }`

#### Compile-time constant evaluation

OpenQASM 3 requires certain expressions to be evaluated at compile time: type
designators (`int[N]`, `qubit[N]`, `angle[N]`, `bit[N]`), array dimensions
(`array[float[64], M, N]`), and const initializers. These can reference `const`
variables and use arithmetic on literals.

A small tree-walking expression interpreter,
`eval_const_expr(expr, symbols) -> Result<ConstValue>`, handles this in
`lower.rs`. It is intentionally designed so it can evolve into the later
general-purpose interpreter instead of being a throwaway constant folder.

It walks the expression tree and:

- Resolves `Var(sym)` → looks up the symbol; if `SymbolKind::Const`, uses its
  stored value. Otherwise returns `NonConstantExpression`.
- Evaluates `IntLit`, `UintLit`, `FloatLit`, and `BoolLit` directly,
  preserving arbitrary-width integer values with `awint` and explicit float
  widths with `FloatValue`.
- Evaluates bitstring literals, timing literals, and sized-angle casts into
  dedicated value variants.
- Folds `Binary`, `Unary`, indexing, and casts on constant operands where the
  spec allows compile-time evaluation. Scalar `bit` values are produced here
  when a constant expression's type is `bit`, not by a dedicated literal node.
- Recognizes built-in constants (`pi`, `tau`, `euler`).
- Evaluates intrinsic calls such as `sin`, `cos`, `tan`, `arcsin`, `arccos`,
  `arctan`, `exp`, `log`, `sqrt`, `ceiling`, `floor`, `mod`, `popcount`,
  `rotl`, `rotr`, `real`, `imag`, and `sizeof` when all operands are const and
  the specific overload is compile-time valid.
- Returns `NonConstantExpression` for user-defined subroutine calls, extern
  calls, runtime array-size queries, and anything else it cannot resolve.

```rust
use awint::Awi;
use num_complex::Complex;

enum FloatValue {
    F32(f32),
    F64(f64),
}

enum ComplexValue {
    F32(Complex<f32>),
    F64(Complex<f64>),
}

enum ConstValue {
    Bool(bool),
    Int(Awi),
    Uint(Awi),
    Float(FloatValue),
    Bitstring(BitVec),
    Angle(Awi),
    Timing(TimingValue),
    Complex(ComplexValue),
}

pub struct TimingValue {
    pub value: TimingNumber,
    pub unit: TimeUnit,
}

pub enum TimingNumber {
    Integer(i64),
    Float(FloatValue),
}
```

This evaluator covers the compile-time forms used by the language for widths,
dimensions, constant initializers, switch case labels, intrinsic folding, and
the shipped fixture programs. Because it is already interpreter-shaped and uses
language-level value types, the next step can extend it instead of replacing it.

### 1.3 `error.rs` — Compile errors

```rust
pub struct CompileError {
    pub kind: ErrorKind,
    pub span: Span,
}

pub enum ErrorKind {
    UndefinedName(String),
    DuplicateDefinition(String),
    TypeMismatch { expected: Type, got: Type },
    NonConstantDesignator,
    NonConstantExpression,
    InvalidWidth(u32),
    IncludeNotFound(String),
    IncludeCycle(Vec<String>),
    MissingSourceContext,
    InvalidContext(String),
    InvalidGateBody(String),
    InvalidSwitch(String),
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, CompileError>;
```

**Verify:** All types compile. `SymbolTable` insert/get round-trips. `resolve_type` maps AST scalar types to IR types.

---

## Phase 2: IR Nodes

### 2.1 `sir.rs` — Program structure

```rust
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
```

### 2.2 Declarations

Declarations are collected into their respective vectors in `Program`. The body
references them by `SymbolId`. Variable/const/qubit/IO declarations appear inline
as statements in the body (they have execution-time effects like initialization).

```rust
pub struct GateDecl {
    pub symbol: SymbolId,
    pub params: Vec<SymbolId>,          // gate params use `Type::Angle(options.system_angle_width)`
    pub qubits: Vec<SymbolId>,          // qubit parameter names
    pub body: GateBody,
    pub span: Span,
}

/// Gate bodies allow only gate applications (including modified gates and
/// `gphase`), `barrier`, and loops whose bodies recursively satisfy the same
/// restriction. No declarations, assignments, subroutine calls, or qubit
/// indexing of gate-argument identifiers are allowed.
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
    /// Classical scalar — passed by value (copy).
    ByValue,
    /// Qubit or qubit register — passed by reference. A given qubit may appear
    /// at most once in a subroutine call (no aliasing).
    QubitRef,
    /// `readonly array[...] name` — array passed by immutable reference.
    ReadonlyRef,
    /// `mutable array[...] name` — array passed by mutable reference.
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
    Expr(Expr),
    Param(SymbolId),
}

pub enum CalibrationOperand {
    Hardware(u32),
    Ident(String),
}

/// Calibration bodies are opaque at this IR level.
pub enum CalibrationBody {
    Opaque(String),
}
```

### 2.3 Statements

Structured control flow preserved. Each statement corresponds directly to an
AST statement, but with resolved names and types.

```rust
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
```

### 2.4 Expressions

Owned, resolved. Every expression carries its resolved type and source span.

```rust
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type,
    pub span: Span,
}

pub enum ExprKind {
    // --- Literals ---
    BoolLit(bool),
    IntLit(Awi),
    UintLit(Awi),
    FloatLit(FloatValue),
    ImagLit(FloatValue),
    BitstringLit(BitVec),
    TimingLit(TimingValue),

    // --- References ---
    Var(SymbolId),
    HardwareQubit(u32),

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
```

### 2.5 Supporting types

```rust
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    BitAnd, BitOr, BitXor, Shl, Shr,
    LogAnd, LogOr,
    Eq, Neq, Lt, Gt, Lte, Gte,
}

pub enum UnOp {
    Neg,
    BitNot,
    LogNot,
}

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
    Pow(Expr),
    /// Number of control qubits. Resolved to a concrete positive integer
    /// via `eval_const_expr` during lowering. Default is 1.
    Ctrl(u32),
    /// Number of negatively-controlled qubits. Same resolution as `Ctrl`.
    NegCtrl(u32),
}

pub enum GateCallTarget {
    /// A resolved gate name.
    Symbol(SymbolId),
    /// Built-in zero-qubit gate.
    GPhase,
}

pub enum QubitOperand {
    /// Single qubit or full register: `q`, `q[0]`, `q[0:3]`
    Indexed {
        symbol: SymbolId,
        indices: Vec<IndexOp>,
    },
    /// Physical qubit: `$0`, `$1`
    Hardware(u32),
}

/// Left-hand side of an assignment.
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
    /// `[{a, b, c}]`
    Set(Vec<Expr>),
    /// `[0]`, `[0, 1]`, `[0:3]`, `[1:2, 0]`
    Items(Vec<IndexItem>),
}

pub enum IndexItem {
    Single(Expr),
    Range(RangeExpr),
}

pub struct RangeExpr {
    pub start: Option<Box<Expr>>,
    pub step: Option<Box<Expr>>,
    pub end: Option<Box<Expr>>,
}

pub enum ForIterable {
    /// `[start:end]` or `[start:step:end]` — inclusive on both ends.
    Range {
        start: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    /// `{a, b, c}` — discrete set of values.
    Set(Vec<Expr>),
    /// Iterate over a bit register, array, or array slice.
    Expr(Expr),
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

pub enum CallTarget {
    Symbol(SymbolId),
    Intrinsic(Intrinsic),
}

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
    Expr(Expr),
    Measure(MeasureExpr),
}

pub enum ReturnValue {
    Expr(Expr),
    Measure(MeasureExpr),
}

pub enum DeclInit {
    Expr(Expr),
    Measure(MeasureExpr),
    ArrayLiteral(ArrayLiteral),
}

pub struct ArrayLiteral {
    pub items: Vec<ArrayLiteralItem>,
    pub span: Span,
}

pub enum ArrayLiteralItem {
    Expr(Expr),
    Nested(ArrayLiteral),
}

pub enum TimeUnit {
    Dt, Ns, Us, Ms, S,
}
```

**Verify:** All IR types compile. Can manually construct the IR for the `teleport.qasm` fixture and confirm it represents the program correctly.

---

## Phase 3: Name Resolution

### 3.1 `resolve.rs` — Scope tracker

The resolver walks the AST and maintains a scope stack mapping names to `SymbolId`s.
It inserts symbols into the `SymbolTable` as it encounters declarations, and resolves
references by searching the scope stack top-down.

```rust
struct Resolver {
    symbols: SymbolTable,
    scopes: Vec<HashMap<String, SymbolId>>,
    include_stack: Vec<PathBuf>,            // active include chain
}
```

Methods:

- `push_scope()` / `pop_scope()`
- `declare(name, kind, ty, span) -> Result<SymbolId>` — error on duplicate in current scope.
- `resolve(name, span) -> Result<SymbolId>` — search scopes top-down, error if not found.
- `resolve_call(name, span) -> Result<CallTarget>` — lexical symbol or intrinsic.
- `push_include(path)` / `pop_include()` — for cycle diagnostics.

### 3.2 Scope rules per OpenQASM 3

| Context            | Creates new scope? | Notes                                                                                                         |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `{ ... }` block    | Yes                | Inherits all names from containing scope                                                                      |
| `gate ... { }`     | Yes                | Use the gate-specific restrictions from `gates.rst`; gate params lower with the configured system angle width |
| `def ... { }`      | Yes                | Params plus visible global consts / previously-defined callables                                              |
| `for (...) { }`    | Yes                | Loop variable bound in body scope; inherits containing scope                                                  |
| `if / while` body  | Yes                | Inherits containing scope                                                                                     |
| `switch` case body | Yes                | Inherits containing scope; no gate/def/qubit/array declarations allowed                                       |
| `cal { }`          | Opaque             | Preserved inline; body not parsed below calibration-grammar boundary                                          |
| `defcal { }`       | Opaque             | Header lowered, body opaque                                                                                   |

Global-only validations enforced during lowering:

- `include`
- `defcalgrammar`
- `gate`, `def`, `defcal`
- `qubit` declarations
- `array[...]` declarations

Block-scope validations enforced during lowering:

- Local block scopes may not declare qubits or arrays.
- `switch` targets must be integer-typed, case labels must be integer const expressions, and duplicate labels are rejected.

### 3.3 Pre-declared names

Before processing the program body, insert built-in names:

- Constants: `pi` (π), `tau` (τ), `euler` (e) as `SymbolKind::Const`.
- Built-in gates: `U` as a 3-parameter, 1-qubit gate.
- Built-in functions are handled as `Intrinsic` call targets, not as ordinary lexical symbols. Use the spec names exactly: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `exp`, `log`, `sqrt`, `ceiling`, `floor`, `mod`, `popcount`, `sizeof`, `rotl`, `rotr`, `real`, `imag`.

### 3.4 Include handling

When `Include(path)` is encountered:

1. Validate that the statement occurs at global scope.
2. Resolve the path relative to the including file's directory from `CompileOptions::source_name`.
3. If `path` is `"stdgates.inc"`, optionally satisfy it from an embedded source string, but preserve ordinary include semantics.
4. Push the resolved path onto `include_stack`; error on cycles.
5. Parse with `oqi-parse`.
6. Recursively lower the parsed AST into the current program's global scope, exactly as though the file contents appeared inline.
7. Pop the path from `include_stack`.

Do **not** silently de-duplicate includes. The language models `include` as
textual inclusion; including the same file twice replays its declarations and
may therefore legitimately produce duplicate-definition diagnostics.

**Verify:** Test that `include "stdgates.inc"` makes `h`, `cx`, `ccx`, etc. available. Test that relative includes resolve against the including file. Test that recursive includes produce `IncludeCycle`. Test that re-including a file replays its declarations as textual inclusion. Test that referencing an undeclared name produces `UndefinedName`. Test scope shadowing.

---

## Phase 4: AST → SIR Lowering

### 4.1 `lower.rs` — Main lowering pass

```rust
pub fn compile_ast(program: &ast::Program<'_>, options: &CompileOptions) -> Result<sir::Program>
pub fn compile_source(source: &str, source_name: Option<&Path>) -> Result<sir::Program>
pub fn compile_file(path: &Path) -> Result<sir::Program>
```

The lowering pass walks the AST top-down, using the resolver for name resolution
and building the SIR nodes. One pass, producing the full `sir::Program`, with
source-path context available for include expansion.

`CompileOptions` carries at least:

- `source_name: Option<PathBuf>` for include resolution and diagnostics.
- `system_angle_width: u32` as the shared implicit-width setting for plain
  `int`, `uint`, `float`, `complex`, and `angle`, including gate parameters.

`compile_source` and `compile_file` are convenience wrappers that populate a
default `CompileOptions`, with `system_angle_width = usize::BITS as u32`,
before delegating to `compile_ast`.

### 4.2 Statement lowering

| AST node                       | SIR node                                       | Notes                                                                |
| ------------------------------ | ---------------------------------------------- | -------------------------------------------------------------------- |
| `StmtKind::Include(path)`      | _(inlined)_                                    | Parse + lower into current program                                   |
| `StmtKind::ClassicalDecl`      | `StmtKind::ClassicalDecl`                      | Resolve type, lower init expr                                        |
| `StmtKind::ConstDecl`          | `StmtKind::ConstDecl`                          | Resolve type, lower init expr, store evaluated const value on symbol |
| `StmtKind::QuantumDecl`        | `StmtKind::QubitDecl`                          | Resolve designator to size                                           |
| `StmtKind::OldStyleDecl`       | `StmtKind::ClassicalDecl` or `QubitDecl`       | `creg` → `Bit`, `qreg` → `Qubit`                                     |
| `StmtKind::IoDecl`             | `StmtKind::IoDecl`                             |                                                                      |
| `StmtKind::Gate`               | → `GateDecl` in `program.gates`                | Not a body statement                                                 |
| `StmtKind::Def`                | → `SubroutineDecl` in `program.subroutines`    | Not a body statement                                                 |
| `StmtKind::Extern`             | → `ExternDecl` in `program.externs`            | Not a body statement                                                 |
| `StmtKind::Defcal`             | → `CalibrationDecl` in `program.calibrations`  | Body kept opaque                                                     |
| `StmtKind::Cal`                | `StmtKind::Cal`                                | Inline opaque calibration block                                      |
| `StmtKind::GateCall`           | `StmtKind::GateCall`                           | Resolve gate name, lower args + operands                             |
| `StmtKind::MeasureArrow`       | `StmtKind::Measure`                            | Lower operand and target                                             |
| `StmtKind::Reset`              | `StmtKind::Reset`                              |                                                                      |
| `StmtKind::Barrier`            | `StmtKind::Barrier`                            |                                                                      |
| `StmtKind::Assignment`         | `StmtKind::Assignment`                         | Lower either expression or measure RHS                               |
| `StmtKind::If`                 | `StmtKind::If`                                 | Lower condition + bodies in new scopes                               |
| `StmtKind::For`                | `StmtKind::For`                                | Declare loop var in body scope                                       |
| `StmtKind::While`              | `StmtKind::While`                              |                                                                      |
| `StmtKind::Switch`             | `StmtKind::Switch`                             | Validate integer target, const labels, duplicates                    |
| `StmtKind::Break`              | `StmtKind::Break`                              |                                                                      |
| `StmtKind::Continue`           | `StmtKind::Continue`                           |                                                                      |
| `StmtKind::Return`             | `StmtKind::Return`                             |                                                                      |
| `StmtKind::End`                | `StmtKind::End`                                |                                                                      |
| `StmtKind::Alias`              | `StmtKind::Alias`                              |                                                                      |
| `StmtKind::Delay`              | `StmtKind::Delay`                              |                                                                      |
| `StmtKind::Box`                | `StmtKind::Box`                                |                                                                      |
| `StmtKind::Nop`                | `StmtKind::Nop`                                |                                                                      |
| `StmtKind::Pragma`             | `StmtKind::Pragma`                             |                                                                      |
| `StmtKind::CalibrationGrammar` | Stored in `program.calibration_grammar`        | Global-only; at most one active grammar                              |
| `StmtKind::Expr`               | `StmtKind::ExprStmt`                           |                                                                      |
| `StmtOrScope::Scope`           | Flattened into parent body with scope push/pop |                                                                      |

### 4.3 Expression lowering

| AST node                    | SIR node                            | Notes                                                             |
| --------------------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `Expr::IntLiteral(s)`       | `ExprKind::IntLit` or `UintLit`     | Parse `0b`, `0o`, `0x`, decimal                                   |
| `Expr::FloatLiteral(s)`     | `ExprKind::FloatLit`                | Parsed into `FloatValue` using `options.system_angle_width`       |
| `Expr::BoolLiteral(s)`      | `ExprKind::BoolLit`                 |                                                                   |
| `Expr::BitstringLiteral(s)` | `ExprKind::BitstringLit`            | Parse `"0110"` → `BitVec`                                         |
| `Expr::ImagLiteral(s)`      | `ExprKind::ImagLit`                 | Strip `im` suffix, parse float using `options.system_angle_width` |
| `Expr::TimingLiteral(s)`    | `ExprKind::TimingLit`               | Parse into `TimingValue`                                          |
| `Expr::HardwareQubit(s)`    | `ExprKind::HardwareQubit`           | Strip `$`, parse u32                                              |
| `Expr::Ident(id)`           | `ExprKind::Var(symbol_id)`          | Resolve name                                                      |
| `Expr::Paren(e)`            | Lower inner expr (parens discarded) |                                                                   |
| `Expr::BinOp`               | `ExprKind::Binary`                  | Map `ast::BinOp` → `sir::BinOp` (1:1)                             |
| `Expr::UnaryOp`             | `ExprKind::Unary`                   | Map `ast::UnOp` → `sir::UnOp` (1:1)                               |
| `Expr::Index`               | `ExprKind::Index`                   | Lower base + index op                                             |
| `Expr::Call`                | `ExprKind::Call`                    | Resolve lexical callee or classify as `Intrinsic`                 |
| `Expr::Cast`                | `ExprKind::Cast`                    | Resolve target type                                               |
| `Expr::DurationOf`          | `ExprKind::DurationOf`              | Lower inner scope to stmt list                                    |

### 4.4 Compound assignment resolution

The AST stores compound ops as `&str` (`"+="`, `"-="`, etc.). The lowering pass maps these to the `AssignOp` enum:

```
"+=" → AddAssign    "-=" → SubAssign    "*=" → MulAssign
"/=" → DivAssign    "%=" → ModAssign    "**=" → PowAssign
"&=" → BitAndAssign "|=" → BitOrAssign  "^=" → BitXorAssign
"<<=" → ShlAssign   ">>=" → ShrAssign
```

### 4.5 Gate body lowering

Gate bodies in OpenQASM 3 are stricter than ordinary block scopes. Allow:

- Built-in gate statements (`U`, `gphase`)
- Calls to previously defined gates
- Gate modifiers on those gate calls
- `barrier`
- Looping constructs whose bodies recursively contain only the forms above

Do NOT allow:

- Classical declarations
- Assignments
- `measure`, `reset`, `delay`, `box`, `nop`
- Subroutine calls
- Gate or subroutine definitions
- Qubit declarations
- Indexing gate-argument identifiers inside the body

The lowering pass validates these restrictions and produces a `GateBody { body: Vec<Stmt> }`.

Inside the gate body scope, gate parameter identifiers lower to
`Type::Angle(options.system_angle_width)`
and qubit arguments lower to `Type::Qubit`. Name resolution for ordinary
expressions in the body is restricted to gate-local parameters plus builtins; gate
call targets may additionally refer to previously defined gates.

**Verify:** Compile `adder.qasm` and `teleport.qasm` end-to-end. Verify symbol count, gate count, and that all references resolve. Compile `stdgates.inc` and verify all 30+ gates produce `GateDecl` entries.

---

## Phase 5: Type Assignment

### 5.1 Expression type inference

During lowering, each `Expr` is assigned a `ty: Type` field.

Stable cases:

| Expression                   | Type                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| `BoolLit`                    | `Bool`                                                     |
| `IntLit`                     | `Int { width: options.system_angle_width, signed: true }`  |
| `UintLit`                    | `Int { width: options.system_angle_width, signed: false }` |
| `FloatLit`                   | `Float(width matching literal payload)`                    |
| `ImagLit`                    | `Complex(width matching literal payload)`                  |
| `BitstringLit(bv)`           | `BitReg(bitstring length)`                                 |
| `TimingLit`                  | `Duration`                                                 |
| `Var(sym)`                   | symbol's declared type                                     |
| `Binary { op, left, right }` | Computed from spec-defined operator rules                  |
| `Unary { op, operand }`      | Same as operand type (Neg/BitNot) or `Bool` (LogNot)       |
| `Cast { target_ty, .. }`     | `target_ty`                                                |
| `Index { base, .. }`         | Element, slice, or sub-array type depending on index       |
| `Call { callee, .. }`        | Return type of symbol or intrinsic overload                |

### 5.2 Operator typing strategy

Do not encode ad-hoc promotion tables in the implementation. Instead:

1. Standard classical types (`bool`, `int`, `uint`, `float`, `complex`) follow the implicit-promotion and conversion rules from `types.rst`, matching C99-style ranking for mixed arithmetic.
2. Special types (`bit`, `angle`, `duration`, `stretch`, arrays, qubits) only participate in operators explicitly allowed by the language spec.
3. `angle` operators follow the dedicated rules in `classical.rst`; in particular:
   - `angle + angle` / `angle - angle` → `angle`
   - `angle * uint` and `uint * angle` → `angle`
   - `angle / uint` → `angle`
   - `angle / angle` → `uint`
4. Boolean values participate in logical operations, not general arithmetic.
5. Intrinsics such as `sizeof`, `real`, and `imag` use dedicated overload resolution rather than the generic lexical-call path.

The type of each expression is computed eagerly during lowering and stored in
`Expr.ty`, so lowering also catches type errors early.

### 5.3 Cast validity table

The lowering pass validates `Cast` expressions against the spec table in
`types.rst`. Invalid casts produce a `CompileError`.

| From \ To    | bool | int | uint | float | angle | bit | duration | qubit |
| ------------ | ---- | --- | ---- | ----- | ----- | --- | -------- | ----- |
| **bool**     | -    | Yes | Yes  | Yes   | No    | Yes | No       | No    |
| **int**      | Yes  | -   | Yes  | Yes   | No    | Yes | No       | No    |
| **uint**     | Yes  | Yes | -    | Yes   | No    | Yes | No       | No    |
| **float**    | Yes  | Yes | Yes  | -     | Yes   | No  | No       | No    |
| **angle**    | Yes  | No  | No   | No    | -     | Yes | No       | No    |
| **bit**      | Yes  | Yes | Yes  | No    | Yes   | -   | No       | No    |
| **duration** | No   | No  | No   | No\*  | No    | No  | -        | No    |
| **qubit**    | No   | No  | No   | No    | No    | No  | No       | -     |

Key semantics called out explicitly in tests:

- **float → angle[N]**: mathematical conversion, `angle_uint = round(value * 2^N / 2π)`
- **angle → bool**: true if nonzero
- **int[n]/uint[n] ↔ bit[m]**: direct bit reinterpretation; requires `n == m`
- **angle[n] ↔ bit[m]**: direct bit reinterpretation; requires `n == m`
- **float → int/uint**: truncation toward zero (C99 semantics)
- **bool → int/uint/float/bit**: `false` = 0, `true` = 1

**Verify:** Test type inference on mixed expressions. Confirm `1 + 2.0` gets type `Float(float_width_from_system_width(options.system_angle_width))`. Confirm `true && false` gets type `Bool`. Confirm `angle / angle` gets `uint`. Confirm `angle[N](float_val)` is valid. Confirm `float(angle_val)` produces a compile error. Confirm `sizeof(fixed_array)` can be const-folded while `sizeof(readonly array[..., #dim = N])` remains runtime.

---

## Phase 6: CLI Integration

### 6.1 Add `compile` subcommand

Add to the CLI `Command` enum:

```rust
Compile {
    /// The OpenQASM file to compile
    path: PathBuf,

    /// Dump the IR to stdout
    #[arg(long)]
    dump: bool,
}
```

### 6.2 IR dump format

Implement `Display` for the SIR types to produce a readable text dump. This is useful
for debugging and testing. Example output for the Bell state program:

```
program v3.0
  symbols:
    %0 = const pi : float[64]
    %1 = gate h(; a)
    %2 = gate cx(; a, b)
    %3 = qubit c : bit
    %4 = qubit q : qubit[2]
  body:
    qubit_decl %3
    qubit_decl %4
    gate_call %1 () [%4[0]]
    gate_call %2 () [%4[0], %4[1]]
    measure %4[0] -> %3
```

### 6.3 Error reporting

Use `ariadne` to render compile errors with span information, matching the existing
formatter error display.

**Verify:** `oqi compile --dump fixtures/qasm/teleport.qasm` prints readable IR.

---

## Phase 7: Testing

### 7.1 Unit tests

- `symbol.rs`: Insert/resolve round-trip, duplicate detection, scope shadowing.
- `types.rs`: `resolve_type` for all AST type variants, including scalar vs register `bit`/`qubit`, `array[...]`, `readonly`/`mutable` array refs, and `#dim`.
- `resolve.rs`: Name resolution across nested scopes, gate scope restrictions, intrinsic classification, include cycles, and global-only validations.
- `lower.rs`: Expression lowering for every `Expr` variant, literal parsing, nested array literals, `return measure q;`, and compound op resolution.
- `lower.rs`: Constant-expression interpreter coverage for bits, bitstrings, timings, sized angles, `FloatValue`, and `ComplexValue`.
- `lower.rs`: `switch` validation (integer target, const integer labels, duplicate rejection).

### 7.2 Integration tests

Compile each fixture and verify no errors (except fixtures that use unsupported features):

| Fixture          | Expected                                                        |
| ---------------- | --------------------------------------------------------------- |
| `teleport.qasm`  | Compiles successfully                                           |
| `adder.qasm`     | Compiles successfully                                           |
| `rus.qasm`       | Compiles successfully                                           |
| `arrays.qasm`    | Compiles successfully                                           |
| `vqe.qasm`       | Compiles successfully (externs declared but not called through) |
| `qft.qasm`       | Compiles successfully                                           |
| `defcal.qasm`    | Compiles (calibration bodies opaque)                            |
| `alignment.qasm` | Compiles (delay/stretch as-is)                                  |

Additional focused regression tests:

- `return measure q;` lowers successfully in subroutines.
- Nested array literals produce `DeclInit::ArrayLiteral`.
- `readonly array[int[32], #dim = 3]` lowers to `Type::ArrayRef { dims: Rank(3), ... }`.
- Re-including the same file replays its declarations as textual inclusion.
- `cal` blocks remain inline statements and do not leak declarations back out.
- `defcalgrammar` is preserved as program metadata.
- Constant evaluation preserves bit, angle, timing, bitstring, float, and complex values with the richer `ConstValue` model.

### 7.3 Round-trip property

For any program that parses successfully, `compile(parse(source))` should either:

- Produce a valid `sir::Program`, or
- Return a well-formed `CompileError` with a valid span

Never panic.

---

## Implementation Order

```
Phase 1  →  verify: types compile, SymbolTable round-trips
Phase 2  →  verify: can manually construct SIR for teleport.qasm
Phase 3  →  verify: name resolution tests pass, builtins + stdgates resolve
Phase 4  →  verify: compile(parse("teleport.qasm")) produces SIR without errors
Phase 5  →  verify: all expressions carry correct types
Phase 6  →  verify: oqi compile --dump works on fixture files
Phase 7  →  verify: all fixture integration tests pass
```
