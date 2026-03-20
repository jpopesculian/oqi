use std::fmt;

use awint::Awi;
use bitvec::vec::BitVec;

use crate::error::{CompileError, ErrorKind};
use crate::sir::*;
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::{ArrayAccess, ArrayRefDims, FloatWidth, Type};
use crate::value::{ComplexValue, ConstValue, FloatValue, TimeUnit, TimingNumber, TimingValue};

// ── Simple type displays ────────────────────────────────────────────

impl fmt::Display for FloatWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatWidth::F32 => write!(f, "32"),
            FloatWidth::F64 => write!(f, "64"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Bool => write!(f, "bool"),
            Type::Bit => write!(f, "bit"),
            Type::BitReg(n) => write!(f, "bit[{n}]"),
            Type::Int {
                width,
                signed: true,
            } => write!(f, "int[{width}]"),
            Type::Int {
                width,
                signed: false,
            } => write!(f, "uint[{width}]"),
            Type::Float(fw) => write!(f, "float[{fw}]"),
            Type::Angle(n) => write!(f, "angle[{n}]"),
            Type::Complex(fw) => write!(f, "complex[float[{fw}]]"),
            Type::Duration => write!(f, "duration"),
            Type::Stretch => write!(f, "stretch"),
            Type::Array { element, dims } => {
                write!(f, "array[{element}")?;
                for d in dims {
                    write!(f, ", {d}")?;
                }
                write!(f, "]")
            }
            Type::ArrayRef {
                element,
                dims,
                access,
            } => {
                match access {
                    ArrayAccess::Readonly => write!(f, "readonly ")?,
                    ArrayAccess::Mutable => write!(f, "mutable ")?,
                }
                write!(f, "array[{element}, ")?;
                match dims {
                    ArrayRefDims::Fixed(ds) => {
                        for (i, d) in ds.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{d}")?;
                        }
                    }
                    ArrayRefDims::Rank(r) => write!(f, "#dim = {r}")?,
                }
                write!(f, "]")
            }
            Type::Qubit => write!(f, "qubit"),
            Type::QubitReg(n) => write!(f, "qubit[{n}]"),
            Type::PhysicalQubit => write!(f, "physical_qubit"),
        }
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolKind::Const => write!(f, "const"),
            SymbolKind::Variable => write!(f, "var"),
            SymbolKind::Input => write!(f, "input"),
            SymbolKind::Output => write!(f, "output"),
            SymbolKind::Qubit => write!(f, "qubit"),
            SymbolKind::Alias => write!(f, "alias"),
            SymbolKind::GateParam => write!(f, "gate_param"),
            SymbolKind::GateQubit => write!(f, "gate_qubit"),
            SymbolKind::SubroutineParam => write!(f, "sub_param"),
            SymbolKind::LoopVar => write!(f, "loop_var"),
            SymbolKind::Gate => write!(f, "gate"),
            SymbolKind::Subroutine => write!(f, "def"),
            SymbolKind::Extern => write!(f, "extern"),
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::Shl => write!(f, "<<"),
            BinOp::Shr => write!(f, ">>"),
            BinOp::LogAnd => write!(f, "&&"),
            BinOp::LogOr => write!(f, "||"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Lte => write!(f, "<="),
            BinOp::Gte => write!(f, ">="),
        }
    }
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::BitNot => write!(f, "~"),
            UnOp::LogNot => write!(f, "!"),
        }
    }
}

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssignOp::Assign => write!(f, "="),
            AssignOp::AddAssign => write!(f, "+="),
            AssignOp::SubAssign => write!(f, "-="),
            AssignOp::MulAssign => write!(f, "*="),
            AssignOp::DivAssign => write!(f, "/="),
            AssignOp::ModAssign => write!(f, "%="),
            AssignOp::PowAssign => write!(f, "**="),
            AssignOp::BitAndAssign => write!(f, "&="),
            AssignOp::BitOrAssign => write!(f, "|="),
            AssignOp::BitXorAssign => write!(f, "^="),
            AssignOp::ShlAssign => write!(f, "<<="),
            AssignOp::ShrAssign => write!(f, ">>="),
        }
    }
}

impl fmt::Display for IoDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoDir::Input => write!(f, "input"),
            IoDir::Output => write!(f, "output"),
        }
    }
}

impl fmt::Display for FloatValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatValue::F32(v) => write!(f, "{v}"),
            FloatValue::F64(v) => write!(f, "{v}"),
        }
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeUnit::Dt => write!(f, "dt"),
            TimeUnit::Ns => write!(f, "ns"),
            TimeUnit::Us => write!(f, "us"),
            TimeUnit::Ms => write!(f, "ms"),
            TimeUnit::S => write!(f, "s"),
        }
    }
}

impl fmt::Display for TimingValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            TimingNumber::Integer(v) => write!(f, "{v}{}", self.unit),
            TimingNumber::Float(v) => write!(f, "{v}{}", self.unit),
        }
    }
}

impl fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Intrinsic::Sin => write!(f, "sin"),
            Intrinsic::Cos => write!(f, "cos"),
            Intrinsic::Tan => write!(f, "tan"),
            Intrinsic::Arcsin => write!(f, "arcsin"),
            Intrinsic::Arccos => write!(f, "arccos"),
            Intrinsic::Arctan => write!(f, "arctan"),
            Intrinsic::Exp => write!(f, "exp"),
            Intrinsic::Log => write!(f, "log"),
            Intrinsic::Sqrt => write!(f, "sqrt"),
            Intrinsic::Ceiling => write!(f, "ceiling"),
            Intrinsic::Floor => write!(f, "floor"),
            Intrinsic::Mod => write!(f, "mod"),
            Intrinsic::Popcount => write!(f, "popcount"),
            Intrinsic::Rotl => write!(f, "rotl"),
            Intrinsic::Rotr => write!(f, "rotr"),
            Intrinsic::Real => write!(f, "real"),
            Intrinsic::Imag => write!(f, "imag"),
            Intrinsic::Sizeof => write!(f, "sizeof"),
        }
    }
}

// ── Error display ───────────────────────────────────────────────────

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UndefinedName(name) => write!(f, "undefined name '{name}'"),
            ErrorKind::DuplicateDefinition(name) => {
                write!(f, "duplicate definition of '{name}'")
            }
            ErrorKind::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            ErrorKind::NonConstantDesignator => {
                write!(f, "designator must be a constant expression")
            }
            ErrorKind::NonConstantExpression => write!(f, "expression must be constant"),
            ErrorKind::InvalidWidth(w) => write!(f, "invalid width: {w}"),
            ErrorKind::IncludeNotFound(path) => write!(f, "include file not found: {path}"),
            ErrorKind::IncludeCycle(chain) => {
                write!(f, "include cycle: {}", chain.join(" -> "))
            }
            ErrorKind::MissingSourceContext => {
                write!(
                    f,
                    "cannot resolve relative include without source file context"
                )
            }
            ErrorKind::InvalidContext(msg) => write!(f, "{msg}"),
            ErrorKind::InvalidGateBody(msg) => write!(f, "invalid gate body: {msg}"),
            ErrorKind::InvalidSwitch(msg) => write!(f, "invalid switch: {msg}"),
            ErrorKind::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for CompileError {}

// ── Awi/BitVec helpers ──────────────────────────────────────────────

fn fmt_awi(f: &mut fmt::Formatter<'_>, awi: &Awi) -> fmt::Result {
    let bw = awi.bw();
    if bw <= 128 {
        let mut val: u128 = 0;
        for i in 0..bw {
            if awi.get(i).unwrap() {
                val |= 1u128 << i;
            }
        }
        write!(f, "{val}")
    } else {
        write!(f, "<{bw}-bit>")
    }
}

fn fmt_bitvec(f: &mut fmt::Formatter<'_>, bv: &BitVec) -> fmt::Result {
    write!(f, "\"")?;
    for bit in bv.iter() {
        write!(f, "{}", if *bit { '1' } else { '0' })?;
    }
    write!(f, "\"")
}

// ── Program display ─────────────────────────────────────────────────

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.version {
            Some(v) => writeln!(f, "program v{v}")?,
            None => writeln!(f, "program")?,
        }

        if let Some(ref grammar) = self.calibration_grammar {
            writeln!(f, "  calibration_grammar: {grammar}")?;
        }

        // Symbol table
        if !self.symbols.is_empty() {
            writeln!(f, "  symbols:")?;
            for sym in self.symbols.iter() {
                write!(f, "    {} = {} {} : {}", sym.id, sym.kind, sym.name, sym.ty)?;
                if let Some(ref cv) = sym.const_value {
                    write!(f, " = ")?;
                    fmt_const_value(f, cv)?;
                }
                writeln!(f)?;
            }
        }

        // Gate declarations
        for gate in &self.gates {
            let name = &self.symbols.get(gate.symbol).name;
            write!(f, "  gate {name}(")?;
            for (i, p) in gate.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.symbols.get(*p).name)?;
            }
            write!(f, "; ")?;
            for (i, q) in gate.qubits.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.symbols.get(*q).name)?;
            }
            writeln!(f, ") {{")?;
            for s in &gate.body.body {
                fmt_stmt(f, s, &self.symbols, 4)?;
            }
            writeln!(f, "  }}")?;
        }

        // Subroutine declarations
        for sub in &self.subroutines {
            let name = &self.symbols.get(sub.symbol).name;
            write!(f, "  def {name}(")?;
            for (i, p) in sub.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let sym = self.symbols.get(p.symbol);
                write!(f, "{} : {}", sym.name, sym.ty)?;
            }
            write!(f, ")")?;
            if let Some(ref ty) = sub.return_ty {
                write!(f, " -> {ty}")?;
            }
            writeln!(f, " {{")?;
            for s in &sub.body {
                fmt_stmt(f, s, &self.symbols, 4)?;
            }
            writeln!(f, "  }}")?;
        }

        // Extern declarations
        for ext in &self.externs {
            let name = &self.symbols.get(ext.symbol).name;
            write!(f, "  extern {name}(")?;
            for (i, ty) in ext.param_types.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{ty}")?;
            }
            write!(f, ")")?;
            if let Some(ref ty) = ext.return_ty {
                write!(f, " -> {ty}")?;
            }
            writeln!(f)?;
        }

        // Body
        if !self.body.is_empty() {
            writeln!(f, "  body:")?;
            for stmt in &self.body {
                fmt_stmt(f, stmt, &self.symbols, 4)?;
            }
        }

        Ok(())
    }
}

// ── Statement display ───────────────────────────────────────────────

fn pad(f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    for _ in 0..indent {
        write!(f, " ")?;
    }
    Ok(())
}

fn fmt_stmt(
    f: &mut fmt::Formatter<'_>,
    stmt: &Stmt,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    for ann in &stmt.annotations {
        pad(f, indent)?;
        write!(f, "@{}", ann.keyword)?;
        if let Some(ref content) = ann.content {
            write!(f, " {content}")?;
        }
        writeln!(f)?;
    }

    pad(f, indent)?;
    match &stmt.kind {
        StmtKind::ClassicalDecl { symbol, init } => {
            write!(f, "classical_decl {symbol}")?;
            if let Some(init) = init {
                write!(f, " = ")?;
                fmt_decl_init(f, init, symbols)?;
            }
            writeln!(f)
        }
        StmtKind::ConstDecl { symbol, init } => {
            write!(f, "const_decl {symbol} = ")?;
            fmt_expr(f, init, symbols)?;
            writeln!(f)
        }
        StmtKind::QubitDecl { symbol } => writeln!(f, "qubit_decl {symbol}"),
        StmtKind::IoDecl { symbol, dir } => writeln!(f, "io_decl {dir} {symbol}"),
        StmtKind::Alias { symbol, value } => {
            write!(f, "alias {symbol} = ")?;
            for (i, v) in value.iter().enumerate() {
                if i > 0 {
                    write!(f, " ++ ")?;
                }
                fmt_expr(f, v, symbols)?;
            }
            writeln!(f)
        }
        StmtKind::GateCall {
            gate,
            modifiers,
            args,
            qubits,
        } => {
            write!(f, "gate_call ")?;
            for m in modifiers {
                fmt_gate_modifier(f, m, symbols)?;
                write!(f, " ")?;
            }
            match gate {
                GateCallTarget::Symbol(id) => write!(f, "{id}")?,
                GateCallTarget::GPhase => write!(f, "gphase")?,
            }
            write!(f, " (")?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(f, arg, symbols)?;
            }
            write!(f, ") [")?;
            for (i, q) in qubits.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_qubit_op(f, q, symbols)?;
            }
            writeln!(f, "]")
        }
        StmtKind::Measure { measure, target } => {
            fmt_measure(f, measure, symbols)?;
            if let Some(lv) = target {
                write!(f, " -> ")?;
                fmt_lvalue(f, lv, symbols)?;
            }
            writeln!(f)
        }
        StmtKind::Reset { operand } => {
            write!(f, "reset ")?;
            fmt_qubit_op(f, operand, symbols)?;
            writeln!(f)
        }
        StmtKind::Barrier { operands } => {
            write!(f, "barrier")?;
            if !operands.is_empty() {
                write!(f, " ")?;
                for (i, q) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_qubit_op(f, q, symbols)?;
                }
            }
            writeln!(f)
        }
        StmtKind::Delay { duration, operands } => {
            write!(f, "delay ")?;
            fmt_expr(f, duration, symbols)?;
            for q in operands {
                write!(f, " ")?;
                fmt_qubit_op(f, q, symbols)?;
            }
            writeln!(f)
        }
        StmtKind::Box { duration, body } => {
            write!(f, "box")?;
            if let Some(dur) = duration {
                write!(f, "[")?;
                fmt_expr(f, dur, symbols)?;
                write!(f, "]")?;
            }
            writeln!(f, " {{")?;
            for s in body {
                fmt_stmt(f, s, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
        StmtKind::Assignment { target, op, value } => {
            write!(f, "assign ")?;
            fmt_lvalue(f, target, symbols)?;
            write!(f, " {op} ")?;
            match value {
                AssignValue::Expr(e) => fmt_expr(f, e, symbols)?,
                AssignValue::Measure(m) => fmt_measure(f, m, symbols)?,
            }
            writeln!(f)
        }
        StmtKind::If {
            condition,
            then_body,
            else_body,
        } => {
            write!(f, "if ")?;
            fmt_expr(f, condition, symbols)?;
            writeln!(f, " {{")?;
            for s in then_body {
                fmt_stmt(f, s, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            if let Some(else_body) = else_body {
                writeln!(f, "}} else {{")?;
                for s in else_body {
                    fmt_stmt(f, s, symbols, indent + 2)?;
                }
                pad(f, indent)?;
                writeln!(f, "}}")
            } else {
                writeln!(f, "}}")
            }
        }
        StmtKind::For {
            var,
            iterable,
            body,
        } => {
            write!(f, "for {var} in ")?;
            fmt_for_iterable(f, iterable, symbols)?;
            writeln!(f, " {{")?;
            for s in body {
                fmt_stmt(f, s, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
        StmtKind::While { condition, body } => {
            write!(f, "while ")?;
            fmt_expr(f, condition, symbols)?;
            writeln!(f, " {{")?;
            for s in body {
                fmt_stmt(f, s, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
        StmtKind::Switch { target, cases } => {
            write!(f, "switch ")?;
            fmt_expr(f, target, symbols)?;
            writeln!(f, " {{")?;
            for case in cases {
                pad(f, indent + 2)?;
                match &case.labels {
                    SwitchLabels::Values(vals) => {
                        write!(f, "case ")?;
                        for (i, v) in vals.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            fmt_expr(f, v, symbols)?;
                        }
                    }
                    SwitchLabels::Default => write!(f, "default")?,
                }
                writeln!(f, " {{")?;
                for s in &case.body {
                    fmt_stmt(f, s, symbols, indent + 4)?;
                }
                pad(f, indent + 2)?;
                writeln!(f, "}}")?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
        StmtKind::Break => writeln!(f, "break"),
        StmtKind::Continue => writeln!(f, "continue"),
        StmtKind::Return(val) => {
            write!(f, "return")?;
            if let Some(rv) = val {
                write!(f, " ")?;
                match rv {
                    ReturnValue::Expr(e) => fmt_expr(f, e, symbols)?,
                    ReturnValue::Measure(m) => fmt_measure(f, m, symbols)?,
                }
            }
            writeln!(f)
        }
        StmtKind::End => writeln!(f, "end"),
        StmtKind::Pragma(content) => writeln!(f, "pragma {content}"),
        StmtKind::Cal { body } => match body {
            CalibrationBody::Opaque(s) => writeln!(f, "cal {{{s}}}"),
        },
        StmtKind::ExprStmt(expr) => {
            fmt_expr(f, expr, symbols)?;
            writeln!(f)
        }
        StmtKind::Nop { operands } => {
            write!(f, "nop")?;
            if !operands.is_empty() {
                write!(f, " ")?;
                for (i, q) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_qubit_op(f, q, symbols)?;
                }
            }
            writeln!(f)
        }
    }
}

// ── Expression display ──────────────────────────────────────────────

fn fmt_expr(f: &mut fmt::Formatter<'_>, expr: &Expr, symbols: &SymbolTable) -> fmt::Result {
    match &expr.kind {
        ExprKind::BoolLit(b) => write!(f, "{b}"),
        ExprKind::IntLit(awi) => fmt_awi(f, awi),
        ExprKind::UintLit(awi) => fmt_awi(f, awi),
        ExprKind::FloatLit(v) => write!(f, "{v}"),
        ExprKind::ImagLit(v) => write!(f, "{v}im"),
        ExprKind::BitstringLit(bv) => fmt_bitvec(f, bv),
        ExprKind::TimingLit(tv) => write!(f, "{tv}"),
        ExprKind::Var(id) => write!(f, "{id}"),
        ExprKind::HardwareQubit(n) => write!(f, "${n}"),
        ExprKind::Binary { op, left, right } => {
            write!(f, "(")?;
            fmt_expr(f, left, symbols)?;
            write!(f, " {op} ")?;
            fmt_expr(f, right, symbols)?;
            write!(f, ")")
        }
        ExprKind::Unary { op, operand } => {
            write!(f, "({op}")?;
            fmt_expr(f, operand, symbols)?;
            write!(f, ")")
        }
        ExprKind::Cast { target_ty, operand } => {
            write!(f, "{target_ty}(")?;
            fmt_expr(f, operand, symbols)?;
            write!(f, ")")
        }
        ExprKind::Index { base, index } => {
            fmt_expr(f, base, symbols)?;
            write!(f, "[")?;
            fmt_index(f, index, symbols)?;
            write!(f, "]")
        }
        ExprKind::Call { callee, args } => {
            match callee {
                CallTarget::Symbol(id) => write!(f, "{id}")?,
                CallTarget::Intrinsic(i) => write!(f, "{i}")?,
            }
            write!(f, "(")?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(f, arg, symbols)?;
            }
            write!(f, ")")
        }
        ExprKind::DurationOf(stmts) => {
            if stmts.is_empty() {
                write!(f, "durationof {{}}")
            } else {
                write!(f, "durationof {{ ... }}")
            }
        }
    }
}

// ── Helper displays ─────────────────────────────────────────────────

fn fmt_qubit_op(
    f: &mut fmt::Formatter<'_>,
    op: &QubitOperand,
    symbols: &SymbolTable,
) -> fmt::Result {
    match op {
        QubitOperand::Indexed { symbol, indices } => {
            write!(f, "{symbol}")?;
            for idx in indices {
                write!(f, "[")?;
                fmt_index(f, idx, symbols)?;
                write!(f, "]")?;
            }
            Ok(())
        }
        QubitOperand::Hardware(n) => write!(f, "${n}"),
    }
}

fn fmt_lvalue(f: &mut fmt::Formatter<'_>, lv: &LValue, symbols: &SymbolTable) -> fmt::Result {
    match lv {
        LValue::Var(id) => write!(f, "{id}"),
        LValue::Indexed { symbol, indices } => {
            write!(f, "{symbol}")?;
            for idx in indices {
                write!(f, "[")?;
                fmt_index(f, idx, symbols)?;
                write!(f, "]")?;
            }
            Ok(())
        }
    }
}

fn fmt_index(f: &mut fmt::Formatter<'_>, idx: &IndexOp, symbols: &SymbolTable) -> fmt::Result {
    match &idx.kind {
        IndexKind::Set(exprs) => {
            write!(f, "{{")?;
            for (i, e) in exprs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(f, e, symbols)?;
            }
            write!(f, "}}")
        }
        IndexKind::Items(items) => {
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match item {
                    IndexItem::Single(e) => fmt_expr(f, e, symbols)?,
                    IndexItem::Range(r) => fmt_range(f, r, symbols)?,
                }
            }
            Ok(())
        }
    }
}

fn fmt_range(f: &mut fmt::Formatter<'_>, range: &RangeExpr, symbols: &SymbolTable) -> fmt::Result {
    if let Some(start) = &range.start {
        fmt_expr(f, start, symbols)?;
    }
    write!(f, ":")?;
    if let Some(step) = &range.step {
        fmt_expr(f, step, symbols)?;
        write!(f, ":")?;
    }
    if let Some(end) = &range.end {
        fmt_expr(f, end, symbols)?;
    }
    Ok(())
}

fn fmt_gate_modifier(
    f: &mut fmt::Formatter<'_>,
    m: &GateModifier,
    symbols: &SymbolTable,
) -> fmt::Result {
    match m {
        GateModifier::Inv => write!(f, "inv @"),
        GateModifier::Pow(expr) => {
            write!(f, "pow(")?;
            fmt_expr(f, expr, symbols)?;
            write!(f, ") @")
        }
        GateModifier::Ctrl(n) => write!(f, "ctrl({n}) @"),
        GateModifier::NegCtrl(n) => write!(f, "negctrl({n}) @"),
    }
}

fn fmt_measure(
    f: &mut fmt::Formatter<'_>,
    m: &MeasureExpr,
    symbols: &SymbolTable,
) -> fmt::Result {
    match &m.kind {
        MeasureExprKind::Measure { operand } => {
            write!(f, "measure ")?;
            fmt_qubit_op(f, operand, symbols)
        }
        MeasureExprKind::QuantumCall {
            callee,
            args,
            qubits,
        } => {
            write!(f, "{callee}(")?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(f, arg, symbols)?;
            }
            write!(f, ") [")?;
            for (i, q) in qubits.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_qubit_op(f, q, symbols)?;
            }
            write!(f, "]")
        }
    }
}

fn fmt_for_iterable(
    f: &mut fmt::Formatter<'_>,
    iter: &ForIterable,
    symbols: &SymbolTable,
) -> fmt::Result {
    match iter {
        ForIterable::Range { start, step, end } => {
            write!(f, "[")?;
            if let Some(s) = start {
                fmt_expr(f, s, symbols)?;
            }
            write!(f, ":")?;
            if let Some(st) = step {
                fmt_expr(f, st, symbols)?;
                write!(f, ":")?;
            }
            if let Some(e) = end {
                fmt_expr(f, e, symbols)?;
            }
            write!(f, "]")
        }
        ForIterable::Set(exprs) => {
            write!(f, "{{")?;
            for (i, e) in exprs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(f, e, symbols)?;
            }
            write!(f, "}}")
        }
        ForIterable::Expr(e) => fmt_expr(f, e, symbols),
    }
}

fn fmt_decl_init(
    f: &mut fmt::Formatter<'_>,
    init: &DeclInit,
    symbols: &SymbolTable,
) -> fmt::Result {
    match init {
        DeclInit::Expr(e) => fmt_expr(f, e, symbols),
        DeclInit::Measure(m) => fmt_measure(f, m, symbols),
        DeclInit::ArrayLiteral(arr) => fmt_array_literal(f, arr, symbols),
    }
}

fn fmt_array_literal(
    f: &mut fmt::Formatter<'_>,
    arr: &ArrayLiteral,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "{{")?;
    for (i, item) in arr.items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        match item {
            ArrayLiteralItem::Expr(e) => fmt_expr(f, e, symbols)?,
            ArrayLiteralItem::Nested(inner) => fmt_array_literal(f, inner, symbols)?,
        }
    }
    write!(f, "}}")
}

fn fmt_const_value(f: &mut fmt::Formatter<'_>, cv: &ConstValue) -> fmt::Result {
    match cv {
        ConstValue::Bool(b) => write!(f, "{b}"),
        ConstValue::Int(awi) | ConstValue::Uint(awi) | ConstValue::Angle(awi) => fmt_awi(f, awi),
        ConstValue::Float(v) => write!(f, "{v}"),
        ConstValue::Bitstring(bv) => fmt_bitvec(f, bv),
        ConstValue::Timing(tv) => write!(f, "{tv}"),
        ConstValue::Complex(cv) => match cv {
            ComplexValue::F32(c) => write!(f, "{}+{}im", c.re, c.im),
            ComplexValue::F64(c) => write!(f, "{}+{}im", c.re, c.im),
        },
    }
}
