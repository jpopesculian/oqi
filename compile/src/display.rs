use std::fmt;

use crate::cfg::{
    BasicBlockId, BlockBoxStmt, BlockCalibrationBody, BlockExpr, BlockExprKind, BlockStmt,
    BlockStmtKind, Cfg, CfgDisplay, Terminator,
};
use crate::error::{CompileError, ErrorKind};
use crate::sir::*;
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::Type;

/// Format an expression in either the SIR or CFG world. Implemented for
/// [`Expr`] (delegates to [`fmt_expr`]) and [`BlockExpr`] (delegates to
/// [`fmt_block_expr`]) so the per-payload helpers (`fmt_qubit_op`,
/// `fmt_index`, etc.) can be generic over the expression type.
trait FmtExpr {
    fn fmt_e(&self, f: &mut fmt::Formatter<'_>, symbols: &SymbolTable) -> fmt::Result;
}

impl FmtExpr for Expr {
    fn fmt_e(&self, f: &mut fmt::Formatter<'_>, symbols: &SymbolTable) -> fmt::Result {
        fmt_expr(f, self, symbols)
    }
}

impl FmtExpr for BlockExpr {
    fn fmt_e(&self, f: &mut fmt::Formatter<'_>, symbols: &SymbolTable) -> fmt::Result {
        fmt_block_expr(f, self, symbols)
    }
}

// ── Simple type displays ────────────────────────────────────────────

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Classical(ty) => write!(f, "{ty}"),
            Type::Unsized(u) => write!(
                f,
                "{}",
                match u {
                    crate::types::UnsizedType::Int => "int",
                    crate::types::UnsizedType::Uint => "uint",
                    crate::types::UnsizedType::Angle => "angle",
                    crate::types::UnsizedType::Float => "float",
                }
            ),
            Type::Stretch => write!(f, "stretch"),
            Type::Qubit => write!(f, "qubit"),
            Type::QubitReg(n) => write!(f, "qubit[{n}]"),
            Type::PhysicalQubit => write!(f, "physical_qubit"),
            Type::Openpulse(ty) => write!(f, "{ty}"),
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
            SymbolKind::ExternPort => write!(f, "extern_port"),
            SymbolKind::ExternFrame => write!(f, "extern_frame"),
            SymbolKind::Temp => write!(f, "temp"),
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
            ErrorKind::DuplicateDefinition { name, .. } => {
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
            ErrorKind::InvalidLiteral(msg) => write!(f, "invalid literal: {msg}"),
            ErrorKind::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            ErrorKind::QubitIndexOutOfRange { index, len } => {
                write!(
                    f,
                    "qubit index {index} out of range for register of length {len}"
                )
            }
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for CompileError {}

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
                    write!(f, "{}", cv)?;
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

        // Calibration declarations
        for cal in &self.calibrations {
            write!(f, "  defcal ")?;
            match &cal.target {
                CalibrationTarget::Measure => write!(f, "measure")?,
                CalibrationTarget::Reset => write!(f, "reset")?,
                CalibrationTarget::Delay => write!(f, "delay")?,
                CalibrationTarget::Named(sym) => write!(f, "{}", self.symbols.get(*sym).name)?,
            }
            if !cal.args.is_empty() {
                write!(f, "(")?;
                for (i, a) in cal.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match a {
                        CalibrationArg::Expr(e) => fmt_expr(f, e, &self.symbols)?,
                        CalibrationArg::Param(sym) => {
                            let s = self.symbols.get(*sym);
                            write!(f, "{} : {}", s.name, s.ty)?;
                        }
                    }
                }
                write!(f, ")")?;
            }
            for (i, op) in cal.operands.iter().enumerate() {
                write!(f, "{}", if i == 0 { " " } else { ", " })?;
                match op {
                    CalibrationOperand::Hardware(n) => write!(f, "${n}")?,
                    CalibrationOperand::Ident(sym) => write!(f, "{}", self.symbols.get(*sym).name)?,
                }
            }
            if let Some(ref ty) = cal.return_ty {
                write!(f, " -> {ty}")?;
            }
            write!(f, " ")?;
            fmt_cal_body(f, &cal.body, &self.symbols, 2, "")?;
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

fn fmt_cal_body(
    f: &mut fmt::Formatter<'_>,
    body: &CalibrationBody,
    symbols: &SymbolTable,
    indent: usize,
    prefix: &str,
) -> fmt::Result {
    match body {
        CalibrationBody::Opaque(s) => {
            if prefix.is_empty() {
                writeln!(f, "{{{s}}}")
            } else {
                writeln!(f, "{prefix} {{{s}}}")
            }
        }
        CalibrationBody::OpenPulse(stmts) => {
            if prefix.is_empty() {
                writeln!(f, "{{")?;
            } else {
                writeln!(f, "{prefix} {{")?;
            }
            for stmt in stmts {
                fmt_stmt(f, stmt, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
    }
}

// ── Statement display ───────────────────────────────────────────────

fn pad(f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    for _ in 0..indent {
        write!(f, " ")?;
    }
    Ok(())
}

fn fmt_annotations(
    f: &mut fmt::Formatter<'_>,
    annotations: &[Annotation],
    indent: usize,
) -> fmt::Result {
    for ann in annotations {
        pad(f, indent)?;
        write!(f, "@{}", ann.keyword)?;
        if let Some(ref content) = ann.content {
            write!(f, " {content}")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn fmt_alias<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    a: &Alias<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "alias {} = ", a.symbol)?;
    for (i, v) in a.value.iter().enumerate() {
        if i > 0 {
            write!(f, " ++ ")?;
        }
        v.fmt_e(f, symbols)?;
    }
    writeln!(f)
}

fn fmt_gate_call<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    g: &GateCall<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "gate_call ")?;
    for m in &g.modifiers {
        fmt_gate_modifier(f, m, symbols)?;
        write!(f, " ")?;
    }
    write!(f, "{}", g.gate)?;
    write!(f, " (")?;
    for (i, arg) in g.args.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        arg.fmt_e(f, symbols)?;
    }
    write!(f, ") [")?;
    for (i, q) in g.qubits.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        fmt_qubit_op(f, q, symbols)?;
    }
    writeln!(f, "]")
}

fn fmt_measure_stmt<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    m: &Measure<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    fmt_measure(f, &m.measure, symbols)?;
    if let Some(lv) = &m.target {
        write!(f, " -> ")?;
        fmt_lvalue(f, lv, symbols)?;
    }
    writeln!(f)
}

fn fmt_reset<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    operand: &QubitOperand<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "reset ")?;
    fmt_qubit_op(f, operand, symbols)?;
    writeln!(f)
}

fn fmt_barrier<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    operands: &[QubitOperand<E>],
    symbols: &SymbolTable,
) -> fmt::Result {
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

fn fmt_delay<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    d: &Delay<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "delay ")?;
    d.duration.fmt_e(f, symbols)?;
    for q in &d.operands {
        write!(f, " ")?;
        fmt_qubit_op(f, q, symbols)?;
    }
    writeln!(f)
}

fn fmt_box_stmt(
    f: &mut fmt::Formatter<'_>,
    b: &BoxStmt,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    write!(f, "box")?;
    if let Some(dur) = &b.duration {
        write!(f, "[")?;
        fmt_expr(f, dur, symbols)?;
        write!(f, "]")?;
    }
    writeln!(f, " {{")?;
    for s in &b.body {
        fmt_stmt(f, s, symbols, indent + 2)?;
    }
    pad(f, indent)?;
    writeln!(f, "}}")
}

fn fmt_assignment<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    a: &Assignment<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "assign ")?;
    fmt_lvalue(f, &a.target, symbols)?;
    write!(f, " = ")?;
    match &a.value {
        RValue::Expr(e) => e.fmt_e(f, symbols)?,
        RValue::Measure(m) => fmt_measure(f, m, symbols)?,
    }
    writeln!(f)
}

fn fmt_nop<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    operands: &[QubitOperand<E>],
    symbols: &SymbolTable,
) -> fmt::Result {
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

fn fmt_stmt(
    f: &mut fmt::Formatter<'_>,
    stmt: &Stmt,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    fmt_annotations(f, &stmt.annotations, indent)?;

    pad(f, indent)?;
    match &stmt.kind {
        StmtKind::Alias(a) => fmt_alias(f, a, symbols),
        StmtKind::GateCall(g) => fmt_gate_call(f, g, symbols),
        StmtKind::Measure(m) => fmt_measure_stmt(f, m, symbols),
        StmtKind::Reset(operand) => fmt_reset(f, operand, symbols),
        StmtKind::Barrier(operands) => fmt_barrier(f, operands, symbols),
        StmtKind::Delay(d) => fmt_delay(f, d, symbols),
        StmtKind::Box(b) => fmt_box_stmt(f, b, symbols, indent),
        StmtKind::Assignment(a) => fmt_assignment(f, a, symbols),
        StmtKind::If(If {
            condition,
            then_body,
            else_body,
        }) => {
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
        StmtKind::For(For {
            var,
            iterable,
            body,
        }) => {
            write!(f, "for {var} in ")?;
            fmt_for_iterable(f, iterable, symbols)?;
            writeln!(f, " {{")?;
            for s in body {
                fmt_stmt(f, s, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
        StmtKind::While(While { condition, body }) => {
            write!(f, "while ")?;
            fmt_expr(f, condition, symbols)?;
            writeln!(f, " {{")?;
            for s in body {
                fmt_stmt(f, s, symbols, indent + 2)?;
            }
            pad(f, indent)?;
            writeln!(f, "}}")
        }
        StmtKind::Switch(Switch { target, cases }) => {
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
                    RValue::Expr(e) => fmt_expr(f, e, symbols)?,
                    RValue::Measure(m) => fmt_measure(f, m, symbols)?,
                }
            }
            writeln!(f)
        }
        StmtKind::End => writeln!(f, "end"),
        StmtKind::Pragma(content) => writeln!(f, "pragma {content}"),
        StmtKind::Cal(body) => fmt_cal_body(f, body, symbols, indent, "cal"),
        StmtKind::ExprStmt(expr) => {
            fmt_expr(f, expr, symbols)?;
            writeln!(f)
        }
        StmtKind::Nop(operands) => fmt_nop(f, operands, symbols),
    }
}

pub(crate) fn fmt_block_stmt(
    f: &mut fmt::Formatter<'_>,
    stmt: &BlockStmt,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    fmt_annotations(f, &stmt.annotations, indent)?;
    pad(f, indent)?;
    match &stmt.kind {
        BlockStmtKind::Alias(a) => fmt_alias(f, a, symbols),
        BlockStmtKind::GateCall(g) => fmt_gate_call(f, g, symbols),
        BlockStmtKind::Measure(m) => fmt_measure_stmt(f, m, symbols),
        BlockStmtKind::Reset(operand) => fmt_reset(f, operand, symbols),
        BlockStmtKind::Barrier(operands) => fmt_barrier(f, operands, symbols),
        BlockStmtKind::Delay(d) => fmt_delay(f, d, symbols),
        BlockStmtKind::Box(b) => fmt_block_box(f, b, symbols, indent),
        BlockStmtKind::Assignment(a) => fmt_assignment(f, a, symbols),
        BlockStmtKind::Pragma(content) => writeln!(f, "pragma {content}"),
        BlockStmtKind::Cal(body) => fmt_block_cal_body(f, body, symbols, indent),
        BlockStmtKind::ExprStmt(expr) => {
            fmt_block_expr(f, expr, symbols)?;
            writeln!(f)
        }
        BlockStmtKind::Nop(operands) => fmt_nop(f, operands, symbols),
    }
}

fn fmt_block_box(
    f: &mut fmt::Formatter<'_>,
    b: &BlockBoxStmt,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    write!(f, "box")?;
    if let Some(dur) = &b.duration {
        write!(f, "[")?;
        fmt_block_expr(f, dur, symbols)?;
        write!(f, "]")?;
    }
    writeln!(f, " {{")?;
    fmt_block_cfg(f, &b.body, symbols, indent + 2)?;
    pad(f, indent)?;
    writeln!(f, "}}")
}

fn fmt_block_cal_body(
    f: &mut fmt::Formatter<'_>,
    body: &BlockCalibrationBody,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    match body {
        BlockCalibrationBody::Opaque(s) => writeln!(f, "cal {{ {s} }}"),
        BlockCalibrationBody::OpenPulse(cfg) => {
            writeln!(f, "cal {{")?;
            fmt_block_cfg(f, cfg, symbols, indent + 2)?;
            pad(f, indent)?;
            writeln!(f, "}}")
        }
    }
}

fn fmt_basic_block_id(f: &mut fmt::Formatter<'_>, id: BasicBlockId) -> fmt::Result {
    write!(f, "bb{}", id.0)
}

fn fmt_terminator(
    f: &mut fmt::Formatter<'_>,
    term: &Terminator,
    symbols: &SymbolTable,
) -> fmt::Result {
    match term {
        Terminator::Goto(target) => {
            write!(f, "goto ")?;
            fmt_basic_block_id(f, *target)?;
            writeln!(f)
        }
        Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } => {
            write!(f, "branch ")?;
            fmt_block_expr(f, cond, symbols)?;
            write!(f, " ? ")?;
            fmt_basic_block_id(f, *then_bb)?;
            write!(f, " : ")?;
            fmt_basic_block_id(f, *else_bb)?;
            writeln!(f)
        }
        Terminator::Switch {
            target,
            cases,
            default,
        } => {
            write!(f, "switch ")?;
            fmt_block_expr(f, target, symbols)?;
            writeln!(f, " {{")?;
            for (labels, bb) in cases {
                match labels {
                    SwitchLabels::Values(vals) => {
                        for (i, v) in vals.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            fmt_block_expr(f, v, symbols)?;
                        }
                    }
                    SwitchLabels::Default => write!(f, "default")?,
                }
                write!(f, " -> ")?;
                fmt_basic_block_id(f, *bb)?;
                writeln!(f)?;
            }
            if let Some(d) = default {
                write!(f, "default -> ")?;
                fmt_basic_block_id(f, *d)?;
                writeln!(f)?;
            }
            writeln!(f, "}}")
        }
        Terminator::Return(val) => {
            write!(f, "return")?;
            if let Some(rv) = val {
                write!(f, " ")?;
                match rv {
                    RValue::Expr(e) => fmt_block_expr(f, e, symbols)?,
                    RValue::Measure(m) => fmt_measure(f, m, symbols)?,
                }
            }
            writeln!(f)
        }
        Terminator::End => writeln!(f, "end"),
        Terminator::Unreachable => writeln!(f, "unreachable"),
    }
}

fn fmt_block_cfg(
    f: &mut fmt::Formatter<'_>,
    cfg: &Cfg,
    symbols: &SymbolTable,
    indent: usize,
) -> fmt::Result {
    for block in &cfg.blocks {
        pad(f, indent)?;
        fmt_basic_block_id(f, block.id)?;
        writeln!(f, ":")?;
        for stmt in &block.stmts {
            fmt_block_stmt(f, stmt, symbols, indent + 2)?;
        }
        pad(f, indent + 2)?;
        fmt_terminator(f, &block.terminator, symbols)?;
    }
    Ok(())
}

impl fmt::Display for CfgDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_block_cfg(f, self.cfg, self.symbols, 0)
    }
}

// ── Expression display ──────────────────────────────────────────────

fn fmt_expr(f: &mut fmt::Formatter<'_>, expr: &Expr, symbols: &SymbolTable) -> fmt::Result {
    match &expr.kind {
        ExprKind::Literal(v) => write!(f, "{}", v),
        ExprKind::Var(id) => write!(f, "{id}"),
        ExprKind::HardwareQubit(n) => write!(f, "${n}"),
        ExprKind::Binary(Binary { op, left, right }) => {
            write!(f, "(")?;
            fmt_expr(f, left, symbols)?;
            write!(f, " {op} ")?;
            fmt_expr(f, right, symbols)?;
            write!(f, ")")
        }
        ExprKind::Unary(Unary { op, operand }) => {
            write!(f, "({op}")?;
            fmt_expr(f, operand, symbols)?;
            write!(f, ")")
        }
        ExprKind::Cast(Cast { target_ty, operand }) => {
            write!(f, "{target_ty}(")?;
            fmt_expr(f, operand, symbols)?;
            write!(f, ")")
        }
        ExprKind::Index(Index { base, index }) => {
            fmt_expr(f, base, symbols)?;
            write!(f, "[")?;
            fmt_index(f, index, symbols)?;
            write!(f, "]")
        }
        ExprKind::Call(Call { callee, args }) => {
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
        ExprKind::ArrayLiteral(arr) => fmt_array_literal(f, arr, symbols),
    }
}

fn fmt_block_expr(
    f: &mut fmt::Formatter<'_>,
    expr: &BlockExpr,
    symbols: &SymbolTable,
) -> fmt::Result {
    match &expr.kind {
        BlockExprKind::Literal(v) => write!(f, "{}", v),
        BlockExprKind::Var(id) => write!(f, "{id}"),
        BlockExprKind::HardwareQubit(n) => write!(f, "${n}"),
        BlockExprKind::Binary(Binary { op, left, right }) => {
            write!(f, "(")?;
            fmt_block_expr(f, left, symbols)?;
            write!(f, " {op} ")?;
            fmt_block_expr(f, right, symbols)?;
            write!(f, ")")
        }
        BlockExprKind::Unary(Unary { op, operand }) => {
            write!(f, "({op}")?;
            fmt_block_expr(f, operand, symbols)?;
            write!(f, ")")
        }
        BlockExprKind::Cast(Cast { target_ty, operand }) => {
            write!(f, "{target_ty}(")?;
            fmt_block_expr(f, operand, symbols)?;
            write!(f, ")")
        }
        BlockExprKind::Index(Index { base, index }) => {
            fmt_block_expr(f, base, symbols)?;
            write!(f, "[")?;
            fmt_index(f, index, symbols)?;
            write!(f, "]")
        }
        BlockExprKind::Call(Call { callee, args }) => {
            match callee {
                CallTarget::Symbol(id) => write!(f, "{id}")?,
                CallTarget::Intrinsic(i) => write!(f, "{i}")?,
            }
            write!(f, "(")?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_block_expr(f, arg, symbols)?;
            }
            write!(f, ")")
        }
        BlockExprKind::DurationOf(_) => write!(f, "durationof {{ <cfg> }}"),
        BlockExprKind::ArrayLiteral(arr) => fmt_array_literal(f, arr, symbols),
    }
}

// ── Helper displays ─────────────────────────────────────────────────

fn fmt_qubit_op<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    op: &QubitOperand<E>,
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

fn fmt_lvalue<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    lv: &LValue<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
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

fn fmt_index<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    idx: &IndexOp<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    match &idx.kind {
        IndexKind::Set(exprs) => {
            write!(f, "{{")?;
            for (i, e) in exprs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                e.fmt_e(f, symbols)?;
            }
            write!(f, "}}")
        }
        IndexKind::Items(items) => {
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match item {
                    IndexItem::Single(e) => e.fmt_e(f, symbols)?,
                    IndexItem::Range(r) => fmt_range(f, r, symbols)?,
                }
            }
            Ok(())
        }
    }
}

fn fmt_range<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    range: &RangeExpr<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    if let Some(start) = &range.start {
        start.fmt_e(f, symbols)?;
    }
    write!(f, ":")?;
    if let Some(step) = &range.step {
        step.fmt_e(f, symbols)?;
        write!(f, ":")?;
    }
    if let Some(end) = &range.end {
        end.fmt_e(f, symbols)?;
    }
    Ok(())
}

fn fmt_gate_modifier<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    m: &GateModifier<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    match m {
        GateModifier::Inv => write!(f, "inv @"),
        GateModifier::Pow(expr) => {
            write!(f, "pow(")?;
            expr.fmt_e(f, symbols)?;
            write!(f, ") @")
        }
        GateModifier::Ctrl(n) => write!(f, "ctrl({n}) @"),
        GateModifier::NegCtrl(n) => write!(f, "negctrl({n}) @"),
    }
}

fn fmt_measure<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    m: &MeasureExpr<E>,
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
                arg.fmt_e(f, symbols)?;
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

fn fmt_array_literal<E: FmtExpr>(
    f: &mut fmt::Formatter<'_>,
    arr: &ArrayLiteral<E>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "{{")?;
    for (i, item) in arr.items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        item.fmt_e(f, symbols)?;
    }
    write!(f, "}}")
}
