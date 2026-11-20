use std::fmt;

use crate::sir::*;
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::Type;

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

impl fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── Error display ───────────────────────────────────────────────────

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
            for s in &gate.body {
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

fn fmt_alias(f: &mut fmt::Formatter<'_>, a: &Alias<Expr>, symbols: &SymbolTable) -> fmt::Result {
    write!(f, "alias {} = ", a.symbol)?;
    for (i, v) in a.value.iter().enumerate() {
        if i > 0 {
            write!(f, " ++ ")?;
        }
        fmt_expr(f, v, symbols)?;
    }
    writeln!(f)
}

fn fmt_gate_call(
    f: &mut fmt::Formatter<'_>,
    g: &GateCall<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "gate_call ")?;
    for m in &g.modifiers {
        fmt_gate_modifier(f, m, symbols)?;
        write!(f, " ")?;
    }
    write!(f, "{}", g.gate)?;
    if let Some(d) = &g.duration {
        write!(f, " dur<")?;
        fmt_expr(f, d, symbols)?;
        write!(f, ">")?;
    }
    write!(f, " (")?;
    for (i, arg) in g.args.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        fmt_expr(f, arg, symbols)?;
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

fn fmt_measure_stmt(
    f: &mut fmt::Formatter<'_>,
    m: &Measure<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
    fmt_measure(f, &m.measure, symbols)?;
    if let Some(lv) = &m.target {
        write!(f, " -> ")?;
        fmt_lvalue(f, lv, symbols)?;
    }
    writeln!(f)
}

fn fmt_reset(
    f: &mut fmt::Formatter<'_>,
    operand: &QubitOperand<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "reset ")?;
    fmt_qubit_op(f, operand, symbols)?;
    writeln!(f)
}

fn fmt_barrier(
    f: &mut fmt::Formatter<'_>,
    operands: &[QubitOperand<Expr>],
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

fn fmt_delay(f: &mut fmt::Formatter<'_>, d: &Delay<Expr>, symbols: &SymbolTable) -> fmt::Result {
    write!(f, "delay ")?;
    fmt_expr(f, &d.duration, symbols)?;
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

fn fmt_assignment(
    f: &mut fmt::Formatter<'_>,
    a: &Assignment<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "assign ")?;
    fmt_lvalue(f, &a.target, symbols)?;
    write!(f, " = ")?;
    match &a.value {
        RValue::Expr(e) => fmt_expr(f, e, symbols)?,
        RValue::Measure(m) => fmt_measure(f, m, symbols)?,
    }
    writeln!(f)
}

fn fmt_nop(
    f: &mut fmt::Formatter<'_>,
    operands: &[QubitOperand<Expr>],
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

// ── Helper displays ─────────────────────────────────────────────────

fn fmt_qubit_op(
    f: &mut fmt::Formatter<'_>,
    op: &QubitOperand<Expr>,
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

fn fmt_lvalue(f: &mut fmt::Formatter<'_>, lv: &LValue<Expr>, symbols: &SymbolTable) -> fmt::Result {
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

fn fmt_index(
    f: &mut fmt::Formatter<'_>,
    idx: &IndexOp<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
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

fn fmt_range(
    f: &mut fmt::Formatter<'_>,
    range: &RangeExpr<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
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
    m: &GateModifier<Expr>,
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
    m: &MeasureExpr<Expr>,
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

fn fmt_array_literal(
    f: &mut fmt::Formatter<'_>,
    arr: &ArrayLiteral<Expr>,
    symbols: &SymbolTable,
) -> fmt::Result {
    write!(f, "{{")?;
    for (i, item) in arr.items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        fmt_expr(f, item, symbols)?;
    }
    write!(f, "}}")
}
