//! Textual disassembly. Format is not standardized; intended for
//! debugging and golden-file tests.

use std::fmt;

use super::types::{
    BcAliasSegment, BcBlock, BcCalArg, BcCalBody, BcCalOperand, BcCalTarget, BcCallTarget,
    BcGateModifier, BcIndex, BcInstr, BcModule, BcOp, BcOperand, BcTerminator, BlockId, ConstId,
    ProcId, ProcOwner, QubitSource, Reg, StringId,
};

impl fmt::Display for BcModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ".module openqasm 3")?;
        writeln!(f, ".version {}.{}", self.version.major, self.version.minor)?;
        writeln!(f, ".entry proc{}", self.entry.0)?;
        if let Some(grammar) = &self.calibration_grammar {
            writeln!(f, ".defcalgrammar {grammar}")?;
        }

        if !self.symbols.is_empty() {
            writeln!(f)?;
            writeln!(f, ".symbols")?;
            for (i, sym) in self.symbols.iter().enumerate() {
                write!(f, "  s{i} {:?} {}", sym.kind, sym.name)?;
                if let Some(ty) = sym.ty {
                    write!(f, " : {ty}")?;
                }
                writeln!(f)?;
            }
        }

        if !self.constants.is_empty() {
            writeln!(f)?;
            writeln!(f, ".constants")?;
            for (i, c) in self.constants.iter().enumerate() {
                writeln!(f, "  k{i} = {c}")?;
            }
        }

        if !self.strings.is_empty() {
            writeln!(f)?;
            writeln!(f, ".strings")?;
            for (i, s) in self.strings.iter().enumerate() {
                writeln!(f, "  $str{i} = {s:?}")?;
            }
        }

        if self.qubits.num_qubits > 0 || !self.qubits.regions.is_empty() {
            writeln!(f)?;
            writeln!(f, ".qubits {}", self.qubits.num_qubits)?;
            for (i, region) in self.qubits.regions.iter().enumerate() {
                write!(f, "  qr{i} =")?;
                for (j, (start, end)) in region.ranges.iter().enumerate() {
                    if j > 0 {
                        write!(f, " ++")?;
                    }
                    write!(f, " [{start}..{end})")?;
                }
                if let Some(sym) = region.origin {
                    write!(f, " ; {}", self.symbol(sym).name)?;
                }
                writeln!(f)?;
            }
        }

        if !self.calibrations.is_empty() {
            writeln!(f)?;
            writeln!(f, ".calibrations")?;
            for (i, cal) in self.calibrations.iter().enumerate() {
                write!(f, "  cal{i} = ")?;
                match &cal.target {
                    BcCalTarget::Measure => write!(f, "measure")?,
                    BcCalTarget::Reset => write!(f, "reset")?,
                    BcCalTarget::Delay => write!(f, "delay")?,
                    BcCalTarget::Gate(s) => write!(f, "gate(s{})", s.0)?,
                }
                if !cal.args.is_empty() {
                    write!(f, "(")?;
                    for (j, a) in cal.args.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        match a {
                            BcCalArg::Param(s) => write!(f, "s{}", s.0)?,
                            BcCalArg::Unsupported => write!(f, "?")?,
                        }
                    }
                    write!(f, ")")?;
                }
                for (j, op) in cal.operands.iter().enumerate() {
                    write!(f, "{}", if j == 0 { " " } else { ", " })?;
                    match op {
                        BcCalOperand::Hardware(n) => write!(f, "${n}")?,
                        BcCalOperand::Any => write!(f, "any")?,
                    }
                }
                if cal.has_return {
                    write!(f, " -> ret")?;
                }
                match &cal.body {
                    BcCalBody::OpenPulse(p) => writeln!(f, " => {}", fmt_proc(*p))?,
                    BcCalBody::Opaque(s) => writeln!(f, " => {}", fmt_str(*s))?,
                }
            }
        }

        for (i, proc) in self.procedures.iter().enumerate() {
            writeln!(f)?;
            write!(f, ".proc {} ", i)?;
            fmt_owner(f, &proc.owner)?;
            write!(f, " (registers={}", proc.register_types.len())?;
            if !proc.params.is_empty() {
                write!(f, ", params=[")?;
                for (j, r) in proc.params.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "r{}", r.0)?;
                }
                write!(f, "]")?;
            }
            writeln!(f, ")")?;
            for block in &proc.blocks {
                fmt_block(f, block)?;
            }
        }
        Ok(())
    }
}

fn fmt_owner(f: &mut fmt::Formatter<'_>, owner: &ProcOwner) -> fmt::Result {
    match owner {
        ProcOwner::TopLevel => write!(f, "top_level"),
        ProcOwner::Subroutine(s) => write!(f, "subroutine(s{})", s.0),
        ProcOwner::Gate(s) => write!(f, "gate(s{})", s.0),
        ProcOwner::Calibration(i) => write!(f, "calibration({i})"),
        ProcOwner::InlineCal => write!(f, "inline_cal"),
        ProcOwner::DurationOf => write!(f, "durationof"),
    }
}

fn fmt_block(f: &mut fmt::Formatter<'_>, block: &BcBlock) -> fmt::Result {
    writeln!(f, "  bb{}:", block.id.0)?;
    for instr in &block.instrs {
        write!(f, "    ")?;
        fmt_instr(f, instr)?;
        writeln!(f)?;
    }
    write!(f, "    ")?;
    fmt_terminator(f, &block.terminator)?;
    writeln!(f)
}

fn fmt_instr(f: &mut fmt::Formatter<'_>, instr: &BcInstr) -> fmt::Result {
    fmt_op(f, &instr.op)
}

fn fmt_op(f: &mut fmt::Formatter<'_>, op: &BcOp) -> fmt::Result {
    match op {
        BcOp::Binary { op, dest, lhs, rhs } => {
            write!(f, "{} = {} ", fmt_reg(*dest), mnemonic(op))?;
            fmt_operand(f, lhs)?;
            write!(f, ", ")?;
            fmt_operand(f, rhs)
        }
        BcOp::Unary { op, dest, src } => {
            write!(f, "{} = {} ", fmt_reg(*dest), mnemonic(op))?;
            fmt_operand(f, src)
        }
        BcOp::Cast {
            dest,
            target_ty,
            src,
        } => {
            write!(f, "{} = cast {} ", fmt_reg(*dest), target_ty)?;
            fmt_operand(f, src)
        }
        BcOp::Move { dest, src } => {
            write!(f, "{} = move ", fmt_reg(*dest))?;
            fmt_operand(f, src)
        }
        BcOp::LoadElement {
            dest,
            base,
            indices,
        } => {
            write!(f, "{} = load_elem ", fmt_reg(*dest))?;
            fmt_operand(f, base)?;
            fmt_index_list(f, indices)
        }
        BcOp::StoreElement {
            target,
            indices,
            value,
        } => {
            write!(f, "store_elem {}", fmt_reg(*target))?;
            fmt_index_list(f, indices)?;
            write!(f, " = ")?;
            fmt_operand(f, value)
        }
        BcOp::Borrow {
            dest,
            base,
            indices,
        } => {
            write!(f, "{} = borrow {}", fmt_reg(*dest), fmt_reg(*base))?;
            fmt_index_list(f, indices)
        }
        BcOp::Release { reg } => write!(f, "release {}", fmt_reg(*reg)),
        BcOp::NewArray { dest, items } => {
            write!(f, "{} = new_array [", fmt_reg(*dest))?;
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_operand(f, it)?;
            }
            write!(f, "]")
        }
        BcOp::Call { dest, callee, args } => {
            if let Some(d) = dest {
                write!(f, "{} = ", fmt_reg(*d))?;
            }
            write!(f, "call ")?;
            fmt_call_target(f, callee)?;
            write!(f, "(")?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_operand(f, a)?;
            }
            write!(f, ")")
        }
        BcOp::GateCall {
            gate,
            modifiers,
            args,
            qubits,
        } => {
            write!(f, "gate_call ")?;
            for m in modifiers {
                fmt_gate_modifier(f, m)?;
                write!(f, " ")?;
            }
            write!(f, "s{}(", gate.0)?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_operand(f, a)?;
            }
            write!(f, ") [")?;
            for (i, q) in qubits.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_operand(f, q)?;
            }
            write!(f, "]")
        }
        BcOp::Measure { dest, qubit } => {
            if let Some(d) = dest {
                write!(f, "{} = ", fmt_reg(*d))?;
            }
            write!(f, "measure ")?;
            fmt_operand(f, qubit)
        }
        BcOp::Reset { qubit } => {
            write!(f, "reset ")?;
            fmt_operand(f, qubit)
        }
        BcOp::Barrier { qubits } => {
            write!(f, "barrier")?;
            for q in qubits {
                write!(f, " ")?;
                fmt_operand(f, q)?;
            }
            Ok(())
        }
        BcOp::Delay { duration, qubits } => {
            write!(f, "delay ")?;
            fmt_operand(f, duration)?;
            for q in qubits {
                write!(f, " ")?;
                fmt_operand(f, q)?;
            }
            Ok(())
        }
        BcOp::Nop { qubits } => {
            write!(f, "nop")?;
            for q in qubits {
                write!(f, " ")?;
                fmt_operand(f, q)?;
            }
            Ok(())
        }
        BcOp::CalOpaque { content } => write!(f, "cal_opaque {}", fmt_str(*content)),
        BcOp::CalOpenPulse { body } => write!(f, "cal_openpulse -> {}", fmt_proc(*body)),
        BcOp::CalLoad { dest, symbol } => {
            write!(f, "{} = cal_load s{}", fmt_reg(*dest), symbol.0)
        }
        BcOp::CalStore { symbol, src } => {
            write!(f, "cal_store s{} = ", symbol.0)?;
            fmt_operand(f, src)
        }
        BcOp::DurationOf { dest, body } => {
            write!(f, "{} = durationof -> {}", fmt_reg(*dest), fmt_proc(*body))
        }
        BcOp::Pragma { content } => write!(f, "pragma {}", fmt_str(*content)),
        BcOp::Alias { symbol, value } => {
            write!(f, "alias s{} = ", symbol.0)?;
            for (i, v) in value.iter().enumerate() {
                if i > 0 {
                    write!(f, " ++ ")?;
                }
                fmt_operand(f, v)?;
            }
            Ok(())
        }
        BcOp::AliasBind { slot, segments } => {
            write!(f, "aliasbind qa{slot} = ")?;
            for (i, seg) in segments.iter().enumerate() {
                if i > 0 {
                    write!(f, " ++ ")?;
                }
                fmt_alias_segment(f, seg)?;
            }
            Ok(())
        }
    }
}

fn fmt_alias_segment(f: &mut fmt::Formatter<'_>, seg: &BcAliasSegment) -> fmt::Result {
    match seg {
        BcAliasSegment::Operand(op) => fmt_operand(f, op),
        BcAliasSegment::Slice {
            source,
            start,
            step,
            end,
        } => {
            fmt_operand(f, source)?;
            write!(f, "[")?;
            if let Some(s) = start {
                fmt_operand(f, s)?;
            }
            write!(f, ":")?;
            if let Some(s) = step {
                fmt_operand(f, s)?;
                write!(f, ":")?;
            }
            if let Some(e) = end {
                fmt_operand(f, e)?;
            }
            write!(f, "]")
        }
    }
}

fn fmt_terminator(f: &mut fmt::Formatter<'_>, term: &BcTerminator) -> fmt::Result {
    match term {
        BcTerminator::Goto(b) => write!(f, "goto {}", fmt_block_id(*b)),
        BcTerminator::Branch {
            cond,
            then_bb,
            else_bb,
        } => {
            write!(f, "branch ")?;
            fmt_operand(f, cond)?;
            write!(
                f,
                " ? {} : {}",
                fmt_block_id(*then_bb),
                fmt_block_id(*else_bb)
            )
        }
        BcTerminator::Switch {
            target,
            cases,
            default,
        } => {
            write!(f, "switch ")?;
            fmt_operand(f, target)?;
            for (labels, bb) in cases {
                write!(f, " | ")?;
                for (i, v) in labels.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_operand(f, v)?;
                }
                write!(f, " -> {}", fmt_block_id(*bb))?;
            }
            write!(f, " | default -> {}", fmt_block_id(*default))
        }
        BcTerminator::Box {
            duration,
            body,
            after,
        } => {
            write!(f, "box[")?;
            fmt_operand(f, duration)?;
            write!(f, "] {} then {}", fmt_block_id(*body), fmt_block_id(*after))
        }
        BcTerminator::Return(rv) => {
            write!(f, "return")?;
            if let Some(v) = rv {
                write!(f, " ")?;
                fmt_operand(f, v)?;
            }
            Ok(())
        }
        BcTerminator::End => write!(f, "end"),
        BcTerminator::Unreachable => write!(f, "unreachable"),
    }
}

/// An operator's mnemonic: its variant name in lower case (`add`, `lte`,
/// `bitnot`, `sin`, …).
fn mnemonic(op: &impl fmt::Debug) -> String {
    format!("{op:?}").to_lowercase()
}

/// Render `[i0, i1, …]` for an element access's per-dimension indices.
fn fmt_index_list(f: &mut fmt::Formatter<'_>, indices: &[BcIndex]) -> fmt::Result {
    write!(f, "[")?;
    for (i, idx) in indices.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        match idx {
            BcIndex::Item(position) => fmt_operand(f, position)?,
            BcIndex::Set(members) => {
                write!(f, "{{")?;
                for (i, member) in members.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_operand(f, member)?;
                }
                write!(f, "}}")?;
            }
            BcIndex::Range { start, step, end } => {
                for (i, bound) in [start, step, end].into_iter().enumerate() {
                    // The step and its colon are left out when it is.
                    if i > 0 && (i != 1 || bound.is_some()) {
                        write!(f, ":")?;
                    }
                    if let Some(bound) = bound {
                        fmt_operand(f, bound)?;
                    }
                }
            }
        }
    }
    write!(f, "]")
}

fn fmt_operand(f: &mut fmt::Formatter<'_>, op: &BcOperand) -> fmt::Result {
    match op {
        BcOperand::Reg(r) => write!(f, "{}", fmt_reg(*r)),
        BcOperand::Const(c) => write!(f, "{}", fmt_const(*c)),
        BcOperand::HardwareQubit(n) => write!(f, "${n}"),
        BcOperand::Qubit(n) => write!(f, "q@{n}"),
        BcOperand::Whole(source) => fmt_qubit_source(f, source),
        BcOperand::Select { source, positions } => {
            fmt_qubit_source(f, source)?;
            write!(f, "{positions:?}")
        }
        BcOperand::Index { source, index } => {
            fmt_qubit_source(f, source)?;
            write!(f, "[")?;
            fmt_operand(f, index)?;
            write!(f, "]")
        }
    }
}

/// Render a qubit source: `qr{n}` for a global memory region, `qs{n}`
/// for a frame-local qubit slot (parameter or runtime alias).
fn fmt_qubit_source(f: &mut fmt::Formatter<'_>, source: &QubitSource) -> fmt::Result {
    match source {
        QubitSource::Region(id) => write!(f, "qr{}", id.0),
        QubitSource::Slot(s) => write!(f, "qs{s}"),
    }
}

fn fmt_call_target(f: &mut fmt::Formatter<'_>, t: &BcCallTarget) -> fmt::Result {
    match t {
        BcCallTarget::Symbol(s) => write!(f, "s{}", s.0),
        BcCallTarget::Intrinsic(i) => write!(f, "{i}"),
    }
}

fn fmt_gate_modifier(f: &mut fmt::Formatter<'_>, m: &BcGateModifier) -> fmt::Result {
    match m {
        BcGateModifier::Inv => write!(f, "inv @"),
        BcGateModifier::Pow(o) => {
            write!(f, "pow(")?;
            fmt_operand(f, o)?;
            write!(f, ") @")
        }
        BcGateModifier::Ctrl(n) => write!(f, "ctrl({n}) @"),
        BcGateModifier::NegCtrl(n) => write!(f, "negctrl({n}) @"),
    }
}

fn fmt_reg(r: Reg) -> String {
    format!("%{}", r.0)
}

fn fmt_const(c: ConstId) -> String {
    format!("k{}", c.0)
}

fn fmt_str(s: StringId) -> String {
    format!("$str{}", s.0)
}

fn fmt_proc(p: ProcId) -> String {
    format!("proc{}", p.0)
}

fn fmt_block_id(b: BlockId) -> String {
    format!("bb{}", b.0)
}
