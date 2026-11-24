//! Consistency checks on a [`Program`], so a decoded one can't send the VM
//! out of bounds.
//!
//! The VM indexes procedures, blocks, registers, constants, strings, symbols,
//! qubit regions and frame slots directly. [`Program::from_bytes`] runs
//! [`validate`] on every program it decodes, and the emitter checks its own
//! output with [`check`] in debug builds.

use oqi_classical::{Array, Scalar, Value};

use super::types::*;
use crate::symbol::SymbolId;

/// `program` with its constants rebuilt through the checked constructors,
/// once [`check`] finds nothing out of range.
pub(crate) fn validate(mut program: Program) -> Result<Program, String> {
    program.constants = std::mem::take(&mut program.constants)
        .into_iter()
        .enumerate()
        .map(|(i, value)| well_formed(value).map_err(|why| format!("constant c{i}: {why}")))
        .collect::<Result<_, _>>()?;
    check(&program)?;
    Ok(program)
}

/// A constant rebuilt so its value fits its type.
fn well_formed(value: Value) -> Result<Value, String> {
    match value {
        Value::Scalar(scalar) => {
            let ty = scalar.ty();
            Scalar::new(scalar.into_value(), ty)
                .map(Value::Scalar)
                .map_err(|e| e.to_string())
        }
        Value::Array(array) => {
            let ty = array.ty();
            Array::new(array.into_values(), ty)
                .map(Value::Array)
                .map_err(|e| e.to_string())
        }
        Value::ArrayRef(_) => Err("a constant cannot be an array reference".into()),
    }
}

/// Every index in `program` names something that exists.
pub(crate) fn check(program: &Program) -> Result<(), String> {
    let p = Checker { program };
    let entry = p.procedure(program.entry)?;
    if !matches!(entry.owner, ProcOwner::TopLevel) {
        return Err(format!(
            "entry p{} is not a top-level body",
            program.entry.0
        ));
    }
    for &(symbol, reg) in program.inputs.iter().chain(&program.outputs) {
        p.symbol(symbol)?;
        p.reg(entry, reg)?;
    }
    for (i, region) in program.qubits.regions.iter().enumerate() {
        for &(start, end) in &region.ranges {
            if start > end || end > program.qubits.num_qubits {
                return Err(format!(
                    "qubit region {i} covers {start}..{end} of {} qubits",
                    program.qubits.num_qubits
                ));
            }
        }
        if let Some(origin) = region.origin {
            p.symbol(origin)?;
        }
    }
    for (i, cal) in program.calibrations.iter().enumerate() {
        if let CalTarget::Gate(gate) = cal.target {
            p.symbol(gate)?;
        }
        for arg in &cal.args {
            if let CalArg::Param(symbol) = arg {
                p.symbol(*symbol)?;
            }
        }
        match cal.body {
            CalBody::OpenPulse(id) => {
                let body = p.procedure(id)?;
                if !matches!(body.owner, ProcOwner::Calibration(owner) if owner as usize == i) {
                    return Err(format!("p{} is not the body of calibration {i}", id.0));
                }
            }
            CalBody::Opaque(id) => p.string(id)?,
        }
    }
    for (i, procedure) in program.procedures.iter().enumerate() {
        p.check_procedure(procedure)
            .map_err(|why| format!("p{i}: {why}"))?;
    }
    Ok(())
}

struct Checker<'p> {
    program: &'p Program,
}

impl<'p> Checker<'p> {
    fn procedure(&self, id: ProcId) -> Result<&'p Procedure, String> {
        self.program
            .procedures
            .get(id.0 as usize)
            .ok_or_else(|| format!("no procedure p{}", id.0))
    }

    fn symbol(&self, id: SymbolId) -> Result<(), String> {
        within(id.0, self.program.symbols.len(), "symbol")
    }

    fn string(&self, id: StringId) -> Result<(), String> {
        within(id.0 as usize, self.program.strings.len(), "string")
    }

    fn reg(&self, procedure: &Procedure, reg: Reg) -> Result<(), String> {
        within(reg.0 as usize, procedure.register_types.len(), "register")
    }

    fn check_procedure(&self, procedure: &Procedure) -> Result<(), String> {
        match procedure.owner {
            ProcOwner::Subroutine(symbol) | ProcOwner::Gate(symbol) => self.symbol(symbol)?,
            ProcOwner::Calibration(i) => {
                within(i as usize, self.program.calibrations.len(), "calibration")?
            }
            ProcOwner::TopLevel | ProcOwner::InlineCal | ProcOwner::DurationOf => {}
        }
        for &reg in &procedure.params {
            self.reg(procedure, reg)?;
        }
        // Block ids are positions: each id names one block.
        let mut seen = vec![false; procedure.blocks.len()];
        for block in &procedure.blocks {
            let id = block.id.0 as usize;
            within(id, seen.len(), "block")?;
            if std::mem::replace(&mut seen[id], true) {
                return Err(format!("two blocks are bb{id}"));
            }
        }
        let block = |id: BlockId| within(id.0 as usize, seen.len(), "block");
        block(procedure.entry)?;

        let body = Body {
            checker: self,
            procedure,
        };
        for b in &procedure.blocks {
            for instr in &b.instrs {
                body.op(&instr.op)?;
            }
            match &b.terminator {
                Terminator::Goto(target) => block(*target)?,
                Terminator::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    body.operand(cond)?;
                    block(*then_bb)?;
                    block(*else_bb)?;
                }
                Terminator::Switch {
                    target,
                    cases,
                    default,
                } => {
                    body.operand(target)?;
                    for (labels, target) in cases {
                        body.operands(labels)?;
                        block(*target)?;
                    }
                    block(*default)?;
                }
                Terminator::Box {
                    duration,
                    body: inside,
                    after,
                } => {
                    body.operand(duration)?;
                    block(*inside)?;
                    block(*after)?;
                }
                Terminator::Return(value) => {
                    if let Some(value) = value {
                        body.operand(value)?;
                    }
                }
                Terminator::End | Terminator::Unreachable => {}
            }
        }
        Ok(())
    }
}

/// The checks that depend on the procedure an instruction is in.
struct Body<'c, 'p> {
    checker: &'c Checker<'p>,
    procedure: &'p Procedure,
}

impl Body<'_, '_> {
    fn reg(&self, reg: Reg) -> Result<(), String> {
        self.checker.reg(self.procedure, reg)
    }

    fn slot(&self, slot: u32) -> Result<(), String> {
        within(
            slot as usize,
            self.procedure.num_qubit_slots as usize,
            "qubit slot",
        )
    }

    fn source(&self, source: &QubitSource) -> Result<(), String> {
        match source {
            QubitSource::Region(id) => within(
                id.0 as usize,
                self.checker.program.qubits.regions.len(),
                "qubit region",
            ),
            QubitSource::Slot(slot) => self.slot(*slot),
        }
    }

    fn operand(&self, operand: &Operand) -> Result<(), String> {
        match operand {
            Operand::Reg(reg) => self.reg(*reg),
            Operand::Const(id) => within(
                id.0 as usize,
                self.checker.program.constants.len(),
                "constant",
            ),
            Operand::Qubit(q) => within(
                *q as usize,
                self.checker.program.qubits.num_qubits as usize,
                "qubit",
            ),
            Operand::HardwareQubit(_) => Ok(()),
            Operand::Whole(source) | Operand::Select { source, .. } => self.source(source),
            Operand::Index { source, index } => {
                self.source(source)?;
                self.operand(index)
            }
        }
    }

    fn operands(&self, operands: &[Operand]) -> Result<(), String> {
        operands.iter().try_for_each(|op| self.operand(op))
    }

    fn optional(&self, operand: &Option<impl AsOperand>) -> Result<(), String> {
        match operand {
            Some(operand) => self.operand(operand.as_operand()),
            None => Ok(()),
        }
    }

    fn subscripts(&self, subscripts: &[Subscript]) -> Result<(), String> {
        subscripts.iter().try_for_each(|subscript| match subscript {
            Subscript::Item(op) => self.operand(op),
            Subscript::Set(ops) => self.operands(ops),
            Subscript::Range { start, step, end } => {
                self.optional(start)?;
                self.optional(step)?;
                self.optional(end)
            }
        })
    }

    /// A procedure `op` runs on this frame: it must be one of `owner` and
    /// share this procedure's frame layout.
    fn nested(&self, id: ProcId, what: &str, owner: fn(&ProcOwner) -> bool) -> Result<(), String> {
        let nested = self.checker.procedure(id)?;
        if !owner(&nested.owner) {
            return Err(format!("p{} is not {what}", id.0));
        }
        if nested.register_types.len() != self.procedure.register_types.len()
            || nested.num_qubit_slots != self.procedure.num_qubit_slots
        {
            return Err(format!(
                "{what} p{} does not share the frame it runs on",
                id.0
            ));
        }
        Ok(())
    }

    fn op(&self, op: &Op) -> Result<(), String> {
        match op {
            Op::Binary { dest, lhs, rhs, .. } => {
                self.reg(*dest)?;
                self.operand(lhs)?;
                self.operand(rhs)
            }
            Op::Unary { dest, src, .. } | Op::Cast { dest, src, .. } | Op::Move { dest, src } => {
                self.reg(*dest)?;
                self.operand(src)
            }
            Op::LoadElement {
                dest,
                base,
                indices,
            } => {
                self.reg(*dest)?;
                self.operand(base)?;
                self.subscripts(indices)
            }
            Op::StoreElement {
                target,
                indices,
                value,
            } => {
                self.reg(*target)?;
                self.subscripts(indices)?;
                self.operand(value)
            }
            Op::NewArray { dest, items } => {
                self.reg(*dest)?;
                self.operands(items)
            }
            Op::Borrow {
                dest,
                base,
                indices,
            } => {
                self.reg(*dest)?;
                self.reg(*base)?;
                self.subscripts(indices)
            }
            Op::Release { reg } => self.reg(*reg),
            Op::Call { dest, callee, args } => {
                if let Some(dest) = dest {
                    self.reg(*dest)?;
                }
                if let Callee::Symbol(symbol) = callee {
                    self.checker.symbol(*symbol)?;
                }
                self.operands(args)
            }
            Op::GateCall {
                gate,
                modifiers,
                args,
                qubits,
            } => {
                self.checker.symbol(*gate)?;
                let mut controls = 0usize;
                for modifier in modifiers {
                    match modifier {
                        Modifier::Pow(exponent) => self.operand(exponent)?,
                        Modifier::Ctrl(n) | Modifier::NegCtrl(n) => controls += *n as usize,
                        Modifier::Inv => {}
                    }
                }
                if controls > qubits.len() {
                    return Err(format!(
                        "a gate call with {controls} controls has {} qubit operands",
                        qubits.len()
                    ));
                }
                self.operands(args)?;
                self.operands(qubits)
            }
            Op::Measure { dest, qubit } => {
                if let Some(dest) = dest {
                    self.reg(*dest)?;
                }
                self.operand(qubit)
            }
            Op::Reset { qubit } => self.operand(qubit),
            Op::Barrier { qubits } | Op::Nop { qubits } => self.operands(qubits),
            Op::Delay { duration, qubits } => {
                self.operand(duration)?;
                self.operands(qubits)
            }
            Op::CalOpaque { content } | Op::Pragma { content } => self.checker.string(*content),
            Op::CalOpenPulse { body } => self.nested(*body, "a cal block body", |o| {
                matches!(o, ProcOwner::InlineCal)
            }),
            Op::DurationOf { dest, body } => {
                self.reg(*dest)?;
                self.nested(*body, "a durationof scope", |o| {
                    matches!(o, ProcOwner::DurationOf)
                })
            }
            Op::AliasBind { slot, segments } => {
                self.slot(*slot)?;
                segments.iter().try_for_each(|segment| match segment {
                    AliasSegment::Operand(op) => self.operand(op),
                    AliasSegment::Slice {
                        source,
                        start,
                        step,
                        end,
                    } => {
                        self.operand(source)?;
                        self.optional(start)?;
                        self.optional(step)?;
                        self.optional(end)
                    }
                })
            }
            Op::CalLoad { dest, symbol } => {
                self.reg(*dest)?;
                self.checker.symbol(*symbol)
            }
            Op::CalStore { symbol, src } => {
                self.checker.symbol(*symbol)?;
                self.operand(src)
            }
        }
    }
}

/// `Operand` and `Box<Operand>` alike, for the optional range bounds.
trait AsOperand {
    fn as_operand(&self) -> &Operand;
}

impl AsOperand for Operand {
    fn as_operand(&self) -> &Operand {
        self
    }
}

impl AsOperand for Box<Operand> {
    fn as_operand(&self) -> &Operand {
        self
    }
}

fn within(index: usize, len: usize, what: &str) -> Result<(), String> {
    if index < len {
        Ok(())
    } else {
        Err(format!("{what} {index} out of range (there are {len})"))
    }
}
