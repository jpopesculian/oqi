//! SIR → bytecode.
//!
//! Builds the control-flow graph of each body of a [`sir::Program`] and the
//! global qubit layout, then emits one [`bc::Procedure`] per graph: every
//! block's statements become [`bc::Instr`]s and its terminator a
//! [`bc::Terminator`]. Each classical variable lives in one register of its
//! procedure's frame; sub-expressions are evaluated into temporaries.
//!
//! The scope of a `durationof` and the body of an inline `cal` block become
//! procedures of their own, referenced by [`ProcId`], that run on the frame
//! of the procedure they appear in and so share its registers and qubit
//! slots.
//!
//! Qubit references are lowered through the [`QubitLayout`]: named
//! registers and resolved `let` aliases become global quantum memory
//! references ([`bc::Operand::Qubit`] or a [`QubitSource::Region`]); gate
//! and subroutine qubit parameters and runtime aliases become frame-local
//! [`QubitSource::Slot`]s bound at call time / by `bc::Op::AliasBind`.

use std::collections::{HashMap, HashSet};

use oqi_classical::{RefAccess, Value, ValueTy};
use oqi_lex::Span;

use crate::cfg::{self, Cfg, Terminator};
use crate::classical::{PrimitiveTy, Scalar, iw};
use crate::error::{CompileError, ErrorKind, Result};
use crate::qubits::{self, QubitLayout, Resolved};
use crate::register::QuantumRegister;
use crate::scope::ScopeId;
use crate::sir::{
    self, CallTarget, Expr, ExprKind, GateModifier, IndexItem, IndexKind, IndexOp, LValue,
    MeasureExpr, MeasureExprKind, QubitOperand, RValue, Stmt, StmtKind,
};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::Type;

use super::types::{
    self as bc, BlockId, ConstId, ProcId, ProcOwner, QubitRegion, QubitRegionId, QubitSource,
    QubitTable, Reg, StringId,
};

/// Lower a SIR program to bytecode.
pub fn emit(program: &sir::Program) -> Result<bc::Program> {
    // Building the graphs adds the hidden variables of `for` loops.
    let mut symbols = program.symbols.clone();
    let cfgs = cfg::build_program(program, &mut symbols)?;
    let layout = qubits::build_layout(program);
    let mut ctx = EmitCtx::new(symbols, &program.calibrations, layout);

    let entry = ctx.emit_root(&cfgs.top_level)?;
    for body in cfgs.subroutines.iter().chain(&cfgs.gates) {
        ctx.emit_root(body)?;
    }
    let calibrations = program
        .calibrations
        .iter()
        .zip(&cfgs.calibrations)
        .map(|(cal, body)| {
            let proc = body.as_ref().map(|body| ctx.emit_root(body)).transpose()?;
            Ok(ctx.lower_calibration(cal, proc))
        })
        .collect::<Result<_>>()?;

    let program = bc::Program {
        symbols: ctx
            .symbols
            .iter()
            .map(|s| bc::Symbol {
                name: s.name.clone(),
                kind: s.kind,
                ty: s.ty.value_ty(),
            })
            .collect(),
        constants: ctx.constants,
        strings: ctx.strings,
        qubits: QubitTable {
            // Physical qubits map directly to global indices, so the
            // register must cover both declared memory and the highest
            // `$n` touched (the latter dominates in hardware-level
            // programs that declare no `qubit` registers at all).
            num_qubits: (ctx.layout.num_qubits() as u32)
                .max(ctx.max_hardware_qubit.map_or(0, |n| n + 1)),
            regions: ctx.qubit_regions,
        },
        procedures: ctx.procedures,
        entry,
        inputs: ctx.inputs,
        outputs: ctx.outputs,
        calibrations,
        calibration_grammar: program.calibration_grammar.clone(),
    };
    // What a decoded program is checked against holds for every emitted one.
    #[cfg(debug_assertions)]
    if let Err(why) = super::validate::check(&program) {
        panic!("emitted inconsistent bytecode: {why}");
    }
    Ok(program)
}

/// Emission state shared across procedures.
struct EmitCtx<'a> {
    symbols: SymbolTable,
    /// The program's calibration declarations; `ProcOwner::Calibration(i)`
    /// indexes into this for the defcal's signature (args/operands).
    calibrations: &'a [sir::CalibrationDecl],
    /// Global qubit layout; gains alias definitions during emission.
    layout: QubitLayout,
    qubit_regions: Vec<QubitRegion>,
    /// Resolved ranges of each entry in `qubit_regions`, for dedup.
    region_index: HashMap<Vec<(u32, u32)>, QubitRegionId>,
    procedures: Vec<bc::Procedure>,
    constants: Vec<Value>,
    /// Postcard encoding of each entry in `constants`, for dedup.
    const_index: HashMap<Vec<u8>, ConstId>,
    strings: Vec<String>,
    /// The program's input contract, resolved while emitting the
    /// top-level body.
    inputs: Vec<(SymbolId, Reg)>,
    /// Named program outputs, resolved while emitting the top-level body.
    outputs: Vec<(SymbolId, Reg)>,
    /// Highest physical qubit index (`$n`) referenced anywhere in the
    /// module, if any. Physical qubits map directly to global memory
    /// indices, so the simulator must be sized to cover the maximum one
    /// even when no `qubit` registers are declared.
    max_hardware_qubit: Option<u32>,
}

/// The base a runtime-alias operand indexes: a declared register /
/// static alias, or another runtime-bound alias.
enum AliasSrc {
    Region {
        reg: QuantumRegister,
        origin: SymbolId,
    },
    Alias(qubits::DynAlias),
    /// A qubit subroutine parameter, bound to a qubit list at call time and
    /// resolved at run time via its [`QubitSource::Slot`].
    Param {
        slot: u32,
        len: Option<usize>,
    },
}

impl AliasSrc {
    /// Statically known length of the base, if any.
    fn len(&self) -> Option<usize> {
        match self {
            AliasSrc::Region { reg, .. } => Some(reg.len()),
            AliasSrc::Alias(da) => da.len,
            AliasSrc::Param { len, .. } => *len,
        }
    }
}

impl<'a> EmitCtx<'a> {
    fn new(
        symbols: SymbolTable,
        calibrations: &'a [sir::CalibrationDecl],
        layout: QubitLayout,
    ) -> Self {
        Self {
            symbols,
            calibrations,
            layout,
            qubit_regions: Vec::new(),
            region_index: HashMap::new(),
            procedures: Vec::new(),
            constants: Vec::new(),
            const_index: HashMap::new(),
            strings: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            max_hardware_qubit: None,
        }
    }

    /// Claim the next procedure id, to be filled in once its body is
    /// emitted (bodies nested in it take the ids after it).
    fn reserve_proc(&mut self, owner: ProcOwner) -> ProcId {
        let id = ProcId(self.procedures.len() as u32);
        self.procedures.push(bc::Procedure {
            owner,
            register_types: Vec::new(),
            params: Vec::new(),
            num_qubit_slots: 0,
            blocks: Vec::new(),
            entry: BlockId(0),
        });
        id
    }

    /// Emit a procedure with a frame of its own.
    fn emit_root(&mut self, cfg: &Cfg) -> Result<ProcId> {
        let owner = cfg.owner.clone();
        let id = self.reserve_proc(owner.clone());
        // Qubit slots: parameters occupy `[0, n)`; runtime aliases bound in
        // the body take the slots after them.
        let qubit_params = match &owner {
            ProcOwner::Gate(s) | ProcOwner::Subroutine(s) => self.layout.qubit_param_count(*s),
            ProcOwner::Calibration(i) => self.calibrations[*i as usize].operands.len() as u32,
            _ => 0,
        };
        let top_level = matches!(owner, ProcOwner::TopLevel);

        let mut body = ProcEmitter {
            ctx: self,
            regs: HashMap::new(),
            bit_aliases: HashMap::new(),
            register_types: Vec::new(),
            alias_slot: qubit_params,
            instrs: Vec::new(),
            assigned: Vec::new(),
            nested: Vec::new(),
        };
        let params = body.param_registers(&owner)?;
        let inputs = match top_level {
            true => body.input_registers()?,
            false => Vec::new(),
        };
        let blocks = body.emit_blocks(cfg)?;
        let ProcEmitter {
            regs,
            register_types,
            alias_slot,
            mut assigned,
            nested,
            ..
        } = body;

        if top_level {
            // Per docs/directives.rst: if any `output` is declared, only
            // those variables are outputs; otherwise every variable is. Only
            // variables the program assigns are reported.
            let has_outputs = self.symbols.iter().any(|s| s.kind == SymbolKind::Output);
            let wanted = match has_outputs {
                true => SymbolKind::Output,
                false => SymbolKind::Variable,
            };
            assigned.sort_by_key(|symbol| symbol.0);
            assigned.dedup();
            assigned.retain(|symbol| self.symbols.get(*symbol).kind == wanted);
            // (A value of the cal scope has no register: it is not an output.)
            let outputs = assigned.iter().filter_map(|s| Some((*s, *regs.get(s)?)));
            self.outputs = outputs.collect();
            self.inputs = inputs;
        }
        // Nested bodies run on this procedure's frame.
        for nested in nested {
            let nested = &mut self.procedures[nested.0 as usize];
            nested.register_types = register_types.clone();
            nested.num_qubit_slots = alias_slot;
        }
        self.procedures[id.0 as usize] = bc::Procedure {
            owner,
            register_types,
            params,
            num_qubit_slots: alias_slot,
            blocks,
            entry: BlockId(cfg.entry.0 as u32),
        };
        Ok(id)
    }

    /// A value of the cal scope (docs/pulses.rst): declared at the top
    /// level of a `cal` block, and shared by `cal` blocks and `defcal`
    /// bodies through the VM's store of them.
    fn is_cal_value(&self, sym: SymbolId) -> bool {
        self.symbols.get(sym).scope == Some(ScopeId::CAL)
    }

    /// The module-level record of one `defcal`: its dispatch signature
    /// plus the body (the already-emitted proc for OpenPulse bodies, the
    /// interned source text for opaque ones).
    fn lower_calibration(
        &mut self,
        cal: &sir::CalibrationDecl,
        proc: Option<ProcId>,
    ) -> bc::Calibration {
        bc::Calibration {
            target: match &cal.target {
                sir::CalibrationTarget::Measure => bc::CalTarget::Measure,
                sir::CalibrationTarget::Reset => bc::CalTarget::Reset,
                sir::CalibrationTarget::Delay => bc::CalTarget::Delay,
                sir::CalibrationTarget::Named(sym) => bc::CalTarget::Gate(*sym),
            },
            args: cal
                .args
                .iter()
                .map(|a| match a {
                    sir::CalibrationArg::Param(sym) => bc::CalArg::Param(*sym),
                    sir::CalibrationArg::Expr(_) => bc::CalArg::Unsupported,
                })
                .collect(),
            operands: cal
                .operands
                .iter()
                .map(|o| match o {
                    sir::CalibrationOperand::Hardware(n) => bc::CalOperand::Hardware(*n as u32),
                    sir::CalibrationOperand::Ident(_) => bc::CalOperand::Any,
                })
                .collect(),
            has_return: cal.return_ty.is_some(),
            body: match &cal.body {
                sir::CalibrationBody::OpenPulse(_) => bc::CalBody::OpenPulse(
                    proc.expect("OpenPulse calibration body emitted as a proc"),
                ),
                sir::CalibrationBody::Opaque(text) => {
                    bc::CalBody::Opaque(self.intern_string(text.clone()))
                }
            },
        }
    }

    /// Build a hardware-qubit operand, recording its index so the
    /// global memory can be sized to include it.
    fn hw_qubit(&mut self, n: u32) -> bc::Operand {
        self.max_hardware_qubit = Some(self.max_hardware_qubit.map_or(n, |m| m.max(n)));
        bc::Operand::HardwareQubit(n)
    }

    fn must_global(&self, reg: &QuantumRegister, local: usize) -> u32 {
        self.layout
            .global_index(reg, local)
            .expect("local index validated against register length") as u32
    }

    /// Intern a region for `reg`, deduplicating on its global ranges.
    fn region_for(&mut self, reg: &QuantumRegister, origin: Option<SymbolId>) -> QubitRegionId {
        let ranges = self.layout.global_ranges(reg);
        if let Some(&id) = self.region_index.get(&ranges) {
            return id;
        }
        let id = QubitRegionId(self.qubit_regions.len() as u32);
        self.qubit_regions.push(QubitRegion {
            ranges: ranges.clone(),
            origin,
        });
        self.region_index.insert(ranges, id);
        id
    }

    /// Declared length of a qubit parameter (`qubit q` → 1, `qubit[n]` → n),
    /// used to resolve static slice bounds. `None` for non-qubit symbols.
    fn qubit_param_len(&self, symbol: SymbolId) -> Option<usize> {
        match self.symbols.get(symbol).ty {
            Type::Qubit => Some(1),
            Type::QubitReg(n) => Some(n),
            _ => None,
        }
    }

    fn intern_const(&mut self, v: Value) -> ConstId {
        // Dedup keyed on the postcard encoding: bit-exact, so values
        // that differ only in float payload (e.g. NaNs) stay distinct.
        let key = postcard::to_allocvec(&v).expect("constant Value should encode");
        if let Some(&id) = self.const_index.get(&key) {
            return id;
        }
        let id = ConstId(self.constants.len() as u32);
        self.constants.push(v);
        self.const_index.insert(key, id);
        id
    }

    /// The pooled constant for a non-negative position.
    fn position_const(&mut self, position: usize) -> bc::Operand {
        bc::Operand::Const(self.intern_const(Value::int(position as i128, iw(64))))
    }

    fn intern_string(&mut self, s: String) -> StringId {
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s);
        id
    }
}

/// Where a classical variable lives.
enum Place {
    /// In a register of the frame.
    Reg(Reg),
    /// Outside the frame's registers, to be fetched from and put back to:
    /// a value of the cal scope (in the VM's store of them), or the bits a
    /// `let` alias refers to (in the registers it names).
    Elsewhere,
}

/// What is assigned or returned: an expression or a measurement.
#[derive(Clone, Copy)]
enum Source<'e> {
    Expr(&'e Expr),
    Measure(&'e MeasureExpr),
}

impl<'e> From<&'e RValue> for Source<'e> {
    fn from(value: &'e RValue) -> Self {
        match value {
            RValue::Expr(e) => Source::Expr(e),
            RValue::Measure(m) => Source::Measure(m),
        }
    }
}

/// Emission state of one frame: a procedure and the bodies nested in it.
struct ProcEmitter<'c, 'a> {
    ctx: &'c mut EmitCtx<'a>,
    /// The register of each classical variable referenced so far.
    regs: HashMap<SymbolId, Reg>,
    /// Type of every register allocated so far, variables and temporaries.
    register_types: Vec<ValueTy>,
    /// The bits each `let` alias of bit registers refers to: the operands
    /// of its `++`, each a (possibly subscripted) register or alias, with
    /// the positions in their subscripts fixed when the `let` ran.
    bit_aliases: HashMap<SymbolId, Vec<Expr>>,
    /// Next free frame-local qubit slot for a runtime-bound alias
    /// (`bc::Op::AliasBind`).
    alias_slot: u32,
    /// Instructions of the block being emitted.
    instrs: Vec<bc::Instr>,
    /// Every variable assigned to, in whole or in part.
    assigned: Vec<SymbolId>,
    /// The procedures emitted for nested bodies.
    nested: Vec<ProcId>,
}

impl ProcEmitter<'_, '_> {
    fn push(&mut self, op: bc::Op, span: Span) {
        self.instrs.push(bc::Instr { op, span });
    }

    /// A fresh register of type `ty`.
    fn temp(&mut self, ty: ValueTy) -> Reg {
        let reg = Reg(self.register_types.len() as u32);
        self.register_types.push(ty);
        reg
    }

    /// The register holding the classical variable `symbol`.
    fn reg(&mut self, symbol: SymbolId, span: Span) -> Result<Reg> {
        if let Some(reg) = self.regs.get(&symbol) {
            return Ok(*reg);
        }
        let sym = self.ctx.symbols.get(symbol);
        let ty = reg_value_ty(&sym.ty).ok_or_else(|| {
            let message = format!("`{}` is not a classical value", sym.name);
            CompileError::at(ErrorKind::InvalidContext(message), span)
        })?;
        let reg = self.temp(ty);
        self.regs.insert(symbol, reg);
        Ok(reg)
    }

    /// The register type of the value of an expression of type `ty`.
    fn value_ty(&self, ty: &Type, span: Span) -> Result<ValueTy> {
        reg_value_ty(ty).ok_or_else(|| {
            let message = format!("a value is needed here, not `{ty}`");
            CompileError::at(ErrorKind::InvalidContext(message), span)
        })
    }

    /// Registers holding the classical parameters of a gate/subroutine/
    /// defcal, in declaration order. Empty for owners that take no
    /// classical parameters.
    fn param_registers(&mut self, owner: &ProcOwner) -> Result<Vec<Reg>> {
        let params: Vec<SymbolId> = match owner {
            ProcOwner::Gate(s) | ProcOwner::Subroutine(s) => {
                self.ctx.layout.classical_params(*s).to_vec()
            }
            // Defcal signatures aren't in the layout (several defcals can
            // share one gate symbol); read the params off the declaration.
            ProcOwner::Calibration(i) => self.ctx.calibrations[*i as usize]
                .args
                .iter()
                .filter_map(|a| match a {
                    sir::CalibrationArg::Param(sym) => Some(*sym),
                    sir::CalibrationArg::Expr(_) => None,
                })
                .collect(),
            _ => return Ok(Vec::new()),
        };
        params
            .into_iter()
            .map(|symbol| self.reg(symbol, self.ctx.symbols.get(symbol).span))
            .collect()
    }

    /// The program's input contract: a `(symbol, reg)` for every
    /// `input`-declared variable, allocated here so even an unread input
    /// has a register to seed. In symbol (declaration) order.
    fn input_registers(&mut self) -> Result<Vec<(SymbolId, Reg)>> {
        let symbols = self.ctx.symbols.iter();
        let inputs: Vec<(SymbolId, Span)> = symbols
            .filter(|s| s.kind == SymbolKind::Input && s.ty.value_ty().is_some())
            .map(|s| (s.id, s.span))
            .collect();
        inputs
            .into_iter()
            .map(|(symbol, span)| Ok((symbol, self.reg(symbol, span)?)))
            .collect()
    }

    fn emit_blocks(&mut self, cfg: &Cfg) -> Result<Vec<bc::Block>> {
        cfg.blocks
            .iter()
            .map(|block| {
                for stmt in &block.stmts {
                    self.emit_stmt(stmt)?;
                }
                let terminator = self.emit_terminator(&block.terminator)?;
                Ok(bc::Block {
                    id: block_id(block.id),
                    instrs: std::mem::take(&mut self.instrs),
                    terminator,
                    span: block.span,
                })
            })
            .collect()
    }

    /// Emit a nested body as a procedure that runs on this frame.
    fn emit_nested(&mut self, body: &[Stmt], owner: ProcOwner) -> Result<ProcId> {
        let cfg = cfg::build_body(body, owner.clone(), false, &mut self.ctx.symbols)?;
        let id = self.ctx.reserve_proc(owner.clone());
        let outer_instrs = std::mem::take(&mut self.instrs);
        let outer_assigned = self.assigned.len();
        let blocks = self.emit_blocks(&cfg);
        self.instrs = outer_instrs;
        // What a nested body assigns is not a result of this one: a cal
        // block's values do not leak out of it, and a `durationof` scope
        // only runs to be timed.
        self.assigned.truncate(outer_assigned);
        let proc = &mut self.ctx.procedures[id.0 as usize];
        proc.blocks = blocks?;
        proc.entry = block_id(cfg.entry);
        self.nested.push(id);
        Ok(id)
    }

    fn emit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        let span = stmt.span;
        match &stmt.kind {
            StmtKind::Alias(sir::Alias { symbol, value })
                if self.ctx.symbols.get(*symbol).ty.value_ty().is_some() =>
            {
                self.bind_bit_alias(*symbol, value)?;
            }
            StmtKind::Alias(sir::Alias { symbol, value }) => {
                // An operand referencing another runtime-bound alias or a
                // qubit parameter forces this alias to be runtime-bound too
                // (parameters are only bound to concrete qubits at call time).
                let layout = &self.ctx.layout;
                let uses_dynamic = value.iter().any(|e| {
                    qubits::peel_index_chain(e).is_some_and(|(s, _)| {
                        layout.dynamic_alias_of(s).is_some() || layout.param_slot(s).is_some()
                    })
                });
                // Static qubit aliases resolve into the layout and emit
                // nothing; a non-constant index falls back to a runtime
                // bind.
                let resolved = match uses_dynamic {
                    true => Some(Resolved::Dynamic),
                    false => layout.resolve_alias_value(value, &self.ctx.symbols)?,
                };
                match resolved {
                    Some(Resolved::Static(reg)) => self.ctx.layout.define_alias(*symbol, reg),
                    Some(Resolved::Dynamic) => self.lower_dynamic_alias(*symbol, value, span)?,
                    // Lowering types an alias of bits, so this one names
                    // qubits the layout does not know here.
                    None => {
                        let name = &self.ctx.symbols.get(*symbol).name;
                        let message = format!("cannot resolve the qubits `{name}` aliases");
                        return Err(CompileError::at(ErrorKind::InvalidContext(message), span));
                    }
                }
            }
            StmtKind::GateCall(sir::GateCall {
                gate,
                modifiers,
                args,
                qubits,
                // Compile-time timing only.
                duration: _,
            }) => {
                // A bare gate-call whose name resolves to a `def` (e.g.
                // `hadamard_layer ancilla;`) is a subroutine call, not a gate.
                if self.ctx.symbols.get(*gate).kind == SymbolKind::Subroutine {
                    if !modifiers.is_empty() {
                        let message = "gate modifiers cannot be applied to a subroutine call";
                        return Err(CompileError::at(
                            ErrorKind::InvalidContext(message.into()),
                            span,
                        ));
                    }
                    return self.emit_call(None, CallTarget::Symbol(*gate), args, qubits, span);
                }
                let modifiers = modifiers
                    .iter()
                    .map(|m| self.lower_gate_modifier(m))
                    .collect::<Result<_>>()?;
                let args = self.operands(args)?;
                let qubits = self.qubit_operands(qubits, span)?;
                let gate = *gate;
                self.push(
                    bc::Op::GateCall {
                        gate,
                        modifiers,
                        args,
                        qubits,
                    },
                    span,
                );
            }
            // `measure q -> lv;` assigns like `lv = measure q;`.
            StmtKind::Measure(sir::Measure { measure, target }) => match target {
                Some(target) => self.emit_assignment(target, Source::Measure(measure), span)?,
                None => self.emit_measure(measure, None, span)?,
            },
            StmtKind::Reset(q) => {
                let qubit = self.qubit_operand(q, span)?;
                self.push(bc::Op::Reset { qubit }, span);
            }
            StmtKind::Barrier(qs) => {
                let qubits = self.qubit_operands(qs, span)?;
                self.push(bc::Op::Barrier { qubits }, span);
            }
            StmtKind::Delay(sir::Delay { duration, operands }) => {
                let duration = self.operand(duration)?;
                let qubits = self.qubit_operands(operands, span)?;
                self.push(bc::Op::Delay { duration, qubits }, span);
            }
            StmtKind::Assignment(sir::Assignment { target, value }) => {
                self.emit_assignment(target, value.into(), span)?;
            }
            StmtKind::Pragma(s) => {
                let content = self.ctx.intern_string(s.clone());
                self.push(bc::Op::Pragma { content }, span);
            }
            StmtKind::Cal(sir::CalibrationBody::Opaque(s)) => {
                let content = self.ctx.intern_string(s.clone());
                self.push(bc::Op::CalOpaque { content }, span);
            }
            StmtKind::Cal(sir::CalibrationBody::OpenPulse(body)) => {
                let body = self.emit_nested(body, ProcOwner::InlineCal)?;
                self.push(bc::Op::CalOpenPulse { body }, span);
            }
            // Evaluated for its effects; the value is dropped.
            StmtKind::ExprStmt(e) => match &e.kind {
                ExprKind::Call(call) => {
                    self.emit_call(None, call.callee.clone(), &call.args, &[], e.span)?
                }
                _ => {
                    self.operand(e)?;
                }
            },
            StmtKind::Nop(qs) => {
                let qubits = self.qubit_operands(qs, span)?;
                self.push(bc::Op::Nop { qubits }, span);
            }
            StmtKind::Box(_)
            | StmtKind::If(_)
            | StmtKind::For(_)
            | StmtKind::While(_)
            | StmtKind::Switch(_)
            | StmtKind::Break
            | StmtKind::Continue
            | StmtKind::Return(_)
            | StmtKind::End => {
                unreachable!("the CFG builder lowers control flow to terminators")
            }
        }
        Ok(())
    }

    fn emit_assignment(&mut self, target: &LValue, value: Source<'_>, span: Span) -> Result<()> {
        let symbol = target.symbol();
        let stores = self.stores(symbol);
        self.assigned.extend(stores);
        let subscripts: Vec<&IndexOp> = match target {
            LValue::Var(_) => Vec::new(),
            LValue::Indexed { indices, .. } => indices.iter().collect(),
        };
        // A whole variable is evaluated into a register of its type: its
        // own, if it lives in one.
        if subscripts.is_empty() {
            let dest = match self.place(symbol, span)? {
                Place::Reg(dest) => return self.emit_to_dest(value, dest, span),
                Place::Elsewhere => {
                    let ty = self.ctx.symbols.get(symbol).ty.clone();
                    self.temp(self.value_ty(&ty, span)?)
                }
            };
            self.emit_to_dest(value, dest, span)?;
            return self.put(symbol, bc::Operand::Reg(dest), span);
        }
        let value = match value {
            Source::Expr(e) => self.operand(e)?,
            Source::Measure(m) => {
                let temp = self.temp(self.value_ty(&m.ty, m.span)?);
                self.emit_measure(m, Some(temp), m.span)?;
                bc::Operand::Reg(temp)
            }
        };
        self.assign(symbol, &subscripts, value, span)
    }

    /// Assign `value` to the part of the variable `symbol` that a chain of
    /// subscripts selects: all of it if there are none.
    fn assign(
        &mut self,
        symbol: SymbolId,
        subscripts: &[&IndexOp],
        value: bc::Operand,
        span: Span,
    ) -> Result<()> {
        let ty = self.ctx.symbols.get(symbol).ty.clone();
        match (self.place(symbol, span)?, subscripts) {
            (Place::Reg(dest), []) => self.push(bc::Op::Move { dest, src: value }, span),
            (Place::Reg(target), _) => self.store_indexed(target, &ty, subscripts, value, span)?,
            (Place::Elsewhere, []) => self.put(symbol, value, span)?,
            // Part of a value held elsewhere: fetch it, store into the
            // copy, and put the copy back.
            (Place::Elsewhere, _) => {
                let held = self.fetch(symbol, span)?;
                self.store_indexed(held, &ty, subscripts, value, span)?;
                self.put(symbol, bc::Operand::Reg(held), span)?;
            }
        }
        Ok(())
    }

    /// Where the classical variable `symbol` lives.
    fn place(&mut self, symbol: SymbolId, span: Span) -> Result<Place> {
        if self.bit_aliases.contains_key(&symbol) || self.ctx.is_cal_value(symbol) {
            return Ok(Place::Elsewhere);
        }
        let sym = self.ctx.symbols.get(symbol);
        // An alias of bits is bound in the body that declares it.
        if sym.kind == SymbolKind::Alias && sym.ty.value_ty().is_some() {
            let message = format!("the alias `{}` is not visible here", sym.name);
            return Err(CompileError::at(ErrorKind::InvalidContext(message), span));
        }
        Ok(Place::Reg(self.reg(symbol, span)?))
    }

    /// A temporary holding the current value of a variable that lives
    /// outside the frame's registers ([`Place::Elsewhere`]).
    fn fetch(&mut self, symbol: SymbolId, span: Span) -> Result<Reg> {
        let ty = self.ctx.symbols.get(symbol).ty.clone();
        let dest = self.temp(self.value_ty(&ty, span)?);
        match self.bit_aliases.get(&symbol).cloned() {
            None => self.push(bc::Op::CalLoad { dest, symbol }, span),
            // One part: the alias is that register, or that part of it.
            Some(parts) if parts.len() == 1 => self.emit_expr_to_dest(&parts[0], dest, span)?,
            Some(parts) => {
                let mut offset = 0;
                for part in &parts {
                    let value = self.operand(part)?;
                    let (bits, width) = self.alias_bits(part, offset);
                    let (target, indices) = (dest, vec![bits]);
                    let op = bc::Op::StoreElement {
                        target,
                        indices,
                        value,
                    };
                    self.push(op, span);
                    offset += width;
                }
            }
        }
        Ok(dest)
    }

    /// Set a variable that lives outside the frame's registers
    /// ([`Place::Elsewhere`]) to `src`.
    fn put(&mut self, symbol: SymbolId, src: bc::Operand, span: Span) -> Result<()> {
        let Some(parts) = self.bit_aliases.get(&symbol).cloned() else {
            self.push(bc::Op::CalStore { symbol, src }, span);
            return Ok(());
        };
        let mut offset = 0;
        for part in &parts {
            let (bits, width) = self.alias_bits(part, offset);
            let piece = match parts.len() {
                1 => src.clone(),
                _ => {
                    let dest = self.temp(self.value_ty(&part.ty, span)?);
                    let (base, indices) = (src.clone(), vec![bits]);
                    let op = bc::Op::LoadElement {
                        dest,
                        base,
                        indices,
                    };
                    self.push(op, span);
                    bc::Operand::Reg(dest)
                }
            };
            let (root, subscripts) =
                qubits::peel_index_chain(part).expect("an alias part is a register reference");
            self.assign(root, &subscripts, piece, span)?;
            offset += width;
        }
        Ok(())
    }

    /// Where the bits of `part` sit in the alias it is a part of, `offset`
    /// bits in: the subscript selecting them, and how many they are.
    fn alias_bits(&mut self, part: &Expr, offset: usize) -> (bc::Subscript, usize) {
        let width = match part.ty.scalar_ty() {
            Some(PrimitiveTy::BitReg(width)) => width as usize,
            // A single bit.
            _ => return (bc::Subscript::Item(self.ctx.position_const(offset)), 1),
        };
        let bits = bc::Subscript::Range {
            start: Some(self.ctx.position_const(offset)),
            step: None,
            end: Some(self.ctx.position_const(offset + width - 1)),
        };
        (bits, width)
    }

    /// The variables that hold the value of `symbol`: itself, or, through
    /// an alias, the registers it refers to.
    fn stores(&self, symbol: SymbolId) -> Vec<SymbolId> {
        let Some(parts) = self.bit_aliases.get(&symbol) else {
            return vec![symbol];
        };
        let roots = parts.iter().filter_map(qubits::peel_index_chain);
        roots.flat_map(|(root, _)| self.stores(root)).collect()
    }

    /// The bits of declared registers that the register reference `place`
    /// names, in order: `None` if a position in its subscripts is only
    /// known at run time.
    fn named_bits(&self, place: &Expr) -> Option<Vec<(SymbolId, usize)>> {
        let (root, subscripts) = qubits::peel_index_chain(place)?;
        let mut bits: Vec<(SymbolId, usize)> = match self.bit_aliases.get(&root) {
            Some(parts) => {
                let parts = parts.iter().map(|part| self.named_bits(part));
                parts.collect::<Option<Vec<_>>>()?.concat()
            }
            None => {
                let width = match self.ctx.symbols.get(root).ty.scalar_ty()? {
                    PrimitiveTy::BitReg(width) => width as usize,
                    _ => 1,
                };
                (0..width).map(|bit| (root, bit)).collect()
            }
        };
        for subscript in subscripts {
            let Ok(Resolved::Static(positions)) = qubits::resolve_index(subscript, bits.len())
            else {
                return None;
            };
            bits = positions
                .into_iter()
                .map(|position| bits[position])
                .collect();
        }
        Some(bits)
    }

    /// Bind a `let` alias of bits (docs/types.rst, "Register concatenation
    /// and slicing") to the bits its operands name now.
    fn bind_bit_alias(&mut self, symbol: SymbolId, parts: &[Expr]) -> Result<()> {
        let mut named = HashSet::new();
        let mut bound = Vec::with_capacity(parts.len());
        for part in parts {
            let (root, _) =
                qubits::peel_index_chain(part).expect("an alias part is a register reference");
            // Also rejects an alias from another body (see `place`).
            self.place(root, part.span)?;
            // "A register cannot be concatenated with any part of itself":
            // checked where the positions are known.
            let bits = self.named_bits(part).unwrap_or_default();
            if bits.iter().any(|bit| named.contains(bit)) {
                let message = "a register cannot be concatenated with any part of itself";
                let kind = ErrorKind::InvalidOperands(message.into());
                return Err(CompileError::at(kind, part.span));
            }
            named.extend(bits);
            let mut part = part.clone();
            self.fix_positions(&mut part)?;
            bound.push(part);
        }
        self.bit_aliases.insert(symbol, bound);
        Ok(())
    }

    /// Evaluate the positions in the subscripts of the register reference
    /// `place` into hidden variables, so that it keeps naming the same bits
    /// whatever becomes of the variables in its index sets.
    fn fix_positions(&mut self, place: &mut Expr) -> Result<()> {
        let mut children = place.children_mut().into_iter();
        let Some(base) = children.next() else {
            return Ok(());
        };
        self.fix_positions(base)?;
        for position in children {
            if qubits::literal_index(position).is_some() {
                continue;
            }
            let hidden = self
                .ctx
                .symbols
                .new_temp(position.ty.clone(), position.span, None);
            let dest = self.reg(hidden, position.span)?;
            self.emit_expr_to_dest(position, dest, position.span)?;
            position.kind = ExprKind::Var(hidden);
        }
        Ok(())
    }

    /// Store `value` to the part of `target` (a register of type `ty`) that
    /// a chain of subscripts selects.
    ///
    /// One store reaches as far as the subscripts index one dimension after
    /// another (see [`direct_subscripts`]). The subscripts after that index
    /// the selection (`a[1:2][0]`), which is copied out, stored into, and
    /// written back.
    fn store_indexed(
        &mut self,
        target: Reg,
        ty: &Type,
        subscripts: &[&IndexOp],
        value: bc::Operand,
        span: Span,
    ) -> Result<()> {
        let (direct, rest) = subscripts.split_at(direct_subscripts(subscripts));
        let indices = self.index_chain(direct.iter().copied())?;
        if rest.is_empty() {
            let op = bc::Op::StoreElement {
                target,
                indices,
                value,
            };
            self.push(op, span);
            return Ok(());
        }
        let selected_ty = selected_type(ty, direct)?;
        let selected = self.temp(self.value_ty(&selected_ty, span)?);
        let load = bc::Op::LoadElement {
            dest: selected,
            base: bc::Operand::Reg(target),
            indices: indices.clone(),
        };
        self.push(load, span);
        self.store_indexed(selected, &selected_ty, rest, value, span)?;
        let store = bc::Op::StoreElement {
            target,
            indices,
            value: bc::Operand::Reg(selected),
        };
        self.push(store, span);
        Ok(())
    }

    fn emit_to_dest(&mut self, value: Source<'_>, dest: Reg, span: Span) -> Result<()> {
        match value {
            Source::Expr(e) => self.emit_expr_to_dest(e, dest, span),
            Source::Measure(m) => self.emit_measure(m, Some(dest), span),
        }
    }

    /// Measure into `dest`, or — for a subroutine called on qubits — call
    /// it for its value.
    fn emit_measure(&mut self, m: &MeasureExpr, dest: Option<Reg>, span: Span) -> Result<()> {
        match &m.kind {
            MeasureExprKind::Measure { operand } => {
                let qubit = self.qubit_operand(operand, m.span)?;
                self.push(bc::Op::Measure { dest, qubit }, span);
                Ok(())
            }
            MeasureExprKind::QuantumCall {
                callee,
                args,
                qubits,
            } => {
                let sym = self.ctx.symbols.get(*callee);
                if sym.kind != SymbolKind::Subroutine || reg_value_ty(&sym.ty).is_none() {
                    let message = format!("`{}` does not return a value", sym.name);
                    return Err(CompileError::at(ErrorKind::InvalidContext(message), m.span));
                }
                self.emit_call(dest, CallTarget::Symbol(*callee), args, qubits, span)
            }
        }
    }

    /// Emit a call with `args` followed by the qubit operands `qubits`
    /// (as the bare `name(args) qubits;` syntax passes them).
    ///
    /// An array (or part of one) passed to a `mutable` parameter is lent
    /// to the callee by reference (docs/subroutines.rst) for the call.
    fn emit_call(
        &mut self,
        dest: Option<Reg>,
        callee: CallTarget,
        args: &[Expr],
        qubits: &[QubitOperand],
        span: Span,
    ) -> Result<()> {
        let params: Vec<SymbolId> = match &callee {
            CallTarget::Symbol(callee) => self.ctx.layout.classical_params(*callee).to_vec(),
            CallTarget::Intrinsic(_) => Vec::new(),
        };
        let mut params = params.into_iter();
        let mut lent = Vec::new();
        // The arrays lent from outside the frame's registers, as fetched.
        let mut fetched = Vec::new();
        let mut operands = Vec::with_capacity(args.len() + qubits.len());
        for arg in args {
            // Qubit arguments bind to slots, not to classical parameters.
            if matches!(arg.kind, ExprKind::HardwareQubit(_)) || self.peel_qubit_ref(arg).is_some()
            {
                operands.push(self.operand(arg)?);
                continue;
            }
            let lend_as =
                params
                    .next()
                    .and_then(|param| match self.ctx.symbols.get(param).ty.value_ty() {
                        Some(ty @ ValueTy::ArrayRef(r)) if r.access() == RefAccess::Mutable => {
                            Some(ty)
                        }
                        _ => None,
                    });
            // Only an array the caller owns needs lending; a reference it
            // was passed itself already refers to the original.
            let owned = qubits::peel_index_chain(arg).filter(|(base, _)| {
                matches!(
                    self.ctx.symbols.get(*base).ty.value_ty(),
                    Some(ValueTy::Array(_))
                )
            });
            match (lend_as, owned) {
                (Some(ty), Some((base, subscripts))) => {
                    let lender = match self.place(base, arg.span)? {
                        Place::Reg(reg) => reg,
                        Place::Elsewhere => {
                            let held = self.fetch(base, arg.span)?;
                            fetched.push((base, held));
                            held
                        }
                    };
                    let mut lent_ty = self.ctx.symbols.get(base).ty.clone();
                    let mut source = lender;
                    let mut rest = subscripts.as_slice();
                    // A subscript that follows a range indexes the
                    // selection (see `direct_subscripts`): borrow from the
                    // reference to it.
                    loop {
                        let (direct, after) = rest.split_at(direct_subscripts(rest));
                        let indices = self.index_chain(direct.iter().copied())?;
                        lent_ty = selected_type(&lent_ty, direct)?;
                        let dest = match after.is_empty() {
                            true => self.temp(ty),
                            false => self.temp(self.value_ty(&lent_ty, arg.span)?),
                        };
                        let base = source;
                        let op = bc::Op::Borrow {
                            dest,
                            base,
                            indices,
                        };
                        self.push(op, arg.span);
                        (source, rest) = (dest, after);
                        if rest.is_empty() {
                            break;
                        }
                    }
                    lent.push(lender);
                    operands.push(bc::Operand::Reg(source));
                }
                _ => operands.push(self.operand(arg)?),
            }
        }
        operands.extend(self.qubit_operands(qubits, span)?);
        let callee = match callee {
            CallTarget::Symbol(s) => bc::Callee::Symbol(s),
            CallTarget::Intrinsic(i) => bc::Callee::Intrinsic(i),
        };
        let args = operands;
        self.push(bc::Op::Call { dest, callee, args }, span);
        lent.dedup();
        for reg in lent {
            self.push(bc::Op::Release { reg }, span);
        }
        for (symbol, held) in fetched {
            self.put(symbol, bc::Operand::Reg(held), span)?;
        }
        Ok(())
    }

    fn emit_expr_to_dest(&mut self, e: &Expr, dest: Reg, span: Span) -> Result<()> {
        match &e.kind {
            ExprKind::Binary(sir::Binary { op, left, right }) => {
                let lhs = self.operand(left)?;
                let rhs = self.operand(right)?;
                let op = *op;
                self.push(bc::Op::Binary { op, dest, lhs, rhs }, span);
            }
            ExprKind::Unary(sir::Unary { op, operand }) => {
                let src = self.operand(operand)?;
                let op = *op;
                self.push(bc::Op::Unary { op, dest, src }, span);
            }
            ExprKind::Cast(sir::Cast { target_ty, operand }) => {
                let src = self.operand(operand)?;
                let target_ty = self.value_ty(target_ty, span)?;
                self.push(
                    bc::Op::Cast {
                        dest,
                        target_ty,
                        src,
                    },
                    span,
                );
            }
            ExprKind::Index(sir::Index { base, index }) => {
                let base = self.operand(base)?;
                let indices = self.index_chain([index].into_iter())?;
                self.push(
                    bc::Op::LoadElement {
                        dest,
                        base,
                        indices,
                    },
                    span,
                );
            }
            ExprKind::Call(sir::Call { callee, args }) => {
                self.emit_call(Some(dest), callee.clone(), args, &[], span)?;
            }
            ExprKind::ArrayLiteral(_) => {
                // Flatten nested literals (multi-dimensional arrays) into a
                // single flat list of scalar leaves; `NewArray` builds the
                // array from the flat values against the destination's full
                // shape (see `oqi_classical::Array::new`).
                let mut items = Vec::new();
                self.collect_array_literal_leaves(e, &mut items)?;
                self.push(bc::Op::NewArray { dest, items }, span);
            }
            ExprKind::DurationOf(scope) => {
                let body = self.emit_nested(scope, ProcOwner::DurationOf)?;
                self.push(bc::Op::DurationOf { dest, body }, span);
            }
            ExprKind::Literal(_) | ExprKind::Var(_) | ExprKind::HardwareQubit(_) => {
                let src = self.operand(e)?;
                self.push(bc::Op::Move { dest, src }, span);
            }
        }
        Ok(())
    }

    /// Lower an expression into a single operand, emitting instructions
    /// into temporaries for any non-trivial sub-expression. Qubit
    /// references (named registers, aliases, parameters — bare or
    /// indexed) are resolved through the layout so that call arguments
    /// carry global qubit references.
    fn operand(&mut self, e: &Expr) -> Result<bc::Operand> {
        if let Some((sym, ops)) = self.peel_qubit_ref(e) {
            return self.resolve_qubit_ref(sym, &ops, e.span);
        }
        Ok(match &e.kind {
            ExprKind::Literal(p) => {
                let ty = match self.value_ty(&e.ty, e.span)? {
                    ValueTy::Scalar(ty) => ty,
                    other => {
                        let message = format!("a literal cannot be of type `{other}`");
                        return Err(CompileError::at(ErrorKind::Unsupported(message), e.span));
                    }
                };
                let value = Value::Scalar(Scalar::new_unchecked(p.clone(), ty));
                bc::Operand::Const(self.ctx.intern_const(value))
            }
            // Compile-time constants (`pi`, `tau`, `euler`, user `const`s)
            // carry their value in the symbol table and are pooled.
            ExprKind::Var(symbol) => match self.ctx.symbols.get(*symbol).const_value.clone() {
                Some(value) => bc::Operand::Const(self.ctx.intern_const(value)),
                None => match self.place(*symbol, e.span)? {
                    Place::Reg(reg) => bc::Operand::Reg(reg),
                    Place::Elsewhere => bc::Operand::Reg(self.fetch(*symbol, e.span)?),
                },
            },
            ExprKind::HardwareQubit(n) => self.ctx.hw_qubit(*n as u32),
            _ => {
                let temp = self.temp(self.value_ty(&e.ty, e.span)?);
                self.emit_expr_to_dest(e, temp, e.span)?;
                bc::Operand::Reg(temp)
            }
        })
    }

    fn operands(&mut self, exprs: &[Expr]) -> Result<Vec<bc::Operand>> {
        exprs.iter().map(|e| self.operand(e)).collect()
    }

    fn optional_operand(&mut self, e: Option<&Expr>) -> Result<Option<bc::Operand>> {
        e.map(|e| self.operand(e)).transpose()
    }

    /// Recursively collect the scalar-leaf operands of a (possibly nested)
    /// array literal in row-major order. Nested `ArrayLiteral`s are descended
    /// into rather than spilled to temporaries, so a multi-dimensional literal
    /// becomes one flat operand list matching its element count.
    fn collect_array_literal_leaves(
        &mut self,
        e: &Expr,
        items: &mut Vec<bc::Operand>,
    ) -> Result<()> {
        match &e.kind {
            ExprKind::ArrayLiteral(arr) => {
                for item in &arr.items {
                    self.collect_array_literal_leaves(item, items)?;
                }
            }
            _ => items.push(self.operand(e)?),
        }
        Ok(())
    }

    /// The positions a chain of subscripts selects, one [`bc::Subscript`] per
    /// subscript position.
    fn index_chain<'e>(
        &mut self,
        subscripts: impl Iterator<Item = &'e IndexOp>,
    ) -> Result<Vec<bc::Subscript>> {
        let mut out = Vec::new();
        for subscript in subscripts {
            match &subscript.kind {
                IndexKind::Set(members) => out.push(bc::Subscript::Set(self.operands(members)?)),
                IndexKind::Items(items) => {
                    for item in items {
                        out.push(match item {
                            IndexItem::Single(e) => bc::Subscript::Item(self.operand(e)?),
                            IndexItem::Range(r) => bc::Subscript::Range {
                                start: self.optional_operand(r.start.as_deref())?,
                                step: self.optional_operand(r.step.as_deref())?,
                                end: self.optional_operand(r.end.as_deref())?,
                            },
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    fn qubit_operand(&mut self, q: &QubitOperand, span: Span) -> Result<bc::Operand> {
        match q {
            QubitOperand::Indexed { symbol, indices } => {
                let indices: Vec<&IndexOp> = indices.iter().collect();
                self.resolve_qubit_ref(*symbol, &indices, span)
            }
            QubitOperand::Hardware(n) => Ok(self.ctx.hw_qubit(*n as u32)),
        }
    }

    fn qubit_operands(&mut self, qs: &[QubitOperand], span: Span) -> Result<Vec<bc::Operand>> {
        qs.iter().map(|q| self.qubit_operand(q, span)).collect()
    }

    /// Resolve a (possibly indexed) named qubit reference to a global
    /// memory operand, or to a positional parameter reference inside
    /// gate/subroutine bodies.
    fn resolve_qubit_ref(
        &mut self,
        symbol: SymbolId,
        indices: &[&IndexOp],
        span: Span,
    ) -> Result<bc::Operand> {
        // Gate / subroutine qubit parameter: bound at call time.
        if let Some(slot) = self.ctx.layout.param_slot(symbol) {
            return match *indices {
                [] => Ok(bc::Operand::Whole(QubitSource::Slot(slot))),
                [io] => match qubits::single_index_expr(io) {
                    Some(e) => {
                        let index = Box::new(self.operand(e)?);
                        Ok(bc::Operand::Index {
                            source: QubitSource::Slot(slot),
                            index,
                        })
                    }
                    None => {
                        let len = self.ctx.qubit_param_len(symbol).ok_or_else(|| {
                            CompileError::at(
                                ErrorKind::Unsupported(
                                    "slice of a non-register qubit parameter".into(),
                                ),
                                io.span,
                            )
                        })?;
                        match qubits::resolve_index(io, len)? {
                            // Static slice / discrete set: resolve positions
                            // against the declared length at compile time.
                            Resolved::Static(positions) => Ok(bc::Operand::Select {
                                source: QubitSource::Slot(slot),
                                positions: positions.iter().map(|&p| p as u32).collect(),
                            }),
                            // Runtime slice bounds: materialize a transient
                            // alias bound to the parameter's qubits at run time.
                            Resolved::Dynamic => {
                                let src = AliasSrc::Param {
                                    slot,
                                    len: Some(len),
                                };
                                let alias = self.bind_transient_alias(&src, &[io], span)?;
                                Ok(bc::Operand::Whole(QubitSource::Slot(alias)))
                            }
                        }
                    }
                },
                _ => Err(multi_dimensional(span)),
            };
        }

        // Declared register or resolved alias: global memory.
        if let Some(reg) = self.ctx.layout.register_of(symbol) {
            let reg = reg.clone();
            return match *indices {
                [] => {
                    if reg.len() == 1 {
                        Ok(bc::Operand::Qubit(self.ctx.must_global(&reg, 0)))
                    } else {
                        Ok(bc::Operand::Whole(QubitSource::Region(
                            self.ctx.region_for(&reg, Some(symbol)),
                        )))
                    }
                }
                [io] => match qubits::resolve_index(io, reg.len())? {
                    Resolved::Static(idxs) => {
                        if let [only] = idxs.as_slice() {
                            Ok(bc::Operand::Qubit(self.ctx.must_global(&reg, *only)))
                        } else {
                            let sub = qubits::select(&reg, &idxs);
                            Ok(bc::Operand::Whole(QubitSource::Region(
                                self.ctx.region_for(&sub, Some(symbol)),
                            )))
                        }
                    }
                    // Runtime single index: the VM maps the logical
                    // index through the region's ranges.
                    Resolved::Dynamic => {
                        match qubits::single_index_expr(io) {
                            Some(e) => {
                                let region = self.ctx.region_for(&reg, Some(symbol));
                                let index = Box::new(self.operand(e)?);
                                Ok(bc::Operand::Index {
                                    source: QubitSource::Region(region),
                                    index,
                                })
                            }
                            // Runtime multi-element slice: materialize a
                            // transient alias and reference it by slot.
                            None => {
                                let src = AliasSrc::Region {
                                    reg: reg.clone(),
                                    origin: symbol,
                                };
                                let slot = self.bind_transient_alias(&src, &[io], span)?;
                                Ok(bc::Operand::Whole(QubitSource::Slot(slot)))
                            }
                        }
                    }
                },
                _ => Err(multi_dimensional(span)),
            };
        }

        // Runtime-bound alias: resolved through its bind slot by the VM.
        if let Some(da) = self.ctx.layout.dynamic_alias_of(symbol) {
            return match *indices {
                [] => Ok(bc::Operand::Whole(QubitSource::Slot(da.slot))),
                [io] => match qubits::single_index_expr(io) {
                    Some(e) => {
                        let index = Box::new(self.alias_index_operand(&da, e)?);
                        Ok(bc::Operand::Index {
                            source: QubitSource::Slot(da.slot),
                            index,
                        })
                    }
                    // Runtime multi-element slice of an alias: re-bind it
                    // through a transient alias slot.
                    None => {
                        let src = AliasSrc::Alias(da);
                        let slot = self.bind_transient_alias(&src, &[io], span)?;
                        Ok(bc::Operand::Whole(QubitSource::Slot(slot)))
                    }
                },
                _ => Err(multi_dimensional(span)),
            };
        }

        Err(CompileError::at(
            ErrorKind::Unsupported(format!(
                "cannot resolve qubit reference `{}`",
                self.ctx.symbols.get(symbol).name
            )),
            span,
        ))
    }

    /// Lower a runtime-bound `let` alias to a [`bc::Op::AliasBind`] and
    /// register its slot/length for later [`QubitSource::Slot`]
    /// references. Each `++` operand becomes one or more segments.
    fn lower_dynamic_alias(
        &mut self,
        alias_sym: SymbolId,
        value: &[Expr],
        span: Span,
    ) -> Result<()> {
        let slot = self.alias_slot;
        self.alias_slot += 1;

        let mut segments: Vec<bc::AliasSegment> = Vec::new();
        let mut total_len: Option<usize> = Some(0);
        for expr in value {
            let len = self.alias_operand_segments(expr, &mut segments)?;
            total_len = match (total_len, len) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
        }

        self.push(bc::Op::AliasBind { slot, segments }, span);
        self.ctx
            .layout
            .define_dynamic_alias(alias_sym, slot, total_len);
        Ok(())
    }

    /// Append the segment(s) for one `++` operand of a runtime alias,
    /// returning its length (`None` if it contains a runtime range).
    fn alias_operand_segments(
        &mut self,
        expr: &Expr,
        out: &mut Vec<bc::AliasSegment>,
    ) -> Result<Option<usize>> {
        if matches!(expr.kind, ExprKind::HardwareQubit(_)) {
            return Err(CompileError::at(
                ErrorKind::Unsupported("physical qubits cannot be aliased".into()),
                expr.span,
            ));
        }
        let Some((sym, ops)) = qubits::peel_index_chain(expr) else {
            return Err(CompileError::at(
                ErrorKind::Unsupported(
                    "cannot mix quantum and classical operands in an alias".into(),
                ),
                expr.span,
            ));
        };
        // The base this operand indexes: a qubit parameter (bound at call
        // time), a declared register / static alias, or another runtime-bound
        // alias.
        let src = if let Some(slot) = self.ctx.layout.param_slot(sym) {
            AliasSrc::Param {
                slot,
                len: self.ctx.qubit_param_len(sym),
            }
        } else if let Some(reg) = self.ctx.layout.register_of(sym) {
            AliasSrc::Region {
                reg: reg.clone(),
                origin: sym,
            }
        } else if let Some(da) = self.ctx.layout.dynamic_alias_of(sym) {
            AliasSrc::Alias(da)
        } else {
            return Err(CompileError::at(
                ErrorKind::Unsupported(format!(
                    "cannot resolve qubit reference `{}`",
                    self.ctx.symbols.get(sym).name
                )),
                expr.span,
            ));
        };

        self.alias_segments_for(&src, &ops, out, expr.span)
    }

    /// Build the alias segment(s) for one indexed qubit operand against
    /// `src`. Shared by [`Self::alias_operand_segments`] (the `let`-alias
    /// path) and [`Self::bind_transient_alias`] (inline runtime slices).
    fn alias_segments_for(
        &mut self,
        src: &AliasSrc,
        ops: &[&IndexOp],
        out: &mut Vec<bc::AliasSegment>,
        span: Span,
    ) -> Result<Option<usize>> {
        match *ops {
            [] => {
                out.push(bc::AliasSegment::Operand(self.alias_whole_operand(src)));
                Ok(src.len())
            }
            [io] => match &io.kind {
                IndexKind::Set(es) => {
                    if es.is_empty() {
                        return Err(CompileError::at(
                            ErrorKind::InvalidContext(
                                "a quantum register cannot be indexed by an empty index set".into(),
                            ),
                            io.span,
                        ));
                    }
                    for e in es {
                        let op = self.alias_element_operand(src, e)?;
                        out.push(bc::AliasSegment::Operand(op));
                    }
                    Ok(Some(es.len()))
                }
                IndexKind::Items(items) => match items.as_slice() {
                    [IndexItem::Single(e)] => {
                        let op = self.alias_element_operand(src, e)?;
                        out.push(bc::AliasSegment::Operand(op));
                        Ok(Some(1))
                    }
                    [IndexItem::Range(r)] => {
                        let source = self.alias_whole_operand(src);
                        let mut lower = |b: Option<&Expr>| -> Result<Option<Box<bc::Operand>>> {
                            match b {
                                Some(e) => Ok(Some(Box::new(self.operand(e)?))),
                                None => Ok(None),
                            }
                        };
                        let start = lower(r.start.as_deref())?;
                        let step = lower(r.step.as_deref())?;
                        let end = lower(r.end.as_deref())?;
                        out.push(bc::AliasSegment::Slice {
                            source,
                            start,
                            step,
                            end,
                        });
                        // Length depends on runtime bounds.
                        Ok(None)
                    }
                    _ => Err(multi_dimensional(io.span)),
                },
            },
            _ => Err(multi_dimensional(span)),
        }
    }

    /// Materialize a runtime multi-element slice (`q[{a, b}]`, `q[i:j]`)
    /// used directly as an operand: allocate an anonymous alias slot, emit
    /// its [`bc::Op::AliasBind`], and return the slot. The caller wraps it in
    /// a [`QubitSource::Slot`].
    fn bind_transient_alias(
        &mut self,
        src: &AliasSrc,
        ops: &[&IndexOp],
        span: Span,
    ) -> Result<u32> {
        let slot = self.alias_slot;
        self.alias_slot += 1;
        let mut segments = Vec::new();
        self.alias_segments_for(src, ops, &mut segments, span)?;
        self.push(bc::Op::AliasBind { slot, segments }, span);
        Ok(slot)
    }

    /// The operand resolving to the whole base list of `src`.
    fn alias_whole_operand(&mut self, src: &AliasSrc) -> bc::Operand {
        let source = match src {
            AliasSrc::Region { reg, origin } => {
                QubitSource::Region(self.ctx.region_for(reg, Some(*origin)))
            }
            AliasSrc::Alias(da) => QubitSource::Slot(da.slot),
            AliasSrc::Param { slot, .. } => QubitSource::Slot(*slot),
        };
        bc::Operand::Whole(source)
    }

    /// A single-qubit operand selecting element `e` of `src`.
    fn alias_element_operand(&mut self, src: &AliasSrc, e: &Expr) -> Result<bc::Operand> {
        match src {
            AliasSrc::Region { reg, origin } => match qubits::literal_index(e) {
                Some(k) => {
                    let local = qubits::normalize_index(k, reg.len())
                        .map_err(|err| err.with_span(e.span))?;
                    Ok(bc::Operand::Qubit(self.ctx.must_global(reg, local)))
                }
                None => {
                    let region = self.ctx.region_for(reg, Some(*origin));
                    let index = Box::new(self.operand(e)?);
                    Ok(bc::Operand::Index {
                        source: QubitSource::Region(region),
                        index,
                    })
                }
            },
            AliasSrc::Alias(da) => {
                let index = Box::new(self.alias_index_operand(da, e)?);
                Ok(bc::Operand::Index {
                    source: QubitSource::Slot(da.slot),
                    index,
                })
            }
            AliasSrc::Param { slot, len } => {
                let index = match (qubits::literal_index(e), len) {
                    (Some(k), Some(n)) => {
                        let local =
                            qubits::normalize_index(k, *n).map_err(|err| err.with_span(e.span))?;
                        self.ctx.position_const(local)
                    }
                    _ => self.operand(e)?,
                };
                Ok(bc::Operand::Index {
                    source: QubitSource::Slot(*slot),
                    index: Box::new(index),
                })
            }
        }
    }

    /// Lower a logical index into a runtime alias. A constant index is
    /// normalized against the alias length when known (so negative
    /// indices and bounds are resolved at compile time); otherwise it is
    /// left for the VM to resolve and bounds-check.
    fn alias_index_operand(&mut self, da: &qubits::DynAlias, e: &Expr) -> Result<bc::Operand> {
        match (qubits::literal_index(e), da.len) {
            (Some(k), Some(len)) => {
                let local = qubits::normalize_index(k, len).map_err(|err| err.with_span(e.span))?;
                Ok(self.ctx.position_const(local))
            }
            (Some(k), None) if k < 0 => Err(CompileError::at(
                ErrorKind::Unsupported("negative index into a runtime-length alias".into()),
                e.span,
            )),
            _ => self.operand(e),
        }
    }

    /// Whether `e` is a (possibly indexed) reference to a named qubit
    /// register, qubit alias, or qubit parameter; returns the base
    /// symbol and the index ops in application order.
    fn peel_qubit_ref<'e>(&self, e: &'e Expr) -> Option<(SymbolId, Vec<&'e IndexOp>)> {
        let (sym, ops) = qubits::peel_index_chain(e)?;
        let qubit_like = self.ctx.layout.param_slot(sym).is_some()
            || self.ctx.layout.register_of(sym).is_some()
            || matches!(
                self.ctx.symbols.get(sym).kind,
                SymbolKind::Qubit | SymbolKind::GateQubit
            );
        qubit_like.then_some((sym, ops))
    }

    fn lower_gate_modifier(&mut self, m: &GateModifier) -> Result<bc::Modifier> {
        Ok(match m {
            GateModifier::Inv => bc::Modifier::Inv,
            GateModifier::Pow(e) => bc::Modifier::Pow(self.operand(e)?),
            GateModifier::Ctrl(n) => bc::Modifier::Ctrl(*n as u32),
            GateModifier::NegCtrl(n) => bc::Modifier::NegCtrl(*n as u32),
        })
    }

    fn emit_terminator(&mut self, term: &Terminator) -> Result<bc::Terminator> {
        Ok(match term {
            Terminator::Goto(b) => bc::Terminator::Goto(block_id(*b)),
            Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => bc::Terminator::Branch {
                cond: self.operand(cond)?,
                then_bb: block_id(*then_bb),
                else_bb: block_id(*else_bb),
            },
            Terminator::Switch {
                target,
                cases,
                default,
            } => {
                let target = self.operand(target)?;
                let cases = cases
                    .iter()
                    .map(|(labels, bb)| Ok((self.operands(labels)?, block_id(*bb))))
                    .collect::<Result<_>>()?;
                bc::Terminator::Switch {
                    target,
                    cases,
                    default: block_id(*default),
                }
            }
            Terminator::Box {
                duration,
                body,
                after,
            } => bc::Terminator::Box {
                duration: self.operand(duration)?,
                body: block_id(*body),
                after: block_id(*after),
            },
            Terminator::Return(None) => bc::Terminator::Return(None),
            Terminator::Return(Some(RValue::Expr(e))) => {
                bc::Terminator::Return(Some(self.operand(e)?))
            }
            Terminator::Return(Some(RValue::Measure(m))) => {
                let temp = self.temp(self.value_ty(&m.ty, m.span)?);
                self.emit_measure(m, Some(temp), m.span)?;
                bc::Terminator::Return(Some(bc::Operand::Reg(temp)))
            }
            Terminator::End => bc::Terminator::End,
            Terminator::Unreachable => bc::Terminator::Unreachable,
        })
    }
}

/// How many leading subscripts of a chain index one dimension after another,
/// as a single subscript would: those up to and including the first that
/// selects a range or set. What follows indexes that selection, starting
/// over at its first dimension.
fn direct_subscripts(chain: &[&IndexOp]) -> usize {
    let selects = |subscript: &&IndexOp| match &subscript.kind {
        IndexKind::Set(_) => true,
        IndexKind::Items(items) => items.iter().any(|item| matches!(item, IndexItem::Range(_))),
    };
    chain
        .iter()
        .position(selects)
        .map_or(chain.len(), |i| i + 1)
}

/// The type of a value of type `ty` after the subscripts `chain`.
fn selected_type(ty: &Type, chain: &[&IndexOp]) -> Result<Type> {
    chain.iter().try_fold(ty.clone(), |ty, subscript| {
        crate::lower::index_result_type(&ty, subscript)
    })
}

fn block_id(id: cfg::BasicBlockId) -> BlockId {
    BlockId(id.0 as u32)
}

/// The register type backing a value of type `ty`, if it has one.
/// Classical types use their value type; openpulse values (ports, frames,
/// waveforms) are opaque handles minted by the VM's pulse handler, carried
/// in a `uint[64]` register.
fn reg_value_ty(ty: &Type) -> Option<ValueTy> {
    match ty {
        Type::Openpulse(_) => Some(ValueTy::Scalar(PrimitiveTy::Uint(iw(64)))),
        _ => ty.value_ty(),
    }
}

/// The error for more than one subscript on a qubit register.
fn multi_dimensional(span: Span) -> CompileError {
    let message = "multi-dimensional index on a quantum register";
    CompileError::at(ErrorKind::Unsupported(message.into()), span)
}
