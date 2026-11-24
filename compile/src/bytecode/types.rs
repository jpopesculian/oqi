//! Bytecode types. Flat, indexed, serializable.
//!
//! [`Program`] and [`Symbol`] are the public face of a compiled program.
//! Everything else here is the instruction set the VM executes: it is
//! public for `oqi-vm`, and changes whenever the compiler does.

use oqi_classical::{Value, ValueTy};
use oqi_lex::Span;
use serde::{Deserialize, Serialize};

use crate::sir::{BinOp, Intrinsic, UnOp};
use crate::symbol::{SymbolId, SymbolKind};

// ── IDs ──────────────────────────────────────────────────────────────

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProcId(pub u32);

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstId(pub u32);

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringId(pub u32);

#[doc(hidden)]
/// Dense per-procedure register identifier. `Reg(0)`..`Reg(n)` cover
/// every classical variable the procedure touches and every temporary it
/// computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Reg(pub u32);

#[doc(hidden)]
/// Index into [`QubitTable::regions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QubitRegionId(pub u32);

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

// ── Module / procedure / block ───────────────────────────────────────

#[doc(hidden)]
/// Global quantum memory: every named register is statically allocated
/// into one flat index space; operands reference it via regions.
#[derive(Clone, Serialize, Deserialize)]
pub struct QubitTable {
    /// Total size of global quantum memory.
    pub num_qubits: u32,
    pub regions: Vec<QubitRegion>,
}

#[doc(hidden)]
/// A (possibly non-contiguous) set of global qubit indices: a declared
/// register, a resolved `let` alias, or a static slice.
#[derive(Clone, Serialize, Deserialize)]
pub struct QubitRegion {
    /// Half-open global index ranges in logical order; adjacent ranges
    /// are merged.
    pub ranges: Vec<(u32, u32)>,
    /// Originating symbol, for disassembly only — regions are deduped
    /// on `ranges`, so distinct symbols may share a region.
    pub origin: Option<SymbolId>,
}

/// A name the program declares: enough to name it, classify it, and (for
/// classical values) type it.
#[derive(Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// The name as written in the source.
    pub name: String,
    /// What the name declares.
    pub kind: SymbolKind,
    /// Classical value type: a variable's declared type, or an `extern`'s
    /// return type. `None` for everything else (qubits, gates, void
    /// externs, …).
    pub ty: Option<ValueTy>,
}

/// A compiled program: what [`compile`](crate::compile) produces and the VM
/// runs. [`Program::to_bytes`] and [`Program::from_bytes`] write and read it
/// in the bytecode format; its `Display` is the disassembly.
///
/// Its fields describe the program to the VM; outside it, use the methods.
#[derive(Clone, Serialize, Deserialize)]
pub struct Program {
    /// The program's symbols, indexed by [`SymbolId`].
    pub symbols: Vec<Symbol>,
    #[doc(hidden)]
    /// Pool of classical values referenced by [`Operand::Const`].
    pub constants: Vec<Value>,
    #[doc(hidden)]
    /// Pool of strings (pragma payloads, opaque cal text).
    pub strings: Vec<String>,
    #[doc(hidden)]
    /// Global quantum memory layout referenced by qubit operands.
    pub qubits: QubitTable,
    #[doc(hidden)]
    pub procedures: Vec<Procedure>,
    #[doc(hidden)]
    /// Index into `procedures` for the program's top-level body.
    pub entry: ProcId,
    #[doc(hidden)]
    /// The program's input contract: each `(symbol, reg)` pairs an
    /// `input`-declared variable with the register in the entry
    /// procedure's register file that holds its value. The host seeds
    /// these before running. One entry per declared `input`, sorted by
    /// symbol id (declaration order).
    pub inputs: Vec<(SymbolId, Reg)>,
    #[doc(hidden)]
    /// Named program outputs: each `(symbol, reg)` pairs a source-level
    /// classical variable with the register in the entry procedure's
    /// register file holding its final value. Follows OpenQASM 3
    /// semantics — if any `output` is declared only those appear, else
    /// every named classical variable. Sorted by symbol id.
    pub outputs: Vec<(SymbolId, Reg)>,
    #[doc(hidden)]
    /// `defcal` declarations, in declaration order. When a pulse handler
    /// is installed, the VM dispatches matching gate/measure/reset
    /// operations on hardware qubits to these bodies.
    pub calibrations: Vec<Calibration>,
    #[doc(hidden)]
    /// The `defcalgrammar` string, quotes stripped (e.g. `openpulse`).
    pub calibration_grammar: Option<String>,
}

impl Program {
    /// How many qubits the program uses: the size of the backend it needs.
    pub fn num_qubits(&self) -> u32 {
        self.qubits.num_qubits
    }

    /// The symbol `id` names.
    ///
    /// # Panics
    ///
    /// If `id` is not one of this program's symbols.
    pub fn symbol(&self, id: SymbolId) -> &Symbol {
        &self.symbols[id.0]
    }

    /// The declared `input` called `name`, with its value type.
    pub fn input(&self, name: &str) -> Option<(SymbolId, ValueTy)> {
        self.inputs
            .iter()
            .map(|(id, _)| (*id, self.symbol(*id)))
            .find(|(_, sym)| sym.name == name)
            .and_then(|(id, sym)| Some((id, sym.ty?)))
    }

    /// Each declared classical `extern` with its return type (`None` = void).
    pub fn externs(&self) -> impl Iterator<Item = (&str, Option<ValueTy>)> {
        self.symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Extern)
            .map(|s| (s.name.as_str(), s.ty))
    }
}

#[doc(hidden)]
/// A body of code and the layout of the frame it runs on.
///
/// A gate, subroutine, `defcal` or the program itself runs on a frame of
/// its own. The body of an inline `cal` block or the scope of a
/// `durationof` is a procedure too, but one that runs on the frame of the
/// procedure it appears in: it sees that procedure's variables, and so
/// repeats its register and qubit-slot layout.
#[derive(Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub owner: ProcOwner,
    /// Classical type of each register, indexed by [`Reg`].
    pub register_types: Vec<ValueTy>,
    /// Registers holding this procedure's classical parameters, in
    /// declaration order. Empty for procedures that take no classical
    /// parameters (top-level, cal/durationof bodies). The calling
    /// convention: a caller binds its positional classical arguments to
    /// these registers before entering the body. Qubit parameters are
    /// addressed separately as frame-local qubit slots
    /// ([`QubitSource::Slot`]), not registers, so they never appear here.
    pub params: Vec<Reg>,
    /// Size of this procedure's frame-local qubit-slot vector: the number
    /// of qubit parameters plus the number of runtime aliases its body
    /// binds. The VM pre-sizes the slot vector to this. Parameters occupy
    /// `[0, n_qubit_params)`; aliases occupy the rest.
    pub num_qubit_slots: u32,
    pub blocks: Vec<Block>,
    pub entry: BlockId,
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum ProcOwner {
    TopLevel,
    Subroutine(SymbolId),
    Gate(SymbolId),
    Calibration(u32),
    /// Body of an inline `cal { ... }` block.
    InlineCal,
    /// Scope of a `durationof({...})` expression.
    DurationOf,
}

// ── Calibrations ─────────────────────────────────────────────────────

#[doc(hidden)]
/// A `defcal` declaration: the operation it calibrates, its signature,
/// and its lowered body. With a pulse handler installed, the VM matches
/// gate/measure/reset operations on hardware qubits against these
/// (most-specific operand match wins; ties go to the first declared).
#[derive(Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub target: CalTarget,
    /// Classical parameter list. A calibration with any
    /// [`CalArg::Unsupported`] entry is never dispatched.
    pub args: Vec<CalArg>,
    pub operands: Vec<CalOperand>,
    /// True for measure defcals, whose body returns the measured value.
    pub has_return: bool,
    pub body: CalBody,
}

#[doc(hidden)]
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalTarget {
    Measure,
    Reset,
    Delay,
    Gate(SymbolId),
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum CalArg {
    /// A declared parameter, bound positionally to the call's argument.
    Param(SymbolId),
    /// An expression specialization (e.g. `defcal rx(pi/2)`); such
    /// calibrations are not dispatched.
    Unsupported,
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum CalOperand {
    /// A specific hardware qubit: the call's operand must be `$n`.
    Hardware(u32),
    /// A named (generic) operand: matches any hardware qubit.
    Any,
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum CalBody {
    /// Lowered OpenPulse body.
    OpenPulse(ProcId),
    /// Unlowered text of a non-OpenPulse grammar.
    Opaque(StringId),
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    pub instrs: Vec<Instr>,
    pub terminator: Terminator,
    pub span: Span,
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Instr {
    pub op: Op,
    pub span: Span,
}

// ── Operands ─────────────────────────────────────────────────────────

#[doc(hidden)]
/// The base an indexable qubit operand selects from: a region of global
/// quantum memory, or a call/`AliasBind`-bound slot of the enclosing
/// frame.
#[derive(Clone, Serialize, Deserialize)]
pub enum QubitSource {
    /// A region of global quantum memory (declared register, resolved
    /// alias, or static slice) — see [`QubitTable::regions`].
    Region(QubitRegionId),
    /// A frame-local qubit slot. Slots `[0, n_qubit_params)` are the
    /// gate/subroutine's qubit parameters, bound positionally at call
    /// time; the rest are runtime aliases bound by [`Op::AliasBind`].
    Slot(u32),
}

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum Operand {
    /// Classical register.
    Reg(Reg),
    /// Pooled constant value.
    Const(ConstId),
    /// Hardware qubit reference: `$0`, `$1`, etc. — a separate
    /// physical namespace, distinct from global quantum memory.
    HardwareQubit(u32),
    /// Statically resolved single qubit: an index into global quantum
    /// memory. The hot path for ordinary single-qubit operands.
    Qubit(u32),
    /// The whole qubit list of a [`QubitSource`] (a region, a bound
    /// parameter, or a runtime alias).
    Whole(QubitSource),
    /// A statically-resolved sub-selection of `source`'s qubit list: the
    /// listed local positions, in order (e.g. `scratch[0:1]` → `[0, 1]`).
    /// Used for ranges and discrete index sets with compile-time bounds.
    Select {
        source: QubitSource,
        positions: Vec<u32>,
    },
    /// A runtime single-qubit index into `source`'s qubit list: the VM
    /// evaluates `index` and maps it through the list at execution time.
    Index {
        source: QubitSource,
        index: Box<Operand>,
    },
}

#[doc(hidden)]
/// One piece of a runtime-bound qubit alias value (see [`Op::AliasBind`]).
#[derive(Clone, Serialize, Deserialize)]
pub enum AliasSegment {
    /// Append the qubit(s) this operand resolves to via the VM's qubit
    /// resolution (single qubit, whole region, runtime index, …).
    Operand(Operand),
    /// Append a runtime slice `source[start : step : end]` of `source`'s
    /// qubit list, following OpenQASM range semantics (defaults: start 0,
    /// step 1, end len-1; negative indices count from the end).
    Slice {
        source: Operand,
        start: Option<Box<Operand>>,
        step: Option<Box<Operand>>,
        end: Option<Box<Operand>>,
    },
}

#[doc(hidden)]
/// How one subscript position selects from a bit register or an array
/// dimension. Negative positions count from the end.
#[derive(Clone, Serialize, Deserialize)]
pub enum Subscript {
    /// One position: `a[i]`.
    Item(Operand),
    /// The listed positions, in order: `a[{i, j}]`.
    Set(Vec<Operand>),
    /// Every `step`-th position from `start` to `end`, both included:
    /// `a[start:step:end]`. A bound left out is the first or last position
    /// in the direction of `step`, which defaults to 1.
    Range {
        start: Option<Operand>,
        step: Option<Operand>,
        end: Option<Operand>,
    },
}

// ── Gate modifiers ───────────────────────────────────────────────────

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum Modifier {
    Inv,
    Pow(Operand),
    Ctrl(u32),
    NegCtrl(u32),
}

// ── Call target ──────────────────────────────────────────────────────

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum Callee {
    Symbol(SymbolId),
    Intrinsic(Intrinsic),
}

// ── Opcodes ──────────────────────────────────────────────────────────

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum Op {
    // ── Classical arithmetic / bit / comparison (TAC) ──────────────
    Binary {
        op: BinOp,
        dest: Reg,
        lhs: Operand,
        rhs: Operand,
    },
    Unary {
        op: UnOp,
        dest: Reg,
        src: Operand,
    },
    Cast {
        dest: Reg,
        target_ty: ValueTy,
        src: Operand,
    },

    // ── Moves & memory ─────────────────────────────────────────────
    /// `dest = src`.
    Move {
        dest: Reg,
        src: Operand,
    },
    /// Indexed read `dest = base[indices]`: an element or sub-array for
    /// [`Subscript::Item`]s, a slice for a final set or range, and bits of an
    /// element once the array dimensions are used up. `indices` are applied
    /// in order, each to the leading dimension of what the previous ones
    /// left.
    LoadElement {
        dest: Reg,
        base: Operand,
        indices: Vec<Subscript>,
    },
    /// Indexed write `target[indices] = value`, in place (see
    /// [`Op::LoadElement`] for `indices`).
    StoreElement {
        target: Reg,
        indices: Vec<Subscript>,
        value: Operand,
    },
    NewArray {
        dest: Reg,
        items: Vec<Operand>,
    },
    /// Lend out the array in `base` (or the part of it `indices` select) to
    /// pass to a `mutable` array parameter: `dest` becomes a reference
    /// through which the callee writes to the array itself. `base` is
    /// shared for as long as the reference is out; [`Op::Release`] ends
    /// that.
    Borrow {
        dest: Reg,
        base: Reg,
        indices: Vec<Subscript>,
    },
    /// Take back sole ownership of the array in `reg` after a
    /// [`Op::Borrow`].
    Release {
        reg: Reg,
    },

    // ── Call ───────────────────────────────────────────────────────
    Call {
        dest: Option<Reg>,
        callee: Callee,
        args: Vec<Operand>,
    },

    // ── Quantum ops ────────────────────────────────────────────────
    GateCall {
        gate: SymbolId,
        modifiers: Vec<Modifier>,
        args: Vec<Operand>,
        qubits: Vec<Operand>,
    },
    Measure {
        dest: Option<Reg>,
        qubit: Operand,
    },
    Reset {
        qubit: Operand,
    },
    Barrier {
        qubits: Vec<Operand>,
    },
    Delay {
        duration: Operand,
        qubits: Vec<Operand>,
    },
    Nop {
        qubits: Vec<Operand>,
    },

    // ── Nested bodies ──────────────────────────────────────────────
    CalOpaque {
        content: StringId,
    },
    /// Run the body of an inline `cal` block on this frame.
    CalOpenPulse {
        body: ProcId,
    },
    /// `dest` = the duration of `body`, measured by a timing pass over it
    /// on a copy of this frame.
    DurationOf {
        dest: Reg,
        body: ProcId,
    },

    // ── Misc ───────────────────────────────────────────────────────
    Pragma {
        content: StringId,
    },
    /// Bind a body-local qubit alias `slot` to the concatenation of the
    /// resolved `segments`, for later [`QubitSource::Slot`] references.
    AliasBind {
        slot: u32,
        segments: Vec<AliasSegment>,
    },

    // ── The cal scope ──────────────────────────────────────────────
    /// Read a value of the cal scope (declared by a `cal` block, shared by
    /// every `cal` block and `defcal` body) from the VM's store of them.
    CalLoad {
        dest: Reg,
        symbol: SymbolId,
    },
    /// Write a value of the cal scope to the VM's store of them.
    CalStore {
        symbol: SymbolId,
        src: Operand,
    },
}

// ── Terminators ──────────────────────────────────────────────────────

#[doc(hidden)]
#[derive(Clone, Serialize, Deserialize)]
pub enum Terminator {
    Goto(BlockId),
    Branch {
        cond: Operand,
        then_bb: BlockId,
        else_bb: BlockId,
    },
    Switch {
        target: Operand,
        cases: Vec<(Vec<Operand>, BlockId)>,
        default: BlockId,
    },
    /// Entry to a `box[duration]`: run its contents from `body`, which
    /// fall through to `after` — or, in a timing pass, count `duration`
    /// and go straight to `after`.
    Box {
        duration: Operand,
        body: BlockId,
        after: BlockId,
    },
    Return(Option<Operand>),
    End,
    Unreachable,
}
