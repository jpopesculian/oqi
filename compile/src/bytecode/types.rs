//! Bytecode types. Flat, indexed, serializable.

use oqi_classical::{Value, ValueTy};
use oqi_lex::Span;
use serde::{Deserialize, Serialize};

use crate::sir::Intrinsic;
use crate::symbol::{SymbolId, SymbolTable};

// ── IDs ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProcId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringId(pub u32);

/// Dense per-procedure register identifier. `Reg(0)`..`Reg(n)` cover
/// every SSA value the procedure produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Reg(pub u32);

/// Index into [`QubitTable::regions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QubitRegionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

// ── Module / procedure / block ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BcVersion {
    pub major: u16,
    pub minor: u16,
}

impl BcVersion {
    pub const CURRENT: BcVersion = BcVersion { major: 0, minor: 4 };
}

/// Global quantum memory: every named register is statically allocated
/// into one flat index space; operands reference it via regions.
#[derive(Clone, Serialize, Deserialize)]
pub struct QubitTable {
    /// Total size of global quantum memory.
    pub num_qubits: u32,
    pub regions: Vec<QubitRegion>,
}

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

#[derive(Clone, Serialize, Deserialize)]
pub struct BcModule {
    pub version: BcVersion,
    pub symbols: SymbolTable,
    /// Pool of classical values referenced by [`BcOperand::Const`].
    pub constants: Vec<Value>,
    /// Pool of strings (pragma payloads, opaque cal text).
    pub strings: Vec<String>,
    /// Global quantum memory layout referenced by qubit operands.
    pub qubits: QubitTable,
    pub procedures: Vec<BcProcedure>,
    /// Index into `procedures` for the program's top-level body.
    pub entry: ProcId,
    /// The program's input contract: each `(symbol, reg)` pairs an
    /// `input`-declared variable with the register in the entry
    /// procedure's register file that holds its value. The host seeds
    /// these before running. One entry per declared `input`, sorted by
    /// symbol id (declaration order).
    pub inputs: Vec<(SymbolId, Reg)>,
    /// Named program outputs: each `(symbol, reg)` pairs a source-level
    /// classical variable with the register in the entry procedure's
    /// register file holding its final value. Follows OpenQASM 3
    /// semantics — if any `output` is declared only those appear, else
    /// every named classical variable. Sorted by symbol id.
    pub outputs: Vec<(SymbolId, Reg)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BcProcedure {
    pub owner: ProcOwner,
    /// Classical type of each register, indexed by [`Reg`].
    pub register_types: Vec<ValueTy>,
    /// Registers holding this procedure's classical parameters, in
    /// declaration order. Empty for procedures that take no classical
    /// parameters (top-level, box/cal/durationof bodies). The calling
    /// convention: a caller binds its positional classical arguments to
    /// these registers before entering the body. Qubit parameters are
    /// addressed separately as positional [`BcOperand::QubitParam`]
    /// slots, not registers, so they never appear here.
    pub params: Vec<Reg>,
    pub blocks: Vec<BcBlock>,
    pub entry: BlockId,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ProcOwner {
    TopLevel,
    Subroutine(SymbolId),
    Gate(SymbolId),
    Calibration(u32),
    /// Lifted body of a `box { ... }` block stmt.
    Box,
    /// Lifted body of an inline `cal { ... }` block stmt.
    InlineCal,
    /// Lifted body of a `durationof({...})` expression.
    DurationOf,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BcBlock {
    pub id: BlockId,
    pub instrs: Vec<BcInstr>,
    pub terminator: BcTerminator,
    pub span: Span,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BcInstr {
    pub op: BcOp,
    pub span: Span,
}

// ── Operands ─────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub enum BcOperand {
    /// Classical SSA register.
    Reg(Reg),
    /// Pooled constant value.
    Const(ConstId),
    /// Hardware qubit reference: `$0`, `$1`, etc. — a separate
    /// physical namespace, distinct from global quantum memory.
    HardwareQubit(u32),
    /// Statically resolved single qubit: an index into global quantum
    /// memory.
    Qubit(u32),
    /// A whole region of global quantum memory (declared register,
    /// resolved alias, or static slice).
    QubitRegion(QubitRegionId),
    /// Runtime-indexed register: the VM maps the logical `index`
    /// through the region's ranges at execution time.
    QubitIndexed {
        region: QubitRegionId,
        index: Box<BcOperand>,
    },
    /// Qubit parameter of the enclosing gate/subroutine body, bound at
    /// call time. For gates the slot is the position in the declared
    /// qubit list (after any `ctrl @` controls); for subroutines it is
    /// the position in the full parameter list.
    QubitParam {
        slot: u32,
        index: Option<Box<BcOperand>>,
    },
    /// Body-local runtime alias, bound by [`BcOp::AliasBind`]. The VM
    /// maps the logical `index` through the bound qubit list at run time;
    /// `index: None` refers to the whole alias.
    QubitAlias {
        slot: u32,
        index: Option<Box<BcOperand>>,
    },
}

/// One piece of a runtime-bound qubit alias value (see [`BcOp::AliasBind`]).
#[derive(Clone, Serialize, Deserialize)]
pub enum BcAliasSegment {
    /// Append the qubit(s) this operand resolves to via the VM's qubit
    /// resolution (single qubit, whole region, runtime index, …).
    Operand(BcOperand),
    /// Append a runtime slice `source[start : step : end]` of `source`'s
    /// qubit list, following OpenQASM range semantics (defaults: start 0,
    /// step 1, end len-1; negative indices count from the end).
    Slice {
        source: BcOperand,
        start: Option<Box<BcOperand>>,
        step: Option<Box<BcOperand>>,
        end: Option<Box<BcOperand>>,
    },
}

// ── Gate modifiers ───────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub enum BcGateModifier {
    Inv,
    Pow(BcOperand),
    Ctrl(u32),
    NegCtrl(u32),
}

// ── Call target ──────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub enum BcCallTarget {
    Symbol(SymbolId),
    Intrinsic(Intrinsic),
}

// ── Opcodes ──────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub enum BcOp {
    // ── Classical arithmetic / bit / comparison (TAC) ──────────────
    Add {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Sub {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Mul {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Div {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Mod {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Pow {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    BitAnd {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    BitOr {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    BitXor {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Shl {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Shr {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    LogAnd {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    LogOr {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Eq {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Neq {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Lt {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Gt {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Le {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },
    Ge {
        dest: Reg,
        lhs: BcOperand,
        rhs: BcOperand,
    },

    Neg {
        dest: Reg,
        src: BcOperand,
    },
    BitNot {
        dest: Reg,
        src: BcOperand,
    },
    LogNot {
        dest: Reg,
        src: BcOperand,
    },
    Cast {
        dest: Reg,
        target_ty: ValueTy,
        src: BcOperand,
    },

    // ── Moves & memory ─────────────────────────────────────────────
    /// `dest = src`. Covers literal-to-register copies and the parallel
    /// register-to-register moves inserted by phi elimination.
    Move {
        dest: Reg,
        src: BcOperand,
    },
    LoadElement {
        dest: Reg,
        base: BcOperand,
        index: BcOperand,
    },
    /// Whole-array kill+def: `new = base; new[index] = value`. Reads
    /// the entire array via `base`, produces a fresh `new` register
    /// after the partial update.
    StoreElement {
        new: Reg,
        base: BcOperand,
        index: BcOperand,
        value: BcOperand,
    },
    /// Whole-register kill+def for a slice target: `new = base;
    /// new[indices] = value`. `value` is a multi-bit register whose bits
    /// are written into `base` at the given (already-resolved) positions,
    /// in order. Used for `reg[a:b] = ...` and discrete-set assignment.
    StoreSlice {
        new: Reg,
        base: BcOperand,
        indices: Vec<u32>,
        value: BcOperand,
    },
    NewArray {
        dest: Reg,
        items: Vec<BcOperand>,
    },

    // ── Call ───────────────────────────────────────────────────────
    Call {
        dest: Option<Reg>,
        callee: BcCallTarget,
        args: Vec<BcOperand>,
    },

    // ── Quantum ops ────────────────────────────────────────────────
    GateCall {
        gate: SymbolId,
        modifiers: Vec<BcGateModifier>,
        args: Vec<BcOperand>,
        qubits: Vec<BcOperand>,
    },
    Measure {
        dest: Option<Reg>,
        qubit: BcOperand,
    },
    Reset {
        qubit: BcOperand,
    },
    Barrier {
        qubits: Vec<BcOperand>,
    },
    Delay {
        duration: BcOperand,
        qubits: Vec<BcOperand>,
    },
    Nop {
        qubits: Vec<BcOperand>,
    },

    // ── Structured timing constructs (lifted to procedures) ────────
    Box {
        duration: Option<BcOperand>,
        body: ProcId,
    },
    CalOpaque {
        content: StringId,
    },
    CalOpenPulse {
        body: ProcId,
    },
    DurationOf {
        dest: Reg,
        body: ProcId,
    },

    // ── Misc ───────────────────────────────────────────────────────
    Pragma {
        content: StringId,
    },
    Alias {
        symbol: SymbolId,
        value: Vec<BcOperand>,
    },
    /// Bind a body-local qubit alias `slot` to the concatenation of the
    /// resolved `segments`, for later [`BcOperand::QubitAlias`] references.
    AliasBind {
        slot: u32,
        segments: Vec<BcAliasSegment>,
    },
}

// ── Terminators ──────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub enum BcTerminator {
    Goto(BlockId),
    Branch {
        cond: BcOperand,
        then_bb: BlockId,
        else_bb: BlockId,
    },
    Switch {
        target: BcOperand,
        cases: Vec<(BcSwitchLabels, BlockId)>,
        default: Option<BlockId>,
    },
    Return(Option<BcOperand>),
    End,
    Unreachable,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum BcSwitchLabels {
    Values(Vec<BcOperand>),
    Default,
}
