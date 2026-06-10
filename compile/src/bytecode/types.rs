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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

// ── Module / procedure / block ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BcVersion {
    pub major: u16,
    pub minor: u16,
}

impl BcVersion {
    pub const CURRENT: BcVersion = BcVersion { major: 0, minor: 1 };
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BcModule {
    pub version: BcVersion,
    pub symbols: SymbolTable,
    /// Pool of classical values referenced by [`BcOperand::Const`].
    pub constants: Vec<Value>,
    /// Pool of strings (pragma payloads, opaque cal text).
    pub strings: Vec<String>,
    pub procedures: Vec<BcProcedure>,
    /// Index into `procedures` for the program's top-level body.
    pub entry: ProcId,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BcProcedure {
    pub owner: ProcOwner,
    /// Classical type of each register, indexed by [`Reg`].
    pub register_types: Vec<ValueTy>,
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
    /// Hardware qubit reference: `$0`, `$1`, etc.
    HardwareQubit(u32),
    /// Symbolic qubit register, optionally indexed by a classical operand.
    QubitReg {
        symbol: SymbolId,
        index: Option<Box<BcOperand>>,
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
    Add { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Sub { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Mul { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Div { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Mod { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Pow { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    BitAnd { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    BitOr { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    BitXor { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Shl { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Shr { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    LogAnd { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    LogOr { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Eq { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Neq { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Lt { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Gt { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Le { dest: Reg, lhs: BcOperand, rhs: BcOperand },
    Ge { dest: Reg, lhs: BcOperand, rhs: BcOperand },

    Neg { dest: Reg, src: BcOperand },
    BitNot { dest: Reg, src: BcOperand },
    LogNot { dest: Reg, src: BcOperand },
    Cast { dest: Reg, target_ty: ValueTy, src: BcOperand },

    // ── Moves & memory ─────────────────────────────────────────────
    /// `dest = src`. Covers literal-to-register copies and the parallel
    /// register-to-register moves inserted by phi elimination.
    Move { dest: Reg, src: BcOperand },
    LoadElement { dest: Reg, base: BcOperand, index: BcOperand },
    /// Whole-array kill+def: `new = base; new[index] = value`. Reads
    /// the entire array via `base`, produces a fresh `new` register
    /// after the partial update.
    StoreElement {
        new: Reg,
        base: BcOperand,
        index: BcOperand,
        value: BcOperand,
    },
    NewArray { dest: Reg, items: Vec<BcOperand> },

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
    Measure { dest: Option<Reg>, qubit: BcOperand },
    Reset { qubit: BcOperand },
    Barrier { qubits: Vec<BcOperand> },
    Delay { duration: BcOperand, qubits: Vec<BcOperand> },
    Nop { qubits: Vec<BcOperand> },

    // ── Structured timing constructs (lifted to procedures) ────────
    Box { duration: Option<BcOperand>, body: ProcId },
    CalOpaque { content: StringId },
    CalOpenPulse { body: ProcId },
    DurationOf { dest: Reg, body: ProcId },

    // ── Misc ───────────────────────────────────────────────────────
    Pragma { content: StringId },
    Alias { symbol: SymbolId, value: Vec<BcOperand> },
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
