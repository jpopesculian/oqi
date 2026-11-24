//! Global quantum memory layout.
//!
//! OpenQASM qubits are all globally scoped with compile-time-constant
//! sizes (docs/types.rst), so every named quantum register can be
//! statically allocated into one flat global quantum memory. This
//! module builds that layout from a [`sir::Program`] and resolves
//! register references — including `let` aliases built from slices,
//! discrete index sets, and `++` concatenation — to global qubit
//! indices via [`QuantumMemory`].
//!
//! The bytecode emitter consumes the layout to rewrite symbolic qubit
//! operands into global references; the register types never cross
//! into the bytecode.
//!
//! Qubit parameters of gates and subroutines cannot be resolved
//! statically (they bind at call time); the layout maps them to
//! positional slots instead. Physical qubits (`$N`) live in a separate
//! hardware namespace and are not part of the layout.

use std::collections::{HashMap, HashSet};

use crate::classical::Primitive;
use crate::error::{CompileError, ErrorKind, Result};
use crate::register::{QuantumMemory, QuantumRegister};
use crate::sir::{self, Expr, ExprKind, IndexItem, IndexKind, IndexOp, RangeExpr};
use crate::symbol::{SymbolId, SymbolKind, SymbolTable};
use crate::types::Type;

/// A runtime-bound qubit alias: its body-local bind slot and, when
/// statically known, its length. `len` is `None` when the alias
/// contains a runtime-valued range (its size is only known at run time).
#[derive(Clone, Copy)]
pub struct DynAlias {
    pub slot: u32,
    pub len: Option<usize>,
}

/// Static layout of all named quantum registers in global memory.
pub struct QubitLayout {
    memory: QuantumMemory,
    /// Global `SymbolKind::Qubit` declarations, allocated in
    /// declaration (`SymbolId`) order.
    registers: HashMap<SymbolId, QuantumRegister>,
    /// Resolved qubit `let` aliases; populated during bytecode
    /// emission, in block order.
    aliases: HashMap<SymbolId, QuantumRegister>,
    /// Runtime-bound qubit `let` aliases (built from non-constant index
    /// expressions or derived from another dynamic alias); populated
    /// during bytecode emission. References resolve to a
    /// [`QubitSource::Slot`](super::bytecode::QubitSource) bound at run
    /// time by [`Op::AliasBind`](super::bytecode::Op).
    dynamic_aliases: HashMap<SymbolId, DynAlias>,
    /// Gate qubit params and qubit-typed subroutine params, mapped to
    /// their positional slot (gate: position in the declared qubit
    /// list; subroutine: position in the full parameter list).
    param_slots: HashMap<SymbolId, u32>,
    /// Classical parameters of each gate/subroutine, keyed by the
    /// gate/subroutine symbol, in declaration order. Used by the
    /// bytecode emitter to record the calling convention.
    classical_params: HashMap<SymbolId, Vec<SymbolId>>,
    /// Number of qubit parameters of each gate/subroutine. Qubit
    /// parameters occupy frame-local slots `[0, count)`; runtime aliases
    /// in the body are allocated slots after them.
    qubit_param_count: HashMap<SymbolId, u32>,
}

/// Allocate every global qubit declaration into a fresh global memory
/// and record the parameter slots of gate/subroutine qubit params.
pub fn build_layout(program: &sir::Program) -> QubitLayout {
    let mut memory = QuantumMemory::new();
    let mut registers = HashMap::new();
    for sym in program.symbols.iter() {
        if sym.kind != SymbolKind::Qubit || sym.scope.is_some() {
            continue;
        }
        let size = match sym.ty {
            Type::Qubit => 1,
            Type::QubitReg(n) => n,
            _ => continue,
        };
        registers.insert(sym.id, memory.alloc(size));
    }

    let mut param_slots = HashMap::new();
    let mut classical_params = HashMap::new();
    let mut qubit_param_count = HashMap::new();
    for gate in &program.gates {
        for (slot, q) in gate.qubits.iter().enumerate() {
            param_slots.insert(*q, slot as u32);
        }
        // Every declared gate parameter is classical (gates take only
        // qubits and classical angle-like params).
        classical_params.insert(gate.symbol, gate.params.clone());
        qubit_param_count.insert(gate.symbol, gate.qubits.len() as u32);
    }
    for sub in &program.subroutines {
        let mut classical = Vec::new();
        // Qubit params get dense slots `[0, n)` (classical params go to
        // registers, not slots), so the layout matches gates.
        let mut qslot = 0u32;
        for &param in &sub.params {
            if matches!(
                program.symbols.get(param).ty,
                Type::Qubit | Type::QubitReg(_)
            ) {
                param_slots.insert(param, qslot);
                qslot += 1;
            } else {
                classical.push(param);
            }
        }
        classical_params.insert(sub.symbol, classical);
        qubit_param_count.insert(sub.symbol, qslot);
    }
    // Defcal operands bind to slots `[0, n)` by position, like gate
    // qubits; only named operands are referenced through a slot ($n
    // references stay hardware-direct). Classical params and the qubit
    // count aren't recorded here — several defcals can share one gate
    // symbol, so the emitter reads those off the declaration instead.
    for cal in &program.calibrations {
        for (slot, op) in cal.operands.iter().enumerate() {
            if let sir::CalibrationOperand::Ident(sym) = op {
                param_slots.insert(*sym, slot as u32);
            }
        }
    }

    QubitLayout {
        memory,
        registers,
        aliases: HashMap::new(),
        dynamic_aliases: HashMap::new(),
        param_slots,
        classical_params,
        qubit_param_count,
    }
}

impl QubitLayout {
    /// Total size of global quantum memory.
    pub fn num_qubits(&self) -> usize {
        self.memory.size()
    }

    /// The register for a declared qubit symbol or a resolved alias.
    pub fn register_of(&self, sym: SymbolId) -> Option<&QuantumRegister> {
        self.registers.get(&sym).or_else(|| self.aliases.get(&sym))
    }

    /// Positional slot of a gate/subroutine qubit parameter.
    pub fn param_slot(&self, sym: SymbolId) -> Option<u32> {
        self.param_slots.get(&sym).copied()
    }

    /// Number of qubit parameters of a gate/subroutine symbol (0 for
    /// anything else). Runtime aliases in its body are slotted after
    /// these.
    pub fn qubit_param_count(&self, sym: SymbolId) -> u32 {
        self.qubit_param_count.get(&sym).copied().unwrap_or(0)
    }

    /// Classical parameters of a gate/subroutine, in declaration order.
    pub fn classical_params(&self, sym: SymbolId) -> &[SymbolId] {
        self.classical_params
            .get(&sym)
            .map_or(&[], |v| v.as_slice())
    }

    pub fn define_alias(&mut self, sym: SymbolId, reg: QuantumRegister) {
        self.aliases.insert(sym, reg);
    }

    /// Record a runtime-bound alias symbol and its bind slot.
    pub fn define_dynamic_alias(&mut self, sym: SymbolId, slot: u32, len: Option<usize>) {
        self.dynamic_aliases.insert(sym, DynAlias { slot, len });
    }

    /// The runtime-bound alias for `sym`, if it is one.
    pub fn dynamic_alias_of(&self, sym: SymbolId) -> Option<DynAlias> {
        self.dynamic_aliases.get(&sym).copied()
    }

    /// Global memory index of the qubit at logical position `local`.
    pub fn global_index(&self, reg: &QuantumRegister, local: usize) -> Option<usize> {
        reg.global_index_of(local)
    }

    /// Global index ranges of `reg`, with adjacent ranges merged.
    pub fn global_ranges(&self, reg: &QuantumRegister) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = Vec::new();
        for r in reg.ranges() {
            if r.is_empty() {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.1 == r.start as u32 => last.1 = r.end as u32,
                _ => out.push((r.start as u32, r.end as u32)),
            }
        }
        out
    }

    /// Resolve the value of a `let` alias of qubits (one expression per
    /// `++` operand) into a register. Returns `Ok(None)` when the first
    /// operand is not a quantum reference the layout knows.
    pub fn resolve_alias_value(
        &self,
        exprs: &[Expr],
        symbols: &SymbolTable,
    ) -> Result<Option<Resolved<QuantumRegister>>> {
        let mut combined = QuantumRegister::new();
        for (i, expr) in exprs.iter().enumerate() {
            match self.resolve_alias_operand(expr, symbols)? {
                Some(Resolved::Static(reg)) => combined = combined.concat(reg),
                Some(Resolved::Dynamic) => return Ok(Some(Resolved::Dynamic)),
                None if i == 0 => return Ok(None),
                None => {
                    return Err(CompileError::at(
                        ErrorKind::Unsupported(
                            "cannot mix quantum and classical operands in an alias".into(),
                        ),
                        expr.span,
                    ));
                }
            }
        }
        let Some(first) = exprs.first() else {
            return Ok(None);
        };

        // "A register cannot be concatenated with any part of itself"
        // (docs/types.rst) — reject overlapping global indices.
        let mut seen = HashSet::new();
        for g in combined.iter() {
            if !seen.insert(g) {
                return Err(CompileError::at(
                    ErrorKind::InvalidContext(
                        "a quantum register cannot be concatenated with any part of itself".into(),
                    ),
                    first.span,
                ));
            }
        }
        Ok(Some(Resolved::Static(combined)))
    }

    /// Resolve a single alias operand to a register, or `Ok(None)` if
    /// it is not a quantum reference.
    fn resolve_alias_operand(
        &self,
        expr: &Expr,
        symbols: &SymbolTable,
    ) -> Result<Option<Resolved<QuantumRegister>>> {
        if matches!(expr.kind, ExprKind::HardwareQubit(_)) {
            return Err(CompileError::at(
                ErrorKind::Unsupported("physical qubits cannot be aliased".into()),
                expr.span,
            ));
        }
        let Some((sym, ops)) = peel_index_chain(expr) else {
            return Ok(None);
        };
        if self.param_slots.contains_key(&sym) {
            return Err(CompileError::at(
                ErrorKind::Unsupported("aliases of qubit parameters are not supported".into()),
                expr.span,
            ));
        }
        let Some(reg) = self.register_of(sym) else {
            return if matches!(
                symbols.get(sym).kind,
                SymbolKind::Qubit | SymbolKind::GateQubit
            ) {
                Err(CompileError::at(
                    ErrorKind::Unsupported(format!(
                        "cannot resolve qubit reference `{}`",
                        symbols.get(sym).name
                    )),
                    expr.span,
                ))
            } else {
                Ok(None)
            };
        };
        match ops.as_slice() {
            [] => Ok(Some(Resolved::Static(reg.clone()))),
            [io] => {
                let idxs = resolve_index(io, reg.len())?;
                Ok(Some(idxs.map(|idxs| select(reg, &idxs))))
            }
            _ => Err(CompileError::at(
                ErrorKind::Unsupported("multi-dimensional index on a quantum register".into()),
                expr.span,
            )),
        }
    }
}

/// Peel an `Index` chain down to its base variable, collecting index
/// ops in application order.
pub fn peel_index_chain(e: &Expr) -> Option<(SymbolId, Vec<&IndexOp>)> {
    match &e.kind {
        ExprKind::Var(symbol) => Some((*symbol, Vec::new())),
        ExprKind::Index(ix) => {
            let (sym, mut ops) = peel_index_chain(&ix.base)?;
            ops.push(&ix.index);
            Some((sym, ops))
        }
        _ => None,
    }
}

/// `Some(i)` iff `e` is an integer literal (const folding during SIR
/// lowering guarantees constant indices arrive as literals).
pub fn literal_index(e: &Expr) -> Option<i128> {
    match &e.kind {
        ExprKind::Literal(Primitive::Int(i)) => Some(*i),
        ExprKind::Literal(Primitive::Uint(u)) => i128::try_from(*u).ok(),
        _ => None,
    }
}

/// The single index expression of `io`, if it is a plain `q[expr]`.
pub fn single_index_expr(io: &IndexOp) -> Option<&Expr> {
    match &io.kind {
        IndexKind::Items(items) => match items.as_slice() {
            [IndexItem::Single(e)] => Some(e),
            _ => None,
        },
        IndexKind::Set(_) => None,
    }
}

/// Something resolved at compile time, unless it depends on an index
/// that is only known at run time.
#[derive(Debug)]
pub enum Resolved<T> {
    Static(T),
    Dynamic,
}

impl<T> Resolved<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Resolved<U> {
        match self {
            Resolved::Static(value) => Resolved::Static(f(value)),
            Resolved::Dynamic => Resolved::Dynamic,
        }
    }
}

/// Evaluate an index op against a register of length `len`, returning
/// the selected local indices in order. Implements the index set
/// semantics from docs/types.rst: single (possibly negative) integers,
/// discrete sets `{a,b,c}`, and inclusive ranges `a:b` / `a:c:b` =
/// {a, a+c, ..., a+mc} with m maximized. Negative members select from
/// the end. The selection is static when every constituent expression
/// is an integer literal, and dynamic otherwise.
pub fn resolve_index(io: &IndexOp, len: usize) -> Result<Resolved<Vec<usize>>> {
    resolve_index_unlocated(io, len).map_err(|e| e.with_span(io.span))
}

fn resolve_index_unlocated(io: &IndexOp, len: usize) -> Result<Resolved<Vec<usize>>> {
    let normalize = |raw: Option<Vec<i128>>| -> Result<Option<Vec<usize>>> {
        let normalized = raw.map(|raw| raw.into_iter().map(|i| normalize_index(i, len)).collect());
        normalized.transpose()
    };
    let idxs = match &io.kind {
        IndexKind::Set(es) => normalize(es.iter().map(literal_index).collect())?,
        IndexKind::Items(items) => match items.as_slice() {
            [IndexItem::Single(e)] => normalize(literal_index(e).map(|i| vec![i]))?,
            [IndexItem::Range(r)] => expand_range(r, len)?,
            _ => {
                return Err(CompileError::at(
                    ErrorKind::Unsupported("multi-dimensional index on a quantum register".into()),
                    io.span,
                ));
            }
        },
    };
    let Some(idxs) = idxs else {
        return Ok(Resolved::Dynamic);
    };
    if idxs.is_empty() {
        return Err(CompileError::at(
            ErrorKind::InvalidContext(
                "a quantum register cannot be indexed by an empty index set".into(),
            ),
            io.span,
        ));
    }
    Ok(Resolved::Static(idxs))
}

/// Build a register selecting `indices` (in order) from `reg`, using a
/// single slice per run of consecutive indices.
pub fn select(reg: &QuantumRegister, indices: &[usize]) -> QuantumRegister {
    let mut out = QuantumRegister::new();
    let mut i = 0;
    while i < indices.len() {
        let start = indices[i];
        let mut end = start + 1;
        i += 1;
        while i < indices.len() && indices[i] == end {
            end += 1;
            i += 1;
        }
        out = out.concat(reg.slice(start..end));
    }
    out
}

/// Map a possibly-negative index into `0..len`.
pub fn normalize_index(idx: i128, len: usize) -> Result<usize> {
    let n = len as i128;
    let adjusted = if idx < 0 { idx + n } else { idx };
    if (0..n).contains(&adjusted) {
        Ok(adjusted as usize)
    } else {
        Err(CompileError::new(ErrorKind::QubitIndexOutOfRange {
            index: idx,
            len,
        }))
    }
}

/// Expand an inclusive range per the spec: the member set is generated
/// arithmetically from the raw (possibly negative) bounds, then each
/// member is normalized as an index. A direction mismatch yields the
/// empty set (rejected by the caller). `None` when a bound or the step
/// is not an integer literal.
pub fn expand_range(r: &RangeExpr, len: usize) -> Result<Option<Vec<usize>>> {
    // A component left out takes `default`.
    let component = |e: &Option<Box<Expr>>, default: i128| match e {
        Some(e) => literal_index(e),
        None => Some(default),
    };
    let Some(step) = component(&r.step, 1) else {
        return Ok(None);
    };
    if step == 0 {
        return Err(CompileError::new(ErrorKind::InvalidContext(
            "range step must be non-zero".into(),
        )));
    }
    let last = len as i128 - 1;
    let (first, last) = if step > 0 { (0, last) } else { (last, 0) };
    let (Some(start), Some(end)) = (component(&r.start, first), component(&r.end, last)) else {
        return Ok(None);
    };

    let mut out = Vec::new();
    let mut cur = start;
    while (step > 0 && cur <= end) || (step < 0 && cur >= end) {
        out.push(normalize_index(cur, len)?);
        cur += step;
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::compile_to_sir;
    use crate::resolve::DefaultIncludeResolver;
    use crate::types::CompileOptions;

    fn compile(src: &str) -> sir::Program {
        compile_to_sir(
            src,
            &DefaultIncludeResolver,
            &CompileOptions::default(),
            None,
        )
        .expect("compile")
    }

    /// Build the layout and resolve every top-level alias in program
    /// order, mirroring what the bytecode emitter does.
    fn try_layout(src: &str) -> Result<(sir::Program, QubitLayout)> {
        let p = compile_to_sir(
            src,
            &DefaultIncludeResolver,
            &CompileOptions::default(),
            None,
        )?;
        let mut layout = build_layout(&p);
        for stmt in &p.body {
            if let sir::StmtKind::Alias(a) = &stmt.kind
                && let Some(Resolved::Static(reg)) =
                    layout.resolve_alias_value(&a.value, &p.symbols)?
            {
                layout.define_alias(a.symbol, reg);
            }
        }
        Ok((p, layout))
    }

    fn layout(src: &str) -> (sir::Program, QubitLayout) {
        match try_layout(src) {
            Ok(out) => out,
            Err(e) => panic!("layout failed: {e}"),
        }
    }

    fn layout_err(src: &str) -> CompileError {
        match try_layout(src) {
            Err(e) => e,
            Ok(_) => panic!("expected a layout error for `{src}`"),
        }
    }

    fn ranges_of(p: &sir::Program, l: &QubitLayout, name: &str) -> Vec<(u32, u32)> {
        let sym = p.symbols.lookup(name).expect("symbol");
        let reg = l.register_of(sym).expect("register");
        l.global_ranges(reg)
    }

    #[test]
    fn registers_allocate_in_declaration_order() {
        let (p, l) = layout("qubit c; qubit[4] a; qubit[4] b;");
        assert_eq!(l.num_qubits(), 9);
        assert_eq!(ranges_of(&p, &l, "c"), vec![(0, 1)]);
        assert_eq!(ranges_of(&p, &l, "a"), vec![(1, 5)]);
        assert_eq!(ranges_of(&p, &l, "b"), vec![(5, 9)]);
        let b = p.symbols.lookup("b").unwrap();
        assert_eq!(l.global_index(l.register_of(b).unwrap(), 0), Some(5));
    }

    #[test]
    fn alias_slice_is_inclusive() {
        // docs/types.rst: `let myreg = q[1:4];` — myreg[0] is q[1].
        let (p, l) = layout("qubit[5] q; let myreg = q[1:4];");
        assert_eq!(ranges_of(&p, &l, "myreg"), vec![(1, 5)]);
    }

    #[test]
    fn alias_concat_merges_adjacent_ranges() {
        let (p, l) = layout("qubit[2] one; qubit[10] two; let concatenated = one ++ two;");
        assert_eq!(ranges_of(&p, &l, "concatenated"), vec![(0, 12)]);
    }

    #[test]
    fn alias_discrete_set() {
        // `two` starts at global 2.
        let (p, l) = layout("qubit[2] one; qubit[10] two; let sel = two[{0, 3, 5}];");
        assert_eq!(ranges_of(&p, &l, "sel"), vec![(2, 3), (5, 6), (7, 8)]);
    }

    #[test]
    fn alias_negative_index_and_range() {
        let (p, l) =
            layout("qubit[2] one; qubit[10] two; let last = two[-1]; let last_three = two[-4:-1];");
        assert_eq!(ranges_of(&p, &l, "last"), vec![(11, 12)]);
        // two[-4:-1] = local {6,7,8,9} = global 8..12.
        assert_eq!(ranges_of(&p, &l, "last_three"), vec![(8, 12)]);
    }

    #[test]
    fn alias_step_range() {
        let (p, l) = layout("qubit[2] one; qubit[10] two; let every_second = two[0:2:8];");
        // local {0,2,4,6,8} = global {2,4,6,8,10}.
        assert_eq!(
            ranges_of(&p, &l, "every_second"),
            vec![(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        );
    }

    #[test]
    fn alias_of_alias() {
        let (p, l) = layout(
            r#"
                qubit[2] one;
                qubit[10] two;
                let concatenated = one ++ two;
                let sliced = concatenated[0:6];
                let last_three = two[-4:-1];
                let both = sliced ++ last_three;
            "#,
        );
        assert_eq!(ranges_of(&p, &l, "sliced"), vec![(0, 7)]);
        assert_eq!(ranges_of(&p, &l, "both"), vec![(0, 7), (8, 12)]);
    }

    #[test]
    fn alias_out_of_range_index_is_rejected() {
        let err = layout_err("qubit[4] q; let bad = q[4];");
        assert!(matches!(
            err.kind,
            ErrorKind::QubitIndexOutOfRange { index: 4, len: 4 }
        ));
    }

    #[test]
    fn alias_self_concatenation_is_rejected() {
        let err = layout_err("qubit[4] q; let bad = q ++ q[0:1];");
        assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
    }

    #[test]
    fn alias_runtime_index_is_dynamic() {
        let p = compile("input uint[32] i; qubit[4] q; let late = q[{i}] ++ q[{0}];");
        let aliases = p.body.iter().filter_map(|stmt| match &stmt.kind {
            sir::StmtKind::Alias(alias) => Some(alias),
            _ => None,
        });
        let alias = { aliases }.next().expect("alias");
        let resolved = build_layout(&p).resolve_alias_value(&alias.value, &p.symbols);
        assert!(matches!(resolved, Ok(Some(Resolved::Dynamic))));
    }

    #[test]
    fn classical_alias_is_not_a_qubit_alias() {
        let (p, l) = layout("bit[4] c; let view = c[0:1];");
        let sym = p.symbols.lookup("view").expect("symbol");
        assert!(l.register_of(sym).is_none());
    }

    #[test]
    fn param_slots_for_gates_and_subroutines() {
        let (p, l) = layout(
            r#"
                qubit[2] b;
                gate g a, c { }
                def f(int n, qubit[2] d) { reset d[0]; }
            "#,
        );
        let gate = &p.gates[0];
        assert_eq!(l.param_slot(gate.qubits[0]), Some(0));
        assert_eq!(l.param_slot(gate.qubits[1]), Some(1));
        assert_eq!(l.qubit_param_count(gate.symbol), 2);
        let sub = &p.subroutines[0];
        // Qubit params get dense slots: classical `n` is skipped, so the
        // qubit param `d` is slot 0.
        assert_eq!(l.param_slot(sub.params[0]), None);
        assert_eq!(l.param_slot(sub.params[1]), Some(0));
        assert_eq!(l.qubit_param_count(sub.symbol), 1);
    }

    #[test]
    fn empty_index_set_is_rejected() {
        // Direction mismatch produces the empty set, which the spec
        // forbids as a register index.
        let err = layout_err("qubit[4] q; let bad = q[3:1];");
        assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
    }
}
