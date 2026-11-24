//! Static analyses over a [`bc::Program`].

use std::collections::{HashMap, HashSet};

use crate::bytecode::types as bc;

/// Whether a shot of this program can be sampled from a single final state —
/// i.e. no quantum operation depends on a measurement outcome, so the
/// pre-measurement state is the same on every shot.
///
/// Sound (conservative) condition: measurements occur only in the entry
/// procedure, and no gate-applying op (`GateCall`/`Reset`/`Call`) is reachable
/// *after* any `Measure` in the entry procedure's control-flow graph. When
/// this holds, the shot-sampling fast path may snapshot the state once and draw
/// every shot from it; measurement-dependent *classical* control flow is still
/// fine (the classical program is replayed per shot). Any doubt returns
/// `false`, falling back to per-shot re-execution.
pub fn is_sample_safe(module: &bc::Program) -> bool {
    let entry = module.entry.0 as usize;
    if entry >= module.procedures.len() {
        return false;
    }
    // Measurements must be confined to the entry procedure — a measure inside a
    // called subroutine would need interprocedural reasoning.
    for (i, proc) in module.procedures.iter().enumerate() {
        if i != entry && has_measure(proc) {
            return false;
        }
    }

    let proc = &module.procedures[entry];
    let pos: HashMap<u32, usize> = proc
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, b)| (b.id.0, idx))
        .collect();

    // Seed the "reachable after a measurement" worklist: for each block that
    // measures, its own suffix is checked in place, and all its successors are
    // post-measure.
    let mut post: HashSet<u32> = HashSet::new();
    let mut stack: Vec<u32> = Vec::new();
    for b in &proc.blocks {
        let mut measured = false;
        for ins in &b.instrs {
            match &ins.op {
                bc::Op::Measure { .. } => measured = true,
                op if measured && is_gate_applying(op) => return false,
                _ => {}
            }
        }
        if measured {
            for s in successors(&b.terminator) {
                if post.insert(s) {
                    stack.push(s);
                }
            }
        }
    }
    // Any gate-applying op in a post-measurement block is a feedback loop.
    while let Some(bid) = stack.pop() {
        let Some(&idx) = pos.get(&bid) else { continue };
        let b = &proc.blocks[idx];
        if b.instrs.iter().any(|ins| is_gate_applying(&ins.op)) {
            return false;
        }
        for s in successors(&b.terminator) {
            if post.insert(s) {
                stack.push(s);
            }
        }
    }
    true
}

/// Ops that apply a unitary / touch the quantum state. `Call` (a subroutine or
/// intrinsic) is treated conservatively — it might contain gates.
fn is_gate_applying(op: &bc::Op) -> bool {
    matches!(
        op,
        bc::Op::GateCall { .. } | bc::Op::Reset { .. } | bc::Op::Call { .. }
    )
}

fn has_measure(proc: &bc::Procedure) -> bool {
    proc.blocks.iter().any(|b| {
        b.instrs
            .iter()
            .any(|i| matches!(i.op, bc::Op::Measure { .. }))
    })
}

fn successors(term: &bc::Terminator) -> Vec<u32> {
    match term {
        bc::Terminator::Goto(b) => vec![b.0],
        bc::Terminator::Branch {
            then_bb, else_bb, ..
        } => vec![then_bb.0, else_bb.0],
        bc::Terminator::Box { body, after, .. } => vec![body.0, after.0],
        bc::Terminator::Switch { cases, default, .. } => {
            let mut v: Vec<u32> = cases.iter().map(|(_, b)| b.0).collect();
            v.push(default.0);
            v
        }
        bc::Terminator::Return(_) | bc::Terminator::End | bc::Terminator::Unreachable => Vec::new(),
    }
}
