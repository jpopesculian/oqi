//! Step A: out-of-SSA via parallel register-to-register moves.
//!
//! For each block `B` with phis, group the phi sources by predecessor
//! and insert a sequentialized parallel-move set at the end of each
//! predecessor's stmt list (just before its terminator). Cycles in the
//! parallel set are broken by routing through a fresh temporary
//! [`SsaValue`].
//!
//! After this pass, every block's `phis` field is empty. The SSA
//! version numbers are preserved as register hints; the bytecode
//! emitter's register allocator dense-numbers them into [`Reg`](super::types::Reg)s.

use std::collections::HashMap;

use crate::cfg::BasicBlockId;
use crate::sir::{Annotation, RValue};
use crate::ssa::{
    Phi, SsaAssignment, SsaBlock, SsaCfg, SsaExpr, SsaExprKind, SsaLValue, SsaStmt, SsaStmtKind,
    SsaValue,
};

/// Replace phi nodes with parallel register-to-register moves at the
/// end of each predecessor block. Output `SsaBlock.phis` is empty.
pub fn deconstruct_phis(mut cfg: SsaCfg) -> SsaCfg {
    // Snapshot phis; clear the field so we can mutate predecessors.
    let n = cfg.blocks.len();
    let mut phis_per_block: Vec<Vec<Phi>> = (0..n).map(|_| Vec::new()).collect();
    for block in &mut cfg.blocks {
        phis_per_block[block.id.0] = std::mem::take(&mut block.phis);
    }

    // Group each block's phi sources by predecessor.
    for phis in &phis_per_block {
        if phis.is_empty() {
            continue;
        }
        let mut per_pred: HashMap<BasicBlockId, Vec<(SsaValue, SsaValue)>> = HashMap::new();
        for phi in phis {
            for &(pred, src) in &phi.sources {
                per_pred.entry(pred).or_default().push((phi.dest, src));
            }
        }
        for (pred_id, pairs) in per_pred {
            insert_parallel_moves(&mut cfg.blocks[pred_id.0], pairs);
        }
    }

    cfg
}

/// Sequentialize a set of parallel `dest = src` moves and append them
/// to `block.stmts` before its terminator. Cycles are broken via a
/// temporary SSA value (high `version` numbers reserved for this use).
fn insert_parallel_moves(block: &mut SsaBlock, pairs: Vec<(SsaValue, SsaValue)>) {
    for (dest, src) in sequentialize(pairs) {
        block.stmts.push(move_stmt(dest, src, block.span));
    }
}

/// Order a set of parallel `dest = src` moves so that executing them
/// sequentially is equivalent to executing them simultaneously.
fn sequentialize(mut pairs: Vec<(SsaValue, SsaValue)>) -> Vec<(SsaValue, SsaValue)> {
    // `d = d` is a no-op (every SsaValue maps to its own register).
    pairs.retain(|(d, s)| d != s);

    let mut emitted: Vec<(SsaValue, SsaValue)> = Vec::new();
    let mut temp_counter: u32 = 0;

    while !pairs.is_empty() {
        // A pair `(d, _)` is "ready" if `d` isn't anyone's source — we
        // can safely write to `d` because no remaining move reads from
        // it.
        let leaf = pairs
            .iter()
            .position(|(d, _)| !pairs.iter().any(|(_, s)| s == d));
        match leaf {
            Some(i) => emitted.push(pairs.swap_remove(i)),
            None => {
                // Cycle. Save the first pair's dest into a fresh temp
                // before it gets overwritten, and redirect all reads
                // of it to the temp; pairs[0] then becomes a leaf.
                let (cycle_dest, _) = pairs[0];
                let temp = SsaValue {
                    symbol: cycle_dest.symbol,
                    // Reserve the top of the version space for temps
                    // — unlikely to collide with any real SSA version.
                    version: u32::MAX - temp_counter,
                };
                temp_counter += 1;
                emitted.push((temp, cycle_dest));
                for (_, s) in pairs.iter_mut() {
                    if *s == cycle_dest {
                        *s = temp;
                    }
                }
            }
        }
    }

    emitted
}

fn move_stmt(dest: SsaValue, src: SsaValue, span: oqi_lex::Span) -> SsaStmt {
    SsaStmt {
        kind: SsaStmtKind::Assignment(SsaAssignment {
            target: SsaLValue::Var(dest),
            value: RValue::Expr(Box::new(var_expr(src, span))),
        }),
        annotations: Vec::<Annotation>::new(),
        span,
    }
}

fn var_expr(v: SsaValue, span: oqi_lex::Span) -> SsaExpr {
    SsaExpr {
        kind: SsaExprKind::Var(v),
        // The bytecode emitter resolves register types from the symbol
        // table, not from the SsaExpr `ty` field, so Void is fine.
        ty: crate::types::Type::Void,
        span,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::sequentialize;
    use crate::ssa::SsaValue;
    use crate::symbol::SymbolId;

    fn v(symbol: usize, version: u32) -> SsaValue {
        SsaValue {
            symbol: SymbolId(symbol),
            version,
        }
    }

    /// Execute `moves` sequentially and assert the result matches the
    /// parallel semantics of `pairs`: every dest holds its src's
    /// *initial* value. Initial values are the SsaValues themselves.
    fn assert_parallel_semantics(pairs: Vec<(SsaValue, SsaValue)>) {
        let mut env: HashMap<SsaValue, SsaValue> = HashMap::new();
        for (dest, src) in sequentialize(pairs.clone()) {
            let val = env.get(&src).copied().unwrap_or(src);
            env.insert(dest, val);
        }
        for (dest, src) in pairs {
            assert_eq!(
                env.get(&dest).copied().unwrap_or(dest),
                src,
                "dest {dest:?} should hold the initial value of {src:?}"
            );
        }
    }

    #[test]
    fn chain_preserves_values() {
        // a2 = a1, b3 = a2. Parallel semantics: b3 gets the OLD a2, so
        // `b3 = a2` must be ordered before `a2 = a1`.
        assert_parallel_semantics(vec![(v(0, 2), v(0, 1)), (v(1, 3), v(0, 2))]);
    }

    #[test]
    fn swap_cycle_preserves_both_values() {
        // a = b, b = a — needs a temp.
        assert_parallel_semantics(vec![(v(0, 2), v(1, 2)), (v(1, 2), v(0, 2))]);
    }

    #[test]
    fn three_cycle_preserves_all_values() {
        // a = b, b = c, c = a.
        assert_parallel_semantics(vec![
            (v(0, 1), v(1, 1)),
            (v(1, 1), v(2, 1)),
            (v(2, 1), v(0, 1)),
        ]);
    }

    #[test]
    fn self_move_is_dropped() {
        assert!(sequentialize(vec![(v(0, 2), v(0, 2))]).is_empty());
    }

    #[test]
    fn cycle_plus_leaf() {
        // d = a (leaf), a = b, b = a (cycle).
        assert_parallel_semantics(vec![
            (v(3, 1), v(0, 1)),
            (v(0, 1), v(1, 1)),
            (v(1, 1), v(0, 1)),
        ]);
    }
}
