use std::collections::HashSet;
use std::path::Path;

use oqi_compile::cfg::{self, BasicBlockId, Cfg, ProgramCfgs, Terminator};
use oqi_compile::error::{CompileError, ErrorKind};
use oqi_compile::lower::compile_source;
use oqi_compile::resolve::DefaultIncludeResolver;
use oqi_compile::sir;

fn compile(src: &str) -> sir::Program {
    compile_source(src, DefaultIncludeResolver, None).expect("compile should succeed")
}

fn compile_fixture(name: &str) -> sir::Program {
    let path_str = format!("../fixtures/qasm/{name}");
    let path = Path::new(&path_str);
    let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path_str}: {e}"));
    compile_source(&source, DefaultIncludeResolver, Some(path)).expect("compile should succeed")
}

fn build(p: &sir::Program) -> ProgramCfgs {
    cfg::build_program(p).expect("CFG build should succeed")
}

fn try_build(p: &sir::Program) -> Result<ProgramCfgs, CompileError> {
    cfg::build_program(p)
}

/// Walk forward from entry collecting every block reachable through any
/// terminator successor.
fn reachable(cfg: &Cfg) -> HashSet<BasicBlockId> {
    let mut seen = HashSet::new();
    let mut stack = vec![cfg.entry];
    while let Some(bb) = stack.pop() {
        if !seen.insert(bb) {
            continue;
        }
        for s in cfg.blocks[bb.0].terminator.successors() {
            stack.push(s);
        }
    }
    seen
}

// ── Straight-line ────────────────────────────────────────────────────

#[test]
fn straight_line_program_has_single_path() {
    let p = compile(
        r#"
            include "stdgates.inc";
            qubit q;
            h q;
            x q;
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    // Entry has no control flow, just falls through to exit.
    assert!(matches!(
        top.blocks[top.entry.0].terminator,
        Terminator::Goto(t) if t == top.exit
    ));
    assert!(matches!(
        top.blocks[top.exit.0].terminator,
        Terminator::Return(None)
    ));
    // Both blocks reachable.
    let r = reachable(top);
    assert!(r.contains(&top.entry));
    assert!(r.contains(&top.exit));
}

// ── If / Else ────────────────────────────────────────────────────────

#[test]
fn if_else_branches_to_then_and_else_then_merges() {
    let p = compile(
        r#"
            include "stdgates.inc";
            qubit q;
            bit r;
            r = measure q;
            if (r == 1) {
                x q;
            } else {
                z q;
            }
            h q;
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    // Find the Branch terminator.
    let branch_bb = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("expected a Branch terminator");
    let Terminator::Branch {
        then_bb, else_bb, ..
    } = branch_bb.terminator
    else {
        unreachable!()
    };

    // Then and else both terminate by Goto to the same merge block.
    let Terminator::Goto(then_succ) = top.blocks[then_bb.0].terminator else {
        panic!("then block should goto merge");
    };
    let Terminator::Goto(else_succ) = top.blocks[else_bb.0].terminator else {
        panic!("else block should goto merge");
    };
    assert_eq!(then_succ, else_succ, "if/else branches must merge");

    // Every block is reachable from entry.
    let r = reachable(top);
    for b in &top.blocks {
        assert!(r.contains(&b.id), "block {:?} should be reachable", b.id);
    }
}

#[test]
fn if_without_else_falls_through_to_merge() {
    let p = compile(
        r#"
            include "stdgates.inc";
            qubit q;
            bit r;
            r = measure q;
            if (r == 1) {
                x q;
            }
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;
    let branch_bb = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("expected a Branch");
    let Terminator::Branch {
        then_bb, else_bb, ..
    } = branch_bb.terminator
    else {
        unreachable!()
    };
    // The else "block" is the empty branch that immediately gotos the merge.
    assert!(matches!(
        top.blocks[else_bb.0].terminator,
        Terminator::Goto(_)
    ));
    let Terminator::Goto(then_succ) = top.blocks[then_bb.0].terminator else {
        panic!("then should goto merge");
    };
    let Terminator::Goto(else_succ) = top.blocks[else_bb.0].terminator else {
        unreachable!()
    };
    assert_eq!(then_succ, else_succ);
}

// ── While ────────────────────────────────────────────────────────────

#[test]
fn while_loop_body_returns_to_header() {
    let p = compile(
        r#"
            int i = 0;
            while (i < 10) {
                i += 1;
            }
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    let header = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("expected a header with Branch");
    let Terminator::Branch {
        then_bb: body_bb,
        else_bb: after_bb,
        ..
    } = header.terminator
    else {
        unreachable!()
    };

    // The body block ends by jumping back to the header.
    let Terminator::Goto(body_succ) = top.blocks[body_bb.0].terminator else {
        panic!("body should goto header");
    };
    assert_eq!(
        body_succ, header.id,
        "while body must return to its header"
    );
    // After block is downstream of header (false branch).
    assert_ne!(after_bb, body_bb);
}

// ── Break / Continue ─────────────────────────────────────────────────

#[test]
fn break_goes_to_loop_after_block() {
    let p = compile(
        r#"
            int i = 0;
            while (i < 10) {
                i += 1;
                if (i == 4) {
                    break;
                }
            }
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    // Find the loop header.
    let header_id = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("header")
        .id;
    let Terminator::Branch {
        else_bb: after_bb, ..
    } = top.blocks[header_id.0].terminator
    else {
        unreachable!()
    };

    // There must exist some block whose Goto target is the loop's after block,
    // and that block is not the header itself (which also gotos the body via
    // Branch, not Goto). That block represents the `break`.
    let break_block_count = top
        .blocks
        .iter()
        .filter(|b| matches!(b.terminator, Terminator::Goto(t) if t == after_bb))
        .count();
    assert!(
        break_block_count >= 1,
        "expected a block that jumps to the loop's after block (break)"
    );
}

#[test]
fn continue_goes_to_loop_continue_target() {
    let p = compile(
        r#"
            int i = 0;
            while (i < 10) {
                i += 1;
                if (i == 2) {
                    continue;
                }
            }
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    // The header is the unique Branch in this CFG (the inner `if` is also a
    // Branch — pick the one whose body terminator goes back to itself or
    // through the inner if).
    let header_id = top
        .blocks
        .iter()
        .filter(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .map(|b| b.id)
        // Pick the outermost Branch — by lowest block index, which is
        // allocated first when entering the while.
        .min_by_key(|id| id.0)
        .expect("expected a header Branch");

    // For while, continue → header. So at least two blocks should goto the
    // header: the body's natural fallthrough and the `continue` block.
    let goto_header_count = top
        .blocks
        .iter()
        .filter(|b| matches!(b.terminator, Terminator::Goto(t) if t == header_id))
        .count();
    assert!(
        goto_header_count >= 2,
        "expected continue + body fallthrough both going to header (got {goto_header_count})"
    );
}

#[test]
fn break_outside_loop_is_an_invalid_context_error() {
    // `break` directly at top level — semantically invalid, surfaced by CFG.
    let p = compile("break;");
    let err = match try_build(&p) {
        Err(e) => e,
        Ok(_) => panic!("break outside loop must error"),
    };
    assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
}

// ── Switch ───────────────────────────────────────────────────────────

#[test]
fn switch_dispatches_to_each_case_then_merges() {
    let p = compile(
        r#"
            int i = 1;
            int x = 0;
            switch (i) {
                case 1 { x = 10; }
                case 2 { x = 20; }
                default { x = 99; }
            }
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    let switch_bb = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Switch { .. }))
        .expect("expected a Switch terminator");
    let Terminator::Switch {
        cases, default, ..
    } = &switch_bb.terminator
    else {
        unreachable!()
    };

    assert_eq!(cases.len(), 2, "two value-labelled cases expected");
    let default_bb = default.expect("default branch should be set");

    // Every case + default should go to the same merge block.
    let merge_targets: HashSet<BasicBlockId> = cases
        .iter()
        .map(|(_, bb)| match top.blocks[bb.0].terminator {
            Terminator::Goto(t) => t,
            _ => panic!("case body should goto merge"),
        })
        .chain(std::iter::once(match top.blocks[default_bb.0].terminator {
            Terminator::Goto(t) => t,
            _ => panic!("default body should goto merge"),
        }))
        .collect();
    assert_eq!(merge_targets.len(), 1, "all cases must merge to one block");
}

// ── For (Range desugar) ──────────────────────────────────────────────

#[test]
fn for_range_loop_desugars_to_header_body_latch() {
    let p = compile(
        r#"
            int sum = 0;
            for int i in [0:2:20] {
                sum += i;
            }
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;

    // Header has Branch on `i <= 20`.
    let header_id = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("for-loop should produce a header Branch")
        .id;

    let Terminator::Branch {
        then_bb: body_bb, ..
    } = top.blocks[header_id.0].terminator
    else {
        unreachable!()
    };

    // Body falls through to a latch block that goes back to header.
    let Terminator::Goto(latch_id) = top.blocks[body_bb.0].terminator else {
        panic!("body should goto latch");
    };
    let Terminator::Goto(latch_succ) = top.blocks[latch_id.0].terminator else {
        panic!("latch should goto header");
    };
    assert_eq!(latch_succ, header_id, "latch must go back to header");

    // Latch contains a synthesized i = i + step assignment.
    assert_eq!(
        top.blocks[latch_id.0].stmts.len(),
        1,
        "latch should hold the synthetic step assignment"
    );
}

/// Comparison op of the (single) Branch terminator in `cfg`.
fn header_cmp_op(cfg: &Cfg) -> sir::BinOp {
    let header = cfg
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("for-loop should produce a header Branch");
    let Terminator::Branch { cond, .. } = &header.terminator else {
        unreachable!()
    };
    let cfg::BlockExprKind::Binary(b) = &cond.kind else {
        panic!("header condition should be a binary comparison");
    };
    b.op
}

#[test]
fn for_range_positive_step_compares_inclusive() {
    // Ranges include `end`: [0:3] iterates 0,1,2,3, so the header
    // must compare with <=.
    let p = compile(
        r#"
            int sum = 0;
            for int i in [0:3] {
                sum += i;
            }
        "#,
    );
    let cfgs = build(&p);
    assert!(matches!(header_cmp_op(&cfgs.top_level), sir::BinOp::Lte));
}

#[test]
fn for_range_negative_step_counts_down() {
    // [10:-1:0] iterates 10,9,...,0, so the header must compare
    // with >=.
    let p = compile(
        r#"
            int sum = 0;
            for int i in [10:-1:0] {
                sum += i;
            }
        "#,
    );
    let cfgs = build(&p);
    assert!(matches!(header_cmp_op(&cfgs.top_level), sir::BinOp::Gte));
}

#[test]
fn for_range_zero_step_is_rejected() {
    let p = compile(
        r#"
            int sum = 0;
            for int i in [0:0:3] {
                sum += i;
            }
        "#,
    );
    let err = match try_build(&p) {
        Err(e) => e,
        Ok(_) => panic!("zero step should be rejected"),
    };
    assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
}

#[test]
fn for_set_iterable_returns_unsupported() {
    let p = compile(
        r#"
            int b = 0;
            for int i in {1, 5, 10} {
                b += i;
            }
        "#,
    );
    let err = match try_build(&p) {
        Err(e) => e,
        Ok(_) => panic!("set-iterable for is not yet supported"),
    };
    assert!(matches!(err.kind, ErrorKind::Unsupported(_)));
}

// ── Subroutines ──────────────────────────────────────────────────────

#[test]
fn subroutines_get_their_own_cfg() {
    let p = compile(
        r#"
            include "stdgates.inc";
            def meas(qubit q) -> bit {
                return measure q;
            }
            qubit q;
            bit c = meas(q);
        "#,
    );
    let cfgs = build(&p);
    assert_eq!(cfgs.subroutines.len(), p.subroutines.len());
    let sub = &cfgs.subroutines[0];
    // The subroutine has an explicit `return`; some block must terminate with Return(Some(_)).
    assert!(
        sub.blocks
            .iter()
            .any(|b| matches!(b.terminator, Terminator::Return(Some(_)))),
        "subroutine should have a Return(Some) terminator"
    );
}

#[test]
fn return_outside_subroutine_is_an_invalid_context_error() {
    let p = compile("return;");
    let err = match try_build(&p) {
        Err(e) => e,
        Ok(_) => panic!("return at top level must error"),
    };
    assert!(matches!(err.kind, ErrorKind::InvalidContext(_)));
}

// ── End / unreachable ────────────────────────────────────────────────

#[test]
fn end_statement_produces_end_terminator() {
    let p = compile(
        r#"
            int x = 0;
            end;
        "#,
    );
    let cfgs = build(&p);
    let top = &cfgs.top_level;
    assert!(
        top.blocks
            .iter()
            .any(|b| matches!(b.terminator, Terminator::End)),
        "expected an End terminator"
    );
}

// ── End-to-end on fixtures ───────────────────────────────────────────

#[test]
fn teleport_fixture_builds_and_is_fully_reachable() {
    let p = compile_fixture("teleport.qasm");
    let cfgs = build(&p);
    let top = &cfgs.top_level;
    let r = reachable(top);
    for b in &top.blocks {
        assert!(
            r.contains(&b.id),
            "teleport CFG block {:?} should be reachable from entry",
            b.id
        );
    }
}
