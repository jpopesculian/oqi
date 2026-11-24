use std::collections::HashSet;
use std::path::Path;

use crate::cfg::{self, BasicBlockId, Cfg, ProgramCfgs, Terminator};
use crate::classical::{IntWidth, ValueTy};
use crate::error::{CompileError, ErrorKind};
use crate::lower::compile_to_sir;
use crate::resolve::DefaultIncludeResolver;
use crate::sir;
use crate::types::{CompileOptions, Type};

fn compile(src: &str) -> sir::Program {
    compile_to_sir(
        src,
        &DefaultIncludeResolver,
        &CompileOptions::default(),
        None,
    )
    .expect("compile should succeed")
}

fn compile_fixture(name: &str) -> sir::Program {
    let path_str = format!("../fixtures/qasm/{name}");
    let path = Path::new(&path_str);
    let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path_str}: {e}"));
    let options = CompileOptions {
        source_name: Some(path.to_path_buf()),
        ..Default::default()
    };
    compile_to_sir(&source, &DefaultIncludeResolver, &options, None)
        .expect("compile should succeed")
}

fn build(p: &sir::Program) -> ProgramCfgs {
    try_build(p).expect("CFG build should succeed")
}

fn try_build(p: &sir::Program) -> Result<ProgramCfgs, CompileError> {
    cfg::build_program(p, &mut p.symbols.clone())
}

/// The blocks `term` can transfer control to.
fn successors(term: &Terminator) -> Vec<BasicBlockId> {
    match term {
        Terminator::Goto(t) => vec![*t],
        Terminator::Branch {
            then_bb, else_bb, ..
        } => vec![*then_bb, *else_bb],
        Terminator::Box { body, after, .. } => vec![*body, *after],
        Terminator::Switch { cases, default, .. } => {
            cases.iter().map(|(_, bb)| *bb).chain([*default]).collect()
        }
        Terminator::Return(_) | Terminator::End | Terminator::Unreachable => Vec::new(),
    }
}

/// The synthetic fall-off-the-end block: always the last one.
fn exit(cfg: &Cfg) -> BasicBlockId {
    cfg.blocks.last().expect("a CFG has an exit block").id
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
        stack.extend(successors(&cfg.blocks[bb.0].terminator));
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
        Terminator::Goto(t) if t == exit(top)
    ));
    assert!(matches!(
        top.blocks[exit(top).0].terminator,
        Terminator::Return(None)
    ));
    // Both blocks reachable.
    let r = reachable(top);
    assert!(r.contains(&top.entry));
    assert!(r.contains(&exit(top)));
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
    assert_eq!(body_succ, header.id, "while body must return to its header");
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
    let Terminator::Switch { cases, default, .. } = &switch_bb.terminator else {
        unreachable!()
    };

    assert_eq!(cases.len(), 2, "two value-labelled cases expected");
    let default_bb = *default;

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

// ── For ──────────────────────────────────────────────────────────────

/// The pieces of the (single) loop in `cfg`: the header's condition, and
/// its body, latch and exit blocks.
fn loop_parts(cfg: &Cfg) -> (&sir::Expr, &cfg::BasicBlock, &cfg::BasicBlock, BasicBlockId) {
    let header = cfg
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("a for-loop should produce a header Branch");
    let Terminator::Branch {
        cond,
        then_bb,
        else_bb,
    } = &header.terminator
    else {
        unreachable!()
    };
    let body = &cfg.blocks[then_bb.0];
    let Terminator::Goto(latch) = body.terminator else {
        panic!("body should goto latch");
    };
    let latch = &cfg.blocks[latch.0];
    assert!(
        matches!(latch.terminator, Terminator::Goto(t) if t == header.id),
        "the latch re-enters the header"
    );
    (cond, body, latch, *else_bb)
}

fn assigned(stmt: &sir::Stmt) -> (crate::symbol::SymbolId, &sir::Expr) {
    match &stmt.kind {
        sir::StmtKind::Assignment(sir::Assignment {
            target: sir::LValue::Var(symbol),
            value: sir::RValue::Expr(value),
        }) => (*symbol, value),
        _ => panic!("expected an assignment to a variable"),
    }
}

#[test]
fn for_range_loop_steps_a_hidden_counter() {
    let p = compile(
        r#"
            int sum = 0;
            for int i in [0:2:20] {
                sum += i;
            }
        "#,
    );
    let cfgs = build(&p);
    let (cond, body, latch, _) = loop_parts(&cfgs.top_level);
    let wide = Type::Classical(ValueTy::int(IntWidth::B128));

    // The header tests the counter, which is wide enough not to overflow.
    let sir::ExprKind::Binary(test) = &cond.kind else {
        panic!("header condition should be a comparison");
    };
    assert!(matches!(test.op, sir::BinOp::Lte));
    let sir::ExprKind::Var(counter) = test.left.kind else {
        panic!("the header should test the counter");
    };
    assert_eq!(test.left.ty, wide);

    // The body starts by giving the loop variable the counter's value.
    let (var, value) = assigned(&body.stmts[0]);
    assert_eq!(p.symbols.get(var).name, "i");
    assert_ne!(var, counter, "assigning `i` must not disturb the iteration");
    assert!(matches!(&value.kind, sir::ExprKind::Cast(c)
        if matches!(c.operand.kind, sir::ExprKind::Var(v) if v == counter)));

    // The latch advances the counter by the step.
    assert_eq!(latch.stmts.len(), 1);
    let (advanced, next) = assigned(&latch.stmts[0]);
    assert_eq!(advanced, counter);
    assert!(matches!(&next.kind, sir::ExprKind::Binary(b) if matches!(b.op, sir::BinOp::Add)));
}

/// Comparison op of the loop header in `cfg`.
fn header_cmp_op(cfg: &Cfg) -> sir::BinOp {
    let (cond, ..) = loop_parts(cfg);
    let sir::ExprKind::Binary(b) = &cond.kind else {
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
fn for_range_run_time_step_tests_both_directions() {
    let p = compile(
        r#"
            int step = 2;
            for int i in [0:step:6] { }
        "#,
    );
    let cfgs = build(&p);
    assert!(matches!(header_cmp_op(&cfgs.top_level), sir::BinOp::LogOr));
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
fn for_set_and_register_loops_index_what_they_iterate() {
    for src in [
        "int b = 0; for int i in {1, 5, 10} { b += i; }",
        r#"bit[4] r = "1010"; int b = 0; for bit x in r { b += int(x); }"#,
    ] {
        let p = compile(src);
        let cfgs = build(&p);
        let (cond, body, ..) = loop_parts(&cfgs.top_level);
        assert!(matches!(&cond.kind, sir::ExprKind::Binary(b) if matches!(b.op, sir::BinOp::Lte)));
        let (_, value) = assigned(&body.stmts[0]);
        let indexed = match &value.kind {
            sir::ExprKind::Cast(c) => &c.operand.kind,
            other => other,
        };
        assert!(matches!(indexed, sir::ExprKind::Index(_)), "{src}");
    }
}

#[test]
fn break_and_continue_leave_the_body() {
    let p = compile(
        r#"
            for int i in [0:9] {
                if (i == 2) { continue; }
                if (i == 5) { break; }
            }
        "#,
    );
    let top = build(&p).top_level;
    let goes_to = |target: BasicBlockId| {
        let blocks = top.blocks.iter();
        blocks.filter(move |b| matches!(b.terminator, Terminator::Goto(t) if t == target))
    };
    let header = top
        .blocks
        .iter()
        .find(|b| matches!(b.terminator, Terminator::Branch { .. }))
        .expect("a loop header");
    let Terminator::Branch { else_bb: after, .. } = header.terminator else {
        unreachable!()
    };
    // Only the latch re-enters the header; `continue` and the end of the
    // body both go to the latch, and `break` to the loop's exit.
    let latches: Vec<_> = goes_to(header.id)
        .filter(|b| b.id.0 > header.id.0)
        .collect();
    let [latch] = latches.as_slice() else {
        panic!("expected one latch, got {}", latches.len());
    };
    assert_eq!(goes_to(latch.id).count(), 2);
    assert_eq!(goes_to(after).count(), 1);
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
