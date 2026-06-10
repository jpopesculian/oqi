//! Final bytecode IR after SSA.
//!
//! The pipeline is `parse → resolve → lower → cfg → ssa → bytecode`.
//! Bytecode is a flat, indexed representation suitable for direct
//! consumption by a VM:
//!
//! - **Phi-free**: SSA phis are deconstructed into parallel
//!   register-to-register moves at the end of each predecessor block.
//! - **Nested bodies lifted**: each `box { ... }`, inline `cal { ... }`,
//!   and `durationof({...})` body becomes its own procedure in the
//!   module's procedure table, referenced by index.
//! - **Constants pooled**: literal values live in a module-level
//!   `Vec<classical::Value>`; instructions reference them by index.
//! - **Strings pooled**: pragma payloads, opaque cal text, etc.
//! - **Both binary and text formats**: postcard-encoded binary (the
//!   primary, VM-ingestable form) plus a textual disassembly (debug
//!   aid).

pub mod binary;
pub mod disasm;
pub mod emit;
pub mod phi_elim;
pub mod regalloc;
pub mod types;

pub use binary::{DecodeError, EncodeError, from_bytes, to_bytes};
pub use emit::emit;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::lower::compile_source;
    use crate::resolve::DefaultIncludeResolver;
    use crate::ssa;

    fn build_bytecode(src: &str) -> BcModule {
        let program = compile_source(src, DefaultIncludeResolver, None).expect("compile");
        let cfgs = cfg::build_program(&program).expect("cfg");
        let ssa = ssa::build_program(&cfgs, &program.symbols);
        emit(&ssa, &program.symbols)
    }

    #[test]
    fn straight_line_emits_at_least_one_proc() {
        let module = build_bytecode(
            r#"
                int x = 1;
                int y = x + 1;
            "#,
        );
        assert!(!module.procedures.is_empty());
        assert_eq!(module.entry, ProcId(0));
        // Top-level procedure should have at least one block with a Return.
        let proc0 = &module.procedures[0];
        assert!(!proc0.blocks.is_empty());
        let last_term = &proc0.blocks.last().unwrap().terminator;
        assert!(matches!(last_term, BcTerminator::Return(_)));
    }

    #[test]
    fn phi_eliminated_in_if_else() {
        let module = build_bytecode(
            r#"
                int c = 0;
                int x = 0;
                if (c == 0) { x = 1; } else { x = 2; }
                int y = x;
            "#,
        );
        // No phi opcode exists in the bytecode. Check that the merge
        // block contains at least one Move into a register.
        let mut found_move = false;
        for proc in &module.procedures {
            for block in &proc.blocks {
                for instr in &block.instrs {
                    if matches!(instr.op, BcOp::Move { .. }) {
                        found_move = true;
                    }
                }
            }
        }
        assert!(
            found_move,
            "phi elimination should have inserted at least one Move"
        );
    }

    #[test]
    fn indexed_measure_target_stores_element() {
        let module = build_bytecode(
            r#"
                qubit q;
                bit[2] c;
                measure q -> c[0];
            "#,
        );
        // The measured bit must land in a temp register that a
        // StoreElement then writes into the array — not overwrite the
        // array's register.
        let mut measure_dest = None;
        let mut store_value = None;
        for proc in &module.procedures {
            for block in &proc.blocks {
                for instr in &block.instrs {
                    match &instr.op {
                        BcOp::Measure { dest: Some(d), .. } => measure_dest = Some(*d),
                        BcOp::StoreElement { value, .. } => store_value = Some(value.clone()),
                        _ => {}
                    }
                }
            }
        }
        let dest = measure_dest.expect("measure should have a dest register");
        let value = store_value.expect("indexed measure target should emit StoreElement");
        assert!(
            matches!(value, BcOperand::Reg(r) if r == dest),
            "StoreElement should read the measured value's register"
        );
    }

    #[test]
    fn binary_roundtrip_simple() {
        let module = build_bytecode("int x = 1; int y = x + 2;");
        let bytes = to_bytes(&module).expect("encode");
        assert!(bytes.starts_with(b"OQIB"));
        let module2 = from_bytes(&bytes).expect("decode");
        assert_eq!(module.procedures.len(), module2.procedures.len());
        assert_eq!(module.constants.len(), module2.constants.len());
        assert_eq!(module.entry, module2.entry);
    }

    #[test]
    fn bad_magic_rejected() {
        let bad = b"NOPE\x00";
        match from_bytes(bad) {
            Err(DecodeError::BadMagic) => {}
            Err(other) => panic!("expected BadMagic, got {other:?}"),
            Ok(_) => panic!("expected BadMagic, got Ok"),
        }
    }

    #[test]
    fn box_lifted_to_separate_proc() {
        let module = build_bytecode(
            r#"
                include "stdgates.inc";
                qubit q;
                box {
                    int y = 2;
                    y = y + 1;
                }
            "#,
        );
        // At least two procedures (top-level + box body).
        assert!(
            module.procedures.len() >= 2,
            "expected ≥2 procedures, got {}",
            module.procedures.len()
        );
        // Top-level should contain a Box opcode referencing some proc.
        let mut found_box_ref: Option<ProcId> = None;
        for block in &module.procedures[0].blocks {
            for instr in &block.instrs {
                if let BcOp::Box { body, .. } = &instr.op {
                    found_box_ref = Some(*body);
                }
            }
        }
        let body_id = found_box_ref.expect("expected a Box opcode in top-level");
        // The referenced procedure should be a Box owner.
        assert!(matches!(
            module.procedures[body_id.0 as usize].owner,
            ProcOwner::Box
        ));
    }

    #[test]
    fn disasm_mentions_gates() {
        let module = build_bytecode(
            r#"
                include "stdgates.inc";
                qubit q;
                h q;
                x q;
            "#,
        );
        let text = format!("{module}");
        assert!(
            text.contains("gate_call"),
            "disasm missing gate_call:\n{text}"
        );
        assert!(text.contains(".module"), "disasm missing .module header");
        assert!(text.contains(".proc"), "disasm missing .proc");
    }

    #[test]
    fn end_to_end_fixture_teleport() {
        let src = include_str!("../../../fixtures/qasm/teleport.qasm");
        let program = compile_source(src, DefaultIncludeResolver, None).expect("compile");
        let cfgs = cfg::build_program(&program).expect("cfg");
        let ssa = ssa::build_program(&cfgs, &program.symbols);
        let module = emit(&ssa, &program.symbols);

        let bytes = to_bytes(&module).expect("encode");
        assert!(bytes.len() > 4, "encoded module should be non-trivial");
        let module2 = from_bytes(&bytes).expect("decode");
        assert_eq!(module.procedures.len(), module2.procedures.len());

        let text = format!("{module}");
        assert!(text.contains("gate_call"));
        // teleport.qasm uses measure and conditionals.
        assert!(text.contains("measure"));
    }

    #[test]
    fn end_to_end_fixture_adder() {
        let src = include_str!("../../../fixtures/qasm/adder.qasm");
        let program = compile_source(src, DefaultIncludeResolver, None).expect("compile");
        let cfgs = cfg::build_program(&program).expect("cfg");
        let ssa = ssa::build_program(&cfgs, &program.symbols);
        let module = emit(&ssa, &program.symbols);
        let bytes = to_bytes(&module).expect("encode");
        assert!(bytes.len() > 4);
        let _ = from_bytes(&bytes).expect("decode");
    }
}
