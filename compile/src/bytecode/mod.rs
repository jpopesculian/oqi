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
        try_build_bytecode(src).expect("emit bytecode")
    }

    fn try_build_bytecode(src: &str) -> Result<BcModule, crate::error::CompileError> {
        let program = compile_source(src, DefaultIncludeResolver, None).expect("compile");
        let cfgs = cfg::build_program(&program).expect("cfg");
        let ssa = ssa::build_program(&cfgs, &program.symbols);
        let layout = crate::qubits::build_layout(&program);
        emit(&ssa, &program.symbols, layout)
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

    /// All ops of every block of every procedure.
    fn all_ops(module: &BcModule) -> impl Iterator<Item = &BcOp> {
        module
            .procedures
            .iter()
            .flat_map(|p| &p.blocks)
            .flat_map(|b| &b.instrs)
            .map(|i| &i.op)
    }

    #[test]
    fn bare_subroutine_call_emits_call_not_gatecall() {
        // `flip q;` invokes a `def` with bare gate-call syntax; it must lower
        // to BcOp::Call (a subroutine call), never BcOp::GateCall.
        let module = build_bytecode(
            r#"
                include "stdgates.inc";
                def flip(qubit[2] qs) { x qs[0]; x qs[1]; }
                qubit[2] q;
                flip q;
            "#,
        );
        let flip = module
            .symbols
            .iter()
            .find(|s| s.name == "flip")
            .expect("flip symbol")
            .id;
        let emits_call = all_ops(&module).any(|op| {
            matches!(op, BcOp::Call { callee: BcCallTarget::Symbol(s), .. } if *s == flip)
        });
        assert!(emits_call, "bare `flip q;` should emit a Call to `flip`:\n{module}");
        let emits_gatecall = all_ops(&module)
            .any(|op| matches!(op, BcOp::GateCall { gate, .. } if *gate == flip));
        assert!(!emits_gatecall, "bare `flip q;` must not emit a GateCall");
    }

    #[test]
    fn static_index_resolves_to_global_qubit() {
        let module = build_bytecode("qubit[3] a; qubit[2] b; reset b[1];");
        assert_eq!(module.qubits.num_qubits, 5);
        // `b` starts at global 3, so b[1] is global 4.
        let found = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::Qubit(4)
                }
            )
        });
        assert!(found, "reset b[1] should lower to Qubit(4):\n{module}");
    }

    #[test]
    fn whole_register_operand_is_region() {
        let module = build_bytecode("qubit[3] a; reset a;");
        let region = all_ops(&module)
            .find_map(|op| match op {
                BcOp::Reset {
                    qubit: BcOperand::QubitRegion(id),
                } => Some(*id),
                _ => None,
            })
            .expect("reset of a register should use a region operand");
        assert_eq!(module.qubits.regions[region.0 as usize].ranges, [(0, 3)]);
    }

    #[test]
    fn static_slice_operand_is_region() {
        let module = build_bytecode("qubit[4] b; reset b[1:2];");
        let region = all_ops(&module)
            .find_map(|op| match op {
                BcOp::Reset {
                    qubit: BcOperand::QubitRegion(id),
                } => Some(*id),
                _ => None,
            })
            .expect("static slice should use a region operand");
        assert_eq!(module.qubits.regions[region.0 as usize].ranges, [(1, 3)]);
    }

    #[test]
    fn runtime_index_uses_qubit_indexed() {
        let module = build_bytecode("input uint[32] i; qubit[3] a; reset a[i];");
        let found = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::QubitIndexed { .. },
                }
            )
        });
        assert!(
            found,
            "runtime index should lower to QubitIndexed:\n{module}"
        );
    }

    #[test]
    fn qubit_alias_resolves_and_emits_no_alias_op() {
        let module = build_bytecode(
            r#"
                qubit[2] one;
                qubit[10] two;
                let concatenated = one ++ two;
                reset concatenated[5];
            "#,
        );
        // concatenated[5] is global qubit 5 (one ++ two is 0..12).
        let found = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::Qubit(5)
                }
            )
        });
        assert!(found, "alias index should resolve statically:\n{module}");
        let alias_ops = all_ops(&module)
            .filter(|op| matches!(op, BcOp::Alias { .. }))
            .count();
        assert_eq!(alias_ops, 0, "qubit aliases should not emit Alias ops");
    }

    #[test]
    fn subroutine_qubit_param_and_call_args() {
        let module = build_bytecode(
            r#"
                qubit[2] b;
                def f(int n, qubit[2] d) {
                    reset d[0];
                }
                f(1, b);
            "#,
        );
        // The body references its qubit param positionally.
        let body_ok = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::QubitParam {
                        slot: 1,
                        index: Some(_),
                    },
                }
            )
        });
        assert!(body_ok, "body should use QubitParam slot 1:\n{module}");
        // The call site passes a resolved global region.
        let call_ok = all_ops(&module).any(|op| match op {
            BcOp::Call { args, .. } => {
                matches!(
                    args.as_slice(),
                    [BcOperand::Const(_), BcOperand::QubitRegion(_)]
                )
            }
            _ => false,
        });
        assert!(call_ok, "call should pass [Const, QubitRegion]:\n{module}");
    }

    #[test]
    fn gate_body_uses_qubit_params() {
        let module = build_bytecode(
            r#"
                include "stdgates.inc";
                gate flip a {
                    x a;
                }
                qubit q;
                flip q;
            "#,
        );
        let body_ok = all_ops(&module).any(|op| match op {
            BcOp::GateCall { qubits, .. } => matches!(
                qubits.as_slice(),
                [BcOperand::QubitParam {
                    slot: 0,
                    index: None,
                }]
            ),
            _ => false,
        });
        assert!(body_ok, "gate body should use QubitParam slot 0:\n{module}");
        let call_ok = all_ops(&module).any(|op| match op {
            BcOp::GateCall { qubits, .. } => {
                matches!(qubits.as_slice(), [BcOperand::Qubit(0)])
            }
            _ => false,
        });
        assert!(call_ok, "gate call should pass the global qubit:\n{module}");
    }

    #[test]
    fn runtime_indexed_alias_emits_aliasbind_and_qubit_alias() {
        // A `let` over a runtime index set binds a slot at run time
        // (BcOp::AliasBind) and references resolve to BcOperand::QubitAlias.
        let module = build_bytecode(
            "input uint[32] i; qubit[8] q; let bp = q[{2*i, 2*i + 1}]; reset bp; reset bp[1];",
        );
        let bind = all_ops(&module).any(|op| {
            matches!(op, BcOp::AliasBind { slot: 0, segments } if segments.len() == 2)
        });
        assert!(bind, "runtime alias should emit AliasBind:\n{module}");
        let whole = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::QubitAlias {
                        slot: 0,
                        index: None
                    }
                }
            )
        });
        assert!(whole, "`reset bp` should use QubitAlias slot 0:\n{module}");
        let indexed = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::QubitAlias {
                        slot: 0,
                        index: Some(_)
                    }
                }
            )
        });
        assert!(indexed, "`reset bp[1]` should index QubitAlias:\n{module}");
        // Static aliases must still resolve at compile time (no AliasBind).
        let static_mod = build_bytecode("qubit[4] q; let a = q[1:2]; reset a;");
        assert!(
            !all_ops(&static_mod).any(|op| matches!(op, BcOp::AliasBind { .. })),
            "static alias should not emit AliasBind:\n{static_mod}"
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
        let layout = crate::qubits::build_layout(&program);
        let module = emit(&ssa, &program.symbols, layout).expect("emit");

        let bytes = to_bytes(&module).expect("encode");
        assert!(bytes.len() > 4, "encoded module should be non-trivial");
        let module2 = from_bytes(&bytes).expect("decode");
        assert_eq!(module.procedures.len(), module2.procedures.len());
        assert_eq!(module.qubits.num_qubits, module2.qubits.num_qubits);
        assert_eq!(module.qubits.regions.len(), module2.qubits.regions.len());

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
        let layout = crate::qubits::build_layout(&program);
        let module = emit(&ssa, &program.symbols, layout).expect("emit");
        // adder.qasm: cin(1) + a(4) + b(4) + cout(1).
        assert_eq!(module.qubits.num_qubits, 10);
        let text = format!("{module}");
        assert!(text.contains(".qubits 10"), "disasm:\n{text}");
        let bytes = to_bytes(&module).expect("encode");
        assert!(bytes.len() > 4);
        let _ = from_bytes(&bytes).expect("decode");
    }
}
