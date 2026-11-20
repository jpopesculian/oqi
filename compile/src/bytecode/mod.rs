//! Final bytecode IR.
//!
//! The pipeline is `parse → resolve → lower → cfg → bytecode`.
//! Bytecode is a flat, indexed representation suitable for direct
//! consumption by a VM:
//!
//! - **Register-based**: each classical variable of a procedure lives in
//!   one register of its frame; expressions are three-address instructions
//!   over registers and pooled constants.
//! - **Nested bodies lifted**: each inline `cal { ... }` body and
//!   `durationof({...})` scope becomes its own procedure in the module's
//!   procedure table, referenced by index, and runs on the frame of the
//!   procedure it appears in.
//! - **Constants pooled**: literal values live in a module-level
//!   `Vec<classical::Value>`; instructions reference them by index.
//! - **Strings pooled**: pragma payloads, opaque cal text, etc.
//! - **Both binary and text formats**: postcard-encoded binary (the
//!   primary, VM-ingestable form) plus a textual disassembly (debug
//!   aid).

mod analysis;
mod binary;
mod disasm;
mod emit;
mod types;

pub use analysis::is_sample_safe;
pub use binary::{DecodeError, from_bytes, to_bytes};
pub use emit::emit;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::DefaultIncludeResolver;
    use crate::types::CompileOptions;

    fn symbol_id(module: &BcModule, name: &str) -> crate::symbol::SymbolId {
        let pos = module.symbols.iter().position(|s| s.name == name);
        crate::symbol::SymbolId(pos.expect("symbol"))
    }

    fn build_bytecode(src: &str) -> BcModule {
        try_build_bytecode(src).expect("emit bytecode")
    }

    fn try_build_bytecode(src: &str) -> Result<BcModule, crate::error::CompileError> {
        crate::compile(
            src,
            DefaultIncludeResolver,
            &CompileOptions::default(),
            None,
        )
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
    fn a_variable_keeps_one_register_across_branches() {
        let module = build_bytecode(
            r#"
                int c = 0;
                int x = 0;
                if (c == 0) { x = 1; } else { x = 2; }
                int y = x;
            "#,
        );
        // Both branches write the register `y` is later read from.
        let top = &module.procedures[module.entry.0 as usize];
        let moves = top
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .filter_map(|i| match &i.op {
                BcOp::Move { dest, src } => Some((*dest, src)),
                _ => None,
            });
        let moves: Vec<_> = moves.collect();
        let (_, read) = moves.last().expect("`y = x`");
        let BcOperand::Reg(x) = read else {
            panic!("`y` is assigned from a register");
        };
        assert_eq!(moves.iter().filter(|(dest, _)| dest == x).count(), 3);
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
        let flip = symbol_id(&module, "flip");
        let emits_call = all_ops(&module).any(
            |op| matches!(op, BcOp::Call { callee: BcCallTarget::Symbol(s), .. } if *s == flip),
        );
        assert!(
            emits_call,
            "bare `flip q;` should emit a Call to `flip`:\n{module}"
        );
        let emits_gatecall =
            all_ops(&module).any(|op| matches!(op, BcOp::GateCall { gate, .. } if *gate == flip));
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
                    qubit: BcOperand::Whole(QubitSource::Region(id)),
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
                    qubit: BcOperand::Whole(QubitSource::Region(id)),
                } => Some(*id),
                _ => None,
            })
            .expect("static slice should use a region operand");
        assert_eq!(module.qubits.regions[region.0 as usize].ranges, [(1, 3)]);
    }

    #[test]
    fn runtime_index_uses_index_operand() {
        let module = build_bytecode("input uint[32] i; qubit[3] a; reset a[i];");
        let found = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::Index {
                        source: QubitSource::Region(_),
                        ..
                    },
                }
            )
        });
        assert!(found, "runtime index should lower to Index:\n{module}");
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
        // The body references its qubit param by its dense qubit slot
        // (slot 0: the only qubit param, after the classical `n`).
        let body_ok = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::Index {
                        source: QubitSource::Slot(0),
                        ..
                    },
                }
            )
        });
        assert!(body_ok, "body should index qubit slot 0:\n{module}");
        // The call site passes a resolved global region.
        let call_ok = all_ops(&module).any(|op| match op {
            BcOp::Call { args, .. } => {
                matches!(
                    args.as_slice(),
                    [
                        BcOperand::Const(_),
                        BcOperand::Whole(QubitSource::Region(_))
                    ]
                )
            }
            _ => false,
        });
        assert!(call_ok, "call should pass [Const, region]:\n{module}");
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
            BcOp::GateCall { qubits, .. } => {
                matches!(qubits.as_slice(), [BcOperand::Whole(QubitSource::Slot(0))])
            }
            _ => false,
        });
        assert!(body_ok, "gate body should use qubit slot 0:\n{module}");
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
        // (BcOp::AliasBind) and references resolve to that QubitSource::Slot.
        let module = build_bytecode(
            "input uint[32] i; qubit[8] q; let bp = q[{2*i, 2*i + 1}]; reset bp; reset bp[1];",
        );
        let bind = all_ops(&module)
            .any(|op| matches!(op, BcOp::AliasBind { slot: 0, segments } if segments.len() == 2));
        assert!(bind, "runtime alias should emit AliasBind:\n{module}");
        let whole = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::Whole(QubitSource::Slot(0))
                }
            )
        });
        assert!(whole, "`reset bp` should use qubit slot 0:\n{module}");
        let indexed = all_ops(&module).any(|op| {
            matches!(
                op,
                BcOp::Reset {
                    qubit: BcOperand::Index {
                        source: QubitSource::Slot(0),
                        ..
                    }
                }
            )
        });
        assert!(
            indexed,
            "`reset bp[1]` should index qubit slot 0:\n{module}"
        );
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
    fn box_runs_in_its_enclosing_procedure() {
        let module = build_bytecode(
            r#"
                include "stdgates.inc";
                qubit q;
                int n = 2;
                box[100ns] {
                    n = n + 1;
                    h q;
                }
            "#,
        );
        // A box is a scope of the procedure it is in, entered through a
        // terminator a timing pass can skip its contents by.
        let top = &module.procedures[module.entry.0 as usize];
        let block = |id: BlockId| top.blocks.iter().find(|b| b.id == id).unwrap();
        let entry = block(top.entry);
        let BcTerminator::Box { body, after, .. } = &entry.terminator else {
            panic!("a timed box is entered through a box terminator");
        };
        let body = block(*body);
        assert!(matches!(body.terminator, BcTerminator::Goto(t) if t == *after));
        assert!(
            body.instrs
                .iter()
                .any(|i| matches!(i.op, BcOp::GateCall { .. }))
        );
        // `n` is one register, written before the box and inside it.
        let writes = |block: &BcBlock| {
            let dests = block.instrs.iter().filter_map(|i| match &i.op {
                BcOp::Move { dest, .. } | BcOp::Binary { dest, .. } => Some(*dest),
                _ => None,
            });
            dests.collect::<Vec<_>>()
        };
        assert_eq!(writes(entry), writes(body));
    }

    #[test]
    fn a_gate_in_a_box_after_a_measurement_is_not_sample_safe() {
        // The box's contents are part of the entry procedure, so the
        // feedback analysis sees them.
        for boxed in ["box { x q; }", "box[100ns] { x q; }"] {
            let src = format!(r#"include "stdgates.inc"; qubit q; bit c = measure q; {boxed}"#);
            assert!(!is_sample_safe(&build_bytecode(&src)), "{boxed}");
        }
        let src = r#"include "stdgates.inc"; qubit q; box { x q; } bit c = measure q;"#;
        assert!(is_sample_safe(&build_bytecode(src)));
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
        let module = build_bytecode(src);

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
        let module = build_bytecode(src);
        // adder.qasm: cin(1) + a(4) + b(4) + cout(1).
        assert_eq!(module.qubits.num_qubits, 10);
        let text = format!("{module}");
        assert!(text.contains(".qubits 10"), "disasm:\n{text}");
        let bytes = to_bytes(&module).expect("encode");
        assert!(bytes.len() > 4);
        let _ = from_bytes(&bytes).expect("decode");
    }

    #[test]
    fn defcal_fixture_calibration_table() {
        let src = include_str!("../../../fixtures/qasm/defcal.qasm");
        let module = build_bytecode(src);

        assert_eq!(module.calibration_grammar.as_deref(), Some("openpulse"));
        // x $0, x $1, rz, measure, zx90_ix, cx — in declaration order.
        assert_eq!(module.calibrations.len(), 6);

        let sym = |name: &str| symbol_id(&module, name);

        // The two `defcal x` share the gate symbol and differ in operand.
        let x = sym("x");
        for (i, hw) in [(0usize, 0u32), (1, 1)] {
            let cal = &module.calibrations[i];
            assert!(matches!(cal.target, BcCalTarget::Gate(s) if s == x));
            assert!(cal.args.is_empty());
            assert!(!cal.has_return);
            assert!(matches!(cal.operands.as_slice(), [BcCalOperand::Hardware(n)] if *n == hw));
        }

        // defcal rz(angle[20] theta) q — one Param arg, one Any operand.
        let rz = sym("rz");
        let cal_rz = &module.calibrations[2];
        assert!(matches!(cal_rz.target, BcCalTarget::Gate(s) if s == rz));
        assert!(matches!(cal_rz.args.as_slice(), [BcCalArg::Param(_)]));
        assert!(matches!(cal_rz.operands.as_slice(), [BcCalOperand::Any]));

        // defcal measure $0 -> bit.
        let cal_measure = &module.calibrations[3];
        assert!(matches!(cal_measure.target, BcCalTarget::Measure));
        assert!(cal_measure.has_return);

        // Every body is OpenPulse and points at its Calibration-owned proc.
        for (i, cal) in module.calibrations.iter().enumerate() {
            let BcCalBody::OpenPulse(p) = &cal.body else {
                panic!("calibration {i} should have an OpenPulse body");
            };
            assert!(matches!(
                module.procedures[p.0 as usize].owner,
                ProcOwner::Calibration(j) if j == i as u32
            ));
        }

        // Roundtrip preserves the table and the grammar.
        let bytes = to_bytes(&module).expect("encode");
        let module2 = from_bytes(&bytes).expect("decode");
        assert_eq!(module2.calibrations.len(), 6);
        assert_eq!(module2.calibration_grammar.as_deref(), Some("openpulse"));

        // Defcal bodies read the shared frames through entry cal_loads;
        // the inline cal block cal_stores the frames it creates.
        let text = format!("{module}");
        assert!(text.contains(".defcalgrammar openpulse"), "disasm:\n{text}");
        assert!(text.contains(".calibrations"), "disasm:\n{text}");
        assert!(text.contains("cal_load"), "disasm:\n{text}");
        assert!(text.contains("cal_store"), "disasm:\n{text}");
    }

    #[test]
    fn opaque_cal_bodies_keep_text() {
        let module = build_bytecode(
            r#"
                defcalgrammar "mypulses";
                cal { raw pulse text }
                defcal x $0 { more raw text }
            "#,
        );
        assert_eq!(module.calibration_grammar.as_deref(), Some("mypulses"));
        assert_eq!(module.calibrations.len(), 1);
        let BcCalBody::Opaque(s) = &module.calibrations[0].body else {
            panic!("non-openpulse defcal should have an opaque body");
        };
        assert!(module.strings[s.0 as usize].contains("more raw text"));
        assert!(
            all_ops(&module).any(|op| matches!(op, BcOp::CalOpaque { .. })),
            "inline cal should emit CalOpaque:\n{module}"
        );
    }
}
