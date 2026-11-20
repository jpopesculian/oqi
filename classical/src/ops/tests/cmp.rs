//! Tests of the operators in `ops::cmp`, one module per operator.

mod eq {
    use crate::DurationUnit;

    use crate::index::Index;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn bool_eq() {
        let r = BinOp::Eq
            .apply(bool_scalar(true), bool_scalar(true))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bool_eq_promotes() {
        let r = BinOp::Eq
            .apply(bit_scalar(true), bool_scalar(false))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_uint16_eq_promotes() {
        let r = BinOp::Eq
            .apply(u_scalar(255, 8), u_scalar(255, 16))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_float_eq_promotes() {
        let r = BinOp::Eq
            .apply(
                i_scalar(3, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(3.0),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_eq_uses_partial_eq() {
        let r = BinOp::Eq
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1_000.0, DurationUnit::Ns),
                    Duration,
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, DurationUnit::Us),
                    Duration,
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_eq_same_width() {
        let r = BinOp::Eq
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::uint(0b1010_u128),
                    BitReg(4),
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::uint(0b1010_u128),
                    BitReg(4),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_eq_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(8),
        ));
        assert!(BinOp::Eq.apply(a, b).is_err());
    }

    #[test]
    fn bit_uint_eq_is_ok() {
        let a = bit_scalar(true);
        let b = u_scalar(1, 8);
        let r = BinOp::Eq.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn array_ref_eq_array_ref() {
        let a = into_ref(u_array(&[1, 2, 3, 4], 8, vec![2, 2]))
            .get(&[Index::Item(0)])
            .unwrap();
        let b = into_ref(u_array(&[1, 9, 3, 8], 8, vec![2, 2]))
            .get(&[Index::Item(0)])
            .unwrap();
        let r = BinOp::Eq.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<bool> = a.values().iter().map(|s| s.as_bit()).collect();
                assert_eq!(vals, vec![true, false]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_eq_scalar() {
        let a = into_ref(u_array(&[7, 8, 7], 8, vec![3]))
            .get(&[Index::Select(vec![2, 1, 0])])
            .unwrap();
        let r = BinOp::Eq.apply(a, u_scalar(7, 8)).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<bool> = a.values().iter().map(|s| s.as_bit()).collect();
                assert_eq!(vals, vec![true, false, true]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod neq {
    use crate::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn bool_neq() {
        let r = BinOp::Neq
            .apply(bool_scalar(true), bool_scalar(false))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bool_neq_promotes() {
        let r = BinOp::Neq
            .apply(bit_scalar(true), bool_scalar(true))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_uint16_neq_promotes() {
        let r = BinOp::Neq.apply(u_scalar(255, 8), u_scalar(0, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_float_neq_promotes() {
        let r = BinOp::Neq
            .apply(
                i_scalar(3, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(4.0),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_neq_uses_eq_then_lognot() {
        let r = BinOp::Neq
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1_000.0, DurationUnit::Ns),
                    Duration,
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(2.0, DurationUnit::Us),
                    Duration,
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_neq_same_width() {
        let r = BinOp::Neq
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::uint(0b1010_u128),
                    BitReg(4),
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::uint(0b0101_u128),
                    BitReg(4),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_neq_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(8),
        ));
        assert!(BinOp::Neq.apply(a, b).is_err());
    }

    #[test]
    fn bit_uint_neq_returns_none() {
        let a = bit_scalar(true);
        let b = u_scalar(0, 8);
        let r = BinOp::Neq.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected error"),
        }
    }
}

mod lt {
    use crate::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn bool_lt() {
        assert!(
            BinOp::Lt
                .apply(bool_scalar(false), bool_scalar(true))
                .is_err()
        );
    }

    #[test]
    fn bit_lt() {
        // docs/classical.rst: bits and registers can be compared.
        let result = BinOp::Lt.apply(bit_scalar(false), bit_scalar(true));
        assert_eq!(result.unwrap().to_string(), "true");
    }

    #[test]
    fn bitreg_lt_same_width() {
        let result = BinOp::Lt.apply(bitreg(0b0011, 4), bitreg(0b0100, 4));
        assert_eq!(result.unwrap().to_string(), "true");
    }

    #[test]
    fn int_float_lt_promotes() {
        let r = BinOp::Lt
            .apply(
                i_scalar(3, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(3.5),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_lt() {
        let r = BinOp::Lt
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, DurationUnit::Us),
                    Duration,
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(2_000.0, DurationUnit::Ns),
                    Duration,
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_lt_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(2.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        assert!(BinOp::Lt.apply(a, b).is_err());
    }

    #[test]
    fn bitreg_lt_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0011),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0100),
            BitReg(8),
        ));
        assert!(BinOp::Lt.apply(a, b).is_err());
    }

    #[test]
    fn bit_uint_lt_returns_none() {
        let a = bit_scalar(true);
        let b = u_scalar(0, 8);
        let r = BinOp::Lt.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected error"),
        }
    }
}

mod gt {
    use crate::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn bool_gt() {
        assert!(
            BinOp::Gt
                .apply(bool_scalar(true), bool_scalar(false))
                .is_err()
        );
    }

    #[test]
    fn bit_gt() {
        // docs/classical.rst: bits and registers can be compared.
        let result = BinOp::Gt.apply(bit_scalar(true), bit_scalar(false));
        assert_eq!(result.unwrap().to_string(), "true");
    }

    #[test]
    fn bitreg_gt_same_width() {
        let result = BinOp::Gt.apply(bitreg(0b0100, 4), bitreg(0b0011, 4));
        assert_eq!(result.unwrap().to_string(), "true");
    }

    #[test]
    fn int_float_gt_promotes() {
        let r = BinOp::Gt
            .apply(
                i_scalar(4, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(3.5),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_gt() {
        let r = BinOp::Gt
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(2_000.0, DurationUnit::Ns),
                    Duration,
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, DurationUnit::Us),
                    Duration,
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_gt_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(2.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        assert!(BinOp::Gt.apply(a, b).is_err());
    }

    #[test]
    fn bitreg_gt_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0100),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0011),
            BitReg(8),
        ));
        assert!(BinOp::Gt.apply(a, b).is_err());
    }

    #[test]
    fn bit_uint_gt_is_ok() {
        let a = bit_scalar(true);
        let b = u_scalar(0, 8);
        let r = BinOp::Gt.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected error"),
        }
    }
}

mod lte {
    use crate::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn bool_lte_is_err() {
        assert!(
            BinOp::Lte
                .apply(bool_scalar(true), bool_scalar(true))
                .is_err()
        );
    }

    #[test]
    fn int_float_lte_via_lt() {
        let r = BinOp::Lte
            .apply(
                i_scalar(3, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(3.5),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_lte_via_eq() {
        let r = BinOp::Lte
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1_000.0, DurationUnit::Ns),
                    Duration,
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, DurationUnit::Us),
                    Duration,
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_lte_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(2.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        assert!(BinOp::Lte.apply(a, b).is_err());
    }
}

mod gte {
    use crate::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn bool_gte_via_eq() {
        assert!(
            BinOp::Gte
                .apply(bool_scalar(false), bool_scalar(false))
                .is_err()
        )
    }

    #[test]
    fn int_float_gte_via_gt() {
        let r = BinOp::Gte
            .apply(
                i_scalar(4, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(3.5),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_gte_via_eq() {
        let r = BinOp::Gte
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1_000.0, DurationUnit::Ns),
                    Duration,
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, DurationUnit::Us),
                    Duration,
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_gte_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(2.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        assert!(BinOp::Gte.apply(a, b).is_err());
    }
}

mod logand {
    use crate::ops::tests::util::*;
    use crate::primitive::PrimitiveTy::*;
    use crate::scalar::Scalar;

    #[test]
    fn bool_logand() {
        let r = BinOp::LogAnd
            .apply(bool_scalar(true), bool_scalar(false))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bool_logand_true_true() {
        let r = BinOp::LogAnd
            .apply(bool_scalar(true), bool_scalar(true))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_logand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(BinOp::LogAnd.apply(a, b).is_err());
    }

    #[test]
    fn bitreg_logand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0b01_u128), BitReg(2)));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0b11_u128), BitReg(2)));
        assert!(BinOp::LogAnd.apply(a, b).is_err());
    }
}

mod logor {
    use crate::ops::tests::util::*;
    use crate::primitive::PrimitiveTy::*;
    use crate::scalar::Scalar;

    #[test]
    fn bool_logor() {
        let r = BinOp::LogOr
            .apply(bool_scalar(true), bool_scalar(false))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bool_logor_false_false() {
        let r = BinOp::LogOr
            .apply(bool_scalar(false), bool_scalar(false))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_logor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(BinOp::LogOr.apply(a, b).is_err());
    }

    #[test]
    fn bitreg_logor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0b01_u128), BitReg(2)));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0b11_u128), BitReg(2)));
        assert!(BinOp::LogOr.apply(a, b).is_err());
    }
}

mod lognot {
    use crate::ops::tests::util::*;
    use crate::primitive::PrimitiveTy::*;
    use crate::scalar::Scalar;

    #[test]
    fn bool_lognot_true() {
        let r = UnOp::LogNot.apply(bool_scalar(true)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bool_lognot_false() {
        let r = UnOp::LogNot.apply(bool_scalar(false)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_lognot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(UnOp::LogNot.apply(a).is_err());
    }

    #[test]
    fn bitreg_lognot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0b01_u128), BitReg(2)));
        assert!(UnOp::LogNot.apply(a).is_err());
    }
}
