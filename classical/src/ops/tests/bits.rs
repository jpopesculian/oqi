//! Tests of the operators in `ops::bits`, one module per operator.

mod bitand {
    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar & Scalar ---

    #[test]
    fn uint_bitand() {
        // 0b1100 & 0b1010 = 0b1000
        let r = BinOp::BitAnd
            .apply(u_scalar(0b1100, 8), u_scalar(0b1010, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0b1000),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_bitand() {
        // -1 & 0x0F = 0x0F (in i128, -1 is all bits set)
        let r = BinOp::BitAnd
            .apply(i_scalar(-1, 8), i_scalar(0x0F, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 0x0F),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bitand() {
        // docs/classical.rst: `bit` supports the bitwise operators.
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        let result = BinOp::BitAnd.apply(a, b).unwrap();
        assert_eq!(result.to_string(), "0");
    }

    #[test]
    fn bitreg_bitand_same_width() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1100_u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1010_u128),
            BitReg(4),
        ));
        let r = BinOp::BitAnd.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1000),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_bitand_uint16_promotes() {
        // 0xFF & 0x0F00 promoted to u16 → 0x00FF & 0x0F00 = 0x0F00 & 0x00FF = 0
        let r = BinOp::BitAnd
            .apply(u_scalar(0xFF, 8), u_scalar(0x0F00, 16))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 0x00FF & 0x0F00);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn float_bitand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(BinOp::BitAnd.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_bitand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(F64),
        ));
        assert!(BinOp::BitAnd.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_bitand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(BinOp::BitAnd.apply(a, b).is_err());
    }

    #[test]
    fn angle_bitand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF_u128),
            Angle(iw(8)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0x0F_u128),
            Angle(iw(8)),
        ));
        assert!(BinOp::BitAnd.apply(a, b).is_err());
    }
}

mod bitor {
    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar | Scalar ---

    #[test]
    fn uint_bitor() {
        // 0b1100 | 0b1010 = 0b1110
        let r = BinOp::BitOr
            .apply(u_scalar(0b1100, 8), u_scalar(0b1010, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0b1110),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_bitor() {
        let r = BinOp::BitOr
            .apply(i_scalar(0x0F, 8), i_scalar(0x70, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 0x7F),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bitor() {
        // docs/classical.rst: `bit` supports the bitwise operators.
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        let result = BinOp::BitOr.apply(a, b).unwrap();
        assert_eq!(result.to_string(), "1");
    }

    #[test]
    fn bitreg_bitor_same_width() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1010_u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0101_u128),
            BitReg(4),
        ));
        let r = BinOp::BitOr.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1111),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_bitor_uint16_promotes() {
        let r = BinOp::BitOr
            .apply(u_scalar(0x00FF, 8), u_scalar(0xFF00, 16))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 0xFFFF);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn float_bitor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(BinOp::BitOr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_bitor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(F64),
        ));
        assert!(BinOp::BitOr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_bitor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(BinOp::BitOr.apply(a, b).is_err());
    }

    #[test]
    fn angle_bitor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF_u128),
            Angle(iw(8)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0x0F_u128),
            Angle(iw(8)),
        ));
        assert!(BinOp::BitOr.apply(a, b).is_err());
    }
}

mod bitxor {
    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar ^ Scalar ---

    #[test]
    fn uint_bitxor() {
        // 0b1100 ^ 0b1010 = 0b0110
        let r = BinOp::BitXor
            .apply(u_scalar(0b1100, 8), u_scalar(0b1010, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0b0110),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_bitxor() {
        let r = BinOp::BitXor
            .apply(i_scalar(0x0F, 8), i_scalar(0xFF_u8 as i128, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 0x0F ^ -1),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bitxor() {
        // docs/classical.rst: `bit` supports the bitwise operators.
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        let result = BinOp::BitXor.apply(a, b).unwrap();
        assert_eq!(result.to_string(), "1");
    }

    #[test]
    fn bitreg_bitxor_same_width() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1100_u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1010_u128),
            BitReg(4),
        ));
        let r = BinOp::BitXor.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b0110),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_bitxor_uint16_promotes() {
        let r = BinOp::BitXor
            .apply(u_scalar(0xFF, 8), u_scalar(0xFF0F, 16))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 0x00FF ^ 0xFF0F);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn float_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(BinOp::BitXor.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(F64),
        ));
        assert!(BinOp::BitXor.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(BinOp::BitXor.apply(a, b).is_err());
    }

    #[test]
    fn angle_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF_u128),
            Angle(iw(8)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0x0F_u128),
            Angle(iw(8)),
        ));
        assert!(BinOp::BitXor.apply(a, b).is_err());
    }
}

mod bitnot {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn bit_bitnot() {
        let r = UnOp::BitNot.apply(bit_scalar(true)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bit));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_bitnot() {
        let r = UnOp::BitNot.apply(u_scalar(0x0F_u128, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(width) if width.get() == 8));
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0xF0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_bitnot() {
        let r = UnOp::BitNot.apply(i_scalar(0x0F_i128, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(iw(8)).unwrap(), -16);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_bitnot() {
        let r = UnOp::BitNot
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(0b0011_u128),
                BitReg(4),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), BitReg(width) if width == 4));
                assert_eq!(s.value().as_uint(iw(4)).unwrap(), 0b1100);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bool_bitnot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bool));
        assert!(UnOp::BitNot.apply(a).is_err());
    }

    #[test]
    fn float_bitnot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(UnOp::BitNot.apply(a).is_err());
    }
}

mod shl {

    use crate::duration::DurationUnit;
    use crate::index::Index;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar << Scalar ---

    #[test]
    fn uint_shl() {
        let r = BinOp::Shl.apply(u_scalar(1, 8), u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 8),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_shl_overflow_returns_truncated() {
        assert!(BinOp::Shl.apply(u_scalar(128, 8), u_scalar(1, 8)).is_ok());
    }

    #[test]
    fn int_shl() {
        let r = BinOp::Shl.apply(i_scalar(1, 8), u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 8),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_shl_overflow_returns_truncated() {
        assert!(BinOp::Shl.apply(i_scalar(64, 8), u_scalar(1, 8)).is_ok());
    }

    // --- Invalid types ---

    #[test]
    fn float_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(BinOp::Shl.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(F64),
        ));
        assert!(BinOp::Shl.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn bit_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(BinOp::Shl.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(BinOp::Shl.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn angle_shl_doubles_numerator() {
        // angle[4] with turns numerator 1 (stored left-aligned as 1 << 124).
        // Shifting left by 1 doubles the numerator to 2.
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(1u128 << 124)),
            Angle(iw(4)),
        ));
        match BinOp::Shl.apply(a, u_scalar(1, 8)).unwrap() {
            Value::Scalar(s) => match s.value() {
                Primitive::Angle(out) => assert_eq!(out.0, 2u128 << 124),
                _ => panic!("expected angle primitive"),
            },
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn float_rhs_returns_none() {
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(BinOp::Shl.apply(u_scalar(1, 8), b).is_err());
    }

    // --- Array << Scalar ---

    #[test]
    fn array_shl_scalar() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let s = u_scalar(2, 8);
        let r = BinOp::Shl.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![4, 8, 12]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Array combinations ---

    #[test]
    fn array_shl_array() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let b = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Shl.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 8, 24]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_shl_array() {
        let s = u_scalar(1, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Shl.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 4, 8]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_shl_scalar() {
        let a = into_ref(u_array(&[1, 2, 3], 8, vec![3]))
            .get(&[Index::Select(vec![2, 0])])
            .unwrap();
        let r = BinOp::Shl.apply(a, u_scalar(1, 8)).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![6, 2]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_shl_array_ref() {
        let a = into_ref(u_array(&[1, 2, 3], 8, vec![3]))
            .get(&[Index::Slice {
                start: Some(0),
                step: Some(2),
                end: Some(3),
            }])
            .unwrap();
        let r = BinOp::Shl.apply(u_scalar(1, 8), a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 8]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod shr {

    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar >> Scalar ---

    #[test]
    fn uint_shr() {
        let r = BinOp::Shr.apply(u_scalar(8, 8), u_scalar(2, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 2),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_shr_arithmetic() {
        // -4 >> 1 = -2 (arithmetic right shift preserves sign)
        let r = BinOp::Shr.apply(i_scalar(-4, 8), u_scalar(1, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), -2),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_shr_positive() {
        let r = BinOp::Shr.apply(i_scalar(16, 8), u_scalar(2, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 4),
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn float_shr_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(8.0), Float(F64)));
        assert!(BinOp::Shr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_shr_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(8.0, 0.0),
            Complex(F64),
        ));
        assert!(BinOp::Shr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn bit_shr_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(BinOp::Shr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_shr_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(8.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(BinOp::Shr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn angle_shr_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(8_u128), Angle(iw(8))));
        assert!(BinOp::Shr.apply(a, u_scalar(1, 8)).is_err());
    }

    #[test]
    fn float_rhs_returns_none() {
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(BinOp::Shr.apply(u_scalar(8, 8), b).is_err());
    }

    // --- Array >> Scalar ---

    #[test]
    fn array_shr_scalar() {
        let a = u_array(&[8, 16, 32], 8, vec![3]);
        let s = u_scalar(2, 8);
        let r = BinOp::Shr.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 4, 8]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Array combinations ---

    #[test]
    fn array_shr_array() {
        let a = u_array(&[8, 16, 32], 8, vec![3]);
        let b = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Shr.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![4, 4, 4]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_shr_array() {
        let s = u_scalar(8, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Shr.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![4, 2, 1]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod rotl {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn rotl_bitreg_by_3() {
        // 0b0010_1010 rotated left 3 = 0b0101_0001
        let r = BinOp::Rotl
            .apply(bitreg(0b0010_1010, 8), i_scalar(3, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), BitReg(w) if w == 8));
                assert_eq!(s.value().as_bitreg(8).unwrap().as_u128(), 0b0101_0001);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_uint_by_1() {
        // 0b1000_0001 rotated left 1 in 8 bits = 0b0000_0011
        let r = BinOp::Rotl
            .apply(u_scalar(0b1000_0001, 8), i_scalar(1, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 8));
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0b0000_0011);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_by_zero() {
        let r = BinOp::Rotl
            .apply(bitreg(0b1010, 4), i_scalar(0, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_by_width_is_identity() {
        let r = BinOp::Rotl
            .apply(bitreg(0b1010, 4), i_scalar(4, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_negative_distance() {
        // rotl(a, -1) == rotr(a, 1): rotate right by 1
        // 0b0001 in 4 bits rotated left by -1 = rotated right by 1 = 0b1000
        let r = BinOp::Rotl
            .apply(bitreg(0b0001, 4), i_scalar(-1, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1000);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_with_uint_distance() {
        let r = BinOp::Rotl
            .apply(bitreg(0b0001, 4), u_scalar(1, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b0010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_float_returns_none() {
        assert!(
            BinOp::Rotl
                .apply(
                    Value::Scalar(Scalar::new_unchecked(
                        Primitive::float(1.0),
                        Float(FloatWidth::F64),
                    )),
                    i_scalar(1, 8)
                )
                .is_err()
        );
    }

    #[test]
    fn rotl_int_value_returns_none() {
        assert!(BinOp::Rotl.apply(i_scalar(5, 8), i_scalar(1, 8)).is_err());
    }
}

mod rotr {
    use crate::ops::tests::util::*;
    use crate::primitive::{PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn rotr_bitreg_by_1() {
        // 0b0001 in 4 bits rotated right by 1 = 0b1000
        let r = BinOp::Rotr
            .apply(bitreg(0b0001, 4), i_scalar(1, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1000);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotr_uint_by_2() {
        // 0b0000_0011 rotated right 2 in 8 bits = 0b1100_0000
        let r = BinOp::Rotr
            .apply(u_scalar(0b0000_0011, 8), i_scalar(2, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 8));
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0b1100_0000);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotr_negative_equals_rotl() {
        // rotr(a, -3) == rotl(a, 3)
        let a = bitreg(0b0010_1010, 8);
        let r = BinOp::Rotr.apply(a, i_scalar(-3, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(8).unwrap().as_u128(), 0b0101_0001);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotr_by_zero() {
        let r = BinOp::Rotr
            .apply(bitreg(0b1010, 4), i_scalar(0, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotr_by_width_is_identity() {
        let r = BinOp::Rotr
            .apply(bitreg(0b1010, 4), i_scalar(4, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(4).unwrap().as_u128(), 0b1010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotr_float_returns_none() {
        assert!(
            BinOp::Rotr
                .apply(
                    Value::Scalar(Scalar::new_unchecked(
                        Primitive::float(1.0),
                        Float(crate::primitive::FloatWidth::F64),
                    )),
                    i_scalar(1, 8)
                )
                .is_err()
        );
    }
}

mod popcount {
    use crate::ops::tests::util::*;
    use crate::primitive::{PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn popcount_all_zeros() {
        let r = UnOp::Popcount.apply(bitreg(0b0000_0000, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 8));
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_all_ones() {
        let r = UnOp::Popcount.apply(bitreg(0xFF, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 8);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_mixed() {
        let r = UnOp::Popcount.apply(bitreg(0b1010_1010, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 4);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_single_bit() {
        let r = UnOp::Popcount.apply(bitreg(0b1, 4)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 4));
                assert_eq!(s.value().as_uint(iw(4)).unwrap(), 1);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_uint_counts_its_bits() {
        // docs/classical.rst: `uint[6] b = 37; popcount(b); // Produces 3.`
        let r = UnOp::Popcount.apply(u_scalar(37, 6)).unwrap();
        assert_eq!(r.ty(), ValueTy::uint(iw(6)));
        assert_eq!(r.to_string(), "3");
    }

    #[test]
    fn popcount_of_a_full_128_bit_register() {
        let r = UnOp::Popcount.apply(bitreg(u128::MAX, 128)).unwrap();
        assert_eq!(r.to_string(), "128");
    }

    #[test]
    fn popcount_float_returns_none() {
        assert!(
            UnOp::Popcount
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(1.0),
                    Float(crate::primitive::FloatWidth::F64),
                )))
                .is_err()
        );
    }
}

mod shift_distance {
    use crate::ops::tests::util::*;
    use crate::primitive::iw;
    use crate::scalar::Scalar;

    fn angle4(bits: u128) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(bits << 124)),
            PrimitiveTy::Angle(iw(4)),
        ))
    }

    #[test]
    fn a_distance_of_the_width_or_more_shifts_every_bit_out() {
        for distance in [8, 127, 128, 200, u128::MAX] {
            let distance = || u_scalar(distance, 128);
            let shl = |v: Value| BinOp::Shl.apply(v, distance()).unwrap().to_string();
            let shr = |v: Value| BinOp::Shr.apply(v, distance()).unwrap().to_string();
            assert_eq!(shl(u_scalar(0xFF, 8)), "0");
            assert_eq!(shr(u_scalar(0xFF, 8)), "0");
            assert_eq!(shl(i_scalar(-1, 8)), "0");
            assert_eq!(
                shr(i_scalar(-1, 8)),
                "-1",
                "arithmetic shift keeps the sign"
            );
            assert_eq!(shr(i_scalar(5, 8)), "0");
            assert_eq!(shl(bitreg(0b1111, 4)), "\"0000\"");
            assert_eq!(shr(bitreg(0b1111, 4)), "\"0000\"");
            assert_eq!(shl(angle4(0b1001)), "0");
            assert_eq!(shr(angle4(0b1001)), "0");
        }
        // A full-width value still shifts by less than its width.
        let r = BinOp::Shl
            .apply(u_scalar(1, 128), u_scalar(127, 8))
            .unwrap();
        assert_eq!(r.to_string(), (1u128 << 127).to_string());
    }

    #[test]
    fn a_negative_distance_reverses_the_direction() {
        let shl = |v: Value, d: i128| BinOp::Shl.apply(v, i_scalar(d, 128)).unwrap().to_string();
        let shr = |v: Value, d: i128| BinOp::Shr.apply(v, i_scalar(d, 128)).unwrap().to_string();
        assert_eq!(shl(u_scalar(0b1000, 8), -2), "2");
        assert_eq!(shr(u_scalar(0b0010, 8), -2), "8");
        // The most negative distance has no positive counterpart in `int`.
        assert_eq!(shl(u_scalar(0xFF, 8), i128::MIN), "0");
        assert_eq!(shr(u_scalar(0xFF, 8), i128::MIN), "0");
    }

    #[test]
    fn angles_shift_as_their_bit_pattern() {
        // docs/classical.rst: `angle[4] a = 9 * (pi / 8);  // "1001"`,
        // `a << 2` is "0100" (pi/2) and `a >> 2` is "0010" (pi/4).
        let a = || angle4(0b1001);
        let two = || u_scalar(2, 8);
        assert_eq!(BinOp::Shl.apply(a(), two()).unwrap().to_string(), "(π/2)");
        assert_eq!(BinOp::Shr.apply(a(), two()).unwrap().to_string(), "(π/4)");
        assert_eq!(
            BinOp::Shr.return_ty(a().ty(), two().ty()).unwrap(),
            a().ty()
        );
    }
}
