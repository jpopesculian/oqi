//! Tests of the operators in `ops::arith`, one module per operator.

mod add {

    use crate::duration::DurationUnit;
    use crate::index::Index;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar + Scalar ---

    #[test]
    fn uint_add() {
        let r = BinOp::Add.apply(u_scalar(3, 8), u_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 7),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_add_wraps() {
        // Unsigned arithmetic is modulo 2^width: 200 + 100 = 300 = 44 (mod 256).
        let r = BinOp::Add
            .apply(u_scalar(200, 8), u_scalar(100, 8))
            .unwrap();
        assert_eq!(r.to_string(), "44");
        let full = BinOp::Add
            .apply(u_scalar(u128::MAX, 128), u_scalar(2, 128))
            .unwrap();
        assert_eq!(full.to_string(), "1");
    }

    #[test]
    fn int_add() {
        // -3 + 5 = 2 in Int(8)
        let r = BinOp::Add.apply(i_scalar(-3, 8), i_scalar(5, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 2),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_overflow_returns_none() {
        // 100 + 100 = 200 overflows Int(8) (max 127)
        let a = i_scalar(100, 8);
        let b = i_scalar(100, 8);
        assert!(BinOp::Add.apply(a, b).is_err());
    }

    #[test]
    fn int_negative_overflow_returns_none() {
        // -100 + -100 = -200 overflows Int(8) (min -128)
        let a = i_scalar(-100, 8);
        let b = i_scalar(-100, 8);
        assert!(BinOp::Add.apply(a, b).is_err());
    }

    #[test]
    fn f64_add() {
        let r = BinOp::Add.apply(f64_scalar(1.5), f64_scalar(2.25)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 3.75),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn c64_add() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 2.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(3.0, -1.0),
            Complex(F64),
        ));
        let r = BinOp::Add.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert_eq!(c.re, 4.0);
                assert_eq!(c.im, 1.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_add() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        let r = BinOp::Add.apply(dur(100.0), dur(200.0)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_duration().unwrap().value, 300.0)
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_add_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(BinOp::Add.apply(a, b).is_err());
    }

    #[test]
    fn bitreg_add_fails() {
        // BitReg is cast to Uint, so addition works
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1010u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0101u128),
            BitReg(4),
        ));
        assert!(BinOp::Add.apply(a, b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_add_uint16_promotes() {
        // Uint(8) + Uint(16) promotes to Uint(16)
        let r = BinOp::Add
            .apply(u_scalar(100, 8), u_scalar(300, 16))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 400);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_add_float_promotes() {
        let r = BinOp::Add.apply(u_scalar(3, 8), f64_scalar(0.5)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 3.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn incompatible_types_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = u_scalar(1, 8);
        assert!(BinOp::Add.apply(a, b).is_err());
    }

    // --- Array + Array ---

    #[test]
    fn array_add_elementwise() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let b = u_array(&[10, 20, 30], 8, vec![3]);
        let r = BinOp::Add.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![11, 22, 33]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_shape_mismatch_returns_none() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let b = u_array(&[1, 2], 8, vec![2]);
        assert!(BinOp::Add.apply(a, b).is_err());
    }

    #[test]
    fn array_element_add_wraps() {
        let a = u_array(&[1, 255, 3], 8, vec![3]);
        let b = u_array(&[0, 1, 0], 8, vec![3]);
        assert_eq!(BinOp::Add.apply(a, b).unwrap().to_string(), "{1, 0, 3}");
    }

    // --- Scalar / Array mixing ---

    #[test]
    fn scalar_add_array() {
        let s = u_scalar(10, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Add.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![11, 12, 13]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_add_scalar() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let s = u_scalar(10, 8);
        let r = BinOp::Add.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![11, 12, 13]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_add_array_ref() {
        let a = into_ref(u_array(&[1, 2, 3, 4], 8, vec![2, 2]))
            .get(&[Index::Item(0)])
            .unwrap();
        let b = into_ref(u_array(&[10, 20, 30, 40], 8, vec![2, 2]))
            .get(&[Index::Item(1)])
            .unwrap();
        let r = BinOp::Add.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![31, 42]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_add_array_ref() {
        let a = into_ref(u_array(&[1, 2, 3], 8, vec![3]))
            .get(&[Index::Select(vec![2, 0])])
            .unwrap();
        let r = BinOp::Add.apply(u_scalar(10, 8), a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![13, 11]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_shape_mismatch_returns_none() {
        let a = into_ref(u_array(&[1, 2, 3, 4], 8, vec![2, 2]))
            .get(&[Index::Item(0)])
            .unwrap();
        let b = into_ref(u_array(&[10, 20, 30], 8, vec![3]))
            .get(&[Index::Select(vec![0])])
            .unwrap();
        assert!(BinOp::Add.apply(a, b).is_err());
    }
}

mod sub {

    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar - Scalar ---

    #[test]
    fn uint_sub() {
        let r = BinOp::Sub.apply(u_scalar(10, 8), u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 7),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_sub_wraps() {
        // 3 - 10 = -7 = 249 (mod 256).
        let r = BinOp::Sub.apply(u_scalar(3, 8), u_scalar(10, 8)).unwrap();
        assert_eq!(r.to_string(), "249");
    }

    #[test]
    fn int_sub() {
        // 5 - (-3) = 8 in Int(8)
        let r = BinOp::Sub.apply(i_scalar(5, 8), i_scalar(-3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), 8),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_overflow_returns_none() {
        // 100 - (-100) = 200 overflows Int(8) (max 127)
        assert!(
            BinOp::Sub
                .apply(i_scalar(100, 8), i_scalar(-100, 8))
                .is_err()
        );
    }

    #[test]
    fn int_negative_overflow_returns_none() {
        // -100 - 100 = -200 overflows Int(8) (min -128)
        assert!(
            BinOp::Sub
                .apply(i_scalar(-100, 8), i_scalar(100, 8))
                .is_err()
        );
    }

    #[test]
    fn f64_sub() {
        let r = BinOp::Sub.apply(f64_scalar(3.75), f64_scalar(1.5)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 2.25),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn c64_sub() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(4.0, 1.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 3.0),
            Complex(F64),
        ));
        let r = BinOp::Sub.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert_eq!(c.re, 3.0);
                assert_eq!(c.im, -2.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_sub() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        let r = BinOp::Sub.apply(dur(300.0), dur(100.0)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_duration().unwrap().value, 200.0),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_sub_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(BinOp::Sub.apply(a, b).is_err());
    }

    #[test]
    fn bitreg_sub_err() {
        // BitReg is cast to Uint, so subtraction works
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(4),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b0101_u128),
            BitReg(4),
        ));
        assert!(BinOp::Sub.apply(a, b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_sub_uint16_promotes() {
        let r = BinOp::Sub
            .apply(u_scalar(300, 16), u_scalar(100, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 200);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_sub_float_promotes() {
        let r = BinOp::Sub.apply(u_scalar(3, 8), f64_scalar(0.5)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 2.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn incompatible_types_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = u_scalar(1, 8);
        assert!(BinOp::Sub.apply(a, b).is_err());
    }

    // --- Array - Array ---

    #[test]
    fn array_sub_elementwise() {
        let a = u_array(&[10, 20, 30], 8, vec![3]);
        let b = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Sub.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![9, 18, 27]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_shape_mismatch_returns_none() {
        let a = u_array(&[10, 20, 30], 8, vec![3]);
        let b = u_array(&[1, 2], 8, vec![2]);
        assert!(BinOp::Sub.apply(a, b).is_err());
    }

    #[test]
    fn array_element_sub_wraps() {
        let a = u_array(&[10, 0, 30], 8, vec![3]);
        let b = u_array(&[1, 1, 1], 8, vec![3]);
        assert_eq!(BinOp::Sub.apply(a, b).unwrap().to_string(), "{9, 255, 29}");
    }

    // --- Scalar / Array mixing ---

    #[test]
    fn scalar_sub_array() {
        let s = u_scalar(100, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = BinOp::Sub.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![99, 98, 97]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_sub_scalar() {
        let a = u_array(&[10, 20, 30], 8, vec![3]);
        let s = u_scalar(5, 8);
        let r = BinOp::Sub.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![5, 15, 25]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod mul {

    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};

    // --- Scalar * Scalar ---

    #[test]
    fn uint_mul() {
        let r = BinOp::Mul.apply(u_scalar(3, 8), u_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 12),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_mul_wraps() {
        // 200 * 2 = 400 = 144 (mod 256).
        let r = BinOp::Mul.apply(u_scalar(200, 8), u_scalar(2, 8)).unwrap();
        assert_eq!(r.to_string(), "144");
    }

    #[test]
    fn int_mul() {
        let r = BinOp::Mul.apply(i_scalar(-3, 8), i_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), -12),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_overflow_returns_none() {
        assert!(BinOp::Mul.apply(i_scalar(100, 8), i_scalar(2, 8)).is_err());
    }

    #[test]
    fn f64_mul() {
        let r = BinOp::Mul.apply(f64_scalar(1.5), f64_scalar(4.0)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 6.0),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn c64_mul() {
        // (1+2i) * (3-1i) = 5+5i
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 2.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(3.0, -1.0),
            Complex(F64),
        ));
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert_eq!(c.re, 5.0);
                assert_eq!(c.im, 5.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn bit_mul_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(BinOp::Mul.apply(a, b).is_err());
    }

    #[test]
    fn duration_mul_returns_none() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        assert!(BinOp::Mul.apply(dur(5.0), dur(2.0)).is_err());
    }

    #[test]
    fn angle_mul_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(3_u128), Angle(iw(8))));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::uint(2_u128), Angle(iw(8))));
        assert!(BinOp::Mul.apply(a, b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_mul_uint16_promotes() {
        let r = BinOp::Mul
            .apply(u_scalar(100, 8), u_scalar(300, 16))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 30_000);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_mul_float_promotes() {
        let r = BinOp::Mul.apply(u_scalar(3, 8), f64_scalar(0.5)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 1.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Dot product: [n] · [n] → scalar ---

    #[test]
    fn dot_product_1d() {
        // [1,2,3] · [4,5,6] = 4+10+18 = 32
        let a = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        let b = f64_array(&[4.0, 5.0, 6.0], vec![3]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(s) => {
                assert_eq!(s.ty().shape().dims(), &[1]);
                let value = s.scalars().next().unwrap().into_value();
                assert_eq!(value.as_float(F64).unwrap(), 32.0)
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn dot_product_length_mismatch_returns_none() {
        let a = f64_array(&[1.0, 2.0], vec![2]);
        let b = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        assert!(BinOp::Mul.apply(a, b).is_err());
    }

    // --- Matrix-vector: [m, n] · [n] → [m] ---

    #[test]
    fn matrix_vector_mul() {
        // [[1,2],[3,4]] · [5,6] = [1*5+2*6, 3*5+4*6] = [17, 39]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[5.0, 6.0], vec![2]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![17.0, 39.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn matrix_vector_dim_mismatch_returns_none() {
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        assert!(BinOp::Mul.apply(a, b).is_err());
    }

    // --- Vector-matrix: [n] · [n, p] → [p] ---

    #[test]
    fn vector_matrix_mul() {
        // [1,2] · [[3,4,5],[6,7,8]] = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        let a = f64_array(&[1.0, 2.0], vec![2]);
        let b = f64_array(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 3]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[3]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![15.0, 18.0, 21.0]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Matrix multiply: [m, n] · [n, p] → [m, p] ---

    #[test]
    fn matrix_matrix_mul() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![19.0, 22.0, 43.0, 50.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn matrix_mul_non_square() {
        // [2,3] · [3,2] → [2,2]
        // [[1,2,3],[4,5,6]] · [[7,8],[9,10],[11,12]]
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = f64_array(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![58.0, 64.0, 139.0, 154.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn matrix_mul_inner_dim_mismatch_returns_none() {
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        assert!(BinOp::Mul.apply(a, b).is_err());
    }

    // --- Batch matmul ---

    #[test]
    fn batch_matmul_3d() {
        // [2,2,2] · [2,2,2]: batch of 2 matrix muls, all ones
        // each 2x2 ones @ 2x2 ones = [[2,2],[2,2]]
        let a = f64_array(&[1.0; 8], vec![2, 2, 2]);
        let b = f64_array(&[1.0; 8], vec![2, 2, 2]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[2, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![2.0; 8]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_broadcast_left() {
        // a=[2,2] single matrix, b=[3,2,2] batch of 3
        // a = [[1,2],[3,4]], b = [I, 2I, swap]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(
            &[
                1.0, 0.0, 0.0, 1.0, // I
                2.0, 0.0, 0.0, 2.0, // 2I
                0.0, 1.0, 1.0, 0.0, // swap
            ],
            vec![3, 2, 2],
        );
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[3, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(
                    vals,
                    vec![
                        1.0, 2.0, 3.0, 4.0, // a @ I
                        2.0, 4.0, 6.0, 8.0, // a @ 2I
                        2.0, 1.0, 4.0, 3.0, // a @ swap
                    ]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_broadcast_right() {
        // a=[3,2,2] batch of 3 identities, b=[2,2] single matrix
        let a = f64_array(
            &[
                1.0, 0.0, 0.0, 1.0, // I
                1.0, 0.0, 0.0, 1.0, // I
                1.0, 0.0, 0.0, 1.0, // I
            ],
            vec![3, 2, 2],
        );
        let b = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[3, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                // I @ b = b, repeated 3 times
                assert_eq!(
                    vals,
                    vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_broadcast_ones() {
        // a=[1,2,2] (batch 1), b=[2,2,2] (batch 2)
        // a's single matrix broadcasts across both batches of b
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let b = f64_array(
            &[
                1.0, 0.0, 0.0, 1.0, // I
                0.0, 1.0, 1.0, 0.0, // swap
            ],
            vec![2, 2, 2],
        );
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[2, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(
                    vals,
                    vec![
                        1.0, 2.0, 3.0, 4.0, // a @ I
                        2.0, 1.0, 4.0, 3.0, // a @ swap
                    ]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_incompatible_returns_none() {
        let a = f64_array(&[1.0; 12], vec![3, 2, 2]);
        let b = f64_array(&[1.0; 8], vec![2, 2, 2]);
        assert!(BinOp::Mul.apply(a, b).is_err());
    }

    // --- Overflow in dot product ---

    #[test]
    fn dot_product_wraps() {
        // 200*200 = 40000 = 64 (mod 256).
        let a = u_array(&[200], 8, vec![1]);
        let b = u_array(&[200], 8, vec![1]);
        assert_eq!(BinOp::Mul.apply(a, b).unwrap().to_string(), "{64}");
    }

    #[test]
    fn signed_dot_product_overflow_is_an_error() {
        let ints = |v: i128| {
            Value::Array(Array::new_unchecked(
                vec![Primitive::int(v)],
                ArrayTy::new(Int(iw(8)), crate::ashape(vec![1])),
            ))
        };
        assert!(BinOp::Mul.apply(ints(100), ints(100)).is_err());
    }

    #[test]
    fn dot_product_accumulation_overflow_returns_none() {
        // 100*1 + 100*1 = 200, but products fit individually; sum would too in u16
        // but 200*1 + 200*1 at u8: 200+200=400 overflows
        let a = u_array(&[200, 200], 16, vec![2]);
        let b = u_array(&[1, 1], 16, vec![2]);
        // 200 + 200 = 400, fits in u16. Should succeed.
        let r = BinOp::Mul.apply(a, b).unwrap();
        match r {
            Value::Array(arr) => {
                assert_eq!(arr.ty().shape().dims(), &[1]);
                let value = arr.scalars().next().unwrap().into_value();
                assert_eq!(value.as_uint(iw(16)).unwrap(), 400)
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Scalar * Array broadcast (scaling) ---

    #[test]
    fn scalar_mul_array_broadcast() {
        let s = f64_scalar(2.0);
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = BinOp::Mul.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().dims(), &[2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![2.0, 4.0, 6.0, 8.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_mul_scalar_broadcast() {
        let a = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        let s = f64_scalar(3.0);
        let r = BinOp::Mul.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![3.0, 6.0, 9.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_broadcast_with_promotion() {
        let s = f64_scalar(0.5);
        let a = u_array(&[2, 4, 6], 8, vec![3]);
        let r = BinOp::Mul.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                assert!(matches!(a.ty().element_ty(), Float(F64)));
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod div {

    use crate::duration::DurationUnit;
    use crate::index::Index;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar / Scalar ---

    #[test]
    fn uint_div() {
        let r = BinOp::Div.apply(u_scalar(10, 8), u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 3),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_div_by_zero_returns_none() {
        assert!(BinOp::Div.apply(u_scalar(10, 8), u_scalar(0, 8)).is_err());
    }

    #[test]
    fn int_div() {
        let r = BinOp::Div.apply(i_scalar(-12, 8), i_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), -3),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_div_by_zero_returns_none() {
        assert!(BinOp::Div.apply(i_scalar(10, 8), i_scalar(0, 8)).is_err());
    }

    #[test]
    fn int_min_div_neg1_returns_none() {
        // -128 / -1 = 128 overflows Int(8)
        assert!(
            BinOp::Div
                .apply(i_scalar(-128, 8), i_scalar(-1, 8))
                .is_err()
        );
    }

    #[test]
    fn f64_div() {
        let r = BinOp::Div.apply(f64_scalar(7.5), f64_scalar(2.5)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 3.0),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn f64_div_by_zero_is_inf() {
        let r = BinOp::Div.apply(f64_scalar(1.0), f64_scalar(0.0)).unwrap();
        match r {
            Value::Scalar(s) => assert!(s.value().as_float(F64).unwrap().is_infinite()),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn c64_div() {
        // (4+2i) / (1+1i) = (4+2i)(1-1i) / (1+1) = (6-2i)/2 = 3-1i
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(4.0, 2.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 1.0),
            Complex(F64),
        ));
        let r = BinOp::Div.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert_eq!(c.re, 3.0);
                assert_eq!(c.im, -1.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn bit_div_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(BinOp::Div.apply(a, b).is_err());
    }

    #[test]
    fn duration_div_returns_float() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        let r = BinOp::Div.apply(dur(10.0), dur(2.0)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), 5.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn angle_div_returns_uint() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(0b1000_u128 << 124)),
            Angle(iw(8)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(0b0010_u128 << 124)),
            Angle(iw(8)),
        ));
        let r = BinOp::Div.apply(a, b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.ty(), Uint(iw(8)));
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 4);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Type promotion ---

    #[test]
    fn uint8_div_uint16_promotes() {
        let r = BinOp::Div.apply(u_scalar(300, 16), u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 100);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_div_float_promotes() {
        let r = BinOp::Div.apply(u_scalar(3, 8), f64_scalar(2.0)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 1.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Array / Scalar ---

    #[test]
    fn array_div_scalar() {
        let a = u_array(&[10, 20, 30], 8, vec![3]);
        let s = u_scalar(5, 8);
        let r = BinOp::Div.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 4, 6]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_div_scalar_with_promotion() {
        let a = u_array(&[1, 2, 4], 8, vec![3]);
        let s = f64_scalar(2.0);
        let r = BinOp::Div.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                assert!(matches!(a.ty().element_ty(), Float(F64)));
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![0.5, 1.0, 2.0]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Array combinations ---

    #[test]
    fn array_div_array() {
        let a = u_array(&[10, 20, 30], 8, vec![3]);
        let b = u_array(&[2, 5, 10], 8, vec![3]);
        let r = BinOp::Div.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![5, 4, 3]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_div_array() {
        let s = u_scalar(100, 8);
        let a = u_array(&[2, 5, 10], 8, vec![3]);
        let r = BinOp::Div.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![50, 20, 10]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_div_array() {
        let a = into_ref(u_array(&[8, 16, 50, 100], 8, vec![2, 2]))
            .get(&[Index::Item(1)])
            .unwrap();
        let b = u_array(&[2, 5], 8, vec![2]);
        let r = BinOp::Div.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![25, 20]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_div_array_ref() {
        let a = into_ref(u_array(&[2, 4, 8], 8, vec![3]))
            .get(&[Index::Select(vec![1, 0])])
            .unwrap();
        let r = BinOp::Div.apply(u_scalar(64, 8), a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![16, 32]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod modulo {

    use crate::duration::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    // --- Scalar % Scalar ---

    #[test]
    fn uint_rem() {
        let r = BinOp::Mod.apply(u_scalar(10, 8), u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(iw(8)).unwrap(), 1),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_rem_by_zero_returns_none() {
        assert!(BinOp::Mod.apply(u_scalar(10, 8), u_scalar(0, 8)).is_err());
    }

    #[test]
    fn int_rem() {
        // -7 % 3 = -1 (Rust truncated remainder)
        let r = BinOp::Mod.apply(i_scalar(-7, 8), i_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(iw(8)).unwrap(), -1),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_rem_by_zero_returns_none() {
        assert!(BinOp::Mod.apply(i_scalar(10, 8), i_scalar(0, 8)).is_err());
    }

    #[test]
    fn f64_rem() {
        let r = BinOp::Mod.apply(f64_scalar(7.5), f64_scalar(2.0)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 1.5),
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn complex_rem_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(4.0, 2.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 1.0),
            Complex(F64),
        ));
        assert!(BinOp::Mod.apply(a, b).is_err());
    }

    #[test]
    fn bit_rem_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(BinOp::Mod.apply(a, b).is_err());
    }

    #[test]
    fn duration_rem_returns_none() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        assert!(BinOp::Mod.apply(dur(10.0), dur(3.0)).is_err());
    }

    #[test]
    fn angle_rem_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(8_u128), Angle(iw(8))));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::uint(3_u128), Angle(iw(8))));
        assert!(BinOp::Mod.apply(a, b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_rem_uint16_promotes() {
        let r = BinOp::Mod.apply(u_scalar(300, 16), u_scalar(7, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 300 % 7);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn incompatible_types_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = u_scalar(1, 8);
        assert!(BinOp::Mod.apply(a, b).is_err());
    }

    // --- Array % Scalar ---

    #[test]
    fn array_rem_scalar() {
        let a = u_array(&[10, 23, 37], 8, vec![3]);
        let s = u_scalar(7, 8);
        let r = BinOp::Mod.apply(a, s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![3, 2, 2]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Array combinations ---

    #[test]
    fn array_rem_array() {
        let a = u_array(&[10, 23, 37], 8, vec![3]);
        let b = u_array(&[3, 5, 10], 8, vec![3]);
        let r = BinOp::Mod.apply(a, b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![1, 3, 7]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_rem_array() {
        let s = u_scalar(100, 8);
        let a = u_array(&[3, 7, 11], 8, vec![3]);
        let r = BinOp::Mod.apply(s, a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(iw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![1, 2, 1]);
            }
            _ => panic!("expected array"),
        }
    }
}

mod pow {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn uint_pow() {
        let r = BinOp::Pow.apply(u_scalar(3, 8), u_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(width) if width.get() == 8));
                assert_eq!(s.value().as_uint(iw(8)).unwrap(), 81);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_pow_wraps() {
        // 16 ** 2 = 256 = 0 (mod 256); 3 ** 5 = 243; 3 ** 6 = 729 = 217.
        let pow = |a, b| {
            BinOp::Pow
                .apply(u_scalar(a, 8), u_scalar(b, 8))
                .unwrap()
                .to_string()
        };
        assert_eq!(pow(16, 2), "0");
        assert_eq!(pow(3, 5), "243");
        assert_eq!(pow(3, 6), "217");
        assert_eq!(pow(7, 0), "1");
    }

    #[test]
    fn uint_pow_takes_exponents_beyond_32_bits() {
        // 3 has odd order mod 2^128: 3 ** 2^127 = 1 there (and so mod 2^64).
        let r = BinOp::Pow
            .apply(u_scalar(3, 64), u_scalar(1 << 127, 128))
            .unwrap();
        assert_eq!(r.to_string(), "1");
    }

    #[test]
    fn int_pow() {
        let r = BinOp::Pow.apply(i_scalar(-3, 8), i_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(iw(8)).unwrap(), -27);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_negative_exponent_returns_zero() {
        let r = BinOp::Pow.apply(i_scalar(3, 8), i_scalar(-1, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(iw(8)).unwrap(), 0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_uint16_pow_promotes() {
        let r = BinOp::Pow.apply(u_scalar(3, 8), u_scalar(4, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(width) if width.get() == 16));
                assert_eq!(s.value().as_uint(iw(16)).unwrap(), 81);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_pow_same_width() {
        assert!(
            BinOp::Pow
                .apply(
                    Value::Scalar(Scalar::new_unchecked(Primitive::bitreg_u128(3), BitReg(8))),
                    Value::Scalar(Scalar::new_unchecked(Primitive::bitreg_u128(4), BitReg(8),))
                )
                .is_err()
        );
    }

    #[test]
    fn float_pow() {
        let r = BinOp::Pow
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(9.0),
                    Float(FloatWidth::F64),
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(0.5),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), 3.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_float_pow_promotes() {
        let r = BinOp::Pow
            .apply(
                i_scalar(9, 8),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::float(0.5),
                    Float(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), 3.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_pow() {
        let r = BinOp::Pow
            .apply(
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::complex(2.0, 0.0),
                    Complex(FloatWidth::F64),
                )),
                Value::Scalar(Scalar::new_unchecked(
                    Primitive::complex(3.0, 0.0),
                    Complex(FloatWidth::F64),
                )),
            )
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(FloatWidth::F64)));
                let c = s.value().as_complex(FloatWidth::F64).unwrap();
                assert!((c.re - 8.0).abs() < 1e-12);
                assert!(c.im.abs() < 1e-12);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_pow_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(2.0, crate::DurationUnit::Ns),
            Duration,
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(3.0, crate::DurationUnit::Ns),
            Duration,
        ));
        assert!(BinOp::Pow.apply(a, b).is_err());
    }
}

mod neg {
    use crate::DurationUnit;
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn int_neg() {
        let r = UnOp::Neg.apply(i_scalar(5, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(iw(8)).unwrap(), -5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_min_neg_wraps() {
        let r = UnOp::Neg.apply(i_scalar(-128, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(iw(8)).unwrap(), -128);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn float_neg() {
        let r = UnOp::Neg.apply(f64_scalar(1.5)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), -1.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_neg() {
        let r = UnOp::Neg
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::complex(1.0, -2.0),
                Complex(FloatWidth::F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(FloatWidth::F64)));
                let c = s.value().as_complex(FloatWidth::F64).unwrap();
                assert_eq!(c.re, -1.0);
                assert_eq!(c.im, 2.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_neg() {
        let r = UnOp::Neg
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(100.0, DurationUnit::Ns),
                Duration,
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Duration));
                let d = s.value().as_duration().unwrap();
                assert_eq!(d.value, -100.0);
                assert!(matches!(d.unit, DurationUnit::Ns));
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn angle_neg_wraps() {
        let r = UnOp::Neg
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::Angle(turns::Angle(0b0010 << 124)),
                Angle(iw(4)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Angle(width) if width.get() == 4));
                assert_eq!(s.value().as_angle(iw(4)).unwrap(), 0b1110 << 124);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_neg_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(UnOp::Neg.apply(a).is_err());
    }

    #[test]
    fn uint_neg_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Uint(iw(8))));
        assert!(UnOp::Neg.apply(a).is_err());
    }
}

mod angle {
    use crate::Error;
    use crate::ops::tests::util::*;
    use crate::primitive::iw;
    use crate::scalar::Scalar;

    fn angle4(bits: u128) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(bits << 124)),
            PrimitiveTy::Angle(iw(4)),
        ))
    }

    /// The worked examples of docs/classical.rst, "Angles".
    #[test]
    fn spec_examples() {
        let a = || angle4(0b0111); // 7 * (pi / 8)
        let b = || angle4(0b0001); // pi / 8
        let c = || angle4(0b1010); // 5 * (pi / 4)
        let two = || u_scalar(2, 4);
        let show = |op: BinOp, l: Value, r: Value| op.apply(l, r).unwrap().to_string();

        assert_eq!(show(BinOp::Add, a(), b()), "π"); // "1000"
        assert_eq!(show(BinOp::Sub, b(), a()), "(5*π/4)"); // "1010"
        assert_eq!(show(BinOp::Div, a(), two()), "(3*π/8)"); // "0011"
        assert_eq!(show(BinOp::Mul, two(), c()), "(π/2)"); // "0100"
        assert_eq!(show(BinOp::Div, c(), b()), "10"); // uint[4] "1010"
    }

    #[test]
    fn dividing_an_angle_by_zero_is_an_error_not_a_panic() {
        let zero_angle = angle4(0);
        let zero = u_scalar(0, 4);
        assert!(matches!(
            BinOp::Div.apply(angle4(0b0111), zero),
            Err(Error::DivideByZero)
        ));
        assert!(matches!(
            BinOp::Div.apply(angle4(0b0111), zero_angle),
            Err(Error::DivideByZero)
        ));
    }
}
