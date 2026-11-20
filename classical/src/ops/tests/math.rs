//! Tests of the operators in `ops::math`, one module per operator.

mod sin {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn sin_zero() {
        let r = UnOp::Sin
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sin_pi_half() {
        let r = UnOp::Sin
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(core::f64::consts::FRAC_PI_2),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sin_angle_quarter_turn() {
        // 8-bit angle: 64/256 = quarter turn = π/2
        let r = UnOp::Sin
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::angle(core::f64::consts::FRAC_PI_2),
                Angle(iw(8)),
            )))
            .unwrap();
        println!("{:?}", Primitive::angle(core::f64::consts::FRAC_PI_2));
        println!(
            "{:?}",
            Primitive::angle(core::f64::consts::FRAC_PI_2).as_angle(iw(8))
        );
        println!("{:?}", r);
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sin_int_promotes_to_float() {
        // Integers promote to `float` implicitly, as for every float input
        // (docs/types.rst: `exp(i1)` with an `int[8]`).
        let r = UnOp::Sin.apply(i_scalar(0, 8)).unwrap();
        assert_eq!(r.ty(), ValueTy::float(FloatWidth::F64));
        assert_eq!(r.to_string(), "0");
    }

    #[test]
    fn sin_bit_returns_none() {
        assert!(
            UnOp::Sin
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::bit(true),
                    Bit
                )))
                .is_err()
        );
    }

    #[test]
    fn sin_duration_returns_none() {
        assert!(
            UnOp::Sin
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, crate::DurationUnit::Ns),
                    Duration,
                )))
                .is_err()
        );
    }
}

mod cos {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn cos_zero() {
        let r = UnOp::Cos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_pi() {
        let r = UnOp::Cos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(core::f64::consts::PI),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() + 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_angle_zero() {
        let r = UnOp::Cos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::Angle(turns::Angle(0)),
                Angle(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_uint_promoted() {
        let r = UnOp::Cos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(0_u128),
                Uint(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_duration_returns_none() {
        assert!(
            UnOp::Cos
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, crate::DurationUnit::Ns),
                    Duration,
                )))
                .is_err()
        );
    }
}

mod tan {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn tan_zero() {
        let r = UnOp::Tan
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn tan_pi_quarter() {
        let r = UnOp::Tan
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(core::f64::consts::FRAC_PI_4),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn tan_angle() {
        // 8-bit angle 0 = 0 radians, tan(0) = 0
        let r = UnOp::Tan
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::Angle(turns::Angle(0)),
                Angle(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn tan_duration_returns_none() {
        assert!(
            UnOp::Tan
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, crate::DurationUnit::Ns),
                    Duration,
                )))
                .is_err()
        );
    }
}

mod arcsin {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn arcsin_zero() {
        let r = UnOp::Arcsin
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arcsin_one() {
        let r = UnOp::Arcsin
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(
                    (s.value().as_float(F64).unwrap() - core::f64::consts::FRAC_PI_2).abs() < 1e-10
                );
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arcsin_int_promoted() {
        let r = UnOp::Arcsin
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::int(0),
                Int(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arcsin_duration_returns_none() {
        assert!(
            UnOp::Arcsin
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, crate::DurationUnit::Ns),
                    Duration,
                )))
                .is_err()
        );
    }
}

mod arccos {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn arccos_one() {
        let r = UnOp::Arccos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arccos_zero() {
        let r = UnOp::Arccos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(
                    (s.value().as_float(F64).unwrap() - core::f64::consts::FRAC_PI_2).abs() < 1e-10
                );
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arccos_int_promoted() {
        // arccos(1) = 0
        let r = UnOp::Arccos
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::int(1),
                Int(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arccos_angle_returns_none() {
        assert!(
            UnOp::Arccos
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::Angle(turns::Angle(0)),
                    Angle(iw(8))
                )))
                .is_err()
        );
    }
}

mod arctan {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn arctan_zero() {
        let r = UnOp::Arctan
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arctan_one() {
        let r = UnOp::Arctan
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(
                    (s.value().as_float(F64).unwrap() - core::f64::consts::FRAC_PI_4).abs() < 1e-10
                );
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arctan_uint_promoted() {
        let r = UnOp::Arctan
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(0_u128),
                Uint(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arctan_complex_returns_none() {
        assert!(
            UnOp::Arctan
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::complex(1.0, 0.0),
                    Complex(F64)
                )))
                .is_err()
        );
    }
}

mod exp {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn exp_zero() {
        let r = UnOp::Exp
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_one() {
        let r = UnOp::Exp
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - core::f64::consts::E).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_complex_pure_imaginary() {
        // e^(iπ) = -1 + 0i
        let r = UnOp::Exp
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::complex(0.0, core::f64::consts::PI),
                Complex(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(F64)));
                let c = s.value().as_complex(F64).unwrap();
                assert!((c.re + 1.0).abs() < 1e-10);
                assert!(c.im.abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_int_promoted() {
        // exp(0) = 1
        let r = UnOp::Exp
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::int(0),
                Int(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_duration_returns_none() {
        assert!(
            UnOp::Exp
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(1.0, crate::DurationUnit::Ns),
                    Duration,
                )))
                .is_err()
        );
    }
}

mod log {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn log_one() {
        let r = UnOp::Log
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_e() {
        let r = UnOp::Log
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(core::f64::consts::E),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_negative_is_nan() {
        let r = UnOp::Log
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(-1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().is_nan());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_uint_promoted() {
        // log(1) = 0
        let r = UnOp::Log
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(1_u128),
                Uint(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_complex_returns_none() {
        assert!(
            UnOp::Log
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::complex(1.0, 0.0),
                    Complex(F64),
                )))
                .is_err()
        );
    }
}

mod sqrt {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, iw};
    use crate::scalar::Scalar;

    #[test]
    fn sqrt_four() {
        let r = UnOp::Sqrt
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(4.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 2.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let r = UnOp::Sqrt
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(-1.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().is_nan());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_complex_neg_one() {
        // sqrt(-1+0i) = 0+1i
        let r = UnOp::Sqrt
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::complex(-1.0, 0.0),
                Complex(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(F64)));
                let c = s.value().as_complex(F64).unwrap();
                assert!(c.re.abs() < 1e-10);
                assert!((c.im - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_complex_four() {
        let r = UnOp::Sqrt
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::complex(4.0, 0.0),
                Complex(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert!((c.re - 2.0).abs() < 1e-10);
                assert!(c.im.abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_uint_promoted() {
        let r = UnOp::Sqrt
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(9_u128),
                Uint(iw(8)),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 3.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_duration_returns_none() {
        assert!(
            UnOp::Sqrt
                .apply(Value::Scalar(Scalar::new_unchecked(
                    Primitive::duration(4.0, crate::DurationUnit::Ns),
                    Duration,
                )))
                .is_err()
        );
    }
}

mod ceiling {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn ceiling_rounds_up() {
        let r = UnOp::Ceiling
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(2.3),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), 3.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn ceiling_negative() {
        let r = UnOp::Ceiling
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(-2.7),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), -2.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn ceiling_exact() {
        let r = UnOp::Ceiling
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(5.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), 5.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn ceiling_int_promotes_to_float() {
        let r = UnOp::Ceiling.apply(i_scalar(5, 8)).unwrap();
        assert_eq!(r.ty(), ValueTy::float(FloatWidth::F64));
        assert_eq!(r.to_string(), "5");
    }
}

mod floor {
    use crate::ops::tests::util::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn floor_rounds_down() {
        let r = UnOp::Floor
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(2.7),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), 2.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn floor_negative() {
        let r = UnOp::Floor
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(-2.3),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), -3.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn floor_exact() {
        let r = UnOp::Floor
            .apply(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(5.0),
                Float(F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), 5.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn floor_uint_promotes_to_float() {
        let r = UnOp::Floor.apply(u_scalar(5, 8)).unwrap();
        assert_eq!(r.ty(), ValueTy::float(FloatWidth::F64));
        assert_eq!(r.to_string(), "5");
    }
}
