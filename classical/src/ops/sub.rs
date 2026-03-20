use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Sub;

impl BinOp for Sub {
    const NAME: &'static str = "-";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        Ok((ty, ty, ty))
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        let result = match (lhs.value(), rhs.value()) {
            (Uint(lhs), Uint(rhs)) => Uint(lhs.checked_sub(rhs).ok_or(Error::Overflow)?),
            (Int(lhs), Int(rhs)) => Int(lhs.checked_sub(rhs).ok_or(Error::Overflow)?),
            (Float(lhs), Float(rhs)) => Float(lhs - rhs),
            (Complex(lhs), Complex(rhs)) => Complex(lhs - rhs),
            (Duration(lhs), Duration(rhs)) => Duration(lhs - rhs),
            (Angle(lhs), Angle(rhs)) => Angle(lhs.wrapping_sub(rhs)),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.assert_fits(out)?, out))
    }
}

impl Value {
    pub fn sub_(self, rhs: Self) -> Result<Self> {
        Sub::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Array, ArrayTy, ashape};
    use crate::duration::DurationUnit;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    fn f64_scalar(v: f64) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::float(v), Float(F64)))
    }

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::uint(v)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    // --- Scalar - Scalar ---

    #[test]
    fn uint_sub() {
        let r = u_scalar(10, 8).sub_(u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 7),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_underflow_returns_none() {
        assert!(u_scalar(3, 8).sub_(u_scalar(10, 8)).is_err());
    }

    #[test]
    fn int_sub() {
        // 5 - (-3) = 8 in Int(8)
        let r = i_scalar(5, 8).sub_(i_scalar(-3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), 8),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_overflow_returns_none() {
        // 100 - (-100) = 200 overflows Int(8) (max 127)
        assert!(i_scalar(100, 8).sub_(i_scalar(-100, 8)).is_err());
    }

    #[test]
    fn int_negative_overflow_returns_none() {
        // -100 - 100 = -200 overflows Int(8) (min -128)
        assert!(i_scalar(-100, 8).sub_(i_scalar(100, 8)).is_err());
    }

    #[test]
    fn f64_sub() {
        let r = f64_scalar(3.75).sub_(f64_scalar(1.5)).unwrap();
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
        let r = a.sub_(b).unwrap();
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
        let r = dur(300.0).sub_(dur(100.0)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_duration().unwrap().value, 200.0),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_sub_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(a.sub_(b).is_err());
    }

    #[test]
    fn bitreg_sub_err() {
        // BitReg is cast to Uint, so subtraction works
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(4)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b0101_u128),
            BitReg(bw(4)),
        ));
        assert!(a.sub_(b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_sub_uint16_promotes() {
        let r = u_scalar(300, 16).sub_(u_scalar(100, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 200);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_sub_float_promotes() {
        let r = u_scalar(3, 8).sub_(f64_scalar(0.5)).unwrap();
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
        assert!(a.sub_(b).is_err());
    }

    // --- Array - Array ---

    #[test]
    fn array_sub_elementwise() {
        let a = u_array(&[10, 20, 30], 8, vec![3]);
        let b = u_array(&[1, 2, 3], 8, vec![3]);
        let r = a.sub_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
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
        assert!(a.sub_(b).is_err());
    }

    #[test]
    fn array_element_underflow_returns_none() {
        let a = u_array(&[10, 0, 30], 8, vec![3]);
        let b = u_array(&[1, 1, 1], 8, vec![3]);
        assert!(a.sub_(b).is_err());
    }

    // --- Scalar / Array mixing ---

    #[test]
    fn scalar_sub_array() {
        let s = u_scalar(100, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = s.sub_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
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
        let r = a.sub_(s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![5, 15, 25]);
            }
            _ => panic!("expected array"),
        }
    }
}
