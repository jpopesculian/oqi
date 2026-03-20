use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{FloatWidth, PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Div;

impl BinOp for Div {
    const NAME: &'static str = "/";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        use PrimitiveTy::*;
        match (lht, rht) {
            (Duration, Duration) => Ok((lht, rht, Float(FloatWidth::F64))),
            (Duration, Float(_)) | (Duration, Uint(_)) | (Duration, Int(_)) => {
                Ok((lht, Float(FloatWidth::F64), Duration))
            }
            (Angle(a), Angle(b)) => {
                let max = a.max(b);
                Ok((Angle(max), Angle(max), Uint(max)))
            }
            (Angle(a), Uint(b)) | (Angle(a), Int(b)) => {
                let max = a.max(b);
                Ok((Angle(max), Uint(max), Angle(max)))
            }
            _ => {
                let ty = promote_arithmetic(lht, rht)
                    .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
                if matches!(
                    ty,
                    BitReg(_) | Uint(_) | Int(_) | Float(_) | Complex(_) | Duration
                ) {
                    Ok((ty, ty, ty))
                } else {
                    Err(unsupported_scalar_binop::<Self>(lht, rht))
                }
            }
        }
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        let result = match (lhs.value(), rhs.value()) {
            (Uint(lhs), Uint(rhs)) => Uint(lhs.checked_div(rhs).ok_or(Error::DivideByZero)?),
            (Int(lhs), Int(rhs)) => Int(lhs.checked_div(rhs).ok_or(if rhs == 0 {
                Error::DivideByZero
            } else {
                Error::Overflow
            })?),
            (Float(lhs), Float(rhs)) => Float(lhs / rhs),
            (Complex(lhs), Complex(rhs)) => Complex(lhs / rhs),
            (Duration(lhs), Duration(rhs)) => Float(lhs / rhs),
            (Duration(lhs), Float(rhs)) => Duration(lhs / rhs),
            (Angle(lhs), Angle(rhs)) => Uint(lhs / rhs),
            (Angle(lhs), Uint(rhs)) => Angle(lhs / rhs),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.assert_fits(out)?, out))
    }
}

impl Value {
    pub fn div_(self, rhs: Self) -> Result<Self> {
        Div::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Array, ArrayTy, ashape};
    use crate::duration::DurationUnit;
    use crate::index::Index;
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

    // --- Scalar / Scalar ---

    #[test]
    fn uint_div() {
        let r = u_scalar(10, 8).div_(u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 3),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_div_by_zero_returns_none() {
        assert!(u_scalar(10, 8).div_(u_scalar(0, 8)).is_err());
    }

    #[test]
    fn int_div() {
        let r = i_scalar(-12, 8).div_(i_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), -3),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_div_by_zero_returns_none() {
        assert!(i_scalar(10, 8).div_(i_scalar(0, 8)).is_err());
    }

    #[test]
    fn int_min_div_neg1_returns_none() {
        // -128 / -1 = 128 overflows Int(8)
        assert!(i_scalar(-128, 8).div_(i_scalar(-1, 8)).is_err());
    }

    #[test]
    fn f64_div() {
        let r = f64_scalar(7.5).div_(f64_scalar(2.5)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 3.0),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn f64_div_by_zero_is_inf() {
        let r = f64_scalar(1.0).div_(f64_scalar(0.0)).unwrap();
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
        let r = a.div_(b).unwrap();
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
        assert!(a.div_(b).is_err());
    }

    #[test]
    fn duration_div_returns_float() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        let r = dur(10.0).div_(dur(2.0)).unwrap();
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
            Primitive::Angle(0b1000_u128 << 124),
            Angle(bw(8)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(0b0010_u128 << 124),
            Angle(bw(8)),
        ));
        let r = a.div_(b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.ty(), Uint(bw(8)));
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 4);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Type promotion ---

    #[test]
    fn uint8_div_uint16_promotes() {
        let r = u_scalar(300, 16).div_(u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 100);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_div_float_promotes() {
        let r = u_scalar(3, 8).div_(f64_scalar(2.0)).unwrap();
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
        let r = a.div_(s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
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
        let r = a.div_(s).unwrap();
        match r {
            Value::Array(a) => {
                assert!(matches!(a.ty().ty(), Float(F64)));
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
        let r = a.div_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
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
        let r = s.div_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![50, 20, 10]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_div_array() {
        let a = u_array(&[8, 16, 50, 100], 8, vec![2, 2])
            .get(&[Index::Item(1)])
            .unwrap();
        let b = u_array(&[2, 5], 8, vec![2]);
        let r = a.div_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![25, 20]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_div_array_ref() {
        let a = u_array(&[2, 4, 8], 8, vec![3])
            .get(&[Index::Select(vec![1, 0])])
            .unwrap();
        let r = u_scalar(64, 8).div_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![16, 32]);
            }
            _ => panic!("expected array"),
        }
    }
}
