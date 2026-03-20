use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Rem;

impl BinOp for Rem {
    const NAME: &'static str = "%";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        if matches!(
            ty,
            PrimitiveTy::BitReg(_)
                | PrimitiveTy::Uint(_)
                | PrimitiveTy::Int(_)
                | PrimitiveTy::Float(_)
        ) {
            Ok((ty, ty, ty))
        } else {
            Err(unsupported_scalar_binop::<Self>(lht, rht))
        }
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        let result = match (lhs.value(), rhs.value()) {
            (Uint(lhs), Uint(rhs)) => Uint(lhs.checked_rem(rhs).ok_or(Error::DivideByZero)?),
            (Int(lhs), Int(rhs)) => Int(lhs.checked_rem(rhs).ok_or(if rhs == 0 {
                Error::DivideByZero
            } else {
                Error::Overflow
            })?),
            (Float(lhs), Float(rhs)) => Float(lhs % rhs),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.assert_fits(out)?, out))
    }
}

impl Value {
    pub fn rem_(self, rhs: Self) -> Result<Self> {
        Rem::checked_op(self, rhs)
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

    // --- Scalar % Scalar ---

    #[test]
    fn uint_rem() {
        let r = u_scalar(10, 8).rem_(u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 1),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_rem_by_zero_returns_none() {
        assert!(u_scalar(10, 8).rem_(u_scalar(0, 8)).is_err());
    }

    #[test]
    fn int_rem() {
        // -7 % 3 = -1 (Rust truncated remainder)
        let r = i_scalar(-7, 8).rem_(i_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), -1),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_rem_by_zero_returns_none() {
        assert!(i_scalar(10, 8).rem_(i_scalar(0, 8)).is_err());
    }

    #[test]
    fn f64_rem() {
        let r = f64_scalar(7.5).rem_(f64_scalar(2.0)).unwrap();
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
        assert!(a.rem_(b).is_err());
    }

    #[test]
    fn bit_rem_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(a.rem_(b).is_err());
    }

    #[test]
    fn duration_rem_returns_none() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        assert!(dur(10.0).rem_(dur(3.0)).is_err());
    }

    #[test]
    fn angle_rem_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(8_u128), Angle(bw(8))));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::uint(3_u128), Angle(bw(8))));
        assert!(a.rem_(b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_rem_uint16_promotes() {
        let r = u_scalar(300, 16).rem_(u_scalar(7, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 300 % 7);
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
        assert!(a.rem_(b).is_err());
    }

    // --- Array % Scalar ---

    #[test]
    fn array_rem_scalar() {
        let a = u_array(&[10, 23, 37], 8, vec![3]);
        let s = u_scalar(7, 8);
        let r = a.rem_(s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
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
        let r = a.rem_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
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
        let r = s.rem_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![1, 2, 1]);
            }
            _ => panic!("expected array"),
        }
    }
}
