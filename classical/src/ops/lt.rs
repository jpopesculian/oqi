use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Lt;

impl BinOp for Lt {
    const NAME: &'static str = "<";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        Ok((ty, ty, PrimitiveTy::Bool))
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match (lhs.value(), rhs.value()) {
            (Primitive::Uint(a), Primitive::Uint(b)) => a < b,
            (Primitive::Int(a), Primitive::Int(b)) => a < b,
            (Primitive::Float(a), Primitive::Float(b)) => a < b,
            (Primitive::Duration(a), Primitive::Duration(b)) => a < b,
            (Primitive::Angle(a), Primitive::Angle(b)) => a < b,
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(Primitive::bit(result), out))
    }
}

impl Value {
    pub fn lt_(self, rhs: Self) -> Result<Self> {
        Lt::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DurationUnit;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn bool_scalar(v: bool) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), Bool))
    }

    fn bit_scalar(v: bool) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), Bit))
    }

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    #[test]
    fn bool_lt() {
        assert!(bool_scalar(false).lt_(bool_scalar(true)).is_err());
    }

    #[test]
    fn bit_lt() {
        assert!(bit_scalar(false).lt_(bit_scalar(true)).is_err());
    }

    #[test]
    fn bitreg_lt_same_width() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::BitReg(0b0011),
                BitReg(bw(4))
            ))
            .lt_(Value::Scalar(Scalar::new_unchecked(
                Primitive::BitReg(0b0100),
                BitReg(bw(4)),
            )))
            .is_err()
        )
    }

    #[test]
    fn int_float_lt_promotes() {
        let r = i_scalar(3, 8)
            .lt_(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(3.5),
                Float(FloatWidth::F64),
            )))
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
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Us),
            Duration,
        ))
        .lt_(Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(2_000.0, DurationUnit::Ns),
            Duration,
        )))
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
        assert!(a.lt_(b).is_err());
    }

    #[test]
    fn bitreg_lt_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b0011),
            BitReg(bw(4)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b0100),
            BitReg(bw(8)),
        ));
        assert!(a.lt_(b).is_err());
    }

    #[test]
    fn bit_uint_lt_returns_none() {
        let a = bit_scalar(true);
        let b = u_scalar(0, 8);
        let r = a.lt_(b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected error"),
        }
    }
}
