use super::{BinOp, Eq, Gt, LogOr};
use crate::Result;
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;

#[cfg(test)]
use crate::Primitive;

pub struct Gte;

impl BinOp for Gte {
    const NAME: &'static str = ">=";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        Gt::scalar_check(lht, rht)
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        LogOr::scalar_op(
            Gt::scalar_op(lhs, rhs, out)?,
            Eq::scalar_op(lhs, rhs, out)?,
            out,
        )
    }
}

impl Value {
    pub fn gte_(self, rhs: Self) -> Result<Self> {
        Gte::checked_op(self, rhs)
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

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    #[test]
    fn bool_gte_via_eq() {
        assert!(bool_scalar(false).gte_(bool_scalar(false)).is_err())
    }

    #[test]
    fn int_float_gte_via_gt() {
        let r = i_scalar(4, 8)
            .gte_(Value::Scalar(Scalar::new_unchecked(
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
    fn duration_gte_via_eq() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1_000.0, DurationUnit::Ns),
            Duration,
        ))
        .gte_(Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Us),
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
    fn complex_gte_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(2.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(FloatWidth::F64),
        ));
        assert!(a.gte_(b).is_err());
    }
}
