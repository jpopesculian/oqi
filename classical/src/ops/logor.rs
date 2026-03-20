use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct LogOr;

impl BinOp for LogOr {
    const NAME: &'static str = "||";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        matches!((lht, rht), (PrimitiveTy::Bool, PrimitiveTy::Bool))
            .then_some((PrimitiveTy::Bool, PrimitiveTy::Bool, PrimitiveTy::Bool))
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match (lhs.value(), rhs.value()) {
            (Primitive::Bit(a), Primitive::Bit(b)) => Primitive::Bit(a || b),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result, out))
    }
}

impl Value {
    pub fn lor_(self, rhs: Self) -> Result<Self> {
        LogOr::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::PrimitiveTy::*;
    use crate::scalar::Scalar;

    fn bool_scalar(v: bool) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), Bool))
    }

    #[test]
    fn bool_logor() {
        let r = bool_scalar(true).lor_(bool_scalar(false)).unwrap();
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
        let r = bool_scalar(false).lor_(bool_scalar(false)).unwrap();
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
        assert!(a.lor_(b).is_err());
    }

    #[test]
    fn bitreg_logor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b01_u128),
            BitReg(crate::bw(2)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b11_u128),
            BitReg(crate::bw(2)),
        ));
        assert!(a.lor_(b).is_err());
    }
}
