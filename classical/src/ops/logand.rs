use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct LogAnd;

impl BinOp for LogAnd {
    const NAME: &'static str = "&&";

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
            (Primitive::Bit(a), Primitive::Bit(b)) => Primitive::Bit(a && b),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result, out))
    }
}

impl Value {
    pub fn land_(self, rhs: Self) -> Result<Self> {
        LogAnd::checked_op(self, rhs)
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
    fn bool_logand() {
        let r = bool_scalar(true).land_(bool_scalar(false)).unwrap();
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
        let r = bool_scalar(true).land_(bool_scalar(true)).unwrap();
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
        assert!(a.land_(b).is_err());
    }

    #[test]
    fn bitreg_logand_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b01_u128),
            BitReg(crate::bw(2)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b11_u128),
            BitReg(crate::bw(2)),
        ));
        assert!(a.land_(b).is_err());
    }
}
