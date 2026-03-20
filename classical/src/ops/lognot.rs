use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct LogNot;

impl UnOp for LogNot {
    const NAME: &'static str = "!";

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        matches!(arg, PrimitiveTy::Bool)
            .then_some((PrimitiveTy::Bool, PrimitiveTy::Bool))
            .ok_or_else(|| unsupported_scalar_unop::<Self>(arg))
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match arg.value() {
            Primitive::Bit(v) => Primitive::Bit(!v),
            _ => return Err(unsupported_scalar_unop::<Self>(arg.ty())),
        };
        Scalar::new(result, out)
    }
}

impl Value {
    pub fn lnot_(self) -> Result<Self> {
        LogNot::checked_op(self)
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
    fn bool_lognot_true() {
        let r = bool_scalar(true).lnot_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bool_lognot_false() {
        let r = bool_scalar(false).lnot_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_lognot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(a.lnot_().is_err());
    }

    #[test]
    fn bitreg_lognot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b01_u128),
            BitReg(crate::bw(2)),
        ));
        assert!(a.lnot_().is_err());
    }
}
