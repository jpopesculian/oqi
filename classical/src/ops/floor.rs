use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Floor;

impl UnOp for Floor {
    const NAME: &'static str = "floor";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        match arg {
            PrimitiveTy::Float(w) => Ok((PrimitiveTy::Float(w), PrimitiveTy::Float(w))),
            _ => Err(unsupported_scalar_unop::<Self>(arg)),
        }
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        match arg.value() {
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.floor()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn floor_(self) -> Result<Self> {
        Floor::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*};
    use crate::scalar::Scalar;

    #[test]
    fn floor_rounds_down() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(2.7), Float(F64)))
            .floor_()
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
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(-2.3), Float(F64)))
            .floor_()
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
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(5.0), Float(F64)))
            .floor_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_float(F64).unwrap(), 5.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn floor_uint_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(5_u128),
                Uint(crate::primitive::bw(8)),
            ))
            .floor_()
            .is_err()
        );
    }
}
