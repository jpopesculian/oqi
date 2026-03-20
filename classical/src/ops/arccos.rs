use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Arccos;

impl UnOp for Arccos {
    const NAME: &'static str = "arccos";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        Ok(match arg {
            PrimitiveTy::Float(w) => (PrimitiveTy::Float(w), PrimitiveTy::Float(w)),
            PrimitiveTy::Int(_) | PrimitiveTy::Uint(_) => (
                PrimitiveTy::Float(FloatWidth::F64),
                PrimitiveTy::Float(FloatWidth::F64),
            ),
            _ => return Err(unsupported_scalar_unop::<Self>(arg)),
        })
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        match arg.value() {
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.acos()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn arccos_(self) -> Result<Self> {
        Arccos::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn arccos_one() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)))
            .arccos_()
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
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .arccos_()
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
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::int(1), Int(bw(8))))
            .arccos_()
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
            Value::Scalar(Scalar::new_unchecked(Primitive::Angle(0), Angle(bw(8))))
                .arccos_()
                .is_err()
        );
    }
}
