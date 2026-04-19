use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Cos;

impl UnOp for Cos {
    const NAME: &'static str = "cos";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        Ok(match arg {
            PrimitiveTy::Float(w) => (PrimitiveTy::Float(w), PrimitiveTy::Float(w)),
            PrimitiveTy::Angle(_) | PrimitiveTy::Int(_) | PrimitiveTy::Uint(_) => (
                PrimitiveTy::Float(FloatWidth::F64),
                PrimitiveTy::Float(FloatWidth::F64),
            ),
            _ => return Err(unsupported_scalar_unop::<Self>(arg)),
        })
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        match arg.value() {
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.cos()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn cos_(self) -> Result<Self> {
        Cos::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn cos_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .cos_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_pi() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(core::f64::consts::PI),
            Float(F64),
        ))
        .cos_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() + 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_angle_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(0)),
            Angle(bw(8)),
        ))
        .cos_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_uint_promoted() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0_u128), Uint(bw(8))))
            .cos_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn cos_duration_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(1.0, crate::DurationUnit::Ns),
                Duration,
            ))
            .cos_()
            .is_err()
        );
    }
}
