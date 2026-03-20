use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Tan;

impl UnOp for Tan {
    const NAME: &'static str = "tan";
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
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.tan()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn tan_(self) -> Result<Self> {
        Tan::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn tan_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .tan_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn tan_pi_quarter() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(core::f64::consts::FRAC_PI_4),
            Float(F64),
        ))
        .tan_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn tan_angle() {
        // 8-bit angle 0 = 0 radians, tan(0) = 0
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::Angle(0), Angle(bw(8))))
            .tan_()
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
    fn tan_duration_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(1.0, crate::DurationUnit::Ns),
                Duration,
            ))
            .tan_()
            .is_err()
        );
    }
}
