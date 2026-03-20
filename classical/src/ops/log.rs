use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Log;

impl UnOp for Log {
    const NAME: &'static str = "log";
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
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.ln()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn log_(self) -> Result<Self> {
        Log::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn log_one() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)))
            .log_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_e() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(core::f64::consts::E),
            Float(F64),
        ))
        .log_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_negative_is_nan() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(-1.0), Float(F64)))
            .log_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().is_nan());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn log_uint_promoted() {
        // log(1) = 0
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Uint(bw(8))))
            .log_()
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
    fn log_complex_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::complex(1.0, 0.0),
                Complex(F64),
            ))
            .log_()
            .is_err()
        );
    }
}
