use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Arcsin;

impl UnOp for Arcsin {
    const NAME: &'static str = "arcsin";
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
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.asin()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn arcsin_(self) -> Result<Self> {
        Arcsin::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn arcsin_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .arcsin_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arcsin_one() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)))
            .arcsin_()
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
    fn arcsin_int_promoted() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::int(0), Int(bw(8))))
            .arcsin_()
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
    fn arcsin_duration_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(1.0, crate::DurationUnit::Ns),
                Duration,
            ))
            .arcsin_()
            .is_err()
        );
    }
}
