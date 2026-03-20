use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Arctan;

impl UnOp for Arctan {
    const NAME: &'static str = "arctan";
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
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.atan()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn arctan_(self) -> Result<Self> {
        Arctan::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn arctan_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .arctan_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arctan_one() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)))
            .arctan_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(
                    (s.value().as_float(F64).unwrap() - core::f64::consts::FRAC_PI_4).abs() < 1e-10
                );
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn arctan_uint_promoted() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::uint(0_u128), Uint(bw(8))))
            .arctan_()
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
    fn arctan_complex_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::complex(1.0, 0.0),
                Complex(F64)
            ))
            .arctan_()
            .is_err()
        );
    }
}
