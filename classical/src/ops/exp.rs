use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};
use num_complex::c64;

pub struct Exp;

impl UnOp for Exp {
    const NAME: &'static str = "exp";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        Ok(match arg {
            PrimitiveTy::Float(w) => (PrimitiveTy::Float(w), PrimitiveTy::Float(w)),
            PrimitiveTy::Complex(w) => (PrimitiveTy::Complex(w), PrimitiveTy::Complex(w)),
            PrimitiveTy::Int(_) | PrimitiveTy::Uint(_) => (
                PrimitiveTy::Float(FloatWidth::F64),
                PrimitiveTy::Float(FloatWidth::F64),
            ),
            _ => return Err(unsupported_scalar_unop::<Self>(arg)),
        })
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match arg.value() {
            Primitive::Float(v) => Primitive::Float(v.exp()),
            Primitive::Complex(c) => {
                let r = c.re.exp();
                Primitive::Complex(c64(r * c.im.cos(), r * c.im.sin()))
            }
            _ => return Err(unsupported_scalar_unop::<Self>(arg.ty())),
        };
        Scalar::new(result, out)
    }
}

impl Value {
    pub fn exp_(self) -> Result<Self> {
        Exp::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn exp_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .exp_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_one() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)))
            .exp_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - core::f64::consts::E).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_complex_pure_imaginary() {
        // e^(iπ) = -1 + 0i
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(0.0, core::f64::consts::PI),
            Complex(F64),
        ))
        .exp_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(F64)));
                let c = s.value().as_complex(F64).unwrap();
                assert!((c.re + 1.0).abs() < 1e-10);
                assert!(c.im.abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn exp_int_promoted() {
        // exp(0) = 1
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::int(0), Int(bw(8))))
            .exp_()
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
    fn exp_duration_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(1.0, crate::DurationUnit::Ns),
                Duration,
            ))
            .exp_()
            .is_err()
        );
    }
}
