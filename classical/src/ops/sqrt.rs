use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};
use num_complex::c64;

pub struct Sqrt;

impl UnOp for Sqrt {
    const NAME: &'static str = "sqrt";
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
            Primitive::Float(v) => Primitive::Float(v.sqrt()),
            Primitive::Complex(c) => {
                let r = (c.re * c.re + c.im * c.im).sqrt();
                let arg = c.im.atan2(c.re);
                let sqrt_r = r.sqrt();
                Primitive::Complex(c64(sqrt_r * (arg / 2.0).cos(), sqrt_r * (arg / 2.0).sin()))
            }
            _ => return Err(unsupported_scalar_unop::<Self>(arg.ty())),
        };
        Scalar::new(result, out)
    }
}

impl Value {
    pub fn sqrt_(self) -> Result<Self> {
        Sqrt::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn sqrt_four() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(4.0), Float(F64)))
            .sqrt_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 2.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(-1.0), Float(F64)))
            .sqrt_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(s.value().as_float(F64).unwrap().is_nan());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_complex_neg_one() {
        // sqrt(-1+0i) = 0+1i
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(-1.0, 0.0),
            Complex(F64),
        ))
        .sqrt_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(F64)));
                let c = s.value().as_complex(F64).unwrap();
                assert!(c.re.abs() < 1e-10);
                assert!((c.im - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_complex_four() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(4.0, 0.0),
            Complex(F64),
        ))
        .sqrt_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert!((c.re - 2.0).abs() < 1e-10);
                assert!(c.im.abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_uint_promoted() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::uint(9_u128), Uint(bw(8))))
            .sqrt_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 3.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sqrt_duration_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(4.0, crate::DurationUnit::Ns),
                Duration,
            ))
            .sqrt_()
            .is_err()
        );
    }
}
