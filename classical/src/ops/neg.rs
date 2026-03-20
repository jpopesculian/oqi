use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Neg;

impl UnOp for Neg {
    const NAME: &'static str = "-";

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        Ok(match arg {
            PrimitiveTy::Int(width) => (PrimitiveTy::Int(width), PrimitiveTy::Int(width)),
            PrimitiveTy::Float(width) => (PrimitiveTy::Float(width), PrimitiveTy::Float(width)),
            PrimitiveTy::Complex(width) => {
                (PrimitiveTy::Complex(width), PrimitiveTy::Complex(width))
            }
            PrimitiveTy::Duration => (PrimitiveTy::Duration, PrimitiveTy::Duration),
            PrimitiveTy::Angle(width) => (PrimitiveTy::Angle(width), PrimitiveTy::Angle(width)),
            _ => return Err(unsupported_scalar_unop::<Self>(arg)),
        })
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match arg.value() {
            Primitive::Int(v) => Primitive::Int(v.checked_neg().ok_or(Error::Overflow)?),
            Primitive::Float(v) => Primitive::Float(-v),
            Primitive::Complex(v) => Primitive::Complex(-v),
            Primitive::Duration(v) => Primitive::Duration(-v),
            Primitive::Angle(v) => Primitive::Angle(v.wrapping_neg()),
            _ => return Err(unsupported_scalar_unop::<Self>(arg.ty())),
        };
        Scalar::new(result.resize(out), out)
    }
}

impl Value {
    pub fn neg_(self) -> Result<Self> {
        Neg::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DurationUnit;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    fn f64_scalar(v: f64) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::float(v),
            Float(FloatWidth::F64),
        ))
    }

    #[test]
    fn int_neg() {
        let r = i_scalar(5, 8).neg_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(bw(8)).unwrap(), -5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_min_neg_wraps() {
        let r = i_scalar(-128, 8).neg_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(bw(8)).unwrap(), -128);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn float_neg() {
        let r = f64_scalar(1.5).neg_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), -1.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_neg() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, -2.0),
            Complex(FloatWidth::F64),
        ))
        .neg_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(FloatWidth::F64)));
                let c = s.value().as_complex(FloatWidth::F64).unwrap();
                assert_eq!(c.re, -1.0);
                assert_eq!(c.im, 2.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_neg() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(100.0, DurationUnit::Ns),
            Duration,
        ))
        .neg_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Duration));
                let d = s.value().as_duration().unwrap();
                assert_eq!(d.value, -100.0);
                assert!(matches!(d.unit, DurationUnit::Ns));
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn angle_neg_wraps() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(0b0010 << 124),
            Angle(bw(4)),
        ))
        .neg_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Angle(width) if width.get() == 4));
                assert_eq!(s.value().as_angle(bw(4)).unwrap(), 0b1110 << 124);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_neg_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(a.neg_().is_err());
    }

    #[test]
    fn uint_neg_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Uint(bw(8))));
        assert!(a.neg_().is_err());
    }
}
