use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Pow;

impl BinOp for Pow {
    const NAME: &'static str = "**";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        Ok((ty, ty, ty))
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match (lhs.value(), rhs.value()) {
            (Primitive::Uint(a), Primitive::Uint(b)) => Primitive::Uint(
                a.checked_pow(b.try_into().map_err(|_| Error::Overflow)?)
                    .ok_or(Error::Overflow)?,
            ),
            (Primitive::Int(a), Primitive::Int(b)) => {
                if b < 0 {
                    Primitive::Int(0)
                } else {
                    Primitive::Int(
                        a.checked_pow(b.try_into().map_err(|_| Error::Overflow)?)
                            .ok_or(Error::Overflow)?,
                    )
                }
            }
            (Primitive::Float(a), Primitive::Float(b)) => Primitive::Float(a.powf(b)),
            (Primitive::Complex(a), Primitive::Complex(b)) => Primitive::Complex(a.powc(b)),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.assert_fits(out)?, out))
    }
}

impl Value {
    pub fn pow_(self, rhs: Self) -> Result<Self> {
        Pow::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    #[test]
    fn uint_pow() {
        let r = u_scalar(3, 8).pow_(u_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(width) if width.get() == 8));
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 81);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_pow_overflow_returns_none() {
        assert!(u_scalar(16, 8).pow_(u_scalar(2, 8)).is_err());
    }

    #[test]
    fn int_pow() {
        let r = i_scalar(-3, 8).pow_(i_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(bw(8)).unwrap(), -27);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_negative_exponent_returns_zero() {
        let r = i_scalar(3, 8).pow_(i_scalar(-1, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(bw(8)).unwrap(), 0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_uint16_pow_promotes() {
        let r = u_scalar(3, 8).pow_(u_scalar(4, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(width) if width.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 81);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_pow_same_width() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(Primitive::BitReg(3), BitReg(bw(8))))
                .pow_(Value::Scalar(Scalar::new_unchecked(
                    Primitive::BitReg(4),
                    BitReg(bw(8)),
                )))
                .is_err()
        );
    }

    #[test]
    fn float_pow() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(9.0),
            Float(FloatWidth::F64),
        ))
        .pow_(Value::Scalar(Scalar::new_unchecked(
            Primitive::float(0.5),
            Float(FloatWidth::F64),
        )))
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), 3.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_float_pow_promotes() {
        let r = i_scalar(9, 8)
            .pow_(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(0.5),
                Float(FloatWidth::F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(FloatWidth::F64)));
                assert_eq!(s.value().as_float(FloatWidth::F64).unwrap(), 3.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn complex_pow() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(2.0, 0.0),
            Complex(FloatWidth::F64),
        ))
        .pow_(Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(3.0, 0.0),
            Complex(FloatWidth::F64),
        )))
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Complex(FloatWidth::F64)));
                let c = s.value().as_complex(FloatWidth::F64).unwrap();
                assert!((c.re - 8.0).abs() < 1e-12);
                assert!(c.im.abs() < 1e-12);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_pow_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(2.0, crate::DurationUnit::Ns),
            Duration,
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(3.0, crate::DurationUnit::Ns),
            Duration,
        ));
        assert!(a.pow_(b).is_err());
    }
}
