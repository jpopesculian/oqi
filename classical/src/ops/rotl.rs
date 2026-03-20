use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Rotl;

pub(crate) fn rotate_left(value: u128, width: u32, distance: i128) -> u128 {
    if width == 0 {
        return 0;
    }
    let d = distance.rem_euclid(width as i128) as u32;
    if d == 0 {
        return value;
    }
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    let v = value & mask;
    ((v << d) | (v >> (width - d))) & mask
}

impl BinOp for Rotl {
    const NAME: &'static str = "<<<";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        match (lht, rht) {
            (PrimitiveTy::BitReg(n), PrimitiveTy::Int(_) | PrimitiveTy::Uint(_)) => {
                Ok((PrimitiveTy::BitReg(n), rht, PrimitiveTy::BitReg(n)))
            }
            (PrimitiveTy::Uint(n), PrimitiveTy::Int(_) | PrimitiveTy::Uint(_)) => {
                Ok((PrimitiveTy::Uint(n), rht, PrimitiveTy::Uint(n)))
            }
            _ => Err(unsupported_scalar_binop::<Self>(lht, rht)),
        }
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let width = out
            .bw()
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty()))?
            .get();
        let distance = match rhs.value() {
            Primitive::Int(v) => v,
            Primitive::Uint(v) => v as i128,
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        let value = match lhs.value() {
            Primitive::BitReg(v) | Primitive::Uint(v) => v,
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        let result = rotate_left(value, width, distance);
        match out {
            PrimitiveTy::BitReg(_) => Ok(Scalar::new_unchecked(Primitive::BitReg(result), out)),
            PrimitiveTy::Uint(_) => Ok(Scalar::new_unchecked(Primitive::Uint(result), out)),
            _ => Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        }
    }
}

impl Value {
    pub fn rotl_(self, rhs: Self) -> Result<Self> {
        Rotl::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn bitreg(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(v),
            BitReg(bw(bits)),
        ))
    }

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    #[test]
    fn rotl_bitreg_by_3() {
        // 0b0010_1010 rotated left 3 = 0b0101_0001
        let r = bitreg(0b0010_1010, 8).rotl_(i_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), BitReg(w) if w.get() == 8));
                assert_eq!(s.value().as_bitreg(bw(8)).unwrap(), 0b0101_0001);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_uint_by_1() {
        // 0b1000_0001 rotated left 1 in 8 bits = 0b0000_0011
        let r = u_scalar(0b1000_0001, 8).rotl_(i_scalar(1, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 8));
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 0b0000_0011);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_by_zero() {
        let r = bitreg(0b1010, 4).rotl_(i_scalar(0, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(bw(4)).unwrap(), 0b1010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_by_width_is_identity() {
        let r = bitreg(0b1010, 4).rotl_(i_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(bw(4)).unwrap(), 0b1010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_negative_distance() {
        // rotl(a, -1) == rotr(a, 1): rotate right by 1
        // 0b0001 in 4 bits rotated left by -1 = rotated right by 1 = 0b1000
        let r = bitreg(0b0001, 4).rotl_(i_scalar(-1, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(bw(4)).unwrap(), 0b1000);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_with_uint_distance() {
        let r = bitreg(0b0001, 4).rotl_(u_scalar(1, 8)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_bitreg(bw(4)).unwrap(), 0b0010);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn rotl_float_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(FloatWidth::F64),
            ))
            .rotl_(i_scalar(1, 8))
            .is_err()
        );
    }

    #[test]
    fn rotl_int_value_returns_none() {
        assert!(i_scalar(5, 8).rotl_(i_scalar(1, 8)).is_err());
    }
}
