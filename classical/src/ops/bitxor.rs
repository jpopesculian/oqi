use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct BitXor;

impl BinOp for BitXor {
    const NAME: &'static str = "^";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        if let (PrimitiveTy::BitReg(lhs_width), PrimitiveTy::BitReg(rhs_width)) = (lht, rht)
            && lhs_width == rhs_width
        {
            return Ok((lht, rht, lht));
        }
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        if matches!(
            ty,
            PrimitiveTy::BitReg(_) | PrimitiveTy::Uint(_) | PrimitiveTy::Int(_)
        ) {
            Ok((ty, ty, ty))
        } else {
            Err(unsupported_scalar_binop::<Self>(lht, rht))
        }
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        let result = match (lhs.value(), rhs.value()) {
            (BitReg(a), BitReg(b)) => BitReg(a ^ b),
            (Uint(a), Uint(b)) => Uint(a ^ b),
            (Int(a), Int(b)) => Int(a ^ b),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.resize(out), out))
    }
}

impl Value {
    pub fn xor_(self, rhs: Self) -> Result<Self> {
        BitXor::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::DurationUnit;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    // --- Scalar ^ Scalar ---

    #[test]
    fn uint_bitxor() {
        // 0b1100 ^ 0b1010 = 0b0110
        let r = u_scalar(0b1100, 8).xor_(u_scalar(0b1010, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 0b0110),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_bitxor() {
        let r = i_scalar(0x0F, 8)
            .xor_(i_scalar(0xFF_u8 as i128, 8))
            .unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), 0x0F ^ -1),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(a.xor_(b).is_err());
    }

    #[test]
    fn bitreg_bitxor_same_width() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg(0b1100_u128),
            BitReg(bw(4)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg(0b1010_u128),
            BitReg(bw(4)),
        ));
        let r = a.xor_(b).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_bitreg(bw(4)).unwrap(), 0b0110),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_bitxor_uint16_promotes() {
        let r = u_scalar(0xFF, 8).xor_(u_scalar(0xFF0F, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 0x00FF ^ 0xFF0F);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn float_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(a.xor_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(F64),
        ));
        assert!(a.xor_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(a.xor_(b).is_err());
    }

    #[test]
    fn angle_bitxor_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF_u128),
            Angle(bw(8)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0x0F_u128),
            Angle(bw(8)),
        ));
        assert!(a.xor_(b).is_err());
    }
}
