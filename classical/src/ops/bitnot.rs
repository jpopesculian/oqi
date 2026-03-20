use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct BitNot;

impl UnOp for BitNot {
    const NAME: &'static str = "~";

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        Ok(match arg {
            PrimitiveTy::Bit => (PrimitiveTy::Bit, PrimitiveTy::Bit),
            PrimitiveTy::Uint(width) => (PrimitiveTy::Uint(width), PrimitiveTy::Uint(width)),
            PrimitiveTy::Int(width) => (PrimitiveTy::Int(width), PrimitiveTy::Int(width)),
            PrimitiveTy::BitReg(width) => (PrimitiveTy::BitReg(width), PrimitiveTy::BitReg(width)),
            _ => return Err(unsupported_scalar_unop::<Self>(arg)),
        })
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match arg.value() {
            Primitive::Bit(v) => Primitive::Bit(!v),
            Primitive::Uint(v) => Primitive::Uint(!v),
            Primitive::Int(v) => Primitive::Int(!v),
            Primitive::BitReg(v) => Primitive::BitReg(!v),
            _ => return Err(unsupported_scalar_unop::<Self>(arg.ty())),
        };
        Scalar::new(result.resize(out), out)
    }
}

impl Value {
    pub fn not_(self) -> Result<Self> {
        BitNot::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn bit_scalar(v: bool) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), Bit))
    }

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    #[test]
    fn bit_bitnot() {
        let r = bit_scalar(true).not_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bit));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_bitnot() {
        let r = u_scalar(0x0F_u128, 8).not_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(width) if width.get() == 8));
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 0xF0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_bitnot() {
        let r = i_scalar(0x0F_i128, 8).not_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Int(width) if width.get() == 8));
                assert_eq!(s.value().as_int(bw(8)).unwrap(), -16);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_bitnot() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b0011_u128),
            BitReg(bw(4)),
        ))
        .not_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), BitReg(width) if width.get() == 4));
                assert_eq!(s.value().as_uint(bw(4)).unwrap(), 0b1100);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bool_bitnot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bool));
        assert!(a.not_().is_err());
    }

    #[test]
    fn float_bitnot_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(a.not_().is_err());
    }
}
