use super::{BinOp, Eq, LogNot, UnOp};
use crate::Result;
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;

#[cfg(test)]
use crate::Primitive;

pub struct Neq;

impl BinOp for Neq {
    const NAME: &'static str = "!=";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        Eq::scalar_check(lht, rht)
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        LogNot::scalar_op(Eq::scalar_op(lhs, rhs, out)?, out)
    }
}

impl Value {
    pub fn neq_(self, rhs: Self) -> Result<Self> {
        Neq::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DurationUnit;
    use crate::primitive::{FloatWidth, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn bool_scalar(v: bool) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), Bool))
    }

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
    fn bool_neq() {
        let r = bool_scalar(true).neq_(bool_scalar(false)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bool_neq_promotes() {
        let r = bit_scalar(true).neq_(bool_scalar(true)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_uint16_neq_promotes() {
        let r = u_scalar(255, 8).neq_(u_scalar(0, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_float_neq_promotes() {
        let r = i_scalar(3, 8)
            .neq_(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(4.0),
                Float(FloatWidth::F64),
            )))
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_neq_uses_eq_then_lognot() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1_000.0, DurationUnit::Ns),
            Duration,
        ))
        .neq_(Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(2.0, DurationUnit::Us),
            Duration,
        )))
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_neq_same_width() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(4)),
        ))
        .neq_(Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b0101_u128),
            BitReg(bw(4)),
        )))
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bitreg_neq_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(4)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(8)),
        ));
        assert!(a.neq_(b).is_err());
    }

    #[test]
    fn bit_uint_neq_returns_none() {
        let a = bit_scalar(true);
        let b = u_scalar(0, 8);
        let r = a.neq_(b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected error"),
        }
    }
}
