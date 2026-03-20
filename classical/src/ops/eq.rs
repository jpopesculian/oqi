use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Eq;

impl BinOp for Eq {
    const NAME: &'static str = "==";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        if lht == rht {
            return Ok((lht, rht, PrimitiveTy::Bool));
        }
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        Ok((ty, ty, PrimitiveTy::Bool))
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        let result = match (lhs.value(), rhs.value()) {
            (Primitive::Bit(a), Primitive::Bit(b)) => a == b,
            (Primitive::BitReg(a), Primitive::BitReg(b)) => a == b,
            (Primitive::Uint(a), Primitive::Uint(b)) => a == b,
            (Primitive::Int(a), Primitive::Int(b)) => a == b,
            (Primitive::Float(a), Primitive::Float(b)) => a == b,
            (Primitive::Complex(a), Primitive::Complex(b)) => a == b,
            (Primitive::Duration(a), Primitive::Duration(b)) => a == b,
            (Primitive::Angle(a), Primitive::Angle(b)) => a == b,
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(Primitive::bit(result), out))
    }
}

impl Value {
    pub fn eq_(self, rhs: Self) -> Result<Self> {
        Eq::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DurationUnit;
    use crate::array::{Array, ArrayTy, ashape};
    use crate::index::Index;
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

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::uint(v)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    #[test]
    fn bool_eq() {
        let r = bool_scalar(true).eq_(bool_scalar(true)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_bool_eq_promotes() {
        let r = bit_scalar(true).eq_(bool_scalar(false)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(!s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint8_uint16_eq_promotes() {
        let r = u_scalar(255, 8).eq_(u_scalar(255, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_float_eq_promotes() {
        let r = i_scalar(3, 8)
            .eq_(Value::Scalar(Scalar::new_unchecked(
                Primitive::float(3.0),
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
    fn duration_eq_uses_partial_eq() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1_000.0, DurationUnit::Ns),
            Duration,
        ))
        .eq_(Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Us),
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
    fn bitreg_eq_same_width() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(4)),
        ))
        .eq_(Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
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
    fn bitreg_eq_mismatched_width_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(4)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_u128),
            BitReg(bw(8)),
        ));
        assert!(a.eq_(b).is_err());
    }

    #[test]
    fn bit_uint_eq_is_ok() {
        let a = bit_scalar(true);
        let b = u_scalar(1, 8);
        let r = a.eq_(b).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Bool));
                assert!(s.value().as_bit());
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn array_ref_eq_array_ref() {
        let a = u_array(&[1, 2, 3, 4], 8, vec![2, 2])
            .get(&[Index::Item(0)])
            .unwrap();
        let b = u_array(&[1, 9, 3, 8], 8, vec![2, 2])
            .get(&[Index::Item(0)])
            .unwrap();
        let r = a.eq_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<bool> = a.values().iter().map(|s| s.as_bit()).collect();
                assert_eq!(vals, vec![true, false]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_eq_scalar() {
        let a = u_array(&[7, 8, 7], 8, vec![3])
            .get(&[Index::Select(vec![2, 1, 0])])
            .unwrap();
        let r = a.eq_(u_scalar(7, 8)).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<bool> = a.values().iter().map(|s| s.as_bit()).collect();
                assert_eq!(vals, vec![true, false, true]);
            }
            _ => panic!("expected array"),
        }
    }
}
