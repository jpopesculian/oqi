use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Add;

impl BinOp for Add {
    const NAME: &'static str = "+";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        let ty = promote_arithmetic(lht, rht)
            .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
        if matches!(
            ty,
            PrimitiveTy::BitReg(_)
                | PrimitiveTy::Uint(_)
                | PrimitiveTy::Int(_)
                | PrimitiveTy::Float(_)
                | PrimitiveTy::Complex(_)
                | PrimitiveTy::Duration
                | PrimitiveTy::Angle(_)
        ) {
            Ok((ty, ty, ty))
        } else {
            Err(unsupported_scalar_binop::<Self>(lht, rht))
        }
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        let result = match (lhs.value(), rhs.value()) {
            (Uint(lhs), Uint(rhs)) => Uint(lhs.checked_add(rhs).ok_or(Error::Overflow)?),
            (Int(lhs), Int(rhs)) => Int(lhs.checked_add(rhs).ok_or(Error::Overflow)?),
            (Float(lhs), Float(rhs)) => Float(lhs + rhs),
            (Complex(lhs), Complex(rhs)) => Complex(lhs + rhs),
            (Duration(lhs), Duration(rhs)) => Duration(lhs + rhs),
            (Angle(lhs), Angle(rhs)) => Angle(lhs + rhs),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.assert_fits(out)?, out))
    }
}

impl Value {
    pub fn add_(self, rhs: Self) -> Result<Self> {
        Add::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Array, ArrayTy, ashape};
    use crate::duration::DurationUnit;
    use crate::index::Index;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    fn f64_scalar(v: f64) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::float(v), Float(F64)))
    }

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::uint(v)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    // --- Scalar + Scalar ---

    #[test]
    fn uint_add() {
        let r = u_scalar(3, 8).add_(u_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 7),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_overflow_returns_none() {
        assert!(u_scalar(200, 8).add_(u_scalar(100, 8)).is_err());
    }

    #[test]
    fn int_add() {
        // -3 + 5 = 2 in Int(8)
        let r = i_scalar(-3, 8).add_(i_scalar(5, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), 2),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_overflow_returns_none() {
        // 100 + 100 = 200 overflows Int(8) (max 127)
        let a = i_scalar(100, 8);
        let b = i_scalar(100, 8);
        assert!(a.add_(b).is_err());
    }

    #[test]
    fn int_negative_overflow_returns_none() {
        // -100 + -100 = -200 overflows Int(8) (min -128)
        let a = i_scalar(-100, 8);
        let b = i_scalar(-100, 8);
        assert!(a.add_(b).is_err());
    }

    #[test]
    fn f64_add() {
        let r = f64_scalar(1.5).add_(f64_scalar(2.25)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 3.75),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn c64_add() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 2.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(3.0, -1.0),
            Complex(F64),
        ));
        let r = a.add_(b).unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert_eq!(c.re, 4.0);
                assert_eq!(c.im, 1.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn duration_add() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        let r = dur(100.0).add_(dur(200.0)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_duration().unwrap().value, 300.0)
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn bit_add_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(false), Bit));
        assert!(a.add_(b).is_err());
    }

    #[test]
    fn bitreg_add_fails() {
        // BitReg is cast to Uint, so addition works
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg(0b1010u128),
            BitReg(bw(4)),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg(0b0101u128),
            BitReg(bw(4)),
        ));
        assert!(a.add_(b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_add_uint16_promotes() {
        // Uint(8) + Uint(16) promotes to Uint(16)
        let r = u_scalar(100, 8).add_(u_scalar(300, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 400);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_add_float_promotes() {
        let r = u_scalar(3, 8).add_(f64_scalar(0.5)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 3.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn incompatible_types_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        let b = u_scalar(1, 8);
        assert!(a.add_(b).is_err());
    }

    // --- Array + Array ---

    #[test]
    fn array_add_elementwise() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let b = u_array(&[10, 20, 30], 8, vec![3]);
        let r = a.add_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![11, 22, 33]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_shape_mismatch_returns_none() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let b = u_array(&[1, 2], 8, vec![2]);
        assert!(a.add_(b).is_err());
    }

    #[test]
    fn array_element_overflow_returns_none() {
        let a = u_array(&[1, 255, 3], 8, vec![3]);
        let b = u_array(&[0, 1, 0], 8, vec![3]);
        assert!(a.add_(b).is_err());
    }

    // --- Scalar / Array mixing ---

    #[test]
    fn scalar_add_array() {
        let s = u_scalar(10, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = s.add_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![11, 12, 13]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_add_scalar() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let s = u_scalar(10, 8);
        let r = a.add_(s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![11, 12, 13]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_add_array_ref() {
        let a = u_array(&[1, 2, 3, 4], 8, vec![2, 2])
            .get(&[Index::Item(0)])
            .unwrap();
        let b = u_array(&[10, 20, 30, 40], 8, vec![2, 2])
            .get(&[Index::Item(1)])
            .unwrap();
        let r = a.add_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![31, 42]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_add_array_ref() {
        let a = u_array(&[1, 2, 3], 8, vec![3])
            .get(&[Index::Select(vec![2, 0])])
            .unwrap();
        let r = u_scalar(10, 8).add_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![13, 11]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_shape_mismatch_returns_none() {
        let a = u_array(&[1, 2, 3, 4], 8, vec![2, 2])
            .get(&[Index::Item(0)])
            .unwrap();
        let b = u_array(&[10, 20, 30], 8, vec![3])
            .get(&[Index::Select(vec![0])])
            .unwrap();
        assert!(a.add_(b).is_err());
    }
}
