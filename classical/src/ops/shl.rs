use super::Shr;
use super::{BinOp, unsupported_scalar_binop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Shl;

impl BinOp for Shl {
    const NAME: &'static str = "<<";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        use PrimitiveTy::*;
        if matches!((lht, rht), (Uint(_) | Int(_) | BitReg(_), Uint(_) | Int(_))) {
            Ok((lht, rht, lht))
        } else {
            Err(unsupported_scalar_binop::<Self>(lht, rht))
        }
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        if let Int(a) = rhs.value()
            && a < 0
        {
            return Shr::scalar_op(lhs, Scalar::new_unchecked(Int(-a), rhs.ty()), out);
        }
        let result = match (lhs.value(), rhs.value()) {
            (BitReg(a), Uint(b)) => BitReg(a << b),
            (Uint(a), Uint(b)) => Uint(a << b),
            (Int(a), Uint(b)) => Int(a << b),
            (BitReg(a), Int(b)) => BitReg(a << b),
            (Uint(a), Int(b)) => Uint(a << b),
            (Int(a), Int(b)) => Int(a << b),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Ok(Scalar::new_unchecked(result.resize(out), out))
    }
}

impl Value {
    pub fn shl_(self, rhs: Self) -> Result<Self> {
        Shl::checked_op(self, rhs)
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

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::uint(v)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    // --- Scalar << Scalar ---

    #[test]
    fn uint_shl() {
        let r = u_scalar(1, 8).shl_(u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 8),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_shl_overflow_returns_truncated() {
        assert!(u_scalar(128, 8).shl_(u_scalar(1, 8)).is_ok());
    }

    #[test]
    fn int_shl() {
        let r = i_scalar(1, 8).shl_(u_scalar(3, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), 8),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_shl_overflow_returns_truncated() {
        assert!(i_scalar(64, 8).shl_(u_scalar(1, 8)).is_ok());
    }

    // --- Invalid types ---

    #[test]
    fn float_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(a.shl_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn complex_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 0.0),
            Complex(F64),
        ));
        assert!(a.shl_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn bit_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(a.shl_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn duration_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::duration(1.0, DurationUnit::Ns),
            Duration,
        ));
        assert!(a.shl_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn angle_shl_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Angle(bw(8))));
        assert!(a.shl_(u_scalar(1, 8)).is_err());
    }

    #[test]
    fn float_rhs_returns_none() {
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::float(1.0), Float(F64)));
        assert!(u_scalar(1, 8).shl_(b).is_err());
    }

    // --- Array << Scalar ---

    #[test]
    fn array_shl_scalar() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let s = u_scalar(2, 8);
        let r = a.shl_(s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![4, 8, 12]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Array combinations ---

    #[test]
    fn array_shl_array() {
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let b = u_array(&[1, 2, 3], 8, vec![3]);
        let r = a.shl_(b).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 8, 24]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_shl_array() {
        let s = u_scalar(1, 8);
        let a = u_array(&[1, 2, 3], 8, vec![3]);
        let r = s.shl_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 4, 8]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_ref_shl_scalar() {
        let a = u_array(&[1, 2, 3], 8, vec![3])
            .get(&[Index::Select(vec![2, 0])])
            .unwrap();
        let r = a.shl_(u_scalar(1, 8)).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![6, 2]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_shl_array_ref() {
        let a = u_array(&[1, 2, 3], 8, vec![3])
            .get(&[Index::Slice {
                start: Some(0),
                step: Some(2),
                end: Some(3),
            }])
            .unwrap();
        let r = u_scalar(1, 8).shl_(a).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<u128> = a
                    .values()
                    .iter()
                    .map(|s| s.as_uint(bw(8)).unwrap())
                    .collect();
                assert_eq!(vals, vec![2, 8]);
            }
            _ => panic!("expected array"),
        }
    }
}
