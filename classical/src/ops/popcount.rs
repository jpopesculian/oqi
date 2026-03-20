use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::PrimitiveTy;
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Popcount;

impl UnOp for Popcount {
    const NAME: &'static str = "popcount";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        match arg {
            PrimitiveTy::BitReg(n) => Ok((PrimitiveTy::BitReg(n), PrimitiveTy::Uint(n))),
            _ => Err(unsupported_scalar_unop::<Self>(arg)),
        }
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        match arg.value() {
            // BitReg-to-BitReg cast produces Scalar::Uint due to cast impl
            Primitive::BitReg(v) | Primitive::Uint(v) => Scalar::new(
                Primitive::Uint(v.count_ones() as u128).assert_fits(out)?,
                out,
            ),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn popcount_(self) -> Result<Self> {
        Popcount::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    fn bitreg(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(v),
            BitReg(bw(bits)),
        ))
    }

    #[test]
    fn popcount_all_zeros() {
        let r = bitreg(0b0000_0000, 8).popcount_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 8));
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_all_ones() {
        let r = bitreg(0xFF, 8).popcount_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 8);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_mixed() {
        let r = bitreg(0b1010_1010, 8).popcount_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert_eq!(s.value().as_uint(bw(8)).unwrap(), 4);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_single_bit() {
        let r = bitreg(0b1, 4).popcount_().unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(w) if w.get() == 4));
                assert_eq!(s.value().as_uint(bw(4)).unwrap(), 1);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn popcount_uint_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::uint(0xFF_u128),
                Uint(bw(8)),
            ))
            .popcount_()
            .is_err()
        );
    }

    #[test]
    fn popcount_float_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::float(1.0),
                Float(crate::primitive::FloatWidth::F64),
            ))
            .popcount_()
            .is_err()
        );
    }
}
