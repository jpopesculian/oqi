use super::UnOp;
use crate::array::{ArrayTy, ashape};
use crate::array_ref::ArrayRefTy;
use crate::error::{Error, Result};
use crate::primitive::{BitWidth, Primitive, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::{Value, ValueTy};

pub struct Sizeof;

const SIZEOF_OUT_TY: PrimitiveTy = PrimitiveTy::Uint(BitWidth::B64);

fn sizeof_result(arg: ValueTy, dim: usize) -> Result<Value> {
    let size = arg
        .size(dim)
        .ok_or_else(|| Error::unsupported_unop(Sizeof::NAME, arg, Sizeof::IS_FUNC))?;
    Ok(Value::Scalar(Scalar::new_unchecked(
        Primitive::uint(size as u128),
        SIZEOF_OUT_TY,
    )))
}

impl UnOp for Sizeof {
    const NAME: &'static str = "sizeof";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        if ValueTy::Scalar(arg).size(0).is_some() {
            Ok((arg, SIZEOF_OUT_TY))
        } else {
            Err(Error::unsupported_unop(
                Self::NAME,
                ValueTy::Scalar(arg),
                Self::IS_FUNC,
            ))
        }
    }

    fn scalar_op(arg: Scalar, _out: PrimitiveTy) -> Result<Scalar> {
        let Value::Scalar(value) = sizeof_result(ValueTy::Scalar(arg.ty()), 0)? else {
            unreachable!("sizeof scalar result is always scalar")
        };
        Ok(value)
    }

    fn arr_check(arg: ArrayTy) -> Result<(ArrayTy, ArrayTy)> {
        if ValueTy::Array(arg).size(0).is_some() {
            Ok((arg, ArrayTy::new(SIZEOF_OUT_TY, ashape(vec![1]))))
        } else {
            Err(Error::unsupported_unop(
                Self::NAME,
                ValueTy::Array(arg),
                Self::IS_FUNC,
            ))
        }
    }

    fn arr_ref_check(arg: ArrayRefTy) -> Result<(ArrayTy, ArrayTy)> {
        if ValueTy::ArrayRef(arg).size(0).is_some() {
            Ok((
                arg.to_owned()?,
                ArrayTy::new(SIZEOF_OUT_TY, ashape(vec![1])),
            ))
        } else {
            Err(Error::unsupported_unop(
                Self::NAME,
                ValueTy::ArrayRef(arg),
                Self::IS_FUNC,
            ))
        }
    }

    fn op(arg: Value, _out: ValueTy) -> Result<Value> {
        sizeof_result(arg.ty(), 0)
    }

    fn checked_op(arg: Value) -> Result<Value> {
        let arg_ty = arg.ty();
        Self::return_ty(arg_ty)?;
        sizeof_result(arg_ty, 0)
    }

    fn return_ty(arg: ValueTy) -> Result<ValueTy> {
        if arg.size(0).is_some() {
            Ok(ValueTy::Scalar(SIZEOF_OUT_TY))
        } else {
            Err(Error::unsupported_unop(Self::NAME, arg, Self::IS_FUNC))
        }
    }
}

impl Value {
    pub fn sizeof_(self) -> Result<Self> {
        Sizeof::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::array_ref::RefAccess;
    use crate::index::Index;
    use crate::primitive::{PrimitiveTy::*, bw};

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&value| Primitive::uint(value)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    #[test]
    fn sizeof_scalar_returns_bitwidth() {
        let result = Value::Scalar(Scalar::new_unchecked(Primitive::uint(5_u128), Uint(bw(16))))
            .sizeof_()
            .unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), SIZEOF_OUT_TY);
                assert_eq!(scalar.value().as_uint(BitWidth::B64), Some(16));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_array_returns_first_dimension() {
        let result = u_array(&[1, 2, 3, 4, 5, 6], 8, vec![2, 3])
            .sizeof_()
            .unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(BitWidth::B64), Some(2));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_fixed_array_ref_returns_first_dimension() {
        let array_ref = u_array(&[1, 2, 3, 4], 8, vec![4])
            .get(&[Index::Select(vec![0, 2])])
            .unwrap();

        let result = array_ref.sizeof_().unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(BitWidth::B64), Some(2));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_rank_only_array_ref_is_rejected() {
        let array_ref_ty = ArrayRefTy::new(Uint(bw(8)), crate::ArrayRefShape::Dim(crate::adim(2)), RefAccess::Readonly);

        assert!(Sizeof::return_ty(ValueTy::ArrayRef(array_ref_ty)).is_err());
    }

    #[test]
    fn sizeof_bool_is_rejected() {
        let value = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bool));

        assert!(value.sizeof_().is_err());
    }
}
