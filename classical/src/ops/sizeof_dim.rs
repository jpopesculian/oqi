use super::BinOp;
use crate::array::{ArrayTy, ashape};
use crate::array_ref::ArrayRefTy;
use crate::error::{Error, Result};
use crate::primitive::{IntWidth, Primitive, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::{Value, ValueTy};

pub struct SizeofDim;

const SIZEOF_DIM_TY: PrimitiveTy = PrimitiveTy::Int(IntWidth::B64);
const SIZEOF_OUT_TY: PrimitiveTy = PrimitiveTy::Uint(IntWidth::B64);

fn sizeof_result(lhs: ValueTy, dim: usize, rhs: ValueTy) -> Result<Value> {
    let size = lhs
        .size(dim)
        .ok_or_else(|| Error::unsupported_binop(SizeofDim::NAME, lhs, rhs, SizeofDim::IS_FUNC))?;
    Ok(Value::Scalar(Scalar::new_unchecked(
        Primitive::uint(size as u128),
        SIZEOF_OUT_TY,
    )))
}

fn scalar_as_usize(value: Scalar) -> Option<usize> {
    let scalar = value.cast(SIZEOF_DIM_TY).ok()?;
    usize::try_from(scalar.value().as_int(IntWidth::B64)?).ok()
}

impl BinOp for SizeofDim {
    const NAME: &'static str = "sizeof";
    const IS_FUNC: bool = true;

    fn scalar_check(
        lhs: PrimitiveTy,
        rhs: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        if ValueTy::Scalar(lhs).size(0).is_none() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                ValueTy::Scalar(lhs),
                ValueTy::Scalar(rhs),
                Self::IS_FUNC,
            ));
        }
        rhs.cast(SIZEOF_DIM_TY)?;
        Ok((lhs, SIZEOF_DIM_TY, SIZEOF_OUT_TY))
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, _out: PrimitiveTy) -> Result<Scalar> {
        let rhs_ty = rhs.ty();
        let dim = scalar_as_usize(rhs).ok_or_else(|| {
            Error::unsupported_binop(
                Self::NAME,
                ValueTy::Scalar(lhs.ty()),
                ValueTy::Scalar(rhs_ty),
                Self::IS_FUNC,
            )
        })?;
        let Value::Scalar(value) =
            sizeof_result(ValueTy::Scalar(lhs.ty()), dim, ValueTy::Scalar(rhs_ty))?
        else {
            unreachable!("sizeof scalar result is always scalar")
        };
        Ok(value)
    }

    fn arr_scalar_check(lhs: ArrayTy, rhs: PrimitiveTy) -> Result<(ArrayTy, PrimitiveTy, ArrayTy)> {
        rhs.cast(SIZEOF_DIM_TY)?;
        if ValueTy::Array(lhs).size(0).is_none() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                ValueTy::Array(lhs),
                ValueTy::Scalar(rhs),
                Self::IS_FUNC,
            ));
        }
        Ok((
            lhs,
            SIZEOF_DIM_TY,
            ArrayTy::new(SIZEOF_OUT_TY, ashape(vec![1])),
        ))
    }

    fn arr_ref_scalar_check(
        lhs: ArrayRefTy,
        rhs: PrimitiveTy,
    ) -> Result<(ArrayRefTy, PrimitiveTy, ArrayRefTy)> {
        rhs.cast(SIZEOF_DIM_TY)?;
        if ValueTy::ArrayRef(lhs).size(0).is_none() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                ValueTy::ArrayRef(lhs),
                ValueTy::Scalar(rhs),
                Self::IS_FUNC,
            ));
        }
        Ok((lhs, SIZEOF_DIM_TY, lhs.with_ty(SIZEOF_OUT_TY)))
    }

    fn op(lhs: Value, rhs: Value, _out: ValueTy) -> Result<Value> {
        let lhs_ty = lhs.ty();
        let rhs_ty = rhs.ty();
        let Value::Scalar(rhs) = rhs.cast(ValueTy::Scalar(SIZEOF_DIM_TY))? else {
            unreachable!("sizeof dimension is always scalar after cast")
        };
        let dim = scalar_as_usize(rhs)
            .ok_or_else(|| Error::unsupported_binop(Self::NAME, lhs_ty, rhs_ty, Self::IS_FUNC))?;
        sizeof_result(lhs_ty, dim, rhs_ty)
    }

    fn checked_op(lhs: Value, rhs: Value) -> Result<Value> {
        let lhs_ty = lhs.ty();
        let rhs_ty = rhs.ty();
        Self::return_ty(lhs_ty, rhs_ty)?;
        Self::op(lhs, rhs, ValueTy::Scalar(SIZEOF_OUT_TY))
    }

    fn return_ty(lhs: ValueTy, rhs: ValueTy) -> Result<ValueTy> {
        if lhs.sizeof_rank().is_none() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                lhs,
                rhs,
                Self::IS_FUNC,
            ));
        }
        rhs.cast(ValueTy::Scalar(SIZEOF_DIM_TY))?;
        Ok(ValueTy::Scalar(SIZEOF_OUT_TY))
    }
}

impl Value {
    pub fn sizeof_dim_(self, rhs: Self) -> Result<Self> {
        SizeofDim::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::index::Index;
    use crate::primitive::{PrimitiveTy::*, iw};

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&value| Primitive::uint(value)).collect(),
            ArrayTy::new(Uint(iw(bits)), ashape(shape)),
        ))
    }

    fn int(value: i128) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::int(value),
            PrimitiveTy::Int(IntWidth::B64),
        ))
    }

    #[test]
    fn sizeof_dim_array_returns_requested_dimension() {
        let result = u_array(&[1, 2, 3, 4, 5, 6], 8, vec![2, 3])
            .sizeof_dim_(int(1))
            .unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), SIZEOF_OUT_TY);
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(3));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_dim_array_returns_element_bitwidth_on_last_dimension() {
        let result = u_array(&[1, 2, 3, 4], 8, vec![2, 2])
            .sizeof_dim_(int(2))
            .unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(8));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_dim_fixed_array_ref_returns_requested_dimension() {
        let array_ref = u_array(&[1, 2, 3, 4, 5, 6], 8, vec![2, 3])
            .get(&[Index::Item(1)])
            .unwrap();

        let result = array_ref.sizeof_dim_(int(0)).unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(3));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_dim_rejects_negative_dimension() {
        let array = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);

        assert!(array.sizeof_dim_(int(-1)).is_err());
    }

    #[test]
    fn sizeof_dim_rejects_out_of_bounds_dimension() {
        let array = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);

        assert!(array.sizeof_dim_(int(3)).is_err());
    }

    #[test]
    fn sizeof_dim_rejects_unsupported_dimension_type() {
        let array = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);
        let angle = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Angle(iw(8))));

        assert!(array.sizeof_dim_(angle).is_err());
    }

    #[test]
    fn sizeof_dim_rank_only_array_ref_type_is_accepted() {
        // An unspecified-length array reference type-checks to `uint`; the
        // concrete dimension is read at run time from the array passed in.
        use crate::array_ref::{ArrayRefTy, RefAccess};
        let array_ref_ty = ArrayRefTy::new(
            Uint(iw(8)),
            crate::ArrayRefShape::Dim(crate::adim(3)),
            RefAccess::Readonly,
        );

        assert_eq!(
            SizeofDim::return_ty(
                ValueTy::ArrayRef(array_ref_ty),
                ValueTy::Scalar(SIZEOF_DIM_TY)
            )
            .unwrap(),
            ValueTy::Scalar(SIZEOF_OUT_TY)
        );
    }
}
