//! The built-in `sizeof(value)` / `sizeof(value, dim)`.
//!
//! Sizes live in types, so `sizeof` reads the *type* of its first argument;
//! only an unspecified-length array reference has to wait for a run-time
//! value to supply one.

use crate::primitive::{IntWidth, PrimitiveTy};
use crate::value::{Value, ValueTy};
use crate::{Error, Result};

const DIM_TY: ValueTy = ValueTy::Scalar(PrimitiveTy::Int(IntWidth::B64));
const OUT_TY: ValueTy = ValueTy::Scalar(PrimitiveTy::Uint(IntWidth::B64));

fn unsupported(value: ValueTy, dim: Option<ValueTy>) -> Error {
    Error::UnsupportedOp {
        op: super::Operator::Sizeof,
        args: std::iter::once(value).chain(dim).collect(),
    }
}

/// The type of `sizeof(value)` or `sizeof(value, dim)`, or an error if
/// `value` has no size or `dim` is not an integer.
///
/// # Errors
///
/// [`Error::UnsupportedOp`] if `value` has no size;
/// [`Error::UnsupportedCast`] if `dim` is not an integer.
pub fn sizeof_ty(value: ValueTy, dim: Option<ValueTy>) -> Result<ValueTy> {
    if value.sizeof_rank().is_none() {
        return Err(unsupported(value, dim));
    }
    if let Some(dim) = dim {
        dim.cast(DIM_TY)?;
    }
    Ok(OUT_TY)
}

/// The size of dimension `dim` (default 0) of a value of type `value`.
///
/// # Errors
///
/// As [`sizeof_ty`], and [`Error::IndexOutOfBounds`] if `dim` is not a
/// dimension of `value` or its size is not known from the type.
pub fn sizeof(value: ValueTy, dim: Option<Value>) -> Result<Value> {
    let dim_ty = dim.as_ref().map(Value::ty);
    sizeof_ty(value, dim_ty)?;
    let dim = match dim {
        None => Some(0),
        Some(dim) => match dim.cast(DIM_TY)? {
            Value::Scalar(dim) => dim
                .value()
                .as_int(IntWidth::B64)
                .and_then(|dim| usize::try_from(dim).ok()),
            _ => None,
        },
    };
    let size = dim
        .and_then(|dim| value.size(dim))
        .ok_or_else(|| unsupported(value, dim_ty))?;
    Ok(Value::uint(size as u128, IntWidth::B64))
}
