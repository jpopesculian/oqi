//! Value constructors shared by the operator tests.

pub(super) use crate::array::{Array, ArrayTy};
pub(super) use crate::array_ref::ArrayRefTy;
pub(super) use crate::ops::{BinOp, UnOp, sizeof, sizeof_ty};
pub(super) use crate::primitive::{FloatWidth, IntWidth, Primitive, PrimitiveTy};
pub(super) use crate::scalar::Scalar;
pub(super) use crate::value::{Value, ValueTy};

use crate::array::ashape;
use crate::primitive::iw;

pub(super) fn bit_scalar(v: bool) -> Value {
    Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), PrimitiveTy::Bit))
}

pub(super) fn bool_scalar(v: bool) -> Value {
    Value::Scalar(Scalar::new_unchecked(Primitive::bit(v), PrimitiveTy::Bool))
}

pub(super) fn bitreg(v: u128, bits: u32) -> Value {
    Value::bitreg_u128(v, bits)
}

pub(super) fn u_scalar(v: u128, bits: u32) -> Value {
    Value::uint(v, iw(bits))
}

pub(super) fn i_scalar(v: i128, bits: u32) -> Value {
    Value::int(v, iw(bits))
}

pub(super) fn f64_scalar(v: f64) -> Value {
    Value::float(v, FloatWidth::F64)
}

pub(super) fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
    Value::Array(Array::new_unchecked(
        values.iter().map(|&v| Primitive::uint(v)).collect(),
        ArrayTy::new(PrimitiveTy::Uint(iw(bits)), ashape(shape)),
    ))
}

pub(super) fn f64_array(values: &[f64], shape: Vec<usize>) -> Value {
    Value::Array(Array::new_unchecked(
        values.iter().map(|&v| Primitive::float(v)).collect(),
        ArrayTy::new(PrimitiveTy::Float(FloatWidth::F64), ashape(shape)),
    ))
}

/// A read-only reference to the whole of an owned array.
pub(super) fn into_ref(array: Value) -> Value {
    match array {
        Value::Array(array) => Value::ArrayRef(array.into_ref()),
        other => other,
    }
}
