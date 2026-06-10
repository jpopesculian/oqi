pub use oqi_classical::{
    ArrayDim, ArrayRefShape, ArrayRefTy, ArrayShape, ArrayTy, BaseArray, BaseArrayRef,
    BaseArrayRefTy, BaseArrayTy, BaseScalar, BaseValue, BaseValueTy, Duration, DurationUnit,
    FloatWidth, IntWidth, Primitive, PrimitiveTy, RefAccess, Scalar, Value, ValueTy, adim, ashape,
    iw,
};

#[inline]
pub fn value_as_usize(value: &Value) -> Option<usize> {
    let Value::Scalar(scalar) = value else {
        return None;
    };
    let scalar = scalar.clone().cast(PrimitiveTy::Int(iw(128))).ok()?;
    usize::try_from(scalar.value().as_int(iw(128))?).ok()
}
