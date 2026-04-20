pub use oqi_classical::{
    ArrayDim, ArrayRefShape, ArrayRefTy, ArrayShape, ArrayTy, BaseArray, BaseArrayRef,
    BaseArrayRefTy, BaseArrayTy, BaseScalar, BaseValue, BaseValueTy, BitWidth, Duration,
    DurationUnit, FloatWidth, Primitive, PrimitiveTy, RefAccess, Scalar, Value, ValueTy, adim,
    ashape, bw,
};

#[inline]
pub fn value_as_usize(value: &Value) -> Option<usize> {
    let Value::Scalar(scalar) = value else {
        return None;
    };
    let scalar = scalar.cast(PrimitiveTy::Int(bw(128))).ok()?;
    usize::try_from(scalar.value().as_int(bw(128))?).ok()
}
