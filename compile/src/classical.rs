pub use oqi_classical::{
    ArrayDim, ArrayRefShape, ArrayRefTy, ArrayShape, ArrayTy, BitWidth, Duration, DurationUnit,
    FloatWidth, Primitive, PrimitiveTy, RefAccess, Scalar, ScalarTy, Value, ValueTy, adim,
};

#[inline]
pub fn bit_width(bits: usize) -> BitWidth {
    oqi_classical::bw(bits.try_into().expect("bit width must fit in u32"))
}

#[inline]
fn scalar(value: Primitive, ty: PrimitiveTy) -> Scalar {
    Scalar::new(value, ty).expect("compile-generated classical value should be valid")
}

#[inline]
pub fn bool_value(value: bool) -> Value {
    scalar(Primitive::bit(value), PrimitiveTy::Bool).into()
}

#[inline]
pub fn int_value(value: i128) -> Value {
    scalar(Primitive::int(value), PrimitiveTy::Int(bit_width(128))).into()
}

#[inline]
pub fn uint_value(value: u128, width: usize) -> Value {
    scalar(Primitive::uint(value), PrimitiveTy::Uint(bit_width(width))).into()
}

#[inline]
pub fn float_value(value: f64, width: FloatWidth) -> Value {
    let value = match width {
        FloatWidth::F32 => value as f32 as f64,
        FloatWidth::F64 => value,
    };
    scalar(Primitive::float(value), PrimitiveTy::Float(width)).into()
}

#[inline]
pub fn complex_value(re: f64, im: f64, width: FloatWidth) -> Value {
    let (re, im) = match width {
        FloatWidth::F32 => (re as f32 as f64, im as f32 as f64),
        FloatWidth::F64 => (re, im),
    };
    scalar(Primitive::complex(re, im), PrimitiveTy::Complex(width)).into()
}

#[inline]
pub fn bitreg_value(bits: u128, width: usize) -> Value {
    scalar(
        Primitive::bitreg(bits),
        PrimitiveTy::BitReg(bit_width(width)),
    )
    .into()
}

#[inline]
pub fn duration_value(duration: Duration) -> Value {
    scalar(Primitive::Duration(duration), PrimitiveTy::Duration).into()
}

#[inline]
pub fn value_as_usize(value: &Value) -> Option<usize> {
    let Value::Scalar(scalar) = value else {
        return None;
    };
    let scalar = scalar.cast(PrimitiveTy::Int(bit_width(128))).ok()?;
    usize::try_from(scalar.value().as_int(bit_width(128))?).ok()
}
