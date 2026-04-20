use std::fmt;

use num_complex::{Complex32, Complex64};

use crate::{
    BitWidth, FloatWidth, Primitive,
    array::{Array, ArrayShape, ArrayTy, BaseArray, BaseArrayTy},
    array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, BaseArrayRef, BaseArrayRefTy, RefAccess},
    duration::{Duration, DurationUnit},
    error::{Error, Result},
    primitive::PrimitiveTy,
    scalar::{BaseScalar, Scalar},
};

#[derive(Clone, Debug)]
pub enum BaseValue<V, T> {
    Scalar(BaseScalar<V, T>),
    Array(BaseArray<V, T>),
    ArrayRef(BaseArrayRef<V, T>),
}

impl<V: Copy, T: Copy> BaseValue<V, T> {
    pub fn ty(&self) -> BaseValueTy<T> {
        match self {
            BaseValue::Scalar(s) => BaseValueTy::Scalar(s.ty()),
            BaseValue::Array(a) => BaseValueTy::Array(a.ty()),
            BaseValue::ArrayRef(ar) => BaseValueTy::ArrayRef(ar.ty()),
        }
    }
}

pub type Value = BaseValue<Primitive, PrimitiveTy>;

impl Value {
    pub const PI: Self = Value::Scalar(Scalar::PI);
    pub const TAU: Self = Value::Scalar(Scalar::TAU);
    pub const E: Self = Value::Scalar(Scalar::E);

    pub fn cast(self, ty: ValueTy) -> Result<Self> {
        match (self, ty) {
            (Value::Scalar(s), ValueTy::Scalar(ty)) => Ok(Value::Scalar(s.cast(ty)?)),
            (Value::Array(a), ValueTy::Array(ty)) => Ok(Value::Array(a.cast(ty)?)),
            (Value::ArrayRef(ar), ValueTy::ArrayRef(ty)) => Ok(Value::Array(ar.cast(ty)?)),
            (Value::Array(ar), ValueTy::ArrayRef(ty)) => {
                Ok(Value::Array(ar.into_ref_mut().cast(ty)?))
            }
            (Value::ArrayRef(ar), ValueTy::Array(ty)) => Ok(Value::Array(ar.cast(ty.as_ref())?)),
            (value, ty) => Err(Error::unsupported_cast(value.ty(), ty)),
        }
    }

    #[inline]
    pub const fn bit(v: bool) -> Self {
        Self::Scalar(Scalar::bit(v))
    }
    #[inline]
    pub const fn int(v: i128, bw: BitWidth) -> Self {
        Self::Scalar(Scalar::int(v, bw))
    }
    #[inline]
    pub const fn uint(v: u128, bw: BitWidth) -> Self {
        Self::Scalar(Scalar::uint(v, bw))
    }
    #[inline]
    pub const fn float(v: f64, fw: FloatWidth) -> Self {
        Self::Scalar(Scalar::float(v, fw))
    }
    #[inline]
    pub const fn complex(re: f64, im: f64, fw: FloatWidth) -> Self {
        Self::Scalar(Scalar::complex(re, im, fw))
    }
    #[inline]
    pub const fn duration(v: f64, unit: DurationUnit) -> Self {
        Self::Scalar(Scalar::duration(v, unit))
    }
    #[inline]
    pub const fn bitreg(bits: u128, bw: BitWidth) -> Self {
        Self::Scalar(Scalar::bitreg(bits, bw))
    }
    #[inline]
    pub fn angle(radians: f64) -> Self {
        Value::Scalar(Scalar::angle(radians))
    }
}

impl From<Scalar> for Value {
    fn from(scalar: Scalar) -> Self {
        Value::Scalar(scalar)
    }
}

impl From<Array> for Value {
    fn from(array: Array) -> Self {
        Value::Array(array)
    }
}

impl From<ArrayRef> for Value {
    fn from(array_ref: ArrayRef) -> Self {
        Value::ArrayRef(array_ref)
    }
}

impl From<Primitive> for Value {
    #[inline]
    fn from(primitive: Primitive) -> Self {
        Value::Scalar(Scalar::from(primitive))
    }
}

macro_rules! impl_from_primitive {
    ($($ty:ty),* $(,)?) => {
        $(
        impl From<$ty> for Value {
            fn from(value: $ty) -> Self {
                Self::Scalar(Scalar::from(value))
            }
        }
        )*
    };
}

impl_from_primitive!(
    bool,
    u8,
    u16,
    u32,
    u64,
    u128,
    i8,
    i16,
    i32,
    i64,
    i128,
    f32,
    f64,
    Complex32,
    Complex64,
    Duration,
    turns::Angle8,
    turns::Angle16,
    turns::Angle32,
    turns::Angle64,
    turns::Angle128,
);

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Scalar(s) => write!(f, "{}", s),
            Value::Array(a) => write!(f, "{}", a),
            Value::ArrayRef(ar) => write!(f, "{}", ar),
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum BaseValueTy<T> {
    Scalar(T),
    Array(BaseArrayTy<T>),
    ArrayRef(BaseArrayRefTy<T>),
}

pub type ValueTy = BaseValueTy<PrimitiveTy>;

impl ValueTy {
    pub fn cast(self, ty: ValueTy) -> Result<Self> {
        match (self, ty) {
            (ValueTy::Scalar(from), ValueTy::Scalar(to)) => Ok(ValueTy::Scalar(from.cast(to)?)),
            (ValueTy::Array(from), ValueTy::Array(to)) => {
                from.cast(to)?;
                Ok(ValueTy::Array(to))
            }
            (ValueTy::ArrayRef(from), ValueTy::ArrayRef(to)) => {
                from.cast(to)?;
                Ok(ValueTy::ArrayRef(to))
            }
            (ValueTy::Array(from), ValueTy::ArrayRef(to)) => {
                from.as_ref_mut().cast(to)?;
                Ok(ValueTy::ArrayRef(to))
            }
            (ValueTy::ArrayRef(from), ValueTy::Array(to)) => {
                from.cast(to.as_ref())?;
                Ok(ValueTy::Array(to))
            }
            (from, to) => Err(Error::unsupported_cast(from, to)),
        }
    }

    pub fn size(&self, dim: usize) -> Option<usize> {
        match self {
            ValueTy::Scalar(ty) if dim == 0 => ty.bw().map(|bw| bw.get() as usize),
            ValueTy::Array(ty) => {
                if dim == ty.shape().dim() {
                    ty.ty().bw().map(|bw| bw.get() as usize)
                } else {
                    ty.shape().get().get(dim).copied()
                }
            }
            ValueTy::ArrayRef(ty) => match ty.shape() {
                ArrayRefShape::Fixed(shape) => {
                    if dim == shape.dim() {
                        ty.ty().bw().map(|bw| bw.get() as usize)
                    } else {
                        shape.get().get(dim).copied()
                    }
                }
                ArrayRefShape::Dim(_) => None,
            },
            _ => None,
        }
    }

    #[inline]
    pub const fn bit() -> Self {
        Self::Scalar(PrimitiveTy::Bit)
    }

    #[inline]
    pub const fn bool() -> Self {
        Self::Scalar(PrimitiveTy::Bool)
    }

    #[inline]
    pub const fn int(bw: BitWidth) -> Self {
        Self::Scalar(PrimitiveTy::Int(bw))
    }

    #[inline]
    pub const fn uint(bw: BitWidth) -> Self {
        Self::Scalar(PrimitiveTy::Uint(bw))
    }

    #[inline]
    pub const fn float(fw: FloatWidth) -> Self {
        Self::Scalar(PrimitiveTy::Float(fw))
    }

    #[inline]
    pub const fn complex(fw: FloatWidth) -> Self {
        Self::Scalar(PrimitiveTy::Complex(fw))
    }

    #[inline]
    pub const fn angle(bw: BitWidth) -> Self {
        Self::Scalar(PrimitiveTy::Angle(bw))
    }

    #[inline]
    pub const fn duration() -> Self {
        Self::Scalar(PrimitiveTy::Duration)
    }

    #[inline]
    pub const fn bitreg(bw: BitWidth) -> Self {
        Self::Scalar(PrimitiveTy::BitReg(bw))
    }

    #[inline]
    pub const fn array(ty: PrimitiveTy, shape: ArrayShape) -> Self {
        Self::Array(ArrayTy::new(ty, shape))
    }

    #[inline]
    pub const fn array_ref(ty: PrimitiveTy, shape: ArrayRefShape, access: RefAccess) -> Self {
        Self::ArrayRef(ArrayRefTy::new(ty, shape, access))
    }
}

impl From<PrimitiveTy> for ValueTy {
    #[inline]
    fn from(scalar_ty: PrimitiveTy) -> Self {
        ValueTy::Scalar(scalar_ty)
    }
}

impl From<ArrayTy> for ValueTy {
    #[inline]
    fn from(array_ty: ArrayTy) -> Self {
        ValueTy::Array(array_ty)
    }
}

impl From<ArrayRefTy> for ValueTy {
    #[inline]
    fn from(array_ref_ty: ArrayRefTy) -> Self {
        ValueTy::ArrayRef(array_ref_ty)
    }
}

impl<T: fmt::Display + Copy> fmt::Display for BaseValueTy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaseValueTy::Scalar(s) => write!(f, "{}", s),
            BaseValueTy::Array(a) => write!(f, "{}", a),
            BaseValueTy::ArrayRef(ar) => write!(f, "{}", ar),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DurationUnit,
        array::{Array, ArrayTy, adim, ashape},
        array_ref::{ArrayRefShape, ArrayRefTy, RefAccess},
        primitive::{FloatWidth::F64, Primitive, PrimitiveTy, PrimitiveTy::*, bw},
        scalar::Scalar,
    };

    fn aty(ty: PrimitiveTy, shape: Vec<usize>) -> ArrayTy {
        ArrayTy::new(ty, ashape(shape))
    }

    #[test]
    fn scalar_cast_delegates_to_scalar_cast() {
        let value = Value::Scalar(Scalar::new_unchecked(Primitive::uint(42_u128), Uint(bw(8))));

        let cast = value.cast(ValueTy::Scalar(Float(F64))).unwrap();

        match cast {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 42.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn array_cast_preserves_shape_and_casts_each_element() {
        let value = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(1_u128), Primitive::uint(2_u128)],
            aty(Uint(bw(8)), vec![2]),
        ));

        let cast = value
            .cast(ValueTy::Array(aty(Float(F64), vec![2])))
            .unwrap();

        match cast {
            Value::Array(a) => {
                assert!(matches!(a.ty().ty(), Float(F64)));
                assert_eq!(a.ty().shape().get(), &[2]);
                assert_eq!(
                    a.values()
                        .iter()
                        .map(|scalar| scalar.as_float(F64).unwrap())
                        .collect::<Vec<_>>(),
                    vec![1.0, 2.0]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_allow_rejects_shape_changes() {
        let value = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(1_u128), Primitive::uint(2_u128)],
            aty(Uint(bw(8)), vec![2]),
        ));

        assert!(
            value
                .cast(ValueTy::Array(aty(Uint(bw(8)), vec![1, 2])))
                .is_ok()
        );
    }

    #[test]
    fn array_cast_rejects_len_changes() {
        let value = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(1_u128), Primitive::uint(2_u128)],
            aty(Uint(bw(8)), vec![2]),
        ));

        assert!(
            value
                .cast(ValueTy::Array(aty(Uint(bw(8)), vec![1, 3])))
                .is_err()
        );
    }

    #[test]
    fn cast_rejects_scalar_array_mismatch() {
        let scalar = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Uint(bw(8))));
        let array = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(1_u128)],
            aty(Uint(bw(8)), vec![1]),
        ));

        assert!(
            scalar
                .cast(ValueTy::Array(aty(Uint(bw(8)), vec![1])))
                .is_err()
        );
        assert!(array.cast(ValueTy::Scalar(Uint(bw(8)))).is_err());
    }

    #[test]
    fn cast_returns_none_when_element_cast_fails() {
        let value = Value::Array(Array::new_unchecked(
            vec![Primitive::duration(1.0, DurationUnit::Ns)],
            aty(Duration, vec![1]),
        ));

        assert!(
            value
                .cast(ValueTy::Array(aty(Uint(bw(8)), vec![1])))
                .is_err()
        );
    }

    #[test]
    fn value_ty_scalar_cast_returns_target_type() {
        let cast = ValueTy::Scalar(Uint(bw(8)))
            .cast(ValueTy::Scalar(Float(F64)))
            .unwrap();

        assert_eq!(cast, ValueTy::Scalar(Float(F64)));
    }

    #[test]
    fn value_ty_array_cast_returns_target_type() {
        let cast = ValueTy::Array(aty(Uint(bw(8)), vec![2]))
            .cast(ValueTy::Array(aty(Float(F64), vec![1, 2])))
            .unwrap();

        assert_eq!(cast, ValueTy::Array(aty(Float(F64), vec![1, 2])));
    }

    #[test]
    fn value_ty_array_to_array_ref_cast_uses_array_ref_rules() {
        let target = ValueTy::ArrayRef(ArrayRefTy::new(
            Float(F64),
            ArrayRefShape::Dim(adim(1)),
            RefAccess::Readonly,
        ));
        let cast = ValueTy::Array(aty(Uint(bw(8)), vec![2]))
            .cast(target)
            .unwrap();

        assert_eq!(cast, target);
    }

    #[test]
    fn value_ty_array_ref_to_array_cast_uses_array_ref_rules() {
        let cast = ValueTy::ArrayRef(ArrayRefTy::new(
            Uint(bw(8)),
            ArrayRefShape::Fixed(ashape(vec![2])),
            RefAccess::Mutable,
        ))
        .cast(ValueTy::Array(aty(Float(F64), vec![2])))
        .unwrap();

        assert_eq!(cast, ValueTy::Array(aty(Float(F64), vec![2])));
    }

    #[test]
    fn value_ty_cast_rejects_scalar_array_mismatch() {
        assert!(
            ValueTy::Scalar(Uint(bw(8)))
                .cast(ValueTy::Array(aty(Uint(bw(8)), vec![1])))
                .is_err()
        );
    }
}
