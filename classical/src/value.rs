use std::fmt;

use num_complex::{Complex32, Complex64};

use crate::{
    BitReg, FloatWidth, IntWidth, Primitive,
    array::{Array, ArrayShape, ArrayTy},
    array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, RefAccess},
    duration::{Duration, DurationUnit},
    error::{Error, Result},
    primitive::PrimitiveTy,
    scalar::Scalar,
};

/// A classical value: a scalar, an owned array, or a reference to part of
/// an array. Equality is structural for scalars and arrays (floats compare
/// as `f64` does); two references are equal when they view the same
/// elements of the same array.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    /// A scalar.
    Scalar(Scalar),
    /// An owned array.
    Array(Array),
    /// A reference to all or part of an array.
    ArrayRef(ArrayRef),
}

impl Value {
    /// The value's type.
    pub fn ty(&self) -> ValueTy {
        match self {
            Value::Scalar(s) => ValueTy::Scalar(s.ty()),
            Value::Array(a) => ValueTy::Array(a.ty()),
            Value::ArrayRef(ar) => ValueTy::ArrayRef(ar.ty()),
        }
    }

    /// Cast to `ty` under the language's cast rules. Arrays and array
    /// references cast element-wise into a new owned array.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedCast`] if the cast is not allowed;
    /// [`Error::BadBorrow`] if a referenced array is being written.
    pub fn cast(self, ty: ValueTy) -> Result<Self> {
        match (self, ty) {
            (Value::Scalar(s), ValueTy::Scalar(ty)) => Ok(Value::Scalar(s.cast(ty)?)),
            (Value::Array(a), ValueTy::Array(ty)) => Ok(Value::Array(a.cast(ty)?)),
            (Value::ArrayRef(ar), ValueTy::ArrayRef(ty)) => Ok(Value::Array(ar.cast(ty)?)),
            (Value::Array(ar), ValueTy::ArrayRef(ty)) => {
                Ok(Value::Array(ar.into_ref_mut().cast(ty)?))
            }
            (Value::ArrayRef(ar), ValueTy::Array(ty)) => {
                Ok(Value::Array(ar.cast(ty.ref_ty(RefAccess::Readonly))?))
            }
            (value, ty) => Err(Error::unsupported_cast(value.ty(), ty)),
        }
    }

    /// Materialise an array reference into (a copy of) the array it views;
    /// scalars and owned arrays pass through.
    ///
    /// # Errors
    ///
    /// [`Error::BadBorrow`] if the referenced array is being written.
    pub fn into_owned(self) -> Result<Self> {
        match self {
            Value::ArrayRef(ar) => ar.to_array().map(Value::Array),
            owned => Ok(owned),
        }
    }

    /// A `bit`.
    #[inline]
    pub const fn bit(v: bool) -> Self {
        Self::Scalar(Scalar::bit(v))
    }
    /// An `int[bw]`, wrapped to `bw` bits.
    #[inline]
    pub const fn int(v: i128, bw: IntWidth) -> Self {
        Self::Scalar(Scalar::int(v, bw))
    }
    /// A `uint[bw]`, wrapped to `bw` bits.
    #[inline]
    pub const fn uint(v: u128, bw: IntWidth) -> Self {
        Self::Scalar(Scalar::uint(v, bw))
    }
    /// A `float[fw]` (rounded to `f32` at 32 bits).
    #[inline]
    pub const fn float(v: f64, fw: FloatWidth) -> Self {
        Self::Scalar(Scalar::float(v, fw))
    }
    /// A `complex[float[fw]]` (each part rounded to `f32` at 32 bits).
    #[inline]
    pub const fn complex(re: f64, im: f64, fw: FloatWidth) -> Self {
        Self::Scalar(Scalar::complex(re, im, fw))
    }
    /// A `duration` of `v` in `unit`.
    #[inline]
    pub const fn duration(v: f64, unit: DurationUnit) -> Self {
        Self::Scalar(Scalar::duration(v, unit))
    }
    /// A `bit[width]` holding the low `width` bits of `reg`.
    #[inline]
    pub fn bitreg(reg: BitReg, width: u32) -> Self {
        Self::Scalar(Scalar::bitreg(reg, width))
    }
    /// A `bit[width]` holding the low `width` bits of `bits`.
    #[inline]
    pub fn bitreg_u128(bits: u128, width: u32) -> Self {
        Self::Scalar(Scalar::bitreg_u128(bits, width))
    }
    /// An `angle` of `radians`, reduced modulo 2π.
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

/// The type of a [`Value`].
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ValueTy {
    /// A scalar type.
    Scalar(PrimitiveTy),
    /// An array type.
    Array(ArrayTy),
    /// An array reference type.
    ArrayRef(ArrayRefTy),
}

impl ValueTy {
    /// Check that a value of this type may be cast to `ty`, and return `ty`.
    /// Scalars follow the spec's table of allowed casts; arrays cast
    /// element-wise and must have the same shape.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedCast`] if the cast is not allowed.
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
                from.ref_ty(RefAccess::Mutable).cast(to)?;
                Ok(ValueTy::ArrayRef(to))
            }
            (ValueTy::ArrayRef(from), ValueTy::Array(to)) => {
                from.cast(to.ref_ty(RefAccess::Readonly))?;
                Ok(ValueTy::Array(to))
            }
            (from, to) => Err(Error::unsupported_cast(from, to)),
        }
    }

    /// Whether a value of this type can stand in for one of `other`
    /// without conversion (see [`PrimitiveTy::is_compatible`] and
    /// [`ArrayRefShape::is_compatible`]).
    pub(crate) fn is_compatible(&self, other: &ValueTy) -> bool {
        match (self, other) {
            (ValueTy::Scalar(a), ValueTy::Scalar(b)) => a.is_compatible(*b),
            (ValueTy::Array(a), ValueTy::Array(b)) => {
                a.element_ty().is_compatible(b.element_ty()) && a.shape() == b.shape()
            }
            (ValueTy::ArrayRef(a), ValueTy::ArrayRef(b)) => {
                a.element_ty().is_compatible(b.element_ty())
                    && a.shape().is_compatible(&b.shape())
                    && a.access() == b.access()
            }
            _ => false,
        }
    }

    /// The queryable rank of a `sizeof`-able type, or `None` for types
    /// that `sizeof` does not accept. A dimension `dim` is valid iff
    /// `dim <= rank`. Unlike [`size`](Self::size), this is known even for
    /// unspecified-length array references (`array[T, #dim = N]`), whose
    /// concrete dimension sizes only arrive at run time.
    pub fn sizeof_rank(&self) -> Option<usize> {
        match self {
            ValueTy::Scalar(ty) => ty.bit_count().map(|_| 0),
            ValueTy::Array(ty) => Some(ty.shape().dim().get()),
            ValueTy::ArrayRef(ty) => Some(ty.shape().dim().get()),
        }
    }

    /// What `sizeof(value, dim)` returns for a value of this type: the size
    /// of dimension `dim` of an array (the bit width at `dim == rank`), the
    /// bit width of a scalar at `dim == 0`. `None` when that is not known
    /// from the type (a `#dim` reference) or `dim` is out of range.
    pub fn size(&self, dim: usize) -> Option<usize> {
        match self {
            ValueTy::Scalar(ty) if dim == 0 => ty.bit_count().map(|bw| bw as usize),
            ValueTy::Array(ty) => {
                if dim == ty.shape().dim().get() {
                    ty.element_ty().bit_count().map(|bw| bw as usize)
                } else {
                    ty.shape().dims().get(dim).copied()
                }
            }
            ValueTy::ArrayRef(ty) => match ty.shape() {
                ArrayRefShape::Fixed(shape) => {
                    if dim == shape.dim().get() {
                        ty.element_ty().bit_count().map(|bw| bw as usize)
                    } else {
                        shape.dims().get(dim).copied()
                    }
                }
                ArrayRefShape::Dim(_) => None,
            },
            _ => None,
        }
    }

    /// `bit`
    #[inline]
    pub const fn bit() -> Self {
        Self::Scalar(PrimitiveTy::Bit)
    }

    /// `bool`
    #[inline]
    pub const fn bool() -> Self {
        Self::Scalar(PrimitiveTy::Bool)
    }

    /// `int[bw]`
    #[inline]
    pub const fn int(bw: IntWidth) -> Self {
        Self::Scalar(PrimitiveTy::Int(bw))
    }

    /// `uint[bw]`
    #[inline]
    pub const fn uint(bw: IntWidth) -> Self {
        Self::Scalar(PrimitiveTy::Uint(bw))
    }

    /// `float[fw]`
    #[inline]
    pub const fn float(fw: FloatWidth) -> Self {
        Self::Scalar(PrimitiveTy::Float(fw))
    }

    /// `complex[float[fw]]`
    #[inline]
    pub const fn complex(fw: FloatWidth) -> Self {
        Self::Scalar(PrimitiveTy::Complex(fw))
    }

    /// `angle[bw]`
    #[inline]
    pub const fn angle(bw: IntWidth) -> Self {
        Self::Scalar(PrimitiveTy::Angle(bw))
    }

    /// `duration`
    #[inline]
    pub const fn duration() -> Self {
        Self::Scalar(PrimitiveTy::Duration)
    }

    /// `bit[width]`
    #[inline]
    pub const fn bitreg(width: u32) -> Self {
        Self::Scalar(PrimitiveTy::BitReg(width))
    }

    /// `array[ty, shape…]`
    #[inline]
    pub const fn array(ty: PrimitiveTy, shape: ArrayShape) -> Self {
        Self::Array(ArrayTy::new(ty, shape))
    }

    /// A `readonly` or `mutable` reference to an array of `ty`.
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

impl fmt::Display for ValueTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueTy::Scalar(s) => write!(f, "{}", s),
            ValueTy::Array(a) => write!(f, "{}", a),
            ValueTy::ArrayRef(ar) => write!(f, "{}", ar),
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
        primitive::{FloatWidth::F64, Primitive, PrimitiveTy, PrimitiveTy::*, iw},
        scalar::Scalar,
    };

    fn aty(ty: PrimitiveTy, shape: Vec<usize>) -> ArrayTy {
        ArrayTy::new(ty, ashape(shape))
    }

    #[test]
    fn scalar_cast_delegates_to_scalar_cast() {
        let value = Value::Scalar(Scalar::new_unchecked(Primitive::uint(42_u128), Uint(iw(8))));

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
            aty(Uint(iw(8)), vec![2]),
        ));

        let cast = value
            .cast(ValueTy::Array(aty(Float(F64), vec![2])))
            .unwrap();

        match cast {
            Value::Array(a) => {
                assert!(matches!(a.ty().element_ty(), Float(F64)));
                assert_eq!(a.ty().shape().dims(), &[2]);
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
            aty(Uint(iw(8)), vec![2]),
        ));

        assert!(
            value
                .cast(ValueTy::Array(aty(Uint(iw(8)), vec![1, 2])))
                .is_ok()
        );
    }

    #[test]
    fn array_cast_rejects_len_changes() {
        let value = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(1_u128), Primitive::uint(2_u128)],
            aty(Uint(iw(8)), vec![2]),
        ));

        assert!(
            value
                .cast(ValueTy::Array(aty(Uint(iw(8)), vec![1, 3])))
                .is_err()
        );
    }

    #[test]
    fn cast_rejects_scalar_array_mismatch() {
        let scalar = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Uint(iw(8))));
        let array = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(1_u128)],
            aty(Uint(iw(8)), vec![1]),
        ));

        assert!(
            scalar
                .cast(ValueTy::Array(aty(Uint(iw(8)), vec![1])))
                .is_err()
        );
        assert!(array.cast(ValueTy::Scalar(Uint(iw(8)))).is_err());
    }

    #[test]
    fn cast_returns_none_when_element_cast_fails() {
        let value = Value::Array(Array::new_unchecked(
            vec![Primitive::duration(1.0, DurationUnit::Ns)],
            aty(Duration, vec![1]),
        ));

        assert!(
            value
                .cast(ValueTy::Array(aty(Uint(iw(8)), vec![1])))
                .is_err()
        );
    }

    #[test]
    fn value_ty_scalar_cast_returns_target_type() {
        let cast = ValueTy::Scalar(Uint(iw(8)))
            .cast(ValueTy::Scalar(Float(F64)))
            .unwrap();

        assert_eq!(cast, ValueTy::Scalar(Float(F64)));
    }

    #[test]
    fn value_ty_array_cast_returns_target_type() {
        let cast = ValueTy::Array(aty(Uint(iw(8)), vec![2]))
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
        let cast = ValueTy::Array(aty(Uint(iw(8)), vec![2]))
            .cast(target)
            .unwrap();

        assert_eq!(cast, target);
    }

    #[test]
    fn value_ty_array_ref_to_array_cast_uses_array_ref_rules() {
        let cast = ValueTy::ArrayRef(ArrayRefTy::new(
            Uint(iw(8)),
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
            ValueTy::Scalar(Uint(iw(8)))
                .cast(ValueTy::Array(aty(Uint(iw(8)), vec![1])))
                .is_err()
        );
    }
}
