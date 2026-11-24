use std::fmt;

use crate::array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, RefAccess};
use crate::primitive::{Primitive, PrimitiveTy};
use crate::scalar::Scalar;
use crate::shared::Shared;
use crate::{Error, Result};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Array {
    values: Vec<Primitive>,
    ty: ArrayTy,
}

impl Array {
    #[inline]
    pub fn new_unchecked(values: Vec<Primitive>, ty: ArrayTy) -> Self {
        Array { values, ty }
    }

    pub fn values(&self) -> &[Primitive] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Primitive] {
        &mut self.values
    }

    pub fn ty(&self) -> ArrayTy {
        self.ty
    }

    /// Build an array from `values`, converting each to the element type.
    pub fn new(values: Vec<Primitive>, ty: ArrayTy) -> Result<Self> {
        if values.len() != ty.shape().total() {
            return Err(Error::LengthMismatch {
                expected: ty.shape().total(),
                received: values.len(),
            });
        }
        let values = values
            .into_iter()
            .map(|value| value.as_ty(ty.ty()))
            .collect::<Result<_>>()?;
        Ok(Array::new_unchecked(values, ty))
    }

    pub fn into_values(self) -> Vec<Primitive> {
        self.values
    }

    #[inline]
    pub fn into_ref(self) -> ArrayRef {
        ArrayRef::new(Shared::new(self), vec![], RefAccess::Readonly).unwrap()
    }

    #[inline]
    pub fn into_ref_mut(self) -> ArrayRef {
        ArrayRef::new(Shared::new(self), vec![], RefAccess::Mutable).unwrap()
    }

    pub fn scalars(&self) -> ScalarIter<'_> {
        ScalarIter {
            slice: self.values(),
            ty: self.ty.ty(),
            index: 0,
        }
    }

    #[inline]
    pub fn cast(self, to: ArrayTy) -> Result<Self> {
        self.ty.cast(to)?;
        Self::new(self.values, to)
    }
}

pub struct ScalarIter<'a> {
    slice: &'a [Primitive],
    ty: PrimitiveTy,
    index: usize,
}

impl<'a> Iterator for ScalarIter<'a> {
    type Item = Scalar;
    fn next(&mut self) -> Option<Self::Item> {
        let next = Scalar::new_unchecked(self.slice.get(self.index)?.clone(), self.ty);
        self.index += 1;
        Some(next)
    }
}

impl ExactSizeIterator for ScalarIter<'_> {
    fn len(&self) -> usize {
        self.slice.len() - self.index
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_values(
            f: &mut fmt::Formatter<'_>,
            shape: &[usize],
            values: &mut ScalarIter,
        ) -> fmt::Result {
            write!(f, "{{")?;

            if let Some((&len, rest)) = shape.split_first() {
                if rest.is_empty() {
                    for (i, value) in values.take(len).enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", value)?;
                    }
                } else {
                    for i in 0..len {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        fmt_values(f, rest, values)?;
                    }
                }
            }

            f.write_str("}")
        }

        fmt_values(f, self.ty.shape.get(), &mut self.scalars())
    }
}

/// The rank of an array: `1..=ArrayDim::MAX` dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "usize", into = "usize")
)]
pub struct ArrayDim(usize);

impl TryFrom<usize> for ArrayDim {
    type Error = Error;

    fn try_from(dim: usize) -> Result<Self> {
        Self::new(dim)
    }
}

impl From<ArrayDim> for usize {
    fn from(dim: ArrayDim) -> usize {
        dim.0
    }
}

impl ArrayDim {
    /// docs/types.rst: arrays have "a maximum of 7 total dimensions".
    pub const MAX: usize = 7;

    #[inline]
    pub const fn new(dim: usize) -> Result<Self> {
        if dim == 0 || dim > Self::MAX {
            return Err(Error::InvalidRank {
                received: dim,
                max: Self::MAX,
            });
        }
        Ok(Self(dim))
    }

    #[inline]
    pub fn get(&self) -> usize {
        self.0
    }
}

/// Shorthand for a rank known to be valid.
///
/// # Panics
/// If `value` is not in `1..=ArrayDim::MAX`; use [`ArrayDim::new`] for ranks
/// that come from a program.
pub fn adim(value: usize) -> ArrayDim {
    ArrayDim::new(value).unwrap()
}

/// The size of each dimension of an array: `1..=ArrayDim::MAX` dimensions
/// and at most [`ArrayShape::MAX_ELEMENTS`] elements in all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "Vec<usize>", into = "Vec<usize>")
)]
pub struct ArrayShape {
    shape: [usize; ArrayDim::MAX],
    dim: ArrayDim,
}

impl TryFrom<Vec<usize>> for ArrayShape {
    type Error = Error;

    fn try_from(shape: Vec<usize>) -> Result<Self> {
        Self::new(shape)
    }
}

impl From<ArrayShape> for Vec<usize> {
    fn from(shape: ArrayShape) -> Vec<usize> {
        shape.get().to_vec()
    }
}

impl fmt::Display for ArrayShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, size) in self.get().iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", size)?;
        }
        Ok(())
    }
}

impl ArrayShape {
    /// The most elements an array holds: a variable holds all of them in
    /// memory at once.
    pub const MAX_ELEMENTS: usize = 1 << 26;

    pub fn new(shape: Vec<usize>) -> Result<Self> {
        let dim = ArrayDim::new(shape.len())?;
        let total = shape
            .iter()
            .try_fold(1usize, |total, &size| total.checked_mul(size));
        if total.is_none_or(|total| total > Self::MAX_ELEMENTS) {
            return Err(Error::TooManyElements {
                max: Self::MAX_ELEMENTS,
            });
        }
        let mut shape_arr = [1; ArrayDim::MAX];
        shape_arr[..shape.len()].copy_from_slice(&shape);
        Ok(Self {
            shape: shape_arr,
            dim,
        })
    }

    #[inline]
    pub fn dim(&self) -> ArrayDim {
        self.dim
    }

    #[inline]
    pub fn get(&self) -> &[usize] {
        &self.shape[..self.dim.get()]
    }

    #[inline]
    pub fn total(&self) -> usize {
        self.get().iter().product()
    }
}

/// Shorthand for a shape known to be valid.
///
/// # Panics
/// If `values` does not have `1..=ArrayDim::MAX` dimensions; use
/// [`ArrayShape::new`] for shapes that come from a program.
pub fn ashape(values: Vec<usize>) -> ArrayShape {
    ArrayShape::new(values).unwrap()
}

#[cfg(test)]
macro_rules! shape {
    ($($dim:expr),+) => {
        crate::array::ashape(vec![$($dim),+])
    }
}
#[cfg(test)]
pub(crate) use shape;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayTy {
    ty: PrimitiveTy,
    shape: ArrayShape,
}

impl ArrayTy {
    #[inline]
    pub const fn new(ty: PrimitiveTy, shape: ArrayShape) -> Self {
        Self { ty, shape }
    }

    #[inline]
    pub const fn shape(&self) -> ArrayShape {
        self.shape
    }

    #[inline]
    pub const fn with_ty(&self, ty: PrimitiveTy) -> Self {
        Self { ty, ..*self }
    }

    #[inline]
    pub const fn with_shape(&self, shape: ArrayShape) -> Self {
        Self::new(self.ty, shape)
    }

    #[inline]
    pub const fn ty(&self) -> PrimitiveTy {
        self.ty
    }

    /// The type of a reference to an array of this type.
    #[inline]
    pub const fn as_ref(&self, access: RefAccess) -> ArrayRefTy {
        ArrayRefTy::new(self.ty, ArrayRefShape::Fixed(self.shape), access)
    }

    #[inline]
    pub fn cast(self, to: ArrayTy) -> Result<Self> {
        if self.shape.total() != to.shape.total() {
            return Err(Error::unsupported_cast(self, to));
        }
        self.ty.cast(to.ty)?;
        Ok(self)
    }
}

impl fmt::Debug for ArrayTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayTy")
            .field("ty", &self.ty)
            .field("shape", &self.shape.get())
            .finish()
    }
}

impl fmt::Display for ArrayTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array[{}, {}]", self.ty, self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{Primitive, PrimitiveTy, PrimitiveTy::Uint, iw};

    #[test]
    fn display_formats_one_dimensional_arrays() {
        let array = Array::new_unchecked(
            vec![
                Primitive::uint(1_u128),
                Primitive::uint(2_u128),
                Primitive::uint(3_u128),
            ],
            ArrayTy::new(Uint(iw(8)), shape![3]),
        );

        assert_eq!(array.to_string(), "{1, 2, 3}");
    }

    #[test]
    fn display_formats_nested_arrays() {
        let array = Array::new_unchecked(
            vec![
                Primitive::uint(1_u128),
                Primitive::uint(2_u128),
                Primitive::uint(3_u128),
                Primitive::uint(4_u128),
                Primitive::uint(5_u128),
                Primitive::uint(6_u128),
            ],
            ArrayTy::new(Uint(iw(8)), shape![2, 3]),
        );

        assert_eq!(array.to_string(), "{{1, 2, 3}, {4, 5, 6}}");
    }

    #[test]
    fn display_formats_zero_length_inner_dimensions() {
        let array = Array::new_unchecked(vec![], ArrayTy::new(PrimitiveTy::Duration, shape![2, 0]));

        assert_eq!(array.to_string(), "{{}, {}}");
    }

    #[test]
    fn ranks_run_from_one_to_seven() {
        // docs/types.rst: "a maximum of 7 total dimensions".
        assert!(ArrayDim::new(0).is_err());
        assert!(ArrayDim::new(1).is_ok());
        assert!(ArrayDim::new(7).is_ok());
        assert!(matches!(
            ArrayDim::new(8),
            Err(Error::InvalidRank {
                received: 8,
                max: 7
            })
        ));
        assert!(ArrayShape::new(vec![2; 7]).is_ok());
        assert!(ArrayShape::new(vec![2; 8]).is_err());
        assert!(ArrayShape::new(vec![]).is_err());
    }

    #[test]
    fn building_an_array_checks_the_element_count() {
        let ty = ArrayTy::new(Uint(iw(8)), shape![2, 2]);
        let err = Array::new(vec![Primitive::uint(1_u128); 3], ty).unwrap_err();
        assert_eq!(err.to_string(), "expected 4 array elements, got 3");
    }

    #[test]
    fn array_ty_display_formats_scalar_type_and_dims() {
        let ty = ArrayTy::new(Uint(iw(8)), shape![2, 3, 4]);

        assert_eq!(ty.to_string(), "array[uint[8], 2, 3, 4]");
    }

    #[test]
    fn array_ty_display_includes_zero_length_dimensions() {
        let ty = ArrayTy::new(PrimitiveTy::Duration, shape![2, 0]);

        assert_eq!(ty.to_string(), "array[duration, 2, 0]");
    }
}
