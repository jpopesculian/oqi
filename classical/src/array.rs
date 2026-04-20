use std::fmt;

use crate::array_ref::{ArrayRef, ArrayRefShape, BaseArrayRefTy, RefAccess};
use crate::primitive::{Primitive, PrimitiveTy};
use crate::scalar::Scalar;
use crate::shared::Shared;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct BaseArray<V, T> {
    values: Vec<V>,
    ty: BaseArrayTy<T>,
}

impl<V, T> BaseArray<V, T> {
    #[inline]
    pub fn new_unchecked(values: Vec<V>, ty: BaseArrayTy<T>) -> Self {
        BaseArray { values, ty }
    }

    pub fn values(&self) -> &[V] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [V] {
        &mut self.values
    }
}

impl<V, T: Copy> BaseArray<V, T> {
    pub fn ty(&self) -> BaseArrayTy<T> {
        self.ty
    }
}

pub type Array = BaseArray<Primitive, PrimitiveTy>;

impl Array {
    pub fn new(mut values: Vec<Primitive>, ty: ArrayTy) -> Result<Self> {
        if values.len() != ty.shape().total() {
            return Err(Error::UnsupportedCast {
                from: Box::new(ty.with_shape(shape![values.len()]).into()),
                to: Box::new(ty.into()),
            });
        }
        for value in values.iter_mut() {
            *value = value.as_ty(ty.ty())?;
        }
        Ok(Array::new_unchecked(values, ty))
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
        Self::new(self.values().to_vec(), to)
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
        let next = Scalar::new_unchecked(*self.slice.get(self.index)?, self.ty);
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrayDim(usize);

impl ArrayDim {
    const MAX: usize = 16;
    #[inline]
    pub const fn new(dim: usize) -> Result<Self> {
        if dim == 0 || dim > 16 {
            return Err(Error::BadDimensions {
                received: dim,
                min: 1,
                max: 16,
            });
        }
        Ok(Self(dim))
    }

    #[inline]
    pub fn get(&self) -> usize {
        self.0
    }
}

pub fn adim(value: usize) -> ArrayDim {
    ArrayDim::new(value).unwrap()
}

impl PartialEq<usize> for ArrayDim {
    fn eq(&self, other: &usize) -> bool {
        self.get() == *other
    }
}

impl PartialEq<ArrayDim> for usize {
    fn eq(&self, other: &ArrayDim) -> bool {
        *self == other.get()
    }
}

impl PartialOrd<usize> for ArrayDim {
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        self.get().partial_cmp(other)
    }
}

impl PartialOrd<ArrayDim> for usize {
    fn partial_cmp(&self, other: &ArrayDim) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.get())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrayShape {
    shape: [usize; ArrayDim::MAX],
    dim: ArrayDim,
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
    pub fn new(shape: Vec<usize>) -> Result<Self> {
        let dim = ArrayDim::new(shape.len())?;
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

impl PartialEq<Vec<usize>> for ArrayShape {
    fn eq(&self, other: &Vec<usize>) -> bool {
        self.get() == other.as_slice()
    }
}

impl PartialEq<ArrayShape> for Vec<usize> {
    fn eq(&self, other: &ArrayShape) -> bool {
        self.as_slice() == other.get()
    }
}

impl<'a> PartialEq<&'a [usize]> for ArrayShape {
    fn eq(&self, other: &&'a [usize]) -> bool {
        self.get() == *other
    }
}

impl PartialEq<ArrayShape> for &'_ [usize] {
    fn eq(&self, other: &ArrayShape) -> bool {
        *self == other.get()
    }
}

pub fn ashape(values: Vec<usize>) -> ArrayShape {
    ArrayShape::new(values).unwrap()
}

macro_rules! shape {
    ($($dim:expr),+) => {
        crate::array::ashape(vec![$($dim),+])
    }
}
pub(crate) use shape;

#[derive(Clone, Copy, PartialEq)]
pub struct BaseArrayTy<T> {
    ty: T,
    shape: ArrayShape,
}

impl<T> BaseArrayTy<T> {
    #[inline]
    pub const fn new(ty: T, shape: ArrayShape) -> Self {
        Self { ty, shape }
    }

    #[inline]
    pub const fn shape(&self) -> ArrayShape {
        self.shape
    }
}

impl<T: Copy> BaseArrayTy<T> {
    #[inline]
    pub const fn with_ty(&self, ty: T) -> Self {
        Self { ty, ..*self }
    }

    #[inline]
    pub const fn with_shape(&self, shape: ArrayShape) -> Self {
        Self::new(self.ty, shape)
    }

    #[inline]
    pub const fn ty(&self) -> T {
        self.ty
    }

    #[inline]
    pub const fn as_ref(&self) -> BaseArrayRefTy<T> {
        BaseArrayRefTy::new(
            self.ty,
            ArrayRefShape::Fixed(self.shape),
            RefAccess::Readonly,
        )
    }

    #[inline]
    pub const fn as_ref_mut(&self) -> BaseArrayRefTy<T> {
        BaseArrayRefTy::new(
            self.ty,
            ArrayRefShape::Fixed(self.shape),
            RefAccess::Mutable,
        )
    }
}

pub type ArrayTy = BaseArrayTy<PrimitiveTy>;

impl ArrayTy {
    #[inline]
    pub fn cast(self, to: ArrayTy) -> Result<Self> {
        if self.shape.total() != to.shape.total() {
            return Err(Error::UnsupportedCast {
                from: Box::new(self.ty().into()),
                to: Box::new(to.into()),
            });
        }
        self.ty.cast(to.ty)?;
        Ok(self)
    }
}

impl<T: fmt::Debug> fmt::Debug for BaseArrayTy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayTy")
            .field("ty", &self.ty)
            .field("shape", &self.shape.get())
            .finish()
    }
}

impl<T: fmt::Display> fmt::Display for BaseArrayTy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array[{}, {}]", self.ty, self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{Primitive, PrimitiveTy, PrimitiveTy::Uint, bw};

    #[test]
    fn display_formats_one_dimensional_arrays() {
        let array = Array::new_unchecked(
            vec![
                Primitive::uint(1_u128),
                Primitive::uint(2_u128),
                Primitive::uint(3_u128),
            ],
            ArrayTy::new(Uint(bw(8)), shape![3]),
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
            ArrayTy::new(Uint(bw(8)), shape![2, 3]),
        );

        assert_eq!(array.to_string(), "{{1, 2, 3}, {4, 5, 6}}");
    }

    #[test]
    fn display_formats_zero_length_inner_dimensions() {
        let array = Array::new_unchecked(vec![], ArrayTy::new(PrimitiveTy::Duration, shape![2, 0]));

        assert_eq!(array.to_string(), "{{}, {}}");
    }

    #[test]
    fn array_ty_display_formats_scalar_type_and_dims() {
        let ty = ArrayTy::new(Uint(bw(8)), shape![2, 3, 4]);

        assert_eq!(ty.to_string(), "array[uint[8], 2, 3, 4]");
    }

    #[test]
    fn array_ty_display_includes_zero_length_dimensions() {
        let ty = ArrayTy::new(PrimitiveTy::Duration, shape![2, 0]);

        assert_eq!(ty.to_string(), "array[duration, 2, 0]");
    }
}
