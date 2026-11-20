use std::fmt;

use serde::{Deserialize, Serialize};

use crate::array::{Array, ArrayDim, ArrayShape, ArrayTy};
use crate::error::{Error, Result};
use crate::index::Index;
use crate::primitive::PrimitiveTy;
use crate::shared::Shared;
use crate::value::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayRef {
    array: Shared<Array>,
    indices: Vec<Index>,
    ty: ArrayRefTy,
}

impl ArrayRef {
    #[inline]
    pub fn array(&self) -> &Shared<Array> {
        &self.array
    }

    #[inline]
    pub fn indices(&self) -> &[Index] {
        &self.indices
    }

    #[inline]
    pub fn ty(&self) -> ArrayRefTy {
        self.ty
    }

    pub fn new(array: Shared<Array>, indices: Vec<Index>, access: RefAccess) -> Result<Self> {
        let array_ty = array.borrow()?.ty();
        let mut shape = array_ty.shape();
        for idx in &indices {
            let new_shape = idx
                .shape(shape.get())
                .filter(|shape| !shape.is_empty())
                .ok_or_else(|| Error::IndexOutOfBounds {
                    value: Box::new(array_ty.into()),
                    index: indices.clone(),
                })?;
            shape = ArrayShape::new(new_shape)?;
        }
        let shape = ArrayRefShape::Fixed(shape);
        Ok(Self {
            ty: ArrayRefTy::new(array_ty.ty(), shape, access),
            array,
            indices,
        })
    }

    pub fn to_owned(&self) -> Result<Array> {
        match self.array.borrow()?.get(&self.indices)? {
            Value::Array(array) => Ok(array),
            Value::Scalar(scalar) => Err(Error::unsupported_cast(
                scalar.ty().into(),
                self.ty.to_owned()?.into(),
            )),
            Value::ArrayRef(array_ref) => Err(Error::unsupported_cast(
                array_ref.ty().into(),
                self.ty.to_owned()?.into(),
            )),
        }
    }

    pub fn cast(&self, ty: ArrayRefTy) -> Result<Array> {
        self.ty.cast(ty)?;
        let owned = self.to_owned()?;
        let shape = match ty.shape {
            ArrayRefShape::Fixed(shape) => shape,
            ArrayRefShape::Dim(_) => owned.ty().shape(),
        };
        owned.cast(ArrayTy::new(ty.ty, shape))
    }
}

impl From<Array> for ArrayRef {
    fn from(array: Array) -> Self {
        array.into_ref_mut()
    }
}

impl fmt::Display for ArrayRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // An ArrayRef carries indices into a larger backing array; show the
        // sub-array it refers to, not the whole store.
        write!(f, "{}", self.to_owned().map_err(|_| fmt::Error)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefAccess {
    Readonly,
    Mutable,
}

impl fmt::Display for RefAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefAccess::Readonly => write!(f, "readonly"),
            RefAccess::Mutable => write!(f, "mutable"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ArrayRefShape {
    Fixed(ArrayShape),
    Dim(ArrayDim),
}

impl From<ArrayShape> for ArrayRefShape {
    fn from(shape: ArrayShape) -> Self {
        Self::Fixed(shape)
    }
}

impl ArrayRefShape {
    pub fn dim(&self) -> ArrayDim {
        match self {
            ArrayRefShape::Fixed(shape) => shape.dim(),
            ArrayRefShape::Dim(dim) => *dim,
        }
    }
}

impl PartialEq for ArrayRefShape {
    fn eq(&self, other: &Self) -> bool {
        use ArrayRefShape::*;
        match (self, other) {
            (Fixed(s), Fixed(o)) => s == o,
            (Dim(s), Dim(o)) => s == o,
            (Fixed(f), Dim(d)) | (Dim(d), Fixed(f)) => f.dim() == *d,
        }
    }
}

impl PartialEq<ArrayDim> for ArrayRefShape {
    fn eq(&self, other: &ArrayDim) -> bool {
        self.dim() == *other
    }
}

impl PartialEq<ArrayRefShape> for ArrayDim {
    fn eq(&self, other: &ArrayRefShape) -> bool {
        other == self
    }
}

impl PartialEq<ArrayShape> for ArrayRefShape {
    fn eq(&self, other: &ArrayShape) -> bool {
        match self {
            ArrayRefShape::Fixed(shape) => shape == other,
            _ => false,
        }
    }
}

impl PartialEq<ArrayRefShape> for ArrayShape {
    fn eq(&self, other: &ArrayRefShape) -> bool {
        other == self
    }
}

impl fmt::Display for ArrayRefShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayRefShape::Fixed(shape) => write!(f, "{}", shape),
            ArrayRefShape::Dim(dim) => write!(f, "#dim = {}", dim.get()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArrayRefTy {
    ty: PrimitiveTy,
    shape: ArrayRefShape,
    access: RefAccess,
}

impl ArrayRefTy {
    #[inline]
    pub const fn new(ty: PrimitiveTy, shape: ArrayRefShape, access: RefAccess) -> Self {
        Self { ty, shape, access }
    }

    #[inline]
    pub fn shape(&self) -> ArrayRefShape {
        self.shape
    }

    #[inline]
    pub fn access(&self) -> RefAccess {
        self.access
    }

    #[inline]
    pub fn readonly(array: ArrayTy) -> Self {
        Self::new(
            array.ty(),
            ArrayRefShape::Fixed(array.shape()),
            RefAccess::Readonly,
        )
    }

    #[inline]
    pub fn mutable(array: ArrayTy) -> Self {
        Self::new(
            array.ty(),
            ArrayRefShape::Fixed(array.shape()),
            RefAccess::Mutable,
        )
    }

    #[inline]
    pub fn ty(&self) -> PrimitiveTy {
        self.ty
    }

    #[inline]
    pub fn with_ty(&self, ty: PrimitiveTy) -> Self {
        Self { ty, ..*self }
    }

    #[inline]
    pub fn with_shape(&self, shape: ArrayRefShape) -> Self {
        Self { shape, ..*self }
    }

    pub fn to_owned(self) -> Result<ArrayTy> {
        let shape = match self.shape {
            ArrayRefShape::Fixed(shape) => shape,
            ArrayRefShape::Dim(dim) => {
                return Err(Error::UnsupportedCast {
                    from: Box::new(self.ty.into()),
                    to: Box::new(
                        ArrayTy::new(self.ty, ArrayShape::new(vec![1; dim.get()])?).into(),
                    ),
                });
            }
        };
        Ok(ArrayTy::new(self.ty, shape))
    }

    pub fn cast(self, ty: ArrayRefTy) -> Result<ArrayRefTy> {
        match (self.shape, ty.shape) {
            (ArrayRefShape::Fixed(s), ArrayRefShape::Fixed(t)) => {
                if s.total() != t.total() {
                    return Err(Error::unsupported_cast(self.into(), ty.into()));
                }
            }
            (this, other) => {
                if this.dim() != other.dim() {
                    return Err(Error::unsupported_cast(self.into(), ty.into()));
                }
            }
        }
        self.ty.cast(ty.ty)?;
        Ok(self)
    }
}

impl From<ArrayTy> for ArrayRefTy {
    #[inline]
    fn from(array_ty: ArrayTy) -> Self {
        array_ty.as_ref_mut()
    }
}

impl fmt::Display for ArrayRefTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} array[{}, {}]", self.access, self.ty, self.shape)
    }
}
