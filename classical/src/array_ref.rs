use std::fmt;

use crate::array::{Array, ArrayDim, ArrayShape, ArrayTy};
use crate::error::{Error, Result};
use crate::index::Index;
use crate::primitive::PrimitiveTy;
use crate::shared::Shared;
use crate::value::{Value, ValueTy};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// A reference to the part of `array` the subscript `indices` selects
    /// (one index per dimension), which must be an array itself.
    pub fn new(array: Shared<Array>, indices: Vec<Index>, access: RefAccess) -> Result<Self> {
        let array_ty = array.borrow()?.ty();
        let ValueTy::Array(viewed) = ValueTy::Array(array_ty).get(&indices)? else {
            return Err(Error::IndexOutOfBounds {
                value: Box::new(array_ty.into()),
                index: indices,
            });
        };
        let shape = ArrayRefShape::Fixed(viewed.shape());
        Ok(Self {
            ty: ArrayRefTy::new(array_ty.ty(), shape, access),
            array,
            indices,
        })
    }

    /// Copy out the (sub-)array this reference views.
    pub fn to_owned(&self) -> Result<Array> {
        match self.array.borrow()?.get(&self.indices)? {
            Value::Array(array) => Ok(array),
            other => Err(Error::unsupported_cast(other.ty(), self.ty.to_owned()?)),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// The shape of an array reference: fully known, or only its rank
/// (`#dim = N`, sizes supplied by the referenced array at run time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Whether arrays of the two shapes can line up element for element:
    /// equal when both are fully known, otherwise equal in rank (the sizes
    /// are then checked at run time).
    pub fn is_compatible(&self, other: &ArrayRefShape) -> bool {
        match (self, other) {
            (ArrayRefShape::Fixed(a), ArrayRefShape::Fixed(b)) => a == b,
            _ => self.dim() == other.dim(),
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// The type of the array a fixed-shape reference views. A `#dim`
    /// reference has no owned type: its sizes are only known at run time.
    pub fn to_owned(self) -> Result<ArrayTy> {
        match self.shape {
            ArrayRefShape::Fixed(shape) => Ok(ArrayTy::new(self.ty, shape)),
            ArrayRefShape::Dim(dim) => Err(Error::unsupported_cast(
                self,
                ArrayTy::new(self.ty, ArrayShape::new(vec![1; dim.get()])?),
            )),
        }
    }

    pub fn cast(self, ty: ArrayRefTy) -> Result<ArrayRefTy> {
        let fits = match (self.shape, ty.shape) {
            (ArrayRefShape::Fixed(s), ArrayRefShape::Fixed(t)) => s.total() == t.total(),
            (this, other) => this.dim() == other.dim(),
        };
        if !fits {
            return Err(Error::unsupported_cast(self, ty));
        }
        self.ty.cast(ty.ty)?;
        Ok(self)
    }
}

impl fmt::Display for ArrayRefTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} array[{}, {}]", self.access, self.ty, self.shape)
    }
}
