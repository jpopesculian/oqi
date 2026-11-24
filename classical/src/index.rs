use std::fmt;

use crate::array::{Array, ArrayDim, ArrayShape, ArrayTy};
use crate::array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, RefAccess};
use crate::bitreg::BitReg;
use crate::scalar::Scalar;
use crate::value::{Value, ValueTy};
use crate::{
    Error, Result,
    primitive::{Primitive, PrimitiveTy, angle_bits, angle_from_bits, resize_int},
};

/// One step of indexing into an array dimension or a bit register.
/// Negative positions count from the end.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Index {
    Item(isize),
    /// `start`, `start + step`, … up to but excluding `end`.
    Slice {
        start: Option<isize>,
        step: Option<isize>,
        end: Option<isize>,
    },
    Select(Vec<isize>),
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bound = |f: &mut fmt::Formatter<'_>, bound: &Option<isize>| match bound {
            Some(i) => write!(f, "{i}"),
            None => Ok(()),
        };
        match self {
            Index::Item(i) => write!(f, "{i}"),
            Index::Slice { start, step, end } => {
                bound(f, start)?;
                f.write_str(":")?;
                if let Some(step) = step {
                    write!(f, "{step}:")?;
                }
                bound(f, end)
            }
            Index::Select(items) => {
                f.write_str("{")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("}")
            }
        }
    }
}

fn resolve_index(index: isize, dim: usize, exclusive: bool) -> Option<usize> {
    let dim = dim as isize;
    let bounds = if exclusive { dim + 1 } else { dim };
    if index >= bounds || index < -bounds {
        return None;
    }
    Some(if index < 0 {
        (dim + index) as usize
    } else {
        index as usize
    })
}

/// Resolve a slice against a dimension of size `dim` to concrete
/// `(start, step, end)` positions, `end` exclusive. `None` if a bound is out
/// of range, the step is zero, or the step points away from `end`.
fn resolve_slice(
    start: Option<isize>,
    step: Option<isize>,
    end: Option<isize>,
    dim: usize,
) -> Option<SliceIter> {
    let start = resolve_index(start.unwrap_or(0), dim, false)?;
    let end = resolve_index(end.unwrap_or(dim as isize), dim, true)?;
    let step = step.unwrap_or(1);
    let valid = match step {
        0 => false,
        1.. => start <= end,
        _ => end <= start,
    };
    valid.then_some(SliceIter { start, step, end })
}

impl Index {
    /// The positions this index selects from a dimension (or register) of
    /// size `dim`, in order.
    fn iter(&self, dim: usize) -> Option<IndexIter> {
        Some(match self {
            Self::Item(i) => IndexIter::Item(Some(resolve_index(*i, dim, false)?)),
            Self::Slice { start, step, end } => {
                IndexIter::Slice(resolve_slice(*start, *step, *end, dim)?)
            }
            Self::Select(indices) => {
                let resolved = indices
                    .iter()
                    .map(|&i| resolve_index(i, dim, false))
                    .collect::<Option<Vec<_>>>()?;
                IndexIter::Select(resolved.into_iter())
            }
        })
    }
}

enum IndexIter {
    Item(Option<usize>),
    Slice(SliceIter),
    Select(std::vec::IntoIter<usize>),
}

impl Iterator for IndexIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Item(opt) => opt.take(),
            Self::Slice(iter) => iter.next(),
            Self::Select(vec_iter) => vec_iter.next(),
        }
    }
}

impl ExactSizeIterator for IndexIter {
    fn len(&self) -> usize {
        match self {
            Self::Item(opt) => usize::from(opt.is_some()),
            Self::Slice(iter) => iter.len(),
            Self::Select(vec_iter) => vec_iter.len(),
        }
    }
}

/// The positions of a resolved slice (see [`resolve_slice`], which
/// guarantees a non-zero step pointing towards `end`).
struct SliceIter {
    start: usize,
    step: isize,
    end: usize,
}

impl Iterator for SliceIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if if self.step > 0 {
            self.start < self.end
        } else {
            self.end < self.start
        } {
            let current = self.start;
            self.start = (self.start as isize + self.step) as usize;
            Some(current)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for SliceIter {
    fn len(&self) -> usize {
        let range = self.end as isize - self.start as isize;
        ((range + self.step - self.step.signum()) / self.step) as usize
    }
}

/// flat index to multi-dimensional index
pub(crate) fn flat_to_multi(shape: &[usize], mut flat: usize) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        result[i] = flat % shape[i];
        flat /= shape[i];
    }
    result
}

fn index_error(value: ValueTy, indices: &[Index]) -> Error {
    Error::IndexOutOfBounds {
        value: Box::new(value),
        index: indices.to_vec(),
    }
}

/// What a subscript selects from an array: the shape of the selection and
/// the flat (row-major) positions of its elements.
struct Selection {
    shape: Vec<usize>,
    positions: Vec<usize>,
}

/// Apply the subscript `indices` to an array of shape `shape`, one index per
/// dimension (docs/types.rst, "Array concatenation and slicing"): an item
/// drops its dimension, a slice or selection keeps it at the size it
/// selects, and the dimensions the subscript does not reach are kept whole.
/// Indices past the last dimension are left to the caller (see
/// [`bit_indices`]).
fn select(shape: &[usize], indices: &[Index]) -> Option<Selection> {
    let mut selection = Selection {
        shape: Vec::new(),
        positions: vec![0],
    };
    for (k, &dim) in shape.iter().enumerate() {
        let picked: Vec<usize> = match indices.get(k) {
            Some(index) => {
                let picked = index.iter(dim)?;
                if !matches!(index, Index::Item(_)) {
                    selection.shape.push(picked.len());
                }
                picked.collect()
            }
            None => {
                selection.shape.push(dim);
                (0..dim).collect()
            }
        };
        selection.positions = selection
            .positions
            .iter()
            .flat_map(|&outer| picked.iter().map(move |&p| outer * dim + p))
            .collect();
    }
    Some(selection)
}

/// The part of a subscript past the `rank` dimensions of an array: it
/// indexes the bits of the element the rest of the subscript selects.
fn bit_indices(rank: usize, indices: &[Index]) -> &[Index] {
    &indices[indices.len().min(rank)..]
}

/// The single subscript that selects from an array of shape `shape` what
/// `next` selects from the selection of `base`: `next` indexes, in order,
/// the dimensions `base` keeps.
pub(crate) fn compose(shape: &[usize], base: &[Index], next: &[Index]) -> Option<Vec<Index>> {
    let mut next = next.iter();
    let mut composed = Vec::with_capacity(shape.len());
    for (k, &dim) in shape.iter().enumerate() {
        let kept = match base.get(k) {
            Some(dropped @ Index::Item(_)) => {
                composed.push(dropped.clone());
                continue;
            }
            kept => kept,
        };
        let Some(index) = next.next() else {
            match kept {
                Some(kept) => composed.push(kept.clone()),
                // Neither subscript reaches this dimension or any after it.
                None => break,
            }
            continue;
        };
        let positions: Vec<usize> = match kept {
            Some(kept) => kept.iter(dim)?.collect(),
            None => (0..dim).collect(),
        };
        let mut picked = index.iter(positions.len())?.map(|p| positions[p] as isize);
        composed.push(match index {
            Index::Item(_) => Index::Item(picked.next()?),
            _ => Index::Select(picked.collect()),
        });
    }
    composed.extend(next.cloned());
    Some(composed)
}

impl PrimitiveTy {
    fn get(&self, indices: &[Index]) -> Result<PrimitiveTy> {
        if indices.is_empty() {
            return Ok(*self);
        }

        let error = || index_error((*self).into(), indices);
        let (Some(width), [index]) = (self.bit_count(), indices) else {
            return Err(error());
        };
        let dim = width as usize;
        match index {
            Index::Item(i) => {
                resolve_index(*i, dim, false).ok_or_else(error)?;
                Ok(PrimitiveTy::Bit)
            }
            other => {
                let iter = other.iter(dim).ok_or_else(error)?;
                Ok(PrimitiveTy::BitReg(iter.len() as u32))
            }
        }
    }
}

impl ArrayTy {
    fn get(self, indices: &[Index]) -> Result<ValueTy> {
        if indices.is_empty() {
            return Ok(ValueTy::Array(self));
        }

        let value_ty = ValueTy::Array(self);
        let error = || index_error(value_ty, indices);
        let shape = self.shape();
        let selected = select(shape.dims(), indices).ok_or_else(error)?;
        let bits = bit_indices(shape.dims().len(), indices);
        if selected.shape.is_empty() {
            self.element_ty().get(bits).map(ValueTy::Scalar)
        } else if bits.is_empty() {
            let shape = ArrayShape::new(selected.shape)?;
            Ok(ValueTy::Array(ArrayTy::new(self.element_ty(), shape)))
        } else {
            Err(error())
        }
    }
}

impl ArrayRefTy {
    fn get(self, indices: &[Index]) -> Result<ValueTy> {
        if indices.is_empty() {
            return Ok(ValueTy::ArrayRef(self));
        }

        let value_ty = ValueTy::ArrayRef(self);
        let error = || index_error(value_ty, indices);
        match self.shape() {
            ArrayRefShape::Fixed(shape) => {
                match ArrayTy::new(self.element_ty(), shape).get(indices)? {
                    ValueTy::Scalar(ty) => Ok(ValueTy::Scalar(ty)),
                    ValueTy::Array(ty) => Ok(ValueTy::ArrayRef(ArrayRefTy::new(
                        ty.element_ty(),
                        ty.shape().into(),
                        self.access(),
                    ))),
                    ValueTy::ArrayRef(_) => {
                        unreachable!("array indexing on ArrayTy cannot yield ArrayRef")
                    }
                }
            }
            // Only the rank is known: count the dimensions the subscript
            // drops (see `select`).
            ArrayRefShape::Dim(dim) => {
                let rank = dim.get();
                let bits = bit_indices(rank, indices);
                let reached = &indices[..indices.len() - bits.len()];
                if reached
                    .iter()
                    .any(|index| matches!(index, Index::Slice { step: Some(0), .. }))
                {
                    return Err(error());
                }
                let dropped = reached
                    .iter()
                    .filter(|index| matches!(index, Index::Item(_)))
                    .count();
                if dropped == rank {
                    self.element_ty().get(bits).map(ValueTy::Scalar)
                } else if bits.is_empty() {
                    Ok(ValueTy::ArrayRef(ArrayRefTy::new(
                        self.element_ty(),
                        ArrayRefShape::Dim(ArrayDim::new(rank - dropped)?),
                        self.access(),
                    )))
                } else {
                    Err(error())
                }
            }
        }
    }
}

impl Scalar {
    /// The bits of a bit-indexable scalar (`bit[n]`, `uint[n]`, `int[n]`,
    /// `angle[n]`) as a register, with its width.
    fn to_bits(&self) -> Option<(BitReg, u32)> {
        let width = self.ty().bit_count()?;
        let reg = match self.value() {
            Primitive::Uint(v) => BitReg::from(*v),
            Primitive::BitReg(reg) => reg.clone(),
            Primitive::Int(v) => BitReg::from(*v as u128),
            Primitive::Angle(a) => BitReg::from(angle_bits(*a, width)),
            _ => return None,
        };
        Some((reg, width))
    }

    /// Inverse of [`to_bits`](Self::to_bits): this scalar's representation
    /// with `reg` as its bits.
    fn with_bits(&self, reg: BitReg) -> Option<Primitive> {
        Some(match (self.value(), self.ty()) {
            (Primitive::Uint(_), _) => Primitive::Uint(reg.as_u128()),
            (Primitive::BitReg(_), _) => Primitive::BitReg(reg),
            (Primitive::Int(_), PrimitiveTy::Int(bw)) => {
                Primitive::Int(resize_int(reg.as_u128() as i128, bw))
            }
            (Primitive::Angle(_), PrimitiveTy::Angle(bw)) => {
                Primitive::Angle(angle_from_bits(reg.as_u128(), bw.get()))
            }
            _ => return None,
        })
    }

    pub fn get(&self, indices: &[Index]) -> Result<Scalar> {
        if indices.is_empty() {
            return Ok(self.clone());
        }

        let value_ty = ValueTy::Scalar(self.ty());
        let error = || index_error(value_ty, indices);
        let (Some((reg, width)), [index]) = (self.to_bits(), indices) else {
            return Err(error());
        };
        let dim = width as usize;
        match index {
            Index::Item(i) => {
                let position = resolve_index(*i, dim, false).ok_or_else(error)?;
                Ok(Scalar::bit(reg.get_bit(position)))
            }
            other => {
                let iter = other.iter(dim).ok_or_else(error)?;
                let new_width = iter.len() as u32;
                Ok(Scalar::bitreg(reg.get_bits(iter), new_width))
            }
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Scalar) -> Result<()> {
        if indices.is_empty() {
            if !self.ty().is_compatible(value.ty()) {
                return Err(Error::unexpected_ty(self.ty(), value.ty()));
            }
            *self = value;
            return Ok(());
        }

        let value_ty = ValueTy::Scalar(self.ty());
        let error = || index_error(value_ty, indices);
        let (Some((mut reg, width)), [index]) = (self.to_bits(), indices) else {
            return Err(error());
        };
        let dim = width as usize;
        match index {
            Index::Item(i) => {
                let Primitive::Bit(bit) = value.value() else {
                    return Err(Error::unexpected_ty(PrimitiveTy::Bit, value.ty()));
                };
                let position = resolve_index(*i, dim, false).ok_or_else(error)?;
                reg.set_bit(position, *bit);
            }
            other => {
                let iter = other.iter(dim).ok_or_else(error)?;
                let mismatch = || Error::unexpected_ty(self.ty(), value.ty());
                // Registers and integers supply bits; an angle needs a cast.
                let (bits, value_width) = value
                    .to_bits()
                    .filter(|_| !matches!(value.ty(), PrimitiveTy::Angle(_)))
                    .ok_or_else(mismatch)?;
                if iter.len() != value_width as usize {
                    return Err(mismatch());
                }
                let bits = bits.iter_bits().take(value_width as usize);
                reg.set_bits(iter.zip(bits));
            }
        }

        let new_value = self.with_bits(reg).ok_or_else(error)?;
        *self = Scalar::new_unchecked(new_value, self.ty());
        Ok(())
    }
}

impl Array {
    pub fn get(&self, indices: &[Index]) -> Result<Value> {
        if indices.is_empty() {
            return Ok(Value::Array(self.clone()));
        }

        let value_ty = ValueTy::Array(self.ty());
        let error = || index_error(value_ty, indices);
        let ty = self.ty().element_ty();
        let shape = self.ty().shape();
        let selected = select(shape.dims(), indices).ok_or_else(error)?;
        let bits = bit_indices(shape.dims().len(), indices);
        let mut values = selected.positions.iter().map(|&p| self.values()[p].clone());
        if selected.shape.is_empty() {
            let element = Scalar::new_unchecked(values.next().ok_or_else(error)?, ty);
            element.get(bits).map(Value::Scalar)
        } else if bits.is_empty() {
            let shape = ArrayShape::new(selected.shape)?;
            Ok(Value::Array(Array::new_unchecked(
                values.collect(),
                ArrayTy::new(ty, shape),
            )))
        } else {
            Err(error())
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Value) -> Result<()> {
        if indices.is_empty() {
            return match value {
                Value::Array(a) => {
                    *self = a.cast(self.ty())?;
                    Ok(())
                }
                Value::ArrayRef(ar) => {
                    *self = ar.cast(self.ty().ref_ty(RefAccess::Mutable))?;
                    Ok(())
                }
                Value::Scalar(s) => Err(Error::unsupported_cast(s.ty(), self.ty())),
            };
        }

        let value_ty = ValueTy::Array(self.ty());
        let error = || index_error(value_ty, indices);
        let ty = self.ty().element_ty();
        let shape = self.ty().shape();
        let selected = select(shape.dims(), indices).ok_or_else(error)?;
        let bits = bit_indices(shape.dims().len(), indices);

        // One element: replace it, or the bits of it the subscript goes on
        // to select.
        if selected.shape.is_empty() {
            let Value::Scalar(value) = value else {
                return Err(Error::unexpected_ty(ty, value.ty()));
            };
            let flat = *selected.positions.first().ok_or_else(error)?;
            let element = if bits.is_empty() {
                value
            } else {
                let mut element = Scalar::new_unchecked(self.values()[flat].clone(), ty);
                element.set(bits, value)?;
                element
            };
            self.values_mut()[flat] = element.into_value();
            return Ok(());
        }

        let source = match value {
            Value::Array(a) if bits.is_empty() => a,
            Value::ArrayRef(ar) if bits.is_empty() => ar.to_array()?,
            _ => return Err(error()),
        };
        if source.ty().shape().dims() != selected.shape.as_slice() {
            return Err(error());
        }
        let elements = self.values_mut();
        for (&flat, element) in selected.positions.iter().zip(source.values()) {
            elements[flat] = element.clone();
        }
        Ok(())
    }
}

impl ArrayRef {
    /// The subscript on the referenced array that `indices` on this
    /// reference amounts to.
    fn through(&self, indices: &[Index]) -> Result<Vec<Index>> {
        let shape = self.array().borrow()?.ty().shape();
        compose(shape.dims(), self.indices(), indices)
            .ok_or_else(|| index_error(self.ty().into(), indices))
    }

    pub fn get(&self, indices: &[Index]) -> Result<Value> {
        if indices.is_empty() {
            return Ok(Value::ArrayRef(self.clone()));
        }

        let through = self.through(indices)?;
        match self.ty().get(indices)? {
            ValueTy::Scalar(_) => self.array().borrow()?.get(&through),
            ValueTy::ArrayRef(_) => {
                ArrayRef::new(self.array().clone(), through, self.ty().access())
                    .map(Value::ArrayRef)
            }
            ValueTy::Array(_) => unreachable!("array ref indexing never yields owned arrays"),
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Value) -> Result<()> {
        if matches!(self.ty().access(), RefAccess::Readonly) {
            return Err(Error::ReadOnly);
        }
        let through = self.through(indices)?;
        // The value may be a reference into this same array (`a[0] = a[1]`):
        // copy it out before borrowing the array to write.
        let value = match value {
            Value::ArrayRef(source) => Value::Array(source.to_array()?),
            value => value,
        };
        self.array().borrow_mut()?.set(&through, value)?;
        Ok(())
    }
}

impl Value {
    /// Index into the value. An owned array yields an owned element or
    /// sub-array (a copy: writing to it does not write to `self`); an array
    /// reference yields a narrower reference to the same array.
    pub fn get(&self, indices: &[Index]) -> Result<Value> {
        if indices.is_empty() {
            return Ok(self.clone());
        }

        match self {
            Value::Scalar(s) => s.get(indices).map(Value::Scalar),
            Value::Array(a) => a.get(indices),
            Value::ArrayRef(ar) => ar.get(indices),
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Value) -> Result<()> {
        if indices.is_empty() {
            if !self.ty().is_compatible(&value.ty()) {
                return Err(Error::unexpected_ty(self.ty(), value.ty()));
            }
            *self = value;
            return Ok(());
        }

        match self {
            Value::Scalar(s) => {
                let Value::Scalar(value) = value else {
                    return Err(Error::unexpected_ty(s.ty(), value.ty()));
                };
                s.set(indices, value)
            }
            Value::Array(a) => a.set(indices, value),
            Value::ArrayRef(ar) => ar.set(indices, value),
        }
    }
}

impl ValueTy {
    pub fn get(&self, indices: &[Index]) -> Result<ValueTy> {
        if indices.is_empty() {
            return Ok(*self);
        }

        match self {
            ValueTy::Scalar(s) => s.get(indices).map(ValueTy::Scalar),
            ValueTy::Array(a) => a.get(indices),
            ValueTy::ArrayRef(a) => a.get(indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayShape, shape};
    use crate::primitive::iw;

    impl Index {
        /// The shape left by applying this index to the leading dimension.
        fn shape(&self, shape: &[usize]) -> Option<Vec<usize>> {
            select(shape, std::slice::from_ref(self)).map(|selected| selected.shape)
        }
    }

    // --- Item ---

    #[test]
    fn item_removes_first_dim() {
        assert_eq!(Index::Item(0).shape(&[3, 4, 5]), Some(vec![4, 5]));
    }

    #[test]
    fn item_on_1d_returns_scalar_shape() {
        assert_eq!(Index::Item(0).shape(&[5]), Some(vec![]));
    }

    // --- Slice defaults (start=None, step=None, end=None → full slice) ---

    #[test]
    fn slice_full_preserves_shape() {
        let idx = Index::Slice {
            start: None,
            step: None,
            end: None,
        };
        // start=0, end=4, step=1 → (4-0)/1 = 4
        assert_eq!(idx.shape(&[4, 3]), Some(vec![4, 3]));
    }

    #[test]
    fn slice_with_step() {
        let idx = Index::Slice {
            start: Some(0),
            step: Some(2),
            end: Some(4),
        };
        // (4-0)/2 = 2
        assert_eq!(idx.shape(&[5, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn slice_step_ceils() {
        // elements [0,2,4] → 3
        let idx = Index::Slice {
            start: Some(0),
            step: Some(2),
            end: Some(5),
        };
        assert_eq!(idx.shape(&[6]), Some(vec![3]));
    }

    #[test]
    fn slice_negative_start() {
        // dim=5, start=-2 → resolved to 3, end defaults to 5, step=1
        // (5-3)/1 = 2
        let idx = Index::Slice {
            start: Some(-2),
            step: None,
            end: None,
        };
        assert_eq!(idx.shape(&[5]), Some(vec![2]));
    }

    #[test]
    fn slice_negative_end() {
        // dim=5, start=0, end=-1 → resolved to 4, step=1
        // (4-0)/1 = 4
        let idx = Index::Slice {
            start: None,
            step: None,
            end: Some(-1),
        };
        assert_eq!(idx.shape(&[5, 2]), Some(vec![4, 2]));
    }

    #[test]
    fn slice_negative_step() {
        // start=4, end=0, step=-1 → is_valid_slice(4,-1,0) → end<=start → 0<=4 → true
        // (0-4)/-1 = 4
        let idx = Index::Slice {
            start: Some(4),
            step: Some(-1),
            end: Some(0),
        };
        assert_eq!(idx.shape(&[5]), Some(vec![4]));
    }

    #[test]
    fn slice_step_zero_returns_none() {
        let idx = Index::Slice {
            start: Some(0),
            step: Some(0),
            end: Some(3),
        };
        assert_eq!(idx.shape(&[5]), None);
    }

    #[test]
    fn slice_invalid_forward_range_returns_none() {
        // step=1 but start > end → invalid
        let idx = Index::Slice {
            start: Some(3),
            step: Some(1),
            end: Some(1),
        };
        assert_eq!(idx.shape(&[5]), None);
    }

    #[test]
    fn slice_invalid_backward_range_returns_none() {
        // step=-1 but start < end → invalid
        let idx = Index::Slice {
            start: Some(1),
            step: Some(-1),
            end: Some(3),
        };
        assert_eq!(idx.shape(&[5]), None);
    }

    #[test]
    fn slice_out_of_bounds_returns_none() {
        // start=6 with dim=5 → resolve_index(6,5) → 6 > 5 → None
        let idx = Index::Slice {
            start: Some(6),
            step: None,
            end: None,
        };
        assert_eq!(idx.shape(&[5]), None);
    }

    #[test]
    fn slice_negative_out_of_bounds_returns_none() {
        // start=-6 with dim=5 → |-6| = 6 > 5 → None
        let idx = Index::Slice {
            start: Some(-6),
            step: None,
            end: None,
        };
        assert_eq!(idx.shape(&[5]), None);
    }

    // --- Select ---

    #[test]
    fn select_replaces_first_dim() {
        assert_eq!(Index::Select(vec![0, 2]).shape(&[5, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn select_empty_gives_zero_dim() {
        assert_eq!(Index::Select(vec![]).shape(&[5, 3]), Some(vec![0, 3]));
    }

    #[test]
    fn select_out_of_bounds_returns_none() {
        assert_eq!(Index::Select(vec![0, 100]).shape(&[5, 3]), None);
    }

    #[test]
    fn select_negative_out_of_bounds_returns_none() {
        assert_eq!(Index::Select(vec![-6]).shape(&[5]), None);
    }

    // --- Item bounds checking ---

    #[test]
    fn item_out_of_bounds_returns_none() {
        assert_eq!(Index::Item(100).shape(&[3, 4]), None);
    }

    #[test]
    fn item_negative_out_of_bounds_returns_none() {
        assert_eq!(Index::Item(-4).shape(&[3, 4]), None);
    }

    #[test]
    fn item_negative_valid() {
        assert_eq!(Index::Item(-1).shape(&[3, 4]), Some(vec![4]));
    }

    // --- get/set helpers ---

    fn arr(values: &[u8], shape: ArrayShape) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::uint(v as u128)).collect(),
            ArrayTy::new(PrimitiveTy::Uint(iw(8)), shape),
        ))
    }

    fn sc(v: u8) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(v as u128),
            PrimitiveTy::Uint(iw(8)),
        ))
    }

    fn to_u8s(val: &Value) -> Vec<u8> {
        match val {
            Value::Array(a) => a
                .values()
                .iter()
                .map(|s| s.as_uint(iw(8)).unwrap() as u8)
                .collect(),
            Value::ArrayRef(ar) => ar
                .to_array()
                .unwrap()
                .values()
                .iter()
                .map(|s| s.as_uint(iw(8)).unwrap() as u8)
                .collect(),
            Value::Scalar(s) => vec![s.value().as_uint(iw(8)).unwrap() as u8],
        }
    }

    // --- get tests ---

    #[test]
    fn get_item_returns_row() {
        let val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let result = val.get(&[Index::Item(0)]).unwrap();
        assert!(matches!(result, Value::Array(_)));
        assert_eq!(to_u8s(&result), vec![0, 1, 2]);
    }

    #[test]
    fn get_item_second_row() {
        let val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let result = val.get(&[Index::Item(1)]).unwrap();
        assert!(matches!(result, Value::Array(_)));
        assert_eq!(to_u8s(&result), vec![3, 4, 5]);
    }

    #[test]
    fn array_ref_displays_indexed_sub_array() {
        // An ArrayRef produced by indexing must display the sub-array it
        // refers to, not its whole backing store.
        let Value::Array(array) = arr(&[1, 2, 3, 4], shape![2, 2]) else {
            unreachable!()
        };
        let row0 = Value::ArrayRef(array.into_ref())
            .get(&[Index::Item(0)])
            .unwrap();
        assert!(matches!(row0, Value::ArrayRef(_)));
        assert_eq!(row0.to_string(), "{1, 2}");

        let val3 = arr(&[1, 2, 3, 4, 5, 6], shape![3, 2]);
        let row1 = val3.get(&[Index::Item(1)]).unwrap();
        assert_eq!(row1.to_string(), "{3, 4}");
    }

    #[test]
    fn indexing_an_owned_array_yields_an_owned_copy() {
        // The row is a value in its own right: writing to it must not look
        // like (or silently fail to be) a write into the array it came from.
        let val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let mut row = val.get(&[Index::Item(1)]).unwrap();
        assert!(matches!(row, Value::Array(_)));
        assert_eq!(row.ty(), val.ty().get(&[Index::Item(1)]).unwrap());

        row.set(&[Index::Item(0)], sc(9)).unwrap();
        assert_eq!(to_u8s(&row), vec![9, 4, 5]);
        assert_eq!(to_u8s(&val), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn indexing_an_array_ref_yields_a_ref_that_writes_through() {
        let Value::Array(array) = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]) else {
            unreachable!()
        };
        let whole = Value::ArrayRef(array.into_ref_mut());
        let mut row = whole.get(&[Index::Item(1)]).unwrap();
        assert!(matches!(row, Value::ArrayRef(_)));

        row.set(&[Index::Item(0)], sc(9)).unwrap();
        assert_eq!(to_u8s(&whole), vec![0, 1, 2, 9, 4, 5]);
    }

    #[test]
    fn a_reference_can_be_assigned_from_the_array_it_views() {
        let Value::Array(array) = arr(&[0, 1, 2, 3], shape![2, 2]) else {
            unreachable!()
        };
        let mut whole = Value::ArrayRef(array.into_ref_mut());
        let row = whole.get(&[Index::Item(1)]).unwrap();
        whole.set(&[Index::Item(0)], row).unwrap();
        assert_eq!(to_u8s(&whole), vec![2, 3, 2, 3]);
    }

    /// Rows 1 and 2 of a three-row array.
    fn last_two_rows() -> Index {
        Index::Slice {
            start: Some(1),
            step: None,
            end: Some(3),
        }
    }

    #[test]
    fn an_index_after_a_slice_indexes_the_next_dimension() {
        // {{0, 1}, {2, 3}, {4, 5}}[1:2, 0] is column 0 of rows 1 and 2.
        let mut val = arr(&[0, 1, 2, 3, 4, 5], shape![3, 2]);
        let column = [last_two_rows(), Index::Item(0)];
        let result = val.get(&column).unwrap();
        assert_eq!(result.ty(), val.ty().get(&column).unwrap());
        assert_eq!(to_u8s(&result), vec![2, 4]);

        val.set(&[last_two_rows(), Index::Item(1)], arr(&[8, 9], shape![2]))
            .unwrap();
        assert_eq!(to_u8s(&val), vec![0, 1, 2, 8, 4, 9]);

        // A second index never re-indexes a dimension the first one kept.
        let one_d = arr(&[0, 1, 2], shape![3]);
        assert!(one_d.get(&[last_two_rows(), Index::Item(0)]).is_err());
    }

    #[test]
    fn a_reference_to_a_slice_is_indexed_within_the_slice() {
        let Value::Array(array) = arr(&[0, 1, 2, 3, 4, 5], shape![3, 2]) else {
            unreachable!()
        };
        let whole = Value::ArrayRef(array.into_ref_mut());
        let mut tail = whole.get(&[last_two_rows()]).unwrap();

        // Row 0 of the slice is row 1 of the array.
        tail.set(&[Index::Item(0), Index::Item(1)], sc(9)).unwrap();
        assert_eq!(to_u8s(&whole), vec![0, 1, 2, 9, 4, 5]);

        let column = tail.get(&[Index::Select(vec![1, 0]), Index::Item(0)]);
        assert_eq!(to_u8s(&column.unwrap()), vec![4, 2]);
        let row = tail.get(&[Index::Item(-1)]).unwrap();
        assert_eq!(to_u8s(&row.get(&[Index::Item(0)]).unwrap()), vec![4]);
    }

    #[test]
    fn get_two_items_returns_scalar() {
        let val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let result = val.get(&[Index::Item(0), Index::Item(2)]).unwrap();
        assert!(matches!(result, Value::Scalar(_)));
        assert_eq!(to_u8s(&result), vec![2]);
    }

    #[test]
    fn get_slice() {
        let val = arr(&[0, 1, 2, 3, 4], shape![5]);
        let idx = Index::Slice {
            start: Some(1),
            step: None,
            end: Some(4),
        };
        let result = val.get(&[idx]).unwrap();
        assert!(matches!(result, Value::Array(_)));
        assert_eq!(to_u8s(&result), vec![1, 2, 3]);
    }

    #[test]
    fn get_select() {
        let val = arr(&[0, 1, 2, 3, 4], shape![5]);
        let result = val.get(&[Index::Select(vec![3, 1, 4])]).unwrap();
        assert!(matches!(result, Value::Array(_)));
        assert_eq!(to_u8s(&result), vec![3, 1, 4]);
    }

    #[test]
    fn get_on_float_scalar_returns_none() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(1.0),
            PrimitiveTy::Float(crate::primitive::FloatWidth::F64),
        ));
        assert!(v.get(&[Index::Item(0)]).is_err());
    }

    #[test]
    fn get_empty_indices_returns_clone() {
        let val = arr(&[0, 1, 2], shape![3]);
        assert_eq!(to_u8s(&val.get(&[]).unwrap()), vec![0, 1, 2]);
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let val = arr(&[0, 1, 2], shape![3]);
        assert!(val.get(&[Index::Item(5)]).is_err());
    }

    #[test]
    fn scalar_get_direct_reads_bits() {
        let scalar = Scalar::new_unchecked(Primitive::uint(0b1010), PrimitiveTy::Uint(iw(4)));
        let result = scalar.get(&[Index::Item(1)]).unwrap();

        assert!(matches!(result.value(), Primitive::Bit(true)));
    }

    // --- set tests ---

    #[test]
    fn set_item_replaces_row() {
        let mut val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        assert!(
            val.set(&[Index::Item(0)], arr(&[7, 8, 9], shape![3]))
                .is_ok()
        );
        assert_eq!(to_u8s(&val), vec![7, 8, 9, 3, 4, 5]);
    }

    #[test]
    fn set_scalar_element() {
        let mut val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        assert!(val.set(&[Index::Item(1), Index::Item(2)], sc(99)).is_ok());
        assert_eq!(to_u8s(&val), vec![0, 1, 2, 3, 4, 99]);
    }

    #[test]
    fn set_wrong_shape_returns_false() {
        let mut val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        assert!(val.set(&[Index::Item(0)], arr(&[7, 8], shape![2])).is_err());
        assert_eq!(to_u8s(&val), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn set_on_scalar_returns_false() {
        let mut val = sc(42);
        assert!(val.set(&[Index::Item(0)], sc(1)).is_err());
    }

    #[test]
    fn set_empty_indices_replaces_value() {
        let mut val = arr(&[0, 1, 2], shape![3]);
        assert!(val.set(&[], arr(&[7, 8, 9], shape![3])).is_ok());
        assert_eq!(to_u8s(&val), vec![7, 8, 9]);
    }

    #[test]
    fn set_select() {
        let mut val = arr(&[0, 1, 2, 3, 4], shape![5]);
        assert!(
            val.set(&[Index::Select(vec![4, 0])], arr(&[40, 99], shape![2]))
                .is_ok()
        );
        assert_eq!(to_u8s(&val), vec![99, 1, 2, 3, 40]);
    }

    #[test]
    fn array_set_direct_updates_element() {
        let mut array = Array::new_unchecked(
            vec![
                Primitive::uint(0_u128),
                Primitive::uint(1_u128),
                Primitive::uint(2_u128),
            ],
            ArrayTy::new(PrimitiveTy::Uint(iw(8)), shape![3]),
        );

        assert!(array.set(&[Index::Item(1)], sc(99)).is_ok());
        assert_eq!(
            array
                .values()
                .iter()
                .map(|v| v.as_uint(iw(8)).unwrap() as u8)
                .collect::<Vec<_>>(),
            vec![0, 99, 2]
        );
    }

    #[test]
    fn get_set_roundtrip() {
        let mut val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let row = val.get(&[Index::Item(1)]).unwrap();
        assert_eq!(to_u8s(&row), vec![3, 4, 5]);
        assert!(val.set(&[Index::Item(0)], row).is_ok());
        assert_eq!(to_u8s(&val), vec![3, 4, 5, 3, 4, 5]);
    }

    // --- bit indexing ---

    fn bit_val(b: bool) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::Bit(b), PrimitiveTy::Bit))
    }

    #[test]
    fn get_bit_from_uint() {
        // 0b1010 = 10
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(8)),
        ));
        assert!(matches!(
            v.get(&[Index::Item(0)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(false))
        ));
        assert!(matches!(
            v.get(&[Index::Item(1)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(true))
        ));
        assert!(matches!(
            v.get(&[Index::Item(3)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(true))
        ));
    }

    #[test]
    fn get_bit_from_bitreg() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0101),
            PrimitiveTy::BitReg(4),
        ));
        assert!(matches!(
            v.get(&[Index::Item(0)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(true))
        ));
        assert!(matches!(
            v.get(&[Index::Item(1)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(false))
        ));
    }

    #[test]
    fn get_bit_from_int() {
        // -1 as Int(8) = 0xFF, all bits set
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::int(-1),
            PrimitiveTy::Int(iw(8)),
        ));
        for i in 0..8 {
            assert!(matches!(
                v.get(&[Index::Item(i)]),
                Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(true))
            ));
        }
    }

    #[test]
    fn get_bit_negative_index() {
        // 0b1000 in 4 bits: bit -1 (= bit 3) is 1
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1000),
            PrimitiveTy::Uint(iw(4)),
        ));
        assert!(matches!(
            v.get(&[Index::Item(-1)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(true))
        ));
        assert!(matches!(
            v.get(&[Index::Item(-4)]),
            Ok(Value::Scalar(s)) if matches!(s.value(), Primitive::Bit(false))
        ));
    }

    #[test]
    fn get_bit_out_of_bounds() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF),
            PrimitiveTy::Uint(iw(8)),
        ));
        assert!(v.get(&[Index::Item(8)]).is_err());
        assert!(v.get(&[Index::Item(-9)]).is_err());
    }

    #[test]
    fn get_bit_slice_uint() {
        // 0b1010_0110, bits 1..5 = bits 1,2,3,4 = 1,1,0,0 → 0b0011
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_0110),
            PrimitiveTy::Uint(iw(8)),
        ));
        let r = v
            .get(&[Index::Slice {
                start: Some(1),
                step: None,
                end: Some(5),
            }])
            .unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 4);
                assert_eq!(*val, 0b0011);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_slice_bitreg() {
        // 0b1100 in 4 bits, bits 0..4 step 2 = bits 0,2 = 0,1 → 0b10
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1100),
            PrimitiveTy::BitReg(4),
        ));
        let r = v
            .get(&[Index::Slice {
                start: Some(0),
                step: Some(2),
                end: Some(4),
            }])
            .unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 2);
                assert_eq!(*val, 0b10);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_slice_full() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));
        let r = v
            .get(&[Index::Slice {
                start: None,
                step: None,
                end: None,
            }])
            .unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 4);
                assert_eq!(*val, 0b1010);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_slice_reversed() {
        // 0b1010 in 4 bits, bits 3 down to 0 (step -1) = bits 3,2,1,0 = 1,0,1,0
        // result bit 0=1, bit 1=0, bit 2=1, bit 3=0 → 0b0101
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));
        let r = v
            .get(&[Index::Slice {
                start: Some(3),
                step: Some(-1),
                end: Some(0),
            }])
            .unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                // bits at positions 3,2,1 → values 1,0,1 → result 0b101
                assert_eq!(w, 3);
                assert_eq!(*val, 0b101);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_select_uint() {
        // 0b1010, select bits [3, 0] → bit 3=1, bit 0=0 → result bit 0=1, bit 1=0 → 0b01
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));
        let r = v.get(&[Index::Select(vec![3, 0])]).unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 2);
                assert_eq!(*val, 0b01);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_select_bitreg() {
        // 0b1111, select bits [0, 2] → both 1 → 0b11
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1111),
            PrimitiveTy::BitReg(4),
        ));
        let r = v.get(&[Index::Select(vec![0, 2])]).unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 2);
                assert_eq!(*val, 0b11);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_select_empty_yields_zero_width_bitreg() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF),
            PrimitiveTy::Uint(iw(8)),
        ));
        match v.get(&[Index::Select(vec![])]).unwrap() {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), PrimitiveTy::BitReg(0)));
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn get_bit_select_out_of_bounds() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFF),
            PrimitiveTy::Uint(iw(8)),
        ));
        assert!(v.get(&[Index::Select(vec![0, 8])]).is_err());
    }

    #[test]
    fn set_bit_on_uint() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0),
            PrimitiveTy::Uint(iw(4)),
        ));
        assert!(v.set(&[Index::Item(1)], bit_val(true)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(*val, 0b0010);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_clear_on_uint() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1111),
            PrimitiveTy::Uint(iw(4)),
        ));
        assert!(v.set(&[Index::Item(2)], bit_val(false)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(*val, 0b1011);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_on_bitreg() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b0000),
            PrimitiveTy::BitReg(4),
        ));
        assert!(v.set(&[Index::Item(3)], bit_val(true)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected bitreg")
                };
                assert_eq!(*val, 0b1000);
            }
            _ => panic!("expected bitreg"),
        }
    }

    #[test]
    fn set_bit_on_int_sign_extends() {
        // Int(8) = 0, set bit 7 (sign bit) → should become -128
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::int(0),
            PrimitiveTy::Int(iw(8)),
        ));
        assert!(v.set(&[Index::Item(7)], bit_val(true)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Int(val) = s.value() else {
                    panic!("expected int")
                };
                assert_eq!(*val, -128);
            }
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn set_bit_on_int_clear_sign() {
        // Int(8) = -1 (0xFF), clear bit 7 → 0x7F = 127
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::int(-1),
            PrimitiveTy::Int(iw(8)),
        ));
        assert!(v.set(&[Index::Item(7)], bit_val(false)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Int(val) = s.value() else {
                    panic!("expected int")
                };
                assert_eq!(*val, 127);
            }
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn set_bit_with_non_bit_value_returns_false() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0),
            PrimitiveTy::Uint(iw(8)),
        ));
        assert!(v.set(&[Index::Item(0)], sc(1)).is_err());
    }

    #[test]
    fn set_bit_out_of_bounds_returns_false() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0),
            PrimitiveTy::Uint(iw(8)),
        ));
        assert!(v.set(&[Index::Item(8)], bit_val(true)).is_err());
    }

    #[test]
    fn get_set_bit_roundtrip() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));
        let bit = v.get(&[Index::Item(1)]).unwrap();
        // bit 1 of 0b1010 is true
        assert!(matches!(
            &bit,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(true))
        ));
        // set bit 0 to that value
        assert!(v.set(&[Index::Item(0)], bit).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(*val, 0b1011);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_slice_uint() {
        // 0b0000_0000, set bits 2..6 to 0b1010 (4-bit BitReg)
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0),
            PrimitiveTy::Uint(iw(8)),
        ));
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1010),
            PrimitiveTy::BitReg(4),
        ));
        assert!(
            v.set(
                &[Index::Slice {
                    start: Some(2),
                    step: None,
                    end: Some(6),
                }],
                src
            )
            .is_ok()
        );
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                // bits 2,3,4,5 set to 0,1,0,1 → 0b00101000
                assert_eq!(*val, 0b00101000);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_select_uint() {
        // 0b0000, set bits [0, 3] to 0b11 (2-bit BitReg)
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0),
            PrimitiveTy::Uint(iw(4)),
        ));
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b11),
            PrimitiveTy::BitReg(2),
        ));
        assert!(v.set(&[Index::Select(vec![0, 3])], src).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(*val, 0b1001);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_slice_bitreg() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b1111),
            PrimitiveTy::BitReg(4),
        ));
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b00),
            PrimitiveTy::BitReg(2),
        ));
        assert!(
            v.set(
                &[Index::Slice {
                    start: Some(1),
                    step: None,
                    end: Some(3),
                }],
                src
            )
            .is_ok()
        );
        match &v {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected bitreg")
                };
                assert_eq!(*val, 0b1001);
            }
            _ => panic!("expected bitreg"),
        }
    }

    #[test]
    fn set_bit_slice_wrong_width_returns_false() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0),
            PrimitiveTy::Uint(iw(8)),
        ));
        // Slice selects 4 bits but value is 3-bit
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b111),
            PrimitiveTy::BitReg(3),
        ));
        assert!(
            v.set(
                &[Index::Slice {
                    start: Some(0),
                    step: None,
                    end: Some(4),
                }],
                src
            )
            .is_err()
        );
    }

    #[test]
    fn angle_bit_index_set_and_get() {
        // An angle[4] starts at numerator 0. Setting bit 0 makes the
        // numerator 1 (stored left-aligned as 1 << 124); reads round-trip.
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(0)),
            PrimitiveTy::Angle(iw(4)),
        ));
        let one_bit = Value::Scalar(Scalar::new_unchecked(
            Primitive::bit(true),
            PrimitiveTy::Bit,
        ));
        v.set(&[Index::Item(0)], one_bit).unwrap();
        match &v {
            Value::Scalar(s) => match s.value() {
                Primitive::Angle(a) => assert_eq!(a.0, 1u128 << 124),
                _ => panic!("expected angle primitive"),
            },
            _ => panic!("expected scalar"),
        }
        let read = |i| match v.get(&[Index::Item(i)]).unwrap() {
            Value::Scalar(s) => s.value().as_bit(),
            _ => panic!("expected scalar"),
        };
        assert!(read(0), "bit 0 should read back set");
        assert!(!read(1), "bit 1 should be unset");
    }

    #[test]
    fn get_set_slice_roundtrip() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_0110),
            PrimitiveTy::Uint(iw(8)),
        ));
        let slice = v
            .get(&[Index::Slice {
                start: Some(0),
                step: None,
                end: Some(4),
            }])
            .unwrap();
        // Lower 4 bits of 0b10100110 = 0b0110, returned as BitReg
        match &slice {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 4);
                assert_eq!(*val, 0b0110);
            }
            _ => panic!("expected BitReg"),
        }
        // Set upper 4 bits to that value
        assert!(
            v.set(
                &[Index::Slice {
                    start: Some(4),
                    step: None,
                    end: Some(8),
                }],
                slice
            )
            .is_ok()
        );
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(*val, 0b0110_0110);
            }
            _ => panic!("expected uint"),
        }
    }

    // --- array then bit indexing ---

    fn bitreg_arr(values: &[u128], bits: u32) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::bitreg_u128(v)).collect(),
            ArrayTy::new(PrimitiveTy::BitReg(bits), shape![values.len()]),
        ))
    }

    #[test]
    fn get_array_then_bit_item() {
        // array of 3 BitReg(4): [0b1010, 0b0101, 0b1100]
        let v = bitreg_arr(&[0b1010, 0b0101, 0b1100], 4);
        // array[1][2] = bit 2 of 0b0101 = 1
        let r = v.get(&[Index::Item(1), Index::Item(2)]).unwrap();
        assert!(matches!(
            r,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(true)) && matches!(s.ty(), PrimitiveTy::Bit)
        ));
        // array[0][0] = bit 0 of 0b1010 = 0
        let r = v.get(&[Index::Item(0), Index::Item(0)]).unwrap();
        assert!(matches!(
            r,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(false)) && matches!(s.ty(), PrimitiveTy::Bit)
        ));
    }

    #[test]
    fn get_array_then_bit_slice() {
        // array[2][0:2] = lower 2 bits of 0b1100 = 0b00
        let v = bitreg_arr(&[0b1010, 0b0101, 0b1100], 4);
        let r = v
            .get(&[
                Index::Item(2),
                Index::Slice {
                    start: Some(0),
                    step: None,
                    end: Some(2),
                },
            ])
            .unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let PrimitiveTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w, 2);
                assert_eq!(*val, 0b00);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn set_array_then_bit_item() {
        let mut v = bitreg_arr(&[0b0000, 0b0000], 4);
        // set array[1][3] = true → second element becomes 0b1000
        assert!(
            v.set(&[Index::Item(1), Index::Item(3)], bit_val(true))
                .is_ok()
        );
        match &v {
            Value::Array(a) => {
                assert_eq!(a.values()[0].as_bitreg(4).unwrap().as_u128(), 0b0000);
                assert_eq!(a.values()[1].as_bitreg(4).unwrap().as_u128(), 0b1000);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn set_array_then_bit_slice() {
        let mut v = bitreg_arr(&[0b0000, 0b0000], 4);
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::bitreg_u128(0b11),
            PrimitiveTy::BitReg(2),
        ));
        // set array[0][1:3] = 0b11 → first element bits 1,2 set → 0b0110
        assert!(
            v.set(
                &[
                    Index::Item(0),
                    Index::Slice {
                        start: Some(1),
                        step: None,
                        end: Some(3),
                    },
                ],
                src
            )
            .is_ok()
        );
        match &v {
            Value::Array(a) => {
                assert_eq!(a.values()[0].as_bitreg(4).unwrap().as_u128(), 0b0110);
                assert_eq!(a.values()[1].as_bitreg(4).unwrap().as_u128(), 0b0000);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn get_set_array_bit_roundtrip() {
        let mut v = bitreg_arr(&[0b1010, 0b0101], 4);
        // get bit 1 of element 0 (=1), set it as bit 0 of element 1
        let bit = v.get(&[Index::Item(0), Index::Item(1)]).unwrap();
        assert!(matches!(
            &bit,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(true))
        ));
        assert!(v.set(&[Index::Item(1), Index::Item(0)], bit).is_ok());
        match &v {
            Value::Array(a) => {
                assert_eq!(a.values()[0].as_bitreg(4).unwrap().as_u128(), 0b1010);
                assert_eq!(a.values()[1].as_bitreg(4).unwrap().as_u128(), 0b0101);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn value_get_ty_scalar_item_returns_bit() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));

        assert!(matches!(
            v.ty().get(&[Index::Item(1)]),
            Ok(ValueTy::Scalar(PrimitiveTy::Bit))
        ));
        assert!(matches!(
            v.get(&[Index::Item(1)]).unwrap().ty(),
            ValueTy::Scalar(PrimitiveTy::Bit)
        ));
    }

    #[test]
    fn value_get_ty_scalar_slice_returns_bitreg_width() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));
        let indices = [Index::Slice {
            start: Some(0),
            step: Some(2),
            end: Some(4),
        }];

        match v.ty().get(&indices).unwrap() {
            ValueTy::Scalar(PrimitiveTy::BitReg(width)) => assert_eq!(width, 2),
            other => panic!("expected BitReg type, got {other:?}"),
        }
        match v.get(&indices).unwrap().ty() {
            ValueTy::Scalar(PrimitiveTy::BitReg(width)) => assert_eq!(width, 2),
            other => panic!("expected BitReg value type, got {other:?}"),
        }
    }

    #[test]
    fn value_get_ty_array_item_returns_subarray() {
        let v = Value::Array(Array::new_unchecked(
            vec![
                Primitive::uint(1),
                Primitive::uint(2),
                Primitive::uint(3),
                Primitive::uint(4),
                Primitive::uint(5),
                Primitive::uint(6),
            ],
            ArrayTy::new(PrimitiveTy::Uint(iw(8)), shape![2, 3]),
        ));

        let row = ValueTy::array(PrimitiveTy::Uint(iw(8)), shape![3]);
        assert_eq!(v.ty().get(&[Index::Item(1)]).unwrap(), row);
        assert_eq!(v.get(&[Index::Item(1)]).unwrap().ty(), row);
    }

    #[test]
    fn value_get_ty_array_then_bit_returns_scalar_bit() {
        let v = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(0b1010), Primitive::uint(0b0101)],
            ArrayTy::new(PrimitiveTy::Uint(iw(4)), shape![2]),
        ));
        let indices = [Index::Item(0), Index::Item(1)];

        assert!(matches!(
            v.ty().get(&indices),
            Ok(ValueTy::Scalar(PrimitiveTy::Bit))
        ));
        assert!(matches!(
            v.get(&indices).unwrap().ty(),
            ValueTy::Scalar(PrimitiveTy::Bit)
        ));
    }

    #[test]
    fn value_get_ty_matches_get_for_empty_select() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010),
            PrimitiveTy::Uint(iw(4)),
        ));

        // Empty select yields a 0-bit BitReg (was previously an error).
        let ty = v.ty().get(&[Index::Select(vec![])]).unwrap();
        let val = v.get(&[Index::Select(vec![])]).unwrap();
        assert!(matches!(ty, ValueTy::Scalar(PrimitiveTy::BitReg(0))));
        assert!(matches!(
            val,
            Value::Scalar(s) if matches!(s.ty(), PrimitiveTy::BitReg(0))
        ));
    }
}
