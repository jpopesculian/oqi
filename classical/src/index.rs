use crate::array::{Array, ArrayDim, ArrayShape, ArrayTy};
use crate::array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, RefAccess};
use crate::scalar::{Scalar, ScalarTy};
use crate::value::{Value, ValueTy};
use crate::{
    Error, Result,
    primitive::{BitWidth, Primitive, resize_int},
};

#[derive(Debug, Clone)]
pub enum Index {
    Item(isize),
    Slice {
        start: Option<isize>,
        step: Option<isize>,
        end: Option<isize>,
    },
    Select(Vec<isize>),
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

fn is_valid_slice(start: usize, step: isize, end: usize) -> bool {
    if step == 0 {
        false
    } else {
        if step > 0 { start <= end } else { end <= start }
    }
}

impl Index {
    // given array dimensions, get new dimensions after applying the index
    pub fn shape(&self, shape: &[usize]) -> Option<Vec<usize>> {
        if shape.is_empty() {
            return None;
        }
        let dim = shape[0];
        Some(match self {
            Self::Item(i) => {
                resolve_index(*i, dim, false)?;
                shape[1..].to_vec()
            }
            Self::Slice { start, step, end } => {
                let start = resolve_index(start.unwrap_or(0), dim, false)?;
                let end = resolve_index(end.unwrap_or(dim as isize), dim, true)?;
                let step = step.unwrap_or(1);
                if !is_valid_slice(start, step, end) {
                    return None;
                }
                let mut new_shape = shape.to_vec();
                let range = end as isize - start as isize;
                new_shape[0] = ((range + step - step.signum()) / step) as usize;
                new_shape
            }
            Self::Select(indices) => {
                for &i in indices {
                    resolve_index(i, dim, false)?;
                }
                let mut new_shape = shape.to_vec();
                new_shape[0] = indices.len();
                new_shape
            }
        })
    }

    // Given the original shape and an index for the new shape, return the index for the original shape
    // e.g. Given Index::Elem(1), shape [3,4] and index: [1] return [1, 1]
    pub fn index(&self, shape: &[usize], index: &[usize]) -> Option<Vec<usize>> {
        if shape.is_empty() {
            return None;
        }
        let dim = shape[0];
        match self {
            Self::Item(i) => {
                let resolved = resolve_index(*i, dim, false)?;
                let mut result = vec![resolved];
                result.extend_from_slice(index);
                Some(result)
            }
            Self::Slice { start, step, end } => {
                let start = resolve_index(start.unwrap_or(0), dim, false)?;
                let end = resolve_index(end.unwrap_or(dim as isize), dim, true)?;
                let step = step.unwrap_or(1);
                if !is_valid_slice(start, step, end) {
                    return None;
                }
                let original = start as isize + *index.first()? as isize * step;
                if original < 0 || original as usize >= dim {
                    return None;
                }
                let mut result = vec![original as usize];
                result.extend_from_slice(&index[1..]);
                Some(result)
            }
            Self::Select(indices) => {
                let i = *index.first()?;
                if i >= indices.len() {
                    return None;
                }
                let original = resolve_index(indices[i], dim, false)?;
                let mut result = vec![original];
                result.extend_from_slice(&index[1..]);
                Some(result)
            }
        }
    }

    pub fn iter(&self, dim: usize) -> Option<IndexIter> {
        Some(match self {
            Self::Item(i) => IndexIter::Item(Some(resolve_index(*i, dim, false)?)),
            Self::Slice { start, step, end } => {
                let start = resolve_index(start.unwrap_or(0), dim, false)?;
                let end = resolve_index(end.unwrap_or(dim as isize), dim, true)?;
                let step = step.unwrap_or(1);
                if !is_valid_slice(start, step, end) {
                    return None;
                }
                IndexIter::Slice(SliceIter { start, step, end })
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

pub enum IndexIter {
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
            Self::Item(opt) => {
                if opt.is_some() {
                    1
                } else {
                    0
                }
            }
            Self::Slice(iter) => iter.len(),
            Self::Select(vec_iter) => vec_iter.len(),
        }
    }
}

pub struct SliceIter {
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

/// multi-dimensional index to flat index
fn flat_to_multi(shape: &[usize], mut flat: usize) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        result[i] = flat % shape[i];
        flat /= shape[i];
    }
    result
}

/// flat index to multi-dimensional index
fn multi_to_flat(shape: &[usize], index: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat += index[i] * stride;
        stride *= shape[i];
    }
    flat
}

fn index_error(value: ValueTy, indices: &[Index]) -> Error {
    Error::IndexOutOfBounds {
        value: Box::new(value),
        index: indices.to_vec(),
    }
}

fn compute_shapes(shape: &[usize], indices: &[Index]) -> Option<Vec<Vec<usize>>> {
    let mut shapes = Vec::with_capacity(indices.len() + 1);
    shapes.push(shape.to_vec());
    for idx in indices {
        if shapes.last().unwrap().is_empty() {
            break;
        }
        shapes.push(idx.shape(shapes.last().unwrap())?);
    }
    Some(shapes)
}

fn resolve_array_position(
    shapes: &[Vec<usize>],
    indices: &[Index],
    mut pos: Vec<usize>,
) -> Option<Vec<usize>> {
    for (i, idx) in indices.iter().enumerate().rev() {
        pos = idx.index(&shapes[i], &pos)?;
    }
    Some(pos)
}

#[inline]
fn get_bit(bits: u128, position: usize) -> u128 {
    (bits >> position) & 1
}

#[inline]
fn get_bits(bits: u128, positions: impl IntoIterator<Item = usize>) -> u128 {
    let mut out = 0;
    for (i, pos) in positions.into_iter().enumerate() {
        out |= get_bit(bits, pos) << i
    }
    out
}

#[inline]
fn iter_bits(bits: u128, bw: usize) -> impl Iterator<Item = bool> {
    (0..bw).map(move |i| get_bit(bits, i) != 0)
}

#[inline]
fn set_bit(bits: u128, position: usize, set: bool) -> u128 {
    let mask = 1u128 << position;
    if set { bits | mask } else { bits & !mask }
}

#[inline]
fn set_bits(bits: u128, ops: impl IntoIterator<Item = (usize, bool)>) -> u128 {
    let mut out = bits;
    for (idx, op) in ops.into_iter() {
        out = set_bit(out, idx, op);
    }
    out
}

impl ScalarTy {
    fn get(&self, indices: &[Index]) -> Result<ScalarTy> {
        if indices.is_empty() {
            return Ok(*self);
        }

        let error = || index_error((*self).into(), indices);
        let Some(dim) = self.bw().map(|bw| bw.get() as usize) else {
            return Err(error());
        };
        if indices.len() != 1 {
            return Err(error());
        }

        match &indices[0] {
            Index::Item(i) => {
                resolve_index(*i, dim, false).ok_or_else(error)?;
                Ok(ScalarTy::Bit)
            }
            other => {
                let iter = other.iter(dim).ok_or_else(error)?;
                let new_bw = BitWidth::new(iter.len() as u32)?;
                Ok(ScalarTy::BitReg(new_bw))
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
        let shapes = compute_shapes(self.shape().get(), indices).ok_or_else(error)?;
        let n_array = shapes.len() - 1;
        let result_shape = shapes.last().unwrap();
        let result_ty = if result_shape.is_empty() {
            ValueTy::Scalar(self.ty())
        } else {
            ValueTy::Array(ArrayTy::new(
                self.ty(),
                ArrayShape::new(result_shape.clone())?,
            ))
        };

        if n_array < indices.len() {
            match result_ty {
                ValueTy::Scalar(ty) => ty.get(&indices[n_array..]).map(ValueTy::Scalar),
                ValueTy::Array(_) => Err(error()),
                ValueTy::ArrayRef(_) => {
                    unreachable!("array indexing on ArrayTy cannot yield ArrayRef")
                }
            }
        } else {
            Ok(result_ty)
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
            ArrayRefShape::Fixed(shape) => match ArrayTy::new(self.ty(), shape).get(indices)? {
                ValueTy::Scalar(ty) => Ok(ValueTy::Scalar(ty)),
                ValueTy::Array(ty) => Ok(ValueTy::ArrayRef(ArrayRefTy::new(
                    ty.ty(),
                    ty.shape().into(),
                    self.access(),
                ))),
                ValueTy::ArrayRef(_) => {
                    unreachable!("array indexing on ArrayTy cannot yield ArrayRef")
                }
            },
            ArrayRefShape::Dim(dim) => {
                let mut result_dim = dim.get();
                let mut n_array = 0;

                for idx in indices {
                    if result_dim == 0 {
                        break;
                    }
                    match idx {
                        Index::Item(_) => result_dim -= 1,
                        Index::Slice { step: Some(0), .. } => return Err(error()),
                        Index::Slice { .. } | Index::Select(_) => {}
                    }
                    n_array += 1;
                }

                let result_ty = if result_dim == 0 {
                    ValueTy::Scalar(self.ty())
                } else {
                    ValueTy::ArrayRef(ArrayRefTy::new(
                        self.ty(),
                        ArrayRefShape::Dim(ArrayDim::new(result_dim)?),
                        self.access(),
                    ))
                };

                if n_array < indices.len() {
                    match result_ty {
                        ValueTy::Scalar(ty) => ty.get(&indices[n_array..]).map(ValueTy::Scalar),
                        ValueTy::Array(_) => {
                            unreachable!("array ref indexing never yields owned arrays")
                        }
                        ValueTy::ArrayRef(_) => Err(error()),
                    }
                } else {
                    Ok(result_ty)
                }
            }
        }
    }
}

impl Scalar {
    pub fn get(&self, indices: &[Index]) -> Result<Scalar> {
        if indices.is_empty() {
            return Ok(*self);
        }

        let value_ty = ValueTy::Scalar(self.ty());
        let error = || index_error(value_ty, indices);
        let Some(dim) = self.ty().bw().map(|bw| bw.get() as usize) else {
            return Err(error());
        };
        if indices.len() != 1 {
            return Err(error());
        }

        let bits = match self.value() {
            Primitive::Uint(v) | Primitive::BitReg(v) => v,
            Primitive::Int(v) => v as u128,
            _ => return Err(error()),
        };

        match &indices[0] {
            Index::Item(i) => {
                let position = resolve_index(*i, dim, false).ok_or_else(error)?;
                Ok(Scalar::new_unchecked(
                    Primitive::Bit(get_bit(bits, position) != 0),
                    ScalarTy::Bit,
                ))
            }
            other => {
                let iter = other.iter(dim).ok_or_else(error)?;
                let new_bw = BitWidth::new(iter.len() as u32)?;
                let new_bits = get_bits(bits, iter);
                Ok(Scalar::new_unchecked(
                    Primitive::BitReg(new_bits),
                    ScalarTy::BitReg(new_bw),
                ))
            }
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Scalar) -> Result<()> {
        if indices.is_empty() {
            if self.ty() != value.ty() {
                return Err(Error::UnexpectedTy {
                    expected: Box::new(self.ty().into()),
                    received: Box::new(value.ty().into()),
                });
            }
            *self = value;
            return Ok(());
        }

        let value_ty = ValueTy::Scalar(self.ty());
        let error = || index_error(value_ty, indices);
        let Some(bw) = self.ty().bw() else {
            return Err(error());
        };
        let dim = bw.get() as usize;
        if indices.len() != 1 {
            return Err(error());
        }

        let bits = match self.value() {
            Primitive::Uint(v) | Primitive::BitReg(v) => v,
            Primitive::Int(v) => v as u128,
            _ => return Err(error()),
        };

        let bits = match &indices[0] {
            Index::Item(i) => {
                let set = match value.value() {
                    Primitive::Bit(b) => b,
                    _ => {
                        return Err(Error::UnexpectedTy {
                            expected: Box::new(ScalarTy::Bit.into()),
                            received: Box::new(value.ty().into()),
                        });
                    }
                };
                let Some(idx) = resolve_index(*i, dim, false) else {
                    return Err(error());
                };
                set_bit(bits, idx, set)
            }
            other => {
                let iter = other.iter(dim).ok_or_else(error)?;
                let (new_bits, bw) = match (value.value(), value.ty()) {
                    (Primitive::BitReg(b), ScalarTy::BitReg(bw)) => (b, bw),
                    (Primitive::Uint(b), ScalarTy::Uint(bw)) => (b, bw),
                    (Primitive::Int(b), ScalarTy::Int(bw)) => (b as u128, bw),
                    _ => {
                        return Err(Error::UnexpectedTy {
                            expected: Box::new(self.ty().into()),
                            received: Box::new(value.ty().into()),
                        });
                    }
                };
                if iter.len() != bw.get() as usize {
                    return Err(Error::UnexpectedTy {
                        expected: Box::new(self.ty().into()),
                        received: Box::new(value.ty().into()),
                    });
                }
                set_bits(bits, iter.zip(iter_bits(new_bits, bw.get() as usize)))
            }
        };
        let value = match self.value() {
            Primitive::Uint(_) => Primitive::Uint(bits),
            Primitive::BitReg(_) => Primitive::BitReg(bits),
            Primitive::Int(_) => Primitive::Int(resize_int(bits as i128, bw)),
            _ => return Err(error()),
        };
        *self = Scalar::new_unchecked(value, self.ty());
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
        let aty = self.ty();
        let ty = aty.ty();
        let shape = aty.shape();
        let shapes = compute_shapes(shape.get(), indices).ok_or_else(error)?;
        let n_array = shapes.len() - 1;
        let array_indices = &indices[..n_array];
        let result_shape = shapes.last().unwrap();

        let total: usize = result_shape.iter().product();
        let mut result = Vec::with_capacity(total);
        for flat in 0..total {
            let pos =
                resolve_array_position(&shapes, array_indices, flat_to_multi(result_shape, flat))
                    .ok_or_else(error)?;
            result.push(self.values()[multi_to_flat(shape.get(), &pos)]);
        }

        let value = if result_shape.is_empty() {
            Value::Scalar(Scalar::new_unchecked(result[0], ty))
        } else {
            Value::Array(Array::new_unchecked(
                result,
                ArrayTy::new(ty, ArrayShape::new(result_shape.clone())?),
            ))
        };

        if n_array < indices.len() {
            match value {
                Value::Scalar(s) => s.get(&indices[n_array..]).map(Value::Scalar),
                Value::Array(_) => Err(error()),
                Value::ArrayRef(_) => {
                    unreachable!("Array::get only produces owned arrays or scalars")
                }
            }
        } else {
            Ok(value)
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
                    *self = ar.cast(self.ty().as_ref_mut())?;
                    Ok(())
                }
                Value::Scalar(s) => Err(Error::unsupported_cast(s.ty().into(), self.ty().into())),
            };
        }

        let value_ty = ValueTy::Array(self.ty());
        let error = || index_error(value_ty, indices);
        let aty = self.ty();
        let ty = aty.ty();
        let shape = aty.shape();
        let shapes = compute_shapes(shape.get(), indices).ok_or_else(error)?;
        let n_array = shapes.len() - 1;
        let array_indices = &indices[..n_array];
        let bit_indices = &indices[n_array..];
        let result_shape = shapes.last().unwrap();

        if !bit_indices.is_empty() {
            if !result_shape.is_empty() {
                return Err(error());
            }
            let Value::Scalar(value) = value else {
                return Err(Error::UnexpectedTy {
                    expected: Box::new(self.ty().ty().into()),
                    received: Box::new(value.ty()),
                });
            };
            let pos = resolve_array_position(&shapes, array_indices, vec![]).ok_or_else(error)?;
            let flat = multi_to_flat(shape.get(), &pos);
            let mut tmp = Scalar::new_unchecked(self.values()[flat], ty);
            tmp.set(bit_indices, value)?;
            self.values_mut()[flat] = tmp.value();
            return Ok(());
        }

        let val_scalars: Vec<Primitive> = match value {
            Value::Scalar(s) => {
                if !result_shape.is_empty() {
                    return Err(error());
                }
                vec![s.value()]
            }
            Value::Array(a) => {
                if a.ty().shape().get() != result_shape.as_slice() {
                    return Err(error());
                }
                a.values().to_vec()
            }
            Value::ArrayRef(ar) => {
                let a = ar.to_owned()?;
                if a.ty().shape().get() != result_shape.as_slice() {
                    return Err(error());
                }
                a.values().to_vec()
            }
        };

        let total: usize = result_shape.iter().product();
        let scalars = self.values_mut();
        for (flat, scalar) in val_scalars.into_iter().enumerate().take(total) {
            let pos =
                resolve_array_position(&shapes, array_indices, flat_to_multi(result_shape, flat))
                    .ok_or_else(error)?;
            scalars[multi_to_flat(shape.get(), &pos)] = scalar;
        }
        Ok(())
    }
}

impl ArrayRef {
    pub fn get(&self, indices: &[Index]) -> Result<Value> {
        if indices.is_empty() {
            return Ok(Value::ArrayRef(self.clone()));
        }

        let mut new_indices = self.indices().to_vec();
        new_indices.extend_from_slice(indices);
        match self.ty().get(indices)? {
            ValueTy::Scalar(_) => self.array().borrow()?.get(&new_indices),
            ValueTy::ArrayRef(_) => {
                ArrayRef::new(self.array().clone(), new_indices, self.ty().access())
                    .map(Value::ArrayRef)
            }
            ValueTy::Array(_) => unreachable!("array ref indexing never yields owned arrays"),
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Value) -> Result<()> {
        if matches!(self.ty().access(), RefAccess::Readonly) {
            return Err(Error::ReadOnly);
        }
        let mut new_indices = self.indices().to_vec();
        new_indices.extend_from_slice(indices);
        self.array().borrow_mut()?.set(&new_indices, value)?;
        Ok(())
    }
}

impl Value {
    pub fn get(&self, indices: &[Index]) -> Result<Value> {
        if indices.is_empty() {
            return Ok(self.clone());
        }

        match self {
            Value::Scalar(s) => s.get(indices).map(Value::Scalar),
            Value::Array(a) => a.clone().into_ref_mut().get(indices),
            Value::ArrayRef(ar) => ar.get(indices),
        }
    }

    pub fn set(&mut self, indices: &[Index], value: Value) -> Result<()> {
        if indices.is_empty() {
            if self.ty() != value.ty() {
                return Err(Error::UnexpectedTy {
                    expected: Box::new(self.ty()),
                    received: Box::new(value.ty()),
                });
            }
            *self = value;
            return Ok(());
        }

        match self {
            Value::Scalar(s) => {
                let Value::Scalar(value) = value else {
                    return Err(Error::UnexpectedTy {
                        expected: Box::new(self.ty()),
                        received: Box::new(value.ty()),
                    });
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
            ValueTy::Array(a) => match a.get(indices)? {
                ValueTy::Scalar(ty) => Ok(ValueTy::Scalar(ty)),
                ValueTy::Array(ty) => Ok(ValueTy::ArrayRef(ty.as_ref_mut())),
                ValueTy::ArrayRef(_) => {
                    unreachable!("array indexing on ArrayTy cannot yield ArrayRef")
                }
            },
            ValueTy::ArrayRef(a) => a.get(indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayShape, shape};
    use crate::primitive::bw;

    // --- Item ---

    #[test]
    fn item_removes_first_dim() {
        assert_eq!(Index::Item(0).shape(&[3, 4, 5]), Some(vec![4, 5]));
    }

    #[test]
    fn item_on_1d_returns_scalar_shape() {
        assert_eq!(Index::Item(0).shape(&[5]), Some(vec![]));
    }

    #[test]
    fn item_on_empty_shape_returns_none() {
        assert_eq!(Index::Item(0).shape(&[]), None);
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

    #[test]
    fn slice_on_empty_shape_returns_none() {
        let idx = Index::Slice {
            start: None,
            step: None,
            end: None,
        };
        assert_eq!(idx.shape(&[]), None);
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
    fn select_on_empty_shape_returns_none() {
        assert_eq!(Index::Select(vec![0]).shape(&[]), None);
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
            ArrayTy::new(ScalarTy::Uint(bw(8)), shape),
        ))
    }

    fn sc(v: u8) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(v as u128),
            ScalarTy::Uint(bw(8)),
        ))
    }

    fn to_u8s(val: &Value) -> Vec<u8> {
        match val {
            Value::Array(a) => a
                .values()
                .iter()
                .map(|s| s.as_uint(bw(8)).unwrap() as u8)
                .collect(),
            Value::ArrayRef(ar) => ar
                .to_owned()
                .unwrap()
                .values()
                .iter()
                .map(|s| s.as_uint(bw(8)).unwrap() as u8)
                .collect(),
            Value::Scalar(s) => vec![s.value().as_uint(bw(8)).unwrap() as u8],
        }
    }

    // --- get tests ---

    #[test]
    fn get_item_returns_row() {
        let val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let result = val.get(&[Index::Item(0)]).unwrap();
        assert!(matches!(result, Value::ArrayRef(_)));
        assert_eq!(to_u8s(&result), vec![0, 1, 2]);
    }

    #[test]
    fn get_item_second_row() {
        let val = arr(&[0, 1, 2, 3, 4, 5], shape![2, 3]);
        let result = val.get(&[Index::Item(1)]).unwrap();
        assert!(matches!(result, Value::ArrayRef(_)));
        assert_eq!(to_u8s(&result), vec![3, 4, 5]);
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
        assert!(matches!(result, Value::ArrayRef(_)));
        assert_eq!(to_u8s(&result), vec![1, 2, 3]);
    }

    #[test]
    fn get_select() {
        let val = arr(&[0, 1, 2, 3, 4], shape![5]);
        let result = val.get(&[Index::Select(vec![3, 1, 4])]).unwrap();
        assert!(matches!(result, Value::ArrayRef(_)));
        assert_eq!(to_u8s(&result), vec![3, 1, 4]);
    }

    #[test]
    fn get_on_float_scalar_returns_none() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(1.0),
            ScalarTy::Float(crate::primitive::FloatWidth::F64),
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
        let scalar = Scalar::new_unchecked(Primitive::uint(0b1010u32), ScalarTy::Uint(bw(4)));
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
            ArrayTy::new(ScalarTy::Uint(bw(8)), shape![3]),
        );

        assert!(array.set(&[Index::Item(1)], sc(99)).is_ok());
        assert_eq!(
            array
                .values()
                .iter()
                .map(|v| v.as_uint(bw(8)).unwrap() as u8)
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
        Value::Scalar(Scalar::new_unchecked(Primitive::Bit(b), ScalarTy::Bit))
    }

    #[test]
    fn get_bit_from_uint() {
        // 0b1010 = 10
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(8)),
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
            Primitive::BitReg(0b0101),
            ScalarTy::BitReg(bw(4)),
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
            ScalarTy::Int(bw(8)),
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
            Primitive::uint(0b1000u32),
            ScalarTy::Uint(bw(4)),
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
            Primitive::uint(0xFFu32),
            ScalarTy::Uint(bw(8)),
        ));
        assert!(v.get(&[Index::Item(8)]).is_err());
        assert!(v.get(&[Index::Item(-9)]).is_err());
    }

    #[test]
    fn get_bit_slice_uint() {
        // 0b1010_0110, bits 1..5 = bits 1,2,3,4 = 1,1,0,0 → 0b0011
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_0110u32),
            ScalarTy::Uint(bw(8)),
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
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 4);
                assert_eq!(val, 0b0011);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_slice_bitreg() {
        // 0b1100 in 4 bits, bits 0..4 step 2 = bits 0,2 = 0,1 → 0b10
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b1100),
            ScalarTy::BitReg(bw(4)),
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
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 2);
                assert_eq!(val, 0b10);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_slice_full() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
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
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 4);
                assert_eq!(val, 0b1010);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_slice_reversed() {
        // 0b1010 in 4 bits, bits 3 down to 0 (step -1) = bits 3,2,1,0 = 1,0,1,0
        // result bit 0=1, bit 1=0, bit 2=1, bit 3=0 → 0b0101
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
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
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                // bits at positions 3,2,1 → values 1,0,1 → result 0b101
                assert_eq!(w.get(), 3);
                assert_eq!(val, 0b101);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_select_uint() {
        // 0b1010, select bits [3, 0] → bit 3=1, bit 0=0 → result bit 0=1, bit 1=0 → 0b01
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
        ));
        let r = v.get(&[Index::Select(vec![3, 0])]).unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 2);
                assert_eq!(val, 0b01);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_select_bitreg() {
        // 0b1111, select bits [0, 2] → both 1 → 0b11
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b1111),
            ScalarTy::BitReg(bw(4)),
        ));
        let r = v.get(&[Index::Select(vec![0, 2])]).unwrap();
        match r {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected BitReg")
                };
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 2);
                assert_eq!(val, 0b11);
            }
            _ => panic!("expected BitReg"),
        }
    }

    #[test]
    fn get_bit_select_empty_returns_none() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFFu32),
            ScalarTy::Uint(bw(8)),
        ));
        assert!(v.get(&[Index::Select(vec![])]).is_err());
    }

    #[test]
    fn get_bit_select_out_of_bounds() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0xFFu32),
            ScalarTy::Uint(bw(8)),
        ));
        assert!(v.get(&[Index::Select(vec![0, 8])]).is_err());
    }

    #[test]
    fn set_bit_on_uint() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0u32),
            ScalarTy::Uint(bw(4)),
        ));
        assert!(v.set(&[Index::Item(1)], bit_val(true)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(val, 0b0010);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_clear_on_uint() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1111u32),
            ScalarTy::Uint(bw(4)),
        ));
        assert!(v.set(&[Index::Item(2)], bit_val(false)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(val, 0b1011);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_on_bitreg() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b0000),
            ScalarTy::BitReg(bw(4)),
        ));
        assert!(v.set(&[Index::Item(3)], bit_val(true)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::BitReg(val) = s.value() else {
                    panic!("expected bitreg")
                };
                assert_eq!(val, 0b1000);
            }
            _ => panic!("expected bitreg"),
        }
    }

    #[test]
    fn set_bit_on_int_sign_extends() {
        // Int(8) = 0, set bit 7 (sign bit) → should become -128
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::int(0),
            ScalarTy::Int(bw(8)),
        ));
        assert!(v.set(&[Index::Item(7)], bit_val(true)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Int(val) = s.value() else {
                    panic!("expected int")
                };
                assert_eq!(val, -128);
            }
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn set_bit_on_int_clear_sign() {
        // Int(8) = -1 (0xFF), clear bit 7 → 0x7F = 127
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::int(-1),
            ScalarTy::Int(bw(8)),
        ));
        assert!(v.set(&[Index::Item(7)], bit_val(false)).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Int(val) = s.value() else {
                    panic!("expected int")
                };
                assert_eq!(val, 127);
            }
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn set_bit_with_non_bit_value_returns_false() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0u32),
            ScalarTy::Uint(bw(8)),
        ));
        assert!(v.set(&[Index::Item(0)], sc(1)).is_err());
    }

    #[test]
    fn set_bit_out_of_bounds_returns_false() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0u32),
            ScalarTy::Uint(bw(8)),
        ));
        assert!(v.set(&[Index::Item(8)], bit_val(true)).is_err());
    }

    #[test]
    fn get_set_bit_roundtrip() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
        ));
        let bit = v.get(&[Index::Item(1)]).unwrap();
        // bit 1 of 0b1010 is true
        assert!(matches!(
            bit,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(true))
        ));
        // set bit 0 to that value
        assert!(v.set(&[Index::Item(0)], bit).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(val, 0b1011);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_slice_uint() {
        // 0b0000_0000, set bits 2..6 to 0b1010 (4-bit BitReg)
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0u32),
            ScalarTy::Uint(bw(8)),
        ));
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b1010),
            ScalarTy::BitReg(bw(4)),
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
                assert_eq!(val, 0b00101000);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_select_uint() {
        // 0b0000, set bits [0, 3] to 0b11 (2-bit BitReg)
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0u32),
            ScalarTy::Uint(bw(4)),
        ));
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b11),
            ScalarTy::BitReg(bw(2)),
        ));
        assert!(v.set(&[Index::Select(vec![0, 3])], src).is_ok());
        match &v {
            Value::Scalar(s) => {
                let Primitive::Uint(val) = s.value() else {
                    panic!("expected uint")
                };
                assert_eq!(val, 0b1001);
            }
            _ => panic!("expected uint"),
        }
    }

    #[test]
    fn set_bit_slice_bitreg() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b1111),
            ScalarTy::BitReg(bw(4)),
        ));
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b00),
            ScalarTy::BitReg(bw(2)),
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
                assert_eq!(val, 0b1001);
            }
            _ => panic!("expected bitreg"),
        }
    }

    #[test]
    fn set_bit_slice_wrong_width_returns_false() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0u32),
            ScalarTy::Uint(bw(8)),
        ));
        // Slice selects 4 bits but value is 3-bit
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b111),
            ScalarTy::BitReg(bw(3)),
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
    fn get_set_slice_roundtrip() {
        let mut v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010_0110u32),
            ScalarTy::Uint(bw(8)),
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
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 4);
                assert_eq!(val, 0b0110);
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
                assert_eq!(val, 0b0110_0110);
            }
            _ => panic!("expected uint"),
        }
    }

    // --- array then bit indexing ---

    fn bitreg_arr(values: &[u128], bits: u32) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::BitReg(v)).collect(),
            ArrayTy::new(ScalarTy::BitReg(bw(bits)), shape![values.len()]),
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
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(true)) && matches!(s.ty(), ScalarTy::Bit)
        ));
        // array[0][0] = bit 0 of 0b1010 = 0
        let r = v.get(&[Index::Item(0), Index::Item(0)]).unwrap();
        assert!(matches!(
            r,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(false)) && matches!(s.ty(), ScalarTy::Bit)
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
                let ScalarTy::BitReg(w) = s.ty() else {
                    panic!("expected BitReg")
                };
                assert_eq!(w.get(), 2);
                assert_eq!(val, 0b00);
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
                assert_eq!(a.values()[0].as_bitreg(bw(4)).unwrap(), 0b0000);
                assert_eq!(a.values()[1].as_bitreg(bw(4)).unwrap(), 0b1000);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn set_array_then_bit_slice() {
        let mut v = bitreg_arr(&[0b0000, 0b0000], 4);
        let src = Value::Scalar(Scalar::new_unchecked(
            Primitive::BitReg(0b11),
            ScalarTy::BitReg(bw(2)),
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
                assert_eq!(a.values()[0].as_bitreg(bw(4)).unwrap(), 0b0110);
                assert_eq!(a.values()[1].as_bitreg(bw(4)).unwrap(), 0b0000);
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
            bit,
            Value::Scalar(s) if matches!(s.value(), Primitive::Bit(true))
        ));
        assert!(v.set(&[Index::Item(1), Index::Item(0)], bit).is_ok());
        match &v {
            Value::Array(a) => {
                assert_eq!(a.values()[0].as_bitreg(bw(4)).unwrap(), 0b1010);
                assert_eq!(a.values()[1].as_bitreg(bw(4)).unwrap(), 0b0101);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn value_get_ty_scalar_item_returns_bit() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
        ));

        assert!(matches!(
            v.ty().get(&[Index::Item(1)]),
            Ok(ValueTy::Scalar(ScalarTy::Bit))
        ));
        assert!(matches!(
            v.get(&[Index::Item(1)]).unwrap().ty(),
            ValueTy::Scalar(ScalarTy::Bit)
        ));
    }

    #[test]
    fn value_get_ty_scalar_slice_returns_bitreg_width() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
        ));
        let indices = [Index::Slice {
            start: Some(0),
            step: Some(2),
            end: Some(4),
        }];

        match v.ty().get(&indices).unwrap() {
            ValueTy::Scalar(ScalarTy::BitReg(width)) => assert_eq!(width.get(), 2),
            other => panic!("expected BitReg type, got {other:?}"),
        }
        match v.get(&indices).unwrap().ty() {
            ValueTy::Scalar(ScalarTy::BitReg(width)) => assert_eq!(width.get(), 2),
            other => panic!("expected BitReg value type, got {other:?}"),
        }
    }

    #[test]
    fn value_get_ty_array_item_returns_subarray() {
        let v = Value::Array(Array::new_unchecked(
            vec![
                Primitive::uint(1u32),
                Primitive::uint(2u32),
                Primitive::uint(3u32),
                Primitive::uint(4u32),
                Primitive::uint(5u32),
                Primitive::uint(6u32),
            ],
            ArrayTy::new(ScalarTy::Uint(bw(8)), shape![2, 3]),
        ));

        match v.ty().get(&[Index::Item(1)]).unwrap() {
            ValueTy::ArrayRef(ty) => {
                assert_eq!(ty.shape(), shape![3]);
                assert_eq!(ty.access(), RefAccess::Mutable);
                assert!(matches!(ty.ty(), ScalarTy::Uint(width) if width.get() == 8));
            }
            other => panic!("expected array ref type, got {other:?}"),
        }
        match v.get(&[Index::Item(1)]).unwrap().ty() {
            ValueTy::ArrayRef(ty) => {
                assert_eq!(ty.shape(), shape![3]);
                assert_eq!(ty.access(), RefAccess::Mutable);
                assert!(matches!(ty.ty(), ScalarTy::Uint(width) if width.get() == 8));
            }
            other => panic!("expected array ref value type, got {other:?}"),
        }
    }

    #[test]
    fn value_get_ty_array_then_bit_returns_scalar_bit() {
        let v = Value::Array(Array::new_unchecked(
            vec![Primitive::uint(0b1010u32), Primitive::uint(0b0101u32)],
            ArrayTy::new(ScalarTy::Uint(bw(4)), shape![2]),
        ));
        let indices = [Index::Item(0), Index::Item(1)];

        assert!(matches!(
            v.ty().get(&indices),
            Ok(ValueTy::Scalar(ScalarTy::Bit))
        ));
        assert!(matches!(
            v.get(&indices).unwrap().ty(),
            ValueTy::Scalar(ScalarTy::Bit)
        ));
    }

    #[test]
    fn value_get_ty_matches_get_errors() {
        let v = Value::Scalar(Scalar::new_unchecked(
            Primitive::uint(0b1010u32),
            ScalarTy::Uint(bw(4)),
        ));

        assert!(v.ty().get(&[Index::Select(vec![])]).is_err());
        assert!(v.get(&[Index::Select(vec![])]).is_err());
    }
}
