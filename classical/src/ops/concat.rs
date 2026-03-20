use crate::array::{Array, ArrayTy, ashape};
use crate::array_ref::{ArrayRefShape, ArrayRefTy};
use crate::error::{Error, Result};
use crate::ops::BinOp;
use crate::value::Value;

pub struct Concat;

fn unsupported(
    lhs: impl Into<crate::value::ValueTy>,
    rhs: impl Into<crate::value::ValueTy>,
) -> Error {
    Error::unsupported_binop(Concat::NAME, lhs.into(), rhs.into(), Concat::IS_FUNC)
}

fn concat_shape(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    if lhs.len() != rhs.len() || lhs[1..] != rhs[1..] {
        return None;
    }
    let mut shape = lhs.to_vec();
    shape[0] += rhs[0];
    Some(shape)
}

impl BinOp for Concat {
    const NAME: &'static str = "++";

    fn arr_arr_check(lhs: ArrayTy, rhs: ArrayTy) -> Result<(ArrayTy, ArrayTy, ArrayTy)> {
        if lhs.ty() != rhs.ty() {
            return Err(unsupported(lhs, rhs));
        }
        let new_shape = concat_shape(lhs.shape().get(), rhs.shape().get())
            .ok_or_else(|| unsupported(lhs, rhs))?;
        let out = ArrayTy::new(lhs.ty(), ashape(new_shape));
        Ok((lhs, rhs, out))
    }

    fn arr_ref_arr_ref_check(
        lhs: ArrayRefTy,
        rhs: ArrayRefTy,
    ) -> Result<(ArrayRefTy, ArrayRefTy, ArrayRefTy)> {
        if lhs.ty() != rhs.ty() || lhs.shape().dim() != rhs.shape().dim() {
            return Err(unsupported(lhs, rhs));
        }
        let out_shape = match (lhs.shape(), rhs.shape()) {
            (ArrayRefShape::Fixed(lhs_shape), ArrayRefShape::Fixed(rhs_shape)) => {
                ArrayRefShape::Fixed(ashape(
                    concat_shape(lhs_shape.get(), rhs_shape.get())
                        .ok_or_else(|| unsupported(lhs, rhs))?,
                ))
            }
            _ => ArrayRefShape::Dim(lhs.shape().dim()),
        };
        Ok((lhs, rhs, lhs.with_shape(out_shape)))
    }

    fn arr_ref_arr_check(
        lhs: ArrayRefTy,
        rhs: ArrayTy,
    ) -> Result<(ArrayRefTy, ArrayTy, ArrayRefTy)> {
        if lhs.ty() != rhs.ty() || lhs.shape().dim() != rhs.shape().dim() {
            return Err(unsupported(lhs, rhs));
        }
        let out_shape = match lhs.shape() {
            ArrayRefShape::Fixed(lhs_shape) => ArrayRefShape::Fixed(ashape(
                concat_shape(lhs_shape.get(), rhs.shape().get())
                    .ok_or_else(|| unsupported(lhs, rhs))?,
            )),
            ArrayRefShape::Dim(_) => ArrayRefShape::Dim(rhs.shape().dim()),
        };
        Ok((lhs, rhs, lhs.with_shape(out_shape)))
    }

    fn arr_arr_ref_check(
        lhs: ArrayTy,
        rhs: ArrayRefTy,
    ) -> Result<(ArrayTy, ArrayRefTy, ArrayRefTy)> {
        if lhs.ty() != rhs.ty() || lhs.shape().dim() != rhs.shape().dim() {
            return Err(unsupported(lhs, rhs));
        }
        let out_shape = match rhs.shape() {
            ArrayRefShape::Fixed(rhs_shape) => ArrayRefShape::Fixed(ashape(
                concat_shape(lhs.shape().get(), rhs_shape.get())
                    .ok_or_else(|| unsupported(lhs, rhs))?,
            )),
            ArrayRefShape::Dim(_) => ArrayRefShape::Dim(lhs.shape().dim()),
        };
        Ok((lhs, rhs, rhs.with_shape(out_shape)))
    }

    fn arr_arr_op(lhs: Array, rhs: Array, out: ArrayTy) -> Result<Array> {
        let new_values = lhs
            .values()
            .iter()
            .chain(rhs.values().iter())
            .copied()
            .collect();
        Array::new(new_values, out)
    }

    fn checked_op(lhs: Value, rhs: Value) -> Result<Value> {
        match (lhs, rhs) {
            (Value::Array(lhs), Value::Array(rhs)) => {
                let (_, _, out) = Self::arr_arr_check(lhs.ty(), rhs.ty())?;
                Self::arr_arr_op(lhs, rhs, out).map(Value::Array)
            }
            (Value::Array(lhs), Value::ArrayRef(rhs)) => {
                let rhs = rhs.to_owned()?;
                let (_, _, out) = Self::arr_arr_check(lhs.ty(), rhs.ty())?;
                Self::arr_arr_op(lhs, rhs, out).map(Value::Array)
            }
            (Value::ArrayRef(lhs), Value::Array(rhs)) => {
                let lhs = lhs.to_owned()?;
                let (_, _, out) = Self::arr_arr_check(lhs.ty(), rhs.ty())?;
                Self::arr_arr_op(lhs, rhs, out).map(Value::Array)
            }
            (Value::ArrayRef(lhs), Value::ArrayRef(rhs)) => {
                let lhs = lhs.to_owned()?;
                let rhs = rhs.to_owned()?;
                let (_, _, out) = Self::arr_arr_check(lhs.ty(), rhs.ty())?;
                Self::arr_arr_op(lhs, rhs, out).map(Value::Array)
            }
            (lhs, rhs) => Err(Error::unsupported_binop(
                Self::NAME,
                lhs.ty(),
                rhs.ty(),
                Self::IS_FUNC,
            )),
        }
    }
}

impl Value {
    pub fn concat_(self, rhs: Self) -> Result<Self> {
        Concat::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use crate::primitive::{Primitive, PrimitiveTy::*, bw};

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&value| Primitive::uint(value)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    fn f_array(values: &[f64], shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values
                .iter()
                .map(|&value| Primitive::float(value))
                .collect(),
            ArrayTy::new(Float(crate::primitive::FloatWidth::F64), ashape(shape)),
        ))
    }

    fn assert_uint_array(value: Value, bits: u32, shape: &[usize], expected: &[u128]) {
        match value {
            Value::Array(array) => {
                assert_eq!(array.ty().shape().get(), shape);
                let values: Vec<u128> = array
                    .values()
                    .iter()
                    .map(|value| value.as_uint(bw(bits)).unwrap())
                    .collect();
                assert_eq!(values, expected);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn concat_vectors_appends_values() {
        let result = u_array(&[1, 2], 8, vec![2])
            .concat_(u_array(&[3, 4, 5], 8, vec![3]))
            .unwrap();

        assert_uint_array(result, 8, &[5], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn concat_multidim_arrays_grows_leading_dimension() {
        let result = u_array(&[1, 2, 3, 4], 8, vec![2, 2])
            .concat_(u_array(&[5, 6], 8, vec![1, 2]))
            .unwrap();

        assert_uint_array(result, 8, &[3, 2], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn concat_array_ref_array_ref() {
        let lhs = u_array(&[1, 2, 3, 4], 8, vec![4])
            .get(&[Index::Slice {
                start: Some(1),
                step: None,
                end: Some(3),
            }])
            .unwrap();
        let rhs = u_array(&[10, 20, 30, 40], 8, vec![4])
            .get(&[Index::Select(vec![3, 0])])
            .unwrap();

        let result = lhs.concat_(rhs).unwrap();

        assert_uint_array(result, 8, &[4], &[2, 3, 40, 10]);
    }

    #[test]
    fn concat_array_array_ref() {
        let rhs = u_array(&[10, 20, 30, 40], 8, vec![4])
            .get(&[Index::Slice {
                start: Some(1),
                step: Some(2),
                end: None,
            }])
            .unwrap();

        let result = u_array(&[1, 2], 8, vec![2]).concat_(rhs).unwrap();

        assert_uint_array(result, 8, &[4], &[1, 2, 20, 40]);
    }

    #[test]
    fn concat_rejects_inner_shape_mismatch() {
        let lhs = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);
        let rhs = u_array(&[5, 6, 7], 8, vec![1, 3]);

        assert!(lhs.concat_(rhs).is_err());
    }

    #[test]
    fn concat_rejects_rank_mismatch_without_panicking() {
        let lhs = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);
        let rhs = u_array(&[5, 6], 8, vec![2]);

        assert!(lhs.concat_(rhs).is_err());
    }

    #[test]
    fn concat_rejects_element_type_mismatch() {
        let lhs = u_array(&[1, 2], 8, vec![2]);
        let rhs = f_array(&[3.0, 4.0], vec![2]);

        assert!(lhs.concat_(rhs).is_err());
    }
}
