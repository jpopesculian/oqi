use super::{Add, BinOp, unsupported_scalar_binop};
use crate::array::{Array, ArrayShape, ArrayTy, shape};
use crate::primitive::{FloatWidth, PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Error, Primitive, Result};

pub struct Mul;

struct ArrArrPlan {
    squeeze_a: bool,
    squeeze_b: bool,
    m: usize,
    k: usize,
    p: usize,
    batch_out: Vec<usize>,
}

impl ArrArrPlan {
    fn output_shape(&self) -> Vec<usize> {
        let mut shape = self.batch_out.clone();
        if !self.squeeze_a {
            shape.push(self.m);
        }
        if !self.squeeze_b {
            shape.push(self.p);
        }
        if shape.is_empty() { vec![1] } else { shape }
    }
}

fn arr_arr_plan(lhs: &[usize], rhs: &[usize]) -> Option<ArrArrPlan> {
    let squeeze_a = lhs.len() == 1;
    let squeeze_b = rhs.len() == 1;

    let (m, k) = if squeeze_a {
        (1, lhs[0])
    } else {
        (lhs[lhs.len() - 2], lhs[lhs.len() - 1])
    };
    let (k_b, p) = if squeeze_b {
        (rhs[0], 1)
    } else {
        (rhs[rhs.len() - 2], rhs[rhs.len() - 1])
    };

    if k == 0 || k != k_b {
        return None;
    }

    let batch_a: &[usize] = if squeeze_a {
        &[]
    } else {
        &lhs[..lhs.len() - 2]
    };
    let batch_b: &[usize] = if squeeze_b {
        &[]
    } else {
        &rhs[..rhs.len() - 2]
    };
    let batch_out = broadcast_shape(batch_a, batch_b)?;

    Some(ArrArrPlan {
        squeeze_a,
        squeeze_b,
        m,
        k,
        p,
        batch_out,
    })
}

impl BinOp for Mul {
    const NAME: &'static str = "*";

    fn scalar_check(
        lht: PrimitiveTy,
        rht: PrimitiveTy,
    ) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        use PrimitiveTy::*;
        match (lht, rht) {
            (Duration, Int(_) | Uint(_) | Float(_)) => {
                Ok((Duration, Float(FloatWidth::F64), Duration))
            }
            (Int(_) | Uint(_) | Float(_), Duration) => {
                Ok((Float(FloatWidth::F64), Duration, Duration))
            }
            (Angle(a), Int(b) | Uint(b)) => {
                let out = a.max(b);
                Ok((Angle(out), Uint(out), Angle(out)))
            }
            (Int(a) | Uint(a), Angle(b)) => {
                let out = a.max(b);
                Ok((Uint(out), Angle(out), Angle(out)))
            }
            _ => {
                let ty = promote_arithmetic(lht, rht)
                    .ok_or_else(|| unsupported_scalar_binop::<Self>(lht, rht))?;
                if matches!(ty, Uint(_) | Int(_) | Float(_) | Complex(_) | BitReg(_)) {
                    Ok((ty, ty, ty))
                } else {
                    Err(unsupported_scalar_binop::<Self>(lht, rht))
                }
            }
        }
    }
    fn scalar_op(lhs: Scalar, rhs: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        use Primitive::*;
        let result = match (lhs.value(), rhs.value()) {
            (Uint(lhs), Uint(rhs)) => Uint(lhs.checked_mul(rhs).ok_or(Error::Overflow)?),
            (Int(lhs), Int(rhs)) => Int(lhs.checked_mul(rhs).ok_or(Error::Overflow)?),
            (Float(lhs), Float(rhs)) => Float(lhs * rhs),
            (Complex(lhs), Complex(rhs)) => Complex(lhs * rhs),
            (Angle(lhs), Uint(rhs)) | (Uint(lhs), Angle(rhs)) => Angle(lhs.wrapping_mul(rhs)),
            (Duration(lhs), Float(rhs)) => Duration(lhs * rhs),
            (Float(lhs), Duration(rhs)) => Duration(lhs * rhs),
            _ => return Err(unsupported_scalar_binop::<Self>(lhs.ty(), rhs.ty())),
        };
        Scalar::new(result.assert_fits(out)?, out)
    }

    fn arr_arr_check(lhs: ArrayTy, rhs: ArrayTy) -> Result<(ArrayTy, ArrayTy, ArrayTy)> {
        let (lht, rht, ty) = Self::scalar_check(lhs.ty(), rhs.ty())?;
        let lhs = lhs.with_ty(lht);
        let rhs = rhs.with_ty(rht);
        let plan = arr_arr_plan(lhs.shape().get(), rhs.shape().get()).ok_or_else(|| {
            Error::unsupported_binop(Mul::NAME, lhs.into(), rhs.into(), Mul::IS_FUNC)
        })?;
        let out = ArrayTy::new(ty, ArrayShape::new(plan.output_shape())?);
        Ok((lhs, rhs, out))
    }

    fn arr_arr_op(lhs: Array, rhs: Array, out: ArrayTy) -> Result<Array> {
        let lhs_ty = lhs.ty();
        let rhs_ty = rhs.ty();
        let plan = arr_arr_plan(lhs_ty.shape().get(), rhs_ty.shape().get()).ok_or_else(|| {
            Error::unsupported_binop(Mul::NAME, lhs_ty.into(), rhs_ty.into(), Mul::IS_FUNC)
        })?;
        let out_scalar_ty = out.ty();

        let ae = lhs.scalars().collect::<Vec<_>>();
        let be = rhs.scalars().collect::<Vec<_>>();

        let batch_total: usize = plan.batch_out.iter().product();
        let a_mat = plan.m * plan.k;
        let b_mat = plan.k * plan.p;
        let mut elements = Vec::with_capacity(batch_total * plan.m * plan.p);

        let batch_a = if plan.squeeze_a {
            vec![]
        } else {
            lhs_ty.shape().get()[..lhs_ty.shape().dim().get() - 2].to_vec()
        };
        let batch_b = if plan.squeeze_b {
            vec![]
        } else {
            rhs_ty.shape().get()[..rhs_ty.shape().dim().get() - 2].to_vec()
        };

        for batch_flat in 0..batch_total {
            let coords = flat_to_coords(&plan.batch_out, batch_flat);
            let a_base = batch_offset(&coords, &batch_a, plan.batch_out.len()) * a_mat;
            let b_base = batch_offset(&coords, &batch_b, plan.batch_out.len()) * b_mat;

            for i in 0..plan.m {
                for j in 0..plan.p {
                    elements.push(
                        dot_product(
                            (&ae, a_base + i * plan.k, 1),
                            (&be, b_base + j, plan.p),
                            plan.k,
                            out_scalar_ty,
                        )?
                        .value(),
                    );
                }
            }
        }

        Array::new(elements, out)
    }
}

fn dot_product(
    // a: (array, offset, stride)
    a: (&[Scalar], usize, usize),
    // b: (array, offset, stride)
    b: (&[Scalar], usize, usize),
    n: usize,
    out: PrimitiveTy,
) -> Result<Scalar> {
    if n == 0 {
        return Err(Error::UnsupportedOp {
            op: Mul::NAME,
            args: vec![
                ArrayTy::new(out, shape![0]).into(),
                ArrayTy::new(out, shape![0]).into(),
            ],
            is_func: Mul::IS_FUNC,
        });
    }
    let mut acc = Mul::scalar_op(a.0[a.1], b.0[b.1], out)?;
    for k in 1..n {
        let prod = Mul::scalar_op(a.0[a.1 + k * a.2], b.0[b.1 + k * b.2], out)?;
        acc = Add::scalar_op(acc, prod, out)?;
    }
    Ok(acc)
}

/// Broadcast two shapes. Returns None if incompatible.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let ad = a.len().checked_sub(len - i).map_or(1, |j| a[j]);
        let bd = b.len().checked_sub(len - i).map_or(1, |j| b[j]);
        if ad == bd {
            out.push(ad);
        } else if ad == 1 {
            out.push(bd);
        } else if bd == 1 {
            out.push(ad);
        } else {
            return None;
        }
    }
    Some(out)
}

/// Convert flat index to multi-dimensional coordinates.
fn flat_to_coords(shape: &[usize], mut flat: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        coords[i] = flat % shape[i];
        flat /= shape[i];
    }
    coords
}

/// Compute flat batch offset in a source array given broadcast output coordinates.
fn batch_offset(out_coords: &[usize], src_batch: &[usize], out_len: usize) -> usize {
    let offset = out_len - src_batch.len();
    let mut flat = 0;
    for (i, &d) in src_batch.iter().enumerate() {
        let coord = if d == 1 { 0 } else { out_coords[i + offset] };
        flat = flat * d + coord;
    }
    flat
}

impl Value {
    pub fn mul_(self, rhs: Self) -> Result<Self> {
        Mul::checked_op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ashape;
    use crate::duration::DurationUnit;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};

    fn u_scalar(v: u128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::uint(v), Uint(bw(bits))))
    }

    fn i_scalar(v: i128, bits: u32) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::int(v), Int(bw(bits))))
    }

    fn f64_scalar(v: f64) -> Value {
        Value::Scalar(Scalar::new_unchecked(Primitive::float(v), Float(F64)))
    }

    fn u_array(values: &[u128], bits: u32, shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::uint(v)).collect(),
            ArrayTy::new(Uint(bw(bits)), ashape(shape)),
        ))
    }

    fn f64_array(values: &[f64], shape: Vec<usize>) -> Value {
        Value::Array(Array::new_unchecked(
            values.iter().map(|&v| Primitive::float(v)).collect(),
            ArrayTy::new(Float(F64), ashape(shape)),
        ))
    }

    // --- Scalar * Scalar ---

    #[test]
    fn uint_mul() {
        let r = u_scalar(3, 8).mul_(u_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_uint(bw(8)).unwrap(), 12),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_overflow_returns_none() {
        assert!(u_scalar(200, 8).mul_(u_scalar(2, 8)).is_err());
    }

    #[test]
    fn int_mul() {
        let r = i_scalar(-3, 8).mul_(i_scalar(4, 8)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_int(bw(8)).unwrap(), -12),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn int_overflow_returns_none() {
        assert!(i_scalar(100, 8).mul_(i_scalar(2, 8)).is_err());
    }

    #[test]
    fn f64_mul() {
        let r = f64_scalar(1.5).mul_(f64_scalar(4.0)).unwrap();
        match r {
            Value::Scalar(s) => assert_eq!(s.value().as_float(F64).unwrap(), 6.0),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn c64_mul() {
        // (1+2i) * (3-1i) = 5+5i
        let a = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(1.0, 2.0),
            Complex(F64),
        ));
        let b = Value::Scalar(Scalar::new_unchecked(
            Primitive::complex(3.0, -1.0),
            Complex(F64),
        ));
        let r = a.mul_(b).unwrap();
        match r {
            Value::Scalar(s) => {
                let c = s.value().as_complex(F64).unwrap();
                assert_eq!(c.re, 5.0);
                assert_eq!(c.im, 5.0);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Invalid types ---

    #[test]
    fn bit_mul_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit));
        assert!(a.mul_(b).is_err());
    }

    #[test]
    fn duration_mul_returns_none() {
        let dur = |v| {
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(v, DurationUnit::Ns),
                Duration,
            ))
        };
        assert!(dur(5.0).mul_(dur(2.0)).is_err());
    }

    #[test]
    fn angle_mul_returns_none() {
        let a = Value::Scalar(Scalar::new_unchecked(Primitive::uint(3_u128), Angle(bw(8))));
        let b = Value::Scalar(Scalar::new_unchecked(Primitive::uint(2_u128), Angle(bw(8))));
        assert!(a.mul_(b).is_err());
    }

    // --- Type promotion ---

    #[test]
    fn uint8_mul_uint16_promotes() {
        let r = u_scalar(100, 8).mul_(u_scalar(300, 16)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Uint(n) if n.get() == 16));
                assert_eq!(s.value().as_uint(bw(16)).unwrap(), 30_000);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn uint_mul_float_promotes() {
        let r = u_scalar(3, 8).mul_(f64_scalar(0.5)).unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert_eq!(s.value().as_float(F64).unwrap(), 1.5);
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Dot product: [n] · [n] → scalar ---

    #[test]
    fn dot_product_1d() {
        // [1,2,3] · [4,5,6] = 4+10+18 = 32
        let a = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        let b = f64_array(&[4.0, 5.0, 6.0], vec![3]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(s) => {
                assert_eq!(s.ty().shape().get(), &[1]);
                let value = s.scalars().next().unwrap().value();
                assert_eq!(value.as_float(F64).unwrap(), 32.0)
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn dot_product_length_mismatch_returns_none() {
        let a = f64_array(&[1.0, 2.0], vec![2]);
        let b = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        assert!(a.mul_(b).is_err());
    }

    // --- Matrix-vector: [m, n] · [n] → [m] ---

    #[test]
    fn matrix_vector_mul() {
        // [[1,2],[3,4]] · [5,6] = [1*5+2*6, 3*5+4*6] = [17, 39]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[5.0, 6.0], vec![2]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![17.0, 39.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn matrix_vector_dim_mismatch_returns_none() {
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        assert!(a.mul_(b).is_err());
    }

    // --- Vector-matrix: [n] · [n, p] → [p] ---

    #[test]
    fn vector_matrix_mul() {
        // [1,2] · [[3,4,5],[6,7,8]] = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        let a = f64_array(&[1.0, 2.0], vec![2]);
        let b = f64_array(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 3]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[3]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![15.0, 18.0, 21.0]);
            }
            _ => panic!("expected array"),
        }
    }

    // --- Matrix multiply: [m, n] · [n, p] → [m, p] ---

    #[test]
    fn matrix_matrix_mul() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![19.0, 22.0, 43.0, 50.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn matrix_mul_non_square() {
        // [2,3] · [3,2] → [2,2]
        // [[1,2,3],[4,5,6]] · [[7,8],[9,10],[11,12]]
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = f64_array(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![58.0, 64.0, 139.0, 154.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn matrix_mul_inner_dim_mismatch_returns_none() {
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        assert!(a.mul_(b).is_err());
    }

    // --- Batch matmul ---

    #[test]
    fn batch_matmul_3d() {
        // [2,2,2] · [2,2,2]: batch of 2 matrix muls, all ones
        // each 2x2 ones @ 2x2 ones = [[2,2],[2,2]]
        let a = f64_array(&[1.0; 8], vec![2, 2, 2]);
        let b = f64_array(&[1.0; 8], vec![2, 2, 2]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[2, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![2.0; 8]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_broadcast_left() {
        // a=[2,2] single matrix, b=[3,2,2] batch of 3
        // a = [[1,2],[3,4]], b = [I, 2I, swap]
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = f64_array(
            &[
                1.0, 0.0, 0.0, 1.0, // I
                2.0, 0.0, 0.0, 2.0, // 2I
                0.0, 1.0, 1.0, 0.0, // swap
            ],
            vec![3, 2, 2],
        );
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[3, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(
                    vals,
                    vec![
                        1.0, 2.0, 3.0, 4.0, // a @ I
                        2.0, 4.0, 6.0, 8.0, // a @ 2I
                        2.0, 1.0, 4.0, 3.0, // a @ swap
                    ]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_broadcast_right() {
        // a=[3,2,2] batch of 3 identities, b=[2,2] single matrix
        let a = f64_array(
            &[
                1.0, 0.0, 0.0, 1.0, // I
                1.0, 0.0, 0.0, 1.0, // I
                1.0, 0.0, 0.0, 1.0, // I
            ],
            vec![3, 2, 2],
        );
        let b = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[3, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                // I @ b = b, repeated 3 times
                assert_eq!(
                    vals,
                    vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_broadcast_ones() {
        // a=[1,2,2] (batch 1), b=[2,2,2] (batch 2)
        // a's single matrix broadcasts across both batches of b
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let b = f64_array(
            &[
                1.0, 0.0, 0.0, 1.0, // I
                0.0, 1.0, 1.0, 0.0, // swap
            ],
            vec![2, 2, 2],
        );
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[2, 2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(
                    vals,
                    vec![
                        1.0, 2.0, 3.0, 4.0, // a @ I
                        2.0, 1.0, 4.0, 3.0, // a @ swap
                    ]
                );
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn batch_incompatible_returns_none() {
        let a = f64_array(&[1.0; 12], vec![3, 2, 2]);
        let b = f64_array(&[1.0; 8], vec![2, 2, 2]);
        assert!(a.mul_(b).is_err());
    }

    // --- Overflow in dot product ---

    #[test]
    fn dot_product_overflow_returns_none() {
        // 200*200 = 40000 overflows Uint(8)
        let a = u_array(&[200], 8, vec![1]);
        let b = u_array(&[200], 8, vec![1]);
        assert!(a.mul_(b).is_err());
    }

    #[test]
    fn dot_product_accumulation_overflow_returns_none() {
        // 100*1 + 100*1 = 200, but products fit individually; sum would too in u16
        // but 200*1 + 200*1 at u8: 200+200=400 overflows
        let a = u_array(&[200, 200], 16, vec![2]);
        let b = u_array(&[1, 1], 16, vec![2]);
        // 200 + 200 = 400, fits in u16. Should succeed.
        let r = a.mul_(b).unwrap();
        match r {
            Value::Array(arr) => {
                assert_eq!(arr.ty().shape().get(), &[1]);
                let value = arr.scalars().next().unwrap().value();
                assert_eq!(value.as_uint(bw(16)).unwrap(), 400)
            }
            _ => panic!("expected scalar"),
        }
    }

    // --- Scalar * Array broadcast (scaling) ---

    #[test]
    fn scalar_mul_array_broadcast() {
        let s = f64_scalar(2.0);
        let a = f64_array(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = s.mul_(a).unwrap();
        match r {
            Value::Array(a) => {
                assert_eq!(a.ty().shape().get(), &[2, 2]);
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![2.0, 4.0, 6.0, 8.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_mul_scalar_broadcast() {
        let a = f64_array(&[1.0, 2.0, 3.0], vec![3]);
        let s = f64_scalar(3.0);
        let r = a.mul_(s).unwrap();
        match r {
            Value::Array(a) => {
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![3.0, 6.0, 9.0]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn scalar_broadcast_with_promotion() {
        let s = f64_scalar(0.5);
        let a = u_array(&[2, 4, 6], 8, vec![3]);
        let r = s.mul_(a).unwrap();
        match r {
            Value::Array(a) => {
                assert!(matches!(a.ty().ty(), Float(F64)));
                let vals: Vec<f64> = a
                    .values()
                    .iter()
                    .map(|s| s.as_float(F64).unwrap())
                    .collect();
                assert_eq!(vals, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("expected array"),
        }
    }
}
