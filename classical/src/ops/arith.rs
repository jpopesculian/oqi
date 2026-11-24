//! Arithmetic: `+ - * / % **`, unary `-`, and the array matrix product.

use super::{BinOp, BinSignature, UnOp, UnSignature};
use crate::array::{Array, ArrayShape, ArrayTy};
use crate::index::flat_to_multi;
use crate::primitive::{FloatWidth, PrimitiveTy, promote_arithmetic, resize_int};
use crate::scalar::Scalar;
use crate::{Error, Primitive, Result};

pub(super) fn signature(op: BinOp, lhs: PrimitiveTy, rhs: PrimitiveTy) -> Option<BinSignature> {
    use BinOp::*;
    use PrimitiveTy::*;
    let real = Float(FloatWidth::F64);
    Some(match (op, lhs, rhs) {
        // A duration scales by a real number; two durations have a ratio.
        (Mul | Div, Duration, Int(_) | Uint(_) | Float(_)) => (Duration, real, Duration),
        (Mul, Int(_) | Uint(_) | Float(_), Duration) => (real, Duration, Duration),
        (Div, Duration, Duration) => (Duration, Duration, real),
        // An angle scales by an unsigned integer; two angles have a ratio.
        (Mul | Div, Angle(a), Int(b) | Uint(b)) => {
            let width = a.max(b);
            (Angle(width), Uint(width), Angle(width))
        }
        (Mul, Int(a) | Uint(a), Angle(b)) => {
            let width = a.max(b);
            (Uint(width), Angle(width), Angle(width))
        }
        (Div, Angle(a), Angle(b)) => {
            let width = a.max(b);
            (Angle(width), Angle(width), Uint(width))
        }
        // Otherwise both operands promote to a common type the operator
        // is defined on.
        _ => {
            let ty = promote_arithmetic(lhs, rhs)?;
            let defined = match op {
                Add | Sub => matches!(
                    ty,
                    Uint(_) | Int(_) | Float(_) | Complex(_) | Duration | Angle(_)
                ),
                Mul | Div | Pow => matches!(ty, Uint(_) | Int(_) | Float(_) | Complex(_)),
                Mod => matches!(ty, Uint(_) | Int(_) | Float(_)),
                _ => false,
            };
            defined.then_some((ty, ty, ty))?
        }
    })
}

pub(super) fn eval(op: BinOp, lhs: &Scalar, rhs: &Scalar, out: PrimitiveTy) -> Result<Primitive> {
    use BinOp::*;
    use Primitive::*;
    let unsupported = || op.unsupported(lhs.ty(), rhs.ty());
    let value = match (op, lhs.value(), rhs.value()) {
        (_, Uint(a), Uint(b)) => Uint(uint(op, *a, *b).ok_or_else(unsupported)??),
        (_, Int(a), Int(b)) => Int(int(op, *a, *b).ok_or_else(unsupported)??),

        (Add, Float(a), Float(b)) => Float(a + b),
        (Sub, Float(a), Float(b)) => Float(a - b),
        (Mul, Float(a), Float(b)) => Float(a * b),
        (Div, Float(a), Float(b)) => Float(a / b),
        (Mod, Float(a), Float(b)) => Float(a % b),
        (Pow, Float(a), Float(b)) => Float(a.powf(*b)),

        (Add, Complex(a), Complex(b)) => Complex(a + b),
        (Sub, Complex(a), Complex(b)) => Complex(a - b),
        (Mul, Complex(a), Complex(b)) => Complex(a * b),
        (Div, Complex(a), Complex(b)) => Complex(a / b),
        (Pow, Complex(a), Complex(b)) => Complex(a.powc(*b)),

        (Add, Duration(a), Duration(b)) => Duration(*a + *b),
        (Sub, Duration(a), Duration(b)) => Duration(*a - *b),
        (Div, Duration(a), Duration(b)) => Float(*a / *b),
        (Mul, Duration(a), Float(b)) => Duration(*a * *b),
        (Mul, Float(a), Duration(b)) => Duration(*a * *b),
        (Div, Duration(a), Float(b)) => Duration(*a / *b),

        // Angles are stored left-aligned, so these act on the turns
        // numerator exactly as on a `uint` of the same width.
        (Add, Angle(a), Angle(b)) => Angle(*a + *b),
        (Sub, Angle(a), Angle(b)) => Angle(*a - *b),
        (Mul, Angle(a), Uint(b)) | (Mul, Uint(b), Angle(a)) => Angle(*a * *b),
        (Div, Angle(a), Uint(b)) => Angle(turns::Angle(
            a.0.checked_div(*b).ok_or(Error::DivideByZero)?,
        )),
        (Div, Angle(a), Angle(b)) => Uint(a.0.checked_div(b.0).ok_or(Error::DivideByZero)?),

        _ => return Err(unsupported()),
    };
    // Signed overflow is an error. Unsigned arithmetic is modulo 2^width
    // (C99, which the spec's standard types follow): converting to `out`
    // wraps it.
    if let (Int(v), PrimitiveTy::Int(width)) = (&value, out)
        && resize_int(*v, width) != *v
    {
        return Err(Error::Overflow);
    }
    Ok(value)
}

/// Unsigned arithmetic, wrapping (the caller masks to the operand width).
fn uint(op: BinOp, a: u128, b: u128) -> Option<Result<u128>> {
    Some(Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => return Some(a.checked_div(b).ok_or(Error::DivideByZero)),
        BinOp::Mod => return Some(a.checked_rem(b).ok_or(Error::DivideByZero)),
        BinOp::Pow => {
            // Square-and-multiply: the exponent may exceed `u32`.
            let (mut base, mut exp, mut acc) = (a, b, 1u128);
            while exp != 0 {
                if exp & 1 == 1 {
                    acc = acc.wrapping_mul(base);
                }
                base = base.wrapping_mul(base);
                exp >>= 1;
            }
            acc
        }
        _ => return None,
    }))
}

/// Signed arithmetic in the 128-bit carrier; overflow of the carrier is an
/// error here, overflow of a narrower width is caught by [`eval`].
fn int(op: BinOp, a: i128, b: i128) -> Option<Result<i128>> {
    let div_error = if b == 0 {
        Error::DivideByZero
    } else {
        Error::Overflow
    };
    Some(match op {
        BinOp::Add => a.checked_add(b).ok_or(Error::Overflow),
        BinOp::Sub => a.checked_sub(b).ok_or(Error::Overflow),
        BinOp::Mul => a.checked_mul(b).ok_or(Error::Overflow),
        BinOp::Div => a.checked_div(b).ok_or(div_error),
        BinOp::Mod => a.checked_rem(b).ok_or(div_error),
        // An integer raised to a negative power truncates to zero.
        BinOp::Pow if b < 0 => Ok(0),
        BinOp::Pow => u32::try_from(b)
            .ok()
            .and_then(|b| a.checked_pow(b))
            .ok_or(Error::Overflow),
        _ => return None,
    })
}

pub(super) fn neg_signature(arg: PrimitiveTy) -> Option<UnSignature> {
    use PrimitiveTy::*;
    matches!(arg, Int(_) | Float(_) | Complex(_) | Duration | Angle(_)).then_some((arg, arg))
}

pub(super) fn neg(arg: &Scalar) -> Result<Primitive> {
    Ok(match arg.value() {
        Primitive::Int(v) => Primitive::Int(v.checked_neg().ok_or(Error::Overflow)?),
        Primitive::Float(v) => Primitive::Float(-*v),
        Primitive::Complex(v) => Primitive::Complex(-*v),
        Primitive::Duration(v) => Primitive::Duration(-*v),
        Primitive::Angle(v) => Primitive::Angle(-*v),
        _ => return Err(UnOp::Neg.unsupported(arg.ty())),
    })
}

// ── Matrix product ──────────────────────────────────────────────────────

/// How two array shapes combine under `*`: the trailing two dimensions are
/// matrices (a 1-D operand is a row/column vector and its dimension is
/// squeezed from the result); leading dimensions broadcast.
struct MatmulPlan {
    squeeze_a: bool,
    squeeze_b: bool,
    m: usize,
    k: usize,
    p: usize,
    batch_a: Vec<usize>,
    batch_b: Vec<usize>,
    batch_out: Vec<usize>,
}

impl MatmulPlan {
    fn new(lhs: &[usize], rhs: &[usize]) -> Option<Self> {
        let squeeze_a = lhs.len() == 1;
        let squeeze_b = rhs.len() == 1;
        let (batch_a, m, k) = match lhs {
            [k] => (&[][..], 1, *k),
            [batch @ .., m, k] => (batch, *m, *k),
            [] => return None,
        };
        let (batch_b, k_b, p) = match rhs {
            [k] => (&[][..], *k, 1),
            [batch @ .., k, p] => (batch, *k, *p),
            [] => return None,
        };
        if k == 0 || k != k_b {
            return None;
        }
        Some(Self {
            squeeze_a,
            squeeze_b,
            m,
            k,
            p,
            batch_out: broadcast_shape(batch_a, batch_b)?,
            batch_a: batch_a.to_vec(),
            batch_b: batch_b.to_vec(),
        })
    }

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

/// The shape of `lhs * rhs`, or `None` if the shapes do not multiply.
pub(super) fn matmul_shape(lhs: ArrayShape, rhs: ArrayShape) -> Option<ArrayShape> {
    let plan = MatmulPlan::new(lhs.dims(), rhs.dims())?;
    ArrayShape::new(plan.output_shape()).ok()
}

/// The rank of `lhs * rhs` when only the operand ranks are known.
pub(super) fn matmul_rank(lhs: usize, rhs: usize) -> usize {
    let batch = lhs.saturating_sub(2).max(rhs.saturating_sub(2));
    (batch + usize::from(lhs > 1) + usize::from(rhs > 1)).max(1)
}

pub(super) fn matmul(lhs: &Array, rhs: &Array, out: ArrayTy) -> Result<Array> {
    let (lhs_ty, rhs_ty) = (lhs.ty(), rhs.ty());
    let plan = MatmulPlan::new(lhs_ty.shape().dims(), rhs_ty.shape().dims())
        .ok_or_else(|| BinOp::Mul.unsupported(lhs_ty, rhs_ty))?;
    let elem = out.element_ty();
    let a: Vec<Scalar> = lhs.scalars().collect();
    let b: Vec<Scalar> = rhs.scalars().collect();

    let batch_total: usize = plan.batch_out.iter().product();
    let mut elements = Vec::with_capacity(batch_total * plan.m * plan.p);
    for batch_flat in 0..batch_total {
        let coords = flat_to_multi(&plan.batch_out, batch_flat);
        let a_base = batch_offset(&coords, &plan.batch_a) * plan.m * plan.k;
        let b_base = batch_offset(&coords, &plan.batch_b) * plan.k * plan.p;
        for i in 0..plan.m {
            for j in 0..plan.p {
                // Row i of a (stride 1) · column j of b (stride p).
                let term = |k: usize| {
                    BinOp::Mul.scalar(
                        &a[a_base + i * plan.k + k],
                        &b[b_base + j + k * plan.p],
                        elem,
                    )
                };
                let mut acc = term(0)?;
                for k in 1..plan.k {
                    acc = BinOp::Add.scalar(&acc, &term(k)?, elem)?;
                }
                elements.push(acc.into_value());
            }
        }
    }
    Array::new(elements, out)
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

/// Flat batch offset in a source array for broadcast output coordinates.
fn batch_offset(out_coords: &[usize], src_batch: &[usize]) -> usize {
    let offset = out_coords.len() - src_batch.len();
    let mut flat = 0;
    for (i, &d) in src_batch.iter().enumerate() {
        let coord = if d == 1 { 0 } else { out_coords[i + offset] };
        flat = flat * d + coord;
    }
    flat
}
