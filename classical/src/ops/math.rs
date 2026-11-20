//! The built-in mathematical functions (docs/types.rst, "Built-in
//! constant expression functions").

use num_complex::Complex64;

use super::{UnOp, UnSignature};
use crate::primitive::{FloatWidth, PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::{Primitive, Result};

/// The real function behind a float built-in, or `None` for the others.
fn float_fn(op: UnOp) -> Option<fn(f64) -> f64> {
    Some(match op {
        UnOp::Sin => f64::sin,
        UnOp::Cos => f64::cos,
        UnOp::Tan => f64::tan,
        UnOp::Arcsin => f64::asin,
        UnOp::Arccos => f64::acos,
        UnOp::Arctan => f64::atan,
        UnOp::Exp => f64::exp,
        UnOp::Log => f64::ln,
        UnOp::Sqrt => f64::sqrt,
        UnOp::Ceiling => f64::ceil,
        UnOp::Floor => f64::floor,
        _ => return None,
    })
}

/// The complex overload of a built-in, where the spec gives one.
fn complex_fn(op: UnOp) -> Option<fn(Complex64) -> Complex64> {
    match op {
        UnOp::Exp => Some(Complex64::exp),
        UnOp::Sqrt => Some(Complex64::sqrt),
        _ => None,
    }
}

pub(super) fn signature(op: UnOp, arg: PrimitiveTy) -> Option<UnSignature> {
    use PrimitiveTy::*;
    let real = Float(FloatWidth::F64);
    if matches!(op, UnOp::Real | UnOp::Imag) {
        // Anything that promotes to a complex number has parts.
        let width = arg.fw().unwrap_or(FloatWidth::F64);
        return match promote_arithmetic(arg, Complex(width))? {
            Complex(width) => Some((Complex(width), Float(width))),
            _ => None,
        };
    }
    float_fn(op)?;
    match arg {
        Float(_) => Some((arg, arg)),
        // Integers promote to a float implicitly, as for any float input.
        Int(_) | Uint(_) => Some((real, real)),
        Angle(_) if matches!(op, UnOp::Sin | UnOp::Cos | UnOp::Tan) => Some((real, real)),
        Complex(_) if complex_fn(op).is_some() => Some((arg, arg)),
        _ => None,
    }
}

pub(super) fn eval(op: UnOp, arg: &Scalar) -> Result<Primitive> {
    Ok(match (op, arg.value()) {
        (UnOp::Real, Primitive::Complex(c)) => Primitive::Float(c.re),
        (UnOp::Imag, Primitive::Complex(c)) => Primitive::Float(c.im),
        (_, Primitive::Float(v)) if let Some(f) = float_fn(op) => Primitive::Float(f(*v)),
        (_, Primitive::Complex(c)) if let Some(f) = complex_fn(op) => Primitive::Complex(f(*c)),
        _ => return Err(op.unsupported(arg.ty())),
    })
}
