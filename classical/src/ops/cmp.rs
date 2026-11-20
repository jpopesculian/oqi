//! Comparisons (`== != < > <= >=`) and Boolean logic (`&& || !`).

use std::cmp::Ordering;

use super::{BinOp, BinSignature, UnOp, UnSignature};
use crate::primitive::{PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::{Primitive, Result};

pub(super) fn signature(op: BinOp, lhs: PrimitiveTy, rhs: PrimitiveTy) -> Option<BinSignature> {
    use BinOp::*;
    use PrimitiveTy::*;
    match op {
        LogAnd | LogOr => (lhs == Bool && rhs == Bool).then_some((Bool, Bool, Bool)),
        // Any type is comparable for equality with itself.
        Eq | Neq if lhs.is_compatible(rhs) => Some((lhs, rhs, Bool)),
        Eq | Neq => promote_arithmetic(lhs, rhs).map(|ty| (ty, ty, Bool)),
        // docs/classical.rst: "Integers, angles, bits, and classical
        // registers can be compared".
        Lt | Gt | Lte | Gte => match (lhs, rhs) {
            (Bit, Bit) => Some((lhs, rhs, Bool)),
            (BitReg(a), BitReg(b)) => (a == b).then_some((lhs, rhs, Bool)),
            _ => {
                let ty = promote_arithmetic(lhs, rhs)?;
                matches!(ty, Uint(_) | Int(_) | Float(_) | Duration | Angle(_))
                    .then_some((ty, ty, Bool))
            }
        },
        _ => None,
    }
}

pub(super) fn eval(op: BinOp, lhs: &Scalar, rhs: &Scalar) -> Result<Primitive> {
    use BinOp::*;
    let unsupported = || op.unsupported(lhs.ty(), rhs.ty());
    let (a, b) = (lhs.value(), rhs.value());
    let result = match op {
        LogAnd | LogOr => match (a, b) {
            (Primitive::Bit(a), Primitive::Bit(b)) if op == LogAnd => *a && *b,
            (Primitive::Bit(a), Primitive::Bit(b)) => *a || *b,
            _ => return Err(unsupported()),
        },
        Eq => equal(a, b).ok_or_else(unsupported)?,
        Neq => !equal(a, b).ok_or_else(unsupported)?,
        // An unordered pair (a NaN operand) compares false.
        Lt | Gt | Lte | Gte => match order(a, b).ok_or_else(unsupported)? {
            Some(Ordering::Less) => matches!(op, Lt | Lte),
            Some(Ordering::Equal) => matches!(op, Lte | Gte),
            Some(Ordering::Greater) => matches!(op, Gt | Gte),
            None => false,
        },
        _ => return Err(unsupported()),
    };
    Ok(Primitive::Bit(result))
}

/// Equality of two values of the same kind; `None` for mixed kinds.
fn equal(a: &Primitive, b: &Primitive) -> Option<bool> {
    use Primitive::*;
    Some(match (a, b) {
        (Bit(a), Bit(b)) => a == b,
        (BitReg(a), BitReg(b)) => a == b,
        (Uint(a), Uint(b)) => a == b,
        (Int(a), Int(b)) => a == b,
        (Float(a), Float(b)) => a == b,
        (Complex(a), Complex(b)) => a == b,
        (Duration(a), Duration(b)) => a == b,
        (Angle(a), Angle(b)) => a == b,
        _ => return None,
    })
}

/// Ordering of two values of the same ordered kind; the outer `None` is
/// for kinds with no order, the inner for unordered values.
fn order(a: &Primitive, b: &Primitive) -> Option<Option<Ordering>> {
    use Primitive::*;
    Some(match (a, b) {
        (Bit(a), Bit(b)) => Some(a.cmp(b)),
        (BitReg(a), BitReg(b)) => Some(a.cmp(b)),
        (Uint(a), Uint(b)) => Some(a.cmp(b)),
        (Int(a), Int(b)) => Some(a.cmp(b)),
        (Float(a), Float(b)) => a.partial_cmp(b),
        (Duration(a), Duration(b)) => a.partial_cmp(b),
        (Angle(a), Angle(b)) => Some(a.0.cmp(&b.0)),
        _ => return None,
    })
}

pub(super) fn not_signature(arg: PrimitiveTy) -> Option<UnSignature> {
    (arg == PrimitiveTy::Bool).then_some((arg, arg))
}

pub(super) fn not(arg: &Scalar) -> Result<Primitive> {
    match arg.value() {
        Primitive::Bit(v) => Ok(Primitive::Bit(!v)),
        _ => Err(UnOp::LogNot.unsupported(arg.ty())),
    }
}
