//! Bitwise operators: `& | ^ ~`, shifts, rotations and `popcount`.

use super::{BinOp, BinSignature, UnOp, UnSignature};
use crate::primitive::{IntWidth, PrimitiveTy, angle_bits, angle_from_bits, promote_arithmetic};
use crate::scalar::Scalar;
use crate::{Primitive, Result};

pub(super) fn signature(op: BinOp, lhs: PrimitiveTy, rhs: PrimitiveTy) -> Option<BinSignature> {
    use BinOp::*;
    use PrimitiveTy::*;
    let integer = matches!(rhs, Uint(_) | Int(_));
    match op {
        BitAnd | BitOr | BitXor => match (lhs, rhs) {
            (BitReg(a), BitReg(b)) => (a == b).then_some((lhs, rhs, lhs)),
            _ => {
                let ty = promote_arithmetic(lhs, rhs)?;
                matches!(ty, Uint(_) | Int(_)).then_some((ty, ty, ty))
            }
        },
        // Shifts and rotations keep the left operand's type and take any
        // integer distance (a negative one reverses the direction).
        Shl | Shr => (integer && matches!(lhs, Uint(_) | Int(_) | BitReg(_) | Angle(_)))
            .then_some((lhs, rhs, lhs)),
        Rotl | Rotr => (integer && matches!(lhs, Uint(_) | BitReg(_))).then_some((lhs, rhs, lhs)),
        _ => None,
    }
}

pub(super) fn eval(op: BinOp, lhs: &Scalar, rhs: &Scalar) -> Result<Primitive> {
    use BinOp::*;
    use Primitive::*;
    let unsupported = || op.unsupported(lhs.ty(), rhs.ty());
    Ok(match (op, lhs.value(), rhs.value()) {
        (BitAnd, BitReg(a), BitReg(b)) => BitReg(a & b),
        (BitAnd, Uint(a), Uint(b)) => Uint(a & b),
        (BitAnd, Int(a), Int(b)) => Int(a & b),
        (BitOr, BitReg(a), BitReg(b)) => BitReg(a | b),
        (BitOr, Uint(a), Uint(b)) => Uint(a | b),
        (BitOr, Int(a), Int(b)) => Int(a | b),
        (BitXor, BitReg(a), BitReg(b)) => BitReg(a ^ b),
        (BitXor, Uint(a), Uint(b)) => Uint(a ^ b),
        (BitXor, Int(a), Int(b)) => Int(a ^ b),

        (Shl | Shr, value, distance) => {
            // `left` is the direction after folding in the distance's sign.
            let (left, amount) = match distance {
                Uint(d) => (op == Shl, *d),
                Int(d) => ((op == Shl) == (*d >= 0), d.unsigned_abs()),
                _ => return Err(unsupported()),
            };
            // Every type here is at most 128 bits wide, or a register
            // whose shift saturates: larger distances all shift every bit out.
            let amount = u32::try_from(amount).unwrap_or(u32::MAX);
            let wide = amount >= u128::BITS;
            match (value, left) {
                (BitReg(a), true) => BitReg(a << amount),
                (BitReg(a), false) => BitReg(a >> amount),
                (Uint(_), _) if wide => Uint(0),
                (Uint(a), true) => Uint(a << amount),
                (Uint(a), false) => Uint(a >> amount),
                (Int(_), true) if wide => Int(0),
                (Int(a), true) => Int(a << amount),
                // Arithmetic shift: the sign bit fills in.
                (Int(a), false) => Int(a >> amount.min(u128::BITS - 1)),
                // An angle shifts as its turns numerator, a `uint` of the
                // same width.
                (Angle(_), _) if wide => Angle(turns::Angle(0)),
                (Angle(a), _) => {
                    let width = lhs.ty().bit_count().ok_or_else(unsupported)?;
                    let bits = angle_bits(*a, width);
                    let bits = if left { bits << amount } else { bits >> amount };
                    Angle(angle_from_bits(bits, width))
                }
                _ => return Err(unsupported()),
            }
        }

        (Rotl | Rotr, value, distance) => {
            let width = lhs.ty().bit_count().ok_or_else(unsupported)?;
            let distance = match distance {
                Int(d) => *d,
                Uint(d) => *d as i128,
                _ => return Err(unsupported()),
            };
            if width == 0 {
                return Ok(value.clone());
            }
            // Rotating right by `d` is rotating left by `width - d`.
            let d = distance.rem_euclid(width as i128) as u32;
            let left = if op == Rotl { d } else { (width - d) % width };
            match value {
                BitReg(reg) => {
                    let mut reg = reg.clone();
                    reg.rotl(0..width, left as usize);
                    BitReg(reg)
                }
                Uint(v) if left == 0 => Uint(*v),
                Uint(v) => {
                    let mask = u128::MAX >> (u128::BITS - width);
                    let v = v & mask;
                    Uint(((v << left) | (v >> (width - left))) & mask)
                }
                _ => return Err(unsupported()),
            }
        }

        _ => return Err(unsupported()),
    })
}

pub(super) fn unary_signature(op: UnOp, arg: PrimitiveTy) -> Option<UnSignature> {
    use PrimitiveTy::*;
    match (op, arg) {
        (UnOp::BitNot, Bit | Uint(_) | Int(_) | BitReg(_)) => Some((arg, arg)),
        // The count of a register is as wide as an integer can hold it.
        (UnOp::Popcount, BitReg(n)) => Some((arg, Uint(IntWidth::new(n.clamp(1, 128)).ok()?))),
        (UnOp::Popcount, Uint(_)) => Some((arg, arg)),
        _ => None,
    }
}

pub(super) fn unary_eval(op: UnOp, arg: &Scalar) -> Result<Primitive> {
    use Primitive::*;
    Ok(match (op, arg.value()) {
        (UnOp::BitNot, Bit(v)) => Bit(!*v),
        (UnOp::BitNot, Uint(v)) => Uint(!*v),
        (UnOp::BitNot, Int(v)) => Int(!*v),
        (UnOp::BitNot, BitReg(v)) => BitReg(!v),
        (UnOp::Popcount, BitReg(reg)) => Uint(reg.count_ones() as u128),
        (UnOp::Popcount, Uint(v)) => Uint(v.count_ones() as u128),
        _ => return Err(op.unsupported(arg.ty())),
    })
}
