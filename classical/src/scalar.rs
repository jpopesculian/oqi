use std::fmt;

use crate::error::Result;
use crate::primitive::{BitWidth, FloatWidth, Primitive, PrimitiveTy};

pub type ScalarTy = PrimitiveTy;

#[derive(Clone, Copy, Debug)]
pub struct Scalar {
    value: Primitive,
    ty: ScalarTy,
}

impl Scalar {
    pub const PI: Self = Self::new_unchecked(Primitive::PI, PrimitiveTy::Float(FloatWidth::F64));
    pub const TAU: Self = Self::new_unchecked(Primitive::TAU, PrimitiveTy::Float(FloatWidth::F64));
    pub const E: Self = Self::new_unchecked(Primitive::E, PrimitiveTy::Float(FloatWidth::F64));

    pub fn new(value: Primitive, ty: ScalarTy) -> Result<Self> {
        Ok(Scalar::new_unchecked(value.as_ty(ty)?, ty))
    }

    #[inline]
    pub const fn new_unchecked(value: Primitive, ty: ScalarTy) -> Self {
        Scalar { value, ty }
    }

    #[inline]
    pub const fn value(&self) -> Primitive {
        self.value
    }

    #[inline]
    pub const fn ty(&self) -> ScalarTy {
        self.ty
    }

    #[inline]
    pub fn cast(self, to: ScalarTy) -> Result<Self> {
        self.ty.cast(to)?;
        Self::new(self.value, to)
    }
}

impl From<Primitive> for Scalar {
    #[inline]
    fn from(value: Primitive) -> Self {
        Self::new_unchecked(value, value.default_ty())
    }
}

fn angle_numden(angle: u128) -> (u128, u128) {
    if angle == 0 {
        return (0, 1);
    }

    let common_twos = angle.trailing_zeros().min(127);
    (angle >> common_twos, 1u128 << (127 - common_twos))
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value() {
            Primitive::Bit(v) => {
                if matches!(self.ty(), PrimitiveTy::Bool) {
                    write!(f, "{}", v)
                } else {
                    write!(f, "{}", if v { 1 } else { 0 })
                }
            }
            Primitive::Uint(v) => write!(f, "{}", v),
            Primitive::Int(v) => write!(f, "{}", v),
            Primitive::Float(v) => write!(f, "{}", v),
            Primitive::Duration(v) => write!(f, "{}", v),
            Primitive::Angle(v) => {
                let (num, den) = angle_numden(v);
                if num == 0 {
                    write!(f, "0")
                } else if den == 1 && num == 1 {
                    write!(f, "π")
                } else if num == 1 {
                    write!(f, "(π/{})", den)
                } else if den == 1 {
                    write!(f, "({}*π)", num)
                } else {
                    write!(f, "({}*π/{})", num, den)
                }
            }
            Primitive::Complex(v) => write!(f, "({}+{}im)", v.re, v.im),
            Primitive::BitReg(v) => {
                let bw = self.ty().bw().unwrap_or(BitWidth::B128).get();
                write!(f, "\"")?;
                for i in 0..bw {
                    write!(f, "{}", v & (1 << (bw - 1 - i)))?;
                }
                write!(f, "\"")?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bw;

    #[test]
    fn angle_numden_reduces_dyadic_pi_fraction() {
        assert_eq!(angle_numden(0), (0, 1));
        assert_eq!(angle_numden(1_u128 << 127), (1, 1));
        assert_eq!(angle_numden(1_u128 << 126), (1, 2));
        assert_eq!(angle_numden(3_u128 << 126), (3, 2));
        assert_eq!(angle_numden(7_u128 << 124), (7, 8));
    }

    #[test]
    fn angle_display_uses_reduced_pi_fraction() {
        let angle4 = |bits: u128| {
            Scalar::new_unchecked(Primitive::Angle(bits << 124), PrimitiveTy::Angle(bw(4)))
        };

        assert_eq!(angle4(0b0000).to_string(), "0");
        assert_eq!(angle4(0b0001).to_string(), "(π/8)");
        assert_eq!(angle4(0b0100).to_string(), "(π/2)");
        assert_eq!(angle4(0b0111).to_string(), "(7*π/8)");
        assert_eq!(angle4(0b1000).to_string(), "π");
        assert_eq!(angle4(0b1010).to_string(), "(5*π/4)");
        assert_eq!(angle4(0b1110).to_string(), "(7*π/4)");
    }
}
