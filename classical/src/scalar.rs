use std::fmt;

use num_complex::{Complex32, Complex64};

use crate::duration::{Duration, DurationUnit};
use crate::error::Result;
use crate::primitive::{BitWidth, FloatWidth, Primitive, PrimitiveTy, bw};

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

    #[inline]
    pub const fn bit(v: bool) -> Self {
        Self::new_unchecked(Primitive::bit(v), PrimitiveTy::Bit)
    }
    #[inline]
    pub const fn int(v: i128, bw: BitWidth) -> Self {
        Self::new_unchecked(Primitive::int(v), PrimitiveTy::Int(bw))
    }
    #[inline]
    pub const fn uint(v: u128, bw: BitWidth) -> Self {
        Self::new_unchecked(Primitive::uint(v), PrimitiveTy::Uint(bw))
    }
    #[inline]
    pub const fn float(v: f64, fw: FloatWidth) -> Self {
        Self::new_unchecked(Primitive::float(v), PrimitiveTy::Float(fw))
    }
    #[inline]
    pub const fn complex(re: f64, im: f64, fw: FloatWidth) -> Self {
        Self::new_unchecked(Primitive::complex(re, im), PrimitiveTy::Complex(fw))
    }
    #[inline]
    pub const fn duration(v: f64, unit: DurationUnit) -> Self {
        Self::new_unchecked(Primitive::duration(v, unit), PrimitiveTy::Duration)
    }
    #[inline]
    pub const fn bitreg(bits: u128, bw: BitWidth) -> Self {
        Self::new_unchecked(Primitive::bitreg(bits), PrimitiveTy::BitReg(bw))
    }
    #[inline]
    pub fn angle(radians: f64) -> Self {
        Self::from(Primitive::angle(radians))
    }
}

impl From<Primitive> for Scalar {
    #[inline]
    fn from(value: Primitive) -> Self {
        Self::new_unchecked(value, value.default_ty())
    }
}

macro_rules! impl_from_int {
    ($($ty:ty: $bw:literal),* $(,)?) => {
        $(
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::int(value as i128, bw($bw))
            }
        }
        )*
    };
}

macro_rules! impl_from_uint {
    ($($ty:ty: $bw:literal),* $(,)?) => {
        $(
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::uint(value as u128, bw($bw))
            }
        }
        )*
    };
}

impl_from_int!(i8: 8, i16: 16, i32: 32, i64: 64, i128: 128);
impl_from_uint!(u8: 8, u16: 16, u32: 32, u64: 64, u128: 128);

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::bit(value)
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self::float(value as f64, FloatWidth::F32)
    }
}

impl From<f64> for Scalar {
    fn from(value: f64) -> Self {
        Self::float(value, FloatWidth::F64)
    }
}

impl From<Complex64> for Scalar {
    fn from(value: Complex64) -> Self {
        Self::complex(value.re, value.im, FloatWidth::F64)
    }
}

impl From<Complex32> for Scalar {
    fn from(value: Complex32) -> Self {
        Self::complex(value.re as f64, value.im as f64, FloatWidth::F32)
    }
}

impl From<Duration> for Scalar {
    fn from(value: Duration) -> Self {
        Self::duration(value.value, value.unit)
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
