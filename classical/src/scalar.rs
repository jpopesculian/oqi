use std::fmt;

use num_complex::{Complex32, Complex64};

use crate::bitreg::BitReg;
use crate::duration::{Duration, DurationUnit};
use crate::error::Result;
use crate::primitive::{
    FloatWidth, IntWidth, Primitive, PrimitiveTy, angle_bits, angle_from_bits, iw, resize_int,
    resize_uint,
};

/// `v` as a `float[fw]` holds it: rounded to single precision for 32 bits.
const fn round_to(v: f64, fw: FloatWidth) -> f64 {
    match fw {
        FloatWidth::F32 => v as f32 as f64,
        _ => v,
    }
}

/// A classical value of a scalar type: a [`Primitive`] and its
/// [`PrimitiveTy`], kept consistent (an `int[8]` holds a value that fits
/// 8 bits). Equality compares both; floats compare as `f64` does.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Scalar {
    value: Primitive,
    ty: PrimitiveTy,
}

impl Scalar {
    /// A scalar of `value` without checking it against `ty`.
    ///
    /// The caller guarantees `value` already has `ty`'s representation (as
    /// [`Scalar::new`] would convert it: an `int[8]` within 8 bits, a
    /// `float[32]` rounded to single precision, …); otherwise arithmetic may
    /// give wrong results.
    #[doc(hidden)]
    #[inline]
    pub const fn new_unchecked(value: Primitive, ty: PrimitiveTy) -> Self {
        Scalar { value, ty }
    }

    /// The value.
    #[inline]
    pub const fn value(&self) -> &Primitive {
        &self.value
    }

    /// The value, without its type.
    #[inline]
    pub fn into_value(self) -> Primitive {
        self.value
    }

    /// The type.
    #[inline]
    pub const fn ty(&self) -> PrimitiveTy {
        self.ty
    }

    /// `value` represented as `ty` (wrapped or rounded to its width).
    ///
    /// # Errors
    ///
    /// [`Error::TypeMismatch`](crate::Error::TypeMismatch) if `value` has no representation in
    /// `ty`.
    pub fn new(value: Primitive, ty: PrimitiveTy) -> Result<Self> {
        Ok(Scalar::new_unchecked(value.as_ty(ty)?, ty))
    }

    /// Cast to `to` under the spec's table of allowed casts.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedCast`](crate::Error::UnsupportedCast) if the cast is not
    /// allowed.
    #[inline]
    pub fn cast(self, to: PrimitiveTy) -> Result<Self> {
        self.ty.cast(to)?;
        // An angle and a register of its width are cast by "a direct copy
        // of the bit pattern" (docs/types.rst), which takes the angle's
        // width to read.
        let value = match (self.value, self.ty, to) {
            (Primitive::Angle(a), PrimitiveTy::Angle(bw), PrimitiveTy::BitReg(_)) => {
                Primitive::bitreg_u128(angle_bits(a, bw.get()))
            }
            (Primitive::BitReg(reg), _, PrimitiveTy::Angle(bw)) => {
                Primitive::Angle(angle_from_bits(reg.as_u128(), bw.get()))
            }
            (value, ..) => value,
        };
        Self::new(value, to)
    }

    /// A `bit`.
    #[inline]
    pub const fn bit(v: bool) -> Self {
        Self::new_unchecked(Primitive::bit(v), PrimitiveTy::Bit)
    }
    /// An `int[bw]`, wrapping `v` to `bw` bits.
    #[inline]
    pub const fn int(v: i128, bw: IntWidth) -> Self {
        Self::new_unchecked(Primitive::int(resize_int(v, bw)), PrimitiveTy::Int(bw))
    }
    /// A `uint[bw]`, wrapping `v` to `bw` bits.
    #[inline]
    pub const fn uint(v: u128, bw: IntWidth) -> Self {
        Self::new_unchecked(Primitive::uint(resize_uint(v, bw)), PrimitiveTy::Uint(bw))
    }
    /// A `float[fw]`, rounding `v` to single precision for `float[32]`.
    #[inline]
    pub const fn float(v: f64, fw: FloatWidth) -> Self {
        Self::new_unchecked(Primitive::float(round_to(v, fw)), PrimitiveTy::Float(fw))
    }
    /// A `complex[float[fw]]`, rounding both parts like [`Scalar::float`].
    #[inline]
    pub const fn complex(re: f64, im: f64, fw: FloatWidth) -> Self {
        Self::new_unchecked(
            Primitive::complex(round_to(re, fw), round_to(im, fw)),
            PrimitiveTy::Complex(fw),
        )
    }
    /// A `duration` of `v` in `unit`.
    #[inline]
    pub const fn duration(v: f64, unit: DurationUnit) -> Self {
        Self::new_unchecked(Primitive::duration(v, unit), PrimitiveTy::Duration)
    }
    /// A `bit[width]` holding the low `width` bits of `reg`.
    #[inline]
    pub fn bitreg(reg: BitReg, width: u32) -> Self {
        Self::new_unchecked(
            Primitive::bitreg(reg.resize(width)),
            PrimitiveTy::BitReg(width),
        )
    }
    /// A `bit[width]` holding the low `width` bits of `bits`.
    #[inline]
    pub fn bitreg_u128(bits: u128, width: u32) -> Self {
        Self::bitreg(BitReg::from(bits), width)
    }
    /// An `angle[128]` of `radians`, reduced modulo 2π.
    #[inline]
    pub fn angle(radians: f64) -> Self {
        Self::from(Primitive::angle(radians))
    }
}

impl From<Primitive> for Scalar {
    #[inline]
    fn from(value: Primitive) -> Self {
        let ty = value.default_ty();
        Self::new_unchecked(value, ty)
    }
}

macro_rules! impl_from_int {
    ($($ty:ty: $bw:literal),* $(,)?) => {
        $(
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::int(value as i128, iw($bw))
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
                Self::uint(value as u128, iw($bw))
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

macro_rules! impl_from_angle {
    ($($ty:ty: $bw:literal),* $(,)?) => {
        $(
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::new_unchecked(Primitive::from(value), PrimitiveTy::Angle(iw($bw)))
            }
        }
        )*
    };
}

impl_from_angle!(
    turns::Angle8: 8,
    turns::Angle16: 16,
    turns::Angle32: 32,
    turns::Angle64: 64,
    turns::Angle128: 128,
);

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.value(), self.ty()) {
            (Primitive::Bit(v), PrimitiveTy::Bool) => write!(f, "{v}"),
            (Primitive::BitReg(reg), ty) => {
                let bw = ty.bit_count().unwrap_or(reg.default_width());
                write!(f, "\"")?;
                reg.fmt_bits(f, bw)?;
                write!(f, "\"")
            }
            (value, _) => write!(f, "{value}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iw;

    /// A typed constructor holds the value its type can: wider integers
    /// wrap, wider registers lose their high bits, `float[32]` rounds.
    #[test]
    fn typed_constructors_normalize_to_their_width() {
        assert_eq!(Scalar::int(200, iw(8)).to_string(), "-56");
        assert_eq!(Scalar::int(-1, iw(8)).to_string(), "-1");
        assert_eq!(Scalar::uint(300, iw(8)).to_string(), "44");
        assert_eq!(Scalar::bitreg_u128(0b1111, 2).to_string(), "\"11\"");
        assert_eq!(
            Scalar::bitreg(BitReg::from(0b1111u128), 2).to_string(),
            "\"11\""
        );
        let third = Scalar::float(1.0 / 3.0, FloatWidth::F32);
        assert_eq!(third.to_string(), f64::from(1.0f32 / 3.0).to_string());
        let c = Scalar::complex(1.0 / 3.0, 0.0, FloatWidth::F32);
        let Primitive::Complex(c) = c.value() else {
            panic!("complex")
        };
        assert_eq!(c.re, f64::from(1.0f32 / 3.0));
    }

    #[test]
    fn angle_and_register_casts_copy_the_bit_pattern() {
        // 9π/8 is 9/16 of a turn: "1001" as an `angle[4]`.
        let angle = Scalar::new_unchecked(
            Primitive::Angle(turns::Angle(0b1001 << 124)),
            PrimitiveTy::Angle(iw(4)),
        );
        let bits = angle.clone().cast(PrimitiveTy::BitReg(4)).unwrap();
        assert_eq!(bits.to_string(), "\"1001\"");
        let back = bits.cast(PrimitiveTy::Angle(iw(4))).unwrap();
        assert_eq!(back.to_string(), angle.to_string());
    }

    #[test]
    fn angle_display_uses_reduced_pi_fraction() {
        let angle4 = |bits: u128| {
            Scalar::new_unchecked(
                Primitive::Angle(turns::Angle(bits << 124)),
                PrimitiveTy::Angle(iw(4)),
            )
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
