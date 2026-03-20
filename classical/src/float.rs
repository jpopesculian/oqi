use core::cmp::{PartialEq, PartialOrd};
use core::num::NonZero;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{
    ConstOne, ConstZero, Float as NumFloat, FloatConst, Num, NumCast, One, Signed, ToPrimitive,
    Zero,
};

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub type fsize = f32;
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub type fsize = f64;

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub const ZERO: Float = Float::F32(f32::ZERO);
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub const ZERO: Float = Float::F64(f64::ZERO);

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub const ONE: Float = Float::F32(f32::ONE);
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub const ONE: Float = Float::F64(f64::ONE);

#[derive(Debug, Clone, Copy)]
pub enum Float {
    F32(f32),
    F64(f64),
}

impl Float {
    pub const ZERO: Self = ZERO;
    pub const ONE: Self = ONE;

    #[inline]
    pub const fn f32(f: f32) -> Self {
        Self::F32(f)
    }

    #[inline]
    pub const fn f64(f: f64) -> Self {
        Self::F64(f)
    }

    #[inline]
    pub const fn as_f32(self) -> f32 {
        match self {
            Self::F32(f) => f,
            Self::F64(f) => f as f32,
        }
    }

    #[inline]
    pub const fn as_f64(self) -> f64 {
        match self {
            Self::F32(f) => f as f64,
            Self::F64(f) => f,
        }
    }

    #[inline]
    pub const fn size(self) -> FloatSize {
        match self {
            Self::F32(_) => FloatSize::F32,
            Self::F64(_) => FloatSize::F64,
        }
    }
}

macro_rules! impl_func {
    ($func:path, $a:expr) => {
        match $a {
            Self::F32(a) => $func(a),
            Self::F64(a) => $func(a),
        }
    };
    ($func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::F32(a), Self::F32(b)) => $func(a, b),
            (Self::F32(a), Self::F64(b)) => $func(a as f64, b),
            (Self::F64(a), Self::F32(b)) => $func(a, b as f64),
            (Self::F64(a), Self::F64(b)) => $func(a, b),
        }
    };
    (& $func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::F32(a), Self::F32(b)) => $func(a, b),
            (Self::F32(a), Self::F64(b)) => $func(&(*a as f64), b),
            (Self::F64(a), Self::F32(b)) => $func(a, &(*b as f64)),
            (Self::F64(a), Self::F64(b)) => $func(a, b),
        }
    };
}

macro_rules! impl_op {
    ($func:path, $a:expr) => {
        match $a {
            Self::F32(a) => Self::F32($func(a)),
            Self::F64(a) => Self::F64($func(a)),
        }
    };
    ($func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::F32(a), Self::F32(b)) => Self::F32($func(a, b)),
            (Self::F32(a), Self::F64(b)) => Self::F64($func(a as f64, b)),
            (Self::F64(a), Self::F32(b)) => Self::F64($func(a, b as f64)),
            (Self::F64(a), Self::F64(b)) => Self::F64($func(a, b)),
        }
    };
    (&$func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::F32(a), Self::F32(b)) => Self::F32($func(a, b)),
            (Self::F32(a), Self::F64(b)) => Self::F64($func(&(*a as f64), b)),
            (Self::F64(a), Self::F32(b)) => Self::F64($func(a, &(*b as f64))),
            (Self::F64(a), Self::F64(b)) => Self::F64($func(a, b)),
        }
    };
}

impl PartialEq for Float {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        impl_func!(&PartialEq::eq, self, other)
    }
}

impl PartialOrd for Float {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        impl_func!(&PartialOrd::partial_cmp, self, other)
    }
}

impl Add for Float {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        impl_op!(Add::add, self, rhs)
    }
}

impl Mul for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        impl_op!(Mul::mul, self, rhs)
    }
}

impl Sub for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        impl_op!(Sub::sub, self, rhs)
    }
}

impl Div for Float {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        impl_op!(Div::div, self, rhs)
    }
}

impl Rem for Float {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        impl_op!(Rem::rem, self, rhs)
    }
}

impl Zero for Float {
    #[inline]
    fn zero() -> Self {
        fsize::zero().into()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        impl_func!(Zero::is_zero, self)
    }

    #[inline]
    fn set_zero(&mut self) {
        impl_func!(Zero::set_zero, self)
    }
}

impl One for Float {
    #[inline]
    fn one() -> Self {
        fsize::one().into()
    }

    #[inline]
    fn is_one(&self) -> bool {
        impl_func!(One::is_one, self)
    }

    #[inline]
    fn set_one(&mut self) {
        impl_func!(One::set_one, self)
    }
}

impl Neg for Float {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        impl_op!(Neg::neg, self)
    }
}

impl Num for Float {
    type FromStrRadixErr = <fsize as Num>::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let high_precision = f64::from_str_radix(str, radix)?;
        #[cfg(not(target_pointer_width = "64"))]
        {
            #[allow(clippy::unnecessary_cast)]
            if high_precision as fsize as f64 == high_precision {
                return Ok((high_precision as fsize).into());
            }
        }
        Ok(high_precision.into())
    }
}

impl ToPrimitive for Float {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        impl_func!(ToPrimitive::to_i64, self)
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        impl_func!(ToPrimitive::to_u64, self)
    }
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        impl_func!(ToPrimitive::to_isize, self)
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        impl_func!(ToPrimitive::to_i8, self)
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        impl_func!(ToPrimitive::to_i16, self)
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        impl_func!(ToPrimitive::to_i32, self)
    }
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        impl_func!(ToPrimitive::to_i128, self)
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        impl_func!(ToPrimitive::to_usize, self)
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        impl_func!(ToPrimitive::to_u8, self)
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        impl_func!(ToPrimitive::to_u16, self)
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        impl_func!(ToPrimitive::to_u32, self)
    }
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        impl_func!(ToPrimitive::to_u128, self)
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        impl_func!(ToPrimitive::to_f32, self)
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        impl_func!(ToPrimitive::to_f64, self)
    }
}

impl NumCast for Float {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let high_precision = <f64 as NumCast>::from(n)?;
        #[cfg(not(target_pointer_width = "64"))]
        {
            #[allow(clippy::unnecessary_cast)]
            if high_precision as fsize as f64 == high_precision {
                return Some((high_precision as fsize).into());
            }
        }
        Some(high_precision.into())
    }
}

impl Signed for Float {
    #[inline]
    fn abs(&self) -> Self {
        impl_op!(Signed::abs, self)
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        impl_op!(&Signed::abs_sub, self, other)
    }
    #[inline]
    fn signum(&self) -> Self {
        impl_op!(Signed::signum, self)
    }
    #[inline]
    fn is_positive(&self) -> bool {
        impl_func!(Signed::is_positive, self)
    }
    #[inline]
    fn is_negative(&self) -> bool {
        impl_func!(Signed::is_negative, self)
    }
}

impl Default for Float {
    #[inline]
    fn default() -> Self {
        fsize::zero().into()
    }
}

impl From<f32> for Float {
    #[inline]
    fn from(f: f32) -> Self {
        Self::F32(f)
    }
}

impl From<f64> for Float {
    #[inline]
    fn from(f: f64) -> Self {
        Self::F64(f)
    }
}

impl NumFloat for Float {
    #[inline]
    fn nan() -> Self {
        fsize::nan().into()
    }
    #[inline]
    fn infinity() -> Self {
        fsize::infinity().into()
    }
    #[inline]
    fn neg_infinity() -> Self {
        fsize::neg_infinity().into()
    }
    #[inline]
    fn neg_zero() -> Self {
        fsize::neg_zero().into()
    }
    #[inline]
    fn min_value() -> Self {
        fsize::min_value().into()
    }
    #[inline]
    fn min_positive_value() -> Self {
        fsize::min_positive_value().into()
    }
    #[inline]
    fn max_value() -> Self {
        fsize::max_value().into()
    }

    #[inline]
    fn is_nan(self) -> bool {
        impl_func!(NumFloat::is_nan, self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        impl_func!(NumFloat::is_infinite, self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        impl_func!(NumFloat::is_finite, self)
    }
    #[inline]
    fn is_normal(self) -> bool {
        impl_func!(NumFloat::is_normal, self)
    }
    #[inline]
    fn classify(self) -> core::num::FpCategory {
        impl_func!(NumFloat::classify, self)
    }

    #[inline]
    fn floor(self) -> Self {
        impl_op!(NumFloat::floor, self)
    }
    #[inline]
    fn ceil(self) -> Self {
        impl_op!(NumFloat::ceil, self)
    }
    #[inline]
    fn round(self) -> Self {
        impl_op!(NumFloat::round, self)
    }
    #[inline]
    fn trunc(self) -> Self {
        impl_op!(NumFloat::trunc, self)
    }
    #[inline]
    fn fract(self) -> Self {
        impl_op!(NumFloat::fract, self)
    }
    #[inline]
    fn abs(self) -> Self {
        impl_op!(NumFloat::abs, self)
    }
    #[inline]
    fn signum(self) -> Self {
        impl_op!(NumFloat::signum, self)
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        impl_func!(NumFloat::is_sign_positive, self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        impl_func!(NumFloat::is_sign_negative, self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        match (self, a, b) {
            (Self::F32(x), Self::F32(a), Self::F32(b)) => Self::F32(x.mul_add(a, b)),
            _ => Self::F64(self.as_f64().mul_add(a.as_f64(), b.as_f64())),
        }
    }
    #[inline]
    fn recip(self) -> Self {
        impl_op!(NumFloat::recip, self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        match self {
            Self::F32(a) => Self::F32(NumFloat::powi(a, n)),
            Self::F64(a) => Self::F64(NumFloat::powi(a, n)),
        }
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        impl_op!(NumFloat::powf, self, n)
    }
    #[inline]
    fn sqrt(self) -> Self {
        impl_op!(NumFloat::sqrt, self)
    }
    #[inline]
    fn exp(self) -> Self {
        impl_op!(NumFloat::exp, self)
    }
    #[inline]
    fn exp2(self) -> Self {
        impl_op!(NumFloat::exp2, self)
    }
    #[inline]
    fn ln(self) -> Self {
        impl_op!(NumFloat::ln, self)
    }
    #[inline]
    fn log(self, base: Self) -> Self {
        impl_op!(NumFloat::log, self, base)
    }
    #[inline]
    fn log2(self) -> Self {
        impl_op!(NumFloat::log2, self)
    }
    #[inline]
    fn log10(self) -> Self {
        impl_op!(NumFloat::log10, self)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        impl_op!(NumFloat::max, self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        impl_op!(NumFloat::min, self, other)
    }
    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        impl_op!(NumFloat::abs_sub, self, other)
    }
    #[inline]
    fn cbrt(self) -> Self {
        impl_op!(NumFloat::cbrt, self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        impl_op!(NumFloat::hypot, self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        impl_op!(NumFloat::sin, self)
    }
    #[inline]
    fn cos(self) -> Self {
        impl_op!(NumFloat::cos, self)
    }
    #[inline]
    fn tan(self) -> Self {
        impl_op!(NumFloat::tan, self)
    }
    #[inline]
    fn asin(self) -> Self {
        impl_op!(NumFloat::asin, self)
    }
    #[inline]
    fn acos(self) -> Self {
        impl_op!(NumFloat::acos, self)
    }
    #[inline]
    fn atan(self) -> Self {
        impl_op!(NumFloat::atan, self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        impl_op!(NumFloat::atan2, self, other)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        match self {
            Self::F32(a) => {
                let (s, c) = NumFloat::sin_cos(a);
                (Self::F32(s), Self::F32(c))
            }
            Self::F64(a) => {
                let (s, c) = NumFloat::sin_cos(a);
                (Self::F64(s), Self::F64(c))
            }
        }
    }
    #[inline]
    fn exp_m1(self) -> Self {
        impl_op!(NumFloat::exp_m1, self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        impl_op!(NumFloat::ln_1p, self)
    }
    #[inline]
    fn sinh(self) -> Self {
        impl_op!(NumFloat::sinh, self)
    }
    #[inline]
    fn cosh(self) -> Self {
        impl_op!(NumFloat::cosh, self)
    }
    #[inline]
    fn tanh(self) -> Self {
        impl_op!(NumFloat::tanh, self)
    }
    #[inline]
    fn asinh(self) -> Self {
        impl_op!(NumFloat::asinh, self)
    }
    #[inline]
    fn acosh(self) -> Self {
        impl_op!(NumFloat::acosh, self)
    }
    #[inline]
    fn atanh(self) -> Self {
        impl_op!(NumFloat::atanh, self)
    }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        impl_func!(NumFloat::integer_decode, self)
    }
}

impl num_traits::float::FloatCore for Float {
    #[inline]
    fn infinity() -> Self {
        NumFloat::infinity()
    }
    #[inline]
    fn neg_infinity() -> Self {
        NumFloat::neg_infinity()
    }
    #[inline]
    fn nan() -> Self {
        NumFloat::nan()
    }
    #[inline]
    fn neg_zero() -> Self {
        NumFloat::neg_zero()
    }
    #[inline]
    fn min_value() -> Self {
        NumFloat::min_value()
    }
    #[inline]
    fn min_positive_value() -> Self {
        NumFloat::min_positive_value()
    }
    #[inline]
    fn epsilon() -> Self {
        NumFloat::epsilon()
    }
    #[inline]
    fn max_value() -> Self {
        NumFloat::max_value()
    }
    #[inline]
    fn classify(self) -> core::num::FpCategory {
        NumFloat::classify(self)
    }
    #[inline]
    fn to_degrees(self) -> Self {
        impl_op!(NumFloat::to_degrees, self)
    }
    #[inline]
    fn to_radians(self) -> Self {
        impl_op!(NumFloat::to_radians, self)
    }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        NumFloat::integer_decode(self)
    }
}

#[allow(non_snake_case)]
impl FloatConst for Float {
    #[inline]
    fn E() -> Self {
        fsize::E().into()
    }
    #[inline]
    fn FRAC_1_PI() -> Self {
        fsize::FRAC_1_PI().into()
    }
    #[inline]
    fn FRAC_1_SQRT_2() -> Self {
        fsize::FRAC_1_SQRT_2().into()
    }
    #[inline]
    fn FRAC_2_PI() -> Self {
        fsize::FRAC_2_PI().into()
    }
    #[inline]
    fn FRAC_2_SQRT_PI() -> Self {
        fsize::FRAC_2_SQRT_PI().into()
    }
    #[inline]
    fn FRAC_PI_2() -> Self {
        fsize::FRAC_PI_2().into()
    }
    #[inline]
    fn FRAC_PI_3() -> Self {
        fsize::FRAC_PI_3().into()
    }
    #[inline]
    fn FRAC_PI_4() -> Self {
        fsize::FRAC_PI_4().into()
    }
    #[inline]
    fn FRAC_PI_6() -> Self {
        fsize::FRAC_PI_6().into()
    }
    #[inline]
    fn FRAC_PI_8() -> Self {
        fsize::FRAC_PI_8().into()
    }
    #[inline]
    fn LN_10() -> Self {
        fsize::LN_10().into()
    }
    #[inline]
    fn LN_2() -> Self {
        fsize::LN_2().into()
    }
    #[inline]
    fn LOG10_E() -> Self {
        fsize::LOG10_E().into()
    }
    #[inline]
    fn LOG2_E() -> Self {
        fsize::LOG2_E().into()
    }
    #[inline]
    fn PI() -> Self {
        fsize::PI().into()
    }
    #[inline]
    fn SQRT_2() -> Self {
        fsize::SQRT_2().into()
    }
}

impl ConstZero for Float {
    const ZERO: Self = ZERO;
}

impl ConstOne for Float {
    const ONE: Self = ONE;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSize {
    F32,
    F64,
}

impl FloatSize {
    #[inline]
    pub const fn bits(self) -> NonZero<usize> {
        match self {
            FloatSize::F32 => unsafe { NonZero::new_unchecked(32) },
            FloatSize::F64 => unsafe { NonZero::new_unchecked(64) },
        }
    }
}
