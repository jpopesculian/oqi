use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_complex::{Complex as NumComplex, ComplexFloat as NumComplexFloat};
use num_traits::{
    ConstOne, ConstZero, Float as NumFloat, FloatConst, Num, NumCast, One, ToPrimitive, Zero,
};

use crate::float::{Float, FloatSize};

#[allow(non_camel_case_types)]
type c32 = NumComplex<f32>;
#[allow(non_camel_case_types)]
type c64 = NumComplex<f64>;

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub type csize = c32;
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub type csize = c64;

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub const ZERO: Complex = Complex::C32(c32::ZERO);
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub const ZERO: Complex = Complex::C64(c64::ZERO);

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub const ONE: Complex = Complex::C32(c32::ONE);
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub const ONE: Complex = Complex::C64(c64::ONE);

#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub const I: Complex = Complex::C32(c32::I);
#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub const I: Complex = Complex::C64(c64::I);

#[derive(Debug, Clone, Copy)]
pub enum Complex {
    C32(c32),
    C64(c64),
}

impl Complex {
    pub const ZERO: Self = ZERO;
    pub const ONE: Self = ONE;
    pub const I: Self = I;

    #[inline]
    pub const fn c32(re: f32, im: f32) -> Self {
        Self::C32(c32 { re, im })
    }

    pub const fn c64(re: f64, im: f64) -> Self {
        Self::C64(c64 { re, im })
    }

    #[inline]
    pub const fn as_c32(self) -> c32 {
        match self {
            Self::C32(c) => c,
            Self::C64(c) => c32 {
                re: c.re as f32,
                im: c.im as f32,
            },
        }
    }

    #[inline]
    pub const fn as_c64(self) -> c64 {
        match self {
            Self::C32(c) => c64 {
                re: c.re as f64,
                im: c.im as f64,
            },
            Self::C64(c) => c,
        }
    }

    #[inline]
    pub const fn size(self) -> FloatSize {
        match self {
            Self::C32(_) => FloatSize::F32,
            Self::C64(_) => FloatSize::F64,
        }
    }
}

macro_rules! impl_func {
    ($func:path, $a:expr) => {
        match $a {
            Self::C32(a) => $func(a),
            Self::C64(a) => $func(a),
        }
    };
    ($func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::C32(a), Self::C32(b)) => $func(a, b),
            (Self::C32(a), Self::C64(b)) => $func(c64 { re: a.re, im: a.im }, b),
            (Self::C64(a), Self::C32(b)) => $func(a, c64 { re: b.re, im: b.im }),
            (Self::C64(a), Self::C64(b)) => $func(a, b),
        }
    };
    (& $func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::C32(a), Self::C32(b)) => $func(a, b),
            (Self::C32(a), Self::C64(b)) => $func(
                &c64 {
                    re: a.re as f64,
                    im: a.im as f64,
                },
                b,
            ),
            (Self::C64(a), Self::C32(b)) => $func(
                a,
                &c64 {
                    re: b.re as f64,
                    im: b.im as f64,
                },
            ),
            (Self::C64(a), Self::C64(b)) => $func(a, b),
        }
    };
}

macro_rules! impl_op {
    ($func:path, $a:expr) => {
        match $a {
            Self::C32(a) => Self::C32($func(a)),
            Self::C64(a) => Self::C64($func(a)),
        }
    };
    ($func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::C32(a), Self::C32(b)) => Self::C32($func(a, b)),
            (Self::C32(a), Self::C64(b)) => Self::C64($func(
                c64 {
                    re: a.re as f64,
                    im: a.im as f64,
                },
                b,
            )),
            (Self::C64(a), Self::C32(b)) => Self::C64($func(
                a,
                c64 {
                    re: b.re as f64,
                    im: b.im as f64,
                },
            )),
            (Self::C64(a), Self::C64(b)) => Self::C64($func(a, b)),
        }
    };
    (&$func:path, $a:expr, $b:expr) => {
        match ($a, $b) {
            (Self::C32(a), Self::C32(b)) => Self::C32($func(a, b)),
            (Self::C32(a), Self::C64(b)) => Self::C64($func(
                &c64 {
                    re: a.re as f64,
                    im: a.im as f64,
                },
                b,
            )),
            (Self::C64(a), Self::C32(b)) => Self::C64($func(
                a,
                &c64 {
                    re: b.re as f64,
                    im: b.im as f64,
                },
            )),
            (Self::C64(a), Self::C64(b)) => Self::C64($func(a, b)),
        }
    };
}

impl PartialEq for Complex {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        impl_func!(&PartialEq::eq, self, other)
    }
}

impl Add for Complex {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        impl_op!(Add::add, self, rhs)
    }
}

impl Mul for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        impl_op!(Mul::mul, self, rhs)
    }
}

impl Sub for Complex {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        impl_op!(Sub::sub, self, rhs)
    }
}

impl Div for Complex {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        impl_op!(Div::div, self, rhs)
    }
}

impl Rem for Complex {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        impl_op!(Rem::rem, self, rhs)
    }
}

impl Zero for Complex {
    #[inline]
    fn zero() -> Self {
        csize::zero().into()
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

impl One for Complex {
    #[inline]
    fn one() -> Self {
        csize::one().into()
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

impl Neg for Complex {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        impl_op!(Neg::neg, self)
    }
}

impl Num for Complex {
    type FromStrRadixErr = <csize as Num>::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let high_precision = c64::from_str_radix(str, radix)?;
        #[cfg(not(target_pointer_width = "64"))]
        {
            use crate::float::fsize;
            #[allow(clippy::unnecessary_cast)]
            if high_precision.re as fsize as f64 == high_precision.re
                && high_precision.im as fsize as f64 == high_precision.im
            {
                return Ok(csize {
                    re: high_precision.re as fsize,
                    im: high_precision.im as fsize,
                }
                .into());
            }
        }
        Ok(high_precision.into())
    }
}

impl ToPrimitive for Complex {
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

impl NumCast for Complex {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let high_precision = <c64 as NumCast>::from(n)?;
        #[cfg(not(target_pointer_width = "64"))]
        {
            use crate::float::fsize;
            #[allow(clippy::unnecessary_cast)]
            if high_precision.re as fsize as f64 == high_precision.re
                && high_precision.im as fsize as f64 == high_precision.im
            {
                return Some(
                    csize {
                        re: high_precision.re as fsize,
                        im: high_precision.im as fsize,
                    }
                    .into(),
                );
            }
        }
        Some(high_precision.into())
    }
}

impl Default for Complex {
    #[inline]
    fn default() -> Self {
        csize::zero().into()
    }
}

impl From<c32> for Complex {
    #[inline]
    fn from(c: c32) -> Self {
        Self::C32(c)
    }
}

impl From<c64> for Complex {
    #[inline]
    fn from(c: c64) -> Self {
        Self::C64(c)
    }
}

/// Generic trait for floating point complex numbers.
///
/// This trait defines methods which are common to complex floating point
/// numbers and regular floating point numbers.
///
/// This trait is the same as [num_complex::ComplexFloat] but unsealed
/// and implemented for all [num_complex::ComplexFloat] types
pub trait ComplexFloat: Num + NumCast + Copy + Neg<Output = Self> {
    /// The type used to represent the real coefficients of this complex number.
    type Real: NumFloat + FloatConst;

    /// Returns `true` if this value is `NaN` and false otherwise.
    fn is_nan(self) -> bool;

    /// Returns `true` if this value is positive infinity or negative infinity and
    /// false otherwise.
    fn is_infinite(self) -> bool;

    /// Returns `true` if this number is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal](http://en.wikipedia.org/wiki/Denormal_number), or `NaN`.
    fn is_normal(self) -> bool;

    /// Take the reciprocal (inverse) of a number, `1/x`. See also [Complex::finv].
    fn recip(self) -> Self;

    /// Raises `self` to a signed integer power.
    fn powi(self, exp: i32) -> Self;

    /// Raises `self` to a real power.
    fn powf(self, exp: Self::Real) -> Self;

    /// Raises `self` to a complex power.
    fn powc(self, exp: NumComplex<Self::Real>) -> NumComplex<Self::Real>;

    /// Take the square root of a number.
    fn sqrt(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> Self;

    /// Returns `2^(self)`.
    fn exp2(self) -> Self;

    /// Returns `base^(self)`.
    fn expf(self, base: Self::Real) -> Self;

    /// Returns the natural logarithm of the number.
    fn ln(self) -> Self;

    /// Returns the logarithm of the number with respect to an arbitrary base.
    fn log(self, base: Self::Real) -> Self;

    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> Self;

    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> Self;

    /// Take the cubic root of a number.
    fn cbrt(self) -> Self;

    /// Computes the sine of a number (in radians).
    fn sin(self) -> Self;

    /// Computes the cosine of a number (in radians).
    fn cos(self) -> Self;

    /// Computes the tangent of a number (in radians).
    fn tan(self) -> Self;

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    fn asin(self) -> Self;

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    fn acos(self) -> Self;

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    fn atan(self) -> Self;

    /// Hyperbolic sine function.
    fn sinh(self) -> Self;

    /// Hyperbolic cosine function.
    fn cosh(self) -> Self;

    /// Hyperbolic tangent function.
    fn tanh(self) -> Self;

    /// Inverse hyperbolic sine function.
    fn asinh(self) -> Self;

    /// Inverse hyperbolic cosine function.
    fn acosh(self) -> Self;

    /// Inverse hyperbolic tangent function.
    fn atanh(self) -> Self;

    /// Returns the real part of the number.
    fn re(self) -> Self::Real;

    /// Returns the imaginary part of the number.
    fn im(self) -> Self::Real;

    /// Returns the absolute value of the number. See also [Complex::norm]
    fn abs(self) -> Self::Real;

    /// Returns the L1 norm `|re| + |im|` -- the [Manhattan distance] from the origin.
    ///
    /// [Manhattan distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
    fn l1_norm(&self) -> Self::Real;

    /// Computes the argument of the number.
    fn arg(self) -> Self::Real;

    /// Computes the complex conjugate of the number.
    ///
    /// Formula: `a+bi -> a-bi`
    fn conj(self) -> Self;
}

macro_rules! forward {
    ($( $base:ident :: $method:ident ( self $( , $arg:ident : $ty:ty )* ) -> $ret:ty ; )*)
        => {$(
            #[inline]
            fn $method(self $( , $arg : $ty )* ) -> $ret {
                $base::$method(self $( , $arg )* )
            }
        )*};
}

impl<T> ComplexFloat for T
where
    T: NumComplexFloat,
{
    type Real = T::Real;

    forward! {
        T::is_nan(self) -> bool;
        T::is_infinite(self) -> bool;
        T::is_finite(self) -> bool;
        T::is_normal(self) -> bool;
        T::recip(self) -> Self;
        T::powi(self, exp: i32) -> Self;
        T::powf(self, exp: Self::Real) -> Self;
        T::powc(self, exp: NumComplex<Self::Real>) -> NumComplex<Self::Real>;
        T::sqrt(self) -> Self;
        T::exp(self) -> Self;
        T::exp2(self) -> Self;
        T::expf(self, base: Self::Real) -> Self;
        T::ln(self) -> Self;
        T::log(self, base: Self::Real) -> Self;
        T::log2(self) -> Self;
        T::log10(self) -> Self;
        T::cbrt(self) -> Self;
        T::sin(self) -> Self;
        T::cos(self) -> Self;
        T::tan(self) -> Self;
        T::asin(self) -> Self;
        T::acos(self) -> Self;
        T::atan(self) -> Self;
        T::sinh(self) -> Self;
        T::cosh(self) -> Self;
        T::tanh(self) -> Self;
        T::asinh(self) -> Self;
        T::acosh(self) -> Self;
        T::atanh(self) -> Self;
        T::re(self) -> Self::Real;
        T::im(self) -> Self::Real;
        T::abs(self) -> Self::Real;
        T::arg(self) -> Self::Real;
        T::conj(self) -> Self;
    }

    #[inline]
    fn l1_norm(&self) -> Self::Real {
        T::l1_norm(self)
    }
}

impl ComplexFloat for Complex {
    type Real = Float;

    #[inline]
    fn is_nan(self) -> bool {
        impl_func!(NumComplexFloat::is_nan, self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        impl_func!(NumComplexFloat::is_infinite, self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        impl_func!(NumComplexFloat::is_finite, self)
    }
    #[inline]
    fn is_normal(self) -> bool {
        impl_func!(NumComplexFloat::is_normal, self)
    }
    #[inline]
    fn recip(self) -> Self {
        impl_op!(NumComplexFloat::recip, self)
    }
    #[inline]
    fn powi(self, exp: i32) -> Self {
        match self {
            Self::C32(a) => Self::C32(NumComplexFloat::powi(a, exp)),
            Self::C64(a) => Self::C64(NumComplexFloat::powi(a, exp)),
        }
    }
    #[inline]
    fn powf(self, exp: Float) -> Self {
        match (self, exp) {
            (Self::C32(a), Float::F32(b)) => Self::C32(NumComplexFloat::powf(a, b)),
            (Self::C32(a), Float::F64(b)) => Self::C64(NumComplexFloat::powf(
                c64 {
                    re: a.re as f64,
                    im: a.im as f64,
                },
                b,
            )),
            (Self::C64(a), Float::F32(b)) => Self::C64(NumComplexFloat::powf(a, b as f64)),
            (Self::C64(a), Float::F64(b)) => Self::C64(NumComplexFloat::powf(a, b)),
        }
    }
    #[inline]
    fn powc(self, exp: NumComplex<Float>) -> NumComplex<Float> {
        match (self, exp.re, exp.im) {
            (Self::C32(a), Float::F32(re), Float::F32(im)) => {
                let result = NumComplexFloat::powc(a, NumComplex { re, im });
                NumComplex {
                    re: Float::F32(result.re),
                    im: Float::F32(result.im),
                }
            }
            (s, re, im) => {
                let result = NumComplexFloat::powc(
                    s.as_c64(),
                    c64 {
                        re: re.as_f64(),
                        im: im.as_f64(),
                    },
                );
                NumComplex {
                    re: Float::F64(result.re),
                    im: Float::F64(result.im),
                }
            }
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        impl_op!(NumComplexFloat::sqrt, self)
    }
    #[inline]
    fn exp(self) -> Self {
        impl_op!(NumComplexFloat::exp, self)
    }
    #[inline]
    fn exp2(self) -> Self {
        impl_op!(NumComplexFloat::exp2, self)
    }
    #[inline]
    fn expf(self, base: Float) -> Self {
        match (self, base) {
            (Self::C32(a), Float::F32(b)) => Self::C32(NumComplexFloat::expf(a, b)),
            (Self::C32(a), Float::F64(b)) => Self::C64(NumComplexFloat::expf(
                c64 {
                    re: a.re as f64,
                    im: a.im as f64,
                },
                b,
            )),
            (Self::C64(a), Float::F32(b)) => Self::C64(NumComplexFloat::expf(a, b as f64)),
            (Self::C64(a), Float::F64(b)) => Self::C64(NumComplexFloat::expf(a, b)),
        }
    }
    #[inline]
    fn ln(self) -> Self {
        impl_op!(NumComplexFloat::ln, self)
    }
    #[inline]
    fn log(self, base: Float) -> Self {
        match (self, base) {
            (Self::C32(a), Float::F32(b)) => Self::C32(NumComplexFloat::log(a, b)),
            (Self::C32(a), Float::F64(b)) => Self::C64(NumComplexFloat::log(
                c64 {
                    re: a.re as f64,
                    im: a.im as f64,
                },
                b,
            )),
            (Self::C64(a), Float::F32(b)) => Self::C64(NumComplexFloat::log(a, b as f64)),
            (Self::C64(a), Float::F64(b)) => Self::C64(NumComplexFloat::log(a, b)),
        }
    }
    #[inline]
    fn log2(self) -> Self {
        impl_op!(NumComplexFloat::log2, self)
    }
    #[inline]
    fn log10(self) -> Self {
        impl_op!(NumComplexFloat::log10, self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        impl_op!(NumComplexFloat::cbrt, self)
    }
    #[inline]
    fn sin(self) -> Self {
        impl_op!(NumComplexFloat::sin, self)
    }
    #[inline]
    fn cos(self) -> Self {
        impl_op!(NumComplexFloat::cos, self)
    }
    #[inline]
    fn tan(self) -> Self {
        impl_op!(NumComplexFloat::tan, self)
    }
    #[inline]
    fn asin(self) -> Self {
        impl_op!(NumComplexFloat::asin, self)
    }
    #[inline]
    fn acos(self) -> Self {
        impl_op!(NumComplexFloat::acos, self)
    }
    #[inline]
    fn atan(self) -> Self {
        impl_op!(NumComplexFloat::atan, self)
    }
    #[inline]
    fn sinh(self) -> Self {
        impl_op!(NumComplexFloat::sinh, self)
    }
    #[inline]
    fn cosh(self) -> Self {
        impl_op!(NumComplexFloat::cosh, self)
    }
    #[inline]
    fn tanh(self) -> Self {
        impl_op!(NumComplexFloat::tanh, self)
    }
    #[inline]
    fn asinh(self) -> Self {
        impl_op!(NumComplexFloat::asinh, self)
    }
    #[inline]
    fn acosh(self) -> Self {
        impl_op!(NumComplexFloat::acosh, self)
    }
    #[inline]
    fn atanh(self) -> Self {
        impl_op!(NumComplexFloat::atanh, self)
    }
    #[inline]
    fn re(self) -> Float {
        match self {
            Self::C32(a) => Float::F32(NumComplexFloat::re(a)),
            Self::C64(a) => Float::F64(NumComplexFloat::re(a)),
        }
    }
    #[inline]
    fn im(self) -> Float {
        match self {
            Self::C32(a) => Float::F32(NumComplexFloat::im(a)),
            Self::C64(a) => Float::F64(NumComplexFloat::im(a)),
        }
    }
    #[inline]
    fn abs(self) -> Float {
        match self {
            Self::C32(a) => Float::F32(NumComplexFloat::abs(a)),
            Self::C64(a) => Float::F64(NumComplexFloat::abs(a)),
        }
    }
    #[inline]
    fn l1_norm(&self) -> Float {
        match self {
            Self::C32(a) => Float::F32(NumComplexFloat::l1_norm(a)),
            Self::C64(a) => Float::F64(NumComplexFloat::l1_norm(a)),
        }
    }
    #[inline]
    fn arg(self) -> Float {
        match self {
            Self::C32(a) => Float::F32(NumComplexFloat::arg(a)),
            Self::C64(a) => Float::F64(NumComplexFloat::arg(a)),
        }
    }
    #[inline]
    fn conj(self) -> Self {
        impl_op!(NumComplexFloat::conj, self)
    }
}

impl ConstOne for Complex {
    const ONE: Self = ONE;
}

impl ConstZero for Complex {
    const ZERO: Self = ZERO;
}
