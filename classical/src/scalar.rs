use core::fmt;
use core::num::NonZero;
use std::f64::consts::TAU;

use awint::{Awi, bw};
use num_traits::Zero;

use crate::complex::{Complex, ComplexFloat};
use crate::float::{Float, FloatSize, fsize};

pub use core::num::{ParseFloatError, ParseIntError};

#[derive(Debug, Clone)]
pub enum Scalar {
    Bool(bool),
    Int(Awi),
    Uint(Awi),
    Float(Float),
    Complex(Complex),
    Angle(Awi),
    Timing(TimingValue),
    Bits(Awi),
}

#[derive(Debug, Clone, Copy)]
pub enum ScalarTy {
    Bool,
    Int(NonZero<usize>),
    Uint(NonZero<usize>),
    Float(FloatSize),
    Complex(FloatSize),
    Angle(NonZero<usize>),
    Timing,
    Bits(NonZero<usize>),
}

impl ScalarTy {
    pub fn bw(self) -> NonZero<usize> {
        match self {
            ScalarTy::Int(n) | ScalarTy::Uint(n) | ScalarTy::Angle(n) | ScalarTy::Bits(n) => n,
            ScalarTy::Bool => bw(1),
            ScalarTy::Float(fs) => fs.bits(),
            ScalarTy::Complex(fs) => fs.bits().saturating_mul(bw(2)),
            ScalarTy::Timing => bw(usize::BITS as usize),
        }
    }
}

impl ScalarTy {
    pub const fn int(n: usize) -> Self {
        Self::Int(bw(n))
    }

    pub const fn uint(n: usize) -> Self {
        Self::Uint(bw(n))
    }

    pub const fn f32() -> Self {
        Self::Float(FloatSize::F32)
    }

    pub const fn f64() -> Self {
        Self::Float(FloatSize::F64)
    }

    pub const fn c32() -> Self {
        Self::Complex(FloatSize::F32)
    }

    pub const fn c64() -> Self {
        Self::Complex(FloatSize::F64)
    }

    pub const fn angle(n: usize) -> Self {
        Self::Angle(bw(n))
    }

    pub const fn timing() -> Self {
        Self::Timing
    }

    pub const fn bits(n: usize) -> Self {
        Self::Bits(bw(n))
    }

    pub const fn bit() -> Self {
        Self::Bool
    }
}

#[derive(Debug, Clone)]
pub struct TimingValue {
    pub value: TimingNumber,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone)]
pub enum TimingNumber {
    Integer(isize),
    Float(fsize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Dt,
    Ns,
    Us,
    Ms,
    S,
}

#[derive(Debug, Clone)]
pub enum ParseBitstringError {
    InvalidChar(char, usize),
}

impl fmt::Display for ParseBitstringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseBitstringError::InvalidChar(c, index) => {
                write!(f, "Invalid character '{}' at position {}", c, index)
            }
        }
    }
}

impl std::error::Error for ParseBitstringError {}

#[derive(Debug, Clone)]
pub struct CastError {
    pub scalar: Scalar,
    pub target: ScalarTy,
    pub kind: CastErrorKind,
}

#[derive(Debug, Clone)]
pub enum CastErrorKind {
    /// Cast between these types is not allowed by the OpenQASM 3 spec.
    Unsupported,
    /// Cast requires matching bit widths (e.g. int[n] <-> bit[m] requires n == m).
    WidthMismatch,
}

impl Scalar {
    pub fn uint_from_str_radix(s: &str, radix: u32) -> Result<Self, ParseIntError> {
        let parsed = u128::from_str_radix(&s.replace('_', ""), radix)?;
        let mut value = Awi::from(parsed);
        value.shrink_to_msb();
        Ok(Self::Uint(value))
    }

    pub fn float_from_str(s: &str) -> Result<Self, ParseFloatError> {
        Ok(Self::Float(Float::F64(s.parse::<f64>()?)))
    }

    pub fn bitstring_from_str(s: &str) -> Result<Self, ParseBitstringError> {
        let bits: Vec<bool> = s
            .chars()
            .enumerate()
            .filter_map(|(index, c)| match c {
                '0' => Some(Ok(false)),
                '1' => Some(Ok(true)),
                '_' => None,
                c => Some(Err(ParseBitstringError::InvalidChar(c, index))),
            })
            .collect::<Result<_, _>>()?;
        let mut awi = Awi::zero(bw(bits.len()));
        for (i, &bit) in bits.iter().enumerate() {
            let _ = awi.set(i, bit);
        }
        Ok(Self::Bits(awi))
    }

    pub fn ty(&self) -> ScalarTy {
        match self {
            Scalar::Bool(_) => ScalarTy::Bool,
            Scalar::Int(inner) => ScalarTy::Int(inner.nzbw()),
            Scalar::Uint(inner) => ScalarTy::Uint(inner.nzbw()),
            Scalar::Float(inner) => ScalarTy::Float(inner.size()),
            Scalar::Complex(inner) => ScalarTy::Complex(inner.size()),
            Scalar::Angle(inner) => ScalarTy::Angle(inner.nzbw()),
            Scalar::Timing(_) => ScalarTy::Timing,
            Scalar::Bits(bits) => ScalarTy::Bits(bits.nzbw()),
        }
    }

    pub fn cast_to(self, ty: ScalarTy) -> Result<Scalar, CastError> {
        match (self, ty) {
            // === Identity (same variant, same size) ===
            (Scalar::Bool(b), ScalarTy::Bool) => Ok(Scalar::Bool(b)),
            (Scalar::Timing(t), ScalarTy::Timing) => Ok(Scalar::Timing(t)),

            // === Bool -> X ===
            (Scalar::Bool(b), ScalarTy::Int(n)) => {
                let mut awi = Awi::zero(n);
                awi.bool_(b);
                Ok(Scalar::Int(awi))
            }
            (Scalar::Bool(b), ScalarTy::Uint(n)) => {
                let mut awi = Awi::zero(n);
                awi.bool_(b);
                Ok(Scalar::Uint(awi))
            }
            (Scalar::Bool(b), ScalarTy::Float(fs)) => Ok(Scalar::Float(match fs {
                FloatSize::F32 => Float::F32(if b { 1.0 } else { 0.0 }),
                FloatSize::F64 => Float::F64(if b { 1.0 } else { 0.0 }),
            })),
            (Scalar::Bool(b), ScalarTy::Complex(fs)) => Ok(Scalar::Complex(match fs {
                FloatSize::F32 => Complex::c32(if b { 1. } else { 0. }, 0.),
                FloatSize::F64 => Complex::c64(if b { 1. } else { 0. }, 0.),
            })),
            (Scalar::Bool(b), ScalarTy::Bits(n)) => {
                let mut awi = Awi::zero(n);
                awi.bool_(b);
                Ok(Scalar::Bits(awi))
            }

            // === Int -> X ===
            (Scalar::Int(awi), ScalarTy::Bool) => Ok(Scalar::Bool(!awi.is_zero())),
            (Scalar::Int(mut awi), ScalarTy::Int(n)) => {
                awi.sign_resize(n);
                Ok(Scalar::Int(awi))
            }
            (Scalar::Int(mut awi), ScalarTy::Uint(n)) => {
                // Preserve bit pattern, then reinterpret as unsigned
                awi.sign_resize(n);
                Ok(Scalar::Uint(awi))
            }
            (Scalar::Int(awi), ScalarTy::Float(fs)) => {
                let v = awi.to_i128() as f64;
                Ok(Scalar::Float(match fs {
                    FloatSize::F32 => Float::F32(v as f32),
                    FloatSize::F64 => Float::F64(v),
                }))
            }
            (Scalar::Int(awi), ScalarTy::Complex(fs)) => {
                let v = awi.to_i128() as f64;
                Ok(Scalar::Complex(match fs {
                    FloatSize::F32 => Complex::c32(v as f32, 0.0),
                    FloatSize::F64 => Complex::c64(v, 0.0),
                }))
            }
            (Scalar::Int(awi), ScalarTy::Bits(n)) if awi.nzbw() == n => Ok(Scalar::Bits(awi)),

            // === Uint -> X ===
            (Scalar::Uint(awi), ScalarTy::Bool) => Ok(Scalar::Bool(!awi.is_zero())),
            (Scalar::Uint(mut awi), ScalarTy::Int(n)) => {
                // Preserve bit pattern, then reinterpret as signed
                awi.zero_resize(n);
                Ok(Scalar::Int(awi))
            }
            (Scalar::Uint(mut awi), ScalarTy::Uint(n)) => {
                awi.zero_resize(n);
                Ok(Scalar::Uint(awi))
            }
            (Scalar::Uint(awi), ScalarTy::Float(fs)) => {
                let v = awi.to_u128() as f64;
                Ok(Scalar::Float(match fs {
                    FloatSize::F32 => Float::F32(v as f32),
                    FloatSize::F64 => Float::F64(v),
                }))
            }
            (Scalar::Uint(awi), ScalarTy::Complex(fs)) => {
                let v = awi.to_u128() as f64;
                Ok(Scalar::Complex(match fs {
                    FloatSize::F32 => Complex::c32(v as f32, 0.0),
                    FloatSize::F64 => Complex::c64(v, 0.0),
                }))
            }
            (Scalar::Uint(awi), ScalarTy::Bits(n)) if awi.nzbw() == n => Ok(Scalar::Bits(awi)),

            // === Float -> X ===
            (Scalar::Float(fv), ScalarTy::Bool) => Ok(Scalar::Bool(!fv.is_zero())),
            (Scalar::Float(fv), ScalarTy::Int(n)) => {
                let mut awi = Awi::from(fv.as_f64() as i128);
                awi.sign_resize(n);
                Ok(Scalar::Int(awi))
            }
            (Scalar::Float(fv), ScalarTy::Uint(n)) => {
                let mut awi = Awi::from(fv.as_f64() as u128);
                awi.zero_resize(n);
                Ok(Scalar::Uint(awi))
            }
            (Scalar::Float(fv), ScalarTy::Float(fs)) => Ok(Scalar::Float(match fs {
                FloatSize::F32 => Float::F32(fv.as_f32()),
                FloatSize::F64 => Float::F64(fv.as_f64()),
            })),
            (Scalar::Float(fv), ScalarTy::Complex(fs)) => Ok(Scalar::Complex(match fs {
                FloatSize::F32 => Complex::c32(fv.as_f32(), 0.0),
                FloatSize::F64 => Complex::c64(fv.as_f64(), 0.0),
            })),
            (Scalar::Float(fv), ScalarTy::Angle(n)) => {
                Ok(Scalar::Angle(float_to_angle(fv.as_f64(), n)))
            }

            // === Complex -> X (C99 semantics: extract real part) ===
            (
                Scalar::Complex(cv),
                ScalarTy::Bool | ScalarTy::Int(_) | ScalarTy::Uint(_) | ScalarTy::Float(_),
            ) => Scalar::Float(cv.re()).cast_to(ty).map_err(|err| CastError {
                scalar: Scalar::Complex(cv),
                target: ty,
                kind: err.kind,
            }),
            (Scalar::Complex(cv), ScalarTy::Complex(fs)) => Ok(Scalar::Complex(match fs {
                FloatSize::F32 => Complex::c32(cv.re().as_f32(), cv.im().as_f32()),
                FloatSize::F64 => Complex::c64(cv.re().as_f64(), cv.im().as_f64()),
            })),

            // === Angle -> X ===
            (Scalar::Angle(awi), ScalarTy::Bool) => Ok(Scalar::Bool(!awi.is_zero())),
            (Scalar::Angle(awi), ScalarTy::Angle(m)) => {
                Ok(Scalar::Angle(resize_angle(awi, m.get())))
            }
            (Scalar::Angle(awi), ScalarTy::Bits(m)) if awi.nzbw() == m => Ok(Scalar::Bits(awi)),

            // === Bitstring -> X ===
            (Scalar::Bits(awi), ScalarTy::Bool) => Ok(Scalar::Bool(!awi.is_zero())),
            (Scalar::Bits(awi), ScalarTy::Int(n)) if awi.nzbw() == n => Ok(Scalar::Int(awi)),
            (Scalar::Bits(awi), ScalarTy::Uint(n)) if awi.nzbw() == n => Ok(Scalar::Uint(awi)),
            (Scalar::Bits(awi), ScalarTy::Angle(n)) if awi.nzbw() == n => Ok(Scalar::Angle(awi)),
            (Scalar::Bits(awi), ScalarTy::Bits(n)) if awi.nzbw() == n => Ok(Scalar::Bits(awi)),

            // === Width mismatch cases (produce specific error) ===
            (scalar @ (Scalar::Int(_) | Scalar::Uint(_) | Scalar::Angle(_)), ScalarTy::Bits(_))
            | (
                scalar @ Scalar::Bits(_),
                ScalarTy::Int(_) | ScalarTy::Uint(_) | ScalarTy::Angle(_) | ScalarTy::Bits(_),
            ) => Err(CastError {
                scalar,
                target: ty,
                kind: CastErrorKind::WidthMismatch,
            }),

            // === Everything else is unsupported ===
            (scalar, _) => Err(CastError {
                scalar,
                target: ty,
                kind: CastErrorKind::Unsupported,
            }),
        }
    }
}

/// Convert a float value to an angle[n] representation.
/// angle_uint = round(value * 2^n / 2π), with ties to even, mod 2^n.
fn float_to_angle(value: f64, n: NonZero<usize>) -> Awi {
    let size = n.get();
    let scaled = value * f64::exp2(size as f64) / TAU;
    let rounded = scaled.round_ties_even();
    // Convert to u128 mod 2^n. We need to handle the modular arithmetic
    // since the rounded value might be negative or larger than 2^n.
    let modulus = if size >= 128 {
        u128::MAX
    } else {
        (1u128 << size).wrapping_sub(1)
    };
    let val = (rounded as i128).rem_euclid(modulus as i128 + 1) as u128;
    let mut awi = Awi::from(val);
    awi.zero_resize(n);
    awi
}

/// Resize an angle from its current width to m bits.
/// Widening: pad LSBs with zeros (shift left).
/// Narrowing: truncate LSBs (shift right).
fn resize_angle(mut awi: Awi, m: usize) -> Awi {
    let n = awi.bw();
    if m == n {
        return awi;
    }
    if m > n {
        // Widen: zero_resize then shift left by (m - n) to pad LSBs
        awi.zero_resize(bw(m));
        let _ = awi.shl_(m - n);
        awi
    } else {
        // Narrow: shift right by (n - m) to truncate LSBs, then resize
        let _ = awi.lshr_(n - m);
        awi.zero_resize(bw(m));
        awi
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<u8> for Scalar {
    fn from(value: u8) -> Self {
        Self::Uint(Awi::from(value))
    }
}

impl From<u16> for Scalar {
    fn from(value: u16) -> Self {
        Self::Uint(Awi::from(value))
    }
}

impl From<u32> for Scalar {
    fn from(value: u32) -> Self {
        Self::Uint(Awi::from(value))
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Self {
        Self::Uint(Awi::from(value))
    }
}

impl From<u128> for Scalar {
    fn from(value: u128) -> Self {
        Self::Uint(Awi::from(value))
    }
}

impl From<i8> for Scalar {
    fn from(value: i8) -> Self {
        Self::Int(Awi::from(value))
    }
}

impl From<i16> for Scalar {
    fn from(value: i16) -> Self {
        Self::Int(Awi::from(value))
    }
}

impl From<i32> for Scalar {
    fn from(value: i32) -> Self {
        Self::Int(Awi::from(value))
    }
}

impl From<i64> for Scalar {
    fn from(value: i64) -> Self {
        Self::Int(Awi::from(value))
    }
}

impl From<i128> for Scalar {
    fn from(value: i128) -> Self {
        Self::Int(Awi::from(value))
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self::Float(value.into())
    }
}

impl From<f64> for Scalar {
    fn from(value: f64) -> Self {
        Self::Float(value.into())
    }
}

impl From<Float> for Scalar {
    fn from(value: Float) -> Self {
        Self::Float(value)
    }
}

impl From<num_complex::Complex<f32>> for Scalar {
    fn from(value: num_complex::Complex<f32>) -> Self {
        Self::Complex(value.into())
    }
}

impl From<num_complex::Complex<f64>> for Scalar {
    fn from(value: num_complex::Complex<f64>) -> Self {
        Self::Complex(value.into())
    }
}

impl From<Complex> for Scalar {
    fn from(value: Complex) -> Self {
        Self::Complex(value)
    }
}

impl From<TimingValue> for Scalar {
    fn from(value: TimingValue) -> Self {
        Self::Timing(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: create an angle scalar with given value and bit width.
    fn angle(val: u128, n: usize) -> Scalar {
        let mut awi = Awi::from(val);
        awi.zero_resize(bw(n));
        Scalar::Angle(awi)
    }

    /// Helper: create a bitstring scalar from a bool slice.
    fn bits(bs: &[bool]) -> Scalar {
        let mut awi = Awi::zero(bw(bs.len()));
        for (i, &b) in bs.iter().enumerate() {
            let _ = awi.set(i, b);
        }
        Scalar::Bits(awi)
    }

    /// Extract the Awi from Int/Uint/Angle/Bitstring scalar and return its u128 value.
    fn to_u128(s: &Scalar) -> u128 {
        match s {
            Scalar::Int(a) | Scalar::Uint(a) | Scalar::Angle(a) | Scalar::Bits(a) => a.to_u128(),
            _ => panic!("expected Int/Uint/Angle/Bitstring"),
        }
    }

    // ── Bool casts ──

    #[test]
    fn bool_to_int() {
        let t = Scalar::Bool(true).cast_to(ScalarTy::int(32)).unwrap();
        assert_eq!(to_u128(&t), 1);
        let f = Scalar::Bool(false).cast_to(ScalarTy::int(32)).unwrap();
        assert_eq!(to_u128(&f), 0);
    }

    #[test]
    fn bool_to_uint() {
        let t = Scalar::Bool(true).cast_to(ScalarTy::uint(16)).unwrap();
        assert_eq!(to_u128(&t), 1);
    }

    #[test]
    fn bool_to_float() {
        let t = Scalar::Bool(true)
            .cast_to(ScalarTy::Float(FloatSize::F64))
            .unwrap();
        assert!(matches!(t, Scalar::Float(Float::F64(v)) if v == 1.0));
        let f = Scalar::Bool(false)
            .cast_to(ScalarTy::Float(FloatSize::F32))
            .unwrap();
        assert!(matches!(f, Scalar::Float(Float::F32(v)) if v == 0.0));
    }

    #[test]
    fn bool_to_bitstring() {
        let t = Scalar::Bool(true).cast_to(ScalarTy::bits(8)).unwrap();
        if let Scalar::Bits(awi) = &t {
            assert_eq!(awi.bw(), 8);
            assert!(awi.get(0).unwrap()); // LSB set
            assert_eq!(awi.to_u128(), 1); // only bit 0 set
        } else {
            panic!();
        }
    }

    #[test]
    fn bool_to_angle_fails() {
        assert!(Scalar::Bool(true).cast_to(ScalarTy::angle(8)).is_err());
    }

    // ── Int casts ──

    #[test]
    fn int_to_bool() {
        let s = Scalar::from(42i32).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(true)));
        let s = Scalar::from(0i32).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(false)));
    }

    #[test]
    fn int_to_uint_preserves_bits() {
        // int[16](-1) → uint[16] should give 0xFFFF (65535)
        let neg = Scalar::from(-1i16).cast_to(ScalarTy::uint(16)).unwrap();
        assert_eq!(to_u128(&neg), 0xFFFF);
    }

    #[test]
    fn int_roundtrip_through_uint() {
        // x == int[n](uint[n](x))
        let original = Scalar::from(-42i32);
        let as_uint = original.cast_to(ScalarTy::uint(32)).unwrap();
        let back = as_uint.cast_to(ScalarTy::int(32)).unwrap();
        assert_eq!(to_u128(&back), (-42i32 as u32) as u128);
    }

    #[test]
    fn int_to_float() {
        let s = Scalar::from(42i32)
            .cast_to(ScalarTy::Float(FloatSize::F64))
            .unwrap();
        assert!(matches!(s, Scalar::Float(Float::F64(v)) if v == 42.0));
    }

    #[test]
    fn int_to_bit_width_mismatch_fails() {
        let s = Scalar::from(42i32).cast_to(ScalarTy::bits(16));
        assert!(matches!(s.unwrap_err().kind, CastErrorKind::WidthMismatch));
    }

    #[test]
    fn int_to_bit_matching_width() {
        // int[8](0xFF) → bit[8]: direct bit reinterpretation
        let s = Scalar::from(-1i8).cast_to(ScalarTy::bits(8)).unwrap();
        if let Scalar::Bits(awi) = &s {
            assert_eq!(awi.to_u128(), 0xFF); // all bits set for -1
        } else {
            panic!();
        }
    }

    #[test]
    fn int_to_angle_fails() {
        assert!(Scalar::from(1i32).cast_to(ScalarTy::angle(8)).is_err());
    }

    // ── Uint casts ──

    #[test]
    fn uint_to_bool() {
        let s = Scalar::from(0u32).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(false)));
        let s = Scalar::from(5u32).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(true)));
    }

    #[test]
    fn uint_widen() {
        let s = Scalar::from(255u8).cast_to(ScalarTy::uint(32)).unwrap();
        assert_eq!(to_u128(&s), 255);
    }

    #[test]
    fn uint_narrow() {
        // uint[32](256) → uint[8] truncates to 0
        let mut awi = Awi::from(256u32);
        awi.zero_resize(bw(32));
        let s = Scalar::Uint(awi).cast_to(ScalarTy::uint(8)).unwrap();
        assert_eq!(to_u128(&s), 0);
    }

    // ── Float casts ──

    #[test]
    fn float_to_bool() {
        let s = Scalar::from(0.0f64).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(false)));
        #[allow(clippy::approx_constant)]
        let s = Scalar::from(3.14f64).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(true)));
    }

    #[test]
    fn float_to_int_truncates() {
        // C99: truncation toward zero
        let s = Scalar::from(3.9f64).cast_to(ScalarTy::int(32)).unwrap();
        assert_eq!(to_u128(&s), 3);
        let s = Scalar::from(-3.9f64).cast_to(ScalarTy::int(32)).unwrap();
        // -3 in 32-bit two's complement
        let awi = match &s {
            Scalar::Int(a) => a,
            _ => panic!(),
        };
        assert_eq!(awi.to_i128(), -3);
    }

    #[test]
    fn float_to_uint_truncates() {
        let s = Scalar::from(7.99f64).cast_to(ScalarTy::uint(16)).unwrap();
        assert_eq!(to_u128(&s), 7);
    }

    #[test]
    fn float_to_bit_fails() {
        assert!(matches!(
            Scalar::from(1.0f64)
                .cast_to(ScalarTy::bits(32))
                .unwrap_err()
                .kind,
            CastErrorKind::Unsupported
        ));
    }

    // ── Float → Angle ──

    #[test]
    fn float_to_angle_pi() {
        // angle[4](π) should be "1000" = 8
        let s = Scalar::from(PI).cast_to(ScalarTy::angle(4)).unwrap();
        assert_eq!(to_u128(&s), 8); // 0b1000
    }

    #[test]
    fn float_to_angle_pi_over_2() {
        // angle[6](π/2) should be "010000" = 16
        let s = Scalar::from(PI / 2.0).cast_to(ScalarTy::angle(6)).unwrap();
        assert_eq!(to_u128(&s), 16); // 0b010000
    }

    #[test]
    fn float_to_angle_ties_to_even() {
        // From the spec: angle[8](two_pi * 127/512) should be "01000000" (64)
        // not "00111111" (63), because of ties-to-even.
        #[allow(clippy::approx_constant)]
        let two_pi: f64 = 6.283185307179586;
        let f = two_pi * (127.0 / 512.0);
        let s = Scalar::from(f).cast_to(ScalarTy::angle(8)).unwrap();
        assert_eq!(to_u128(&s), 64); // 0b01000000
    }

    // ── Angle casts ──

    #[test]
    fn angle_to_bool() {
        let s = angle(0, 8).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(false)));
        let s = angle(1, 8).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(true)));
    }

    #[test]
    fn angle_widen_pads_lsb() {
        // angle[4] "1000" (= π) → angle[6] should be "100000" = 32
        let s = angle(0b1000, 4).cast_to(ScalarTy::angle(6)).unwrap();
        assert_eq!(to_u128(&s), 0b100000);
    }

    #[test]
    fn angle_narrow_truncates_lsb() {
        // angle[6] "010000" (= π/2) → angle[4] should be "0100" = 4
        let s = angle(0b010000, 6).cast_to(ScalarTy::angle(4)).unwrap();
        assert_eq!(to_u128(&s), 0b0100);
    }

    #[test]
    fn angle_to_bit_matching_width() {
        let s = angle(0b1010, 4).cast_to(ScalarTy::bits(4)).unwrap();
        if let Scalar::Bits(awi) = &s {
            assert_eq!(awi.bw(), 4);
            assert_eq!(awi.to_u128(), 0b1010);
        } else {
            panic!();
        }
    }

    #[test]
    fn angle_to_bit_width_mismatch_fails() {
        assert!(angle(0, 4).cast_to(ScalarTy::bits(8)).is_err());
    }

    #[test]
    fn angle_to_int_fails() {
        assert!(angle(0, 8).cast_to(ScalarTy::int(8)).is_err());
    }

    #[test]
    fn angle_to_float_fails() {
        assert!(
            angle(0, 8)
                .cast_to(ScalarTy::Float(FloatSize::F64))
                .is_err()
        );
    }

    // ── Bitstring casts ──

    #[test]
    fn bit_to_bool() {
        let s = bits(&[false, false, false])
            .cast_to(ScalarTy::Bool)
            .unwrap();
        assert!(matches!(s, Scalar::Bool(false)));
        let s = bits(&[true, false]).cast_to(ScalarTy::Bool).unwrap();
        assert!(matches!(s, Scalar::Bool(true)));
    }

    #[test]
    fn bit_to_uint_matching() {
        // bit[8] "00001111" → uint[8] = 15 (little-endian: bits 0-3 set)
        let s = bits(&[true, true, true, true, false, false, false, false])
            .cast_to(ScalarTy::uint(8))
            .unwrap();
        assert_eq!(to_u128(&s), 0b00001111);
    }

    #[test]
    fn bit_to_int_matching() {
        // bit[8] all ones → int[8] = -1
        let s = bits(&[true; 8]).cast_to(ScalarTy::int(8)).unwrap();
        let awi = match &s {
            Scalar::Int(a) => a,
            _ => panic!(),
        };
        assert_eq!(awi.to_i128(), -1);
    }

    #[test]
    fn bit_to_float_fails() {
        assert!(
            bits(&[true, false])
                .cast_to(ScalarTy::Float(FloatSize::F64))
                .is_err()
        );
    }

    #[test]
    fn bit_to_angle_matching() {
        let s = bits(&[false, false, false, true])
            .cast_to(ScalarTy::angle(4))
            .unwrap();
        // bit 3 set → value 8
        assert_eq!(to_u128(&s), 0b1000);
    }

    #[test]
    fn bit_to_uint_width_mismatch_fails() {
        assert!(bits(&[true, false]).cast_to(ScalarTy::uint(8)).is_err());
    }

    // ── Unsupported casts ──

    #[test]
    fn timing_to_anything_fails() {
        let timing = Scalar::Timing(TimingValue {
            value: TimingNumber::Integer(100),
            unit: TimeUnit::Ns,
        });
        assert!(timing.clone().cast_to(ScalarTy::Bool).is_err());
        assert!(timing.clone().cast_to(ScalarTy::int(32)).is_err());
        assert!(timing.cast_to(ScalarTy::Float(FloatSize::F64)).is_err());
    }

    #[test]
    fn anything_to_timing_fails() {
        assert!(Scalar::Bool(true).cast_to(ScalarTy::Timing).is_err());
        assert!(Scalar::from(1i32).cast_to(ScalarTy::Timing).is_err());
        assert!(Scalar::from(1.0f64).cast_to(ScalarTy::Timing).is_err());
    }
}
