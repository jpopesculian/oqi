use std::fmt;

use crate::{
    DurationUnit,
    duration::Duration,
    error::{Error, Result},
    value::ValueTy,
};
use num_complex::{Complex64, c64};

#[derive(Clone, Copy)]
pub enum Primitive {
    Bit(bool),
    BitReg(u128),
    Uint(u128),
    Int(i128),
    Float(f64),
    Complex(Complex64),
    Duration(Duration),
    Angle(u128),
}

impl Primitive {
    pub const PI: Self = Self::Float(core::f64::consts::PI);
    pub const TAU: Self = Self::Float(core::f64::consts::TAU);
    pub const E: Self = Self::Float(core::f64::consts::E);

    pub fn bit(v: bool) -> Self {
        Self::Bit(v)
    }

    pub fn int(v: impl Into<i128>) -> Self {
        Self::Int(v.into())
    }

    pub fn uint(v: impl Into<u128>) -> Self {
        Self::Uint(v.into())
    }

    pub fn float(v: impl Into<f64>) -> Self {
        Self::Float(v.into())
    }

    pub fn complex(re: impl Into<f64>, im: impl Into<f64>) -> Self {
        Self::Complex(c64(re.into(), im.into()))
    }

    pub fn duration(v: impl Into<f64>, unit: DurationUnit) -> Self {
        Self::Duration(Duration::new(v.into(), unit))
    }

    pub fn angle(radians: impl Into<f64>) -> Self {
        Self::Angle(radians_to_angle(radians.into()))
    }

    pub fn bitreg(bits: impl Into<u128>) -> Self {
        Self::BitReg(bits.into())
    }

    #[inline]
    pub const fn as_bit(self) -> bool {
        match self {
            Self::Bit(b) => b,
            Self::Uint(v) | Self::Angle(v) | Self::BitReg(v) => v != 0, // truthiness: nonzero -> true
            Self::Int(v) => v != 0,     // truthiness: nonzero -> true
            Self::Float(v) => v != 0.0, // truthiness: nonzero -> true
            Self::Complex(c) => c.re != 0.0 || c.im != 0.0, // truthiness: nonzero -> true
            Self::Duration(d) => d.value != 0.0, // truthiness: nonzero -> true
        }
    }

    #[inline]
    pub fn as_int(self, bits: BitWidth) -> Option<i128> {
        let v = match self {
            Self::Bit(b) => {
                if b {
                    1
                } else {
                    0
                }
            }
            Self::BitReg(b) => b as i128,
            Self::Int(v) => v,
            Self::Float(v) => v as i128,
            Self::Complex(c) => c.re as i128, // only take real part for uint conversion
            Self::Duration(d) => d.value as i128,
            Self::Uint(v) => v as i128,
            Self::Angle(_v) => return None,
        };
        Some(resize_int(v, bits))
    }

    #[inline]
    pub fn as_uint(self, bits: BitWidth) -> Option<u128> {
        let v = match self {
            Self::Bit(b) => {
                if b {
                    1
                } else {
                    0
                }
            }
            Self::Int(v) => v as u128,
            Self::Float(v) => v as u128,
            Self::Complex(c) => c.re as u128, // only take real part for uint conversion
            Self::Duration(d) => d.value as u128,
            Self::Uint(v) | Self::BitReg(v) => v,
            Self::Angle(_v) => return None,
        };
        Some(resize_uint(v, bits))
    }

    #[inline]
    pub fn as_float(self, width: FloatWidth) -> Option<f64> {
        let v = match self {
            Self::Bit(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Float(v) => v,
            Self::Int(v) => v as f64,
            Self::Uint(v) => v as f64,
            Self::Complex(c) => c.re, // only take real part for float conversion
            Self::Duration(d) => d.value,
            Self::Angle(v) => angle_to_radians(v),
            _ => return None,
        };
        Some(match width {
            FloatWidth::F32 => v as f32 as f64, // round-trip through f32 to truncate precision
            FloatWidth::F64 => v,
        })
    }

    #[inline]
    pub fn as_complex(self, width: FloatWidth) -> Option<Complex64> {
        let v = match self {
            Self::Bit(b) => c64(if b { 1.0 } else { 0.0 }, 0.0),
            Self::Complex(c) => c,
            Self::Int(v) => c64(v as f64, 0.0),
            Self::Uint(v) => c64(v as f64, 0.0),
            Self::Float(v) => c64(v, 0.0),
            Self::Duration(d) => c64(d.value, 0.0),
            Self::Angle(v) => c64(angle_to_radians(v), 0.0),
            _ => return None,
        };
        Some(match width {
            FloatWidth::F32 => c64(v.re as f32 as f64, v.im as f32 as f64), // round-trip through
            // f32 to truncate precision
            FloatWidth::F64 => v,
        })
    }

    #[inline]
    pub fn as_duration(self) -> Option<Duration> {
        match self {
            Self::Duration(d) => Some(d),
            _ => None,
        }
    }

    #[inline]
    pub fn as_angle(self, bw: BitWidth) -> Option<u128> {
        Some(match self {
            Self::Bit(b) => {
                if b {
                    u128::MAX
                } else {
                    0
                }
            }
            Self::Angle(v) => resize_angle(v, bw),
            Self::Float(f) => radians_to_angle_bw(f, bw),
            Self::Complex(Complex64 { re: f, .. }) => radians_to_angle_bw(f, bw),
            _ => return None,
        })
    }

    #[inline]
    pub fn as_bitreg(self, bw: BitWidth) -> Option<u128> {
        let v = match self {
            Self::Bit(b) => {
                if b {
                    1
                } else {
                    0
                }
            }
            Self::Uint(v) => v,
            Self::Int(v) => v as u128,
            Self::BitReg(b) => b,
            _ => return None,
        };
        Some(resize_uint(v, bw))
    }

    pub fn assert_fits(self, ty: PrimitiveTy) -> Result<Self> {
        let Some(bw) = ty.bw() else { return Ok(self) };
        let overflow = match self {
            Self::Uint(v) => v >> bw.get() != 0,
            Self::Int(v) => {
                let shifted = v >> (bw.get() - 1);
                shifted != 0 && shifted != -1
            }
            _ => false,
        };
        if overflow {
            Err(Error::Overflow)
        } else {
            Ok(self)
        }
    }

    pub const fn resize(self, ty: PrimitiveTy) -> Primitive {
        let Some(bw) = ty.bw() else { return self };
        match self {
            Self::Uint(v) => Self::Uint(resize_uint(v, bw)),
            Self::Int(v) => Self::Int(resize_int(v, bw)),
            Self::BitReg(v) => Self::Uint(resize_uint(v, bw)),
            Self::Angle(v) => Self::Angle(resize_angle(v, bw)),
            other => other,
        }
    }

    #[inline]
    pub const fn default_ty(self) -> PrimitiveTy {
        match self {
            Primitive::Bit(_) => PrimitiveTy::Bit,
            Primitive::BitReg(_) => PrimitiveTy::BitReg(BitWidth::B128),
            Primitive::Int(_) => PrimitiveTy::Int(BitWidth::B128),
            Primitive::Uint(_) => PrimitiveTy::Uint(BitWidth::B128),
            Primitive::Float(_) => PrimitiveTy::Float(FloatWidth::F64),
            Primitive::Complex(_) => PrimitiveTy::Complex(FloatWidth::F64),
            Primitive::Duration(_) => PrimitiveTy::Duration,
            Primitive::Angle(_) => PrimitiveTy::Angle(BitWidth::B128),
        }
    }

    pub fn as_ty(self, ty: PrimitiveTy) -> Result<Primitive> {
        use PrimitiveTy::*;
        match ty {
            Bit | Bool => Some(Self::Bit(self.as_bit())),
            BitReg(bw) => self.as_bitreg(bw).map(Self::BitReg),
            Int(bw) => self.as_int(bw).map(Self::Int),
            Uint(bw) => self.as_uint(bw).map(Self::Uint),
            Float(fw) => self.as_float(fw).map(Self::Float),
            Complex(fw) => self.as_complex(fw).map(Self::Complex),
            Duration => self.as_duration().map(Self::Duration),
            Angle(bw) => self.as_angle(bw).map(Self::Angle),
        }
        .ok_or(Error::TypeMismatch { value: self, ty })
    }

    pub fn cast(self, from: PrimitiveTy, to: PrimitiveTy) -> Result<Self> {
        from.cast(to)?;
        self.as_ty(from)?.as_ty(to)
    }
}

impl fmt::Debug for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Primitive::*;
        match self {
            Bit(b) => f.debug_tuple("Bit").field(b).finish(),
            Uint(u) => f.debug_tuple("Uint").field(u).finish(),
            Int(i) => f.debug_tuple("Int").field(i).finish(),
            Float(fl) => f.debug_tuple("Float").field(fl).finish(),
            Complex(c) => fmt::Debug::fmt(c, f),
            Duration(d) => fmt::Debug::fmt(d, f),
            Angle(a) => f
                .debug_tuple("Angle")
                .field(&format_args!("{:0128b}", a))
                .finish(),
            BitReg(r) => f
                .debug_tuple("BitReg")
                .field(&format_args!("{:0128b}", r))
                .finish(),
        }
    }
}

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug)]
pub enum FloatWidth {
    F32,
    F64,
}

impl FloatWidth {
    pub const fn get(self) -> usize {
        match self {
            FloatWidth::F32 => 32,
            FloatWidth::F64 => 64,
        }
    }
}

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug)]
pub struct BitWidth(u32);

impl BitWidth {
    pub const B128: BitWidth = BitWidth(128);
    pub const B72: BitWidth = BitWidth(72);
    pub const B64: BitWidth = BitWidth(64);
    pub const B32: BitWidth = BitWidth(32);
    pub const B16: BitWidth = BitWidth(16);
    pub const B8: BitWidth = BitWidth(8);

    #[inline]
    pub const fn new(bits: u32) -> Result<Self> {
        if bits == 0 || bits > 128 {
            Err(Error::BadDimensions {
                received: bits as usize,
                min: 1,
                max: 128,
            })
        } else {
            Ok(Self(bits))
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl PartialEq<u32> for BitWidth {
    fn eq(&self, other: &u32) -> bool {
        self.get() == *other
    }
}

impl PartialEq<BitWidth> for u32 {
    fn eq(&self, other: &BitWidth) -> bool {
        *self == other.get()
    }
}

impl PartialOrd<u32> for BitWidth {
    fn partial_cmp(&self, other: &u32) -> Option<std::cmp::Ordering> {
        self.get().partial_cmp(other)
    }
}

impl PartialOrd<BitWidth> for u32 {
    fn partial_cmp(&self, other: &BitWidth) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.get())
    }
}

#[inline]
pub fn bw(bits: u32) -> BitWidth {
    BitWidth::new(bits).unwrap()
}

#[derive(Clone, Copy, Debug)]
pub enum PrimitiveTy {
    Bool,
    Bit,
    BitReg(BitWidth),
    Int(BitWidth),
    Uint(BitWidth),
    Float(FloatWidth),
    Complex(FloatWidth),
    Duration,
    Angle(BitWidth),
}

impl PrimitiveTy {
    pub const fn bw(&self) -> Option<BitWidth> {
        match self {
            PrimitiveTy::BitReg(n)
            | PrimitiveTy::Int(n)
            | PrimitiveTy::Uint(n)
            | PrimitiveTy::Angle(n) => Some(*n),
            _ => None,
        }
    }

    pub fn fw(&self) -> Option<FloatWidth> {
        match self {
            PrimitiveTy::Float(w) | PrimitiveTy::Complex(w) => Some(*w),
            _ => None,
        }
    }

    pub fn cast(self, ty: PrimitiveTy) -> Result<Self> {
        use PrimitiveTy::*;
        match (self, ty) {
            // no-op
            (Bit, Bit) | (Bit, Bool) | (Bool, Bit) | (Bool, Bool) | (Duration, Duration)
            // Bit-like <-> Unsigned-like
            | (Bit | Bool, Uint(_) )
            | (Uint(_) , Bit | Bool)
            // Bit-like <-> Signed
            | (Bit | Bool, Int(_))
            // Bit-like <-> Float
            | (Bit | Bool, Float(_))
            | (Float(_), Bit | Bool)
            // Bit-like <-> Complex
            | (Bit | Bool, Complex(_))
            | (Complex(_), Bit | Bool)
            // Unsigned-like <-> Unsigned-like
            | (Uint(_) , Uint(_) )
            // Unsigned-like <-> Signed
            | (Uint(_) , Int(_))
            | (Int(_), Uint(_) )
            // Signed <-> Signed
            | (Int(_), Int(_))

            // Integer-like -> Float
            | (Uint(_) , Float(_))
            | (Int(_), Float(_))
            // Float -> Integer-like
            | (Float(_), Uint(_) )
            | (Float(_), Int(_))
            // Float <-> Float
            | (Float(_), Float(_))
            // Float <-> Angle
            | (Float(_), Angle(_))
            | (Angle(_), Float(_))
            // Complex <-> Angle
            | (Complex(_), Angle(_))
            | (Angle(_), Complex(_))

            // Numeric -> Complex
            | (Uint(_) , Complex(_))
            | (Int(_), Complex(_))
            | (Float(_), Complex(_))
            // Complex <-> Complex
            | (Complex(_), Complex(_))

            // Angle <-> Angle
            | (Angle(_), Angle(_)) => {},

            // Bitreg-like <-> Bitreg-like
            (BitReg(a), BitReg(b))
            | (Uint(a), BitReg(b))
            | (BitReg(a), Int(b))
            | (Int(a), BitReg(b))
            | (BitReg(a), Uint(b))
            | (Angle(a), BitReg(b))
            | (BitReg(a), Angle(b))
            if a == b => {}
            _ => {
                return Err(Error::UnsupportedCast {
                    from: Box::new(ValueTy::Scalar(self)),
                    to: Box::new(ValueTy::Scalar(ty)),
                });
            }
        }
        Ok(ty)
    }
}

impl PartialEq for PrimitiveTy {
    fn eq(&self, other: &Self) -> bool {
        use PrimitiveTy::*;
        match (self, other) {
            (Bit, Bit) | (Bit, Bool) | (Bool, Bit) | (Bool, Bool) => true,
            (BitReg(a), BitReg(b)) if a == b => true,
            (Int(a), Int(b)) if a == b => true,
            (Uint(a), Uint(b)) if a == b => true,
            (Float(a), Float(b)) if a == b => true,
            (Complex(a), Complex(b)) if a == b => true,
            (Duration, Duration) => true,
            (Angle(a), Angle(b)) if a == b => true,
            _ => false,
        }
    }
}

pub fn promote_arithmetic(lhs: PrimitiveTy, rhs: PrimitiveTy) -> Option<PrimitiveTy> {
    use PrimitiveTy::*;
    Some(match (lhs, rhs) {
        (Bit, Uint(a)) | (Uint(a), Bit) => Uint(a),
        (Bit, Int(a)) | (Int(a), Bit) => Int(a),
        (Bit, Float(a)) | (Float(a), Bit) => Float(a),
        (Bit, Complex(a)) | (Complex(a), Bit) => Complex(a),
        (Uint(a), Int(b)) | (Int(b), Uint(a)) => {
            if a >= b {
                Uint(a)
            } else {
                Int(b)
            }
        }
        (Uint(a), Uint(b)) => Uint(a.max(b)),
        (Int(a), Int(b)) => Int(a.max(b)),
        (Uint(_), Float(fw))
        | (Float(fw), Uint(_))
        | (Int(_), Float(fw))
        | (Float(fw), Int(_))
        | (Float(fw), Angle(_))
        | (Angle(_), Float(fw)) => Float(fw),
        (Uint(_), Complex(fw))
        | (Complex(fw), Uint(_))
        | (Int(_), Complex(fw))
        | (Complex(fw), Int(_))
        | (Angle(_), Complex(fw))
        | (Complex(fw), Angle(_)) => Complex(fw),
        (Float(a), Float(b)) => Float(a.max(b)),
        (Float(a), Complex(b)) | (Complex(b), Float(a)) => Complex(a.max(b)),
        (Complex(a), Complex(b)) => Complex(a.max(b)),
        (Duration, Duration) => Duration,
        (Angle(a), Angle(b)) => Angle(a.max(b)),
        _ => return None,
    })
}

impl fmt::Display for PrimitiveTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PrimitiveTy::*;
        match self {
            Bool => write!(f, "bool"),
            Bit => write!(f, "bit"),
            BitReg(bw) => write!(f, "bit[{}]", bw.get()),
            Uint(bw) => write!(f, "uint[{}]", bw.get()),
            Int(bw) => write!(f, "int[{}]", bw.get()),
            Float(fw) => write!(f, "float[{}]", fw.get()),
            Complex(fw) => write!(f, "complex[float[{}]]", fw.get()),
            Angle(bw) => write!(f, "angle[{}]", bw.get()),
            Duration => write!(f, "duration"),
        }
    }
}

#[inline]
const fn resize_uint(v: u128, bw: BitWidth) -> u128 {
    let mask = if matches!(bw, BitWidth(128)) {
        u128::MAX
    } else {
        (1u128 << bw.get()) - 1
    };
    v & mask
}

#[inline]
pub(crate) const fn resize_int(v: i128, bw: BitWidth) -> i128 {
    let shift = 128 - bw.get();
    (v << shift) >> shift
}

#[inline]
const fn resize_angle(v: u128, bw: BitWidth) -> u128 {
    let mask = if matches!(bw, BitWidth(128)) {
        u128::MAX
    } else {
        !(u128::MAX >> bw.get())
    };
    v & mask
}

#[inline]
fn radians_to_angle(radians: f64) -> u128 {
    radians_to_angle_bw(radians, BitWidth::B128)
}

#[inline]
fn radians_to_angle_bw(radians: f64, bw: BitWidth) -> u128 {
    use std::f64::consts::TAU;

    // Map radians into [0, TAU)
    let normalized = radians.rem_euclid(TAU);

    // Convert to a fraction of a full turn
    let turns = normalized / TAU;

    // Quantize into 2^bitwidth discrete values
    let buckets = 2.0_f64.powi(bw.get() as i32);
    let quantized = (turns * buckets).round_ties_even() as u128;

    // Wrap 2^bitwidth back to 0
    let wrapped = match bw {
        BitWidth(128) => quantized,
        _ => quantized % (1u128 << bw.get()),
    };

    // Store in the high bits, zeroing the low bits
    wrapped << (128 - bw.get())
}

#[inline]
fn angle_to_radians(angle: u128) -> f64 {
    angle_to_radians_bw(angle, BitWidth::B128)
}

#[inline]
fn angle_to_radians_bw(angle: u128, bw: BitWidth) -> f64 {
    angle as f64 * core::f64::consts::TAU / f64::exp2(bw.get() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use FloatWidth::*;
    use PrimitiveTy::*;

    // --- Uint <-> Uint ---

    #[test]
    fn uint_to_uint_widen() {
        let s = Primitive::uint(200_u128);
        let r = s.cast(Uint(bw(8)), Uint(bw(16))).unwrap();
        assert_eq!(r.as_uint(bw(16)).unwrap(), 200);
    }

    #[test]
    fn uint_to_uint_truncate() {
        let s = Primitive::uint(0x1FF_u128); // 511, 9 bits
        let r = s.cast(Uint(bw(16)), Uint(bw(8))).unwrap();
        assert_eq!(r.as_uint(bw(8)).unwrap(), 0xFF); // truncated to 8 bits
    }

    // --- Int <-> Int ---

    #[test]
    fn int_to_int_widen() {
        let s = Primitive::int(-5_i128);
        let r = s.cast(Int(bw(8)), Int(bw(16))).unwrap();
        assert_eq!(r.as_int(bw(16)).unwrap(), -5);
    }

    #[test]
    fn int_to_int_truncate() {
        let s = Primitive::int(-1_i128);
        let r = s.cast(Int(bw(32)), Int(bw(8))).unwrap();
        assert_eq!(r.as_int(bw(8)).unwrap(), -1);
    }

    // --- Uint <-> Int ---

    #[test]
    fn uint_to_int() {
        let s = Primitive::uint(42_u128);
        let r = s.cast(Uint(bw(8)), Int(bw(16))).unwrap();
        assert_eq!(r.as_int(bw(16)).unwrap(), 42);
    }

    #[test]
    fn int_to_uint() {
        let s = Primitive::int(-1_i128);
        let r = s.cast(Int(bw(8)), Uint(bw(8))).unwrap();
        assert_eq!(r.as_uint(bw(8)).unwrap(), 0xFF);
    }

    // --- Bit <-> Uint/Int ---

    #[test]
    fn bit_to_uint() {
        let s = Primitive::bit(true);
        let r = s.cast(Bit, Uint(bw(32))).unwrap();
        assert_eq!(r.as_uint(bw(32)).unwrap(), 1);
    }

    #[test]
    fn uint_to_bit_nonzero() {
        let s = Primitive::uint(42_u128);
        let r = s.cast(Uint(bw(8)), Bit).unwrap();
        assert!(r.as_bit()); // nonzero -> true (truthiness)
    }

    #[test]
    fn uint_to_bit_one() {
        let s = Primitive::uint(1_u128);
        let r = s.cast(Uint(bw(8)), Bit).unwrap();
        assert!(r.as_bit());
    }

    // --- Uint/Int -> Float ---

    #[test]
    fn uint_to_f32() {
        let s = Primitive::uint(100_u128);
        let r = s.cast(Uint(bw(8)), Float(F32)).unwrap();
        assert_eq!(r.as_float(F32).unwrap(), 100.0);
    }

    #[test]
    fn uint_to_f64() {
        let s = Primitive::uint(1_000_000_u128);
        let r = s.cast(Uint(bw(32)), Float(F64)).unwrap();
        assert_eq!(r.as_float(F64).unwrap(), 1_000_000.0);
    }

    #[test]
    fn int_to_f64() {
        let s = Primitive::int(-50_i128);
        let r = s.cast(Int(bw(8)), Float(F64)).unwrap();
        assert_eq!(r.as_float(F64).unwrap(), -50.0);
    }

    // --- Float -> Uint/Int ---

    #[test]
    fn f64_to_uint() {
        let s = Primitive::float(3.7);
        let r = s.cast(Float(F64), Uint(bw(32))).unwrap();
        assert_eq!(r.as_uint(bw(32)).unwrap(), 3); // truncates toward zero
    }

    #[test]
    fn f64_negative_to_uint_saturates() {
        let s = Primitive::float(-1.0);
        let r = s.cast(Float(F64), Uint(bw(32))).unwrap();
        assert_eq!(r.as_uint(bw(32)).unwrap(), 0); // saturates to 0
    }

    #[test]
    fn f64_nan_to_uint_saturates() {
        let s = Primitive::float(f64::NAN);
        let r = s.cast(Float(F64), Uint(bw(32))).unwrap();
        assert_eq!(r.as_uint(bw(32)).unwrap(), 0); // NaN -> 0
    }

    #[test]
    fn f64_inf_to_int_saturates() {
        let s = Primitive::float(f64::INFINITY);
        let r = s.cast(Float(F64), Int(bw(32))).unwrap();
        // f64::INFINITY as i128 saturates to i128::MAX, then masked to 32 bits
        assert!(r.as_int(bw(32)).unwrap() != 0);
    }

    #[test]
    fn f64_to_int() {
        let s = Primitive::float(-7.9);
        let r = s.cast(Float(F64), Int(bw(32))).unwrap();
        assert_eq!(r.as_int(bw(32)).unwrap(), -7);
    }

    #[test]
    fn float_to_angle_rounds_ties_to_even() {
        let two_pi = core::f64::consts::TAU;
        let f = two_pi * (127.0 / 512.0);
        println!("{:?}", Primitive::angle(f));
        let r = Primitive::float(f).cast(Float(F64), Angle(bw(8))).unwrap();
        println!("{:?}", r);
        assert_eq!(r.as_angle(bw(8)).unwrap(), 0b0100_0000 << 120);
    }

    #[test]
    fn float_inf_to_angle_returns_0() {
        let res = Primitive::float(f64::INFINITY)
            .cast(Float(F64), Angle(bw(8)))
            .unwrap();
        assert_eq!(res.as_angle(bw(8)).unwrap(), 0)
    }

    // --- Float <-> Float ---

    #[test]
    fn f32_to_f64() {
        let s = Primitive::float(1.23_f32);
        let r = s.cast(Float(F32), Float(F64)).unwrap();
        assert!((r.as_float(F64).unwrap() - 1.23f32 as f64).abs() < 1e-10);
    }

    #[test]
    fn f64_to_f32() {
        let s = Primitive::float(2.5);
        let r = s.cast(Float(F64), Float(F32)).unwrap();
        assert_eq!(r.as_float(F32).unwrap(), 2.5);
    }

    // --- Numeric -> Complex ---

    #[test]
    fn uint_to_c64() {
        let s = Primitive::uint(5_u128);
        let r = s.cast(Uint(bw(8)), Complex(F64)).unwrap();
        let c = r.as_complex(F64).unwrap();
        assert_eq!(c.re, 5.0);
        assert_eq!(c.im, 0.0);
    }

    #[test]
    fn f64_to_c64() {
        let s = Primitive::float(2.5);
        let r = s.cast(Float(F64), Complex(F64)).unwrap();
        let c = r.as_complex(F64).unwrap();
        assert_eq!(c.re, 2.5);
        assert_eq!(c.im, 0.0);
    }

    // --- Complex <-> Complex ---

    #[test]
    fn c32_to_c64() {
        let s = Primitive::complex(1.5, -2.5);
        let r = s.cast(Complex(F32), Complex(F64)).unwrap();
        let c = r.as_complex(F64).unwrap();
        assert!((c.re - 1.5).abs() < 1e-6);
        assert!((c.im - -2.5).abs() < 1e-6);
    }

    #[test]
    fn c64_to_c32() {
        let s = Primitive::complex(1.0, -1.0);
        let r = s.cast(Complex(F64), Complex(F32)).unwrap();
        let c = r.as_complex(F32).unwrap();
        assert_eq!(c.re, 1.0);
        assert_eq!(c.im, -1.0);
    }

    // --- Complex -> non-complex returns None ---

    #[test]
    fn complex_to_float_returns_none() {
        let s = Primitive::complex(1.0, 2.0);
        assert!(s.cast(Complex(F64), Float(F64)).is_err());
    }

    #[test]
    fn complex_to_uint_returns_none() {
        let s = Primitive::complex(1.0, 0.0);
        assert!(s.cast(Complex(F64), Uint(bw(32))).is_err());
    }

    // --- Duration <-> Duration ---

    #[test]
    fn duration_to_duration_is_noop() {
        let s = Primitive::duration(42.0, crate::duration::DurationUnit::Ns);
        let r = s.cast(Duration, Duration).unwrap();
        assert_eq!(r.as_duration().unwrap().value, 42.0);
    }

    #[test]
    fn scalar_angle_round_trips_radians() {
        let angle = Primitive::angle(core::f64::consts::FRAC_PI_2);
        let radians = angle.as_float(F64).unwrap();
        assert!((radians - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    // --- Angle <-> Angle ---

    #[test]
    fn angle_widen() {
        // 0b1010 in 4-bit angle, widened to 8-bit, preserves the value
        let s = Primitive::Angle(0b1010_u128 << 124);
        let r = s.cast(Angle(bw(4)), Angle(bw(8))).unwrap();
        assert_eq!(r.as_angle(bw(8)).unwrap(), 0b1010u128 << 124);
    }

    #[test]
    fn angle_narrow() {
        // 0b10100011 in 8-bit angle, narrowed to 4-bit, truncates to lower 4 bits
        let s = Primitive::Angle(0b10100011_u128 << 120);
        let r = s.cast(Angle(bw(8)), Angle(bw(4))).unwrap();
        assert_eq!(r.as_angle(bw(4)).unwrap(), 0b1010u128 << 124);
    }

    // --- BitReg <-> Uint ---

    #[test]
    fn bitreg_to_uint() {
        let s = Primitive::uint(0b1101_u128);
        let r = s.cast(BitReg(bw(4)), Uint(bw(4))).unwrap();
        assert_eq!(r.as_uint(bw(8)).unwrap(), 0b1101);
    }

    // --- Invalid cross-category ---

    #[test]
    fn duration_to_uint_returns_none() {
        let s = Primitive::duration(100.0, crate::duration::DurationUnit::Ns);
        assert!(s.cast(Duration, Uint(bw(32))).is_err());
    }

    #[test]
    fn angle_to_float_returns_some() {
        let s = Primitive::Angle(42_u128);
        assert!(s.cast(Angle(bw(8)), Float(F64)).is_ok());
    }

    #[test]
    fn angle_to_float_uses_angle_width() {
        let s = Primitive::Angle(0b0100_0000_u128 << 120);
        let r = s.cast(Angle(bw(8)), Float(F64)).unwrap();
        assert!((r.as_float(F64).unwrap() - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn uint_to_duration_returns_none() {
        let s = Primitive::uint(100_u128);
        assert!(s.cast(Uint(bw(32)), Duration).is_err());
    }

    #[test]
    fn radians_to_angle_conversion() {
        let angle = radians_to_angle(core::f64::consts::TAU);
        assert_eq!(angle, 0);
        let angle = radians_to_angle(core::f64::consts::PI);
        assert_eq!(angle, 1u128 << 127); // 0.5 of the full circle, so 2^128 / 2 = 2^127
        let angle = radians_to_angle(core::f64::consts::FRAC_PI_2);
        assert_eq!(angle, 1u128 << 126); // 0.25 of the full circle, so 2^128 / 4 = 2^126
    }
}
