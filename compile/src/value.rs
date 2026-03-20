use awint::Awi;
use bitvec::vec::BitVec;
use num_complex::Complex;

#[derive(Debug, Clone)]
pub enum FloatValue {
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone)]
pub enum ComplexValue {
    F32(Complex<f32>),
    F64(Complex<f64>),
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    Bool(bool),
    Int(Awi),
    Uint(Awi),
    Float(FloatValue),
    Bitstring(BitVec),
    Angle(Awi),
    Timing(TimingValue),
    Complex(ComplexValue),
}

#[derive(Debug, Clone)]
pub struct TimingValue {
    pub value: TimingNumber,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone)]
pub enum TimingNumber {
    Integer(i64),
    Float(FloatValue),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Dt,
    Ns,
    Us,
    Ms,
    S,
}

impl ConstValue {
    /// Extract a u32 value for use as a type designator (width/size).
    /// Returns `None` if the value is not a non-negative integer that fits in u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            ConstValue::Int(awi) => {
                let bw = awi.bw();
                // Check sign bit — if set, value is negative
                if bw > 0 && awi.msb() {
                    return None;
                }
                // Check if it fits in 32 bits
                if bw > 33 {
                    // Could still fit if upper bits are 0
                    for i in 32..bw {
                        if awi.get(i).unwrap() {
                            return None;
                        }
                    }
                }
                let mut val: u32 = 0;
                for i in 0..bw.min(32) {
                    if awi.get(i).unwrap() {
                        val |= 1 << i;
                    }
                }
                Some(val)
            }
            ConstValue::Uint(awi) => {
                let bw = awi.bw();
                if bw > 32 {
                    for i in 32..bw {
                        if awi.get(i).unwrap() {
                            return None;
                        }
                    }
                }
                let mut val: u32 = 0;
                for i in 0..bw.min(32) {
                    if awi.get(i).unwrap() {
                        val |= 1 << i;
                    }
                }
                Some(val)
            }
            _ => None,
        }
    }
}
