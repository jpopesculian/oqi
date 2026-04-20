use std::fmt;

use num_complex::Complex64;
use oqi_classical::{BaseValueTy, BitWidth, FloatWidth};
use turns::Angle128;

#[derive(Debug, Clone)]
pub struct Frame<P> {
    pub port: P,
    pub freq: f64,
    pub phase: Angle128,
}

#[derive(Debug, Clone, Default)]
pub struct Waveform(pub Vec<Complex64>);

#[derive(Debug, Clone)]
pub enum Primitive<P> {
    Port(P),
    Frame(Frame<P>),
    Waveform(Waveform),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTy {
    Port,
    Frame(FloatWidth, BitWidth),
    Waveform(FloatWidth),
}

pub type ValueTy = BaseValueTy<PrimitiveTy>;

impl PrimitiveTy {
    #[inline]
    pub const fn port() -> Self {
        Self::Port
    }

    #[inline]
    pub const fn frame(fw: FloatWidth, bw: BitWidth) -> Self {
        Self::Frame(fw, bw)
    }

    #[inline]
    pub const fn waveform(fw: FloatWidth) -> Self {
        Self::Waveform(fw)
    }
}

impl fmt::Display for PrimitiveTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveTy::Port => write!(f, "port"),
            PrimitiveTy::Frame(fw, bw) => write!(f, "frame[{}, {}]", fw.get(), bw.get()),
            PrimitiveTy::Waveform(fw) => write!(f, "waveform[{}]", fw.get()),
        }
    }
}

