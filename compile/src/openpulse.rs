use std::fmt;

use crate::classical::{FloatWidth, IntWidth};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTy {
    Port,
    Frame(FloatWidth, IntWidth),
    Waveform(FloatWidth),
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
