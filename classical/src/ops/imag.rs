use super::UnOp;
use crate::Primitive;
use crate::error::{Error, Result};
use crate::primitive::{FloatWidth, PrimitiveTy, promote_arithmetic};
use crate::scalar::Scalar;
use crate::value::Value;

pub struct Imag;

impl UnOp for Imag {
    const NAME: &'static str = "imag";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        match promote_arithmetic(arg, PrimitiveTy::Complex(arg.fw().unwrap_or(FloatWidth::F64))) {
            Some(PrimitiveTy::Complex(fw)) => Ok((PrimitiveTy::Complex(fw), PrimitiveTy::Float(fw))),
            _ => Err(Error::unsupported_unop(
                Self::NAME,
                arg.into(),
                Self::IS_FUNC,
            )),
        }
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        match arg.value() {
            Primitive::Complex(c) => Scalar::new(Primitive::Float(c.im), out),
            _ => Err(Error::unsupported_unop(
                Self::NAME,
                arg.ty().into(),
                Self::IS_FUNC,
            )),
        }
    }
}

impl Value {
    pub fn imag_(self) -> Result<Self> {
        Imag::checked_op(self)
    }
}
