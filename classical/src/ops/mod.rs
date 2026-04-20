mod add;
mod arccos;
mod arcsin;
mod arctan;
mod bitand;
mod bitnot;
mod bitor;
mod bitxor;
mod ceiling;
mod concat;
mod cos;
mod div;
mod eq;
mod exp;
mod floor;
mod gt;
mod gte;
mod imag;
mod log;
mod logand;
mod lognot;
mod logor;
mod lt;
mod lte;
mod mul;
mod neg;
mod neq;
mod popcount;
mod pow;
mod real;
mod rem;
mod rotl;
mod rotr;
mod shl;
mod shr;
mod sin;
mod sizeof;
mod sizeof_dim;
mod sqrt;
mod sub;
mod tan;

pub use add::Add;
pub use arccos::Arccos;
pub use arcsin::Arcsin;
pub use arctan::Arctan;
pub use bitand::BitAnd;
pub use bitnot::BitNot;
pub use bitor::BitOr;
pub use bitxor::BitXor;
pub use ceiling::Ceiling;
pub use concat::Concat;
pub use cos::Cos;
pub use div::Div;
pub use eq::Eq;
pub use exp::Exp;
pub use floor::Floor;
pub use gt::Gt;
pub use gte::Gte;
pub use imag::Imag;
pub use log::Log;
pub use logand::LogAnd;
pub use lognot::LogNot;
pub use logor::LogOr;
pub use lt::Lt;
pub use lte::Lte;
pub use mul::Mul;
pub use neg::Neg;
pub use neq::Neq;
pub use popcount::Popcount;
pub use pow::Pow;
pub use real::Real;
pub use rem::Rem;
pub use rotl::Rotl;
pub use rotr::Rotr;
pub use shl::Shl;
pub use shr::Shr;
pub use sin::Sin;
pub use sizeof::Sizeof;
pub use sizeof_dim::SizeofDim;
pub use sqrt::Sqrt;
pub use sub::Sub;
pub use tan::Tan;

use crate::{
    Error, Result,
    array::{Array, ArrayTy},
    array_ref::{ArrayRef, ArrayRefTy},
    primitive::PrimitiveTy,
    scalar::Scalar,
    value::{Value, ValueTy},
};

pub trait BinOp {
    const NAME: &'static str;
    const IS_FUNC: bool = false;

    // given the inputs, get lhs cast, rhs cast and ouput type
    fn scalar_check(lht: PrimitiveTy, rht: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy, PrimitiveTy)> {
        Err(Error::unsupported_scalar_binop(
            Self::NAME,
            lht,
            rht,
            Self::IS_FUNC,
        ))
    }

    fn scalar_op(lhs: Scalar, rhs: Scalar, _out: PrimitiveTy) -> Result<Scalar> {
        Err(Error::unsupported_scalar_binop(
            Self::NAME,
            lhs.ty(),
            rhs.ty(),
            Self::IS_FUNC,
        ))
    }

    fn arr_arr_check(lhs: ArrayTy, rhs: ArrayTy) -> Result<(ArrayTy, ArrayTy, ArrayTy)> {
        if lhs.shape() != rhs.shape() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                lhs.into(),
                rhs.into(),
                Self::IS_FUNC,
            ));
        }
        let (lhc, rhc, out) = Self::scalar_check(lhs.ty(), rhs.ty())?;
        let lhc = lhs.with_ty(lhc);
        let rhc = rhs.with_ty(rhc);
        let out = lhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn arr_scalar_check(lhs: ArrayTy, rhs: PrimitiveTy) -> Result<(ArrayTy, PrimitiveTy, ArrayTy)> {
        let (lhc, rhc, out) = Self::scalar_check(lhs.ty(), rhs)?;
        let lhc = lhs.with_ty(lhc);
        let out = lhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn scalar_arr_check(lhs: PrimitiveTy, rhs: ArrayTy) -> Result<(PrimitiveTy, ArrayTy, ArrayTy)> {
        let (lhc, rhc, out) = Self::scalar_check(lhs, rhs.ty())?;
        let rhc = rhs.with_ty(rhc);
        let out = rhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn arr_arr_op(lhs: Array, rhs: Array, out: ArrayTy) -> Result<Array> {
        let scalar_out_ty = out.ty();
        let values = lhs
            .scalars()
            .zip(rhs.scalars())
            .map(|(a, b)| Ok(Self::scalar_op(a, b, scalar_out_ty)?.value()))
            .collect::<Result<Vec<_>>>()?;
        Array::new(values, out)
    }

    fn arr_scalar_op(lhs: Array, rhs: Scalar, out: ArrayTy) -> Result<Array> {
        let scalar_out_ty = out.ty();
        let values = lhs
            .scalars()
            .map(|a| Ok(Self::scalar_op(a, rhs, scalar_out_ty)?.value()))
            .collect::<Result<Vec<_>>>()?;
        Array::new(values, out)
    }

    fn scalar_arr_op(lhs: Scalar, rhs: Array, out: ArrayTy) -> Result<Array> {
        let scalar_out_ty = out.ty();
        let values = rhs
            .scalars()
            .map(|b| Ok(Self::scalar_op(lhs, b, scalar_out_ty)?.value()))
            .collect::<Result<Vec<_>>>()?;
        Array::new(values, out)
    }

    fn arr_ref_arr_check(
        lhs: ArrayRefTy,
        rhs: ArrayTy,
    ) -> Result<(ArrayRefTy, ArrayTy, ArrayRefTy)> {
        if lhs.shape() != rhs.shape() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                lhs.into(),
                rhs.into(),
                Self::IS_FUNC,
            ));
        }
        let (lhc, rhc, out) = Self::scalar_check(lhs.ty(), rhs.ty())?;
        let lhc = lhs.with_ty(lhc);
        let rhc = rhs.with_ty(rhc);
        let out = lhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn arr_arr_ref_check(
        lhs: ArrayTy,
        rhs: ArrayRefTy,
    ) -> Result<(ArrayTy, ArrayRefTy, ArrayRefTy)> {
        if lhs.shape() != rhs.shape() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                lhs.into(),
                rhs.into(),
                Self::IS_FUNC,
            ));
        }
        let (lhc, rhc, out) = Self::scalar_check(lhs.ty(), rhs.ty())?;
        let lhc = lhs.with_ty(lhc);
        let rhc = rhs.with_ty(rhc);
        let out = rhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn arr_ref_arr_ref_check(
        lhs: ArrayRefTy,
        rhs: ArrayRefTy,
    ) -> Result<(ArrayRefTy, ArrayRefTy, ArrayRefTy)> {
        if lhs.shape() != rhs.shape() {
            return Err(Error::unsupported_binop(
                Self::NAME,
                lhs.into(),
                rhs.into(),
                Self::IS_FUNC,
            ));
        }
        let (lhc, rhc, out) = Self::scalar_check(lhs.ty(), rhs.ty())?;
        let lhc = lhs.with_ty(lhc);
        let rhc = rhs.with_ty(rhc);
        let out = lhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn arr_ref_scalar_check(
        lhs: ArrayRefTy,
        rhs: PrimitiveTy,
    ) -> Result<(ArrayRefTy, PrimitiveTy, ArrayRefTy)> {
        let (lhc, rhc, out) = Self::scalar_check(lhs.ty(), rhs)?;
        let lhc = lhs.with_ty(lhc);
        let out = lhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn scalar_arr_ref_check(
        lhs: PrimitiveTy,
        rhs: ArrayRefTy,
    ) -> Result<(PrimitiveTy, ArrayRefTy, ArrayRefTy)> {
        let (lhc, rhc, out) = Self::scalar_check(lhs, rhs.ty())?;
        let rhc = rhs.with_ty(rhc);
        let out = rhs.with_ty(out);
        Ok((lhc, rhc, out))
    }

    fn arr_ref_arr_ref_op(lhs: ArrayRef, rhs: ArrayRef, out: ArrayRefTy) -> Result<Array> {
        let lhs = lhs.to_owned()?;
        let rhs = rhs.to_owned()?;
        let out = lhs.ty().with_ty(out.ty());
        Self::arr_arr_op(lhs, rhs, out)
    }

    fn arr_arr_ref_op(lhs: Array, rhs: ArrayRef, out: ArrayRefTy) -> Result<Array> {
        let rhs = rhs.to_owned()?;
        let out = lhs.ty().with_ty(out.ty());
        Self::arr_arr_op(lhs, rhs, out)
    }

    fn arr_ref_arr_op(lhs: ArrayRef, rhs: Array, out: ArrayRefTy) -> Result<Array> {
        let lhs = lhs.to_owned()?;
        let out = lhs.ty().with_ty(out.ty());
        Self::arr_arr_op(lhs, rhs, out)
    }

    fn arr_ref_scalar_op(lhs: ArrayRef, rhs: Scalar, out: ArrayRefTy) -> Result<Array> {
        let lhs = lhs.to_owned()?;
        let out = lhs.ty().with_ty(out.ty());
        Self::arr_scalar_op(lhs, rhs, out)
    }

    fn scalar_arr_ref_op(lhs: Scalar, rhs: ArrayRef, out: ArrayRefTy) -> Result<Array> {
        let rhs = rhs.to_owned()?;
        let out = rhs.ty().with_ty(out.ty());
        Self::scalar_arr_op(lhs, rhs, out)
    }

    fn check(lhs: ValueTy, rhs: ValueTy) -> Result<(ValueTy, ValueTy, ValueTy)> {
        use crate::value::BaseValueTy::*;
        match (lhs, rhs) {
            (Scalar(lht), Scalar(rht)) => Self::scalar_check(lht, rht)
                .map(|(lht, rht, out)| (Scalar(lht), Scalar(rht), Scalar(out))),
            (Array(lh_aty), Array(rh_aty)) => Self::arr_arr_check(lh_aty, rh_aty)
                .map(|(lhc, rhc, out)| (Array(lhc), Array(rhc), Array(out))),
            (Array(lh_aty), Scalar(rh_sty)) => Self::arr_scalar_check(lh_aty, rh_sty)
                .map(|(lhc, rhc, out)| (Array(lhc), Scalar(rhc), Array(out))),
            (Scalar(lh_sty), Array(rh_aty)) => Self::scalar_arr_check(lh_sty, rh_aty)
                .map(|(lhc, rhc, out)| (Scalar(lhc), Array(rhc), Array(out))),
            (ArrayRef(lh_aty), ArrayRef(rh_aty)) => Self::arr_ref_arr_ref_check(lh_aty, rh_aty)
                .map(|(lhc, rhc, out)| (ArrayRef(lhc), ArrayRef(rhc), ArrayRef(out))),
            (ArrayRef(lh_aty), Array(rh_aty)) => Self::arr_ref_arr_check(lh_aty, rh_aty)
                .map(|(lhc, rhc, out)| (ArrayRef(lhc), Array(rhc), ArrayRef(out))),
            (Array(lh_aty), ArrayRef(rh_aty)) => Self::arr_arr_ref_check(lh_aty, rh_aty)
                .map(|(lhc, rhc, out)| (Array(lhc), ArrayRef(rhc), ArrayRef(out))),
            (ArrayRef(lh_aty), Scalar(rh_sty)) => Self::arr_ref_scalar_check(lh_aty, rh_sty)
                .map(|(lhc, rhc, out)| (ArrayRef(lhc), Scalar(rhc), ArrayRef(out))),
            (Scalar(lh_sty), ArrayRef(rh_aty)) => Self::scalar_arr_ref_check(lh_sty, rh_aty)
                .map(|(lhc, rhc, out)| (Scalar(lhc), ArrayRef(rhc), ArrayRef(out))),
        }
    }

    fn op(lhs: Value, rhs: Value, out: ValueTy) -> Result<Value> {
        match (lhs, rhs, out) {
            (Value::Scalar(lh_s), Value::Scalar(rh_s), ValueTy::Scalar(out_ty)) => {
                Self::scalar_op(lh_s, rh_s, out_ty).map(Value::Scalar)
            }
            (Value::Array(lh_a), Value::Array(rh_a), ValueTy::Array(out)) => {
                Self::arr_arr_op(lh_a, rh_a, out).map(Value::Array)
            }
            (Value::Array(lh_a), Value::Scalar(rh_s), ValueTy::Array(out_aty)) => {
                Self::arr_scalar_op(lh_a, rh_s, out_aty).map(Value::Array)
            }
            (Value::Scalar(lh_s), Value::Array(rh_a), ValueTy::Array(out_aty)) => {
                Self::scalar_arr_op(lh_s, rh_a, out_aty).map(Value::Array)
            }
            (Value::Array(lh_a), Value::Array(rh_a), ValueTy::ArrayRef(out)) => {
                let out = lh_a.ty().with_ty(out.ty());
                Self::arr_arr_op(lh_a, rh_a, out).map(Value::Array)
            }
            (Value::Array(lh_a), Value::Scalar(rh_s), ValueTy::ArrayRef(out_aty)) => {
                let out = lh_a.ty().with_ty(out_aty.ty());
                Self::arr_scalar_op(lh_a, rh_s, out).map(Value::Array)
            }
            (Value::Scalar(lh_s), Value::Array(rh_a), ValueTy::ArrayRef(out_aty)) => {
                let out = rh_a.ty().with_ty(out_aty.ty());
                Self::scalar_arr_op(lh_s, rh_a, out).map(Value::Array)
            }
            (Value::ArrayRef(lh_a), Value::ArrayRef(rh_a), ValueTy::ArrayRef(out)) => {
                Self::arr_ref_arr_ref_op(lh_a, rh_a, out).map(Value::Array)
            }
            (Value::ArrayRef(lh_a), Value::Array(rh_a), ValueTy::ArrayRef(out)) => {
                Self::arr_ref_arr_op(lh_a, rh_a, out).map(Value::Array)
            }
            (Value::Array(lh_a), Value::ArrayRef(rh_a), ValueTy::ArrayRef(out)) => {
                Self::arr_arr_ref_op(lh_a, rh_a, out).map(Value::Array)
            }
            (Value::ArrayRef(lh_a), Value::Scalar(rh_s), ValueTy::ArrayRef(out_aty)) => {
                Self::arr_ref_scalar_op(lh_a, rh_s, out_aty).map(Value::Array)
            }
            (Value::Scalar(lh_s), Value::ArrayRef(rh_a), ValueTy::ArrayRef(out_aty)) => {
                Self::scalar_arr_ref_op(lh_s, rh_a, out_aty).map(Value::Array)
            }
            (lhs, _, out) => Err(Error::unsupported_cast(lhs.ty(), out)),
        }
    }

    fn checked_op(lhs: Value, rhs: Value) -> Result<Value> {
        let (lhs_ty, rhs_ty, out_ty) = Self::check(lhs.ty(), rhs.ty())?;
        let lhs = lhs.cast(lhs_ty)?;
        let rhs = rhs.cast(rhs_ty)?;
        Self::op(lhs, rhs, out_ty)
    }

    fn return_ty(lht: ValueTy, rht: ValueTy) -> Result<ValueTy> {
        Self::check(lht, rht).map(|(_, _, out)| out)
    }
}

pub trait UnOp {
    const NAME: &'static str;
    const IS_FUNC: bool = false;

    // given the input, get cast and ouput type
    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)>;
    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar>;
    fn arr_check(arg: ArrayTy) -> Result<(ArrayTy, ArrayTy)> {
        // as of right now, we don't support any array unops
        Err(Error::unsupported_unop(
            Self::NAME,
            arg.ty().into(),
            Self::IS_FUNC,
        ))
    }
    fn arr_ref_check(arg: ArrayRefTy) -> Result<(ArrayTy, ArrayTy)> {
        // as of right now, we don't support any array unops
        Err(Error::unsupported_unop(
            Self::NAME,
            arg.ty().into(),
            Self::IS_FUNC,
        ))
    }
    fn arr_op(arg: Array, out: ArrayTy) -> Result<Array> {
        let scalar_out_ty = out.ty();
        let values = arg
            .scalars()
            .map(|scalar| Ok(Self::scalar_op(scalar, scalar_out_ty)?.value()))
            .collect::<Result<Vec<_>>>()?;
        Array::new(values, out)
    }
    fn arr_ref_op(arg: ArrayRef, out: ArrayTy) -> Result<Array> {
        let scalar_out_ty = out.ty();
        let values = arg
            .array()
            .borrow()?
            .scalars()
            .map(|scalar| Ok(Self::scalar_op(scalar, scalar_out_ty)?.value()))
            .collect::<Result<Vec<_>>>()?;
        Array::new(values, out)
    }

    fn check(arg: ValueTy) -> Result<(ValueTy, ValueTy)> {
        use crate::value::BaseValueTy::*;
        match arg {
            Scalar(ty) => {
                Self::scalar_check(ty).map(|(arg_ty, out_ty)| (Scalar(arg_ty), Scalar(out_ty)))
            }
            Array(aty) => {
                Self::arr_check(aty).map(|(arg_ty, out_ty)| (Array(arg_ty), Array(out_ty)))
            }
            ArrayRef(ar_ty) => {
                Self::arr_ref_check(ar_ty).map(|(arg_ty, out_ty)| (Array(arg_ty), Array(out_ty)))
            }
        }
    }

    fn op(arg: Value, out: ValueTy) -> Result<Value> {
        match (arg, out) {
            (Value::Scalar(arg), ValueTy::Scalar(out)) => {
                Self::scalar_op(arg, out).map(Value::Scalar)
            }
            (Value::Array(arg), ValueTy::Array(out)) => Self::arr_op(arg, out).map(Value::Array),
            (Value::ArrayRef(arg), ValueTy::Array(out)) => {
                Self::arr_ref_op(arg, out).map(Value::Array)
            }
            (arg, out) => Err(Error::unsupported_cast(arg.ty(), out)),
        }
    }

    fn checked_op(arg: Value) -> Result<Value> {
        let (arg_ty, out_ty) = Self::check(arg.ty())?;
        let arg = arg.cast(arg_ty)?;
        Self::op(arg, out_ty)
    }

    fn return_ty(arg: ValueTy) -> Result<ValueTy> {
        Self::check(arg).map(|(_, out)| out)
    }
}

pub(crate) fn unsupported_scalar_binop<O: BinOp>(lhs: PrimitiveTy, rhs: PrimitiveTy) -> Error {
    Error::unsupported_scalar_binop(O::NAME, lhs, rhs, O::IS_FUNC)
}

pub(crate) fn unsupported_scalar_unop<O: UnOp>(arg: PrimitiveTy) -> Error {
    Error::unsupported_scalar_unop(O::NAME, arg, O::IS_FUNC)
}
