use crate::index::Index;
use crate::primitive::{Primitive, PrimitiveTy};
use crate::value::ValueTy;

#[derive(Debug)]
pub enum Error {
    UnsupportedCast {
        from: Box<ValueTy>,
        to: Box<ValueTy>,
    },
    UnexpectedTy {
        expected: Box<ValueTy>,
        received: Box<ValueTy>,
    },
    UnsupportedOp {
        op: &'static str,
        is_func: bool,
        args: Vec<ValueTy>,
    },
    IndexOutOfBounds {
        value: Box<ValueTy>,
        index: Vec<Index>,
    },
    Overflow,
    DivideByZero,
    TypeMismatch {
        value: Primitive,
        ty: PrimitiveTy,
    },
    BadDimensions {
        received: usize,
        min: usize,
        max: usize,
    },
    ReadOnly,
    BadBorrow,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn unsupported_binop(op: &'static str, lhs: ValueTy, rhs: ValueTy, is_func: bool) -> Self {
        Self::UnsupportedOp {
            op,
            args: vec![lhs, rhs],
            is_func,
        }
    }

    pub fn unsupported_scalar_binop(
        op: &'static str,
        lhs: PrimitiveTy,
        rhs: PrimitiveTy,
        is_func: bool,
    ) -> Self {
        Self::unsupported_binop(op, ValueTy::Scalar(lhs), ValueTy::Scalar(rhs), is_func)
    }

    pub fn unsupported_unop(op: &'static str, arg: ValueTy, is_func: bool) -> Self {
        Self::UnsupportedOp {
            op,
            args: vec![arg],
            is_func,
        }
    }

    pub fn unsupported_scalar_unop(op: &'static str, arg: PrimitiveTy, is_func: bool) -> Self {
        Self::unsupported_unop(op, ValueTy::Scalar(arg), is_func)
    }

    pub fn unsupported_cast(from: ValueTy, to: ValueTy) -> Self {
        Self::UnsupportedCast {
            from: Box::new(from),
            to: Box::new(to),
        }
    }

    pub fn unsupported_scalar_cast(from: PrimitiveTy, to: PrimitiveTy) -> Self {
        Self::unsupported_cast(ValueTy::Scalar(from), ValueTy::Scalar(to))
    }
}
