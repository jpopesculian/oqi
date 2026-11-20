use std::fmt;

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
    /// `op` has no definition for operands of these types. `is_func`
    /// distinguishes `sin(x)`-style built-ins from infix/prefix operators.
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
    /// `value` has no representation in `ty`.
    TypeMismatch {
        value: Primitive,
        ty: PrimitiveTy,
    },
    /// An `int`/`uint`/`angle` width outside `1..=max` bits.
    InvalidWidth {
        received: usize,
        max: usize,
    },
    /// An array rank outside `1..=max` dimensions.
    InvalidRank {
        received: usize,
        max: usize,
    },
    /// An array built from the wrong number of elements.
    LengthMismatch {
        expected: usize,
        received: usize,
    },
    ReadOnly,
    BadBorrow,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn unsupported_cast(from: impl Into<ValueTy>, to: impl Into<ValueTy>) -> Self {
        Self::UnsupportedCast {
            from: Box::new(from.into()),
            to: Box::new(to.into()),
        }
    }

    pub(crate) fn unexpected_ty(
        expected: impl Into<ValueTy>,
        received: impl Into<ValueTy>,
    ) -> Self {
        Self::UnexpectedTy {
            expected: Box::new(expected.into()),
            received: Box::new(received.into()),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedCast { from, to } => write!(f, "cannot cast `{from}` to `{to}`"),
            Error::UnexpectedTy { expected, received } => {
                write!(f, "expected `{expected}`, got `{received}`")
            }
            Error::UnsupportedOp { op, is_func, args } => {
                f.write_str("`")?;
                match args.as_slice() {
                    [lhs, rhs] if !is_func => write!(f, "{lhs} {op} {rhs}")?,
                    [arg] if !is_func => write!(f, "{op}{arg}")?,
                    args => {
                        write!(f, "{op}(")?;
                        for (i, arg) in args.iter().enumerate() {
                            if i > 0 {
                                f.write_str(", ")?;
                            }
                            write!(f, "{arg}")?;
                        }
                        f.write_str(")")?;
                    }
                }
                f.write_str("` is not defined")
            }
            Error::IndexOutOfBounds { value, index } => {
                f.write_str("index [")?;
                for (i, idx) in index.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{idx}")?;
                }
                write!(f, "] is out of bounds for `{value}`")
            }
            Error::Overflow => f.write_str("integer overflow"),
            Error::DivideByZero => f.write_str("division by zero"),
            Error::TypeMismatch { value, ty } => {
                write!(f, "`{value}` has no representation as `{ty}`")
            }
            Error::InvalidWidth { received, max } => write!(
                f,
                "a width of {received} bits is not supported (must be between 1 and {max})"
            ),
            Error::InvalidRank { received, max } => write!(
                f,
                "an array of {received} dimensions is not supported (must be between 1 and {max})"
            ),
            Error::LengthMismatch { expected, received } => {
                write!(f, "expected {expected} array elements, got {received}")
            }
            Error::ReadOnly => f.write_str("cannot write through a readonly array reference"),
            Error::BadBorrow => f.write_str("array is already borrowed"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth, iw};
    use crate::{ArrayTy, ashape};

    #[test]
    fn errors_display_in_source_terms() {
        let int8 = ValueTy::int(iw(8));
        let float = ValueTy::float(FloatWidth::F64);
        let array = ValueTy::Array(ArrayTy::new(PrimitiveTy::Int(iw(8)), ashape(vec![2, 3])));
        let cases = [
            (
                Error::unsupported_cast(float, ValueTy::bit()),
                "cannot cast `float[64]` to `bit`",
            ),
            (
                Error::unexpected_ty(int8, float),
                "expected `int[8]`, got `float[64]`",
            ),
            (
                Error::IndexOutOfBounds {
                    value: Box::new(array),
                    index: vec![
                        Index::Item(5),
                        Index::Slice {
                            start: Some(1),
                            step: None,
                            end: Some(3),
                        },
                        Index::Slice {
                            start: None,
                            step: Some(-1),
                            end: None,
                        },
                        Index::Select(vec![0, 2]),
                    ],
                },
                "index [5, 1:3, :-1:, {0, 2}] is out of bounds for `array[int[8], 2, 3]`",
            ),
            (Error::Overflow, "integer overflow"),
            (Error::DivideByZero, "division by zero"),
            (
                Error::TypeMismatch {
                    value: Primitive::int(3),
                    ty: PrimitiveTy::Duration,
                },
                "`3` has no representation as `duration`",
            ),
            (
                Error::InvalidRank {
                    received: 8,
                    max: 7,
                },
                "an array of 8 dimensions is not supported (must be between 1 and 7)",
            ),
            (
                Error::ReadOnly,
                "cannot write through a readonly array reference",
            ),
            (Error::BadBorrow, "array is already borrowed"),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }
}
