//! Classical operators as values.
//!
//! [`BinOp`] and [`UnOp`] name an operation; `return_ty` type-checks it and
//! `apply` evaluates it. Each operator has one *signature* (which operand
//! types it is defined on, what they are converted to, and the result
//! type). The checker is that signature; the evaluator consults it before
//! computing, and `tests/agreement.rs` checks that every signature has an
//! evaluation, so the two cannot disagree about what is supported.
//!
//! Binary operators lift over arrays: element-wise for array ∘ array and
//! array ∘ scalar, except `*` between two arrays, which is a matrix product.
//! Unary operators are scalar-only.

mod arith;
mod bits;
mod cmp;
mod math;
mod sizeof;
#[cfg(test)]
mod tests;

use std::fmt;

pub use sizeof::{sizeof, sizeof_ty};

use crate::{
    Error, Result,
    array::{Array, ArrayDim, ArrayTy},
    array_ref::{ArrayRefShape, ArrayRefTy, RefAccess},
    primitive::{Primitive, PrimitiveTy},
    scalar::Scalar,
    value::{Value, ValueTy},
};

/// A binary operator, or a two-argument built-in function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinOp {
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*` (on arrays, the matrix product).
    Mul,
    /// `/`.
    Div,
    /// `%`, and the built-in `mod(a, b)`.
    Mod,
    /// `**`.
    Pow,
    /// `&`.
    BitAnd,
    /// `|`.
    BitOr,
    /// `^`.
    BitXor,
    /// `<<`.
    Shl,
    /// `>>`.
    Shr,
    /// The built-in `rotl(value, distance)`.
    Rotl,
    /// The built-in `rotr(value, distance)`.
    Rotr,
    /// `&&`.
    LogAnd,
    /// `||`.
    LogOr,
    /// `==`.
    Eq,
    /// `!=`.
    Neq,
    /// `<`.
    Lt,
    /// `>`.
    Gt,
    /// `<=`.
    Lte,
    /// `>=`.
    Gte,
}

/// A unary operator, or a one-argument built-in function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnOp {
    /// `-x`.
    Neg,
    /// `~x`.
    BitNot,
    /// `!x`.
    LogNot,
    /// `sin(x)`.
    Sin,
    /// `cos(x)`.
    Cos,
    /// `tan(x)`.
    Tan,
    /// `arcsin(x)`.
    Arcsin,
    /// `arccos(x)`.
    Arccos,
    /// `arctan(x)`.
    Arctan,
    /// `exp(x)`.
    Exp,
    /// `log(x)`, the natural logarithm.
    Log,
    /// `sqrt(x)`.
    Sqrt,
    /// `ceiling(x)`.
    Ceiling,
    /// `floor(x)`.
    Floor,
    /// `popcount(x)`, the number of set bits.
    Popcount,
    /// `real(z)`.
    Real,
    /// `imag(z)`.
    Imag,
}

/// An operation an [`Error::UnsupportedOp`] reports: what was applied to
/// operands it is not defined for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Operator {
    /// A binary operator or two-argument built-in.
    Binary(BinOp),
    /// A unary operator or one-argument built-in.
    Unary(UnOp),
    /// The built-in `sizeof`.
    Sizeof,
}

impl Operator {
    /// The operation as spelled in source: `+`, `sin`, `sizeof`, ….
    pub const fn name(self) -> &'static str {
        match self {
            Operator::Binary(op) => op.name(),
            Operator::Unary(op) => op.name(),
            Operator::Sizeof => "sizeof",
        }
    }

    /// Whether the operation is written as a call (`sin(x)`) rather than
    /// with an operator symbol (`a + b`, `-x`).
    pub(crate) const fn is_func(self) -> bool {
        match self {
            Operator::Binary(op) => op.is_func(),
            Operator::Unary(op) => op.is_func(),
            Operator::Sizeof => true,
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Operand conversions and result type of a binary operator on scalars.
type BinSignature = (PrimitiveTy, PrimitiveTy, PrimitiveTy);
/// Operand conversion and result type of a unary operator.
type UnSignature = (PrimitiveTy, PrimitiveTy);

impl BinOp {
    /// The operator as spelled in source: `+`, `<<`, `rotl`, ….
    pub const fn name(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Pow => "**",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Rotl => "rotl",
            BinOp::Rotr => "rotr",
            BinOp::LogAnd => "&&",
            BinOp::LogOr => "||",
            BinOp::Eq => "==",
            BinOp::Neq => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Lte => "<=",
            BinOp::Gte => ">=",
        }
    }

    /// Whether the operation is written as a call (`rotl(a, b)`) rather
    /// than infix (`a + b`).
    const fn is_func(self) -> bool {
        matches!(self, BinOp::Rotl | BinOp::Rotr)
    }

    /// The type of `lhs op rhs`, or an error if the operator is not
    /// defined for those operands.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOp`] for operand types the operator is not defined
    /// for.
    pub fn return_ty(self, lhs: ValueTy, rhs: ValueTy) -> Result<ValueTy> {
        Ok(self.check(lhs, rhs)?.2)
    }

    /// Evaluate `lhs op rhs`.
    ///
    /// # Errors
    ///
    /// As [`return_ty`](Self::return_ty), plus run-time failures:
    /// [`Error::DivideByZero`], [`Error::Overflow`] for a signed integer
    /// result that does not fit its width, [`Error::BadBorrow`] for an array
    /// reference in use.
    pub fn apply(self, lhs: Value, rhs: Value) -> Result<Value> {
        // References are read once, up front: every array below is owned.
        let (lhs, rhs) = (lhs.into_owned()?, rhs.into_owned()?);
        let (lhs_elem, rhs_elem, out) = self.check(lhs.ty(), rhs.ty())?;
        let (lhs, rhs) = (convert(lhs, lhs_elem)?, convert(rhs, rhs_elem)?);
        let elementwise = |out: ArrayTy, values: Result<Vec<Primitive>>| {
            Array::new(values?, out).map(Value::Array)
        };
        match (lhs, rhs, out) {
            (Value::Scalar(l), Value::Scalar(r), ValueTy::Scalar(out)) => {
                self.scalar(&l, &r, out).map(Value::Scalar)
            }
            (Value::Array(l), Value::Array(r), ValueTy::Array(out)) if self == BinOp::Mul => {
                arith::matmul(&l, &r, out).map(Value::Array)
            }
            (Value::Array(l), Value::Array(r), ValueTy::Array(out)) => elementwise(
                out,
                l.scalars()
                    .zip(r.scalars())
                    .map(|(l, r)| self.eval(&l, &r, out.element_ty()))
                    .collect(),
            ),
            (Value::Array(l), Value::Scalar(r), ValueTy::Array(out)) => elementwise(
                out,
                l.scalars()
                    .map(|l| self.eval(&l, &r, out.element_ty()))
                    .collect(),
            ),
            (Value::Scalar(l), Value::Array(r), ValueTy::Array(out)) => elementwise(
                out,
                r.scalars()
                    .map(|r| self.eval(&l, &r, out.element_ty()))
                    .collect(),
            ),
            (lhs, rhs, _) => Err(self.unsupported(lhs.ty(), rhs.ty())),
        }
    }

    /// The element types the operands are converted to, and the result
    /// type. A result involving an array reference is itself typed as a
    /// reference: it is the only array type that can carry a `#dim` shape.
    fn check(self, lhs: ValueTy, rhs: ValueTy) -> Result<(PrimitiveTy, PrimitiveTy, ValueTy)> {
        let unsupported = || self.unsupported(lhs, rhs);
        let ((lhs_elem, lhs_shape), (rhs_elem, rhs_shape)) = (parts(lhs), parts(rhs));
        let (lhs_elem, rhs_elem, out) =
            self.signature(lhs_elem, rhs_elem).ok_or_else(unsupported)?;
        let shape = match (lhs_shape, rhs_shape) {
            (None, None) => return Ok((lhs_elem, rhs_elem, ValueTy::Scalar(out))),
            (Some(shape), None) | (None, Some(shape)) => shape,
            (Some(l), Some(r)) => self.array_shape(l, r).ok_or_else(unsupported)?,
        };
        let access = [lhs, rhs].into_iter().find_map(|ty| match ty {
            ValueTy::ArrayRef(ty) => Some(ty.access()),
            _ => None,
        });
        let out = match (shape, access) {
            (ArrayRefShape::Fixed(shape), None) => ValueTy::Array(ArrayTy::new(out, shape)),
            (shape, access) => ValueTy::ArrayRef(ArrayRefTy::new(
                out,
                shape,
                access.unwrap_or(RefAccess::Readonly),
            )),
        };
        Ok((lhs_elem, rhs_elem, out))
    }

    /// The result shape of an array ∘ array operation.
    fn array_shape(self, lhs: ArrayRefShape, rhs: ArrayRefShape) -> Option<ArrayRefShape> {
        use ArrayRefShape::{Dim, Fixed};
        match (self, lhs, rhs) {
            (BinOp::Mul, Fixed(l), Fixed(r)) => arith::matmul_shape(l, r).map(Fixed),
            (BinOp::Mul, l, r) => {
                let rank = arith::matmul_rank(l.dim().get(), r.dim().get());
                ArrayDim::new(rank).ok().map(Dim)
            }
            // Element-wise: prefer whichever side knows its sizes.
            (_, Dim(_), Fixed(_)) => lhs.is_compatible(&rhs).then_some(rhs),
            _ => lhs.is_compatible(&rhs).then_some(lhs),
        }
    }

    /// How the operator applies to scalar operands, or `None` if it is not
    /// defined for them. The single source of truth for what is supported.
    fn signature(self, lhs: PrimitiveTy, rhs: PrimitiveTy) -> Option<BinSignature> {
        use BinOp::*;
        match self {
            Add | Sub | Mul | Div | Mod | Pow => arith::signature(self, lhs, rhs),
            BitAnd | BitOr | BitXor | Shl | Shr | Rotl | Rotr => bits::signature(self, lhs, rhs),
            LogAnd | LogOr | Eq | Neq | Lt | Gt | Lte | Gte => cmp::signature(self, lhs, rhs),
        }
    }

    /// Evaluate on operands already converted per [`signature`](Self::signature).
    fn eval(self, lhs: &Scalar, rhs: &Scalar, out: PrimitiveTy) -> Result<Primitive> {
        use BinOp::*;
        match self {
            Add | Sub | Mul | Div | Mod | Pow => arith::eval(self, lhs, rhs, out),
            BitAnd | BitOr | BitXor | Shl | Shr | Rotl | Rotr => bits::eval(self, lhs, rhs),
            LogAnd | LogOr | Eq | Neq | Lt | Gt | Lte | Gte => cmp::eval(self, lhs, rhs),
        }
    }

    fn scalar(self, lhs: &Scalar, rhs: &Scalar, out: PrimitiveTy) -> Result<Scalar> {
        Scalar::new(self.eval(lhs, rhs, out)?, out)
    }

    fn unsupported(self, lhs: impl Into<ValueTy>, rhs: impl Into<ValueTy>) -> Error {
        Error::UnsupportedOp {
            op: Operator::Binary(self),
            args: vec![lhs.into(), rhs.into()],
        }
    }
}

impl UnOp {
    /// The operator as spelled in source: `-`, `~`, `sin`, ….
    pub const fn name(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::BitNot => "~",
            UnOp::LogNot => "!",
            UnOp::Sin => "sin",
            UnOp::Cos => "cos",
            UnOp::Tan => "tan",
            UnOp::Arcsin => "arcsin",
            UnOp::Arccos => "arccos",
            UnOp::Arctan => "arctan",
            UnOp::Exp => "exp",
            UnOp::Log => "log",
            UnOp::Sqrt => "sqrt",
            UnOp::Ceiling => "ceiling",
            UnOp::Floor => "floor",
            UnOp::Popcount => "popcount",
            UnOp::Real => "real",
            UnOp::Imag => "imag",
        }
    }

    /// Whether the operation is written as a call (`sin(x)`) rather than
    /// prefix (`-x`).
    const fn is_func(self) -> bool {
        !matches!(self, UnOp::Neg | UnOp::BitNot | UnOp::LogNot)
    }

    /// The type of `op arg`, or an error if the operator is not defined
    /// for that operand.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOp`] for an operand type the operator is not
    /// defined for.
    pub fn return_ty(self, arg: ValueTy) -> Result<ValueTy> {
        self.check(arg).map(|(_, out)| ValueTy::Scalar(out))
    }

    /// Evaluate `op arg`.
    ///
    /// # Errors
    ///
    /// As [`return_ty`](Self::return_ty), and [`Error::Overflow`] for a
    /// negation that does not fit its width.
    pub fn apply(self, arg: Value) -> Result<Value> {
        let (operand, out) = self.check(arg.ty())?;
        let Value::Scalar(arg) = arg else {
            unreachable!("unary operators only type-check on scalars");
        };
        let arg = Scalar::new(arg.into_value(), operand)?;
        Scalar::new(self.eval(&arg)?, out).map(Value::Scalar)
    }

    fn check(self, arg: ValueTy) -> Result<UnSignature> {
        match arg {
            ValueTy::Scalar(ty) => self.signature(ty),
            ValueTy::Array(_) | ValueTy::ArrayRef(_) => None,
        }
        .ok_or_else(|| self.unsupported(arg))
    }

    /// How the operator applies to its operand, or `None` if it is not
    /// defined for it. The single source of truth for what is supported.
    fn signature(self, arg: PrimitiveTy) -> Option<UnSignature> {
        match self {
            UnOp::Neg => arith::neg_signature(arg),
            UnOp::BitNot | UnOp::Popcount => bits::unary_signature(self, arg),
            UnOp::LogNot => cmp::not_signature(arg),
            _ => math::signature(self, arg),
        }
    }

    /// Evaluate on an operand already converted per [`signature`](Self::signature).
    fn eval(self, arg: &Scalar) -> Result<Primitive> {
        match self {
            UnOp::Neg => arith::neg(arg),
            UnOp::BitNot | UnOp::Popcount => bits::unary_eval(self, arg),
            UnOp::LogNot => cmp::not(arg),
            _ => math::eval(self, arg),
        }
    }

    fn unsupported(self, arg: impl Into<ValueTy>) -> Error {
        Error::UnsupportedOp {
            op: Operator::Unary(self),
            args: vec![arg.into()],
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An operand type as its element type and, for an array or array
/// reference, its shape.
fn parts(ty: ValueTy) -> (PrimitiveTy, Option<ArrayRefShape>) {
    match ty {
        ValueTy::Scalar(elem) => (elem, None),
        ValueTy::Array(array) => (array.element_ty(), Some(array.shape().into())),
        ValueTy::ArrayRef(array) => (array.element_ty(), Some(array.shape())),
    }
}

/// Convert the element(s) of an owned operand to `elem`. This is the
/// conversion an operator's signature asks for (a promotion such as
/// `angle -> float`), not a user-level cast, so the cast table does not apply.
fn convert(value: Value, elem: PrimitiveTy) -> Result<Value> {
    match value {
        Value::Scalar(scalar) => Scalar::new(scalar.into_value(), elem).map(Value::Scalar),
        Value::Array(array) => {
            let ty = array.ty().with_ty(elem);
            Array::new(array.into_values(), ty).map(Value::Array)
        }
        Value::ArrayRef(array) => convert(Value::Array(array.to_array()?), elem),
    }
}
