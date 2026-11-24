//! `return_ty` and `apply` must agree: whatever type-checks evaluates
//! (to a value of the promised type, or a genuine run-time error such as
//! division by zero), and whatever is rejected by one is rejected by both.

use crate::array::ashape;
use crate::ops::{BinOp, UnOp};
use crate::primitive::{FloatWidth, Primitive, PrimitiveTy, iw};
use crate::scalar::Scalar;
use crate::value::{Value, ValueTy};
use crate::{DurationUnit, Error};

const BIN_OPS: [BinOp; 21] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Mod,
    BinOp::Pow,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::BitXor,
    BinOp::Shl,
    BinOp::Shr,
    BinOp::Rotl,
    BinOp::Rotr,
    BinOp::LogAnd,
    BinOp::LogOr,
    BinOp::Eq,
    BinOp::Neq,
    BinOp::Lt,
    BinOp::Gt,
    BinOp::Lte,
    BinOp::Gte,
];

const UN_OPS: [UnOp; 17] = [
    UnOp::Neg,
    UnOp::BitNot,
    UnOp::LogNot,
    UnOp::Sin,
    UnOp::Cos,
    UnOp::Tan,
    UnOp::Arcsin,
    UnOp::Arccos,
    UnOp::Arctan,
    UnOp::Exp,
    UnOp::Log,
    UnOp::Sqrt,
    UnOp::Ceiling,
    UnOp::Floor,
    UnOp::Popcount,
    UnOp::Real,
    UnOp::Imag,
];

/// One value of every scalar kind, at two widths where a kind has widths.
fn samples() -> Vec<Value> {
    let scalar = |value, ty| Value::Scalar(Scalar::new_unchecked(value, ty));
    vec![
        scalar(Primitive::bit(true), PrimitiveTy::Bit),
        scalar(Primitive::bit(true), PrimitiveTy::Bool),
        Value::bitreg_u128(0b0110, 4),
        Value::bitreg_u128(0b0110_1001, 8),
        Value::uint(3, iw(8)),
        Value::uint(3, iw(16)),
        Value::int(-3, iw(8)),
        Value::int(3, iw(32)),
        Value::float(1.5, FloatWidth::F32),
        Value::float(0.5, FloatWidth::F64),
        Value::complex(1.0, 2.0, FloatWidth::F64),
        Value::duration(10.0, DurationUnit::Ns),
        Value::Scalar(Scalar::new(Primitive::angle(1.0), PrimitiveTy::Angle(iw(8))).unwrap()),
        Value::Scalar(Scalar::new(Primitive::angle(2.0), PrimitiveTy::Angle(iw(16))).unwrap()),
    ]
}

/// A result is "supported" unless the operator itself was rejected.
fn is_unsupported<T>(result: &crate::Result<T>) -> bool {
    matches!(result, Err(Error::UnsupportedOp { .. }))
}

#[test]
fn binary_check_and_evaluation_agree_on_every_scalar_pair() {
    for op in BIN_OPS {
        for lhs in samples() {
            for rhs in samples() {
                let ty = op.return_ty(lhs.ty(), rhs.ty());
                let value = op.apply(lhs.clone(), rhs.clone());
                let context = format!("{} {op} {}", lhs.ty(), rhs.ty());
                match (&ty, &value) {
                    (Ok(ty), Ok(value)) => assert_eq!(*ty, value.ty(), "{context}"),
                    (Ok(_), Err(e)) => assert!(
                        matches!(e, Error::Overflow | Error::DivideByZero),
                        "{context} type-checks but fails to evaluate: {e}"
                    ),
                    (Err(_), _) => assert!(
                        is_unsupported(&ty) && is_unsupported(&value),
                        "{context} is rejected inconsistently: {ty:?} vs {value:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn unary_check_and_evaluation_agree_on_every_scalar() {
    for op in UN_OPS {
        for arg in samples() {
            let ty = op.return_ty(arg.ty());
            let value = op.apply(arg.clone());
            let context = format!("{op} on {}", arg.ty());
            match (&ty, &value) {
                (Ok(ty), Ok(value)) => assert_eq!(*ty, value.ty(), "{context}"),
                (Ok(_), Err(e)) => panic!("{context} type-checks but fails to evaluate: {e}"),
                (Err(_), _) => assert!(
                    is_unsupported(&ty) && is_unsupported(&value),
                    "{context} is rejected inconsistently: {ty:?} vs {value:?}"
                ),
            }
        }
    }
}

#[test]
fn unary_operators_reject_arrays_by_their_full_type() {
    let array = Value::Array(
        crate::Array::new(
            vec![Primitive::float(1.0), Primitive::float(2.0)],
            crate::ArrayTy::new(PrimitiveTy::Float(FloatWidth::F64), ashape(vec![2])),
        )
        .unwrap(),
    );
    for op in UN_OPS {
        let err = op.apply(array.clone()).unwrap_err();
        let Error::UnsupportedOp { args, .. } = err else {
            panic!("expected an unsupported-operator error for {op}");
        };
        assert_eq!(args, vec![array.ty()]);
        assert!(op.return_ty(array.ty()).is_err());
    }
}

#[test]
fn array_references_are_read_once_and_behave_like_the_array() {
    let array = |values: &[i128], shape: Vec<usize>| {
        crate::Array::new(
            values.iter().map(|&v| Primitive::int(v)).collect(),
            crate::ArrayTy::new(PrimitiveTy::Int(iw(32)), ashape(shape)),
        )
        .unwrap()
    };
    let a = array(&[1, 2, 3, 4], vec![2, 2]);
    let b = array(&[5, 6, 7, 8], vec![2, 2]);
    let owned = |op: BinOp| {
        op.apply(Value::Array(a.clone()), Value::Array(b.clone()))
            .unwrap()
            .to_string()
    };
    for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Eq] {
        let refs = [
            (
                Value::ArrayRef(a.clone().into_ref()),
                Value::Array(b.clone()),
            ),
            (
                Value::Array(a.clone()),
                Value::ArrayRef(b.clone().into_ref()),
            ),
            (
                Value::ArrayRef(a.clone().into_ref_mut()),
                Value::ArrayRef(b.clone().into_ref()),
            ),
        ];
        for (lhs, rhs) in refs {
            let result = op.apply(lhs, rhs).unwrap();
            assert!(
                matches!(result, Value::Array(_)),
                "{op} yields an owned array"
            );
            assert_eq!(result.to_string(), owned(op), "{op}");
        }
    }
    // `*` between arrays is the matrix product, references included.
    assert_eq!(owned(BinOp::Mul), "{{19, 22}, {43, 50}}");
}

#[test]
fn a_reference_operand_makes_the_result_type_a_reference() {
    use crate::array::adim;
    use crate::{ArrayRefShape, ArrayRefTy, ArrayTy, RefAccess};
    let elem = PrimitiveTy::Int(iw(8));
    let owned = ValueTy::Array(ArrayTy::new(elem, ashape(vec![3])));
    let fixed = |access| ValueTy::ArrayRef(ArrayRefTy::new(elem, ashape(vec![3]).into(), access));
    let ranked = ValueTy::ArrayRef(ArrayRefTy::new(
        elem,
        ArrayRefShape::Dim(adim(1)),
        RefAccess::Readonly,
    ));

    assert_eq!(BinOp::Add.return_ty(owned, owned).unwrap(), owned);
    assert_eq!(
        BinOp::Add
            .return_ty(fixed(RefAccess::Mutable), owned)
            .unwrap(),
        fixed(RefAccess::Mutable)
    );
    // Against a rank-only reference, the side that knows its sizes wins.
    assert_eq!(
        BinOp::Add.return_ty(ranked, owned).unwrap(),
        fixed(RefAccess::Readonly)
    );
    assert_eq!(BinOp::Add.return_ty(ranked, ranked).unwrap(), ranked);
    // Ranks must still match.
    let matrix = ValueTy::Array(ArrayTy::new(elem, ashape(vec![3, 3])));
    assert!(BinOp::Add.return_ty(ranked, matrix).is_err());
}

#[test]
fn unsupported_operations_display_as_source() {
    let float = ValueTy::float(FloatWidth::F64);
    let show = |e: Error| e.to_string();
    assert_eq!(
        show(UnOp::Sin.return_ty(ValueTy::bool()).unwrap_err()),
        "`sin(bool)` is not defined"
    );
    assert_eq!(
        show(UnOp::LogNot.return_ty(float).unwrap_err()),
        "`!float[64]` is not defined"
    );
    assert_eq!(
        show(
            BinOp::Add
                .return_ty(float, ValueTy::duration())
                .unwrap_err()
        ),
        "`float[64] + duration` is not defined"
    );
    assert_eq!(
        show(BinOp::Rotl.return_ty(float, float).unwrap_err()),
        "`rotl(float[64], float[64])` is not defined"
    );
}
