//! Classical values and types for OpenQASM 3: the integers, floats,
//! angles, complex numbers, durations, bits and bit registers of the
//! language, and arrays of them, with the language's casts and operators.
//!
//! A [`Value`] is a [`Scalar`], an owned [`Array`] or an [`ArrayRef`]; its
//! [`ValueTy`] describes it. Operators are values too: [`ops::BinOp`] and
//! [`ops::UnOp`] type-check with `return_ty` and evaluate with `apply`.
//!
//! This crate is a building block of [`oqi`](https://docs.rs/oqi), the
//! OpenQASM 3 compiler and simulator, which re-exports it as
//! `oqi::classical`: program inputs, outputs and `extern` arguments are
//! these values.
//!
//! ```
//! use oqi_classical::{FloatWidth, IntWidth, Value, ValueTy, ops::BinOp};
//!
//! let a = Value::int(200, IntWidth::new(32)?);
//! let b = Value::float(0.5, FloatWidth::F64);
//! assert_eq!(BinOp::Mul.return_ty(a.ty(), b.ty())?, ValueTy::float(FloatWidth::F64));
//! assert_eq!(BinOp::Mul.apply(a, b)?.to_string(), "100");
//! # Ok::<(), oqi_classical::Error>(())
//! ```
//!
//! # Casts
//!
//! Every `cast` follows docs/types.rst, "Allowed casts". On a type
//! (`PrimitiveTy`, `ValueTy`, `ArrayTy`, `ArrayRefTy`), `cast(to)` checks
//! that the cast is allowed and returns `to`; on a value (`Scalar`,
//! `Value`, `Array`, `ArrayRef`), `cast(to)` returns the converted value.
//! Operator promotions (`angle + float`, say) are not casts: the operators
//! convert their operands themselves.
//!
//! # Features
//!
//! - `serde`: serialize and deserialize values and types (used by the
//!   bytecode format of `oqi-compile`).
#![warn(missing_docs)]

mod array;
mod array_ref;
mod bitreg;
mod duration;
mod error;
mod index;
pub mod ops;
mod primitive;
mod scalar;
mod shared;
mod value;

pub use num_complex;
pub use turns;

pub use array::{Array, ArrayDim, ArrayShape, ArrayTy, ashape};
pub use array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, RefAccess};
pub use bitreg::BitReg;
pub use duration::{Duration, DurationUnit, DurationUnitParseError};
pub use error::{Error, Result};
pub use index::Index;
pub use primitive::{FloatWidth, IntWidth, Primitive, PrimitiveTy, iw};
pub use scalar::Scalar;
pub use value::{Value, ValueTy};

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use crate::array::{ArrayTy, ashape};
    use crate::array_ref::{ArrayRefShape, RefAccess};
    use crate::primitive::{FloatWidth, Primitive, PrimitiveTy, iw};
    use crate::value::{Value, ValueTy};
    use crate::{Array, BitReg, Duration, DurationUnit, Scalar};

    fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned>(v: &T) -> T {
        let bytes: Vec<u8> = postcard::to_allocvec(v).expect("encode");
        postcard::from_bytes(&bytes).expect("decode")
    }

    fn ty_roundtrip<
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    >(
        v: T,
    ) {
        let v2 = roundtrip(&v);
        assert_eq!(v, v2);
    }

    #[test]
    fn value_ty_roundtrips_each_variant() {
        ty_roundtrip(ValueTy::bit());
        ty_roundtrip(ValueTy::bool());
        ty_roundtrip(ValueTy::int(iw(32)));
        ty_roundtrip(ValueTy::uint(iw(64)));
        ty_roundtrip(ValueTy::float(FloatWidth::F32));
        ty_roundtrip(ValueTy::float(FloatWidth::F64));
        ty_roundtrip(ValueTy::complex(FloatWidth::F64));
        ty_roundtrip(ValueTy::angle(iw(128)));
        ty_roundtrip(ValueTy::duration());
        ty_roundtrip(ValueTy::bitreg(8));
        ty_roundtrip(ValueTy::array(PrimitiveTy::Int(iw(32)), ashape(vec![2, 3])));
        ty_roundtrip(ValueTy::array_ref(
            PrimitiveTy::Float(FloatWidth::F64),
            ArrayRefShape::Fixed(ashape(vec![4])),
            RefAccess::Readonly,
        ));
    }

    fn assert_value_roundtrips_via_string(v: Value) {
        let bytes: Vec<u8> = postcard::to_allocvec(&v).expect("encode");
        let v2: Value = postcard::from_bytes(&bytes).expect("decode");
        // `Value` doesn't impl PartialEq (floats/complex). Display-equality
        // is good enough for the leaf payloads we exercise here.
        assert_eq!(format!("{}", v), format!("{}", v2));
    }

    #[test]
    fn value_scalar_variants_roundtrip() {
        assert_value_roundtrips_via_string(Value::bit(true));
        assert_value_roundtrips_via_string(Value::int(-42, iw(32)));
        assert_value_roundtrips_via_string(Value::uint(0xDEAD_BEEF_u128, iw(64)));
        assert_value_roundtrips_via_string(Value::float(2.5, FloatWidth::F64));
        assert_value_roundtrips_via_string(Value::complex(1.0, 2.0, FloatWidth::F64));
        assert_value_roundtrips_via_string(Value::duration(150.0, DurationUnit::Ns));
        assert_value_roundtrips_via_string(Value::angle(1.5));
        assert_value_roundtrips_via_string(Value::bitreg_u128(0b1010_1100, 8));
    }

    #[test]
    fn value_array_roundtrips() {
        let arr = Array::new_unchecked(
            vec![Primitive::int(1), Primitive::int(2), Primitive::int(3)],
            ArrayTy::new(PrimitiveTy::Int(iw(32)), ashape(vec![3])),
        );
        let v = Value::Array(arr);
        let bytes: Vec<u8> = postcard::to_allocvec(&v).expect("encode");
        let _v2: Value = postcard::from_bytes(&bytes).expect("decode");
    }

    #[test]
    fn value_array_ref_roundtrips_through_shared() {
        let arr = Array::new_unchecked(
            vec![Primitive::float(1.5), Primitive::float(2.5)],
            ArrayTy::new(PrimitiveTy::Float(FloatWidth::F64), ashape(vec![2])),
        );
        let arr_ref = arr.into_ref();
        let v = Value::ArrayRef(arr_ref);
        let bytes: Vec<u8> = postcard::to_allocvec(&v).expect("encode");
        let _v2: Value = postcard::from_bytes(&bytes).expect("decode");
    }

    #[test]
    fn duration_roundtrips() {
        ty_roundtrip(Duration::new(100.0, DurationUnit::Ns));
        ty_roundtrip(Duration::new(1.5, DurationUnit::Ms));
    }

    #[test]
    fn bitreg_roundtrips() {
        let b = BitReg::from(0xCAFE_F00D_u128);
        let bytes: Vec<u8> = postcard::to_allocvec(&b).expect("encode");
        let b2: BitReg = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(format!("{:?}", b), format!("{:?}", b2));
    }

    #[test]
    fn scalar_roundtrips() {
        let s = Scalar::int(42, iw(32));
        let bytes: Vec<u8> = postcard::to_allocvec(&s).expect("encode");
        let s2: Scalar = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(format!("{}", s), format!("{}", s2));
    }
}
