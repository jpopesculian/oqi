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
mod ty;
mod value;

pub use array::{
    Array, ArrayDim, ArrayShape, ArrayTy, BaseArray, BaseArrayTy, ScalarIter, adim, ashape,
};
pub use array_ref::{ArrayRef, ArrayRefShape, ArrayRefTy, BaseArrayRef, BaseArrayRefTy, RefAccess};
pub use bitreg::BitReg;
pub use duration::{Duration, DurationUnit};
pub use error::{Error, Result};
pub use index::Index;
pub use primitive::{FloatWidth, IntWidth, Primitive, PrimitiveTy, iw};
pub use scalar::{BaseScalar, Scalar};
pub use shared::Shared;
pub use value::{BaseValue, BaseValueTy, Value, ValueTy};

#[cfg(test)]
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
        let b = BitReg::Stack(0xCAFE_F00D);
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
