//! Tests of the operators in `ops::sizeof`, one module per operator.

mod whole {

    use crate::array_ref::RefAccess;
    use crate::index::Index;
    use crate::ops::tests::util::*;
    use crate::primitive::{PrimitiveTy::*, iw};

    #[test]
    fn sizeof_scalar_returns_bitwidth() {
        let result = sizeof(
            Value::Scalar(Scalar::new_unchecked(Primitive::uint(5_u128), Uint(iw(16)))).ty(),
            None,
        )
        .unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), Uint(IntWidth::B64));
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(16));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_array_returns_first_dimension() {
        let result = sizeof(u_array(&[1, 2, 3, 4, 5, 6], 8, vec![2, 3]).ty(), None).unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(2));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_fixed_array_ref_returns_first_dimension() {
        let array_ref = into_ref(u_array(&[1, 2, 3, 4], 8, vec![4]))
            .get(&[Index::Select(vec![0, 2])])
            .unwrap();
        assert!(matches!(array_ref, Value::ArrayRef(_)));

        let result = sizeof(array_ref.ty(), None).unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(2));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_rank_only_array_ref_is_accepted() {
        // An unspecified-length array reference (`array[T, #dim = N]`) has a
        // known rank; `sizeof` type-checks to `uint` and is resolved at run
        // time against the concrete array passed in.
        let array_ref_ty = ArrayRefTy::new(
            Uint(iw(8)),
            crate::ArrayRefShape::Dim(crate::array::adim(2)),
            RefAccess::Readonly,
        );

        assert_eq!(
            sizeof_ty(ValueTy::ArrayRef(array_ref_ty), None).unwrap(),
            ValueTy::uint(IntWidth::B64)
        );
    }

    #[test]
    fn sizeof_bool_is_rejected() {
        let value = Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bool));

        assert!(sizeof(value.ty(), None).is_err());
    }
}

mod dimension {

    use crate::index::Index;
    use crate::ops::tests::util::*;
    use crate::primitive::{PrimitiveTy::*, iw};

    fn int(value: i128) -> Value {
        Value::Scalar(Scalar::new_unchecked(
            Primitive::int(value),
            PrimitiveTy::Int(IntWidth::B64),
        ))
    }

    #[test]
    fn sizeof_dim_array_returns_requested_dimension() {
        let result = sizeof(
            u_array(&[1, 2, 3, 4, 5, 6], 8, vec![2, 3]).ty(),
            Some(int(1)),
        )
        .unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.ty(), Uint(IntWidth::B64));
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(3));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_dim_array_returns_element_bitwidth_on_last_dimension() {
        let result = sizeof(u_array(&[1, 2, 3, 4], 8, vec![2, 2]).ty(), Some(int(2))).unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(8));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_dim_fixed_array_ref_returns_requested_dimension() {
        let array_ref = into_ref(u_array(&[1, 2, 3, 4, 5, 6], 8, vec![2, 3]))
            .get(&[Index::Item(1)])
            .unwrap();
        assert!(matches!(array_ref, Value::ArrayRef(_)));

        let result = sizeof(array_ref.ty(), Some(int(0))).unwrap();

        match result {
            Value::Scalar(scalar) => {
                assert_eq!(scalar.value().as_uint(IntWidth::B64), Some(3));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn sizeof_dim_rejects_negative_dimension() {
        let array = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);

        assert!(sizeof(array.ty(), Some(int(-1))).is_err());
    }

    #[test]
    fn sizeof_dim_rejects_out_of_bounds_dimension() {
        let array = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);

        assert!(sizeof(array.ty(), Some(int(3))).is_err());
    }

    #[test]
    fn sizeof_dim_rejects_unsupported_dimension_type() {
        let array = u_array(&[1, 2, 3, 4], 8, vec![2, 2]);
        let angle = Value::Scalar(Scalar::new_unchecked(Primitive::uint(1_u128), Angle(iw(8))));

        assert!(sizeof(array.ty(), Some(angle)).is_err());
    }

    #[test]
    fn sizeof_dim_rank_only_array_ref_type_is_accepted() {
        // An unspecified-length array reference type-checks to `uint`; the
        // concrete dimension is read at run time from the array passed in.
        use crate::array_ref::RefAccess;
        let array_ref_ty = ArrayRefTy::new(
            Uint(iw(8)),
            crate::ArrayRefShape::Dim(crate::array::adim(3)),
            RefAccess::Readonly,
        );

        assert_eq!(
            sizeof_ty(
                ValueTy::ArrayRef(array_ref_ty),
                Some(ValueTy::int(IntWidth::B64))
            )
            .unwrap(),
            ValueTy::uint(IntWidth::B64)
        );
    }
}
