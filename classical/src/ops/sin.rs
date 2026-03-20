use super::{UnOp, unsupported_scalar_unop};
use crate::primitive::{FloatWidth, PrimitiveTy};
use crate::scalar::Scalar;
use crate::value::Value;
use crate::{Primitive, Result};

pub struct Sin;

impl UnOp for Sin {
    const NAME: &'static str = "sin";
    const IS_FUNC: bool = true;

    fn scalar_check(arg: PrimitiveTy) -> Result<(PrimitiveTy, PrimitiveTy)> {
        Ok(match arg {
            PrimitiveTy::Float(w) => (PrimitiveTy::Float(w), PrimitiveTy::Float(w)),
            PrimitiveTy::Angle(_) => (
                PrimitiveTy::Float(FloatWidth::F64),
                PrimitiveTy::Float(FloatWidth::F64),
            ),
            _ => return Err(unsupported_scalar_unop::<Self>(arg)),
        })
    }

    fn scalar_op(arg: Scalar, out: PrimitiveTy) -> Result<Scalar> {
        match arg.value() {
            Primitive::Float(v) => Scalar::new(Primitive::Float(v.sin()), out),
            _ => Err(unsupported_scalar_unop::<Self>(arg.ty())),
        }
    }
}

impl Value {
    pub fn sin_(self) -> Result<Self> {
        Sin::checked_op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{FloatWidth::*, PrimitiveTy::*, bw};
    use crate::scalar::Scalar;

    #[test]
    fn sin_zero() {
        let r = Value::Scalar(Scalar::new_unchecked(Primitive::float(0.0), Float(F64)))
            .sin_()
            .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!(s.value().as_float(F64).unwrap().abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sin_pi_half() {
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::float(core::f64::consts::FRAC_PI_2),
            Float(F64),
        ))
        .sin_()
        .unwrap();
        match r {
            Value::Scalar(s) => {
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sin_angle_quarter_turn() {
        // 8-bit angle: 64/256 = quarter turn = π/2
        let r = Value::Scalar(Scalar::new_unchecked(
            Primitive::angle(core::f64::consts::FRAC_PI_2),
            Angle(bw(8)),
        ))
        .sin_()
        .unwrap();
        println!("{:?}", Primitive::angle(core::f64::consts::FRAC_PI_2));
        println!(
            "{:?}",
            Primitive::angle(core::f64::consts::FRAC_PI_2).as_angle(bw(8))
        );
        println!("{:?}", r);
        match r {
            Value::Scalar(s) => {
                assert!(matches!(s.ty(), Float(F64)));
                assert!((s.value().as_float(F64).unwrap() - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn sin_int_err() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(Primitive::int(0), Int(bw(8))))
                .sin_()
                .is_err()
        )
    }

    #[test]
    fn sin_bit_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(Primitive::bit(true), Bit))
                .sin_()
                .is_err()
        );
    }

    #[test]
    fn sin_duration_returns_none() {
        assert!(
            Value::Scalar(Scalar::new_unchecked(
                Primitive::duration(1.0, crate::DurationUnit::Ns),
                Duration,
            ))
            .sin_()
            .is_err()
        );
    }
}
