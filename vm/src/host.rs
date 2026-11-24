//! Host values: the bridge between a front door's native values (CLI
//! strings, JS values, Python objects) and the VM's typed classical
//! [`Value`]s.
//!
//! A front door only maps its native values to and from [`HostValue`];
//! everything type-directed — coercing an input or an extern's return value
//! to its declared type, presenting an output — lives here, so every front
//! door accepts and produces the same things.

use std::collections::HashMap;

use num_complex::Complex64;
use oqi_classical::{
    Array, BitReg, DurationUnit, FloatWidth, Primitive, PrimitiveTy, Scalar, Value, ValueTy,
};
use oqi_compile::bytecode::BcModule;
use oqi_compile::symbol::SymbolId;

use crate::error::{VmError, VmErrorKind, VmResult};
use crate::vm::RunResult;

/// Largest integer an `f64` represents exactly (2^53 − 1).
const MAX_SAFE_INTEGER: f64 = ((1u64 << 53) - 1) as f64;

/// A value as a host language sees it, before (inputs, extern results) or
/// after (outputs, extern arguments) it is typed.
#[derive(Debug, Clone, PartialEq)]
pub enum HostValue {
    Bool(bool),
    Int(i128),
    Uint(u128),
    Float(f64),
    Complex(Complex64),
    Text(String),
    List(Vec<HostValue>),
}

impl HostValue {
    /// How a host receives an extern-call argument. Like the output form
    /// ([`From<&Value>`]) except `angle` is a float in radians and `bit[n]`
    /// an MSB-first "0101" string, so values round-trip with what
    /// [`coerce`] accepts for an extern's return value.
    pub fn extern_arg(value: &Value) -> HostValue {
        if let Value::Scalar(s) = value {
            match s.value() {
                angle @ Primitive::Angle(_) => {
                    if let Some(radians) = angle.as_float(FloatWidth::F64) {
                        return HostValue::Float(radians);
                    }
                }
                Primitive::BitReg(reg) => {
                    if let PrimitiveTy::BitReg(width) = s.ty() {
                        return HostValue::Text(bit_string(reg, width));
                    }
                }
                _ => {}
            }
        }
        HostValue::from(value)
    }

    fn to_i128(&self, what: &str) -> Result<i128, String> {
        match self {
            HostValue::Int(i) => Ok(*i),
            HostValue::Uint(u) => {
                i128::try_from(*u).map_err(|_| format!("{what}: {u} is out of range"))
            }
            HostValue::Float(f) if f.fract() == 0.0 && f.abs() <= MAX_SAFE_INTEGER => {
                Ok(*f as i128)
            }
            HostValue::Text(s) => s
                .parse()
                .map_err(|_| format!("{what}: `{s}` is not an integer")),
            _ => Err(format!("{what} must be an integer or a decimal string")),
        }
    }

    fn to_u128(&self, what: &str) -> Result<u128, String> {
        match self {
            HostValue::Uint(u) => Ok(*u),
            HostValue::Int(i) if *i >= 0 => Ok(*i as u128),
            HostValue::Float(f) if f.fract() == 0.0 && (0.0..=MAX_SAFE_INTEGER).contains(f) => {
                Ok(*f as u128)
            }
            HostValue::Text(s) => s
                .parse()
                .map_err(|_| format!("{what}: `{s}` is not an unsigned integer")),
            _ => Err(format!(
                "{what} must be a non-negative integer or a decimal string"
            )),
        }
    }

    fn to_f64(&self, what: &str) -> Result<f64, String> {
        match self {
            HostValue::Float(f) => Ok(*f),
            HostValue::Int(i) => Ok(*i as f64),
            HostValue::Uint(u) => Ok(*u as f64),
            HostValue::Text(s) => s
                .parse()
                .map_err(|_| format!("{what}: `{s}` is not a float")),
            _ => Err(format!("{what} must be a number or a decimal string")),
        }
    }

    fn to_complex(&self, what: &str) -> Result<Complex64, String> {
        match self {
            HostValue::Complex(c) => Ok(*c),
            HostValue::Text(s) => parse_complex(s)
                .ok_or_else(|| format!("{what}: `{s}` is not a complex number (e.g. `1.5-2im`)")),
            HostValue::Bool(_) | HostValue::List(_) => Err(format!(
                "{what} must be a number or a complex string like `1.5-2im`"
            )),
            real => Ok(Complex64::new(real.to_f64(what)?, 0.0)),
        }
    }

    /// An SI timing literal: a number, then `ns`, `us`, `µs`, `ms` or `s`.
    fn to_duration(&self, what: &str) -> Result<(f64, DurationUnit), String> {
        let err = || format!("{what} must be a duration with an SI unit, like `100ns`");
        let HostValue::Text(s) = self else {
            return Err(err());
        };
        let s = s.trim();
        let unit_start = s.find(|c: char| c.is_alphabetic()).ok_or_else(err)?;
        let (number, unit) = s.split_at(unit_start);
        let value: f64 = number.trim().parse().map_err(|_| err())?;
        let unit: DurationUnit = unit.parse().map_err(|_| err())?;
        Ok((value, unit))
    }

    fn to_bool(&self, what: &str) -> Result<bool, String> {
        match self {
            HostValue::Bool(b) => Ok(*b),
            HostValue::Int(0) | HostValue::Uint(0) => Ok(false),
            HostValue::Int(1) | HostValue::Uint(1) => Ok(true),
            HostValue::Float(f) if *f == 0.0 => Ok(false),
            HostValue::Float(f) if *f == 1.0 => Ok(true),
            HostValue::Text(s) if s == "0" || s == "false" => Ok(false),
            HostValue::Text(s) if s == "1" || s == "true" => Ok(true),
            _ => Err(format!("{what} is not a bit (use 0/1/true/false)")),
        }
    }

    /// "0101"-style string, MSB first, exactly `width` chars of 0/1.
    fn to_bitreg(&self, what: &str, width: u32) -> Result<BitReg, String> {
        let err = || format!("{what} must be a {width}-character string of 0s and 1s");
        let HostValue::Text(s) = self else {
            return Err(err());
        };
        if s.len() != width as usize || !s.bytes().all(|b| matches!(b, b'0' | b'1')) {
            return Err(err());
        }
        let mut reg = BitReg::zeros(width);
        for (i, b) in s.bytes().enumerate() {
            reg.set_bit(width as usize - 1 - i, b == b'1');
        }
        Ok(reg)
    }

    /// Flatten nested lists into their scalar leaves, row-major, so both
    /// `[1, 2, 3, 4]` and `[[1, 2], [3, 4]]` feed a 2×2 array.
    fn leaves<'a>(&'a self, out: &mut Vec<&'a HostValue>) {
        match self {
            HostValue::List(items) => items.iter().for_each(|item| item.leaves(out)),
            scalar => out.push(scalar),
        }
    }
}

/// Parse `a+bim`, `a-bim`, `bim` or `a`, optionally in parentheses and
/// with spaces, as written in OpenQASM (and as complex outputs print).
fn parse_complex(text: &str) -> Option<Complex64> {
    let text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let text = text
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .unwrap_or(&text);
    let Some(body) = text.strip_suffix("im") else {
        return Some(Complex64::new(text.parse().ok()?, 0.0));
    };
    // The imaginary part starts at the first sign that is neither leading
    // nor an exponent's (its own sign may follow: `1+-2im`).
    let bytes = body.as_bytes();
    let split = (1..bytes.len())
        .find(|&i| matches!(bytes[i], b'+' | b'-') && !matches!(bytes[i - 1], b'e' | b'E'));
    let (re, im) = match split {
        Some(i) => (body[..i].parse().ok()?, &body[i..]),
        None => (0.0, body),
    };
    let im = im.strip_prefix('+').unwrap_or(im);
    let im = match im {
        "" | "-" => return None,
        im => im.parse().ok()?,
    };
    Some(Complex64::new(re, im))
}

/// How a host receives a program output: `bit`/`bool` as a boolean, `int`,
/// `uint`, `float` and `complex` natively, everything else (bit registers,
/// angles, durations, arrays) as its OpenQASM text form.
impl From<&Value> for HostValue {
    fn from(value: &Value) -> Self {
        if let Value::Scalar(s) = value {
            match s.value() {
                Primitive::Bit(b) => return HostValue::Bool(*b),
                Primitive::Int(i) => return HostValue::Int(*i),
                Primitive::Uint(u) => return HostValue::Uint(*u),
                Primitive::Float(f) => return HostValue::Float(*f),
                Primitive::Complex(c) => return HostValue::Complex(*c),
                _ => {}
            }
        }
        HostValue::Text(value.to_string())
    }
}

/// Coerce a host value to the declared type `ty`; `what` names the value in
/// error messages. Integers and floats also accept decimal strings; angles
/// are radians (wrapped mod 2π and quantized to the declared width); complex
/// numbers take a host complex, a real, or text like `1.5-2im`; durations
/// take SI timing text like `100ns`; bit registers are MSB-first "0101"
/// strings of the declared width; arrays take a (possibly nested) list
/// whose leaves coerce to the element type.
pub fn coerce(what: &str, raw: &HostValue, ty: ValueTy) -> Result<Value, String> {
    Ok(match ty {
        ValueTy::Scalar(PrimitiveTy::Int(w)) => Value::int(raw.to_i128(what)?, w),
        ValueTy::Scalar(PrimitiveTy::Uint(w)) => Value::uint(raw.to_u128(what)?, w),
        ValueTy::Scalar(PrimitiveTy::Float(w)) => Value::float(raw.to_f64(what)?, w),
        ValueTy::Scalar(PrimitiveTy::Bit | PrimitiveTy::Bool) => Value::bit(raw.to_bool(what)?),
        ValueTy::Scalar(PrimitiveTy::Angle(w)) => {
            Scalar::new(Primitive::float(raw.to_f64(what)?), PrimitiveTy::Angle(w))
                .map_err(|e| format!("{what}: not representable as `{ty}`: {e}"))?
                .into()
        }
        ValueTy::Scalar(PrimitiveTy::BitReg(w)) => Value::bitreg(raw.to_bitreg(what, w)?, w),
        ValueTy::Scalar(PrimitiveTy::Complex(w)) => {
            let c = raw.to_complex(what)?;
            Value::complex(c.re, c.im, w)
        }
        ValueTy::Scalar(PrimitiveTy::Duration) => {
            let (value, unit) = raw.to_duration(what)?;
            Value::duration(value, unit)
        }
        ValueTy::Array(aty) => {
            if !matches!(raw, HostValue::List(_)) {
                return Err(format!("{what} must be a list of `{}` values", aty.ty()));
            }
            let mut leaves = Vec::new();
            raw.leaves(&mut leaves);
            let elem_ty = ValueTy::Scalar(aty.ty());
            let elems = leaves
                .into_iter()
                .enumerate()
                .map(
                    |(i, leaf)| match coerce(&format!("{what}[{i}]"), leaf, elem_ty)? {
                        Value::Scalar(s) => Ok(s.into_value()),
                        _ => unreachable!("a scalar type coerces to a scalar value"),
                    },
                )
                .collect::<Result<Vec<Primitive>, String>>()?;
            // `Array::new` validates the element count against the shape.
            Value::Array(Array::new(elems, aty).map_err(|e| format!("{what}: {e}"))?)
        }
        _ => {
            return Err(format!(
                "{what} has type `{ty}`, which a host value cannot supply"
            ));
        }
    })
}

/// Resolve named host values against `module`'s declared `input`s, coercing
/// each to its declared type. Fails on the first (in the given order) name
/// that isn't a declared input or value that doesn't fit its type.
pub fn coerce_inputs(
    module: &BcModule,
    named: &[(String, HostValue)],
) -> VmResult<HashMap<SymbolId, Value>> {
    let mut inputs = HashMap::new();
    for (name, raw) in named {
        let (sym, ty) = module
            .input(name)
            .ok_or_else(|| VmError::new(VmErrorKind::UnknownInput(name.clone())))?;
        let value = coerce(&format!("input `{name}`"), raw, ty)
            .map_err(|message| VmError::new(VmErrorKind::Type(message)))?;
        inputs.insert(sym, value);
    }
    Ok(inputs)
}

/// A value's histogram bucket key (see [`Histogram::counts`]).
fn label(value: &Value) -> String {
    if let Value::Scalar(s) = value
        && let Primitive::Bit(b) = s.value()
    {
        return (if *b { "1" } else { "0" }).to_string();
    }
    value.to_string().trim_matches('"').to_string()
}

/// How often each distinct value of one output variable was observed across
/// a set of shots.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// The output variable's name.
    pub name: String,
    /// `(label, count)` per distinct value. A label is the value's canonical
    /// text: `bit`/`bool` as `0`/`1`, `bit[n]` as an unquoted MSB-first bit
    /// string, everything else its OpenQASM text form. Sorted by value:
    /// numerically when every label is a plain decimal integer, else
    /// lexicographically — so fixed-width bit strings keep their binary
    /// order.
    pub counts: Vec<(String, u32)>,
}

/// Tally the outputs of `shots` (e.g. from [`Vm::sample`](crate::Vm::sample))
/// into one [`Histogram`] per output variable, in program order.
pub fn histograms(module: &BcModule, shots: &[RunResult]) -> Vec<Histogram> {
    let mut tallies: HashMap<SymbolId, HashMap<String, u32>> = HashMap::new();
    for shot in shots {
        for (sym, value) in &shot.outputs {
            *tallies
                .entry(*sym)
                .or_default()
                .entry(label(value))
                .or_insert(0) += 1;
        }
    }
    module
        .outputs
        .iter()
        .filter_map(|(sym, _)| {
            let mut counts: Vec<(String, u32)> = tallies.remove(sym)?.into_iter().collect();
            let numeric = counts.iter().all(|(label, _)| {
                let digits = label.strip_prefix('-').unwrap_or(label);
                digits.parse::<i128>().is_ok() && (digits.len() == 1 || !digits.starts_with('0'))
            });
            if numeric {
                counts.sort_by_key(|(label, _)| label.parse::<i128>().unwrap());
            } else {
                counts.sort();
            }
            Some(Histogram {
                name: module.symbol(*sym).name.clone(),
                counts,
            })
        })
        .collect()
}

/// Unquoted MSB-first bit string of `reg`'s low `width` bits.
fn bit_string(reg: &BitReg, width: u32) -> String {
    (0..width as usize)
        .rev()
        .map(|i| if reg.get_bit(i) { '1' } else { '0' })
        .collect()
}

#[cfg(test)]
mod tests {
    use oqi_classical::{ashape, iw};

    use super::*;

    fn int8() -> ValueTy {
        ValueTy::int(iw(8))
    }

    #[test]
    fn integers_accept_ints_integral_floats_and_decimal_strings() {
        for raw in [
            HostValue::Int(5),
            HostValue::Uint(5),
            HostValue::Float(5.0),
            HostValue::Text("5".into()),
        ] {
            assert_eq!(coerce("x", &raw, int8()).unwrap().to_string(), "5");
        }
        assert!(coerce("x", &HostValue::Float(5.5), int8()).is_err());
        assert!(coerce("x", &HostValue::Bool(true), int8()).is_err());
        assert!(coerce("x", &HostValue::Int(-1), ValueTy::uint(iw(8))).is_err());
    }

    #[test]
    fn bits_accept_booleans_zero_one_and_their_names() {
        for raw in [
            HostValue::Bool(true),
            HostValue::Int(1),
            HostValue::Float(1.0),
            HostValue::Text("true".into()),
        ] {
            assert_eq!(coerce("b", &raw, ValueTy::bit()).unwrap().to_string(), "1");
        }
        let err = coerce("b", &HostValue::Int(2), ValueTy::bit()).unwrap_err();
        assert_eq!(err, "b is not a bit (use 0/1/true/false)");
    }

    #[test]
    fn bit_registers_and_angles_round_trip_through_extern_args() {
        let reg = coerce("r", &HostValue::Text("0110".into()), ValueTy::bitreg(4)).unwrap();
        assert_eq!(HostValue::extern_arg(&reg), HostValue::Text("0110".into()));
        assert_eq!(label(&reg), "0110");
        assert!(coerce("r", &HostValue::Text("011".into()), ValueTy::bitreg(4)).is_err());

        let angle = coerce("a", &HostValue::Float(1.0), ValueTy::angle(iw(32))).unwrap();
        let HostValue::Float(radians) = HostValue::extern_arg(&angle) else {
            panic!("angle should surface as radians");
        };
        assert!((radians - 1.0).abs() < 1e-6);
    }

    #[test]
    fn arrays_flatten_nested_lists_and_check_the_element_count() {
        let ty = ValueTy::array(PrimitiveTy::Int(iw(8)), ashape(vec![2, 2]));
        let nested = HostValue::List(vec![
            HostValue::List(vec![HostValue::Int(1), HostValue::Int(2)]),
            HostValue::List(vec![HostValue::Int(3), HostValue::Text("4".into())]),
        ]);
        assert!(coerce("a", &nested, ty).is_ok());
        let short = HostValue::List(vec![HostValue::Int(1)]);
        assert!(coerce("a", &short, ty).is_err());
        let err = coerce("a", &HostValue::Int(1), ty).unwrap_err();
        assert!(err.contains("must be a list"), "{err}");
    }

    #[test]
    fn outputs_surface_natively_or_as_text() {
        assert_eq!(HostValue::from(&Value::bit(true)), HostValue::Bool(true));
        assert_eq!(HostValue::from(&Value::int(-3, iw(8))), HostValue::Int(-3));
        assert_eq!(
            HostValue::from(&Value::bitreg_u128(0b101, 3)),
            HostValue::Text("\"101\"".into())
        );
    }

    fn complex64() -> ValueTy {
        ValueTy::complex(FloatWidth::F64)
    }

    #[test]
    fn complex_values_accept_complex_reals_and_text() {
        let c = |re: f64, im: f64| Value::complex(re, im, FloatWidth::F64).to_string();
        for (raw, re, im) in [
            (HostValue::Complex(Complex64::new(1.5, -2.0)), 1.5, -2.0),
            (HostValue::Float(0.5), 0.5, 0.0),
            (HostValue::Int(-3), -3.0, 0.0),
            (HostValue::Text("1+2im".into()), 1.0, 2.0),
            (HostValue::Text(" 1.5 - 2.5 im ".into()), 1.5, -2.5),
            (HostValue::Text("(1+-2im)".into()), 1.0, -2.0),
            (HostValue::Text("-2im".into()), 0.0, -2.0),
            (HostValue::Text("1e-3+1e2im".into()), 1e-3, 1e2),
            (HostValue::Text("4".into()), 4.0, 0.0),
        ] {
            let value = coerce("z", &raw, complex64()).unwrap_or_else(|e| panic!("{raw:?}: {e}"));
            assert_eq!(value.to_string(), c(re, im), "{raw:?}");
        }
        for raw in [
            HostValue::Text("1+2i".into()),
            HostValue::Text("im".into()),
            HostValue::Bool(true),
        ] {
            assert!(coerce("z", &raw, complex64()).is_err(), "{raw:?}");
        }
    }

    #[test]
    fn durations_accept_si_timing_text() {
        let ty = ValueTy::duration();
        for (raw, value) in [
            ("100ns", Value::duration(100.0, DurationUnit::Ns)),
            ("1.5 us", Value::duration(1.5, DurationUnit::Us)),
            ("2\u{b5}s", Value::duration(2.0, DurationUnit::Us)),
            ("3ms", Value::duration(3.0, DurationUnit::Ms)),
            ("0s", Value::duration(0.0, DurationUnit::S)),
        ] {
            let got = coerce("d", &HostValue::Text(raw.into()), ty).unwrap();
            assert_eq!(got.to_string(), value.to_string(), "{raw}");
        }
        for raw in [
            HostValue::Text("10dt".into()),
            HostValue::Text("10".into()),
            HostValue::Text("ns".into()),
            HostValue::Float(1.0),
        ] {
            assert!(coerce("d", &raw, ty).is_err(), "{raw:?}");
        }
    }
}
