use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;

#[derive(Clone, Copy, Debug)]
pub struct Duration {
    pub value: f64,
    pub unit: DurationUnit,
}

impl Duration {
    #[inline]
    pub const fn new(value: f64, unit: DurationUnit) -> Self {
        Self { value, unit }
    }

    #[inline]
    pub fn to_unit(self, unit: DurationUnit) -> Self {
        let multiplier = self.unit.multiplier() / unit.multiplier();
        Self {
            value: self.value * multiplier,
            unit,
        }
    }
}

impl Add for Duration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let rhs_converted = rhs.to_unit(self.unit);
        Self {
            value: self.value + rhs_converted.value,
            unit: self.unit,
        }
    }
}

impl Sub for Duration {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let rhs_converted = rhs.to_unit(self.unit);
        Self {
            value: self.value - rhs_converted.value,
            unit: self.unit,
        }
    }
}

impl Mul<f64> for Duration {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value * rhs,
            unit: self.unit,
        }
    }
}

impl Mul<Duration> for f64 {
    type Output = Duration;

    fn mul(self, rhs: Duration) -> Self::Output {
        Duration {
            value: self * rhs.value,
            unit: rhs.unit,
        }
    }
}

impl Div for Duration {
    type Output = f64;

    fn div(self, rhs: Self) -> Self::Output {
        let rhs_converted = rhs.to_unit(self.unit);
        self.value / rhs_converted.value
    }
}

impl Div<f64> for Duration {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value / rhs,
            unit: self.unit,
        }
    }
}

impl Rem for Duration {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let rhs_converted = rhs.to_unit(self.unit);
        Self {
            value: self.value % rhs_converted.value,
            unit: self.unit,
        }
    }
}

impl Rem<f64> for Duration {
    type Output = Self;

    fn rem(self, rhs: f64) -> Self::Output {
        Self {
            value: self.value % rhs,
            unit: self.unit,
        }
    }
}

impl Neg for Duration {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            unit: self.unit,
        }
    }
}

impl PartialEq for Duration {
    fn eq(&self, other: &Self) -> bool {
        let other_converted = other.to_unit(self.unit);
        self.value == other_converted.value
    }
}

impl PartialOrd for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let other_converted = other.to_unit(self.unit);
        self.value.partial_cmp(&other_converted.value)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.value, self.unit)
    }
}

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug)]
pub enum DurationUnit {
    Ns,
    Us,
    Ms,
    S,
}

impl DurationUnit {
    #[inline]
    const fn multiplier(&self) -> f64 {
        match self {
            DurationUnit::Ns => 1.,
            DurationUnit::Us => 1_000.,
            DurationUnit::Ms => 1_000_000.,
            DurationUnit::S => 1_000_000_000.,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DurationUnitParseError;

impl fmt::Display for DurationUnitParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid duration unit. expected one of: ns, us, μs, ms or s"
        )
    }
}

impl std::error::Error for DurationUnitParseError {}

impl FromStr for DurationUnit {
    type Err = DurationUnitParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ns" => Ok(DurationUnit::Ns),
            "us" | "μs" => Ok(DurationUnit::Us),
            "ms" => Ok(DurationUnit::Ms),
            "s" => Ok(DurationUnit::S),
            _ => Err(DurationUnitParseError),
        }
    }
}

impl fmt::Display for DurationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let unit_str = match self {
            DurationUnit::Ns => "ns",
            DurationUnit::Us => "us",
            DurationUnit::Ms => "ms",
            DurationUnit::S => "s",
        };
        write!(f, "{}", unit_str)
    }
}
