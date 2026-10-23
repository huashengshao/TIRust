//! Frozen exact binary scaling for numerical Dual evaluation.
//!
//! Symbolica supplies logarithms, integer rounding, powers and division. The
//! logarithm only chooses a convenient exponent; it does not alter any input.
//! Every actual scaling factor is an exact power of two constructed separately
//! at each component's existing precision. Inverse factors are never formed.

use crate::{Complex, NativeFloat};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BinaryScale {
    exponent: i64,
}

impl BinaryScale {
    pub(crate) fn exponent(self) -> i64 {
        self.exponent
    }
    /// Select a scale only outside a moderate range. The complete factor need
    /// not be representable: `divide` uses bounded exact factors and never
    /// constructs a full factor or its reciprocal.
    pub(crate) fn for_positive_real<T: NativeFloat>(pivot: &Complex<T>) -> Option<Self> {
        if !pivot.im.is_zero() || !pivot.re.is_finite() || pivot.re <= pivot.re.zero() {
            return None;
        }
        let two = pivot.re.from_i64(2);
        let limit = two.pow(128);
        if pivot.re >= limit.inv() && pivot.re <= limit {
            return None;
        }
        // DoubleFloat's logarithm does not terminate for some subnormal
        // arguments. Use the existing scalar classification before calling it;
        // this exact shift brings every binary64 subnormal into the normal
        // range without inspecting or rounding its represented limbs.
        let (log_argument, offset) =
            if pivot.re.real_classify() == Some(std::num::FpCategory::Subnormal) {
                (pivot.re.clone() * two.pow(256), -256)
            } else {
                (pivot.re.clone(), 0)
            };
        let exponent = (log_argument.log() / two.log())
            .round_to_nearest_integer()
            .to_i64()?
            .checked_add(offset)?;
        Some(Self { exponent })
    }

    /// Retain the largest requested binary shift that is exactly reversible
    /// for every represented component. Widely separated scale/seed values
    /// must not overflow, underflow, or lose subnormal bits during preparation.
    pub(crate) fn bounded_by<T: NativeFloat>(self, values: &[Complex<T>]) -> Option<Self> {
        let safe = |exponent| {
            let scale = Self { exponent };
            let inverse = Self {
                exponent: -exponent,
            };
            values.iter().all(|value| {
                let mut shifted = value.clone();
                scale.divide(&mut shifted);
                if !shifted.re.is_finite() || !shifted.im.is_finite() {
                    return false;
                }
                inverse.divide(&mut shifted);
                shifted == *value
            })
        };
        if safe(self.exponent) {
            return Some(self);
        }
        let sign = self.exponent.signum();
        let mut lower = 0;
        let mut upper = self.exponent.unsigned_abs();
        while upper - lower > 1 {
            let middle = lower + (upper - lower) / 2;
            if safe(sign * middle as i64) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        (lower != 0).then_some(Self {
            exponent: sign * lower as i64,
        })
    }

    fn factor<T: NativeFloat>(&self, prototype: &T) -> T {
        let two = prototype.from_i64(2);
        if self.exponent < 0 {
            // Inverting 2^|exponent| first would overflow for subnormals.
            two.inv().pow(self.exponent.unsigned_abs())
        } else {
            two.pow(self.exponent as u64)
        }
    }

    /// Divide components independently, retaining their individual precisions.
    pub(crate) fn divide<T: NativeFloat>(&self, value: &mut Complex<T>) {
        // The existing DoubleFloat quotient can overflow its internal
        // reciprocal when the divisor is subnormal, even if the quotient is
        // moderate. Moderate exact factors also keep its product residuals
        // representable. This is numerical rescaling, not symbolic algebra.
        let mut remaining = self.exponent;
        while remaining != 0 {
            let step = remaining.clamp(-256, 256);
            let chunk = Self { exponent: step };
            value.re = value.re.clone() / chunk.factor(&value.re);
            value.im = value.im.clone() / chunk.factor(&value.im);
            remaining -= step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DoubleFloat, Float};
    use symbolica::domains::float::{FloatLike, Real};

    fn roundtrip<T: NativeFloat>(prototype: T, exponent: i64) {
        let scale = BinaryScale { exponent };
        let factor = scale.factor(&prototype);
        assert!(factor.is_finite() && !factor.is_zero());
        let mut point = Complex::new(factor.clone() * prototype.from_i64(3), prototype.zero());
        scale.divide(&mut point);
        assert_eq!(point.re, prototype.from_i64(3));
        assert!(point.im.is_zero());
    }

    #[test]
    fn exact_binary_scaling_handles_large_and_subnormal_factors() {
        for exponent in [-1024, -600, 600, 1020] {
            roundtrip(0.0_f64, exponent);
            roundtrip(DoubleFloat::from(0.), exponent);
        }
        for exponent in [-10000, -600, 600, 10000] {
            roundtrip(Float::new(256), exponent);
        }
    }

    #[test]
    fn frozen_power_of_two_keeps_mixed_float_precisions() {
        let pivot = Complex::new(Float::with_val(128, 2).pow(600), Float::new(64));
        let scale = BinaryScale::for_positive_real(&pivot).unwrap();
        let mut value = Complex::new(
            Float::with_val(256, 3) * Float::with_val(256, 2).pow(600),
            Float::with_val(64, 5) * Float::with_val(64, 2).pow(600),
        );
        scale.divide(&mut value);
        assert_eq!(value.re.get_precision(), 256);
        assert_eq!(value.im.get_precision(), 64);
        assert_eq!(value.re, Float::with_val(256, 3));
        assert_eq!(value.im, Float::with_val(64, 5));
        assert!(value.re.norm().is_finite());
    }

    #[test]
    fn subnormal_pivot_never_requires_its_overflowing_reciprocal() {
        let pivot = Complex::new(0.5_f64.pow(1070), 0.);
        let scale = BinaryScale::for_positive_real(&pivot).unwrap();
        let mut value = pivot;
        scale.divide(&mut value);
        assert_eq!(value, Complex::new(1., 0.));
        let pivot = Complex::new(DoubleFloat::from(0.5).pow(1070), DoubleFloat::from(0.));
        let scale = BinaryScale::for_positive_real(&pivot).unwrap();
        let mut value = pivot;
        scale.divide(&mut value);
        assert_eq!(
            value,
            Complex::new(DoubleFloat::from(1.), DoubleFloat::from(0.))
        );
    }
}
