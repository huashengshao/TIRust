//! Relative conditioning of derived scalar-master invariants.
//!
//! Symbolica maps rational polynomial coefficients to their absolute values.
//! The executor evaluates these positive polynomials, with the same Dualizer,
//! on the norms of individual input Taylor coefficients. This includes internal
//! cancellation inside a monomial's Taylor convolution; taking the norm only
//! after evaluating a signed monomial would miss it. No ABS is differentiated.
//!
//! These conservative operation-count estimates follow the existing stability
//! monitor; they are not certified error bounds. Exact zeros are checked by the
//! independent polynomial certificate, before any scalar-master branch.

use iregi_masters::NativeFloat;
use std::num::FpCategory;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{float::Complex, rational::Q},
    poly::PolyVariable,
};

use crate::ReductionError;

use super::{dyadic, operation_count};

#[derive(Clone, Debug)]
pub(crate) struct ArgumentStabilityPlan {
    expressions: Vec<Atom>,
    operations: Vec<usize>,
}

impl ArgumentStabilityPlan {
    pub(crate) fn prepare(
        parameters: &[Atom],
        polynomials: &[Atom],
    ) -> Result<Self, ReductionError> {
        let mut expressions = Vec::with_capacity(polynomials.len());
        let mut operations = Vec::with_capacity(polynomials.len());
        for expression in polynomials {
            let polynomial = expression
                .try_to_polynomial::<_, u32>(&Q, parameters)
                .map_err(|error| ReductionError::Preparation(error.to_string()))?;
            if polynomial.variables().iter().any(|variable| {
                !matches!(variable, PolyVariable::Symbol(symbol) if parameters.contains(&Atom::var(*symbol)))
            }) {
                return Err(ReductionError::Preparation(
                    "argument conditioning requires rational polynomials in the declared input symbols".into(),
                ));
            }
            // Symbolica owns coefficient mapping, polynomial reconstruction,
            // expansion and subsequent ordered Taylor multiplication.
            let envelope = polynomial
                .map_coeff(|coefficient| coefficient.abs(), Q)
                .to_expression();
            operations.push(
                operation_count(expression.as_view())
                    .max(operation_count(envelope.as_view()))
                    .max(1),
            );
            expressions.push(envelope);
        }
        Ok(Self {
            expressions,
            operations,
        })
    }

    /// Evaluate on `Complex(hypot(re,im), 0)` for every input Taylor
    /// component, through a separate instruction-major batch evaluator.
    pub(crate) fn expressions(&self) -> &[Atom] {
        &self.expressions
    }

    pub(crate) fn inspect_lane<T: NativeFloat>(
        &self,
        values: &[Complex<T>],
        envelopes: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        effective_input_bits: u32,
    ) -> Option<String> {
        let Some(width) = self.expressions.len().checked_mul(components) else {
            return Some("derived-invariant conditioning buffer size overflow".into());
        };
        if components == 0 || values.len() != width || envelopes.len() != width {
            return Some("invalid derived-invariant conditioning buffer".into());
        }
        let input_bits = prototype.get_precision().min(effective_input_bits);
        if input_bits == 0 {
            return Some("derived-invariant conditioning requires positive input precision".into());
        }
        let tolerance = dyadic(prototype, 19i64 - i64::from(accuracy_bits));
        for (constraint, operations) in self.operations.iter().enumerate() {
            // A coefficient of a Dual product has at most components² ordered
            // input pairs. This is resource accounting, not Taylor arithmetic.
            let count = operations
                .saturating_mul(8)
                .saturating_mul(components)
                .saturating_mul(components);
            for component in 0..components {
                let offset = constraint * components + component;
                let value = &values[offset];
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Some(format!(
                        "nonfinite derived distance {constraint}, Taylor component {component}"
                    ));
                }
                if value.re.is_zero() && value.im.is_zero() {
                    // True zero must remain usable, including a zero primal
                    // with nonzero transverse derivatives. Its exact proof is
                    // handled by check_selected_batch after this monitor.
                    continue;
                }
                let envelope = &envelopes[offset];
                if !envelope.re.is_finite()
                    || !envelope.im.is_zero()
                    || envelope.re < prototype.zero()
                {
                    return Some(format!(
                        "nonfinite or invalid derived-distance envelope {constraint}, Taylor component {component}"
                    ));
                }
                let magnitude = NativeFloat::hypot(&value.re, &value.im);
                // Float input operands retain their own precision in the
                // evaluator. A high-precision workspace does not upgrade an
                // 80-bit input multiplication to 256-bit arithmetic.
                let envelope_bits = input_bits.min(envelope.re.get_precision());
                let unit = dyadic(prototype, 1i64 - i64::from(envelope_bits));
                let roundoff = unit.clone() * prototype.from_usize(count);
                let estimate = roundoff.clone() * &envelope.re;
                let output_bits = [&value.re, &value.im]
                    .into_iter()
                    .filter(|number| !number.is_zero())
                    .fold(magnitude.get_precision(), |bits, number| {
                        bits.min(number.get_precision())
                    })
                    .min(prototype.get_precision());
                let output_unit = dyadic(prototype, 1i64 - i64::from(output_bits));
                // Tracked output precision already accounts for cancellation.
                // Applying that precision to the large input envelope would
                // count the same loss twice. Check its relative uncertainty
                // separately; Float's reported precision may be two bits high.
                let output_estimate = output_unit.clone() * prototype.from_usize(8) * &magnitude;
                let target = tolerance.clone() * &magnitude;
                // The relative roundoff model does not cover gradual
                // underflow. In particular, a rounded-zero estimate is not
                // evidence that a nonzero subnormal invariant is accurate.
                // Use Symbolica's numeric classification rather than inventing
                // a minimum exponent shared by f64, DoubleFloat and Float.
                // Precision rescue can reevaluate these lanes with a wider
                // exponent range; exact zeros still use their certificate.
                if [
                    &unit,
                    &tolerance,
                    &roundoff,
                    &magnitude,
                    &envelope.re,
                    &estimate,
                    &output_unit,
                    &output_estimate,
                    &target,
                ]
                .into_iter()
                .any(|number| {
                    !number.is_finite()
                        || number.is_zero()
                        || number.real_classify() == Some(FpCategory::Subnormal)
                }) || estimate > target
                    || output_estimate > target
                {
                    return Some(format!(
                        "derived distance {constraint}, Taylor component {component}, exceeds the relative conditioning target 2^(19-{accuracy_bits}) at working precision {} and input precision {input_bits}; this operation-count estimate is not a certified error bound",
                        prototype.get_precision(),
                    ));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::{
        domains::{
            dual::HyperDual,
            float::{Float, FloatLike, SingleFloat},
            rational::Rational,
        },
        evaluate::{Dualizer, ExpressionEvaluator},
    };

    fn c(value: f64) -> Complex<f64> {
        Complex::new(value, 0.)
    }

    #[test]
    fn positive_argument_envelope_retains_internal_monomial_taylor_cancellation() {
        let x = Atom::var(symbolica::symbol!("iregi_argument_conditioning::x"));
        let plan = ArgumentStabilityPlan::prepare(&[x.clone()], &[x.pow(2)]).unwrap();
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(plan.expressions(), &[x])
                .build()
                .unwrap();
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1], vec![2]]);
        let mut evaluator = exact
            .vectorize(&Dualizer::new(shape, Vec::new()))
            .unwrap()
            .map_coeff(&|value| Complex::new(value.re.to_f64(), value.im.to_f64()))
            .into_batch_evaluator();
        // For x=(1,2^27-1,-2^53+2^27+2), the exact quadratic
        // coefficient of x² is5, but binary64 convolution produces4.
        let positive = [
            c(1.),
            c(2_f64.powi(27) - 1.),
            c(2_f64.powi(53) - 2_f64.powi(27) - 2.),
        ];
        let mut envelope = vec![c(0.); 3];
        evaluator
            .evaluate_batch(&positive, &mut envelope, 1)
            .unwrap();
        assert!(envelope[2].re > 2_f64.powi(54));
        let value = [c(1.), c(2. * (2_f64.powi(27) - 1.)), c(4.)];
        assert!(
            plan.inspect_lane(&value, &envelope, 3, &0_f64, 53, 53)
                .unwrap()
                .contains("Taylor component 2")
        );
        // Already-collapsed monomial jets would miss this cancellation.
        assert!(
            plan.inspect_lane(&value, &value, 3, &0_f64, 53, 53)
                .is_none()
        );
    }

    #[test]
    fn argument_conditioning_respects_requested_accuracy_and_defers_exact_zeros() {
        let x = Atom::var(symbolica::symbol!("iregi_argument_conditioning::x"));
        let y = Atom::var(symbolica::symbol!("iregi_argument_conditioning::y"));
        let plan = ArgumentStabilityPlan::prepare(&[x.clone(), y.clone()], &[&x - &y]).unwrap();
        assert!(
            plan.inspect_lane(&[c(1.)], &[c(1e6)], 1, &0_f64, 53, 53)
                .is_some()
        );
        assert!(
            plan.inspect_lane(&[c(1.)], &[c(1e6)], 1, &0_f64, 40, 53)
                .is_none()
        );
        assert!(
            plan.inspect_lane(&[c(0.)], &[c(f64::INFINITY)], 1, &0_f64, 53, 53)
                .is_none()
        );
        assert!(
            plan.inspect_lane(&[c(1.)], &[c(f64::INFINITY)], 1, &0_f64, 53, 53)
                .is_some()
        );
    }

    #[test]
    fn subnormal_distance_cannot_pass_with_an_underflowed_error_estimate() {
        let parameters: Vec<_> = (0..4)
            .map(|component| {
                Atom::var(symbolica::symbol!(format!(
                    "iregi_argument_underflow::p{component}"
                )))
            })
            .collect();
        let polynomial = parameters[0].pow(2)
            - parameters[1].pow(2)
            - parameters[2].pow(2)
            - parameters[3].pow(2);
        let plan = ArgumentStabilityPlan::prepare(&parameters, &[polynomial]).unwrap();
        // In units delta=2^-1074, this represented input has exact square
        // (10^10-3/16)*delta. Binary64's individual products instead give
        // (10^10+1)*delta, a relative error greater than 1e-10.
        let scale = 2_f64.powi(-537);
        let time = 200000.75 * scale;
        let space = 100000.5 * scale;
        let squared_time = time * time;
        let squared_space = space * space;
        let value = squared_time - squared_space - squared_space - squared_space;
        let envelope = squared_time + squared_space + squared_space + squared_space;
        assert_eq!(value.to_bits(), 10_000_000_001);
        assert_eq!(envelope.to_bits(), 70_000_600_001);
        let roundoff = 2_f64.powi(-52) * (plan.operations[0] * 8) as f64;
        assert_eq!(roundoff * envelope, 0.);
        assert!(2_f64.powi(-34) * value > 0.);
        assert!(
            plan.inspect_lane(&[c(value)], &[c(envelope)], 1, &0_f64, 53, 53)
                .is_some()
        );
    }

    #[test]
    fn mixed_float_precision_is_respected_without_counting_cancellation_twice() {
        let x = Atom::var(symbolica::symbol!("iregi_argument_mixed_precision::x"));
        let plan = ArgumentStabilityPlan::prepare(&[x.clone()], &[x.pow(2)]).unwrap();
        let prototype = Float::new(256);
        let complex = |value| Complex::new(value, Float::new(256));
        // These are the rounded value and positive envelope scales for
        // q=(2^53,2^53-1,2^27,2) evaluated with 80-bit Float operands.
        // The exact represented-input square is -5, and its products lose +1.
        let value = [complex(Float::with_val(80, -4))];
        let envelope = [complex(Float::with_val(80, 1_u128 << 107))];
        assert!(
            plan.inspect_lane(&value, &envelope, 1, &prototype, 53, 80)
                .is_some()
        );
        // For the smaller q=(2^27,2^27-1,2^14,2), 128-bit input
        // arithmetic leaves ample accuracy after roughly 53 cancelled bits.
        // Applying the 75-bit output precision to its 2^55 envelope would
        // incorrectly reject it by counting the cancellation twice.
        let value = [complex(Float::with_val(75, -5))];
        let envelope = [complex(Float::with_val(128, 1_u64 << 55))];
        assert!(
            plan.inspect_lane(&value, &envelope, 1, &prototype, 53, 128)
                .is_none()
        );
    }

    #[test]
    fn mixed_precision_higher_taylor_coefficients_keep_their_own_precision() {
        let x = Atom::var(symbolica::symbol!("iregi_argument_mixed_taylor::x"));
        let plan = ArgumentStabilityPlan::prepare(&[x.clone()], &[x.pow(2)]).unwrap();
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(plan.expressions(), &[x])
                .build()
                .unwrap();
        let prototype = Float::new(256);
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1], vec![2]]);
        let mut evaluator = exact
            .vectorize(&Dualizer::new(shape, Vec::new()))
            .unwrap()
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        prototype.from_rational(&value.re),
                        prototype.from_rational(&value.im),
                    )
                },
                256,
            )
            .into_batch_evaluator();
        let complex = |value| Complex::new(value, Float::new(256));
        let input = [
            complex(Float::with_val(80, 1)),
            complex(Float::with_val(160, 3)),
            complex(Float::with_val(256, 5)),
            complex(Float::with_val(256, 1)),
            complex(Float::with_val(80, (1_u64 << 53) - 1)),
            complex(Float::with_val(256, -(1_i128 << 105) + (1_i128 << 53) + 2)),
        ];
        let positive: Vec<_> = input
            .iter()
            .map(|value| complex(NativeFloat::hypot(&value.re, &value.im)))
            .collect();
        let mut values = vec![complex(Float::new(256)); 6];
        let mut envelopes = values.clone();
        evaluator.evaluate_batch(&input, &mut values, 2).unwrap();
        evaluator
            .evaluate_batch(&positive, &mut envelopes, 2)
            .unwrap();
        assert!(values[2].re.get_precision() > 80);
        assert!(
            plan.inspect_lane(&values[..3], &envelopes[..3], 3, &prototype, 53, 80)
                .is_none()
        );
        assert!(
            plan.inspect_lane(&values[3..], &envelopes[3..], 3, &prototype, 53, 80)
                .unwrap()
                .contains("Taylor component 2")
        );
    }
}
