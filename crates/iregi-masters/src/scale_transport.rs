//! Exact Laurent scale transport for lanes with constant kinematics.
//!
//! The normalization is `(mu_squared)^epsilon`. Symbolica expands its ratio,
//! extracts the Laurent coefficients, and differentiates the requested Dual
//! shape. No kinematic scalar-master expression is differentiated here.

use crate::{Complex, HyperDual, MasterError, NativeFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    evaluate::{BatchExpressionEvaluator, EvaluationDomain},
    id::Replacement,
};

#[derive(Clone)]
pub(crate) struct ScaleTransport<T: NativeFloat> {
    evaluator: BatchExpressionEvaluator<Complex<T>>,
}

impl<T: NativeFloat> ScaleTransport<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(crate) fn new(shape: &[Vec<usize>], prototype: &T) -> Result<Self, MasterError> {
        // finite, total single pole, double pole, UV pole, mu², primal mu²,
        // and a frozen exact binary exponent offset.
        let p: Vec<_> = (0..7)
            .map(|i| symbolica::symbol!(format!("iregi_scale_transport::p{i}")).to_atom())
            .collect();
        let epsilon_symbol = symbolica::symbol!("iregi_scale_transport::epsilon");
        let epsilon = Atom::var(epsilon_symbol);
        let logarithm = p[4].log() - p[5].log() + &p[6] * Atom::num(2).log();
        // Expand in a formal logarithm first. Expanding a composed logarithmic
        // expression can introduce exp(logarithm) intermediates, which need
        // not be representable even when its Laurent coefficients are finite.
        let formal_logarithm =
            symbolica::symbol!("iregi_scale_transport::formal_logarithm").to_atom();
        let series = ((&epsilon * &formal_logarithm).exp()
            * (&p[0] + &p[1] / &epsilon + &p[2] / epsilon.pow(2)))
        .series(epsilon_symbol, 0, 0)
        .map_err(|error| MasterError::Preparation(error.to_string()))?;
        let coefficient = |power: i64| {
            series
                .coefficient(power.into())
                .unwrap_or_else(|| Atom::num(0))
                .replace_multiple(
                    [Replacement::new(
                        formal_logarithm.clone(),
                        logarithm.clone(),
                    )]
                    .iter(),
                )
        };
        let single = coefficient(-1);
        let outputs = [
            coefficient(0),
            single.clone(),
            coefficient(-2),
            p[3].clone(),
            single - &p[3],
            Atom::num(0),
        ];
        let context = oneloop::OneLoopExpressions::new();
        let exact = context
            .dual_evaluator(&outputs, &p, HyperDual::<f64>::new(shape.to_vec()))
            .map_err(MasterError::Preparation)?;
        let evaluator = exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        prototype.from_rational(&value.re),
                        prototype.from_rational(&value.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator();
        Ok(Self { evaluator })
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<(), MasterError> {
        self.evaluator
            .evaluate_batch(input, output, rows)
            .map_err(|error| MasterError::Evaluation(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_binary_log_offset_does_not_require_its_exponential() {
        let shape = vec![vec![0], vec![1], vec![2]];
        let mut evaluator = ScaleTransport::new(&shape, &0.0_f64).unwrap();
        let mut input = Vec::new();
        for exponent in [-10000., -1070., 1070., 10000.] {
            let mut row = vec![Complex::new(0., 0.); 21];
            row[3].re = 1.; // total single pole
            row[9].re = 1.; // UV
            row[12].re = 1.; // scale primal
            row[13].re = 1.; // scale seed
            row[15].re = 1.; // reference scale
            row[18].re = exponent;
            input.extend(row);
        }
        let mut output = vec![Complex::new(0., 0.); 4 * 18];
        evaluator.evaluate_batch(&input, &mut output, 4).unwrap();
        for (row, exponent) in output
            .chunks_exact(18)
            .zip([-10000., -1070., 1070., 10000.])
        {
            assert!(
                row.iter()
                    .all(|value| value.re.is_finite() && value.im.is_finite())
            );
            assert!((row[0].re - exponent * 2_f64.ln()).abs() < 1e-10);
            assert_eq!(row[1], Complex::new(1., 0.));
            assert_eq!(row[2], Complex::new(-0.5, 0.));
        }
    }
}
