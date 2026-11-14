//! Row-major batches over Symbolica's scalar [`ExpressionEvaluator`].

use symbolica::{
    atom::Atom,
    domains::float::Real,
    evaluate::{EvaluationError, ExpressionEvaluator},
};

/// Evaluates `rows` row-major samples with one scalar evaluation per row.
///
/// Every row runs the interpreter program of a single evaluation, so the
/// results are those of [`ExpressionEvaluator::evaluate`] on each row. Input
/// and output lengths must be exactly `rows * get_input_len()` and
/// `rows * get_output_len()`; shape errors are reported before any row is
/// evaluated and leave `out` unchanged. Empty batches are supported.
#[derive(Clone)]
pub struct RowEvaluator<T> {
    evaluator: ExpressionEvaluator<T>,
    inputs: usize,
    outputs: usize,
}

impl<T: Real> From<ExpressionEvaluator<T>> for RowEvaluator<T> {
    fn from(mut evaluator: ExpressionEvaluator<T>) -> Self {
        // `get_input_len`/`get_output_len` require `T: Default`, which
        // `Complex` does not implement: read both counts from the shape errors
        // of `try_evaluate`, which are reported before anything is evaluated.
        let shape_error = |error| unreachable!("unexpected evaluator probe error: {error}");
        let (inputs, outputs) = match evaluator.try_evaluate(&[], &mut []) {
            Ok(()) => (0, 0),
            Err(EvaluationError::InvalidOutputCount { expected, .. }) => (0, expected),
            Err(EvaluationError::InvalidParameterCount { expected, .. }) => {
                let zeros = vec![T::new_zero(); expected];
                match evaluator.try_evaluate(&zeros, &mut []) {
                    Ok(()) => (expected, 0),
                    Err(EvaluationError::InvalidOutputCount {
                        expected: outputs, ..
                    }) => (expected, outputs),
                    Err(error) => shape_error(error),
                }
            }
            Err(error) => shape_error(error),
        };
        Self {
            evaluator,
            inputs,
            outputs,
        }
    }
}

/// `evaluator.into_row_evaluator()`, for builder chains.
pub trait IntoRowEvaluator<T> {
    fn into_row_evaluator(self) -> RowEvaluator<T>;
}

impl<T: Real> IntoRowEvaluator<T> for ExpressionEvaluator<T> {
    fn into_row_evaluator(self) -> RowEvaluator<T> {
        RowEvaluator::from(self)
    }
}

impl<T> RowEvaluator<T> {
    /// Number of parameters in each input row.
    pub fn get_input_len(&self) -> usize {
        self.inputs
    }

    /// Number of results in each output row.
    pub fn get_output_len(&self) -> usize {
        self.outputs
    }

    /// Return the scalar evaluator.
    pub fn into_expression_evaluator(self) -> ExpressionEvaluator<T> {
        self.evaluator
    }
}

impl<T: Real> RowEvaluator<T> {
    pub fn evaluate_batch(
        &mut self,
        parameters: &[T],
        out: &mut [T],
        rows: usize,
    ) -> Result<(), EvaluationError> {
        let (inputs, outputs) = (self.inputs, self.outputs);
        let overflow = || EvaluationError::EvaluationFailed {
            expression: Atom::num(0),
            reason: "Batch dimensions exceed the addressable buffer size".into(),
        };
        let expected = rows.checked_mul(inputs).ok_or_else(overflow)?;
        if parameters.len() != expected {
            return Err(EvaluationError::InvalidParameterCount {
                expected,
                actual: parameters.len(),
            });
        }
        let expected = rows.checked_mul(outputs).ok_or_else(overflow)?;
        if out.len() != expected {
            return Err(EvaluationError::InvalidOutputCount {
                expected,
                actual: out.len(),
            });
        }
        for row in 0..rows {
            self.evaluator.try_evaluate(
                &parameters[row * inputs..(row + 1) * inputs],
                &mut out[row * outputs..(row + 1) * outputs],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::{atom::AtomCore, parse};

    #[test]
    fn rows_match_scalar_evaluation_and_shape_errors_leave_output() {
        let exact = parse!("x^2+2*x*y+y")
            .evaluator(&[parse!("x"), parse!("y")])
            .build()
            .unwrap();
        let mut scalar = exact.map_coeff(&|c| c.re.to_f64());
        let mut rows_evaluator = scalar.clone().into_row_evaluator();
        assert_eq!((rows_evaluator.get_input_len(), rows_evaluator.get_output_len()), (2, 1));
        for rows in [0, 1, 8, 3] {
            let parameters = (0..rows)
                .flat_map(|i| [i as f64 / 7., (rows - i) as f64 / 3.])
                .collect::<Vec<_>>();
            let mut actual = vec![0.; rows];
            rows_evaluator.evaluate_batch(&parameters, &mut actual, rows).unwrap();
            for (input, result) in parameters.chunks_exact(2).zip(actual.chunks_exact(1)) {
                let mut expected = [0.; 1];
                scalar.evaluate(input, &mut expected);
                assert_eq!(result, expected);
            }
        }
        let mut out = [42.; 2];
        assert!(matches!(
            rows_evaluator.evaluate_batch(&[1.; 3], &mut out, 2),
            Err(EvaluationError::InvalidParameterCount { .. })
        ));
        assert!(matches!(
            rows_evaluator.evaluate_batch(&[1.; 4], &mut out[..1], 2),
            Err(EvaluationError::InvalidOutputCount { .. })
        ));
        assert_eq!(out, [42.; 2]);
        let mut constant = parse!("pi")
            .evaluator::<Atom>(&[])
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64())
            .into_row_evaluator();
        let mut out = [0.; 5];
        constant.evaluate_batch(&[], &mut out, 5).unwrap();
        assert_eq!(out, [std::f64::consts::PI; 5]);
    }
}
