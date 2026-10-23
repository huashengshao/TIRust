//! Compiled Symbolica coefficient evaluation shared by the CLI and benchmarks.
use iregi_masters::{Complex, NativeFloat};
use iregi_prepare::PreparedGraph;
use iregi_rust::{NumeratorBatch, basis::Monomial};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::rational::Rational,
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
};

use crate::input::{Sample, parse_complex};

pub struct CoefficientPlan {
    names: Vec<String>,
    support: Vec<Monomial>,
    exact: Option<ExpressionEvaluator<Complex<Rational>>>,
}

impl CoefficientPlan {
    pub fn new(graph: &PreparedGraph, names: Vec<String>) -> Result<Self, String> {
        let parameters = names
            .iter()
            .map(|name| Atom::parse(name, "iregi_cli", Default::default()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("numerator parameter: {error}"))?;
        let expressions: Vec<_> = graph
            .numerator
            .terms
            .iter()
            .map(|term| term.coefficient.clone())
            .collect();
        let exact = if expressions.is_empty() {
            None
        } else {
            Some(Atom::evaluator_multiple(&expressions, &parameters)
                .direct_translation(true).horner_iterations(0).cores(1).build()
                .map_err(|error| format!("numerator parameters incomplete or unsupported: {error}; bind remaining coefficient symbols/functions in the input parameters map"))?)
        };
        Ok(Self {
            names,
            support: graph
                .numerator
                .terms
                .iter()
                .map(|term| Monomial(term.exponents))
                .collect(),
            exact,
        })
    }

    pub fn support(&self) -> &[Monomial] {
        &self.support
    }

    /// Parsing belongs to numerical input preparation, outside repeated kernels.
    pub fn pack<T: NativeFloat>(
        &self,
        samples: &[Sample],
        prototype: &T,
    ) -> Result<Vec<Complex<T>>, String> {
        let mut values = Vec::with_capacity(samples.len() * self.names.len());
        for sample in samples {
            if sample.parameters.keys().ne(self.names.iter()) {
                return Err("all samples must bind the same numerator parameter names".into());
            }
            for name in &self.names {
                values.push(parse_complex(&sample.parameters[name], prototype, false)?);
            }
        }
        Ok(values)
    }
}

pub struct CoefficientWorkspace<T: NativeFloat> {
    evaluator: Option<BatchExpressionEvaluator<Complex<T>>>,
    input_width: usize,
    width: usize,
    prototype: T,
}

impl<T: NativeFloat> CoefficientWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &CoefficientPlan, prototype: T) -> Self {
        let evaluator = plan.exact.as_ref().map(|exact| {
            exact
                .clone()
                .map_coeff_with_prec(
                    &|value| {
                        Complex::new(
                            prototype.from_rational(&value.re),
                            prototype.from_rational(&value.im),
                        )
                    },
                    prototype.get_precision(),
                )
                .into_batch_evaluator()
        });
        Self {
            evaluator,
            input_width: plan.names.len(),
            width: plan.support.len(),
            prototype,
        }
    }

    /// Evaluates every coefficient for every lane. Nonfinite values remain in
    /// that lane so the contraction API can report its independent failure.
    pub fn evaluate(
        &mut self,
        parameters: &[Complex<T>],
        rows: usize,
    ) -> Result<NumeratorBatch<T>, String> {
        if parameters.len()
            != rows
                .checked_mul(self.input_width)
                .ok_or("numerator batch size overflow")?
        {
            return Err("numerator parameter batch shape mismatch".into());
        }
        let size = rows
            .checked_mul(self.width)
            .ok_or("numerator output size overflow")?;
        let mut values = vec![Complex::new(self.prototype.zero(), self.prototype.zero()); size];
        if rows != 0
            && let Some(evaluator) = &mut self.evaluator
        {
            evaluator
                .evaluate_batch(parameters, &mut values, rows)
                .map_err(|error| format!("numerator coefficient evaluation: {error}"))?;
        }
        Ok(NumeratorBatch {
            rows,
            components: 1,
            values,
        })
    }
}
