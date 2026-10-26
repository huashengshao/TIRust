//! Reusable contraction of a numerical polynomial with reduced tensor moments.
//!
//! Graph import and symbolic preparation of numerator coefficients are external
//! concerns. This module accepts coefficients of ordinary Cartesian monomials
//! and delegates the contraction, including Dual products, to a Symbolica
//! instruction-major batch evaluator.

use std::collections::HashMap;

use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        float::{Complex, FloatLike},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

use crate::{
    basis::Monomial,
    convention::multinomial,
    plan::ReductionPlan,
    types::{BatchResult, Convention, LaneStatus, ReductionError, TensorWeights},
};

/// Immutable polynomial support and contraction convention.
///
/// Each supplied coefficient multiplies exactly `q0^a0 q1^a1 q2^a2 q3^a3`,
/// with no multinomial factor. The loop momentum is the same `q` used by the
/// reduction denominators. The plan removes stored tensor multiplicities and
/// compensates the MadLoop odd-rank moment sign. The contracted result retains
/// the chosen integral normalization and four separate Laurent slots.
#[derive(Clone, Debug)]
pub struct ContractionPlan {
    basis: Vec<Monomial>,
    convention: Convention,
    weights: TensorWeights,
    monomials: Vec<Monomial>,
    positions: Vec<usize>,
    parameters: Vec<Atom>,
    expressions: [Atom; 4],
}

impl ContractionPlan {
    pub fn new(reduction: &ReductionPlan, monomials: &[Monomial]) -> Result<Self, ReductionError> {
        Self::from_basis(
            reduction.basis(),
            reduction.settings().convention,
            reduction.settings().tensor_weights,
            monomials,
        )
    }

    pub(crate) fn from_basis(
        basis: &[Monomial],
        convention: Convention,
        weights: TensorWeights,
        monomials: &[Monomial],
    ) -> Result<Self, ReductionError> {
        let lookup: HashMap<_, _> = basis
            .iter()
            .copied()
            .enumerate()
            .map(|(index, monomial)| (monomial, index))
            .collect();
        let positions = monomials
            .iter()
            .map(|monomial| {
                lookup.get(monomial).copied().ok_or_else(|| {
                    ReductionError::InvalidInput(format!(
                        "numerator monomial {:?} is absent from the reduction basis",
                        monomial.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coefficients: Vec<_> = (0..monomials.len())
            .map(|index| {
                Atom::var(symbolica::symbol!(format!(
                    "iregi_contract::coefficient_{index}"
                )))
            })
            .collect();
        let mut parameters = coefficients.clone();
        let mut expressions: [Atom; 4] = std::array::from_fn(|_| Atom::Zero);
        for (index, monomial) in monomials.iter().enumerate() {
            let mut coefficient = coefficients[index].clone();
            if convention == Convention::MadLoop && monomial.rank() % 2 == 1 {
                coefficient = -coefficient;
            }
            if weights == TensorWeights::Multinomial {
                coefficient = coefficient / multinomial(*monomial);
            }
            for (pole, expression) in expressions.iter_mut().enumerate() {
                let moment = Atom::var(symbolica::symbol!(format!(
                    "iregi_contract::moment_{index}_{pole}"
                )));
                *expression = &*expression + &coefficient * &moment;
                parameters.push(moment);
            }
        }
        Ok(Self {
            basis: basis.to_vec(),
            convention,
            weights,
            monomials: monomials.to_vec(),
            positions,
            parameters,
            expressions,
        })
    }

    pub fn monomials(&self) -> &[Monomial] {
        &self.monomials
    }
    pub fn term_count(&self) -> usize {
        self.monomials.len()
    }
}

/// Numerical coefficients of `ContractionPlan::monomials()`.
///
/// Storage is row-major, then polynomial term, then Taylor component. For
/// ordinary numbers `components` is one. An empty polynomial has no values
/// even for a nonempty batch and contracts successfully to zero.
#[derive(Clone, Debug)]
pub struct NumeratorBatch<T: FloatLike> {
    pub rows: usize,
    pub components: usize,
    pub values: Vec<Complex<T>>,
}

/// Contracted `[UV1, IR1, IR2, finite]` coefficients, with Taylor components
/// contiguous inside each slot. Failed input lanes retain their diagnostics.
#[derive(Clone, Debug)]
pub struct ContractionResult<T: FloatLike> {
    pub components: usize,
    pub statuses: Vec<LaneStatus>,
    pub fallback_used: Vec<bool>,
    pub analytic_used: Vec<bool>,
    values: Vec<Complex<T>>,
}

impl<T: FloatLike> ContractionResult<T> {
    pub fn rows(&self) -> usize {
        self.statuses.len()
    }
    pub fn lane(&self, row: usize) -> Result<&[Complex<T>], &LaneStatus> {
        if self.statuses[row] != LaneStatus::Success {
            return Err(&self.statuses[row]);
        }
        let width = 4 * self.components;
        Ok(&self.values[row * width..(row + 1) * width])
    }
}

/// Owned numerical evaluator and reusable scratch buffers.
///
/// Constants are converted directly from exact Symbolica coefficients at the
/// requested precision. Static and dynamic Dual layouts are lowered with
/// Symbolica's Dualizer; derivative components never become sample lanes.
#[derive(Clone)]
pub struct ContractionWorkspace<T> {
    prototype: T,
    plan: ContractionPlan,
    components: usize,
    shape: Option<Vec<Vec<usize>>>,
    evaluator: BatchExpressionEvaluator<Complex<T>>,
    input: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
    active: Vec<usize>,
    evaluated_rows: usize,
}

impl<T: NativeFloat> ContractionWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &ContractionPlan, prototype: T) -> Result<Self, ReductionError> {
        Self::prepare::<HyperDual<Complex<Rational>>>(plan, prototype, None)
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &ContractionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        Self::prepare(plan, prototype, Some(&dual))
    }

    fn prepare<D: DualNumberStructure>(
        plan: &ContractionPlan,
        prototype: T,
        dual: Option<&D>,
    ) -> Result<Self, ReductionError> {
        if prototype.get_precision() == 0 {
            return Err(ReductionError::Preparation(
                "working precision must be positive".into(),
            ));
        }
        if let Some(layout) = dual {
            iregi_masters::validate_dual_structure(layout)
                .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        }
        let shape = dual.map(|layout| {
            layout
                .get_shape()
                .into_iter()
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>()
        });
        let mut exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&plan.expressions, &plan.parameters)
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        if let Some(layout) = dual {
            exact = exact
                .vectorize(&Dualizer::new(BorrowedDual(layout), Vec::new()))
                .map_err(ReductionError::Preparation)?;
        } else {
            exact.optimize_stack();
        }
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
        Ok(Self {
            prototype,
            plan: plan.clone(),
            components: shape.as_ref().map_or(1, Vec::len),
            shape,
            evaluator,
            input: Vec::new(),
            output: Vec::new(),
            active: Vec::new(),
            evaluated_rows: 0,
        })
    }

    pub fn component_count(&self) -> usize {
        self.components
    }
    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.shape.as_deref()
    }
    pub fn binary_precision(&self) -> u32 {
        self.prototype.get_precision()
    }
    /// Number of sample lanes evaluated by the last instruction-major call.
    pub fn last_evaluated_rows(&self) -> usize {
        self.evaluated_rows
    }

    /// Accept complex static or dynamic Dual coefficients directly, preserving
    /// their component order. Moments remain the result of `Workspace::new_dual`.
    pub fn contract_typed_dual_batch<D: DualNumber<Complex<T>>>(
        &mut self,
        moments: &BatchResult<T>,
        coefficients: &[D],
    ) -> Result<ContractionResult<T>, ReductionError> {
        self.check_dual_coefficients(moments.rows(), coefficients)?;
        let values = coefficients
            .iter()
            .flat_map(|coefficient| coefficient.values().iter().cloned())
            .collect();
        self.contract_batch(
            moments,
            &NumeratorBatch {
                rows: moments.rows(),
                components: self.components,
                values,
            },
        )
    }

    /// Accept real static or dynamic Dual coefficients, embedding every Taylor
    /// component in the complex domain at its original precision.
    pub fn contract_real_dual_batch<D: DualNumber<T>>(
        &mut self,
        moments: &BatchResult<T>,
        coefficients: &[D],
    ) -> Result<ContractionResult<T>, ReductionError> {
        self.check_dual_coefficients(moments.rows(), coefficients)?;
        let values = coefficients
            .iter()
            .flat_map(|coefficient| {
                coefficient
                    .values()
                    .iter()
                    .map(|value| Complex::new(value.clone(), value.zero()))
            })
            .collect();
        self.contract_batch(
            moments,
            &NumeratorBatch {
                rows: moments.rows(),
                components: self.components,
                values,
            },
        )
    }

    fn check_dual_coefficients<D: DualNumberStructure>(
        &self,
        rows: usize,
        coefficients: &[D],
    ) -> Result<(), ReductionError> {
        let shape = self.shape.as_ref().ok_or_else(|| {
            ReductionError::InvalidInput(
                "use new_dual before supplying Dual numerator coefficients".into(),
            )
        })?;
        if coefficients.len() != size(rows, self.plan.term_count())? {
            return Err(ReductionError::InvalidInput(
                "wrong number of Dual numerator coefficients".into(),
            ));
        }
        let expected_shape: Vec<_> = shape.iter().map(Vec::as_slice).collect();
        if coefficients.iter().any(|coefficient| {
            coefficient.get_len() != self.components || coefficient.get_shape() != expected_shape
        }) {
            return Err(ReductionError::InvalidInput(
                "numerator Dual shape differs from prepared shape".into(),
            ));
        }
        Ok(())
    }

    /// Contract a batch produced by the same reduction plan used at creation.
    /// The basis, integral convention, tensor weights and full Taylor layout
    /// are checked before touching a lane.
    pub fn contract_batch(
        &mut self,
        moments: &BatchResult<T>,
        numerator: &NumeratorBatch<T>,
    ) -> Result<ContractionResult<T>, ReductionError> {
        self.evaluated_rows = 0;
        if moments.basis != self.plan.basis {
            return Err(ReductionError::InvalidInput(
                "contraction and reduction bases differ".into(),
            ));
        }
        if moments.metadata().convention() != self.plan.convention
            || moments.metadata().tensor_weights() != self.plan.weights
        {
            return Err(ReductionError::InvalidInput(
                "contraction and reduction conventions or tensor weights differ".into(),
            ));
        }
        if moments.metadata().dual_shape() != self.shape.as_deref() {
            return Err(ReductionError::InvalidInput(
                "contraction and reduction Dual shapes differ".into(),
            ));
        }
        if numerator.rows != moments.rows()
            || numerator.components != self.components
            || moments.components != self.components
        {
            return Err(ReductionError::InvalidInput(
                "numerator, moments and contraction workspace have different batch/Taylor dimensions".into()));
        }
        let coefficient_width = size(self.plan.term_count(), self.components)?;
        if numerator.values.len() != size(numerator.rows, coefficient_width)? {
            return Err(ReductionError::InvalidInput(
                "numerator coefficient buffer has wrong size".into(),
            ));
        }
        let moment_width = size(size(self.plan.basis.len(), 4)?, self.components)?;
        if moments.values.len() != size(moments.rows(), moment_width)?
            || moments.fallback_used.len() != moments.rows()
            || moments.analytic_used.len() != moments.rows()
        {
            return Err(ReductionError::InvalidInput(
                "reduction result buffer has wrong size".into(),
            ));
        }
        let output_width = size(4, self.components)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = ContractionResult {
            components: self.components,
            statuses: moments.statuses.clone(),
            fallback_used: moments.fallback_used.clone(),
            analytic_used: moments.analytic_used.clone(),
            values: vec![zero.clone(); size(moments.rows(), output_width)?],
        };
        self.input.clear();
        self.output.clear();
        self.active.clear();
        for row in 0..moments.rows() {
            if result.statuses[row] != LaneStatus::Success {
                continue;
            }
            let coefficients =
                &numerator.values[row * coefficient_width..(row + 1) * coefficient_width];
            if coefficients.iter().any(|value| !finite(value)) {
                result.statuses[row] =
                    LaneStatus::InvalidInput("nonfinite numerator coefficient".into());
                continue;
            }
            let data = moments
                .lane(row)
                .expect("successful moment lane was checked");
            if self.plan.positions.iter().any(|&position| {
                data[position * output_width..(position + 1) * output_width]
                    .iter()
                    .any(|value| !finite(value))
            }) {
                result.statuses[row] = LaneStatus::Unstable("nonfinite tensor moment".into());
                continue;
            }
            self.active.push(row);
            self.input.extend_from_slice(coefficients);
            for &position in &self.plan.positions {
                self.input.extend_from_slice(
                    &data[position * output_width..(position + 1) * output_width],
                );
            }
        }
        if self.active.is_empty() || self.plan.term_count() == 0 {
            return Ok(result);
        }
        self.output
            .resize(size(self.active.len(), output_width)?, zero);
        self.evaluator
            .evaluate_batch(&self.input, &mut self.output, self.active.len())
            .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
        self.evaluated_rows = self.active.len();
        for (compact, &row) in self.active.iter().enumerate() {
            let values = &self.output[compact * output_width..(compact + 1) * output_width];
            if values.iter().any(|value| !finite(value)) {
                result.statuses[row] =
                    LaneStatus::Unstable("nonfinite numerator contraction".into());
            } else {
                result.values[row * output_width..(row + 1) * output_width]
                    .clone_from_slice(values);
            }
        }
        Ok(result)
    }
}

struct BorrowedDual<'a, D>(&'a D);
impl<D: DualNumberStructure> DualNumberStructure for BorrowedDual<'_, D> {
    fn get_len(&self) -> usize {
        self.0.get_len()
    }
    fn get_shape(&self) -> Vec<&[usize]> {
        self.0.get_shape()
    }
    fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
        self.0.get_multiplication_table()
    }
    fn get_max_depth(&self) -> usize {
        self.0.get_max_depth()
    }
}

fn finite<T: NativeFloat>(value: &Complex<T>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
fn size(left: usize, right: usize) -> Result<usize, ReductionError> {
    left.checked_mul(right)
        .ok_or_else(|| ReductionError::InvalidInput("contraction dimensions overflow usize".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::tensor_basis;
    use symbolica::domains::float::{DoubleFloat, Float, SingleFloat};

    symbolica::create_hyperdual_from_components!(StaticCoefficient, [[0], [1]]);

    fn c(value: f64) -> Complex<f64> {
        Complex::new(value, 0.)
    }

    #[test]
    fn rejects_mismatched_result_conventions_and_tensor_weights() {
        let basis = tensor_basis(2);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::MadLoop,
            TensorWeights::Multinomial,
            &[Monomial([1, 1, 0, 0])],
        )
        .unwrap();
        let mut workspace = ContractionWorkspace::new(&plan, 0.).unwrap();
        let mut moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                Convention::MadLoop,
                TensorWeights::Components,
                None,
            ),
            values: vec![c(2.); basis.len() * 4],
            basis,
            components: 1,
            statuses: vec![LaneStatus::Success],
            fallback_used: vec![false],
            analytic_used: vec![false],
            primal_envelopes: vec![None; 1],
            primal_envelope_errors: vec![None; 1],
        };
        let numerator = NumeratorBatch {
            rows: 1,
            components: 1,
            values: vec![c(1.)],
        };
        let error = workspace.contract_batch(&moments, &numerator).unwrap_err();
        assert!(error.to_string().contains("conventions or tensor weights"));
        assert_eq!(workspace.last_evaluated_rows(), 0);

        moments.metadata =
            crate::ResultMetadata::new(Convention::OneLoop, TensorWeights::Multinomial, None);
        let error = workspace.contract_batch(&moments, &numerator).unwrap_err();
        assert!(error.to_string().contains("conventions or tensor weights"));
        moments.metadata =
            crate::ResultMetadata::new(Convention::MadLoop, TensorWeights::Multinomial, None);
        assert_eq!(
            workspace
                .contract_batch(&moments, &numerator)
                .unwrap()
                .lane(0)
                .unwrap(),
            &[c(1.); 4]
        );
    }

    #[test]
    fn rejects_equal_component_counts_with_different_dual_shapes() {
        let basis = tensor_basis(0);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::OneLoop,
            TensorWeights::Components,
            &basis,
        )
        .unwrap();
        let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
        let mut workspace =
            ContractionWorkspace::new_dual(&plan, 0., HyperDual::<f64>::new(shape.clone()))
                .unwrap();
        let mut moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                Convention::OneLoop,
                TensorWeights::Components,
                Some(vec![vec![0], vec![1], vec![2]]),
            ),
            basis,
            components: 3,
            values: [5., 7., 11.].map(c).repeat(4),
            statuses: vec![LaneStatus::Success],
            fallback_used: vec![false],
            analytic_used: vec![false],
            primal_envelopes: vec![None; 1],
            primal_envelope_errors: vec![None; 1],
        };
        let numerator = NumeratorBatch {
            rows: 1,
            components: 3,
            values: [2., 3., 4.].map(c).to_vec(),
        };
        let error = workspace.contract_batch(&moments, &numerator).unwrap_err();
        assert!(error.to_string().contains("Dual shapes differ"));
        assert_eq!(workspace.last_evaluated_rows(), 0);
        moments.metadata =
            crate::ResultMetadata::new(Convention::OneLoop, TensorWeights::Components, Some(shape));
        let result = workspace.contract_batch(&moments, &numerator).unwrap();
        for slot in result.lane(0).unwrap().chunks_exact(3) {
            assert_eq!(slot, &[c(10.), c(29.), c(42.)]);
        }
    }

    #[test]
    fn rejects_malformed_dual_metadata_before_contraction_compilation() {
        let basis = tensor_basis(0);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::OneLoop,
            TensorWeights::Components,
            &basis,
        )
        .unwrap();
        // Symbolica's dynamic constructor admits duplicate multi-indices;
        // these are ambiguous even though the component count is consistent.
        let duplicate = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![1]]);
        let error = match ContractionWorkspace::new_dual(&plan, 0., duplicate) {
            Ok(_) => panic!("duplicate Dual components must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unique"));
    }

    #[test]
    fn sparse_contraction_corrects_weights_and_madloop_sign_in_each_lane() {
        let basis = tensor_basis(2);
        let terms = [Monomial([1, 0, 0, 0]), Monomial([1, 1, 0, 0])];
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::MadLoop,
            TensorWeights::Multinomial,
            &terms,
        )
        .unwrap();
        let mut workspace = ContractionWorkspace::new(&plan, 0.).unwrap();
        let mut moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                plan.convention,
                plan.weights,
                workspace.shape.clone(),
            ),
            basis,
            components: 1,
            values: vec![c(0.); 3 * 15 * 4],
            statuses: vec![
                LaneStatus::Success,
                LaneStatus::MasterFailure("fixture failure".into()),
                LaneStatus::Success,
            ],
            fallback_used: vec![false, true, true],
            analytic_used: vec![false, false, true],
            primal_envelopes: vec![None; 3],
            primal_envelope_errors: vec![None; 3],
        };
        for (row, values) in [[-5., 14.], [-7., 18.], [-11., 26.]].iter().enumerate() {
            for (term, &position) in plan.positions.iter().enumerate() {
                for pole in 0..4 {
                    moments.values[row * 60 + position * 4 + pole] = c(values[term]);
                }
            }
        }
        let numerator = NumeratorBatch {
            rows: 3,
            components: 1,
            values: [2., 3., 2., 3., 2., 3.].map(c).to_vec(),
        };
        let result = workspace.contract_batch(&moments, &numerator).unwrap();
        assert_eq!(result.lane(0).unwrap(), &[c(31.); 4]);
        assert_eq!(result.lane(2).unwrap(), &[c(61.); 4]);
        assert_eq!(result.statuses[1], moments.statuses[1]);
        assert_eq!(result.fallback_used, moments.fallback_used);
        assert_eq!(result.analytic_used, moments.analytic_used);
        assert!(result.lane(1).is_err());
        assert_eq!(workspace.last_evaluated_rows(), 2);
    }

    #[test]
    fn second_order_dual_products_and_invalid_coefficients_remain_separate_lanes() {
        let basis = tensor_basis(0);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::OneLoop,
            TensorWeights::Components,
            &basis,
        )
        .unwrap();
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = ContractionWorkspace::new_dual(&plan, 0., shape).unwrap();
        let moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                plan.convention,
                plan.weights,
                workspace.shape.clone(),
            ),
            basis,
            components: 3,
            values: [5., 7., 11.].map(c).repeat(12),
            statuses: vec![LaneStatus::Success; 3],
            fallback_used: vec![false; 3],
            analytic_used: vec![false; 3],
            primal_envelopes: vec![None; 3],
            primal_envelope_errors: vec![None; 3],
        };
        let numerator = NumeratorBatch {
            rows: 3,
            components: 3,
            values: [2., 3., 4., f64::NAN, 0., 0., 2., 3., 4.].map(c).to_vec(),
        };
        let result = workspace.contract_batch(&moments, &numerator).unwrap();
        for row in [0, 2] {
            for slot in result.lane(row).unwrap().chunks_exact(3) {
                assert_eq!(slot, &[c(10.), c(29.), c(63.)]);
            }
        }
        assert!(matches!(result.statuses[1], LaneStatus::InvalidInput(_)));
        assert_eq!(workspace.last_evaluated_rows(), 2);
        let typed: Vec<_> = [[2., 3., 4.], [f64::NAN, 0., 0.], [2., 3., 4.]]
            .into_iter()
            .map(|values| {
                let mut value = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
                value.values = values.to_vec();
                value
            })
            .collect();
        let typed_result = workspace
            .contract_real_dual_batch(&moments, &typed)
            .unwrap();
        for row in [0, 2] {
            assert_eq!(typed_result.lane(row).unwrap(), result.lane(row).unwrap());
        }
        assert_eq!(typed_result.statuses, result.statuses);
    }

    #[test]
    fn empty_batches_and_zero_polynomials_preserve_failure_status() {
        let basis = tensor_basis(0);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::OneLoop,
            TensorWeights::Components,
            &[],
        )
        .unwrap();
        let mut workspace = ContractionWorkspace::new(&plan, 0.).unwrap();
        let empty = BatchResult {
            metadata: crate::ResultMetadata::new(
                plan.convention,
                plan.weights,
                workspace.shape.clone(),
            ),
            basis: basis.clone(),
            components: 1,
            values: Vec::new(),
            statuses: Vec::new(),
            fallback_used: Vec::new(),
            analytic_used: Vec::new(),
            primal_envelopes: vec![None; 0],
            primal_envelope_errors: vec![None; 0],
        };
        assert_eq!(
            workspace
                .contract_batch(
                    &empty,
                    &NumeratorBatch {
                        rows: 0,
                        components: 1,
                        values: Vec::new()
                    }
                )
                .unwrap()
                .rows(),
            0
        );
        let moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                plan.convention,
                plan.weights,
                workspace.shape.clone(),
            ),
            basis,
            components: 1,
            values: vec![c(0.); 8],
            statuses: vec![LaneStatus::Success, LaneStatus::Unstable("fixture".into())],
            fallback_used: vec![false; 2],
            analytic_used: vec![false; 2],
            primal_envelopes: vec![None; 2],
            primal_envelope_errors: vec![None; 2],
        };
        let result = workspace
            .contract_batch(
                &moments,
                &NumeratorBatch {
                    rows: 2,
                    components: 1,
                    values: Vec::new(),
                },
            )
            .unwrap();
        assert_eq!(result.lane(0).unwrap(), &[c(0.); 4]);
        assert!(result.lane(1).is_err());
    }

    #[test]
    fn arbitrary_precision_coefficients_do_not_round_through_binary64() {
        let basis = tensor_basis(0);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::OneLoop,
            TensorWeights::Components,
            &basis,
        )
        .unwrap();
        let prototype = Float::with_val(256, 0);
        let mut workspace = ContractionWorkspace::new(&plan, prototype.clone()).unwrap();
        let tiny = prototype.from_rational(&Rational::from((1, 1_000_000_000_000_000_000_i64)));
        let one = Float::with_val(256, 1);
        let coefficient = one.clone() + tiny.clone();
        let complex = |value| Complex::new(value, prototype.clone());
        let moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                plan.convention,
                plan.weights,
                workspace.shape.clone(),
            ),
            basis,
            components: 1,
            values: vec![complex(one.clone()); 4],
            statuses: vec![LaneStatus::Success],
            fallback_used: vec![false],
            analytic_used: vec![false],
            primal_envelopes: vec![None; 1],
            primal_envelope_errors: vec![None; 1],
        };
        let numerator = NumeratorBatch {
            rows: 1,
            components: 1,
            values: vec![complex(coefficient.clone())],
        };
        let result = workspace.contract_batch(&moments, &numerator).unwrap();
        assert!(result.lane(0).unwrap()[3].re > one);
        assert_eq!(result.lane(0).unwrap()[3].re, coefficient);
    }

    #[test]
    fn static_complex_dual_coefficients_preserve_doublefloat_low_component() {
        let basis = tensor_basis(0);
        let plan = ContractionPlan::from_basis(
            &basis,
            Convention::OneLoop,
            TensorWeights::Components,
            &basis,
        )
        .unwrap();
        let zero = DoubleFloat::from(0.);
        let tiny = zero.from_rational(&Rational::from((1_i64, 1_i64 << 54)));
        let complex = |real: i64, imag: i64| Complex::new(zero.from_i64(real), zero.from_i64(imag));
        let mut coefficient = StaticCoefficient::new_variable(0, complex(2, 1));
        coefficient.values[0].re += tiny;
        coefficient.values[1] = complex(3, 4);
        let mut workspace =
            ContractionWorkspace::new_dual(&plan, zero, coefficient.clone()).unwrap();
        let moments = BatchResult {
            metadata: crate::ResultMetadata::new(
                plan.convention,
                plan.weights,
                workspace.shape.clone(),
            ),
            basis,
            components: 2,
            values: [complex(5, 2), complex(7, -1)].repeat(4),
            statuses: vec![LaneStatus::Success],
            fallback_used: vec![false],
            analytic_used: vec![false],
            primal_envelopes: vec![None; 1],
            primal_envelope_errors: vec![None; 1],
        };
        let result = workspace
            .contract_typed_dual_batch(&moments, &[coefficient])
            .unwrap();
        let expected = [
            Complex::new(
                zero.from_i64(8) + zero.from_i64(5) * tiny,
                zero.from_i64(9) + zero.from_i64(2) * tiny,
            ),
            Complex::new(
                zero.from_i64(22) + zero.from_i64(7) * tiny,
                zero.from_i64(31) - tiny,
            ),
        ];
        for slot in result.lane(0).unwrap().chunks_exact(2) {
            assert_eq!(slot, &expected);
        }
        assert_ne!(result.lane(0).unwrap()[0].re, zero.from_i64(8));
    }
}
