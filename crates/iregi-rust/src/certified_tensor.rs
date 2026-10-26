//! Opt-in supplied-product Cartesian tensors with exact full-jet compatibility.
//!
//! This opt-in interface supports N6 tensors through rank six, including positive powers and dimension shifts.
//! It retains the original Cartesian shifts for moments and supplied products
//! for scalar/master arguments. Each batch must pass the existing rank-four
//! Gram certificate and every relative Cartesian/product polynomial identity.
//! No numerical projection, hidden shift reconstruction or unchecked plan is
//! exposed. Generalized requests retain genuine scalar IBP and its shifted-master guards.
//!
//! Numerical qualification is incomplete. Every returned lane retains its
//! stability status; accepting a numeric/Dual type is not an achieved-accuracy
//! statement.

use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::DualNumberStructure, float::Complex},
    evaluate::EvaluationDomain,
};

use crate::{
    BatchResult, CertificateExecutionStats, ContractionPlan, ExactInvariantScalar,
    ExecutionProfile, ExecutionStats, InputLayout, InputPolicy, InvariantInput, KinematicsBatch,
    ReductionError, ReductionSettings,
    basis::Monomial,
    certified_scalar::{CertifiedScalarReductionPlan, CertifiedScalarWorkspace},
};

pub(crate) fn settings_supported(settings: &ReductionSettings) -> bool {
    settings.propagators == 6
        && settings.rank <= 6
        && settings.powers.len() == 6
        && settings.powers.iter().all(|&power| power > 0)
        && settings.input_policy == InputPolicy::Preserve
        && settings.invariants == InvariantInput::SuppliedScalarProducts
}

/// The packed PV engine has a deliberately separate ordinary domain. Hybrid
/// and generalized requests retain their existing algorithms and fallback.
pub(crate) fn packed_settings_supported(settings: &ReductionSettings) -> bool {
    settings_supported(settings) && settings.algorithm == crate::Algorithm::PassarinoVeltman
        && settings.dimension_shift == 0 && settings.powers == [1;6]
}

/// Symbolica owns every subtraction/product and the later complete Dualization.
/// Parameters are original p, then upper-triangle supplied G. Masses and scale
/// are independent of this equality and retain the ordinary runtime guards.
pub(crate) fn compatibility_constraints(n: usize) -> (Vec<Atom>, Vec<Atom>) {
    let shifts: Vec<[Atom; 4]> = (0..n)
        .map(|i| {
            std::array::from_fn(|mu| {
                Atom::var(symbolica::symbol!(format!(
                    "iregi_tensor_compatibility::p{i}_{mu}"
                )))
            })
        })
        .collect();
    let mut parameters = shifts.iter().flatten().cloned().collect::<Vec<_>>();
    let mut products = vec![vec![Atom::new(); n]; n];
    for i in 0..n {
        for j in i..n {
            let value = Atom::var(symbolica::symbol!(format!(
                "iregi_tensor_compatibility::g{i}_{j}"
            )));
            products[i][j] = value.clone();
            products[j][i] = value.clone();
            parameters.push(value);
        }
    }
    let mut residuals = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in i..n {
            let relative_g = &products[i][j] - &products[0][i] - &products[0][j] + &products[0][0];
            let cartesian = (0..4)
                .map(|mu| {
                    Atom::num(if mu == 0 { 1 } else { -1 })
                        * (&shifts[i][mu] - &shifts[0][mu])
                        * (&shifts[j][mu] - &shifts[0][mu])
                })
                .sum::<Atom>();
            residuals.push(relative_g - cartesian);
        }
    }
    (parameters, residuals)
}

/// Immutable checked tensor geometry; structural caches contain no lane proof.
///
/// Input is the existing full supplied-G layout: N four-vectors, N(N+1)/2 upper
/// triangle products including diagonals, N squared masses and real positive mu.
/// The constructor explicitly selects supplied-product input. All other caller
/// settings, including requested algorithm/convention/weights, are preserved.
/// The domain is N6, rank<=6, positive powers, supported scalar dimension shifts and preserved inputs.
/// Only a regular exact rank-four chart is admitted. Lower-span strata remain
/// explicit failures; this wrapper never reconstructs momenta from covectors.
#[derive(Clone, Debug)]
pub struct CertifiedTensorReductionPlan {
    inner: CertifiedScalarReductionPlan,
    layout: InputLayout,
    basis: Vec<Monomial>,
}

impl CertifiedTensorReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let inner = CertifiedScalarReductionPlan::new_tensor(settings)?;
        Ok(Self {
            layout: inner.tensor_input_layout(),
            basis: inner.basis(),
            inner,
        })
    }
    pub fn settings(&self) -> &ReductionSettings {
        self.inner.settings()
    }
    pub fn input_layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn basis(&self) -> &[Monomial] {
        &self.basis
    }
    pub fn parameter_count(&self) -> usize {
        self.inner.argument_count()
    }
    pub fn contraction_plan(
        &self,
        monomials: &[Monomial],
    ) -> Result<ContractionPlan, ReductionError> {
        self.inner.contraction_plan(monomials)
    }
}

/// Two distinct exact checks precede every accepted chart dispatch. The work
/// counts include all evaluated lane/refinement passes, not derivative lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorCertificateExecutionStats {
    pub cartesian: CertificateExecutionStats,
    pub gram: CertificateExecutionStats,
}

/// Owned shared checked-chart runtime. Both certificates and chart workspaces
/// have independent numerical scratch; only immutable structural plans/cache
/// entries are shared. Each input lane is recertified on every call, including
/// every component of an explicit order-zero/static/dynamic Dual shape.
#[derive(Clone)]
pub struct CertifiedTensorWorkspace<T: NativeFloat + ExactInvariantScalar> {
    inner: CertifiedScalarWorkspace<T>,
}

impl<T: NativeFloat + ExactInvariantScalar> CertifiedTensorWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &CertifiedTensorReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Ok(Self {
            inner: CertifiedScalarWorkspace::new(&plan.inner, prototype)?,
        })
    }
    pub fn new_dual<D: DualNumberStructure>(
        plan: &CertifiedTensorReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        Ok(Self {
            inner: CertifiedScalarWorkspace::new_dual(&plan.inner, prototype, dual)?,
        })
    }
    pub fn component_count(&self) -> usize {
        self.inner.component_count()
    }
    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.inner.dual_shape()
    }
    pub fn set_certificate_precision_budget(
        &mut self,
        initial: u32,
        maximum: u32,
    ) -> Result<(), ReductionError> {
        self.inner
            .set_certificate_precision_budget(initial, maximum)
    }
    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        self.inner.set_accuracy_precision(bits)
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.inner.set_profiling(enabled);
    }
    pub fn last_execution(&self) -> ExecutionStats {
        self.inner.last_execution()
    }
    /// Actual nested ordinary-PV work for every dispatched checked chart.
    /// Empty for scalar-IBP/generalized routes; cleared on every call.
    pub fn last_packed_execution(&self)->&[crate::SixPointExecutionStats] {
        self.inner.packed_execution()
    }
    pub fn last_certificate_execution(&self) -> TensorCertificateExecutionStats {
        TensorCertificateExecutionStats {
            cartesian: self.inner.compatibility_execution(),
            gram: self.inner.last_certificate_execution(),
        }
    }
    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        self.inner.last_profile()
    }
    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        self.inner.last_diagnostics()
    }
    pub fn reduce_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.inner.reduce_tensor_batch(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Algorithm, ReductionPlan,
        exact_zero_certificate::{ExactZeroCertificate, ExactZeroConstraintPlan},
    };
    use symbolica::domains::{dual::HyperDual, rational::Rational};

    fn c(value: i64) -> Complex<Rational> {
        Complex::new(Rational::from(value), Rational::from(0))
    }
    fn geometry(n: usize) -> Vec<Complex<Rational>> {
        let points = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ];
        let mut row = points[..n]
            .iter()
            .flatten()
            .map(|&v| c(v))
            .collect::<Vec<_>>();
        for i in 0..n {
            for j in i..n {
                let a = points[i];
                let b = points[j];
                row.push(c(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]));
            }
        }
        row
    }

    #[test]
    fn checked_tensor_domain_keeps_default_guard_and_generalized_boundary() {
        for n in [6] {
            for rank in [0, 1, 6] {
                for algorithm in [
                    Algorithm::Ibp,
                    Algorithm::PassarinoVeltman,
                    Algorithm::Hybrid,
                ] {
                    let mut settings = ReductionSettings::new(n, rank);
                    settings.invariants = InvariantInput::SuppliedScalarProducts;
                    settings.algorithm = algorithm;
                    assert!(settings_supported(&settings));
                    assert!(matches!(
                        ReductionPlan::new(settings.clone()),
                        Err(ReductionError::Unsupported(_))
                    ));
                    for h in [-1, 1] {
                        let mut invalid = settings.clone();
                        invalid.dimension_shift = h;
                        assert!(settings_supported(&invalid));
                    }
                    let mut dotted = settings.clone();
                    dotted.powers[0] = 2;
                    assert!(settings_supported(&dotted));
                    dotted.powers[0] = 0;
                    assert!(!settings_supported(&dotted));
                    let mut high = settings.clone();
                    high.rank = 7;
                    assert!(!settings_supported(&high));
                }
            }
        }
    }

    #[test]
    fn rank_four_invariant_scale_does_not_imply_cartesian_compatibility() {
        for n in [6, 7] {
            let (parameters, residuals) = compatibility_constraints(n);
            let mut proof = ExactZeroConstraintPlan::new(&parameters, &residuals)
                .unwrap()
                .instantiate();
            let good = geometry(n);
            let mut bad = good.clone();
            for value in &mut bad[4 * n..] {
                value.re *= Rational::from(2);
            }
            // This is still a rank-four Gram with identical dependencies, but
            // its lengths disagree with the caller's Cartesian metric.
            let result = proof
                .check_batch(&[good.clone(), bad, good].concat(), 3)
                .unwrap();
            assert_eq!(result[0], ExactZeroCertificate::Zero);
            assert!(matches!(
                result[1],
                ExactZeroCertificate::Nonzero { component: 0, .. }
            ));
            assert_eq!(result[2], ExactZeroCertificate::Zero);
        }
    }

    #[test]
    fn compatibility_checks_entire_ordered_mixed_shape_and_preserves_relative_translation() {
        let n = 6;
        let shape = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let (parameters, residuals) = compatibility_constraints(n);
        let plan = ExactZeroConstraintPlan::new_dual(
            &parameters,
            &residuals,
            &HyperDual::<u8>::new(shape),
        )
        .unwrap();
        let mut proof = plan.instantiate();
        let mut good = geometry(n)
            .into_iter()
            .flat_map(|value| [value, c(0), c(0), c(0)])
            .collect::<Vec<_>>();
        // Same full jet translation for every vector; relative momenta and
        // supplied products stay constant. Absolute G equality is unnecessary.
        for i in 0..n {
            good[4 * (4 * i) + 1] = c(2);
            good[4 * (4 * i + 1) + 2] = c(3);
            good[4 * (4 * i + 2) + 3] = c(5);
        }
        let mut bad = good.clone();
        bad[4 * (4 * n) + 3] = c(1); // mixed G00 only; primal unchanged
        let actual = proof.check_batch(&[good, bad].concat(), 2).unwrap();
        assert_eq!(actual[0], ExactZeroCertificate::Zero);
        assert!(matches!(
            actual[1],
            ExactZeroCertificate::Nonzero { component: 3, .. }
        ));
    }

    #[test]
    fn tiny_exact_discrepancy_is_unresolved_or_rejected_never_projected() {
        let n = 6;
        let (parameters, residuals) = compatibility_constraints(n);
        let mut proof = ExactZeroConstraintPlan::new(&parameters, &residuals)
            .unwrap()
            .instantiate();
        let mut row = geometry(n);
        row[4 * n].re += Rational::from((1, 2)).pow(300);
        proof.set_precision_budget(128, 128).unwrap();
        assert!(!matches!(
            proof.check_batch(&row, 1).unwrap()[0],
            ExactZeroCertificate::Zero
        ));
        proof.set_precision_budget(128, 1024).unwrap();
        assert!(matches!(
            proof.check_batch(&row, 1).unwrap()[0],
            ExactZeroCertificate::Nonzero { .. }
        ));
    }
    #[test]
    #[ignore = "genuine N6 checked tensor/master evaluation; run only under a separate finite qualification grant"]
    fn checked_tensor_full_rows_match_coordinate_api_and_isolate_bad_neighbors() {
        use crate::{ContractionWorkspace, NumeratorBatch, TensorReductionPlan, TensorWorkspace};
        use symbolica::domains::float::{Float, FloatLike, SingleFloat};
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../validation/highpoint-seven-point-current-cargo/fixture.json"
        ))
        .unwrap();
        let prototype = Float::new(128);
        let exact = |text: &str| {
            let (n, d) = text.split_once('/').unwrap_or((text, "1"));
            Rational::from((
                n.parse::<symbolica::domains::integer::Integer>().unwrap(),
                d.parse::<symbolica::domains::integer::Integer>().unwrap(),
            ))
        };
        let numeric = |value: &Complex<Rational>| {
            Complex::new(
                prototype.from_rational(&value.re),
                prototype.from_rational(&value.im),
            )
        };
        for n in [6] {
            let propagators = fixture["samples"][0]["propagators"].as_array().unwrap();
            let points = propagators[..n]
                .iter()
                .map(|p| {
                    std::array::from_fn::<_, 4, _>(|mu| exact(p["shift"][mu].as_str().unwrap()))
                })
                .collect::<Vec<_>>();
            let mut row = points
                .iter()
                .flatten()
                .cloned()
                .map(|value| Complex::new(value, Rational::from(0)))
                .collect::<Vec<_>>();
            for i in 0..n {
                for j in i..n {
                    let value = &points[i][0] * &points[j][0]
                        - &points[i][1] * &points[j][1]
                        - &points[i][2] * &points[j][2]
                        - &points[i][3] * &points[j][3];
                    row.push(Complex::new(value, Rational::from(0)));
                }
            }
            row.extend(propagators[..n].iter().map(|p| {
                Complex::new(
                    exact(p["mass_squared"][0].as_str().unwrap()),
                    exact(p["mass_squared"][1].as_str().unwrap()),
                )
            }));
            row.push(c(3));
            let mut requested = ReductionSettings::new(n, 1);
            requested.algorithm = Algorithm::PassarinoVeltman;
            let checked = CertifiedTensorReductionPlan::new(requested.clone()).unwrap();
            let mut scratch = CertifiedTensorWorkspace::new(&checked, prototype.clone()).unwrap();
            scratch.set_accuracy_precision(53).unwrap();
            let valid = row.iter().map(numeric).collect::<Vec<_>>();
            let mut bad = valid.clone();
            // Same rank-four relative Gram, different Cartesian lengths.
            for value in &mut bad[4 * n..4 * n + n * (n + 1) / 2] {
                value.re *= prototype.from_i64(2);
            }
            let input = KinematicsBatch {
                rows: 3,
                components: 1,
                values: [valid.clone(), bad, valid.clone()].concat(),
            };
            let before = input.values.clone();
            let result = scratch.reduce_batch(&input).unwrap();
            assert_eq!(input.values, before);
            assert!(result.lane(0).is_ok() && result.lane(1).is_err() && result.lane(2).is_ok(),
                "statuses={:?}; diagnostics={:?}", result.statuses, scratch.last_diagnostics());
            assert_eq!(result.lane(0).unwrap(), result.lane(2).unwrap());
            assert!(matches!(
                result.statuses[1],
                crate::LaneStatus::InvalidInput(_)
            ));
            assert!(scratch.last_diagnostics()[0].iter().any(|note|
                note.contains("requested PassarinoVeltman; actual")));
            let counts = scratch.last_certificate_execution();
            assert_eq!(counts.cartesian.input_rows, 3);
            assert_eq!(counts.gram.reduction_batches, 1);
            assert_eq!(scratch.last_execution().input_rows, 3);
            let packed=scratch.last_packed_execution();
            assert_eq!(packed.len(),1);
            assert_eq!(packed[0].child_batch_calls,1);assert_eq!(packed[0].packed_child_rows,12);
            assert_eq!(packed[0].parent_batch_calls,1);assert_eq!(packed[0].packed_parent_rows,2);
            assert_eq!(scratch.last_execution().reduction_rows,
                packed[0].child_execution.reduction_rows+packed[0].parent_execution.reduction_rows);
            let coordinate = TensorReductionPlan::new(requested).unwrap();
            let mut reference = TensorWorkspace::new(&coordinate, prototype.clone()).unwrap();
            reference.set_accuracy_precision(53).unwrap();
            let coordinate_row = valid[..4 * n]
                .iter()
                .chain(valid[4 * n + n * (n + 1) / 2..].iter())
                .cloned()
                .collect();
            let expected = reference
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: coordinate_row,
                })
                .unwrap();
            let tolerance = prototype.from_rational(&Rational::from((1i64, 10_000_000_000i64)));
            for (actual, expected) in result
                .lane(0)
                .unwrap()
                .iter()
                .zip(expected.lane(0).unwrap())
            {
                // The dimensionless retained rank-one family is checked here
                // against its existing coordinate API; a new independent
                // defining-integral/quad anchor remains a separate gate.
                let difference = actual - expected;
                let error = difference.re.clone() * &difference.re + difference.im.clone() * &difference.im;
                let unit = prototype.one();
                let norm = expected.re.clone() * &expected.re + expected.im.clone() * &expected.im;
                let scale = if norm > unit { norm } else { unit };
                assert!(error <= tolerance.clone() * &tolerance * scale);
            }
            let support = [Monomial([0; 4]), Monomial([1, 0, 0, 0])];
            let contraction = checked.contraction_plan(&support).unwrap();
            let mut contract = ContractionWorkspace::new(&contraction, prototype.clone()).unwrap();
            let coefficients = NumeratorBatch {
                rows: 3,
                components: 1,
                values: Vec::new(),
            };
            // The qualified contraction API and metadata are used unchanged;
            // explicit vector construction avoids confusing rows with terms.
            let mut coefficients = coefficients;
            coefficients.values = (0..3)
                .flat_map(|_| [numeric(&c(2)), numeric(&c(3))])
                .collect();
            let contracted = contract.contract_batch(&result, &coefficients).unwrap();
            assert!(contracted.lane(0).is_ok() && contracted.lane(1).is_err());
            let empty = scratch
                .reduce_batch(&KinematicsBatch {
                    rows: 0,
                    components: 1,
                    values: Vec::new(),
                })
                .unwrap();
            assert!(empty.statuses.is_empty());
            assert_eq!(scratch.last_execution().input_rows, 0);
            assert!(scratch.last_diagnostics().is_empty());
            assert!(scratch.last_packed_execution().is_empty());
            assert!(
                scratch
                    .reduce_batch(&KinematicsBatch {
                        rows: 1,
                        components: 1,
                        values: Vec::new()
                    })
                    .is_err()
            );
            let recovered = scratch
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: valid,
                })
                .unwrap();
            assert_eq!(recovered.lane(0).unwrap(), result.lane(0).unwrap());
        }
    }
}
