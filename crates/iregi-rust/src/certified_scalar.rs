//! Opt-in scalar reduction with exact full-Taylor external-Gram certification.
//!
//! A regular rank-four chart supplies linear denominator relations. The
//! certificate never changes the input, and its covectors are never exposed as
//! Cartesian momenta. Accepted samples run together in each chart's existing
//! instruction-major reducer.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use iregi_masters::NativeFloat;
use symbolica::{
    domains::{
        dual::{DualNumberStructure, HyperDual},
        float::Complex,
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

use crate::{
    Algorithm, BatchResult, ExecutionProfile, ExecutionStats, InvariantInput, KinematicsBatch,
    LaneStatus, ReductionError, ReductionPlan, ReductionSettings, ResultMetadata,
    ScalarKinematicsBatch, Workspace,
    types::PrimalEnvelopeRoute,
    basis::Monomial,
    invariant_certificate::{
        ExactInvariantScalar, InvariantCertificate, InvariantCertificatePlan,
        InvariantCertificateWorkspace,
    },
};

type Shape = Option<Vec<Vec<usize>>>;
type Prepared<P> = Arc<OnceLock<Result<Arc<P>, String>>>;
type ChartCache = Arc<Mutex<HashMap<[usize; 4], Prepared<ReductionPlan>>>>;
type CertificateCache = Arc<Mutex<HashMap<Shape, Prepared<InvariantCertificatePlan>>>>;

/// Opt-in rank-zero IBP plan for six or seven supplied-G propagators.
///
/// Every accepted sample has an exactly certified regular rank-four relative
/// Gram matrix, including every Taylor component. Incompatible input is an
/// invalid lane; unresolved certificates and absent regular charts are unstable
/// lanes. They never produce successful zero results. Scalar analytic
/// continuation is allowed: real Minkowski inertia is not imposed.
///
/// This deliberately narrow interface does not expose the unchecked internal
/// chart plans. Use [`crate::ScalarReductionPlan`] for up to five propagators.
#[derive(Clone)]
pub struct CertifiedScalarReductionPlan {
    settings: ReductionSettings,
    charts: ChartCache,
    certificates: CertificateCache,
}

impl std::fmt::Debug for CertifiedScalarReductionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CertifiedScalarReductionPlan")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl CertifiedScalarReductionPlan {
    pub fn new(mut settings: ReductionSettings) -> Result<Self, ReductionError> {
        if settings.rank != 0 {
            return Err(ReductionError::InvalidInput(
                "certified scalar reduction requires rank zero".into(),
            ));
        }
        if !(6..=7).contains(&settings.propagators) || settings.algorithm != Algorithm::Ibp {
            return Err(ReductionError::Unsupported(
                "certified supplied-G reduction currently requires six or seven propagators and IBP".into(),
            ));
        }
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        let plan = Self {
            settings,
            charts: Arc::default(),
            certificates: Arc::default(),
        };
        // Validate settings and compile one immutable chart at preparation
        // time. Other charts are prepared once on first demand.
        plan.chart([1, 2, 3, 4])?;
        Ok(plan)
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.settings
    }

    pub fn argument_count(&self) -> usize {
        let n = self.settings.propagators;
        n * (n + 1) / 2 + n + 1
    }

    fn chart(&self, basis: [usize; 4]) -> Result<Arc<ReductionPlan>, ReductionError> {
        let entry = self
            .charts
            .lock()
            .map_err(|_| preparation("certified scalar chart cache poisoned"))?
            .entry(basis)
            .or_default()
            .clone();
        entry
            .get_or_init(|| {
                ReductionPlan::new_certified_scalar(self.settings.clone(), basis)
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(preparation)
    }

    fn certificate(&self, shape: &Shape) -> Result<Arc<InvariantCertificatePlan>, ReductionError> {
        let entry = self
            .certificates
            .lock()
            .map_err(|_| preparation("invariant certificate cache poisoned"))?
            .entry(shape.clone())
            .or_default()
            .clone();
        entry
            .get_or_init(|| {
                match shape {
                    None => InvariantCertificatePlan::new(self.settings.propagators),
                    Some(shape) => InvariantCertificatePlan::new_dual(
                        self.settings.propagators,
                        &HyperDual::<u8>::new(shape.clone()),
                    ),
                }
                .map(Arc::new)
                .map_err(|error| error.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(preparation)
    }
}

/// Work performed by the full-Taylor certificate before numeric reduction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CertificateExecutionStats {
    /// Input samples, including rejected input and unresolved certificates.
    pub input_rows: usize,
    /// Active sample rows over all precision-refinement passes.
    pub evaluated_rows: usize,
    pub passes: usize,
    /// Calls to chart-specific batch reduction, independently of sample count.
    pub reduction_batches: usize,
}

/// Owned numeric scratch for [`CertifiedScalarReductionPlan`].
///
/// Numeric workspaces and certificate refinement buffers are never shared.
/// Structural plans are shared across clones, precisions and numeric values.
#[derive(Clone)]
pub struct CertifiedScalarWorkspace<T: NativeFloat + ExactInvariantScalar> {
    plan: CertifiedScalarReductionPlan,
    prototype: T,
    shape: Shape,
    components: usize,
    certificate: InvariantCertificateWorkspace,
    charts: HashMap<[usize; 4], Workspace<T>>,
    accuracy_bits: u32,
    profiling: bool,
    stats: ExecutionStats,
    certificate_stats: CertificateExecutionStats,
    profile: ExecutionProfile,
    diagnostics: Vec<Vec<String>>,
}

impl<T: NativeFloat + ExactInvariantScalar> CertifiedScalarWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &CertifiedScalarReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Self::prepare(plan, prototype, None)
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &CertifiedScalarReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(&dual).map_err(preparation)?;
        let shape = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect();
        Self::prepare(plan, prototype, Some(shape))
    }

    fn prepare(
        plan: &CertifiedScalarReductionPlan,
        prototype: T,
        shape: Shape,
    ) -> Result<Self, ReductionError> {
        let accuracy_bits = prototype.get_precision();
        if accuracy_bits == 0 {
            return Err(preparation("working precision must be positive"));
        }
        Ok(Self {
            plan: plan.clone(),
            prototype,
            components: shape.as_ref().map_or(1, Vec::len),
            certificate: plan.certificate(&shape)?.instantiate(),
            shape,
            charts: HashMap::new(),
            accuracy_bits,
            profiling: false,
            stats: ExecutionStats::default(),
            certificate_stats: CertificateExecutionStats::default(),
            profile: ExecutionProfile::default(),
            diagnostics: Vec::new(),
        })
    }

    pub fn component_count(&self) -> usize {
        self.components
    }

    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.shape.as_deref()
    }

    /// Limit exact-certificate refinement. Exhaustion returns an unstable lane,
    /// never acceptance by a numerical tolerance. Defaults: 128 to 16384 bits.
    pub fn set_certificate_precision_budget(
        &mut self,
        initial_bits: u32,
        maximum_bits: u32,
    ) -> Result<(), ReductionError> {
        self.certificate
            .set_precision_budget(initial_bits, maximum_bits)
    }

    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(
                "accuracy precision must be positive and no greater than working precision".into(),
            ));
        }
        for workspace in self.charts.values_mut() {
            workspace.set_accuracy_precision(bits)?;
        }
        self.accuracy_bits = bits;
        Ok(())
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        for workspace in self.charts.values_mut() {
            workspace.set_profiling(enabled);
        }
    }

    pub fn last_execution(&self) -> ExecutionStats {
        self.stats
    }

    pub fn last_certificate_execution(&self) -> CertificateExecutionStats {
        self.certificate_stats
    }

    /// The argument stage includes exact certificate checking and chart setup.
    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }

    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }

    pub fn reduce_batch(
        &mut self,
        input: &ScalarKinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let start = self.profiling.then(Instant::now);
        self.stats = ExecutionStats {
            input_rows: input.rows,
            ..Default::default()
        };
        self.certificate_stats = CertificateExecutionStats {
            input_rows: input.rows,
            ..Default::default()
        };
        self.profile = ExecutionProfile::default();
        self.diagnostics.clear();
        if input.components != self.components {
            return Err(ReductionError::InvalidInput(
                "scalar input Taylor layout does not match workspace".into(),
            ));
        }
        let width = product(self.plan.argument_count(), self.components)?;
        let expected = product(width, input.rows)?;
        if input.values.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "scalar batch expects {expected} values, received {}",
                input.values.len(),
            )));
        }
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let output_width = product(4, self.components)?;
        let output_len = product(output_width, input.rows)?;
        self.diagnostics.resize_with(input.rows, Vec::new);
        let mut result = BatchResult {
            basis: vec![Monomial([0; 4])],
            components: self.components,
            metadata: ResultMetadata::new(
                self.plan.settings.convention,
                self.plan.settings.tensor_weights,
                self.shape.clone(),
            ),
            values: vec![zero.clone(); output_len],
            // Success is assigned only after both certification and reduction.
            statuses: vec![
                LaneStatus::Unstable("invariant certificate not evaluated".into());
                input.rows
            ],
            fallback_used: vec![false; input.rows],
            analytic_used: vec![false; input.rows],
            primal_envelopes: Vec::new(),
            primal_envelope_errors: Vec::new(),
        };
        let n = self.plan.settings.propagators;
        let gram_width = product(n * (n + 1) / 2, self.components)?;
        let mut exact = Vec::with_capacity(product(gram_width, input.rows)?);
        let mut lanes = Vec::with_capacity(input.rows);
        for (lane, row) in input.values.chunks_exact(width).enumerate() {
            let previous_len = exact.len();
            for (slot, value) in row[..gram_width].iter().enumerate() {
                match (
                    value.re.exact_invariant_value(),
                    value.im.exact_invariant_value(),
                ) {
                    (Some(re), Some(im)) => exact.push(Complex::new(re, im)),
                    _ => {
                        result.statuses[lane] = LaneStatus::InvalidInput(format!(
                            "nonfinite scalar product at argument {}, Taylor component {}",
                            slot / self.components,
                            slot % self.components,
                        ));
                        break;
                    }
                }
            }
            if exact.len() == previous_len + gram_width {
                lanes.push(lane);
            } else {
                exact.truncate(previous_len);
            }
        }
        let certificates = self.certificate.check_batch(&exact, lanes.len())?;
        self.certificate_stats.evaluated_rows = self.certificate.evaluated_rows();
        self.certificate_stats.passes = self.certificate.passes();
        let mut groups = BTreeMap::<[usize; 4], Vec<usize>>::new();
        for (lane, certificate) in lanes.into_iter().zip(certificates) {
            match certificate {
                InvariantCertificate::Compatible { basis } => {
                    groups.entry(basis).or_default().push(lane);
                    self.diagnostics[lane].push(format!(
                        "exact full-Taylor rank-four Gram certificate, chart {basis:?}"
                    ));
                }
                InvariantCertificate::Incompatible {
                    basis,
                    pair,
                    component,
                    ..
                } => {
                    result.statuses[lane] = LaneStatus::InvalidInput(format!(
                        "external Gram rank exceeds four: chart {basis:?}, residual {pair:?}, Taylor component {component} is certified nonzero"
                    ));
                }
                InvariantCertificate::NoRegularChart => {
                    result.statuses[lane] = LaneStatus::Unstable(
                        "no regular primal rank-four Gram chart was certified; rank and compatibility remain undetermined".into(),
                    );
                }
                InvariantCertificate::Unresolved { working_bits } => {
                    result.statuses[lane] = LaneStatus::Unstable(format!(
                        "exact Gram certificate unresolved at {working_bits} bits; input was preserved"
                    ));
                }
            }
        }
        if let Some(start) = start {
            self.profile.arguments = start.elapsed();
        }
        for (basis, lanes) in groups {
            if !self.charts.contains_key(&basis) {
                let preparation_start = self.profiling.then(Instant::now);
                let prepared = self.prepare_chart(basis);
                if let Some(start) = preparation_start {
                    self.profile.arguments += start.elapsed();
                }
                match prepared {
                    Ok(workspace) => {
                        self.charts.insert(basis, workspace);
                    }
                    Err(error) => {
                        for lane in lanes {
                            result.statuses[lane] = LaneStatus::MasterFailure(format!(
                                "certified chart preparation failed: {error}"
                            ));
                        }
                        continue;
                    }
                }
            }
            let padding = product(4 * n, self.components)?;
            let packed_width = width.checked_add(padding).ok_or_else(size_error)?;
            let mut packed = KinematicsBatch {
                rows: lanes.len(),
                components: self.components,
                values: vec![zero.clone(); product(packed_width, lanes.len())?],
            };
            for (index, &lane) in lanes.iter().enumerate() {
                packed.values[index * packed_width + padding..(index + 1) * packed_width]
                    .clone_from_slice(&input.values[lane * width..(lane + 1) * width]);
            }
            let workspace = self.charts.get_mut(&basis).expect("prepared chart");
            self.certificate_stats.reduction_batches += 1;
            let reduced = workspace.reduce_batch(&packed)?;
            add_stats(&mut self.stats, workspace.last_execution());
            if let Some(profile) = workspace.last_profile() {
                self.profile.arguments += profile.arguments;
                self.profile.scalar_masters += profile.scalar_masters;
                self.profile.reduction += profile.reduction;
            }
            for (index, &lane) in lanes.iter().enumerate() {
                result.statuses[lane] = reduced.statuses[index].clone();
                result.fallback_used[lane] = reduced.fallback_used[index];
                result.analytic_used[lane] = reduced.analytic_used[index];
                self.diagnostics[lane].extend_from_slice(&workspace.last_diagnostics()[index]);
                if let Ok(values) = reduced.lane(index) {
                    result.values[lane * output_width..(lane + 1) * output_width]
                        .clone_from_slice(values);
                    result.inherit_primal_envelope(
                        lane, &reduced, index, PrimalEnvelopeRoute::CertifiedScalar,
                    );
                }
            }
        }
        if let Some(start) = start {
            self.profile.total = start.elapsed();
        }
        Ok(result)
    }

    fn prepare_chart(&self, basis: [usize; 4]) -> Result<Workspace<T>, ReductionError> {
        let plan = self.plan.chart(basis)?;
        let mut workspace = match &self.shape {
            None => Workspace::new(&plan, self.prototype.clone()),
            Some(shape) => Workspace::new_dual(
                &plan,
                self.prototype.clone(),
                HyperDual::<Complex<Rational>>::new(shape.clone()),
            ),
        }?;
        workspace.set_accuracy_precision(self.accuracy_bits)?;
        workspace.set_profiling(self.profiling);
        Ok(workspace)
    }
}

fn add_stats(total: &mut ExecutionStats, value: ExecutionStats) {
    total.argument_rows += value.argument_rows;
    total.master_family_batches += value.master_family_batches;
    total.master_rows += value.master_rows;
    total.pentagon_box_rows += value.pentagon_box_rows;
    total.reduction_rows += value.reduction_rows;
    total.fallback_rows += value.fallback_rows;
    total.fallback_successes += value.fallback_successes;
    total.analytic_rows += value.analytic_rows;
}

fn product(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b).ok_or_else(size_error)
}

fn size_error() -> ReductionError {
    ReductionError::InvalidInput("certified scalar batch size overflow".into())
}

fn preparation(error: impl ToString) -> ReductionError {
    ReductionError::Preparation(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convention, ScalarReductionPlan};

    fn c(value: f64) -> Complex<f64> {
        Complex::new(value, 0.)
    }

    fn shifts(n: usize) -> Vec<[f64; 4]> {
        [
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
            [2., 3., 4., 5.],
        ][..n]
            .to_vec()
    }

    fn scalar_row(q: &[[f64; 4]]) -> Vec<Complex<f64>> {
        let mut row = Vec::new();
        for (i, a) in q.iter().enumerate() {
            for b in &q[i..] {
                row.push(c(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]));
            }
        }
        row.extend((0..q.len()).map(|i| c(30. + i as f64)));
        row.push(c(3.));
        row
    }

    fn settings(n: usize) -> ReductionSettings {
        let mut settings = ReductionSettings::new(n, 0);
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        settings
    }

    #[test]
    fn certificate_opt_in_does_not_weaken_existing_interface_guards() {
        for n in [6, 7] {
            assert!(ScalarReductionPlan::new(settings(n)).is_err());
            let mut tensor = settings(n);
            tensor.invariants = InvariantInput::SuppliedScalarProducts;
            assert!(ReductionPlan::new(tensor).is_err());
        }
        for algorithm in [Algorithm::Hybrid, Algorithm::PassarinoVeltman] {
            let mut input = settings(6);
            input.algorithm = algorithm;
            assert!(matches!(
                CertifiedScalarReductionPlan::new(input),
                Err(ReductionError::Unsupported(_))
            ));
        }
        let mut input = settings(6);
        input.rank = 1;
        assert!(matches!(
            CertifiedScalarReductionPlan::new(input),
            Err(ReductionError::InvalidInput(_))
        ));
        assert!(CertifiedScalarReductionPlan::new(settings(5)).is_err());
    }

    #[test]
    fn certified_scalar_batch_matches_coordinate_interface_and_isolates_failures() {
        for n in [6, 7] {
            let started = Instant::now();
            let phase = |label: &str| {
                eprintln!("certified scalar n={n} phase={label} elapsed={:?}", started.elapsed());
            };
            let q = shifts(n);
            let row = scalar_row(&q);
            phase("chart_prepare_begin");
            let plan = CertifiedScalarReductionPlan::new(settings(n)).unwrap();
            phase("chart_prepare_end");
            assert!(
                !plan
                    .chart([1, 2, 3, 4])
                    .unwrap()
                    .zero_profile_dispatch_allowed()
            );
            phase("certificate_prepare_begin");
            let mut workspace = CertifiedScalarWorkspace::new(&plan, 0_f64).unwrap();
            phase("certificate_prepare_end");
            assert!(
                workspace
                    .reduce_batch(&ScalarKinematicsBatch {
                        rows: usize::MAX,
                        components: 1,
                        values: Vec::new(),
                    })
                    .is_err()
            );
            workspace.set_profiling(true);
            let gram_arguments = n * (n + 1) / 2;
            let mut incompatible = row.clone();
            incompatible[gram_arguments - 1].re += 1.;
            let mut lower_rank = row.clone();
            lower_rank[..gram_arguments].fill(c(0.));
            let mut nonfinite = row.clone();
            nonfinite[1] = c(f64::NAN);
            let mut invalid_scale = row.clone();
            *invalid_scale.last_mut().unwrap() = c(-1.);
            let input = ScalarKinematicsBatch {
                rows: 7,
                components: 1,
                values: [
                    row.clone(),
                    incompatible,
                    row.clone(),
                    lower_rank,
                    nonfinite,
                    invalid_scale,
                    row.clone(),
                ]
                .concat(),
            };
            phase("certified_reduce_begin");
            let reduced = workspace.reduce_batch(&input).unwrap();
            phase("certified_reduce_end");
            assert!(matches!(reduced.statuses[1], LaneStatus::InvalidInput(_)));
            assert!(matches!(reduced.statuses[3], LaneStatus::Unstable(_)));
            assert!(matches!(reduced.statuses[4], LaneStatus::InvalidInput(_)));
            assert!(matches!(reduced.statuses[5], LaneStatus::InvalidInput(_)));
            for lane in [1, 3, 4, 5] {
                assert!(reduced.lane(lane).is_err());
            }
            assert_eq!(workspace.last_certificate_execution().reduction_batches, 1);
            assert_eq!(workspace.last_certificate_execution().evaluated_rows, 6);
            assert_eq!(workspace.last_execution().input_rows, 7);
            assert_eq!(workspace.last_execution().reduction_rows, 3);
            assert!(
                workspace.last_profile().unwrap().total
                    >= workspace.last_profile().unwrap().arguments
            );

            phase("coordinate_plan_prepare_begin");
            let reference_plan = ReductionPlan::new(settings(n)).unwrap();
            phase("coordinate_plan_prepare_end");
            phase("coordinate_workspace_prepare_begin");
            let mut reference = Workspace::new(&reference_plan, 0_f64).unwrap();
            phase("coordinate_workspace_prepare_end");
            let mut coordinates: Vec<_> = q.iter().flatten().copied().map(c).collect();
            coordinates.extend_from_slice(&row[gram_arguments..]);
            phase("coordinate_reduce_begin");
            let expected = reference
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: coordinates,
                })
                .unwrap();
            phase("coordinate_reduce_end");
            for lane in [0, 2, 6] {
                for (actual, expected) in reduced
                    .lane(lane)
                    .unwrap()
                    .iter()
                    .zip(expected.lane(0).unwrap())
                {
                    let scale = expected.re.hypot(expected.im).max(1e-14);
                    assert!(
                        (actual.re - expected.re).hypot(actual.im - expected.im) <= 1e-10 * scale,
                        "n={n}, lane={lane}: {actual:?} versus {expected:?}"
                    );
                }
            }
            let empty = workspace
                .reduce_batch(&ScalarKinematicsBatch {
                    rows: 0,
                    components: 1,
                    values: Vec::new(),
                })
                .unwrap();
            assert_eq!(empty.rows(), 0);
            assert_eq!(workspace.last_certificate_execution().evaluated_rows, 0);
            assert_eq!(workspace.last_certificate_execution().reduction_batches, 0);
        }
    }

    #[test]
    fn certified_scalar_rejects_an_incompatible_mixed_taylor_component() {
        let n = 6;
        let row = scalar_row(&shifts(n));
        let gram_arguments = n * (n + 1) / 2;
        let plan = CertifiedScalarReductionPlan::new(settings(n)).unwrap();
        let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let dual = HyperDual::<f64>::new(shape.clone());
        let mut workspace = CertifiedScalarWorkspace::new_dual(&plan, 0_f64, dual).unwrap();
        let mut values = Vec::new();
        for (argument, value) in row.iter().enumerate() {
            // The Gram jet is a common scalar multiple of a physical rank-four
            // matrix, so all determinant constraints vanish as Taylor series.
            let factor = if argument < gram_arguments {
                value.re
            } else {
                0.
            };
            values.extend([value.clone(), c(factor), c(2. * factor), c(3. * factor)]);
        }
        let mut bad = values.clone();
        bad[(gram_arguments - 1) * 4 + 3].re += 1.;
        let result = workspace
            .reduce_batch(&ScalarKinematicsBatch {
                rows: 2,
                components: 4,
                values: [values, bad].concat(),
            })
            .unwrap();
        assert_eq!(result.metadata().dual_shape(), Some(shape.as_slice()));
        assert!(
            matches!(&result.statuses[1], LaneStatus::InvalidInput(message) if message.contains("Taylor component 3"))
        );
        assert!(result.lane(1).is_err());
        assert_eq!(workspace.last_certificate_execution().reduction_batches, 1);
        assert_eq!(workspace.last_execution().input_rows, 2);
        assert_eq!(workspace.last_execution().argument_rows, 1);
        assert_eq!(
            result.statuses[0],
            LaneStatus::Success,
            "{:?}",
            workspace.last_diagnostics()
        );
    }
}
