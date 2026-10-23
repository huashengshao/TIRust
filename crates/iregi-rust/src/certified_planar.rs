//! Certified constant-one identities for planar five-point tensor integrals.
//!
//! This first interface covers preserved Cartesian inputs with a regular
//! rank-three constant-one chart. The exact represented coordinate determinant
//! must vanish in every requested Taylor component, and the selected chart
//! determinant must have a nonzero primal. Numerical values are never projected.
//! Accepted lanes run together through the ordinary instruction-major reducer,
//! including its complete scalar-master and stability checks.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use iregi_masters::NativeFloat;
use symbolica::{
    atom::Atom,
    domains::{
        dual::{DualNumberStructure, HyperDual},
        float::Complex,
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

use crate::{
    Algorithm, BatchResult, CertificateExecutionStats, ExecutionProfile, ExecutionStats,
    InputLayout, InputPolicy, InvariantInput, KinematicsBatch, LaneStatus, ReductionError,
    ReductionPlan, ReductionSettings, ResultMetadata, Workspace,
    basis::{Monomial, tensor_basis},
    exact_zero_certificate::{
        ExactZeroCertificate, ExactZeroConstraintPlan, ExactZeroConstraintWorkspace,
    },
    execution::validate_input_row,
    pv::planar_chart_polynomials,
};

type Shape = Option<Vec<Vec<usize>>>;
type Prepared<P> = Arc<OnceLock<Result<Arc<P>, String>>>;
type ChartCache = Arc<Mutex<HashMap<usize, Prepared<ReductionPlan>>>>;
type CertificateCache = Arc<Mutex<HashMap<Shape, Prepared<ExactZeroConstraintPlan>>>>;

/// Checked PV reduction for planar N=5 Cartesian tensors of rank at most seven.
///
/// Unit powers, D=4-2epsilon, preserved inputs and the PV algorithm are required.
/// A sample without a regular constant-one chart is reported as unstable; this
/// includes lower-rank and homogeneous-only families. A transverse Dual jet is
/// rejected whenever it leaves the planar variety at a requested Taylor order.
/// The unchecked chart plans are private and cannot bypass batch certification.
#[derive(Clone)]
pub struct CertifiedPlanarReductionPlan {
    settings: ReductionSettings,
    basis: Vec<Monomial>,
    layout: InputLayout,
    parameters: Vec<Atom>,
    constraints: Vec<Atom>,
    charts: ChartCache,
    certificates: CertificateCache,
}

impl std::fmt::Debug for CertifiedPlanarReductionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CertifiedPlanarReductionPlan")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl CertifiedPlanarReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        if settings.propagators != 5
            || settings.rank > 7
            || settings.powers != [1; 5]
            || settings.dimension_shift != 0
            || settings.algorithm != Algorithm::PassarinoVeltman
            || settings.invariants != InvariantInput::FromShifts
            || settings.input_policy != InputPolicy::Preserve
        {
            return Err(ReductionError::Unsupported(
                "certified planar reduction requires five preserved Cartesian propagators, rank at most seven, unit powers, D=4, and PV".into(),
            ));
        }
        let shifts: Vec<[Atom; 4]> = (0..5)
            .map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..5).map(|i| variable(&format!("m2_{i}"))).collect();
        let mut parameters: Vec<_> = shifts.iter().flatten().cloned().collect();
        parameters.extend(masses.iter().cloned());
        parameters.push(variable("mu"));
        Ok(Self {
            basis: tensor_basis(settings.rank),
            layout: InputLayout {
                shifts: 0..20,
                scalar_products: None,
                masses: 20..25,
                scale: 25,
            },
            constraints: planar_chart_polynomials(&shifts, &masses).map_err(preparation)?,
            settings,
            parameters,
            charts: Arc::default(),
            certificates: Arc::default(),
        })
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.settings
    }
    pub fn basis(&self) -> &[Monomial] {
        &self.basis
    }
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    pub fn input_layout(&self) -> &InputLayout {
        &self.layout
    }

    fn chart(&self, omitted: usize) -> Result<Arc<ReductionPlan>, ReductionError> {
        let entry = self
            .charts
            .lock()
            .map_err(|_| preparation("planar chart cache poisoned"))?
            .entry(omitted)
            .or_default()
            .clone();
        entry
            .get_or_init(|| {
                ReductionPlan::new_certified_planar(self.settings.clone(), omitted)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(preparation)
    }

    fn certificate(&self, shape: &Shape) -> Result<Arc<ExactZeroConstraintPlan>, ReductionError> {
        let entry = self
            .certificates
            .lock()
            .map_err(|_| preparation("planar certificate cache poisoned"))?
            .entry(shape.clone())
            .or_default()
            .clone();
        entry
            .get_or_init(|| {
                match shape {
                    None => ExactZeroConstraintPlan::new(&self.parameters, &self.constraints),
                    Some(shape) => ExactZeroConstraintPlan::new_dual(
                        &self.parameters,
                        &self.constraints,
                        &HyperDual::<u8>::new(shape.clone()),
                    ),
                }
                .map(Arc::new)
                .map_err(|e| e.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(preparation)
    }
}

/// Owned scratch for the certified planar interface. Structural plans are
/// shared; all numeric evaluators, sample buffers and certificate state are owned.
#[derive(Clone)]
pub struct CertifiedPlanarWorkspace<T: NativeFloat> {
    plan: CertifiedPlanarReductionPlan,
    prototype: T,
    shape: Shape,
    components: usize,
    certificate: ExactZeroConstraintWorkspace,
    charts: HashMap<usize, Workspace<T>>,
    accuracy_bits: u32,
    profiling: bool,
    stats: ExecutionStats,
    certificate_stats: CertificateExecutionStats,
    profile: ExecutionProfile,
    diagnostics: Vec<Vec<String>>,
}

impl<T: NativeFloat> CertifiedPlanarWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &CertifiedPlanarReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Self::prepare(plan, prototype, None)
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &CertifiedPlanarReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(&dual).map_err(preparation)?;
        Self::prepare(
            plan,
            prototype,
            Some(
                dual.get_shape()
                    .into_iter()
                    .map(<[usize]>::to_vec)
                    .collect(),
            ),
        )
    }

    fn prepare(
        plan: &CertifiedPlanarReductionPlan,
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
    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }
    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }

    pub fn reduce_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.profile = ExecutionProfile::default();
        let started = self.profiling.then(Instant::now);
        let result = self.reduce_batch_impl(input);
        if let Some(started) = started {
            self.profile.total = started.elapsed();
        }
        result
    }

    fn reduce_batch_impl(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let started = self.profiling.then(Instant::now);
        self.stats = ExecutionStats {
            input_rows: input.rows,
            ..Default::default()
        };
        self.certificate_stats = CertificateExecutionStats {
            input_rows: input.rows,
            ..Default::default()
        };
        self.diagnostics.clear();
        if input.components != self.components {
            return Err(ReductionError::InvalidInput(
                "planar input Taylor layout does not match workspace".into(),
            ));
        }
        let width = product(self.plan.parameter_count(), self.components)?;
        if input.values.len() != product(width, input.rows)? {
            return Err(ReductionError::InvalidInput(
                "planar input buffer has the wrong length".into(),
            ));
        }
        let output_width = product(product(self.plan.basis.len(), 4)?, self.components)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = BatchResult {
            basis: self.plan.basis.clone(),
            components: self.components,
            metadata: ResultMetadata::new(
                self.plan.settings.convention,
                self.plan.settings.tensor_weights,
                self.shape.clone(),
            ),
            values: vec![zero; product(output_width, input.rows)?],
            statuses: vec![
                LaneStatus::Unstable("planar certificate not evaluated".into());
                input.rows
            ],
            fallback_used: vec![false; input.rows],
            analytic_used: vec![false; input.rows],
        };
        self.diagnostics.resize_with(input.rows, Vec::new);
        let mut exact = Vec::with_capacity(input.values.len());
        let mut lanes = Vec::with_capacity(input.rows);
        for (lane, row) in input.values.chunks_exact(width).enumerate() {
            if let Some(status) = validate_input_row(row, &self.plan.layout, self.components) {
                result.statuses[lane] = status;
                continue;
            }
            let previous = exact.len();
            for value in row {
                match (value.re.exact_rational(), value.im.exact_rational()) {
                    (Some(re), Some(im)) => exact.push(Complex::new(re, im)),
                    _ => {
                        result.statuses[lane] = LaneStatus::Unstable(
                            "numeric type cannot certify its exact represented planar inputs"
                                .into(),
                        );
                        break;
                    }
                }
            }
            if exact.len() == previous + width {
                lanes.push(lane);
            } else {
                exact.truncate(previous);
            }
        }
        let rank_selection = vec![
            (0..self.components)
                .map(|component| (0, component))
                .collect();
            lanes.len()
        ];
        let certificates =
            self.certificate
                .check_selected_batch(&exact, lanes.len(), &rank_selection)?;
        self.add_certificate_stats()?;
        let mut planar_lanes = Vec::new();
        let mut planar_exact = Vec::new();
        for (compact, (&lane, certificate)) in lanes.iter().zip(certificates).enumerate() {
            match certificate {
                ExactZeroCertificate::Zero => {
                    planar_lanes.push(lane);
                    planar_exact.extend_from_slice(&exact[compact * width..(compact + 1) * width]);
                }
                ExactZeroCertificate::Nonzero { component, .. } => {
                    result.statuses[lane] = LaneStatus::InvalidInput(format!(
                        "planar coordinate determinant is certified nonzero in Taylor component {component}; transverse jets and nonplanar points require the ordinary interface"
                    ))
                }
                ExactZeroCertificate::Unresolved { working_bits } => {
                    result.statuses[lane] = LaneStatus::Unstable(format!(
                        "exact planar rank certificate unresolved at {working_bits} bits; input was preserved"
                    ))
                }
            }
        }
        // A Nonzero result is deliberately required here. The first certified
        // nonzero chart primal is selected; unresolved or zero determinants
        // cannot activate a numerical IF branch or a singular formal inverse.
        let chart_selection =
            vec![(1..5).map(|constraint| (constraint, 0)).collect(); planar_lanes.len()];
        let certificates = self.certificate.check_selected_batch(
            &planar_exact,
            planar_lanes.len(),
            &chart_selection,
        )?;
        self.add_certificate_stats()?;
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        for (lane, certificate) in planar_lanes.into_iter().zip(certificates) {
            match certificate {
                ExactZeroCertificate::Nonzero { constraint, component: 0 } if (1..5).contains(&constraint) => {
                    let omitted = constraint - 1;
                    groups.entry(omitted).or_default().push(lane);
                    self.diagnostics[lane].push(format!("exact full-Taylor planar identity with regular constant-one chart omitting coordinate row {omitted}"));
                }
                ExactZeroCertificate::Zero => result.statuses[lane] = LaneStatus::Unstable("no regular primal constant-one pentagon chart; lower-rank and homogeneous-only families require another reduction path".into()),
                ExactZeroCertificate::Unresolved { working_bits } => result.statuses[lane] = LaneStatus::Unstable(format!("constant-one chart certificate unresolved at {working_bits} bits; input was preserved")),
                _ => return Err(ReductionError::Evaluation("invalid planar chart certificate result".into())),
            }
        }
        if let Some(started) = started {
            self.profile.arguments = started.elapsed();
        }
        for (omitted, lanes) in groups {
            if !self.charts.contains_key(&omitted) {
                let preparation_started = self.profiling.then(Instant::now);
                let prepared = self.prepare_chart(omitted);
                if let Some(started) = preparation_started {
                    self.profile.arguments += started.elapsed();
                }
                match prepared {
                    Ok(workspace) => {
                        self.charts.insert(omitted, workspace);
                    }
                    Err(error) => {
                        for lane in lanes {
                            result.statuses[lane] = LaneStatus::MasterFailure(format!(
                                "certified planar chart preparation failed: {error}"
                            ));
                        }
                        continue;
                    }
                }
            }
            let mut packed = KinematicsBatch {
                rows: lanes.len(),
                components: self.components,
                values: Vec::with_capacity(product(lanes.len(), width)?),
            };
            for &lane in &lanes {
                packed
                    .values
                    .extend_from_slice(&input.values[lane * width..(lane + 1) * width]);
            }
            let workspace = self.charts.get_mut(&omitted).expect("prepared chart");
            self.certificate_stats.reduction_batches = self
                .certificate_stats
                .reduction_batches
                .checked_add(1)
                .ok_or_else(|| preparation("planar certificate counter overflow"))?;
            self.stats.analytic_rows = self
                .stats
                .analytic_rows
                .checked_add(lanes.len())
                .ok_or_else(|| preparation("planar execution counter overflow"))?;
            let reduced = workspace.reduce_batch(&packed);
            add_stats(&mut self.stats, workspace.last_execution())?;
            if let Some(profile) = workspace.last_profile() {
                self.profile.arguments += profile.arguments;
                self.profile.scalar_masters += profile.scalar_masters;
                self.profile.reduction += profile.reduction;
            }
            let reduced = reduced?;
            for (compact, &lane) in lanes.iter().enumerate() {
                result.statuses[lane] = reduced.statuses[compact].clone();
                result.fallback_used[lane] = reduced.fallback_used[compact];
                // Match ordinary analytic-profile routing: an attempted chart
                // remains visible even when its numerical guard rejects the lane.
                result.analytic_used[lane] = true;
                self.diagnostics[lane].extend_from_slice(&workspace.last_diagnostics()[compact]);
                if let Ok(values) = reduced.lane(compact) {
                    result.values[lane * output_width..(lane + 1) * output_width]
                        .clone_from_slice(values);
                }
            }
        }
        Ok(result)
    }

    fn add_certificate_stats(&mut self) -> Result<(), ReductionError> {
        self.certificate_stats.evaluated_rows = self
            .certificate_stats
            .evaluated_rows
            .checked_add(self.certificate.evaluated_rows())
            .ok_or_else(|| preparation("planar certificate counter overflow"))?;
        self.certificate_stats.passes = self
            .certificate_stats
            .passes
            .checked_add(self.certificate.passes())
            .ok_or_else(|| preparation("planar certificate counter overflow"))?;
        Ok(())
    }

    fn prepare_chart(&self, omitted: usize) -> Result<Workspace<T>, ReductionError> {
        let plan = self.plan.chart(omitted)?;
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

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("iregi::{name}")))
}
fn product(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b)
        .ok_or_else(|| ReductionError::InvalidInput("planar batch size overflow".into()))
}
fn preparation(error: impl ToString) -> ReductionError {
    ReductionError::Preparation(error.to_string())
}
fn add_stats(total: &mut ExecutionStats, value: ExecutionStats) -> Result<(), ReductionError> {
    macro_rules! add {
        ($field:ident) => {
            total.$field = total
                .$field
                .checked_add(value.$field)
                .ok_or_else(|| preparation("planar execution counter overflow"))?;
        };
    }
    add!(argument_rows);
    add!(master_family_batches);
    add!(master_rows);
    add!(pentagon_box_rows);
    add!(reduction_rows);
    add!(fallback_rows);
    add!(fallback_successes);
    // Each fixed-chart lane is already counted as one analytic-profile
    // attempt above. Its child work is aggregated without counting it twice.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convention, TensorWeights};
    use symbolica::{
        atom::AtomCore,
        domains::float::{Float, FloatLike, SingleFloat},
        evaluate::{Dualizer, ExpressionEvaluator},
    };

    fn settings(rank: u16) -> ReductionSettings {
        let mut settings = ReductionSettings::new(5, rank);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.convention = Convention::OneLoop;
        settings
    }

    // Match StabilityPlan::kinematic_scale: routing components have mass
    // dimension one, squared masses dimension two; mu is used only if scaleless.
    // This test-only helper normalizes the independently compared tensor output.
    fn dimensional_reference<T: NativeFloat>(
        prototype: &T,
        input_row: &[Complex<T>],
        components: usize,
        monomial: &Monomial,
        expected: &Complex<T>,
    ) -> T {
        assert_eq!(input_row.len(), 26 * components);
        let mut scale = prototype.zero();
        for argument in 0..20 {
            let value = &input_row[argument * components];
            let magnitude = NativeFloat::hypot(&value.re, &value.im);
            if magnitude > scale {
                scale = magnitude;
            }
        }
        for argument in 20..25 {
            let value = &input_row[argument * components];
            let magnitude = NativeFloat::hypot(&value.re, &value.im).sqrt();
            if magnitude > scale {
                scale = magnitude;
            }
        }
        if scale.is_zero() {
            scale = input_row[25 * components].re.clone();
        }
        let dimension = i32::try_from(monomial.rank()).unwrap() - 6;
        let power = scale.pow(u64::from(dimension.unsigned_abs()));
        let natural = if dimension < 0 { power.inv() } else { power };
        assert!(natural.is_finite() && !natural.is_zero());
        let norm = NativeFloat::hypot(&expected.re, &expected.im);
        if norm > natural { norm } else { natural }
    }

    fn points() -> [[i64; 4]; 5] {
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
        ]
    }

    fn row<T: NativeFloat>(
        prototype: &T,
        points: &[[i64; 4]; 5],
        components: usize,
    ) -> Vec<Complex<T>> {
        let mut values = vec![Complex::new(prototype.zero(), prototype.zero()); 26 * components];
        let primal = points
            .iter()
            .flatten()
            .copied()
            .chain([4, 9, 16, 25, 36, 3]);
        for (argument, value) in primal.enumerate() {
            values[argument * components].re = prototype.from_rational(&Rational::from(value));
        }
        values
    }

    fn exact_row(values: &[Complex<f64>]) -> Vec<Complex<Rational>> {
        values
            .iter()
            .map(|value| {
                Complex::new(
                    value.re.exact_rational().unwrap(),
                    value.im.exact_rational().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn planar_constructor_rejects_unsupported_contexts_without_an_unchecked_plan() {
        for invalid in [
            settings(8),
            settings(1),
            settings(1),
            settings(1),
            settings(1),
            settings(1),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, mut s)| {
            match i {
                1 => s.powers[0] = 2,
                2 => s.dimension_shift = 1,
                3 => s.invariants = InvariantInput::SuppliedScalarProducts,
                4 => s.input_policy = InputPolicy::legacy_onshell(),
                5 => s.algorithm = Algorithm::Ibp,
                _ => (),
            }
            s
        }) {
            assert!(CertifiedPlanarReductionPlan::new(invalid.clone()).is_err());
        }
        assert!(ReductionPlan::new_certified_planar(settings(0), 4).is_err());
    }

    #[test]
    fn planar_certificate_checks_full_jets_and_primal_chart_units() {
        let plan = CertifiedPlanarReductionPlan::new(settings(2)).unwrap();
        let shape = Some(vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![2, 0],
            vec![0, 2],
        ]);
        let components = 6;
        let mut tangent = row(&0., &points(), components);
        // The plane rotates: p1.time=s, p2.time=t, p4.time=s+t.
        tangent[(4 * 1) * components + 2].re = 1.;
        tangent[(4 * 2) * components + 1].re = 1.;
        tangent[(4 * 4) * components + 2].re = 1.;
        tangent[(4 * 4) * components + 1].re = 1.;
        let mut first = row(&0., &points(), components);
        first[16 * components + 2].re = 1.;
        let mut mixed = row(&0., &points(), components);
        mixed[16 * components + 3].re = 1.;
        let mut second = row(&0., &points(), components);
        second[16 * components + 4].re = 1.;
        let nilpotent_points = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 2, 3, 0],
        ];
        let mut nilpotent = row(&0., &nilpotent_points, components);
        nilpotent[15 * components + 2].re = 1.;
        let input = exact_row(&[tangent, first, mixed, second, nilpotent].concat());
        let mut workspace = plan.certificate(&shape).unwrap().instantiate();
        let selection = vec![(0..components).map(|component| (0, component)).collect(); 5];
        let results = workspace
            .check_selected_batch(&input, 5, &selection)
            .unwrap();
        assert_eq!(results[0], ExactZeroCertificate::Zero);
        for (lane, component) in [(1, 2), (2, 3), (3, 4)] {
            assert_eq!(
                results[lane],
                ExactZeroCertificate::Nonzero {
                    constraint: 0,
                    component
                }
            );
        }
        assert_eq!(results[4], ExactZeroCertificate::Zero);
        let primal_charts = vec![(1..5).map(|constraint| (constraint, 0)).collect(); 5];
        let results = workspace
            .check_selected_batch(&input, 5, &primal_charts)
            .unwrap();
        assert!(matches!(
            results[0],
            ExactZeroCertificate::Nonzero { component: 0, .. }
        ));
        assert_eq!(
            results[4],
            ExactZeroCertificate::Zero,
            "a nilpotent chart determinant is not a unit"
        );
    }

    #[test]
    fn rounded_zero_rank_determinant_and_homogeneous_families_never_succeed() {
        let plan = CertifiedPlanarReductionPlan::new(settings(0)).unwrap();
        let mut rounded = row(&0., &points(), 1);
        let epsilon = 2_f64.powi(-52);
        let q = [
            [0., 0., 0., 0.],
            [1. + epsilon, 1. + 2. * epsilon, 0., 0.],
            [1., 1. + epsilon, 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ];
        for (slot, value) in q.into_iter().flatten().enumerate() {
            rounded[slot].re = value;
        }
        let mut homogeneous = row(&0., &points(), 1);
        for (slot, mass) in [16., 15., 15., 15., 13.].into_iter().enumerate() {
            homogeneous[20 + slot].re = mass;
        }
        let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0.).unwrap();
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values: [rounded, homogeneous].concat(),
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::InvalidInput(reason) if reason.contains("certified nonzero"))
        );
        assert!(
            matches!(&result.statuses[1], LaneStatus::Unstable(reason) if reason.contains("constant-one"))
        );
        assert!(result.lane(0).is_err() && result.lane(1).is_err());
        assert_eq!(workspace.last_certificate_execution().reduction_batches, 0);
        assert_eq!(workspace.last_execution().master_rows, 0);
        assert!(
            workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 0,
                    components: 1,
                    values: vec![]
                })
                .unwrap()
                .statuses
                .is_empty()
        );
        assert!(
            workspace
                .reduce_batch(&KinematicsBatch {
                    rows: usize::MAX,
                    components: 1,
                    values: vec![]
                })
                .is_err()
        );
    }

    /// Independent fixed-weight box identity, evaluated using Symbolica. Five
    /// pinches of every input lane share one ordinary box reduction batch.
    fn box_identity<T: NativeFloat>(
        configuration: &ReductionSettings,
        prototype: T,
        accuracy: u32,
        shape: &Shape,
        input: &KinematicsBatch<T>,
        coefficient_jets: &[Complex<T>],
    ) -> Vec<Complex<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        let components = input.components;
        let mut configuration = configuration.clone();
        configuration.propagators = 4;
        configuration.powers = vec![1; 4];
        let plan = ReductionPlan::new(configuration).unwrap();
        let mut workspace = match shape {
            None => Workspace::new(&plan, prototype.clone()),
            Some(shape) => Workspace::new_dual(
                &plan,
                prototype.clone(),
                HyperDual::<u8>::new(shape.clone()),
            ),
        }
        .unwrap();
        workspace.set_accuracy_precision(accuracy).unwrap();
        let mut packed = KinematicsBatch {
            rows: input.rows * 5,
            components,
            values: Vec::new(),
        };
        for row in input.values.chunks_exact(26 * components) {
            for omitted in 0..5 {
                for i in 0..5 {
                    if i != omitted {
                        packed
                            .values
                            .extend_from_slice(&row[4 * i * components..4 * (i + 1) * components]);
                    }
                }
                for i in 0..5 {
                    if i != omitted {
                        packed
                            .values
                            .extend_from_slice(&row[(20 + i) * components..(21 + i) * components]);
                    }
                }
                packed
                    .values
                    .extend_from_slice(&row[25 * components..26 * components]);
            }
        }
        let boxes = workspace.reduce_batch(&packed).unwrap();
        eprintln!(
            "independent boxes at {} bits, accuracy {accuracy}: {:?}",
            prototype.get_precision(),
            boxes.statuses
        );
        assert!(
            boxes
                .statuses
                .iter()
                .all(|status| *status == LaneStatus::Success),
            "{:?}",
            boxes.statuses
        );
        let parameters: Vec<_> = (0..10)
            .map(|i| variable(&format!("independent_planar_box_{i}")))
            .collect();
        let expression = (0..5).fold(Atom::Zero, |sum, i| {
            sum + &parameters[i] * &parameters[5 + i]
        });
        let mut exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(&[expression], &parameters)
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        if let Some(shape) = shape {
            exact = exact
                .vectorize(&Dualizer::new(
                    HyperDual::<u8>::new(shape.clone()),
                    Vec::new(),
                ))
                .unwrap();
        }
        let mut evaluator = exact
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
        let slots = plan.basis().len() * 4;
        let mut inputs = Vec::new();
        for lane in 0..input.rows {
            for slot in 0..slots {
                inputs.extend_from_slice(
                    &coefficient_jets[lane * 5 * components..(lane + 1) * 5 * components],
                );
                for pinch in 0..5 {
                    inputs.extend_from_slice(
                        &boxes.lane(lane * 5 + pinch).unwrap()
                            [slot * components..(slot + 1) * components],
                    );
                }
            }
        }
        let mut output =
            vec![Complex::new(prototype.zero(), prototype.zero()); input.rows * slots * components];
        evaluator
            .evaluate_batch(&inputs, &mut output, input.rows * slots)
            .unwrap();
        output
    }

    fn constant_weights<T: NativeFloat>(
        prototype: &T,
        divisor: i64,
        components: usize,
    ) -> Vec<Complex<T>> {
        let mut coefficients =
            vec![Complex::new(prototype.zero(), prototype.zero()); 5 * components];
        for (i, weight) in [2, -1, -1, -1, 1].into_iter().enumerate() {
            coefficients[i * components].re =
                prototype.from_rational(&Rational::from((weight, divisor)));
        }
        coefficients
    }

    #[test]
    #[ignore = "bounded rank-seven qualification; use the exact-output artifact test documented in docs/research/certified-planar-api.md"]
    fn certified_planar_all_330_moments_match_independent_box_identity() {
        let prototype = Float::new(128);
        let configuration = settings(7);
        let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
        let mut workspace = CertifiedPlanarWorkspace::new(&plan, prototype.clone()).unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let input = KinematicsBatch {
            rows: 1,
            components: 1,
            values: row(&prototype, &points(), 1),
        };
        let reduced = workspace.reduce_batch(&input).unwrap();
        assert_eq!(reduced.statuses, [LaneStatus::Success]);
        assert_eq!(reduced.basis.len(), 330);
        assert!(!plan.chart(0).unwrap().zero_profile_dispatch_allowed());
        let expected = box_identity(
            &configuration,
            prototype.clone(),
            53,
            &None,
            &input,
            &constant_weights(&prototype, 6, 1),
        );
        let tolerance = prototype.from_rational(&Rational::from((1, 10_000_000_000_i64)));
        for (index, (actual, expected)) in
            reduced.lane(0).unwrap().iter().zip(&expected).enumerate()
        {
            let difference = NativeFloat::hypot(
                &(actual.re.clone() - &expected.re),
                &(actual.im.clone() - &expected.im),
            );
            let scale = dimensional_reference(
                &prototype,
                &input.values,
                input.components,
                &reduced.basis[index / 4],
                expected,
            );
            assert!(
                difference <= tolerance.clone() * scale,
                "tensor slot {index}"
            );
        }
        assert_eq!(workspace.last_certificate_execution().reduction_batches, 1);
        assert_eq!(workspace.last_execution().reduction_rows, 1);
        assert_eq!(workspace.last_execution().analytic_rows, 1);
    }

    #[test]
    fn certified_planar_spatial_binary64_guard_rejection_is_explicit() {
        let configuration = settings(2);
        let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
        let first = row(&0., &points(), 1);
        let alternate = [
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [2, 0, 1, 1],
        ];
        let second = row(&0., &alternate, 1);
        let mut transverse = first.clone();
        transverse[16].re = 1.;
        let mut bad_scale = first.clone();
        bad_scale[25].re = -1.;
        let input = KinematicsBatch {
            rows: 7,
            components: 1,
            values: [
                first.clone(),
                transverse,
                second.clone(),
                bad_scale,
                first.clone(),
                second.clone(),
                first,
            ]
            .concat(),
        };
        let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0.).unwrap();
        let reduced = workspace.reduce_batch(&input).unwrap();
        eprintln!("primary planar binary64 statuses: {:?}", reduced.statuses);
        // This exact family previously failed a test that incorrectly required
        // binary64 Success. Preserve the unchanged guard and expose its outcome.
        for lane in [0, 2, 4, 5, 6] {
            assert!(
                matches!(&reduced.statuses[lane], LaneStatus::Unstable(message)
                if message.contains("cancellation") && message.contains("Taylor component 0"))
            );
            assert!(reduced.lane(lane).is_err());
            assert!(reduced.analytic_used[lane]);
        }
        for lane in [1, 3] {
            assert!(matches!(
                reduced.statuses[lane],
                LaneStatus::InvalidInput(_)
            ));
            assert!(reduced.lane(lane).is_err());
            assert!(!reduced.analytic_used[lane]);
        }
        assert_eq!(reduced.fallback_used, [false; 7]);
        assert_eq!(workspace.last_certificate_execution().reduction_batches, 2);
        assert_eq!(workspace.last_execution().analytic_rows, 5);
        assert_eq!(workspace.last_execution().argument_rows, 5);
        assert_eq!(workspace.last_execution().reduction_rows, 5);
        assert_eq!(workspace.last_execution().fallback_successes, 0);
    }

    fn precise_batch(bits: u32) -> Vec<Complex<Float>> {
        let prototype = Float::new(bits);
        let configuration = settings(2);
        let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
        let first = row(&prototype, &points(), 1);
        let alternate = [
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [2, 0, 1, 1],
        ];
        let second = row(&prototype, &alternate, 1);
        let mut transverse = first.clone();
        transverse[16].re = prototype.one();
        let mut bad_scale = first.clone();
        bad_scale[25].re = -prototype.one();
        let input = KinematicsBatch {
            rows: 7,
            components: 1,
            values: [
                first.clone(),
                transverse,
                second.clone(),
                bad_scale,
                first.clone(),
                second.clone(),
                first.clone(),
            ]
            .concat(),
        };
        let mut workspace = CertifiedPlanarWorkspace::new(&plan, prototype.clone()).unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let reduced = workspace.reduce_batch(&input).unwrap();
        eprintln!(
            "primary planar {bits}-bit statuses, requested53: {:?}",
            reduced.statuses
        );
        let expected = box_identity(
            &configuration,
            prototype.clone(),
            53,
            &None,
            &KinematicsBatch {
                rows: 2,
                components: 1,
                values: [first, second].concat(),
            },
            &[
                constant_weights(&prototype, 6, 1),
                constant_weights(&prototype, 6, 1),
            ]
            .concat(),
        );
        for lane in [0, 2, 4, 5, 6] {
            assert_eq!(
                reduced.statuses[lane],
                LaneStatus::Success,
                "precision {bits}, lane {lane}"
            );
        }
        for lane in [1, 3] {
            assert!(matches!(
                reduced.statuses[lane],
                LaneStatus::InvalidInput(_)
            ));
            assert!(reduced.lane(lane).is_err());
        }
        assert_eq!(workspace.last_certificate_execution().reduction_batches, 2);
        assert_eq!(workspace.last_execution().reduction_rows, 5);
        assert_eq!(workspace.last_execution().analytic_rows, 5);
        assert_eq!(reduced.lane(0).unwrap(), reduced.lane(4).unwrap());
        assert_eq!(reduced.lane(0).unwrap(), reduced.lane(6).unwrap());
        assert_eq!(reduced.lane(2).unwrap(), reduced.lane(5).unwrap());
        let slots = reduced.basis.len() * 4;
        let values: Vec<_> = [reduced.lane(0).unwrap(), reduced.lane(2).unwrap()].concat();
        let tolerance = prototype.from_rational(&Rational::from((1, 10_000_000_000_i64)));
        for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
            let difference = NativeFloat::hypot(
                &(actual.re.clone() - &expected.re),
                &(actual.im.clone() - &expected.im),
            );
            let original_lane = [0_usize, 2][index / slots];
            let reference = dimensional_reference(
                &prototype,
                &input.values[original_lane * 26..(original_lane + 1) * 26],
                1,
                &reduced.basis[(index % slots) / 4],
                &expected,
            );
            assert!(
                difference <= tolerance.clone() * reference,
                "precision {bits}, tensor slot {index}"
            );
        }
        values
    }

    #[test]
    fn certified_planar_float_batches_preserve_binary64_target_and_converge() {
        let low = precise_batch(128);
        let high = precise_batch(256);
        let prototype = Float::new(256);
        let tolerance = prototype.from_rational(&Rational::from((1, 2)).pow(65));
        for (index, (low, high)) in low.iter().zip(&high).enumerate() {
            let low = Complex::new(
                prototype.from_rational(&low.re.exact_rational().unwrap()),
                prototype.from_rational(&low.im.exact_rational().unwrap()),
            );
            let difference = NativeFloat::hypot(&(low.re - &high.re), &(low.im - &high.im));
            // Both chart fixtures have identical physical scale six; obtain
            // it from their actual common masses and representative routing.
            let input_row = row(&prototype, &points(), 1);
            let basis = tensor_basis(2);
            let reference = dimensional_reference(
                &prototype,
                &input_row,
                1,
                &basis[(index % (basis.len() * 4)) / 4],
                high,
            );
            assert!(
                difference <= tolerance.clone() * reference,
                "128-to-256 convergence, tensor slot {index}"
            );
        }
    }

    #[test]
    fn certified_planar_rotating_mixed_jets_match_independent_box_identity() {
        let shape = Some(vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![2, 0],
            vec![0, 2],
        ]);
        let components = 6;
        let prototype = Float::new(128);
        let mut configuration = settings(2);
        configuration.convention = Convention::MadLoop;
        configuration.tensor_weights = TensorWeights::Multinomial;
        let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
        let mut tangent = row(&prototype, &points(), components);
        tangent[4 * components + 2].re = prototype.one();
        tangent[8 * components + 1].re = prototype.one();
        tangent[16 * components + 2].re = prototype.one();
        tangent[16 * components + 1].re = prototype.one();
        // An independent common mass variation remains tangent as well.
        for mass in 20..25 {
            tangent[mass * components + 3].re = prototype.one();
        }
        let mut transverse = tangent.clone();
        transverse[16 * components + 3].re = prototype.one();
        let input = KinematicsBatch {
            rows: 3,
            components,
            values: [tangent.clone(), transverse, tangent.clone()].concat(),
        };
        let mut workspace = CertifiedPlanarWorkspace::new_dual(
            &plan,
            prototype.clone(),
            HyperDual::<u8>::new(shape.clone().unwrap()),
        )
        .unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let reduced = workspace.reduce_batch(&input).unwrap();
        eprintln!("mixed Float128/request53 statuses: {:?}", reduced.statuses);
        assert_eq!(reduced.statuses[0], LaneStatus::Success);
        assert!(matches!(reduced.statuses[1], LaneStatus::InvalidInput(_)));
        assert_eq!(reduced.statuses[2], LaneStatus::Success);
        assert_eq!(reduced.metadata().dual_shape(), shape.as_deref());
        assert_eq!(reduced.lane(0).unwrap(), reduced.lane(2).unwrap());
        // C(s,t)=6+2st gives 1/C=1/6-st/18 through the retained order.
        let mut coefficients = constant_weights(&prototype, 6, components);
        for (i, weight) in [2_i64, -1, -1, -1, 1].into_iter().enumerate() {
            coefficients[i * components + 3].re =
                prototype.from_rational(&Rational::from((-weight, 18)));
        }
        let expected = box_identity(
            &configuration,
            prototype.clone(),
            53,
            &shape,
            &KinematicsBatch {
                rows: 1,
                components,
                values: tangent,
            },
            &coefficients,
        );
        let tolerance = prototype.from_rational(&Rational::from((1, 10_000_000_000_i64)));
        for (index, (actual, expected)) in
            reduced.lane(0).unwrap().iter().zip(&expected).enumerate()
        {
            let difference = NativeFloat::hypot(
                &(actual.re.clone() - &expected.re),
                &(actual.im.clone() - &expected.im),
            );
            let reference = dimensional_reference(
                &prototype,
                &input.values[..26 * components],
                components,
                &reduced.basis[index / (4 * components)],
                expected,
            );
            assert!(
                difference <= tolerance.clone() * reference,
                "mixed tensor component {index}"
            );
        }
        assert_eq!(workspace.last_certificate_execution().reduction_batches, 1);
        assert_eq!(workspace.last_execution().reduction_rows, 2);
        assert_eq!(workspace.last_execution().analytic_rows, 2);
    }

    #[test]
    fn certified_planar_tetrahedron_rank_two_binary64_isolates_failed_rows() {
        let configuration = settings(2);
        let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
        let tetrahedron = [
            [0, -1, -1, -1],
            [0, 1, 1, -1],
            [0, 1, -1, 1],
            [0, -1, 1, 1],
            [0, 0, 0, 0],
        ];
        let mut valid = row(&0_f64, &tetrahedron, 1);
        for argument in 20..26 {
            valid[argument].re = 1.; // Five squared masses and the public scale mu.
        }
        let mut nonplanar = valid.clone();
        nonplanar[16].re = 1.; // Move only p4 out of the spatial hyperplane.
        let mut bad_scale = valid.clone();
        bad_scale[25].re = -1.;
        let input = KinematicsBatch {
            rows: 5,
            components: 1,
            values: [
                valid.clone(),
                nonplanar,
                valid.clone(),
                bad_scale,
                valid.clone(),
            ]
            .concat(),
        };
        let original_values = input.values.clone();
        let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0_f64).unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let reduced = workspace.reduce_batch(&input).unwrap();

        assert_eq!(input.values, original_values);
        assert_eq!(reduced.rows(), 5);
        assert_eq!(reduced.components, 1);
        assert_eq!(reduced.basis.as_slice(), plan.basis());
        assert_eq!(reduced.basis.len(), 15);
        assert_eq!(reduced.metadata().convention(), configuration.convention);
        assert_eq!(
            reduced.metadata().tensor_weights(),
            configuration.tensor_weights
        );
        assert_eq!(reduced.metadata().dual_shape(), None);
        for lane in [0, 2, 4] {
            assert_eq!(
                reduced.statuses[lane],
                LaneStatus::Success,
                "tetrahedron lane {lane}: {:?}",
                reduced.statuses
            );
            assert!(
                workspace.last_diagnostics()[lane]
                    .iter()
                    .any(|message| message.contains("omitting coordinate row 0"))
            );
        }
        assert!(matches!(
            &reduced.statuses[1],
            LaneStatus::InvalidInput(reason) if reason.contains("certified nonzero")
        ));
        assert!(matches!(
            &reduced.statuses[3],
            LaneStatus::InvalidInput(reason) if reason.contains("mu must be positive and real")
        ));
        assert!(reduced.lane(1).is_err());
        assert!(reduced.lane(3).is_err());
        assert_eq!(reduced.analytic_used, [true, false, true, false, true]);
        assert_eq!(reduced.fallback_used, [false; 5]);
        assert_eq!(reduced.lane(0).unwrap(), reduced.lane(2).unwrap());
        assert_eq!(reduced.lane(0).unwrap(), reduced.lane(4).unwrap());

        let certificates = workspace.last_certificate_execution();
        assert_eq!(certificates.input_rows, 5);
        assert_eq!(certificates.reduction_batches, 1);
        assert!(certificates.passes >= 2);
        assert!(certificates.evaluated_rows >= 7);
        let execution = workspace.last_execution();
        assert_eq!(execution.input_rows, 5);
        assert_eq!(execution.reduction_rows, 3);
        // Invalid scale is rejected before certificates; the nonplanar row is
        // rejected by the rank certificate. Only three valid rows enter the chart.
        assert_eq!(execution.analytic_rows, 3);
        assert_eq!(execution.fallback_rows, 0);
        assert_eq!(execution.fallback_successes, 0);
        assert!(execution.master_rows >= 3);

        // At equal masses, 4*D4 - D0 - D1 - D2 - D3 = 12. Supply these
        // independent constant weights without using the wrapper's chart solve.
        let coefficients: Vec<_> = [-1_i64, -1, -1, -1, 4]
            .into_iter()
            .map(|weight| Complex::new(0_f64.from_rational(&Rational::from((weight, 12_i64))), 0.))
            .collect();
        let expected = box_identity(
            &configuration,
            0_f64,
            53,
            &None,
            &KinematicsBatch {
                rows: 1,
                components: 1,
                values: valid.clone(),
            },
            &coefficients,
        );
        let slots = plan.basis().len() * 4;
        assert_eq!(expected.len(), slots);
        for lane in [0, 2, 4] {
            let actual = reduced.lane(lane).unwrap();
            assert_eq!(actual.len(), slots);
            for (slot, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let error = (actual.re - expected.re).hypot(actual.im - expected.im);
                let scale =
                    dimensional_reference(&0_f64, &valid, 1, &reduced.basis[slot / 4], expected);
                assert!(
                    error <= 1e-10 * scale,
                    "lane {lane}, tensor {}, Laurent slot {}: {actual:?} versus {expected:?}",
                    slot / 4,
                    slot % 4,
                );
            }
        }
    }

    include!("certified_planar/public_api_tests.rs");
}
