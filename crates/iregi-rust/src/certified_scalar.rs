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
    exact_zero_certificate::{ExactZeroCertificate, ExactZeroConstraintPlan, ExactZeroConstraintWorkspace},
    invariant_certificate::{
        ExactInvariantScalar, InvariantCertificate, InvariantCertificatePlan,
        InvariantCertificateWorkspace,
    },
};

type Shape = Option<Vec<Vec<usize>>>;
type Prepared<P> = Arc<OnceLock<Result<Arc<P>, String>>>;
type ChartCache = Arc<Mutex<HashMap<[usize; 4], Prepared<ReductionPlan>>>>;
type CompatibilityCache = Arc<Mutex<HashMap<Shape, Prepared<ExactZeroConstraintPlan>>>>;
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
    /// Private shared runtime mode; scalar and tensor structural caches never alias.
    cartesian_tensor: bool,
    compatibilities: CompatibilityCache,
}

impl std::fmt::Debug for CertifiedScalarReductionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CertifiedScalarReductionPlan")
            .field("settings", &self.settings)
            .field("cartesian_tensor", &self.cartesian_tensor)
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
            cartesian_tensor: false,
            compatibilities: Arc::default(),
        };
        // Validate settings and compile one immutable chart at preparation
        // time. Other charts are prepared once on first demand.
        plan.chart([1, 2, 3, 4])?;
        Ok(plan)
    }

    pub(crate) fn new_tensor(mut settings: ReductionSettings) -> Result<Self, ReductionError> {
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        if !crate::certified_tensor::settings_supported(&settings) {
            return Err(ReductionError::Unsupported(
                "checked Cartesian supplied-G tensors require N6, rank<=6, positive powers and preserved inputs".into()));
        }
        let plan = Self { settings, charts: Arc::default(), certificates: Arc::default(),
            cartesian_tensor: true, compatibilities: Arc::default() };
        // No chart is numerically accepted by construction. Other structural
        // charts share this immutable plan's once-only cache, as for scalars.
        if crate::certified_tensor::packed_settings_supported(&plan.settings) {
            plan.chart([1, 2, 3, 4])?;
        } else {
            // Validate the private checked-product composition without eagerly
            // building the old, fully inlined high-point IBP tensor graph.
            crate::AffinePowerReductionPlan::new_checked_supplied(plan.settings.clone())?;
        }
        Ok(plan)
    }

    pub(crate) fn basis(&self) -> Vec<Monomial> {
        crate::basis::tensor_basis(self.settings.rank)
    }

    pub(crate) fn tensor_input_layout(&self) -> crate::InputLayout {
        let n = self.settings.propagators;
        let gram_end = 4 * n + n * (n + 1) / 2;
        crate::InputLayout { shifts: 0..4*n, scalar_products: Some(4*n..gram_end),
            masses: gram_end..gram_end+n, scale: gram_end+n }
    }

    pub(crate) fn contraction_plan(&self, monomials: &[Monomial]) -> Result<crate::ContractionPlan, ReductionError> {
        crate::ContractionPlan::new(self.chart([1, 2, 3, 4])?.as_ref(), monomials)
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.settings
    }

    pub fn argument_count(&self) -> usize {
        let n = self.settings.propagators;
        (if self.cartesian_tensor { 4*n } else { 0 }) + n * (n + 1) / 2 + n + 1
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
                (if self.cartesian_tensor {
                    ReductionPlan::new_certified_tensor(self.settings.clone(), basis)
                } else {
                    ReductionPlan::new_certified_scalar(self.settings.clone(), basis)
                }).map(Arc::new)
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
    fn compatibility(&self, shape: &Shape) -> Result<Arc<ExactZeroConstraintPlan>, ReductionError> {
        let entry = self.compatibilities.lock()
            .map_err(|_| preparation("Cartesian compatibility cache poisoned"))?
            .entry(shape.clone()).or_default().clone();
        entry.get_or_init(|| {
            let (parameters, residuals) = crate::certified_tensor::compatibility_constraints(self.settings.propagators);
            match shape {
                None => ExactZeroConstraintPlan::new(&parameters, &residuals),
                Some(shape) => ExactZeroConstraintPlan::new_dual(&parameters, &residuals,
                    &HyperDual::<u8>::new(shape.clone())),
            }.map(Arc::new).map_err(|error| error.to_string())
        }).as_ref().cloned().map_err(preparation)
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

/// Ordinary checked PV keeps imported child moments as graph inputs instead
/// of expanding six high-rank pentagon recurrences into one evaluator.
#[derive(Clone)]
enum ChartWorkspace<T:NativeFloat> {
    Generic(Workspace<T>),
    Packed(crate::thin_n6::ThinSixWorkspace<T>),
    AffinePower(Box<crate::AffinePowerWorkspace<T>>),
}
impl<T:NativeFloat> ChartWorkspace<T> where Complex<T>:EvaluationDomain {
    fn set_accuracy_precision(&mut self,bits:u32)->Result<(),ReductionError> {
        match self {Self::Generic(workspace)=>workspace.set_accuracy_precision(bits),Self::Packed(workspace)=>workspace.set_accuracy_precision(bits),Self::AffinePower(workspace)=>workspace.set_accuracy_precision(bits)}
    }
    fn set_profiling(&mut self,enabled:bool) {
        match self {Self::Generic(workspace)=>workspace.set_profiling(enabled),Self::Packed(workspace)=>workspace.set_profiling(enabled),Self::AffinePower(workspace)=>workspace.set_profiling(enabled)}
    }
    fn last_profile(&self)->Option<ExecutionProfile> {
        match self {Self::Generic(workspace)=>workspace.last_profile(),Self::Packed(workspace)=>workspace.last_profile(),Self::AffinePower(workspace)=>workspace.last_profile()}
    }
    fn last_diagnostics(&self)->&[Vec<String>] {
        match self {Self::Generic(workspace)=>workspace.last_diagnostics(),Self::Packed(workspace)=>workspace.last_diagnostics(),Self::AffinePower(workspace)=>workspace.last_diagnostics()}
    }
    fn reduce_batch(&mut self,input:&KinematicsBatch<T>)->Result<BatchResult<T>,ReductionError> {
        match self {Self::Generic(workspace)=>workspace.reduce_batch(input),Self::Packed(workspace)=>workspace.reduce_batch(input),Self::AffinePower(workspace)=>workspace.reduce_batch(input)}
    }
    fn packed_execution(&self)->Option<&crate::SixPointExecutionStats> {
        match self {Self::Generic(_)|Self::AffinePower(_)=>None,Self::Packed(workspace)=>Some(workspace.last_execution())}
    }
    fn affine_execution(&self) -> Option<&crate::AffinePowerExecutionStats> {
        match self {
            Self::AffinePower(workspace) => Some(workspace.last_execution()),
            Self::Generic(_) | Self::Packed(_) => None,
        }
    }
    fn execution(&self)->Result<ExecutionStats,ReductionError> {
        match self {
            Self::Generic(workspace)=>Ok(workspace.last_execution()),
            Self::AffinePower(workspace) => {
                let nested = workspace.last_execution();
                let mut total = ExecutionStats { input_rows: nested.input_rows, ..Default::default() };
                add_stats(&mut total, nested.child_execution)?;
                add_stats(&mut total, nested.parent_execution)?;
                Ok(total)
            }
            Self::Packed(workspace)=>{
                let nested=workspace.last_execution();
                let mut total=ExecutionStats{input_rows:nested.parent_input_rows,..Default::default()};
                add_stats(&mut total,nested.child_execution)?;add_stats(&mut total,nested.parent_execution)?;Ok(total)
            }
        }
    }
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
    compatibility: Option<ExactZeroConstraintWorkspace>,
    charts: HashMap<[usize; 4], ChartWorkspace<T>>,
    packed_execution:Vec<crate::SixPointExecutionStats>,
    affine_execution:Vec<crate::AffinePowerExecutionStats>,
    accuracy_bits: u32,
    profiling: bool,
    stats: ExecutionStats,
    certificate_stats: CertificateExecutionStats,
    compatibility_stats: CertificateExecutionStats,
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
            compatibility: if plan.cartesian_tensor { Some(plan.compatibility(&shape)?.instantiate()) } else { None },
            shape,
            charts: HashMap::new(),
            packed_execution:Vec::new(),
            affine_execution:Vec::new(),
            accuracy_bits,
            profiling: false,
            stats: ExecutionStats::default(),
            certificate_stats: CertificateExecutionStats::default(),
            compatibility_stats: CertificateExecutionStats::default(),
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
        self.certificate.set_precision_budget(initial_bits, maximum_bits)?;
        if let Some(compatibility) = &mut self.compatibility {
            compatibility.set_precision_budget(initial_bits, maximum_bits)?;
        }
        Ok(())
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
        self.reduce_values(input.rows, input.components, &input.values)
    }

    pub(crate) fn reduce_tensor_batch(&mut self, input: &KinematicsBatch<T>) -> Result<BatchResult<T>, ReductionError> {
        if !self.plan.cartesian_tensor {
            return Err(ReductionError::InvalidInput("scalar workspace cannot dispatch Cartesian tensors".into()));
        }
        self.reduce_values(input.rows, input.components, &input.values)
    }

    pub(crate) fn compatibility_execution(&self) -> CertificateExecutionStats { self.compatibility_stats }
    pub(crate) fn packed_execution(&self)->&[crate::SixPointExecutionStats] { &self.packed_execution }

    pub(crate) fn affine_execution(&self) -> &[crate::AffinePowerExecutionStats] { &self.affine_execution }

    fn reduce_values(&mut self, rows: usize, components: usize, values: &[Complex<T>]) -> Result<BatchResult<T>, ReductionError> {
        let start = self.profiling.then(Instant::now);
        self.stats = ExecutionStats {
            input_rows: rows,
            ..Default::default()
        };
        self.certificate_stats = CertificateExecutionStats {
            input_rows: rows,
            ..Default::default()
        };
        self.compatibility_stats = CertificateExecutionStats { input_rows: rows, ..Default::default() };
        self.profile = ExecutionProfile::default();
        self.diagnostics.clear();
        self.packed_execution.clear();
        self.affine_execution.clear();
        if components != self.components {
            return Err(ReductionError::InvalidInput(
                (if self.plan.cartesian_tensor { "tensor input Taylor layout does not match workspace" } else { "scalar input Taylor layout does not match workspace" }).into(),
            ));
        }
        let width = product(self.plan.argument_count(), self.components)?;
        let expected = product(width, rows)?;
        if values.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "{} batch expects {expected} values, received {}",
                if self.plan.cartesian_tensor { "tensor" } else { "scalar" }, values.len(),
            )));
        }
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let output_basis = self.plan.basis();
        let output_width = product(product(4, output_basis.len())?, self.components)?;
        let output_len = product(output_width, rows)?;
        self.diagnostics.resize_with(rows, Vec::new);
        let mut result = BatchResult {
            basis: output_basis,
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
                rows
            ],
            fallback_used: vec![false; rows],
            analytic_used: vec![false; rows],
            primal_envelopes: Vec::new(),
            primal_envelope_errors: Vec::new(),
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        };
        let n = self.plan.settings.propagators;
        let gram_width = product(n * (n + 1) / 2, self.components)?;
        let coordinate_width = if self.plan.cartesian_tensor { product(4*n, self.components)? } else { 0 };
        let geometry_width = coordinate_width.checked_add(gram_width).ok_or_else(size_error)?;
        let mut exact = Vec::with_capacity(product(geometry_width, rows)?);
        let mut lanes = Vec::with_capacity(rows);
        for (lane, row) in values.chunks_exact(width).enumerate() {
            let previous_len = exact.len();
            for (slot, value) in row[..geometry_width].iter().enumerate() {
                match (
                    value.re.exact_invariant_value(),
                    value.im.exact_invariant_value(),
                ) {
                    (Some(re), Some(im)) => exact.push(Complex::new(re, im)),
                    _ => {
                        result.statuses[lane] = LaneStatus::InvalidInput(format!(
                            "nonfinite {} at argument {}, Taylor component {}",
                            if self.plan.cartesian_tensor { "certified geometry" } else { "scalar product" }, slot / self.components,
                            slot % self.components,
                        ));
                        break;
                    }
                }
            }
            if exact.len() == previous_len + geometry_width {
                lanes.push(lane);
            } else {
                exact.truncate(previous_len);
            }
        }
        if let Some(compatibility) = &mut self.compatibility {
            let proofs = compatibility.check_batch(&exact, lanes.len())?;
            self.compatibility_stats.evaluated_rows = compatibility.evaluated_rows();
            self.compatibility_stats.passes = compatibility.passes();
            let mut gram = Vec::with_capacity(exact.len());
            let mut accepted = Vec::with_capacity(lanes.len());
            for ((lane, row), proof) in lanes.into_iter().zip(exact.chunks_exact(geometry_width)).zip(proofs) {
                match proof {
                    ExactZeroCertificate::Zero => {
                        accepted.push(lane);
                        gram.extend_from_slice(&row[coordinate_width..]);
                        self.diagnostics[lane].push("exact full-Taylor supplied-G/Cartesian relative-product compatibility".into());
                    }
                    ExactZeroCertificate::Nonzero { constraint, component } => {
                        result.statuses[lane] = LaneStatus::InvalidInput(format!(
                            "supplied scalar products disagree with Cartesian relative momenta: constraint {constraint}, Taylor component {component} is certified nonzero"));
                    }
                    ExactZeroCertificate::Unresolved { working_bits } => {
                        result.statuses[lane] = LaneStatus::Unstable(format!(
                            "exact supplied-G/Cartesian compatibility unresolved at {working_bits} bits; input was preserved"));
                    }
                }
            }
            exact = gram;
            lanes = accepted;
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
            let padding = if self.plan.cartesian_tensor { 0 } else { product(4 * n, self.components)? };
            let packed_width = width.checked_add(padding).ok_or_else(size_error)?;
            let mut packed = KinematicsBatch {
                rows: lanes.len(),
                components: self.components,
                values: vec![zero.clone(); product(packed_width, lanes.len())?],
            };
            for (index, &lane) in lanes.iter().enumerate() {
                packed.values[index * packed_width + padding..(index + 1) * packed_width]
                    .clone_from_slice(&values[lane * width..(lane + 1) * width]);
            }
            if self.plan.cartesian_tensor && crate::certified_tensor::packed_settings_supported(self.plan.settings()) {
                let chart = self.plan.chart(basis)?;
                let selection = format!(
                    "checked Cartesian supplied-G chart {basis:?}: requested {:?}; actual {:?}",
                    self.plan.settings.algorithm, chart.algorithm_used(),
                );
                for &lane in &lanes {
                    self.diagnostics[lane].push(selection.clone());
                    self.diagnostics[lane].extend_from_slice(chart.preparation_notes());
                }
            }
            let workspace = self.charts.get_mut(&basis).expect("prepared chart");
            self.certificate_stats.reduction_batches += 1;
            let reduced = workspace.reduce_batch(&packed);
            add_stats(&mut self.stats, workspace.execution()?)?;
            if let Some(nested)=workspace.packed_execution() {self.packed_execution.push(nested.clone());}
            if let Some(nested) = workspace.affine_execution() {
                self.affine_execution.push(nested.clone());
                let notes: Vec<_> = nested.child_methods.iter().flat_map(|method| {
                    std::iter::once(format!(
                        "checked supplied-G affine child N{}, powers {:?}: requested {:?}; actual {:?}",
                        method.propagators, method.powers, method.requested, method.actual,
                    )).chain(method.notes.iter().cloned())
                }).collect();
                for &lane in &lanes { self.diagnostics[lane].extend(notes.iter().cloned()); }
            }
            let reduced=reduced?;
            if reduced.basis != result.basis || reduced.components != self.components
                || reduced.metadata() != result.metadata() || reduced.rows() != lanes.len()
                || reduced.values.len() != product(output_width, lanes.len())?
                || reduced.fallback_used.len() != lanes.len() || reduced.analytic_used.len() != lanes.len() {
                return Err(ReductionError::Evaluation("checked chart returned incompatible tensor layout".into()));
            }
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
                        lane, &reduced, index, if self.plan.cartesian_tensor {
                            PrimalEnvelopeRoute::CertifiedTensor
                        } else { PrimalEnvelopeRoute::CertifiedScalar },
                    );
                }
            }
        }
        if let Some(start) = start {
            self.profile.total = start.elapsed();
        }
        Ok(result)
    }

    fn prepare_chart(&self, basis: [usize; 4]) -> Result<ChartWorkspace<T>, ReductionError> {
        if self.plan.cartesian_tensor && !crate::certified_tensor::packed_settings_supported(self.plan.settings()) {
            let plan = crate::AffinePowerReductionPlan::new_checked_supplied(self.plan.settings().clone())?;
            let mut workspace = match &self.shape {
                None => crate::AffinePowerWorkspace::new_with_master_policy(
                    &plan, self.prototype.clone(), crate::MasterExecutionPolicy::Symbolica,
                ),
                Some(shape) => crate::AffinePowerWorkspace::new_dual_with_master_policy(
                    &plan, self.prototype.clone(), HyperDual::<Complex<Rational>>::new(shape.clone()),
                    crate::MasterExecutionPolicy::Symbolica,
                ),
            }?;
            workspace.set_accuracy_precision(self.accuracy_bits)?;
            workspace.set_profiling(self.profiling);
            return Ok(ChartWorkspace::AffinePower(Box::new(workspace)));
        }
        let plan = self.plan.chart(basis)?;
        if self.plan.cartesian_tensor && crate::certified_tensor::packed_settings_supported(self.plan.settings()) {
            let packed=crate::thin_n6::ThinSixReductionPlan::from_parent(plan.as_ref().clone())?;
            let mut workspace=match &self.shape {
                None=>crate::thin_n6::ThinSixWorkspace::new(&packed,self.prototype.clone()),
                Some(shape)=>crate::thin_n6::ThinSixWorkspace::new_dual(&packed,self.prototype.clone(),HyperDual::<Complex<Rational>>::new(shape.clone())),
            }?;
            workspace.set_accuracy_precision(self.accuracy_bits)?;workspace.set_profiling(self.profiling);
            return Ok(ChartWorkspace::Packed(workspace));
        }
        // The opt-in tensor wrapper has an explicit Symbolica master contract.
        // Preserve the existing scalar API's feature-dependent Default policy.
        let master_policy = if self.plan.cartesian_tensor {
            crate::MasterExecutionPolicy::Symbolica
        } else {
            crate::MasterExecutionPolicy::Default
        };
        let mut workspace = match &self.shape {
            None => Workspace::new_with_master_policy(&plan, self.prototype.clone(), master_policy),
            Some(shape) => Workspace::new_dual_with_master_policy(
                &plan,
                self.prototype.clone(),
                HyperDual::<Complex<Rational>>::new(shape.clone()),
                master_policy,
            ),
        }?;
        workspace.set_accuracy_precision(self.accuracy_bits)?;
        workspace.set_profiling(self.profiling);
        Ok(ChartWorkspace::Generic(workspace))
    }
}

fn add_stats(total: &mut ExecutionStats, value: ExecutionStats)->Result<(),ReductionError> {
    let add=|a:usize,b:usize|a.checked_add(b).ok_or_else(||ReductionError::Evaluation("checked chart execution counter overflow".into()));
    total.argument_rows=add(total.argument_rows,value.argument_rows)?;
    total.master_family_batches=add(total.master_family_batches,value.master_family_batches)?;
    total.master_rows=add(total.master_rows,value.master_rows)?;
    total.pentagon_box_rows=add(total.pentagon_box_rows,value.pentagon_box_rows)?;
    total.reduction_rows=add(total.reduction_rows,value.reduction_rows)?;
    total.fallback_rows=add(total.fallback_rows,value.fallback_rows)?;
    total.fallback_successes=add(total.fallback_successes,value.fallback_successes)?;
    total.analytic_rows=add(total.analytic_rows,value.analytic_rows)?;Ok(())
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
            // The retained coordinate f64 observer can be conservatively
            // Unstable from cancellation at requested53. Keep that actual
            // failure as evidence and compare against a higher-precision
            // observer with the same requested accuracy and tolerance.
            use symbolica::domains::float::{Float, SingleFloat};
            let prototype = Float::new(128);
            let lift = |value: &Complex<f64>| Complex::new(
                prototype.from_rational(&value.re.exact_rational().unwrap()),
                prototype.from_rational(&value.im.exact_rational().unwrap()),
            );
            let mut reference = Workspace::new(&reference_plan, prototype.clone()).unwrap();
            reference.set_accuracy_precision(53).unwrap();
            phase("coordinate_workspace_prepare_end");
            let mut coordinates: Vec<_> = q.iter().flatten().copied().map(c).collect();
            coordinates.extend_from_slice(&row[gram_arguments..]);
            phase("coordinate_reduce_begin");
            let expected = reference
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: coordinates.iter().map(lift).collect(),
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
                    let actual = lift(actual);
                    let norm = NativeFloat::hypot(&expected.re, &expected.im);
                    let floor = prototype.from_rational(&(1e-14_f64).exact_rational().unwrap());
                    let scale = if norm > floor { norm } else { floor };
                    let tolerance = prototype.from_rational(&(1e-10_f64).exact_rational().unwrap());
                    let difference = &actual - expected;
                    assert!(NativeFloat::hypot(&difference.re, &difference.im) <= tolerance * scale,
                        "n={n}, lane={lane}: {actual:?} versus {expected:?}");
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
    #[test]
    fn generalized_checked_tensor_constructor_does_not_inline_the_highpoint_graph() {
        for algorithm in [Algorithm::Ibp, Algorithm::PassarinoVeltman, Algorithm::Hybrid] {
            let mut settings = ReductionSettings::new(6, 6);
            settings.algorithm = algorithm;
            settings.invariants = InvariantInput::SuppliedScalarProducts;
            settings.powers[0] = 2;
            settings.dimension_shift = 1;
            let plan = CertifiedScalarReductionPlan::new_tensor(settings.clone()).unwrap();
            assert!(plan.charts.lock().unwrap().is_empty());
            assert_eq!(plan.settings().algorithm, algorithm);
            assert_eq!(plan.settings().powers, settings.powers);
            assert_eq!(plan.settings().dimension_shift, 1);
            assert_eq!(plan.settings().invariants, InvariantInput::SuppliedScalarProducts);
            assert_eq!(plan.basis().len(), 210);
            assert!(plan.cartesian_tensor);
        }
    }

}
