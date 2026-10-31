//! Private checked affine owner. Immediate production candidate: N3/r1,
//! ordinary preserved Cartesian Hybrid/PV through rank6; no generalized claims.
use crate::{
    Algorithm, BatchResult, CertificateExecutionStats, ExecutionProfile, ExecutionStats,
    InputPolicy, InvariantInput, KinematicsBatch, LaneStatus, ReductionError, ReductionPlan,
    ReductionSettings, ResultMetadata, Workspace,
    affine_descent_guard::{
        AffineDescentGuardPlan, AffineDescentGuardWorkspace, AffineDescentSelection,
    },
    basis::{Monomial, tensor_basis},
    execution::WorkspacePolicy,
    plan::CertifiedAffineChart,
    pv::AffineDescentBudget,
    types::PrimalEnvelopeRoute,
};
use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use symbolica::{
    domains::{
        dual::{DualNumberStructure, HyperDual},
        float::Complex,
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

type Shape = Option<Vec<Vec<usize>>>;
type Key = (usize, Vec<usize>);
type Prepared<P> = Arc<OnceLock<Result<Arc<P>, String>>>;

pub(crate) fn settings_supported(s: &ReductionSettings) -> bool {
    s.propagators == 3
        && s.rank <= 6
        && s.powers == [1; 3]
        && s.dimension_shift == 0
        && s.invariants == InvariantInput::FromShifts
        && s.input_policy == InputPolicy::Preserve
        && matches!(s.algorithm, Algorithm::PassarinoVeltman | Algorithm::Hybrid)
}

#[derive(Clone)]
pub(crate) struct CertifiedAffinePlan {
    settings: ReductionSettings,
    basis: Vec<Monomial>,
    guards: Arc<Mutex<HashMap<Shape, Prepared<AffineDescentGuardPlan>>>>,
    charts: Arc<Mutex<HashMap<Key, Prepared<ReductionPlan>>>>,
}

impl CertifiedAffinePlan {
    pub(crate) fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        if !settings_supported(&settings) {
            return Err(ReductionError::Unsupported("checked affine activation requires N3/rank<=6, preserved Cartesian unit powers, h0, Hybrid/PV".into()));
        }
        Ok(Self {
            basis: tensor_basis(settings.rank),
            settings,
            guards: Arc::default(),
            charts: Arc::default(),
        })
    }
    fn guard(&self, shape: &Shape) -> Result<Arc<AffineDescentGuardPlan>, ReductionError> {
        let entry = self
            .guards
            .lock()
            .map_err(|_| prep("affine guard cache poisoned"))?
            .entry(shape.clone())
            .or_default()
            .clone();
        entry
            .get_or_init(|| {
                let budget = AffineDescentBudget::default();
                match shape {
                    None => AffineDescentGuardPlan::new(3, 1, budget),
                    Some(shape) => AffineDescentGuardPlan::new_dual(
                        3,
                        1,
                        budget,
                        &HyperDual::<u8>::new(shape.clone()),
                    ),
                }
                .map(Arc::new)
                .map_err(|e| e.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(prep)
    }
    fn chart(
        &self,
        key: &Key,
        guard: &AffineDescentGuardPlan,
    ) -> Result<Arc<ReductionPlan>, ReductionError> {
        let context = CertifiedAffineChart {
            key: guard
                .charts()
                .get(key.0)
                .ok_or_else(|| prep("affine chart index out of range"))?
                .clone(),
            inactive: key.1.clone(),
        };
        let entry = self
            .charts
            .lock()
            .map_err(|_| prep("affine chart cache poisoned"))?
            .entry(key.clone())
            .or_default()
            .clone();
        entry
            .get_or_init(|| {
                ReductionPlan::new_certified_affine(self.settings.clone(), context)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .cloned()
            .map_err(prep)
    }
}

#[derive(Clone)]
pub(crate) struct CertifiedAffineWorkspace<T: NativeFloat> {
    plan: CertifiedAffinePlan,
    guard_plan: Arc<AffineDescentGuardPlan>,
    guard: AffineDescentGuardWorkspace,
    prototype: T,
    shape: Shape,
    components: usize,
    policy: WorkspacePolicy,
    masters: MasterExecutionPolicy,
    charts: HashMap<Key, Workspace<T>>,
    accuracy_bits: u32,
    profiling: bool,
    stats: ExecutionStats,
    certificate_stats: CertificateExecutionStats,
    profile: ExecutionProfile,
    diagnostics: Vec<Vec<String>>,
    attempts: Vec<bool>,
    selected: Vec<bool>,
}

impl<T: NativeFloat> CertifiedAffineWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(crate) fn new(
        plan: &CertifiedAffinePlan,
        prototype: T,
        shape: Shape,
        policy: WorkspacePolicy,
        masters: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        masters.validate_available().map_err(prep)?;
        if prototype.get_precision() == 0 {
            return Err(prep("working precision must be positive"));
        }
        let guard_plan = plan.guard(&shape)?;
        Ok(Self {
            plan: plan.clone(),
            guard: guard_plan.instantiate(),
            guard_plan,
            accuracy_bits: prototype.get_precision(),
            prototype,
            components: shape.as_ref().map_or(1, Vec::len),
            shape,
            policy,
            masters,
            charts: HashMap::new(),
            profiling: false,
            stats: Default::default(),
            certificate_stats: Default::default(),
            profile: Default::default(),
            diagnostics: Vec::new(),
            attempts: Vec::new(),
            selected: Vec::new(),
        })
    }
    pub(crate) fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(
                "invalid affine accuracy precision".into(),
            ));
        }
        for workspace in self.charts.values_mut() {
            workspace.set_accuracy_precision(bits)?;
        }
        self.accuracy_bits = bits;
        Ok(())
    }
    pub(crate) fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        for w in self.charts.values_mut() {
            w.set_profiling(enabled);
        }
    }
    pub(crate) fn last_execution(&self) -> ExecutionStats {
        self.stats
    }
    pub(crate) fn last_certificate_execution(&self) -> CertificateExecutionStats {
        self.certificate_stats
    }
    pub(crate) fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }
    pub(crate) fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }
    pub(crate) fn last_chart_attempts(&self) -> &[bool] {
        &self.attempts
    }
    pub(crate) fn last_selected(&self) -> &[bool] {
        &self.selected
    }
    fn prepare_chart(&self, key: &Key) -> Result<Workspace<T>, ReductionError> {
        let plan = self.plan.chart(key, &self.guard_plan)?;
        let mut workspace = match &self.shape {
            None => Workspace::new_with_policies(
                &plan,
                self.prototype.clone(),
                self.policy,
                self.masters,
            ),
            Some(shape) => Workspace::new_dual_with_policies(
                &plan,
                self.prototype.clone(),
                HyperDual::<Complex<Rational>>::new(shape.clone()),
                self.policy,
                self.masters,
            ),
        }?;
        workspace.set_accuracy_precision(self.accuracy_bits)?;
        workspace.set_profiling(self.profiling);
        Ok(workspace)
    }
    pub(crate) fn reduce_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.profile = Default::default();
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
        self.diagnostics = vec![Vec::new(); input.rows];
        self.attempts = vec![false; input.rows];
        self.selected = vec![false; input.rows];
        let width = product(16, self.components)?;
        if input.components != self.components || input.values.len() != product(input.rows, width)?
        {
            return Err(ReductionError::InvalidInput(
                "affine input dimensions/Taylor shape differ".into(),
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
            values: vec![zero; product(input.rows, output_width)?],
            statuses: vec![LaneStatus::Unstable("affine proof not accepted".into()); input.rows],
            fallback_used: vec![false; input.rows],
            analytic_used: vec![false; input.rows],
            primal_envelopes: if self.policy.retain_primal_envelopes {
                vec![None; input.rows]
            } else {
                Vec::new()
            },
            primal_envelope_errors: if self.policy.retain_primal_envelopes {
                vec![None; input.rows]
            } else {
                Vec::new()
            },
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        };
        let selected = self.guard.select_batch(input);
        self.certificate_stats.evaluated_rows = self.guard.evaluated_rows();
        self.certificate_stats.passes = self.guard.passes();
        let selected = selected?;
        let mut groups = BTreeMap::<Key, Vec<usize>>::new();
        for (lane, selection) in selected.into_iter().enumerate() {
            match selection {
                AffineDescentSelection::Chart { index, inactive } => {
                    self.selected[lane] = true;
                    self.diagnostics[lane].push(format!("full-Taylor certified N3 affine chart {index}, inactive children {inactive:?}; original q and inputs retained"));
                    groups.entry((index, inactive)).or_default().push(lane);
                }
                AffineDescentSelection::InvalidInput(reason) => {
                    result.statuses[lane] = LaneStatus::InvalidInput(reason)
                }
                other => {
                    result.statuses[lane] = LaneStatus::Unstable(format!(
                        "constant-one affine profile not available: {other:?}; no projection or successful zero"
                    ))
                }
            }
        }
        if let Some(started) = started {
            self.profile.arguments = started.elapsed();
        }
        for (key, lanes) in groups {
            if !self.charts.contains_key(&key) {
                let started = self.profiling.then(Instant::now);
                let prepared = self.prepare_chart(&key);
                if let Some(started) = started {
                    self.profile.arguments += started.elapsed();
                }
                match prepared {
                    Ok(workspace) => {
                        self.charts.insert(key.clone(), workspace);
                    }
                    Err(error) => {
                        for lane in lanes {
                            result.statuses[lane] = LaneStatus::MasterFailure(format!(
                                "affine chart preparation failed: {error}"
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
                self.attempts[lane] = true;
            }
            self.certificate_stats.reduction_batches = self
                .certificate_stats
                .reduction_batches
                .checked_add(1)
                .ok_or_else(|| prep("affine certificate counter overflow"))?;
            self.stats.analytic_rows = self
                .stats
                .analytic_rows
                .checked_add(lanes.len())
                .ok_or_else(|| prep("affine analytic counter overflow"))?;
            let workspace = self.charts.get_mut(&key).expect("prepared affine chart");
            let reduced = workspace.reduce_batch(&packed);
            add_stats(&mut self.stats, workspace.last_execution())?;
            if let Some(p) = workspace.last_profile() {
                self.profile.arguments += p.arguments;
                self.profile.scalar_masters += p.scalar_masters;
                self.profile.reduction += p.reduction;
            }
            for (compact, &lane) in lanes.iter().enumerate() {
                if let Some(notes) = workspace.last_diagnostics().get(compact) {
                    self.diagnostics[lane].extend_from_slice(notes);
                }
            }
            let reduced = match reduced {
                Ok(reduced) => reduced,
                Err(error) => {
                    for &lane in &lanes {
                        result.statuses[lane] = LaneStatus::MasterFailure(format!(
                            "affine chart batch failed: {error}"
                        ));
                    }
                    continue;
                }
            };
            if reduced.basis != self.plan.basis
                || reduced.components != self.components
                || reduced.metadata() != result.metadata()
                || reduced.rows() != lanes.len()
                || reduced.values.len() != product(lanes.len(), output_width)?
                || reduced.fallback_used.len() != lanes.len()
                || reduced.analytic_used.len() != lanes.len()
            {
                for &lane in &lanes {
                    result.statuses[lane] = LaneStatus::MasterFailure(
                        "affine child returned inconsistent metadata or complete buffers".into(),
                    );
                }
                continue;
            }
            for (compact, &lane) in lanes.iter().enumerate() {
                result.statuses[lane] = reduced.statuses[compact].clone();
                result.analytic_used[lane] = true;
                result.fallback_used[lane] = reduced.fallback_used[compact];
                if let Ok(values) = reduced.lane(compact) {
                    result.values[lane * output_width..(lane + 1) * output_width]
                        .clone_from_slice(values);
                    result.inherit_primal_envelope(
                        lane,
                        &reduced,
                        compact,
                        PrimalEnvelopeRoute::CertifiedAffine,
                    );
                }
            }
        }
        Ok(result)
    }
}
fn prep(error: impl ToString) -> ReductionError {
    ReductionError::Preparation(error.to_string())
}
fn product(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b)
        .ok_or_else(|| prep("affine buffer overflow"))
}
fn add_stats(total: &mut ExecutionStats, work: ExecutionStats) -> Result<(), ReductionError> {
    macro_rules! add {
        ($field:ident) => {
            total.$field = total
                .$field
                .checked_add(work.$field)
                .ok_or_else(|| prep("affine work counter overflow"))?;
        };
    }
    add!(argument_rows);
    add!(master_family_batches);
    add!(master_rows);
    add!(pentagon_box_rows);
    add!(reduction_rows);
    add!(fallback_rows);
    add!(fallback_successes);
    Ok(())
}

#[cfg(test)]
#[path = "certified_affine_tests.rs"]
mod tests;
