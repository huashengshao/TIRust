//! Constant-denominator descent before the existing low-point planner.
//!
//! The relation is the existing ScalarCompiler constant-one routing branch.
//! Symbolica owns its matrix solve, recurrence expressions, Taylor products,
//! stability programs and imported quadrature contraction. This module owns
//! only structural profiles, power-state scheduling and packed numerical rows.
mod dag;
mod packing;
mod parent;

use crate::{
    Algorithm, BatchResult, CertificateExecutionStats, ExecutionProfile, ExecutionStats,
    InputLayout, InputPolicy, InvariantInput, KinematicsBatch, LaneStatus, ReductionError,
    ReductionPlan, ReductionSettings, ResultMetadata, Workspace,
    affine_descent_guard::{
        AffineDescentGuardPlan, AffineDescentGuardWorkspace, AffineDescentSelection,
    },
    basis::{Monomial, tensor_basis},
    execution::{
        WorkspacePolicy,
        quadrature_context::{
            CoupledQuadratureBatch, QuadratureErrorContext, QuadratureErrorOrigin,
        },
    },
    pv::AffineDescentBudget,
    stability::quadrature::imported_affine::ImportedAffineQuadrature,
    types::PrimalEnvelopeRoute,
};
use dag::{AffinePowerDag, DagBudget};
use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use packing::PowerBatchLayout;
use parent::ParentRecipe;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
    time::Instant,
};
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        float::Complex,
    },
    evaluate::EvaluationDomain,
};

type Shape = Option<Vec<Vec<usize>>>;
type ProfileKey = (usize, Vec<usize>);

/// Explicit preparation and row budgets, rather than mathematical point or
/// power limits. A rejected budget produces an error, never a partial tensor.
#[derive(Clone, Copy, Debug)]
pub struct AffinePowerBudget {
    pub max_charts: usize,
    pub max_constraints: usize,
    pub max_cofactors: usize,
    pub max_dag_nodes: usize,
    pub max_dag_edges: usize,
    pub max_terminals: usize,
    pub max_parent_parameters: usize,
    pub max_packed_child_rows: usize,
}
impl Default for AffinePowerBudget {
    fn default() -> Self {
        Self {
            max_charts: 4096,
            max_constraints: 4096,
            max_cofactors: 65536,
            max_dag_nodes: 65536,
            max_dag_edges: 262144,
            max_terminals: 65536,
            max_parent_parameters: 1_000_000,
            max_packed_child_rows: 1_000_000,
        }
    }
}
impl AffinePowerBudget {
    fn chart(self) -> AffineDescentBudget {
        AffineDescentBudget {
            max_propagators: usize::MAX,
            max_charts: self.max_charts,
            max_constraints: self.max_constraints,
            max_cofactors: self.max_cofactors,
        }
    }
    fn dag(self) -> DagBudget {
        DagBudget {
            max_nodes: self.max_dag_nodes,
            max_edges: self.max_dag_edges,
            max_terminals: self.max_terminals,
        }
    }
}

struct PreparedProfile {
    dag: AffinePowerDag,
    packing: PowerBatchLayout,
    parent: ParentRecipe,
}
#[derive(Clone)]
enum ChildPlan {
    LowPoint(ReductionPlan),
    Affine(Box<AffinePowerReductionPlan>),
}

/// Immutable numerical-input interface. Dynamic point count is reduced before
/// the inner low-point planner is invoked. A regular constant-one chart must
/// be freshly certified for every complete represented input Taylor row.
#[derive(Clone)]
pub struct AffinePowerReductionPlan {
    settings: ReductionSettings,
    budget: AffinePowerBudget,
    basis: Vec<Monomial>,
    layout: InputLayout,
    guards: Arc<Mutex<HashMap<Shape, Arc<AffineDescentGuardPlan>>>>,
    profiles: Arc<Mutex<HashMap<(Shape, ProfileKey), Arc<PreparedProfile>>>>,
    children: Arc<Mutex<HashMap<Vec<u16>, ChildPlan>>>,
}
impl std::fmt::Debug for AffinePowerReductionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AffinePowerReductionPlan")
            .field("settings", &self.settings)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}
impl AffinePowerReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Self::with_budget(settings, AffinePowerBudget::default())
    }
    pub fn with_budget(
        settings: ReductionSettings,
        budget: AffinePowerBudget,
    ) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        if n < 2
            || settings.powers.len() != n
            || settings.powers.contains(&0)
            || settings.invariants != InvariantInput::FromShifts
            || settings.input_policy != InputPolicy::Preserve
        {
            return Err(invalid(
                "affine power descent requires N>=2, N positive powers and preserved Cartesian routing",
            ));
        }
        if settings.rank > 6 {
            return Err(ReductionError::Unsupported(
                "affine power descent currently qualifies tensor rank through six".into(),
            ));
        }
        let geometry = parameters(n)?;
        if [
            budget.max_charts,
            budget.max_constraints,
            budget.max_cofactors,
            budget.max_dag_nodes,
            budget.max_dag_edges,
            budget.max_terminals,
            budget.max_parent_parameters,
            budget.max_packed_child_rows,
        ]
        .contains(&0)
        {
            return Err(invalid("affine power resource budgets must be positive"));
        }
        // At least one admissible chart needs N cofactor entries. Check this
        // necessary resource lower bound before allocating symbolic shifts.
        if geometry > budget.max_parent_parameters || n > budget.max_cofactors {
            return Err(invalid(
                "affine geometry exceeds the explicit parameter/cofactor resource budget",
            ));
        }
        let basis = tensor_basis(settings.rank);
        let layout = InputLayout {
            shifts: 0..mul(n, 4)?,
            scalar_products: None,
            masses: mul(n, 4)?..mul(n, 5)?,
            scale: mul(n, 5)?,
        };
        Ok(Self {
            settings,
            budget,
            basis,
            layout,
            guards: Arc::default(),
            profiles: Arc::default(),
            children: Arc::default(),
        })
    }
    pub fn settings(&self) -> &ReductionSettings {
        &self.settings
    }
    pub fn basis(&self) -> &[Monomial] {
        &self.basis
    }
    pub fn input_layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn parameter_count(&self) -> usize {
        self.layout.scale + 1
    }
    pub(crate) fn contraction_plan(
        &self,
        monomials: &[Monomial],
    ) -> Result<crate::ContractionPlan, ReductionError> {
        crate::ContractionPlan::from_basis(
            &self.basis,
            self.settings.convention,
            self.settings.tensor_weights,
            monomials,
        )
    }
    fn guard(&self, shape: &Shape) -> Result<Arc<AffineDescentGuardPlan>, ReductionError> {
        if let Some(p) = self
            .guards
            .lock()
            .map_err(|_| prep("affine guard cache poisoned"))?
            .get(shape)
            .cloned()
        {
            return Ok(p);
        }
        let n = self.settings.propagators;
        let _timer = crate::pv::PreparationTimer::new(
            std::env::var_os("IREGI_PREPARATION_PROFILE").is_some(),
            format_args!(
                "affine-power N{n} guard structural preparation, components {}",
                shape.as_ref().map_or(1, Vec::len)
            ),
        );
        let rank = 4usize.min(n - 2);
        let guard = Arc::new(match shape {
            None => AffineDescentGuardPlan::new(n, rank, self.budget.chart()),
            Some(s) => AffineDescentGuardPlan::new_dual(
                n,
                rank,
                self.budget.chart(),
                &HyperDual::<u8>::new(s.clone()),
            ),
        }?);
        let mut cache = self
            .guards
            .lock()
            .map_err(|_| prep("affine guard cache poisoned"))?;
        Ok(cache.entry(shape.clone()).or_insert(guard).clone())
    }
    fn profile(
        &self,
        shape: &Shape,
        key: &ProfileKey,
        guard: &AffineDescentGuardPlan,
    ) -> Result<Arc<PreparedProfile>, ReductionError> {
        let cache_key = (shape.clone(), key.clone());
        if let Some(p) = self
            .profiles
            .lock()
            .map_err(|_| prep("affine profile cache poisoned"))?
            .get(&cache_key)
            .cloned()
        {
            return Ok(p);
        }
        let _timer = crate::pv::PreparationTimer::new(
            std::env::var_os("IREGI_PREPARATION_PROFILE").is_some(),
            format_args!(
                "affine-power N{} chart {} power-DAG parent preparation",
                self.settings.propagators, key.0
            ),
        );
        let dag = AffinePowerDag::new(self.settings.powers.clone(), &key.1, self.budget.dag())
            .map_err(prep)?;
        let packing = PowerBatchLayout::new(&self.settings, &dag.terminals)?;
        let chart = guard
            .charts()
            .get(key.0)
            .ok_or_else(|| prep("affine chart index out of range"))?;
        let parent = ParentRecipe::prepare(
            &self.settings,
            chart,
            &key.1,
            &dag,
            guard.parameters(),
            shape,
            self.budget.chart(),
            self.budget.max_parent_parameters,
        )?;
        let prepared = Arc::new(PreparedProfile {
            dag,
            packing,
            parent,
        });
        let mut cache = self
            .profiles
            .lock()
            .map_err(|_| prep("affine profile cache poisoned"))?;
        Ok(cache.entry(cache_key).or_insert(prepared).clone())
    }
    fn child(&self, settings: &ReductionSettings) -> Result<ChildPlan, ReductionError> {
        if let Some(p) = self
            .children
            .lock()
            .map_err(|_| prep("affine child-plan cache poisoned"))?
            .get(&settings.powers)
            .cloned()
        {
            return Ok(p);
        }
        // Point descent happens BEFORE any inner fixed planner resource guard.
        let child = if settings.propagators <= 5 {
            ChildPlan::LowPoint(ReductionPlan::new(settings.clone())?)
        } else {
            ChildPlan::Affine(Box::new(Self::with_budget(settings.clone(), self.budget)?))
        };
        let mut cache = self
            .children
            .lock()
            .map_err(|_| prep("affine child-plan cache poisoned"))?;
        Ok(cache
            .entry(settings.powers.clone())
            .or_insert(child)
            .clone())
    }
}

#[derive(Clone, Debug)]
pub struct AffinePowerChildMethod {
    pub propagators: usize,
    pub powers: Vec<u16>,
    pub requested: Algorithm,
    /// None denotes another affine composite; see its retained nested notes.
    pub actual: Option<Algorithm>,
    pub rows: usize,
    pub notes: Vec<String>,
}
#[derive(Clone, Debug, Default)]
pub struct AffinePowerExecutionStats {
    pub input_rows: usize,
    pub certificate: CertificateExecutionStats,
    pub profiles: usize,
    pub dag_nodes: usize,
    pub dag_edges: usize,
    pub packed_child_rows: usize,
    pub child_batch_calls: usize,
    pub parent_batch_calls: usize,
    pub parent_rows: usize,
    pub quadrature_work_rows: usize,
    pub child_execution: ExecutionStats,
    pub parent_execution: ExecutionStats,
    pub child_methods: Vec<AffinePowerChildMethod>,
}

/// Coupled result with the separate actual-route quadrature diagnostic.
/// Estimates remain successive-rule estimates, not certified error bounds.
#[derive(Clone)]
pub struct AffinePowerResult<T: NativeFloat> {
    owned: CoupledQuadratureBatch<T>,
    working: u32,
    accuracy: u32,
}
impl<T: NativeFloat> AffinePowerResult<T> {
    pub fn result(&self) -> &BatchResult<T> {
        self.owned.result()
    }
    pub fn into_result(self) -> BatchResult<T> {
        self.owned.into_parts().0
    }
    pub fn quadrature_estimates(&self, row: usize) -> Result<&[Complex<T>], String> {
        self.owned.errors(row, self.working, self.accuracy)
    }
}

/// Actual immediate-child contexts, retained only by an enabled finite test
/// observer. Release builds contain no observation field or clone operation.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct AffinePowerChildObservation<T: NativeFloat> {
    pub(crate) settings: ReductionSettings,
    pub(crate) input: KinematicsBatch<T>,
    pub(crate) result: BatchResult<T>,
    pub(crate) origins: Vec<(usize, usize)>,
    pub(crate) quadrature_origins: Vec<Result<QuadratureErrorOrigin, String>>,
    pub(crate) quadrature_errors: Vec<Result<Vec<Complex<T>>, String>>,
}

#[derive(Clone)]
enum ChildWorkspace<T: NativeFloat> {
    LowPoint {
        workspace: Box<Workspace<T>>,
        actual: Algorithm,
        notes: Vec<String>,
    },
    Affine(Box<AffinePowerWorkspace<T>>),
}
#[derive(Clone)]
struct ProfileWorkspace<T: NativeFloat> {
    prepared: Arc<PreparedProfile>,
    parent: Workspace<T>,
    quadrature: ImportedAffineQuadrature<T>,
}
#[derive(Clone)]
pub struct AffinePowerWorkspace<T: NativeFloat> {
    plan: AffinePowerReductionPlan,
    prototype: T,
    accuracy_bits: u32,
    shape: Shape,
    components: usize,
    metadata: ResultMetadata,
    masters: MasterExecutionPolicy,
    guard_plan: Arc<AffineDescentGuardPlan>,
    guard: AffineDescentGuardWorkspace,
    profiles: HashMap<ProfileKey, ProfileWorkspace<T>>,
    children: HashMap<Vec<u16>, ChildWorkspace<T>>,
    stats: AffinePowerExecutionStats,
    diagnostics: Vec<Vec<String>>,
    profiling: bool,
    profile: ExecutionProfile,
    #[cfg(test)]
    component_probe_children: Option<Vec<AffinePowerChildObservation<T>>>,
}

impl<T: NativeFloat> AffinePowerWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &AffinePowerReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Self::prepare(plan, prototype, None, MasterExecutionPolicy::Default)
    }
    pub fn new_with_master_policy(
        plan: &AffinePowerReductionPlan,
        prototype: T,
        masters: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        Self::prepare(plan, prototype, None, masters)
    }
    pub fn new_dual<D: DualNumberStructure>(
        plan: &AffinePowerReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        Self::new_dual_with_master_policy(plan, prototype, dual, MasterExecutionPolicy::Default)
    }
    pub fn new_dual_with_master_policy<D: DualNumberStructure>(
        plan: &AffinePowerReductionPlan,
        prototype: T,
        dual: D,
        masters: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(&dual).map_err(|e| invalid(e.to_string()))?;
        let shape = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect::<Vec<_>>();
        analytic_order(&Some(shape.clone()))?;
        Self::prepare(plan, prototype, Some(shape), masters)
    }
    fn prepare(
        plan: &AffinePowerReductionPlan,
        prototype: T,
        shape: Shape,
        masters: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        masters
            .validate_available()
            .map_err(|e| prep(e.to_string()))?;
        if prototype.get_precision() == 0 {
            return Err(invalid("working precision must be positive"));
        }
        let guard_plan = plan.guard(&shape)?;
        let guard = guard_plan.instantiate();
        Ok(Self {
            plan: plan.clone(),
            accuracy_bits: prototype.get_precision(),
            prototype,
            components: shape.as_ref().map_or(1, Vec::len),
            metadata: ResultMetadata::new(
                plan.settings.convention,
                plan.settings.tensor_weights,
                shape.clone(),
            ),
            shape,
            masters,
            guard_plan,
            guard,
            profiles: HashMap::new(),
            children: HashMap::new(),
            stats: Default::default(),
            diagnostics: Vec::new(),
            profiling: false,
            profile: Default::default(),
            #[cfg(test)]
            component_probe_children: None,
        })
    }
    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(invalid(
                "accuracy precision must be within working precision",
            ));
        }
        for child in self.children.values_mut() {
            match child {
                ChildWorkspace::LowPoint { workspace, .. } => {
                    workspace.set_accuracy_precision(bits)?
                }
                ChildWorkspace::Affine(w) => w.set_accuracy_precision(bits)?,
            }
        }
        for p in self.profiles.values_mut() {
            p.parent.set_accuracy_precision(bits)?;
        }
        self.accuracy_bits = bits;
        Ok(())
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        for c in self.children.values_mut() {
            match c {
                ChildWorkspace::LowPoint { workspace, .. } => workspace.set_profiling(enabled),
                ChildWorkspace::Affine(w) => w.set_profiling(enabled),
            }
        }
        for p in self.profiles.values_mut() {
            p.parent.set_profiling(enabled);
        }
    }
    pub fn last_execution(&self) -> &AffinePowerExecutionStats {
        &self.stats
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
    #[cfg(test)]
    pub(crate) fn enable_component_child_observer(&mut self, enabled: bool) {
        self.component_probe_children = enabled.then(Vec::new);
    }
    #[cfg(test)]
    pub(crate) fn component_child_observations(&self) -> Option<&[AffinePowerChildObservation<T>]> {
        self.component_probe_children.as_deref()
    }
    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }
    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }
    fn reset(&mut self, rows: usize) {
        self.stats = AffinePowerExecutionStats {
            input_rows: rows,
            ..Default::default()
        };
        self.diagnostics.clear();
        self.profile = Default::default();
        #[cfg(test)]
        if let Some(children) = &mut self.component_probe_children {
            children.clear();
        }
    }
    pub fn reduce_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        Ok(self.reduce_coupled_batch(input)?.into_result())
    }
    pub fn reduce_coupled_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<AffinePowerResult<T>, ReductionError> {
        self.reset(input.rows);
        let started = self.profiling.then(Instant::now);
        let result = self.reduce_impl(input);
        if let Some(started) = started {
            self.profile.total = started.elapsed();
        }
        result.map(|owned| AffinePowerResult {
            owned,
            working: self.prototype.get_precision(),
            accuracy: self.accuracy_bits,
        })
    }
    pub fn reduce_typed_dual_batch<D: DualNumber<Complex<T>>>(
        &mut self,
        input: &[D],
        rows: usize,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.reset(rows);
        if input.len() != mul(rows, self.plan.parameter_count())? {
            return Err(invalid(
                "affine typed input length differs from rows and parameters",
            ));
        }
        let shape = self
            .shape
            .as_ref()
            .ok_or_else(|| invalid("typed Dual input requires a Dual workspace"))?;
        let expected: Vec<_> = shape.iter().map(Vec::as_slice).collect();
        let mut values = Vec::new();
        values
            .try_reserve_exact(mul(input.len(), self.components)?)
            .map_err(|_| invalid("affine typed input allocation failed"))?;
        for value in input {
            if value.get_shape() != expected || value.values().len() != self.components {
                return Err(invalid("affine typed ordered Dual shape differs"));
            }
            values.extend_from_slice(value.values());
        }
        self.reduce_batch(&KinematicsBatch {
            rows,
            components: self.components,
            values,
        })
    }
    fn make_profile(&self, key: &ProfileKey) -> Result<ProfileWorkspace<T>, ReductionError> {
        let prepared = self.plan.profile(&self.shape, key, &self.guard_plan)?;
        let mut parent = workspace(
            &prepared.parent.plan,
            &self.prototype,
            &self.shape,
            WorkspacePolicy::AFFINE_QUADRATURE_PARENT,
            self.masters,
        )?;
        parent.set_accuracy_precision(self.accuracy_bits)?;
        parent.set_profiling(self.profiling);
        let quadrature = prepared.parent.quadrature.instantiate(&self.prototype);
        Ok(ProfileWorkspace {
            prepared,
            parent,
            quadrature,
        })
    }
    fn make_child(
        &self,
        settings: &ReductionSettings,
    ) -> Result<ChildWorkspace<T>, ReductionError> {
        let _timer = crate::pv::PreparationTimer::new(
            std::env::var_os("IREGI_PREPARATION_PROFILE").is_some(),
            format_args!(
                "affine-power N{} full-rank {} child h={} powers {:?}",
                settings.propagators, settings.rank, settings.dimension_shift, settings.powers
            ),
        );
        let child = self.plan.child(settings)?;
        Ok(match child {
            ChildPlan::LowPoint(plan) => {
                let actual = plan.algorithm_used();
                let notes = plan.preparation_notes().to_vec();
                let mut child = workspace(
                    &plan,
                    &self.prototype,
                    &self.shape,
                    WorkspacePolicy::AFFINE_QUADRATURE_CHILD,
                    self.masters,
                )?;
                child.set_accuracy_precision(self.accuracy_bits)?;
                child.set_profiling(self.profiling);
                ChildWorkspace::LowPoint {
                    workspace: Box::new(child),
                    actual,
                    notes,
                }
            }
            ChildPlan::Affine(plan) => {
                let mut child = Self::prepare(
                    &plan,
                    self.prototype.clone(),
                    self.shape.clone(),
                    self.masters,
                )?;
                child.set_accuracy_precision(self.accuracy_bits)?;
                child.set_profiling(self.profiling);
                ChildWorkspace::Affine(Box::new(child))
            }
        })
    }
    fn reduce_impl(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<CoupledQuadratureBatch<T>, ReductionError> {
        let input_width = mul(self.plan.parameter_count(), self.components)?;
        if input.components != self.components
            || mul(input.rows, input_width)? != input.values.len()
        {
            return Err(invalid(
                "affine input dimensions or ordered component count differ",
            ));
        }
        let mut result = blank_result(
            &self.plan.basis,
            &self.metadata,
            self.components,
            input.rows,
            &self.prototype,
        )?;
        let mut context = QuadratureErrorContext::missing(input.rows);
        self.diagnostics.resize_with(input.rows, Vec::new);
        if input.rows == 0 {
            return CoupledQuadratureBatch::new(result, context);
        }
        let started = self.profiling.then(Instant::now);
        let selections = self.guard.select_batch(input)?;
        self.stats.certificate = CertificateExecutionStats {
            input_rows: input.rows,
            evaluated_rows: self.guard.evaluated_rows(),
            passes: self.guard.passes(),
            ..Default::default()
        };
        let mut groups = BTreeMap::<ProfileKey, Vec<usize>>::new();
        for (lane, selected) in selections.into_iter().enumerate() {
            match selected {
                AffineDescentSelection::Chart { index, inactive } => {
                    groups.entry((index, inactive)).or_default().push(lane);
                }
                AffineDescentSelection::InvalidInput(reason) => {
                    result.statuses[lane] = LaneStatus::InvalidInput(reason)
                }
                other => {
                    result.statuses[lane] = LaneStatus::Unstable(format!(
                        "constant-one affine certificate not accepted: {other:?}; no denominator projection"
                    ))
                }
            }
        }
        if let Some(started) = started {
            self.profile.arguments += started.elapsed();
        }
        for (key, lanes) in groups {
            if !self.profiles.contains_key(&key) {
                match self.make_profile(&key) {
                    Ok(p) => {
                        self.profiles.insert(key.clone(), p);
                    }
                    Err(e) => {
                        for lane in lanes {
                            result.statuses[lane] = LaneStatus::MasterFailure(format!(
                                "affine power profile preparation failed: {e}"
                            ));
                        }
                        continue;
                    }
                }
            }
            let prepared = self.profiles[&key].prepared.clone();
            self.stats.profiles = add(self.stats.profiles, 1)?;
            self.stats.dag_nodes = add(self.stats.dag_nodes, prepared.dag.nodes.len())?;
            self.stats.dag_edges = add(self.stats.dag_edges, prepared.dag.edge_count)?;
            let mut physical = KinematicsBatch {
                rows: lanes.len(),
                components: self.components,
                values: Vec::with_capacity(mul(lanes.len(), input_width)?),
            };
            for &lane in &lanes {
                physical
                    .values
                    .extend_from_slice(&input.values[lane * input_width..(lane + 1) * input_width]);
                self.diagnostics[lane].push(format!("constant-one affine power DAG chart {}, inactive {:?}; unchanged q, rank {}, h={}",key.0,key.1,self.plan.settings.rank,self.plan.settings.dimension_shift));
            }
            let demanded = vec![(0..prepared.dag.terminals.len()).collect(); physical.rows];
            let packed = prepared.packing.pack(
                &physical,
                &demanded,
                self.plan.budget.max_packed_child_rows,
            )?;
            self.stats.packed_child_rows = add(self.stats.packed_child_rows, packed.total_rows)?;
            let mut children = Vec::with_capacity(packed.groups.len());
            for (group, packed_group) in packed.groups.iter().enumerate() {
                if packed_group.input.rows == 0 {
                    children.push(None);
                    continue;
                }
                let settings = &prepared.packing.child_settings()[group];
                if !self.children.contains_key(&settings.powers) {
                    match self.make_child(settings) {
                        Ok(c) => {
                            self.children.insert(settings.powers.clone(), c);
                        }
                        Err(e) => {
                            children.push(None);
                            for origin in &packed_group.origins {
                                let lane = lanes[origin.parent_row];
                                self.diagnostics[lane].push(format!(
                                    "child powers {:?} preparation failed: {e}",
                                    settings.powers
                                ));
                            }
                            continue;
                        }
                    }
                }
                self.stats.child_batch_calls = add(self.stats.child_batch_calls, 1)?;
                let child = self
                    .children
                    .get_mut(&settings.powers)
                    .expect("prepared affine child");
                let (evaluated, work, profile, notes, actual) = match child {
                    ChildWorkspace::LowPoint {
                        workspace,
                        actual,
                        notes,
                    } => {
                        let value =
                            workspace.reduce_batch_with_quadrature_context(&packed_group.input);
                        (
                            value,
                            workspace.last_execution(),
                            workspace.last_profile(),
                            workspace.last_diagnostics().to_vec(),
                            Some((*actual, notes.clone())),
                        )
                    }
                    ChildWorkspace::Affine(workspace) => {
                        let value = workspace
                            .reduce_coupled_batch(&packed_group.input)
                            .map(|v| v.owned);
                        let mut work = workspace.stats.child_execution;
                        sum_work(&mut work, &workspace.stats.parent_execution)?;
                        (
                            value,
                            work,
                            workspace.last_profile(),
                            workspace.last_diagnostics().to_vec(),
                            None,
                        )
                    }
                };
                sum_work(&mut self.stats.child_execution, &work)?;
                if let Some(profile) = profile {
                    sum_profile(&mut self.profile, profile);
                }
                self.stats.child_methods.push(AffinePowerChildMethod{propagators:settings.propagators,powers:settings.powers.clone(),requested:settings.algorithm,
                    actual:actual.as_ref().map(|(a,_)|*a),rows:packed_group.input.rows,notes:actual.map_or_else(||vec!["nested constant-one affine power descent; inspect nested lane diagnostics".into()],|(_,n)|n)});
                for (row, origin) in packed_group.origins.iter().enumerate() {
                    if let Some(notes) = notes.get(row) {
                        self.diagnostics[lanes[origin.parent_row]].extend(notes.iter().map(|n| {
                            format!(
                                "terminal {} powers {:?}: {n}",
                                origin.terminal_index, settings.powers
                            )
                        }));
                    }
                }
                match evaluated {
                    Ok(value) => {
                        #[cfg(test)]
                        if let Some(observations) = &mut self.component_probe_children {
                            observations.push(AffinePowerChildObservation {
                                settings: settings.clone(),
                                input: packed_group.input.clone(),
                                result: value.result().clone(),
                                quadrature_origins: (0..value.result().rows())
                                    .map(|row| value.origin(row, self.prototype.get_precision(), self.accuracy_bits))
                                    .collect(),
                                quadrature_errors: (0..value.result().rows())
                                    .map(|row| value.errors(row, self.prototype.get_precision(), self.accuracy_bits).map(<[Complex<T>]>::to_vec))
                                    .collect(),
                                origins: packed_group
                                    .origins
                                    .iter()
                                    .map(|origin| (lanes[origin.parent_row], origin.terminal_index))
                                    .collect(),
                            });
                        }
                        children.push(Some(value));
                    }
                    Err(e) => {
                        for origin in &packed_group.origins {
                            self.diagnostics[lanes[origin.parent_row]]
                                .push(format!("child batch failed: {e}"));
                        }
                        children.push(None);
                    }
                }
            }
            let output_width = mul(mul(self.plan.basis.len(), 4)?, self.components)?;
            let mut parent_lanes = Vec::new();
            let mut parent_input = KinematicsBatch {
                rows: 0,
                components: self.components,
                values: Vec::new(),
            };
            let mut geometry = KinematicsBatch {
                rows: 0,
                components: self.components,
                values: Vec::new(),
            };
            let mut gathered = blank_result(
                &self.plan.basis,
                &self.metadata,
                self.components,
                mul(lanes.len(), prepared.dag.terminals.len())?,
                &self.prototype,
            )?;
            let mut gathered_context = QuadratureErrorContext::missing(gathered.rows());
            let mut source_rows = Vec::new();
            let mut origins = Vec::new();
            for (row, &lane) in lanes.iter().enumerate() {
                let mut validated = Vec::new();
                let mut reason = None;
                let mut has_quadrature = false;
                for terminal in 0..prepared.dag.terminals.len() {
                    let location = packed.require_location(row, terminal)?;
                    let Some(child) = children[location.group].as_ref() else {
                        reason = Some(format!(
                            "terminal {terminal} child preparation/evaluation unavailable"
                        ));
                        break;
                    };
                    let source = child.result();
                    if let Err(e) = validate_layout(
                        source,
                        &self.plan.basis,
                        &self.metadata,
                        self.components,
                        packed.groups[location.group].input.rows,
                    ) {
                        reason = Some(e.to_string());
                        break;
                    }
                    result.fallback_used[lane] |= source.fallback_used[location.child_row];
                    result.analytic_used[lane] |= source.analytic_used[location.child_row];
                    let values = match source.lane(location.child_row) {
                        Ok(v) => v,
                        Err(status) => {
                            reason = Some(format!("terminal {terminal}: {status:?}"));
                            break;
                        }
                    };
                    let envelope = match source.validated_primal_envelope(
                        location.child_row,
                        self.prototype.get_precision(),
                        self.accuracy_bits,
                    ) {
                        Ok(e) => e,
                        Err(e) => {
                            reason = Some(format!(
                                "terminal {terminal} actual-route primal context unavailable: {e}"
                            ));
                            break;
                        }
                    };
                    let amplitudes = if self.components == 1 {
                        envelope
                            .operands
                            .iter()
                            .map(|o| o.amplitude.clone())
                            .collect::<Vec<_>>()
                    } else {
                        match source.validated_component_amplitudes(
                            location.child_row,
                            self.prototype.get_precision(),
                            self.accuracy_bits,
                        ) {
                            Ok(v) => v.to_vec(),
                            Err(e) => {
                                reason = Some(format!(
                                    "terminal {terminal} full-component context unavailable: {e}"
                                ));
                                break;
                            }
                        }
                    };
                    let errors = match child.errors(
                        location.child_row,
                        self.prototype.get_precision(),
                        self.accuracy_bits,
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            reason = Some(format!(
                                "terminal {terminal} actual-route quadrature context unavailable: {e}"
                            ));
                            break;
                        }
                    };
                    let origin = child
                        .origin(
                            location.child_row,
                            self.prototype.get_precision(),
                            self.accuracy_bits,
                        )
                        .map_err(ReductionError::Evaluation)?;
                    has_quadrature |= origin == QuadratureErrorOrigin::PropagatedEuclidean;
                    validated.push((
                        source,
                        location.child_row,
                        values,
                        amplitudes,
                        errors,
                        origin,
                    ));
                }
                if reason.is_none() && self.components > 1 && has_quadrature {
                    reason=Some("full-Dual imported quadrature and roundoff joint budget is not yet qualified".into());
                }
                if let Some(reason) = reason {
                    self.diagnostics[lane].push(reason.clone());
                    result.statuses[lane] = LaneStatus::Unstable(reason);
                    continue;
                }
                let compact = parent_lanes.len();
                parent_lanes.push(lane);
                origins.push(has_quadrature);
                geometry.values.extend_from_slice(
                    &physical.values[row * input_width..(row + 1) * input_width],
                );
                geometry.rows += 1;
                parent_input.values.extend_from_slice(
                    &physical.values[row * input_width..(row + 1) * input_width],
                );
                parent_input.rows += 1;
                for (_, _, values, _, _, _) in &validated {
                    parent_input.values.extend_from_slice(values);
                }
                for (_, _, _, amplitudes, _, _) in &validated {
                    parent_input.values.extend_from_slice(amplitudes);
                }
                let mut mapping = Vec::new();
                for (terminal, (source, source_row, values, _, errors, origin)) in
                    validated.into_iter().enumerate()
                {
                    let target = compact * prepared.dag.terminals.len() + terminal;
                    mapping.push(target);
                    gathered.values[target * output_width..(target + 1) * output_width]
                        .clone_from_slice(values);
                    gathered.statuses[target] = LaneStatus::Success;
                    gathered.fallback_used[target] = source.fallback_used[source_row];
                    gathered.analytic_used[target] = source.analytic_used[source_row];
                    gathered_context
                        .install_dense(
                            &gathered,
                            target,
                            self.prototype.get_precision(),
                            self.accuracy_bits,
                            errors,
                            origin,
                        )
                        .map_err(ReductionError::Evaluation)?;
                    result.fallback_used[lane] |= source.fallback_used[source_row];
                    result.analytic_used[lane] |= source.analytic_used[source_row];
                }
                source_rows.push(mapping);
            }
            if parent_lanes.is_empty() {
                continue;
            }
            let gathered = CoupledQuadratureBatch::new(gathered, gathered_context)?;
            let profile = self
                .profiles
                .get_mut(&key)
                .expect("prepared affine profile");
            self.stats.parent_batch_calls = add(self.stats.parent_batch_calls, 1)?;
            self.stats.parent_rows = add(self.stats.parent_rows, parent_input.rows)?;
            let parent = profile.parent.reduce_batch(&parent_input);
            sum_work(
                &mut self.stats.parent_execution,
                &profile.parent.last_execution(),
            )?;
            if let Some(p) = profile.parent.last_profile() {
                sum_profile(&mut self.profile, p);
            }
            for (compact, &lane) in parent_lanes.iter().enumerate() {
                if let Some(notes) = profile.parent.last_diagnostics().get(compact) {
                    self.diagnostics[lane].extend_from_slice(notes);
                }
            }
            let parent = match parent {
                Ok(v) => v,
                Err(e) => {
                    for &lane in &parent_lanes {
                        result.statuses[lane] =
                            LaneStatus::MasterFailure(format!("affine parent batch failed: {e}"));
                    }
                    continue;
                }
            };
            validate_layout(
                &parent,
                &self.plan.basis,
                &self.metadata,
                self.components,
                parent_lanes.len(),
            )?;
            profile.quadrature.evaluate(
                &geometry,
                &gathered,
                &source_rows,
                &self.plan.basis,
                &self.metadata,
                &self.prototype,
                self.accuracy_bits,
            )?;
            self.stats.quadrature_work_rows = add(
                self.stats.quadrature_work_rows,
                profile.quadrature.last_work_rows(),
            )?;
            for (compact, &lane) in parent_lanes.iter().enumerate() {
                result.statuses[lane] = parent.statuses[compact].clone();
                result.analytic_used[lane] = true;
                let Ok(values) = parent.lane(compact) else {
                    continue;
                };
                if let Some(reason) = profile.quadrature.failure(compact) {
                    self.diagnostics[lane].push(reason.to_string());
                    result.statuses[lane] = LaneStatus::Unstable(reason.to_string());
                    continue;
                }
                let mut errors = profile
                    .quadrature
                    .lane(compact, self.components)
                    .ok_or_else(|| prep("missing affine propagated error lane"))?
                    .to_vec();
                // Existing final-output UV power counting made these exact
                // Symbolica zeros; no numerical cancellation or primal-only
                // branch test is used to erase an uncertainty component.
                for (slot, &is_zero) in prepared.parent.exact_zero_outputs.iter().enumerate() {
                    if is_zero {
                        for value in
                            &mut errors[slot * self.components..(slot + 1) * self.components]
                        {
                            *value = Complex::new(self.prototype.zero(), self.prototype.zero());
                        }
                    }
                }
                let parent_width = mul(prepared.parent.plan.parameter_count(), self.components)?;
                let checked = profile.parent.inspect_imported_quadrature_output(
                    &prepared.parent.checks,
                    &parent,
                    compact,
                    &parent_input.values[compact * parent_width..(compact + 1) * parent_width],
                    &errors,
                );
                match checked {
                    Err(reason) | Ok(Some(reason)) => {
                        self.diagnostics[lane].push(reason.clone());
                        result.statuses[lane] = LaneStatus::Unstable(reason);
                        continue;
                    }
                    Ok(None) => {}
                }
                result.values[lane * output_width..(lane + 1) * output_width]
                    .clone_from_slice(values);
                result.inherit_primal_envelope(
                    lane,
                    &parent,
                    compact,
                    PrimalEnvelopeRoute::Analytic,
                );
                if let Err(reason) = result.validated_primal_envelope(
                    lane,
                    self.prototype.get_precision(),
                    self.accuracy_bits,
                ) {
                    self.diagnostics[lane].push(reason.clone());
                    result.statuses[lane] = LaneStatus::Unstable(reason);
                    continue;
                }
                let origin = if origins[compact] {
                    QuadratureErrorOrigin::PropagatedEuclidean
                } else {
                    QuadratureErrorOrigin::KnownNoQuadrature
                };
                context
                    .install_dense(
                        &result,
                        lane,
                        self.prototype.get_precision(),
                        self.accuracy_bits,
                        &errors,
                        origin,
                    )
                    .map_err(ReductionError::Evaluation)?;
            }
        }
        CoupledQuadratureBatch::new(result, context)
    }
}

fn invalid(message: impl Into<String>) -> ReductionError {
    ReductionError::InvalidInput(message.into())
}
fn prep(message: impl Into<String>) -> ReductionError {
    ReductionError::Preparation(message.into())
}
fn mul(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("affine dimensions overflow"))
}
fn add(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("affine counter overflow"))
}
fn parameters(n: usize) -> Result<usize, ReductionError> {
    mul(n, 5)?
        .checked_add(1)
        .ok_or_else(|| invalid("affine parameter count overflow"))
}
fn analytic_order(shape: &Shape) -> Result<u16, ReductionError> {
    shape.as_ref().map_or(Ok(0), |s| {
        s.iter().try_fold(0u16, |maximum, row| {
            row.iter()
                .try_fold(0usize, |sum, &v| sum.checked_add(v))
                .and_then(|v| u16::try_from(v).ok())
                .map(|v| maximum.max(v))
                .ok_or_else(|| invalid("affine analytic order exceeds u16"))
        })
    })
}
fn workspace<T: NativeFloat>(
    plan: &ReductionPlan,
    prototype: &T,
    shape: &Shape,
    policy: WorkspacePolicy,
    masters: MasterExecutionPolicy,
) -> Result<Workspace<T>, ReductionError>
where
    Complex<T>: EvaluationDomain,
{
    match shape {
        None => Workspace::new_with_policies(plan, prototype.clone(), policy, masters),
        Some(s) => Workspace::new_dual_with_policies(
            plan,
            prototype.clone(),
            HyperDual::<u8>::new(s.clone()),
            policy,
            masters,
        ),
    }
}
fn blank_result<T: NativeFloat>(
    basis: &[Monomial],
    metadata: &ResultMetadata,
    components: usize,
    rows: usize,
    prototype: &T,
) -> Result<BatchResult<T>, ReductionError> {
    let length = mul(mul(mul(rows, basis.len())?, 4)?, components)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| invalid("affine result allocation failed"))?;
    values.resize(length, Complex::new(prototype.zero(), prototype.zero()));
    Ok(BatchResult {
        basis: basis.to_vec(),
        components,
        metadata: metadata.clone(),
        values,
        statuses: vec![LaneStatus::Unstable("affine output was not evaluated".into()); rows],
        fallback_used: vec![false; rows],
        analytic_used: vec![false; rows],
        primal_envelopes: vec![None; rows],
        primal_envelope_errors: vec![None; rows],
    })
}
fn validate_layout<T: NativeFloat>(
    result: &BatchResult<T>,
    basis: &[Monomial],
    metadata: &ResultMetadata,
    components: usize,
    rows: usize,
) -> Result<(), ReductionError> {
    if result.basis != basis
        || result.metadata() != metadata
        || result.components != components
        || result.rows() != rows
        || result.values.len() != mul(mul(mul(rows, basis.len())?, 4)?, components)?
        || result.fallback_used.len() != rows
        || result.analytic_used.len() != rows
        || result.primal_envelopes.len() != rows
        || result.primal_envelope_errors.len() != rows
    {
        return Err(prep("affine child complete result interpretation differs"));
    }
    Ok(())
}
fn sum_work(target: &mut ExecutionStats, source: &ExecutionStats) -> Result<(), ReductionError> {
    target.input_rows = add(target.input_rows, source.input_rows)?;
    target.argument_rows = add(target.argument_rows, source.argument_rows)?;
    target.master_family_batches = add(target.master_family_batches, source.master_family_batches)?;
    target.master_rows = add(target.master_rows, source.master_rows)?;
    target.pentagon_box_rows = add(target.pentagon_box_rows, source.pentagon_box_rows)?;
    target.reduction_rows = add(target.reduction_rows, source.reduction_rows)?;
    target.fallback_rows = add(target.fallback_rows, source.fallback_rows)?;
    target.fallback_successes = add(target.fallback_successes, source.fallback_successes)?;
    target.analytic_rows = add(target.analytic_rows, source.analytic_rows)?;
    Ok(())
}
fn sum_profile(target: &mut ExecutionProfile, source: ExecutionProfile) {
    target.arguments += source.arguments;
    target.scalar_masters += source.scalar_masters;
    target.reduction += source.reduction;
}

#[cfg(test)]
#[path = "affine_power/tests.rs"]
mod tests;

// Finite test-only public-facade capture; independent jets helper is reused.
#[cfg(test)]
#[path = "affine_power/numeric_probe.rs"]
mod numeric_probe;
