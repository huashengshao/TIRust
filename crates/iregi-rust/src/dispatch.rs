//! Shared tensor-reduction front door, independent of graph preparation.
//!
//! Settings retain their requested semantics. Ordinary PV requests use the
//! existing public six-/seven-point wrappers only when their own domain guards
//! accept them. Explicit Duals, profiling and native master-policy requests
//! use the existing generic implementation. A feature-dependent native default
//! also retains generic execution. No numerical result is cached here.

use std::sync::{Arc, OnceLock};

use iregi_masters::{MasterBackend, MasterExecutionPolicy, NativeFloat};
use symbolica::{
    domains::{dual::DualNumberStructure, float::Complex},
    evaluate::EvaluationDomain,
};

use crate::{
    BatchResult, CertificateExecutionStats, ContractionPlan, ExecutionProfile, ExecutionStats,
    InputLayout, KinematicsBatch, PrecisionRescueWorkspace, ReductionError, ReductionPlan,
    ReductionSettings, RescueScalar, RescueSettings, SevenPointExecutionStats,
    SevenPointReductionPlan, SevenPointWorkspace, SixPointExecutionStats, SixPointReductionPlan,
    SixPointWorkspace, Workspace, basis::Monomial,
};

/// The actual reduction implementation, distinct from its requested algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorEngineKind {
    Generic,
    SixPoint,
    SevenPoint,
}

#[derive(Clone, Debug)]
enum OrdinaryPlan {
    Generic(Arc<ReductionPlan>),
    SixPoint(Arc<SixPointReductionPlan>),
    SevenPoint(Arc<SevenPointReductionPlan>),
}

/// Immutable ordinary dispatch with a separately cached generic structural plan.
///
/// `new` prepares the selected ordinary plan eagerly. A later Dual, profiling,
/// rescue or generic-only policy request lazily prepares the generic plan with the
/// same requested settings. Thus an ordinary thin-plan preparation error is
/// returned before any later workspace request; it is never silently retried.
#[derive(Clone, Debug)]
pub struct TensorReductionPlan {
    requested_settings: ReductionSettings,
    ordinary: OrdinaryPlan,
    generic: Arc<OnceLock<Result<Arc<ReductionPlan>, ReductionError>>>,
    notes: Vec<String>,
}

fn ordinary_kind(settings: &ReductionSettings) -> TensorEngineKind {
    if crate::six_point::validate_settings(settings).is_ok() {
        TensorEngineKind::SixPoint
    } else if crate::seven_point::validate_settings(settings).is_ok() {
        TensorEngineKind::SevenPoint
    } else {
        TensorEngineKind::Generic
    }
}

fn clone_error(error: &ReductionError) -> ReductionError {
    match error {
        ReductionError::InvalidInput(text) => ReductionError::InvalidInput(text.clone()),
        ReductionError::Preparation(text) => ReductionError::Preparation(text.clone()),
        ReductionError::Unsupported(text) => ReductionError::Unsupported(text.clone()),
        ReductionError::Evaluation(text) => ReductionError::Evaluation(text.clone()),
    }
}

impl TensorReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let requested_settings = settings.clone();
        let ordinary = match ordinary_kind(&settings) {
            TensorEngineKind::Generic => {
                OrdinaryPlan::Generic(Arc::new(ReductionPlan::new(settings)?))
            }
            TensorEngineKind::SixPoint => {
                OrdinaryPlan::SixPoint(Arc::new(SixPointReductionPlan::new(settings)?))
            }
            TensorEngineKind::SevenPoint => {
                OrdinaryPlan::SevenPoint(Arc::new(SevenPointReductionPlan::new(settings)?))
            }
        };
        let generic = match &ordinary {
            OrdinaryPlan::Generic(plan) => Arc::new(OnceLock::from(Ok(plan.clone()))),
            _ => Arc::new(OnceLock::new()),
        };
        let notes = match &ordinary {
            OrdinaryPlan::Generic(plan) => plan.preparation_notes().to_vec(),
            OrdinaryPlan::SixPoint(_) => vec!["Facade ordinary selection: SixPointReductionPlan; specialized preparation logs are not exposed by that wrapper.".into()],
            OrdinaryPlan::SevenPoint(_) => vec!["Facade ordinary selection: SevenPointReductionPlan; specialized preparation logs are not exposed by that wrapper.".into()],
        };
        Ok(Self {
            requested_settings,
            ordinary,
            generic,
            notes,
        })
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.requested_settings
    }

    /// Eligible specialized ordinary engine. Workspace capabilities and the
    /// feature-dependent default master backend may select generic execution.
    pub fn engine_kind(&self) -> TensorEngineKind {
        match &self.ordinary {
            OrdinaryPlan::Generic(_) => TensorEngineKind::Generic,
            OrdinaryPlan::SixPoint(_) => TensorEngineKind::SixPoint,
            OrdinaryPlan::SevenPoint(_) => TensorEngineKind::SevenPoint,
        }
    }

    pub fn basis(&self) -> &[Monomial] {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => plan.basis(),
            OrdinaryPlan::SixPoint(plan) => plan.basis(),
            OrdinaryPlan::SevenPoint(plan) => plan.basis(),
        }
    }

    pub fn input_layout(&self) -> &InputLayout {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => plan.input_layout(),
            OrdinaryPlan::SixPoint(plan) => plan.input_layout(),
            OrdinaryPlan::SevenPoint(plan) => plan.input_layout(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => plan.parameter_count(),
            OrdinaryPlan::SixPoint(plan) => plan.parameter_count(),
            OrdinaryPlan::SevenPoint(plan) => plan.parameter_count(),
        }
    }

    /// Existing generic preparation notes or factual specialized-selection metadata.
    pub fn preparation_notes(&self) -> &[String] {
        &self.notes
    }

    pub fn contraction_plan(
        &self,
        monomials: &[Monomial],
    ) -> Result<ContractionPlan, ReductionError> {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => ContractionPlan::new(plan, monomials),
            OrdinaryPlan::SixPoint(plan) => plan.contraction_plan(monomials),
            OrdinaryPlan::SevenPoint(plan) => plan.contraction_plan(monomials),
        }
    }

    /// Reuse only immutable generic preparation, preserving error variants.
    pub fn generic_plan(&self) -> Result<Arc<ReductionPlan>, ReductionError> {
        self.generic
            .get_or_init(|| ReductionPlan::new(self.requested_settings.clone()).map(Arc::new))
            .as_ref()
            .map(Arc::clone)
            .map_err(clone_error)
    }

    /// Existing generic precision rescue, without widening ordinary numeric bounds.
    /// Master-policy/profiling options are not accepted by the rescue API.
    pub fn rescue_workspace<T: RescueScalar>(
        &self,
        prototype: T,
        settings: RescueSettings,
    ) -> Result<PrecisionRescueWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        let generic = self.generic_plan()?;
        PrecisionRescueWorkspace::new(generic.as_ref(), prototype, settings)
    }

    pub fn rescue_dual_workspace<T: RescueScalar, D: DualNumberStructure>(
        &self,
        prototype: T,
        settings: RescueSettings,
        dual: D,
    ) -> Result<PrecisionRescueWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        let generic = self.generic_plan()?;
        PrecisionRescueWorkspace::new_dual(generic.as_ref(), prototype, settings, dual)
    }
}

/// Capabilities requested when constructing an owned numerical workspace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorWorkspaceOptions {
    /// Preserve the compiled default backend, or explicitly select Symbolica
    /// to permit eligible specialized wrappers. Native policies use generic.
    pub master_policy: MasterExecutionPolicy,
    /// Use the existing generic profiler, including for eligible high points.
    pub profiling: bool,
}

impl TensorWorkspaceOptions {
    fn permits_thin(self) -> bool {
        !self.profiling
            && match self.master_policy {
                MasterExecutionPolicy::Symbolica => true,
                MasterExecutionPolicy::Default => {
                    iregi_masters::default_master_backend() == MasterBackend::SymbolicaBatch
                }
                MasterExecutionPolicy::NativeF64Abc => false,
            }
    }
}

#[derive(Clone)]
enum WorkspaceEngine<T: NativeFloat> {
    Generic(Workspace<T>),
    SixPoint(SixPointWorkspace<T>),
    SevenPoint(SevenPointWorkspace<T>),
}

/// Exact tagged execution detail, including actual nested certificate work.
#[derive(Clone, Debug)]
pub enum TensorEngineExecution {
    Generic {
        execution: ExecutionStats,
        certificate_execution: CertificateExecutionStats,
    },
    SixPoint(SixPointExecutionStats),
    SevenPoint(SevenPointExecutionStats),
}

impl TensorEngineExecution {
    /// Logical input lanes with child/parent work counted once.
    ///
    /// Fallback and analytic counts measure executed work and can exceed the
    /// logical parent-lane count. Packed row counts and certificate counters
    /// remain in the tagged detail and are not added to these work fields.
    pub fn summary(&self) -> Result<ExecutionStats, ReductionError> {
        match self {
            Self::Generic { execution, .. } => Ok(*execution),
            Self::SixPoint(stats) => summarize(
                stats.parent_input_rows,
                stats.child_execution,
                stats.parent_execution,
            ),
            Self::SevenPoint(stats) => summarize(
                stats.parent_input_rows,
                stats.child_execution,
                stats.parent_execution,
            ),
        }
    }
}

fn summarize(
    input_rows: usize,
    child: ExecutionStats,
    parent: ExecutionStats,
) -> Result<ExecutionStats, ReductionError> {
    let add = |a: usize, b: usize| {
        a.checked_add(b).ok_or_else(|| {
            ReductionError::Evaluation("tensor execution summary counter overflow".into())
        })
    };
    Ok(ExecutionStats {
        input_rows,
        argument_rows: add(child.argument_rows, parent.argument_rows)?,
        master_family_batches: add(child.master_family_batches, parent.master_family_batches)?,
        master_rows: add(child.master_rows, parent.master_rows)?,
        pentagon_box_rows: add(child.pentagon_box_rows, parent.pentagon_box_rows)?,
        reduction_rows: add(child.reduction_rows, parent.reduction_rows)?,
        fallback_rows: add(child.fallback_rows, parent.fallback_rows)?,
        fallback_successes: add(child.fallback_successes, parent.fallback_successes)?,
        analytic_rows: add(child.analytic_rows, parent.analytic_rows)?,
    })
}

/// Owned batch scratch. The engine tag describes this workspace's actual path.
#[derive(Clone)]
pub struct TensorWorkspace<T: NativeFloat> {
    engine: WorkspaceEngine<T>,
    requested_master_policy: MasterExecutionPolicy,
    notes: Vec<String>,
}

impl<T: NativeFloat> TensorWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &TensorReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Self::new_with_options(plan, prototype, TensorWorkspaceOptions::default())
    }

    pub fn new_with_options(
        plan: &TensorReductionPlan,
        prototype: T,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        options
            .master_policy
            .validate_available()
            .map_err(|e| ReductionError::Preparation(e.to_string()))?;
        if !options.permits_thin() {
            return Self::generic(plan, prototype, options);
        }
        let engine = match &plan.ordinary {
            OrdinaryPlan::Generic(generic) => WorkspaceEngine::Generic(
                Workspace::new_with_master_policy(generic, prototype, options.master_policy)?,
            ),
            OrdinaryPlan::SixPoint(six) => {
                WorkspaceEngine::SixPoint(SixPointWorkspace::new(six, prototype)?)
            }
            OrdinaryPlan::SevenPoint(seven) => {
                WorkspaceEngine::SevenPoint(SevenPointWorkspace::new(seven, prototype)?)
            }
        };
        Ok(Self {
            engine,
            requested_master_policy: options.master_policy,
            notes: plan.preparation_notes().to_vec(),
        })
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &TensorReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        Self::new_dual_with_options(plan, prototype, dual, TensorWorkspaceOptions::default())
    }

    pub fn new_dual_with_options<D: DualNumberStructure>(
        plan: &TensorReductionPlan,
        prototype: T,
        dual: D,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        options
            .master_policy
            .validate_available()
            .map_err(|e| ReductionError::Preparation(e.to_string()))?;
        let generic = plan.generic_plan()?;
        let mut workspace = Workspace::new_dual_with_master_policy(
            &generic,
            prototype,
            dual,
            options.master_policy,
        )?;
        workspace.set_profiling(options.profiling);
        let mut notes = vec!["Facade workspace selection: generic; explicit Dual layout.".into()];
        notes.extend_from_slice(generic.preparation_notes());
        Ok(Self {
            engine: WorkspaceEngine::Generic(workspace),
            requested_master_policy: options.master_policy,
            notes,
        })
    }

    fn generic(
        plan: &TensorReductionPlan,
        prototype: T,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        let generic = plan.generic_plan()?;
        let mut workspace =
            Workspace::new_with_master_policy(&generic, prototype, options.master_policy)?;
        workspace.set_profiling(options.profiling);
        let mut notes = vec![format!(
            "Facade workspace selection: generic; profiling={}, requested master policy={:?}.",
            options.profiling, options.master_policy
        )];
        notes.extend_from_slice(generic.preparation_notes());
        Ok(Self {
            engine: WorkspaceEngine::Generic(workspace),
            requested_master_policy: options.master_policy,
            notes,
        })
    }

    pub fn engine_kind(&self) -> TensorEngineKind {
        match &self.engine {
            WorkspaceEngine::Generic(_) => TensorEngineKind::Generic,
            WorkspaceEngine::SixPoint(_) => TensorEngineKind::SixPoint,
            WorkspaceEngine::SevenPoint(_) => TensorEngineKind::SevenPoint,
        }
    }

    /// Requested policy, not a claim about an observed scalar backend. The
    /// specialized public wrappers explicitly use Symbolica internally.
    pub fn requested_master_policy(&self) -> MasterExecutionPolicy {
        self.requested_master_policy
    }
    pub fn preparation_notes(&self) -> &[String] {
        &self.notes
    }

    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        match &mut self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.set_accuracy_precision(bits),
            WorkspaceEngine::SixPoint(workspace) => workspace.set_accuracy_precision(bits),
            WorkspaceEngine::SevenPoint(workspace) => workspace.set_accuracy_precision(bits),
        }
    }

    /// Thin workspaces require `profiling: true` at construction to select the
    /// existing generic profiler. Disabling profiling is always supported.
    pub fn set_profiling(&mut self, enabled: bool) -> Result<(), ReductionError> {
        match &mut self.engine {
            WorkspaceEngine::Generic(workspace) => {
                workspace.set_profiling(enabled);
                Ok(())
            }
            _ if !enabled => Ok(()),
            _ => Err(ReductionError::Unsupported(
                "construct TensorWorkspace with profiling enabled to use the generic profiler"
                    .into(),
            )),
        }
    }

    pub fn component_count(&self) -> usize {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.component_count(),
            WorkspaceEngine::SixPoint(workspace) => workspace.component_count(),
            WorkspaceEngine::SevenPoint(workspace) => workspace.component_count(),
        }
    }

    pub fn binary_precision(&self) -> u32 {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.binary_precision(),
            WorkspaceEngine::SixPoint(workspace) => workspace.binary_precision(),
            WorkspaceEngine::SevenPoint(workspace) => workspace.binary_precision(),
        }
    }

    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::SixPoint(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::SevenPoint(workspace) => workspace.last_diagnostics(),
        }
    }

    pub fn engine_execution(&self) -> TensorEngineExecution {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => TensorEngineExecution::Generic {
                execution: workspace.last_execution(),
                certificate_execution: workspace.last_planar_certificate_execution(),
            },
            WorkspaceEngine::SixPoint(workspace) => {
                TensorEngineExecution::SixPoint(workspace.last_execution().clone())
            }
            WorkspaceEngine::SevenPoint(workspace) => {
                TensorEngineExecution::SevenPoint(workspace.last_execution().clone())
            }
        }
    }

    pub fn execution_summary(&self) -> Result<ExecutionStats, ReductionError> {
        self.engine_execution().summary()
    }

    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.last_profile(),
            _ => None,
        }
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        match &mut self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.reduce_batch(batch),
            WorkspaceEngine::SixPoint(workspace) => workspace.reduce_batch(batch),
            WorkspaceEngine::SevenPoint(workspace) => workspace.reduce_batch(batch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Algorithm, ContractionWorkspace, Convention, InputPolicy, InvariantInput, LaneStatus,
        NumeratorBatch, TensorWeights,
    };
    use symbolica::domains::dual::HyperDual;

    fn pv(n: usize, rank: u16) -> ReductionSettings {
        let mut settings = ReductionSettings::new(n, rank);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings
    }

    #[test]
    fn cheap_selection_uses_wrapper_domains_and_preserves_settings() {
        for rank in 0..=7 {
            assert_eq!(ordinary_kind(&pv(6, rank)), TensorEngineKind::SixPoint);
        }
        assert_eq!(ordinary_kind(&pv(7, 0)), TensorEngineKind::SevenPoint);
        for (n, rank) in [(6, 3), (7, 0)] {
            let settings = pv(n, rank);
            let mut variants = Vec::new();
            for algorithm in [Algorithm::Ibp, Algorithm::Hybrid] {
                let mut s = settings.clone();
                s.algorithm = algorithm;
                variants.push(s);
            }
            for convention in [Convention::OneLoop, Convention::MadLoop] {
                let mut s = settings.clone();
                s.convention = convention;
                variants.push(s);
            }
            let mut s = settings.clone();
            s.tensor_weights = TensorWeights::Multinomial;
            variants.push(s);
            let mut s = settings.clone();
            s.invariants = InvariantInput::SuppliedScalarProducts;
            variants.push(s);
            let mut s = settings.clone();
            s.input_policy = InputPolicy::legacy_onshell();
            variants.push(s);
            let mut s = settings.clone();
            s.powers[0] = 2;
            variants.push(s);
            let mut s = settings.clone();
            s.powers.pop();
            variants.push(s);
            let mut s = settings.clone();
            s.dimension_shift = 1;
            variants.push(s);
            let mut s = settings.clone();
            s.dimension_shift = -1;
            variants.push(s);
            let mut s = settings.clone();
            s.rank = if n == 6 { 8 } else { 1 };
            variants.push(s);
            for variant in variants {
                let before = serde_json::to_value(&variant).unwrap();
                assert_eq!(ordinary_kind(&variant), TensorEngineKind::Generic);
                assert_eq!(serde_json::to_value(&variant).unwrap(), before);
            }
        }
        for n in 1..=7 {
            assert_eq!(
                ordinary_kind(&ReductionSettings::new(n, 0)),
                TensorEngineKind::Generic
            );
        }
    }

    #[test]
    fn options_keep_generic_only_capabilities_explicit() {
        assert_eq!(
            TensorWorkspaceOptions::default().permits_thin(),
            iregi_masters::default_master_backend() == MasterBackend::SymbolicaBatch
        );
        assert!(
            TensorWorkspaceOptions {
                master_policy: MasterExecutionPolicy::Symbolica,
                profiling: false
            }
            .permits_thin()
        );
        assert!(
            !TensorWorkspaceOptions {
                profiling: true,
                ..Default::default()
            }
            .permits_thin()
        );
        assert!(
            !TensorWorkspaceOptions {
                master_policy: MasterExecutionPolicy::NativeF64Abc,
                ..Default::default()
            }
            .permits_thin()
        );
    }

    #[test]
    fn nested_summary_preserves_logical_rows_and_rejects_overflow() {
        let child = ExecutionStats {
            input_rows: 21,
            argument_rows: 22,
            master_family_batches: 3,
            master_rows: 24,
            pentagon_box_rows: 25,
            reduction_rows: 26,
            fallback_rows: 27,
            fallback_successes: 28,
            analytic_rows: 29,
        };
        let parent = ExecutionStats {
            input_rows: 2,
            argument_rows: 4,
            master_family_batches: 5,
            master_rows: 6,
            pentagon_box_rows: 7,
            reduction_rows: 8,
            fallback_rows: 9,
            fallback_successes: 10,
            analytic_rows: 11,
        };
        let expected = ExecutionStats {
            input_rows: 3,
            argument_rows: 26,
            master_family_batches: 8,
            master_rows: 30,
            pentagon_box_rows: 32,
            reduction_rows: 34,
            fallback_rows: 36,
            fallback_successes: 38,
            analytic_rows: 40,
        };
        let six = TensorEngineExecution::SixPoint(SixPointExecutionStats {
            parent_input_rows: 3,
            packed_child_rows: 1000,
            child_execution: child,
            parent_execution: parent,
            ..Default::default()
        });
        let seven = TensorEngineExecution::SevenPoint(SevenPointExecutionStats {
            parent_input_rows: 3,
            packed_child_rows: 2000,
            child_execution: child,
            parent_execution: parent,
            ..Default::default()
        });
        assert_eq!(six.summary().unwrap(), expected);
        assert_eq!(seven.summary().unwrap(), expected);
        let mut enormous = child;
        enormous.master_rows = usize::MAX;
        assert!(matches!(
            summarize(3, enormous, parent),
            Err(ReductionError::Evaluation(_))
        ));
    }

    fn tadpole_batch() -> KinematicsBatch<f64> {
        let c = |x| Complex::new(x, 0.);
        KinematicsBatch {
            rows: 2,
            components: 1,
            values: [0., 0., 0., 0., 2., 3., 0., 0., 0., 0., 2., 0.]
                .map(c)
                .to_vec(),
        }
    }

    #[test]
    fn generic_batch_contraction_and_state_match_existing_api() {
        let settings = ReductionSettings::new(1, 1);
        let facade = TensorReductionPlan::new(settings.clone()).unwrap();
        let generic = facade.generic_plan().unwrap();
        assert!(Arc::ptr_eq(&generic, &facade.generic_plan().unwrap()));
        assert_eq!(facade.settings().algorithm, Algorithm::Hybrid);
        assert_eq!(facade.parameter_count(), generic.parameter_count());
        assert_eq!(facade.basis(), generic.basis());
        assert_eq!(facade.input_layout().scale, generic.input_layout().scale);
        let mut direct = Workspace::new(&generic, 0.).unwrap();
        let mut workspace = TensorWorkspace::new(&facade, 0.).unwrap();
        let input = tadpole_batch();
        let before = input.values.clone();
        let expected = direct.reduce_batch(&input).unwrap();
        let actual = workspace.reduce_batch(&input).unwrap();
        assert_eq!(workspace.engine_kind(), TensorEngineKind::Generic);
        assert_eq!(input.values, before);
        assert_eq!(actual.lane(0).unwrap(), expected.lane(0).unwrap());
        assert!(matches!(actual.lane(1), Err(LaneStatus::InvalidInput(_))));
        assert_eq!(
            workspace.execution_summary().unwrap(),
            direct.last_execution()
        );
        let support = [Monomial([0, 0, 0, 0]), Monomial([1, 0, 0, 0])];
        let mut contraction =
            ContractionWorkspace::new(&facade.contraction_plan(&support).unwrap(), 0.).unwrap();
        let numerator = NumeratorBatch {
            rows: 2,
            components: 1,
            values: [1., 2., 3., 4.].map(|x| Complex::new(x, 0.)).to_vec(),
        };
        let contracted = contraction.contract_batch(&actual, &numerator).unwrap();
        assert_eq!(
            contracted.lane(0).unwrap(),
            actual.lane(0).unwrap().get(..4).unwrap()
        );
        assert!(contracted.lane(1).is_err());
        let empty = KinematicsBatch {
            rows: 0,
            components: 1,
            values: Vec::new(),
        };
        workspace.reduce_batch(&empty).unwrap();
        assert_eq!(
            workspace.execution_summary().unwrap(),
            ExecutionStats::default()
        );
        assert!(workspace.last_diagnostics().is_empty());
        let malformed = KinematicsBatch {
            rows: 1,
            components: 1,
            values: Vec::new(),
        };
        assert!(workspace.reduce_batch(&malformed).is_err());
        assert_eq!(
            workspace.execution_summary().unwrap(),
            ExecutionStats {
                input_rows: 1,
                ..Default::default()
            }
        );
        assert!(workspace.last_diagnostics().is_empty());
    }

    #[test]
    fn explicit_dual_and_profile_options_delegate_to_generic() {
        let plan = TensorReductionPlan::new(ReductionSettings::new(1, 0)).unwrap();
        let options = TensorWorkspaceOptions {
            profiling: true,
            master_policy: MasterExecutionPolicy::Symbolica,
        };
        let shape = HyperDual::<f64>::new(vec![vec![0]]);
        let mut dual = TensorWorkspace::new_dual_with_options(&plan, 0., shape, options).unwrap();
        assert_eq!(dual.engine_kind(), TensorEngineKind::Generic);
        assert_eq!(dual.component_count(), 1);
        assert!(dual.last_profile().is_some());
        let result = dual.reduce_batch(&tadpole_batch()).unwrap();
        assert_eq!(result.metadata().dual_shape(), Some([vec![0]].as_slice()));
        assert!(result.lane(0).is_ok());
        assert!(
            dual.preparation_notes()
                .iter()
                .any(|n| n.contains("explicit Dual"))
        );
        let mut plain = TensorWorkspace::new_with_options(&plan, 0., options).unwrap();
        assert_eq!(plain.engine_kind(), TensorEngineKind::Generic);
        plain.reduce_batch(&tadpole_batch()).unwrap();
        assert!(plain.last_profile().is_some());
        plain.set_profiling(false).unwrap();
        assert!(plain.last_profile().is_none());
        assert_eq!(
            plain.requested_master_policy(),
            MasterExecutionPolicy::Symbolica
        );
    }

    #[test]
    fn explicit_master_policy_is_never_silently_replaced() {
        let plan = TensorReductionPlan::new(ReductionSettings::new(1, 0)).unwrap();
        let options = TensorWorkspaceOptions {
            master_policy: MasterExecutionPolicy::NativeF64Abc,
            profiling: false,
        };
        let result = TensorWorkspace::new_with_options(&plan, 0., options);
        if cfg!(feature = "native-f64-abc") {
            let workspace = result.unwrap();
            assert_eq!(workspace.engine_kind(), TensorEngineKind::Generic);
            assert_eq!(workspace.requested_master_policy(), options.master_policy);
        } else {
            assert!(matches!(result, Err(ReductionError::Preparation(message))
                if message.contains("native-f64-abc")));
        }
    }

    #[test]
    #[ignore = "separately bounded Symbolica Float128 N6/N7 fixture delegation"]
    fn specialized_workspaces_delegate_retained_scalar_fixtures() {
        use symbolica::domains::{
            float::{Float, SingleFloat},
            rational::Rational,
        };

        // Exact quarter units from the retained seventh_time_quarter fixture.
        // Its first six propagators are the conditioned N6 base point.
        let shifts = [
            [0, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 4],
            [2, 0, 0, 0],
            [0, 4, 4, 0],
            [1, 0, 4, 4],
        ];
        let masses = [1, 2, 3, 4, 2, 3, 4];
        for n in [6, 7] {
            let zero = Float::new(128);
            let value = |numerator: i64, denominator: i64| {
                Complex::new(
                    zero.from_rational(&Rational::from((numerator, denominator))),
                    zero.clone(),
                )
            };
            let plan = TensorReductionPlan::new(pv(n, 0)).unwrap();
            assert!(plan.generic.get().is_none());
            let mut row = Vec::new();
            for shift in &shifts[..n] {
                row.extend(shift.iter().map(|&component| value(component, 4)));
            }
            row.extend(masses[..n].iter().map(|&mass| value(mass, 1)));
            row.push(value(3, 1));
            let mut invalid = row.clone();
            *invalid.last_mut().unwrap() = value(0, 1);
            let input = KinematicsBatch {
                rows: 2,
                components: 1,
                values: row.into_iter().chain(invalid).collect(),
            };
            let original = input.values.clone();
            let mut workspace = TensorWorkspace::new_with_options(
                &plan,
                zero.clone(),
                TensorWorkspaceOptions {
                    master_policy: MasterExecutionPolicy::Symbolica,
                    profiling: false,
                },
            )
            .unwrap();
            workspace.set_accuracy_precision(53).unwrap();
            let actual = workspace.reduce_batch(&input).unwrap();
            let expected = match &plan.ordinary {
                OrdinaryPlan::SixPoint(six) => {
                    let mut direct = SixPointWorkspace::new(six, zero.clone()).unwrap();
                    direct.set_accuracy_precision(53).unwrap();
                    direct.reduce_batch(&input).unwrap()
                }
                OrdinaryPlan::SevenPoint(seven) => {
                    let mut direct = SevenPointWorkspace::new(seven, zero.clone()).unwrap();
                    direct.set_accuracy_precision(53).unwrap();
                    direct.reduce_batch(&input).unwrap()
                }
                _ => panic!("eligible scalar fixture did not select its public wrapper"),
            };
            assert_eq!(input.values, original);
            assert_eq!(actual.lane(0).unwrap(), expected.lane(0).unwrap());
            assert!(matches!(actual.lane(1), Err(LaneStatus::InvalidInput(_))));
            assert!(plan.generic.get().is_none());
            assert!(workspace.last_profile().is_none());
            assert!(matches!(
                workspace.set_profiling(true),
                Err(ReductionError::Unsupported(_))
            ));
            let summary = workspace.execution_summary().unwrap();
            assert_eq!(summary.input_rows, 2);
            match workspace.engine_execution() {
                TensorEngineExecution::SixPoint(stats) => {
                    assert_eq!(n, 6);
                    assert_eq!(stats.packed_child_rows, 6);
                    assert_eq!(stats.packed_parent_rows, 1);
                }
                TensorEngineExecution::SevenPoint(stats) => {
                    assert_eq!(n, 7);
                    assert_eq!(stats.packed_child_rows, 21);
                    assert_eq!(stats.packed_parent_rows, 1);
                }
                _ => panic!("specialized execution lost its actual tag"),
            }
            let contract = plan.contraction_plan(&[Monomial([0; 4])]).unwrap();
            let mut contraction = ContractionWorkspace::new(&contract, zero.clone()).unwrap();
            let numerator = NumeratorBatch {
                rows: 2,
                components: 1,
                values: vec![value(1, 1); 2],
            };
            let contracted = contraction.contract_batch(&actual, &numerator).unwrap();
            assert_eq!(contracted.lane(0).unwrap(), actual.lane(0).unwrap());
            assert!(contracted.lane(1).is_err());
            workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 0,
                    components: 1,
                    values: Vec::new(),
                })
                .unwrap();
            assert_eq!(
                workspace.execution_summary().unwrap(),
                ExecutionStats::default()
            );
            assert!(workspace.last_diagnostics().is_empty());
        }
    }
}
