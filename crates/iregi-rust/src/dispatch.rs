//! Shared tensor-reduction front door, independent of graph preparation.
//!
//! Settings retain their requested semantics. Ordinary PV requests use the
//! existing public six-/seven-point wrappers only when their own domain guards
//! accept them. Preserved Cartesian positive-power requests may instead use
//! constant-denominator affine descent before the low-point planner. That
//! composite retains explicit Duals, profiling and requested master policies;
//! old thin wrappers use generic execution for those capabilities. No numerical
//! result is cached here.

use std::sync::{Arc, OnceLock};

#[cfg(test)]
use crate::affine_power::AffinePowerChildObservation;

use iregi_masters::{MasterBackend, MasterExecutionPolicy, NativeFloat};
use symbolica::{
    domains::{dual::DualNumberStructure, float::Complex},
    evaluate::EvaluationDomain,
};

use crate::{
    AffinePowerExecutionStats, AffinePowerReductionPlan, AffinePowerWorkspace, BatchResult,
    CertificateExecutionStats, CertifiedTensorReductionPlan, CertifiedTensorWorkspace,
    ContractionPlan, ExecutionProfile, ExecutionStats, InputLayout, KinematicsBatch,
    MassTaylorExecutionStats, MassTaylorReductionPlan, MassTaylorWorkspace, ReductionError,
    ReductionPlan, ReductionSettings, RescueScalar, RescueSettings, SevenPointExecutionStats,
    SevenPointReductionPlan, SevenPointWorkspace, SixPointExecutionStats, SixPointReductionPlan,
    SixPointWorkspace, TensorCertificateExecutionStats, TensorRescueWorkspace, Workspace,
    basis::Monomial,
};

mod six_hybrid;
use six_hybrid::{SixHybridPlan, SixHybridWorkspace};

mod legacy_noop;
use legacy_noop::{LegacyNoOpPlan, LegacyNoOpWorkspace, highpoint_legacy_domain};

/// The actual reduction implementation, distinct from its requested algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorEngineKind {
    Generic,
    SixPoint,
    SixPointHybrid,
    CertifiedTensor,
    SevenPoint,
    AffinePower,
    MassTaylor,
    LegacyNoOp,
}

#[derive(Clone, Debug)]
enum OrdinaryPlan {
    Generic(Arc<ReductionPlan>),
    SixPoint(Arc<SixPointReductionPlan>),
    SixPointHybrid(Arc<SixHybridPlan>),
    CertifiedTensor(Arc<CertifiedTensorReductionPlan>),
    SevenPoint(Arc<SevenPointReductionPlan>),
    AffinePower(Arc<AffinePowerReductionPlan>),
    MassTaylor(Arc<MassTaylorReductionPlan>),
    LegacyNoOp(Arc<LegacyNoOpPlan>),
}

/// Immutable ordinary dispatch with a separately cached generic structural plan.
///
/// `new` prepares the selected ordinary plan eagerly. Existing thin wrappers
/// use a separately cached generic plan for unsupported workspace capabilities.
/// Affine-power composition keeps its own engine for Duals, profiling, native
/// policies and ordinary precision rescue; it never prepares the full generic
/// high-point graph implicitly. Explicit generic_plan() remains a separate request.
#[derive(Clone, Debug)]
pub struct TensorReductionPlan {
    requested_settings: ReductionSettings,
    ordinary: OrdinaryPlan,
    generic: Arc<OnceLock<Result<Arc<ReductionPlan>, ReductionError>>>,
    affine: Arc<OnceLock<Result<Arc<AffinePowerReductionPlan>, ReductionError>>>,
    notes: Vec<String>,
}

fn ordinary_kind(settings: &ReductionSettings) -> TensorEngineKind {
    // This proof-only front stage must precede generic recurrence preparation.
    if highpoint_legacy_domain(settings) {
        TensorEngineKind::LegacyNoOp
    } else if crate::six_point::validate_settings(settings).is_ok() {
        TensorEngineKind::SixPoint
    } else if settings.algorithm == crate::Algorithm::Hybrid && {
        let mut primary = settings.clone();
        primary.algorithm = crate::Algorithm::PassarinoVeltman;
        crate::six_point::validate_settings(&primary).is_ok()
    } {
        TensorEngineKind::SixPointHybrid
    } else if crate::seven_point::validate_settings(settings).is_ok() {
        TensorEngineKind::SevenPoint
    } else if affine_power_eligible(settings) {
        TensorEngineKind::AffinePower
    } else {
        TensorEngineKind::Generic
    }
}

/// Facade selection domain; actual charts and resource bounds are checked by
/// the affine plan/guard. There is no mathematical maximum point count here.
pub(crate) fn affine_power_eligible(settings: &ReductionSettings) -> bool {
    settings.propagators >= 6
        && settings.rank <= 6
        && settings.powers.len() == settings.propagators
        && !settings.powers.contains(&0)
        && settings.invariants == crate::InvariantInput::FromShifts
        && settings.input_policy == crate::InputPolicy::Preserve
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
    /// Explicit auxiliary mass-Taylor strategy for N5, h=0 positive powers.
    ///
    /// Differentiated squared masses must be nonzero on admitted lanes. This
    /// analytic-domain restriction belongs to this explicit strategy; default
    /// dispatch retains direct recurrences for dimensionally regulated dots.
    pub fn new_mass_taylor(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let plan = Arc::new(MassTaylorReductionPlan::new(settings)?);
        Ok(Self {
            requested_settings: plan.settings().clone(),
            notes: plan.preparation_notes().to_vec(),
            ordinary: OrdinaryPlan::MassTaylor(plan),
            generic: Arc::new(OnceLock::new()),
            affine: Arc::new(OnceLock::new()),
        })
    }

    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let requested_settings = settings.clone();
        let ordinary = match ordinary_kind(&settings) {
            TensorEngineKind::MassTaylor => {
                OrdinaryPlan::MassTaylor(Arc::new(MassTaylorReductionPlan::new(settings)?))
            }
            TensorEngineKind::Generic => {
                OrdinaryPlan::Generic(Arc::new(ReductionPlan::new(settings)?))
            }
            TensorEngineKind::SixPoint => {
                OrdinaryPlan::SixPoint(Arc::new(SixPointReductionPlan::new(settings)?))
            }
            TensorEngineKind::SixPointHybrid => {
                OrdinaryPlan::SixPointHybrid(Arc::new(SixHybridPlan::new(settings)?))
            }
            TensorEngineKind::CertifiedTensor => {
                unreachable!("checked tensors require the explicit constructor")
            }
            TensorEngineKind::SevenPoint => {
                OrdinaryPlan::SevenPoint(Arc::new(SevenPointReductionPlan::new(settings)?))
            }
            TensorEngineKind::AffinePower => {
                OrdinaryPlan::AffinePower(Arc::new(AffinePowerReductionPlan::new(settings)?))
            }
            TensorEngineKind::LegacyNoOp => {
                OrdinaryPlan::LegacyNoOp(Arc::new(LegacyNoOpPlan::new(settings)?))
            }
        };
        let generic = match &ordinary {
            OrdinaryPlan::Generic(plan) => Arc::new(OnceLock::from(Ok(plan.clone()))),
            _ => Arc::new(OnceLock::new()),
        };
        let affine = match &ordinary {
            OrdinaryPlan::AffinePower(plan) => Arc::new(OnceLock::from(Ok(plan.clone()))),
            _ => Arc::new(OnceLock::new()),
        };
        let notes = match &ordinary {
            OrdinaryPlan::MassTaylor(plan) => plan.preparation_notes().to_vec(),
            OrdinaryPlan::Generic(plan) => plan.preparation_notes().to_vec(),
            OrdinaryPlan::LegacyNoOp(plan) => plan.preparation_notes().to_vec(),
            OrdinaryPlan::SixPoint(_) => vec!["Facade ordinary selection: SixPointReductionPlan; specialized preparation logs are not exposed by that wrapper.".into()],
            OrdinaryPlan::SixPointHybrid(_) => vec!["Facade ordinary selection: specialized six-point PV primary with lazy genuine batched IBP fallback; requested Hybrid settings retained.".into()],
            OrdinaryPlan::CertifiedTensor(_) => unreachable!("ordinary constructor never selects checked input"),
            OrdinaryPlan::SevenPoint(_) => vec!["Facade ordinary selection: SevenPointReductionPlan; specialized preparation logs are not exposed by that wrapper.".into()],
            OrdinaryPlan::AffinePower(_) => vec!["Facade selection: affine positive-power descent into packed same-dimension child plans; actual requested/selected child algorithms and notes are retained in tagged execution metadata.".into()],
        };
        Ok(Self {
            requested_settings,
            ordinary,
            generic,
            affine,
            notes,
        })
    }

    /// Explicitly select the existing checked Cartesian supplied-product API.
    ///
    /// This constructor selects `SuppliedScalarProducts`, as does the direct
    /// CertifiedTensorReductionPlan constructor. All other requested settings
    /// are preserved. Every lane must pass both full-component compatibility
    /// checks before chart execution. Native policies and unchecked generic
    /// conversion are unavailable; this is never selected automatically by new.
    /// Generalized and Dual numeric qualification remains explicitly scoped.
    pub fn new_checked_cartesian(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let checked = Arc::new(CertifiedTensorReductionPlan::new(settings)?);
        Ok(Self {
            requested_settings: checked.settings().clone(),
            ordinary: OrdinaryPlan::CertifiedTensor(checked),
            generic: Arc::new(OnceLock::new()),
            affine: Arc::new(OnceLock::new()),
            notes: vec![
                "Facade selection: opt-in checked Cartesian supplied-G tensors; Cartesian compatibility and regular rank-four Gram certificates precede each accepted chart dispatch.".into(),
                "Checked tensor workspaces use Symbolica. Numeric qualification of generalized or ordered Dual requests is separate from API acceptance.".into(),
            ],
        })
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.requested_settings
    }

    /// Eligible ordinary engine. Capabilities may send old thin wrappers to
    /// generic execution; affine composition preserves its own selected engine.
    pub fn engine_kind(&self) -> TensorEngineKind {
        match &self.ordinary {
            OrdinaryPlan::Generic(_) => TensorEngineKind::Generic,
            OrdinaryPlan::MassTaylor(_) => TensorEngineKind::MassTaylor,
            OrdinaryPlan::SixPoint(_) => TensorEngineKind::SixPoint,
            OrdinaryPlan::SixPointHybrid(_) => TensorEngineKind::SixPointHybrid,
            OrdinaryPlan::CertifiedTensor(_) => TensorEngineKind::CertifiedTensor,
            OrdinaryPlan::SevenPoint(_) => TensorEngineKind::SevenPoint,
            OrdinaryPlan::AffinePower(_) => TensorEngineKind::AffinePower,
            OrdinaryPlan::LegacyNoOp(_) => TensorEngineKind::LegacyNoOp,
        }
    }

    pub fn basis(&self) -> &[Monomial] {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => plan.basis(),
            OrdinaryPlan::SixPoint(plan) => plan.basis(),
            OrdinaryPlan::SixPointHybrid(plan) => plan.primary.basis(),
            OrdinaryPlan::CertifiedTensor(plan) => plan.basis(),
            OrdinaryPlan::SevenPoint(plan) => plan.basis(),
            OrdinaryPlan::AffinePower(plan) => plan.basis(),
            OrdinaryPlan::MassTaylor(plan) => plan.basis(),
            OrdinaryPlan::LegacyNoOp(plan) => plan.basis(),
        }
    }

    pub fn input_layout(&self) -> &InputLayout {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => plan.input_layout(),
            OrdinaryPlan::SixPoint(plan) => plan.input_layout(),
            OrdinaryPlan::SixPointHybrid(plan) => plan.primary.input_layout(),
            OrdinaryPlan::CertifiedTensor(plan) => plan.input_layout(),
            OrdinaryPlan::SevenPoint(plan) => plan.input_layout(),
            OrdinaryPlan::AffinePower(plan) => plan.input_layout(),
            OrdinaryPlan::MassTaylor(plan) => plan.input_layout(),
            OrdinaryPlan::LegacyNoOp(plan) => plan.input_layout(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        match &self.ordinary {
            OrdinaryPlan::Generic(plan) => plan.parameter_count(),
            OrdinaryPlan::SixPoint(plan) => plan.parameter_count(),
            OrdinaryPlan::SixPointHybrid(plan) => plan.primary.parameter_count(),
            OrdinaryPlan::CertifiedTensor(plan) => plan.parameter_count(),
            OrdinaryPlan::SevenPoint(plan) => plan.parameter_count(),
            OrdinaryPlan::AffinePower(plan) => plan.parameter_count(),
            OrdinaryPlan::MassTaylor(plan) => plan.parameter_count(),
            OrdinaryPlan::LegacyNoOp(plan) => plan.parameter_count(),
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
            OrdinaryPlan::SixPointHybrid(plan) => plan.primary.contraction_plan(monomials),
            OrdinaryPlan::CertifiedTensor(plan) => plan.contraction_plan(monomials),
            OrdinaryPlan::SevenPoint(plan) => plan.contraction_plan(monomials),
            OrdinaryPlan::AffinePower(plan) => plan.contraction_plan(monomials),
            OrdinaryPlan::MassTaylor(plan) => plan.contraction_plan(monomials),
            OrdinaryPlan::LegacyNoOp(plan) => plan.contraction_plan(monomials),
        }
    }

    /// Reuse only immutable generic preparation, preserving error variants.
    /// For LegacyNoOp this explicitly requests the original Legacy settings;
    /// it never bypasses the no-op certificate by requesting a Preserve graph.
    pub fn generic_plan(&self) -> Result<Arc<ReductionPlan>, ReductionError> {
        if matches!(&self.ordinary, OrdinaryPlan::CertifiedTensor(_)) {
            return Err(ReductionError::Unsupported(
                "checked Cartesian plans cannot bypass their certificates through an unchecked generic plan".into()));
        }
        self.generic
            .get_or_init(|| ReductionPlan::new(self.requested_settings.clone()).map(Arc::new))
            .as_ref()
            .map(Arc::clone)
            .map_err(clone_error)
    }

    /// Immutable alternate composition recipe for capabilities unsupported by
    /// old thin wrappers. Its shape/child caches are shared across workspaces;
    /// requesting it never constructs the generic full high-point recurrence.
    fn affine_plan(&self) -> Result<Arc<AffinePowerReductionPlan>, ReductionError> {
        if !affine_power_eligible(&self.requested_settings) {
            return Err(ReductionError::Unsupported(
                "settings are outside the affine-power facade domain".into(),
            ));
        }
        self.affine
            .get_or_init(|| {
                AffinePowerReductionPlan::new(self.requested_settings.clone()).map(Arc::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(clone_error)
    }

    /// Precision rescue through the same selected ordinary engine. Specialized
    /// stages retain their actual engine details. Explicit Dual rescue stays
    /// generic in existing low-point domains; affine joint-context rescue is
    /// explicitly unsupported until precision transport is qualified.
    /// Master-policy/profiling options are not accepted by the rescue API.
    pub fn rescue_workspace<T: RescueScalar>(
        &self,
        prototype: T,
        settings: RescueSettings,
    ) -> Result<TensorRescueWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        TensorRescueWorkspace::new(self, prototype, settings)
    }

    pub fn rescue_dual_workspace<T: RescueScalar, D: DualNumberStructure>(
        &self,
        prototype: T,
        settings: RescueSettings,
        dual: D,
    ) -> Result<TensorRescueWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        TensorRescueWorkspace::new_dual(self, prototype, settings, dual)
    }
}

/// Capabilities requested when constructing an owned numerical workspace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorWorkspaceOptions {
    /// Preserve the compiled default backend, or explicitly select Symbolica
    /// to permit eligible old thin wrappers. Affine composition forwards native
    /// policies to its actual child workspaces instead of forcing generic.
    pub master_policy: MasterExecutionPolicy,
    /// Enable the generic profiler or the affine composite's child/parent profiler.
    /// Existing thin wrappers require a generic workspace for profiling.
    pub profiling: bool,
}

impl TensorWorkspaceOptions {
    fn validate_checked(self) -> Result<(), ReductionError> {
        let symbolic = match self.master_policy {
            MasterExecutionPolicy::Symbolica => true,
            MasterExecutionPolicy::Default => {
                iregi_masters::default_master_backend() == MasterBackend::SymbolicaBatch
            }
            MasterExecutionPolicy::NativeF64Abc => false,
        };
        if !symbolic {
            return Err(ReductionError::Unsupported(
                "checked Cartesian workspaces require the Symbolica master policy; native policies cannot bypass the supplied-product certificates".into()));
        }
        Ok(())
    }

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
    SixPointHybrid(SixHybridWorkspace<T>),
    CertifiedTensor {
        workspace: CertifiedTensorWorkspace<T>,
        binary_precision: u32,
    },
    SevenPoint(SevenPointWorkspace<T>),
    AffinePower(AffinePowerWorkspace<T>),
    MassTaylor(MassTaylorWorkspace<T>),
    LegacyNoOp(Box<LegacyNoOpWorkspace<T>>),
}

/// Exact tagged execution detail, including actual nested certificate work.
#[derive(Clone, Debug)]
pub enum TensorEngineExecution {
    Generic {
        execution: ExecutionStats,
        certificate_execution: CertificateExecutionStats,
    },
    SixPoint(SixPointExecutionStats),
    CertifiedTensor {
        execution: ExecutionStats,
        certificate_execution: TensorCertificateExecutionStats,
    },
    SixPointHybrid {
        primary: SixPointExecutionStats,
        fallback: ExecutionStats,
        fallback_certificate_execution: CertificateExecutionStats,
        fallback_rows: usize,
        fallback_successes: usize,
        fallback_batches: usize,
        fallback_preparation_attempts: usize,
        fallback_preparation_error: Option<String>,
    },
    SevenPoint(SevenPointExecutionStats),
    /// Composite counters retain actual child methods, powers and notes.
    AffinePower(AffinePowerExecutionStats),
    /// Mass-Taylor extraction followed by a fresh dotted-dimension guard.
    MassTaylor(MassTaylorExecutionStats),
    /// Conservative no-op proof for requested Legacy snapping. Certificate
    /// work is separate from the actual, lazily prepared Preserve child.
    LegacyNoOp {
        input_rows: usize,
        accepted_rows: usize,
        rejected_rows: usize,
        query_count: usize,
        certificate_execution: CertificateExecutionStats,
        child: Option<Box<TensorEngineExecution>>,
        child_preparation_attempts: usize,
        child_preparation_error: Option<String>,
    },
}

impl TensorEngineExecution {
    /// Logical input lanes with child/parent work counted once.
    ///
    /// Fallback and analytic counts measure executed work and can exceed the
    /// logical parent-lane count. Packed row counts and certificate counters
    /// remain in the tagged detail and are not added to these work fields.
    pub fn summary(&self) -> Result<ExecutionStats, ReductionError> {
        match self {
            Self::Generic { execution, .. } | Self::CertifiedTensor { execution, .. } => {
                Ok(*execution)
            }
            Self::SixPoint(stats) => summarize(
                stats.parent_input_rows,
                stats.child_execution,
                stats.parent_execution,
            ),
            Self::SixPointHybrid {
                primary,
                fallback,
                fallback_rows,
                fallback_successes,
                ..
            } => {
                let primary = summarize(
                    primary.parent_input_rows,
                    primary.child_execution,
                    primary.parent_execution,
                )?;
                let mut result = summarize(primary.input_rows, primary, *fallback)?;
                result.fallback_rows = result
                    .fallback_rows
                    .checked_add(*fallback_rows)
                    .ok_or_else(counter_overflow)?;
                result.fallback_successes = result
                    .fallback_successes
                    .checked_add(*fallback_successes)
                    .ok_or_else(counter_overflow)?;
                Ok(result)
            }
            Self::SevenPoint(stats) => summarize(
                stats.parent_input_rows,
                stats.child_execution,
                stats.parent_execution,
            ),
            Self::AffinePower(stats) => summarize(
                stats.input_rows,
                stats.child_execution,
                stats.parent_execution,
            ),
            Self::MassTaylor(stats) => summarize(
                stats.input_rows,
                stats.unit_execution,
                stats.projection_execution,
            ),
            Self::LegacyNoOp {
                input_rows, child, ..
            } => {
                let mut result = match child {
                    Some(child) => child.summary()?,
                    None => ExecutionStats::default(),
                };
                result.input_rows = *input_rows;
                Ok(result)
            }
        }
    }
}

fn counter_overflow() -> ReductionError {
    ReductionError::Evaluation("tensor execution summary counter overflow".into())
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
        if let OrdinaryPlan::LegacyNoOp(legacy) = &plan.ordinary {
            let workspace = LegacyNoOpWorkspace::new(legacy.clone(), prototype, options)?;
            return Ok(Self {
                engine: WorkspaceEngine::LegacyNoOp(Box::new(workspace)),
                requested_master_policy: options.master_policy,
                notes: plan.preparation_notes().to_vec(),
            });
        }
        if let OrdinaryPlan::CertifiedTensor(checked) = &plan.ordinary {
            options.validate_checked()?;
            let binary_precision = prototype.get_precision();
            let mut workspace = CertifiedTensorWorkspace::new(checked, prototype)?;
            workspace.set_profiling(options.profiling);
            return Ok(Self {
                engine: WorkspaceEngine::CertifiedTensor {
                    workspace,
                    binary_precision,
                },
                requested_master_policy: options.master_policy,
                notes: plan.preparation_notes().to_vec(),
            });
        }
        options
            .master_policy
            .validate_available()
            .map_err(|e| ReductionError::Preparation(e.to_string()))?;
        if let OrdinaryPlan::MassTaylor(mass) = &plan.ordinary {
            let mut workspace = MassTaylorWorkspace::new_with_master_policy(
                mass,
                prototype,
                options.master_policy,
            )?;
            workspace.set_profiling(options.profiling);
            return Ok(Self {
                engine: WorkspaceEngine::MassTaylor(workspace),
                requested_master_policy: options.master_policy,
                notes: mass.preparation_notes().to_vec(),
            });
        }
        if matches!(&plan.ordinary, OrdinaryPlan::AffinePower(_)) {
            return Self::affine(plan, prototype, options);
        }
        if !options.permits_thin() {
            if affine_power_eligible(plan.settings()) {
                return Self::affine(plan, prototype, options);
            }
            return Self::generic(plan, prototype, options);
        }
        let engine = match &plan.ordinary {
            OrdinaryPlan::MassTaylor(_) => {
                unreachable!("mass-Taylor workspace returns before thin dispatch")
            }
            OrdinaryPlan::Generic(generic) => WorkspaceEngine::Generic(
                Workspace::new_with_master_policy(generic, prototype, options.master_policy)?,
            ),
            OrdinaryPlan::SixPoint(six) => {
                WorkspaceEngine::SixPoint(SixPointWorkspace::new(six, prototype)?)
            }
            OrdinaryPlan::SixPointHybrid(six) => {
                WorkspaceEngine::SixPointHybrid(SixHybridWorkspace::new(six.clone(), prototype)?)
            }
            OrdinaryPlan::CertifiedTensor(_) => {
                unreachable!("checked workspace returns before generic policy dispatch")
            }
            OrdinaryPlan::AffinePower(_) => {
                unreachable!("affine workspace returns before thin policy dispatch")
            }
            OrdinaryPlan::LegacyNoOp(_) => {
                unreachable!("Legacy no-op workspace returns before generic policy dispatch")
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
        if let OrdinaryPlan::LegacyNoOp(legacy) = &plan.ordinary {
            let workspace =
                LegacyNoOpWorkspace::new_dual(legacy.clone(), prototype, dual, options)?;
            return Ok(Self {
                engine: WorkspaceEngine::LegacyNoOp(Box::new(workspace)),
                requested_master_policy: options.master_policy,
                notes: plan.preparation_notes().to_vec(),
            });
        }
        if let OrdinaryPlan::CertifiedTensor(checked) = &plan.ordinary {
            options.validate_checked()?;
            let binary_precision = prototype.get_precision();
            let mut workspace = CertifiedTensorWorkspace::new_dual(checked, prototype, dual)?;
            workspace.set_profiling(options.profiling);
            let mut notes = plan.preparation_notes().to_vec();
            notes.push("Facade checked workspace retains the existing full ordered Dual certificates; this does not qualify private thin highpoint Dual engines.".into());
            return Ok(Self {
                engine: WorkspaceEngine::CertifiedTensor {
                    workspace,
                    binary_precision,
                },
                requested_master_policy: options.master_policy,
                notes,
            });
        }
        options
            .master_policy
            .validate_available()
            .map_err(|e| ReductionError::Preparation(e.to_string()))?;
        if let OrdinaryPlan::MassTaylor(mass) = &plan.ordinary {
            let mut workspace = MassTaylorWorkspace::new_dual_with_master_policy(
                mass,
                prototype,
                dual,
                options.master_policy,
            )?;
            workspace.set_profiling(options.profiling);
            return Ok(Self {
                engine: WorkspaceEngine::MassTaylor(workspace),
                requested_master_policy: options.master_policy,
                notes: mass.preparation_notes().to_vec(),
            });
        }
        if affine_power_eligible(plan.settings()) {
            let affine = plan.affine_plan()?;
            let mut workspace = AffinePowerWorkspace::new_dual_with_master_policy(
                &affine,
                prototype,
                dual,
                options.master_policy,
            )?;
            workspace.set_profiling(options.profiling);
            let mut notes = vec!["Facade actual workspace: AffinePower; supplied ordered Dual layout and selected master policy retained; propagated-quadrature joint-Dual acceptance remains explicitly guarded.".into()];
            notes.extend_from_slice(plan.preparation_notes());
            return Ok(Self {
                engine: WorkspaceEngine::AffinePower(workspace),
                requested_master_policy: options.master_policy,
                notes,
            });
        }
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

    fn affine(
        plan: &TensorReductionPlan,
        prototype: T,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        let affine = plan.affine_plan()?;
        let mut workspace = AffinePowerWorkspace::new_with_master_policy(
            &affine,
            prototype,
            options.master_policy,
        )?;
        workspace.set_profiling(options.profiling);
        let mut notes = vec![format!(
            "Facade actual workspace: AffinePower; profiling={}, requested master policy={:?}.",
            options.profiling, options.master_policy,
        )];
        notes.extend_from_slice(plan.preparation_notes());
        Ok(Self {
            engine: WorkspaceEngine::AffinePower(workspace),
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
            WorkspaceEngine::MassTaylor(_) => TensorEngineKind::MassTaylor,
            WorkspaceEngine::SixPoint(_) => TensorEngineKind::SixPoint,
            WorkspaceEngine::SixPointHybrid(_) => TensorEngineKind::SixPointHybrid,
            WorkspaceEngine::CertifiedTensor { .. } => TensorEngineKind::CertifiedTensor,
            WorkspaceEngine::SevenPoint(_) => TensorEngineKind::SevenPoint,
            WorkspaceEngine::AffinePower(_) => TensorEngineKind::AffinePower,
            WorkspaceEngine::LegacyNoOp(_) => TensorEngineKind::LegacyNoOp,
        }
    }

    /// Requested policy, not a claim about an observed scalar backend. The
    /// old thin wrappers explicitly use Symbolica internally; affine composites
    /// forward the requested policy to their actual child workspaces.
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
            WorkspaceEngine::SixPointHybrid(workspace) => workspace.set_accuracy_precision(bits),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => {
                workspace.set_accuracy_precision(bits)
            }
            WorkspaceEngine::SevenPoint(workspace) => workspace.set_accuracy_precision(bits),
            WorkspaceEngine::AffinePower(workspace) => workspace.set_accuracy_precision(bits),
            WorkspaceEngine::MassTaylor(workspace) => workspace.set_accuracy_precision(bits),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.set_accuracy_precision(bits),
        }
    }

    /// Set the refinement budget of both exact certificates for checked input.
    /// Other engines do not expose this opt-in certificate operation.
    pub fn set_certificate_precision_budget(
        &mut self,
        initial: u32,
        maximum: u32,
    ) -> Result<(), ReductionError> {
        match &mut self.engine {
            WorkspaceEngine::CertifiedTensor { workspace, .. } => {
                workspace.set_certificate_precision_budget(initial, maximum)
            }
            _ => Err(ReductionError::Unsupported(
                "certificate refinement budgets require an opt-in checked Cartesian workspace"
                    .into(),
            )),
        }
    }

    /// Affine workspaces profile their actual child and parent stages. Old thin
    /// workspaces require profiling at construction to select the generic path.
    /// Disabling profiling is always supported.
    pub fn set_profiling(&mut self, enabled: bool) -> Result<(), ReductionError> {
        match &mut self.engine {
            WorkspaceEngine::Generic(workspace) => {
                workspace.set_profiling(enabled);
                Ok(())
            }
            WorkspaceEngine::CertifiedTensor { workspace, .. } => {
                workspace.set_profiling(enabled);
                Ok(())
            }
            WorkspaceEngine::MassTaylor(workspace) => {
                workspace.set_profiling(enabled);
                Ok(())
            }
            WorkspaceEngine::AffinePower(workspace) => {
                workspace.set_profiling(enabled);
                Ok(())
            }
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.set_profiling(enabled),
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
            WorkspaceEngine::SixPointHybrid(workspace) => workspace.component_count(),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => workspace.component_count(),
            WorkspaceEngine::SevenPoint(workspace) => workspace.component_count(),
            WorkspaceEngine::AffinePower(workspace) => workspace.component_count(),
            WorkspaceEngine::MassTaylor(workspace) => workspace.component_count(),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.component_count(),
        }
    }

    /// Ordered Taylor directions for generic, checked, or affine execution.
    /// The ordinary six-/seven-point thin wrappers accept scalar inputs only.
    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.dual_shape(),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => workspace.dual_shape(),
            WorkspaceEngine::AffinePower(workspace) => workspace.dual_shape(),
            WorkspaceEngine::MassTaylor(workspace) => workspace.dual_shape(),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.dual_shape(),
            WorkspaceEngine::SixPoint(_)
            | WorkspaceEngine::SixPointHybrid(_)
            | WorkspaceEngine::SevenPoint(_) => None,
        }
    }

    pub fn binary_precision(&self) -> u32 {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.binary_precision(),
            WorkspaceEngine::SixPoint(workspace) => workspace.binary_precision(),
            WorkspaceEngine::SixPointHybrid(workspace) => workspace.binary_precision(),
            WorkspaceEngine::CertifiedTensor {
                binary_precision, ..
            } => *binary_precision,
            WorkspaceEngine::SevenPoint(workspace) => workspace.binary_precision(),
            WorkspaceEngine::AffinePower(workspace) => workspace.binary_precision(),
            WorkspaceEngine::MassTaylor(workspace) => workspace.binary_precision(),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.binary_precision(),
        }
    }

    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::SixPoint(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::SixPointHybrid(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => workspace.last_diagnostics(),
            WorkspaceEngine::SevenPoint(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::AffinePower(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::MassTaylor(workspace) => workspace.last_diagnostics(),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.last_diagnostics(),
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
            WorkspaceEngine::SixPointHybrid(workspace) => workspace.engine_execution(),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => {
                TensorEngineExecution::CertifiedTensor {
                    execution: workspace.last_execution(),
                    certificate_execution: workspace.last_certificate_execution(),
                }
            }
            WorkspaceEngine::SevenPoint(workspace) => {
                TensorEngineExecution::SevenPoint(workspace.last_execution().clone())
            }
            WorkspaceEngine::AffinePower(workspace) => {
                TensorEngineExecution::AffinePower(workspace.last_execution().clone())
            }
            WorkspaceEngine::MassTaylor(workspace) => {
                TensorEngineExecution::MassTaylor(workspace.last_execution().clone())
            }
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.engine_execution(),
        }
    }

    /// Private qualification observer; absent from release builds and disabled
    /// by default. It retains complete real child inputs/results when enabled.
    #[cfg(test)]
    pub(crate) fn enable_affine_power_child_observer(
        &mut self,
        enabled: bool,
    ) -> Result<(), ReductionError> {
        match &mut self.engine {
            WorkspaceEngine::AffinePower(workspace) => {
                workspace.enable_component_child_observer(enabled);
                Ok(())
            }
            _ => Err(ReductionError::Unsupported(
                "child observations require an actual affine-power workspace".into(),
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn affine_power_child_observations(
        &self,
    ) -> Option<&[AffinePowerChildObservation<T>]> {
        match &self.engine {
            WorkspaceEngine::AffinePower(workspace) => workspace.component_child_observations(),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn affine_power_parent_observation(
        &self,
    ) -> Option<&crate::execution::quadrature_context::CoupledQuadratureBatch<T>> {
        match &self.engine {
            WorkspaceEngine::AffinePower(workspace) => workspace.component_parent_observation(),
            _ => None,
        }
    }

    pub fn execution_summary(&self) -> Result<ExecutionStats, ReductionError> {
        self.engine_execution().summary()
    }

    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        match &self.engine {
            WorkspaceEngine::Generic(workspace) => workspace.last_profile(),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => workspace.last_profile(),
            WorkspaceEngine::AffinePower(workspace) => workspace.last_profile(),
            WorkspaceEngine::MassTaylor(workspace) => workspace.last_profile(),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.last_profile(),
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
            WorkspaceEngine::SixPointHybrid(workspace) => workspace.reduce_batch(batch),
            WorkspaceEngine::CertifiedTensor { workspace, .. } => workspace.reduce_batch(batch),
            WorkspaceEngine::SevenPoint(workspace) => workspace.reduce_batch(batch),
            WorkspaceEngine::AffinePower(workspace) => workspace.reduce_batch(batch),
            WorkspaceEngine::MassTaylor(workspace) => workspace.reduce_batch(batch),
            WorkspaceEngine::LegacyNoOp(workspace) => workspace.reduce_batch(batch),
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
            for algorithm in [Algorithm::Ibp, Algorithm::Hybrid]
                .into_iter()
                .filter(|algorithm| n != 6 || *algorithm != Algorithm::Hybrid)
            {
                let mut s = settings.clone();
                s.algorithm = algorithm;
                variants.push((s, TensorEngineKind::AffinePower));
            }
            for convention in [Convention::OneLoop, Convention::MadLoop] {
                let mut s = settings.clone();
                s.convention = convention;
                variants.push((s, TensorEngineKind::AffinePower));
            }
            let mut s = settings.clone();
            s.tensor_weights = TensorWeights::Multinomial;
            variants.push((s, TensorEngineKind::AffinePower));
            let mut s = settings.clone();
            s.invariants = InvariantInput::SuppliedScalarProducts;
            variants.push((s, TensorEngineKind::Generic));
            let mut s = settings.clone();
            s.input_policy = InputPolicy::legacy_onshell();
            variants.push((
                s,
                if n == 6 {
                    TensorEngineKind::LegacyNoOp
                } else {
                    TensorEngineKind::Generic
                },
            ));
            let mut s = settings.clone();
            s.powers[0] = 2;
            variants.push((s, TensorEngineKind::AffinePower));
            let mut s = settings.clone();
            s.powers.pop();
            variants.push((s, TensorEngineKind::Generic));
            let mut s = settings.clone();
            s.dimension_shift = 1;
            variants.push((s, TensorEngineKind::AffinePower));
            let mut s = settings.clone();
            s.dimension_shift = -1;
            variants.push((s, TensorEngineKind::AffinePower));
            let mut s = settings.clone();
            s.rank = if n == 6 { 8 } else { 1 };
            variants.push((
                s,
                if n == 6 {
                    TensorEngineKind::Generic
                } else {
                    TensorEngineKind::AffinePower
                },
            ));
            for (variant, expected) in variants {
                let before = serde_json::to_value(&variant).unwrap();
                assert_eq!(ordinary_kind(&variant), expected);
                assert_eq!(serde_json::to_value(&variant).unwrap(), before);
            }
        }
        for n in 1..=7 {
            assert_eq!(
                ordinary_kind(&ReductionSettings::new(n, 0)),
                if n == 6 {
                    TensorEngineKind::SixPointHybrid
                } else if n == 7 {
                    TensorEngineKind::AffinePower
                } else {
                    TensorEngineKind::Generic
                }
            );
        }
        let requested = ReductionSettings::new(6, 6);
        let before = serde_json::to_value(&requested).unwrap();
        assert_eq!(ordinary_kind(&requested), TensorEngineKind::SixPointHybrid);
        assert_eq!(serde_json::to_value(&requested).unwrap(), before);
    }

    #[test]
    fn legacy_noop_selection_precedes_highpoint_generic_preparation() {
        for n in [5, 6] {
            for algorithm in [
                Algorithm::Ibp,
                Algorithm::PassarinoVeltman,
                Algorithm::Hybrid,
            ] {
                for rank in [0, 6] {
                    let mut requested = ReductionSettings::new(n, rank);
                    requested.input_policy = InputPolicy::legacy_onshell();
                    requested.algorithm = algorithm;
                    requested.powers[0] = 2;
                    let before = serde_json::to_value(&requested).unwrap();
                    assert_eq!(ordinary_kind(&requested), TensorEngineKind::LegacyNoOp);
                    assert_eq!(serde_json::to_value(&requested).unwrap(), before);
                }
            }
        }
        let mut seven = ReductionSettings::new(7, 6);
        seven.input_policy = InputPolicy::legacy_onshell();
        assert_eq!(ordinary_kind(&seven), TensorEngineKind::Generic);
        let mut supplied = ReductionSettings::new(6, 6);
        supplied.input_policy = InputPolicy::legacy_onshell();
        supplied.invariants = InvariantInput::SuppliedScalarProducts;
        assert_eq!(ordinary_kind(&supplied), TensorEngineKind::Generic);
        let mut zero_power = ReductionSettings::new(5, 0);
        zero_power.input_policy = InputPolicy::legacy_onshell();
        zero_power.powers[0] = 0;
        assert_eq!(ordinary_kind(&zero_power), TensorEngineKind::Generic);
    }

    #[test]
    fn legacy_noop_constructor_retains_requested_settings_without_generic_graph() {
        for n in [5, 6] {
            let mut settings = ReductionSettings::new(n, 6);
            settings.input_policy = InputPolicy::legacy_onshell();
            settings.powers[0] = 2;
            settings.algorithm = Algorithm::Ibp;
            let before = serde_json::to_value(&settings).unwrap();
            let plan = TensorReductionPlan::new(settings).unwrap();
            assert_eq!(plan.engine_kind(), TensorEngineKind::LegacyNoOp);
            assert_eq!(serde_json::to_value(plan.settings()).unwrap(), before);
            assert!(plan.generic.get().is_none());
            assert!(plan.affine.get().is_none());
            assert_eq!(plan.parameter_count(), 5 * n + 1);
            assert_eq!(plan.basis().len(), 210);
            assert!(plan.contraction_plan(&[Monomial([0, 0, 0, 0])]).is_ok());
            assert!(plan.generic.get().is_none());
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
        let affine = TensorEngineExecution::AffinePower(AffinePowerExecutionStats {
            input_rows: 3,
            packed_child_rows: 3000,
            quadrature_work_rows: 840,
            child_execution: child,
            parent_execution: parent,
            ..Default::default()
        });
        assert_eq!(affine.summary().unwrap(), expected);
        let mut enormous = child;
        enormous.master_rows = usize::MAX;
        assert!(matches!(
            summarize(3, enormous, parent),
            Err(ReductionError::Evaluation(_))
        ));
        assert!(
            TensorEngineExecution::AffinePower(AffinePowerExecutionStats {
                input_rows: 3,
                child_execution: enormous,
                parent_execution: parent,
                ..Default::default()
            })
            .summary()
            .is_err()
        );
    }

    #[test]
    fn legacy_noop_summary_retains_outer_rows_and_counts_child_once() {
        let child_execution = ExecutionStats {
            input_rows: 2,
            argument_rows: 17,
            master_family_batches: 3,
            master_rows: 11,
            pentagon_box_rows: 5,
            reduction_rows: 23,
            fallback_rows: 7,
            fallback_successes: 6,
            analytic_rows: 29,
        };
        let execution = TensorEngineExecution::LegacyNoOp {
            input_rows: 5,
            accepted_rows: 2,
            rejected_rows: 3,
            query_count: 101,
            certificate_execution: CertificateExecutionStats::default(),
            child: Some(Box::new(TensorEngineExecution::Generic {
                execution: child_execution,
                certificate_execution: CertificateExecutionStats::default(),
            })),
            child_preparation_attempts: 1,
            child_preparation_error: None,
        };
        assert_eq!(
            execution.summary().unwrap(),
            ExecutionStats {
                input_rows: 5,
                ..child_execution
            }
        );
        let rejected = TensorEngineExecution::LegacyNoOp {
            input_rows: 5,
            accepted_rows: 0,
            rejected_rows: 5,
            query_count: 101,
            certificate_execution: CertificateExecutionStats::default(),
            child: None,
            child_preparation_attempts: 0,
            child_preparation_error: None,
        };
        assert_eq!(
            rejected.summary().unwrap(),
            ExecutionStats {
                input_rows: 5,
                ..Default::default()
            }
        );
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

    #[test]
    #[ignore = "separately bounded construction of three actual specialized precision stages"]
    fn specialized_precision_rescue_does_not_prepare_unrelated_generic_plan() {
        if iregi_masters::default_master_backend() != MasterBackend::SymbolicaBatch {
            return;
        }
        let plan = TensorReductionPlan::new(pv(6, 0)).unwrap();
        assert!(plan.generic.get().is_none());
        let rescue = plan
            .rescue_workspace(0., RescueSettings::default())
            .unwrap();
        assert_eq!(rescue.engine_kind(), TensorEngineKind::SixPoint);
        assert_eq!(rescue.algorithm_used(), Algorithm::PassarinoVeltman);
        assert!(rescue.primary_engine_execution().is_none());
        assert!(rescue.working_engine_execution().is_none());
        assert!(rescue.verification_engine_execution().is_none());
        assert!(rescue.last_primary_statuses().is_empty());
        assert!(rescue.last_rescue_statuses().is_empty());
        assert_eq!(rescue.last_execution(), crate::RescueStats::default());
        assert!(plan.generic.get().is_none());
    }
}

#[cfg(test)]
mod checked_tests;

#[cfg(test)]
mod affine_power_tests;

#[cfg(test)]
#[path = "dispatch/mass_taylor_tests.rs"]
mod mass_taylor_tests;
