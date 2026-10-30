//! Direct numerical tensor reduction for unit-power, unshifted requests.
//!
//! The symbolic planners compile one expression per Cartesian moment and
//! interpret it per call. This engine instead stores the recurrence structure
//! of the validated PV planner (`pv.rs`) as short linear instructions and
//! evaluates them numerically per lane, with scalar masters batched across
//! lanes. It targets ordinary binary64 throughput. Lanes whose Gram, Cayley or
//! six-point coordinate systems are ill-conditioned report `Unstable` without
//! coefficients, so callers can rerun them on a robust engine.

mod exec;
mod numeric;
mod plan;

use std::sync::Arc;

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

pub use exec::{DirectMasters, DirectProfile};
use exec::DirectExecutor;
use plan::DirectPlanData;

use crate::{BatchResult, KinematicsBatch, ReductionError, ReductionSettings, basis::Monomial};

/// Immutable structural plan of the direct engine. No sampled value is stored.
#[derive(Clone, Debug)]
pub struct DirectReductionPlan {
    data: Arc<DirectPlanData>,
}

impl DirectReductionPlan {
    /// Whether `settings` lie inside the direct engine's implemented domain.
    pub fn supports(settings: &ReductionSettings) -> bool {
        DirectPlanData::supported(settings).is_ok()
    }

    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Ok(Self {
            data: Arc::new(DirectPlanData::new(settings)?),
        })
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.data.settings
    }

    pub fn basis(&self) -> &[Monomial] {
        &self.data.basis
    }

    /// Number of recurrence instructions and their total term count.
    pub fn instruction_counts(&self) -> (usize, usize) {
        (
            self.data.coefficient_program.len() + self.data.moment_program.len(),
            self.data.terms.len(),
        )
    }

    /// Workspace with the default direct OneLOop master kernels.
    pub fn workspace<T: NativeFloat>(&self, prototype: T) -> Result<DirectWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        self.workspace_with_masters(prototype, DirectMasters::Native)
    }

    /// Workspace with batched `MasterEvaluator` masters under `policy`.
    pub fn workspace_with_policy<T: NativeFloat>(
        &self,
        prototype: T,
        policy: MasterExecutionPolicy,
    ) -> Result<DirectWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        self.workspace_with_masters(prototype, DirectMasters::Policy(policy))
    }

    pub fn workspace_with_masters<T: NativeFloat>(
        &self,
        prototype: T,
        masters: DirectMasters,
    ) -> Result<DirectWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        Ok(DirectWorkspace {
            executor: DirectExecutor::new(self.data.clone(), prototype, masters)?,
        })
    }
}

/// Owned numerical scratch storage for one direct plan.
pub struct DirectWorkspace<T: NativeFloat> {
    executor: DirectExecutor<T>,
}

impl<T: NativeFloat> DirectWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.executor.reduce_batch(batch)
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.executor.set_profiling(enabled);
    }

    pub fn last_profile(&self) -> DirectProfile {
        self.executor.last_profile()
    }

    /// Minimal accepted relative determinant (raised to the recursion depth).
    pub fn set_stability_threshold(&mut self, threshold: T) {
        self.executor.set_stability_threshold(threshold);
    }
}
