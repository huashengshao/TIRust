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
pub(crate) mod shared;
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
    /// `(number of propagators, coefficient rank)` of every PV subset.
    pub(crate) fn subset_ranks(&self) -> Vec<(usize, u16)> {
        self.data
            .subsets
            .iter()
            .filter(|subset| subset.kind == plan::SubsetKind::Pv && subset.coefficient_rank >= 0)
            .map(|subset| (subset.members.len(), subset.coefficient_rank as u16))
            .collect()
    }

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

    /// Workspace with the default direct oneloopmaster master kernels.
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
#[derive(Clone)]
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

    /// Scalar-master evaluations of the most recent call.
    pub fn last_master_rows(&self) -> usize {
        self.executor.last_master_rows()
    }

    /// Minimal accepted relative determinant (raised to the recursion depth).
    /// Partial rescue (binary64 workspaces): boxes and triangles refused by
    /// their Gram gate are computed in double-double with acceptance down to
    /// `threshold` and injected; the rest of the lane stays binary64.
    pub fn enable_partial_rescue(&mut self, threshold: f64) {
        self.executor.enable_partial_rescue(threshold);
    }

    /// Refused boxes and triangles refuse the lane again (the default).
    pub fn disable_partial_rescue(&mut self) {
        self.executor.disable_partial_rescue();
    }

    /// Boxes refused by their Gram gate are computed in this workspace's
    /// precision by the small-Gram expansion from triangles `boost` ranks
    /// higher (Denner-Dittmaier); lanes it does not converge on are refused.
    /// The expansion is not checked here: callers verify the lanes (the
    /// public facade by the loop-rotation test).
    pub fn enable_gram_expansion(&mut self, boost: u16) {
        self.executor.enable_gram_expansion(boost);
    }

    /// Boxes and triangles refused by their Gram gate are computed by the
    /// alternative PV reduction through the modified Cayley matrix
    /// (Denner-Dittmaier sect. 3.2) only. Unchecked, like the expansion.
    pub fn enable_gram_expansion_everywhere(&mut self, boost: u16, aux_threshold: T) {
        self.executor.enable_gram_expansion_everywhere(boost, aux_threshold);
    }

    /// COLLIER-style a priori method choice: boxes and triangles whose
    /// Passarino-Veltman estimate 2^-52 w^r exceeds 2^-36 are computed by the
    /// small-Gram expansions or the alternative PV reduction (unchecked).
    pub fn enable_estimated_method_choice(&mut self, boost: u16, aux_threshold: T) {
        self.executor.enable_estimated_method_choice(boost, aux_threshold);
    }

    /// Evaluate the a priori estimate of `enable_estimated_method_choice`
    /// per lane without acting on it (every box and triangle by PV); see
    /// `last_estimate_lanes`.
    pub fn enable_estimate_flags(&mut self) {
        self.executor.enable_estimate_flags();
    }

    /// Share per-subset results with other workspaces holding the same store
    /// (binary64 only; `None`: no sharing).
    pub(crate) fn set_shared_subsets(&mut self, store: Option<std::sync::Arc<shared::SubsetStore>>) {
        self.executor.set_shared_subsets(store);
    }

    pub fn enable_alternative_pv(&mut self) {
        self.executor.enable_alternative_pv();
    }

    pub fn set_stability_threshold(&mut self, threshold: T) {
        self.executor.set_stability_threshold(threshold);
    }

    /// Per lane of the most recent call: for lanes refused on conditioning,
    /// the refused estimate (the quantity compared with the stability
    /// threshold; zero for an exactly singular pivot), else `None`. Callers
    /// use it to choose a working precision for a retry.
    pub fn last_conditions(&self) -> &[Option<f64>] {
        self.executor.last_conditions()
    }

    /// Per lane of the most recent call: some box or triangle was computed by
    /// the small-Gram expansions or the alternative PV reduction.
    pub fn last_gram_lanes(&self) -> &[bool] {
        self.executor.last_gram_lanes()
    }

    /// Per lane of the most recent call: some box or triangle exceeds the a
    /// priori estimate (with `enable_estimate_flags`).
    pub fn last_estimate_lanes(&self) -> &[bool] {
        self.executor.last_estimate_lanes()
    }
}
