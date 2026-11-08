//! Several one-loop integrals evaluated together at the same phase-space
//! points, with their common work shared.
//!
//! The integrals of an amplitude contain the same boxes and triangles: a
//! pinched integral's are its parent's. On difficult (physical) kinematics
//! most of the binary64 cost is per box or triangle: refused ones computed in
//! double-double (partial rescue) and the method-choice retries' small-Gram
//! expansions. An [`AmplitudeWorkspace`] gives every integral's workspace one
//! store of these per-subset results, keyed exactly by the method's settings
//! and the subset's propagators bit for bit, so each is computed once per
//! call. A stored result is what the integral would have computed itself:
//! results are bit-identical with sharing on and off (the
//! `share_subsets` flag), only faster on (a top-loop gg -> gggg hexagon and
//! its 56 pinches on 32 physical points: 2.7 s -> 1.4 s per call).
//!
//! Integrals need a box or triangle at different ranks (a pinched box at
//! rank 4, its parents' boxes at rank 5): inside an amplitude workspace
//! every injected one is computed at the amplitude's highest rank for its
//! size and each integral uses the leading coefficients it needs, with
//! sharing on and off alike. Results can therefore differ from a standalone
//! [`TensorWorkspace`] at the level of the expansions' truncation (both
//! checked by the rotation test); with sharing on and off they are
//! bit-identical.
//!
//! The store is cleared at the start and end of every call, so memory does
//! not grow over a Monte Carlo run.
use std::sync::Arc;

use iregi_masters::NativeFloat;
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use crate::{
    BatchResult, KinematicsBatch, ReductionError, TensorReductionPlan, TensorWorkspace,
    TensorWorkspaceOptions, direct::shared::SubsetStore,
};

/// Options of an [`AmplitudeWorkspace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmplitudeOptions {
    /// Share per-subset results between the integrals (default `true`).
    /// Results are bit-identical either way; `false` evaluates every integral
    /// independently, as separate [`TensorWorkspace`]s would.
    pub share_subsets: bool,
    /// Options of every integral's workspace.
    pub workspace: TensorWorkspaceOptions,
}

impl Default for AmplitudeOptions {
    fn default() -> Self {
        Self { share_subsets: true, workspace: TensorWorkspaceOptions::default() }
    }
}

/// Per-subset results of the last call: reused from the store and computed.
/// New counters may be added (`#[non_exhaustive]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SubsetSharingStats {
    /// Coefficient results (scalar masters, PV sub-integrals, rescues,
    /// expansions).
    pub reused: u64,
    pub computed: u64,
    /// Cartesian moments of gathered subsets (of the sizes and ranks some
    /// integral gathers inside a larger one).
    pub moments_reused: u64,
    pub moments_computed: u64,
    /// Double-double scalar masters of the partial rescues and the
    /// double-double tier (exact arguments, in each executor's own
    /// propagator order).
    pub dd_masters_reused: u64,
    pub dd_masters_computed: u64,
    /// PV sub-integrals, Gram-free pentagons and gathered moments of the
    /// double-double tiers (exact inputs, limb for limb).
    pub dd_subsets_reused: u64,
    pub dd_subsets_computed: u64,
}

/// One workspace per integral of an amplitude, evaluated together.
pub struct AmplitudeWorkspace<T: NativeFloat> {
    workspaces: Vec<TensorWorkspace<T>>,
    store: Arc<SubsetStore>,
    share: bool,
    last: SubsetSharingStats,
}

impl<T: NativeFloat> AmplitudeWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Workspaces for `plans` (one per integral) with default options
    /// (sharing on).
    pub fn new(plans: &[TensorReductionPlan], prototype: T) -> Result<Self, ReductionError> {
        Self::new_with_options(plans, prototype, AmplitudeOptions::default())
    }

    pub fn new_with_options(
        plans: &[TensorReductionPlan],
        prototype: T,
        options: AmplitudeOptions,
    ) -> Result<Self, ReductionError> {
        let workspaces = plans
            .iter()
            .map(|plan| TensorWorkspace::new_with_options(plan, prototype.clone(), options.workspace))
            .collect::<Result<Vec<_>, _>>()?;
        // Every injected box or triangle at the amplitude's highest rank for
        // its size, with sharing on and off alike (see `direct::shared`).
        let store = Arc::new(SubsetStore::default());
        // The double-double tiers share their sub-integrals, pentagons and
        // moments too: a refused lane of a large integral is usually refused
        // in some of its pinches as well.
        store.set_dd_subsets(true);
        for plan in plans {
            for (size, rank) in TensorWorkspace::<T>::direct_subset_ranks(plan) {
                store.raise_floor(size, rank);
            }
            // Moments of subsets an integral gathers inside a larger one
            // (another integral may gather them as its own).
            for (size, rank) in TensorWorkspace::<T>::direct_nested_gathers(plan) {
                store.share_moments(size, rank);
            }
        }
        let mut amplitude = Self { workspaces, store, share: false, last: SubsetSharingStats::default() };
        for workspace in &mut amplitude.workspaces {
            workspace.set_shared_subsets(Some(amplitude.store.clone()));
        }
        amplitude.set_subset_sharing(options.share_subsets);
        Ok(amplitude)
    }

    /// Turn sharing between the integrals on or off. Results are
    /// bit-identical either way.
    pub fn set_subset_sharing(&mut self, enabled: bool) {
        self.share = enabled;
        self.store.set_enabled(enabled);
        self.store.clear();
    }

    pub fn subset_sharing(&self) -> bool {
        self.share
    }

    /// Number of integrals.
    pub fn len(&self) -> usize {
        self.workspaces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workspaces.is_empty()
    }

    /// Integral `index`'s workspace (diagnostics, execution statistics).
    pub fn workspace(&self, index: usize) -> &TensorWorkspace<T> {
        &self.workspaces[index]
    }

    /// Integral `index`'s workspace, e.g. to change its settings.
    pub fn workspace_mut(&mut self, index: usize) -> &mut TensorWorkspace<T> {
        &mut self.workspaces[index]
    }

    /// Evaluate integral `k` on `batches[k]`, for every `k` (one batch per
    /// integral, in the order of the plans; usually the same phase-space
    /// points, each batch holding the integral's own propagators).
    pub fn reduce_batches(&mut self, batches: &[KinematicsBatch<T>]) -> Result<Vec<BatchResult<T>>, ReductionError> {
        if batches.len() != self.workspaces.len() {
            return Err(ReductionError::InvalidInput(format!(
                "{} batches for {} integrals",
                batches.len(),
                self.workspaces.len()
            )));
        }
        self.store.clear();
        let results = self
            .workspaces
            .iter_mut()
            .zip(batches)
            .map(|(workspace, batch)| workspace.reduce_batch(batch))
            .collect::<Result<Vec<_>, _>>();
        let (reused, computed) = self.store.counts();
        let (moments_reused, moments_computed) = self.store.moment_counts();
        let (dd_masters_reused, dd_masters_computed) = self.store.dd_master_counts();
        let (dd_subsets_reused, dd_subsets_computed) = self.store.dd_subset_counts();
        self.last = SubsetSharingStats {
            reused,
            computed,
            moments_reused,
            moments_computed,
            dd_masters_reused,
            dd_masters_computed,
            dd_subsets_reused,
            dd_subsets_computed,
        };
        self.store.clear();
        results
    }

    /// Sharing statistics of the last [`Self::reduce_batches`] call (zero
    /// with sharing off).
    pub fn last_sharing(&self) -> SubsetSharingStats {
        self.last
    }
}
