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
pub use plan::DirectCapabilities;

/// Width of each propagator's field in the packed IBP keys (`exec`).
pub(crate) fn exec_ibp_field_bits(n: usize) -> u32 {
    exec::ibp_field_bits(n)
}

/// Largest power those fields hold.
pub(crate) fn exec_ibp_field_max(n: usize) -> u64 {
    exec::ibp_field_max(exec::ibp_field_bits(n))
}
use exec::DirectExecutor;
use plan::DirectPlanData;

use crate::{
    BatchResult, InputLayout, InvariantInput, KinematicsBatch, ReductionError, ReductionSettings, TensorStore,
    basis::Monomial,
};

/// `batch` with every lane's propagators relabelled cyclically by `rotation`
/// (see [`DirectReductionPlan::rotated_batch`]): rows of `layout.width()`
/// ordinary numbers, the component count kept. Also the facade's
/// loop-rotation test and cyclic-order checks.
pub(crate) fn rotate_lanes<T: NativeFloat>(
    batch: &KinematicsBatch<T>,
    layout: &InputLayout,
    rotation: usize,
) -> KinematicsBatch<T> {
    let n = layout.masses.len();
    let width = layout.width();
    let from = |k: usize| (k + rotation) % n.max(1);
    let mut values = Vec::with_capacity(batch.values.len());
    for lane in 0..batch.rows {
        let row = &batch.values[lane * width..(lane + 1) * width];
        for k in 0..n {
            values.extend_from_slice(&row[4 * from(k)..4 * from(k) + 4]);
        }
        if let Some(supplied) = &layout.scalar_products {
            // The invariants s_ij, i < j, of the relabelled propagators.
            for i in 0..n {
                for j in i + 1..n {
                    let (a, b) = (from(i), from(j));
                    let (a, b) = (a.min(b), a.max(b));
                    values.push(row[supplied.start + InvariantInput::invariant_index(n, a, b)].clone());
                }
            }
        }
        for k in 0..n {
            values.push(row[layout.masses.start + from(k)].clone());
        }
        values.push(row[layout.scale].clone());
    }
    KinematicsBatch { rows: batch.rows, components: batch.components, values }
}

/// Largest a priori estimate (COLLIER's w_pv2 with the modified Cayley
/// determinant's cancellation, see [`DirectWorkspace::set_alternative_first`])
/// at which the direct engine accepts the alternative PV reduction alone
/// ([`DirectWorkspace::enable_alternative_pv`]): 2^-36; also the natural
/// `limit` of [`DirectWorkspace::set_alternative_first`].
pub const ALTERNATIVE_PV_LIMIT: f64 = 1.0 / (1u64 << 36) as f64;

/// Largest relative change of the last pass at which the direct engine's
/// small-Gram expansions accept their result (2^-30). As the `limit` of
/// [`DirectWorkspace::set_alternative_first`]: the alternative PV first
/// wherever its estimate is within what the expansions themselves accept.
pub const EXPANSION_TOLERANCE: f64 = 1.0 / (1u64 << 30) as f64;

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

    /// `(size, rank)` of the subsets whose Cartesian moments are gathered
    /// inside a larger integral (not as the plan's own top subset).
    pub(crate) fn nested_gathers(&self) -> Vec<(usize, u16)> {
        let top = self.data.settings.propagators;
        self.data
            .gathers
            .iter()
            .map(|gather| (self.data.subsets[gather.subset].members.len(), gather.moment_rank))
            .filter(|&(size, _)| size < top)
            .collect()
    }

    /// Whether `settings` lie inside the direct engine's implemented domain.
    pub fn supports(settings: &ReductionSettings) -> bool {
        DirectPlanData::supported(settings).is_ok()
    }

    /// The capability query: what the direct engine serves for the carrier
    /// of `prototype` (largest point count, rank, raised power and dimension
    /// shift; see [`DirectCapabilities`]).
    pub fn capabilities<T: NativeFloat>(prototype: &T) -> DirectCapabilities {
        DirectCapabilities::of_representation(prototype.get_precision())
    }

    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Ok(Self {
            data: Arc::new(DirectPlanData::new(settings)?),
        })
    }

    pub fn settings(&self) -> &ReductionSettings {
        &self.data.settings
    }

    /// `batch` with every lane's propagators relabelled cyclically by
    /// `rotation`: propagator `k` of the result is propagator
    /// `(k + rotation) mod N` of `batch` (its shift, its supplied invariants
    /// and its squared mass; the scale is kept). The loop momentum is only
    /// relabelled, so the integral and its Cartesian moments are unchanged,
    /// while the direct engine reduces it through other PV systems (another
    /// origin: the propagator first in the new order), as in the loop-rotation
    /// test of [`crate::TensorWorkspace`]. Propagators of unequal powers would
    /// move away from their powers: such plans are refused.
    pub fn rotated_batch<T: NativeFloat>(
        &self,
        batch: &KinematicsBatch<T>,
        rotation: usize,
    ) -> Result<KinematicsBatch<T>, ReductionError> {
        let settings = &self.data.settings;
        if settings.powers.iter().any(|&power| power != settings.powers[0]) {
            return Err(ReductionError::Unsupported(
                "a rotated copy needs propagators of equal powers (the powers are fixed by the plan)".into(),
            ));
        }
        if batch.components != 1 {
            return Err(ReductionError::Unsupported(
                "the direct engine evaluates ordinary numbers; use the generic workspace for Dual components".into(),
            ));
        }
        let layout = InputLayout::of(settings);
        let width = layout.width();
        if batch.values.len() != batch.rows * width {
            return Err(ReductionError::InvalidInput(format!("expected {width} values per row")));
        }
        Ok(rotate_lanes(batch, &layout, rotation))
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

    /// [`Self::reduce_batch`] as a check pass: the masters provider of an
    /// attached store is never consulted (see
    /// [`Self::reduce_batch_with_rotation`]).
    pub(crate) fn reduce_check_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.executor.reduce_check_batch(batch)
    }

    /// Reduce `batch` and its rotated copy in one call (U3): returns
    /// `(primary, copy)`, where `copy` holds the moments of
    /// [`DirectReductionPlan::rotated_batch`]`(batch, rotation)`, lane for
    /// lane in the order of `batch`. The copy's moments are the same
    /// integrals in the same loop-momentum frame, reduced through other PV
    /// systems, so a lane and its copy agree to the reduction's accuracy (a
    /// stability check without a second call or a re-expanded numerator).
    ///
    /// The copy is a check pass: it never consults a masters provider
    /// ([`crate::TensorStoreOptions::masters`]). Without one it is bit for bit
    /// what [`Self::reduce_batch`] returns for the rotated batch, and its
    /// subsets that keep their propagator order (those without a propagator
    /// moved to the end; `rotation = 1` keeps every subset without
    /// propagator 0) repeat the primary's work exactly: they are taken from
    /// the attached [`TensorStore`] or, without one, from a store private to
    /// this call. That private store holds the masters and, in a binary64
    /// workspace that injects boxes or triangles (partial rescue, small-Gram
    /// expansions, method choice), the sub-integrals, pentagons, rescues,
    /// expansions and moments too; a binary64 workspace on its gates alone
    /// (where a sub-integral or moment key costs about what it saves) and a
    /// double-double one share the masters only, and their copy computes its
    /// PV sub-integrals, pentagons and moments again. With a provider the
    /// copy is computed in full by TIRust's kernels at the store's rank
    /// floors, bit for bit [`Self::reduce_batch`] on the rotated batch with
    /// [`TensorStore::check_view`] attached instead of the store: a provider
    /// that gives several argument orders one value (a canonical-order
    /// cache) would hand the copy the primary's boxes and triangles bit for
    /// bit and the check would no longer see the masters' rounding. Per-lane
    /// diagnostics (`last_conditions`, `last_gram_lanes`,
    /// `last_estimate_lanes`) describe the primary; `last_profile` and
    /// `last_master_rows` cover both.
    pub fn reduce_batch_with_rotation(
        &mut self,
        batch: &KinematicsBatch<T>,
        rotation: usize,
    ) -> Result<(BatchResult<T>, BatchResult<T>), ReductionError> {
        let plan = DirectReductionPlan { data: self.executor.base_plan().clone() };
        let rotated = plan.rotated_batch(batch, rotation)?;
        let private = self.executor.shared_subsets().is_none().then(|| {
            let store = Arc::new(shared::SubsetStore::default());
            store.set_enabled(true);
            // Without injected boxes and triangles, a sub-integral or moment
            // key costs about what it saves; the masters are shared.
            store.set_masters_only(!self.executor.injects_subsets());
            for (size, rank) in plan.nested_gathers() {
                store.share_moments(size, rank);
            }
            store
        });
        if let Some(store) = &private {
            self.executor.set_shared_subsets(Some(store.clone()));
        }
        let outcome = (|| {
            let primary = self.executor.reduce_batch(batch)?;
            let first = self.executor.take_call();
            let copy = self.executor.reduce_check_batch(&rotated)?;
            self.executor.merge_call(first);
            Ok((primary, copy))
        })();
        if private.is_some() {
            self.executor.set_shared_subsets(None);
        }
        outcome
    }

    /// Share per-subset results with every workspace holding `store`, across
    /// calls, until the caller clears it (`None`: detach; see
    /// [`TensorStore`]). Binary64 workspaces share sub-integrals, pentagons,
    /// rescues, expansions, moments and scalar masters (and take the store's
    /// masters provider); double-double ones their scalar masters; MPFR ones
    /// nothing. Attaching registers this plan's nested moment gathers with
    /// the store. Results are bit-identical with and without the store
    /// unless [`TensorStore::align_ranks`] raised its rank floor (or its
    /// masters provider returns other values than TIRust's kernels). Check
    /// passes (the rotation copy) never consult the provider.
    pub fn set_tensor_store(&mut self, store: Option<&TensorStore>) {
        if let Some(store) = store {
            store.register(&DirectReductionPlan { data: self.executor.base_plan().clone() });
        }
        self.executor.set_shared_subsets(store.map(|store| store.shared().clone()));
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

    /// Every box and triangle (refused by its Gram gate or not) by the
    /// small-Gram expansions, else the alternative PV reduction; their own
    /// triangles gated at `aux_threshold`. Unchecked, like the expansion.
    pub fn enable_gram_expansion_everywhere(&mut self, boost: u16, aux_threshold: T) {
        self.executor.enable_gram_expansion_everywhere(boost, aux_threshold);
    }

    /// COLLIER-style a priori method choice: boxes and triangles whose
    /// Passarino-Veltman estimate 2^-52 w^r exceeds 2^-36 are computed by the
    /// small-Gram expansions or the alternative PV reduction (unchecked).
    /// The expansions' convergence test bounds their truncation (the last
    /// pass's relative change at most [`EXPANSION_TOLERANCE`]), not the
    /// rounding of the higher-rank triangles they read: a served lane can be
    /// off by more than that (up to 1.6e-6 on pyAmpliCol's binary64
    /// g g > g g g g pentagons), and no per-lane estimate is reported.
    pub fn enable_estimated_method_choice(&mut self, boost: u16, aux_threshold: T) {
        self.executor.enable_estimated_method_choice(boost, aux_threshold);
    }

    /// With a small-Gram method choice (`enable_gram_expansion`,
    /// `enable_gram_expansion_everywhere`, `enable_estimated_method_choice`;
    /// before or after enabling it): a box or triangle it serves whose a
    /// priori estimate of the alternative PV reduction,
    /// 2^-53 (max(1, w) max(1, v))^ceil((r + 1) / 2) with COLLIER's
    /// w = max_j |(Zadj f)_j| / |Delta| (w_pv2) and the cancellation v of
    /// the modified Cayley determinant det X = 2 m_0^2 Delta - f Zadj f (sum
    /// of the terms' sizes over |det X|; the reduction divides by det X), is
    /// at most `limit` gets that reduction first, from triangles of its own
    /// rank; the expansions follow only if it fails. `None` (the default):
    /// the expansions first, the alternative PV as their check and fallback.
    /// Natural limits: [`ALTERNATIVE_PV_LIMIT`] (the alternative-only
    /// reduction's) and [`EXPANSION_TOLERANCE`] (the expansions' own).
    /// Like the expansions, the choice is unchecked: callers verify the
    /// lanes. It applies to the workspace's own boxes and triangles (the
    /// expansions' auxiliary triangles keep their order).
    pub fn set_alternative_first(&mut self, limit: Option<f64>) {
        self.executor.set_alternative_first(limit);
    }

    /// Evaluate the a priori estimate of `enable_estimated_method_choice`
    /// per lane without acting on it (every box and triangle by PV); see
    /// `last_estimate_lanes`.
    pub fn enable_estimate_flags(&mut self) {
        self.executor.enable_estimate_flags();
    }

    /// Share per-subset results with other workspaces holding the same store
    /// (`None`: no sharing). Binary64 workspaces share sub-integrals, rescues,
    /// expansions and moments; double-double ones their scalar masters, and
    /// where the store allows it (an amplitude's store, see
    /// `shared::SubsetStore::set_dd_subsets`) their sub-integrals, Gram-free
    /// pentagons and moments too.
    pub(crate) fn set_shared_subsets(&mut self, store: Option<std::sync::Arc<shared::SubsetStore>>) {
        self.executor.set_shared_subsets(store);
    }

    /// Boxes and triangles refused by their Gram gate are computed by the
    /// alternative PV reduction through the modified Cayley matrix
    /// (Denner-Dittmaier sect. 3.2) only, where its a priori estimate (as
    /// for [`Self::set_alternative_first`], with det X's cancellation) is at
    /// most [`ALTERNATIVE_PV_LIMIT`]; lanes with a box or triangle above it
    /// are refused. Unchecked otherwise, like the expansion.
    pub fn enable_alternative_pv(&mut self) {
        self.executor.enable_alternative_pv();
    }

    pub fn set_stability_threshold(&mut self, threshold: T) {
        self.executor.set_stability_threshold(threshold);
    }

    /// Precision (bits) of the inputs this workspace evaluates; by default
    /// its working precision. A bubble is lightlike (closed-form coefficients)
    /// when |p^2| is at most 2^-(bits - 13) of its scale: 2^-40 for binary64
    /// inputs, also when they are evaluated at a higher precision (rescue
    /// tiers), and relative to the working precision for inputs given at it.
    pub fn set_input_precision(&mut self, bits: u32) {
        self.executor.set_input_precision(bits);
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

#[cfg(test)]
mod dd_sharing_tests {
    use std::sync::Arc;

    use symbolica::domains::float::{Complex, DoubleFloat};

    use super::{DirectReductionPlan, shared::SubsetStore};
    use crate::{KinematicsBatch, ReductionSettings, TensorStore};

    /// Two generic massive pentagon lanes (well conditioned in every
    /// subset); the first three propagators are common to both.
    fn pentagon_lanes() -> KinematicsBatch<DoubleFloat> {
        const SHIFTS: [[[f64; 4]; 5]; 2] = [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.7, 0.2, -0.1, 0.3],
                [1.3, -0.4, 0.25, 0.1],
                [0.9, 0.35, 0.6, -0.45],
                [0.45, -0.3, -0.55, 0.2],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.7, 0.2, -0.1, 0.3],
                [1.3, -0.4, 0.25, 0.1],
                [1.1, -0.25, 0.4, 0.6],
                [0.3, 0.5, -0.35, -0.15],
            ],
        ];
        let dd = |x: f64| Complex::new(DoubleFloat::from(x), DoubleFloat::from(0.0));
        let mut values = Vec::new();
        for lane in &SHIFTS {
            for shift in lane {
                values.extend(shift.iter().map(|&x| dd(x)));
            }
            values.extend((0..5).map(|_| dd(0.6)));
            values.push(dd(1.0));
        }
        KinematicsBatch { rows: SHIFTS.len(), components: 1, values }
    }

    /// A double-double workspace at the engine's default gate.
    fn workspace(plan: &DirectReductionPlan) -> super::DirectWorkspace<DoubleFloat> {
        plan.workspace(DoubleFloat::from(0.0)).unwrap()
    }

    /// Every lane's array, limb for limb.
    fn limbs(result: &crate::BatchResult<DoubleFloat>) -> Vec<Vec<[u64; 4]>> {
        (0..result.rows())
            .map(|lane| {
                result.lane(lane).map_or_else(|_| Vec::new(), |values| {
                    values
                        .iter()
                        .map(|z| {
                            let (re, im) = (z.re.into_inner(), z.im.into_inner());
                            [re.hi().to_bits(), re.lo().to_bits(), im.hi().to_bits(), im.lo().to_bits()]
                        })
                        .collect()
                })
            })
            .collect()
    }

    /// A store as an amplitude's: sharing on, double-double sub-integrals,
    /// pentagons and moments too, and the moments of the gathers `shapes`
    /// (`(size, rank)`).
    fn amplitude_store(shapes: &[(usize, u16)]) -> Arc<SubsetStore> {
        let store = Arc::new(SubsetStore::default());
        store.set_enabled(true);
        store.set_dd_subsets(true);
        for &(size, rank) in shapes {
            store.share_moments(size, rank);
        }
        store
    }

    #[test]
    fn double_double_subsets_from_a_store_are_limb_for_limb_what_the_executor_computes() {
        let plan = DirectReductionPlan::new(ReductionSettings::new(5, 5)).unwrap();
        let batch = pentagon_lanes();
        let alone = workspace(&plan).reduce_batch(&batch).unwrap();
        assert!(limbs(&alone).iter().all(|lane| !lane.is_empty()), "a lane was refused");
        // The moments an amplitude shares (`AmplitudeWorkspace`,
        // `TensorStore::register`): those its integrals gather inside a
        // larger one, here a rank-six hexagon's pentagons (gathered at rank
        // five, as this plan gathers its own). Their keys hold the gather
        // scale and the coefficients, limb for limb.
        let hexagon = DirectReductionPlan::new(ReductionSettings::new(6, 6)).unwrap();
        let shapes = hexagon.nested_gathers();
        assert!(shapes.contains(&(5, 5)), "{shapes:?}");
        let mut lookups = Vec::new();
        for shapes in [&[][..], &shapes[..]] {
            let store = amplitude_store(shapes);
            let mut workspace = workspace(&plan);
            workspace.set_shared_subsets(Some(store.clone()));
            // The first call computes and stores (its two lanes share their
            // first three propagators, so some triangles meet already), the
            // second takes every stored sub-integral, pentagon and moment
            // from the store.
            let first = workspace.reduce_batch(&batch).unwrap();
            let (hits, misses) = store.dd_subset_counts();
            assert!(misses > 0, "{shapes:?}: ({hits}, {misses})");
            let second = workspace.reduce_batch(&batch).unwrap();
            let (hits_after, misses_after) = store.dd_subset_counts();
            assert_eq!(misses_after, misses, "{shapes:?}: the second call computed again");
            assert_eq!(hits_after - hits, hits + misses, "{shapes:?}: the second call missed a lookup");
            assert_eq!(limbs(&first), limbs(&alone), "{shapes:?}");
            assert_eq!(limbs(&second), limbs(&alone), "{shapes:?}");
            lookups.push(hits + misses);
        }
        // The moments add lookups of their own, all of which the second call
        // reused.
        assert!(lookups[1] > lookups[0], "no double-double moment shared: {lookups:?}");
    }

    #[test]
    fn double_double_settings_differing_in_the_low_limb_share_nothing() {
        // The real settings in the sub-integral and pentagon keys are keyed
        // by both limbs (`exec::real_key_words`): two workspaces whose gate,
        // or whose Gram expansions' auxiliary gate, agrees in the high limb
        // and differs in the low one compute everything themselves.
        let plan = DirectReductionPlan::new(ReductionSettings::new(5, 5)).unwrap();
        let batch = pentagon_lanes();
        let gate = DoubleFloat::from(1e-12);
        let nudged = gate.clone() + DoubleFloat::from(2f64.powi(-100));
        assert_eq!(gate.into_inner().hi(), nudged.into_inner().hi());
        assert_ne!(gate.into_inner().lo(), nudged.into_inner().lo());
        let gated = |threshold: &DoubleFloat| {
            let mut workspace = workspace(&plan);
            workspace.set_stability_threshold(threshold.clone());
            workspace
        };
        // The a priori method choice keeps these well-conditioned boxes and
        // triangles on PV: the auxiliary gate enters through the Gram
        // signature of the key only.
        let auxiliary = |threshold: &DoubleFloat| {
            let mut workspace = workspace(&plan);
            workspace.set_stability_threshold(gate.clone());
            workspace.enable_estimated_method_choice(12, threshold.clone());
            workspace
        };
        type Make<'a> = &'a dyn Fn(&DoubleFloat) -> super::DirectWorkspace<DoubleFloat>;
        for (setting, make) in [("gate", &gated as Make), ("auxiliary gate", &auxiliary as Make)] {
            // One call of a workspace with each setting in turn, on one store
            // (no moments: their keys hold the coefficients, not the
            // settings): `(hits, misses)` of each call.
            let in_turn = |a: &DoubleFloat, b: &DoubleFloat| {
                let store = amplitude_store(&[]);
                [a, b].map(|threshold| {
                    let (hits, misses) = store.dd_subset_counts();
                    let mut workspace = make(threshold);
                    workspace.set_shared_subsets(Some(store.clone()));
                    workspace.reduce_batch(&batch).unwrap();
                    let (hits_after, misses_after) = store.dd_subset_counts();
                    (hits_after - hits, misses_after - misses)
                })
            };
            // Equal settings: the second workspace reuses every entry.
            let [first, second] = in_turn(&gate, &gate);
            assert!(first.1 > 0, "{setting}: nothing computed {first:?}");
            assert_eq!(second, (first.0 + first.1, 0), "{setting}: equal settings");
            // Low limbs differ: neither workspace reuses the other's entries
            // (the second meets only its own, as the first did).
            for (a, b) in [(&gate, &nudged), (&nudged, &gate)] {
                let [first, second] = in_turn(a, b);
                assert_eq!(second, first, "{setting}: a low-limb change reused an entry");
            }
        }
    }

    #[test]
    fn a_tensor_store_shares_only_double_double_masters() {
        // Outside an amplitude workspace double-double executors keep their
        // sub-integrals, pentagons and moments to themselves.
        let plan = DirectReductionPlan::new(ReductionSettings::new(5, 5)).unwrap();
        let store = TensorStore::new();
        let mut workspace = workspace(&plan);
        workspace.set_tensor_store(Some(&store));
        workspace.reduce_batch(&pentagon_lanes()).unwrap();
        workspace.reduce_batch(&pentagon_lanes()).unwrap();
        assert_eq!(store.shared().dd_subset_counts(), (0, 0));
        assert!(store.statistics().dd_masters_reused > 0, "{:?}", store.statistics());
    }
}
