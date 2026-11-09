//! A persistent tensor store: per-subset results of the direct engine shared
//! between workspaces and across calls, with a lifetime the caller owns.
//!
//! [`crate::AmplitudeWorkspace`] shares the common boxes, triangles and
//! pentagons of an amplitude's integrals inside one call and clears its
//! store at the start and end of that call. An embedder that reduces the
//! integrals of a phase-space point in several calls (one per propagator
//! count and rank, say: a parent family and its pinched families in
//! different batches) needs the results to survive between those calls. A
//! [`TensorStore`] is that store with the lifetime left to the caller:
//! attach it to any number of [`crate::DirectWorkspace`]s (or
//! [`crate::TensorWorkspace`]s on the direct engine), reduce, and call
//! [`TensorStore::clear`] when the points change (once per point chunk).
//!
//! **What is shared, and when it hits.** Every entry is a pure function of
//! its exact key, so values are bit-identical with the store attached or
//! not, whatever the order of the calls and the number of threads sharing
//! it (the store is `Send + Sync`; concurrent workspaces may compute the same
//! entry twice, never a different one):
//!
//! - binary64 scalar masters (tadpole, bubble, triangle, box), keyed by the
//!   family and the bits of their arguments: the invariants (supplied or
//!   formed from the shifts), the squared masses and mu^2. The key holds no
//!   shift, so it is translation-invariant: with supplied invariants
//!   ([`crate::InvariantInput::SuppliedInvariants`]) a box meets its copy in
//!   any loop-momentum frame, whenever its invariants are supplied bit for
//!   bit and its propagators come in the same order;
//! - binary64 PV sub-integrals and Gram-free pentagons, partial rescues and
//!   small-Gram expansions of boxes and triangles, keyed by what was
//!   computed (the method's settings, the subset's plan signature, rank and
//!   value count) and the subset's propagators bit for bit (shifts, masses,
//!   supplied invariants) with mu^2;
//! - gathered binary64 Cartesian moments of the subsets of a size and rank
//!   that some attached plan gathers inside a larger integral (a hexagon's
//!   pentagons, for instance), keyed by everything the gather reads,
//!   including the subset's origin;
//! - double-double scalar masters (double-double workspaces, partial rescues
//!   and tiers), keyed limb for limb.
//!
//! The sub-integral and moment keys hold the subset's shifts, so parent and
//! pinched families meet when the caller hands them over in one **aligned**
//! loop-momentum frame: a propagator shared by two families has the same
//! shift bit for bit in both, and the families list their common propagators
//! in the same relative order (a subset's first propagator is its PV
//! origin). Keys on differences of shifts would not do better in binary64:
//! `(p_i + a) - (p_j + a)` rounds differently from `p_i - p_j`, so two frames
//! related by a translation rarely give equal difference bits, and the
//! Cartesian moments themselves depend on the origin. In other frames
//! nothing false is ever shared: their keys simply differ.
//!
//! **Use.** [`TensorStore::register`] every plan that will be reduced with
//! the store (every (N, R) bucket) before the first call, then per chunk of
//! phase-space points: reduce, and [`TensorStore::clear`] when the points
//! change (registrations and rank floors survive it). Moments are shared
//! for the subset sizes and ranks some registered or attached plan gathers
//! inside a larger integral; without registering first, which moments meet
//! depends on the order in which workspaces attach (a hexagon's pentagon
//! moments are reused by a later pentagon call, not by an earlier one).
//! Values never depend on it.
//!
//! **Lifetime and memory.** Nothing is removed until [`TensorStore::clear`].
//! [`TensorStoreOptions::byte_budget`] (none by default: the caller's clears
//! bound the store) stops inserting entries once their estimated size
//! reaches it; a result not inserted is computed again where it is needed,
//! never changed.
//!
//! **Rank alignment (optional).** Integrals need a box or triangle at
//! different coefficient ranks; sub-integral keys include the rank, so by
//! default only equal ranks meet and every workspace computes exactly what
//! it computes alone. [`TensorStore::align_ranks`] raises the store's rank
//! floor for the injected (partially rescued or expanded) boxes and
//! triangles to a plan's ranks, as an `AmplitudeWorkspace` does for its
//! integrals: more of them meet, and every attached workspace's injected
//! boxes and triangles are then computed at the floor (results differ from a
//! workspace without the store at the expansions' truncation level, and stay
//! bit-identical with sharing on and off).
//!
//! **Masters provider (optional).** A [`MasterProvider`] given in
//! [`TensorStoreOptions::masters`] supplies the binary64 scalar masters of
//! every attached binary64 workspace (and of their binary64 auxiliary
//! executors) on a miss of the store's own master maps, before TIRust's
//! kernels: an embedder can serve them from its own cache. Its values are
//! used as returned (results change only if they differ from TIRust's
//! kernels); declining a master (`None`) leaves it to TIRust's kernels.
//! The mass derivatives of dotted propagators (IBP jets), workspaces with
//! batched [`crate::MasterExecutionPolicy`] evaluators and double-double or
//! MPFR workspaces never consult it; nothing signals that beyond
//! [`TensorStoreStatistics::masters_provided`] (an embedder crediting its
//! cache checks that count).
//!
//! **Check passes never see the provider.** A check compares the same
//! integrals in another propagator order or frame with a primary pass. A
//! provider that gives several argument orders one value (a cache in a
//! canonical propagator order: a cyclic rotation is one of a box's dihedral
//! relabellings) would hand the check the primary's boxes and triangles bit
//! for bit, and the check would no longer see their rounding. TIRust's own
//! check passes (the rotation copy of
//! [`crate::DirectWorkspace::reduce_batch_with_rotation`], the facade's
//! loop-rotation test and cyclic-order checks) therefore reduce through
//! [`TensorStore::check_view`]: TIRust's kernels at the store's rank floors,
//! no entry of the store read (any binary64 entry may hold the provider's
//! masters). An embedder's own check passes (other frames, rescaled
//! re-tests) attach [`TensorStore::check_view`] to the workspaces they use,
//! or a store without a provider.
use std::sync::Arc;

use symbolica::domains::float::Complex;

use crate::{DirectReductionPlan, direct::shared::SubsetStore};

/// A scalar master family of the direct engine's native kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MasterFamily {
    /// `A0(m^2)`; arguments `[m^2, mu^2]`.
    Tadpole,
    /// `B0(p^2; m_0^2, m_1^2)`; arguments `[p^2, m_0^2, m_1^2, mu^2]`.
    Bubble,
    /// `C0`; arguments `[s_01, s_12, s_02, m_0^2, m_1^2, m_2^2, mu^2]`.
    Triangle,
    /// `D0`; arguments `[s_01, s_12, s_23, s_03, s_02, s_13, m_0^2, m_1^2,
    /// m_2^2, m_3^2, mu^2]`.
    Box,
}

impl MasterFamily {
    /// Number of arguments, mu^2 included.
    pub fn arguments(self) -> usize {
        match self {
            Self::Tadpole => 2,
            Self::Bubble => 4,
            Self::Triangle => 7,
            Self::Box => 11,
        }
    }

    pub(crate) fn of_index(index: usize) -> Self {
        match index {
            0 => Self::Tadpole,
            1 => Self::Bubble,
            2 => Self::Triangle,
            _ => Self::Box,
        }
    }
}

/// Binary64 scalar masters supplied by the embedder (see the module docs).
///
/// `arguments` are those of oneloopmaster's native kernels, in the
/// subset's propagator order (see [`MasterFamily`]): `s_ij` is the invariant
/// `(p_i - p_j)^2` of propagators `i` and `j` of the subset (supplied, or
/// formed from the shifts), then the squared masses, then mu^2. The value is
/// `[finite, single pole, double pole]` in the OneLOop normalisation of
/// [`crate::Convention::OneLoop`] (the single pole is the total, UV plus IR),
/// exactly what the kernel returns for these arguments. Implementations are
/// called from every thread that reduces with the store and must give the
/// same value for the same arguments. Check passes never call it (see the
/// module docs): a provider may serve masters in a canonical argument order.
pub trait MasterProvider: Send + Sync {
    /// The master at `arguments`, or `None` to leave it to TIRust's kernel.
    fn master(&self, family: MasterFamily, arguments: &[Complex<f64>]) -> Option<[Complex<f64>; 3]>;
}

/// Options of a [`TensorStore`]. Write them as
/// `TensorStoreOptions { byte_budget: Some(b), ..Default::default() }`:
/// options may be added (as for [`crate::EmbedderOptions`]).
#[derive(Clone, Default)]
pub struct TensorStoreOptions {
    /// Stop inserting entries once their estimated size reaches this many
    /// bytes (`None`, the default: no bound; the caller's
    /// [`TensorStore::clear`] bounds the store). Results are unchanged
    /// either way.
    pub byte_budget: Option<u64>,
    /// Binary64 scalar masters from the embedder, consulted on a miss of the
    /// store's master maps (`None`, the default: TIRust's kernels).
    pub masters: Option<Arc<dyn MasterProvider>>,
}

impl std::fmt::Debug for TensorStoreOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorStoreOptions")
            .field("byte_budget", &self.byte_budget)
            .field("masters", &self.masters.as_ref().map(|_| "MasterProvider"))
            .finish()
    }
}

/// Counts of a [`TensorStore`] since its last [`TensorStore::clear`] (counts
/// may be added).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TensorStoreStatistics {
    /// Binary64 sub-integrals, pentagons, rescues and expansions reused and
    /// computed (scalar masters not included).
    pub subsets_reused: u64,
    pub subsets_computed: u64,
    /// Binary64 scalar masters reused, computed (by TIRust's kernels or the
    /// provider) and, of those computed, served by the provider.
    pub masters_reused: u64,
    pub masters_computed: u64,
    pub masters_provided: u64,
    /// Gathered Cartesian moments reused and computed.
    pub moments_reused: u64,
    pub moments_computed: u64,
    /// Double-double scalar masters reused and computed.
    pub dd_masters_reused: u64,
    pub dd_masters_computed: u64,
    /// Stored entries and their estimated size in bytes. Since 3bb83e5 a
    /// binary64 sub-integral, pentagon or moment entry counts 8 bytes more
    /// (one more key word: the low limb of a real setting, zero in binary64),
    /// so a given byte budget admits slightly fewer such entries.
    pub entries: u64,
    pub bytes: u64,
    /// Results computed but not inserted because of the byte budget.
    pub not_stored: u64,
}

/// Per-subset results shared between workspaces and across calls (see the
/// module docs). Cloning gives another handle to the same store.
#[derive(Clone, Debug)]
pub struct TensorStore {
    inner: Arc<SubsetStore>,
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorStore {
    /// An empty store: sharing on, no byte budget, TIRust's own masters.
    pub fn new() -> Self {
        Self::with_options(TensorStoreOptions::default())
    }

    pub fn with_options(options: TensorStoreOptions) -> Self {
        let inner = SubsetStore::with_provider(options.masters);
        inner.set_enabled(true);
        inner.set_byte_budget(options.byte_budget);
        Self { inner: Arc::new(inner) }
    }

    /// Forget every entry and reset the counts (the caller's lifetime: once
    /// per chunk of phase-space points). Attached workspaces stay attached;
    /// registrations ([`Self::register`]) and rank floors
    /// ([`Self::align_ranks`]) stay.
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Share (the default) or not. Results are bit-identical either way; off,
    /// nothing is looked up or inserted (the rank floors and the masters
    /// provider still apply).
    pub fn set_sharing(&self, enabled: bool) {
        self.inner.set_enabled(enabled);
    }

    pub fn sharing(&self) -> bool {
        self.inner.enabled()
    }

    /// See [`TensorStoreOptions::byte_budget`]; entries already stored stay.
    pub fn set_byte_budget(&self, bytes: Option<u64>) {
        self.inner.set_byte_budget(bytes);
    }

    pub fn byte_budget(&self) -> Option<u64> {
        self.inner.byte_budget()
    }

    /// Share the moments of the subsets `plan` gathers inside its top
    /// integral (attaching a workspace does this for its own plan). Call it
    /// for every plan before the first call (registrations survive
    /// [`Self::clear`]): otherwise which moments meet depends on the order
    /// of the calls (see the module docs). Values do not depend on it.
    pub fn register(&self, plan: &DirectReductionPlan) {
        for (size, rank) in plan.nested_gathers() {
            self.inner.share_moments(size, rank);
        }
    }

    /// Raise the rank floor of the injected boxes and triangles to `plan`'s
    /// ranks (see the module docs: this changes the values of attached
    /// workspaces' injected subsets at the truncation level). Call it for
    /// every plan before the first call; floors only rise.
    pub fn align_ranks(&self, plan: &DirectReductionPlan) {
        for (size, rank) in plan.subset_ranks() {
            self.inner.raise_floor(size, rank);
        }
    }

    /// The store for check passes (the same integrals in another propagator
    /// order or frame, compared with a primary pass; see the module docs):
    /// without a masters provider this store itself (its entries are
    /// TIRust's own results for their exact keys, what a check computes);
    /// with one, a store with this store's rank floors as they are now, no
    /// provider and sharing off, so that the check computes every master
    /// with TIRust's kernels.
    pub fn check_view(&self) -> TensorStore {
        match self.inner.check_view() {
            Some(view) => Self { inner: Arc::new(view) },
            None => self.clone(),
        }
    }

    /// Whether the store has a masters provider.
    pub fn has_master_provider(&self) -> bool {
        self.inner.provider().is_some()
    }

    pub fn statistics(&self) -> TensorStoreStatistics {
        let (hits, misses) = self.inner.counts();
        let (masters_reused, masters_computed, masters_provided) = self.inner.master_counts();
        let (moments_reused, moments_computed) = self.inner.moment_counts();
        let (dd_masters_reused, dd_masters_computed) = self.inner.dd_master_counts();
        let (bytes, not_stored) = self.inner.byte_counts();
        TensorStoreStatistics {
            subsets_reused: hits.saturating_sub(masters_reused),
            subsets_computed: misses.saturating_sub(masters_computed),
            masters_reused,
            masters_computed,
            masters_provided,
            moments_reused,
            moments_computed,
            dd_masters_reused,
            dd_masters_computed,
            entries: self.inner.entries(),
            bytes,
            not_stored,
        }
    }

    pub(crate) fn shared(&self) -> &Arc<SubsetStore> {
        &self.inner
    }
}
