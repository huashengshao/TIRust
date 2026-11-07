//! Numerical execution of a direct plan.
//!
//! Lanes are processed in three stages. All master arguments of a batch are
//! collected first and each scalar family is evaluated once across lanes. The
//! recurrence program and the Cartesian gather then run per lane on flat,
//! reused buffers. A lane whose required Gram, Cayley or thin-six system is
//! numerically singular reports `Unstable` and exposes no coefficients.

use std::{collections::HashMap, sync::Arc, time::Instant};

mod gram;

use iregi_masters::{MasterEvaluator, MasterExecutionPolicy, NativeFloat, ScalarIntegral};
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use super::{
    numeric::{
        C, L4, cadd, cfma, cfma_real, cinv, cmul, cnorm2, cscale, csub, czero, div_dimension, lfma,
        lfma_slots, lzero, mul_dimension, real_lfma_slots, solve_in_place,
    },
    plan::{DirectPlanData, Gather, NO_GEOMETRY, PostOp, Subset, SubsetKind, Term},
    shared::{MASTER_KEY_WORDS, SharedValue, SubsetStore},
};
use crate::{
    types::filled,
    BatchResult, Convention, InputLayout, KinematicsBatch, LaneStatus, ReductionError,
    ReductionSettings, ScaleInput, TensorWeights,
    types::ResultMetadata,
};

/// Scalar-master implementation used by a direct workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DirectMasters {
    /// oneloopmaster's direct Rust kernels (hand-written binary64 C0/D0 sectors),
    /// evaluated per lane inside the lane loop.
    #[default]
    Native,
    /// `MasterEvaluator` with an explicit policy, batched across lanes.
    Policy(MasterExecutionPolicy),
}

/// Marks a monomial not yet reached by the product being built.
const NO_SLOT: u32 = u32::MAX;

/// A bubble is lightlike when |p^2| is at most `2^-(p - LIGHTLIKE_MARGIN_BITS)`
/// of its largest scale (|p^2|, |m0^2|, |m1^2|), `p` the precision of the
/// inputs (by default the working precision): 2^-40 for binary64 inputs, also
/// when they are evaluated at a higher precision (rescue tiers).
const LIGHTLIKE_MARGIN_BITS: i64 = 13;

/// Stage timings of the most recent profiled call.
#[derive(Clone, Copy, Debug, Default)]
pub struct DirectProfile {
    pub geometry: std::time::Duration,
    pub masters: std::time::Duration,
    pub recurrence: std::time::Duration,
    pub gather: std::time::Duration,
    pub output: std::time::Duration,
}

/// Why a lane's geometry cannot use the current plan variant.
enum GeometryFailure {
    /// Singular or ill-conditioned Gram system of this subset; it may be
    /// affinely dependent and reducible by a denominator identity.
    Singular(u16, String),
    /// A relation or lightlike assumption of the current variant does not
    /// hold on this lane.
    Relation(String),
    /// A bubble with (numerically) lightlike or vanishing momentum.
    Lightlike(u16),
    Unstable(String),
}

/// Partial rescue of a binary64 executor: boxes and triangles refused by
/// their Gram gate are evaluated by double-double executors of their own
/// plans (same members, origin and coefficient rank), and the rounded
/// coefficients replace the binary64 recursion; everything else stays
/// binary64.
#[derive(Clone)]
pub(crate) struct PartialRescue {
    executors: HashMap<(usize, u16), DirectExecutor<symbolica::domains::float::DoubleFloat>>,
    threshold: f64,
}

/// Most step-1 schedules an executor keeps (one per zero pattern of the
/// momenta; real kinematics rarely have more than a few).
const MAX_PRODUCT_SCHEDULES: usize = 8;

/// The fewest distinct momenta for which the gather keeps real products
/// (see [`DirectExecutor::gather_with`]).
const REAL_PRODUCT_MOMENTA: usize = 2;

/// A recorded step-1 pass of [`DirectExecutor::real_products`]: where every
/// term of the sparse product lists goes, for one plan and zero pattern.
#[derive(Clone, Debug)]
struct ProductSchedule {
    plan: Arc<DirectPlanData>,
    /// Per distinct momentum: its nonzero components (bits 0-3); then, when
    /// products are skipped, `u8::MAX` and whether each product is needed.
    key: Vec<u8>,
    /// Local monomial of every entry before exact zeros are dropped, in order.
    locals: Vec<u32>,
    /// Whether each of those entries is kept (not an exact zero).
    keep: Vec<bool>,
    /// Per product: its entries before and after dropping exact zeros.
    counts: Vec<(u32, u32)>,
    /// Entry (from the product's first) each (parent entry, axis) term
    /// accumulates into, in order.
    slots: Vec<u32>,
}

#[derive(Clone)]
pub(crate) struct DirectExecutor<T: NativeFloat> {
    /// Partial rescue (binary64 only) and the subsets it serves on this lane.
    partial: Option<Box<PartialRescue>>,
    external: Vec<usize>,
    /// Small-Gram expansion of refused boxes (`gram.rs`) and the boxes it
    /// serves on this lane.
    gram: Option<Box<gram::GramRescue<T>>>,
    gram_external: Vec<usize>,
    /// Auxiliary executor of the expansion: its top box is filled from its
    /// triangles up to this rank instead of its Gram system, and lanes
    /// return no moments.
    gram_top: Option<u16>,
    /// Auxiliary executor of the alternative-PV stage: skip the expansions.
    gram_alt_only: bool,
    /// Auxiliary executor of a nested triangle whose expansion cannot
    /// converge: the alternative PV reduction without its a priori estimate
    /// (as after a failed expansion).
    gram_alt_direct: bool,
    /// Alternative PV first for refused boxes and triangles whose a priori
    /// estimate is at most this (with a small-Gram method choice).
    gram_alternative_first: Option<f64>,
    /// Plan-derived tables of `expand_top`, per plan variant (built once).
    gram_tables: Vec<(Arc<DirectPlanData>, Arc<gram::TopTables>)>,
    /// Reused coefficient storage of `expand_top`.
    gram_pool: gram::CoefficientPool<T>,
    /// Auxiliary expansion executor: ranks the next larger boost adds (0 at
    /// the largest). When neither expansion converges and their observed
    /// convergence predicts that those ranks suffice, the executor reports
    /// failure so that the next boost is tried; otherwise it falls back to
    /// the alternative PV reduction.
    gram_extra_ranks: u16,
    /// Every box and triangle by the expansions or the alternative PV
    /// reduction, whatever its Gram conditioning (COLLIER chooses per box).
    gram_everywhere: bool,
    /// A priori choice: a box or triangle leaves PV when its estimate
    /// 2^-52 w^r exceeds 2^-36; refusals as before.
    gram_estimated: bool,
    /// Evaluate the same a priori estimate without acting on it: lanes where
    /// some box or triangle exceeds it are only recorded (`lane_estimate`).
    estimate_flags: bool,
    /// Per-subset results shared with other executors (binary64 only; see
    /// `super::shared`).
    shared: Option<Arc<SubsetStore>>,
    /// The same store for a double-double executor: its scalar masters only.
    dd_masters: Option<Arc<SubsetStore>>,
    /// The current lane's geometry found such a box or triangle.
    estimate_tripped: bool,
    gram_top_active: bool,
    /// mu^2 of the current lane (keys of shared sub-integrals).
    lane_mu2: Option<C<T>>,
    /// Plan variant of the current lane.
    plan: Arc<DirectPlanData>,
    base_plan: Arc<DirectPlanData>,
    /// Variants keyed by their minimal degenerate subsets.
    variants: HashMap<(Vec<u16>, Vec<u16>), (Arc<DirectPlanData>, Arc<Vec<T>>)>,
    lightlike_profile: Vec<u16>,
    base_constants: Arc<Vec<T>>,
    relation_profile: Vec<u16>,
    prototype: T,
    zero: C<T>,
    one: C<T>,
    constants: Arc<Vec<T>>,
    evaluators: [Option<MasterEvaluator<T>>; 4],
    native: [Option<oneloop::NativeEvaluator<T>>; 4],
    native_input: Vec<C<T>>,
    native_output: Vec<C<T>>,
    /// Key of the current double-double master in the shared store: family,
    /// arity and both limbs of every argument (reused buffer). Binary64
    /// masters use a fixed-size key of `MASTER_KEY_WORDS` words instead.
    master_key: Vec<u64>,
    master_inputs: [Vec<C<T>>; 4],
    /// Per lane, per family, per master: four Laurent slots.
    master_values: Vec<L4<T>>,
    master_errors: Vec<Option<String>>,
    master_offsets: [usize; 4],
    master_count: usize,
    values: Vec<L4<T>>,
    geometry: Vec<C<T>>,
    distances: Vec<C<T>>,
    /// Row layout of the plan's settings (shifts, supplied invariants,
    /// masses, scale).
    layout: InputLayout,
    /// The scale argument is mu^2 (`ScaleInput::MuSquared`), not mu.
    scale_squared: bool,
    /// The current lane's scale argument as given (mu or mu^2).
    lane_scale: C<T>,
    /// Precision (bits) of the inputs: the lightlike bubble rule is relative
    /// to it. The working precision unless set (`set_input_precision`).
    input_bits: u32,
    shifts: Vec<[C<T>; 4]>,
    masses: Vec<C<T>>,
    /// `g_lambda` blocks, `lambda * stride + monomial`.
    g_buffer: Vec<L4<T>>,
    sparse_products: Vec<(u32, C<T>)>,
    /// `sparse_products` of real momenta (binary64, see `gather_with`).
    sparse_real: Vec<(u32, T)>,
    /// Recorded step-1 passes of `real_products` by zero pattern.
    product_schedules: Vec<ProductSchedule>,
    /// Distinct momenta (geometry offsets) of the products of a plan, and
    /// each product's index among them (`u16::MAX` for degree zero).
    product_momenta: Option<(Arc<DirectPlanData>, Vec<u32>, Vec<u16>)>,
    /// Zero pattern of the current pass and its recorded entries.
    schedule_key: Vec<u8>,
    schedule_slots: Vec<u32>,
    product_ranges: Vec<(u32, u32)>,
    /// Per local monomial of the product being built: its entry in
    /// `sparse_products`, or `NO_SLOT`.
    product_slot: Vec<u32>,
    /// Products some gather of this lane computes (moments sharing on).
    product_needed: Vec<bool>,
    qexp_constants: Vec<Vec<T>>,
    binomials: Vec<Vec<T>>,
    tpow: Vec<C<T>>,
    matrix: Vec<C<T>>,
    rhs: Vec<C<T>>,
    native_constants: NativeConstants<T>,
    output_flags: Vec<OutputFlags<T>>,
    /// Taylor orders `1..K` of values and geometry in the dotted squared mass.
    values_jet: Vec<Vec<L4<T>>>,
    geometry_jet: Vec<Vec<C<T>>>,
    /// Per-lane IBP memo of raised-power scalars, keyed by packed powers.
    ibp_memo: HashMap<u64, L4<T>>,
    /// Per-lane inverse modified Cayley matrices, keyed by subset mask.
    rinv_memo: HashMap<u16, Vec<C<T>>>,
    /// `(-2)^h` of dimension-shifted gathers.
    gather_scale: T,
    /// The current lane's masters have no IR poles (two-slot kernels).
    ir_free: bool,
    /// Scalar-master evaluations of the most recent call.
    master_rows: usize,
    /// Minimal accepted relative determinant raised to its recursion depth.
    stability_threshold: T,
    /// Conditioning estimate of the most recent refused system (zero for an
    /// exactly singular pivot), for precision escalation by the caller.
    failed_condition: Option<f64>,
    /// Per lane of the most recent call: `failed_condition` of lanes refused
    /// on conditioning, `None` otherwise.
    lane_conditions: Vec<Option<f64>>,
    /// Per lane of the most recent call: some box or triangle was computed
    /// by the expansions or the alternative PV reduction.
    lane_gram: Vec<bool>,
    /// Per lane of the most recent call: some box or triangle exceeds the a
    /// priori estimate (`estimate_flags` only).
    lane_estimate: Vec<bool>,
    /// `8 * 2^-precision`: a squared distance below this multiple of its
    /// sum of squared components is zero within its own rounding.
    rounding: T,
    profiling: bool,
    profile: DirectProfile,
    /// Partial-rescue executor: only the top subset's coefficients are read,
    /// so lanes whose outputs are certainly finite stop after the
    /// coefficient program (see `top_outputs_finite`).
    coefficients_only: bool,
}

/// Per-output-moment conversion flags.
#[derive(Clone)]
struct OutputFlags<T> {
    /// UV power counting: no UV pole at this rank.
    uv_zero: bool,
    /// Odd rank (MadLoop sign).
    odd: bool,
    /// Multinomial weight, if requested.
    weight: Option<T>,
}

#[derive(Clone)]
struct NativeConstants<T> {
    a: T,
    a_half_squared_minus_b: T,
    normalization: C<T>,
}

impl<T: NativeFloat> DirectExecutor<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(
        plan: Arc<DirectPlanData>,
        prototype: T,
        masters: DirectMasters,
    ) -> Result<Self, ReductionError> {
        let zero = czero(&prototype);
        let one = C::new(prototype.one(), prototype.zero());
        let constants: Arc<Vec<T>> = Arc::new(
            plan.constants
                .iter()
                .map(|value| prototype.from_rational(value))
                .collect(),
        );
        let families = [
            ScalarIntegral::A0,
            ScalarIntegral::B0,
            ScalarIntegral::C0,
            ScalarIntegral::D0,
        ];
        let mut evaluators: [Option<MasterEvaluator<T>>; 4] = Default::default();
        let mut native: [Option<oneloop::NativeEvaluator<T>>; 4] = Default::default();
        for (family_index, family) in families.into_iter().enumerate() {
            if plan.masters[family_index].is_empty() {
                continue;
            }
            match masters {
                DirectMasters::Native => {
                    native[family_index] = Some(
                        oneloop::NativeEvaluator::with_binary_precision(
                            family,
                            prototype.get_precision(),
                        )
                        .map_err(ReductionError::Preparation)?,
                    );
                }
                DirectMasters::Policy(policy) => {
                    evaluators[family_index] = Some(
                        MasterEvaluator::new_with_policy(family, prototype.clone(), policy)
                            .map_err(|error| ReductionError::Preparation(error.to_string()))?,
                    );
                }
            }
        }
        let mut master_offsets = [0; 4];
        let mut master_count = 0;
        for family in 0..4 {
            master_offsets[family] = master_count;
            master_count += plan.masters[family].len();
        }
        let pi = prototype.pi();
        let a = prototype.euler() + &pi.log();
        let b = pi.clone() * &pi / &prototype.from_usize(12);
        let half = prototype.one() / &prototype.from_usize(2);
        let a_half_squared_minus_b = a.clone() * &a * &half - &b;
        let mut pi_power = prototype.one();
        for _ in 0..2 + i32::from(plan.settings.dimension_shift) {
            pi_power *= &pi;
        }
        let normalization = C::new(prototype.zero(), pi_power);
        let mut gather_scale = prototype.one();
        for _ in 0..plan.lambda_shift {
            gather_scale *= &prototype.from_i64(-2);
        }
        let jet_orders = plan.jet.map_or(0, |(_, k)| k - 1);
        let gather_degree = plan.basis.len();
        let max_size = plan.subsets.iter().map(|s| s.members.len()).max().unwrap_or(1);
        let stability_threshold = prototype.from_rational(&symbolica::domains::rational::Rational::from((
            symbolica::domains::integer::Integer::from(1),
            symbolica::domains::integer::Integer::from(1_000_000_000_000i64),
        )));
        Ok(Self {
            partial: None,
            external: Vec::new(),
            gram: None,
            gram_external: Vec::new(),
            gram_top: None,
            gram_alt_only: false,
            gram_alt_direct: false,
            gram_alternative_first: None,
            gram_tables: Vec::new(),
            gram_pool: gram::CoefficientPool::default(),
            gram_extra_ranks: 0,
            gram_everywhere: false,
            gram_estimated: false,
            estimate_flags: false,
            lane_mu2: None,
            shared: None,
            estimate_tripped: false,
            gram_top_active: false,
            values: vec![lzero(&zero); plan.value_count],
            geometry: vec![zero.clone(); plan.geometry_count.max(1)],
            distances: vec![zero.clone(); plan.settings.propagators * plan.settings.propagators],
            layout: InputLayout::of(&plan.settings),
            scale_squared: plan.settings.scale_input == ScaleInput::MuSquared,
            lane_scale: zero.clone(),
            input_bits: prototype.get_precision(),
            shifts: vec![[zero.clone(), zero.clone(), zero.clone(), zero.clone()]; plan.settings.propagators],
            masses: vec![zero.clone(); plan.settings.propagators],
            g_buffer: vec![lzero(&zero); gather_degree * (usize::from(plan.settings.rank) / 2 + 1)],
            sparse_products: Vec::new(),
            sparse_real: Vec::new(),
            product_schedules: Vec::new(),
            product_momenta: None,
            schedule_key: Vec::new(),
            schedule_slots: Vec::new(),
            product_ranges: Vec::new(),
            product_slot: vec![NO_SLOT; gather_degree],
            product_needed: Vec::new(),
            qexp_constants: plan
                .monomials
                .qexp
                .iter()
                .map(|terms| terms.iter().map(|(_, _, c)| prototype.from_rational(c)).collect())
                .collect(),
            // Coefficient ranks may exceed the tensor rank (boosted pinches).
            binomials: (0..=usize::from(plan.settings.rank)
                .max(plan.subsets.iter().map(|subset| subset.coefficient_rank.max(0) as usize).max().unwrap_or(0)))
                // Pascal's rule in the working arithmetic: exact, no overflow.
                .scan(Vec::<T>::new(), |row: &mut Vec<T>, _| {
                    let mut next = Vec::with_capacity(row.len() + 1);
                    next.push(prototype.one());
                    for k in 1..row.len() {
                        next.push(row[k - 1].clone() + &row[k]);
                    }
                    if !row.is_empty() {
                        next.push(prototype.one());
                    }
                    *row = next.clone();
                    Some(next)
                })
                .collect(),
            tpow: vec![zero.clone(); usize::from(plan.settings.rank) + 1],
            matrix: vec![zero.clone(); max_size.max(5) * max_size.max(5)],
            rhs: vec![zero.clone(); 5 * 5],
            values_jet: vec![vec![lzero(&zero); plan.value_count]; jet_orders],
            geometry_jet: vec![vec![zero.clone(); plan.geometry_count.max(1)]; jet_orders],
            ibp_memo: HashMap::new(),
            rinv_memo: HashMap::new(),
            gather_scale,
            ir_free: false,
            master_rows: 0,
            native_input: vec![zero.clone(); 11],
            native_output: vec![zero.clone(); 3],
            master_key: Vec::with_capacity(25),
            native,
            master_inputs: Default::default(),
            master_values: Vec::new(),
            master_errors: Vec::new(),
            master_offsets,
            master_count,
            output_flags: {
                let settings = &plan.settings;
                let base = 4 + 2 * i32::from(settings.dimension_shift)
                    - 2 * settings.powers.iter().map(|&p| i32::from(p)).sum::<i32>();
                plan.basis
                    .iter()
                    .map(|monomial| {
                        let rank = monomial.rank() as i32;
                        OutputFlags {
                            uv_zero: base + rank < 0,
                            odd: rank % 2 == 1,
                            weight: (settings.tensor_weights == TensorWeights::Multinomial)
                                .then(|| prototype.from_usize(multinomial(monomial.0) as usize)),
                        }
                    })
                    .collect()
            },
            native_constants: NativeConstants {
                a,
                a_half_squared_minus_b,
                normalization,
            },
            stability_threshold,
            failed_condition: None,
            lane_conditions: Vec::new(),
            lane_gram: Vec::new(),
            lane_estimate: Vec::new(),
            rounding: {
                let half = prototype.one() / &prototype.from_usize(2);
                (0..prototype.get_precision()).fold(prototype.from_usize(8), |acc, _| acc * &half)
            },
            profiling: false,
            profile: DirectProfile::default(),
            coefficients_only: false,
            dd_masters: None,
            base_plan: plan.clone(),
            base_constants: constants.clone(),
            variants: HashMap::new(),
            relation_profile: Vec::new(),
            lightlike_profile: Vec::new(),
            constants,
            evaluators,
            prototype,
            zero,
            one,
            plan,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    pub fn last_profile(&self) -> DirectProfile {
        self.profile
    }

    pub fn last_master_rows(&self) -> usize {
        self.master_rows
    }

    pub fn set_stability_threshold(&mut self, threshold: T) {
        self.stability_threshold = threshold;
    }

    pub fn set_input_precision(&mut self, bits: u32) {
        self.input_bits = bits;
    }

    pub fn last_conditions(&self) -> &[Option<f64>] {
        &self.lane_conditions
    }

    pub(crate) fn last_gram_lanes(&self) -> &[bool] {
        &self.lane_gram
    }

    pub(crate) fn last_estimate_lanes(&self) -> &[bool] {
        &self.lane_estimate
    }

    fn timer(&self) -> Option<Instant> {
        self.profiling.then(Instant::now)
    }

    fn read_lane(&mut self, batch: &KinematicsBatch<T>, row: usize) -> Result<(), String> {
        let n = self.plan.settings.propagators;
        let width = self.layout.width();
        let values = &batch.values[row * width..(row + 1) * width];
        for (i, shift) in self.shifts.iter_mut().enumerate() {
            for a in 0..4 {
                shift[a] = values[4 * i + a].clone();
            }
        }
        let masses = self.layout.masses.start;
        for i in 0..n {
            self.masses[i] = values[masses + i].clone();
        }
        if values.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
            return Err("nonfinite kinematic input".into());
        }
        let scale = &values[self.layout.scale];
        if !scale.im.is_zero() || scale.re <= scale.re.zero() {
            return Err(if self.scale_squared {
                "mu^2 must be positive and real".into()
            } else {
                "mu must be positive and real".into()
            });
        }
        self.lane_scale = scale.clone();
        if self.masses.iter().any(|mass| mass.im > mass.im.zero()) {
            return Err("squared masses require nonpositive imaginary parts".into());
        }
        if let Some(supplied) = self.layout.scalar_products.clone() {
            // Supplied invariants are authoritative: read exactly, never
            // recomputed or snapped.
            let mut index = supplied.start;
            for i in 0..n {
                self.distances[i * n + i] = self.zero.clone();
                for j in i + 1..n {
                    let value = values[index].clone();
                    index += 1;
                    self.distances[j * n + i] = value.clone();
                    self.distances[i * n + j] = value;
                }
            }
            return Ok(());
        }
        // Upper triangle only: d(j, i) equals d(i, j) bit for bit (the
        // difference changes sign, which neither its square nor the error
        // bound below sees), and d(i, i) is the zero the tolerance test
        // below returns for it.
        for i in 0..n {
            self.distances[i * n + i] = self.zero.clone();
            for j in i + 1..n {
                let mut value = self.zero.clone();
                let mut size = self.prototype.zero();
                for a in 0..4 {
                    let delta = csub(&self.shifts[i][a], &self.shifts[j][a]);
                    let square = cmul(&delta, &delta);
                    size += cnorm2(&square).sqrt();
                    // The difference of two represented shifts carries their
                    // representation error u (|q_ia| + |q_ja|), which enters
                    // the square as 2 |delta| u (|q_ia| + |q_ja|).
                    let spread = cnorm2(&self.shifts[i][a]).sqrt() + cnorm2(&self.shifts[j][a]).sqrt();
                    size += cnorm2(&delta).sqrt() * &spread * &self.prototype.from_usize(2);
                    value = if a == 0 {
                        cadd(&value, &square)
                    } else {
                        csub(&value, &square)
                    };
                }
                // A (numerically lightlike) invariant within the rounding error
                // of its represented inputs is zero: replacing it perturbs the
                // inputs by no more than their working-precision representation
                // does, and it keeps scalar masters on their exact massless-leg
                // branches (the general ones cancel like m^2/p^2 there, and
                // oneloopmaster's binary64 D0 returns NaN for two opposite legs
                // at ~1e-10 against m^2 = 3e4).
                if cnorm2(&value).sqrt() <= size * &self.rounding {
                    value = self.zero.clone();
                }
                self.distances[j * n + i] = value.clone();
                self.distances[i * n + j] = value;
            }
        }
        Ok(())
    }

    /// mu^2 of `row`: the supplied value, or the square of the supplied mu.
    fn mu_squared(&self, batch: &KinematicsBatch<T>, row: usize) -> C<T> {
        let scale = &batch.values[row * self.layout.width() + self.layout.scale];
        if self.scale_squared {
            scale.clone()
        } else {
            cmul(scale, scale)
        }
    }

    /// Settings of an auxiliary single-subset plan (partial rescue, small-Gram
    /// expansion): `size` unit-power propagators at `rank`, with this plan's
    /// input conventions (supplied invariants, scale).
    pub(super) fn subset_settings(&self, size: usize, rank: u16) -> ReductionSettings {
        let mut settings = ReductionSettings::new(size, rank);
        settings.invariants = self.base_plan.settings.invariants;
        settings.scale_input = self.base_plan.settings.scale_input;
        settings
    }

    /// The current lane's row of the auxiliary plan of `members` (see
    /// [`Self::subset_settings`]), its values mapped by `map`: their shifts,
    /// supplied invariants, masses and the scale as given.
    pub(super) fn subset_row<U: NativeFloat>(&self, members: &[usize], map: impl Fn(&C<T>) -> C<U>) -> Vec<C<U>> {
        let n = self.plan.settings.propagators;
        let size = members.len();
        let supplied = self.layout.scalar_products.is_some();
        let mut values = Vec::with_capacity(5 * size + 1 + if supplied { size * size / 2 } else { 0 });
        for &member in members {
            values.extend(self.shifts[member].iter().map(&map));
        }
        if supplied {
            for (a, &i) in members.iter().enumerate() {
                for &j in &members[a + 1..] {
                    values.push(map(&self.distances[i * n + j]));
                }
            }
        }
        values.extend(members.iter().map(|&member| map(&self.masses[member])));
        values.push(map(&self.lane_scale));
        values
    }

    #[inline(never)]
    fn evaluate_masters(&mut self, batch: &KinematicsBatch<T>) -> Result<(), ReductionError> {
        let n = self.plan.settings.propagators;
        let rows = batch.rows;
        for family in 0..4 {
            self.master_inputs[family].clear();
        }
        self.master_values.clear();
        self.master_values
            .resize(rows * self.master_count, lzero(&self.zero));
        self.master_errors.clear();
        self.master_errors.resize(rows, None);
        let plan = self.plan.clone();
        for row in 0..rows {
            if let Err(error) = self.read_lane(batch, row) {
                self.master_errors[row] = Some(error);
            }
            let mu2 = self.mu_squared(batch, row);
            for family in 0..4 {
                for &(subset, _) in &plan.masters[family] {
                    let members = &plan.subsets[subset].members;
                    let d = |i: usize, j: usize| self.distances[members[i] * n + members[j]].clone();
                    let input = &mut self.master_inputs[family];
                    match family {
                        0 => {}
                        1 => input.push(d(0, 1)),
                        2 => input.extend([d(0, 1), d(1, 2), d(0, 2)]),
                        _ => input.extend([d(0, 1), d(1, 2), d(2, 3), d(0, 3), d(0, 2), d(1, 3)]),
                    }
                    input.extend(members.iter().map(|&i| self.masses[i].clone()));
                    input.push(mu2.clone());
                }
            }
        }
        for family in 0..4 {
            let Some(evaluator) = self.evaluators[family].as_mut() else {
                continue;
            };
            let per_lane = plan.masters[family].len();
            let batch_rows = rows * per_lane;
            self.master_rows += batch_rows;
            let result = evaluator
                .evaluate_batch(&self.master_inputs[family], batch_rows)
                .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
            for row in 0..rows {
                for (k, &(subset, _)) in plan.masters[family].iter().enumerate() {
                    let index = row * self.master_count + self.master_offsets[family] + k;
                    match result.lane(row * per_lane + k) {
                        Ok(lane) => {
                            let members = &plan.subsets[subset].members;
                            let mut value = [
                                lane.uv[0].clone(),
                                lane.ir[0].clone(),
                                lane.ir2[0].clone(),
                                lane.finite[0].clone(),
                            ];
                            // Physical pole slots, as in the symbolic planner.
                            match family {
                                0 => {
                                    value[0] = self.master_inputs[0][(row * per_lane + k) * 2].clone();
                                    value[1] = self.zero.clone();
                                    value[2] = self.zero.clone();
                                }
                                1 => value[0] = self.one.clone(),
                                _ => value[0] = self.zero.clone(),
                            }
                            let _ = members;
                            self.master_values[index] = value;
                        }
                        Err(error) => {
                            if self.master_errors[row].is_none() {
                                self.master_errors[row] = Some(format!("{error}"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluate this lane's masters with the direct oneloopmaster kernels, writing
    /// the four native Laurent slots straight into the value array.
    fn native_masters(&mut self, plan: &DirectPlanData, mu2: &C<T>) -> Result<(), String> {
        let n = plan.settings.propagators;
        // Scalar masters shared with the other integrals of an amplitude,
        // keyed by family and arguments bit for bit (binary64).
        let store = self.shared.clone().filter(|store| store.enabled());
        for family in 0..4 {
            let Some(evaluator) = self.native[family].as_mut() else {
                continue;
            };
            for &(subset, index) in &plan.masters[family] {
                let members = &plan.subsets[subset].members;
                let d = |i: usize, j: usize| self.distances[members[i] * n + members[j]].clone();
                let input = &mut self.native_input;
                let arity = match family {
                    0 => {
                        input[0] = self.masses[members[0]].clone();
                        1
                    }
                    1 => {
                        input[0] = d(0, 1);
                        3
                    }
                    2 => {
                        input[0] = d(0, 1);
                        input[1] = d(1, 2);
                        input[2] = d(0, 2);
                        6
                    }
                    _ => {
                        input[0] = d(0, 1);
                        input[1] = d(1, 2);
                        input[2] = d(2, 3);
                        input[3] = d(0, 3);
                        input[4] = d(0, 2);
                        input[5] = d(1, 3);
                        10
                    }
                };
                let first_mass = arity - members.len();
                for (k, &member) in members.iter().enumerate() {
                    input[first_mass + k] = self.masses[member].clone();
                }
                input[arity] = mu2.clone();
                // The arguments bit for bit (the family fixes their number).
                let mut key = [0u64; MASTER_KEY_WORDS];
                let words = 2 * (arity + 1);
                let hit = store.as_ref().is_some_and(|store| {
                    for (k, z) in input[..arity + 1].iter().enumerate() {
                        key[2 * k] = z.re.to_f64().to_bits();
                        key[2 * k + 1] = z.im.to_f64().to_bits();
                    }
                    let Some(v) = store.master(family, &key[..words]) else { return false };
                    for k in 0..3 {
                        self.native_output[k] =
                            C::new(exact_f64(&self.prototype, v[2 * k]), exact_f64(&self.prototype, v[2 * k + 1]));
                    }
                    true
                });
                // Double-double: masters shared limb for limb (a stored value
                // is the kernels' output for exactly these arguments).
                let dd_store = self.dd_masters.as_ref().filter(|store| store.enabled());
                let dd_key = dd_store.is_some() && dd_master_key(&mut self.master_key, family, arity, &input[..arity + 1]);
                let dd_hit = dd_key
                    && dd_store.is_some_and(|store| {
                        let output = &mut self.native_output;
                        store.dd_master_with(&self.master_key, |stored| write_dd_master(output, stored))
                    });
                if !hit && !dd_hit {
                    evaluator.evaluate(&input[..arity + 1], &mut self.native_output)?;
                    if let (true, Some(store)) = (dd_key, dd_store) {
                        if let Some(value) = read_dd_master(&self.native_output) {
                            store.put_dd_master(self.master_key.clone(), value);
                        }
                    }
                    if let Some(store) = store.as_ref() {
                        let o = &self.native_output;
                        let v = [o[0].re.to_f64(), o[0].im.to_f64(), o[1].re.to_f64(), o[1].im.to_f64(),
                            o[2].re.to_f64(), o[2].im.to_f64()];
                        store.put_master(family, &key[..words], v);
                    }
                }
                self.master_rows += 1;
                // The oneloopmaster order is [finite, total single, double]. The UV
                // residues are exact: A0 m^2, B0 one, C0/D0 zero.
                let uv = match family {
                    0 => self.native_input[0].clone(),
                    1 => self.one.clone(),
                    _ => self.zero.clone(),
                };
                let [finite, single, double] = [
                    self.native_output[0].clone(),
                    self.native_output[1].clone(),
                    self.native_output[2].clone(),
                ];
                self.values[index as usize] = [uv.clone(), csub(&single, &uv), double, finite];
            }
        }
        Ok(())
    }

    /// Enable the partial rescue (binary64 executors only) with the given
    /// double-double acceptance threshold.
    pub(crate) fn enable_partial_rescue(&mut self, threshold: f64) {
        if T::FIXED_BITS == Some(53) {
            self.partial = Some(Box::new(PartialRescue { executors: HashMap::new(), threshold }));
        }
    }

    pub(crate) fn disable_partial_rescue(&mut self) {
        self.partial = None;
    }

    /// Boxes refused by their Gram gate are computed by the small-Gram
    /// expansion from triangles `boost` ranks higher (binary64 lanes keep
    /// their precision; the caller checks the result a posteriori).
    pub(crate) fn enable_gram_expansion(&mut self, boost: u16) {
        self.gram = Some(Box::new(gram::GramRescue::new(boost)));
        self.set_alternative_first(self.gram_alternative_first);
    }

    /// Every box and triangle (refused or not) by the small-Gram expansions,
    /// else the alternative PV reduction; their own triangles gated at
    /// `aux_threshold`.
    pub(crate) fn enable_gram_expansion_everywhere(&mut self, boost: u16, aux_threshold: T) {
        self.gram = Some(Box::new(gram::GramRescue::everywhere(boost, aux_threshold)));
        self.set_alternative_first(self.gram_alternative_first);
        self.gram_everywhere = true;
    }

    /// COLLIER-style a priori method choice in this executor's own pass.
    pub(crate) fn enable_estimated_method_choice(&mut self, boost: u16, aux_threshold: T) {
        self.gram = Some(Box::new(gram::GramRescue::everywhere(boost, aux_threshold)));
        self.set_alternative_first(self.gram_alternative_first);
        self.gram_estimated = true;
    }

    pub(crate) fn set_alternative_first(&mut self, limit: Option<f64>) {
        self.gram_alternative_first = limit;
        if let Some(gram) = self.gram.as_mut() {
            gram.set_alternative_first(limit);
        }
    }

    /// Record, per lane, whether the a priori choice would leave PV for
    /// some box or triangle, while computing every one by PV.
    pub(crate) fn enable_estimate_flags(&mut self) {
        self.estimate_flags = true;
    }

    /// Refused boxes and triangles by the alternative PV reduction alone.
    pub(crate) fn enable_alternative_pv(&mut self) {
        self.gram = Some(Box::new(gram::GramRescue::alternative()));
    }

    /// Coefficient values of the top subset of the current lane's variant,
    /// when it is a PV subset.
    pub(crate) fn top_coefficients(&self) -> Option<Vec<L4<T>>> {
        let id = self.plan.subsets.len().checked_sub(1)?;
        let subset = &self.plan.subsets[id];
        if subset.kind != SubsetKind::Pv || subset.lightlike {
            return None;
        }
        let start = subset.master? as usize;
        let count = self.plan.coefficient_counts[id] as usize;
        Some(self.values[start..start + count].to_vec())
    }

    /// Share per-subset results (partial rescue, expansions) through `store`
    /// (binary64 executors; `None`: compute every one here). A double-double
    /// executor shares only its scalar masters, keyed limb for limb.
    pub(crate) fn set_shared_subsets(&mut self, store: Option<Arc<SubsetStore>>) {
        self.dd_masters = store.clone().filter(|_| T::FIXED_BITS == Some(106));
        self.shared = store.filter(|_| T::FIXED_BITS == Some(53));
    }

    /// Exact key of a shared per-subset result: what it is (`tag`), the
    /// subset's size, rank and value count, the method's settings, and the
    /// subset's propagators bit for bit. `None` without a store.
    fn subset_key(&self, tag: u64, members: &[usize], rank: u16, count: usize, config: &[u64], mu: &C<T>) -> Option<Vec<u64>> {
        if !self.shared.as_ref()?.enabled() {
            return None;
        }
        let mut key = Vec::new();
        self.fill_subset_key(&mut key, tag, members, rank, count, config, mu);
        Some(key)
    }

    /// [`Self::subset_key`] into a reused buffer.
    #[allow(clippy::too_many_arguments)]
    fn fill_subset_key(&self, key: &mut Vec<u64>, tag: u64, members: &[usize], rank: u16, count: usize, config: &[u64], mu: &C<T>) {
        key.clear();
        key.extend([tag, members.len() as u64, u64::from(rank), count as u64, config.len() as u64]);
        key.extend_from_slice(config);
        let bits = |z: &C<T>| [z.re.to_f64().to_bits(), z.im.to_f64().to_bits()];
        for &member in members {
            for a in 0..4 {
                key.extend(bits(&self.shifts[member][a]));
            }
            key.extend(bits(&self.masses[member]));
        }
        if self.layout.scalar_products.is_some() {
            // Supplied invariants are inputs of their own.
            let n = self.plan.settings.propagators;
            for (a, &i) in members.iter().enumerate() {
                for &j in &members[a + 1..] {
                    key.extend(bits(&self.distances[i * n + j]));
                }
            }
        }
        key.extend(bits(mu));
    }

    fn shared_get(&self, key: Option<&Vec<u64>>) -> Option<SharedValue> {
        self.shared.as_ref()?.get(key?)
    }

    /// Coefficient rank at which an injected `size`-propagator subset needed
    /// at `rank` is computed (the amplitude's floor, see `super::shared`).
    fn injection_rank(&self, size: usize, rank: u16) -> u16 {
        self.shared.as_ref().map_or(rank, |store| store.rank_for(size, rank))
    }

    fn shared_put(&self, key: Option<Vec<u64>>, outcome: &Result<Vec<[f64; 8]>, String>) {
        if let (Some(store), Some(key)) = (self.shared.as_ref(), key) {
            store.put(key, outcome.clone());
        }
    }

    /// Binary64 values of a computed subset, for the store.
    fn shared_values<U: NativeFloat>(values: &[L4<U>]) -> Vec<[f64; 8]> {
        values
            .iter()
            .map(|slots| std::array::from_fn(|k| if k % 2 == 0 { slots[k / 2].re.to_f64() } else { slots[k / 2].im.to_f64() }))
            .collect()
    }

    /// Write the leading `count` shared binary64 values (exact in this
    /// executor's binary64 domain).
    fn shared_write(&mut self, start: usize, count: usize, values: &[[f64; 8]]) -> Result<(), String> {
        if write_shared(&mut self.values[start..], &self.prototype, count, values) {
            Ok(())
        } else {
            Err("shared subset result: coefficient layouts differ".into())
        }
    }

    /// Laurent slots a gather computes: all four, or the UV and finite ones
    /// of an IR-free lane (the others are then zero times the scale).
    const fn gather_slots<const FULL: bool>() -> &'static [usize] {
        if FULL { &[0, 1, 2, 3] } else { &[0, 3] }
    }

    /// Exact key of a gather's Cartesian moments: everything the gather
    /// reads (the computed slots, the scale, rank, momenta and origin, and
    /// the coefficient values of those slots in the order they are summed).
    fn moment_key<const FULL: bool>(&self, gather: &Gather) -> Vec<u64> {
        let subset = &self.plan.subsets[gather.subset];
        let pv = subset.pv.unwrap();
        let rank = usize::from(gather.moment_rank);
        let momenta = subset.members.len() - 1;
        let bits = |z: &C<T>| [z.re.to_f64().to_bits(), z.im.to_f64().to_bits()];
        let slots = Self::gather_slots::<FULL>();
        let used = gather.coefficient.iter().filter(|&&index| index != u32::MAX).count();
        let mut key = Vec::with_capacity(5 + 8 * (momenta + 1) + 2 * slots.len() * used);
        key.extend([5, u64::from(FULL), rank as u64, momenta as u64, self.gather_scale.to_f64().to_bits()]);
        let p = pv.p as usize;
        for z in self.geometry[p..p + 4 * momenta].iter().chain(&self.geometry[pv.origin as usize..pv.origin as usize + 4]) {
            key.extend(bits(z));
        }
        let kappas = gather.kappas.len();
        for lambda in 0..=rank / 2 {
            let top = if momenta == 0 { 0 } else { rank - 2 * lambda };
            for (k, kappa) in gather.kappas.iter().enumerate() {
                if usize::from(kappa.degree) <= top {
                    let value = &self.values[gather.coefficient[lambda * kappas + k] as usize];
                    for &slot in slots {
                        key.extend(bits(&value[slot]));
                    }
                }
            }
        }
        debug_assert_eq!(key.len(), 5 + 8 * (momenta + 1) + 2 * slots.len() * used);
        key
    }

    /// Whether a lane's gather and output are certainly finite, from the
    /// top subset's coefficients and geometry alone: the lane's plan (base
    /// or variant) has a PV top subset whose gather is its only one, no jets
    /// and no moment recursions, every coefficient is finite and a bound on
    /// every output stays far below the overflow threshold. Then no output
    /// can overflow, and without an infinite operand none is NaN, so the
    /// lane's status after the full gather and output is `Success`.
    ///
    /// Bound (rank `r`, coefficients at most `c`, momenta of l1 norm at most
    /// `m`, origin components at most `o`): a product `prod_i (p_i.z)^k_i /
    /// k_i!` has coefficients of l1 norm at most `m^|k|`; `g_lambda` sums at
    /// most `(r + 1)(r + 2)/2` of them; the metric expansion multiplies by at
    /// most `r!` over at most `(r/2 + 1)^4` terms; the gather scale by `|s|`;
    /// each axis translation by at most `(1 + o)^r`; the output by at most
    /// `(1 + 2|a| + |a^2/2 - b|) (1 + |normalization|) 4^r` (multinomial
    /// weights). The product of these bounds is required below `2^990`.
    fn top_outputs_finite(&self, plan: &DirectPlanData) -> bool {
        let Some(id) = plan.subsets.len().checked_sub(1) else { return false };
        let subset = &plan.subsets[id];
        if subset.kind != SubsetKind::Pv
            || subset.lightlike
            || plan.jet.is_some()
            || !plan.moment_program.is_empty()
            || plan.gathers.len() != 1
            || plan.gathers[0].subset != id
        {
            return false;
        }
        let (Some(start), Some(pv)) = (subset.master, subset.pv) else { return false };
        let count = plan.coefficient_counts[id] as usize;
        let mut largest = 0.0f64;
        for value in &self.values[start as usize..start as usize + count] {
            for z in value {
                if !(z.re.is_finite() && z.im.is_finite()) {
                    return false;
                }
                largest = largest.max(z.re.to_f64().abs()).max(z.im.to_f64().abs());
            }
        }
        let size = |z: &C<T>| z.re.to_f64().abs() + z.im.to_f64().abs();
        let momenta = subset.members.len() - 1;
        let m = (0..momenta)
            .map(|i| (0..4).map(|a| size(&self.geometry[pv.p as usize + 4 * i + a])).sum::<f64>())
            .fold(0.0f64, f64::max);
        let o = (0..4).map(|a| size(&self.geometry[pv.origin as usize + a])).fold(0.0f64, f64::max);
        let constants = &self.native_constants;
        let output = (1.0 + 2.0 * constants.a.to_f64().abs() + constants.a_half_squared_minus_b.to_f64().abs())
            * (1.0 + size(&constants.normalization));
        let r = f64::from(plan.settings.rank);
        let terms = (1..=plan.settings.rank).map(|k| f64::from(k).log2()).sum::<f64>()
            + 4.0 * (r / 2.0 + 1.0).log2()
            + ((r + 1.0) * (r + 2.0) / 2.0).log2();
        let log2 = largest.max(f64::MIN_POSITIVE).log2()
            + terms
            + r * m.max(1.0).log2()
            + self.gather_scale.to_f64().abs().max(1.0).log2()
            + 4.0 * r * (1.0 + o).log2()
            + output.log2()
            + 2.0 * r;
        log2.is_finite() && log2 < 990.0
    }

    /// Evaluate every external subset of this lane in double-double and
    /// inject its rounded coefficients.
    fn inject_external(&mut self) -> Result<(), String> {
        use symbolica::domains::float::DoubleFloat;
        let plan = self.plan.clone();
        let external = self.external.clone();
        let mut partial = self.partial.take().ok_or("no partial rescue")?;
        let outcome = (|| -> Result<(), String> {
            for &id in &external {
                let subset = &plan.subsets[id];
                let size = subset.members.len();
                let rank = self.injection_rank(size, subset.coefficient_rank.max(0) as u16);
                let count = plan.coefficient_counts[id] as usize;
                let start = subset.master.ok_or("partial rescue: no master")? as usize;
                let key = self.subset_key(1, &subset.members, rank, 0, &[partial.threshold.to_bits()], &self.lane_scale);
                if let Some(shared) = self.shared_get(key.as_ref()) {
                    self.shared_write(start, count, &shared?)?;
                    continue;
                }
                let outcome = (|| -> Result<Vec<[f64; 8]>, String> {
                let executor = match partial.executors.entry((size, rank)) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let small = DirectPlanData::new(self.subset_settings(size, rank))
                            .map_err(|e| e.to_string())?;
                        let mut executor = DirectExecutor::<DoubleFloat>::new(Arc::new(small), DoubleFloat::from(0.0), DirectMasters::Native)
                            .map_err(|e| e.to_string())?;
                        // It evaluates this lane's inputs, at their precision.
                        executor.input_bits = self.input_bits;
                        executor.set_stability_threshold(DoubleFloat::from(partial.threshold));
                        executor.coefficients_only = true;
                        entry.insert(executor)
                    }
                };
                let values = self.subset_row(&subset.members, |z| {
                    Complex::new(DoubleFloat::from(z.re.to_f64()), DoubleFloat::from(z.im.to_f64()))
                });
                // Its scalar masters shared with the amplitude's other
                // double-double evaluations (limb-exact keys).
                executor.set_shared_subsets(self.shared.clone());
                let result = executor
                    .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values })
                    .map_err(|e| e.to_string())?;
                if result.statuses[0] != LaneStatus::Success {
                    return Err(format!("partial rescue of {:?}: {:?}", subset.members, result.statuses[0]));
                }
                let coefficients = executor.top_coefficients().ok_or("partial rescue: no PV coefficients")?;
                if coefficients.len() < count {
                    return Err("partial rescue: coefficient layouts differ".into());
                }
                Ok(Self::shared_values(&coefficients))
                })();
                self.shared_put(key, &outcome);
                self.shared_write(start, count, &outcome?)?;
            }
            Ok(())
        })();
        self.partial = Some(partial);
        outcome
    }

    /// `2^-bits` at working precision.
    fn binary_tolerance(&self, bits: u64) -> T {
        self.prototype.one() / &self.prototype.from_usize(2).pow(bits)
    }

    /// `2^-(p - margin)` for working precision `p` bits: a tolerance `margin`
    /// bits above the working precision's unit roundoff, the same at every
    /// precision relative to it (binary64: `2^-(53 - margin)`).
    fn precision_tolerance(&self, margin: i64) -> T {
        self.tolerance_at(self.prototype.get_precision(), margin)
    }

    /// `2^-(bits - margin)` at working precision.
    fn tolerance_at(&self, bits: u32, margin: i64) -> T {
        let bits = i64::from(bits) - margin;
        if bits >= 0 {
            self.binary_tolerance(bits as u64)
        } else {
            self.prototype.from_usize(2).pow((-bits) as u64)
        }
    }

    /// The constant-one identity `sum_i c_i D_i = 1` over `members`, if their
    /// external points are affinely dependent: `sum c_i = 0`, `sum c_i q_i = 0`
    /// and `sum c_i (q_i^2 - m_i^2) = 1`. `c` follows the member order.
    fn affine_relation(&self, members: &[usize]) -> Result<Vec<C<T>>, String> {
        let k = members.len();
        if k < 2 {
            return Err("a single propagator has no denominator identity".into());
        }
        let n = k - 1;
        let origin = members[0];
        let mut p = vec![self.zero.clone(); 4 * n];
        let mut norms = Vec::with_capacity(n);
        let mut scale = self.prototype.zero();
        for j in 0..n {
            let mut norm = self.prototype.zero();
            for a in 0..4 {
                let value = csub(&self.shifts[members[j + 1]][a], &self.shifts[origin][a]);
                norm += cnorm2(&value);
                p[a * n + j] = value;
            }
            let norm = norm.sqrt();
            if norm > scale {
                scale = norm.clone();
            }
            norms.push(norm);
        }
        let original = p.clone();
        // Complete pivoting, so the numerical rank is explicit.
        let tolerance = scale * &self.binary_tolerance(40);
        let mut columns: Vec<usize> = (0..n).collect();
        let mut rank = 0;
        for step in 0..n.min(4) {
            let mut best = self.prototype.zero();
            let mut pivot = (step, step);
            for row in step..4 {
                for column in step..n {
                    let candidate = cnorm2(&p[row * n + columns[column]]);
                    if candidate > best {
                        best = candidate;
                        pivot = (row, column);
                    }
                }
            }
            if best.sqrt() <= tolerance {
                break;
            }
            if pivot.0 != step {
                for column in 0..n {
                    p.swap(step * n + column, pivot.0 * n + column);
                }
            }
            columns.swap(step, pivot.1);
            let inverse = cinv(&p[step * n + columns[step]]);
            for row in step + 1..4 {
                let factor = cmul(&p[row * n + columns[step]], &inverse);
                for column in step..n {
                    let update = cmul(&factor, &p[step * n + columns[column]]);
                    p[row * n + columns[column]] = csub(&p[row * n + columns[column]], &update);
                }
            }
            rank += 1;
        }
        if rank == n {
            return Err(format!("external points of {members:?} are affinely independent"));
        }
        let mut v = vec![self.zero.clone(); n];
        v[columns[rank]] = self.one.clone();
        for step in (0..rank).rev() {
            let mut sum = self.zero.clone();
            for column in step + 1..n {
                cfma(&mut sum, &p[step * n + columns[column]], &v[columns[column]]);
            }
            v[columns[step]] = csub(&self.zero, &cmul(&sum, &cinv(&p[step * n + columns[step]])));
        }
        // Residual and normalization from the original coordinates.
        let mut bound = self.prototype.zero();
        for j in 0..n {
            bound += cnorm2(&v[j]).sqrt() * &norms[j];
        }
        for a in 0..4 {
            let mut residual = self.zero.clone();
            for j in 0..n {
                cfma(&mut residual, &v[j], &original[a * n + j]);
            }
            if cnorm2(&residual).sqrt() > bound.clone() * &self.binary_tolerance(33) {
                return Err(format!("no exact affine relation among {members:?}"));
            }
        }
        let nprop = self.plan.settings.propagators;
        let mut constant = self.zero.clone();
        let mut constant_bound = self.prototype.zero();
        for j in 0..n {
            let member = members[j + 1];
            let f = cadd(
                &csub(&self.distances[origin * nprop + member], &self.masses[member]),
                &self.masses[origin],
            );
            constant_bound += (cnorm2(&v[j]) * &cnorm2(&f)).sqrt();
            cfma(&mut constant, &v[j], &f);
        }
        if cnorm2(&constant).sqrt() <= constant_bound * &self.binary_tolerance(33) {
            return Err(format!("homogeneous denominator relation among {members:?}"));
        }
        let inverse = cinv(&constant);
        let mut weights = Vec::with_capacity(k);
        let mut sum = self.zero.clone();
        for value in &v {
            sum = cadd(&sum, value);
        }
        weights.push(csub(&self.zero, &cmul(&sum, &inverse)));
        for value in &v {
            weights.push(cmul(value, &inverse));
        }
        Ok(weights)
    }

    /// Minimal support of the affine relation of a singular subset.
    #[inline(never)]
    fn relation_support(&self, mask: u16) -> Result<u16, String> {
        let members: Vec<usize> = (0..16).filter(|&i| mask & (1 << i) != 0).collect();
        let weights = self.affine_relation(&members)?;
        let largest = weights
            .iter()
            .map(cnorm2)
            .fold(self.prototype.zero(), |a, b| if b > a { b } else { a });
        let cut = largest * &self.binary_tolerance(80);
        let mut support = 0u16;
        for (weight, &member) in weights.iter().zip(&members) {
            if cnorm2(weight) > cut {
                support |= 1 << member;
            }
        }
        let support_members: Vec<usize> = (0..16).filter(|&i| support & (1 << i) != 0).collect();
        self.affine_relation(&support_members)?;
        Ok(support)
    }

    /// Switch the lane to the plan variant of `relation_profile`.
    fn select_variant(&mut self) -> Result<(), ReductionError> {
        let key = (self.relation_profile.clone(), self.lightlike_profile.clone());
        if self.relation_profile.is_empty() && self.lightlike_profile.is_empty() {
            self.plan = self.base_plan.clone();
            self.constants = self.base_constants.clone();
        } else if let Some((plan, constants)) = self.variants.get(&key) {
            self.plan = plan.clone();
            self.constants = constants.clone();
        } else {
            let plan = Arc::new(DirectPlanData::with_profile_boost(
                self.base_plan.settings.clone(),
                &self.relation_profile,
                &self.lightlike_profile,
                self.base_plan.top_child_boost,
            )?);
            let constants: Arc<Vec<T>> = Arc::new(
                plan.constants
                    .iter()
                    .map(|value| self.prototype.from_rational(value))
                    .collect(),
            );
            self.variants.insert(key, (plan.clone(), constants.clone()));
            self.plan = plan;
            self.constants = constants;
        }
        let plan = &self.plan;
        if self.values.len() < plan.value_count {
            self.values.resize(plan.value_count, lzero(&self.zero));
        }
        if self.geometry.len() < plan.geometry_count {
            self.geometry.resize(plan.geometry_count, self.zero.clone());
        }
        let (value_count, geometry_count) = (plan.value_count, plan.geometry_count);
        for values in &mut self.values_jet {
            if values.len() < value_count {
                values.resize(value_count, lzero(&self.zero));
            }
        }
        for geometry in &mut self.geometry_jet {
            if geometry.len() < geometry_count {
                geometry.resize(geometry_count, self.zero.clone());
            }
        }
        Ok(())
    }

    /// Compute every subset's numeric geometry for the current variant.
    fn geometry(&mut self, plan: &DirectPlanData) -> Result<(), GeometryFailure> {
        self.failed_condition = None;
        self.external.clear();
        self.gram_external.clear();
        self.gram_top_active = false;
        self.estimate_tripped = false;
        let n = plan.settings.propagators;
        let threshold = self.stability_threshold.clone();
        for relation in &plan.relations {
            let weights = self
                .affine_relation(&relation.members)
                .map_err(GeometryFailure::Relation)?;
            for (j, weight) in weights.into_iter().enumerate() {
                self.geometry[relation.weights as usize + j] = weight;
            }
        }
        let top_mask = plan.subsets.last().map(|subset| subset.mask);
        let partial = self.partial.is_some() && plan.jet.is_none() && plan.lambda_shift == 0;
        let gram = self.gram.is_some() && plan.jet.is_none() && plan.lambda_shift == 0;
        let flags = self.estimate_flags && plan.jet.is_none() && plan.lambda_shift == 0;
        for (subset_id, subset) in plan.subsets.iter().enumerate() {
            let members = &subset.members;
            let origin = members[0];
            let d = |i: usize, j: usize| &self.distances[i * n + j];
            match subset.kind {
                SubsetKind::Pv => {
                    let geometry = subset.pv.unwrap();
                    let momenta = geometry.n;
                    self.geometry[geometry.m0 as usize] = self.masses[origin].clone();
                    for i in 0..momenta {
                        let member = members[i + 1];
                        self.geometry[geometry.f as usize + i] = cadd(
                            &csub(d(origin, member), &self.masses[member]),
                            &self.masses[origin],
                        );
                        for a in 0..4 {
                            self.geometry[geometry.p as usize + 4 * i + a] =
                                csub(&self.shifts[member][a], &self.shifts[origin][a]);
                        }
                    }
                    for a in 0..4 {
                        self.geometry[geometry.origin as usize + a] = self.shifts[origin][a].clone();
                    }
                    if momenta == 1 && (subset.lightlike || subset.coefficient_rank >= 1) {
                        // Lightlike or coincident bubble: |p^2| negligible
                        // against its masses (or exactly scaleless).
                        let p2 = d(origin, members[1]);
                        let scale = [p2, &self.masses[origin], &self.masses[members[1]]]
                            .iter()
                            .map(|z| cnorm2(z).sqrt())
                            .fold(self.prototype.zero(), |a, b| if b > a { b } else { a });
                        // Precision-relative: 2^-40 in binary64 (13 bits above
                        // its roundoff), 2^-93 in double-double, 2^-(p-13) at p
                        // MPFR bits; never a fixed binary64 bound at higher
                        // precision.
                        let lightlike = cnorm2(p2).sqrt() <= scale * &self.tolerance_at(self.input_bits, LIGHTLIKE_MARGIN_BITS);
                        if subset.lightlike {
                            if !lightlike {
                                return Err(GeometryFailure::Relation(format!(
                                    "bubble {members:?} is not lightlike on this lane"
                                )));
                            }
                            continue;
                        }
                        if lightlike {
                            return Err(GeometryFailure::Lightlike(subset.mask));
                        }
                    }
                    if subset.coefficient_rank >= 1 && momenta > 0 {
                        let size = momenta;
                        let aux_top = self.gram_top.is_some() && Some(subset.mask) == top_mask;
                        let choose = gram && (self.gram_everywhere || self.gram_estimated);
                        let flag = flags && !self.estimate_tripped;
                        if (choose || flag) && !aux_top && (2..=3).contains(&size) {
                            // COLLIER's per-box choice: off Passarino-Veltman
                            // where its estimate is worse (w > 2) and the
                            // adjugates do not vanish.
                            let zero = self.zero.clone();
                            let z: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
                                std::array::from_fn(|j| {
                                    if i >= size || j >= size {
                                        return zero.clone();
                                    }
                                    let (mi, mj) = (members[i + 1], members[j + 1]);
                                    csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj))
                                })
                            });
                            let f: [C<T>; 3] = std::array::from_fn(|i| {
                                if i < size { self.geometry[geometry.f as usize + i].clone() } else { zero.clone() }
                            });
                            let rank = subset.coefficient_rank.max(1);
                            let switch = |w: f64| if self.gram_everywhere {
                                w > 2.0
                            } else {
                                2f64.powi(-52) * w.powi(rank) > 2f64.powi(-36)
                            };
                            if gram::gram_indicator(&z, &f, size).is_some_and(switch) {
                                if choose {
                                    self.gram_external.push(subset_id);
                                    continue;
                                }
                                self.estimate_tripped = true;
                            }
                        }
                        if self.gram_top.is_some() && (2..=3).contains(&size) && Some(subset.mask) == top_mask {
                            // Auxiliary expansion executor: no Gram system.
                            self.gram_top_active = true;
                            continue;
                        }
                        // Kinematic scale of the recursion: Gram, f and m0^2.
                        let mut scale = cnorm2(&self.masses[origin]).sqrt();
                        for i in 0..size {
                            let value = cnorm2(&self.geometry[geometry.f as usize + i]).sqrt();
                            if value > scale {
                                scale = value;
                            }
                        }
                        for i in 0..size {
                            for j in 0..size {
                                let (mi, mj) = (members[i + 1], members[j + 1]);
                                self.matrix[i * size + j] =
                                    csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
                                self.rhs[i * size + j] =
                                    if i == j { self.one.clone() } else { self.zero.clone() };
                            }
                        }
                        for value in &self.matrix[..size * size] {
                            let value = cnorm2(value).sqrt();
                            if value > scale {
                                scale = value;
                            }
                        }
                        let quality = solve_in_place(
                            &mut self.matrix[..size * size],
                            &mut self.rhs[..size * size],
                            size,
                            size,
                        )
                        .ok_or_else(|| {
                            GeometryFailure::Singular(
                                subset.mask,
                                format!("singular Gram matrix for {members:?}"),
                            )
                        });
                        if quality.is_err() {
                            // Exactly singular pivot: conditioning zero.
                            self.failed_condition = Some(0.0);
                        }
                        let quality = quality?;
                        let power = subset.coefficient_rank as u64;
                        // Hadamard-relative and scale-relative conditioning:
                        // |det Z|^(1/n) against the recursion's kinematic scale.
                        let mut scaled = quality.determinant.clone();
                        for _ in 1..size {
                            scaled = scaled.sqrt();
                        }
                        if size == 3 {
                            scaled = quality.determinant.clone().powf(&(self.prototype.one() / &self.prototype.from_usize(3)));
                        }
                        let scaled = if scale.is_zero() { scale.clone() } else { scaled / &scale };
                        let condition = if scaled < quality.relative_determinant {
                            scaled
                        } else {
                            quality.relative_determinant.clone()
                        };
                        if condition.clone().pow(power) < threshold {
                            if gram && (2..=3).contains(&size) {
                                // Small-Gram expansion (box or triangle), injected in the lane loop
                                // (with the a priori choice too: its retry covers refused lanes).
                                let estimate = condition.clone().pow(power).to_f64();
                                self.failed_condition =
                                    Some(self.failed_condition.map_or(estimate, |f: f64| f.min(estimate)));
                                self.gram_external.push(subset_id);
                                continue;
                            }
                            if partial && (2..=3).contains(&size) && Some(subset.mask) != top_mask {
                                // Computed in double-double and injected.
                                let estimate = condition.clone().pow(power).to_f64();
                                self.failed_condition =
                                    Some(self.failed_condition.map_or(estimate, |f: f64| f.min(estimate)));
                                self.external.push(subset_id);
                                continue;
                            }
                            self.failed_condition = Some(condition.clone().pow(power).to_f64());
                            return Err(GeometryFailure::Singular(subset.mask, format!(
                                "ill-conditioned Gram matrix for {members:?}: conditioning {:e}",
                                condition
                            )));
                        }
                        for j in 0..size {
                            let mut zs = self.zero.clone();
                            let mut zf = self.zero.clone();
                            for i in 0..size {
                                let value = self.rhs[j * size + i].clone();
                                zs = cadd(&zs, &value);
                                cfma(&mut zf, &value, &self.geometry[geometry.f as usize + i]);
                                self.geometry[geometry.zinv as usize + j * size + i] = value;
                            }
                            self.geometry[geometry.zs as usize + j] = zs;
                            self.geometry[geometry.zf as usize + j] = zf;
                        }
                    }
                }
                SubsetKind::Pentagon => self.geometry_pentagon(subset, n)?,
                SubsetKind::Relation(_) => {}
                SubsetKind::Hexagon => self.geometry_hexagon(subset, n)?,
            }
        }
        Ok(())
    }

    /// Scalar pentagon geometry of `subset` (the Cayley system, and the
    /// modified Cayley and Gram systems of its coefficients): out of line,
    /// as only five or more points reach it.
    #[inline(never)]
    fn geometry_pentagon(&mut self, subset: &Subset, n: usize) -> Result<(), GeometryFailure> {
        let threshold = self.stability_threshold.clone();
        let pentagon_calibration = self.binary_tolerance(7);
        let members = &subset.members;
        let origin = members[0];
        let d = |i: usize, j: usize| &self.distances[i * n + j];
        let geometry = subset.pentagon.unwrap();
        // b = R^-1 1 for the scalar pentagon.
        for i in 0..5 {
            for j in 0..5 {
                let (mi, mj) = (members[i], members[j]);
                self.matrix[i * 5 + j] = csub(
                    &csub(d(mi, mj), &self.masses[mi]),
                    &self.masses[mj],
                );
            }
            self.rhs[i] = self.one.clone();
        }
        let quality = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..5], 5, 1)
            .ok_or_else(|| GeometryFailure::Unstable(format!("singular Cayley matrix for {members:?}")));
        if quality.is_err() {
            // Exactly singular pivot: conditioning zero.
            self.failed_condition = Some(0.0);
        }
        let quality = quality?;
        if quality.relative_determinant < threshold {
            self.failed_condition = Some(quality.relative_determinant.to_f64());
            return Err(GeometryFailure::Unstable(format!(
                "ill-conditioned Cayley matrix for {members:?}: relative determinant {:e}",
                quality.relative_determinant
            )));
        }
        for i in 0..5 {
            self.geometry[geometry.b as usize + i] = self.rhs[i].clone();
        }
        if geometry.dd_rank >= 1 {
            // Gram-free coefficients: momenta and origin for the
            // gather, X^-1 of X = [[2 m0^2, f^T], [f, Z]] and the
            // second adjugates Xtt_(kn)(0i) / det X.
            let pv = subset.pv.unwrap();
            for i in 0..4 {
                let member = members[i + 1];
                let f = cadd(&csub(d(origin, member), &self.masses[member]), &self.masses[origin]);
                self.matrix[i + 1] = f.clone();
                self.matrix[(i + 1) * 5] = f;
                for j in 0..4 {
                    let (mi, mj) = (members[i + 1], members[j + 1]);
                    self.matrix[(i + 1) * 5 + j + 1] =
                        csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
                }
                for a in 0..4 {
                    self.geometry[pv.p as usize + 4 * i + a] =
                        csub(&self.shifts[member][a], &self.shifts[origin][a]);
                }
            }
            for a in 0..4 {
                self.geometry[pv.origin as usize + a] = self.shifts[origin][a].clone();
            }
            self.matrix[0] = cadd(&self.masses[origin], &self.masses[origin]);
            for i in 0..5 {
                for j in 0..5 {
                    self.rhs[i * 5 + j] = if i == j { self.one.clone() } else { self.zero.clone() };
                }
            }
            let quality = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..25], 5, 5)
                .ok_or_else(|| GeometryFailure::Unstable(format!("singular modified Cayley matrix for {members:?}")));
            if quality.is_err() {
                self.failed_condition = Some(0.0);
            }
            let quality = quality?;
            if quality.relative_determinant < threshold {
                self.failed_condition = Some(quality.relative_determinant.to_f64());
                return Err(GeometryFailure::Unstable(format!(
                    "ill-conditioned modified Cayley matrix for {members:?}: relative determinant {:e}",
                    quality.relative_determinant
                )));
            }
            let xinv: Vec<C<T>> = self.rhs[..25].to_vec();
            for (i, value) in xinv.iter().enumerate() {
                self.geometry[geometry.xinv as usize + i] = value.clone();
            }
            let sums = geometry.sums as usize;
            let mut column = self.zero.clone();
            for m in 1..=4 {
                column = cadd(&column, &xinv[5 * m]);
            }
            self.geometry[sums] = column;
            for k in 1..=4 {
                let mut row = self.zero.clone();
                for m in 0..5 {
                    row = cadd(&row, &xinv[5 * k + m]);
                }
                self.geometry[sums + k] = row;
            }
            for k in 1..=4 {
                for i in 1..=4 {
                    let mut total = self.zero.clone();
                    for m in 1..=4 {
                        let value = csub(
                            &cmul(&xinv[m], &xinv[5 * i + k]),
                            &cmul(&xinv[5 * i + m], &xinv[k]),
                        );
                        if m != k {
                            total = cadd(&total, &value);
                        }
                        self.geometry[geometry.adjadj as usize + 16 * (k - 1) + 4 * (m - 1) + (i - 1)] = value;
                    }
                    self.geometry[sums + 5 + 4 * (k - 1) + (i - 1)] = total;
                }
            }
        }
        if subset.moment_rank > geometry.dd_rank.max(0) {
            let mut f = [self.zero.clone(), self.zero.clone(), self.zero.clone(), self.zero.clone()];
            for i in 0..4 {
                let member = members[i + 1];
                f[i] = cadd(&csub(d(origin, member), &self.masses[member]), &self.masses[origin]);
                for j in 0..4 {
                    let (mi, mj) = (members[i + 1], members[j + 1]);
                    self.matrix[i * 4 + j] =
                        csub(&cadd(d(origin, mi), d(origin, mj)), d(mi, mj));
                    self.rhs[i * 4 + j] =
                        if i == j { self.one.clone() } else { self.zero.clone() };
                }
            }
            // Row-sum norm of Z before the solve destroys it.
            let mut norm = self.prototype.zero();
            for i in 0..4 {
                let mut row = self.prototype.zero();
                for j in 0..4 {
                    row += cnorm2(&self.matrix[i * 4 + j]).sqrt();
                }
                if row > norm {
                    norm = row;
                }
            }
            let quality = solve_in_place(&mut self.matrix[..16], &mut self.rhs[..16], 4, 4)
                .ok_or_else(|| GeometryFailure::Singular(subset.mask, format!("singular pentagon Gram matrix for {members:?}")));
            if quality.is_err() {
                // Exactly singular pivot: conditioning zero.
                self.failed_condition = Some(0.0);
            }
            let _ = quality?;
            // The moment recursion applies Z^-1 once per rank, so
            // rounding grows like kappa^r with kappa the condition
            // number ||Z|| ||Z^-1|| (row sums; rhs now holds Z^-1).
            // On physical 2->3 points the binary64 error was within
            // ~50 kappa^r u (u = 2^-53); the Hadamard-relative
            // determinant missed this (accepted lanes with E/A
            // up to 3e-4), though kappa alone also overestimates
            // (an opening N6 row with kappa 60 is good to 2e-11).
            // This gate only removes hopeless lanes: 2^-7/kappa^r
            // against the threshold accepts kappa^r <= 8.6e9 in
            // binary64 (1e-12) and kappa^r <= 2^(bits-47) at a
            // rescue tier. The binary64 hybrid checks the
            // remaining N>=5 lanes a posteriori (loop rotation).
            let mut inverse_norm = self.prototype.zero();
            for i in 0..4 {
                let mut row = self.prototype.zero();
                for j in 0..4 {
                    row += cnorm2(&self.rhs[i * 4 + j]).sqrt();
                }
                if row > inverse_norm {
                    inverse_norm = row;
                }
            }
            let kappa = norm * &inverse_norm;
            let condition = if kappa.is_zero() || !kappa.is_finite() {
                self.prototype.zero()
            } else {
                pentagon_calibration.clone()
                    / &kappa.clone().pow((subset.moment_rank - geometry.dd_rank.max(0)) as u64)
            };
            if condition < threshold {
                self.failed_condition = Some(condition.to_f64());
                return Err(GeometryFailure::Singular(subset.mask, format!(
                    "ill-conditioned pentagon Gram matrix for {members:?}: condition number {:e}",
                    kappa
                )));
            }
            for mu in 0..4 {
                let mut a = csub(&self.zero, &self.shifts[origin][mu]);
                let mut s = self.zero.clone();
                for j in 0..4 {
                    let mut w = self.zero.clone();
                    for i in 0..4 {
                        let p = csub(&self.shifts[members[i + 1]][mu], &self.shifts[origin][mu]);
                        cfma(&mut w, &p, &self.rhs[i * 4 + j]);
                    }
                    let wf = cmul(&w, &f[j]);
                    a = csub(&a, &wf);
                    s = cadd(&s, &w);
                    self.geometry[geometry.w as usize + 4 * mu + j] = w;
                }
                self.geometry[geometry.a as usize + mu] = a;
                self.geometry[geometry.s as usize + mu] = s;
            }
        }
        Ok(())
    }

    /// Six-point coordinate system of `subset` (six or more points): out of
    /// line, as only six or more points reach it.
    #[inline(never)]
    fn geometry_hexagon(&mut self, subset: &Subset, n: usize) -> Result<(), GeometryFailure> {
        let threshold = self.stability_threshold.clone();
        let members = &subset.members;
        let d = |i: usize, j: usize| &self.distances[i * n + j];
        // Thin identity sum_i w_i D_i = 1 (column 0) or q^mu
        // (column mu+1) over six members: the 5x5 system of their
        // four momentum components and f row. A seven-point lane
        // drops the member whose remaining six give the best
        // conditioned system (its weight is zero); an N-point lane
        // drops the N - 6 members whose remaining six do.
        let weights = subset.hexagon.unwrap() as usize;
        let size = members.len();
        let half = self.prototype.one() / &self.prototype.from_usize(2);
        // Dropped members as bit sets over the member positions.
        let drops: Vec<u32> = (0u32..(1 << size)).filter(|m| m.count_ones() as usize + 6 == size).collect();
        let mut best: Option<(T, Vec<C<T>>, u32)> = None;
        for drop in drops {
            let chosen: Vec<usize> = (0..size).filter(|&k| drop & (1 << k) == 0).collect();
            let origin = members[chosen[0]];
            for column in 0..5 {
                let member = members[chosen[column + 1]];
                for mu in 0..4 {
                    self.matrix[mu * 5 + column] =
                        csub(&self.shifts[member][mu], &self.shifts[origin][mu]);
                }
                self.matrix[4 * 5 + column] =
                    cadd(&csub(d(origin, member), &self.masses[member]), &self.masses[origin]);
            }
            for row in 0..5 {
                for column in 0..5 {
                    self.rhs[row * 5 + column] = if column == 0 {
                        if row == 4 { self.one.clone() } else { self.zero.clone() }
                    } else if row == 4 {
                        csub(&self.zero, &self.shifts[origin][column - 1])
                    } else if row == column - 1 {
                        let sign = if row == 0 { half.clone() } else { -half.clone() };
                        C::new(sign, self.prototype.zero())
                    } else {
                        self.zero.clone()
                    };
                }
            }
            let Some(quality) = solve_in_place(&mut self.matrix[..25], &mut self.rhs[..25], 5, 5) else {
                continue;
            };
            if best.as_ref().is_none_or(|(relative, _, _)| quality.relative_determinant > *relative) {
                best = Some((quality.relative_determinant, self.rhs[..25].to_vec(), drop));
            }
        }
        let Some((relative, solution, drop)) = best else {
            // Exactly singular pivot: conditioning zero.
            self.failed_condition = Some(0.0);
            return Err(GeometryFailure::Unstable(format!(
                "singular six-point coordinate system for {members:?}"
            )));
        };
        if relative < threshold {
            self.failed_condition = Some(relative.to_f64());
            return Err(GeometryFailure::Unstable(format!(
                "ill-conditioned six-point coordinate system: relative determinant {:e}",
                relative
            )));
        }
        let chosen: Vec<usize> = (0..size).filter(|&k| drop & (1 << k) == 0).collect();
        for column in 0..5 {
            let mut sum = self.zero.clone();
            for i in 0..5 {
                let value = solution[i * 5 + column].clone();
                sum = cadd(&sum, &value);
                self.geometry[weights + size * column + chosen[i + 1]] = value;
            }
            self.geometry[weights + size * column + chosen[0]] = csub(&self.zero, &sum);
            for dropped in (0..size).filter(|&k| drop & (1 << k) != 0) {
                self.geometry[weights + size * column + dropped] = self.zero.clone();
            }
        }
        Ok(())
    }

    /// PV coefficients of every lightlike bubble of the current variant,
    /// from the Feynman-parameter representation at `p^2 = 0`:
    /// `C(lambda,k) = (-1)^k/(2^lambda lambda!) [I0/eps_UV + H_lambda I0 - I1]`,
    /// `I0 = int_0^1 x^k D^lambda`, `I1 = int_0^1 x^k D^lambda log(D/mu^2)`,
    /// `D(x) = m0^2 + x (m1^2 - m0^2)`. Scaleless bubbles keep their
    /// cancelling UV/IR poles at `lambda = 0` and vanish otherwise.
    #[inline(never)]
    fn lightlike_coefficients(&mut self, mu2: &C<T>) -> Result<(), String> {
        use oneloop::rust_codegen::runtime::primitives::log;
        let plan = self.plan.clone();
        // Scratch buffers, reused across subsets and entries.
        let (mut l_values, mut m_values, mut a_powers, mut b_powers): (Vec<C<T>>, Vec<C<T>>, Vec<C<T>>, Vec<C<T>>) =
            Default::default();
        for (subset, entries) in &plan.lightlike {
            let members = &plan.subsets[*subset].members;
            let a = self.masses[members[0]].clone();
            let b = csub(&self.masses[members[1]], &a);
            let max_k = entries.iter().map(|&(l, k, _)| usize::from(l + k)).max().unwrap_or(0);
            let a_zero = a.re.is_zero() && a.im.is_zero();
            let b_zero = b.re.is_zero() && b.im.is_zero();
            // m1^2 itself, not a + b, which cancels when m1 << m0.
            let end = self.masses[members[1]].clone();
            let end_zero = end.re.is_zero() && end.im.is_zero();
            let real_negative = |z: &C<T>| z.re < z.re.zero();
            if (!a_zero && real_negative(&a)) || (!end_zero && real_negative(&end)) {
                return Err("lightlike bubble with a negative real squared mass".into());
            }
            // L[n] = int_0^1 x^n log(a + b x) dx, n <= max_k.
            l_values.clear();
            l_values.resize(max_k + 1, self.zero.clone());
            if a_zero && b_zero {
                // Scaleless; L is not used.
            } else if b_zero {
                let log_a = log(&a);
                for (n, value) in l_values.iter_mut().enumerate() {
                    *value = cscale(&log_a, &(self.prototype.one() / &self.prototype.from_usize(n + 1)));
                }
            } else if end_zero {
                // m1 = 0: int x^n log(a (1-x)) = log(a)/(n+1) - H_(n+1)/(n+1).
                let log_a = log(&a);
                let mut harmonic = self.prototype.zero();
                for (n, value) in l_values.iter_mut().enumerate() {
                    let inverse = self.prototype.one() / &self.prototype.from_usize(n + 1);
                    harmonic += &inverse;
                    *value = csub(
                        &cscale(&log_a, &inverse),
                        &C::new(harmonic.clone() * &inverse, self.prototype.zero()),
                    );
                }
            } else if a_zero {
                let log_b = log(&b);
                for (n, value) in l_values.iter_mut().enumerate() {
                    let inverse = self.prototype.one() / &self.prototype.from_usize(n + 1);
                    *value = csub(&cscale(&log_b, &inverse), &C::new(inverse.clone() * &inverse, self.prototype.zero()));
                }
            } else {
                // M[k] = int_0^1 x^k / (a + b x) dx, k <= max_k + 1.
                m_values.clear();
                m_values.resize(max_k + 2, self.zero.clone());
                let inverse_a = cinv(&a);
                let ratio = cmul(&b, &inverse_a);
                if cnorm2(&ratio) <= self.prototype.one() / &self.prototype.from_usize(4) {
                    let minus_ratio = csub(&self.zero, &ratio);
                    // Truncated 7 bits below the working precision's roundoff
                    // (2^-60 in binary64). |ratio| <= 1/2, so the m-th term is
                    // at most 2^-m of the first and the tolerance is reached
                    // within p + 8 terms; the bound only stops a nonfinite sum.
                    let tolerance = self.precision_tolerance(-7);
                    let terms = self.prototype.get_precision() as usize + 64;
                    for (k, value) in m_values.iter_mut().enumerate() {
                        let mut sum = self.zero.clone();
                        let mut power = self.one.clone();
                        for m in 0..terms {
                            let term = cscale(&power, &(self.prototype.one() / &self.prototype.from_usize(k + m + 1)));
                            sum = cadd(&sum, &term);
                            if cnorm2(&term).sqrt() <= cnorm2(&sum).sqrt() * &tolerance {
                                break;
                            }
                            power = cmul(&power, &minus_ratio);
                        }
                        *value = cmul(&sum, &inverse_a);
                    }
                } else {
                    let inverse_b = cinv(&b);
                    m_values[0] = cmul(&csub(&log(&end), &log(&a)), &inverse_b);
                    for k in 1..m_values.len() {
                        let step = csub(
                            &C::new(self.prototype.one() / &self.prototype.from_usize(k), self.prototype.zero()),
                            &cmul(&a, &m_values[k - 1]),
                        );
                        m_values[k] = cmul(&step, &inverse_b);
                    }
                }
                let log_end = log(&end);
                for (n, value) in l_values.iter_mut().enumerate() {
                    let inverse = self.prototype.one() / &self.prototype.from_usize(n + 1);
                    *value = csub(&cscale(&log_end, &inverse), &cscale(&cmul(&b, &m_values[n + 1]), &inverse));
                }
            }
            let log_mu2 = log(mu2);
            for &(lambda, k, index) in entries {
                let mut value = lzero(&self.zero);
                if a_zero && b_zero {
                    if lambda == 0 {
                        let residue = self.prototype.one() / &self.prototype.from_usize(usize::from(k) + 1);
                        let residue = if k % 2 == 0 { residue } else { -residue };
                        value[0] = C::new(residue.clone(), self.prototype.zero());
                        value[1] = C::new(-residue, self.prototype.zero());
                    }
                } else {
                    let (mut i0, mut i1) = (self.zero.clone(), self.zero.clone());
                    // (a + b x)^lambda = sum_i binom(lambda,i) a^(lambda-i) b^i x^i.
                    a_powers.clear();
                    a_powers.resize(usize::from(lambda) + 1, self.one.clone());
                    b_powers.clear();
                    b_powers.resize(usize::from(lambda) + 1, self.one.clone());
                    for i in 1..=usize::from(lambda) {
                        a_powers[i] = cmul(&a_powers[i - 1], &a);
                        b_powers[i] = cmul(&b_powers[i - 1], &b);
                    }
                    for i in 0..=usize::from(lambda) {
                        let n = usize::from(k) + i;
                        let weight = cscale(
                            &cmul(&a_powers[usize::from(lambda) - i], &b_powers[i]),
                            &self.binomials[usize::from(lambda)][i],
                        );
                        let inverse = self.prototype.one() / &self.prototype.from_usize(n + 1);
                        i0 = cadd(&i0, &cscale(&weight, &inverse));
                        let log_term = csub(&l_values[n], &cscale(&log_mu2, &inverse));
                        i1 = cadd(&i1, &cmul(&weight, &log_term));
                    }
                    let mut harmonic = self.prototype.zero();
                    let mut normalization = self.prototype.one();
                    for i in 1..=usize::from(lambda) {
                        harmonic += &(self.prototype.one() / &self.prototype.from_usize(i));
                        normalization = normalization / &self.prototype.from_usize(2 * i);
                    }
                    if k % 2 == 1 {
                        normalization = -normalization;
                    }
                    value[0] = cscale(&i0, &normalization);
                    value[3] = cscale(&csub(&cscale(&i0, &harmonic), &i1), &normalization);
                }
                self.values[index as usize] = value;
            }
        }
        Ok(())
    }

    /// `d f / d delta` of `f_member = d(origin, member) - m_member^2 + m_origin^2`.
    fn f_derivative(&self, origin: usize, member: usize, dotted: usize) -> T {
        let mut value = self.prototype.zero();
        if origin == dotted {
            value += &self.prototype.one();
        }
        if member == dotted {
            value -= &self.prototype.one();
        }
        value
    }

    /// Taylor orders `1..K` of every mass-dependent geometry entry. The
    /// dotted squared mass `m_d^2 + delta` enters `m0^2`, `f`, the Cayley
    /// matrices of pentagons, the six-point `f` row and relation constants.
    #[inline(never)]
    fn geometry_jets(&mut self) -> Result<(), String> {
        let Some((dotted, k_len)) = self.plan.jet else {
            return Ok(());
        };
        let plan = self.plan.clone();
        let n = plan.settings.propagators;
        for order in &mut self.geometry_jet {
            for value in order.iter_mut().take(plan.geometry_count) {
                *value = self.zero.clone();
            }
        }
        let real = |value: T| C::new(value, self.prototype.zero());
        for relation in &plan.relations {
            // w(delta) = w0 / (1 + t delta), t = sum_j w0_j df_j.
            let base = relation.weights as usize;
            let origin = relation.members[0];
            let mut t = self.zero.clone();
            for (j, &member) in relation.members.iter().enumerate().skip(1) {
                let df = self.f_derivative(origin, member, dotted);
                t = cadd(&t, &cscale(&self.geometry[base + j], &df));
            }
            let minus_t = csub(&self.zero, &t);
            for j in 0..relation.members.len() {
                let mut value = self.geometry[base + j].clone();
                for order in 0..k_len - 1 {
                    value = cmul(&value, &minus_t);
                    self.geometry_jet[order][base + j] = value.clone();
                }
            }
        }
        for subset in &plan.subsets {
            let members = &subset.members;
            let origin = members[0];
            match subset.kind {
                SubsetKind::Pv => {
                    let geometry = subset.pv.unwrap();
                    if origin == dotted {
                        self.geometry_jet[0][geometry.m0 as usize] = self.one.clone();
                    }
                    let mut df = Vec::with_capacity(geometry.n);
                    for i in 0..geometry.n {
                        let value = self.f_derivative(origin, members[i + 1], dotted);
                        self.geometry_jet[0][geometry.f as usize + i] = real(value.clone());
                        df.push(value);
                    }
                    if subset.coefficient_rank >= 1 && geometry.n > 0 {
                        for j in 0..geometry.n {
                            let mut zf = self.zero.clone();
                            for i in 0..geometry.n {
                                let zinv = &self.geometry[geometry.zinv as usize + j * geometry.n + i];
                                zf = cadd(&zf, &cscale(zinv, &df[i]));
                            }
                            self.geometry_jet[0][geometry.zf as usize + j] = zf;
                        }
                    }
                }
                SubsetKind::Pentagon => {
                    let geometry = subset.pentagon.unwrap();
                    let local = members.iter().position(|&m| m == dotted);
                    // R(delta) = R0 - delta E, E_ij = [i=d] + [j=d]:
                    // b_k = R0^-1 E b_(k-1).
                    if let Some(local) = local {
                        let mut previous: Vec<C<T>> =
                            (0..5).map(|i| self.geometry[geometry.b as usize + i].clone()).collect();
                        for order in 0..k_len - 1 {
                            let mut sum = self.zero.clone();
                            for value in &previous {
                                sum = cadd(&sum, value);
                            }
                            for i in 0..5 {
                                for j in 0..5 {
                                    let (mi, mj) = (members[i], members[j]);
                                    self.matrix[i * 5 + j] = csub(
                                        &csub(&self.distances[mi * n + mj], &self.masses[mi]),
                                        &self.masses[mj],
                                    );
                                }
                                self.rhs[i] = previous[local].clone();
                                if i == local {
                                    self.rhs[i] = cadd(&self.rhs[i], &sum);
                                }
                            }
                            solve_in_place(&mut self.matrix[..25], &mut self.rhs[..5], 5, 1)
                                .ok_or_else(|| "singular pentagon Cayley jet system".to_string())?;
                            for i in 0..5 {
                                previous[i] = self.rhs[i].clone();
                                self.geometry_jet[order][geometry.b as usize + i] = previous[i].clone();
                            }
                        }
                    }
                    if subset.moment_rank >= 1 {
                        for mu in 0..4 {
                            let mut a = self.zero.clone();
                            for j in 0..4 {
                                let df = self.f_derivative(origin, members[j + 1], dotted);
                                let w = &self.geometry[geometry.w as usize + 4 * mu + j];
                                a = csub(&a, &cscale(w, &df));
                            }
                            self.geometry_jet[0][geometry.a as usize + mu] = a;
                        }
                    }
                }
                SubsetKind::Hexagon => {
                    // A(delta) = A0 + delta A1, A1 only in the f row:
                    // X_k = -A0^-1 A1 X_(k-1) = y (-a1 . X_(k-1)), y = A0^-1 e_4.
                    let weights = subset.hexagon.unwrap() as usize;
                    let a1: Vec<T> = (0..5)
                        .map(|c| self.f_derivative(origin, members[c + 1], dotted))
                        .collect();
                    if a1.iter().all(|value| value.is_zero()) {
                        continue;
                    }
                    for column in 0..5 {
                        let member = members[column + 1];
                        for mu in 0..4 {
                            self.matrix[mu * 5 + column] =
                                csub(&self.shifts[member][mu], &self.shifts[origin][mu]);
                        }
                        self.matrix[4 * 5 + column] = cadd(
                            &csub(&self.distances[origin * n + member], &self.masses[member]),
                            &self.masses[origin],
                        );
                    }
                    for row in 0..5 {
                        self.rhs[row] = if row == 4 { self.one.clone() } else { self.zero.clone() };
                    }
                    solve_in_place(&mut self.matrix[..25], &mut self.rhs[..5], 5, 1)
                        .ok_or_else(|| "singular six-point jet system".to_string())?;
                    let y: Vec<C<T>> = self.rhs[..5].to_vec();
                    for order in 0..k_len - 1 {
                        for column in 0..5 {
                            let mut r = self.zero.clone();
                            for c in 0..5 {
                                let previous = if order == 0 {
                                    &self.geometry[weights + 6 * column + c + 1]
                                } else {
                                    &self.geometry_jet[order - 1][weights + 6 * column + c + 1]
                                };
                                r = csub(&r, &cscale(previous, &a1[c]));
                            }
                            let mut sum = self.zero.clone();
                            for i in 0..5 {
                                let value = cmul(&y[i], &r);
                                sum = cadd(&sum, &value);
                                self.geometry_jet[order][weights + 6 * column + i + 1] = value;
                            }
                            self.geometry_jet[order][weights + 6 * column] = csub(&self.zero, &sum);
                        }
                    }
                }
                SubsetKind::Relation(_) => {}
            }
        }
        Ok(())
    }

    /// Inverse of the modified Cayley matrix `R_ij = d_ij - m_i^2 - m_j^2`.
    fn cayley_inverse(&mut self, mask: u16) -> Result<Vec<C<T>>, String> {
        if let Some(inverse) = self.rinv_memo.get(&mask) {
            return Ok(inverse.clone());
        }
        let n = self.plan.settings.propagators;
        let members: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        let size = members.len();
        let mut matrix = vec![self.zero.clone(); size * size];
        let mut inverse = vec![self.zero.clone(); size * size];
        for i in 0..size {
            for j in 0..size {
                matrix[i * size + j] = csub(
                    &csub(&self.distances[members[i] * n + members[j]], &self.masses[members[i]]),
                    &self.masses[members[j]],
                );
            }
            inverse[i * size + i] = self.one.clone();
        }
        let quality = solve_in_place(&mut matrix, &mut inverse, size, size)
            .ok_or_else(|| format!("singular Cayley matrix for {members:?}"));
        if quality.is_err() {
            // Exactly singular pivot: conditioning zero.
            self.failed_condition = Some(0.0);
        }
        let quality = quality?;
        if quality.relative_determinant < self.stability_threshold {
            self.failed_condition = Some(quality.relative_determinant.to_f64());
            return Err(format!(
                "ill-conditioned Cayley matrix for {members:?}: relative determinant {:e}",
                quality.relative_determinant
            ));
        }
        self.rinv_memo.insert(mask, inverse.clone());
        Ok(inverse)
    }

    /// Unit-dimension scalar with propagator powers `powers` (4 bits each),
    /// reduced to the lane's masters by Eq. (19) of hep-ph/0303184, as in
    /// the validated symbolic `ibp.rs`:
    /// `(v_l - 1) J(v) = sum_i Rinv_li [-(5-|v|-2eps) J(w) + sum_j w_j J(w-e_i+e_j)]`,
    /// `w = v - e_l`. Every child has a smaller total power.
    fn ibp(&mut self, powers: u64) -> Result<L4<T>, String> {
        if let Some(value) = self.ibp_memo.get(&powers) {
            return Ok(value.clone());
        }
        let n = self.plan.settings.propagators;
        let power = |packed: u64, i: usize| (packed >> (4 * i)) & 15;
        let mask: u16 = (0..n).filter(|&i| power(powers, i) > 0).fold(0, |m, i| m | (1 << i));
        if mask == 0 {
            return Ok(lzero(&self.zero));
        }
        let Some(l) = (0..n).find(|&i| power(powers, i) > 1) else {
            let subset = self
                .plan
                .subsets
                .iter()
                .find(|subset| subset.mask == mask && subset.master.is_some())
                .ok_or_else(|| format!("no master for subset {mask:#b}"))?;
            return Ok(self.values[subset.master.unwrap() as usize].clone());
        };
        let total: u64 = (0..n).map(|i| power(powers, i)).sum();
        let active: Vec<usize> = (0..n).filter(|&i| power(powers, i) > 0).collect();
        let inverse = self.cayley_inverse(mask)?;
        let local_l = active.iter().position(|&i| i == l).unwrap();
        let w = powers - (1 << (4 * l));
        let base = self.ibp(w)?;
        let mut common = base;
        mul_dimension(&mut common, &self.prototype.from_i64(5 - total as i64));
        let common: L4<T> = std::array::from_fn(|slot| csub(&self.zero, &common[slot]));
        let mut result = lzero(&self.zero);
        let size = active.len();
        for (local_i, &i) in active.iter().enumerate() {
            let mut term = common.clone();
            for &j in &active {
                let wj = power(w, j);
                if wj == 0 {
                    continue;
                }
                let child = w - (1 << (4 * i)) + (1 << (4 * j));
                let value = self.ibp(child)?;
                lfma(&mut term, &C::new(self.prototype.from_usize(wj as usize), self.prototype.zero()), &value);
            }
            lfma(&mut result, &inverse[local_l * size + local_i], &term);
        }
        let divisor = self.prototype.one() / &self.prototype.from_usize(power(w, l) as usize);
        let result: L4<T> = std::array::from_fn(|slot| cscale(&result[slot], &divisor));
        self.ibp_memo.insert(powers, result.clone());
        Ok(result)
    }

    /// Taylor coefficients `J(1_S + k e_d)` of every master containing `d`.
    fn master_jets(&mut self) -> Result<(), String> {
        let Some((dotted, k_len)) = self.plan.jet else {
            return Ok(());
        };
        self.ibp_memo.clear();
        self.rinv_memo.clear();
        let plan = self.plan.clone();
        for order in &mut self.values_jet {
            for value in order.iter_mut().take(plan.value_count) {
                *value = lzero(&self.zero);
            }
        }
        for family in &plan.masters {
            for &(subset, index) in family {
                let mask = plan.subsets[subset].mask;
                if mask & (1 << dotted) == 0 {
                    continue;
                }
                let unit: u64 = (0..16).filter(|&i| mask & (1 << i) != 0).map(|i| 1u64 << (4 * i)).sum();
                for order in 1..k_len {
                    let value = self.ibp(unit + ((order as u64) << (4 * dotted)))?;
                    self.values_jet[order - 1][index as usize] = value;
                }
            }
        }
        Ok(())
    }

    /// Recurrence program on Taylor jets: order k accumulates
    /// `sum_(a+b=k) geometry_a constant value_b`.
    #[inline(never)]
    fn run_program_jets(&mut self, plan: &DirectPlanData, program_is_moments: bool) {
        let k_len = plan.jet.map_or(1, |(_, k)| k);
        let program = if program_is_moments {
            &plan.moment_program
        } else {
            &plan.coefficient_program
        };
        let mut accumulator = vec![lzero(&self.zero); k_len];
        let mut factors = vec![self.zero.clone(); k_len];
        for instruction in program {
            for value in &mut accumulator {
                *value = lzero(&self.zero);
            }
            for term in &plan.terms[instruction.terms.0 as usize..instruction.terms.1 as usize] {
                let constant = &self.constants[term.constant as usize];
                let orders = if term.geometry == NO_GEOMETRY {
                    factors[0] = C::new(constant.clone(), self.prototype.zero());
                    1
                } else {
                    let g = term.geometry as usize;
                    factors[0] = cscale(&self.geometry[g], constant);
                    for a in 1..k_len {
                        factors[a] = cscale(&self.geometry_jet[a - 1][g], constant);
                    }
                    k_len
                };
                let src = term.src as usize;
                for a in 0..orders {
                    if factors[a].re.is_zero() && factors[a].im.is_zero() {
                        continue;
                    }
                    for b in 0..k_len - a {
                        let value = if b == 0 {
                            &self.values[src]
                        } else {
                            &self.values_jet[b - 1][src]
                        };
                        lfma(&mut accumulator[a + b], &factors[a], value);
                    }
                }
            }
            for value in &mut accumulator {
                if let PostOp::DivideDimension { inverse } = instruction.post {
                    div_dimension(value, &self.constants[inverse as usize]);
                }
            }
            let dst = instruction.dst as usize;
            self.values[dst] = accumulator[0].clone();
            for order in 1..k_len {
                self.values_jet[order - 1][dst] = accumulator[order].clone();
            }
        }
    }

    /// Gather the Cartesian moments of Taylor order `order` (zero is the
    /// ordinary value array). The gather is linear and mass independent.
    fn gather_order(&mut self, plan: &Arc<DirectPlanData>, order: usize) {
        if order == 0 {
            self.gather(plan);
        } else {
            std::mem::swap(&mut self.values, &mut self.values_jet[order - 1]);
            self.gather(plan);
            std::mem::swap(&mut self.values, &mut self.values_jet[order - 1]);
        }
    }

    fn run_program(&mut self, plan: &DirectPlanData, program_is_moments: bool) {
        match (self.plan.jet, self.ir_free) {
            (Some((_, 2)), true) => self.run_program_jet::<false, 2>(plan, program_is_moments),
            (Some((_, 2)), false) => self.run_program_jet::<true, 2>(plan, program_is_moments),
            (Some((_, 3)), true) => self.run_program_jet::<false, 3>(plan, program_is_moments),
            (Some((_, 3)), false) => self.run_program_jet::<true, 3>(plan, program_is_moments),
            (Some((_, 4)), true) => self.run_program_jet::<false, 4>(plan, program_is_moments),
            (Some((_, 4)), false) => self.run_program_jet::<true, 4>(plan, program_is_moments),
            (Some((_, 5)), true) => self.run_program_jet::<false, 5>(plan, program_is_moments),
            (Some((_, 5)), false) => self.run_program_jet::<true, 5>(plan, program_is_moments),
            (Some(_), _) => self.run_program_jets(plan, program_is_moments),
            (None, true) => self.run_program_ordinary::<false>(plan, program_is_moments),
            (None, false) => self.run_program_ordinary::<true>(plan, program_is_moments),
        }
    }

    /// Whether every master (and closed-form) value of this lane has
    /// vanishing IR slots. Recurrences are linear and never move UV or
    /// finite parts into IR slots, so those slots then stay exactly zero.
    fn detect_ir_free(&mut self, plan: &DirectPlanData) {
        let zero = |z: &C<T>| z.re.is_zero() && z.im.is_zero();
        let mut free = true;
        for family in &plan.masters {
            for &(_, index) in family {
                let value = &self.values[index as usize];
                free &= zero(&value[1]) && zero(&value[2]);
                for order in &self.values_jet {
                    let value = &order[index as usize];
                    free &= zero(&value[1]) && zero(&value[2]);
                }
            }
        }
        for (_, entries) in &plan.lightlike {
            for &(_, _, index) in entries {
                let value = &self.values[index as usize];
                free &= zero(&value[1]) && zero(&value[2]);
            }
        }
        self.ir_free = free;
    }

    /// Taylor length `K`: order `k` accumulates `sum_(a+b=k) g_a v_b`, skipping
    /// the vanishing orders of mass-independent geometry.
    #[inline(never)]
    fn run_program_jet<const FULL: bool, const K: usize>(&mut self, plan: &DirectPlanData, program_is_moments: bool) {
        let program = if program_is_moments {
            &plan.moment_program
        } else {
            &plan.coefficient_program
        };
        for instruction in program {
            let mut acc: [L4<T>; K] = std::array::from_fn(|_| lzero(&self.zero));
            for term in &plan.terms[instruction.terms.0 as usize..instruction.terms.1 as usize] {
                let constant = &self.constants[term.constant as usize];
                let src = term.src as usize;
                if term.geometry == NO_GEOMETRY {
                    let factor = C::new(constant.clone(), self.prototype.zero());
                    lfma_slots::<T, FULL>(&mut acc[0], &factor, &self.values[src]);
                    for b in 1..K {
                        lfma_slots::<T, FULL>(&mut acc[b], &factor, &self.values_jet[b - 1][src]);
                    }
                    continue;
                }
                let g = term.geometry as usize;
                for a in 0..K {
                    let ga = if a == 0 { &self.geometry[g] } else { &self.geometry_jet[a - 1][g] };
                    if ga.re.is_zero() && ga.im.is_zero() {
                        continue;
                    }
                    let factor = if term.constant == 0 { ga.clone() } else { cscale(ga, constant) };
                    lfma_slots::<T, FULL>(&mut acc[a], &factor, &self.values[src]);
                    for b in 1..K - a {
                        lfma_slots::<T, FULL>(&mut acc[a + b], &factor, &self.values_jet[b - 1][src]);
                    }
                }
            }
            if let PostOp::DivideDimension { inverse } = instruction.post {
                let inverse = &self.constants[inverse as usize];
                for value in &mut acc {
                    div_dimension(value, inverse);
                }
            }
            let dst = instruction.dst as usize;
            self.values[dst] = acc[0].clone();
            for order in 1..K {
                self.values_jet[order - 1][dst] = acc[order].clone();
            }
        }
    }

    #[inline(never)]
    fn run_program_ordinary<const FULL: bool>(&mut self, plan: &DirectPlanData, program_is_moments: bool) {
        let program = if program_is_moments {
            &plan.moment_program
        } else {
            &plan.coefficient_program
        };
        if program.is_empty() {
            // Nothing to run, skip or share.
            return;
        }
        // Instructions of externally computed (injected) subsets are skipped.
        let skip: Vec<(u32, u32)> = if program_is_moments {
            Vec::new()
        } else {
            let top = (self.gram_top_active).then(|| plan.subsets.len() - 1);
            self.external
                .iter()
                .chain(&self.gram_external)
                .chain(top.iter())
                .map(|&id| plan.coefficient_ranges[id])
                .collect()
        };
        // Binary64 PV sub-integrals and Gram-free pentagons shared with the
        // other integrals of an amplitude: a stored subset is injected and
        // its instructions skipped, the others are stored after the program.
        // A subset's coefficients are fixed by its propagators and mu^2, the
        // executor's settings and the lane's IR-free flag, and its plan
        // signature (the ranks, kinds and lightlike flags of the subsets
        // beneath it, which decide their gates, injections and closed forms;
        // a lightlike flag holds exactly on the lanes whose kinematics make
        // the bubble lightlike, so plan variants share too).
        let mut skip = skip;
        let mut computed: Vec<(usize, usize, Vec<u64>)> = Vec::new();
        let store = self.shared.clone().filter(|store| store.enabled() && !program_is_moments);
        if let (Some(store), Some(mu2)) = (store.as_ref(), self.lane_mu2.clone()) {
            if self.gram_top.is_none() {
                let gram = self.gram.as_ref();
                let mut config = [
                    u64::from(FULL),
                    self.stability_threshold.to_f64().to_bits(),
                    self.partial.as_ref().map_or(u64::MAX, |partial| partial.threshold.to_bits()),
                    gram.map_or(u64::MAX, |gram| gram.signature()),
                    u64::from(self.gram_everywhere),
                    u64::from(self.gram_estimated),
                    u64::from(self.gram_alt_only),
                    u64::from(self.gram_alt_direct),
                    u64::from(self.gram_extra_ranks),
                    0,
                ];
                let mut key = Vec::with_capacity(96);
                for (id, subset) in plan.subsets.iter().enumerate() {
                    let (a, b) = plan.coefficient_ranges[id];
                    if a == b || subset.kind != SubsetKind::Pv || subset.lightlike {
                        continue;
                    }
                    // A short program costs less than a lookup.
                    if plan.coefficient_terms[id] < SHARED_MIN_TERMS {
                        continue;
                    }
                    if skip.iter().any(|&(x, y)| x == a && y == b) {
                        continue;
                    }
                    let Some(start) = subset.master.map(|m| m as usize) else { continue };
                    let count = plan.coefficient_counts[id] as usize;
                    let rank = subset.coefficient_rank.max(0) as u16;
                    config[9] = plan.subset_signatures[id];
                    self.fill_subset_key(&mut key, 3, &subset.members, rank, count, &config, &mu2);
                    let (values, prototype) = (&mut self.values[start..], &self.prototype);
                    let hit = store.get_with(&key, |stored| stored.as_ref().is_ok_and(|v| write_shared(values, prototype, count, v)));
                    if hit == Some(true) {
                        skip.push((a, b));
                    } else {
                        computed.push((start, count, key.clone()));
                    }
                }
                // Gram-free pentagons: a pentagon integral and the hexagon
                // containing it compute the same pentagon from the same boxes.
                for (id, block) in plan.pentagon_blocks.iter().enumerate() {
                    let Some(block) = block else { continue };
                    let subset = &plan.subsets[id];
                    let (start, count) = (block.start as usize, block.count as usize);
                    let rank = subset.coefficient_rank.max(0) as u16;
                    config[9] = plan.subset_signatures[id];
                    self.fill_subset_key(&mut key, 6, &subset.members, rank, count, &config, &mu2);
                    let (values, prototype) = (&mut self.values[start..], &self.prototype);
                    let hit = store.get_with(&key, |stored| stored.as_ref().is_ok_and(|v| write_shared(values, prototype, count, v)));
                    if hit == Some(true) {
                        skip.push(block.instructions);
                    } else {
                        computed.push((start, count, key.clone()));
                    }
                }
            }
        }
        // Disjoint borrows: the slices stay in registers across the loop
        // instead of being reloaded through `self` after every store.
        let constants: &[T] = &self.constants;
        let geometry: &[C<T>] = &self.geometry;
        let terms: &[Term] = &plan.terms;
        let zero = &self.zero;
        let real_zero = self.prototype.zero();
        let values = &mut self.values;
        // The instructions outside `skip`, in program order: the run before
        // each skipped range, then the rest.
        skip.sort_unstable();
        let (mut at, mut next) = (0usize, 0usize);
        loop {
            let end = skip.get(next).map_or(program.len(), |&(a, _)| (a as usize).min(program.len()));
            for instruction in program.get(at..end).unwrap_or(&[]) {
                let mut accumulator = lzero(zero);
                for term in &terms[instruction.terms.0 as usize..instruction.terms.1 as usize] {
                    let factor = if term.geometry == NO_GEOMETRY {
                        C::new(constants[term.constant as usize].clone(), real_zero.clone())
                    } else {
                        let geometry = &geometry[term.geometry as usize];
                        if geometry.re.is_zero() && geometry.im.is_zero() {
                            continue;
                        }
                        if term.constant == 0 {
                            geometry.clone()
                        } else {
                            cscale(geometry, &constants[term.constant as usize])
                        }
                    };
                    lfma_slots::<T, FULL>(&mut accumulator, &factor, &values[term.src as usize]);
                }
                match instruction.post {
                    PostOp::None => {}
                    PostOp::DivideDimension { inverse } => {
                        div_dimension(&mut accumulator, &constants[inverse as usize]);
                    }
                }
                values[instruction.dst as usize] = accumulator;
            }
            let Some(&(_, b)) = skip.get(next) else { break };
            at = at.max(b as usize);
            next += 1;
        }
        if let Some(store) = store {
            for (start, count, key) in computed {
                store.put(key, Ok(Self::shared_values(&self.values[start..start + count])));
            }
        }
    }

    /// Step 1 of [`Self::gather_with`]: the momentum products as sparse
    /// complex lists.
    fn complex_products(&mut self, plan: &DirectPlanData, skip_unneeded: bool) {
        let table = &plan.monomials;
        self.sparse_products.clear();
        self.product_ranges.clear();
        for (index, product) in plan.products.iter().enumerate() {
            let start = self.sparse_products.len();
            if skip_unneeded && !self.product_needed[index] {
                self.product_ranges.push((start as u32, start as u32));
                continue;
            }
            if product.degree == 0 {
                self.sparse_products.push((0, self.one.clone()));
                self.product_ranges.push((start as u32, start as u32 + 1));
                continue;
            }
            let block = table.block(usize::from(product.degree));
            let parent_block = table.block(usize::from(product.degree) - 1);
            let rows = &table.multiply[parent_block];
            let (from, to) = self.product_ranges[product.parent as usize];
            let inverse = self.prototype.one() / &self.prototype.from_usize(usize::from(product.divisor));
            let p = product.momentum as usize;
            let mut axes = [0usize; 4];
            let mut count = 0;
            for a in 0..4 {
                let component = &self.geometry[p + a];
                if !(component.re.is_zero() && component.im.is_zero()) {
                    axes[count] = a;
                    count += 1;
                }
            }
            let axes = &axes[..count];
            for entry in from as usize..to as usize {
                let (local, ref value) = self.sparse_products[entry];
                let value = cscale(value, &inverse);
                let row = &rows[local as usize];
                for &a in axes {
                    let target = row[a] as usize - block.start;
                    let mut slot = self.product_slot[target];
                    if slot == NO_SLOT {
                        // First reached: starts from zero.
                        slot = self.sparse_products.len() as u32;
                        self.product_slot[target] = slot;
                        self.sparse_products.push((target as u32, self.zero.clone()));
                    }
                    // slot + value * component (binary64: as written).
                    cfma(&mut self.sparse_products[slot as usize].1, &value, &self.geometry[p + a]);
                }
            }
            // Release the slots and drop exact zeros, keeping the order.
            let mut end = start;
            for index in start..self.sparse_products.len() {
                let (target, ref value) = self.sparse_products[index];
                self.product_slot[target as usize] = NO_SLOT;
                if !(value.re.is_zero() && value.im.is_zero()) {
                    if end != index {
                        self.sparse_products.swap(end, index);
                    }
                    end += 1;
                }
            }
            self.sparse_products.truncate(end);
            self.product_ranges.push((start as u32, end as u32));
        }
    }

    /// The number of distinct momenta of `plan`'s products (offsets into the
    /// geometry), recorded with each product's index among them.
    fn distinct_momenta(&mut self, plan: &Arc<DirectPlanData>) -> usize {
        if !self.product_momenta.as_ref().is_some_and(|(of, _, _)| Arc::ptr_eq(of, plan)) {
            let mut offsets: Vec<u32> = Vec::new();
            let momentum_of = plan
                .products
                .iter()
                .map(|product| {
                    if product.degree == 0 {
                        return u16::MAX;
                    }
                    let index = offsets.iter().position(|&offset| offset == product.momentum).unwrap_or_else(|| {
                        offsets.push(product.momentum);
                        offsets.len() - 1
                    });
                    index as u16
                })
                .collect();
            self.product_momenta = Some((plan.clone(), offsets, momentum_of));
        }
        self.product_momenta.as_ref().expect("set above").1.len()
    }

    /// Step 1 of [`Self::gather_with`] for binary64 with real momenta: the
    /// products as sparse real lists, in the order and with the values of
    /// [`Self::complex_products`] (their imaginary parts are +0). `false`,
    /// with nothing usable written, when a momentum is not real or a product
    /// is not finite.
    ///
    /// Where the lists go depends only on which momentum components are
    /// zero (and which products are needed) and on which sums cancel to an
    /// exact zero (dropped; massless momenta along an axis cancel exactly).
    /// A pass is recorded as a [`ProductSchedule`] for its zero pattern, and
    /// later passes with that pattern replay it, the same terms in the same
    /// order without finding the entries again, if their exact zeros are the
    /// recorded ones (otherwise this pass runs and replaces the record).
    fn real_products(&mut self, plan: &Arc<DirectPlanData>, skip_unneeded: bool) -> bool {
        let table = &plan.monomials;
        // The zero pattern: per distinct momentum of the products, its
        // nonzero components (bits 0-3); then, when products are skipped,
        // which ones are needed.
        self.distinct_momenta(plan);
        let (_, offsets, _) = self.product_momenta.as_ref().expect("set above");
        self.schedule_key.clear();
        for &offset in offsets {
            let mut mask = 0u8;
            for a in 0..4 {
                let component = &self.geometry[offset as usize + a];
                if !component.im.is_zero() {
                    return false;
                }
                if !component.re.is_zero() {
                    mask |= 1 << a;
                }
            }
            self.schedule_key.push(mask);
        }
        if skip_unneeded {
            self.schedule_key.push(u8::MAX);
            self.schedule_key.extend(self.product_needed.iter().map(|&needed| u8::from(needed)));
        }
        let found = self
            .product_schedules
            .iter()
            .position(|schedule| Arc::ptr_eq(&schedule.plan, plan) && schedule.key == self.schedule_key);
        if let Some(index) = found {
            if let Some(finite) = self.replay_products(plan, index) {
                return finite;
            }
        }
        self.sparse_real.clear();
        self.product_ranges.clear();
        self.schedule_slots.clear();
        let mut locals = Vec::new();
        let mut keep = Vec::new();
        let mut counts = Vec::with_capacity(plan.products.len());
        let mut finite = true;
        for (index, product) in plan.products.iter().enumerate() {
            let start = self.sparse_real.len();
            if skip_unneeded && !self.product_needed[index] {
                self.product_ranges.push((start as u32, start as u32));
                counts.push((0, 0));
                continue;
            }
            if product.degree == 0 {
                self.sparse_real.push((0, self.one.re.clone()));
                self.product_ranges.push((start as u32, start as u32 + 1));
                locals.push(0);
                keep.push(true);
                counts.push((1, 1));
                continue;
            }
            let block = table.block(usize::from(product.degree));
            let parent_block = table.block(usize::from(product.degree) - 1);
            let rows = &table.multiply[parent_block];
            let (from, to) = self.product_ranges[product.parent as usize];
            let inverse = self.prototype.one() / &self.prototype.from_usize(usize::from(product.divisor));
            let p = product.momentum as usize;
            let mut axes = [0usize; 4];
            let mut count = 0;
            for a in 0..4 {
                if !self.geometry[p + a].re.is_zero() {
                    axes[count] = a;
                    count += 1;
                }
            }
            let axes = &axes[..count];
            for entry in from as usize..to as usize {
                let (local, ref value) = self.sparse_real[entry];
                let value = value.clone() * &inverse;
                let row = &rows[local as usize];
                for &a in axes {
                    let target = row[a] as usize - block.start;
                    let mut slot = self.product_slot[target];
                    if slot == NO_SLOT {
                        slot = self.sparse_real.len() as u32;
                        self.product_slot[target] = slot;
                        self.sparse_real.push((target as u32, self.zero.re.clone()));
                    }
                    self.schedule_slots.push(slot - start as u32);
                    let total = &mut self.sparse_real[slot as usize].1;
                    *total = total.clone() + value.clone() * &self.geometry[p + a].re;
                }
            }
            // Release the slots and drop exact zeros, keeping the order.
            let mut end = start;
            for index in start..self.sparse_real.len() {
                let (target, ref value) = self.sparse_real[index];
                self.product_slot[target as usize] = NO_SLOT;
                finite &= value.to_f64().is_finite();
                let kept = !value.is_zero();
                locals.push(target);
                keep.push(kept);
                if kept {
                    if end != index {
                        self.sparse_real.swap(end, index);
                    }
                    end += 1;
                }
            }
            counts.push(((self.sparse_real.len() - start) as u32, (end - start) as u32));
            self.sparse_real.truncate(end);
            self.product_ranges.push((start as u32, end as u32));
        }
        if finite {
            let schedule = ProductSchedule {
                plan: plan.clone(),
                key: self.schedule_key.clone(),
                locals,
                keep,
                counts,
                slots: self.schedule_slots.clone(),
            };
            match found {
                // Its replay found other exact zeros: this pass replaces it.
                Some(index) => self.product_schedules[index] = schedule,
                None => {
                    if self.product_schedules.len() >= MAX_PRODUCT_SCHEDULES {
                        self.product_schedules.remove(0);
                    }
                    self.product_schedules.push(schedule);
                }
            }
        }
        finite
    }

    /// [`Self::real_products`] by its recorded schedule `index`: `None` when
    /// the exact zeros (dropped entries) are not the recorded ones.
    fn replay_products(&mut self, plan: &DirectPlanData, index: usize) -> Option<bool> {
        let slots = std::mem::take(&mut self.product_schedules[index].slots);
        let recorded = &self.product_schedules[index];
        let (_, _, momentum_of) = self.product_momenta.as_ref().expect("set by real_products");
        self.sparse_real.clear();
        self.product_ranges.clear();
        let mut cursor = 0;
        let mut at = 0;
        let mut same = true;
        'products: for (k, product) in plan.products.iter().enumerate() {
            let start = self.sparse_real.len();
            let (before, after) = (recorded.counts[k].0 as usize, recorded.counts[k].1 as usize);
            if before == 0 {
                self.product_ranges.push((start as u32, start as u32));
                continue;
            }
            if product.degree == 0 {
                self.sparse_real.push((0, self.one.re.clone()));
                self.product_ranges.push((start as u32, start as u32 + 1));
                at += 1;
                continue;
            }
            let zero = &self.zero.re;
            self.sparse_real.extend(recorded.locals[at..at + before].iter().map(|&local| (local, zero.clone())));
            let (from, to) = self.product_ranges[product.parent as usize];
            let inverse = self.prototype.one() / &self.prototype.from_usize(usize::from(product.divisor));
            let p = product.momentum as usize;
            let mask = self.schedule_key[usize::from(momentum_of[k])];
            let mut components: [T; 4] = std::array::from_fn(|_| self.zero.re.clone());
            let mut count = 0;
            for a in 0..4 {
                if mask & (1 << a) != 0 {
                    components[count] = self.geometry[p + a].re.clone();
                    count += 1;
                }
            }
            let components = &components[..count];
            let terms = (to - from) as usize * count;
            let product_slots = &slots[cursor..cursor + terms];
            cursor += terms;
            if count > 0 {
                // Parents precede the product's entries; each term adds to
                // one of them (slots from the product's first entry).
                let (parents, entries) = self.sparse_real.split_at_mut(start);
                let parents = &parents[from as usize..to as usize];
                for ((_, parent), terms) in parents.iter().zip(product_slots.chunks_exact(count)) {
                    let value = parent.clone() * &inverse;
                    for (&slot, component) in terms.iter().zip(components) {
                        let total = &mut entries[slot as usize].1;
                        *total = total.clone() + value.clone() * component;
                    }
                }
            }
            if after != before {
                // Drop the recorded exact zeros, which must be the only ones.
                let mut end = start;
                for (i, &kept) in recorded.keep[at..at + before].iter().enumerate() {
                    let index = start + i;
                    if !self.sparse_real[index].1.is_zero() != kept {
                        same = false;
                        break 'products;
                    }
                    if kept {
                        if end != index {
                            self.sparse_real.swap(end, index);
                        }
                        end += 1;
                    }
                }
                self.sparse_real.truncate(end);
            }
            at += before;
            self.product_ranges.push((start as u32, (start + after) as u32));
        }
        self.product_schedules[index].slots = slots;
        if !same {
            return None;
        }
        // No other exact zero; finite (the dropped zeros are).
        let mut finite = true;
        let mut zero = false;
        for (_, value) in &self.sparse_real {
            let x = value.to_f64();
            finite &= x.is_finite();
            zero |= x == 0.0;
        }
        (!zero).then_some(finite)
    }

    /// Cartesian moments of every gathered PV subset.
    ///
    /// 1. Momentum products `P_k(z) = prod_i (p_i.z)^(k_i)/k_i!`.
    /// 2. `g_lambda(f) = sum_(|k|=|f|) C(lambda, k) [z^f] P_k`.
    /// 3. Untranslated moments by the closed metric expansion
    ///    `L(e) = sum_a e! (-1)^(a1+a2+a3)/(2^|a| a!) g_|a|(e-2a)`.
    /// 4. Translation to the loop momentum, `M(e) = sum_f binom(e,f)
    ///    (-o)^(e-f) L(f)`, as separable 1-D binomial transforms along the
    ///    nonzero axes of the subset origin `o`.
    fn gather(&mut self, plan: &Arc<DirectPlanData>) {
        if self.ir_free {
            self.gather_with::<false>(plan, plan);
        } else {
            self.gather_with::<true>(plan, plan);
        }
    }

    #[inline(never)]
    fn gather_with<const FULL: bool>(&mut self, plan_arc: &Arc<DirectPlanData>, plan: &DirectPlanData) {
        let table = &plan.monomials;
        let stride = table.block(usize::from(plan.settings.rank)).end;
        // 0. Moments shared with the other integrals of an amplitude (binary64
        //    only, sizes and ranks the store shares): stored moments are
        //    written now and their gathers skipped; the others are stored
        //    once gathered. The key holds every input, so a stored entry is
        //    exactly what the gather computes.
        enum Moments {
            Stored,
            Store(Vec<u64>),
            Compute,
        }
        let store = self.shared.clone().filter(|store| store.enabled());
        let slots = Self::gather_slots::<FULL>();
        let mut moments: Vec<Moments> = Vec::new();
        if let Some(store) = store.as_ref() {
            // The slots a gather leaves: zero times the scale (step 3).
            let left = cscale(&lzero(&self.zero)[1], &self.gather_scale);
            self.product_needed.clear();
            self.product_needed.resize(plan.products.len(), false);
            for gather in &plan.gathers {
                let size = plan.subsets[gather.subset].members.len();
                if !store.shares_moments(size, gather.moment_rank) {
                    // Computed here and not stored.
                    for &product in &gather.products {
                        self.product_needed[product as usize] = true;
                    }
                    moments.push(Moments::Compute);
                    continue;
                }
                let key = self.moment_key::<FULL>(gather);
                let base = gather.moment_base as usize;
                let len = table.block(usize::from(gather.moment_rank)).end;
                let (values, prototype) = (&mut self.values, &self.prototype);
                let hit = store.moments_with(&key, |stored| {
                    if stored.len() != 2 * slots.len() * len {
                        return false;
                    }
                    for (e, v) in stored.chunks_exact(2 * slots.len()).enumerate() {
                        let moment = &mut values[base + e];
                        if !FULL {
                            moment[1] = left.clone();
                            moment[2] = left.clone();
                        }
                        for (k, &slot) in slots.iter().enumerate() {
                            moment[slot] = C::new(exact_f64(prototype, v[2 * k]), exact_f64(prototype, v[2 * k + 1]));
                        }
                    }
                    true
                });
                if hit {
                    moments.push(Moments::Stored);
                } else {
                    for &product in &gather.products {
                        self.product_needed[product as usize] = true;
                    }
                    moments.push(Moments::Store(key));
                }
            }
        }
        // 1. Products as sparse (local monomial, value) lists, once for all
        //    gathers of an origin (only those a computed gather uses). Each
        //    list is in the order its monomials are first reached: that order
        //    fixes the order of every later sum, so it is kept exactly. A
        //    product's parent belongs to every gather using the product.
        //    Binary64 with real momenta (real kinematics) and finite
        //    coefficients: the products are real and kept as reals, and step
        //    2 multiplies by a real factor. Bit-identical: the complex
        //    arithmetic adds only the zero products of a zero imaginary part,
        //    and an accumulator started at +0 never becomes -0, so adding a
        //    zero leaves it unchanged; a non-finite product falls back. Only
        //    from two distinct momenta (triangles and up): the lists of
        //    tadpoles and bubbles are short, and recording and replaying them
        //    costs more than the real arithmetic saves (a tadpole, with no
        //    momentum, would also compare empty keys, whose dangling pointers
        //    cost glibc's memcmp about 200 cycles in its masked loads on AMD
        //    Zen 4).
        let mut finite = T::FIXED_BITS == Some(53) && self.distinct_momenta(plan_arc) >= REAL_PRODUCT_MOMENTA;
        'check: for (index, gather) in plan.gathers.iter().enumerate().filter(|_| finite) {
            if matches!(moments.get(index), Some(Moments::Stored)) {
                continue;
            }
            let rank = usize::from(gather.moment_rank);
            let momenta = plan.subsets[gather.subset].members.len() - 1;
            let kappas = gather.kappas.len();
            for lambda in 0..=rank / 2 {
                let top = if momenta == 0 { 0 } else { rank - 2 * lambda };
                for (k, kappa) in gather.kappas.iter().enumerate() {
                    if usize::from(kappa.degree) > top {
                        continue;
                    }
                    let value = &self.values[gather.coefficient[lambda * kappas + k] as usize];
                    let finite_slot =
                        |slot: usize| value[slot].re.to_f64().is_finite() && value[slot].im.to_f64().is_finite();
                    if !slots.iter().all(|&slot| finite_slot(slot)) {
                        finite = false;
                        break 'check;
                    }
                }
            }
        }
        let real = finite && self.real_products(plan_arc, store.is_some());
        if !real {
            self.complex_products(plan, store.is_some());
        }
        for (index, gather) in plan.gathers.iter().enumerate() {
            let key = match moments.get_mut(index).map(|m| std::mem::replace(m, Moments::Compute)) {
                Some(Moments::Stored) => continue,
                Some(Moments::Store(key)) => Some(key),
                Some(Moments::Compute) | None => None,
            };
            let subset = &plan.subsets[gather.subset];
            let geometry = subset.pv.unwrap();
            let rank = usize::from(gather.moment_rank);
            let momenta = subset.members.len() - 1;
            // 2. g_lambda blocks; without momenta only degree zero exists.
            let kappas = gather.kappas.len();
            for lambda in 0..=rank / 2 {
                let top = if momenta == 0 { 0 } else { rank - 2 * lambda };
                let base = lambda * stride;
                for slot in &mut self.g_buffer[base..base + table.block(top).end] {
                    *slot = lzero(&self.zero);
                }
                for (k, kappa) in gather.kappas.iter().enumerate() {
                    let degree = usize::from(kappa.degree);
                    if degree > top {
                        continue;
                    }
                    let coefficient = gather.coefficient[lambda * kappas + k];
                    let value = &self.values[coefficient as usize];
                    let block_start = base + table.block(degree).start;
                    let (from, to) = self.product_ranges[gather.products[k] as usize];
                    if real {
                        for (local, product) in &self.sparse_real[from as usize..to as usize] {
                            let total = &mut self.g_buffer[block_start + *local as usize];
                            real_lfma_slots::<T, FULL>(total, product, value);
                        }
                    } else {
                        for (local, product) in &self.sparse_products[from as usize..to as usize] {
                            lfma_slots::<T, FULL>(&mut self.g_buffer[block_start + *local as usize], product, value);
                        }
                    }
                }
            }
            // 3. Untranslated moments.
            let moment_base = gather.moment_base as usize;
            for e in 0..table.block(rank).end {
                let mut total = lzero(&self.zero);
                // One constant per term of `qexp[e]` (built from it, same length).
                for (&(src, lambda, _), constant) in table.qexp[e].iter().zip(&self.qexp_constants[e]) {
                    if momenta == 0 && src != 0 {
                        continue;
                    }
                    let value = &self.g_buffer[usize::from(lambda) * stride + src as usize];
                    for slot in if FULL { 0..4 } else { 0..1 } {
                        cfma_real(&mut total[slot], &value[slot], constant);
                    }
                    if !FULL {
                        cfma_real(&mut total[3], &value[3], constant);
                    }
                }
                for slot in &mut total {
                    *slot = cscale(slot, &self.gather_scale);
                }
                self.values[moment_base + e] = total;
            }
            // 4. Translation along the nonzero origin axes.
            let origin = geometry.origin as usize;
            for axis in 0..4 {
                let shift = &self.geometry[origin + axis];
                if shift.re.is_zero() && shift.im.is_zero() {
                    continue;
                }
                let t = csub(&self.zero, shift);
                self.tpow[0] = self.one.clone();
                for k in 1..=rank {
                    self.tpow[k] = cmul(&self.tpow[k - 1], &t);
                }
                for (others, line) in &table.lines[axis] {
                    let others = usize::from(*others);
                    if others > rank {
                        continue;
                    }
                    let length = rank - others;
                    for j in (1..=length).rev() {
                        let mut total = self.values[moment_base + line[j] as usize].clone();
                        for k in 0..j {
                            let factor = cscale(&self.tpow[j - k], &self.binomials[j][k]);
                            lfma_slots::<T, FULL>(&mut total, &factor, &self.values[moment_base + line[k] as usize]);
                        }
                        self.values[moment_base + line[j] as usize] = total;
                    }
                }
            }
            if let (Some(store), Some(key)) = (store.as_ref(), key) {
                let len = table.block(rank).end;
                let mut stored = Vec::with_capacity(2 * slots.len() * len);
                for moment in &self.values[moment_base..moment_base + len] {
                    for &slot in slots {
                        stored.extend([moment[slot].re.to_f64(), moment[slot].im.to_f64()]);
                    }
                }
                store.put_moments(key, stored);
            }
        }
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        if batch.components != 1 {
            return Err(ReductionError::Unsupported(
                "the direct engine evaluates ordinary numbers; use the generic workspace for Dual components".into(),
            ));
        }
        let width = self.layout.width();
        if batch.values.len() != batch.rows * width {
            return Err(ReductionError::InvalidInput(format!(
                "expected {} values per row",
                width
            )));
        }
        self.profile = DirectProfile::default();
        self.master_rows = 0;
        let batched_masters = self.evaluators.iter().any(Option::is_some);
        if batched_masters {
            let started = self.timer();
            self.evaluate_masters(batch)?;
            if let Some(started) = started {
                self.profile.masters += started.elapsed();
            }
        }
        let basis_len = self.base_plan.basis.len();
        let (convention, tensor_weights) = (self.base_plan.settings.convention, self.base_plan.settings.tensor_weights);
        let mut values = Vec::with_capacity(batch.rows * basis_len * 4);
        let mut statuses = Vec::with_capacity(batch.rows);
        self.lane_conditions.clear();
        self.lane_conditions.resize(batch.rows, None);
        self.lane_gram.clear();
        self.lane_gram.resize(batch.rows, false);
        self.lane_estimate.clear();
        self.lane_estimate.resize(batch.rows, false);
        for row in 0..batch.rows {
            let lane_values = values.len();
            values.resize(lane_values + basis_len * 4, self.zero.clone());
            if batched_masters {
                if let Some(error) = &self.master_errors[row] {
                    statuses.push(LaneStatus::MasterFailure(error.clone()));
                    continue;
                }
            }
            let started = self.timer();
            if let Err(error) = self.read_lane(batch, row) {
                statuses.push(LaneStatus::InvalidInput(error));
                continue;
            }
            // Reuse the previous lane's variant; on a relation that does not
            // hold here, restart from the base plan. Each affinely dependent
            // subset the geometry discovers switches to a relation variant.
            let mut failure = None;
            // The lane's plan, cloned once (an atomic increment and
            // decrement) and lent to every step below.
            let plan = loop {
                let plan = self.plan.clone();
                match self.geometry(&plan) {
                    Ok(()) => break plan,
                    Err(GeometryFailure::Relation(error)) => {
                        if self.relation_profile.is_empty() && self.lightlike_profile.is_empty() {
                            failure = Some(error);
                            break plan;
                        }
                        self.relation_profile.clear();
                        self.lightlike_profile.clear();
                        self.select_variant()?;
                    }
                    Err(GeometryFailure::Lightlike(mask)) => {
                        if self.plan.jet.is_some() || self.lightlike_profile.contains(&mask) {
                            failure = Some(format!(
                                "lightlike bubble {mask:#b} unsupported for dotted propagators"
                            ));
                            break plan;
                        }
                        self.lightlike_profile.push(mask);
                        self.lightlike_profile.sort_unstable();
                        self.select_variant()?;
                    }
                    Err(GeometryFailure::Unstable(error)) => {
                        failure = Some(error);
                        break plan;
                    }
                    Err(GeometryFailure::Singular(mask, error)) => {
                        match self.relation_support(mask) {
                            Ok(support)
                                if !self.relation_profile.contains(&support)
                                    && !(batched_masters && support != 0) =>
                            {
                                self.relation_profile.push(support);
                                self.relation_profile.sort_unstable();
                                self.select_variant()?;
                            }
                            _ => {
                                failure = Some(error);
                                break plan;
                            }
                        }
                    }
                }
            };
            self.lane_estimate[row] = self.estimate_tripped;
            if let Some(error) = failure {
                self.lane_conditions[row] = self.failed_condition;
                statuses.push(LaneStatus::Unstable(error));
                continue;
            }
            if let Some(started) = started {
                self.profile.geometry += started.elapsed();
            }
            let started = self.timer();
            if batched_masters {
                for family in 0..4 {
                    for (k, &(_, index)) in plan.masters[family].iter().enumerate() {
                        self.values[index as usize] = self.master_values
                            [row * self.master_count + self.master_offsets[family] + k]
                            .clone();
                    }
                }
            } else {
                let mu2 = self.mu_squared(batch, row);
                self.lane_mu2 = Some(mu2.clone());
                if let Err(error) = self.native_masters(&plan, &mu2) {
                    statuses.push(LaneStatus::MasterFailure(error));
                    continue;
                }
            }
            if let Some(started) = started {
                self.profile.masters += started.elapsed();
            }
            if !self.plan.lightlike.is_empty() {
                let mu2 = self.mu_squared(batch, row);
                if let Err(error) = self.lightlike_coefficients(&mu2) {
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
            }
            if self.plan.jet.is_some() {
                let started = self.timer();
                if let Err(error) = self.geometry_jets() {
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
                self.failed_condition = None;
                if let Err(error) = self.master_jets() {
                    self.lane_conditions[row] = self.failed_condition;
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
                if let Some(started) = started {
                    self.profile.masters += started.elapsed();
                }
            }
            if !self.external.is_empty() {
                if let Err(error) = self.inject_external() {
                    self.lane_conditions[row] = self.failed_condition;
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
            }
            if !self.gram_external.is_empty() {
                self.lane_gram[row] = true;
                if let Err(error) = self.inject_gram() {
                    self.lane_conditions[row] = self.failed_condition;
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
            }
            if let Some(unit) = plan.unit_finite {
                self.values[unit as usize] = [self.zero.clone(), self.zero.clone(), self.zero.clone(), self.one.clone()];
            }
            self.detect_ir_free(&plan);
            let started = self.timer();
            self.run_program(&plan, false);
            if let Some(started) = started {
                self.profile.recurrence += started.elapsed();
            }
            if let Some(target) = self.gram_top {
                // Auxiliary expansion executor: coefficients only.
                let outcome = if self.gram_top_active { self.expand_top(target) } else { Ok(()) };
                statuses.push(match outcome {
                    Ok(()) => LaneStatus::Success,
                    Err(error) => {
                        self.lane_conditions[row] = Some(0.0);
                        LaneStatus::Unstable(error)
                    }
                });
                continue;
            }
            if self.coefficients_only && self.top_outputs_finite(&plan) {
                // The gather and output would give finite values, hence
                // `Success`: the status is the one the full lane reports, and
                // the caller reads only the coefficients.
                statuses.push(LaneStatus::Success);
                continue;
            }
            let started = self.timer();
            match self.plan.jet {
                None => self.gather(&plan),
                // Without moment recursions only the requested order is used.
                Some((_, k_len)) if self.plan.moment_program.is_empty() => {
                    self.gather_order(&plan, k_len - 1)
                }
                Some((_, k_len)) => {
                    for order in 0..k_len {
                        self.gather_order(&plan, order);
                    }
                }
            }
            if let Some(started) = started {
                self.profile.gather += started.elapsed();
            }
            let started = self.timer();
            self.run_program(&plan, true);
            if let Some(started) = started {
                self.profile.recurrence += started.elapsed();
            }
            let started = self.timer();
            let mut finite = true;
            let output_order = plan.jet.map_or(0, |(_, k)| k - 1);
            let source = if output_order == 0 {
                &self.values
            } else {
                &self.values_jet[output_order - 1]
            };
            let constants = &self.native_constants;
            // The native normalization is purely imaginary, i pi^(2+h).
            let p = &constants.normalization.im;
            let lane_out = &mut values[lane_values..lane_values + 4 * basis_len];
            for (index, flags) in self.output_flags.iter().enumerate() {
                let v = &source[plan.output[index] as usize];
                let uv = if flags.uv_zero { self.zero.clone() } else { v[0].clone() };
                let mut out: L4<T> = match convention {
                    Convention::OneLoop => [uv, v[1].clone(), v[2].clone(), v[3].clone()],
                    Convention::MadLoop => {
                        let mut out = [uv, v[1].clone(), v[2].clone(), v[3].clone()];
                        if flags.odd {
                            for slot in &mut out {
                                *slot = csub(&self.zero, slot);
                            }
                        }
                        out
                    }
                    Convention::IregiNative => {
                        let single_ir = csub(&v[1], &cscale(&v[2], &constants.a));
                        let finite_part = cadd(
                            &csub(&v[3], &cscale(&cadd(&uv, &v[1]), &constants.a)),
                            &cscale(&v[2], &constants.a_half_squared_minus_b),
                        );
                        [uv, single_ir, v[2].clone(), finite_part]
                            .map(|z| C::new(-(z.im * p), z.re * p))
                    }
                };
                if let Some(weight) = &flags.weight {
                    for slot in &mut out {
                        *slot = cscale(slot, weight);
                    }
                }
                for (slot, entry) in out.into_iter().enumerate() {
                    finite &= entry.re.is_finite() && entry.im.is_finite();
                    lane_out[4 * index + slot] = entry;
                }
            }
            if let Some(started) = started {
                self.profile.output += started.elapsed();
            }
            statuses.push(if finite {
                LaneStatus::Success
            } else {
                // Overflow or a vanishing divisor at this precision: no
                // conditioning estimate, so report zero (a rescue starts high).
                self.lane_conditions[row] = Some(0.0);
                LaneStatus::Unstable("nonfinite direct-engine coefficient".into())
            });
        }
        let rows = batch.rows;
        let _ = cnorm2::<T>;
        Ok(BatchResult {
            basis: self.base_plan.basis.clone(),
            components: 1,
            metadata: ResultMetadata::new(convention, tensor_weights, None),
            values,
            statuses,
            fallback_used: filled(rows, false),
            analytic_used: filled(rows, false),
            primal_envelopes: std::iter::repeat_with(|| None).take(rows).collect(),
            primal_envelope_errors: std::iter::repeat_with(|| None).take(rows).collect(),
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        })
    }
}

/// Subsets with fewer coefficient terms than this are computed even when
/// shared: a lookup builds, hashes and compares a key of about 50 words and
/// reads the stored values from memory, which costs about as many cycles as
/// recomputing a triangle (59 terms at rank 3, 108 at rank 4); boxes (287
/// terms at rank 4, 543 at rank 5) and pentagons (3441) are shared.
const SHARED_MIN_TERMS: u32 = 256;

/// Write the leading `count` shared binary64 values into `out` (exact in
/// the executor's domain); false if fewer were stored.
fn write_shared<T: NativeFloat>(out: &mut [L4<T>], prototype: &T, count: usize, values: &[[f64; 8]]) -> bool {
    if values.len() < count {
        return false;
    }
    for (slot_values, v) in out[..count].iter_mut().zip(values) {
        *slot_values = std::array::from_fn(|slot| C::new(exact_f64(prototype, v[2 * slot]), exact_f64(prototype, v[2 * slot + 1])));
    }
    true
}

/// Key of a double-double scalar master: family, arity and the limbs of
/// every argument bit for bit (`false`: `T` is not double-double).
fn dd_master_key<T: NativeFloat>(key: &mut Vec<u64>, family: usize, arity: usize, input: &[C<T>]) -> bool {
    use symbolica::domains::float::DoubleFloat;
    key.clear();
    key.extend([7, family as u64, arity as u64]);
    for z in input {
        let Some(z) = (z as &dyn std::any::Any).downcast_ref::<C<DoubleFloat>>() else { return false };
        let (re, im) = (z.re.into_inner(), z.im.into_inner());
        key.extend([re.hi().to_bits(), re.lo().to_bits(), im.hi().to_bits(), im.lo().to_bits()]);
    }
    true
}

/// A stored double-double master into the kernels' output slots.
fn write_dd_master<T: NativeFloat>(output: &mut [C<T>], stored: &[C<symbolica::domains::float::DoubleFloat>; 3]) -> bool {
    for (slot, value) in output.iter_mut().zip(stored) {
        let Some(slot) = (slot as &mut dyn std::any::Any).downcast_mut::<C<symbolica::domains::float::DoubleFloat>>() else {
            return false;
        };
        *slot = value.clone();
    }
    true
}

/// The kernels' output as a double-double master to store.
fn read_dd_master<T: NativeFloat>(output: &[C<T>]) -> Option<[C<symbolica::domains::float::DoubleFloat>; 3]> {
    let read = |k: usize| (&output[k] as &dyn std::any::Any).downcast_ref::<C<symbolica::domains::float::DoubleFloat>>().cloned();
    Some([read(0)?, read(1)?, read(2)?])
}

/// The binary64 value `x` exactly in the executor's number type.
fn exact_f64<T: NativeFloat>(prototype: &T, x: f64) -> T {
    match (&x as &dyn std::any::Any).downcast_ref::<T>() {
        Some(value) => value.clone(),
        None => symbolica::domains::float::Float::with_val(53, x)
            .try_to_rational()
            .map_or_else(|| prototype.zero(), |exact| prototype.from_rational(&exact)),
    }
}

fn multinomial(e: [u16; 4]) -> u64 {
    // Product of binomials in u128 (exact; fits u64 up to rank 32).
    let binomial = |n: u128, k: u128| (0..k).fold(1u128, |acc, i| acc * (n - i) / (i + 1));
    let mut total = 0u128;
    let mut value = 1u128;
    for &x in &e {
        total += u128::from(x);
        value *= binomial(total, u128::from(x));
    }
    u64::try_from(value).expect("multinomial beyond u64 (rank above 32)")
}
