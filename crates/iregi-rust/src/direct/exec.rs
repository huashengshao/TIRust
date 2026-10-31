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
        C, L4, cadd, cfma, cinv, cmul, cnorm2, cscale, csub, czero, div_dimension, lfma,
        lfma_slots, lzero, mul_dimension, solve_in_place,
    },
    plan::{DirectPlanData, NO_GEOMETRY, PostOp, SubsetKind},
};
use crate::{
    BatchResult, Convention, KinematicsBatch, LaneStatus, ReductionError, TensorWeights,
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
    Singular(u8, String),
    /// A relation or lightlike assumption of the current variant does not
    /// hold on this lane.
    Relation(String),
    /// A bubble with (numerically) lightlike or vanishing momentum.
    Lightlike(u8),
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
    gram_top_active: bool,
    /// Plan variant of the current lane.
    plan: Arc<DirectPlanData>,
    base_plan: Arc<DirectPlanData>,
    /// Variants keyed by their minimal degenerate subsets.
    variants: HashMap<(Vec<u8>, Vec<u8>), (Arc<DirectPlanData>, Arc<Vec<T>>)>,
    lightlike_profile: Vec<u8>,
    base_constants: Arc<Vec<T>>,
    relation_profile: Vec<u8>,
    prototype: T,
    zero: C<T>,
    one: C<T>,
    constants: Arc<Vec<T>>,
    evaluators: [Option<MasterEvaluator<T>>; 4],
    native: [Option<oneloop::NativeEvaluator<T>>; 4],
    native_input: Vec<C<T>>,
    native_output: Vec<C<T>>,
    master_inputs: [Vec<C<T>>; 4],
    /// Per lane, per family, per master: four Laurent slots.
    master_values: Vec<L4<T>>,
    master_errors: Vec<Option<String>>,
    master_offsets: [usize; 4],
    master_count: usize,
    values: Vec<L4<T>>,
    geometry: Vec<C<T>>,
    distances: Vec<C<T>>,
    shifts: Vec<[C<T>; 4]>,
    masses: Vec<C<T>>,
    /// `g_lambda` blocks, `lambda * stride + monomial`.
    g_buffer: Vec<L4<T>>,
    sparse_products: Vec<(u32, C<T>)>,
    product_ranges: Vec<(u32, u32)>,
    dense: Vec<C<T>>,
    dense_mark: Vec<bool>,
    touched: Vec<u32>,
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
    rinv_memo: HashMap<u8, Vec<C<T>>>,
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
    /// `8 * 2^-precision`: a squared distance below this multiple of its
    /// sum of squared components is zero within its own rounding.
    rounding: T,
    profiling: bool,
    profile: DirectProfile,
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
            gram_top_active: false,
            values: vec![lzero(&zero); plan.value_count],
            geometry: vec![zero.clone(); plan.geometry_count.max(1)],
            distances: vec![zero.clone(); plan.settings.propagators * plan.settings.propagators],
            shifts: vec![[zero.clone(), zero.clone(), zero.clone(), zero.clone()]; plan.settings.propagators],
            masses: vec![zero.clone(); plan.settings.propagators],
            g_buffer: vec![lzero(&zero); gather_degree * (usize::from(plan.settings.rank) / 2 + 1)],
            sparse_products: Vec::new(),
            product_ranges: Vec::new(),
            dense: vec![zero.clone(); gather_degree],
            dense_mark: vec![false; gather_degree],
            touched: Vec::new(),
            qexp_constants: plan
                .monomials
                .qexp
                .iter()
                .map(|terms| terms.iter().map(|(_, _, c)| prototype.from_rational(c)).collect())
                .collect(),
            // Coefficient ranks may exceed the tensor rank (boosted pinches).
            binomials: (0..=usize::from(plan.settings.rank)
                .max(plan.subsets.iter().map(|subset| subset.coefficient_rank.max(0) as usize).max().unwrap_or(0)))
                .map(|j| {
                    (0..=j)
                        .map(|k| {
                            let value: u64 = ((k as u64 + 1)..=(j as u64)).product::<u64>()
                                / (1..=(j - k) as u64).product::<u64>();
                            prototype.from_usize(value as usize)
                        })
                        .collect()
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
            rounding: {
                let half = prototype.one() / &prototype.from_usize(2);
                (0..prototype.get_precision()).fold(prototype.from_usize(8), |acc, _| acc * &half)
            },
            profiling: false,
            profile: DirectProfile::default(),
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

    pub fn last_conditions(&self) -> &[Option<f64>] {
        &self.lane_conditions
    }

    fn timer(&self) -> Option<Instant> {
        self.profiling.then(Instant::now)
    }

    fn read_lane(&mut self, batch: &KinematicsBatch<T>, row: usize) -> Result<(), String> {
        let n = self.plan.settings.propagators;
        let width = 5 * n + 1;
        let values = &batch.values[row * width..(row + 1) * width];
        for (i, shift) in self.shifts.iter_mut().enumerate() {
            for a in 0..4 {
                shift[a] = values[4 * i + a].clone();
            }
        }
        for i in 0..n {
            self.masses[i] = values[4 * n + i].clone();
        }
        if values.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
            return Err("nonfinite kinematic input".into());
        }
        let mu = &values[5 * n];
        if !mu.im.is_zero() || mu.re <= mu.re.zero() {
            return Err("mu must be positive and real".into());
        }
        if self.masses.iter().any(|mass| mass.im > mass.im.zero()) {
            return Err("squared masses require nonpositive imaginary parts".into());
        }
        for i in 0..n {
            for j in 0..n {
                let mut value = self.zero.clone();
                let mut size = self.prototype.zero();
                for a in 0..4 {
                    let delta = csub(&self.shifts[i][a], &self.shifts[j][a]);
                    let square = cmul(&delta, &delta);
                    size += cnorm2(&square).sqrt();
                    value = if a == 0 {
                        cadd(&value, &square)
                    } else {
                        csub(&value, &square)
                    };
                }
                // A (numerically lightlike) invariant within its own rounding
                // error of zero is zero: replacing it perturbs the inputs by
                // no more than their working-precision representation does,
                // and it keeps scalar masters on their exact massless-leg
                // branches (the general ones cancel like m^2/p^2 there).
                if cnorm2(&value).sqrt() <= size * &self.rounding {
                    value = self.zero.clone();
                }
                self.distances[i * n + j] = value;
            }
        }
        Ok(())
    }

    fn mu_squared(&self, batch: &KinematicsBatch<T>, row: usize) -> C<T> {
        let n = self.plan.settings.propagators;
        let mu = &batch.values[row * (5 * n + 1) + 5 * n];
        cmul(mu, mu)
    }

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
    fn native_masters(&mut self, mu2: &C<T>) -> Result<(), String> {
        let plan = self.plan.clone();
        let n = plan.settings.propagators;
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
                evaluator.evaluate(&input[..arity + 1], &mut self.native_output)?;
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

    /// Evaluate every external subset of this lane in double-double and
    /// inject its rounded coefficients.
    fn inject_external(&mut self, mu: f64) -> Result<(), String> {
        use symbolica::domains::float::{DoubleFloat, RealLike};
        let plan = self.plan.clone();
        let external = self.external.clone();
        let mut partial = self.partial.take().ok_or("no partial rescue")?;
        let outcome = (|| -> Result<(), String> {
            for &id in &external {
                let subset = &plan.subsets[id];
                let size = subset.members.len();
                let rank = subset.coefficient_rank.max(0) as u16;
                let executor = match partial.executors.entry((size, rank)) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let small = DirectPlanData::new(crate::ReductionSettings::new(size, rank))
                            .map_err(|e| e.to_string())?;
                        let mut executor = DirectExecutor::<DoubleFloat>::new(Arc::new(small), DoubleFloat::from(0.0), DirectMasters::Native)
                            .map_err(|e| e.to_string())?;
                        executor.set_stability_threshold(DoubleFloat::from(partial.threshold));
                        entry.insert(executor)
                    }
                };
                let mut values = Vec::with_capacity(5 * size + 1);
                for &member in &subset.members {
                    for a in 0..4 {
                        let z = &self.shifts[member][a];
                        values.push(Complex::new(DoubleFloat::from(z.re.to_f64()), DoubleFloat::from(z.im.to_f64())));
                    }
                }
                for &member in &subset.members {
                    let z = &self.masses[member];
                    values.push(Complex::new(DoubleFloat::from(z.re.to_f64()), DoubleFloat::from(z.im.to_f64())));
                }
                values.push(Complex::new(DoubleFloat::from(mu), DoubleFloat::from(0.0)));
                let result = executor
                    .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values })
                    .map_err(|e| e.to_string())?;
                if result.statuses[0] != LaneStatus::Success {
                    return Err(format!("partial rescue of {:?}: {:?}", subset.members, result.statuses[0]));
                }
                let coefficients = executor.top_coefficients().ok_or("partial rescue: no PV coefficients")?;
                if coefficients.len() != plan.coefficient_counts[id] as usize {
                    return Err("partial rescue: coefficient layouts differ".into());
                }
                let start = subset.master.ok_or("partial rescue: no master")? as usize;
                for (k, slots) in coefficients.iter().enumerate() {
                    self.values[start + k] = std::array::from_fn(|slot| {
                        let z = &slots[slot];
                        C::new(exact_f64(&self.prototype, z.re.to_f64()), exact_f64(&self.prototype, z.im.to_f64()))
                    });
                }
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
    fn relation_support(&self, mask: u8) -> Result<u8, String> {
        let members: Vec<usize> = (0..8).filter(|&i| mask & (1 << i) != 0).collect();
        let weights = self.affine_relation(&members)?;
        let largest = weights
            .iter()
            .map(cnorm2)
            .fold(self.prototype.zero(), |a, b| if b > a { b } else { a });
        let cut = largest * &self.binary_tolerance(80);
        let mut support = 0u8;
        for (weight, &member) in weights.iter().zip(&members) {
            if cnorm2(weight) > cut {
                support |= 1 << member;
            }
        }
        let support_members: Vec<usize> = (0..8).filter(|&i| support & (1 << i) != 0).collect();
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
    fn geometry(&mut self) -> Result<(), GeometryFailure> {
        self.failed_condition = None;
        self.external.clear();
        self.gram_external.clear();
        self.gram_top_active = false;
        let plan = self.plan.clone();
        let n = plan.settings.propagators;
        let threshold = self.stability_threshold.clone();
        let pentagon_calibration = self.binary_tolerance(7);
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
                        let lightlike = cnorm2(p2).sqrt() <= scale * &self.binary_tolerance(40);
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
                        if self.gram_top.is_some() && size == 3 && Some(subset.mask) == top_mask {
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
                            if gram && size == 3 {
                                // Small-Gram expansion, injected in the lane loop.
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
                SubsetKind::Pentagon => {
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
                }
                SubsetKind::Relation(_) => {}
                SubsetKind::Hexagon => {
                    // Thin identity sum_i w_i D_i = 1 (column 0) or q^mu
                    // (column mu+1) over six members: the 5x5 system of their
                    // four momentum components and f row. A seven-point lane
                    // drops the member whose remaining six give the best
                    // conditioned system (its weight is zero).
                    let weights = subset.hexagon.unwrap() as usize;
                    let size = members.len();
                    let half = self.prototype.one() / &self.prototype.from_usize(2);
                    let drops: Vec<usize> = if size == 6 { vec![usize::MAX] } else { (0..size).collect() };
                    let mut best: Option<(T, Vec<C<T>>, usize)> = None;
                    for drop in drops {
                        let chosen: Vec<usize> = (0..size).filter(|&k| k != drop).collect();
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
                    let chosen: Vec<usize> = (0..size).filter(|&k| k != drop).collect();
                    for column in 0..5 {
                        let mut sum = self.zero.clone();
                        for i in 0..5 {
                            let value = solution[i * 5 + column].clone();
                            sum = cadd(&sum, &value);
                            self.geometry[weights + size * column + chosen[i + 1]] = value;
                        }
                        self.geometry[weights + size * column + chosen[0]] = csub(&self.zero, &sum);
                        if drop != usize::MAX {
                            self.geometry[weights + size * column + drop] = self.zero.clone();
                        }
                    }
                }
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
    fn lightlike_coefficients(&mut self, mu2: &C<T>) -> Result<(), String> {
        use oneloop::rust_codegen::runtime::primitives::log;
        let plan = self.plan.clone();
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
            let mut l_values = vec![self.zero.clone(); max_k + 1];
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
                let mut m_values = vec![self.zero.clone(); max_k + 2];
                let inverse_a = cinv(&a);
                let ratio = cmul(&b, &inverse_a);
                if cnorm2(&ratio) <= self.prototype.one() / &self.prototype.from_usize(4) {
                    let minus_ratio = csub(&self.zero, &ratio);
                    let tolerance = self.binary_tolerance(60);
                    for (k, value) in m_values.iter_mut().enumerate() {
                        let mut sum = self.zero.clone();
                        let mut power = self.one.clone();
                        for m in 0..200 {
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
                    let mut a_powers = vec![self.one.clone(); usize::from(lambda) + 1];
                    let mut b_powers = vec![self.one.clone(); usize::from(lambda) + 1];
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
    fn cayley_inverse(&mut self, mask: u8) -> Result<Vec<C<T>>, String> {
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
        let mask: u8 = (0..n).filter(|&i| power(powers, i) > 0).fold(0, |m, i| m | (1 << i));
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
                let unit: u64 = (0..8).filter(|&i| mask & (1 << i) != 0).map(|i| 1u64 << (4 * i)).sum();
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
    fn run_program_jets(&mut self, program_is_moments: bool) {
        let plan = self.plan.clone();
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
    fn gather_order(&mut self, order: usize) {
        if order == 0 {
            self.gather();
        } else {
            std::mem::swap(&mut self.values, &mut self.values_jet[order - 1]);
            self.gather();
            std::mem::swap(&mut self.values, &mut self.values_jet[order - 1]);
        }
    }

    fn run_program(&mut self, program_is_moments: bool) {
        match (self.plan.jet, self.ir_free) {
            (Some((_, 2)), true) => self.run_program_jet::<false, 2>(program_is_moments),
            (Some((_, 2)), false) => self.run_program_jet::<true, 2>(program_is_moments),
            (Some((_, 3)), true) => self.run_program_jet::<false, 3>(program_is_moments),
            (Some((_, 3)), false) => self.run_program_jet::<true, 3>(program_is_moments),
            (Some((_, 4)), true) => self.run_program_jet::<false, 4>(program_is_moments),
            (Some((_, 4)), false) => self.run_program_jet::<true, 4>(program_is_moments),
            (Some((_, 5)), true) => self.run_program_jet::<false, 5>(program_is_moments),
            (Some((_, 5)), false) => self.run_program_jet::<true, 5>(program_is_moments),
            (Some(_), _) => self.run_program_jets(program_is_moments),
            (None, true) => self.run_program_ordinary::<false>(program_is_moments),
            (None, false) => self.run_program_ordinary::<true>(program_is_moments),
        }
    }

    /// Whether every master (and closed-form) value of this lane has
    /// vanishing IR slots. Recurrences are linear and never move UV or
    /// finite parts into IR slots, so those slots then stay exactly zero.
    fn detect_ir_free(&mut self) {
        let zero = |z: &C<T>| z.re.is_zero() && z.im.is_zero();
        let plan = self.plan.clone();
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
    fn run_program_jet<const FULL: bool, const K: usize>(&mut self, program_is_moments: bool) {
        let plan = self.plan.clone();
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

    fn run_program_ordinary<const FULL: bool>(&mut self, program_is_moments: bool) {
        let plan = self.plan.clone();
        let program = if program_is_moments {
            &plan.moment_program
        } else {
            &plan.coefficient_program
        };
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
        for (index, instruction) in program.iter().enumerate() {
            if !skip.is_empty() && skip.iter().any(|&(a, b)| a <= index as u32 && (index as u32) < b) {
                continue;
            }
            let mut accumulator = lzero(&self.zero);
            for term in &plan.terms[instruction.terms.0 as usize..instruction.terms.1 as usize] {
                let constant = &self.constants[term.constant as usize];
                let factor = if term.geometry == NO_GEOMETRY {
                    C::new(constant.clone(), self.prototype.zero())
                } else {
                    let geometry = &self.geometry[term.geometry as usize];
                    if geometry.re.is_zero() && geometry.im.is_zero() {
                        continue;
                    }
                    if term.constant == 0 {
                        geometry.clone()
                    } else {
                        cscale(geometry, constant)
                    }
                };
                lfma_slots::<T, FULL>(&mut accumulator, &factor, &self.values[term.src as usize]);
            }
            match instruction.post {
                PostOp::None => {}
                PostOp::DivideDimension { inverse } => {
                    div_dimension(&mut accumulator, &self.constants[inverse as usize]);
                }
            }
            self.values[instruction.dst as usize] = accumulator;
        }
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
    fn gather(&mut self) {
        if self.ir_free {
            self.gather_with::<false>();
        } else {
            self.gather_with::<true>();
        }
    }

    fn gather_with<const FULL: bool>(&mut self) {
        let plan = self.plan.clone();
        let table = &plan.monomials;
        let stride = table.block(usize::from(plan.settings.rank)).end;
        for gather in &plan.gathers {
            let subset = &plan.subsets[gather.subset];
            let geometry = subset.pv.unwrap();
            let rank = usize::from(gather.moment_rank);
            let momenta = subset.members.len() - 1;
            // 1. Products as sparse (local monomial, value) lists per kappa.
            self.sparse_products.clear();
            self.product_ranges.clear();
            for kappa in &gather.kappas {
                let start = self.sparse_products.len() as u32;
                if kappa.degree == 0 {
                    self.sparse_products.push((0, self.one.clone()));
                    self.product_ranges.push((start, start + 1));
                    continue;
                }
                let block = table.block(usize::from(kappa.degree));
                let parent = &gather.kappas[kappa.parent as usize];
                let parent_start = table.block(usize::from(parent.degree)).start;
                let (from, to) = self.product_ranges[kappa.parent as usize];
                let inverse = self.prototype.one() / &self.prototype.from_usize(usize::from(kappa.divisor));
                let p = geometry.p as usize + 4 * usize::from(kappa.momentum);
                self.touched.clear();
                for entry in from..to {
                    let (local, value) = self.sparse_products[entry as usize].clone();
                    let value = cscale(&value, &inverse);
                    let monomial = parent_start + local as usize;
                    for a in 0..4 {
                        let component = &self.geometry[p + a];
                        if component.re.is_zero() && component.im.is_zero() {
                            continue;
                        }
                        let target = table.multiply[monomial][a] as usize - block.start;
                        if !self.dense_mark[target] {
                            self.dense_mark[target] = true;
                            self.dense[target] = self.zero.clone();
                            self.touched.push(target as u32);
                        }
                        let update = cmul(&value, component);
                        self.dense[target] = cadd(&self.dense[target], &update);
                    }
                }
                for &target in &self.touched {
                    self.dense_mark[target as usize] = false;
                    let value = self.dense[target as usize].clone();
                    if !(value.re.is_zero() && value.im.is_zero()) {
                        self.sparse_products.push((target, value));
                    }
                }
                let end = self.sparse_products.len() as u32;
                self.product_ranges.push((start, end));
            }
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
                    let (from, to) = self.product_ranges[k];
                    for (local, product) in &self.sparse_products[from as usize..to as usize] {
                        lfma_slots::<T, FULL>(&mut self.g_buffer[block_start + *local as usize], product, value);
                    }
                }
            }
            // 3. Untranslated moments.
            let moment_base = gather.moment_base as usize;
            for e in 0..table.block(rank).end {
                let mut total = lzero(&self.zero);
                for (term, &(src, lambda, _)) in table.qexp[e].iter().enumerate() {
                    if momenta == 0 && src != 0 {
                        continue;
                    }
                    let value = &self.g_buffer[usize::from(lambda) * stride + src as usize];
                    let constant = &self.qexp_constants[e][term];
                    for slot in if FULL { 0..4 } else { 0..1 } {
                        total[slot] = cadd(&total[slot], &cscale(&value[slot], constant));
                    }
                    if !FULL {
                        total[3] = cadd(&total[3], &cscale(&value[3], constant));
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
        }
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let n = self.plan.settings.propagators;
        if batch.components != 1 {
            return Err(ReductionError::Unsupported(
                "the direct engine evaluates ordinary numbers; use the generic workspace for Dual components".into(),
            ));
        }
        let width = 5 * n + 1;
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
        let base_plan = self.base_plan.clone();
        let basis_len = base_plan.basis.len();
        let mut values = Vec::with_capacity(batch.rows * basis_len * 4);
        let mut statuses = Vec::with_capacity(batch.rows);
        self.lane_conditions.clear();
        self.lane_conditions.resize(batch.rows, None);
        let settings = &base_plan.settings;
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
            loop {
                match self.geometry() {
                    Ok(()) => break,
                    Err(GeometryFailure::Relation(error)) => {
                        if self.relation_profile.is_empty() && self.lightlike_profile.is_empty() {
                            failure = Some(error);
                            break;
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
                            break;
                        }
                        self.lightlike_profile.push(mask);
                        self.lightlike_profile.sort_unstable();
                        self.select_variant()?;
                    }
                    Err(GeometryFailure::Unstable(error)) => {
                        failure = Some(error);
                        break;
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
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(error) = failure {
                self.lane_conditions[row] = self.failed_condition;
                statuses.push(LaneStatus::Unstable(error));
                continue;
            }
            let plan = self.plan.clone();
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
                if let Err(error) = self.native_masters(&mu2) {
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
                let n = self.plan.settings.propagators;
                let mu = batch.values[row * (5 * n + 1) + 5 * n].re.to_f64();
                if let Err(error) = self.inject_external(mu) {
                    self.lane_conditions[row] = self.failed_condition;
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
            }
            if !self.gram_external.is_empty() {
                let n = self.plan.settings.propagators;
                let mu = batch.values[row * (5 * n + 1) + 5 * n].clone();
                if let Err(error) = self.inject_gram(&mu) {
                    self.lane_conditions[row] = self.failed_condition;
                    statuses.push(LaneStatus::Unstable(error));
                    continue;
                }
            }
            if let Some(unit) = plan.unit_finite {
                self.values[unit as usize] = [self.zero.clone(), self.zero.clone(), self.zero.clone(), self.one.clone()];
            }
            self.detect_ir_free();
            let started = self.timer();
            self.run_program(false);
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
            let started = self.timer();
            match self.plan.jet {
                None => self.gather(),
                // Without moment recursions only the requested order is used.
                Some((_, k_len)) if self.plan.moment_program.is_empty() => {
                    self.gather_order(k_len - 1)
                }
                Some((_, k_len)) => {
                    for order in 0..k_len {
                        self.gather_order(order);
                    }
                }
            }
            if let Some(started) = started {
                self.profile.gather += started.elapsed();
            }
            let started = self.timer();
            self.run_program(true);
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
                let mut out: L4<T> = match settings.convention {
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
            basis: base_plan.basis.clone(),
            components: 1,
            metadata: ResultMetadata::new(settings.convention, settings.tensor_weights, None),
            values,
            statuses,
            fallback_used: vec![false; rows],
            analytic_used: vec![false; rows],
            primal_envelopes: vec![None; rows],
            primal_envelope_errors: vec![None; rows],
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        })
    }
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
    let factorial = |n: u16| (1..=u64::from(n)).product::<u64>();
    let rank: u16 = e.iter().sum();
    factorial(rank) / e.iter().map(|&x| factorial(x)).product::<u64>()
}
