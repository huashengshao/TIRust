//! Embedder mode of the facade: the direct engine at the workspace's own
//! precision, with no hidden check, no hidden escalation and never the
//! symbolic engine.
//!
//! A host that runs its own stability checks (a re-evaluation in another
//! frame, a comparison with another backend) needs the reduction to do exactly
//! what it was asked: [`TensorWorkspace::new_embedded`] builds such a
//! workspace for every number type the direct engine serves (binary64,
//! double-double, MPFR). Every lane first runs through the direct engine's
//! conditioning gates. Then, in this order and only as configured
//! ([`EmbedderOptions`]):
//!
//! 1. *method choice* (on by default): lanes the gates refuse are evaluated
//!    again with COLLIER's a priori per-box choice (boxes and triangles whose
//!    Passarino-Veltman estimate exceeds the request, or that their gate
//!    refuses, by the small-Gram expansions or the alternative PV reduction).
//!    Unchecked here: the lanes are flagged ([`TensorWorkspace::last_lane_routes`])
//!    for the host's own check;
//! 2. *partial rescue* (binary64 only, off by default): boxes and triangles a
//!    gate refuses inside a binary64 lane are computed in double-double;
//! 3. *precision tiers* (binary64 only, off by default): lanes still refused
//!    on conditioning are evaluated again by the direct engine at 106, 256 and
//!    512 bits (conditioning gates only, no rotation test) and rounded once.
//!
//! There is no loop-rotation test, no rescue-first routing of refused lanes
//! and no symbolic fallback: a lane nothing accepted stays `Unstable` with the
//! reason in [`TensorWorkspace::last_diagnostics`], a scalar-master failure
//! stays `MasterFailure`. The facade's ordinary binary64 workspace (with its
//! rotation test, retries, rescue and fallback) is unchanged.

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{
    domains::float::{Complex, DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

use super::direct_hybrid::{
    DirectExecutionStats, DirectRescueTier, RESCUE_MARGIN, add_direct_profile, first_tier_with_margin, masters_for,
    prepare_tier, run_tier,
};
use crate::{
    BatchResult, DirectReductionPlan, DirectWorkspace, ExecutionProfile, KinematicsBatch, LaneStatus, ReductionError,
    types::filled,
};

/// Configuration of an embedder-mode workspace ([`TensorWorkspace::new_embedded`]).
///
/// [`TensorWorkspace::new_embedded`]: super::TensorWorkspace::new_embedded
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbedderOptions {
    /// Gate-refused lanes get COLLIER's a priori method choice (unchecked,
    /// flagged per lane). Default `true`.
    pub method_choice: bool,
    /// Binary64 workspaces: boxes and triangles refused inside a lane are
    /// computed in double-double. Default `false`. Their double-double gate
    /// is fixed at 2^-66 (40 bits of headroom);
    /// [`Self::double_double_headroom_bits`] does not change it.
    pub partial_rescue: bool,
    /// Binary64 workspaces: lanes still refused on conditioning are evaluated
    /// again by the direct engine at 106, 256 and 512 bits. Default `false`.
    pub precision_tiers: bool,
    /// Minimal accepted relative determinant of the first pass (raised to the
    /// recursion depth); `None` keeps the direct engine's default (1e-12).
    pub gate: Option<f64>,
    /// Gate of the method-choice pass; `None`: 1e-15, the gate the binary64
    /// facade's choice runs with (its rows are checked by the host).
    pub method_choice_gate: Option<f64>,
    /// Scalar-master implementation (see [`MasterExecutionPolicy`]).
    pub master_policy: MasterExecutionPolicy,
    /// Precision tiers: bits of margin by which a lane's binary64
    /// conditioning estimate must clear a tier's limit for the lane to start
    /// at that tier (default 8). A lane whose estimate falls inside the
    /// margin skips the tier, on the expectation that the tier would refuse
    /// it. Every tier still applies its own gate (accuracy is unchanged);
    /// with 0, a lane starts at the lowest tier whose limit admits its
    /// estimate and moves up only if that tier refuses it. On amplitudes
    /// with many near-limit lanes (massless QCD), 0 replaces most 256-bit
    /// evaluations by double-double ones.
    pub tier_margin_bits: u32,
    /// Precision tiers: bits the double-double tier keeps between its working
    /// precision (106 bits) and the conditioning it accepts (default 40: it
    /// accepts estimates down to 2^-66, amplifying its rounding by at most
    /// 2^-40). A smaller headroom accepts worse-conditioned lanes in
    /// double-double instead of 256 bits, with a correspondingly larger bound
    /// (26: down to 2^-80, at most 2^-26). The MPFR tiers keep 40 bits, and
    /// [`Self::partial_rescue`] keeps its 2^-66 subset gate whatever this
    /// headroom. Values of 106 and above are clamped to 105: the tier then
    /// accepts only estimates of at least 2^-1, so almost every gate-refused
    /// lane goes on to 256 bits.
    pub double_double_headroom_bits: u32,
}

impl Default for EmbedderOptions {
    fn default() -> Self {
        Self {
            method_choice: true,
            partial_rescue: false,
            precision_tiers: false,
            gate: None,
            method_choice_gate: None,
            master_policy: MasterExecutionPolicy::Default,
            tier_margin_bits: RESCUE_MARGIN,
            double_double_headroom_bits: TIER_HEADROOM,
        }
    }
}

/// How the most recent call of an embedder-mode workspace computed a lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRoute {
    /// The first pass: the direct engine with its gates (and, if enabled,
    /// double-double boxes and triangles inside the lane).
    Direct,
    /// The a priori method choice (unchecked: the host checks it).
    MethodChoice,
    /// The direct engine at this many bits (precision tiers), rounded once.
    Tier(u32),
    /// No value: refused, invalid input or a master failure (see the status).
    Refused,
}

// The facade's own constants: the method choice runs with its gate and
// small-Gram boost, the precision tiers on its rescue ladder.
use super::direct_hybrid::{
    GRAM_BOOST, RESCUE_BITS as TIER_BITS, RESCUE_HEADROOM as TIER_HEADROOM,
    ROTATION_GATE as METHOD_CHOICE_GATE, from_binary64,
};

/// A tier accepts conditioning down to `2^-(bits - TIER_HEADROOM)`: its
/// rounding error amplified by at most `2^-40` relative, below binary64's
/// accuracy (no a posteriori check in this mode).
fn tier_threshold(bits: u32) -> f64 {
    2f64.powi(-((bits - TIER_HEADROOM) as i32))
}

#[derive(Clone)]
pub(crate) struct EmbeddedWorkspace<T: NativeFloat> {
    plan: DirectReductionPlan,
    options: EmbedderOptions,
    prototype: T,
    direct: DirectWorkspace<T>,
    choice: Option<Box<DirectWorkspace<T>>>,
    double: Option<Box<DirectWorkspace<DoubleFloat>>>,
    mpfr: [Option<Box<DirectWorkspace<Float>>>; 2],
    routes: Vec<LaneRoute>,
    diagnostics: Vec<Vec<String>>,
    last: DirectExecutionStats,
    profiling: bool,
    last_profile: Option<ExecutionProfile>,
    /// The caller's tensor store (`crate::TensorStore`), for every direct
    /// workspace of this lane pipeline, present and future.
    store: Option<std::sync::Arc<crate::direct::shared::SubsetStore>>,
}

impl<T: NativeFloat> EmbeddedWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(crate) fn new(plan: DirectReductionPlan, prototype: T, options: EmbedderOptions) -> Result<Self, ReductionError> {
        let binary64 = Self::is_binary64(&prototype);
        if !binary64 && (options.partial_rescue || options.precision_tiers) {
            return Err(ReductionError::Unsupported(
                "partial rescue and precision tiers serve binary64 workspaces; a double-double or MPFR workspace runs at its own precision".into(),
            ));
        }
        let mut direct = plan.workspace_with_masters(prototype.clone(), masters_for(options.master_policy))?;
        if let Some(gate) = options.gate {
            direct.set_stability_threshold(from_binary64(&prototype, gate));
        }
        if options.partial_rescue {
            direct.enable_partial_rescue(tier_threshold(TIER_BITS[0]));
        }
        Ok(Self {
            plan,
            options,
            prototype,
            direct,
            choice: None,
            double: None,
            mpfr: [None, None],
            routes: Vec::new(),
            diagnostics: Vec::new(),
            last: DirectExecutionStats::default(),
            profiling: false,
            last_profile: None,
            store: None,
        })
    }

    /// Share through the caller's tensor store (`None`: detach): the direct
    /// pass, the method choice and the double-double tier.
    pub(crate) fn set_shared_subsets(&mut self, store: Option<std::sync::Arc<crate::direct::shared::SubsetStore>>) {
        self.direct.set_shared_subsets(store.clone());
        if let Some(choice) = self.choice.as_mut() {
            choice.set_shared_subsets(store.clone());
        }
        if let Some(double) = self.double.as_mut() {
            double.set_shared_subsets(store.clone());
        }
        self.store = store;
    }

    pub(crate) fn plan(&self) -> &DirectReductionPlan {
        &self.plan
    }

    fn is_binary64(prototype: &T) -> bool {
        T::FIXED_BITS == Some(53) && prototype.get_precision() == 53
    }

    pub(crate) fn options(&self) -> EmbedderOptions {
        self.options
    }

    pub(crate) fn binary_precision(&self) -> u32 {
        self.prototype.get_precision()
    }

    pub(crate) fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }

    pub(crate) fn last_routes(&self) -> &[LaneRoute] {
        &self.routes
    }

    pub(crate) fn last_execution(&self) -> &DirectExecutionStats {
        &self.last
    }

    pub(crate) fn last_profile(&self) -> Option<ExecutionProfile> {
        self.last_profile
    }

    pub(crate) fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        self.direct.set_profiling(enabled);
        if let Some(choice) = self.choice.as_mut() {
            choice.set_profiling(enabled);
        }
    }

    pub(crate) fn reduce_batch(&mut self, batch: &KinematicsBatch<T>) -> Result<BatchResult<T>, ReductionError> {
        let started = self.profiling.then(std::time::Instant::now);
        let mut profile = ExecutionProfile::default();
        let mut result = self.direct.reduce_batch(batch)?;
        result.estimated_errors = filled(batch.rows, None);
        result.best_effort = filled(batch.rows, false);
        add_direct_profile(&mut profile, self.direct.last_profile());
        self.last = DirectExecutionStats {
            input_rows: batch.rows,
            master_rows: self.direct.last_master_rows(),
            ..DirectExecutionStats::default()
        };
        self.routes.clear();
        self.routes.extend(result.statuses.iter().map(|status| {
            if *status == LaneStatus::Success { LaneRoute::Direct } else { LaneRoute::Refused }
        }));
        self.diagnostics.clear();
        self.diagnostics.resize(batch.rows, Vec::new());
        self.last.direct_successes = self.routes.iter().filter(|&&route| route == LaneRoute::Direct).count();
        // Lanes refused on conditioning (a master failure or invalid input is
        // final), with their binary64 conditioning estimate.
        let mut conditions: Vec<Option<f64>> = self.direct.last_conditions().to_vec();
        let mut refused: Vec<usize> = (0..batch.rows)
            .filter(|&lane| matches!(result.statuses[lane], LaneStatus::Unstable(_)) && conditions[lane].is_some())
            .collect();
        for lane in 0..batch.rows {
            if let LaneStatus::Unstable(note) | LaneStatus::MasterFailure(note) = &result.statuses[lane] {
                self.diagnostics[lane].push(format!("direct engine: {note}"));
            }
        }
        if self.options.method_choice && !refused.is_empty() {
            self.method_choice(batch, &refused, &mut result, &mut conditions, &mut profile)?;
            refused.retain(|&lane| result.statuses[lane] != LaneStatus::Success && conditions[lane].is_some());
        }
        if self.options.precision_tiers && !refused.is_empty() {
            self.precision_tiers(batch, &refused, &conditions, &mut result, &mut profile)?;
        }
        if let Some(started) = started {
            profile.total = started.elapsed();
            self.last_profile = Some(profile);
        } else {
            self.last_profile = None;
        }
        Ok(result)
    }

    fn sub_batch(batch: &KinematicsBatch<T>, lanes: &[usize]) -> KinematicsBatch<T> {
        let width = batch.values.len() / batch.rows.max(1);
        let mut values = Vec::with_capacity(lanes.len() * width);
        for &lane in lanes {
            values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
        }
        KinematicsBatch { rows: lanes.len(), components: batch.components, values }
    }

    fn method_choice(
        &mut self,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        result: &mut BatchResult<T>,
        conditions: &mut [Option<f64>],
        profile: &mut ExecutionProfile,
    ) -> Result<(), ReductionError> {
        if self.choice.is_none() {
            let mut workspace = self
                .plan
                .workspace_with_masters(self.prototype.clone(), masters_for(self.options.master_policy))?;
            let gate = from_binary64(&self.prototype, self.options.method_choice_gate.unwrap_or(METHOD_CHOICE_GATE));
            workspace.set_stability_threshold(gate.clone());
            workspace.enable_estimated_method_choice(GRAM_BOOST, gate);
            if self.options.partial_rescue {
                workspace.enable_partial_rescue(tier_threshold(TIER_BITS[0]));
            }
            workspace.set_profiling(self.profiling);
            workspace.set_shared_subsets(self.store.clone());
            self.choice = Some(Box::new(workspace));
        }
        let workspace = self.choice.as_mut().unwrap();
        let chosen = workspace.reduce_batch(&Self::sub_batch(batch, lanes))?;
        self.last.master_rows += workspace.last_master_rows();
        add_direct_profile(profile, workspace.last_profile());
        self.last.expansion_rows += lanes.len();
        let stride = result.basis.len() * 4 * result.components;
        let choice_conditions = workspace.last_conditions();
        for (k, &lane) in lanes.iter().enumerate() {
            result.fallback_used[lane] = true;
            if chosen.statuses[k] == LaneStatus::Success {
                result.values[lane * stride..(lane + 1) * stride]
                    .clone_from_slice(&chosen.values[k * stride..(k + 1) * stride]);
                result.statuses[lane] = LaneStatus::Success;
                self.routes[lane] = LaneRoute::MethodChoice;
                self.last.expansion_successes += 1;
                self.diagnostics[lane].push("a priori method choice: accepted (unchecked)".into());
            } else {
                self.diagnostics[lane].push(format!("a priori method choice: {:?}", chosen.statuses[k]));
                if !matches!(chosen.statuses[k], LaneStatus::Unstable(_)) {
                    conditions[lane] = None;
                } else if let Some(condition) = choice_conditions.get(k).copied().flatten() {
                    // The smaller (worse) estimate decides the first tier.
                    conditions[lane] = Some(conditions[lane].map_or(condition, |c| c.min(condition)));
                }
            }
        }
        Ok(())
    }

    /// The conditioning a precision tier of `bits` accepts (see
    /// [`EmbedderOptions::double_double_headroom_bits`]).
    fn tier_limit(&self, bits: u32) -> f64 {
        if bits == TIER_BITS[0] {
            2f64.powi(-((bits - self.options.double_double_headroom_bits.min(bits - 1)) as i32))
        } else {
            tier_threshold(bits)
        }
    }

    fn precision_tiers(
        &mut self,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        conditions: &[Option<f64>],
        result: &mut BatchResult<T>,
        profile: &mut ExecutionProfile,
    ) -> Result<(), ReductionError> {
        // The first tier admitting the binary64 estimate with margin (an
        // exactly singular binary64 pivot carries a zero estimate: 256 bits).
        let mut pending: Vec<(usize, usize)> = lanes
            .iter()
            .map(|&lane| {
                let condition = conditions[lane].unwrap_or(0.0);
                (lane, first_tier_with_margin(condition, |bits| self.tier_limit(bits), self.options.tier_margin_bits))
            })
            .collect();
        self.last.fallback_rows += pending.len();
        let masters = masters_for(self.options.master_policy);
        for (tier, &bits) in TIER_BITS.iter().enumerate() {
            let current: Vec<usize> =
                pending.iter().filter(|&&(_, start)| start <= tier).map(|&(lane, _)| lane).collect();
            if current.is_empty() {
                continue;
            }
            // Accepted lanes are rounded into `result` by the facade's tier runner.
            let outcomes = if tier == 0 {
                if self.double.is_none() {
                    self.double = Some(prepare_tier::<DoubleFloat>(
                        self.plan.workspace_with_masters::<DoubleFloat>(DoubleFloat::from(0.0), masters),
                        DoubleFloat::from(self.tier_limit(bits)),
                        self.profiling,
                    )?);
                }
                let workspace = self.double.as_mut().unwrap();
                workspace.set_shared_subsets(self.store.clone());
                run_tier::<T, DoubleFloat>(workspace, DoubleFloat::from, batch, &current, &self.prototype, result, profile, &mut self.last)?
            } else {
                let slot = &mut self.mpfr[tier - 1];
                if slot.is_none() {
                    *slot = Some(prepare_tier::<Float>(
                        self.plan.workspace_with_masters::<Float>(Float::new(bits), masters),
                        Float::with_val(bits, tier_threshold(bits)),
                        self.profiling,
                    )?);
                }
                let workspace = slot.as_mut().unwrap();
                run_tier::<T, Float>(workspace, |x| Float::with_val(bits, x), batch, &current, &self.prototype, result, profile, &mut self.last)?
            };
            let mut successes = 0;
            for (&lane, (status, condition)) in current.iter().zip(&outcomes) {
                if *status == LaneStatus::Success {
                    result.statuses[lane] = LaneStatus::Success;
                    result.fallback_used[lane] = true;
                    self.routes[lane] = LaneRoute::Tier(bits);
                    self.diagnostics[lane].push(format!("direct engine at {bits} bits: accepted"));
                    successes += 1;
                    pending.retain(|&(other, _)| other != lane);
                } else {
                    self.diagnostics[lane].push(format!("direct engine at {bits} bits: {status:?}"));
                    if condition.is_none() || tier + 1 == TIER_BITS.len() {
                        pending.retain(|&(other, _)| other != lane);
                    }
                }
            }
            self.last.fallback_successes += successes;
            self.last.precision_rescue.push(DirectRescueTier { bits, rows: current.len(), successes });
        }
        Ok(())
    }
}
