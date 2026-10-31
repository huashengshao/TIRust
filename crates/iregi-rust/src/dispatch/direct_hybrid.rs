//! Binary64 default: the direct numerical engine with per-lane fallback.
//!
//! The direct engine evaluates the validated PV recurrence numerically. With
//! five or more propagators (pentagon Gram recursions, whose rounding a
//! conditioning estimate predicts poorly) every binary64 lane is evaluated a
//! second time with its propagators rotated, the same integral through
//! different Gram systems, as in MadLoop's loop-direction test; a relative
//! difference above the requested accuracy (2^-34 by default) or a refused
//! rotated copy refuses the lane.
//!
//! Lanes the a priori conditioning gate refuses (four or more propagators,
//! equal powers) are evaluated again in binary64 with the gate lowered to
//! `CHECKED_GATE` and kept when their rotated copy agrees two bits better
//! than the tolerance: the a posteriori test replaces the a priori bound,
//! which is pessimistic for most such lanes.
//!
//! With [`PrecisionRescue::Automatic`] (the default) lanes still refused on
//! conditioning are rerun by the same direct engine at higher precision:
//! double-double, then 256 and 512 MPFR bits, starting at the tier the
//! binary64 conditioning estimate calls for; boxes and triangles refused
//! inside a lane are computed in double-double first (partial rescue). The
//! binary64 inputs are lifted exactly, so a tier evaluates the caller's
//! kinematics, and the accepted coefficients are rounded once. With
//! [`PrecisionRescue::Disabled`] these lanes are returned unstable. Lanes no
//! tier accepts (or, rescue disabled, lanes without a conditioning estimate)
//! and scalar-master failures are rerun on the existing binary64 symbolic
//! engine, whose plan and workspace are only prepared when such a lane first
//! occurs. Invalid inputs are rejected by the direct engine itself and are not
//! retried.

use std::sync::{Arc, OnceLock};

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{
    domains::float::{Complex, DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

use super::{
    PrecisionRescue, TensorEngineExecution, TensorReductionPlan, TensorWorkspace,
    TensorWorkspaceOptions,
};
use crate::{
    BatchResult, DirectMasters, DirectReductionPlan, DirectWorkspace, ExecutionProfile,
    InputLayout, KinematicsBatch, LaneStatus, ReductionError, ReductionSettings,
    basis::Monomial,
};

/// Structural direct plan and the lazily prepared fallback plan.
#[derive(Debug)]
pub(crate) struct DirectHybridPlan {
    pub(crate) direct: DirectReductionPlan,
    pub(crate) layout: InputLayout,
    fallback: OnceLock<Result<Arc<TensorReductionPlan>, String>>,
}

/// Settings for which `TensorReductionPlan::new` selects the direct engine.
pub(crate) fn direct_eligible(settings: &ReductionSettings) -> bool {
    DirectReductionPlan::supports(settings)
}

impl DirectHybridPlan {
    pub(crate) fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        let direct = DirectReductionPlan::new(settings)?;
        Ok(Self {
            direct,
            layout: InputLayout {
                shifts: 0..4 * n,
                scalar_products: None,
                masses: 4 * n..5 * n,
                scale: 5 * n,
            },
            fallback: OnceLock::new(),
        })
    }

    pub(crate) fn basis(&self) -> &[Monomial] {
        self.direct.basis()
    }

    /// The existing ordinary plan for the same settings, prepared once.
    pub(crate) fn fallback(&self) -> Result<Arc<TensorReductionPlan>, ReductionError> {
        self.fallback
            .get_or_init(|| {
                TensorReductionPlan::new_without_direct(self.direct.settings().clone())
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            })
            .clone()
            .map_err(ReductionError::Preparation)
    }
}

/// Working precisions of the direct-engine rescue: double-double, then MPFR.
const RESCUE_BITS: [u32; 3] = [106, 256, 512];
/// A tier accepts conditioning down to `2^-(bits - RESCUE_HEADROOM)`, so the
/// estimated amplification of its rounding error stays below `2^-40`
/// (stricter than the binary64 gate of `1e-12` at 53 bits).
const RESCUE_HEADROOM: u32 = 40;
/// Bits added to the binary64 estimate when choosing the first tier.
const RESCUE_MARGIN: u32 = 8;

/// Largest accepted E/A between a binary64 lane and its rotated copy.
const ROTATION_TOLERANCE: f64 = 1.0 / (1u64 << 34) as f64;
/// Binary64 conditioning gate of plans with the loop-rotation test (N >= 5,
/// uniform powers). The test checks every accepted lane a posteriori, so the
/// a priori gate only removes hopeless lanes: on physical gg->Hgg pentagons
/// and gg->gggg hexagons (stability-20261030) the default `1e-12` refused
/// lanes of which most were good to 1e-10, while `1e-15` keeps the
/// accepted tail at <= 1e-9 before the rotation test.
const ROTATION_GATE: f64 = 1e-15;

/// Gate of the checked binary64 retry. On the stability-20261030 points
/// (boxes to heptagons) no lane below 1e-21 passed the rotation check.
const CHECKED_GATE: f64 = 1e-30;
/// Bits by which a lane below the a priori gate must beat the rotation
/// tolerance. Its rotated copy shares the ill-conditioned subsystems, so the
/// difference underestimates the error more often: on stability-20261030
/// such lanes accepted at the plain tolerance reached E/A 6.6 times it, with
/// two bits at most 2.4 times it (the binary64 tail without the retry).
const CHECKED_MARGIN_BITS: i32 = 2;
/// Extra triangle ranks of the small-Gram expansion retry (each adds a
/// correction of order Delta / (Zadj f)_j). On the gate-refused
/// stability-20261030 rank-6 boxes the expansion converged on 9, 155, 239
/// lanes with 4, 6, 8 and on no more with 10-16; every converged lane agreed
/// with 256 bits to E/A <= 3.4e-11.
const GRAM_BOOST: u16 = 8;

/// Whether the binary64 pass runs the loop-rotation test.
fn rotation_tested(settings: &ReductionSettings) -> bool {
    settings.propagators >= 5 && uniform_powers(settings)
}

/// Whether lanes refused by the a priori gate get the checked retry.
fn rotation_checkable(settings: &ReductionSettings) -> bool {
    settings.propagators >= 4 && uniform_powers(settings)
}

fn uniform_powers(settings: &ReductionSettings) -> bool {
    settings.powers.iter().all(|&power| power == settings.powers[0])
}

/// The batch with every lane's propagators rotated by one (shifts and squared
/// masses; the loop momentum is relabelled, the integral is unchanged).
fn rotated<T: NativeFloat>(batch: &KinematicsBatch<T>, n: usize) -> KinematicsBatch<T> {
    let width = 5 * n + 1;
    let mut values = Vec::with_capacity(batch.values.len());
    for lane in 0..batch.rows {
        let row = &batch.values[lane * width..(lane + 1) * width];
        for k in 0..n {
            let from = (k + 1) % n;
            values.extend_from_slice(&row[4 * from..4 * from + 4]);
        }
        for k in 0..n {
            values.push(row[4 * n + (k + 1) % n].clone());
        }
        values.push(row[5 * n].clone());
    }
    KinematicsBatch { rows: batch.rows, components: batch.components, values }
}

/// E/A between two lanes' arrays, moments weighted by mu^-degree.
fn lane_difference<T: NativeFloat>(
    basis: &[Monomial],
    a: &[Complex<T>],
    b: &[Complex<T>],
    mu: f64,
) -> f64 {
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (m, monomial) in basis.iter().enumerate() {
        let weight = mu.powi(monomial.rank() as i32);
        for slot in 0..4 {
            let (x, y) = (&a[4 * m + slot], &b[4 * m + slot]);
            let (xr, xi, yr, yi) = (x.re.to_f64(), x.im.to_f64(), y.re.to_f64(), y.im.to_f64());
            error = error.max((xr - yr).hypot(xi - yi) / weight);
            amplitude = amplitude.max(yr.hypot(yi) / weight);
        }
    }
    if amplitude > 0.0 { error / amplitude } else if error == 0.0 { 0.0 } else { f64::INFINITY }
}

/// Headroom of the double-double tier. Its acceptance is checked a
/// posteriori by the loop-rotation test for N >= 5 (`RESCUE_ROTATION_TOLERANCE`):
/// on physical gg -> ggggg heptagons the conditioning estimate called for 256
/// bits, while the 106-bit direct engine with no gate at all agreed with 256
/// bits to E/A <= 1.7e-23 (30 points).
const RESCUE_DD_HEADROOM: u32 = 20;
/// Largest accepted E/A between a 106-bit lane and its rotated copy, both
/// rounded to binary64 (a few ulps).
const RESCUE_ROTATION_TOLERANCE: f64 = 1.0 / (1u64 << 46) as f64;

fn rescue_threshold(bits: u32) -> f64 {
    let headroom = if bits == RESCUE_BITS[0] { RESCUE_DD_HEADROOM } else { RESCUE_HEADROOM };
    2f64.powi(-((bits - headroom) as i32))
}

/// First tier that admits the binary64 conditioning estimate with margin. An
/// exactly singular binary64 pivot carries no estimate and starts at 256 bits.
fn first_tier(condition: f64) -> usize {
    if condition > 0.0 {
        RESCUE_BITS
            .iter()
            .position(|&bits| condition >= rescue_threshold(bits) * 2f64.powi(RESCUE_MARGIN as i32))
            .unwrap_or(RESCUE_BITS.len() - 1)
    } else {
        1
    }
}

/// The binary64 value `x` in the caller's domain, whose precision is 53 bits.
fn from_binary64<T: NativeFloat>(prototype: &T, x: f64) -> T {
    match (&x as &dyn std::any::Any).downcast_ref::<T>() {
        Some(value) => value.clone(),
        None => Float::with_val(53, x)
            .try_to_rational()
            .map_or_else(|| prototype.zero(), |exact| prototype.from_rational(&exact)),
    }
}

/// Direct-engine work at one rescue precision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectRescueTier {
    pub bits: u32,
    pub rows: usize,
    pub successes: usize,
}

/// Counters of the most recent direct-engine call.
#[derive(Clone, Debug, Default)]
pub struct DirectExecutionStats {
    pub input_rows: usize,
    /// Scalar-master evaluations by the direct engine, at every precision.
    pub master_rows: usize,
    pub direct_successes: usize,
    /// Lanes rerun after the binary64 direct pass (by the precision rescue,
    /// the symbolic engine or both), and how many of them succeeded.
    pub fallback_rows: usize,
    pub fallback_successes: usize,
    /// Direct-engine reruns per precision tier, in order.
    pub precision_rescue: Vec<DirectRescueTier>,
    /// Lanes passed on to the symbolic engine.
    pub symbolic_rows: usize,
    /// Binary64 lanes re-evaluated with rotated propagators (N >= 5), and
    /// those whose rotated copy disagreed or was refused.
    pub rotation_rows: usize,
    pub rotation_failures: usize,
    /// Lanes refused by the a priori binary64 gate and evaluated again in
    /// binary64 below it, and those kept by the rotation check.
    pub checked_rows: usize,
    pub checked_successes: usize,
    /// Lanes still refused, evaluated in binary64 with refused boxes by the
    /// small-Gram expansion, and those kept by the rotation check.
    pub expansion_rows: usize,
    pub expansion_successes: usize,
    pub fallback_preparation_error: Option<String>,
    pub fallback: Option<Box<TensorEngineExecution>>,
}

/// Lazily prepared direct workspaces of the rescue tiers.
#[derive(Clone, Default)]
struct RescueWorkspaces {
    double: Option<Box<DirectWorkspace<DoubleFloat>>>,
    mpfr: [Option<Box<DirectWorkspace<Float>>>; 2],
}

#[derive(Clone)]
pub(crate) struct DirectHybridWorkspace<T: NativeFloat> {
    plan: Arc<DirectHybridPlan>,
    direct: DirectWorkspace<T>,
    prototype: T,
    options: TensorWorkspaceOptions,
    accuracy_bits: Option<u32>,
    profiling: bool,
    precision_rescue: PrecisionRescue,
    /// Binary64 workspaces of the checked retry and of the small-Gram
    /// expansion retry, prepared on first use.
    checked: Option<Box<DirectWorkspace<T>>>,
    expanded: Option<Box<DirectWorkspace<T>>>,
    rescue: RescueWorkspaces,
    fallback: Option<Box<TensorWorkspace<T>>>,
    last: DirectExecutionStats,
    last_profile: Option<ExecutionProfile>,
    diagnostics: Vec<Vec<String>>,
}

fn masters_for(options: TensorWorkspaceOptions) -> DirectMasters {
    match options.master_policy {
        MasterExecutionPolicy::Default | MasterExecutionPolicy::NativeF64Abc => {
            DirectMasters::Native
        }
        MasterExecutionPolicy::Symbolica => DirectMasters::Policy(MasterExecutionPolicy::Symbolica),
    }
}

impl<T: NativeFloat> DirectHybridWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Binary64 ordinary numbers only; other domains use the fallback plan.
    pub(crate) fn eligible(prototype: &T) -> bool {
        T::FIXED_BITS == Some(53) && prototype.get_precision() == 53
    }

    pub(crate) fn new(
        plan: Arc<DirectHybridPlan>,
        prototype: T,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        let mut direct = plan
            .direct
            .workspace_with_masters(prototype.clone(), masters_for(options))?;
        direct.set_profiling(options.profiling);
        if rotation_tested(plan.direct.settings()) {
            direct.set_stability_threshold(from_binary64(&prototype, ROTATION_GATE));
        }
        if options.precision_rescue == PrecisionRescue::Automatic {
            direct.enable_partial_rescue(rescue_threshold(RESCUE_BITS[0]));
        }
        Ok(Self {
            plan,
            direct,
            prototype,
            options,
            accuracy_bits: None,
            profiling: options.profiling,
            precision_rescue: options.precision_rescue,
            checked: None,
            expanded: None,
            rescue: RescueWorkspaces::default(),
            fallback: None,
            last: DirectExecutionStats::default(),
            last_profile: None,
            diagnostics: Vec::new(),
        })
    }

    pub(crate) fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(format!(
                "requested accuracy {bits} bits is outside 1..={}",
                self.prototype.get_precision()
            )));
        }
        self.accuracy_bits = Some(bits);
        if let Some(fallback) = &mut self.fallback {
            fallback.set_accuracy_precision(bits)?;
        }
        Ok(())
    }

    pub(crate) fn set_precision_rescue(&mut self, rescue: PrecisionRescue) {
        self.precision_rescue = rescue;
        self.options.precision_rescue = rescue;
        for workspace in std::iter::once(&mut self.direct).chain(self.checked.as_deref_mut()) {
            match rescue {
                PrecisionRescue::Automatic => {
                    workspace.enable_partial_rescue(rescue_threshold(RESCUE_BITS[0]))
                }
                PrecisionRescue::Disabled => workspace.disable_partial_rescue(),
            }
        }
    }

    pub(crate) fn precision_rescue_setting(&self) -> PrecisionRescue {
        self.precision_rescue
    }

    /// Largest accepted E/A between a binary64 lane and its rotated copy:
    /// the requested accuracy `2^(19 - bits)` (2^-34 by default).
    fn rotation_tolerance(&self) -> f64 {
        match self.accuracy_bits {
            Some(bits) => 2f64.powi(19 - bits as i32),
            None => ROTATION_TOLERANCE,
        }
    }

    pub(crate) fn set_profiling(&mut self, enabled: bool) -> Result<(), ReductionError> {
        self.profiling = enabled;
        self.direct.set_profiling(enabled);
        for workspace in self.checked.iter_mut().chain(self.expanded.iter_mut()) {
            workspace.set_profiling(enabled);
        }
        if let Some(workspace) = &mut self.rescue.double {
            workspace.set_profiling(enabled);
        }
        for workspace in self.rescue.mpfr.iter_mut().flatten() {
            workspace.set_profiling(enabled);
        }
        if let Some(fallback) = &mut self.fallback {
            fallback.set_profiling(enabled)?;
        }
        Ok(())
    }

    pub(crate) fn binary_precision(&self) -> u32 {
        self.prototype.get_precision()
    }

    pub(crate) fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }

    pub(crate) fn last_execution(&self) -> &DirectExecutionStats {
        &self.last
    }

    pub(crate) fn last_profile(&self) -> Option<ExecutionProfile> {
        self.last_profile
    }

    fn fallback_workspace(&mut self) -> Result<&mut TensorWorkspace<T>, ReductionError> {
        if self.fallback.is_none() {
            let plan = self.plan.fallback()?;
            let mut options = self.options;
            options.profiling = self.profiling;
            let mut workspace =
                TensorWorkspace::new_with_options(&plan, self.prototype.clone(), options)?;
            if let Some(bits) = self.accuracy_bits {
                workspace.set_accuracy_precision(bits)?;
            }
            self.fallback = Some(Box::new(workspace));
        }
        Ok(self.fallback.as_mut().unwrap())
    }

    /// Rerun `lanes` at rescue tier `tier`. Returns, per lane, the status and
    /// conditioning estimate of the tier, and stores accepted coefficients.
    fn rescue_tier(
        &mut self,
        tier: usize,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        result: &mut BatchResult<T>,
        profile: &mut ExecutionProfile,
    ) -> Result<Vec<(LaneStatus, Option<f64>)>, ReductionError> {
        let bits = RESCUE_BITS[tier];
        let masters = masters_for(self.options);
        let plan = &self.plan.direct;
        let profiling = self.profiling;
        if tier == 0 {
            if self.rescue.double.is_none() {
                self.rescue.double = Some(prepare_tier::<DoubleFloat>(
                    plan.workspace_with_masters::<DoubleFloat>(DoubleFloat::from(0.0), masters),
                    DoubleFloat::from(rescue_threshold(bits)),
                    profiling,
                )?);
            }
            let workspace = self.rescue.double.as_mut().unwrap();
            run_tier::<T, DoubleFloat>(workspace, DoubleFloat::from, batch, lanes, &self.prototype, result, profile, &mut self.last)
        } else {
            let slot = &mut self.rescue.mpfr[tier - 1];
            if slot.is_none() {
                *slot = Some(prepare_tier::<Float>(
                    plan.workspace_with_masters::<Float>(Float::new(bits), masters),
                    Float::with_val(bits, rescue_threshold(bits)),
                    profiling,
                )?);
            }
            let workspace = slot.as_mut().unwrap();
            run_tier::<T, Float>(workspace, |x| Float::with_val(bits, x), batch, lanes, &self.prototype, result, profile, &mut self.last)
        }
    }

    /// Precision rescue of the conditioning-refused lanes among `retry`.
    /// Returns the lanes left for the symbolic engine.
    fn precision_rescue(
        &mut self,
        batch: &KinematicsBatch<T>,
        retry: &[usize],
        conditions: &[Option<f64>],
        result: &mut BatchResult<T>,
        profile: &mut ExecutionProfile,
    ) -> Result<Vec<usize>, ReductionError> {
        let mut pending: Vec<(usize, usize)> = Vec::new();
        let mut symbolic = Vec::new();
        for &lane in retry {
            match (&result.statuses[lane], conditions.get(lane).copied().flatten()) {
                (LaneStatus::Unstable(_), Some(condition)) => {
                    pending.push((lane, first_tier(condition)))
                }
                _ => symbolic.push(lane),
            }
        }
        for tier in 0..RESCUE_BITS.len() {
            let lanes: Vec<usize> = pending
                .iter()
                .filter(|&&(_, first)| first <= tier)
                .map(|&(lane, _)| lane)
                .collect();
            if lanes.is_empty() {
                continue;
            }
            let bits = RESCUE_BITS[tier];
            let mut outcomes = self.rescue_tier(tier, batch, &lanes, result, profile)?;
            if tier == 0 && rotation_tested(self.plan.direct.settings()) {
                // The double-double tier's looser gate is checked a posteriori:
                // the accepted lanes are evaluated again with rotated
                // propagators; a disagreement escalates the lane.
                let accepted: Vec<usize> = lanes
                    .iter()
                    .zip(&outcomes)
                    .filter(|(_, (status, _))| *status == LaneStatus::Success)
                    .map(|(&lane, _)| lane)
                    .collect();
                if !accepted.is_empty() {
                    let n = self.plan.direct.settings().propagators;
                    let mut check = result.clone();
                    let checks = self.rescue_tier(tier, &rotated(batch, n), &accepted, &mut check, profile)?;
                    let stride = result.basis.len() * 4;
                    let width = 5 * n + 1;
                    for (&lane, (status, _)) in accepted.iter().zip(checks) {
                        let difference = if status == LaneStatus::Success {
                            let mu = batch.values[lane * width + 5 * n].re.to_f64();
                            lane_difference(
                                &result.basis,
                                &result.values[lane * stride..(lane + 1) * stride],
                                &check.values[lane * stride..(lane + 1) * stride],
                                mu,
                            )
                        } else {
                            f64::INFINITY
                        };
                        if difference > RESCUE_ROTATION_TOLERANCE {
                            let index = lanes.iter().position(|&other| other == lane).unwrap();
                            outcomes[index] = (
                                LaneStatus::Unstable(format!(
                                    "loop-rotation test at {bits} bits: arrays differ by E/A {difference:e}"
                                )),
                                Some(0.0),
                            );
                        }
                    }
                }
            }
            let mut successes = 0;
            for (&lane, (status, condition)) in lanes.iter().zip(outcomes) {
                if status == LaneStatus::Success {
                    successes += 1;
                    result.fallback_used[lane] = true;
                    result.statuses[lane] = LaneStatus::Success;
                    self.diagnostics[lane]
                        .push(format!("direct engine at {bits} bits: accepted"));
                    pending.retain(|&(other, _)| other != lane);
                    continue;
                }
                self.diagnostics[lane].push(format!("direct engine at {bits} bits: {status:?}"));
                // Escalate only lanes refused on conditioning again.
                if condition.is_none() || tier + 1 == RESCUE_BITS.len() {
                    pending.retain(|&(other, _)| other != lane);
                    symbolic.push(lane);
                }
            }
            self.last.fallback_successes += successes;
            self.last.precision_rescue.push(DirectRescueTier {
                bits,
                rows: lanes.len(),
                successes,
            });
        }
        symbolic.sort_unstable();
        Ok(symbolic)
    }

    /// Checked binary64 retry of `lanes`, refused by the a priori gate: they
    /// are evaluated again in binary64 with the gate at `CHECKED_GATE`, with
    /// rotated propagators too, and kept when the two arrays agree within
    /// `tolerance * 2^-CHECKED_MARGIN_BITS`. Lanes not kept stay refused with
    /// their a priori conditioning estimate, which picks the rescue tier.
    fn checked_retry(
        &mut self,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        tolerance: f64,
        result: &mut BatchResult<T>,
        profile: &mut ExecutionProfile,
        expansion: bool,
    ) -> Result<(), ReductionError> {
        let n = self.plan.direct.settings().propagators;
        let slot = if expansion { &mut self.expanded } else { &mut self.checked };
        if slot.is_none() {
            let mut workspace = self
                .plan
                .direct
                .workspace_with_masters(self.prototype.clone(), masters_for(self.options))?;
            if expansion {
                // The a priori gate decides which boxes are expanded.
                if rotation_tested(self.plan.direct.settings()) {
                    workspace.set_stability_threshold(from_binary64(&self.prototype, ROTATION_GATE));
                }
                workspace.enable_gram_expansion(GRAM_BOOST);
            } else {
                workspace.set_stability_threshold(from_binary64(&self.prototype, CHECKED_GATE));
                if self.precision_rescue == PrecisionRescue::Automatic {
                    workspace.enable_partial_rescue(rescue_threshold(RESCUE_BITS[0]));
                }
            }
            workspace.set_profiling(self.profiling);
            *slot = Some(Box::new(workspace));
        }
        let workspace = slot.as_mut().unwrap();
        let width = batch.values.len() / batch.rows.max(1);
        let mut values = Vec::with_capacity(lanes.len() * width);
        for &lane in lanes {
            values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
        }
        let sub = KinematicsBatch { rows: lanes.len(), components: batch.components, values };
        let first = workspace.reduce_batch(&sub)?;
        self.last.master_rows += workspace.last_master_rows();
        let mut child = workspace.last_profile();
        let second = workspace.reduce_batch(&rotated(&sub, n))?;
        self.last.master_rows += workspace.last_master_rows();
        let rotated_profile = workspace.last_profile();
        child.geometry += rotated_profile.geometry;
        child.masters += rotated_profile.masters;
        child.recurrence += rotated_profile.recurrence;
        child.gather += rotated_profile.gather;
        child.output += rotated_profile.output;
        profile.arguments += child.geometry;
        profile.scalar_masters += child.masters;
        profile.reduction += child.recurrence + child.gather + child.output;
        if expansion {
            self.last.expansion_rows += lanes.len();
        } else {
            self.last.checked_rows += lanes.len();
        }
        let required = tolerance * 2f64.powi(-CHECKED_MARGIN_BITS);
        let stride = result.basis.len() * 4 * result.components;
        let width_lane = 5 * n + 1;
        for (k, &lane) in lanes.iter().enumerate() {
            let note = match (&first.statuses[k], &second.statuses[k]) {
                (LaneStatus::Success, LaneStatus::Success) => {
                    let mu = batch.values[lane * width_lane + 5 * n].re.to_f64();
                    let (a, b) = (
                        &first.values[k * stride..(k + 1) * stride],
                        &second.values[k * stride..(k + 1) * stride],
                    );
                    let difference = lane_difference(&result.basis, a, b, mu);
                    if difference <= required {
                        result.values[lane * stride..(lane + 1) * stride].clone_from_slice(a);
                        result.statuses[lane] = LaneStatus::Success;
                        result.primal_envelopes[lane] = first.primal_envelopes[k].clone();
                        result.primal_envelope_errors[lane] = first.primal_envelope_errors[k].clone();
                        let stage = if expansion {
                            self.last.expansion_successes += 1;
                            "binary64 small-Gram expansion"
                        } else {
                            self.last.checked_successes += 1;
                            "binary64 below the conditioning gate"
                        };
                        self.diagnostics[lane]
                            .push(format!("{stage}: kept, rotated copy agrees to E/A {difference:e}"));
                        continue;
                    }
                    format!("{stage} the rotated copy differs by E/A {difference:e}", stage = if expansion { "small-Gram expansion:" } else { "below the gate" })
                }
                (LaneStatus::Success, status) => format!("{} the rotated copy is refused ({status:?})", if expansion { "small-Gram expansion:" } else { "below the gate" }),
                (status, _) => format!("{} ({status:?})", if expansion { "small-Gram expansion refused" } else { "below the gate refused again" }),
            };
            if let LaneStatus::Unstable(previous) = &result.statuses[lane] {
                result.statuses[lane] = LaneStatus::Unstable(format!("{previous}; {note}"));
            }
        }
        Ok(())
    }

    pub(crate) fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let started = self.profiling.then(std::time::Instant::now);
        let mut result = self.direct.reduce_batch(batch)?;
        let mut direct_profile = self.direct.last_profile();
        self.last = DirectExecutionStats {
            input_rows: batch.rows,
            master_rows: self.direct.last_master_rows(),
            ..DirectExecutionStats::default()
        };
        let mut conditions = self.direct.last_conditions().to_vec();
        let settings = self.plan.direct.settings();
        let n = settings.propagators;
        let (tested, checkable) = (rotation_tested(settings), rotation_checkable(settings));
        let tolerance = self.rotation_tolerance();
        // Lanes refused by the a priori gate (N >= 5: or whose rotated copy
        // was), for the checked binary64 retry.
        let mut checked: Vec<usize> = Vec::new();
        if checkable {
            for lane in 0..batch.rows {
                if matches!(result.statuses[lane], LaneStatus::Unstable(_))
                    && conditions[lane].is_some_and(|condition| condition >= CHECKED_GATE)
                {
                    checked.push(lane);
                }
            }
        }
        if tested {
            let check = self.direct.reduce_batch(&rotated(batch, n))?;
            self.last.master_rows += self.direct.last_master_rows();
            let rotated_profile = self.direct.last_profile();
            direct_profile.geometry += rotated_profile.geometry;
            direct_profile.masters += rotated_profile.masters;
            direct_profile.recurrence += rotated_profile.recurrence;
            direct_profile.gather += rotated_profile.gather;
            direct_profile.output += rotated_profile.output;
            self.last.rotation_rows = batch.rows;
            let rotated_conditions = self.direct.last_conditions().to_vec();
            let stride = result.basis.len() * 4;
            let width = 5 * n + 1;
            for lane in 0..batch.rows {
                if result.statuses[lane] != LaneStatus::Success {
                    continue;
                }
                let note = if check.statuses[lane] == LaneStatus::Success {
                    let mu = batch.values[lane * width + 5 * n].re.to_f64();
                    let difference = lane_difference(
                        &result.basis,
                        &result.values[lane * stride..(lane + 1) * stride],
                        &check.values[lane * stride..(lane + 1) * stride],
                        mu,
                    );
                    if difference <= tolerance {
                        continue;
                    }
                    // Rounding estimate ~ amplification * 2^-53.
                    conditions[lane] = Some(2f64.powi(-53) / difference);
                    format!("loop-rotation test: binary64 arrays differ by E/A {difference:e}")
                } else {
                    conditions[lane] = rotated_conditions[lane].or(Some(0.0));
                    if checkable
                        && matches!(check.statuses[lane], LaneStatus::Unstable(_))
                        && rotated_conditions[lane].is_some_and(|condition| condition >= CHECKED_GATE)
                    {
                        checked.push(lane);
                    }
                    format!("loop-rotation test: rotated lane refused ({:?})", check.statuses[lane])
                };
                self.last.rotation_failures += 1;
                result.statuses[lane] = LaneStatus::Unstable(note);
            }
        }
        self.diagnostics.clear();
        self.diagnostics.resize(batch.rows, Vec::new());
        self.last.direct_successes = result
            .statuses
            .iter()
            .filter(|status| **status == LaneStatus::Success)
            .count();
        let mut profile = ExecutionProfile {
            arguments: direct_profile.geometry,
            scalar_masters: direct_profile.masters,
            reduction: direct_profile.recurrence + direct_profile.gather + direct_profile.output,
            total: Default::default(),
        };
        if !checked.is_empty() {
            checked.sort_unstable();
            checked.dedup();
            self.checked_retry(batch, &checked, tolerance, &mut result, &mut profile, false)?;
            // Lanes still refused: refused boxes by the small-Gram expansion.
            checked.retain(|&lane| result.statuses[lane] != LaneStatus::Success);
            if !checked.is_empty() {
                self.checked_retry(batch, &checked, tolerance, &mut result, &mut profile, true)?;
            }
        }
        let retry: Vec<usize> = result
            .statuses
            .iter()
            .enumerate()
            .filter(|(_, status)| {
                matches!(status, LaneStatus::Unstable(_) | LaneStatus::MasterFailure(_))
            })
            .map(|(lane, _)| lane)
            .collect();
        let disabled = self.precision_rescue == PrecisionRescue::Disabled;
        let mut rerun = Vec::with_capacity(retry.len());
        for &lane in &retry {
            // Rescue disabled: lanes refused on conditioning are final.
            let last = disabled
                && matches!(result.statuses[lane], LaneStatus::Unstable(_))
                && conditions[lane].is_some();
            if let LaneStatus::Unstable(note) | LaneStatus::MasterFailure(note) =
                &result.statuses[lane]
            {
                let action = if last { "precision rescue disabled" } else { "fallback requested" };
                self.diagnostics[lane].push(format!("direct engine: {note}; {action}"));
            }
            if !last {
                rerun.push(lane);
            }
        }
        self.last.fallback_rows = rerun.len();
        let retry = if rerun.is_empty() || disabled {
            rerun
        } else {
            self.precision_rescue(batch, &rerun, &conditions, &mut result, &mut profile)?
        };
        if !retry.is_empty() {
            self.last.symbolic_rows = retry.len();
            let width = batch.values.len() / batch.rows.max(1);
            let mut values = Vec::with_capacity(retry.len() * width);
            for &lane in &retry {
                values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
            }
            let sub = KinematicsBatch {
                rows: retry.len(),
                components: batch.components,
                values,
            };
            match self.fallback_workspace() {
                Ok(fallback) => {
                    let fallback_result = fallback.reduce_batch(&sub)?;
                    if let Some(child) = fallback.last_profile() {
                        profile.arguments += child.arguments;
                        profile.scalar_masters += child.scalar_masters;
                        profile.reduction += child.reduction;
                    }
                    let fallback_diagnostics = fallback.last_diagnostics().to_vec();
                    self.last.fallback = Some(Box::new(fallback.engine_execution()));
                    let stride = result.basis.len() * 4 * result.components;
                    for (k, &lane) in retry.iter().enumerate() {
                        result.fallback_used[lane] = true;
                        result.analytic_used[lane] = fallback_result.analytic_used[k];
                        result.statuses[lane] = fallback_result.statuses[k].clone();
                        if let Some(notes) = fallback_diagnostics.get(k) {
                            self.diagnostics[lane].extend(notes.iter().cloned());
                        }
                        if fallback_result.statuses[k] == LaneStatus::Success {
                            self.last.fallback_successes += 1;
                            result.values[lane * stride..(lane + 1) * stride].clone_from_slice(
                                &fallback_result.values[k * stride..(k + 1) * stride],
                            );
                        } else {
                            for value in &mut result.values[lane * stride..(lane + 1) * stride] {
                                *value = Complex::new(self.prototype.zero(), self.prototype.zero());
                            }
                        }
                        result.primal_envelopes[lane] =
                            fallback_result.primal_envelopes.get(k).cloned().flatten();
                        result.primal_envelope_errors[lane] =
                            fallback_result.primal_envelope_errors.get(k).cloned().flatten();
                    }
                }
                Err(error) => {
                    let note = error.to_string();
                    self.last.fallback_preparation_error = Some(note.clone());
                    for &lane in &retry {
                        result.fallback_used[lane] = true;
                        self.diagnostics[lane].push(format!("fallback preparation failed: {note}"));
                    }
                }
            }
        }
        if let Some(started) = started {
            profile.total = started.elapsed();
            self.last_profile = Some(profile);
        } else {
            self.last_profile = None;
        }
        Ok(result)
    }
}

/// Run `lanes` of `batch` on a rescue-tier workspace with exactly lifted
/// inputs; round accepted lanes into `result`.
#[allow(clippy::too_many_arguments)]
fn run_tier<T: NativeFloat, R: NativeFloat>(
    workspace: &mut DirectWorkspace<R>,
    lift: impl Fn(f64) -> R,
    batch: &KinematicsBatch<T>,
    lanes: &[usize],
    prototype: &T,
    result: &mut BatchResult<T>,
    profile: &mut ExecutionProfile,
    stats: &mut DirectExecutionStats,
) -> Result<Vec<(LaneStatus, Option<f64>)>, ReductionError>
where
    Complex<R>: EvaluationDomain,
{
    let width = batch.values.len() / batch.rows.max(1);
    let mut values = Vec::with_capacity(lanes.len() * width);
    for &lane in lanes {
        // Exact: the caller's domain has 53 bits.
        values.extend(
            batch.values[lane * width..(lane + 1) * width]
                .iter()
                .map(|z| Complex::new(lift(z.re.to_f64()), lift(z.im.to_f64()))),
        );
    }
    let sub = KinematicsBatch {
        rows: lanes.len(),
        components: batch.components,
        values,
    };
    let rescued = workspace.reduce_batch(&sub)?;
    stats.master_rows += workspace.last_master_rows();
    let child = workspace.last_profile();
    profile.arguments += child.geometry;
    profile.scalar_masters += child.masters;
    profile.reduction += child.recurrence + child.gather + child.output;
    let stride = result.basis.len() * 4 * result.components;
    let conditions = workspace.last_conditions();
    let mut outcomes = Vec::with_capacity(lanes.len());
    for (k, &lane) in lanes.iter().enumerate() {
        if rescued.statuses[k] == LaneStatus::Success {
            for (out, value) in result.values[lane * stride..(lane + 1) * stride]
                .iter_mut()
                .zip(&rescued.values[k * stride..(k + 1) * stride])
            {
                *out = Complex::new(
                    from_binary64(prototype, value.re.to_f64()),
                    from_binary64(prototype, value.im.to_f64()),
                );
            }
        }
        outcomes.push((rescued.statuses[k].clone(), conditions.get(k).copied().flatten()));
    }
    Ok(outcomes)
}

fn prepare_tier<R: NativeFloat>(
    workspace: Result<DirectWorkspace<R>, ReductionError>,
    threshold: R,
    profiling: bool,
) -> Result<Box<DirectWorkspace<R>>, ReductionError>
where
    Complex<R>: EvaluationDomain,
{
    let mut workspace = workspace?;
    workspace.set_stability_threshold(threshold);
    workspace.set_profiling(profiling);
    Ok(Box::new(workspace))
}
