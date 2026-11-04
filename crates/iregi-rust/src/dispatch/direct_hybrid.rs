//! Binary64 default: the direct numerical engine with per-lane fallback.
//!
//! The direct engine evaluates the validated PV recurrence numerically. With
//! four or more propagators and equal powers (Gram recursions, whose rounding
//! a conditioning estimate predicts poorly) every binary64 lane is evaluated a
//! second time with its propagators rotated, the same integral through
//! different Gram systems, as in MadLoop's loop-direction test; a relative
//! difference above the requested accuracy (2^-34 by default) or a refused
//! rotated copy refuses the lane. A lane within `NEAR_MARGIN_BITS` of the
//! tolerance is evaluated in a third cyclic order, which must agree too.
//!
//! Lanes the a priori conditioning gate refuses (four or more propagators,
//! equal powers) are evaluated again in binary64 with the gate lowered to
//! `CHECKED_GATE` and kept when their rotated copy agrees two bits better
//! than the tolerance: the a posteriori test replaces the a priori bound,
//! which is pessimistic for most such lanes.
//!
//! A lane the binary64 pass refuses (five to seven propagators, rank <= 7)
//! is first evaluated again with COLLIER's per-box choice: boxes and
//! triangles whose a priori PV estimate 2^-52 w^r exceeds the request, or
//! that the gate refuses, by the small-Gram expansions or the alternative PV
//! reduction; it is kept at the retries' stricter level.
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
    basis::Monomial, direct::shared::SubsetStore,
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
/// Binary64 conditioning gate of plans with the loop-rotation test (N >= 4,
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
/// correction of order Delta / (Zadj f)_j; the small-Cayley variant gains one
/// order per two ranks). On the gate-refused stability-20261030 boxes the
/// expansions converged on 22 of 34 rank-4 and 239 of 370 rank-6 lanes with
/// 8, and on 34 and 254 with 12 (15 of them by the small-Cayley variant);
/// every converged lane agreed with 256 bits to E/A <= 3.4e-11.
const GRAM_BOOST: u16 = 12;
/// A lane whose rotated copy agrees within this many bits of the tolerance is
/// evaluated in a third cyclic order, which must agree with the first too.
const NEAR_MARGIN_BITS: i32 = 4;
/// Whether lanes the binary64 pass refuses get the a priori method choice
/// as their first retry. (Validated up to seven
/// points and rank 7; larger plans would build many high-rank auxiliary
/// plans.) The estimate 2^-52 w^r is pessimistic: it exceeds the request on
/// every opening benchmark point from N5R5 to N7R7, where PV passes the
/// rotation test (E/A 4e-14 to 7e-12) and the expansions would cost 13-40 ms
/// per lane against 0.07-2 ms; on stability-20261030 no PV lane the rotation
/// test accepted was improved by them.
fn estimate_choice(settings: &ReductionSettings) -> bool {
    rotation_tested(settings) && (4..=7).contains(&settings.propagators) && settings.rank <= 7
}

/// Refused lanes the per-box choice kept with an agreement worse than this
/// also try every box and triangle with w > 2 off PV (kept if better).
const HARDER_AGREEMENT: f64 = 1.0 / (1u64 << 48) as f64;
/// Consensus of cyclic orders: at least this many other orders must agree
/// with the kept one, each within the tolerance times 2^-CONSENSUS_MARGIN_BITS.
/// On stability-20261030 (all cyclic orders, binary64, against 256 bits) this
/// kept hexagon lanes at E/A median 1.3e-14, worst 4.2e-12, and certified
/// none beyond the request; two partners instead kept 3 of the 32 opening
/// N7R7 lanes, one at E/A 1.2e-10 with the orders agreeing to 3.3e-12
/// (their orders share errors near 1e-10). Probe: hexagon median 3.8e-14 ->
/// 2.7e-14, pentagon p99 5.9e-12 -> 3.9e-12, nothing certified beyond.
const CONSENSUS_PARTNERS: usize = 3;
const CONSENSUS_MARGIN_BITS: i32 = 4;

/// Whether the binary64 pass runs the loop-rotation test. Boxes too: on
/// stability-20261030 the conditioning gate alone certified 86 of 10000
/// rank-4 and 49 of 10000 rank-6 boxes beyond the tolerance (up to E/A
/// 9.6e-9); the rotation test refuses 85 and 49 of them.
fn rotation_tested(settings: &ReductionSettings) -> bool {
    settings.propagators >= 4 && uniform_powers(settings)
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
    // Squared magnitudes and weights mu^-2r (no hypot or division per
    // entry: this test runs on every lane); `robust_lane_difference` if a
    // square overflows.
    let mut inverse = [0.0f64; 33];
    let step = 1.0 / (mu * mu);
    let mut weight = 1.0;
    for slot in &mut inverse {
        *slot = weight;
        weight *= step;
    }
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (m, monomial) in basis.iter().enumerate() {
        let rank = monomial.rank() as usize;
        let Some(&weight) = inverse.get(rank) else {
            return robust_lane_difference(basis, a, b, mu);
        };
        for slot in 0..4 {
            let (x, y) = (&a[4 * m + slot], &b[4 * m + slot]);
            let (xr, xi, yr, yi) = (x.re.to_f64(), x.im.to_f64(), y.re.to_f64(), y.im.to_f64());
            let (dr, di) = (xr - yr, xi - yi);
            error = error.max((dr * dr + di * di) * weight);
            amplitude = amplitude.max((yr * yr + yi * yi) * weight);
        }
    }
    if !(error.is_finite() && amplitude.is_finite() && amplitude > f64::MIN_POSITIVE) {
        return robust_lane_difference(basis, a, b, mu);
    }
    (error / amplitude).sqrt()
}

fn robust_lane_difference<T: NativeFloat>(
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
/// posteriori by the loop-rotation test for N >= 4 (`RESCUE_ROTATION_TOLERANCE`):
/// on physical gg -> ggggg heptagons the conditioning estimate called for 256
/// bits, while the 106-bit direct engine with no gate at all agreed with 256
/// bits to E/A <= 1.7e-23 (30 points).
const RESCUE_DD_HEADROOM: u32 = 20;
/// Largest accepted E/A between a 106-bit lane and its rotated copy, both
/// rounded to binary64 (a few ulps).
const RESCUE_ROTATION_TOLERANCE: f64 = 1.0 / (1u64 << 46) as f64;
/// Binary64 conditioning estimate (2^-53 over the rotation difference) at or
/// above which a lane rescued at double-double skips that tier's rotation
/// check: its binary64 difference was at most 2^-20, so the double-double
/// error is ~2^-73, far below `RESCUE_ROTATION_TOLERANCE` (the opening N7R7
/// batch of 32 spent half of its time in that check).
const RESCUE_CHECK_CONDITION: f64 = 1.0 / (1u64 << 33) as f64;

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
    /// Binary64 lanes re-evaluated with rotated propagators (N >= 4, equal powers), and
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
    /// Per-subset results shared with the other integrals of an amplitude
    /// (`crate::AmplitudeWorkspace`), given to every binary64 workspace here.
    shared: Option<Arc<SubsetStore>>,
    /// Binary64 workspaces of the checked retry and of the small-Gram
    /// expansion retry, prepared on first use.
    checked: Option<Box<DirectWorkspace<T>>>,
    expanded: Option<Box<DirectWorkspace<T>>>,
    /// ... and of the alternative-PV retry (refused boxes and triangles by
    /// the modified Cayley reduction only).
    alternative: Option<Box<DirectWorkspace<T>>>,
    /// ... and of the stage computing every box and triangle by the
    /// expansions or the alternative PV reduction (COLLIER's per-box choice).
    everywhere: Option<Box<DirectWorkspace<T>>>,
    /// ... and of the a priori method choice (COLLIER-style: boxes and
    /// triangles whose PV estimate exceeds the request leave PV).
    estimated: Option<Box<DirectWorkspace<T>>>,
    /// Binary64 workspace without any gate, for best-effort arrays of lanes
    /// no other binary64 stage computed.
    open: Option<Box<DirectWorkspace<T>>>,
    /// Per lane of the current call: the best binary64 array computed for a
    /// lane its checks refused, with the E/A between it and its rotated copy.
    /// Every binary64 array computed for a refused lane, with the E/A to its
    /// rotated copy (NaN: no rotated copy).
    candidates: Vec<Vec<(Vec<Complex<T>>, f64)>>,
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
            alternative: None,
            everywhere: None,
            estimated: None,
            open: None,
            shared: None,
            candidates: Vec::new(),
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

    /// Share per-subset results through `store` in every binary64 workspace,
    /// present and future (`None`: no sharing).
    pub(crate) fn set_shared_subsets(&mut self, store: Option<Arc<SubsetStore>>) {
        self.shared = store;
        self.direct.set_shared_subsets(self.shared.clone());
        for workspace in [
            &mut self.checked,
            &mut self.expanded,
            &mut self.alternative,
            &mut self.everywhere,
            &mut self.estimated,
            &mut self.open,
        ]
        .into_iter()
        .flatten()
        {
            workspace.set_shared_subsets(self.shared.clone());
        }
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
        for workspace in self.checked.iter_mut().chain(self.expanded.iter_mut()).chain(self.alternative.iter_mut()).chain(self.open.iter_mut()) {
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

    /// The double-double tier's looser gate, checked a posteriori where the
    /// rotation test applies: the lanes it accepted (`outcomes`, written to
    /// `result`) are evaluated again with rotated propagators, and a
    /// disagreement refuses the lane (the caller escalates it). Lanes whose
    /// binary64 estimate is mild (refused by a rotation difference d <= 2^-20:
    /// a double-double error ~ d 2^-53) are not checked again.
    fn check_double_double(
        &mut self,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        conditions: &[Option<f64>],
        outcomes: &mut [(LaneStatus, Option<f64>)],
        result: &BatchResult<T>,
        profile: &mut ExecutionProfile,
    ) -> Result<(), ReductionError> {
        if !rotation_tested(self.plan.direct.settings()) {
            return Ok(());
        }
        let bits = RESCUE_BITS[0];
        let accepted: Vec<usize> = lanes
            .iter()
            .zip(outcomes.iter())
            .filter(|(lane, (status, _))| {
                *status == LaneStatus::Success
                    && !conditions.get(**lane).copied().flatten().is_some_and(|c| c >= RESCUE_CHECK_CONDITION)
            })
            .map(|(&lane, _)| lane)
            .collect();
        if accepted.is_empty() {
            return Ok(());
        }
        let n = self.plan.direct.settings().propagators;
        let mut check = result.clone();
        let checks = self.rescue_tier(0, &rotated(batch, n), &accepted, &mut check, profile)?;
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
        Ok(())
    }

    /// With the precision rescue on, lanes the a priori method choice kept
    /// without a clear agreement (`HARDER_AGREEMENT`) are evaluated by the
    /// double-double tier (with its rotation check) before the binary64 retry
    /// with every box and triangle off PV, which costs about four times as
    /// much per lane and bettered only part of them. A lane the tier accepts
    /// takes its double-double array; the lanes it refuses are returned (for
    /// that binary64 retry) with the choice's array.
    fn upgrade_to_double_double(
        &mut self,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        conditions: &[Option<f64>],
        result: &mut BatchResult<T>,
        profile: &mut ExecutionProfile,
    ) -> Result<Vec<usize>, ReductionError> {
        let mut trial = result.clone();
        let mut outcomes = self.rescue_tier(0, batch, lanes, &mut trial, profile)?;
        self.check_double_double(batch, lanes, conditions, &mut outcomes, &trial, profile)?;
        let stride = result.basis.len() * 4 * result.components;
        let bits = RESCUE_BITS[0];
        let mut successes = 0;
        let mut refused = Vec::new();
        for (&lane, (status, _)) in lanes.iter().zip(outcomes) {
            if status == LaneStatus::Success {
                successes += 1;
                result.values[lane * stride..(lane + 1) * stride]
                    .clone_from_slice(&trial.values[lane * stride..(lane + 1) * stride]);
                result.fallback_used[lane] = true;
                result.estimated_errors[lane] = None;
                self.diagnostics[lane].push(format!("direct engine at {bits} bits: accepted"));
            } else {
                self.diagnostics[lane].push(format!("direct engine at {bits} bits: {status:?}"));
                refused.push(lane);
            }
        }
        self.last.fallback_successes += successes;
        self.last.precision_rescue.push(DirectRescueTier { bits, rows: lanes.len(), successes });
        Ok(refused)
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
            if tier == 0 {
                self.check_double_double(batch, &lanes, conditions, &mut outcomes, result, profile)?;
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
        stage: u8,
    ) -> Result<(), ReductionError> {
        // 0: below the gate; 1: small-Gram expansions (alternative PV as
        // their last resort); 2: alternative PV alone; 3: every box and
        // triangle off PV where w > 2; 4: the a priori method choice. Stages
        // 1 to 4 count as expansion rows.
        let expansion = stage >= 1;
        let n = self.plan.direct.settings().propagators;
        let slot = match stage {
            0 => &mut self.checked,
            1 => &mut self.expanded,
            2 => &mut self.alternative,
            3 => &mut self.everywhere,
            _ => &mut self.estimated,
        };
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
                if stage == 4 {
                    let gate = from_binary64(&self.prototype, ROTATION_GATE);
                    workspace.enable_estimated_method_choice(GRAM_BOOST, gate);
                    if self.precision_rescue == PrecisionRescue::Automatic {
                        workspace.enable_partial_rescue(rescue_threshold(RESCUE_BITS[0]));
                    }
                } else if stage == 3 {
                    let gate = if rotation_tested(self.plan.direct.settings()) { ROTATION_GATE } else { 1e-12 };
                    workspace.enable_gram_expansion_everywhere(GRAM_BOOST, from_binary64(&self.prototype, gate));
                } else if stage == 2 {
                    workspace.enable_alternative_pv();
                } else {
                    workspace.enable_gram_expansion(GRAM_BOOST);
                }
            } else {
                workspace.set_stability_threshold(from_binary64(&self.prototype, CHECKED_GATE));
                if self.precision_rescue == PrecisionRescue::Automatic {
                    workspace.enable_partial_rescue(rescue_threshold(RESCUE_BITS[0]));
                }
            }
            workspace.set_profiling(self.profiling);
            workspace.set_shared_subsets(self.shared.clone());
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
        let mu = |k: usize| batch.values[lanes[k] * width_lane + 5 * n].re.to_f64();
        let mut differences: Vec<Option<f64>> = (0..lanes.len())
            .map(|k| {
                (first.statuses[k] == LaneStatus::Success && second.statuses[k] == LaneStatus::Success).then(|| {
                    lane_difference(
                        &result.basis,
                        &first.values[k * stride..(k + 1) * stride],
                        &second.values[k * stride..(k + 1) * stride],
                        mu(k),
                    )
                })
            })
            .collect();
        // Near misses: a third cyclic order must agree with the first too
        // (stability-20261030 hexagon #1208: kept by the expansion at a
        // difference of 3.6e-12, E/A 2.0e-10).
        let near: Vec<usize> = (0..lanes.len())
            .filter(|&k| {
                differences[k].is_some_and(|d| d <= required && d > required * 2f64.powi(-NEAR_MARGIN_BITS))
            })
            .collect();
        let mut third_refused = vec![false; lanes.len()];
        if !near.is_empty() && n >= 3 {
            let mut values = Vec::with_capacity(near.len() * width);
            for &k in &near {
                values.extend_from_slice(&sub.values[k * width..(k + 1) * width]);
            }
            let near_batch = KinematicsBatch { rows: near.len(), components: batch.components, values };
            let third = workspace.reduce_batch(&rotated(&rotated(&near_batch, n), n))?;
            self.last.master_rows += workspace.last_master_rows();
            let third_profile = workspace.last_profile();
            profile.arguments += third_profile.geometry;
            profile.scalar_masters += third_profile.masters;
            profile.reduction += third_profile.recurrence + third_profile.gather + third_profile.output;
            for (j, &k) in near.iter().enumerate() {
                if third.statuses[j] == LaneStatus::Success {
                    let difference = lane_difference(
                        &result.basis,
                        &first.values[k * stride..(k + 1) * stride],
                        &third.values[j * stride..(j + 1) * stride],
                        mu(k),
                    );
                    differences[k] = differences[k].map(|d| d.max(difference));
                } else {
                    third_refused[k] = true;
                }
            }
        }
        for (k, &lane) in lanes.iter().enumerate() {
            let note = match (&first.statuses[k], &second.statuses[k]) {
                (LaneStatus::Success, LaneStatus::Success) if third_refused[k] => {
                    self.candidates[lane].push((first.values[k * stride..(k + 1) * stride].to_vec(), differences[k].unwrap_or(f64::NAN)));
                    format!("{} a third order is refused", if expansion { "small-Gram expansion:" } else { "below the gate" })
                }
                (LaneStatus::Success, LaneStatus::Success) => {
                    let a = &first.values[k * stride..(k + 1) * stride];
                    let difference = differences[k].unwrap_or(f64::NAN);
                    // An accepted lane (the harder retry after the choice) is
                    // replaced only by a better agreement.
                    let better = result.statuses[lane] != LaneStatus::Success
                        || result.estimated_errors[lane].map_or(true, |e| difference < e);
                    if difference <= required && better {
                        result.values[lane * stride..(lane + 1) * stride].clone_from_slice(a);
                        result.statuses[lane] = LaneStatus::Success;
                        result.estimated_errors[lane] = Some(difference);
                        result.primal_envelopes[lane] = first.primal_envelopes[k].clone();
                        result.primal_envelope_errors[lane] = first.primal_envelope_errors[k].clone();
                        let stage = if expansion {
                            self.last.expansion_successes += 1;
                            match stage {
                                2 => "binary64 alternative PV",
                                4 => "binary64 a priori method choice",
                                _ => "binary64 small-Gram expansion",
                            }
                        } else {
                            self.last.checked_successes += 1;
                            "binary64 below the conditioning gate"
                        };
                        self.diagnostics[lane]
                            .push(format!("{stage}: kept, rotated copy agrees to E/A {difference:e}"));
                        continue;
                    }
                    self.candidates[lane].push((a.to_vec(), difference));
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

    /// Consensus of cyclic orders for refused lanes: every cyclic relabelling
    /// of the propagators is evaluated in binary64 (the pass's own
    /// configuration), and an order that at least `CONSENSUS_PARTNERS` other
    /// orders agree with within `tolerance * 2^-CONSENSUS_MARGIN_BITS` is kept
    /// (most partners, then the closest). The orders' Gram systems differ, so
    /// their errors differ widely (a heptagon lane: 1.2e-9 in orders 0 and 1,
    /// 7.7e-11..9.1e-11 in orders 2, 3, 6).
    fn order_consensus(
        &mut self,
        batch: &KinematicsBatch<T>,
        lanes: &[usize],
        tolerance: f64,
        result: &mut BatchResult<T>,
    ) -> Result<(), ReductionError> {
        let n = self.plan.direct.settings().propagators;
        let width = batch.values.len() / batch.rows.max(1);
        let mut values = Vec::with_capacity(lanes.len() * width);
        for &lane in lanes {
            values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
        }
        let mut sub = KinematicsBatch { rows: lanes.len(), components: batch.components, values };
        let mut orders = Vec::with_capacity(n);
        for k in 0..n {
            if k > 0 {
                sub = rotated(&sub, n);
            }
            orders.push(self.direct.reduce_batch(&sub)?);
            self.last.master_rows += self.direct.last_master_rows();
        }
        self.last.checked_rows += lanes.len();
        let required = tolerance * 2f64.powi(-CONSENSUS_MARGIN_BITS);
        let stride = result.basis.len() * 4 * result.components;
        let width_lane = 5 * n + 1;
        for (j, &lane) in lanes.iter().enumerate() {
            let mu = batch.values[lane * width_lane + 5 * n].re.to_f64();
            let ok: Vec<usize> = (0..n).filter(|&k| orders[k].statuses[j] == LaneStatus::Success).collect();
            let mut best: Option<(usize, f64, usize)> = None;
            for &a in &ok {
                let mut partners = Vec::new();
                for &b in &ok {
                    if a != b {
                        let difference = lane_difference(
                            &result.basis,
                            &orders[a].values[j * stride..(j + 1) * stride],
                            &orders[b].values[j * stride..(j + 1) * stride],
                            mu,
                        );
                        if difference <= required {
                            partners.push(difference);
                        }
                    }
                }
                if partners.len() >= CONSENSUS_PARTNERS {
                    let worst = partners.iter().copied().fold(0.0f64, f64::max);
                    let better = match best {
                        None => true,
                        Some((count, spread, _)) => partners.len() > count || (partners.len() == count && worst < spread),
                    };
                    if better {
                        best = Some((partners.len(), worst, a));
                    }
                }
            }
            if let Some((count, spread, a)) = best {
                result.values[lane * stride..(lane + 1) * stride].clone_from_slice(&orders[a].values[j * stride..(j + 1) * stride]);
                result.statuses[lane] = LaneStatus::Success;
                result.estimated_errors[lane] = Some(spread);
                result.primal_envelopes[lane] = orders[a].primal_envelopes[j].clone();
                result.primal_envelope_errors[lane] = orders[a].primal_envelope_errors[j].clone();
                self.last.checked_successes += 1;
                self.diagnostics[lane].push(format!(
                    "binary64 cyclic orders: kept order {a}, {count} other orders agree to E/A {spread:e}"
                ));
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
        result.estimated_errors = vec![None; batch.rows];
        result.best_effort = vec![false; batch.rows];
        self.candidates.clear();
        self.candidates.resize(batch.rows, Vec::new());
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
        let choice_enabled = estimate_choice(settings);
        let tolerance = self.rotation_tolerance();
        // Lanes refused by the a priori gate (N >= 4: or whose rotated copy
        // was), for the checked binary64 retry.
        let mut checked: Vec<usize> = Vec::new();
        // Lanes whose binary64 arrays passed the gate but whose cyclic orders
        // disagree (rotation test or third order).
        let mut order_refused = vec![false; batch.rows];
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
            // Lanes accepted within NEAR_MARGIN_BITS of the tolerance, with
            // their difference, for the third-order check.
            let mut near: Vec<(usize, f64)> = Vec::new();
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
                        result.estimated_errors[lane] = Some(difference);
                        if difference > tolerance * 2f64.powi(-NEAR_MARGIN_BITS) {
                            near.push((lane, difference));
                        }
                        continue;
                    }
                    self.candidates[lane].push((result.values[lane * stride..(lane + 1) * stride].to_vec(), difference));
                    // Rounding estimate ~ amplification * 2^-53.
                    conditions[lane] = Some(2f64.powi(-53) / difference);
                    order_refused[lane] = true;
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
            if !near.is_empty() && n >= 3 {
                // Near misses: the two orders can share an error several
                // times the tolerance (stability-20261030 pentagons: E/A
                // 1.4e-10 at a difference of 2.4e-11). A third cyclic order
                // must agree with the first as well.
                let mut values = Vec::with_capacity(near.len() * width);
                for &(lane, _) in &near {
                    values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
                }
                let sub = KinematicsBatch { rows: near.len(), components: batch.components, values };
                let third = self.direct.reduce_batch(&rotated(&rotated(&sub, n), n))?;
                self.last.master_rows += self.direct.last_master_rows();
                let third_profile = self.direct.last_profile();
                direct_profile.geometry += third_profile.geometry;
                direct_profile.masters += third_profile.masters;
                direct_profile.recurrence += third_profile.recurrence;
                direct_profile.gather += third_profile.gather;
                direct_profile.output += third_profile.output;
                self.last.rotation_rows += near.len();
                for (k, &(lane, first)) in near.iter().enumerate() {
                    let note = if third.statuses[k] == LaneStatus::Success {
                        let mu = batch.values[lane * width + 5 * n].re.to_f64();
                        let difference = lane_difference(
                            &result.basis,
                            &result.values[lane * stride..(lane + 1) * stride],
                            &third.values[k * stride..(k + 1) * stride],
                            mu,
                        );
                        if difference <= tolerance {
                            result.estimated_errors[lane] = Some(first.max(difference));
                            continue;
                        }
                        let difference = first.max(difference);
                        conditions[lane] = Some(2f64.powi(-53) / difference);
                        order_refused[lane] = true;
                        format!("loop-rotation test: a third order differs by E/A {difference:e}")
                    } else {
                        conditions[lane] = self.direct.last_conditions().get(k).copied().flatten().or(Some(0.0));
                        format!("loop-rotation test: third order refused ({:?})", third.statuses[k])
                    };
                    self.candidates[lane].push((result.values[lane * stride..(lane + 1) * stride].to_vec(), first));
                    result.estimated_errors[lane] = None;
                    self.last.rotation_failures += 1;
                    result.statuses[lane] = LaneStatus::Unstable(note);
                }
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
        // Consensus of cyclic orders first (five and six points; on heptagons
        // it did not help, and the opening N7R7 batch paid 12% for it).
        if tested && (5..=6).contains(&n) {
            let refused: Vec<usize> = (0..batch.rows)
                .filter(|&lane| matches!(result.statuses[lane], LaneStatus::Unstable(_)))
                .collect();
            if !refused.is_empty() {
                self.order_consensus(batch, &refused, tolerance, &mut result)?;
                checked.retain(|&lane| result.statuses[lane] != LaneStatus::Success);
            }
        }
        // COLLIER-style a priori method choice (boxes and triangles whose PV
        // estimate exceeds the request, or that the gate refuses, by the
        // expansions or the alternative PV) for every lane the binary64 pass
        // refused (gate or rotation test), before the other retries: on
        // stability-20261030 hexagons the lanes it keeps reach E/A median
        // 3.7e-14 against 2.8e-13 for PV below the gate.
        // With the precision rescue on, refused heptagon (and larger) lanes go
        // to it directly: their binary64 retries rarely succeed there (3 of
        // 495 stability heptagons) and cost the opening N7R7 batch of 32
        // about 0.6 s of 1.3 s. Accepted lanes are not retried for accuracy
        // beyond the request (the former upgrade of accepted box and pentagon
        // lanes the a priori estimate flagged: on the stability points their
        // 99th percentiles 5e-13..3e-12 against 5e-12..8e-12 without it, still
        // below COLLIER's, for 20% of the hexagon and 34% of the pentagon
        // amplitude's time).
        let binary64_retries = !(self.precision_rescue == PrecisionRescue::Automatic && n >= 7);
        // With the precision rescue on, lanes the a priori choice does not keep
        // go to the rescue directly: the binary64 retries below (below the
        // gate, by the expansions, by the alternative PV, every box and
        // triangle off PV) kept 2 of 8 such box lanes of the hexagon amplitude
        // at about 2 ms per lane, where the double-double tier takes about
        // 0.6 ms per box lane and gives 106 bits (on stability-20261030 the 15
        // box points these retries kept reach E/A 2e-31 or below that way).
        let rescue_after_choice =
            choice_enabled && binary64_retries && self.precision_rescue == PrecisionRescue::Automatic;
        let estimated: Vec<usize> = if choice_enabled && binary64_retries {
            (0..batch.rows)
                .filter(|&lane| matches!(result.statuses[lane], LaneStatus::Unstable(_)))
                .collect()
        } else {
            Vec::new()
        };
        if !estimated.is_empty() {
            let refused: Vec<bool> = estimated.iter().map(|&lane| result.statuses[lane] != LaneStatus::Success).collect();
            self.checked_retry(batch, &estimated, tolerance, &mut result, &mut profile, 4)?;
            checked.retain(|&lane| result.statuses[lane] != LaneStatus::Success);
            // Lanes the pass refused and the choice kept without a clear
            // agreement: every box and triangle with w > 2 off PV as well,
            // kept where its rotated copy agrees better; with the precision
            // rescue on, the double-double tier first
            // (`upgrade_to_double_double`), this retry for the lanes it refuses.
            if n >= 5 {
                let harder: Vec<usize> = estimated
                    .iter()
                    .zip(&refused)
                    .filter(|&(&lane, &was)| {
                        was && result.statuses[lane] == LaneStatus::Success
                            && result.estimated_errors[lane].is_some_and(|e| e > HARDER_AGREEMENT)
                    })
                    .map(|(&lane, _)| lane)
                    .collect();
                let harder = if self.precision_rescue == PrecisionRescue::Automatic && !harder.is_empty() {
                    self.upgrade_to_double_double(batch, &harder, &conditions, &mut result, &mut profile)?
                } else {
                    harder
                };
                if !harder.is_empty() {
                    self.checked_retry(batch, &harder, tolerance, &mut result, &mut profile, 3)?;
                }
            }
        }
        if !checked.is_empty() && !rescue_after_choice {
            checked.sort_unstable();
            checked.dedup();
            self.checked_retry(batch, &checked, tolerance, &mut result, &mut profile, 0)?;
            // Lanes still refused: refused boxes by the small-Gram expansion.
            checked.retain(|&lane| result.statuses[lane] != LaneStatus::Success);
            if !checked.is_empty() {
                self.checked_retry(batch, &checked, tolerance, &mut result, &mut profile, 1)?;
                // Lanes still refused: the alternative PV reduction alone
                // (each method wins on different lanes; the rotation test
                // decides, as COLLIER's error estimates choose its method).
                checked.retain(|&lane| result.statuses[lane] != LaneStatus::Success);
                if !checked.is_empty() {
                    self.checked_retry(batch, &checked, tolerance, &mut result, &mut profile, 2)?;
                }

            }
        }
        // Every lane still refused on a conditioning estimate (gate or
        // rotation test): every box and triangle by the expansions or the
        // alternative PV reduction, as COLLIER chooses its method per box.
        // Not for heptagon (and larger) lanes whose orders disagreed after
        // passing the gate: on the 500 stability heptagons this stage
        // certified only gate-refused lanes, and on the opening N7R7 batch of
        // 32 (precision rescue disabled) it kept none of its 12 such lanes in
        // 0.51 s of 0.80 s.
        if checkable && binary64_retries && !rescue_after_choice {
            let rest: Vec<usize> = (0..batch.rows)
                .filter(|&lane| {
                    matches!(result.statuses[lane], LaneStatus::Unstable(_))
                        && conditions[lane].is_some()
                        && !(n >= 7 && order_refused[lane])
                })
                .collect();
            if !rest.is_empty() {
                self.checked_retry(batch, &rest, tolerance, &mut result, &mut profile, 3)?;
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
        // Refused lanes keep a binary64 array, flagged (COLLIER-style) with an
        // estimated error. Lanes without one so far get the expansion stage,
        // then one evaluation without any gate (each with its rotated copy),
        // then the binary64 symbolic engine.
        let stride = result.basis.len() * 4 * result.components;
        let width = batch.values.len() / batch.rows.max(1);
        let unresolved = |result: &BatchResult<T>, candidates: &[Vec<(Vec<Complex<T>>, f64)>]| -> Vec<usize> {
            (0..batch.rows)
                .filter(|&lane| matches!(result.statuses[lane], LaneStatus::Unstable(_)) && candidates[lane].is_empty())
                .collect()
        };
        for stage in 0..2 {
            // The gate-free stage covers every refused lane of five or more
            // points: one of its cyclic orders is often far more accurate
            // than the retries' arrays (stability hexagons left best-effort:
            // E/A median 3.3e-11 -> 1.9e-13 by the order closest to two
            // others; COLLIER 1.2e-12).
            let missing = if stage == 1 && n >= 5 {
                (0..batch.rows).filter(|&lane| matches!(result.statuses[lane], LaneStatus::Unstable(_))).collect()
            } else {
                unresolved(&result, &self.candidates)
            };
            if missing.is_empty() {
                continue;
            }
            let slot = if stage == 0 { &mut self.expanded } else { &mut self.open };
            if slot.is_none() {
                let mut workspace = self
                    .plan
                    .direct
                    .workspace_with_masters(self.prototype.clone(), masters_for(self.options))?;
                if stage == 0 {
                    if rotation_tested(self.plan.direct.settings()) {
                        workspace.set_stability_threshold(from_binary64(&self.prototype, ROTATION_GATE));
                    }
                    workspace.enable_gram_expansion(GRAM_BOOST);
                } else {
                    workspace.set_stability_threshold(self.prototype.zero());
                }
                workspace.set_profiling(self.profiling);
                workspace.set_shared_subsets(self.shared.clone());
                *slot = Some(Box::new(workspace));
            }
            let workspace = slot.as_mut().unwrap();
            let mut values = Vec::with_capacity(missing.len() * width);
            for &lane in &missing {
                values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
            }
            let sub = KinematicsBatch { rows: missing.len(), components: batch.components, values };
            // The integral does not depend on the order of its propagators:
            // the gate-free stage tries every cyclic relabelling (the others
            // may avoid an exactly singular subsystem of this order).
            let orders = if stage == 0 { 2 } else { n.max(2) };
            let mut evaluations = Vec::with_capacity(orders);
            let mut relabelled = sub;
            for order in 0..orders {
                if order > 0 {
                    relabelled = rotated(&relabelled, n);
                }
                evaluations.push(workspace.reduce_batch(&relabelled)?);
            }
            for (k, &lane) in missing.iter().enumerate() {
                let mu = batch.values[lane * width + 5 * n].re.to_f64();
                let good: Vec<&[Complex<T>]> = evaluations
                    .iter()
                    .filter(|evaluation| evaluation.statuses[k] == LaneStatus::Success)
                    .map(|evaluation| &evaluation.values[k * stride..(k + 1) * stride])
                    .collect();
                for (i, a) in good.iter().enumerate() {
                    // E/A to its second-nearest other order (with fewer than
                    // three orders, to the other one; NaN: the only one).
                    let mut distances: Vec<f64> = (0..good.len())
                        .filter(|&j| j != i)
                        .map(|j| lane_difference(&result.basis, a, good[j], mu))
                        .collect();
                    distances.sort_by(f64::total_cmp);
                    let difference = match distances.len() {
                        0 => f64::NAN,
                        1 => distances[0],
                        _ => distances[1],
                    };
                    self.candidates[lane].push((a.to_vec(), difference));
                    if stage == 0 {
                        // The expansion stage keeps its first order only.
                        break;
                    }
                }
            }
        }
        let missing = unresolved(&result, &self.candidates);
        // (Not for seven points: their symbolic plans are too large to prepare.)
        if !missing.is_empty() && n <= 6 {
            let mut values = Vec::with_capacity(missing.len() * width);
            for &lane in &missing {
                values.extend_from_slice(&batch.values[lane * width..(lane + 1) * width]);
            }
            let sub = KinematicsBatch { rows: missing.len(), components: batch.components, values };
            if let Ok(fallback) = self.fallback_workspace() {
                if let Ok(symbolic) = fallback.reduce_batch(&sub) {
                    for (k, &lane) in missing.iter().enumerate() {
                        result.fallback_used[lane] = true;
                        if symbolic.statuses[k] == LaneStatus::Success {
                            let a = symbolic.values[k * stride..(k + 1) * stride].to_vec();
                            self.candidates[lane].push((a, f64::NAN));
                        }
                    }
                }
            }
        }
        for lane in 0..batch.rows {
            if result.statuses[lane] == LaneStatus::Success {
                continue;
            }
            let candidates = std::mem::take(&mut self.candidates[lane]);
            let key = |d: f64| if d.is_nan() { f64::INFINITY } else { d };
            let Some(best) = (0..candidates.len()).min_by(|&a, &b| key(candidates[a].1).total_cmp(&key(candidates[b].1)))
            else {
                continue;
            };
            let mu = batch.values[lane * width + 5 * n].re.to_f64();
            let (values, own) = &candidates[best];
            // The rotated copy can share the chosen array's error: the
            // nearest array of another binary64 stage bounds the estimate
            // from below.
            let cross = (0..candidates.len())
                .filter(|&other| other != best)
                .map(|other| lane_difference(&result.basis, values, &candidates[other].0, mu))
                .fold(f64::INFINITY, f64::min);
            let estimate = match (own.is_nan(), cross.is_finite()) {
                (false, true) => own.max(cross),
                (false, false) => *own,
                (true, true) => cross,
                (true, false) => f64::NAN,
            };
            result.values[lane * stride..(lane + 1) * stride].clone_from_slice(values);
            result.best_effort[lane] = true;
            result.estimated_errors[lane] = (!estimate.is_nan()).then_some(estimate);
            self.diagnostics[lane].push(match result.estimated_errors[lane] {
                Some(estimate) => format!("best-effort binary64 array returned, estimated E/A {estimate:e}"),
                None => "best-effort binary64 array returned (no estimate)".into(),
            });
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
