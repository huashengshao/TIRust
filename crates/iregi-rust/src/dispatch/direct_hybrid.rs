//! Binary64 default: the direct numerical engine with per-lane fallback.
//!
//! The direct engine evaluates the validated PV recurrence numerically. Lanes
//! refused on conditioning (ill-conditioned Gram, Cayley or six-point systems
//! without a denominator identity) are first rerun by the same direct engine
//! at higher precision: double-double, then 256 and 512 MPFR bits, starting at
//! the tier the binary64 conditioning estimate calls for. The binary64 inputs
//! are lifted exactly, so a tier evaluates the caller's kinematics, and the
//! accepted coefficients are rounded once. Lanes no tier accepts, and
//! scalar-master failures, are rerun on the existing symbolic engine, whose
//! plan and workspace are only prepared when such a lane first occurs.
//! Invalid inputs are rejected by the direct engine itself and are not
//! retried.

use std::sync::{Arc, OnceLock};

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{
    domains::float::{Complex, DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

use super::{TensorEngineExecution, TensorReductionPlan, TensorWorkspace, TensorWorkspaceOptions};
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

fn rescue_threshold(bits: u32) -> f64 {
    2f64.powi(-((bits - RESCUE_HEADROOM) as i32))
}

/// First tier that admits the binary64 conditioning estimate with margin. An
/// exactly singular binary64 pivot carries no estimate and starts at 256 bits.
fn first_tier(condition: f64) -> usize {
    if condition > 0.0 {
        RESCUE_BITS
            .iter()
            .position(|&bits| condition >= rescue_threshold(bits - RESCUE_MARGIN))
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
        Ok(Self {
            plan,
            direct,
            prototype,
            options,
            accuracy_bits: None,
            profiling: options.profiling,
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

    pub(crate) fn set_profiling(&mut self, enabled: bool) -> Result<(), ReductionError> {
        self.profiling = enabled;
        self.direct.set_profiling(enabled);
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
        result: &mut BatchResult<T>,
        profile: &mut ExecutionProfile,
    ) -> Result<Vec<usize>, ReductionError> {
        let conditions = self.direct.last_conditions().to_vec();
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
            let outcomes = self.rescue_tier(tier, batch, &lanes, result, profile)?;
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

    pub(crate) fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let started = self.profiling.then(std::time::Instant::now);
        let mut result = self.direct.reduce_batch(batch)?;
        self.last = DirectExecutionStats {
            input_rows: batch.rows,
            master_rows: self.direct.last_master_rows(),
            ..DirectExecutionStats::default()
        };
        self.diagnostics.clear();
        self.diagnostics.resize(batch.rows, Vec::new());
        let retry: Vec<usize> = result
            .statuses
            .iter()
            .enumerate()
            .filter(|(_, status)| {
                matches!(status, LaneStatus::Unstable(_) | LaneStatus::MasterFailure(_))
            })
            .map(|(lane, _)| lane)
            .collect();
        self.last.direct_successes = result
            .statuses
            .iter()
            .filter(|status| **status == LaneStatus::Success)
            .count();
        for &lane in &retry {
            if let LaneStatus::Unstable(note) | LaneStatus::MasterFailure(note) =
                &result.statuses[lane]
            {
                self.diagnostics[lane].push(format!("direct engine: {note}; fallback requested"));
            }
        }
        let direct_profile = self.direct.last_profile();
        let mut profile = ExecutionProfile {
            arguments: direct_profile.geometry,
            scalar_masters: direct_profile.masters,
            reduction: direct_profile.recurrence + direct_profile.gather + direct_profile.output,
            total: Default::default(),
        };
        self.last.fallback_rows = retry.len();
        let retry = if retry.is_empty() {
            retry
        } else {
            self.precision_rescue(batch, &retry, &mut result, &mut profile)?
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
