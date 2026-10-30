//! Binary64 default: the direct numerical engine with per-lane fallback.
//!
//! The direct engine evaluates the validated PV recurrence numerically. Lanes
//! it cannot reduce reliably (ill-conditioned Gram, Cayley or six-point
//! systems without a denominator identity, or scalar-master failures) are
//! rerun on the existing symbolic engine, whose plan and workspace are only
//! prepared when such a lane first occurs. Invalid inputs are rejected by the
//! direct engine itself and are not retried.

use std::sync::{Arc, OnceLock};

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

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

/// Counters of the most recent direct-engine call.
#[derive(Clone, Debug, Default)]
pub struct DirectExecutionStats {
    pub input_rows: usize,
    pub direct_successes: usize,
    pub fallback_rows: usize,
    pub fallback_successes: usize,
    pub fallback_preparation_error: Option<String>,
    pub fallback: Option<Box<TensorEngineExecution>>,
}

#[derive(Clone)]
pub(crate) struct DirectHybridWorkspace<T: NativeFloat> {
    plan: Arc<DirectHybridPlan>,
    direct: DirectWorkspace<T>,
    prototype: T,
    options: TensorWorkspaceOptions,
    accuracy_bits: Option<u32>,
    profiling: bool,
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

    pub(crate) fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let started = self.profiling.then(std::time::Instant::now);
        let mut result = self.direct.reduce_batch(batch)?;
        self.last = DirectExecutionStats {
            input_rows: batch.rows,
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
        if !retry.is_empty() {
            self.last.fallback_rows = retry.len();
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
