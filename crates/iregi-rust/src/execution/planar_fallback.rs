//! Optional outer retry through the checked constant-one planar interface.

use super::*;
use crate::{
    CertificateExecutionStats, CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace,
    ReductionSettings,
};

#[cfg(test)]
mod policy_composition_tests {
    use super::*;
    use crate::Algorithm;

    #[test]
    fn composition_planar_retry_forwards_policies_to_lazy_masked_charts() {
        let mut policies = vec![MasterExecutionPolicy::Symbolica];
        if MasterExecutionPolicy::NativeF64Abc.validate_available().is_ok() {
            policies.push(MasterExecutionPolicy::NativeF64Abc);
        }
        for masters in policies {
            let mut settings = ReductionSettings::new(5, 0);
            settings.algorithm = Algorithm::PassarinoVeltman;
            let retry = PlanarFallback::new(settings, WorkspacePolicy::SIX_POINT_CHILD, masters);
            assert_eq!(retry.policy, WorkspacePolicy::SIX_POINT_CHILD);
            assert_eq!(retry.master_policy, masters);
            assert!(retry.workspace.is_none());
            for shape in [None, Some(vec![vec![0]])] {
                let workspace = retry.prepare(0., &shape, 53, false).unwrap();
                assert_eq!(workspace.dual_shape(), shape.as_deref());
                crate::certified_planar::policy_composition_tests::assert_chart_context(&workspace, masters);
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct PlanarFallback<T: NativeFloat> {
    settings: ReductionSettings,
    workspace: Option<Result<Box<CertifiedPlanarWorkspace<T>>, String>>,
    policy: WorkspacePolicy,
    master_policy: MasterExecutionPolicy,
}

impl<T: NativeFloat> PlanarFallback<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn new(settings: ReductionSettings, policy: WorkspacePolicy, master_policy: MasterExecutionPolicy) -> Self {
        Self {
            settings,
            workspace: None,
            policy,
            master_policy,
        }
    }

    pub(super) fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if let Some(Ok(workspace)) = &mut self.workspace {
            workspace.set_accuracy_precision(bits)?;
        }
        Ok(())
    }

    pub(super) fn set_profiling(&mut self, enabled: bool) {
        if let Some(Ok(workspace)) = &mut self.workspace {
            workspace.set_profiling(enabled);
        }
    }

    fn prepare(
        &self,
        prototype: T,
        shape: &Option<Vec<Vec<usize>>>,
        accuracy_bits: u32,
        profiling: bool,
    ) -> Result<Box<CertifiedPlanarWorkspace<T>>, String> {
        let plan =
            CertifiedPlanarReductionPlan::new_for_six_point_child(self.settings.clone()).map_err(|e| e.to_string())?;
        let mut workspace = match shape {
            None => CertifiedPlanarWorkspace::new_with_policies(&plan, prototype, self.policy, self.master_policy),
            Some(shape) => CertifiedPlanarWorkspace::new_dual_with_policies(
                &plan,
                prototype,
                HyperDual::<Complex<Rational>>::new(shape.clone()),
                self.policy,
                self.master_policy,
            ),
        }
        .map_err(|e| e.to_string())?;
        workspace
            .set_accuracy_precision(accuracy_bits)
            .map_err(|e| e.to_string())?;
        workspace.set_profiling(profiling);
        Ok(Box::new(workspace))
    }
}

impl<T: NativeFloat> Workspace<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Current-call planar certificate work, including executed analytic and
    /// Hybrid children. No cached child's previous-call counters are harvested.
    pub(crate) fn last_planar_certificate_execution(&self) -> CertificateExecutionStats {
        self.planar_certificate_stats
    }

    pub(super) fn retry_planar(
        &mut self,
        batch: &KinematicsBatch<T>,
        result: &mut BatchResult<T>,
    ) -> Result<(), ReductionError> {
        let Some(state) = &mut self.planar_fallback else {
            return Ok(());
        };
        let input_width = checked_mul(self.parameters, self.components)?;
        let output_width = checked_mul(checked_mul(self.basis.len(), 4)?, self.components)?;
        // Recheck the original rows: retained active-lane lists can omit a
        // failed analytic profile or a lane rejected by a numerical guard.
        let lanes: Vec<_> = batch
            .values
            .chunks_exact(input_width)
            .enumerate()
            .filter_map(|(lane, row)| {
                (result.statuses[lane] != LaneStatus::Success
                    && validate_input_row(row, &self.layout, self.components).is_none())
                .then_some(lane)
            })
            .collect();
        if lanes.is_empty() {
            return Ok(());
        }
        self.stats.fallback_rows = self
            .stats
            .fallback_rows
            .checked_add(lanes.len())
            .ok_or_else(size_error)?;
        for &lane in &lanes {
            result.fallback_used[lane] = true;
        }
        if state.workspace.is_none() {
            let started = self.profiling.then(Instant::now);
            state.workspace = Some(state.prepare(
                self.prototype.clone(),
                &self.shape,
                self.accuracy_bits,
                self.profiling,
            ));
            record_time(started, &mut self.profile.arguments);
        }
        let workspace = match state.workspace.as_mut().expect("prepared planar retry") {
            Ok(workspace) => workspace,
            Err(error) => {
                for &lane in &lanes {
                    self.diagnostics[lane].push(format!(
                        "planar fallback preparation failed; original failure retained: {error}"
                    ));
                }
                return Ok(());
            }
        };
        let mut packed = KinematicsBatch {
            rows: lanes.len(),
            components: self.components,
            values: Vec::with_capacity(checked_mul(lanes.len(), input_width)?),
        };
        for &lane in &lanes {
            packed
                .values
                .extend_from_slice(&batch.values[lane * input_width..(lane + 1) * input_width]);
        }
        let retried = workspace.reduce_batch(&packed);
        // Harvest certificate work before any returned child error or later
        // accounting check. Chart workspaces cannot recursively own this retry.
        add_certificate_work(&mut self.planar_certificate_stats, workspace.last_certificate_execution())?;
        self.stats.add_policy_child_work(workspace.last_execution())?;
        if let Some(profile) = workspace.last_profile() {
            self.profile.add_stages(profile);
        }
        for (compact, &lane) in lanes.iter().enumerate() {
            if let Some(notes) = workspace.last_diagnostics().get(compact) {
                self.diagnostics[lane].extend_from_slice(notes);
            }
            if workspace.last_chart_attempts().get(compact) == Some(&true) {
                result.analytic_used[lane] = true;
            }
        }
        let retried = match retried {
            Ok(retried) => retried,
            Err(error) => {
                for &lane in &lanes {
                    self.diagnostics[lane].push(format!(
                        "planar fallback evaluation failed; original failure retained: {error}"
                    ));
                }
                return Ok(());
            }
        };
        // A component count alone does not establish a Dual algebra or the
        // normalization of coefficients. Validate the whole result before any
        // failed parent lane can become successful.
        if retried.basis != self.basis
            || retried.metadata() != &self.metadata
            || retried.components != self.components
            || retried.rows() != lanes.len()
            || retried.fallback_used.len() != lanes.len()
            || retried.analytic_used.len() != lanes.len()
            || retried.values.len() != checked_mul(lanes.len(), output_width)?
        {
            for &lane in &lanes {
                self.diagnostics[lane].push("planar fallback returned inconsistent basis, metadata or buffer dimensions; original failure retained".into());
            }
            return Ok(());
        }
        for (compact, &lane) in lanes.iter().enumerate() {
            result.analytic_used[lane] |= retried.analytic_used[compact];
            result.fallback_used[lane] |= retried.fallback_used[compact];
            match retried.lane(compact) {
                Ok(values) => {
                    result.values[lane * output_width..(lane + 1) * output_width]
                        .clone_from_slice(values);
                    result.statuses[lane] = LaneStatus::Success;
                    result.inherit_primal_envelope(
                        lane, &retried, compact, PrimalEnvelopeRoute::AutomaticPlanar,
                    );
                    self.stats.fallback_successes = self
                        .stats
                        .fallback_successes
                        .checked_add(1)
                        .ok_or_else(size_error)?;
                }
                Err(status) => self.diagnostics[lane].push(format!(
                    "planar fallback returned {status:?}; original failure retained"
                )),
            }
        }
        Ok(())
    }
}

