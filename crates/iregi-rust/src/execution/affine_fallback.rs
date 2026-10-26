//! Automatic checked affine prepass before parent arguments/master activation.
use super::*;
use crate::{
    ReductionSettings,
    certified_affine::{CertifiedAffinePlan, CertifiedAffineWorkspace},
};

#[derive(Clone)]
pub(super) struct AffineFallback<T: NativeFloat> {
    settings: ReductionSettings,
    workspace: Option<Result<Box<CertifiedAffineWorkspace<T>>, String>>,
    policy: WorkspacePolicy,
    master_policy: MasterExecutionPolicy,
}

impl<T: NativeFloat> Workspace<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Exact analytic prepass. Raw inputs are validated, but no parent
    /// argument, Gram inverse or scalar-master instruction has run yet.
    pub(super) fn dispatch_affine(
        &mut self,
        batch: &KinematicsBatch<T>,
        result: &mut BatchResult<T>,
    ) -> Result<(), ReductionError> {
        let Some(state) = &mut self.affine_fallback else {
            return Ok(());
        };
        let lanes = self.valid_lanes.clone();
        if lanes.is_empty() {
            return Ok(());
        }
        let input_width = checked_mul(self.parameters, self.components)?;
        let output_width = checked_mul(checked_mul(self.basis.len(), 4)?, self.components)?;
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
        let workspace = match state.workspace.as_mut().expect("prepared affine prepass") {
            Ok(workspace) => workspace,
            Err(error) => {
                for &lane in &lanes {
                    self.diagnostics[lane].push(format!(
                        "affine prepass unavailable; ordinary profile retained: {error}"
                    ));
                }
                return Ok(());
            }
        };
        let packed = KinematicsBatch {
            rows: lanes.len(),
            components: self.components,
            values: self.input.clone(),
        };
        let reduced = workspace.reduce_batch(&packed);
        add_certificate_work(
            &mut self.planar_certificate_stats,
            workspace.last_certificate_execution(),
        )?;
        self.stats
            .add_policy_child_work(workspace.last_execution())?;
        if let Some(profile) = workspace.last_profile() {
            self.profile.add_stages(profile);
        }
        for (compact, &lane) in lanes.iter().enumerate() {
            if let Some(notes) = workspace.last_diagnostics().get(compact) {
                self.diagnostics[lane].extend_from_slice(notes);
            }
        }
        let reduced = reduced?;
        if reduced.basis != self.basis
            || reduced.metadata() != &self.metadata
            || reduced.components != self.components
            || reduced.rows() != lanes.len()
            || reduced.fallback_used.len() != lanes.len()
            || reduced.analytic_used.len() != lanes.len()
            || reduced.values.len() != checked_mul(lanes.len(), output_width)?
            || workspace.last_selected().len() != lanes.len()
        {
            return Err(ReductionError::Evaluation(
                "affine prepass returned inconsistent metadata/buffers".into(),
            ));
        }
        let mut ordinary = Vec::new();
        for (compact, &lane) in lanes.iter().enumerate() {
            if !workspace.last_selected()[compact] {
                // These lanes never entered an affine numeric recipe. Their
                // original exact input and ordinary algorithm are retained.
                ordinary.push(compact);
                continue;
            }
            result.analytic_used[lane] = true;
            result.fallback_used[lane] = reduced.fallback_used[compact];
            result.statuses[lane] = reduced.statuses[compact].clone();
            if let Ok(values) = reduced.lane(compact) {
                result.values[lane * output_width..(lane + 1) * output_width]
                    .clone_from_slice(values);
                result.inherit_primal_envelope(
                    lane,
                    &reduced,
                    compact,
                    PrimalEnvelopeRoute::AutomaticAffine,
                );
            }
            // Accepted rank-deficient lanes never reach the parent Gram path,
            // even if their independently guarded child reduction failed.
        }
        self.retain_lanes(&ordinary, input_width, 0);
        Ok(())
    }
}

impl<T: NativeFloat> AffineFallback<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn new(
        settings: ReductionSettings,
        policy: WorkspacePolicy,
        master_policy: MasterExecutionPolicy,
    ) -> Self {
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
    ) -> Result<Box<CertifiedAffineWorkspace<T>>, String> {
        let plan = CertifiedAffinePlan::new(self.settings.clone()).map_err(|e| e.to_string())?;
        let mut workspace = CertifiedAffineWorkspace::new(
            &plan,
            prototype,
            shape.clone(),
            self.policy,
            self.master_policy,
        )
        .map_err(|e| e.to_string())?;
        workspace
            .set_accuracy_precision(accuracy_bits)
            .map_err(|e| e.to_string())?;
        workspace.set_profiling(profiling);
        Ok(Box::new(workspace))
    }
}
