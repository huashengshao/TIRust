//! Six-point Hybrid orchestration. Existing Symbolica PV and IBP engines own
//! all mathematical operations. This module only batches, guards and scatters.
use super::*;
use crate::{
    Algorithm, LaneStatus, ResultMetadata, execution::WorkspacePolicy, types::PrimalEnvelopeRoute,
};

#[derive(Clone, Debug)]
pub(super) struct SixHybridPlan {
    pub(super) primary: Arc<SixPointReductionPlan>,
    fallback: LazyIbpPlan,
}
#[derive(Clone, Debug)]
struct LazyIbpPlan {
    settings: ReductionSettings,
    prepared: Arc<OnceLock<Result<Arc<ReductionPlan>, String>>>,
}
impl LazyIbpPlan {
    fn new(mut settings: ReductionSettings) -> Self {
        settings.algorithm = Algorithm::Ibp;
        Self {
            settings,
            prepared: Arc::new(OnceLock::new()),
        }
    }
    fn get(&self) -> Result<Arc<ReductionPlan>, String> {
        self.prepared
            .get_or_init(|| {
                let plan = ReductionPlan::new(self.settings.clone()).map_err(|e| e.to_string())?;
                require_ibp(&plan)?;
                Ok(Arc::new(plan))
            })
            .clone()
    }
}
fn require_ibp(plan: &ReductionPlan) -> Result<(), String> {
    if plan.algorithm_used() != Algorithm::Ibp {
        return Err(format!(
            "genuine IBP fallback unavailable: requested IBP preparation uses {:?} tensor completion; this is not an independent IBP numerical fallback",
            plan.algorithm_used()
        ));
    }
    Ok(())
}
impl SixHybridPlan {
    pub(super) fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        if settings.algorithm != Algorithm::Hybrid {
            return Err(ReductionError::InvalidInput(
                "six-point Hybrid plan requires requested Hybrid".into(),
            ));
        }
        let fallback = LazyIbpPlan::new(settings.clone());
        let mut pv = settings;
        pv.algorithm = Algorithm::PassarinoVeltman;
        Ok(Self {
            primary: Arc::new(SixPointReductionPlan::new(pv)?),
            fallback,
        })
    }
}

#[derive(Clone)]
enum FallbackState<T: NativeFloat> {
    Unprepared,
    Ready(Box<Workspace<T>>),
    Unavailable(String),
}
#[derive(Clone, Debug, Default)]
struct HybridStats {
    primary: SixPointExecutionStats,
    fallback: ExecutionStats,
    fallback_certificate_execution: CertificateExecutionStats,
    fallback_rows: usize,
    fallback_successes: usize,
    fallback_batches: usize,
    fallback_preparation_attempts: usize,
    fallback_preparation_error: Option<String>,
}
impl HybridStats {
    fn detail(&self) -> TensorEngineExecution {
        TensorEngineExecution::SixPointHybrid {
            primary: self.primary.clone(),
            fallback: self.fallback,
            fallback_certificate_execution: self.fallback_certificate_execution,
            fallback_rows: self.fallback_rows,
            fallback_successes: self.fallback_successes,
            fallback_batches: self.fallback_batches,
            fallback_preparation_attempts: self.fallback_preparation_attempts,
            fallback_preparation_error: self.fallback_preparation_error.clone(),
        }
    }
}
struct FallbackOutcome<T: NativeFloat> {
    result: Result<BatchResult<T>, String>,
    execution: ExecutionStats,
    certificate: CertificateExecutionStats,
    diagnostics: Vec<Vec<String>>,
    batches: usize,
    preparation_attempts: usize,
    preparation_error: Option<String>,
}
impl<T: NativeFloat> FallbackOutcome<T> {
    fn unavailable(error: String, attempts: usize) -> Self {
        Self {
            result: Err(error.clone()),
            execution: Default::default(),
            certificate: Default::default(),
            diagnostics: Vec::new(),
            batches: 0,
            preparation_attempts: attempts,
            preparation_error: Some(error),
        }
    }
}

#[derive(Clone)]
pub(super) struct SixHybridWorkspace<T: NativeFloat> {
    plan: Arc<SixHybridPlan>,
    primary: SixPointWorkspace<T>,
    fallback: FallbackState<T>,
    prototype: T,
    accuracy_bits: u32,
    stats: HybridStats,
    diagnostics: Vec<Vec<String>>,
    profiling: bool,
    profile: ExecutionProfile,
}
impl<T: NativeFloat> SixHybridWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn new(plan: Arc<SixHybridPlan>, prototype: T) -> Result<Self, ReductionError> {
        let primary = SixPointWorkspace::new(&plan.primary, prototype.clone())?;
        Ok(Self {
            plan,
            primary,
            fallback: FallbackState::Unprepared,
            accuracy_bits: prototype.get_precision(),
            prototype,
            stats: Default::default(),
            diagnostics: Vec::new(),
            profiling: false,
            profile: ExecutionProfile::default(),
        })
    }
    pub(super) fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        self.primary.set_accuracy_precision(bits)?;
        if let FallbackState::Ready(fallback) = &mut self.fallback {
            fallback.set_accuracy_precision(bits)?;
        }
        self.accuracy_bits = bits;
        Ok(())
    }
    pub(super) fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        self.primary.set_profiling(enabled);
        if let FallbackState::Ready(fallback) = &mut self.fallback {
            fallback.set_profiling(enabled);
        }
    }
    pub(super) fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }
    pub(super) fn component_count(&self) -> usize {
        self.primary.component_count()
    }
    pub(super) fn binary_precision(&self) -> u32 {
        self.primary.binary_precision()
    }
    pub(super) fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }
    pub(super) fn engine_execution(&self) -> TensorEngineExecution {
        self.stats.detail()
    }
    pub(super) fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.profile = ExecutionProfile::default();
        let started = self.profiling.then(std::time::Instant::now);
        let result = self.reduce_batch_impl(batch);
        if let Some(started) = started {
            self.profile.total = started.elapsed();
        }
        result
    }
    fn reduce_batch_impl(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.stats = HybridStats::default();
        self.diagnostics.clear();
        let result = self.primary.reduce_batch(batch);
        if let Some(profile) = self.primary.last_profile() {
            add_profile_stages(&mut self.profile, profile);
        }
        self.stats.primary = self.primary.last_execution().clone();
        self.diagnostics
            .extend_from_slice(self.primary.last_diagnostics());
        let mut result = result?;
        let expected_metadata = ResultMetadata::new(
            self.plan.primary.settings().convention,
            self.plan.primary.settings().tensor_weights,
            None,
        );
        validate_layout(
            &result,
            self.plan.primary.basis(),
            &expected_metadata,
            self.component_count(),
            batch.rows,
        )
        .map_err(ReductionError::Evaluation)?;
        let plan = &self.plan.fallback;
        let state = &mut self.fallback;
        let prototype = &self.prototype;
        let accuracy = self.accuracy_bits;
        let profiling = self.profiling;
        let mut fallback_profile = None;
        let retried = retry_unstable(
            batch,
            &mut result,
            &mut self.diagnostics,
            &mut self.stats,
            |packed| {
                let mut attempts = 0;
                if matches!(&*state, FallbackState::Unprepared) {
                    attempts = 1;
                    let prepared = plan.get().and_then(|plan| {
                        let mut workspace = Workspace::new_with_policies(
                            &plan,
                            prototype.clone(),
                            WorkspacePolicy::RETAIN_ENVELOPES,
                            MasterExecutionPolicy::Symbolica,
                        )
                        .map_err(|e| e.to_string())?;
                        workspace
                            .set_accuracy_precision(accuracy)
                            .map_err(|e| e.to_string())?;
                        workspace.set_profiling(profiling);
                        Ok(workspace)
                    });
                    *state = match prepared {
                        Ok(workspace) => FallbackState::Ready(Box::new(workspace)),
                        Err(error) => FallbackState::Unavailable(error),
                    };
                }
                match state {
                    FallbackState::Unavailable(error) => {
                        FallbackOutcome::unavailable(error.clone(), attempts)
                    }
                    FallbackState::Ready(workspace) => {
                        let result = workspace.reduce_batch(packed).map_err(|e| e.to_string());
                        fallback_profile = workspace.last_profile();
                        FallbackOutcome {
                            result,
                            execution: workspace.last_execution(),
                            certificate: workspace.last_planar_certificate_execution(),
                            diagnostics: workspace.last_diagnostics().to_vec(),
                            batches: 1,
                            preparation_attempts: attempts,
                            preparation_error: None,
                        }
                    }
                    FallbackState::Unprepared => unreachable!("fallback preparation changes state"),
                }
            },
        );
        if let Some(profile) = fallback_profile {
            add_profile_stages(&mut self.profile, profile);
        }
        retried?;
        Ok(result)
    }
}

fn add_profile_stages(total: &mut ExecutionProfile, child: ExecutionProfile) {
    total.arguments += child.arguments;
    total.scalar_masters += child.scalar_masters;
    total.reduction += child.reduction;
}
fn dimension_error() -> ReductionError {
    ReductionError::InvalidInput("six-point Hybrid batch dimensions overflow".into())
}
fn validate_layout<T: NativeFloat>(
    result: &BatchResult<T>,
    basis: &[Monomial],
    metadata: &ResultMetadata,
    components: usize,
    rows: usize,
) -> Result<(), String> {
    let count = rows
        .checked_mul(basis.len())
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_mul(components))
        .ok_or("six-point Hybrid result dimensions overflow")?;
    let sidecars = result.primal_envelopes.len();
    if components == 0
        || result.basis != basis
        || result.metadata() != metadata
        || result.components != components
        || result.rows() != rows
        || result.values.len() != count
        || result.fallback_used.len() != rows
        || result.analytic_used.len() != rows
        || result.primal_envelope_errors.len() != sidecars
        || (sidecars != 0 && sidecars != rows)
    {
        return Err("six-point Hybrid complete result layout mismatch".into());
    }
    Ok(())
}

/// One callback for the complete selected batch, including analytic Unstable
/// lanes. Invalid input and master failures are not silently retried. A global
/// fallback error cannot discard successful neighbors from the PV primary.
fn retry_unstable<T: NativeFloat>(
    batch: &KinematicsBatch<T>,
    result: &mut BatchResult<T>,
    diagnostics: &mut [Vec<String>],
    stats: &mut HybridStats,
    evaluate: impl FnOnce(&KinematicsBatch<T>) -> FallbackOutcome<T>,
) -> Result<(), ReductionError> {
    let components = result.components;
    validate_layout(
        result,
        &result.basis,
        result.metadata(),
        components,
        batch.rows,
    )
    .map_err(ReductionError::Evaluation)?;
    let input_width = 31usize
        .checked_mul(components)
        .ok_or_else(dimension_error)?;
    let output_width = result
        .basis
        .len()
        .checked_mul(4)
        .and_then(|n| n.checked_mul(components))
        .ok_or_else(dimension_error)?;
    if diagnostics.len() != batch.rows
        || batch.components != components
        || batch.rows.checked_mul(input_width) != Some(batch.values.len())
    {
        return Err(ReductionError::InvalidInput(
            "six-point Hybrid input layout mismatch".into(),
        ));
    }
    let lanes: Vec<_> = result
        .statuses
        .iter()
        .enumerate()
        .filter_map(|(lane, status)| matches!(status, LaneStatus::Unstable(_)).then_some(lane))
        .collect();
    if lanes.is_empty() {
        return Ok(());
    }
    let mut packed = KinematicsBatch {
        rows: lanes.len(),
        components,
        values: Vec::with_capacity(
            lanes
                .len()
                .checked_mul(input_width)
                .ok_or_else(dimension_error)?,
        ),
    };
    for &lane in &lanes {
        packed
            .values
            .extend_from_slice(&batch.values[lane * input_width..(lane + 1) * input_width]);
        result.fallback_used[lane] = true;
    }
    stats.fallback_rows = lanes.len();
    let outcome = evaluate(&packed);
    stats.fallback = outcome.execution;
    stats.fallback_certificate_execution = outcome.certificate;
    stats.fallback_batches = outcome.batches;
    stats.fallback_preparation_attempts = outcome.preparation_attempts;
    stats.fallback_preparation_error = outcome.preparation_error;
    let retried = outcome.result.and_then(|retried| {
        validate_layout(
            &retried,
            &result.basis,
            result.metadata(),
            components,
            lanes.len(),
        )?;
        Ok(retried)
    });
    for (compact, &lane) in lanes.iter().enumerate() {
        if let Some(notes) = outcome.diagnostics.get(compact) {
            diagnostics[lane].extend_from_slice(notes);
        }
        let primary = format!("{:?}", result.statuses[lane]);
        match &retried {
            Ok(retried) => {
                result.analytic_used[lane] |= retried.analytic_used[compact];
                match retried.lane(compact) {
                    Ok(values) => {
                        result.values[lane * output_width..(lane + 1) * output_width]
                            .clone_from_slice(values);
                        result.statuses[lane] = LaneStatus::Success;
                        result.inherit_primal_envelope(
                            lane,
                            retried,
                            compact,
                            PrimalEnvelopeRoute::Hybrid,
                        );
                        stats.fallback_successes =
                            stats.fallback_successes.checked_add(1).ok_or_else(|| {
                                ReductionError::Evaluation(
                                    "six-point Hybrid success counter overflow".into(),
                                )
                            })?;
                        diagnostics[lane].push("six-point Hybrid: specialized PV primary was unstable; genuine batched IBP fallback succeeded".into());
                    }
                    Err(status) => {
                        result.statuses[lane] = LaneStatus::Unstable(format!(
                            "six-point Hybrid PV primary {primary}; genuine IBP fallback returned {status:?}"
                        ));
                    }
                }
            }
            Err(error) => {
                result.statuses[lane] = LaneStatus::Unstable(format!(
                    "six-point Hybrid PV primary {primary}; IBP fallback failed or unavailable: {error}"
                ));
            }
        }
        if result.statuses[lane] != LaneStatus::Success {
            if !result.primal_envelopes.is_empty() {
                result.primal_envelopes[lane] = None;
                result.primal_envelope_errors[lane] = Some(format!("{:?}", result.statuses[lane]));
            }
            diagnostics[lane].push(format!("{:?}", result.statuses[lane]));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Convention, TensorWeights,
        types::{PrimalEnvelope, PrimalOperand, PrimalOperandKind},
    };

    fn batch(rows: usize, components: usize) -> KinematicsBatch<f64> {
        KinematicsBatch {
            rows,
            components,
            values: (0..rows * 31 * components)
                .map(|i| Complex::new(i as f64, -(i as f64)))
                .collect(),
        }
    }
    fn output(
        statuses: Vec<LaneStatus>,
        shape: Option<Vec<Vec<usize>>>,
        seed: f64,
    ) -> BatchResult<f64> {
        let components = shape.as_ref().map_or(1, Vec::len);
        let rows = statuses.len();
        let basis = vec![Monomial([0; 4])];
        let metadata =
            ResultMetadata::new(Convention::IregiNative, TensorWeights::Components, shape);
        let mut values: Vec<_> = (0..rows * 4 * components)
            .map(|i| Complex::new(seed + i as f64, 0.))
            .collect();
        for row in 0..rows {
            values[(row + 1) * 4 * components - 1].im = -0.;
        }
        let primal_envelopes = (0..rows)
            .map(|row| {
                Some(PrimalEnvelope {
                    basis: basis.clone(),
                    metadata: metadata.clone(),
                    components,
                    working_precision: 53,
                    accuracy_bits: 53,
                    operands: (0..4)
                        .map(|output_index| PrimalOperand {
                            output_index,
                            amplitude: Complex::new(1000. + output_index as f64, 0.),
                            kind: PrimalOperandKind::UncancelledMasters,
                        })
                        .collect(),
                    output_primals: (0..4).map(|i| values[(row * 4 + i) * components]).collect(),
                    component_amplitudes: Some(vec![Complex::new(2000., 0.); 4 * components]),
                    output_components: Some(
                        values[row * 4 * components..(row + 1) * 4 * components].to_vec(),
                    ),
                    routes: vec![PrimalEnvelopeRoute::Ordinary],
                })
            })
            .collect();
        BatchResult {
            basis,
            components,
            metadata,
            values,
            statuses,
            fallback_used: vec![false; rows],
            analytic_used: vec![false; rows],
            primal_envelopes,
            primal_envelope_errors: vec![None; rows],
        }
    }
    fn outcome(result: BatchResult<f64>) -> FallbackOutcome<f64> {
        FallbackOutcome {
            result: Ok(result),
            execution: ExecutionStats {
                input_rows: 2,
                argument_rows: 2,
                master_family_batches: 1,
                master_rows: 7,
                reduction_rows: 2,
                ..Default::default()
            },
            certificate: CertificateExecutionStats::default(),
            diagnostics: vec![vec!["actual IBP work".into()]; 2],
            batches: 1,
            preparation_attempts: 1,
            preparation_error: None,
        }
    }

    #[test]
    fn hybrid_profile_sums_primary_and_fallback_stages_without_nested_wall_time() {
        use std::time::Duration;

        let primary = ExecutionProfile {
            arguments: Duration::from_nanos(2),
            scalar_masters: Duration::from_nanos(3),
            reduction: Duration::from_nanos(5),
            total: Duration::from_nanos(20),
        };
        let fallback = ExecutionProfile {
            arguments: Duration::from_nanos(7),
            scalar_masters: Duration::from_nanos(11),
            reduction: Duration::from_nanos(13),
            total: Duration::from_nanos(50),
        };
        let mut combined = ExecutionProfile {
            total: Duration::from_nanos(100),
            ..Default::default()
        };
        add_profile_stages(&mut combined, primary);
        assert_eq!(combined.scalar_masters, primary.scalar_masters);
        assert_eq!(combined.total, Duration::from_nanos(100));
        add_profile_stages(&mut combined, fallback);
        assert_eq!(combined.arguments, Duration::from_nanos(9));
        assert_eq!(combined.scalar_masters, Duration::from_nanos(14));
        assert_eq!(combined.reduction, Duration::from_nanos(18));
        assert_eq!(combined.total, Duration::from_nanos(100));
        assert!(combined.arguments + combined.scalar_masters + combined.reduction <= combined.total);
    }

    #[test]
    fn unstable_lanes_are_one_full_component_batch_and_scatter_coupled_results() {
        let shape = Some(vec![vec![0, 0], vec![1, 0], vec![0, 1]]);
        let input = batch(5, 3);
        let before = input.values.clone();
        let mut primary = output(
            vec![
                LaneStatus::Success,
                LaneStatus::Unstable("PV one".into()),
                LaneStatus::InvalidInput("mu".into()),
                LaneStatus::Unstable("PV three".into()),
                LaneStatus::MasterFailure("master".into()),
            ],
            shape.clone(),
            10.,
        );
        primary.analytic_used[1] = true;
        let original = primary.clone();
        let retried = output(
            vec![
                LaneStatus::Success,
                LaneStatus::Unstable("IBP denominator".into()),
            ],
            shape,
            100.,
        );
        let expected = retried.clone();
        let mut calls = 0;
        let mut stats = HybridStats::default();
        let mut notes = vec![Vec::new(); 5];
        retry_unstable(&input, &mut primary, &mut notes, &mut stats, |packed| {
            calls += 1;
            assert_eq!((packed.rows, packed.components), (2, 3));
            let width = 31 * 3;
            assert_eq!(&packed.values[..width], &input.values[width..2 * width]);
            assert_eq!(&packed.values[width..], &input.values[3 * width..4 * width]);
            outcome(retried)
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(input.values, before);
        assert_eq!(primary.lane(0).unwrap(), original.lane(0).unwrap());
        assert_eq!(primary.lane(1).unwrap(), expected.lane(0).unwrap());
        assert_eq!(primary.statuses[2], original.statuses[2]);
        assert_eq!(primary.statuses[4], original.statuses[4]);
        assert!(
            matches!(&primary.statuses[3],LaneStatus::Unstable(e) if e.contains("PV three") && e.contains("IBP denominator"))
        );
        assert_eq!(primary.fallback_used, [false, true, false, true, false]);
        assert!(primary.analytic_used[1]);
        assert!(
            primary
                .lane(1)
                .unwrap()
                .last()
                .unwrap()
                .im
                .is_sign_negative()
        );
        assert!(primary.primal_envelope(3).is_err());
        let envelope = primary.primal_envelope(1).unwrap();
        assert_eq!(
            envelope.output_components,
            expected.primal_envelope(0).unwrap().output_components
        );
        assert_eq!(
            envelope.component_amplitudes,
            expected.primal_envelope(0).unwrap().component_amplitudes
        );
        assert_eq!(
            envelope.routes,
            [PrimalEnvelopeRoute::Ordinary, PrimalEnvelopeRoute::Hybrid]
        );
        assert_eq!(
            (
                stats.fallback_rows,
                stats.fallback_successes,
                stats.fallback_batches,
                stats.fallback_preparation_attempts
            ),
            (2, 1, 1, 1)
        );
        assert_eq!(stats.fallback.master_rows, 7);
    }

    #[test]
    fn failed_preparation_or_bad_layout_preserves_successful_neighbors() {
        for malformed in [false, true] {
            let input = batch(3, 1);
            let mut result = output(
                vec![
                    LaneStatus::Success,
                    LaneStatus::Unstable("PV".into()),
                    LaneStatus::InvalidInput("mu".into()),
                ],
                None,
                5.,
            );
            let original = result.clone();
            let mut notes = vec![Vec::new(); 3];
            let mut stats = HybridStats::default();
            retry_unstable(&input, &mut result, &mut notes, &mut stats, |_| {
                if malformed {
                    let mut wrong = output(vec![LaneStatus::Success], None, 100.);
                    wrong.values.pop();
                    outcome(wrong)
                } else {
                    FallbackOutcome::unavailable("IBP preparation uses PV completion".into(), 1)
                }
            })
            .unwrap();
            assert_eq!(result.lane(0).unwrap(), original.lane(0).unwrap());
            assert_eq!(result.statuses[2], original.statuses[2]);
            assert!(matches!(result.statuses[1], LaneStatus::Unstable(_)));
            assert_eq!(result.fallback_used, [false, true, false]);
            assert!(result.primal_envelope(1).is_err());
            if !malformed {
                assert_eq!(stats.fallback, ExecutionStats::default());
                assert_eq!(stats.fallback_batches, 0);
                assert!(stats.fallback_preparation_error.is_some());
            }
        }
    }

    #[test]
    fn ordinary_success_does_not_invent_missing_fallback_error_context() {
        let input = batch(1, 1);
        let mut result = output(vec![LaneStatus::Unstable("PV".into())], None, 1.);
        let mut notes = vec![Vec::new()];
        let mut stats = HybridStats::default();
        retry_unstable(&input, &mut result, &mut notes, &mut stats, |_| {
            let mut value = output(vec![LaneStatus::Success], None, 42.);
            value.primal_envelopes.clear();
            value.primal_envelope_errors.clear();
            outcome(value)
        })
        .unwrap();
        assert_eq!(result.lane(0).unwrap()[0].re, 42.);
        assert!(result.primal_envelope(0).is_err());
        assert!(result.primal_envelope_errors[0].is_some());
    }

    #[test]
    fn no_unstable_or_empty_batch_does_not_prepare_or_execute_fallback() {
        for statuses in [
            vec![],
            vec![LaneStatus::Success, LaneStatus::InvalidInput("mu".into())],
        ] {
            let input = batch(statuses.len(), 1);
            let mut result = output(statuses, None, 1.);
            let mut notes = vec![Vec::new(); input.rows];
            let mut stats = HybridStats::default();
            retry_unstable(&input, &mut result, &mut notes, &mut stats, |_| {
                panic!("fallback must remain lazy")
            })
            .unwrap();
            assert_eq!(stats.fallback_rows, 0);
            assert_eq!(stats.fallback_batches, 0);
            assert_eq!(stats.fallback_preparation_attempts, 0);
        }
    }

    #[test]
    fn immutable_ibp_preparation_is_lazy_shared_and_rejects_pv_completion() {
        let lazy = LazyIbpPlan::new(ReductionSettings::new(1, 0));
        let other = lazy.clone();
        assert!(lazy.prepared.get().is_none());
        let first = lazy.get().unwrap();
        let second = other.get().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.algorithm_used(), Algorithm::Ibp);
        let mut pv = ReductionSettings::new(1, 0);
        pv.algorithm = Algorithm::PassarinoVeltman;
        let pv = ReductionPlan::new(pv).unwrap();
        assert!(
            require_ibp(&pv)
                .unwrap_err()
                .contains("not an independent IBP")
        );
        assert_eq!(first.settings().algorithm, Algorithm::Ibp);
    }

    #[test]
    fn hybrid_summary_counts_nested_work_once_and_checks_outer_overflow() {
        let mut stats = HybridStats::default();
        stats.primary.parent_input_rows = 5;
        stats.primary.packed_child_rows = 1000;
        stats.primary.child_execution = ExecutionStats {
            input_rows: 21,
            master_rows: 30,
            fallback_rows: 2,
            fallback_successes: 1,
            ..Default::default()
        };
        stats.primary.parent_execution = ExecutionStats {
            input_rows: 5,
            reduction_rows: 5,
            ..Default::default()
        };
        stats.fallback = ExecutionStats {
            input_rows: 2,
            master_rows: 7,
            reduction_rows: 2,
            fallback_rows: 1,
            fallback_successes: 1,
            ..Default::default()
        };
        stats.fallback_rows = 2;
        stats.fallback_successes = 1;
        let summary = stats.detail().summary().unwrap();
        assert_eq!(summary.input_rows, 5);
        assert_eq!(summary.master_rows, 37);
        assert_eq!(summary.reduction_rows, 7);
        assert_eq!(summary.fallback_rows, 5);
        assert_eq!(summary.fallback_successes, 3);
        stats.fallback_rows = usize::MAX;
        assert!(stats.detail().summary().is_err());
    }

    #[test]
    fn retained_generic_ibp_fallback_runs_actual_numeric_work() {
        let lazy = LazyIbpPlan::new(ReductionSettings::new(1, 0));
        let plan = lazy.get().unwrap();
        let mut workspace = Workspace::new_with_policies(
            &plan,
            0.,
            WorkspacePolicy::RETAIN_ENVELOPES,
            MasterExecutionPolicy::Symbolica,
        )
        .unwrap();
        let input = KinematicsBatch {
            rows: 2,
            components: 1,
            values: [0., 0., 0., 0., 2., 3., 0., 0., 0., 0., 2., 0.]
                .map(|x| Complex::new(x, 0.))
                .to_vec(),
        };
        let result = workspace.reduce_batch(&input).unwrap();
        assert!(result.lane(0).is_ok());
        assert!(matches!(result.statuses[1], LaneStatus::InvalidInput(_)));
        assert!(result.primal_envelope(0).is_ok());
        let work = workspace.last_execution();
        assert_eq!(work.input_rows, 2);
        assert!(work.master_rows > 0);
        assert!(work.reduction_rows > 0);
        let mut independent = workspace.clone();
        independent
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 1,
                values: vec![],
            })
            .unwrap();
        assert_eq!(independent.last_execution(), ExecutionStats::default());
        assert_eq!(workspace.last_execution(), work);
    }

    #[test]
    #[ignore = "separately bounded Float128 rank-six specialized Hybrid/PV full-array delegation"]
    fn ordinary_rank_six_hybrid_uses_specialized_pv_and_keeps_fallback_lazy() {
        use symbolica::domains::{
            float::{Float, FloatLike, SingleFloat},
            rational::Rational,
        };
        let zero = Float::new(128);
        let value = |a: i64, b: i64| {
            Complex::new(zero.from_rational(&Rational::from((a, b))), zero.clone())
        };
        let shifts = [
            [0, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 4],
            [2, 0, 0, 0],
            [0, 4, 4, 0],
        ];
        let masses = [1, 2, 3, 4, 2, 3];
        let settings = ReductionSettings::new(6, 6);
        let plan = TensorReductionPlan::new(settings.clone()).unwrap();
        assert_eq!(plan.settings().algorithm, Algorithm::Hybrid);
        assert_eq!(plan.engine_kind(), TensorEngineKind::SixPointHybrid);
        assert!(plan.generic.get().is_none());
        let primary = match &plan.ordinary {
            OrdinaryPlan::SixPointHybrid(hybrid) => hybrid.primary.clone(),
            _ => panic!("wrong ordinary engine"),
        };
        let mut row = Vec::new();
        for shift in shifts {
            row.extend(shift.into_iter().map(|x| value(x, 4)));
        }
        row.extend(masses.map(|x| value(x, 1)));
        row.push(value(3, 1));
        let mut invalid = row.clone();
        *invalid.last_mut().unwrap() = value(0, 1);
        let input = KinematicsBatch {
            rows: 2,
            components: 1,
            values: row.into_iter().chain(invalid).collect(),
        };
        let before = input.values.clone();
        let mut hybrid = TensorWorkspace::new_with_options(
            &plan,
            zero.clone(),
            TensorWorkspaceOptions {
                master_policy: MasterExecutionPolicy::Symbolica,
                profiling: false,
            },
        )
        .unwrap();
        hybrid.set_accuracy_precision(53).unwrap();
        assert!(hybrid.last_profile().is_none());
        hybrid.set_profiling(true).unwrap();
        assert_eq!(hybrid.engine_kind(), TensorEngineKind::SixPointHybrid);
        assert!(hybrid.last_profile().is_some());
        let actual = hybrid.reduce_batch(&input).unwrap();
        let profile = hybrid.last_profile().unwrap();
        assert!(profile.scalar_masters > std::time::Duration::ZERO);
        assert!(profile.arguments + profile.scalar_masters + profile.reduction <= profile.total);
        let mut pv = SixPointWorkspace::new(&primary, zero).unwrap();
        pv.set_accuracy_precision(53).unwrap();
        let direct = pv.reduce_batch(&input).unwrap();
        assert_eq!(actual.basis.len(), 210);
        assert_eq!(actual.lane(0).unwrap().len(), 840);
        assert_eq!(actual.lane(0).unwrap(), direct.lane(0).unwrap());
        assert_eq!(actual.metadata(), direct.metadata());
        assert_eq!(input.values, before);
        assert!(matches!(actual.statuses[1], LaneStatus::InvalidInput(_)));
        assert_eq!(actual.fallback_used, direct.fallback_used);
        assert_eq!(actual.analytic_used, direct.analytic_used);
        assert!(actual.primal_envelope(0).is_ok());
        match hybrid.engine_execution() {
            TensorEngineExecution::SixPointHybrid {
                primary,
                fallback,
                fallback_rows,
                fallback_batches,
                fallback_preparation_attempts,
                fallback_preparation_error,
                ..
            } => {
                assert_eq!(primary.parent_input_rows, 2);
                assert_eq!(primary.packed_child_rows, 6);
                assert_eq!(fallback, ExecutionStats::default());
                assert_eq!(fallback_rows, 0);
                assert_eq!(fallback_batches, 0);
                assert_eq!(fallback_preparation_attempts, 0);
                assert!(fallback_preparation_error.is_none());
            }
            _ => panic!("Hybrid lost its actual engine tag"),
        }
        match &hybrid.engine {
            WorkspaceEngine::SixPointHybrid(workspace) => {
                assert!(matches!(workspace.fallback, FallbackState::Unprepared))
            }
            _ => unreachable!(),
        }
        assert!(plan.generic.get().is_none());
        hybrid.set_profiling(false).unwrap();
        assert!(hybrid.last_profile().is_none());
        let snap = |z: &Complex<Float>| {
            serde_json::json!({
                "real":z.re.exact_rational().map(|x|x.to_string()),
                "imaginary":z.im.exact_rational().map(|x|x.to_string()),
                "real_precision":z.re.get_precision(),
                "imaginary_precision":z.im.get_precision(),
                "real_negative_sign":z.re.negative_sign(),
                "imaginary_negative_sign":z.im.negative_sign(),
            })
        };
        let work = |s: ExecutionStats| {
            serde_json::json!({
                "input_rows":s.input_rows,"argument_rows":s.argument_rows,
                "master_family_batches":s.master_family_batches,"master_rows":s.master_rows,
                "pentagon_box_rows":s.pentagon_box_rows,"reduction_rows":s.reduction_rows,
                "fallback_rows":s.fallback_rows,"fallback_successes":s.fallback_successes,
                "analytic_rows":s.analytic_rows,
            })
        };
        let detail = match hybrid.engine_execution() {
            TensorEngineExecution::SixPointHybrid {
                primary,
                fallback,
                fallback_certificate_execution,
                fallback_rows,
                fallback_successes,
                fallback_batches,
                fallback_preparation_attempts,
                fallback_preparation_error,
            } => serde_json::json!({
                "engine":"SixPointHybrid","primary":{
                    "parent_input_rows":primary.parent_input_rows,
                    "packed_child_rows":primary.packed_child_rows,
                    "packed_parent_rows":primary.packed_parent_rows,
                    "child_execution":work(primary.child_execution),
                    "parent_execution":work(primary.parent_execution),
                    "full_detail":format!("{primary:?}"),
                },"fallback":work(fallback),"fallback_rows":fallback_rows,
                "fallback_successes":fallback_successes,"fallback_batches":fallback_batches,
                "fallback_preparation_attempts":fallback_preparation_attempts,
                "fallback_preparation_error":fallback_preparation_error,
                "fallback_certificate_execution":format!("{fallback_certificate_execution:?}"),
            }),
            _ => unreachable!(),
        };
        println!(
            "TIRUST_SIX_HYBRID_R6_CAPTURE {}",
            serde_json::json!({
                "rank":6,"propagators":6,"working_precision":128,"requested_accuracy":53,
                "requested_algorithm":"Hybrid","primary_algorithm":"PassarinoVeltman",
                "basis":actual.basis.iter().map(|m|m.0).collect::<Vec<_>>(),
                "convention":"IregiNative","tensor_weights":"Components","components":1,
                "input_before":before.iter().map(&snap).collect::<Vec<_>>(),
                "input_after":input.values.iter().map(&snap).collect::<Vec<_>>(),
                "lane_statuses":actual.statuses.iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),
                "fallback_used":actual.fallback_used,"analytic_used":actual.analytic_used,
            "pv_fallback_used":direct.fallback_used,"pv_analytic_used":direct.analytic_used,
                "lanes":[actual.lane(0).unwrap().iter().map(&snap).collect::<Vec<_>>(),serde_json::Value::Null],
                "pv_lane":direct.lane(0).unwrap().iter().map(&snap).collect::<Vec<_>>(),
                "diagnostics":hybrid.last_diagnostics(),"execution":detail,
                "summary":work(hybrid.execution_summary().unwrap()),
                "independent_quad_agreement_checked_in_test":false,
            })
        );
        hybrid
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 1,
                values: vec![],
            })
            .unwrap();
        assert_eq!(
            hybrid.execution_summary().unwrap(),
            ExecutionStats::default()
        );
        assert!(hybrid.last_diagnostics().is_empty());
    }
}
