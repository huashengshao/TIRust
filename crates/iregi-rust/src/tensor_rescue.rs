//! Precision rescue through the same tensor facade used for ordinary evaluation.
//!
//! Eligible scalar thin and affine-power engines stay selected at every precision.
//! Explicit Dual layouts retain generic rescue where supported; affine joint-Dual
//! rescue is explicitly unavailable until coupled-context transport is qualified.
//! Higher-precision convergence is a numerical check, not a certified error bound.

use iregi_masters::MasterBackend;
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure},
        float::{Complex, Float, FloatLike},
    },
    evaluate::EvaluationDomain,
};

use crate::{
    Algorithm, BatchResult, KinematicsBatch, LaneStatus, PrecisionRescueWorkspace, ReductionError,
    RescueScalar, RescueSettings, RescueStats, RescueStatus, RescueTrigger, TensorEngineExecution,
    TensorEngineKind, TensorReductionPlan, TensorWorkspace,
    rescue::{
        RescueComparison, accept_rescued_row, promote_row, reject, validate_rescue_settings,
        verify_rescued_row,
    },
};

/// Owned primary and two higher-precision tensor workspaces.
///
/// The constructor preserves the compiled master-backend policy. Specialized
/// scalar engines are used when the ordinary facade permits them. Existing
/// supported generic Dual domains use `PrecisionRescueWorkspace`; eligible
/// affine high-point Dual rescue returns Unsupported before preparation. Selected
/// samples form one batch at each higher precision, never extra Taylor lanes.
/// Accepted rows have no scalar sensitivity envelope: convergence and rounding
/// do not transport that coupled context through a precision conversion.
#[derive(Clone)]
pub struct TensorRescueWorkspace<T: RescueScalar> {
    engine: RescueEngine<T>,
    algorithm_used: Algorithm,
    notes: Vec<String>,
}

#[derive(Clone)]
enum RescueEngine<T: RescueScalar> {
    Generic(PrecisionRescueWorkspace<T>),
    Tensor(TensorPrecisionRescue<T>),
}

#[derive(Clone)]
struct TensorPrecisionRescue<T: RescueScalar> {
    prototype: T,
    plan: TensorReductionPlan,
    primary: TensorWorkspace<T>,
    working: TensorWorkspace<Float>,
    verification: TensorWorkspace<Float>,
    settings: RescueSettings,
    accuracy_bits: u32,
    statuses: Vec<RescueStatus>,
    primary_statuses: Vec<LaneStatus>,
    diagnostics: Vec<Vec<String>>,
    stats: RescueStats,
    primary_execution: Option<TensorEngineExecution>,
    working_execution: Option<TensorEngineExecution>,
    verification_execution: Option<TensorEngineExecution>,
}

impl<T: RescueScalar> TensorRescueWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(
        plan: &TensorReductionPlan,
        prototype: T,
        settings: RescueSettings,
    ) -> Result<Self, ReductionError> {
        let accuracy_bits = validate_rescue_settings(prototype.get_precision(), settings)?;
        // Precision rescue compares working precisions on one engine: use
        // the symbolic plan even where binary64 defaults to direct recurrences.
        let symbolic = plan.without_direct()?;
        let plan = symbolic.as_ref();
        if !specialized_rescue_eligible(plan.engine_kind())
            && !crate::dispatch::affine_power_eligible(plan.settings())
        {
            let generic = plan.generic_plan()?;
            let mut notes = vec![
                "Facade precision rescue: existing generic engine; compiled default master policy retained.".into(),
            ];
            notes.extend_from_slice(generic.preparation_notes());
            return Ok(Self {
                algorithm_used: generic.algorithm_used(),
                engine: RescueEngine::Generic(PrecisionRescueWorkspace::new(
                    &generic, prototype, settings,
                )?),
                notes,
            });
        }
        // Do not obtain plan.generic_plan() here: it can prepare a much larger
        // unrelated generic graph before any specialized sample is evaluated.
        let mut primary = TensorWorkspace::new(plan, prototype.clone())?;
        let mut working = TensorWorkspace::<Float>::new(plan, Float::new(settings.working_bits))?;
        let mut verification =
            TensorWorkspace::<Float>::new(plan, Float::new(settings.verification_bits))?;
        primary.set_accuracy_precision(accuracy_bits)?;
        working.set_accuracy_precision(accuracy_bits)?;
        verification.set_accuracy_precision(accuracy_bits)?;
        let mut notes = primary.preparation_notes().to_vec();
        notes.push(format!(
            "Facade precision rescue: actual {:?} workspaces at output, {} and {} bits; requested accuracy {} bits.",
            primary.engine_kind(), settings.working_bits, settings.verification_bits, accuracy_bits,
        ));
        Ok(Self {
            algorithm_used: plan.settings().algorithm,
            engine: RescueEngine::Tensor(TensorPrecisionRescue {
                prototype,
                plan: plan.clone(),
                primary,
                working,
                verification,
                settings,
                accuracy_bits,
                statuses: Vec::new(),
                primary_statuses: Vec::new(),
                diagnostics: Vec::new(),
                stats: RescueStats::default(),
                primary_execution: None,
                working_execution: None,
                verification_execution: None,
            }),
            notes,
        })
    }

    /// Existing generic Dual domains retain their original ordered shape.
    /// Affine high-point joint-context rescue is explicitly unsupported; a
    /// rounded result cannot silently discard the required coupled context.
    pub fn new_dual<D: DualNumberStructure>(
        plan: &TensorReductionPlan,
        prototype: T,
        settings: RescueSettings,
        dual: D,
    ) -> Result<Self, ReductionError> {
        validate_rescue_settings(prototype.get_precision(), settings)?;
        if crate::dispatch::affine_power_eligible(plan.settings())
            || plan.engine_kind() == TensorEngineKind::MassTaylor
        {
            return Err(ReductionError::Unsupported(
                "composite tensor precision rescue with an explicit Dual layout requires qualified joint-context precision transport".into()));
        }
        let generic = plan.generic_plan()?;
        let mut notes =
            vec!["Facade precision rescue: existing generic engine; explicit Dual layout.".into()];
        notes.extend_from_slice(generic.preparation_notes());
        Ok(Self {
            algorithm_used: generic.algorithm_used(),
            engine: RescueEngine::Generic(PrecisionRescueWorkspace::new_dual(
                &generic, prototype, settings, dual,
            )?),
            notes,
        })
    }

    pub fn engine_kind(&self) -> TensorEngineKind {
        match &self.engine {
            RescueEngine::Generic(_) => TensorEngineKind::Generic,
            RescueEngine::Tensor(workspace) => workspace.primary.engine_kind(),
        }
    }
    pub fn algorithm_used(&self) -> Algorithm {
        self.algorithm_used
    }
    pub fn preparation_notes(&self) -> &[String] {
        &self.notes
    }

    pub fn component_count(&self) -> usize {
        match &self.engine {
            RescueEngine::Generic(workspace) => workspace.component_count(),
            RescueEngine::Tensor(workspace) => workspace.primary.component_count(),
        }
    }
    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        match &self.engine {
            RescueEngine::Generic(workspace) => workspace.dual_shape(),
            RescueEngine::Tensor(_) => None,
        }
    }
    pub fn last_execution(&self) -> RescueStats {
        match &self.engine {
            RescueEngine::Generic(workspace) => workspace.last_execution(),
            RescueEngine::Tensor(workspace) => workspace.stats,
        }
    }
    pub fn last_rescue_statuses(&self) -> &[RescueStatus] {
        match &self.engine {
            RescueEngine::Generic(workspace) => workspace.last_rescue_statuses(),
            RescueEngine::Tensor(workspace) => &workspace.statuses,
        }
    }
    pub fn last_primary_statuses(&self) -> &[LaneStatus] {
        match &self.engine {
            RescueEngine::Generic(workspace) => workspace.last_primary_statuses(),
            RescueEngine::Tensor(workspace) => &workspace.primary_statuses,
        }
    }
    /// Per-lane notes from the primary and each actually executed higher stage.
    /// Generic rescue retains its existing primary diagnostics interface.
    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        match &self.engine {
            RescueEngine::Generic(workspace) => {
                if workspace.last_primary_statuses().is_empty() {
                    &[]
                } else {
                    workspace.last_diagnostics()
                }
            }
            RescueEngine::Tensor(workspace) => &workspace.diagnostics,
        }
    }

    /// Actual tagged stages of the last call. `None` means that stage did not
    /// run, or its detail is not exposed by the legacy generic rescue engine.
    pub fn primary_engine_execution(&self) -> Option<TensorEngineExecution> {
        match &self.engine {
            RescueEngine::Generic(_) => None,
            RescueEngine::Tensor(workspace) => workspace.primary_execution.clone(),
        }
    }
    pub fn working_engine_execution(&self) -> Option<TensorEngineExecution> {
        match &self.engine {
            RescueEngine::Generic(_) => None,
            RescueEngine::Tensor(workspace) => workspace.working_execution.clone(),
        }
    }
    pub fn verification_engine_execution(&self) -> Option<TensorEngineExecution> {
        match &self.engine {
            RescueEngine::Generic(_) => None,
            RescueEngine::Tensor(workspace) => workspace.verification_execution.clone(),
        }
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        match &mut self.engine {
            RescueEngine::Generic(workspace) => workspace.reduce_batch(batch),
            RescueEngine::Tensor(workspace) => workspace.reduce_batch(batch),
        }
    }

    pub fn reduce_typed_dual_batch<D: DualNumber<Complex<T>>>(
        &mut self,
        input: &[D],
        rows: usize,
    ) -> Result<BatchResult<T>, ReductionError> {
        match &mut self.engine {
            RescueEngine::Generic(workspace) => {
                // The facade reports this call even when typed-shape validation
                // rejects it before the legacy numerical entry is reached.
                workspace.clear_last_report();
                workspace.reduce_typed_dual_batch(input, rows)
            }
            RescueEngine::Tensor(workspace) => {
                workspace.reset();
                Err(ReductionError::InvalidInput(
                    "use TensorRescueWorkspace::new_dual before passing Dual values".into(),
                ))
            }
        }
    }
}

fn specialized_rescue_eligible(kind: TensorEngineKind) -> bool {
    matches!(
        kind,
        TensorEngineKind::AffinePower | TensorEngineKind::MassTaylor
    ) || (matches!(
        kind,
        TensorEngineKind::SixPoint
            | TensorEngineKind::SixPointHybrid
            | TensorEngineKind::SevenPoint
    ) && iregi_masters::default_master_backend() == MasterBackend::SymbolicaBatch)
}

impl<T: RescueScalar> TensorPrecisionRescue<T>
where
    Complex<T>: EvaluationDomain,
{
    fn reset(&mut self) {
        self.stats = RescueStats::default();
        self.statuses.clear();
        self.primary_statuses.clear();
        self.diagnostics.clear();
        self.primary_execution = None;
        self.working_execution = None;
        self.verification_execution = None;
    }

    fn comparison(&self) -> RescueComparison<'_> {
        RescueComparison {
            settings: self.plan.settings(),
            basis: self.plan.basis(),
            layout: self.plan.input_layout(),
            verification_bits: self.settings.verification_bits,
            accuracy_bits: self.accuracy_bits,
        }
    }

    fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.reset();
        let primary = self.primary.reduce_batch(batch);
        let execution = self.primary.engine_execution();
        self.stats.primary = execution.summary()?;
        self.primary_execution = Some(execution);
        self.diagnostics = self.primary.last_diagnostics().to_vec();
        let mut result = primary?;
        self.primary_statuses.clone_from(&result.statuses);
        self.statuses.resize(batch.rows, RescueStatus::NotRequested);
        self.diagnostics.resize_with(batch.rows, Vec::new);
        let input_width = self
            .plan
            .parameter_count()
            .checked_mul(batch.components)
            .ok_or_else(|| {
                ReductionError::InvalidInput("rescue input dimensions overflow".into())
            })?;
        let mut lanes = Vec::new();
        let mut low_inputs = Vec::new();
        let mut high_inputs = Vec::new();
        for lane in 0..batch.rows {
            let requested = match result.statuses[lane] {
                LaneStatus::Unstable(_) => true,
                LaneStatus::Success => self.settings.trigger == RescueTrigger::VerifyAllSuccessful,
                _ => false,
            };
            if !requested {
                continue;
            }
            self.stats.requested_rows += 1;
            let row = &batch.values[lane * input_width..(lane + 1) * input_width];
            match promote_row(
                row,
                self.settings.working_bits,
                self.settings.verification_bits,
            ) {
                Ok((low, high)) => {
                    lanes.push(lane);
                    low_inputs.extend(low);
                    high_inputs.extend(high);
                }
                Err(reason) => reject(&mut result, &mut self.statuses, lane, reason),
            }
        }
        if lanes.is_empty() {
            return Ok(result);
        }
        let low_batch = KinematicsBatch {
            rows: lanes.len(),
            components: batch.components,
            values: low_inputs,
        };
        let high_batch = KinematicsBatch {
            rows: lanes.len(),
            components: batch.components,
            values: high_inputs,
        };
        let low = self.working.reduce_batch(&low_batch);
        let execution = self.working.engine_execution();
        self.stats.working = execution.summary()?;
        self.working_execution = Some(execution);
        append_stage_notes(
            &mut self.diagnostics,
            &lanes,
            "working precision",
            self.working.last_diagnostics(),
        );
        let low = low?;
        let high = self.verification.reduce_batch(&high_batch);
        let execution = self.verification.engine_execution();
        self.stats.verification = execution.summary()?;
        self.verification_execution = Some(execution);
        append_stage_notes(
            &mut self.diagnostics,
            &lanes,
            "verification precision",
            self.verification.last_diagnostics(),
        );
        let high = high?;
        for (compact, &lane) in lanes.iter().enumerate() {
            let input = &high_batch.values[compact * input_width..(compact + 1) * input_width];
            let attempt = verify_rescued_row(
                &self.prototype,
                self.settings,
                self.comparison(),
                &low,
                &high,
                compact,
                lane,
                input,
                batch.components,
                result.lane(lane).ok(),
            );
            match attempt {
                Ok((values, primary_agreed)) => {
                    accept_rescued_row(
                        &mut result,
                        lane,
                        &values,
                        low.fallback_used[compact] || high.fallback_used[compact],
                        low.analytic_used[compact] || high.analytic_used[compact],
                    );
                    self.statuses[lane] = RescueStatus::Converged {
                        working_bits: self.settings.working_bits,
                        verification_bits: self.settings.verification_bits,
                        primary_agreed,
                    };
                    self.stats.accepted_rows += 1;
                    if primary_agreed == Some(false) {
                        self.stats.corrected_successful_rows += 1;
                    }
                }
                Err(reason) => reject(&mut result, &mut self.statuses, lane, reason),
            }
        }
        Ok(result)
    }
}

fn append_stage_notes(
    destination: &mut [Vec<String>],
    lanes: &[usize],
    stage: &str,
    notes: &[Vec<String>],
) {
    for (compact, &lane) in lanes.iter().enumerate() {
        if let Some(notes) = notes.get(compact) {
            destination[lane].extend(notes.iter().map(|note| format!("{stage}: {note}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Convention, InputLayout, ReductionSettings, ResultMetadata, TensorWeights,
        basis::Monomial,
        types::{PrimalEnvelope, PrimalEnvelopeRoute, PrimalOperand, PrimalOperandKind},
    };
    use symbolica::domains::dual::HyperDual;

    fn scalar_result<T: RescueScalar>(
        prototype: &T,
        rows: usize,
        components: usize,
    ) -> BatchResult<T> {
        BatchResult {
            basis: vec![Monomial([0; 4])],
            components,
            metadata: ResultMetadata::new(
                Convention::IregiNative,
                TensorWeights::Components,
                (components == 3).then(|| vec![vec![0], vec![1], vec![2]]),
            ),
            values: vec![Complex::new(prototype.clone(), prototype.clone()); rows * 4 * components],
            statuses: vec![LaneStatus::Success; rows],
            fallback_used: vec![false; rows],
            analytic_used: vec![false; rows],
            primal_envelopes: vec![None; rows],
            primal_envelope_errors: vec![None; rows],
        }
    }

    #[test]
    fn rescue_configuration_and_backend_eligibility_are_explicit() {
        let settings = RescueSettings::default();
        assert_eq!(validate_rescue_settings(53, settings).unwrap(), 53);
        assert!(validate_rescue_settings(128, settings).is_err());
        assert!(
            validate_rescue_settings(
                53,
                RescueSettings {
                    accuracy_bits: Some(54),
                    ..settings
                }
            )
            .is_err()
        );
        assert!(!specialized_rescue_eligible(TensorEngineKind::Generic));
        assert!(!specialized_rescue_eligible(
            TensorEngineKind::CertifiedTensor
        ));
        assert!(specialized_rescue_eligible(TensorEngineKind::AffinePower));
        for kind in [
            TensorEngineKind::SixPoint,
            TensorEngineKind::SixPointHybrid,
            TensorEngineKind::SevenPoint,
        ] {
            assert_eq!(
                specialized_rescue_eligible(kind),
                iregi_masters::default_master_backend() == MasterBackend::SymbolicaBatch
            );
        }
    }

    #[test]
    fn shared_verification_checks_every_slot_and_dual_component_before_rounding() {
        let settings = ReductionSettings::new(2, 0);
        let layout = InputLayout {
            shifts: 0..8,
            scalar_products: None,
            masses: 8..10,
            scale: 10,
        };
        let basis = [Monomial([0; 4])];
        let policy = RescueSettings::default();
        let comparison = RescueComparison {
            settings: &settings,
            basis: &basis,
            layout: &layout,
            verification_bits: 256,
            accuracy_bits: 53,
        };
        let input = vec![Complex::new(Float::with_val(256, 1), Float::with_val(256, 0)); 11 * 3];
        let mut low = scalar_result(&Float::with_val(128, 0), 1, 3);
        let mut high = scalar_result(&Float::with_val(256, 0), 1, 3);
        for index in 0..12 {
            low.values[index].re = Float::with_val(128, (index + 1) as i64);
            high.values[index].re = Float::with_val(256, (index + 1) as i64);
        }
        let old = vec![Complex::new(0.0, 0.0); 12];
        let (rounded, agreed) = verify_rescued_row(
            &0.0,
            policy,
            comparison,
            &low,
            &high,
            0,
            0,
            &input,
            3,
            Some(&old),
        )
        .unwrap();
        assert_eq!(agreed, Some(false));
        assert_eq!(rounded.len(), 12);
        for index in 0..12 {
            let saved = high.values[index].clone();
            high.values[index].im = Float::with_val(256, 1);
            let error =
                verify_rescued_row(&0.0, policy, comparison, &low, &high, 0, 0, &input, 3, None)
                    .unwrap_err();
            assert!(error.contains(&format!("Laurent slot {}", index / 3)));
            assert!(error.contains(&format!("Taylor component {}", index % 3)));
            high.values[index] = saved;
        }
        // Convergence alone must not allow an output-domain overflow.
        for value in &mut low.values {
            value.re = Float::with_val(128, f64::MAX) * Float::with_val(128, 2);
        }
        for value in &mut high.values {
            value.re = Float::with_val(256, f64::MAX) * Float::with_val(256, 2);
        }
        assert!(
            verify_rescued_row(&0.0, policy, comparison, &low, &high, 0, 0, &input, 3, None)
                .unwrap_err()
                .contains("finite range")
        );
    }

    #[test]
    fn accepted_scatter_discards_only_the_replaced_coupled_context() {
        let mut result = scalar_result(&0.0, 3, 1);
        for lane in 0..3 {
            result.values[lane * 4..(lane + 1) * 4].fill(Complex::new((lane + 1) as f64, 0.0));
            result.primal_envelopes[lane] = Some(PrimalEnvelope {
                basis: result.basis.clone(),
                metadata: result.metadata.clone(),
                components: 1,
                working_precision: 53,
                accuracy_bits: 53,
                operands: (0..4)
                    .map(|output_index| PrimalOperand {
                        output_index,
                        amplitude: Complex::new(1.0, 0.0),
                        kind: PrimalOperandKind::NoMaster,
                    })
                    .collect(),
                output_primals: result.values[lane * 4..(lane + 1) * 4].to_vec(),
                component_amplitudes: None,
                output_components: None,
                routes: vec![PrimalEnvelopeRoute::Ordinary],
            });
            assert!(result.primal_envelope(lane).is_ok());
        }
        let neighbor = result.lane(2).unwrap().to_vec();
        let mut statuses = vec![RescueStatus::NotRequested; 3];
        result.statuses[0] = LaneStatus::Unstable("primary cancellation".into());
        accept_rescued_row(
            &mut result,
            0,
            &vec![Complex::new(7.0, -0.0); 4],
            true,
            true,
        );
        assert!(result.lane(0).is_ok());
        assert!(
            result
                .primal_envelope(0)
                .unwrap_err()
                .contains("unavailable after precision conversion")
        );
        assert!(result.primal_envelopes[0].is_none());
        assert!(result.fallback_used[0] && result.analytic_used[0]);
        reject(
            &mut result,
            &mut statuses,
            1,
            "independent convergence failed".into(),
        );
        assert!(matches!(result.lane(1), Err(LaneStatus::Unstable(_))));
        assert_eq!(result.lane(2).unwrap(), neighbor);
        assert!(result.primal_envelope(2).is_ok());
        assert!(!result.fallback_used[2] && !result.analytic_used[2]);
    }

    #[test]
    fn compact_stage_diagnostics_preserve_original_sample_indices() {
        let mut notes = vec![
            vec!["primary 0".into()],
            vec!["primary 1".into()],
            vec!["primary 2".into()],
        ];
        append_stage_notes(
            &mut notes,
            &[2, 0],
            "verification precision",
            &[
                vec!["selected lane 2".into()],
                vec!["selected lane 0".into()],
            ],
        );
        assert_eq!(notes[1], vec!["primary 1"]);
        assert_eq!(notes[2][1], "verification precision: selected lane 2");
        assert_eq!(notes[0][1], "verification precision: selected lane 0");
    }

    #[test]
    fn generic_facade_rescue_keeps_dual_axes_clone_empty_and_invalid_reports_separate() {
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.convention = Convention::OneLoop;
        let plan = TensorReductionPlan::new(settings).unwrap();
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = plan
            .rescue_dual_workspace(
                0.0,
                RescueSettings {
                    trigger: RescueTrigger::VerifyAllSuccessful,
                    ..RescueSettings::default()
                },
                shape,
            )
            .unwrap();
        assert_eq!(workspace.engine_kind(), TensorEngineKind::Generic);
        assert_eq!(workspace.component_count(), 3);
        let c = |re| Complex::new(re, 0.0);
        let mut values = vec![c(0.0); 3 * 6 * 3];
        for lane in 0..3 {
            values[lane * 18 + 12] = c(2.0 + lane as f64);
            values[lane * 18 + 13] = c(1.0);
            values[lane * 18 + 15] = c(3.0);
        }
        values[2 * 18 + 15] = c(-1.0);
        let batch = KinematicsBatch {
            rows: 3,
            components: 3,
            values,
        };
        let mut cloned = workspace.clone();
        let result = workspace.reduce_batch(&batch).unwrap();
        let other = cloned.reduce_batch(&batch).unwrap();
        for lane in 0..2 {
            assert_eq!(result.lane(lane).unwrap(), other.lane(lane).unwrap());
            assert_eq!(
                &result.lane(lane).unwrap()[..3],
                &[c(2.0 + lane as f64), c(1.0), c(0.0)]
            );
            assert!(
                result
                    .primal_envelope(lane)
                    .unwrap_err()
                    .contains("precision conversion")
            );
        }
        assert!(matches!(result.lane(2), Err(LaneStatus::InvalidInput(_))));
        assert_eq!(workspace.last_execution().accepted_rows, 2);
        assert_eq!(workspace.last_execution().working.input_rows, 2);
        assert_eq!(workspace.last_execution().verification.input_rows, 2);
        let empty = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 3,
                values: vec![],
            })
            .unwrap();
        assert_eq!(empty.rows(), 0);
        assert_eq!(workspace.last_execution(), RescueStats::default());
        cloned
            .reduce_typed_dual_batch::<HyperDual<Complex<f64>>>(&[], 1)
            .unwrap_err();
        assert_eq!(cloned.last_execution(), RescueStats::default());
        assert!(cloned.last_primary_statuses().is_empty());
        assert!(cloned.last_rescue_statuses().is_empty());
        assert!(cloned.last_diagnostics().is_empty());
        assert!(workspace.primary_engine_execution().is_none());
        assert!(workspace.working_engine_execution().is_none());
        assert!(workspace.verification_engine_execution().is_none());
    }
}
