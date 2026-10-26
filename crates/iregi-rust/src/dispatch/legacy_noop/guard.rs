//! Existing Legacy branch monitor and selected exact residuals, before any
//! reduction recipe. This module owns lane bookkeeping, never snap algebra.
use super::*;
use crate::execution::legacy_compatibility::{
    LegacyCompatibilityWorkspace, PreparedLegacyCompatibility,
};
use crate::execution::{compile_exact, map_exact, validate_input_row};
use symbolica::domains::rational::Rational;
use symbolica::evaluate::{BatchExpressionEvaluator, ExpressionEvaluator};

pub(super) struct GuardSelection {
    pub accepted: Vec<usize>,
    pub statuses: Vec<LaneStatus>,
    pub diagnostics: Vec<Vec<String>>,
    pub certificate: CertificateExecutionStats,
}

#[derive(Clone)]
pub(super) struct NoOpGuard<T: NativeFloat> {
    compatibility: LegacyCompatibilityWorkspace<T>,
    monitor: BatchExpressionEvaluator<Complex<T>>,
    labels: Vec<String>,
    layout: InputLayout,
    parameters: usize,
    components: usize,
    positive_threshold: bool,
    prototype: T,
    certificate_execution: CertificateExecutionStats,
}

impl<T: NativeFloat> NoOpGuard<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn prepare<D: DualNumberStructure>(
        queries: &queries::LegacyNoOpQueries,
        prototype: T,
        dual: Option<&D>,
    ) -> Result<Self, ReductionError> {
        let active_queries: Vec<_> = queries
            .queries
            .iter()
            .cloned()
            .map(|query| (query, Atom::num(1)))
            .collect();
        let prepared = PreparedLegacyCompatibility::prepare_from_queries(
            &queries.parameters,
            queries.layout.masses.clone(),
            &active_queries,
            &queries.aliases,
            dual,
        )?;
        let mut expressions: Vec<_> = queries
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.expression.clone())
            .collect();
        expressions.push(queries.threshold.clone());
        // Strict comparisons inspect the primal, as in the existing executor.
        // Their chosen residual is separately proved in every full jet slot.
        let monitor: ExpressionEvaluator<Complex<Rational>> =
            compile_exact::<D>(&expressions, &queries.parameters, &queries.aliases, None)?;
        Ok(Self {
            compatibility: LegacyCompatibilityWorkspace::instantiate(&prepared, &prototype),
            monitor: map_exact(&monitor, &prototype),
            labels: queries
                .diagnostics
                .iter()
                .map(|d| d.label.clone())
                .collect(),
            layout: queries.layout.clone(),
            parameters: queries.parameters.len(),
            components: dual.map_or(1, |d| d.get_shape().len()),
            positive_threshold: !queries.threshold.is_zero(),
            prototype,
            certificate_execution: CertificateExecutionStats::default(),
        })
    }

    pub(super) fn last_certificate_execution(&self) -> CertificateExecutionStats {
        self.certificate_execution
    }

    pub(super) fn select(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<GuardSelection, ReductionError> {
        self.certificate_execution = CertificateExecutionStats {
            input_rows: input.rows,
            ..Default::default()
        };
        let width = mul(self.parameters, self.components)?;
        if input.components != self.components || input.values.len() != mul(input.rows, width)? {
            return Err(invalid("Legacy no-op full input/Taylor dimensions differ"));
        }
        let mut statuses = vec![LaneStatus::Success; input.rows];
        let mut diagnostics = vec![Vec::new(); input.rows];
        let mut valid = Vec::new();
        let mut primal = Vec::new();
        for (lane, row) in input.values.chunks_exact(width).enumerate() {
            if let Some(status) = validate_input_row(row, &self.layout, self.components) {
                statuses[lane] = status;
            } else {
                valid.push(lane);
                primal.extend(row.chunks_exact(self.components).map(|jet| jet[0].clone()));
            }
        }
        let monitor_width = add(self.labels.len(), 1)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut monitored = vec![zero; mul(valid.len(), monitor_width)?];
        self.monitor
            .evaluate_batch(&primal, &mut monitored, valid.len())
            .map_err(|e| ReductionError::Evaluation(e.to_string()))?;
        let mut selected_lanes = Vec::new();
        let mut selected_values = Vec::new();
        for (compact, &lane) in valid.iter().enumerate() {
            let outputs = &monitored[compact * monitor_width..(compact + 1) * monitor_width];
            let threshold = &outputs[self.labels.len()];
            let mut failure = if !finite(threshold)
                || !threshold.im.is_zero()
                || (self.positive_threshold && threshold.re.is_zero())
            {
                Some("legacy no-op threshold cannot be represented at the working precision".into())
            } else {
                None
            };
            for (value, label) in outputs.iter().zip(&self.labels) {
                if failure.is_some() {
                    break;
                }
                if !finite(value) || !value.im.is_zero() {
                    failure = Some(format!(
                        "legacy on-shell branch cannot be classified at the working precision: {label}"
                    ));
                } else if value.re.is_zero() {
                    if self.components > 1 {
                        failure = Some(format!(
                            "legacy on-shell policy is nondifferentiable at an active strict-comparison boundary: {label}"
                        ));
                    } else {
                        diagnostics[lane].push(format!("legacy on-shell strict-comparison boundary selected its equality branch: {label}"));
                    }
                }
            }
            if let Some(reason) = failure {
                statuses[lane] = LaneStatus::Unstable(reason);
            } else {
                selected_lanes.push(lane);
                selected_values.extend_from_slice(&input.values[lane * width..(lane + 1) * width]);
            }
        }
        let failures = self.compatibility.failures(
            &selected_values,
            selected_lanes.len(),
            self.components,
            &self.prototype,
        );
        let mut certificate = self.compatibility.last_certificate_execution();
        certificate.input_rows = input.rows;
        self.certificate_execution = certificate;
        let failures = failures?;
        for (compact, status) in failures {
            let &lane = selected_lanes.get(compact).ok_or_else(|| {
                ReductionError::Evaluation("Legacy no-op proof lane map differs".into())
            })?;
            statuses[lane] = status;
        }
        let accepted = selected_lanes
            .into_iter()
            .filter(|&lane| statuses[lane] == LaneStatus::Success)
            .collect();
        for (lane, status) in statuses.iter().enumerate() {
            if *status != LaneStatus::Success {
                diagnostics[lane].push("Conservative pre-recurrence Legacy query superset did not prove an unchanged operation; no Preserve child was executed for this lane. This is not an exact demanded-query classification.".into());
            }
        }
        Ok(GuardSelection {
            accepted,
            statuses,
            diagnostics,
            certificate,
        })
    }
}

fn finite<T: NativeFloat>(value: &Complex<T>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
