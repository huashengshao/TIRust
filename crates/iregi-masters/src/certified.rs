//! Explicit accuracy-aware scalar-master evaluation.
//!
//! Certified finite D0 results enclose the function of the exact represented
//! input components. Routing or shift roundoff before this call requires its
//! own enclosure. The bounded series supports orders two and four, at most
//! 32 ancestor-closed Taylor components, and total derivative degree at most four.
//!
//! Explicit Rational budgets determine acceptance. Ordinary fallback is opt-in
//! and carries no certificate. The ordinary `MasterEvaluator` API is unchanged.

use crate::{
    Complex, EvaluationDomain, LaneError, MasterBatch, MasterError, MasterEvaluator,
    MasterLane, NativeFloat, OUTPUTS, ScalarIntegral, near_vacuum,
};

pub use near_vacuum::{
    AccuracyContext as MasterAccuracyContext, ComponentBound as FiniteComponentBound,
    Execution as SeriesExecution, Status as SeriesStatus,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertificatePolicy {
    RequireFiniteCertificate,
    AllowOrdinaryUncertified,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Formula {
    Rejected,
    OrdinaryUncertified,
    NearVacuum { order: usize },
}

#[derive(Clone, Debug)]
pub struct SeriesAttempt {
    pub order: usize,
    pub status: SeriesStatus,
    pub rho: Option<symbolica::domains::rational::Rational>,
}

#[derive(Clone, Debug)]
pub struct LaneDiagnostic {
    pub requested_bits: u32,
    pub formula: Formula,
    pub attempts: Vec<SeriesAttempt>,
    pub finite: Option<Vec<FiniteComponentBound>>,
    /// A valid ordinary request may be outside the bounded series facility.
    pub series_unavailable: Option<String>,
}

pub struct AccuracyMasterBatch<T: NativeFloat> {
    results: MasterBatch<T>,
    diagnostics: Vec<LaneDiagnostic>,
    pub series_execution: Vec<(usize, SeriesExecution)>,
    pub ordinary_rows: usize,
}

impl<T: NativeFloat> AccuracyMasterBatch<T> {
    pub fn row_count(&self) -> usize {
        self.results.row_count()
    }
    pub fn component_count(&self) -> usize {
        self.results.component_count()
    }
    pub fn lane(&self, lane: usize) -> Result<MasterLane<'_, T>, &LaneError> {
        self.results.lane(lane)
    }
    pub fn errors(&self) -> &[Option<LaneError>] {
        self.results.errors()
    }
    pub fn diagnostics(&self) -> &[LaneDiagnostic] {
        &self.diagnostics
    }
}

/// Owns optional series workspaces separately from the unchanged ordinary API.
/// Preparation is opt-in and lazy; ordinary callers incur no new conversion.
pub struct AccuracyMasterEvaluator<T: NativeFloat> {
    ordinary: MasterEvaluator<T>,
    series2: Option<near_vacuum::Workspace<T>>,
    series4: Option<near_vacuum::Workspace<T>>,
}

impl<T: NativeFloat> AccuracyMasterEvaluator<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(ordinary: MasterEvaluator<T>) -> Self {
        Self {
            ordinary,
            series2: None,
            series4: None,
        }
    }

    /// Preparation diagnostics; derivative components are not sample lanes.
    pub fn prepared_series_orders(&self) -> Vec<usize> {
        [(2, self.series2.is_some()), (4, self.series4.is_some())]
            .into_iter()
            .filter_map(|(order, prepared)| prepared.then_some(order))
            .collect()
    }

    pub fn evaluate_batch_with_accuracy(
        &mut self,
        input: &[Complex<T>],
        rows: usize,
        accuracy: &MasterAccuracyContext,
        policy: CertificatePolicy,
    ) -> Result<AccuracyMasterBatch<T>, MasterError> {
        let c = self.ordinary.components;
        accuracy.validate(c).map_err(MasterError::InvalidAccuracy)?;
        let width = self
            .ordinary
            .family
            .arity()
            .checked_mul(c)
            .ok_or(MasterError::SizeOverflow)?;
        let expected = rows.checked_mul(width).ok_or(MasterError::SizeOverflow)?;
        if input.len() != expected {
            return Err(MasterError::InputLength {
                expected,
                actual: input.len(),
            });
        }
        let result_width = OUTPUTS.checked_mul(c).ok_or(MasterError::SizeOverflow)?;
        let result_size = rows
            .checked_mul(result_width)
            .ok_or(MasterError::SizeOverflow)?;
        let zero = Complex::new(self.ordinary.prototype.zero(), self.ordinary.prototype.zero());
        let mut result = AccuracyMasterBatch {
            results: MasterBatch {
                components: c,
                values: vec![zero; result_size],
                errors: vec![Some(LaneError::UncertifiedFinite(None)); rows],
            },
            diagnostics: (0..rows)
                .map(|_| LaneDiagnostic {
                    requested_bits: accuracy.requested_bits,
                    formula: Formula::Rejected,
                    attempts: Vec::new(),
                    finite: None,
                    series_unavailable: None,
                })
                .collect(),
            series_execution: Vec::new(),
            ordinary_rows: 0,
        };
        let mut pending = Vec::new();
        for (lane, point) in input.chunks_exact(width).enumerate() {
            match self.ordinary.validate_point(point) {
                Ok(()) => pending.push(lane),
                Err(error) => result.results.errors[lane] = Some(error),
            }
        }

        // Physical scale validation above applies even though a finite D0 has
        // no scale dependence. Only physical arguments enter the series.
        let mut ordinary_pending = Vec::new();
        let shape = self
            .ordinary
            .shape
            .clone()
            .unwrap_or_else(|| vec![Vec::new()]);
        if self.ordinary.family == ScalarIntegral::D0 && !pending.is_empty() {
            if let Err(reason) = near_vacuum::NearVacuumPlan::validate_shape(&shape) {
                for &lane in &pending {
                    result.diagnostics[lane].series_unavailable = Some(reason.clone());
                }
                ordinary_pending.append(&mut pending);
            }
            for order in [2, 4] {
                if pending.is_empty() {
                    break;
                }
                let workspace = if order == 2 {
                    &mut self.series2
                } else {
                    &mut self.series4
                };
                if workspace.is_none() {
                    let plan = near_vacuum::NearVacuumPlan::new(order, shape.clone())
                        .map_err(MasterError::Preparation)?;
                    *workspace = Some(plan.instantiate(self.ordinary.prototype.clone()));
                }
                let workspace = workspace.as_mut().unwrap();
                let physical_width = near_vacuum::PHYSICAL_ARGUMENTS
                    .checked_mul(c)
                    .ok_or(MasterError::SizeOverflow)?;
                let capacity = pending
                    .len()
                    .checked_mul(physical_width)
                    .ok_or(MasterError::SizeOverflow)?;
                let mut prepared = Vec::with_capacity(capacity);
                for &lane in &pending {
                    prepared.extend_from_slice(
                        &input[lane * width..lane * width + physical_width],
                    );
                }
                let evaluated = workspace
                    .evaluate_batch(&prepared, pending.len(), accuracy)
                    .map_err(MasterError::Evaluation)?;
                result
                    .series_execution
                    .push((order, workspace.execution.clone()));
                let mut next = Vec::new();
                for (lane, evaluated) in pending.into_iter().zip(evaluated) {
                    result.diagnostics[lane].attempts.push(SeriesAttempt {
                        order,
                        status: evaluated.status.clone(),
                        rho: evaluated.rho,
                    });
                    if evaluated.status == SeriesStatus::Success {
                        let finite = evaluated.values.ok_or_else(|| {
                            MasterError::Evaluation("certified lane omitted values".into())
                        })?;
                        if finite.len() != c || evaluated.bounds.len() != c {
                            return Err(MasterError::Evaluation(
                                "certified lane omitted components".into(),
                            ));
                        }
                        // Every pole coefficient was initialized to zero. This
                        // branch is finite throughout its proved neighborhood.
                        let start = lane * result_width;
                        result.results.values[start..start + c].clone_from_slice(&finite);
                        result.results.errors[lane] = None;
                        result.diagnostics[lane].formula = Formula::NearVacuum { order };
                        result.diagnostics[lane].finite = Some(evaluated.bounds);
                    } else {
                        let refine = order == 2
                            && matches!(&evaluated.status, SeriesStatus::UncertifiedComponent(_));
                        result.results.errors[lane] = Some(LaneError::UncertifiedFinite(
                            match evaluated.status {
                                SeriesStatus::UncertifiedComponent(component) => Some(component),
                                _ => None,
                            },
                        ));
                        if refine {
                            next.push(lane);
                        } else {
                            ordinary_pending.push(lane);
                        }
                    }
                }
                pending = next;
            }
        } else {
            ordinary_pending.append(&mut pending);
        }

        if policy == CertificatePolicy::AllowOrdinaryUncertified && !ordinary_pending.is_empty() {
            let capacity = ordinary_pending
                .len()
                .checked_mul(width)
                .ok_or(MasterError::SizeOverflow)?;
            let mut ordinary_input = Vec::with_capacity(capacity);
            for &lane in &ordinary_pending {
                ordinary_input.extend_from_slice(&input[lane * width..(lane + 1) * width]);
            }
            result.ordinary_rows = ordinary_pending.len();
            let ordinary = self.ordinary.evaluate_batch(&ordinary_input, ordinary_pending.len())?;
            for (prepared, lane) in ordinary_pending.into_iter().enumerate() {
                result.results.errors[lane] = ordinary.errors[prepared].clone();
                if ordinary.errors[prepared].is_none() {
                    result.results.values[lane * result_width..(lane + 1) * result_width]
                        .clone_from_slice(
                            &ordinary.values[prepared * result_width..(prepared + 1) * result_width],
                        );
                    result.diagnostics[lane].formula = Formula::OrdinaryUncertified;
                }
            }
        }
        Ok(result)
    }
}
