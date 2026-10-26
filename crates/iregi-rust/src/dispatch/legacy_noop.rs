//! Conservative proof that the requested highpoint Legacy operation is
//! unchanged, before preparing any Legacy recurrence. Accepted physical rows
//! enter the existing Preserve facade as one batch; all arithmetic remains in
//! the existing policy, exact-certificate and reduction engines.
mod guard;
mod queries;
use super::*;
use crate::{LaneStatus, ResultMetadata, basis::tensor_basis};
use guard::NoOpGuard;
use std::time::Instant;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::HyperDual,
};

pub(super) fn highpoint_legacy_domain(settings: &ReductionSettings) -> bool {
    crate::snapping::highpoint_legacy_domain(settings)
        && settings.powers.len() == settings.propagators
        && !settings.powers.contains(&0)
}
fn invalid(s: impl Into<String>) -> ReductionError {
    ReductionError::InvalidInput(s.into())
}
fn mul(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("Legacy no-op buffer dimensions overflow"))
}
fn add(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Legacy no-op buffer dimensions overflow"))
}

#[derive(Clone, Debug)]
pub(super) struct LegacyNoOpPlan {
    settings: ReductionSettings,
    queries: Arc<queries::LegacyNoOpQueries>,
    basis: Vec<Monomial>,
    preserve: Arc<OnceLock<Result<Arc<TensorReductionPlan>, ReductionError>>>,
    notes: Vec<String>,
}
impl LegacyNoOpPlan {
    pub(super) fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let queries = queries::LegacyNoOpQueries::prepare(&settings)?;
        Ok(Self { basis:tensor_basis(settings.rank), settings, queries:Arc::new(queries),
            preserve:Arc::new(OnceLock::new()), notes:vec![
                "LegacyNoOp: complete finite superset of possible original distance/PV Gram queries; selected zero/real-mass residuals must vanish in every ordered Taylor component before Preserve execution.".into(),
                "This conservative admission can reject a changed query unused by the final recurrence. No exact demanded-query or geometry-changing Legacy coverage is claimed; explicit generic_plan retains the original requested Legacy path.".into(),
            ] })
    }
    pub(super) fn settings(&self) -> &ReductionSettings {
        &self.settings
    }
    pub(super) fn basis(&self) -> &[Monomial] {
        &self.basis
    }
    pub(super) fn input_layout(&self) -> &InputLayout {
        &self.queries.layout
    }
    pub(super) fn parameter_count(&self) -> usize {
        self.queries.parameters.len()
    }
    pub(super) fn preparation_notes(&self) -> &[String] {
        &self.notes
    }
    pub(super) fn contraction_plan(
        &self,
        terms: &[Monomial],
    ) -> Result<ContractionPlan, ReductionError> {
        ContractionPlan::from_basis(
            &self.basis,
            self.settings.convention,
            self.settings.tensor_weights,
            terms,
        )
    }
    fn preserve_plan(&self) -> Result<Arc<TensorReductionPlan>, ReductionError> {
        self.preserve
            .get_or_init(|| {
                TensorReductionPlan::new(self.queries.preserve_settings.clone()).map(Arc::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(clone_error)
    }
}

#[derive(Clone)]
pub(super) struct LegacyNoOpWorkspace<T: NativeFloat> {
    plan: Arc<LegacyNoOpPlan>,
    prototype: T,
    options: TensorWorkspaceOptions,
    shape: Option<Vec<Vec<usize>>>,
    components: usize,
    accuracy_bits: u32,
    guard: NoOpGuard<T>,
    child: Option<Box<TensorWorkspace<T>>>,
    preparation_error: Option<Arc<ReductionError>>,
    input_rows: usize,
    accepted_rows: usize,
    rejected_rows: usize,
    certificate: CertificateExecutionStats,
    last_child: Option<Box<TensorEngineExecution>>,
    preparation_attempts: usize,
    last_preparation_error: Option<String>,
    diagnostics: Vec<Vec<String>>,
    profile: ExecutionProfile,
}

impl<T: NativeFloat> LegacyNoOpWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn new(
        plan: Arc<LegacyNoOpPlan>,
        prototype: T,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        Self::prepare::<HyperDual<u8>>(plan, prototype, None, options)
    }
    pub(super) fn new_dual<D: DualNumberStructure>(
        plan: Arc<LegacyNoOpPlan>,
        prototype: T,
        dual: D,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(&dual).map_err(|e| invalid(e.to_string()))?;
        Self::prepare(plan, prototype, Some(&dual), options)
    }
    fn prepare<D: DualNumberStructure>(
        plan: Arc<LegacyNoOpPlan>,
        prototype: T,
        dual: Option<&D>,
        options: TensorWorkspaceOptions,
    ) -> Result<Self, ReductionError> {
        options
            .master_policy
            .validate_available()
            .map_err(|e| ReductionError::Preparation(e.to_string()))?;
        if prototype.get_precision() == 0 {
            return Err(invalid("working precision must be positive"));
        }
        let shape = dual.map(|d| {
            d.get_shape()
                .into_iter()
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>()
        });
        let guard = NoOpGuard::prepare(&plan.queries, prototype.clone(), dual)?;
        Ok(Self {
            plan,
            accuracy_bits: prototype.get_precision(),
            prototype,
            options,
            components: shape.as_ref().map_or(1, Vec::len),
            shape,
            guard,
            child: None,
            preparation_error: None,
            input_rows: 0,
            accepted_rows: 0,
            rejected_rows: 0,
            certificate: Default::default(),
            last_child: None,
            preparation_attempts: 0,
            last_preparation_error: None,
            diagnostics: Vec::new(),
            profile: Default::default(),
        })
    }
    pub(super) fn component_count(&self) -> usize {
        self.components
    }
    pub(super) fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.shape.as_deref()
    }
    pub(super) fn binary_precision(&self) -> u32 {
        self.prototype.get_precision()
    }
    pub(super) fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(invalid(
                "accuracy precision must be within working precision",
            ));
        }
        if let Some(child) = &mut self.child {
            child.set_accuracy_precision(bits)?;
        }
        self.accuracy_bits = bits;
        Ok(())
    }
    pub(super) fn set_profiling(&mut self, on: bool) -> Result<(), ReductionError> {
        if let Some(child) = &mut self.child {
            child.set_profiling(on)?;
        }
        self.options.profiling = on;
        Ok(())
    }
    pub(super) fn last_profile(&self) -> Option<ExecutionProfile> {
        self.options.profiling.then_some(self.profile)
    }
    pub(super) fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }
    pub(super) fn engine_execution(&self) -> TensorEngineExecution {
        TensorEngineExecution::LegacyNoOp {
            input_rows: self.input_rows,
            accepted_rows: self.accepted_rows,
            rejected_rows: self.rejected_rows,
            query_count: self.plan.queries.queries.len(),
            certificate_execution: self.certificate,
            child: self.last_child.clone(),
            child_preparation_attempts: self.preparation_attempts,
            child_preparation_error: self.last_preparation_error.clone(),
        }
    }
    fn reset(&mut self, rows: usize) {
        self.input_rows = rows;
        self.accepted_rows = 0;
        self.rejected_rows = 0;
        self.certificate = Default::default();
        self.last_child = None;
        self.preparation_attempts = 0;
        self.last_preparation_error = None;
        self.diagnostics.clear();
        self.diagnostics.resize_with(rows, Vec::new);
        self.profile = Default::default();
    }
    fn prepare_child(&mut self) -> Result<(), ReductionError> {
        if self.child.is_some() {
            return Ok(());
        }
        if let Some(error) = &self.preparation_error {
            return Err(clone_error(error));
        }
        self.preparation_attempts = add(self.preparation_attempts, 1)?;
        let prepared = (|| {
            let plan = self.plan.preserve_plan()?;
            let mut child = match &self.shape {
                None => {
                    TensorWorkspace::new_with_options(&plan, self.prototype.clone(), self.options)?
                }
                Some(shape) => TensorWorkspace::new_dual_with_options(
                    &plan,
                    self.prototype.clone(),
                    HyperDual::<u8>::new(shape.clone()),
                    self.options,
                )?,
            };
            child.set_accuracy_precision(self.accuracy_bits)?;
            Ok::<_, ReductionError>(child)
        })();
        match prepared {
            Ok(child) => {
                self.child = Some(Box::new(child));
                Ok(())
            }
            Err(error) => {
                self.last_preparation_error = Some(error.to_string());
                self.preparation_error = Some(Arc::new(clone_error(&error)));
                Err(error)
            }
        }
    }
    pub(super) fn reduce_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.reset(input.rows);
        let started = self.options.profiling.then(Instant::now);
        let evaluated = self.reduce_impl(input);
        if let Some(started) = started {
            self.profile.total = started.elapsed();
        }
        evaluated
    }
    fn reduce_impl(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        let guard_started = self.options.profiling.then(Instant::now);
        let selected = self.guard.select(input);
        self.certificate = self.guard.last_certificate_execution();
        if let Some(started) = guard_started {
            self.profile.arguments += started.elapsed();
        }
        let selected = selected?;
        self.certificate = selected.certificate;
        self.accepted_rows = selected.accepted.len();
        self.rejected_rows = input.rows - self.accepted_rows;
        self.diagnostics = selected.diagnostics;
        let output_width = mul(mul(self.plan.basis.len(), 4)?, self.components)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = BatchResult {
            basis: self.plan.basis.clone(),
            metadata: ResultMetadata::new(
                self.plan.settings.convention,
                self.plan.settings.tensor_weights,
                self.shape.clone(),
            ),
            components: self.components,
            values: vec![zero; mul(input.rows, output_width)?],
            statuses: selected.statuses,
            fallback_used: vec![false; input.rows],
            analytic_used: vec![false; input.rows],
            primal_envelopes: Vec::new(),
            primal_envelope_errors: Vec::new(),
        };
        if selected.accepted.is_empty() {
            return Ok(result);
        }
        // No selected proof can escape as initialized successful zeros.
        for &lane in &selected.accepted {
            result.statuses[lane] =
                LaneStatus::MasterFailure("proven Legacy no-op child did not complete".into());
        }
        if let Err(error) = self.prepare_child() {
            self.last_preparation_error = Some(error.to_string());
            for &lane in &selected.accepted {
                result.statuses[lane] = LaneStatus::MasterFailure(format!(
                    "Legacy no-op Preserve preparation failed: {error}"
                ));
            }
            return Ok(result);
        }
        let input_width = mul(self.plan.parameter_count(), self.components)?;
        let mut packed = KinematicsBatch {
            rows: selected.accepted.len(),
            components: self.components,
            values: Vec::with_capacity(mul(selected.accepted.len(), input_width)?),
        };
        for &lane in &selected.accepted {
            packed
                .values
                .extend_from_slice(&input.values[lane * input_width..(lane + 1) * input_width]);
        }
        let child = self.child.as_mut().expect("prepared Preserve child");
        let evaluated = child.reduce_batch(&packed);
        self.last_child = Some(Box::new(child.engine_execution()));
        let child_notes = child.last_diagnostics().to_vec();
        if let Some(p) = child.last_profile() {
            self.profile.arguments += p.arguments;
            self.profile.scalar_masters += p.scalar_masters;
            self.profile.reduction += p.reduction;
        }
        let source = match evaluated {
            Ok(source) => source,
            Err(error) => {
                for &lane in &selected.accepted {
                    result.statuses[lane] = LaneStatus::MasterFailure(format!(
                        "Legacy no-op Preserve batch failed: {error}"
                    ));
                }
                return Ok(result);
            }
        };
        if let Err(error) = validate_layout(&source, &result, selected.accepted.len()) {
            for &lane in &selected.accepted {
                result.statuses[lane] = LaneStatus::MasterFailure(error.clone());
            }
            return Ok(result);
        }
        if !source.primal_envelopes.is_empty() {
            result.primal_envelopes = vec![None; input.rows];
            result.primal_envelope_errors = vec![None; input.rows];
        }
        for (compact, &lane) in selected.accepted.iter().enumerate() {
            if let Some(notes) = child_notes.get(compact) {
                self.diagnostics[lane].extend_from_slice(notes);
            }
            self.diagnostics[lane].push("Requested LegacyOnShell retained: every selected target in the conservative original-query superset was proved identical in all Taylor components; this row used the unchanged Preserve child engine.".into());
            result.statuses[lane] = source.statuses[compact].clone();
            result.fallback_used[lane] = source.fallback_used[compact];
            result.analytic_used[lane] = source.analytic_used[compact];
            if let Ok(values) = source.lane(compact) {
                result.values[lane * output_width..(lane + 1) * output_width]
                    .clone_from_slice(values);
                result.inherit_primal_envelope(
                    lane,
                    &source,
                    compact,
                    crate::types::PrimalEnvelopeRoute::Ordinary,
                );
            } else if !source.primal_envelope_errors.is_empty() {
                result.primal_envelope_errors[lane] =
                    source.primal_envelope_errors[compact].clone();
            }
        }
        Ok(result)
    }
}
fn validate_layout<T: NativeFloat>(
    source: &BatchResult<T>,
    target: &BatchResult<T>,
    rows: usize,
) -> Result<(), String> {
    let width = source
        .basis
        .len()
        .checked_mul(4)
        .and_then(|n| n.checked_mul(source.components))
        .ok_or("Legacy no-op child output dimensions overflow")?;
    let sidecars = source.primal_envelopes.len();
    if source.basis != target.basis
        || source.metadata() != target.metadata()
        || source.components != target.components
        || source.rows() != rows
        || rows.checked_mul(width) != Some(source.values.len())
        || source.fallback_used.len() != rows
        || source.analytic_used.len() != rows
        || source.primal_envelope_errors.len() != sidecars
        || (sidecars != 0 && sidecars != rows)
    {
        return Err("Legacy no-op child complete result interpretation differs".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
