//! Private restricted N7 dispatcher: one coalesced N5 batch, then a guarded
//! two-level Symbolica parent. No numerical cofactor pruning is performed.
//!
//! Every one of the 21 N5 subsets must succeed. Active inner N6 inverse failures
//! remain failures, even if a different N7 chart is regular. This conservative
//! domain is not general exceptional-kinematics or shifted-master completion.
use crate::{
    Algorithm, BatchResult, CertificateExecutionStats, Convention, ExecutionStats, InputLayout, KinematicsBatch, LaneStatus,
    ReductionError, ReductionPlan, ReductionSettings, ResultMetadata, TensorWeights, Workspace,
    basis::Monomial, execution::{WorkspacePolicy, validate_input_row},
};
use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{
    domains::{dual::DualNumberStructure, float::Complex},
    evaluate::EvaluationDomain,
};

#[derive(Clone, Debug)]
pub(crate) struct ThinSevenReductionPlan {
    parent: ReductionPlan,
    child: ReductionPlan,
}

impl ThinSevenReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        let child_rank = settings.rank.saturating_sub(2);
        let profile = std::env::var_os("IREGI_PV_PREPARATION_PROFILE").is_some();
        let _parent = crate::pv::PreparationTimer::new(profile, format_args!("thin N7 parent plan"));
        let parent = ReductionPlan::new_thin_seven(settings, 0)?;
        drop(_parent);
        let mut child_settings = ReductionSettings::new(5, child_rank);
        child_settings.algorithm = Algorithm::PassarinoVeltman;
        let _child = crate::pv::PreparationTimer::new(profile, format_args!("thin N7 generic N5 child plan"));
        let child = ReductionPlan::new(child_settings)?;
        drop(_child);
        Ok(Self { parent, child })
    }

    pub fn basis(&self) -> &[Monomial] {
        self.parent.basis()
    }

    pub fn settings(&self) -> &ReductionSettings {
        self.parent.settings()
    }

    pub fn alias_count(&self) -> usize {
        self.parent.aliases().len()
    }

    pub(crate) fn input_layout(&self) -> &InputLayout {
        self.parent.input_layout()
    }

    pub(crate) fn contraction_plan(&self, monomials: &[Monomial]) -> Result<crate::ContractionPlan, ReductionError> {
        crate::ContractionPlan::new(&self.parent, monomials)
    }
}

#[derive(Clone, Debug, Default)]
pub struct SevenPointExecutionStats {
    pub parent_input_rows: usize,
    pub child_batch_calls: usize,
    pub packed_child_rows: usize,
    pub parent_batch_calls: usize,
    pub packed_parent_rows: usize,
    pub child_execution: ExecutionStats,
    pub parent_execution: ExecutionStats,
    pub child_certificate_execution: CertificateExecutionStats,
    pub parent_certificate_execution: CertificateExecutionStats,
}

#[derive(Clone)]
pub(crate) struct ThinSevenWorkspace<T: NativeFloat> {
    prototype: T,
    accuracy_bits: u32,
    components: usize,
    metadata: ResultMetadata,
    basis: Vec<Monomial>,
    child_basis: Vec<Monomial>,
    child: Workspace<T>,
    parent: Workspace<T>,
    stats: SevenPointExecutionStats,
    diagnostics: Vec<Vec<String>>,
    #[cfg(test)]
    pub(crate) component_probe_children: Option<Box<BatchResult<T>>>,
}

impl<T: NativeFloat> ThinSevenWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &ThinSevenReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        let child = Workspace::new_with_policies(&plan.child, prototype.clone(), WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::Symbolica)?;
        let parent = Workspace::new_with_policies(&plan.parent, prototype.clone(), WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica)?;
        Ok(Self {
            accuracy_bits: prototype.get_precision(),
            prototype,
            components: 1,
            metadata: ResultMetadata::new(Convention::IregiNative, TensorWeights::Components, None),
            basis: plan.basis().to_vec(),
            child_basis: plan.child.basis().to_vec(),
            child,
            parent,
            stats: Default::default(),
            diagnostics: Vec::new(),
            #[cfg(test)] component_probe_children: None,
        })
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &ThinSevenReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(&dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        let shape: Vec<Vec<usize>> = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect();
        let order = shape.iter().try_fold(0u16, |max, row| {
            row.iter()
                .try_fold(0usize, |sum, &n| sum.checked_add(n))
                .and_then(|n| u16::try_from(n).ok())
                .map(|n| max.max(n))
                .ok_or_else(|| ReductionError::Preparation("Dual order exceeds u16".into()))
        })?;
        let parent_plan = ReductionPlan::new_thin_seven(plan.settings().clone(), order)?;
        let child = Workspace::new_dual_with_policies(&plan.child, prototype.clone(), BorrowedDual(&dual), WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::Symbolica)?;
        let parent = Workspace::new_dual_with_policies(&parent_plan, prototype.clone(), BorrowedDual(&dual), WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica)?;
        Ok(Self {
            accuracy_bits: prototype.get_precision(),
            prototype,
            components: shape.len(),
            metadata: ResultMetadata::new(
                Convention::IregiNative,
                TensorWeights::Components,
                Some(shape),
            ),
            basis: plan.basis().to_vec(),
            child_basis: plan.child.basis().to_vec(),
            child,
            parent,
            stats: Default::default(),
            diagnostics: Vec::new(),
            #[cfg(test)] component_probe_children: None,
        })
    }

    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(
                "invalid thin N7 accuracy precision".into(),
            ));
        }
        self.child.set_accuracy_precision(bits)?;
        self.parent.set_accuracy_precision(bits)?;
        self.accuracy_bits = bits;
        Ok(())
    }

    pub(crate) fn component_count(&self) -> usize { self.components }

    pub(crate) fn binary_precision(&self) -> u32 { self.prototype.get_precision() }

    pub fn last_execution(&self) -> &SevenPointExecutionStats {
        &self.stats
    }

    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }

    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.stats = SevenPointExecutionStats {
            parent_input_rows: batch.rows,
            ..Default::default()
        };
        self.diagnostics.clear();
        #[cfg(test)] { self.component_probe_children = None; }
        let input_width = checked_product(36, self.components)?;
        if self.components == 0
            || batch.components != self.components
            || batch.rows.checked_mul(input_width) != Some(batch.values.len())
        {
            return Err(ReductionError::InvalidInput(
                "thin N7 input dimensions do not match plan".into(),
            ));
        }
        let width = checked_product(checked_product(self.basis.len(), 4)?, self.components)?;
        let output_len = checked_product(batch.rows, width)?;
        // Check every row-sized dimension before allocating diagnostic/output buffers.
        self.diagnostics.resize_with(batch.rows, Vec::new);
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = BatchResult {
            basis: self.basis.clone(),
            components: self.components,
            metadata: self.metadata.clone(),
            values: vec![zero.clone(); output_len],
            statuses: vec![LaneStatus::Success; batch.rows],
            fallback_used: vec![false; batch.rows],
            analytic_used: vec![false; batch.rows],
            primal_envelopes: vec![None; batch.rows],
            primal_envelope_errors: vec![None; batch.rows],
        };
        let layout = InputLayout {
            shifts: 0..28,
            scalar_products: None,
            masses: 28..35,
            scale: 35,
        };
        let mut active = Vec::new();
        for (lane, row) in batch.values.chunks_exact(input_width).enumerate() {
            if let Some(status) = validate_input_row(row, &layout, self.components) {
                result.statuses[lane] = status;
            } else {
                active.push(lane);
            }
        }
        if active.is_empty() {
            return Ok(result);
        }
        let subsets = five_subsets();
        let child_rows = checked_product(subsets.len(), active.len())?;
        let child_width = checked_product(26, self.components)?;
        let mut child_input = KinematicsBatch {
            rows: child_rows,
            components: self.components,
            values: Vec::with_capacity(checked_product(child_rows, child_width)?),
        };
        // Each canonical five-subset appears once, regardless of its two
        // ordered pinch paths. Preserve absolute shifts and every Taylor value.
        for subset in &subsets {
            for &lane in &active {
                let row = &batch.values[lane * input_width..(lane + 1) * input_width];
                for &i in subset {
                    child_input.values.extend_from_slice(
                        &row[4 * i * self.components..4 * (i + 1) * self.components],
                    );
                }
                for &i in subset {
                    child_input.values.extend_from_slice(
                        &row[(28 + i) * self.components..(29 + i) * self.components],
                    );
                }
                child_input
                    .values
                    .extend_from_slice(&row[35 * self.components..36 * self.components]);
            }
        }
        self.stats.child_batch_calls = 1;
        self.stats.packed_child_rows = child_input.rows;
        // This public child entry starts a fresh proof domain. No parent
        // certificate is transported across the changed subset/lane layout.
        let children = self.child.reduce_batch(&child_input);
        self.stats.child_execution = self.child.last_execution();
        self.stats.child_certificate_execution = self.child.last_planar_certificate_execution();
        for (compact, &lane) in active.iter().enumerate() {
            for subset in 0..21 {
                if let Some(notes) = self
                    .child
                    .last_diagnostics()
                    .get(subset * active.len() + compact)
                {
                    self.diagnostics[lane].extend(
                        notes
                            .iter()
                            .map(|note| format!("N5 subset {subset}: {note}")),
                    );
                }
            }
        }
        // All attempted work is harvested before propagating a global error.
        let children = children?;
        validate_result_layout(
            &children,
            &self.child_basis,
            self.components,
            &self.metadata,
            child_input.rows,
        )?;
        #[cfg(test)] if self.components > 1 {
            self.component_probe_children = Some(Box::new(children.clone()));
        }
        let imported_slots = checked_product(21, checked_product(self.child_basis.len(), 4)?)?;
        let parent_width = checked_product(
            36usize
                .checked_add(checked_product(2, imported_slots)?)
                .ok_or_else(dimension_error)?,
            self.components,
        )?;
        let mut parent_input = KinematicsBatch {
            rows: 0,
            components: self.components,
            values: Vec::with_capacity(checked_product(active.len(), parent_width)?),
        };
        let mut parent_lanes = Vec::new();
        for (compact, &lane) in active.iter().enumerate() {
            let state = inspect_required_children(
                &children,
                active.len(),
                compact,
                self.prototype.get_precision(),
                self.accuracy_bits,
            );
            result.fallback_used[lane] = state.fallback_used;
            result.analytic_used[lane] = state.analytic_used;
            if let Some(error) = state.failure {
                self.diagnostics[lane].push(error.clone());
                result.statuses[lane] = LaneStatus::Unstable(error);
                continue;
            }
            parent_lanes.push(lane);
            parent_input.rows += 1;
            parent_input
                .values
                .extend_from_slice(&batch.values[lane * input_width..(lane + 1) * input_width]);
            for subset in 0..21 {
                parent_input.values.extend_from_slice(
                    children
                        .lane(subset * active.len() + compact)
                        .expect("validated required child"),
                );
            }
            for subset in 0..21 {
                let child_row = subset * active.len() + compact;
                if self.components > 1 {
                    // These are separately prepared, positive Taylor sensitivity
                    // operands, never derivatives of the primal ABS estimate.
                    parent_input.values.extend_from_slice(
                        children.validated_component_amplitudes(
                            child_row, self.prototype.get_precision(), self.accuracy_bits,
                        ).expect("validated required child component context"),
                    );
                } else {
                    let envelope = children.validated_primal_envelope(
                        child_row, self.prototype.get_precision(), self.accuracy_bits,
                    ).expect("validated required child envelope");
                    for operand in &envelope.operands {
                        parent_input.values.push(operand.amplitude.clone());
                    }
                }
            }
        }
        if parent_input.rows == 0 {
            return Ok(result);
        }
        if parent_input.values.len() != checked_product(parent_input.rows, parent_width)? {
            return Err(ReductionError::Evaluation(
                "thin N7 imported operand width mismatch".into(),
            ));
        }
        self.stats.parent_batch_calls = 1;
        self.stats.packed_parent_rows = parent_input.rows;
        let parent = self.parent.reduce_batch(&parent_input);
        self.stats.parent_execution = self.parent.last_execution();
        self.stats.parent_certificate_execution = self.parent.last_planar_certificate_execution();
        for (compact, &lane) in parent_lanes.iter().enumerate() {
            if let Some(notes) = self.parent.last_diagnostics().get(compact) {
                self.diagnostics[lane].extend(notes.iter().cloned());
            }
        }
        let parent = parent?;
        validate_result_layout(
            &parent,
            &self.basis,
            self.components,
            &self.metadata,
            parent_input.rows,
        )?;
        for (compact, &lane) in parent_lanes.iter().enumerate() {
            result.statuses[lane] = parent.statuses[compact].clone();
            result.fallback_used[lane] |= parent.fallback_used[compact];
            result.analytic_used[lane] |= parent.analytic_used[compact];
            if let Ok(values) = parent.lane(compact) {
                result.values[lane * width..(lane + 1) * width].clone_from_slice(values);
                result.primal_envelopes[lane] = parent.primal_envelopes[compact].clone();
                result.primal_envelope_errors[lane] =
                    parent.primal_envelope_errors[compact].clone();
            }
        }
        Ok(result)
    }
}

/// Canonical subset order: lexicographic pair of omitted original lines.
fn five_subsets() -> Vec<[usize; 5]> {
    (0..7)
        .flat_map(|a| {
            ((a + 1)..7).map(move |b| {
                let selected: Vec<_> = (0..7).filter(|&i| i != a && i != b).collect();
                std::array::from_fn(|i| selected[i])
            })
        })
        .collect()
}

pub(crate) fn subset_index(a: usize, b: usize) -> Option<usize> {
    if a >= 7 || b >= 7 || a == b {
        return None;
    }
    let pair = (a.min(b), a.max(b));
    (0..7)
        .flat_map(|i| ((i + 1)..7).map(move |j| (i, j)))
        .position(|p| p == pair)
}

struct RequiredChildState {
    fallback_used: bool,
    analytic_used: bool,
    failure: Option<String>,
}

fn inspect_required_children<T: NativeFloat>(
    children: &BatchResult<T>,
    packed_parent_rows: usize,
    compact: usize,
    working_bits: u32,
    accuracy_bits: u32,
) -> RequiredChildState {
    let mut state = RequiredChildState {
        fallback_used: false,
        analytic_used: false,
        failure: None,
    };
    if compact >= packed_parent_rows
        || packed_parent_rows.checked_mul(21) != Some(children.rows())
        || children.fallback_used.len() != children.rows()
        || children.analytic_used.len() != children.rows()
    {
        state.failure = Some("thin N7 required-child row layout mismatch".into());
        return state;
    }
    for subset in 0..21 {
        let row = subset * packed_parent_rows + compact;
        state.fallback_used |= children.fallback_used[row];
        state.analytic_used |= children.analytic_used[row];
        let failure = match children.statuses[row] {
            LaneStatus::Success => children
                .validated_primal_envelope(row, working_bits, accuracy_bits)
                .and_then(|_| {
                    if children.components > 1 {
                        children.validated_component_amplitudes(row, working_bits, accuracy_bits)
                            .map(|_| ())
                    } else {
                        Ok(())
                    }
                })
                .err(),
            ref status => Some(format!("{status:?}")),
        };
        if state.failure.is_none() {
            state.failure = failure.map(|error| format!("N5 subset {subset}: required child or actual-route sensitivity context unavailable: {error}"));
        }
    }
    state
}

fn validate_result_layout<T: NativeFloat>(
    result: &BatchResult<T>,
    basis: &[Monomial],
    components: usize,
    metadata: &ResultMetadata,
    rows: usize,
) -> Result<(), ReductionError> {
    let values = checked_product(
        checked_product(checked_product(rows, basis.len())?, 4)?,
        components,
    )?;
    if components == 0
        || result.basis != basis
        || result.components != components
        || result.metadata() != metadata
        || result.rows() != rows
        || result.values.len() != values
        || result.fallback_used.len() != rows
        || result.analytic_used.len() != rows
        || result.primal_envelopes.len() != rows
        || result.primal_envelope_errors.len() != rows
    {
        return Err(ReductionError::Evaluation(
            "thin N7 complete result layout mismatch".into(),
        ));
    }
    Ok(())
}

fn dimension_error() -> ReductionError {
    ReductionError::InvalidInput("thin N7 dimensions overflow".into())
}
fn checked_product(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b).ok_or_else(dimension_error)
}

struct BorrowedDual<'a, D>(&'a D);
impl<D: DualNumberStructure> DualNumberStructure for BorrowedDual<'_, D> {
    fn get_len(&self) -> usize {
        self.0.get_len()
    }
    fn get_shape(&self) -> Vec<&[usize]> {
        self.0.get_shape()
    }
    fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
        self.0.get_multiplication_table()
    }
    fn get_max_depth(&self) -> usize {
        self.0.get_max_depth()
    }
}

#[cfg(test)]
mod tests;

// Private ignored correctness probe; does not expand the public N7 domain.
#[cfg(test)]
mod numeric_probe;
