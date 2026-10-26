//! Instruction-major execution of prepared reduction and scalar-master graphs.
//!
//! Sample lanes and Taylor components are independent dimensions. Symbolica
//! compiles all argument transformations and reduction arithmetic; this module
//! only validates and packs their numerical buffers.

#[cfg(test)]
#[path = "master_policy_tests.rs"]
mod master_policy_tests;

#[cfg(test)]
#[path = "execution/policy_composition_tests.rs"]
pub(crate) mod policy_composition_tests;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use iregi_masters::{
    ExactMasterPlan, LaneError, MasterBackend, MasterEvaluator, MasterExecutionPolicy, NativeFloat, Pentagon6Evaluator,
    Pentagon6LaneError, Pentagon6Options, Pentagon6Plan, ScalarIntegral,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        float::Complex,
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

use crate::{
    CertificateExecutionStats,
    basis::Monomial,
    exact_zero_certificate::{
        ExactZeroCertificate, ExactZeroConstraintPlan, ExactZeroConstraintWorkspace,
    },
    ibp::MasterLeafKind,
    plan::{ReductionPlan, ZeroInvariantProfile},
    stability::{
        GuardContext, StabilityPlan, arguments::ArgumentStabilityPlan,
        components::{ComponentSensitivityRecipe, ComponentSensitivityWorkspace},
        master_argument_precision_failure, quadrature::QuadratureStabilityPlan,
    },
    types::{
        BatchResult, InputLayout, KinematicsBatch, LaneStatus, PrimalEnvelope, PrimalEnvelopeRoute,
        ReductionError, ResultMetadata,
    },
};

#[path = "execution/legacy_compatibility.rs"]
mod legacy_compatibility;
use legacy_compatibility::{PreparedLegacyCompatibility, LegacyCompatibilityWorkspace};

pub(crate) mod quadrature_context;
use quadrature_context::{CoupledQuadratureBatch, QuadratureErrorContext, QuadratureErrorOrigin};
#[cfg(test)]
#[path = "execution/quadrature_context_tests.rs"]
mod quadrature_context_tests;

mod planar_fallback;
use planar_fallback::PlanarFallback;
mod affine_fallback;
use affine_fallback::AffineFallback;

/// Immutable execution-only options. Ordinary public constructors never opt in.
/// Numeric child caches belong to one workspace and therefore one policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkspacePolicy {
    pub(crate) retain_primal_envelopes: bool,
    pub(crate) planar_retry: bool,
    /// Retain owning, actual-route output quadrature diagnostics separately.
    pub(crate) retain_quadrature_context: bool,
    /// Imported uncertain observables require an outer, coupled error kernel.
    pub(crate) quadrature_imports_pending: bool,
}

impl WorkspacePolicy {
    pub(crate) const ORDINARY: Self = Self {
        retain_primal_envelopes: false,
        planar_retry: false,
        retain_quadrature_context: false,
        quadrature_imports_pending: false,
    };
    pub(crate) const SIX_POINT_CHILD: Self = Self {
        retain_primal_envelopes: true,
        planar_retry: true,
        retain_quadrature_context: false,
        quadrature_imports_pending: false,
    };
    pub(crate) const RETAIN_ENVELOPES: Self = Self {
        retain_primal_envelopes: true,
        planar_retry: false,
        retain_quadrature_context: false,
        quadrature_imports_pending: false,
    };
    /// Complete base-child recipe: no imported uncertain observables.
    pub(crate) const AFFINE_QUADRATURE_CHILD: Self = Self {
        retain_primal_envelopes: true,
        planar_retry: false,
        retain_quadrature_context: true,
        quadrature_imports_pending: false,
    };
    /// Parent values may depend on imported child integrals. Local absence of
    /// E6 sources cannot establish absence of propagated quadrature errors.
    pub(crate) const AFFINE_QUADRATURE_PARENT: Self = Self {
        retain_primal_envelopes: true,
        planar_retry: false,
        retain_quadrature_context: true,
        quadrature_imports_pending: true,
    };

    fn validate_quadrature_recipe(self, has_external_seeds: bool) -> Result<(), ReductionError> {
        if self.quadrature_imports_pending && !self.retain_quadrature_context {
            return Err(ReductionError::Preparation("pending quadrature imports require retained context".into()));
        }
        if self.retain_quadrature_context && !self.retain_primal_envelopes {
            return Err(ReductionError::Preparation("quadrature context requires actual-route primal envelopes".into()));
        }
        if self.retain_quadrature_context && !self.quadrature_imports_pending && has_external_seeds {
            return Err(ReductionError::Preparation("base-child quadrature policy cannot accept imported observables".into()));
        }
        Ok(())
    }
}

/// Work performed by the last batch, independently of Taylor component count.
///
/// Each scalar family is called once for all its leaves and active samples.
/// Argument and reduction evaluators execute instructions across sample lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    pub input_rows: usize,
    pub argument_rows: usize,
    pub master_family_batches: usize,
    pub master_rows: usize,
    /// D0 rows used inside genuine D6 pentagon quadrature, including refinement.
    /// These are separate from `master_rows`, which counts requested leaves.
    pub pentagon_box_rows: usize,
    pub reduction_rows: usize,
    /// Samples retried by the hybrid algorithm after a failed PV stability check.
    pub fallback_rows: usize,
    pub fallback_successes: usize,
    /// Samples routed through an exact analytic kinematic-limit profile.
    pub analytic_rows: usize,
}

/// Optional elapsed times for one outer reduction call, including its children.
///
/// `arguments` measures invariant/argument and policy-diagnostic evaluators; `scalar_masters`
/// measures the complete master evaluator calls; `reduction` measures the
/// recurrence evaluator and stability inspection. `total` also includes input
/// validation, buffer packing, routing, and result assembly. Stage times are
/// summed across analytic profiles and hybrid retries; total is never summed
/// recursively. Profiling is disabled by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionProfile {
    pub arguments: Duration,
    pub scalar_masters: Duration,
    pub reduction: Duration,
    pub total: Duration,
}

#[derive(Clone)]
struct MasterGroup<T: NativeFloat> {
    leaves: Vec<usize>,
    evaluator: ScalarMasterEvaluator<T>,
    input: Vec<Complex<T>>,
    requests: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarFamily {
    OneLoop(ScalarIntegral),
    Pentagon6D,
}

impl ScalarFamily {
    fn arity(self) -> usize {
        match self {
            Self::OneLoop(family) => family.arity(),
            Self::Pentagon6D => 16,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::OneLoop(family) => family.name(),
            Self::Pentagon6D => "E6",
        }
    }
}

#[derive(Clone)]
enum ScalarMasterEvaluator<T: NativeFloat> {
    OneLoop(MasterEvaluator<T>),
    Pentagon6 {
        evaluator: Pentagon6Evaluator<T>,
        options: Pentagon6Options<T>,
    },
}

impl<T: NativeFloat> ScalarMasterEvaluator<T>
where
    Complex<T>: EvaluationDomain,
{
    fn family(&self) -> ScalarFamily {
        match self {
            Self::OneLoop(evaluator) => ScalarFamily::OneLoop(evaluator.family()),
            Self::Pentagon6 { .. } => ScalarFamily::Pentagon6D,
        }
    }
}

pub(crate) type ExactExecutionCache = Arc<
    Mutex<HashMap<(Option<Vec<Vec<usize>>>, MasterExecutionPolicy), Arc<OnceLock<Result<Arc<PreparedExecution>, String>>>>>,
>;

struct PreparedMasterGroup {
    leaves: Vec<usize>,
    plan: PreparedMasterPlan,
}

enum PreparedMasterPlan {
    Exact(Arc<ExactMasterPlan>),
    Native(ScalarIntegral),
    Pentagon6(Arc<Pentagon6Plan>),
}

#[derive(Default)]
struct PreparedMasterCache {
    one_loop: Vec<(ScalarIntegral, Arc<ExactMasterPlan>)>,
    pentagon6: Option<Arc<Pentagon6Plan>>,
}

/// Original-input distance polynomials and their positive conditioning
/// envelopes. Exact certificates additionally protect numerical zero branches.
struct PreparedDistanceZeros {
    argument_offset: usize,
    precision_arguments: std::ops::Range<usize>,
    pairs: Vec<(usize, usize)>,
    certificate: ExactZeroConstraintPlan,
    conditioning: Arc<ArgumentStabilityPlan>,
    envelope: ExpressionEvaluator<Complex<Rational>>,
}

#[derive(Clone)]
struct DistanceZeroWorkspace<T: NativeFloat> {
    argument_offset: usize,
    precision_arguments: std::ops::Range<usize>,
    pairs: Vec<(usize, usize)>,
    certificate: ExactZeroConstraintWorkspace,
    conditioning: Arc<ArgumentStabilityPlan>,
    envelope: BatchExpressionEvaluator<Complex<T>>,
    envelope_inputs: Vec<Complex<T>>,
    envelope_values: Vec<Complex<T>>,
    input: Vec<Complex<Rational>>,
    selected: Vec<Vec<(usize, usize)>>,
    lanes: Vec<usize>,
}

impl<T: NativeFloat> DistanceZeroWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    fn instantiate(prepared: &PreparedDistanceZeros, prototype: &T) -> Self {
        Self {
            argument_offset: prepared.argument_offset,
            precision_arguments: prepared.precision_arguments.clone(),
            pairs: prepared.pairs.clone(),
            certificate: prepared.certificate.instantiate(),
            conditioning: prepared.conditioning.clone(),
            envelope: map_exact(&prepared.envelope, prototype),
            envelope_inputs: Vec::new(),
            envelope_values: Vec::new(),
            input: Vec::new(),
            selected: Vec::new(),
            lanes: Vec::new(),
        }
    }

    fn failures(
        &mut self,
        input: &[Complex<T>],
        arguments: &[Complex<T>],
        input_width: usize,
        argument_width: usize,
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
    ) -> Result<Vec<(usize, String)>, ReductionError> {
        self.input.clear();
        self.selected.clear();
        self.lanes.clear();
        // Map each original Taylor coefficient to its magnitude, then let
        // Symbolica perform the same positive polynomial convolution. Taking
        // abs after a jet product would miss cancellation inside that product.
        self.envelope_inputs.clear();
        self.envelope_inputs.extend(
            input.iter().map(|value| {
                Complex::new(NativeFloat::hypot(&value.re, &value.im), prototype.zero())
            }),
        );
        let rows = input.len() / input_width;
        let envelope_width = checked_mul(self.pairs.len(), components)?;
        self.envelope_values.resize(
            checked_mul(rows, envelope_width)?,
            Complex::new(prototype.zero(), prototype.zero()),
        );
        self.envelope
            .evaluate_batch(&self.envelope_inputs, &mut self.envelope_values, rows)
            .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
        let mut failures = Vec::new();
        for (compact, (row, values)) in input
            .chunks_exact(input_width)
            .zip(arguments.chunks_exact(argument_width))
            .enumerate()
        {
            let raw_start = self.argument_offset * components;
            let raw_values = &values[raw_start..raw_start + envelope_width];
            let envelopes =
                &self.envelope_values[compact * envelope_width..(compact + 1) * envelope_width];
            // Symbolica preserves each input coefficient's precision while
            // evaluating a batch. A high-precision prototype cannot restore
            // precision already lost in lower-precision input operations.
            // Only shift coordinates (or authoritative supplied scalar
            // products) enter raw distances. Mass and renormalization-scale
            // precision belongs to the later active-master request checks.
            let effective_input_bits = row[self.precision_arguments.start * components
                ..self.precision_arguments.end * components]
                .iter()
                .flat_map(|value| [&value.re, &value.im])
                .filter(|value| !value.is_zero())
                .fold(prototype.get_precision(), |bits, value| {
                    bits.min(value.get_precision())
                });
            if let Some(reason) = self.conditioning.inspect_lane(
                raw_values,
                envelopes,
                components,
                prototype,
                accuracy_bits,
                effective_input_bits,
            ) {
                failures.push((compact, reason));
                continue;
            }
            let mut selected = Vec::new();
            for constraint in 0..self.pairs.len() {
                for component in 0..components {
                    if complex_zero(
                        &values[(self.argument_offset + constraint) * components + component],
                    ) {
                        selected.push((constraint, component));
                    }
                }
            }
            if selected.is_empty() {
                continue;
            }
            let start = self.input.len();
            for value in row {
                let (Some(re), Some(im)) = (value.re.exact_rational(), value.im.exact_rational())
                else {
                    self.input.truncate(start);
                    failures.push((compact,
                        "rounded-zero distance cannot be certified from this numeric type's represented inputs".into()));
                    break;
                };
                self.input.push(Complex::new(re, im));
            }
            if self.input.len() != start + input_width {
                continue;
            }
            self.selected.push(selected);
            self.lanes.push(compact);
        }
        if self.lanes.is_empty() {
            return Ok(failures);
        }
        let certificates =
            self.certificate
                .check_selected_batch(&self.input, self.lanes.len(), &self.selected)?;
        for (&compact, certificate) in self.lanes.iter().zip(certificates) {
            let reason = match certificate {
                ExactZeroCertificate::Zero => continue,
                ExactZeroCertificate::Nonzero {
                    constraint,
                    component,
                } => {
                    let (i, j) = self.pairs[constraint];
                    format!(
                        "rounded-zero distance ({i},{j}), Taylor component {component}, is nonzero in the exact represented inputs"
                    )
                }
                ExactZeroCertificate::Unresolved { working_bits } => format!(
                    "rounded-zero distance proof is unresolved at {working_bits} certificate bits"
                ),
            };
            failures.push((compact, reason));
        }
        Ok(failures)
    }
}

struct PreparedQuadrature {
    plan: Arc<QuadratureStabilityPlan>,
    coefficients: ExpressionEvaluator<Complex<Rational>>,
    contraction: ExpressionEvaluator<Complex<Rational>>,
    coefficient_count: usize,
    output_count: usize,
}

impl PreparedQuadrature {
    fn prepare<D: DualNumberStructure>(
        plan: &ReductionPlan,
        stability: &StabilityPlan,
        dual: Option<&D>,
    ) -> Result<Option<Self>, ReductionError> {
        let Some(quadrature) = QuadratureStabilityPlan::prepare(plan, stability)? else {
            return Ok(None);
        };
        let coefficient_count = quadrature.coefficient_expressions().len();
        if coefficient_count == 0
            || quadrature.coefficient_parameters() != plan.parameters.as_slice()
            || quadrature.contraction_parameters().len()
                != coefficient_count + quadrature.leaf_indices().len()
        {
            return Err(ReductionError::Preparation(
                "inconsistent quadrature-stability parameter layout".into(),
            ));
        }
        let coefficients = compile_exact(
            quadrature.coefficient_expressions(),
            quadrature.coefficient_parameters(),
            quadrature.coefficient_aliases(),
            dual,
        )?;
        let contraction = compile_exact(
            quadrature.contraction_expressions(),
            quadrature.contraction_parameters(),
            &[],
            dual,
        )?;
        Ok(Some(Self {
            coefficient_count,
            output_count: quadrature.contraction_expressions().len(),
            plan: Arc::new(quadrature),
            coefficients,
            contraction,
        }))
    }
}

#[derive(Clone)]
struct QuadratureWorkspace<T: NativeFloat> {
    plan: Arc<QuadratureStabilityPlan>,
    coefficients: BatchExpressionEvaluator<Complex<T>>,
    contraction: BatchExpressionEvaluator<Complex<T>>,
    coefficient_count: usize,
    output_count: usize,
    leaf_offsets: Vec<Option<usize>>,
    errors: Vec<Complex<T>>,
    successful_errors: Vec<Complex<T>>,
    coefficient_inputs: Vec<Complex<T>>,
    coefficient_values: Vec<Complex<T>>,
    contraction_inputs: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
}

impl<T: NativeFloat> QuadratureWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    fn instantiate(
        prepared: &PreparedQuadrature,
        prototype: &T,
        leaves: usize,
    ) -> Result<Self, ReductionError> {
        let mut leaf_offsets = vec![None; leaves];
        for (offset, &leaf) in prepared.plan.leaf_indices().iter().enumerate() {
            let slot = leaf_offsets.get_mut(leaf).ok_or_else(|| {
                ReductionError::Preparation("quadrature-stability leaf is out of bounds".into())
            })?;
            if slot.replace(offset).is_some() {
                return Err(ReductionError::Preparation(
                    "duplicate quadrature-stability leaf".into(),
                ));
            }
        }
        Ok(Self {
            plan: prepared.plan.clone(),
            coefficients: map_exact(&prepared.coefficients, prototype),
            contraction: map_exact(&prepared.contraction, prototype),
            coefficient_count: prepared.coefficient_count,
            output_count: prepared.output_count,
            leaf_offsets,
            errors: Vec::new(),
            successful_errors: Vec::new(),
            coefficient_inputs: Vec::new(),
            coefficient_values: Vec::new(),
            contraction_inputs: Vec::new(),
            output: Vec::new(),
        })
    }

    fn reset(
        &mut self,
        rows: usize,
        components: usize,
        zero: &Complex<T>,
    ) -> Result<(), ReductionError> {
        let width = checked_mul(self.plan.leaf_indices().len(), components)?;
        self.errors.resize(checked_mul(rows, width)?, zero.clone());
        self.errors.fill(zero.clone());
        self.successful_errors.clear();
        Ok(())
    }

    fn store_errors(
        &mut self,
        leaf: usize,
        compact: usize,
        components: usize,
        estimates: &[T],
        zero: &T,
    ) -> Result<(), String> {
        if estimates.len() != components {
            return Err("E6 backend returned an inconsistent Taylor error layout".into());
        }
        if estimates
            .iter()
            .any(|estimate| !estimate.is_finite() || estimate < zero)
        {
            return Err("E6 backend returned an invalid Taylor error estimate".into());
        }
        if let Some(offset) = self.leaf_offsets[leaf] {
            let start = (compact * self.plan.leaf_indices().len() + offset) * components;
            for (target, estimate) in self.errors[start..start + components]
                .iter_mut()
                .zip(estimates)
            {
                *target = Complex::new(estimate.clone(), zero.clone());
            }
        }
        Ok(())
    }

    fn select_row(&mut self, compact: usize, components: usize) {
        let width = self.plan.leaf_indices().len() * components;
        self.successful_errors
            .extend_from_slice(&self.errors[compact * width..(compact + 1) * width]);
    }

    fn evaluate(
        &mut self,
        inputs: &[Complex<T>],
        rows: usize,
        input_width: usize,
        reduction_width: usize,
        components: usize,
        zero: &Complex<T>,
    ) -> Result<(), ReductionError> {
        let coefficient_width = checked_mul(self.coefficient_count, components)?;
        let error_width = checked_mul(self.plan.leaf_indices().len(), components)?;
        if self.successful_errors.len() != checked_mul(rows, error_width)? {
            return Err(ReductionError::Evaluation(
                "inconsistent selected quadrature-error rows".into(),
            ));
        }
        self.coefficient_inputs.clear();
        for row in inputs.chunks_exact(reduction_width) {
            self.coefficient_inputs
                .extend_from_slice(&row[..input_width]);
        }
        self.coefficient_values
            .resize(checked_mul(rows, coefficient_width)?, zero.clone());
        self.coefficients
            .evaluate_batch(&self.coefficient_inputs, &mut self.coefficient_values, rows)
            .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
        self.contraction_inputs.clear();
        for (row, coefficients) in self
            .coefficient_values
            .chunks_exact(coefficient_width)
            .enumerate()
        {
            self.contraction_inputs
                .extend(coefficients.iter().map(|value| {
                    Complex::new(NativeFloat::hypot(&value.re, &value.im), zero.im.clone())
                }));
            self.contraction_inputs.extend_from_slice(
                &self.successful_errors[row * error_width..(row + 1) * error_width],
            );
        }
        self.output.resize(
            checked_mul(rows, checked_mul(self.output_count, components)?)?,
            zero.clone(),
        );
        // Symbolica's Dualizer evaluates every positive Taylor product. We
        // take norms only after evaluating coefficient jets, so no absolute
        // value is differentiated and no jet multiplication is handwritten.
        self.contraction
            .evaluate_batch(&self.contraction_inputs, &mut self.output, rows)
            .map_err(|error| ReductionError::Evaluation(error.to_string()))
    }

    fn lane(&self, row: usize, components: usize) -> &[Complex<T>] {
        let width = self.output_count * components;
        &self.output[row * width..(row + 1) * width]
    }
}

/// Immutable exact instructions. This graph contains no numeric samples,
/// rounded coefficients, scratch buffers, or reference back to ReductionPlan.
pub(crate) struct PreparedExecution {
    master_policy: MasterExecutionPolicy,
    has_external_primal_seeds: bool,
    basis: Vec<Monomial>,
    parameters: usize,
    layout: InputLayout,
    shape: Option<Vec<Vec<usize>>>,
    metadata: ResultMetadata,
    argument_offsets: Vec<usize>,
    argument_count: usize,
    classification_offset: Option<usize>,
    activation_offset: usize,
    zero_profile_offset: Option<usize>,
    zero_profile_pairs: Vec<(usize, usize)>,
    distance_zeros: Option<PreparedDistanceZeros>,
    arguments: ExpressionEvaluator<Complex<Rational>>,
    reduction: ExpressionEvaluator<Complex<Rational>>,
    reduction_outputs: usize,
    primal_fused: bool,
    primal_monitor: Option<ExpressionEvaluator<Complex<Rational>>>,
    primal_monitor_outputs: usize,
    snapping_monitor: Option<ExpressionEvaluator<Complex<Rational>>>,
    legacy_compatibility: Option<PreparedLegacyCompatibility>,
    snapping_labels: Vec<String>,
    snapping_positive_threshold: bool,
    snapping_zero_image: bool,
    stability: Arc<StabilityPlan>,
    quadrature: Option<PreparedQuadrature>,
    component_sensitivity: Option<Arc<ComponentSensitivityRecipe>>,
    masters: Vec<PreparedMasterGroup>,
    fallback: Option<Arc<PreparedExecution>>,
    lightlike_equal: Option<Arc<PreparedExecution>>,
    lightlike_unequal: Option<Arc<PreparedExecution>>,
}

impl std::fmt::Debug for PreparedExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedExecution")
            .field("parameters", &self.parameters)
            .field("shape", &self.shape)
            .field("masters", &self.masters.len())
            .finish_non_exhaustive()
    }
}

impl PreparedExecution {
    fn build<D: DualNumberStructure>(
        plan: &ReductionPlan,
        dual: Option<&D>,
        master_plans: &mut PreparedMasterCache,
        master_policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        let shape = dual.map(|value| {
            value
                .get_shape()
                .into_iter()
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>()
        });
        if plan.leaves.len() != plan.master_arguments.len()
            || plan.basis.len() != plan.moments.len()
        {
            return Err(ReductionError::Preparation(
                "inconsistent reduction plan dimensions".into(),
            ));
        }
        let mut argument_offsets = vec![0usize];
        let mut argument_expressions = Vec::new();
        let mut reduction_parameters = plan.parameters.clone();
        let mut grouped_leaves: Vec<(ScalarFamily, Vec<usize>)> = Vec::new();
        for (index, (leaf, arguments)) in plan.leaves.iter().zip(&plan.master_arguments).enumerate()
        {
            let family = match (leaf.kind, leaf.active.len()) {
                (MasterLeafKind::FourDimensional, 1) => ScalarFamily::OneLoop(ScalarIntegral::A0),
                (MasterLeafKind::FourDimensional, 2) => ScalarFamily::OneLoop(ScalarIntegral::B0),
                (MasterLeafKind::FourDimensional, 3) => ScalarFamily::OneLoop(ScalarIntegral::C0),
                (MasterLeafKind::FourDimensional, 4) => ScalarFamily::OneLoop(ScalarIntegral::D0),
                (MasterLeafKind::SixDimensionalPentagon, 5) => ScalarFamily::Pentagon6D,
                (kind, n) => {
                    return Err(ReductionError::Preparation(format!(
                        "scalar leaf {index} has unsupported kind {kind:?} with {n} propagators",
                    )));
                }
            };
            if arguments.len() != family.arity() {
                return Err(ReductionError::Preparation(format!(
                    "scalar leaf {index} expects {} arguments, received {}",
                    family.arity(),
                    arguments.len(),
                )));
            }
            argument_expressions.extend_from_slice(arguments);
            argument_offsets.push(argument_expressions.len());
            reduction_parameters.extend_from_slice(&leaf.slots);
            if let Some((_, leaves)) = grouped_leaves.iter_mut().find(|(kind, _)| *kind == family) {
                leaves.push(index);
            } else {
                grouped_leaves.push((family, vec![index]));
            }
        }
        let analytic_bubble = plan.settings.propagators == 2 && !plan.special_kinematics;
        let classification_offset = analytic_bubble.then_some(argument_expressions.len());
        if analytic_bubble {
            argument_expressions.push(plan.distances[0][1].clone());
            let masses = plan.layout.masses.start;
            argument_expressions.push(&plan.parameters[masses + 1] - &plan.parameters[masses]);
        }
        let zero_profile_offset = plan
            .zero_profile_dispatch_allowed()
            .then_some(argument_expressions.len());
        let mut zero_profile_pairs = Vec::new();
        if zero_profile_offset.is_some() {
            for i in 0..plan.settings.propagators {
                for j in i + 1..plan.settings.propagators {
                    zero_profile_pairs.push((i, j));
                    argument_expressions.push(plan.distances[i][j].clone());
                }
            }
        }
        let distance_offset = argument_expressions.len();
        let mut distance_pairs = Vec::new();
        let mut distance_polynomials = Vec::new();
        for i in 0..plan.settings.propagators {
            for j in i + 1..plan.settings.propagators {
                let polynomial = &plan.raw_distances[i][j];
                if !polynomial.is_zero() {
                    distance_pairs.push((i, j));
                    distance_polynomials.push(polynomial.clone());
                }
            }
        }
        let distance_zeros = if distance_polynomials.is_empty() {
            None
        } else {
            let _conditioning = crate::pv::PreparationTimer::new(
                std::env::var_os("IREGI_PREPARATION_PROFILE").is_some(),
                format_args!("workspace distance-zero conditioning"),
            );
            let conditioning = Arc::new(ArgumentStabilityPlan::prepare(
                &plan.parameters,
                &distance_polynomials,
            )?);
            drop(_conditioning);
            let _envelope = crate::pv::PreparationTimer::new(
                std::env::var_os("IREGI_PREPARATION_PROFILE").is_some(),
                format_args!("workspace distance-zero exact envelope"),
            );
            let envelope = compile_exact(conditioning.expressions(), &plan.parameters, &[], dual)?;
            drop(_envelope);
            let _certificate = crate::pv::PreparationTimer::new(
                std::env::var_os("IREGI_PREPARATION_PROFILE").is_some(),
                format_args!("workspace distance-zero exact certificate"),
            );
            let certificate = match dual {
                Some(layout) => ExactZeroConstraintPlan::new_dual(
                    &plan.parameters,
                    &distance_polynomials,
                    layout,
                ),
                None => ExactZeroConstraintPlan::new(&plan.parameters, &distance_polynomials),
            }?;
            drop(_certificate);
            argument_expressions.extend(distance_polynomials);
            Some(PreparedDistanceZeros {
                argument_offset: distance_offset,
                precision_arguments: plan
                    .layout
                    .scalar_products
                    .clone()
                    .unwrap_or_else(|| plan.layout.shifts.clone()),
                pairs: distance_pairs,
                certificate,
                conditioning,
                envelope,
            })
        };
        let started = preparation_timer();
        let stability = Arc::new(StabilityPlan::prepare(plan)?);
        report_preparation("stability expressions", started);
        let started = preparation_timer();
        let quadrature = PreparedQuadrature::prepare(plan, &stability, dual)?;
        report_preparation("quadrature-stability exact evaluators", started);
        let activation_offset = argument_expressions.len();
        argument_expressions.extend_from_slice(stability.master_activations());
        let argument_count = argument_expressions.len();
        // Snapped invariants can occur in master arguments even when no
        // activation predicate needs their aliases. Preserve the dependency
        // order and avoid redeclaring aliases present in both closures.
        let mut argument_aliases = plan.snapping_aliases.clone();
        for alias in stability.master_activation_aliases() {
            if !argument_aliases
                .iter()
                .any(|(symbol, _)| symbol == &alias.0)
            {
                argument_aliases.push(alias.clone());
            }
        }
        let started = preparation_timer();
        let arguments = compile_exact(
            &argument_expressions,
            &plan.parameters,
            &argument_aliases,
            dual,
        )?;
        report_preparation("argument exact evaluator", started);
        let legacy_compatibility = PreparedLegacyCompatibility::prepare(plan, &stability, dual)?;
        let started = preparation_timer();
        let snapping_labels = stability.snapping_labels().to_vec();
        let snapping_zero_image = stability.snapping_zero_image().is_some();
        let snapping_monitor = if let Some(threshold) = stability.snapping_threshold() {
            let mut expressions = stability.snapping_expressions().to_vec();
            expressions.push(threshold.clone());
            expressions.extend(stability.snapping_zero_image().cloned());
            Some(compile_exact::<D>(
                &expressions,
                &plan.parameters,
                stability.snapping_aliases(),
                None,
            )?)
        } else {
            None
        };
        let snapping_positive_threshold = plan
            .snapping_threshold
            .as_ref()
            .is_some_and(|threshold| !threshold.is_zero());
        report_preparation("snapping diagnostic exact evaluator", started);
        let mut reduction_expressions: Vec<_> = plan.moments.iter().flatten().cloned().collect();
        reduction_expressions.extend_from_slice(stability.expressions());
        let mut reduction_aliases: Vec<_> = plan
            .aliases
            .iter()
            .chain(stability.aliases())
            .cloned()
            .collect();
        let primal_monitor_outputs = stability.primal_expressions().len();
        // Ordinary outputs and conditioning envelopes share the same primal
        // inputs. Lower them together so Symbolica reuses their common DAG.
        // Dual layouts retain a separate primal evaluator: differentiating an
        // absolute-value envelope would change the meaning of the estimate.
        let primal_fused = dual.is_none() && primal_monitor_outputs != 0;
        if primal_fused {
            reduction_expressions.extend_from_slice(stability.primal_expressions());
            reduction_aliases.extend_from_slice(stability.primal_aliases());
        }
        let reduction_outputs = reduction_expressions.len();
        let started = preparation_timer();
        let reduction = compile_exact(
            &reduction_expressions,
            &reduction_parameters,
            &reduction_aliases,
            dual,
        )?;
        report_preparation("reduction exact evaluator", started);
        let started = preparation_timer();
        let primal_monitor = if primal_fused || primal_monitor_outputs == 0 {
            None
        } else {
            let aliases: Vec<_> = reduction_aliases
                .iter()
                .chain(stability.primal_aliases())
                .cloned()
                .collect();
            Some(compile_exact::<D>(
                stability.primal_expressions(),
                &reduction_parameters,
                &aliases,
                None,
            )?)
        };
        report_preparation("primal exact evaluator", started);
        let started = preparation_timer();
        let mut masters = Vec::with_capacity(grouped_leaves.len());
        for (family, leaves) in grouped_leaves {
            let family = match family {
                ScalarFamily::Pentagon6D => {
                    let exact = if let Some(exact) = &master_plans.pentagon6 {
                        exact.clone()
                    } else {
                        let exact = Arc::new(
                            match dual {
                                Some(layout) => Pentagon6Plan::new_dual_with_policy(BorrowedDual(layout), master_policy),
                                None => Pentagon6Plan::new_with_policy(master_policy),
                            }
                            .map_err(|error| ReductionError::Preparation(error.to_string()))?,
                        );
                        master_plans.pentagon6 = Some(exact.clone());
                        exact
                    };
                    masters.push(PreparedMasterGroup {
                        leaves,
                        plan: PreparedMasterPlan::Pentagon6(exact),
                    });
                    continue;
                }
                ScalarFamily::OneLoop(family) => family,
            };
            if master_policy == MasterExecutionPolicy::Default
                && dual.is_none()
                && iregi_masters::default_master_backend() == MasterBackend::RustBatch
            {
                masters.push(PreparedMasterGroup {
                    leaves,
                    plan: PreparedMasterPlan::Native(family),
                });
                continue;
            }
            let exact = if let Some((_, exact)) = master_plans
                .one_loop
                .iter()
                .find(|(kind, _)| *kind == family)
            {
                exact.clone()
            } else {
                let exact = Arc::new(
                    match dual {
                        Some(layout) => ExactMasterPlan::new_dual(family, BorrowedDual(layout)),
                        None => ExactMasterPlan::new(family),
                    }
                    .map_err(|error| ReductionError::Preparation(error.to_string()))?,
                );
                master_plans.one_loop.push((family, exact.clone()));
                exact
            };
            masters.push(PreparedMasterGroup {
                leaves,
                plan: PreparedMasterPlan::Exact(exact),
            });
        }
        report_preparation("scalar master exact evaluators", started);
        let fallback = plan
            .fallback
            .as_deref()
            .map(|fallback| prepare_exact(fallback, dual, master_plans, master_policy))
            .transpose()?;
        let jet_order = shape
            .as_ref()
            .map(|shape| maximum_jet_order(shape.iter().map(Vec::as_slice)))
            .transpose()?
            .unwrap_or(0);
        let (lightlike_equal, lightlike_unequal) = if analytic_bubble {
            let equal = ReductionPlan::lightlike_bubble(plan.settings.clone(), jet_order, true)?;
            let unequal = ReductionPlan::lightlike_bubble(plan.settings.clone(), jet_order, false)?;
            (
                Some(prepare_exact(&equal, dual, master_plans, master_policy)?),
                Some(prepare_exact(&unequal, dual, master_plans, master_policy)?),
            )
        } else {
            (None, None)
        };
        let metadata = ResultMetadata::new(
            plan.settings.convention,
            plan.settings.tensor_weights,
            shape.clone(),
        );
        let component_sensitivity=shape.as_ref().filter(|shape|shape.len()>1)
            .map(|_|Arc::new(ComponentSensitivityRecipe::new(plan)));
        Ok(Self {
            master_policy,
            has_external_primal_seeds: !plan.external_primal_seeds.is_empty(),
            basis: plan.basis.clone(),
            parameters: plan.parameters.len(),
            layout: plan.layout.clone(),
            shape,
            metadata,
            argument_offsets,
            argument_count,
            classification_offset,
            activation_offset,
            zero_profile_offset,
            zero_profile_pairs,
            distance_zeros,
            arguments,
            reduction,
            reduction_outputs,
            primal_fused,
            primal_monitor,
            primal_monitor_outputs,
            snapping_monitor,
            legacy_compatibility,
            snapping_labels,
            snapping_positive_threshold,
            snapping_zero_image,
            stability,
            component_sensitivity,
            quadrature,
            masters,
            fallback,
            lightlike_equal,
            lightlike_unequal,
        })
    }
}

fn prepare_exact<D: DualNumberStructure>(
    plan: &ReductionPlan,
    dual: Option<&D>,
    master_plans: &mut PreparedMasterCache,
    master_policy: MasterExecutionPolicy,
) -> Result<Arc<PreparedExecution>, ReductionError> {
    let shape = dual.map(|layout| {
        layout
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect()
    });
    // Release the map lock before entering recursive preparation. OnceLock
    // prevents concurrent callers from compiling the same ordered shape twice.
    let entry = plan
        .execution_cache
        .lock()
        .map_err(|_| ReductionError::Preparation("exact execution cache lock poisoned".into()))?
        .entry((shape, master_policy))
        .or_default()
        .clone();
    entry
        .get_or_init(|| {
            PreparedExecution::build(plan, dual, master_plans, master_policy)
                .map(Arc::new)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map(Arc::clone)
        .map_err(|error| ReductionError::Preparation(error.clone()))
}

/// Owned evaluators and scratch storage for one immutable reduction plan.
///
/// Independent workspaces may evaluate the same plan concurrently. Constants
/// are mapped from exact rationals at `prototype` precision, without a binary64
/// intermediate. Construct a new workspace when increasing working precision.
/// Input Float values keep their represented values and arithmetic precision;
/// the prototype does not promote them. The requested accuracy is unchanged
/// when conservative active-master argument and output-precision checks mark a
/// sample `Unstable` for explicit precision rescue.
#[derive(Clone)]
pub struct Workspace<T: NativeFloat> {
    policy: WorkspacePolicy,
    master_policy: MasterExecutionPolicy,
    planar_fallback: Option<PlanarFallback<T>>,
    affine_fallback: Option<AffineFallback<T>>,
    planar_certificate_stats: CertificateExecutionStats,
    prototype: T,
    accuracy_bits: u32,
    basis: Vec<Monomial>,
    parameters: usize,
    layout: InputLayout,
    components: usize,
    shape: Option<Vec<Vec<usize>>>,
    metadata: ResultMetadata,
    argument_offsets: Vec<usize>,
    argument_count: usize,
    classification_offset: Option<usize>,
    activation_offset: usize,
    zero_profile_offset: Option<usize>,
    zero_profile_pairs: Vec<(usize, usize)>,
    distance_zeros: Option<DistanceZeroWorkspace<T>>,
    zero_profile_plan: Option<Box<ReductionPlan>>,
    zero_profiles: HashMap<ZeroInvariantProfile, Result<Box<Workspace<T>>, String>>,
    arguments: BatchExpressionEvaluator<Complex<T>>,
    reduction: BatchExpressionEvaluator<Complex<T>>,
    reduction_outputs: usize,
    primal_fused: bool,
    primal_monitor: Option<BatchExpressionEvaluator<Complex<T>>>,
    primal_monitor_outputs: usize,
    snapping_monitor: Option<BatchExpressionEvaluator<Complex<T>>>,
    legacy_compatibility: Option<LegacyCompatibilityWorkspace<T>>,
    snapping_labels: Vec<String>,
    snapping_positive_threshold: bool,
    snapping_zero_image: bool,
    stability: Arc<StabilityPlan>,
    guard_context: GuardContext<T>,
    quadrature: Option<QuadratureWorkspace<T>>,
    last_quadrature_context: QuadratureErrorContext<T>,
    component_sensitivity: Option<ComponentSensitivityWorkspace<T>>,
    masters: Vec<MasterGroup<T>>,
    valid_lanes: Vec<usize>,
    successful_lanes: Vec<usize>,
    input: Vec<Complex<T>>,
    argument_values: Vec<Complex<T>>,
    reduction_inputs: Vec<Complex<T>>,
    successful_inputs: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
    primal_inputs: Vec<Complex<T>>,
    primal_output: Vec<Complex<T>>,
    fallback: Option<Box<Workspace<T>>>,
    lightlike_equal: Option<Box<Workspace<T>>>,
    lightlike_unequal: Option<Box<Workspace<T>>>,
    stats: ExecutionStats,
    profiling: bool,
    profile: ExecutionProfile,
    diagnostics: Vec<Vec<String>>,
}

impl<T: NativeFloat> Workspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &ReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Self::new_with_master_policy(plan, prototype, MasterExecutionPolicy::Default)
    }

    /// Prepare an explicit scalar-master execution policy for this workspace.
    ///
    /// `NativeF64Abc` requires the `native-f64-abc` feature and selects native
    /// A0/B0/C0 only for plain f64 inputs. All other types/families use Symbolica.
    /// The policy is inherited by recursive and lazily specialized workspaces.
    pub fn new_with_master_policy(
        plan: &ReductionPlan,
        prototype: T,
        master_policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        Self::new_with_policies(plan, prototype, WorkspacePolicy::ORDINARY, master_policy)
    }

    pub(crate) fn new_with_policy(
        plan: &ReductionPlan,
        prototype: T,
        policy: WorkspacePolicy,
    ) -> Result<Self, ReductionError> {
        Self::new_with_policies(plan, prototype, policy, MasterExecutionPolicy::Default)
    }

    pub(crate) fn new_with_policies(
        plan: &ReductionPlan,
        prototype: T,
        policy: WorkspacePolicy,
        master_policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        Self::prepare::<HyperDual<Complex<Rational>>>(plan, prototype, None, policy, master_policy)
    }

    /// Compile a static or dynamic Symbolica Dual layout. Inputs and outputs
    /// retain the supplied component order and Taylor coefficient convention.
    pub fn new_dual<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        Self::new_dual_with_master_policy(plan, prototype, dual, MasterExecutionPolicy::Default)
    }

    /// Prepare a Dual layout with an explicit master policy.
    /// Explicit Symbolica and selective-native policies keep the complete Dual
    /// path, including lazily evaluated constant jets, on Symbolica.
    pub fn new_dual_with_master_policy<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: D,
        master_policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        Self::new_dual_with_policies(plan, prototype, dual, WorkspacePolicy::ORDINARY, master_policy)
    }

    pub(crate) fn new_dual_with_policy<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: D,
        policy: WorkspacePolicy,
    ) -> Result<Self, ReductionError> {
        Self::new_dual_with_policies(plan, prototype, dual, policy, MasterExecutionPolicy::Default)
    }

    pub(crate) fn new_dual_with_policies<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: D,
        policy: WorkspacePolicy,
        master_policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        policy.validate_quadrature_recipe(!plan.external_primal_seeds.is_empty())?;
        master_policy.validate_available()
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        iregi_masters::validate_dual_structure(&dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        let order = maximum_jet_order(dual.get_shape())?;
        if order == plan.analytic_jet_order {
            return Self::prepare(plan, prototype, Some(&dual), policy, master_policy);
        }
        let specialized = plan.with_analytic_jet_order(order)?;
        Self::prepare(&specialized, prototype, Some(&dual), policy, master_policy)
    }

    fn prepare<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: Option<&D>,
        policy: WorkspacePolicy,
        master_policy: MasterExecutionPolicy,
    ) -> Result<Self, ReductionError> {
        policy.validate_quadrature_recipe(!plan.external_primal_seeds.is_empty())?;
        master_policy.validate_available()
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        if prototype.get_precision() == 0 {
            return Err(ReductionError::Preparation(
                "working precision must be positive".into(),
            ));
        }
        if let Some(layout) = dual {
            iregi_masters::validate_dual_structure(layout)
                .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        }
        let exact = prepare_exact(plan, dual, &mut PreparedMasterCache::default(), master_policy)?;
        Self::instantiate(Some(plan), prototype, &exact, policy)
    }

    fn instantiate(
        plan: Option<&ReductionPlan>,
        prototype: T,
        exact: &PreparedExecution,
        policy: WorkspacePolicy,
    ) -> Result<Self, ReductionError> {
        policy.validate_quadrature_recipe(exact.has_external_primal_seeds)?;
        let started = preparation_timer();
        let arguments = map_exact(&exact.arguments, &prototype);
        let reduction = map_exact(&exact.reduction, &prototype);
        let primal_monitor = exact
            .primal_monitor
            .as_ref()
            .map(|value| map_exact(value, &prototype));
        let snapping_monitor = exact
            .snapping_monitor
            .as_ref()
            .map(|value| map_exact(value, &prototype));
        let legacy_compatibility = exact.legacy_compatibility.as_ref()
            .map(|prepared| LegacyCompatibilityWorkspace::instantiate(prepared, &prototype));
        let quadrature = exact
            .quadrature
            .as_ref()
            .map(|prepared| {
                QuadratureWorkspace::instantiate(
                    prepared,
                    &prototype,
                    exact.argument_offsets.len() - 1,
                )
            })
            .transpose()?;
        let component_sensitivity = if policy.retain_primal_envelopes {
            exact.component_sensitivity.as_ref().map(|recipe|recipe.instantiate(
                exact.shape.as_ref().expect("component sensitivity requires a Dual shape"), &prototype)).transpose()?
        } else { None };
        let masters = exact
            .masters
            .iter()
            .map(|group| {
                Ok(MasterGroup {
                    leaves: group.leaves.clone(),
                    evaluator: match &group.plan {
                        PreparedMasterPlan::Exact(plan) => plan
                            .instantiate_with_policy(prototype.clone(), exact.master_policy)
                            .map(ScalarMasterEvaluator::OneLoop),
                        PreparedMasterPlan::Native(family) => MasterEvaluator::new_with_backend(
                            *family,
                            prototype.clone(),
                            MasterBackend::RustBatch,
                        )
                        .map(ScalarMasterEvaluator::OneLoop),
                        PreparedMasterPlan::Pentagon6(plan) => {
                            plan.instantiate(prototype.clone()).map(|evaluator| {
                                let mut options = Pentagon6Options::new(&prototype);
                                options.relative_tolerance = pentagon_relative_tolerance(
                                    &prototype,
                                    prototype.get_precision(),
                                );
                                ScalarMasterEvaluator::Pentagon6 { evaluator, options }
                            })
                        }
                    }
                    .map_err(|error| ReductionError::Preparation(error.to_string()))?,
                    input: Vec::new(),
                    requests: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, ReductionError>>()?;
        report_preparation("numeric coefficient mapping", started);
        let fallback = exact
            .fallback
            .as_ref()
            .map(|child| {
                Self::instantiate(
                    plan.and_then(|plan| plan.fallback.as_deref()),
                    prototype.clone(),
                    child,
                    policy,
                )
                .map(Box::new)
            })
            .transpose()?;
        let lightlike_equal = exact
            .lightlike_equal
            .as_ref()
            .map(|child| Self::instantiate(None, prototype.clone(), child, policy).map(Box::new))
            .transpose()?;
        let lightlike_unequal = exact
            .lightlike_unequal
            .as_ref()
            .map(|child| Self::instantiate(None, prototype.clone(), child, policy).map(Box::new))
            .transpose()?;
        let zero_profile_plan = if exact.zero_profile_offset.is_some() {
            Some(Box::new(
                plan.ok_or_else(|| {
                    ReductionError::Preparation(
                        "missing structural plan for exact-zero dispatch".into(),
                    )
                })?
                .clone(),
            ))
        } else {
            None
        };
        let distance_zeros = exact
            .distance_zeros
            .as_ref()
            .map(|prepared| DistanceZeroWorkspace::instantiate(prepared, &prototype));
        let guard_context =
            GuardContext::new(&exact.stability, &prototype, prototype.get_precision());
        Ok(Self {
            policy,
            master_policy: exact.master_policy,
            planar_fallback: if policy.planar_retry {
                plan.and_then(ReductionPlan::automatic_planar_settings)
                    .map(|settings| PlanarFallback::new(settings, policy, exact.master_policy))
            } else {
                None
            },
            affine_fallback: plan.and_then(ReductionPlan::automatic_affine_settings)
                .map(|settings|AffineFallback::new(settings,policy,exact.master_policy)),
            planar_certificate_stats: CertificateExecutionStats::default(),
            guard_context,
            accuracy_bits: prototype.get_precision(),
            prototype,
            basis: exact.basis.clone(),
            parameters: exact.parameters,
            layout: exact.layout.clone(),
            components: exact.shape.as_ref().map_or(1, Vec::len),
            shape: exact.shape.clone(),
            metadata: exact.metadata.clone(),
            argument_offsets: exact.argument_offsets.clone(),
            argument_count: exact.argument_count,
            classification_offset: exact.classification_offset,
            activation_offset: exact.activation_offset,
            zero_profile_offset: exact.zero_profile_offset,
            zero_profile_pairs: exact.zero_profile_pairs.clone(),
            distance_zeros,
            zero_profile_plan,
            zero_profiles: HashMap::new(),
            arguments,
            reduction,
            reduction_outputs: exact.reduction_outputs,
            primal_fused: exact.primal_fused,
            primal_monitor,
            primal_monitor_outputs: exact.primal_monitor_outputs,
            snapping_monitor,
            legacy_compatibility,
            snapping_labels: exact.snapping_labels.clone(),
            snapping_positive_threshold: exact.snapping_positive_threshold,
            snapping_zero_image: exact.snapping_zero_image,
            stability: exact.stability.clone(),
            quadrature,
            last_quadrature_context: QuadratureErrorContext::missing(0),
            component_sensitivity,
            masters,
            fallback,
            lightlike_equal,
            lightlike_unequal,
            valid_lanes: Vec::new(),
            successful_lanes: Vec::new(),
            input: Vec::new(),
            argument_values: Vec::new(),
            reduction_inputs: Vec::new(),
            successful_inputs: Vec::new(),
            output: Vec::new(),
            primal_inputs: Vec::new(),
            primal_output: Vec::new(),
            stats: ExecutionStats::default(),
            profiling: false,
            profile: ExecutionProfile::default(),
            diagnostics: Vec::new(),
        })
    }

    /// Requested policy; individual master evaluators report their actual backend.
    pub fn master_execution_policy(&self) -> MasterExecutionPolicy {
        self.master_policy
    }

    pub fn component_count(&self) -> usize {
        self.components
    }

    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.shape.as_deref()
    }

    pub fn binary_precision(&self) -> u32 {
        self.prototype.get_precision()
    }

    /// Set the accuracy target used by cancellation estimates and finite-master
    /// quadrature. Working
    /// precision remains unchanged; a higher-precision rescue can target the
    /// accuracy originally requested by its caller.
    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(
                "accuracy precision must be positive and no greater than working precision".into(),
            ));
        }
        self.accuracy_bits = bits;
        if let Some(state) = &mut self.planar_fallback {
            state.set_accuracy_precision(bits)?;
        }
        if let Some(state)=&mut self.affine_fallback { state.set_accuracy_precision(bits)?; }
        self.guard_context.set_accuracy(&self.prototype, bits);
        for group in &mut self.masters {
            if let ScalarMasterEvaluator::Pentagon6 { options, .. } = &mut group.evaluator {
                options.relative_tolerance = pentagon_relative_tolerance(&self.prototype, bits);
            }
        }
        for child in [
            &mut self.fallback,
            &mut self.lightlike_equal,
            &mut self.lightlike_unequal,
        ]
        .into_iter()
        .flatten()
        {
            child.set_accuracy_precision(bits)?;
        }
        for child in self
            .zero_profiles
            .values_mut()
            .filter_map(|value| value.as_mut().ok())
        {
            child.set_accuracy_precision(bits)?;
        }
        Ok(())
    }

    pub fn last_execution(&self) -> ExecutionStats {
        self.stats
    }

    /// Per-sample notes from the last call. A failed optional exact-profile
    /// preparation is recorded here while the ordinary reduction is retained.
    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        &self.diagnostics
    }

    /// Enable or disable timing for this workspace and its analytic/fallback plans.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if let Some(state) = &mut self.planar_fallback {
            state.set_profiling(enabled);
        }
        if let Some(state)=&mut self.affine_fallback { state.set_profiling(enabled); }
        for child in [
            &mut self.fallback,
            &mut self.lightlike_equal,
            &mut self.lightlike_unequal,
        ] {
            if let Some(child) = child {
                child.set_profiling(enabled);
            }
        }
        for child in self
            .zero_profiles
            .values_mut()
            .filter_map(|value| value.as_mut().ok())
        {
            child.set_profiling(enabled);
        }
    }

    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }

    /// Evaluate the whole batch with per-lane input, master, and stability status.
    /// Only malformed whole-buffer dimensions return an input error directly.
    pub fn reduce_batch(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        // Reset before even malformed whole-buffer preflight can return.
        self.last_quadrature_context.clear();
        self.planar_certificate_stats = CertificateExecutionStats::default();
        self.profile = ExecutionProfile::default();
        let started = self.profiling.then(Instant::now);
        let mut result = self.reduce_batch_impl(batch);
        if self.policy.planar_retry {
            if let Ok(reduced) = &mut result {
                if self.policy.retain_quadrature_context {
                    for lane in 0..reduced.rows() {
                        if reduced.statuses[lane] != LaneStatus::Success {
                            let _ = self.last_quadrature_context.reject(lane,
                                "automatic planar replacement has no coupled quadrature transport".into());
                        }
                    }
                }
                if let Err(error) = self.retry_planar(batch, reduced) {
                    result = Err(error);
                }
            }
        }
        if let Some(started) = started {
            self.profile.total = started.elapsed();
        }
        if result.is_err() {
            self.last_quadrature_context.clear();
        }
        result
    }

    /// Owning boundary used only by explicit affine composition policies.
    /// Missing/error contexts remain per-lane failures at this boundary; they
    /// never alter the ordinary result's standalone success contract.
    pub(crate) fn reduce_batch_with_quadrature_context(
        &mut self, batch: &KinematicsBatch<T>,
    ) -> Result<CoupledQuadratureBatch<T>, ReductionError> {
        self.last_quadrature_context.clear();
        if !self.policy.retain_quadrature_context {
            self.stats = ExecutionStats { input_rows: batch.rows, ..ExecutionStats::default() };
            self.profile = ExecutionProfile::default();
            self.diagnostics.clear();
            self.planar_certificate_stats = CertificateExecutionStats::default();
            return Err(ReductionError::Unsupported(
                "workspace did not opt into quadrature-context retention".into()));
        }
        let result = self.reduce_batch(batch)?;
        let context = std::mem::replace(&mut self.last_quadrature_context,
            QuadratureErrorContext::missing(0));
        CoupledQuadratureBatch::new(result, context)
    }

    /// Reuse the existing uncached inspector with the amplitudes that describe
    /// these returned values, including actual fallback/scatter routes. The
    /// established combined budget includes roundoff only for component zero;
    /// joint higher-Dual quadrature acceptance requires a separate policy gate.
    pub(crate) fn inspect_imported_quadrature_output(
        &self, checks: &QuadratureStabilityPlan, result: &BatchResult<T>, lane: usize,
        input: &[Complex<T>], dense_errors: &[Complex<T>],
    ) -> Result<Option<String>, String> {
        if result.basis != self.basis || result.metadata() != &self.metadata
            || result.components != self.components
            || self.parameters.checked_mul(self.components) != Some(input.len()) {
            return Err("imported quadrature result/input interpretation differs".into());
        }
        // This validates the row and buffer dimensions before lane() indexes.
        let envelope = result.validated_primal_envelope(lane,
            self.prototype.get_precision(), self.accuracy_bits)?;
        let output = result.lane(lane)
            .map_err(|status| format!("imported quadrature requires success: {status:?}"))?;
        let actual_amplitudes: Vec<_> = envelope.operands.iter()
            .map(|operand| operand.amplitude.clone()).collect();
        Ok(self.stability.inspect_quadrature_lane(checks, dense_errors, output, input,
            self.components, &self.prototype, self.accuracy_bits, Some(&actual_amplitudes)))
    }

    fn reduce_batch_impl(
        &mut self,
        batch: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.stats = ExecutionStats {
            input_rows: batch.rows,
            ..ExecutionStats::default()
        };
        self.diagnostics.clear();
        if batch.components != self.components {
            return Err(ReductionError::InvalidInput(format!(
                "workspace expects {} Taylor components, received {}",
                self.components, batch.components,
            )));
        }
        let input_width = checked_mul(self.parameters, self.components)?;
        let expected = checked_mul(batch.rows, input_width)?;
        if batch.values.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "expected {expected} input values, received {}",
                batch.values.len(),
            )));
        }
        self.diagnostics.resize_with(batch.rows, Vec::new);
        if self.policy.retain_quadrature_context {
            self.last_quadrature_context = QuadratureErrorContext::missing(batch.rows);
        }
        let output_width = checked_mul(checked_mul(self.basis.len(), 4)?, self.components)?;
        let output_len = checked_mul(batch.rows, output_width)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = BatchResult {
            metadata: self.metadata.clone(),
            basis: self.basis.clone(),
            components: self.components,
            values: vec![zero.clone(); output_len],
            statuses: vec![LaneStatus::Success; batch.rows],
            fallback_used: vec![false; batch.rows],
            analytic_used: vec![false; batch.rows],
            primal_envelopes: if self.policy.retain_primal_envelopes { vec![None; batch.rows] } else { Vec::new() },
            primal_envelope_errors: if self.policy.retain_primal_envelopes { vec![None; batch.rows] } else { Vec::new() },
        };
        self.valid_lanes.clear();
        self.successful_lanes.clear();
        self.input.clear();
        for (lane, row) in batch.values.chunks_exact(input_width).enumerate() {
            if let Some(status) = validate_input_row(row, &self.layout, self.components) {
                result.statuses[lane] = status;
                continue;
            }
            self.valid_lanes.push(lane);
            self.input.extend_from_slice(row);
        }
        let mut valid = self.valid_lanes.len();
        if valid == 0 {
            return Ok(result);
        }
        // Prove and dispatch constant-one affine lanes before any parent
        // argument/Gram evaluation or scalar-master activation.
        self.dispatch_affine(batch,&mut result)?;
        if self.policy.retain_quadrature_context {
            for lane in 0..result.rows() {
                if result.analytic_used[lane] {
                    self.last_quadrature_context.reject(lane,
                        "automatic certified affine route has no coupled quadrature transport".into())
                        .map_err(ReductionError::Evaluation)?;
                }
            }
        }
        valid=self.valid_lanes.len();
        if valid==0{return Ok(result);}
        self.stats.argument_rows=self.stats.argument_rows.checked_add(valid).ok_or_else(size_error)?;
        if let Some(monitor) = &mut self.snapping_monitor {
            // Strict policy comparisons only have primal branch semantics.
            // Their residuals must never be differentiated, and rejected
            // lanes must not reach scalar masters or hybrid retries.
            self.primal_inputs.clear();
            self.primal_inputs.extend(
                self.input
                    .chunks_exact(self.components)
                    .map(|components| components[0].clone()),
            );
            let width = self.snapping_labels.len() + 1 + usize::from(self.snapping_zero_image);
            self.primal_output
                .resize(checked_mul(valid, width)?, zero.clone());
            let started = self.profiling.then(Instant::now);
            let evaluated =
                monitor.evaluate_batch(&self.primal_inputs, &mut self.primal_output, valid);
            record_time(started, &mut self.profile.arguments);
            evaluated.map_err(|error| ReductionError::Evaluation(error.to_string()))?;
            let mut selected = Vec::with_capacity(valid);
            for compact in 0..valid {
                let lane = self.valid_lanes[compact];
                let values = &self.primal_output[compact * width..(compact + 1) * width];
                let threshold = &values[self.snapping_labels.len()];
                let invalid_threshold = !finite(threshold)
                    || !threshold.im.is_zero()
                    || (self.snapping_positive_threshold && complex_zero(threshold));
                let mut failure = None;
                for (value, label) in values.iter().zip(&self.snapping_labels) {
                    if invalid_threshold || failure.is_some() {
                        break;
                    }
                    if !finite(value) || !value.im.is_zero() {
                        failure = Some(format!(
                            "legacy on-shell branch cannot be classified at the working precision: {label}"
                        ));
                    } else if complex_zero(value) {
                        if self.components > 1 {
                            failure = Some(format!(
                                "legacy on-shell policy is nondifferentiable at an active strict-comparison boundary: {label}"
                            ));
                        } else {
                            self.diagnostics[lane].push(format!(
                                "legacy on-shell strict-comparison boundary selected its equality branch: {label}"
                            ));
                        }
                    }
                }
                let unsupported_image = if self.snapping_zero_image {
                    let image = &values[self.snapping_labels.len() + 1];
                    if !finite(image) || !image.im.is_zero() {
                        failure = Some(
                            "legacy on-shell zero-image compatibility cannot be classified at the working precision"
                                .into(),
                        );
                        false
                    } else {
                        !complex_zero(image)
                    }
                } else {
                    false
                };
                if invalid_threshold {
                    result.statuses[lane] = LaneStatus::InvalidInput(
                        "legacy on-shell threshold is not representable at the working precision"
                            .into(),
                    );
                } else if unsupported_image {
                    result.statuses[lane] = LaneStatus::InvalidInput(
                        "legacy on-shell policy maps a zero scalar product to a nonzero value; this nonphysical Cayley-diagonal domain is unsupported".into(),
                    );
                } else if let Some(reason) = failure {
                    result.statuses[lane] = LaneStatus::Unstable(reason);
                } else {
                    selected.push(compact);
                }
            }
            self.retain_lanes(&selected, input_width, 0);
            valid = selected.len();
            if valid == 0 {
                return Ok(result);
            }
        }
        if let Some(compatibility) = &mut self.legacy_compatibility {
            let started = self.profiling.then(Instant::now);
            let failures = compatibility.failures(&self.input, valid, self.components, &self.prototype)?;
            record_time(started, &mut self.profile.arguments);
            let mut rejected = vec![false; valid];
            for (compact, status) in failures {
                let lane = self.valid_lanes[compact];
                self.diagnostics[lane].push(format!("{status:?}"));
                result.statuses[lane] = status;
                rejected[compact] = true;
            }
            let selected: Vec<_> = (0..valid).filter(|&row| !rejected[row]).collect();
            self.retain_lanes(&selected, input_width, 0);
            valid = selected.len();
            if valid == 0 { return Ok(result); }
        }
        let argument_width = checked_mul(self.argument_count, self.components)?;
        self.argument_values
            .resize(checked_mul(valid, argument_width)?, zero.clone());
        let started = self.profiling.then(Instant::now);
        let evaluated =
            self.arguments
                .evaluate_batch(&self.input, &mut self.argument_values, valid);
        record_time(started, &mut self.profile.arguments);
        evaluated.map_err(|error| ReductionError::Evaluation(error.to_string()))?;

        if let Some(checks) = &mut self.distance_zeros {
            let started = self.profiling.then(Instant::now);
            let checked = checks.failures(
                &self.input,
                &self.argument_values,
                input_width,
                argument_width,
                self.components,
                &self.prototype,
                self.accuracy_bits,
            );
            record_time(started, &mut self.profile.arguments);
            let failures = checked?;
            if !failures.is_empty() {
                for (compact, reason) in failures {
                    result.statuses[self.valid_lanes[compact]] = LaneStatus::Unstable(reason);
                }
                let retained: Vec<_> = self
                    .valid_lanes
                    .iter()
                    .enumerate()
                    .filter_map(|(compact, &lane)| {
                        (result.statuses[lane] == LaneStatus::Success).then_some(compact)
                    })
                    .collect();
                // Same-precision fallback cannot bypass an invariant check.
                // Unstable results remain available to precision rescue.
                self.retain_lanes(&retained, input_width, argument_width);
                valid = retained.len();
                if valid == 0 {
                    return Ok(result);
                }
            }
        }

        if let Some(offset) = self.classification_offset {
            let mut equal = Vec::new();
            let mut unequal = Vec::new();
            let mut ordinary = Vec::new();
            for compact in 0..valid {
                let point =
                    &self.argument_values[compact * argument_width..(compact + 1) * argument_width];
                let s = &point[offset * self.components];
                let delta = &point[(offset + 1) * self.components];
                let nonzero_mass = !complex_zero(
                    &self.input[compact * input_width + self.layout.masses.start * self.components],
                );
                if complex_zero(s) && !complex_zero(delta) {
                    unequal.push(self.valid_lanes[compact]);
                } else if complex_zero(s) && complex_zero(delta) && nonzero_mass {
                    equal.push(self.valid_lanes[compact]);
                } else {
                    ordinary.push(compact);
                }
            }
            if !equal.is_empty() {
                dispatch_profile_with_context(
                    self.lightlike_equal.as_mut().unwrap(),
                    &equal,
                    batch,
                    &mut result,
                    &mut self.stats,
                    &mut self.profile,
                    &mut self.diagnostics,
                    &mut self.planar_certificate_stats,
                    &mut self.last_quadrature_context,
                )?;
            }
            if !unequal.is_empty() {
                dispatch_profile_with_context(
                    self.lightlike_unequal.as_mut().unwrap(),
                    &unequal,
                    batch,
                    &mut result,
                    &mut self.stats,
                    &mut self.profile,
                    &mut self.diagnostics,
                    &mut self.planar_certificate_stats,
                    &mut self.last_quadrature_context,
                )?;
            }
            self.retain_lanes(&ordinary, input_width, argument_width);
            valid = ordinary.len();
            if valid == 0 {
                return Ok(result);
            }
        }

        if let Some(offset) = self.zero_profile_offset {
            let mut groups: Vec<(ZeroInvariantProfile, Vec<usize>)> = Vec::new();
            let mut ordinary = Vec::new();
            for compact in 0..valid {
                let input = &self.input[compact * input_width..(compact + 1) * input_width];
                let arguments =
                    &self.argument_values[compact * argument_width..(compact + 1) * argument_width];
                let masses: Vec<_> = self
                    .layout
                    .masses
                    .clone()
                    .map(|argument| {
                        input[argument * self.components..(argument + 1) * self.components]
                            .iter()
                            .all(complex_zero)
                    })
                    .collect();
                let distances: Vec<_> = self
                    .zero_profile_pairs
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &pair)| {
                        let start = (offset + index) * self.components;
                        arguments[start..start + self.components]
                            .iter()
                            .all(complex_zero)
                            .then_some(pair)
                    })
                    .collect();
                if !masses.iter().any(|&zero| zero) && distances.is_empty() {
                    ordinary.push(compact);
                    continue;
                }
                let profile = ZeroInvariantProfile { masses, distances };
                if let Some((_, lanes)) = groups.iter_mut().find(|(key, _)| key == &profile) {
                    lanes.push(compact);
                } else {
                    groups.push((profile, vec![compact]));
                }
            }
            for (profile, compact_lanes) in groups {
                if !self.zero_profiles.contains_key(&profile) {
                    let prepared =
                        self.zero_profile_plan
                            .as_ref()
                            .unwrap()
                            .with_zero_invariants(profile.clone())
                            .and_then(|plan| {
                                let layout = self.shape.as_ref().map(|shape| {
                                    HyperDual::<Complex<Rational>>::new(shape.clone())
                                });
                                Self::prepare(&plan, self.prototype.clone(), layout.as_ref(), self.policy, self.master_policy)
                            })
                            .and_then(|mut workspace| {
                                workspace.set_profiling(self.profiling);
                                workspace.set_accuracy_precision(self.accuracy_bits)?;
                                Ok(Box::new(workspace))
                            })
                            .map_err(|error| error.to_string());
                    self.zero_profiles.insert(profile.clone(), prepared);
                }
                match self.zero_profiles.get_mut(&profile).unwrap() {
                    Ok(workspace) => {
                        let lanes: Vec<_> = compact_lanes
                            .iter()
                            .map(|&compact| self.valid_lanes[compact])
                            .collect();
                        dispatch_profile_with_context(
                            workspace,
                            &lanes,
                            batch,
                            &mut result,
                            &mut self.stats,
                            &mut self.profile,
                            &mut self.diagnostics,
                            &mut self.planar_certificate_stats,
                            &mut self.last_quadrature_context,
                        )?;
                        for (&compact, &lane) in compact_lanes.iter().zip(&lanes) {
                            if result.statuses[lane] != LaneStatus::Success {
                                self.diagnostics[lane].push(format!(
                                    "exact-zero profile evaluation failed; ordinary reduction retained: {:?}",
                                    result.statuses[lane]
                                ));
                                result.statuses[lane] = LaneStatus::Success;
                                result.analytic_used[lane] = false;
                                result.fallback_used[lane] = false;
                                if self.policy.retain_quadrature_context {
                                    self.last_quadrature_context.reject(lane,
                                        "failed optional profile context discarded before ordinary retry".into())
                                        .map_err(ReductionError::Evaluation)?;
                                }
                                ordinary.push(compact);
                            }
                        }
                    }
                    Err(reason) => {
                        for &compact in &compact_lanes {
                            self.diagnostics[self.valid_lanes[compact]].push(format!(
                                "exact-zero profile preparation failed; ordinary reduction retained: {reason}"
                            ));
                        }
                        ordinary.extend(compact_lanes);
                    }
                }
            }
            // Retain source ordering so in-place compaction never overwrites
            // an unread sample after a failed optional profile preparation.
            ordinary.sort_unstable();
            self.retain_lanes(&ordinary, input_width, argument_width);
            valid = ordinary.len();
            if valid == 0 {
                return Ok(result);
            }
        }

        let master_slots = checked_mul(self.argument_offsets.len() - 1, 4)?;
        let reduction_width = checked_mul(
            self.parameters
                .checked_add(master_slots)
                .ok_or_else(size_error)?,
            self.components,
        )?;
        self.reduction_inputs
            .resize(checked_mul(valid, reduction_width)?, zero.clone());
        // Inactive leaves receive deterministic placeholders. Their values
        // are never inspected by the selected reduction/monitor branches.
        self.reduction_inputs.fill(zero.clone());
        if let Some(quadrature) = &mut self.quadrature {
            quadrature.reset(valid, self.components, &zero)?;
        }
        for (compact, input) in self.input.chunks_exact(input_width).enumerate() {
            self.reduction_inputs
                [compact * reduction_width..compact * reduction_width + input_width]
                .clone_from_slice(input);
        }

        for group in &mut self.masters {
            let family = group.evaluator.family();
            let width = checked_mul(family.arity(), self.components)?;
            group.input.clear();
            group.requests.clear();
            for &leaf in &group.leaves {
                let offset = self.argument_offsets[leaf] * self.components;
                let active = (self.activation_offset + leaf) * self.components;
                for (compact, row) in self
                    .argument_values
                    .chunks_exact(argument_width)
                    .enumerate()
                {
                    if complex_zero(&row[active]) {
                        continue;
                    }
                    group.requests.push((leaf, compact));
                    group.input.extend_from_slice(&row[offset..offset + width]);
                }
            }
            let rows = group.requests.len();
            if rows == 0 {
                continue;
            }
            self.stats.master_family_batches += 1;
            self.stats.master_rows = self
                .stats
                .master_rows
                .checked_add(rows)
                .ok_or_else(size_error)?;
            let started = self.profiling.then(Instant::now);
            match &mut group.evaluator {
                ScalarMasterEvaluator::OneLoop(evaluator) => {
                    let evaluated = evaluator.evaluate_batch(&group.input, rows);
                    record_time(started, &mut self.profile.scalar_masters);
                    let values =
                        evaluated.map_err(|error| ReductionError::Evaluation(error.to_string()))?;
                    for (request, &(leaf, compact)) in group.requests.iter().enumerate() {
                        let lane = self.valid_lanes[compact];
                        match values.lane(request) {
                            Err(error) => {
                                if matches!(
                                    result.statuses[lane],
                                    LaneStatus::Success | LaneStatus::Unstable(_)
                                ) {
                                    result.statuses[lane] = LaneStatus::MasterFailure(format!(
                                        "{} leaf {leaf}: {error}",
                                        family.name(),
                                    ));
                                }
                            }
                            Ok(value) => {
                                if let Some(reason) = master_argument_precision_failure(
                                    &group.input[request * width..(request + 1) * width],
                                    &self.input[compact * input_width
                                        + self.layout.scale * self.components
                                        ..compact * input_width
                                            + (self.layout.scale + 1) * self.components],
                                    self.components,
                                    &self.prototype,
                                    self.accuracy_bits,
                                ) {
                                    if result.statuses[lane] == LaneStatus::Success {
                                        result.statuses[lane] = LaneStatus::Unstable(format!(
                                            "{} leaf {leaf}: {reason}",
                                            family.name(),
                                        ));
                                    }
                                }
                                let start = compact * reduction_width
                                    + input_width
                                    + leaf * 4 * self.components;
                                for (slot, coefficients) in
                                    [value.uv, value.ir, value.ir2, value.finite]
                                        .into_iter()
                                        .enumerate()
                                {
                                    let offset = start + slot * self.components;
                                    self.reduction_inputs[offset..offset + self.components]
                                        .clone_from_slice(coefficients);
                                }
                            }
                        }
                    }
                }
                ScalarMasterEvaluator::Pentagon6 { evaluator, options } => {
                    // One E6 batch contains every active leaf and sample. Its
                    // backend batches all quadrature nodes and D0 pinches;
                    // Taylor components remain inside each numerical row.
                    let evaluated = evaluator.evaluate_batch(&group.input, rows, options);
                    record_time(started, &mut self.profile.scalar_masters);
                    let values = match evaluated {
                        Ok(values) => values,
                        Err(error) => {
                            for &(leaf, compact) in &group.requests {
                                let lane = self.valid_lanes[compact];
                                let reason =
                                    format!("E6 leaf {leaf}: batch evaluation failed: {error}");
                                self.diagnostics[lane].push(reason.clone());
                                if matches!(
                                    result.statuses[lane],
                                    LaneStatus::Success | LaneStatus::Unstable(_)
                                ) {
                                    result.statuses[lane] = LaneStatus::MasterFailure(reason);
                                }
                            }
                            continue;
                        }
                    };
                    for (request, &(leaf, compact)) in group.requests.iter().enumerate() {
                        let lane = self.valid_lanes[compact];
                        let diagnostic = &values.diagnostics()[request];
                        self.stats.pentagon_box_rows = self
                            .stats
                            .pentagon_box_rows
                            .checked_add(diagnostic.box_rows)
                            .ok_or_else(size_error)?;
                        self.diagnostics[lane].push(format!(
                            "E6 leaf {leaf}: quadrature order {}, box rows {}, error estimate {:?}",
                            diagnostic.order, diagnostic.box_rows, diagnostic.error_estimate,
                        ));
                        match values.lane(request) {
                            Err(error) => {
                                let reason = format!(
                                    "E6 leaf {leaf}: {error:?} (quadrature order {}, box rows {}, error estimate {:?})",
                                    diagnostic.order,
                                    diagnostic.box_rows,
                                    diagnostic.error_estimate,
                                );
                                self.diagnostics[lane].push(reason.clone());
                                let numerical = matches!(
                                    error,
                                    Pentagon6LaneError::NonConvergent
                                        | Pentagon6LaneError::NonFinite
                                        | Pentagon6LaneError::SingularGeometry
                                        | Pentagon6LaneError::BoxFailure {
                                            error: LaneError::NonFiniteOutput,
                                            ..
                                        }
                                );
                                if numerical {
                                    if result.statuses[lane] == LaneStatus::Success {
                                        result.statuses[lane] = LaneStatus::Unstable(reason);
                                    }
                                } else if matches!(
                                    result.statuses[lane],
                                    LaneStatus::Success | LaneStatus::Unstable(_)
                                ) {
                                    // An endpoint/domain failure takes priority
                                    // over a retryable numerical error in another
                                    // leaf of the same sample.
                                    result.statuses[lane] = LaneStatus::MasterFailure(reason);
                                }
                            }
                            Ok(finite) => {
                                if let Some(reason) = master_argument_precision_failure(
                                    &group.input[request * width..(request + 1) * width],
                                    &self.input[compact * input_width
                                        + self.layout.scale * self.components
                                        ..compact * input_width
                                            + (self.layout.scale + 1) * self.components],
                                    self.components,
                                    &self.prototype,
                                    self.accuracy_bits,
                                ) {
                                    if result.statuses[lane] == LaneStatus::Success {
                                        result.statuses[lane] = LaneStatus::Unstable(format!(
                                            "E6 leaf {leaf}: {reason}",
                                        ));
                                    }
                                }
                                if let Some(quadrature) = &mut self.quadrature {
                                    if let Err(reason) = quadrature.store_errors(
                                        leaf,
                                        compact,
                                        self.components,
                                        &diagnostic.component_error_estimates,
                                        &self.prototype.zero(),
                                    ) {
                                        self.diagnostics[lane].push(reason.clone());
                                        if matches!(
                                            result.statuses[lane],
                                            LaneStatus::Success | LaneStatus::Unstable(_)
                                        ) {
                                            result.statuses[lane] =
                                                LaneStatus::MasterFailure(reason);
                                        }
                                        continue;
                                    }
                                }
                                // Only a backend-qualified finite endpoint can
                                // enter this branch. Failed lanes never expose
                                // their initialized zero values as coefficients.
                                let start = compact * reduction_width
                                    + input_width
                                    + leaf * 4 * self.components;
                                self.reduction_inputs[start..start + 3 * self.components]
                                    .fill(zero.clone());
                                self.reduction_inputs
                                    [start + 3 * self.components..start + 4 * self.components]
                                    .clone_from_slice(finite);
                            }
                        }
                    }
                }
            }
        }
        self.successful_inputs.clear();
        for (compact, &lane) in self.valid_lanes.iter().enumerate() {
            if result.statuses[lane] == LaneStatus::Success {
                self.successful_lanes.push(lane);
                if let Some(quadrature) = &mut self.quadrature {
                    quadrature.select_row(compact, self.components);
                }
                self.successful_inputs.extend_from_slice(
                    &self.reduction_inputs
                        [compact * reduction_width..(compact + 1) * reduction_width],
                );
            }
        }
        let successful = self.successful_lanes.len();
        self.stats.reduction_rows = self
            .stats
            .reduction_rows
            .checked_add(successful)
            .ok_or_else(size_error)?;
        if successful == 0 {
            return Ok(result);
        }
        if let Some(quadrature) = &mut self.quadrature {
            let started = self.profiling.then(Instant::now);
            let evaluated = quadrature.evaluate(
                &self.successful_inputs,
                successful,
                input_width,
                reduction_width,
                self.components,
                &zero,
            );
            record_time(started, &mut self.profile.reduction);
            evaluated?;
        }
        let evaluation_width = checked_mul(self.reduction_outputs, self.components)?;
        self.output
            .resize(checked_mul(successful, evaluation_width)?, zero.clone());
        let started = self.profiling.then(Instant::now);
        if let Err(error) =
            self.reduction
                .evaluate_batch(&self.successful_inputs, &mut self.output, successful)
        {
            record_time(started, &mut self.profile.reduction);
            return Err(ReductionError::Evaluation(error.to_string()));
        }
        let primal_width = reduction_width / self.components;
        if let Some(monitor) = &mut self.primal_monitor {
            // Absolute-value envelopes are numerical conditioning estimates,
            // not analytic expressions to differentiate. Keep this evaluator
            // primal-only while retaining the separate all-Taylor guards.
            self.primal_inputs.clear();
            self.primal_inputs.extend(
                self.successful_inputs
                    .chunks_exact(self.components)
                    .map(|components| components[0].clone()),
            );
            self.primal_output
                .resize(checked_mul(successful, self.primal_monitor_outputs)?, zero);
            if let Err(error) =
                monitor.evaluate_batch(&self.primal_inputs, &mut self.primal_output, successful)
            {
                record_time(started, &mut self.profile.reduction);
                return Err(ReductionError::Evaluation(error.to_string()));
            }
        }
        if let Some(sensitivity) = &mut self.component_sensitivity {
            let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
            sensitivity.evaluate(&self.successful_inputs, successful, self.components, &zero)?;
        }
        let local_monitor_width = checked_mul(self.stability.expressions().len(), self.components)?;
        for (compact, &lane) in self.successful_lanes.iter().enumerate() {
            self.guard_context.begin_lane();
            let evaluated =
                &self.output[compact * evaluation_width..(compact + 1) * evaluation_width];
            let (output, monitor_outputs) = evaluated.split_at(output_width);
            let (monitors, fused_primal) = monitor_outputs.split_at(local_monitor_width);
            let input_row =
                &self.successful_inputs[compact * reduction_width..(compact + 1) * reduction_width];
            let primal_values = if self.primal_fused {
                Some(fused_primal)
            } else {
                self.primal_monitor.as_ref().map(|_| {
                    &self.primal_output[compact * self.primal_monitor_outputs
                        ..(compact + 1) * self.primal_monitor_outputs]
                })
            };
            if !output.iter().all(finite) {
                result.statuses[lane] = LaneStatus::Unstable(
                    "reduction has nonfinite coefficients at this kinematic point".into(),
                );
            } else if let Some(reason) = self.stability.inspect_output_precision_lane_context(
                output,
                input_row,
                self.components,
                &self.prototype,
                self.accuracy_bits,
                &mut self.guard_context,
            ) {
                result.statuses[lane] = LaneStatus::Unstable(reason);
            } else if let Some(reason) = self.stability.inspect_lane_with_accuracy_context(
                monitors,
                input_row,
                self.components,
                &self.prototype,
                self.accuracy_bits,
                &mut self.guard_context,
            ) {
                result.statuses[lane] = LaneStatus::Unstable(reason);
            } else if let Some(reason) = primal_values.and_then(|values| {
                self.stability.inspect_primal_lane_context(
                    values,
                    if self.primal_fused {
                        input_row
                    } else {
                        &self.primal_inputs[compact * primal_width..(compact + 1) * primal_width]
                    },
                    &self.prototype,
                    self.accuracy_bits,
                    &mut self.guard_context,
                )
            }) {
                result.statuses[lane] = LaneStatus::Unstable(reason);
            } else if let Some(reason) = self.component_sensitivity.as_ref().and_then(|sensitivity| {
                if let Some(reason)=sensitivity.invalid_lane(compact) {return Some(reason);}
                self.stability.inspect_component_amplitude_lane(
                    sensitivity.lane(compact, self.components), output, input_row, self.components,
                    &self.prototype, self.accuracy_bits,
                )
            }) {
                result.statuses[lane] = LaneStatus::Unstable(reason);
            } else if let Some(reason) = self.quadrature.as_ref().and_then(|quadrature| {
                self.stability.inspect_quadrature_lane_context(
                    &quadrature.plan,
                    quadrature.lane(compact, self.components),
                    output,
                    input_row,
                    self.components,
                    &self.prototype,
                    self.accuracy_bits,
                    primal_values,
                    &mut self.guard_context,
                )
            }) {
                result.statuses[lane] = LaneStatus::Unstable(reason);
            } else {
                result.values[lane * output_width..(lane + 1) * output_width]
                    .clone_from_slice(output);
                // Export only the buffers from this guarded successful route.
                // Disabled ordinary calls allocate or clone no sidecar data.
                if self.policy.retain_primal_envelopes {
                    if self.quadrature.is_some() && !self.policy.retain_quadrature_context {
                        result.primal_envelope_errors[lane] = Some(
                            "quadrature error transport is unsupported by the six-point parent".into(),
                        );
                    } else {
                        match self.stability.export_primal_operands(
                            primal_values.unwrap_or(&[]), output, self.components,
                        ) {
                            Ok(operands) => {
                                result.primal_envelopes[lane] = Some(PrimalEnvelope {
                                    basis: self.basis.clone(),
                                    metadata: self.metadata.clone(),
                                    components: self.components,
                                    working_precision: self.prototype.get_precision(),
                                    accuracy_bits: self.accuracy_bits,
                                    operands,
                                    output_primals: output.chunks_exact(self.components)
                                        .map(|slot| slot[0].clone()).collect(),
                                    component_amplitudes: self.component_sensitivity.as_ref()
                                        .map(|sensitivity|sensitivity.lane(compact,self.components).to_vec()),
                                    output_components: self.component_sensitivity.as_ref().map(|_|output.to_vec()),
                                    routes: vec![PrimalEnvelopeRoute::Ordinary],
                                });
                                if let Err(reason) = result.primal_envelope(lane) {
                                    result.primal_envelopes[lane] = None;
                                    result.primal_envelope_errors[lane] = Some(reason);
                                }
                            }
                            Err(reason) => result.primal_envelope_errors[lane] = Some(reason),
                        }
                    }
                }
                if self.policy.retain_quadrature_context {
                    let dense = if self.policy.quadrature_imports_pending {
                        Err("imported child quadrature context is pending the outer affine kernel".to_string())
                    } else if let Some(quadrature) = &self.quadrature {
                        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
                        quadrature.plan.dense_output_errors(
                            quadrature.lane(compact, self.components), self.components, &zero)
                            .map(|errors| (errors, QuadratureErrorOrigin::PropagatedEuclidean))
                            .map_err(|error| error.to_string())
                    } else {
                        // Base-child construction proves this complete recipe
                        // has no imported uncertain values, not merely no local E6.
                        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
                        Ok((vec![zero; output_width], QuadratureErrorOrigin::KnownNoQuadrature))
                    };
                    let retained = dense.and_then(|(errors, origin)| {
                        result.validated_primal_envelope(lane,
                            self.prototype.get_precision(), self.accuracy_bits)?;
                        self.last_quadrature_context.install_dense(&result, lane,
                            self.prototype.get_precision(), self.accuracy_bits, &errors, origin)
                    });
                    if let Err(reason) = retained {
                        self.last_quadrature_context.reject(lane, reason)
                            .map_err(ReductionError::Evaluation)?;
                    }
                }
            }
        }
        record_time(started, &mut self.profile.reduction);
        if let Some(fallback) = &mut self.fallback {
            let lanes: Vec<_> = self
                .valid_lanes
                .iter()
                .filter_map(|&lane| {
                    (matches!(result.statuses[lane], LaneStatus::Unstable(_))
                        && !result.analytic_used[lane])
                        .then_some(lane)
                })
                .collect();
            if !lanes.is_empty() {
                let mut values = Vec::with_capacity(checked_mul(lanes.len(), input_width)?);
                for &lane in &lanes {
                    values.extend_from_slice(
                        &batch.values[lane * input_width..(lane + 1) * input_width],
                    );
                }
                let retried = fallback.reduce_batch(&KinematicsBatch {
                    rows: lanes.len(),
                    components: self.components,
                    values,
                });
                if self.policy != WorkspacePolicy::ORDINARY {
                    add_certificate_work(&mut self.planar_certificate_stats, fallback.last_planar_certificate_execution())?;
                }
                if self.policy != WorkspacePolicy::ORDINARY && retried.is_err() {
                    // Preserve actual child work even when no result can scatter.
                    // Ordinary error accounting remains on its existing path.
                    self.stats.fallback_rows = self.stats.fallback_rows
                        .checked_add(lanes.len()).ok_or_else(size_error)?;
                    self.stats.add_policy_child_work(fallback.last_execution())?;
                    if let Some(profile) = fallback.last_profile() {
                        self.profile.add_stages(profile);
                    }
                    for (compact, &lane) in lanes.iter().enumerate() {
                        if let Some(notes) = fallback.last_diagnostics().get(compact) {
                            self.diagnostics[lane].extend_from_slice(notes);
                        }
                    }
                }
                let retried = retried?;
                if self.policy == WorkspacePolicy::ORDINARY {
                    self.stats.fallback_rows = lanes.len();
                } else {
                    self.stats.fallback_rows = self.stats.fallback_rows
                        .checked_add(lanes.len()).ok_or_else(size_error)?;
                }
                for (compact, &lane) in lanes.iter().enumerate() {
                    result.fallback_used[lane] = true;
                    if self.policy != WorkspacePolicy::ORDINARY {
                        result.analytic_used[lane] |= retried.analytic_used[compact];
                    }
                    self.diagnostics[lane].extend_from_slice(&fallback.diagnostics[compact]);
                    match retried.lane(compact) {
                        Ok(output) => {
                            result.values[lane * output_width..(lane + 1) * output_width]
                                .clone_from_slice(output);
                            result.statuses[lane] = LaneStatus::Success;
                            if self.policy.retain_primal_envelopes {
                                result.inherit_primal_envelope(lane, &retried, compact, PrimalEnvelopeRoute::Hybrid);
                            }
                            if self.policy.retain_quadrature_context {
                                // Values and their error estimates must come from
                                // the same replacement route, including descendants.
                                let _ = self.last_quadrature_context.inherit_lane(&result, lane,
                                    &retried, &fallback.last_quadrature_context, compact,
                                    self.prototype.get_precision(), self.accuracy_bits);
                            }
                            self.stats.fallback_successes += 1;
                        }
                        Err(status) => {
                            if self.policy.retain_quadrature_context {
                                self.last_quadrature_context.reject(lane,
                                    "replacement IBP route did not return a successful coupled context".into())
                                    .map_err(ReductionError::Evaluation)?;
                            }
                            let primary = &result.statuses[lane];
                            result.statuses[lane] = LaneStatus::Unstable(format!(
                                "PV reduction failed its stability check ({primary:?}); IBP fallback returned {status:?}",
                            ));
                        }
                    }
                }
                let work = fallback.last_execution();
                if self.policy != WorkspacePolicy::ORDINARY {
                    // Count descendant stages separately from this outer retry.
                    self.stats.fallback_rows = self.stats.fallback_rows
                        .checked_add(work.fallback_rows).ok_or_else(size_error)?;
                    self.stats.fallback_successes = self.stats.fallback_successes
                        .checked_add(work.fallback_successes).ok_or_else(size_error)?;
                    self.stats.analytic_rows = self.stats.analytic_rows
                        .checked_add(work.analytic_rows).ok_or_else(size_error)?;
                }
                if let Some(profile) = fallback.last_profile() {
                    self.profile.add_stages(profile);
                }
                self.stats.argument_rows = self
                    .stats
                    .argument_rows
                    .checked_add(work.argument_rows)
                    .ok_or_else(size_error)?;
                self.stats.master_family_batches = self
                    .stats
                    .master_family_batches
                    .checked_add(work.master_family_batches)
                    .ok_or_else(size_error)?;
                self.stats.master_rows = self
                    .stats
                    .master_rows
                    .checked_add(work.master_rows)
                    .ok_or_else(size_error)?;
                self.stats.pentagon_box_rows = self
                    .stats
                    .pentagon_box_rows
                    .checked_add(work.pentagon_box_rows)
                    .ok_or_else(size_error)?;
                self.stats.reduction_rows = self
                    .stats
                    .reduction_rows
                    .checked_add(work.reduction_rows)
                    .ok_or_else(size_error)?;
            }
        }
        Ok(result)
    }

    fn retain_lanes(&mut self, selected: &[usize], input_width: usize, argument_width: usize) {
        // No input or Taylor coefficient changes during routing.
        for (destination, &source) in selected.iter().enumerate() {
            self.valid_lanes[destination] = self.valid_lanes[source];
            if destination != source {
                for component in 0..input_width {
                    self.input[destination * input_width + component] =
                        self.input[source * input_width + component].clone();
                }
                for component in 0..argument_width {
                    self.argument_values[destination * argument_width + component] =
                        self.argument_values[source * argument_width + component].clone();
                }
            }
        }
        self.valid_lanes.truncate(selected.len());
        self.input.truncate(selected.len() * input_width);
        self.argument_values
            .truncate(selected.len() * argument_width);
    }

    /// Flatten caller-defined static Dual values without treating derivative
    /// components as extra sample lanes. Pass `|dual| &dual.values`.
    pub fn reduce_dual_batch<D: DualNumberStructure>(
        &mut self,
        input: &[D],
        rows: usize,
        components: impl for<'a> Fn(&'a D) -> &'a [Complex<T>],
    ) -> Result<BatchResult<T>, ReductionError> {
        self.last_quadrature_context.clear();
        // Typed-input validation can return before reduce_batch resets the
        // observable state. Attribute every early error to this requested call.
        self.stats = ExecutionStats { input_rows: rows, ..ExecutionStats::default() };
        self.profile = ExecutionProfile::default();
        self.diagnostics.clear();
        self.planar_certificate_stats = CertificateExecutionStats::default();
        let expected = checked_mul(rows, self.parameters)?;
        if input.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "expected {expected} Dual input values, received {}",
                input.len(),
            )));
        }
        let shape = self.shape.as_ref().ok_or_else(|| {
            ReductionError::InvalidInput(
                "use Workspace::new_dual before supplying Dual values".into(),
            )
        })?;
        let expected_shape: Vec<_> = shape.iter().map(Vec::as_slice).collect();
        let mut values = Vec::with_capacity(checked_mul(input.len(), self.components)?);
        for value in input {
            if value.get_shape() != expected_shape || components(value).len() != self.components {
                return Err(ReductionError::InvalidInput(
                    "input Dual shape differs from workspace shape".into(),
                ));
            }
            values.extend_from_slice(components(value));
        }
        self.reduce_batch(&KinematicsBatch {
            rows,
            components: self.components,
            values,
        })
    }

    pub fn reduce_hyperdual_batch(
        &mut self,
        input: &[HyperDual<Complex<T>>],
        rows: usize,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.reduce_dual_batch(input, rows, |value| &value.values)
    }

    /// Evaluate any Symbolica static or dynamic Dual carrier directly.
    pub fn reduce_typed_dual_batch<D: DualNumber<Complex<T>>>(
        &mut self,
        input: &[D],
        rows: usize,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.reduce_dual_batch(input, rows, DualNumber::values)
    }
}

/// Evaluate using a caller-owned reusable workspace.
pub fn reduce_batch<T: NativeFloat>(
    workspace: &mut Workspace<T>,
    batch: &KinematicsBatch<T>,
) -> Result<BatchResult<T>, ReductionError>
where
    Complex<T>: EvaluationDomain,
{
    workspace.reduce_batch(batch)
}

#[cfg(test)]
fn dispatch_profile<T: NativeFloat>(
    workspace: &mut Workspace<T>, lanes: &[usize], batch: &KinematicsBatch<T>,
    result: &mut BatchResult<T>, stats: &mut ExecutionStats,
    profile: &mut ExecutionProfile, diagnostics: &mut [Vec<String>],
    certificate_stats: &mut CertificateExecutionStats,
) -> Result<(), ReductionError>
where Complex<T>: EvaluationDomain {
    let mut unavailable = QuadratureErrorContext::missing(result.rows());
    dispatch_profile_with_context(workspace, lanes, batch, result, stats, profile,
        diagnostics, certificate_stats, &mut unavailable)
}

fn dispatch_profile_with_context<T: NativeFloat>(
    workspace: &mut Workspace<T>,
    lanes: &[usize],
    batch: &KinematicsBatch<T>,
    result: &mut BatchResult<T>,
    stats: &mut ExecutionStats,
    profile: &mut ExecutionProfile,
    diagnostics: &mut [Vec<String>],
    certificate_stats: &mut CertificateExecutionStats,
    quadrature_context: &mut QuadratureErrorContext<T>,
) -> Result<(), ReductionError>
where
    Complex<T>: EvaluationDomain,
{
    let input_width = checked_mul(workspace.parameters, workspace.components)?;
    let output_width = checked_mul(checked_mul(result.basis.len(), 4)?, result.components)?;
    let mut values = Vec::with_capacity(checked_mul(lanes.len(), input_width)?);
    for &lane in lanes {
        values.extend_from_slice(&batch.values[lane * input_width..(lane + 1) * input_width]);
    }
    let evaluated = workspace.reduce_batch(&KinematicsBatch {
        rows: lanes.len(),
        components: batch.components,
        values,
    });
    if workspace.policy != WorkspacePolicy::ORDINARY {
        add_certificate_work(certificate_stats, workspace.last_planar_certificate_execution())?;
    }
    if workspace.policy != WorkspacePolicy::ORDINARY && evaluated.is_err() {
        stats.analytic_rows = stats.analytic_rows.checked_add(lanes.len()).ok_or_else(size_error)?;
        stats.add_policy_child_work(workspace.last_execution())?;
        if let Some(child_profile) = workspace.last_profile() {
            profile.add_stages(child_profile);
        }
        for (compact, &lane) in lanes.iter().enumerate() {
            if let Some(notes) = workspace.last_diagnostics().get(compact) {
                diagnostics[lane].extend_from_slice(notes);
            }
        }
    }
    let evaluated = evaluated?;
    for (compact, &lane) in lanes.iter().enumerate() {
        diagnostics[lane].extend_from_slice(&workspace.diagnostics[compact]);
        result.analytic_used[lane] = true;
        result.fallback_used[lane] = evaluated.fallback_used[compact];
        result.statuses[lane] = evaluated.statuses[compact].clone();
        if let Ok(output) = evaluated.lane(compact) {
            result.values[lane * output_width..(lane + 1) * output_width].clone_from_slice(output);
            result.inherit_primal_envelope(lane, &evaluated, compact, PrimalEnvelopeRoute::Analytic);
            if workspace.policy.retain_quadrature_context {
                let _ = quadrature_context.inherit_lane(result, lane, &evaluated,
                    &workspace.last_quadrature_context, compact,
                    workspace.prototype.get_precision(), workspace.accuracy_bits);
            }
        } else {
            if !result.primal_envelopes.is_empty() {
                result.primal_envelopes[lane] = None;
                result.primal_envelope_errors[lane] = None;
            }
            if workspace.policy.retain_quadrature_context {
                quadrature_context.reject(lane,
                    "analytic profile did not return a successful coupled quadrature context".into())
                    .map_err(ReductionError::Evaluation)?;
            }
        }
    }
    stats.analytic_rows = stats
        .analytic_rows
        .checked_add(lanes.len())
        .ok_or_else(size_error)?;
    let work = workspace.last_execution();
    if workspace.policy != WorkspacePolicy::ORDINARY {
        stats.analytic_rows = stats.analytic_rows
            .checked_add(work.analytic_rows).ok_or_else(size_error)?;
    }
    if let Some(child_profile) = workspace.last_profile() {
        profile.add_stages(child_profile);
    }
    stats.argument_rows = stats
        .argument_rows
        .checked_add(work.argument_rows)
        .ok_or_else(size_error)?;
    stats.master_family_batches = stats
        .master_family_batches
        .checked_add(work.master_family_batches)
        .ok_or_else(size_error)?;
    stats.master_rows = stats
        .master_rows
        .checked_add(work.master_rows)
        .ok_or_else(size_error)?;
    stats.pentagon_box_rows = stats
        .pentagon_box_rows
        .checked_add(work.pentagon_box_rows)
        .ok_or_else(size_error)?;
    stats.reduction_rows = stats
        .reduction_rows
        .checked_add(work.reduction_rows)
        .ok_or_else(size_error)?;
    stats.fallback_rows = stats
        .fallback_rows
        .checked_add(work.fallback_rows)
        .ok_or_else(size_error)?;
    stats.fallback_successes = stats
        .fallback_successes
        .checked_add(work.fallback_successes)
        .ok_or_else(size_error)?;
    Ok(())
}

pub(crate) fn compile_exact<D: DualNumberStructure>(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: &[(Atom, Atom)],
    dual: Option<&D>,
) -> Result<ExpressionEvaluator<Complex<Rational>>, ReductionError> {
    let mut exact: ExpressionEvaluator<Complex<Rational>> =
        Atom::evaluator_multiple(expressions, parameters)
            .add_aliases(aliases.iter().cloned())
            .map_err(|error| ReductionError::Preparation(error.to_string()))?
            .direct_translation(true)
            .horner_iterations(0)
            .cores(1)
            .build()
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
    if let Some(layout) = dual {
        exact = exact
            .vectorize(&Dualizer::new(BorrowedDual(layout), Vec::new()))
            .map_err(ReductionError::Preparation)?;
    } else {
        exact.optimize_stack();
    }
    Ok(exact)
}

pub(crate) fn map_exact<T: NativeFloat>(
    exact: &ExpressionEvaluator<Complex<Rational>>,
    prototype: &T,
) -> BatchExpressionEvaluator<Complex<T>>
where
    Complex<T>: EvaluationDomain,
{
    exact
        .clone()
        .map_coeff_with_prec(
            &|value| {
                Complex::new(
                    prototype.from_rational(&value.re),
                    prototype.from_rational(&value.im),
                )
            },
            prototype.get_precision(),
        )
        .into_batch_evaluator()
}

fn preparation_timer() -> Option<Instant> {
    std::env::var_os("IREGI_PREPARATION_PROFILE")
        .is_some()
        .then(Instant::now)
}

fn report_preparation(stage: &str, started: Option<Instant>) {
    if let Some(started) = started {
        eprintln!(
            "iregi preparation {stage}: {:.6}s",
            started.elapsed().as_secs_f64()
        );
    }
}

// Borrow existing Symbolica metadata without regenerating multiplication tables
// or requiring caller-defined static Dual types to implement Clone.
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

/// Shared public-input validation. Call only after whole-buffer shape checks.
/// The check order and error strings match the ordinary numerical interface.
pub(crate) fn validate_input_row<T: NativeFloat>(
    row: &[Complex<T>],
    layout: &InputLayout,
    components: usize,
) -> Option<LaneStatus> {
    if let Some((slot, _)) = row.iter().enumerate().find(|(_, value)| !finite(value)) {
        return Some(LaneStatus::InvalidInput(format!(
            "nonfinite input at argument {}, Taylor component {}",
            slot / components,
            slot % components,
        )));
    }
    // mu, rather than mu², is the public interface. Squaring must not
    // hide a negative or complex scale before master validation.
    let scale = &row[layout.scale * components];
    if !scale.im.is_zero() || scale.re <= scale.re.zero() {
        return Some(LaneStatus::InvalidInput(
            "mu must be positive and real".into(),
        ));
    }
    if layout
        .shifts
        .clone()
        .chain(layout.scalar_products.clone().into_iter().flatten())
        .any(|argument| !row[argument * components].im.is_zero())
    {
        return Some(LaneStatus::InvalidInput(
            "momentum-shift and scalar-product primal components must be real".into(),
        ));
    }
    if layout.masses.clone().any(|argument| {
        let mass = &row[argument * components];
        mass.im > mass.im.zero()
    }) {
        return Some(LaneStatus::InvalidInput(
            "squared masses must have nonpositive imaginary parts".into(),
        ));
    }
    None
}

fn finite<T: NativeFloat>(value: &Complex<T>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn complex_zero<T: NativeFloat>(value: &Complex<T>) -> bool {
    value.re.is_zero() && value.im.is_zero()
}

fn maximum_jet_order<'a>(
    shape: impl IntoIterator<Item = &'a [usize]>,
) -> Result<u16, ReductionError> {
    shape.into_iter().try_fold(0, |maximum, component| {
        component
            .iter()
            .try_fold(0usize, |order, &degree| order.checked_add(degree))
            .and_then(|order| u16::try_from(order).ok())
            .map(|order| maximum.max(order))
            .ok_or_else(|| ReductionError::Preparation("Dual derivative order exceeds u16".into()))
    })
}

fn size_error() -> ReductionError {
    ReductionError::InvalidInput("batch dimensions overflow usize".into())
}

fn checked_mul(left: usize, right: usize) -> Result<usize, ReductionError> {
    left.checked_mul(right).ok_or_else(size_error)
}

fn pentagon_relative_tolerance<T: NativeFloat>(prototype: &T, accuracy_bits: u32) -> T {
    // The reduction accuracy target is 2^(19-bits). Reserve two guard bits
    // for quadrature, including when high working precision rescues a lower
    // requested accuracy. Construct the power directly in the numeric domain.
    let two = prototype.from_i64(2);
    if accuracy_bits >= 17 {
        two.pow(u64::from(accuracy_bits - 17)).inv()
    } else {
        two.pow(u64::from(17 - accuracy_bits))
    }
}

fn add_certificate_work(
    total: &mut CertificateExecutionStats,
    work: CertificateExecutionStats,
) -> Result<(), ReductionError> {
    // Commit the aggregate atomically if every checked addition succeeds.
    *total = CertificateExecutionStats {
        input_rows: total.input_rows.checked_add(work.input_rows).ok_or_else(size_error)?,
        evaluated_rows: total.evaluated_rows.checked_add(work.evaluated_rows).ok_or_else(size_error)?,
        passes: total.passes.checked_add(work.passes).ok_or_else(size_error)?,
        reduction_batches: total.reduction_batches.checked_add(work.reduction_batches).ok_or_else(size_error)?,
    };
    Ok(())
}

impl ExecutionStats {
    /// Used only by opted-in policy routes; ordinary accounting stays unchanged.
    fn add_policy_child_work(&mut self, work: Self) -> Result<(), ReductionError> {
        macro_rules! add {
            ($field:ident) => {
                self.$field = self.$field.checked_add(work.$field).ok_or_else(size_error)?;
            };
        }
        add!(argument_rows);
        add!(master_family_batches);
        add!(master_rows);
        add!(pentagon_box_rows);
        add!(reduction_rows);
        add!(fallback_rows);
        add!(fallback_successes);
        add!(analytic_rows);
        Ok(())
    }
}

impl ExecutionProfile {
    fn add_stages(&mut self, child: Self) {
        self.arguments += child.arguments;
        self.scalar_masters += child.scalar_masters;
        self.reduction += child.reduction;
    }
}

fn record_time(started: Option<Instant>, duration: &mut Duration) {
    if let Some(started) = started {
        *duration += started.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, Convention, ReductionSettings};
    use iregi_masters::{DoubleFloat, Float};
    use symbolica::domains::float::{FloatLike, Real, SingleFloat};

    fn c(value: f64) -> Complex<f64> {
        Complex::new(value, 0.)
    }

    #[test]
    fn pentagon_quadrature_budget_tracks_requested_accuracy_with_two_guard_bits() {
        let binary_target = pentagon_relative_tolerance(&0., 53);
        let tighter_target = pentagon_relative_tolerance(&0., 64);
        assert_eq!(binary_target, 2.0_f64.powi(-36));
        assert_eq!(tighter_target, 2.0_f64.powi(-47));
        assert!(tighter_target < binary_target);
        assert_eq!(pentagon_relative_tolerance(&0., 17), 1.);
        assert_eq!(pentagon_relative_tolerance(&0., 16), 2.);
        // Higher working precision preserves the requested binary64 budget;
        // its additional bits improve master evaluation, not the user's target.
        let prototype = Float::new(128);
        assert_eq!(
            pentagon_relative_tolerance(&prototype, 53),
            prototype.from_i64(2).pow(36).inv()
        );
        assert_eq!(
            pentagon_relative_tolerance(&prototype, 128),
            prototype.from_i64(2).pow(111).inv()
        );
    }

    #[test]
    fn quadrature_error_packing_preserves_selected_lanes_and_taylor_order() {
        // A graph-only fixture isolates executor packing from the independently
        // validated E6 integration. No scalar backend is instantiated here.
        let mut plan = plan(1, 0);
        plan.leaves[0].kind = MasterLeafKind::SixDimensionalPentagon;
        plan.leaves[0].active = vec![0, 1, 2, 3, 4];
        plan.aliases.clear();
        plan.moments[0] = [
            Atom::new(),
            Atom::new(),
            Atom::new(),
            &plan.parameters[0] * &plan.leaves[0].slots[3],
        ];
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![2], vec![1]]);
        let prepared = PreparedQuadrature::prepare(&plan, &stability, Some(&shape))
            .unwrap()
            .unwrap();
        let mut quadrature = QuadratureWorkspace::instantiate(&prepared, &0., 1).unwrap();
        quadrature.reset(3, 3, &c(0.)).unwrap();
        quadrature
            .store_errors(0, 0, 3, &[2., 7., 5.], &0.)
            .unwrap();
        quadrature
            .store_errors(0, 2, 3, &[1., 4., 3.], &0.)
            .unwrap();
        // Select the surviving lanes in reversed order; the omitted lane must
        // not shift another sample's component errors into its contraction.
        quadrature.select_row(2, 3);
        quadrature.select_row(0, 3);
        let input_width = plan.parameters.len() * 3;
        let reduction_width = input_width + 4 * 3;
        let mut inputs = vec![c(0.); 2 * reduction_width];
        inputs[..3].copy_from_slice(&[c(-2.), c(5.), c(0.)]);
        inputs[reduction_width..reduction_width + 3].copy_from_slice(&[c(3.), c(0.), c(4.)]);
        quadrature
            .evaluate(&inputs, 2, input_width, reduction_width, 3, &c(0.))
            .unwrap();
        assert_eq!(quadrature.lane(0, 3), &[c(4.), c(26.), c(12.)]);
        assert_eq!(quadrature.lane(1, 3), &[c(12.), c(82.), c(46.)]);
    }

    fn plan(propagators: usize, rank: u16) -> ReductionPlan {
        let mut settings = ReductionSettings::new(propagators, rank);
        settings.convention = Convention::OneLoop;
        settings.algorithm = Algorithm::PassarinoVeltman;
        ReductionPlan::new(settings).unwrap()
    }

    fn close(actual: Complex<f64>, expected: Complex<f64>) {
        assert!(
            (actual.re - expected.re).abs() <= 1e-11 * (1. + expected.re.abs()),
            "{actual:?} != {expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= 1e-11 * (1. + expected.im.abs()),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn rounded_zero_distance_is_rejected_before_massless_master_and_profile_dispatch() {
        let plan = plan(2, 0);
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let large = 134_217_728.; // 2^27, exactly represented.
        let previous = large - 2_f64.powi(-26);
        let row = |q: [f64; 4]| {
            [0., 0., 0., 0.]
                .into_iter()
                .chain(q)
                .chain([0., 0., 1.])
                .map(c)
                .collect::<Vec<_>>()
        };
        let false_zero = row([large, previous, 2., 0.]);
        let genuine_zero = row([1., 1., 0., 0.]);
        let regular = row([0., 1., 0., 0.]);
        let batch = KinematicsBatch {
            rows: 3,
            components: 1,
            values: [false_zero.clone(), genuine_zero, regular.clone()].concat(),
        };
        let result = workspace.reduce_batch(&batch).unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::Unstable(reason)
            if reason.contains("rounded-zero distance") && reason.contains("exact represented")),
            "{:?}",
            result.statuses
        );
        assert!(!result.analytic_used[0]);
        assert!(!result.fallback_used[0]);
        close(result.lane(1).unwrap()[0], c(1.));
        close(result.lane(1).unwrap()[1], c(-1.));
        assert_eq!(result.statuses[2], LaneStatus::Success);
        assert_eq!(workspace.last_execution().master_rows, 1);
        let isolated = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: regular,
            })
            .unwrap();
        assert_eq!(isolated.lane(0).unwrap(), result.lane(2).unwrap());

        // Increasing arithmetic precision retains the exact same represented
        // inputs and resolves q²=-2^-52, without selecting the scaleless branch.
        let mut precise = Workspace::new(&plan, Float::new(256)).unwrap();
        precise.set_accuracy_precision(53).unwrap();
        let resolved = precise
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: false_zero
                    .iter()
                    .map(|value| {
                        Complex::new(
                            Float::with_val(256, value.re),
                            Float::with_val(256, value.im),
                        )
                    })
                    .collect(),
            })
            .unwrap();
        assert_eq!(resolved.statuses[0], LaneStatus::Success);
        assert!(resolved.lane(0).unwrap()[1].re.is_zero());
        assert!(resolved.lane(0).unwrap()[1].im.is_zero());
        assert!(precise.distance_zeros.as_ref().unwrap().lanes.is_empty());
    }

    #[test]
    fn derived_distance_guards_preserve_nonzero_jets_and_isolate_unstable_rows() {
        let plan = plan(2, 0);
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let large = 134_217_728.;
        let previous = large - 2_f64.powi(-26);
        let row = |q: [f64; 4], derivative: [f64; 4]| {
            let primal = [0., 0., 0., 0.].into_iter().chain(q).chain([4., 4., 1.]);
            let tangent = [0., 0., 0., 0.]
                .into_iter()
                .chain(derivative)
                .chain([0., 0., 0.]);
            primal
                .zip(tangent)
                .flat_map(|(value, jet)| [c(value), c(jet)])
                .collect::<Vec<_>>()
        };
        let batch = KinematicsBatch {
            rows: 2,
            components: 2,
            values: [
                // Exactly lightlike primal, nonzero first derivative of s.
                row([1., 1., 0., 0.], [1., 0., 0., 0.]),
                // Primal s≈-1, but ds=-2^-51 falsely rounds to zero.
                row([large, previous, 2., 1.], [large, previous, 2., 0.]),
            ]
            .concat(),
        };
        let result = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert!(result.analytic_used[0]);
        assert!(
            matches!(&result.statuses[1], LaneStatus::Unstable(reason)
            if reason.contains("derived distance") || reason.contains("rounded-zero distance")),
            "{:?}",
            result.statuses
        );
        assert!(!result.analytic_used[1]);
        assert!(!result.fallback_used[1]);
        let first_derivative = &result.lane(0).unwrap()[7];
        assert!(!complex_zero(first_derivative));
    }

    #[test]
    fn nonzero_derived_distance_cancellation_is_rejected_and_resolved_at_higher_precision() {
        let plan = plan(2, 0);
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let row = |q: [f64; 4]| {
            [0.; 4]
                .into_iter()
                .chain(q)
                .chain([0., 0., 1.])
                .map(c)
                .collect::<Vec<_>>()
        };
        // Exact represented q²=-5; binary64 arithmetic can instead produce -4.
        // Both are nonzero, so an exact-zero check alone cannot protect B0.
        let ill_conditioned = row([134_217_728., 134_217_727., 16_384., 2.]);
        let regular = row([0., 1., 2., 0.]);
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values: [ill_conditioned.clone(), regular.clone()].concat(),
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::Unstable(reason)
                if reason.contains("derived distance") && reason.contains("relative conditioning")),
            "{:?}",
            result.statuses
        );
        assert_eq!(result.statuses[1], LaneStatus::Success);
        assert!(!result.analytic_used[0]);
        assert!(!result.fallback_used[0]);
        assert_eq!(workspace.last_execution().master_rows, 1);

        let mut precise = Workspace::new(&plan, Float::new(128)).unwrap();
        precise.set_accuracy_precision(53).unwrap();
        let resolved = precise
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values: [ill_conditioned, regular]
                    .concat()
                    .iter()
                    .map(|value| {
                        Complex::new(
                            Float::with_val(128, value.re),
                            Float::with_val(128, value.im),
                        )
                    })
                    .collect(),
            })
            .unwrap();
        assert_eq!(resolved.statuses, vec![LaneStatus::Success; 2]);
        // Cancellation consumes about 54 working bits, which Symbolica
        // records in the result precision. Require twenty decimal digits of
        // mixed agreement in all four Laurent slots, well beyond the requested
        // binary64 target. The erroneous s=-4 finite part differs by log(5/4).
        let tolerance = Float::with_val(128, 10).pow(20).inv();
        let actual = resolved.lane(0).unwrap();
        let expected = resolved.lane(1).unwrap();
        assert_eq!(actual.len(), 4);
        for (slot, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = NativeFloat::hypot(
                &(actual.re.clone() - &expected.re),
                &(actual.im.clone() - &expected.im),
            );
            let limit = tolerance.clone()
                * (Float::with_val(128, 1) + NativeFloat::hypot(&expected.re, &expected.im));
            assert!(
                delta <= limit,
                "Laurent slot {slot}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn derived_distance_guard_respects_input_precision_within_one_batch() {
        let plan = plan(2, 0);
        let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let row = |bits, q: [f64; 4]| {
            [0.; 4]
                .into_iter()
                .chain(q)
                .chain([0., 0., 1.])
                .map(|value| Complex::new(Float::with_val(bits, value), Float::new(bits)))
                .collect::<Vec<_>>()
        };
        // These integer coordinates are exactly represented even at 80 bits,
        // but their squares require 107 bits to retain the unit term. The
        // exact invariant is -5; 80-bit products can instead yield -4.
        let q = [
            9_007_199_254_740_992.,
            9_007_199_254_740_991.,
            134_217_728.,
            2.,
        ];
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 1,
                values: [row(80, q), row(256, q), row(256, [0., 1., 2., 0.])].concat(),
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::Unstable(reason)
                if reason.contains("derived distance")),
            "{:?}",
            result.statuses
        );
        assert_eq!(
            &result.statuses[1..],
            &[LaneStatus::Success, LaneStatus::Success]
        );
        assert!(!result.analytic_used[0]);
        assert!(!result.fallback_used[0]);
        assert_eq!(workspace.last_execution().master_rows, 2);
        let tolerance = Float::with_val(256, 10).pow(20).inv();
        let actual = result.lane(1).unwrap();
        let expected = result.lane(2).unwrap();
        assert_eq!(actual.len(), 4);
        for (slot, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = NativeFloat::hypot(
                &(actual.re.clone() - &expected.re),
                &(actual.im.clone() - &expected.im),
            );
            let limit = tolerance.clone()
                * (Float::with_val(256, 1) + NativeFloat::hypot(&expected.re, &expected.im));
            assert!(
                delta <= limit,
                "Laurent slot {slot}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn derived_distance_envelope_checks_internal_second_taylor_cancellation() {
        let plan = plan(2, 0);
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut bad = vec![c(0.); 11 * 3];
        bad[4 * 3] = c(1.);
        bad[4 * 3 + 1] = c(2_f64.powi(27) - 1.);
        bad[4 * 3 + 2] = c(-2_f64.powi(53) + 2_f64.powi(27) + 2.);
        bad[8 * 3] = c(4.);
        bad[9 * 3] = c(4.);
        bad[10 * 3] = c(1.);
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 3,
                values: bad,
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::Unstable(reason)
                if reason.contains("derived distance") && reason.contains("Taylor component 2")),
            "{:?}",
            result.statuses
        );
        assert_eq!(workspace.last_execution().master_rows, 0);
        assert!(!result.analytic_used[0]);
        assert!(!result.fallback_used[0]);
    }

    #[test]
    fn supplied_scalar_products_use_exact_distance_zero_proofs() {
        let mut settings = ReductionSettings::new(2, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.invariants = crate::InvariantInput::SuppliedScalarProducts;
        settings.convention = Convention::OneLoop;
        let plan = ReductionPlan::new(settings).unwrap();
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let large = 18_014_398_509_481_984.; // 2^54.
        let row = |g11| {
            [0.; 8]
                .into_iter()
                .chain([large, large / 2., g11])
                .chain([4., 4., 1.])
                .map(c)
                .collect::<Vec<_>>()
        };
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values: [row(1.), row(0.)].concat(),
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::Unstable(reason)
            if reason.contains("rounded-zero distance")),
            "{:?}",
            result.statuses
        );
        assert_eq!(result.statuses[1], LaneStatus::Success);
    }

    #[test]
    fn scalar_tadpole_empty_mixed_lanes_and_reused_workspace() {
        let plan = plan(1, 0);
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let empty = KinematicsBatch {
            rows: 0,
            components: 1,
            values: Vec::new(),
        };
        assert_eq!(workspace.reduce_batch(&empty).unwrap().rows(), 0);

        let valid = [c(0.), c(0.), c(0.), c(0.), c(2.), c(1.)];
        let mut invalid_input = valid;
        invalid_input[0] = c(f64::NAN);
        let mut invalid_mass = valid;
        invalid_mass[4] = Complex::new(2., 1.);
        let mut invalid_scale = valid;
        invalid_scale[5] = c(-1.);
        let values = [valid, invalid_input, valid, invalid_mass, invalid_scale].concat();
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 5,
                components: 1,
                values,
            })
            .unwrap();
        assert!(matches!(result.statuses[1], LaneStatus::InvalidInput(_)));
        assert!(matches!(result.statuses[3], LaneStatus::InvalidInput(_)));
        assert!(matches!(result.statuses[4], LaneStatus::InvalidInput(_)));
        assert!(result.lane(1).is_err());
        assert!(result.lane(3).is_err());
        for lane in [0, 2] {
            let actual = result.lane(lane).unwrap();
            close(actual[0], c(2.));
            close(actual[1], c(0.));
            close(actual[2], c(0.));
            close(actual[3], c(2. * (1. - 2_f64.ln())));
        }
        assert_eq!(
            workspace.last_execution(),
            ExecutionStats {
                input_rows: 5,
                argument_rows: 2,
                master_family_batches: 1,
                master_rows: 2,
                pentagon_box_rows: 0,
                reduction_rows: 2,
                fallback_rows: 0,
                fallback_successes: 0,
                analytic_rows: 0,
            }
        );
        let single = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: valid.to_vec(),
            })
            .unwrap();
        assert_eq!(single.lane(0).unwrap(), result.lane(0).unwrap());
    }

    #[test]
    fn equal_mass_rank_one_bubble_respects_routing_and_batch_order() {
        let plan = plan(2, 1);
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let mut values = Vec::new();
        for spatial in [1., 2., 3., 4., 5.] {
            values.extend([0., 0., 0., 0., 0., spatial, 0., 0., 2., 2., 3.].map(c));
        }
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 5,
                components: 1,
                values: values.clone(),
            })
            .unwrap();
        for (lane, spatial) in [1., 2., 3., 4., 5.].into_iter().enumerate() {
            let actual = result.lane(lane).unwrap();
            for slot in 0..4 {
                close(actual[4 + slot], c(0.));
                close(actual[8 + slot], actual[slot] * c(-spatial / 2.));
                close(actual[12 + slot], c(0.));
                close(actual[16 + slot], c(0.));
            }
        }
        assert_eq!(workspace.last_execution().master_family_batches, 2);
        assert_eq!(workspace.last_execution().reduction_rows, 5);
        for lane in (0..5).rev() {
            let single = workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: values[lane * 11..(lane + 1) * 11].to_vec(),
                })
                .unwrap();
            for (&actual, &expected) in single
                .lane(0)
                .unwrap()
                .iter()
                .zip(result.lane(lane).unwrap())
            {
                close(actual, expected);
            }
        }
    }

    #[test]
    fn scale_squared_and_mass_taylor_components_propagate_through_all_stages() {
        let plan = plan(1, 0);
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut values = vec![c(0.); 18];
        values[12] = c(2.);
        values[13] = c(1.);
        values[15] = c(3.);
        values[16] = c(1.);
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 3,
                values: values.repeat(3),
            })
            .unwrap();
        // A0 = m² * (1 + log(mu²/m²)). Both m² and mu vary with
        // the same Taylor variable; this checks the nonlinear mu² argument.
        let logarithm = (9_f64 / 2.).ln();
        for lane in 0..3 {
            let actual = result.lane(lane).unwrap();
            assert_eq!(&actual[..3], &[c(2.), c(1.), c(0.)]);
            close(actual[9], c(2. * (1. + logarithm)));
            close(actual[10], c(logarithm + 4. / 3.));
            close(actual[11], c(-0.25 + 2. / 3. - 2. / 9.));
        }
        assert_eq!(workspace.last_execution().master_rows, 3);
        assert_eq!(workspace.last_execution().reduction_rows, 3);
    }

    #[test]
    fn reduction_keeps_double_double_and_arbitrary_precision_inputs() {
        let plan = plan(1, 0);
        let one = DoubleFloat::from(1.);
        let zero = DoubleFloat::from(0.);
        let mass = one + one.from_usize(2).pow(80).inv();
        let mut workspace = Workspace::new(&plan, zero).unwrap();
        let values = [zero, zero, zero, zero, mass, one]
            .map(|value| Complex::new(value, zero))
            .to_vec();
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values,
            })
            .unwrap();
        assert_eq!(result.lane(0).unwrap()[0].re, mass);
        assert_ne!(result.lane(0).unwrap()[0].re, one);

        for bits in [128, 256] {
            let zero = Float::new(bits);
            let one = Float::with_val(bits, 1);
            let mass = one.clone() + Float::with_val(bits, 2).pow(100).inv();
            let mut workspace = Workspace::new(&plan, zero.clone()).unwrap();
            let values = [
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                mass.clone(),
                one.clone(),
            ]
            .map(|value| Complex::new(value, zero.clone()))
            .to_vec();
            let result = workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values,
                })
                .unwrap();
            assert_eq!(result.lane(0).unwrap()[0].re, mass);
            assert_ne!(result.lane(0).unwrap()[0].re, one);
            assert_eq!(workspace.binary_precision(), bits);
        }
    }

    symbolica::create_hyperdual_from_components!(StaticDual, [[0], [1], [2]]);

    #[test]
    fn typed_static_and_dynamic_dual_carriers_agree() {
        let plan = plan(1, 0);
        let mut mass = StaticDual::new_variable(0, c(2.));
        let mut constant = mass.clone();
        constant.values = [c(0.); 3];
        let mut scale = constant.clone();
        scale.values[0] = c(1.);
        mass.values[2] = c(0.);
        let mut workspace = Workspace::new_dual(&plan, 0., mass.clone()).unwrap();
        let input = [
            constant.clone(),
            constant.clone(),
            constant.clone(),
            constant,
            mass,
            scale,
        ];
        let static_result = workspace.reduce_typed_dual_batch(&input, 1).unwrap();
        let shape = vec![vec![0], vec![1], vec![2]];
        let dynamic: Vec<_> = input
            .iter()
            .map(|value| HyperDual::from_values(shape.clone(), value.values.to_vec()))
            .collect();
        let dynamic_result = workspace.reduce_hyperdual_batch(&dynamic, 1).unwrap();
        assert_eq!(
            static_result.lane(0).unwrap(),
            dynamic_result.lane(0).unwrap()
        );
        let actual = static_result.lane(0).unwrap();
        close(actual[9], c(2. * (1. - 2_f64.ln())));
        close(actual[10], c(-2_f64.ln()));
        close(actual[11], c(-0.25));
    }

    #[test]
    fn lightlike_unequal_mass_lanes_use_analytic_profiles() {
        let mut settings = ReductionSettings::new(2, 1);
        settings.convention = Convention::OneLoop;
        let hybrid = ReductionPlan::new(settings.clone()).unwrap();
        settings.algorithm = Algorithm::Ibp;
        let ibp = ReductionPlan::new(settings).unwrap();
        let mut workspace = Workspace::new(&hybrid, 0.).unwrap();
        // The second momentum is nonzero and lightlike: its Gram determinant
        // vanishes, while unequal masses keep the Cayley matrix nonsingular.
        let values = [
            [0., 0., 0., 0., 0., 1., 0., 0., 2., 3., 1.],
            [0., 0., 0., 0., 1., 1., 0., 0., 2., 3., 1.],
        ]
        .into_iter()
        .flatten()
        .map(c)
        .collect();
        let batch = KinematicsBatch {
            rows: 2,
            components: 1,
            values,
        };
        let result = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(workspace.last_execution().fallback_rows, 0);
        assert_eq!(workspace.last_execution().analytic_rows, 1);
        assert_eq!(result.analytic_used, [false, true]);
        let expected = Workspace::new(&ibp, 0.)
            .unwrap()
            .reduce_batch(&batch)
            .unwrap();
        for lane in 0..2 {
            for (&actual, &expected) in result
                .lane(lane)
                .unwrap()
                .iter()
                .zip(expected.lane(lane).unwrap())
            {
                close(actual, expected);
            }
        }
    }

    #[test]
    fn distinct_master_leaves_preserve_mixed_dual_components_and_lane_scatter() {
        let plan = plan(2, 1);
        let shape = HyperDual::<f64>::new(vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![2, 0],
            vec![1, 1],
        ]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut values = Vec::new();
        for lane in 0..3 {
            let mut row = vec![c(0.); 55];
            row[5 * 5] = c(1. + f64::from(lane) / 4.);
            row[5 * 5 + 2] = c(1.);
            row[8 * 5] = c(2. + f64::from(lane) / 2.);
            row[8 * 5 + 1] = c(1.);
            row[9 * 5] = c(3. + f64::from(lane));
            row[10 * 5] = c(1. + f64::from(lane) / 3.);
            if lane == 1 {
                row[0] = c(f64::NAN);
            }
            values.extend(row);
        }
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 5,
                values: values.clone(),
            })
            .unwrap();
        assert!(matches!(result.statuses[1], LaneStatus::InvalidInput(_)));
        assert_eq!(workspace.last_execution().argument_rows, 2);
        assert_eq!(workspace.last_execution().master_family_batches, 2);
        for lane in [2, 0] {
            let single = workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 5,
                    values: values[lane * 55..(lane + 1) * 55].to_vec(),
                })
                .unwrap();
            for (&actual, &expected) in single
                .lane(0)
                .unwrap()
                .iter()
                .zip(result.lane(lane).unwrap())
            {
                close(actual, expected);
            }
            // B1's UV coefficient is -p_x/2; its mass derivative vanishes.
            let uv_x = &result.lane(lane).unwrap()[2 * 4 * 5..2 * 4 * 5 + 5];
            close(uv_x[0], c(-(1. + lane as f64 / 4.) / 2.));
            close(uv_x[1], c(0.));
            close(uv_x[2], c(-0.5));
            close(uv_x[3], c(0.));
            close(uv_x[4], c(0.));
        }
    }

    #[test]
    fn native_normalization_constants_use_the_requested_precision() {
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let plan = ReductionPlan::new(settings).unwrap();
        // Populate the structural cache from binary64 first. Subsequent
        // higher-precision constants must still be mapped from exact IR.
        let _binary = Workspace::new(&plan, 0.).unwrap();
        let bits = 256;
        let zero = Float::new(bits);
        let one = Float::with_val(bits, 1);
        let mass = Float::with_val(bits, 2);
        let values = [
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            mass.clone(),
            one.clone(),
        ]
        .map(|value| Complex::new(value, zero.clone()))
        .to_vec();
        let result = Workspace::new(&plan, zero.clone())
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values,
            })
            .unwrap();
        let actual = result.lane(0).unwrap();
        let pi = zero.pi();
        let normalization = pi.pow(2);
        let expected_uv = normalization.clone() * &mass;
        let expected_finite = normalization * &mass * (one - mass.log() - zero.euler() - pi.log());
        let tolerance = Float::with_val(bits, 2).pow(220).inv();
        assert!((actual[0].im.clone() - expected_uv).norm() < tolerance);
        assert!((actual[3].im.clone() - expected_finite).norm() < tolerance);
        assert!(actual[0].re.is_zero());
        assert!(actual[3].re.is_zero());
    }

    #[test]
    fn equal_mass_lightlike_profile_retains_transverse_and_mixed_taylor_components() {
        let plan = plan(2, 1);
        let shape = HyperDual::<f64>::new(vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![2, 0],
            vec![1, 1],
            vec![0, 2],
        ]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut values = Vec::new();
        for mass in [2., 3., 4.] {
            let mut row = vec![c(0.); 66];
            row[4 * 6] = c(1.);
            row[4 * 6 + 2] = c(1.);
            row[5 * 6] = c(1.);
            row[8 * 6] = c(mass);
            row[8 * 6 + 1] = c(1.);
            row[9 * 6] = c(mass);
            row[10 * 6] = c(1.);
            values.extend(row);
        }
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 6,
                values,
            })
            .unwrap();
        assert_eq!(result.analytic_used, [true; 3]);
        assert_eq!(workspace.last_execution().analytic_rows, 3);
        // Independent derivatives of -integral_0^1 log((1-x)m0+x*m1-x(1-x)s) dx.
        // Here m0=m+t, m1=m and s=(1+u)^2-1=2u+u².
        for (lane, mass) in [2_f64, 3., 4.].into_iter().enumerate() {
            let actual = result.lane(lane).unwrap();
            for (component, expected) in [
                -mass.ln(),
                -1. / (2. * mass),
                1. / (3. * mass),
                1. / (6. * mass * mass),
                -1. / (6. * mass * mass),
                1. / (6. * mass) + 1. / (15. * mass * mass),
            ]
            .into_iter()
            .enumerate()
            {
                close(actual[18 + component], c(expected));
            }
            close(actual[0], c(1.));
            close(actual[4 * 6], c(-0.5));
            close(actual[4 * 6 + 2], c(-0.5));
            close(actual[2 * 4 * 6], c(-0.5));
            close(actual[(2 * 4 + 3) * 6], c(0.5 * mass.ln()));
        }
    }

    #[test]
    fn supplied_scalar_products_match_shift_derived_batches_and_dual_jets() {
        let from_shifts = plan(2, 1);
        let mut settings = from_shifts.settings().clone();
        settings.invariants = crate::InvariantInput::SuppliedScalarProducts;
        let supplied = ReductionPlan::new(settings).unwrap();
        let mut shifts_values = Vec::new();
        let mut supplied_values = Vec::new();
        for point in [
            [0., 0., 0., 0., 0., 2., 0., 0., 2., 3., 1.],
            [0., 0., 0., 0., 1., 1., 0., 0., 2., 2., 1.],
            [0., 0., 0., 0., 0., 3., 0., 0., 4., 3., 2.],
        ] {
            shifts_values.extend(point.map(c));
            supplied_values.extend(point[..8].iter().copied().map(c));
            supplied_values.extend([0., 0., point[4] * point[4] - point[5] * point[5]].map(c));
            supplied_values.extend(point[8..].iter().copied().map(c));
        }
        let ordinary = Workspace::new(&from_shifts, 0.)
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 1,
                values: shifts_values,
            })
            .unwrap();
        let explicit = Workspace::new(&supplied, 0.)
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 1,
                values: supplied_values,
            })
            .unwrap();
        assert_eq!(explicit.analytic_used, [false, true, false]);
        for lane in 0..3 {
            for (&actual, &expected) in explicit
                .lane(lane)
                .unwrap()
                .iter()
                .zip(ordinary.lane(lane).unwrap())
            {
                close(actual, expected);
            }
        }

        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1]]);
        let mut shifts = vec![c(0.); 22];
        shifts[8] = c(1.);
        shifts[9] = c(1.);
        shifts[10] = c(1.);
        shifts[16] = c(2.);
        shifts[18] = c(2.);
        shifts[20] = c(1.);
        let mut products = shifts[..16].to_vec();
        products.extend([c(0.), c(0.), c(0.), c(0.), c(0.), c(2.)]);
        products.extend_from_slice(&shifts[16..]);
        let ordinary = Workspace::new_dual(&from_shifts, 0., shape.clone())
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 2,
                values: shifts,
            })
            .unwrap();
        let explicit = Workspace::new_dual(&supplied, 0., shape)
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 2,
                values: products,
            })
            .unwrap();
        for (&actual, &expected) in explicit
            .lane(0)
            .unwrap()
            .iter()
            .zip(ordinary.lane(0).unwrap())
        {
            close(actual, expected);
        }
        close(explicit.lane(0).unwrap()[7], c(1. / 6.));
    }

    #[test]
    fn finite_cancellation_retries_and_optional_profiles_include_child_stages() {
        let mut settings = ReductionSettings::new(2, 1);
        settings.convention = Convention::OneLoop;
        let plan = ReductionPlan::new(settings).unwrap();
        let values = [1., 1e-5, 0.]
            .into_iter()
            .flat_map(|spatial| [0., 0., 0., 0., 0., spatial, 0., 0., 4., 9., 1.].map(c))
            .collect();
        let batch = KinematicsBatch {
            rows: 3,
            components: 1,
            values,
        };
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        assert!(workspace.last_profile().is_none());
        let ordinary = workspace.reduce_batch(&batch).unwrap();
        assert!(workspace.last_profile().is_none());
        workspace.set_profiling(true);
        let profiled = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(profiled.fallback_used, [false, true, false]);
        assert_eq!(profiled.analytic_used, [false, false, true]);
        assert_eq!(workspace.last_execution().fallback_rows, 1);
        assert!(
            matches!(profiled.statuses[1], LaneStatus::Unstable(_)),
            "both conditioned recurrences must report the unresolved lane: {:?}",
            profiled.statuses
        );
        assert!(profiled.lane(1).is_err());
        assert_eq!(profiled.statuses, ordinary.statuses);
        for lane in [0, 2] {
            assert_eq!(profiled.lane(lane).unwrap(), ordinary.lane(lane).unwrap());
        }
        let profile = workspace.last_profile().unwrap();
        assert!(profile.arguments + profile.scalar_masters + profile.reduction <= profile.total);
        workspace.set_profiling(false);
        assert!(workspace.last_profile().is_none());
    }

    #[test]
    fn inactive_master_branches_are_batched_without_false_dual_failures() {
        use symbolica::atom::Symbol;

        let mut settings = ReductionSettings::new(2, 1);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.convention = Convention::OneLoop;
        let mut plan = ReductionPlan::new(settings).unwrap();
        let tadpole = plan
            .leaves
            .iter()
            .find(|leaf| leaf.active == [1])
            .unwrap()
            .slots
            .clone();
        let bubble = plan
            .leaves
            .iter()
            .find(|leaf| leaf.active.len() == 2)
            .unwrap()
            .slots
            .clone();
        let control = plan.parameters[4].clone();
        plan.basis.truncate(1);
        plan.moments = vec![std::array::from_fn(|slot| {
            Symbol::IF.call((&control, &bubble[slot], &tadpole[slot]))
        })];
        // This synthetic expression tests dispatch independently of the
        // automatic whole-bubble profile. Its analytic degree is already 1.
        plan.special_kinematics = true;
        plan.analytic_jet_order = 1;
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut values = Vec::new();
        for (time, space, first_mass, second_mass) in
            [(0., 0., 0., 9.), (1., 2., 4., 9.), (1., 1., 0., 0.)]
        {
            for (argument, value) in [
                0.,
                0.,
                0.,
                0.,
                time,
                space,
                0.,
                0.,
                first_mass,
                second_mass,
                1.,
            ]
            .into_iter()
            .enumerate()
            {
                values.extend([
                    c(value),
                    c(if argument == 8 || argument == 9 {
                        1.
                    } else {
                        0.
                    }),
                ]);
            }
        }
        let batch = KinematicsBatch {
            rows: 3,
            components: 2,
            values,
        };
        let evaluated = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(evaluated.statuses[0], LaneStatus::Success);
        assert_eq!(evaluated.statuses[1], LaneStatus::Success);
        assert!(matches!(
            evaluated.statuses[2],
            LaneStatus::MasterFailure(_)
        ));
        assert_eq!(workspace.last_execution().master_rows, 3);
        assert_eq!(workspace.last_execution().master_family_batches, 2);
        close(evaluated.lane(0).unwrap()[0], c(9.));
        close(evaluated.lane(0).unwrap()[1], c(1.));
        close(evaluated.lane(0).unwrap()[6], c(9. * (1. - 9f64.ln())));
        close(evaluated.lane(0).unwrap()[7], c(-9f64.ln()));
        close(evaluated.lane(1).unwrap()[0], c(1.));
        close(evaluated.lane(1).unwrap()[1], c(0.));

        let repeated = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 2,
                values: batch.values[..22].to_vec(),
            })
            .unwrap();
        assert_eq!(repeated.lane(0).unwrap(), evaluated.lane(0).unwrap());
        assert_eq!(workspace.last_execution().master_rows, 1);
        assert_eq!(workspace.last_execution().master_family_batches, 1);
    }

    #[test]
    fn exact_zero_profiles_preserve_shifted_scaleless_poles_and_batch_neighbors() {
        let mut settings = ReductionSettings::new(2, 1);
        settings.convention = Convention::OneLoop;
        let plan = ReductionPlan::new(settings).unwrap();
        let batch = KinematicsBatch {
            rows: 3,
            components: 1,
            values: [
                [0., 0., 0., 0., 0., 1., 0., 0., 4., 9., 1.],
                [2., 1., 0., 0., 2., 1., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
            ]
            .into_iter()
            .flatten()
            .map(c)
            .collect(),
        };
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let result = workspace.reduce_batch(&batch).unwrap();
        assert!(
            result
                .statuses
                .iter()
                .all(|status| *status == LaneStatus::Success),
            "{:?}",
            result.statuses
        );
        assert_eq!(result.analytic_used, [false, true, true]);
        let scalar = result.lane(1).unwrap();
        close(scalar[0], c(1.));
        close(scalar[1], c(-1.));
        close(scalar[2], c(0.));
        close(scalar[3], c(0.));
        let time = result
            .basis
            .iter()
            .position(|basis| basis.0 == [1, 0, 0, 0])
            .unwrap();
        close(scalar[time * 4], c(-2.));
        close(scalar[time * 4 + 1], c(2.));
        close(result.lane(2).unwrap()[3], c(2.));
        let isolated = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: batch.values[11..22].to_vec(),
            })
            .unwrap();
        assert_eq!(isolated.lane(0).unwrap(), scalar);
        assert_eq!(workspace.last_execution().master_rows, 0);
        assert_eq!(workspace.last_execution().analytic_rows, 1);
    }

    #[test]
    fn zero_constraints_check_higher_dual_components_without_freezing_routing() {
        let mut settings = ReductionSettings::new(1, 1);
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        settings.powers = vec![2];
        let plan = ReductionPlan::new(settings).unwrap();
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut tangent = vec![c(0.); 18];
        tangent[0] = c(2.);
        tangent[1] = c(1.);
        tangent[15] = c(1.);
        let mut transverse = tangent.clone();
        transverse[14] = c(1.); // m² has only a second-order transverse seed.
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 3,
                values: tangent.into_iter().chain(transverse).collect(),
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert!(matches!(result.statuses[1], LaneStatus::MasterFailure(_)));
        assert_eq!(result.analytic_used, [true, false]);
        let values = result.lane(0).unwrap();
        close(values[0], c(1.));
        close(values[3], c(-1.));
        let time = result
            .basis
            .iter()
            .position(|basis| basis.0 == [1, 0, 0, 0])
            .unwrap();
        close(values[time * 12], c(-2.));
        close(values[time * 12 + 1], c(-1.));
        close(values[time * 12 + 3], c(2.));
        close(values[time * 12 + 4], c(1.));
    }

    #[test]
    fn fused_ordinary_monitors_match_separate_primal_graph_and_lazy_lanes() {
        let mut plan = plan(1, 0);
        let uv = plan.leaves[0].slots[0].clone();
        let intermediate = Atom::var(symbolica::symbol!("iregi::test_fused_amplitude"));
        plan.aliases.push((
            intermediate.clone(),
            Atom::num(1_000_000) * uv - Atom::num(3_996_000),
        ));
        plan.moments[0] = [
            symbolica::atom::Symbol::IF.call((
                &plan.parameters[0],
                intermediate - Atom::num(3_996),
                Atom::num(7),
            )),
            Atom::new(),
            Atom::new(),
            Atom::new(),
        ];
        plan.special_kinematics = true;
        let mut fused = Workspace::new(&plan, 0.).unwrap();
        let mut separate = Workspace::new_dual(
            &plan,
            0.,
            HyperDual::<Complex<Rational>>::new(vec![vec![0]]),
        )
        .unwrap();
        assert!(fused.primal_fused);
        assert!(fused.primal_monitor.is_none());
        assert!(!separate.primal_fused);
        assert!(separate.primal_monitor.is_some());
        let mut values = Vec::new();
        for (selector, mass) in [
            (1., c(4.)),
            (1., c(5.)),
            (0., c(4.)),
            (1., Complex::new(4., 1.)),
        ] {
            values.extend([c(selector), c(0.), c(0.), c(0.), mass, c(1.)]);
        }
        let batch = KinematicsBatch {
            rows: 4,
            components: 1,
            values,
        };
        let actual = fused.reduce_batch(&batch).unwrap();
        let reference = separate.reduce_batch(&batch).unwrap();
        assert_eq!(actual.statuses, reference.statuses);
        assert!(matches!(&actual.statuses[0], LaneStatus::Unstable(reason)
            if reason.contains("compounded master cancellation")));
        assert!(matches!(actual.statuses[3], LaneStatus::InvalidInput(_)));
        for lane in [1, 2] {
            assert_eq!(actual.lane(lane).unwrap(), reference.lane(lane).unwrap());
        }
        close(actual.lane(1).unwrap()[0], c(1_000_004.));
        close(actual.lane(2).unwrap()[0], c(7.));
        assert_eq!(fused.last_execution().master_rows, 2);
        assert_eq!(fused.last_execution().reduction_rows, 3);
        assert!(fused.primal_inputs.is_empty());
        assert!(fused.primal_output.is_empty());

        // Reordering and shrinking a retained workspace must preserve branch
        // masks, failure isolation, and the fused output offsets.
        let reordered = KinematicsBatch {
            rows: 3,
            components: 1,
            values: [2, 0, 1]
                .into_iter()
                .flat_map(|lane| batch.values[lane * 6..(lane + 1) * 6].iter().cloned())
                .collect(),
        };
        let reused = fused.reduce_batch(&reordered).unwrap();
        for (new, old) in [2, 0, 1].into_iter().enumerate() {
            assert_eq!(reused.statuses[new], actual.statuses[old]);
            if let Ok(values) = actual.lane(old) {
                assert_eq!(reused.lane(new).unwrap(), values);
            }
        }
        assert!(fused.primal_inputs.is_empty());
        assert!(fused.primal_output.is_empty());
    }

    #[test]
    fn compounded_primal_monitor_preserves_dual_stride_and_lane_isolation() {
        let mut plan = plan(1, 0);
        let uv = plan.leaves[0].slots[0].clone();
        let intermediate = Atom::var(symbolica::symbol!("iregi::test_compounded_amplitude"));
        plan.aliases.push((
            intermediate.clone(),
            Atom::num(1_000_000) * uv - Atom::num(3_996_000),
        ));
        plan.moments[0] = [
            intermediate - Atom::num(3_996),
            Atom::new(),
            Atom::new(),
            Atom::new(),
        ];
        plan.analytic_jet_order = 1;
        plan.special_kinematics = true;
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape).unwrap();
        let mut local_only = workspace.clone();
        local_only.primal_monitor = None;
        let batch = KinematicsBatch {
            rows: 2,
            components: 2,
            values: [4., 5.]
                .into_iter()
                .flat_map(|mass| {
                    [0., 0., 0., 0., mass, 1.].into_iter().enumerate().flat_map(
                        |(argument, value)| [c(value), c(if argument == 4 { 1. } else { 0. })],
                    )
                })
                .collect(),
        };
        let local_result = local_only.reduce_batch(&batch).unwrap();
        assert_eq!(
            local_result.statuses,
            [LaneStatus::Success, LaneStatus::Success]
        );
        let result = workspace.reduce_batch(&batch).unwrap();
        match &result.statuses[0] {
            LaneStatus::Unstable(reason) => assert!(
                reason.contains("compounded master cancellation"),
                "{reason}"
            ),
            status => panic!("compounded cancellation was not detected: {status:?}"),
        }
        assert!(result.lane(0).is_err());
        assert_eq!(result.statuses[1], LaneStatus::Success);
        assert_eq!(result.lane(1).unwrap(), local_result.lane(1).unwrap());
        close(result.lane(1).unwrap()[0], c(1_000_004.));
        close(result.lane(1).unwrap()[1], c(1_000_000.));
        assert_eq!(workspace.last_execution().reduction_rows, 2);
    }

    #[test]
    fn exact_preparation_is_shared_across_precisions_and_keyed_by_dual_shape() {
        fn thread_safe<T: Send + Sync>() {}
        thread_safe::<PreparedExecution>();
        thread_safe::<ReductionPlan>();

        let plan = plan(1, 0);
        let binary = Workspace::new(&plan, 0.).unwrap();
        let precise = Workspace::new(&plan, Float::new(256)).unwrap();
        assert!(matches!(
            &binary.masters[0].evaluator,
            ScalarMasterEvaluator::OneLoop(evaluator)
                if evaluator.backend() == iregi_masters::default_master_backend()
        ));
        assert!(matches!(
            &precise.masters[0].evaluator,
            ScalarMasterEvaluator::OneLoop(evaluator)
                if evaluator.backend() == iregi_masters::default_master_backend()
        ));
        assert!(Arc::ptr_eq(&binary.stability, &precise.stability));
        assert_eq!(plan.execution_cache.lock().unwrap().len(), 1);

        // An explicit primal-only Dual carrier is a different semantic
        // layout, despite having the same number of flattened components.
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0]]);
        let dual = Workspace::new_dual(&plan, 0., shape).unwrap();
        assert!(matches!(
            &dual.masters[0].evaluator,
            ScalarMasterEvaluator::OneLoop(evaluator)
                if evaluator.backend() == MasterBackend::SymbolicaBatch
        ));
        assert!(!Arc::ptr_eq(&binary.stability, &dual.stability));
        assert_eq!(plan.execution_cache.lock().unwrap().len(), 2);
    }

    #[test]
    fn legacy_boundaries_are_primal_only_and_isolated_before_master_evaluation() {
        let mut settings = ReductionSettings::new(2, 0);
        settings.convention = Convention::OneLoop;
        settings.input_policy = crate::InputPolicy::LegacyOnShell {
            threshold: "1/4".into(),
        };
        let plan = ReductionPlan::new(settings).unwrap();
        let rows: Vec<_> = [0.5, 1., 2.]
            .into_iter()
            .map(|time| [0., 0., 0., 0., time, 0., 0., 0., 4., 9., 3.])
            .collect();
        let original: Vec<_> = rows.iter().flatten().copied().map(c).collect();
        let batch = KinematicsBatch {
            rows: 3,
            components: 1,
            values: original.clone(),
        };
        let mut ordinary = Workspace::new(&plan, 0.).unwrap();
        let result = ordinary.reduce_batch(&batch).unwrap();
        assert!(
            result
                .statuses
                .iter()
                .all(|status| *status == LaneStatus::Success)
        );
        assert!(
            ordinary.last_diagnostics()[1]
                .iter()
                .any(|note| note.contains("strict-comparison boundary"))
        );
        assert_eq!(batch.values, original);

        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let mut workspace = Workspace::new_dual(&plan, 0., shape.clone()).unwrap();
        let mut values = Vec::new();
        for row in &rows {
            for (argument, &value) in row.iter().enumerate() {
                // Both a routing and a mass direction are present. Snapping
                // to m0² must retain its jet while discarding the routing jet.
                values.extend([
                    c(value),
                    c(if argument == 4 || argument == 8 {
                        1.
                    } else {
                        0.
                    }),
                ]);
            }
        }
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 3,
                components: 2,
                values,
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert_eq!(result.statuses[2], LaneStatus::Success);
        assert!(matches!(&result.statuses[1], LaneStatus::Unstable(reason)
            if reason.contains("nondifferentiable") && reason.contains("relative zero threshold")));
        assert!(!result.fallback_used[1]);

        let mut master = MasterEvaluator::new_dual(ScalarIntegral::B0, 0., shape).unwrap();
        let expected_input = [
            c(0.),
            c(0.),
            c(4.),
            c(1.),
            c(9.),
            c(0.),
            c(9.),
            c(0.),
            c(4.),
            c(1.),
            c(4.),
            c(1.),
            c(9.),
            c(0.),
            c(9.),
            c(0.),
        ];
        let expected = master.evaluate_batch(&expected_input, 2).unwrap();
        for (expected_lane, result_lane) in [0, 2].into_iter().enumerate() {
            let scalar = expected.lane(expected_lane).unwrap();
            for (slot, coefficients) in [scalar.uv, scalar.ir, scalar.ir2, scalar.finite]
                .into_iter()
                .enumerate()
            {
                for (component, &value) in coefficients.iter().enumerate() {
                    close(
                        result.lane(result_lane).unwrap()[slot * 2 + component],
                        value,
                    );
                }
            }
        }
    }

    #[test]
    fn legacy_threshold_representability_respects_precision_and_unused_queries() {
        let mut settings = ReductionSettings::new(2, 0);
        settings.convention = Convention::OneLoop;
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.input_policy = crate::InputPolicy::LegacyOnShell {
            threshold: "1e-1000".into(),
        };
        let plan = ReductionPlan::new(settings.clone()).unwrap();
        let values = [0., 0., 0., 0., 1., 0., 0., 0., 4., 9., 3.].map(c).to_vec();
        let mut binary = Workspace::new(&plan, 0.).unwrap();
        let result = binary
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: values.clone(),
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::InvalidInput(reason)
            if reason.contains("threshold is not representable"))
        );
        assert_eq!(binary.last_execution().master_rows, 0);

        let prototype = Float::new(256);
        let mut precise = Workspace::new(&plan, prototype.clone()).unwrap();
        let values = [0, 0, 0, 0, 1, 0, 0, 0, 4, 9, 3]
            .into_iter()
            .map(|value| {
                Complex::new(
                    prototype.from_rational(&Rational::from((value, 1))),
                    prototype.zero(),
                )
            })
            .collect();
        let result = precise
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values,
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);

        // A scalar tadpole has no scalar-product queries; preparing the
        // reference-mass list alone must not activate a policy diagnostic.
        settings.propagators = 1;
        settings.powers = vec![1];
        let plan = ReductionPlan::new(settings).unwrap();
        let mut unused = Workspace::new(&plan, 0.).unwrap();
        let result = unused
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: [0., 0., 0., 0., 4., 3.].map(c).to_vec(),
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert!(unused.snapping_monitor.is_none());
    }

    #[test]
    fn failed_optional_zero_profile_rejoins_ordinary_batch_in_original_order() {
        let normal_plan = plan(1, 0);
        let mut workspace = Workspace::new(&normal_plan, 0.).unwrap();
        let mut without_profile = workspace.clone();
        without_profile.zero_profile_offset = None;
        without_profile.zero_profile_plan = None;
        let mut failing_plan = plan(1, 0);
        failing_plan.special_kinematics = true;
        failing_plan.moments[0] = [
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            Atom::num(1) / &failing_plan.parameters[failing_plan.layout.masses.start],
        ];
        let failing = Workspace::new(&failing_plan, 0.).unwrap();
        workspace.zero_profiles.insert(
            ZeroInvariantProfile {
                masses: vec![true],
                distances: Vec::new(),
            },
            Ok(Box::new(failing)),
        );
        let batch = KinematicsBatch {
            rows: 3,
            components: 1,
            values: [4., 0., 9.]
                .into_iter()
                .flat_map(|mass| [0., 0., 0., 0., mass, 3.].map(c))
                .collect(),
        };
        let expected = without_profile.reduce_batch(&batch).unwrap();
        let actual = workspace.reduce_batch(&batch).unwrap();
        for lane in 0..3 {
            assert_eq!(actual.statuses[lane], LaneStatus::Success);
            assert_eq!(actual.lane(lane).unwrap(), expected.lane(lane).unwrap());
            assert!(!actual.analytic_used[lane]);
        }
        assert!(
            workspace.last_diagnostics()[1]
                .iter()
                .any(|note| note.contains("exact-zero profile evaluation failed")
                    && note.contains("ordinary reduction retained"))
        );
        assert!(workspace.last_diagnostics()[0].is_empty());
        assert!(workspace.last_diagnostics()[2].is_empty());
        // The failed attempt stays visible in measured work, while all
        // ordinary lanes still run together in a single instruction batch.
        assert_eq!(workspace.last_execution().analytic_rows, 1);
        assert_eq!(workspace.last_execution().reduction_rows, 4);
        assert_eq!(workspace.last_execution().master_rows, 3);
    }

    #[test]
    fn legacy_pv_does_not_replace_its_gram_system_with_an_ibp_zero_profile() {
        let mut settings = ReductionSettings::new(3, 1);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.convention = Convention::OneLoop;
        settings.input_policy = crate::InputPolicy::legacy_onshell();
        let plan = ReductionPlan::new(settings.clone()).unwrap();
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let mut wrong_profile_plan = ReductionPlan::new(settings).unwrap();
        wrong_profile_plan.special_kinematics = true;
        for moment in &mut wrong_profile_plan.moments {
            *moment = [Atom::Zero, Atom::Zero, Atom::Zero, Atom::num(99)];
        }
        let wrong_profile = Workspace::new(&wrong_profile_plan, 0.).unwrap();
        workspace.zero_profiles.insert(
            ZeroInvariantProfile {
                masses: vec![false, true, false],
                distances: Vec::new(),
            },
            Ok(Box::new(wrong_profile)),
        );
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: [
                    0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 4., 0., 9., 3.,
                ]
                .map(c)
                .to_vec(),
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert!(!result.analytic_used[0]);
        assert_eq!(workspace.last_execution().analytic_rows, 0);
        assert_ne!(result.lane(0).unwrap()[3], c(99.));
    }

    #[test]
    fn legacy_zero_image_rejects_only_lanes_using_an_unsupported_diagonal() {
        let mut settings = ReductionSettings::new(2, 0);
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        settings.input_policy = crate::InputPolicy::LegacyOnShell {
            threshold: "0".into(),
        };
        let plan = ReductionPlan::new(settings).unwrap();
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let result = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 2,
                components: 1,
                values: [-4., 4.]
                    .into_iter()
                    .flat_map(|mass| [0., 0., 0., 0., 0., 1., 0., 0., mass, 9., 3.].map(c))
                    .collect(),
            })
            .unwrap();
        assert!(
            matches!(&result.statuses[0], LaneStatus::InvalidInput(reason)
            if reason.contains("nonphysical Cayley-diagonal domain is unsupported"))
        );
        assert_eq!(result.statuses[1], LaneStatus::Success);
        assert_eq!(workspace.last_execution().master_rows, 1);

        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        settings.input_policy = crate::InputPolicy::LegacyOnShell {
            threshold: "0".into(),
        };
        let plan = ReductionPlan::new(settings).unwrap();
        let mut scalar = Workspace::new(&plan, 0.).unwrap();
        let result = scalar
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: [0., 0., 0., 0., -4., 3.].map(c).to_vec(),
            })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert!(scalar.snapping_monitor.is_none());
    }
    include!("execution/guard_context_tests.rs");
    include!("execution/policy_tests.rs");
}
