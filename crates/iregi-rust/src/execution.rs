//! Instruction-major execution of prepared reduction and scalar-master graphs.
//!
//! Sample lanes and Taylor components are independent dimensions. Symbolica
//! compiles all argument transformations and reduction arithmetic; this module
//! only validates and packs their numerical buffers.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use iregi_masters::{MasterEvaluator, NativeFloat, ScalarIntegral};
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
    basis::Monomial,
    plan::{ReductionPlan, ZeroInvariantProfile},
    stability::StabilityPlan,
    types::{
        BatchResult, InputLayout, KinematicsBatch, LaneStatus, ReductionError, ResultMetadata,
    },
};

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
    pub reduction_rows: usize,
    /// Samples retried by the hybrid algorithm after a failed PV stability check.
    pub fallback_rows: usize,
    pub fallback_successes: usize,
    /// Samples routed through an exact analytic kinematic-limit profile.
    pub analytic_rows: usize,
}

/// Optional elapsed times for one outer reduction call, including its children.
///
/// `arguments` measures the invariant/argument evaluator; `scalar_masters`
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
    evaluator: MasterEvaluator<T>,
    input: Vec<Complex<T>>,
    requests: Vec<(usize, usize)>,
}

/// Owned evaluators and scratch storage for one immutable reduction plan.
///
/// Independent workspaces may evaluate the same plan concurrently. Constants
/// are mapped from exact rationals at `prototype` precision, without a binary64
/// intermediate. Construct a new workspace when increasing working precision.
#[derive(Clone)]
pub struct Workspace<T: NativeFloat> {
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
    zero_profile_plan: Option<Box<ReductionPlan>>,
    zero_profiles: HashMap<ZeroInvariantProfile, Result<Box<Workspace<T>>, String>>,
    arguments: BatchExpressionEvaluator<Complex<T>>,
    reduction: BatchExpressionEvaluator<Complex<T>>,
    reduction_outputs: usize,
    stability: StabilityPlan,
    masters: Vec<MasterGroup<T>>,
    valid_lanes: Vec<usize>,
    successful_lanes: Vec<usize>,
    input: Vec<Complex<T>>,
    argument_values: Vec<Complex<T>>,
    reduction_inputs: Vec<Complex<T>>,
    successful_inputs: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
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
        Self::prepare::<HyperDual<Complex<Rational>>>(plan, prototype, None)
    }

    /// Compile a static or dynamic Symbolica Dual layout. Inputs and outputs
    /// retain the supplied component order and Taylor coefficient convention.
    pub fn new_dual<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(&dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        let order = maximum_jet_order(dual.get_shape())?;
        if order == plan.analytic_jet_order {
            return Self::prepare(plan, prototype, Some(&dual));
        }
        let specialized = plan.with_analytic_jet_order(order)?;
        Self::prepare(&specialized, prototype, Some(&dual))
    }

    fn prepare<D: DualNumberStructure>(
        plan: &ReductionPlan,
        prototype: T,
        dual: Option<&D>,
    ) -> Result<Self, ReductionError> {
        if prototype.get_precision() == 0 {
            return Err(ReductionError::Preparation(
                "working precision must be positive".into(),
            ));
        }
        if let Some(layout) = dual {
            iregi_masters::validate_dual_structure(layout)
                .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        }
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
        let mut grouped_leaves: Vec<(ScalarIntegral, Vec<usize>)> = Vec::new();
        for (index, (leaf, arguments)) in plan.leaves.iter().zip(&plan.master_arguments).enumerate()
        {
            let family = match leaf.active.len() {
                1 => ScalarIntegral::A0,
                2 => ScalarIntegral::B0,
                3 => ScalarIntegral::C0,
                4 => ScalarIntegral::D0,
                n => {
                    return Err(ReductionError::Preparation(format!(
                        "scalar leaf {index} has unsupported propagator count {n}",
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
        let zero_profile_offset = (!plan.special_kinematics).then_some(argument_expressions.len());
        let mut zero_profile_pairs = Vec::new();
        if zero_profile_offset.is_some() {
            for i in 0..plan.settings.propagators {
                for j in i + 1..plan.settings.propagators {
                    zero_profile_pairs.push((i, j));
                    argument_expressions.push(plan.distances[i][j].clone());
                }
            }
        }
        let stability = StabilityPlan::prepare(plan)?;
        let activation_offset = argument_expressions.len();
        argument_expressions.extend_from_slice(stability.master_activations());
        let argument_count = argument_expressions.len();
        let arguments = compile(
            &argument_expressions,
            &plan.parameters,
            stability.master_activation_aliases(),
            &prototype,
            dual,
        )?;
        let mut reduction_expressions: Vec<_> = plan.moments.iter().flatten().cloned().collect();
        reduction_expressions.extend_from_slice(stability.expressions());
        let reduction_aliases: Vec<_> = plan
            .aliases
            .iter()
            .chain(stability.aliases())
            .cloned()
            .collect();
        let reduction_outputs = reduction_expressions.len();
        let reduction = compile(
            &reduction_expressions,
            &reduction_parameters,
            &reduction_aliases,
            &prototype,
            dual,
        )?;
        let mut masters = Vec::with_capacity(grouped_leaves.len());
        for (family, leaves) in grouped_leaves {
            let evaluator = match dual {
                Some(layout) => {
                    MasterEvaluator::new_dual(family, prototype.clone(), BorrowedDual(layout))
                }
                None => MasterEvaluator::new(family, prototype.clone()),
            }
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
            masters.push(MasterGroup {
                leaves,
                evaluator,
                input: Vec::new(),
                requests: Vec::new(),
            });
        }
        let fallback = plan
            .fallback
            .as_deref()
            .map(|fallback| Self::prepare(fallback, prototype.clone(), dual).map(Box::new))
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
                Some(Box::new(Self::prepare(&equal, prototype.clone(), dual)?)),
                Some(Box::new(Self::prepare(&unequal, prototype.clone(), dual)?)),
            )
        } else {
            (None, None)
        };
        let metadata = ResultMetadata::new(
            plan.settings.convention,
            plan.settings.tensor_weights,
            shape.clone(),
        );
        Ok(Self {
            accuracy_bits: prototype.get_precision(),
            prototype,
            basis: plan.basis.clone(),
            parameters: plan.parameters.len(),
            layout: plan.layout.clone(),
            components: shape.as_ref().map_or(1, Vec::len),
            shape,
            metadata,
            argument_offsets,
            argument_count,
            classification_offset,
            activation_offset,
            zero_profile_offset,
            zero_profile_pairs,
            zero_profile_plan: (!plan.special_kinematics).then(|| Box::new(plan.clone())),
            zero_profiles: HashMap::new(),
            arguments,
            reduction,
            reduction_outputs,
            stability,
            masters,
            valid_lanes: Vec::new(),
            successful_lanes: Vec::new(),
            input: Vec::new(),
            argument_values: Vec::new(),
            reduction_inputs: Vec::new(),
            successful_inputs: Vec::new(),
            output: Vec::new(),
            fallback,
            lightlike_equal,
            lightlike_unequal,
            stats: ExecutionStats::default(),
            profiling: false,
            profile: ExecutionProfile::default(),
            diagnostics: Vec::new(),
        })
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

    /// Set the accuracy target used by cancellation estimates. Working
    /// precision remains unchanged; a higher-precision rescue can target the
    /// accuracy originally requested by its caller.
    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(
                "accuracy precision must be positive and no greater than working precision".into(),
            ));
        }
        self.accuracy_bits = bits;
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
        self.profile = ExecutionProfile::default();
        let started = self.profiling.then(Instant::now);
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
        };
        self.valid_lanes.clear();
        self.successful_lanes.clear();
        self.input.clear();
        for (lane, row) in batch.values.chunks_exact(input_width).enumerate() {
            if let Some((slot, _)) = row.iter().enumerate().find(|(_, value)| !finite(value)) {
                result.statuses[lane] = LaneStatus::InvalidInput(format!(
                    "nonfinite input at argument {}, Taylor component {}",
                    slot / self.components,
                    slot % self.components,
                ));
                continue;
            }
            // mu, rather than mu², is the public interface. Squaring must not
            // hide a negative or complex scale before master validation.
            let scale = &row[self.layout.scale * self.components];
            if !scale.im.is_zero() || scale.re <= scale.re.zero() {
                result.statuses[lane] =
                    LaneStatus::InvalidInput("mu must be positive and real".into());
                continue;
            }
            if self
                .layout
                .shifts
                .clone()
                .chain(self.layout.scalar_products.clone().into_iter().flatten())
                .any(|argument| !row[argument * self.components].im.is_zero())
            {
                result.statuses[lane] = LaneStatus::InvalidInput(
                    "momentum-shift and scalar-product primal components must be real".into(),
                );
                continue;
            }
            if self.layout.masses.clone().any(|argument| {
                let mass = &row[argument * self.components];
                mass.im > mass.im.zero()
            }) {
                result.statuses[lane] = LaneStatus::InvalidInput(
                    "squared masses must have nonpositive imaginary parts".into(),
                );
                continue;
            }
            self.valid_lanes.push(lane);
            self.input.extend_from_slice(row);
        }
        let mut valid = self.valid_lanes.len();
        if valid == 0 {
            return Ok(result);
        }
        self.stats.argument_rows = valid;
        let argument_width = checked_mul(self.argument_count, self.components)?;
        self.argument_values
            .resize(checked_mul(valid, argument_width)?, zero.clone());
        let started = self.profiling.then(Instant::now);
        let evaluated =
            self.arguments
                .evaluate_batch(&self.input, &mut self.argument_values, valid);
        record_time(started, &mut self.profile.arguments);
        evaluated.map_err(|error| ReductionError::Evaluation(error.to_string()))?;

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
                dispatch_profile(
                    self.lightlike_equal.as_mut().unwrap(),
                    &equal,
                    batch,
                    &mut result,
                    &mut self.stats,
                    &mut self.profile,
                )?;
            }
            if !unequal.is_empty() {
                dispatch_profile(
                    self.lightlike_unequal.as_mut().unwrap(),
                    &unequal,
                    batch,
                    &mut result,
                    &mut self.stats,
                    &mut self.profile,
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
                                Self::prepare(&plan, self.prototype.clone(), layout.as_ref())
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
                        dispatch_profile(
                            workspace,
                            &lanes,
                            batch,
                            &mut result,
                            &mut self.stats,
                            &mut self.profile,
                        )?;
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
        for (compact, input) in self.input.chunks_exact(input_width).enumerate() {
            self.reduction_inputs
                [compact * reduction_width..compact * reduction_width + input_width]
                .clone_from_slice(input);
        }

        for group in &mut self.masters {
            let width = checked_mul(group.evaluator.family().arity(), self.components)?;
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
            let started = self.profiling.then(Instant::now);
            let evaluated = group.evaluator.evaluate_batch(&group.input, rows);
            record_time(started, &mut self.profile.scalar_masters);
            let values =
                evaluated.map_err(|error| ReductionError::Evaluation(error.to_string()))?;
            self.stats.master_family_batches += 1;
            self.stats.master_rows = self
                .stats
                .master_rows
                .checked_add(rows)
                .ok_or_else(size_error)?;
            for (request, &(leaf, compact)) in group.requests.iter().enumerate() {
                let lane = self.valid_lanes[compact];
                match values.lane(request) {
                    Err(error) => {
                        if result.statuses[lane] == LaneStatus::Success {
                            result.statuses[lane] = LaneStatus::MasterFailure(format!(
                                "{} leaf {leaf}: {error}",
                                group.evaluator.family().name(),
                            ));
                        }
                    }
                    Ok(value) => {
                        let start =
                            compact * reduction_width + input_width + leaf * 4 * self.components;
                        for (slot, coefficients) in [value.uv, value.ir, value.ir2, value.finite]
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
        self.successful_inputs.clear();
        for (compact, &lane) in self.valid_lanes.iter().enumerate() {
            if result.statuses[lane] == LaneStatus::Success {
                self.successful_lanes.push(lane);
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
        let evaluation_width = checked_mul(self.reduction_outputs, self.components)?;
        self.output
            .resize(checked_mul(successful, evaluation_width)?, zero);
        let started = self.profiling.then(Instant::now);
        if let Err(error) =
            self.reduction
                .evaluate_batch(&self.successful_inputs, &mut self.output, successful)
        {
            record_time(started, &mut self.profile.reduction);
            return Err(ReductionError::Evaluation(error.to_string()));
        }
        for (compact, &lane) in self.successful_lanes.iter().enumerate() {
            let evaluated =
                &self.output[compact * evaluation_width..(compact + 1) * evaluation_width];
            let (output, monitors) = evaluated.split_at(output_width);
            if !output.iter().all(finite) {
                result.statuses[lane] = LaneStatus::Unstable(
                    "reduction has nonfinite coefficients at this kinematic point".into(),
                );
            } else if let Some(reason) = self.stability.inspect_lane_with_accuracy(
                monitors,
                &self.successful_inputs[compact * reduction_width..(compact + 1) * reduction_width],
                self.components,
                &self.prototype,
                self.accuracy_bits,
            ) {
                result.statuses[lane] = LaneStatus::Unstable(reason);
            } else {
                result.values[lane * output_width..(lane + 1) * output_width]
                    .clone_from_slice(output);
            }
        }
        record_time(started, &mut self.profile.reduction);
        if let Some(fallback) = &mut self.fallback {
            let lanes: Vec<_> = result
                .statuses
                .iter()
                .enumerate()
                .filter_map(|(lane, status)| {
                    (matches!(status, LaneStatus::Unstable(_)) && !result.analytic_used[lane])
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
                })?;
                self.stats.fallback_rows = lanes.len();
                for (compact, &lane) in lanes.iter().enumerate() {
                    result.fallback_used[lane] = true;
                    self.diagnostics[lane].extend_from_slice(&fallback.diagnostics[compact]);
                    match retried.lane(compact) {
                        Ok(output) => {
                            result.values[lane * output_width..(lane + 1) * output_width]
                                .clone_from_slice(output);
                            result.statuses[lane] = LaneStatus::Success;
                            self.stats.fallback_successes += 1;
                        }
                        Err(status) => {
                            let primary = &result.statuses[lane];
                            result.statuses[lane] = LaneStatus::Unstable(format!(
                                "PV reduction failed its stability check ({primary:?}); IBP fallback returned {status:?}",
                            ));
                        }
                    }
                }
                let work = fallback.last_execution();
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

fn dispatch_profile<T: NativeFloat>(
    workspace: &mut Workspace<T>,
    lanes: &[usize],
    batch: &KinematicsBatch<T>,
    result: &mut BatchResult<T>,
    stats: &mut ExecutionStats,
    profile: &mut ExecutionProfile,
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
    })?;
    for (compact, &lane) in lanes.iter().enumerate() {
        result.analytic_used[lane] = true;
        result.fallback_used[lane] = evaluated.fallback_used[compact];
        result.statuses[lane] = evaluated.statuses[compact].clone();
        if let Ok(output) = evaluated.lane(compact) {
            result.values[lane * output_width..(lane + 1) * output_width].clone_from_slice(output);
        }
    }
    stats.analytic_rows = stats
        .analytic_rows
        .checked_add(lanes.len())
        .ok_or_else(size_error)?;
    let work = workspace.last_execution();
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

fn compile<T: NativeFloat, D: DualNumberStructure>(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: &[(Atom, Atom)],
    prototype: &T,
    dual: Option<&D>,
) -> Result<BatchExpressionEvaluator<Complex<T>>, ReductionError>
where
    Complex<T>: EvaluationDomain,
{
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
    }
    Ok(exact
        .map_coeff_with_prec(
            &|value| {
                Complex::new(
                    prototype.from_rational(&value.re),
                    prototype.from_rational(&value.im),
                )
            },
            prototype.get_precision(),
        )
        .into_batch_evaluator())
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
}
