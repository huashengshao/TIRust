//! Genuine finite D=6-2ε pentagons through a common squared-mass contour.
//!
//! `J6(m) = ∫₀∞ J4(m+t) dt`, with `J4 = Σᵢ bᵢ D0ᵢ` and
//! `bᵢ(t)=bᵢ(0)/(1-2t Σⱼbⱼ(0))`. Symbolica prepares all matrix algebra,
//! differentiation, and quadrature polynomials. Sampling and weighted sums are
//! numerical operations. See `docs/research/shifted-pentagon.md` for the proof.

use crate::{
    Complex, DualNumberStructure, ExactMasterPlan, HyperDual, LaneError, MasterBackend,
    MasterError, MasterEvaluator, MasterExecutionPolicy, NativeFloat, ScalarIntegral, default_master_backend,
    validate_dual_structure,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        float::SingleFloat,
        integer::Integer,
        rational::{Q, Rational},
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
    tensors::matrix::Matrix,
};

const INPUTS: usize = 16;
const GEOMETRY: usize = 6;
const CONTRACTION: usize = 12;
const EDGES: [(usize, usize); 10] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
];

/// Exact, precision-independent preparation for the genuine D6 pentagon.
///
/// Inputs are the ten squared distances in lexicographic `(i,j), i<j` order,
/// five squared masses, then `mu_squared`. Each argument contains all ordered
/// Taylor/Dual components; sample lanes are a separate outer dimension.
/// The evaluator currently requires either a strictly positive sufficient
/// Feynman-parameter bound or strictly negative widths for every mass. Singular
/// geometry and unqualified endpoints are reported independently per lane.
#[derive(Clone)]
pub struct Pentagon6Plan {
    geometry: ExpressionEvaluator<Complex<Rational>>,
    contraction: ExpressionEvaluator<Complex<Rational>>,
    boxes: PentagonBoxPlan,
    components: usize,
    master_policy: MasterExecutionPolicy,
    rules: Arc<Mutex<HashMap<(usize, u32), Arc<Vec<(Rational, Rational)>>>>>,
}

#[derive(Clone)]
enum PentagonBoxPlan {
    Exact(ExactMasterPlan),
    Native,
}

impl Pentagon6Plan {
    pub fn new() -> Result<Self, MasterError> {
        Self::new_with_policy(MasterExecutionPolicy::Default)
    }

    /// Select the policy for scalar dependencies. Explicit policies retain
    /// Symbolica D0 boxes; the pentagon geometry and quadrature are unchanged.
    pub fn new_with_policy(policy: MasterExecutionPolicy) -> Result<Self, MasterError> {
        policy.validate_available()?;
        Self::build::<HyperDual<f64>>(None, policy)
    }

    pub fn new_dual<D: DualNumberStructure>(dual: D) -> Result<Self, MasterError> {
        Self::new_dual_with_policy(dual, MasterExecutionPolicy::Default)
    }

    /// Prepare a Dual pentagon with the same scalar-dependency policy.
    pub fn new_dual_with_policy<D: DualNumberStructure>(
        dual: D,
        policy: MasterExecutionPolicy,
    ) -> Result<Self, MasterError> {
        policy.validate_available()?;
        validate_dual_structure(&dual)?;
        Self::build(Some(&dual), policy)
    }

    fn build<D: DualNumberStructure>(
        dual: Option<&D>,
        policy: MasterExecutionPolicy,
    ) -> Result<Self, MasterError> {
        let parameters: Vec<_> = (0..INPUTS).map(|i| variable(&format!("a{i}"))).collect();
        let mut distances = vec![vec![Atom::new(); 5]; 5];
        for (index, &(i, j)) in EDGES.iter().enumerate() {
            distances[i][j] = parameters[index].clone();
            distances[j][i] = parameters[index].clone();
        }
        let masses = &parameters[10..15];
        // Solve the equivalent 4x4 external Gram system. Keeping inverse,
        // v=Z^-1 f, and B as aliases prevents expanding mass polynomials.
        let gram = (1..5)
            .map(|i| {
                (1..5)
                    .map(|j| &distances[0][i] + &distances[0][j] - &distances[i][j])
                    .collect()
            })
            .collect();
        let inverse = Matrix::from_nested_vec(gram, exact_field())
            .map_err(|e| preparation(format!("Gram construction: {e:?}")))?
            .inv()
            .map_err(|e| preparation(format!("Gram inverse: {e:?}")))?;
        let f: Vec<_> = (1..5)
            .map(|i| &distances[0][i] - &masses[i] + &masses[0])
            .collect();
        let mut aliases = Vec::new();
        let mut v = Vec::new();
        for i in 0..4 {
            let mut row = Atom::new();
            for j in 0..4 {
                let entry = variable(&format!("z{i}{j}"));
                aliases.push((entry.clone(), inverse[(i, j)].clone()));
                row += entry * &f[j as usize];
            }
            let handle = variable(&format!("v{i}"));
            aliases.push((handle.clone(), row));
            v.push(handle);
        }
        let mut denominator = -Atom::num(2) * &masses[0];
        for (fi, vi) in f.iter().zip(&v) {
            denominator += fi * vi;
        }
        let b_sum = variable("B");
        aliases.push((b_sum.clone(), Atom::num(1) / denominator));
        let mut b = vec![Atom::num(1)];
        for vi in &v {
            b[0] -= vi;
        }
        b.extend(v);
        for value in &mut b {
            *value = &b_sum * &*value;
        }
        b.push(b_sum);
        let geometry = compile(&b, &parameters, &aliases, dual)?;

        // Each box and coefficient carries all Taylor components. Symbolica's
        // Dualizer performs their products; there is no handwritten jet algebra.
        let contraction_parameters: Vec<_> = (0..CONTRACTION)
            .map(|i| variable(&format!("c{i}")))
            .collect();
        let mut sum = Atom::new();
        for i in 0..5 {
            sum += &contraction_parameters[i] * &contraction_parameters[7 + i];
        }
        sum /=
            Atom::num(1) - Atom::num(2) * &contraction_parameters[6] * &contraction_parameters[5];
        let contraction = compile(&[sum], &contraction_parameters, &[], dual)?;
        let components = dual.map_or(1, DualNumberStructure::get_len);
        let boxes = match dual {
            Some(dual) => PentagonBoxPlan::Exact(ExactMasterPlan::new_dual(
                ScalarIntegral::D0,
                BorrowedDual(dual),
            )?),
            None if policy == MasterExecutionPolicy::Default
                && default_master_backend() == MasterBackend::RustBatch => PentagonBoxPlan::Native,
            None => PentagonBoxPlan::Exact(ExactMasterPlan::new(ScalarIntegral::D0)?),
        };
        Ok(Self {
            geometry,
            contraction,
            components,
            boxes,
            master_policy: policy,
            rules: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn component_count(&self) -> usize {
        self.components
    }

    pub fn instantiate<T: NativeFloat>(
        &self,
        prototype: T,
    ) -> Result<Pentagon6Evaluator<T>, MasterError>
    where
        Complex<T>: EvaluationDomain,
    {
        if prototype.get_precision() == 0 {
            return Err(MasterError::InvalidPrecision);
        }
        Ok(Pentagon6Evaluator {
            geometry: map(&self.geometry, &prototype),
            contraction: map(&self.contraction, &prototype),
            boxes: match &self.boxes {
                PentagonBoxPlan::Exact(plan) => {
                    plan.instantiate_with_policy(prototype.clone(), self.master_policy)?
                }
                PentagonBoxPlan::Native => MasterEvaluator::new_with_backend(
                    ScalarIntegral::D0,
                    prototype.clone(),
                    MasterBackend::RustBatch,
                )?,
            },
            prototype,
            components: self.components,
            rules: self.rules.clone(),
            geometry_workspace: Vec::new(),
            box_workspace: Vec::new(),
            contraction_workspace: Vec::new(),
            contraction_output: Vec::new(),
        })
    }
}

/// Numerical accuracy and contour controls, expressed at the chosen precision.
#[derive(Clone, Debug)]
pub struct Pentagon6Options<T: NativeFloat> {
    pub relative_tolerance: T,
    pub absolute_tolerance: T,
    /// The contour is `t = scale * (1 - i*contour_slope) * x/(1-x)`.
    /// A strictly positive slope moves every quadrature mass below its cut.
    pub contour_slope: T,
    pub initial_order: usize,
    pub maximum_order: usize,
}

impl<T: NativeFloat> Pentagon6Options<T> {
    pub fn new(prototype: &T) -> Self {
        Self {
            relative_tolerance: prototype
                .from_i64(2)
                .pow(u64::from(prototype.get_precision().saturating_mul(2) / 3))
                .inv(),
            absolute_tolerance: prototype.zero(),
            contour_slope: prototype.one(),
            initial_order: 8,
            maximum_order: 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pentagon6LaneError {
    InvalidInput(LaneError),
    /// A zero mass at the integration endpoint needs separate IR qualification.
    UnqualifiedMasslessBoundary,
    /// Neither a positive real Feynman denominator bound nor uniformly
    /// negative mass widths qualifies the integration endpoint as finite.
    UnqualifiedEndpoint,
    SingularGeometry,
    BoxFailure {
        pinch: usize,
        error: LaneError,
    },
    UnexpectedBoxPole,
    NonFinite,
    NonConvergent,
}

#[derive(Clone, Debug)]
pub struct Pentagon6Diagnostic<T: NativeFloat> {
    pub order: usize,
    pub box_rows: usize,
    /// Maximum successive-rule difference across all Taylor components.
    pub error_estimate: T,
    /// Absolute L-infinity complex difference between the last two rules,
    /// separately for each ordered Taylor component, in J6 normalization.
    /// These are convergence estimates, not rigorous interval bounds.
    pub component_error_estimates: Vec<T>,
    pub error: Option<Pentagon6LaneError>,
}

/// Finite coefficients and independent convergence/failure information per lane.
/// The qualified massive D6 pentagon has zero UV and IR pole coefficients.
#[derive(Clone, Debug)]
pub struct Pentagon6Batch<T: NativeFloat> {
    components: usize,
    values: Vec<Complex<T>>,
    diagnostics: Vec<Pentagon6Diagnostic<T>>,
}

impl<T: NativeFloat> Pentagon6Batch<T> {
    pub fn row_count(&self) -> usize {
        self.diagnostics.len()
    }
    pub fn component_count(&self) -> usize {
        self.components
    }
    pub fn diagnostics(&self) -> &[Pentagon6Diagnostic<T>] {
        &self.diagnostics
    }
    pub fn lane(&self, lane: usize) -> Result<&[Complex<T>], &Pentagon6LaneError> {
        if let Some(error) = &self.diagnostics[lane].error {
            return Err(error);
        }
        Ok(&self.values[lane * self.components..(lane + 1) * self.components])
    }
}

/// Owned scratch and generic instruction-major evaluators.
#[derive(Clone)]
pub struct Pentagon6Evaluator<T: NativeFloat> {
    prototype: T,
    components: usize,
    geometry: BatchExpressionEvaluator<Complex<T>>,
    contraction: BatchExpressionEvaluator<Complex<T>>,
    boxes: MasterEvaluator<T>,
    rules: Arc<Mutex<HashMap<(usize, u32), Arc<Vec<(Rational, Rational)>>>>>,
    geometry_workspace: Vec<Complex<T>>,
    box_workspace: Vec<Complex<T>>,
    contraction_workspace: Vec<Complex<T>>,
    contraction_output: Vec<Complex<T>>,
}

impl<T: NativeFloat> Pentagon6Evaluator<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Every refinement sends all active lanes, nodes, and box pinches through
    /// one D0 batch. Unsuccessful lanes are removed before the next refinement.
    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        rows: usize,
        options: &Pentagon6Options<T>,
    ) -> Result<Pentagon6Batch<T>, MasterError> {
        let c = self.components;
        #[cfg(test)]
        if precision_trace::enabled() {
            precision_trace::record("e6_begin", &[
                ("rows", rows.to_string()),
                ("components", c.to_string()),
                ("prototype_precision", self.prototype.get_precision().to_string()),
                ("observed_box_backend", precision_trace::string(&format!("{:?}", self.boxes.backend()))),
                ("initial_order", options.initial_order.to_string()),
                ("maximum_order", options.maximum_order.to_string()),
                ("relative_tolerance", precision_trace::real(&options.relative_tolerance)),
                ("absolute_tolerance", precision_trace::real(&options.absolute_tolerance)),
                ("contour_slope", precision_trace::real(&options.contour_slope)),
                ("inputs", precision_trace::complexes(input)),
            ]);
        }
        let expected = rows
            .checked_mul(INPUTS)
            .and_then(|n| n.checked_mul(c))
            .ok_or(MasterError::SizeOverflow)?;
        if input.len() != expected {
            return Err(MasterError::InputLength {
                expected,
                actual: input.len(),
            });
        }
        if options.initial_order < 2
            || options.initial_order % 2 != 0
            || options.maximum_order < options.initial_order.saturating_mul(4)
            || options.relative_tolerance <= self.prototype.zero()
            || options.absolute_tolerance < self.prototype.zero()
            || options.contour_slope <= self.prototype.zero()
            || !options.relative_tolerance.is_finite()
            || !options.absolute_tolerance.is_finite()
            || !options.contour_slope.is_finite()
        {
            return Err(MasterError::InvalidShape(
                "invalid pentagon quadrature controls".into(),
            ));
        }
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = Pentagon6Batch {
            components: c,
            values: vec![zero.clone(); rows * c],
            diagnostics: (0..rows)
                .map(|_| Pentagon6Diagnostic {
                    order: 0,
                    box_rows: 0,
                    error_estimate: self.prototype.zero(),
                    component_error_estimates: vec![self.prototype.zero(); c],
                    error: None,
                })
                .collect(),
        };
        if rows == 0 {
            return Ok(result);
        }
        self.geometry_workspace
            .resize(rows * GEOMETRY * c, zero.clone());
        self.geometry
            .evaluate_batch(input, &mut self.geometry_workspace, rows)
            .map_err(|e| MasterError::Evaluation(e.to_string()))?;
        #[cfg(test)]
        if precision_trace::enabled() {
            for lane in 0..rows {
                precision_trace::record("geometry", &[
                    ("lane", lane.to_string()),
                    ("values", precision_trace::complexes(&self.geometry_workspace[lane * GEOMETRY * c..(lane + 1) * GEOMETRY * c])),
                ]);
            }
        }
        let mut active = Vec::new();
        let mut scales = vec![self.prototype.zero(); rows];
        for lane in 0..rows {
            let point = &input[lane * INPUTS * c..(lane + 1) * INPUTS * c];
            if let Some(error) = validate(point, c) {
                #[cfg(test)]
                if precision_trace::enabled() {
                    precision_trace::record("endpoint_rejected", &[
                        ("lane", lane.to_string()),
                        ("error", precision_trace::string(&format!("{error:?}"))),
                    ]);
                }
                result.diagnostics[lane].error = Some(error);
                continue;
            }
            let geometry = &self.geometry_workspace[lane * GEOMETRY * c..(lane + 1) * GEOMETRY * c];
            if geometry.iter().any(|v| !finite(v)) {
                result.diagnostics[lane].error = Some(Pentagon6LaneError::SingularGeometry);
                continue;
            }
            // Scale is an arbitrary numerical contour parameter. Holding it
            // fixed while differentiating preserves the exact integral; every
            // physical mass/invariant jet remains in the evaluated integrand.
            for argument in 0..15 {
                let value = &point[argument * c];
                let magnitude = max_norm(value);
                if magnitude > scales[lane] {
                    scales[lane] = magnitude;
                }
            }
            #[cfg(test)]
            if precision_trace::enabled() {
                precision_trace::record("endpoint_accepted", &[
                    ("lane", lane.to_string()),
                    ("scale", precision_trace::real(&scales[lane])),
                ]);
            }
            active.push(lane);
        }
        let mut previous = vec![zero.clone(); rows * c];
        let mut successes = vec![0usize; rows];
        let mut order = options.initial_order;
        while !active.is_empty() && order <= options.maximum_order {
            let rule = self.rule(order)?;
            #[cfg(test)]
            if precision_trace::enabled() {
                for (node, (x, w)) in rule.iter().enumerate() {
                    precision_trace::record("exact_gauss_node", &[
                        ("order", order.to_string()),
                        ("node", node.to_string()),
                        ("x", precision_trace::string(&x.to_string())),
                        ("weight", precision_trace::string(&w.to_string())),
                    ]);
                }
            }
            let rule: Vec<_> = rule
                .iter()
                .map(|(x, w)| {
                    (
                        self.prototype.from_rational(x),
                        self.prototype.from_rational(w),
                    )
                })
                .collect();
            let node_rows = active
                .len()
                .checked_mul(order)
                .ok_or(MasterError::SizeOverflow)?;
            let box_rows = node_rows.checked_mul(5).ok_or(MasterError::SizeOverflow)?;
            let box_values = box_rows
                .checked_mul(11)
                .and_then(|n| n.checked_mul(c))
                .ok_or(MasterError::SizeOverflow)?;
            let contraction_values = node_rows
                .checked_mul(CONTRACTION)
                .and_then(|n| n.checked_mul(c))
                .ok_or(MasterError::SizeOverflow)?;
            self.box_workspace.clear();
            self.box_workspace.reserve(box_values);
            self.contraction_workspace.clear();
            self.contraction_workspace
                .resize(contraction_values, zero.clone());
            let mut weights = Vec::with_capacity(node_rows);
            for (prepared, &lane) in active.iter().enumerate() {
                let point = &input[lane * INPUTS * c..(lane + 1) * INPUTS * c];
                let geometry =
                    &self.geometry_workspace[lane * GEOMETRY * c..(lane + 1) * GEOMETRY * c];
                let ray = Complex::new(
                    scales[lane].clone(),
                    -(scales[lane].clone() * &options.contour_slope),
                );
                for (node, (x, w)) in rule.iter().enumerate() {
                    let one_minus = self.prototype.one() - x;
                    let t = &ray * Complex::new(x.clone() / &one_minus, self.prototype.zero());
                    let weight =
                        &ray * Complex::new(w.clone() / one_minus.pow(2), self.prototype.zero());
                    weights.push(weight);
                    let index = prepared * order + node;
                    #[cfg(test)]
                    if precision_trace::enabled() {
                        precision_trace::record("contour_node", &[
                            ("order", order.to_string()),
                            ("lane", lane.to_string()),
                            ("node", node.to_string()),
                            ("x", precision_trace::real(x)),
                            ("gauss_weight", precision_trace::real(w)),
                            ("ray", precision_trace::complex(&ray)),
                            ("t", precision_trace::complex(&t)),
                            ("weight", precision_trace::complex(&weights[index])),
                        ]);
                    }
                    let contraction = &mut self.contraction_workspace
                        [index * CONTRACTION * c..(index + 1) * CONTRACTION * c];
                    contraction[..GEOMETRY * c].clone_from_slice(geometry);
                    contraction[6 * c] = t.clone();
                    for pinch in 0..5 {
                        let remaining: Vec<_> = (0..5).filter(|&i| i != pinch).collect();
                        for (i, j) in [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)] {
                            let edge = edge_index(remaining[i], remaining[j]);
                            self.box_workspace
                                .extend_from_slice(&point[edge * c..(edge + 1) * c]);
                        }
                        for i in remaining {
                            let start = self.box_workspace.len();
                            self.box_workspace
                                .extend_from_slice(&point[(10 + i) * c..(11 + i) * c]);
                            self.box_workspace[start] += &t;
                        }
                        self.box_workspace.extend_from_slice(&point[15 * c..16 * c]);
                    }
                }
            }
            #[cfg(test)]
            if precision_trace::enabled() {
                for (prepared, &lane) in active.iter().enumerate() {
                    for node in 0..order {
                        for pinch in 0..5 {
                            let index = (prepared * order + node) * 5 + pinch;
                            precision_trace::record("box_input", &[
                                ("order", order.to_string()),
                                ("lane", lane.to_string()),
                                ("node", node.to_string()),
                                ("pinch", pinch.to_string()),
                                ("packed_box_row", index.to_string()),
                                ("values", precision_trace::complexes(&self.box_workspace[index * 11 * c..(index + 1) * 11 * c])),
                            ]);
                        }
                    }
                }
            }
            let boxes = self.boxes.evaluate_batch(&self.box_workspace, box_rows)?;
            #[cfg(test)]
            if precision_trace::enabled() {
                for (prepared, &lane) in active.iter().enumerate() {
                    for node in 0..order {
                        for pinch in 0..5 {
                            let index = (prepared * order + node) * 5 + pinch;
                            match boxes.lane(index) {
                                Ok(value) => precision_trace::record("box_result", &[
                                    ("order", order.to_string()),
                                    ("lane", lane.to_string()),
                                    ("node", node.to_string()),
                                    ("pinch", pinch.to_string()),
                                    ("packed_box_row", index.to_string()),
                                    ("error", "null".into()),
                                    ("finite", precision_trace::complexes(value.finite)),
                                    ("uv", precision_trace::complexes(value.uv)),
                                    ("ir", precision_trace::complexes(value.ir)),
                                    ("ir2", precision_trace::complexes(value.ir2)),
                                    ("total_single", precision_trace::complexes(value.total_single)),
                                ]),
                                Err(error) => precision_trace::record("box_result", &[
                                    ("order", order.to_string()),
                                    ("lane", lane.to_string()),
                                    ("node", node.to_string()),
                                    ("pinch", pinch.to_string()),
                                    ("packed_box_row", index.to_string()),
                                    ("error", precision_trace::string(&format!("{error:?}"))),
                                ]),
                            }
                        }
                    }
                }
            }
            for (prepared, &lane) in active.iter().enumerate() {
                result.diagnostics[lane].order = order;
                result.diagnostics[lane].box_rows += order * 5;
                for node in 0..order {
                    let index = prepared * order + node;
                    for pinch in 0..5 {
                        match boxes.lane(index * 5 + pinch) {
                            Err(error) => {
                                result.diagnostics[lane].error =
                                    Some(Pentagon6LaneError::BoxFailure {
                                        pinch,
                                        error: error.clone(),
                                    });
                            }
                            Ok(value) => {
                                if value
                                    .uv
                                    .iter()
                                    .chain(value.ir)
                                    .chain(value.ir2)
                                    .any(|v| !v.is_zero())
                                {
                                    result.diagnostics[lane].error =
                                        Some(Pentagon6LaneError::UnexpectedBoxPole);
                                }
                                let start = (index * CONTRACTION + 7 + pinch) * c;
                                self.contraction_workspace[start..start + c]
                                    .clone_from_slice(value.finite);
                            }
                        }
                    }
                }
            }
            self.contraction_output.resize(node_rows * c, zero.clone());
            self.contraction
                .evaluate_batch(
                    &self.contraction_workspace,
                    &mut self.contraction_output,
                    node_rows,
                )
                .map_err(|e| MasterError::Evaluation(e.to_string()))?;
            #[cfg(test)]
            if precision_trace::enabled() {
                for (prepared, &lane) in active.iter().enumerate() {
                    for node in 0..order {
                        let index = prepared * order + node;
                        // Observation only: the production accumulator below is unchanged.
                        let weighted: Vec<_> = self.contraction_output[index * c..(index + 1) * c]
                            .iter().map(|value| &weights[index] * value).collect();
                        precision_trace::record("node_contraction", &[
                            ("order", order.to_string()),
                            ("lane", lane.to_string()),
                            ("node", node.to_string()),
                            ("inputs", precision_trace::complexes(&self.contraction_workspace[index * CONTRACTION * c..(index + 1) * CONTRACTION * c])),
                            ("values", precision_trace::complexes(&self.contraction_output[index * c..(index + 1) * c])),
                            ("weight", precision_trace::complex(&weights[index])),
                            ("weighted_values", precision_trace::complexes(&weighted)),
                        ]);
                    }
                }
            }
            let mut next = Vec::new();
            for (prepared, &lane) in active.iter().enumerate() {
                if result.diagnostics[lane].error.is_some() {
                    continue;
                }
                let mut current = vec![zero.clone(); c];
                for node in 0..order {
                    let index = prepared * order + node;
                    for component in 0..c {
                        current[component] +=
                            &weights[index] * &self.contraction_output[index * c + component];
                    }
                }
                #[cfg(test)]
                if precision_trace::enabled() {
                    precision_trace::record("rule_sum_diagnostic_only", &[
                        ("order", order.to_string()),
                        ("lane", lane.to_string()),
                        ("current", precision_trace::complexes(&current)),
                        ("previous", precision_trace::complexes(&previous[lane * c..(lane + 1) * c])),
                        ("success_streak_before", successes[lane].to_string()),
                    ]);
                }
                if current.iter().any(|v| !finite(v)) {
                    result.diagnostics[lane].error = Some(Pentagon6LaneError::NonFinite);
                    continue;
                }
                let (error, within_tolerance, component_errors) = match assess_convergence(
                    &current,
                    &previous[lane * c..(lane + 1) * c],
                    options,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        result.diagnostics[lane].error = Some(error);
                        continue;
                    }
                };
                let converged = order > options.initial_order && within_tolerance;
                result.diagnostics[lane].error_estimate = error;
                result.diagnostics[lane].component_error_estimates = component_errors;
                if converged {
                    successes[lane] += 1;
                } else {
                    successes[lane] = 0;
                }
                #[cfg(test)]
                if precision_trace::enabled() {
                    precision_trace::record("assessment", &[
                        ("order", order.to_string()),
                        ("lane", lane.to_string()),
                        ("within_tolerance", within_tolerance.to_string()),
                        ("converged_comparison", converged.to_string()),
                        ("success_streak_after", successes[lane].to_string()),
                        ("error_estimate", precision_trace::real(&result.diagnostics[lane].error_estimate)),
                        ("component_error_estimates", precision_trace::reals(&result.diagnostics[lane].component_error_estimates)),
                    ]);
                }
                previous[lane * c..(lane + 1) * c].clone_from_slice(&current);
                if successes[lane] >= 2 {
                    result.values[lane * c..(lane + 1) * c].clone_from_slice(&current);
                } else {
                    next.push(lane);
                }
            }
            active = next;
            order = order.checked_mul(2).ok_or(MasterError::SizeOverflow)?;
        }
        for lane in active {
            result.diagnostics[lane].error = Some(Pentagon6LaneError::NonConvergent);
        }
        #[cfg(test)]
        if precision_trace::enabled() {
            for lane in 0..rows {
                let diagnostic = &result.diagnostics[lane];
                let error = diagnostic.error.as_ref().map_or_else(|| "null".into(), |error| precision_trace::string(&format!("{error:?}")));
                let public_values = match result.lane(lane) {
                    Ok(values) => precision_trace::complexes(values),
                    Err(_) => "null".into(),
                };
                precision_trace::record("e6_lane_final", &[
                    ("lane", lane.to_string()),
                    ("order", diagnostic.order.to_string()),
                    ("box_rows", diagnostic.box_rows.to_string()),
                    ("error", error),
                    ("error_estimate", precision_trace::real(&diagnostic.error_estimate)),
                    ("component_error_estimates", precision_trace::reals(&diagnostic.component_error_estimates)),
                    ("public_values", public_values),
                    ("last_approximant_diagnostic_only", precision_trace::complexes(&previous[lane * c..(lane + 1) * c])),
                ]);
            }
        }
        Ok(result)
    }

    fn rule(&self, order: usize) -> Result<Arc<Vec<(Rational, Rational)>>, MasterError> {
        let key = (order, self.prototype.get_precision());
        {
            let rules = self
                .rules
                .lock()
                .map_err(|_| preparation("quadrature cache poisoned"))?;
            if let Some((_, rule)) = rules
                .iter()
                .filter(|((n, bits), _)| *n == order && *bits >= key.1)
                .min_by_key(|((_, bits), _)| *bits)
            {
                return Ok(rule.clone());
            }
        }
        let rule = Arc::new(gauss_legendre(order, key.1)?);
        self.rules
            .lock()
            .map_err(|_| preparation("quadrature cache poisoned"))?
            .insert(key, rule.clone());
        Ok(rule)
    }
}

// Rodrigues' polynomial, differentiation, certified root isolation/refinement,
// and rational polynomial evaluation all use existing Symbolica operations.
fn gauss_legendre(order: usize, bits: u32) -> Result<Vec<(Rational, Rational)>, MasterError> {
    let n = u32::try_from(order).map_err(|_| MasterError::SizeOverflow)?;
    if order == 0 || order % 2 != 0 {
        return Err(preparation("quadrature order must be positive and even"));
    }
    let symbol = symbolica::symbol!("iregi_pentagon6::quadrature_x");
    let x = Atom::var(symbol);
    let mut expression = (x.pow(2) - Atom::num(1)).pow(order as i64).expand();
    for _ in 0..order {
        expression = expression.derivative(symbol);
    }
    expression /= Atom::num(Integer::from(2).pow(order as u64) * Integer::factorial(n));
    let polynomial = expression
        .to_polynomial::<_, u32>(&Q, None)
        .to_univariate_from_univariate(0);
    let derivative = polynomial.derivative();
    let tolerance = Rational::from((Integer::from(1), Integer::from(2).pow(u64::from(bits) + 32)));
    let intervals = polynomial.isolate_real_root_intervals();
    if intervals.len() != order {
        return Err(preparation(
            "Legendre root isolation returned wrong root count",
        ));
    }
    let mut result = Vec::with_capacity(order);
    for (lo, hi, multiplicity) in intervals {
        if multiplicity != 1 {
            return Err(preparation("Legendre root was not simple"));
        }
        let (lo, hi) = polynomial.refine_root_interval((lo, hi), &tolerance);
        let root = (lo + hi) / Rational::from(2);
        let slope = derivative.evaluate(&root);
        let weight = Rational::from(1) / ((Rational::from(1) - root.pow(2)) * slope.pow(2));
        result.push(((root + Rational::from(1)) / Rational::from(2), weight));
    }
    Ok(result)
}

fn validate<T: NativeFloat>(point: &[Complex<T>], c: usize) -> Option<Pentagon6LaneError> {
    for (i, value) in point.iter().enumerate() {
        if !finite(value) {
            return Some(Pentagon6LaneError::InvalidInput(
                LaneError::NonFiniteInput {
                    argument: i / c,
                    component: i % c,
                },
            ));
        }
    }
    for i in 0..10 {
        if !point[i * c].im.is_zero() {
            return Some(Pentagon6LaneError::InvalidInput(
                LaneError::ComplexInvariant { argument: i },
            ));
        }
    }
    for i in 10..15 {
        if point[i * c].im > point[i * c].im.zero() {
            return Some(Pentagon6LaneError::InvalidInput(
                LaneError::UpperHalfPlaneMass { argument: i },
            ));
        }
        if point[i * c].is_zero() {
            return Some(Pentagon6LaneError::UnqualifiedMasslessBoundary);
        }
    }
    if !point[15 * c].im.is_zero() || point[15 * c].re <= point[15 * c].re.zero() {
        return Some(Pentagon6LaneError::InvalidInput(LaneError::InvalidScale));
    }
    let zero = point[10 * c].re.zero();
    if (10..15).all(|i| point[i * c].im < zero) {
        // On the parameter simplex, Im(F) is a convex combination of these
        // strictly negative widths, so it cannot vanish at the endpoint.
        return None;
    }
    let mut minimum_mass = point[10 * c].re.clone();
    for i in 11..15 {
        if point[i * c].re < minimum_mass {
            minimum_mass = point[i * c].re.clone();
        }
    }
    if minimum_mass <= zero {
        return Some(Pentagon6LaneError::UnqualifiedEndpoint);
    }
    let mut scale = minimum_mass.clone();
    for i in 0..10 {
        if point[i * c].re > scale {
            scale = point[i * c].re.clone();
        }
    }
    let four = zero.from_i64(4);
    let mut positive_invariants = zero.clone();
    for i in 0..10 {
        if point[i * c].re > zero {
            // x_i*x_j <= 1/4 on the simplex. Normalize before summing so
            // overflow or uniformly subnormal inputs cannot defeat the bound.
            positive_invariants += (point[i * c].re.clone() / &scale) / &four;
            if !positive_invariants.is_finite() {
                return Some(Pentagon6LaneError::UnqualifiedEndpoint);
            }
        }
    }
    let bound = minimum_mass / &scale - positive_invariants;
    if bound.is_finite() && bound > zero {
        None
    } else {
        Some(Pentagon6LaneError::UnqualifiedEndpoint)
    }
}

fn edge_index(i: usize, j: usize) -> usize {
    let edge = if i < j { (i, j) } else { (j, i) };
    EDGES
        .iter()
        .position(|&p| p == edge)
        .expect("distinct pentagon vertices")
}
fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("iregi_pentagon6::{name}")))
}
fn preparation(message: impl Into<String>) -> MasterError {
    MasterError::Preparation(message.into())
}
fn finite<T: NativeFloat>(value: &Complex<T>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
fn max_norm<T: NativeFloat>(value: &Complex<T>) -> T {
    let real = value.re.norm();
    let imaginary = value.im.norm();
    if real > imaginary { real } else { imaginary }
}
fn assess_convergence<T: NativeFloat>(
    current: &[Complex<T>],
    previous: &[Complex<T>],
    options: &Pentagon6Options<T>,
) -> Result<(T, bool, Vec<T>), Pentagon6LaneError> {
    let mut error = options.absolute_tolerance.zero();
    let mut converged = true;
    let mut component_errors = Vec::with_capacity(current.len());
    for (current, previous) in current.iter().zip(previous) {
        let difference = current - previous;
        if !finite(current) || !finite(previous) || !finite(&difference) {
            return Err(Pentagon6LaneError::NonFinite);
        }
        let magnitude = max_norm(&difference);
        let norm = max_norm(current);
        let threshold =
            options.absolute_tolerance.clone() + options.relative_tolerance.clone() * &norm;
        #[cfg(test)]
        if precision_trace::enabled() {
            // In the trace this immediately follows rule_sum_diagnostic_only,
            // which supplies the unchanged caller's order and lane context.
            precision_trace::record("convergence_component", &[
                ("component", component_errors.len().to_string()),
                ("current", precision_trace::complex(current)),
                ("previous", precision_trace::complex(previous)),
                ("difference", precision_trace::complex(&difference)),
                ("magnitude", precision_trace::real(&magnitude)),
                ("current_norm", precision_trace::real(&norm)),
                ("threshold", precision_trace::real(&threshold)),
            ]);
        }
        if !magnitude.is_finite() || !norm.is_finite() || !threshold.is_finite() {
            return Err(Pentagon6LaneError::NonFinite);
        }
        if magnitude > error {
            error = magnitude.clone();
        }
        if magnitude > threshold {
            converged = false;
        }
        component_errors.push(magnitude);
    }
    Ok((error, converged, component_errors))
}
fn exact_field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: Some(Box::new(|view, out| {
            *out = view.together();
            true
        })),
    }
}

fn compile<D: DualNumberStructure>(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: &[(Atom, Atom)],
    dual: Option<&D>,
) -> Result<ExpressionEvaluator<Complex<Rational>>, MasterError> {
    let mut evaluator = Atom::evaluator_multiple(expressions, parameters)
        .add_aliases(aliases.iter().cloned())
        .map_err(|e| preparation(e.to_string()))?
        .direct_translation(true)
        .horner_iterations(0)
        .cores(1)
        .build()
        .map_err(|e| preparation(e.to_string()))?;
    if let Some(dual) = dual {
        evaluator = evaluator
            .vectorize(&Dualizer::new(BorrowedDual(dual), Vec::new()))
            .map_err(preparation)?;
    }
    Ok(evaluator)
}
fn map<T: NativeFloat>(
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


// This observer is compiled only in tests and is activated only by the ignored
// probe below. It records existing kernel buffers without changing acceptance.
#[cfg(test)]
mod precision_trace {
    use super::*;
    use std::{cell::RefCell, fs::OpenOptions, io::{BufWriter, Write}};
    use symbolica::domains::float::Real;

    thread_local! {
        static SINK: RefCell<Option<BufWriter<std::fs::File>>> = const { RefCell::new(None) };
    }

    pub(super) const OUTPUT_ENV: &str = "TIRUST_E6_TERMINAL0_TRACE_JSONL";

    pub(super) fn begin() {
        let path = std::env::var_os(OUTPUT_ENV).expect("set the explicit diagnostic JSONL output path");
        let file = OpenOptions::new().write(true).create_new(true).open(path)
            .expect("create a fresh diagnostic output file without overwriting evidence");
        SINK.with(|sink| {
            assert!(sink.borrow().is_none(), "one diagnostic observer per thread");
            *sink.borrow_mut() = Some(BufWriter::new(file));
        });
    }

    pub(super) fn finish() {
        SINK.with(|sink| {
            if let Some(mut writer) = sink.borrow_mut().take() {
                writer.flush().expect("flush diagnostic evidence");
            }
        });
    }

    pub(super) fn enabled() -> bool { SINK.with(|sink| sink.borrow().is_some()) }

    pub(super) fn string(value: &str) -> String {
        let mut output = String::from("\"");
        for character in value.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                c if c < '\u{20}' => {
                    use std::fmt::Write as _;
                    write!(output, "\\u{:04x}", c as u32).unwrap();
                }
                c => output.push(c),
            }
        }
        output.push('"');
        output
    }

    pub(super) fn real<T: NativeFloat>(value: &T) -> String {
        let rational = value.exact_rational().map_or_else(|| "null".into(), |value| string(&value.to_string()));
        format!("{{\"exact_rational\":{rational},\"precision\":{},\"negative_sign\":{},\"finite\":{},\"display\":{}}}",
            value.get_precision(), value.negative_sign(), value.is_finite(), string(&format!("{value}")))
    }

    pub(super) fn complex<T: NativeFloat>(value: &Complex<T>) -> String {
        format!("{{\"re\":{},\"im\":{}}}", real(&value.re), real(&value.im))
    }

    pub(super) fn complexes<T: NativeFloat>(values: &[Complex<T>]) -> String {
        format!("[{}]", values.iter().map(complex).collect::<Vec<_>>().join(","))
    }

    pub(super) fn reals<T: NativeFloat>(values: &[T]) -> String {
        format!("[{}]", values.iter().map(real).collect::<Vec<_>>().join(","))
    }

    pub(super) fn record(event: &str, fields: &[(&str, String)]) {
        SINK.with(|sink| {
            let mut sink = sink.borrow_mut();
            let Some(writer) = sink.as_mut() else { return; };
            write!(writer, "{{\"event\":{}", string(event)).expect("write diagnostic event");
            for (key, value) in fields {
                write!(writer, ",{}:{}", string(key), value).expect("write diagnostic field");
            }
            writeln!(writer, "}}").expect("write diagnostic line");
            // A native child may hit its explicit time limit. Keep every
            // completed observation on disk instead of retaining a final buffer.
            writer.flush().expect("flush diagnostic line");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DoubleFloat, Float};
    use symbolica::domains::float::FloatLike;
    #[test]
    fn explicit_policy_keeps_pentagon_boxes_symbolica() {
        let mut policies = vec![MasterExecutionPolicy::Symbolica];
        if cfg!(feature = "native-f64-abc") {
            policies.push(MasterExecutionPolicy::NativeF64Abc);
        }
        for policy in policies {
            let plan = Pentagon6Plan::new_with_policy(policy).unwrap();
            assert_eq!(plan.master_policy, policy);
            assert!(matches!(&plan.boxes, PentagonBoxPlan::Exact(_)));
            assert_eq!(plan.instantiate(0.).unwrap().boxes.backend(), MasterBackend::SymbolicaBatch);
        }
    }

    /// Capture only: success of this test means the one observation completed,
    /// not that the scalar or any parent tensor was numerically validated.
    #[test]
    #[ignore = "bounded explicit diagnostic; requires a fresh TIRUST_E6_TERMINAL0_TRACE_JSONL path"]
    fn trace_terminal0_float256_diagnostic_only() {
        precision_trace::begin();
        precision_trace::record("probe_metadata", &[
            ("purpose", precision_trace::string("one reconstructed terminal-0 scalar E6 observation; no numerical acceptance claim")),
            ("numeric_agreement_claimed", "false".into()),
            ("working_bits", "256".into()),
            ("requested_accuracy_bits", "53".into()),
            ("input_construction", precision_trace::string("exact endpoint rationals mapped directly into Float256; original core master-argument tracked precisions were not captured")),
            ("limitation", precision_trace::string("nonreproduction at this ideal reconstructed input does not qualify or fix the original six-point tensor failure; no captured input precision is padded")),
            ("endpoint_source", precision_trace::string("original lines [0,2,3,4,5], edge order (01,02,03,04,12,13,14,23,24,34)")),
            ("maximum_d0_rows", "1240".into()),
        ]);
        let prototype = Float::new(256);
        let endpoints: [(i64, i64); INPUTS] = [
            (-1, 1), (-1, 1), (1, 4), (-2, 1), (-2, 1),
            (-3, 4), (-1, 1), (-3, 4), (-3, 1), (-7, 4),
            (1, 1), (3, 1), (4, 1), (2, 1), (3, 1), (9, 1),
        ];
        let input: Vec<_> = endpoints.into_iter().map(|fraction| {
            Complex::new(prototype.from_rational(&Rational::from(fraction)), prototype.zero())
        }).collect();
        let plan = Pentagon6Plan::new().expect("prepare the unchanged default scalar plan");
        assert_eq!(plan.master_policy, MasterExecutionPolicy::Default);
        assert_eq!(plan.component_count(), 1);
        let mut evaluator = plan.instantiate(prototype.clone()).expect("instantiate Float256");
        assert_eq!(evaluator.boxes.backend(), MasterBackend::SymbolicaBatch);
        let mut options = Pentagon6Options::new(&prototype);
        // The original tensor caller requests 53 accuracy bits and applies
        // 2^(17-53). Preserve that target rather than the Float256 default.
        options.relative_tolerance = prototype.from_i64(2).pow(36).inv();
        options.absolute_tolerance = prototype.zero();
        options.contour_slope = prototype.one();
        options.initial_order = 8;
        options.maximum_order = 128;
        match evaluator.evaluate_batch(&input, 1, &options) {
            Ok(batch) => {
                let diagnostic = &batch.diagnostics()[0];
                assert!(diagnostic.box_rows <= 1240, "the fixed single-point row cap");
                let status = match batch.lane(0) {
                    Ok(_) => "API_success_diagnostic_only",
                    Err(_) => "API_failure_retained",
                };
                precision_trace::record("probe_complete", &[
                    ("status", precision_trace::string(status)),
                    ("numeric_agreement_claimed", "false".into()),
                    ("observed_rows", batch.row_count().to_string()),
                    ("observed_components", batch.component_count().to_string()),
                    ("observed_box_rows", diagnostic.box_rows.to_string()),
                ]);
            }
            Err(error) => {
                precision_trace::record("probe_complete", &[
                    ("status", precision_trace::string("outer_master_error_retained")),
                    ("error", precision_trace::string(&format!("{error:?}"))),
                    ("numeric_agreement_claimed", "false".into()),
                ]);
            }
        }
        precision_trace::finish();
    }

    const SAMPLE: [i64; 16] = [1, -1, -1, -1, 0, 0, 0, -2, -2, -2, 10, 12, 15, 18, 22, 1];
    fn sample() -> Vec<Complex<f64>> {
        SAMPLE.map(|re| Complex::new(re as f64, 0.)).to_vec()
    }
    fn decimal(value: &str) -> Rational {
        let (integer, fraction) = value.split_once('.').unwrap();
        Rational::from((
            format!("{integer}{fraction}").parse::<Integer>().unwrap(),
            Integer::from(10).pow(fraction.len() as u64),
        ))
    }
    #[test]
    fn symbolica_gauss_rule_integrates_polynomial_moments() {
        let rule = gauss_legendre(8, 80).unwrap();
        for degree in 0..16 {
            let result = rule
                .iter()
                .fold(Rational::from(0), |sum, (x, w)| sum + w * &x.pow(degree));
            let error = (result - Rational::from((1, degree as i64 + 1))).abs();
            assert!(error < Rational::from((Integer::from(1), Integer::from(2).pow(70))));
        }
    }
    #[test]
    fn common_mass_rank_one_identity_is_exact() {
        let point = sample();
        let b: [Atom; 5] = [25, 1, -6, -9, -13].map(|v| Atom::num(v) / Atom::num(325));
        let t = variable("test_t");
        let sum = b.iter().fold(Atom::new(), |v, b| v + b);
        for i in 0..5 {
            let mut value = Atom::new();
            for j in 0..5 {
                let s = if i == j {
                    0
                } else {
                    point[edge_index(i, j)].re as i64
                };
                let r = Atom::num(s - point[10 + i].re as i64 - point[10 + j].re as i64)
                    - Atom::num(2) * &t;
                value += r * &b[j] / (Atom::num(1) - Atom::num(2) * &t * &sum);
            }
            assert!((value - Atom::num(1)).together().is_zero());
        }
    }
    #[test]
    fn genuine_pentagon_matches_independent_parameter_oracle_in_batch() {
        let plan = Pentagon6Plan::new().unwrap();
        let mut evaluator = plan.instantiate(0.).unwrap();
        let options = Pentagon6Options::new(&0.);
        let mut inputs = sample().repeat(3);
        inputs[16 + 10].im = 1.;
        let results = evaluator.evaluate_batch(&inputs, 3, &options).unwrap();
        for lane in [0, 2] {
            let got = &results.lane(lane).unwrap()[0];
            assert!(
                (got.re + 0.0001771401382640081806076521310069).abs() < 2e-14,
                "{got:?}"
            );
            assert!(got.im.abs() < 2e-14, "{got:?}");
            assert!(results.diagnostics()[lane].order >= 32);
        }
        assert!(matches!(
            results.lane(1),
            Err(Pentagon6LaneError::InvalidInput(
                LaneError::UpperHalfPlaneMass { .. }
            ))
        ));
        assert_eq!(
            evaluator
                .evaluate_batch(&[], 0, &options)
                .unwrap()
                .row_count(),
            0
        );
    }

    fn check_precision<T: NativeFloat>(plan: &Pentagon6Plan, prototype: T)
    where
        Complex<T>: EvaluationDomain,
    {
        let input: Vec<_> = SAMPLE
            .iter()
            .map(|&value| Complex::new(prototype.from_i64(value), prototype.zero()))
            .collect();
        let mut evaluator = plan.instantiate(prototype.clone()).unwrap();
        let mut options = Pentagon6Options::new(&prototype);
        // Enough to demonstrate digits inaccessible to binary64 without making
        // this focused qualification depend on very high quadrature orders.
        options.relative_tolerance = prototype.from_i64(10).pow(20).inv();
        let result = evaluator.evaluate_batch(&input, 1, &options).unwrap();
        let actual = &result.lane(0).unwrap()[0];
        let expected =
            prototype.from_rational(&decimal("-0.0001771401382640081806076521310069008176"));
        let error = (actual.re.clone() - expected).norm() + actual.im.norm();
        assert!(
            error < prototype.from_i64(10).pow(23).inv(),
            "precision={} error={error}",
            prototype.get_precision()
        );
    }

    #[test]
    fn genuine_pentagon_preserves_doublefloat_and_arbitrary_precision() {
        let plan = Pentagon6Plan::new().unwrap();
        check_precision(&plan, Float::new(128));
        check_precision(&plan, DoubleFloat::from(0.));
    }

    #[test]
    fn common_mass_dual_derivative_equals_negative_d4() {
        let plan = Pentagon6Plan::new_dual(HyperDual::<f64>::new(vec![vec![0], vec![1]])).unwrap();
        assert_eq!(plan.component_count(), 2);
        let mut evaluator = plan.instantiate(0.).unwrap();
        let input: Vec<_> = SAMPLE
            .iter()
            .enumerate()
            .flat_map(|(argument, &value)| {
                [
                    Complex::new(value as f64, 0.),
                    Complex::new(if (10..15).contains(&argument) { 1. } else { 0. }, 0.),
                ]
            })
            .collect();
        let options = Pentagon6Options::new(&0.);
        let result = evaluator
            .evaluate_batch(&input.repeat(3), 3, &options)
            .unwrap();
        assert_eq!(result.component_count(), 2);
        for lane in 0..3 {
            let jet = result.lane(lane).unwrap();
            assert!((jet[0].re + 0.000177140138264008180607652131).abs() < 2e-14);
            assert!(
                (jet[1].re - 0.0000236059994160983338257053727).abs() < 2e-14,
                "{:?}",
                jet[1]
            );
            assert!(jet[1].im.abs() < 2e-14);
        }
    }

    #[test]
    fn convergence_rejects_overflow_in_differences_and_thresholds() {
        let mut options = Pentagon6Options::new(&0.);
        let high = Complex::new(f64::MAX, f64::MAX);
        let low = Complex::new(-f64::MAX, -f64::MAX);
        assert_eq!(
            assess_convergence(&[high], &[low], &options),
            Err(Pentagon6LaneError::NonFinite)
        );
        options.relative_tolerance = f64::MAX;
        assert_eq!(
            assess_convergence(&[high], &[high], &options),
            Err(Pentagon6LaneError::NonFinite)
        );
    }

    #[test]
    fn endpoint_guard_accepts_positive_feynman_bound_and_negative_widths() {
        let mut point = sample();
        assert_eq!(validate(&point, 1), None);
        point[10].im = -1.;
        assert_eq!(validate(&point, 1), None);

        // A timelike endpoint with an excluded real bound remains qualified
        // when every mass has a strictly negative imaginary part.
        for value in &mut point[..10] {
            value.re = 1000.;
        }
        for value in &mut point[10..15] {
            value.im = -1.;
        }
        assert_eq!(validate(&point, 1), None);
        point[14].im = 0.;
        assert_eq!(
            validate(&point, 1),
            Some(Pentagon6LaneError::UnqualifiedEndpoint)
        );
    }

    #[test]
    fn endpoint_guard_rejects_zero_width_threshold_and_keeps_massless_diagnostic() {
        let mut point = sample();
        for value in &mut point[..10] {
            value.re = 0.;
        }
        for value in &mut point[10..15] {
            value.re = 1.;
        }
        point[0].re = 4.;
        assert_eq!(
            validate(&point, 1),
            Some(Pentagon6LaneError::UnqualifiedEndpoint)
        );
        point[0].re = 5.;
        assert_eq!(
            validate(&point, 1),
            Some(Pentagon6LaneError::UnqualifiedEndpoint)
        );
        point[10].re = 0.;
        assert_eq!(
            validate(&point, 1),
            Some(Pentagon6LaneError::UnqualifiedMasslessBoundary)
        );
    }

    #[test]
    fn endpoint_guard_handles_large_finite_bounds_without_overflow_acceptance() {
        let mut point = sample();
        for value in &mut point[..10] {
            value.re = 0.;
        }
        for value in &mut point[10..15] {
            value.re = f64::MAX;
        }
        point[0].re = f64::MAX;
        point[1].re = f64::MAX;
        assert_eq!(validate(&point, 1), None);
        for value in &mut point[..10] {
            value.re = f64::MAX;
        }
        assert_eq!(
            validate(&point, 1),
            Some(Pentagon6LaneError::UnqualifiedEndpoint)
        );
        for value in &mut point[10..15] {
            value.im = -1.;
        }
        assert_eq!(validate(&point, 1), None);
    }

    #[test]
    fn endpoint_guard_does_not_lose_subnormal_invariant_penalties() {
        let mut point = sample();
        for value in &mut point[..15] {
            value.re = f64::from_bits(1);
        }
        assert_eq!(
            validate(&point, 1),
            Some(Pentagon6LaneError::UnqualifiedEndpoint)
        );
        for value in &mut point[..10] {
            value.re = -value.re;
        }
        assert_eq!(validate(&point, 1), None);
    }

    #[test]
    fn endpoint_guard_uses_primal_components_and_preserves_precision() {
        let mut point: Vec<_> = sample()
            .into_iter()
            .flat_map(|primal| [primal, Complex::new(f64::MAX, -1.)])
            .collect();
        assert_eq!(validate(&point, 2), None);
        for i in 0..10 {
            point[2 * i].re = 1000.;
        }
        assert_eq!(
            validate(&point, 2),
            Some(Pentagon6LaneError::UnqualifiedEndpoint)
        );

        fn check_margin<T: NativeFloat>(prototype: T) {
            let mut point = vec![Complex::new(prototype.zero(), prototype.zero()); INPUTS];
            point[0].re = prototype.from_i64(4);
            let mass = prototype.one() + prototype.from_i64(2).pow(100).inv();
            for value in &mut point[10..15] {
                value.re = mass.clone();
            }
            point[15].re = prototype.one();
            assert_eq!(validate(&point, 1), None);
            point[10].re = prototype.one();
            assert_eq!(
                validate(&point, 1),
                Some(Pentagon6LaneError::UnqualifiedEndpoint)
            );
        }
        check_margin(DoubleFloat::from(0.));
        check_margin(Float::new(160));
    }

    #[test]
    fn lower_contour_and_tiny_scale_are_explicit() {
        let plan = Pentagon6Plan::new().unwrap();
        let mut evaluator = plan.instantiate(0.).unwrap();
        let mut options = Pentagon6Options::new(&0.);
        options.contour_slope = 0.;
        assert!(matches!(
            evaluator.evaluate_batch(&sample(), 1, &options),
            Err(MasterError::InvalidShape(_))
        ));
        options.contour_slope = 1.;
        let scale = 1e-20;
        let mut point = sample();
        for value in &mut point[..15] {
            value.re *= scale;
        }
        let result = evaluator.evaluate_batch(&point, 1, &options).unwrap();
        let value = &result.lane(0).unwrap()[0];
        assert!(
            (value.re * scale * scale + 0.0001771401382640081806).abs() < 2e-14,
            "{value:?}"
        );
        assert!((value.im * scale * scale).abs() < 2e-14, "{value:?}");
    }

    #[test]
    fn negative_width_contours_match_independent_parameter_integral() {
        // Above the zero-width two-particle threshold. The common negative
        // width keeps Im F=-5 uniformly away from the singular integration
        // surface. The original OneLOop-3.4 reference returns zero for one
        // pinched box, so the oracle is direct parameter integration instead;
        // all failing reference outputs are retained in contours.json.
        let plan = Pentagon6Plan::new().unwrap();
        let mut evaluator = plan.instantiate(0.).unwrap();
        let point: Vec<_> = [16, -1, -1, -1, 15, 15, 15, -2, -2, -2, 1, 2, 3, 4, 5, 1]
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                Complex::new(value as f64, if (10..15).contains(&i) { -5. } else { 0. })
            })
            .collect();
        let mut options = Pentagon6Options::new(&0.);
        options.relative_tolerance = 1e-11;
        for slope in [1., 2.] {
            options.contour_slope = slope;
            let result = evaluator.evaluate_batch(&point, 1, &options).unwrap();
            let value = &result.lane(0).unwrap()[0];
            assert!(
                (value.re - 0.00115669079120771305).abs() < 1e-12,
                "slope={slope}: {value:?}"
            );
            assert!(
                (value.im + 0.000473960524532898804).abs() < 1e-12,
                "slope={slope}: {value:?}"
            );
        }
    }
}
