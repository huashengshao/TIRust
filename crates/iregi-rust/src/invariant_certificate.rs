//! Exact rank certificates for the invariant-only high-point scalar interface.
//!
//! This module checks
//! the exact represented input, without projecting an almost-compatible Gram
//! matrix. Symbolica prepares all determinants, Taylor coefficients and batched
//! arithmetic. Chart bookkeeping and numeric conversion are the only local
//! operations. See `docs/research/exact-invariant-certificate.md`.

use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        dual::{DualNumberStructure, HyperDual},
        float::{Complex, ComplexBall, Float, SingleFloat},
        rational::{Q, Rational},
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, ExpressionEvaluator},
    poly::PolyVariable,
    tensors::matrix::Matrix,
};

use crate::ReductionError;

/// Convert the represented value exactly, rather than its decimal rendering.
/// The distinction matters for compatibility residuals smaller than one ulp.
pub trait ExactInvariantScalar {
    /// Return the exact represented value, or `None` for a nonfinite value.
    /// Implementations must not round, truncate, or pass through binary64.
    fn exact_invariant_value(&self) -> Option<Rational>;
}

// NativeFloat already requires lossless represented-value conversion. This
// bridge adds no weaker decimal/binary64 path, and the public trait remains
// independently implementable by exact scalar types that are not NativeFloat.
impl<T: iregi_masters::NativeFloat> ExactInvariantScalar for T {
    fn exact_invariant_value(&self) -> Option<Rational> {
        iregi_masters::NativeFloat::exact_rational(self)
    }
}

#[derive(Clone, Debug)]
struct Chart {
    /// Relative vectors are labelled by their original propagator indices.
    basis: [usize; 4],
    determinant: usize,
    residuals: Vec<((usize, usize), usize)>,
}

/// Exact compatibility with a regular rank-four invariant chart.
///
/// A certificate permits complex external-vector analytic continuation. It
/// does not impose the inertia of a Gram matrix of real Minkowski vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum InvariantCertificate {
    Compatible {
        basis: [usize; 4],
    },
    Incompatible {
        basis: [usize; 4],
        pair: (usize, usize),
        component: usize,
        /// A certified enclosure that excludes zero, not a rounded residual.
        residual: ComplexBall,
    },
    /// No principal chart has an invertible primal. This includes legitimate
    /// lower-rank inputs and requires a separate analytic profile.
    NoRegularChart,
    /// The exact predicate was unresolved within the certificate budget.
    Unresolved {
        working_bits: u32,
    },
}

/// Immutable polynomial instructions, shared across numeric precisions.
#[derive(Clone)]
pub(crate) struct InvariantCertificatePlan {
    arguments: usize,
    components: usize,
    charts: Vec<Chart>,
    exact: ExpressionEvaluator<Complex<Rational>>,
}

impl InvariantCertificatePlan {
    /// Reuse the exact polynomial execution engine for constraints unrelated
    /// to Gram rank. The unit chart has no kinematic precondition.
    pub(crate) fn from_zero_polynomials(
        parameters: &[Atom],
        residuals: &[Atom],
        shape: Option<Vec<Vec<usize>>>,
    ) -> Result<Self, ReductionError> {
        let mut symbols = std::collections::HashSet::new();
        for parameter in parameters {
            let symbolica::atom::AtomView::Var(variable) = parameter.as_view() else {
                return Err(preparation(
                    "exact polynomial certificate parameters must be symbols",
                ));
            };
            if !symbols.insert(variable.get_symbol()) {
                return Err(preparation(
                    "exact polynomial certificate parameters must be unique",
                ));
            }
        }
        let mut expressions = Vec::with_capacity(residuals.len() + 1);
        expressions.push(Atom::num(1));
        for residual in residuals {
            // Symbolica converts non-polynomial pieces into extra variables.
            // Inspect those variables, then compile its polynomial expression,
            // so IF/function/negative-power semantics cannot enter ball proofs.
            let polynomial = residual
                // Discover only used polynomial variables; compilation below still
                // binds every declared parameter and the original Taylor shape.
                .try_to_polynomial::<_, u32>(&Q, None)
                .map_err(preparation)?;
            if polynomial.variables().iter().any(|variable| {
                !matches!(variable, PolyVariable::Symbol(symbol) if symbols.contains(symbol))
            }) {
                return Err(preparation(
                    "exact zero certificates require rational polynomials in the supplied symbols; functions, non-polynomial powers, and unknown symbols are unsupported",
                ));
            }
            expressions.push(polynomial.to_expression());
        }
        let exact = compile_polynomials(&expressions, parameters, Vec::new(), &shape)?;
        Ok(Self {
            arguments: parameters.len(),
            components: shape.as_ref().map_or(1, Vec::len),
            charts: vec![Chart {
                basis: [0; 4],
                determinant: 0,
                residuals: (0..residuals.len()).map(|i| ((i, 0), i + 1)).collect(),
            }],
            exact,
        })
    }

    pub(crate) fn new(propagators: usize) -> Result<Self, ReductionError> {
        Self::build(propagators, None)
    }

    pub(crate) fn new_dual<D: DualNumberStructure>(
        propagators: usize,
        dual: &D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        Self::build(
            propagators,
            Some(
                dual.get_shape()
                    .into_iter()
                    .map(<[usize]>::to_vec)
                    .collect(),
            ),
        )
    }

    fn build(propagators: usize, shape: Option<Vec<Vec<usize>>>) -> Result<Self, ReductionError> {
        if !(6..=7).contains(&propagators) {
            return Err(ReductionError::Unsupported(
                "the exact invariant certificate currently covers six and seven propagators".into(),
            ));
        }
        let arguments = propagators * (propagators + 1) / 2;
        let parameters: Vec<_> = (0..arguments).map(|i| variable(&format!("g{i}"))).collect();
        let mut aliases = Vec::new();
        let mut gram = vec![vec![Atom::Zero; propagators]; propagators];
        for i in 1..propagators {
            for j in i..propagators {
                let value = variable(&format!("k{i}_{j}"));
                aliases.push((
                    value.clone(),
                    &parameters[packed_index(propagators, i, j)]
                        - &parameters[packed_index(propagators, 0, i)]
                        - &parameters[packed_index(propagators, 0, j)]
                        + &parameters[0],
                ));
                gram[i][j] = value.clone();
                gram[j][i] = value;
            }
        }
        let mut expressions = Vec::new();
        let mut charts = Vec::new();
        for a in 1..propagators {
            for b in a + 1..propagators {
                for c in b + 1..propagators {
                    for d in c + 1..propagators {
                        let basis = [a, b, c, d];
                        let determinant = expressions.len();
                        expressions.push(minor(&gram, &basis, &basis)?);
                        let outside: Vec<_> =
                            (1..propagators).filter(|i| !basis.contains(i)).collect();
                        let mut residuals = Vec::new();
                        for (position, &u) in outside.iter().enumerate() {
                            for &v in &outside[position..] {
                                let rows = [a, b, c, d, u];
                                let columns = [a, b, c, d, v];
                                residuals.push(((u, v), expressions.len()));
                                expressions.push(minor(&gram, &rows, &columns)?);
                            }
                        }
                        charts.push(Chart {
                            basis,
                            determinant,
                            residuals,
                        });
                    }
                }
            }
        }
        let exact = compile_polynomials(&expressions, &parameters, aliases, &shape)?;
        let components = shape.as_ref().map_or(1, Vec::len);
        Ok(Self {
            arguments,
            components,
            charts,
            exact,
        })
    }

    pub(crate) fn instantiate(&self) -> InvariantCertificateWorkspace {
        InvariantCertificateWorkspace {
            arguments: self.arguments,
            components: self.components,
            charts: self.charts.clone(),
            exact: self.exact.clone(),
            evaluators: Vec::new(),
            initial_bits: 128,
            maximum_bits: 16384,
            evaluated_rows: 0,
            passes: 0,
            input: Vec::new(),
            output: Vec::new(),
        }
    }
}

fn compile_polynomials(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: Vec<(Atom, Atom)>,
    shape: &Option<Vec<Vec<usize>>>,
) -> Result<ExpressionEvaluator<Complex<Rational>>, ReductionError> {
    let mut exact: ExpressionEvaluator<Complex<Rational>> =
        Atom::evaluator_multiple(expressions, parameters)
            .add_aliases(aliases)
            .map_err(preparation)?
            .direct_translation(true)
            .horner_iterations(0)
            .cores(1)
            .build()
            .map_err(preparation)?;
    if let Some(shape) = shape {
        let dual = HyperDual::<Complex<Rational>>::new(shape.clone());
        exact = exact
            .vectorize(&Dualizer::new(dual, Vec::new()))
            .map_err(preparation)?;
    }
    Ok(exact)
}

/// The input layout is lane, packed upper-triangular G argument, Taylor
/// component. Certified polynomial instructions run across active sample lanes;
/// only exact zero enclosures can certify an equality.
#[derive(Clone)]
pub(crate) struct InvariantCertificateWorkspace {
    arguments: usize,
    components: usize,
    charts: Vec<Chart>,
    exact: ExpressionEvaluator<Complex<Rational>>,
    evaluators: Vec<(u32, BatchExpressionEvaluator<ComplexBall>)>,
    initial_bits: u32,
    maximum_bits: u32,
    evaluated_rows: usize,
    passes: usize,
    input: Vec<ComplexBall>,
    output: Vec<ComplexBall>,
}

impl InvariantCertificateWorkspace {
    pub(crate) fn set_precision_budget(
        &mut self,
        initial_bits: u32,
        maximum_bits: u32,
    ) -> Result<(), ReductionError> {
        if initial_bits < 2 || maximum_bits < initial_bits {
            return Err(ReductionError::InvalidInput(
                "certificate precision must satisfy 2 <= initial <= maximum".into(),
            ));
        }
        self.initial_bits = initial_bits;
        self.maximum_bits = maximum_bits;
        Ok(())
    }

    /// Actual active lanes evaluated, including higher-precision retries.
    pub(crate) fn evaluated_rows(&self) -> usize {
        self.evaluated_rows
    }

    pub(crate) fn passes(&self) -> usize {
        self.passes
    }

    pub(crate) fn check_batch(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
    ) -> Result<Vec<InvariantCertificate>, ReductionError> {
        self.check_batch_inner(input, rows, None)
    }

    /// Only unit-chart polynomial plans may select residual components. Gram
    /// compatibility always checks every required component of every minor.
    pub(crate) fn check_selected_batch(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
        selected: &[Vec<(usize, usize)>],
    ) -> Result<Vec<InvariantCertificate>, ReductionError> {
        self.check_batch_inner(input, rows, Some(selected))
    }

    fn check_batch_inner(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
        selected: Option<&[Vec<(usize, usize)>]>,
    ) -> Result<Vec<InvariantCertificate>, ReductionError> {
        self.evaluated_rows = 0;
        self.passes = 0;
        if let Some(selected) = selected {
            if selected.len() != rows
                || self.charts.len() != 1
                || self.charts[0].basis != [0; 4]
                || selected.iter().flatten().any(|&(constraint, component)| {
                    constraint >= self.charts[0].residuals.len() || component >= self.components
                })
            {
                return Err(ReductionError::InvalidInput(
                    "invalid selected polynomial certificate components or non-unit chart".into(),
                ));
            }
        }
        let expected = self
            .arguments
            .checked_mul(self.components)
            .and_then(|width| width.checked_mul(rows))
            .ok_or_else(size_error)?;
        if input.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "invariant certificate expects {expected} values, received {}",
                input.len(),
            )));
        }
        if rows == 0 {
            self.output.clear();
            return Ok(Vec::new());
        }
        let input_width = self.arguments * self.components;
        let mut result = vec![
            InvariantCertificate::Unresolved {
                working_bits: self.initial_bits
            };
            rows
        ];
        let mut active: Vec<_> = (0..rows)
            .filter(|&lane| {
                if selected.is_some_and(|selected| selected[lane].is_empty()) {
                    result[lane] = InvariantCertificate::Compatible { basis: [0; 4] };
                    false
                } else {
                    true
                }
            })
            .collect();
        if active.is_empty() {
            self.output.clear();
            return Ok(result);
        }
        let mut bits = self.initial_bits;
        let radius = Rational::from(0);
        loop {
            let index =
                if let Some(index) = self.evaluators.iter().position(|entry| entry.0 == bits) {
                    index
                } else {
                    let evaluator = self
                        .exact
                        .clone()
                        .map_coeff_with_prec(
                            &|value| ComplexBall::from_rational_ball(value, &radius, bits),
                            bits,
                        )
                        .into_batch_evaluator();
                    self.evaluators.push((bits, evaluator));
                    self.evaluators.len() - 1
                };
            let evaluator = &mut self.evaluators[index].1;
            let width = evaluator.get_output_len();
            let output_len = width.checked_mul(active.len()).ok_or_else(size_error)?;
            self.input.clear();
            for &lane in &active {
                self.input.extend(
                    input[lane * input_width..(lane + 1) * input_width]
                        .iter()
                        .map(|value| ComplexBall::from_rational_ball(value, &radius, bits)),
                );
            }
            let zero = ComplexBall::from_rational_ball(
                &Complex::new(Rational::from(0), Rational::from(0)),
                &radius,
                bits,
            );
            self.output.resize(output_len, zero);
            evaluator
                .evaluate_batch(&self.input, &mut self.output, active.len())
                .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
            self.evaluated_rows = self
                .evaluated_rows
                .checked_add(active.len())
                .ok_or_else(size_error)?;
            self.passes += 1;
            let mut unresolved = Vec::new();
            for (&lane, point) in active.iter().zip(self.output.chunks_exact(width)) {
                result[lane] = inspect(
                    point,
                    &self.charts,
                    self.components,
                    bits,
                    selected.map(|selected| selected[lane].as_slice()),
                );
                if matches!(result[lane], InvariantCertificate::Unresolved { .. }) {
                    unresolved.push(lane);
                }
            }
            if unresolved.is_empty() || bits == self.maximum_bits {
                return Ok(result);
            }
            bits = bits.saturating_mul(2).min(self.maximum_bits);
            active = unresolved;
        }
    }
}

fn inspect(
    point: &[ComplexBall],
    charts: &[Chart],
    components: usize,
    bits: u32,
    selected: Option<&[(usize, usize)]>,
) -> InvariantCertificate {
    let mut all_singular = true;
    for chart in charts {
        let determinant = &point[chart.determinant * components];
        if exact_zero(determinant) {
            continue;
        }
        all_singular = false;
        if !finite_ball(determinant) || determinant.contains_zero() {
            continue;
        }
        let mut all_zero = true;
        for &(pair, index) in &chart.residuals {
            for (component, value) in point[index * components..(index + 1) * components]
                .iter()
                .enumerate()
            {
                if selected.is_some_and(|selected| !selected.contains(&(pair.0, component))) {
                    continue;
                }
                if exact_zero(value) {
                    continue;
                }
                all_zero = false;
                if finite_ball(value) && !value.contains_zero() {
                    return InvariantCertificate::Incompatible {
                        basis: chart.basis,
                        pair,
                        component,
                        residual: value.clone(),
                    };
                }
            }
        }
        if all_zero {
            return InvariantCertificate::Compatible { basis: chart.basis };
        }
    }
    if all_singular {
        InvariantCertificate::NoRegularChart
    } else {
        InvariantCertificate::Unresolved { working_bits: bits }
    }
}

fn exact_zero(value: &ComplexBall) -> bool {
    value.re.center.is_zero()
        && value.re.radius.is_zero()
        && value.im.center.is_zero()
        && value.im.radius.is_zero()
}

fn finite_ball(value: &ComplexBall) -> bool {
    value.re.center.is_finite()
        && value.re.radius.is_finite()
        && value.im.center.is_finite()
        && value.im.radius.is_finite()
}

fn minor(gram: &[Vec<Atom>], rows: &[usize], columns: &[usize]) -> Result<Atom, ReductionError> {
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: Some(Box::new(|value, output| {
            *output = value.together();
            true
        })),
    };
    Matrix::from_nested_vec(
        rows.iter()
            .map(|&row| {
                columns
                    .iter()
                    .map(|&column| gram[row][column].clone())
                    .collect()
            })
            .collect(),
        field,
    )
    .map_err(preparation)?
    .det()
    .map(|value| value.together().expand())
    .map_err(|error| preparation(format!("Gram minor: {error:?}")))
}

/// Linear coordinates for the existing scalar denominator-relation engine.
///
/// These are the four covector projections `K_ai`, not physical time/spatial
/// components. Only a successful full-Taylor certificate for this exact chart
/// permits using their linear dependencies. The scalar compiler must retain
/// the original Cayley matrix when constructing denominator constants.
pub(crate) fn certified_relation_coordinates(
    propagators: usize,
    packed_gram: &[Atom],
    basis: [usize; 4],
) -> Result<Vec<[Atom; 4]>, ReductionError> {
    if !(6..=7).contains(&propagators)
        || packed_gram.len() != propagators * (propagators + 1) / 2
        || basis.iter().any(|&i| i == 0 || i >= propagators)
        || basis
            .iter()
            .enumerate()
            .any(|(i, value)| basis[..i].contains(value))
    {
        return Err(ReductionError::InvalidInput(
            "invalid certified invariant relation chart".into(),
        ));
    }
    Ok((0..propagators)
        .map(|i| {
            std::array::from_fn(|column| {
                if i == 0 {
                    Atom::Zero
                } else {
                    let a = basis[column];
                    &packed_gram[packed_index(propagators, a, i)]
                        - &packed_gram[packed_index(propagators, 0, a)]
                        - &packed_gram[packed_index(propagators, 0, i)]
                        + &packed_gram[0]
                }
            })
        })
        .collect())
}

fn packed_index(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    i * (2 * n - i + 1) / 2 + j - i
}

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!(
        "iregi_invariant_certificate::{name}"
    )))
}

fn preparation(error: impl ToString) -> ReductionError {
    ReductionError::Preparation(error.to_string())
}

fn size_error() -> ReductionError {
    ReductionError::InvalidInput("invariant-certificate batch size overflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::float::DoubleFloat;
    fn real(value: i64) -> Complex<Rational> {
        Complex::new(Rational::from(value), Rational::from(0))
    }

    // q = [0,e0,e1,e2,e3,(1,1,1,1),(2,3,4,5)], metric (+---).
    fn seven_point_gram() -> Vec<Complex<Rational>> {
        [
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, -1, 0, 0, -1, -3, -1, 0, -1, -4, -1, -1, -5, -2,
            -10, -46,
        ]
        .into_iter()
        .map(real)
        .collect()
    }

    #[test]
    fn exact_gram_certificate_isolates_lanes_and_uses_alternative_charts() {
        let plan = InvariantCertificatePlan::new(7).unwrap();
        let mut workspace = plan.instantiate();
        let good = seven_point_gram();
        let mut bad = good.clone();
        bad[packed_index(7, 5, 5)] += real(1);
        let relabel = [0, 0, 1, 2, 3, 4, 5];
        let mut alternative = Vec::new();
        for i in 0..7 {
            for j in i..7 {
                alternative.push(good[packed_index(7, relabel[i], relabel[j])].clone());
            }
        }
        let input = [good.clone(), bad, alternative, good.clone(), good].concat();
        let result = workspace.check_batch(&input, 5).unwrap();
        assert!(matches!(result[0], InvariantCertificate::Compatible { .. }));
        assert!(matches!(
            result[1],
            InvariantCertificate::Incompatible { component: 0, .. }
        ));
        assert_eq!(
            result[2],
            InvariantCertificate::Compatible {
                basis: [2, 3, 4, 5]
            }
        );
        assert_eq!(result[0], result[3]);
        assert_eq!(result[0], result[4]);
        assert!(workspace.check_batch(&[], 0).unwrap().is_empty());
        assert!(workspace.check_batch(&[], usize::MAX).is_err());
    }

    #[test]
    fn certificate_checks_every_mixed_taylor_component() {
        let shape = HyperDual::<Complex<Rational>>::new(vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![1, 1],
            vec![2, 0],
        ]);
        let plan = InvariantCertificatePlan::new_dual(7, &shape).unwrap();
        let mut workspace = plan.instantiate();
        // q(t,u)=(1+t+u)q; coefficients are those of (1+t+u)^2.
        let good: Vec<_> = seven_point_gram()
            .iter()
            .flat_map(|value| [1, 2, 2, 2, 1].map(|weight| value * real(weight)))
            .collect();
        let mut bad = good.clone();
        bad[packed_index(7, 5, 6) * 5 + 3] += real(1);
        let result = workspace.check_batch(&[good, bad].concat(), 2).unwrap();
        assert!(matches!(result[0], InvariantCertificate::Compatible { .. }));
        assert!(matches!(
            result[1],
            InvariantCertificate::Incompatible { component: 3, .. }
        ));
    }

    #[test]
    fn scalar_certificate_preserves_translation_and_diagnoses_rank_loss() {
        let plan = InvariantCertificatePlan::new(7).unwrap();
        let mut workspace = plan.instantiate();
        let mut translated = seven_point_gram();
        for i in 0..7 {
            for j in i..7 {
                translated[packed_index(7, i, j)] += real((i + j + 11) as i64);
            }
        }
        assert!(matches!(
            workspace.check_batch(&translated, 1).unwrap()[0],
            InvariantCertificate::Compatible { .. }
        ));
        assert_eq!(
            workspace.check_batch(&vec![real(0); 28], 1).unwrap(),
            [InvariantCertificate::NoRegularChart]
        );
    }

    #[test]
    fn absent_regular_chart_does_not_claim_lower_rank() {
        let plan = InvariantCertificatePlan::new(6).unwrap();
        let mut workspace = plan.instantiate();
        let mut input = vec![real(0); 21];
        // K=-I_5+J_5/4 has eigenvalues -1,-1,-1,-1,1/4, but every
        // principal four-dimensional submatrix has a zero eigenvalue.
        // Absence of a regular chart does not certify rank at most four.
        for i in 1..6 {
            for j in i..6 {
                input[packed_index(6, i, j)] = Complex::new(
                    Rational::from((if i == j { -3 } else { 1 }, 4)),
                    Rational::from(0),
                );
            }
        }
        assert_eq!(
            workspace.check_batch(&input, 1).unwrap(),
            [InvariantCertificate::NoRegularChart]
        );
    }

    #[test]
    fn numeric_conversion_retains_bits_beyond_binary64() {
        let perturbation = Rational::from((1, 2)).pow(100);
        let exact = Rational::from(1) + &perturbation;
        let arbitrary = Float::new(128).from_rational(&exact);
        let double = DoubleFloat::from_compensated_sum(1.0, 2_f64.powi(-100));
        assert_eq!(arbitrary.exact_invariant_value(), Some(exact.clone()));
        assert_eq!(double.exact_invariant_value(), Some(exact));
        assert_eq!(1_f64.exact_invariant_value(), Some(Rational::from(1)));
        assert!(f64::INFINITY.exact_invariant_value().is_none());
        let sparse = DoubleFloat::from_compensated_sum(1.0, 2_f64.powi(-300));
        assert_eq!(
            sparse.exact_invariant_value(),
            Some(Rational::from(1) + Rational::from((1, 2)).pow(300))
        );
        assert_eq!(
            f64::from_bits(1).exact_invariant_value(),
            Some(Rational::from((1, 2)).pow(1074))
        );
    }

    #[test]
    fn tiny_nonzero_residual_is_never_accepted_by_a_roundoff_tolerance() {
        let plan = InvariantCertificatePlan::new(7).unwrap();
        let mut workspace = plan.instantiate();
        let good = seven_point_gram();
        let mut bad = good.clone();
        bad[packed_index(7, 5, 5)].re += Rational::from((1, 2)).pow(300);
        workspace.set_precision_budget(128, 128).unwrap();
        assert_eq!(
            workspace.check_batch(&bad, 1).unwrap(),
            [InvariantCertificate::Unresolved { working_bits: 128 }]
        );
        workspace.set_precision_budget(128, 1024).unwrap();
        let result = workspace.check_batch(&[good, bad].concat(), 2).unwrap();
        assert!(matches!(result[0], InvariantCertificate::Compatible { .. }));
        assert!(matches!(
            result[1],
            InvariantCertificate::Incompatible { .. }
        ));
        assert!(workspace.passes() > 1);
        assert!(workspace.evaluated_rows() > 2);
    }

    #[test]
    fn six_point_scalar_continuation_needs_no_real_minkowski_signature() {
        let plan = InvariantCertificatePlan::new(6).unwrap();
        let mut workspace = plan.instantiate();
        // Positive semidefinite Euclidean rank-four Gram: q=[0,e0,e1,e2,e3,1].
        let input: Vec<_> = [
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 4,
        ]
        .into_iter()
        .map(real)
        .collect();
        assert!(matches!(
            workspace.check_batch(&input, 1).unwrap()[0],
            InvariantCertificate::Compatible { .. }
        ));
    }

    #[test]
    fn certified_covectors_keep_linear_relations_after_translation() {
        let original: Vec<_> = seven_point_gram()
            .iter()
            .map(|value| Atom::num(value.re.clone()))
            .collect();
        let mut translated = original.clone();
        for i in 0..7 {
            for j in i..7 {
                translated[packed_index(7, i, j)] += Atom::num((i + j + 11) as i64);
            }
        }
        let coordinates = certified_relation_coordinates(7, &original, [1, 2, 3, 4]).unwrap();
        let shifted = certified_relation_coordinates(7, &translated, [1, 2, 3, 4]).unwrap();
        assert_eq!(coordinates, shifted);
        for column in 0..4 {
            let residual =
                &coordinates[5][column] - (1..5).map(|i| &coordinates[i][column]).sum::<Atom>();
            assert!(residual.is_zero());
            let residual = &coordinates[6][column]
                - (1..5)
                    .map(|i| Atom::num((i + 1) as i64) * &coordinates[i][column])
                    .sum::<Atom>();
            assert!(residual.is_zero());
        }
        assert!(certified_relation_coordinates(7, &original, [1, 1, 3, 4]).is_err());
        assert!(certified_relation_coordinates(7, &original, [0, 2, 3, 4]).is_err());
    }

    #[test]
    fn sparse_zero_polynomials_preserve_wide_ordered_mixed_certificate_layout() {
        let x = Atom::var(symbolica::symbol!("iregi_sparse_zero_width::x"));
        let y = Atom::var(symbolica::symbol!("iregi_sparse_zero_width::y"));
        let small_parameters = [x.clone(), y.clone()];
        let mut wide_parameters = (0..21_202)
            .map(|i| {
                Atom::var(symbolica::symbol!(format!(
                    "iregi_sparse_zero_width::imported_{i}"
                )))
            })
            .collect::<Vec<_>>();
        let y_position = 7;
        let x_position = 21_190;
        wide_parameters.insert(y_position, y.clone());
        wide_parameters.insert(x_position, x.clone());
        assert_eq!(wide_parameters.len(), 21_204);
        let residuals = [&x - &y, &x * &x - &y * &y, Atom::num(0)];
        // Deliberately ordered primal,u,t,tu, not lexicographic derivative order.
        let shape = Some(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
        let small = InvariantCertificatePlan::from_zero_polynomials(
            &small_parameters,
            &residuals,
            shape.clone(),
        )
        .unwrap();
        let wide =
            InvariantCertificatePlan::from_zero_polynomials(&wide_parameters, &residuals, shape)
                .unwrap();
        assert_eq!(wide.arguments, wide_parameters.len());
        assert_eq!(wide.components, 4);
        let mut small_input = Vec::new();
        let mut wide_input = Vec::new();
        for lane in 0..3 {
            let mut x_jet = [1, 2, 3, 4].map(real);
            let y_jet = [1, 2, 3, 4].map(real);
            if lane == 1 {
                x_jet[3].im += Rational::from(1);
            }
            if lane == 2 {
                x_jet[0].re += Rational::from(1);
            }
            small_input.extend(x_jet.iter().cloned());
            small_input.extend(y_jet.iter().cloned());
            for position in 0..wide_parameters.len() {
                if position == x_position {
                    wide_input.extend(x_jet.iter().cloned());
                } else if position == y_position {
                    wide_input.extend(y_jet.iter().cloned());
                } else {
                    wide_input.extend((0..4).map(|component| real(17 + lane * 3 + component)));
                }
            }
        }
        let mut small_workspace = small.instantiate();
        let mut wide_workspace = wide.instantiate();
        small_workspace.set_precision_budget(128, 128).unwrap();
        wide_workspace.set_precision_budget(128, 128).unwrap();
        let small_result = small_workspace.check_batch(&small_input, 3).unwrap();
        let wide_result = wide_workspace.check_batch(&wide_input, 3).unwrap();
        assert_eq!(small_result, wide_result);
        assert!(matches!(
            wide_result[0],
            InvariantCertificate::Compatible { .. }
        ));
        assert!(matches!(
            wide_result[1],
            InvariantCertificate::Incompatible {
                pair: (0, 0),
                component: 3,
                ..
            }
        ));
        assert!(matches!(
            wide_result[2],
            InvariantCertificate::Incompatible {
                pair: (0, 0),
                component: 0,
                ..
            }
        ));
        // A primal-zero residual with a nonzero transverse coefficient remains
        // compatible only for the exact selected components, never all jets.
        let selected = vec![vec![], vec![(0, 0), (0, 1), (0, 2)], vec![(2, 3)]];
        assert_eq!(
            small_workspace
                .check_selected_batch(&small_input, 3, &selected)
                .unwrap(),
            wide_workspace
                .check_selected_batch(&wide_input, 3, &selected)
                .unwrap()
        );
        assert!(
            wide_workspace
                .check_selected_batch(&wide_input, 3, &selected)
                .unwrap()
                .iter()
                .all(|r| matches!(r, InvariantCertificate::Compatible { .. }))
        );
        assert!(wide_workspace.check_batch(&[], 0).unwrap().is_empty());
        // Declaration validation is retained independently of polynomial use.
        assert!(
            InvariantCertificatePlan::from_zero_polynomials(
                &[x.clone(), x.clone()],
                &[Atom::num(0)],
                None
            )
            .is_err()
        );
        assert!(
            InvariantCertificatePlan::from_zero_polynomials(&[&x + &y], &[Atom::num(0)], None)
                .is_err()
        );
    }
}
