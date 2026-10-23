//! Propagate scalar-master quadrature estimates through all tensor jets.
//!
//! Symbolica extracts signed affine coefficients in the E6 masters. The
//! executor Dualizes those coefficients, takes their numerical complex norms,
//! and evaluates a second Symbolica-Dualized positive sum of products with
//! the per-component master errors. Thus Symbolica owns every Taylor product
//! and multiplicity; ABS is never differentiated into a putative error bound.

use std::collections::{BTreeSet, HashMap};

use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{atom::AtomField, float::Complex},
};

use crate::{ibp::MasterLeafKind, plan::ReductionPlan, types::ReductionError};

use super::{
    GuardContext, OperandRoundoff, StabilityPlan, collect_amplitude_variables,
    depends_on_amplitudes, dyadic, if_parts, signed_power,
};

#[derive(Clone, Debug)]
struct OutputCheck {
    output: usize,
    dimension: i32,
    primal_amplitude: usize,
    label: String,
}

/// Exact coefficients and positive Taylor products for the E6 error estimates.
/// Both expression lists must use the same ordered Dual layout as reduction.
#[derive(Clone, Debug)]
pub(crate) struct QuadratureStabilityPlan {
    leaf_indices: Vec<usize>,
    coefficient_parameters: Vec<Atom>,
    coefficient_expressions: Vec<Atom>,
    coefficient_aliases: Vec<(Atom, Atom)>,
    contraction_parameters: Vec<Atom>,
    contraction_expressions: Vec<Atom>,
    checks: Vec<OutputCheck>,
    moment_outputs: usize,
}

impl QuadratureStabilityPlan {
    pub(crate) fn prepare(
        plan: &ReductionPlan,
        stability: &StabilityPlan,
    ) -> Result<Option<Self>, ReductionError> {
        let leaf_indices: Vec<_> = plan
            .leaves
            .iter()
            .enumerate()
            .filter_map(|(index, leaf)| {
                (leaf.kind == MasterLeafKind::SixDimensionalPentagon).then_some(index)
            })
            .collect();
        if leaf_indices.is_empty() {
            return Ok(None);
        }
        let count = leaf_indices.len();
        let mut projections = HashMap::new();
        for (index, leaf) in plan.leaves.iter().enumerate() {
            for (pole, slot) in leaf.slots.iter().enumerate() {
                let mut projected = vec![Atom::new(); count];
                if pole == 3 {
                    if let Some(component) = leaf_indices.iter().position(|&leaf| leaf == index) {
                        projected[component] = Atom::num(1);
                    }
                }
                projections.insert(slot.clone(), projected);
            }
        }
        let mut projected_aliases = Vec::new();
        for (alias, body) in &plan.aliases {
            if !depends_on_projection(body, &projections) {
                continue;
            }
            let mut projected = affine_projection(body, &projections, count)?;
            for (leaf, expression) in projected.iter_mut().enumerate() {
                if expression.is_zero() {
                    continue;
                }
                let handle = variable(&format!("coefficient_{leaf}_{}", projected_aliases.len()));
                projected_aliases.push((handle.clone(), expression.clone()));
                *expression = handle;
            }
            projections.insert(alias.clone(), projected);
        }
        let errors: Vec<_> = (0..count)
            .map(|leaf| variable(&format!("error_{leaf}")))
            .collect();
        let base_dimension = 4 + 2 * i32::from(plan.settings.dimension_shift)
            - 2 * plan
                .settings
                .powers
                .iter()
                .map(|&power| i32::from(power))
                .sum::<i32>();
        let mut result = Self {
            leaf_indices,
            coefficient_parameters: plan.parameters.clone(),
            coefficient_expressions: Vec::new(),
            coefficient_aliases: Vec::new(),
            contraction_parameters: Vec::new(),
            contraction_expressions: Vec::new(),
            checks: Vec::new(),
            moment_outputs: plan.moments.len() * 4,
        };
        for (moment, (basis, slots)) in plan.basis.iter().zip(&plan.moments).enumerate() {
            for (pole, expression) in slots.iter().enumerate() {
                let projected = affine_projection(expression, &projections, count)?;
                if projected.iter().all(AtomCore::is_zero) {
                    continue;
                }
                let primal_amplitude = stability
                    .primal_checks
                    .iter()
                    .find(|check| stability.primal_expressions[check.value] == *expression)
                    .map(|check| check.amplitude)
                    .ok_or_else(|| {
                        ReductionError::Preparation(
                            "E6-dependent output is missing its primal stability amplitude".into(),
                        )
                    })?;
                let mut contraction = Atom::new();
                for (leaf, coefficient) in projected.into_iter().enumerate() {
                    if coefficient.is_zero() {
                        continue;
                    }
                    let argument = variable(&format!(
                        "magnitude_{}",
                        result.coefficient_expressions.len()
                    ));
                    result.coefficient_expressions.push(coefficient);
                    result.contraction_parameters.push(argument.clone());
                    // Backend errors use the complex infinity norm. A factor
                    // of two bounds its conversion to a complex Euclidean
                    // norm without an inexact sqrt(2) preparation constant.
                    contraction += Atom::num(2) * argument * &errors[leaf];
                }
                result.contraction_expressions.push(contraction);
                result.checks.push(OutputCheck {
                    output: moment * 4 + pole,
                    dimension: base_dimension + basis.rank() as i32,
                    primal_amplitude,
                    label: format!("moment {:?}, Laurent slot {pole}", basis.0),
                });
            }
        }
        if result.checks.is_empty() {
            return Ok(None);
        }
        result.contraction_parameters.extend(errors);
        let all_aliases: Vec<_> = plan
            .aliases
            .iter()
            .cloned()
            .chain(projected_aliases)
            .collect();
        let definitions: HashMap<_, _> = all_aliases.iter().cloned().collect();
        let mut reachable = BTreeSet::new();
        let mut pending = result.coefficient_expressions.clone();
        while let Some(expression) = pending.pop() {
            for symbol in expression.get_all_symbols(false) {
                let alias = Atom::var(symbol);
                if let Some(body) = definitions.get(&alias) {
                    if reachable.insert(alias) {
                        pending.push(body.clone());
                    }
                } else if projections.contains_key(&alias) {
                    return Err(ReductionError::Preparation(
                        "E6 error coefficient still depends on a scalar master".into(),
                    ));
                }
            }
        }
        result.coefficient_aliases = all_aliases
            .into_iter()
            .filter(|(alias, _)| reachable.contains(alias))
            .collect();
        Ok(Some(result))
    }

    pub(crate) fn leaf_indices(&self) -> &[usize] {
        &self.leaf_indices
    }
    pub(crate) fn coefficient_parameters(&self) -> &[Atom] {
        &self.coefficient_parameters
    }
    pub(crate) fn coefficient_expressions(&self) -> &[Atom] {
        &self.coefficient_expressions
    }
    pub(crate) fn coefficient_aliases(&self) -> &[(Atom, Atom)] {
        &self.coefficient_aliases
    }
    pub(crate) fn contraction_parameters(&self) -> &[Atom] {
        &self.contraction_parameters
    }
    pub(crate) fn contraction_expressions(&self) -> &[Atom] {
        &self.contraction_expressions
    }
}

impl StabilityPlan {
    /// Check propagated quadrature estimates for every requested output jet.
    /// The estimates are successive-rule differences, not interval bounds.
    /// For primals, include the existing working-ULP master-sensitivity estimate
    /// in the same budget rather than granting quadrature an additional budget.
    pub(crate) fn inspect_quadrature_lane<T: NativeFloat>(
        &self,
        quadrature: &QuadratureStabilityPlan,
        errors: &[Complex<T>],
        moments: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        primal: Option<&[Complex<T>]>,
    ) -> Option<String> {
        self.inspect_quadrature_lane_impl::<T, false>(
            quadrature,
            errors,
            moments,
            input,
            components,
            prototype,
            accuracy_bits,
            primal,
            None,
        )
    }

    pub(crate) fn inspect_quadrature_lane_context<T: NativeFloat>(
        &self,
        quadrature: &QuadratureStabilityPlan,
        errors: &[Complex<T>],
        moments: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        primal: Option<&[Complex<T>]>,
        context: &mut GuardContext<T>,
    ) -> Option<String> {
        self.inspect_quadrature_lane_impl::<T, true>(
            quadrature,
            errors,
            moments,
            input,
            components,
            prototype,
            accuracy_bits,
            primal,
            Some(context),
        )
    }

    fn inspect_quadrature_lane_impl<T: NativeFloat, const CACHED: bool>(
        &self,
        quadrature: &QuadratureStabilityPlan,
        errors: &[Complex<T>],
        moments: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
        primal: Option<&[Complex<T>]>,
        mut context: Option<&mut GuardContext<T>>,
    ) -> Option<String> {
        if components == 0
            || errors.len() != quadrature.checks.len() * components
            || moments.len() != quadrature.moment_outputs * components
        {
            return Some("invalid propagated quadrature-error buffer".into());
        }
        let Some(primal) = primal.filter(|values| values.len() == self.primal_expressions.len())
        else {
            return Some("missing primal amplitudes for propagated quadrature errors".into());
        };
        let precision = prototype.get_precision();
        let mut roundoff = if CACHED {
            OperandRoundoff::with_unit(
                prototype,
                context.as_deref().unwrap().prototype_unit.clone(),
            )
        } else {
            OperandRoundoff::new(prototype)
        };
        let tolerance = if CACHED {
            context.as_deref().unwrap().tolerance.clone()
        } else {
            dyadic(prototype, 19i64 - i64::from(accuracy_bits))
        };
        let scale = if CACHED {
            None
        } else {
            Some(self.kinematic_scale(input, components, prototype))
        };
        for (index, check) in quadrature.checks.iter().enumerate() {
            let natural = if CACHED {
                let context = context.as_deref_mut().unwrap();
                context.natural(
                    context.output_slots[check.output],
                    self,
                    input,
                    components,
                    prototype,
                )
            } else {
                signed_power(scale.as_ref().unwrap(), check.dimension)
            };
            if !natural.is_finite() || natural.is_zero() {
                return Some(format!(
                    "quadrature-error normalization is outside the numeric range for {}",
                    check.label
                ));
            }
            for component in 0..components {
                let error = &errors[index * components + component];
                let moment = &moments[check.output * components + component];
                if !error.re.is_finite()
                    || !error.im.is_zero()
                    || error.re < error.re.zero()
                    || !moment.re.is_finite()
                    || !moment.im.is_finite()
                {
                    return Some(format!(
                        "invalid propagated quadrature estimate for {}, Taylor component {component}",
                        check.label
                    ));
                }
                let mut estimate = error.re.clone();
                if component == 0 {
                    let amplitude = &primal[check.primal_amplitude];
                    if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                        return Some(format!(
                            "nonfinite primal amplitude for quadrature estimate in {}",
                            check.label
                        ));
                    }
                    let primal_error = match roundoff.complex(amplitude) {
                        Ok(value) => value,
                        Err(reason) => return Some(format!("{reason} for {}", check.label)),
                    };
                    estimate += primal_error;
                }
                let magnitude = NativeFloat::hypot(&moment.re, &moment.im);
                let reference = if magnitude > natural {
                    magnitude
                } else {
                    natural.clone()
                };
                let target = tolerance.clone() * reference;
                if !target.is_finite() || target.is_zero() {
                    return Some(format!(
                        "quadrature-error tolerance is outside the numeric range for {}",
                        check.label
                    ));
                }
                if !estimate.is_finite() || estimate > target {
                    return Some(format!(
                        "propagated E6 quadrature estimate exceeds 2^(19-{accuracy_bits}) at working precision {precision} in {}, Taylor component {component}; successive-rule estimate, not a certified error bound",
                        check.label,
                    ));
                }
            }
        }
        None
    }
}

fn depends_on_projection(expression: &Atom, projections: &HashMap<Atom, Vec<Atom>>) -> bool {
    expression
        .get_all_symbols(false)
        .into_iter()
        .any(|symbol| projections.contains_key(&Atom::var(symbol)))
}

/// Project an affine body onto its E6 finite-master coefficients. Symbolica
/// extracts the local polynomial; branch predicates stay unchanged and lazy.
fn affine_projection(
    expression: &Atom,
    projections: &HashMap<Atom, Vec<Atom>>,
    count: usize,
) -> Result<Vec<Atom>, ReductionError> {
    if let Some(projected) = projections.get(expression) {
        return Ok(projected.clone());
    }
    if !depends_on_projection(expression, projections) {
        return Ok(vec![Atom::new(); count]);
    }
    if let Some((condition, yes, no)) = if_parts(expression.as_view()) {
        if depends_on_projection(&condition.to_owned(), projections) {
            return Err(ReductionError::Preparation(
                "E6 error propagation requires scalar-master-independent branch predicates".into(),
            ));
        }
        let yes = affine_projection(&yes.to_owned(), projections, count)?;
        let no = affine_projection(&no.to_owned(), projections, count)?;
        return Ok(yes
            .into_iter()
            .zip(no)
            .map(|(yes, no)| {
                if yes == no {
                    yes
                } else {
                    Symbol::IF.call((condition, yes, no))
                }
            })
            .collect());
    }
    let dependencies: HashMap<_, _> = expression
        .get_all_symbols(false)
        .into_iter()
        .map(Atom::var)
        .filter(|key| projections.contains_key(key))
        .map(|key| (key, Atom::num(1)))
        .collect();
    let mut variables = BTreeSet::new();
    collect_amplitude_variables(expression.as_view(), &dependencies, &mut variables);
    let variables: Vec<_> = variables.into_iter().collect();
    let projected_variables = variables
        .iter()
        .map(|value| affine_projection(value, projections, count))
        .collect::<Result<Vec<_>, _>>()?;
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    };
    let polynomial = expression.to_polynomial_in_vars_with_field::<u16>(&variables, &field);
    let mut result = vec![Atom::new(); count];
    for term in &polynomial {
        if depends_on_amplitudes(term.coefficient, &dependencies)
            || term
                .exponents
                .iter()
                .map(|&power| u32::from(power))
                .sum::<u32>()
                > 1
        {
            return Err(ReductionError::Preparation(
                "E6 error propagation requires affine scalar-master dependence".into(),
            ));
        }
        if let Some(index) = term.exponents.iter().position(|&power| power == 1) {
            for (coefficient, projected) in result.iter_mut().zip(&projected_variables[index]) {
                *coefficient = &*coefficient + term.coefficient * projected;
            }
        }
    }
    Ok(result)
}

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!(
        "iregi_quadrature_error::{name}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, Convention, ReductionSettings};
    use symbolica::{
        domains::{dual::HyperDual, float::SingleFloat, rational::Rational},
        evaluate::{Dualizer, ExpressionEvaluator},
    };

    fn c(value: f64) -> Complex<f64> {
        Complex::new(value, 0.0)
    }

    // Graph-only fixture: physical E6 values are tested by the executor.
    fn fixture() -> ReductionPlan {
        let mut settings = ReductionSettings::new(1, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.convention = Convention::OneLoop;
        let mut plan = ReductionPlan::new(settings).unwrap();
        plan.leaves[0].kind = MasterLeafKind::SixDimensionalPentagon;
        plan.leaves[0].active = vec![0, 1, 2, 3, 4];
        plan.aliases.clear();
        plan.moments = vec![[
            Atom::new(),
            Atom::new(),
            Atom::new(),
            plan.leaves[0].slots[3].clone(),
        ]];
        plan
    }

    fn evaluate(
        expressions: &[Atom],
        parameters: &[Atom],
        aliases: &[(Atom, Atom)],
        shape: &[Vec<usize>],
        input: &[Complex<f64>],
        rows: usize,
    ) -> Vec<Complex<f64>> {
        let exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(expressions, parameters)
                .add_aliases(aliases.iter().cloned())
                .unwrap()
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        let exact = exact
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(shape.to_vec()),
                Vec::new(),
            ))
            .unwrap();
        let mut batch = exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        0.0f64.from_rational(&value.re),
                        0.0f64.from_rational(&value.im),
                    )
                },
                53,
            )
            .into_batch_evaluator();
        let mut output = vec![c(0.0); rows * expressions.len() * shape.len()];
        batch.evaluate_batch(input, &mut output, rows).unwrap();
        output
    }

    #[test]
    fn signed_coefficient_jets_feed_positive_symbolica_convolution() {
        let mut plan = fixture();
        let x = plan.parameters[0].clone();
        let master = plan.leaves[0].slots[3].clone();
        let first = variable("test_first");
        let second = variable("test_second");
        plan.aliases = vec![
            (first.clone(), &x * &master + Atom::num(7)),
            (second.clone(), &x * &first - Atom::num(7) * &x),
        ];
        plan.moments[0][3] = second;
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let quadrature = QuadratureStabilityPlan::prepare(&plan, &stability)
            .unwrap()
            .unwrap();
        let shape = vec![vec![0], vec![1], vec![2]];
        let mut input = vec![c(0.0); plan.parameters.len() * 3];
        input[0] = c(2.0);
        input[1] = c(1.0);
        input[4 * 3] = c(1.0);
        input[5 * 3] = c(1.0);
        let coefficients = evaluate(
            quadrature.coefficient_expressions(),
            quadrature.coefficient_parameters(),
            quadrature.coefficient_aliases(),
            &shape,
            &input,
            1,
        );
        assert_eq!(coefficients, [c(4.0), c(4.0), c(1.0)]);
        let mut contraction: Vec<_> = coefficients
            .iter()
            .map(|value| c(value.re.hypot(value.im)))
            .collect();
        contraction.extend([c(1.0), c(2.0), c(3.0)]);
        let errors = evaluate(
            quadrature.contraction_expressions(),
            quadrature.contraction_parameters(),
            &[],
            &shape,
            &contraction,
            1,
        );
        // Independently: (4+4t+t²)(1+2t+3t²), including factor2 for
        // conversion of complex-infinity-norm master estimates.
        assert_eq!(errors, [c(8.0), c(24.0), c(42.0)]);
    }

    #[test]
    fn mixed_jets_and_inactive_singular_coefficients_preserve_lanes() {
        let mut plan = fixture();
        let x = plan.parameters[0].clone();
        let y = plan.parameters[1].clone();
        let selected = plan.parameters[2].clone();
        let master = plan.leaves[0].slots[3].clone();
        plan.moments[0][3] = Symbol::IF.call((&selected, &master / &x, &x * &y * &master));
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let quadrature = QuadratureStabilityPlan::prepare(&plan, &stability)
            .unwrap()
            .unwrap();
        let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let mut input = vec![c(0.0); plan.parameters.len() * 4 * 2];
        input[1] = c(1.0); // x=0: inactive inverse must remain unused.
        input[4] = c(3.0);
        input[6] = c(1.0);
        input[16] = c(1.0);
        input[20] = c(1.0);
        let row = plan.parameters.len() * 4;
        input[row] = c(2.0);
        input[row + 1] = c(1.0);
        input[row + 4] = c(3.0);
        input[row + 6] = c(1.0);
        input[row + 8] = c(1.0); // This lane selects the nonsingular inverse.
        input[row + 16] = c(1.0);
        input[row + 20] = c(1.0);
        let coefficients = evaluate(
            quadrature.coefficient_expressions(),
            quadrature.coefficient_parameters(),
            quadrature.coefficient_aliases(),
            &shape,
            &input,
            2,
        );
        assert_eq!(&coefficients[..4], &[c(0.0), c(3.0), c(0.0), c(1.0)]);
        assert_eq!(&coefficients[4..], &[c(0.5), c(-0.25), c(0.0), c(0.0)]);
        let mut contraction = Vec::new();
        for coefficients in coefficients.chunks_exact(4) {
            contraction.extend(coefficients.iter().map(|value| c(value.re.hypot(value.im))));
            contraction.extend([c(1.0), c(2.0), c(3.0), c(4.0)]);
        }
        let errors = evaluate(
            quadrature.contraction_expressions(),
            quadrature.contraction_parameters(),
            &[],
            &shape,
            &contraction,
            2,
        );
        assert_eq!(&errors[..4], &[c(0.0), c(6.0), c(0.0), c(20.0)]);
        assert_eq!(&errors[4..], &[c(1.0), c(2.5), c(3.0), c(5.5)]);
    }

    #[test]
    fn backend_budget_is_rechecked_after_cancellation_and_mixed_jet_amplification() {
        let mut plan = fixture();
        let x = plan.parameters[0].clone();
        let master = plan.leaves[0].slots[3].clone();
        // The central master value is exactly one. Its small uncertainty
        // survives this cancellation and can dominate the zero output.
        plan.moments[0][3] = Atom::num(1024) * &x * (&master - Atom::num(1));
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let quadrature = QuadratureStabilityPlan::prepare(&plan, &stability)
            .unwrap()
            .unwrap();
        let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let components = shape.len();
        let rows = 3;
        let mut parameters = plan.parameters.clone();
        parameters.extend(plan.leaves.iter().flat_map(|leaf| leaf.slots.clone()));
        let input_width = parameters.len() * components;
        let mut input = vec![c(0.0); rows * input_width];
        for (lane, values) in input.chunks_exact_mut(input_width).enumerate() {
            values[0] = c(if lane == 1 { 1.0 } else { 1.0 / 1024.0 });
            if lane == 2 {
                values[3] = c(1.0); // Amplification only in the mixed jet.
            }
            values[4 * components] = c(1.0); // mass squared
            values[5 * components] = c(1.0); // scale squared
            values[(plan.parameters.len() + 3) * components] = c(1.0);
        }
        let kinematics: Vec<_> = input
            .chunks_exact(input_width)
            .flat_map(|values| values[..plan.parameters.len() * components].iter().cloned())
            .collect();
        let coefficients = evaluate(
            quadrature.coefficient_expressions(),
            quadrature.coefficient_parameters(),
            quadrature.coefficient_aliases(),
            &shape,
            &kinematics,
            rows,
        );
        assert_eq!(
            coefficients,
            [
                1.0, 0.0, 0.0, 0.0, 1024.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1024.0
            ]
            .map(c)
        );

        let backend_error = dyadic(&0.0f64, -40);
        let backend_budget = dyadic(&0.0f64, 17 - 53);
        // These are exactly representable successive-rule master values.
        // The primal passes the backend's relative budget, while its zero
        // derivative components have zero errors and pass their zero budgets.
        assert_eq!(1.0 - (1.0 - backend_error), backend_error);
        assert!(backend_error <= backend_budget);
        let mut contraction = Vec::new();
        for values in coefficients.chunks_exact(components) {
            contraction.extend(values.iter().map(|value| c(value.re.hypot(value.im))));
            contraction.extend([c(backend_error), c(0.0), c(0.0), c(0.0)]);
        }
        let errors = evaluate(
            quadrature.contraction_expressions(),
            quadrature.contraction_parameters(),
            &[],
            &shape,
            &contraction,
            rows,
        );
        assert_eq!(errors[0], c(2.0 * backend_error));
        assert_eq!(errors[4], c(2048.0 * backend_error));
        assert_eq!(errors[8], c(2.0 * backend_error));
        assert_eq!(errors[11], c(2048.0 * backend_error));

        let moments = evaluate(
            &plan.moments.iter().flatten().cloned().collect::<Vec<_>>(),
            &parameters,
            &plan.aliases,
            &shape,
            &input,
            rows,
        );
        assert!(moments.iter().all(|value| *value == c(0.0)));
        let primal_input: Vec<_> = input.iter().step_by(components).cloned().collect();
        let primal_aliases: Vec<_> = plan
            .aliases
            .iter()
            .chain(stability.aliases())
            .chain(stability.primal_aliases())
            .cloned()
            .collect();
        let primal = evaluate(
            stability.primal_expressions(),
            &parameters,
            &primal_aliases,
            &[vec![0]],
            &primal_input,
            rows,
        );
        let primal_width = stability.primal_expressions().len();
        let inspect = |lane: usize, accuracy_bits| {
            stability.inspect_quadrature_lane(
                &quadrature,
                &errors[lane * components..(lane + 1) * components],
                &moments[lane * 4 * components..(lane + 1) * 4 * components],
                &input[lane * input_width..(lane + 1) * input_width],
                components,
                &0.0,
                accuracy_bits,
                Some(&primal[lane * primal_width..(lane + 1) * primal_width]),
            )
        };
        assert!(inspect(0, 53).is_none());
        assert!(inspect(1, 53).unwrap().contains("Taylor component 0"));
        assert!(inspect(2, 53).unwrap().contains("Taylor component 3"));
        let stricter = inspect(0, 64).unwrap();
        assert!(stricter.contains("2^(19-64)"));
        assert!(stricter.contains("Taylor component 0"));
    }

    #[test]
    fn amplified_derivative_errors_and_invalid_estimates_are_rejected() {
        let plan = fixture();
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let quadrature = QuadratureStabilityPlan::prepare(&plan, &stability)
            .unwrap()
            .unwrap();
        let input = [
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(1.0),
            c(0.0),
            c(1.0),
            c(0.0),
        ];
        let moments = vec![c(0.0); 8];
        let primal = vec![c(0.0); stability.primal_expressions().len()];
        let inspect = |errors: &[Complex<f64>], primal| {
            stability.inspect_quadrature_lane(
                &quadrature,
                errors,
                &moments,
                &input,
                2,
                &0.0,
                53,
                primal,
            )
        };
        assert!(inspect(&[c(1e-12), c(1e-12)], Some(&primal)).is_none());
        assert!(
            inspect(&[c(1e-12), c(1e-8)], Some(&primal))
                .unwrap()
                .contains("Taylor component 1")
        );
        assert!(inspect(&[c(f64::NAN), c(0.0)], Some(&primal)).is_some());
        assert!(inspect(&[c(-1.0), c(0.0)], Some(&primal)).is_some());
        assert!(inspect(&[c(0.0)], Some(&primal)).is_some());
        assert!(inspect(&[c(0.0), c(0.0)], None).is_some());
    }

    #[test]
    fn combined_quadrature_budget_uses_actual_primal_amplitude_precision() {
        use symbolica::domains::float::Float;

        let plan = fixture();
        let stability = StabilityPlan::prepare(&plan).unwrap();
        let quadrature = QuadratureStabilityPlan::prepare(&plan, &stability)
            .unwrap()
            .unwrap();
        let prototype = Float::new(256);
        let number = |bits, value| Complex::new(Float::with_val(bits, value), Float::new(bits));
        let input = [0, 0, 0, 0, 1, 1].map(|value| number(256, value));
        let moments = vec![number(256, 0); 4];
        let errors = vec![number(256, 0); quadrature.checks.len()];
        let inspect = |bits, amplitude| {
            let mut primal = vec![number(256, 0); stability.primal_expressions.len()];
            for check in &quadrature.checks {
                primal[check.primal_amplitude] = number(bits, amplitude);
            }
            stability.inspect_quadrature_lane(
                &quadrature,
                &errors,
                &moments,
                &input,
                1,
                &prototype,
                128,
                Some(&primal),
            )
        };
        assert!(
            inspect(80, 1)
                .unwrap()
                .contains("quadrature estimate exceeds")
        );
        assert!(inspect(256, 1).is_none());
        assert!(inspect(1, 0).is_none());
    }
}
