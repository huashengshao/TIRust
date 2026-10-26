//! Opt-in full-Taylor transport of the existing heuristic master amplitudes.
//!
//! Signed local affine coefficients are Dualized before taking their component
//! norms. A second Symbolica-Dualized positive alias DAG owns every convolution.
//! No ABS is differentiated and no Taylor arithmetic is implemented here.
use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, OnceLock},
};

use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{atom::AtomField, dual::HyperDual, float::Complex, rational::Rational},
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
};

use super::{
    branch_path, collect_alias_uses, collect_amplitude_variables, depends_on_amplitudes, guarded,
    if_parts, union_paths,
};
use crate::{
    execution::{compile_exact, map_exact},
    plan::ReductionPlan,
    types::ReductionError,
};

#[derive(Clone, Debug)]
struct Seed {
    input: usize,
    imported: bool,
}

/// The recipe deliberately owns no ReductionPlan or execution-cache reference.
/// Only retaining multi-component workspaces force its shared exact programs.
/// One recipe belongs to one PreparedExecution, whose existing cache key
/// contains the complete ordered shape. The OnceLock must never cross shapes.
pub(crate) struct ComponentSensitivityRecipe {
    parameters: Vec<Atom>,
    aliases: Vec<(Atom, Atom)>,
    outputs: Vec<Atom>,
    native_slots: Vec<Atom>,
    external_seeds: Vec<(Atom, Atom, i32)>,
    prepared: OnceLock<Result<Arc<PreparedComponentSensitivity>, String>>,
}

impl ComponentSensitivityRecipe {
    pub(crate) fn new(plan: &ReductionPlan) -> Self {
        let mut parameters = plan.parameters.clone();
        let native_slots: Vec<_> = plan
            .leaves
            .iter()
            .flat_map(|leaf| leaf.slots.iter().cloned())
            .collect();
        parameters.extend(native_slots.iter().cloned());
        Self {
            parameters,
            aliases: plan.aliases.clone(),
            outputs: plan.moments.iter().flatten().cloned().collect(),
            native_slots,
            external_seeds: plan.external_primal_seeds.clone(),
            prepared: OnceLock::new(),
        }
    }

    pub(crate) fn instantiate<T: NativeFloat>(
        &self,
        shape: &[Vec<usize>],
        prototype: &T,
    ) -> Result<ComponentSensitivityWorkspace<T>, ReductionError>
    where
        Complex<T>: EvaluationDomain,
    {
        let exact = self
            .prepared
            .get_or_init(|| {
                PreparedComponentSensitivity::prepare(self, shape)
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| ReductionError::Preparation(error.clone()))?;
        if exact.shape.as_slice() != shape {
            return Err(ReductionError::Preparation(
                "component sensitivity ordered shape mismatch".into(),
            ));
        }
        Ok(ComponentSensitivityWorkspace {
            exact: exact.clone(),
            coefficients: map_exact(&exact.coefficients, prototype),
            positive: map_exact(&exact.positive, prototype),
            coefficient_values: Vec::new(),
            positive_inputs: Vec::new(),
            output: Vec::new(),
            invalid: Vec::new(),
        })
    }
}

struct PreparedComponentSensitivity {
    shape: Vec<Vec<usize>>,
    input_count: usize,
    coefficient_count: usize,
    output_count: usize,
    seeds: Vec<Seed>,
    coefficients: ExpressionEvaluator<Complex<Rational>>,
    positive: ExpressionEvaluator<Complex<Rational>>,
}

struct Builder {
    coefficients: Vec<Atom>,
    coefficient_parameters: Vec<Atom>,
    // Exact Symbolica Atom identity includes the active demand mask. Reuse
    // numerical coefficient inputs without removing any positive edge use.
    coefficient_indices: HashMap<Atom, usize>,
    coefficient_uses: usize,
}

impl Builder {
    fn magnitude(&mut self, expression: &Atom, active: &Atom) -> Atom {
        self.coefficient_uses += 1;
        let signed = guarded(active, expression.clone(), false);
        if let Some(index) = self.coefficient_indices.get(&signed) {
            return self.coefficient_parameters[*index].clone();
        }
        let index = self.coefficients.len();
        let parameter = var(&format!("coefficient_{index}"));
        self.coefficients.push(signed.clone());
        self.coefficient_parameters.push(parameter.clone());
        self.coefficient_indices.insert(signed, index);
        parameter
    }

    fn local(
        &mut self,
        expression: &Atom,
        active: &Atom,
        amplitudes: &HashMap<Atom, Atom>,
    ) -> Result<Atom, ReductionError> {
        if let Some(amplitude) = amplitudes.get(expression) {
            return Ok(amplitude.clone());
        }
        if !depends_on_amplitudes(expression, amplitudes) {
            return Ok(self.magnitude(expression, active));
        }
        if let Some((condition, yes, no)) = if_parts(expression.as_view()) {
            if depends_on_amplitudes(&condition.to_owned(), amplitudes) {
                return Err(ReductionError::Preparation(
                    "component sensitivity requires master-independent predicates".into(),
                ));
            }
            // The signed evaluator chooses this mask on the original inputs.
            // Positive inputs never reclassify a branch from magnitude data.
            let mask = Symbol::IF.call((condition, Atom::num(1), Atom::new()));
            let mask = self.magnitude(&mask, active);
            let yes = self.local(
                &yes.to_owned(),
                &branch_path(active, condition, true),
                amplitudes,
            )?;
            let no = self.local(
                &no.to_owned(),
                &branch_path(active, condition, false),
                amplitudes,
            )?;
            return Ok(Symbol::IF.call((mask, yes, no)));
        }
        let mut variables = BTreeSet::new();
        collect_amplitude_variables(expression.as_view(), amplitudes, &mut variables);
        let variables: Vec<_> = variables.into_iter().collect();
        let variable_amplitudes = variables
            .iter()
            .map(|value| self.local(value, active, amplitudes))
            .collect::<Result<Vec<_>, _>>()?;
        let field = AtomField {
            statistical_zero_test: false,
            cancel_check_on_division: false,
            custom_normalization: None,
        };
        let polynomial = expression.to_polynomial_in_vars_with_field::<u16>(&variables, &field);
        let mut contributions = Vec::with_capacity(polynomial.nterms());
        for term in &polynomial {
            if depends_on_amplitudes(term.coefficient, amplitudes)
                || term
                    .exponents
                    .iter()
                    .map(|&power| u32::from(power))
                    .sum::<u32>()
                    > 1
            {
                return Err(ReductionError::Preparation(
                    "component sensitivity requires immediate affine master dependence".into(),
                ));
            }
            let coefficient = self.magnitude(term.coefficient, active);
            contributions.push(
                if let Some(index) = term.exponents.iter().position(|&power| power == 1) {
                    coefficient * &variable_amplitudes[index]
                } else {
                    coefficient
                },
            );
        }
        Ok(Atom::add_many(contributions))
    }
}

impl PreparedComponentSensitivity {
    fn prepare(
        recipe: &ComponentSensitivityRecipe,
        shape: &[Vec<usize>],
    ) -> Result<Self, ReductionError> {
        let mut amplitudes = HashMap::new();
        let mut positive_parameters = Vec::new();
        let mut seeds = Vec::new();
        for (value, imported) in recipe.native_slots.iter().map(|value| (value, None)).chain(
            recipe
                .external_seeds
                .iter()
                .map(|(value, amplitude, _)| (value, Some(amplitude))),
        ) {
            let source = imported.unwrap_or(value);
            let input = recipe
                .parameters
                .iter()
                .position(|parameter| parameter == source)
                .ok_or_else(|| {
                    ReductionError::Preparation(
                        "component sensitivity seed missing from exact parameter map".into(),
                    )
                })?;
            let parameter = var(&format!("seed_{}", seeds.len()));
            if amplitudes
                .insert(value.clone(), parameter.clone())
                .is_some()
            {
                return Err(ReductionError::Preparation(
                    "duplicate component sensitivity seed".into(),
                ));
            }
            positive_parameters.push(parameter);
            seeds.push(Seed {
                input,
                imported: imported.is_some(),
            });
        }
        // Retain original demand paths so extracting a coefficient cannot make
        // an inactive singular recurrence an eager signed-evaluator root.
        let definitions: HashMap<_, _> = recipe.aliases.iter().cloned().collect();
        let mut uses = HashMap::new();
        let always = Atom::num(1);
        for expression in &recipe.outputs {
            collect_alias_uses(expression.as_view(), &always, &definitions, &mut uses);
        }
        let mut reachable = HashMap::new();
        for (alias, body) in recipe.aliases.iter().rev() {
            if let Some(paths) = uses.remove(alias) {
                let active = union_paths(paths);
                collect_alias_uses(body.as_view(), &active, &definitions, &mut uses);
                reachable.insert(alias.clone(), active);
            }
        }
        let mut builder = Builder {
            coefficients: Vec::new(),
            coefficient_parameters: Vec::new(),
            coefficient_indices: HashMap::new(),
            coefficient_uses: 0,
        };
        let mut positive_aliases = Vec::new();
        for (alias, body) in &recipe.aliases {
            let Some(active) = reachable.get(alias) else {
                continue;
            };
            if !depends_on_amplitudes(body, &amplitudes) {
                continue;
            }
            let body = builder.local(body, active, &amplitudes)?;
            let handle = var(&format!("amplitude_{}", positive_aliases.len()));
            positive_aliases.push((handle.clone(), body));
            amplitudes.insert(alias.clone(), handle);
        }
        let positive_outputs = recipe
            .outputs
            .iter()
            .map(|expression| builder.local(expression, &always, &amplitudes))
            .collect::<Result<Vec<_>, _>>()?;
        if std::env::var_os("TIRUST_TRACE_BATCH_ALLOC").is_some() {
            eprintln!(
                "TIRUST_COMPONENT_COEFFICIENTS requests={} unique={} shape_components={} outputs={}",
                builder.coefficient_uses, builder.coefficients.len(), shape.len(), recipe.outputs.len()
            );
        }
        let mut selected = BTreeSet::new();
        let mut pending = builder.coefficients.clone();
        while let Some(expression) = pending.pop() {
            for symbol in expression.get_all_symbols(false) {
                let alias = Atom::var(symbol);
                if amplitudes.contains_key(&alias) {
                    return Err(ReductionError::Preparation(
                        "signed component coefficient still depends on a master or uncertain alias"
                            .into(),
                    ));
                }
                if let Some(body) = definitions.get(&alias) {
                    if selected.insert(alias) {
                        pending.push(body.clone());
                    }
                }
            }
        }
        let coefficient_aliases: Vec<_> = recipe
            .aliases
            .iter()
            .filter(|(alias, _)| selected.contains(alias))
            .cloned()
            .collect();
        positive_parameters.extend(builder.coefficient_parameters);
        let dual = HyperDual::<Complex<Rational>>::new(shape.to_vec());
        let coefficients = compile_exact(
            &builder.coefficients,
            &recipe.parameters,
            &coefficient_aliases,
            Some(&dual),
        )?;
        let positive = compile_exact(
            &positive_outputs,
            &positive_parameters,
            &positive_aliases,
            Some(&dual),
        )?;
        Ok(Self {
            shape: shape.to_vec(),
            input_count: recipe.parameters.len(),
            coefficient_count: builder.coefficients.len(),
            output_count: positive_outputs.len(),
            seeds,
            coefficients,
            positive,
        })
    }
}

#[derive(Clone)]
pub(crate) struct ComponentSensitivityWorkspace<T: NativeFloat> {
    exact: Arc<PreparedComponentSensitivity>,
    coefficients: BatchExpressionEvaluator<Complex<T>>,
    positive: BatchExpressionEvaluator<Complex<T>>,
    coefficient_values: Vec<Complex<T>>,
    positive_inputs: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
    invalid: Vec<Option<String>>,
}

impl<T: NativeFloat> ComponentSensitivityWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(crate) fn evaluate(
        &mut self,
        inputs: &[Complex<T>],
        rows: usize,
        components: usize,
        zero: &Complex<T>,
    ) -> Result<(), ReductionError> {
        let input_width = width(self.exact.input_count, components)?;
        let coefficient_width = width(self.exact.coefficient_count, components)?;
        let output_width = width(self.exact.output_count, components)?;
        if components != self.exact.shape.len() || inputs.len() != width(rows, input_width)? {
            return Err(ReductionError::Evaluation(
                "component sensitivity input layout mismatch".into(),
            ));
        }
        self.coefficient_values
            .resize(width(rows, coefficient_width)?, zero.clone());
        self.coefficients
            .evaluate_batch(inputs, &mut self.coefficient_values, rows)
            .map_err(|error| ReductionError::Evaluation(error.to_string()))?;
        self.positive_inputs.clear();
        self.invalid.clear();
        self.invalid.resize(rows, None);
        for (row, input) in inputs.chunks_exact(input_width).enumerate() {
            for seed in &self.exact.seeds {
                for value in &input[seed.input * components..(seed.input + 1) * components] {
                    if seed.imported {
                        if !valid_amplitude(value) {
                            self.invalid[row] =
                                Some("invalid imported component sensitivity amplitude".into());
                            self.positive_inputs.push(zero.clone());
                        } else {
                            self.positive_inputs.push(value.clone());
                        }
                    } else {
                        self.positive_inputs.push(Complex::new(
                            NativeFloat::hypot(&value.re, &value.im),
                            zero.im.clone(),
                        ));
                    }
                }
            }
            self.positive_inputs.extend(
                self.coefficient_values[row * coefficient_width..(row + 1) * coefficient_width]
                    .iter()
                    .map(|value| {
                        Complex::new(NativeFloat::hypot(&value.re, &value.im), zero.im.clone())
                    }),
            );
        }
        self.output.resize(width(rows, output_width)?, zero.clone());
        self.positive
            .evaluate_batch(&self.positive_inputs, &mut self.output, rows)
            .map_err(|error| ReductionError::Evaluation(error.to_string()))
    }

    pub(crate) fn lane(&self, row: usize, components: usize) -> &[Complex<T>] {
        let width = self.exact.output_count * components;
        &self.output[row * width..(row + 1) * width]
    }

    pub(crate) fn invalid_lane(&self, row: usize) -> Option<String> {
        self.invalid[row].clone()
    }
}

impl super::StabilityPlan {
    /// Same dimensional working-ULP heuristic as the primal monitor, now for
    /// every coefficient. This remains a sensitivity estimate, not a bound on
    /// backend errors or the entire arithmetic computation.
    pub(crate) fn inspect_component_amplitude_lane<T: NativeFloat>(
        &self,
        amplitudes: &[Complex<T>],
        moments: &[Complex<T>],
        input: &[Complex<T>],
        components: usize,
        prototype: &T,
        accuracy_bits: u32,
    ) -> Option<String> {
        let Some(expected) = self.output_dimensions.len().checked_mul(components) else {
            return Some("component sensitivity dimensions overflow".into());
        };
        if components == 0 || amplitudes.len() != expected || moments.len() != expected {
            return Some("invalid component sensitivity monitor dimensions".into());
        }
        let tolerance = super::dyadic(prototype, 19i64 - i64::from(accuracy_bits));
        let scale = self.kinematic_scale(input, components, prototype);
        let mut roundoff = super::OperandRoundoff::new(prototype);
        for (output, &dimension) in self.output_dimensions.iter().enumerate() {
            let natural = super::signed_power(&scale, dimension);
            if !natural.is_finite() || natural.is_zero() {
                return Some(
                    "component sensitivity normalization is outside the numeric range".into(),
                );
            }
            for component in 0..components {
                let index = output * components + component;
                let amplitude = &amplitudes[index];
                if !valid_amplitude(amplitude) {
                    return Some(format!(
                        "invalid component sensitivity amplitude in output {output}, Taylor component {component}"
                    ));
                }
                let estimate = match roundoff.complex(amplitude) {
                    Ok(value) => value,
                    Err(reason) => return Some(reason.into()),
                };
                let value = &moments[index];
                let magnitude = NativeFloat::hypot(&value.re, &value.im);
                let reference = if magnitude > natural {
                    magnitude
                } else {
                    natural.clone()
                };
                let target = tolerance.clone() * reference;
                if !target.is_finite()
                    || target.is_zero()
                    || !estimate.is_finite()
                    || estimate > target
                {
                    return Some(format!(
                        "estimated compounded master cancellation exceeds 2^(19-{accuracy_bits}) in output {output}, Taylor component {component}; full-component sensitivity estimate, not a certified error bound"
                    ));
                }
            }
        }
        None
    }
}

pub(crate) fn valid_amplitude<T: NativeFloat>(value: &Complex<T>) -> bool {
    value.re.is_finite()
        && value.im.is_finite()
        && value.im.is_zero()
        && value.re >= value.re.zero()
}

fn width(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b).ok_or_else(|| {
        ReductionError::Evaluation("component sensitivity dimensions overflow".into())
    })
}
fn var(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!(
        "iregi_component_sensitivity::{name}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(x: f64) -> Complex<f64> {
        Complex::new(x, 0.)
    }
    fn fixture(
        parameters: Vec<Atom>,
        aliases: Vec<(Atom, Atom)>,
        finite: Atom,
        native: Vec<Atom>,
        external: Vec<(Atom, Atom, i32)>,
    ) -> ComponentSensitivityRecipe {
        ComponentSensitivityRecipe {
            parameters,
            aliases,
            outputs: vec![Atom::new(), Atom::new(), Atom::new(), finite],
            native_slots: native,
            external_seeds: external,
            prepared: OnceLock::new(),
        }
    }
    fn mixed() -> Vec<Vec<usize>> {
        vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![2, 0],
            vec![1, 1],
            vec![0, 2],
        ]
    }

    #[test]
    fn repeated_guarded_coefficients_keep_every_signed_edge_and_mixed_product() {
        let x = var("intern_weight");
        let a = var("intern_master_a");
        let b = var("intern_master_b");
        let aa = var("intern_amplitude_a");
        let ab = var("intern_amplitude_b");
        let shape = mixed();
        let dual = HyperDual::<Complex<Rational>>::new(shape.clone());
        // An independent positive Symbolica evaluator supplies all six Taylor
        // products. No convolution or derivative arithmetic is implemented here.
        let positive = compile_exact(
            &[&x * (&a + &b)],
            &[x.clone(), a.clone(), b.clone()],
            &[],
            Some(&dual),
        ).unwrap();
        let mut expected_evaluator = map_exact(&positive, &0f64);
        let weight = [2., 3., 5., 7., 11., 13.];
        let amplitudes_a = [1., 17., 19., 23., 29., 31.];
        let amplitudes_b = [2., 37., 41., 43., 47., 53.];
        let reference_input = weight.into_iter().chain(amplitudes_a).chain(amplitudes_b)
            .map(c).collect::<Vec<_>>();
        let mut expected = vec![c(0.); 6];
        expected_evaluator.evaluate_batch(&reference_input, &mut expected, 1).unwrap();
        for sign in [1, -1] {
            let recipe = fixture(
                vec![x.clone(), a.clone(), b.clone(), aa.clone(), ab.clone()],
                vec![],
                &x * &a + Atom::num(sign) * &x * &b,
                vec![],
                vec![(a.clone(), aa.clone(), 0), (b.clone(), ab.clone(), 0)],
            );
            let mut workspace = recipe.instantiate(&shape, &0f64).unwrap();
            // Three zero Laurent outputs share one zero coefficient. The two
            // finite edges reuse x only when their SIGNED coefficients match.
            assert_eq!(workspace.exact.coefficient_count, if sign == 1 { 2 } else { 3 });
            let input = weight.into_iter().map(|v| -v).chain([0.; 12])
                .chain(amplitudes_a).chain(amplitudes_b).map(c).collect::<Vec<_>>();
            workspace.evaluate(&input, 1, 6, &c(0.)).unwrap();
            assert_eq!(&workspace.lane(0, 6)[18..], expected.as_slice());
            assert_eq!(&workspace.lane(0, 6)[..18], [c(0.); 18]);
        }
    }

    #[test]
    fn coefficient_interning_keeps_independent_demand_masks_per_lane() {
        let condition_a = var("intern_condition_a");
        let condition_b = var("intern_condition_b");
        let x = var("intern_branch_weight");
        let a = var("intern_branch_master_a");
        let b = var("intern_branch_master_b");
        let aa = var("intern_branch_amplitude_a");
        let ab = var("intern_branch_amplitude_b");
        let finite = Symbol::IF.call((&condition_a, &x * &a, Atom::new()))
            + Symbol::IF.call((&condition_b, &x * &b, Atom::new()));
        let parameters = vec![condition_a.clone(), condition_b.clone(), x.clone(), a.clone(), b.clone()];
        let shape = mixed();
        let positive = compile_exact(&[finite.clone()], &parameters, &[],
            Some(&HyperDual::<Complex<Rational>>::new(shape.clone()))).unwrap();
        let recipe = fixture(
            parameters.into_iter().chain([aa.clone(), ab.clone()]).collect(),
            vec![], finite, vec![], vec![(a, aa, 0), (b, ab, 0)],
        );
        let mut workspace = recipe.instantiate(&shape, &0f64).unwrap();
        let mut expected_evaluator = map_exact(&positive, &0f64);
        let mut inputs = Vec::new();
        let mut reference = Vec::new();
        for (ca, cb) in [(1., 0.), (0., 1.), (1., 1.)] {
            let conditions = [ca, 0., 0., 0., 0., 0., cb, 0., 0., 0., 0., 0.];
            let weight = [2., 3., 5., 7., 11., 13.];
            let amps_a = [1., 17., 19., 23., 29., 31.];
            let amps_b = [2., 37., 41., 43., 47., 53.];
            inputs.extend(conditions.into_iter().chain(weight.map(|v| -v)).chain([0.; 12])
                .chain(amps_a).chain(amps_b).map(c));
            reference.extend(conditions.into_iter().chain(weight).chain(amps_a).chain(amps_b).map(c));
        }
        let mut expected = vec![c(0.); 18];
        expected_evaluator.evaluate_batch(&reference, &mut expected, 3).unwrap();
        workspace.evaluate(&inputs, 3, 6, &c(0.)).unwrap();
        for row in 0..3 {
            assert_eq!(&workspace.lane(row, 6)[18..], &expected[row*6..(row+1)*6]);
        }
    }

    #[test]
    fn earlier_alias_cancellation_and_all_mixed_products_remain_visible() {
        let x = var("test_x");
        let m = var("test_m");
        let a = var("test_a");
        let b = var("test_b");
        // Signed b=x*(x*m+7)-7*x simplifies globally to x²*m. The
        // immediate-alias amplitude must retain both contributions of 7*x.
        let recipe = fixture(
            vec![x.clone(), m.clone()],
            vec![
                (a.clone(), &x * &m + Atom::num(7)),
                (b.clone(), &x * &a - Atom::num(7) * &x),
            ],
            b,
            vec![m],
            vec![],
        );
        let mut workspace = recipe.instantiate(&mixed(), &0f64).unwrap();
        let inputs = [2., 1., 3., 0., 0., 0., -1., 2., -3., 5., -7., 11.].map(c);
        workspace.evaluate(&inputs, 1, 6, &c(0.)).unwrap();
        assert_eq!(
            &workspace.lane(0, 6)[18..],
            [32., 26., 66., 29., 70., 89.].map(c)
        );
        assert_eq!(&workspace.lane(0, 6)[..18], [c(0.); 18]);
        // A signed global projection has primal amplitude four, not 32.
        assert!(workspace.lane(0, 6)[18].re > 4.);
    }

    #[test]
    fn imported_derivative_amplitude_survives_zero_primal_weight() {
        let x = var("test_weight");
        let value = var("test_child");
        let amplitude = var("test_child_amplitude");
        let recipe = fixture(
            vec![x.clone(), value.clone(), amplitude.clone()],
            vec![],
            &x * &value,
            vec![],
            vec![(value, amplitude, 0)],
        );
        let shape = vec![vec![0], vec![1], vec![2]];
        let mut workspace = recipe.instantiate(&shape, &0f64).unwrap();
        let inputs = [0., 1., 0., 1., 0., 0., 1., 1099511627776., 3.].map(c);
        workspace.evaluate(&inputs, 1, 3, &c(0.)).unwrap();
        assert_eq!(
            &workspace.lane(0, 3)[9..],
            [c(0.), c(1.), c(1099511627776.)]
        );
    }

    #[test]
    fn complex_coefficient_magnitudes_are_not_derivatives_of_abs() {
        let x = var("test_complex_coefficient");
        let m = var("test_complex_value");
        let recipe = fixture(vec![x.clone(), m.clone()], vec![], &x * &m, vec![m], vec![]);
        let shape = vec![vec![0], vec![1]];
        let mut workspace = recipe.instantiate(&shape, &0f64).unwrap();
        workspace
            .evaluate(
                &[
                    Complex::new(3., 4.),
                    Complex::new(-5., 12.),
                    c(2.),
                    Complex::new(0., -3.),
                ],
                1,
                2,
                &c(0.),
            )
            .unwrap();
        // |x[0]|=5, |x[1]|=13, |m[0]|=2, |m[1]|=3.
        // The norm derivative of x would be 33/5, which gives a different result.
        assert_eq!(&workspace.lane(0, 2)[6..], [c(10.), c(41.)]);
    }

    #[test]
    fn signed_predicates_keep_inactive_singular_coefficient_lazy_per_lane() {
        let x = var("test_divisor");
        let condition = var("test_condition");
        let m = var("test_lazy_master");
        let expression = Symbol::IF.call((&condition, &m, &m / &x));
        let recipe = fixture(
            vec![x, condition, m.clone()],
            vec![],
            expression,
            vec![m],
            vec![],
        );
        let shape = vec![vec![0], vec![1]];
        let mut workspace = recipe.instantiate(&shape, &0f64).unwrap();
        let inputs = [0., 1., 1., 0., 1., 2., 2., 0., 0., 0., 1., 2.].map(c);
        workspace.evaluate(&inputs, 2, 2, &c(0.)).unwrap();
        assert_eq!(&workspace.lane(0, 2)[6..], [c(1.), c(2.)]);
        assert_eq!(&workspace.lane(1, 2)[6..], [c(0.5), c(1.)]);
    }

    #[test]
    fn nonlinear_masters_and_master_dependent_predicates_are_explicitly_rejected() {
        let m = var("test_bad_master");
        let shape = vec![vec![0], vec![1]];
        let nonlinear = fixture(vec![m.clone()], vec![], m.pow(2), vec![m.clone()], vec![]);
        assert!(matches!(
            nonlinear.instantiate(&shape, &0f64),
            Err(ReductionError::Preparation(_))
        ));
        let conditional = fixture(
            vec![m.clone()],
            vec![],
            Symbol::IF.call((&m, &m, Atom::new())),
            vec![m],
            vec![],
        );
        assert!(matches!(
            conditional.instantiate(&shape, &0f64),
            Err(ReductionError::Preparation(_))
        ));
    }

    #[test]
    fn invalid_imported_lane_does_not_pollute_neighbor_or_next_call() {
        let value = var("test_import");
        let amplitude = var("test_import_a");
        let recipe = fixture(
            vec![value.clone(), amplitude.clone()],
            vec![],
            value.clone(),
            vec![],
            vec![(value, amplitude, 0)],
        );
        let shape = vec![vec![0], vec![1]];
        let mut workspace = recipe.instantiate(&shape, &0f64).unwrap();
        workspace
            .evaluate(&[1., 0., 1., -3., 1., 0., 1., 7.].map(c), 2, 2, &c(0.))
            .unwrap();
        assert!(workspace.invalid_lane(0).is_some());
        assert!(workspace.invalid_lane(1).is_none());
        assert_eq!(&workspace.lane(1, 2)[6..], [c(1.), c(7.)]);
        workspace
            .evaluate(&[1., 0., 1., 9.].map(c), 1, 2, &c(0.))
            .unwrap();
        assert!(workspace.invalid_lane(0).is_none());
        assert_eq!(workspace.lane(0, 2)[7], c(9.));
    }

    #[test]
    fn exact_prepared_sensitivity_never_reinterprets_reordered_shape() {
        let x = var("test_shape_x");
        let recipe = fixture(vec![x.clone()], vec![], x, vec![], vec![]);
        let shape = mixed();
        recipe.instantiate(&shape, &0f64).unwrap();
        let mut reordered = shape;
        reordered.swap(1, 2);
        assert!(matches!(
            recipe.instantiate(&reordered, &0f64),
            Err(ReductionError::Preparation(_))
        ));
    }

    #[test]
    fn full_component_guard_detects_uncertainty_that_primal_does_not_contain() {
        let plan = ReductionPlan::new(crate::ReductionSettings::new(1, 0)).unwrap();
        let stability = super::super::StabilityPlan::prepare(&plan).unwrap();
        let components = 3;
        let mut inputs = vec![c(0.); plan.parameters.len() * components];
        inputs[plan.layout.masses.start * components] = c(1.);
        inputs[plan.layout.scale * components] = c(1.);
        let moments = vec![c(0.); 4 * components];
        let mut amplitudes = vec![c(0.); 4 * components];
        amplitudes[9] = c(1.);
        amplitudes[11] = c(2f64.powi(100));
        let reason = stability
            .inspect_component_amplitude_lane(&amplitudes, &moments, &inputs, components, &0f64, 53)
            .unwrap();
        assert!(reason.contains("Taylor component 2"));
    }
}
