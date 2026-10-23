//! Validation-only finite D0 neighborhood. All polynomial/jet/ball operations
//! belong to Symbolica; production master dispatch is deliberately untouched.

use oneloop::NativeFloat;
use std::sync::Arc;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        dual::{DualNumberStructure, HyperDual},
        float::{Complex, ComplexBall, RealBall},
        integer::Integer,
        rational::{Q, Rational},
    },
    evaluate::{
        BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator, Instruction,
    },
    poly::PolyVariable,
    tensors::matrix::Matrix,
};

pub type Exact = Complex<Rational>;
pub const PHYSICAL_ARGUMENTS: usize = 10; // six invariants, four squared masses
const PREPARED_ARGUMENTS: usize = 11; // normalized invariants/deviations, fixed M

pub fn pow2_minus(bits: u32) -> Rational {
    Rational::from((Integer::from(1), Integer::from(2).pow(u64::from(bits))))
}
pub fn zero() -> Exact {
    Complex::new(Rational::from(0), Rational::from(0))
}
pub fn abs(x: &Rational) -> Rational {
    if x < &Rational::from(0) {
        -x.clone()
    } else {
        x.clone()
    }
}
fn upper_abs(x: &Exact) -> Rational {
    abs(&x.re) + abs(&x.im)
}

/// Explicit error contract. Each component has a natural absolute budget;
/// relative accuracy uses that component's returned complex magnitude.
/// Acceptance is error² <= max(absolute², relative²*|returned|²).
#[derive(Clone)]
pub struct AccuracyContext {
    pub requested_bits: u32,
    pub absolute_per_component: Vec<Rational>,
    pub relative: Rational,
}
impl AccuracyContext {
    pub fn binary(bits: u32, components: usize) -> Self {
        Self {
            requested_bits: bits,
            absolute_per_component: vec![pow2_minus(bits); components],
            relative: pow2_minus(bits),
        }
    }
    fn validate(&self, components: usize) -> Result<(), String> {
        if self.requested_bits == 0
            || self.absolute_per_component.len() != components
            || self
                .absolute_per_component
                .iter()
                .any(|x| x <= &Rational::from(0))
            || self.relative < Rational::from(0)
        {
            return Err("invalid requested accuracy context".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Success,
    InvalidInput(String),
    OutsideAnalyticDomain,
    UncertifiedComponent(usize),
    NonFiniteOutput,
}
#[derive(Clone, Debug)]
pub struct ComponentBound {
    pub truncation: Rational,
    pub real_lower: Rational,
    pub real_upper: Rational,
    pub imaginary_lower: Rational,
    pub imaginary_upper: Rational,
    pub maximum_error_squared: Rational,
    pub allowed_error_squared: Rational,
}
pub struct Lane<T: NativeFloat> {
    pub status: Status,
    // Rejected lanes deliberately expose no coefficients.
    pub values: Option<Vec<Complex<T>>>,
    pub bounds: Vec<ComponentBound>,
    pub rho: Option<Rational>,
}
#[derive(Clone, Debug, Default)]
pub struct Execution {
    pub input_rows: usize,
    pub active_rows: usize,
    pub numeric_batch_calls: usize,
    pub enclosure_batch_calls: usize,
    pub tail_batch_calls: usize,
    pub estimated_exported_instruction_lane_visits: usize,
}

#[derive(Clone)]
pub struct NearVacuumPlan {
    pub order: usize,
    pub shape: Vec<Vec<usize>>,
    pub integrated_polynomial: Atom,
    pub instructions: usize,
    exact: ExpressionEvaluator<Exact>,
    tail: ExpressionEvaluator<Exact>,
}

fn checked_program(exact: &ExpressionEvaluator<Exact>) -> Result<usize, String> {
    let exported = exact.export_instructions();
    if !exported.sub_evaluators.is_empty()
        || !exported.constant_functions.is_empty()
        || exported.instructions.iter().any(|instruction| {
            !matches!(
                instruction,
                Instruction::Add(..)
                    | Instruction::Mul(..)
                    | Instruction::Pow(..)
                    | Instruction::Assign(..)
            )
        })
    {
        return Err(
            "certificate program contains an operation outside certified rational arithmetic"
                .into(),
        );
    }
    if exported.instructions.len() > 500_000 {
        return Err("validation program exceeds the 500000-instruction preparation limit".into());
    }
    Ok(exported.instructions.len())
}

fn compile(
    expression: &Atom,
    parameters: &[Atom],
    shape: &[Vec<usize>],
) -> Result<ExpressionEvaluator<Exact>, String> {
    let exact = expression
        .evaluator(parameters)
        .direct_translation(true)
        .horner_iterations(0)
        .cores(1)
        .build()
        .map_err(|e| e.to_string())?;
    let exact = if shape.len() == 1 {
        exact
    } else {
        exact.vectorize(&Dualizer::new(
            HyperDual::<Exact>::new(shape.to_vec()),
            Vec::new(),
        ))?
    };
    checked_program(&exact)?;
    Ok(exact)
}

impl NearVacuumPlan {
    pub fn from_dual<D: DualNumberStructure>(order: usize, dual: &D) -> Result<Self, String> {
        Self::new(order, dual.get_shape().iter().map(|x| x.to_vec()).collect())
    }

    pub fn new(order: usize, shape: Vec<Vec<usize>>) -> Result<Self, String> {
        let preparation_started = std::time::Instant::now();
        if ![2, 4].contains(&order) {
            return Err("validation stage supports only orders two and four".into());
        }
        if shape.is_empty()
            || shape.len() > 32
            || shape[0].iter().any(|&n| n != 0)
            || shape.iter().any(|x| {
                x.len() != shape[0].len()
                    || x.iter()
                        .try_fold(0usize, |sum, &n| sum.checked_add(n))
                        .is_none_or(|n| n > 4)
            })
        {
            return Err("invalid or oversized validation Dual shape".into());
        }
        for (i, component) in shape.iter().enumerate() {
            if shape[..i].contains(component) {
                return Err("duplicate Dual component".into());
            }
            for axis in 0..component.len() {
                if component[axis] > 0 {
                    let mut parent = component.clone();
                    parent[axis] -= 1;
                    if !shape.contains(&parent) {
                        return Err("Dual shape is not ancestor-closed".into());
                    }
                }
            }
        }
        let parameters: Vec<_> = (0..PREPARED_ARGUMENTS)
            .map(|i| Atom::var(symbolica::symbol!(format!("iregi_near_vacuum::p{i}"))))
            .collect();
        let cube_symbols: Vec<_> = (0..3)
            .map(|i| symbolica::symbol!(format!("iregi_near_vacuum::u{i}")))
            .collect();
        let cube: Vec<_> = cube_symbols.iter().copied().map(Atom::var).collect();
        let mut remaining = Atom::num(1);
        let mut simplex = Vec::new();
        for coordinate in &cube {
            simplex.push(&remaining * coordinate);
            remaining *= Atom::num(1) - coordinate;
        }
        simplex.push(remaining);
        let field = AtomField {
            statistical_zero_test: false,
            cancel_check_on_division: false,
            custom_normalization: None,
        };
        let jacobian = Matrix::from_nested_vec(
            simplex[..3]
                .iter()
                .map(|coordinate| {
                    cube_symbols
                        .iter()
                        .map(|&variable| coordinate.derivative(variable))
                        .collect()
                })
                .collect(),
            field,
        )
        .map_err(|e| format!("Jacobian construction: {e:?}"))?
        .det()
        .map_err(|e| format!("Jacobian: {e:?}"))?;
        eprintln!(
            "near-vacuum preparation Jacobian complete: {:?}",
            preparation_started.elapsed()
        );
        let mut deviation: Atom = (0..4).map(|i| &parameters[6 + i] * &simplex[i]).sum();
        for (i, (a, b)) in [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
            .into_iter()
            .enumerate()
        {
            deviation -= &parameters[i] * &simplex[a] * &simplex[b];
        }
        let lambda_symbol = symbolica::symbol!("iregi_near_vacuum::lambda");
        let lambda = Atom::var(lambda_symbol);
        let series = (Atom::num(1) / (Atom::num(1) + &lambda * deviation).pow(2))
            .series(lambda_symbol, 0, order as u32)
            .map_err(|e| e.to_string())?
            .to_atom()
            .replace(lambda_symbol)
            .with(Atom::num(1));
        eprintln!(
            "near-vacuum preparation series complete: {:?}",
            preparation_started.elapsed()
        );
        // Exact rational coefficients avoid expression-coefficient swell.
        // Physical parameters follow the three cube variables in Symbolica's
        // variable map and remain variables after definite integration.
        let variables = Arc::new(
            cube.iter()
                .cloned()
                .map(PolyVariable::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut polynomial = (series * jacobian)
            .try_to_polynomial::<_, u16>(&Q, Some(variables))
            .map_err(|e| e.to_string())?;
        eprintln!(
            "near-vacuum preparation polynomial complete: {:?}, {} terms",
            preparation_started.elapsed(),
            polynomial.nterms()
        );
        // N4 in the complete ten-parameter basis has 103243 cube-expanded
        // monomials before exact integration reduces it to 1001 terms.
        if polynomial.nterms() > 150_000 {
            return Err("validation parameter polynomial exceeds term limit".into());
        }
        for variable in 0..3 {
            let antiderivative = polynomial.integrate(variable);
            polynomial = &antiderivative.replace(variable, &Rational::from(1))
                - &antiderivative.replace(variable, &Rational::from(0));
            eprintln!(
                "near-vacuum preparation integral{variable} complete: {:?}, {} terms",
                preparation_started.elapsed(),
                polynomial.nterms()
            );
        }
        if (0..3).any(|coordinate| polynomial.degree(coordinate) != 0) {
            return Err("parameter integration left coordinates".into());
        }
        let integrated_polynomial = polynomial.to_expression() / parameters[10].pow(2);
        eprintln!(
            "near-vacuum preparation exact expression ready: {:?}",
            preparation_started.elapsed()
        );
        let exact = compile(&integrated_polynomial, &parameters, &shape)?;
        eprintln!(
            "near-vacuum preparation compiled/dualized polynomial: {:?}",
            preparation_started.elapsed()
        );
        let instructions = checked_program(&exact)?;
        let b = Atom::var(symbolica::symbol!("iregi_near_vacuum::bound"));
        let l = Atom::var(symbolica::symbol!("iregi_near_vacuum::scale_lower"));
        let one_minus = Atom::num(1) - &b;
        let tail = b.pow((order + 1) as i64)
            * (Atom::num(order + 1) / &one_minus + Atom::num(1) / one_minus.pow(2))
            / (Atom::num(6) * l.pow(2));
        let tail = compile(&tail, &[b, l], &shape)?;
        eprintln!(
            "near-vacuum preparation tail complete: {:?}",
            preparation_started.elapsed()
        );
        Ok(Self {
            order,
            shape,
            integrated_polynomial,
            instructions,
            exact,
            tail,
        })
    }

    pub fn instantiate<T: NativeFloat>(&self, prototype: T) -> Workspace<T>
    where
        Complex<T>: EvaluationDomain,
    {
        let bits = prototype.get_precision();
        Workspace {
            plan: self.clone(),
            prototype: prototype.clone(),
            numeric: self
                .exact
                .clone()
                .map_coeff_with_prec(
                    &|x| {
                        Complex::new(
                            prototype.from_rational(&x.re),
                            prototype.from_rational(&x.im),
                        )
                    },
                    bits,
                )
                .into_batch_evaluator(),
            enclosure: self
                .exact
                .clone()
                .map_coeff_with_prec(
                    &|x| ComplexBall::from_rational_ball(x, &Rational::from(0), bits),
                    bits,
                )
                .into_batch_evaluator(),
            tail: self
                .tail
                .clone()
                .map_coeff_with_prec(
                    &|x| {
                        assert_eq!(x.im, Rational::from(0));
                        RealBall::from_rational_bounds(&x.re, &x.re, bits)
                    },
                    bits,
                )
                .into_batch_evaluator(),
            execution: Execution::default(),
        }
    }
}

pub struct Workspace<T: NativeFloat> {
    plan: NearVacuumPlan,
    prototype: T,
    numeric: BatchExpressionEvaluator<Complex<T>>,
    enclosure: BatchExpressionEvaluator<ComplexBall>,
    tail: BatchExpressionEvaluator<RealBall>,
    pub execution: Execution,
}

impl<T: NativeFloat> Workspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        rows: usize,
        accuracy: &AccuracyContext,
    ) -> Result<Vec<Lane<T>>, String> {
        let components = self.plan.shape.len();
        accuracy.validate(components)?;
        if input.len()
            != rows
                .checked_mul(PHYSICAL_ARGUMENTS * components)
                .ok_or("batch size overflow")?
        {
            return Err("invalid row-major input length".into());
        }
        self.execution = Execution {
            input_rows: rows,
            ..Execution::default()
        };
        let mut results: Vec<_> = (0..rows)
            .map(|_| Lane {
                status: Status::OutsideAnalyticDomain,
                values: None,
                bounds: Vec::new(),
                rho: None,
            })
            .collect();
        let mut active = Vec::new();
        let mut exact_arguments = Vec::new();
        let mut bounds_input = Vec::new();
        for (lane, row) in input
            .chunks_exact(PHYSICAL_ARGUMENTS * components)
            .enumerate()
        {
            let point: Option<Vec<_>> = row
                .iter()
                .map(|v| Some(Complex::new(v.re.exact_rational()?, v.im.exact_rational()?)))
                .collect();
            let Some(point) = point else {
                results[lane].status =
                    Status::InvalidInput("nonfinite or nonexact-representable input".into());
                continue;
            };
            if (0..6)
                .any(|i| (0..components).any(|j| point[i * components + j].im != Rational::from(0)))
                || (6..10).any(|i| {
                    point[i * components].re <= Rational::from(0)
                        || point[i * components].im > Rational::from(0)
                })
                || (6..10).any(|i| {
                    point[i * components].im == Rational::from(0)
                        && (1..components)
                            .any(|j| point[i * components + j].im != Rational::from(0))
                })
            {
                results[lane].status =
                    Status::InvalidInput("physical invariant/mass input domain".into());
                continue;
            }
            let mass = point[6 * components].clone();
            let lower = abs(&mass.re).max(abs(&mass.im));
            let mut envelope = Vec::new();
            for component in 0..components {
                let sigma = (0..6)
                    .map(|i| upper_abs(&point[i * components + component]))
                    .max()
                    .unwrap();
                let mu = (6..10)
                    .map(|i| {
                        upper_abs(
                            &(&point[i * components + component]
                                - if component == 0 { mass.clone() } else { zero() }),
                        )
                    })
                    .max()
                    .unwrap();
                envelope.push((mu + Rational::from((3, 8)) * sigma) / &lower);
            }
            results[lane].rho = Some(envelope[0].clone());
            if envelope[0] > Rational::from((1, 4)) {
                continue;
            }
            for argument in 0..PHYSICAL_ARGUMENTS {
                for component in 0..components {
                    let mut value = point[argument * components + component].clone();
                    if argument >= 6 && component == 0 {
                        value -= &mass;
                    }
                    exact_arguments.push(value / &mass);
                }
            }
            exact_arguments.extend(
                (0..components).map(|component| if component == 0 { mass.clone() } else { zero() }),
            );
            bounds_input.extend(envelope);
            bounds_input.extend((0..components).map(|component| {
                if component == 0 {
                    lower.clone()
                } else {
                    Rational::from(0)
                }
            }));
            active.push(lane);
        }
        let count = active.len();
        self.execution.active_rows = count;
        if count == 0 {
            return Ok(results);
        }
        let bits = self.prototype.get_precision();
        let numeric_input: Vec<_> = exact_arguments
            .iter()
            .map(|x| {
                Complex::new(
                    self.prototype.from_rational(&x.re),
                    self.prototype.from_rational(&x.im),
                )
            })
            .collect();
        let ball_input: Vec<_> = exact_arguments
            .iter()
            .map(|x| ComplexBall::from_rational_ball(x, &Rational::from(0), bits))
            .collect();
        let mut numeric_output =
            vec![Complex::new(self.prototype.zero(), self.prototype.zero()); count * components];
        let mut ball_output =
            vec![
                ComplexBall::from_rational_ball(&zero(), &Rational::from(0), bits);
                count * components
            ];
        let bound_inputs: Vec<_> = bounds_input
            .iter()
            .map(|x| RealBall::from_rational_bounds(x, x, bits))
            .collect();
        let mut tails =
            vec![
                RealBall::from_rational_bounds(&Rational::from(0), &Rational::from(0), bits);
                count * components
            ];
        self.numeric
            .evaluate_batch(&numeric_input, &mut numeric_output, count)
            .map_err(|e| e.to_string())?;
        self.execution.numeric_batch_calls = 1;
        self.enclosure
            .evaluate_batch(&ball_input, &mut ball_output, count)
            .map_err(|e| e.to_string())?;
        self.execution.enclosure_batch_calls = 1;
        self.tail
            .evaluate_batch(&bound_inputs, &mut tails, count)
            .map_err(|e| e.to_string())?;
        self.execution.tail_batch_calls = 1;
        self.execution.estimated_exported_instruction_lane_visits = self.plan.instructions * count;
        for (compact, lane) in active.into_iter().enumerate() {
            results[lane].status = Status::Success;
            for component in 0..components {
                let index = compact * components + component;
                let Some(tail) = tails[index].upper_bound().exact_rational() else {
                    results[lane].status = Status::NonFiniteOutput;
                    break;
                };
                if tail < Rational::from(0) {
                    return Err("positive tail program returned an invalid bound".into());
                }
                let value = &numeric_output[index];
                let ball = &ball_output[index];
                let raw = || -> Option<_> {
                    Some((
                        value.re.exact_rational()?,
                        value.im.exact_rational()?,
                        ball.re.lower_bound().exact_rational()?,
                        ball.re.upper_bound().exact_rational()?,
                        ball.im.lower_bound().exact_rational()?,
                        ball.im.upper_bound().exact_rational()?,
                    ))
                };
                let Some((re, im, rl, ru, il, iu)) = raw() else {
                    results[lane].status = Status::NonFiniteOutput;
                    break;
                };
                let rl = rl - &tail;
                let ru = ru + &tail;
                let il = il - &tail;
                let iu = iu + &tail;
                let real_distance = abs(&(&re - &rl)).max(abs(&(&re - &ru)));
                let imaginary_distance = abs(&(&im - &il)).max(abs(&(&im - &iu)));
                let maximum_error_squared = real_distance.pow(2) + imaginary_distance.pow(2);
                let allowed_error_squared = accuracy.absolute_per_component[component]
                    .pow(2)
                    .max(accuracy.relative.pow(2) * (re.pow(2) + im.pow(2)));
                if maximum_error_squared > allowed_error_squared
                    && results[lane].status == Status::Success
                {
                    results[lane].status = Status::UncertifiedComponent(component);
                }
                results[lane].bounds.push(ComponentBound {
                    truncation: tail,
                    real_lower: rl,
                    real_upper: ru,
                    imaginary_lower: il,
                    imaginary_upper: iu,
                    maximum_error_squared,
                    allowed_error_squared,
                });
            }
            if results[lane].status == Status::Success {
                results[lane].values =
                    Some(numeric_output[compact * components..(compact + 1) * components].to_vec());
            }
        }
        Ok(results)
    }
}
