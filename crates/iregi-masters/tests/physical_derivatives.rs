//! Physical-sheet derivatives checked against independently evaluated native
//! scalar masters. Finite-difference weights are solved exactly by Symbolica.
use iregi_masters::{Complex, Float, HyperDual, MasterEvaluator, ScalarIntegral};
use symbolica::{
    domains::{
        float::{FloatLike, Real, SingleFloat},
        rational::{Q, Rational},
    },
    tensors::matrix::Matrix,
};
const BITS: u32 = 384;
const SHAPE: [[usize; 2]; 6] = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];
type C = Complex<Float>;
fn c(re: i64, im: i64) -> C {
    C::new(Float::with_val(BITS, re), Float::with_val(BITS, im))
}
fn fraction(n: i64, d: i64) -> C {
    c(n, 0) / c(d, 0)
}
fn size(v: &C) -> Float {
    v.re.norm() + v.im.norm()
}
fn map(v: &Complex<Rational>) -> C {
    C::new(
        v.re.to_multi_prec_float(BITS),
        v.im.to_multi_prec_float(BITS),
    )
}
fn stencil(order: usize) -> Vec<C> {
    let nodes: Vec<i64> = (-2..=2).collect();
    let matrix = Matrix::from_nested_vec(
        (0..5)
            .map(|degree| {
                nodes
                    .iter()
                    .map(|x| Rational::from(x.pow(degree)))
                    .collect()
            })
            .collect(),
        Q,
    )
    .unwrap();
    let rhs = Matrix::new_vec(
        (0..5)
            .map(|degree| Rational::from(i64::from(degree == order)))
            .collect(),
        Q,
    );
    matrix
        .solve(&rhs)
        .unwrap()
        .into_vec()
        .iter()
        .map(|v| map(&Complex::new(v.clone(), Rational::from(0))))
        .collect()
}
struct Case {
    label: &'static str,
    base: Vec<C>,
    first: Vec<C>,
    second: Vec<C>,
}
fn case(label: &'static str, values: &[i64], momentum: usize, mass: usize, complex: bool) -> Case {
    let mut base: Vec<_> = values.iter().map(|&v| c(v, 0)).collect();
    let mut first = vec![c(0, 0); base.len()];
    let mut second = first.clone();
    first[momentum] = c(1, 0);
    second[mass] = if complex { c(1, 1) / c(3, 0) } else { c(1, 0) };
    if complex {
        for k in mass..base.len() - 1 {
            base[k] = base[k].clone() - c(0, (k - mass + 1) as i64) / c(20, 0);
        }
    }
    Case {
        label,
        base,
        first,
        second,
    }
}
fn qualify(family: ScalarIntegral, cases: Vec<Case>) {
    let filter = std::env::var("IREGI_DUAL_CASE_FILTER").ok();
    let skip = std::env::var("IREGI_DUAL_SKIP_CASE").ok();
    let cases: Vec<_> = cases
        .into_iter()
        .filter(|case| {
            filter
                .as_ref()
                .is_none_or(|filter| case.label.contains(filter))
                && skip.as_ref().is_none_or(|skip| !case.label.contains(skip))
        })
        .collect();
    assert!(
        !cases.is_empty(),
        "the requested derivative fixture filter is empty"
    );
    eprintln!("preparing {} six-component batch evaluator", family.name());
    let preparation_started = std::time::Instant::now();
    let mut dual = MasterEvaluator::new_dual(
        family,
        Float::new(BITS),
        HyperDual::<f64>::new(SHAPE.map(Vec::from).to_vec()),
    )
    .unwrap();
    eprintln!(
        "prepared {} evaluator in {:?}",
        family.name(),
        preparation_started.elapsed()
    );
    let mut native =
        oneloop::NativeEvaluator::<Float>::with_binary_precision(family, BITS).unwrap();
    let weights: Vec<_> = (0..=2).map(stencil).collect();
    let mut failed = Vec::new();
    let input: Vec<_> = cases
        .iter()
        .flat_map(|fixture| {
            fixture
                .base
                .iter()
                .zip(&fixture.first)
                .zip(&fixture.second)
                .flat_map(|((p, a), b)| {
                    [p.clone(), a.clone(), b.clone(), c(0, 0), c(0, 0), c(0, 0)]
                })
        })
        .collect();
    eprintln!(
        "evaluating {} physical {} lanes",
        cases.len(),
        family.name()
    );
    let evaluation_started = std::time::Instant::now();
    let outputs = dual.evaluate_batch(&input, cases.len()).unwrap();
    eprintln!(
        "evaluated {} batch in {:?}",
        family.name(),
        evaluation_started.elapsed()
    );
    eprintln!("checking {} native finite differences", family.name());
    for (lane, fixture) in cases.into_iter().enumerate() {
        let actual = outputs
            .lane(lane)
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.label))
            .finite;
        let mut prior: Option<Float> = None;
        for step in [12, 18, 24] {
            let h = Float::with_val(BITS, 2).pow(step).inv();
            let hc = C::new(h.clone(), h.zero());
            let mut grid = Vec::new();
            for x in -2..=2 {
                for y in -2..=2 {
                    let point: Vec<_> = fixture
                        .base
                        .iter()
                        .zip(&fixture.first)
                        .zip(&fixture.second)
                        .map(|((p, a), b)| p + a * &hc * c(x, 0) + b * &hc * c(y, 0))
                        .collect();
                    let mut output = [c(0, 0), c(0, 0), c(0, 0)];
                    native.evaluate(&point, &mut output).unwrap();
                    grid.push(output[0].clone());
                }
            }
            let mut worst = Float::new(BITS);
            for (component, powers) in SHAPE.iter().enumerate() {
                let mut expected = c(0, 0);
                for x in 0..5 {
                    for y in 0..5 {
                        expected +=
                            &grid[x * 5 + y] * &weights[powers[0]][x] * &weights[powers[1]][y];
                    }
                }
                expected /= hc.pow((powers[0] + powers[1]) as u64);
                let error = size(&(&actual[component] - &expected))
                    / (Float::with_val(BITS, 1) + size(&expected));
                if !error.is_finite() || error > worst {
                    worst = error.clone();
                }
                if step == 24
                    && (!error.is_finite() || error > Float::with_val(BITS, 2).pow(70).inv())
                {
                    failed.push(format!(
                        "{} component{component}: error={error:e}, dual={}, FD={expected}",
                        fixture.label, actual[component]
                    ));
                }
            }
            eprintln!(
                "{} h=2^-{step}: worst scaled error={worst:e}",
                fixture.label
            );
            if let Some(previous) = prior {
                if worst > Float::with_val(BITS, 2).pow(90).inv()
                    && worst.clone() * Float::with_val(BITS, 100) > previous
                {
                    failed.push(format!("{} no convergence at {step}", fixture.label));
                }
            }
            prior = Some(worst);
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
#[test]
fn physical_triangle_first_second_and_mixed_derivatives() {
    let mut reference = case("C0 complex reference", &[1, 2, 3, 1, 1, 1, 1], 0, 3, true);
    reference.base[3] = fraction(1, 2) - c(0, 1) / c(100, 0);
    reference.base[4] = fraction(7, 10) - c(0, 2) / c(100, 0);
    reference.base[5] = fraction(7, 5) - c(0, 5) / c(100, 0);
    qualify(
        ScalarIntegral::C0,
        vec![
            case("C0 subthreshold", &[1, 2, 3, 5, 7, 11, 1], 0, 3, false),
            case("C0 absorptive", &[-1, -2, 30, 2, 3, 5, 1], 2, 3, false),
            reference,
            case(
                "C0 complex absorptive",
                &[-1, -2, 30, 2, 3, 5, 1],
                2,
                3,
                true,
            ),
        ],
    );
}

#[test]
fn physical_bubble_pseudothreshold_first_second_and_mixed_derivatives() {
    for family in [ScalarIntegral::B0, ScalarIntegral::DB0] {
        qualify(
            family,
            vec![case("massive pseudothreshold", &[1, 4, 9, 1], 0, 1, false)],
        );
    }
}
#[test]
fn physical_box_first_second_and_mixed_derivatives() {
    let mut reference = case(
        "D0 equal mass absorptive",
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        4,
        6,
        false,
    );
    for (i, (n, d)) in [
        (2171, 144),
        (211, 16),
        (-5, 4),
        (-587, 72),
        (-311, 72),
        (27, 16),
    ]
    .into_iter()
    .enumerate()
    {
        reference.base[i] = fraction(n, d);
    }
    let mut equal_complex = case(
        "D0 equal complex masses",
        &[-1, -2, -3, -4, 12, -5, 2, 2, 2, 2, 1],
        4,
        8,
        true,
    );
    for mass in &mut equal_complex.base[6..10] {
        *mass = c(2, 0) - c(0, 1) / c(10, 0);
    }
    qualify(
        ScalarIntegral::D0,
        vec![
            case(
                "D0 subthreshold",
                &[1, 1, 1, 1, 8, -2, 20, 30, 40, 50, 1],
                4,
                6,
                false,
            ),
            case(
                "D0 absorptive",
                &[1, 1, 1, 1, 40, -8, 2, 3, 4, 5, 1],
                4,
                6,
                false,
            ),
            case(
                "D0 complex absorptive",
                &[-1, -2, -3, -4, 12, -5, 2, 3, 4, 5, 1],
                4,
                6,
                true,
            ),
            reference,
            equal_complex,
        ],
    );
}

#[test]
fn analytic_grouping_preserves_zero_primal_jets() {
    let context = oneloop::OneLoopExpressions::new();
    let x = symbolica::symbol!("physical_grouping_x").to_atom();
    let expression = symbolica::atom::Symbol::IF.call((&x, &x, 0));
    let mut evaluator = context
        .dual_evaluator(
            &[expression],
            &[x],
            HyperDual::<f64>::new(SHAPE.map(Vec::from).to_vec()),
        )
        .unwrap()
        .map_coeff_with_prec(&map, BITS)
        .into_batch_evaluator();
    let input = vec![c(0, 0), c(1, 2), c(3, 4), c(5, 6), c(7, 8), c(9, 10)];
    let mut actual = vec![c(0, 0); SHAPE.len()];
    evaluator.evaluate_batch(&input, &mut actual, 1).unwrap();
    assert_eq!(
        actual, input,
        "an algebraic identity must preserve every Taylor component at zero"
    );
    // A genuine branch is not an analytic identity and must remain untouched.
    let x = symbolica::symbol!("physical_grouping_x").to_atom();
    let conditional = symbolica::atom::Symbol::IF.call((&x, &x * symbolica::atom::Atom::num(2), 0));
    let mut conditional = context
        .dual_evaluator(
            &[conditional],
            &[x.clone()],
            HyperDual::<f64>::new(SHAPE.map(Vec::from).to_vec()),
        )
        .unwrap()
        .map_coeff_with_prec(&map, BITS)
        .into_batch_evaluator();
    conditional.evaluate_batch(&input, &mut actual, 1).unwrap();
    assert!(actual.iter().all(|v| v.re.is_zero() && v.im.is_zero()));

    // Retain the arithmetic grouping used to compare precisely equal large
    // masses while keeping an independently represented tiny invariant.
    let y = symbolica::symbol!("physical_grouping_y").to_atom();
    let p = symbolica::symbol!("physical_grouping_p").to_atom();
    let difference = &x - &y;
    let grouped = symbolica::atom::Symbol::IF.call((&difference, &difference, 0)) - &p;
    let mut grouped = context
        .dual_evaluator(
            &[grouped],
            &[x, y, p],
            HyperDual::<f64>::new(SHAPE.map(Vec::from).to_vec()),
        )
        .unwrap()
        .map_coeff_with_prec(&map, BITS)
        .into_batch_evaluator();
    let mut arguments = vec![c(0, 0); 3 * SHAPE.len()];
    arguments[0] = c(29929, 0);
    arguments[SHAPE.len()] = c(29929, 0);
    arguments[1] = c(1, 0);
    arguments[SHAPE.len() + 2] = c(1, 0);
    let tiny = c(2, 0).pow(114).inv();
    arguments[2 * SHAPE.len()] = tiny.clone();
    grouped.evaluate_batch(&arguments, &mut actual, 1).unwrap();
    assert_eq!(actual[0], -tiny);
    assert_eq!(actual[1], c(1, 0));
    assert_eq!(actual[2], c(-1, 0));
    assert!(actual[3..].iter().all(|v| v.re.is_zero() && v.im.is_zero()));
}
