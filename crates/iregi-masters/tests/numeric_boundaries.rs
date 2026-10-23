use iregi_masters::{
    Complex, DoubleFloat, ExactMasterPlan, Float, FloatLike, HyperDual, LaneError, MasterEvaluator,
    NativeFloat, ScalarIntegral,
};
use symbolica::{
    domains::float::{Real, SingleFloat},
    evaluate::EvaluationDomain,
};

fn assert_close<T: NativeFloat>(actual: &Complex<T>, expected: &Complex<T>, prototype: &T) {
    let tolerance = prototype
        .from_i64(2)
        .pow(u64::from(prototype.get_precision() * 2 / 3))
        .inv();
    let error =
        (actual.re.clone() - &expected.re).norm() + (actual.im.clone() - &expected.im).norm();
    assert!(
        error <= (prototype.one() + expected.re.norm() + expected.im.norm()) * tolerance,
        "actual={actual}, expected={expected}, error={error}"
    );
}

fn lightlike_jet<T: NativeFloat>(plan: &ExactMasterPlan, prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let c = |value| Complex::new(prototype.from_i64(value), prototype.zero());
    let point = [
        c(0),
        c(1),
        c(0),
        c(2),
        c(0),
        c(0),
        c(2),
        c(0),
        c(0),
        c(1),
        c(0),
        c(0),
    ];
    let mut evaluator = plan.instantiate(prototype.clone()).unwrap();
    assert_eq!(evaluator.binary_precision(), prototype.get_precision());
    let input: Vec<_> = (0..3).flat_map(|_| point.iter().cloned()).collect();
    let result = evaluator.evaluate_batch(&input, 3).unwrap();
    let expected = if plan.family() == ScalarIntegral::B0 {
        vec![-c(2).log(), c(12).inv(), c(240).inv()]
    } else {
        vec![c(12).inv(), c(120).inv(), c(1120).inv()]
    };
    for lane in 0..3 {
        let lane = result.lane(lane).unwrap();
        for (actual, expected) in lane.finite.iter().zip(&expected) {
            assert_close(actual, expected, &prototype);
        }
        assert!(lane.ir.iter().chain(lane.ir2).all(SingleFloat::is_zero));
    }
}

#[test]
fn exact_plan_reuses_lightlike_dual_instructions_across_domains_and_precisions() {
    for family in [ScalarIntegral::B0, ScalarIntegral::DB0] {
        let shape = vec![vec![0], vec![1], vec![2]];
        let plan = ExactMasterPlan::new_dual(family, HyperDual::<f64>::new(shape.clone())).unwrap();
        assert_eq!(plan.component_count(), 3);
        assert_eq!(plan.dual_shape(), Some(shape.as_slice()));
        lightlike_jet(&plan, 0_f64);
        lightlike_jet(&plan, DoubleFloat::from(0.));
        for bits in [128, 256, 512] {
            lightlike_jet(&plan, Float::new(bits));
        }
    }
}

#[test]
fn mixed_input_precisions_and_dual_seed_precisions_are_preserved_on_reuse() {
    let plan = ExactMasterPlan::new_dual(
        ScalarIntegral::A0,
        HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]),
    )
    .unwrap();
    for prototype_bits in [64, 256] {
        let mut evaluator = plan.instantiate(Float::new(prototype_bits)).unwrap();
        for precisions in [
            vec![64, 128, 256, 128, 64],
            vec![256],
            vec![],
            vec![128, 256, 64],
        ] {
            let mut input = Vec::new();
            for &bits in &precisions {
                let mass = Float::with_val(bits, 2)
                    + Float::with_val(bits, 2).pow(u64::from(bits - 10)).inv();
                input.extend([
                    Complex::new(mass, Float::new(128)),
                    Complex::new(Float::with_val(256, 2).pow(180).inv(), Float::new(64)),
                    Complex::new(Float::with_val(128, 2).pow(100).inv(), Float::new(256)),
                    Complex::new(Float::with_val(128, 1), Float::new(64)),
                    Complex::new(Float::new(64), Float::new(256)),
                    Complex::new(Float::new(256), Float::new(128)),
                ]);
            }
            let output = evaluator.evaluate_batch(&input, precisions.len()).unwrap();
            for lane in 0..precisions.len() {
                for (actual, original) in output
                    .lane(lane)
                    .unwrap()
                    .uv
                    .iter()
                    .zip(&input[lane * 6..lane * 6 + 3])
                {
                    assert_eq!(actual, original);
                    assert_eq!(actual.re.get_precision(), original.re.get_precision());
                    assert_eq!(actual.im.get_precision(), original.im.get_precision());
                }
            }
        }
    }
}

#[test]
fn branch_diagnostics_are_lane_local_and_keep_valid_neighbors() {
    let c = |value| Complex::new(value, 0.);
    let good = vec![c(-3.), c(2.), c(5.), c(1.)];
    let mut complex_invariant = good.clone();
    complex_invariant[0].im = 1.;
    let mut upper_mass = good.clone();
    upper_mass[1].im = 1.;
    let mut complex_scale = good.clone();
    complex_scale[3].im = 1.;
    let mut scalar = MasterEvaluator::new(ScalarIntegral::B0, 0.).unwrap();
    let output = scalar
        .evaluate_batch(
            &[
                good.clone(),
                complex_invariant,
                upper_mass,
                complex_scale,
                good,
            ]
            .concat(),
            5,
        )
        .unwrap();
    assert_eq!(
        output.lane(0).unwrap().finite,
        output.lane(4).unwrap().finite
    );
    assert_eq!(
        output.lane(1).unwrap_err(),
        &LaneError::ComplexInvariant { argument: 0 }
    );
    assert_eq!(
        output.lane(2).unwrap_err(),
        &LaneError::UpperHalfPlaneMass { argument: 1 }
    );
    assert_eq!(output.lane(3).unwrap_err(), &LaneError::InvalidScale);

    let good: Vec<_> = [-1., -5., -4., 2., 3., 5., 1.]
        .into_iter()
        .enumerate()
        .flat_map(|(argument, value)| [c(value), c(if argument == 0 { 1. } else { 0. })])
        .collect();
    let mut imaginary_direction = good.clone();
    imaginary_direction[1] = Complex::new(0., 1.);
    let mut width_boundary = good.clone();
    width_boundary[7] = Complex::new(0., 1.);
    let mut gram_boundary = good.clone();
    gram_boundary[2] = c(-4.);
    gram_boundary[4] = c(-9.);
    let mut triangle = MasterEvaluator::new_dual(
        ScalarIntegral::C0,
        0.,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let output = triangle
        .evaluate_batch(
            &[
                good.clone(),
                imaginary_direction,
                width_boundary,
                gram_boundary,
                good,
            ]
            .concat(),
            5,
        )
        .unwrap();
    assert_eq!(
        output.lane(0).unwrap().finite,
        output.lane(4).unwrap().finite
    );
    for lane in 1..4 {
        assert_eq!(
            output.lane(lane).unwrap_err(),
            &LaneError::UnsupportedDerivativeDomain
        );
    }
}

#[test]
fn bubble_imaginary_invariant_direction_matches_analytic_derivative() {
    let c = |value| Complex::new(value, 0.);
    let mut bubble = MasterEvaluator::new_dual(
        ScalarIntegral::B0,
        0.,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let mut derivative = MasterEvaluator::new(ScalarIntegral::DB0, 0.).unwrap();
    let points = [[c(-3.), c(2.), c(5.), c(1.)], [c(30.), c(2.), c(5.), c(1.)]];
    let inputs: Vec<_> = points
        .iter()
        .flat_map(|point| {
            point.iter().enumerate().flat_map(|(index, value)| {
                [
                    *value,
                    if index == 0 {
                        Complex::new(0., 1.)
                    } else {
                        c(0.)
                    },
                ]
            })
        })
        .collect();
    let actual = bubble.evaluate_batch(&inputs, 2).unwrap();
    let expected = derivative.evaluate_batch(&points.concat(), 2).unwrap();
    for lane in 0..2 {
        assert_close(
            &actual.lane(lane).unwrap().finite[1],
            &(expected.lane(lane).unwrap().finite[0] * Complex::new(0., 1.)),
            &0_f64,
        );
    }
}
