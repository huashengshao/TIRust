use iregi_masters::{
    Complex, DoubleFloat, ExactMasterPlan, Float, FloatLike, HyperDual, LaneError, NativeFloat,
    ScalarIntegral,
};
use symbolica::{domains::float::Real, evaluate::EvaluationDomain};

fn scaled_chart<T: NativeFloat>(prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
    let z = || Complex::new(prototype.zero(), prototype.zero());
    let c = |value: T| Complex::new(value, prototype.zero());
    for family in [ScalarIntegral::B0, ScalarIntegral::DB0] {
        let plan = ExactMasterPlan::new_dual(family, shape.clone()).unwrap();
        let mut evaluator = plan.instantiate(prototype.clone()).unwrap();
        let delta = prototype.from_i64(2).pow(40).inv();
        let mut scales = vec![prototype.one()];
        for scale in [
            prototype.from_i64(2).pow(600).inv(),
            prototype.from_i64(2).pow(600),
        ] {
            scales.push(scale.clone() * (prototype.one() - &delta));
            scales.push(scale.clone());
            scales.push(scale * (prototype.one() + &delta));
        }
        let input: Vec<_> = scales
            .iter()
            .flat_map(|scale| {
                [
                    c(scale.clone()),
                    c(scale.clone()),
                    z(),
                    c(prototype.from_i64(4) * scale),
                    z(),
                    z(),
                    c(prototype.from_i64(9) * scale),
                    z(),
                    z(),
                    c(scale.clone()),
                    z(),
                    z(),
                ]
            })
            .collect();
        let batch = evaluator.evaluate_batch(&input, scales.len()).unwrap();
        let baseline = batch.lane(0).unwrap();
        let tolerance = prototype
            .from_i64(2)
            .pow(u64::from(prototype.get_precision() / 2))
            .inv();
        for (lane, scale) in scales.iter().enumerate() {
            let actual = batch.lane(lane).unwrap_or_else(|error| {
                panic!(
                    "{} at {} bits, scale {scale}: {error}",
                    family.name(),
                    prototype.get_precision()
                )
            });
            for (value, expected) in actual.finite.iter().zip(baseline.finite) {
                let rescaled = if family == ScalarIntegral::DB0 {
                    value * &c(scale.clone())
                } else {
                    value.clone()
                };
                let error =
                    (rescaled.re - &expected.re).norm() + (rescaled.im - &expected.im).norm();
                assert!(error < tolerance, "{} scale{scale}: {error}", family.name());
            }
        }
    }
}

#[test]
fn normalized_float_jets_keep_mixed_input_and_seed_precisions() {
    let plan = ExactMasterPlan::new_dual(
        ScalarIntegral::B0,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let mut evaluator = plan.instantiate(Float::new(128)).unwrap();
    for tiny in [false, true] {
        let scaled = |bits, integer| {
            let two = Float::with_val(bits, 2);
            let scale = if tiny {
                two.inv().pow(600)
            } else {
                two.pow(600)
            };
            Complex::new(Float::with_val(bits, integer) * scale, Float::new(bits))
        };
        let zero = |bits| Complex::new(Float::new(bits), Float::new(bits));
        let input = [
            scaled(128, 1),
            scaled(256, 1),
            scaled(64, 4),
            zero(192),
            scaled(192, 9),
            zero(64),
            scaled(128, 1),
            zero(256),
        ];
        let before = input.clone();
        let output = evaluator.evaluate_batch(&input, 1).unwrap();
        let actual = output.lane(0).unwrap();
        let expected = Float::with_val(128, 5)
            * (Float::with_val(128, 3) / Float::with_val(128, 2)).log()
            - Float::with_val(128, 2);
        let error = (actual.finite[1].re.clone() - expected).norm();
        assert!(error < Float::with_val(128, 2).pow(40).inv(), "{error}");
        for (value, original) in input.iter().zip(&before) {
            assert_eq!(value, original);
            assert_eq!(value.re.get_precision(), original.re.get_precision());
            assert_eq!(value.im.get_precision(), original.im.get_precision());
        }
    }
}

#[test]
fn pseudothreshold_chart_is_scale_safe_across_numeric_domains() {
    scaled_chart(0.0_f64);
    scaled_chart(DoubleFloat::from(0.));
    scaled_chart(Float::new(256));
}

#[test]
fn genuine_threshold_derivatives_fail_and_constant_duals_use_scalar_values() {
    for family in [ScalarIntegral::B0, ScalarIntegral::DB0] {
        let plan = ExactMasterPlan::new_dual(family, HyperDual::<f64>::new(vec![vec![0], vec![1]]))
            .unwrap();
        let mut evaluator = plan.instantiate(0.0).unwrap();
        let row = |s, seed| [s, seed, 4., 0., 9., 0., 1., 0.].map(|value| Complex::new(value, 0.));
        let input = [row(1., 1.), row(25., 1.), row(25., 0.)].concat();
        let batch = evaluator.evaluate_batch(&input, 3).unwrap();
        assert!(batch.lane(0).is_ok());
        assert_eq!(
            batch.lane(1).unwrap_err(),
            &LaneError::UnsupportedDerivativeDomain
        );
        let mut native = oneloop::NativeEvaluator::<f64>::new(family).unwrap();
        let mut expected = [Complex::new(0., 0.); 3];
        native
            .evaluate(
                &[25., 4., 9., 1.].map(|value| Complex::new(value, 0.)),
                &mut expected,
            )
            .unwrap();
        let constant = batch.lane(2).unwrap();
        assert!((constant.finite[0].re - expected[0].re).abs() < 1e-12);
        assert_eq!(constant.finite[1], Complex::new(0., 0.));
    }
}

fn separated_scales<T: NativeFloat>(prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let plan = ExactMasterPlan::new_dual(
        ScalarIntegral::B0,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let mut evaluator = plan.instantiate(prototype.clone()).unwrap();
    let two = prototype.from_i64(2);
    let ratio_log = (prototype.from_i64(4) / prototype.from_i64(9)).log();
    let expected_seed = prototype.from_i64(5)
        * (prototype.from_i64(3) / prototype.from_i64(2)).log()
        - prototype.from_i64(2);
    for exponent in [-1070, 1000] {
        let scale = if exponent < 0 {
            two.inv().pow((-exponent) as u64)
        } else {
            two.pow(exponent as u64)
        };
        let z = prototype.zero();
        let input = [
            scale.clone(),
            scale.clone(),
            prototype.from_i64(4) * &scale,
            z.clone(),
            prototype.from_i64(9) * &scale,
            z.clone(),
            prototype.one(),
            z.clone(),
        ]
        .map(|value| Complex::new(value, z.clone()));
        let before = input.clone();
        let batch = evaluator.evaluate_batch(&input, 1).unwrap();
        let lane = batch.lane(0).unwrap_or_else(|error| {
            panic!(
                "{} bits, exponent{exponent}: {error}",
                prototype.get_precision()
            )
        });
        let expected = prototype.from_i64(2)
            - prototype.from_i64(4).log()
            - prototype.from_i64(exponent) * two.log()
            + prototype.from_i64(3) * &ratio_log;
        let tolerance = two.pow((prototype.get_precision() / 3) as u64).inv();
        assert!((lane.finite[0].re.clone() - expected).norm() < tolerance);
        assert!((lane.finite[1].re.clone() - &expected_seed).norm() < tolerance);
        assert_eq!(input, before);
    }
}

#[test]
fn kinematic_and_renormalization_scales_can_have_extreme_separation() {
    separated_scales(0.0_f64);
    separated_scales(DoubleFloat::from(0.));
    separated_scales(Float::new(256));
}

#[test]
fn scale_only_jets_retain_scalar_thresholds_and_infrared_laurent_transport() {
    let shape = HyperDual::<f64>::new(vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0],
        vec![1, 1],
        vec![0, 2],
    ]);
    for (family, point) in [
        (ScalarIntegral::A0, vec![3., 1.]),
        (ScalarIntegral::B0, vec![25., 4., 9., 1.]),
        (ScalarIntegral::DB0, vec![25., 4., 9., 1.]),
        (ScalarIntegral::C0, vec![0., 0., -3., 0., 0., 0., 1.]),
        (
            ScalarIntegral::D0,
            vec![0., 0., 0., 0., -3., -4., 0., 0., 0., 0., 1.],
        ),
    ] {
        let plan = ExactMasterPlan::new_dual(family, shape.clone()).unwrap();
        let mut evaluator = plan.instantiate(0.0).unwrap();
        let mut input = vec![Complex::new(0., 0.); point.len() * 6];
        for (argument, value) in point.iter().enumerate() {
            input[argument * 6].re = *value;
        }
        let offset = (point.len() - 1) * 6;
        input[offset + 1].re = 1.;
        input[offset + 2].re = 1.;
        let batch = evaluator.evaluate_batch(&input, 1).unwrap();
        let lane = batch
            .lane(0)
            .unwrap_or_else(|error| panic!("{}: {error}", family.name()));
        let single = lane.total_single[0];
        let double = lane.ir2[0];
        let expected = [
            lane.finite[0],
            single,
            single,
            (double - single) / 2.,
            double - single,
            (double - single) / 2.,
        ];
        for (actual, expected) in lane.finite.iter().zip(expected) {
            assert!(
                (actual.re - expected.re).abs() + (actual.im - expected.im).abs() < 2e-12,
                "{}: {actual} != {expected}",
                family.name()
            );
        }
        assert_eq!(lane.total_single[1], double);
        assert_eq!(lane.uv[1], Complex::new(0., 0.));
    }
}
