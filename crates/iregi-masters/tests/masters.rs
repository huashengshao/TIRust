use iregi_masters::{
    Complex, DoubleFloat, Float, HyperDual, LaneError, MasterBackend, MasterEvaluator, NativeFloat,
    ScalarIntegral,
};
use symbolica::{
    domains::float::{FloatLike, Real, SingleFloat},
    evaluate::EvaluationDomain,
};

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn repeat<T: Clone>(point: &[T], rows: usize) -> Vec<T> {
    (0..rows).flat_map(|_| point.iter().cloned()).collect()
}

fn close(actual: Complex<f64>, expected: Complex<f64>) {
    let error = (actual.re - expected.re).abs() + (actual.im - expected.im).abs();
    assert!(
        error <= 2.0e-10 * (1. + expected.re.abs() + expected.im.abs()),
        "actual={actual}, expected={expected}, error={error:e}"
    );
}

#[test]
fn scalar_families_match_direct_native_and_isolate_bad_lanes() {
    let fixtures = [
        (ScalarIntegral::A0, vec![2., 1.]),
        (ScalarIntegral::B0, vec![-3., 2., 5., 1.]),
        (ScalarIntegral::DB0, vec![-3., 2., 5., 1.]),
        (ScalarIntegral::C0, vec![-1., -5., -4., 2., 3., 5., 1.]),
        (
            ScalarIntegral::D0,
            vec![-1., -2., -3., -4., -5., -6., 2., 3., 4., 5., 1.],
        ),
    ];
    for (family, point) in fixtures {
        let point: Vec<_> = point.into_iter().map(c).collect();
        let mut native = oneloop::NativeEvaluator::<f64>::new(family).unwrap();
        let mut expected = [c(0.); 3];
        native.evaluate(&point, &mut expected).unwrap();
        let mut evaluator = MasterEvaluator::new(family, 0.).unwrap();
        let mut interpreted =
            MasterEvaluator::new_with_backend(family, 0., MasterBackend::SymbolicaBatch).unwrap();
        for rows in [0, 1, 3, 5, 31] {
            let mut input: Vec<_> = (0..rows).flat_map(|_| point.clone()).collect();
            if rows > 1 {
                input[2 * point.len() - 1] = c(-1.);
            }
            let batch = evaluator.evaluate_batch(&input, rows).unwrap();
            let interpreted = interpreted.evaluate_batch(&input, rows).unwrap();
            assert_eq!(batch.errors(), interpreted.errors());
            assert_eq!(batch.row_count(), rows);
            assert_eq!(batch.component_count(), 1);
            for lane in 0..rows {
                if lane == 1 {
                    assert_eq!(batch.lane(lane).unwrap_err(), &LaneError::InvalidScale);
                } else {
                    let actual = batch.lane(lane).unwrap();
                    close(actual.finite[0], interpreted.lane(lane).unwrap().finite[0]);
                    close(
                        actual.total_single[0],
                        interpreted.lane(lane).unwrap().total_single[0],
                    );
                    close(actual.finite[0], expected[0]);
                    close(actual.total_single[0], expected[1]);
                    close(actual.ir2[0], expected[2]);
                    close(actual.uv[0] + actual.ir[0], expected[1]);
                }
            }
        }
    }
}

#[test]
fn mixed_branches_and_scaleless_uv_ir_are_preserved() {
    let rows = [
        [0., 0., 0., 1.],
        [0., 2., 2., 1.],
        [-3., 2., 5., 1.],
        [7., 0., 0., 1.],
        [0., 0., 0., 1.],
    ];
    let mut evaluator = MasterEvaluator::new(ScalarIntegral::B0, 0.).unwrap();
    let input: Vec<_> = rows.iter().flatten().copied().map(c).collect();
    let batch = evaluator.evaluate_batch(&input, rows.len()).unwrap();
    let mut native = oneloop::NativeEvaluator::<f64>::new(ScalarIntegral::B0).unwrap();
    for (lane, row) in rows.iter().enumerate() {
        let mut expected = [c(0.); 3];
        native.evaluate(&row.map(c), &mut expected).unwrap();
        let actual = batch.lane(lane).unwrap();
        close(actual.finite[0], expected[0]);
        close(actual.total_single[0], expected[1]);
        assert_eq!(actual.uv, &[c(1.)]);
    }
    for lane in [0, 4] {
        let actual = batch.lane(lane).unwrap();
        assert_eq!(actual.finite, &[c(0.)]);
        assert_eq!(actual.total_single, &[c(0.)]);
        assert_eq!(actual.ir, &[c(-1.)]);
    }
    for family in [ScalarIntegral::A0, ScalarIntegral::C0, ScalarIntegral::D0] {
        let mut evaluator = MasterEvaluator::new(family, 0.).unwrap();
        let mut input = vec![c(0.); family.arity() * 3];
        for row in input.chunks_exact_mut(family.arity()) {
            row[family.arity() - 1] = c(1.);
        }
        let batch = evaluator.evaluate_batch(&input, 3).unwrap();
        for lane in 0..3 {
            let result = batch.lane(lane).unwrap();
            assert_eq!(result.finite, &[c(0.)]);
            assert_eq!(result.uv, &[c(0.)]);
            assert_eq!(result.ir, &[c(0.)]);
        }
    }
}

#[test]
fn doubledouble_and_arbitrary_precision_inputs_do_not_round_through_f64() {
    let one = DoubleFloat::from(1.);
    let delta = one.from_usize(2).pow(80).inv();
    let mass = one + delta;
    let zero = DoubleFloat::from(0.);
    let mut evaluator = MasterEvaluator::new(ScalarIntegral::A0, zero).unwrap();
    let point = [Complex::new(mass, zero), Complex::new(one, zero)];
    let batch = evaluator.evaluate_batch(&repeat(&point, 3), 3).unwrap();
    for lane in 0..3 {
        assert_eq!(batch.lane(lane).unwrap().uv[0].re, mass);
        assert_ne!(batch.lane(lane).unwrap().uv[0].re, one);
    }

    for bits in [128, 256, 512] {
        let zero = Float::new(bits);
        let one = Float::with_val(bits, 1);
        let mass = one.clone() + Float::with_val(bits, 2).pow(100).inv();
        let point = [
            Complex::new(mass.clone(), zero.clone()),
            Complex::new(one.clone(), zero.clone()),
        ];
        let mut evaluator = MasterEvaluator::new(ScalarIntegral::A0, zero).unwrap();
        let batch = evaluator.evaluate_batch(&repeat(&point, 3), 3).unwrap();
        for lane in 0..3 {
            let actual = batch.lane(lane).unwrap();
            assert_eq!(actual.uv[0].re, mass);
            assert_ne!(actual.uv[0].re, one);
            assert!(actual.finite[0].re.is_finite());
            let expected = mass.clone() * (one.clone() - mass.log());
            let error = (actual.finite[0].re.clone() - expected).norm();
            assert!(error < Float::with_val(bits, 2).pow((bits / 2).into()).inv());
        }
    }
}

fn check_precise_dual<T: NativeFloat>(prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let z = || Complex::new(prototype.zero(), prototype.zero());
    let c = |value| Complex::new(prototype.from_i64(value), prototype.zero());
    let mut evaluator = MasterEvaluator::new_dual(
        ScalarIntegral::A0,
        prototype.clone(),
        HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]),
    )
    .unwrap();
    let point = [c(2), c(1), z(), c(1), z(), z()];
    let batch = evaluator.evaluate_batch(&repeat(&point, 3), 3).unwrap();
    let expected = -prototype.from_i64(2).log();
    let tolerance = prototype
        .from_i64(2)
        .pow(u64::from(prototype.get_precision() / 2))
        .inv();
    for lane in 0..3 {
        let actual = batch.lane(lane).unwrap();
        let error = (actual.finite[1].re.clone() - &expected).norm();
        assert!(error < tolerance, "derivative error={error}");
        assert_eq!(actual.finite[2].re, -prototype.from_i64(4).inv());
        assert!(actual.ir.iter().all(SingleFloat::is_zero));
    }
}

#[test]
fn dual_transcendentals_keep_doublefloat_and_arbitrary_precision() {
    check_precise_dual(DoubleFloat::from(0.));
    for bits in [128, 256, 512] {
        check_precise_dual(Float::new(bits));
    }
}

fn check_generic_master_families<T: NativeFloat>(prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let fixtures = [
        (ScalarIntegral::A0, vec![2, 1]),
        (ScalarIntegral::B0, vec![-3, 2, 5, 1]),
        (ScalarIntegral::C0, vec![-1, -5, -4, 2, 3, 5, 1]),
        (
            ScalarIntegral::D0,
            vec![-1, -2, -3, -4, -5, -6, 2, 3, 4, 5, 1],
        ),
    ];
    let bits = prototype.get_precision();
    let tolerance = prototype.from_i64(2).pow(u64::from(bits * 2 / 3)).inv();
    for (family, point) in fixtures {
        let point: Vec<_> = point
            .into_iter()
            .map(|value| Complex::new(prototype.from_i64(value), prototype.zero()))
            .collect();
        let mut native =
            oneloop::NativeEvaluator::<T>::with_binary_precision(family, bits).unwrap();
        let mut expected = vec![Complex::new(prototype.zero(), prototype.zero()); 3];
        native.evaluate(&point, &mut expected).unwrap();
        let mut evaluator = MasterEvaluator::new(family, prototype.clone()).unwrap();
        let batch = evaluator.evaluate_batch(&repeat(&point, 3), 3).unwrap();
        for lane in 0..3 {
            let actual = batch.lane(lane).unwrap();
            for (actual, expected) in [&actual.finite[0], &actual.total_single[0], &actual.ir2[0]]
                .into_iter()
                .zip(&expected)
            {
                let error = (actual.re.clone() - &expected.re).norm()
                    + (actual.im.clone() - &expected.im).norm();
                let scale = prototype.one() + expected.re.norm() + expected.im.norm();
                assert!(
                    error <= scale * &tolerance,
                    "{} at {bits} bits: error={error}",
                    family.name()
                );
            }
        }
    }
}

#[test]
fn all_master_families_preserve_doublefloat_and_float_numerics() {
    check_generic_master_families(DoubleFloat::from(0.));
    check_generic_master_families(Float::new(256));
}

symbolica::create_hyperdual_from_components!(StaticDual, [[0], [1], [2]]);

#[test]
fn static_dynamic_dual_agree_and_keep_taylor_components() {
    let shape = vec![vec![0], vec![1], vec![2]];
    let mass = StaticDual::new_variable(0, c(2.));
    let mut scale = StaticDual::new_variable(0, c(1.));
    scale.values[1] = c(0.);
    let mut static_evaluator =
        MasterEvaluator::new_dual(ScalarIntegral::A0, 0., mass.clone()).unwrap();
    let static_result = static_evaluator
        .evaluate_dual_batch(
            &[mass.clone(), scale.clone(), mass.clone(), scale.clone()],
            2,
            |value| &value.values,
        )
        .unwrap();
    let mass = HyperDual::from_values(shape.clone(), mass.values.to_vec());
    let scale = HyperDual::from_values(shape.clone(), scale.values.to_vec());
    let mut dynamic_evaluator =
        MasterEvaluator::new_dual(ScalarIntegral::A0, 0., mass.clone()).unwrap();
    let dynamic_result = dynamic_evaluator
        .evaluate_hyperdual_batch(&[mass.clone(), scale.clone(), mass, scale], 2)
        .unwrap();
    for lane in 0..2 {
        let actual = dynamic_result.lane(lane).unwrap();
        assert_eq!(actual.finite, static_result.lane(lane).unwrap().finite);
        close(actual.finite[0], c(2. * (1. - 2_f64.ln())));
        close(actual.finite[1], c(-2_f64.ln()));
        close(actual.finite[2], c(-0.25));
        assert_eq!(actual.uv, &[c(2.), c(1.), c(0.)]);
        assert_eq!(actual.ir, &[c(0.); 3]);
    }
    let mass = StaticDual::new_variable(0, 2.);
    let mut scale = StaticDual::new_variable(0, 1.);
    scale.values[1] = 0.;
    let real = static_evaluator
        .evaluate_real_dual_batch(&[mass, scale], 1, |value| &value.values)
        .unwrap();
    assert_eq!(
        real.lane(0).unwrap().finite,
        static_result.lane(0).unwrap().finite
    );
}

#[test]
fn finite_triangle_and_box_dual_homogeneity() {
    for (family, point, degree) in [
        (ScalarIntegral::C0, vec![-1., -5., -4., 2., 3., 5., 1.], -1.),
        (
            ScalarIntegral::D0,
            vec![-1., -2., -3., -4., -5., -6., 2., 3., 4., 5., 1.],
            -2.,
        ),
    ] {
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1]]);
        let mut evaluator = MasterEvaluator::new_dual(family, 0., shape).unwrap();
        let input: Vec<_> = point
            .iter()
            .enumerate()
            .flat_map(|(i, &value)| [c(value), c(if i + 1 == point.len() { 0. } else { value })])
            .collect();
        let batch = evaluator.evaluate_batch(&input.repeat(3), 3).unwrap();
        for lane in 0..3 {
            let result = batch.lane(lane).unwrap();
            close(result.finite[1], result.finite[0] * c(degree));
            assert!(result.uv.iter().all(SingleFloat::is_zero));
            assert!(result.ir.iter().all(SingleFloat::is_zero));
        }
    }
}

#[test]
fn invalid_components_and_undefined_derivatives_are_explicit() {
    let mut evaluator = MasterEvaluator::new_dual(
        ScalarIntegral::B0,
        0.,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let rows = [
        [c(0.), c(1.), c(0.), c(0.), c(0.), c(0.), c(1.), c(0.)],
        [
            c(-1.),
            c(0.),
            c(2.),
            c(f64::NAN),
            c(3.),
            c(0.),
            c(1.),
            c(0.),
        ],
    ];
    let batch = evaluator.evaluate_batch(&rows.concat(), 2).unwrap();
    assert_eq!(
        batch.lane(0).unwrap_err(),
        &LaneError::UnsupportedDerivativeDomain
    );
    assert_eq!(
        batch.lane(1).unwrap_err(),
        &LaneError::NonFiniteInput {
            argument: 1,
            component: 1
        }
    );
    let mut derivative = MasterEvaluator::new(ScalarIntegral::DB0, 0.).unwrap();
    let batch = derivative
        .evaluate_batch(&[c(0.), c(0.), c(0.), c(1.)], 1)
        .unwrap();
    assert_eq!(
        batch.lane(0).unwrap_err(),
        &LaneError::UndefinedScalelessDerivative
    );
    assert!(matches!(
        derivative.evaluate_batch(&[], usize::MAX),
        Err(iregi_masters::MasterError::SizeOverflow)
    ));
}

#[test]
fn lightlike_bubble_jets_and_threshold_errors_are_preserved() {
    let mut evaluator = MasterEvaluator::new_dual(
        ScalarIntegral::B0,
        0.,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let rows = [
        [c(0.), c(1.), c(2.), c(0.), c(2.), c(0.), c(1.), c(0.)],
        [c(0.), c(0.), c(2.), c(0.), c(2.), c(1.), c(1.), c(0.)],
        [c(0.), c(0.), c(2.), c(1.), c(2.), c(1.), c(1.), c(0.)],
        [c(0.), c(1.), c(0.), c(0.), c(2.), c(0.), c(1.), c(0.)],
        [c(0.), c(1.), c(2.), c(0.), c(5.), c(0.), c(1.), c(0.)],
        [c(0.), c(0.), c(0.), c(1.), c(2.), c(0.), c(1.), c(0.)],
    ];
    let batch = evaluator
        .evaluate_batch(&rows.concat(), rows.len())
        .unwrap();
    close(batch.lane(0).unwrap().finite[1], c(1. / 12.));
    close(batch.lane(1).unwrap().finite[1], c(-0.25));
    close(batch.lane(2).unwrap().finite[1], c(-0.5));
    close(batch.lane(3).unwrap().finite[1], c(0.25));
    close(
        batch.lane(4).unwrap().finite[1],
        c(7. / 18. - 10. * 2.5_f64.ln() / 27.),
    );
    assert_eq!(
        batch.lane(5).unwrap_err(),
        &LaneError::UnsupportedDerivativeDomain
    );

    let mut threshold = MasterEvaluator::new_dual(
        ScalarIntegral::DB0,
        0.,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let lightlike = [c(0.), c(1.), c(2.), c(0.), c(2.), c(0.), c(1.), c(0.)];
    let result = threshold.evaluate_batch(&lightlike, 1).unwrap();
    close(result.lane(0).unwrap().finite[0], c(1. / 12.));
    close(result.lane(0).unwrap().finite[1], c(1. / 120.));
    let point = [c(8.), c(1.), c(2.), c(0.), c(2.), c(0.), c(1.), c(0.)];
    let result = threshold.evaluate_batch(&point, 1).unwrap();
    assert_eq!(
        result.lane(0).unwrap_err(),
        &LaneError::UnsupportedDerivativeDomain
    );
}

#[test]
fn soft_box_relabelling_survives_mixed_batches_and_constant_duals() {
    // Three-mass fixture and relabelling from oneloopmaster's independently
    // checked d0_infrared_relabelling regression (mu² = 10^6).
    let soft = [
        90000., 29929., 500000., 300000., 20000., 6464., 8315., 0., 29929., 6464., 1e6,
    ]
    .map(c);
    let mut relabelled = soft;
    assert!(oneloop::regular_d0_labelling(&mut relabelled));
    let ordinary = [-1., -2., -3., -4., -5., -6., 2., 3., 4., 5., 1.].map(c);
    let mut invalid = ordinary;
    invalid[10] = c(-1.);
    let input = [soft, ordinary, invalid, soft].concat();
    let mut evaluator = MasterEvaluator::new(ScalarIntegral::D0, 0.).unwrap();
    let actual = evaluator.evaluate_batch(&input, 4).unwrap();
    assert_eq!(actual.lane(2).unwrap_err(), &LaneError::InvalidScale);
    let mut native = oneloop::NativeEvaluator::<f64>::new(ScalarIntegral::D0).unwrap();
    let mut expected = [c(0.); 3];
    native.evaluate(&soft, &mut expected).unwrap();
    let reference = [
        Complex::new(-6.595062782740979e-10, 4.402997824384324e-11),
        Complex::new(-9.273699493321599e-11, 8.310761273761105e-11),
    ];
    for lane in [0, 3] {
        let result = actual.lane(lane).unwrap();
        for (coefficient, observed) in [result.finite[0], result.total_single[0]]
            .into_iter()
            .enumerate()
        {
            for expected in [expected[coefficient], reference[coefficient]] {
                let error = (observed.re - expected.re).abs() + (observed.im - expected.im).abs();
                assert!(
                    error < 2e-9 * (expected.re.abs() + expected.im.abs()),
                    "error={error:e}"
                );
            }
        }
        assert_eq!(result.uv, &[c(0.)]);
    }
    let mut dual = MasterEvaluator::new_dual(
        ScalarIntegral::D0,
        0.,
        HyperDual::<f64>::new(vec![vec![0], vec![1]]),
    )
    .unwrap();
    let flattened: Vec<_> = input.iter().flat_map(|&primal| [primal, c(0.)]).collect();
    let dual = dual.evaluate_batch(&flattened, 4).unwrap();
    for lane in [0, 1, 3] {
        let result = dual.lane(lane).unwrap();
        let expected = actual.lane(lane).unwrap().finite[0];
        let error =
            (result.finite[0].re - expected.re).abs() + (result.finite[0].im - expected.im).abs();
        assert!(error < 3e-14 * (expected.re.abs() + expected.im.abs()));
        assert_eq!(result.finite[1], c(0.));
    }
}
