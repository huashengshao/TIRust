//! Public-constructor regression controls, with private inspection of the sixth
//! domain output and lazy Dual child. These contain no copied kernel arithmetic.
use super::*;
use oneloop::rust_codegen::runtime::BatchExecutionPolicy;
use symbolica::domains::float::Real;

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn metadata<T: NativeFloat>(values: &[Complex<T>]) -> Vec<(Option<String>, u32, bool)> {
    values.iter().flat_map(|z| [&z.re, &z.im]).map(|x| {
        let rational = x.exact_rational().map(|value| value.to_string());
        assert_eq!(rational.is_some(), x.is_finite());
        (rational, x.get_precision(), x.negative_sign())
    }).collect()
}

fn snapshot<T: NativeFloat>(
    batch: &MasterBatch<T>,
    components: usize,
    expected: &[Option<LaneError>],
) -> Vec<Option<Vec<(Option<String>, u32, bool)>>> {
    assert_eq!(batch.errors(), expected);
    assert_eq!(batch.row_count(), expected.len());
    assert_eq!(batch.component_count(), components);
    assert_eq!(batch.values.len(), expected.len() * OUTPUTS * components);
    expected.iter().enumerate().map(|(lane, error)| {
        if let Some(error) = error {
            assert_eq!(batch.lane(lane).unwrap_err(), error);
            None
        } else {
            assert!(batch.lane(lane).is_ok());
            let values = &batch.values[lane * OUTPUTS * components..(lane + 1) * OUTPUTS * components];
            assert!(values.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
            Some(metadata(values))
        }
    }).collect()
}

#[test]
fn existing_defaults_and_explicit_symbolica_remain_distinct() {
    assert_eq!(MasterExecutionPolicy::default(), MasterExecutionPolicy::Default);
    MasterExecutionPolicy::Default.validate_available().unwrap();
    MasterExecutionPolicy::Symbolica.validate_available().unwrap();
    assert_eq!(MasterEvaluator::new(ScalarIntegral::A0, 0.).unwrap().backend(), default_master_backend());
    assert_eq!(MasterEvaluator::new_with_policy(ScalarIntegral::A0, 0., MasterExecutionPolicy::Default).unwrap().backend(), default_master_backend());
    let mut explicit = MasterEvaluator::new_with_policy(ScalarIntegral::A0, 0., MasterExecutionPolicy::Symbolica).unwrap();
    assert_eq!(explicit.backend(), MasterBackend::SymbolicaBatch);
    assert!(!explicit.set_native_batch_policy(BatchExecutionPolicy::Whole));
    let plan = ExactMasterPlan::new(ScalarIntegral::A0).unwrap();
    assert_eq!(plan.instantiate(0.).unwrap().backend(), MasterBackend::SymbolicaBatch);
    assert_eq!(plan.instantiate_with_policy(0., MasterExecutionPolicy::Default).unwrap().backend(), MasterBackend::SymbolicaBatch);
}

#[cfg(not(feature = "native-f64-abc"))]
#[test]
fn unavailable_selection_errors_before_fallback_or_preparation() {
    let policy = MasterExecutionPolicy::NativeF64Abc;
    assert!(matches!(policy.validate_available(), Err(MasterError::Preparation(_))));
    for family in [ScalarIntegral::A0, ScalarIntegral::D0] {
        assert!(matches!(MasterEvaluator::new_with_policy(family, 0., policy), Err(MasterError::Preparation(_))));
    }
    assert!(matches!(MasterEvaluator::new_dual_with_policy(ScalarIntegral::A0, 0., HyperDual::<f64>::new(vec![vec![0]]), policy), Err(MasterError::Preparation(_))));
    let plan = ExactMasterPlan::new(ScalarIntegral::A0).unwrap();
    assert!(matches!(plan.instantiate_with_policy(0., policy), Err(MasterError::Preparation(_))));
    assert!(matches!(Pentagon6Plan::new_with_policy(policy), Err(MasterError::Preparation(_))));
    assert!(matches!(Pentagon6Plan::new_dual_with_policy(HyperDual::<f64>::new(vec![vec![0]]), policy), Err(MasterError::Preparation(_))));
}

#[cfg(all(feature = "native-f64-abc", not(feature = "native-batch")))]
#[test]
fn selective_feature_does_not_enable_the_legacy_broad_constructor() {
    assert!(matches!(MasterEvaluator::new_with_backend(ScalarIntegral::A0, 0., MasterBackend::RustBatch), Err(MasterError::Preparation(_))));
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn selected_batches_keep_six_outputs_errors_and_own_repeats() {
    let fixtures = [
        (ScalarIntegral::A0, vec![2., 1.], vec![5., 4.], 0..1),
        (ScalarIntegral::B0, vec![-3., 2., 5., 1.], vec![30., 2., 5., 1.], 1..3),
        (ScalarIntegral::C0, vec![1., 2., 3., 5., 7., 11., 1.], vec![-1., -2., 30., 2., 3., 5., 1.], 3..6),
    ];
    for (family, first, physical, masses) in fixtures {
        let first: Vec<_> = first.into_iter().map(c).collect();
        let physical: Vec<_> = physical.into_iter().map(c).collect();
        let mut complex = physical.clone();
        for argument in masses { complex[argument].im = -0.1; }
        let mut invalid = first.clone();
        *invalid.last_mut().unwrap() = c(-1.);
        let mut scaleless = vec![c(0.); family.arity()];
        *scaleless.last_mut().unwrap() = c(1.);
        let points = [first, physical, complex, invalid, scaleless];
        let plan = ExactMasterPlan::new(family).unwrap();
        let mut baseline = plan.instantiate_with_policy(0., MasterExecutionPolicy::Symbolica).unwrap();
        let mut selected = plan.instantiate_with_policy(0., MasterExecutionPolicy::NativeF64Abc).unwrap();
        assert_eq!(baseline.backend(), MasterBackend::SymbolicaBatch);
        assert_eq!(selected.backend(), MasterBackend::RustBatch);
        assert!(selected.dual_shape().is_none());
        for storage in [BatchExecutionPolicy::Whole, BatchExecutionPolicy::FixedRows(std::num::NonZeroUsize::new(2).unwrap())] {
            assert!(selected.set_native_batch_policy(storage));
            for rows in [0, 1, 5, 3, 0, 1] {
                let input = points[..rows].concat();
                let original = metadata(&input);
                let errors: Vec<_> = (0..rows).map(|lane| (lane == 3).then_some(LaneError::InvalidScale)).collect();
                let a = baseline.evaluate_batch(&input, rows).unwrap();
                let b = selected.evaluate_batch(&input, rows).unwrap();
                let av = snapshot(&a, 1, &errors);
                let bv = snapshot(&b, 1, &errors);
                for lane in 0..rows {
                    if errors[lane].is_some() { continue; }
                    let scale = points[lane].iter().map(|z| z.re.hypot(z.im)).fold(1_f64, f64::max);
                    let unit = match family { ScalarIntegral::A0 => scale, ScalarIntegral::C0 => 1. / scale, _ => 1. };
                    for coefficient in 0..OUTPUTS {
                        let x = &a.values[lane * OUTPUTS + coefficient];
                        let y = &b.values[lane * OUTPUTS + coefficient];
                        let denominator = (if coefficient == 5 { 1. } else { unit }) + x.re.hypot(x.im);
                        assert!(denominator.is_finite() && denominator > 0.);
                        let error = (x.re - y.re).hypot(x.im - y.im) / denominator;
                        assert!(error.is_finite() && error <= 1e-10);
                    }
                }
                for _ in 0..2 {
                    assert_eq!(snapshot(&baseline.evaluate_batch(&input, rows).unwrap(), 1, &errors), av);
                    assert_eq!(snapshot(&selected.evaluate_batch(&input, rows).unwrap(), 1, &errors), bv);
                }
                assert_eq!(metadata(&input), original);
                assert!(matches!(selected.evaluate_batch(&[], 1), Err(MasterError::InputLength { .. })));
                assert!(matches!(selected.evaluate_batch(&[], usize::MAX), Err(MasterError::SizeOverflow)));
            }
        }
    }
}

#[cfg(feature = "native-f64-abc")]
fn unchanged_numeric<T: NativeFloat>(prototype: T, point: &[Complex<T>])
where Complex<T>: EvaluationDomain {
    let original = metadata(point);
    let plan = ExactMasterPlan::new(ScalarIntegral::A0).unwrap();
    let mut baseline = plan.instantiate_with_policy(prototype.clone(), MasterExecutionPolicy::Symbolica).unwrap();
    let mut selected = plan.instantiate_with_policy(prototype, MasterExecutionPolicy::NativeF64Abc).unwrap();
    assert_eq!(selected.backend(), MasterBackend::SymbolicaBatch);
    let expected = snapshot(&baseline.evaluate_batch(point, 1).unwrap(), 1, &[None]);
    for _ in 0..3 {
        assert_eq!(snapshot(&selected.evaluate_batch(point, 1).unwrap(), 1, &[None]), expected);
    }
    assert_eq!(metadata(point), original);
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn higher_precision_and_unselected_families_remain_exact_symbolica() {
    let one = DoubleFloat::from(1.);
    let zero = DoubleFloat::from(0.);
    let mass = one + one.from_i64(2).pow(80).inv();
    assert_ne!(mass, one);
    unchanged_numeric(zero, &[Complex::new(mass, zero), Complex::new(one, zero)]);
    let mass = Float::with_val(256, 1) + Float::with_val(256, 2).pow(100).inv();
    assert_ne!(mass, Float::with_val(256, 1));
    unchanged_numeric(Float::new(128), &[Complex::new(mass, Float::new(64)), Complex::new(Float::with_val(128, 1), Float::new(256))]);
    for (family, values) in [
        (ScalarIntegral::DB0, vec![-3., 2., 5., 1.]),
        (ScalarIntegral::D0, vec![-1., -2., -3., -4., -5., -6., 2., 3., 4., 5., 1.]),
    ] {
        let plan = ExactMasterPlan::new(family).unwrap();
        let point: Vec<_> = values.into_iter().map(c).collect();
        let mut a = plan.instantiate_with_policy(0., MasterExecutionPolicy::Symbolica).unwrap();
        let mut b = plan.instantiate_with_policy(0., MasterExecutionPolicy::NativeF64Abc).unwrap();
        assert_eq!(b.backend(), MasterBackend::SymbolicaBatch);
        let expected = snapshot(&a.evaluate_batch(&point, 1).unwrap(), 1, &[None]);
        for _ in 0..3 { assert_eq!(snapshot(&b.evaluate_batch(&point, 1).unwrap(), 1, &[None]), expected); }
    }
}

#[test]
fn explicit_dual_policy_keeps_lazy_scalar_children_symbolica() {
    let mut policies = vec![MasterExecutionPolicy::Symbolica];
    if cfg!(feature = "native-f64-abc") { policies.push(MasterExecutionPolicy::NativeF64Abc); }
    let plan = ExactMasterPlan::new_dual(ScalarIntegral::A0, HyperDual::<f64>::new(vec![vec![0], vec![1]])).unwrap();
    let point = [c(2.), c(0.), c(1.), c(0.)];
    let mut reference = plan.instantiate_with_policy(0., MasterExecutionPolicy::Symbolica).unwrap();
    let expected = snapshot(&reference.evaluate_batch(&point, 1).unwrap(), 2, &[None]);
    for policy in policies {
        let mut evaluator = plan.instantiate_with_policy(0., policy).unwrap();
        assert_eq!(evaluator.backend(), MasterBackend::SymbolicaBatch);
        for _ in 0..3 {
            assert_eq!(snapshot(&evaluator.evaluate_batch(&point, 1).unwrap(), 2, &[None]), expected);
            let child = evaluator.constant_evaluator.as_ref().expect("constant Dual scalar child");
            assert_eq!(child.backend(), MasterBackend::SymbolicaBatch);
            assert_eq!(child.execution_policy, MasterExecutionPolicy::Symbolica);
        }
    }
    let mut legacy = plan.instantiate(0.).unwrap();
    snapshot(&legacy.evaluate_batch(&point, 1).unwrap(), 2, &[None]);
    assert_eq!(legacy.constant_evaluator.as_ref().unwrap().backend(), default_master_backend());
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn one_component_and_seeded_dual_shapes_never_select_native() {
    for shape in [vec![vec![0]], vec![vec![0], vec![1]]] {
        for (family, values) in [
            (ScalarIntegral::A0, vec![2., 1.]),
            (ScalarIntegral::C0, vec![-1., -5., -4., 2., 3., 5., 1.]),
        ] {
            let components = shape.len();
            let plan = ExactMasterPlan::new_dual(family, HyperDual::<f64>::new(shape.clone())).unwrap();
            let mut input: Vec<_> = values.into_iter().flat_map(|value| std::iter::once(c(value)).chain(std::iter::repeat_n(c(0.), components - 1))).collect();
            if components > 1 { input[1] = c(1.); }
            let mut a = plan.instantiate_with_policy(0., MasterExecutionPolicy::Symbolica).unwrap();
            let mut b = plan.instantiate_with_policy(0., MasterExecutionPolicy::NativeF64Abc).unwrap();
            assert_eq!(b.backend(), MasterBackend::SymbolicaBatch);
            assert!(b.dual_shape().is_some());
            let expected = snapshot(&a.evaluate_batch(&input, 1).unwrap(), components, &[None]);
            for _ in 0..3 { assert_eq!(snapshot(&b.evaluate_batch(&input, 1).unwrap(), components, &[None]), expected); }
        }
    }
}

symbolica::create_hyperdual_from_components!(PolicyStaticDual, [[0], [1]]);

#[cfg(feature = "native-f64-abc")]
#[test]
fn static_and_dynamic_dual_adapters_keep_policy_and_shape() {
    let mass = PolicyStaticDual::new_variable(0, c(2.));
    let mut scale = PolicyStaticDual::new_variable(0, c(1.));
    scale.values[1] = c(0.);
    let mut stat = MasterEvaluator::new_dual_with_policy(ScalarIntegral::A0, 0., mass.clone(), MasterExecutionPolicy::NativeF64Abc).unwrap();
    assert_eq!(stat.backend(), MasterBackend::SymbolicaBatch);
    let expected = snapshot(&stat.evaluate_dual_batch(&[mass.clone(), scale.clone()], 1, |x| &x.values).unwrap(), 2, &[None]);
    let shape = vec![vec![0], vec![1]];
    let input = [HyperDual::from_values(shape.clone(), mass.values.to_vec()), HyperDual::from_values(shape.clone(), scale.values.to_vec())];
    let mut dynamic = MasterEvaluator::new_dual_with_policy(ScalarIntegral::A0, 0., HyperDual::<f64>::new(shape), MasterExecutionPolicy::NativeF64Abc).unwrap();
    assert_eq!(snapshot(&dynamic.evaluate_hyperdual_batch(&input, 1).unwrap(), 2, &[None]), expected);
    let wrong = [HyperDual::from_values(vec![vec![0]], vec![c(2.)]), HyperDual::from_values(vec![vec![0]], vec![c(1.)])];
    assert!(matches!(dynamic.evaluate_hyperdual_batch(&wrong, 1), Err(MasterError::InvalidShape(_))));
    assert_eq!(snapshot(&dynamic.evaluate_hyperdual_batch(&input, 1).unwrap(), 2, &[None]), expected);
}
