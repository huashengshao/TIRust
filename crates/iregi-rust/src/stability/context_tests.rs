// Injected into candidate stability::tests, with uncached guards as the oracle.
fn context_same<T: NativeFloat>(a: &T, b: &T) {
    assert!(a == b, "represented values differ");
    assert_eq!(a.get_precision(), b.get_precision());
    assert_eq!(a.negative_sign(), b.negative_sign());
}

fn context_scale_plan() -> StabilityPlan {
    let mut plan = synthetic(2, false);
    plan.layout = InputLayout {
        shifts: 0..2,
        masses: 2..2,
        scalar_products: None,
        scale: 2,
    };
    plan.output_dimensions = vec![-2, 0, 1, 2];
    plan
}

#[test]
fn guard_context_preserves_tied_scale_precision_and_ignores_higher_jets() {
    let plan = context_scale_plan();
    let prototype = Float::new(256);
    let mut context = GuardContext::new(&plan, &prototype, 128);
    for bits in [[80, 256], [256, 80]] {
        let primal = [
            complex(Float::with_val(bits[0], 2)),
            complex(Float::with_val(bits[1], 2)),
            complex(Float::with_val(128, 1)),
        ];
        let jets: Vec<_> = primal
            .iter()
            .flat_map(|p| {
                [
                    p.clone(),
                    complex(Float::with_val(16, 1000000)),
                    complex(Float::with_val(1, -1000000)),
                ]
            })
            .collect();
        let expected = plan.kinematic_scale(&primal, 1, &prototype);
        assert_eq!(expected.get_precision(), bits[0]);
        context.begin_lane();
        let actual = context.natural(context.output_slots[2], &plan, &jets, 3, &prototype);
        context_same(&actual, &expected);
        let same_primal = context.natural(context.output_slots[2], &plan, &primal, 1, &prototype);
        context_same(&same_primal, &expected);
    }
}

#[test]
fn guard_context_resets_equal_values_with_different_precision_and_clones_hits() {
    let plan = context_scale_plan();
    let prototype = Float::new(256);
    let mut context = GuardContext::new(&plan, &prototype, 128);
    for bits in [256, 80, 128, 256] {
        let input = [
            complex(Float::with_val(bits, 2)),
            complex(Float::new(1)),
            complex(Float::with_val(256, 1)),
        ];
        let scale = plan.kinematic_scale(&input, 1, &prototype);
        context.begin_lane();
        for (i, dimension) in plan.output_dimensions.iter().enumerate() {
            let expected = signed_power(&scale, *dimension);
            let slot = context.output_slots[i];
            let mut value = context.natural(slot, &plan, &input, 1, &prototype);
            context_same(&value, &expected);
            value.set_prec(16);
            value *= Float::with_val(16, 3);
            context_same(
                &context.natural(slot, &plan, &input, 1, &prototype),
                &expected,
            );
        }
    }
}

#[test]
fn guard_context_scaleless_scale_uses_primal_mu_without_promoting_it() {
    let plan = context_scale_plan();
    let prototype = Float::new(256);
    let primal = [
        complex(Float::new(256)),
        complex(Float::new(128)),
        complex(Float::with_val(80, 3)),
    ];
    let jets: Vec<_> = primal
        .iter()
        .flat_map(|p| [p.clone(), complex(Float::with_val(16, 1000000))])
        .collect();
    let mut context = GuardContext::new(&plan, &prototype, 53);
    for (i, dimension) in plan.output_dimensions.iter().enumerate() {
        let expected = signed_power(&primal[2].re, *dimension);
        let actual = context.natural(context.output_slots[i], &plan, &jets, 2, &prototype);
        context_same(&actual, &expected);
    }
}

#[test]
fn guard_context_local_errors_keep_original_precedence_and_denominators_skip_powers() {
    for denominator in [false, true] {
        let plan = synthetic(4, denominator);
        let input = [complex(1e100)];
        let mut context = GuardContext::new(&plan, &0.0, 53);
        for values in [
            [complex(f64::NAN), complex(0.0), complex(0.0)],
            [complex(0.0), complex(f64::NAN), complex(0.0)],
            [complex(0.0), complex(0.0), complex(0.0)],
            [complex(1.0), complex(1.0), complex(0.0)],
        ] {
            context.begin_lane();
            let expected = plan.inspect_lane_with_accuracy(&values, &input, 1, &0.0, 53);
            let actual =
                plan.inspect_lane_with_accuracy_context(&values, &input, 1, &0.0, 53, &mut context);
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn guard_context_output_skip_paths_leave_unused_invalid_powers_unobserved() {
    let mut plan = synthetic(4, false);
    plan.output_dimensions = vec![4];
    let mut context = GuardContext::new(&plan, &0.0, 53);
    assert_eq!(
        plan.inspect_output_precision_lane_context(&[complex(1.0)], &[], 1, &0.0, 53, &mut context),
        None
    );
    let prototype = iregi_masters::DoubleFloat::from(0.0);
    let input = [complex(iregi_masters::DoubleFloat::from(1e100))];
    let mut context = GuardContext::new(&plan, &prototype, 106);
    for value in [0.0, 1.0] {
        context.begin_lane();
        let values = [complex(iregi_masters::DoubleFloat::from(value))];
        assert_eq!(
            plan.inspect_output_precision_lane_context(
                &values,
                &input,
                1,
                &prototype,
                106,
                &mut context
            ),
            plan.inspect_output_precision_lane(&values, &input, 1, &prototype, 106)
        );
    }
}

#[test]
fn guard_context_does_not_reject_infinite_scale_before_valid_zero_power() {
    let mut plan = synthetic(0, false);
    plan.layout = InputLayout {
        shifts: 0..0,
        masses: 0..1,
        scalar_products: None,
        scale: 1,
    };
    let input = [Complex::new(f64::MAX, -f64::MAX), complex(1.0)];
    assert!(plan.kinematic_scale(&input, 1, &0.0).is_infinite());
    let values = [complex(0.0), complex(0.0), complex(0.0)];
    let mut context = GuardContext::new(&plan, &0.0, 53);
    let expected = plan.inspect_lane_with_accuracy(&values, &input, 1, &0.0, 53);
    assert_eq!(expected, None);
    assert_eq!(
        plan.inspect_lane_with_accuracy_context(&values, &input, 1, &0.0, 53, &mut context),
        expected
    );
}

#[test]
fn guard_context_primal_error_precedence_and_invalid_buffers_match() {
    let mut plan = synthetic(4, false);
    plan.primal_expressions = vec![Atom::num(0), Atom::num(0)];
    plan.primal_checks = vec![PrimalCheck {
        value: 0,
        amplitude: 1,
        dimension: 4,
        label: "primal fixture".into(),
    }];
    let input = [complex(1e100)];
    let mut context = GuardContext::new(&plan, &0.0, 53);
    for values in [
        [complex(f64::NAN), complex(0.0)],
        [complex(0.0), complex(0.0)],
    ] {
        context.begin_lane();
        assert_eq!(
            plan.inspect_primal_lane_context(&values, &input, &0.0, 53, &mut context),
            plan.inspect_primal_lane(&values, &input, &0.0, 53)
        );
    }
    assert_eq!(
        plan.inspect_lane_with_accuracy_context(&[], &[], 1, &0.0, 53, &mut context),
        plan.inspect_lane_with_accuracy(&[], &[], 1, &0.0, 53)
    );
    assert_eq!(
        plan.inspect_primal_lane_context(&[], &[], &0.0, 53, &mut context),
        plan.inspect_primal_lane(&[], &[], &0.0, 53)
    );
    plan.output_dimensions = vec![4];
    assert_eq!(
        plan.inspect_output_precision_lane_context(&[], &[], 1, &0.0, 53, &mut context),
        plan.inspect_output_precision_lane(&[], &[], 1, &0.0, 53)
    );
}

#[test]
fn guard_context_typed_constants_refresh_accuracy_without_changing_roundoff() {
    let plan = synthetic(0, false);
    let prototype = Float::new(256);
    let mut context = GuardContext::new(&plan, &prototype, 256);
    let unit = context.prototype_unit.clone();
    for accuracy in [256, 53, 128, 53] {
        context.set_accuracy(&prototype, accuracy);
        context_same(
            &context.tolerance,
            &dyadic(&prototype, 19 - i64::from(accuracy)),
        );
        context_same(&context.prototype_unit, &unit);
        let values = [
            complex(Float::with_val(256, 1)),
            complex(Float::with_val(80, 1)),
            complex(Float::new(1)),
        ];
        let input = [complex(Float::with_val(256, 1))];
        context.begin_lane();
        assert_eq!(
            plan.inspect_lane_with_accuracy_context(
                &values,
                &input,
                1,
                &prototype,
                accuracy,
                &mut context
            ),
            plan.inspect_lane_with_accuracy(&values, &input, 1, &prototype, accuracy)
        );
    }
    let mut cloned = context.clone();
    cloned.set_accuracy(&prototype, 128);
    context_same(&context.tolerance, &dyadic(&prototype, 19 - 53));
    context_same(&cloned.tolerance, &dyadic(&prototype, 19 - 128));
}

#[test]
fn guard_context_prototype_unit_preserves_low_precision_and_range_estimates() {
    let plan = synthetic(0, false);
    let prototype = Float::new(256);
    let context = GuardContext::new(&plan, &prototype, 128);
    let mut baseline = OperandRoundoff::new(&prototype);
    let mut candidate = OperandRoundoff::with_unit(&prototype, context.prototype_unit.clone());
    for value in [
        Complex::new(Float::with_val(256, 1), Float::new(1)),
        Complex::new(Float::with_val(80, 3), Float::with_val(53, 4)),
        complex(Float::with_val(1, 2).pow(100).inv()),
        complex(Float::with_val(1, 1)),
        complex(Float::with_val(80, f64::INFINITY)),
    ] {
        match (baseline.complex(&value), candidate.complex(&value)) {
            (Ok(a), Ok(b)) => context_same(&a, &b),
            (Err(a), Err(b)) => assert_eq!(a, b),
            _ => panic!("roundoff status differs"),
        }
    }
}
