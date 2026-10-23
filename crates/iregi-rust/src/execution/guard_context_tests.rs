// Spliced into the isolated candidate's execution::tests module.
// Exercise the actual Workspace lifecycle without exposing GuardContext fields.

fn guard_context_lifecycle_number(bits: u32, value: i64) -> Complex<Float> {
    Complex::new(Float::with_val(bits, value), Float::new(bits))
}

/// A coefficient with 80 retained bits passes the 53-bit target and fails the
/// 128-bit target. Compare the cached guard to the uncached arithmetic as well
/// as asserting that the selected target makes a measurable difference.
fn guard_context_lifecycle_target_probe(workspace: &mut Workspace<Float>, bits: u32) {
    assert_eq!(workspace.accuracy_bits, bits);
    let mut values = vec![
        guard_context_lifecycle_number(256, 0);
        workspace.basis.len() * 4 * workspace.components
    ];
    values[0] = guard_context_lifecycle_number(80, 1);
    let mut input =
        vec![guard_context_lifecycle_number(256, 0); workspace.parameters * workspace.components];
    for mass in workspace.layout.masses.clone() {
        input[mass * workspace.components] = guard_context_lifecycle_number(256, 4);
    }
    input[workspace.layout.scale * workspace.components] = guard_context_lifecycle_number(256, 1);
    let expected = workspace.stability.inspect_output_precision_lane(
        &values,
        &input,
        workspace.components,
        &workspace.prototype,
        bits,
    );
    assert_eq!(expected.is_some(), bits >= 128, "{expected:?}");
    workspace.guard_context.begin_lane();
    let actual = workspace.stability.inspect_output_precision_lane_context(
        &values,
        &input,
        workspace.components,
        &workspace.prototype,
        bits,
        &mut workspace.guard_context,
    );
    assert_eq!(actual, expected);
}

fn guard_context_lifecycle_probe_tree(workspace: &mut Workspace<Float>, bits: u32) {
    guard_context_lifecycle_target_probe(workspace, bits);
    for child in [
        &mut workspace.fallback,
        &mut workspace.lightlike_equal,
        &mut workspace.lightlike_unequal,
    ]
    .into_iter()
    .flatten()
    {
        guard_context_lifecycle_probe_tree(child, bits);
    }
    for child in workspace
        .zero_profiles
        .values_mut()
        .filter_map(|child| child.as_mut().ok())
    {
        guard_context_lifecycle_probe_tree(child, bits);
    }
}

fn guard_context_lifecycle_same_result(actual: &BatchResult<Float>, expected: &BatchResult<Float>) {
    assert_eq!(actual.basis, expected.basis);
    assert_eq!(actual.components, expected.components);
    assert_eq!(actual.metadata, expected.metadata);
    assert_eq!(actual.statuses, expected.statuses);
    assert_eq!(actual.analytic_used, expected.analytic_used);
    assert_eq!(actual.fallback_used, expected.fallback_used);
    assert_eq!(actual.values.len(), expected.values.len());
    for (actual, expected) in actual.values.iter().zip(&expected.values) {
        for (actual, expected) in [(&actual.re, &expected.re), (&actual.im, &expected.im)] {
            assert_eq!(actual, expected);
            assert_eq!(actual.get_precision(), expected.get_precision());
            assert_eq!(actual.negative_sign(), expected.negative_sign());
        }
    }
}

#[test]
fn guard_context_lifecycle_accuracy_refresh_rejects_invalid_targets_and_clones_independently() {
    let plan = plan(1, 0);
    let mut original = Workspace::new(&plan, Float::new(256)).unwrap();
    guard_context_lifecycle_target_probe(&mut original, 256);
    original.set_accuracy_precision(53).unwrap();
    guard_context_lifecycle_target_probe(&mut original, 53);

    // Clone after the context has actually cached a scale and natural power.
    let mut cloned = original.clone();
    cloned.set_accuracy_precision(128).unwrap();
    guard_context_lifecycle_target_probe(&mut cloned, 128);
    guard_context_lifecycle_target_probe(&mut original, 53);
    for invalid in [0, 257] {
        assert!(cloned.set_accuracy_precision(invalid).is_err());
        guard_context_lifecycle_target_probe(&mut cloned, 128);
        guard_context_lifecycle_target_probe(&mut original, 53);
    }
    cloned.set_accuracy_precision(53).unwrap();
    original.set_accuracy_precision(128).unwrap();
    guard_context_lifecycle_target_probe(&mut cloned, 53);
    guard_context_lifecycle_target_probe(&mut original, 128);
}

#[test]
fn guard_context_lifecycle_accuracy_refresh_reaches_existing_hybrid_children() {
    let mut settings = ReductionSettings::new(2, 1);
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    assert!(workspace.fallback.is_some());
    assert!(workspace.lightlike_equal.is_some());
    assert!(workspace.lightlike_unequal.is_some());
    for bits in [53, 128, 53] {
        workspace.set_accuracy_precision(bits).unwrap();
        guard_context_lifecycle_probe_tree(&mut workspace, bits);
    }
    let mut cloned = workspace.clone();
    cloned.set_accuracy_precision(128).unwrap();
    guard_context_lifecycle_probe_tree(&mut cloned, 128);
    guard_context_lifecycle_probe_tree(&mut workspace, 53);
}

#[test]
fn guard_context_lifecycle_accuracy_refresh_reaches_existing_and_late_zero_profiles() {
    let mut settings = ReductionSettings::new(2, 0);
    settings.algorithm = Algorithm::Ibp;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    assert!(workspace.zero_profile_offset.is_some());
    assert!(workspace.zero_profiles.is_empty());
    let row = |masses: [i64; 2]| KinematicsBatch {
        rows: 1,
        components: 1,
        values: [0, 0, 0, 0, 0, 1, 0, 0, masses[0], masses[1], 1]
            .map(|value| guard_context_lifecycle_number(256, value))
            .to_vec(),
    };
    for (bits, masses, profiles) in [(53, [0, 4], 1), (128, [4, 0], 2), (53, [0, 4], 2)] {
        workspace.set_accuracy_precision(bits).unwrap();
        // This checks existing children before the next profile is created.
        guard_context_lifecycle_probe_tree(&mut workspace, bits);
        let input = row(masses);
        let actual = workspace.reduce_batch(&input).unwrap();
        let mut fresh = Workspace::new(&plan, Float::new(256)).unwrap();
        fresh.set_accuracy_precision(bits).unwrap();
        let expected = fresh.reduce_batch(&input).unwrap();
        guard_context_lifecycle_same_result(&actual, &expected);
        assert_eq!(workspace.last_execution(), fresh.last_execution());
        assert_eq!(workspace.last_diagnostics(), fresh.last_diagnostics());
        assert_eq!(actual.statuses, [LaneStatus::Success]);
        assert_eq!(actual.analytic_used, [true]);
        assert_eq!(workspace.zero_profiles.len(), profiles);
        assert!(workspace.zero_profiles.values().all(Result::is_ok));
        // Newly created children must inherit the target already selected.
        guard_context_lifecycle_probe_tree(&mut workspace, bits);
    }
}

#[test]
fn guard_context_lifecycle_reused_batches_match_fresh_workspaces_after_empty_and_invalid_calls() {
    let plan = plan(1, 0);
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    let row = |bits, mass| {
        [0, 0, 0, 0, mass, 1]
            .map(|value| guard_context_lifecycle_number(bits, value))
            .to_vec()
    };
    for order in [
        [(256, 2), (80, 2), (128, 3)],
        [(128, 3), (256, 2), (80, 2)],
        [(80, 2), (128, 3), (256, 2)],
    ] {
        let input = KinematicsBatch {
            rows: order.len(),
            components: 1,
            values: order
                .into_iter()
                .flat_map(|(bits, mass)| row(bits, mass))
                .collect(),
        };
        let actual = workspace.reduce_batch(&input).unwrap();
        let mut fresh = Workspace::new(&plan, Float::new(256)).unwrap();
        fresh.set_accuracy_precision(53).unwrap();
        let expected = fresh.reduce_batch(&input).unwrap();
        guard_context_lifecycle_same_result(&actual, &expected);
        assert_eq!(workspace.last_execution(), fresh.last_execution());
        assert_eq!(workspace.last_diagnostics(), fresh.last_diagnostics());
        assert!(
            actual
                .statuses
                .iter()
                .all(|status| *status == LaneStatus::Success)
        );

        let empty = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 1,
                values: vec![],
            })
            .unwrap();
        assert_eq!(empty.rows(), 0);
        let mut invalid = row(256, 2);
        invalid[5] = guard_context_lifecycle_number(256, -1);
        let rejected = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: invalid,
            })
            .unwrap();
        assert!(matches!(rejected.statuses[0], LaneStatus::InvalidInput(_)));
        assert!(
            workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: vec![],
                })
                .is_err()
        );
    }
}

#[test]
fn guard_context_lifecycle_rescue_retains_the_callers_target_at_each_precision() {
    let mut plan = plan(1, 0);
    let factor = Atom::num(2).pow(70);
    let ultraviolet = plan.leaves[0].slots[0].clone();
    // Symbolica constructs the expression. At mass squared 4 its exact value
    // is one, but the subtraction deliberately consumes about 70 bits.
    plan.moments[0] = [
        &factor * ultraviolet - (&factor * Atom::num(4) - Atom::num(1)),
        Atom::new(),
        Atom::new(),
        Atom::new(),
    ];
    plan.special_kinematics = true;
    let input = KinematicsBatch {
        rows: 1,
        components: 1,
        values: [0., 0., 0., 0., 4., 1.].map(c).to_vec(),
    };
    let precise_input = KinematicsBatch {
        rows: 1,
        components: 1,
        values: [0, 0, 0, 0, 4, 1]
            .map(|value| guard_context_lifecycle_number(128, value))
            .to_vec(),
    };
    let mut strict = Workspace::new(&plan, Float::new(128)).unwrap();
    let strict_result = strict.reduce_batch(&precise_input).unwrap();
    assert!(matches!(strict_result.statuses[0], LaneStatus::Unstable(_)));
    strict.set_accuracy_precision(53).unwrap();
    let relaxed = strict.reduce_batch(&precise_input).unwrap();
    assert_eq!(relaxed.statuses, [LaneStatus::Success]);
    assert_eq!(relaxed.lane(0).unwrap()[0].re, Float::with_val(128, 1));

    let mut rescue =
        crate::PrecisionRescueWorkspace::new(&plan, 0., crate::RescueSettings::default()).unwrap();
    let actual = rescue.reduce_batch(&input).unwrap();
    assert!(matches!(
        rescue.last_primary_statuses()[0],
        LaneStatus::Unstable(_)
    ));
    assert_eq!(actual.statuses, [LaneStatus::Success]);
    assert_eq!(actual.lane(0).unwrap(), &[c(1.), c(0.), c(0.), c(0.)]);
    assert_eq!(rescue.last_execution().requested_rows, 1);
    assert_eq!(rescue.last_execution().accepted_rows, 1);
    assert!(matches!(
        rescue.last_rescue_statuses()[0],
        crate::RescueStatus::Converged {
            working_bits: 128,
            verification_bits: 256,
            ..
        }
    ));
}

// Run rescue::tests::verified_batch_keeps_dual_axes_and_invalid_lanes_separate
// unchanged in BOTH frozen builds. It exercises retained caller accuracy in
// the primary/working/verification workspaces, second-order Dual coefficients,
// invalid-lane isolation and an empty follow-up batch without duplicating it.
