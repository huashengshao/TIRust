// Included inside execution::tests. These tests exercise only the new policy
// boundary; the existing ordinary and certified tests retain their own scope.

fn assert_policy_children(workspace: &Workspace<f64>, expected: WorkspacePolicy) {
    assert_eq!(workspace.policy, expected);
    for child in [&workspace.fallback, &workspace.lightlike_equal, &workspace.lightlike_unequal]
        .into_iter().flatten()
    {
        assert_policy_children(child, expected);
    }
    for child in workspace.zero_profiles.values().filter_map(|child| child.as_ref().ok()) {
        assert_policy_children(child, expected);
    }
}

#[test]
fn workspace_policy_default_preserves_values_without_sidecar_allocation() {
    let plan = plan(1, 0);
    let mut ordinary = Workspace::new(&plan, 0.).unwrap();
    let mut retained = Workspace::new_with_policy(&plan, 0., WorkspacePolicy::RETAIN_ENVELOPES).unwrap();
    let input = KinematicsBatch {
        rows: 2,
        components: 1,
        values: [0., 0., 0., 0., 2., 1., f64::NAN, 0., 0., 0., 2., 1.].map(c).to_vec(),
    };
    let baseline = ordinary.reduce_batch(&input).unwrap();
    let actual = retained.reduce_batch(&input).unwrap();
    assert_eq!(baseline.statuses, actual.statuses);
    assert_eq!(baseline.values, actual.values);
    assert_eq!(baseline.metadata, actual.metadata);
    assert_eq!(baseline.fallback_used, actual.fallback_used);
    assert_eq!(baseline.analytic_used, actual.analytic_used);
    assert_eq!(ordinary.last_execution(), retained.last_execution());
    assert_eq!(ordinary.last_diagnostics(), retained.last_diagnostics());
    assert_eq!(baseline.primal_envelopes.capacity(), 0);
    assert_eq!(baseline.primal_envelope_errors.capacity(), 0);
    assert_eq!(actual.statuses[0], LaneStatus::Success);
    assert!(actual.validated_primal_envelope(0, 53, 53).is_ok());
    assert!(actual.primal_envelopes[1].is_none());
    assert!(ordinary.planar_fallback.is_none() && retained.planar_fallback.is_none());
}

#[test]
fn workspace_policy_isolates_retry_while_sharing_exact_arithmetic_cache() {
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    let plan = ReductionPlan::new(settings).unwrap();
    assert!(plan.automatic_planar_settings().is_some());
    let ordinary = Workspace::new(&plan, 0.).unwrap();
    let opted = Workspace::new_with_policy(&plan, 0., WorkspacePolicy::SIX_POINT_CHILD).unwrap();
    assert_eq!(ordinary.policy, WorkspacePolicy::ORDINARY);
    assert_eq!(opted.policy, WorkspacePolicy::SIX_POINT_CHILD);
    assert!(ordinary.planar_fallback.is_none());
    assert!(opted.planar_fallback.is_some());
    // The policy does not change a single prepared arithmetic expression.
    assert_eq!(plan.execution_cache.lock().unwrap().len(), 1);
    assert!(ordinary.zero_profiles.is_empty() && opted.zero_profiles.is_empty());
}

#[test]
fn workspace_policy_survives_dual_specialization_eager_and_lazy_children() {
    let mut settings = ReductionSettings::new(2, 1);
    settings.algorithm = Algorithm::Hybrid;
    let bubble = ReductionPlan::new(settings).unwrap();
    let shape = vec![vec![0], vec![1], vec![2]];
    let dual = HyperDual::<Complex<Rational>>::new(shape.clone());
    let ordinary = Workspace::new_dual(&bubble, 0., dual.clone()).unwrap();
    let opted = Workspace::new_dual_with_policy(&bubble, 0., dual, WorkspacePolicy::SIX_POINT_CHILD).unwrap();
    assert_eq!(opted.dual_shape(), Some(shape.as_slice()));
    assert!(opted.fallback.is_some());
    assert!(opted.lightlike_equal.is_some() && opted.lightlike_unequal.is_some());
    assert_policy_children(&ordinary, WorkspacePolicy::ORDINARY);
    assert_policy_children(&opted, WorkspacePolicy::SIX_POINT_CHILD);

    let tadpole = plan(1, 0);
    let mut lazy = Workspace::new_with_policy(&tadpole, 0., WorkspacePolicy::SIX_POINT_CHILD).unwrap();
    let result = lazy.reduce_batch(&KinematicsBatch {
        rows: 1, components: 1, values: [0., 0., 0., 0., 0., 1.].map(c).to_vec(),
    }).unwrap();
    assert_eq!(result.statuses, [LaneStatus::Success]);
    assert_eq!(result.analytic_used, [true]);
    assert!(!lazy.zero_profiles.is_empty());
    assert_policy_children(&lazy, WorkspacePolicy::SIX_POINT_CHILD);
    assert!(result.validated_primal_envelope(0, 53, 53).is_ok());
}

#[test]
fn workspace_policy_certificate_totals_reset_before_empty_and_invalid_calls() {
    let plan = plan(1, 0);
    let mut workspace = Workspace::new_with_policy(&plan, 0., WorkspacePolicy::SIX_POINT_CHILD).unwrap();
    let previous = CertificateExecutionStats { input_rows: 6, evaluated_rows: 30, passes: 5, reduction_batches: 1 };
    workspace.planar_certificate_stats = previous;
    workspace.reduce_batch(&KinematicsBatch { rows: 0, components: 1, values: Vec::new() }).unwrap();
    assert_eq!(workspace.last_planar_certificate_execution(), CertificateExecutionStats::default());
    workspace.planar_certificate_stats = previous;
    assert!(workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: Vec::new() }).is_err());
    assert_eq!(workspace.last_planar_certificate_execution(), CertificateExecutionStats::default());
    workspace.planar_certificate_stats = previous;
    let input: Vec<HyperDual<Complex<f64>>> = Vec::new();
    assert!(workspace.reduce_hyperdual_batch(&input, 1).is_err());
    assert_eq!(workspace.last_planar_certificate_execution(), CertificateExecutionStats::default());
}

#[test]
fn workspace_policy_counts_nested_analytic_work_without_changing_ordinary_counts() {
    let plan = plan(1, 0);
    let batch = KinematicsBatch {
        rows: 1, components: 1, values: [0., 0., 0., 0., 0., 1.].map(c).to_vec(),
    };
    for policy in [WorkspacePolicy::ORDINARY, WorkspacePolicy::RETAIN_ENVELOPES] {
        let mut child = Workspace::new_with_policy(&plan, 0., policy).unwrap();
        let mut result = child.reduce_batch(&batch).unwrap();
        assert_eq!(child.last_execution().analytic_rows, 1);
        let mut stats = ExecutionStats::default();
        let mut profile = ExecutionProfile::default();
        let mut diagnostics = vec![Vec::new()];
        let mut certificates = CertificateExecutionStats::default();
        // One outer dispatch plus the child's actual zero-mass dispatch.
        // The child's cached workspace must perform the second call afresh.
        dispatch_profile(&mut child, &[0], &batch, &mut result, &mut stats,
            &mut profile, &mut diagnostics, &mut certificates).unwrap();
        assert_eq!(child.last_execution().analytic_rows, 1);
        assert_eq!(stats.analytic_rows, if policy == WorkspacePolicy::ORDINARY { 1 } else { 2 });
        assert_eq!(stats.argument_rows, child.last_execution().argument_rows);
        assert_eq!(stats.reduction_rows, child.last_execution().reduction_rows);
        assert_eq!(stats.fallback_rows, 0);
        assert_eq!(result.statuses, [LaneStatus::Success]);
        assert_eq!(certificates, CertificateExecutionStats::default());
    }
}
