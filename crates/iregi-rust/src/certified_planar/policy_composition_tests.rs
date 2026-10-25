//! Lazy masked charts retain the independent backend and sidecar choices.
use super::*;

pub(crate) fn assert_chart_context(
    workspace: &CertifiedPlanarWorkspace<f64>,
    expected: MasterExecutionPolicy,
) {
    assert_eq!(workspace.policy, WorkspacePolicy::RETAIN_ENVELOPES);
    assert_eq!(workspace.master_policy, expected);
    assert!(workspace.charts.is_empty());
    let mut chart = workspace.prepare_chart(3, 8).unwrap();
    assert_eq!(chart.dual_shape(), workspace.shape.as_deref());
    assert!(crate::execution::policy_composition_tests::assert_context(
        &chart, WorkspacePolicy::RETAIN_ENVELOPES, expected, true,
    ) > 0);
    let invalid = chart.reduce_batch(&KinematicsBatch {
        rows: 1,
        components: workspace.components,
        values: vec![Complex::new(f64::NAN, 0.); workspace.plan.parameters.len() * workspace.components],
    }).unwrap();
    assert_ne!(invalid.statuses[0], LaneStatus::Success);
    assert_eq!(invalid.primal_envelopes.len(), 1);
    assert!(invalid.primal_envelopes[0].is_none());
    assert_eq!(invalid.primal_envelope_errors.len(), 1);
    assert_eq!(chart.last_execution().master_rows, 0);
    let empty = chart.reduce_batch(&KinematicsBatch {
        rows: 0, components: workspace.components, values: vec![],
    }).unwrap();
    assert!(empty.statuses.is_empty());
}

#[test]
fn composition_masked_planar_chart_preserves_explicit_symbolica() {
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    let plan = CertifiedPlanarReductionPlan::new_for_six_point_child(settings).unwrap();
    let workspace = CertifiedPlanarWorkspace::new_with_policies(
        &plan, 0., WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::Symbolica,
    ).unwrap();
    assert_chart_context(&workspace, MasterExecutionPolicy::Symbolica);
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn composition_masked_planar_chart_preserves_selected_policy_for_both_shapes() {
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    let plan = CertifiedPlanarReductionPlan::new_for_six_point_child(settings).unwrap();
    let ordinary = CertifiedPlanarWorkspace::new_with_policies(
        &plan, 0., WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc,
    ).unwrap();
    assert_chart_context(&ordinary, MasterExecutionPolicy::NativeF64Abc);
    let dual = CertifiedPlanarWorkspace::new_dual_with_policies(
        &plan, 0., HyperDual::<Complex<Rational>>::new(vec![vec![0]]),
        WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc,
    ).unwrap();
    assert_eq!(dual.shape, Some(vec![vec![0]]));
    assert_chart_context(&dual, MasterExecutionPolicy::NativeF64Abc);
}
