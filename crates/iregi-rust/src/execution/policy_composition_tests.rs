//! Composition checks for independent sidecar/retry and master policies.
use super::*;
use crate::{Algorithm, ReductionSettings};

// Shared only by sibling test modules: no production accessor or public API.
pub(crate) fn assert_context<T: NativeFloat>(
    workspace: &Workspace<T>,
    sidecars: WorkspacePolicy,
    masters: MasterExecutionPolicy,
    plain_f64: bool,
) -> usize
where
    Complex<T>: EvaluationDomain,
{
    assert_eq!(workspace.policy, sidecars);
    assert_eq!(workspace.master_execution_policy(), masters);
    let mut groups = 0;
    for group in &workspace.masters {
        if let ScalarMasterEvaluator::OneLoop(evaluator) = &group.evaluator {
            let expected = match masters {
                MasterExecutionPolicy::Default if workspace.shape.is_none() => iregi_masters::default_master_backend(),
                MasterExecutionPolicy::NativeF64Abc
                    if plain_f64 && workspace.shape.is_none()
                        && matches!(evaluator.family(), ScalarIntegral::A0 | ScalarIntegral::B0 | ScalarIntegral::C0) => MasterBackend::RustBatch,
                _ => MasterBackend::SymbolicaBatch,
            };
            assert_eq!(evaluator.backend(), expected);
            groups += 1;
        }
    }
    for child in [
        workspace.fallback.as_deref(),
        workspace.lightlike_equal.as_deref(),
        workspace.lightlike_unequal.as_deref(),
    ].into_iter().flatten() {
        groups += assert_context(child, sidecars, masters, plain_f64);
    }
    for child in workspace.zero_profiles.values() {
        groups += assert_context(child.as_ref().expect("lazy policy preparation failed"), sidecars, masters, plain_f64);
    }
    groups
}

fn tadpole() -> ReductionPlan {
    ReductionPlan::new(ReductionSettings::new(1, 0)).unwrap()
}

fn massless_tadpole() -> KinematicsBatch<f64> {
    KinematicsBatch {
        rows: 1,
        components: 1,
        values: [0., 0., 0., 0., 0., 1.].map(|x| Complex::new(x, 0.)).to_vec(),
    }
}

#[test]
fn composition_sidecars_share_exact_cache_but_master_policies_do_not() {
    let plan = tadpole();
    let mut ordinary = Workspace::new_with_policies(
        &plan, 0., WorkspacePolicy::ORDINARY, MasterExecutionPolicy::Symbolica,
    ).unwrap();
    let first = plan.execution_cache.lock().unwrap()[&(None, MasterExecutionPolicy::Symbolica)].clone();
    let mut retained = Workspace::new_with_policies(
        &plan, 0., WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica,
    ).unwrap();
    {
        let cache = plan.execution_cache.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(Arc::ptr_eq(&first, &cache[&(None, MasterExecutionPolicy::Symbolica)]));
    }
    assert!(ordinary.planar_fallback.is_none() && retained.planar_fallback.is_none());
    assert!(assert_context(&ordinary, WorkspacePolicy::ORDINARY, MasterExecutionPolicy::Symbolica, true) > 0);
    assert!(assert_context(&retained, WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica, true) > 0);
    let defaults = Workspace::new(&plan, 0.).unwrap();
    assert_eq!(defaults.policy, WorkspacePolicy::ORDINARY);
    {
        let cache = plan.execution_cache.lock().unwrap();
        assert_eq!(cache.len(), 2);
        assert!(!Arc::ptr_eq(&first, &cache[&(None, MasterExecutionPolicy::Default)]));
    }
    let batch = KinematicsBatch {
        rows: 1, components: 1,
        values: [0., 0., 0., 0., 2., 1.].map(|x| Complex::new(x, 0.)).to_vec(),
    };
    let baseline = ordinary.reduce_batch(&batch).unwrap();
    let actual = retained.reduce_batch(&batch).unwrap();
    assert_eq!(baseline.statuses, [LaneStatus::Success]);
    assert_eq!(actual.statuses, baseline.statuses);
    assert_eq!(actual.values, baseline.values);
    for (a, b) in actual.values.iter().zip(&baseline.values) {
        assert_eq!(a.re.to_bits(), b.re.to_bits());
        assert_eq!(a.im.to_bits(), b.im.to_bits());
    }
    assert_eq!(actual.metadata, baseline.metadata);
    assert_eq!(actual.fallback_used, baseline.fallback_used);
    assert_eq!(actual.analytic_used, baseline.analytic_used);
    assert_eq!(ordinary.last_execution(), retained.last_execution());
    assert_eq!(ordinary.last_diagnostics(), retained.last_diagnostics());
    assert_eq!(baseline.primal_envelopes.capacity(), 0);
    assert_eq!(baseline.primal_envelope_errors.capacity(), 0);
    assert!(actual.validated_primal_envelope(0, 53, 53).is_ok());
}

#[test]
fn composition_ordered_dual_shapes_and_master_policy_separate_exact_entries() {
    use symbolica::domains::float::Float;
    let plan = tadpole().with_analytic_jet_order(1).unwrap();
    let shapes = [vec![vec![0, 0], vec![1, 0], vec![0, 1]], vec![vec![0, 0], vec![0, 1], vec![1, 0]]];
    for shape in &shapes {
        for policy in [MasterExecutionPolicy::Default, MasterExecutionPolicy::Symbolica] {
            let workspace = Workspace::new_dual_with_policies(
                &plan, 0., HyperDual::<Complex<Rational>>::new(shape.clone()),
                WorkspacePolicy::RETAIN_ENVELOPES, policy,
            ).unwrap();
            assert_eq!(workspace.dual_shape(), Some(shape.as_slice()));
            assert_eq!(workspace.policy, WorkspacePolicy::RETAIN_ENVELOPES);
            assert_eq!(workspace.master_execution_policy(), policy);
        }
    }
    let keys: Vec<_> = shapes.iter().flat_map(|shape| [MasterExecutionPolicy::Default, MasterExecutionPolicy::Symbolica]
        .map(|policy| (Some(shape.clone()), policy))).collect();
    let first = {
        let cache = plan.execution_cache.lock().unwrap();
        assert_eq!(cache.len(), 4);
        for (i, key) in keys.iter().enumerate() {
            for other in &keys[i + 1..] {
                assert!(!Arc::ptr_eq(&cache[key], &cache[other]));
            }
        }
        cache[&keys[1]].clone()
    };
    let precise = Workspace::new_dual_with_policies(
        &plan, Float::new(128), HyperDual::<Complex<Rational>>::new(shapes[0].clone()),
        WorkspacePolicy::ORDINARY, MasterExecutionPolicy::Symbolica,
    ).unwrap();
    assert!(assert_context(&precise, WorkspacePolicy::ORDINARY, MasterExecutionPolicy::Symbolica, false) > 0);
    let cache = plan.execution_cache.lock().unwrap();
    assert_eq!(cache.len(), 4);
    assert!(Arc::ptr_eq(&first, &cache[&keys[1]]));
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn composition_selected_policy_survives_eager_dual_and_lazy_children() {
    let mut settings = ReductionSettings::new(2, 1);
    settings.algorithm = Algorithm::Hybrid;
    let bubble = ReductionPlan::new(settings).unwrap();
    for shape in [None, Some(vec![vec![0], vec![1], vec![2]])] {
        let workspace = match &shape {
            None => Workspace::new_with_policies(&bubble, 0., WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc),
            Some(shape) => Workspace::new_dual_with_policies(&bubble, 0., HyperDual::<Complex<Rational>>::new(shape.clone()), WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc),
        }.unwrap();
        assert_eq!(workspace.dual_shape(), shape.as_deref());
        assert!(workspace.fallback.is_some());
        assert!(workspace.lightlike_equal.is_some() && workspace.lightlike_unequal.is_some());
        assert!(assert_context(&workspace, WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc, true) > 0);
    }
    let plan = tadpole();
    let mut lazy = Workspace::new_with_policies(&plan, 0., WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc).unwrap();
    for _ in 0..2 {
        let result = lazy.reduce_batch(&massless_tadpole()).unwrap();
        assert_eq!(result.statuses, [LaneStatus::Success]);
        assert_eq!(result.analytic_used, [true]);
        assert!(result.validated_primal_envelope(0, 53, 53).is_ok());
        assert!(!lazy.zero_profiles.is_empty());
        assert!(assert_context(&lazy, WorkspacePolicy::SIX_POINT_CHILD, MasterExecutionPolicy::NativeF64Abc, true) > 0);
    }
}
