//! Public-constructor regression controls for explicit workspace master policy.
use super::*;
use crate::{Convention, ReductionSettings};
use symbolica::domains::float::Float;

fn bubble(rank: u16) -> ReductionPlan {
    let mut settings = ReductionSettings::new(2, rank);
    settings.convention = Convention::OneLoop;
    ReductionPlan::new(settings).unwrap()
}

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn assert_interpreter<T: NativeFloat>(workspace: &Workspace<T>)
where
    Complex<T>: EvaluationDomain,
{
    for group in &workspace.masters {
        if let ScalarMasterEvaluator::OneLoop(evaluator) = &group.evaluator {
            assert_eq!(evaluator.backend(), MasterBackend::SymbolicaBatch);
        }
    }
    for child in [
        workspace.fallback.as_deref(),
        workspace.lightlike_equal.as_deref(),
        workspace.lightlike_unequal.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        assert_interpreter(child);
    }
}

#[test]
fn explicit_symbolica_overrides_the_legacy_feature_and_preserves_default_cache() {
    let plan = bubble(1);
    let explicit = Workspace::new_with_master_policy(&plan, 0., MasterExecutionPolicy::Symbolica)
        .unwrap();
    assert_eq!(explicit.master_execution_policy(), MasterExecutionPolicy::Symbolica);
    assert_interpreter(&explicit);
    let ordinary = Workspace::new(&plan, 0.).unwrap();
    assert_eq!(ordinary.master_execution_policy(), MasterExecutionPolicy::Default);
    assert!(ordinary.masters.iter().all(|group| match &group.evaluator {
        ScalarMasterEvaluator::OneLoop(evaluator) => {
            evaluator.backend() == iregi_masters::default_master_backend()
        }
        _ => true,
    }));
    assert_eq!(plan.execution_cache.lock().unwrap().len(), 2);
    let _second = Workspace::new_with_master_policy(
        &plan, Float::new(128), MasterExecutionPolicy::Symbolica,
    ).unwrap();
    assert_eq!(plan.execution_cache.lock().unwrap().len(), 2);
}

#[cfg(not(feature = "native-f64-abc"))]
#[test]
fn unavailable_selective_policy_fails_before_exact_preparation() {
    // A downstream crate can enable the dependency feature directly through
    // Cargo feature unification, without enabling this crate's forwarding flag.
    if MasterExecutionPolicy::NativeF64Abc.validate_available().is_ok() {
        return;
    }
    let plan = bubble(0);
    let result = Workspace::new_with_master_policy(
        &plan, 0., MasterExecutionPolicy::NativeF64Abc,
    );
    assert!(matches!(result, Err(ReductionError::Preparation(reason))
        if reason.contains("native-f64-abc")));
    assert!(plan.execution_cache.lock().unwrap().is_empty());
    let dual = HyperDual::<Complex<Rational>>::new(vec![vec![0]]);
    let result = Workspace::new_dual_with_master_policy(
        &plan, 0., dual, MasterExecutionPolicy::NativeF64Abc,
    );
    assert!(matches!(result, Err(ReductionError::Preparation(reason))
        if reason.contains("native-f64-abc")));
    assert!(plan.execution_cache.lock().unwrap().is_empty());
}

#[cfg(feature = "native-f64-abc")]
fn assert_selected_tree(workspace: &Workspace<f64>) {
    assert_eq!(workspace.master_execution_policy(), MasterExecutionPolicy::NativeF64Abc);
    for group in &workspace.masters {
        if let ScalarMasterEvaluator::OneLoop(evaluator) = &group.evaluator {
            let expected = if workspace.shape.is_none()
                && matches!(evaluator.family(), ScalarIntegral::A0 | ScalarIntegral::B0 | ScalarIntegral::C0)
            {
                MasterBackend::RustBatch
            } else {
                MasterBackend::SymbolicaBatch
            };
            assert_eq!(evaluator.backend(), expected);
        }
    }
    for child in [
        workspace.fallback.as_deref(),
        workspace.lightlike_equal.as_deref(),
        workspace.lightlike_unequal.as_deref(),
    ].into_iter().flatten() {
        assert_selected_tree(child);
    }
    for child in workspace.zero_profiles.values() {
        assert_selected_tree(child.as_ref().expect("selected zero-profile preparation failed"));
    }
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn selective_workspaces_keep_cache_policy_and_recursive_lane_semantics() {
    let plan = bubble(1);
    let mut native = Workspace::new_with_master_policy(
        &plan, 0., MasterExecutionPolicy::NativeF64Abc,
    ).unwrap();
    let mut reference = Workspace::new_with_master_policy(
        &plan, 0., MasterExecutionPolicy::Symbolica,
    ).unwrap();
    assert_eq!(plan.execution_cache.lock().unwrap().len(), 2);
    assert_selected_tree(&native);
    let batch = KinematicsBatch {
        rows: 3,
        components: 1,
        values: [
            [0., 0., 0., 0., 0., 1., 0., 0., 4., 9., 1.],
            [2., 1., 0., 0., 2., 1., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
        ].into_iter().flatten().map(c).collect(),
    };
    let actual = native.reduce_batch(&batch).unwrap();
    let expected = reference.reduce_batch(&batch).unwrap();
    assert_eq!(actual.basis, expected.basis);
    assert_eq!(actual.components, expected.components);
    assert_eq!(actual.metadata, expected.metadata);
    assert_eq!(actual.statuses, expected.statuses);
    assert_eq!(actual.fallback_used, expected.fallback_used);
    assert_eq!(actual.analytic_used, expected.analytic_used);
    for lane in 0..3 {
        assert_eq!(actual.lane(lane).unwrap().len(), expected.lane(lane).unwrap().len());
        for (left, right) in actual.lane(lane).unwrap().iter().zip(expected.lane(lane).unwrap()) {
            assert!((left.re - right.re).abs() <= 1e-10 * (1. + right.re.abs()));
            assert!((left.im - right.im).abs() <= 1e-10 * (1. + right.im.abs()));
        }
    }
    assert!(!native.zero_profiles.is_empty());
    assert_selected_tree(&native);
    let reordered = KinematicsBatch {
        rows: 3,
        components: 1,
        values: [2, 0, 1].into_iter()
            .flat_map(|lane| batch.values[lane * 11..(lane + 1) * 11].iter().cloned())
            .collect(),
    };
    let again = native.reduce_batch(&reordered).unwrap();
    for (new, old) in [2, 0, 1].into_iter().enumerate() {
        assert_eq!(again.lane(new).unwrap(), actual.lane(old).unwrap());
        for (left, right) in again.lane(new).unwrap().iter().zip(actual.lane(old).unwrap()) {
            assert_eq!(left.re.to_bits(), right.re.to_bits());
            assert_eq!(left.im.to_bits(), right.im.to_bits());
        }
    }
    let empty = native.reduce_batch(&KinematicsBatch {
        rows: 0, components: 1, values: Vec::new(),
    }).unwrap();
    assert!(empty.statuses.is_empty());
}

#[cfg(feature = "native-f64-abc")]
#[test]
fn selective_policy_keeps_high_precision_and_component_one_duals_on_symbolica() {
    let plan = bubble(0);
    let precise = Workspace::new_with_master_policy(
        &plan, Float::new(128), MasterExecutionPolicy::NativeF64Abc,
    ).unwrap();
    assert_interpreter(&precise);
    let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0]]);
    let mut dual = Workspace::new_dual_with_master_policy(
        &plan, 0., shape, MasterExecutionPolicy::NativeF64Abc,
    ).unwrap();
    assert_interpreter(&dual);
    let mut reference = Workspace::new_dual_with_master_policy(
        &plan, 0., HyperDual::<Complex<Rational>>::new(vec![vec![0]]), MasterExecutionPolicy::Symbolica,
    ).unwrap();
    let batch = KinematicsBatch {
        rows: 1, components: 1,
        values: [0., 0., 0., 0., 0., 1., 0., 0., 4., 9., 1.].map(c).to_vec(),
    };
    let actual = dual.reduce_batch(&batch).unwrap();
    let expected = reference.reduce_batch(&batch).unwrap();
    assert_eq!(actual.basis, expected.basis);
    assert_eq!(actual.components, expected.components);
    assert_eq!(actual.metadata, expected.metadata);
    assert_eq!(actual.statuses, expected.statuses);
    assert_eq!(actual.lane(0).unwrap(), expected.lane(0).unwrap());
    for (left, right) in actual.lane(0).unwrap().iter().zip(expected.lane(0).unwrap()) {
        assert_eq!(left.re.to_bits(), right.re.to_bits());
        assert_eq!(left.im.to_bits(), right.im.to_bits());
    }
    assert_interpreter(&dual);
}
