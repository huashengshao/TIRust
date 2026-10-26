//! Facade orchestration and pre-master proof tests; no new reduction algebra.
use super::*;
use crate::{
    Algorithm, Convention, ExactInvariantScalar, InvariantInput, LaneStatus, TensorWeights,
};
use symbolica::domains::{
    dual::HyperDual,
    float::{DoubleFloat, Float, FloatLike, SingleFloat},
    rational::Rational,
};

fn settings(rank: u16) -> ReductionSettings {
    let mut s = ReductionSettings::new(6, rank);
    s.algorithm = Algorithm::PassarinoVeltman;
    s.invariants = InvariantInput::SuppliedScalarProducts;
    s
}
fn symbolic(profiling: bool) -> TensorWorkspaceOptions {
    TensorWorkspaceOptions {
        master_policy: MasterExecutionPolicy::Symbolica,
        profiling,
    }
}
/// Literal quarter-unit fixture. The 21 packed products are independently
/// specified input data, not reconstructed by the facade or these tests.
fn row<T: NativeFloat>(zero: &T) -> Vec<Complex<T>> {
    let shifts = [
        0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 4, 4, 0,
    ];
    let products = [
        0, 0, 0, 0, 0, 0, -4, 0, 0, 0, -4, -4, 0, 0, -4, -4, 0, 0, 1, 0, -8,
    ];
    shifts
        .into_iter()
        .chain(products)
        .map(|n| Complex::new(zero.from_rational(&Rational::from((n, 4))), zero.zero()))
        .chain(
            [1, 2, 3, 4, 2, 3, 3]
                .map(|n| Complex::new(zero.from_rational(&Rational::from(n)), zero.zero())),
        )
        .collect()
}

#[test]
fn native_exact_bridge_keeps_beyond_binary64_and_non_native_implementors() {
    fn native<T: NativeFloat>(value: &T) -> Option<Rational> {
        value.exact_invariant_value()
    }
    let expected = Rational::from(((1u64 << 60) + 1, 1u64 << 60));
    let zero = Float::new(192);
    let arbitrary = zero.from_rational(&expected);
    assert_eq!(native(&arbitrary), Some(expected));
    let sparse = DoubleFloat::from_compensated_sum(1., 2_f64.powi(-300));
    assert_eq!(
        native(&sparse),
        iregi_masters::NativeFloat::exact_rational(&sparse)
    );
    assert_ne!(native(&sparse), Some(Rational::from(1)));
    assert!(native(&f64::INFINITY).is_none());
    struct ExactOnly(Rational);
    impl ExactInvariantScalar for ExactOnly {
        fn exact_invariant_value(&self) -> Option<Rational> {
            Some(self.0.clone())
        }
    }
    let standalone = ExactOnly(Rational::from((7, 11)));
    assert_eq!(
        standalone.exact_invariant_value(),
        Some(Rational::from((7, 11)))
    );
}

#[test]
fn checked_master_policy_never_downgrades_to_unchecked_generic() {
    for profiling in [false, true] {
        assert!(symbolic(profiling).validate_checked().is_ok());
        let native = TensorWorkspaceOptions {
            master_policy: MasterExecutionPolicy::NativeF64Abc,
            profiling,
        };
        assert!(matches!(
            native.validate_checked(),
            Err(ReductionError::Unsupported(_))
        ));
        let default = TensorWorkspaceOptions {
            profiling,
            ..Default::default()
        }
        .validate_checked();
        assert_eq!(
            default.is_ok(),
            iregi_masters::default_master_backend() == MasterBackend::SymbolicaBatch
        );
    }
}

#[test]
fn checked_facade_preserves_layout_and_profiling_and_rejects_before_masters() {
    let mut requested = settings(0);
    requested.convention = Convention::OneLoop;
    requested.tensor_weights = TensorWeights::Multinomial;
    assert!(matches!(
        TensorReductionPlan::new(requested.clone()),
        Err(ReductionError::Unsupported(_))
    ));
    let plan = TensorReductionPlan::new_checked_cartesian(requested.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(plan.settings()).unwrap(),
        serde_json::to_value(&requested).unwrap()
    );
    assert_eq!(plan.engine_kind(), TensorEngineKind::CertifiedTensor);
    assert_eq!(plan.parameter_count(), 52);
    assert_eq!(plan.input_layout().scalar_products, Some(24..45));
    assert_eq!(plan.basis(), [Monomial([0; 4])]);
    assert!(matches!(
        plan.generic_plan(),
        Err(ReductionError::Unsupported(_))
    ));
    assert!(plan.generic.get().is_none());
    assert!(matches!(
        plan.rescue_workspace(0., RescueSettings::default()),
        Err(ReductionError::Unsupported(_))
    ));
    assert!(matches!(
        plan.rescue_dual_workspace(
            0.,
            RescueSettings::default(),
            HyperDual::<u8>::new(vec![vec![0]])
        ),
        Err(ReductionError::Unsupported(_))
    ));
    let native = TensorWorkspaceOptions {
        master_policy: MasterExecutionPolicy::NativeF64Abc,
        profiling: true,
    };
    assert!(matches!(
        TensorWorkspace::new_with_options(&plan, 0., native),
        Err(ReductionError::Unsupported(_))
    ));
    assert!(plan.generic.get().is_none());
    let mut workspace = TensorWorkspace::new_with_options(&plan, 0., symbolic(true)).unwrap();
    assert_eq!(workspace.engine_kind(), TensorEngineKind::CertifiedTensor);
    assert_eq!(workspace.binary_precision(), 53);
    assert_eq!(workspace.component_count(), 1);
    assert!(workspace.last_profile().is_some());
    workspace
        .set_certificate_precision_budget(128, 256)
        .unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    let mut wrong = row(&0.);
    wrong[30] = Complex::new(0., 0.);
    let mut nonfinite = row(&0.);
    nonfinite[0] = Complex::new(f64::INFINITY, 0.);
    let input = KinematicsBatch {
        rows: 2,
        components: 1,
        values: wrong.into_iter().chain(nonfinite).collect(),
    };
    let before = input.values.clone();
    let output = workspace.reduce_batch(&input).unwrap();
    assert_eq!(input.values, before);
    assert!(
        output
            .statuses
            .iter()
            .all(|s| matches!(s, LaneStatus::InvalidInput(_)))
    );
    assert_eq!(workspace.execution_summary().unwrap().master_rows, 0);
    assert_eq!(workspace.execution_summary().unwrap().reduction_rows, 0);
    match workspace.engine_execution() {
        TensorEngineExecution::CertifiedTensor {
            execution,
            certificate_execution,
        } => {
            assert_eq!(execution.input_rows, 2);
            assert_eq!(execution.argument_rows, 0);
            assert_eq!(certificate_execution.cartesian.input_rows, 2);
            assert_eq!(certificate_execution.cartesian.evaluated_rows, 1);
            assert_eq!(certificate_execution.gram.input_rows, 2);
            assert_eq!(certificate_execution.gram.evaluated_rows, 0);
        }
        _ => panic!("checked execution lost both proof counters"),
    }
    let profile = workspace.last_profile().unwrap();
    assert!(profile.total >= profile.arguments);
    workspace.set_profiling(false).unwrap();
    assert!(workspace.last_profile().is_none());
    workspace
        .reduce_batch(&KinematicsBatch {
            rows: 0,
            components: 1,
            values: vec![],
        })
        .unwrap();
    assert_eq!(
        workspace.execution_summary().unwrap(),
        ExecutionStats::default()
    );
    assert!(workspace.last_diagnostics().is_empty());
    assert!(
        workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 2,
                values: vec![]
            })
            .is_err()
    );
    assert_eq!(workspace.execution_summary().unwrap().master_rows, 0);
    assert!(plan.generic.get().is_none());
}

#[test]
fn checked_ordered_dual_transverse_component_cannot_reach_gram_or_masters() {
    let plan = TensorReductionPlan::new_checked_cartesian(settings(0)).unwrap();
    let shape = HyperDual::<u8>::new(vec![vec![0], vec![1]]);
    let mut workspace =
        TensorWorkspace::new_dual_with_options(&plan, 0., shape, symbolic(true)).unwrap();
    assert_eq!(workspace.engine_kind(), TensorEngineKind::CertifiedTensor);
    assert_eq!(workspace.component_count(), 2);
    let mut values = row(&0.)
        .into_iter()
        .flat_map(|v| [v, Complex::new(0., 0.)])
        .collect::<Vec<_>>();
    values[30 * 2 + 1] = Complex::new(1., 0.); // independent G11 first component, fixed Cartesian vectors
    let before = values.clone();
    let input = KinematicsBatch {
        rows: 1,
        components: 2,
        values,
    };
    let output = workspace.reduce_batch(&input).unwrap();
    assert_eq!(input.values, before);
    assert_eq!(
        output.metadata().dual_shape(),
        Some([vec![0], vec![1]].as_slice())
    );
    assert!(
        matches!(&output.statuses[0],LaneStatus::InvalidInput(reason)if reason.contains("component 1"))
    );
    match workspace.engine_execution() {
        TensorEngineExecution::CertifiedTensor {
            execution,
            certificate_execution,
        } => {
            assert_eq!(execution.master_rows, 0);
            assert_eq!(execution.reduction_rows, 0);
            assert_eq!(certificate_execution.cartesian.evaluated_rows, 1);
            assert_eq!(certificate_execution.gram.evaluated_rows, 0);
        }
        _ => panic!("Dual certificate was silently bypassed"),
    }
    assert!(plan.generic.get().is_none());
}

#[test]
#[ignore = "separately bounded actual Float128 checked-facade/direct-wrapper full-array delegation"]
fn checked_facade_matches_direct_wrapper_with_valid_and_invalid_lanes() {
    let zero = Float::new(128);
    let plan = TensorReductionPlan::new_checked_cartesian(settings(1)).unwrap();
    let direct = match &plan.ordinary {
        OrdinaryPlan::CertifiedTensor(checked) => checked.clone(),
        _ => unreachable!(),
    };
    let good = row(&zero);
    let mut wrong = good.clone();
    wrong[30] = Complex::new(zero.zero(), zero.zero());
    let input = KinematicsBatch {
        rows: 3,
        components: 1,
        values: good.iter().cloned().chain(wrong).chain(good.clone()).collect(),
    };
    let before = input.values.clone();
    let mut facade =
        TensorWorkspace::new_with_options(&plan, zero.clone(), symbolic(true)).unwrap();
    facade.set_accuracy_precision(53).unwrap();
    let actual = facade.reduce_batch(&input).unwrap();
    let mut reference = CertifiedTensorWorkspace::new(&direct, zero).unwrap();
    reference.set_accuracy_precision(53).unwrap();
    reference.set_profiling(true);
    let expected = reference.reduce_batch(&input).unwrap();
    assert_eq!(actual.statuses, expected.statuses);
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(actual.basis, expected.basis);
    assert_eq!(actual.basis.len(), 5);
    assert_eq!(actual.lane(0).unwrap().len(), 20);
    for lane in [0, 2] {
        assert_eq!(actual.lane(lane).unwrap(), expected.lane(lane).unwrap());
    }
    assert!(matches!(actual.statuses[1], LaneStatus::InvalidInput(_)));
    assert_eq!(input.values, before);
    assert_eq!(
        facade.execution_summary().unwrap(),
        reference.last_execution()
    );
    match facade.engine_execution() {
        TensorEngineExecution::CertifiedTensor {
            certificate_execution,
            ..
        } => assert_eq!(
            certificate_execution,
            reference.last_certificate_execution()
        ),
        _ => panic!("checked engine detail was erased"),
    }
    assert!(facade.last_profile().is_some());
    assert!(plan.generic.get().is_none());
    let contraction = plan.contraction_plan(&[Monomial([0; 4])]).unwrap();
    let mut contraction = crate::ContractionWorkspace::new(&contraction, Float::new(128)).unwrap();
    let numerator = crate::NumeratorBatch {
        rows: 3,
        components: 1,
        values: vec![Complex::new(Float::with_val(128, 1), Float::new(128)); 3],
    };
    let contracted = contraction.contract_batch(&actual, &numerator).unwrap();
    for lane in [0, 2] {
        assert_eq!(
            contracted.lane(lane).unwrap(),
            &actual.lane(lane).unwrap()[..4]
        );
    }
    assert!(contracted.lane(1).is_err());
}
