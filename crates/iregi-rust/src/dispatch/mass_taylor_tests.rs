use super::*;
use crate::{Algorithm, Convention, InputPolicy, InvariantInput, TensorWeights};

#[test]
fn mass_taylor_selection_preserves_algorithms_and_supplied_product_semantics() {
    for algorithm in [
        Algorithm::Ibp,
        Algorithm::PassarinoVeltman,
        Algorithm::Hybrid,
    ] {
        for convention in [
            Convention::IregiNative,
            Convention::OneLoop,
            Convention::MadLoop,
        ] {
            for invariants in [
                InvariantInput::FromShifts,
                InvariantInput::SuppliedScalarProducts,
            ] {
                for power in [2, 3] {
                    let mut settings = ReductionSettings::new(5, 6);
                    settings.algorithm = algorithm;
                    settings.convention = convention;
                    settings.invariants = invariants;
                    settings.tensor_weights = TensorWeights::Multinomial;
                    settings.powers[0] = power;
                    let before = serde_json::to_value(&settings).unwrap();
                    assert!(crate::mass_taylor::settings_supported(&settings));
                    // Settings alone cannot establish the auxiliary analytic
                    // mass domain. Default dispatch keeps zero-mass dot support.
                    assert_eq!(ordinary_kind(&settings), TensorEngineKind::Generic);
                    assert_eq!(serde_json::to_value(&settings).unwrap(), before);
                }
            }
        }
    }
    let mut explicit = ReductionSettings::new(5, 0);
    explicit.algorithm = Algorithm::PassarinoVeltman;
    explicit.powers[0] = 2;
    let before = serde_json::to_value(&explicit).unwrap();
    let plan = TensorReductionPlan::new_mass_taylor(explicit).unwrap();
    assert_eq!(plan.engine_kind(), TensorEngineKind::MassTaylor);
    assert_eq!(serde_json::to_value(plan.settings()).unwrap(), before);
}

#[test]
fn mass_taylor_does_not_change_unit_low_point_or_high_point_descent_domains() {
    let mut dot = ReductionSettings::new(5, 6);
    dot.powers[0] = 2;
    for h in [-1, 1] {
        let mut shifted = dot.clone();
        shifted.dimension_shift = h;
        assert_eq!(ordinary_kind(&shifted), TensorEngineKind::Generic);
    }
    let mut legacy = dot.clone();
    legacy.input_policy = InputPolicy::legacy_onshell();
    // The separate no-op proof stage preserves requested Legacy semantics.
    assert_eq!(ordinary_kind(&legacy), TensorEngineKind::LegacyNoOp);
    let mut too_high = dot.clone();
    too_high.rank = 7;
    assert_eq!(ordinary_kind(&too_high), TensorEngineKind::Generic);
    for n in [1, 2, 3, 4] {
        let mut low = ReductionSettings::new(n, 6);
        low.powers[0] = 2;
        assert_eq!(ordinary_kind(&low), TensorEngineKind::Generic);
    }
    for n in [6, 8, 17] {
        let mut high = ReductionSettings::new(n, 6);
        high.powers[0] = 2;
        assert_eq!(ordinary_kind(&high), TensorEngineKind::AffinePower);
    }
    assert_eq!(
        ordinary_kind(&ReductionSettings::new(5, 6)),
        TensorEngineKind::Generic
    );
    for malformed in [vec![0, 1, 1, 1, 1], vec![2, 1]] {
        let mut bad = dot.clone();
        bad.powers = malformed;
        assert_eq!(ordinary_kind(&bad), TensorEngineKind::Generic);
    }
}

#[test]
fn mass_taylor_summary_counts_physical_rows_once_and_retains_both_stages() {
    let unit = ExecutionStats {
        input_rows: 3,
        argument_rows: 3,
        master_family_batches: 4,
        master_rows: 21,
        reduction_rows: 3,
        ..Default::default()
    };
    let projection = ExecutionStats {
        input_rows: 3,
        argument_rows: 3,
        reduction_rows: 3,
        ..Default::default()
    };
    let stats = MassTaylorExecutionStats {
        input_rows: 3,
        user_components: 6,
        internal_components: 18,
        unit_execution: unit,
        projection_execution: projection,
        unit_algorithm_used: Algorithm::PassarinoVeltman,
        requested_algorithm: Algorithm::Ibp,
        ..Default::default()
    };
    let summary = TensorEngineExecution::MassTaylor(stats).summary().unwrap();
    assert_eq!(summary.input_rows, 3);
    assert_eq!(summary.argument_rows, 6);
    assert_eq!(summary.reduction_rows, 6);
    assert_eq!(summary.master_rows, 21);
    assert_eq!(summary.master_family_batches, 4);
    let mut overflow = projection;
    overflow.master_rows = usize::MAX;
    let bad = MassTaylorExecutionStats {
        input_rows: 3,
        user_components: 6,
        internal_components: 18,
        unit_execution: unit,
        projection_execution: overflow,
        unit_algorithm_used: Algorithm::PassarinoVeltman,
        requested_algorithm: Algorithm::Hybrid,
        ..Default::default()
    };
    assert!(TensorEngineExecution::MassTaylor(bad).summary().is_err());
}
