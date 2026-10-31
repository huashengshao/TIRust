//! Full branch/jet proof tests with no integral evaluation. The ignored real
//! array gate is supplied by the retained CLI fixture after API qualification.
use super::*;
use crate::InputPolicy;

fn settings() -> ReductionSettings {
    let mut s = ReductionSettings::new(5, 6);
    s.algorithm = crate::Algorithm::PassarinoVeltman;
    s.input_policy = InputPolicy::LegacyOnShell {
        threshold: "1/1000000".into(),
    };
    s
}
fn row() -> Vec<Complex<f64>> {
    let raw = [
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.5, 0., 0., 0., 1., 2.,
        3., 4., 2., 3.,
    ];
    raw.into_iter().map(|x| Complex::new(x, 0.)).collect()
}
fn policy(s: &ReductionSettings) -> queries::LegacyNoOpQueries {
    queries::LegacyNoOpQueries::prepare(s).unwrap()
}
fn guard(s: &ReductionSettings) -> NoOpGuard<f64> {
    NoOpGuard::prepare::<HyperDual<u8>>(&policy(s), 0., None).unwrap()
}
fn lift(row: &[Complex<f64>], components: usize) -> Vec<Complex<f64>> {
    row.iter()
        .flat_map(|&value| {
            std::iter::once(value).chain(std::iter::repeat_n(Complex::new(0., 0.), components - 1))
        })
        .collect()
}

#[test]
fn original_noop_queries_isolate_changed_neighbor_and_preserve_input() {
    let mut g = guard(&settings());
    let good = row();
    let mut changed = good.clone();
    changed[16].re = 1. + 2_f64.powi(-24);
    let input = KinematicsBatch {
        rows: 3,
        components: 1,
        values: [good.clone(), changed, good.clone()].concat(),
    };
    let saved = input.values.clone();
    let selected = g.select(&input).unwrap();
    assert_eq!(input.values, saved);
    assert_eq!(selected.accepted, [0, 2]);
    assert!(
        matches!(&selected.statuses[1],LaneStatus::InvalidInput(reason)if reason.contains("Taylor component 0"))
    );
    assert!(
        selected.diagnostics[1]
            .iter()
            .any(|s| s.contains("not an exact demanded-query"))
    );
    assert!(selected.certificate.evaluated_rows > 0);
    assert_eq!(selected.certificate.input_rows, 3);
    assert_eq!(
        g.select(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: good
        })
        .unwrap()
        .accepted,
        [0]
    );
}

#[test]
fn selected_real_mass_and_zero_queries_compare_complete_reordered_mixed_jets() {
    let shape = HyperDual::<u8>::new(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    let mut g = NoOpGuard::prepare(&policy(&settings()), 0., Some(&shape)).unwrap();
    let mut scalar = row();
    scalar[16].re = 2.; // exact norm4 and original real reference mass4
    let mut good = lift(&scalar, 4);
    for line in 0..5 {
        good[(4 * line) * 4 + 1].re = 1.;
    } // common routing jet
    let mut transverse = good.clone();
    transverse[16 * 4 + 3].re = 1.;
    let input = KinematicsBatch {
        rows: 3,
        components: 4,
        values: [good.clone(), transverse, good].concat(),
    };
    let saved = input.values.clone();
    let selected = g.select(&input).unwrap();
    assert_eq!(input.values, saved);
    assert_eq!(selected.accepted, [0, 2]);
    assert!(
        matches!(&selected.statuses[1],LaneStatus::InvalidInput(reason)if reason.contains("Taylor component 3"))
    );
    assert_eq!(selected.certificate.reduction_batches, 0);
}

#[test]
fn complex_width_retains_original_mass_and_reference_order() {
    let mut g = guard(&settings());
    let mut good = row();
    good[16].re = 1.;
    good[20].im = -0.125;
    let saved = good.clone();
    let input = KinematicsBatch {
        rows: 1,
        components: 1,
        values: good,
    };
    assert_eq!(g.select(&input).unwrap().accepted, [0]);
    assert_eq!(input.values, saved);
}

#[test]
fn strict_boundary_is_diagnostic_for_ordinary_and_rejects_dual_before_child() {
    let mut s = settings();
    s.input_policy = InputPolicy::LegacyOnShell {
        threshold: "0".into(),
    };
    let mut massless = row();
    for m in &mut massless[20..25] {
        m.re = 0.;
    }
    let mut ordinary = guard(&s);
    let selected = ordinary
        .select(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: massless.clone(),
        })
        .unwrap();
    assert_eq!(selected.accepted, [0]);
    assert!(
        selected.diagnostics[0]
            .iter()
            .any(|s| s.contains("strict-comparison boundary"))
    );
    let shape = HyperDual::<u8>::new(vec![vec![0], vec![1]]);
    let mut dual = NoOpGuard::prepare(&policy(&s), 0., Some(&shape)).unwrap();
    let selected = dual
        .select(&KinematicsBatch {
            rows: 1,
            components: 2,
            values: lift(&massless, 2),
        })
        .unwrap();
    assert!(selected.accepted.is_empty());
    assert!(
        matches!(&selected.statuses[0],LaneStatus::Unstable(reason)if reason.contains("nondifferentiable"))
    );
    assert_eq!(selected.certificate.evaluated_rows, 0);
}

#[test]
fn signed_reference_zero_image_and_unrepresentable_threshold_cannot_claim_noop() {
    let mut s = settings();
    s.input_policy = InputPolicy::LegacyOnShell {
        threshold: "0".into(),
    };
    let mut negative = row();
    for (i, m) in negative[20..25].iter_mut().enumerate() {
        m.re = -(i as f64 + 1.);
    }
    let selected = guard(&s)
        .select(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: negative,
        })
        .unwrap();
    assert!(selected.accepted.is_empty());
    assert!(matches!(selected.statuses[0], LaneStatus::InvalidInput(_)));
    s.input_policy = InputPolicy::LegacyOnShell {
        threshold: "1e-1000".into(),
    };
    let selected = guard(&s)
        .select(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: row(),
        })
        .unwrap();
    assert!(
        matches!(&selected.statuses[0],LaneStatus::Unstable(reason)if reason.contains("threshold"))
    );
}

#[test]
fn empty_invalid_and_malformed_calls_do_not_prepare_any_reduction_graph() {
    let plan = Arc::new(LegacyNoOpPlan::new(settings()).unwrap());
    assert!(plan.preserve.get().is_none());
    let mut workspace = LegacyNoOpWorkspace::new(
        plan.clone(),
        0.,
        TensorWorkspaceOptions {
            master_policy: MasterExecutionPolicy::Symbolica,
            profiling: true,
            ..Default::default()
        },
    )
    .unwrap();
    let empty = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 0,
            components: 1,
            values: vec![],
        })
        .unwrap();
    assert_eq!(empty.rows(), 0);
    assert!(workspace.child.is_none());
    assert!(plan.preserve.get().is_none());
    let mut bad = row();
    bad[0].re = f64::NAN;
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: bad,
        })
        .unwrap();
    assert!(matches!(result.statuses[0], LaneStatus::InvalidInput(_)));
    assert!(workspace.child.is_none());
    assert!(plan.preserve.get().is_none());
    assert!(matches!(
        workspace.engine_execution(),
        TensorEngineExecution::LegacyNoOp {
            child: None,
            child_preparation_attempts: 0,
            rejected_rows: 1,
            ..
        }
    ));
    assert!(
        workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: vec![]
            })
            .is_err()
    );
    assert!(matches!(
        workspace.engine_execution(),
        TensorEngineExecution::LegacyNoOp {
            child: None,
            child_preparation_attempts: 0,
            certificate_execution: CertificateExecutionStats {
                evaluated_rows: 0,
                ..
            },
            ..
        }
    ));
    assert!(workspace.last_profile().is_some());
}
