//! Public regression for a high-precision workspace receiving lower-precision
//! Float values. A requested accuracy must not be inferred from the prototype
//! when the actual scalar-master arithmetic retained the input precision.

use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, Convention, InvariantInput, KinematicsBatch, LaneStatus, PrecisionRescueWorkspace,
    ReductionPlan, ReductionSettings, RescueSettings, RescueStatus, Workspace,
};
use symbolica::domains::{
    dual::HyperDual,
    float::{Complex, Float, FloatLike, Real},
};

fn tadpole_row(bits: u32, mass: i64) -> Vec<Complex<Float>> {
    [0, 0, 0, 0, mass, 1]
        .into_iter()
        .map(|value| Complex::new(Float::with_val(bits, value), Float::new(bits)))
        .collect()
}

#[test]
fn requested_accuracy_respects_actual_float_arithmetic_precision() {
    let mut settings = ReductionSettings::new(1, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    // The mathematical inputs are exactly representable at every precision.
    // There is no uncertainty about the supplied mass or renormalization scale.
    let mut irrelevant_low_precision = tadpole_row(256, 2);
    irrelevant_low_precision[0] = Complex::new(Float::with_val(16, 1), Float::new(16));
    let batch = KinematicsBatch {
        rows: 4,
        components: 1,
        values: [
            tadpole_row(80, 2),
            tadpole_row(256, 2),
            tadpole_row(80, 0),
            irrelevant_low_precision,
        ]
        .concat(),
    };
    let two = Float::with_val(512, 2);
    let expected = two.clone() * (Float::with_val(512, 1) - two.log());
    let mut failed_targets = Vec::new();
    let mut unexpected_statuses = Vec::new();
    for accuracy_bits in [53, 128, 256] {
        workspace.set_accuracy_precision(accuracy_bits).unwrap();
        let result = workspace.reduce_batch(&batch).unwrap();
        let tolerance = Float::with_val(512, 2)
            .pow(u64::from(accuracy_bits - 19))
            .inv();
        println!(
            "accuracy_bits={accuracy_bits} statuses={:?}",
            result.statuses
        );
        for lane in 0..2 {
            if let Ok(values) = result.lane(lane) {
                let error = (values[3].re.at_precision(512) - &expected).norm();
                let scaled = error / Float::with_val(512, 2);
                println!(
                    "lane={lane} finite={} precision={} scaled_error={scaled:e} target={tolerance:e}",
                    values[3].re,
                    values[3].re.get_precision(),
                );
                if scaled > tolerance {
                    failed_targets.push((accuracy_bits, lane));
                }
            }
        }
        assert_eq!(result.statuses[1], LaneStatus::Success);
        assert_eq!(result.statuses[2], LaneStatus::Success);
        assert_eq!(result.statuses[3], LaneStatus::Success);
        if accuracy_bits == 53 {
            assert_eq!(result.statuses[0], LaneStatus::Success);
        } else if !matches!(result.statuses[0], LaneStatus::Unstable(_)) {
            unexpected_statuses.push((accuracy_bits, result.statuses[0].clone()));
        }
    }
    assert!(
        failed_targets.is_empty(),
        "successful lanes missed their requested accuracy: {failed_targets:?}"
    );
    assert!(unexpected_statuses.is_empty(), "{unexpected_statuses:?}");
}

#[test]
fn low_precision_taylor_seed_is_checked_without_a_local_sum() {
    let mut settings = ReductionSettings::new(1, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new_dual(
        &plan,
        Float::new(256),
        HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]),
    )
    .unwrap();
    workspace.set_accuracy_precision(128).unwrap();
    let mut values = Vec::new();
    for seed_bits in [80, 256] {
        for (argument, primal) in tadpole_row(256, 2).into_iter().enumerate() {
            values.push(primal);
            values.push(Complex::new(
                Float::with_val(seed_bits, if argument == 4 { 3 } else { 0 }),
                Float::new(seed_bits),
            ));
            values.push(Complex::new(Float::new(256), Float::new(256)));
        }
    }
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 2,
            components: 3,
            values,
        })
        .unwrap();
    assert!(
        matches!(result.statuses[0], LaneStatus::Unstable(_)),
        "{:?}",
        result.statuses
    );
    assert_eq!(result.statuses[1], LaneStatus::Success);
}

#[test]
fn insufficient_input_arithmetic_can_be_rescued_by_exact_promotion() {
    let mut settings = ReductionSettings::new(1, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = PrecisionRescueWorkspace::new(
        &plan,
        Float::new(256),
        RescueSettings {
            working_bits: 384,
            verification_bits: 512,
            accuracy_bits: Some(128),
            ..Default::default()
        },
    )
    .unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: tadpole_row(80, 2),
        })
        .unwrap();
    assert!(matches!(
        workspace.last_primary_statuses()[0],
        LaneStatus::Unstable(_)
    ));
    assert_eq!(result.statuses, [LaneStatus::Success]);
    assert!(matches!(
        workspace.last_rescue_statuses()[0],
        RescueStatus::Converged { .. }
    ));
    assert_eq!(workspace.last_execution().requested_rows, 1);
    let two = Float::with_val(512, 2);
    let expected = two.clone() * (Float::with_val(512, 1) - two.log());
    let error = (result.lane(0).unwrap()[3].re.at_precision(512) - expected).norm();
    assert!(error < Float::with_val(512, 2).pow(200).inv());
}

#[test]
fn low_precision_scale_cannot_hide_a_nonzero_bubble_finite_part() {
    let mut settings = ReductionSettings::new(2, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    let mass = Float::with_val(256, 4) + Float::with_val(256, 2).pow(100).inv();
    let mut values = Vec::new();
    for (scale_bits, scaleless) in [(80, false), (256, false), (80, true)] {
        values.extend((0..8).map(|_| Complex::new(Float::new(256), Float::new(256))));
        values.extend((0..2).map(|_| {
            Complex::new(
                if scaleless {
                    Float::new(256)
                } else {
                    mass.clone()
                },
                Float::new(256),
            )
        }));
        values.push(Complex::new(
            Float::with_val(scale_bits, 2),
            Float::new(scale_bits),
        ));
    }
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 3,
            components: 1,
            values,
        })
        .unwrap();
    println!("rounded-zero bubble statuses={:?}", result.statuses);
    let expected = -(mass.at_precision(512) / Float::with_val(512, 4)).log();
    if let Ok(value) = result.lane(0) {
        println!(
            "rounded-zero bubble finite={} bits={} absolute_error={:e}",
            value[3].re,
            value[3].re.get_precision(),
            (value[3].re.at_precision(512) - &expected).norm(),
        );
    }
    assert!(
        matches!(result.statuses[0], LaneStatus::Unstable(_)),
        "{:?}",
        result.statuses
    );
    assert_eq!(result.statuses[1], LaneStatus::Success);
    assert_eq!(result.statuses[2], LaneStatus::Success);
    let error = (result.lane(1).unwrap()[3].re.at_precision(512) - expected).norm();
    assert!(error < Float::with_val(512, 2).pow(235).inv());
}

#[test]
fn distance_precision_uses_only_its_declared_kinematic_inputs() {
    let mut settings = ReductionSettings::new(2, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings.clone()).unwrap();
    let mut values: Vec<_> = [0, 0, 0, 0, 0, 1, 0, 0, 2, 3]
        .into_iter()
        .map(|value| Complex::new(Float::with_val(256, value), Float::new(256)))
        .collect();
    values.push(Complex::new(Float::with_val(16, 1), Float::new(16)));
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        })
        .unwrap();
    let LaneStatus::Unstable(reason) = &result.statuses[0] else {
        panic!(
            "weak master scale must request rescue: {:?}",
            result.statuses
        );
    };
    assert!(reason.contains("active scalar-master"), "{reason}");

    settings.invariants = InvariantInput::SuppliedScalarProducts;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut values: Vec<_> = [0, 0, 0, 0, 0, 1, 0, 0]
        .into_iter()
        .map(|value| Complex::new(Float::with_val(16, value), Float::new(16)))
        .collect();
    values.extend(
        [0, 0, -1, 2, 3, 1]
            .into_iter()
            .map(|value| Complex::new(Float::with_val(256, value), Float::new(256))),
    );
    let mut workspace = Workspace::new(&plan, Float::new(256)).unwrap();
    workspace.set_accuracy_precision(128).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        })
        .unwrap();
    assert_eq!(result.statuses, [LaneStatus::Success]);
}

#[test]
fn original_scale_jet_precision_survives_a_rounded_zero_square_coefficient() {
    let mut settings = ReductionSettings::new(1, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new_dual(
        &plan,
        Float::new(256),
        HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]),
    )
    .unwrap();
    workspace.set_accuracy_precision(128).unwrap();
    let mut values = Vec::new();
    for scale_seed_bits in [80, 256] {
        for (argument, primal) in tadpole_row(256, 2).into_iter().enumerate() {
            let (primal, first, second) = if argument == 5 {
                (
                    Complex::new(
                        Float::with_val(256, 1) + Float::with_val(256, 2).pow(100).inv(),
                        Float::new(256),
                    ),
                    Complex::new(Float::with_val(256, 3), Float::new(256)),
                    Complex::new(
                        -Float::with_val(scale_seed_bits, 9) / Float::with_val(scale_seed_bits, 2),
                        Float::new(scale_seed_bits),
                    ),
                )
            } else {
                (
                    primal,
                    Complex::new(Float::new(256), Float::new(256)),
                    Complex::new(Float::new(256), Float::new(256)),
                )
            };
            values.extend([primal, first, second]);
        }
    }
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 2,
            components: 3,
            values,
        })
        .unwrap();
    if let (Ok(weak), Ok(strong)) = (result.lane(0), result.lane(1)) {
        println!(
            "original-scale second finite weak={} bits={} strong={} scaled_discrepancy={:e}",
            weak[11].re,
            weak[11].re.get_precision(),
            strong[11].re,
            (weak[11].re.at_precision(512) - strong[11].re.at_precision(512)).norm()
                / Float::with_val(512, 2),
        );
    }
    assert!(
        matches!(result.statuses[0], LaneStatus::Unstable(_)),
        "{:?}",
        result.statuses
    );
    assert_eq!(result.statuses[1], LaneStatus::Success);
}
