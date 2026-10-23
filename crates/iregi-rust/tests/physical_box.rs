//! Reducer-only regression for the canonical massive gg -> HH box.
//!
//! Its lightlike edges select an optional exact-zero IBP profile. Equal internal
//! masses make that profile singular in pinched bubbles; ordinary PV must still
//! get the opportunity to evaluate the sample through its analytic bubble jets.

use iregi_rust::{
    Algorithm, Convention, KinematicsBatch, LaneStatus, PrecisionRescueWorkspace, ReductionPlan,
    ReductionSettings, RescueSettings, RescueStatus,
};
use symbolica::domains::float::Complex;

fn number(value: &serde_json::Value) -> f64 {
    value
        .as_str()
        .map(|value| value.parse().unwrap())
        .unwrap_or_else(|| value.as_f64().unwrap())
}

#[test]
fn physical_equal_mass_box_requires_verified_precision_rescue() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../validation/physical/gghh-tensor-reference.json"
    ))
    .unwrap();
    let input = &fixture["input"];
    let propagators = input["propagators"].as_array().unwrap();
    let mut settings = ReductionSettings::new(propagators.len(), 4);
    settings.algorithm = Algorithm::Hybrid;
    settings.convention = Convention::IregiNative;
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = PrecisionRescueWorkspace::new(
        &plan,
        0.0_f64,
        RescueSettings {
            working_bits: 128,
            verification_bits: 256,
            accuracy_bits: Some(53),
            ..Default::default()
        },
    )
    .unwrap();
    let mut values = Vec::new();
    let mut scale = number(&input["mu"]);
    for propagator in propagators {
        for component in propagator["shift"].as_array().unwrap() {
            let component = number(component);
            scale = scale.max(component.abs());
            values.push(Complex::new(component, 0.0));
        }
    }
    for propagator in propagators {
        let mass = &propagator["mass_squared"];
        let (re, im) = (number(&mass[0]), number(&mass[1]));
        scale = scale.max(re.hypot(im).sqrt());
        values.push(Complex::new(re, im));
    }
    values.push(Complex::new(number(&input["mu"]), 0.0));
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        })
        .unwrap();
    // Do not turn the binary64 rejection into an implicit binary64 success.
    assert!(matches!(
        workspace.last_primary_statuses()[0],
        LaneStatus::Unstable(_)
    ));
    assert_eq!(
        result.statuses,
        [LaneStatus::Success],
        "{:?}",
        workspace.last_rescue_statuses()
    );
    assert_eq!(workspace.last_execution().accepted_rows, 1);
    assert_eq!(
        workspace.last_rescue_statuses(),
        &[RescueStatus::Converged {
            working_bits: 128,
            verification_bits: 256,
            primary_agreed: None,
        }]
    );
    let expected = fixture["expected"]["moments"].as_array().unwrap();
    let actual = result.lane(0).unwrap();
    assert_eq!(expected.len(), 70);
    assert_eq!(actual.len(), expected.len() * 4);
    for (moment, poles) in expected.iter().enumerate() {
        let natural = scale.powi(plan.basis()[moment].rank() as i32 - 4);
        for pole in 0..4 {
            let (re, im) = (number(&poles[pole][0]), number(&poles[pole][1]));
            let value = &actual[4 * moment + pole];
            let error = (value.re / natural - re / natural)
                .hypot(value.im / natural - im / natural)
                / (re / natural).hypot(im / natural).max(1.0);
            assert!(
                error.is_finite() && error <= 1e-10,
                "moment {moment}, pole {pole}: normalized error {error}; got ({},{}), expected ({re},{im})",
                value.re,
                value.im
            );
        }
    }
}
