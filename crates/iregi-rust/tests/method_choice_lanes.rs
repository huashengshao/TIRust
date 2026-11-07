//! pyAmpliCol's method-choice lanes (m-lit lit-tir; fixture
//! `fixtures/method_choice_lanes.json`, generated from lit-tir's lane dumps):
//! every lane of g g > g g g whose binary64 method choice was off by more than
//! 1e-3 at TIRust main 3558433 (massless triangles, boxes and pentagon
//! sub-systems with lightlike legs, Gram conditioning 6e-5 .. 4e-3: the
//! last-resort alternative PV reduction divided by a modified Cayley
//! determinant that vanishes for these IR-divergent integrals), and the
//! g g > g g g g top-loop pentagon and hexagon lanes (sub-triangle Gram
//! conditioning 1.9e-6) the method choice serves. The truth is NinjaRS's
//! generic engine at 40 digits on the same binary64 inputs: with the exact
//! S-matrix for g g > g g g (lightlike legs exactly 0), with the S-matrix
//! computed from the binary64 shifts for the top-loop lanes (pyAmpliCol's
//! S-matrix differs from those shifts by up to 1.7e-11 relative, which moves
//! NinjaRS, masters from the S-matrix and integrand from the shifts, by
//! 2.7e-4 and 2.5e-5 there; TIRust's 256-bit engine gives the same value with
//! either).
//!
//! The method choice as pyAmpliCol's TIR backend runs it (gate 1e-15, boost
//! 12, with and without the alternative PV first) must serve a lane only
//! where it is accurate: within 1e-6 of the lane's size against the truth,
//! in both invariant arms (from the shifts, where the binary64 lightlike rule
//! puts the lightlike legs on shell, and supplied). Refusing a lane
//! (`Unstable`) is allowed; a wrong `Success` is not. (TIRust's 256-bit
//! engine is checked against the truth with the supplied S-matrix only: from
//! the shifts it keeps the lightlike legs off shell, a different integral.)

mod common;

use common::{batch, contract, error, lanes, reference, settings};
use iregi_rust::DirectReductionPlan;
use iregi_rust::direct::EXPANSION_TOLERANCE;

const FIXTURE: &str = include_str!("fixtures/method_choice_lanes.json");

/// Largest accepted error of a served lane (relative to the lane's size).
const SERVED_LIMIT: f64 = 1e-6;

#[test]
fn method_choice_serves_pyamplicol_lanes_accurately_or_refuses_them() {
    let mut failures = Vec::new();
    let (mut served, mut refused) = (0usize, 0usize);
    for lane in lanes(FIXTURE) {
        let high = reference(&lane, true)
            .unwrap_or_else(|status| panic!("{}: 256-bit reference {status:?}", lane.label));
        let e = error(&high, &lane.truth);
        assert!(
            e <= 1e-8,
            "{}: 256-bit reference off the truth by {e:e}",
            lane.label
        );
        for supplied in [false, true] {
            let batch = batch(&lane, supplied);
            let plan = DirectReductionPlan::new(settings(&lane, supplied)).unwrap();
            for alternative_first in [None, Some(EXPANSION_TOLERANCE)] {
                let mut workspace = plan.workspace(0.0f64).unwrap();
                workspace.set_stability_threshold(1e-15);
                workspace.enable_estimated_method_choice(12, 1e-15);
                workspace.set_alternative_first(alternative_first);
                let result = workspace.reduce_batch(&batch).unwrap();
                let Ok(array) = result.lane(0) else {
                    refused += 1;
                    println!(
                        "{} supplied={supplied} alternative_first={alternative_first:?}: {:?}",
                        lane.label, result.statuses[0]
                    );
                    continue;
                };
                served += 1;
                let e = error(&contract(&result.basis, array, &lane), &lane.truth);
                println!(
                    "{} supplied={supplied} alternative_first={alternative_first:?}: served, error {e:e}",
                    lane.label
                );
                if e > SERVED_LIMIT {
                    failures.push(format!(
                        "{} supplied={supplied} alternative_first={alternative_first:?}: {e:e}",
                        lane.label
                    ));
                }
            }
        }
    }
    println!("served {served}, refused {refused}");
    assert!(
        failures.is_empty(),
        "method-choice lanes served beyond {SERVED_LIMIT:e}: {failures:#?}"
    );
}
