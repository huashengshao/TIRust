//! Lanes the direct engine's default gate serves, reported from pyAmpliCol
//! (m-lit lit-tir fix 1) as wrong beyond 1e-5 with every check blind: two
//! g g > g g g g boxes at a point with sub-triangle Gram conditioning ~1e-5
//! (massless, and a top loop) and 111 lanes of three over-rank synthetic
//! rows (LI-32 N = 4 at R = 5, LI-33 N = 5 at R = 6, massive and massless
//! loops; random numerator coefficients), up to 5.5e-4 against NinjaRS's
//! generic engine at 40 digits. That truth used the on-shell S-matrix with
//! binary64 shifts it is inconsistent with (by up to 2.3e-10 relative);
//! NinjaRS, masters from the S-matrix and integrand from the shifts, moves
//! by up to 5.4e-4 on such inputs at any precision. On inputs consistent
//! with the binary64 shifts (lightlike pairs made exactly lightlike by
//! moving energies ~1e-16, the S-matrix derived exactly) every one of the
//! 113 lanes as served is within 7.0e-6 of the truth (11 beyond 1e-6, all on
//! the R = 5 rows), and within about 2 times (massive) to 5 times
//! (massless, at 4e-8) the larger of pyAmpliCol's two re-test estimates.
//!
//! The fixture (`fixtures/gate_passed_lanes.json`) keeps the two boxes and,
//! per synthetic row, the lane farthest from that truth. The default
//! workspace (pyAmpliCol's settings) must serve each lane, invariants from
//! the shifts or supplied, within 1e-5 of its size or refuse it.

mod common;

use common::{batch, contract, error, lanes, reference, settings};
use iregi_rust::DirectReductionPlan;

const FIXTURE: &str = include_str!("fixtures/gate_passed_lanes.json");

/// Largest accepted error of a served lane (relative to the lane's size).
const SERVED_LIMIT: f64 = 1e-5;

#[test]
fn gate_passed_lanes_are_accurate_or_refused() {
    let mut failures = Vec::new();
    for lane in lanes(FIXTURE) {
        for supplied in [false, true] {
            let plan = DirectReductionPlan::new(settings(&lane, supplied)).unwrap();
            let result = plan
                .workspace(0.0f64)
                .unwrap()
                .reduce_batch(&batch(&lane, supplied))
                .unwrap();
            let Ok(array) = result.lane(0) else {
                println!(
                    "{} supplied={supplied}: {:?}",
                    lane.label, result.statuses[0]
                );
                continue;
            };
            let e = error(&contract(&result.basis, array, &lane), &lane.truth);
            println!(
                "{} supplied={supplied}: served, error {e:e} (256-bit engine {:?})",
                lane.label,
                reference(&lane, supplied).map(|high| error(&high, &lane.truth))
            );
            if e > SERVED_LIMIT {
                failures.push(format!("{} supplied={supplied}: {e:e}", lane.label));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gate-passed lanes served beyond {SERVED_LIMIT:e}: {failures:#?}"
    );
}
