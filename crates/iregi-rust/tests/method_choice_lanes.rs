//! pyAmpliCol's method-choice lanes (m-lit lit-tir; fixture
//! `fixtures/method_choice_lanes.json`, generated from lit-tir's lane dumps):
//! every lane of g g > g g g whose binary64 method choice was off by more than
//! 1e-3 at TIRust main 3558433 (massless triangles, boxes and pentagon
//! sub-systems with lightlike legs, Gram conditioning 6e-5 .. 4e-3: the
//! last-resort alternative PV reduction divided by a modified Cayley
//! determinant that vanishes for these IR-divergent integrals), and the
//! g g > g g g g top-loop pentagon and hexagon lanes (sub-triangle Gram
//! conditioning 1.9e-6) the method choice serves. Each lane carries
//! pyAmpliCol's numerator coefficients (NinjaRS monomials `q0^a q1^b q2^c
//! q3^d`) and the truth `[finite, 1/eps, 1/eps^2]` by NinjaRS's generic
//! engine at 40 digits on the same binary64 inputs: with the exact S-matrix
//! for g g > g g g, with the S-matrix computed from the binary64 shifts for
//! the top-loop lanes (pyAmpliCol's S-matrix differs from those shifts by up
//! to 1.7e-11 relative, which moves NinjaRS, masters from the S-matrix and
//! integrand from the shifts, by 2.7e-4 and 2.5e-5 there; TIRust's 256-bit
//! engine gives the same value with either).
//!
//! The method choice as pyAmpliCol's TIR backend runs it
//! (`Convention::OneLoop`, `TensorWeights::Components`, gate 1e-15, boost
//! 12, with and without the alternative PV first) must serve a lane only
//! where it is accurate: within 1e-6 of the lane's size against TIRust's
//! 256-bit direct engine on the same inputs (invariants from the shifts or
//! the supplied S-matrix) and, with the supplied S-matrix, against the
//! truth. Refusing a lane (`Unstable`) is allowed; a wrong `Success` is not.

use iregi_rust::direct::EXPANSION_TOLERANCE;
use iregi_rust::{
    Convention, DirectReductionPlan, InvariantInput, KinematicsBatch, LaneStatus,
    ReductionSettings, TensorWeights,
};
use serde_json::Value;
use symbolica::domains::float::{Complex, Float, RealLike};

const FIXTURE: &str = include_str!("fixtures/method_choice_lanes.json");

/// Largest accepted error of a served lane (relative to the lane's size).
const SERVED_LIMIT: f64 = 1e-6;

struct Lane {
    label: String,
    n: usize,
    rank: u16,
    shifts: Vec<[f64; 4]>,
    masses: Vec<Complex<f64>>,
    mu2: f64,
    invariants: Vec<Complex<f64>>,
    coefficients: Vec<([u16; 4], Complex<f64>)>,
    truth: [Complex<f64>; 3],
}

fn complex(value: &Value) -> Complex<f64> {
    Complex::new(value[0].as_f64().unwrap(), value[1].as_f64().unwrap())
}

fn lanes() -> Vec<Lane> {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    fixture["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lane| Lane {
            label: lane["label"].as_str().unwrap().to_string(),
            n: lane["n"].as_u64().unwrap() as usize,
            rank: lane["rank"].as_u64().unwrap() as u16,
            shifts: lane["shifts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| std::array::from_fn(|a| v[a].as_f64().unwrap()))
                .collect(),
            masses: lane["masses"]
                .as_array()
                .unwrap()
                .iter()
                .map(complex)
                .collect(),
            mu2: lane["mu2"].as_f64().unwrap(),
            invariants: lane["invariants"]
                .as_array()
                .unwrap()
                .iter()
                .map(complex)
                .collect(),
            coefficients: lane["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        std::array::from_fn(|a| c[0][a].as_u64().unwrap() as u16),
                        complex(&c[1]),
                    )
                })
                .collect(),
            truth: std::array::from_fn(|slot| complex(&lane["truth"][slot])),
        })
        .collect()
}

fn settings(lane: &Lane, supplied: bool) -> ReductionSettings {
    let mut settings = ReductionSettings::new(lane.n, lane.rank);
    settings.convention = Convention::OneLoop;
    settings.tensor_weights = TensorWeights::Components;
    if supplied {
        settings.invariants = InvariantInput::SuppliedInvariants;
    }
    settings
}

fn row(lane: &Lane, supplied: bool) -> Vec<Complex<f64>> {
    let mut values = Vec::new();
    for shift in &lane.shifts {
        values.extend(shift.map(|x| Complex::new(x, 0.0)));
    }
    if supplied {
        values.extend_from_slice(&lane.invariants);
    }
    values.extend_from_slice(&lane.masses);
    values.push(Complex::new(lane.mu2.sqrt(), 0.0));
    values
}

/// pyAmpliCol's lane `[finite, 1/eps, 1/eps^2]` from a lane array.
fn contract(
    basis: &[iregi_rust::basis::Monomial],
    array: &[Complex<f64>],
    lane: &Lane,
) -> [Complex<f64>; 3] {
    let mut out = [Complex::new(0.0, 0.0); 3];
    for (alpha, coefficient) in &lane.coefficients {
        let b = basis
            .iter()
            .position(|m| m.0 == *alpha)
            .expect("monomial in the basis");
        let s = &array[4 * b..4 * b + 4];
        for (slot, value) in [s[3], s[0] + s[1], s[2]].into_iter().enumerate() {
            out[slot] += *coefficient * value;
        }
    }
    out
}

fn error(value: &[Complex<f64>; 3], truth: &[Complex<f64>; 3]) -> f64 {
    let size = |z: &Complex<f64>| z.re.hypot(z.im);
    let scale = truth.iter().map(size).fold(0.0, f64::max);
    value
        .iter()
        .zip(truth)
        .map(|(a, b)| size(&(*a - *b)))
        .fold(0.0, f64::max)
        / scale
}

/// The lane by the 256-bit direct engine (accepting conditioning down to
/// 2^-216).
fn reference(lane: &Lane, supplied: bool) -> [Complex<f64>; 3] {
    let lifted: Vec<Complex<Float>> = row(lane, supplied)
        .iter()
        .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
        .collect();
    let mut workspace = DirectReductionPlan::new(settings(lane, supplied))
        .unwrap()
        .workspace(Float::new(256))
        .unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: lifted,
        })
        .unwrap();
    assert_eq!(
        result.statuses[0],
        LaneStatus::Success,
        "{}: 256-bit reference",
        lane.label
    );
    let array: Vec<Complex<f64>> = result
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
        .collect();
    contract(&result.basis, &array, lane)
}

#[test]
fn method_choice_serves_pyamplicol_lanes_accurately_or_refuses_them() {
    let mut failures = Vec::new();
    let (mut served, mut refused) = (0usize, 0usize);
    for lane in lanes() {
        for supplied in [false, true] {
            let high = reference(&lane, supplied);
            if supplied {
                // The reference is sound where the inputs are exact.
                let e = error(&high, &lane.truth);
                assert!(
                    e <= 1e-8,
                    "{}: 256-bit reference off the truth by {e:e}",
                    lane.label
                );
            }
            let batch = KinematicsBatch {
                rows: 1,
                components: 1,
                values: row(&lane, supplied),
            };
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
                let value = contract(&result.basis, array, &lane);
                let e = error(&value, &high);
                let t = supplied.then(|| error(&value, &lane.truth));
                println!(
                    "{} supplied={supplied} alternative_first={alternative_first:?}: served, error {e:e} (truth {t:?})",
                    lane.label
                );
                if e > SERVED_LIMIT || t.is_some_and(|t| t > SERVED_LIMIT) {
                    failures.push(format!(
                        "{} supplied={supplied} alternative_first={alternative_first:?}: {e:e} (truth {t:?})",
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
