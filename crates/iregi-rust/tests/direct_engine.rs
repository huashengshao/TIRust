//! Direct numerical engine: full-array agreement with the archived opening
//! reference tensors (independently admitted against quadruple-precision
//! original IREGI), agreement with the existing engine on perturbed lanes,
//! the denominator-identity path and lane isolation.
use std::path::Path;

use iregi_rust::{
    DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionSettings, TensorReductionPlan,
    TensorWorkspace,
};
use serde_json::Value;
use symbolica::domains::{
    float::{Complex, SingleFloat},
    integer::Integer,
    rational::Rational,
};

const SUITE: &str = "../../validation/benchmark/opening-rank6-20261026";

fn exact(text: &str) -> f64 {
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let q = Rational::from((n.parse::<Integer>().unwrap(), d.parse::<Integer>().unwrap()));
    0.0f64.from_rational(&q)
}

struct Case {
    id: String,
    n: usize,
    rank: u16,
    row: Vec<Complex<f64>>,
    mu: f64,
    reference: Vec<Complex<f64>>,
}

fn ordinary_cases() -> Vec<Case> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(SUITE);
    let inputs: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("inputs.json")).unwrap()).unwrap();
    let mut cases = Vec::new();
    for case in inputs["cases"].as_array().unwrap() {
        if case["table"] != "ordinary" {
            continue;
        }
        let request = &case["driver_request"];
        let id = request["id"].as_str().unwrap().to_string();
        let sample = &request["samples"][0];
        let mut row = Vec::new();
        for shift in sample["shifts"].as_array().unwrap() {
            for component in shift.as_array().unwrap() {
                row.push(Complex::new(exact(component.as_str().unwrap()), 0.0));
            }
        }
        for mass in sample["masses_squared"].as_array().unwrap() {
            row.push(Complex::new(
                exact(mass[0].as_str().unwrap()),
                exact(mass[1].as_str().unwrap()),
            ));
        }
        let mu = exact(sample["mu"].as_str().unwrap());
        row.push(Complex::new(mu, 0.0));
        let result: Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("cases").join(&id).join("result.json")).unwrap(),
        )
        .unwrap();
        let reference = result["first_output"]["lanes"][0]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| {
                Complex::new(
                    exact(z[0]["exact_rational"].as_str().unwrap()),
                    exact(z[1]["exact_rational"].as_str().unwrap()),
                )
            })
            .collect();
        cases.push(Case {
            id,
            n: request["n"].as_u64().unwrap() as usize,
            rank: request["rank"].as_u64().unwrap() as u16,
            row,
            mu,
            reference,
        });
    }
    cases
}

/// Dimensionally rescaled whole-array discrepancy `E/A` (unit powers, h=0).
fn array_error(
    basis: &[iregi_rust::basis::Monomial],
    n: usize,
    mu: f64,
    value: &[Complex<f64>],
    reference: &[Complex<f64>],
) -> f64 {
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (m, monomial) in basis.iter().enumerate() {
        let weight = mu.powi(monomial.rank() as i32 + 4 - 2 * n as i32);
        for slot in 0..4 {
            let (a, b) = (value[4 * m + slot], reference[4 * m + slot]);
            error = error.max((a.re - b.re).hypot(a.im - b.im) / weight);
            amplitude = amplitude.max(b.re.hypot(b.im) / weight);
        }
    }
    error / amplitude
}

#[test]
fn ordinary_opening_rows_match_reference_arrays() {
    let cases = ordinary_cases();
    assert_eq!(cases.len(), 12);
    for case in &cases {
        let plan = DirectReductionPlan::new(ReductionSettings::new(case.n, case.rank)).unwrap();
        let mut workspace = plan.workspace(0.0f64).unwrap();
        let batch = KinematicsBatch {
            rows: 1,
            components: 1,
            values: case.row.clone(),
        };
        let result = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success, "{}", case.id);
        let lane = result.lane(0).unwrap();
        assert_eq!(lane.len(), case.reference.len(), "{}", case.id);
        let error = array_error(&result.basis, case.n, case.mu, lane, &case.reference);
        // The N5/N6 references and IREGI's own binary64 IBP agree to ~1e-11.
        let tolerance = if case.n <= 4 { 1e-13 } else { 1e-10 };
        assert!(error <= tolerance, "{}: E/A = {error:e}", case.id);
        // A repeated call on the same workspace is bitwise reproducible.
        let again = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(again.lane(0).unwrap(), lane, "{}", case.id);
    }
}

/// Mildly perturbed lanes, binary-exact, IR finite.
fn perturbed(row: &[Complex<f64>], n: usize, lane: usize) -> Vec<Complex<f64>> {
    let mut row = row.to_vec();
    let scale = 1.0 + lane as f64 / 256.0;
    for value in &mut row[..4 * n] {
        value.re *= scale;
    }
    for value in &mut row[4 * n..5 * n] {
        value.re += lane as f64 / 64.0;
    }
    row
}

#[test]
fn batches_match_the_existing_engine_and_isolate_invalid_lanes() {
    let cases = ordinary_cases();
    for id in ["ordinary-n2-r1", "ordinary-n3-r3", "ordinary-n4-r4"] {
        let case = cases.iter().find(|c| c.id == id).unwrap();
        let mut values = Vec::new();
        let lanes = 5;
        for lane in 0..lanes {
            let mut row = perturbed(&case.row, case.n, lane);
            if lane == 2 {
                // Invalid scale: this lane must fail alone.
                *row.last_mut().unwrap() = Complex::new(0.0, 0.0);
            }
            values.extend(row);
        }
        let batch = KinematicsBatch {
            rows: lanes,
            components: 1,
            values,
        };
        let settings = ReductionSettings::new(case.n, case.rank);
        let direct = DirectReductionPlan::new(settings.clone())
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&batch)
            .unwrap();
        let plan = TensorReductionPlan::new(settings).unwrap();
        let mut existing = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        let reference = existing.reduce_batch(&batch).unwrap();
        for lane in 0..lanes {
            if lane == 2 {
                assert!(direct.lane(lane).is_err(), "{id}: invalid lane accepted");
                continue;
            }
            let a = direct.lane(lane).unwrap();
            let b = reference.lane(lane).unwrap();
            let error = array_error(&direct.basis, case.n, case.mu, a, b);
            assert!(error <= 1e-12, "{id} lane {lane}: E/A = {error:e}");
        }
        let empty = DirectReductionPlan::new(ReductionSettings::new(case.n, case.rank))
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 1,
                values: Vec::new(),
            })
            .unwrap();
        assert_eq!(empty.rows(), 0);
    }
}

#[test]
fn degenerate_box_uses_the_constant_one_identity() {
    // Shifts 0, e_x, e_y, e_x+e_y: coplanar, so the box Gram matrix is
    // singular but D0 - D1 - D2 + D3 is a nonzero constant.
    let shifts = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]];
    let masses = [1.0, 2.0, 3.0, 3.0];
    let mut row = Vec::new();
    for shift in shifts {
        row.extend(shift.map(|x| Complex::new(x, 0.0)));
    }
    row.extend(masses.map(|m| Complex::new(m, 0.0)));
    row.push(Complex::new(3.0, 0.0));
    let rank = 3;
    let batch = KinematicsBatch {
        rows: 1,
        components: 1,
        values: row.clone(),
    };
    let direct = DirectReductionPlan::new(ReductionSettings::new(4, rank))
        .unwrap()
        .workspace(0.0f64)
        .unwrap()
        .reduce_batch(&batch)
        .unwrap();
    assert_eq!(direct.statuses[0], LaneStatus::Success);
    // Independent check: the same identity applied to the four triangles.
    // c = (1, -1, -1, 1) / K with K = sum c_i (q_i^2 - m_i^2).
    let q2m = |i: usize| {
        let s = shifts[i];
        s[0] * s[0] - s[1] * s[1] - s[2] * s[2] - s[3] * s[3] - masses[i]
    };
    let c = [1.0, -1.0, -1.0, 1.0];
    let k: f64 = (0..4).map(|i| c[i] * q2m(i)).sum();
    let mut expected = vec![Complex::new(0.0, 0.0); direct.lane(0).unwrap().len()];
    for omitted in 0..4 {
        let members: Vec<usize> = (0..4).filter(|&i| i != omitted).collect();
        let mut triangle = Vec::new();
        for &i in &members {
            triangle.extend(shifts[i].map(|x| Complex::new(x, 0.0)));
        }
        for &i in &members {
            triangle.push(Complex::new(masses[i], 0.0));
        }
        triangle.push(Complex::new(3.0, 0.0));
        let value = DirectReductionPlan::new(ReductionSettings::new(3, rank))
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: triangle,
            })
            .unwrap();
        let lane = value.lane(0).unwrap();
        for (target, source) in expected.iter_mut().zip(lane) {
            *target += *source * (c[omitted] / k);
        }
    }
    let error = array_error(&direct.basis, 4, 3.0, direct.lane(0).unwrap(), &expected);
    assert!(error <= 1e-13, "E/A = {error:e}");
}
