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
    powers: Vec<u16>,
    h: i16,
    row: Vec<Complex<f64>>,
    mu: f64,
    reference: Vec<Complex<f64>>,
}

fn ordinary_cases() -> Vec<Case> {
    opening_cases().into_iter().filter(|case| case.id.starts_with("ordinary")).collect()
}

fn opening_cases() -> Vec<Case> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(SUITE);
    let inputs: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("inputs.json")).unwrap()).unwrap();
    let mut cases = Vec::new();
    for case in inputs["cases"].as_array().unwrap() {
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
            powers: request["powers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_u64().unwrap() as u16)
                .collect(),
            h: request["dimension_shift"].as_i64().unwrap() as i16,
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
    general_array_error(basis, 4 - 2 * n as i32, mu, value, reference)
}

/// `E/A` with weights `mu^(rank + 4 + 2h - 2 sum(powers))`.
fn general_array_error(
    basis: &[iregi_rust::basis::Monomial],
    base: i32,
    mu: f64,
    value: &[Complex<f64>],
    reference: &[Complex<f64>],
) -> f64 {
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (m, monomial) in basis.iter().enumerate() {
        let weight = mu.powi(monomial.rank() as i32 + base);
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

#[test]
fn dotted_and_dimension_shifted_rows_match_reference_arrays() {
    let cases: Vec<_> = opening_cases()
        .into_iter()
        .filter(|case| case.id.starts_with("extended"))
        .collect();
    assert_eq!(cases.len(), 6);
    for case in &cases {
        let mut settings = ReductionSettings::new(case.n, case.rank);
        settings.powers = case.powers.clone();
        settings.dimension_shift = case.h;
        assert!(DirectReductionPlan::supports(&settings), "{}", case.id);
        let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(0.0f64).unwrap();
        let batch = KinematicsBatch {
            rows: 1,
            components: 1,
            values: case.row.clone(),
        };
        let result = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success, "{}", case.id);
        let base = 4 + 2 * i32::from(case.h)
            - 2 * case.powers.iter().map(|&p| i32::from(p)).sum::<i32>();
        let error = general_array_error(
            &result.basis,
            base,
            case.mu,
            result.lane(0).unwrap(),
            &case.reference,
        );
        // Original IREGI's own binary64 IBP reaches ~1e-13 on the N4R6 rows.
        assert!(error <= 1e-12, "{}: E/A = {error:e}", case.id);
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

/// One batch row from shifts (t,x,y,z), complex squared masses and mu.
fn row(shifts: &[[f64; 4]], masses: &[(f64, f64)], mu: f64) -> Vec<Complex<f64>> {
    let mut values = Vec::new();
    for shift in shifts {
        values.extend(shift.map(|x| Complex::new(x, 0.0)));
    }
    values.extend(masses.iter().map(|&(re, im)| Complex::new(re, im)));
    values.push(Complex::new(mu, 0.0));
    values
}

#[test]
fn lightlike_and_massless_kinematics_match_the_symbolic_engine() {
    // Exactly representable lightlike momenta (also k4 = -(k1+k2+k3)).
    let k1 = [5.0, 0.0, 0.0, 5.0];
    let k2 = [5.0, 0.0, 0.0, -5.0];
    let k3 = [-5.0, -3.0, 0.0, -4.0];
    let add = |a: [f64; 4], b: [f64; 4]| std::array::from_fn::<f64, 4, _>(|i| a[i] + b[i]);
    let zero = [0.0; 4];
    let q2 = add(k1, k2);
    let q3 = add(q2, k3);
    // (label, rank, shifts, masses)
    let cases: Vec<(&str, u16, Vec<[f64; 4]>, Vec<(f64, f64)>)> = vec![
        ("bubble unequal", 4, vec![zero, k1], vec![(1.0, 0.0), (2.0, 0.0)]),
        ("bubble equal", 4, vec![zero, k1], vec![(2.0, 0.0), (2.0, 0.0)]),
        ("bubble one massless", 3, vec![zero, k1], vec![(0.0, 0.0), (3.0, 0.0)]),
        ("bubble other massless", 3, vec![zero, k1], vec![(3.0, 0.0), (0.0, 0.0)]),
        ("bubble close masses", 4, vec![zero, k1], vec![(4.0, 0.0), (5.0, 0.0)]),
        ("bubble scaleless", 3, vec![zero, k1], vec![(0.0, 0.0), (0.0, 0.0)]),
        ("bubble complex", 3, vec![zero, k1], vec![(1.0, -0.1), (2.0, -0.3)]),
        ("ggH triangle", 3, vec![zero, k1, q2], vec![(3.0, 0.0); 3]),
        // The massless (IR-divergent) box is checked against original IREGI
        // in validation/benchmark/direct-engine-20261030/lightlike-check.py.
        // The ggHH-like box (old engine refuses it in binary64) is also in the
        // IREGI cross-check.
    ];
    for (label, rank, shifts, masses) in cases {
        let n = shifts.len();
        let batch = KinematicsBatch {
            rows: 1,
            components: 1,
            values: row(&shifts, &masses, 3.0),
        };
        let settings = ReductionSettings::new(n, rank);
        let direct = DirectReductionPlan::new(settings.clone())
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&batch)
            .unwrap();
        assert_eq!(direct.statuses[0], LaneStatus::Success, "{label}: {:?}", direct.statuses[0]);
        let plan = TensorReductionPlan::new_without_direct(settings).unwrap();
        let mut symbolic = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        let reference = symbolic.reduce_batch(&batch).unwrap();
        let reference = match reference.lane(0) {
            Ok(lane) => lane.to_vec(),
            Err(status) => {
                // The symbolic engine refuses a lightlike bubble with one
                // massless line. Check continuity instead: the m -> 0 closed
                // form against the generic branch at m^2 = 1e-20, which the
                // other cases validate against the symbolic engine.
                eprintln!("{label}: symbolic reference unavailable ({status:?}); checking m^2 -> 0 continuity");
                assert!(masses.iter().any(|m| *m == (0.0, 0.0)), "{label}: no reference");
                let mut nearby = masses.clone();
                for mass in &mut nearby {
                    if mass.0 == 0.0 && mass.1 == 0.0 {
                        mass.0 = 1e-20;
                    }
                }
                let near = DirectReductionPlan::new(ReductionSettings::new(n, rank))
                    .unwrap()
                    .workspace(0.0f64)
                    .unwrap()
                    .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: row(&shifts, &nearby, 3.0) })
                    .unwrap();
                near.lane(0).unwrap().to_vec()
            }
        };
        let error = array_error(&direct.basis, n, 3.0, direct.lane(0).unwrap(), &reference);
        eprintln!("{label}: E/A = {error:e}");
        assert!(error <= 1e-12, "{label}: E/A = {error:e}");
    }
}

#[test]
fn near_degenerate_boxes_are_rescued_at_higher_precision() {
    // Shifts 0, e_x, e_y, e_x+e_y+delta e_z: nearly coplanar. Binary64
    // refuses the Gram system. For delta = 1/64 and 1/1024 (rank 4) the
    // binary64 small-Gram expansion keeps the lane (E/A 6e-14 and 3e-14; the
    // binary64 PV recursion is off by up to 1e-4); for delta = 2^-24 (rank 6)
    // it does not, and the facade reruns the lane by the direct engine at the
    // precision its conditioning calls for (512 bits). The independent check
    // is a 512-bit direct evaluation of the same exactly lifted inputs.
    use iregi_rust::TensorEngineExecution;
    use symbolica::domains::float::{Float, RealLike};
    let masses = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 0.0)];
    for (rank, delta, bits) in [(4u16, 1.0 / 64.0, 106u32), (4, 1.0 / 1024.0, 256), (6, 1.0 / 16_777_216.0, 512)] {
        let shifts = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, delta]];
        let values = row(&shifts, &masses, 3.0);
        let settings = ReductionSettings::new(4, rank);
        let direct = DirectReductionPlan::new(settings.clone())
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() })
            .unwrap();
        assert!(matches!(direct.statuses[0], LaneStatus::Unstable(_)), "{:?}", direct.statuses[0]);
        let plan = TensorReductionPlan::new(settings.clone()).unwrap();
        let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        let result = workspace
            .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() })
            .unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        let TensorEngineExecution::Direct(stats) = workspace.engine_execution() else {
            panic!("expected the direct engine");
        };
        assert_eq!(stats.symbolic_rows, 0, "{stats:?}");
        // Either the binary64 small-Gram expansion (checked by the rotated
        // copy) or the precision rescue at the tier the conditioning calls for.
        let tolerance = if result.fallback_used[0] {
            let accepted = stats.precision_rescue.last().unwrap();
            assert_eq!((accepted.bits, accepted.successes), (bits, 1), "{stats:?}");
            1e-14
        } else {
            assert_eq!(stats.expansion_successes, 1, "{stats:?}");
            1e-10
        };
        let lifted: Vec<Complex<Float>> = values
            .iter()
            .map(|z| Complex::new(Float::with_val(512, z.re), Float::with_val(512, z.im)))
            .collect();
        let mut reference = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(512)).unwrap();
        reference.set_stability_threshold(Float::with_val(512, 2f64.powi(-472)));
        let reference = reference
            .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lifted })
            .unwrap();
        let reference: Vec<Complex<f64>> = reference
            .lane(0)
            .unwrap()
            .iter()
            .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
            .collect();
        let error = array_error(&result.basis, 4, 3.0, result.lane(0).unwrap(), &reference);
        assert!(error <= tolerance, "rank {rank} delta {delta:e}: E/A = {error:e}");
    }
}

/// Facade result of a top-loop pentagon (sqrt(s) = 1 TeV) and its E/A
/// against a 256-bit direct evaluation.
fn physical_pentagon(shifts: &[[f64; 4]; 5]) -> (iregi_rust::BatchResult<f64>, iregi_rust::DirectExecutionStats, f64) {
    use iregi_rust::TensorEngineExecution;
    use symbolica::domains::float::{Float, RealLike};
    let mt2 = 173.0f64 * 173.0;
    let values = row(shifts, &[(mt2, 0.0); 5], 1000.0);
    let settings = ReductionSettings::new(5, 5);
    let plan = TensorReductionPlan::new(settings.clone()).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() })
        .unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    let TensorEngineExecution::Direct(stats) = workspace.engine_execution() else {
        panic!("expected the direct engine");
    };
    let stats = stats.clone();
    assert_eq!(stats.rotation_rows, 1, "{stats:?}");
    let lifted: Vec<Complex<Float>> = values
        .iter()
        .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
        .collect();
    let mut reference = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(256)).unwrap();
    reference.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let reference = reference
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lifted })
        .unwrap();
    let reference: Vec<Complex<f64>> = reference
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
        .collect();
    let error = general_array_error(&result.basis, 4 - 2 * 5, 1000.0, result.lane(0).unwrap(), &reference);
    (result, stats, error)
}

#[test]
fn physical_pentagons_stay_accurate_in_binary64() {
    // gg -> Hgg top-loop pentagons (stability study points pentagon-r5 #0 and
    // #1569). Their gluon legs have binary64 invariants of ~1e-10 against
    // m_t^2 = 3e4. The general B0 expression cancelled like m^2/p^2 there
    // (E/A 1e-4 in #0) and the generated D0 returned NaN (#1569), so both
    // lanes used to be rescued at 106 bits. With oneloopmaster's small-momentum
    // B0 and invariants snapped to zero within their own rounding they stay in
    // binary64 and agree with 256 bits.
    let points: [[[f64; 4]; 5]; 2] = [
        [
            [0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 500.0],
            [1000.0, 0.0, 0.0, 0.0],
            [636.4389996374975, 200.64958343246792, -293.7037081332071, 75.1962597591425],
            [381.0252510177986, -5.258369435281395, -357.7820947151467, 130.9372527789567],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 500.0],
            [1000.0, 0.0, 0.0, 0.0],
            [646.3875758715252, -252.10932067317907, -153.54887423380967, -194.69304083416594],
            [243.0269224508989, 71.54019379005626, -232.21322716968106, -4.5937823063976],
        ],
    ];
    for shifts in points {
        let (result, stats, error) = physical_pentagon(&shifts);
        assert!(!result.fallback_used[0], "{stats:?}");
        assert!(error <= 1e-12, "E/A = {error:e}");
    }
}

#[test]
fn inaccurate_binary64_pentagons_are_caught_and_rescued() {
    // Point pentagon-r5 #212: its box Gram systems pass the binary64 gate of
    // rotation-tested plans, yet the recursion keeps only ~2.5e-9. The
    // facade's loop-rotation test must catch it; the rescued array must match
    // a 256-bit direct evaluation.
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [664.7752755465025, -264.68605966338976, 196.959906569897, 59.360769105269235],
        [366.1961747737811, -226.26921095725032, 197.98348258528515, -209.05602888970952],
    ];
    let (result, stats, error) = physical_pentagon(&shifts);
    assert!(result.fallback_used[0], "{stats:?}");
    assert!(error <= 1e-14, "E/A = {error:e}");
}

#[test]
fn gram_free_pentagon_is_stable_in_binary64() {
    // gg -> Hgg top-loop pentagon (stability study pilot point 16): the
    // four-dimensional Gram recursion refused it in binary64; the
    // Denner-Dittmaier reduction (only det X of the modified Cayley matrix
    // inverted) accepts it and agrees with a 256-bit evaluation.
    use symbolica::domains::float::{Float, RealLike};
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [738.7657509103415, 13.903324306995643, -192.0742500575477, -176.51490853602957],
        [345.579043544914, -190.85951632951324, 105.70685581443107, -267.99921827816706]
    ];
    let mt2 = 173.0f64 * 173.0;
    let values = row(&shifts, &[(mt2, 0.0); 5], 1000.0);
    let settings = ReductionSettings::new(5, 5);
    let plan = DirectReductionPlan::new(settings).unwrap();
    let direct = plan
        .workspace(0.0f64)
        .unwrap()
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() })
        .unwrap();
    assert_eq!(direct.statuses[0], LaneStatus::Success);
    let lifted: Vec<Complex<Float>> = values
        .iter()
        .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
        .collect();
    let mut reference = plan.workspace(Float::new(256)).unwrap();
    reference.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let reference = reference
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lifted })
        .unwrap();
    let reference: Vec<Complex<f64>> = reference
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
        .collect();
    let error = general_array_error(&direct.basis, 4 - 2 * 5, 1000.0, direct.lane(0).unwrap(), &reference);
    assert!(error <= 1e-11, "E/A = {error:e}");
}

/// Cartesian array of a direct evaluation at 256 bits (accepting
/// conditioning down to 2^-216), rounded to binary64.
fn direct_256(n: usize, rank: u16, values: &[Complex<f64>]) -> (Vec<iregi_rust::basis::Monomial>, Vec<Complex<f64>>) {
    use symbolica::domains::float::{Float, RealLike};
    let lifted: Vec<Complex<Float>> = values
        .iter()
        .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
        .collect();
    let mut workspace = DirectReductionPlan::new(ReductionSettings::new(n, rank))
        .unwrap()
        .workspace(Float::new(256))
        .unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let result = workspace
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lifted })
        .unwrap();
    let array = result
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
        .collect();
    (result.basis.clone(), array)
}

/// Facade result of a gg -> HH top-loop box (sqrt(s) = 1 TeV) with the given
/// precision rescue, its counters and notes, and its E/A against a 256-bit
/// direct evaluation (when the lane succeeded).
fn physical_box(
    shifts: &[[f64; 4]; 4],
    rank: u16,
    rescue: iregi_rust::PrecisionRescue,
) -> (iregi_rust::BatchResult<f64>, iregi_rust::DirectExecutionStats, Vec<String>, Option<f64>) {
    use iregi_rust::{TensorEngineExecution, TensorWorkspaceOptions};
    let mt2 = 173.0f64 * 173.0;
    let values = row(shifts, &[(mt2, 0.0); 4], 1000.0);
    let plan = TensorReductionPlan::new(ReductionSettings::new(4, rank)).unwrap();
    let options = TensorWorkspaceOptions { precision_rescue: rescue, ..Default::default() };
    let mut workspace = TensorWorkspace::new_with_options(&plan, 0.0f64, options).unwrap();
    assert_eq!(workspace.precision_rescue(), rescue);
    let result = workspace
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() })
        .unwrap();
    let TensorEngineExecution::Direct(stats) = workspace.engine_execution() else {
        panic!("expected the direct engine");
    };
    let stats = stats.clone();
    let notes = workspace.last_diagnostics()[0].clone();
    let error = result.lane(0).ok().map(|lane| {
        let (_, reference) = direct_256(4, rank, &values);
        general_array_error(&result.basis, 4 - 2 * 4, 1000.0, lane, &reference)
    });
    (result, stats, notes, error)
}

#[test]
fn gate_refused_binary64_box_is_kept_by_the_rotation_check() {
    // gg -> HH rank-6 box (stability point box-r6 #759): the a priori Gram
    // gate refuses it (estimate 3.8e-17), yet binary64 keeps E/A 3.8e-12.
    // The checked retry evaluates it below the gate with rotated propagators
    // too; the copies agree, so the binary64 array is returned, with the
    // precision rescue disabled or not.
    use iregi_rust::PrecisionRescue;
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [499.99999999999994, 21.6322892694395, -42.18555103069406, 481.7960391547444],
    ];
    for rescue in [PrecisionRescue::Disabled, PrecisionRescue::Automatic] {
        let (result, stats, notes, error) = physical_box(&shifts, 6, rescue);
        assert_eq!(result.statuses[0], LaneStatus::Success, "{notes:?}");
        assert!(!result.fallback_used[0], "{stats:?}");
        assert_eq!((stats.checked_rows, stats.checked_successes), (1, 1), "{stats:?}");
        assert!(stats.precision_rescue.is_empty(), "{stats:?}");
        let error = error.unwrap();
        assert!(error <= 1e-11, "E/A = {error:e}");
    }
}

#[test]
fn small_gram_box_is_kept_in_binary64_by_the_expansion() {
    // gg -> HH rank-4 box (box-r4 #5046, small Higgs pT): the PV recursion
    // keeps only E/A 1.1e-3 in binary64, below its gate or not. The small-Gram
    // expansion (Denner-Dittmaier) gets the box from its triangles in
    // binary64; the rotated copy agrees, so the lane stays binary64 with the
    // precision rescue disabled.
    use iregi_rust::PrecisionRescue;
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [499.9999999999999, 8.840408903567376, -6.981798561273453, -483.99184048831603],
    ];
    let (result, stats, notes, error) = physical_box(&shifts, 4, PrecisionRescue::Disabled);
    assert_eq!(result.statuses[0], LaneStatus::Success, "{notes:?}");
    assert!(!result.fallback_used[0], "{stats:?}");
    assert_eq!((stats.checked_rows, stats.checked_successes), (1, 0), "{stats:?}");
    assert_eq!((stats.expansion_rows, stats.expansion_successes), (1, 1), "{stats:?}");
    let error = error.unwrap();
    assert!(error <= 1e-10, "E/A = {error:e}");
}

#[test]
fn disabled_precision_rescue_returns_inaccurate_binary64_lanes_unstable() {
    // gg -> HH rank-4 box (box-r4 #8739): binary64 PV keeps E/A 1.2e-9 below
    // its gate, and the small-Gram expansion does not converge, so nothing in
    // binary64 meets the default request 2^-34. With the rescue disabled the
    // lane is refused, never rerun, with the reason in its diagnostics; by
    // default it is rescued at higher precision.
    use iregi_rust::PrecisionRescue;
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [499.99999999999994, 3.270999215852638, -7.550202940230169, 484.05298780163656],
    ];
    let (result, stats, notes, _) = physical_box(&shifts, 4, PrecisionRescue::Disabled);
    assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "{:?}", result.statuses[0]);
    assert!(!result.fallback_used[0], "{stats:?}");
    assert_eq!((stats.checked_rows, stats.checked_successes), (1, 0), "{stats:?}");
    assert_eq!((stats.expansion_rows, stats.expansion_successes), (1, 0), "{stats:?}");
    assert!(stats.precision_rescue.is_empty() && stats.symbolic_rows == 0, "{stats:?}");
    assert!(notes.iter().any(|note| note.contains("precision rescue disabled")), "{notes:?}");
    let (result, stats, _, error) = physical_box(&shifts, 4, PrecisionRescue::Automatic);
    assert_eq!(result.statuses[0], LaneStatus::Success);
    assert!(result.fallback_used[0], "{stats:?}");
    let error = error.unwrap();
    assert!(error <= 1e-14, "E/A = {error:e}");
}

#[test]
fn precision_rescue_switches_on_an_existing_workspace() {
    // Pentagon-r5 #212 (binary64 keeps ~2.5e-9, caught by the loop-rotation
    // test): the same workspace refuses it with the rescue disabled and
    // rescues it again once re-enabled.
    use iregi_rust::PrecisionRescue;
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [664.7752755465025, -264.68605966338976, 196.959906569897, 59.360769105269235],
        [366.1961747737811, -226.26921095725032, 197.98348258528515, -209.05602888970952],
    ];
    let mt2 = 173.0f64 * 173.0;
    let batch = KinematicsBatch { rows: 1, components: 1, values: row(&shifts, &[(mt2, 0.0); 5], 1000.0) };
    let plan = TensorReductionPlan::new(ReductionSettings::new(5, 5)).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    assert_eq!(workspace.precision_rescue(), PrecisionRescue::Automatic);
    workspace.set_precision_rescue(PrecisionRescue::Disabled);
    let refused = workspace.reduce_batch(&batch).unwrap();
    assert!(matches!(refused.statuses[0], LaneStatus::Unstable(_)), "{:?}", refused.statuses[0]);
    assert!(!refused.fallback_used[0]);
    workspace.set_precision_rescue(PrecisionRescue::Automatic);
    let rescued = workspace.reduce_batch(&batch).unwrap();
    assert_eq!(rescued.statuses[0], LaneStatus::Success);
    assert!(rescued.fallback_used[0]);
    let (_, reference) = direct_256(5, 5, &batch.values);
    let error = general_array_error(&rescued.basis, 4 - 2 * 5, 1000.0, rescued.lane(0).unwrap(), &reference);
    assert!(error <= 1e-14, "E/A = {error:e}");
}

#[test]
fn seven_point_rank_seven_heptagon_matches_high_precision() {
    // gg -> ggggg top-loop heptagon at sqrt(s) = 1 TeV (stability study point
    // heptagon-r7 #0). The thin identity over six of the seven members (the
    // best-conditioned choice per lane) reduces it to six hexagons.
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [908.1355877778915, 66.9766183664414, 52.69764788265707, 34.29519984990652],
        [607.9858386731116, 213.22343582276903, -153.34551353846933, 196.30740896744675],
        [533.5298788231431, 156.90681517490697, -114.13194173864258, 225.19380685096203],
        [270.01829905083963, 130.41975484643868, -236.33633415496422, -6.760660196794646],
    ];
    let mt2 = 173.0f64 * 173.0;
    let values = row(&shifts, &[(mt2, 0.0); 7], 1000.0);
    let (basis, reference) = direct_256(7, 7, &values);
    // (COLLIER and original IREGI cross-check heptagons in the stability
    // study; the old rank-0 seven-point wrapper takes minutes per point.)
    assert_eq!(basis.len(), 330);

    // Binary64 facade (direct engine, loop-rotation test, precision rescue).
    let plan = TensorReductionPlan::new(ReductionSettings::new(7, 7)).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() })
        .unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    let error = general_array_error(&basis, 4 - 2 * 7, 1000.0, result.lane(0).unwrap(), &reference);
    assert!(error <= 1e-12, "binary64 E/A = {error:e}");

    // Relabelled propagators (a different thin system, origin and pinch
    // order) give the same integral at 256 bits.
    for shift in [1, 3] {
        let rotated_shifts: Vec<[f64; 4]> = (0..7).map(|i| shifts[(i + shift) % 7]).collect();
        let rotated = row(&rotated_shifts, &[(mt2, 0.0); 7], 1000.0);
        let (_, array) = direct_256(7, 7, &rotated);
        let error = general_array_error(&basis, 4 - 2 * 7, 1000.0, &array, &reference);
        assert!(error <= 1e-15, "rotation by {shift}: E/A = {error:e}");
    }

    // The scalar heptagon stays in binary64 (no rescue).
    let (scalar_basis, scalar) = direct_256(7, 0, &values);
    let plan = TensorReductionPlan::new(ReductionSettings::new(7, 0)).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values })
        .unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    assert!(!result.fallback_used[0]);
    let error = general_array_error(&scalar_basis, 4 - 2 * 7, 1000.0, result.lane(0).unwrap(), &scalar);
    assert!(error <= 1e-13, "scalar binary64 E/A = {error:e}");
}
