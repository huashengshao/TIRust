//! S-wave quarkonium one-loop integrals (benchmarks/onia): the heavy quark
//! and antiquark carry the same momentum, p_c = p_c~ = P/2, in g g > c c~ and
//! g g > c c~ g. On these exact lattice inputs every Gram determinant that
//! involves both heavy-quark legs (and every pentagon's) vanishes exactly.
//!
//! The fixture (`benchmarks/onia/make_fixture.py`) holds two kinds of lanes:
//! - degenerate: affinely dependent shifts with a nonzero constant in
//!   sum_i c_i D_i = K. The reference is that exact identity applied
//!   recursively down to non-degenerate integrals, each confirmed by quad
//!   original IREGI (or by COLLIER's scalar master where quad IREGI's bundled
//!   OneLOop is wrong). It is independent of TIRust's handling of the
//!   degenerate kinematics; original IREGI itself is off by up to 1e+72 on
//!   such pentagons.
//! - Coulomb exchange: linearly dependent propagators (2 D_g = D_c + D_c~,
//!   K = 0). TIRust does not support these yet. They must come back refused
//!   per lane, typed, never as an array. The fixture keeps original IREGI's
//!   array (consistent with its partial-fraction combination) for when they
//!   are supported.
use iregi_rust::{KinematicsBatch, LaneStatus, ReductionSettings, TensorReductionPlan, TensorWorkspace};
use serde_json::Value;
use symbolica::domains::float::Complex;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/onia-swave.json")).unwrap()
}

fn number(value: &Value) -> f64 {
    value.as_str().expect("decimal string").parse().unwrap()
}

fn batch(case: &Value, mu: f64) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    for shift in case["shifts"].as_array().unwrap() {
        values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(number(x), 0.0)));
    }
    values.extend(case["masses_squared"].as_array().unwrap().iter().map(|m| Complex::new(number(m), 0.0)));
    values.push(Complex::new(mu, 0.0));
    KinematicsBatch { rows: 1, components: 1, values }
}

/// Whole-array E/A, moments rescaled by mu^(|alpha| + 4 - 2N) (as in
/// benchmarks/onia/run.py).
fn e_over_a(values: &[Complex<f64>], reference: &Value, ranks: &[usize], n: usize, mu: f64) -> f64 {
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (m, &rank) in ranks.iter().enumerate() {
        let weight = mu.powi(rank as i32 + 4 - 2 * n as i32);
        for slot in 0..4 {
            let pair = &reference[m][slot];
            let (re, im) = (number(&pair[0]), number(&pair[1]));
            let z = &values[4 * m + slot];
            error = error.max((z.re - re).hypot(z.im - im) / weight);
            amplitude = amplitude.max(re.hypot(im) / weight);
        }
    }
    error / amplitude
}

fn evaluate(case: &Value, mu: f64) -> (Result<Vec<Complex<f64>>, LaneStatus>, Vec<usize>) {
    let settings = ReductionSettings::new(case["n"].as_u64().unwrap() as usize, case["rank"].as_u64().unwrap() as u16);
    let plan = TensorReductionPlan::new(settings).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let result = workspace.reduce_batch(&batch(case, mu)).unwrap();
    let ranks = result.basis.iter().map(|m| m.rank()).collect();
    (result.lane(0).map(<[_]>::to_vec).map_err(Clone::clone), ranks)
}

#[test]
fn degenerate_quarkonium_integrals_match_the_exact_identity_reference() {
    let fixture = fixture();
    let mu = number(&fixture["mu"]);
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap().iter().filter(|c| c["kind"] == "degenerate") {
        let label = format!("{} {} #{}", case["process"], case["topology"], case["index"]);
        let (values, ranks) = evaluate(case, mu);
        let values = values.unwrap_or_else(|status| panic!("{label}: {status:?}"));
        let n = case["n"].as_u64().unwrap() as usize;
        let ea = e_over_a(&values, &case["reference"], &ranks, n, mu);
        assert!(ea <= 1e-9, "{label}: E/A {ea:e} against the identity reference");
        checked += 1;
    }
    assert_eq!(checked, 12);
}

#[test]
fn coulomb_exchange_integrals_are_refused_per_lane() {
    let fixture = fixture();
    let mu = number(&fixture["mu"]);
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap().iter().filter(|c| c["kind"] == "coulomb") {
        let label = format!("{} {} #{}", case["process"], case["topology"], case["index"]);
        let (values, _) = evaluate(case, mu);
        match values {
            Err(LaneStatus::Unstable(_) | LaneStatus::MasterFailure(_)) => {}
            other => panic!("{label}: expected a typed refusal, got {other:?}"),
        }
        assert!(case["iregi_reference"].is_array(), "{label}: IREGI array kept for future support");
        checked += 1;
    }
    assert_eq!(checked, 3);
}
