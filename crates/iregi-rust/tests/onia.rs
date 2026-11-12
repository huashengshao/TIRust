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
//!   K = 0). The direct engine reduces them by that partial fraction to
//!   pinches with a doubled gluon (its homogeneous-relation variant), whose
//!   IR-singular on-shell bases have closed forms. They must agree with
//!   original IREGI's arrays (quad IREGI, equal to the exact doubled-gluon
//!   combination of IREGI's general API; for the triangle also to the closed
//!   form of benchmarks/onia/coulomb_closed_form.py), and the triangle's
//!   scalar moment with the v = 0 value C0 = -(m^2)^(-1-eps) Gamma(1+eps) /
//!   (2 eps (1 + 2 eps)), independent of IREGI.
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
    evaluate_with_diagnostics(case, mu).0
}

/// The lane's array and basis ranks, and its facade diagnostics (empty when
/// the binary64 first pass accepts the lane, rotation test included).
fn evaluate_with_diagnostics(
    case: &Value,
    mu: f64,
) -> ((Result<Vec<Complex<f64>>, LaneStatus>, Vec<usize>), Vec<String>) {
    let settings = ReductionSettings::new(case["n"].as_u64().unwrap() as usize, case["rank"].as_u64().unwrap() as u16);
    let plan = TensorReductionPlan::new(settings).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let result = workspace.reduce_batch(&batch(case, mu)).unwrap();
    let ranks = result.basis.iter().map(|m| m.rank()).collect();
    let diagnostics = workspace.last_diagnostics().first().cloned().unwrap_or_default();
    ((result.lane(0).map(<[_]>::to_vec).map_err(Clone::clone), ranks), diagnostics)
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
fn coulomb_exchange_integrals_match_iregi_and_the_v0_c0() {
    let fixture = fixture();
    let mu = number(&fixture["mu"]);
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap().iter().filter(|c| c["kind"] == "coulomb") {
        let label = format!("{} {} #{}", case["process"], case["topology"], case["index"]);
        let ((values, ranks), diagnostics) = evaluate_with_diagnostics(case, mu);
        let values = values.unwrap_or_else(|status| panic!("{label}: {status:?}"));
        let n = case["n"].as_u64().unwrap() as usize;
        let ea = e_over_a(&values, &case["iregi_reference"], &ranks, n, mu);
        assert!(ea <= 1e-9, "{label}: E/A {ea:e} against original IREGI");
        // The binary64 pass accepts the lane, loop-rotation test included: a
        // rotated copy must find its own Coulomb variant (the raised member
        // is the gluon in every labelling), not fall back to a rescue.
        assert!(diagnostics.is_empty(), "{label}: binary64 pass refused: {diagnostics:?}");
        if n == 3 {
            // IREGI native: i pi^2 times OneLOop's normalization, the finite
            // part shifted by -(gamma_E + ln pi) times the single pole.
            let m2 = case["masses_squared"].as_array().unwrap().iter().map(number).fold(0.0f64, f64::max);
            let pi2 = std::f64::consts::PI.powi(2);
            let pole = -1.0 / (2.0 * m2);
            let finite = 1.0 / m2 - (mu * mu / m2).ln() / (2.0 * m2);
            let shift = 0.577_215_664_901_532_9 + std::f64::consts::PI.ln();
            let expected = [0.0, pi2 * pole, 0.0, pi2 * (finite - shift * pole)];
            for (slot, &want) in expected.iter().enumerate() {
                let z = &values[slot];
                let scale = expected.iter().map(|x| x.abs()).fold(0.0, f64::max);
                assert!(
                    z.re.abs() <= 1e-12 * scale && (z.im - want).abs() <= 1e-12 * scale,
                    "{label}: scalar slot {slot} = {z:?}, v = 0 C0 gives i {want:e}"
                );
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 5);
}

/// The Coulomb box of g g > c c~ and all its pinches, in the box's routing,
/// as one amplitude (benchmarks/onia/amplitude.py): every member succeeds,
/// sharing per-subset results is bit-identical to evaluating the members
/// separately (the homogeneous-relation variants share like any other), and
/// the box itself agrees with original IREGI.
#[test]
fn coulomb_family_in_amplitude_workspace() {
    use iregi_rust::AmplitudeWorkspace;
    let fixture = fixture();
    let mu = number(&fixture["mu"]);
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "coulomb" && c["process"] == "gg_QQ" && c["n"] == 4)
        .expect("the gg_QQ Coulomb box");
    let parent = batch(case, mu);
    let n = 4;
    let mut plans = Vec::new();
    let mut batches = Vec::new();
    let mut subsets = Vec::new();
    for mask in (1u32..(1 << n)).rev().filter(|m| m.count_ones() >= 2) {
        let members: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        plans.push(TensorReductionPlan::new(ReductionSettings::new(members.len(), members.len() as u16)).unwrap());
        let mut values = Vec::new();
        for &i in &members {
            values.extend_from_slice(&parent.values[4 * i..4 * i + 4]);
        }
        values.extend(members.iter().map(|&i| parent.values[4 * n + i]));
        values.push(parent.values[5 * n]);
        batches.push(KinematicsBatch { rows: 1, components: 1, values });
        subsets.push(members);
    }
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64).unwrap();
    let bits = |results: &[iregi_rust::BatchResult<f64>]| -> Vec<Vec<u64>> {
        results
            .iter()
            .map(|r| r.lane(0).expect("every member succeeds").iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect())
            .collect()
    };
    amplitude.set_subset_sharing(false);
    let separate = amplitude.reduce_batches(&batches).unwrap();
    amplitude.set_subset_sharing(true);
    let shared = amplitude.reduce_batches(&batches).unwrap();
    assert_eq!(bits(&separate), bits(&shared), "sharing changes the Coulomb family");
    assert_eq!(subsets[0].len(), 4);
    let box_values = shared[0].lane(0).unwrap().to_vec();
    let ranks: Vec<usize> = shared[0].basis.iter().map(|m| m.rank()).collect();
    let ea = e_over_a(&box_values, &case["iregi_reference"], &ranks, n, mu);
    assert!(ea <= 1e-9, "Coulomb box in the amplitude workspace: E/A {ea:e}");
}
