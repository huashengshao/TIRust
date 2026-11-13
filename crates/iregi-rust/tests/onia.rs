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
use iregi_rust::{
    DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionSettings, TensorReductionPlan, TensorWorkspace,
};
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

/// Physical S-wave amplitudes at one loop (benchmarks/onia/swave.py; fixture
/// by make_swave_fixture.py): g g > c c~ [1S0] colour singlet and octet at
/// points 0 and 5, and g g > c c~ [3S1, singlet] g and [3S1, octet] g at one
/// point each (the octet with numeric colour wavefunctions). Each case
/// holds the FeynKit diagrams' S-wave-projected numerators, contracted per
/// distinct loop integral (tree factors included) at fixed polarizations.
/// TIRust's amplitude X (native slots [UV, IR1, IR2, finite], X / (2 pi)^4
/// at the poles, G = 1) must agree with the reference amplitude to 1e-9 and
/// satisfy, against the Born A0 with n external gluons, the pole structure of
/// the bare amplitude: X_IR2 = -n C_A pi^2 A0; IR1/IR2 + gamma_E + ln(pi) -
/// (the fixture's log part, sum of ln(mu^2/(-s_ij)) terms) = the fixture's
/// constant (1 - C_F/C_A singlet, 3/2 - C_F/C_A octet, 1 - 2 C_F/(3 C_A) for
/// 3S1), or for the g g > c c~ g octet X_IR1 + (gamma_E + ln pi) X_IR2 = the
/// fixture's colour-space Becher-Neubert value pi^2 S A0; X_UV = pi^2 [(n C_A
/// + C_F) A0 + 3 C_F m dA0/dm]. Point 5 is
/// boosted, and its soft triangles are IR-divergent only at the exact lattice
/// m^2 (an ulp-off mass regulates them). The 3S1 case has 84 integrals,
/// pentagons included, among them the Coulomb pentagons on which original
/// IREGI returns no array.
#[test]
fn swave_amplitudes_match_the_reference_and_the_pole_structure() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-swave-amplitude.json")).unwrap();
    let abs = |z: Complex<f64>| z.norm_squared().sqrt();
    let pair = |v: &Value| Complex::new(number(&v[0]), number(&v[1]));
    let (ca, cf) = (3.0, 4.0 / 3.0);
    let pi2 = std::f64::consts::PI.powi(2);
    let euler_pi = 0.577_215_664_901_532_9 + std::f64::consts::PI.ln();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for case in cases {
        let label = format!("{} point {}", case["state"], case["point"]);
        let mu = number(&case["mu"]);
        let mut x = [Complex::new(0.0, 0.0); 4];
        for lane in case["lanes"].as_array().unwrap() {
            let (moments, _) = evaluate(lane, mu);
            let moments = moments.unwrap_or_else(|status| panic!("{label}: lane fails: {status:?}"));
            for term in lane["coefficients"].as_array().unwrap() {
                let m = term[0].as_u64().unwrap() as usize;
                let coefficient = Complex::new(number(&term[1]), number(&term[2]));
                for (slot, value) in x.iter_mut().enumerate() {
                    *value += coefficient * moments[4 * m + slot];
                }
            }
        }
        let reference: Vec<_> = (0..4).map(|slot| pair(&case["reference"][slot])).collect();
        let scale = reference.iter().map(|&z| abs(z)).fold(0.0, f64::max);
        let error = (0..4).map(|slot| abs(x[slot] - reference[slot])).fold(0.0, f64::max) / scale;
        assert!(error <= 1e-9, "{label}: amplitude E/A {error:e}");

        let (born, born_dm) = (pair(&case["born"]), pair(&case["born_mass_derivative"]));
        let n = case["gluons"].as_u64().unwrap() as f64;
        let double = abs(x[2] / (born * Complex::new(-n * ca * pi2, 0.0)) - Complex::new(1.0, 0.0));
        assert!(double <= 1e-10, "{label}: double pole {double:e}");
        if case.get("single_pole_prediction").is_some() {
            // Octet: the MS-bar single pole X_IR1 + (gamma_E + ln pi) X_IR2
            // against the colour-space prediction pi^2 S A0.
            let prediction = pair(&case["single_pole_prediction"]);
            let single = abs(x[1] + x[2] * Complex::new(euler_pi, 0.0) - prediction) / abs(prediction);
            assert!(single <= 1e-9, "{label}: single pole {single:e}");
        } else {
            let ratio = x[1] / x[2] + Complex::new(euler_pi, 0.0) - pair(&case["single_pole_log"]);
            let single = abs(ratio - pair(&case["single_pole_target"]));
            assert!(single <= 1e-9, "{label}: single pole ratio {ratio:?}");
        }
        let uv_born = (born * Complex::new(n * ca + cf, 0.0) + born_dm * Complex::new(3.0 * cf, 0.0)) * Complex::new(pi2, 0.0);
        let uv = abs(x[0] / uv_born - Complex::new(1.0, 0.0));
        assert!(uv <= 1e-10, "{label}: UV pole {uv:e}");
    }
}

/// Kinematics of one lane from shifts, masses squared and mu.
fn coulomb_lane(shifts: &[[f64; 4]], masses: &[f64], mu: f64) -> Vec<Complex<f64>> {
    let mut values: Vec<Complex<f64>> =
        shifts.iter().flat_map(|shift| shift.iter().map(|&x| Complex::new(x, 0.0))).collect();
    values.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
    values.push(Complex::new(mu, 0.0));
    values
}

fn plus(a: [f64; 4], b: [f64; 4], s: f64) -> [f64; 4] {
    std::array::from_fn(|i| a[i] + s * b[i])
}

/// Whole-array E/A without rescaling (mu fixed).
fn relative_difference(values: &[Complex<f64>], reference: &[Complex<f64>]) -> f64 {
    assert_eq!(values.len(), reference.len());
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (z, r) in values.iter().zip(reference) {
        error = error.max((z.re - r.re).hypot(z.im - r.im));
        amplitude = amplitude.max(r.re.hypot(r.im));
    }
    error / amplitude
}

fn bits(values: &[Complex<f64>]) -> Vec<u64> {
    values.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect()
}

/// Statuses and arrays of a binary64 `DirectWorkspace` batch.
fn direct_lanes(n: usize, rank: u16, lanes: &[Vec<Complex<f64>>]) -> Vec<(LaneStatus, Vec<Complex<f64>>)> {
    let mut workspace =
        DirectReductionPlan::new(ReductionSettings::new(n, rank)).unwrap().workspace(0.0f64).unwrap();
    let batch = KinematicsBatch { rows: lanes.len(), components: 1, values: lanes.concat() };
    let result = workspace.reduce_batch(&batch).unwrap();
    (0..lanes.len())
        .map(|row| (result.statuses[row].clone(), result.lane(row).map(<[_]>::to_vec).unwrap_or_default()))
        .collect()
}

/// The facade's status and array of one lane.
fn facade_lane(settings: ReductionSettings, lane: &[Complex<f64>]) -> (LaneStatus, Vec<Complex<f64>>) {
    let plan = TensorReductionPlan::new(settings).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let batch = KinematicsBatch { rows: 1, components: 1, values: lane.to_vec() };
    let result = workspace.reduce_batch(&batch).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

/// Pentagons with the Coulomb triple D1, D2, D3 (shifts q1, q1 + k, q1 + 2k,
/// masses m^2, 0, m^2 with k^2 = m^2: D1 + D3 = 2 D2) at the rank given:
/// 1/(D1 D2 D3) = [1/(D2^2 D3) + 1/(D1 D2^2)] / 2, so the reference is half
/// the sum of the two boxes with the gluon doubled, from the facade.
fn coulomb_pentagon_reference(shifts: &[[f64; 4]], masses: &[f64], mu: f64, rank: u16) -> Vec<Complex<f64>> {
    let mut total: Vec<Complex<f64>> = Vec::new();
    for dropped in [1usize, 3] {
        let keep: Vec<usize> = (0..5).filter(|&i| i != dropped).collect();
        let mut settings = ReductionSettings::new(4, rank);
        settings.powers[keep.iter().position(|&i| i == 2).unwrap()] = 2;
        let lane = coulomb_lane(
            &keep.iter().map(|&i| shifts[i]).collect::<Vec<_>>(),
            &keep.iter().map(|&i| masses[i]).collect::<Vec<_>>(),
            mu,
        );
        let (status, values) = facade_lane(settings, &lane);
        assert_eq!(status, LaneStatus::Success, "doubled-gluon box without D{dropped}, rank {rank}");
        if total.is_empty() {
            total = values.iter().map(|z| Complex::new(0.5 * z.re, 0.5 * z.im)).collect();
        } else {
            for (t, z) in total.iter_mut().zip(&values) {
                *t = Complex::new(t.re + 0.5 * z.re, t.im + 0.5 * z.im);
            }
        }
    }
    total
}

/// A Coulomb pentagon's lane does not depend on the other lanes of its
/// batch (PR #32 review): a plan variant needs master families its base plan
/// has not (the rank-1 base pentagon has only C0 and D0), and their
/// evaluators are created on first use instead of the variant reading the
/// previous lane's values. Every rank agrees with the partial fraction, the
/// scalar pentagon included (its singular Cayley matrix leads to the
/// relation search), alone, after or before a generic pentagon, twice, and
/// through the facade.
#[test]
fn coulomb_lanes_do_not_depend_on_their_batch() {
    let (k, q1, q4) = ([5.0, 0.0, 0.0, 4.0], [1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 3.0, -1.0]);
    let shifts = [[0.0; 4], q1, plus(q1, k, 1.0), plus(q1, k, 2.0), q4];
    let masses = [0.0, 9.0, 0.0, 9.0, 2.0];
    let mu = 3.0;
    let coulomb = coulomb_lane(&shifts, &masses, mu);
    let generic = coulomb_lane(
        &[[0.0; 4], [1.0, 0.3, 0.2, 0.1], [2.5, -0.7, 0.4, 1.1], [0.6, 1.9, -0.8, 0.3], [-1.2, 0.5, 1.7, -0.4]],
        &[1.0, 2.0, 0.5, 3.0, 1.5],
        mu,
    );
    for rank in 0u16..=3 {
        let reference = coulomb_pentagon_reference(&shifts, &masses, mu, rank);
        let alone = direct_lanes(5, rank, &[coulomb.clone()]).remove(0);
        assert_eq!(alone.0, LaneStatus::Success, "rank {rank}");
        let ea = relative_difference(&alone.1, &reference);
        assert!(ea <= 1e-12, "Coulomb pentagon alone, rank {rank}: E/A {ea:e}");
        let generic_alone = direct_lanes(5, rank, &[generic.clone()]).remove(0);
        for (lanes, at) in [
            (vec![generic.clone(), coulomb.clone()], 1),
            (vec![coulomb.clone(), generic.clone()], 0),
            (vec![coulomb.clone(), coulomb.clone()], 1),
        ] {
            let results = direct_lanes(5, rank, &lanes);
            assert_eq!(results[at].0, LaneStatus::Success, "rank {rank}");
            assert_eq!(bits(&results[at].1), bits(&alone.1), "Coulomb lane in a batch, rank {rank}");
            let other = &results[1 - at];
            if lanes[1 - at] == generic {
                assert_eq!(bits(&other.1), bits(&generic_alone.1), "generic lane next to a Coulomb lane, rank {rank}");
            }
        }
        let (status, values) = facade_lane(ReductionSettings::new(5, rank), &coulomb);
        assert_eq!(status, LaneStatus::Success, "facade, rank {rank}");
        let ea = relative_difference(&values, &reference);
        assert!(ea <= 1e-12, "Coulomb pentagon through the facade, rank {rank}: E/A {ea:e}");
    }
}

/// Partial fractions run before the recursions that read them (PR #32
/// review): a planar pentagon (shifts in the (t, z) plane) is a relation
/// subset whose pinch {1, 2, 3, 4} holds the Coulomb triple. Its moments read
/// that pinch's partial fraction; before, they read it before it was written
/// (two copies in one batch differed by O(1), and the facade accepted a wrong
/// value at 256 bits). Lanes the engine accepts agree with the partial
/// fraction; the scalar case is refused (its Coulomb box reaches the scalar
/// D0 master without a Gram check), never wrong.
#[test]
fn coulomb_partial_fractions_precede_their_parents() {
    let (k, q1, q4) = ([5.0, 0.0, 0.0, 4.0], [2.0, 0.0, 0.0, 1.0], [-2.0, 0.0, 0.0, 3.0]);
    let shifts = [[0.0; 4], q1, plus(q1, k, 1.0), plus(q1, k, 2.0), q4];
    let masses = [0.0, 9.0, 0.0, 9.0, 2.0];
    let mu = 3.0;
    let planar = coulomb_lane(&shifts, &masses, mu);
    for rank in 0u16..=2 {
        let reference = coulomb_pentagon_reference(&shifts, &masses, mu, rank);
        let twice = direct_lanes(5, rank, &[planar.clone(), planar.clone()]);
        assert_eq!(bits(&twice[0].1), bits(&twice[1].1), "two copies of the planar lane, rank {rank}");
        let (status, values) = facade_lane(ReductionSettings::new(5, rank), &planar);
        for (what, status, values) in [("direct", &twice[0].0, &twice[0].1), ("facade", &status, &values)] {
            if rank >= 1 {
                assert_eq!(*status, LaneStatus::Success, "{what}, rank {rank}");
            }
            if *status == LaneStatus::Success {
                let ea = relative_difference(values, &reference);
                assert!(ea <= 1e-12, "planar Coulomb pentagon, {what}, rank {rank}: E/A {ea:e}");
            }
        }
    }
}

/// Coulomb triangles and boxes at every rank: each accepted rank agrees with
/// the leading moments of the full-rank array (the homogeneous variant at
/// rank N). At rank 0 the scalar master has no Gram check and the lane may be
/// refused (oneloopmaster's master is not finite there), never wrong.
#[test]
fn coulomb_integrals_agree_across_ranks() {
    let (k, q1) = ([5.0, 0.0, 0.0, 4.0], [1.0, 2.0, -1.0, 0.5]);
    let cases = [
        (vec![q1, plus(q1, k, 1.0), plus(q1, k, 2.0)], vec![9.0, 0.0, 9.0]),
        (vec![[0.0; 4], q1, plus(q1, k, 1.0), plus(q1, k, 2.0)], vec![2.0, 9.0, 0.0, 9.0]),
    ];
    for (shifts, masses) in cases {
        let n = shifts.len();
        let lane = coulomb_lane(&shifts, &masses, 3.0);
        let full = direct_lanes(n, n as u16, &[lane.clone()]).remove(0);
        assert_eq!(full.0, LaneStatus::Success, "N{n} at rank {n}");
        for rank in 0..n as u16 {
            let (status, values) = direct_lanes(n, rank, &[lane.clone()]).remove(0);
            if rank >= 1 {
                assert_eq!(status, LaneStatus::Success, "N{n} at rank {rank}");
            }
            if status == LaneStatus::Success {
                let ea = relative_difference(&values, &full.1[..values.len()]);
                assert!(ea <= 1e-12, "N{n} at rank {rank} against rank {n}: E/A {ea:e}");
            }
        }
    }
}
