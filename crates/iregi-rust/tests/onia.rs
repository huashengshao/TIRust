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
use symbolica::domains::float::{Complex, Float, RealLike};

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

const BITS: u32 = 256;

/// The direct engine at 256 bits on one lane (as benchmarks/onia's
/// direct256).
fn direct256_float(settings: ReductionSettings, lane: &[Complex<f64>]) -> (LaneStatus, Vec<Complex<Float>>) {
    let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(BITS)).unwrap();
    workspace.set_stability_threshold(Float::with_val(BITS, 2f64.powi(-(BITS as i32 - 40))));
    let values = lane.iter().map(|z| Complex::new(Float::with_val(BITS, z.re), Float::with_val(BITS, z.im))).collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

fn to_f64(values: &[Complex<Float>]) -> Vec<Complex<f64>> {
    values.iter().map(|z| Complex::new(z.re.to_f64(), z.im.to_f64())).collect()
}

/// `direct256_float`, rounded to binary64.
fn direct256(settings: ReductionSettings, lane: &[Complex<f64>]) -> (LaneStatus, Vec<Complex<f64>>) {
    let (status, values) = direct256_float(settings, lane);
    (status, to_f64(&values))
}

/// The exact identity of a triple D_a, D_b, D_c with equally spaced
/// collinear shifts, D_a + D_c - 2 D_b = K (a constant): the lane is
/// [I(without D_a) + I(without D_c) - 2 I(without D_b)] / K, with 256-bit
/// pinches summed at 256 bits (the sum cancels like 1/K).
fn triple_identity(shifts: &[[f64; 4]], masses: &[f64], mu: f64, rank: u16, (a, b, c): (usize, usize, usize)) -> Vec<Complex<f64>> {
    // K from the binary64 inputs at 256 bits (exact: in binary64 its terms
    // round by more than K = -18 delta).
    let constant = |i: usize| {
        let q = |x: f64| Float::with_val(BITS, x);
        let [t, x, y, z] = shifts[i].map(q);
        t.clone() * &t - x.clone() * &x - y.clone() * &y - z.clone() * &z - q(masses[i])
    };
    let k = constant(a) + constant(c) - constant(b) * Float::with_val(BITS, 2.0);
    let n = shifts.len();
    let mut total: Vec<Complex<Float>> = Vec::new();
    for (dropped, weight) in [(a, 1.0), (c, 1.0), (b, -2.0)] {
        let keep: Vec<usize> = (0..n).filter(|&i| i != dropped).collect();
        let pinch = coulomb_lane(
            &keep.iter().map(|&i| shifts[i]).collect::<Vec<_>>(),
            &keep.iter().map(|&i| masses[i]).collect::<Vec<_>>(),
            mu,
        );
        let (status, values) = direct256_float(ReductionSettings::new(n - 1, rank), &pinch);
        assert_eq!(status, LaneStatus::Success, "256-bit pinch without D{dropped}, rank {rank}");
        let weight = Float::with_val(BITS, weight) / k.clone();
        if total.is_empty() {
            total = vec![Complex::new(Float::new(BITS), Float::new(BITS)); values.len()];
        }
        for (t, z) in total.iter_mut().zip(&values) {
            *t = Complex::new(t.re.clone() + weight.clone() * &z.re, t.im.clone() + weight.clone() * &z.im);
        }
    }
    to_f64(&total)
}

/// Nearly linearly dependent propagators (PR #33 review). The Coulomb
/// triangle (shifts q1, q1 + k, q1 + 2k) and the review's pentagon with the
/// heavy mass M = 9 (1 + delta), so that D1 + D3 - 2 D2 = -18 delta, or with
/// the third heavy-quark point moved off the line by eps, have a small
/// nonzero relative velocity. Their integrals are Coulomb singular (they
/// grow like delta^(-1/2)), not the v = 0 value. Linear dependence is
/// accepted only to the working precision's rounding of the inputs, so
/// binary64 refuses these lanes, typed, at every rank with a Gram or Cayley
/// check. The scalar triangle has none: its master is evaluated at the lane's
/// own kinematics, bit-identical to main. Every lane that succeeds, directly
/// or through the facade (whose 256-bit tier accepts the moved lanes), agrees
/// with a 256-bit reference: the exact identity over 256-bit pinches, or the
/// 256-bit generic reduction of the moved lane.
/// A heavy mass one ulp above 9 is within binary64's rounding (|K| is below
/// 2^-53 of its inputs' size): a binary64 `DirectWorkspace` accepts it as
/// linearly dependent, and its array equals the exact lane's. (Through the
/// facade, the triangle's loop-rotation test then refuses it; the pentagon
/// passes that test.)
/// The rounding window is wider than one ulp (PR #38 review): the tests
/// bound the rounding by sums of magnitudes, so binary64 accepts the
/// dependence up to |delta| = 1.5e-14 here (about 80 ulp of M), and 1e-13
/// after a common translation of the shifts by (40, -30, 25, 10). Inside it,
/// directly, at a stability threshold of zero and through the facade, a lane
/// is refused, agrees with the 256-bit reference, or has the v = 0 value (the
/// exact lane's array). At threshold zero the near on-shell bubbles must
/// still take their closed forms, not their near-singular Cayley inverses.
#[test]
fn nearly_dependent_lanes_are_refused() {
    let k = [5.0, 0.0, 0.0, 4.0];
    let (q1, pq1, q4) = ([1.5, 0.5, 2.0, -1.0], [1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 3.0, -1.0]);
    let mu = 3.0;
    let still = [0.0; 4];
    let mut cases: Vec<(String, f64, f64, [f64; 4])> = Vec::new();
    for delta in [1.75e-10, 1e-11, 1e-13, -1e-13, 4e-14, 3e-14, 2e-14] {
        cases.push((format!("M = 9 (1 + {delta:e})"), delta, 9.0 * (1.0 + delta), still));
    }
    for (eps, axis) in [(1e-12, 1), (1e-14, 1), (1e-12, 2), (1e-14, 2)] {
        let mut offset = still;
        offset[axis] = eps;
        cases.push((format!("third point moved by {eps:e} along axis {axis}"), 0.0, 9.0, offset));
    }
    let lanes = |m: f64, offset: [f64; 4]| {
        [
            ("triangle", vec![q1, plus(q1, k, 1.0), plus(plus(q1, k, 2.0), offset, 1.0)], vec![m, 0.0, m], (0, 1, 2)),
            (
                "pentagon",
                vec![[0.0; 4], pq1, plus(pq1, k, 1.0), plus(plus(pq1, k, 2.0), offset, 1.0), q4],
                vec![0.0, m, 0.0, m, 2.0],
                (1, 2, 3),
            ),
        ]
    };
    for (label, delta, m, offset) in &cases {
        for (name, shifts, masses, triple) in lanes(*m, *offset) {
            let n = shifts.len();
            let lane = coulomb_lane(&shifts, &masses, mu);
            for rank in 0..=2u16 {
                let settings = ReductionSettings::new(n, rank);
                let reference = if *offset == still {
                    triple_identity(&shifts, &masses, mu, rank, triple)
                } else {
                    let (status, values) = direct256(settings.clone(), &lane);
                    assert_eq!(status, LaneStatus::Success, "{label}, {name}, rank {rank}: 256-bit reference");
                    values
                };
                let direct = direct_lanes(n, rank, &[lane.clone()]).remove(0);
                if n == 5 || rank >= 1 {
                    assert_ne!(direct.0, LaneStatus::Success, "{label}, {name}, rank {rank}: binary64 accepted the lane");
                }
                let facade = facade_lane(settings, &lane);
                for (what, (status, values)) in [("direct", direct), ("facade", facade)] {
                    if status == LaneStatus::Success {
                        // The scalar triangle is oneloopmaster's binary64 C0,
                        // bit-identical to main, conditioned like 1/delta
                        // near the threshold.
                        let tolerance = if n == 3 && rank == 0 && *delta != 0.0 { 1e-15 / delta.abs() } else { 1e-8 };
                        let ea = relative_difference(&values, &reference);
                        assert!(ea <= tolerance, "{label}, {name}, rank {rank}, {what}: E/A {ea:e} against 256 bits");
                    }
                }
            }
        }
    }
    let ulp = f64::from_bits(9.0f64.to_bits() + 1);
    for (exact, rounded) in lanes(9.0, still).into_iter().zip(lanes(ulp, still)) {
        let n = exact.1.len();
        for rank in 1..=2u16 {
            let exact_lane = direct_lanes(n, rank, &[coulomb_lane(&exact.1, &exact.2, mu)]).remove(0);
            let rounded_lane = direct_lanes(n, rank, &[coulomb_lane(&rounded.1, &rounded.2, mu)]).remove(0);
            assert_eq!(rounded_lane.0, LaneStatus::Success, "{}, M one ulp above 9, rank {rank}", exact.0);
            let ea = relative_difference(&rounded_lane.1, &exact_lane.1);
            assert!(ea <= 1e-12, "{}, M one ulp above 9, rank {rank}: E/A {ea:e} against the exact lane", exact.0);
        }
    }
    let translation = [40.0, -30.0, 25.0, 10.0];
    let translate = |shifts: Vec<[f64; 4]>, by: [f64; 4]| -> Vec<[f64; 4]> {
        shifts.into_iter().map(|q| plus(q, by, 1.0)).collect()
    };
    let mut window: Vec<(String, f64, [f64; 4])> = Vec::new();
    for delta in [2e-16, -2e-16, 5e-16, 1e-15, -1e-15, 3e-15, 5e-15, 1e-14, 1.5e-14, -1.5e-14] {
        window.push((format!("M = 9 (1 + {delta:e})"), delta, still));
    }
    for delta in [1e-14, 5e-14, 1e-13] {
        window.push((format!("M = 9 (1 + {delta:e}), translated"), delta, translation));
    }
    for (label, delta, by) in &window {
        for (near, exact) in lanes(9.0 * (1.0 + delta), still).into_iter().zip(lanes(9.0, still)) {
            let (name, shifts, masses, triple) = near;
            let shifts = translate(shifts, *by);
            let n = shifts.len();
            let lane = coulomb_lane(&shifts, &masses, mu);
            let exact_lane = coulomb_lane(&translate(exact.1, *by), &exact.2, mu);
            for rank in 0..=2u16 {
                let settings = ReductionSettings::new(n, rank);
                let reference = triple_identity(&shifts, &masses, mu, rank, triple);
                // The v = 0 value: the exact lane (the facade rescues the
                // scalar triangle, refused directly on exact inputs).
                let v0 = match direct_lanes_with(settings.clone(), &exact_lane) {
                    (LaneStatus::Success, values) => values,
                    _ => {
                        let (status, values) = facade_lane(settings.clone(), &exact_lane);
                        assert_eq!(status, LaneStatus::Success, "{name}, rank {rank}: the exact lane");
                        values
                    }
                };
                let paths = [
                    ("direct", direct_lanes_with(settings.clone(), &lane)),
                    ("direct at threshold 0", direct_lane_at_threshold(settings.clone(), &lane, 0.0)),
                    ("facade", facade_lane(settings, &lane)),
                ];
                for (what, (status, values)) in paths {
                    if status != LaneStatus::Success {
                        continue;
                    }
                    let tolerance = if n == 3 && rank == 0 { 1e-15 / delta.abs() } else { 1e-8 };
                    let (against_reference, against_v0) =
                        (relative_difference(&values, &reference), relative_difference(&values, &v0));
                    assert!(
                        against_reference <= tolerance || against_v0 <= 1e-10,
                        "{label}, {name}, rank {rank}, {what}: E/A {against_reference:e} against 256 bits, \
                         {against_v0:e} against the v = 0 value"
                    );
                }
            }
        }
    }
}

/// Statuses and array of one binary64 `DirectWorkspace` lane at `settings`,
/// with the stability threshold given (zero: an embedder's last resort).
fn direct_lane_at_threshold(
    settings: ReductionSettings,
    lane: &[Complex<f64>],
    threshold: f64,
) -> (LaneStatus, Vec<Complex<f64>>) {
    let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(0.0f64).unwrap();
    workspace.set_stability_threshold(threshold);
    let batch = KinematicsBatch { rows: 1, components: 1, values: lane.to_vec() };
    let result = workspace.reduce_batch(&batch).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

/// Central difference in m^2 with one Richardson step, of a lane's direct
/// binary64 array: the integral with that propagator doubled.
fn mass_derivative(shifts: &[[f64; 4]], masses: &[f64], mu: f64, rank: u16, line: usize) -> Vec<Complex<f64>> {
    let at = |m2: f64| {
        let mut moved = masses.to_vec();
        moved[line] = m2;
        let (status, values) = direct_lanes(shifts.len(), rank, &[coulomb_lane(shifts, &moved, mu)]).remove(0);
        assert_eq!(status, LaneStatus::Success, "undotted lane at m^2 = {m2}");
        values
    };
    let central = |h: f64| -> Vec<Complex<f64>> {
        let (up, down) = (at(masses[line] + h), at(masses[line] - h));
        up.iter().zip(&down).map(|(a, b)| Complex::new((a.re - b.re) / (2.0 * h), (a.im - b.im) / (2.0 * h))).collect()
    };
    let h = 1e-3 * masses[line];
    let (coarse, fine) = (central(h), central(h / 2.0));
    coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| Complex::new((4.0 * f.re - c.re) / 3.0, (4.0 * f.im - c.im) / 3.0))
        .collect()
}

/// Raised powers outside the Coulomb variant (PR #33 review): the closed
/// forms of the IR-singular IBP bases also serve main's dotted plans. Main
/// refuses every lane below ("singular Cayley matrix"), directly and through
/// the facade (the bubble of equal masses only directly).
/// - Bubbles, rank 0, against the closed forms: J(2,1) and J(1,2) of the
///   on-shell bubble p^2 = m^2 with a massless partner, and the lightlike
///   bubble of equal masses with one line doubled, int 1/(l^2 - m^2)^3 =
///   -1/(2 m^2).
/// - Ranks 0 to 2 against the mass derivative of the undotted integral (IR
///   finite and smooth in that mass): a triangle and a box with one on-shell
///   pair (a massless line and a massive one with p^2 = m^2) and a third
///   line doubled, whose IBP reaches the scaleless doubled massless tadpole;
///   and a box with a lightlike equal-mass pair, the doubled line inside or
///   outside the pair.
#[test]
fn raised_powers_use_the_singular_base_closed_forms() {
    let mu = 3.0;
    let pi2 = std::f64::consts::PI.powi(2);
    let euler_pi = 0.577_215_664_901_532_9 + std::f64::consts::PI.ln();
    // [uv, ir1, ir2, finite] in OneLOop's normalization, to the facade's: i
    // pi^2 times it, the finite part shifted by -(gamma_E + ln pi) ir1.
    let convention = |x: [f64; 4]| -> Vec<Complex<f64>> {
        vec![
            Complex::new(0.0, pi2 * x[0]),
            Complex::new(0.0, pi2 * x[1]),
            Complex::new(0.0, pi2 * x[2]),
            Complex::new(0.0, pi2 * (x[3] - euler_pi * x[1])),
        ]
    };
    let (p, m2): ([f64; 4], f64) = ([5.0, 0.0, 0.0, 4.0], 9.0);
    let log = (mu * mu / m2).ln();
    let inv = 1.0 / (2.0 * m2);
    let bubbles = [
        ("on-shell bubble, massless line doubled", vec![[0.0; 4], p], vec![0.0, m2], vec![2, 1], [0.0, -inv, 0.0, -(log - 2.0) * inv]),
        ("on-shell bubble, massive line doubled", vec![[0.0; 4], p], vec![0.0, m2], vec![1, 2], [0.0, inv, 0.0, log * inv]),
        ("lightlike equal-mass bubble", vec![[0.0; 4], [3.0, 0.0, 0.0, 3.0]], vec![4.0, 4.0], vec![2, 1], [0.0, 0.0, 0.0, -1.0 / 8.0]),
    ];
    for (label, shifts, masses, powers, closed_form) in bubbles {
        let mut settings = ReductionSettings::new(2, 0);
        settings.powers = powers;
        let lane = coulomb_lane(&shifts, &masses, mu);
        let expected = convention(closed_form);
        let direct = direct_lanes_with(settings.clone(), &lane);
        for (what, (status, values)) in [("direct", direct), ("facade", facade_lane(settings, &lane))] {
            assert_eq!(status, LaneStatus::Success, "{label}, {what}");
            let ea = relative_difference(&values, &expected);
            assert!(ea <= 1e-13, "{label}, {what}: E/A {ea:e} against the closed form");
        }
    }
    let (q2, q3) = ([3.0, 1.0, -1.0, 2.0], [-2.0, 0.5, 1.0, -1.0]);
    let lightlike = [3.0, 0.0, 0.0, 3.0];
    let dotted: [(&str, Vec<[f64; 4]>, Vec<f64>, usize); 4] = [
        ("triangle with an on-shell pair", vec![[0.0; 4], p, q2], vec![0.0, m2, 4.0], 2),
        ("box with an on-shell pair", vec![[0.0; 4], p, q2, q3], vec![0.0, m2, 4.0, 1.0], 2),
        ("lightlike equal-mass box, pair line doubled", vec![[0.0; 4], lightlike, q2, q3], vec![4.0, 4.0, 5.0, 2.0], 0),
        ("lightlike equal-mass box, other line doubled", vec![[0.0; 4], lightlike, q2, q3], vec![4.0, 4.0, 5.0, 2.0], 2),
    ];
    for (label, shifts, masses, line) in dotted {
        let n = shifts.len();
        let lane = coulomb_lane(&shifts, &masses, mu);
        for rank in 0..=2u16 {
            let mut settings = ReductionSettings::new(n, rank);
            settings.powers[line] = 2;
            let reference = mass_derivative(&shifts, &masses, mu, rank, line);
            let direct = direct_lanes_with(settings.clone(), &lane);
            for (what, (status, values)) in [("direct", direct), ("facade", facade_lane(settings, &lane))] {
                assert_eq!(status, LaneStatus::Success, "{label}, D{line} doubled, rank {rank}, {what}");
                let ea = relative_difference(&values, &reference);
                assert!(ea <= 1e-8, "{label}, D{line} doubled, rank {rank}, {what}: E/A {ea:e} against d/dm^2");
            }
        }
    }
}

/// Statuses and array of one binary64 `DirectWorkspace` lane at `settings`.
fn direct_lanes_with(settings: ReductionSettings, lane: &[Complex<f64>]) -> (LaneStatus, Vec<Complex<f64>>) {
    let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(0.0f64).unwrap();
    let batch = KinematicsBatch { rows: 1, components: 1, values: lane.to_vec() };
    let result = workspace.reduce_batch(&batch).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

/// MadLoop's QP rescue of g g > eta_c(1S0[8]) at v = 0 (onia_loop): its
/// 256-bit direct workspace evaluates exact binary128 inputs
/// (`set_input_precision(113)`), which hold p_Q = p_Qbar and p_Q^2 = m^2
/// only to binary128 rounding (here s_02 and 4 m^2 differ by 2^-112 of
/// their size). The Coulomb triangle (heavy, gluon, heavy; shifts 0, p, 2p)
/// and a box containing that triple must be decided linearly dependent at
/// the inputs' rounding, as binary64 decides its own inputs, and agree with
/// binary64 and with 512 bits. The triangle's origin also carries 5.5e-35
/// of MadLoop's momentum non-conservation in its z component, where the
/// other shifts are 1e-8 (zero to the inputs' rounding of the members'
/// largest component, 3.1). Moving the third point by 1e-25 along an empty
/// axis, which the inputs resolve, is still refused.
#[test]
fn binary128_inputs_keep_their_coulomb_relations_at_256_bits() {
    use iregi_rust::{Convention, InvariantInput, ScaleInput, TensorWeights};
    const P: &str = "1.550000000000000150417313013730877941058";
    const TWO_P: &str = "3.100000000000000300834626027461755882116";
    const M2: &str = "2.402500000000000301980662698042578995228";
    const S02: &str = "9.610000000000001207922650792170314440167";
    const MU2: &str = "9.610000000000000550670620214077651938730";
    const ORIGIN_Z: &str = "5.456311585857560400126433672016202759864e-35";
    const K: &str = "1.550000012818463695028389592261958698906";
    let settings = |n: usize, rank: u16| {
        let mut s = ReductionSettings::new(n, rank);
        s.convention = Convention::MadLoop;
        s.tensor_weights = TensorWeights::Components;
        s.invariants = InvariantInput::SuppliedInvariants;
        s.scale_input = ScaleInput::MuSquared;
        s
    };
    // Shifts, the upper triangle of the supplied invariants s_ij (i < j)
    // and the squared masses, as MadLoop's QP records write them.
    let triangle = |third_x: &'static str| {
        (
            vec![
                ["0", "0", "0", ORIGIN_Z],
                ["-1.550000000000000150417313013730877941058", "0", "0", "-1.281846354461107657853108063427026083915e-8"],
                ["-3.100000000000000300834626027461755882116", third_x, "0", "-2.563692708922215315706216126854052167830e-8"],
            ],
            vec![M2, S02, M2],
            vec![M2, "0", M2],
        )
    };
    let boxed = (
        vec![
            ["0", "0", "0", ORIGIN_Z],
            [K, "0", "0", K],
            [TWO_P, "0", "0", "2.563692708922215315706216132310363753687e-8"],
            [P, "0", "0", "1.281846354461107657853108068883337669773e-8"],
        ],
        vec!["0", S02, M2, "0", "-2.402500000000000301980662698042578995228", M2],
        vec![M2, M2, M2, "0"],
    );
    let values = |(shifts, invariants, masses): &(Vec<[&str; 4]>, Vec<&str>, Vec<&str>), bits: u32| {
        let parse = |text: &str| {
            let mut value = Float::parse(text, Some(113)).unwrap();
            value.set_prec(bits);
            Complex::new(value, Float::new(bits))
        };
        let mut lane: Vec<Complex<Float>> = shifts.iter().flat_map(|q| q.iter().map(|x| parse(x))).collect();
        lane.extend(invariants.iter().map(|x| parse(x)));
        lane.extend(masses.iter().map(|x| parse(x)));
        lane.push(parse(MU2));
        lane
    };
    let qp = |n: usize, rank: u16, lane: Vec<Complex<Float>>, bits: u32| {
        let mut workspace = DirectReductionPlan::new(settings(n, rank)).unwrap().workspace(Float::new(bits)).unwrap();
        workspace.set_input_precision(113);
        // The bridge's QP gate: 2^-(bits - 120).
        workspace.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 120))));
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lane }).unwrap();
        (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
    };
    let triangle_lane = triangle("0");
    for (name, n, rank, lane) in [("triangle", 3, 2u16, &triangle_lane), ("box", 4, 3, &boxed)] {
        let (status, at_256) = qp(n, rank, values(lane, 256), 256);
        assert_eq!(status, LaneStatus::Success, "{name}: binary128 inputs at 256 bits");
        let (status, at_512) = qp(n, rank, values(lane, 512), 512);
        assert_eq!(status, LaneStatus::Success, "{name}: binary128 inputs at 512 bits");
        // Componentwise, relative to the largest |value| of each slot (squared).
        let norm2 = |re: &Float, im: &Float| re.clone() * re + im.clone() * im;
        let mut worst = 0.0f64;
        for slot in 0..4 {
            let scale = (slot..at_512.len())
                .step_by(4)
                .map(|k| norm2(&at_512[k].re, &at_512[k].im))
                .fold(Float::new(512), |a, b| if b > a { b } else { a });
            for k in (slot..at_512.len()).step_by(4) {
                let difference = norm2(&(at_256[k].re.clone() - &at_512[k].re), &(at_256[k].im.clone() - &at_512[k].im));
                worst = worst.max((difference / &scale).to_f64());
            }
        }
        assert!(worst < 1e-140, "{name}: 256 bits {:e} off 512 bits", worst.sqrt());
        // Binary64: the same inputs rounded, through the facade.
        let rounded: Vec<Complex<f64>> = values(lane, 256).iter().map(|z| Complex::new(z.re.to_f64(), z.im.to_f64())).collect();
        let (status, binary64) = facade_lane(settings(n, rank), &rounded);
        assert_eq!(status, LaneStatus::Success, "{name}: binary64");
        let ea = relative_difference(&binary64, &to_f64(&at_256));
        assert!(ea <= 1e-14, "{name}: binary64 E/A {ea:e} against 256 bits");
    }
    // A third point moved by 1e-25 along x: a relative velocity the binary128
    // inputs resolve, Coulomb singular rather than the v = 0 value.
    let (status, _) = qp(3, 2, values(&triangle("1e-25"), 256), 256);
    assert_ne!(status, LaneStatus::Success, "triangle moved by 1e-25: binary128 accepted the lane");
}

/// `{g^lambda p^k}` at the tensor component with these Lorentz indices: the
/// sum over the partitions of the indices into `lambda` metric pairs
/// (diag(1, -1, -1, -1)) and single momenta, each partition once.
fn pv_structure(indices: &[usize], lambda: usize, p: &[f64; 4]) -> f64 {
    let Some((&first, tail)) = indices.split_first() else {
        return if lambda == 0 { 1.0 } else { 0.0 };
    };
    let mut sum = p[first] * pv_structure(tail, lambda, p);
    if lambda > 0 {
        for (i, &other) in tail.iter().enumerate() {
            if other == first {
                let mut rest = tail.to_vec();
                rest.remove(i);
                let metric = if first == 0 { 1.0 } else { -1.0 };
                sum += metric * pv_structure(&rest, lambda - 1, p);
            }
        }
    }
    sum
}

/// P-wave phase 1: two-point integrals at p^2 = 0 with one line doubled
/// (the lightlike bubbles of the dotted rank-(R+1) integrals), ranks 0 to 4,
/// against their PV coefficients from the d-dimensional Feynman-parameter
/// representation (benchmarks/onia/make_dotted_lightlike.py, mpmath; OneLOop
/// normalization). Massless doubled lines (soft poles at every rank for
/// line 1, below rank m for line 0), massive ones (with a massless, equal or
/// unequal partner, both branches of the inverse-power moments) and
/// scaleless pairs (cancelling poles at lambda = 1). Binary64 directly and
/// through the facade, and 256 bits. Main refuses every case of rank 1 or
/// more ("lightlike bubble unsupported for dotted propagators").
#[test]
fn dotted_lightlike_bubbles_match_their_d_dimensional_coefficients() {
    use iregi_rust::Convention;
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-dotted-lightlike.json")).unwrap();
    let rank = fixture["max_rank"].as_u64().unwrap() as u16;
    let basis = iregi_rust::basis::tensor_basis(rank);
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let shifts: Vec<[f64; 4]> = case["shifts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| std::array::from_fn(|a| q[a].as_f64().unwrap()))
            .collect();
        let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
        let powers: Vec<u16> = case["powers"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u16).collect();
        let mu = case["mu_squared"].as_f64().unwrap().sqrt();
        let p = shifts[1];
        let coefficient = |lambda: usize, k: usize, slot: usize| -> Float {
            let text = case["coefficients"][format!("{lambda},{k}")][slot].as_str().unwrap();
            Float::parse(text, Some(BITS)).unwrap()
        };
        // Expected components, slots [uv, ir1, ir2, finite], at 256 bits.
        let mut expected = Vec::new();
        for monomial in &basis {
            let indices: Vec<usize> = (0..4).flat_map(|a| std::iter::repeat(a).take(usize::from(monomial.0[a]))).collect();
            let r = indices.len();
            for slot in 0..4 {
                let mut value = Float::new(BITS);
                for lambda in 0..=r / 2 {
                    let weight = pv_structure(&indices, lambda, &p);
                    if weight != 0.0 {
                        value += &(coefficient(lambda, r - 2 * lambda, slot) * &Float::with_val(BITS, weight));
                    }
                }
                expected.push(Complex::new(value, Float::new(BITS)));
            }
        }
        let mut settings = ReductionSettings::new(2, rank);
        settings.convention = Convention::OneLoop;
        settings.powers = powers;
        let lane = coulomb_lane(&shifts, &masses, mu);
        let reference = to_f64(&expected);
        for (what, (status, values)) in
            [("direct", direct_lanes_with(settings.clone(), &lane)), ("facade", facade_lane(settings.clone(), &lane))]
        {
            assert_eq!(status, LaneStatus::Success, "{label}, {what}");
            let ea = relative_difference(&values, &reference);
            assert!(ea <= 1e-13, "{label}, {what}: E/A {ea:e} against the d-dimensional coefficients");
        }
        let (status, values) = direct256_float(settings, &lane);
        assert_eq!(status, LaneStatus::Success, "{label}, 256 bits");
        let (mut error, mut amplitude) = (0.0f64, 0.0f64);
        for (z, r) in values.iter().zip(&expected) {
            let (dr, di) = ((z.re.clone() - &r.re).to_f64(), (z.im.clone() - &r.im).to_f64());
            error = error.max(dr.hypot(di));
            amplitude = amplitude.max(r.re.to_f64().hypot(r.im.to_f64()));
        }
        assert!(error / amplitude <= 1e-30, "{label}, 256 bits: E/A {:e}", error / amplitude);
    }
}

/// P-wave phase 1: triangles and boxes whose lightlike bubbles contain the
/// doubled (massive) line, at ranks that reach those bubbles' tensor
/// coefficients, against the mass derivative of the undotted integral (IR
/// finite and smooth in that mass). Main refuses them all.
#[test]
fn dotted_lines_in_lightlike_pairs_match_the_mass_derivative() {
    let mu = 3.0;
    let (k, q2, q3) = ([3.0, 0.0, 0.0, 3.0], [3.0, 1.0, -1.0, 2.0], [-2.0, 0.5, 1.0, -1.0]);
    let cases: [(&str, Vec<[f64; 4]>, Vec<f64>, usize, std::ops::RangeInclusive<u16>); 6] = [
        ("triangle, equal-mass pair, line 0 doubled", vec![[0.0; 4], k, q2], vec![4.0, 4.0, 5.0], 0, 2..=4),
        ("triangle, equal-mass pair, line 1 doubled", vec![[0.0; 4], k, q2], vec![4.0, 4.0, 5.0], 1, 2..=4),
        ("triangle, unequal pair, line 1 doubled", vec![[0.0; 4], k, q2], vec![4.0, 9.0, 5.0], 1, 2..=3),
        ("triangle, near-equal pair, line 0 doubled", vec![[0.0; 4], k, q2], vec![4.0, 5.0, 2.0], 0, 2..=3),
        ("triangle, massless partner, line 0 doubled", vec![[0.0; 4], k, q2], vec![4.0, 0.0, 5.0], 0, 2..=3),
        ("box, equal-mass pair, line 0 doubled", vec![[0.0; 4], k, q2, q3], vec![4.0, 4.0, 5.0, 2.0], 0, 3..=4),
    ];
    for (label, shifts, masses, line, ranks) in cases {
        let n = shifts.len();
        let lane = coulomb_lane(&shifts, &masses, mu);
        for rank in ranks {
            let mut settings = ReductionSettings::new(n, rank);
            settings.powers[line] = 2;
            let reference = mass_derivative(&shifts, &masses, mu, rank, line);
            let direct = direct_lanes_with(settings.clone(), &lane);
            for (what, (status, values)) in [("direct", direct), ("facade", facade_lane(settings, &lane))] {
                assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, {what}");
                let ea = relative_difference(&values, &reference);
                assert!(ea <= 1e-8, "{label}, rank {rank}, {what}: E/A {ea:e} against d/dm^2");
            }
        }
    }
}

/// P-wave phase 1: IR-divergent triangles with a doubled line, which the
/// soft-line recursion reduces (a massless line k with every other line on
/// shell against it empties row k of the Cayley matrix; main refuses them
/// with "singular Cayley matrix"). The other rows of the IBP identity of
/// (l + r_k) at unit powers,
///   (D - 4) J(1,1,1) - sum_(i != k) J(1 + e_i - e_k) + sum_i R_ki J(1 + e_i) = 0,
/// tie the three singly dotted scalars to the undotted triangle and the
/// doubled pinches, and the recursion uses none of them. Binary64 and 256
/// bits: a soft triangle with one massive line (on-shell leg) and the
/// massless triangle with two lightlike legs.
#[test]
fn soft_lines_reduce_dotted_ir_divergent_triangles() {
    use iregi_rust::Convention;
    let mu = 3.0;
    let k = [3.0, 0.0, 0.0, 3.0];
    let cases: [(&str, Vec<[f64; 4]>, Vec<f64>, usize); 2] = [
        ("soft triangle, on-shell massive leg", vec![[0.0; 4], k, [5.0, 0.0, 0.0, 3.0]], vec![0.0, 0.0, 4.0], 1),
        ("massless triangle, two lightlike legs", vec![[0.0; 4], k, [6.0, 0.0, 0.0, 0.0]], vec![0.0, 0.0, 0.0], 1),
    ];
    let minkowski = |a: &[f64; 4], b: &[f64; 4]| {
        let d: [f64; 4] = std::array::from_fn(|i| a[i] - b[i]);
        d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]
    };
    let big = |x: f64| Complex::new(Float::with_val(BITS, x), Float::new(BITS));
    for (label, shifts, masses, soft) in cases {
        // Scalars of the lines given with the powers given, at 256 bits and
        // in binary64 (both carried at 256 bits for the rows).
        let scalar = |lines: &[usize], powers: Vec<u16>| -> [Vec<Complex<Float>>; 2] {
            let mut settings = ReductionSettings::new(lines.len(), 0);
            settings.convention = Convention::OneLoop;
            settings.powers = powers;
            let sub: Vec<[f64; 4]> = lines.iter().map(|&l| shifts[l]).collect();
            let m: Vec<f64> = lines.iter().map(|&l| masses[l]).collect();
            let lane = coulomb_lane(&sub, &m, mu);
            let (status, binary64) = direct_lanes_with(settings.clone(), &lane);
            assert_eq!(status, LaneStatus::Success, "{label}: lines {lines:?}, binary64");
            let (status, wide) = direct256_float(settings, &lane);
            assert_eq!(status, LaneStatus::Success, "{label}: lines {lines:?}, 256 bits");
            [binary64.iter().map(|z| Complex::new(Float::with_val(BITS, z.re), Float::with_val(BITS, z.im))).collect(), wide]
        };
        let all = [0, 1, 2];
        let base = scalar(&all, vec![1, 1, 1]);
        let dotted: Vec<_> = (0..3).map(|i| scalar(&all, (0..3).map(|l| if l == i { 2 } else { 1 }).collect())).collect();
        for (precision, tolerance) in [(0, 1e-12), (1, 1e-60)] {
            for (i, value) in dotted.iter().enumerate() {
                let ir = value[precision][1].re.to_f64().hypot(value[precision][1].im.to_f64())
                    + value[precision][2].re.to_f64().hypot(value[precision][2].im.to_f64());
                assert!(ir > 0.0, "{label}: D{i} doubled has no soft pole");
            }
            for row in (0..3).filter(|&r| r != soft) {
                // (D - 4) J(1,1,1) = -2 eps J(1,1,1): [uv, ir1, ir2, finite] times -2 eps.
                let j = &base[precision];
                let mut residual = [
                    big(0.0),
                    j[2].clone() * &big(-2.0),
                    big(0.0),
                    (j[0].clone() + &j[1]) * &big(-2.0),
                ];
                let mut size = 0.0f64;
                for i in (0..3).filter(|&i| i != row) {
                    let lines: Vec<usize> = (0..3).filter(|&l| l != row).collect();
                    let pinch = scalar(&lines, lines.iter().map(|&l| if l == i { 2 } else { 1 }).collect());
                    for slot in 0..4 {
                        residual[slot] = residual[slot].clone() - &pinch[precision][slot];
                        size = size.max(pinch[precision][slot].re.to_f64().hypot(pinch[precision][slot].im.to_f64()));
                    }
                }
                for i in 0..3 {
                    let r = minkowski(&shifts[row], &shifts[i]) - masses[row] - masses[i];
                    for slot in 0..4 {
                        let term = dotted[i][precision][slot].clone() * &big(r);
                        size = size.max(term.re.to_f64().hypot(term.im.to_f64()));
                        residual[slot] = residual[slot].clone() + &term;
                    }
                }
                let worst = residual.iter().map(|z| z.re.to_f64().hypot(z.im.to_f64())).fold(0.0, f64::max) / size;
                assert!(worst <= tolerance, "{label}: IBP row {row}, {}: residual {worst:e}", ["binary64", "256 bits"][precision]);
            }
        }
    }
}

/// Two-onium pentagons of onia_loop's e+ e- > J/psi eta_c trace at v = 0
/// (binary64, MadLoop's supplied invariants): each has two Coulomb triples
/// sharing a heavy line, whose partial fractions double a gluon on shell
/// against both heavy lines. That soft line empties its row of the
/// triangle's Cayley matrix; main refuses the lanes ("singular Cayley
/// matrix"), the soft-line recursion serves them. Reference: original
/// IREGI's quad IBP arrays, which it reports stable
/// (benchmarks/onia/make_two_onium_fixture.py).
#[test]
fn two_onium_pentagons_with_soft_coulomb_triangles_match_quad_iregi() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-two-onium-pentagons.json")).unwrap();
    supplied_invariant_lanes_match(&fixture, false);
}

/// The cases of a fixture of MadLoop lanes (supplied invariants and mu^2)
/// against quad IREGI arrays: binary64 directly and through the facade to
/// 1e-12, and 256 bits to 1e-15 if `wide`.
fn supplied_invariant_lanes_match(fixture: &Value, wide: bool) {
    use iregi_rust::{InvariantInput, ScaleInput};
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let parse = |v: &Value| v.as_str().unwrap().parse::<f64>().unwrap();
        let shifts: Vec<f64> =
            case["shifts"].as_array().unwrap().iter().flat_map(|q| q.as_array().unwrap().iter().map(parse)).collect();
        let invariants: Vec<f64> = case["invariants"].as_array().unwrap().iter().map(parse).collect();
        let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(parse).collect();
        let n = masses.len();
        let rank = case["rank"].as_u64().unwrap() as u16;
        let mut lane: Vec<Complex<f64>> = shifts.iter().map(|&x| Complex::new(x, 0.0)).collect();
        // The packed strict upper triangle of the (column-major) invariants.
        for i in 0..n {
            for j in i + 1..n {
                lane.push(Complex::new(invariants[i + n * j], 0.0));
            }
        }
        lane.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
        lane.push(Complex::new(parse(&case["mu_squared"]), 0.0));
        let reference: Vec<Complex<f64>> = case["reference"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|monomial| monomial.as_array().unwrap().iter().map(|z| Complex::new(parse(&z[0]), parse(&z[1]))))
            .collect();
        let mut settings = ReductionSettings::new(n, rank);
        settings.invariants = InvariantInput::SuppliedInvariants;
        settings.scale_input = ScaleInput::MuSquared;
        let mut runs =
            vec![("direct", direct_lanes_with(settings.clone(), &lane)), ("facade", facade_lane(settings.clone(), &lane))];
        if wide {
            runs.push(("256 bits", direct256(settings.clone(), &lane)));
        }
        for (what, (status, values)) in runs {
            assert_eq!(status, LaneStatus::Success, "{label}, {what}");
            let ea = relative_difference(&values, &reference);
            let tolerance = if what == "256 bits" { 1e-15 } else { 1e-12 };
            assert!(ea <= tolerance, "{label}, {what}: E/A {ea:e} against quad IREGI");
        }
    }
}

/// Collinear pairs (two massless lines whose invariant vanishes): a triangle
/// with one has a singular Cayley matrix that no soft line empties, and its
/// null vector's IBP identity reduces it (main refuses it, "singular Cayley
/// matrix"). A generic triangle and box sharing the pair, either pair member
/// doubled, and a pentagon with its massive line doubled, against original
/// IREGI's quad scalars with powers (benchmarks/onia/make_collinear_fixture.py).
/// The pentagon with a pair member doubled is refused: its Cayley reduction
/// drops a (D + 2)-dimensional term whose mass derivative keeps a collinear
/// pole (1e-4 of the integral here).
#[test]
fn collinear_pairs_reduce_dotted_triangles_and_refuse_unreduced_pentagons() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-collinear.json")).unwrap();
    let shifts: Vec<[f64; 4]> = fixture["shifts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| std::array::from_fn(|a| q[a].as_f64().unwrap()))
        .collect();
    let masses: Vec<f64> = fixture["masses_squared"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
    let mu = fixture["mu"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (n, dotted) = (case["n"].as_u64().unwrap() as usize, case["dotted"].as_u64().unwrap() as usize);
        let reference: Vec<Complex<f64>> = case["reference"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| Complex::new(z[0].as_str().unwrap().parse().unwrap(), z[1].as_str().unwrap().parse().unwrap()))
            .collect();
        let mut settings = ReductionSettings::new(n, 0);
        settings.powers[dotted] = 2;
        let lane = coulomb_lane(&shifts[..n], &masses[..n], mu);
        for (what, (status, values)) in
            [("direct", direct_lanes_with(settings.clone(), &lane)), ("facade", facade_lane(settings.clone(), &lane))]
        {
            assert_eq!(status, LaneStatus::Success, "n={n}, D{dotted} doubled, {what}");
            let ea = relative_difference(&values, &reference);
            assert!(ea <= 1e-13, "n={n}, D{dotted} doubled, {what}: E/A {ea:e} against quad IREGI");
        }
        let (status, values) = direct256(settings, &lane);
        assert_eq!(status, LaneStatus::Success, "n={n}, D{dotted} doubled, 256 bits");
        let ea = relative_difference(&values, &reference);
        assert!(ea <= 1e-15, "n={n}, D{dotted} doubled, 256 bits: E/A {ea:e} against quad IREGI");
    }
    for dotted in [0, 1] {
        let mut settings = ReductionSettings::new(5, 0);
        settings.powers[dotted] = 2;
        let (status, _) = direct_lanes_with(settings, &coulomb_lane(&shifts, &masses, mu));
        assert!(
            matches!(&status, LaneStatus::Unstable(e) if e.contains("collinear pair of pentagon")),
            "pentagon, D{dotted} doubled: {status:?}"
        );
    }
}

/// P-wave phase 2: a Coulomb box with its gluon doubled (MadLoop's P-wave
/// virtual at threshold; main refuses it, "linearly dependent propagators
/// with a raised power"). The partial fraction 2 D_g = D_Q + D_Qbar raises
/// the doubled gluon once more, a Taylor order 2 in its squared mass:
/// J(X, Q, g^2, Qbar) = [J(X, g^3, Qbar) + J(X, Q, g^3)] / 2, whose
/// triangles are no longer linearly dependent. Ranks 1 to 4 (P-wave's
/// R + 1), binary64 directly and through the facade, and 256 bits. (At rank 0
/// the box is a scalar master, with no Gram check to detect the linear
/// dependence (benchmarks/onia/README.md, Limits): the IBP of its raised
/// power inverts the box's Cayley matrix, singular at v = 0, and the lane is
/// refused. Its partial-fraction triangles are served at rank 0 too.)
#[test]
fn doubled_coulomb_gluons_take_a_second_taylor_order() {
    let (mu, m2) = (3.0, 4.0);
    let (q1, k) = ([3.0, 1.0, -1.0, 2.0], [2.0, 0.0, 0.0, 0.0]);
    let shifts = vec![[0.0; 4], q1, plus(q1, k, 1.0), plus(q1, k, 2.0)];
    let masses = vec![1.0, m2, 0.0, m2];
    let lane = coulomb_lane(&shifts, &masses, mu);
    let pinch = |drop: usize| {
        let lines: Vec<usize> = (0..4).filter(|&l| l != drop).collect();
        let sub: Vec<[f64; 4]> = lines.iter().map(|&l| shifts[l]).collect();
        let m: Vec<f64> = lines.iter().map(|&l| masses[l]).collect();
        (lines, coulomb_lane(&sub, &m, mu))
    };
    for rank in 1..=4u16 {
        let mut settings = ReductionSettings::new(4, rank);
        settings.powers[2] = 2;
        let mut reference: Vec<Vec<Complex<f64>>> = Vec::new();
        for drop in [1, 3] {
            let (lines, sub) = pinch(drop);
            let mut s = ReductionSettings::new(3, rank);
            s.powers[lines.iter().position(|&l| l == 2).unwrap()] = 3;
            let (status, values) = direct256(s, &sub);
            assert_eq!(status, LaneStatus::Success, "rank {rank}: pinch without line {drop}");
            reference.push(values);
        }
        let reference: Vec<Complex<f64>> =
            reference[0].iter().zip(&reference[1]).map(|(a, b)| Complex::new((a.re + b.re) / 2.0, (a.im + b.im) / 2.0)).collect();
        for (what, (status, values)) in [
            ("direct", direct_lanes_with(settings.clone(), &lane)),
            ("facade", facade_lane(settings.clone(), &lane)),
            ("256 bits", direct256(settings.clone(), &lane)),
        ] {
            assert_eq!(status, LaneStatus::Success, "rank {rank}, {what}");
            let ea = relative_difference(&values, &reference);
            let tolerance = if what == "256 bits" { 1e-14 } else { 1e-11 };
            assert!(ea <= tolerance, "rank {rank}, {what}: E/A {ea:e} against the partial fraction");
        }
    }
}

/// Dotted threshold bubbles (equal masses at p^2 = 4 m^2): their Cayley
/// matrix is singular, and the IBP base is the closed form of
/// Delta = m^2 (1 - 2x)^2, finite in dimensional regularization (main
/// refuses them). The bubble with raised powers and a triangle containing it
/// with each line doubled, against original IREGI's quad scalars with powers
/// (benchmarks/onia/make_threshold_fixture.py): binary64 directly and
/// through the facade, and 256 bits.
#[test]
fn dotted_threshold_bubbles_take_their_closed_form() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-threshold-bubbles.json")).unwrap();
    let shifts: Vec<[f64; 4]> = fixture["shifts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| std::array::from_fn(|a| q[a].as_f64().unwrap()))
        .collect();
    let masses: Vec<f64> = fixture["masses_squared"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
    let mu = fixture["mu"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let powers: Vec<u16> = case["powers"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u16).collect();
        let n = powers.len();
        let reference: Vec<Complex<f64>> = case["reference"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| Complex::new(z[0].as_str().unwrap().parse().unwrap(), z[1].as_str().unwrap().parse().unwrap()))
            .collect();
        let mut settings = ReductionSettings::new(n, 0);
        settings.powers = powers.clone();
        let lane = coulomb_lane(&shifts[..n], &masses[..n], mu);
        for (what, (status, values)) in [
            ("direct", direct_lanes_with(settings.clone(), &lane)),
            ("facade", facade_lane(settings.clone(), &lane)),
            ("256 bits", direct256(settings.clone(), &lane)),
        ] {
            assert_eq!(status, LaneStatus::Success, "powers {powers:?}, {what}");
            let ea = relative_difference(&values, &reference);
            assert!(ea <= 1e-13, "powers {powers:?}, {what}: E/A {ea:e} against quad IREGI");
        }
    }
}

/// Two-onium boxes of onia_loop's e+ e- > J/psi eta_c trace at v = 0
/// (binary64, MadLoop's supplied invariants), rank 3: a Coulomb triple and
/// two heavy pairs at p^2 = 4 m^2. The partial fraction doubles the gluon,
/// and the doubled triangles reduce to threshold bubbles with raised powers
/// (main refuses the lanes). Reference: original IREGI's quad IBP arrays
/// with the heavy pairs exactly on threshold
/// (benchmarks/onia/make_two_onium_box_fixture.py). From the binary64
/// shifts, off threshold by their rounding, IREGI moves by up to 9e-9: an
/// integral with a threshold pair depends on sqrt(1 - 4 m^2 / p^2).
#[test]
fn two_onium_boxes_with_threshold_pairs_match_quad_iregi() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-two-onium-boxes.json")).unwrap();
    supplied_invariant_lanes_match(&fixture, true);
}

/// MadLoop's supplied invariants in binary64 (BUILD_KINEMATIC_MATRIX of
/// CT_interface.inc, then S + m_i^2 + m_j^2 as TIRUST_interface.inc): the
/// squared shift differences, projected on shell or to zero within
/// OSThres = 1e-8, never to 4 m^2. Column-major, n x n.
fn madloop_invariants(shifts: &[[f64; 4]], masses: &[f64]) -> Vec<f64> {
    let n = shifts.len();
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut d = (shifts[i][0] - shifts[j][0]).powi(2);
            for k in 1..4 {
                d -= (shifts[i][k] - shifts[j][k]).powi(2);
            }
            let mut entry = d - masses[i] - masses[j];
            if masses[i] != 0.0 && ((d - masses[i]) / masses[i]).abs() < 1e-8 {
                entry = -masses[j];
            }
            if masses[j] != 0.0 && ((d - masses[j]) / masses[j]).abs() < 1e-8 {
                entry = -masses[i];
            }
            let reference = ((shifts[i][0].abs() + shifts[j][0].abs()) / 2.0).powi(2);
            if reference != 0.0 && (d / reference).abs() < 1e-8 {
                entry = -(masses[i] + masses[j]);
            }
            s[i + n * j] = entry + masses[i] + masses[j];
        }
    }
    s
}

/// A supplied-invariant lane: shifts, the packed strict upper triangle of
/// the column-major invariants, masses and mu^2.
fn supplied_lane(shifts: &[[f64; 4]], invariants: &[f64], masses: &[f64], mu2: f64) -> Vec<Complex<f64>> {
    let n = shifts.len();
    let mut lane: Vec<Complex<f64>> = shifts.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
    for i in 0..n {
        for j in i + 1..n {
            lane.push(Complex::new(invariants[i + n * j], 0.0));
        }
    }
    lane.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
    lane.push(Complex::new(mu2, 0.0));
    lane
}

/// The frame-independent contractions of a rank-`rank` array (ML5
/// components): the integrals of (q.u)^r, r = 0 to `rank`, each with its
/// four components.
fn contractions(values: &[Complex<f64>], rank: u16, u: [f64; 4]) -> Vec<Complex<f64>> {
    let factorial = |m: usize| (1..=m).product::<usize>() as f64;
    let eta = [1.0, -1.0, -1.0, -1.0];
    let mut out = vec![Complex::new(0.0, 0.0); 4 * (usize::from(rank) + 1)];
    for (k, monomial) in iregi_rust::basis::tensor_basis(rank).iter().enumerate() {
        let r = monomial.rank();
        let mut weight = factorial(r);
        for mu in 0..4 {
            let a = usize::from(monomial.0[mu]);
            weight *= (eta[mu] * u[mu]).powi(a as i32) / factorial(a);
        }
        for c in 0..4 {
            let z = &values[4 * k + c];
            out[4 * r + c] = Complex::new(out[4 * r + c].re + weight * z.re, out[4 * r + c].im + weight * z.im);
        }
    }
    out
}

fn lorentz_apply(lambda: &[[f64; 4]; 4], p: &[f64; 4]) -> [f64; 4] {
    std::array::from_fn(|i| (0..4).map(|k| lambda[i][k] * p[k]).sum())
}

fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 4]; 4] {
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    let [x, y, z] = axis.map(|v| v / norm);
    let (c, s) = (angle.cos(), angle.sin());
    let t = 1.0 - c;
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c + x * x * t, x * y * t - z * s, x * z * t + y * s],
        [0.0, y * x * t + z * s, c + y * y * t, y * z * t - x * s],
        [0.0, z * x * t - y * s, z * y * t + x * s, c + z * z * t],
    ]
}

fn boost(direction: [f64; 3], beta: f64) -> [[f64; 4]; 4] {
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    let n = direction.map(|v| v / norm);
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    let mut lambda = [[0.0; 4]; 4];
    lambda[0][0] = gamma;
    for i in 0..3 {
        lambda[0][i + 1] = -gamma * beta * n[i];
        lambda[i + 1][0] = -gamma * beta * n[i];
        for j in 0..3 {
            lambda[i + 1][j + 1] = f64::from(u8::from(i == j)) + (gamma - 1.0) * n[i] * n[j];
        }
    }
    lambda
}

/// A two-onium box (the first case of onia-two-onium-boxes.json) in rotated
/// and boosted frames, with the invariants MadLoop supplies there (its DP
/// stability check rotates the momenta). Rounded in binary64, they put the
/// threshold pairs up to about 1e-13 off 4 m^2 (the frame and the
/// arithmetic decide how far), inside the inputs' rounding: every frame is
/// served, and its frame-independent contractions stay within 2 beta of the
/// original frame's, beta = sqrt(|s - 4 m^2| / 4 m^2) (the integral's own
/// sensitivity to those inputs; measured up to 3.4e-8), plus 1e-10 for the
/// rounding of a boosted rank-3 array.
#[test]
fn two_onium_boxes_are_served_in_rotated_and_boosted_frames() {
    use iregi_rust::{InvariantInput, ScaleInput};
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-two-onium-boxes.json")).unwrap();
    let case = &fixture["cases"][0];
    let parse = |v: &Value| v.as_str().unwrap().parse::<f64>().unwrap();
    let shifts: Vec<[f64; 4]> =
        case["shifts"].as_array().unwrap().iter().map(|q| std::array::from_fn(|a| parse(&q[a]))).collect();
    let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(parse).collect();
    let mu2 = parse(&case["mu_squared"]);
    let rank = case["rank"].as_u64().unwrap() as u16;
    let threshold = 4.0 * masses[0];
    let mut settings = ReductionSettings::new(shifts.len(), rank);
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    let us = [[1.3, 0.4, -0.7, 0.2], [0.5, -1.1, 0.3, 0.9], [2.0, 0.1, 0.6, -0.8]];
    let lane = supplied_lane(&shifts, &madloop_invariants(&shifts, &masses), &masses, mu2);
    let (status, base) = direct_lanes_with(settings.clone(), &lane);
    assert_eq!(status, LaneStatus::Success, "original frame");
    let base: Vec<Complex<f64>> = us.iter().flat_map(|&u| contractions(&base, rank, u)).collect();
    let product = |a: [[f64; 4]; 4], b: [[f64; 4]; 4]| -> [[f64; 4]; 4] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
    };
    let mut offsets = Vec::new();
    for (name, lambda) in [
        ("rotation", rotation([-0.9, 0.2, 0.4], 2.1)),
        ("boost", boost([0.2, 0.7, -0.4], 0.6)),
        ("rotated boost", product(rotation([0.1, 0.9, 0.3], 1.3), boost([-0.6, 0.1, 0.5], 0.95))),
    ] {
        let moved: Vec<[f64; 4]> = shifts.iter().map(|p| lorentz_apply(&lambda, p)).collect();
        let invariants = madloop_invariants(&moved, &masses);
        let offset = invariants.iter().map(|s| (s - threshold).abs()).filter(|&d| d < 1e-6).fold(0.0, f64::max);
        offsets.push(offset);
        let lane = supplied_lane(&moved, &invariants, &masses, mu2);
        for (what, (status, values)) in
            [("direct", direct_lanes_with(settings.clone(), &lane)), ("facade", facade_lane(settings.clone(), &lane))]
        {
            assert_eq!(status, LaneStatus::Success, "{name}, {what}");
            let moved: Vec<Complex<f64>> =
                us.iter().flat_map(|u| contractions(&values, rank, lorentz_apply(&lambda, u))).collect();
            let ea = relative_difference(&moved, &base);
            let tolerance = 2.0 * (offset / threshold).sqrt() + 1e-10;
            eprintln!("{name}, {what}: threshold pairs off 4 m^2 by up to {offset:e}, E/A {ea:e} against the original frame");
            assert!(ea <= tolerance, "{name}, {what}: E/A {ea:e} against the original frame (offset {offset:e})");
        }
    }
    assert!(offsets.iter().any(|&o| o > 0.0), "no frame moved a threshold pair off 4 m^2: {offsets:?}");
}

/// The threshold gate against the exact inputs: the first two-onium box
/// with only the cross-onium pair moved, s_23 = 4 m^2 + delta (not in the
/// Coulomb triple, so the integral is regular in delta and moves like
/// beta = sqrt(|delta| / 4 m^2)). Inside the inputs' rounding the binary64
/// lane takes the threshold bubbles' closed form, and stays within beta of
/// the integral of its exact inputs (256 bits; measured 0.26 beta), at 256
/// bits too when the inputs are declared binary64. Beyond the rounding,
/// binary64 refuses (an ill-conditioned Cayley matrix) and 256 bits with
/// binary64 inputs agree with the exact inputs.
#[test]
fn threshold_pairs_within_the_rounding_stay_within_beta_of_the_exact_inputs() {
    use iregi_rust::{InvariantInput, ScaleInput};
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-two-onium-boxes.json")).unwrap();
    let case = &fixture["cases"][0];
    let parse = |v: &Value| v.as_str().unwrap().parse::<f64>().unwrap();
    let shifts: Vec<[f64; 4]> =
        case["shifts"].as_array().unwrap().iter().map(|q| std::array::from_fn(|a| parse(&q[a]))).collect();
    let supplied: Vec<f64> = case["invariants"].as_array().unwrap().iter().map(parse).collect();
    let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(parse).collect();
    let mu2 = parse(&case["mu_squared"]);
    let rank = case["rank"].as_u64().unwrap() as u16;
    let (n, threshold) = (shifts.len(), 4.0 * masses[0]);
    assert_eq!(supplied[2 + n * 3], threshold);
    let mut settings = ReductionSettings::new(n, rank);
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    // 256 bits with the inputs declared binary64 (MadLoop's QP rescue of a
    // DP lane): the same rounding, and so the same gate, as binary64.
    let rounded256 = |settings: ReductionSettings, lane: &[Complex<f64>]| {
        let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(BITS)).unwrap();
        workspace.set_stability_threshold(Float::with_val(BITS, 2f64.powi(-(BITS as i32 - 40))));
        workspace.set_input_precision(53);
        let values = lane.iter().map(|z| Complex::new(Float::with_val(BITS, z.re), Float::with_val(BITS, z.im))).collect();
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
        (result.statuses[0].clone(), to_f64(result.lane(0).unwrap_or_default()))
    };
    for (s23, inside) in [(threshold + 1.8e-15, true), (threshold - 1.1e-13, true), (threshold + 1e-12, false)] {
        let delta = s23 - threshold;
        let mut invariants = supplied.clone();
        invariants[2 + n * 3] = s23;
        invariants[3 + n * 2] = s23;
        let lane = supplied_lane(&shifts, &invariants, &masses, mu2);
        let (status, exact) = direct256(settings.clone(), &lane);
        assert_eq!(status, LaneStatus::Success, "s_23 - 4 m^2 = {delta:e}, exact inputs");
        let beta = (delta.abs() / threshold).sqrt();
        let (status, values) = direct_lanes_with(settings.clone(), &lane);
        if inside {
            assert_eq!(status, LaneStatus::Success, "s_23 - 4 m^2 = {delta:e}, binary64");
            let ea = relative_difference(&values, &exact);
            assert!(ea <= beta, "s_23 - 4 m^2 = {delta:e}, binary64: E/A {ea:e} against the exact inputs (beta {beta:e})");
        } else {
            assert!(
                matches!(&status, LaneStatus::Unstable(e) if e.contains("Cayley matrix for [2, 3]")),
                "s_23 - 4 m^2 = {delta:e}, binary64: {status:?}"
            );
        }
        let (status, values) = rounded256(settings.clone(), &lane);
        assert_eq!(status, LaneStatus::Success, "s_23 - 4 m^2 = {delta:e}, 256 bits with binary64 inputs");
        let ea = relative_difference(&values, &exact);
        let tolerance = if inside { beta } else { 1e-14 };
        assert!(ea <= tolerance, "s_23 - 4 m^2 = {delta:e}, 256 bits with binary64 inputs: E/A {ea:e} against the exact inputs");
    }
}

/// The direct engine at `bits` on one lane, its inputs declared
/// `input_bits` wide (53: MadLoop's QP rescue of a DP lane; `None`: exact).
fn direct_with_inputs(
    bits: u32,
    settings: ReductionSettings,
    lane: &[Complex<f64>],
    input_bits: Option<u32>,
) -> (LaneStatus, Vec<Complex<Float>>) {
    let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(bits)).unwrap();
    workspace.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
    if let Some(input_bits) = input_bits {
        workspace.set_input_precision(input_bits);
    }
    let values = lane.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

/// Whole-array E/A of two multiprecision arrays.
fn relative_difference_float(values: &[Complex<Float>], reference: &[Complex<Float>]) -> f64 {
    assert_eq!(values.len(), reference.len());
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (z, r) in values.iter().zip(reference) {
        let (dr, di) = ((z.re.clone() - &r.re).to_f64(), (z.im.clone() - &r.im).to_f64());
        error = error.max(dr.hypot(di));
        amplitude = amplitude.max(r.re.to_f64().hypot(r.im.to_f64()));
    }
    error / amplitude
}

/// A supplied-invariant lane's integral with its lightlike equal-mass pairs
/// (s_ij = 0, m_i^2 = m_j^2 != 0) moved off the light cone, to
/// s_ij = (1 + k/8) m_i^2 2^-e for the k-th pair, at e and e + 1, and
/// extrapolated linearly to zero (inputs `input_bits` wide). The general IBP
/// (the Cayley inverse) then evaluates the bubbles that the lightlike closed
/// form takes at s_ij = 0, and the integral is analytic in s_ij there (its
/// threshold is at 4 m^2). It divides by s_ij more than once, so it runs at
/// 512 bits.
fn off_light_cone(
    settings: &ReductionSettings,
    shifts: &[[f64; 4]],
    invariants: &[f64],
    masses: &[f64],
    mu2: f64,
    e: i32,
    input_bits: Option<u32>,
) -> Vec<Complex<Float>> {
    let n = shifts.len();
    let at = |e: i32| {
        let mut moved = invariants.to_vec();
        let mut k = 0;
        for i in 0..n {
            for j in i + 1..n {
                if masses[i] != 0.0 && masses[i] == masses[j] && invariants[i + n * j] == 0.0 {
                    let s = masses[i] * (1.0 + k as f64 / 8.0) * 2f64.powi(-e);
                    (moved[i + n * j], moved[j + n * i]) = (s, s);
                    k += 1;
                }
            }
        }
        assert!(k > 0, "no lightlike equal-mass pair");
        let lane = supplied_lane(shifts, &moved, masses, mu2);
        let (status, values) = direct_with_inputs(512, settings.clone(), &lane, input_bits);
        assert_eq!(status, LaneStatus::Success, "off the light cone by 2^-{e}");
        values
    };
    let (coarse, fine) = (at(e), at(e + 1));
    coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| Complex::new(f.re.clone() + &f.re - &c.re, f.im.clone() + &f.im - &c.im))
        .collect()
}

/// Lightlike equal-mass bubbles (p^2 = 0) with higher powers: their Cayley
/// matrix is singular, and the IBP base is the tadpole of the total power
/// nu, (-1)^nu (m^2)^(2 - nu) / ((nu - 1) (nu - 2)), finite for nu >= 3
/// (#51 takes nu = 3 only, and refuses every case below, "singular
/// Cayley matrix for [0, 1]"). The bubble with one line raised to the third
/// or fourth power, and a triangle containing it with each line so raised:
/// - rank 0 against original IREGI's quad scalars with powers
///   (benchmarks/onia/make_lightlike_powers_fixture.py), binary64 directly
///   and through the facade, and 256 bits, to 1e-13; quad IREGI's bubbles
///   are the tadpole;
/// - ranks 1 to 3 against the lanes with the pair off the light cone
///   (`off_light_cone`, exact inputs, from 2^-80): binary64 to 1e-12, 256
///   bits to 1e-40.
#[test]
fn lightlike_bubbles_with_higher_powers_take_the_tadpole() {
    use iregi_rust::{InvariantInput, ScaleInput};
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-lightlike-powers.json")).unwrap();
    let shifts: Vec<[f64; 4]> = fixture["shifts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| std::array::from_fn(|a| q[a].as_f64().unwrap()))
        .collect();
    let masses: Vec<f64> = fixture["masses_squared"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
    let mu = fixture["mu"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let powers: Vec<u16> = case["powers"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u16).collect();
        let n = powers.len();
        let reference: Vec<Complex<f64>> = case["reference"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| Complex::new(z[0].as_str().unwrap().parse().unwrap(), z[1].as_str().unwrap().parse().unwrap()))
            .collect();
        if n == 2 {
            // IREGI's native convention: i pi^2 times OneLOop's finite part.
            let nu = i32::from(powers[0] + powers[1]);
            let tadpole = (-1f64).powi(nu) * masses[0].powi(2 - nu) / f64::from((nu - 1) * (nu - 2));
            let expected = std::f64::consts::PI.powi(2) * tadpole;
            assert!((reference[3].im / expected - 1.0).abs() <= 1e-15, "powers {powers:?}: quad IREGI against the tadpole");
        }
        let mut settings = ReductionSettings::new(n, 0);
        settings.powers = powers.clone();
        let lane = coulomb_lane(&shifts[..n], &masses[..n], mu);
        for (what, (status, values)) in [
            ("direct", direct_lanes_with(settings.clone(), &lane)),
            ("facade", facade_lane(settings.clone(), &lane)),
            ("256 bits", direct256(settings.clone(), &lane)),
        ] {
            assert_eq!(status, LaneStatus::Success, "powers {powers:?}, {what}");
            let ea = relative_difference(&values, &reference);
            assert!(ea <= 1e-13, "powers {powers:?}, {what}: E/A {ea:e} against quad IREGI");
        }
        let invariants = madloop_invariants(&shifts[..n], &masses[..n]);
        let lane = supplied_lane(&shifts[..n], &invariants, &masses[..n], mu * mu);
        for rank in 1..=3u16 {
            let mut settings = ReductionSettings::new(n, rank);
            settings.powers = powers.clone();
            settings.invariants = InvariantInput::SuppliedInvariants;
            settings.scale_input = ScaleInput::MuSquared;
            let reference = off_light_cone(&settings, &shifts[..n], &invariants, &masses[..n], mu * mu, 80, None);
            let reference64 = to_f64(&reference);
            for (what, (status, values)) in
                [("direct", direct_lanes_with(settings.clone(), &lane)), ("facade", facade_lane(settings.clone(), &lane))]
            {
                assert_eq!(status, LaneStatus::Success, "powers {powers:?}, rank {rank}, {what}");
                let ea = relative_difference(&values, &reference64);
                assert!(ea <= 1e-12, "powers {powers:?}, rank {rank}, {what}: E/A {ea:e} off the light cone");
            }
            let (status, values) = direct_with_inputs(BITS, settings.clone(), &lane, None);
            assert_eq!(status, LaneStatus::Success, "powers {powers:?}, rank {rank}, 256 bits");
            let ea = relative_difference_float(&values, &reference);
            assert!(ea <= 1e-40, "powers {powers:?}, rank {rank}, 256 bits: E/A {ea:e} off the light cone");
        }
    }
}

/// The doubled Coulomb gluon in pentagons: onia_loop's g g > chi_c0(3P0[1]) g
/// at sqrt(s) = 20 GeV, m_c = 1.55 (binary64 momenta, MadLoop's invariants),
/// rank 5 (P-wave's R + 1), which #51 refuses ("singular Cayley matrix for
/// [3, 4]" and "[0, 4]"). Every heavy pair across an external gluon is
/// lightlike with equal masses, and the second Taylor order of the doubled
/// gluon raises a heavy line to the third power, so the IBP reaches those
/// bubbles at total power 4. Both layouts of the four refused calls: the
/// gluon third (MadLoop's loop groups 195 and 197, lines as MadLoop reported
/// them) and fourth (199 and 200, the incoming gluons first). Against
/// `off_light_cone` with binary64 inputs (from 2^-30, above the lightlike
/// gate's 2^-40): binary64 directly and through the facade to 1e-12, and 256
/// bits with binary64 inputs to 1e-17.
#[test]
fn doubled_coulomb_gluons_in_pentagons_reach_lightlike_bubbles() {
    use iregi_rust::{InvariantInput, ScaleInput};
    let p1 = [10.0, 0.0, 0.0, 10.0];
    let p2 = [10.0, 0.0, 0.0, -10.0];
    let onium = [1.024_024_999_999_999_79e1, -7.422_135_593_064_672_85, 4.207_721_234_534_722_58, -4.739_167_153_953_114_39];
    let gluon = [9.759_749_999_999_998_59, 7.422_135_593_064_672_85, -4.207_721_234_534_722_58, 4.739_167_153_953_114_39];
    let half: [f64; 4] = std::array::from_fn(|a| -onium[a] / 2.0);
    let (m2, mu2) = (1.55f64 * 1.55, 3.1f64 * 3.1);
    let layouts: [(&str, [[f64; 4]; 4], usize); 2] = [("gluon third", [p1, half, half, p2], 2), ("gluon fourth", [p1, p2, half, half], 3)];
    for (label, legs, gluon_line) in layouts {
        let mut shifts = vec![[0.0; 4]];
        for leg in legs {
            let last = *shifts.last().unwrap();
            shifts.push(std::array::from_fn(|a| last[a] + leg[a]));
        }
        let closing: [f64; 4] = std::array::from_fn(|a| shifts[0][a] - shifts[4][a]);
        assert!(closing.iter().zip(&gluon).all(|(x, g)| (x + g).abs() <= 1e-13), "{label}: momentum conservation");
        let masses: Vec<f64> = (0..5).map(|l| if l == gluon_line { 0.0 } else { m2 }).collect();
        let invariants = madloop_invariants(&shifts, &masses);
        let lane = supplied_lane(&shifts, &invariants, &masses, mu2);
        let mut settings = ReductionSettings::new(5, 5);
        settings.powers[gluon_line] = 2;
        settings.invariants = InvariantInput::SuppliedInvariants;
        settings.scale_input = ScaleInput::MuSquared;
        let reference = off_light_cone(&settings, &shifts, &invariants, &masses, mu2, 30, Some(53));
        let reference64 = to_f64(&reference);
        for (what, (status, values)) in
            [("direct", direct_lanes_with(settings.clone(), &lane)), ("facade", facade_lane(settings.clone(), &lane))]
        {
            assert_eq!(status, LaneStatus::Success, "{label}, {what}");
            let ea = relative_difference(&values, &reference64);
            assert!(ea <= 1e-12, "{label}, {what}: E/A {ea:e} off the light cone");
        }
        let (status, values) = direct_with_inputs(BITS, settings.clone(), &lane, Some(53));
        assert_eq!(status, LaneStatus::Success, "{label}, 256 bits with binary64 inputs");
        let ea = relative_difference_float(&values, &reference);
        assert!(ea <= 1e-17, "{label}, 256 bits with binary64 inputs: E/A {ea:e} off the light cone");
    }
}

/// Binary64 inputs at 512 bits, as a DP lane in the embedder's last tier
/// (`set_input_precision(53)`). The Coulomb triangle (heavy, gluon, heavy)
/// of onia_loop's g g > Upsilon g at sqrt(s) = 9.51 GeV holds its threshold
/// only to binary64 rounding: s_02 is 2 ulps below 4 m^2, and the origin is
/// 1.8e-15 off zero. Its Gram system is singular to that rounding, but its
/// conditioning (5e-68 at depth 2) passes the 512-bit threshold, which
/// evaluated the off-threshold integral (values of 1e9, 1/v enhanced) where
/// binary64 and 256 bits take the v = 0 partial fraction. It must agree with
/// both. The Coulomb vertex of e+ e- > t t~ a 46 ulps above threshold (the
/// MadLoop bridge's stress point), beyond that rounding, is still evaluated
/// exactly, 256 and 512 bits alike.
#[test]
fn binary64_inputs_keep_their_coulomb_relations_at_512_bits() {
    use iregi_rust::{Convention, InvariantInput, ScaleInput, TensorWeights};
    let settings = || {
        let mut s = ReductionSettings::new(3, 2);
        s.convention = Convention::MadLoop;
        s.tensor_weights = TensorWeights::Components;
        s.invariants = InvariantInput::SuppliedInvariants;
        s.scale_input = ScaleInput::MuSquared;
        s
    };
    // Shifts, the upper triangle of the supplied invariants (s_01, s_02,
    // s_12), the squared masses and mu^2, as binary64 bits.
    let lane = |p: &[u64; 12], s: &[u64; 3], m: &[u64; 3], mu2: u64| -> Vec<Complex<f64>> {
        let real = |bits: u64| Complex::new(f64::from_bits(bits), 0.0);
        p.iter().chain(s).chain(m).chain([&mu2]).map(|&b| real(b)).collect()
    };
    let upsilon = lane(
        &[
            0xbce0000000000000, 0x0000000000000000, 0x0000000000000000, 0x3cb0400000000000,
            0xc0130000b06aa477, 0x3f63fe1f0447ac1d, 0xbf575928c55a6639, 0x3f70e1b676d4a682,
            0xc0230000b06aa477, 0x3f73fe1f0447ac1d, 0xbf675928c55a6639, 0x3f80e1b676d4a682,
        ],
        &[0x4036900000000000, 0x40568ffffffffffd, 0x4036900000000000],
        &[0x4036900000000000, 0x0000000000000000, 0x4036900000000000],
        0x405e400000000000,
    );
    let ttxa = lane(
        &[
            0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x3d15692f40000000,
            0xc065a999999999e4, 0x3ef2b440f5517ccb, 0x3ef4242f861f91e8, 0x3edb20b1d08a5fdc,
            0xc075a999999999c9, 0x3eff4ff39a75e1a6, 0x3ef26d0393420659, 0xbea3ae92954b6860,
        ],
        &[0x40dd5438f5c28f5d, 0x40fd5438f5c28f8b, 0x40dd5438f5c28f5d],
        &[0x40dd5438f5c28f5d, 0x0000000000000000, 0x40dd5438f5c28f5d],
        0x40fd5439582bf9b6,
    );
    for (name, lane, binary64) in [("Upsilon Coulomb triangle", &upsilon, true), ("t t~ a Coulomb vertex", &ttxa, false)] {
        let (status, at_256) = direct_with_inputs(BITS, settings(), lane, Some(53));
        assert_eq!(status, LaneStatus::Success, "{name}: 256 bits");
        let (status, at_512) = direct_with_inputs(512, settings(), lane, Some(53));
        assert_eq!(status, LaneStatus::Success, "{name}: 512 bits");
        let ea = relative_difference_float(&at_256, &at_512);
        assert!(ea <= 1e-60, "{name}: 256 bits E/A {ea:e} against 512 bits");
        if binary64 {
            for (what, (status, values)) in
                [("direct", direct_lanes_with(settings(), lane)), ("facade", facade_lane(settings(), lane))]
            {
                assert_eq!(status, LaneStatus::Success, "{name}, {what}");
                let ea = relative_difference(&values, &to_f64(&at_512));
                assert!(ea <= 1e-14, "{name}, {what}: E/A {ea:e} against 512 bits");
            }
        }
    }
}

/// Two-onium pentagons of onia_loop's e+ e- > J/psi eta_c trace at v = 0
/// that hold both onia's Coulomb triples (MadLoop's loop groups 41 and 42;
/// binary64, MadLoop's supplied invariants), rank 3. The two triples share a
/// heavy line, so neither gluon can be raised for both: the partial
/// fractions raise the shared heavy line twice, and the pinch of the first
/// holds the second's partial fraction (Taylor order 2). The parent refuses
/// the lanes ("raising different propagators"). Reference: original IREGI's
/// quad IBP arrays with both onia exactly on threshold
/// (benchmarks/onia/make_double_coulomb_fixture.py):
/// - binary64 directly and through the facade to 1e-12, and 256 bits with
///   the inputs declared binary64 (MadLoop's QP rescue of a DP lane) to
///   1e-14. IREGI's invariants come from its shifts and agree with the
///   supplied ones to their rounding, which the pentagon amplifies to 5e-15.
///   As exact inputs the second triple is refused: its binary64 middle shift
///   is the rounded midpoint, so the relation does not hold at 256 bits'
///   rounding.
/// - 256 bits on IREGI's own inputs (binary128 shifts on threshold, their
///   squared differences) to 1e-15: 1.1e-16 on both, quad IREGI's floor (as
///   on the two-onium boxes), while 256 and 512 bits agree to 1e-75.
#[test]
fn double_coulomb_pentagons_match_quad_iregi() {
    use iregi_rust::{InvariantInput, ScaleInput};
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-double-coulomb.json")).unwrap();
    supplied_invariant_lanes_match(&fixture, false);
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let parse = |v: &Value| v.as_str().unwrap().parse::<f64>().unwrap();
        let shifts: Vec<[f64; 4]> =
            case["shifts"].as_array().unwrap().iter().map(|q| std::array::from_fn(|a| parse(&q[a]))).collect();
        let invariants: Vec<f64> = case["invariants"].as_array().unwrap().iter().map(parse).collect();
        let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(parse).collect();
        let reference: Vec<Complex<f64>> = case["reference"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|monomial| monomial.as_array().unwrap().iter().map(|z| Complex::new(parse(&z[0]), parse(&z[1]))))
            .collect();
        let mut settings = ReductionSettings::new(masses.len(), case["rank"].as_u64().unwrap() as u16);
        settings.invariants = InvariantInput::SuppliedInvariants;
        settings.scale_input = ScaleInput::MuSquared;
        let lane = supplied_lane(&shifts, &invariants, &masses, parse(&case["mu_squared"]));
        let (status, values) = direct_with_inputs(BITS, settings.clone(), &lane, Some(53));
        assert_eq!(status, LaneStatus::Success, "{label}, 256 bits with binary64 inputs");
        let ea = relative_difference(&to_f64(&values), &reference);
        assert!(ea <= 1e-14, "{label}, 256 bits with binary64 inputs: E/A {ea:e} against quad IREGI");
        // IREGI's own inputs (binary128 shifts exactly on threshold, and
        // their squared differences): only the arithmetic differs.
        let quad = |v: &Value| {
            let mut x = Float::parse(v.as_str().unwrap(), Some(113)).unwrap();
            x.set_prec(BITS);
            Complex::new(x, Float::new(BITS))
        };
        let n = masses.len();
        let threshold_invariants: Vec<&Value> = case["threshold_invariants"].as_array().unwrap().iter().collect();
        let mut exact: Vec<Complex<Float>> =
            case["threshold_shifts"].as_array().unwrap().iter().flat_map(|q| q.as_array().unwrap().iter().map(quad)).collect();
        for i in 0..n {
            for j in i + 1..n {
                exact.push(quad(threshold_invariants[i + n * j]));
            }
        }
        exact.extend(lane[4 * n + n * (n - 1) / 2..].iter().map(|z| Complex::new(Float::with_val(BITS, z.re), Float::new(BITS))));
        let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(BITS)).unwrap();
        workspace.set_stability_threshold(Float::with_val(BITS, 2f64.powi(-(BITS as i32 - 40))));
        workspace.set_input_precision(113);
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: exact }).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success, "{label}, IREGI's binary128 inputs at 256 bits");
        let quad_reference: Vec<Complex<Float>> = case["reference"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|monomial| {
                monomial.as_array().unwrap().iter().map(|z| {
                    let part = |v: &Value| {
                        let mut x = Float::parse(v.as_str().unwrap().trim(), Some(113)).unwrap();
                        x.set_prec(BITS);
                        x
                    };
                    Complex::new(part(&z[0]), part(&z[1]))
                })
            })
            .collect();
        let ea = relative_difference_float(result.lane(0).unwrap(), &quad_reference);
        assert!(ea <= 1e-15, "{label}, IREGI's binary128 inputs at 256 bits: E/A {ea:e} against quad IREGI");
    }
}

/// P-wave's doubled heavy line in a Coulomb triple: the partial fraction
/// raises the dotted line itself (the parent raised the gluon and refused
/// the lane, "linearly dependent propagators with a raised power"). The box
/// (X, Q, g, Qbar) of `doubled_coulomb_gluons_take_a_second_taylor_order`
/// with Q doubled is, from D_Q + D_Qbar = 2 D_g,
/// J(X, Q^2, g, Qbar) = 2 J(X, Q^3, Qbar) - J(X, Q^3, g), two triangles
/// that are not linearly dependent, and with Qbar doubled the same with Q and
/// Qbar exchanged. Ranks 1 to 4, binary64 directly and through the facade,
/// and 256 bits, against those triangles at 256 bits. (At rank 0 the box is
/// a scalar master; README, Limits.)
#[test]
fn doubled_heavy_lines_in_coulomb_triples_raise_themselves() {
    let (mu, m2) = (3.0, 4.0);
    let (q1, k) = ([3.0, 1.0, -1.0, 2.0], [2.0, 0.0, 0.0, 0.0]);
    let shifts = vec![[0.0; 4], q1, plus(q1, k, 1.0), plus(q1, k, 2.0)];
    let masses = vec![1.0, m2, 0.0, m2];
    let lane = coulomb_lane(&shifts, &masses, mu);
    let pinch = |lines: [usize; 3], rank: u16, cubed: usize| {
        let sub: Vec<[f64; 4]> = lines.iter().map(|&l| shifts[l]).collect();
        let m: Vec<f64> = lines.iter().map(|&l| masses[l]).collect();
        let mut s = ReductionSettings::new(3, rank);
        s.powers[lines.iter().position(|&l| l == cubed).unwrap()] = 3;
        let (status, values) = direct256(s, &coulomb_lane(&sub, &m, mu));
        assert_eq!(status, LaneStatus::Success, "rank {rank}: pinch {lines:?} with line {cubed} cubed");
        values
    };
    // (doubled line, its partner): 2 J(X, dot^3, partner) - J(X, dot^3, g).
    for (dotted, partner) in [(1usize, 3usize), (3, 1)] {
        for rank in 1..=4u16 {
            let sorted = |mut lines: [usize; 3]| {
                lines.sort_unstable();
                lines
            };
            let a = pinch(sorted([0, dotted, partner]), rank, dotted);
            let b = pinch(sorted([0, dotted, 2]), rank, dotted);
            let reference: Vec<Complex<f64>> =
                a.iter().zip(&b).map(|(x, y)| Complex::new(2.0 * x.re - y.re, 2.0 * x.im - y.im)).collect();
            let mut settings = ReductionSettings::new(4, rank);
            settings.powers[dotted] = 2;
            for (what, (status, values)) in [
                ("direct", direct_lanes_with(settings.clone(), &lane)),
                ("facade", facade_lane(settings.clone(), &lane)),
                ("256 bits", direct256(settings.clone(), &lane)),
            ] {
                assert_eq!(status, LaneStatus::Success, "line {dotted} doubled, rank {rank}, {what}");
                let ea = relative_difference(&values, &reference);
                let tolerance = if what == "256 bits" { 1e-14 } else { 1e-11 };
                assert!(ea <= tolerance, "line {dotted} doubled, rank {rank}, {what}: E/A {ea:e} against the partial fraction");
            }
        }
    }
}

/// `EmbedderOptions::relation_tiers`: a P-wave pentagon of u u~ > chi_c0 g at
/// sqrt(s) = 1 TeV (onia_loop, point 7 of its stability profile, MadLoop's
/// call 76), rank 4, the heavy line doubled, MadLoop's binary64 inputs. It
/// reduces through the constant identity D_0 + D_3 - 2 D_4 = 4 m^2 = 9.6,
/// against invariants of 1e6: the identity cancels its pinches by about s/K.
/// The bridge's binary64 embedder accepts it at its first pass, 2e-10 off the
/// integral of its inputs; MadLoop's P-wave sum amplifies that by about 1e8.
/// With `relation_tiers` the lane goes to the double-double tier (inputs
/// declared binary64, so the relation still holds there): within 1e-15 of
/// 512 bits. A lane reduced without a relation keeps its binary64 route.
#[test]
fn relation_tiers_take_affine_relation_lanes_to_double_double() {
    use iregi_rust::{EmbedderOptions, LaneRoute};
    let pentagon = relation_pentagon();
    let mut powered = bridge_settings(5, 4);
    powered.powers[4] = 2;
    let (status, exact) = direct_with_inputs(512, powered.clone(), &pentagon, Some(53));
    assert_eq!(status, LaneStatus::Success, "512 bits with binary64 inputs");
    let exact = to_f64(&exact);
    let embedded = |settings: ReductionSettings, lane: &[Complex<f64>], relation_tiers: bool| {
        let plan = TensorReductionPlan::new(settings).unwrap();
        let options = EmbedderOptions { relation_tiers, ..bridge_options() };
        let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lane.to_vec() }).unwrap();
        let route = workspace.last_lane_routes().unwrap()[0];
        (result.statuses[0].clone(), route, result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
    };
    let (status, route, values) = embedded(powered.clone(), &pentagon, false);
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Direct), "without relation_tiers");
    let ea = relative_difference(&values, &exact);
    assert!(ea >= 1e-11, "without relation_tiers: E/A {ea:e}, expected the binary64 cancellation");
    let (status, route, values) = embedded(powered, &pentagon, true);
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Tier(106)), "with relation_tiers");
    let ea = relative_difference(&values, &exact);
    assert!(ea <= 1e-15, "with relation_tiers: E/A {ea:e} against 512 bits");
    // No relation: a generic triangle keeps its first pass.
    let triangle = coulomb_lane(&[[0.0; 4], [3.0, 1.0, -1.0, 2.0], [-2.0, 0.5, 1.0, -1.0]], &[1.0, 4.0, 0.0], 3.0);
    let plan = TensorReductionPlan::new(ReductionSettings::new(3, 2)).unwrap();
    let options = EmbedderOptions { method_choice: false, precision_tiers: true, relation_tiers: true, ..EmbedderOptions::default() };
    let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: triangle }).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    assert_eq!(workspace.last_lane_routes().unwrap()[0], LaneRoute::Direct, "a lane without a relation");
}

/// The settings of MadLoop's bridge: supplied invariants, tensor components,
/// MadLoop's convention, mu^2 as the scale input.
fn bridge_settings(n: usize, rank: u16) -> ReductionSettings {
    use iregi_rust::{Convention, InvariantInput, ScaleInput, TensorWeights};
    let mut s = ReductionSettings::new(n, rank);
    s.convention = Convention::MadLoop;
    s.tensor_weights = TensorWeights::Components;
    s.invariants = InvariantInput::SuppliedInvariants;
    s.scale_input = ScaleInput::MuSquared;
    s
}

/// The bridge's embedder options, without its gate.
fn bridge_options() -> iregi_rust::EmbedderOptions {
    iregi_rust::EmbedderOptions {
        method_choice: false,
        precision_tiers: true,
        tier_margin_bits: 0,
        double_double_headroom_bits: 26,
        ..iregi_rust::EmbedderOptions::default()
    }
}

/// The pentagon of `relation_tiers_take_affine_relation_lanes_to_double_double`
/// as MadLoop supplied it (heavy line 4, rank 4): shifts, the upper triangle
/// of the supplied invariants, masses^2, mu^2.
fn relation_pentagon() -> Vec<Complex<f64>> {
    let real = |bits: u64| Complex::new(f64::from_bits(bits), 0.0);
    [
        0x0000000000000000u64, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0x407f400000000001, 0x0000000000000000, 0x0000000000000000, 0x407f400000000001,
        0x408f400000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0x407f4013ae685db8, 0xc0608e55f47bd8ca, 0x4054a9a3ee72991b, 0xc07daff1547d55d3,
        0x406f4013ae685db8, 0xc0508e55f47bd8ca, 0x4044a9a3ee72991b, 0xc06daff1547d55d3,
        0x0000000000000000, 0x412e848000000002, 0x40233851eb850000, 0x40033851eb851eb9, 0x0000000000000000,
        0xc12dc11f3a183d27, 0xc11dc128d64132e9, 0x0000000000000000, 0x411e847663d70a3f, 0x40033851eb851eb9,
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x40033851eb851eb9,
        0x412e848000000000,
    ]
    .iter()
    .map(|&b| real(b))
    .collect()
}

/// `EmbedderOptions::low_words` on the pentagon of
/// `relation_tiers_take_affine_relation_lanes_to_double_double`: the
/// double-double tier serves it, and value plus low word is that tier's value:
/// 3.2e-26 from 512 bits on the same binary64 inputs (the rounded value
/// alone: 5.3e-17). Values, statuses and routes are bit-identical with the
/// option off. A lane the binary64 pass serves has zero low words, and
/// without the option there are none.
#[test]
fn low_words_carry_the_double_double_tier_value() {
    use iregi_rust::{EmbedderOptions, LaneRoute};
    let pentagon = relation_pentagon();
    let mut powered = bridge_settings(5, 4);
    powered.powers[4] = 2;
    let (status, exact) = direct_with_inputs(512, powered.clone(), &pentagon, Some(53));
    assert_eq!(status, LaneStatus::Success, "512 bits with binary64 inputs");
    let run = |settings: ReductionSettings, lane: &[Complex<f64>], relation_tiers: bool, low_words: bool| {
        let plan = TensorReductionPlan::new(settings).unwrap();
        let options = EmbedderOptions { relation_tiers, low_words, ..bridge_options() };
        let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lane.to_vec() }).unwrap();
        let route = workspace.last_lane_routes().unwrap()[0];
        let low = workspace.last_low_words().map(<[_]>::to_vec);
        (result.statuses[0].clone(), route, result.lane(0).map(<[_]>::to_vec).unwrap_or_default(), low)
    };
    let (status, route, values, low) = run(powered.clone(), &pentagon, true, true);
    assert_eq!((status.clone(), route), (LaneStatus::Success, LaneRoute::Tier(106)));
    let low = low.expect("low words with the option on");
    assert_eq!(low.len(), values.len());
    let (_, plain_route, plain, none) = run(powered.clone(), &pentagon, true, false);
    assert_eq!((plain_route, bits(&plain)), (route, bits(&values)), "values and route with the option off");
    assert!(none.is_none(), "no low words with the option off");
    let at = |z: &Complex<f64>| Complex::new(Float::with_val(512, z.re), Float::with_val(512, z.im));
    let carried: Vec<Complex<Float>> = values
        .iter()
        .zip(&low)
        .map(|(hi, lo)| {
            let (hi, lo) = (at(hi), at(lo));
            Complex::new(hi.re + &lo.re, hi.im + &lo.im)
        })
        .collect();
    let rounded: Vec<Complex<Float>> = values.iter().map(at).collect();
    let (with_low, without) = (relative_difference_float(&carried, &exact), relative_difference_float(&rounded, &exact));
    assert!(with_low <= 1e-24, "value plus low word: E/A {with_low:e} against 512 bits");
    assert!(without >= 1e-18, "the rounded value alone: E/A {without:e}");
    assert!(low.iter().any(|z| z.re != 0.0 || z.im != 0.0), "nonzero low words");
    // Without relation_tiers the binary64 pass serves it: zero low words.
    let (status, route, _, low) = run(powered, &pentagon, false, true);
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Direct));
    assert!(low.unwrap().iter().all(|z| z.re == 0.0 && z.im == 0.0), "binary64 lanes have zero low words");
}

/// A Coulomb triangle (Q, g, Qbar), masses^2 (1, 0, 1), rank 2, through the
/// bridge's embedder options with `relation_tiers` and `low_words`, as hi + lo
/// inputs: shifts a, (a + b) / 2, b with a - b = (2 + e, 0, 0, 0) and the
/// whole lane translated by (t, 0, 0, 0), and the invariants they give
/// (s_QQbar = (2 + e)^2, s_Qg = s_gQbar = (1 + e / 2)^2). Hi words, lo words.
fn coulomb_triangle_low_inputs(e: f64, t: f64) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let real = |x: f64| Complex::new(x, 0.0);
    let zero = real(0.0);
    let (mut hi, mut lo) = (Vec::new(), Vec::new());
    // e and t are below half an ulp of the hi words (powers of two).
    for (first, low) in [(3.0, t), (2.0, t - e / 2.0), (1.0, t - e)] {
        hi.extend([real(first), real(1.0), real(-1.0), real(2.0)]);
        lo.extend([real(low), zero, zero, zero]);
    }
    // s_Qg, s_QQbar, s_gQbar, to 106 bits: (1 + e/2)^2 = 1 + e + e^2/4.
    for (value, low) in [(1.0, e), (4.0, 4.0 * e), (1.0, e)] {
        hi.push(real(value));
        lo.push(real(low));
    }
    hi.extend([real(1.0), zero, real(1.0), real(3.0)]);
    lo.extend([zero, zero, zero, zero]);
    (hi, lo)
}

/// `reduce_batch_with_low_inputs` decides at the declared precision of its
/// inputs, the same exact-threshold and near-threshold points in both modes:
/// - zero low words reduce bit for bit as `reduce_batch`;
/// - exactly at threshold but translated by t = 2^-60 (which binary64 cannot
///   hold), the tiers serve the v = 0 value of hi + lo: 2.1e-32 from 512
///   bits on the same inputs at 106 bits (bound 1e-25). The binary64 inputs
///   lose the translation, 7.5e-19;
/// - off threshold by e = 2^-80 (relative), the binary64 inputs are at
///   threshold (the v = 0 value). Declared at 106 bits, the offset is beyond
///   the inputs' rounding, so the v = 0 rule does not apply: refused, as at
///   512 bits on the same inputs (the gluon's bubbles are in the undecided
///   window). Declared at 64 bits, it is within the rounding again: the v = 0
///   value, 9.3e-33 from 512 bits and 6.4e-24 from the binary64 inputs' value.
/// - with `relation_tiers` off, the binary64 pass serves the hi words at
///   threshold (route Direct). With low words it keeps no first-pass value:
///   the tiers decide, so the 2^-80 lane is refused at 106 bits and the v = 0
///   value at 64 bits, and the translated lane is served from the tier.
#[test]
fn low_inputs_decide_thresholds_at_their_declared_precision() {
    use iregi_rust::{EmbedderOptions, LaneRoute};
    let settings = bridge_settings(3, 2);
    let plan = TensorReductionPlan::new(settings.clone()).unwrap();
    let options = EmbedderOptions { relation_tiers: true, low_words: true, ..bridge_options() };
    let batch = |values: Vec<Complex<f64>>| KinematicsBatch { rows: 1, components: 1, values };
    let run_with = |options: EmbedderOptions, hi: &[Complex<f64>], lo: Option<(&[Complex<f64>], u32)>| {
        let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
        let result = match lo {
            None => workspace.reduce_batch(&batch(hi.to_vec())),
            Some((lo, bits)) => workspace.reduce_batch_with_low_inputs(&batch(hi.to_vec()), &batch(lo.to_vec()), bits),
        }
        .unwrap();
        let route = workspace.last_lane_routes().unwrap()[0];
        let low = workspace.last_low_words().unwrap().to_vec();
        (result.statuses[0].clone(), route, result.lane(0).map(<[_]>::to_vec).unwrap_or_default(), low)
    };
    let run = |hi: &[Complex<f64>], lo: Option<(&[Complex<f64>], u32)>| run_with(options, hi, lo);
    // 512 bits on hi + lo, declared at `bits`.
    let exact = |hi: &[Complex<f64>], lo: &[Complex<f64>], bits: u32| {
        let mut workspace = DirectReductionPlan::new(settings.clone()).unwrap().workspace(Float::new(512)).unwrap();
        workspace.set_stability_threshold(Float::with_val(512, 2f64.powi(-(512 - 40))));
        workspace.set_input_precision(bits);
        let at = |x: f64| Float::with_val(512, x);
        let values = hi.iter().zip(lo).map(|(h, l)| Complex::new(at(h.re) + &at(l.re), at(h.im) + &at(l.im))).collect();
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
        (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
    };
    let carried = |values: &[Complex<f64>], low: &[Complex<f64>]| -> Vec<Complex<Float>> {
        let at = |x: f64| Float::with_val(512, x);
        values.iter().zip(low).map(|(h, l)| Complex::new(at(h.re) + &at(l.re), at(h.im) + &at(l.im))).collect()
    };
    // Zero low words: as reduce_batch, bit for bit.
    let (hi, lo) = coulomb_triangle_low_inputs(0.0, 0.0);
    assert!(lo.iter().all(|z| z.re == 0.0));
    let plain = run(&hi, None);
    assert_eq!(plain.0, LaneStatus::Success);
    assert_eq!(plain.1, LaneRoute::Tier(106), "a relation lane");
    let zero = run(&hi, Some((&lo, 106)));
    assert_eq!((zero.0.clone(), zero.1), (plain.0.clone(), plain.1));
    assert_eq!((bits(&zero.2), bits(&zero.3)), (bits(&plain.2), bits(&plain.3)), "zero low words");
    // Exactly at threshold, translated by 2^-60.
    let t = 2f64.powi(-60);
    let (hi_t, lo_t) = coulomb_triangle_low_inputs(0.0, t);
    let (status, route, values, low) = run(&hi_t, Some((&lo_t, 106)));
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Tier(106)), "translated, at threshold");
    let (reference_status, reference) = exact(&hi_t, &lo_t, 106);
    assert_eq!(reference_status, LaneStatus::Success);
    let ea = relative_difference_float(&carried(&values, &low), &reference);
    assert!(ea <= 1e-25, "translated, at threshold: E/A {ea:e} against 512 bits");
    let binary64 = relative_difference_float(&carried(&plain.2, &plain.3), &reference);
    assert!(binary64 >= 1e-20, "the translation, lost by binary64 inputs: E/A {binary64:e}");
    // Off threshold by 2^-80.
    let e = 2f64.powi(-80);
    let (hi_e, lo_e) = coulomb_triangle_low_inputs(e, 0.0);
    assert_eq!(bits(&hi_e), bits(&hi), "the offset is below binary64's rounding");
    let (status, ..) = run(&hi_e, Some((&lo_e, 106)));
    assert!(matches!(status, LaneStatus::Unstable(_)), "at 106 bits, 2^-80 off threshold: {status:?}");
    let (reference_status, _) = exact(&hi_e, &lo_e, 106);
    assert!(matches!(reference_status, LaneStatus::Unstable(_)), "512 bits at 106-bit inputs: {reference_status:?}");
    let (status, route, values, low) = run(&hi_e, Some((&lo_e, 64)));
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Tier(106)), "at 64 bits, within the rounding");
    let (reference_status, reference) = exact(&hi_e, &lo_e, 64);
    assert_eq!(reference_status, LaneStatus::Success);
    let ea = relative_difference_float(&carried(&values, &low), &reference);
    assert!(ea <= 1e-25, "at 64 bits: E/A {ea:e} against 512 bits");
    let v0 = relative_difference_float(&carried(&values, &low), &carried(&plain.2, &plain.3));
    assert!(v0 <= 1e-20, "at 64 bits: the v = 0 value, {v0:e} from the binary64 inputs'");
    // Without relation_tiers the binary64 pass serves the lane at threshold
    // (route Direct). With low words, nothing the binary64 pass decided at
    // its rounding is kept: the tiers decide, with the same answers.
    let first_pass = EmbedderOptions { relation_tiers: false, ..options };
    let (status, route, ..) = run_with(first_pass, &hi_e, None);
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Direct), "binary64 inputs: the first pass");
    let (status, route, ..) = run_with(first_pass, &hi_e, Some((&lo_e, 106)));
    assert!(matches!(status, LaneStatus::Unstable(_)), "first pass on, 106 bits, 2^-80 off: {status:?}");
    assert_eq!(route, LaneRoute::Refused);
    let (status, route, values, low) = run_with(first_pass, &hi_e, Some((&lo_e, 64)));
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Tier(106)), "first pass on, 64 bits");
    let ea = relative_difference_float(&carried(&values, &low), &reference);
    assert!(ea <= 1e-25, "first pass on, 64 bits: E/A {ea:e} against 512 bits");
    let (status, route, values, low) = run_with(first_pass, &hi_t, Some((&lo_t, 106)));
    assert_eq!((status, route), (LaneStatus::Success, LaneRoute::Tier(106)), "first pass on, translated");
    let (_, translated) = exact(&hi_t, &lo_t, 106);
    let ea = relative_difference_float(&carried(&values, &low), &translated);
    assert!(ea <= 1e-25, "first pass on, translated: E/A {ea:e} against 512 bits");
    // Invalid declarations.
    let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
    for bits in [53, 107] {
        assert!(workspace.reduce_batch_with_low_inputs(&batch(hi_t.clone()), &batch(lo_t.clone()), bits).is_err());
    }
}

/// The direct engine at 512 bits on one lane with `powers`.
fn direct512_powers(n: usize, rank: u16, powers: &[u16], lane: &[Complex<f64>]) -> (LaneStatus, Vec<Complex<Float>>) {
    let mut settings = ReductionSettings::new(n, rank);
    settings.powers = powers.to_vec();
    direct_with_inputs(512, settings, lane, None)
}

/// The mass derivative of the 512-bit lane with `powers`, in the squared
/// mass of `line` (massive, and analytic there): a central difference with
/// one Richardson step at `h = 2^-40 m^2`, exact binary64 offsets of the
/// dyadic masses. That is the integral with `line`'s power raised by one,
/// to about `h^4`.
fn mass_derivative512(
    shifts: &[[f64; 4]],
    masses: &[f64],
    mu: f64,
    rank: u16,
    powers: &[u16],
    line: usize,
) -> Vec<Complex<Float>> {
    let n = shifts.len();
    let at = |offset: f64| {
        let mut moved = masses.to_vec();
        moved[line] += offset;
        let (status, values) = direct512_powers(n, rank, powers, &coulomb_lane(shifts, &moved, mu));
        assert_eq!(status, LaneStatus::Success, "powers {powers:?}, m^2 offset {offset:e}");
        values
    };
    let central = |h: f64| -> Vec<Complex<Float>> {
        let (up, down) = (at(h), at(-h));
        let scale = Float::with_val(512, 2.0 * h);
        up.iter()
            .zip(&down)
            .map(|(a, b)| Complex::new((a.re.clone() - &b.re) / &scale, (a.im.clone() - &b.im) / &scale))
            .collect()
    };
    let h = masses[line] * 2f64.powi(-40);
    let (coarse, fine) = (central(h), central(h / 2.0));
    let (four, three) = (Float::with_val(512, 4.0), Float::with_val(512, 3.0));
    coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| {
            Complex::new((f.re.clone() * &four - &c.re) / &three, (f.im.clone() * &four - &c.im) / &three)
        })
        .collect()
}

/// Two P-wave states (onia_loop's two-P-wave virtuals): integrals with two
/// doubled lines, and with a doubled line outside a Coulomb triple, whose
/// jets have two or three Taylor variables (the dotted lines and the
/// triple's raised gluon). Each against the mass derivative of the same
/// integral with one line fewer doubled, at 512 bits, where that line is
/// massive and outside every Coulomb triple; and in binary64.
/// - A generic massive box and pentagon with two doubled lines.
/// - A box of a Coulomb triple (Q, g, Qbar at threshold) and a massive line
///   doubled outside it, against the unit-power box, which main reduces.
/// - A pentagon of a Coulomb triple and two massive lines, one or both
///   doubled: the first against the unit-power pentagon, the second against
///   the first.
#[test]
fn two_doubled_lines_match_the_mass_derivative_of_one() {
    let mu = 3.0;
    let (k, q2, q3, q4) = ([3.0, 0.0, 0.0, 3.0], [3.0, 1.0, -1.0, 2.0], [-2.0, 0.5, 1.0, -1.0], [1.5, -1.0, 0.5, 0.25]);
    let (kc, q1) = ([5.0, 0.0, 0.0, 4.0], [1.0, 2.0, -1.0, 0.5]);
    let triple = [q1, plus(q1, kc, 1.0), plus(q1, kc, 2.0)];
    let generic_box = vec![[0.0; 4], k, q2, q3];
    let generic_pentagon = vec![[0.0; 4], k, q2, q3, q4];
    let coulomb_box = vec![[0.0; 4], triple[0], triple[1], triple[2]];
    let coulomb_pentagon = vec![[0.0; 4], triple[0], triple[1], triple[2], q4];
    // (label, shifts, masses, powers, derivative line, ranks, binary64 E/A bound)
    type Case = (&'static str, Vec<[f64; 4]>, Vec<f64>, Vec<u16>, usize, std::ops::RangeInclusive<u16>, f64);
    let cases: Vec<Case> = vec![
        ("generic box, lines 1 and 2", generic_box.clone(), vec![1.25, 2.0, 0.75, 1.5], vec![1, 2, 2, 1], 2, 0..=4, 1e-10),
        ("generic box, lines 0 and 3", generic_box, vec![1.25, 2.0, 0.75, 1.5], vec![2, 1, 1, 2], 3, 0..=4, 1e-10),
        ("generic pentagon, lines 1 and 3", generic_pentagon, vec![1.25, 2.0, 0.75, 1.5, 1.0], vec![1, 2, 1, 2, 1], 3, 0..=5, 1e-9),
        ("Coulomb box, line 0 outside the triple", coulomb_box, vec![2.0, 9.0, 0.0, 9.0], vec![2, 1, 1, 1], 0, 1..=4, 1e-9),
        ("Coulomb pentagon, line 4 outside", coulomb_pentagon.clone(), vec![2.0, 9.0, 0.0, 9.0, 1.5], vec![1, 1, 1, 1, 2], 4, 1..=5, 1e-8),
        ("Coulomb pentagon, lines 0 and 4 outside", coulomb_pentagon, vec![2.0, 9.0, 0.0, 9.0, 1.5], vec![2, 1, 1, 1, 2], 0, 1..=5, 1e-8),
    ];
    for (label, shifts, masses, powers, line, ranks, bound) in cases {
        let n = shifts.len();
        let lane = coulomb_lane(&shifts, &masses, mu);
        let mut lower = powers.clone();
        lower[line] -= 1;
        for rank in ranks {
            let reference = mass_derivative512(&shifts, &masses, mu, rank, &lower, line);
            let (status, values) = direct512_powers(n, rank, &powers, &lane);
            assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, 512 bits");
            let ea = relative_difference_float(&values, &reference);
            assert!(ea <= 1e-30, "{label}, rank {rank}: 512 bits {ea:e} from d/dm^2");
            let mut settings = ReductionSettings::new(n, rank);
            settings.powers = powers.clone();
            let (status, values) = direct_lanes_with(settings, &lane);
            assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, binary64");
            let ea = relative_difference(&values, &to_f64(&reference));
            assert!(ea <= bound, "{label}, rank {rank}: binary64 {ea:e} from d/dm^2");
        }
    }
}

/// Two P-wave states at v = 0: MadLoop's lanes of e+ e- > chic0 hc and
/// g g > chic0 chic0 (one per propagator count, rank, powers and massive
/// lines) that TIRust refused before, two doubled lines or a doubled line
/// outside a Coulomb triple (benchmarks/onia/make_two_dotted_fixture.py).
/// - Their scalar components at MadLoop's rank, in binary64 and at 256 bits
///   (inputs declared binary64), against the reference: exact partial
///   fractions of the lane's linearly dependent subsets down to integrals
///   without one, from original IREGI's quad scalars with powers. IREGI's
///   direct value is wrong on six of these lanes (by 2.1e-6 to 5.6); its
///   tensor API with powers stops on them. Measured: 2.3e-9 at both
///   precisions, where MadLoop's supplied invariants and the reference's
///   shifts differ by their rounding near threshold.
/// - The whole array against that of the lines in reverse order, which
///   raises the other dotted member of a linearly dependent set holding both
///   and changes every origin: 8.6e-10 in binary64, 7.9e-10 at 256 bits, the
///   same inputs' consistency (exact inputs:
///   `two_dotted_members_of_a_coulomb_triple_do_not_depend_on_their_labels`).
#[test]
fn two_dotted_lanes_match_their_reference_and_their_relabelling() {
    use iregi_rust::{InvariantInput, ScaleInput};
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-two-dotted.json")).unwrap();
    let parse = |v: &Value| v.as_str().unwrap().parse::<f64>().unwrap();
    let (mut largest, mut largest_wide, mut largest_relabelled, mut largest_relabelled_wide) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let shifts: Vec<[f64; 4]> = case["shifts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| std::array::from_fn(|a| parse(&q.as_array().unwrap()[a])))
            .collect();
        let invariants: Vec<f64> = case["invariants"].as_array().unwrap().iter().map(parse).collect();
        let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(parse).collect();
        let powers: Vec<u16> = case["powers"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u16).collect();
        let mu2 = parse(&case["mu_squared"]);
        let reference: Vec<Complex<f64>> =
            case["reference"][0].as_array().unwrap().iter().map(|z| Complex::new(parse(&z[0]), parse(&z[1]))).collect();
        let n = masses.len();
        let rank = case["rank"].as_u64().unwrap() as u16;
        let settings_for = |powers: Vec<u16>| {
            let mut settings = ReductionSettings::new(n, rank);
            settings.invariants = InvariantInput::SuppliedInvariants;
            settings.scale_input = ScaleInput::MuSquared;
            settings.powers = powers;
            settings
        };
        let settings = settings_for(powers.clone());
        let lane = supplied_lane(&shifts, &invariants, &masses, mu2);
        let binary64 = direct_lanes_with(settings.clone(), &lane);
        // At 256 bits with the inputs declared binary64, as MadLoop's tiers.
        let wide = direct_with_inputs(256, settings, &lane, Some(53));
        for (what, (status, values)) in [("binary64", binary64.clone()), ("256 bits", (wide.0.clone(), to_f64(&wide.1)))] {
            assert_eq!(status, LaneStatus::Success, "{label}, {what}");
            let ea = relative_difference(&values[..4], &reference);
            let worst = if what == "binary64" { &mut largest } else { &mut largest_wide };
            *worst = worst.max(ea);
            assert!(ea <= 1e-8, "{label}, {what}: E/A {ea:e} against the reference");
        }
        let reversed: Vec<usize> = (0..n).rev().collect();
        let moved_shifts: Vec<[f64; 4]> = reversed.iter().map(|&i| shifts[i]).collect();
        let moved_masses: Vec<f64> = reversed.iter().map(|&i| masses[i]).collect();
        let mut moved = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                moved[a + n * b] = invariants[reversed[a] + n * reversed[b]];
            }
        }
        let lane = supplied_lane(&moved_shifts, &moved, &moved_masses, mu2);
        let moved_settings = settings_for(reversed.iter().map(|&i| powers[i]).collect());
        let (status, values) = direct_lanes_with(moved_settings.clone(), &lane);
        assert_eq!(status, LaneStatus::Success, "{label}, reversed lines");
        let ea = relative_difference(&values, &binary64.1);
        largest_relabelled = largest_relabelled.max(ea);
        assert!(ea <= 1e-8, "{label}: binary64 E/A {ea:e} against the reversed lines");
        // The same inputs at 256 bits agree only to the inputs' own
        // consistency: MadLoop's invariants hold the thresholds to their
        // rounding (2e-15 here), and the labellings raise different members
        // and read different invariants. Exact inputs: the dyadic Coulomb
        // lanes of `two_dotted_members_of_a_coulomb_triple_do_not_depend_on_their_labels`.
        let (status, values) = direct_with_inputs(256, moved_settings, &lane, Some(53));
        assert_eq!(status, LaneStatus::Success, "{label}, reversed lines, 256 bits");
        let ea = relative_difference_float(&values, &wide.1);
        largest_relabelled_wide = largest_relabelled_wide.max(ea);
        assert!(ea <= 1e-8, "{label}: 256-bit E/A {ea:e} against the reversed lines");
    }
    eprintln!(
        "two-dotted lanes: scalars against the reference, binary64 {largest:e}, 256 bits {largest_wide:e}; \
         reversed lines, binary64 {largest_relabelled:e}, 256 bits {largest_relabelled_wide:e}"
    );
}

/// Two dotted members of one Coulomb triple (Q, g, Qbar at threshold), on
/// exact dyadic inputs: the partial fraction raises the first dotted member
/// and keeps the other's line (the set's own order with a power moved to
/// the raised member). Reversing the lines raises the other member, so both
/// routes must give the same array at 512 bits; and in binary64. A box (a
/// massive line and the triple) and a pentagon (two massive lines and the
/// triple), each with the heavy line and the gluon, or both heavy lines,
/// doubled.
#[test]
fn two_dotted_members_of_a_coulomb_triple_do_not_depend_on_their_labels() {
    let mu = 3.0;
    let (kc, q1, q4) = ([5.0, 0.0, 0.0, 4.0], [1.0, 2.0, -1.0, 0.5], [1.5, -1.0, 0.5, 0.25]);
    let triple = [q1, plus(q1, kc, 1.0), plus(q1, kc, 2.0)];
    let coulomb_box = vec![[0.0; 4], triple[0], triple[1], triple[2]];
    let coulomb_pentagon = vec![[0.0; 4], triple[0], triple[1], triple[2], q4];
    type Case = (&'static str, Vec<[f64; 4]>, Vec<f64>, Vec<u16>, std::ops::RangeInclusive<u16>);
    let cases: Vec<Case> = vec![
        ("box, heavy line and gluon", coulomb_box.clone(), vec![2.0, 9.0, 0.0, 9.0], vec![1, 2, 2, 1], 1..=4),
        ("box, both heavy lines", coulomb_box, vec![2.0, 9.0, 0.0, 9.0], vec![1, 2, 1, 2], 1..=4),
        ("pentagon, gluon and heavy line", coulomb_pentagon.clone(), vec![2.0, 9.0, 0.0, 9.0, 1.5], vec![1, 1, 2, 2, 1], 1..=5),
        ("pentagon, both heavy lines", coulomb_pentagon, vec![2.0, 9.0, 0.0, 9.0, 1.5], vec![1, 2, 1, 2, 1], 1..=5),
    ];
    for (label, shifts, masses, powers, ranks) in cases {
        let n = shifts.len();
        let lane = coulomb_lane(&shifts, &masses, mu);
        let reversed: Vec<usize> = (0..n).rev().collect();
        let moved_shifts: Vec<[f64; 4]> = reversed.iter().map(|&i| shifts[i]).collect();
        let moved_masses: Vec<f64> = reversed.iter().map(|&i| masses[i]).collect();
        let moved_powers: Vec<u16> = reversed.iter().map(|&i| powers[i]).collect();
        let moved = coulomb_lane(&moved_shifts, &moved_masses, mu);
        for rank in ranks {
            let (status, values) = direct512_powers(n, rank, &powers, &lane);
            assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, 512 bits");
            let (status, other) = direct512_powers(n, rank, &moved_powers, &moved);
            assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, 512 bits, reversed");
            let ea = relative_difference_float(&other, &values);
            assert!(ea <= 1e-60, "{label}, rank {rank}: 512 bits {ea:e} against the reversed lines");
            let mut settings = ReductionSettings::new(n, rank);
            settings.powers = powers.clone();
            let (status, binary64) = direct_lanes_with(settings, &lane);
            assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, binary64");
            let ea = relative_difference(&binary64, &to_f64(&values));
            assert!(ea <= 1e-8, "{label}, rank {rank}: binary64 {ea:e} from 512 bits");
        }
    }
}

/// A g g > chic0 chic0 hexagon (unit powers, rank 4, MadLoop's binary64
/// inputs at sqrt(s) = 20 GeV) with three linearly dependent sets, two of
/// which share a heavy line that the third does not hold. 59aee45 raised
/// the shared heavy line in the first two and kept it when the third came,
/// so the geometry rejected the massive raised member, the lane restarted
/// from the base plan, and it built the same profile again, forever. Each
/// set now raises its own member again when no member is common to all,
/// and a lane restarts at most once. The lane returns (refused, as on
/// d17085a, where sets without a common member were refused).
#[test]
fn linearly_dependent_sets_without_a_common_member_return() {
    use iregi_rust::{InvariantInput, ScaleInput};
    const HANG_M2: [u64; 6] = [0x40033851eb851eb9, 0x40033851eb851eb9, 0x40033851eb851eb9, 0x0000000000000000, 0x40033851eb851eb9, 0x0000000000000000];
    const HANG_S: [[u64; 6]; 6] = [
        [0x0000000000000000, 0x0000000000000000, 0x4079000000000000, 0x40694ce147ae1479, 0x40233851eb851ec4, 0x40033851eb851eb9],
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xc0609be8cd135c44, 0xc0704f07856547c8, 0xc0609be8cd135c43],
        [0x4079000000000000, 0x0000000000000000, 0x0000000000000000, 0x40033851eb851eb9, 0x40233851eb851ed4, 0x40694ce147ae147b],
        [0x40694ce147ae1479, 0xc0609be8cd135c44, 0x40033851eb851eb9, 0x0000000000000000, 0x40033851eb851eb9, 0x4059000000000000],
        [0x40233851eb851ec4, 0xc0704f07856547c8, 0x40233851eb851ed4, 0x40033851eb851eb9, 0x0000000000000000, 0x40033851eb851eb9],
        [0x40033851eb851eb9, 0xc0609be8cd135c43, 0x40694ce147ae147b, 0x4059000000000000, 0x40033851eb851eb9, 0x0000000000000000],
    ];
    const HANG_P: [[u64; 4]; 6] = [
        [0xbce0000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4023ffffffffffff, 0x0000000000000000, 0x0000000000000000, 0x4023ffffffffffff],
        [0x4033ffffffffffff, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x402dfffffffffffe, 0x400e2fb56247673b, 0xc002541aee163b5a, 0xbffc384084d604c8],
        [0x4023fffffffffffe, 0x401e2fb56247673b, 0xc012541aee163b5a, 0xc00c384084d604c8],
        [0x4013fffffffffffd, 0x400e2fb56247673b, 0xc002541aee163b5a, 0xbffc384084d604c8],
    ];
    const HANG_MU2: u64 = 0x4079000000000000;
    let f = f64::from_bits;
    let shifts: Vec<[f64; 4]> = HANG_P.iter().map(|q| q.map(f)).collect();
    let invariants: Vec<f64> = (0..36).map(|k| f(HANG_S[k % 6][k / 6])).collect();
    let masses: Vec<f64> = HANG_M2.iter().map(|&m| f(m)).collect();
    let lane = supplied_lane(&shifts, &invariants, &masses, f(HANG_MU2));
    let mut settings = ReductionSettings::new(6, 4);
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    let (status, _) = direct_lanes_with(settings, &lane);
    assert!(matches!(status, LaneStatus::Unstable(_)), "{status:?}");
}

/// The pole slots of the two-dotted lanes' tensors (onia-two-dotted.json),
/// which `two_dotted_lanes_match_their_reference_and_their_relabelling`
/// references only in their scalar components. A numerator factor
/// D_i = q^2 + 2 q.p_i + p_i^2 - m_i^2 cancels a power of line i:
///   sum_mu eta_mu M(e + 2 e_mu) + 2 sum_mu eta_mu p_i^mu M(e + e_mu)
///     + (p_i^2 - m_i^2) M(e) = M'(e),  |e| <= R - 2,
/// with M' the lane with line i's power lowered (a pinch at power 1). The
/// (D - 4)-dimensional part of q^2 adds finite rational terms only, so the
/// UV and IR slots satisfy it exactly. For a doubled line M' is the
/// single-dotted array (one jet variable); otherwise it is a smaller
/// two-dotted integral. At 256 bits, inputs declared binary64. Measured:
/// 1e-10 of the pole slots' scale at most, 4e-15 on e+ e- > chic0 hc
/// (against the larger of both sides' scales: a pinch may have no poles).
#[test]
fn two_dotted_pole_slots_satisfy_the_cancellation_identity() {
    use iregi_rust::{InvariantInput, ScaleInput, basis::tensor_basis};
    let fixture: Value = serde_json::from_str(include_str!("fixtures/onia-two-dotted.json")).unwrap();
    let parse = |v: &Value| v.as_str().unwrap().parse::<f64>().unwrap();
    let eta = [1.0, -1.0, -1.0, -1.0];
    let mut relations = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let shifts: Vec<[f64; 4]> = case["shifts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| std::array::from_fn(|a| parse(&q.as_array().unwrap()[a])))
            .collect();
        let invariants: Vec<f64> = case["invariants"].as_array().unwrap().iter().map(parse).collect();
        let masses: Vec<f64> = case["masses_squared"].as_array().unwrap().iter().map(parse).collect();
        let powers: Vec<u16> = case["powers"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u16).collect();
        let mu2 = parse(&case["mu_squared"]);
        let (n, rank) = (masses.len(), case["rank"].as_u64().unwrap() as u16);
        // The lane of `lines` (the same routing) with `powers` at `rank`.
        let run = |lines: &[usize], powers: Vec<u16>, rank: u16| {
            let m = lines.len();
            let s: Vec<[f64; 4]> = lines.iter().map(|&k| shifts[k]).collect();
            let ms: Vec<f64> = lines.iter().map(|&k| masses[k]).collect();
            let mut inv = vec![0.0; m * m];
            for a in 0..m {
                for b in 0..m {
                    inv[a + m * b] = invariants[lines[a] + n * lines[b]];
                }
            }
            let mut settings = ReductionSettings::new(m, rank);
            settings.invariants = InvariantInput::SuppliedInvariants;
            settings.scale_input = ScaleInput::MuSquared;
            settings.powers = powers;
            let (status, values) = direct_with_inputs(256, settings, &supplied_lane(&s, &inv, &ms, mu2), Some(53));
            assert_eq!(status, LaneStatus::Success, "{label}: lines {lines:?}");
            to_f64(&values)
        };
        let all: Vec<usize> = (0..n).collect();
        let lhs = run(&all, powers.clone(), rank);
        // The size of the lane's own pole slots (a pinch may have none).
        let lhs_scale = lhs.chunks(4).flat_map(|z| z[..3].iter()).map(|z| z.re.hypot(z.im)).fold(0.0f64, f64::max);
        let basis = tensor_basis(rank);
        let index = |e: [u16; 4]| basis.iter().position(|m| m.0 == e).unwrap();
        for i in 0..n {
            let rhs = if powers[i] > 1 {
                let mut lower = powers.clone();
                lower[i] -= 1;
                run(&all, lower, rank - 2)
            } else {
                let keep: Vec<usize> = all.iter().copied().filter(|&k| k != i).collect();
                run(&keep, keep.iter().map(|&k| powers[k]).collect(), rank - 2)
            };
            let p = shifts[i];
            let p2: f64 = (0..4).map(|mu| eta[mu] * p[mu] * p[mu]).sum();
            let (mut worst, mut scale) = (0.0f64, 0.0f64);
            for (k, m) in tensor_basis(rank - 2).iter().enumerate() {
                for slot in 0..3 {
                    let mut acc = lhs[4 * index(m.0) + slot] * Complex::new(p2 - masses[i], 0.0);
                    for mu in 0..4 {
                        let (mut e2, mut e1) = (m.0, m.0);
                        e2[mu] += 2;
                        e1[mu] += 1;
                        acc = acc + lhs[4 * index(e2) + slot] * Complex::new(eta[mu], 0.0);
                        acc = acc + lhs[4 * index(e1) + slot] * Complex::new(2.0 * eta[mu] * p[mu], 0.0);
                    }
                    let r = rhs[4 * k + slot];
                    worst = worst.max((acc.re - r.re).hypot(acc.im - r.im));
                    scale = scale.max(r.re.hypot(r.im));
                }
            }
            assert!(worst <= 1e-9 * scale.max(lhs_scale), "{label}, line {i}: pole slots {worst:e} off, scale {scale:e}");
            relations += 1;
        }
    }
    assert_eq!(relations, 224);
}

/// Two doubled gluon lines with a lightlike difference: a scaleless bubble
/// whose two lines are both jet variables (g g > chic0 chic0 at 20 GeV,
/// MadLoop's binary64 lanes at rank 6: a massless box with powers
/// (1, 1, 2, 2) and a pentagon with one heavy line, (1, 1, 1, 2, 2); TIRust
/// refused both before), and that bubble alone with powers (2, 2) and
/// (2, 3). The box and pentagon read only orders of the bubble that vanish
/// (not logarithmically divergent): doubling the residue of the others
/// leaves them bit-identical. The bubbles alone read that residue.
///
/// No reference evaluates these tensors. At 256 bits, inputs declared
/// binary64, they are checked against binary64, against themselves with
/// their lines reversed, and against smaller integrals by two exact
/// identities:
/// - `2 q.(p_i - p_j) = D_i - D_j - (p_i^2 - m_i^2) + (p_j^2 - m_j^2)`, in
///   every slot (`q.p` has no `(-2 eps)`-dimensional part), for every pair of
///   lines and the components up to rank 5. For the two doubled lines, the
///   smaller integrals have one doubled line;
/// - the cancellation identity of
///   `two_dotted_pole_slots_satisfy_the_cancellation_identity` (`q^2` from
///   the four-dimensional components), up to rank 4, in the UV and double
///   pole slots, and in every slot for the bubbles alone. The `(-2 eps)`-
///   dimensional part of `q^2` adds the same tensor to every line's identity;
///   it vanishes for a scaleless bubble. When a doubled massless line lies
///   between an on-shell heavy leg and a lightlike one (the pentagon's line
///   3), its soft region gives it a single pole: the pentagon with only that
///   line doubled, one jet variable, misses the identity by 9% in that slot,
///   as this pentagon does by 8%.
///
/// The box and pentagon are UV finite (`4 + 6 - 2 sum nu < 0`): their UV slot
/// vanishes. Measured: binary64 8.1e-10 from 256 bits (the pentagon; the
/// others 1.3e-14 at most); identities 3.7e-16 of their terms at most;
/// reversed lines 2e-74.
#[test]
fn two_doubled_lines_of_a_scaleless_bubble_satisfy_exact_identities() {
    use iregi_rust::{InvariantInput, ScaleInput, basis::tensor_basis};
    const BOX_S: [[u64; 4]; 4] = [
        [0x0000000000000000, 0x0000000000000000, 0xc058fffffffffffe, 0x4058fffffffffffe],
        [0x0000000000000000, 0x0000000000000000, 0x4058fffffffffffe, 0xc058fffffffffffe],
        [0xc058fffffffffffe, 0x4058fffffffffffe, 0x0000000000000000, 0x0000000000000000],
        [0x4058fffffffffffe, 0xc058fffffffffffe, 0x0000000000000000, 0x0000000000000000],
    ];
    const BOX_P: [[u64; 4]; 4] = [
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4023ffffffffffff, 0x0000000000000000, 0x0000000000000000, 0x4023ffffffffffff],
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x4023ffffffffffff],
        [0x4023ffffffffffff, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
    ];
    const PENTAGON_M2: [u64; 5] = [0, 0, 0x40033851eb851eb9, 0, 0];
    const PENTAGON_S: [[u64; 5]; 5] = [
        [0x0000000000000000, 0x0000000000000000, 0xc0609be8cd135c43, 0xc058fffffffffffe, 0x4058fffffffffffe],
        [0x0000000000000000, 0x0000000000000000, 0x40033851eb851eb9, 0x4058fffffffffffe, 0xc058fffffffffffe],
        [0xc0609be8cd135c43, 0x40033851eb851eb9, 0x0000000000000000, 0x40033851eb851eb9, 0xc0609be8cd135c43],
        [0xc058fffffffffffe, 0x4058fffffffffffe, 0x40033851eb851eb9, 0x0000000000000000, 0x0000000000000000],
        [0x4058fffffffffffe, 0xc058fffffffffffe, 0xc0609be8cd135c43, 0x0000000000000000, 0x0000000000000000],
    ];
    const PENTAGON_P: [[u64; 4]; 5] = [
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4023ffffffffffff, 0x0000000000000000, 0x0000000000000000, 0x4023ffffffffffff],
        [0x4013ffffffffffff, 0xc00e2fb56247673b, 0x4002541aee163b5a, 0x40278708109ac098],
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x4023ffffffffffff],
        [0x4023ffffffffffff, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
    ];
    const MU2: u64 = 0x4079000000000000;
    let f = f64::from_bits;
    let eta = [1.0, -1.0, -1.0, -1.0];
    let rank = 6u16;
    let norm = |z: Complex<f64>| z.re.hypot(z.im);
    let square = |p: [f64; 4]| (0..4).map(|mu| eta[mu] * p[mu] * p[mu]).sum::<f64>();
    let box_shifts: Vec<[f64; 4]> = BOX_P.iter().map(|q| q.map(f)).collect();
    // (label, powers, squared masses, invariants, shifts)
    let cases: [(&str, Vec<u16>, Vec<f64>, Vec<f64>, Vec<[f64; 4]>); 4] = [
        ("box", vec![1, 1, 2, 2], vec![0.0; 4], BOX_S.iter().flatten().map(|&x| f(x)).collect(), box_shifts.clone()),
        (
            "pentagon",
            vec![1, 1, 1, 2, 2],
            PENTAGON_M2.map(f).to_vec(),
            PENTAGON_S.iter().flatten().map(|&x| f(x)).collect(),
            PENTAGON_P.iter().map(|q| q.map(f)).collect(),
        ),
        ("bubble (2, 2)", vec![2, 2], vec![0.0; 2], vec![0.0; 4], vec![box_shifts[2], box_shifts[3]]),
        ("bubble (2, 3)", vec![2, 3], vec![0.0; 2], vec![0.0; 4], vec![box_shifts[2], box_shifts[3]]),
    ];
    let (mut relations, mut measured) = (0, [0.0f64; 3]);
    for (label, powers, masses, invariants, shifts) in &cases {
        let n = masses.len();
        // The settings and lane of `lines` (the same routing) with `powers`
        // at `rank`.
        let input = |lines: &[usize], powers: Vec<u16>, rank: u16| {
            let m = lines.len();
            let s: Vec<[f64; 4]> = lines.iter().map(|&k| shifts[k]).collect();
            let ms: Vec<f64> = lines.iter().map(|&k| masses[k]).collect();
            let mut inv = vec![0.0; m * m];
            for a in 0..m {
                for b in 0..m {
                    inv[a + m * b] = invariants[lines[a] + n * lines[b]];
                }
            }
            let mut settings = ReductionSettings::new(m, rank);
            settings.invariants = InvariantInput::SuppliedInvariants;
            settings.scale_input = ScaleInput::MuSquared;
            settings.powers = powers;
            (settings, supplied_lane(&s, &inv, &ms, f(MU2)))
        };
        let run = |lines: &[usize], powers: Vec<u16>, rank: u16| {
            let (settings, lane) = input(lines, powers, rank);
            let (status, values) = direct_with_inputs(256, settings, &lane, Some(53));
            assert_eq!(status, LaneStatus::Success, "{label}: lines {lines:?}");
            to_f64(&values)
        };
        let all: Vec<usize> = (0..n).collect();
        // The lane with line `i` lowered by one power (dropped at power 1).
        let lowered = |i: usize, rank: u16| {
            if powers[i] > 1 {
                let mut lower = powers.clone();
                lower[i] -= 1;
                run(&all, lower, rank)
            } else {
                let keep: Vec<usize> = all.iter().copied().filter(|&k| k != i).collect();
                run(&keep, keep.iter().map(|&k| powers[k]).collect(), rank)
            }
        };
        let full = run(&all, powers.clone(), rank);
        let scale = full.iter().map(|&z| norm(z)).fold(0.0f64, f64::max);
        let (settings, lane) = input(&all, powers.clone(), rank);
        let (status, binary64) = direct_lanes_with(settings, &lane);
        assert_eq!(status, LaneStatus::Success, "{label}: binary64");
        let ea = relative_difference(&binary64, &full);
        assert!(ea <= 1e-8, "{label}: binary64 {ea:e} from 256 bits");
        let bubble = n == 2;
        if !bubble {
            let uv = full.chunks(4).map(|z| norm(z[0])).fold(0.0f64, f64::max);
            assert!(uv <= 1e-30 * scale, "{label}: UV slot {uv:e} of a UV-finite integral");
        }
        let basis = tensor_basis(rank);
        let index = |e: [u16; 4]| basis.iter().position(|m| m.0 == e).unwrap();
        // 2 q.(p_i - p_j).
        let reduced: Vec<Vec<Complex<f64>>> = (0..n).map(|i| lowered(i, rank - 1)).collect();
        for i in 0..n {
            for j in i + 1..n {
                let d: [f64; 4] = std::array::from_fn(|mu| shifts[i][mu] - shifts[j][mu]);
                let delta = square(shifts[i]) - masses[i] - square(shifts[j]) + masses[j];
                for slot in 0..4 {
                    let (mut worst, mut terms_scale) = (0.0f64, 0.0f64);
                    for (k, m) in tensor_basis(rank - 1).iter().enumerate() {
                        let mut terms = vec![
                            reduced[i][4 * k + slot],
                            -reduced[j][4 * k + slot],
                            -full[4 * index(m.0) + slot] * Complex::new(delta, 0.0),
                        ];
                        for mu in 0..4 {
                            let mut e = m.0;
                            e[mu] += 1;
                            terms.push(full[4 * index(e) + slot] * Complex::new(-2.0 * eta[mu] * d[mu], 0.0));
                        }
                        let sum = terms.iter().fold(Complex::new(0.0, 0.0), |a, &t| a + t);
                        worst = worst.max(norm(sum));
                        terms_scale = terms.iter().fold(terms_scale, |s, &t| s.max(norm(t)));
                    }
                    assert!(
                        worst <= 1e-9 * terms_scale,
                        "{label}, lines {i}, {j}, slot {slot}: {worst:e} off, terms {terms_scale:e}"
                    );
                    if terms_scale > 0.0 {
                        measured[0] = measured[0].max(worst / terms_scale);
                    }
                }
                relations += 1;
            }
        }
        // q^2 = sum_mu eta_mu (q^mu)^2 + (-2 eps)-dimensional part.
        let slots: &[usize] = if bubble { &[0, 1, 2, 3] } else { &[0, 2] };
        for i in 0..n {
            let rhs = lowered(i, rank - 2);
            let p = shifts[i];
            for &slot in slots {
                let (mut worst, mut terms_scale) = (0.0f64, 0.0f64);
                for (k, m) in tensor_basis(rank - 2).iter().enumerate() {
                    let mut terms = vec![
                        full[4 * index(m.0) + slot] * Complex::new(square(p) - masses[i], 0.0),
                        -rhs[4 * k + slot],
                    ];
                    for mu in 0..4 {
                        let (mut e2, mut e1) = (m.0, m.0);
                        e2[mu] += 2;
                        e1[mu] += 1;
                        terms.push(full[4 * index(e2) + slot] * Complex::new(eta[mu], 0.0));
                        terms.push(full[4 * index(e1) + slot] * Complex::new(2.0 * eta[mu] * p[mu], 0.0));
                    }
                    let sum = terms.iter().fold(Complex::new(0.0, 0.0), |a, &t| a + t);
                    worst = worst.max(norm(sum));
                    terms_scale = terms.iter().fold(terms_scale, |s, &t| s.max(norm(t)));
                }
                assert!(
                    worst <= 1e-9 * terms_scale,
                    "{label}, line {i}, slot {slot}: {worst:e} off, terms {terms_scale:e}"
                );
                if terms_scale > 0.0 {
                    measured[1] = measured[1].max(worst / terms_scale);
                }
            }
            relations += 1;
        }
        // The lines in reverse order.
        let reversed: Vec<usize> = all.iter().rev().copied().collect();
        let relabelled = run(&reversed, reversed.iter().map(|&k| powers[k]).collect(), rank);
        let worst = full.iter().zip(&relabelled).map(|(&a, &b)| norm(a - b)).fold(0.0f64, f64::max);
        assert!(worst <= 1e-12 * scale, "{label} reversed: {worst:e} off, scale {scale:e}");
        measured[2] = measured[2].max(worst / scale);
        eprintln!("{label}: binary64 {ea:e}");
    }
    eprintln!("identities and reversal: {measured:?}");
    assert_eq!(relations, (6 + 4) + (10 + 5) + 2 * (1 + 2));
}
