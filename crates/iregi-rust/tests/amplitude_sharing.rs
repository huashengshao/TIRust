//! Per-subset sharing between the integrals of an amplitude
//! (`AmplitudeWorkspace`): bit-identical results with sharing on and off and
//! against standalone workspaces, and actual reuse on physical kinematics.
use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, KinematicsBatch, ReductionSettings, TensorReductionPlan,
    TensorWorkspace,
};
use symbolica::domains::float::Complex;

/// Four top-loop gg -> gggg hexagon points of the binary64 stability study
/// (sqrt(s) = 1 TeV; shifts of the six propagators), m_t^2 = 29929, mu = 1000.
const POINTS: [[[f64; 4]; 6]; 4] = [
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [792.7874729156998, -82.84167294384719, -92.33145751079824, -165.97948836665938],
        [496.1459371514851, -309.53730217508144, -274.5389438821032, -224.33807246752178],
        [365.75699134826516, -254.15872221430251, -237.93327663297802, -112.1127847893476],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [628.3485871137328, -126.38562954754646, -64.66171687618484, 343.46805923528484],
        [493.6035005043037, -75.99778103019892, 59.08875070855528, 360.8785405535477],
        [342.2594345535183, -151.0438342019983, 151.3291327069494, 267.2578797427977],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [720.5411460864439, 41.03231623557652, 269.0508926250419, -63.44459971258532],
        [580.5195723069281, 86.55172419413077, 347.3583949823081, 43.335359537087676],
        [328.9555327352336, -146.2746278878062, 292.60284001366847, -34.62735579481515],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [684.7727360619465, -266.1383293706448, -32.469872857286525, 165.7839706555041],
        [491.59350180547904, -99.14409525080174, -121.72713563298282, 127.51801502347416],
        [243.28769056900498, 107.6510206808816, -17.784570933495488, 217.44853912012346],
    ],
];
const MT2: f64 = 29929.0;
const MU: f64 = 1000.0;

/// The hexagon and its pinches with at least four propagators (6 pentagons,
/// 15 boxes), each at rank = number of propagators.
fn integrals() -> Vec<Vec<usize>> {
    (1u32..1 << 6)
        .filter(|m| m.count_ones() >= 4)
        .map(|m| (0..6).filter(|i| m & (1 << i) != 0).collect())
        .collect()
}

fn batch(members: &[usize]) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    for point in &POINTS {
        for &i in members {
            values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
        }
        values.extend(members.iter().map(|_| Complex::new(MT2, 0.0)));
        values.push(Complex::new(MU, 0.0));
    }
    KinematicsBatch { rows: POINTS.len(), components: 1, values }
}

/// Every lane's array (best-effort ones included), statuses and estimates, bit for bit.
fn fingerprint(result: &BatchResult<f64>) -> (Vec<u64>, String, Vec<Option<u64>>) {
    let mut values = Vec::new();
    for lane in 0..result.rows() {
        match result.lane_with_estimate(lane) {
            Some(best) => values.extend(best.values.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()])),
            None => values.push(u64::MAX),
        }
    }
    let estimates = result.estimated_errors.iter().map(|e| e.map(f64::to_bits)).collect();
    (values, format!("{:?}", result.statuses), estimates)
}

#[test]
fn shared_subsets_give_bit_identical_results_and_are_reused() {
    let members = integrals();
    let plans: Vec<TensorReductionPlan> = members
        .iter()
        .map(|m| TensorReductionPlan::new(ReductionSettings::new(m.len(), m.len() as u16)).unwrap())
        .collect();
    let batches: Vec<KinematicsBatch<f64>> = members.iter().map(|m| batch(m)).collect();

    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64).unwrap();
    assert!(amplitude.subset_sharing(), "sharing is on by default");
    let shared = amplitude.reduce_batches(&batches).unwrap();
    let stats = amplitude.last_sharing();
    assert!(stats.reused > 0 && stats.computed > 0, "no per-subset result was reused: {stats:?}");

    amplitude.set_subset_sharing(false);
    let separate = amplitude.reduce_batches(&batches).unwrap();
    assert_eq!(amplitude.last_sharing(), Default::default());
    for (k, (a, b)) in shared.iter().zip(&separate).enumerate() {
        assert_eq!(fingerprint(a), fingerprint(b), "integral {:?} differs with sharing", members[k]);
    }

    // Off from construction, and standalone workspaces: the same arrays.
    let options = AmplitudeOptions { share_subsets: false, ..Default::default() };
    let mut unshared = AmplitudeWorkspace::new_with_options(&plans, 0.0f64, options).unwrap();
    assert!(!unshared.subset_sharing());
    for (k, result) in unshared.reduce_batches(&batches).unwrap().iter().enumerate() {
        assert_eq!(fingerprint(result), fingerprint(&shared[k]));
    }
    let hexagon = members.iter().position(|m| m.len() == 6).unwrap();
    let mut standalone = TensorWorkspace::new(&plans[hexagon], 0.0f64).unwrap();
    assert_eq!(fingerprint(&standalone.reduce_batch(&batches[hexagon]).unwrap()), fingerprint(&shared[hexagon]));
}

#[test]
fn one_batch_per_integral_is_required() {
    let plans = vec![TensorReductionPlan::new(ReductionSettings::new(4, 4)).unwrap(); 2];
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64).unwrap();
    assert_eq!(amplitude.len(), 2);
    assert!(amplitude.reduce_batches(&[batch(&[0, 1, 2, 3])]).is_err());
}
