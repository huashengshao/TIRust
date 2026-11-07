//! Per-subset sharing between the integrals of an amplitude
//! (`AmplitudeWorkspace`): bit-identical results with sharing on and off and
//! against standalone workspaces, and actual reuse on physical kinematics.
use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, KinematicsBatch, ReductionSettings, TensorReductionPlan,
    TensorWorkspace,
};
use symbolica::domains::float::Complex;

mod common;
use common::hexagon::{MT2, MU, POINTS};

/// The hexagon and its pinches with at least four propagators (6 pentagons,
/// 15 boxes), each at rank = number of propagators.
fn integrals() -> Vec<Vec<usize>> {
    (1u32..1 << 6)
        .filter(|m| m.count_ones() >= 4)
        .map(|m| (0..6).filter(|i| m & (1 << i) != 0).collect())
        .collect()
}

fn batch(members: &[usize]) -> KinematicsBatch<f64> {
    batch_with_mass(members, MT2)
}

fn batch_with_mass(members: &[usize], mass2: f64) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    for point in &POINTS {
        for &i in members {
            values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
        }
        values.extend(members.iter().map(|_| Complex::new(mass2, 0.0)));
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
    // The hexagon gathers the moments of its six Gram-free pentagons, which
    // the pentagon integrals have gathered already: reused on every lane.
    assert!(stats.moments_reused >= 6 * POINTS.len() as u64, "too few moments reused: {stats:?}");
    // The partial rescues and the double-double tier evaluate the same box
    // and triangle masters repeatedly: some are reused.
    assert!(stats.dd_masters_reused > 0 && stats.dd_masters_computed > 0, "no double-double master reused: {stats:?}");
    eprintln!("{stats:?}");

    amplitude.set_subset_sharing(false);
    let separate = amplitude.reduce_batches(&batches).unwrap();
    let off = amplitude.last_sharing();
    assert_eq!((off.dd_masters_reused, off.dd_masters_computed), (0, 0), "double-double masters counted with sharing off");
    assert_eq!(off, Default::default());
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
    // The hexagon needs no subset above its own ranks: bit-identical to a
    // standalone workspace. The pinched integrals take their injected boxes
    // and triangles at the amplitude's rank floor: equal to their standalone
    // arrays within the certified accuracy.
    let hexagon = members.iter().position(|m| m.len() == 6).unwrap();
    let mut worst = 0.0f64;
    for (k, plan) in plans.iter().enumerate() {
        let mut standalone = TensorWorkspace::new(plan, 0.0f64).unwrap();
        let alone = standalone.reduce_batch(&batches[k]).unwrap();
        if k == hexagon {
            assert_eq!(fingerprint(&alone), fingerprint(&shared[k]));
        }
        assert_eq!(format!("{:?}", alone.statuses), format!("{:?}", shared[k].statuses));
        for lane in 0..alone.rows() {
            let (Ok(a), Ok(b)) = (alone.lane(lane), shared[k].lane(lane)) else { continue };
            let size = a.iter().map(|z| z.re.hypot(z.im)).fold(0.0f64, f64::max);
            let difference = a.iter().zip(b).map(|(x, y)| (x.re - y.re).hypot(x.im - y.im)).fold(0.0f64, f64::max);
            worst = worst.max(difference / size);
        }
    }
    eprintln!("largest relative difference to standalone workspaces: {worst:e}");
    assert!(worst <= 1e-10, "amplitude results differ from standalone ones by {worst:e}");
}

/// Massless propagators: the integrals have IR poles, so every Laurent slot
/// of the shared moments is gathered and stored.
#[test]
fn shared_moments_with_ir_poles_are_bit_identical() {
    let members = integrals();
    let plans: Vec<TensorReductionPlan> = members
        .iter()
        .map(|m| TensorReductionPlan::new(ReductionSettings::new(m.len(), m.len() as u16)).unwrap())
        .collect();
    let batches: Vec<KinematicsBatch<f64>> = members.iter().map(|m| batch_with_mass(m, 0.0)).collect();
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64).unwrap();
    let shared = amplitude.reduce_batches(&batches).unwrap();
    let stats = amplitude.last_sharing();
    eprintln!("{stats:?}");
    assert!(stats.moments_reused > 0, "no moments reused: {stats:?}");
    let poles = shared.iter().any(|result| {
        (0..result.rows()).any(|lane| {
            result.lane(lane).is_ok_and(|v| v.chunks(4).any(|s| [s[1], s[2]].iter().any(|z| z.re != 0.0 || z.im != 0.0)))
        })
    });
    assert!(poles, "no IR pole in any array");
    amplitude.set_subset_sharing(false);
    let separate = amplitude.reduce_batches(&batches).unwrap();
    for (k, (a, b)) in shared.iter().zip(&separate).enumerate() {
        assert_eq!(fingerprint(a), fingerprint(b), "integral {:?} differs with sharing", members[k]);
    }
}

#[test]
fn one_batch_per_integral_is_required() {
    let plans = vec![TensorReductionPlan::new(ReductionSettings::new(4, 4)).unwrap(); 2];
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64).unwrap();
    assert_eq!(amplitude.len(), 2);
    assert!(amplitude.reduce_batches(&[batch(&[0, 1, 2, 3])]).is_err());
}
