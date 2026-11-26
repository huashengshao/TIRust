//! Rank >= 1 triangles near a vanishing Källén function lambda
//! (huashengshao/TIRust#62), with supplied invariants as MadLoop supplies
//! them.
//!
//! The PV reduction of a triangle solves its 2 x 2 Gram system once per
//! rank, and the rounding of each right-hand side is amplified by about
//! `scale ||Z^-1||`. The Gram gate measured the system by its Hadamard ratio
//! and `|det Z|^(1/2) / scale`, which miss that amplification as lambda
//! vanishes: by `lambda^(1/2)` per step near collinear legs, and by `lambda`
//! next to a lightlike leg whose other two invariants are nearly equal (the
//! Hadamard ratio scales the lightlike leg's short Gram row away). Through
//! MadGraph's bridge options binary64 served such lanes up to 1.1e-1 off and
//! the double-double tier up to 5e22 off. For a triangle integral the gate now
//! holds `(G D^(1/2) B)^rank D^(1/2)` against the threshold
//! (`D = |det Z| / max|Z_ij|^2`, `B` the Gram system's row balance), and
//! binary64 and double-double also refuse one whose Gram system alone costs
//! them more than `p - 34` bits, whatever the threshold.
//!
//! Each lane must now come from the precision tier its estimate admits,
//! within 1e-13 of the same binary64 inputs at 1024 bits (relative to each
//! slot's largest entry); a generic triangle stays in binary64.

use iregi_rust::{
    Convention, DirectReductionPlan, EmbedderOptions, InvariantInput, KinematicsBatch, LaneRoute, LaneStatus,
    ReductionSettings, ScaleInput, TensorReductionPlan, TensorWeights, TensorWorkspace,
};
use symbolica::domains::float::{Complex, DoubleFloat, Float, RealLike};

fn settings(rank: u16) -> ReductionSettings {
    let mut settings = ReductionSettings::new(3, rank);
    settings.convention = Convention::MadLoop;
    settings.tensor_weights = TensorWeights::Components;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    settings
}

/// A lane: shifts (three `[E, x, y, z]`), packed invariants `(s01, s02, s12)`,
/// squared masses and mu^2.
#[derive(Clone)]
struct Lane {
    label: &'static str,
    rank: u16,
    values: Vec<Complex<f64>>,
}

fn lane(label: &'static str, rank: u16, shifts: &[f64], invariants: [f64; 3], masses: [f64; 3], mu2: f64) -> Lane {
    let mut values: Vec<Complex<f64>> = shifts.iter().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend(invariants.map(|x| Complex::new(x, 0.0)));
    values.extend(masses.map(|m| Complex::new(m, 0.0)));
    values.push(Complex::new(mu2, 0.0));
    Lane { label, rank, values }
}

fn square(d: [f64; 4]) -> f64 {
    d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]
}

/// Shifts `offset + d` for `d` in `(0, d1, d2)`, and their invariants in
/// binary64 (`s01` given, as MadLoop's on-shell projection sets it).
fn from_shifts(d1: [f64; 4], d2: [f64; 4], s01: Option<f64>) -> (Vec<f64>, [f64; 3]) {
    let offset = [0.3, -0.7, 0.2, 1.1];
    let points: [[f64; 4]; 3] =
        [offset, std::array::from_fn(|k| offset[k] + d1[k]), std::array::from_fn(|k| offset[k] + d2[k])];
    let invariant = |i: usize, j: usize| square(std::array::from_fn(|k| points[i][k] - points[j][k]));
    let shifts = points.iter().flatten().copied().collect();
    (shifts, [s01.unwrap_or_else(|| invariant(0, 1)), invariant(0, 2), invariant(1, 2)])
}

/// `qcd_n4_uux_zzgg`, call 558753 of MadGraph's TIRust pin qualification
/// (issue #62's reproducer): massless lines, s01 = 0, s12 and s02 equal to
/// 5e-4, rank 2. Served by binary64 5.1e-4 off.
fn madloop_lightlike_forward() -> Lane {
    let bits = |b: &[u64]| b.iter().map(|&x| f64::from_bits(x)).collect::<Vec<f64>>();
    let shifts = bits(&[
        0, 0, 0, 0,
        0xc062c491605eb7a9, 0xc02a14a7a7116eda, 0xc06153719ae7e3cd, 0xc04c1c32c04695ed,
        0xc056d96573e66afa, 0xc023086cb106fb70, 0x406becf467ba34dd, 0xc088bb7469ed2f2b,
    ]);
    let [s01, s02, s12] = [0, 0xc12462d688689107, 0xc12463ecba23e8bb].map(f64::from_bits);
    lane("MadLoop lightlike-forward, rank 2", 2, &shifts, [s01, s02, s12], [0.0; 3], f64::from_bits(0x412e848000000000))
}

/// Timelike legs a, b nearly collinear: s02 = (a + b)^2 (1 + eta), all
/// invariants nonzero. Rank 1 at eta = 1e-11 was served by binary64 7.6e-6 off.
fn near_collinear(rank: u16, eta: f64) -> Lane {
    let (a, b) = (1.3f64, 2.1f64);
    let c2 = (a + b) * (a + b) * (1.0 + eta);
    let alpha = (c2 - b * b + a * a) / (2.0 * a);
    let (shifts, invariants) = from_shifts([a, 0.0, 0.0, 0.0], [alpha, 0.0, (alpha * alpha - c2).sqrt(), 0.0], None);
    lane("near-collinear", rank, &shifts, invariants, [3.1, 5.7, 2.2], 9.0)
}

/// A lightlike leg p1 = a (1, 1, 0, 0) (s01 = 0 as projected) and
/// p2 = (c + eps, c, y, 0), so s02 - s12 = 2 p1.p2 = 2 a eps. Rank 3 at
/// eps = 1e-6 was served by the double-double tier 8.7e8 off.
fn lightlike_forward(rank: u16, eps: f64) -> Lane {
    let (a, c, y) = (1.3, 1.7, 0.9);
    let (shifts, invariants) = from_shifts([a, a, 0.0, 0.0], [c + eps, c, y, 0.0], Some(0.0));
    lane("lightlike-forward", rank, &shifts, invariants, [2.3, 0.0, 6.4], 9.0)
}

/// Issue #68's reproducer: two shifts that coincide to about 1e-12, a
/// lightlike leg (s01 = 0, s12 = 0, s02 = -7.6e-9) and three equal masses
/// against mu^2 of the same size. Its G is about 1e-12 while the Gram system
/// itself is well conditioned, so the triangle estimate refuses rank 1 in
/// binary64 and rank 2 in double-double where the Gram gate alone passes.
fn coincident_shifts(rank: u16) -> Lane {
    let bits = |b: &[u64]| b.iter().map(|&x| f64::from_bits(x)).collect::<Vec<f64>>();
    let shifts = bits(&[
        0, 0, 0, 0,
        0x404f358c01babe7b, 0xc04eda752ce9e7a6, 0x4022c25e6e356758, 0xbfe30ba7fe9a3718,
        0x404f358c01ba980d, 0xc04eda752ce9e87b, 0x4022c25e6e34d29f, 0xbfe30ba7fe97cb43,
    ]);
    let mass = f64::from_bits(0x40b93cc89cb252cf);
    lane("coincident shifts", rank, &shifts, [0.0, f64::from_bits(0xbe405ee5d4000000), 0.0], [mass; 3], mass)
}

/// A generic triangle (lambda of order its invariants squared).
fn generic(rank: u16) -> Lane {
    let (shifts, invariants) = from_shifts([1.9, 0.4, -0.3, 0.2], [2.6, -0.5, 1.1, -0.4], None);
    lane("generic", rank, &shifts, invariants, [0.7, 2.9, 1.6], 9.0)
}

fn batch<T: symbolica::domains::float::SingleFloat>(values: Vec<Complex<T>>) -> KinematicsBatch<T> {
    KinematicsBatch { rows: 1, components: 1, values }
}

fn lifted(lane: &Lane, bits: u32) -> KinematicsBatch<Float> {
    batch(lane.values.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect())
}

fn double_double(lane: &Lane) -> KinematicsBatch<DoubleFloat> {
    batch(lane.values.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect())
}

/// The same binary64 inputs at 1024 bits.
fn reference(lane: &Lane) -> Vec<(f64, f64)> {
    let mut workspace = DirectReductionPlan::new(settings(lane.rank)).unwrap().workspace(Float::new(1024)).unwrap();
    workspace.set_input_precision(53);
    workspace.set_stability_threshold(Float::with_val(1024, 2f64.powi(-(1024 - 120))));
    let result = workspace.reduce_batch(&lifted(lane, 1024)).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success, "{}: 1024-bit reference", lane.label);
    result.lane(0).unwrap().iter().map(|z| (z.re.to_f64(), z.im.to_f64())).collect()
}

/// Largest error of a slot over that slot's largest reference entry. Both
/// arrays must be complete and finite: `f64::max` would drop a NaN.
fn error(values: &[Complex<f64>], reference: &[(f64, f64)]) -> f64 {
    assert_eq!(values.len(), reference.len(), "value and reference lengths");
    assert!(values.iter().all(|z| z.re.is_finite() && z.im.is_finite()), "non-finite value: {values:?}");
    assert!(reference.iter().all(|z| z.0.is_finite() && z.1.is_finite()), "non-finite reference: {reference:?}");
    (0..4)
        .map(|slot| {
            let entries: Vec<usize> = (slot..values.len()).step_by(4).collect();
            let scale = entries.iter().map(|&k| reference[k].0.hypot(reference[k].1)).fold(0.0, f64::max);
            let worst = entries
                .iter()
                .map(|&k| (values[k].re - reference[k].0).hypot(values[k].im - reference[k].1))
                .fold(0.0, f64::max);
            if scale == 0.0 { worst } else { worst / scale }
        })
        .fold(0.0, f64::max)
}

/// The binary64 embedder workspace with MadGraph's bridge options.
fn madloop(rank: u16, gate: Option<f64>) -> TensorWorkspace<f64> {
    let plan = TensorReductionPlan::new(settings(rank)).unwrap();
    TensorWorkspace::new_embedded(
        &plan,
        0.0f64,
        EmbedderOptions {
            method_choice: false,
            precision_tiers: true,
            tier_margin_bits: 0,
            double_double_headroom_bits: 26,
            gate,
            ..EmbedderOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn near_degenerate_triangles_come_from_the_tier_their_estimate_admits() {
    let cases = [
        (madloop_lightlike_forward(), LaneRoute::Tier(256)),
        (near_collinear(1, 1e-11), LaneRoute::Tier(106)),
        (lightlike_forward(3, 1e-6), LaneRoute::Tier(256)),
    ];
    for (lane, expected) in cases {
        let exact = reference(&lane);
        for gate in [None, Some(1e-10)] {
            let mut workspace = madloop(lane.rank, gate);
            let result = workspace.reduce_batch(&batch(lane.values.clone())).unwrap();
            let route = workspace.last_lane_routes().unwrap()[0];
            assert_eq!(result.statuses[0], LaneStatus::Success, "{}, gate {gate:?}", lane.label);
            assert_eq!(route, expected, "{}, gate {gate:?}", lane.label);
            let error = error(result.lane(0).unwrap(), &exact);
            assert!(error <= 1e-13, "{}, gate {gate:?}: {route:?} {error:e} off", lane.label);
        }
    }
}

#[test]
fn fixed_precision_refuses_near_degenerate_triangles_at_any_threshold() {
    // Binary64 with its gate off still refuses the nearly collinear lane, with
    // a conditioning estimate for the tiers; the tiers' precisions serve it.
    let collinear = near_collinear(1, 1e-11);
    let plan = DirectReductionPlan::new(settings(1)).unwrap();
    let mut binary64 = plan.workspace(0.0f64).unwrap();
    binary64.set_stability_threshold(0.0);
    let result = binary64.reduce_batch(&batch(collinear.values.clone())).unwrap();
    // The diagnostic names the floor that refused it, not the threshold.
    match &result.statuses[0] {
        LaneStatus::Unstable(reason) => {
            assert!(reason.contains("below the fixed-precision floor"), "{reason}");
            assert!(!reason.contains("below the threshold"), "{reason}");
        }
        status => panic!("{status:?}"),
    }
    let estimate = binary64.last_conditions()[0].expect("a conditioning estimate");
    assert!(estimate > 2f64.powi(-80) && estimate < 2f64.powi(-19), "estimate {estimate:e}");
    let mut double = plan.workspace(DoubleFloat::from(0.0)).unwrap();
    double.set_stability_threshold(DoubleFloat::from(0.0));
    let served = double.reduce_batch(&double_double(&collinear)).unwrap();
    assert_eq!(served.statuses[0], LaneStatus::Success);
    // Double-double refuses the lightlike-forward rank-3 lane it served 8.7e8
    // off; 256 bits with its gate off serves it.
    let forward = lightlike_forward(3, 1e-6);
    let plan = DirectReductionPlan::new(settings(3)).unwrap();
    let mut double = plan.workspace(DoubleFloat::from(0.0)).unwrap();
    double.set_stability_threshold(DoubleFloat::from(0.0));
    let result = double.reduce_batch(&double_double(&forward)).unwrap();
    assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "{:?}", result.statuses[0]);
    let mut wide = plan.workspace(Float::new(256)).unwrap();
    wide.set_input_precision(53);
    wide.set_stability_threshold(Float::with_val(256, 0.0));
    assert_eq!(wide.reduce_batch(&lifted(&forward, 256)).unwrap().statuses[0], LaneStatus::Success);
}

#[test]
fn triangles_refused_by_their_estimate_stay_out_of_the_relation_variants() {
    // A lane the triangle estimate alone refuses used to reach the relation
    // variants, whose rank test took the coincident shifts as dependent: the
    // denominator identity built on them (weights of 1e8 for equal masses)
    // served it O(1) off as a success (binary64 rank 1 1.0 off, double-double
    // rank 2 2.2 off; issue #68). It must now be refused, and the facade's
    // tiers must serve it.
    let rank1 = coincident_shifts(1);
    let plan = DirectReductionPlan::new(settings(1)).unwrap();
    let mut binary64 = plan.workspace(0.0f64).unwrap();
    binary64.set_stability_threshold(1e-12);
    let result = binary64.reduce_batch(&batch(rank1.values.clone())).unwrap();
    assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "binary64 rank 1: {:?}", result.statuses[0]);
    let rank2 = coincident_shifts(2);
    let plan = DirectReductionPlan::new(settings(2)).unwrap();
    let mut double = plan.workspace(DoubleFloat::from(0.0)).unwrap();
    double.set_stability_threshold(DoubleFloat::from(1e-24));
    let result = double.reduce_batch(&double_double(&rank2)).unwrap();
    assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "double-double rank 2: {:?}", result.statuses[0]);
    for (lane, expected) in [(rank1, LaneRoute::Tier(106)), (rank2, LaneRoute::Tier(256))] {
        let exact = reference(&lane);
        for gate in [None, Some(1e-10)] {
            let mut workspace = madloop(lane.rank, gate);
            let result = workspace.reduce_batch(&batch(lane.values.clone())).unwrap();
            let route = workspace.last_lane_routes().unwrap()[0];
            assert_eq!(result.statuses[0], LaneStatus::Success, "rank {}, gate {gate:?}", lane.rank);
            assert_eq!(route, expected, "rank {}, gate {gate:?}", lane.rank);
            let error = error(result.lane(0).unwrap(), &exact);
            assert!(error <= 1e-13, "rank {}, gate {gate:?}: {route:?} {error:e} off", lane.rank);
        }
    }
}

#[test]
fn generic_triangles_stay_in_binary64() {
    for rank in 1..=3 {
        let lane = generic(rank);
        let exact = reference(&lane);
        let mut workspace = madloop(rank, None);
        let result = workspace.reduce_batch(&batch(lane.values.clone())).unwrap();
        assert_eq!(workspace.last_lane_routes().unwrap()[0], LaneRoute::Direct, "rank {rank}");
        let error = error(result.lane(0).unwrap(), &exact);
        assert!(error <= 1e-13, "rank {rank}: {error:e} off");
    }
}
