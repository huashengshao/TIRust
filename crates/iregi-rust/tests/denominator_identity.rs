//! The denominator identity of a triangle whose external points are affinely
//! dependent (huashengshao/TIRust#73), with supplied invariants as MadLoop
//! supplies them.
//!
//! Where the Gram gate refuses a subset, the direct engine reduces it by the
//! affine relation of its shifts: for a constant relation, by the identity
//! `sum_i c_i D_i = 1`. On points dependent only to `delta` of their size,
//! that identity is off by `2 (l + q_0).r` with `r = sum_i c_i q_i`. That is
//! up to `2 |r| |p|` relative, about `delta / eps` for a constant that cancels
//! to `eps` of its size, at every precision. Binary64 and double-double
//! served such lanes up to 1e-2 off as a success, and so did the facade
//! through MadGraph's bridge options. The identity must now hold to the
//! inputs' rounding: either the shifts are dependent there, or `2 |r| |p|`
//! is within it.

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
    label: String,
    rank: u16,
    values: Vec<Complex<f64>>,
}

fn lane(label: String, rank: u16, shifts: &[f64], invariants: [f64; 3], masses: [f64; 3], mu2: f64) -> Lane {
    let mut values: Vec<Complex<f64>> = shifts.iter().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend(invariants.map(|x| Complex::new(x, 0.0)));
    values.extend(masses.map(|m| Complex::new(m, 0.0)));
    values.push(Complex::new(mu2, 0.0));
    Lane { label, rank, values }
}

fn square(d: [f64; 4]) -> f64 {
    d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]
}

/// Issue #73's reproducer: q0 = 0, q1 = p and q2 = 2 p + delta |p| u (u a unit
/// vector off the line), with m0^2 = 0.7 s, m1^2 = 1.3 s and m2^2 such that
/// f2 = 2 f1 (1 - eps), where f_j = s_0j - m_j^2 + m0^2 and s = |p|^2
/// (Euclidean). The invariants are those of the binary64 shifts, and
/// mu^2 = s. The identity is about delta / eps off.
fn nearly_collinear(rank: u16, delta: f64, eps: f64) -> Lane {
    let p = [5.0, 1.0, 2.0, 0.5];
    let u = [0.3, 0.5, -0.7, 0.2];
    let un = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    let pn = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q2: [f64; 4] = std::array::from_fn(|a| 2.0 * p[a] + delta * pn * u[a] / un);
    let points = [[0.0; 4], p, q2];
    let inv = |i: usize, j: usize| square(std::array::from_fn(|k| points[i][k] - points[j][k]));
    let shifts: Vec<f64> = points.iter().flatten().copied().collect();
    let s = pn * pn;
    let (m0, m1) = (0.7 * s, 1.3 * s);
    let f1 = inv(0, 1) - m1 + m0;
    let m2 = inv(0, 2) + m0 - 2.0 * f1 * (1.0 - eps);
    let label = format!("delta {delta:e}, eps {eps:e}, rank {rank}");
    lane(label, rank, &shifts, [inv(0, 1), inv(0, 2), inv(1, 2)], [m0, m1, m2], s)
}

/// Exactly collinear points q2 = 2 q1 (exact in binary64, so the invariants
/// put the Källén function at exactly zero) with a constant of order its
/// size: the relation holds to the inputs' rounding.
fn collinear(rank: u16) -> Lane {
    let p = [5.0, 1.0, 2.0, 0.5];
    let points = [[0.0; 4], p, p.map(|x| 2.0 * x)];
    let inv = |i: usize, j: usize| square(std::array::from_fn(|k| points[i][k] - points[j][k]));
    let shifts: Vec<f64> = points.iter().flatten().copied().collect();
    let s = p.iter().map(|v| v * v).sum::<f64>();
    let label = format!("collinear, rank {rank}");
    lane(label, rank, &shifts, [inv(0, 1), inv(0, 2), inv(1, 2)], [0.7 * s, 1.3 * s, 2.9 * s], s)
}

/// MadGraph's soft-collinear stress point of e+e- > t t~ a at its TIRust pin
/// (a photon of E = 1e-4 at 1e-6 rad to the beam, sqrt(s) = 1 TeV): calls 9282
/// (masses (M_Z^2, 0, 0)) and 9375 (masses (0, M_W^2, M_W^2)) at rank 2. The
/// shifts are collinear to 1.4e-13 of their size, and MadLoop snaps the three
/// invariants to exactly zero, so the Gram system is singular at every
/// precision. The identity served both in binary64, its UV pole's
/// off-diagonal components 4.0e-5 and 5.2e-5 of the slot off zero.
fn madloop_soft_collinear(masses: [f64; 3]) -> Lane {
    let shifts = [
        0.0, 0.0, 0.0, 5.684341886080802e-14,
        500.0000000000002, -4.441719503878831e-14, 0.0, 500.0000000000002,
        499.99990000000025, -1.000444171950221e-10, 0.0, 499.99990000000025,
    ];
    lane(format!("soft-collinear, masses {masses:?}"), 2, &shifts, [0.0; 3], masses, 1e6)
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

/// The binary64 and double-double direct workspaces at their default
/// threshold and at 1e-10: each status.
fn fixed_precision_statuses(lane: &Lane) -> Vec<(String, LaneStatus)> {
    let plan = DirectReductionPlan::new(settings(lane.rank)).unwrap();
    let mut statuses = Vec::new();
    for gate in [None, Some(1e-10)] {
        let mut binary64 = plan.workspace(0.0f64).unwrap();
        if let Some(gate) = gate {
            binary64.set_stability_threshold(gate);
        }
        let result = binary64.reduce_batch(&batch(lane.values.clone())).unwrap();
        statuses.push((format!("binary64, threshold {gate:?}"), result.statuses[0].clone()));
        let mut double = plan.workspace(DoubleFloat::from(0.0)).unwrap();
        if let Some(gate) = gate {
            double.set_stability_threshold(DoubleFloat::from(gate));
        }
        let result = double.reduce_batch(&double_double(lane)).unwrap();
        statuses.push((format!("double-double, threshold {gate:?}"), result.statuses[0].clone()));
    }
    statuses
}

/// Each fixed-precision workspace refuses the lane, naming the identity.
fn assert_identity_refused(lane: &Lane) {
    for (workspace, status) in fixed_precision_statuses(lane) {
        match status {
            LaneStatus::Unstable(reason) => {
                assert!(reason.contains("denominator identity"), "{}, {workspace}: {reason}", lane.label)
            }
            status => panic!("{}, {workspace}: {status:?}", lane.label),
        }
    }
}

#[test]
fn nearly_collinear_triangles_come_from_a_tier_that_resolves_them() {
    // Binary64 and double-double served these 4.3e-11 to 9.7e-3 off. The Gram
    // system resolves the points at 256 bits up to rank 2, and at 512 bits at
    // rank 3.
    for (delta, eps) in [(1e-13, 1e-2), (1e-13, 1e-6), (6e-13, 1e-9)] {
        for rank in 1..=3 {
            let lane = nearly_collinear(rank, delta, eps);
            assert_identity_refused(&lane);
            let exact = reference(&lane);
            let expected = if rank < 3 { LaneRoute::Tier(256) } else { LaneRoute::Tier(512) };
            for gate in [None, Some(1e-10)] {
                let mut workspace = madloop(rank, gate);
                let result = workspace.reduce_batch(&batch(lane.values.clone())).unwrap();
                let route = workspace.last_lane_routes().unwrap()[0];
                assert_eq!(result.statuses[0], LaneStatus::Success, "{}, gate {gate:?}", lane.label);
                assert_eq!(route, expected, "{}, gate {gate:?}", lane.label);
                let error = error(result.lane(0).unwrap(), &exact);
                assert!(error <= 1e-13, "{}, gate {gate:?}: {route:?} {error:e} off", lane.label);
            }
        }
    }
}

#[test]
fn exactly_collinear_triangles_keep_their_identity() {
    for rank in 1..=3 {
        let lane = collinear(rank);
        let exact = reference(&lane);
        for (workspace, status) in fixed_precision_statuses(&lane) {
            assert_eq!(status, LaneStatus::Success, "{}, {workspace}", lane.label);
        }
        for gate in [None, Some(1e-10)] {
            let mut workspace = madloop(rank, gate);
            let result = workspace.reduce_batch(&batch(lane.values.clone())).unwrap();
            let route = workspace.last_lane_routes().unwrap()[0];
            assert_eq!(result.statuses[0], LaneStatus::Success, "{}, gate {gate:?}", lane.label);
            assert_eq!(route, LaneRoute::Direct, "{}, gate {gate:?}", lane.label);
            let error = error(result.lane(0).unwrap(), &exact);
            assert!(error <= 1e-13, "{}, gate {gate:?}: {error:e} off", lane.label);
        }
    }
}

#[test]
fn snapped_soft_collinear_triangles_are_refused() {
    // MadGraph prefers the refusal to projecting the shifts onto the relation
    // the snapped invariants assert (issue #73).
    for masses in [[8315.251344, 0.0, 0.0], [0.0, 6467.215560999999, 6467.215560999999]] {
        let lane = madloop_soft_collinear(masses);
        assert_identity_refused(&lane);
        for gate in [None, Some(1e-10)] {
            let mut workspace = madloop(lane.rank, gate);
            let result = workspace.reduce_batch(&batch(lane.values.clone())).unwrap();
            let route = workspace.last_lane_routes().unwrap()[0];
            assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "{}, gate {gate:?}: {:?}", lane.label, result.statuses[0]);
            assert_eq!(route, LaneRoute::Refused, "{}, gate {gate:?}", lane.label);
        }
    }
}
