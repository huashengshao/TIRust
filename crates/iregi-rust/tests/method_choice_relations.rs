//! Boxes and triangles whose external points have an affine relation, under
//! the small-Gram methods (huashengshao/TIRust#79).
//!
//! Where the Gram gate refuses such a subset, the direct engine reduces it by
//! the relation, and refuses the lane where the relation does not hold to the
//! inputs' rounding (huashengshao/TIRust#76). The small-Gram expansions and
//! the alternative PV reduction, which the a priori method choice takes, do
//! not see that geometry. They served these refused lanes far off as
//! `Success` (up to 9e13 off), from the method choice as pyAmpliCol runs it
//! and from the facade with its default options (route `MethodChoice`).
//! Such a subset now takes the gate and the relation search under these
//! methods too: the method choice refuses these lanes, and the facade serves
//! them from its precision tiers.

use iregi_rust::direct::EXPANSION_TOLERANCE;
use iregi_rust::{
    Convention, DirectReductionPlan, EmbedderOptions, InvariantInput, KinematicsBatch, LaneRoute, LaneStatus,
    ReductionSettings, ScaleInput, TensorReductionPlan, TensorWeights, TensorWorkspace,
};
use symbolica::domains::float::{Complex, Float, RealLike};

fn settings(rank: u16) -> ReductionSettings {
    let mut settings = ReductionSettings::new(3, rank);
    settings.convention = Convention::MadLoop;
    settings.tensor_weights = TensorWeights::Components;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    settings
}

/// Binary64 bit patterns of a lane: shifts (three `[E, x, y, z]`), packed
/// invariants `(s01, s02, s12)`, squared masses and mu^2.
fn lane(bits: &[u64; 19]) -> Vec<Complex<f64>> {
    bits.iter().map(|&b| Complex::new(f64::from_bits(b), 0.0)).collect()
}

/// Issue #79's reproducer: two shifts that coincide to about 1e-10 of their
/// size, next to lightlike legs. #76 refuses its denominator identity.
fn coincident() -> Vec<Complex<f64>> {
    lane(&[
        0, 0, 0, 0,
        0x403f8b4a733541c1, 0xc01fd0a472a1846c, 0xc02a12290005ef47, 0xc03b99fe69de4e3a,
        0x403f8b4a7323063f, 0xc01fd0a4728f41a3, 0xc02a1228fff6d82b, 0xc03b99fe69ce5906,
        0, 0xbd58000000000000, 0,
        0x40be2d1fe456191f, 0x40c3e524baa1d41b, 0x40c0ede134ff4eba,
        0x40b93cc89cb252cf,
    ])
}

/// A collinear triple rounded to binary64 (q2 - o = alpha (q1 - o), alpha
/// not dyadic, offset origin o) with a relation constant 1e-10 of its size:
/// the relation is homogeneous only beyond the inputs' rounding, and binary64
/// finds the Gram matrix exactly singular. The a priori method choice took
/// the triangle to the expansions before its Gram system.
fn nearly_homogeneous() -> Vec<Complex<f64>> {
    lane(&[
        0x40327e6fbf23688a, 0xbffe89eb9cc06850, 0x3ff16610840c5190, 0x3ffa30fb3247d060,
        0x40349000e0c2ea11, 0x4008e78f2b1b85c0, 0x402304f841b2f932, 0xc029c0134b11e026,
        0x403398d030a549b5, 0x3fe89be32acc92b8, 0x4016503f97f79c81, 0xc01867a511042ac1,
        0xc072e798d1ff00d6, 0xc05580165291d5b5, 0xc05079d6a5c32f66,
        0x408b8c005fc3a9c6, 0x40793a0ea13406f1, 0x4085efc97f1cf090,
        0x4081d02bb23abdf9,
    ])
}

fn batch<T: symbolica::domains::float::SingleFloat>(values: Vec<Complex<T>>) -> KinematicsBatch<T> {
    KinematicsBatch { rows: 1, components: 1, values }
}

/// The same binary64 inputs at 1024 bits.
fn reference(values: &[Complex<f64>], rank: u16) -> Vec<(f64, f64)> {
    let bits = 1024;
    let mut workspace = DirectReductionPlan::new(settings(rank)).unwrap().workspace(Float::new(bits)).unwrap();
    workspace.set_input_precision(53);
    workspace.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 120))));
    let lifted = values.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect();
    let result = workspace.reduce_batch(&batch(lifted)).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success, "1024-bit reference at rank {rank}");
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

#[test]
fn relation_points_leave_the_small_gram_methods() {
    for (name, values) in [("coincident", coincident()), ("nearly homogeneous", nearly_homogeneous())] {
        for rank in 1..=4u16 {
            let label = format!("{name}, rank {rank}");
            let plan = DirectReductionPlan::new(settings(rank)).unwrap();
            // The method choice as pyAmpliCol's TIR backend runs it.
            for first in [None, Some(EXPANSION_TOLERANCE)] {
                let mut workspace = plan.workspace(0.0f64).unwrap();
                workspace.set_stability_threshold(1e-15);
                workspace.enable_estimated_method_choice(12, 1e-15);
                workspace.set_alternative_first(first);
                let result = workspace.reduce_batch(&batch(values.clone())).unwrap();
                assert!(
                    matches!(result.statuses[0], LaneStatus::Unstable(_)),
                    "{label}, method choice (alternative first {first:?}): {:?}",
                    result.statuses[0]
                );
            }
            let tensor = TensorReductionPlan::new(settings(rank)).unwrap();
            // The facade's default options: refused, not served by the method choice.
            let mut facade = TensorWorkspace::new_embedded(&tensor, 0.0f64, EmbedderOptions::default()).unwrap();
            let result = facade.reduce_batch(&batch(values.clone())).unwrap();
            assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "{label}, facade: {:?}", result.statuses[0]);
            assert_eq!(facade.last_lane_routes().unwrap()[0], LaneRoute::Refused, "{label}, facade route");
            // With the precision tiers: from a tier, within 1e-12 of 1024 bits.
            let options = EmbedderOptions { precision_tiers: true, ..EmbedderOptions::default() };
            let mut facade = TensorWorkspace::new_embedded(&tensor, 0.0f64, options).unwrap();
            let result = facade.reduce_batch(&batch(values.clone())).unwrap();
            assert_eq!(result.statuses[0], LaneStatus::Success, "{label}, facade with tiers");
            let route = facade.last_lane_routes().unwrap()[0];
            assert!(matches!(route, LaneRoute::Tier(_)), "{label}, facade with tiers: route {route:?}");
            let e = error(result.lane(0).unwrap(), &reference(&values, rank));
            assert!(e <= 1e-12, "{label}, facade with tiers: {route:?} {e:e} off 1024 bits");
        }
    }
}
