//! Denominator identities that amplify rounding (huashengshao/TIRust#75).
//!
//! A subset whose Gram or coordinate system is singular is reduced by the
//! identity `sum_i c_i D_i = 1` of its affine relation, into the integrals that
//! the identity pinches. Their errors reach the result multiplied by the
//! identity's amplification `sum_j |c_j| |f_j|`: the relation constant's bound
//! over the constant itself. Next to a Coulomb triple `zeta` off threshold,
//! that constant is about `zeta m^2`, and nothing held the amplification
//! against anything. Binary64 served unit-power heptagons with two onia about
//! `1e-8 / zeta` off from zeta = 1 down to 1e-2, and up to 1e-2 below, by route
//! `Direct` through MadGraph's bridge options too. Double-double served them
//! exactly.
//!
//! A subset under an identity must now pass its gate with its estimate divided
//! by the amplification. Binary64 and double-double also refuse an identity
//! whose amplification alone costs more than `p - 34` bits.

use iregi_rust::{
    DirectReductionPlan, EmbedderOptions, KinematicsBatch, LaneRoute, LaneStatus, ReductionSettings, ScaleInput,
    TensorReductionPlan, TensorWorkspace,
};
use symbolica::domains::float::{Complex, Float, RealLike};

fn settings(rank: u16) -> ReductionSettings {
    let mut settings = ReductionSettings::new(7, rank);
    settings.scale_input = ScaleInput::MuSquared;
    settings
}

/// Issue #75's reproducer: a seven-point unit-power integral with two Coulomb
/// triples (two quark pairs at P/2 each, exact in binary64), the heavy line
/// 1 moved off threshold by a relative `zeta`. Native convention, invariants
/// from the shifts, mu^2 = 4.
fn two_onia(zeta: f64) -> Vec<Complex<f64>> {
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.5, 2.25, 1.375],
        [0.0625, 0.25, 3.125, 1.0],
        [-1.875, 0.0, 4.0, 0.625],
        [-3.8125, -0.875, 3.75, 1.0],
        [-5.75, -1.75, 3.5, 1.375],
        [-0.25, 1.0, -0.625, 6.75],
    ];
    let m = 2.78515625;
    let masses = [1.0, m * (1.0 + zeta), 0.0, m, 0.0, m, 0.0];
    let mut values: Vec<Complex<f64>> = shifts.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend(masses.iter().map(|&x| Complex::new(x, 0.0)));
    values.push(Complex::new(4.0, 0.0));
    values
}

fn batch<T: symbolica::domains::float::SingleFloat>(values: Vec<Complex<T>>) -> KinematicsBatch<T> {
    KinematicsBatch { rows: 1, components: 1, values }
}

/// The same binary64 inputs at 1024 bits.
fn reference(values: &[Complex<f64>], rank: u16) -> Vec<(f64, f64)> {
    let mut workspace = DirectReductionPlan::new(settings(rank)).unwrap().workspace(Float::new(1024)).unwrap();
    workspace.set_input_precision(53);
    workspace.set_stability_threshold(Float::with_val(1024, 2f64.powi(-(1024 - 120))));
    let lifted = values.iter().map(|z| Complex::new(Float::with_val(1024, z.re), Float::with_val(1024, z.im))).collect();
    let result = workspace.reduce_batch(&batch(lifted)).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success, "1024-bit reference");
    result.lane(0).unwrap().iter().map(|z| (z.re.to_f64(), z.im.to_f64())).collect()
}

/// Largest error over the tensor's largest reference entry. Both arrays must
/// be complete and finite.
fn error(values: &[Complex<f64>], reference: &[(f64, f64)]) -> f64 {
    assert_eq!(values.len(), reference.len(), "value and reference lengths");
    assert!(values.iter().all(|z| z.re.is_finite() && z.im.is_finite()), "non-finite value: {values:?}");
    let scale = reference.iter().map(|z| z.0.hypot(z.1)).fold(0.0, f64::max);
    values.iter().zip(reference).map(|(v, r)| (v.re - r.0).hypot(v.im - r.1)).fold(0.0, f64::max) / scale
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
fn heptagons_with_onia_off_threshold_leave_binary64() {
    // Binary64 served these 1.9e-5 to 3.6e-4 off (zeta 1e-4 and 1e-5), and the
    // lane 1.5e-7 off threshold 1.5e-2 to 2.9e-2 off on #71's branch (main
    // refused it by accident: a singular jet system).
    for zeta in [1e-4, 1e-5, 1.5319040112857056e-7] {
        for rank in [2, 3] {
            let values = two_onia(zeta);
            let plan = DirectReductionPlan::new(settings(rank)).unwrap();
            let mut binary64 = plan.workspace(0.0f64).unwrap();
            let result = binary64.reduce_batch(&batch(values.clone())).unwrap();
            match &result.statuses[0] {
                LaneStatus::Unstable(reason) => {
                    assert!(reason.contains("denominator identity"), "zeta {zeta:e}, rank {rank}: {reason}")
                }
                status => panic!("zeta {zeta:e}, rank {rank}: binary64 {status:?}"),
            }
            let exact = reference(&values, rank);
            for gate in [None, Some(1e-10)] {
                let mut workspace = madloop(rank, gate);
                let result = workspace.reduce_batch(&batch(values.clone())).unwrap();
                let route = workspace.last_lane_routes().unwrap()[0];
                assert_eq!(result.statuses[0], LaneStatus::Success, "zeta {zeta:e}, rank {rank}, gate {gate:?}");
                assert_eq!(route, LaneRoute::Tier(106), "zeta {zeta:e}, rank {rank}, gate {gate:?}");
                let error = error(result.lane(0).unwrap(), &exact);
                assert!(error <= 1e-13, "zeta {zeta:e}, rank {rank}, gate {gate:?}: {error:e} off");
            }
        }
    }
}

#[test]
fn heptagons_with_onia_far_from_threshold_stay_in_binary64() {
    // Far from threshold the identities amplify little, and binary64 keeps
    // these lanes, as before (1.1e-8 and 1.6e-8 off).
    for rank in [2, 3] {
        let values = two_onia(1.0);
        let exact = reference(&values, rank);
        let mut workspace = madloop(rank, None);
        let result = workspace.reduce_batch(&batch(values.clone())).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success, "rank {rank}");
        assert_eq!(workspace.last_lane_routes().unwrap()[0], LaneRoute::Direct, "rank {rank}");
        let error = error(result.lane(0).unwrap(), &exact);
        assert!(error <= 1e-7, "rank {rank}: {error:e} off");
    }
}
