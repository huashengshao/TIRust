//! The benchmark's amplitudes (huashengshao/TIRust#26 review): the top-loop
//! gg -> Hgg pentagon, gg -> gggg hexagon and gg -> ggggg heptagon of the
//! binary64 stability study, each with every pinch of at least two
//! propagators at rank = its number of propagators, on the study's first 32
//! points (`fixtures/amplitude-*.json`, the inputs of amplitude captures
//! 026 to 028). `AmplitudeWorkspace` evaluates them as
//! `examples/amplitude_timing.rs` does, and the FNV-1a fingerprint of every
//! lane's array (best effort included), status and estimate, bit for bit, is
//! pinned, sharing on and off alike: with the loop-rotation test (the
//! default) capture 026's, measured again in 027 and 028; with
//! `RotationTest::Disabled` those of 027 and 028 (release
//! `tirust-bench-20261109`).
use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, KinematicsBatch, ReductionSettings, RotationTest,
    TensorReductionPlan, TensorWorkspaceOptions,
};
use serde_json::Value;
use symbolica::domains::float::Complex;

/// FNV-1a as in `examples/amplitude_timing.rs`.
fn fingerprint(results: &[BatchResult<f64>]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for byte in x.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for result in results {
        for lane in 0..result.rows() {
            match result.lane_with_estimate(lane) {
                Some(best) => best.values.iter().for_each(|z| {
                    eat(z.re.to_bits());
                    eat(z.im.to_bits());
                }),
                None => eat(u64::MAX),
            }
            for byte in format!("{:?}", result.statuses[lane]).bytes() {
                eat(u64::from(byte));
            }
            eat(result.estimated_errors.get(lane).copied().flatten().map_or(u64::MAX, f64::to_bits));
        }
    }
    hash
}

/// One plan and one batch per integral; every integral takes its members'
/// shifts and masses from each lane (the amplitude's common routing).
fn amplitude(spec: &str) -> (Vec<TensorReductionPlan>, Vec<KinematicsBatch<f64>>) {
    let spec: Value = serde_json::from_str(spec).unwrap();
    let lanes = spec["lanes"].as_array().unwrap();
    let mut plans = Vec::new();
    let mut batches = Vec::new();
    for integral in spec["integrals"].as_array().unwrap() {
        let members: Vec<usize> =
            integral["members"].as_array().unwrap().iter().map(|m| m.as_u64().unwrap() as usize).collect();
        let rank = integral["rank"].as_u64().unwrap() as u16;
        plans.push(TensorReductionPlan::new(ReductionSettings::new(members.len(), rank)).unwrap());
        let mut values = Vec::new();
        for lane in lanes {
            for &i in &members {
                values.extend(lane["shifts"][i].as_array().unwrap().iter().map(|x| Complex::new(x.as_f64().unwrap(), 0.0)));
            }
            values.extend(members.iter().map(|&i| Complex::new(lane["masses"][i].as_f64().unwrap(), 0.0)));
            values.push(Complex::new(lane["mu"].as_f64().unwrap(), 0.0));
        }
        batches.push(KinematicsBatch { rows: lanes.len(), components: 1, values });
    }
    (plans, batches)
}

fn check(label: &str, spec: &str, enabled: u64, disabled: u64) {
    let (plans, batches) = amplitude(spec);
    for (test, expected) in [(RotationTest::Enabled, enabled), (RotationTest::Disabled, disabled)] {
        let options = AmplitudeOptions {
            workspace: TensorWorkspaceOptions { rotation_test: test, ..Default::default() },
            ..Default::default()
        };
        let mut workspace = AmplitudeWorkspace::new_with_options(&plans, 0.0f64, options).unwrap();
        for share in [true, false] {
            workspace.set_subset_sharing(share);
            let results = workspace.reduce_batches(&batches).unwrap();
            assert_eq!(
                format!("{:016x}", fingerprint(&results)),
                format!("{expected:016x}"),
                "{label}, {test:?}, sharing {share}"
            );
        }
    }
}

#[test]
fn pentagon_amplitude_fingerprints_are_the_captures() {
    check("pentagon", include_str!("fixtures/amplitude-pentagon-r5.json"), 0x2e7a_57a2_a747_69b4, 0x5b55_fb4e_1b7d_4877);
}

#[test]
fn hexagon_amplitude_fingerprints_are_the_captures() {
    check("hexagon", include_str!("fixtures/amplitude-hexagon-r6.json"), 0x0007_2b2b_c3bf_86bc, 0x4fd7_39ec_d746_b99d);
}

#[test]
fn heptagon_amplitude_fingerprints_are_the_captures() {
    check("heptagon", include_str!("fixtures/amplitude-heptagon-r7.json"), 0xb23a_5b5d_c0e2_5cb2, 0x3925_2438_efef_2cac);
}
