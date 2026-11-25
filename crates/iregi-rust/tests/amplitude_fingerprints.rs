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
//!
//! Triangles have had the rotation test since the S-wave quarkonium
//! triangles (`binary64_gate_refuses_lightlike_forward_quarkonium_triangles`
//! in `tests/direct_engine.rs`), so with the test their lanes now carry an
//! estimate, which changed the default's hashes. In the heptagon amplitude
//! the test refuses 4 of the 1,120 triangle lanes, which the double-double
//! tier now serves: against 256 bits they were off by E/A 7.9e-11 to 2.1e-9
//! (beyond the request 2^-34, unflagged) and are now within 1.3e-16.
//!
//! Since huashengshao/TIRust#62 the triangle integrals' Gram gate also holds
//! them to their PV error growth near a vanishing Källén function, and to a
//! 34-bit floor of their Gram system in binary64. Of the 2,080 rank-3
//! triangle lanes of the three amplitudes, 279 changed (pentagon 19,
//! hexagon 74, heptagon 186), 167 of them now refused by the binary64 gate
//! and served by the double-double tier. Against 256 bits they were within
//! 1.7e-10 (30 above 1e-12, 3 above 1e-10, unflagged) and are now within
//! 1.9e-15 (the change touches 3-point plans only). So no amplitude keeps
//! capture 026's hash with the triangles' estimates absent.
//!
//! Platform: the hashes cover every bit, so they depend on the C library's
//! `log`, `exp`, `atan2` and other libm functions behind the scalar masters.
//! They were captured on x86_64 Linux with glibc 2.42's libm and Rust
//! 1.97.1; the captures' release builds (`+fma`, `native-f64-abc`) and this
//! suite's debug build give the same hashes there. Another libm (macOS,
//! musl) or architecture (aarch64) may round a last bit differently, so a
//! first mismatch there is not by itself a regression. To regenerate the
//! hashes on such a platform, check first that the other tests pass. Then run
//! `examples/amplitude_timing.rs` on each fixture with `FINAL` on stdin, once
//! as is and once with `TIRUST_ROTATION_TEST=disabled`, for example from
//! `crates/iregi-rust`:
//! `echo FINAL | cargo run --example amplitude_timing --
//! tests/fixtures/amplitude-pentagon-r5.json`. Its `fingerprint_on` and
//! `fingerprint_off` (sharing on and off) must agree, and that value goes
//! here. The captures' hash with the test (the triangles' estimates absent)
//! is what the same run prints with a build from before the triangles' test
//! (main 854b79d).
use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, KinematicsBatch, ReductionSettings, RotationTest,
    TensorReductionPlan, TensorWorkspaceOptions,
};
use serde_json::Value;
use symbolica::domains::float::Complex;

/// FNV-1a as in `examples/amplitude_timing.rs`; the estimates of the
/// integrals `masked` marks are hashed as absent.
fn fingerprint(results: &[BatchResult<f64>], masked: &[bool]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for byte in x.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for (result, &masked) in results.iter().zip(masked) {
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
            let estimate = if masked { None } else { result.estimated_errors.get(lane).copied().flatten() };
            eat(estimate.map_or(u64::MAX, f64::to_bits));
        }
    }
    hash
}

/// One plan and one batch per integral; every integral takes its members'
/// shifts and masses from each lane (the amplitude's common routing).
fn amplitude(spec: &str) -> (Vec<TensorReductionPlan>, Vec<KinematicsBatch<f64>>, Vec<bool>) {
    let spec: Value = serde_json::from_str(spec).unwrap();
    let lanes = spec["lanes"].as_array().unwrap();
    let mut plans = Vec::new();
    let mut batches = Vec::new();
    let mut triangles = Vec::new();
    for integral in spec["integrals"].as_array().unwrap() {
        let members: Vec<usize> =
            integral["members"].as_array().unwrap().iter().map(|m| m.as_u64().unwrap() as usize).collect();
        let rank = integral["rank"].as_u64().unwrap() as u16;
        triangles.push(members.len() == 3);
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
    (plans, batches, triangles)
}

/// `enabled` and `disabled`: the hashes with the test and without it;
/// `capture`: capture 026's with the test, which the hash with the
/// triangles' estimates as absent must equal (`None`: some triangle arrays
/// changed since).
fn check(label: &str, spec: &str, enabled: u64, capture: Option<u64>, disabled: u64) {
    let (plans, batches, triangles) = amplitude(spec);
    let none = vec![false; plans.len()];
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
                format!("{:016x}", fingerprint(&results, &none)),
                format!("{expected:016x}"),
                "{label}, {test:?}, sharing {share}"
            );
            if let (RotationTest::Enabled, Some(capture)) = (test, capture) {
                assert_eq!(
                    format!("{:016x}", fingerprint(&results, &triangles)),
                    format!("{capture:016x}"),
                    "{label}, triangles' estimates absent, sharing {share}"
                );
            }
        }
    }
}

#[test]
fn pentagon_amplitude_fingerprints_are_the_captures() {
    let spec = include_str!("fixtures/amplitude-pentagon-r5.json");
    check("pentagon", spec, 0x7ccb_ea34_219f_f68e, None, 0x3bc0_1008_ebcf_0412);
}

#[test]
fn hexagon_amplitude_fingerprints_are_the_captures() {
    let spec = include_str!("fixtures/amplitude-hexagon-r6.json");
    check("hexagon", spec, 0xb98a_8e86_3b18_1bc5, None, 0x5b28_9dea_381e_5c94);
}

#[test]
fn heptagon_amplitude_fingerprints_are_the_captures() {
    // Capture 026's 0xb23a_5b5d_c0e2_5cb2 no longer applies: triangle lanes
    // are now rescued (see the module notes).
    let spec = include_str!("fixtures/amplitude-heptagon-r7.json");
    check("heptagon", spec, 0xb874_9985_867b_c1d2, None, 0xe55a_01cc_309b_547e);
}
