//! The direct engine's lightlike bubble rule is relative to the working
//! precision: a bubble counts as lightlike when |p^2| is at most 2^-(p - 13)
//! of its scale (2^-40 in binary64, as before; 2^-93 in double-double,
//! 2^-243 at 256 bits). A near-lightlike bubble at 256 bits is therefore
//! evaluated as the function of its p^2 (its tensor coefficients move
//! linearly with p^2), while binary64 still takes the lightlike closed form.
//!
//! Inputs declared as binary64 (`set_input_precision(53)`, as the facade's
//! precision tiers do) widen the rule to 2^-40 in binary64 and double-double
//! only; MPFR workspaces keep their working precision's tolerance.
use iregi_rust::{
    DirectReductionPlan, EmbedderOptions, InvariantInput, KinematicsBatch, LaneStatus, ReductionSettings, ScaleInput,
    TensorReductionPlan, TensorWeights, TensorWorkspace,
};
use symbolica::domains::{
    float::{Complex, Float, RealLike},
    integer::Integer,
    rational::Rational,
};

/// Rank-1 bubble lane: shifts 0 and (1, 0, 0, 1), supplied p^2, masses 1 and
/// 4, mu = 3, in the carrier of `prototype`.
fn bubble<T: iregi_masters::NativeFloat>(prototype: &T, p2: &Rational) -> KinematicsBatch<T> {
    let real = |x: T| Complex::new(x, prototype.zero());
    let int = |x: i64| prototype.from_rational(&Rational::from(Integer::from(x)));
    let values = vec![
        real(int(0)), real(int(0)), real(int(0)), real(int(0)),
        real(int(1)), real(int(0)), real(int(0)), real(int(1)),
        real(prototype.from_rational(p2)),
        real(int(1)), real(int(4)),
        real(int(3)),
    ];
    KinematicsBatch { rows: 1, components: 1, values }
}

fn settings() -> ReductionSettings {
    let mut settings = ReductionSettings::new(2, 1);
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings
}

fn power_of_two(exponent: i64) -> Rational {
    if exponent >= 0 {
        Rational::from(Integer::from(2).pow(exponent as u64))
    } else {
        Rational::from((Integer::from(1), Integer::from(2).pow((-exponent) as u64)))
    }
}

fn mpfr(p2: &Rational) -> Vec<Complex<Float>> {
    let prototype = Float::new(256);
    let mut workspace = DirectReductionPlan::new(settings()).unwrap().workspace(prototype.clone()).unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let result = workspace.reduce_batch(&bubble(&prototype, p2)).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    result.lane(0).unwrap().to_vec()
}

#[test]
fn near_lightlike_bubbles_follow_p2_at_high_precision() {
    // |p^2| = 2^-60 of the bubble's scale (m1^2 = 4): lightlike in binary64,
    // not at 256 bits.
    let epsilon = power_of_two(-58);
    let zero = mpfr(&Rational::from(Integer::from(0)));
    let one = mpfr(&epsilon);
    let two = mpfr(&(&epsilon * &Rational::from(Integer::from(2))));
    // The rank-1 moments p^mu B1 (p fixed) move linearly with p^2 through B1
    // (second order 2^-58 smaller); the lightlike closed form of B1 would not
    // move at all (only the scalar master B0 sees p^2 there).
    let (mut first, mut curvature) = (0.0f64, 0.0f64);
    for ((z, a), b) in zero.iter().zip(&one).zip(&two).skip(4) {
        for (z, a, b) in [(&z.re, &a.re, &b.re), (&z.im, &a.im, &b.im)] {
            let d1 = a.clone() - z;
            let d2 = b.clone() - z;
            let second = d2 - &(d1.clone() * &Float::with_val(256, 2));
            first = first.max(d1.to_f64().abs());
            curvature = curvature.max(second.to_f64().abs());
        }
    }
    println!("256 bits: first-order move {first:e}, second-order {curvature:e}");
    assert!(first > 0.0 && first < 1e-15, "{first:e}");
    assert!(curvature <= 1e-6 * first, "{curvature:e} vs {first:e}");
    // Binary64: the lightlike closed form, bit for bit the p^2 = 0 lane.
    let reduce64 = |p2: &Rational| {
        let mut workspace = DirectReductionPlan::new(settings()).unwrap().workspace(0.0f64).unwrap();
        let result = workspace.reduce_batch(&bubble(&0.0f64, p2)).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        result.lane(0).unwrap().iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect::<Vec<_>>()
    };
    assert_eq!(reduce64(&epsilon), reduce64(&Rational::from(Integer::from(0))));
}

/// The lane of huashengshao/TIRust#85 (from the review of #84): a triangle
/// whose shifts q1 (lightlike) and q2 coincide to about 6e-13 of their size,
/// with heavy unequal masses. Shifts, supplied invariants (s01, s02, s12; s02
/// is q2^2 exactly), squared masses and mu^2, as binary64 bit patterns.
const COINCIDENT: [u64; 19] = [
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x40275808a1eb79c8, 0xc018ff5a24384f6c, 0xc0185cd6941aed36, 0xc01f0162319c2fdc,
    0x40275808a1eb8919, 0xc018ff5a24384291, 0xc0185cd6941b0873, 0xc01f0162319c3573,
    0x0000000000000000, 0x3ddc940000000000, 0x0000000000000000,
    0x40c29d91f4195764, 0x40c245fae5813f0e, 0x40c0e5911fcf8534,
    0x40c03d96d59b5282,
];

fn coincident_settings(rank: u16) -> ReductionSettings {
    let mut settings = ReductionSettings::new(3, rank);
    settings.tensor_weights = TensorWeights::Components;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    settings
}

/// The lane at 1024 bits, its inputs declared as binary64 or not.
fn coincident_1024(rank: u16, declared: bool) -> Vec<Complex<Float>> {
    let bits = 1024;
    let mut workspace = DirectReductionPlan::new(coincident_settings(rank)).unwrap().workspace(Float::new(bits)).unwrap();
    if declared {
        workspace.set_input_precision(53);
    }
    workspace.set_stability_threshold(Float::with_val(bits, 2f64.powi(-904)));
    let values = COINCIDENT
        .iter()
        .map(|&x| Complex::new(Float::with_val(bits, f64::from_bits(x)), Float::with_val(bits, 0.0)))
        .collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success, "rank {rank}, declared {declared}");
    result.lane(0).unwrap().to_vec()
}

#[test]
fn declared_binary64_inputs_keep_the_working_tolerance_in_mpfr() {
    // q2^2 = 9.2e-11 is 1e-14 of the masses, inside binary64's 2^-40. With
    // that tolerance at 1024 bits the bubble [0, 2] was taken at p^2 = 0, and
    // the triangle's Gram system (determinant -(q2^2 - s12)^2 / 4 with
    // q1^2 = 0) amplified the error: 8.3e-3 off at rank 2, 1.1e10 at rank 3,
    // 2.1e21 at rank 4, and MadGraph's facade served those values from its
    // 256-bit tier.
    for rank in 1..=4u16 {
        let exact = coincident_1024(rank, false);
        let declared = coincident_1024(rank, true);
        let scale = exact.iter().map(|z| z.re.to_f64().hypot(z.im.to_f64())).fold(0.0, f64::max);
        let difference = exact
            .iter()
            .zip(&declared)
            .map(|(a, b)| (a.re.clone() - &b.re).to_f64().hypot((a.im.clone() - &b.im).to_f64()))
            .fold(0.0, f64::max);
        assert!(difference <= 1e-200 * scale, "rank {rank}: declared {:e} off", difference / scale);
        if rank <= 2 {
            // MadGraph's facade options. At ranks 3 and 4 its 256-bit tier
            // still misses on this lane (4e-12, 0.97): a small Gram
            // determinant against the masses, which its gate does not see
            // (#65's item 2), not this rule.
            let options = EmbedderOptions {
                method_choice: false,
                precision_tiers: true,
                tier_margin_bits: 0,
                double_double_headroom_bits: 26,
                ..EmbedderOptions::default()
            };
            let plan = TensorReductionPlan::new(coincident_settings(rank)).unwrap();
            let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
            let values = COINCIDENT.iter().map(|&x| Complex::new(f64::from_bits(x), 0.0)).collect();
            let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
            assert_eq!(result.statuses[0], LaneStatus::Success, "rank {rank}");
            let error = result
                .lane(0)
                .unwrap()
                .iter()
                .zip(&exact)
                .map(|(v, r)| (v.re - r.re.to_f64()).hypot(v.im - r.im.to_f64()))
                .fold(0.0, f64::max)
                / scale;
            assert!(error <= 1e-15, "rank {rank}: facade {error:e} off");
        }
    }
}
