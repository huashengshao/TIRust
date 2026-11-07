//! The direct engine's lightlike bubble rule is relative to the working
//! precision: a bubble counts as lightlike when |p^2| is at most 2^-(p - 13)
//! of its scale (2^-40 in binary64, as before; 2^-93 in double-double,
//! 2^-243 at 256 bits). A near-lightlike bubble at 256 bits is therefore
//! evaluated as the function of its p^2 (its tensor coefficients move
//! linearly with p^2), while binary64 still takes the lightlike closed form.
use iregi_rust::{DirectReductionPlan, InvariantInput, KinematicsBatch, LaneStatus, ReductionSettings};
use symbolica::domains::{
    float::{Complex, Float, Real, RealLike, SingleFloat},
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
