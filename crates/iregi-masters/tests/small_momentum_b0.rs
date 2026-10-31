//! Small-momentum B0: binary64 and double-double finite parts against the
//! general expression at 1024 bits, for `|p²|` from 1e-14 to 1/8 of the
//! larger squared mass (equal, comparable, hierarchical, one massless and
//! complex masses; both signs of p²).
use oneloop::{NativeEvaluator, ScalarIntegral};
use symbolica::domains::float::{Complex, DoubleFloat, Float, RealLike};

const BITS: u32 = 1024;

fn reference(input: [(f64, f64); 4]) -> [Complex<Float>; 3] {
    let mut native = NativeEvaluator::<Float>::with_binary_precision(ScalarIntegral::B0, BITS).unwrap();
    native.set_small_momentum(false);
    let point: Vec<_> = input
        .iter()
        .map(|&(re, im)| Complex::new(Float::with_val(BITS, re), Float::with_val(BITS, im)))
        .collect();
    let mut out = vec![Complex::new(Float::new(BITS), Float::new(BITS)); 3];
    native.evaluate(&point, &mut out).unwrap();
    [out[0].clone(), out[1].clone(), out[2].clone()]
}

fn relative<T: RealLike>(value: &[Complex<T>], reference: &[Complex<Float>; 3]) -> f64 {
    let (r, i) = (reference[0].re.to_f64(), reference[0].im.to_f64());
    let scale = r.hypot(i).max(reference[1].re.to_f64().hypot(reference[1].im.to_f64()));
    let mut worst = 0.0f64;
    for slot in 0..3 {
        let (rr, ri) = (reference[slot].re.to_f64(), reference[slot].im.to_f64());
        worst = worst.max((value[slot].re.to_f64() - rr).hypot(value[slot].im.to_f64() - ri));
    }
    worst / scale
}

fn cases() -> Vec<[(f64, f64); 4]> {
    let mu = (1.0e6, 0.0);
    let masses: [((f64, f64), (f64, f64)); 7] = [
        ((29929.0, 0.0), (29929.0, 0.0)),
        ((29929.0, 0.0), (6464.16, 0.0)),
        ((1.0, 0.0), (0.7, 0.0)),
        ((1.0, 0.0), (1.0e-6, 0.0)),
        ((0.0, 0.0), (8317.0, 0.0)),
        ((8317.0, -2.1e3), (8317.0, -2.1e3)),
        ((29929.0, 0.0), (6464.16, -1.3e2)),
    ];
    let mut out = Vec::new();
    for (m0, m1) in masses {
        let big = m0.0.hypot(m0.1).max(m1.0.hypot(m1.1));
        for exponent in [-14, -12, -10, -8, -6, -4, -3, -2, -1] {
            for sign in [1.0, -1.0] {
                let p2 = sign * big * 10f64.powi(exponent) * 1.234_567;
                out.push([(p2, 0.0), m0, m1, mu]);
                out.push([(p2, 0.0), m1, m0, mu]);
            }
        }
    }
    out
}

#[test]
fn small_momentum_b0_is_accurate_in_binary64_and_double_double() {
    let mut double = NativeEvaluator::<f64>::new(ScalarIntegral::B0).unwrap();
    let mut dd = NativeEvaluator::<DoubleFloat>::new(ScalarIntegral::B0).unwrap();
    let mut general = NativeEvaluator::<f64>::new(ScalarIntegral::B0).unwrap();
    general.set_small_momentum(false);
    let (mut worst64, mut worst_dd, mut worst_general) = (0.0f64, 0.0f64, 0.0f64);
    for case in cases() {
        let exact = reference(case);
        let point: Vec<_> = case.iter().map(|&(re, im)| Complex::new(re, im)).collect();
        let mut out = vec![Complex::new(0.0, 0.0); 3];
        double.evaluate(&point, &mut out).unwrap();
        let e64 = relative(&out, &exact);
        general.evaluate(&point, &mut out).unwrap();
        let eg = relative(&out, &exact);
        let point: Vec<_> = case
            .iter()
            .map(|&(re, im)| Complex::new(DoubleFloat::from(re), DoubleFloat::from(im)))
            .collect();
        let mut out = vec![Complex::new(DoubleFloat::from(0.0), DoubleFloat::from(0.0)); 3];
        dd.evaluate(&point, &mut out).unwrap();
        let edd = relative(&out, &exact);
        if std::env::var("VERBOSE").is_ok() {
            println!("{case:?}: binary64 {e64:.2e} general {eg:.2e} double-double {edd:.2e}");
        }
        assert!(e64 < 4e-15, "{case:?}: binary64 {e64:e}");
        assert!(edd < 1e-29, "{case:?}: double-double {edd:e}");
        worst64 = worst64.max(e64);
        worst_dd = worst_dd.max(edd);
        worst_general = worst_general.max(eg);
    }
    println!("worst binary64 {worst64:e}, double-double {worst_dd:e}; general expression {worst_general:e}");
    // The general expression is what the small-momentum path repairs.
    assert!(worst_general > 1e-8);
}
