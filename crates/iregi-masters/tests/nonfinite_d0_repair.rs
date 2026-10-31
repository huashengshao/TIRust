//! oneloopmaster's binary64 D0 kernels returned NaN for a top-loop box with
//! two opposite numerically lightlike legs (p^2 = 1.7e-10 and -2.6e-10 from
//! binary64 rounding, m^2 = 29929, s ~ 5e5); the evaluator now retries such a
//! point with invariants within 2^-30 of the scale set to zero (PATCHES.md).
use oneloop::{NativeEvaluator, ScalarIntegral};
use symbolica::domains::float::{Complex, Float, RealLike};

const POINT: [f64; 11] = [
    1.6916601452976465e-10,
    15624.999999999956,
    -2.6193447411060333e-10,
    1e6,
    478321.0969658119,
    517226.6469296405,
    29929.0,
    29929.0,
    29929.0,
    29929.0,
    1e6,
];

#[test]
fn binary64_d0_with_tiny_opposite_invariants_is_finite_and_accurate() {
    let mut reference = NativeEvaluator::<Float>::with_binary_precision(ScalarIntegral::D0, 256).unwrap();
    let point: Vec<_> = POINT.iter().map(|&x| Complex::new(Float::with_val(256, x), Float::with_val(256, 0.0))).collect();
    let mut exact = vec![Complex::new(Float::new(256), Float::new(256)); 3];
    reference.evaluate(&point, &mut exact).unwrap();
    let finite = (exact[0].re.to_f64(), exact[0].im.to_f64());
    for handwritten in [true, false] {
        let mut native = NativeEvaluator::<f64>::new(ScalarIntegral::D0).unwrap();
        native.set_handwritten(handwritten);
        let point: Vec<_> = POINT.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut out = vec![Complex::new(0.0, 0.0); 3];
        native.evaluate(&point, &mut out).unwrap();
        let error = (out[0].re - finite.0).hypot(out[0].im - finite.1) / finite.0.hypot(finite.1);
        assert!(error <= 1e-12, "handwritten {handwritten}: {:?} vs {finite:?} (relative {error:e})", out[0]);
    }
}
