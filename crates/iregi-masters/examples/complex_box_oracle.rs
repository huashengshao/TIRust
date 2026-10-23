//! A complex-mass box checked against a defining Feynman-parameter integral.
//!
//! Reproduce with `cargo run -p iregi-masters --example complex_box_oracle`.
//! The independent 18-point-per-axis, 50-decimal-digit quadrature and the
//! reference Fortran zero-result defect are recorded in
//! `validation/shifted-pentagon/contours.json` (pinched box 1). The tolerance
//! allows quadrature truncation; this example does not claim 50 correct digits.

use iregi_masters::{
    Complex, DoubleFloat, Float, MasterBackend, MasterEvaluator, NativeFloat, ScalarIntegral,
};
use std::str::FromStr;
use symbolica::{
    domains::{integer::Integer, rational::Rational},
    evaluate::EvaluationDomain,
};

fn decimal(text: &str) -> Rational {
    let (whole, fraction) = text.split_once('.').unwrap();
    Rational::new(
        Integer::from_str(&format!("{whole}{fraction}")).unwrap(),
        Integer::from_str(&format!("1{}", "0".repeat(fraction.len()))).unwrap(),
    )
}

fn check<T: NativeFloat>(prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let point: Vec<_> = [-1, -2, -2, -1, -1, -2, 1, 3, 4, 5, 1]
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            Complex::new(
                prototype.from_i64(value),
                prototype.from_i64(if (6..10).contains(&index) { -5 } else { 0 }),
            )
        })
        .collect();
    let expected = Complex::new(
        prototype.from_rational(&decimal(
            "-0.00139413754633931043939422440291610357134826883",
        )),
        prototype.from_rational(&decimal(
            "0.00400233718928142794565742273960032031088933789",
        )),
    );
    let tolerance = prototype.from_i64(10).pow(24).inv()
        + prototype.from_i64(32)
            * prototype
                .from_i64(2)
                .pow(u64::from(prototype.get_precision()))
                .inv();
    let mut evaluator = MasterEvaluator::new_with_backend(
        ScalarIntegral::D0,
        prototype.clone(),
        MasterBackend::SymbolicaBatch,
    )
    .unwrap();
    let batch = evaluator.evaluate_batch(&point, 1).unwrap();
    let actual = batch.lane(0).unwrap();
    let error = (actual.finite[0].re.clone() - &expected.re).norm()
        + (actual.finite[0].im.clone() - &expected.im).norm();
    assert!(
        error <= tolerance,
        "independent parameter-integral error: {error}"
    );
    for component in [actual.uv, actual.ir, actual.ir2] {
        assert!(component[0].re.is_zero() && component[0].im.is_zero());
    }
    println!(
        "{} bits: finite={}, independent integral error={error}",
        prototype.get_precision(),
        actual.finite[0]
    );
}

fn main() {
    check(0.0_f64);
    check(DoubleFloat::from(0.));
    check(Float::new(128));
    check(Float::new(256));
}
