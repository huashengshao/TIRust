//! Fused double-double kernels without fused multiply-add.
//!
//! xprec's double-double product calls `f64::mul_add` three times. A build
//! without the FMA target feature (baseline x86-64) runs each as two
//! out-of-line calls (`compiler_builtins`' runtime-dispatched `fma`), and every
//! product and sum is renormalized on its own (xprec: 9 flops per product,
//! 20 per sum). These kernels evaluate `acc + x1 y1 +- x2 y2` as one exact sum
//! instead: the leading products by Dekker's splitting (exact for finite limbs
//! below 2^996 without underflow; larger or nonfinite operands keep
//! `mul_add`), their rounding errors and all lower-order terms gathered in one
//! binary64 sum, and one final exact two-sum. The error is a few u^2 times the
//! sum of the magnitudes of the terms (u = 2^-53), the class of xprec's own
//! operations (Joldes, Muller, Popescu, ACM TOMS 44 (2018)), at about 60% of
//! their flops.
//!
//! Every operation is symmetric under exchanging the factors of a product and
//! odd in each factor, so `complex_mul(a, b) == complex_mul(b, a)`,
//! `complex_mul(a, -b) == -complex_mul(a, b)` and the imaginary part of
//! `z conj(z)` is exactly zero, as the exact predicates of the master
//! expressions require (see `NativeFloat::complex_mul` for DoubleFloat).
use symbolica::domains::float::{Complex, DoubleFloat};
use xprec::Df64;

/// 2^27 + 1.
const SPLITTER: f64 = 134_217_729.0;
/// 2^996: Dekker's splitting overflows above it.
const SPLIT_LIMIT: f64 = 6.696_928_794_914_171e299;

#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let high = c - (c - a);
    (high, a - high)
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virtual = s - a;
    (s, (a - (s - b_virtual)) + (b - b_virtual))
}

#[inline(always)]
fn limbs(x: &DoubleFloat) -> (f64, f64) {
    let x = Df64::from(*x);
    (x.hi(), x.lo())
}

#[inline(always)]
fn join(hi: f64, lo: f64) -> DoubleFloat {
    let (hi, lo) = two_sum(hi, lo);
    DoubleFloat::from(Df64::new_full(hi, lo))
}

/// A double-double factor with its leading limb split once, for repeated
/// products with the same factor.
#[derive(Clone, Copy, Debug)]
pub struct Factor {
    hi: f64,
    lo: f64,
    high: f64,
    low: f64,
    splittable: bool,
}

#[inline(always)]
pub fn factor(x: &DoubleFloat) -> Factor {
    let (hi, lo) = limbs(x);
    let (high, low) = split(hi);
    Factor { hi, lo, high, low, splittable: hi.abs() < SPLIT_LIMIT }
}

/// `x y = p + e` up to O(u^2 |x y|): the leading product and the rest. The
/// rest sums the two cross products in a fixed order of their values, so it
/// is symmetric in `x` and `y`.
#[inline(always)]
fn product(x: &Factor, y: &DoubleFloat) -> (f64, f64) {
    let (y_hi, y_lo) = limbs(y);
    let p = x.hi * y_hi;
    let error = if x.splittable && y_hi.abs() < SPLIT_LIMIT {
        let (bh, bl) = split(y_hi);
        // Exact (Dekker): each step is error free.
        ((x.high * bh - p) + x.high * bl + x.low * bh) + x.low * bl
    } else {
        x.hi.mul_add(y_hi, -p)
    };
    (p, error + (x.hi * y_lo + x.lo * y_hi))
}

/// `acc + x1 y1 + s x2 y2` with `s = -1` (`subtract`) or `+1`.
#[inline(always)]
fn sum2(acc: (f64, f64), x1: &Factor, y1: &DoubleFloat, x2: &Factor, y2: &DoubleFloat, subtract: bool) -> DoubleFloat {
    let (p1, e1) = product(x1, y1);
    let (p2, e2) = product(x2, y2);
    let (p2, e2) = if subtract { (-p2, -e2) } else { (p2, e2) };
    let (s, f) = two_sum(acc.0, p1);
    let (t, g) = two_sum(s, p2);
    join(t, acc.1 + (e1 + e2) + (f + g))
}

/// `acc + x y`.
#[inline(always)]
fn sum1(acc: (f64, f64), x: &Factor, y: &DoubleFloat) -> DoubleFloat {
    let (p, e) = product(x, y);
    let (s, f) = two_sum(acc.0, p);
    join(s, acc.1 + e + f)
}

/// Whether `x` is exactly zero.
#[inline(always)]
pub fn is_zero(x: &DoubleFloat) -> bool {
    limbs(x).0 == 0.0
}

/// `acc += a b` for a real factor `a` prepared by [`factor`].
#[inline(always)]
pub fn real_fma(acc: &mut Complex<DoubleFloat>, a: &Factor, b: &Complex<DoubleFloat>) {
    let re = sum1(limbs(&acc.re), a, &b.re);
    let im = sum1(limbs(&acc.im), a, &b.im);
    acc.re = re;
    acc.im = im;
}

/// `a b` for a real factor `a`.
#[inline(always)]
pub fn real_scale(a: &Factor, b: &Complex<DoubleFloat>) -> Complex<DoubleFloat> {
    Complex::new(sum1((0.0, 0.0), a, &b.re), sum1((0.0, 0.0), a, &b.im))
}

/// `acc += a b`, with the factor `a` prepared by [`factor`].
#[inline(always)]
pub fn complex_fma(acc: &mut Complex<DoubleFloat>, a: &(Factor, Factor), b: &Complex<DoubleFloat>) {
    let re = sum2(limbs(&acc.re), &a.0, &b.re, &a.1, &b.im, true);
    let im = sum2(limbs(&acc.im), &a.0, &b.im, &a.1, &b.re, false);
    acc.re = re;
    acc.im = im;
}

/// `a b`.
#[inline(always)]
pub fn complex_mul(a: &Complex<DoubleFloat>, b: &Complex<DoubleFloat>) -> Complex<DoubleFloat> {
    let a = (factor(&a.re), factor(&a.im));
    Complex::new(
        sum2((0.0, 0.0), &a.0, &b.re, &a.1, &b.im, true),
        sum2((0.0, 0.0), &a.0, &b.im, &a.1, &b.re, false),
    )
}

/// `x y` (real).
#[inline(always)]
pub fn mul(x: &DoubleFloat, y: &DoubleFloat) -> DoubleFloat {
    let (p, e) = product(&factor(x), y);
    join(p, e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::float::{Float, Real, RealLike};

    fn wide(x: &DoubleFloat) -> Float {
        let (hi, lo) = limbs(x);
        Float::with_val(300, hi) + Float::with_val(300, lo)
    }

    fn sample(k: u64) -> DoubleFloat {
        // Deterministic operands with nonzero low limbs over a wide exponent range.
        let a = ((k * 2_654_435_761) % 1_000_003) as f64 / 1_000_003.0 - 0.5;
        let b = ((k * 40_503) % 997) as f64 / 997.0 * 1e-17;
        let scale = 2f64.powi((k % 61) as i32 - 30);
        join(a * scale, b * scale)
    }

    #[test]
    fn products_are_double_double_accurate_and_symmetric() {
        for k in 1..2000u64 {
            let (x, y) = (sample(k), sample(3 * k + 7));
            let exact = wide(&x) * wide(&y);
            let value = mul(&x, &y);
            let error = (wide(&value) - &exact).norm().to_f64();
            assert!(error <= 4.0 * 2f64.powi(-106) * exact.norm().to_f64(), "k = {k}: error {error:e}");
            let (a, b) = (Complex::new(x, sample(k + 11)), Complex::new(y, sample(k + 13)));
            let (ab, ba) = (complex_mul(&a, &b), complex_mul(&b, &a));
            assert_eq!(limbs(&ab.re), limbs(&ba.re));
            assert_eq!(limbs(&ab.im), limbs(&ba.im));
            let negated = complex_mul(&a, &Complex::new(-b.re, -b.im));
            assert_eq!(limbs(&negated.re), limbs(&-ab.re));
            assert_eq!(limbs(&negated.im), limbs(&-ab.im));
            let conjugate = Complex::new(a.re, -a.im);
            assert_eq!(limbs(&complex_mul(&a, &conjugate).im), (0.0, 0.0));
            let re = wide(&a.re) * wide(&b.re) - wide(&a.im) * wide(&b.im);
            let im = wide(&a.re) * wide(&b.im) + wide(&a.im) * wide(&b.re);
            let bound = 8.0 * 2f64.powi(-106) * (wide(&a.re).norm().to_f64() + wide(&a.im).norm().to_f64())
                * (wide(&b.re).norm().to_f64() + wide(&b.im).norm().to_f64());
            assert!((wide(&ab.re) - &re).norm().to_f64() <= bound, "k = {k}");
            assert!((wide(&ab.im) - &im).norm().to_f64() <= bound, "k = {k}");
            let mut acc = Complex::new(sample(k + 17), sample(k + 19));
            let start = (wide(&acc.re), wide(&acc.im));
            complex_fma(&mut acc, &(factor(&a.re), factor(&a.im)), &b);
            let magnitude = bound + wide(&acc.re).norm().to_f64() + wide(&acc.im).norm().to_f64();
            assert!((wide(&acc.re) - (start.0 + &re)).norm().to_f64() <= 8.0 * 2f64.powi(-106) * magnitude, "k = {k}");
            assert!((wide(&acc.im) - (start.1 + &im)).norm().to_f64() <= 8.0 * 2f64.powi(-106) * magnitude, "k = {k}");
        }
    }
}
