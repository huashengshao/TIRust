//! Hand-written binary64 and double-double kernels for the dominant C0/D0
//! mass sectors.
//!
//! Each kernel is a direct transcription of the exact expressions, branch for
//! branch: the three-mass triangle of `triangle.rs` (OneLOop `trif3`), the
//! contour box of `box_complex.rs` (`boxc`, used for three and four massive
//! propagators where OneLOop selects it) and the four-mass Denner-Nierste-
//! Scharf box of `box_integral.rs` (`boxf4`), with the same exact zero tests,
//! comparisons, infinitesimal directions and sheets (`sheet_exact.rs`, mirrored
//! by [`sheet::Sheet`]). Unlike the generated code they evaluate one sector
//! dispatch per call, never evaluate inactive branches, and keep real values
//! and predicates real. Sectors not covered return `None` and use the
//! generated code, which also remains the implementation for `Float`: its
//! tracked precision follows the expression's own grouping.
//!
//! The triangle kernel is generic over [`NativeFloat`]; the box kernels run in
//! binary64 and double-double ([`real::BoxReal`]). Double-double takes the
//! expressions' double-double branches: the arithmetic decides the precision
//! tests that the expressions form at run time (the far-root series, `1 + x`
//! rounding to 1), and the contour box keeps every line product of a
//! parallel chart without binary64's probe (`box_complex::tfun_selected`).
//! In double-double the triangle forms the r-functions' discriminant `k² - 4`
//! of real arguments from the exact threshold polynomial instead of the
//! rounded `k` (`triangle::r_function_of`): the same value, without the
//! cancellation that left about 1e-24 next to a threshold.
use super::primitives::{C, NativeFloat, inverse, mul, sqrt};

mod box_contour;
mod boxes;
mod real;
pub(crate) mod sheet;
pub(crate) mod triangle;

/// Working-precision constants, prepared once per evaluator.
#[derive(Clone)]
pub(crate) struct Constants<T: NativeFloat> {
    pub(crate) one: T,
    pub(crate) two: T,
    pub(crate) four: T,
    pub(crate) half: T,
    pub(crate) pi: T,
    pub(crate) pi2_6: T,
    pub(crate) one_c: C<T>,
    pub(crate) two_fifty_six: T,
    small_ints: [T; 9],
    /// `B_n/(n+1)!` for `n = 1, 2, 4, ..., 22` (`sheet_exact::series_body`).
    pub(crate) bernoulli: [T; 12],
}

impl<T: NativeFloat> Constants<T> {
    pub(crate) fn new(prototype: &T) -> Self {
        let one = prototype.one();
        let two = prototype.from_i64(2);
        let pi = prototype.pi();
        let pi2_6 = pi.clone() * &pi / &prototype.from_i64(6);
        // B_n/(n+1)! for n = 1, 2, 4, ..., 22, as exact rationals.
        const BERNOULLI: [(i64, i64); 12] = [
            (-1, 2),
            (1, 6),
            (-1, 30),
            (1, 42),
            (-1, 30),
            (5, 66),
            (-691, 2730),
            (7, 6),
            (-3617, 510),
            (43867, 798),
            (-174611, 330),
            (854513, 138),
        ];
        let bernoulli = core::array::from_fn(|k| {
            let n = if k == 0 { 1 } else { 2 * k as u32 };
            let (numerator, denominator) = BERNOULLI[k];
            let factorial = symbolica::domains::integer::Integer::factorial(n + 1);
            let value = symbolica::domains::rational::Rational::from((numerator, denominator))
                / &symbolica::domains::rational::Rational::from(factorial);
            prototype.from_rational(&value)
        });
        Self {
            bernoulli,
            half: one.clone() / &two,
            four: prototype.from_i64(4),
            two_fifty_six: prototype.from_i64(256),
            one_c: C::new(one.clone(), prototype.zero()),
            small_ints: core::array::from_fn(|i| prototype.from_i64(i as i64 - 4)),
            one,
            two,
            pi,
            pi2_6,
        }
    }

    #[inline]
    pub(crate) fn from_int(&self, value: i32) -> T {
        if (-4..=4).contains(&value) {
            self.small_ints[(value + 4) as usize].clone()
        } else {
            self.one.from_i64(value as i64)
        }
    }
}

#[inline]
pub(crate) fn is_zero<T: NativeFloat>(z: &C<T>) -> bool {
    z.re.is_fully_zero() && z.im.is_fully_zero()
}

/// `sheet_exact::negative` of a real value: `x - |x|` is nonzero (x < 0,
/// +-inf or NaN, as the expression's own test).
#[inline]
pub(crate) fn negative<T: NativeFloat>(x: &T) -> bool {
    !(x.clone() - &x.norm()).is_fully_zero()
}

/// `sheet_exact::sign_nonnegative` of a real value: -1 iff `x - |x|` is nonzero.
#[inline]
pub(crate) fn sign_nonnegative<T: NativeFloat>(x: &T) -> i32 {
    if negative(x) { -1 } else { 1 }
}

#[inline]
pub(crate) fn add<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    C::new(a.re.clone() + &b.re, a.im.clone() + &b.im)
}

#[inline]
pub(crate) fn sub<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    C::new(a.re.clone() - &b.re, a.im.clone() - &b.im)
}

#[inline]
pub(crate) fn neg<T: NativeFloat>(a: &C<T>) -> C<T> {
    C::new(-a.re.clone(), -a.im.clone())
}

#[inline]
pub(crate) fn scale<T: NativeFloat>(a: &C<T>, f: &T) -> C<T> {
    C::new(a.re.clone() * f, a.im.clone() * f)
}

#[inline]
pub(crate) fn div<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    mul(a, &inverse(b))
}

#[inline]
pub(crate) fn real<T: NativeFloat>(x: T) -> C<T> {
    let zero = x.zero();
    C::new(x, zero)
}

/// `physical_sqrt` (`sqrt_lower`): principal off the real axis, lower lip on
/// the negative real axis.
#[inline]
pub(crate) fn physical_sqrt<T: NativeFloat>(z: &C<T>) -> C<T> {
    if !z.im.is_fully_zero() {
        sqrt(z)
    } else if negative(&z.re) {
        C::new(z.re.zero(), -(-z.re.clone()).sqrt())
    } else {
        C::new(z.re.clone().sqrt(), z.im.clone())
    }
}

/// `triangle::r_function`: the root r of r + 1/r = k selected like OneLOop's
/// rfun, and r - 1/r, taken directly from the square root added to k (it is
/// exactly zero at a threshold and keeps its sign next to one).
#[inline]
pub(crate) fn r_function<T: NativeFloat>(k: &C<T>, c: &Constants<T>) -> (C<T>, C<T>) {
    if !k.im.is_fully_zero() {
        let two = real(c.two.clone());
        let difference = mul(&sqrt(&sub(k, &two)), &sqrt(&add(k, &two)));
        (scale(&add(k, &difference), &c.half), difference)
    } else {
        let x = &k.re;
        r_function_real(x, x.clone() * x - &c.four, c)
    }
}

/// [r_function] of a real `k = x` given its discriminant `x² - 4`.
#[inline]
pub(crate) fn r_function_real<T: NativeFloat>(x: &T, discriminant: T, c: &Constants<T>) -> (C<T>, C<T>) {
    if negative(&discriminant) {
        let root = (-discriminant).sqrt();
        (
            C::new(x.clone() * &c.half, -(root.clone() * &c.half)),
            C::new(x.zero(), -root),
        )
    } else {
        let root = discriminant.sqrt();
        let difference = if negative(x) { -root } else { root };
        (
            C::new((x.clone() + &difference) * &c.half, x.zero()),
            C::new(difference, x.zero()),
        )
    }
}

/// C0 `[finite, pole, double pole]` for the covered sectors: three massive
/// propagators (`register_triangles` sector 7) with nonzero momenta and a
/// nonnegative Källén function (OneLOop trif3).
#[inline]
pub(crate) fn c0<T: NativeFloat>(a: &[C<T>], c: &Constants<T>) -> Option<[C<T>; 3]> {
    if a[3..6].iter().any(is_zero) {
        return None;
    }
    let finite = triangle::three_masses([&a[0], &a[1], &a[2]], [&a[3], &a[4], &a[5]], c)?;
    let zero = real(c.one.zero());
    Some([finite, zero.clone(), zero])
}

/// D0 `[finite, pole, double pole]` for the covered sectors (binary64 and
/// double-double).
#[inline]
pub(crate) fn d0<T: NativeFloat>(a: &[C<T>], c: &Constants<T>) -> Option<[C<T>; 3]> {
    use core::any::Any;
    use symbolica::domains::float::DoubleFloat;
    let a: &[C<T>; 11] = a.try_into().ok()?;
    let c = c as &dyn Any;
    let values = a as &dyn Any;
    if let (Some(a), Some(c)) = (values.downcast_ref::<[C<f64>; 11]>(), c.downcast_ref::<Constants<f64>>()) {
        let values = d0_with(a, c)?;
        return (&values as &dyn Any).downcast_ref::<[C<T>; 3]>().cloned();
    }
    if let (Some(a), Some(c)) = (values.downcast_ref::<[C<DoubleFloat>; 11]>(), c.downcast_ref::<Constants<DoubleFloat>>()) {
        let values = d0_with(a, c)?;
        return (&values as &dyn Any).downcast_ref::<[C<T>; 3]>().cloned();
    }
    None
}

#[inline]
fn d0_with<T: real::BoxReal>(a: &[C<T>; 11], c: &Constants<T>) -> Option<[C<T>; 3]> {
    let finite = boxes::d0(&a.clone().map(real::Z::of), c)?;
    let zero = C::new(T::lit(0.0), T::lit(0.0));
    Some([finite.c(), zero.clone(), zero])
}


/// The hand-written kernels in double-double against the generated
/// double-double code, both against the generated expressions at 256 (or 512)
/// bits: the randomized kinematics of `tests/native_handwritten.rs` (complex
/// masses and triangle thresholds included), and boxes with invariants on
/// two-particle thresholds.
#[cfg(test)]
mod double_double_tests {
    use crate::{NativeEvaluator, ScalarIntegral};
    use symbolica::domains::float::{Complex, DoubleFloat, Float, FloatLike, RealLike};

    type C64 = Complex<f64>;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn uniform(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn nice(&mut self, scale: f64) -> f64 {
            match self.below(4) {
                0 => (self.below(9) as f64 - 4.0) * scale / 4.0,
                _ => (2.0 * self.uniform() - 1.0) * scale,
            }
        }
    }

    fn real(x: f64) -> C64 {
        C64::new(x, 0.0)
    }

    /// [p1,p2,p3,m1,m2,m3,mu2] with three or two nonzero masses, some momenta
    /// on two-particle thresholds (`tests/native_handwritten.rs`).
    fn triangle_point(rng: &mut Rng) -> Vec<C64> {
        let mass = |rng: &mut Rng, common: f64| -> C64 {
            let re = match rng.below(3) {
                0 => common,
                1 => (1 + rng.below(4)) as f64 * 0.25,
                _ => 0.05 + rng.uniform(),
            };
            let im = if rng.below(4) == 0 { -0.1 * rng.uniform() } else { 0.0 };
            C64::new(re, im)
        };
        let common = 0.05 + rng.uniform();
        let mut m: Vec<C64> = (0..3).map(|_| mass(rng, common)).collect();
        if rng.below(3) == 0 {
            m[rng.below(3)] = C64::new(0.0, 0.0);
        }
        let momentum = |rng: &mut Rng, m: &[C64]| -> f64 {
            match rng.below(6) {
                0 => 0.0,
                1 => m[rng.below(3)].re,
                2 => {
                    let (a, b) = (m[rng.below(3)].re.sqrt(), m[rng.below(3)].re.sqrt());
                    (a + b) * (a + b)
                }
                3 => -rng.uniform() * 4.0,
                _ => rng.nice(6.0),
            }
        };
        let mut p: Vec<C64> = (0..3).map(|_| real(momentum(rng, &m))).collect();
        if rng.below(5) == 0 {
            p[rng.below(3)] = p[rng.below(3)];
        }
        p.extend(m);
        p.push(real(0.3 + rng.uniform()));
        p
    }

    /// [p1..p4,s12,s23,m1..m4,mu2] with four or three nonzero masses
    /// (`tests/native_handwritten.rs`).
    fn box_point(rng: &mut Rng, complex: bool) -> Vec<C64> {
        let common = 0.05 + rng.uniform();
        let mut m: Vec<C64> = (0..4)
            .map(|_| {
                let re = match rng.below(3) {
                    0 | 1 => common,
                    _ => 0.05 + rng.uniform(),
                };
                let im = if complex && rng.below(2) == 0 { -0.05 * rng.uniform() } else { 0.0 };
                C64::new(re, im)
            })
            .collect();
        if rng.below(3) == 0 {
            m[rng.below(4)] = C64::new(0.0, 0.0);
        }
        let mut p: Vec<C64> = (0..6)
            .map(|_| {
                real(match rng.below(6) {
                    0 => 0.0,
                    1 => m[rng.below(4)].re,
                    2 => -rng.uniform() * 3.0,
                    _ => rng.nice(8.0),
                })
            })
            .collect();
        if rng.below(4) == 0 {
            p[rng.below(6)] = p[rng.below(6)];
        }
        p.extend(m);
        p.push(real(0.3 + rng.uniform()));
        p
    }

    fn wide(x: &DoubleFloat, bits: u32) -> Float {
        let pair = x.into_inner();
        Float::with_val(bits, pair.hi()) + Float::with_val(bits, pair.lo())
    }

    fn evaluate_dd(evaluator: &mut NativeEvaluator<DoubleFloat>, hand: bool, input: &[C64]) -> Vec<Complex<DoubleFloat>> {
        evaluator.set_handwritten(hand);
        let input: Vec<_> = input.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect();
        let zero = Complex::new(DoubleFloat::from(0.0), DoubleFloat::from(0.0));
        let mut out = vec![zero; 3];
        evaluator.evaluate(&input, &mut out).unwrap();
        out
    }

    fn evaluate_float(evaluator: &mut NativeEvaluator<Float>, input: &[C64], bits: u32) -> Vec<Complex<Float>> {
        evaluator.set_handwritten(false);
        let input: Vec<_> = input.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect();
        let mut out: Vec<_> = (0..3).map(|_| Complex::new(Float::new(bits), Float::new(bits))).collect();
        evaluator.evaluate(&input, &mut out).unwrap();
        out
    }

    fn tracked_bits(values: &[Complex<Float>]) -> u32 {
        values.iter().flat_map(|z| [&z.re, &z.im]).filter(|x| !x.is_fully_zero()).map(|x| x.prec()).min().unwrap_or(u32::MAX)
    }

    /// max_k |a_k - b_k| / max_k |b_k| (infinite for a NaN).
    fn difference(a: &[Complex<DoubleFloat>], b: &[Complex<Float>]) -> f64 {
        let scale = b.iter().map(|z| z.re.to_f64().hypot(z.im.to_f64())).fold(0.0, f64::max).max(1e-300);
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let re = (wide(&x.re, 512) - &y.re).to_f64();
                let im = (wide(&x.im, 512) - &y.im).to_f64();
                re.hypot(im) / scale
            })
            .fold(0.0, |w: f64, d| if d.is_nan() { f64::INFINITY } else { w.max(d) })
    }

    fn count(default: usize, variable: &str) -> usize {
        std::env::var(variable).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// As `tests/native_handwritten.rs` in binary64: near thresholds both
    /// codes lose digits to the same conditioning and which one is luckier
    /// varies point by point, so those cases are counted both ways (beyond
    /// 2^-90, well above double-double rounding); a gross deviation (beyond
    /// 2^-40 and 16 times the generated code's) fails.
    fn run(family: ScalarIntegral, points: &[Vec<C64>]) {
        let mut dd = NativeEvaluator::<DoubleFloat>::new(family).unwrap();
        let mut precise = NativeEvaluator::<Float>::with_binary_precision(family, 256).unwrap();
        let mut very_precise = NativeEvaluator::<Float>::with_binary_precision(family, 512).unwrap();
        let show = std::env::var("HANDWRITTEN_SHOW").is_ok();
        let (mut worst_hand, mut worst_generated) = (0.0f64, 0.0f64);
        let (mut unresolved, mut compared, mut worse, mut better) = (0, 0, 0, 0);
        let mut failures = vec![];
        for input in points {
            let mut truth = evaluate_float(&mut precise, input, 256);
            if tracked_bits(&truth) < 140 {
                truth = evaluate_float(&mut very_precise, input, 512);
            }
            // A reference that comes out exactly zero (where it is not) is
            // unresolved, as one with fewer tracked bits or a nonfinite one.
            let zero = truth.iter().all(|z| z.re.is_fully_zero() && z.im.is_fully_zero());
            let nonfinite = truth.iter().any(|z| !z.re.to_f64().is_finite() || !z.im.to_f64().is_finite());
            if zero || nonfinite || tracked_bits(&truth) < 120 {
                unresolved += 1;
                continue;
            }
            compared += 1;
            let hand = evaluate_dd(&mut dd, true, input);
            let generated = evaluate_dd(&mut dd, false, input);
            let (h, g) = (difference(&hand, &truth), difference(&generated, &truth));
            worst_hand = worst_hand.max(h);
            if g.is_finite() {
                worst_generated = worst_generated.max(g);
            }
            let small = 2f64.powi(-90);
            let report = |label: &str| {
                if show {
                    eprintln!("{label}: hand {h:.2e} generated {g:.2e}: {input:?}\n  hand={:?}\n  generated={:?}", hand[0], generated[0]);
                }
            };
            if h > small && h > 16.0 * g {
                worse += 1;
                report("worse");
            }
            if g > small && g > 16.0 * h {
                better += 1;
                report("better");
            }
            if h > 2f64.powi(-40) && !(h <= 16.0 * g) {
                failures.push(format!("hand {h:.2e} generated {g:.2e}: {input:?}"));
            }
        }
        eprintln!(
            "{family:?} double-double: {} points, {compared} compared ({unresolved} unresolved): worst hand {worst_hand:.2e}, generated {worst_generated:.2e}; hand more than 16x less accurate on {worse}, more accurate on {better}",
            points.len()
        );
        assert!(worse <= better + points.len() / 500, "hand-written double-double less accurate: {worse} against {better}");
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
    }

    #[test]
    fn double_double_c0_matches_generated_expressions() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let points: Vec<_> = (0..count(3000, "HANDWRITTEN_POINTS")).map(|_| triangle_point(&mut rng)).collect();
        run(ScalarIntegral::C0, &points);
    }

    #[test]
    fn double_double_d0_matches_generated_expressions() {
        let mut rng = Rng(0x2545f4914f6cdd1d);
        let points: Vec<_> = (0..count(600, "HANDWRITTEN_BOX_POINTS")).map(|i| box_point(&mut rng, i % 3 == 0)).collect();
        run(ScalarIntegral::D0, &points);
    }

    /// Invariants on the two-particle thresholds `(√mi ± √mj)²` of their
    /// propagator pairs (rounded to binary64).
    #[test]
    fn double_double_d0_thresholds() {
        // Propagator pairs of p1..p4, s12, s23.
        const PAIRS: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)];
        let mut rng = Rng(0x5851f42d4c957f2d);
        let points: Vec<_> = (0..count(600, "HANDWRITTEN_BOX_POINTS"))
            .map(|i| {
                let mut x = box_point(&mut rng, i % 4 == 0);
                for (slot, &(a, b)) in PAIRS.iter().enumerate() {
                    if rng.below(3) == 0 {
                        let (ra, rb) = (x[6 + a].re.sqrt(), x[6 + b].re.sqrt());
                        let root = if rng.below(4) == 0 { ra - rb } else { ra + rb };
                        x[slot] = real(root * root);
                    }
                }
                x
            })
            .collect();
        run(ScalarIntegral::D0, &points);
    }
}
