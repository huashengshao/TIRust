//! Scalar primitives used by the generated native expression DAG.

use symbolica::domains::float::{
    Complex, DoubleFloat, Float, FloatLike, Real, RealLike, SingleFloat,
};
use symbolica::domains::rational::Rational;

pub type C<T> = Complex<T>;

// Resolve Symbolica's public polylog(2, z) implementation once per domain.
// Enter state before acquiring our cache: initialization can reenter OneLOop.
fn resolve_dilog<T: symbolica::evaluate::EvaluationDomain>()
-> Box<dyn symbolica::evaluate::ExternalFunction<T>> {
    let tag = symbolica::atom::Atom::num(2);
    T::resolve_function(
        &[tag.as_view()],
        symbolica::get_symbol!("symbolica::polylog")
            .unwrap()
            .get_evaluation_info()
            .unwrap(),
    )
    .expect("Symbolica provides complex polylog evaluation")
}

/// The double-double dilogarithm through a 160-bit MPFR evaluation (the
/// previous route), as a reference for the native double-double kernel.
#[doc(hidden)]
pub fn double_dilog_via_float(z: &C<DoubleFloat>) -> C<DoubleFloat> {
    let value = C::new(double_to_float(z.re, 160), double_to_float(z.im, 160));
    let value = <Float as NativeFloat>::dilog(&value);
    C::new(value.re.to_double_float(), value.im.to_double_float())
}

/// Symbolica's generic `polylog(2, z)` at the precision of `z`: the reference
/// for the native arbitrary-precision kernel (`native::dilog`).
#[doc(hidden)]
pub fn symbolica_dilog_float(z: &C<Float>) -> C<Float> {
    dilog_float_callback()(std::slice::from_ref(z))
}

fn dilog_float_callback() -> &'static dyn symbolica::evaluate::ExternalFunction<C<Float>> {
    static CALLBACK: std::sync::OnceLock<Box<dyn symbolica::evaluate::ExternalFunction<C<Float>>>> =
        std::sync::OnceLock::new();
    if let Some(callback) = CALLBACK.get() {
        return callback.as_ref();
    }
    let _ = symbolica::get_symbol!("symbolica::polylog");
    CALLBACK.get_or_init(resolve_dilog).as_ref()
}

/// A real scalar supported by the native integral evaluator.
pub trait NativeFloat: Real + RealLike + SingleFloat + PartialOrd + Send + Sync + 'static {
    const FIXED_BITS: Option<u32>;
    const DEFAULT_BITS: u32;

    fn zero_at(bits: u32) -> Self;
    fn at_precision(&self, bits: u32) -> Self;
    fn negative_sign(&self) -> bool;
    fn hypot(&self, other: &Self) -> Self;
    fn dilog(z: &C<Self>) -> C<Self>;

    /// The exact represented value, for certified algebraic branch predicates.
    /// No rounding or binary64 intermediary is permitted. Custom numeric
    /// domains may leave this unsupported; callers must then fail explicitly
    /// when an exact predicate is required. Nonfinite values return `None`.
    fn exact_rational(&self) -> Option<Rational> {
        None
    }

    // Keep ordinary machine arithmetic inline, while allowing larger numeric
    // domains to share these operations across the generated expression DAG.
    #[doc(hidden)]
    #[inline]
    fn complex_add(a: &C<Self>, b: &C<Self>) -> C<Self> {
        inline_add(a, b)
    }

    #[doc(hidden)]
    #[inline]
    fn complex_mul(a: &C<Self>, b: &C<Self>) -> C<Self> {
        inline_mul(a, b)
    }

    #[doc(hidden)]
    #[inline]
    fn complex_inverse(a: &C<Self>) -> C<Self> {
        scaled_inverse(a)
    }

    #[doc(hidden)]
    fn exact_i64(&self) -> Option<i64>;
}

/// Binary64 magnitudes whose squares neither overflow nor underflow, and for
/// which the square of the smaller component may only underflow when it is
/// negligible against the larger one.
const F64_SAFE_MIN: f64 = 3.054936363499605e-151; // 2^-500
const F64_SAFE_MAX: f64 = 3.273390607896142e150; // 2^500

impl NativeFloat for f64 {
    const FIXED_BITS: Option<u32> = Some(53);
    const DEFAULT_BITS: u32 = 53;

    fn exact_rational(&self) -> Option<Rational> {
        Float::with_val(53, *self).try_to_rational()
    }

    #[inline]
    fn zero_at(_: u32) -> Self {
        0.0
    }
    #[inline]
    fn at_precision(&self, _: u32) -> Self {
        *self
    }
    #[inline]
    fn negative_sign(&self) -> bool {
        self.is_sign_negative()
    }
    #[inline]
    fn hypot(&self, other: &Self) -> Self {
        // Exact axes need no libm norm computation. abs also gives +0 for
        // every signed-zero pair. Non-axis NaN/Inf cases still reach hypot,
        // preserving its infinity precedence (including hypot(NaN, Inf)).
        if *other == 0.0 {
            self.abs()
        } else if *self == 0.0 {
            other.abs()
        } else {
            // In the safe range the direct form is within about one ulp,
            // without libm's rescaling; NaN/Inf and extreme scales keep hypot.
            let (x, y) = (self.abs(), other.abs());
            let largest = x.max(y);
            if largest > F64_SAFE_MIN && largest < F64_SAFE_MAX {
                (x * x + y * y).sqrt()
            } else {
                f64::hypot(*self, *other)
            }
        }
    }
    #[inline]
    fn complex_inverse(a: &C<Self>) -> C<Self> {
        // Off the axes and in the safe range, one division replaces the
        // scaled algorithm's dependent divisions (within about two ulps).
        // A finite value on an axis takes the scaled algorithm's own exact
        // shortcut here (the same values), so that the common real inverses
        // need no call: `scaled_inverse` is generic and not `#[inline]`, and
        // where rustc's partition of the codegen units leaves it outside the
        // caller's unit, the opaque call cost the hand-written box kernel 6%
        // more instructions per D0 call. NaN/Inf (including overflowing
        // intermediates, which must stay visible) and extreme scales keep the
        // scaled algorithm.
        let (x, y) = (a.re.abs(), a.im.abs());
        if x > 0.0 && y > 0.0 {
            let largest = x.max(y);
            if largest > F64_SAFE_MIN && largest < F64_SAFE_MAX {
                let scale = 1.0 / (a.re * a.re + a.im * a.im);
                return C::new(a.re * scale, -a.im * scale);
            }
        }
        if a.im == 0.0 && a.re.is_finite() {
            return C::new(1.0 / a.re, -a.im);
        }
        if a.re == 0.0 && a.im.is_finite() {
            return C::new(a.re, -(1.0 / a.im));
        }
        scaled_inverse(a)
    }
    #[inline]
    fn dilog(z: &C<Self>) -> C<Self> {
        // Symbolica's binary64 kernel, also behind its registered polylog(2, z)
        // callback; calling it directly skips the callback and its lookup.
        symbolica::transcendental::dilog_complex_f64(*z)
    }
    #[inline]
    fn exact_i64(&self) -> Option<i64> {
        if self.is_finite() && *self >= i64::MIN as f64 && *self < 9_223_372_036_854_775_808.0 {
            let integer = *self as i64;
            (*self == integer as f64).then_some(integer)
        } else {
            None
        }
    }
}

fn double_to_float(value: DoubleFloat, bits: u32) -> Float {
    let pair = value.into_inner();
    if pair.hi() == 0.0 && pair.lo() == 0.0 {
        return Float::with_val(bits, pair.hi());
    }
    Float::with_val(bits, pair.hi()) + Float::with_val(bits, pair.lo())
}

impl NativeFloat for DoubleFloat {
    // Compensated arithmetic expands to many instructions. Sharing these two
    // operations avoids repeating that expansion in every generated branch.
    #[inline(never)]
    fn complex_add(a: &C<Self>, b: &C<Self>) -> C<Self> {
        inline_add(a, b)
    }

    // The exact predicates of the expressions test analytically real values
    // such as `1 - z conj(z)` with `x - |x|`, which is nonzero for any
    // imaginary residue. With a commutative product the imaginary part of
    // `z conj(z)`, `re (-im) + im re`, is exactly zero, as in binary64 and
    // MPFR; otherwise a residue of a few 1e-35 decided |z| > 1 for |z| = 0.09
    // and inverted a dilogarithm argument (wrong four-mass D0 real part).
    // The fused kernel is commutative and keeps that residue exactly zero.
    #[inline(never)]
    fn complex_mul(a: &C<Self>, b: &C<Self>) -> C<Self> {
        super::double_double::complex_mul(a, b)
    }

    const FIXED_BITS: Option<u32> = Some(106);
    const DEFAULT_BITS: u32 = 106;

    fn exact_rational(&self) -> Option<Rational> {
        let pair = self.into_inner();
        Some(
            Float::with_val(53, pair.hi()).try_to_rational()?
                + Float::with_val(53, pair.lo()).try_to_rational()?,
        )
    }

    fn zero_at(_: u32) -> Self {
        0.0.into()
    }
    fn at_precision(&self, _: u32) -> Self {
        *self
    }
    fn negative_sign(&self) -> bool {
        let pair = self.into_inner();
        if pair.hi() == 0.0 && pair.lo() != 0.0 {
            pair.lo().is_sign_negative()
        } else {
            pair.hi().is_sign_negative()
        }
    }
    fn hypot(&self, other: &Self) -> Self {
        let a = self.into_inner().hi();
        let b = other.into_inner().hi();
        if a.is_infinite() || b.is_infinite() {
            return f64::INFINITY.into();
        }
        if a.is_nan() || b.is_nan() {
            return f64::NAN.into();
        }
        let a = self.norm();
        let b = other.norm();
        let (large, small) = if a >= b { (a, b) } else { (b, a) };
        // An axis (most calls: sign tests `x - |x|` of real values) is the
        // value below exactly, without its division and square root.
        if large.is_fully_zero() || small.is_fully_zero() {
            return large;
        }
        let ratio = small / large;
        large * (large.one() + ratio * ratio).sqrt()
    }
    fn dilog(z: &C<Self>) -> C<Self> {
        // Native double-double evaluation (both limbs, no binary64
        // intermediate): the arbitrary-precision kernel with coefficients
        // rounded once to double-double.
        super::dilog::dilog(z, super::dilog::double_coefficients(), 106)
    }
    fn exact_i64(&self) -> Option<i64> {
        double_to_float(*self, 128).exact_i64()
    }
}

impl NativeFloat for Float {
    const FIXED_BITS: Option<u32> = None;
    const DEFAULT_BITS: u32 = 128;

    fn exact_rational(&self) -> Option<Rational> {
        self.try_to_rational()
    }

    fn zero_at(bits: u32) -> Self {
        Float::new(bits)
    }
    fn at_precision(&self, bits: u32) -> Self {
        let mut value = self.clone();
        value.set_prec(bits);
        value
    }
    fn negative_sign(&self) -> bool {
        self.is_sign_negative()
    }
    fn hypot(&self, other: &Self) -> Self {
        use symbolica::domains::backend::float::MultiPrecisionFloat;
        if self.is_finite() && other.is_finite() {
            if !self.is_fully_zero() && other.is_fully_zero() {
                return self.norm();
            }
            if self.is_fully_zero() && !other.is_fully_zero() {
                return other.norm();
            }
            if !self.is_fully_zero() {
                let a = self.norm();
                let b = other.norm();
                let (large, small) = if a >= b { (a, b) } else { (b, a) };
                let ratio = small / &large;
                // Only this exact constant uses the strongest precision.
                // Tracked arithmetic retains uncertainty from either input;
                // evaluating raw MPFR hypot at max precision would conceal a
                // weak dominant component's uncertainty, even off the axes.
                let one = Self::with_val(self.prec().max(other.prec()), 1);
                return large * (one + ratio.clone() * ratio).sqrt();
            }
        }
        // Keep MPFR's established NaN/Inf precedence and both-zero behavior.
        MultiPrecisionFloat::with_val(
            self.prec().max(other.prec()),
            self.as_raw().hypot_ref(other.as_raw()),
        )
        .into()
    }
    fn dilog(z: &C<Self>) -> C<Self> {
        let bits = z.re.get_precision().max(z.im.get_precision());
        // Lift both parts to the working precision first: binary operations
        // keep the precision of their left operand, so a part whose tracked
        // precision dropped (a cancelled, nearly zero real part) would
        // otherwise degrade the whole series.
        let (mut re, mut im) = (z.re.clone(), z.im.clone());
        re.set_prec(bits);
        im.set_prec(bits);
        let coefficients = super::dilog::float_coefficients(bits);
        super::dilog::dilog(&C::new(re, im), &coefficients, bits)
    }
    fn exact_i64(&self) -> Option<i64> {
        let rational = self.try_to_rational()?;
        (rational.denominator() == 1)
            .then(|| rational.numerator().to_i64())
            .flatten()
    }
}

#[inline]
pub fn add<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    T::complex_add(a, b)
}

#[inline]
pub fn inline_add<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    C::new(a.re.clone() + &b.re, a.im.clone() + &b.im)
}

#[inline]
pub fn mul<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    T::complex_mul(a, b)
}

#[inline]
pub fn inline_mul<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    // Keep separately rounded products, including exactly cancelling pairs.
    C::new(
        a.re.clone() * &b.re - a.im.clone() * &b.im,
        a.re.clone() * &b.im + a.im.clone() * &b.re,
    )
}

#[inline]
fn constant_prototype<T: NativeFloat>(a: &C<T>) -> &T {
    // Exact constants must not inherit the poor relative precision of a
    // cancelled component when the other component retains more bits. This
    // selects precision only for new constants; computed values are not padded.
    if a.re.get_precision() >= a.im.get_precision() {
        &a.re
    } else {
        &a.im
    }
}

#[inline]
pub(crate) fn inverse<T: NativeFloat>(a: &C<T>) -> C<T> {
    T::complex_inverse(a)
}

fn scaled_inverse<T: NativeFloat>(a: &C<T>) -> C<T> {
    // An overflowing intermediate must stay visible to the caller's finite
    // checks: the axis shortcuts would otherwise turn infinity into zero.
    if !a.is_finite() {
        return a.inv();
    }
    if a.im.is_fully_zero() {
        return C::new(a.re.inv(), -a.im.clone());
    }
    if a.re.is_fully_zero() {
        return C::new(a.re.clone(), -a.im.inv());
    }
    let x = a.re.norm();
    let y = a.im.norm();
    let scale = if x >= y { x } else { y };
    let x = a.re.clone() / &scale;
    let y = a.im.clone() / &scale;
    let norm = x.clone() * &x + y.clone() * &y;
    C::new((x / &norm) / &scale, (-y / norm) / scale)
}

#[inline]
pub fn powi<T: NativeFloat>(a: &C<T>, exponent: i64, _is_real: bool) -> C<T> {
    if exponent == 1 {
        return a.clone();
    }
    if exponent == -1 {
        return inverse(a);
    }
    let mut power = exponent.unsigned_abs();
    if power == 0 {
        let prototype = constant_prototype(a);
        return C::new(prototype.one(), prototype.zero());
    }
    let mut value = None;
    let mut base = a.clone();
    while power != 0 {
        if power & 1 != 0 {
            value = Some(match value {
                Some(value) => mul(&value, &base),
                None => base.clone(),
            });
        }
        power >>= 1;
        if power != 0 {
            base = mul(&base, &base);
        }
    }
    let value = value.expect("a nonzero exponent contains a set bit");
    if exponent < 0 { inverse(&value) } else { value }
}

#[inline]
pub fn powf<T: NativeFloat>(a: &C<T>, b: &C<T>, is_real: bool) -> C<T> {
    if b.im.is_fully_zero() {
        let half = b.re.one() / b.re.from_i64(2);
        if b.re == half {
            return sqrt(a);
        }
        if b.re == -half {
            return inverse(&sqrt(a));
        }
        let three_halves = b.re.from_i64(3) / b.re.from_i64(2);
        if b.re == three_halves {
            return mul(a, &sqrt(a));
        }
        if b.re == -three_halves {
            // Reciprocate first: a*sqrt(a) may overflow even when its inverse
            // remains representable (for example 1e210 raised to -3/2).
            return mul(&inverse(a), &inverse(&sqrt(a)));
        }
        if let Some(exponent) = b.re.exact_i64() {
            return powi(a, exponent, is_real);
        }
        if a.im.is_fully_zero() && a.re >= a.re.zero() {
            return C::new(a.re.powf(&b.re), a.im.zero());
        }
    }
    let exponent = mul(b, &log(a));
    let magnitude = exponent.re.exp();
    C::new(
        magnitude.clone() * exponent.im.cos(),
        magnitude * exponent.im.sin(),
    )
}

#[inline]
pub fn log<T: NativeFloat>(a: &C<T>) -> C<T> {
    let radius = NativeFloat::hypot(&a.re, &a.im);
    let magnitude = if !radius.is_finite() && a.re.is_finite() && a.im.is_finite() {
        let x = a.re.norm();
        let y = a.im.norm();
        let scale = if x >= y { x } else { y };
        scale.log() + NativeFloat::hypot(&(a.re.clone() / &scale), &(a.im.clone() / scale)).log()
    } else {
        radius.log()
    };
    let angle = if a.im.is_fully_zero() {
        if a.re.negative_sign() {
            let pi = constant_prototype(a).pi();
            if a.im.negative_sign() { -pi } else { pi }
        } else {
            a.im.clone()
        }
    } else if a.re.is_fully_zero() {
        let prototype = constant_prototype(a);
        let half_pi = prototype.pi() / prototype.from_i64(2);
        if a.im.negative_sign() {
            -half_pi
        } else {
            half_pi
        }
    } else {
        a.im.atan2(&a.re)
    };
    C::new(magnitude, angle)
}

#[inline]
pub fn sqrt<T: NativeFloat>(a: &C<T>) -> C<T> {
    if a.im.is_fully_zero() {
        let root = a.re.norm().sqrt();
        return if a.re < a.re.zero() {
            C::new(a.re.zero(), if a.im.negative_sign() { -root } else { root })
        } else {
            C::new(root, a.im.clone())
        };
    }
    let x = a.re.norm();
    let y = a.im.norm();
    let scale = if x >= y { x } else { y };
    let x = a.re.clone() / &scale;
    let y = a.im.clone() / &scale;
    let radius = NativeFloat::hypot(&x, &y);
    let two = constant_prototype(a).from_i64(2);
    // Scaled rectangular formula: no overflowing |z| intermediate and no
    // halving of a subnormal input before its square root is taken.
    let large = scale.sqrt() * ((radius + x.norm()) / &two).sqrt();
    if a.re >= a.re.zero() {
        let small = a.im.clone() / &large / &two;
        C::new(large, small)
    } else {
        let small = a.im.norm() / &large / two;
        C::new(small, if a.im.negative_sign() { -large } else { large })
    }
}

#[inline]
pub fn abs<T: NativeFloat>(a: &C<T>) -> C<T> {
    C::new(NativeFloat::hypot(&a.re, &a.im), a.im.zero())
}
#[inline]
pub fn conj<T: NativeFloat>(a: &C<T>) -> C<T> {
    C::new(a.re.clone(), -a.im.clone())
}
#[inline]
#[cfg(test)]
#[allow(dead_code)] // Exercised by the standalone primitive integration target.
pub fn re<T: NativeFloat>(a: &C<T>) -> C<T> {
    C::new(a.re.clone(), a.im.zero())
}
#[inline]
#[cfg(test)]
#[allow(dead_code)] // Exercised by the standalone primitive integration target.
pub fn im<T: NativeFloat>(a: &C<T>) -> C<T> {
    C::new(a.im.clone(), a.im.zero())
}
#[inline]
pub fn polylog2<T: NativeFloat>(a: &C<T>) -> C<T> {
    T::dilog(a)
}
#[inline]
pub fn truth<T: NativeFloat>(a: &C<T>) -> bool {
    !a.re.is_fully_zero() || !a.im.is_fully_zero()
}
