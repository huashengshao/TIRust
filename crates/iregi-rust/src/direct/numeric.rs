//! Scalar kernels of the direct engine: complex helpers, the four native
//! Laurent slots and a pivoted dense solve for the small Gram/Cayley systems.
//!
//! Reuse checks (AGENTS.md): Symbolica's `Matrix` provides exact and generic
//! field inversion, used by the symbolic planner. It allocates per operation
//! and is generic over Symbolica fields, which dominates the cost of the 1x1
//! to 5x5 per-sample systems here. FeynKit/gammaloop offer no numeric dense
//! solver. This module is a numeric kernel only; all algebra that defines the
//! recurrences remains in the symbolic planner and its tests.

use std::any::Any;

use iregi_masters::NativeFloat;
use symbolica::domains::float::{Complex, DoubleFloat};

pub(crate) type C<T> = Complex<T>;
/// `[UV single, IR single, IR double, finite]` coefficients.
pub(crate) type L4<T> = [C<T>; 4];

/// Double-double kernels: oneloopmaster's fused multiply-accumulate (one
/// exact sum per component, Dekker products, no `fma` calls), with half the
/// products for real factors. xprec's separately renormalized products and
/// sums, each product through out-of-line `fma` calls on a baseline x86-64
/// build, took two to three times as long in the precision rescue (the
/// opening N7R7 batch of 32, 106-bit tier: recurrence 112 -> 37 ms, gather
/// 57 -> 23 ms). Binary64 and MPFR keep the generic expressions.
use oneloop::double_double as dd;

/// The double-double view of a generic complex value (`T = DoubleFloat`);
/// the precision test folds at compile time, so other domains keep their
/// generic code unchanged.
#[inline(always)]
fn as_double<T: NativeFloat>(z: &C<T>) -> Option<&C<DoubleFloat>> {
    if T::FIXED_BITS == Some(106) { (z as &dyn Any).downcast_ref() } else { None }
}

#[inline(always)]
fn as_double_mut<T: NativeFloat>(z: &mut C<T>) -> Option<&mut C<DoubleFloat>> {
    if T::FIXED_BITS == Some(106) { (z as &mut dyn Any).downcast_mut() } else { None }
}

#[inline(always)]
fn as_double_l4<T: NativeFloat>(z: &L4<T>) -> Option<&L4<DoubleFloat>> {
    if T::FIXED_BITS == Some(106) { (z as &dyn Any).downcast_ref() } else { None }
}

#[inline(always)]
fn as_double_l4_mut<T: NativeFloat>(z: &mut L4<T>) -> Option<&mut L4<DoubleFloat>> {
    if T::FIXED_BITS == Some(106) { (z as &mut dyn Any).downcast_mut() } else { None }
}

#[inline(always)]
fn as_double_real<T: NativeFloat>(x: &T) -> Option<&DoubleFloat> {
    if T::FIXED_BITS == Some(106) { (x as &dyn Any).downcast_ref() } else { None }
}

#[inline(always)]
pub(crate) fn cmul<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    if let (Some(x), Some(y)) = (as_double(a), as_double(b)) {
        // Real factors (real kinematics): half the products.
        let product = if dd::is_zero(&y.im) {
            dd::real_scale(&dd::factor(&y.re), x)
        } else if dd::is_zero(&x.im) {
            dd::real_scale(&dd::factor(&x.re), y)
        } else {
            dd::complex_mul(x, y)
        };
        let mut out = a.clone();
        if let Some(slot) = as_double_mut(&mut out) {
            *slot = product;
            return out;
        }
    }
    C::new(
        a.re.clone() * &b.re - a.im.clone() * &b.im,
        a.re.clone() * &b.im + a.im.clone() * &b.re,
    )
}

#[inline(always)]
pub(crate) fn cadd<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    C::new(a.re.clone() + &b.re, a.im.clone() + &b.im)
}

#[inline(always)]
pub(crate) fn csub<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    C::new(a.re.clone() - &b.re, a.im.clone() - &b.im)
}

/// `acc += a * b`.
#[inline(always)]
pub(crate) fn cfma<T: NativeFloat>(acc: &mut C<T>, a: &C<T>, b: &C<T>) {
    if let (Some(x), Some(y)) = (as_double(a), as_double(b)) {
        if let Some(total) = as_double_mut(acc) {
            if dd::is_zero(&x.im) {
                dd::real_fma(total, &dd::factor(&x.re), y);
            } else {
                dd::complex_fma(total, &(dd::factor(&x.re), dd::factor(&x.im)), y);
            }
            return;
        }
    }
    acc.re += a.re.clone() * &b.re - a.im.clone() * &b.im;
    acc.im += a.re.clone() * &b.im + a.im.clone() * &b.re;
}

#[inline(always)]
pub(crate) fn cscale<T: NativeFloat>(a: &C<T>, s: &T) -> C<T> {
    if let (Some(x), Some(factor)) = (as_double(a), as_double_real(s)) {
        let product = dd::real_scale(&dd::factor(factor), x);
        let mut out = a.clone();
        if let Some(slot) = as_double_mut(&mut out) {
            *slot = product;
            return out;
        }
    }
    C::new(a.re.clone() * s, a.im.clone() * s)
}

/// `acc += value * s` for a real `s` (binary64: `acc + s value` as written).
#[inline(always)]
pub(crate) fn cfma_real<T: NativeFloat>(acc: &mut C<T>, value: &C<T>, s: &T) {
    if let (Some(x), Some(factor)) = (as_double(value), as_double_real(s)) {
        if let Some(total) = as_double_mut(acc) {
            dd::real_fma(total, &dd::factor(factor), x);
            return;
        }
    }
    *acc = cadd(acc, &cscale(value, s));
}

#[inline(always)]
pub(crate) fn cnorm2<T: NativeFloat>(a: &C<T>) -> T {
    if let Some(x) = as_double(a) {
        let norm = dd::mul(&x.re, &x.re) + dd::mul(&x.im, &x.im);
        let mut out = a.re.clone();
        if let Some(slot) = (&mut out as &mut dyn Any).downcast_mut::<DoubleFloat>() {
            *slot = norm;
            return out;
        }
    }
    a.re.clone() * &a.re + a.im.clone() * &a.im
}

pub(crate) fn cinv<T: NativeFloat>(a: &C<T>) -> C<T> {
    T::complex_inverse(a)
}

#[inline(always)]
pub(crate) fn czero<T: NativeFloat>(prototype: &T) -> C<T> {
    C::new(prototype.zero(), prototype.zero())
}

#[inline(always)]
pub(crate) fn lzero<T: NativeFloat>(zero: &C<T>) -> L4<T> {
    [zero.clone(), zero.clone(), zero.clone(), zero.clone()]
}

/// `acc += factor * value` for all four Laurent slots.
#[inline(always)]
pub(crate) fn lfma<T: NativeFloat>(acc: &mut L4<T>, factor: &C<T>, value: &L4<T>) {
    if let (Some(x), Some(values)) = (as_double(factor), as_double_l4(value)) {
        if let Some(total) = as_double_l4_mut(acc) {
            if dd::is_zero(&x.im) {
                let x = dd::factor(&x.re);
                for slot in 0..4 {
                    dd::real_fma(&mut total[slot], &x, &values[slot]);
                }
            } else {
                let x = (dd::factor(&x.re), dd::factor(&x.im));
                for slot in 0..4 {
                    dd::complex_fma(&mut total[slot], &x, &values[slot]);
                }
            }
            return;
        }
    }
    for slot in 0..4 {
        cfma(&mut acc[slot], factor, &value[slot]);
    }
}

/// `acc += factor * value` in the active slots: all four, or only UV and
/// finite when the lane has no IR poles (`FULL = false`).
#[inline(always)]
pub(crate) fn lfma_slots<T: NativeFloat, const FULL: bool>(
    acc: &mut L4<T>,
    factor: &C<T>,
    value: &L4<T>,
) {
    if FULL {
        lfma(acc, factor, value);
    } else {
        if let (Some(x), Some(values)) = (as_double(factor), as_double_l4(value)) {
            if let Some(total) = as_double_l4_mut(acc) {
                if dd::is_zero(&x.im) {
                    let x = dd::factor(&x.re);
                    dd::real_fma(&mut total[0], &x, &values[0]);
                    dd::real_fma(&mut total[3], &x, &values[3]);
                } else {
                    let x = (dd::factor(&x.re), dd::factor(&x.im));
                    dd::complex_fma(&mut total[0], &x, &values[0]);
                    dd::complex_fma(&mut total[3], &x, &values[3]);
                }
                return;
            }
        }
        cfma(&mut acc[0], factor, &value[0]);
        cfma(&mut acc[3], factor, &value[3]);
    }
}

/// Multiply by `(a - 2 epsilon)` through the finite term.
pub(crate) fn mul_dimension<T: NativeFloat>(value: &mut L4<T>, a: &T) {
    let two = a.from_usize(2);
    let [uv, ir1, ir2, finite] = value.clone();
    value[0] = cscale(&uv, a);
    value[1] = csub(&cscale(&ir1, a), &cscale(&ir2, &two));
    value[2] = cscale(&ir2, a);
    value[3] = csub(&cscale(&finite, a), &cscale(&cadd(&uv, &ir1), &two));
}

/// Multiply by `(a - 2 epsilon)^-1` through the finite term, `a != 0`.
/// `inv_a = 1/a` is supplied by the caller at working precision.
pub(crate) fn div_dimension<T: NativeFloat>(value: &mut L4<T>, inv_a: &T) {
    let two = inv_a.from_usize(2);
    let four = inv_a.from_usize(4);
    let inv2 = inv_a.clone() * inv_a;
    let inv3 = inv2.clone() * inv_a;
    let [uv, ir1, ir2, finite] = value.clone();
    value[0] = cscale(&uv, inv_a);
    value[1] = cadd(&cscale(&ir1, inv_a), &cscale(&ir2, &(two.clone() * &inv2)));
    value[2] = cscale(&ir2, inv_a);
    value[3] = cadd(
        &cadd(&cscale(&finite, inv_a), &cscale(&cadd(&uv, &ir1), &(two * &inv2))),
        &cscale(&ir2, &(four * &inv3)),
    );
}

/// Outcome of a small dense solve.
pub(crate) struct SolveQuality<T> {
    /// `|det(A)| / prod_i ||row_i||_2`, at most one; zero for singular input.
    pub relative_determinant: T,
    /// `|det(A)|`.
    pub determinant: T,
}

/// Solve `A X = B` in place by Gaussian elimination with partial pivoting.
/// `a` is `n x n` row-major and is destroyed; `b` is `n x m` row-major and
/// receives `X`. Returns `None` for an exactly singular pivot.
pub(crate) fn solve_in_place<T: NativeFloat>(
    a: &mut [C<T>],
    b: &mut [C<T>],
    n: usize,
    m: usize,
) -> Option<SolveQuality<T>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * m);
    let one = a[0].re.one();
    // Hadamard bound for the relative determinant estimate.
    let mut hadamard = one.clone();
    for row in 0..n {
        let mut norm = a[0].re.zero();
        for column in 0..n {
            norm += cnorm2(&a[row * n + column]);
        }
        hadamard *= norm.sqrt();
    }
    let mut determinant_norm = one.clone();
    for column in 0..n {
        let mut pivot = column;
        let mut best = cnorm2(&a[column * n + column]);
        for row in column + 1..n {
            let candidate = cnorm2(&a[row * n + column]);
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best.is_zero() || !best.is_finite() {
            return None;
        }
        if pivot != column {
            for k in 0..n {
                a.swap(column * n + k, pivot * n + k);
            }
            for k in 0..m {
                b.swap(column * m + k, pivot * m + k);
            }
        }
        determinant_norm *= best.sqrt();
        let inverse = cinv(&a[column * n + column]);
        for row in column + 1..n {
            let factor = cmul(&a[row * n + column], &inverse);
            if factor.re.is_zero() && factor.im.is_zero() {
                continue;
            }
            for k in column + 1..n {
                let update = cmul(&factor, &a[column * n + k]);
                a[row * n + k] = csub(&a[row * n + k], &update);
            }
            for k in 0..m {
                let update = cmul(&factor, &b[column * m + k]);
                b[row * m + k] = csub(&b[row * m + k], &update);
            }
        }
    }
    for column in (0..n).rev() {
        let inverse = cinv(&a[column * n + column]);
        for k in 0..m {
            let mut value = b[column * m + k].clone();
            for j in column + 1..n {
                let update = cmul(&a[column * n + j], &b[j * m + k]);
                value = csub(&value, &update);
            }
            b[column * m + k] = cmul(&value, &inverse);
        }
    }
    let relative_determinant = if hadamard.is_zero() {
        hadamard
    } else {
        determinant_norm.clone() / &hadamard
    };
    Some(SolveQuality {
        relative_determinant,
        determinant: determinant_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C<f64> {
        C::new(re, im)
    }

    #[test]
    fn pivoted_solve_inverts_a_complex_matrix() {
        let n = 3;
        let matrix = vec![
            c(0., 0.),
            c(2., 1.),
            c(1., 0.),
            c(1., -1.),
            c(3., 0.),
            c(0., 2.),
            c(4., 0.),
            c(1., 1.),
            c(2., -3.),
        ];
        let mut a = matrix.clone();
        let mut b = vec![c(0., 0.); 9];
        for i in 0..3 {
            b[i * 3 + i] = c(1., 0.);
        }
        let quality = solve_in_place(&mut a, &mut b, n, n).unwrap();
        assert!(quality.relative_determinant > 0.1);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = c(0., 0.);
                for k in 0..3 {
                    cfma(&mut sum, &matrix[i * 3 + k], &b[k * 3 + j]);
                }
                let expected = if i == j { 1. } else { 0. };
                assert!((sum.re - expected).abs() < 1e-14 && sum.im.abs() < 1e-14);
            }
        }
    }

    #[test]
    fn dimension_factors_round_trip() {
        let mut value = [c(1., 2.), c(-3., 0.5), c(0.25, -1.), c(7., 3.)];
        let original = value;
        mul_dimension(&mut value, &3.0);
        div_dimension(&mut value, &(1. / 3.0));
        for slot in 0..4 {
            assert!((value[slot].re - original[slot].re).abs() < 1e-14);
            assert!((value[slot].im - original[slot].im).abs() < 1e-14);
        }
    }
}
