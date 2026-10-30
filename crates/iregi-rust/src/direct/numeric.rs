//! Scalar kernels of the direct engine: complex helpers, the four native
//! Laurent slots and a pivoted dense solve for the small Gram/Cayley systems.
//!
//! Reuse checks (AGENTS.md): Symbolica's `Matrix` provides exact and generic
//! field inversion, used by the symbolic planner. It allocates per operation
//! and is generic over Symbolica fields, which dominates the cost of the 1x1
//! to 5x5 per-sample systems here. FeynKit/gammaloop offer no numeric dense
//! solver. This module is a numeric kernel only; all algebra that defines the
//! recurrences remains in the symbolic planner and its tests.

use iregi_masters::NativeFloat;
use symbolica::domains::float::Complex;

pub(crate) type C<T> = Complex<T>;
/// `[UV single, IR single, IR double, finite]` coefficients.
pub(crate) type L4<T> = [C<T>; 4];

#[inline(always)]
pub(crate) fn cmul<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
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
    acc.re += a.re.clone() * &b.re - a.im.clone() * &b.im;
    acc.im += a.re.clone() * &b.im + a.im.clone() * &b.re;
}

#[inline(always)]
pub(crate) fn cscale<T: NativeFloat>(a: &C<T>, s: &T) -> C<T> {
    C::new(a.re.clone() * s, a.im.clone() * s)
}

#[inline(always)]
pub(crate) fn cnorm2<T: NativeFloat>(a: &C<T>) -> T {
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
