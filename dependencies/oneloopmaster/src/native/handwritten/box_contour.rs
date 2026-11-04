//! Port of the contour box `box_complex.rs` (OneLOop `boxc`:
//! Dao Thi Nhung & Le Duc Ninh, 't Hooft & Veltman), branch for branch, in
//! binary64 and double-double ([`super::real::BoxReal`]).
//!
//! Function names follow the registered `__olo_boxc_*` definitions. Every
//! exact zero test, comparison, infinitesimal direction ("Continued" first and
//! second coefficients) and sheet choice is the expression's; only the
//! evaluation order differs (inactive branches are never evaluated).
#![allow(clippy::too_many_arguments)] // The source's named polynomial coefficients.
use super::super::primitives::NativeFloat;
use super::Constants;
use super::sheet::Sheet;
use core::cell::Cell;

mod vertex;

pub(super) use super::real::{BoxReal, Z, dd, dd_given, dv, inverse, norm, nz, one, r, rv, zero, zsheet, zupper, SheetZ};

/// `coefficient_sum`: the left-to-right sum of `terms`, or the invariant
/// `exact` it equals when the sum cancels by more than 2^16.
#[inline]
fn coefficient_sum<T: BoxReal>(terms: &[Z<T>], exact: Z<T>) -> Z<T> {
    let numeric = terms.iter().fold(zero(), |sum, &term| sum + term);
    let scale = terms.iter().fold(T::lit(-0.0), |sum, &term| sum + norm(term));
    if negative(T::lit(4294967296.0) * norm(exact) - scale) { exact } else { numeric }
}

/// `negative` of a real value: `x - |x|` is nonzero (x < 0, inf or NaN).
#[inline(always)]
pub(super) fn negative<T: BoxReal>(x: T) -> bool {
    !(x - x.abs()).is_fully_zero()
}

/// `box_complex::below`: `2^bits x < y` for squared norms (binary64's half
/// precision for 52 and 56, as on every expression route).
#[inline(always)]
fn below<T: BoxReal>(bits: i32, x: T, y: T) -> bool {
    negative(T::lit(2f64.powi(bits)) * x - y)
}

/// `box_complex::agree`: `|v - w| < 2^-26 |w|`.
#[inline(always)]
fn agree<T: BoxReal>(v: Z<T>, w: Z<T>) -> bool {
    below(52, norm(v - w), norm(w))
}

/// `box_complex::close`: `|v - w| < 2^-bits |w|`.
#[inline(always)]
fn close<T: BoxReal>(bits: i32, v: Z<T>, w: Z<T>) -> bool {
    below(2 * bits, norm(v - w), norm(w))
}

/// Finite real and imaginary parts.
#[inline(always)]
fn finite<T: BoxReal>(z: Z<T>) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// `sign_nonnegative` of a real value.
#[inline(always)]
fn sgn<T: BoxReal>(x: T) -> i32 {
    if negative(x) { -1 } else { 1 }
}

/// `imaginary_sign(z, fallback)`.
#[inline(always)]
fn isign<T: BoxReal>(z: Z<T>, fallback: i32) -> i32 {
    if !z.im.is_fully_zero() { sgn(z.im) } else { fallback }
}

/// A value and the first two coefficients of its infinitesimal continuation.
#[derive(Clone, Copy, Debug)]
struct Cont<T> {
    v: Z<T>,
    f: Z<T>,
    s: Z<T>,
}

impl<T: BoxReal> Cont<T> {
    #[inline]
    fn sign(&self) -> i32 {
        isign(self.v, isign(self.f, isign(self.s, 1)))
    }

    #[inline]
    fn sheet(&self) -> Sheet<T> {
        zsheet(self.v, self.sign())
    }

    #[inline]
    fn complement(&self, constant: f64) -> Self {
        Self {
            v: r(constant) - self.v,
            f: -self.f,
            s: -self.s,
        }
    }

    #[inline]
    fn minus(&self, other: &Self) -> Self {
        Self {
            v: self.v - other.v,
            f: self.f - other.f,
            s: self.s - other.s,
        }
    }
}

/// `mass_directions`: -i*|Re m| for exactly real masses.
#[inline]
fn mass_directions<T: BoxReal>(m: [Z<T>; 4]) -> [Z<T>; 4] {
    m.map(|mass| {
        if !mass.im.is_fully_zero() {
            zero()
        } else {
            Z {
                re: T::lit(0.0),
                im: -mass.re.abs(),
            }
        }
    })
}

#[inline]
fn continued_root<T: BoxReal>(y: Z<T>, k: Z<T>, l: Z<T>, dl: Z<T>, dn: Z<T>, dn2: Z<T>) -> Cont<T> {
    continued_root_with_slope(y, r(2.0) * k * y + l, k, dl, dn, dn2)
}

/// `continued_root` given the slope `2 k y + l` at the root.
#[inline]
fn continued_root_with_slope<T: BoxReal>(y: Z<T>, derivative: Z<T>, k: Z<T>, dl: Z<T>, dn: Z<T>, dn2: Z<T>) -> Cont<T> {
    let inv = inverse(&derivative);
    let first = -((dl * y + dn) * inv);
    let second = -((k * first * first + dl * first + dn2) * inv);
    Cont {
        v: y,
        f: first,
        s: second,
    }
}

/// `principal_sqrt`: exact axes and rectangular components off the axis.
#[inline]
fn principal_sqrt<T: BoxReal>(v: Z<T>) -> Z<T> {
    if v.im.is_fully_zero() {
        if negative(v.re) {
            Z {
                re: T::lit(0.0),
                im: (-v.re).sqrt(),
            }
        } else {
            Z {
                re: v.re.sqrt(),
                im: v.im,
            }
        }
    } else {
        let absolute = <T as NativeFloat>::hypot(&v.re, &v.im);
        if negative(v.re) {
            let w = T::lit(sgn(v.im) as f64) * ((absolute - v.re) / T::lit(2.0)).sqrt();
            Z {
                re: v.im / (T::lit(2.0) * w),
                im: w,
            }
        } else {
            let u = ((absolute + v.re) / T::lit(2.0)).sqrt();
            Z {
                re: u,
                im: v.im / (T::lit(2.0) * u),
            }
        }
    }
}

/// `kallen`: λ(p, mi, mj) = (p - (√mi + √mj)²)(p - (√mi - √mj)²), each factor
/// formed from the invariants (exactly zero at a threshold).
#[inline]
fn kallen<T: BoxReal>(p: Z<T>, mi: Z<T>, mj: Z<T>) -> Z<T> {
    if p.im.is_fully_zero() && mi.im.is_fully_zero() && mj.im.is_fully_zero() && !negative(mi.re) && !negative(mj.re) {
        // Real masses and invariant: the same operations in real arithmetic.
        let (p, mi, mj) = (p.re, mi.re, mj.re);
        let plus = (mi + mj) + T::lit(2.0) * (mi * mj).sqrt();
        let minus = if mj.is_fully_zero() {
            mi
        } else if mi.is_fully_zero() {
            mj
        } else {
            (mi - mj) * (mi - mj) / plus
        };
        return rv((p - plus) * (p - minus));
    }
    let plus = (mi + mj) + r(2.0) * principal_sqrt(mi * mj);
    let minus = if !nz(mj) {
        mi
    } else if !nz(mi) {
        mj
    } else {
        let difference = mi - mj;
        difference * difference * inverse(&plus)
    };
    (p - plus) * (p - minus)
}

/// `edge_kallens`: [λ_a, λ_abc, λ_bc] of a `tfun` conic with corner values
/// `f`, `q10`, `q11`.
#[inline]
fn edge_kallens<T: BoxReal>(a: Z<T>, b: Z<T>, abc: Z<T>, f: Z<T>, q10: Z<T>, q11: Z<T>) -> [Z<T>; 3] {
    [kallen(a, q10, f), kallen(abc, q11, f), kallen(b, q11, q10)]
}

/// `root_1`, `root_2` (solabc labels and product relation).
#[inline]
fn roots<T: BoxReal>(a: Z<T>, b: Z<T>, c: Z<T>) -> [Z<T>; 2] {
    if !nz(c) {
        return [if nz(a) { -dv(b, a) } else { zero() }, zero()];
    }
    if !nz(a) {
        let linear = if nz(b) { -dv(c, b) } else { zero() };
        return [linear, linear];
    }
    let delta = principal_sqrt(b * b - r(4.0) * a * c);
    let plus = -b + delta;
    let minus = -b - delta;
    if negative(norm(plus) - norm(minus)) {
        [dv(r(2.0) * c, minus), dv(minus, r(2.0) * a)]
    } else {
        [dv(plus, r(2.0) * a), dv(r(2.0) * c, plus)]
    }
}

/// `eta(a, sa, b, sb, c, sc)` = i*pi*sc*(1+sa*sb)*(1-sa*sc)/2.
#[inline]
fn eta<T: BoxReal>(a: Z<T>, sa: i32, b: Z<T>, sb: i32, c: Z<T>, sc: i32, k: &Constants<T>) -> Z<T> {
    let sa = isign(a, sa);
    let sb = isign(b, sb);
    let sc = isign(c, sc);
    let value = sc * (1 + sa * sb) * (1 - sa * sc) / 2;
    if value == 0 {
        zero()
    } else {
        Z {
            re: T::lit(0.0),
            im: k.pi * T::lit(value as f64),
        }
    }
}

#[inline]
fn lom<T: BoxReal>(q: &Sheet<T>, k: &Constants<T>) -> Z<T> {
    q.zlom(k)
}

fn r0<T: BoxReal>(y1: &Cont<T>, y2: &Cont<T>, k: &Constants<T>) -> Z<T> {
    let (q0, n0, q1, n1) = r0_quotients(y1, y2);
    let value = dv(lom(&q0, k), y1.v) + dv(lom(&q1, k), one() - y1.v);
    let winding = n0 - n1;
    if winding != 0 {
        let pi_n = k.pi * T::lit(winding as f64);
        value + dv(Z { re: T::lit(0.0), im: pi_n }, y1.v - y2.v)
    } else {
        value
    }
}

/// The unwound quotients `y2/y1` and `(1 - y2)/(1 - y1)` of `r0` and their
/// windings.
#[inline(always)]
fn r0_quotients<T: BoxReal>(y1: &Cont<T>, y2: &Cont<T>) -> (Sheet<T>, i32, Sheet<T>, i32) {
    // `r0`: the windings of the two quotients are taken out and added as
    // i pi (n0 - n1)/(y1 - y2), with the exact pole difference.
    let split = |q: Sheet<T>| {
        let winding = if q.odd { 0 } else { q.phase };
        (
            Sheet {
                phase: q.phase - winding,
                ..q
            },
            winding,
        )
    };
    let (q0, n0) = split(y2.complement(0.0).sheet().div(&y1.complement(0.0).sheet()));
    let (q1, n1) = split(y2.complement(1.0).sheet().div(&y1.complement(1.0).sheet()));
    (q0, n0, q1, n1)
}

/// `r0_line`: `r0` given `delta = y1 - y2`, as `(log(q0) - log(q1) + i pi
/// (n0 - n1))/delta` with the logarithms from the offsets `-delta/y1` and
/// `delta/(1 - y1)`; `r0` itself at `delta = 0`.
fn r0_line<T: BoxReal>(y1: &Cont<T>, y2: &Cont<T>, delta: Z<T>, k: &Constants<T>) -> Z<T> {
    if !nz(delta) {
        return r0(y1, y2, k);
    }
    let (q0, n0, q1, n1) = r0_quotients(y1, y2);
    // -delta/y1 and delta/(1 - y1) from one reciprocal.
    let oy1 = one() - y1.v;
    let scaled = delta * inverse(&(y1.v * oy1));
    let mut logarithms = q0.zlog_given(-(scaled * oy1), k) - q1.zlog_given(scaled * y1.v, k);
    logarithms.im = logarithms.im + k.pi * T::lit((n0 - n1) as f64);
    dv(logarithms, delta)
}

/// `far_root_series`: `r1` for a far root, fy log(-z) - sum_{n<=8} M_n/(n zⁿ).
fn far_root_series<T: BoxReal>(zz: &Cont<T>, yy1: &Cont<T>, yy2: &Cont<T>, fy: Z<T>, k: &Constants<T>) -> Z<T> {
    let l2 = yy2.complement(1.0).sheet().zlog(k) - yy2.complement(0.0).sheet().zlog(k);
    let sum_y = yy1.v + yy2.v;
    let product_y = yy1.v * yy2.v;
    let inv = inverse(&zz.v);
    let mut m = [zero(); 9];
    m[0] = fy;
    m[1] = yy1.v * fy + l2;
    for n in 2..=8 {
        m[n] = rv(T::lit(1.0) / T::lit((n - 1) as f64)) + sum_y * m[n - 1] - product_y * m[n - 2];
    }
    let mut sum = m[8] * r(1.0 / 8.0);
    for n in (1..8).rev() {
        sum = m[n] * rv(T::lit(1.0) / T::lit(n as f64)) + inv * sum;
    }
    fy * zz.complement(0.0).sheet().zlog(k) - inv * sum
}

fn r1<T: BoxReal>(zz: &Cont<T>, yy1: &Cont<T>, yy2: &Cont<T>, fy: Z<T>, k: &Constants<T>) -> Z<T> {
    r1_with(zz, yy1, yy2, fy, None, k)
}

/// `r1` (no line) or `r1_line`: where the nearer pole's `y - z` cancels by
/// more than 64 (`4096 |y - z|² < |z|²`), its offset is the line's product
/// `(y1 - z)(y2 - z)` (`LineProduct`) over the farther pole's, except at
/// coincident poles (the product's two factors are then one offset).
#[inline(always)]
fn r1_with<T: BoxReal>(zz: &Cont<T>, yy1: &Cont<T>, yy2: &Cont<T>, fy: Z<T>, line: Option<&LineProduct<'_, T>>, k: &Constants<T>) -> Z<T> {
    let (z, y1, y2) = (zz.v, yy1.v, yy2.v);
    let oz = one() - z;
    let (mut yz1, mut yz2) = (y1 - z, y2 - z);
    let (mut hz1, mut hz2) = (norm(yz1), norm(yz2));
    let hzz = norm(z);
    let mut given = [false; 2];
    if let Some(line) = line {
        let nearer_first = negative(hz1 - hz2);
        let nearer = if nearer_first { hz1 } else { hz2 };
        if negative(T::lit(4096.0) * nearer - hzz) && nz(y1 - y2) {
            // An exactly real `y - z` stays real (its lip follows the
            // directions); a vanishing product keeps `y - z`.
            let over = |near: Z<T>, far: Z<T>| {
                line.over(far).map(|offset| if !near.im.is_fully_zero() { offset } else { rv(offset.re) })
            };
            if nearer_first {
                if let Some(offset) = over(yz1, yz2) {
                    line.record(0, offset, yz1, nearer, hzz);
                    yz1 = offset;
                    hz1 = norm(yz1);
                    given[0] = true;
                }
            } else if let Some(offset) = over(yz2, yz1) {
                line.record(1, offset, yz2, nearer, hzz);
                yz2 = offset;
                hz2 = norm(yz2);
                given[1] = true;
            }
        }
    }
    let q1 = Cont { v: yz1, ..yy1.minus(zz) }.sheet();
    let q2 = Cont { v: yz2, ..yy2.minus(zz) }.sheet();
    let offsets = [yz1, yz2];
    let r0 = |y: &Cont<T>, z: &Cont<T>, index: usize, k: &Constants<T>| {
        if given[index] { r0_line(y, z, offsets[index], k) } else { r0(y, z, k) }
    };
    let qz = zz.complement(0.0).sheet();
    let qo = zz.complement(1.0).sheet();
    let h12 = norm(y1 - y2);
    let hoz = norm(oz);
    // `far_root`; below |z|² = 2^14 its bound is at least 2^-52.
    if hzz >= T::lit(16384.0) {
        let larger = |a: T, b: T| if negative(a - b) { b } else { a };
        let scale = larger(larger(T::lit(1.0), norm(y1)), norm(y2));
        let ratio = scale / hzz;
        let squared = ratio * ratio;
        let bound = T::lit(16.0) * squared * squared;
        if ((T::lit(1.0) + bound) - T::lit(1.0)).is_fully_zero() {
            return far_root_series(zz, yy1, yy2, fy, k);
        }
    }
    let zsmall = negative(hzz - hz1) && negative(hzz - hz2) && negative(hzz - hoz);
    let osmall = negative(hoz - hz1) && negative(hoz - hz2);
    let logarithms = if zsmall || osmall {
        let product_log = q1.zlog(k) + q2.zlog(k);
        let y2sq = y2 * y2;
        let inv_y2sq = inverse(&y2sq);
        let yratio = Cont {
            v: dv(y2 - one(), y2),
            f: yy2.f * inv_y2sq,
            s: yy2.s * inv_y2sq - yy2.f * yy2.f * inverse(&(y2sq * y2)),
        };
        let ylog = yratio.sheet().zlog(k);
        let ratio = lom(&q1.div(&q2), k);
        let half = product_log * T::lit(0.5);
        let bracket = if zsmall {
            half + ylog - qo.zlog(k)
        } else {
            -half + ylog + qz.zlog(k)
        };
        fy * q1.zlog(k) - dv(bracket * ratio, yz2)
    } else if !negative(hz2 - h12) && !negative(hz1 - hz2) {
        fy * q1.zlog(k) - r0(yy2, zz, 1, k) * lom(&q1.div(&q2), k)
    } else if !negative(hz1 - h12) && !negative(hz2 - hz1) {
        fy * q2.zlog(k) - r0(yy1, zz, 0, k) * lom(&q2.div(&q1), k)
    } else {
        let first = if nz(yz1) {
            yz1 * q1.zlog(k) * r0(yy1, zz, 0, k)
        } else {
            zero()
        };
        let second = if nz(yz2) {
            yz2 * q2.zlog(k) * r0(yy2, zz, 1, k)
        } else {
            zero()
        };
        dv(first - second, y1 - y2)
    };
    // The complements 1 - x of the dilogarithms' arguments from the poles
    // (`r1`): a pole next to an end of the interval puts one next to one.
    // They are formed only where the dilogarithm takes them (`dd_given`).
    let dilog_z = if zsmall {
        if nz(z) {
            dv(dd_given(&qz.div(&q1), || dv(y1, yz1), &qz.div(&q2), || dv(y2, yz2), k) * z, yz1 * yz2)
        } else {
            zero()
        }
    } else {
        dv(dd_given(&q1.div(&qz), || dv(y1, z), &q2.div(&qz), || dv(y2, z), k), z)
    };
    let dilog_o = if !zsmall && osmall {
        if nz(oz) {
            dv(dd_given(&qo.div(&q1), || dv(y1 - one(), yz1), &qo.div(&q2), || dv(y2 - one(), yz2), k) * oz, yz1 * yz2)
        } else {
            zero()
        }
    } else {
        dv(dd_given(&q1.div(&qo), || dv(one() - y1, oz), &q2.div(&qo), || dv(one() - y2, oz), k), oz)
    };
    logarithms + dilog_z + dilog_o
}

fn s3_double_real<T: BoxReal>(y1: Z<T>, y2: Z<T>, z: Z<T>, a: Z<T>, k: &Constants<T>) -> Z<T> {
    let a1 = z - y1;
    let a2 = z - y2;
    let oz = one() - z;
    let q01 = zupper(-y1).div(&zupper(a1));
    let q02 = zupper(-y2).div(&zupper(a2));
    let q11 = zupper(one() - y1).div(&zupper(a1));
    let q12 = zupper(one() - y2).div(&zupper(a2));
    let log = |x: Z<T>| Z::of(super::super::primitives::log(&x.c()));
    let jdd = dv(
        z * dd(&q01, &q02, k) + oz * dd(&q11, &q12, k)
            - dv(z * log(z) * lom(&q01.div(&q02), k), q02.zvalue())
            - dv(oz * log(oz) * lom(&q11.div(&q12), k), q12.zvalue()),
        a1 * a2,
    );
    let pole = |y: Z<T>| Cont {
        v: y,
        f: zero(),
        s: zero(),
    };
    r0(&pole(y1), &pole(y2), k) * log(a) + r(2.0) * jdd
}

#[inline]
fn outside_closed_interval<T: BoxReal>(y: Z<T>) -> bool {
    !y.im.is_fully_zero() || negative(y.re) || negative(T::lit(1.0) - y.re)
}

/// `s3`: the y1, y2 poles are mapped by y -> (d + y)/e; a quadratic passes
/// its two-point subgraph `[lambda, p, near, far]` (`Edge`), and a linear
/// b x + c passes `far` = b + c.
fn s3<T: BoxReal>(
    y1: &Cont<T>, y2: &Cont<T>, d: Z<T>, e: Z<T>, a: Z<T>, b: Z<T>, c: Z<T>, db: Z<T>, dc: Z<T>, edge: [Z<T>; 4],
    k: &Constants<T>,
) -> Z<T> {
    s3_with(y1, y2, d, e, a, b, c, db, dc, edge, None, k)
}

/// `s3` (no line) or, for `a = db = 0`, `s3_line`: the linear term with the
/// line's product of the mapped poles' offsets from its root.
#[inline(always)]
fn s3_with<T: BoxReal>(
    y1: &Cont<T>, y2: &Cont<T>, d: Z<T>, e: Z<T>, a: Z<T>, b: Z<T>, c: Z<T>, db: Z<T>, dc: Z<T>, [lambda, p, near, far]: [Z<T>; 4],
    line: Option<&LineProduct<'_, T>>,
    k: &Constants<T>,
) -> Z<T> {
    if !nz(e) {
        return zero();
    }
    let simc = isign(c, isign(dc, isign(b, isign(db, -1))));
    let inv_e = inverse(&e);
    let map = |y: &Cont<T>| Cont {
        v: (d + y.v) * inv_e,
        f: y.f * inv_e,
        s: y.s * inv_e,
    };
    let y1 = map(y1);
    let y2 = map(y2);
    let fy = r0(&y1, &y2, k);
    let value = if nz(a) {
        // A massless end (far = 0 or near = 0) is a root.
        let z0 = if !nz(far) {
            one()
        } else if !nz(near) {
            zero()
        } else {
            -dv(b, r(2.0) * a)
        };
        let double = !nz(lambda);
        let real_coefficients = a.im.is_fully_zero() && b.im.is_fully_zero() && c.im.is_fully_zero();
        let double_root = double
            && real_coefficients
            && negative(-a.re)
            && negative(-z0.re)
            && negative(z0.re - T::lit(1.0))
            && outside_closed_interval(y1.v)
            && outside_closed_interval(y2.v);
        if double_root {
            s3_double_real(y1.v, y2.v, z0, a, k)
        } else {
            let rea = sgn(a.re);
            let (mut z1, mut z2) = if double {
                // An exact double root splits like sqrt(eps), one root up and
                // one down, dominating every first-order shift of y1 and y2.
                let size = |z: Z<T>| z.re.libm_hypot(z.im);
                let dominant = T::lit(2.0) * (T::lit(1.0) + size(y1.f) + size(y1.s) + size(y2.f) + size(y2.s));
                let centre = z0;
                (
                    Cont { v: centre, f: Z { re: T::lit(0.0), im: dominant }, s: zero() },
                    Cont { v: centre, f: Z { re: T::lit(0.0), im: -dominant }, s: zero() },
                )
            } else if !nz(far) || !nz(near) {
                // A massless end is a root and the other lies slope/p from it,
                // with the slope from the invariants; both continue from the
                // end value's direction.
                let (end, slope, direction) = if !nz(far) { (one(), near - p, db + dc) } else { (zero(), p - far, dc) };
                let inv_slope = inverse(&slope);
                let end_first = direction * inv_slope;
                let near_first = -dv(db, a) - direction * inv_slope;
                (
                    Cont {
                        v: end,
                        f: end_first,
                        s: (a * end_first * end_first + db * end_first) * inv_slope,
                    },
                    Cont {
                        v: end + dv(slope, rv(p.re)),
                        f: near_first,
                        s: dv(near_first * a * direction, slope * slope),
                    },
                )
            } else {
                // Roots with slopes 2 a z + b = +sqrt(lambda), -sqrt(lambda).
                let ra = rv(a.re);
                let delta = principal_sqrt(lambda);
                let plus = -b + delta;
                let minus = -b - delta;
                let (root1, root2) = if negative(norm(plus) - norm(minus)) {
                    (dv(r(2.0) * c, minus), dv(minus, r(2.0) * ra))
                } else {
                    (dv(plus, r(2.0) * ra), dv(r(2.0) * c, plus))
                };
                (
                    continued_root_with_slope(root1, delta, a, db, dc, zero()),
                    continued_root_with_slope(root2, -delta, a, db, dc, zero()),
                )
            };
            let fallback1 = simc * rea * sgn(z2.v.re);
            let fallback2 = simc * rea * sgn(z1.v.re);
            for (z, fallback) in [(&mut z1, fallback1), (&mut z2, fallback2)] {
                let known = !z.v.im.is_fully_zero() || !z.f.im.is_fully_zero() || !z.s.im.is_fully_zero();
                if !known {
                    z.s = Z {
                        re: T::lit(0.0),
                        im: T::lit(fallback as f64) * z.v.re.abs(),
                    };
                }
            }
            let s1 = z1.sign();
            let s2 = z2.sign();
            fy * (zsheet(a, simc).zlog(k)
                + eta(-z1.v, -s1, -z2.v, -s2, zero(), simc * rea, k))
                + r1(&z1, &y1, &y2, fy, k)
                + r1(&z2, &y1, &y2, fy, k)
        }
    } else if nz(b) {
        let root = if nz(far) { -dv(c, b) } else { one() };
        let mut z = continued_root(root, zero(), b, db, dc, zero());
        let known = !z.v.im.is_fully_zero() || !z.f.im.is_fully_zero() || !z.s.im.is_fully_zero();
        if !known {
            z.s = Z {
                re: T::lit(0.0),
                im: T::lit(-(simc * sgn(b.re)) as f64) * z.v.re.abs(),
            };
        }
        let sz = z.sign();
        let sb = isign(db, simc);
        fy * (zsheet(b, sb).zlog(k) + eta(b, sb, -z.v, -sz, c, simc, k))
            + r1_with(&z, &y1, &y2, fy, line, k)
    } else if nz(c) {
        zsheet(c, simc).zlog(k) * fy
    } else {
        zero()
    };
    value * inv_e
}

/// `planar`; `real` marks exactly real logarithm arguments
/// (`real_intersections_absent`).
fn planar<T: BoxReal>(y1: &Cont<T>, y2: &Cont<T>, p1: bool, p2: bool, a: Z<T>, b: Z<T>, c: Z<T>, dc: Z<T>, real: bool, k: &Constants<T>) -> Z<T> {
    if !p1 && !p2 {
        return zero();
    }
    let x1 = b * y1.v + c;
    let x2 = b * y2.v + c;
    let sheet = |y: &Cont<T>, value: Z<T>| {
        let first = b * y.f + dc;
        let second = b * y.s;
        let v2 = value * value;
        let ratio = dv(a, value);
        Cont {
            v: if real { rv(ratio.re) } else { ratio },
            f: -dv(a * first, v2),
            s: dv(a * first * first, v2 * value) - dv(a * second, v2),
        }
        .sheet()
    };
    let value = if p1 && p2 {
        let q1 = sheet(y1, x1);
        let q2 = sheet(y2, x2);
        dv(lom(&q2.div(&q1), k) * b, x2)
    } else if p1 {
        dv(sheet(y1, x1).zlog(k), y1.v - y2.v)
    } else {
        dv(sheet(y2, x2).zlog(k), y2.v - y1.v)
    };
    // 2*i*pi*value
    Z {
        re: T::lit(-2.0) * k.pi * value.im,
        im: T::lit(2.0) * k.pi * value.re,
    }
}

/// `s3` of a linear `b x + c`, with `far` = b + c.
#[inline]
fn s3_linear<T: BoxReal>(y1: &Cont<T>, y2: &Cont<T>, d: Z<T>, e: Z<T>, b: Z<T>, c: Z<T>, dc: Z<T>, far: Z<T>, k: &Constants<T>) -> Z<T> {
    s3(y1, y2, d, e, zero(), b, c, zero(), dc, [zero(), zero(), c, far], k)
}

/// `s3_line`: a line edge of `tfun_selected` (its `[d, e, b, c, far]` in
/// the `LineProduct`), whose product is given only for a line nearly parallel
/// to the contour direction (`parallel`).
fn s3_line<T: BoxReal>(y1: &Cont<T>, y2: &Cont<T>, dc: Z<T>, line: &LineProduct<'_, T>, parallel: bool, k: &Constants<T>) -> Z<T> {
    let [d, e, b, c, far] = line.line;
    s3_with(y1, y2, d, e, zero(), b, c, zero(), dc, [zero(), zero(), c, far], parallel.then_some(line), k)
}

/// `line_product`: the product `(Y1 - z)(Y2 - z)` of the mapped poles'
/// offsets from the root `z` of a line edge `b x + c` with `[d, e, b, c,
/// far]`, `a1 (Q(u) - ratio B(u))/(K e²)` at the preimage `u = e z - d`,
/// formed by `r1_with` only where an offset cancels. With the scale `s = b`
/// (1 for `far = 0`), `U = s u` and `ratio = num/den`, the nearer offset is
/// `a1 (den (Q_b U² + Q_e U s + Q_f s²) - num s (b1 U + c1 s))/(K s² den e²)`
/// over the farther one, with one division. `a1` is the poles' own, or its
/// rounding level `2^-52 g` where it vanishes (`tfun_selected`); `conic`
/// holds `b, e, f` and then `a, c, d`.
struct LineProduct<'a, T> {
    line: [Z<T>; 5],
    ratio: [Z<T>; 2],
    conic: &'a [Z<T>; 6],
    data: &'a BetaData<T>,
    a1: Z<T>,
    probe: Option<&'a Cell<Probe<T>>>,
}

/// The largest disagreement of a product offset with the `y - z` it
/// replaces, in units of that subtraction's rounding `2^-52 |z|` (squared),
/// with its pole and the pole's shift `e (offset - (y - z))` that the
/// product implies (`tfun_selected`).
type Probe<T> = (T, usize, Z<T>);

impl<T: BoxReal> LineProduct<'_, T> {
    /// Records the product `offset` that replaces the rounded `y - z`
    /// (`direct`, with `nearer = |y - z|²` and `hzz = |z|²`) of pole `index`.
    fn record(&self, index: usize, offset: Z<T>, direct: Z<T>, nearer: T, hzz: T) {
        if let Some(probe) = self.probe
            && nz(offset)
        {
            let difference = offset - direct;
            let units = norm(difference) * nearer
                / (T::lit(f64::EPSILON) * T::lit(f64::EPSILON) * hzz * norm(offset));
            if negative(probe.get().0 - units) {
                probe.set((units, index, difference * self.line[1]));
            }
        }
    }

    /// The product over the farther pole's offset `far` (zero for `far = 0`),
    /// or `None` where the product vanishes or the line's intersection with
    /// the edge lies on the conic to within the square root of the rounding
    /// (`line_product`; the expression's `r1_line` then keeps `y - z`).
    fn over(&self, far_offset: Z<T>) -> Option<Z<T>> {
        let [d, e, b, c, far] = self.line;
        let [qb, qe, qf, ..] = *self.conic;
        let BetaData { b: b1, c: c1, k: kk, .. } = *self.data;
        let a1 = self.a1;
        let (scale, scaled_u) = if nz(far) { (b, -(e * c + d * b)) } else { (one(), e - d) };
        // ratio = 0 for a vanishing denominator, as the expression's if_nonzero.
        let [num, den] = if nz(self.ratio[1]) { self.ratio } else { [zero(), one()] };
        let quadratic = (qb * scaled_u + qe * scale) * scaled_u + qf * scale * scale;
        let linear = b1 * scaled_u + c1 * scale;
        let value = den * quadratic - num * scale * linear;
        // The conic's value against its coefficients, with the value's scale
        // den s² (`line_product`).
        let conic_norm = self.conic.iter().fold(T::lit(0.0), |sum, &q| sum + norm(q));
        if !negative(conic_norm * norm(den * scale * scale) - T::lit(4503599627370496.0) * norm(value)) {
            return None;
        }
        let numerator = a1 * value;
        if !nz(numerator) {
            return None;
        }
        let denominator = kk * scale * scale * den * e * e * far_offset;
        Some(if nz(far_offset) && nz(e) { dv(numerator, denominator) } else { zero() })
    }
}

/// The line edges of a parallel chart that take their product
/// (`box_complex::products_kept`, `scale = |g|² + |beta h|²`). Every one for
/// an `a1` within 16 units of its rounding (`2^96 |a1|² < scale`) or where a
/// product replaces a rounded `y - z` that vanishes. Otherwise, among the
/// edges whose nearer pole's `y - z` cancels by more than 64: none where one
/// of them has no product, and else each but one whose pole lies within 8
/// offsets of an end of the edge (`min(|y|², |1 - y|²) < 64 |y - z|²`). A
/// kept product whose offset is more than 2^10 times `a1` relative to its
/// root (`2^20 |z|² |a1|² < |y - z|² scale`) is probed (`tfun_selected`) as
/// one that disagrees with its pole by at least 128 units: the chart's
/// `probe` records it with that disagreement (or its own where larger).
#[cold]
#[inline(never)]
fn products_kept<T: BoxReal>(lines: &[LineProduct<'_, T>; 3], data: &BetaData<T>, scale: T) -> [bool; 3] {
    let mut kept = [true; 3];
    // An a1 within 16 units of its rounding (`2^96 |a1|² < scale`) keeps them all.
    if negative(T::lit(79228162514264337593543950336.0) * norm(lines[0].a1) - scale) {
        return kept;
    }
    let mut chart = true;
    let mut unscaled: [Option<Probe<T>>; 3] = [None; 3];
    for (index, line) in lines.iter().enumerate() {
        let [d, e, b, c, far] = line.line;
        if !nz(e) || !nz(b) {
            continue;
        }
        let inv_e = inverse(&e);
        let [p1, p2] = data.y.map(|y| (d + y.v) * inv_e);
        let z = if nz(far) { -dv(c, b) } else { one() };
        let [o1, o2] = [p1 - z, p2 - z];
        let [n1, n2] = [norm(o1), norm(o2)];
        let (pole, nearer, far_offset) = if negative(n1 - n2) { (p1, n1, o2) } else { (p2, n2, o1) };
        let hzz = norm(z);
        if !(negative(T::lit(4096.0) * nearer - hzz) && nz(p1 - p2)) {
            continue;
        }
        let product = line.over(far_offset);
        // A product where the rounded `y - z` vanishes keeps them all.
        if nearer.is_fully_zero() && product.is_some() {
            return [true; 3];
        }
        let Some(offset) = product else {
            chart = false;
            continue;
        };
        let [to_zero, to_one] = [norm(pole), norm(pole - one())];
        let end = if negative(to_zero - to_one) { to_zero } else { to_one };
        if negative(end - T::lit(64.0) * nearer) {
            kept[index] = false;
        } else if negative(T::lit(1048576.0) * hzz * norm(line.a1) - nearer * scale) {
            // The disagreement `r1_with` records for this product (an exactly
            // real `y - z` takes the product's real part), at least 128 units.
            let first = negative(n1 - n2);
            let direct = if first { o1 } else { o2 };
            let offset = if !direct.im.is_fully_zero() { offset } else { rv(offset.re) };
            let difference = offset - direct;
            let units = if nz(offset) {
                norm(difference) * nearer / (T::lit(f64::EPSILON) * T::lit(f64::EPSILON) * hzz * norm(offset))
            } else {
                T::lit(0.0)
            };
            let units = if negative(units - T::lit(128.0)) { T::lit(128.0) } else { units };
            unscaled[index] = Some((units, if first { 0 } else { 1 }, difference * e));
        }
    }
    if !chart {
        return [false; 3];
    }
    for (line, probe) in lines.iter().zip(unscaled) {
        if let (Some(cell), Some(probe)) = (line.probe, probe)
            && negative(cell.get().0 - probe.0)
        {
            cell.set(probe);
        }
    }
    kept
}

/// The line products of `tfun_lines`: none (`y - z` everywhere), those
/// that `products_kept` keeps, or every one of a parallel chart (the chart
/// tests of `tfun_checked`, `box_complex::tfun_chart`).
#[derive(Clone, Copy, PartialEq)]
enum Products {
    None,
    Kept,
    All,
}

/// `t1`; `gh`, `gj`, `ghj` are the line's g + h, g + j, g + h + j.
fn t1<T: BoxReal>(
    [a, c, g, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, ac, lambda_ac, far_ac, lambda_a, far_a, gh, gj, ghj]: [Z<T>; 22],
    k: &Constants<T>,
) -> Z<T> {
    let kk = h * a - c * g;
    let l = h * d - c * j - e * g;
    let n = h * f - e * j;
    let dl = h * dd_ - c * dj - de * g;
    let dn = h * df - de * j - e * dj;
    let dn2 = -de * dj - Z { re: T::lit(0.0), im: f.re.abs() } * h;
    let [r1_, r2_] = roots(kk, l, n);
    let y1 = continued_root(r1_, kk, l, dl, dn, dn2);
    let y2 = continued_root(r2_, kk, l, dl, dn, dn2);
    let sum = coefficient_sum(&[a, c], ac);
    dv(
        -s3(&y1, &y2, zero(), one(), sum, dpe, f, ddpe, df, [lambda_ac, ac, f, far_ac], k)
            + s3_linear(&y1, &y2, zero(), one(), gh, j, dj, ghj, k)
            - s3_linear(&y1, &y2, zero(), one(), g, j, dj, gj, k)
            + s3(&y1, &y2, zero(), one(), a, d, f, dd_, df, [lambda_a, a, f, far_a], k),
        kk,
    )
}

fn t13<T: BoxReal>(
    [
        a, c, g, h, d, e, f, j, dpe, dpj, dpf, dd_, de, df, dj, ddpe, ddpj, ddpf, lambda_ae, lambda_gj,
        lambda_gh, lambda_ac, far_ae, far_gj, gh, ac,
    ]: [Z<T>; 26],
    k: &Constants<T>,
) -> Z<T> {
    let kk = h * a - c * g;
    let l = a * d + h * e - d * g - c * j;
    let n = d * (e - j) + (h - c) * f;
    let dl = (a - g) * dd_ + h * de - c * dj;
    let dn = dd_ * (e - j) + d * (de - dj) + (h - c) * df;
    let dn2 = dd_ * (de - dj) - Z { re: T::lit(0.0), im: f.re.abs() } * (h - c);
    let [r1_, r2_] = roots(kk, l, n);
    let y1 = continued_root(r1_, kk, l, dl, dn, dn2);
    let y2 = continued_root(r2_, kk, l, dl, dn, dn2);
    dv(
        -s3(&y1, &y2, zero(), one(), a, e + c, dpf, de, ddpf, [lambda_ae, a, dpf, far_ae], k)
            + s3(&y1, &y2, zero(), one(), g, j + h, dpf, dj, ddpf, [lambda_gj, g, dpf, far_gj], k)
            - s3(&y1, &y2, zero(), one(), g + h, dpj, f, ddpj, df, [lambda_gh, gh, f, far_gj], k)
            + s3(&y1, &y2, zero(), one(), a + c, dpe, f, ddpe, df, [lambda_ac, ac, f, far_ae], k),
        kk,
    )
}

/// Barycentric coordinates in the triangle (0, -beta, 1-beta).
#[inline]
fn coordinates<T: BoxReal>(beta: Z<T>, y: Z<T>) -> [T; 2] {
    let sum = -y.im / beta.im;
    let x2 = y.re + beta.re * sum;
    let x1 = sum - x2;
    [x1, x2]
}

#[inline]
fn continued_nonnegative<T: BoxReal>(value: T, first: T, second: T) -> bool {
    if !value.is_fully_zero() {
        !negative(value)
    } else if !first.is_fully_zero() {
        !negative(first)
    } else {
        !negative(second)
    }
}

fn inside<T: BoxReal>(beta: Z<T>, y: &Cont<T>) -> bool {
    if beta.im.is_fully_zero() {
        return false;
    }
    let [x1, x2] = coordinates(beta, y.v);
    let [d1, d2] = coordinates(beta, y.f);
    let [dd1, dd2] = coordinates(beta, y.s);
    continued_nonnegative(x1, d1, dd1)
        && continued_nonnegative(x2, d2, dd2)
        && continued_nonnegative(T::lit(1.0) - x1 - x2, -d1 - d2, -dd1 - dd2)
}

fn distance<T: BoxReal>(beta: Z<T>, y: &Cont<T>) -> T {
    if inside(beta, y) {
        let [x1, x2] = coordinates(beta, y.v);
        if negative(x1 - x2) { x1 } else { x2 }
    } else {
        T::lit(1.0)
    }
}

struct BetaData<T> {
    a: Z<T>,
    b: Z<T>,
    c: Z<T>,
    dc: Z<T>,
    k: Z<T>,
    y: [Cont<T>; 2],
}

/// `beta_data`; `w` is beta - 1, and nearer 1 than 0 the coefficients are
/// expanded about beta = 1 (`dpe` = d + e, `ddpe` = dd + de).
#[inline(always)]
fn beta_data<T: BoxReal>(
    beta: Z<T>, w: Z<T>, b: Z<T>, c: Z<T>, g: Z<T>, h: Z<T>, d: Z<T>, e: Z<T>, f: Z<T>, j: Z<T>, dpe: Z<T>, dd_: Z<T>, de: Z<T>, df: Z<T>,
    dj: Z<T>, ddpe: Z<T>, far_pole: bool,
) -> BetaData<T> {
    let (a1, b1, c1, dc1) = if negative(norm(w) - norm(beta)) {
        ((g + h) + w * h, (c + r(2.0) * b) + r(2.0) * w * b, dpe + w * e, ddpe + w * de)
    } else {
        (g + beta * h, c + r(2.0) * beta * b, d + beta * e, dd_ + beta * de)
    };
    let kk = b * a1 - h * b1;
    let l = e * a1 - h * c1 - j * b1;
    let n = f * a1 - j * c1;
    // In the chart (`far_pole`), a K that rounds to zero puts its far pole
    // 2^52 (1 + |N/L|²) away (`box_complex::beta_data`).
    let kk = if nz(kk) || !far_pole {
        kk
    } else if nz(l) {
        -dv(l, rv(T::lit(4503599627370496.0) * (T::lit(1.0) + norm(dv(n, l)))))
    } else {
        zero()
    };
    let dl = de * a1 - h * dc1 - dj * b1;
    let dn = df * a1 - dj * c1 - j * dc1;
    let dn2 = -dj * dc1 - Z { re: T::lit(0.0), im: f.re.abs() } * a1;
    let [r1_, r2_] = roots(kk, l, n);
    BetaData {
        a: a1,
        b: b1,
        c: c1,
        dc: dc1,
        k: kk,
        y: [
            continued_root(r1_, kk, l, dl, dn, dn2),
            continued_root(r2_, kk, l, dl, dn, dn2),
        ],
    }
}

/// `beta_data` with `K` replaced by `kk` (`box_complex::tfun_chart`'s chart
/// with K at its rounding level).
#[cold]
fn beta_data_with_k<T: BoxReal>(
    beta: Z<T>, w: Z<T>, b: Z<T>, c: Z<T>, g: Z<T>, h: Z<T>, d: Z<T>, e: Z<T>, f: Z<T>, j: Z<T>, dpe: Z<T>, dd_: Z<T>, de: Z<T>, df: Z<T>,
    dj: Z<T>, ddpe: Z<T>, kk: Z<T>,
) -> BetaData<T> {
    let (a1, b1, c1, dc1) = if negative(norm(w) - norm(beta)) {
        ((g + h) + w * h, (c + r(2.0) * b) + r(2.0) * w * b, dpe + w * e, ddpe + w * de)
    } else {
        (g + beta * h, c + r(2.0) * beta * b, d + beta * e, dd_ + beta * de)
    };
    let l = e * a1 - h * c1 - j * b1;
    let n = f * a1 - j * c1;
    let dl = de * a1 - h * dc1 - dj * b1;
    let dn = df * a1 - dj * c1 - j * dc1;
    let dn2 = -dj * dc1 - Z { re: T::lit(0.0), im: f.re.abs() } * a1;
    let [r1_, r2_] = roots(kk, l, n);
    BetaData {
        a: a1,
        b: b1,
        c: c1,
        dc: dc1,
        k: kk,
        y: [
            continued_root(r1_, kk, l, dl, dn, dn2),
            continued_root(r2_, kk, l, dl, dn, dn2),
        ],
    }
}

/// `real_intersections_absent`: real coefficients whose line and conic meet at
/// complex conjugate points (`n^T adj(M) n > 0`).
fn real_intersections_absent<T: BoxReal>(coefficients: [Z<T>; 9]) -> bool {
    if coefficients.iter().any(|z| !z.im.is_fully_zero()) {
        return false;
    }
    let [a, b, c, g, h, d, e, f, j] = coefficients.map(|z| z.re);
    let adj11 = b * f - e * e / T::lit(4.0);
    let adj22 = a * f - d * d / T::lit(4.0);
    let adj33 = a * b - c * c / T::lit(4.0);
    let adj12 = d * e / T::lit(4.0) - c * f / T::lit(2.0);
    let adj13 = c * e / T::lit(4.0) - b * d / T::lit(2.0);
    let adj23 = c * d / T::lit(4.0) - a * e / T::lit(2.0);
    let delta = g * g * adj11
        + h * h * adj22
        + j * j * adj33
        + T::lit(2.0) * (g * h * adj12 + g * j * adj13 + h * j * adj23);
    negative(-delta)
}

/// `tfun_selected` (`box_complex::tfun_selected`): `tfun_lines` with the
/// line products, unless a product offset differs from the rounded `y - z`
/// it replaces by more than 8 units of that subtraction's rounding
/// `2^-52 |z|` (the pole itself is then that inaccurate). There the pole is
/// moved by the shift the product implies, and the variant (the products, or
/// `y - z` everywhere as 09b2c74 took) whose value moves less is taken
/// (`tfun_probed`). `w` is beta - 1 and the sums are the line's g + h,
/// g + j and g + h + j.
fn tfun_selected<T: BoxReal>(
    a: Z<T>, b: Z<T>, c: Z<T>, g: Z<T>, h: Z<T>, d: Z<T>, e: Z<T>, f: Z<T>, j: Z<T>, dpe: Z<T>, dpf: Z<T>, beta: Z<T>, w: Z<T>,
    data: &BetaData<T>, dd_: Z<T>, de: Z<T>, df: Z<T>, dj: Z<T>, ddpe: Z<T>, ddpf: Z<T>, abc: Z<T>,
    edges: [Z<T>; 5], sums: [Z<T>; 3], k: &Constants<T>,
) -> Z<T> {
    let coefficients = [a, b, c, g, h, d, e, f, j, dpe, dpf, dd_, de, df, dj, ddpe, ddpf, abc];
    if !T::BINARY64 {
        return tfun_lines(coefficients, beta, w, data, edges, sums, Products::Kept, None, k);
    }
    let probe = Cell::new((T::lit(0.0), 0, zero()));
    let value = tfun_lines(coefficients, beta, w, data, edges, sums, Products::Kept, Some(&probe), k);
    probed(coefficients, beta, w, data, edges, sums, value, probe.get(), k)
}

/// `value` (`tfun_lines` with the products), or where `probe` shows a pole
/// that a product disagrees with by more than 8 units, `tfun_probed`.
#[inline(always)]
fn probed<T: BoxReal>(
    coefficients: [Z<T>; 18], beta: Z<T>, w: Z<T>, data: &BetaData<T>, edges: [Z<T>; 5], sums: [Z<T>; 3], value: Z<T>,
    (units, index, shift): Probe<T>, k: &Constants<T>,
) -> Z<T> {
    if negative(T::lit(64.0) - units) {
        tfun_probed(coefficients, beta, w, data, edges, sums, value, index, shift, k)
    } else {
        value
    }
}

/// The variant of a chart whose pole `index` the products show to be
/// inaccurate: with the products (`value`) or with `y - z` everywhere,
/// whichever changes less when that pole moves by `shift`. Where the
/// rounded pole carries the error, the line and conic terms of an edge
/// share it and cancel it with `y - z`, which the products' exact offsets
/// in the line terms alone break: with the products alone, the
/// hand-written route of v1 8793, 9431, 13198, 13274, 14824 and 15393 of
/// the verifier's 32,600-row sweep lost 1e-12 to 7.4e-6 where 09b2c74 kept
/// 6.1e-14 to 2e-8 (`box_complex::tfun_selected`). Where the subtraction alone
/// loses the digits, `y - z` moves with the pole and the products are kept
/// (the near-threshold boxes of tests/data/near_threshold_boxes.txt).
#[cold]
#[inline(never)]
fn tfun_probed<T: BoxReal>(
    coefficients: [Z<T>; 18], beta: Z<T>, w: Z<T>, data: &BetaData<T>, edges: [Z<T>; 5], sums: [Z<T>; 3], value: Z<T>,
    index: usize, shift: Z<T>, k: &Constants<T>,
) -> Z<T> {
    let mut moved = BetaData { y: data.y, ..*data };
    moved.y[index].v = moved.y[index].v + shift;
    let lines = |data: &BetaData<T>, products: Products| tfun_lines(coefficients, beta, w, data, edges, sums, products, None, k);
    let moved_products = lines(&moved, Products::Kept);
    let direct = lines(data, Products::None);
    let moved_direct = lines(&moved, Products::None);
    let finite_all = finite(moved_products) && finite(direct) && finite(moved_direct);
    if finite_all && !negative(norm(moved_products - value) - norm(moved_direct - direct)) { direct } else { value }
}

/// `tfun_lines`; the line products only where `products` (and the line is
/// nearly parallel to the contour direction; `Products::Kept` those of
/// `products_kept`), recording their disagreement with `y - z` in `probe`.
fn tfun_lines<T: BoxReal>(
    [a, b, c, g, h, d, e, f, j, dpe, dpf, dd_, de, df, dj, ddpe, ddpf, abc]: [Z<T>; 18],
    beta: Z<T>, w: Z<T>, data: &BetaData<T>,
    [lambda_a, lambda_abc, lambda_bc, q10, q11]: [Z<T>; 5], [gh, gj, ghj]: [Z<T>; 3],
    products: Products, probe: Option<&Cell<Probe<T>>>, k: &Constants<T>,
) -> Z<T> {
    let [y1, y2] = &data.y;
    let p1 = inside(beta, y1);
    let p2 = inside(beta, y2);
    let sum = coefficient_sum(&[a, b, c], abc);
    // The line edges x = 1, y = x and y = 0 as the `[d, e, b, c, far]` of
    // their s3 terms, and h u + j over a1 at their roots' preimages u.
    let conic = [b, e, f, a, c, d];
    // The products only for a line nearly parallel to the contour direction
    // (a1 = g + beta h cancels by more than 64), with an a1 that rounds to
    // zero at its rounding level 2^-52 g (`box_complex::tfun_selected`).
    let scale = norm(g) + norm(beta * h);
    let parallel = products != Products::None && negative(T::lit(4096.0) * norm(data.a) - scale);
    let a1 = if nz(data.a) { data.a } else { g * r(1.0 / 4503599627370496.0) };
    let line = |line: [Z<T>; 5], ratio: [Z<T>; 2]| LineProduct { line, ratio, conic: &conic, data, a1, probe };
    let lines = [
        line([beta, one(), h, gj, ghj], [r(-1.0), one()]),
        line([zero(), -w, gh, j, ghj], [j, gh]),
        line([zero(), -beta, g, j, gj], [j, g]),
    ];
    let kept = match (parallel, products) {
        (false, _) => [false; 3],
        (true, Products::Kept) if T::BINARY64 => products_kept(&lines, data, scale),
        (true, _) => [true; 3],
    };
    let integral = s3_line(y1, y2, dj, &lines[0], kept[0], k) - s3_line(y1, y2, dj, &lines[1], kept[1], k)
        + s3_line(y1, y2, dj, &lines[2], kept[2], k)
        - s3(y1, y2, beta, one(), b, c + e, a + dpf, de, ddpf, [lambda_bc, b, q10, q11], k)
        + s3(y1, y2, zero(), -w, sum, dpe, f, ddpe, df, [lambda_abc, abc, f, q11], k)
        - s3(y1, y2, zero(), -beta, a, d, f, dd_, df, [lambda_a, a, f, q10], k);
    let real = real_intersections_absent([a, b, c, g, h, d, e, f, j]);
    let residue = planar(y1, y2, p1, p2, data.a, data.b, data.c, data.dc, real, k);
    let residue = if negative(T::lit(0.0) - beta.im) { -residue } else { residue };
    dv(integral + residue, data.k)
}

/// The rare charts of `tfun` (`box_complex::tfun_chart`): the selected
/// contour (`[beta, w]`, `data`) with a far pole where its K rounds to zero;
/// where `check`, the other contour's chart too, taken unless the two agree
/// to 2^-26 (or it is not finite). The chart used is tested where it is the
/// selected `flat` one without a regular chart that agrees with it (NaN
/// unless the chart with the line's g moved to `g (1 + 2^-52)`, and its sums
/// by the same exact amount, keeps 10 bits of it) and where its K rounds to
/// zero or it is the other chart (whose K is `b a1` of the flat selected
/// one): NaN unless the chart with K at its rounding level keeps 13 bits.
/// Where the other chart's K is a rounding residue too (below 2^-52 of its
/// terms), neither chart is regular, and of two that disagree the selected
/// one is kept where K at its rounding level moves the other more than 2^10
/// times as much as the selected moves with its line; it must then keep 20
/// bits under that move (`box_complex::tfun_chart`).
#[cold]
#[inline(never)]
fn tfun_checked<T: BoxReal>(
    [a, b, c, g, h, d, e, f, j, dpe, dpf, dd_, de, df, dj, ddpe, ddpf, abc]: [Z<T>; 18],
    [beta, w, other_beta, other_w]: [Z<T>; 4],
    [data, other]: [&BetaData<T>; 2],
    check: bool,
    flat: bool,
    edges: [Z<T>; 5],
    sums: [Z<T>; 3],
    k: &Constants<T>,
) -> Z<T> {
    let coefficients = [a, b, c, g, h, d, e, f, j, dpe, dpf, dd_, de, df, dj, ddpe, ddpf, abc];
    let chart_with = |beta: Z<T>, w: Z<T>, data: &BetaData<T>, products: Products, probe: Option<&Cell<Probe<T>>>| {
        let far = (!nz(data.k)).then(|| beta_data(beta, w, b, c, g, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, true));
        let data = far.unwrap_or(BetaData { y: data.y, ..*data });
        (tfun_lines(coefficients, beta, w, &data, edges, sums, products, probe, k), data)
    };
    let chart = |beta: Z<T>, w: Z<T>, data: &BetaData<T>, products: Products| chart_with(beta, w, data, products, None).0;
    // The chart with the line's g moved to `g (1 + 2^-52)` (its sums by the
    // same exact amount), and the chart with K at the rounding level of its
    // terms, `2^-52 (|b a1| + |h b1|)`, both with every line product.
    let moved_line = |beta: Z<T>, w: Z<T>| {
        let shifted = g * r(1.0 + f64::EPSILON);
        let delta = shifted - g;
        let moved = beta_data(beta, w, b, c, shifted, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, true);
        tfun_lines([a, b, c, shifted, h, d, e, f, j, dpe, dpf, dd_, de, df, dj, ddpe, ddpf, abc], beta, w, &moved, edges, sums.map(|sum| sum + delta), Products::All, None, k)
    };
    let rounding_k = |beta: Z<T>, w: Z<T>, data: &BetaData<T>| {
        let magnitude = |z: Z<T>| z.re.abs() + z.im.abs();
        let rounding = rv((magnitude(b * data.a) + magnitude(h * data.b)) / T::lit(4503599627370496.0));
        let probe = beta_data_with_k(beta, w, b, c, g, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, rounding);
        tfun_lines(coefficients, beta, w, &probe, edges, sums, Products::All, None, k)
    };
    // The tests and the agreement compare the charts with every line product
    // (`used`, `Products::All`); a chart they keep gives `result`, with the
    // products that `products_kept` keeps.
    let tested = |beta: Z<T>, w: Z<T>, data: &BetaData<T>, used: Z<T>, result: Z<T>, flat: bool, zero: bool| {
        let keeps = |moved: Z<T>, bits: i32| finite(moved) && close(bits, used, moved);
        if flat && !keeps(moved_line(beta, w), 10) {
            return Z { re: T::lit(f64::NAN), im: T::lit(f64::NAN) };
        }
        if zero && !keeps(rounding_k(beta, w, data), 13) {
            return Z { re: T::lit(f64::NAN), im: T::lit(f64::NAN) };
        }
        result
    };
    // The selected chart, tested with every line product; where the tests keep
    // it, its value (with the products kept) is probed as in `tfun_selected`.
    let probe = Cell::new((T::lit(0.0), 0, zero()));
    // Binary64 alone probes (`tfun_selected`): without records, `probed` keeps the value.
    let (value, used) = chart_with(beta, w, data, Products::Kept, T::BINARY64.then_some(&probe));
    let all = chart(beta, w, data, Products::All);
    let probed = |result: Z<T>| {
        if finite(value) && result.re.same(value.re) && result.im.same(value.im) {
            probed(coefficients, beta, w, &used, edges, sums, value, probe.get(), k)
        } else {
            result
        }
    };
    if check {
        let regular = chart(other_beta, other_w, other, Products::All);
        if finite(regular) && !(finite(all) && agree(all, regular)) {
            // The other chart's K is a rounding residue (below 2^-52 of its
            // terms): the selected chart where K at its rounding level moves
            // the other (relatively) more than 2^10 times as much as the
            // selected moves with its line, NaN unless the selected keeps 20
            // bits under that move (and 13 under K at its rounding level where
            // its own K rounds to zero).
            if finite(all) && below(104, norm(other.k), norm(b * other.a) + norm(h * other.b)) {
                let line = moved_line(beta, w);
                let at_rounding = rounding_k(other_beta, other_w, other);
                if finite(line)
                    && (!finite(at_rounding)
                        || !negative(
                            norm(at_rounding - regular) * norm(all) - T::lit(1048576.0) * norm(line - all) * norm(regular),
                        ))
                {
                    let keeps = |moved: Z<T>, bits: i32| finite(moved) && close(bits, all, moved);
                    let kept = keeps(line, 20) && (nz(data.k) || keeps(rounding_k(beta, w, data), 13));
                    return probed(if kept { value } else { Z { re: T::lit(f64::NAN), im: T::lit(f64::NAN) } });
                }
            }
            let kept = chart(other_beta, other_w, other, Products::Kept);
            return tested(other_beta, other_w, other, regular, kept, false, true);
        }
        return probed(if finite(regular) { value } else { tested(beta, w, data, all, value, true, !nz(data.k)) });
    }
    probed(if flat || !nz(data.k) { tested(beta, w, data, all, value, flat, !nz(data.k)) } else { value })
}

/// `tfun`; `edges` holds `[λ_a, λ_abc, λ_bc, q10, q11]` (`edge_kallens` and
/// the conic's values at the corners (1, 0) and (1, 1)), `line_sums` the
/// line's g + h, g + j and g + h + j from the invariants.
fn tfun<T: BoxReal>(
    [a, b, c, gin, hin, d, e, f, jin, dpe, dpf, dd_, de, df, djin, ddpe, ddpf, abc]: [Z<T>; 18],
    edges: [Z<T>; 5],
    line_sums: [Z<T>; 3],
    k: &Constants<T>,
) -> Z<T> {
    let [lambda_a, lambda_abc, lambda_bc, q10, q11] = edges;
    let sj = isign(jin, isign(djin, -1));
    let s = r(-(sj as f64));
    let g = s * gin;
    let h = s * hin;
    let j = s * jin;
    let dj = s * djin;
    let [gh, gj, ghj] = line_sums.map(|v| s * v);
    let value = if nz(b) {
        if nz(a) {
            let [root1, root2] = roots(b, c, a);
            // beta - 1, the root nearer 1 from (beta1 - 1)(beta2 - 1) = abc/b.
            let [o1, o2] = [root1 - one(), root2 - one()];
            let w1 = if negative(norm(o1) - norm(o2)) { dv(abc, b * o2) } else { o1 };
            let w2 = if negative(norm(o2) - norm(o1)) { dv(abc, b * o1) } else { o2 };
            let swap = negative(norm(root2) - norm(root1));
            let ((beta1, w1), (beta2, w2)) = if swap { ((root2, w2), (root1, w1)) } else { ((root1, w1), (root2, w2)) };
            let data1 = beta_data(beta1, w1, b, c, g, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, false);
            let data2 = beta_data(beta2, w2, b, c, g, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, false);
            let min = |x: T, y: T| if negative(x - y) { x } else { y };
            let d1 = min(distance(beta1, &data1.y[0]), distance(beta1, &data1.y[1]));
            let d2 = min(distance(beta2, &data2.y[0]), distance(beta2, &data2.y[1]));
            // By default the first (`tfun`). Where the selected contour's a1
            // keeps fewer than 24 bits (`2^56 |a1|² < |g|² + |beta h|²`): if
            // the other's does not and the other has a pole within 2^7, the
            // other chart is computed too and taken unless the two agree to
            // 2^-26; otherwise the chart is tested against a1 moved by one
            // rounding unit (`tfun_chart`).
            let use_first = if !(d1 - T::lit(1.0)).is_fully_zero() { negative(d2 - d1) } else { true };
            let flat = |data: &BetaData<T>, beta: Z<T>| below(56, norm(data.a), norm(g) + norm(beta * h));
            let near = |data: &BetaData<T>| negative(norm(data.y[0].v) - T::lit(16384.0)) || negative(norm(data.y[1].v) - T::lit(16384.0));
            let ((beta, data), (other_beta, other)) = if use_first { ((beta1, &data1), (beta2, &data2)) } else { ((beta2, &data2), (beta1, &data1)) };
            let flat_selected = flat(data, beta);
            let check = flat_selected && !flat(other, other_beta) && near(other);
            if flat_selected || !nz(data.k) {
                let [w, other_w] = if use_first { [w1, w2] } else { [w2, w1] };
                let coefficients = [a, b, c, g, h, d, e, f, j, dpe, dpf, dd_, de, df, dj, ddpe, ddpf, abc];
                return -(r(sj as f64) * tfun_checked(coefficients, [beta, w, other_beta, other_w], [data, other], check, flat_selected, edges, [gh, gj, ghj], k));
            }
            let w = if use_first { w1 } else { w2 };
            tfun_selected(a, b, c, g, h, d, e, f, j, dpe, dpf, beta, w, data, dd_, de, df, dj, ddpe, ddpf, abc, edges, [gh, gj, ghj], k)
        } else {
            // a = 0: b + c = abc, and the new a + c is b. The new conic takes
            // the values q11, q10 and f at the corners (0, 0), (1, 0), (1, 1),
            // and the new line (-g - h, g, g + h + j) the sums -h, j, g + j.
            let bc = coefficient_sum(&[b, c], abc);
            t1(
                [
                    bc,
                    -c,
                    -gh,
                    g,
                    -dpe - r(2.0) * bc,
                    d + c,
                    q11,
                    ghj,
                    -e - r(2.0) * b - c,
                    -ddpe,
                    dd_,
                    ddpe + df,
                    dj,
                    -de,
                    b,
                    lambda_bc,
                    q10,
                    lambda_abc,
                    f,
                    -h,
                    j,
                    gj,
                ],
                k,
            )
        }
    } else {
        // b = 0: a + c = abc.
        t1(
            [a, c, g, h, d, e, f, j, dpe, dd_, de, df, dj, ddpe, abc, lambda_abc, q11, lambda_a, q10, gh, gj, ghj],
            k,
        )
    };
    -(r(sj as f64) * value)
}

/// `opposite_regular`: the auxiliary poles avoid the integration endpoints.
fn opposite_regular<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4]) -> bool {
    if nz(p[0]) || nz(p[2]) {
        return false;
    }
    let a = p[4] - p[0];
    let c = p[3] - p[4] - p[2];
    let g = p[1];
    let h = p[5] - p[1] - p[2];
    let d = m[2] - m[3] - p[2];
    let e = m[0] - m[2] + p[2] - p[3];
    let j = m[1] - m[2] - p[5] + p[2];
    let kk = h * a - c * g;
    let l = a * d + h * e - d * g - c * j;
    let n = d * (e - j) + (h - c) * m[3];
    nz(n * (kk + l + n))
}

fn opposite_chart<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    let dm = mass_directions(m);
    let a = p[4] - p[0];
    let c = p[3] - p[4] - p[2];
    let g = p[1];
    let h = p[5] - p[1] - p[2];
    let d = m[2] - m[3] - p[2];
    let e = m[0] - m[2] + p[2] - p[3];
    let f = m[3];
    let kk = m[1] - m[2] - p[5] + p[2];
    let dpe = m[0] - m[3] - p[3];
    let dpk = m[1] - m[3] - p[5];
    let dpf = m[2] - p[2];
    let far_ae = m[0] - p[0] - p[2];
    let far_gk = m[1] - p[2];
    t13(
        [
            a,
            c,
            g,
            h,
            d,
            e,
            f,
            kk,
            dpe,
            dpk,
            dpf,
            dm[2] - dm[3],
            dm[0] - dm[2],
            dm[3],
            dm[1] - dm[2],
            dm[0] - dm[3],
            dm[1] - dm[3],
            dm[2],
            kallen(a, dpf, far_ae),
            kallen(g, dpf, far_gk),
            // g + h and a + c, simplified as in the expression.
            kallen(p[5] - p[2], f, far_gk),
            kallen(p[3] - p[0] - p[2], f, far_ae),
            far_ae,
            far_gk,
            p[5] - p[2],
            p[3] - p[0] - p[2],
        ],
        k,
    )
}

fn opposite<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    const CYCLES: [[usize; 6]; 4] = [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 0, 5, 4],
        [2, 3, 0, 1, 4, 5],
        [3, 0, 1, 2, 5, 4],
    ];
    let chart = |rotation: usize| {
        let cycle = CYCLES[rotation];
        (
            cycle.map(|i| p[i]),
            core::array::from_fn::<_, 4, _>(|i| m[cycle[i]]),
        )
    };
    let selected = (0..4)
        .find(|&rotation| {
            let (p, m) = chart(rotation);
            opposite_regular(p, m)
        })
        .unwrap_or(0);
    let (p, m) = chart(selected);
    opposite_chart(p, m, k)
}

/// `zero_external`: `zero_external_chart` reflected where `m2` is massless,
/// the line passes nearer the corner (1, 0) than (0, 0) by more than 1/128
/// (`64 |at_10|² < 63 |at_00|²`, `box_complex::zero_external`), neither of
/// the reflected chart's conics has a double contour direction (a
/// discriminant `c² - 4 a b` within 2^-26 of its terms, `below`), the chart
/// itself has a pole that the rounding resolves poorly next to the corner
/// (1, 0) (`vertex::zero_external_suspicious`), and the reflected chart does
/// not select a flat contour without a regular alternative where the
/// original selects none (`vertex::reflection_unresolved`).
fn zero_external<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    if !nz(m[2]) && nz(m[3]) && nz(p[1]) {
        let at_10 = m[0] - m[1] + p[1] - p[4];
        let at_00 = m[0] - m[1] + p[5] - p[3];
        if negative(T::lit(64.0) * norm(at_10) - T::lit(63.0) * norm(at_00)) {
            let reflected = [p[0], p[3], p[2], p[1], p[5], p[4]];
            let reflected_m = [m[1], m[0], m[3], m[2]];
            let distinct = |a: Z<T>, b: Z<T>, c: Z<T>| {
                let [cc, ab] = [c * c, r(4.0) * a * b];
                !below(52, norm(cc - ab), norm(cc) + norm(ab))
            };
            let [p0, p1, p2, p3, p4, p5] = reflected;
            if distinct(p2, p1, p5 - p1 - p2)
                && distinct(p2, p4 - p0, p3 - p4 - p2)
                && vertex::zero_external_suspicious(p, m)
                && !vertex::reflection_unresolved(reflected, reflected_m, p, m)
            {
                return zero_external_chart(reflected, reflected_m, k);
            }
        }
    }
    zero_external_chart(p, m, k)
}

fn zero_external_chart<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    let dm = mass_directions(m);
    let a = p[2];
    let b = p[1];
    let c = p[5] - p[1] - p[2];
    let h = p[3] - p[4] - p[5] + p[1];
    let j = p[4] - p[0] - p[1];
    let d = m[2] - m[3] - p[2];
    let e = m[1] - m[2] - p[5] + p[2];
    let kk = m[0] - m[1] + p[5] - p[3];
    let f = m[3];
    let cph = p[3] - p[4] - p[2];
    let dpe = m[1] - m[3] - p[5];
    let epk = m[0] - m[2] + p[2] - p[3];
    let dek = m[0] - m[3] - p[3];
    let dpf = m[2] - p[2];
    let [lambda_a, lambda_abc, lambda_bc] = edge_kallens(a, b, p[5], f, m[2], m[1]);
    // b + j, a + b + j + cph and the conic at (1, 1), simplified as in the
    // expression.
    let bj = p[4] - p[0];
    let corner = m[0] - p[0];
    let lambda_abj = kallen(p[3] - p[0], corner, f);
    let lambda_bj = kallen(bj, corner, m[2]);
    // The line (h, j, kk): h + j, h + kk and h + j + kk.
    let line = [p[3] - p[5] - p[0], m[0] - m[1] + p[1] - p[4], m[0] - m[1] - p[0]];
    tfun(
        [
            a,
            b,
            c,
            h,
            j,
            d,
            e,
            f,
            kk,
            dpe,
            dpf,
            dm[2] - dm[3],
            dm[1] - dm[2],
            dm[3],
            dm[0] - dm[1],
            dm[1] - dm[3],
            dm[2],
            // a + b + c, simplified as in the expression.
            p[5],
        ],
        [lambda_a, lambda_abc, lambda_bc, m[2], m[1]],
        line,
        k,
    ) - tfun(
        [
            a,
            bj,
            cph,
            h,
            j,
            d,
            epk,
            f,
            kk,
            dek,
            dpf,
            dm[2] - dm[3],
            dm[0] - dm[2],
            dm[3],
            dm[0] - dm[1],
            dm[0] - dm[3],
            dm[2],
            p[3] - p[0],
        ],
        [lambda_a, lambda_abj, lambda_bj, m[2], corner],
        line,
        k,
    )
}

fn ordered<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    let dm = mass_directions(m);
    let a = p[2];
    let b = p[1];
    let g = p[0];
    let c = p[5] - p[1] - p[2];
    let h = p[3] - p[4] - p[5] + p[1];
    let j = p[4] - p[0] - p[1];
    let d = m[2] - m[3] - p[2];
    let e = m[1] - m[2] - p[5] + p[2];
    let kk = m[0] - m[1] + p[5] - p[3];
    let f = m[3];
    let abc = p[5];
    let bgj = p[4];
    let jph = p[3] - p[0] - p[5];
    let cph = p[3] - p[4] - p[2];
    let dpe = m[1] - m[3] - p[5];
    let epk = m[0] - m[2] + p[2] - p[3];
    let dek = m[0] - m[3] - p[3];
    let dpf = m[2] - p[2];
    let def = m[1] - p[5];
    let [x1, x2] = roots(g, j, b);
    let alpha = if negative(x1.re * x1.re - x2.re * x2.re) { x2 } else { x1 };
    let o1 = one() - alpha;
    let j1 = j + r(2.0) * g * alpha;
    let e1 = e + kk * alpha;
    let de1 = dm[1] - dm[2] + (dm[0] - dm[1]) * alpha;
    // The six Källén functions of the box (`edge_kallens` of the three tfun).
    let l01 = kallen(p[0], m[0], m[1]);
    let l12 = kallen(p[1], m[1], m[2]);
    let l23 = kallen(p[2], m[2], m[3]);
    let l03 = kallen(p[3], m[0], m[3]);
    let l02 = kallen(p[4], m[0], m[2]);
    let l13 = kallen(p[5], m[1], m[3]);
    // The lines and their coefficient sums (g + h, g + j, g + h + j).
    let sums = |[g, h, j]: [Z<T>; 3]| [g + h, g + j, (g + h) + j];
    let line1 = [c + r(2.0) * b + (h + j) * alpha, j1, e1];
    let line2 = [c + h * alpha, o1 * j1, e1];
    let line3 = [c + h * alpha, -j1 * alpha, e1];
    -tfun(
        [
            abc,
            g,
            jph,
            line1[0],
            line1[1],
            dpe,
            kk,
            f,
            e1,
            dek,
            def,
            dm[1] - dm[3],
            dm[0] - dm[1],
            dm[3],
            de1,
            dm[0] - dm[3],
            dm[1],
            p[3],
        ],
        [l13, l03, l01, m[1], m[0]],
        sums(line1),
        k,
    ) + o1
        * tfun(
            [
                a,
                bgj,
                cph,
                line2[0],
                line2[1],
                d,
                epk,
                f,
                e1,
                dek,
                dpf,
                dm[2] - dm[3],
                dm[0] - dm[2],
                dm[3],
                de1,
                dm[0] - dm[3],
                dm[2],
                p[3],
            ],
            [l23, l03, l02, m[2], m[0]],
            sums(line2),
            k,
        )
        + alpha
            * tfun(
                [
                    a,
                    b,
                    c,
                    line3[0],
                    line3[1],
                    d,
                    e,
                    f,
                    e1,
                    dpe,
                    dpf,
                    dm[2] - dm[3],
                    dm[1] - dm[2],
                    dm[3],
                    de1,
                    dm[1] - dm[3],
                    dm[2],
                    p[5],
                ],
                [l23, l13, l12, m[2], m[1]],
                sums(line3),
                k,
            )
}

const ORD: [[usize; 10]; 24] = [
    [1, 2, 3, 4, 1, 2, 3, 4, 5, 6],
    [4, 2, 3, 1, 6, 2, 5, 4, 3, 1],
    [2, 4, 3, 1, 6, 3, 5, 1, 2, 4],
    [1, 4, 3, 2, 4, 3, 2, 1, 5, 6],
    [4, 1, 3, 2, 4, 5, 2, 6, 3, 1],
    [2, 1, 3, 4, 1, 5, 3, 6, 2, 4],
    [2, 3, 4, 1, 2, 3, 4, 1, 6, 5],
    [1, 3, 4, 2, 5, 3, 6, 1, 4, 2],
    [3, 1, 4, 2, 5, 4, 6, 2, 3, 1],
    [2, 1, 4, 3, 1, 4, 3, 2, 6, 5],
    [1, 2, 4, 3, 1, 6, 3, 5, 4, 2],
    [3, 2, 4, 1, 2, 6, 4, 5, 3, 1],
    [3, 4, 1, 2, 3, 4, 1, 2, 5, 6],
    [2, 4, 1, 3, 6, 4, 5, 2, 1, 3],
    [4, 2, 1, 3, 6, 1, 5, 3, 4, 2],
    [3, 2, 1, 4, 2, 1, 4, 3, 5, 6],
    [2, 3, 1, 4, 2, 5, 4, 6, 1, 3],
    [4, 3, 1, 2, 3, 5, 1, 6, 4, 2],
    [4, 1, 2, 3, 4, 1, 2, 3, 6, 5],
    [3, 1, 2, 4, 5, 1, 6, 3, 2, 4],
    [1, 3, 2, 4, 5, 2, 6, 4, 1, 3],
    [4, 3, 2, 1, 3, 2, 1, 4, 6, 5],
    [3, 4, 2, 1, 3, 6, 1, 5, 2, 4],
    [1, 4, 2, 3, 4, 6, 2, 5, 1, 3],
];

const PRM_A: [usize; 24] = [
    17, 7, 22, 18, 8, 21, 23, 13, 4, 24, 14, 3, 5, 19, 10, 6, 20, 9, 11, 1, 16, 12, 2, 15,
];
const PRM_B: [usize; 24] = [
    5, 4, 1, 6, 3, 2, 11, 10, 7, 12, 9, 8, 17, 16, 13, 18, 15, 14, 23, 22, 19, 24, 21, 20,
];

/// `sort_four`: the leaf of the expression's comparison tree (1-based).
fn sort_four<T: BoxReal>(a: T, b: T, c: T, d: T) -> usize {
    const OUTPUTS: [usize; 24] = [
        10, 9, 8, 4, 11, 12, 7, 3, 24, 23, 22, 2, 14, 13, 18, 5, 15, 16, 17, 6, 19, 20, 21, 1,
    ];
    let leaf = |index: usize| OUTPUTS[index - 1];
    // le(x, y) = 1 - less(y, x)
    let le = |x: T, y: T| !negative(y - x);
    let choose = |x: T, y: T, yes: usize, no: usize| if le(x, y) { yes } else { no };
    let branch1 = || choose(c, d, leaf(1), choose(b, d, leaf(2), choose(a, d, leaf(3), leaf(4))));
    let branch2 = || choose(b, d, leaf(5), choose(c, d, leaf(6), choose(a, d, leaf(7), leaf(8))));
    let branch3 = || choose(b, d, leaf(9), choose(a, d, leaf(10), choose(c, d, leaf(11), leaf(12))));
    let branch4 = || choose(c, d, leaf(13), choose(a, d, leaf(14), choose(b, d, leaf(15), leaf(16))));
    let branch5 = || choose(a, d, leaf(17), choose(c, d, leaf(18), choose(b, d, leaf(19), leaf(20))));
    let branch6 = || choose(a, d, leaf(21), choose(b, d, leaf(22), choose(c, d, leaf(23), leaf(24))));
    if le(a, b) {
        if le(b, c) {
            branch1()
        } else if le(a, c) {
            branch2()
        } else {
            branch3()
        }
    } else if le(a, c) {
        branch4()
    } else if le(b, c) {
        branch5()
    } else {
        branch6()
    }
}

/// Re of the four Källén functions of the three-point sub-kinematics.
#[inline]
fn lambdas<T: BoxReal>(p: [Z<T>; 6]) -> [T; 4] {
    [(0, 1, 4), (1, 2, 5), (2, 3, 4), (3, 0, 5)].map(|(a, b, c)| {
        let x = p[c] - p[a] - p[b];
        (x * x - r(4.0) * p[a] * p[b]).re
    })
}

fn general<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    let lambda = lambdas(p);
    let sorted = sort_four(lambda[0], lambda[1], lambda[2], lambda[3]);
    let all_positive = lambda.iter().all(|l| !negative(*l));
    let permutation = if all_positive {
        PRM_B[sorted - 1]
    } else {
        PRM_A[sorted - 1]
    };
    let row = ORD[permutation - 1];
    let selected_m = core::array::from_fn(|slot| m[row[slot] - 1]);
    let selected_p = core::array::from_fn(|slot| p[row[slot + 4] - 1]);
    ordered(selected_p, selected_m, k)
}

/// `__olo_boxc_box`: the finite contour box for one mass/momentum labeling.
pub(super) fn boxc<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    const PERM: [[usize; 6]; 16] = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [3, 0, 1, 2, 5, 4],
        [0, 1, 2, 3, 4, 5],
        [2, 3, 0, 1, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [3, 0, 1, 2, 5, 4],
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 0, 5, 4],
        [1, 2, 3, 0, 5, 4],
        [3, 0, 1, 2, 5, 4],
        [1, 2, 3, 0, 5, 4],
        [2, 3, 0, 1, 4, 5],
        [2, 3, 0, 1, 4, 5],
        [3, 0, 1, 2, 5, 4],
        [0, 1, 2, 3, 4, 5],
    ];
    const CASE: [usize; 16] = [0, 1, 1, 2, 1, 5, 2, 3, 1, 2, 5, 3, 2, 3, 3, 4];
    let mask = (0..4).fold(0, |s, i| s + if nz(p[i]) { 1 << (3 - i) } else { 0 });
    let order = PERM[mask];
    let pp = order.map(|i| p[i]);
    let mm = core::array::from_fn(|i| m[order[i]]);
    if mask == 15 {
        general(pp, mm, k)
    } else if matches!(CASE[mask], 0 | 1 | 5) {
        opposite(pp, mm, k)
    } else {
        zero_external(pp, mm, k)
    }
}

/// `supported_kinematics` of the contour representation.
pub(super) fn supported_kinematics<T: BoxReal>(p: [Z<T>; 6]) -> bool {
    if !nz(p[4]) || !nz(p[5]) {
        return false;
    }
    if p[..4].iter().all(|v| nz(*v)) {
        lambdas(p).iter().any(|l| negative(-*l))
    } else {
        true
    }
}

/// Each ported definition against the registered `__olo_boxc_*` expression,
/// evaluated by Symbolica in binary64 on random (often degenerate) arguments.
#[cfg(test)]
mod expression_tests {
    use super::*;
    use crate::definitions::FunctionMap;
    use symbolica::atom::{Atom, AtomCore};
    use symbolica::domains::float::Complex;
    use symbolica::prelude::RealLike;

    // The binary64 port.
    type Z = super::Z<f64>;
    type Cont = super::Cont<f64>;
    type BetaData = super::BetaData<f64>;
    const ZERO: Z = Z { re: 0.0, im: 0.0 };
    const ONE: Z = Z { re: 1.0, im: 0.0 };

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
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn real(&mut self) -> f64 {
            match self.below(5) {
                0 => 0.0,
                1 => (self.below(9) as f64 - 4.0) / 2.0,
                _ => 4.0 * self.uniform() - 2.0,
            }
        }
        fn complex(&mut self) -> Z {
            let re = self.real();
            let im = if self.below(2) == 0 { 0.0 } else { self.real() };
            Z { re, im }
        }
    }

    fn evaluator(name: &str, arity: usize) -> impl FnMut(&[Z]) -> Z {
        let vars: Vec<Atom> = (0..arity)
            .map(|i| symbolica::symbol!(format!("handwritten_boxc_arg_{i}")).to_atom())
            .collect();
        let call = symbolica::symbol!(format!("__olo_boxc_{name}")).call(&vars);
        let mut map = FunctionMap::new();
        crate::sheet_exact::register(&mut map);
        crate::box_complex::register(&mut map);
        let mut evaluator = Atom::evaluator_multiple(&[call.as_view()], &vars)
            .function_map(map.into())
            .direct_translation(true)
            .build()
            .unwrap()
            .map_coeff(&|v| Complex::new(v.re.to_f64(), v.im.to_f64()));
        move |input: &[Z]| {
            let input: Vec<Complex<f64>> = input.iter().map(|z| z.c()).collect();
            let mut out = [Complex::new(0.0, 0.0)];
            evaluator.evaluate(&input, &mut out);
            Z::of(out[0])
        }
    }

    /// The registered definition at 256 bits (rounded to binary64).
    fn precise(name: &str, x: &[Z]) -> Z {
        use symbolica::domains::float::Float;
        let vars: Vec<Atom> = (0..x.len())
            .map(|i| symbolica::symbol!(format!("handwritten_boxc_arg_{i}")).to_atom())
            .collect();
        let call = symbolica::symbol!(format!("__olo_boxc_{name}")).call(&vars);
        let mut map = FunctionMap::new();
        crate::sheet_exact::register(&mut map);
        crate::box_complex::register(&mut map);
        let mut evaluator = Atom::evaluator_multiple(&[call.as_view()], &vars)
            .function_map(map.into())
            .direct_translation(true)
            .build()
            .unwrap()
            .map_coeff_with_prec(
                &|v| symbolica::domains::float::Complex::new(v.re.to_multi_prec_float(256), v.im.to_multi_prec_float(256)),
                256,
            );
        let input: Vec<_> = x
            .iter()
            .map(|z| symbolica::domains::float::Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
            .collect();
        let mut out = [symbolica::domains::float::Complex::new(Float::new(256), Float::new(256))];
        evaluator.evaluate(&input, &mut out);
        Z::new(out[0].re.to_f64(), out[0].im.to_f64())
    }

    fn cont(x: &[Z]) -> Cont {
        Cont { v: x[0], f: x[1], s: x[2] }
    }

    /// `allowed`: tolerated disagreements with the 256-bit expression, for a
    /// composition whose binary64 sheet tests can be decided by roundoff (the
    /// generated binary64 code shares them; tests/native_handwritten.rs checks
    /// the D0 accuracy against it).
    fn compare(name: &str, arity: usize, samples: usize, allowed: usize, port: impl Fn(&[Z]) -> Z, generate: &dyn Fn(&mut Rng) -> Vec<Z>) {
        let mut exact = evaluator(name, arity);
        let mut rng = Rng(0x9e3779b97f4a7c15 ^ arity as u64);
        let mut failures = vec![];
        let mut compared = 0;
        let mut rescued = 0;
        for _ in 0..samples {
            let x = generate(&mut rng);
            assert_eq!(x.len(), arity);
            let want = exact(&x);
            if !(want.re.is_finite() && want.im.is_finite()) {
                continue;
            }
            compared += 1;
            let got = port(&x);
            let abs = |z: Z| z.re.hypot(z.im);
            let error = abs(got - want) / abs(want).max(1e-300);
            if !(error < 1e-8) && abs(got - want) > 1e-10 {
                // Decide with the expression at 256 bits: near removable
                // singularities binary64 rounding can select another branch.
                let truth = precise(name, &x);
                if abs(got - truth) <= 1e-8 * abs(truth).max(1e-2) {
                    rescued += 1;
                } else {
                    failures.push(format!("{error:.2e}: {x:?}\n   port={got:?}\n   expr={want:?}\n   256-bit expr={truth:?}"));
                }
            }
        }
        eprintln!(
            "{name}: {compared} compared, {} failures; {rescued} binary64 expression values off the 256-bit expression where the port agrees with it",
            failures.len()
        );
        assert!(
            failures.len() <= allowed,
            "{name}: {} failures\n{}",
            failures.len(),
            failures[..failures.len().min(8)].join("\n")
        );
    }

    fn k() -> Constants<f64> {
        Constants::new(&0.0)
    }

    fn any(n: usize) -> impl Fn(&mut Rng) -> Vec<Z> {
        move |rng: &mut Rng| (0..n).map(|_| rng.complex()).collect()
    }

    /// Physical box arguments: real momenta, masses with nonpositive widths.
    fn kinematics(rng: &mut Rng) -> Vec<Z> {
        let mut x: Vec<Z> = (0..6).map(|_| r(rng.real() * 2.0)).collect();
        // The contour box is used with three or four massive propagators.
        let massless = if rng.below(3) == 0 { rng.below(4) } else { 4 };
        for slot in 0..4 {
            let re = match rng.below(3) {
                _ if slot == massless => 0.0,
                0 => 0.5,
                _ => 0.05 + rng.uniform(),
            };
            let im = if rng.below(3) == 0 { -0.05 * rng.uniform() } else { 0.0 };
            x.push(Z { re, im });
        }
        if rng.below(3) == 0 {
            let (i, j) = (rng.below(10) as usize, rng.below(10) as usize);
            if (i < 6) == (j < 6) {
                x[i] = x[j];
            }
        }
        x
    }

    fn split(x: &[Z]) -> ([Z; 6], [Z; 4]) {
        (core::array::from_fn(|i| x[i]), core::array::from_fn(|i| x[6 + i]))
    }

    /// `r1` of a far root (|z| = 1e2 to 1e17, real or complex, poles in the
    /// strip around the interval): the hand-written port and the binary64
    /// expression against the expression at 256 bits, within 1e-12 (below the
    /// series threshold, |z| ~ 140, the general form keeps about eps |z|). Its
    /// dilogarithm terms, taken next to the cut on different sheets, cancel by
    /// eps |z| (1e-6 at |z| = 1e9, O(1) at 1e15) unless `far_root_series` is
    /// used.
    #[test]
    fn far_root_r1_matches_high_precision() {
        crate::tests::large_stack(|| {
            let c = k();
            let mut exact = evaluator("r1", 10);
            let mut rng = Rng(0x5eed);
            let mut failures = vec![];
            for sample in 0..32 {
                let y = |rng: &mut Rng| Cont {
                    v: Z { re: 2.0 * rng.uniform() - 0.5, im: rng.uniform() - 0.5 },
                    f: Z { re: rng.uniform() - 0.5, im: rng.uniform() - 0.5 },
                    s: Z { re: rng.uniform() - 0.5, im: rng.uniform() - 0.5 },
                };
                let (y1, y2) = (y(&mut rng), y(&mut rng));
                let exponent = 2 + (sample % 16) as i32;
                let sign = if rng.below(2) == 0 { 1.0 } else { -1.0 };
                let big = sign * 10f64.powi(exponent) * (1.0 + rng.uniform());
                let im = if sample % 3 == 0 { big * (rng.uniform() - 0.5) } else { 0.0 };
                let zz = Cont { v: Z { re: big, im }, f: Z { re: 0.0, im: -big }, s: ZERO };
                let fy = r0(&y1, &y2, &c);
                let args = [zz.v, zz.f, zz.s, y1.v, y1.f, y1.s, y2.v, y2.f, y2.s, fy];
                let truth = precise("r1", &args);
                let abs = |z: Z| z.re.hypot(z.im);
                for (route, value) in [("port", r1(&zz, &y1, &y2, fy, &c)), ("expression", exact(&args))] {
                    let error = abs(value - truth) / abs(truth);
                    if !(error <= 1e-12) {
                        failures.push(format!("{route} |z| = {:.1e}: {error:.1e} ({value:?} vs {truth:?})", abs(zz.v)));
                    }
                }
            }
            assert!(failures.is_empty(), "{failures:#?}");
        });
    }

    #[test]
    fn contour_box_definitions_match_expressions() {
        crate::tests::large_stack(|| {
            let c = k();
            compare("root_1", 3, 3000, 0, |x| roots(x[0], x[1], x[2])[0], &any(3));
            compare("root_2", 3, 3000, 0, |x| roots(x[0], x[1], x[2])[1], &any(3));
            compare("r0", 6, 3000, 0, |x| r0(&cont(&x[0..3]), &cont(&x[3..6]), &c), &any(6));
            compare("r1", 10, 3000, 0, |x| r1(&cont(&x[0..3]), &cont(&x[3..6]), &cont(&x[6..9]), x[9], &c), &any(10));
            // `r0` given `delta = y1 - y2`.
            compare("r0_line", 7, 3000, 0, |x| r0_line(&cont(&x[0..3]), &cont(&x[3..6]), x[6], &c), &|rng: &mut Rng| {
                let mut x = any(6)(rng);
                x.push(x[0] - x[3]);
                x
            });
            // The line variants take the product (y1 - z)(y2 - z) of the
            // poles' offsets; a third of the samples put a pole within 1e-6
            // of the root, where the product replaces its offset.
            let conic = [ZERO, ZERO, ONE, ZERO, ZERO, ZERO];
            let data = |product: Z| BetaData { a: product, b: ZERO, c: ZERO, dc: ZERO, k: ONE, y: [Cont { v: ZERO, f: ZERO, s: ZERO }; 2] };
            fn given<'a>(data: &'a BetaData, conic: &'a [Z; 6]) -> LineProduct<'a, f64> {
                LineProduct { line: [ZERO, ONE, ONE, ZERO, ONE], ratio: [ZERO, ONE], conic, data, a1: data.a, probe: None }
            }
            let near = |rng: &mut Rng, root: Z| root + Z { re: 1e-6 * rng.real(), im: 1e-6 * rng.real() };
            compare(
                "r1_line",
                11,
                3000,
                0,
                |x| r1_with(&cont(&x[0..3]), &cont(&x[3..6]), &cont(&x[6..9]), x[9], Some(&given(&data(x[10]), &conic)), &c),
                &|rng: &mut Rng| {
                    let mut x = any(10)(rng);
                    if rng.below(3) == 0 {
                        x[3] = near(rng, x[0]);
                    }
                    x.push((x[3] - x[0]) * (x[6] - x[0]));
                    x
                },
            );
            compare(
                "s3_line",
                13,
                3000,
                0,
                |x| {
                    // `over` maps the product by 1/e², the expression takes it mapped.
                    let data = data(x[12] * x[7] * x[7]);
                    let line = LineProduct { line: [x[6], x[7], x[8], x[9], x[11]], ..given(&data, &conic) };
                    s3_line(&cont(&x[0..3]), &cont(&x[3..6]), x[10], &line, true, &c)
                },
                &|rng: &mut Rng| {
                    let mut x = any(11)(rng);
                    let (d, e, b) = (x[6], x[7], x[8]);
                    // far = b + c, or a massless end (c = -b, root 1).
                    let massless = rng.below(4) == 0;
                    if massless {
                        x[9] = -b;
                    }
                    let far = if massless { ZERO } else { b + x[9] };
                    let root = if nz(far) { -dv(x[9], b) } else { ONE };
                    if rng.below(3) == 0 {
                        // A pole whose image (d + y)/e lies next to the root.
                        x[0] = near(rng, e * root - d);
                    }
                    let mapped = |y: Z| dv(d + y, e) - root;
                    x.extend([far, mapped(x[0]) * mapped(x[3])]);
                    x
                },
            );
            compare("kallen", 3, 3000, 0, |x| kallen(x[0], x[1], x[2]), &any(3));
            // A quadratic's two-point subgraph [lambda, p, near, far]; often a
            // massless end. `s3` takes the roots of re(a) x² + b x + c (a is real
            // for every edge of the box), so lambda is their discriminant.
            compare(
                "s3",
                17,
                3000,
                0,
                |x| {
                    let edge = [x[13], x[14], x[15], x[16]];
                    s3(&cont(&x[0..3]), &cont(&x[3..6]), x[6], x[7], x[8], x[9], x[10], x[11], x[12], edge, &c)
                },
                &|rng: &mut Rng| {
                    let mut x = any(13)(rng);
                    let (a, b) = (x[8], x[9]);
                    let end = rng.below(4);
                    if end == 0 {
                        x[10] = -(a + b);
                    } else if end == 1 {
                        x[10] = ZERO;
                    }
                    let c = x[10];
                    x.extend([b * b - r(4.0) * r(a.re) * c, a, c, if end == 0 { ZERO } else { a + b + c }]);
                    x
                },
            );
            compare(
                "planar",
                13,
                2000,
                0,
                |x| planar(&cont(&x[0..3]), &cont(&x[3..6]), nz(x[6]), nz(x[7]), x[8], x[9], x[10], x[11], nz(x[12]), &c),
                &|rng: &mut Rng| {
                    let mut x = any(13)(rng);
                    x[6] = r(rng.below(2) as f64);
                    x[7] = r(rng.below(2) as f64);
                    x[12] = r(rng.below(2) as f64);
                    x
                },
            );
            // `a + c` and the discriminants and values at x = 1 of the edge
            // quadratics, which the callers form exactly; the first 14 keep the
            // random stream of the 14-argument comparison.
            let stream = std::cell::RefCell::new(Rng(0x9e3779b97f4a7c15 ^ 14));
            compare("t1", 22, 1500, 0, |x| t1(core::array::from_fn(|i| x[i]), &c), &|_| {
                let mut x = any(14)(&mut stream.borrow_mut());
                let [a, c, g, h, d, f, j, dpe] = [x[0], x[1], x[2], x[3], x[4], x[6], x[7], x[8]];
                let ac = a + c;
                x.extend([ac, dpe * dpe - r(4.0) * r(ac.re) * f, ac + dpe + f, d * d - r(4.0) * r(a.re) * f, a + d + f]);
                x.extend([g + h, g + j, g + h + j]);
                x
            });
            compare("t13", 26, 1500, 0, |x| t13(core::array::from_fn(|i| x[i]), &c), &|rng: &mut Rng| {
                let mut x = any(18)(rng);
                let [a, c, g, h, e, f, j, dpf] = [x[0], x[1], x[2], x[3], x[5], x[6], x[7], x[10]];
                let far_ae = a + e + c + dpf;
                let far_gj = g + j + h + dpf;
                // dpe and dpj consistent with the conic's corner values.
                let dpe = far_ae - a - c - f;
                let dpj = far_gj - g - h - f;
                x[8] = dpe;
                x[9] = dpj;
                x.extend([
                    (e + c) * (e + c) - r(4.0) * r(a.re) * dpf,
                    (j + h) * (j + h) - r(4.0) * r(g.re) * dpf,
                    dpj * dpj - r(4.0) * r((g + h).re) * f,
                    dpe * dpe - r(4.0) * r((a + c).re) * f,
                    far_ae,
                    far_gj,
                    g + h,
                    a + c,
                ]);
                x
            });
            // Physical structure: real momentum combinations, masses with
            // nonpositive widths, imaginary infinitesimal directions.
            let _structured = |rng: &mut Rng| {
                let mut x: Vec<Z> = (0..5).map(|_| r(rng.real())).collect();
                for _ in 5..11 {
                    let mut z = r(rng.real());
                    if rng.below(3) == 0 {
                        z.im = -0.2 * rng.uniform();
                    }
                    x.push(z);
                }
                for _ in 11..17 {
                    x.push(Z { re: 0.0, im: rng.real() });
                }
                x
            };
            // tfun alone is not checked on random arguments: with a real
            // contour parameter its two candidate betas need not agree for
            // inconsistent random widths, and a tie |beta1| = |beta2| is broken
            // by rounding. The physical box compositions below are checked.
            // The charts are composed by `box` (each is only used inside its
            // own external-zero sector); compare on supported kinematics.
            compare(
                "box",
                10,
                3000,
                3,
                |x| {
                    let (p, m) = split(x);
                    boxc(p, m, &c)
                },
                &|rng: &mut Rng| loop {
                    let x = kinematics(rng);
                    if supported_kinematics(split(&x).0) {
                        break x;
                    }
                },
            );
        });
    }
}
