//! D0 dispatch for the hand-written box sectors (binary64 and double-double; `box_series`,
//! `select_four_series`, `box_finite_four_masses`, `box_four_masses_chart`,
//! `box_four_masses_dns`, `finite_box_representation`).
use super::box_contour::{boxc, negative, supported_kinematics};
use super::real::{BoxReal, Z, dd, dv, nz, physical_sqrt, r, r_function, rv, zero, zlower, zsheet, SheetZ};
use super::sheet::Sheet;
use super::super::primitives::NativeFloat;
use super::Constants;

/// `triangle::r_function_velocities`: `[D(r), D(d)]` of each r-function for
/// a velocity `D(k)`, and the weight of the first-order velocities (zero when
/// an r-function sits at a threshold, where it moves like `sqrt(eps)`).
fn r_function_velocities<T: BoxReal>(functions: [[Z<T>; 4]; 2]) -> [[Z<T>; 2]; 3] {
    let singular = functions.map(|[_, _, d, dk]| !nz(d) && nz(dk));
    let any_singular = singular.iter().any(|&s| s);
    let velocities: [[Z<T>; 2]; 2] = core::array::from_fn(|index| {
        let [k, r_, d, dk] = functions[index];
        if any_singular {
            if singular[index] {
                let step = -dk.re.libm_hypot(dk.im).sqrt();
                [rv(step), rv(T::lit(2.0) * step)]
            } else {
                [r(0.0), r(0.0)]
            }
        } else if nz(d) {
            [dv(r_ * dk, d), dv(k * dk, d)]
        } else {
            [r(0.0), r(0.0)]
        }
    });
    let weight = r(if any_singular { 0.0 } else { 1.0 });
    [velocities[0], velocities[1], [weight, weight]]
}

/// `triangle::root_lip_or`: the lip of a real root `x` of `a x² - b x + c`
/// from its first-order shift `D(x)`, or `fallback` when that vanishes.
fn root_lip_or<T: BoxReal>(x: Z<T>, a: Z<T>, b: Z<T>, [da, db, dc]: [Z<T>; 3], fallback: i32) -> i32 {
    let numerator = db * x - da * x * x - dc;
    let slope = r(2.0) * a * x - b;
    let shift = (numerator
        * Z {
            re: slope.re,
            im: -slope.im,
        })
    .re;
    if !shift.is_fully_zero() {
        if negative(shift) { -1 } else { 1 }
    } else {
        fallback
    }
}

/// [root_lip_or] and the root's velocity `D(x)` (zero at a double root).
fn root_lip_and_velocity<T: BoxReal>(x: Z<T>, a: Z<T>, b: Z<T>, [da, db, dc]: [Z<T>; 3], fallback: i32) -> (i32, Z<T>) {
    let numerator = db * x - da * x * x - dc;
    let slope = r(2.0) * a * x - b;
    let shift = (numerator
        * Z {
            re: slope.re,
            im: -slope.im,
        })
    .re;
    let lip = if !shift.is_fully_zero() {
        if negative(shift) { -1 } else { 1 }
    } else {
        fallback
    };
    (lip, if nz(slope) { dv(numerator, slope) } else { r(0.0) })
}

/// `box_four_masses_dns` (OneLOop boxf4, Denner-Nierste-Scharf) as it stands,
/// for complex masses and for real masses without an invariant at or below its
/// pseudo-threshold.
fn four_masses_dns<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    let roots = m.map(|x| physical_sqrt(&x));
    let kk = |mi: usize, mj: usize, momentum: Z<T>| dv(m[mi] + m[mj] - momentum, roots[mi] * roots[mj]);
    let k12 = kk(0, 1, p[0]);
    let k13 = kk(0, 2, p[4]);
    let k14 = kk(0, 3, p[3]);
    let k23 = kk(1, 2, p[1]);
    let k24 = kk(1, 3, p[5]);
    let k34 = kk(2, 3, p[2]);
    let (r12, _) = r_function(&k12, k);
    let (r13, d13) = r_function(&k13, k);
    let (r14, _) = r_function(&k14, k);
    let (r23, _) = r_function(&k23, k);
    let (r24, d24) = r_function(&k24, k);
    let (r34, _) = r_function(&k34, k);
    let a = dv(k34, r24) + r13 * k12 - dv(k14 * r13, r24) - k23;
    if !nz(a) {
        return zero();
    }
    let b = d13 * d24 + k12 * k34 - k14 * k23;
    let c = dv(k12, r13) + r24 * k34 - dv(k14 * r24, r13) - k23;
    let discriminant = physical_sqrt(&(b * b - r(4.0) * a * c));
    let two_a = r(2.0) * a;
    let x1 = dv(b + discriminant, two_a);
    let x2 = dv(b - discriminant, two_a);
    let base = (k23 - r13 * k12 - r24 * k34 + r13 * r24 * k14).re * a.re;
    let sign = |x: T| if negative(x) { -1 } else { 1 };
    // Real roots inherit their lips from OneLOop's p -> p(1 + i eps); its
    // explicit sign applies only when that first-order shift vanishes.
    let lip = |root: Z<T>, other: Z<T>| {
        if !root.im.is_fully_zero() || !negative(root.re) {
            // Sheet::with_sign ignores the lip off the negative real axis.
            return 1;
        }
        let dk = |momentum: Z<T>, i: usize, j: usize| -dv(momentum, roots[i] * roots[j]);
        let (dk13, dk24) = (dk(p[4], 0, 2), dk(p[5], 1, 3));
        let [[dr13, dd13], [dr24, dd24], weight] =
            r_function_velocities([[k13, r13, d13, dk13], [k24, r24, d24, dk24]]);
        let weight = weight[0];
        let [dk12, dk14, dk23, dk34] = [
            dk(p[0], 0, 1),
            dk(p[3], 0, 3),
            dk(p[1], 1, 2),
            dk(p[2], 2, 3),
        ]
        .map(|v| v * weight);
        let r24_2 = r24 * r24;
        let r13_2 = r13 * r13;
        let da = dv(dk34, r24) - dv(k34 * dr24, r24_2) + dr13 * k12 + r13 * dk12
            - dv(dk14 * r13, r24)
            - dv(k14 * dr13, r24)
            + dv(k14 * r13 * dr24, r24_2)
            - dk23;
        let db = dd13 * d24 + d13 * dd24 + dk12 * k34 + k12 * dk34 - dk14 * k23 - k14 * dk23;
        let dc = dv(dk12, r13) - dv(k12 * dr13, r13_2) + dr24 * k34 + r24 * dk34
            - dv(dk14 * r24, r13)
            - dv(k14 * dr24, r13)
            + dv(k14 * r24 * dr13, r13_2)
            - dk23;
        root_lip_or(root, a, b, [da, db, dc], sign(base * other.re))
    };
    let qx1 = zsheet(x1, lip(x1, x2));
    let qx2 = zsheet(x2, lip(x2, x1));
    let [q12, q13, q14, q23, q24, q34] = [r12, r13, r14, r23, r24, r34].map(zlower);
    let y1 = qx1.div(&q24);
    let y2 = qx2.div(&q24);
    let mut finite = dv(
        dd(&y1.mul(&q12), &y2.mul(&q12), k) * r12 + dv(dd(&y1.div(&q12), &y2.div(&q12), k), r12),
        r24,
    );
    let ratio = dv(r13, r24);
    let qr = zsheet(ratio, -sign(r24.re));
    let z1 = qx1.mul(&qr);
    let z2 = qx2.mul(&qr);
    finite = finite
        - (dd(&z1.mul(&q23), &z2.mul(&q23), k) * r23 + dv(dd(&z1.div(&q23), &z2.div(&q23), k), r23))
            * ratio;
    let w1 = qx1.mul(&q13);
    let w2 = qx2.mul(&q13);
    finite = finite
        + (dd(&w1.mul(&q34), &w2.mul(&q34), k) * r34 + dv(dd(&w1.div(&q34), &w2.div(&q34), k), r34))
            * r13;
    finite = finite
        - dd(&qx1.mul(&q14), &qx2.mul(&q14), k) * r14
        - dv(dd(&qx1.div(&q14), &qx2.div(&q14), k), r14);
    -dv(finite, a * roots[0] * roots[1] * roots[2] * roots[3])
}

/// `sign_nonnegative`, or `fallback` for an exact zero.
#[inline(always)]
fn sign_or<T: BoxReal>(x: T, fallback: i32) -> i32 {
    if !x.is_fully_zero() { if negative(x) { -1 } else { 1 } } else { fallback }
}

/// `box_four_masses_chart`: boxf4 for four real masses with `r24 >= 1`
/// (Denner-Dittmaier eq. (3.61)): `p23` static, principal `x r13`, and the
/// explicit `eta(x, r13) Log R` term, with every decision of the expression.
/// `roots` are the masses' square roots and `k24` the selecting r-function
/// argument, as [four_masses] formed them.
fn four_masses_chart<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], roots: [Z<T>; 4], k24: Z<T>, k: &Constants<T>) -> Z<T> {
    let kk = |mi: usize, mj: usize, momentum: Z<T>| dv(m[mi] + m[mj] - momentum, roots[mi] * roots[mj]);
    let k12 = kk(0, 1, p[0]);
    let k13 = kk(0, 2, p[4]);
    let k14 = kk(0, 3, p[3]);
    let k23 = kk(1, 2, p[1]);
    let k34 = kk(2, 3, p[2]);
    let (r12, _) = r_function(&k12, k);
    let (r13, d13) = r_function(&k13, k);
    let (r14, _) = r_function(&k14, k);
    let (r23, _) = r_function(&k23, k);
    let (r24, d24) = r_function(&k24, k);
    let (r34, _) = r_function(&k34, k);
    let a = dv(k34, r24) + r13 * k12 - dv(k14 * r13, r24) - k23;
    if !nz(a) {
        return zero();
    }
    let b = d13 * d24 + k12 * k34 - k14 * k23;
    let c = dv(k12, r13) + r24 * k34 - dv(k14 * r24, r13) - k23;
    let square = b * b - r(4.0) * a * c;
    let discriminant = physical_sqrt(&square);
    let two_a = r(2.0) * a;
    let x1 = dv(b + discriminant, two_a);
    let x2 = dv(b - discriminant, two_a);
    // First-order velocities of p -> p (1 + i eps) with p23 static, needed
    // for real roots and for a real R.
    let dk = |momentum: Z<T>, i: usize, j: usize| -dv(momentum, roots[i] * roots[j]);
    let [[dr13, dd13], _, weight] = r_function_velocities([
        [k13, r13, d13, dk(p[4], 0, 2)],
        [k24, r24, d24, r(0.0)],
    ]);
    let weight = weight[0];
    // At a threshold r13 = r0 = -1 or 1 the root moves along (r0 - i) sqrt(eps)
    // (box_four_masses_chart): D(r13) is proportional to -(1 + i r0).
    let tilt = if weight.re.is_fully_zero() { Z::new(T::lit(1.0), r13.re) } else { r(1.0) };
    let [dr13, dd13] = [dr13 * tilt, dd13 * tilt];
    let velocities = || {
        let [dk12, dk14, dk23, dk34] = [
            dk(p[0], 0, 1),
            dk(p[3], 0, 3),
            dk(p[1], 1, 2),
            dk(p[2], 2, 3),
        ]
        .map(|v| v * weight);
        let r13_2 = r13 * r13;
        let da = dv(dk34, r24) + dr13 * k12 + r13 * dk12
            - dv(dk14 * r13, r24)
            - dv(k14 * dr13, r24)
            - dk23;
        let db = dd13 * d24 + dk12 * k34 + k12 * dk34 - dk14 * k23 - k14 * dk23;
        let dc = dv(dk12, r13) - dv(k12 * dr13, r13_2) + r24 * dk34 - dv(dk14 * r24, r13)
            + dv(k14 * r24 * dr13, r13_2)
            - dk23;
        [da, db, dc, dk23, dk34]
    };
    let sign = |x: T| if negative(x) { -1 } else { 1 };
    // r13 is real for |k13| >= 2 and on the unit circle for |k13| <= 2.
    let r13_discriminant = k13.re * k13.re - T::lit(4.0);
    let r13_real = !negative(r13_discriminant);
    let r13_circle = !negative(-r13_discriminant);
    let below = negative(square.re);
    let x_real = r13_real && !below;
    let ratio_real = x_real || (r13_circle && below);
    let mut cache = None;
    let mut lips = [1, 1];
    // A positive real root times a positive r13 has neither a lip nor a
    // winding (eta vanishes for any of their directions): skip their velocities.
    let positive = |x: Z<T>| x_real && !negative(x.re) && !negative(r13.re);
    // D(x) of the real roots that need their lip.
    let mut speeds = [r(0.0), r(0.0)];
    if x_real && !(positive(x1) && positive(x2)) {
        let [da, db, dc, ..] = *cache.get_or_insert_with(velocities);
        let base = (k23 - r13 * k12 - r24 * k34 + r13 * r24 * k14).re * a.re;
        for (index, (root, other)) in [(x1, x2), (x2, x1)].into_iter().enumerate() {
            if !positive(root) {
                let fallback = sign(base * other.re);
                (lips[index], speeds[index]) =
                    root_lip_and_velocity(root, a, b, [da, db, dc], fallback);
            }
        }
    }
    // The direction of r13 (zero for a static p12, where eta vanishes).
    let s13 = if r13_real { sign_or(dr13.re, 0) } else { sign(r13.im) };
    let mut principal = [zsheet(r(0.0), 1), zsheet(r(0.0), 1)];
    let mut winding = [r(0.0), r(0.0)];
    let mut windings = [0, 0];
    // r24 >= 1 is exactly real: divide by it as a real number.
    let inverse24 = T::lit(1.0) / r24.re;
    let scale = |z: Z<T>| Z::new(z.re * inverse24, z.im * inverse24);
    for (index, (x, lip)) in [(x1, lips[0]), (x2, lips[1])].into_iter().enumerate() {
        let v = x * r13;
        // D(x r13), from the root's velocity (computed above for real roots).
        let root_velocity = |[da, db, dc, ..]: [Z<T>; 5]| {
            let dx = if x_real {
                speeds[index]
            } else {
                let slope = r(2.0) * a * x - b;
                if nz(slope) { dv(db * x - da * x * x - dc, slope) } else { r(0.0) }
            };
            dx * r13 + x * dr13
        };
        // The lip of the principal value x r13, real exactly with x.
        let lv = if positive(x) {
            1
        } else if x_real {
            let dvx = root_velocity(*cache.get_or_insert_with(velocities));
            sign_or(dvx.re, lip * sign(r13.re))
        } else {
            sign(v.im)
        };
        // eta(x, r13) / (2 pi i) from the three imaginary-part signs.
        let sx = if x_real { lip } else { sign(x.im) };
        let n = if positive(x) { 0 } else { -sx * s13 * s13 * (1 + sx * s13) * (1 - sx * lv) / 4 };
        if n != 0 {
            // R = P(x) of the r13 terms; its side, when real, from its velocity.
            let u = scale(v);
            let numerator = r(1.0) - u * k23 + u * u;
            let denominator = r(1.0) - v * k34 + v * v;
            let ratio = dv(numerator, denominator);
            let side = if ratio_real && negative(ratio.re) {
                let speeds = *cache.get_or_insert_with(velocities);
                let [.., dk23, dk34] = speeds;
                let dvx = root_velocity(speeds);
                let du = scale(dvx);
                let dnumerator = -(du * k23) - u * dk23 + r(2.0) * u * du;
                let ddenominator = -(dvx * k34) - v * dk34 + r(2.0) * v * dvx;
                let dratio = ratio * (dv(dnumerator, numerator) - dv(ddenominator, denominator));
                sign_or(dratio.re, 1)
            } else {
                1
            };
            let value = if ratio_real { rv(ratio.re) } else { ratio };
            winding[index] = r(n as f64) * zsheet(value, side).zlog(k);
            windings[index] = n;
        }
        principal[index] = zsheet(v, lv);
    }
    let qx1 = zsheet(x1, lips[0]);
    let qx2 = zsheet(x2, lips[1]);
    let [q12, q14, q23, q24, q34] = [r12, r14, r23, r24, r34].map(zlower);
    let y1 = qx1.div(&q24);
    let y2 = qx2.div(&q24);
    let mut finite = scale(
        dd(&y1.mul(&q12), &y2.mul(&q12), k) * r12 + dv(dd(&y1.div(&q12), &y2.div(&q12), k), r12),
    );
    let ratio = scale(r13);
    let [z1, z2] = principal.clone().map(|s| Sheet {
        v: scale(Z::of(s.v)).c(),
        phase: s.phase,
        odd: s.odd,
    });
    finite = finite
        - (dd(&z1.mul(&q23), &z2.mul(&q23), k) * r23 + dv(dd(&z1.div(&q23), &z2.div(&q23), k), r23))
            * ratio;
    let [w1, w2] = &principal;
    finite = finite
        + (dd(&w1.mul(&q34), &w2.mul(&q34), k) * r34 + dv(dd(&w1.div(&q34), &w2.div(&q34), k), r34))
            * r13;
    finite = finite
        - dd(&qx1.mul(&q14), &qx2.mul(&q14), k) * r14
        - dv(dd(&qx1.div(&q14), &qx2.div(&q14), k), r14);
    // The explicit term, -sum_k (-1)^(k+1) eta_k Log R_k / (x1 - x2).
    if windings != [0, 0] {
        let gap = x1 - x2;
        let explicit = if nz(gap) {
            dv(winding[0] - winding[1], gap)
        } else {
            // d Log R / dx at a double root (x1 = x2).
            let v = x1 * r13;
            let u = scale(v);
            let numerator = r(1.0) - u * k23 + u * u;
            let denominator = r(1.0) - v * k34 + v * v;
            let slope = dv(r(2.0) * u - k23, r24 * numerator) - dv(r(2.0) * v - k34, denominator);
            r(windings[0] as f64) * r13 * slope
        };
        finite = finite - Z::new(T::lit(0.0), T::lit(2.0) * k.pi) * explicit;
    }
    -dv(finite, a * roots[0] * roots[1] * roots[2] * roots[3])
}

/// `box_finite_four_masses`' choice of the chart: the first invariant with a
/// real r-function of at least one (real `k >= 2`) in the `p23` slot of
/// [four_masses_chart], or OneLOop's boxf4 when there is none or a mass is
/// complex.
fn four_masses<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4], k: &Constants<T>) -> Z<T> {
    if m.iter().all(|x| x.im.is_fully_zero()) {
        let roots = m.map(|x| physical_sqrt(&x));
        for (order, masses) in crate::box_integral::FOUR_MASS_CHART_ORDERS {
            let kv = dv(
                m[masses[1]] + m[masses[3]] - p[order[5]],
                roots[masses[1]] * roots[masses[3]],
            );
            if !negative(kv.re * kv.re - T::lit(4.0)) && !negative(kv.re) {
                let (p, m, roots) = (order.map(|i| p[i]), masses.map(|i| m[i]), masses.map(|i| roots[i]));
                return four_masses_chart(p, m, roots, kv, k);
            }
        }
    }
    four_masses_dns(p, m, k)
}

/// `box_series`: exchange the channel pair when s12 or s23 vanishes exactly.
#[inline]
fn hard_kinematics<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4]) -> ([Z<T>; 6], [Z<T>; 4]) {
    if nz(p[4]) && nz(p[5]) {
        return (p, m);
    }
    let abs = p.map(|v| <T as NativeFloat>::hypot(&v.re, &v.im));
    let min = |a: T, b: T| if negative(a - b) { a } else { b };
    let min13 = min(abs[0], abs[2]);
    let min24 = min(abs[1], abs[3]);
    let min56 = min(abs[4], abs[5]);
    if negative(min24 - min13) && negative(min56 - min13) {
        ([4, 1, 5, 3, 0, 2].map(|i| p[i]), [0, 2, 1, 3].map(|i| m[i]))
    } else if negative(min13 - min24) && negative(min56 - min24) {
        ([0, 5, 2, 4, 3, 1].map(|i| p[i]), [0, 1, 3, 2].map(|i| m[i]))
    } else {
        (p, m)
    }
}

/// `finite_box_representation`: OneLOop's contour representation for complex
/// masses and these timelike configurations, where it supports the kinematics.
#[inline]
fn use_contour<T: BoxReal>(p: [Z<T>; 6], m: [Z<T>; 4]) -> bool {
    let complex_mass = m.iter().any(|x| !x.im.is_fully_zero());
    let all_external_nonnegative = p[..4].iter().all(|x| !negative(x.re));
    let channels_nonnegative = !negative(p[4].re) && !negative(p[5].re);
    (complex_mass || all_external_nonnegative || channels_nonnegative) && supported_kinematics(p)
}

/// The finite part of a covered D0 sector, or `None` for the generated code.
pub(super) fn d0<T: BoxReal>(a: &[Z<T>], k: &Constants<T>) -> Option<Z<T>> {
    let p: [Z<T>; 6] = core::array::from_fn(|i| a[i]);
    let m: [Z<T>; 4] = core::array::from_fn(|i| a[6 + i]);
    let (p, m) = hard_kinematics(p, m);
    let sector = (0..4).fold(0, |s, i| s + if nz(m[i]) { 8 >> i } else { 0 });
    match sector {
        15 => {
            // box_finite_four_masses
            let nonvacuum = p.iter().any(|x| nz(*x)) || m[1..].iter().any(|x| nz(*x - m[0]));
            if !nonvacuum {
                return Some(dv(r(1.0), r(6.0) * m[0] * m[0]));
            }
            Some(if use_contour(p, m) {
                boxc(p, m, k)
            } else {
                four_masses(p, m, k)
            })
        }
        7 | 11 | 13 | 14 => {
            // box_three_masses_ir in the canonical labeling with m1 = 0.
            let shift = match sector {
                7 => 0,
                11 => 1,
                13 => 2,
                _ => 3,
            };
            const ORDER: [[usize; 6]; 4] = [
                [0, 1, 2, 3, 4, 5],
                [1, 2, 3, 0, 5, 4],
                [2, 3, 0, 1, 4, 5],
                [3, 0, 1, 2, 5, 4],
            ];
            let p = ORDER[shift].map(|i| p[i]);
            let m: [Z<T>; 4] = core::array::from_fn(|i| m[(i + shift) % 4]);
            // The infrared sector (both massless-adjacent legs on shell) and
            // the non-contour finite formula use the generated code, as does
            // a soft line with one leg and the channel s12 on shell, which
            // the generated code evaluates in another labelling.
            let finite = if nz(p[0] - m[1]) {
                nz(p[3] - m[3]) || nz(p[4] - m[2])
            } else {
                nz(p[3] - m[3]) && nz(p[4] - m[2])
            };
            // `finite_three_mass_representation`: the contour wherever it
            // supports the kinematics.
            (finite && supported_kinematics(p)).then(|| boxc(p, m, k))
        }
        _ => None,
    }
}
