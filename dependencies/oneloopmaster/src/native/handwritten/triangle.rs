//! C0 with three nonzero internal masses (`register_triangles` sector 7).
use super::super::primitives::{C, NativeFloat, inverse, mul, sqrt};
use super::sheet::{Sheet, dd};
use super::{
    Constants, add, div, is_zero, neg, negative, physical_sqrt, r_function, r_function_real, real, scale,
    sign_nonnegative, sub,
};

/// The finite part of sector 7, or `None` where the generated code is used.
pub(crate) fn three_masses<T: NativeFloat>(
    p: [&C<T>; 3],
    m: [&C<T>; 3],
    c: &Constants<T>,
) -> Option<C<T>> {
    if p.iter().all(|z| is_zero(z)) {
        return None;
    }
    let lambda = kallen(p, c);
    if negative(&lambda) {
        return Some(hooft_veltman(p, m, c));
    }
    Some(select(p, m, c))
}

/// Sign of the imaginary part of `z`, or `real_axis` when it vanishes.
#[inline]
fn im_sign<T: NativeFloat>(z: &C<T>, real_axis: i32) -> i32 {
    if z.im.is_fully_zero() {
        real_axis
    } else {
        sign_nonnegative(&z.im)
    }
}

/// `eta(sa, sb, sc) / (i pi)` of `triangle_hv::eta`, an integer.
#[inline]
fn eta(sa: i32, sb: i32, sc: i32) -> i32 {
    sc * (1 + sa * sb) * (1 - sa * sc) / 2
}

/// `triangle_hv::s3`: int_0^1 [log(a y'^2 + b y' + c) - log(a y^2 + b y + c)]/(y' - y)
/// with b = m1 - m2 - a and c = m2, in the 't Hooft--Veltman form, with the
/// same root lips and eta continuation terms as the exact expression.
fn s3<T: NativeFloat>(a: &C<T>, m1: &C<T>, m2: &C<T>, y: &C<T>, z: &C<T>, c: &Constants<T>) -> C<T> {
    let b = sub(&sub(m1, m2), a);
    let disc = sqrt(&sub(&mul(&b, &b), &scale(&mul(a, m2), &c.four)));
    let inverse_two_a = inverse(&scale(a, &c.two));
    let roots = [
        mul(&sub(&disc, &b), &inverse_two_a),
        mul(&neg(&add(&b, &disc)), &inverse_two_a),
    ];
    let lips = [-1, 1];
    let term = |r: &C<T>, lip: i32| {
        let qy = Sheet::with_sign(sub(y, r), lip);
        sub(
            &Sheet::with_sign(neg(r), lip).div(&qy).dilog(c),
            &Sheet::with_sign(sub(&c.one_c, r), lip).div(&qy).dilog(c),
        )
    };
    let below_a = -sign_nonnegative(&a.re);
    let z_sign = im_sign(z, below_a);
    let m2_over_a = mul(m2, &inverse(a));
    let correction = eta(
        im_sign(&neg(&roots[0]), lips[0]),
        im_sign(&neg(&roots[1]), lips[1]),
        im_sign(&m2_over_a, below_a),
    ) - eta(
        im_sign(&sub(y, &roots[0]), lips[0]),
        im_sign(&sub(y, &roots[1]), lips[1]),
        z_sign,
    ) - if negative(&a.re) { 1 - z_sign } else { 0 };
    let mut result = add(&term(&roots[0], lips[0]), &term(&roots[1], lips[1]));
    if correction != 0 {
        let log = Sheet::upper(mul(&sub(y, &c.one_c), &inverse(y))).log(c);
        let factor = c.pi.clone() * &c.from_int(correction);
        result = add(&result, &C::new(-(log.im.clone() * &factor), log.re.clone() * &factor));
    }
    result
}

/// `triangle_hv::s3_call`, the root `i sqrt(-lambda)` on the upper lip.
fn s3_call<T: NativeFloat>(
    a: &C<T>,
    m1: &C<T>,
    m2: &C<T>,
    t1: &C<T>,
    t2: &C<T>,
    lambda: &T,
    lambda_root: &C<T>,
    c: &Constants<T>,
) -> C<T> {
    let inverse_two_a = inverse(&scale(a, &c.two));
    let alpha = mul(&add(&neg(t2), lambda_root), &inverse_two_a);
    let b = sub(&sub(m1, m2), a);
    let y = neg(&mul(&add(t1, &mul(&b, &alpha)), &inverse(lambda_root)));
    let u = sub(t1, &mul(&mul(&b, t2), &inverse_two_a));
    let inverse_a = inverse(a);
    let z = sub(
        &add(&scale(&mul(&u, &u), &(c.one.clone() / lambda)), &mul(m2, &inverse_a)),
        &scale(&mul(&mul(&b, &b), &mul(&inverse_a, &inverse_a)), &(c.one.clone() / &c.four)),
    );
    s3(a, m1, m2, &y, &z, c)
}

/// `triangle_hv::expression`: the three-mass finite part for a negative
/// Källén function, after the same edge sorting by squared invariants.
fn hooft_veltman<T: NativeFloat>(p: [&C<T>; 3], m: [&C<T>; 3], c: &Constants<T>) -> C<T> {
    let mut p = p.map(Clone::clone);
    let mut m = m.map(Clone::clone);
    for (a, b, cc, d) in [(0, 1, 0, 2), (1, 2, 0, 1), (0, 1, 0, 2)] {
        let swap = negative(&(p[b].re.clone() * &p[b].re - p[a].re.clone() * &p[a].re));
        if swap {
            p.swap(a, b);
            m.swap(cc, d);
        }
    }
    let lambda = kallen([&p[0], &p[1], &p[2]], c);
    let lambda_root = C::new(c.one.zero(), (-lambda.clone()).sqrt());
    let mass_difference = sub(&m[0], &m[1]);
    let first = s3_call(
        &p[0],
        &m[0],
        &m[1],
        &add(&sub(&m[1], &m[2]), &p[1]),
        &sub(&sub(&p[2], &p[0]), &p[1]),
        &lambda,
        &lambda_root,
        c,
    );
    let second = s3_call(
        &p[2],
        &m[0],
        &m[2],
        &sub(&sub(&p[2], &p[1]), &mass_difference),
        &sub(&sub(&p[1], &p[0]), &p[2]),
        &lambda,
        &lambda_root,
        c,
    );
    let third = s3_call(
        &p[1],
        &m[1],
        &m[2],
        &sub(&sub(&p[2], &p[1]), &mass_difference),
        &sub(&add(&p[0], &p[1]), &p[2]),
        &lambda,
        &lambda_root,
        c,
    );
    neg(&mul(&add(&sub(&first, &second), &third), &inverse(&lambda_root)))
}

/// Re[(p3-p1-p2)² - 4 p1 p2] for real momenta.
#[inline]
fn kallen<T: NativeFloat>(p: [&C<T>; 3], c: &Constants<T>) -> T {
    let d = p[2].re.clone() - &p[0].re - &p[1].re;
    d.clone() * &d - c.four.clone() * &p[0].re * &p[1].re
}

#[inline]
fn rotate<'a, T: NativeFloat>(
    p: [&'a C<T>; 3],
    m: [&'a C<T>; 3],
) -> ([&'a C<T>; 3], [&'a C<T>; 3]) {
    ([p[2], p[0], p[1]], [m[2], m[0], m[1]])
}

/// The zero test of the leading coefficient's resultant (exact polynomial).
#[inline]
fn resultant<T: NativeFloat>(p: [&C<T>; 3], m: [&C<T>; 3]) -> C<T> {
    let u = sub(m[2], p[1]);
    let v = sub(m[0], p[0]);
    let w = sub(&add(m[0], m[2]), p[2]);
    let first = mul(&mul(m[0], &u), &u);
    let second = mul(&mul(&w, &u), &v);
    let third = mul(&mul(m[2], &v), &v);
    add(&sub(&first, &second), &third)
}

/// A chart with a nonzero leading coefficient that is not a double root at
/// zero (vanishing Källén function with p3 = 0).
#[inline]
fn usable<T: NativeFloat>(p: [&C<T>; 3], m: [&C<T>; 3], c: &Constants<T>) -> bool {
    !is_zero(&resultant(p, m)) && (!kallen(p, c).is_fully_zero() || !is_zero(p[2]))
}

fn select<T: NativeFloat>(p: [&C<T>; 3], m: [&C<T>; 3], c: &Constants<T>) -> C<T> {
    // trif3's relabeling when the middle propagator has the largest width
    // (always on the real axis).
    let middle_largest =
        !negative(&(m[0].im.clone() - &m[1].im)) && !negative(&(m[2].im.clone() - &m[1].im));
    let (p, m) = if middle_largest { rotate(p, m) } else { (p, m) };
    // first(p, m)
    if usable(p, m, c) {
        return ordered(p, m, c);
    }
    // second(R(p, m))
    let (p, m) = rotate(p, m);
    if usable(p, m, c) {
        return ordered(p, m, c);
    }
    let (p, m) = rotate(p, m);
    ordered(p, m, c)
}

/// Modulus of a complex value (`Symbol::ABS`).
#[inline]
fn modulus<T: NativeFloat>(z: &C<T>) -> T {
    <T as NativeFloat>::hypot(&z.re, &z.im)
}

/// `triangle::zero_momentum_root` at `p = 0`: the exact root `E` (`s_j/s_i`
/// or `s_i/s_j`, nearest to the computed root) and whether it is `s_j/s_i`.
/// Its shift `δ` vanishes exactly at `p = 0`.
fn zero_momentum_root<T: NativeFloat>(
    mass_i: &C<T>,
    mass_j: &C<T>,
    sqrt_i: &C<T>,
    sqrt_j: &C<T>,
    c: &Constants<T>,
) -> (C<T>, bool) {
    let ratio = div(sqrt_j, sqrt_i);
    let inverse = div(sqrt_i, sqrt_j);
    let (root_zero, _) = r_function(&div(&add(mass_i, mass_j), &mul(sqrt_i, sqrt_j)), c);
    let nearer_ratio =
        !negative(&(modulus(&sub(&root_zero, &inverse)) - &modulus(&sub(&root_zero, &ratio))));
    (if nearer_ratio { ratio } else { inverse }, nearer_ratio)
}

/// `triangle::r_function_of`: exact roots at `p = 0` and at (pseudo-)
/// thresholds, [r_function] of `k = (m_i + m_j - p)/(s_i s_j)` otherwise.
fn r_function_of<T: NativeFloat>(
    momentum: &C<T>,
    mass_i: &C<T>,
    mass_j: &C<T>,
    sqrt_i: &C<T>,
    sqrt_j: &C<T>,
    k: &C<T>,
    c: &Constants<T>,
) -> (C<T>, C<T>) {
    let zero = || real(c.one.zero());
    let unequal = !is_zero(&sub(mass_i, mass_j));
    if !is_zero(momentum) {
        // Equal masses test p - 4 m directly: the Källén form would round 3 m.
        let threshold = if unequal {
            let shifted = sub(&sub(momentum, mass_i), mass_j);
            sub(
                &mul(&shifted, &shifted),
                &scale(&mul(mass_i, mass_j), &c.four),
            )
        } else {
            sub(momentum, &scale(mass_i, &c.four))
        };
        if is_zero(&threshold) {
            let one = if negative(&k.re) {
                -c.one.clone()
            } else {
                c.one.clone()
            };
            return (real(one), zero());
        }
        // Real arguments in double-double: `k² - 4 = λ(p, m_i, m_j)/(m_i m_j)`
        // with the threshold polynomial (`p (p - 4 m)` for equal masses). Next
        // to a threshold `k² - 4` cancels the rounding of `k` (relative 2^-106
        // of 4) down to the square root's argument, which then keeps about half
        // its bits (1e-24 in C0's imaginary part where p sits on a
        // binary64-rounded threshold); the polynomial is exact for binary64
        // invariants and masses. Binary64 and `Float` keep the expression's
        // form (where exact-zero tests decide a degenerate chart, a `Float`
        // evaluation follows the expression's own branches).
        if matches!(T::FIXED_BITS, Some(bits) if bits > 53)
            && k.im.is_fully_zero()
            && momentum.im.is_fully_zero()
            && mass_i.im.is_fully_zero()
            && mass_j.im.is_fully_zero()
        {
            let lambda = if unequal {
                threshold.re.clone()
            } else {
                momentum.re.clone() * &threshold.re
            };
            let discriminant = lambda / &(mass_i.re.clone() * &mass_j.re);
            return r_function_real(&k.re, discriminant, c);
        }
        return r_function(k, c);
    }
    if !unequal {
        return (real(c.one.clone()), zero());
    }
    let (exact, nearer_ratio) = zero_momentum_root(mass_i, mass_j, sqrt_i, sqrt_j, c);
    let ratio = div(sqrt_j, sqrt_i);
    let inverse = div(sqrt_i, sqrt_j);
    let difference = if nearer_ratio {
        sub(&ratio, &inverse)
    } else {
        sub(&inverse, &ratio)
    };
    (exact, difference)
}

/// `triangle::r_function_inverse_gap`: `s_i/s_j - 1/r`, exactly zero at
/// `p = 0` when `r = s_j/s_i` (equal masses included).
fn r_function_inverse_gap<T: NativeFloat>(
    momentum: &C<T>,
    mass_i: &C<T>,
    mass_j: &C<T>,
    sqrt_i: &C<T>,
    sqrt_j: &C<T>,
    root: &C<T>,
    c: &Constants<T>,
) -> C<T> {
    let regular = || sub(&div(sqrt_i, sqrt_j), &inverse(root));
    if !is_zero(momentum) {
        return regular();
    }
    if is_zero(&sub(mass_i, mass_j)) || zero_momentum_root(mass_i, mass_j, sqrt_i, sqrt_j, c).1 {
        return real(c.one.zero());
    }
    regular()
}

/// `triangle::ordered_negative_quadratic_roots`: roots `X1, X2` of
/// `a X² - b X + c` in OneLOop's solabc order, `X2` from Vieta's product,
/// and `X1 = b/a - X2` at `c = 0`.
fn ordered_roots<T: NativeFloat>(
    a: &C<T>,
    b: &C<T>,
    cc: &C<T>,
    discriminant: &C<T>,
    c: &Constants<T>,
) -> (C<T>, C<T>) {
    let plus = add(b, discriminant);
    let minus = sub(b, discriminant);
    let larger = if negative(&(modulus(&plus) - &modulus(&minus))) {
        minus
    } else {
        plus
    };
    let small = if is_zero(&larger) {
        real(c.one.zero())
    } else {
        div(&scale(cc, &c.two), &larger)
    };
    let first = if is_zero(cc) {
        sub(&div(b, a), &small)
    } else {
        div(&larger, &scale(a, &c.two))
    };
    let second = if is_zero(discriminant) {
        first.clone()
    } else {
        small
    };
    (first, second)
}

/// `triangle::root_lip_or` with fallback 1 (`root_lip`).
fn root_lip<T: NativeFloat>(
    x: &C<T>,
    a: &C<T>,
    b: &C<T>,
    [da, db, dc]: [&C<T>; 3],
    c: &Constants<T>,
) -> i32 {
    let numerator = sub(&sub(&mul(db, x), &mul(&mul(da, x), x)), dc);
    let slope = sub(&mul(&scale(a, &c.two), x), b);
    let shift = numerator.re.clone() * &slope.re + numerator.im.clone() * &slope.im;
    if shift.is_fully_zero() {
        1
    } else if negative(&shift) {
        -1
    } else {
        1
    }
}

/// `triangle_finite_three_masses_ordered` (OneLOop trif3, DNS representation).
fn ordered<T: NativeFloat>(p: [&C<T>; 3], m: [&C<T>; 3], c: &Constants<T>) -> C<T> {
    let sqrt_1 = physical_sqrt(m[0]);
    let sqrt_2 = physical_sqrt(m[1]);
    let sqrt_3 = physical_sqrt(m[2]);
    let k12 = div(&sub(&add(m[0], m[1]), p[0]), &mul(&sqrt_1, &sqrt_2));
    let k13 = div(&sub(&add(m[0], m[2]), p[2]), &mul(&sqrt_1, &sqrt_3));
    let k23 = div(&sub(&add(m[1], m[2]), p[1]), &mul(&sqrt_2, &sqrt_3));
    let (r12, _) = r_function_of(p[0], m[0], m[1], &sqrt_1, &sqrt_2, &k12, c);
    let (r13, d13) = r_function_of(p[2], m[0], m[2], &sqrt_1, &sqrt_3, &k13, c);
    let (r23, _) = r_function_of(p[1], m[1], m[2], &sqrt_2, &sqrt_3, &k23, c);
    let a = add(
        &sub(&div(&sqrt_2, &sqrt_3), &k23),
        &mul(&r13, &sub(&k12, &div(&sqrt_2, &sqrt_1))),
    );
    if is_zero(&a) {
        return real(c.one.zero());
    }
    let b = sub(
        &add(&div(&d13, &sqrt_2), &div(&k12, &sqrt_3)),
        &div(&k23, &sqrt_1),
    );
    let lambda = kallen(p, c);
    let masses = mul(&mul(m[0], m[1]), m[2]);
    let discriminant = physical_sqrt(&div(&real(lambda), &masses));
    let cc = div(
        &r_function_inverse_gap(p[2], m[0], m[2], &sqrt_1, &sqrt_3, &r13, c),
        &mul(&sqrt_1, &sqrt_2),
    );
    let (root_1, root_2) = ordered_roots(&a, &b, &cc, &discriminant, c);
    // Real roots inherit their lips from OneLOop's p -> p(1 + i eps); the lip
    // only matters on the negative real axis.
    let lip = |root: &C<T>| -> i32 {
        if !root.im.is_fully_zero() || !negative(&root.re) {
            return 1;
        }
        let velocity = |momentum: &C<T>, si: &C<T>, sj: &C<T>| neg(&div(momentum, &mul(si, sj)));
        let dk13 = velocity(p[2], &sqrt_1, &sqrt_3);
        let singular = is_zero(&d13) && !is_zero(&dk13);
        let zero = real(c.one.zero());
        let (dr13, dd13, weight) = if singular {
            let step = -modulus(&dk13).sqrt();
            (
                real(step.clone()),
                real(step.clone() * &c.two),
                zero.clone(),
            )
        } else if is_zero(&d13) {
            (zero.clone(), zero.clone(), real(c.one.clone()))
        } else {
            let ratio = div(&dk13, &d13);
            (mul(&r13, &ratio), mul(&k13, &ratio), real(c.one.clone()))
        };
        let dk12 = mul(&velocity(p[0], &sqrt_1, &sqrt_2), &weight);
        let dk23 = mul(&velocity(p[1], &sqrt_2, &sqrt_3), &weight);
        let da = add(
            &add(&neg(&dk23), &mul(&dr13, &sub(&k12, &div(&sqrt_2, &sqrt_1)))),
            &mul(&r13, &dk12),
        );
        let db = sub(
            &add(&div(&dd13, &sqrt_2), &div(&dk12, &sqrt_3)),
            &div(&dk23, &sqrt_1),
        );
        let dc = div(&dr13, &mul(&mul(&mul(&r13, &r13), &sqrt_1), &sqrt_2));
        root_lip(root, &a, &b, [&da, &db, &dc], c)
    };
    let qx1 = Sheet::with_sign(root_1.clone(), lip(&root_1));
    let qx2 = Sheet::with_sign(root_2.clone(), lip(&root_2));
    let q12 = Sheet::lower(r12.clone());
    let q13 = Sheet::lower(r13.clone());
    let q23 = Sheet::lower(r23.clone());
    let qm1 = Sheet::lower(sqrt_1.clone());
    let qm2 = Sheet::lower(sqrt_2.clone());
    let qm3 = Sheet::lower(sqrt_3.clone());

    let z1 = qx1.mul(&qm2);
    let z2 = qx2.mul(&qm2);
    let mut finite = mul(
        &add(
            &mul(&dd(&z1.mul(&q12), &z2.mul(&q12), c), &r12),
            &div(&dd(&z1.div(&q12), &z2.div(&q12), c), &r12),
        ),
        &sqrt_2,
    );
    let temp = q13.mul(&qm2);
    let t1 = qx1.mul(&temp);
    let t2 = qx2.mul(&temp);
    finite = sub(
        &finite,
        &mul(
            &mul(
                &add(
                    &mul(&dd(&t1.mul(&q23), &t2.mul(&q23), c), &r23),
                    &div(&dd(&t1.div(&q23), &t2.div(&q23), c), &r23),
                ),
                &r13,
            ),
            &sqrt_2,
        ),
    );
    let y1 = qx1.mul(&q13);
    let y2 = qx2.mul(&q13);
    finite = add(
        &finite,
        &sub(
            &mul(&mul(&dd(&y1.mul(&qm3), &y2.mul(&qm3), c), &r13), &sqrt_3),
            &mul(&dd(&qx1.mul(&qm1), &qx2.mul(&qm1), c), &sqrt_1),
        ),
    );
    if !is_zero(&root_2) {
        let log_m3 = qm3.log(c);
        let log_m1 = qm1.log(c);
        let first = mul(
            &y1.div(&y2).log_over_one_minus(c),
            &sub(&add(&y1.log(c), &y2.log(c)), &scale(&log_m3, &c.two)),
        );
        let second = mul(
            &qx1.div(&qx2).log_over_one_minus(c),
            &sub(&add(&qx1.log(c), &qx2.log(c)), &scale(&log_m1, &c.two)),
        );
        finite = add(
            &finite,
            &div(&sub(&first, &second), &scale(&root_2, &c.two)),
        );
    }
    div(&finite, &mul(&mul(&mul(&a, &sqrt_1), &sqrt_2), &sqrt_3))
}
