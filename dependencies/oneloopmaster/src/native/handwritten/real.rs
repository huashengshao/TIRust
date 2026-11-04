//! Scalars and complex values of the hand-written box kernels, which run in
//! binary64 and in double-double.
//!
//! `Z<T>` is a plain complex pair whose sums, differences and negations are
//! taken component by component and whose products go through the scalar's
//! own `NativeFloat::complex_mul`: in binary64 the operations of
//! `Complex<f64>` (the kernels' results are bit for bit those of the binary64
//! port), in double-double the fused product of `double_double`, which is
//! commutative and odd in each factor (the exact predicates of the
//! expressions need the exactly zero imaginary part of `z conj(z)`).
use super::super::double_double;
use super::super::primitives::{self, C, NativeFloat};
use super::Constants;
use super::sheet::{self, Sheet};
use core::ops::{Add, Div, Mul, Neg, Sub};
use symbolica::domains::float::DoubleFloat;

/// A real scalar of the box kernels.
pub(crate) trait BoxReal:
    NativeFloat
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// `box_complex::rounds_like_binary64`: binary64 takes the line products
    /// that `products_kept` keeps and probes the poles they disagree with
    /// (`tfun_selected`); double-double keeps every product of a parallel
    /// chart and takes it without the probe, as its expression does.
    const BINARY64: bool;

    /// A constant that binary64 represents exactly.
    fn lit(x: f64) -> Self;

    fn abs(self) -> Self;

    /// The port's own Euclidean norms: `f64::hypot` in binary64.
    fn libm_hypot(self, other: Self) -> Self;

    /// `z k` for a real `k`, component by component.
    fn scale(z: C<Self>, k: Self) -> C<Self>;

    /// Bitwise identity.
    fn same(self, other: Self) -> bool;
}

impl BoxReal for f64 {
    const BINARY64: bool = true;

    #[inline(always)]
    fn lit(x: f64) -> Self {
        x
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline(always)]
    fn libm_hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }

    #[inline(always)]
    fn scale(z: C<Self>, k: Self) -> C<Self> {
        C::new(z.re * k, z.im * k)
    }

    #[inline(always)]
    fn same(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl BoxReal for DoubleFloat {
    const BINARY64: bool = false;

    #[inline(always)]
    fn lit(x: f64) -> Self {
        DoubleFloat::from(x)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        symbolica::domains::float::Real::norm(&self)
    }

    #[inline(always)]
    fn libm_hypot(self, other: Self) -> Self {
        <Self as NativeFloat>::hypot(&self, &other)
    }

    #[inline(always)]
    fn scale(z: C<Self>, k: Self) -> C<Self> {
        double_double::real_scale(&double_double::factor(&k), &z)
    }

    #[inline(always)]
    fn same(self, other: Self) -> bool {
        let (a, b) = (self.into_inner(), other.into_inner());
        a.hi().to_bits() == b.hi().to_bits() && a.lo().to_bits() == b.lo().to_bits()
    }
}

/// A complex value of the box kernels (see the module documentation).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Z<T> {
    pub(crate) re: T,
    pub(crate) im: T,
}

impl<T: BoxReal> Z<T> {
    #[inline(always)]
    pub(crate) fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub(crate) fn c(self) -> C<T> {
        C::new(self.re, self.im)
    }

    #[inline(always)]
    pub(crate) fn of(z: C<T>) -> Self {
        Self { re: z.re, im: z.im }
    }
}

impl<T: BoxReal> Add for Z<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl<T: BoxReal> Sub for Z<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl<T: BoxReal> Neg for Z<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl<T: BoxReal> Mul for Z<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self::of(T::complex_mul(&self.c(), &other.c()))
    }
}

impl<T: BoxReal> Mul<T> for Z<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, k: T) -> Self {
        Self::of(T::scale(self.c(), k))
    }
}

#[inline(always)]
pub(crate) fn zero<T: BoxReal>() -> Z<T> {
    Z::new(T::lit(0.0), T::lit(0.0))
}

#[inline(always)]
pub(crate) fn one<T: BoxReal>() -> Z<T> {
    Z::new(T::lit(1.0), T::lit(0.0))
}

/// A real constant.
#[inline(always)]
pub(crate) fn r<T: BoxReal>(x: f64) -> Z<T> {
    Z::new(T::lit(x), T::lit(0.0))
}

/// A real value.
#[inline(always)]
pub(crate) fn rv<T: BoxReal>(x: T) -> Z<T> {
    Z::new(x, T::lit(0.0))
}

#[inline(always)]
pub(crate) fn inverse<T: BoxReal>(z: &Z<T>) -> Z<T> {
    Z::of(primitives::inverse(&z.c()))
}

#[inline(always)]
pub(crate) fn dv<T: BoxReal>(a: Z<T>, b: Z<T>) -> Z<T> {
    a * inverse(&b)
}

#[inline(always)]
pub(crate) fn nz<T: BoxReal>(a: Z<T>) -> bool {
    !a.re.is_fully_zero() || !a.im.is_fully_zero()
}

#[inline(always)]
pub(crate) fn norm<T: BoxReal>(a: Z<T>) -> T {
    a.re * a.re + a.im * a.im
}

#[inline(always)]
pub(crate) fn physical_sqrt<T: BoxReal>(z: &Z<T>) -> Z<T> {
    Z::of(super::physical_sqrt(&z.c()))
}

#[inline(always)]
pub(crate) fn r_function<T: BoxReal>(k: &Z<T>, c: &Constants<T>) -> (Z<T>, Z<T>) {
    let (r, d) = super::r_function(&k.c(), c);
    (Z::of(r), Z::of(d))
}

#[inline(always)]
pub(crate) fn zsheet<T: BoxReal>(value: Z<T>, zero_sign: i32) -> Sheet<T> {
    Sheet::with_sign(value.c(), zero_sign)
}

#[inline(always)]
pub(crate) fn zupper<T: BoxReal>(value: Z<T>) -> Sheet<T> {
    Sheet::upper(value.c())
}

#[inline(always)]
pub(crate) fn zlower<T: BoxReal>(value: Z<T>) -> Sheet<T> {
    Sheet::lower(value.c())
}

#[inline(always)]
pub(crate) fn dd<T: BoxReal>(first: &Sheet<T>, second: &Sheet<T>, c: &Constants<T>) -> Z<T> {
    Z::of(sheet::dd(first, second, c))
}

#[inline(always)]
pub(crate) fn dd_given<T: BoxReal>(
    first: &Sheet<T>,
    w1: impl FnOnce() -> Z<T>,
    second: &Sheet<T>,
    w2: impl FnOnce() -> Z<T>,
    c: &Constants<T>,
) -> Z<T> {
    Z::of(sheet::dd_given(first, || w1().c(), second, || w2().c(), c))
}

/// The sheet operations of the kernels on `Z`.
pub(crate) trait SheetZ<T: BoxReal> {
    fn zlog(&self, c: &Constants<T>) -> Z<T>;
    fn zlom(&self, c: &Constants<T>) -> Z<T>;
    fn zlog_given(&self, w: Z<T>, c: &Constants<T>) -> Z<T>;
    fn zvalue(&self) -> Z<T>;
}

impl<T: BoxReal> SheetZ<T> for Sheet<T> {
    #[inline(always)]
    fn zlog(&self, c: &Constants<T>) -> Z<T> {
        Z::of(self.log(c))
    }
    #[inline(always)]
    fn zlom(&self, c: &Constants<T>) -> Z<T> {
        Z::of(self.log_over_one_minus(c))
    }
    #[inline(always)]
    fn zlog_given(&self, w: Z<T>, c: &Constants<T>) -> Z<T> {
        Z::of(self.log_given(&w.c(), c))
    }
    #[inline(always)]
    fn zvalue(&self) -> Z<T> {
        Z::of(self.value())
    }
}
