# Genuine shifted pentagons: proposed scalar-master extension

Status: independently validated mathematical identity and reference audit,
2026-10-23. Eight focused generic Rust tests passed in
`crates/iregi-masters/src/pentagon6.rs`, including two negative-width contour
angles. Public-core binary64 integration now passes independent positive-F
and negative-width fixtures, mixed-lane endpoint rejection, dimensional
scaling, empty batches, and internal box-row counters. Five finite-domain
guard tests, four Symbolica uncertainty-propagation tests, one budget test,
and one executor packing test pass. Arbitrary physical-domain coverage and
performance parity are not established.
The defining parameter integral disagrees with the reference's standalone
shifted-pentagon output at the finite nondegenerate point recorded below.

## Current boundary and provenance

The dimension recurrence has a zero epsilon-independent denominator for five
unit propagator powers in `D=6-2 epsilon`. Finite A0--D0 coefficients alone do
not determine this request by dividing the dimension recurrence. The
distinction between finite D4 pentagons and higher epsilon orders is discussed
by [Kozlov and Lee](https://arxiv.org/html/1512.01165v2), including their Eq. (6).
Their paper concerns the massless on-shell case; its specialized formula is
not assumed to cover arbitrary massive kinematics here. The general scalar
recurrences follow [Duplancic and Nizic](https://arxiv.org/abs/hep-ph/0303184).

The supplied source `references/madgraph/vendor/IREGI/src/si_reduce.f90`, lines
212--218 and 791--797, sends the zero-divisor case through `IBP2PAVE` and a PV
coefficient. For five unit powers and `idim=2`, lines 1159--1178 give a factor
`-2*pi` multiplying `E00`; after removing the extra pi from the shifted
measure, this corresponds to `J_1=-2 E00`. The pure-metric five-point branch in
`pave_reduce.f90` near line 1689 constructs that coefficient from box tensor
coefficients. The independent audit below establishes that this standalone
scalar result does not equal the defining shifted integral at its test point.
Agreement of a complete four-dimensional tensor does not by itself determine
the coefficient of a redundant metric in a four-vector basis.

`ScalarCompiler::set_routing` now gives exact six/seven-point denominator
relations at unchanged dimension and arbitrary positive powers, reducing those
families to at most five active denominators. A degenerate external span can
also make a shifted pentagon reducible. A generic rank-four shifted pentagon
uses the genuine finite master below when the compiler enables that extension;
the standalone compiler retains an explicit missing-master diagnostic when
the extension is disabled. The public generalized reduction plans enable it
and enforce the backend's qualified endpoint domain at runtime.

## Common mass-shift identity

Use the compiler normalization `J_h=pi^(-h) I_(4+2h-2 epsilon)`, with the common
`i*pi²` and the same epsilon normalization stripped. For five unit powers let

```
F(x) = sum_i x_i m_i² - sum_(i<j) x_i x_j (p_i-p_j)²,
sum_i x_i = 1, x_i >= 0.
```

The Feynman prescription is the lower-half-plane boundary value of `F`.
The ordinary Dirichlet parameter representation gives

```
J_h = -Gamma(3-h+epsilon) integral_simplex F(x)^(-3+h-epsilon) dx.
```

A common squared-mass shift `m_i² -> m_i²+t` changes `F` to `F+t`. Therefore

```
d/dt J_1(m_i²+t) = -J_0(m_i²+t),
J_1(m_i²) = integral_0^infinity J_0(m_i²+t) dt.
```

This is a derivation for the proposed implementation, rather than a claim that
the cited paper provides this particular quadrature. At epsilon zero it can
also be checked directly: `J_0=-2 integral(F+t)^(-3)dx`, whose integral over t
is `-integral F^(-2)dx=J_1`. No extra pi factor remains in this normalization.

For a strictly positive lower bound on F, exchanging the simplex and t
integrations is justified by absolute convergence. The integrand decreases as
`t^-3`, and the omitted tail of the t integral decreases as `t^-2`. In this
region the result is UV and IR finite. Taking epsilon to zero under the
integral must be justified separately at massless or exceptional boundaries;
an unqualified finite quadrature must not replace Laurent poles there.

## Physical continuation

For real external invariants and masses with nonpositive imaginary parts,
`Im F <= 0`. Singularities in the common mass-shift variable occur at `t=-F`,
in the upper half-plane or its prescribed boundary. A contour
`t=exp(-i theta) u`, `u>=0`, `0<theta<pi/2`, stays below these singularities.
The proposed expression is

```
J_1 = exp(-i theta) integral_0^infinity
      J_0(m_i²+exp(-i theta)u) du.
```

The arc at infinity vanishes by the cubic decay. Every shifted mass has the
allowed Feynman sign. This argument covers analytic continuation from the
absolutely convergent region while avoiding endpoint pinches. It does not
justify crossing a genuine Landau singularity at the endpoint, nor arbitrary
complex external invariants with uncontrolled `Im F`. Such cases need explicit
domain diagnostics or additional analytic work. Independence of theta must be
verified numerically before physical use.

## Pure Rust architecture

Add a genuine five-point D6 master request to the optional scalar-master
adapter. Generate the D4 pentagon-to-box coefficients with Symbolica, evaluate
all quadrature nodes and input lanes in instruction-major batches, and use
oneloopmaster for the box evaluations. Keep the quadrature implementation
numeric; expression manipulation, differentiation, exact constants, and
matrix algebra remain Symbolica operations.

Map the half-line to a finite interval with a positive squared-mass scale S,
for example `u=S*x/(1-x)`, retaining its exact Jacobian. Generate nodes and
weights directly at the chosen precision, with no intermediate binary64
values. Adaptive state belongs to the workspace, and lanes may refine or fail
independently. Return an error estimate, refinement count, contour information,
and an explicit nonconvergence diagnostic. Cache structural plans, never
sample-valued quadrature results under a topology-only key.

For Dual inputs, keep each node's complete mass/invariant jets. Differentiate
through the same analytic contour with Symbolica's Dualizer only where a
uniform integrable bound permits exchanging differentiation and integration.
Do not silently freeze contour-dependent input derivatives or replace
massless transverse derivatives by zero.

The box expansion can itself have a Cayley denominator that vanishes for a
quadrature node. A deformed contour and invariant-aware coefficient evaluation
must be qualified together. A numerical integration failure cannot produce a
successful zero master. The runtime cost must be reported as its own master
family; a convention-dependent reference coefficient is not an established
performance oracle for this request.

The common shift admits a particularly useful exact update. With `1` the
column of ones,

```
R(t) = R(0) - 2t 1 1^T,
b(0) = R(0)^(-1) 1, B(0) = sum_i b_i(0),
b_i(t) = b_i(0) / (1 - 2t B(0)).
```

Multiplication by `R(t)` verifies this identity directly. The prototype
prepares the original geometry once with Symbolica, then uses this update at
every quadrature node. It solves the equivalent four-dimensional external
Gram system `Z v=f`, where `Z_ij=s_0i+s_0j-s_ij` and
`f_i=s_0i-m_i²+m_0²`, and obtains
`B=1/(f.v-2m_0²)`, `b_i=B v_i`, `b_0=B(1-sum_i v_i)`. Symbolica aliases retain
the factorization. A singular external Gram matrix needs a different
reduction path; this prototype cannot silently apply this chart there.

The pinned Numerica integration API and its Symbolica example expose adaptive
Monte Carlo grids, rather than deterministic one-dimensional quadrature.
Source and example searches in Symbolica/Numerica and FeynKit/gammaloop found
no Gauss--Legendre wrapper. The prototype therefore orchestrates a numerical
Gauss rule, while delegating Rodrigues' polynomial, its derivative, certified
real-root isolation/refinement, and exact rational weight evaluation to
Symbolica's existing polynomial APIs. Nodes and weights are converted
directly from rationals at the requested precision. Neither a polynomial
engine nor a root finder is reimplemented.

## Independent qualification and reference investigation

### Completed independent finite-point audit

Run from the project directory:

```sh
source scripts/env.sh
python3 validation/shifted-pentagon/adjudicate.py --write
```

The committed evidence is `validation/shifted-pentagon/evidence.json`.
The routing is `0,e0,e1,e2,e3` with metric `(+---)` and squared masses
`10,12,15,18,22`. Its Cayley determinant is 2600 and external Gram determinant
is -16. The only positive pair distance is `(p0-p1)²=1`, so
`F >= min(m_i²)-x0*x1 >= 39/4`; the integral has no UV or IR poles.

The independent oracle applies mpmath Gauss--Legendre rules directly to the
four-dimensional simplex integral, with the standard stick-breaking map and
its Jacobian. Increasing orders 14, 18, 22 per axis at 40, 50, 70 working
digits gives

```
J_1 = -0.0001771401382640081806076521310069008176...
```

The order-18 and order-22 D6 results agree to better than `2.2e-29` absolute;
this is a convergence observation, not an interval-arithmetic error bound.
The reference's scalar and generalized interfaces in all three modes instead
give `-0.00014953025024128455`, while reporting stability. Conversion divides
native output by `i*pi^3`. Ordinary D4 evaluation agrees with its independent
parameter integral to `4.0e-19` absolute. Coalesced equal-mass D4 and D6
integrals reproduce `-1/(12 M^3)` and `-1/(24 M^2)` for `M=m²=10`, confirming
the extra dimension-shift pi factor separately.

A second check evaluates the exact split

```
J_1(m) = integral_0^20 J_0(m+t) dt + J_1(m+20).
```

Here the first term uses binary64 reference D4 values with one-dimensional
Gauss--Legendre orders 12, 20, 28, while the tail is independently evaluated
as the smooth four-dimensional parameter integral. Their sums agree with the
direct D6 oracle within `4.93e-15` absolute. This establishes the mass-shift
identity independently of the reference's D6 fallback. It does not qualify
production quadrature or compare performance. The mismatch is a standalone
shifted-scalar issue; it does not adjudicate a discrepancy between complete
physical tensor calculations.

The defining identity also requires the common-mass derivative of `J_1` to
equal `-J_0`. Centered differences of the reference D6 result, with steps
`0.01,0.003,0.001`, converge near `2.1801215e-5`, whereas independently checked
`-J_0=2.36059994160983338e-5`. This additional check excludes an isolated
constant offset as the explanation for the standalone discrepancy.

The complete untruncated half-line identity is now checked separately using
only the five binary64 reference D0 boxes at each node, combined with the
exact rank-one coefficients above. Orders 12, 20, 32, 48, 64 agree with the
direct D6 oracle within `8.5e-17` absolute. The committed evidence records each
value and all box stability flags. This is an independent validation of the
proposed box representation, not a benchmark of the new Rust evaluator.

An initial exploratory half-line integral that called the reference's complete
D4 pentagon at every node was substantially less accurate. Repeating its
five-point numerical matrix solve at large common masses introduced errors
that disappeared when the exact rank-one coefficients were reused. This
example demonstrates why quadrature refinement alone cannot certify the
underlying scalar evaluations: conditioning must be considered separately.

### Remaining qualification

The first five Rust tests passed as one serial run (90.72 s, a test-suite
duration rather than a performance result). They cover the exact rank-one
identity, Gauss polynomial moments, complete finite D6 integration with mixed
valid/invalid batch lanes and empty input, DoubleFloat and 128-bit Float
agreement beyond binary64 accuracy, and a three-lane first-order Dual check
of `sum_i d/dm_i² J_1 = -J_0`. The implemented D0 requests are instruction-major
batches over active lanes, quadrature nodes, and the five box pinches.

1. Overflow/nonfinite convergence guards, tiny-scale covariance, five
   finite-domain classification tests, and the public-core integration test
   pass. An initial public-core attempt reached the 128-node budget before
   its four-bit-margin backend tolerance; the last-rule difference was
   `1.0159e-15`. This failure is
   retained in `validation/shifted-pentagon/core-four-bit-margin-failure.log`.
   The backend budget now reserves two bits instead of four below the
   unchanged public output target; the Symbolica affine uncertainty monitor
   still enforces that target after recurrence amplification. Requalification
   passes, including explicit cancellation/mixed-Dual amplification rejection
   and rejection when the requested output accuracy increases. No numerical
   floor is treated as convergence. The successful core log is retained in
   `validation/shifted-pentagon/core-qualification.log`; its test duration is
   not a performance measurement.
   The core accepts only a proven positive-F lower bound or strictly negative
   widths for all masses. Other endpoints fail explicitly.
2. Validate higher and mixed Dual derivatives and additional independent
   finite differences. A first-order common-mass derivative is already checked.
3. Check mass permutations, routing translations, and negative
   imaginary mass deformations. Verify agreement across at least two contour
   angles and increasing node/precision budgets. The two-angle Rust test
   using the independent oracle below now passes.
4. Only then extend to generic physical and massless limits, generalized
   powers, and mixed Dual shapes.
   Keep failed/unsupported points in the benchmark inventory.

### Retained complex-mass reference counterexample

`validation/shifted-pentagon/contours.json` retains a failed reference-based
contour comparison at routing `0,4e0,e1,e2,e3`, masses
`(1,2,3,4,5)-5i`, and `mu_squared=1`. Direct four-dimensional parameter
integration gives

```
J_1 = 0.00115669079120771304992653728965311
      -0.000473960524532898804360784172426185 i.
```

Rules with 22 and 26 nodes per axis agree within `3.64e-18` absolute. The
reference box contours at two slopes and two node counts converge to a
different value. This discrepancy is traced to one scalar box, rather than
the mass-shift contour identity: removing `4e0` leaves shifts `0,e1,e2,e3`
and masses `(1,3,4,5)-5i`. Its direct parameter integral is
`-0.0013941375463393104+0.004002337189281428i`, while bundled OneLOop-3.4
returns zero. The probe `validation/shifted-pentagon/box_probe.f90` links
only the bundled OneLOop object, bypassing IREGI entirely.

The source `avh_olo_foriregi.f90` explicitly returns zero when its complex
box formula finds no positive external Källén discriminant. Here those four
tests are exactly `(-4,-12,-4,-4)`. This is an independently verifiable
reference limitation: all six distances of that box are nonpositive and all
real mass parts are positive, so `F=A-5i` with `A>0`. Consequently
`Im(F^-2)=10A/(A²+25)²>0` throughout its simplex and the box cannot vanish.
The other four pinched boxes agree with direct parameter integration within
`4.3e-14`. The independent D6 value is retained as a Rust continuation test;
the failed reference contour is not treated as a Rust failure or a passing
reference comparison.

The Rust scalar adapter and direct oneloopmaster have subsequently passed this
isolated box check in binary64, DoubleFloat, and 128/256-bit Float. For example,
the 256-bit finite coefficient begins
`-0.00139413754633931043939422347109+0.00400233718928142794565742388257i`.
The complete Rust D6 contour subsequently passed at slopes 1 and 2 against
the independent parameter value. The focused runtime log is retained in
`validation/shifted-pentagon/rust-focused.log`; its elapsed times are test
durations and are not benchmark measurements. The original-reference failure
remains visible in `contours.json`.

The production finite-domain guard requires either
`min_i Re(m_i²) - sum_(s_ij>0) s_ij/4 > 0` or strictly negative imaginary
parts for all five squared masses. The first condition follows from
`x_i*x_j <= 1/4`; the second gives a uniform negative bound on `Im F` by
convexity. Its arithmetic is normalized before summing to avoid overflow and
subnormal-loss false acceptance. These are sufficient conditions, not a
classification of every finite pentagon. Massless endpoints and real
threshold configurations outside the positive bound remain explicit errors.

The current routing solver also has a separate numerical limitation: a
symbolically selected rational pivot can vanish on an individual lane even
when another pivot works. Alternate determinant charts or a constrained plan
are needed for that lane. Higher precision alone cannot repair an exactly
zero selected pivot.
