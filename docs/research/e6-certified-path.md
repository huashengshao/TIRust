# A bounded path to certified shifted pentagons

Source/design audit, 2026-10-23. This note proposes work; it does not add an
implemented certificate, new qualification, numerical result, or timing. The
working tree was inspected at `bdb55c9516e2c8f5700dcfb3d2c48fa87e6697a0`.
An independent agent checked the recurrence distinction and the remainder
factors below. No production source, builds, or numerical runs were changed
during this audit.

The smallest useful public extension is an explicit accuracy-aware E6 method
which encloses the existing Symbolica geometry, actual quadrature nodes and
weights, shifted box inputs, certified box values, and final contraction, then
adds an analytic integration remainder. The first domain should remain
restricted to points for which every enclosure and every box certificate
succeeds. Ordinary D4 unit-power N5/N6 already have an analytic route which
avoids E6; that route does not determine a genuine D6 pentagon.

## What the current implementation proves

`crates/iregi-masters/src/pentagon6.rs:78` prepares a factor-preserving exact
Symbolica geometry program. For four independent external differences it uses

```
Z_ij = s_0i + s_0j - s_ij,       f_i = s_0i - m_i² + m_0²,
v = Z^-1 f,                      B = 1/(f.v - 2m_0²),
b_0 = B(1 - sum_i v_i),          b_i = B v_i.
```

Its numerical instantiation at line 166 uses point-valued `Complex<T>`, not
interval geometry. Lines 419–451 form the contour map, rounded shifted box
inputs and ordinary D0 batch. Lines 490–504 contract and sum them. Lines
510–540 accept two successive sufficiently small rule differences; those
differences are estimates, not analytic error bounds.

The Gauss generator at line 571 already uses Symbolica's certified real-root
isolation and rational interval refinement. It verifies the number of roots
and simple multiplicities. At line 600 it discards each interval and evaluates
the weight at the rational midpoint. Keeping an accurate midpoint does not
enclose the exact Gauss node or weight.

`crates/iregi-rust/src/stability/quadrature.rs:206` explicitly describes its
inputs as successive-rule estimates. Its Symbolica affine-coefficient
extraction and positive Taylor error convolution can be reused, but must not
be relabelled as a certificate. A certificate also includes coefficient
rounding, input uncertainty and their cross terms.

The separate private experiment in
`validation/near-vacuum-input-enclosure/qualification.json` establishes the
one-point shifted-input/box/contraction proof with exact supplied unshifted
geometry. Its SHA-256 is
`9a2987758e106c2b199388178b1766a743420efbeafbf7f7cbf71f15178dce82`.
It does not enclose a computed geometry solve or an integral over the contour.
That distinction remains unchanged here.

## Three reuse checks

The checked Symbolica/Numerica base is revision
`9cdc2949cdb153478c1eca0d40690dc0abfc616e` with the locally recorded patches and
interval facilities; see `dependencies/symbolica/IREGI_PROVENANCE.json` and
`dependencies/symbolica-patches/README.md`. The FeynKit/gammaloop pin in
`Cargo.lock` is `865afdc43fb4e6b950439d34b4ec2c38352873e7`.

| Check | Existing facilities and exact source pointers | Consequence |
| --- | --- | --- |
| Symbolica APIs/source | Numerica `tensors/matrix.rs:430`, `:1731`, `:1949`, `:1990`: fraction-free solve, inverse, solve and solve-any. `domains/float/interval.rs:109`, `:120`, `:128`, `:147`, `:220`: directed real bounds, exact rational input conversion, endpoints and complex balls. Division/inversion at `:353`/`:511` return an unbounded interval if a denominator contains zero. Symbolica `poly/univariate/roots.rs:2157`/`:2176`: exact rational root isolation/refinement. | Reuse the existing exact rational geometry and interval expression evaluation. Retain isolated roots; no new matrix elimination, polynomial algebra, root finder, or jet arithmetic. |
| Existing usage/tests | Numerica matrix inverse tests at `tensors/matrix.rs:2177`; interval rational/complex arithmetic tests at `domains/float/interval.rs:900`, `:944`, `:1014`, `:1042`; Symbolica root isolation/refinement tests at `poly/univariate/roots/tests.rs:24`. Current E6 Gauss moment test at `pentagon6.rs:815`. Existing exact-program-to-ComplexBall batch mapping and bounded precision refinement at `crates/iregi-rust/src/invariant_certificate.rs:390`. | The required primitives have existing usage and tests. The current Gauss moment test is an accuracy check, not a quadrature certificate. Reuse the invariant-certificate orchestration pattern without claiming its determinant proof already covers E6. |
| FeynKit/gammaloop/oneloopmaster equivalents | Searched Rust/Markdown in `../references/gammaloop_feynkit`, `dependencies/feynkit-integrals`, and `dependencies/oneloopmaster` for `gauss_legendre`, `Gauss.Legendre`, `Gauss.Kronrod`, `validated quadrature`, `rigorous quadrature`, `RealBall`, `ComplexBall`, `isolate_real_root_intervals`, and `refine_root_interval`. No matching implementation was found. Numerica `numerical_integration.rs` and `examples/integration.rs` provide Monte Carlo grids/statistics, rather than a deterministic validated Gauss remainder facility. | New orchestration and an analytically justified bound are needed in this checked scope. This is not a claim that no such facility exists elsewhere. |

Paths in the first two rows beginning with `tensors/` or `domains/` are relative
to `dependencies/symbolica/lib/numerica/src/`; those beginning with `poly/`
are relative to `dependencies/symbolica/src/`.

In particular, ordinary generic `Matrix::inv()` over an interval field is not
automatically a validated interval linear solver: its pivot and zero decisions
need their own justification. The narrower approach is to evaluate the
already-derived exact rational chart with directed intervals, prove all active
denominators exclude zero, and reject or refine uncertain lanes. Singular
charts require a separately valid chart, not a pseudoinverse or snapping.

## Proposed narrow API and implementation boundary

Add a separate method to `Pentagon6Evaluator<T>` alongside its existing
`evaluate_batch` (`pentagon6.rs:304`). Proposed names are illustrative:

```
evaluate_batch_with_accuracy(
    input, rows,
    optional_exact_input_rectangles,
    &MasterAccuracyContext,
    &CertifiedPentagon6Options,
) -> CertifiedPentagon6Batch<T>
```

Reuse the explicit rational per-component budgets and finite-component bounds
exported by `crates/iregi-masters/src/certified.rs:16`; the underlying error
contract is in `near_vacuum.rs:45`. An absent rectangle means the exact
represented input, not an unspecified intended decimal value. A supplied
rectangle must contain every represented component. External routing-to-
invariant roundoff remains outside the certificate unless that transformation
is also enclosed by a Symbolica program. Scale/domain validation remains
mandatory even where the finite value is scale independent.
The first path inherits the bounded box facility's branch contract for every
component: invariant rectangles lie entirely on the real axis, with its
existing mass-jet restrictions at a zero-width boundary. Real centers with
complex invariant uncertainty do not meet this contract.

The options need finite maximum working precision, rule order and refinement
counts. The result should distinguish domain rejection, unresolved geometry,
unresolved node/weight enclosure, unavailable box certificate, exhausted
quadrature bound and budget failure. Failure exposes diagnostics but no
successful value. An ordinary fallback may be offered only as explicitly
uncertified, as in `CertificatePolicy`; it must never satisfy a request which
requires a certificate. Keep ordinary callers free of new ball conversions.

The smallest implementation sequence is:

1. Map the existing geometry/alias program to `ComplexBall`, as already done
   for exact invariant programs. Enclose every Taylor component. Check the
   fixed Gram chart and `B` denominators, including the updated denominator
   `1-2tB` at every active contour node. An interval containing zero is an
   unresolved chart even if its midpoint is finite.
2. Retain each exact rational isolated root interval `[l,u]`. Refine until it
   is inside `(-1,1)` and interval evaluation of `P'_n` excludes zero. Evaluate
   the exact expressions `x=(r+1)/2` and
   `w=1/((1-r²)P'_n(r)²)` with `RealBall`. Require finite positive weight bounds.
   Cache the proven intervals by order and precision. The root API's stopping
   criterion is relative width; use the actual endpoints, not an inferred
   absolute radius.
3. Use the existing Symbolica contour and Dualizer programs with node
   intervals, `t=a*x/(1-x)` and weighted Jacobian `a*w/(1-x)²`. Freeze the
   auxiliary contour parameter at an exact represented admissible primal
   value; preserve every physical input jet. Enclose its numerical
   representation in the same calculation. Require `0<x<1` for each node
   interval and lower-half-plane admissibility of all shifted input boxes.
4. Promote the already qualified private physical-input-rectangle contract
   separately, then obtain a finite certificate for every pinched D0. Batch
   all active samples/nodes/pinches instruction-major. The current bounded
   N2/N4 D0 facility limits this first E6 domain; it is not a general DNS
   certificate. Keep shapes within its qualified component/degree limits.
5. Evaluate the geometry/box/weight contraction and accumulation with directed
   intervals. Add the independent analytic Gauss remainder below to each
   component rectangle. Accept only if the full distance from the returned
   numeric component to every point in that rectangle meets its explicit
   budget. Midpoint agreement or intersecting two enclosures is insufficient.
6. Only after the standalone path is qualified, extend the core's affine
   leaf propagation to the certified finite E6 output. Propagate the full
   reduction coefficient enclosure, including coefficient/leaf cross terms;
   keep successive-rule estimates as separate diagnostics. Requested budgets
   must reach the backend without a fixed-bit acceptance floor.

This is a separate opt-in method before a default-core change. No requested
shape, domain, failure guard or budget should be relaxed to make it succeed.

## Analytic integration remainder

Use the existing project's stripped normalization and five-parameter simplex
`Δ4`, of volume `1/24`. The Feynman parameter convention and distinction between
four-dimensional numerators and regulated denominators follow
[IREGI Appendix C.1](https://arxiv.org/html/1405.0301v2#A3.SS1).
For fixed physical inputs define

```
F(α) = sum_i α_i m_i² - sum_(i<j) s_ij α_i α_j,
a = S(1-i β),  S>0, β>0,
H(x,α) = F(α)(1-x) + a x,
g(x) = -2a integral_Δ4 (1-x) H(x,α)^(-3) dα.
```

Then the genuine finite D6 pentagon is `integral_0^1 g(x) dx` on the qualified
contour. The following bound is derived here, not attributed to a ready-made
quadrature implementation. It assumes the contour deformation and a uniform
analytic neighborhood are valid; the endpoint and branch guards in
`docs/research/shifted-pentagon.md` still apply.

Find rigorous `A>=|a|`, `d>=sup_α |a-F(α)|`, and `h>0` such that `|H|>=h`
uniformly over `x∈[0,1]`, `α∈Δ4`, and the supplied physical input rectangle.
Two sufficient alternatives with invariant rectangles confined to the real
axis are:

```
Re F >= f_* > 0, Re a > 0:         h = min(f_*, Re a);
-Im F >= w_* > 0, -Im a > 0:       h = min(w_*, -Im a).
```

The second condition follows if every mass rectangle has a strictly negative
imaginary upper bound. A conservative first condition is the lower bound on
the smallest real mass minus one quarter of the sum of positive invariant
upper bounds. A valid bound on `|F|` is
`F_max = max_i sup |m_i²| + (1/4) sum_(i<j) sup |s_ij|`, giving `d=A+F_max`.
A sharper bound can use
`d=max_i sup |a-m_i²| + (2/5) max_(i<j) sup |s_ij|`.
The five-parameter maximum-invariant factor is **2/5**, since
`sum_(i<j) α_i α_j <= 2/5`; the D0 four-parameter factor 3/8 is inapplicable.
Complex rectangle norms must be bounded outward, e.g. by the L1 norm.

For `k>=1`, differentiating the displayed rational integrand gives

```
sup_(0<=x<=1) |g^(k)(x)| <= M_k,
M_k = A/24 * [ (k+2)! d^k / h^(k+3)
              + k (k+1)! d^(k-1) / h^(k+2) ].
```

The factor follows from the two Leibniz terms for `(1-x)H^-3` and the simplex
volume; an independent review checked it. Generate and verify the derivatives
with Symbolica `Atom::derivative`, exact integer factorials and existing
polynomial/series operations. This note has not executed that symbolic check.

The n-point Gauss–Legendre exact-node remainder on `[0,1]` obeys, separately
for the real and imaginary parts,

```
|E_n| <= K_n M_(2n),
K_n = (n!)^4 / [ (2n+1) ((2n)!)^3 ].
```

This is the affine transformation of the real Gauss remainder and Legendre
norm in [NIST DLMF 3.5.19](https://dlmf.nist.gov/3.5.E19) and
[3.5.21](https://dlmf.nist.gov/3.5.E21). Apply the real theorem separately;
there need not be a common complex mean-value point. Inflate both rectangle
coordinates, then apply the existing complex distance budget. This remainder
covers the exact Gauss rule; node, weight, input, D0, contraction and summation
uncertainties must also be included as described above.

The bound can be valid but unusably large. Its leading geometric factor is
`[d/(4h)]^(2n)` times powers of n. Increasing the order alone is not a promised
solution when `d>=4h`. A later finite, bounded panel subdivision may use the
same rule and interval-scaled remainder. Exhaustion remains a visible failure;
neither a timing nor adequacy of the current maximum order 128 is claimed.

For native Taylor components, let `P(z)` have zero constant term and
nonnegative coefficients majorizing all nonconstant coefficients of `F`,
uniformly over the simplex and input rectangles. With auxiliary `a` fixed,
replace `d` by `d+P(z)` and `h` by `h-P(z)` in `M_k`, and extract coefficients
with Symbolica's series/Dualizer. This produces a positive formal majorant
without differentiating an absolute value. Keep the native factorial
convention and ancestor-closed component shape. If `a` is differentiated,
its prefactor and both denominator/difference majorants need additional terms;
freezing a common admissible contour avoids that extra scope.

There is **no omitted infinite tail** with the current full compactification:
`H(1)=a!=0` and `g` extends smoothly to `x=1` with value zero. If a later method
cuts the ray `t=a r` at finite `R`, and proves `|F+a r|>=c+b r`, `c,b>0`, then

```
|integral_R^infinity a J5^(D4)(a r) dr|
    <= A / [24 b (c+bR)^2].
```

Use `c=f_*`, `b=Re a` for the real-part guard, or `c=w_*`, `b=-Im a` for the
imaginary guard. With fixed `a,b`, `c -> c-P(z)` gives the Taylor tail majorant.
This optional cutoff needs both that tail and its finite-interval quadrature
bound; it is not part of the current implementation.

## The ordinary N5/N6 alternative and its limit

For `R_ij=s_ij-m_i²-m_j²`, `b=R^-1 1`, `B=sum b_i`, the project convention gives

```
J_0^(5) = sum_i b_i J_0^(4),i - 2 ε B J_1^(5).
```

When the shifted remainder is finite in a common analytic neighborhood, it
does not contribute through the finite Laurent coefficient. This ordinary
pentagon relation appears in [Fleischer–Riemann, (4.1)–(4.2)](
https://arxiv.org/html/1009.4436v3). It is already implemented at
`crates/iregi-rust/src/ibp.rs:573`; its Cayley solve starts at line 1214. Retain
all box pole slots even if the final finite-domain integral has no poles.

For six or more denominators with four-dimensional external coordinates,
Symbolica solves the exact denominator relation

```
sum_i c_i = 0,   sum_i c_i q_i^μ = 0,
sum_i c_i(q_i²-m_i²) = 1,         sum_i c_i D_i = 1.
```

Insertion pinches a denominator at unchanged integration dimension. The
extra-dimensional loop square cancels with `sum c_i=0`. This is the scalar
specialization of [Denner–Dittmaier Appendix D](
https://arxiv.org/html/hep-ph/0509141v2). Current dispatch at `ibp.rs:503`,
Symbolica solves at `:734`, and insertion/power transfer at `:830` already
provide the route. A homogeneous-only relation transfers powers and has a
different scope from a constant-one relation.

Thus an ordinary scalar N6 can reduce to pentagons and boxes without E6. The
next certificate for this path can reuse certified box values and enclose
the existing exact reduction coefficients. No new reduction formula is
needed. `plan.rs:862` already enables genuine E6 only for nonzero dimension
shift or nonunit powers; ordinary four-dimensional tensors keep PV completion.

This does not remove the standalone shifted pentagon. Solving its relation
backwards divides by `-2εB`; finite E6 then needs unavailable O(ε) information
about the D4 pentagon/box combination. The explicit guards at `ibp.rs:509` and
`:606` are mathematically necessary. An exactly lower-dimensional external
family may admit a further valid denominator identity; a small nonzero Gram
determinant cannot be substituted for that identity. The existing independent
counterexample to the reference's standalone IBP2PAVE shortcut is recorded in
`docs/research/shifted-pentagon.md`.

Preserve the following boundaries in either path: exact four-dimensional
constraints in **every** Taylor component, derivative coefficients, valid
nonzero coefficient charts, consistent lower-half-plane branches, generalized
powers and dimension shifts, and existing exceptional/IR treatment. The
scalar-product-only N>=6 guard at `crates/iregi-rust/src/scalar.rs:36` and the
transverse-Taylor concern at `pv.rs:944` must not be bypassed. A recurrence is a
mathematical identity, not a numerical error certificate.

## Bounded qualification after implementation approval

Start with the independent tiny-routing exact-series E6 oracle, positive F,
and the already qualified one-point contour fixture. Enclose the geometry,
every actual rule node/weight, all pinches and the integrated result at
requested 128 and 256 bits with finite working-precision limits. Add unequal
masses and mixed physical jets only within the box facility's qualified
shape. Independently audit every accepted output against the exact oracle
interval and the full returned rectangle, not only its center.

Required negative controls include zero/uncertain Gram charts, a contour
update denominator enclosing zero, broad root or input intervals, an
ineligible D0 lane, insufficient Gauss remainder budget, maximum-order
exhaustion, a zero-primal nonzero-jet remainder, and a genuine endpoint
singularity. Retain failed lanes alongside successful lanes, reversed/chunked
samples and separate workspaces. Qualification must include batching counters.
These are proposed checks, not checks executed by this audit.
