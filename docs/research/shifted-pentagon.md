# Genuine shifted pentagons: proposed scalar-master extension

Status: mathematical design and validation proposal, 2026-10-23. Numerical
quadrature is not implemented or qualified. No numerical defect in the
reference's standalone shifted-pentagon output is asserted by this note.

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
coefficients. The precise standalone scalar interpretation must be tested.
Agreement of a complete four-dimensional tensor does not by itself determine
the coefficient of a redundant metric in a four-vector basis.

`ScalarCompiler::set_routing` now gives exact six/seven-point denominator
relations at unchanged dimension and arbitrary positive powers, reducing those
families to at most five active denominators. A degenerate external span can
also make a shifted pentagon reducible. A generic rank-four shifted pentagon
remains an explicit missing-master diagnostic.

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

## Independent qualification and reference investigation

1. Start from real routing points `0,e0,e1,e2,e3` and squared masses
   `10,12,15,18,22`. These span four dimensions, while the masses dominate the
   positive invariants and make F strictly positive. Confirm the bound and
   nonzero Cayley/Gram determinants exactly with Symbolica.
2. Evaluate `-integral_simplex F^-2` by an independent four-dimensional
   parameter quadrature at increasing precision. Separately evaluate the
   proposed one-dimensional mass-shift integral of D4 pentagons. Require
   convergent, overlapping error estimates.
3. Compare the repaired generated IREGI generalized interface at `idim=2`,
   all unit powers, using the documented factorial initialization repair and
   recording its stability flag. Apply the exact pi/normalization conversion.
   Classify any discrepancy only after the two independent defining-integral
   evaluations agree. Preserve the point, all Laurent slots, revisions, and
   reproduction commands; do not infer a defect solely from code inspection.
4. Check `sum_i d/dm_i² J_1 = -J_0` against independently evaluated D4
   pentagons, both with high-precision finite differences and Dual seeds away
   from singularities. This probes normalization and the scalar interpretation.
5. Check scaling, mass permutations, routing translations, and negative
   imaginary mass deformations. Verify agreement across at least two contour
   angles and increasing node/precision budgets.
6. Only then extend to generic physical and massless limits, generalized
   powers, batches, DoubleFloat, arbitrary precision, and mixed Dual shapes.
   Keep failed/unsupported points in the benchmark inventory.

The current routing solver also has a separate numerical limitation: a
symbolically selected rational pivot can vanish on an individual lane even
when another pivot works. Alternate determinant charts or a constrained plan
are needed for that lane. Higher precision alone cannot repair an exactly
zero selected pivot.
