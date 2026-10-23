# Direct near-vacuum E6 series: a smaller certification step

Source/design assessment, 2026-10-23. No implementation, build, numerical
evaluation, certificate or performance result is claimed here. The separate
contour proposal `e6-certified-path.md` remains unchanged, with SHA-256
`03a859ecc4da5ece34cf83a1a1ea68821a149494243be8826222fd0754744bd0`.

**The direct expansion works mathematically in a bounded small-deviation
domain and is the smaller next implementation for the retained tiny-routing
E6 case.** Reuse the existing D0 near-vacuum preparation, numeric/ball batch
evaluation and full-Taylor tail program, changing the parameter simplex,
normalization and envelope constants. No Gram solve, pinched D0 evaluation,
contour deformation, quadrature nodes or integration-error estimate is needed
inside that domain. This is not a replacement for the general E6 contour path.

## Three reuse checks before implementation

1. **Symbolica APIs/source.** `AtomCore::series` is implemented in
   `dependencies/symbolica/src/derivative.rs:377`. The multivariate polynomial
   APIs at `src/poly/polynomial.rs:6667`, `:2659`, and `:2801` provide
   antiderivatives, exact substitution and complete evaluation. Numerica's
   `tensors/matrix.rs:1256` supplies the exact Jacobian determinant. Existing
   `Dualizer` and `ComplexBall`/`RealBall` evaluate the polynomial and positive
   remainder in the required native Taylor convention with directed rounding.
   No monomial integration or Dirichlet-moment engine is missing.
2. **Existing usage/tests.** Production
   `crates/iregi-masters/src/near_vacuum.rs:206–302` already constructs and
   exactly integrates the D0 series with those APIs. Lines 305–346 instantiate
   the three batch programs; lines 363 onward validate, normalize, evaluate
   and check complete component enclosures. The independent E6 example
   `crates/iregi-rust/examples/e6_parameter_series.rs:118–223` already uses
   four cube coordinates, a Symbolica-derived Jacobian and four exact
   polynomial integrations for the retained fixture. Symbolica's
   `src/poly/univariate/tests.rs:175` checks integration/differentiation, and
   `src/derivative.rs:903` begins its series tests. The existing E6 oracle
   construction is documented in `validation/shifted-pentagon/series-oracle.md`.
3. **FeynKit/gammaloop/oneloopmaster.**
   `../references/gammaloop_feynkit/crates/feynkit-graph/src/integrals/parametric.rs:376`
   provides `IntegralFamily::symanzik`; its test at line 813 checks the
   one-loop mass/invariant sign convention. Its documentation explicitly
   leaves measures, powers and prescriptions to the caller. Searches for
   Dirichlet integration, finite near-vacuum expansion and parameter
   integration in the supplied FeynKit/gammaloop and oneloopmaster sources
   did not find a ready E6 finite-series certificate. Oneloopmaster's contour,
   threshold and removable-singularity expansions target other functions.
   The existing project series builder supplies the needed integration
   workflow, so there is no reason to implement new symbolic algebra.

Dependency paths above are within the inspected Symbolica/Numerica base
`9cdc2949cdb153478c1eca0d40690dc0abfc616e` with recorded local patches;
FeynKit/gammaloop is pinned at
`865afdc43fb4e6b950439d34b4ec2c38352873e7`. These checks describe the inspected
scope, not an assertion about all external libraries.

## Normalization and exact preparation

In the project's stripped normalization `J_h=π^(-h) I_(4+2h-2ε)`, the finite
unit-power cases are

```
D0 = +Γ(2) integral_Δ3 F^-2 dx = +integral_Δ3 F^-2 dx,
E6 = -Γ(2) integral_Δ4 F^-2 dx = -integral_Δ4 F^-2 dx,
F  = sum_i x_i m_i² - sum_(i<j) s_ij x_i x_j.
```

This uses the parameter convention in
[IREGI Appendix C.1](https://arxiv.org/html/1405.0301v2#A3.SS1) and the
normalization already established in `docs/research/shifted-pentagon.md`.
The simplex volumes are `vol(Δ3)=1/6` and `vol(Δ4)=1/24`. If every squared
mass equals `M` and every invariant vanishes, the values are therefore
`D0=1/(6M²)` and **`E6=-1/(24M²)`**. Both have mass dimension −4: `M` is a
squared mass. Scaling every squared mass and invariant by λ scales the finite
value by λ^-2 within the same analytic branch. The generic E6 polynomial
cannot be obtained by multiplying the D0 polynomial by −1/4; its parameter
moments differ.

Choose an auxiliary nonzero complex `M` from the lane's represented primal
first squared mass and keep it fixed during differentiation. Set

```
e(z,x) = [sum_i x_i (m_i²(z)-M) - sum_(i<j) s_ij(z) x_i x_j] / M,
E6(z)  = -M^-2 integral_Δ4 (1+e(z,x))^-2 dx.
```

All physical mass jets, including those of the selected first mass, remain in
`m_i²(z)-M`. Freezing this auxiliary expansion center does not freeze a physical
derivative. Apply the current Symbolica bookkeeping-variable series in λ to
`(1+λe)^-2` through order N, set λ=1, and integrate it exactly.

The existing stick-breaking construction needs four cube coordinates and
five simplex coordinates. Its Jacobian is still obtained by Symbolica
differentiation and `Matrix::det`; its four integrations use polynomial
`integrate` and endpoint `replace` exactly as the existing oracle does.
Retain rational coefficients over the combined cube/physical variable map,
which avoids the previously observed expression-coefficient swell. No
factorial/Dirichlet summation routine should be written.

## Reusing the full-component certificate

Let `L=max(|Re M|,|Im M|)>0`, a rigorous lower bound for `|M|`. For each native
Taylor multi-index α form nonnegative exact bounds

```
b_α = [max_i |m²_(i,α)-δ_(α,0)M|_1
       + (2/5) max_(i<j) |s_(ij,α)|_1] / L,
B(z) = sum_α b_α z^α,             ρ = b_0.
```

For interval inputs the magnitudes are suprema over the complete rectangle.
The factor 2/5 follows from `sum_(i<j) x_i x_j <= 2/5` on the five-parameter
simplex; the existing D0 factor 3/8 must change. The L1 complex norm safely
majorizes the Euclidean magnitude. Require the existing conservative
admissibility cutoff `ρ<=1/4`, which in particular ensures `ρ<1` and `F!=0`.
The positive Taylor majorant is

```
T_N(B,L) = B^(N+1) [ (N+1)/(1-B) + 1/(1-B)^2 ] / (24 L²).
```

Each coefficient `[z^α]T_N` bounds that native Taylor component of the omitted
series. It is exactly the current D0 positive tail expression at
`near_vacuum.rs:289–299`, with denominator 6 changed to 24. The overall minus
sign does not affect its magnitude. Generate its jets with the same Symbolica
Dualizer and evaluate its upper bound with RealBall. The current algebra
operation whitelist, finite outputs and exact rational budget checks apply
unchanged. An independent source/math review checked these constants.

A zero primal deviation does not set every remainder component to zero.
For example, at common squared mass `M+t` and zero routing,

```
E6(M+t) = -1/[24(M+t)^2]
       = -1/(24M²) + t/(12M³) - t²/(8M⁴) + t³/(6M⁵) + ... .
```

An N2 expansion has a nonzero cubic remainder even when `ρ=0`; its positive
tail must reject an insufficient cubic-component budget. An N4 expansion is
exact through total Taylor degree four at exact vacuum. For nonzero ρ, every
component still requires its own complete tail and rounding check.

The numeric result, polynomial ComplexBall enclosure and tail RealBall
enclosure remain separate instruction-major batches over active samples.
Convert represented inputs directly to exact rationals before normalization,
then to the requested generic numeric type and directed balls, as the existing
workspace does. No intermediate binary64 values are needed. Reuse
`MasterAccuracyContext` and its rational absolute/relative component budgets;
accept only the complete worst-corner complex distance, including the tail.
Input, branch, shape or budget failures expose no successful value.

The first domain should inherit the existing bounded D0 input contract:
real invariant components, positive-real-part lower-half-plane squared masses,
and the existing mass-jet restriction at a zero-width boundary. Supplied input
rectangles, if supported, must enforce those conditions over the whole
rectangle and contain the represented inputs. The strictly nonzero F bound
proves a finite integral and analytic neighborhood here; all three pole slots
are zero. Scale input validation remains required even though the finite
coefficient has no scale dependence. This does not authorize singular,
massless or different-branch inputs.

## Smallest implementation and its limits

First make a validation-only **full fifteen-parameter N2** E6 specialization
of the current builder and workspace with this fixed family data:

| Property | Current D0 | Proposed genuine E6 |
| --- | --- | --- |
| Simplex/cube coordinates | 4 / 3 | 5 / 4 |
| Pair invariants / squared masses | 6 / 4 | 10 / 5 |
| Pair order | Existing D0 order | Existing E6 lexicographic order |
| Integrated sign | +1 | −1 |
| Tail volume denominator | 6 | 24 |
| Maximum-pair envelope factor | 3/8 | 2/5 |

Share or factor the existing exact preparation and certificate evaluator only
after this specialization is qualified; avoid introducing an unrestricted
N-point framework. Start with N2, keeping N4 as a later bounded extension,
and retain the current 32-component/degree-four shape limits and preparation
checks. A later explicit public E6 accuracy
method can own this plan independently of `Pentagon6Plan`'s Gram preparation,
so a finite zero/small-Gram lane does not first enter an irrelevant geometry
solve. Default contour behavior and core routing/constraint guards remain
unchanged until separately integrated and qualified.

Keep the physical and normalized input layouts distinct: the existing E6
public row has ten distances, five squared masses and `μ²`; the proposed
prepared polynomial has fifteen normalized physical parameters and fixed
`M`. Both happen to have sixteen arguments. Slot 15 must not silently change
meaning through a length-only adapter. Validate the physical scale before
packing; its finite-only Taylor derivatives are zero in this analytic domain.

The full E6 polynomial has fifteen physical variables rather than ten. Its
cube-expanded preparation may exceed the present 150000-term or
500000-instruction limits even when the integrated result is manageable.
This source audit has not measured that cost or established that N4 fits.
Keep explicit preparation failure and bounded profiling; reuse Symbolica
polynomial/integration facilities if operation ordering needs improvement.
Do not evade the bounds with a custom monomial integrator or assume the
existing D0 preparation timings apply.

For the retained fixture `m_i²=1`, `η=2^-64`, and maximum pair magnitude `2η`,
the generic envelope gives `ρ=4η/5`. Thus the N2 primal tail scales as η³,
already giving ample truncation margin for a 128-bit absolute budget, and
the N4 primal tail as η⁵, small enough for the retained 256-bit target.
This does **not** establish numeric polynomial
rounding, preparation feasibility, arbitrary jet budgets or public-core
agreement. The existing independent oracle remains unchanged and must be
used to check the eventual returned rectangle and numeric value.

The first qualification should cover that exact oracle, the common-mass
identity above, heterogeneous mass jets, complex unequal masses, empty and
mixed-success batches, and explicit shape/domain/budget failures. Preserve
the zero-primal cubic-jet rejection. Oracle construction must remain separate
from the candidate preparation; sharing its integrated polynomial would
destroy that independent check.

This route certifies the genuine **unit-power D6 N5** leaf in its bounded
domain. It does not reinterpret it as D4, remove powers/dimension shifts,
invent missing Laurent orders, or relax Taylor constraints. Generalized
reductions may consume the leaf through their existing recurrences only after
their coefficient/rounding uncertainty is also propagated. Outside the
near-vacuum domain or finite preparation/accuracy limits, retain explicit
failure or a separately labelled uncertified contour fallback. The broader
contour certification design is still needed for those cases.
