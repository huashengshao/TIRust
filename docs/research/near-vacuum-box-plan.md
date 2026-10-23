# Bounded near-vacuum D0 branch: implementation proposal

Status: validation-only prototype implemented; selected mixed-derivative
Float256/requested128 fixtures pass independent oracle and enclosure checks.
Production has not changed. The isolated DNS baseline and existing-helper candidate both fail the
exact node target; `validation/shifted-pentagon/d0-node-qualification.json`
retains those results. This proposal addresses the regular finite-box domain
directly and retains DNS and the existing contour formulas elsewhere.

## Existing facilities checked

The three required checks found reusable algebra operations, but no existing
integrated small-momentum D0 implementation covering the failing family.

1. **Symbolica APIs/source.** `AtomCore::series`,
   `MultivariatePolynomial::integrate`, `replace`, and exact `evaluate` already
   provide the expansion and parameter integration. `Dualizer` uses the native
   Taylor coefficient convention, including higher and mixed components.
   `RealBall`/`ComplexBall` certify rational construction, basic arithmetic,
   reciprocal and integer powers using directed rounding. Their inherited
   transcendental functions are not certified, so this proposal uses none.
   Sources: `dependencies/symbolica/src/poly/polynomial.rs`,
   `src/evaluate/dual.rs`, and
   `lib/numerica/src/domains/float/interval.rs` within that dependency.
2. **Existing usage/tests.** Symbolica's `examples/series.rs`, Dualizer's higher
   and mixed derivative tests, and ball arithmetic tests exercise these APIs.
   `e6_parameter_series.rs` and `d0_contour_node.rs` already construct independent
   exact oracles with series, stick-breaking Jacobians, polynomial integration
   and endpoint substitution. `crates/iregi-masters/src/bubble_neighborhood.rs`
   demonstrates analytic Taylor neighborhoods with arbitrary supplied jets;
   its exact-zero bubble formulas do not supply a finite nonzero-momentum box.
   Oneloopmaster's `d0_tiny_invariant` test covers a different contour-admissible
   family. Its mass-first grouping is useful but does not cover this DNS branch.
3. **FeynKit/gammaloop/oneloopmaster.** FeynKit
   `IntegralFamily::symanzik` already constructs equivalent parameter data.
   Its `uv_expansion_of` implements a local massive UV integrand expansion,
   truncated by superficial degree, rather than the integrated finite-box
   expansion and error bounds required here. Vakint's vacuum and epsilon
   expansion facilities target a different workflow. Oneloopmaster has the
   exact all-zero-invariant equal-mass result `1/(6*M^2)`, plus removable
   logarithm/dilogarithm series and threshold helpers, but no bounded
   small-invariant D0 neighborhood. The private `r_function_of` experiment
   improves this node without recovering the requested accuracy.

No new polynomial, jet, Dirichlet-moment, symbolic integration, or interval
engine is proposed. Coefficients must be obtained by existing Symbolica
operations. Do not implement a handwritten factorial/multinomial integrator.

## Finite analytic domain and expansion

For four propagators, on the simplex `sum(x_i)=1`, define

```text
F(t,x) = sum_i m_i_squared(t)*x_i - sum_(i<j) s_ij(t)*x_i*x_j.
```

Choose an auxiliary complex constant `M` from a lane's primal squared mass.
For the initial domain require `Re(M)>0`, physical lower-half-plane squared
masses, and real primal invariants, using the existing input rules. Keep `M`
fixed while differentiating. It is an additional numerical argument with zero
nonprimal components, not a replacement of the physical mass inputs. Thus

```text
E(t,x) = F(t,x)-M,
e(t,x) = E(t,x)/M,
D0(t) = M^(-2) integral_simplex (1+e(t,x))^(-2) dx.
```

All derivatives of the original masses remain in `E`. Holding this auxiliary
constant fixed is an exact identity, not an approximation to the derivative.
If `sup_x |e(0,x)| <= rho < 1`, the segment `M+lambda*E(0,x)`,
`0<=lambda<=1`, avoids zero. With the physical mass prescription it continues
the same finite branch. The three pole outputs are zero throughout this
analytic domain. Exact-zero Gram or Cayley determinants alone do not invalidate
it; eligibility must follow the parameter-domain proof rather than silently
discarding derivative components on those boundaries.

Construct the order-N polynomial by applying Symbolica's series operation to
`(1+lambda*E/M)^(-2)` and then setting the bookkeeping variable `lambda=1`.
Apply the existing three-coordinate stick-breaking substitution, derive its
Jacobian through Symbolica, and integrate the polynomial with
`integrate`/endpoint replacement. Keep physical parameters as coefficient
expressions where useful. In the first implementation, expression-valued
coefficients caused a preparation timeout even at N2. Existing Symbolica exact
rational polynomials over the cube and physical variables instead prepare N2
in about 48 ms and N4 in about 2.6 s in diagnostic runs. Integrating the first
three variables leaves 66 and 1001 physical-parameter monomials, respectively.
The N4 intermediate has 103243 monomials, within the explicit 150000-term
preparation cap. These are preparation observations, not performance gates.
Compile the final rational expression with the existing evaluator
and instruction-major Rust emitter. Dualizer produces the requested shape.

## A bound for every Taylor component

Let the native ancestor-closed shape contain indices `alpha`, with component
values already divided by `alpha!`. All bounds below refer to those Taylor
coefficients, not raw derivatives. No factorial conversion is added.

Construct exact rational upper bounds from the represented input components.
For a complex component use `abs(Re)+abs(Im)` as an upper bound on its norm.
Let `L=max(abs(Re(M)),abs(Im(M)))>0`, so `L<=|M|`. Define

```text
mu_alpha = max_i upper_abs([t^alpha](m_i_squared-M)),
sigma_alpha = max_(i<j) upper_abs([t^alpha]s_ij),
b_alpha = (mu_alpha + (3/8)*sigma_alpha)/L.
```

Here `M` has only its primal coefficient. The bound follows from
`sum_i x_i=1` and `sum_(i<j)x_i*x_j<=3/8` for four nonnegative parameters.
It is conservative and valid for unequal nearby masses and transverse mass
jets. Set `rho=b_0`; reject this branch if `rho>=1`. A stricter initial
eligibility cap, such as `rho<=1/4`, is reasonable for bounded work.

Create a positive Taylor polynomial `b(t)` with these coefficients. Its
coefficientwise nonnegative powers majorize the absolute coefficients of
`e(t,x)^n`, uniformly in the simplex. Consequently the truncation error in each
retained component is bounded by the corresponding coefficient of

```text
H_N(b(t))/(6*L^2),
H_N(z) = sum_(n>N) (n+1)*z^n
       = z^(N+1)*((N+1)/(1-z)+1/(1-z)^2).
```

The displayed positive form avoids cancellation in the equivalent numerator
`(N+2)-(N+1)*z`. The infinite scalar sum converges at `rho<1`; nilpotent
components are handled by its finite Taylor jet. An optional simpler majorant is
`G_N(z)=(N+2)*z^(N+1)/(1-z)^2`, which bounds the scalar tail coefficientwise.
Generate and evaluate the bound jet using Symbolica Dualizer or its existing
hyperdual arithmetic over exact rationals. Do not write coefficient
convolutions. Retain bounds for every component, including components whose
primal is zero. At `rho=0`, an order at least the maximum total requested jet
degree makes this tail vanish exactly; a smaller order may miss derivatives
even though its primal truncation error is zero.

For nonzero `rho`, selecting N from the primal tail alone is invalid: each
derivative can lower the power of `rho`. Increase N only within an explicit
finite preparation/evaluation budget until **every** component bound fits its
allocated accuracy. Large seed components can legitimately reject the branch.
For example, at `rho=0`, N=1 and `h=b10*t1+b01*t2`, the omitted mixed
coefficient is `6*b10*b01` despite a zero primal tail. With `rho` near
`2^-64` and order-one derivative seeds, N=4 can have d-th derivative tails of
order `rho^max(5-d,0)`, so its scalar accuracy does not imply comparable Dual
accuracy.

## Rounding and acceptance

The series remainder is a mathematical truncation bound, not a floating-point
roundoff certificate. Reuse `ComplexBall` to evaluate the same prepared
rational program over exact input enclosures. Its arithmetic and integer
powers are certified; reject any unexpected transcendental or uncertified
operation during preparation. Symbolica Dualizer reduces each requested jet
to those same arithmetic operations, so the ball result encloses every
truncated coefficient. Add its component truncation bound with directed
rounding. `ComplexBall` is a rectangle, so certify the worst-case complex
distance from the returned value to the entire rectangle, including the added
tail. For each Cartesian part take the larger distance to its two outward
endpoints, then compare the sum of the two squared distances against the
unchanged requested absolute/relative component budget squared. Membership in
the enclosure, or independently comparing its two radii to the same complex
budget, is insufficient.

At first, exact-rational controls plus the ball evaluator provide a slower,
auditable exceptional path. A later performance change may reduce overhead
only while preserving the same guarantees. Tracked precision alone is not a
certified rounding bound. Working precision may be inadequate even if N is
sufficient; report that separately from truncation or domain failure.

The current scalar-master workspace does not accept a requested accuracy
context, so that interface needs an explicit additive control rather than
assuming working bits equal requested bits. Pass the requested context from
the core and from E6 to its D0 batches. Track per-component truncation and
roundoff bounds in diagnostics. When E6 combines boxes, propagate their
component bounds through its existing Symbolica contraction/weighted sum;
do not compare only successive quadrature orders and forget master errors.
This propagation can also use the existing ball evaluator, with the remainder
radius included in each box component. Quadrature truncation remains a
separate estimate until independently bounded.

## Batching, resource limits and proposed stages

Keep numerical data in workspaces and structural programs in immutable plans.
Cache exact integrated programs by order, output/derivative shape and formula
version; instantiated evaluator keys additionally include working precision and
backend. Do not cache results across numerical, precision or derivative
contexts. Group eligible lanes by selected order and evaluate each instruction
across its active lanes, including the positive-majorant and enclosure
programs. Derivative components remain separate from sample lanes.

1. Prepare a validation-only Symbolica program for N=2 and N=4, first primal,
   then one common-mass derivative and one mixed invariant/mass shape. Keep
   the independent fixed-node oracle and the failed DNS records. Bound symbolic
   preparation time, term count and compiled instructions explicitly.
2. At the same exact Float256 node, verify the integrated polynomial, rigorous
   rounding enclosure, tail certificate and every derivative component.
   Compare exact vacuum derivatives, independent mass-derivative identities,
   and higher-precision finite differences away from branch boundaries. Add
   an independently integrated parameter quadrature check so a shared series
   construction is not the only numerical oracle.
3. Exercise unequal near-common primal masses, transverse and higher/mixed
   jets, complex phases, permutations, branch-boundary rejection, and scaled
   copies. Validate tails at multiple orders and precisions. Require explicit
   failure when a full-jet budget cannot be satisfied. Huge shapes/orders must
   hit a documented resource limit, never incomplete zero coefficients.
4. Only after those checks, add the guarded branch to oneloopmaster or the
   scalar-master adapter, with ROOT coordinating the source owner. Retain the
   existing general formulas when the analytic/certificate domain does not
   apply. Successfully certified series lanes must bypass the general-formula
   derivative-discriminant rejection in `iregi-masters/src/lib.rs`; otherwise
   exact-vacuum seeded boxes are still rejected even after the new parameter
   proof establishes regularity. Preserve the existing physical input-domain
   checks. Test interleaved ordinary/series/rejected lanes and batch tails.
5. Revisit E6 on the original exact fixture only after the D0 branch qualifies.
   Preserve the existing E6 failures, isolate its remaining geometry precision
   loss, and propagate master error bounds before making an E6 accuracy claim.
   Benchmark preparation and scalar evaluation independently before enabling
   this route by default for any broader family.

Independent review by `/root/pentagon_precision/e6_series_review` confirmed the
mathematical bounds and native Dual convention. Its clarifications about
higher-order tail powers, rectangular enclosure acceptance and derivative
dispatch are incorporated above. The first qualification is recorded in
`validation/near-vacuum-box/qualification.json`: order four certifies 24
components across four selected lanes at requested 128 bits with 256-bit
working arithmetic; order two correctly rejects insufficient mixed-derivative
tails. The moderately small-invariant overlap is certified separately at
24 bits, including agreement with the ordinary DNS primal formula. Historical
preparation timeouts and the intermediate resource-limit rejection remain
visible. The subsequent package also qualifies primal/mixed f64 at 40 requested
bits, DoubleFloat at 80, Float128 at 100, and Float256 at 128; heterogeneous
exact-vacuum jets through degree three; and unequal masses distinguished beyond
53 bits. There are 15 distinct settings plus three final source-provenance
repetitions. Independent quadrature, broader finite differences, production
integration, full requested 256-bit convergence, and performance remain
separate gates.
The validation implementation is in `examples/support/near_vacuum_box.rs`, with
independent oracles and drivers beside it. This proposal does not authorize a
production change by itself.
