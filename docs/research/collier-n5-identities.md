# Independent identities for the admitted N5 comparisons

Source proposal, 24 October 2026. No preparation CLI, reference kernel, new
build or timing has been run for this proposal. Its initial targets are the
six N5 rank0/1/2 singleton/batch32 rows admitted numerically in shared
attempt004. The source-domain proofs cover positive real masses and the
three retained conditioned momenta; inputs and all failures remain unchanged.

## Reuse and algebra ownership

Reuse the existing implementation in
`validation/collier-reference/n4/check_identities.py`: it already obtains
metric-trace, denominator-difference and translated-numerator coefficients
through the Symbolica `prepare` CLI, contracts prepared coefficients against
native Cartesian moments, and retains rejected reference diagnostics separately.
Do not copy its direct Python scalar construction of invariant polynomials into
the extension: ask Symbolica to prepare the exact invariant and shift constants
as well as the numerator polynomials. Python may enumerate indices and gather
wire inputs; it must not implement a new polynomial expansion or matrix solver.

The three reuse checks are:

1. Symbolica APIs/source: `AtomCore::expand`, replacement machinery and
   `to_polynomial_in_vars_with_field` already perform expansion, shifts and
   coefficient extraction. `crates/iregi-prepare/src/lib.rs` exposes
   `polynomial_from_expression` and scalar tensor-component preparation.
2. Existing usage/tests: the N4 identity checker and its committed
   `diagnostics-v1/identity-complete.json.gz` contain actual Symbolica-prepared
   cancellation, trace and pinch-zero translation witnesses. The preparation
   library has graph/numerator tests and retains every exact nonzero term.
3. FeynKit equivalents: `feynkit-graph/src/integrals/mapping.rs` provides
   `IntegralFamily::mapping_to`, `find_mapping`, `quadratic_denominator` and
   `MomentumMapping::apply`; routing exposes `contracted_momentum_basis_of`,
   `momentum_replacements` and `route_expression`. Family denominator maps
   and powers already encode deletion of a denominator. Idenso/Spenso own
   metric/index contractions. These source checks establish reuse candidates;
   match them to the locked dependency revision before introducing a new call.

No missing algebra operation has been identified or is proposed. Prefer the
already available CLI for the first independent witness, binding its executable
hash and explicitly recording whether its source/build provenance is contextual
or source-matched. Do not silently rebuild or upgrade the dependency closure.

## Analytic identities and dimensional scope

Write `D_i=(q_D+p_i)^2-m_i²+i0`, `p_0=0`, and let `E^alpha` be the N5 tensor
with numerator in the original four-dimensional loop components. Let `D_i^alpha`
be the N4 tensor formed by deleting denominator i, with the same original
numerator. This superscript is a tensor moment, not a propagator power.

For i=1,...,4, Symbolica must establish the exact integrand cancellation for
`f_i=p_i²-m_i²+m_0²` and generate coefficients for

```text
2 p_i · E^(vector) = D_i^(scalar) - D_0^(scalar) - f_i E^(scalar),
2 p_i_mu E^(mu,nu) = D_i^nu - D_0^nu - f_i E^nu,   nu=0,...,3.
```

These same-D denominator-difference relations test the odd-rank sign, metric
signs, complete mixed components, denominator order and pinch routing without
an evanescent remainder. Use every native Laurent slot and each independent
vector component; an agreement only after summing components is insufficient.

For rank2 the metric trace is, through the finite Laurent term,

```text
E^(00)-E^(11)-E^(22)-E^(33) = D_0^(scalar)+m_0² E^(scalar).
```

As in the existing N4 analysis, this is not an identity to all orders in
epsilon for a four-dimensional numerator. Its exact defect is proportional to
epsilon times the six-dimensional scalar N5 integral. The latter has degree
`6-2*5=-4`, is UV convergent, and is IR finite under the retained positive
Feynman-polynomial bound. Therefore its Laurent poles vanish and it contributes
no correction through the finite coefficient here. Preserve this derivation in
the executable witness; do not extrapolate it to higher ranks or other domains.
The existing N4 derivation and the denominator definitions in COLLIER source
provide the convention cross-check.

There is no time-reflection zero test for these fixtures: their time components
are nonzero. Reusing the purely spatial N4 time-reflection test would be invalid.

## Pinched input and translated-numerator preparation

For deletion i>0, the remaining origin is already zero. For deletion0, choose
`q'=q+p_1` and shifts `[0,p_2-p_1,p_3-p_1,p_4-p_1]`, with unchanged ordered
squared masses `[m_1²,m_2²,m_3²,m_4²]` and scale. Have Symbolica prepare all
exact shifted components and the four local numerator polynomials
`q'_nu-p_1^nu`; use these to recover `D_0^nu` in the original q routing.

All five N4 pinches need moments only through rank1. The existing immutable
original-IREGI executable supports those inputs directly. Capture Hybrid,
PV and IBP separately to expose any lower-point mode disagreement. Use only
accepted complete N4 responses as identity RHS evidence. A rejected COLLIER
pinch may be retained as an optional diagnostic, but never used to rehabilitate
its own status or to substitute for a missing accepted RHS.

There are15 distinct pinch inputs (five deletions times the same three exact
conditioned samples), grouped by unchanged scales3/6. A batch32 identity can
reuse these exact identical-input RHS values in its original lane order, with
explicit proof-key matching and a clear statement that there are three distinct
kinematic branches. Never present this deduplication as32 independent branches.

## Routing and propagator-order witnesses

As separate checks, cyclically move denominator1 to the origin, using
`q'=q+p_1` and the corresponding cyclic ordering of masses and shifts. Have
Symbolica prepare every original numerator monomial through rank2 as
`(q'-p_1)^alpha` before contracting the new full tensor array. Compare every
reconstructed original Cartesian moment and all four Laurent slots against
the retained parent array, in each accepted reference policy.

Also reverse the nonzero denominators while retaining denominator0 and the
same q variable. Every original moment should remain identical within the
unchanged dimension-normalized1e-10 threshold. Preserve each new routing as a
distinct labelled verification input; do not substitute it into the original
benchmark family or erase a rejected routing.

## Proposed bounded qualification stages

First freeze an independently reviewed producer and exact request inventory.
Have the existing Symbolica CLI prepare the cancellation/translation/metric
and invariant polynomials under a60s preparation deadline (20s per request),
preserving every DOT, prepared coefficient array and CLI error. Bind the original
three input proof keys and the five parent denominators explicitly. No reduction
or custom algebra is required during this phase.

Then propose a separate correctness-only reference capture: five pinches times
two scale groups times three original IREGI modes gives30 requests. The cyclic
routing and reversed-order witnesses, each with two groups and three IREGI modes
plus COLI, add16 requests. No library rebuild is needed. Use CPU26,20s per
request,120s total reference deadline and135s outer cap; preserve all46 requested
entries if the cap is reached. This is a proposal, not execution authorization.

Evaluate all prepared identities against both the retained accepted COLI N5
arrays and original-IREGI N5 arrays, using independently captured accepted
pinched RHS values. Check every native Laurent coefficient at dimensionally
normalized1e-10, report cancellation magnitudes, and keep all original engine
flags alongside the identity result. An identity pass cannot override a native
failure. Audit source/input/binary bindings and the complete lane/permutation
gather before promoting the six N5 rows beyond numerical admission. Performance
cohorts remain separate and require their own controlled, reviewed protocol.
