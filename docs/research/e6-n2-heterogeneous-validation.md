# Next private E6 N2 validation: independent invariant and mass jets

Source-only proposal, 2026-10-23, inspected at
`e3669d08e81e24efdb5f98d32eb0aa2b410b8c74`. No implementation, build,
numerical run, new certificate or timing result is reported here. The typed
checkpoint `validation/e6-direct-n2-typed/qualification.json`, SHA-256
`65d60be9347114c7543f3ba686ae5857d5443a383642d55f8f310ed52f0c9883`,
remains frozen. Its candidate engine SHA-256 remains
`9abf3dc1bbe6c2d2437f6e06d637593cd53a57767107635824b7c0d9560e0513`.

The smallest useful next experiment is one **Float256, requested128,
Mixed6** group using the unchanged N2 engine. Independently integrate exact
vacuum derivatives, then compare a nonzero-routing, unequal-mass point
against a bounded physical Feynman-parameter oracle. Use the existing
oracle's order four; do not prepare a generic N4 candidate or change public
or default E6 dispatch. A 120-second process cap is a proposed resource
limit, not a prediction that this new oracle will finish within it.

## Existing facilities: three reuse checks

| Check | Sources inspected | Reuse and limitation |
| --- | --- | --- |
| Symbolica APIs/source | Vendored `dependencies/symbolica/src/atom/core.rs:694`, `:734`, `:1399`, `:1555`; `src/poly/polynomial.rs:2659`, `:2801`, `:2822`, `:6667` | Derivatives, series, exact rational polynomial conversion, evaluation, polynomial endpoint substitution and antiderivatives already exist. No polynomial, Dirichlet-moment, jet or integration engine is needed. A formal series alone does not certify its remainder. |
| Existing usage/tests | `crates/iregi-rust/examples/e6_parameter_series.rs:118–223`; `examples/support/d0_near_vacuum_oracle.rs:110`, `:122`, `:145`, `:399–470`, `:570`; Symbolica `src/poly/univariate/tests.rs:175` and `src/derivative.rs:903` | Reuse the independent E6 sign/measure and four-parameter integration, the D0 oracle's exact complex coefficient evaluator, normalized derivatives, heterogeneous vacuum jets and positive full-jet remainder. These helpers call neither reduction nor scalar masters. |
| FeynKit/gammaloop/oneloopmaster | `../references/gammaloop_feynkit/crates/feynkit-graph/src/integrals/parametric.rs:364–376`; `examples/reproducers/symbolica-expansion/polynomial_replacement.rs:75–114`; `../references/oneloopmaster/tests/d0_one_mass_width.rs:1–30`; existing searches documented in `e6-certified-path.md` and `e6-near-vacuum-direct.md` | FeynKit supplies `IntegralFamily::symanzik` with explicit Minkowski signs and no inferred measure, powers or prescription; its replacement reproducer verifies Symbolica polynomial substitution. The oneloopmaster reference data concern D0 and are not an arbitrary-precision E6 derivative oracle. No ready certified E6 parameter quadrature was found in the checked sources. |

The first row uses the vendored Symbolica revision recorded by the frozen
dependency closure, rather than line numbers from the newer reference clone.
The direct nested-simplex alternative is already expressible using
`integrate` and `replace_with_poly`; the latter currently recomputes endpoint
powers per monomial. Reuse the already validated cube/Jacobian orchestration
for this smallest experiment. A second coordinate implementation is not
needed to claim independence from the candidate's truncation and recurrence.

## Freeze one physical point and heterogeneous higher jets

Use the existing ancestor-closed shape
`[00,10,01,20,11,21]`. These are native Taylor coefficients, with factorials
already divided out. Let

```
eta = 2^-64, delta = 2^-32, M = 2 + 2^-80 - i,
q0 = 0,
q1 = delta*(1,0,0,0), q2 = delta*(0,2,0,0),
q3 = delta*(0,0,3,0), q4 = delta*(0,0,0,4),
m_i^2 = M + eta*c_i, c = [-2,-1,0,1,2].
```

Symbolica must derive the ten lexicographic squared distances in metric
`(+---)` and check the four-vector Gram determinant is nonzero. The expected
distance list is `eta*[1,-4,-9,-16,-3,-8,-15,-13,-20,-25]`. All ten input
slots are nonzero. This is a physical real routing, unlike an unrestricted
arbitrary symmetric matrix whose inertia might not fit this metric.

Use these nonprimal input coefficients, with every unlisted one zero:

| Physical input | Component | Value |
| --- | --- | --- |
| `s01` | `10` | `eta` |
| `s23` | `01` | `2*eta` |
| `m1^2` | `10` | `eta*(1+i/2)` |
| `m3^2` | `01` | `eta*(-1/2+i)` |
| `m4^2` | `20` | `eta/2` |
| `m2^2` | `11` | `eta/4` |
| `m0^2` | `21` | `eta/8` |

The invariant jets are separate analytic directions in the supplied
invariants; they are not claimed to come from a common variation of the
displayed routing. Mass jets differ among propagators and include complex
and direct higher mixed components. The strictly negative mass widths avoid
the existing zero-width derivative boundary. The direct `20`, `11`, `21`
inputs give leading contributions of order eta, rather than testing those
outputs only below the absolute accuracy floor. Their nonzero size is a
prospective exact-oracle gate below, not an assumed result.

The candidate expands around the represented first physical mass `M-2*eta`.
The independent oracle uses M. Thus their truncations are different even
before comparing N2 with the existing order-four reference. Float256
represents every displayed dyadic input exactly, including the beyond-53-bit
mass difference. Serialize those actual represented values as usual.

## Two independent physical-integral oracles

First use the same nonprimal coefficients with zero primal invariants and
all primal masses equal to M. Reuse the heterogeneous vacuum-oracle pattern
at `d0_near_vacuum_oracle.rs:570`: construct the full physical integrand
`-F(t,x)^-2`, differentiate it with Symbolica in the requested directions,
divide by the factorials, then set `t=0` **before** parameter integration.
Since `F(0,x)=M`, the resulting parameter polynomials integrate exactly.
Their reference remainder is zero. The five-parameter simplex normalization is
`E6=-integral_Delta4 F^-2`, with volume `1/24`.

For the nonzero primal point form independently

```
F(x;t) = M + eta*R(x;t),
R = sum_i c_i(t)*x_i - sum_(i<j) d_ij(t)*x_i*x_j,
sum_i x_i = 1.
```

Build R from the frozen fixture data and routing-derived distances, not from
the candidate's prepared expression or parameter-mapping helper. Use
Symbolica's order-four series in the single bookkeeping ratio `z=eta/M`.
Apply Symbolica normalized derivatives and zero substitution in the two jet
variables, and exactly integrate each requested parameter polynomial. The
formal imaginary-unit variable and rational numerator/denominator evaluator
from the existing D0 oracle preserve complex coefficients over Q. Keep the
symbolic integration over a combined rational polynomial variable map; the
historically slow AtomField coefficient route is unnecessary.

Reuse the existing cube map with its Symbolica-derived Jacobian and four
polynomial antiderivatives. Check simplex normalization, volume, and the
zero-deviation identity `-1/(24*M^2)` explicitly. Sharing Symbolica's algebra
primitives and the standard parameter measure is acceptable; sharing the
candidate's integrated N2 polynomial would not be independent validation.
The order-four reference plus a justified tail encloses the physical
integral. Agreement between two N2 expressions alone would only check the
same finite polynomial.

For each requested monomial alpha, bound its parameter coefficient by the
sum of the L1 norms of **all** its mass and invariant coefficients:
`B_alpha=sum(|Re c|+|Im c|)+sum(|d|)`. This uses `0<=x_i<=1` and is
deliberately different from the candidate's maximum-based `2/5` envelope.
Let `B(t)=sum B_alpha*t^alpha` and `L=max(|Re M|,|Im M|)`. The independent
positive remainder generator is

```
H4(eta*B(t)/L)/(24*L^2),
H_N(z)=z^(N+1)*((N+1)/(1-z)+1/(1-z)^2).
```

Symbolica must verify the generating-function identity and extract the
normalized coefficient for **every** requested component. Each coefficient
is a nonnegative exact rational complex-modulus remainder bound. Because
all physical jets include eta, differentiation in t does not lower the eta
power. A primal-only radius would be incorrect. As a coarse independent
preflight, this fixture gives `B(1,1)<128` and `L>2`, hence
`eta*B(1,1)/L < 2^-58`. Positivity bounds every component tail by
`H4(2^-58)/96`, far below `2^-256`; verify the inequalities exactly in the
future driver instead of treating these source estimates as measured output.

No new reference series order is proposed. The full fifteen-parameter N4
candidate is excluded; fix all physical fixture coefficients before
polynomial conversion. If this finite reference does not fit its resource
cap, preserve the failure and revise the preparation separately.

## Accuracy and failure gates

Retain requested128 and relative tolerance `2^-128`. For this new experiment
strengthen the explicit nonprimal absolute budgets to
`eta*2^-96=2^-160`; keep the primal budget `2^-128`. This changes only the
validation caller's explicit accuracy context and does not relax any old
budget or modify the engine. These proposed budgets must be frozen before
execution.

For each successful component use the independent center C and radius r to
check the returned number and the entire candidate rectangle. A conservative
exact squared test enlarges each real/imaginary coordinate distance to C by
r before summing squares; no square root or approximate comparison is
needed. Require the complete oracle coordinate intervals `C_re±r`,
`C_im±r` to lie inside the candidate rectangle as well. Intersection alone
is insufficient. Recompute the candidate's existing allowed budget and
rounding-plus-tail distance exactly from emitted evidence.

For every nonprimal component require an exact lower bound
`max(|Re C|,|Im C|)-r >= eta/2^15`, an upper modulus bound below eta, and an
absolute allowed budget exactly `2^-160`. Those gates establish at least
81 bits of relative accuracy rather than merely accepting a tiny output
under an absolute threshold. They are additional qualification gates: a
failure stays visible and is not repaired by omitting the component,
weakening the budget, or changing the fixture within the same attempt.

The candidate's original full-jet tail and ball-rounding checks remain
authoritative for exposing a successful value. Retain the large common-mass
linear-jet cubic rejection with zero primal deviation; exact physical
derivatives can be nonzero where the N2 polynomial is zero.

## Smallest bounded batch and retained evidence

Use an odd five-row primary batch:

1. Exact vacuum with the heterogeneous jet pattern.
2. Nonzero routing and unequal masses with that pattern.
3. The same physical integral after a propagator permutation, permuting
   every mass and invariant component together. Its first physical mass
   changes, so the candidate's frozen auxiliary center changes too.
4. An upper-half-plane mass in an otherwise valid neighboring row.
5. The existing large-jet cubic rejection control.

The planned counts are five scale-valid packed rows and four active series
lanes, with three successful rows and two hidden-value rejections; these are
expected assertions, not observed results. Check the permutation of the
physical F polynomial with Symbolica, separately from output agreement.
Replay the nonzero row with changed physical `mu_squared` and a scale jet,
and check unchanged finite results. Preserve empty, reversed, chunked,
fresh-workspace and singleton comparisons, plus two independent concurrent
workspaces. Record numeric/ball/tail batch-call counts and active rows;
label exported instruction/lane products as structural estimates, not traces.

After source review, a private native-object build can have a 60-second
cap and the single correctness process a 120-second cap, with stage logs
for exact oracle construction, each integration, candidate preparation and
batch checks. Retain the existing finite symbolic term/instruction limits;
the external cap must also protect a conversion before its post-conversion
term count is available. Reuse the frozen dependency closure and engine hash;
no Cargo dependency rebuild or unreviewed algebra fallback is needed. On a
substantive failure stop, keep the source, binary, raw log and failed state,
and do not reinterpret a timeout as a numerical disagreement. These are
correctness limits, not benchmark timings.

## Reference choices that do not close this proof

Original IREGI's real scalar-matrix entry point exists in
`../references/madgraph/vendor/IREGI/src/si_reduce.f90:17`, but its genuine
D6 unit-power pentagon uses the known `IBP2PAVE -> E00` shortcut at lines
212–218; the coordinate path at 791–797 and complex analogue at
`csi_reduce.f90:216–223` have the same issue. The factorial initialization
repair does not fix this. Existing evidence in `shifted-pentagon.md:163–215`
and `validation/shifted-pentagon/adjudicate.py:134–174` records
stable-reported disagreement and a failed mass-derivative identity.
That result is diagnostic, not a correctness oracle for this experiment.

The current Fortran harness `validation/reference/driver.f90:85–95` accepts
real four-vectors, not arbitrary invariant matrices. Even a new matrix
protocol would not cure the D6 issue. The independent identity
`sum_i dE6/d(m_i^2) = -E4` remains useful, but its existing production
higher-derivative test at `crates/iregi-rust/tests/shifted_pentagon.rs:196–217`
is ignored and it would add a second master/oracle
family to this smallest experiment. Keep it as a later cross-check.

Likewise, existing E6 successive-rule quadrature differences are estimates,
not rigorous derivative bounds. The one-point shifted-box enclosure proof
does not certify upstream geometry or quadrature. Neither should replace
the exact parameter oracle here. This proposal makes no claim of general
physical kinematics, unequal-mass coverage beyond its fixture, generalized
powers or dimensions, further numeric types, or public/default E6 support.
