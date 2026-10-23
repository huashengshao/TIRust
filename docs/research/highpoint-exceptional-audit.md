# Exceptional high-point reduction audit

This is a source audit and a set of derived test fixtures, not runtime
qualification. No Symbolica evaluator or new numerical comparison was run for
this note. The public API still lacks complete coverage of finite lower-rank
high-point kinematics; raising precision cannot repair an exactly singular
algebraic representation.

A subsequent PV prototype adds a constant denominator identity before the
pentagon Gram inverse. Literal or structurally constrained coordinates use
Symbolica's exact solve; generic ordinary coordinates have four lazy rank-three
charts. Public activation remains disabled until the formula tests and exact
input certificates are qualified. Generic Dual branches remain disabled until
the complete rank constraint is certified in every Taylor coefficient. The
table below describes the public path and the gaps motivating that prototype.

For comparison, the ordinary full-rank six-point fixture now has a complete
[precision adjudication](../../validation/pv-high-rank/oracle-runs/n6-base-precision-adjudication.json).
Its 256-bit native imaginary component 321 is
`-0.004930291971105481824004352852333172450213477064209421463462981704747608819063664650231`,
within approximately `3.2e-18` of the separately generated
[simplex oracle](../../validation/pv-high-rank/sixpoint-simplex-oracle.json).
That oracle's estimated uncertainty remains `6.07e-14`; it has not been
retuned to the reduction result. This full-rank agreement does not close the
lower-rank coverage gaps below.

## Existing primitives and missing dispatch

| Domain | Existing implementation | Remaining limitation |
| --- | --- | --- |
| Ordinary N=5 Cartesian tensors | `PvCompiler::moment` reconstructs one numerator component using the full four-vector basis and `geometry().inv()` | Rank above zero requires an invertible 4-by-4 Gram matrix. A planar or collinear pentagon can have finite, nonzero transverse moments even when that inverse does not exist. |
| N=5 scalar rank zero | `PvCompiler::scalar` delegates to `ScalarCompiler` | Exact constrained scalar matrices can use exceptional Cayley recurrences. A generic symbolic plan evaluated on a singular lane does not automatically acquire those structural constraints. |
| Ordinary N=6/7 tensors | `denominator_identity` solves the constant-one relation. N=7 has six lazily selected full-rank charts. | A homogeneous-only relation is not accepted by this PV path. The six full-rank charts do not cover lower-rank systems. A lower-rank numeric system with a consistent constant-one relation can already succeed through `solve_any`; rank deficiency alone is not sufficient to characterize failure. |
| Scalar N>=6 with routing | `ScalarCompiler::routing_relation` tries a constant-one solve, then constrained homogeneous solves. `compile_routing_relation` pinches or transfers powers. | The capability exists in the scalar compiler, but generic coordinate/invariant aliases hide sample-specific rank and homogeneous constraints. It is not a complete runtime tensor fallback. |
| Singular Cayley matrices | `compile_exceptional` uses invertible augmented S or constrained null vectors; `sum_b==0` uses the pinch identity without division by B | These paths require exact structural singularities. They do not classify arbitrary sampled Gram/Cayley degeneracies. |
| Repeated denominators | `degenerate_family` combines powers when the active Cayley rows are exactly identical. Existing tests cover coincident unit and raised powers. | Generic plans do not encode equality classes of equal nonzero masses and shifts. Current zero-invariant profiles only describe zero masses and pair distances. |
| Supplied high-point invariants | Both `ScalarReductionPlan::new` and `ReductionPlan::build` reject N>=6 supplied scalar products | This is an explicit, safe API gap. `invariant_certificate.rs` is not connected to the public path, and its regular rank-four charts intentionally return `NoRegularChart` for lower rank. |

Relevant source locations are `pv.rs` functions `moment`, `geometry`, and
`denominator_identity`; `ibp.rs` functions `compile`, `degenerate_family`,
`routing_relation`, `compile_routing_relation`, and `compile_exceptional`;
`plan.rs` functions `build` and `compile_ibp`; and the exact-zero profile dispatch
in `execution.rs`. `Algorithm::Ibp` may complete ordinary high-point tensors
through PV. A hybrid plan explicitly omits a nominal IBP fallback when that
fallback itself uses PV. Consequently the algorithm selector alone does not
establish an independent exceptional path.

`compile_ibp` currently introduces independent squared-distance aliases. Even
when their definitions later evaluate to constrained coordinates, a generic
symbolic matrix solve does not know that a planar relation or mass equality
holds on a particular lane. The existing zero-invariant dispatch cannot express
arbitrary linear dependence or equality of nonzero masses.

## Valid fallback using existing algebra

For `D_i=(q+p_i)^2-m_i^2`, define `k_i=p_i-p_0` and
`f_i=k_i^2-m_i^2+m_0^2`. Form the five-row matrix whose columns are `(k_i,f_i)`,
for `i>0`. All matrices, ranks, solves, and simplifications should remain owned
by Symbolica.

1. Solve `A*x=(0,0,0,0,1)`. Set `c_i=x_i` for `i>0` and
   `c_0=-sum(i>0,c_i)`. This yields `sum_i c_i D_i=1`. It is valid at any tensor
   rank and dimension and directly reduces one propagator power. It is useful
   for singular N=5 systems as well as N>=6.
2. If that system is inconsistent, find a nonzero solution of `A*x=0` by
   appending a unit constraint, as `ScalarCompiler` already does. Then
   `sum_i c_i D_i=0`. For any nonzero `c_a`,

   ```text
   J(v) = -sum(i!=a) (c_i/c_a) J(v+e_a-e_i).
   ```

   The numerator is unchanged. Choosing a nonzero coefficient with maximal
   propagator power gives the existing scalar compiler's terminating transfer
   strategy; eventually another denominator is pinched.
3. Homogeneous transfer creates raised powers, so the present unit-power
   `PvCompiler` cannot continue directly. Reuse the existing tensor generating
   polynomial in `compile_ibp` and `ScalarCompiler` for those children. Do not
   add a second symbolic tensor or partial-fraction engine.
4. For a singular N=5 external span, there is a nontrivial vector annihilating
   the coordinate rows. Its contraction with `f` either yields a constant-one
   relation or a homogeneous relation. Therefore these two paths cover the
   denominator dependence without inventing a Gram pseudoinverse. Descendant
   boxes and triangles can themselves be exceptional and need the same scalar
   infrastructure.

A Gram pseudoinverse alone is incorrect: reconstruction along the external
span leaves transverse tensor moments undetermined. The repeated-denominator
fixture below has zero external span and nonzero quadratic moments. The
existing Davydychev generating polynomial retains those metric terms.

Implementation should first expose constrained structural families and group
compatible lanes into their own batch evaluator. Merely retrying a singular
sample in higher precision, or calling a scalar reducer in an outer lane loop,
does not satisfy the requested fallback/batching semantics. A broader
rank-adapted transverse-projector method is possible, but is a larger change
than reusing the already implemented scalar relations.

## Equality and Dual requirements

Coincident coordinate denominators may be merged only when their shift
four-vectors and squared masses agree as complete Taylor values. For an
invariant scalar family, equality of the complete active Cayley rows is the
relevant exact predicate. Preserve the surviving anchor shift and sum the
propagator powers. This preserves translated Cartesian moments through the
existing generating polynomial.

Equality of primal values alone is insufficient. For example, masses
`m0^2=4+t` and `m1^2=4-t` coincide at `t=0` but their mixed/second derivatives
are not represented by blindly merging the two propagators into a single
mass-dependent power. Coalescing transverse jets need an explicit analytic
expansion, analogous to the existing bubble jet infrastructure. The same
restriction applies to a Dual direction that leaves a planar Gram variety.
The profile key must include equality/rank structure and derivative layout;
numerical values remain sample inputs. Small nonzero determinants need
conditioning diagnostics or analytic expansions, not exact-zero projection.

## Derived fixtures

### Distinct planar pentagon: preferred first regression

Use `mu=3`, unit powers, D=4, rank two initially, and

```text
p0=(0,0,0,0), p1=(0,1,0,0), p2=(0,0,1,0),
p3=(0,0,0,1), p4=(0,1,1,1)
m^2=(4,9,16,25,36).
```

The external span has rank three. All four-point pinches are nondegenerate
spacelike tetrahedra, so this fixture isolates the N=5 gap rather than requiring
singular box handling. Every mass is positive and all pair distances are
nonpositive, hence `F>=4` throughout the Feynman simplex. The exact identity is

```text
2*D0 - D1 - D2 - D3 + D4 = 6,
T5[P] = (2*T_pinch0[P] - T_pinch1[P] - T_pinch2[P]
         - T_pinch3[P] + T_pinch4[P])/6.
```

This holds for every requested Cartesian monomial `P`, including the nonzero
transverse `q0^2` moment, with no scalar-master approximation. It supplies a
separate box-only comparison for every Laurent component. All poles vanish
through rank five by UV power counting and positive F. Rank six and seven
must retain possible UV terms. The current exact N=5 PV positive-rank path
rejects the Gram inverse by inspection; a new runtime regression remains to be
run under the shared Symbolica license.

### Homogeneous-only six-point family

Use `p=(0,e_x,e_y,e_z,e_t,e_t+e_x+e_y+e_z)`, squared masses
`(16,15,15,15,17,14)`, and `mu=3`. These obey `m_i^2=p_i^2+16`, so every `f_i`
is zero and no constant-one denominator identity exists. Nevertheless,

```text
3*D0 - D1 - D2 - D3 - D4 + D5 = 0.
```

All masses are positive; the time components lie in `[0,1]`, which gives
`F>=14-1/4=13.75`. This is a regular integral with a homogeneous denominator
relation. It should exercise `ScalarCompiler` power transfer instead of the
PV constant-one solver. Existing `ibp.rs::routing_relations_satisfy_exact_denominator_identities`
already tests the same construction with mass offset 7, but does not establish
public batched tensor coverage of this sample. A seventh shift with
`m_6^2=p_6^2+16` gives the corresponding rank-deficient N=7 test.

### Identical-denominator analytic check

For N=5, all `p_i=0`, all `m_i^2=4`, unit powers and rank two, the integrand
has denominator `(q^2-4+i0)^5`. Its OneLoop-normalized values are

```text
scalar = -1/768,
q0^2   = +1/768,
qx^2 = qy^2 = qz^2 = -1/768,
all linear and mixed quadratic moments = 0.
```

Multiply by `i*pi^2` for native IREGI; all poles vanish. These values follow
from the ordinary vacuum Gaussian integral, or from
`q^2=(q^2-4)+4` and Lorentz symmetry. Compare against the one-propagator
request of power five as well. A common nonzero shift tests preservation of
the anchor routing; a common Dual mass tests tangential derivatives. Independent
mass jets that only coincide in the primal require a separate limit test.

## Existing algebra checked

Symbolica's Numerica matrix implementation provides `rank`, `solve`,
`solve_any`, `inv`, and `det` in
`references/symbolica/lib/numerica/src/tensors/matrix.rs`. Its in-tree tests and
`lib/numerica/tests/api_regressions.rs` exercise rank and underdetermined
`solve_any`, and this project's scalar recurrences already use those exact APIs.
No handwritten elimination is needed.

The FeynKit/HEPKit equivalents were also inspected.
`crates/feynkit-graph/src/integrals.rs` provides `IntegralFamily` rank and
propagator-coordinate solves, but requires a dependent family to be
partial-fractioned before completing it. Its
`integrals/partial_fraction.rs` submodule supplies that operation through
`IntegralFamily::partial_fraction(powers, max_states)`: it handles homogeneous
and constant relations, repeated powers, support-based relation caching, and
explicit state-budget failures. Its tests reconstruct the original rational
function. This existing facility must be considered before adding a separate
partial-fraction implementation; it does not itself integrate the resulting
scalar families or dispatch numeric/Dual constraint profiles. An earlier
version of this audit missed that submodule.
`crates/feynkit-tensor/src/reduction.rs` supports
external-basis projection and symmetry reductions; its external Gram basis
must itself be invertible. It can help organize contractions after choosing a
valid smaller basis, but is not an automatic cure for the current singular
N=5 reconstruction.

The scalar recurrence background is
[Duplancic–Nizic](https://arxiv.org/abs/hep-ph/0303184), whose method treats
vanishing kinematic determinants. Alternative tensor schemes and small-Gram
expansions are discussed by
[Denner–Dittmaier](https://arxiv.org/abs/hep-ph/0509141). The proposed immediate
fallback above is inferred from the exact denominator identities and existing
source primitives; this note does not claim those external papers already
supply an implemented Rust dispatcher.
