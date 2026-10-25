# Generalized N6/N7: exact routing solves and FeynKit affine decomposition

Source-only review, 2026-10-25. No production code was changed and no build,
Symbolica evaluation, numerical reference job, or timing experiment was run.
The proposed tests below are **not test results**. This closes the research
question in HANDOFF §9.4, not the generalized high-point acceptance gate.

At the root agent's subsequent request, a private source-only prototype is now
at `target/routing-rational-proposal/{README.md,proposal.json,proposal.patch}`.
It selects a deliberately narrower explicit-affine matrix subset (at most five
rows/six columns), admits only the caller's zero/one RHS entries, and otherwise
preserves AtomField. Thirteen focused exact tests plus three existing routing
tests are listed for future execution; none was executed here. The new helper
was formatted and both source files parsed by rustfmt; the FeynKit extraction's
fourteen static provenance checks passed without loading Symbolica. Independent
source review corrected an earlier unbounded exponent-admission issue. No
performance, compilation or numerical result is implied by this proposal.

## Recommendation and smallest next patch

First prepare a private patch changing **only the coefficient domain of
`ScalarCompiler::routing_relation`'s existing solves**. Use Symbolica's
`RationalPolynomialField` for structurally eligible exact entries, convert final
coefficients back to `Atom`, and retain the same constant-first,
constrained-homogeneous fallback, active-index ordering, aliases and cache.
This requires no new recurrence, graph wrapper, numerical workspace or dependency.
It benefits the existing arbitrary-power/dimension-shift route directly.

After that patch is independently qualified, consider FeynKit's existing
`AffineFamily::partial_fraction` as a separately measured bounded preparation
alternative. Do not implement another partial-fraction engine. Its benefit would
be coalescing exponent states before compiling descendants; it does not introduce
an identity missing from the current scalar compiler.

The restricted public `SixPointReductionPlan` and private N7 dispatcher should
keep their current accepted domains until separately qualified. A faster generic
scalar solve does not authorize dots, shifted dimensions or Duals in those APIs.

## What is already implemented

`crates/iregi-rust/src/ibp.rs` contains:

- `ScalarRequest` (line 40): original-order unsigned powers, zeros for pinches,
  and `dimension_shift=h` for `D=4+2h-2ε`.
- `compile_uncached` (line 462): for at least six active propagators with routing,
  use an exact routing relation **before** Cayley inversion, regardless of the
  propagator powers or `h`.
- `routing_relation` (line 734): solve five rows of relative shifts and
  `f_i=R_i0-R_00`, first against `(0,0,0,0,1)`, then constrained homogeneous
  systems if needed. Every solve is currently `Matrix::solve_any` over
  `AtomField`; its custom normalization calls `together()` after algebraic
  operations.
- `compile_routing_relation` (line 830): lower a power for an inhomogeneous
  relation or transfer powers for a homogeneous relation. Its recurrence cache
  already keys the complete `ScalarRequest`, while relations key active support.
- `routing_relations_satisfy_exact_denominator_identities` (line 1734) and
  `routing_reduction_preserves_generalized_powers_and_negative_dimensions`
  (line 1898): existing exact source tests, including N6 raised powers and `h=-1`.

The ordinary generic `ReductionPlan` accepts positive propagator powers and
dimension shifts; its nonordinary preparation selects the generalized scalar
compiler. It is not subject to the public thin N6 rank-three/unit-power restriction.
See `plan.rs:300`, `plan.rs:435`, and `compile_scalar_tensors` at `plan.rs:887`.
The latter introduces squared-distance aliases before constructing `R`, registers
preserved Cartesian routing or certified covectors, and enables genuine D6
pentagon leaves for generalized requests.

The actual generic call chain is `compile_ibp` (`plan.rs:820`) **or**
`compile_generalized_pv` (`plan.rs:850`) → `compile_scalar_tensors`
(`plan.rs:890`). At lines 914–920 the latter always assigns bare `s{i}_{j}`
symbols to off-diagonal distances and saves each original quadratic coordinate
expression as an external alias definition. It forms
`R_ij=s_ij-m_i²-m_j²` at lines 923–930, then calls `set_routing` at 953 and
`enable_aliases` at 957. Its final routing row is therefore
`R_i0-R_00=s_i0-m_i²+m_0²`, affine in the aliases and masses; the first four
rows are affine `p_i^μ-p_0^μ`. This is the intended proposal's eligible path.

Crucially, `ScalarCompiler::with_aliases` itself only enables recurrence-node
aliases; it does **not** create squared-distance aliases. A direct caller
passing raw quadratic coordinate expressions in `R` correctly retains the
old AtomField route under this narrow prototype. Added actual-compiler API
tests use `with_aliases` and `set_routing`, with a thread-local cfg(test)-only
domain trace, to distinguish fully generic N6/N7 distance-alias input from an
unaliased quadratic exact family. They also check the relation cache and all
six coefficients of the latter. These are proposed tests, not observations.

Consequently, generic coordinate N6/N7 scalar preparation has no missing generic
denominator relation: a six/seven-denominator family is overcomplete in the five
affine loop variables below. Existing generic N6 square/N7 underdetermined
systems have solutions. Real problems remain expression growth, sample-specific
singular pivots, constrained specializations, descendant stability, and genuine
shifted masters. No new actual dotted N6/N7 preparation was timed in this review;
do not label an unexecuted request as a structural failure or a timeout.

## Three explicit algebra reuse checks

1. **Symbolica APIs/source.** `dependencies/symbolica/src/atom/core.rs:459`
   exposes `AtomCore::apart_multivariate`; its rational-polynomial implementation
   is `src/domains/rational_polynomial.rs:1403`. It factors/reduces general
   multivariate rational functions, rather than returning bounded original-order
   propagator exponent vectors. Existing exact linear algebra is reexported as
   `symbolica::tensors::matrix::Matrix`; its implementation is
   `dependencies/symbolica/lib/numerica/src/tensors/matrix.rs`, with `solve`
   at 1949, `solve_any` at 1990, `rank` at 2023, `det` at 1256 and `inv` at 1731.
   No local elimination or rank algorithm is needed.
2. **Existing Symbolica usage/tests.**
   `dependencies/symbolica/examples/solve_linear_system.rs::solve_from_matrix`
   converts all entries with a shared `Arc<Vec<PolyVariable>>`, constructs a
   rational-polynomial field, calls the library solve, and converts results with
   `to_expression()`. Matrix in-file `solve`, `solve_any` and
   `solve_fraction_free` tests are at 2079, 2121 and 2282. The retained private
   N7 implementation at
   `target/public-highpoint/thin-n7-rational-domain/core/crates/iregi-rust/src/pv/thin_seven.rs`
   already supplies `geometry_variables`, `exact_geometry_matrix` and
   `solve_chart_rational`, with exact solution and determinant-domain checks.
   Its rational-field success is evidence for that private geometry only, not
   a measurement of the proposed scalar solve.
3. **FeynKit/gammaloop equivalents.** The pinned supplied source has
   `crates/feynkit-graph/src/integrals.rs` and
   `integrals/partial_fraction.rs`. The already extracted workspace package
   `dependencies/feynkit-integrals` preserves the affine engine, coefficient
   field aliases, four exact polynomial/matrix helper methods and provenance.
   Its `src/partial_fraction.rs` uses only Symbolica algebra. Existing tensor
   projection in `crates/feynkit-tensor/src/reduction/external.rs` and
   `orbit_projector.rs` also delegates matrix algebra to Symbolica; it does not
   replace the denominator decomposition or resolve singular Gram geometry.

See `dependencies/feynkit-integrals/{UPSTREAM.md,VALIDATION.md}` for extraction
boundaries and recorded validation. The package is already a root workspace
member and in `Cargo.lock`, but **is not a dependency of `iregi-rust`** and is
not called by its current core. Do not confuse workspace membership with an
integrated reduction path.

## Exact identities and the dimensionally regulated loop square

Use the current routing/metric convention, `η=diag(+,-,-,-)`, and write

```text
D_i = L + 2 Σ_μ η_μ p_i^μ q^μ + p_i² - m_i² + i0,
L   = q_D²,                 D = 4 + 2h - 2ε.
```

For affine decomposition, `L,q⁰,q¹,q²,q³` are five independent **bare symbols**.
Do not replace `L` by the Cartesian square: that loses the perpendicular loop
components and invalidates shifted-dimensional/tensor identities. Coefficients
`c_i` in `Σ c_i D_i=C` must obey

```text
Σ c_i = 0,     Σ c_i p_i^μ = 0,     Σ c_i(p_i²-m_i²) = C.
```

The first condition also cancels the common `i0`. All products here are full
Taylor products for Dual input. With `F_ν=Π D_i^(-ν_i)` and the same numerator
on every term:

```text
C != 0:  F_ν = Σ_i (c_i/C) F_(ν-e_i).
C == 0:  F_ν = -Σ_(i!=j) (c_i/c_j) F_(ν+e_j-e_i),  c_j != 0.
```

These identities preserve `h`, numerator rank, loop routing and Laurent-slot
ordering. An inhomogeneous step lowers total denominator power; homogeneous
transfer preserves it and ultimately removes positive support. FeynKit chooses
the first dependent row as pivot and orders exponent vectors monotonically,
coalescing paths with `BTreeMap`. Keep that existing bounded state machine.
Homogeneous input can create dotted descendants even from unit powers.

The formulas act linearly on every native `[UV1,IR1,IR2,finite]` slot. Keep
scaleless descendants and their cancelling UV/IR poles; the affine package does
not decide scalelessness, integration contours or normalization.

### Dimension shifts are not removed by partial fractions

At most five independent positive-power denominators remain, but a generic
unit-power pentagon at `h=1` is a genuine finite D6 master. The existing
`MasterLeafKind::SixDimensionalPentagon`, its qualified-domain numerical backend,
and propagated quadrature estimates remain necessary. A four-dimensional
Cartesian tensor completion cannot stand in for that scalar. Higher `h` can
descend to the same independent master. Negative shifts can use the existing
dimension/power recurrences; partial fractions alone do not extend their domain.

For comparison, existing `ibp.rs:624` implements, when its divisors are admissible,

```text
J_h(ν) = [J_(h-1)(ν) - Σ_i b_i J_(h-1)(ν-e_i)]
         / [B (3+2h-|ν|-2ε)],
b=R^(-1)1, B=Σb_i.
```

`J_h` is the code's π-normalized scalar, not the native integral with its
`iπ^(2+h)` factor. Preserve the existing separate `B=0`, singular-Cayley and
zero-ε-independent-divisor branches and epsilon-to-finite pole mixing. An
independent useful dimension/power test in this normalization is
`J_h(ν) = -Σ_i ν_i J_(h+1)(ν+e_i)`, obtained from `Σx_i=1` in the Feynman
parameter representation. This is a test identity, not a proposal to introduce
another dimension recursion or bypass unavailable masters. The existing
`lowering_tadpole_dimension_matches_raising_its_power` at `ibp.rs:2260` is its
one-denominator instance.

## Minimal rational-field routing patch

Keep `routing_relation`'s current matrix and RHS construction, including distance
aliases and certified covectors. For a private eligible branch:

1. Gather one complete ordered variable map for **all** matrix entries, RHS and
   added homogeneous unit-constraint rows. Include `s_i_j` aliases present in
   the actual scalar compiler; the restricted N7 helper's map of shifts/masses
   alone is insufficient here.
2. Convert once through Symbolica's exact
   `try_to_rational_polynomial(&Q,&Z,Some(map))` and use
   `RationalPolynomialField<IntegerRing,u16>`. Build each augmented constrained
   matrix by appending existing field entries, with the same fixed-index order.
   Call library `solve_any`, **not** `solve`: N7's generic system is
   underdetermined and its free variables must still be set by the library.
3. Keep the same constant-first and homogeneous searches. Verify
   `matrix * solution == rhs` exactly in the field. Convert only final values
   with `to_expression()`, prepend `c_0=-Σ_(i>0)c_i`, then call the unchanged
   alias/cache code.
4. Select domain eligibility structurally before solving. The N7 restricted
   helper excludes opaque/nonpolynomial inputs and uses an AtomField
   compatibility branch for exact complex fixture coefficients after Gaussian
   validation. Generic scalar callers can contain rational covectors or opaque
   aliases; do not accidentally narrow existing support. Initially retain the
   old AtomField path for cases outside a documented rational-polynomial subset.
   Complex **runtime values of symbolic inputs** remain supported by a Q→Z
   symbolic preparation; this is distinct from literal Gaussian coefficients
   in direct exact fixture expressions.
5. Conversion/map/certificate failures must remain explicit internal errors.
   Only linear-system outcomes should enter the existing no-constant-solution
   homogeneous search. Do not hide a failed conversion in a chained `.ok()` or
   retry an algebraically failed rational path with different semantics.

The current private source conservatively admits only explicit-affine matrix
syntax with at most five rows/six columns and constant zero/one RHS. Thus the
normal aliased generic plan above is eligible, while arbitrary direct polynomial
or rational expressions stay AtomField. This avoids creating a new exponent
range limit for those callers without implementing a local degree analyzer.

This preserves structural singular/homogeneous behavior over the same rational
function field. It does **not** add sample-specific alternative chart selection.
Generic coefficients may have poles on specializations where a different free
solution would be regular; keep such lanes guarded or use separately certified
profiles. Canonical rational-function cancellation may remove spurious elimination
factors, so assert identity and accepted/rejected lane behavior rather than
requiring byte-identical coefficient expressions. Use matched private builds to
measure conversion, solve, descendant preparation and complete output separately.

Coordination: the concurrent N7 API port owns `pv/thin_seven.rs` and does not edit
`ibp.rs`. That helper is currently private and chart-specific, not a reusable
public API. Avoid moving it or broadening its restrictions as part of this
minimal scalar experiment.

## Smallest FeynKit adapter after the solve experiment

A private rank-zero N6/N7 scalar prepass is sufficient. At plan preparation,
construct ordered symbolic `D_i` with the five affine placeholders above and
call `AffineFamily::new(variables,denominators)?.partial_fraction(powers,budget)`.
For each returned `(coefficient, original_order_powers)`, compile its request
with unchanged `h` through **one existing `ScalarCompiler`**, then assemble the
four slots using Symbolica. Nonnegative initial powers stay nonnegative here;
check every conversion back to `u16`, overflow and processed-state limits.

This keeps common descendant/geometry/master caches and compiles one complete
Symbolica DAG evaluated instruction-major across lanes. It avoids a new
outer-loop scalar workspace per sample, child-result sensitivity transport,
or premature public generalized wrapper. Existing final-DAG guards still apply.
Only add the core's path dependency in a coordinated private patch. Do not
rewrite the extracted FeynKit engine. A later tensor adapter can use the same
integrand identity with an unchanged numerator, but child-plan composition and
its error transport are additional work; rank-zero is the smallest useful gate.

Whether this beats the existing support-cached recursive compiler is unknown.
The latter already coalesces requests in its cache; FeynKit adds an explicit
state budget and earlier canonical coefficient coalescing, at the cost of a
second preparation stage. Sparse tensors and high total powers may have very
different costs. Measure instead of assuming a gain.

## Exceptional geometry and Dual requirements

Cartesian preserved-shift identities hold for arbitrary coordinate jets because
they are derived from those same symbols. Supplied scalar products require the
existing exact full-Taylor rank-four Gram certificate and regular primal chart;
see `invariant_certificate.rs`, `certified_scalar.rs`, and
`exact_zero_certificate.rs`. A small residual or primal-zero residual is not a
certificate that higher/mixed jets stay on the physical invariant variety.

A Taylor denominator is invertible only if its **primal is nonzero**. A
coefficient with zero primal and nonzero derivative must remain active. A
homogeneous specialization requires the complete `C` jet to be zero; `C(0)=0`
alone cannot authorize replacing the inhomogeneous relation. A degenerate Gram
matrix need not make the denominator identity unusable: rank reduction may
produce a regular constant relation, homogeneous transfer, or coincident-line
power merging. None authorizes an unguarded inverse of a singular generic chart.

All-jet algebraic correctness is separate from derivative uncertainty. The thin
N6/N7 composition transports child **primal-only** sensitivity operands
(`stability.rs:875`, `thin_n6.rs`), inserting zero derivative amplitudes. The
existing local all-component cancellation checks do not certify accumulated
derivative errors from these children. Do not promote a public Dual high-point
API merely because an affine or dilation identity passes. The proposed single
core-DAG prepass avoids creating another such child boundary, but still needs
independent full-jet precision and mass-derivative qualification.

## Concrete exact tests before any numerical claim

Use only Symbolica for reconstruction, simplification, differentiation and
Taylor expansion; integer loops may enumerate original-order exponent vectors.
The current `dependencies/feynkit-integrals/tests/affine.rs::verify` already
reconstructs `Σ coefficient ΠD_i^(-power_i)`, checks independence of each positive
support, and calls `.together().is_zero()` on the complete residual.

- **N6 geometric identity:** let `p_0=0`, `p_1,…,p_4=e_0,…,e_3`,
  `p_5=(1,1,1,1)`, `m_0²,…,m_4²=M²`, `m_5²=M²+Δ`. Then
  `-3D_0+D_1+D_2+D_3+D_4-D_5=Δ`. Verify with independent `L`, including
  powers `[2,1,1,1,1,1]`, `[2,2,1,1,1,1]`, and `[1,1,1,1,1,3]`.
- **N7 extension:** `p_6=(2,1,0,1)`, `m_6²=M²+2+Θ` gives
  `-3D_0+2D_1+D_2+D_4-D_6=Θ`. Check every returned original-order vector
  has support at most five, with dots on both lines 5 and 6 and reordered input.
- **Homogeneous profile:** specialize `Δ=0` before family construction.
  Reconstruct the exact dotted output, preserve the total denominator degree,
  and cover a case where all unit inputs generate a raised descendant. Test
  coincident/proportional lines and fully zero relation constants separately.
- **Field equivalence:** compare current AtomField and proposed rational-field
  constant/homogeneous relations on the existing N6/N7 exact fixtures, including
  underdetermined N7, lower-rank consistent, inconsistent constant but regular
  homogeneous, rational coefficients, distance aliases, certified covectors,
  and complex-literal compatibility. Preserve explicit no-result errors.
- **Complete symbolic scalars:** compare all four slots from old routing,
  rational routing, and FeynKit-prepass descendants on bounded exact massive
  cases at `h=-1,0,1`; allow genuine D6 leaf symbols rather than pretending they
  are boxes. Include the dimension/power identity above and mass derivatives
  `∂_(m_j²)J_h(ν)=ν_j J_h(ν+e_j)` and mixed/higher versions.
- **Dual geometry:** full mixed jets of shifts/masses, a zero-primal coefficient
  with nonzero derivative, homogeneous `C=0` versus `C=t` at `t=0`, and supplied
  invariant jets transverse to the rank-four variety. The latter must reject;
  the first must retain its derivative contribution.
- **Budgets/failures:** state-limit exhaustion, zero state budget, power overflow,
  zero denominator, malformed vectors, nonaffine/hidden loop dependence and
  rejected floating coefficients return errors without partial decompositions.

Only after exact checks should a new bounded numerical protocol admit complete
Laurent arrays against original IREGI, increased-precision convergence, mass-Dual
identities, batched mixed successful/failing lanes and zero-Gram controls.
Keep all historical N6 rank-seven and original-PV failures visible. Performance
and support expansion remain separate gates.
