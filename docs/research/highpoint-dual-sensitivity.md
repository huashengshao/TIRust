# Full-component sensitivity transport for high-point Dual reduction

Investigation and private prototype, 2026-10-25. The original source snapshot
below is `main` after wind-down checkpoint `2c1f334`; it describes the public
boundary that this isolated prototype addresses. The prototype has now passed
19 selected synthetic/regression tests and the narrow numerical cohort described
below. It does not change a public constructor or provide an error certificate.
Independent review of the new saved numerical outputs has passed.

## Private qualification checkpoint

Portable evidence is committed under `validation/highpoint-dual-sensitivity`:
its sealed capsule replays all source patches through the existing committed
cache parent, preserves the actual tests and raw numerical records, and checks
without the original ignored target tree or native dependencies. Run
`python3 validation/highpoint-dual-sensitivity/check.py` from the project root.
The 1.425 MB payload inherits generated master sources instead of duplicating
them. A separate `validation/highpoint-dual-rectangular-proposal` bundle retains
the explicitly uncompiled follow-up source; its checker reconstructs the
93-file proposal in memory and never runs Rust tests or a Symbolica evaluator.

The frozen five-file core extension is in
`target/highpoint-dual-sensitivity/source-v3.json` (SHA256
`a0ae4b8190091e4f25530d9f5f47b68d0463ab404e45b2b542878f169b8a23c8`).
Its actual bounded build and 19 tests passed, including immediate-alias
cancellation, complex coefficient norms, higher/mixed component ordering, lazy
singular branches, stale snapshots and signed zeros. Two independent saved
reviews checked the actual logs and input guards. The numerical capture overlay
adds observation only and uses the identical original six-component scale helper.

The new correctness protocol is
`target/highpoint-dual-sensitivity/numerical-v1/protocol.json` (SHA256
`36c9555660757cc064d7987590d3414809a5805ea65fedca0907f88c02674410`).
It completed both 128/256-bit first squared-mass jets and both 128/256-bit
six-component dilation jets, including the complete fixed batch schedules.
All 528 parent complex Laurent/Taylor slots, 156 packed child rows, full
value/precision/signed-zero/route sidecars and mutation probes passed the saved
checks. The first mass derivative agrees with independent dotted generic-IBP
results to normalized errors `6.83e-34` (128 bits) and `4.71e-74` (256 bits),
and with both original dotted interfaces within `1.18e-13`. All 264 paired
128-to-256 complex components converge within `5.28e-34`, stronger than the
unchanged `2^-80` gate. All four Laurent slots were checked.

The saved comparison is
`target/highpoint-dual-sensitivity/numerical-v1/saved-gate-001/comparison.json`
(SHA256 `bcd751217d117a2dbca29b37ca787be5ddc1e9a6b8c3459a2d5107e37f5f59fc`).
The local `qualification-summary.json` records every bounded stage and its
receipt. The independent raw saved audit is
`target/resume-20261025-review/dual-sensitivity-numerical-v1-independent-saved-review.json`
(SHA256 `dfa12c20ef5500f26affa1527683785600d63f798e81a35efe8a3dbaa4a75402`).
It independently checked 171,834 conditions, 182 full parent/child contexts,
944 mutation probes and all seven terminal process groups, without importing
the producer/comparator or running an integral. It retained all 528 paired
scalar-precision and 868 input-precision representation differences explicitly.
These are a new correctness cohort with explicitly larger predeclared
budgets; all historical 180-second cache-pilot timeouts remain unchanged.
Elapsed stage times include setup and guards and are not performance results.
This remains a private, rank-zero, unit-power N6 Euclidean fixture. Six dilation
jets cover one physical direction. Independent higher/mixed mass/shift
directions, finite differences, broader branches, public highpoint Dual
promotion and E6 backend uncertainty transport remain separate gates.

A separate source-only follow-up uses rectangular shape
`[[0,0],[1,0],[0,1],[1,1]]`: additive first squared mass `a` and independent
dilation `lambda=1+b`. Changing the integration variable gives
`F(a,b)=lambda^-8 I6(m0_squared+a/lambda^2,...)`, so the mixed target is
minus ten times the independently evaluated first-dotted scalar. Symbolica
generates both the exact inputs and, through a separate ordinary derivative
route, the unit/dot coefficient matrix. The local `rectangular-v1` proposal has
six unexecuted helper tests and an isolated capture overlay. It has not been
compiled or evaluated and carries no numerical execution authorization.

The minimal useful extension is a **positive formal Taylor series of
uncancelled contributions**, evaluated with Symbolica's existing Dualizer.
Transport its individual coefficients between the actual child and parent
routes. Do not differentiate the current primal amplitude, and do not put its
primal in component zero and fill all other components with zero. The existing
E6 quadrature propagation already implements the essential two-stage numerical
pattern needed here. A rigorous interval backend is a separate possible
improvement, not a prerequisite for maintaining the ordinary workspace's
explicitly heuristic stability contract.

## Current boundary and the concrete gap

- `crates/iregi-rust/src/stability.rs:409` constructs an amplitude through
  immediate affine recurrence aliases. `master_amplitude` at line 1183 uses
  Symbolica polynomial extraction and absolute coefficients. This preserves
  cancellation from earlier aliases instead of replacing every alias by its
  already cancelled returned value.
- `StabilityPlan::inspect_lane_with_accuracy_impl` at line 767 checks local
  additive cancellation independently in every Taylor component. Denominator
  checks inspect the primal. `inspect_output_precision_lane_impl` at line 1030
  checks tracked output precision. These remain necessary.
- `inspect_primal_lane_impl` at line 910 is intentionally primal-only. Its own
  documentation excludes compounded Taylor error. `export_primal_operands` at
  line 459 exports one amplitude per tensor/Laurent output, not per jet.
- `execution.rs:1965` extracts only primal inputs for this amplitude evaluator.
  Successful actual-route envelopes are exported around line 2064. A
  quadrature-dependent child deliberately lacks supported six-point transport.
- `types.rs:168` stores `PrimalEnvelope` and line 270 validates its association
  with the result. Only output primals are snapshotted and compared; changing
  a nonprimal component would not invalidate that association.
- `thin_n6.rs:136` imports each amplitude followed by `components - 1` zeros.
  This is correct for the ordinary one-component public interface. It is
  insufficient as full-component uncertainty information for its existing
  private `new_dual` implementation.
- `six_point.rs:14` therefore correctly defers public numerical Dual support.
  The algebraic mixed-Dual assembly tests do not close this numerical gap.

These are stability-model limitations, not evidence that Symbolica failed to
compute the signed reduction jets. The private workspace already copies every
kinematic and returned child Taylor coefficient, including primal-zero weights.

## Three reuse checks

1. **Symbolica APIs/source.**
   `dependencies/symbolica/src/evaluate/dual.rs:762` provides `Dualizer::new`;
   `ExpressionEvaluator::vectorize` is at line 229. Multiplication at lines
   995–1038 uses the supplied `DualNumberStructure` multiplication table.
   Automatic external-function differentiation at lines 843–873 explicitly
   restores the factorial-normalized Taylor convention. Thus Symbolica already
   owns higher/mixed products and multiplicities. `AtomCore::derivative` is in
   `src/atom/core.rs:694`; no derivative engine is needed. Existing
   `to_polynomial_in_vars_with_field`, `AtomField`, `Atom::add_many`, aliases,
   lazy `IF`, `map_coeff_with_prec`, and `into_batch_evaluator` provide the
   remaining operations. The ordinary `ABS` derivative at `evaluate/dual.rs:1200`
   differentiates the mathematical norm; it does **not** turn it into an error
   majorant. Its existence is a reason not to misuse it here.
2. **Existing usage/tests.**
   `stability/quadrature.rs:382` has `affine_projection`, using Symbolica's
   polynomial API, explicit degree checks and preserved branch predicates.
   `QuadratureStabilityPlan::prepare` builds signed coefficient expressions and
   positive contraction expressions. `WorkingQuadrature::evaluate` in
   `execution.rs:496` evaluates signed coefficient jets, takes the norm of each
   already evaluated complex coefficient, then runs a second Dualized positive
   contraction. Its `signed_coefficient_jets_feed_positive_symbolica_convolution`
   test at `quadrature.rs:536` exercises precisely this distinction.
   `stability/arguments.rs:30` maps exact polynomial coefficients with `abs`
   and evaluates on individual input-component norms, retaining cancellation
   within a signed monomial's Taylor product. Its first test is an additional
   reusable positive-envelope fixture. `thin_n6.rs:225` already verifies the
   actual assembly's mixed components with exact Rational evaluation.
3. **FeynKit/gammaloop/Spenso equivalents.**
   Searches of `../references/gammaloop_feynkit` for `Dualizer`,
   `ErrorPropagatingFloat`, `RealBall`, `ComplexBall`, `sensitivity`, and
   uncertainty/error propagation found generic domain/evaluator integration,
   not an affine recurrence amplitude sidecar. For example,
   `crates/gammalooprs/src/integrands/process/evaluators.rs:3102` Dualizes once
   before mapping numeric domains, and
   `crates/gammalooprs/src/utils/evaluation_domain_tests.rs:170` tests native
   higher derivatives. Spenso's numerous dual-index operations refer to tensor
   variance and must not be confused with numeric Dual jets. No replacement
   for the existing Symbolica/quadrature primitives was found in this checked
   scope. This is a scoped search result, not a claim of global absence.

No new algebraic manipulation is proposed. The new work is preparation of
existing Symbolica expressions, numerical buffer packing and sidecar lifecycle.

## Positive component amplitudes, rather than differentiated amplitudes

For one immediate affine recurrence body write

`v = b + sum_e c_e * u_e`.

Here `u_e` are immediate master-dependent aliases, native scalar slots, or
imported child tensor slots. The coefficients and `b` must have no unresolved
dependence on those uncertain values. Symbolica extracts them locally; do not
expand the entire recurrence into scalar masters.

First evaluate each **signed** `c_e` and `b` in the actual ordered Dual algebra.
For every component `alpha`, set the numerical input
`C[e, alpha] = hypot(c[e, alpha].re, c[e, alpha].im)` and similarly `B[alpha]`.
These are individual coefficient magnitudes, not derivatives of `abs(c_e)`.
Seed scalar/imported amplitudes `A_u[alpha]` densely. Then prepare the positive
formal expression

`A_v = B + sum_e C_e * A_u_e`.

Compile this expression with **the same** Symbolica Dualizer shape. Evaluating
it on the nonnegative component sequences performs the positive Taylor
convolution. For instance, a `(1,1)` component includes all four pairs
`(0,0)*(1,1)`, `(1,0)*(0,1)`, `(0,1)*(1,0)`, `(1,1)*(0,0)`.
This illustration specifies the intended result; production code must not
implement that pair sum or its multiplicities itself.

Construct one amplitude alias for each reachable immediate recurrence alias,
in the existing topological order. This matters: projecting the whole answer
onto signed scalar-master coefficients can cancel contributions from separate
recurrence paths and lose the ordinary implementation's conditioning model.
`QuadratureStabilityPlan` legitimately uses a signed global projection for a
different purpose, namely sensitivity to backend E6 errors. Reuse its local
extraction and two-evaluator mechanics here, rather than substituting that
global projection for the existing alias amplitude graph.

For an output with no master dependence, seed its per-component magnitude,
matching the current `PrimalOperandKind::NoMaster` behavior. Do not interpret a
missing monitor as zero uncertainty. This amplitude estimates sensitivity to
perturbations in native operands. Coefficient evaluation, input rounding,
backend error, underflow and branch correctness are not automatically bounded
by this sum; all existing argument, local, output-precision and backend guards
remain active. In particular, norms taken after computing a signed coefficient
jet do not diagnose cancellation *inside* that computation. Keep the existing
argument positive-polynomial checks and local coefficient checks, and document
this limitation instead of calling the new amplitude a rigorous bound.

## Proposed private API and execution prototype

The following names are illustrative API design, not code already present:

```rust
struct ComponentSensitivityPlan {
    coefficient_parameters: Vec<Atom>,
    coefficient_expressions: Vec<Atom>,
    coefficient_aliases: Vec<(Atom, Atom)>,
    positive_parameters: Vec<Atom>,
    positive_aliases: Vec<(Atom, Atom)>,
    positive_outputs: Vec<Atom>,
    output_dimensions: Vec<i32>,
}

struct ComponentSensitivityOperand<T: FloatLike> {
    output_index: usize,
    amplitudes: Vec<Complex<T>>, // exact ordered shape; nonnegative real entries
    kind: PrimalOperandKind,
}

struct ComponentSensitivityEnvelope<T: FloatLike> {
    basis: Vec<Monomial>,
    metadata: ResultMetadata, // includes exact ordered Dual shape
    working_precision: u32,
    accuracy_bits: u32,
    operands: Vec<ComponentSensitivityOperand<T>>,
    output_components: Vec<Complex<T>>, // every jet, not just primals
    routes: Vec<PrimalEnvelopeRoute>,
}
```

1. Share the local affine extraction already used in `master_amplitude` and
   `affine_projection`. Treat the immediate uncertain aliases as generators;
   require total degree at most one and reject unresolved uncertain
   coefficients. Build signed local coefficient roots, a signed constant root,
   and the positive amplitude alias graph. Keep structural budgets; no recursive
   alias substitution or fixed-size tensor arrays.
2. Build the signed coefficient evaluator using the current `compile_exact`
   pattern at `execution.rs:2395`: `Atom::evaluator_multiple`, `add_aliases`,
   `direct_translation(true)`, `horner_iterations(0)`, `cores(1)`, `build`, then
   `vectorize(&Dualizer::new(...))`. Retain only needed kinematic aliases, as
   the current quadrature preparation does. Reject a remaining scalar-master
   dependency before any numerical evaluation.
3. Build the positive amplitude evaluator in the same way, with its own
   parameter list and topologically ordered amplitude aliases. Its input jets
   are formal nonnegative sequences; they are not analytic derivatives of a
   norm. Map exact coefficients with the existing precision-preserving
   `map_exact`, never through `f64`.
4. Evaluate all signed coefficient instructions across all active sample lanes.
   Norm-map every complex Taylor component using `NativeFloat::hypot` into
   reusable scratch. Add dense native-master or imported-child amplitudes.
   Evaluate the positive graph across the same active lanes. This adds two
   instruction-major passes, rather than an outer scalar sample loop.
5. Inspect every output/component using the existing `OperandRoundoff`,
   dimensional natural scale and requested `2^(19 - accuracy_bits)` target.
   Apply operand-dependent precision to the amplitude as in the primal model;
   do not apply already tracked cancellation loss to the large upstream
   amplitude a second time. Reject nonfinite, negative, complex, truncated or
   wrong-shape amplitude data and unsupported range/underflow situations.
6. Export only after the **actual successful route** has passed its ordinary
   guards. Validate all output-component values **and their stored precisions**
   against the coupled `BatchResult`, plus basis, Laurent ordering, convention,
   tensor weights, exact Dual shape, working precision, requested accuracy and
   dense output map. Keep unmodified precision when cloning operands. This
   extends the current stale-primal check rather than weakening it.
7. At the N6 import boundary copy each complete child amplitude sequence.
   Replace the `push(primal); extend(zeros)` packing; the parent positive graph
   consumes that complete sequence. Import six children in the existing
   subset-major packed batch. No magnitude or derivative may be dropped because
   the corresponding weight or child output has zero primal.
8. Inherit the sidecar in exactly the branch that scatters a successful retry,
   analytic, certified-planar, automatic-planar or scalar-fallback result.
   Clear it on failures and on each new call. Until E6 backend error estimates
   are separately transported and budgeted, keep the current explicit rejection
   of a quadrature-dependent six-point child. Do not invent zero backend error
   to obtain a public Dual success.

The first private prototype should be opt-in, leaving ordinary one-component
execution and existing policy/cache keys unchanged. Add a separately keyed
prepared sensitivity graph and buffers only to workspaces retaining component
sidecars. The source-only design does not justify enabling a public constructor
before the validation below succeeds.

## Branch handling

Use the **original signed kinematics** to select every coefficient branch.
`affine_projection` already retains `IF(condition, yes_coefficient,
no_coefficient)` and rejects master-dependent predicates. Evaluate those
coefficient expressions before taking norms. The positive graph then consumes
the selected coefficient jets; it must not reevaluate the original predicate on
magnitude inputs. This also keeps a singular inactive denominator lazy.

Symbolica's `Join` vectorization at `evaluate/dual.rs:1310` selects with
`c.index(0)`, as intended for a fixed analytic branch. Preserve the present
master checks: `iregi-masters/src/lib.rs:816` rejects transverse seeds where a
scalar boundary formula lacks a supported analytic neighborhood; existing
snapping residuals report active comparison boundaries. Do not ban every exact
zero uniformly: several implemented analytic limit charts have valid jets.
Conversely, matching primals or a small envelope does not authorize a derivative
across an unsupported branch. A master-dependent predicate should remain an
explicit unsupported sidecar route in the initial prototype.

## Small failure models the current import cannot represent

1. **Compounded derivative cancellation.** Consider a child
   `B = R0 - R1` with `B[0] = 1`, large nearly cancelling first derivatives in
   `R0` and `R1`, and a small returned `B[1]`. Let another child `C` have
   `C[0] = 1` and `C[1] = 0`. The parent is `F = P * (B - C)` with large
   master-independent `P`. The child's derivative estimate can pass an
   absolute dimensional tolerance, while `P` amplifies that estimate beyond
   the parent's tolerance after its primal cancels. The parent sees small
   returned derivative terms and primal amplitudes of order `P`; only the
   imported derivative amplitude retains the earlier large `R0[1]` and
   `R1[1]`. For a schematic 128-bit unit `2^-127`, child derivative amplitudes
   near `2^80` and `P = 2^30` separate these scales by many bits, even allowing
   ordinary local operation factors. This is an algebraic failure model, not
   a claimed measured physical fixture.
2. **Zero-primal weight.** For `F = t * B(t)`, the second Taylor coefficient
   depends on `B[1]`, although the weight's primal is zero. Importing only
   `A_B[0]` cannot represent the uncertainty in this coefficient. The existing
   signed assembly already keeps the term; its uncertainty transport must do
   the same.
3. **Mixed components.** For `F(x,y) = w(x,y) * B(x,y)`, the `(1,1)` estimate
   needs `A_B[1,1]`, `A_B[1,0]` and `A_B[0,1]`, as well as the primal. A
   six-component shape with zeros filled after its primal has the right length
   but the wrong meaning.
4. **Stale sidecar.** Mutate only a returned mixed component while preserving
   the primal. Current primal snapshots cannot notice. Full-component snapshot
   validation must reject this before child data is imported into a parent.

## Why existing error/interval domains are not a shorter replacement

`numerica/src/domains/float/error.rs:13` supplies
`ErrorPropagatingFloat<T>`, but stores `abs_err: f64`. `new` and precision
accessors at lines 294–332 use `to_f64`. It models linearized errors, not a
precision-preserving arbitrary-exponent component amplitude. Replacing every
numeric type by this wrapper would require a separate generic-master audit and
would introduce the intermediate binary64 error conversions forbidden by the
current goal. Its evaluator example in `symbolica/tests/evaluation.rs:367`
demonstrates successful evaluation, not this transport contract. Also note that
the inspected `EvaluationDomain` implementation for
`Complex<ErrorPropagatingFloat<Float>>` declares `Some(53)` at
`symbolica/src/evaluate/domain.rs:336`; it must not silently choose constant
precision for an arbitrary-precision extension.

`RealBall` and `ComplexBall` already provide directed-rounding rational algebra
in `numerica/src/domains/float/interval.rs:13`. Their evaluator integration at
`symbolica/src/evaluate/domain.rs:42` intentionally refuses to invent enclosures
from rounded complex constants. Inherited transcendental functions are not all
certifying. Branch decisions on balls need the separate source whitelist and
guard work described in `docs/research/dag-roundoff-guard.md`. They can support
an optional, conditionally certified recurrence experiment, but do not give
certified scalar-master transcendental errors or a valid branch merely by
wrapping a midpoint. Requiring such a whole-integral certificate before any
highpoint Dual interface would exceed the current ordinary heuristic standard.

## Required private validation before public exposure

- Exact Rational graph tests: ordinary and higher/mixed shapes; pure second
  derivatives; complex coefficient jets; primal-zero weights; successive alias
  cancellation; reversed/noncanonical component ordering; static and dynamic
  structures. Compare generated positive jets with an **independently prepared
  Symbolica polynomial** whose coefficients have known sentinel values.
- Branch tests: nested/shared lazy branches, inactive singular denominators,
  branch changes in different sample lanes, rejected unsupported transverse
  boundaries and explicit failure of master-dependent sidecars. Never relax
  the existing boundary checks to pass the fixture.
- Sidecar tests: stale nonprimal values, altered precision, wrong shape of the
  same length, sparse/duplicate maps, negative/nonfinite operands, differing
  accuracy target, all retry routes, previous-success then failure, empty and
  mixed-validity batches. Confirm disabled ordinary calls retain no buffers.
- Numerical N6 rank-zero then ranks one through three at increasing working
  precision with a retained requested target. Use **independent mass and shift
  directions**, not only common dilation; compare every tensor/Laurent jet and
  contraction, including mixed and pure second derivatives. Include inputs
  distinguishable beyond 53 bits and static/dynamic equivalence.
- For `D_i = (l+p_i)^2 - m_i^2`, the mass derivative is
  `d I(nu)/d m_i^2 = nu_i * I(nu + e_i)`. A unit-power pure second-order
  **Taylor coefficient** is `I(nu + 2 e_i)`, and a two-mass `(1,1)` coefficient
  is `I(nu + e_i + e_j)`. Compare against stable original generalized-interface
  reference runs and independent high-precision ordinary finite differences.
  The dotted N6 oracle need not be a newly exposed TIRust generalized path.
- Use central finite differences and mixed corner differences with multiple
  exactly represented step sizes, and precision refinement sufficient to
  resolve the subtraction. Divide raw pure second derivatives by `2!` before
  comparing Taylor coefficients. Retain failed/noisy differences explicitly;
  a single agreeable step size does not validate a derivative.
- Common dilation remains a useful control: with shifts and mu scaled by
  lambda and squared masses by lambda squared, an N6 rank-r moment scales as
  `lambda^(r-8)` coefficient by coefficient. For rank zero, additive
  `lambda = 1+a*x+b*y` gives mixed coefficient `72*a*b`, while product
  `lambda = (1+a*x)*(1+b*y)` gives `64*a*b`. It is a correlated direction and
  cannot replace independent mass/shift checks or demonstrate general jets.
- Preserve batch statistics: one packed N5 child batch and one parent batch,
  derivative components distinct from sample lanes, odd sizes/chunking/
  reordering, repeated calls and independent concurrent workspaces. Measure
  preparation and new monitor passes separately before optimization claims.

Completion of this prototype would provide an honest extension of the current
heuristic model to all requested Taylor components. It would not settle rank
seven reference discrepancies, generalized highpoint reduction, broad scalar
branch coverage, or performance parity.
