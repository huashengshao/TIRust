# Local OneLOop integration

Origin: https://github.com/alphal00p/oneloopmaster

Baseline revision: `8ec256fc26cb361b20e10f59c19beb35717ef32c`.

This directory contains the baseline package manifest, README, Rust sources and
prebuilt expression assets. It retains the upstream manifest's licensing status.
The parent workspace supplies the exact Symbolica/Numerica/Graphica patch pins.

## Analytic grouping in Dual compilation

`src/definitions.rs` and `src/expressions.rs` replace the exact identity
`if(z,z,0)` with a transparent Symbolica identity function only when preparing a
Dual evaluator. The original identity is an intentional arithmetic grouping
barrier. Its scalar zero arm, however, erases every nonconstant Taylor component
when the primal of `z` is zero. Independent equal-mass derivatives need those
components.

The mapped identity stays opaque during Atom normalization. The existing direct
translation path evaluates its argument once into the caller's instruction
scope, then inlines the identity body by returning that slot. This keeps the
rounding/grouping needed by complex-axis tests, preserves every component, and
introduces neither a runtime callback nor an extra conditional branch. A guard
with identical nonzero/zero branches was avoided because Symbolica deliberately
uses separate expression caches for the two branch scopes.

All transformations use Symbolica's `replace_map_bottom_up`; only structurally
identical condition/true arguments with a literal zero false argument match.
The ordinary scalar evaluator, generated Rust scalar kernels, and non-identity
conditionals remain unchanged. Regression and physical finite-difference tests
are in `../../crates/iregi-masters/tests/physical_derivatives.rs`.

## Bounded helper inlining before Dual compilation

The same Dual-only map rebuild asks Symbolica to inline leaf helper bodies of at
most 256 encoded bytes, and pure forwarding aliases whose arguments are only
variables or numbers. Symbolica's existing symbol inspection and function
registration options perform this transformation; no algebra is implemented
locally. Helpers containing other registered helper calls stay outlined unless
they are pure aliases, bounding expansion of nested expression graphs.

This targets redundant helper-component evaluation after Dualizer retains all
arguments of an outlined helper. Scalar interpretation and generated scalar Rust
remain unchanged. Qualification compares the same physical high-precision finite
differences and legacy Dual tests with the mapped-identity baseline; measured
results are recorded in the master's validation evidence once available.

## Euler–Mascheroni constant in reusable Rust emission

The exact constant metadata and native constant table recognize Symbolica's
`euler_gamma()` constant, which appears in IREGI's normalization. Generated code
initializes it with Symbolica's `Real::euler` at the requested numeric precision;
there is no binary64 intermediary or local approximation. This extends the
existing constant export used for pi and dilogarithms, allowing the same Rust
batch emitter to accept normalized reduction graphs.

The two compile-time unit-test fixtures in `tests/data/` are unchanged copies
from the baseline revision above. They are retained so the dependency's own
unit-test target can compile when qualifying emitter/runtime modifications.

## Exact represented-value access for certified zero predicates

`NativeFloat::exact_rational` exposes exact represented input values without a
binary64 detour. Binary64 uses Symbolica's 53-bit float conversion, DoubleFloat
adds its two exact limb rationals (including sparse limbs), and arbitrary Float
uses its existing exact conversion. Nonfinite inputs return `None`. A default
`None` preserves custom numeric implementations while allowing callers to report
that a required certificate is unavailable. IREGI uses this before treating a
rounded scalar invariant as exactly zero; no scalar-master arithmetic changes.

## Opt-in ordinary-root stack-allocation experiment

`OneLoopExpressions::evaluator_with_stack_reuse` builds the same exact ordinary
graph as `evaluator`, then calls Symbolica's existing `optimize_stack` once.
Default compilation, Dual compilation, prebuilt assets and generated production
kernels remain unchanged. The allocator only reassigns temporary storage; it
does not simplify expressions or reorder arithmetic. This is an experimental
entry point whose correctness and performance evidence must be recorded before
any default adoption.

Reuse checks: Symbolica's `src/evaluate/tree.rs` returns fresh root SSA directly
from `linearize_multiple` for direct translation with zero Horner iterations;
`src/evaluate/optimize.rs` provides the existing lifetime allocator. Its branch
test `nested_multi_result_branches_keep_common_operations_in_scope`, and the
shared-function tests in `src/evaluate.rs`, exercise allocation with lazy
branches. FeynKit's `soft-ct-evaluator-memory` reproduction programs likewise
apply it to freshly compiled zero-Horner graphs. No alternative algebra or
allocator is implemented here. Already allocated graphs must not be allocated
again: Dualizer and shared-body compilation already perform this step.

`examples/stack_reuse_probe.rs` qualifies the opt-in in an explicit artifact
directory. It records original exact represented inputs, Laurent output order,
scale/normalization conventions, root and deduplicated helper slot counts,
ordered instruction provenance, and numerical batch parity. Its C0/D0 inputs
are copied from the recorded `validation/native-policy` experiment. It checks
empty, singleton, odd, batch-32, reordered and reused batches, plus DoubleFloat
and Float256. `--emit` exports paired baseline/candidate Rust batch modules and
one exact constant table to that artifact directory only. These modules retain
the Whole batch policy. The probe reports no timing measurements; compiled
native parity and controlled timing remain separate qualification steps.

The transparent C0/D0 root probe passed 40 batch/domain configurations per
family, comparing 672 complex outputs each with identical represented values,
precision metadata and signed zeros. Root temporary counts changed from 132 to
7 for C0 and from 24,777 to 24 for D0; shared helpers were unchanged. Exact
instruction provenance also agreed. The evidence in
`../../validation/stack-reuse/qualification.json` explicitly separates these
three Laurent outputs from the complete IREGI scalar adapter and records the
initial three-call wrapper probe, which showed no root allocation reduction.
These are correctness and storage results, not runtime measurements.

The complete six-output IREGI scalar factory also passed the paired C0/D0
probe: 40 configurations and 1,440 complex outputs per family, including
massless and scaleless controls. The actual factory root uses 220 to 8
slots for C0 and 25,010 to 25 for D0; shared helpers are unchanged. This
candidate changed only the ordinary factory call in an isolated source
snapshot. After independent review and paired core/master/DD-E6 checks, the
IREGI ordinary factory adopted this call; current-source workspace validation
is tracked separately in `adoption.json`. Exact source, binary and dependency
hashes, inputs, instruction provenance and parity records are in
`../../validation/stack-reuse/production-factory/qualification.json`.
Paired core execution (30 tests per variant), focused scalar integration
(three per variant), and the three-lane DD E6 fixture passed. All 12 complex
E6 Laurent values, precision metadata and lane statuses matched exactly.
The observed peak RSS was 2,052,724 KiB before allocation and 558,752 KiB after;
these single runs used per-child resource accounting, with other builds
active. They are diagnostic observations, not a controlled timing gate.
The frozen pair predates the latest low-relative-precision guard correction,
so current-source Float/stability checks remain required. Generated native
kernels and Dual preparation are unchanged; native regeneration and parity
remain separate qualification steps.


## Preserving correlated complex-box line sums

`src/box_complex.rs` now reuses the existing `coefficient_sum` helper for the
shared `g+h` coefficient of the first two lines in `ordered`. Symbolica proves
that the two original expressions equal
`(p5-p2+p4-p0) + (p3-p5+p0)*alpha` using the already imposed alpha quadratic.
The invariant form retains exact zeros which otherwise arise from cancelling
irrational expressions. It prevents a spurious far-root branch when a common
complex mass shift leaves this coefficient identically zero. The existing
severe-cancellation threshold, root selection, sheets and continuation remain
unchanged; no new algebra implementation or precision adjustment is introduced.

The first qualification point was copied from an actual failed Float256 E6
contour row, preserving all 22 real components, tracked precisions and signs.
The prior result retained only eight and six bits in its real and imaginary
parts, respectively; the repaired ordinary Symbolica result retains 254 and
251 bits. Independent fixed order-20 and order-32 positive-F parameter rules
agree with all three raw OneLOop coefficients at `1e-10`, and with each other
at `1e-12`. The maximum combined normalized difference is `3.553e-15`.
The native IREGI phase is divided out symbolically once. Its negative common
width and positive real F also require a positive raw imaginary part; the
legacy result violates that sign while the repaired result satisfies it.
The evidence is recorded in
`../../target/e6-exact-d0-parameter-anchor/capture-001/receipt.json`
(`c62b112f917322a3bcd144672dca24b9e8614605d2b11e32fd948a4eef98d041`).
Ten exact-identity, policy and existing scalar/Dual regression tests also pass.

The unchanged Float256 E6 scalar contour then converges at order 64 with
600 D0 evaluations and an observed successive-rule estimate of `7.391e-27`.
All 13,200 represented real input components of the 600 common old/new D0
rows are identical. Only the 120 formerly unstable pinch-1 finite values
change; the other 480 finite records and all pole records remain identical.
This is an actual kernel convergence result, separately from the independent
one-point value check. It does not certify a quadrature remainder or complete
parent tensor arrays. The scalar receipt is
`../../target/ordinary-gh-parent-requalification/e6-proposal/capture-001/receipt.json`
(`234ec3a42f787d5aab7a355f583ab3b351eb40b7734c4edec3db48319c1787ce`).

`src/expressions.rs` explicitly restores the previous `ordered` helper body
before all existing Dual preparation. Thus this repair changes the ordinary
Symbolica evaluator only. A primal zero of the shared coefficient is not a
certificate that every transverse Taylor component vanishes, so the repair
must not silently change those Dual germs. Existing seeded-Dual behavior is
preserved rather than newly certified. Handwritten native and generated
kernels are unchanged and are not covered by this ordinary Symbolica result.
The fixed rules provide numerical evidence, not a rigorous quadrature error
bound or a claim of 256 accurate output bits.

## Hand-written binary64 C0 for negative Källén functions (30 October 2026)

`src/native/handwritten/triangle.rs` adds `hooft_veltman`, a direct port of
`triangle_hv::expression` (the 't Hooft--Veltman form the exact expressions
select for three massive propagators when the Källén function is negative):
the same edge sorting, `i sqrt(-lambda)` on the upper lip, root lips `-1/+1`,
`Sheet::with_sign`/`div`/`dilog` continuations and integer eta corrections
with the `Q::upper((y-1)/y)` logarithm. `three_masses` now returns it instead
of deferring to the generated code. Sectors with a zero mass, nonnegative
Källén functions and all non-binary64 kernels are unchanged.

Validation (`iregi-rust/crates/iregi-rust/examples/c0_hv_check.rs`): 80,000
random points (Euclidean and below-threshold timelike invariants, real and
complex masses) agree with the generated code to at most 2.0e-12 relative;
against 128-bit generated evaluations on 8,000 points both paths stay within
3e-12 (hand-written max 2.8e-12, generated 1.2e-12 in the worst class, each
worse on some points). Hand-written 1.71 µs versus generated 3.23 µs per C0.

## Native arbitrary-precision complex dilogarithm (30 October 2026)

`src/native/dilog.rs` replaces Symbolica's generic `polylog(2, z)` evaluation
behind `NativeFloat::dilog` for MPFR `Float` (at the argument's precision) and
for `DoubleFloat` (still through a 160-bit `Float`, both limbs retained). It
ports the algorithm and branch conventions of Symbolica's binary64 kernel
`dilog_complex_f64`, which the binary64 path keeps using: the maps to
`|z| <= 1, Re z <= 1/2` (via `1-z` and `1/z`), `log1p` near the origin and one,
an exactly real `x > 1` with imaginary part `-pi ln x`, a negative zero at
`x = -1`, and the Bernoulli series in `u = -ln(1-z)`, here summed to the
working precision with exact coefficients `B_2k/(2k+1)!` from the tangent
numbers (Brent and Harvey), rounded once per precision and thread.

Validation (`iregi-rust/crates/iregi-rust/examples/direct_tier_profile.rs`,
`DILOG_COST=1`): 636 points covering every argument map, the real axis, the
branch point and the neighbourhoods of 0 and 1 agree with Symbolica's
`polylog(2, z)` at the same precision to relative 1.5e-31 (106 bits),
1.0e-47 (160), 5.8e-77 (256) and 1.5e-153 (512), with identical signs of zero
imaginary parts. Unit tests check the tangent numbers, the first coefficients
and the binary64 instance against `dilog_complex_f64`. Cost per call:
15/21/34/73 µs at 106/160/256/512 bits versus 198/292/468/1058 µs for the
generic evaluation; a four-point box with masses at 256 bits drops from 40 ms
to 7 ms of master evaluation.

## Native double-double dilogarithm (31 October 2026)

`DoubleFloat::dilog` now runs the same kernel (`src/native/dilog.rs`) in
double-double arithmetic with the coefficients rounded once from 160 bits
(`double_coefficients`), instead of converting both limbs to a 160-bit MPFR
value and back. The previous route stays available as
`double_dilog_via_float` for validation. On the 636-point grid
(`examples/direct_tier_profile.rs`, `DILOG_COST=1`) the two agree to relative
6.3e-32; at exactly real `x = -1` the native kernel returns the negative-zero
imaginary part of the binary64 convention (the MPFR route lost the sign in the
conversion back). Cost 1.6 us instead of 20.9 us per call; a 106-bit physical
gg->Hgg pentagon (direct engine, rank 5) drops from 8.2 to 4.8 ms.

## Small-momentum B0 (31 October 2026)

For `0 < |p²| <= max(|m0²|, |m1²|)/16` the generated B0 expression cancels
like `max(m²)/p²`: a numerically lightlike gluon leg (binary64 `p² ~ 1e-10`)
next to a top mass (`m² = 3e4`) kept two digits in binary64 (relative 1.8e-2
on the grid below), and every triangle and box above that bubble inherited
the error. `NativeEvaluator` now evaluates B0 there at `p² = 0` exactly (the
generated zero-momentum branch) and adds the finite-part shift
`B0(p²) − B0(0) = −∫₀¹ ln(1 − x(1−x)p²/D(x)) dx` from
`src/native/small_momentum.rs`: from the zeros of `L(x) = D(x) − x(1−x)p²`,
for comparable masses as a series of power-sum differences generated with an
explicit factor `p²` in every term, for hierarchical masses in closed form
through `log1p` of `δ/(x0−1)` and `δ/x0` with the zero shift `δ` taken from a
quadratic for `δ` itself. Poles do not depend on `p²`. Outside that regime, or
where neither form converges fast, the general expression is kept. The public
`b0_momentum_shift` lets batch kernels apply the same repair
(`iregi-masters` does so for every ordinary B0 lane); `set_small_momentum`
(hidden) restores the general expression for validation.

Validation (`../../crates/iregi-masters/tests/small_momentum_b0.rs`): 252
points (equal, comparable, hierarchical, one massless and complex masses; both
signs of `p²` from 1e-14 to 1/8 of the larger squared mass) against the general
expression at 1024 bits: worst relative 6.6e-16 in binary64 and 3.1e-33 in
double-double, where the general expression in binary64 is off by 1.8e-2.

## Native dilogarithm: lift the argument to the working precision (31 October 2026)

`Float::dilog` now sets both parts of its argument to the working precision
(`max` of their precisions) before the series. Binary operations keep the
precision of their left operand, so a part whose tracked precision had dropped
(a cancelled, nearly zero real part with 2 bits next to a 255-bit imaginary
part, found by upstream's `native_256_bits_matches_oneloop` C0 rows) degraded
the whole result to a few bits. Symbolica's own `polylog` converts its
argument the same way. iregi-rust's direct-engine and regulator tests pass.

## Non-finite C0/D0: retry with tiny invariants at zero (31 October 2026)

Both binary64 D0 kernels (hand-written and generated) returned NaN for a
top-loop box with two opposite numerically lightlike legs: `p1² = 1.7e-10`,
`p3² = -2.6e-10` (binary64 rounding of massless momenta given as differences of
shifts of order 1e3), `p2² = 15625`, `p4² = 1e6`, `s = 4.8e5`, `t = 5.2e5`,
`m² = 29929`; double-double and 256 bits are finite. `evaluate_row` now
retries a C0 or D0 point whose result is not finite with every invariant
within `2^-30` of the largest argument set to exactly zero, and keeps the
retry when it is finite. The change of the inputs is below `2^-30` of the
kinematic scale; on this point the retried binary64 value agrees with 256 bits
to 1e-15. Points with finite results are unchanged.

Validation: `../../crates/iregi-masters/tests/nonfinite_d0_repair.rs` (both
binary64 kernels against 256 bits, relative <= 1e-12). In iregi-rust this
removed the last stability-study lanes (heptagon-r7 #10) whose binary64 arrays
were not finite.

## Fused double-double kernels without fused multiply-add (2 November 2026)

`DoubleFloat` products used xprec's algorithm 12, three `f64::mul_add` calls
each; without the FMA target feature (baseline x86-64, how iregi-rust is
built) every one runs as two out-of-line calls (`compiler_builtins`'
runtime-dispatched `fma`), and each product and sum is renormalized on its own.
New `native::double_double` (public `oneloop::double_double`): `acc + x1 y1 ±
x2 y2` as one exact sum (Dekker products, their errors and the low-order terms
in one binary64 sum, one final two-sum; error a few u² times the sum of the
term magnitudes). It is symmetric in the factors and odd in each, so
`DoubleFloat::complex_mul` now uses it in place of `commutative_product` and
keeps `a b == b a`, `a (-b) == -(a b)` and an exactly zero imaginary part of
`z conj(z)`. The dilog series multiplies with the domain's `complex_mul`
(binary64 and `Float`: the plain expression, unchanged), and
`DoubleFloat::hypot` returns an axis value directly (the value its general
formula gives). iregi-rust also uses these kernels in its double-double
recurrences.

Validation: `native::double_double::tests` (2000 operand sets against 300-bit
MPFR: real products within 4 u², complex products and multiply-accumulates
within 8 u² of the term magnitudes; commutativity, oddness, zero imaginary part
of `z conj(z)`); iregi-masters' double-double master tests and iregi-rust's
direct-engine and rescue tests pass. Opening N7R7 batch of 32 in iregi-rust:
106-bit masters 95 -> 71 ms, rescued arrays within 1.3e-26 of 256 bits.
Upstream: alphal00p/oneloopmaster#16, merged (b667e78).
