# Stability amplitude preparation: isolated n-way sum experiment

The ordinary public N6 rank-seven probe reaches 330 requested moments, 31,670
aliases and 56 scalar masters, then exceeds its workspace preparation bound.
The baseline and promoted-plan captures have 180 and 300 second bounds,
respectively. Neither reaches numerical reduction. Independent planar phase
profiling locates its remaining delay in primal amplitude construction; this
localizes a stage but does not establish which operation dominates N6.

`stability.rs::master_amplitude` uses Symbolica to extract a polynomial in
immediate recurrence inputs, checks affine dependence, and sums absolute-value
weighted contributions. Its repeated `result = result + contribution` performs
an increasingly large canonical sum merge. The isolated candidate collects the
identical contributions **after** polynomial coefficient combination and invokes
Symbolica's `Atom::add_many`. It leaves polynomial extraction, coefficient
absolute values, amplitude handles, lazy `IF` expressions, branch preparation
checks, and rejection of nonlinear master dependence unchanged.

Three existing-API checks precede this experiment:

1. Symbolica `dependencies/symbolica/src/atom.rs:4646` explicitly recommends
   `Atom::add_many` over repeated addition. `FromIterator` and `Sum` delegate to
   it. Its n-way merge implementation in `src/normalize.rs:1980` preserves input
   order for equal terms and returns exact zero for an empty input.
2. Symbolica's `add_normalized_slice` test at `src/normalize.rs:2504` checks
   cancellation, repeated terms, constants and nested rational sums. The current
   stability unit test checks nested conditional amplitudes and nonlinear
   errors; the experiment retains the complete prior implementation for exact
   whole-expression comparisons.
3. Existing ecosystem uses include FeynKit
   `crates/feynkit-generator/src/grouping.rs:817`, Idenso
   `crates/idenso/src/shorthands/chain.rs:453`, and Spenso tests in
   `crates/spenso-hep-lib/tests/color_trace_validation.rs`. These sources are in
   the supplied `references/gammaloop_feynkit` checkout.

No custom algebra is introduced. Production sources remain unchanged. The
private source snapshot derives from the frozen promoted-plan candidate, uses
its matching scalar-master archives, and adds phase/substage timers. It retains
`legacy`, `nway`, and `verify` runtime modes. Verify mode asserts equality of
complete canonical Atoms, including the full expression produced by nested
conditional recursion, and matching preparation errors.

The separate exact-expression probe passed 20 complete Atom/error assertions on
2026-10-23. It covers zero/constants, alias identities,
signed affine coefficients, cancellation to an empty extracted polynomial,
duplicate contribution merging, factorized and rational
coefficients, nested conditional products and structurally preserved singular branches, asserted
nonlinear errors, and wide sums of
32, 128 and 512 master variables. Output contains the complete resulting
expressions; the runner stores their SHA256s and peak resident memory. The
high-point runner records the same immutable input/source/binary checks and
hard subprocess deadlines as previous captures.

This is a candidate, not an established speed improvement. The contribution
vector retains terms until merging, so both time and peak memory matter. A
source audit cannot establish that summation dominates polynomial extraction,
recursive variable handling, or symbol-dependence checks. Timings from the
instrumented development build remain diagnostic rather than benchmark gates.
See `validation/public-highpoint/nway-build.json` and subsequent run manifests
for the exact scope and current qualification status.

The final witness is `nway3`; earlier `nway` and `nway2` build-only snapshots
are retained as unqualified iterations. `nway-qualification.json` binds all 20
asserted results to the immutable source and binary and verifies matching
baseline scalar-master and Symbolica archives. Exact checks ran on CPU14 using
the private user-provided license, with one Rayon thread. The license value is
excluded from artifacts. This license/CPU change and instrumented development
build preclude treating the diagnostic timings as a controlled comparison to
prior CPU0 captures.

The bounded public N6 run subsequently timed out during local checks before
reaching primal amplitude construction. Its completed reverse-pass span was
118.047s; the last local-check span was 126.342s. The n-way optimization therefore
has no measured high-point effect from this run. A separate planar case that
reaches primal construction is the appropriate next witness. The N6 preparation
work is being profiled by its existing reverse/union/traversal stages.
