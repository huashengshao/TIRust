# Complete-array correctness checkpoint — 25 October 2026

This checkpoint follows `0ce5bf0`. The priority is complete, independently
validated IREGI reduction through rank seven. Low-rank COLLIER speed and the
choice of interpreter or native execution are not separate completion gates.
The overall implementation goal remains open.

## Ordinary N1–N5

An actual locked Cargo build, with the adopted Symbolica routing/Cayley
preparation change, produced every cumulative tensor moment through rank seven
for three exact massive positive-F samples at each topology and both Float128
and Float256. All 30 lanes succeeded. The checks cover 39,600 complex native
Laurent coefficients against original quadruple-arithmetic IREGI IBP, 19,800
precision pairs and 13,200 scaling identities. The unchanged gates are `1e-10`
for the independent reference and `2^-80` for precision/scaling comparisons.
The largest reference discrepancy is approximately `1.84e-16`.

Both independent mathematical and terminal/batch audits pass. All applicable
19,800 IR and 996 convergent UV coefficients are exactly zero. These results
qualify the retained ordinary PV family; they do not establish every algorithm,
mass domain, precision, derivative direction or generalized interface.
See [the actual-Cargo capsule](../../validation/public-rank7-current-cargo-n1-n5/README.md).

The separate [small-N original reference corpus](../../validation/quad-original-iregi-small-n-rank7/README.md)
preserves all 26 requested captures. Original PV produces no result for three
bubble requests and disagrees in 486 finite coefficients across six stable
triangle/box arrays. These outputs are retained as failures, and never used as
accuracy anchors. The admitted comparisons use original IBP.

## Ordinary N6

The [rank-seven lifecycle and contraction capsule](../../validation/highpoint-six-point-rank7-lifecycle/README.md)
passes 21,120 tensor coefficients, 128 sparse/dense contraction coefficients,
and 2,656 precision pairs. It exercises mixed valid/invalid lanes, odd batches,
reordering, chunking, empty calls, independent cloned scratch, malformed-buffer
counter reset and recovery at both precisions. Cloned workspaces were evaluated
serially. The captured implementation precedes the Cayley change; its exact
source context is preserved separately.

Actual Cargo runs after the Cayley change also pass separately requested
rank-four, five, six and seven plans at both precisions: all 5,888 complex
reference coefficients, 5,888 prior rank-seven prefix regressions, and 2,944
precision pairs pass. The largest reference discrepancy remains approximately
`1.27e-16`. This cohort also retains all 51 passing IBP module tests and the
actual build artifacts' provenance. See
[the literal-rank capsule](../../validation/highpoint-six-point-literal-ranks4-7-current-cargo/README.md).

## Implementation and remaining work

The routing system and eligible five-point Cayley inverse now use Symbolica's
existing rational-polynomial matrix operations, with exact reconstruction and
bilateral inverse certificates. Other expression domains retain AtomField.
Actual physical singularities keep their exceptional recurrence path;
conversion or certificate errors propagate as preparation errors. No local
symbolic solver, scalar-master formula or sampled numerical specialization was
introduced.

Typed-Dual preflight failures now clear the previous call's statistics,
profiling times and diagnostic messages before returning an error. Two actual
public regression tests pass, including genuine mass derivatives, malformed
shape/storage, overflow and recovery. The
[targeted capsule](../../validation/typed-dual-state-reset/README.md) preserves
the initial test-driver failures. This change does not alter numerical formulas
or extend high-point Dual qualification.

These are correctness runs. Their stage durations are diagnostics, not kernel
benchmarks. Existing timing data and failed/noisy comparisons retain their
original contexts.

The public seven-point interface remains restricted to scalars while private
positive-rank arrays are qualified. Complete N7 rank-seven arrays and lifecycle,
high-point Dual uncertainty transport, broader generalized/complex/exceptional
cases, supplied-invariant high-point tensors, and integration of the validated
high-point engines into the CLI/Python interface remain substantive work. New
IBP-mode and typed-numeric checks proceed separately. The objective must not be
marked complete on the strength of this scoped checkpoint.
