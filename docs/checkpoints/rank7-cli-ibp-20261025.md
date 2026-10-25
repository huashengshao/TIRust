# Shared CLI and retained rank-seven IBP checkpoint

This checkpoint follows `a69b754` on 25 October 2026. The complete implementation
objective remains active. It publishes the shared CLI frontend and additional
independent evidence; it does not expand public seven-point support beyond rank
zero or claim a new performance result.

## Implemented and checked

The CLI now constructs `TensorReductionPlan` and `TensorWorkspace`, delegates
numerator contraction to the shared library helper, and exposes the selected
engine and nested execution counters. Graph import and Symbolica numerator
preparation remain separate. Hybrid defaults, precision rescue, numeric parsing
and master policy are preserved. Missing rescue observations are JSON null.

All thirteen exact CLI tests pass. Four low-point smoke calls cover the retained
bubble reference arrays, DOT numerators and contraction, DoubleFloat and precision
rescue. The six-point rank-three call checks all 140 complex Laurent coefficients;
the seven-point scalar call checks all four coefficients and 21 packed children.
Both include explicit invalid-neighbor behavior. A first N7 timeout and a
test-only diagnostic assertion failure remain saved beside their separately
bounded successful continuations. These durations are correctness diagnostics.

The N2 rank-seven IBP rerun after Symbolica's scoped-condition reuse passes all
7,920 reference comparisons, 3,960 precision pairs, 2,640 scaling checks and
3,960 IR-zero checks at Float128/256. Its largest reference discrepancy is
`1.8383864179821657e-16`, with the unchanged `1e-10` gate. Precision/scaling gates
are `2^-80`. Earlier N2 timeouts remain failed historical records.

Twelve complete original-IREGI quad arrays provide ten new generalized N2/N3
anchors plus two ordinary controls: first/last dots, first power three and unit
powers in D2/D6. The existing narrow factorial-pair repair is confined to the
generated reference copy. Both controls match the earlier ordinary array exactly.
These anchors support subsequent Rust validation; they do not establish the
correctness of every generalized finite coefficient.

## Explicit remaining failure

A fresh actual-main private seven-point rank-seven Float128 attempt fails
allocation under a 32 GiB address-space limit while preparing the workspace.
It emits zero tensor coefficients; Float256 and the pair gate are unrun. The
coarse failed log cannot uniquely identify the expensive substage. Public N7
continues to reject positive rank. The failure, successful cleanup, source
snapshot and denied read-only debugger sample are retained independently of
subsequent instrumentation and optimization work.

## Reproduction and boundaries

Run from the repository root after `source scripts/env.sh`:

```sh
python3 validation/public-rank7-ibp-n2-postfix/check.py
python3 validation/cli-shared-facade/check.py
python3 validation/quad-original-iregi-generalized-rank7/check.py
bash scripts/build-benchmark-report.sh
python3 scripts/check-benchmark-report.py --preserve-from a69b754
```

These Python checkers replay saved evidence without running a reducer or
compiler. The CLI capsule inherits its exact 77-source snapshot from the N2
capsule. Later full-Taylor transport, fine preparation timers, planar generalized
scalar work and the Numerica Python precision fix have their own worktree
qualification and are not implicitly admitted by these older artifacts.

The updated report has 41 pages and retains all 347 preceding raw result values
byte for byte. No new kernel timings or performance ratios are introduced.
The next work remains complete N7 arrays/lifecycle, broader algorithms and
kinematics, generalized tensors, precision/Dual qualification and fresh frontend
regressions after the dependency correction.
