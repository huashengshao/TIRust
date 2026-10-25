# Generalized planar scalars and accepted completion envelope

This checkpoint follows `2ad3cbe` on 25 October 2026. The user now accepts a
feature-complete, independently validated implementation through **six
propagators and rank six**, including dots, as the stopping point. After that
version is tested, update the PDF, commit and push, mark the objective achieved
and stop. Performance optimization is separate later work. Seven-point/rank-seven
limitations must be explained; no extension beyond seven points/rank seven is
required. Both copies of `PLAN.md` record this clarification.

The functional objective is **still active**. Complete ordinary Float arrays
already include every rank-six prefix. Current N6 generalized tensor,
precision/Dual, supplied-invariant and frontend qualification must close before
claiming completion.

## Validated feature slice

The certified-planar scalar API now accepts the D6 unit-power pentagon and one
dot at any of its five positions in D4/D6, using its existing full-Taylor planar
certificate and denominator identity. Symbolica owns the algebra. These cases
reduce to at most four-propagator masters, without a genuine D6 pentagon leaf.

Thirty actual Rust captures at Float128/256 retain 196 successful sample lanes,
26 explicit invalid/null lanes and 880 complex Laurent/Taylor components.
Independent defining-integral quadrature and exact saved-data audits pass all
2,064 decisions. All native Laurent slots are checked before contractions.
Coverage includes all dot positions, first mass derivatives, ordinary finite
differences, scale changes, common translations, repeated lanes and unchanged
inputs. The maximum reducer/oracle error is `2.748042e-18` at the unchanged
`1e-10` gate; precision pairs pass `2^-80`.

The original orders 12/16/20 pass direct accuracy but fail 24 convergence
decisions. They remain inconclusive. A separate oracle-only refinement to
24/28/32, retaining every Rust capture, passes the original convergence gates.
Three earlier missing-optional-import attempts and both independent audits
remain in the portable capsule. No failed result is overwritten or excluded.

## Additional method coverage

Six more retained rank-seven families (and therefore all their rank-six
prefixes) pass complete Float128/256 checks: Hybrid N1–N3, IBP/Hybrid N4 and
Hybrid N5. The cohort checks 15,840 quad-reference coefficients, 7,920 precision
pairs, 7,920 IR-zero slots and 15,840 previous-PV controls.

N5 requested IBP explicitly completes using PV because a dimension recurrence
has an epsilon-dependent divisor. Its 128-bit output passes all 1,320 reference
and previous-PV checks. The strict pure-IBP gate remains failed and the 256-bit
partner remains unrun. Correct numerical completion is distinguished from pure
method coverage; genuine shifted scalars are not replaced by that completion.

## Preparation correctness and seven-point challenge

The instrumented private N7/rank7 attempt again aborts allocation under 32 GiB,
with zero coefficients. Its trace identifies expensive distance conditioning and
monitor construction, followed by failure during master activation. Public N7
remains scalar-only. Fresh positive-rank arrays and lifecycle qualification are
not claimed, and long N7 runs are held under the accepted six-point priority.

Three narrow preparation changes pass an actual Cargo build and nineteen
focused tests: Symbolica polynomial variable discovery at the distance and
exact-certificate conversions, plus an early return for master activation when
there are no master leaves. Full parameter order, precision, Taylor shape,
declaration checks and other monitors remain intact. Tests include 21,204 input
parameters, a late used symbol, reordered mixed Taylor coefficients, exact
certificates and the unchanged active-master path.

The component-amplitude transport and private high-point probes are retained in
the source. They use Symbolica's Dualizer and coupled metadata/precision/context
validation. Public high-point Dual promotion still requires complete numerical
qualification. Focused context tests do not substitute for those arrays.

## Reproduction

From the repository root, after `source scripts/env.sh`:

```sh
python3 validation/certified-planar-generalized-scalar-qualification/check.py
python3 validation/public-rank7-method-completion-n1-n5/check.py
bash scripts/build-benchmark-report.sh
python3 scripts/check-benchmark-report.py --preserve-from 2ad3cbe
```

The Python checkers replay saved bytes and exact arithmetic without running an
integral, compiler or reference binary. The planar capsule is self-contained and
retains 75 captured source files. The resource-fix and failed N7 capsules retain
their exact source and build/test receipts, excluding executables and caches.

The report has 43 pages and preserves all 356 preceding raw result values byte
for byte. No new kernel timings or performance ratios are introduced. Later
N6 generalized, full-tensor Dual, checked supplied-invariant/legacy and corrected
Python-wheel work retain their own contexts and remain separate gates.
