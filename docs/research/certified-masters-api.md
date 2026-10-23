# Certified scalar masters

`iregi_masters::certified::AccuracyMasterEvaluator<T>` wraps an existing
`MasterEvaluator<T>` and owns lazy, separate workspaces for the finite D0
near-vacuum series. The ordinary scalar-master interface remains available.

Call `evaluate_batch_with_accuracy(input, rows, accuracy, policy)` with the
ordinary evaluator's argument order and Taylor layout. Sample lanes remain
separate from Taylor components. Invalid input lanes retain errors; `lane()`
does not expose their internal placeholder values as successful results.

`MasterAccuracyContext` contains one positive exact `Rational` absolute budget
per component and a nonnegative exact relative budget. For a returned complex
component `z`, acceptance requires the maximum squared distance from `z` to
the reported rectangular enclosure to be at most
`max(absolute_budget², relative_budget² * |z|²)`. `requested_bits` records the
requested accuracy; explicit budgets determine acceptance. The `binary`
constructor supplies matching binary budgets.

`CertificatePolicy::RequireFiniteCertificate` rejects a lane if the bounded
series cannot establish that contract. `AllowOrdinaryUncertified` permits the
ordinary evaluator to handle eligible remaining lanes; diagnostics identify
those results as `Formula::OrdinaryUncertified`, with no finite enclosure.
This policy does not turn an ordinary result into an accuracy certificate.

The certified branch supports finite D0 integrals in its checked analytic
neighborhood, series orders two and four, at most 32 ancestor-closed Taylor
components, and total derivative degree at most four. It validates physical
arguments before selecting a branch, including the scale even though finite
D0 is scale independent. Orders are evaluated across active lanes, and
`series_execution` exposes the actual batch work. Unsupported families and
Taylor shapes can use the explicitly permitted ordinary path.

The enclosure includes series truncation and evaluation rounding for the exact
represented input components. It does not include rounding incurred while
constructing those inputs from routing shifts, nor any later quadrature error.
This API therefore does not by itself certify dimension-shifted pentagon
quadrature or automatically alter ordinary tensor reduction.

The current-dependency qualification is retained in
[`validation/near-vacuum-adapter-current/qualification.json`](../../validation/near-vacuum-adapter-current/qualification.json).
It covers binary64, DoubleFloat, Float128/256, derivative components, policy
contracts and an independent contour comparison. Production cleanup is recorded
in [`validation/near-vacuum-production-ready/manifest.json`](../../validation/near-vacuum-production-ready/manifest.json).
The public integration tests additionally exercise lane isolation, reversed and
empty batches, changing accuracy contracts, exact vacuum mass-derivative
identities, scale-only jets and explicit unsupported-shape fallback:

```sh
source scripts/env.sh
source .local/symbolica-license.env
cargo test --locked --offline -p iregi-masters --test certified -- --test-threads=1
```
