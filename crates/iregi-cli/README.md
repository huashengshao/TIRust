# Standalone CLI

Run from the repository root after `source scripts/env.sh`:

```sh
cargo run -p iregi-cli -- prepare examples/graphs/bubble.dot --model fixtures/models/phi3.json
cargo run -p iregi-cli -- reduce --input examples/inputs/bubble.json --rank 2 --algorithm ibp
cargo run -p iregi-cli -- run examples/graphs/bubble.dot --model fixtures/models/phi3.json --input examples/inputs/bubble.json
cargo run -p iregi-cli -- run examples/graphs/triangle.dot --model fixtures/models/phi3.json --input examples/inputs/triangle.json --precision dd
cargo run -p iregi-cli -- reduce --input examples/inputs/bubble.json --precision float --bits 192
cargo run -p iregi-cli -- reduce --input examples/inputs/bubble.json --precision float --bits 128 --accuracy-bits 53
cargo run -p iregi-cli -- reduce --input examples/inputs/bubble-scalar-products.json --algorithm pv
cargo run -p iregi-cli -- compare --fixtures validation/reference/fixtures.json
cargo run -p iregi-cli -- compare --case C0_power2_1_1_rank2 --algorithm ibp --rescue
cargo run -p iregi-cli -- compare --fixtures validation/reference/snapping/reductions.json --case bubble_near_mass_mode1
```

`prepare` uses FeynKit, Spenso, Idenso and Symbolica to return exact numerator
coefficients, their four loop-component exponents, and the propagator routing.
Pass `--feynman-rules` for physical graphs, `--external-wavefunctions` to include
external states, and `--component 0,0` when selecting residual tensor axes.

`reduce` and `run` accept a single sample or `{"samples":[...]}`. Each sample
contains `propagators`, `mu`, and optionally `id`, `rank`, `dimension_shift`,
`scalar_products`, `snap`, `threshold`, and `parameters`. A propagator contains four `shift` components and `mass_squared`
(either a real string or a pair of real/imaginary strings), with optional
positive integer `power` and graph `edge_id`. All kinematic numbers must be
quoted; bare JSON numbers are reserved for the captured binary64 reference
fixtures used by `compare`. Decimal strings go directly through Symbolica at
the requested precision, and strings such as `"1/3"` use exact rational parsing.
Decimal negative zero retains its sign.

`--accuracy-bits N` sets the core's accuracy parameter independently of working
precision. Its nominal tolerance is `2^(19-N)`; for example, `N=53` gives about
`5.82e-11`. It must be positive and no greater than the output precision. Omitting
the option retains the existing default, which uses the output precision.
This target governs numerical stability checks, not a certified error bound.
The separate `compare --tolerance` controls agreement with reference fixtures.

`--rescue` independently verifies every successful or unstable tensor lane at
two higher precisions, then rounds converged results back to the output type.
For binary64 and DoubleFloat the defaults are 128 and 256 bits. For arbitrary
precision, the first rescue precision is the next power of two above the output
precision, at least 128 bits; verification uses twice that precision.
`--rescue-bits` and `--verification-bits` override these choices and require
`output bits < rescue bits < verification bits`. Invalid input and unsupported
master domains remain failures.

Rescue promotes the already parsed input exactly; it cannot recover decimal
digits rounded during binary64 parsing. Use `--precision float` to parse the
original decimal strings at higher precision. Verification covers tensor
moments. Numerator coefficients and contraction run at output precision and are
not independently verified by tensor rescue.

The input `dimension_shift` is the full even offset in
`D=4+dimension_shift-2*epsilon`; the CLI translates this to the core's half-shift
convention. A batch shares topology, powers, dimension shift, requested rank,
and snapping policy/threshold
while every sample may have different shifts, masses, scale and parameters.
The metric is `(+---)`, shifts are `(time,x,y,z)`, and `mu` is the scale, not its
square. The default `snap: false` preserves derived scalar products.
`snap: true` explicitly selects the original on-shell snapping policy, with
default exact threshold `1/1000000`. `threshold` accepts a nonnegative decimal,
scientific or rational string; a JSON numeric threshold token is also retained
exactly without a binary64 conversion. The threshold is used only when snapping
is enabled. Zero disables relative matching for positive reference masses but
retains the original massless absolute `1e-3` rule.

Legacy snapping acts at internal real scalar-product calls, including PV cross
products. Supplied shifts, masses and widths remain unchanged. Coordinate input
currently supports this policy through four propagators; five or more returns
an explicit unsupported error. A zero threshold with a negative reference mass
can make the original policy map a zero scalar product to a nonzero value;
affected lanes are rejected because that Cayley-diagonal domain is unsupported.
Some pure-IBP kinematics also return explicit instability diagnostics; enabling
snapping does not imply that every original-IREGI configuration is qualified.
Independent legacy-policy comparisons currently cover ordinary rank-two
bubbles, triangles and boxes. Generalized powers and dimension shifts receive
the same policy, but do not yet have separate numerical qualification.
Optional `scalar_products` supplies the real invariants `p_i.p_j` directly as
quoted strings in upper-triangle order including the diagonal: for three
propagators `[p0.p0,p0.p1,p0.p2,p1.p1,p1.p2,p2.p2]`. A batch uses the same
invariant input convention throughout. Explicit shifts remain necessary for
Cartesian numerator moments. Omitting this field computes invariants from shifts.
Supplied scalar products remain authoritative and bypass legacy snapping,
including its coordinate-input restriction. This bypass is recorded in the
preparation notes. The core separately rejects supplied scalar products with
six or more propagators until their four-dimensional Gram constraints can be
validated across every Taylor component.

`run` requires explicit canonical `(q+p_i)` propagators in the order printed by
`prepare`. It validates supplied `edge_id` values against that order. For a
reported original propagator `s*q+r` with `s=-1`, supply canonical shift `s*r`.
Graph-derived external kinematics are not yet inferred. `parameters` maps the
remaining coefficient symbols or function expressions, including namespaces,
to a real numeric string or `[real,imaginary]` strings. All samples must bind
the same parameter keys. Symbolica evaluates and contracts the numerator across
the batch; there is no separate polynomial engine in the CLI.

The rank defaults to the prepared numerator degree for `run`, otherwise the
input rank or zero. `--rank` overrides it and cannot be below the numerator
degree. Without an override, explicitly declared sample ranks must match the
inferred rank. `--convention native` retains the original IREGI measure and separate
UV/IR poles. `oneloop` and `madloop` select the corresponding core conventions;
`run` also changes odd-degree numerator coefficients into the MadLoop loop
variable convention before contracting MadLoop moments.
`reduce` and `run` select their method with `--algorithm`; the `mode` field is
used only when importing reference fixtures through `compare`.

The CLI constructs the core's shared `TensorReductionPlan` and `TensorWorkspace`.
The core selects eligible public six-/seven-point ordinary PV wrappers using
its existing domain guards; the CLI maintains no separate eligibility list.
`--algorithm` still defaults to **Hybrid**. This integration does not change
Hybrid to PV or widen a wrapper's qualified rank, convention, power, invariant,
kinematic or Dual domain. `--rescue` uses the facade's existing generic precision
rescue path. The CLI retains its unprofiled/default master-policy behavior;
capability/backend selection remains the core's responsibility. The outer CLI
stage timings do not enable the generic workspace profiler.

Results are JSON with decimal-string complex values, basis exponents, Laurent
slots `[uv1,ir1,ir2,finite]`, per-sample status, fallback and analytic-limit usage, and batch
execution counters. The output also records requested and prepared algorithms
and the actual workspace preparation notes. `engine_execution.engine` identifies
`Generic`, `SixPoint` or `SevenPoint` execution. For specialized engines its tagged
record retains packed child/parent rows, batch-call counts, nested reduction work
and both certificate counters. The legacy `execution` fields are the facade's
logical-input/work summary: child and parent work is counted once, while packed
rows and certificate passes remain in the tagged detail. Fallback/analytic work
can exceed the number of logical samples; do not add packed rows to the summary.
The algorithm field describes the selected engine's algorithm; the engine tag and
nested counters distinguish the actual composition.

Ordinary `diagnostics` contains the actual per-lane core messages. The existing
precision-rescue API does not expose its internal certificate counters or those
messages; rescue outputs mark these fields null instead of inventing zero counts
or empty messages. `input_policy` records the requested preserve/snapping
policy and exact threshold; comparison cases carry the same field.
`precision_control` records the accuracy and verification
policy. Per-sample `precision_rescue` preserves the primary status, convergence
result, and whether a successful primary result agreed with verification.
`rescue_execution` records requested, accepted and corrected samples, with
separate counters for both higher-precision evaluations; `execution` retains
the primary counters. Failed reductions have a reason and no tensor values;
failed numerator contractions retain successfully reduced moments and omit the
integral. Other samples remain available. Exit status is zero when all
samples succeed, one for reported numerical/comparison failures, and two for
input or setup errors. Preparation and one reduction execution are timed
separately; these observations are not statistically qualified benchmarks.

`compare` checks every component and pole against captured original IREGI
fixtures. It groups matching settings into batches, retains failures in the
report, and reports the worst error after scaling each moment by
`Q^(4+dimension_shift+rank-2*sum(powers))`, where
`Q=max(mu,abs(shift components),sqrt(abs(mass_squared)))`, matching the core
oracle's characteristic scale. The mixed error denominator is
`max(1,abs(scaled_reference))`. The default tolerance is `1e-10`.
Nonfinite normalization or error arithmetic is reported as failure.
Comparison cases also retain the tagged actual engine and per-lane diagnostics.
Execution counters on comparison cases describe the shared settings batch named
by `execution_batch_ids`; those repeated counters must not be summed per case.
`--case bubble` selects IDs containing a substring and `--algorithm ibp`
overrides each fixture's stored method. Higher-precision comparisons still use
the original binary64 expected values and cannot certify all extra digits.
After building the binary, `python3 validation/cli_smoke.py` checks graph import,
all three precision modes, contraction, supplied invariants, and lane isolation.

The physical top-loop `g g -> h` example includes model-derived vertices,
external helicity wavefunctions, and a selected equal-color component:

```sh
cargo run -p iregi-cli -- run examples/graphs/ggh_top_triangle.dot --model fixtures/models/sm.json --input validation/physical/ggh-cli-input.json --feynman-rules --external-wavefunctions --component 0,0 --convention oneloop
python3 validation/physical/compare_cli.py --cli target/debug/iregi-rust
```

The second command requires the isolated original IREGI executable built by
`bash scripts/build-reference.sh`. It checks all four Laurent slots of every
tensor moment at the FeynKit routing, then compares the integrated numerator to
the separately captured MadLoop graph with its documented color factor. Its
report records input, executable, and reference hashes. MadLoop's own tensor
variable differs by an affine shift, so its tensors are not compared directly
to FeynKit's tensors. The physical comparison excludes R2 and UV counterterms,
and retains bare poles and R1.

The CLI crate also exposes `iregi_cli::input` for its exact input schema and
parsing, and `iregi_cli::numerator::{CoefficientPlan, CoefficientWorkspace}`
for preparing and reusing batched Symbolica coefficient evaluators. Numerical
coefficient evaluation is separate from graph preparation and core tensor
reduction. The in-process `physical_benchmark` example uses these same helpers;
its measurement boundaries and correctness gates are documented in
[`validation/benchmark/PHYSICAL.md`](../../validation/benchmark/PHYSICAL.md).
