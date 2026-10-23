# Collector output qualification

The private alias-use collector candidate is compared with an unchanged core
from revision `6b24c0372022e22606a1ac85f30c7fa579fb41f0`. Both use the same
retained master and Symbolica archives. This capture tests numerical and
diagnostic equivalence; its development-profile runtimes do not qualify a
performance improvement.

The first attempt placed all 27 baseline cases and their candidate counterparts
under one 300-second bound. It saved the complete baseline after approximately
261.5 seconds, then timed out while evaluating the candidate. That attempt is
retained in `collector-capture-run.json`; it establishes no complete paired
agreement. Three of its baseline ordinary lanes were rejected: f64 N4 rank two,
f64 N4 rank-one second-order Dual, and f64 N5 rank one. They remain explicit
guard-parity controls in the next protocol.

The second protocol retains all 27 inputs and adds two conditioned f64 N4
controls. Each baseline or candidate child has an independent 300-second bound,
and writes a durable checkpoint after each case. The controls use zero and
orthogonal unit spatial shifts, squared masses `[1, 2, 3, 4]`, and scale 3 in
their ordinary lanes. Only their regular lanes change; the original deliberate
exceptional and invalid-scale lanes remain present.

The two new controls have passed a separate bounded probe. Both the ordinary
rank-two case and rank-one second-order Dual case return complete arrays in
the real-mass and complex-mass lanes, with exact baseline/candidate agreement.
Their other three lanes remain guarded. An untouched original IREGI executable
independently checks every native Laurent slot of the real-mass primal outputs.
The largest dimensionally normalized discrepancies are `5.97e-14` and
`6.50e-14`, respectively, against the required `1e-10` tolerance. The higher
Dual components have snapshot equivalence here; this probe does not independently
validate those derivatives.

`collector-v2-probe-qualification.json` binds the probe results, checkpoints,
build, source and captured original-IREGI request/response. Both full 29-case
children completed within their bounds, and their complete output records are
exactly equal. The first qualifier nevertheless rejected historical bitwise
preservation: the retained Float128 N5 analytic-limit lane has 29 different
numeric, precision or signed-zero leaves compared with the older baseline.
The largest numeric difference is exactly `2^-128`. All historical inputs,
statuses, flags and layout metadata agree. The strict checker and its failure
log remain immutable.

A fresh replay of the unchanged original binary, with the original case order,
reproduced all 27 historical records exactly. The changed preparation history
in the new driver is therefore a plausible source of the last-bit differences.
A source audit traces a possible mechanism through process-global alias
namespaces, Symbolica symbol ordering and floating-point accumulation order;
the specific mechanism has not been demonstrated by an instrumented execution.
The source audit has a separate frozen snapshot so later production edits do
not erase its evidence.

The final versioned qualification passed independent source review and its
complete provenance/numerical checks. It requires exact current baseline/candidate records, exact historical inputs,
statuses, flags and layout, and all historical numerical slots within the
existing dimensionally normalized `1e-10` tolerance. The previous additional
historical bitwise requirement is recorded as failed, with every changed leaf
retained. No cross-preparation bitwise reproducibility is claimed.

The next checker also exposed a cross-preparation bitwise difference between
the isolated two-control probe and the full-run controls: 178 numeric leaves,
with a maximum raw difference of about `1.10e-14`. That checker and failure log
are retained separately. All control inputs and nonarray metadata are exact.
The final protocol applies the same unchanged numerical tolerance to both
historical comparisons and directly compares the actual full-run control
primal arrays with the untouched original-IREGI response. Those direct checks
cover 60 and 20 complex Laurent slots, with maximum normalized errors of
`6.06e-14` and `6.74e-14`. All higher Dual components and complex-mass lanes are
included in paired and cross-preparation comparisons; they do not acquire an
independent original-IREGI derivative claim from the primal reference.

The qualifier requires 26 successful ordinary
lanes, separately counts the three retained ordinary guard cases, verifies all
27 original input records, and binds both new controls to the reference
probe. Mutation tests exercise its rejection of missing cases, altered
inputs, matching failures, truncated arrays, zero first or second Dual components,
changed control records, and missing provenance links. Those tests reuse prior
captures and do not substitute for the completed 29-case evaluation. Nine
additional tests cover the historical numerical comparator and its complete
Laurent/Dual/repeat/contraction coverage.

`collector-capture-v4-qualification.json` records 29 exact current pairs,
26 required ordinary successes, three retained ordinary guard cases and 86
successful lanes. These contain 4,520 complex tensor/Laurent/Dual components and
656 contracted components; every repeated tensor array also matches. Twelve
positive second-order Dual cases exercise nonzero first and second components.
The largest normalized historical discrepancy is `5.43e-37`; the largest
full-control/probe discrepancy is `8.65e-14`. This qualifies the reviewed
collector change on these inputs. It establishes neither generic high-point
coverage nor a controlled performance improvement.

The completed protocol used these commands:

```sh
source scripts/env.sh
source .local/symbolica-license.env
python3 validation/public-highpoint/run_collector_capture_v2.py collector-v2-baseline baseline --selection all --cpu 14 --limit 300
python3 validation/public-highpoint/run_collector_capture_v2.py collector-v2-candidate candidate --selection all --cpu 15 --limit 300
python3 validation/public-highpoint/qualify_collector_capture_v2.py # retained historical bitwise failure
python3 validation/public-highpoint/replay_collector_original.py
python3 validation/public-highpoint/qualify_collector_capture_v3.py # retained probe bitwise failure
python3 validation/public-highpoint/qualify_collector_capture_v4.py
```

The runners refuse to overwrite existing captures. A fresh reproduction needs
new output names and corresponding qualifier bindings. The original timeouts and
guards remain visible regardless of the new protocol's outcome. No full N6/N7
array agreement follows from these lower-point cases.
