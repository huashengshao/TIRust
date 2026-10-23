# Full-path N6 capture after exact alias-use deduplication

The private candidate freezes core revision
`6b24c0372022e22606a1ac85f30c7fa579fb41f0`, applies only the independently
reviewed 52-line alias-use patch, and adds stderr phase timers. The amplitude
accumulator remains byte-identical to that revision. No reference collector,
early-stop hook, n-way accumulator change, reference scalar substitution or
stability relaxation is included.

The frozen build passes all 38 stability and 35 execution unit tests. A separate
qualifier binds the complete module membership to the test binary's listing of
174 tests, checks every captured test name and verifies zero failed or ignored
tests. This includes inactive poles, per-sample master activation and
master-dependent activation closure.

The ordinary public N6 rank-seven call uses the same conditioned-mass fixture,
exact coordinate inputs, all 330 requested moments, Float128 working precision
and a 53-bit accuracy request. It reached its 600-second limit during workspace
preparation. Process termination and evidence collection completed in
601.010 seconds. Numerical reduction was never entered; no tensor arrays exist
for comparison.

| Preparation stage | Diagnostic elapsed time | Observed extent |
| --- | ---: | --- |
| Structural plan | 44.305 s | 31,670 aliases; 56 masters |
| Reverse activation | 10.190 s | All 25,700 reachable aliases |
| Local stability checks | 186.845 s | 19,254 checks; 576,834 expressions |
| Primal amplitude construction | At least 344.409 s | Last progress marker: alias 30,720 of 31,670 |

The last phase has no completion marker. The evidence therefore localizes the
remaining preparation cost but does not isolate time spent discovering
variables, extracting Symbolica polynomial coefficients or accumulating terms.
A finer diagnostic can separate those operations before selecting another
optimization. Increasing the timeout alone is not treated as an improvement.

This run used CPU14, one Rayon thread, the private user-supplied Symbolica
license and the same retained master/Symbolica/OneLoop archives as the earlier
public baseline. The core uses development-profile flags and the driver uses
optimization level 2. Timings are diagnostics, not performance-gate evidence;
the earlier 180/300-second timeouts remain retained.

`validation/public-highpoint/use-dedup-full-capture-qualification.json` binds
the full source/build/input/runtime ledger and explicitly records
`complete_successful_array_captured=false`. The source and dependency audit
found only the intended collector and timer changes, with all 608 Rust archive
or shared-library inputs unchanged through compilation. Native library paths
are retained in the build commands; the 608-file guard covers Rust artifacts.

The separate numerical comparator passes 16 new and 10 existing Python
regression tests. Single-array mode can inspect strict `1e-10` dimensionally
normalized original-IREGI agreement and independent zero poles. Optional
two-array mode additionally checks increasing working precision on the same
frozen build. Neither mode was run on this timeout capture, which has no
successful numerical array.

Reproduction commands, with a fresh capture name, are:

```sh
source scripts/env.sh
source .local/symbolica-license.env
python3 validation/public-highpoint/run_use_dedup_full.py fresh-name 6 --limit 600 --cpu 14
```

The runner requires the completed semantic qualification for its frozen build
and refuses to overwrite existing captures. The bounded-capture qualifier is
bound to the retained capture name; adapt that binding for a fresh reproduction.
