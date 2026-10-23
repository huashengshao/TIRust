# Public N6 rank-seven preparation after collector and sum changes

The frozen combined candidate still does not produce a complete public N6
tensor array within the unchanged 600-second bound. The process was killed
after 601.2318 seconds while preparing the workspace. Numerical reduction
never started. The earlier 180-, 300-, and collector-only 600-second failures
remain separate evidence.

This candidate uses the frozen `6b24c03` core, the qualified alias-use
collector, the separately qualified `Atom::add_many` amplitude sum, and fine
diagnostic hooks. It retains the matched historical master archives; it is
not a build of the later production core or latest master implementation.
The input is the existing conditioned N6 fixture, all 330 moments through
rank seven, PV, Float with 128 working bits and 53 requested bits, one lane.
The public coordinate-input plan/workspace path and all guards are enabled.

The final log localizes the timeout more precisely than its coarse
`preparing_workspace` status:

| Diagnostic phase | Seconds | Outcome |
| --- | ---: | --- |
| Structural plan | 41.7608 | Completed |
| Reverse alias-use propagation | 10.5369 | Completed, 25,700 reachable aliases |
| Local stability checks | 185.9402 | Completed, 19,254 checks and 576,834 roots |
| Primal amplitudes | 346.5734 | Completed |
| Master activations | — | Entered; no completion before timeout |

Phase times overlap where the log reports nested work and must not be added
indiscriminately. The primal phase counted 951,437 dependency scans totaling
17.6991 seconds. Selected per-expression timings suggest variable discovery
deserves investigation, but they are not aggregate proof that it dominates
every phase. The diagnostic build and instrumentation do not establish a
performance improvement or pass a performance gate.

The prerequisite exact-expression witness passed 20 cases, and 38 stability
plus 35 execution tests passed. The separate ordinary capture qualification
compares 29 complete baseline/candidate records exactly, with 86 successful
lanes and 4,520 complex moment/Laurent/Dual slots. Those checks establish their
recorded scopes, not full N6 numerical coverage. Historical preparation-order
differences and all failed strict qualifiers remain recorded in
`highpoint-collector-capture.md`.

Durable evidence under `validation/public-highpoint/`:

- `collector-nway-full-qualification.json`, SHA256
  `6e95ce5798c410f72ee07146147c665e86868a4e77b8f71391b1186db9dc8f6a`.
- `n6-rank7-collector-nway-128-53-run.json`, SHA256
  `913bf0c1a841be48e67d5ac5483c05dce49db2bfcf927bd79fe8e01733ed6915`.
- `n6-rank7-collector-nway-128-53.json`, SHA256
  `d44d454db92dd02d70cbffc85b878650c868170376663a959da550a95c04bcef`.
- `n6-rank7-collector-nway-128-53.log`, SHA256
  `0cbe690d20264eab20a6b09189251514db4d203eeaa89f57100c2d5ba70ec8c6`.

The runner verified all 25 frozen source files, binary, input, and prerequisite
hashes. Reproduction requires sourcing `scripts/env.sh` and the private
license environment, then running the following with a new capture name to
preserve the existing artifacts:

```sh
python3 -B validation/public-highpoint/run_collector_nway_full.py \
  NEW_CAPTURE_NAME 6 --rank 7 --algorithm pv --working-bits 128 \
  --requested-bits 53 --rows 1 --cpu 14 --limit 600
```

The collector's later production promotion is distinct from this frozen
run. The reviewed collector patch is byte-identical to the qualified patch;
the pre-adoption review manifest remains a historical record. No numerical
N6/N7 agreement or controlled performance qualification follows from it.
