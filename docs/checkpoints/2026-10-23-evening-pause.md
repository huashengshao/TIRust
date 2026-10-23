# User-requested pause, 23 October 2026

All numerical workers and their reviewers are idle. The current bounded run
finished before this checkpoint; no retry, statistical analysis, new numerical
study or production adoption was started during the pause. The implementation
goal remains incomplete and is paused at the user's request.

The preceding pushed checkpoint is `e3669d08e81e24efdb5f98d32eb0aa2b410b8c74`.
This checkpoint updates the plan, working instructions and report presentation.
It preserves raw records for the last quiet window without changing measured
performance conclusions. Production crates, dependency sources and Cargo files
are unchanged by this checkpoint. The requested reference PLAN remains identical
to the committed project PLAN.

## Public state and limitations

- The Rust library and CLI already implement batched tensor reduction and
  contraction with generic precision and Dual support. Positive propagator
  powers (dotted propagators) are supported; the strongest generalized-interface
  qualification is through four propagators. Complete seven-propagator,
  rank-seven qualification remains open.
- Benchmark naming is **TIRust**. All displayed implementation/reference ratios
  place TIRust in the numerator; below one means faster. Historical reference-only
  IREGI/COLLIER comparisons use absolute timings and outcomes. Never construct a
  TIRust/COLLIER ratio from measurements taken in different sessions.
- Full reduction performance parity is not established. Native A0/B0/C0 kernel
  improvements do not establish whole-reduction parity. Existing slow, noisy,
  unstable and unsupported cases remain visible in the report.
- The new PyO3 interface and marimo gallery are private source drafts. They have
  not been compiled, installed, type checked or executed. No usable Python
  release or notebook runtime qualification is claimed.

## Last quiet window: raw data saved, analysis deferred

`validation/benchmark-quiet/20261023e/completion.json` binds the immutable CPU254
quiet-window manifest and byte-preserved raw records.

1. Selected native masters, baseline TIRust, original IREGI and COLLIER:
   preflight stopped after 9.623 seconds, before any calibration or timing.
   All eight cases remain unmeasured. One baseline C0 finite imaginary component
   differs by one ULP between CASE-first and CONTROL-first preparation schedules.
   The cause is unresolved; no gate was relaxed and no retry was performed.
   The copied `selected-native/handoff.md` records the exact leaf and next
   proposed diagnostic.
2. Lazy versus eager receipt capture: all 14 cases and 1,148 measurement rows
   completed in 256.685 seconds within the existing 600-second cap. Runner
   provenance checks passed. CSV, log, receipt and frozen analyzer are saved in
   `lazy-capture/`. **The analyzer has not been executed.** Do not infer a speedup
   from capture success. On resume, analyze all rows, independently reconstruct
   the statistics and retain inconclusive or regressing families. No rerun is
   implied by this checkpoint.

## Private work preserved locally

These paths are under the ignored `target/` directory. Their compiled artifacts
and unqualified source drafts are intentionally not promoted into production by
this commit. Preserve them when resuming on this workspace.

| Path | State and next step |
| --- | --- |
| `target/selected-native-optimized/` | Optimized matched full-call A/B/C candidate passed numerical admission; four-engine timing failed its strict preflight as above. Inspect deterministic evaluator preparation before proposing a bounded diagnostic. |
| `target/raw-distance-lazy-capture-optimized/` | Fourteen-family exact preflight and final timing capture completed. Analyze the frozen capture before any optimization/adoption decision. |
| `target/public-highpoint/thin-n6-mask/` | Exact-zero cofactor pruning uses full Taylor components and Symbolica certificates. Build passed; the first exact-test attempt hit its retained 30-second cap before completing a test. The independently reviewed all-14-test, 180-second protocol remains unrun. |
| `target/public-highpoint/thin-n6-mask-phase/` | Instrumented diagnostic built and one mixed-shape test passed in 35.04 seconds. This is not an integral or convergence qualification. No causal performance diagnosis was established. |
| `target/batch-binary-fold-candidate/` | Private Symbolica binary Add/Mul loop change built. Paired public-API witnesses matched byte-for-byte across 440 calls and 40,256 real components per variant. Matched downstream whole-reduction builds, full controls and performance remain pending. Never mix baseline and candidate Symbolica crate identities. |
| `target/python-interface-stage/HANDOFF.md` | PyO3 API, one-native-host design, bindings, facade and stub drafts saved; 18-file inventory is bound in `source-inventory.json`. No build. Exponent overflow validation and owner API exposure remain open. |
| `target/marimo-stage/DESIGN.md` | Three notebooks, presentation helpers and tests saved: DOT/numerator reduction, dots/derivatives and benchmark exploration. Real-wheel integration, lint and headless export remain unrun. |

The Python design follows the pinned Community extension guide recorded in PLAN:
actual upstream PyO3 wrappers, direct Rust calls and one native host/kernel.
Avoid serializing expressions between layers or installing incompatible native
hosts together. The pinned guide's current dependency revisions differ from the
validated numerical closure; no silent dependency upgrade is authorized by the
source draft. DoubleFloat transport must preserve widely separated low parts
using existing Numerica conversion APIs; the source fixes still need runtime
round-trip tests. Polynomial variable validation is drafted; exponent overflow
and Symbolica's unchecked casts require a complete fallible path.

## Completed local evidence awaiting a later evidence milestone

The following untracked sources and evidence remain in the workspace. Their
qualification records distinguish source-only proposals from actual captures;
they are not added to the report by this presentation checkpoint.

- `validation/highpoint-child-localization/`: exact Symbolica localization of
  two singular pentagon children and a mixed-derivative counterexample proving
  that primal-zero pruning is invalid.
- `validation/e6-direct-n2-typed/`: bounded private N2 expansion captures across
  f64, DoubleFloat, Float and static/dynamic mixed Dual shapes. Accepted fixtures
  use common complex mass and zero invariants; heterogeneous derivatives remain
  unqualified. The initial compile failure is retained.
- `validation/e6-direct-n2-expansion-design/` and
  `docs/research/e6-n2-heterogeneous-validation.md`: reviewed source-only proposal
  for heterogeneous momenta/masses and higher derivatives; no execution.
- `validation/collier-reference/shared/`: two preserved extended-reference
  attempts. The corrected attempt reproduces the existing dedicated B/C/D
  interfaces exactly and accepts three N5 singleton cases. New mixed-scale
  batch transport and an N6 COLLIER heap abort remain unresolved. No timings.
- `docs/research/collier-n5-n7-harness-design.md`: extended harness design. Keep
  rejected, crashed and unmeasured rows visible; isolate reference failures and
  transport unchanged scale groups before retrying any new cohort.

## Resume order

Read this file, PLAN.md and AGENTS.md; inspect the saved agent handoffs and actual
repository status. Resume the existing goal only when the user asks. Source the
existing environment and ignored license file without printing its value.

First analyze the already finished lazy/eager cohort and independently audit it.
Resolve the four-engine control determinism issue before another timing attempt.
Then continue the bounded high-point exact tests and the PyO3 owner/API fixes,
coordinating all shared Cargo changes and builds. Qualify fresh Python wheels
and notebooks before presenting them as available. Preserve all earlier failed
attempts and enforce the original numerical and performance acceptance gates.

Commit and push reviewed evidence and implementation milestones without force
pushes. Rebuild the Typst/PDF report when qualified coverage or measured
performance changes, and give the user the pushed GitHub PDF link.
