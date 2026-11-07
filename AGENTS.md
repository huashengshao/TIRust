Binding: docs/PYAMPLICOL-COLLABORATION.md (pyAmpliCol collaboration protocol, 6 Nov 2026) overrides anything conflicting below, in PLAN.md and in HANDOFF.md.

# iregi-rust development

Follow PLAN.md and keep the active implementation goal open until its acceptance
criteria are fulfilled. Work from this repository; references are read-only except
for the explicitly requested ../references/PLAN.md. Build reference copies under
target/reference, never in the original source trees.

All production implementation is Rust. Keep graph/numerator preparation separate
from the reduction library. All symbolic algebra must use Symbolica or the
FeynKit/Spenso/Idenso ecosystem. Before introducing missing algebra, record three
reuse checks: Symbolica APIs/source, usage/tests, and FeynKit/gammaloop facilities.
No independent polynomial engine, symbolic simplifier, or matrix solver.

Batches must propagate through reduction and scalar masters. Preserve generic
precision and Dual components; no f64 round trips. A failed lane must not corrupt
other samples. Do not present unsupported domains or partial coefficients as
successful results. Maintain native UV/IR separation including scaleless cases.

Use original IREGI and graph-isolated MadLoop as independent references. Exclude
R2 and UV counterterms from physical comparisons, retaining bare poles and R1.
Reference repairs must be narrow, separately documented and independently tested.

TIRust's (Huasheng's) agent never pushes to, commits on or merges into main;
pyAmpliCol's agents push to main and merge pull requests (protocol sections
0-2). TIRust's agent commits validated milestones as huashengshao
<erdissshaw@gmail.com> on feature branches hs/<topic> and proposes each as a
pull request against main with reviewer ValentinHirschi, without force pushes
(protocol sections 0 and 3, including the freeze check before every push).
Before touching Cargo manifests, Cargo.lock, pins, LICENSE, .githooks or the
protocol's section 3 APIs, it opens an issue mentioning @ValentinHirschi
(labelled pyamplicol, plus pins for pin changes or api-change for API changes;
protocol section 6). Keep the published report link and measurement provenance
current; never invent timings or claim unperformed validation.

Slimness (protocol sections 2 item 6 and 5): validation/ keeps only the
fixtures that tests read and its README.md (scripts/keep-validation.txt); the
evidence lives in the private backup named in validation/README.md. New captures, raw
JSON/JSONL, logs and report data (docs/data/) are git-ignored; never force-add
them. A new fixture that a test must read goes to the test's own crate (for
example crates/iregi-rust/tests/fixtures/), not under validation/.

Blob-size policy (protocol section 5): MAX_BLOB_BYTES = 1 MiB (1048576 bytes).
No push adds a larger blob unless its exact path is in .blob-size-allowlist,
and none adds a blob under dependencies/symbolica/,
dependencies/symbolica-patches/ or dependencies/oneloopmaster/. Change this
value only by pull request.

At each milestone update docs/benchmark-summary.json (scripts/build-benchmark-summary.py)
and, if needed, docs/benchmark-summary.typ, then rebuild docs/benchmark.pdf with
bash scripts/build-benchmark-report.sh (fixed creation timestamp from the data,
embedded fonts only; --check verifies a byte-identical rebuild).
docs/benchmark.pdf stays tracked as a slim report of mostly the summary tables,
committed by pull request (keep it under 1 MiB, or add its exact path to
.blob-size-allowlist in the same pull request with a one-line justification);
give the user the link to it. The full report and the evidence go to
huashengshao/iregi-rust-evidence or a release asset (gh release create
tirust-bench-<date> ...). Run the freeze check first:
no tag or release while the FREEZE issue is open (protocol section 4); gh
release create makes the tag on the server, so the pre-push hook does not see
it (amendment of 6 Nov 2026 in docs/PYAMPLICOL-COLLABORATION.md). Include the measured original
IREGI-versus-COLLIER comparison on shared undotted topologies, separately from
Rust-versus-IREGI results; retain failed, noisy and unmeasured cases explicitly.

Use TIRust as the benchmark and Python-facing implementation name. Every
displayed implementation/reference ratio is TIRust/reference (below one is
faster). Preserve historical raw evidence in the archive
(huashengshao/iregi-rust-evidence), not in this repository; show reference-only
comparisons as absolute timings and outcomes, never infer TIRust ratios across
sessions.

Python bindings follow the pinned Community extension guide in PLAN.md: actual
upstream shared PyO3 types, direct Rust calls, one native host/kernel, rlib
registration and optional standalone host. Keep the Rust core Python-free.
Preserve precision and Dual shapes; provide typed stubs, fresh-wheel tests and
executable marimo examples. Cargo and feature changes follow the issue rule
above (protocol section 6); never compile during a timing capture.
