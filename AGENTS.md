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

Never push to, commit on or merge into main: pyAmpliCol's coordinator maintains
it. Commit validated milestones as huashengshao <erdissshaw@gmail.com> on
feature branches hs/<topic> and propose each as a pull request against main
with reviewer ValentinHirschi, without force pushes (protocol sections 0 and 3,
including the freeze check before every push). Coordinate shared-file changes
with the root agent. Keep docs/benchmark.pdf and measurement provenance
current; never invent timings or claim unperformed validation.

Blob-size policy (protocol section 5): MAX_BLOB_BYTES = 1 MiB (1048576 bytes).
No push adds a larger blob unless its exact path is in .blob-size-allowlist,
and none adds a blob under dependencies/symbolica/,
dependencies/symbolica-patches/ or dependencies/oneloopmaster/. Change this
value only by pull request.

Update docs/benchmark.typ at each milestone. Build docs/benchmark.pdf locally
(git-ignored), publish it as a release asset (gh release create
tirust-bench-<date> docs/benchmark.pdf) or in huashengshao/iregi-rust-evidence,
and give the user that link. Include the measured original
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
executable marimo examples. Coordinate Cargo/feature changes and compilation
with active numerical provenance guards and benchmark quiet windows.
