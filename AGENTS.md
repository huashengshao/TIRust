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

Commit and push validated milestones to origin/main, without force pushes, as
huashengshao <erdissshaw@gmail.com>. Coordinate shared-file changes with the root
agent. Keep docs/benchmark.pdf and measurement provenance current; never invent
timings or claim unperformed validation.

Update docs/benchmark.typ and docs/benchmark.pdf at each pushed milestone and
give the user a GitHub link to that pushed PDF. Include the measured original
IREGI-versus-COLLIER comparison on shared undotted topologies, separately from
Rust-versus-IREGI results; retain failed, noisy and unmeasured cases explicitly.

Use TIRust as the benchmark and Python-facing implementation name. Every
displayed implementation/reference ratio is TIRust/reference (below one is
faster). Preserve historical raw evidence; show reference-only comparisons as
absolute timings and outcomes, never infer TIRust ratios across sessions.

Python bindings follow the pinned Community extension guide in PLAN.md: actual
upstream shared PyO3 types, direct Rust calls, one native host/kernel, rlib
registration and optional standalone host. Keep the Rust core Python-free.
Preserve precision and Dual shapes; provide typed stubs, fresh-wheel tests and
executable marimo examples. Coordinate Cargo/feature changes and compilation
with active numerical provenance guards and benchmark quiet windows.
