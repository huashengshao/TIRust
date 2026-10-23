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
