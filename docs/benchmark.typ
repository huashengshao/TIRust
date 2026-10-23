// Measured values live in data/results.json. null means no measurement.
#let data = json("data/results.json")
#set document(title: "iregi-rust benchmark", author: "Hua-Sheng Shao")
#set page(paper: "a4", margin: (x: 1.6cm, y: 1.8cm), numbering: "1 / 1",
  header: context {
    if counter(page).get().first() > 1 {
      set text(8pt, fill: luma(110))
      [iregi-rust benchmark #h(1fr) #data.date]
    }
  },
)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.6em)
  block(text(14pt, fill: rgb("#1f3b63"), it))
  v(0.2em)
}
#show heading.where(level: 2): it => block(text(11.5pt, fill: rgb("#1f3b63"), it))
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
#show link: set text(fill: rgb("#1f5fa8"))
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let amber = rgb("#fff0cc")
#let pending = rgb("#f0f1f3")
#let palette = gradient.linear(rgb("#5aae61"), rgb("#d9f0d3"), white, rgb("#fddbc7"), rgb("#d6604d"))
#let is-num(x) = type(x) == int or type(x) == float
#let ratio(a, b) = if is-num(a) and is-num(b) and b != 0 { a / b } else { none }
#let ratio-fill(r) = if r == none { pending } else {
  palette.sample(calc.clamp(0.5 + calc.log(r, base: 2) / 4, 0, 1) * 100%)
}
#let fixed(x, digits: 2) = {
  let parts = str(calc.round(x, digits: digits)).split(".")
  let tail = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + if digits > 0 { "." + tail + "0" * (digits - tail.len()) } else { "" }
}
#let cmp-cell(a, b, reference: "IREGI") = {
  let r = ratio(a, b)
  table.cell(fill: ratio-fill(r), {
    set par(leading: 0.35em, justify: false)
    if r == none {
      text(size: 8.5pt, weight: "semibold", fill: luma(85))[Unmeasured]
      linebreak()
      text(size: 7pt, fill: luma(115))[vs #reference]
    } else {
      text(size: 9pt, weight: "bold")[#fixed(a) µs]
      linebreak()
      text(size: 7pt, fill: luma(70))[#reference #fixed(b) #h(3pt) ×#fixed(r)]
    }
  })
}
#let component(record, name) = if record == none { none } else { record.at(name, default: none) }
#let table-stroke = (x, y) => (
  top: if y <= 1 { none } else { 0.4pt + luma(225) },
  left: if x >= 2 and y > 0 { 0.4pt + luma(225) } else { none },
)
#let status-cell(status) = table.cell(fill: if status == "source-audited" or status == "partial" { pale } else if status == "verified" { rgb("#d9f0d3") } else { amber },
  text(size: 8pt, weight: "semibold", if status == "source-audited" { "Source audited" } else if status == "verified" { "Verified" } else if status == "partial" { "Partial" } else { "Pending" }))
#let report-table(body) = {
  set text(font: "DejaVu Sans", size: 8pt)
  set par(justify: false)
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  body
}

#let bench = data.qualified_benchmark
#let native = data.qualified_native_benchmark
#let recent = data.qualified_5c9_benchmark
#let current = data.qualified_80949_benchmark
#let physical = data.physical_80949_benchmark
#let benchmark-cell(r) = table.cell(fill: ratio-fill(r.paired_ratio), {
  set par(leading: 0.25em, justify: false)
  text(size: 9pt, weight: "bold")[#fixed(r.rust_us_per_point, digits: 3) µs]
  linebreak()
  text(size: 7pt)[IREGI #fixed(r.iregi_us_per_point, digits: 3) µs · ×#fixed(r.paired_ratio, digits: 3)]
  linebreak()
  text(size: 6.5pt)[99% CI #fixed(r.paired_ratio_ci99.at(0), digits: 3)–#fixed(r.paired_ratio_ci99.at(1), digits: 3)]
})
#let stage-colors = (rgb("#4c78a8"), rgb("#59a14f"), rgb("#f28e2b"), rgb("#b8b8b8"))
#let recent-cell(r) = table.cell(fill: if r.gate == "inconclusive-noise" { amber } else { ratio-fill(r.paired_ratio) }, {
  set par(leading: 0.3em, justify: false)
  text(size: 11pt, weight: "bold")[#fixed(r.rust_us_per_point, digits: 3) µs]
  linebreak()
  text(size: 8pt)[IREGI #fixed(r.iregi_us_per_point, digits: 3) µs · ×#fixed(r.paired_ratio, digits: 3)]
  linebreak()
  text(size: 7pt)[99% CI #fixed(r.paired_ratio_ci99.at(0), digits: 3)–#fixed(r.paired_ratio_ci99.at(1), digits: 3)]
  linebreak()
  text(size: 7.5pt, weight: "bold")[#if r.gate == "inconclusive-noise" { "Inconclusive: noise" } else { "Regression" }]
})
#let stage-bar(r, maximum: 40) = {
  let v = r.profile_us_per_point
  let vals = (v.arguments, v.masters, v.reduction, v.other)
  box(width: 10.5cm, height: 12pt, fill: rgb("#f3f4f6"))[
    #stack(dir: ltr, spacing: 0pt, ..vals.enumerate().map(((i, val)) =>
      rect(width: 10.5cm * val / maximum, height: 12pt, fill: stage-colors.at(i), stroke: none)))
  ]
}

#align(center)[
  #v(0.4em)
  #text(22pt, weight: "bold", fill: navy)[iregi-rust benchmark]
  #v(-0.3em)
  #text(12pt, fill: luma(80))[Batched one-loop tensor reduction in Rust versus IREGI and MadLoop]
  #v(0.2em)
  #text(9.5pt, fill: luma(110))[#data.date · #data.host]
]
#v(0.6em)
#block(fill: pale, inset: 10pt, radius: 4pt, width: 100%)[
  #set text(9.5pt)
  *Reading guide.* Every timing ratio is _iregi-rust time ÷ reference time_.
  Green means faster, white parity and red slower; timings are per phase-space
  point. Gray cells are unmeasured; amber marks pending or inconclusive evidence.
  Source inspection does not establish numerical agreement. #data.measurement_note
]
#v(0.4em)
#block(stroke: 0.6pt + luma(200), inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Acceptance gates*
  #table(columns: (1fr, 2fr, auto), stroke: none, inset: (x: 4pt, y: 4pt),
    [*Correctness*], [Agreement with original IREGI and isolated MadLoop graphs], [*Pending*],
    [*Kernel performance*], [Equal or better on identical inputs; separate measured snapshots], [*Unmet*],
    [*Complete evaluation*], [Six physical rows measured; two failures retained], [*Inconclusive*],
  )
]
= Historical kernel baseline: #raw("4c32ece")
#figure(kind: table, caption: [*Massive bubble/triangle, binary64, hybrid mode.*
  Rust and IREGI total medians in µs/point. Ratios are paired-block medians;
  intervals are 99% bootstrap confidence intervals. Batch 1 is latency;
  batch 32 is amortized cost per point. Each backend uses its own paired reference.],
  report-table(table(
    columns: (0.8fr, auto, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    inset: (x: 5pt, y: 10pt), stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([Case], [Rank], [Interpreter\ Batch 1], [Interpreter\ Batch 32], [Native\ Batch 1], [Native\ Batch 32]),
    ..for r in bench.rows.filter(r => r.batch_width == 1) {
      let batch = bench.rows.find(x => x.id == r.id and x.batch_width == 32)
      let nr = native.rows.find(x => x.id == r.id and x.batch_width == 1)
      let nb = native.rows.find(x => x.id == r.id and x.batch_width == 32)
      ([*#r.label*], [#r.rank], benchmark-cell(r), benchmark-cell(batch), benchmark-cell(nr), benchmark-cell(nb))
    },
  )),
)
#align(center)[
  #set text(8pt)
  #box(width: 9cm, height: 0.35cm, fill: palette, radius: 2pt)
  #linebreak()
  #box(width: 9cm)[×1/4 #h(1fr) ×1/2 #h(1fr) ×1 #h(1fr) ×2 #h(1fr) ×4]
]
#text(size: 9pt)[Both backends pass all eight numerical checks at $10^(-10)$.
The interpreter has seven regressions and one improvement; native has five
regressions, one parity result and two improvements. *Absolute IREGI timings
changed substantially between sessions*, despite identical inputs/reference
binary and CPU settings. Each verdict uses its own paired run; no direct
backend speedup is inferred across sessions. Both use the historical #raw("4c32ece") revision;
higher-rank performance remains unqualified.]

#text(size: 9pt)[The newer optimized interpreter snapshot #raw("5c9e3dd")
passes eight numerical checks, with six timing regressions and two noisy rows.
The latest #raw("80949db") default-stack/generalized-PV/vector-solve snapshot
also passes all eight checks: five regressions and three noisy rows.
Separate same-session comparisons follow; performance acceptance is unmet.]

#text(size: 9pt)[Newer validation includes four integrated graphs. The top
pentagon at Float 128/256 agrees with PV/hybrid and MadLoop; original IBP still
fails. The latest physical timing comparisons remain inconclusive.]

#pagebreak()
= Historical stage profiles: #raw("4c32ece")
Rust medians from separate instrumented passes, in µs/point. All bars use
0–40 µs/point; S denotes the Symbolica interpreter, N the generated native
backend. Sessions are separate: compare each with its own reference on page 1,
not with the other backend's absolute bar length. Raw ns/batch are divided
by $1000 times "batch width"$.

#report-table(table(
  columns: (3.5cm, 10.5cm, 1.5cm), inset: (x: 3pt, y: 3pt), stroke: none,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case / batch / backend], [Rust stage time: 0 #h(1fr) 20 #h(1fr) 40 µs/point], [Total]),
  ..for r in bench.rows {
    let nr = native.rows.find(x => x.id == r.id and x.batch_width == r.batch_width)
    ([#r.label r#r.rank / #r.batch_width · S], stage-bar(r), [#fixed(r.profile_us_per_point.total, digits: 3)],
     [#r.label r#r.rank / #r.batch_width · N], stage-bar(nr), [#fixed(nr.profile_us_per_point.total, digits: 3)])
  },
))
#align(center)[
  #set text(font: "DejaVu Sans", size: 8pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(0)) Arguments #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(1)) Scalar masters #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(2)) Reduction #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(3)) Other
]
#v(5pt)
#text(size: 8.5pt)[“Other” is total median minus named stage medians, not a
separately timed region. Scalar-master time includes complete master calls;
reduction includes stability checks. IREGI has no measured stage split here.
For native triangle rank 0, scalar masters cost 3.282 µs at batch 1, but 249.977 µs
per batch 32, or 7.812 µs/point: this batch path has higher amortized master
cost. The observation identifies a profiling target, not a cross-session
speedup or a proven cause. Full stage values and samples remain in the raw data.]

== Preparation and measurement conditions
#report-table(table(
  columns: (1.2fr, 1fr, 1fr, 1fr, 1fr), inset: (x: 6pt, y: 6pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Backend], [Bubble r0], [Bubble r1], [Triangle r0], [Triangle r1]),
  [Interpreter], ..bench.rows.filter(r => r.preparation_seconds != none).map(r =>
    [#fixed(r.preparation_seconds, digits: 3) s]),
  [Native], ..native.rows.filter(r => r.preparation_seconds != none).map(r =>
    [#fixed(r.preparation_seconds, digits: 3) s]),
))
Preparation values are single observations without repeated uncertainty,
excluded from warm kernel ratios. Batch32 reuses the plan; numerator
preparation is outside this tensor-array benchmark.

Both backends of the isolated pushed revision #raw("4c32ece") ran on AMD EPYC 9754 CPU 0,
with 5 warm-ups and 21 paired blocks targeting 30 ms, randomized order with
seed 271828. Rust used release opt-level3, thin LTO and one codegen unit;
Fortran used #raw("-O3"). Structural plans and allocation capacity were reused;
numerical results were recomputed, and original recurrence caches were freed
per sample. Subprocess transport is excluded. All source/binary provenance
checks pass, and every ratio interval is no wider than 2% of its median.
Raw samples, inputs, flags and hashes are retained in
#raw("validation/benchmark/baseline-4c32ece/") and
#raw("validation/benchmark/native-4c32ece/").

#text(size: 8.5pt)[A separate frozen-native policy experiment keeps #raw("Whole"):
all 18 C0/D0 comparisons across three orderings find #raw("FixedRows4/8/16")
slower, with per-comparison 99% intervals excluding parity. It uses 32 valid
lanes, CPU 0 and 21 rotated blocks. This selects a batch policy, not IREGI
speedup; see #raw("validation/native-policy/").]

#text(size: 8.5pt)[A later native allocator candidate is rejected: the corrected
24-row protocol gives two singleton parities and 22 noisy rows, with no qualified
batch32 gain. Earlier results (two regressions/22 noisy rows) remain retained.
D0 compilation/qualification is pending. See #raw("validation/native-stack-batch/").]

#pagebreak()
= Latest kernel snapshot: #raw("80949db")
All eight massive bubble/triangle checks pass the unchanged $10^(-10)$
criterion, maximum normalized error $2.46 times 10^(-13)$. Five timing rows
regress; three exceed the unchanged 2% interval-width limit and remain
*inconclusive*. This optimized snapshot includes the adopted master stack
allocation, generalized PV and lazy Cayley solve. Later core-allocation and
experimental changes are excluded. The performance requirement remains unmet.

#figure(kind: table, caption: [*Binary64 hybrid interpreter, µs/point.*
  Latency at batch 1 and amortized cost at batch 32. Each ratio uses its
  same-run paired original-IREGI measurement.], report-table(table(
  columns: (1fr, auto, 2fr, 2fr),
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  inset: (x: 7pt, y: 10pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case], [Rank], [Batch 1], [Batch 32]),
  ..for r in current.rows.filter(r => r.batch_width == 1) {
    let batch = current.rows.find(x => x.id == r.id and x.batch_width == 32)
    ([*#r.label*], [#r.rank], recent-cell(r), recent-cell(batch))
  },
)))

== Separately instrumented component timings
#report-table(table(
  columns: (3.5cm, 10.5cm, 1.5cm), inset: (x: 3pt, y: 3pt), stroke: none,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case / batch], [Rust stage time: 0 #h(1fr) 25 #h(1fr) 50 µs/point], [Total]),
  ..for r in current.rows {
    ([#r.label r#r.rank / #r.batch_width], stage-bar(r, maximum: 50), [#fixed(r.profile_us_per_point.total, digits: 3)])
  },
))
#align(center)[
  #set text(font: "DejaVu Sans", size: 8pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(0)) Arguments #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(1)) Scalar masters #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(2)) Reduction #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(3)) Other
]
#text(size: 8.5pt)[Stages use a separate instrumented pass. “Other” is its
total median minus named medians; stage sums are not the uninstrumented gate.
Graph/numerator preparation is outside this tensor-array benchmark. The
single plan-preparation observations remain in the data, not in speed ratios.]

CPU 0, five warm-ups and 21 randomized paired blocks targeting 30 ms use
identical inputs, complete tensor arrays and all native Laurent slots.
Structural allocations are reused; numerical results are recomputed and
reference caches cleared per sample. Transport is excluded. Rust 1.99 uses
opt-level3, thin LTO and one codegen unit; GFortran 15.2 uses #raw("-O3").

All 401 immutable build-source hashes, both binaries and the harness match
the retained provenance. Other project computations were held; a pending
native compiler was suspended. Host activity outside the project was not
controlled. The accepted license contains no tracked key material.
This is a separate session from the historical tables; its absolute timings
establish no revision speedup. Raw samples and reproduction commands:
#raw("validation/benchmark/baseline-80949db-licensed/").

#pagebreak()
= Historical kernel snapshot: #raw("5c9e3dd")
All eight massive bubble/triangle checks pass the unchanged $10^(-10)$
criterion. Six timing rows regress; two exceed the allowed 2% relative
confidence-interval width and remain *inconclusive*. Amber cells denote noise,
not improvement. This commit predates default stack reuse, generalized PV
and the vector-solve candidate. The overall performance requirement is unmet.

#figure(kind: table, caption: [*Binary64 hybrid interpreter, µs/point.*
  Latency at batch 1 and amortized cost at batch 32. Each ratio uses its
  same-run paired original-IREGI measurement.], report-table(table(
  columns: (1fr, auto, 2fr, 2fr),
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  inset: (x: 7pt, y: 10pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case], [Rank], [Batch 1], [Batch 32]),
  ..for r in recent.rows.filter(r => r.batch_width == 1) {
    let batch = recent.rows.find(x => x.id == r.id and x.batch_width == 32)
    ([*#r.label*], [#r.rank], recent-cell(r), recent-cell(batch))
  },
)))

== Separately instrumented component timings
#report-table(table(
  columns: (3.5cm, 10.5cm, 1.5cm), inset: (x: 3pt, y: 3pt), stroke: none,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case / batch], [Rust stage time: 0 #h(1fr) 20 #h(1fr) 40 µs/point], [Total]),
  ..for r in recent.rows {
    ([#r.label r#r.rank / #r.batch_width], stage-bar(r), [#fixed(r.profile_us_per_point.total, digits: 3)])
  },
))
#align(center)[
  #set text(font: "DejaVu Sans", size: 8pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(0)) Arguments #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(1)) Scalar masters #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(2)) Reduction #h(9pt)
  #box(width: 8pt, height: 8pt, fill: stage-colors.at(3)) Other
]
#text(size: 8.5pt)[The stage sum belongs to a separate instrumented pass;
“Other” is its total median minus named medians. These bars locate costs,
not differences between revisions. Preparation is a single observation:
bubble r0/r1 0.822/0.112 s; triangle r0/r1 0.067/0.061 s. It is excluded from
the warm kernel ratios.]

CPU 0, five warm-ups and 21 randomized paired blocks targeting 30 ms use identical
inputs and all requested tensor slots. Structural allocations are reused;
numerical results are recomputed. Transport is excluded. Rust 1.99 uses
opt-level3, thin LTO and one codegen unit; GFortran 15.2 uses #raw("-O3").
All 391 source hashes, both binaries and the harness remain
unchanged under the retained provenance checks. Absolute timings from earlier
sessions cannot establish a revision speedup. Raw blocks, inputs, manifests
and reproduction commands: #raw("validation/benchmark/baseline-5c9e3dd/").

#pagebreak()
= Physical pipeline measurements: #raw("80949db")
Six preserved-input binary64 comparisons pass complete canonical tensors
against all three original modes and all three isolated graph Laurent slots
against MadLoop. *Every measured timing row remains inconclusive*: its 99%
interval exceeds the unchanged 2% relative-width threshold. Both top-pentagon
rows fail during preparation; no output or timing substitutes for them.

#let physical-cell(r) = table.cell(fill: amber, {
  set par(leading: 0.3em, justify: false)
  if r.status != "measured" {
    text(size: 10pt, weight: "bold")[Failed before timing]
    linebreak()
    text(size: 8pt)[Preserved-input conditioning guard]
  } else {
    text(size: 11pt, weight: "bold")[#fixed(r.rust_ms_per_point, digits: 3) ms]
    linebreak()
    text(size: 8pt)[MadLoop #fixed(r.madloop_ms_per_point, digits: 3) ms · ×#fixed(r.ratio, digits: 3)]
    linebreak()
    text(size: 7pt)[99% CI #fixed(r.ci99.at(0), digits: 3)–#fixed(r.ci99.at(1), digits: 3)]
    linebreak()
    text(size: 7.5pt, weight: "bold")[Inconclusive: noise]
  }
})
#figure(kind: table, caption: [*Equivalent complete numerical pipelines,
  ms/point.* Repeated captured point; amortized batch32 cost does not establish
  throughput over varying phase-space samples.], report-table(table(
  columns: (1.2fr, auto, 2fr, 2fr), inset: (x: 7pt, y: 10pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Graph], [Ranks\ Rust/ML], [Batch 1], [Batch 32]),
  ..for r in physical.rows.filter(r => r.batch_width == 1) {
    let batch = physical.rows.find(x => x.case == r.case and x.batch_width == 32)
    ([*#r.label*], [#r.rust_rank / #r.madloop_rank], physical-cell(r), physical-cell(batch))
  },
)))

== Scope and uncertainty
The timers include numerical numerator coefficients, reduction, conversion,
contraction, status checks and returned-buffer/cache cleanup. Graph/model and
external-wavefunction preparation, input packing, final comparison adapters
and process transport are excluded from both. Bare poles and R1 are retained;
R2 and UV counterterm graphs are excluded. Routing and rank differences are
declared algorithmic choices: *no tensor-stage timing ratio is valid*.

Both engines use Hybrid, 53 working/requested bits, input preservation, CPU 0,
five warm-ups and 21 blocks targeting 30 ms. Separate engine runs use an
independent bootstrap ratio of medians (10,000 resamples). These intervals
describe within-run variation and do not bound drift between engines.
Project computations and producer writes were frozen; external host activity
was not controlled. Stage samples and raw full responses remain available.

Attempt002 preserves all source/build identities. Attempt001 is retained as
unqualified because producer inputs changed during execution. The optimized
top-box numerical pass is specific to this captured harness path; earlier
CLI binary64 guard rejections remain part of the ledger. Top-pentagon failure
is the actual preserved-input derived-distance conditioning diagnostic.
No failure is removed from the family count.

Sources, contracts, stage timings and reproduction:
#raw("validation/benchmark/physical-joint-quiet/"). The preceding three-block
ggH smoke and 80 Python boundary/provenance tests are separately retained in
#raw("validation/benchmark/physical-joint-smoke/").

== Precision follow-up on the failed pentagon point
Promoting the same 95 represented binary64 input components exactly to
Float256/512, with requested accuracy53 unchanged, gives convergence over
all 280 complex tensor slots within $2.17 times 10^(-55)$. Original PV/hybrid
agree within $4.62 times 10^(-14)$ and the preserved MadLoop graph within
$9.29 times 10^(-14)$. Original IBP still differs by 0.60013; binary64/128 guard
failures remain. The first 256-bit driver captured valid outputs before a
rational-input comparison-parser failure; the retained separate comparison
passes. This is one-point manual precision qualification, not automatic
rescue or a timing gate. See #raw("validation/physical/gghhh-guard/").

#pagebreak()
= Current implementation checkpoint
These experiments have separate source and binary identities. They extend
the correctness evidence; they do not replace the historical timing tables.

== Ordinary scalar-master storage
The default interpreter now reuses root temporaries through Symbolica's
existing allocator. All five families preserve every output, precision and
signed zero in 40 configurations each across f64, DoubleFloat and Float256.
Shared helpers and Dual preparation are unchanged.
#report-table(table(
  columns: (1.5fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  inset: (x: 6pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Root temporary slots], [A0], [B0], [dB0], [C0], [D0]),
  [Before], [38], [503], [1,951], [220], [25,010],
  [After], [6], [13], [24], [8], [25],
))
The default-stack workspace snapshot compiled with unchanged source hashes
before the generalized-PV promotion below. Its current-guard acceptance
passes 30 execution, 29 stability, six public mixed-precision and three
scalar integration tests.

The optimized #raw("80949db") snapshot also passes eight complete bubble/
triangle kernel checks at batches 1/32, maximum normalized error
$2.46 times 10^(-13)$. Three short blocks, zero warm-up and concurrent builds
make this a correctness smoke only; its recorded timing ratios are unqualified.

The paired DD E6 experiment preserves all three statuses and 12 complex
Laurent outputs exactly, with the independent $10^(-23)$ check passing in
both variants. Each uses order 128 and 3,720 D0 rows.
#report-table(table(
  columns: (1.4fr, 1fr, 1fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Diagnostic observation], [Before], [After]),
  [Whole validation duration], [46.682 s], [43.822 s],
  [Per-child peak resident memory], [2,052,724 KiB], [558,752 KiB],
))
*These are single-run diagnostics, with other compilation active; no timing
gate or speed ratio is inferred.* The paired experiment uses an earlier,
stricter Float guard; fixed-precision DD is unaffected. Current-guard tests
are separate. Sources: #raw("validation/stack-reuse/production-factory/").

== Generalized PV and unresolved precision
Generalized PV for $N<=4$ passes nine focused tests and 36 original-IREGI
comparisons across 312 complex slots; maximum normalized error is
$7.04 times 10^(-12)$ at the unchanged $10^(-10)$ gate. The promoted PV and
lazy-solve core also passes five supplemental behaviors: supplied products,
multinomial weights, complex scalar/tensor comparisons and Hybrid isolation.
The latter uses working128/256, requested53; its binary64 fixture failure is
retained. These checks use frozen former-default master archives; combined
default-stack integration is separate. Higher points retain IBP/E6. Sources:
#raw("validation/generalized-pv/") qualification, promotion and supplemental
manifests.

The exact small-routing E6 oracle exposes a remaining production failure:
#report-table(table(
  columns: (1fr, 1fr, 2fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Working bits], [Requested bits], [Public result]),
  [128], [128], [Unstable: nonfinite box output],
  [256], [128], [Unstable: nonconvergent quadrature],
))
Both runs finish within their bounds and expose no successful values.
No oracle comparison or requested-128-bit agreement is claimed. The frozen
binary predates default stack reuse. See
#raw("validation/shifted-pentagon/production-series-qualification.json").

#pagebreak()
= Validation coverage
#figure(kind: table, caption: [*Evidence ledger.* “Source audited” records code
  inspection; “verified” identifies executed checks described in that row.],
  report-table(table(
    columns: (1.1fr, 1.5fr, auto), inset: (x: 6pt, y: 5pt), stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([Capability], [Required evidence], [Status]),
    ..for row in data.coverage { ([*#row.feature*], [#row.requirement \ #text(size: 7pt, fill: luma(90), row.evidence)], status-cell(row.status)) },
  )),
)
#block(fill: pale, inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Snapshot boundary.* #data.validation_snapshot_note
  Four physical triangle and five massive/complex box cases pass
  six-component mixed derivative checks. Historical timeouts remain visible.
  The N5 rank-six public rescue verifies all 210 moments / 840 slots at
  128/256 bits, including an independent coefficient. Binary64 Rust and
  original IBP still exceed the fixed $10^(-10)$ criterion for that point.
  The frozen oracle passes 31 cases: 30 at binary64 and one generalized
  triangle through verified rescue. Broader domains remain pending.
]

== Known original-reference defect
#let issue = data.known_issues.first()
For shifts $(0,0,0,0)$ and $(0,1,0,0)$, squared masses $(4,9)$, $mu=3$ and
numerator $q_x^4$, original PV and hybrid report stability but disagree with
an independent Feynman-parameter integral. Their UV residues agree.
#report-table(table(
  columns: (1fr, 1.8fr), inset: (x: 6pt, y: 5pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Method], [Finite coefficient / $i$]),
  [Parameter integral], table.cell(fill: rgb("#d9f0d3"), [#issue.independent_finite_imaginary.slice(0, 25)…]),
  [Original IBP], table.cell(fill: rgb("#d9f0d3"), [#str(issue.original_ibp_finite_imaginary)]),
  [Original PV / hybrid], table.cell(fill: rgb("#fddbc7"), [#str(issue.original_pv_finite_imaginary)]),
))
#text(size: 8.5pt)[Evidence: #raw(issue.evidence).
  The original PV branch is retained unchanged; the relevant Rust comparisons
  use original IBP and the independent parameter-integral result.]

#pagebreak()
= Measurement protocol
Pin release builds to one CPU core and record thread counts, hardware, OS,
toolchain, flags, dependency/input hashes, seeds and warm-up. Identify cold
or warm caches. Retain raw repeated-block observations, median, dispersion
and an output checksum.

Preparation includes graph import, numerator construction and evaluator
compilation. Time numerator, reduction, masters and total independently;
summed instrumented stages are only diagnostic. Separate library costs
from subprocess/file I/O and match physical routing, modes and stage boundaries.

Use identical samples at batch widths 1, 4, 8, 16, 32 and 64, matching numerical
precision and including rescue costs. Retain failed lanes. Every speed gate
requires correctness: stable binary64 uses the fixed $10^(-10)$ mixed tolerance
on dimensionally normalized outputs. Gate singleton latency and batch-32
throughput per topology/rank/mass family, with IBP, PV and hybrid separate.
The target envelope includes seven propagators, rank seven, generalized
powers and dimension shifts.

The kernel harness now records every requested configuration before starting
subprocesses, preserving interrupted and unrun rows. The latest Python suite
passes 80 ledger, provenance, protocol and comparison checks. The optimized
joint ggH smoke passes at batches 1/3; its short zero-warm-up timings are
diagnostic. The subsequent quiet physical results are reported separately.

An identified debug binary passes the $g g arrow H$ harness smoke at batches
1 and 3: all 15 moments / four slots agree with three IREGI modes within
$3.55 times 10^(-15)$, and MadLoop within $8.02 times 10^(-15)$.
Seven-stage protocol checks pass; short debug timings, live-source provenance
and repeated-point scaling remain unqualified. See
#raw("validation/benchmark/physical-smoke/qualification.json").

With the accepted license, 360 batched calls across two workers match sequential
A0–D0 values, status and metadata for f64/Float128 and second-order Dual branches.
A separate four-worker cold-specialization test passes shared ownership and
cleanup checks. These qualify retained SymbolicaBatch snapshots; native kernels
remain separate. The earlier rank-one box rejection is retained in
#raw("validation/concurrent-workspaces/").

= Rebuilding this report
Update #raw("docs/data/results.json") only from retained measurements and
rebuild with #raw("bash scripts/build-benchmark-report.sh").
The report uses Libertinus Serif and DejaVu Sans/Mono. Supply their installation
directories through Typst's #raw("--font-path") option when needed.
Reference checkout revisions are retained in the JSON data. They record the
audited source trees and must not be mistaken for a verified dependency build.

== Validation reproduction and remaining checks
Run recorded commands with the pinned environment from #raw("scripts/env.sh").
Exact commands and source hashes accompany each JSON result. The frozen
31-case oracle includes one explicit 128/256-bit rescue; all 30 legacy
fixtures pass through 22 primary results and eight verified rescues. Three
active rescue, four bubble-Taylor, three scalar API and nineteen CLI smoke
checks pass. The new 68-test default-stack acceptance is a separate source
scope; earlier failures remain retained.

The final frozen master source passes 27 checks, including extreme-scale B0,
six-component scale-only jets in all families, mixed input precision and the
full physical D0 derivative batch; see
#raw("validation/scalar-masters/qualification.json"). The separate exact
factory-parity/default-stack qualification is described above. Generated
native kernels remain opt-in; no new timing gate passes.

An independent Symbolica Gamma expansion checks 200 tensor and 40 generalized
requests with analytic bubbles disabled/enabled: 480 evaluations and 1920
slots. Nine extracted FeynKit partial-fraction and two planar formula tests
pass; public integration remains pending.

N6 inline native-master 128/256-bit arrays converge over 1320 complex slots
within $6.46 times 10^(-29)$; public N6/N7 tensor arrays remain unqualified.
Public rank-seven baseline/current/n-way probes time out during preparation at
180/300/300 s, without values. Plans have 330 moments, 31,670 aliases and 56
masters. The private n-way expression change passes 20 exact checks but its
changed preparation stage is not reached; no speed gate follows. Source,
CPU and diagnostic durations are retained in #raw("validation/public-highpoint/").

#pagebreak()
= Physical fixture qualification
#report-table(table(
  columns: (1.3fr, 1.3fr, 1.3fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Single graph], [Pointwise numerator], [Integrated Rust result]),
  [Top triangle: $g g arrow H$], status-cell("verified"), status-cell("verified"),
  [Top box: $g g arrow H H$], status-cell("verified"), status-cell("verified"),
  [Top pentagon: $g g arrow H H H$], status-cell("verified"), table.cell(fill: pale)[*PV/hybrid + MadLoop*],
  [Massless up-quark box], status-cell("verified"), status-cell("verified"),
))

== Physical reference contract
Fixtures preserve parameters, actual HELAS polarizations, graph factors and
source hashes. The triangle uses two orientations with grouping disabled.
MadLoop shifts are $(0,p_1,p_1+p_2)$ with polynomial variable $x=-Q$;
its adapter includes the odd-rank sign. For the FeynKit routing,
$N_"FK"(q)=-1/2 P_("ML",1)(p_1-q)$ with matched couplings/polarizations.
Do not apply color or direction factors twice.

Five captures contain 40 graph records, including a second triangle helicity;
these are not 40 integrated validations. Thirteen preparation and four selected
physical numerator comparisons pass. Actual evaluated momenta are exported
because MadLoop may correct the requested point.

Fresh preserved-input Fortran replay matches all 301 numerator coefficients,
903 tensor Laurent entries and 12 contracted slots across four graphs exactly;
eight alignment boundary tests pass. Legacy-default ggH is unchanged. This is
reference-only qualification; later Rust checks belong to the separate joint run.
Equivalent full-graph timing retains Rust/MadLoop ranks 2/3 for ggH, 4/4 for
both boxes and 4/5 for ggHHH, and their declared routing. It does not qualify
tensor-stage ratios. Sources: #raw("validation/benchmark/physical-alignment/").

The recorded binary64 $g g arrow H$ CLI passes all 15 moments / 60 slots
against original IREGI within $1.86 times 10^(-15)$ normalized error and
MadLoop within $9.38 times 10^(-15)$, including poles. Values and binary identity
remain in #raw("validation/physical/ggh-cli-comparison.json"); this is not a
fresh replay. Nineteen CLI smoke checks pass. Binary64 comparison arithmetic
certifies no extra digits from higher working precision; short unpinned
timings are excluded.

The historical top-box binary64 CLI rejects: PV fails its cancellation guard and IBP
fallback is nonfinite. Float128/256 reruns pass all 280 slots against all
three reference modes within $1.15 times 10^(-13)$, and MadLoop within
$4.77 times 10^(-15)$. Each freshly prepares/evaluates the numerator. Their
normalized tensor difference is $6.09 times 10^(-37)$ under binary64 comparison
arithmetic. See #raw("validation/physical/gghh-cli-precision-convergence.json").
The Symbolica-backed model adapter passes four boundary, thirteen preparation
and four numerator tests while preserving other numerical tokens.

The current optimized CLI also passes the massless up-quark box. All 70
moments / 280 slots agree with each original mode: maximum error
$2.85 times 10^(-14)$ against IBP and $7.11 times 10^(-15)$ against PV/hybrid.
The isolated MadLoop finite/single/double coefficients agree within
$6.74 times 10^(-15)$, with fixed-color factor $-1/8$. Evidence is retained in
#raw("validation/physical/gggg-up-cli-comparison.json"); its one-run wall time
is not a qualified performance measurement.

Complete top-pentagon CLI reruns at Float 128/256 now agree with original
PV/hybrid for all 280 slots within $4.63 times 10^(-14)$, and with MadLoop
within $9.30 times 10^(-14)$. Between-run tensor difference is
$7.79 times 10^(-17)$ under binary64 comparison arithmetic. Original IBP
still differs by $0.60014$, so the all-mode report remains failed. See
#raw("validation/physical/gghhh-cli-precision-convergence.json") and linked
reports. Earlier PV/hybrid replay of 378 captured MadLoop slots was exact.

== Physical derivative qualification
Four triangle cases at 384 bits cover real subthreshold/absorptive and
complex-mass kinematics, each with six first, second and mixed Dual components.
Finite differences converge with maximum final scaled error
$5.47 times 10^(-28)$. One equal-complex D0 case also passes, with error
$1.11 times 10^(-34)$. The complete five-case D0 batch now passes at 384 bits,
maximum final scaled error $2.53 times 10^(-30)$ under the unchanged $2^(-70)$
criterion. Its 224.83 s validation duration is not a benchmark; earlier
180 s timeouts remain retained. See #raw("validation/scalar-masters/physical_box_full.json").
Exact commands and scope are retained in
#raw("crates/iregi-masters/tests/validation_evidence.json").

#pagebreak()
= Open qualification ledger
#report-table(table(
  columns: (1.1fr, 2.4fr), inset: (x: 7pt, y: 4pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Unresolved item], [Observed evidence and remaining check]),
  [Binary64 baseline accuracy], [Historical Rust error $2.23 times 10^(-10)$
    and original IBP error $3.40 times 10^(-10)$ fail the unchanged gate.
    Public 128/256-bit rescue now passes at the retained rank-six point.],
  [Broader pentagon rescue], [Rank six passes 840 slots; the saved vendored
    baseline passes 1320 rank-seven slots and one independent coefficient.
    Later precision/routing edits and broader domains remain unqualified.],
  [Physical pentagon reference], [Complete Float 128/256 graph evaluations
    pass PV/hybrid and MadLoop. Original IBP still disagrees; all-mode reports
    remain failed. Binary64 and broader kinematics remain unqualified.],
  [Shifted scalar domain], [Thirteen standalone checks and public finite
    fixtures pass. The paired DD stack candidate also passes exactly.
    Small-routing E6 at working128/256 and requested128 returns Unstable;
    broader precision/derivative domains and native E6 remain unqualified.],
  [Broader derivative domains], [The full five-case mixed D0 batch now passes
    with error below $2^(-70)$. True thresholds and unsupported transverse
    massless/Gram/Cayley domains retain typed errors; old timeouts remain.],
  [Physical top-box binary64], [The historical CLI guard rejection remains
    visible. Complete Float 128/256 evaluations pass; the retained 80949db
    physical harness also passes binary64. Broader kinematics remain pending.],
  [Supplied-product high points], [Certified N6 scalar: working128/256,
    requested53, zero poles and invalid-lane isolation pass. Both binary64
    variants reject at the same guard. Independent quadrature differs by
    $1.53 times 10^(-23)$, below estimated uncertainty $6.11 times 10^(-22)$
    (not rigorous). N7/tensor/mixed-Dual coverage remains pending.],
  [Legacy snapping scope], [All 30 retained cases pass: 22 primary plus eight
    verified 128/256-bit rescues. Earlier failures are preserved; generalized
    requests and broader domains remain unqualified.],
  [Mixed-precision accuracy], [Three retained false-success cases are repaired:
    29 stability and six public tests pass. The active-master guard remains
    conservative and its estimates noncertified; isolated overhead diagnostics
    establish no optimized IREGI gate.],
  [Runtime performance], [Interpreter: 7 regressions/1 improvement. Native:
    5 regressions/1 parity/2 improvements, in a separate paired session.
    5c9e3dd: 6 regressions/2 noisy rows; latest80949db: 5 regressions/3 noisy.
    Physical: six noisy rows/two failures. No overall performance gate passes.],
))

== Optimization and experimental evidence
#report-table(table(
  columns: (1fr, 3fr), inset: (x: 6pt, y: 5pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Candidate], [Executed evidence and limit]),
  [Core stack reuse], [76 test executions and 12 paired cases pass; a quiet
    opt2 run preserves 92 outputs. Batch ratios 0.9873/0.9802; singleton intervals
    include parity. Promoted #raw("d1a4b8e") passes 67 current opt2 checks;
    its optimized IREGI gate remains unmeasured.],
  [Guard context], [62/76 isolated checks and 120 snapshots pass; six measured
    improvements, six noisy rows, no regression. Rebased source passes 76 checks
    and 120 snapshots; final cleanup passes 14. Adopted in #raw("6b24c03").
    Its optimized IREGI gate remains unmeasured.],
  [Roundoff cache], [29/37 stability tests and 120 snapshots pass. Measured
    candidate rejected: two regressions, ten noisy rows, no qualified gain.
    Separate accuracy-guard diagnostic: two regressions/six noisy rows.],
  [Certified planar], [Private Float128/request53 rank7 passes 1320 complex slots
    against five boxes at dimensionally normalized $10^(-10)$. Rank≤2 normalized
    precision/Dual checks pass. Historical failures/timeouts remain; no production fallback.],
  [Near-vacuum box], [18 validation executions / 15 settings pass, including
    mixed jets and unequal masses. Requested bits: f64 40, DD 80, Float128 100,
    Float256 128. Rejections/timeouts remain; production D0/E6 is unchanged.],
))
#text(size: 8.5pt)[Artifact paths and full candidate scopes are retained in
#raw("docs/data/results.json"). Planar rank7 Float256/Dual and public N6
remain unqualified. Separate allocator, native and guard effects are not multiplied.]
#pagebreak()
= Independent pentagon adjudication
These checks establish selected mathematical values independently of an
original-reference stability flag. They do not qualify every pentagon tensor,
physical graph, runtime domain or numerical precision.

== Rank-six Euclidean tensor
For squared masses $(4,9,16,25,36)$, $mu=3$ and numerator $q_0^5 q_3$,
direct simplex integration gives native finite coefficient
$-0.002438378935447123 i$. Independent coordinate mappings agree within
$4.34 times 10^(-19)$; reported quadrature error is at most
$2.77 times 10^(-15)$. The bound $F>=3.75$, angular integration and power
counting exclude poles. This uses no reduction/master library; numerical
error estimates are not interval certificates. See
#raw("validation/pv-high-rank/pentagon-simplex-oracle.json").
Historical binary64 errors remain $2.23 times 10^(-10)$ for Rust and
$3.40 times 10^(-10)$ for IREGI. Public rescue flags the primary unstable and
verifies all 840 slots at 128/256 bits plus this coefficient below $10^(-13)$.
#raw("validation/rescue-pentagon-rank6.json") retains the binary and assertions;
full arrays were not printed and its exact build-source snapshot was not saved.

The coherent vendored Symbolica baseline passes four internal scope tests,
57 focused core tests and all 1320 rank-seven slots at 128/256 bits, including
the independent coefficient. Its saved source predates later
precision/routing edits. Current workspace compilation passes with all 329
source/manifest hashes unchanged; later runtime checks remain separate. The
109.44 s cold run is diagnostic, not a performance benchmark. See #raw("validation/symbolica-production-qualification.json").
The earlier 240 s timeout and 115.35 s isolated-patch run remain historical.

== Physical canonical moment 54
For $g g arrow H H H$, rank-four moment $q_0 q_3^3$ has UV degree $-2$;
its UV pole must vanish. Original IBP instead returns approximately
$3.33691766067 times 10^(-6) i$ in native normalization. The 128/256-bit
PV evaluation suppresses its OneLoop-normalized UV residual from
$7.9 times 10^(-31)$ to $2.3 times 10^(-69)$, with both IR slots zero.
Its converged OneLoop finite coefficient is approximately
$3.3474440772594743330 times 10^(-13)
 + 2.9254797223941252086 times 10^(-13) i$.
This supports original PV/hybrid and rejects the original IBP result for this
component. It is not a full physical tensor comparison. Raw precision evidence
is in #raw("validation/pv-high-rank/canonical-pentagon-precision.json");
#raw("canonical-pentagon-adjudication.json") records fixed-gate comparisons.

== Standalone six-dimensional scalar
Use $J_1=I_6/(i pi^3)$, shifts $(0,e_0,e_1,e_2,e_3)$ and squared masses
$(10,12,15,18,22)$. Here $F>=39/4$, $det R=2600$ and $det G=-16$.
#report-table(table(
  columns: (1.3fr, 2fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Method], [Finite $J_1$]),
  [Independent simplex], table.cell(fill: rgb("#d9f0d3"), [$-0.0001771401382640081806076521310$]),
  [Original scalar/general APIs], table.cell(fill: rgb("#fddbc7"), [$-0.00014953025024128455$; every mode reports stable]),
))
Increasing quadrature order and precision through 70 digits converges to the
first value. Independent integration of ordinary D4 values over a common mass
shift, with a separately integrated tail, agrees within $4.93 times 10^(-15)$.
A box-only half-line implementation also agrees to approximately
$8.42 times 10^(-17)$ or better in the recorded checks. Ordinary D4 and
coalesced D4/D6 normalization checks pass. Evidence and reproduction source
are in #raw("validation/shifted-pentagon/evidence.json"). Thirteen consolidated
D6 tests cover eight genuine-master checks and five endpoints. Retained public
f64 fixtures pass finite values, poles, scaling, rejected endpoints and lane
isolation; the earlier quadrature-budget failure remains historical.
Pre-vendor public DoubleFloat and Float128 tests now pass three scaled lanes
at requested 80-bit accuracy and $10^(-23)$ normalized oracle tolerance.
First common-mass Dual primal/derivative checks pass at $2 times 10^(-14)$,
with zero poles. The DD/Float validation durations total 165.69 s, exceeding
the historical 150 s combined timeout; this is no performance benchmark.
Build identities and limits are in
#raw("validation/shifted-pentagon/public-core-qualification.json"). A separate
vendored paired DD E6 test passes exact stack-allocation parity. Full requested
128-bit accuracy fails for the new small-routing fixture; higher/mixed Dual,
broader endpoints and native E6 remain unqualified. The original complex-box reference discrepancy
is documented in #raw("validation/shifted-pentagon/contours.json").


#pagebreak()
#include "conventions.typ"
#pagebreak()
#include "algebra-reuse.typ"
