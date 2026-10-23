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
#let status-cell(status) = table.cell(fill: if status == "source-audited" { pale } else { amber },
  text(size: 8pt, weight: "semibold", if status == "source-audited" { "Source audited" } else { "Pending" }))
#let report-table(body) = {
  set text(font: "DejaVu Sans", size: 8pt)
  set par(justify: false)
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  body
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
  point. Gray cells are unmeasured and amber cells are pending verification.
  Source inspection does not establish numerical agreement. #data.measurement_note
]
#v(0.4em)
#block(stroke: 0.6pt + luma(200), inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Acceptance gates*
  #table(columns: (1fr, 2fr, auto), stroke: none, inset: (x: 4pt, y: 4pt),
    [*Correctness*], [Agreement with original IREGI and isolated MadLoop graphs], [*Pending*],
    [*Pure reduction*], [Equal or better performance on identical inputs], [*Unmeasured*],
    [*Complete evaluation*], [Report preparation, numerator, reduction and masters separately], [*Unmeasured*],
  )
]
= Identical-input comparisons
#figure(kind: table, caption: [*Synthetic inputs versus original IREGI.*
  Each measured comparison must share routing, masses, scale, rank, numerator
  coefficients and reduction mode. The current cells contain no measurements.],
  report-table(table(
    columns: (1.5fr, auto, 1fr, 1fr, 1fr),
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    inset: (x: 5pt, y: 9pt), stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([Case], [$n$ / $r$], [Reduction], [Masters], [Total]),
    ..for c in data.cases.filter(c => c.id != "physical-diagram") {
      (table.cell([*#c.label* \ #text(size: 7pt, fill: luma(90), c.description)]),
       [#c.n / #c.rank],
       cmp-cell(component(c.rust_us, "reduction"), component(c.iregi_us, "reduction")),
       cmp-cell(component(c.rust_us, "masters"), component(c.iregi_us, "masters"), reference: "OneLOop"),
       cmp-cell(component(c.rust_us, "total"), component(c.iregi_us, "total")))
    },
  )),
)
#align(center)[
  #set text(8pt)
  #box(width: 9cm, height: 0.35cm, fill: palette, radius: 2pt)
  #linebreak()
  #box(width: 9cm)[×1/4 #h(1fr) ×1/2 #h(1fr) ×1 #h(1fr) ×2 #h(1fr) ×4]
]
= Physical diagram comparisons
#block(fill: amber, inset: 10pt, radius: 4pt, width: 100%)[
  *Pending.* Start with top-loop $g g arrow H$ triangles and $g g arrow H H$
  boxes. FeynKit-generated physical numerators and graph-isolated MadLoop
  comparisons have not yet been measured. R2 and UV counterterms are excluded;
  bare UV/IR poles and R1 remain. A physical graph comparison must identify
  its model, routing, normalization, helicity and color configuration.
]

#pagebreak()
= Validation coverage
#figure(kind: table, caption: [*Evidence ledger.* “Source audited” records code
  inspection only. Every other capability still requires executable evidence.],
  report-table(table(
    columns: (1.1fr, 1.5fr, auto), inset: (x: 6pt, y: 8pt), stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([Capability], [Required evidence], [Status]),
    ..for row in data.coverage { ([*#row.feature*], [#row.requirement \ #text(size: 7pt, fill: luma(90), row.evidence)], status-cell(row.status)) },
  )),
)
= Measurement protocol
Run release builds on one pinned CPU core with fixed thread counts and record
hardware, operating system, toolchain, compile flags, dependency revisions,
input hashes, seeds, warm-up, sample count and raw observations. Every result
must identify whether caches are cold or warm. Report median and dispersion
over repeated sufficiently long timed blocks; retain an output checksum.

Preparation comprises graph import, rules, tensor identities, coefficient
extraction and evaluator construction. Time numerator evaluation, pure
reduction, scalar masters and complete evaluation independently. Measure the
complete total directly: adding separately instrumented components is only a
diagnostic. Distinguish reference-library calls from subprocess and file I/O.

Use the same samples at batch widths 1, 4, 8, 16, 32 and 64 and report time per
point. For precision comparisons, match the numerical precision and include
rescue costs. Record failed lanes; do not omit unfavorable points silently.
Absolute/scaled coefficient errors, high-precision adjudication and derivative
checks accompany performance results. No speed gate passes without correctness.
The stable binary64 target is a mixed absolute/relative tolerance of $10^(-10)$
on dimensionally normalized outputs. Gate singleton latency and batch-32
throughput separately for each topology/rank/mass family. Record IBP, PaVe and
hybrid mode independently. The eventual reference envelope is seven
propagators and rank seven, including generalized powers and dimension shifts.

= Rebuilding this report
Update #raw("docs/data/results.json") only from retained measurements and
rebuild with #raw("typst compile docs/benchmark.typ docs/benchmark.pdf").
The report uses Libertinus Serif and DejaVu Sans/Mono. Supply their installation
directories through Typst's #raw("--font-path") option when needed.
Reference checkout revisions are retained in the JSON data. They record the
audited source trees and must not be mistaken for a verified dependency build.

#pagebreak()
#include "conventions.typ"
#pagebreak()
#include "algebra-reuse.typ"
