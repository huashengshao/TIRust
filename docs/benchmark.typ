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
  *Integrated top triangle verified.* Graph-isolated
  top-loop $g g arrow H$ records are available for helicities 1 and 4, with
  35 numerator coefficients and 105 tensor Laurent entries per graph.
  All four selected FeynKit numerators agree: top triangle, box and pentagon,
  plus a massless up-quark box, at the recorded real/complex probes.
  R2 and UV counterterms are excluded; bare UV/IR poles and R1 remain.
  The complete $g g arrow H$ path agrees: 60 tensor slots have maximum
  normalized error $1.86 times 10^(-15)$; the contracted finite coefficient
  differs by $9.38 times 10^(-15)$ in the recorded executable. Current CLI smoke
  checks pass; other integrated graphs and timings remain pending.
]

#pagebreak()
= Validation coverage
#figure(kind: table, caption: [*Evidence ledger.* “Source audited” records code
  inspection; “verified” identifies executed checks described in that row.],
  report-table(table(
    columns: (1.1fr, 1.5fr, auto), inset: (x: 6pt, y: 6pt), stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([Capability], [Required evidence], [Status]),
    ..for row in data.coverage { ([*#row.feature*], [#row.requirement \ #text(size: 7pt, fill: luma(90), row.evidence)], status-cell(row.status)) },
  )),
)
#block(fill: pale, inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Snapshot boundary.* #data.validation_snapshot_note
  Master precision checks include binary64, DoubleFloat and Float at 128, 256
  and 512 bits. Verified jets include static/dynamic second-order tadpoles and
  Euclidean massive triangle/box scaling derivatives. General physical
  first, second and mixed derivatives remain under audit.
  *Open precision gate:* for the five-propagator rank-six case, 128- and
  256-bit evaluation converge. Binary64 Rust and original IBP differ from that
  value by $2.23 times 10^(-10)$ and $3.40 times 10^(-10)$ respectively.
  Both exceed the fixed $10^(-10)$ criterion and need validated precision rescue.
  The frozen oracle passes all 31 cases: 30 at binary64 and one generalized
  triangle through explicit 128/256-bit rescue. The five-point rank-six
  rescue attempt timed out after 180 seconds and remains unqualified.
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

== Validation reproduction and remaining checks
The standalone checks reported here run with the pinned dependencies prepared
by #raw("scripts/env.sh"). The master suite uses
#raw("RUST_MIN_STACK=67108864 target/iregi-masters-tests --test-threads=1 --nocapture");
the IBP suite uses #raw("target/ibp-unit --test-threads=1 --nocapture").
The current checkpoint passes ten master, twenty-one IBP, eight PV exact,
fourteen execution, eight contraction, ten stability and six CLI unit tests.
Twelve current CLI smoke checks pass, including model import and the reference
comparison command. The frozen oracle passes all 31 cases:
30 at binary64 and #raw("C0_power2_1_1_rank2") through explicit batched
128/256-bit precision rescue with unchanged tolerance. This is not 31
binary64 successes. Three rescue tests cover exact promotion, generalized
triangle coefficients, second-order Dual values and invalid-lane isolation.
The full five-point rank-six rescue test timed out after 180 seconds;
its numerical qualification remains open. Four exact bubble-Taylor tests
also pass; see #raw("crates/iregi-masters/src/bubble_neighborhood.rs").

Generated instruction-major Rust master kernels remain opt-in through
#raw("native-batch"). Scalar parity passes for all five families with
binary64 batches of 0, 1, 3, 5 and 31 lanes, including an invalid lane;
DoubleFloat/Float parity passes for A0–D0. A grouping regression preserves
zero-primal jets and genuine conditional branches. These checks do not
qualify physical mixed D0 derivatives or establish runtime speed.

An independent Dirichlet-simplex Gamma formula, expanded by Symbolica,
matches 200 tensor-coefficient scalar requests through rank seven and 40
generalized power/dimension requests in all four native Laurent slots.
Run #raw("cargo test -p iregi-rust --test triangle_gamma -- --test-threads=1").
This is exact algebra for the massless triangle with one off-shell leg;
it is not a broad numerical or performance result.

PV numerical checks compare all tensor moments and native pole slots using
original scalar-master leaf values. The strict, dimensionally normalized
comparison passes all 330 moments through rank seven for one to four
propagators, including both complex components. The first open failure is
$N=5$, rank six, numerator $q_0^5 q_3$. Both binary64 codes exceed the
unchanged criterion against the converged 128/256-bit result; the numerical
values and rescue timeout remain visible in the open qualification ledger.
Evidence and unqualified higher-point cases remain in
#raw("validation/pv-high-rank/rust-coverage.json"). A genuinely singular
six-propagator point is an explicit error, not a successful reduction.
The scalar IBP identities include selected singular triangles, coincident
denominators and lightlike bubbles. A standalone six-dimensional pentagon
needs information beyond the finite orders of the A0–D0 basis and must produce
an explicit unsupported-domain diagnostic when that information is unavailable.

#pagebreak()
= Physical fixture qualification
#report-table(table(
  columns: (1.3fr, 1.3fr, 1.3fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Single graph], [Pointwise numerator], [Integrated Rust result]),
  [Top triangle: $g g arrow H$], status-cell("verified"), status-cell("verified"),
  [Top box: $g g arrow H H$], status-cell("verified"), status-cell("pending"),
  [Top pentagon: $g g arrow H H H$], status-cell("verified"), status-cell("pending"),
  [Massless up-quark box], status-cell("verified"), status-cell("pending"),
))

== Physical reference contract
The export selects two oriented top triangles with loop grouping disabled.
Fixtures preserve model parameters, actual HELAS polarizations, graph factors,
source hashes and numerical probes. Original MadLoop uses denominator shifts
$(0,p_1,p_1+p_2)$ with a polynomial in $x=-Q$; its convention adapter already
includes the odd-rank sign. With the supplied FeynKit routing, numerator
comparison is $N_"FK"(q)=-1/2 P_("ML",1)(p_1-q)$ after matching couplings and
polarizations. Do not apply the fixed-color or direction factor twice.

The committed exports are
#raw("validation/physical/madloop/fixtures/ggh-helicity-1.json") and
#raw("ggh-helicity-4.json"). The compact numerator reference is
#raw("validation/physical/ggh-numerator-reference.json").
All thirteen preparation tests and all four selected physical numerator
comparisons pass. The five schema-2 capture files contain 40 graph records,
including the second triangle helicity. This is four selected-graph numerator
checks, not 40 integrated validations. The pentagon discrepancy was resolved
by exporting MadLoop's actual evaluated external momenta; requested driver
momenta can differ after phase-space correction. The corrected massless-up
binding also passes. Captured numerical results were unchanged by the metadata
correction.

The complete $g g arrow H$ CLI path passes at the recorded binary64 point.
All 15 tensor moments / 60 Laurent slots agree with original IREGI at the
FeynKit canonical routing to $1.86 times 10^(-15)$ normalized error. The
fixed-color raw finite coefficient is approximately $-147.86843349622455$;
its MadLoop comparison error is $9.38 times 10^(-15)$, with matching single
and double poles. The normalized finite result is approximately
$-0.9363877940734862$. Values, small imaginary residuals, hashes and reproduction
commands are retained in #raw("validation/physical/ggh-cli-comparison.json").
The other three integrated graph comparisons remain pending. These references
are binary64; higher working precision alone cannot certify additional digits.
Short, unpinned timing smoke runs are excluded from benchmark evidence.
The recorded ggH result retains its exact tested executable hash. The current
CLI smoke suite passes twelve checks; the physical result is not a fresh replay.
A model-import regression was resolved by a narrow Symbolica-backed quantum-
number adapter. Four boundary, thirteen preparation and four physical numerator
tests pass with the fix; other numerical tokens retain their precision.

#pagebreak()
= Open qualification ledger
#report-table(table(
  columns: (1.1fr, 2.4fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Unresolved item], [Observed evidence and remaining check]),
  [Five-point rank-six accuracy], [Rust binary64 error $2.23 times 10^(-10)$;
    original IBP error $3.40 times 10^(-10)$ versus the converged 128/256-bit
    value. Both fail the unchanged $10^(-10)$ gate.],
  [Full pentagon rescue], [Three active rescue tests pass, including the
    generalized triangle. The full rank-six pentagon regression timed out
    after 180 seconds; its success is unqualified.],
  [Physical pentagon reference], [Original IBP and PV/hybrid disagree by
    $0.6001338887$ normalized error at rank four, although every mode reports
    stability. The correct result has not been adjudicated.],
  [Physical mixed D0 derivatives], [The full mixed-derivative suite remains
    pending. Euclidean scaling and scalar batch parity are narrower checks.],
  [Runtime performance], [No qualified kernel timing, speedup or parity result
    has been measured. Short unpinned timing smoke runs remain excluded.],
))

== Canonical physical reference audit
At the FeynKit canonical input for $g g arrow H H H$, original IBP and
PV/hybrid disagree in a finite rank-four tensor coefficient. The largest
normalized difference is $0.6001338886545732$ at moment index 54; all modes
claim stability. This reference-only audit does not identify which result is
correct and does not run Rust reduction. The other three physical families
agree across original modes within $2.22 times 10^(-14)$. Inputs, hashes and
all four Laurent-slot comparisons are retained in
#raw("validation/physical/canonical-reference-audit.json"); reproduce with
#raw("validation/physical/audit_reference.py").

The retained #raw("validation/pv-high-rank/reference-coverage.json") also records
rank-seven original PV instability for five to seven propagators and a falsely
stable four-propagator disagreement. These reference failures remain visible;
stable IBP reference values do not themselves certify Rust high-rank coverage.

== Independent integral evidence
The rank-four bubble check is reproducible with
#raw("python3 validation/pv-high-rank/parameter_check.py --write").
Its 50- and 80-digit quadratures agree with original IBP. The suspected
quadratic-root parenthesization error is a source diagnosis; the numerical
discrepancy is established independently of that diagnosis.
Independent lightlike-bubble quadratures at 60 and 100 digits, including
first invariant derivatives, are retained in
#raw("validation/pv-high-rank/lightlike-feynman-parameters.json").
The five-propagator precision probe is retained in
#raw("validation/pv-high-rank/precision-probe.json"); its source and reproduction
command are recorded with the complete input and 128/256-bit results.
Correctly rounded master inputs reduce the observed finite-component error
to $3.11 times 10^(-12)$ with 256-bit reduction and $2.23 times 10^(-11)$
with binary64 reduction. This isolates amplified scalar-master error; it
does not qualify the original failing binary64 route.

#pagebreak()
#include "conventions.typ"
#pagebreak()
#include "algebra-reuse.typ"
