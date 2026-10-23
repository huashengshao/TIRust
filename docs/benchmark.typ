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
#let benchmark-cell(r) = table.cell(fill: ratio-fill(r.paired_ratio), {
  set par(leading: 0.25em, justify: false)
  text(size: 9pt, weight: "bold")[#fixed(r.rust_us_per_point, digits: 3) µs]
  linebreak()
  text(size: 7pt)[IREGI #fixed(r.iregi_us_per_point, digits: 3) µs · ×#fixed(r.paired_ratio, digits: 3)]
  linebreak()
  text(size: 6.5pt)[99% CI #fixed(r.paired_ratio_ci99.at(0), digits: 3)–#fixed(r.paired_ratio_ci99.at(1), digits: 3)]
})
#let stage-colors = (rgb("#4c78a8"), rgb("#59a14f"), rgb("#f28e2b"), rgb("#b8b8b8"))
#let stage-bar(r) = {
  let v = r.profile_us_per_point
  let vals = (v.arguments, v.masters, v.reduction, v.other)
  box(width: 10.5cm, height: 12pt, fill: rgb("#f3f4f6"))[
    #stack(dir: ltr, spacing: 0pt, ..vals.enumerate().map(((i, val)) =>
      rect(width: 10.5cm * val / 40, height: 12pt, fill: stage-colors.at(i), stroke: none)))
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
  point. Gray cells are unmeasured and amber cells are pending verification.
  Source inspection does not establish numerical agreement. #data.measurement_note
]
#v(0.4em)
#block(stroke: 0.6pt + luma(200), inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Acceptance gates*
  #table(columns: (1fr, 2fr, auto), stroke: none, inset: (x: 4pt, y: 4pt),
    [*Correctness*], [Agreement with original IREGI and isolated MadLoop graphs], [*Pending*],
    [*Kernel performance*], [Equal or better on identical inputs; historical baseline], [*Unmet in both runs*],
    [*Complete evaluation*], [Report preparation, numerator, reduction and masters separately], [*Unmeasured*],
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
current-tree, higher-rank and physical performance remain unmeasured.]

#text(size: 9pt)[Newer validation includes four integrated graphs. The top
pentagon at Float128/256 agrees with PV/hybrid and MadLoop; original IBP still
fails. Physical timings are unmeasured.]

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
  Master precision checks include binary64, DoubleFloat and Float at 128, 256
  and 512 bits. Verified jets include static/dynamic second-order tadpoles and
  Euclidean massive triangle/box scaling derivatives. Four physical triangle
  cases and all five massive/complex box cases pass six-component mixed
  derivative checks. Earlier 180-second box timeouts remain historical evidence.
  *Verified precision rescue:* for the five-propagator rank-six case, 128- and
  256-bit evaluation converge. Binary64 Rust and original IBP differ from that
  value by $2.23 times 10^(-10)$ and $3.40 times 10^(-10)$ respectively.
  Both exceed the fixed $10^(-10)$ criterion. The public rescue now verifies
  all 210 moments / 840 slots, with an independent coefficient assertion.
  The frozen oracle passes all 31 cases: 30 at binary64 and one generalized
  triangle through explicit 128/256-bit rescue. Broader high-point ranks and
  physical domains remain pending.
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
Thirty execution and five argument-guard checks pass, including mixed
Float80/256, underflow, false/nonzero distance cancellation and higher Taylor
components. All nineteen CLI smoke checks pass. The frozen oracle passes
31 cases: 30 at binary64 and #raw("C0_power2_1_1_rank2")
through explicit batched 128/256-bit rescue with unchanged tolerance.
Three active rescue tests and the complete public rank-six pentagon rescue
pass. The primary binary64 lane is correctly flagged unstable; 840 coefficients
converge at 128/256 bits and the independent coefficient assertion passes. Four exact bubble-Taylor tests and three supplied-product
scalar API tests pass. All 30 retained legacy-snapping cases now pass:
22 at primary precision and eight through explicit 128/256-bit rescue.
Earlier failures remain recorded; broader/generalized policy cases are unqualified.

The final frozen master source passes 27 checks, recorded with logs in
#raw("validation/scalar-masters/qualification.json"). Independent B0 checks
cover kinematic scales $2^(-1070)$ and $2^(1000)$ at $mu^2=1$ for f64,
DoubleFloat and Float256. Six-component scale-only jets pass for all five
master families. Mixed input/seed precision, physical B0/C0 derivatives and
subnormal normalization pass, including the complete mixed D0 fixture batch.
Native kernels remain opt-in, with the earlier all-family and tail parity checks.

An independent Symbolica Gamma expansion checks 200 tensor and 40 generalized
requests with analytic bubbles both disabled and enabled: 480 requests and
1920 native slots for the massless one-off-shell triangle. Nine extracted
FeynKit partial-fraction tests pass; planner integration is pending. Two planar
formula tests pass, but public activation awaits exact rank certification.
These checks establish no broader numerical or performance result.

Exact PV coefficients with original Fortran scalar leaves pass the N1–4
corpus: 330 moments through rank seven, all native slots. N6 binary64 with
those leaves has 430 discrepancies. Separately, inline native Rust masters
at 128/256 bits converge for all 1320 complex slots within
$6.46 times 10^(-29)$; original IBP has 566 failures against that result, and
retained binary64 values also fail. This is internal consistency, not public
workspace qualification. Public N6/N7 and routing variants remain pending;
see #raw("validation/pv-high-rank/rust-coverage.json"). N5 rank-six public
rescue is qualified separately. An experimental Symbolica patch now verifies
all 1320 N5 rank-seven slots; production integration remains pending. Earlier
N5 240 s and inline N7 90 s preparation timeouts remain in #raw("data/results.json").

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

Five schema-2 captures under #raw("validation/physical/madloop/fixtures/")
contain 40 graph records, including a second triangle helicity. All thirteen
preparation tests and four selected physical numerator comparisons pass;
this is not 40 integrated validations. Actual evaluated momenta are exported
because MadLoop can correct the requested phase-space point. The compact
triangle reference is #raw("validation/physical/ggh-numerator-reference.json").

The recorded binary64 $g g arrow H$ CLI passes all 15 moments / 60 slots
against original IREGI within $1.86 times 10^(-15)$ normalized error and
MadLoop within $9.38 times 10^(-15)$, including poles. Its fixed-color raw
finite value is $-147.86843349622455$, normalized to $-0.9363877940734862$.
Values and executable provenance are retained in
#raw("validation/physical/ggh-cli-comparison.json"); this is not a fresh replay.
All nineteen current CLI smoke checks pass. Reference data and comparison
arithmetic are binary64; higher working precision certifies no extra digits.
Short unpinned timing checks are excluded from benchmark evidence.
The top-box binary64 rejection remains recorded: PV fails its cancellation
guard and IBP fallback is nonfinite. Complete Float128 and Float256 CLI reruns
now pass all 280 tensor slots against all three reference modes, within
$1.15 times 10^(-13)$; MadLoop contraction error is at most
$4.77 times 10^(-15)$. Each run freshly prepares/evaluates the numerator.
Their normalized tensor difference is $6.09 times 10^(-37)$, with identical
contracted values after binary64 conversion. That comparison arithmetic
certifies no extra digits. Evidence is retained in
#raw("validation/physical/gghh-cli-precision-convergence.json") and its
linked Float128/256 reports.
A model-import regression was resolved by a narrow Symbolica-backed quantum-
number adapter. Four boundary, thirteen preparation and four physical numerator
tests pass with the fix; other numerical tokens retain their precision.

The current optimized CLI also passes the massless up-quark box. All 70
moments / 280 slots agree with each original mode: maximum error
$2.85 times 10^(-14)$ against IBP and $7.11 times 10^(-15)$ against PV/hybrid.
The isolated MadLoop finite/single/double coefficients agree within
$6.74 times 10^(-15)$, with fixed-color factor $-1/8$. Evidence is retained in
#raw("validation/physical/gggg-up-cli-comparison.json"); its one-run wall time
is not a qualified performance measurement.

Complete top-pentagon CLI reruns at Float128/256 now agree with original
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
  columns: (1.1fr, 2.4fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Unresolved item], [Observed evidence and remaining check]),
  [Binary64 baseline accuracy], [Historical Rust error $2.23 times 10^(-10)$
    and original IBP error $3.40 times 10^(-10)$ fail the unchanged gate.
    Public128/256-bit rescue now passes at the retained rank-six point.],
  [Broader pentagon rescue], [Rank six passes 840 slots. An isolated Symbolica
    patch verifies all1320 rank-seven slots at128/256 bits and one independent
    coefficient. Production integration and broader domains remain pending.],
  [Physical pentagon reference], [Complete Float128/256 graph evaluations
    pass PV/hybrid and MadLoop. Original IBP still disagrees; all-mode reports
    remain failed. Binary64 and broader kinematics remain unqualified.],
  [Shifted scalar domain], [Thirteen consolidated master/endpoint checks and
    retained public finite fixtures pass. Unsupported endpoints are rejected;
    public DD/Float batch-3 timed out at 150 s. Broader domains/native E6 remain open.],
  [Broader derivative domains], [The full five-case mixed D0 batch now passes
    with error below $2^(-70)$. True thresholds and unsupported transverse
    massless/Gram/Cayley domains retain typed errors; old timeouts remain.],
  [Physical top-box binary64], [The original binary64 guard rejection remains
    visible. Complete Float128/256 evaluations now pass; this does not establish
    binary64 stability or performance.],
  [Supplied-product high points], [Pure PV/hybrid scalar requests with six or
    more supplied-product propagators require domain validation. The public
    certificate wrapper timed out at 90 s. Bounded argument guards pass; broader
    mixed-precision recurrence conditioning remains unqualified.],
  [Legacy snapping scope], [All30 retained cases pass:22 primary plus eight
    verified128/256-bit rescues. Earlier failures are preserved; generalized
    requests and broader domains remain unqualified.],
  [Runtime performance], [Interpreter: 7 regressions/1 improvement. Native:
    5 regressions/1 parity/2 improvements, in a separate paired session.
    Newer and broader performance remains unmeasured; the goal is unmet.],
))

== Canonical physical reference audit
Original IBP and PV/hybrid disagree in both UV and finite rank-four slots for
$g g arrow H H H$, despite stable flags. The largest normalized finite
error is $0.6001338886545732$ at moment 54. UV power counting and converged
high-precision evaluation now adjudicate that component; details follow on
the next page. The other three reference families agree across original modes
within $2.22 times 10^(-14)$. All-slot comparisons are retained in
#raw("validation/physical/canonical-reference-audit.json").

The retained #raw("validation/pv-high-rank/reference-coverage.json") also records
rank-seven original PV instability for five to seven propagators and a falsely
stable four-propagator disagreement. These reference failures remain visible;
stable IBP reference values do not themselves certify Rust high-rank coverage.

== Independent integral evidence
The rank-four bubble's 50/80-digit parameter integrals agree with original
IBP; independent lightlike-bubble 60/100-digit integrals include first
invariant derivatives. Their fixtures and reproduction scripts are retained
under #raw("validation/pv-high-rank"). Correctly rounded master inputs reduce
the synthetic pentagon error to $3.11 times 10^(-12)$ with 256-bit reduction
and $2.23 times 10^(-11)$ with binary64 reduction; this isolates amplified
master error. Full independent pentagon evidence follows.

#pagebreak()
= Independent pentagon adjudication
These checks establish selected mathematical values independently of an
original-reference stability flag. They do not qualify every pentagon tensor,
physical graph, runtime domain or numerical precision.

== Rank-six Euclidean tensor
For the retained five-propagator input with squared masses $(4,9,16,25,36)$,
$mu=3$ and numerator $q_0^5 q_3$, direct Feynman-simplex integration gives
native finite coefficient $-0.002438378935447123 i$. Two independent simplex
coordinate mappings agree within $4.34 times 10^(-19)$; the largest reported
quadrature error is $2.77 times 10^(-15)$. The result agrees with the converged
256-bit PV value, and all poles vanish by angular integration and power
counting. The positive denominator bound is $F >= 3.75$.
This numerical quadrature uses no tensor-reduction or scalar-master library;
its error estimates are not interval certificates. Evidence and the standalone
script are in #raw("validation/pv-high-rank/pentagon-simplex-oracle.json").
The original binary64 errors remain $2.23 times 10^(-10)$ for Rust and
$3.40 times 10^(-10)$ for IREGI, above the unchanged gate. The complete public
rescue now flags that primary lane unstable and verifies all 210 moments /
840 slots at 128/256 bits, plus an independent absolute-error assertion below
$10^(-13)$ for this coefficient. Evidence is in
#raw("validation/rescue-pentagon-rank6.json"). Its retained binary hash and
successful assertions establish this run; full output arrays were not printed
and the exact source snapshot was not saved at compilation.
An isolated Symbolica branch-cache patch also passes all 1320 rank-seven
slots at 128/256 bits and this independent coefficient assertion. Its frozen
core matches the timed-out baseline; masters and the dependency lock differ.
Exact lowering takes 1.784 s in a 115.35 s cold test, not a controlled benchmark.
Production integration is pending. The earlier 240 s timeout is retained in
#raw("validation/rescue-pentagon-rank7.json").

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
are in #raw("validation/shifted-pentagon/evidence.json"). The new genuine
Rust D6 master passes thirteen consolidated tests: eight genuine-master
checks and five endpoint checks,
including a common-mass Dual derivative, failed-lane isolation, Float128/
DoubleFloat comparison, numerical guards and independent negative-width
agreement at two contour slopes. Public integration now passes a mixed
three-lane test: positive-F and negative-width values, rejected endpoint,
poles, scale covariance, empty input and internal box count. The unchanged
output target is checked after error amplification; the earlier budget failure
is retained. See #raw("validation/shifted-pentagon/core-qualification.log").
The public DD/Float batch-3 test timed out at 150 s without a numerical result.
Broader endpoints, higher Dual orders and native-feature E6 remain unqualified;
no performance claim follows. The separate original complex-box reference discrepancy
is documented in #raw("validation/shifted-pentagon/contours.json").

#pagebreak()
#include "conventions.typ"
#pagebreak()
#include "algebra-reuse.typ"
