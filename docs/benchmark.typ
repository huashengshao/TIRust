// Measured values live in data/results.json. null means no measurement.
#let data = json("data/results.json")
#let report-date = data.at("report_updated_on", default: data.date)
#set document(title: "TIRust benchmark", author: "Hua-Sheng Shao")
#set page(paper: "a4", margin: (x: 1.6cm, y: 1.8cm), numbering: "1 / 1",
  header: context {
    if counter(page).get().first() > 1 {
      set text(8pt, fill: luma(110))
      [TIRust benchmark #h(1fr) #report-date]
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
#let historical809 = data.qualified_80949_benchmark
#let current = data.qualified_6b24_benchmark
#let physical = data.physical_80949_benchmark
#let collier = data.at("qualified_collier_benchmark", default: none)
#let latest-experiments = data.at("latest_isolated_timing_cohorts", default: none)
#let lazy-cohort = data.at("lazy_receipt_capture_performance", default: none)
#let typed-e6 = data.at("private_direct_e6_n2_typed", default: none)
#let thin-mask = data.at("private_thin_n6_rank2_mask", default: none)
#let shared-collier = data.at("collier_shared_attempt004_summary", default: none)
#let joint-direct = data.at("selected_native_four_engine_row_local", default: none)
#let python-checkpoint = data.at("python_interface_compile_checkpoint", default: none)
#let n5-identities = data.at("collier_n5_identity_audit_qualification", default: none)
#let n7-exact = data.at("private_thin_n7_exact_qualification", default: none)
#let rank7-failed = data.at("private_thin_n6_rank7_failed128_audit", default: none)
#let python-runtime = data.at("python_interface_baseline_runtime_qualification", default: none)
#let notebook-native = data.at("native_notebook_baseline_partial_qualification", default: none)
#let rank7-oracle = data.at("private_thin_n6_rank7_component126_oracle", default: none)
#let fixed-wheel = data.at("python_interface_fixed_wheel_qualification", default: none)
#let fixed-gallery = data.at("native_notebook_fixed_wheel_qualification", default: none)
#let native-capability = data.at("native_f64_opt_in_capability_qualification", default: none)
#let native-metadata = data.at("native_f64_opt_in_compatibility_completion", default: none)
#let n7-failed = data.at("private_n7_rank0_failed_attempt", default: none)
#let scheduler-fixture = data.at("private_uniform_branch_scheduler_fixture", default: none)
#let public-six = data.at("public_six_point_api_qualification", default: none)
#let public-six-cargo = data.at("public_six_point_api_current_cargo", default: none)
#let current-dual = data.at("c0d0_dual_identity_qualification", default: none)
#let collier-cell(r) = table.cell(
  fill: if not r.correctness_passed or r.gate in ("inconclusive-noise", "inconclusive-short-block") {
    amber
  } else { pale }, {
    set par(leading: 0.35em, justify: false)
    let verdict = if not r.correctness_passed { "Unqualified" }
    else if r.gate == "inconclusive-noise" { "Inconclusive: noise" }
    else if r.gate == "inconclusive-short-block" { "Inconclusive: short block" }
    else if r.gate == "improvement" { "COLLIER faster" }
    else if r.gate == "regression" { "IREGI faster" }
    else if r.gate == "parity" { "Compatible with parity" } else { "Unqualified" }
    text(size: 9pt, weight: "bold", verdict)
  },
)
#let reference-time-cell(value) = table.cell(fill: pale, {
  if is-num(value) { text(size: 12pt, weight: "bold")[#fixed(value, digits: 3)] }
  else { text(size: 9pt)[Unmeasured] }
})
#let lazy-ratio-cell(row) = table.cell(
  fill: ratio-fill(row.candidate_over_baseline_latency_ratio), {
    set par(leading: 0.3em, justify: false)
    let verdict = if row.classification == "improvement" { "Gain" }
      else if row.classification == "regression" { "Regression" } else { "Compatible with parity" }
    text(size: 12pt, weight: "bold")[#fixed(row.candidate_over_baseline_latency_ratio, digits: 4)]
    linebreak()
    text(size: 7pt)[99% CI #fixed(row.latency_ratio_ci99.at(0), digits: 4)–#fixed(row.latency_ratio_ci99.at(1), digits: 4)]
    linebreak()
    text(size: 8pt, weight: "semibold", verdict)
  },
)
#let joint-ratio-cell(row) = table.cell(
  fill: if row.gate == "inconclusive-short-block" { amber }
    else { ratio-fill(row.statistics.median) }, {
    set par(leading: 0.25em, justify: false)
    let verdict = if row.gate == "inconclusive-short-block" { "Short block" }
      else if row.gate == "improvement" { "Gain" }
      else if row.gate == "regression" { "Slower" } else { "Inconclusive" }
    text(size: 11pt, weight: "bold")[#fixed(row.statistics.median, digits: 3)]
    linebreak()
    text(size: 6.4pt)[#fixed(row.statistics.ci99.at(0), digits: 3)–#fixed(row.statistics.ci99.at(1), digits: 3)]
    linebreak()
    text(size: 7pt, weight: "semibold", verdict)
  },
)
#let reference-coverage-cell(row) = table.cell(
  fill: if row.status == "accepted-COLI" { rgb("#d9f0d3") } else { amber }, {
    set par(leading: 0.25em, justify: false)
    text(size: 9pt, weight: "semibold", if row.status == "accepted-COLI" {
      "COLI admitted"
    } else { "Unqualified" })
  },
)
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
  #text(22pt, weight: "bold", fill: navy)[TIRust benchmark]
  #v(-0.3em)
  #text(12pt, fill: luma(80))[Batched one-loop reduction: TIRust, IREGI, COLLIER and MadLoop]
  #v(0.2em)
  #text(9.5pt, fill: luma(110))[#report-date · #data.host]
]
#v(0.6em)
#block(fill: pale, inset: 10pt, radius: 4pt, width: 100%)[
  #set text(9.5pt)
  *Reading guide.* Every displayed timing ratio has TIRust in the numerator:
  _TIRust / IREGI_, _TIRust / COLLIER_, or
  _TIRust(candidate) / TIRust(baseline)_. Physical graphs use _TIRust / MadLoop_.
  Green means TIRust is faster, white parity and red slower. The archival
  reference-only table shows absolute times and outcomes. Gray is unmeasured;
  amber marks pending or inconclusive evidence.
  Source inspection does not establish numerical agreement. #data.measurement_note
]
#v(0.4em)
#block(stroke: 0.6pt + luma(200), inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Current priorities*
  #table(columns: (1fr, 2fr, auto), stroke: none, inset: (x: 4pt, y: 4pt),
    [*Feature completion*], [Independently validate all agreed features through six points and rank six, including dots], [*Pending*],
    [*Performance optimization*], [Separate later work; publish the tested version and stop once the feature gate passes], [*Deferred*],
    [*Complete evaluation*], [Six physical rows measured; two failures retained], [*Inconclusive*],
  )
]
= Current coverage
#text(size: 9.5pt)[The local Python interface and example gallery execute successfully with
repaired wheel004. Complete mathematical acceptance remains open. Low-rank
COLLIER speed is not a completion gate under the latest user priority;
all historical measurements below retain their original source snapshots.]
#report-table(table(
  columns: (1.05fr, 3.6fr, 0.8fr), inset: (x: 6pt, y: 8pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Interface / domain], [Executed scope or remaining gate], [Status]),
  [Local PyO3 wheel], [17/17 numerical and 12/12 interoperability tests, including the actual SVG renderer; installed consumer typing has zero errors/warnings.], [Passed],
  [Dotted propagators], [Generalized powers/dimension shifts have their strongest retained coverage at N≤4. The current wheel/gallery checks the dotted bubble against its mass Dual derivative.], [Scoped],
  [Example gallery], [4/4 native test methods; three successful HTML exports; all three rendered at desktop and narrow dark sizes in real Chromium.], [Passed],
  [Tensor coverage], [Actual Cargo N1–N5 rank7 massive arrays, massless N2–N4 arrays and N6 ranks4–7 pass retained-family checks at Float128/256. N6 lifecycle/contractions pass. Public N7 is still scalar-only.], [Scoped],
  [Performance], [Historical private selected TIRust/COLLIER: seven slower families and one short inconclusive family. The fresh scheduler cohort is ineligible because of host contention; no default adoption.], [Measured],
))
#text(size: 8.5pt)[Local Linux/Python 3.13 execution qualifies the named examples and
focused API tests. Release distributions, other platforms, separate Community-host
loading and a reactive numerical browser frontend remain unqualified.
C0/D0 mass-Taylor identities now pass on wheel004 at working 128/256 bits with
requested accuracy 53. Historical broader mixed-Dual finite-difference evidence
remains separate; these identities do not rerun that complete suite.]

#pagebreak()
= Rank-seven correctness checkpoint
#let six-rank7 = data.six_point_rank7_current_cargo_qualification.snapshot
#let arrays-n1-n5 = data.rank7_arrays_n1_n5_current_qualification.snapshot
#let arrays-six-life = data.rank7_arrays_six_lifecycle_qualification.snapshot
#let arrays-six-literal = data.rank7_arrays_six_literal_qualification.snapshot
#block(fill: pale, inset: 10pt, width: 100%)[
  *Updated stopping envelope: six points and rank six, with all agreed features
  including dots.* Retain the rank-seven checks already obtained and explain
  the seven-point challenge. Further optimization is deferred. The complete
  functional gate remains open; historical timings retain their measured contexts.
]

#report-table(table(
  columns: (1.2fr, 3.4fr, 0.9fr), inset: (x: 7pt, y: 11pt),
  stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 4 { rgb("#d9f0d3") } else { amber },
  table.header([Domain], [Executed scope], [Outcome]),
  [N1–N5 / rank7], [Actual Cargo PV: three massive samples per topology, Float128/256, 39,600 complex reference coefficients and 19,800 precision pairs.], [Pass¹],
  [Massless N2–N4], [Complete rank7 arrays: 7,920 reference comparisons and 3,960 precision pairs. Separate UV, IR and IR² slots retained, including the scaleless bubble.], [Pass¹],
  [N6 / rank7 batches], [21,120 tensor and 128 sparse/dense contraction coefficients; empty, mixed failed lanes, reorder/chunks, cloned scratch and malformed-call recovery.], [Pass¹],
  [N6 / ranks4–7], [Separately requested plans, Float128/256: 5,888 reference coefficients and 2,944 precision pairs. All previous rank7 prefixes retained.], [Pass¹],
  [N7 / rank0], [Actual Cargo Float128/256 singleton agreement. Nine-call lifecycle stopped after six completed calls; later scratch/recovery checks remain open.], [Partial],
  [N6 / rank7 COLI], [Native accuracy flag −2; 315 missing diagnostic coefficients and one finite discrepancy. All three pole probes return failure sentinels.], [Rejected],
  [Accepted functional goal], [Through N6/rank6: generalized tensors, broader numeric/Dual and supplied-invariant domains still need qualification. N7/rank7 remains a separate challenge.], [Open],
))

¹ Retained massive and massless families, native raw tensor components, requested
accuracy 53 and working precision 128/256. Every moment includes all four native
Laurent slots. Independent-reference gates remain $10^(-10)$; precision and
scaling gates remain $2^(-80)$. Maximum reference discrepancies are
$1.839 times 10^(-16)$ for massive N1–N5, $2.522 times 10^(-16)$ for massless
N2–N4 and $1.265 times 10^(-16)$ at N6.
Working precision does not assert full working-precision accuracy. These
cohorts retain their exact captured source contexts, including the older N6
lifecycle context before the Cayley change. No kernel timing claim follows.

#pagebreak()
== Complete arrays, batches and preparation
The N1–N5 corpus contains 30 successful sample lanes and passes 13,200 uniform
scaling checks, 19,800 exact IR-zero checks and 996 convergent UV-zero checks.
Original IBP supplies the reference arrays. Three original bubble PV requests
produce no result; six stable triangle/box PV arrays disagree in 486 finite
positions. Those failures remain visible and are never substituted as accuracy
anchors.

The N6 rank-seven batch/contraction cohort passes at both precisions with
16 successful and eight explicitly rejected lanes. Sparse odd/mixed support
and dense coefficients covering all 330 moments use the existing Symbolica
contraction evaluator. The maximum contracted-reference discrepancy is
$5.090 times 10^(-18)$. All actual planar-child fallbacks are retained.
Cloned scratch was exercised serially; concurrent execution is a separate gate.

The adopted routing and eligible five-point Cayley preparation uses Symbolica
rational-polynomial matrices and exact identity certificates. Other syntax
retains AtomField. Physical singularities preserve exceptional recurrence;
conversion/certificate errors remain preparation failures. All 51 focused IBP
tests pass. The subsequent literal N6 rank4–7 checks cover complete arrays in
this source context. No local algebra solver or numerical specialization was
introduced.

Typed-Dual malformed-input failures now clear earlier counters, profiling times
and diagnostic messages. Two public regression tests pass, including a genuine
mass derivative and valid recovery. Initial test-driver failures remain saved.
This diagnostic fix does not change numerical formulas or qualify high-point
Dual inputs.

== Reference accuracy adjudicated
Original IREGI and its bundled OneLOop were built with uniform quad arithmetic,
retaining the original complex-matrix sources, formulas, thresholds and
limited-digit constants. Runtime type canaries and four scalar normalization
controls pass. Stable IBP agrees with both complete TIRust arrays. The older
binary64 IBP comparison still fails 90 slots; those failures remain preserved.
Quad PV is unstable and disagrees in 139 positions. Quad Hybrid reports
stability but disagrees in 19, and remains unqualified. Stored 113-bit
arithmetic does not establish achieved 113-bit result accuracy.

The rejected COLI capture is not an independent accuracy witness. Its zero
explicit Laurent-pole residual count does not turn its failed pole-parameter
probes into passing checks. No high-rank kernel timings are inferred from
validation-stage durations.

Saved-evidence replay, without new integral evaluation:
#raw("python3 validation/public-rank7-current-cargo-n1-n5/check.py")\
#raw("python3 validation/highpoint-six-point-rank7-lifecycle/check.py")\
#raw("python3 validation/highpoint-six-point-literal-ranks4-7-current-cargo/check.py")\
#raw("python3 validation/quad-original-iregi-small-n-rank7/check.py").
The separate rejection record is
#raw("validation/collier-n6-rank7-diagnostic/qualification.json").

#pagebreak()
== Massless poles and shared execution
The three massless points separately exercise a lightlike scaleless bubble,
an on-shell triangle and a scattering box. Original quad IBP supplies all three
reference arrays. All six Float128/256 arrays pass; the largest normalized
reference discrepancy is $2.522 times 10^(-16)$. The largest precision-pair
discrepancy is $5.891 times 10^(-37)$. All six actual routes are analytic PV,
with no fallback. In particular, the bubble's nonzero UV and IR single poles
are retained separately even though their sum vanishes. No positive-mass
IR-zero rule or zero projection is applied to these data.

This cohort uses the same frozen actual-Cargo artifact as the earlier N1–N5
arrays. It does not qualify later source edits or the whole massless domain.
The independent reviewer reconstructs all 11,884 comparison records. Per-case
terminal records pass; controller phase order and maximum concurrency are
executor-attested because absolute UTC start/finish records were not saved.
Elapsed durations are diagnostics, not performance measurements.

The additive #raw("TensorReductionPlan") / #raw("TensorWorkspace") facade
shares eligibility and contraction with the existing generic, six-point and
seven-point engines. Six ordinary tests and a separately selected actual
Float128 scalar delegation test pass. The latter checks N6/N7 direct-wrapper
equality, invalid-lane isolation, contractions, nested execution counters and
empty-call reset. It introduces no recurrence or scalar formula. Explicit
Dual layouts, profiling and nondefault policies retain generic execution;
requested algorithms and compiled scalar-master defaults are preserved.
These core tests alone do not qualify new CLI or Python wheel builds.

The earlier separate rank-seven IBP captures complete at N1/N3 and pass 15,840 reference
comparisons, 7,920 precision pairs and 5,280 scaling identities. Both N2
captures time out during workspace preparation and produce no coefficient
array. The corpus remains incomplete. Original cleanup failures and later
process-absence observations are retained separately. No timing gain or
complete-method qualification follows from the available results.

The private seven-point probe completes ranks one and two at one regular
positive-mass point, passing 160 complex reference comparisons and 80
precision pairs. Its earlier rank-seven attempts produce no arrays: Float128
reaches its preparation deadline, and Float256 is deliberately stopped before
its deadline after the same preparation/resource problem. Both original
cleanup failures remain recorded; later process-absence observations are
additive. These are private positive-rank witnesses and failed rank-seven
attempts, not public rank-seven qualification.

A targeted Symbolica compiler candidate reuses an exact SSA condition's
zero/nonzero fact inside its dominating branch. The numeric condition value
is preserved, and branch facts are restored before siblings and function
scopes. Seven new and eleven relevant existing tests pass, including mixed
higher Duals and instruction-major batching. The initial optimized test build
times out; a test-only opt0/debug0 build completes with the same sources and
features. These engineering records establish focused compiler semantics;
the subsequent numerical N2 pass and N7 resource failure are reported below.

Additional saved-evidence replay:
#raw("python3 validation/public-rank7-massless-ir/check.py")\
#raw("python3 validation/shared-tensor-facade/check.py")\
#raw("python3 validation/public-rank7-ibp-owned-artifact-n1-n3/check.py").

#pagebreak()
= Shared CLI and additional rank-seven IBP
#let ibp-n2-postfix = data.rank7_ibp_n2_postfix_qualification.snapshot
#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 12pt),
  stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
  table.header([Check], [Executed scope], [Outcome]),
  [N2 / rank7 / IBP], [Three massive rows at Float128/256: 7,920 reference coefficients, 3,960 precision pairs, 2,640 scaling checks and 3,960 IR-zero checks. Actual method IBP; no fallback.], [Pass],
  [CLI / low points], [Thirteen exact CLI tests and four smoke calls: full bubble reference arrays, DOT numerator preparation/contraction, DoubleFloat, precision rescue and invalid-neighbor isolation.], [Pass],
  [CLI / N6 / rank3], [All 140 complex Laurent coefficients agree with the retained quad reference prefix. The shared facade selects the six-point engine and reports nested counters.], [Pass],
  [CLI / N7 / rank0], [All four coefficients agree with the retained quad reference. Twenty-one packed children, 105 master requests, contraction and invalid-neighbor handling are checked.], [Pass],
))

The new N2 comparison follows the scoped Symbolica condition-fact change.
Its largest normalized reference discrepancy is $1.839 times 10^(-16)$;
the largest precision discrepancy is $7.579 times 10^(-37)$. Scaling and IR
residuals are exactly zero in the captured representation. The two old N2
timeouts remain failed historical attempts. The new complete arrays close
this particular retained-family gap, without qualifying the whole IBP domain.

The CLI now uses the library's shared tensor plan, workspace and contraction
helper. Graph import and Symbolica numerator preparation remain separate.
Default Hybrid, precision rescue and scalar-master policy are preserved.
Output exposes the selected engine, nested counters and per-lane failures;
unavailable rescue observations are null. Explicit Dual layouts continue
through the generic engine. No high-point Dual domain is implied.

The first CLI test run fails a test-only diagnostic-wording assertion; the
one-assertion correction passes all thirteen tests. The first N7 scalar smoke
times out before a result. A separately bounded attempt with the same
executable, inputs and accuracy gates passes on a free core. Both original
failures remain visible. Actual build and validation durations are engineering
records, not comparable kernel timings. No new performance ratios are added.

The compact CLI evidence inherits the exact 77-file source snapshot of the
N2 capture and retains its corrected test source separately. These successful
artifacts precede the later component-Taylor and preparation-instrumentation
changes. They do not silently qualify those later sources.

Saved-only reproduction:
#raw("python3 validation/public-rank7-ibp-n2-postfix/check.py")\
#raw("python3 validation/cli-shared-facade/check.py").

#pagebreak()
= Generalized reference anchors and the N7 blocker
#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 13pt),
  stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 1 { pale } else { amber },
  table.header([Domain], [Saved evidence], [Outcome]),
  [Original IREGI / generalized rank7], [Twelve complete N2/N3 arrays: ten new first/last-dot, first-power-three and D2/D6 anchors, plus two unit-power controls. 15,840 complex slots retained.], [Anchors],
  [TIRust / N7 / rank7], [Actual current-Cargo private Float128 attempt fails allocation under a 32 GiB address-space cap during workspace preparation. Zero output coefficients. Float256 and pair gates unrun.], [Failed],
  [Public N7 tensor support], [The public guard remains rank zero. Private rank-one/two evidence and the new resource failure do not authorize positive-rank promotion.], [Open],
))

The generalized reference build applies only the already reviewed factorial-pair
binomial-recurrence repair to its generated copy. The original source and binary
remain untouched. The two unit-power controls match the earlier ordinary quad
array and each other exactly. Independent checks cover 22,321 saved records,
including UV power counting, positive-mass IR zeros and reflection identities.
These are reference anchors for subsequent Rust comparisons. They are not a
proof of every generalized finite coefficient or a new Rust/Dual qualification.
Uniform quad storage retains the original limited-digit constants and stopping
criteria; achieved 33-digit accuracy is not asserted.

The new N7 failure uses a freshly built main-source test executable and a
333-file source snapshot. Its Float128 child aborts after about 612 seconds;
the timeout flag is false and owned cleanup succeeds. Its one memory snapshot
is not a lifetime peak. A single read-only debugger sample is denied by system
tracing policy and produces no stack profile. The failed attempt's coarse log
does not uniquely identify the expensive substage. Later fine instrumentation
and any optimization are separate work, with no passing array claimed here.

This preserves the distinction between an unsupported public domain, a failed
private resource attempt and a numerical disagreement. All remain visible in
the acceptance envelope. No preparation duration is interpreted as a speedup
against original IREGI or COLLIER.

Saved evidence:
#raw("python3 validation/quad-original-iregi-generalized-rank7/check.py")\
#raw("validation/highpoint-seven-point-main-rank7-resource-failure/qualification.json").

#pagebreak()
= Independently validated planar generalized scalars
#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 4 { amber } else { rgb("#d9f0d3") },
  table.header([Check], [Executed scope], [Outcome]),
  [N5 / planar scalars], [D6 unit powers, plus one positive dot at each of the five positions in D4 and D6. Eleven requests, Float128/256 and retained D4 derivative controls.], [Pass¹],
  [Full saved cohort], [196 successful lanes, 26 explicit invalid lanes with null output, 880 native Laurent/Taylor components and 2,064 independent decisions.], [Pass¹],
  [Derivatives and routing], [Mass Dual versus dotted scalar, ordinary finite differences, common translations, scale changes, repeated samples and input preservation.], [Pass¹],
  [Original quadrature grids], [All direct accuracy checks pass, but 24 refinement decisions fail. The original inconclusive result and three failed optional imports remain preserved.], [Retained],
))

¹ One massive positive-F planar family: shifts
#raw("0, ex, ey, ez, ex+ey+ez"), squared masses
#raw("4, 9, 16, 25, 36") and scale energy three. The parameter denominator
satisfies $F >= 4$. Symbolica prepares the polynomial, Jacobian, input jets
and derivatives independently of reduction. The numerical oracle integrates
the defining parameter integral, calling no reduction or scalar-master backend.

The public certified-planar path uses its full-Taylor geometry certificate and
constant-one denominator relation. These requests reduce to masters with at
most four propagators; they do not evaluate a genuine full-rank D6 pentagon.
Every dot position is checked in both dimensions. Native UV1, IR1, IR2 and
finite slots remain separate.

Initial orders 12/16/20 give 2,040 passing and 24 failing decisions. The separately
bounded oracle-only refinement uses orders 24/28/32, retaining all thirty Rust
captures. It passes every decision with unchanged thresholds: oracle differences
at most $1.065 times 10^(-13)$ then $5.398 times 10^(-16)$, below $10^(-12)$;
reducer/oracle discrepancy at most $2.749 times 10^(-18)$, below $10^(-10)$.
Precision-pair discrepancy is at most $7.854 times 10^(-32)$, below $2^(-80)$.
Mass-Dual/dot discrepancy is $2.943 times 10^(-32)$ and finite-difference/dot
discrepancy $5.906 times 10^(-21)$.

An independent saved-data audit reconstructs every exact squared error,
normalization, threshold and decision, with eight negative controls. Its
portable replay preserves both the original failures and refined success.
Earlier optional-import failures evaluate no quadrature; a pinned environment
continuation is recorded separately. No Rust evaluation is repeated to obtain
the refined oracle agreement.

This closes the named scalar slice. Generalized tensors, other endpoints,
lower-span charts, higher Dual orders and genuine D6 pentagons require their
own coverage. Numerical convergence estimates are not rigorous enclosures.
No validation duration is interpreted as a performance benchmark.

Saved-only replay:
#raw("python3 validation/certified-planar-generalized-scalar-qualification/check.py").
The self-contained capsule retains 75 captured source files, raw arrays,
all six successful quadrature outputs and both independent audits.

#pagebreak()
= Method coverage and the seven-point challenge
#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 12pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 2 { rgb("#d9f0d3") } else { amber },
  table.header([Check], [Executed scope], [Outcome]),
  [Additional ordinary methods], [N1/N2/N3 Hybrid, N4 IBP/Hybrid and N5 Hybrid: six Float128/256 pairs with complete rank7 arrays, including every rank6 prefix.], [Pass¹],
  [Preparation resource fix], [Narrow Symbolica variable discovery and no-leaf activation short-circuit; nineteen focused tests, including 21,204 declared inputs and reordered mixed Taylor components.], [Focused],
  [N5 requested IBP], [Actual PV completion is reported. Its 128-bit output agrees with the reference, but the strict pure-IBP method gate fails; 256-bit partner unrun.], [Partial],
  [N7 / rank7 diagnostic], [Instrumented main-source Float128 attempt aborts allocation under 32 GiB during master-activation preparation. No tensor coefficients; no public promotion.], [Failed],
))

¹ The six successful pairs pass 15,840 quad-reference comparisons, 7,920
precision pairs, 7,920 IR-zero checks and 15,840 retained-PV controls.
Maximum normalized reference discrepancy is $1.839 times 10^(-16)$.
These are one-row retained families, not a complete kinematic domain.
At N5, an epsilon-dependent IBP dimension divisor prevents completion from
the available A0–D0 Laurent orders. The explicitly reported PV completion
passes 1,320 reference and 1,320 previous-PV checks. This is a correct-output
diagnostic, not pure-IBP qualification. True generalized D6 scalars are separate.

The instrumented seven-point attempt makes the preparation challenge explicit.
Its child workspace prepares successfully. The parent spends about 214 seconds
in distance conditioning, 197 in output monitors and 107 in denominator
monitors, then aborts during master activation. These are measurements of one
failed diagnostic, not kernel benchmark comparisons or a complete memory peak.

Source inspection finds two avoidable resource costs. Rational-polynomial
conversion supplies 21,204 declared inputs although each distance uses only
a few shifts. Existing Symbolica variable discovery constructs the same
polynomial using its actual variables; the full evaluator input order, precision
and Taylor shape remain unchanged. The exact certificate uses the same approach
and retains declared-symbol validation. Separately, a parent with no own
scalar-master leaves needs no master-activation graph traversal. Other stability
monitors and every path with leaves remain intact.

The actual Cargo build and nineteen focused tests pass, including unknown-symbol
rejection, wide late-input/mixed-jet equality, exact certificate contexts,
no-leaf output/monitor preservation and existing active-master controls.
The initial preflight and new-test API compile failures are retained separately.
This establishes the focused preparation fix; it does not turn the failed N7
attempt into a successful array. No further long N7 run is required before
closing the accepted six-point/rank-six objective.

Seven-point rank-seven completion would still need fresh full arrays at two
precisions, an independent accuracy check, batched lifecycle/contractions and
the remaining numeric/generalized domains. Public N7 remains scalar-only.

Saved-only replay:
#raw("python3 validation/public-rank7-method-completion-n1-n5/check.py").
Additional source and scope:
#raw("validation/sparse-polynomial-activation-resource-fix/qualification.json")\
#raw("validation/highpoint-seven-point-fine-profile-resource-failure/qualification.json").

#pagebreak()
= Additional correctness milestones
These retained private results support the remaining implementation work.
They do not expand the qualified public domain by themselves.

#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 15pt),
  stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Scope], [Complete saved checks], [Status]),
  [Generalized N6/N7], [N6 dotted, N7 dotted and N7 h=−1: 240 complex outputs at Float128/256; original agreement, scaling, repeated/chunked batch profiles and 80 exactly retained control values.], [Private pass],
  [N6 scalar Dual], [19 tests and four captures: 528 parent Taylor slots, 156 packed child rows and 264 precision pairs. Independent first mass derivative/dotted anchor and coherent dilation jets.], [Private pass],
  [N7 scalar API], [35 focused actual-Cargo tests; singleton Float128/256 checks pass. The partial lifecycle retains 28 moment and 28 unit-contraction slots, including rejected neighbors and reordered/chunked inputs.], [Partial],
))

The generalized preparation change uses Symbolica's existing rational-polynomial
matrix operations, with exact reconstruction certificates and an AtomField
fallback. It also distinguishes a singular matrix from a preparation failure.
The retained cohort passes 7,415 independent saved-data checks, covering all
2,060 input slots and all complete outputs. The preparation adapter is now
adopted into the main Cargo source and passes the ordinary-array checks above.
This older generalized numerical cohort retains its private dependency context;
it has not been rerun in the new Cargo context. Higher-rank generalized requests,
genuine D6 domains, complex masses and Dual directions remain separate work.

The component-sensitivity prototype transports complete heuristic Taylor
sensitivity operands rather than padding derivative entries with zeros. Its
mass derivative agrees with independent dotted IBP/original anchors; its
coherent scale jets pass homogeneity and precision checks. These are numerical
sensitivity estimates, not rigorous derivative uncertainty enclosures. General
independent mixed directions, finite differences and public N6/N7 Dual
constructors remain unqualified. The rectangular mixed-direction follow-up is
source-only.

The N7 lifecycle stopped at its fixed deadline during the independent-scratch
call. Six preceding calls are retained; the remaining scratch, later empty and
malformed-call recovery checks are not silently treated as passing. The first
failed Float256 attempt also remains alongside its separately bounded successful
follow-up. Positive-rank seven-point validation is ongoing.

Exact-master preparation-cache experiments are preserved separately: their
38-request whole-output regression passes, while earlier typed schedules remain
incomplete. No cache candidate or timing gain is adopted by this checkpoint.
No new high-rank benchmark measurements are added here.

Sources and saved-evidence replay:
#raw("validation/routing-rational-cayley-full-core/") and
#raw("validation/highpoint-dual-sensitivity/"). The actual N7 source/build
boundary is recorded in
#raw("validation/highpoint-seven-point-current-cargo/qualification.json").

#pagebreak()
= Historical kernel baseline: #raw("4c32ece")
#figure(kind: table, caption: [*Massive bubble/triangle, binary64, hybrid mode.*
  TIRust and IREGI total medians in µs/point. *TIRust / IREGI* ratios are paired-block medians;
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
The #raw("80949db") snapshot gives five regressions/three noisy rows.
Latest #raw("6b24c03") passes all eight checks: seven regressions/one noisy row.
Separate paired comparisons follow; performance acceptance is unmet.]

#text(size: 9pt)[Newer validation includes four integrated graphs. The top
pentagon at Float 128/256 agrees with PV/hybrid and MadLoop; original IBP still
fails. The latest physical timing comparisons remain inconclusive.]

#pagebreak()
= Historical stage profiles: #raw("4c32ece")
TIRust medians from separate instrumented passes, in µs/point. All bars use
0–40 µs/point; S denotes the Symbolica interpreter, N the generated native
backend. Sessions are separate: compare each with its own reference in the preceding baseline table,
not with the other backend's absolute bar length. Raw ns/batch are divided
by $1000 times "batch width"$.

#report-table(table(
  columns: (3.5cm, 10.5cm, 1.5cm), inset: (x: 3pt, y: 3pt), stroke: none,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case / batch / backend], [TIRust stage time: 0 #h(1fr) 20 #h(1fr) 40 µs/point], [Total]),
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
seed 271828. TIRust used release opt-level3, thin LTO and one codegen unit;
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

#text(size: 8.5pt)[Native allocator adoption remains rejected. Historical 24 timing
rows give two singleton parities/22 noisy rows; new D0 timings include two
batch32 regressions. D0 exact old/new parity passes 40 configurations/1,488 complex
outputs (f64/DD/Float256); interpreter differences remain. Separate candidate
decisions appear in the optimization ledger and #raw("validation/native-stack-batch/").]

#pagebreak()
= Latest kernel snapshot: #raw("6b24c03")
All eight massive bubble/triangle checks pass $10^(-10)$, with maximum
normalized error $2.46 times 10^(-13)$. Seven timing rows regress; the rank-one
triangle singleton exceeds the unchanged 2% interval-width limit (2.357%)
and remains *inconclusive*. This measured snapshot predates the standalone
planar and finite-D0 certification APIs and later expression preparation.
The performance requirement remains unmet.

#figure(kind: table, caption: [*Binary64 hybrid interpreter, µs/point.*
  Latency at batch 1 and amortized cost at batch 32. Ratios are
  *TIRust / IREGI*, using the same-run paired original-IREGI measurement.], report-table(table(
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
  table.header([Case / batch], [TIRust stage time: 0 #h(1fr) 20 #h(1fr) 40 µs/point], [Total]),
  ..for r in current.rows {
    ([#r.label r#r.rank / #r.batch_width], stage-bar(r, maximum: 40), [#fixed(r.profile_us_per_point.total, digits: 3)])
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

CPU 254, five warm-ups and 41 randomized paired blocks targeting 100 ms use
identical inputs, complete tensor arrays and all native Laurent slots.
Structural allocations are reused; numerical results are recomputed and
reference caches cleared per sample. Transport is excluded. rustc 1.99 uses
opt-level3, thin LTO and one codegen unit; GFortran 15.2 uses #raw("-O3").

The complete optimized workspace build exits successfully. All 414 immutable
build-source hashes, both binaries, feature selection and harness match the
retained provenance. Other project computations were held. CPU 254 was selected
by a read-only load survey because unrelated work occupied CPU 0; external
host activity remained uncontrolled. No system settings were changed.
Historical sessions used different CPUs and timing protocols; only each
session's paired reference supports its verdict, with no causal revision speedup.
Raw samples and reproduction: #raw("validation/benchmark/baseline-6b24c03-licensed/").

#if joint-direct != none [
  #pagebreak()
  = Direct TIRust / IREGI / COLLIER comparison
  Private optimized prototypes compare the complete reduction call on the same
  massive Euclidean N2/N3 rank0/1 inputs. *B* is baseline TIRust with Symbolica
  masters; *S* is the same TIRust prototype selecting native A0/B0/C0 for plain
  f64. IREGI uses Hybrid and COLLIER uses COLI mode 1. Every ratio below has
  TIRust in the numerator; all forty results remain visible.

  #figure(kind: table, caption: [*Same-cohort paired latency ratios and 99% CIs.*
    B and S denote TIRust variants, not separate reference libraries.
    Below one favors the numerator. Amber preserves the short-block family.],
    report-table(table(
      columns: (1fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr),
      align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
      inset: (x: 5pt, y: 10pt), stroke: table-stroke,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header([Family / batch], [TIRust B /\ IREGI], [TIRust S /\ IREGI],
        [TIRust B /\ COLLIER], [TIRust S /\ COLLIER], [TIRust S /\ TIRust B]),
      ..for row in joint-direct.snapshot.rows {
        let label = if row.id.starts-with("n2") { "Bubble" } else { "Triangle" }
        let rank = if row.id.contains("r0") { 0 } else { 1 }
        ([*#label r#rank*\ batch #row.batch], ..row.ratios.map(joint-ratio-cell))
      },
    )),
  )

  *Selected versus COLLIER: seven regressions, one inconclusive family.*
  Selected versus IREGI has six regressions, one gain (triangle rank 0 singleton)
  and one inconclusive family. Selected versus baseline has seven gains and
  one inconclusive family. Baseline/reference comparisons each retain seven
  regressions and one inconclusive family. No production/default adoption or
  complete performance acceptance follows.

  *Explicit posthoc analysis revision2.* The original frozen analyzer marked
  five ratios short-block-inconclusive and 35 cohort-failed because one local
  short block set the aggregate status. A separately reviewed saved-data
  correction verifies clean termination and every original global check before
  assigning family gates. All forty sample arrays, medians, CIs, widths and
  numerical order-stratum statistics are unchanged; the conservative verdict remains
  archived. No new timing, threshold change, omitted family or repeated cohort.

  One COLLIER block in bubble rank 0/batch 1 lasts 99.893971 ms, below 100 ms;
  all five ratios of that family remain inconclusive. Protocol: CPU 254, ten
  warm calls, 48 quartets per family containing all 24 permutations of the four
  engines twice, shared fixed iterations, 100 ms blocks, 10,000 paired bootstrap
  resamples, 99% CIs and 2% width gate. Every complete tensor/Laurent array and
  native pole probe passes before and after timing. Transport is excluded.
  This is a source-bound historical prototype, not the current production
  build. Sources: #raw("validation/selected-native-four-engine/v3-frozen/") and
  #raw("validation/selected-native-four-engine/posthoc-revision2/").
]

#if collier != none [
  #pagebreak()
  = Archival reference timings: IREGI and COLLIER
  Identical massive Euclidean tensor integrals compare original IREGI Hybrid
  (mode 2) with COLLIER 1.2.10 COLI (mode 1). Each row requests the complete
  Cartesian tensor array and all four native Laurent slots. This is a
  reference-library comparison, separate from the TIRust performance gate.
  No TIRust/COLLIER ratio is inferred from different sessions. The direct
  four-engine cohort has its own separate source-bound table.

  #figure(kind: table, caption: [*Archival binary64 reference timings, µs/point.*
    Batch 1 is latency; batch 32 is amortized cost per point.
    Qualified comparison outcomes are retained without reference/reference
    ratio displays.], report-table(table(
    columns: (1.15fr, 0.55fr, 1fr, 1fr, 1.65fr),
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    inset: (x: 7pt, y: 9pt), stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([Topology / rank], [Batch], [IREGI\ Hybrid], [COLLIER\ COLI], [Qualified outcome]),
    ..for r in collier.rows {
      ([*#r.label*\ rank #r.rank], [#r.batch_width],
        reference-time-cell(r.iregi_us_per_point),
        reference-time-cell(r.collier_us_per_point), collier-cell(r))
    },
  )))

  #text(size: 9pt)[The 99% interval width must be at most 2% of the paired
  median, and every block must reach 100 ms. Amber rows remain inconclusive.
  Raw ns/batch values are divided by $1000 times "batch width"$ for this display.
  Original paired ratios and intervals remain archival metadata in the JSON.]

  *Numerical scope.* #collier.correctness_note

  *Cache policy.* #collier.cache_policy

  #text(size: 9pt)[COLLIER timing includes invariant setup and native conversion;
  IREGI includes mass setup and cache disposal. Both include the full tensor
  call and status checks.]

  *Measurement protocol.* #collier.protocol_note

  Ordinary non-dotted, fixed-dimension cases have zero IR poles; regulator
  probes are untimed. This archival timing cohort does not qualify higher
  points/ranks, singular or IR-divergent families or physical numerators.
  Raw observations and reproduction:
  #raw(collier.source).
]

#if shared-collier != none [
  #pagebreak()
  = Extended reference coverage: N2–N7
  A separate correctness-only capture extends the ordinary, massive Euclidean
  reference matrix. Each cell keeps the native COLLIER admission status;
  numerical output from a rejected call cannot qualify a comparison. This
  table contains no additional timings and no new TIRust performance claim.

  #figure(kind: table, caption: [*All 34 reference rows, including failures.*
    COLI-admitted means native status accepted and the complete native
    tensor/Laurent array agrees with IREGI within $10^(-10)$.
    This table does not establish independent higher-point identities.],
    report-table(table(
      columns: (1fr, 1.4fr, 1.4fr),
      align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
      inset: (x: 7pt, y: 5pt), stroke: table-stroke,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header([Propagators / rank], [Batch 1], [Batch 32]),
      ..for row in shared-collier.snapshot.rows.filter(r => r.batch_width == 1) {
        let paired = shared-collier.snapshot.rows.find(r => r.id == row.id and r.batch_width == 32)
        ([*N#row.n / rank #row.rank*], reference-coverage-cell(row), reference-coverage-cell(paired))
      },
    )),
  )

  *17 COLI-admitted, 17 unqualified rows.* The six N5 rank 0–2 rows
  have maximum normalized disagreement $4.76 times 10^(-12)$; the N6
  rank 0 singleton reaches $2.60 times 10^(-13)$. Existing N4 rank 1–3
  rejections remain visible. Other higher-point calls retain COLI accuracy
  flags −1/−2 and DD error flag −10. Both policies and all raw statuses
  remain in the evidence; library code and rejection criteria are unchanged.

  Attempt004 retained 239 validated isolated responses. Attempt005 appends
  only its eight missing requests, with no repeated capture or timing. The
  final N7 rank 2/batch 32 case retains unstable original PV, a nominally
  successful IBP result disagreeing with Hybrid by 0.520 normalized, and
  rejected COLLIER outputs. The N6 rank 2 mass-deformed IREGI disagreement
  of 1.97 also remains unresolved.

  The independent saved-artifact/lane-gather audit verifies 81,086 assertions,
  54,224 admitted native complex pairs and 5,427 rejected null slots.
  This is distinct from independent mathematical identities. IR-divergent,
  dotted and shifted-dimension COLLIER coverage is unqualified. Sources:
  #raw("validation/collier-reference/shared/attempt004/") and
  #raw("validation/collier-reference/shared/attempt005/").
]

#if n5-identities != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Independent checks and retained failures
  == Massive N5 reference identities
  Three retained positive-F massive, undotted N5 branches at ranks 0–2 pass
  independent prefix, pinch/denominator/trace and cyclic/reversed-routing checks.
  Batch 32 cycles those same three branches; it does not add 32 distinct
  kinematic configurations. These synthetic checks use the reference codes,
  separately from TIRust reduction, physical numerators and MadLoop.

  #report-table(table(
    columns: (1.6fr, 1fr, 1fr, 1.4fr), inset: (x: 7pt, y: 9pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
    table.header([Independent check], [Residual records], [Complex slots], [Maximum normalized error]),
    [Rank prefixes], [924], [3696], [0],
    [Pinch / denominator / trace], [756], [3024], [$3.135 times 10^(-12)$],
    [Cyclic / reversed routing], [360], [1440], [$4.101 times 10^(-12)$],
    [Finiteness / pole zeros], [3357], [10071], [0],
  ))

  All 5397 residual records / 18,231 complex comparisons pass the unchanged
  $10^(-10)$ gate. The saved-data audit independently verifies 511,509
  assertions and 346 archive members, including 465 admitted native lanes and
  13,428 unique native complex pairs. The earlier capture's audit-pending
  status is immutable; the separate completed audit supplement supplies this
  result. No higher-rank, singular, dotted, dimension-shift, physical or timing
  qualification follows. Source:
  #raw("validation/collier-reference/shared/identities006/correctness001/audit001/").

  #if n7-exact != none [
    == Private N7: exact tests only
    A frozen private N7 source build passes 17 exact tests. Its ignored
    generic variant-constructor test remains unrun. Source/build/test receipt review is distinct
    from numerical integral evaluation: no N7 full-array agreement, precision
    convergence, public adoption or performance is established. Source:
    #raw("validation/highpoint-thin-n7-exact/").
  ]

  #if rank7-failed != none [
    == Private N6 rank 7: failed Float128 comparison
    The retained Float128/requested-53 singleton returns 330 moments / 1320
    native complex Laurent slots, including 990 exact zero pole slots. Its
    unchanged original-IBP comparison fails 90 finite slots: 12 at rank 6 and
    78 at rank 7. Maximum normalized error is $2.539 times 10^(-9)$ against
    $10^(-10)$. A completed process and reported stable status do not qualify
    the complete array.

    For this real positive-F point, the independent analytic real-zero identity
    has no TIRust violations; original IBP and Hybrid each violate it in 88
    slots. This explains 88 of the failed original-IBP complex comparisons from the
    reference real residual alone. It does not certify the remaining imaginary
    values or overturn the failed complete-array gate. Original PV stays
    explicitly unstable. Two saved quadrature witnesses are limited diagnostics,
    without rigorous error bounds or a causal cache-bug conclusion.

    A separate Float256 diagnostic saves the same 1320 slots and 990 exact zero
    poles. Across Float128/256, the maximum observed normalized difference is
    $5.421 times 10^(-32)$, below $2^(-80)$. Original IBP, PV and Hybrid still
    disagree in 90, 151 and 109 slots respectively; unstable PV remains
    unqualified. This observed refinement is diagnostic: complete-array
    accuracy and precision-convergence acceptance remain false, with no
    high-rank adoption, lifecycle or timing result.
    The immutable saved-data packet and its initial missing-file failure are
    preserved at
    #raw("validation/checkpoint-audits/2026-10-24-thin-n6-rank7-128/").
    The separate terminal capture and independent audit are at
    #raw("validation/highpoint-thin-n6-rank7-phase/").
    #if rank7-oracle != none [
      #parbreak()
      *One-component quadrature diagnostic.* For moment 126, $q_0^6$, six
      fixed-rule calculations (orders 12/16/20, two maps) give
      $(0, -0.05166894942216114)$. The estimated normalized uncertainty is
      $3.113 times 10^(-15)$ and is nonrigorous. Discrepancy plus that estimated
      margin is $3.249 times 10^(-15)$ for TIRust Float128, within the
      $10^(-10)$ diagnostic gate; original IBP and Hybrid give
      $4.274 times 10^(-10)$ and $4.267 times 10^(-10)$, outside it.
      PV remains natively rejected. This supports one finite component;
      complete-array acceptance stays failed. Source: #raw(rank7-oracle.source).
    ]
  ]
]

#pagebreak()
= Historical kernel snapshot: #raw("80949db")
All eight massive bubble/triangle checks pass the unchanged $10^(-10)$
criterion, maximum normalized error $2.46 times 10^(-13)$. Five timing rows
regress; three exceed the unchanged 2% interval-width limit and remain
*inconclusive*. This optimized snapshot includes the adopted master stack
allocation, generalized PV and lazy Cayley solve. Later core-allocation and
experimental changes are excluded. The performance requirement remains unmet.

#figure(kind: table, caption: [*Binary64 hybrid interpreter, µs/point.*
  Latency at batch 1 and amortized cost at batch 32. Ratios are
  *TIRust / IREGI*, using the same-run paired original-IREGI measurement.], report-table(table(
  columns: (1fr, auto, 2fr, 2fr),
  align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
  inset: (x: 7pt, y: 10pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case], [Rank], [Batch 1], [Batch 32]),
  ..for r in historical809.rows.filter(r => r.batch_width == 1) {
    let batch = historical809.rows.find(x => x.id == r.id and x.batch_width == 32)
    ([*#r.label*], [#r.rank], recent-cell(r), recent-cell(batch))
  },
)))

== Separately instrumented component timings
#report-table(table(
  columns: (3.5cm, 10.5cm, 1.5cm), inset: (x: 3pt, y: 3pt), stroke: none,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Case / batch], [TIRust stage time: 0 #h(1fr) 25 #h(1fr) 50 µs/point], [Total]),
  ..for r in historical809.rows {
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
reference caches cleared per sample. Transport is excluded. rustc 1.99 uses
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
  Latency at batch 1 and amortized cost at batch 32. Ratios are
  *TIRust / IREGI*, using the same-run paired original-IREGI measurement.], report-table(table(
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
  table.header([Case / batch], [TIRust stage time: 0 #h(1fr) 20 #h(1fr) 40 µs/point], [Total]),
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
numerical results are recomputed. Transport is excluded. rustc 1.99 uses
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
  ms/point, TIRust / MadLoop.* Repeated captured point; amortized batch32 cost does not establish
  throughput over varying phase-space samples.], report-table(table(
  columns: (1.2fr, auto, 2fr, 2fr), inset: (x: 7pt, y: 10pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Graph], [Ranks\ TIRust/ML], [Batch 1], [Batch 32]),
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
Production snapshot #raw("06706e2") and private experiments have separate source
and binary identities. Correctness evidence does not replace timing gates.

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
The retained original-IREGI dotted-integral comparisons cover positive
propagator powers and dimension shifts; they are separate from COLLIER's
ordinary undotted slice. Generalized PV for $N<=4$ passes nine focused tests
and 36 original-IREGI comparisons across 312 complex slots; maximum normalized error is
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
  128/256 bits, including an independent coefficient. Binary64 TIRust and
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
  The original PV branch is retained unchanged; the relevant TIRust comparisons
  use original IBP and the independent parameter-integral result.]

#text(size: 8.5pt)[The dedicated original-IREGI/COLLIER table covers ordinary,
non-dotted N2/N3 rank0/1 inputs, batches1/32. Its numerical and timing results
do not qualify broader N4–N7, singular or IR-divergent families.]

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

Private slim RealBall passes 88 core checks, eight kernel and 15 typed cases,
two ggH packets, and 10 separate planar-interaction checks with retained
dependencies. Its first four timing rows give two noisy real-lane results,
one complex-lane parity and one gain. The longer cohort is reported separately;
the candidate remains unadopted.

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
pass; the separately qualified explicit planar API is recorded below.

N6 inline 128/256-bit arrays converge over 1320 slots within
$6.46 times 10^(-29)$; N6 rank7 and N7 arrays remain unqualified. The later
restricted public N6 rank2/3 qualification is recorded separately below. Adopted
alias-use collection preserves 29 paired cases, 86 successful lanes and 4520
components. Frozen collector-plus-sum and private amplitude-variable probes
reach the 600 s bound (601.232/601.329 s with teardown), after primal amplitudes
but before master activations finish; neither produces arrays. The latest
private variant passes 48 variable-set, 34 amplitude/error and 73 semantic checks.
Earlier timeouts remain in #raw("validation/public-highpoint/"); no timing gate follows.

#pagebreak()
= Physical fixture qualification
#report-table(table(
  columns: (1.3fr, 1.3fr, 1.3fr), inset: (x: 7pt, y: 9pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Single graph], [Pointwise numerator], [Integrated TIRust result]),
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
reference-only qualification; later TIRust checks belong to the separate joint run.
Equivalent full-graph timing retains TIRust/MadLoop ranks 2/3 for ggH, 4/4 for
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

The 80949db massless up-quark box passes all 70 moments / 280 slots against
each original mode: maximum error
$2.85 times 10^(-14)$ against IBP and $7.11 times 10^(-15)$ against PV/hybrid.
MadLoop finite/single/double coefficients agree within $6.74 times 10^(-15)$
with fixed-color factor $-1/8$. Its one-run wall time is unqualified; see
#raw("validation/physical/gggg-up-cli-comparison.json").

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
  columns: (1fr, 2.7fr), inset: (x: 6pt, y: 4pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Unresolved item], [Observed evidence and remaining check]),
  [Binary64 baseline], [Historical TIRust/IBP errors $2.23 times 10^(-10)$ /
    $3.40 times 10^(-10)$ fail. Rank-six rescue passes at the retained point.],
  [Pentagon coverage], [Rank-six rescue passes 840 slots; saved rank-seven
    baseline passes 1320. Later precision/routing edits and broader domains
    require qualification.],
  [Physical pentagon], [Float 128/256 agrees with PV/hybrid and MadLoop;
    original IBP disagrees. Binary64 and all-mode acceptance remain incomplete.],
  [Shifted scalar], [Thirteen standalone checks and public f64 fixtures pass.
    Production small-routing E6 remains Unstable at requested 128. New private
    mixed-jet checks cover common masses/zero invariants only; broader domains
    and production integration remain unqualified.],
  [Derivative domains], [Five-case mixed D0 checks pass $2^(-70)$.
    Thresholds and unsupported transverse massless/Gram/Cayley domains retain errors.],
  [Physical top box], [Historical binary64 guard rejection remains visible;
    Float 128/256 and preserved 80949db binary64 checks pass at retained points.],
  [Supplied-product N6], [Certified scalar working 128/256, requested 53 passes
    with zero poles and invalid-lane isolation; binary64 rejects. Independent
    difference $1.53 times 10^(-23)$ is below estimated (nonrigorous)
    $6.11 times 10^(-22)$. N7/tensors/mixed Dual remain pending.],
  [Legacy and precision], [All 30 retained cases pass: 22 primary and 8 rescues.
    Three false-success cases are repaired; conservative active-master error
    estimates remain noncertified. Broader domains remain unqualified.],
  [Performance], [Historical 4c32 interpreter: 7 regressions/1 gain; native:
    5 regressions/1 parity/2 gains. 5c9: 6 regressions/2 noisy; 80949: 5/3.
    Latest 6b24: 7/1. Physical: six noisy/two failures. Overall acceptance is unmet.],
))

== Optimization and experimental evidence
#report-table(table(
  columns: (1fr, 3fr), inset: (x: 6pt, y: 4pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Candidate], [Executed evidence and decision]),
  [Adopted allocation / guards], [Core allocation: 67 current opt2 checks;
    quiet paired run preserves 92 outputs. Guard context: 76 checks and 120
    snapshots, 14 cleanup checks; six isolated gains/six noisy rows.
    Combined 6b24 optimized IREGI gate remains seven regressions/one noisy row.],
  [Roundoff cache], [29/37 stability checks and 120 snapshots pass.
    Candidate rejected: two regressions/ten noisy rows. Separate accuracy-guard
    diagnostic: two regressions/six noisy rows.],
  [CLI coefficients], [72 ordinary/12 mixed batches pass. Eight opt2 timing rows:
    five gains/one parity/two noisy; ggHH32 ratio 1.0367, CI 1.0237–1.0501 retains
    possible regression; unadopted. CSE proves 892 coefficient/223 homogeneity
    identities. 1281 raw failures coexist with zero dimension-normalized
    diagnostic failures; no physical acceptance or new timing claim.],
  [Native candidates], [Dense f64 B0/C0: two gains/six singleton parities/four
    batch32 regressions. D0 allocator: three singleton gains/two batch32
    regressions/one noisy. Shared B0/C0 kernels: one gain/two parities/seven
    regressions/two noisy. All rejected; D0 singleton source is unchanged.
    Separate actual A0–C0 replay passes 1782 outputs from 297 rows at $10^(-10)$;
    native/interpreter values are not bitwise identical. New replay timings
    are reported separately.],
  [Certificate candidates], [Slim RealBall module: 23 checks, including 8
    baseline controls. Separate exact-distance receipts: 49 checks and 120 exact
    snapshots against the same private RealBall baseline; argument rows unchanged.
    Fourteen timing cases give eight gains/two parities/four noisy, no qualified
    regression. Private final integration passes 12 receipt/6 accounting checks,
    counted once with the automatic-planar followup. The final timing cohort
    is reported separately.],
  [N5 planar API], [Promoted explicit unit-power API: rank7 ordinary128/256
    passes 1320 slots/convergence; mixed128 passes 5280 slots per accepted lane
    versus256 boxes, requested53. Transverse jets reject; 9 focused/49 ordinary
    checks and 20 Fortran-leaf coefficient cases pass. Earlier timeout remains.
    Private automatic rank1/2 dispatch passes 69 checks; the final 18-test followup
    is separate. No automatic rank7, production fallback or timing claim.],
  [Finite D0 certification], [Explicit API promoted: three actual public tests
    and current core Cargo checking pass. Exact represented inputs receive
    component-budget enclosures in the checked near-vacuum domain; upstream
    routing and quadrature errors are excluded. Prior typed targets are f64/40,
    DD/80, Float128/100 and Float256/128. Ordinary evaluation and default E6
    remain unchanged; no automatic certified-master dispatch.],
))

#if latest-experiments != none [
  #pagebreak()
  = Latest isolated optimization cohorts
  These private candidates were measured in one coordinated quiet window on
  CPU 254. Each has its own frozen sources, binaries and registered protocol;
  results do not qualify a combined implementation or replace the TIRust/IREGI
  performance gate. Candidate effects are never multiplied. Failed controls,
  inconclusive rows and earlier cohorts remain in the evidence.

  #report-table(table(
    columns: (1.1fr, 1.25fr, 2.65fr), inset: (x: 8pt, y: 12pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { pale },
    table.header([Private cohort], [Per-family outcomes], [Controls, scope and decision]),
    ..for row in latest-experiments.rows {
      ([*#row.label*], [#row.outcomes], table.cell(fill: amber)[#row.decision])
    },
  ))

  Ratios in these cohorts are *TIRust(candidate) / TIRust(baseline)*,
  using each cohort's own implementation and baseline. Every measured family is retained in the linked JSON,
  with raw paired observations, 99% confidence intervals and unchanged
  qualification thresholds. Project compute was held, but unrelated host
  activity remained uncontrolled. Earlier measurements use different frozen
  sources or protocols and remain separate. No new production adoption is
  implied by this page.

  == Exact and source-only research
  Additional Symbolica CSE leaves A0/B0/C0 exports unchanged: 43/506/223
  instructions and 6/13/8 root slots. Eighteen root identities and 470 control
  checks pass under the structured-SSA assumption. With no instruction saving,
  the candidate stops before numerical or performance qualification.

  The E6 design notes remain proposals. The initial private direct N2 prototype
  at Float256/request128 certifies 16 components across four successful lanes
  with one-axis common-mass jets; five expected rejections hide values.
  Independent audit: 513 checks and 140 hashes pass. Its later typed followup
  appears separately below. Neither snapshot qualifies public/default E6,
  general geometry or performance.

  A separate private rank-lowering N6 rank-one candidate passes all 20 complex
  Laurent slots against original IBP at Float128/request53, maximum normalized
  error $3.17 times 10^(-13)$. The old public path rejects nonfinite coefficients,
  so this is not paired old/new numerical agreement. Exact constraint tests
  accompany the frozen development build. Its rank-two probe rejects nonfinite
  coefficients in 57.823 s, within the 180 s bound, without arrays. Rank-seven,
  N7, convergence and performance qualification remain pending.

]
#if lazy-cohort != none [
  #pagebreak()
  = Lazy proof capture: complete paired comparison
  *TIRust(lazy) / TIRust(eager)* measures the entire reduction call, including
  result construction and disposal. Variants share the eager-final baseline,
  current master adapter and unchanged optimized numerical dependencies; only
  receipt-capture allocation changes. Symbolica masters remain selected; N5
  retry is inactive.

  #figure(kind: table, caption: [*All fourteen workloads are retained.*
    Ratios below one favor lazy capture. Each cell shows the paired latency
    ratio, its 99% bootstrap interval and the unchanged per-family verdict.],
    report-table(table(
      columns: (1.2fr, 1.5fr, 1.5fr), inset: (x: 8pt, y: 12pt),
      stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { none },
      align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
      table.header([Workload], [Batch 1], [Batch 32]),
      ..for singleton in lazy-cohort.snapshot.performance.results.filter(row => row.rows == 1) {
        let batch = lazy-cohort.snapshot.performance.results.find(row => row.workload == singleton.workload and row.rows == 32)
        let labels = (
          "bubble-mixed-dual": "Bubble · mixed Dual",
          "bubble-ordinary": "Bubble · ordinary",
          "bubble-zero-complex": "Bubble · complex / exact zeros",
          "bubble-zero-real": "Bubble · real / exact zeros",
          "ggh-mixed-complex": "ggH · mixed complex",
          "ggh-zero-real": "ggH · real / exact zeros",
          "triangle-ordinary": "Triangle · ordinary",
        )
        ([*#labels.at(singleton.workload)*], lazy-ratio-cell(singleton), lazy-ratio-cell(batch))
      },
    )),
  )

  *Five gains, six parity results and three qualified regressions; no noisy
  intervals. No adoption.* Regressions occur for the real exact-zero bubble
  at batch 32, real exact-zero ggH at batch 32, and ordinary triangle singleton.
  Every 99% interval meets the fixed 2% width gate. The all-family no-regression
  criterion fails; all families remain retained.

  Fourteen optimized preflight pairs preserve 10,560 unique complex output slots
  and exact status/metadata/context/counters. The earlier 26 debug tests remain
  separate. CPU 254; five warm calls; 41 paired blocks with 21/20 alternating
  order; retained calibration ≥200 ms on the faster side; 20,000 paired bootstrap
  resamples; all 1,148 observations retained. Lane/output checks precede and
  follow timing; iterations unwrap the top-level result. The host is shared.

  Scope: N2/N3 f64 and flattened mixed Dual only; no new TIRust/IREGI or
  TIRust/COLLIER gate. Earlier eager/core and slim-RealBall cohorts remain
  separate; their ratios cannot be multiplied. Source:
  #raw("validation/lazy-receipt-performance/").
]
#pagebreak()
= Additional bounded correctness checks
These completed checks extend the evidence ledger without changing production
or any measured timing row. Each private candidate retains its own source and
dependency snapshot; the raw-master replay gains do not establish whole-reduction
performance for a different candidate.

#report-table(table(
  columns: (1.1fr, 2.8fr), inset: (x: 8pt, y: 12pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([Bounded study], [Observed result and remaining limit]),
  [Selected native A0–C0], [Eight ordinary f64 N2/N3 rank0/1 families at batches
    1/32 pass all 1584 complete tensor coefficients per backend against original
    IREGI and the Symbolica baseline. Maximum normalized errors are
    $3.38 times 10^(-13)$ against IREGI and $2.07 times 10^(-13)$ between backends.
    The 297 actual master rows retain all 1782 six-output coefficients.
    Private opt0 build; whole-call performance is unmeasured.],
  [Lazy proof capture], [26 tests pass: 12 receipt, 8 lifecycle and 6 accounting
    checks. Fourteen paired workloads match exact values, status, metadata and
    counters through seven calls per implementation. Capture is created only
    when evidence is needed and discarded after each call. This private debug
    qualification is separate from the later optimized timing table; no adoption.],
  [COLLIER N4 extension], table.cell(fill: amber)[Eight fixed massive cases cover
    ranks0–3 at batches1/32. Both scalar rows are accepted, maximum normalized
    error $2.76 times 10^(-14)$. All six higher-rank COLI rows retain
    #raw("AccFlag=-1"), despite raw agreement within $2.39 times 10^(-12)$.
    DD mode returns #raw("ErrFlag=-10") for all eight; those arrays are invalid
    reference evidence. No inputs or thresholds changed; no timings were run.],
  [Fused core CSE], [The N2 rank1 native/OneLoop exports remain byte-identical:
    1013/850 instructions and 237/147 slots. All 422 exact roots and 730 control
    checks pass under the structured-SSA assumption. With no structural saving,
    the candidate stops before numerical or timing work; other topologies
    remain untested.],
))

== Scope of the independent checks
The selected-native study also passes 34 adapter cases, ten unchanged master
paths covering 33 cases and three unchanged whole-core paths. Repeated whole
responses are retained; direct adapter repeats are executable assertions with
the common output saved. Initial build and fixture failures remain in the
package. Native D0, high-point tensors and a production backend switch are outside
this qualification.

The lazy-capture allocation occurs inside the existing argument-stage timer.
The earlier eager allocation was outside that stage; the later paired comparison
therefore uses total latency. The earlier final receipt cohort and its two
qualified regressions remain unchanged.

N4 checks use 238 Symbolica-prepared polynomial arrays for metric traces,
pinch translation, denominator-difference signs, UV finiteness and odd-time
reflection. Their largest identity discrepancy is $2.89 times 10^(-12)$;
these constraints do not independently determine every spatial tensor
component or override a rejected library status. The initial eight measured
IREGI/COLLIER rows remain unchanged.

A retained *initial unmasked* thin-N6 rank2 probe passes seven exact/sidecar
checks, then both certified-planar retries fail among six packed N5 children;
that attempt produces no parent arrays. The later private mask followup has
separate source, numerical and lifecycle qualification below. Those private
capsules establish no production or performance gate; the later explicit
public six-point API has its own qualification below.

Historical compile/type/lint checks appear separately from later wheel004/gallery
execution. Binding and notebook sources follow the pinned Community guide in
#raw("PLAN.md").

Source bindings, raw outputs, failures and reproduction commands are retained
in #raw("docs/data/results.json"). No elapsed correctness-run duration on this
page is a benchmark. These historical records bind their named source
revisions; the optimized TIRust performance gate remains the measured
#raw("6b24c03") snapshot, without the later six-point API changes.

#if thin-mask != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Additional execution qualification
  == Private N6 rank-two mask followup
  Fourteen exact checks qualify the thin child assembly, full-shape zero-mask
  certificates, cache/context binding and route envelopes. Float128 and
  Float256, both requested 53, return all 60 native complex Laurent slots for
  the retained base point. Maximum normalized disagreement with original IREGI
  is $1.165 times 10^(-12)$ against the unchanged $10^(-10)$ gate; maximum
  128/256 difference is $1.474 times 10^(-34)$ against $2^(-80)$.

  #report-table(table(
    columns: (1.15fr, 1fr, 2.3fr), inset: (x: 8pt, y: 8pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { pale },
    table.header([Check], [Retained count], [Scope]),
    [Precision comparison], [2 × 60 slots], [One retained base point at Float128/256; requested 53.],
    [Lifecycle], [11 calls / 17 accepted lanes], [1020 complex slots; seven explicit rejection controls and two empty resets.],
    [Singleton agreement], [Zero differences], [Values, precision and signed zeros agree with separate singleton calls.],
    [Independent audit], [327 bindings / 1140 slots], [Saved-data arithmetic and source-archive audit; no kernel rerun.],
  ))

  The lifecycle corpus includes base/scaled points, mixed valid/invalid batches,
  reordering, chunks and resets. It shows isolation and absence of partial
  successful arrays; a rejected status alone does not prove the mathematical
  failure cause. The ordinary primal sensitivity estimate remains nonrigorous
  for compounded Taylor errors, and imported quadrature errors are unsupported.

  Strict scope: private FromShifts/Preserve, unit powers, no dimension shift,
  N6 rank 2, native raw components. Numerical Dual, complex masses, N7, rank 7,
  generalized inputs, production adoption and performance remain unqualified.
  The earlier unmasked rejection and the 30 s exact-test timeout with zero completed
  tests remain preserved. The later unchanged 14-test gate passes within its
  separately declared 180 s bound. Sources:
  #raw("validation/highpoint-thin-n6-mask/") and its independent checkpoint audit.

  == Private generic rank-three extension
  A separate generic assembly passes 25 exact structural/integrand/context
  tests. At the retained N6 rank 3 base point, Float128/256 with requested 53
  each return all 140 native complex Laurent slots, including 105 exact zero
  poles. Maximum original-IREGI discrepancy is $2.426 times 10^(-12)$ against
  $10^(-10)$; precision difference is $1.474 times 10^(-34)$ against $2^(-80)$.
  The exact tests cover structural identities through rank 7; numerical rank 7
  is not qualified. This separate package adds no rank 3 lifecycle, numerical
  Dual, generalized/SuppliedG, N7, public adoption or performance result.
  The initial test compile failure and historical rank 7 timeouts remain.
  Source: #raw("validation/highpoint-thin-n6-generic/").

  == Initialization-order diagnostic
  Two fresh processes retain 55 complete responses: seven control requests and
  48 case requests. All numerical admissions pass; 37 within-process exact
  comparisons have zero differences. CASE-first reproduces all 32 saved case
  responses and all three controls exactly. CONTROL-first reproduces a one-ULP
  C0 difference and changes 24 finite imaginary tensor values per backend in
  the batch 32 rank-one triangle, by at most nine ULP ($2.50 times 10^(-16)$).
  The whole-case master captures and input/status/diagnostic records remain exact.

  Registration/operand-order source facts suggest a mechanism, but no paired
  instruction exports establish the cause. This is an untimed, schedule-associated
  diagnostic, not universal determinism or an accuracy-threshold change.
  The failed v2 preflight remains unmeasured. A later four-engine cohort uses
  its separately reviewed CASE-first protocol; no diagnostic duration is a
  benchmark. Source: #raw("validation/selected-native-initialization-order/").

  #if python-checkpoint != none [
    == Python interface: historical compile and source typing
    The PyO3 standalone/native crate passes Cargo checking with one shared
    Symbolica/Numerica/Graphica/PyO3/FeynKit owner identity. Its ordinary source
    consumer passes basedpyright with zero errors and warnings. This checks
    compilation and source-facing type signatures; installed-wheel typing and
    execution require the separate runtime evidence below.
    The initial build timeout and retained mypy/upstream-stub failures are
    preserved rather than counted as successful checks.

    The initial notebook source capsule retains its original strict lint and
    nine display-helper tests. Later explorer, browser and native notebook
    checks are separately source-bound; historical results do not qualify a
    later source revision. Sources:
    #raw("validation/python-interface/checkpoint-compile/") and
    #raw("validation/python-notebooks/qualification.json").
  ]
]

#if notebook-native != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Historical Python and notebook runtime
  #if python-runtime != none [
    == Inspected native wheel baseline
    The exact baseline wheel builds, installs and passes shared-owner identity,
    packaged font/stub checks and installed-consumer typing. Of 29 focused
    tests, 28 pass; the real upstream graph renderer fails with
    #raw("painted coordinates must be numbers"). Numerical, precision, Dual,
    dotted, batch, contraction and shared tensor/graph interoperability checks
    pass independently of that rendering failure. Tests use the actual shared
    upstream types from the installed native host.

    Release-wheel performance, general mathematical-domain acceptance and full
    Community-host integration remain unqualified. Historical
    #raw("6b24c03") measurements describe an earlier kernel.
    Source: #raw(python-runtime.source).
  ]

  == Actual native notebook execution: partial result
  The inspected wheel is installed into the marimo 0.24.0 notebook environment.
  Four native test methods complete: two pass, two have three rendering-error
  occurrences. The successful controls check the dotted bubble against its
  mass Dual derivative, precision distinctions beyond binary64, failed-lane
  isolation and an edited-topology guard. Finite-difference values are
  displayed; this test adds no independent finite-difference acceptance gate.

  #report-table(table(
    columns: (1fr, 1.45fr, 1.2fr, 1.75fr), inset: (x: 7pt, y: 12pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else if y == 1 { amber } else { rgb("#d9f0d3") },
    table.header([Notebook], [Actual checks], [HTML export], [Qualified scope]),
    [01: Graph → integral], [Real renderer errors], [21 cells / 1 error], [Bubble, triangle Float and bubble DoubleFloat stop before their numerical assertions.],
    [02: Dots and derivatives], [Numerical controls pass], [19 successful cells], [Massive bubble identity, exact-input digits and invalid-neighbor masking.],
    [03: Benchmarks], [3 tests / 6 cohort choices], [11 successful cells], [Frozen evidence only; no numerical extension imported.],
  ))

  Both native exports and the original error remain archived. Rendering-dependent
  numerical assertions were unrun in this baseline. The later wheel004 followup
  below qualifies the named examples and their static graph/math browser displays. Source:
  #raw("validation/python-notebooks/native-wheel003/").

  == Benchmark explorer and readable narrow charts
  The separate explorer source revision passes strict lint and 15
  evidence/display-helper tests, including all forty saved ratios and eighty
  CI endpoints. Actual marimo execution retains every cohort and its original
  short/noisy gates. Static Chromium desktop/narrow-dark checks preserve eight
  ratio cells and render without request, HTTP, console or page failures.

  The chart now keeps a 930 px SVG inside controlled horizontal overflow,
  preserving readable labels within a 348 px narrow viewport; the document
  stays 390 px wide. Desktop cells and narrow/dark grid/chart crops were
  visually inspected. Earlier loader failure and small-chart screenshots
  remain historical. The export requires the official marimo CDN frontend and
  has no reactive backend; browser snapshots do not qualify numerical controls.
  Sources: #raw("validation/python-notebooks/chart-readable/") and its separate
  source/runtime/browser predecessors.
]


#if fixed-wheel != none and fixed-gallery != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Repaired Python interface and executed gallery
  == Exact JSON and actual upstream graph rendering
  GDB inspection of the unchanged baseline wheel finds that exact JSON's
  #raw("arbitrary_precision") feature turns coordinates serialized through
  #raw("serde_json::Value") into CBOR private-number marker maps. The actual
  numeric-coordinate decoder rejects those maps. The pinned upstream owner
  now uses typed line/cubic command tuples and typed collision-line records
  at both serialization sites. The exact-JSON feature and actual renderer
  remain in use.

  Two earlier owner-test setup/fixture failures remain archived. The final
  two consumer regression tests pass, exercising both stroke decoding and
  collision search under exact JSON. Rebuilt local wheel004 then passes all
  29 actual focused tests: 17 numerical and 12 interoperability, including
  real graph rendering, with no failures or skips. Shared-owner identity,
  packaged assets/stubs and installed consumer typing pass. This is a local
  development wheel; build durations are validation diagnostics, not timing
  comparisons. Sources: #raw(fixed-wheel.source) and
  #raw("validation/python-interface/rendering-diagnosis-003/").

  == Unchanged notebooks, fresh installed wheel
  The approved protocol freezes eleven notebook/test/graph/environment inputs
  plus the exact wheel and owner receipts. A separate fresh environment
  preserves both the failed baseline003 environment and the inspected owner004
  environment. All four native test methods pass. They exercise bubble f64,
  bubble DoubleFloat and triangle Float graph preparation, full tensor output
  shape, batching and contraction, plus the dotted bubble/Dual mass identity,
  precision distinctions, invalid-neighbor isolation and the edited-topology
  guard. Finite-difference values are computed for the display; this test adds
  no separate finite-difference accuracy threshold.

  #report-table(table(
    columns: (1.1fr, 1.1fr, 1.3fr, 2fr), inset: (x: 7pt, y: 12pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
    table.header([Notebook], [Native / evidence checks], [Successful HTML cells], [Real static browser display]),
    [01: Graph → integral], [Graph/numerical controls pass], [21 / 21], [Actual upstream bubble SVG, formulas and four contracted Laurent cards.],
    [02: Dots and derivatives], [Identity/precision/isolation pass], [19 / 19], [Formulas and four dotted-integral Laurent cards.],
    [03: Benchmarks], [3 tests / 6 cohorts], [11 / 11], [Eight ratios, retained short gate and readable horizontal chart.],
  ))

  All three strict lint checks and 15 display/evidence tests pass. One Chromium
  pass inspects each export in desktop light and narrow dark views: six views,
  1,918 normal frontend requests, no failed requests, HTTP, console or page
  errors. The mobile document stays 390 px wide; the 930 px chart pans inside
  its 348 px viewport. Actual graph, numerical cards and chart crops were
  visually reviewed. The static export uses marimo's official CDN frontend
  and has no reactive numerical backend. No numerical mock or replacement
  renderer is used. Release/platform/Community-host and full mathematical or
  performance qualification remain open. Source: #raw(fixed-gallery.source).
]

#if public-six != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Explicit public six-point reduction
  The source now exposes #raw("SixPointReductionPlan") and
  #raw("SixPointWorkspace<T>") as a separate, restricted API. It accepts six
  unit-power propagators, ranks 0–3, dimension shift zero, preserved Cartesian
  shifts, PV mode and native IREGI raw components. Unsupported settings fail
  during preparation. Ordinary reduction, hybrid dispatch and CLI defaults
  retain their existing behavior.

  #report-table(table(
    columns: (1.15fr, 1.15fr, 2.4fr), inset: (x: 7pt, y: 10pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { pale },
    table.header([Isolated qualification], [Retained count], [What was checked]),
    [Selected exact tests], [34 passed], [Assembly, context, certificates and API controls in the frozen isolated build.],
    [Existing ordinary API], [12 paired cases / 2200 slots], [42 successful lanes and 18 matching failed controls; values, precision, signed zero, metadata and counters preserved.],
    [N6 rank 3], [140 slots × 2 precisions], [Float128/256, requested accuracy 53; all 105 pole slots are exactly zero at each precision.],
    [N6 rank 2 lifecycle], [11 calls / 17 accepted lanes], [1020 complex moments and 68 contractions; seven rejection controls and two empty resets.],
  ))

  The rank-three base point agrees with original IREGI within
  $2.426 times 10^(-12)$ under the unchanged dimensionally normalized
  $10^(-10)$ gate. The 128/256-bit difference is at most
  $1.474 times 10^(-34)$ against $2^(-80)$. The rank-two lifecycle retains
  mixed valid/invalid batches, scaling, reordering, chunks and post-error reuse.
  Its contraction check preserves the requested native convention and tensor
  metadata; it does not require bitwise equality after reassociation.

  The implementation packs all six five-point children across active parent
  lanes, preserves child sensitivity diagnostics and checks complete Taylor
  cofactor constraints before dropping zero contributions. Failed lanes expose
  no partial successful coefficient array. These are implementation properties
  covered by the named tests, not a rigorous numerical error enclosure.

  #block(fill: pale, inset: 9pt, radius: 4pt)[
    *Separate current-Cargo replay.* The actual locked non-Python Cargo library
    test and example builds pass, followed by all 34 selected tests. At the
    retained rank-three point, both complete 140-slot arrays pass the unchanged
    original-IREGI and precision-convergence gates. Independent saved-data
    arithmetic verifies all 280 original comparisons and 140 convergence checks.
    The isolated qualification remains unchanged; its 12-case ordinary comparison
    and rank-two lifecycle are not reported as additional current-Cargo runs.
  ]

  Provenance binds the actual linked executables and 674 runtime inputs,
  including recorded Cargo artifacts and Rust source guards. This is not a
  complete native build-input closure: three native archives have supplemental
  post-run hashes only, without a retroactive pre/post guard claim.

  The accepted high-point numerical scope is the retained ordinary Float
  rank2/3 corpus at requested accuracy 53. Public N6 Dual, binary64/DoubleFloat,
  arbitrary complex masses, generalized powers/dimension shifts, rank 7,
  public N7 and performance remain unqualified. Existing ordinary Dual controls above
  preserve the old API; they do not qualify high-point Dual evaluation. Earlier
  failed high-point attempts and original-IREGI rank7 discrepancies remain visible.
  Sources: #raw(public-six.source) and #raw(public-six-cargo.source);
  public usage: #raw("docs/six-point.md").
]

#if current-dual != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Current-wheel C0/D0 mass-Taylor identities
  The unchanged installed local wheel004 now passes first, second and mixed
  squared-mass Taylor identities for rank-zero C0 and D0. Each family uses one
  fixed massive spacelike point, working precision 128 and 256 bits, and
  requested accuracy 53. The ordered six-component Taylor layout is
  #raw("[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]").

  #report-table(table(
    columns: (0.6fr, 0.9fr, 1fr, 1fr, 1.35fr), inset: (x: 7pt, y: 12pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
    table.header([Family], [Successful evaluations], [Laurent identity checks], [Precision checks], [Maximum normalized identity error]),
    [C0], [14 / 14], [48 / 48], [48 / 48], [$2.509 times 10^(-36)$],
    [D0], [14 / 14], [48 / 48], [48 / 48], [$3.712 times 10^(-36)$],
  ))

  All 28 evaluations succeed. The 96 identity checks compare the Taylor
  coefficients with independently prepared raised-power IBP reductions at the
  appropriate factorial normalization. Another 96 checks compare the two
  working precisions. Every check includes native UV single, IR single,
  IR double and finite coefficients; all three poles are exactly zero at
  these points. The finite coefficients remain general complex values.
  The saved-data audit checks 192 evaluated complex Laurent slots through
  9238 assertions using exact integer ratios and the original dimensionally
  normalized $10^(-10)$ criterion.

  Both families use the existing native convention and preserved inputs.
  No fallback or analytic rescue occurs. There is one numerical attempt per
  family and no retry. The original C0 saved-data audit incorrectly required
  a zero real finite part; its failed receipt is retained. The corrected audit
  checks the full complex value, with unchanged numerical output, input,
  pole tests and tolerance. No new integral evaluation was used to repair it.

  #block(fill: pale, inset: 9pt, radius: 4pt)[
    *Evidence identity.* This is wheel004 evidence, bound to native module
    #raw("cd4df512a134…") and its saved source archive. It is independent of
    the later public six-point source/Cargo check. The raised-power comparison
    and independent saved-data arithmetic are not a new external-library or
    finite-difference validation.
  ]

  Working at 128/256 bits does not claim that many achieved accuracy bits:
  the requested target is 53. These two cases do not add general static-Dual
  or DoubleFloat coverage, massless or threshold derivatives, collinear/Gram/
  Cayley boundaries, arbitrary complex-mass domains, further sample batching,
  physical/MadLoop comparison or performance qualification. Historical broader
  C0/D0 finite-difference evidence remains a separately bound source scope.
  Source: #raw(current-dual.source).
]

#if native-capability != none and n7-failed != none [
  #pagebreak()
  #set text(size: 9.5pt)
  #set par(leading: 0.5em)
  = Private capability and retained high-point failures
  == Selective native-f64 masters: correctness and metadata
  The private opt-in A0/B0/C0 backend passes 17 focused tests and a retained
  38-request capture. Per backend, the whole N2/N3 rank0/1 batch1/32 fixtures
  contain 1,584 complex tensor outputs and 1,782 complex outputs across all six
  master coefficients. The largest candidate/original dimensionally normalized
  error is $3.38 times 10^(-13)$ against $10^(-10)$. Separate selected, unchanged
  precision/Dual, cache and failure controls remain visible in the package.

  The metadata compatibility matrix now has all 16 entries passing across
  default, selective native, broad native-batch and combined features. The
  independent continuation completes only the two unfinished metadata checks;
  earlier timeouts, cancellation and the false final guard remain historical.
  Metadata checking does not execute the broad/combined numerical test binaries.
  No production/default change, performance qualification or adoption follows.
  Sources: #raw(native-capability.source) and #raw(native-metadata.source).

  == Uniform-branch scheduler: local fixture qualification
  The private Symbolica scheduler candidate passes 15 actual tests: five
  existing batch, six focused and four nested/shared-function tests. Actual
  old-batch, candidate and scalar paths are compared, including precision,
  signed zero and active instruction/lane traces for f64, DoubleFloat, mixed
  Float precision and static/dynamic mixed-order Dual shapes. Two earlier
  fixture compile failures remain archived. The patch is not applied to live
  Symbolica; dependent masters/core, full reduction and performance are not
  qualified by these local tests. Source: #raw(scheduler-fixture.source).

  == First private N7 scalar capture: no Rust numerical array
  The retained dyadic N7 rank0 fixture passes its exact-domain audit, including
  seven six-point determinants and 21 child coordinate ranks. Its generic
  structural plan times out at #raw("preparing_plan"), before workspace or
  scalar evaluation. There are zero Rust numerical slots. Cleanup completes,
  but total capture time including guards is 300.928 s, exceeding the frozen
  300 s bound; the comparison is not spawned.

  Original IBP and hybrid report stable matching finite values; PV is unstable
  and its zeros remain rejected. An independent scalar oracle finishes all six
  fixed rules (orders 8/12/16, two mappings), but its empirical normalized
  uncertainty estimate $7.696 times 10^(-10)$ exceeds $10^(-10)$. That oracle
  exits with insufficient convergence. No numerical agreement, precision
  convergence, general N7, higher rank, Dual or performance acceptance follows.
  The earlier 17 selected exact tests remain a separate structural result.
  Source: #raw(n7-failed.source).

  A separate marker-only constructor probe also times out: 110 s compute,
  116.604 s total within its 150 s bound. Its first six-line chart takes
  107.091 s inclusive, containing a 91.136 s determinant, 11.652 s inverse and
  4.297 s product; the second determinant is active at termination. These
  nested observations diagnose that probe only. They are not summed, used as
  performance comparisons or retrospectively attributed to the earlier capture.
  The constructor remains incomplete, with no workspace, masters or arrays.
  Source: #raw(data.private_n7_preparation_phase_diagnostic.source).

  A private followup uses Symbolica's #raw("RationalPolynomialField") for the
  matrix coefficient domain. All 22 selected exact tests pass; the rank-zero
  constructor completes in 58.442 s (66.660 s including guards, within the
  150 s bound). This preparation diagnostic creates no workspace, masters or
  numerical arrays. It establishes no numerical or performance gate and is
  not applied to live production. Both earlier timeouts remain unchanged.
  Source: #raw(data.private_n7_rational_domain_qualification.source).
]

#if typed-e6 != none [
  #pagebreak()
  = Precision and high-point research
  == Typed direct E6: bounded common-mass fixtures
  The unchanged private N2 direct genuine D6-pentagon engine is checked on
  common complex masses and zero invariants, with four- and six-component
  mixed Taylor shapes. Across ten groups, 65 lanes succeed and 65 intentionally
  reject; all 320 successful components satisfy the requested enclosure budget.
  Rejected lanes hide values. This extends the earlier one-axis prototype;
  it does not integrate the engine into the public reduction path.

  #report-table(table(
    columns: (1fr, 1.1fr, 1.3fr, 1fr), inset: (x: 8pt, y: 10pt),
    stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { pale },
    table.header([Base type], [Working / requested bits], [Accepted / rejected lanes], [Accepted components]),
    [f64], [53 / 40], [13 / 13], [64],
    [DoubleFloat], [106 / 80], [13 / 13], [64],
    [Float], [128 / 80], [13 / 13], [64],
    [Float], [256 / 128], [13 / 13], [64],
    [Float], [384 / 128], [13 / 13], [64],
  ))

  Independent exact audit: 24,889 checks and 166 hashes, zero failures; the root
  independently reran that audit. Inputs distinguish $2^(-80)$ where the type
  permits it, retain mixed input precisions and verify scale-jet independence.
  Static Mixed4/Mixed6 and dynamic HyperDual metadata transport use Symbolica's
  Dualizer with NativeFloat workspaces; this is *not*
  #raw("Workspace<StaticDual<T>>") qualification. Empty, odd, reordered,
  chunked, fresh, invalid-neighbor and independent concurrent workspaces pass
  compiled assertions; complete repeated arrays are not serialized.

  Float256→384 arithmetic widths shrink for 102 positive coordinates and remain
  zero for 26. This comparison removes twice the retained truncation radius;
  it does not establish convergence of the fixed N2 mixed-cubic truncation.
  Instruction-lane visits are structural estimates. Unequal masses, nonzero
  invariants in this typed extension, generalized powers/dimensions, N4,
  contour/geometry certification and performance remain outside its scope.
  Production E6 failures remain unchanged. Source:
  #raw("validation/e6-direct-n2-typed/").

  == Exact child localization and mixed-jet counterexample
  Two retained regular planar charts locate an ordinary child with Gram
  rank 2/determinant 0 and Cayley rank 4/determinant −4. Its multiplier vanishes
  at this exact ordinary input; generic symbolic aliasing still requests the
  singular PV child. This is preparation/identity evidence only: it does not
  identify the first nonfinite runtime instruction or evaluate a scalar master.

  A separate exact mixed-jet counterexample has zero primal and first
  components but nonzero mixed cofactors −1/−1 and weights −2/−1 in the two
  charts. *Primal zero is insufficient for pruning.* Any future mask must
  certify the complete requested cofactor jet while retaining planar and chart
  regularity checks. No pruning implementation, successful N6 array, production
  change or performance result is claimed by these sources:
  #raw("validation/highpoint-child-localization/").
]

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
Historical binary64 errors remain $2.23 times 10^(-10)$ for TIRust and
$3.40 times 10^(-10)$ for IREGI. Public rescue flags the primary unstable and
verifies all 840 slots at 128/256 bits plus this coefficient below $10^(-13)$.
#raw("validation/rescue-pentagon-rank6.json") retains the binary and assertions;
full arrays were not printed and its exact build-source snapshot was not saved.

The vendored Symbolica baseline passes four scope tests, 57 core tests and
all 1320 rank-seven slots at 128/256 bits, including the independent coefficient.
Its source predates later precision/routing edits; its workspace compilation
preserves 329 source/manifest hashes. The 109.44 s cold run, earlier 240 s timeout
and 115.35 s isolated run are historical diagnostics. See
#raw("validation/symbolica-production-qualification.json").

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
with zero poles. DD/Float validation takes 165.69 s, exceeding its historical
150 s bound; this is not a benchmark. Build scope is in
#raw("validation/shifted-pentagon/public-core-qualification.json"). Paired
vendored DD E6 passes exact allocation parity. Requested128 fails for the
small-routing fixture; higher/mixed Dual, broader endpoints and native E6 remain
unqualified. The original complex-box discrepancy is retained in
#raw("validation/shifted-pentagon/contours.json").

#text(size: 9pt)[A private input-enclosure extension passes five D0 pinches and
two contractions at the single exact contour point $x=2/5$, Float256/request128:
40 rectangles and 20 rounded mass slots checked. Exact unshifted geometry
remains assumed; no quadrature certificate or production E6 qualification follows.]


#pagebreak()
#include "conventions.typ"
#pagebreak()
#include "algebra-reuse.typ"


#pagebreak()
#set text(size: 9.5pt)
#set par(leading: 0.5em)
= Final private numerical checks and handoff
The following results extend the saved evidence. They do not apply the private
seven-point implementation or scheduler patch to the public default reducer,
and they introduce no new kernel timings. Earlier failed attempts, reference
instabilities and inconclusive oracle or timing rows remain unchanged.

== Seven-point scalar: one fixed ordinary lane
The private rational-polynomial implementation completes generic symbolic
preparation independently at Float128 and Float256, with requested accuracy 53.
It packs all 21 five-point children into one batch and returns all four native
Laurent slots. Every slot agrees with the retained stable original IREGI IBP
and Hybrid results at the unchanged dimensionally normalized $10^(-10)$ gate.
The original PV result remains unstable and is not counted as agreement.

#report-table(table(
  columns: (1.5fr, 1fr, 1fr), inset: (x: 8pt, y: 12pt),
  stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
  table.header([Saved numerical gate], [Complete checks], [Maximum normalized error]),
  [Original stable IBP / Hybrid, both precisions], [16 Laurent slots], [$4.783 times 10^(-12)$],
  [Float128 / Float256 convergence], [4 Laurent slots], [$9.102 times 10^(-34)$],
))
All three poles are exactly zero at both precisions. An independent audit
reconstructs acceptance using exact rational squared inequalities, without
importing the producer comparator. The full complex norm is compared: the tiny
real finite component has tracked precision 2, so these results do not claim
128/256-bit componentwise accuracy. The previous fixed scalar oracle remains
inconclusive. This is one private rank-zero case, not a qualification of N7
higher ranks, derivatives, generalized interfaces, numerical types, batches,
physical graphs or performance.

Capture totals including final provenance checks are 153.623 s and 164.176 s,
both within their predeclared 300 s bounds. They include generic preparation,
workspace construction and diagnostics and are not benchmark measurements.
Source: #raw("validation/highpoint-thin-n7-rank0-rational-numeric/qualification.json").

== Uniform-branch scheduler: complete retained dependent corpus
Separate matching Symbolica/OneLoop/masters/core chains pass 60 requests,
30 per process. Exact complete records agree across schedulers and repeated or
retained workspaces: eight ordinary N2/N3 rank0/1 batch1/32 families cover
1,584 tensor and 1,782 master complex values. Direct master controls cover
67 cases and 133 rows, including 106 successes and 27 explicit error/null rows.
All six master outputs, requested Taylor components, represented values,
precision, signed zero, diagnostics, masks and execution counters are checked.
Independent saved-data and byte-copy audits pass; earlier compiler failures
and the stricter admission-schema revision are retained.

This qualifies the frozen opt0 corpus only. The patch is not applied to
production, and neither optimized performance nor full-domain acceptance is
established. Fresh matched optimized chains are still needed: the older
optimized closure predates a current Symbolica polynomial guard. Source:
#raw("validation/uniform-branch-dependent/qualification.json").

#pagebreak()
= Resumed native-master validation
#let composed-native = data.at("native_policy_six_point_constructor_qualification", default: none)
Development resumed on 25 October 2026. The seven previously unrun constructor
cases pass in the frozen default/selective builds, with two complete test
listings and independent audits. All 646 bound inputs remained unchanged and
all nine owned process groups exited. Together with twenty earlier ordinary
cases, this completes the 27-case focused inventory across two separate runs.
The earlier twenty cases were not rerun.

#report-table(table(
  columns: (1.6fr, 1fr, 1.5fr), inset: (x: 8pt, y: 12pt),
  stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
  table.header([New constructor gate], [Complete cases], [Interpretation]),
  [Default / selective policy composition], [3 + 4 passed],
  [Construction and policy routing; no six-point integral values],
))
The bounded run used #fixed(composed-native.snapshot.new_execution.elapsed_seconds, digits: 3) s
including provenance and cleanup, within its unchanged 300 s limit. This is a
diagnostic test duration, not a kernel benchmark. Source:
#raw(composed-native.source).

== Complete native-policy tensor comparisons
Distinct baseline and selective-policy cores pass the original 38-request
corpus: 30 TIRust requests and eight original IREGI requests. The eight ordinary
bubble/triangle rank-zero/one families include singleton and batch-32 inputs.
All complete tensor and captured master arrays are checked before accepting
the policy; repeated workspaces and retained controls preserve full records.

#report-table(table(
  columns: (1.7fr, 1fr, 1.3fr), inset: (x: 8pt, y: 12pt),
  stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
  table.header([Whole-call comparison], [Complex slots / variant], [Maximum normalized error]),
  [Complete tensor arrays], [1,584], [$3.378 times 10^(-13)$ versus IREGI],
  [Captured scalar masters], [1,782], [$1.695 times 10^(-14)$],
  [Baseline versus candidate tensors], [1,584], [$2.061 times 10^(-13)$],
))
The unchanged gate is $10^(-10)$. D0, bubble derivatives, Dual and precision
controls remain exact by represented values, precision and sign. This is an
unoptimized frozen dependency context, not a current-Cargo or performance
measurement. Source:
#raw("validation/native-f64-sixpoint-whole/qualification.json").

== Retained six-point behavior under both feature configurations
All ten numerical producers and eight saved gates pass in the default and
selective combined source chains. The explicitly supported six-point parent
and children use Symbolica. Checks include 24 ordinary paired cases, 4,400
complex Taylor slots, 560 rank-three original-reference comparisons and 280
128/256-bit convergence comparisons. The largest normalized errors are
$2.426 times 10^(-12)$ against IREGI and $1.474 times 10^(-34)$ across precision.
All 420 checked pole slots are exactly zero. Twenty-two lifecycle calls cover
reordered/repeated/chunked samples, failures, empty calls and contractions;
values, precision and sign match retained singleton results exactly.

Independent numerical and provenance audits pass; all 17 owned process groups
exited. No f64/DoubleFloat high-point accuracy or public six-point Dual claim
follows. Source:
#raw("validation/native-f64-sixpoint-regression/qualification.json").

#pagebreak()
= High-point numeric pilot and open qualifications
The private rank-zero six-point pilot evaluates one massive real physical
point. Repeated and batched profiles do not increase the number of independent
physical points. Missing arrays remain failures.

#report-table(table(
  columns: (1.15fr, 1.2fr, 2.15fr), inset: (x: 8pt, y: 12pt),
  stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { none },
  table.header([Numeric domain], [Completed Laurent slots], [Outcome at unchanged bounds]),
  [f64], table.cell(fill: amber, [0 / 40]),
  table.cell(fill: amber, [All ten intended positive lanes rejected by cancellation guard; two invalid controls rejected.]),
  [DoubleFloat], table.cell(fill: rgb("#d9f0d3"), [40 / 40]),
  table.cell(fill: rgb("#d9f0d3"), [Ten successful lanes, two invalid controls and one empty call pass.]),
  [Float128 dynamic Dual], table.cell(fill: amber, [24 / 240]),
  table.cell(fill: amber, [One affine singleton passes; later schedule times out. Full domain fails.]),
  [Float256 dynamic Dual], table.cell(fill: amber, [0 / 240]),
  table.cell(fill: amber, [Work deadline reached before a complete output. Precision-pair gate has zero comparisons.]),
))
DoubleFloat agrees with original IREGI within $8.175 times 10^(-14)$ and the
retained Float256 scalar target within $6.420 times 10^(-30)$. The available
Float128 prefix has 24 complete complex coefficients, 18 exactly zero poles,
and maximum dilation-identity error $1.013 times 10^(-34)$. Two formal variables
describe one physical dilation direction; the nonlinear mixed profile remains
incomplete. Independent checks reconstruct all 64 available outputs and 1,736
exact input components, including seeds below binary64 resolution.

Each domain retained its 180-second aggregate bound and 20-second postwork
reserve; no failed attempt was retried or extended in this pilot. Its timings
are lifecycle diagnostics. Public high-point Dual accuracy, derivative error
transport, general independent derivative directions and full convergence
remain unqualified. Source:
#raw("validation/highpoint-six-point-numeric-types/qualification.json").

== Exact routing preparation
A private optimization delegates bounded affine routing matrices to
Symbolica's rational-polynomial matrix solver. All 16 selected exact algebra
tests pass, including actual generic N6/N7 scalar-compiler routing, generalized
power/dimension identities, Gaussian coefficients and explicit fallback for
opaque or nonpolynomial entries. The initial missing-environment build failure
and interrupted scheduling attempt remain recorded. This is not a full-core
numerical or performance qualification and is not applied to production.
Source: #raw("validation/routing-rational-preparation/qualification.json").

The optional native policy passes the current locked-Cargo checks detailed below.
The restricted public seven-point proposal and shared exact-master preparation
cache remain separate candidates. Matched optimized scheduler products pass
fresh numerical admission. Their timing cohort is retained below, with final
performance eligibility false because of substantial sibling-core activity.
Existing IREGI/COLLIER timings retain their contexts and TIRust-numerator ratios.
The full goal remains active: complete N6/N7 rank/power/dimension coverage,
high-point precision/Dual qualification, broader physical/exceptional checks
and performance parity per family remain required.

#pagebreak()
= Optional native policy: current Cargo integration
The Rust working tree integrates the explicit
#raw("MasterExecutionPolicy::NativeF64Abc") policy through the
#raw("native-f64-abc") feature. Oneloopmaster's generated Rust evaluators handle
ordinary f64 A0/B0/C0 requests instruction-major across active lanes. D0,
bubble derivatives, DoubleFloat, arbitrary precision and every static or
dynamic Dual layout retain Symbolica under the selective policy. Selecting
the policy without its feature returns an explicit preparation error.

Actual locked, offline Cargo builds and runtime checks pass for the default
and selective configurations, without private capture instrumentation or
substituted project archives. Current-source metadata records verify the
expected feature forwarding. The executable copies are taken from their exact
Cargo artifact records before another configuration can overwrite a shared
example path.

#report-table(table(
  columns: (1.2fr, 1fr, 2fr), inset: (x: 8pt, y: 14pt),
  stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { rgb("#d9f0d3") },
  table.header([Current configuration], [Selected exact tests], [Public five-lane triangle probe]),
  [Default], [9 / 9 pass], [Default and explicit Symbolica records agree exactly.],
  [Selective feature], [17 / 17 pass], [Default, Symbolica and native policies pass complete tensor checks.],
))
These 26 selected cases cover policy routing, ordered Dual layouts, independent
scratch, precision integrity and family exclusions. They are a focused subset,
not a claim that every project test was rerun. The N3/rank-one probe uses three
retained physical points, one invalid-scale neighbor and one repeated point.
All 400 complex Laurent comparisons against original IREGI pass the unchanged
dimensionally normalized $10^(-10)$ gate, with maximum error
$2.346 times 10^(-13)$. The 80 direct selected-versus-Symbolica comparisons have
maximum error $1.486 times 10^(-13)$. Invalid lanes contain no successful tensor
array; values, precision, signs and execution counters remain checked.

== Retained compatibility failure
#report-table(table(
  columns: (1.6fr, 1fr, 2fr), inset: (x: 8pt, y: 13pt),
  stroke: table-stroke, fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Legacy feature configuration], [Metadata outcome], [Qualification boundary]),
  [Broad #raw("native-batch")], [Timeout], [Reached the fixed 240 s frontend limit; group cleanup and unchanged sources verified.],
  [Broad + selective], [Unrun], [Later matrix step was not attempted after the timeout.],
))
The failed matrix attempt remains preserved, with no retry or extended bound.
A separate reviewed continuation admits the already-passed default/selective
metadata rows for their actual builds and numerical checks. Neither the complete
legacy matrix nor broad native runtime is qualified by this checkpoint. The
large legacy D0 evaluator received no LLVM code-generation request.

All 39 owned process groups exited. Independent source, artifact, numerical and
saved-data audits pass for the stated native-only scope. Six-point numerical
regressions remain the separate preceding capsule; no new N6/N7 domain or
performance claim follows from the Cargo checks. Backend selection is exposed
through the Rust API; CLI/Python selection is future work. Default constructors
retain their backend behavior. The explicit six-point parent and children use
Symbolica. Source and reproduction commands:
#raw("validation/native-f64-sixpoint-current-cargo/qualification.json") and
#raw("docs/native-masters.md").

#pagebreak()
= Fresh scheduler experiment: retained, ineligible
#let fresh-scheduler = data.scheduler_four_engine_fresh_2c1_performance.snapshot
#block(fill: amber, inset: 10pt, width: 100%)[
  *Final performance eligibility: false.* The experiment ran on CPU 34 while
  its SMT sibling, CPU 290, had substantial unrelated activity (69.6% non-idle
  over the enclosing observation interval). All mathematical checks pass;
  timing arithmetic and raw gates are retained. These measurements establish
  no uncontended parity or production-default improvement.
]

This private comparison pins #raw("2c1f334") and generic Symbolica masters in
the same inherited optimized dependency context for both TIRust variants.
It does not measure the later opt-in generated f64 policy. Fourteen matched
compiler products, captured and uncaptured complete-array admission, and
six-process preflight pass. The earlier linker-map admission failure remains
recorded; its five valid products were reused and nine were built subsequently.

The one-shot run retains all eight massive Euclidean bubble/triangle families,
48 balanced quartets per family, 1,768 requests and 1,600 timed replies.
Transport lies outside the timed kernel. All six processes exit successfully
and final controls pass. No short or noisy family is removed or rerun.

== Absolute kernel times
Medians in *µs per complete batch call*. For batch 32, divide by 32 for the
amortized per-sample time; the machine-readable record contains both units.
These absolute medians and the paired ratio medians on the next page are
different statistics: ratios are not calculated by dividing this table.

#report-table(table(
  columns: (1.3fr, 1fr, 1fr, 1fr, 1fr),
  inset: (x: 6pt, y: 12pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { pale },
  table.header([N / rank / batch], [TIRust baseline], [TIRust candidate], [IREGI], [COLLIER]),
  ..for r in fresh-scheduler.rows {
    let t = r.median_ns_per_complete_batch_call
    ([#r.n / #r.rank / #r.batch],
     [#fixed(t.baselineTIRust / 1000, digits: 3)],
     [#fixed(t.candidateTIRust / 1000, digits: 3)],
     [#fixed(t.iregi / 1000, digits: 3)],
     [#fixed(t.collier / 1000, digits: 3)])
  },
))

The raw candidate/baseline observations include about 6% lower time for
N2/rank0/batch32 and N3/rank1/batch32, with 2–3% singleton regressions for
N2/rank0 and N3/rank0. This mixed result does not support globally adopting
the scheduler candidate, even before considering contention. The production
scheduler remains unchanged. Higher-precision and Dual performance are not
measured in this cohort.

Reproduce the saved-evidence checks without evaluating new integrals:
#raw("python3 validation/scheduler-four-engine-fresh-2c1/verify.py").
Then run #raw("python3 validation/scheduler-four-engine-fresh-2c1-supplement-check.py")
to check the two historical nested manifests omitted by the original seal.
Machine-readable source: #raw(data.scheduler_four_engine_fresh_2c1_performance.source).

#pagebreak()
= Fresh scheduler experiment: all paired ratios
Every numerator is TIRust. *B* denotes baseline and *C* the private candidate.
Each large cell gives the paired median and 99% bootstrap interval. Color
shows the raw ratio only; *every cell is ineligible for final acceptance*
because of the shared-core context. Short blocks remain amber.

#let fresh-ratio-cell(r) = table.cell(
  fill: if r.gate == "inconclusive-short-block" { amber } else { ratio-fill(r.median) }, {
    set par(leading: 0.25em, justify: false)
    text(size: 11pt, weight: "bold")[#fixed(r.median, digits: 3)]
    linebreak()
    text(size: 6.3pt)[#fixed(r.ci99.at(0), digits: 3)–#fixed(r.ci99.at(1), digits: 3)]
    linebreak()
    text(size: 6.5pt, if r.gate == "inconclusive-short-block" { "Short block" }
      else if r.gate == "inconclusive-noise" { "Noise" }
      else if r.gate == "regression" { "Slower" }
      else if r.gate == "improvement" { "Lower time" } else { r.gate })
  },
)
#report-table(table(
  columns: (1.05fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  inset: (x: 4pt, y: 12pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { none },
  table.header([N / rank / batch], [B / IREGI], [C / IREGI], [B / COLLIER], [C / COLLIER], [C / B]),
  ..for r in fresh-scheduler.rows {
    ([#r.n / #r.rank / #r.batch], ..r.raw_ratios.map(fresh-ratio-cell))
  },
))

Raw gates: *12 regressions, 16 noise-inconclusive, 10 short-block-inconclusive
and 2 improvements*. The short families are N2/rank1/batch32 (one short
block) and N3/rank0/batch32 (50 short blocks). The unchanged minimum block
duration is 100 ms. All 384 quartets and all 40 confidence intervals remain
available, including these failures of timing admission.

Independent saved-data review reproduces the numerical admissions, complete
request accounting, quartet orders and all ratio arithmetic. A separate
environment review sets final acceptance false without rewriting raw gates.
A future cohort needs fresh placement checks and a new measurement protocol;
no extrapolation to current native-policy or high-point performance is made.

Source and full scope: #raw(data.scheduler_four_engine_fresh_2c1_qualification.source).
