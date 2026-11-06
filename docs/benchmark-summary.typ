// Slim benchmark summary: docs/benchmark.pdf. Mostly summary tables; the
// captures, raw deltas, logs and full reports live in the evidence archive
// (see `data.evidence`). Data: docs/benchmark-summary.json, written by
// scripts/build-benchmark-summary.py from one set of captures.
#let data = json("benchmark-summary.json")
#let tc = data.captures.timing.environment
#let ac = data.captures.amplitude.environment
#let report-date = tc.finished.slice(0, 10)
#set document(title: "TIRust benchmark summary", author: "Hua-Sheng Shao")
#set page(paper: "a4", margin: (x: 1.5cm, y: 1.6cm), numbering: "1 / 1",
  header: context {
    if counter(page).get().first() > 1 {
      set text(8pt, fill: luma(110))
      [TIRust benchmark summary #h(1fr) #report-date]
    }
  },
)
#set text(font: "Libertinus Serif", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.4em); block(text(13pt, fill: rgb("#1f3b63"), it)); v(0.1em) }
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
#show link: set text(fill: rgb("#1f5fa8"))
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)

#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let pending = rgb("#f0f1f3")
#let palette = gradient.linear(rgb("#5aae61"), rgb("#d9f0d3"), white, rgb("#fddbc7"), rgb("#d6604d"))
#let is-num(x) = type(x) == int or type(x) == float
#let ratio-fill(r) = if r == none { pending } else {
  palette.sample(calc.clamp(0.5 + calc.log(r, base: 2) / 4, 0, 1) * 100%)
}
#let fixed(x, digits: 2) = {
  let parts = str(calc.round(x, digits: digits)).split(".")
  let tail = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + if digits > 0 { "." + tail + "0" * (digits - tail.len()) } else { "" }
}
// Time with three significant figures.
#let us(x) = if x == none { [—] } else if x >= 100 { fixed(x, digits: 0) } else if x >= 10 { fixed(x, digits: 1) } else { fixed(x, digits: 2) }
#let sci(x) = if x == none { [—] } else if x == 0 { [0] } else {
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  let m = calc.round(x / calc.pow(10.0, e), digits: 1)
  if calc.abs(m) >= 10 { m = m / 10; e = e + 1 }
  [#fixed(m, digits: 1)e#e]
}
// A ratio with two significant figures below 0.1, two decimals otherwise.
#let rfmt(r) = if r >= 0.1 { fixed(r) } else {
  fixed(r, digits: calc.min(6, 1 - calc.floor(calc.log(r, base: 10))))
}
#let flags = data.tirust_builds.timing.rustflags
#let flags-text = if flags == none [no extra target flags] else [#raw(flags)]
#let head(body) = table.cell(fill: navy, text(fill: white, weight: "bold", size: 8.5pt, body))
#let table-stroke = (x, y) => (top: if y <= 1 { none } else { 0.4pt + luma(225) })

// A comparison cell: the paired TIRust/reference ratio (median, 99% CI) and
// the reference's time; an explicit outcome when there is no ratio.
#let vs-cell(entry, ratio, unit: "µs") = {
  let r = if ratio == none { none } else { ratio.median }
  table.cell(fill: ratio-fill(r), {
    set par(leading: 0.3em, justify: false)
    if r == none {
      let status = if entry == none { "not measured" } else { entry.status }
      text(size: 8pt, weight: "semibold", fill: luma(85))[#status]
    } else {
      text(size: 9pt, weight: "bold")[×#rfmt(r)]
      if ratio.ci99 != none { text(size: 6.5pt, fill: luma(80))[ [#rfmt(ratio.ci99.at(0)), #rfmt(ratio.ci99.at(1))]] }
      linebreak()
      text(size: 7pt, fill: luma(70))[#us(entry.us) #unit]
    }
  })
}

#align(center)[
  #text(18pt, weight: "bold", fill: navy)[TIRust benchmark summary]
  #v(-0.4em)
  #text(10pt, fill: luma(80))[#report-date #h(0.6em)·#h(0.6em) TIRust #raw(data.tirust_builds.timing.git_head.slice(0, 10)) #h(0.6em)·#h(0.6em) #tc.cpu_model]
]

Ratios are *TIRust/reference*, each measured in the same capture as paired block medians with a 99% bootstrap interval (below one: TIRust faster; green faster, red slower). No ratio is reconstructed across captures. TIRust is built in release mode (thin LTO, one codegen unit) with #flags-text (bit-identical outputs; huashengshao/TIRust\#5); IREGI and COLLIER are built with gfortran `-O3`, baseline x86-64, and NinjaRS with its own release profile. Unmeasured, skipped and failed cases are kept and named. This document keeps the summary tables only; the captures, per-point accuracy deltas and the full reports are archived at #link(data.evidence)[#data.evidence].

= Single integrals

#let timing-table(batch) = {
  let rows = data.timing.rows
  figure(kind: table, caption: [#if batch == "batch1" [Single phase-space point] else [Batch of 32 points, time per point]: TIRust (default path, automatic precision rescue) against original IREGI, COLLIER and NinjaRS (contraction with a dense random numerator, against NinjaRS's integrand reduction of it)])[
    #set text(size: 8.5pt)
    #table(columns: (auto, 1fr, 1.25fr, 1.25fr, 1.25fr), stroke: table-stroke, inset: (x: 4pt, y: 3pt), align: (left, right, center, center, center),
      head[Integral], head[TIRust µs], head[vs IREGI], head[vs COLLIER], head[vs NinjaRS],
      ..rows.map(row => {
        let b = row.at(batch)
        let label = [#row.label#if row.table == "extended" [ #text(size: 7pt, fill: luma(90))[(#row.id.split("-").slice(3).join(", "))]]]
        (label, [#us(b.tirust.us)],
         vs-cell(b.iregi, b.ratios.tirust_over_iregi),
         vs-cell(b.collier, b.ratios.tirust_over_collier),
         vs-cell(b.ninjars, b.ratios.tirust_contracted_over_ninjars))
      }).flatten())
  ]
}
#timing-table("batch1")
#timing-table("batch32")

== Original IREGI against COLLIER

Reference-only comparison on the shared undotted rows, as absolute times and outcomes from the same capture (no ratio is formed here).

#figure(kind: table, caption: [Original IREGI (its accepted mode) and COLLIER, µs per point, and their lane outcomes])[
  #set text(size: 8.5pt)
  #table(columns: (auto, 1fr, 1fr, 1fr, 1fr), stroke: table-stroke, inset: (x: 4pt, y: 3pt), align: (left, right, right, right, right),
    head[Integral], head[IREGI, 1 point], head[COLLIER, 1 point], head[IREGI, 32 points], head[COLLIER, 32 points],
    ..data.timing.rows.filter(row => row.table == "ordinary").map(row => {
      let cell(e, mode: none) = if e.us == none [#e.status] else [#us(e.us)#if mode != none [ #text(size: 7pt)[#mode]]#if e.lanes != none and e.lanes.keys() != ("success",) [ #text(size: 7pt, fill: luma(90))[(#e.lanes.pairs().map(p => p.at(0) + " " + str(p.at(1))).join(", "))]]]
      (row.label, cell(row.batch1.iregi, mode: row.batch1.iregi_mode), cell(row.batch1.collier),
       cell(row.batch32.iregi, mode: row.batch32.iregi_mode), cell(row.batch32.collier))
    }).flatten())
]

= Amplitudes

Every integral of a one-loop amplitude in one call, #data.amplitudes.at(0).lanes phase-space points per call. TIRust shares sub-results between the integrals of a call (results bit-identical to separate evaluation); COLLIER caches them. Each TIRust lane keeps its loop-rotation test and precision rescue.

#figure(kind: table, caption: [Time per amplitude call and same-capture ratios (ms per call; paired block medians). Shared: one call for all integrals (TIRust's subset store, COLLIER's cache); separate: each integral on its own (no store, no cache)])[
  #set text(size: 8.5pt, hyphenate: false)
  #set par(justify: false)
  #table(columns: (1.6fr, 0.8fr, 1.4fr, 0.9fr, 1.4fr, 1.4fr), stroke: table-stroke, inset: (x: 4pt, y: 3pt), align: (left, right, center, right, center, left),
    head[Amplitude], head[TIRust shared], head[vs COLLIER, cache], head[TIRust separate], head[vs COLLIER, no cache], head[TIRust lanes],
    ..data.amplitudes.map(a => (
      [#a.label #text(size: 7pt, fill: luma(90))[(#a.integrals integrals)]],
      [#fixed(a.ms.at("tirust-shared"), digits: 1)],
      vs-cell((status: "", us: a.ms.at("collier-cache")), a.ratios.tirust_shared_over_collier_cache, unit: "ms"),
      [#fixed(a.ms.at("tirust-separate"), digits: 1)],
      vs-cell((status: "", us: a.ms.at("collier-nocache")), a.ratios.tirust_separate_over_collier_nocache, unit: "ms"),
      [#a.tirust_lanes.successful successful, #a.tirust_lanes.best_effort best effort#if a.tirust_lanes.sharing_on_equals_off [; shared = separate, bit for bit]]
    )).flatten())
]

= Accuracy

E/A against the high-precision reference (#data.captures.stability.accuracy_metric) on the stability-study points; NinjaRS on the contracted dense random numerator. "Flagged" counts lanes the code itself marks (TIRust: lanes that used the precision rescue).

#figure(kind: table, caption: [Per-point E/A: median, 99th percentile and maximum, points worse than 10#super[−3] and points without an array])[
  #set text(size: 8pt)
  #table(columns: (auto, auto, 1fr, 1fr, 1fr, auto, auto, auto), stroke: table-stroke, inset: (x: 4pt, y: 2.5pt), align: (left, left, right, right, right, right, right, right),
    head[Integral], head[Code], head[median], head[p99], head[max], head[> 10#super[−3]], head[no array], head[flagged],
    ..data.stability.map(it => it.codes.enumerate().map(((k, c)) => (
      if k == 0 { table.cell(rowspan: it.codes.len())[#it.label #text(size: 7pt, fill: luma(90))[(#it.evaluated points)]] },
      [#c.code#if c.at("contracted", default: false) [ (contracted)]],
      [#sci(c.median)], [#sci(c.p99)], [#sci(c.max)],
      table.cell(fill: if is-num(c.at("above_1e-3")) and c.at("above_1e-3") > 0 { rgb("#fff0cc") } else { none })[#c.at("above_1e-3")],
      [#if c.no_array == none [—] else [#c.no_array]], [#if c.flagged == none [—] else [#c.flagged]],
    ).filter(x => x != none)).flatten()).flatten())
]

= Provenance

#set text(size: 8.5pt)
- Timing capture: #tc.started to #tc.finished on CPU #tc.cpu (#tc.cpu_model), load #tc.loadavg_start.join(" ") at the start; #tc.rounds.batch1 rounds at batch 1 and #tc.rounds.batch32 at batch 32. Amplitude capture: #ac.started on CPU #ac.cpu, #ac.rounds rounds.
- TIRust #raw(data.tirust_builds.timing.git_head)#if data.tirust_builds.values().any(b => b.source_tree_dirty) [ (with uncommitted changes)] else [ (clean tree for all three builds)]; binaries: timing #raw(data.tirust_builds.timing.sha256.slice(0, 16)), amplitude #raw(data.tirust_builds.amplitude.sha256.slice(0, 16)), stability #raw(data.tirust_builds.stability.sha256.slice(0, 16)) (SHA-256 prefixes), all with #flags-text.
- References (SHA-256 prefixes, flags): #data.reference_builds.timing.pairs().map(((name, r)) => [#name #raw(r.sha256.slice(0, 16))#if r.flags != none [ (#r.flags.join(" "))]]).join(", ").
- Capture summaries (SHA-256): timing #raw(data.captures.timing.summary_sha256.slice(0, 16)), amplitude #raw(data.captures.amplitude.summary_sha256.slice(0, 16)), stability #raw(data.captures.stability.summary_sha256.slice(0, 16)); archived with the build records at #link(data.evidence)[the evidence release].
