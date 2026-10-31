// Measured values live in data/results.json. null means no measurement.
#let data = json("data/results.json")
#let report-date = data.at("direct_engine_report_updated_on", default: data.at("rank6_current_report_updated_on", default: data.at("report_updated_on", default: data.date)))
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
#let facade-wheel002 = data.python_shared_facade_wheel002_qualification.snapshot
#let facade-replay002 = data.python_shared_facade_wheel002_portable_replay.snapshot.stdout
#let native-operand-vendor = data.numerica_native_operand_precision_vendor_reproduction.snapshot
#let complex-parameter-oracle = data.complex_parameter_tensor_oracle_manifest.snapshot
#let generalized-quad-six = data.n6_rank6_generalized_quad_qualification.snapshot
#let facade-wheel003 = data.python_shared_facade_wheel003_qualification.snapshot
#let facade-replay003 = data.python_shared_facade_wheel003_portable_replay.snapshot
#let low-N-generalized = data.rank6_low_N_generalized_current_manifest.snapshot
#let checked-G-unit = data.rank6_checked_supplied_G_unit_manifest.snapshot
#let current-physical = data.current_cli_physical_integration_qualification.snapshot
#let current-wheel004 = data.python_current_facade_wheel004_replay.snapshot
#let opening-examples = data.at("opening_double_precision_examples", default: none)
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

// Optimistic binary64 opening examples are a new additive cohort. Historical
// mixed-precision/strict-admission snapshots remain unchanged in results.json.
#let example-phase-colors = (rgb("#dbe8f5"), rgb("#dcefdc"), rgb("#c7d9ed"))
#let example-duration(us) = {
  if us == none { (value: "Unmeasured", unit: "", divisor: 1) }
  else if us < 1 { (value: fixed(us * 1000, digits: 1), unit: "ns / point", divisor: 0.001) }
  else if us < 1000 { (value: fixed(us, digits: 2), unit: "µs / point", divisor: 1) }
  else if us < 1000000 { (value: fixed(us / 1000, digits: 2), unit: "ms / point", divisor: 1000) }
  else { (value: fixed(us / 1000000, digits: 2), unit: "s / point", divisor: 1000000) }
}
#let example-error(error) = {
  if not is-num(error) { "unavailable" }
  else if error < 0 { "invalid" }
  else if error == 0 { "0" }
  else if error >= 1e300 { str(error) }
  else {
    let exponent = calc.floor(calc.log(error, base: 10))
    fixed(error / calc.pow(10, exponent), digits: 1) + "e" + str(exponent)
  }
}
#let example-accuracy(record) = {
  let accuracy = if record == none { none } else { record.at("accuracy", default: none) }
  if accuracy == none { [accuracy unavailable] }
  else {
    let error = accuracy.at("relative_error", default: none)
    let nonfinite = accuracy.at("nonfinite_complex_fields", default: 0)
    if accuracy.at("measured", default: false) and is-num(error) {
      if nonfinite > 0 { [finite $delta_"rel"$ = #example-error(error)] }
      else { [$delta_"rel"$ = #example-error(error)] }
    } else { [accuracy unavailable] }
    if nonfinite > 0 { [ · #nonfinite nonfinite] }
  }
}
#let example-warning(record) = {
  let accuracy = if record == none { none } else { record.at("accuracy", default: none) }
  if accuracy == none { false } else {
    let error = accuracy.at("relative_error", default: none)
    (accuracy.at("nonfinite_complex_fields", default: 0) > 0 or
      (is-num(error) and error > 0.001))
  }
}
#let example-refused(record) = {
  record != none and record.at("successful_finite_full_array", default: none) == false
}
#let example-failure(record) = {
  if record == none { "No capture" } else {
    let direct = record.at("failure_label", default: none)
    if direct != none { direct }
    else if record.at("status", default: "unmeasured") != "unmeasured" {
      record.status.replace("_", " ")
    } else {
      let attempted = record.at("attempts", default: ()).find(
        item => item.at("status", default: "unmeasured") != "unmeasured")
      if attempted == none { "No warm timing" }
      else { attempted.status.replace("_", " ") }
    }
  }
}
#let example-timing-cell(stats, phase: 0, record: none, show-accuracy: false) = {
  let measured = stats != none
  let shown = example-duration(if measured { stats.mean } else { none })
  table.cell(fill: if not measured { pending }
    else if example-warning(record) or example-refused(record) { amber }
    else if phase == 3 { rgb("#fbe7ce") }
    else { example-phase-colors.at(phase) }, {
    set par(justify: false, leading: 0.13em)
    align(center)[
      #text(size: if measured { 12.5pt } else { 9pt }, weight: "bold", fill: navy)[
        #if measured { shown.value } else { "Unmeasured" }
      ]
      #linebreak()
      #text(size: 6.6pt, fill: luma(65))[
        #if measured { shown.unit } else { example-failure(record) }
        #if measured and stats.at("standard_error", default: none) != none {
          let error = stats.standard_error / shown.divisor
          [ · SE #fixed(error, digits: if error < 0.01 { 3 } else { 2 })]
        }
      ]
      #if show-accuracy {
        [#linebreak() #text(size: 6.4pt, fill: luma(65))[
          #example-accuracy(record)
          #if phase == 3 and record != none and record.at("mode", default: none) != none { [ · #record.mode] }
        ]]
      }
      #if measured and show-accuracy and example-refused(record) {
        [#linebreak() #text(size: 6.2pt, fill: luma(70))[Refused output · diagnostic timing]]
      } else if measured and show-accuracy and example-warning(record) {
        [#linebreak() #text(size: 6.2pt, fill: luma(70))[Diagnostic output]]
      }
    ]
  })
}
#let example-input-cell(row) = table.cell(fill: pale, {
  assert(row.numeric_type == "f64", message: "Opening TIRust timings must use f64")
  assert(row.working_bits == 53, message: "Opening TIRust timings must use binary64")
  set par(justify: false, leading: 0.15em)
  text(size: 9.5pt, weight: "bold", fill: navy, row.label)
  linebreak()
  text(size: 7pt, fill: luma(65))[f64 · #row.moments #if row.moments == 1 { "moment" } else { "moments" }]
  let engine = row.at("engine", default: none)
  if engine != none {
    linebreak()
    text(size: 6.2pt, fill: navy, engine)
  }
})
#let example-powers-cell(row) = table.cell({
  set par(justify: false, leading: 0.2em)
  let powers-label = if row.powers.all(power => power == 1) { "all 1" }
    else { "(" + row.powers.map(str).join(",") + ")" }
  text(size: 8pt, powers-label)
  linebreak()
  text(size: 7pt, fill: luma(80))[$h=#row.dimension_shift$]
})
#let example-timing-table(rows, extended: false) = report-table(table(
  columns: if extended { (2.6cm, 2.0cm, 3.0cm, 3.4cm, 3.4cm, 3.4cm) }
    else { (2.7cm, 1.25cm, 3.4cm, 3.4cm, 3.5cm, 3.55cm) },
  inset: (x: 5pt, y: 4pt),
  stroke: (x, y) => if y == 0 { none } else { 0.6pt + white },
  fill: (x, y) => if y == 0 { navy } else { white },
  table.header(
    [Input integral], [Powers / shift],
    ..if extended { ([Outside direct API],) } else { () },
    [TIRust excl. masters], [Scalar masters], [TIRust total],
    ..if not extended { ([COLLIER · f64],) } else { () },
  ),
  ..for row in rows {
    let timing = row.at("timing", default: (:))
    let total = timing.at("outer_total_us", default: none)
    let exclusive = timing.at("tirust_exclusive_us", default: none)
    let masters = timing.at("scalar_masters_us", default: none)
    let reference = row.at("collier", default: none)
    if reference != none {
      assert(reference.numeric_type in ("f64", "binary64"), message: "Opening COLLIER timings must use binary64")
      assert(reference.working_bits == 53, message: "Opening COLLIER timings must use 53-bit mantissas")
    }
    (
      example-input-cell(row), example-powers-cell(row),
      ..if extended { (table.cell(fill: rgb("#f1e9f6"), {
        set text(size: 7pt)
        if row.powers.any(power => power > 1) { [Dotted propagator powers] }
        else { [Independent dimension shift] }
      }),) } else { () },
      example-timing-cell(exclusive, phase: 0, record: row),
      example-timing-cell(masters, phase: 1, record: row),
      example-timing-cell(total, phase: 2, record: row, show-accuracy: true),
      ..if not extended { (example-timing-cell(
        if reference == none { none } else { reference.at("timing_us", default: none) },
        phase: 3, record: reference, show-accuracy: true),) } else { () },
    )
  },
))
// Same-session direct-engine campaign (30 October 2026): TIRust public
// binary64 default versus original IREGI and COLLIER, interleaved per block.
// The latest bound capture of each campaign is shown; earlier ones stay in the data.
#let direct-campaign-id = ("004", "003", "002").find(id => ("direct_engine_20261030_capture" + id) in data)
#let direct-campaign = if direct-campaign-id != none { data.at("direct_engine_20261030_capture" + direct-campaign-id) } else { data.at("direct_engine_20261030", default: none) }
#let direct-first-capture = data.at("direct_engine_20261030", default: none)
#let de-sci(x) = {
  if x == none { [n/a] } else if x == 0 { [0] } else {
    let e = calc.floor(calc.log(x, base: 10))
    let m = x / calc.pow(10.0, e)
    [#fixed(m, digits: 1)×10#super[#e]]
  }
}
#let de-time(us) = {
  if us == none { [—] }
  else if us < 1 { [#fixed(us * 1000, digits: 0) ns] }
  else if us < 1000 { [#fixed(us, digits: 2) µs] }
  else if us < 1000000 { [#fixed(us / 1000, digits: 2) ms] }
  else { [#fixed(us / 1000000, digits: 2) s] }
}
#let de-label(row) = {
  set par(leading: 0.3em, justify: false)
  text(size: 9pt, weight: "bold", row.label)
  let dotted = row.powers.enumerate().filter(((i, p)) => p > 1)
  if dotted.len() > 0 or row.dimension_shift != 0 {
    linebreak()
    text(size: 7pt, fill: luma(80))[
      #for (i, p) in dotted [ν#sub[#(i + 1)]=#p ]
      #if row.dimension_shift != 0 [D=#(4 + 2 * row.dimension_shift)]
    ]
  }
  linebreak()
  text(size: 6.5pt, fill: luma(110))[#row.moments #if row.moments == 1 [moment] else [moments] × 4 slots]
}
#let de-engine-cell(record, batch, kind: "TIRust") = {
  let ok = record != none and record.accepted
  table.cell(fill: if record == none { pending } else if ok { pale } else { amber }, {
    set par(leading: 0.3em, justify: false)
    if record == none {
      text(size: 8pt, fill: luma(90))[#if kind == "COLLIER" [Outside public API] else [No accepted mode]]
    } else {
      let t = if record.timing_us == none { none } else { record.timing_us.median / batch }
      text(size: 10pt, weight: "bold", de-time(t))
      if batch > 1 { text(size: 6.5pt)[ /point] }
      linebreak()
      text(size: 6.5pt, fill: luma(70))[
        #if record.mode != none and kind != "TIRust" [#record.mode.replace("mode1-ten6-perm0", "COLI").replace("mode2-ten6-perm0", "DD").replace("mode1-ten6-perm1", "COLI, perm.") · ]
        #if ok [E/A #de-sci(record.relative_error)] else [#if not record.complete_finite [nonfinite/refused] else [E/A #de-sci(record.relative_error)] · diagnostic]
      ]
    }
  })
}
#let de-ratio-cell(ratio, accepted) = {
  if ratio == none or not accepted {
    table.cell(fill: pending, text(size: 7.5pt, fill: luma(100))[No ratio])
  } else {
    table.cell(fill: ratio-fill(ratio.median), {
      set par(leading: 0.3em, justify: false)
      text(size: 10pt, weight: "bold")[×#if ratio.median < 0.01 { fixed(ratio.median, digits: 4) } else { fixed(ratio.median, digits: 3) }]
      linebreak()
      text(size: 6.3pt)[99% CI #fixed(ratio.ci99.at(0), digits: 4)–#fixed(ratio.ci99.at(1), digits: 4)]
    })
  }
}
#let de-table(rows, batch) = {
  let key = "batch" + str(batch)
  figure(kind: table, caption: [#if batch == 1 [Single phase-space point] else [Batch of 32 points, time per point] — median of paired block means])[
    #table(
      columns: (1.05fr, 1fr, 1.1fr, 0.95fr, 1.05fr, 0.95fr),
      inset: (x: 5pt, y: 5pt), stroke: table-stroke, align: left + horizon,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header(..([Input], [TIRust (direct)], [IREGI (best mode)], [TIRust / IREGI], [COLLIER 1.2.10], [TIRust / COLLIER]).map(h => text(fill: white, size: 8.5pt, weight: "bold", h))),
      ..rows.map(row => {
        let b = row.at(key)
        let tirust-ok = b.tirust.accepted
        let iregi-ok = b.iregi != none and b.iregi.accepted
        let collier-ok = b.collier != none and b.collier.accepted
        (
          table.cell(de-label(row)),
          de-engine-cell(b.tirust, batch),
          de-engine-cell(b.iregi, batch, kind: "IREGI"),
          de-ratio-cell(b.ratios.at("tirust_over_iregi", default: none), tirust-ok and iregi-ok),
          de-engine-cell(b.collier, batch, kind: "COLLIER"),
          de-ratio-cell(b.ratios.at("tirust_over_collier", default: none), tirust-ok and collier-ok),
        )
      }).flatten(),
    )
  ]
}
// TIRust with the contraction of a dense numerator versus NinjaRS, which
// integrates that numerator directly (capture 003 on).
#let de-ninja-cell(record, batch) = {
  table.cell(fill: if record == none { pending } else if record.accepted { pale } else { amber }, {
    set par(leading: 0.3em, justify: false)
    if record == none {
      text(size: 8pt, fill: luma(90))[Outside API (rank > N+2, dotted or D≠4)]
    } else {
      let t = if record.timing_us == none { none } else { record.timing_us.median / batch }
      text(size: 10pt, weight: "bold", de-time(t))
      if batch > 1 { text(size: 6.5pt)[ /point] }
      linebreak()
      text(size: 6.5pt, fill: luma(70))[
        #record.at("ninjars_engine", default: "fast") engine
        #if record.at("fast_failed_lanes", default: 0) > 0 [(fast: #record.fast_failed_lanes Unstable lanes)] ·
        #if record.relative_error == none [no value] else [contracted E/A #de-sci(record.relative_error)]
      ]
    }
  })
}
#let de-ninja-table(rows, batch) = {
  let key = "batch" + str(batch)
  let rows = rows.filter(r => r.at(key).at("ninjars", default: none) != none or r.table == "ordinary")
  figure(kind: table, caption: [TIRust (array and contraction with a dense random numerator) against NinjaRS
    (integrand reduction of that numerator); #if batch == 1 [single point] else [batch of 32, time per point]])[
    #table(
      columns: (1.05fr, 1fr, 1fr, 1.25fr, 0.95fr),
      inset: (x: 5pt, y: 5pt), stroke: table-stroke, align: left + horizon,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header(..([Input], [TIRust (array only)], [TIRust (array + contraction)], [NinjaRS], [TIRust (contracted) / NinjaRS]).map(h => text(fill: white, size: 8.5pt, weight: "bold", h))),
      ..rows.map(row => {
        let b = row.at(key)
        let contracted = b.at("tirust_contracted", default: none)
        let ninja = b.at("ninjars", default: none)
        (
          table.cell(de-label(row)),
          de-engine-cell(b.tirust, batch),
          de-engine-cell(contracted, batch),
          de-ninja-cell(ninja, batch),
          de-ratio-cell(b.ratios.at("tirust_contracted_over_ninjars", default: none),
            contracted != none and contracted.accepted and ninja != none and ninja.accepted),
        )
      }).flatten(),
    )
  ]
}
#let direct-engine-pages(snapshot) = {
  let env = snapshot.environment
  [= Direct numerical engine: same-session comparison]
  block(fill: rgb("#d9f0d3"), inset: 9pt, radius: 4pt, width: 100%)[
    #set text(9pt)
    *What changed.* The public binary64 default (`TensorReductionPlan::new` + `TensorWorkspace`)
    now evaluates the validated PV recurrences numerically per lane with direct oneloopmaster scalar kernels,
    instead of interpreting compiled symbolic expressions. Affinely dependent sub-topologies use their
    constant-one denominator identity; a dotted propagator is a Taylor jet in its squared mass with
    master derivatives from the IBP relation; D=6 uses #raw("C(λ+h,κ)·(−2)^h"). Lanes with
    ill-conditioned systems, and for N≥5 lanes failing a loop-rotation test, are rerun by the same direct engine
    at 106/256/512 bits (see the stability section), then by the symbolic engine.
  ]
  v(0.3em)
  text(size: 8.5pt)[
    *Protocol.* All codes ran in one session on CPU #env.cpu (#env.cpu_model), one pinned core,
    load average #env.loadavg_start.join(" / ") at start. For every row and batch size each
    candidate got a fresh process; a pilot fixed the calls per block (≈20 ms); then
    #env.rounds.batch1 (batch 1) or #env.rounds.batch32 (batch 32) rounds ran one block per candidate in
    seeded random order. Ratios are medians of same-round block ratios with 99% bootstrap intervals;
    every ratio has TIRust in the numerator (green: TIRust faster). E/A is the dimensionally rescaled
    whole-array error of lane 0 versus the archived reference tensors (admitted against quadruple-precision
    original IREGI). IREGI uses its fastest mode with a complete finite array within E/A≤10#super[−3].
    COLLIER uses the archived selected public configuration. Rejected or nonfinite outputs are shown
    amber with diagnostic times and never enter a ratio. Builds: TIRust release (thin LTO), Fortran
    `-O3`, all baseline x86-64; binary SHA-256 in #raw("validation/benchmark/direct-engine-20261030/summary" + (if direct-campaign-id == none { "" } else { "-" + direct-campaign-id }) + ".json").
    TIRust source: commit #raw(snapshot.at("tirust_source_commit", default: snapshot.tirust_git_head).slice(0, 7)).
    This capture (004, 31 October) repeats capture 003 after the double-double partial rescue and the
    double-double-first precision rescue (stability section); both act only on lanes the binary64 checks
    refuse, which the opening inputs do not have, and TIRust's times agree with capture 003 to a few percent
    (separate sessions; no ratio across them). Capture 003 added the N7R7 row (the opening shifts and masses
    plus q#sub[6] = (1/4, 3/8, 1/8, 5/8), m#sub[6]#super[2] = 4; reference: TIRust at 256 bits confirmed by
    quad IREGI IBP to 6.3×10#super[−17], quad PV fails, COLLIER returns NaN, `validation/mre/collier-n7-nan`)
    and NinjaRS. TIRust's N≥5 rows include the loop-rotation test and the Gram-free pentagon reduction,
    which keep physical pentagons accurate in binary64 (capture 002, without them: N5R5 65 µs, N6R6 216 µs).
    Captures 001 (TIRust #raw(direct-first-capture.snapshot.tirust_git_head.slice(0, 7))), 002 and 003
    remain bound unchanged in the report data.
  ]
  v(0.3em)
  de-table(snapshot.rows, 1)
  pagebreak()
  de-table(snapshot.rows, 32)
  if snapshot.rows.any(r => r.batch1.at("ninjars", default: none) != none) {
    pagebreak()
    text(size: 8.5pt)[
      *TIRust and NinjaRS.* NinjaRS (#link("https://github.com/alphal00p/ninjars")[alphal00p/ninjars], a Rust port of
      Ninja's integrand reduction) integrates a given numerator, not the tensor array. Each row therefore gets a
      dense random numerator of its rank, and TIRust is timed with the contraction of its complete array with it
      inside every timed call. This is not like for like: TIRust's array serves every numerator of the topology,
      NinjaRS repeats its reduction per numerator. NinjaRS runs `case-bench` on the same lanes (pinned, one fresh
      process and ≈20 ms block per round, interleaved with the other candidates). Its fast engine, like original Ninja,
      marks cuts whose two momenta span a Euclidean plane `Unstable`; the opening inputs (spatial unit vectors) have
      such planes throughout, so these rows use NinjaRS's generic engine, as its documentation prescribes. E/A of
      NinjaRS: its lane-0 value against the reference array contracted exactly with the same numerator. On the
      N6R6 and N7R7 rows NinjaRS's generic engine fails its lane-0 cut construction (`CutFailed`, singular
      parameter; the opening points contain coplanar quadruples such as 0, e#sub[x], e#sub[y], e#sub[x]+e#sub[y]),
      so those times have no accepted value.
      Reference build: upstream 41a4c00 with the serde_json `float_roundtrip` repair (upstream pull request),
      without which NinjaRS reads about 3% of binary64 inputs one ulp off.
    ]
    de-ninja-table(snapshot.rows, 1)
    de-ninja-table(snapshot.rows, 32)
  }
  v(0.3em)
  let cross = snapshot.rows.map(r => r.batch32.tirust_iregi_cross_error_lanes).filter(x => x != none)
  text(size: 8.5pt)[
    Lanes 1–31 of each batch are binary-exact perturbations of the reference point (shifts×(256+l)/256,
    masses²+l/64). TIRust and same-session original-IREGI IBP agree on all of them with
    maximum E/A #de-sci(calc.max(..cross)). IREGI modes that STOP, refuse or exceed the accuracy gate are
    retained in the snapshot (`iregi_modes`). Reproduce with
    #raw("python3 validation/benchmark/direct-engine-20261030/build.py") and
    #raw("run.py --cpu N"), then #raw("summarize.py").
  ]
  pagebreak()
}
// Binary64 stability study (30 October 2026): cumulative accuracy profiles
// in the style of Hirschi and Peraro, arXiv:1604.01363, figs. 2-4.
#let stability-campaign-id = ("007", "006", "005", "004", "003", "002").find(id => ("stability_20261030_capture" + id) in data)
#let stability-campaign = if stability-campaign-id != none { data.at("stability_20261030_capture" + stability-campaign-id) } else { data.at("stability_20261030", default: none) }
#let stability-first-capture = data.at("stability_20261030", default: none)
#let st-style = (
  "tirust": (paint: rgb("#1a9850"), thickness: 1.7pt),
  "tirust-binary64": (paint: rgb("#1a9850"), thickness: 0.9pt, dash: "dashed"),
  "tirust-binary64-r20": (paint: rgb("#a6d96a"), thickness: 1.1pt, dash: "dash-dotted"),
  "iregi-ibp": (paint: rgb("#d7301f"), thickness: 1.1pt),
  "iregi-pv": (paint: rgb("#fc8d59"), thickness: 1.1pt, dash: "dashed"),
  "collier-coli": (paint: rgb("#2c7fb8"), thickness: 1.1pt),
  "collier-dd": (paint: rgb("#7570b3"), thickness: 1.1pt, dash: "dashed"),
  "ninjars": (paint: rgb("#e7298a"), thickness: 1.3pt, dash: "dotted"),
)
#let st-plot(integral, grid, width: 6.6cm, height: 4.3cm, codes: none) = {
  let codes = if codes == none { integral.codes } else { codes }
  let x0 = -17.0
  let x1 = 0.0
  let ymin = calc.floor(calc.log(1 / integral.with_reference, base: 10))
  let ylow = ymin - 0.3
  let px(x) = (x - x0) / (x1 - x0) * width
  let py(y) = -y / -ylow * height
  let ox = 1.05cm
  let oy = 0.45cm
  box(width: width + ox + 0.15cm, height: height + oy + 0.8cm, {
    place(dx: ox, dy: 0pt, text(size: 7.5pt, weight: "bold")[#integral.label.replace("->", "→")])
    for e in range(-16, 1, step: 2) {
      place(dx: ox + px(e), dy: oy, line(length: height, angle: 90deg, stroke: 0.3pt + luma(225)))
      place(dx: ox + px(e) - 0.4cm, dy: oy + height + 0.08cm,
        box(width: 0.8cm, align(center, text(size: 6pt)[10#super[#e]])))
    }
    for e in range(ymin, 1) {
      place(dx: ox, dy: oy + py(e), line(length: width, stroke: 0.3pt + luma(225)))
      place(dx: 0.25cm, dy: oy + py(e) - 0.16cm,
        box(width: ox - 0.33cm, align(right, text(size: 6pt)[10#super[#e]])))
    }
    place(dx: 0.12cm - height / 2, dy: oy + height / 2 - 0.15cm, rotate(-90deg,
      box(width: height, height: 0.3cm, align(center + horizon, text(size: 6pt)[fraction with E/A > Δ]))))
    place(dx: ox + px(-3) - 1pt, dy: oy, rect(width: 2pt, height: height, fill: luma(205), stroke: none))
    for code in codes {
      let pts = ()
      for (x, f) in grid.zip(code.curve) {
        if f == none { continue }
        if f <= 0 { pts.push((x, ylow)); break }
        pts.push((x, calc.max(calc.log(f, base: 10), ylow)))
      }
      if pts.len() > 1 {
        place(dx: ox, dy: oy, curve(stroke: st-style.at(code.code),
          curve.move((px(pts.at(0).at(0)), py(pts.at(0).at(1)))),
          ..pts.slice(1).map(p => curve.line((px(p.at(0)), py(p.at(1))))),
        ))
      }
    }
    place(dx: ox, dy: oy, rect(width: width, height: height, stroke: 0.6pt + luma(60)))
    place(dx: ox, dy: oy + height + 0.45cm,
      box(width: width, align(center, text(size: 6.5pt)[target relative accuracy Δ])))
  })
}
#let st-legend(codes) = {
  set text(size: 7pt)
  grid(columns: 3, column-gutter: 12pt, row-gutter: 5pt,
    ..codes.map(code => box[#box(width: 0.8cm, height: 0.5em, align(horizon,
      line(length: 0.8cm, stroke: st-style.at(code.code)))) #h(3pt) #code.label]))
}
#let st-cell(code, total) = {
  let worse = code.at("above_1e-3")
  table.cell(fill: if worse == 0 { rgb("#d9f0d3") } else if worse / total < 0.01 { pale } else { amber }, {
    set par(leading: 0.3em, justify: false)
    text(size: 8pt, weight: "bold")[#if code.median == none [—] else [#de-sci(code.median)]]
    linebreak()
    text(size: 6.3pt, fill: luma(70))[p99 #de-sci(code.p99) · >10#super[−3]: #worse/#total
      #if code.no_array > 0 [· no array #code.no_array] #if code.flagged > 0 and not code.code.starts-with("tirust") [· flagged #code.flagged] #if code.at("best_effort", default: 0) > 0 [· best-effort #code.best_effort]]
  })
}
#let stability-pages(snapshot) = {
  let setup = snapshot.setup
  [= Binary64 stability: TIRust, original IREGI and COLLIER]
  context [#metadata(here().page()) <stability-page>]
  text(size: 8.5pt)[
    *Setup* (as in the stability study of Hirschi and Peraro, arXiv:1604.01363, figs. 2–4):
    random flat phase-space points at √s = #setup.sqrt_s GeV with p#sub[T] > #setup.pt_min GeV for every
    final-state gluon and ΔR > #setup.dr_min between gluons (seed #setup.seed); top-quark loops,
    m#sub[t] = 173 GeV, m#sub[H] = #setup.mh GeV, μ = #setup.mu GeV. Every code reads the same binary64
    inputs, as a user would run it: massless legs are lightlike only to binary64 rounding
    (|p#super[2]| ≈ 10#super[−16] s), and each code treats them as exactly lightlike — TIRust by its lightlike
    tolerance, IREGI by its default on-shell snapping (as in MadLoop), COLLIER through exact zero invariants
    (passing the rounded invariants instead made COLI return NaN and DD wrong arrays on most pentagons and
    hexagons). Δ is the E/A of the complete returned array (all Cartesian moments, all four Laurent slots,
    dimensionally rescaled) against the reference. Each curve gives the fraction of points whose array is worse
    than Δ; points without an array (refused, nonfinite) count as worse than every Δ. The gray bar marks
    Δ = 10#super[−3], the usual Monte-Carlo rescue threshold.
  ]
  v(0.2em)
  figure(caption: [Cumulative binary64 stability profiles (fraction of points with E/A > Δ).
    TIRust (default) is the public binary64 API: lanes its own checks refuse (conditioning gates; for N≥5 a
    loop-rotation test comparing two binary64 evaluations) are rerun by the direct engine at 106, 256 or 512
    bits before results are returned. #if snapshot.integrals.at(0).codes.any(c => c.at("best_effort", default: none) != none) [
    "TIRust binary64 only" is the same facade with the precision rescue disabled (every lane in binary64), at the
    default accuracy request 2#super[−34] and at 2#super[−20]; a lane its checks refuse is scored by the best
    binary64 array it kept, uncertified and flagged like COLLIER's flagged lanes (table: best-effort), and counts
    as missing only without one.] else if snapshot.integrals.at(0).codes.any(c => c.code == "tirust-binary64-r20") [
    "TIRust binary64 only" is the same facade with the precision rescue disabled (every lane in binary64; refused
    lanes count as missing), at the default accuracy request 2#super[−34] and at 2#super[−20].] else [
    The second TIRust curve takes the default path's array unless the lane was rerun at higher precision; refused
    lanes count as missing. From capture 004 on, boxes and triangles refused inside such a lane are computed in
    double-double (partial rescue), so this curve is not binary64 only there.] IREGI and COLLIER are binary64 throughout.])[
    #grid(columns: 2, column-gutter: 4pt, row-gutter: 2pt,
      ..snapshot.integrals.map(integral => st-plot(integral, snapshot.grid_log10)))
    #v(2pt)
    #st-legend(snapshot.integrals.at(0).codes)
  ]
  if snapshot.integrals.at(0).at("contracted_codes", default: none) != none {
    v(0.2em)
    figure(caption: [Contracted dense random numerator: fraction of points with
      |I − I#sub[ref]|/max|I#sub[ref]| > Δ over the finite part, the (UV + IR) single pole and the double pole,
      I = Σ#sub[α] c#sub[α] ∫ q#super[α] with c#sub[α] = (u + i v)/μ#super[|α|], u, v uniform in [−1, 1], fixed per point.
      The array codes contract their arrays; NinjaRS (alphal00p/ninjars, integrand reduction, binary64 fast engine;
      lanes it marks Unstable count as without value) reduces the numerator directly.])[
      #grid(columns: 2, column-gutter: 4pt, row-gutter: 2pt,
        ..snapshot.integrals.map(integral => st-plot(integral, snapshot.grid_log10, codes: integral.contracted_codes)))
      #v(2pt)
      #st-legend(snapshot.integrals.at(0).contracted_codes)
    ]
  }
  v(0.2em)
  let codes = snapshot.integrals.at(0).codes
  block(breakable: false, figure(kind: table, caption: [Per-point E/A: median, 99th percentile and the number of points worse than
    10#super[−3] (green: none; blue: below 1%; amber: 1% or more). "No array": refused or nonfinite output;
    "flagged": COLLIER accuracy flag or IREGI non-success status (values still scored).])[
    #table(
      columns: (1.25fr,) + (1fr,) * codes.len(),
      inset: (x: 4pt, y: 4pt), stroke: table-stroke, align: left + horizon,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header(..([Integral],) .map(h => text(fill: white, size: 7.5pt, weight: "bold", h)),
        ..codes.map(c => text(fill: white, size: 7pt, weight: "bold", c.label))),
      ..snapshot.integrals.map(integral => (
        table.cell(text(size: 7.5pt)[*#integral.id* \ #text(size: 6.3pt, fill: luma(80))[#integral.with_reference points]]),
        ..integral.codes.map(code => st-cell(code, integral.with_reference)),
      )).flatten(),
    )
  ])
  text(size: 8.5pt)[
    *Findings (capture 007, TIRust #raw(snapshot.tirust_git_head.slice(0, 7))).* TIRust's default returns an
    accurate array on every point of all five integrals (worst E/A 9.6×10#super[−9] on the rank-4 box,
    ≤3.4×10#super[−10] elsewhere). "TIRust binary64 only" (precision rescue disabled) now reports every
    point with a binary64 array, as COLLIER does: lanes that pass TIRust's checks are certified, the others
    return their best binary64 array flagged with an estimated error, so its curves have no plateau. At the
    default request (2#super[−34]) binary64 certifies 9997/10000 rank-4 and 9990/10000 rank-6 boxes,
    4868/5000 pentagons, 1740/2000 hexagons and 361/495 heptagons; the flagged rest stays below
    2×10#super[−6] on pentagons and 1.7×10#super[−4] on hexagons, and one heptagon reaches
    8.9×10#super[−3] (flagged with an estimate of 4.8×10#super[−3]). Since capture 005 TIRust follows
    COLLIER further (Denner and Dittmaier, hep-ph/0509141): small-Gram expansions for triangles as well
    as boxes, the expansion for small Gram and small modified Cayley determinants (sect. 3.5), and the
    alternative PV reduction through the modified Cayley matrix (sect. 3.2), which divides by the Gram
    determinant only once per two ranks. Invariant snapping now includes the representation error of
    the shifts, and oneloopmaster's binary64 D0, which returned NaN for two opposite numerically
    lightlike legs, retries such points with those invariants at zero (upstream pull request
    alphal00p/oneloopmaster\#15). COLLIER (both branches) stays accurate everywhere without flags; the
    flagged N≥5 binary64 tails show what TIRust still lacks (COLLIER's small-momentum expansions and
    its choice of method by estimated error). Original IREGI's PV fails on every rank-6 box and on every
    heptagon, its IBP on the hexagons and heptagons (E/A up to 10#super[97]), all reported as successful
    (minimal reproducers in `validation/mre`). NinjaRS, reducing the contracted numerator, returns a
    value on every point with tails beyond 10#super[−3] on one rank-6 box and a few hexagons and
    heptagons. Heptagon references are confirmed by COLLIER COLI and DD (≤10#super[−9]) instead of quad
    IREGI, which fails there in both modes; COLLIER's heptagon curves are therefore conditioned on that
    agreement (5 of 500 points excluded); six hexagon points are confirmed the same way. Captures
    001-006 remain bound unchanged in the report data.
  ]
  v(0.2em)
  let rescued = snapshot.integrals.map(i => (i.id, i.codes.at(0).at("rescued_by_bits", default: (:))))
  block(breakable: false, figure(kind: table, caption: [Reference and TIRust precision rescue per integral.])[
    #table(
      columns: (1.1fr, 1.5fr, 1.3fr, 1.5fr, 0.8fr, 1.6fr),
      inset: (x: 4pt, y: 4pt), stroke: table-stroke, align: left + horizon,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header(..([Integral], [256-bit reference confirmed by], [Confirmation E/A median / max],
        [Confirmed only above 10#super[−12] (COLLIER sides with 256 bits)], [Excluded],
        [TIRust lanes rescued (bits: lanes)]).map(h => text(fill: white, size: 7pt, weight: "bold", h))),
      ..snapshot.integrals.map(i => {
        let rules = i.reference_rules
        let tiers = rescued.find(r => r.at(0) == i.id).at(1)
        (
          text(size: 7.5pt)[*#i.id*],
          text(size: 7pt)[#for (rule, count) in rules [#rule.replace("256-bit, confirmed by ", "").replace("quadIBP", "quad IBP").replace("quadPV", "quad PV"): #count \ ]],
          text(size: 7pt)[#de-sci(i.reference_checks.confirmation_median) / #de-sci(i.reference_checks.confirmation_max)],
          text(size: 7pt)[#i.reference_checks.at("confirmed_above_1e-12") (#i.reference_checks.of_which_collier_closer_to_256bit)],
          text(size: 7pt)[#(i.points - i.with_reference) / #i.points],
          text(size: 7pt)[#if tiers.len() == 0 [none] else { tiers.pairs().map(((b, c)) => [#b: #c]).join(", ") }],
        )
      }).flatten(),
    )
  ])
  text(size: 8pt)[
    *Reference.* The TIRust direct engine at 256 bits on the same binary64 inputs, accepted per point only when
    quadruple-precision original IREGI (IBP or PV, whichever agrees better; on-shell snapping as above) confirms
    it to E/A ≤ 10#super[−9]; otherwise the quad IBP array when both quad modes agree, else the point is
    excluded. Quad IREGI alone cannot be the reference: its IREGI-native normalization carries a binary64 π
    (every array is off by 6.3×10#super[−17]), quad PV fails on the rank-6 box (rank above N), quad IBP on the
    hexagon, and quad PV keeps only 10#super[−12]–10#super[−10] on some hexagon points, where COLLIER (independent
    of both) agrees with the 256-bit array instead. Below the confirmation level Δ is not independently verified.
    TIRust lanes rescued at 256 bits equal the reference after rounding (Δ = 0). This capture: TIRust
    #raw(snapshot.tirust_git_head.slice(0, 7)). Binaries and the TIRust commit are in #raw("validation/benchmark/stability-20261030/summary" + (if stability-campaign-id == none { "" } else { "-" + stability-campaign-id }) + ".json"); reproduce with
    #raw("make-points.py"), #raw("build.py"), #raw("run.py") and #raw("summarize.py") in that directory.
  ]
}

#let opening-timing-pages(examples) = {
  pagebreak()
  heading(level: 1)[Example tensor integrals: double-precision timing]
  text(size: 8.8pt)[Representative $N <= 6$, cumulative rank $R <= 6$ inputs.
  *Every timing in these two opening tables uses binary64 (f64).* Each call
  requests all Cartesian moments $q^alpha$ with $|alpha| <= R$ and all four
  native Laurent slots. Colors identify timing components, not relative speed.]
  v(3pt)
  block(fill: pale, inset: 6pt, radius: 3pt, width: 100%)[
    #set text(size: 7.6pt)
    *Exact inputs:* metric $(+,-,-,-)$; use the first $N$ shifts from
    $(0,e_x,e_y,e_z,e_t/2,e_x+e_y)$ and squared masses $(1,2,3,4,2,3)$;
    scale $mu=3$. Denominators are $(q+p_i)^2-m_i^2+i 0$, powers $nu_i$;
    $D=4+2h-2epsilon$. Numerator components stay four-dimensional.
    *Requested accuracy:* for ordinary N5/R5, N5/R6 and N6/R6, TIRust targets
    $2^(-20) approx 9.54 times 10^(-7)$ and COLLIER targets $10^(-6)$.
    Other rows use TIRust's $2^(-34)$ target and COLLIER's $10^(-10)$ target.
    These are requests, not achieved precision; no working-precision promotion is used.
  ]
  v(4pt)
  example-timing-table(examples.rows.filter(row => row.table == "ordinary"))
  v(3pt)
  text(size: 7.3pt)[*Optimistic selection:* fastest arithmetic mean among the
  fixed measured candidates returning a complete finite array with
  $delta_"rel" <= 10^(-3)$. If none qualifies, any observed duration remains
  visible as diagnostic. A missed $10^(-6)$ request or an internal accuracy
  flag alone does not suppress the timing. All candidate attempts are retained.]
  v(2pt)
  text(size: 7.3pt)[*Phase split:* TIRust excl. masters = total call wall time −
  nested scalar-master wall time. Means of the same timed calls add before
  rounding. SE is descriptive standard error across blocks, not a
  controlled-host confidence guarantee. Conversion, dispatch, stability checks
  and tensor assembly are included; cold preparation, first call, warm-up,
  result checks and I/O are excluded. Plans are reused; numerical results are recomputed.]
  v(2pt)
  text(size: 7.3pt)[*COLLIER:* 1.2.10, binary64, scalar masters included.
  Its selected public mode and all raw flags are recorded with each row.
  *DD* denotes the Denner–Dittmaier algorithm, still in binary64; it does not
  mean DoubleFloat or double-double arithmetic.
  Both tools' total cells show measured output error where available, defined on
  the next page. Amber denotes error above $10^(-3)$, nonfinite or refused output,
  rather than failure to reach a stricter accuracy request.]

  pagebreak()
  heading(level: 1)[Generalized requests outside COLLIER's direct API]
  text(size: 8.8pt)[Dotted and independently shifted requests, all in *binary64*.
  The same exact input family, full cumulative tensor outputs, optimistic
  selection and component timing definitions apply.]
  v(4pt)
  example-timing-table(examples.rows.filter(row => row.table == "extended"), extended: true)
  v(5pt)
  text(size: 8pt)[*Direct-API exclusion:* dots mean $nu_i > 1$; independent
  shifts request $D=4+2h-2epsilon$ with $h != 0$. COLLIER 1.2.10's public tensor
  routines accept momenta, masses and rank, without propagator-power or general
  dimension-shift inputs. Internally used higher-dimensional objects and
  equivalent derivative/coincident-denominator constructions are different
  interfaces and were not benchmarked here. See the
  #link("https://arxiv.org/abs/1604.06792")[COLLIER manual].]
  v(4pt)
  block(fill: pale, inset: 7pt, radius: 3pt, width: 100%)[
    #set text(size: 8pt)
    *Measured relative error:* $delta_"rel" = E/A$, with
    $E = max_j (|x_j-x_j^"ref"| / w_j)$,
    $A = max_j (|x_j^"ref"| / w_j)$ and
    $w_j = mu^(d_j+4+2h-2 sum_i nu_i)$, where $d_j$ is the numerator degree.
    The maxima cover all cumulative moments and all four complex Laurent slots.
    This is a dimensionally rescaled array-relative discrepancy against the
    qualified saved finite-precision tensors, not an exact-integral error bound.
    The older componentwise mixed metric remains in the raw audit.
    Withheld or unavailable arrays have *accuracy unavailable*, never a zero error;
    nonfinite counts are shown separately. A cell marked “finite” compares only
    its available finite fields and cannot certify the incomplete array.
  ]
  v(4pt)
  text(size: 8pt)[*Measured path:* public #raw("TensorReductionPlan / TensorWorkspace"),
  selected from 38 initial algorithm/master-policy candidates and four PV
  trials added after dotted IBP refusals. The latter only permute denominator
  triples; the integral, loop variable and accuracy request stay unchanged.
  Engine names, candidate IDs and profiling scope are saved. These paths are not a formal
  default-path or equal-accuracy speed comparison. No automatic higher-precision
  rescue is permitted in this cohort.]
  v(4pt)
  text(size: 8pt)[*TIRust protocol:* singleton input lanes. For $N <= 4$, three warm-ups
  and 11 blocks target 30 ms, capped at 10,000 calls per block; for $N >= 5$,
  one warm-up and five one-call blocks. Plan/workspace construction and first-call
  specialization are recorded separately from warm calls. Complete finite arrays,
  repeated-output consistency and unchanged inputs are checked outside the timer.]
  v(4pt)
  text(size: 7.5pt)[*Environment and provenance:* 2026-10-26, AMD EPYC 9754,
  pinned logical CPU 18, one thread; Rust 1.99.0 release build.
  Exact build/binary hashes, all candidates, requested tolerances, numerical
  flags, raw calls and selection decisions are retained in
  #raw("validation/benchmark/opening-double-20261026/").
  Prior mixed-precision opening captures remain archived unchanged.]
  v(4pt)
  text(size: 7.5pt)[*Reproduction (fresh output copy; see suite README):*
    #linebreak()
    #raw("python3 validation/benchmark/opening-double-20261026/run.py build")
    #linebreak()
    #raw("python3 validation/benchmark/opening-double-20261026/run.py run")
    #linebreak()
    #raw("python3 validation/benchmark/opening-double-20261026/run.py summarize")
    #linebreak()
    *No cross-session IREGI/COLLIER speed ratio is inferred from these absolute times.*
  ]
  pagebreak()
}

#align(center)[
  #v(0.4em)
  #text(22pt, weight: "bold", fill: navy)[TIRust benchmark]
  #v(-0.3em)
  #text(12pt, fill: luma(80))[Batched one-loop reduction: TIRust, IREGI, COLLIER and MadLoop]
  #v(0.2em)
  #text(9.5pt, fill: luma(110))[#report-date · Archival comparisons: #data.host]
  #if direct-campaign != none {
    linebreak()
    text(9pt, fill: luma(110))[Direct engine: same-session TIRust/IREGI/COLLIER · AMD EPYC 9754 · CPU #direct-campaign.snapshot.environment.cpu · binary64]
  }
  #if opening-examples != none {
    linebreak()
    text(9pt, fill: luma(110))[Historical opening examples (26 October): CPU 18 · previous TIRust engine]
  }
]
#v(0.6em)
#block(fill: pale, inset: 10pt, radius: 4pt, width: 100%)[
  #set text(9.5pt)
  *Reading guide.* Every displayed timing ratio has TIRust in the numerator:
  _TIRust / IREGI_, _TIRust / COLLIER_, or
  _TIRust(candidate) / TIRust(baseline)_. Physical graphs use _TIRust / MadLoop_.
  In comparison cells, green means TIRust is faster, white parity and red slower. The archival
  reference-only table shows absolute times and outcomes. Gray is unmeasured;
  amber marks pending or inconclusive evidence.
  #if opening-examples != none { [The opening example tables use blue/green to identify reduction/master components, not relative speed.] }
  Source inspection does not establish numerical agreement. *Archival comparison note:* #data.measurement_note
]
#v(0.4em)
#block(stroke: 0.6pt + luma(200), inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Current priorities*
  #table(columns: (1fr, 2fr, auto), stroke: none, inset: (x: 4pt, y: 4pt),
    [*Feature completion*], [The documented finite N≤6/rank≤6 numerical checkpoint, including dots, is validated. Higher-N qualification remains separate.], [*Validated*],
    [*Performance optimization*], [Direct numerical engine is the binary64 default; same-session TIRust/IREGI/COLLIER tables follow. Further optimization continues.], [*Active*],
    [*Physical validation*], [Current fixed-graph checks appear below; historical timing failures remain visible], [*Scoped*],
  )
]
#if direct-campaign != none { direct-engine-pages(direct-campaign.snapshot) }
#if stability-campaign != none { stability-pages(stability-campaign.snapshot) }
#if opening-examples != none { opening-timing-pages(opening-examples.snapshot) }
= Current coverage
#text(size: 9.5pt)[The documented N≤6/rank≤6 finite numerical checkpoint now passes
the agreed ordinary, dotted and shifted-array gates. The original quadrature-bearing
Dual witness passes complete384 DD/context/lifecycle checks and SAME_SOURCE 256/384
consistency at unchanged2⁻⁸⁰. FAILED 128/256 (628 fields) remains explicit. Performance is deferred.]
#report-table(table(
  columns: (1.05fr, 3.6fr, 0.8fr), inset: (x: 6pt, y: 6pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Interface / domain], [Executed scope or remaining gate], [Status]),
  [Local PyO3 facade], [Restored-source c65b wheel: 10 installed smoke cases / 2709 saved checks, 205 selected bodies / 128 declared coherent of the actual 210-source build. Its N6 smoke is rank one; older 65API/7marimo/typing stay historical.], [Scoped],
  [Dotted propagators], [N1/N4 first-dot/triple and N5 rank-six first-dot results retain their scoped passes. Current N6 first-dot256/prior128 strict pairs now pass; failed pre-repair attempts remain visible.], [Scoped],
  [Example gallery], [Seven historical runtime cases and earlier three HTML exports/six browser views keep their own source contexts. No new gallery or typing run accompanies the current 10-case smoke.], [Historical],
  [Tensor / API coverage], [Current1f source24/c65 wheel10 remain inherited. Original-SAMPLE128/256/384 pass complete DD/context/lifecycle checks; strict 256/384 passes, while FAILED 128/256 (628 fields) remains visible. API point count is dynamic; this qualification is finite N≤6/rank≤6.], [Scoped],
  [Public rank-six Dual], [Historical h0 PV/Hybrid jets retain scope. The shifted requested-PV / selected-IBP original witness passes complete384 and strict 256/384 gates. Failed 128/256 and all earlier resource/math attempts remain archived. No new default-Hybrid h1 invocation is inferred.], [Scoped],
  [Performance], [Historical private selected TIRust/COLLIER: seven slower families and one short inconclusive family. The fresh scheduler cohort is ineligible because of host contention; no default adoption.], [Measured],
))
#text(size: 8.5pt)[Evidence is local Linux/Python3.13 and source-specific.
Other platforms, separate Community-host loading and broader tensor numeric/exceptional
domains remain unqualified. Earlier C0/D0 identities retain their wheel004 context.]

// Historical intermediate outcomes are retained below the current validated checkpoint.
// Final sealed numerical qualification; publication verification is recorded separately.
#pagebreak()
= Rank-six feature checkpoint: validated
#block(fill: rgb("#d9f0d3"), inset: 9pt, radius: 4pt, width: 100%)[
  *Validated numerical checkpoint.* All named finite numerical gates
  have sealed evidence. The documented scope is N≤6/rank≤6; performance is deferred.
  All454 prior archival records and timings remain unchanged.
]

#report-table(table(
  columns: (1.25fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (1, 2, 3, 4) { rgb("#d9f0d3") } else { amber },
  table.header([Feature gate], [Sealed evidence and qualified scope], [State]),
  [Repaired parent arrays], [Current Cartesian N6/rank-six first-dot and D2 pass 3360 current/prior reference slots, 1680 strict prior128/current256 cross-source pairs and 2520 pole controls. Pair maxima≈4.47284×10⁻²⁸ /2.63578×10⁻²⁷; same-source convergence is not claimed.], [Sealed pass],
  [Independent D6], [Current256 passes both complete 840-slot defining-integral rules (maxima≈1.13558×10⁻¹¹ /6.67806×10⁻¹⁵) and 840 strict cross-source pairs (max≈1.77715×10⁻²⁶). All 84 degree-six UV slots, including 20 macroscopic values, remain intact.], [Sealed pass],
  [Quadrature-bearing Dual], [Complete384 passes both2520-field DD rules (max≈3.97433×10⁻¹⁵), raw poles,3parent/6child full contexts and lifecycle/terminal gates. SAME_SOURCE 256/384 passes2520 at unchanged2⁻⁸⁰, max≈3.89996×10⁻⁶⁰. FAILED 128/256 (628 fields) stays archived.], [Sealed pass],
  [Release source join], [Current c65 wheel passes 10 installed cases / 2709 saved checks: 205 selected / 128 coherent of restored 210-source 1f build. Native 8978 equals historical safe 5ef. Its N6 smoke is rank one; ccee/2674 and older typing/gallery stay historical.], [Inherited pass],
  [Feature domain], [Generic cumulative rank≤6 with positive powers and agreed numeric/batch/API mechanisms. The API has no fixed N ceiling. The documented N≤6/rank≤6 families are a finite checkpoint; arbitrary-N and every-chart qualification are not inferred.], [Scoped],
  [Performance], [Optimization is deferred. Keep every historical timing record unchanged and every displayed ratio with TIRust in the numerator. No new speed or parity gate is inferred from validation durations.], [Deferred],
))

The ordinary parent capsules remain sealed. The earlier e37b bridge is
historical. Historical403/ccee and its96 GiB tensor failures retain their own epoch.
Current restored1f binds24 focused passes; currentc65 binds10 installed cases.
The final capsule binds the actual 210-build-snapshot source aliases; historical
canonical aliases are not substituted. Complete128/256/384 DD/context/lifecycle schedules pass. Full256/384 consistency passes at2⁻⁸⁰; the failed 128/256 gate remains separate.
The failed 128/256 strict pair is inherited without duplicate capture/source
bodies; full256/384 convergence and the additive final capsule are now sealed.

The numerical gates are complete on this finite checkpoint. Preserve all wider numerical-domain limits on the next page.

#pagebreak()
= Retained failures and limits of the completed checkpoint
This checkpoint records supported routes, their earlier failures and the
precise limits of the accepted numerical evidence.
The saved b2/de3c schema audit, root scratch-override failure and corrected final
archive source-path assertion remain separate from the ten failed native witnesses. Strict128/256 is also a separate
comparison failure; it remains visible beside the qualified256/384 pass.
No missing coefficient, failed lane or unrun comparison may become a zero or a
successful result.

#report-table(table(
  columns: (1.25fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Retained domain], [Required qualification boundary], [Outcome]),
  [Ten full witnesses], [Retain eight resource/time no-array attempts and two96 GiB nonfinite tensor failures. Focused/compiler and saved-checker histories remain separate; strict 128/256 (628 fields) remains failed. The latter each record one failed call; empty/recovery is unrun. A two-second/96-sample profile of the earlier12864 GiB run is not the exact256 failure stage.], [Retained],
  [Earlier scalar/tensor failures], [Old Cartesian dot/D2256 E6 failures, rejected original D6 finite comparisons and earlier cost stops remain unchanged. New successful source epochs have separate manifests and actual comparisons.], [Retained],
  [Scalar Dual charts], [Unsupported massless C0/D0 derivative neighborhoods remain explicit failure paths. A rejected implementation chart does not prove mathematical nonanalyticity, and no broader branch qualification is inferred from one passing anchor.], [Scoped limit],
  [Independent accuracy], [The qualified256/384 precision consistency and two-rule agreement are numerical evidence. Retain FAILED 128/256 (628 fields) of2520 adjacent. No rigorous quadrature remainder or certificate of achieved working-bit accuracy follows. Cross-source pairs are labeled explicitly.], [Scoped limit],
  [Point count and kinematics], [The tensor API has no fixed point-count ceiling. The independently tested finite envelope is N≤6/rank≤6; arbitrary-N, arbitrary-kinematics and broader derivative charts remain unqualified.], [Scoped limit],
  [Completion boundary], [Broader powers/shifts, accepted Legacy/rescue and exceptional domains retain their existing mechanism-witness scope. They are not blanket new-grid requirements. Performance optimization remains separate future work.], [Scoped limit],
))

The historical optional-profile skip passed24 focused checks but failed both
full96 GiB tensors. Restored1f source passes24 focused checks, including finite first-order R0
contexts and an honestly refused transverse E6 lane; currentc65 passes10 installed
cases. The unchanged original-fixture mass-second128/256/384 arrays pass actual
DD/context/lifecycle checks. The stricter128/256 gate failed628 fields; full256/384
consistency and its final capsule now pass without relabeling that failure. The observed requested-PV
route selects genuine IBP (0PV/610IBP requests, runtime fallback0). Future
qualification must retain actual profile/analytic rows rather than demand
actual PV or zero analytic rows. In the retained nonfinite failures, accepted masters
and converged E6 endpoints did not establish finite tensors; the first bad node remains unknown.

Final source bindings must identify which current core, scalar dependencies and
frontend release were actually exercised. Reuse unchanged source/archive data
without implying a complete hermetic dependency closure. Historical old wheel
and test-source contexts remain distinguishable from the final installed smoke.
Physical comparisons retain their rank-four graph/CLI scope; no new rank-six
physical or broad external Community-host qualification is implied.

// Certificate prose: complete Gram/Cartesian/Taylor checks apply CHECKED high-point supplied-G routes; ordinary lower-point supplied-G remains authoritative.

// FINAL MATERIALIZATION: relabel the da7 precision-boundary page explicitly HISTORICAL.
// Its384-active/no-array statement is the old cutoff, never current final prose.

#pagebreak()
= Historical da7 precision boundary: retained128/256 failure
#block(fill: amber, inset: 7pt, radius: 4pt, width: 100%)[
  *HISTORICAL — da7 cutoff.* The following384-active/no-array and pending statements
  describe the archived earlier milestone. The validated checkpoint above supersedes
  those runtime states, preserving every original comparison, failure and raw value.
]

#block(fill: amber, inset: 9pt, radius: 4pt, width: 100%)[
  *Intermediate checkpoint — objective open.* The original restored fixture
  agrees with the independent defining integral, while the stronger
  same-artifact 128/256 precision gate fails. Its threshold is unchanged.
]

#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 11pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (1, 2) { rgb("#d9f0d3") } else if y == 3 { rgb("#fddbc7") } else { amber },
  table.header([Gate / scope], [Observed result or pending admission], [Outcome]),
  [Defining-integral agreement], [Both complete 128/256 runs pass 2520 fields per DD rule at 1e-10 (all four maxima≤3.97433×10⁻¹⁵). The DD rule pair also passes. Four retained 1806-control sets preserve raw reducer/reference poles without projection; the joint saved replay counts each dataset explicitly.], [DD pass],
  [Owned contexts / lifecycle], [Each precision completes 3 parent / 6 child owning contexts with higher errors 0/1/2, bit-identical duplicate/recovery, isolated invalid null/context refusal and actual empty reset. Both terminal codes are 0; guards and cleanup pass. These are actual complete schedules.], [Complete],
  [Strict 128/256 consistency], [FAILED 628 of 2520 fields at unchanged 2⁻⁸⁰≈8.27181×10⁻²⁵. Maximum normalized difference≈2.92162×10⁻²¹. The independent saved-array replay confirms the failure. Retain all failed fields and the original comparison; no relaxed gate or exclusion.], [Failed],
  [Precision interpretation], [Requested 53 uses heuristic admission 2⁻³⁴  / E6 target 2⁻³⁶; Success does not promise 80-bit consistency. All 630 finite differences are consistent with combined estimates (saved 128 Q≈9.99×10⁻²¹). Order 64 / 2400 boxes are unchanged; tracked imaginary precision 94→222 bits. This fits precision-dependent cancellation/E6 noise, not a rigorous bound.], [Scoped],
  [Higher precision], [One 384-bit producer is now active on the same 749b / current 1f source, fixture, 53-bit request and mathematics, under 256 GiB / 7200 native seconds. No 384 result is admitted. Full 384 DD/context/lifecycle and strict 256/384 comparison at 2⁻⁸⁰ remain pending; failed 128/256 stays visible.], [Active],
  [Preserved scope], [All 451 prior raw values and nested publication history stay exact. Ten old native witnesses and the separate saved-schema failure remain visible; this precision-gate failure is separate. No new performance claim; TIRust remains the timing-ratio numerator.], [Retained],
))

The DD comparisons and the strict precision comparison answer separate
qualification questions. Passing the former does not erase the failed latter.
Source/error estimates are heuristic, not certified achieved-bit accuracy.
No source formula, input or threshold is changed for the proposed continuation.

Sealed capsule cd656689 owns the actual complete 128/256 packet once. This
intermediate binds that immutable capsule
and its checkpoint. Future 384 capture files are excluded while mutable.
The API has no fixed point-count ceiling; independently documented families
remain N≤6 / rank≤6. Final feature completion remains pending.

#text(size: 8pt)[Source: #raw("validation/e6-dual-rank6-precision-boundary").
The adjacent checkpoint and root saved-replay packet preserve exact comparisons,
source/guard context and harness history. Saved replay success retains the
FAILED precision gate; it does not promote it to a pass.]

#pagebreak()
= Earlier exact-zero restoration (a7ccffb)
#text(size: 8.5pt, fill: luma(100))[Historical a7 cutoff: full128 was preparing and 256 was held.
The current precision-boundary page supersedes those runtime states, with no change to their archived evidence.]
#block(fill: amber, inset: 9pt, radius: 4pt, width: 100%)[
  *Intermediate checkpoint — objective open.* The historical optional-profile
  bypass did not produce a valid full rank-six tensor. Its failures are
  retained; the restored route has a separate source and release epoch.
]

#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 11pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (2, 3) { rgb("#d9f0d3") } else { amber },
  table.header([Evidence], [Sealed evidence; objective remains open], [State]),
  [Two 96 GiB failures], [Both original-SAMPLE runs return 101 with a failed first-call record. Four N5 rows fail the nonfinite-output guard; valid parents are Unstable, failed arrays null and the invalid neighbor isolated. No 2520 tensor comparison or owned error context passes. Capsule 792765c1 preserves 526 nonbinary guards.], [Retained],
  [Restored source], [Sealed 1f1a source capsule removes the 15-line bypass: 24 focused checks pass in 26.8588 s from build b069 / product 749b. The first-order Dual regression accepts finite R0 rows and retains the transverse E6NonConvergent refusal. Earlier 23-pass / failed-expectation invocation b804 and compiler boundaries remain intact. Compiler opt3 disables debug assertions and integer overflow checks; old debug403 enables them. Numerical precision/branch/error guards remain unchanged; no compiler-equivalence claim.], [Sealed pass],
  [Installed frontend], [Sealed c65b wheel passes the same 10 installed cases / 2709 saved checks: 205 selected, 128 coherent of 210. Actual build 118.851 s and fresh install/import bind the restored source. Native 8978 matches historical safe 5ef byte-for-byte. N6 smoke is rank one; old typing/gallery remain separate.], [Sealed pass],
  [Retained generic route], [The failed captures record requested PV selecting actual IBP: 0 PV / 610 IBP scalar requests, runtime fallback 0. Finite-master admission and converged E6 endpoints did not establish finite tensors. Actual profile/analytic rows must remain visible; old PV-only / zero-analytic expectations are obsolete.], [Scoped],
  [Full Dual witness], [Restored original SAMPLE 128 is root-released with 192 GiB / 7200 native seconds; 256 is held. Full 2520 fields, same-source pair, DD/core agreement, owning error contexts and lifecycle remain pending. Active capture files are not bindings.], [Pending],
  [Coverage and timings], [Ordinary dot/D2/D6 and DD reference are inherited without duplicate counts. No fixed N ceiling is added to the API; independently documented families remain N≤6 / rank≤6. No new performance measurements; TIRust remains the timing-ratio numerator.], [Unchanged],
))

Ten unsuccessful full-witness attempts remain visible: eight earlier resource/time cases and
these two returned failed-call records. Empty and singleton recovery remain
unrun in the latter. The first nonfinite evaluator node is unknown; exact-null
geometry as its cause remains a hypothesis. Saved guard checks are archival
integrity evidence, not independent tensor comparisons. Separate focused-test
and compiler-boundary failures remain in the source history; they are not
counted among these ten full witnesses.

This additive checkpoint preserves all 446 baseline raw values and nested
publication history. It records restored-source checks and source epochs;
it does not close the remaining numerical gates or promise a resource bound.

#text(size: 8pt)[Sources: #raw("validation/e6-dual-postfix-nonfinite-boundaries"),
#raw("validation/e6-dual-restored-zero-profile-core-tests") and
#raw("validation/python-interface/restored-zero-profile-wheel-refresh-smoke").
Root saved replays are archived in the adjacent restoration checkpoint; no new
performance or full rank-six numerical qualification is inferred.]

#pagebreak()
= Earlier resource boundaries (c112865)
At the c112865 cutoff, the original-SAMPLE N6/rank-six/h1 mass-second witness
had no admitted full tensor array. Four post-fix attempts had ended without
completing a call; four earlier failures were retained. Two larger-budget
96 GiB attempts were then active. Both subsequently failed at the nonfinite
tensor guards documented above. That diagnostic checkpoint preserved all 443
prior raw values and added no TIRust numerical or performance qualification.

#report-table(table(
  columns: (1.2fr, 3.6fr, 0.8fr), inset: (x: 7pt, y: 11pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else { amber },
  table.header([Frozen attempt / model], [Terminal evidence or precisely bounded interpretation], [Outcome]),
  [Post-fix Float128/256], [Both earlier 2400-second attempts complete generic construction (about 1778s each), then hit the native execution deadline without a completed call or array. They retain the source 210/product 4f430732 context and requested accuracy 53; these are resource limits, not numerical disagreements.], [No array],
  [Float128 /64 GiB], [The 4200-budget attempt aborts in numerical allocation after 3733.628 native seconds; GMP cannot allocate 16 bytes. The final receipt retains 250 unchanged guarded bindings, empty cleanup and an absent owned process group. No array is returned.], [No array],
  [Float256 /64 GiB], [The 4200-budget attempt aborts after 2912.259 native seconds; GMP cannot reallocate 16→40 bytes. The same 250-binding terminal guard/cleanup checks pass. This does not establish the exact evaluator, slot or refinement active at failure.], [No array],
  [Then-active 96 GiB attempts], [At this historical cutoff, separate 128/256 requests were active: 7200 native / 7380 total / 120 cleanup seconds each, explicit PV, source 210 / product 4f430732 and unchanged fixture/shape/accuracy 53. Their later failed-call records are separately bound by 7927 above.], [Historical],
  [Two-second /96-sample profile], [The sample concerns one numerical instant of the older Float128 /64 GiB run. It sees complex multiprecision arithmetic, copies and allocation. It does not identify the later Float256 failure stage, whole-run cost, peak memory, or 96 GiB sufficiency.], [Diagnostic],
))

The source model is finite but is not an exact peak bound. For master-batch capacity
B>1, Symbolica reserves B×(root slots + shared-body slots), including inactive
shared bodies, and retains the largest capacity. E6 with R active requests,
order q and c components prepares 5Rq box rows, 11c input values and 6c allocated
master outputs per box row. At c=3, that is 33 complex inputs and 18 allocated outputs per
box row (including internal quantities), before other retained workspaces, graphs and allocator storage.

One active request through default orders 8–128 accounts for 1240 cumulative
box rows and a largest refinement of 640 rows. These are structural formulas,
not observations of the failed stage. Initially one-bit Float zeroes can acquire
larger limbs when touched at working precision. First-touch expansion and
allocator retention are plausible explanations of growth, not a proven fault
cause, leak or plateau. The saved artifacts lack the exact live slot inventory.

#block(fill: rgb("#d9f0d3"), inset: 7pt, radius: 3pt, width: 100%)[
  *Newly archived reference.* The already completed DD defining-integral stream
  and all 2520 parent rule pairs pass (maximum≈2.44997×10⁻¹⁵). This archives
  reference qualification; it adds no TIRust numerical success. TIRust/DD
  comparisons remain pending.
]

The initial offline symbolization timeout remains failed; the later no-inline
report is separate. Raw perf data is omitted by historical hash. At this cutoff,
96 GiB terminal results were not yet bound. Ordinary/core/wheel counts stay inherited. Final
completion waits for actual full-Dual arrays, strict pairs and independent
comparisons; performance remains deferred.

#text(size: 8pt)[Source: #raw("validation/e6-dual-postfix-resource-boundaries") and
#raw("validation/e6-dual-rank6-mass-second-defining-integral").
Memory formulas are source-based diagnostics, not a complete dependency closure
or a certified byte/accuracy bound.]

#pagebreak()
= Earlier ordinary closure (dfcae98)
The documented ordinary N6/rank-six first-dot, D2 and genuine-D6 families now
have complete current Float256 results and strict cross-source precision comparisons.
The full rank-six feature goal remains open while the quadrature-bearing Dual
witness runs. All 437 earlier raw report values and timings remain unchanged;
no performance or arbitrary-N claim is added.

#report-table(table(
  columns: (1.25fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 6 { amber } else { rgb("#d9f0d3") },
  table.header([Executed domain], [Complete sealed evidence and boundary], [Outcome]),
  [Cartesian first dot / D2], [Together: 3360 current/prior original-quad slots, 1680 strict cross-source pairs and 2520 pole controls pass. Current256 maxima are≈9.11420×10⁻¹⁶ and 2.67157×10⁻¹⁷; pair maxima≈4.47284×10⁻²⁸ and 2.63578×10⁻²⁷.], [Pass],
  [Genuine D6 current Float256], [Both complete 840-slot defining-integral rules pass 10⁻¹⁰ (maxima≈1.13558×10⁻¹¹ /6.67806×10⁻¹⁵). All 840 strict old Float128/current Float256 pairs pass 2⁻⁸⁰ (max≈1.77715×10⁻²⁶). The 5332-gate independent audit and all 84 degree-six UV slots remain preserved.], [Pass],
  [Actual ordinary batching], [The D6 parent uses four packed same-dimension N5/IBP rows, 124 scalar masters and 2400 pentagon-box rows. Requested and actual methods remain explicit; broader mixed-method tensor completion is not called pure IBP.], [Scoped],
  [Earlier final wheel], [The 5ef source epoch passes 10 installed smoke cases with 204 selected files and 128 coherent core bodies; 2614 saved bindings pass. This wheel predates the resource-route change below and remains immutable.], [Historical pass],
  [Current wheel refresh], [The ccee epoch repeats 10 cases after the route fix: 8 low/interop, 1 checked-Dual rejection, 1 N6 rank-one lifecycle/contraction. 205 selected files match 128 declared core bodies within the actual 210-source build; 2674 saved checks pass.], [Pass],
  [Still open], [Original-SAMPLE N6/rank-six explicit PV Float128/256 arrays with owned full-component quadrature contexts, their strict pair, independent mass-derivative anchor agreement, and final coherent publication remain pending.], [Open],
))

Every precision pair above is explicitly CROSS_SOURCE: current Float256 is compared
to a separately frozen prior Float128 result. No same-build precision convergence or
achieved working-bit accuracy is inferred. Rejected original D6 finite values
remain archived and are not the defining-integral gate. The earlier dot/D2256
E6 failures remain failed in their original source epochs.

Both wheel epochs use the same exact ten API cases. The N6 smoke has rank one;
full rank-six mathematical validation is separate. Earlier 65API/7marimo and
consumer typing remain historical contexts. There is no new notebook, typing,
platform, hermetic dependency-closure or independent-integral claim from smoke.

#text(size: 8pt)[Sources: #raw("validation/ordinary-gh-n6-generalized-requalification"),
#raw("validation/ordinary-gh-n6-d6-requalification"),
#raw("validation/python-interface/final-rank6-wheel-smoke"),
#raw("validation/python-interface/final-rank6-wheel-refresh-smoke").]

#pagebreak()
= Optional zero-profile route: correction and bounded tests
The retained 64GiB Hybrid and explicitPV preparations reach an optional
exact-zero runtime specialization after their primary N5 child completes, then
fail before a returned array. Saved stage ordering establishes this precise
attribution. The earlier Hybrid 32 GiB failure occurs in primary mapping instead.

#report-table(table(
  columns: (1.25fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 12pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (2, 3, 4) { rgb("#d9f0d3") } else { amber },
  table.header([Stage], [Actual source/test scope and retained limit], [Outcome]),
  [Corrected old diagnosis], [PV128 generic 12441-alias mapping and the primary N5 child finish before second 12318-alias preparation. Saved Hybrid64/PV128/PV256 logs show the same post-child order. The constructor eager-fallback stage precedes this later optional zero-profile; Hybrid32 fails earlier primary mapping.], [Retained failure],
  [Narrow route fix], [For genuine E6 multi-component quadrature Dual workspaces, retain the generic route and skip optional zero-profile preparation. Full-Taylor zero classification and generic branch, precision, coupled-context and error guards remain; actual skipped lane groups are diagnosed.], [Source pass],
  [Concrete current tests], [210 actual compiled source bodies; 24 distinct focused tests pass. The first stage has 23 passes and a 60-second timeout on the new real N5/D6 Dual test; the separate exact-test continuation passes in 146.707 total seconds. The original timeout remains failed.], [Scoped pass],
  [Coherent frontend], [The current ccee 10-case wheel refresh binds the changed execution source and regression module. Its 205 selected files are tied to the fresh 210-source context; source identity is not a successful full N6/R6 Dual result.], [Pass],
  [Full original fixture], [Post-fix original-SAMPLE explicit PV Float128/256 runs are active. No completed parent array or imported-context accuracy is admitted at this cutoff. The half-time replacement proposal remains held and unrun.], [Pending],
  [Independent anchor], [The DD reference and rule-pair stream are ready in unsealed staging. They do not qualify the reducer until actual full parent/Taylor/Laurent comparison and owned propagated-context gates pass. No pending target packet is source-bound here.], [Pending],
))

The 24-test source capsule is a finite route witness, including one real small
N5/D6 Dual regression. The initial new-test 60s timeout remains failed; its
separate successful continuation does not rewrite that receipt. No tolerance,
master branch, analytic domain or algebraic formula is relaxed by the fix.
Ordinary and prior frontend evidence retain their own compiled source scopes.

All four earlier Hybrid/PV preparation/resource failures remain
visible, irrespective of later success with the repaired generic route.
Unsupported massless scalar derivative charts, conservative high-point Legacy
admission, high-point explicit-Dual rescue, and arbitrary-N or arbitrary-chart
qualification remain limits rather than blanket new-grid completion gates.
Performance optimization stays deferred. All following sections retain their
historical outcomes and original source contexts.

#text(size: 8pt)[Source: #raw("validation/e6-dual-optional-zero-profile-core-tests").
No active-capture mutable input or unsealed DD packet is a report binding.]

#pagebreak()
= Earlier ordinary/source follow-up (fccffab)
This follow-up preserves all 425 earlier raw result values and timings. Each
new test or numerical result is tied to its actual source epoch. The generic
rank-six goal remains open; independent D6 adjudication is on the next page.
No performance measurement or fresh Python-wheel qualification is added.

#report-table(table(
  columns: (1.25fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Executed scope], [Saved evidence and precise boundary], [Outcome]),
  [Current core], [22 exact focused tests pass: 15 joint quadrature-budget, five zero-mass strategy and two existing MassTaylor ownership/projection guards. The emitted Cargo artifact was fresh; its 128 selected source bodies include the current scalar-master repair.], [Tests pass],
  [Joint-budget source], [The separate earlier source capsule preserves 15 tests and three contexts, including the failed synthetic fixture and Arc compile stage. Higher Taylor components combine their own propagated quadrature and operand-roundoff estimates; no actual propagated-quadrature Dual integral is qualified.], [Scoped],
  [Zero-mass supplied G], [Private N5/rank0 first-dot h=0 ordinary and scale-only Dual witnesses pass 102 exact gates (34+68), including mixed invalid lanes and empty/reset/recovery. Only μ=3+3t has nonzero jets; all kinematic jets vanish.], [Scoped],
  [Earlier Cartesian dot], [Pre-repair CLI003 N6/rank6 first-dot Float128 passes 840 quad slots and 630 pole controls. Float256 fails E6 convergence at order128, error≈2.3794×10⁻⁴, with no parent array or precision pair.], [Partial],
  [Earlier D2 shift], [The same pre-repair epoch at h=−1 passes 840 Float128 quad slots and 630 pole controls. Float256 fails E6 terminal1 at order128, error≈2.499×10⁻⁴; its 840 reference and 840 pair comparisons remain unrun.], [Partial],
  [Still open], [Repaired Cartesian dot/D2 and D6 parent precision pairs, a genuine quadrature-bearing Dual with independent anchor, and the final current release remain the completion gates.], [Open],
))

The private zero-mass fallback preserves the authoritative supplied products
and fresh complete geometry proof. The failed derivative in another positive
mass remains an unsupported existing massless C0/D0 derivative chart. It does
not establish mathematical nonanalyticity. This rank-zero ordinary/scale
witness does not qualify zero-mass rank-six or general kinematic Duals.

The source-test counts overlap: the later 22-test cohort reuses the 15 earlier
joint-budget tests. Neither count is added to the historical 120 as a claimed
unique total. The installed 65-case facade retains its prior wheel context.
Broader powers/shifts, Legacy/rescue and exceptional domains retain scoped
qualification limits; no new broad grids are required here. Original failures
and unavailable arrays remain unchanged. Source-test
success does not imply numerical array agreement, achieved working precision,
rigorous integration error or speed.

#text(size: 8pt)[Sources: #raw("validation/rank6-current-focused-core-tests"),
#raw("validation/propagated-dual-joint-budget-core-tests"),
#raw("validation/checked-g-zero-mass-fallback"),
#raw("validation/affine-power-generalized-memo-dot-rank6"),
#raw("validation/public-n6-rank6-d2-partial-evidence").]

#pagebreak()
= Independent D6 adjudication and scalar-master repair
A defining Feynman-parameter calculation now resolves the original D6 finite
reference contradiction at the fixed N6/rank-six point. Separately, a shared
coefficient repair fixes a scalar-box precision loss and restores one failed
E6 evaluation. These are distinct numerical claims with distinct source
contexts; parent precision-pair runs remain outside this checkpoint.

#report-table(table(
  columns: (1.25fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 11pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 2 or y == 6 { amber } else { rgb("#d9f0d3") },
  table.header([Independent check], [Executed numerical evidence], [Outcome]),
  [Complete D6 array], [Two fixed rules each check all 210 tensor moments / 840 Laurent slots against TIRust Float128 at 10⁻¹⁰. Maximum errors: order12≈1.13558×10⁻¹¹, order16≈6.67806×10⁻¹⁵. All 840 rule differences pass, maximum≈1.13491×10⁻¹¹.], [Pass],
  [Original finite values], [All 210 earlier finite disagreements remain archived. The original scalar/time coefficients also violate an exact positive-F mean bound. The independent parameter calculation agrees with TIRust; the earlier raw reference mismatch is not silently rewritten or counted as a Rust accuracy failure.], [Rejected reference],
  [Ordinary D0 repair], [Existing Symbolica coefficient_sum preserves the exact shared line coefficient. Ten regressions and exact chart identities pass. One copied Float256 D0 value agrees with independent order20/32 rules: maximum≈3.5524×10⁻¹⁵ at 10⁻¹⁰; rule comparisons pass 10⁻¹².], [Scoped pass],
  [Actual E6 recovery], [The old failed trace is retained. The repaired ordinary scalar path converges at order64 with 600 box rows and estimate≈7.39054×10⁻²⁷. This is actual convergence, not an independent E6 value or parent-tensor accuracy certificate.], [Scoped pass],
  [Common-node identity], [All 600 common nodes preserve exact inputs; 13200 component checks replay. Of their D0 values, 480 are unchanged and 120 at the isolated pinch change. Failed compiler, preflight and broad-observer preparation stages remain saved.], [Pass],
  [Precision limits], [The D6 oracle uses 106-bit DoubleFloat with exactly saved binary64 rule data. No rigorous remainder, Float256 parent pair, full Dual branch, native/generated parity, arbitrary-kinematics or performance claim is added.], [Limits],
))

D6 replay checks 40 windows, 5068 acknowledgements and exact limb/digest/pole
data. Its 84 degree-six UV slots include 20 macroscopic all-even values. The
missing-NumPy launch, MPFR stop and finite-reference failure remain saved.
Rule agreement is numerical convergence evidence.

The ordinary-only D0 repair preserves the complete prior seeded-Dual helper.
Its existing Dual regressions retain their own scope. The full imported-error
Dual path still needs actual numerical qualification. All following sections
retain their earlier immutable checkpoints and original outcomes.

#text(size: 8pt)[Sources: #raw("validation/ordinary-complex-box-correlation")
and #raw("validation/affine-power-generalized-current-d6-rank6").]

#pagebreak()
= Earlier feature boundary (6a826a0)
The current implementation checkpoint remains incomplete. Rank-six numerical
coverage has advanced through named public derivative and positive-power
paths; arbitrary point counts and the full generalized domain are not implied.
No performance measurement is added, and every preceding raw result remains
unchanged. Failed requests keep their original outcomes and source contexts.

#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 6 { amber } else { rgb("#d9f0d3") },
  table.header([Executed domain], [Evidence and precise boundary], [Outcome]),
  [Current core], [158 canonical source bodies, nine additional versions and four compile contexts. All 120 distinct focused tests pass across a test-only bridge: 95 completed first, then the remaining 25. Six compiler errors and the stale Legacy engine-tag observer failure remain retained.], [Pass],
  [Fresh Python facade], [65 installed API cases, seven marimo runtime cases and clean consumer typing. All 158 declared core bodies match the wheel input context within 201 selected files; 1575 portable saved checks pass.], [Pass],
  [N5 first dot], [Rank-six Float128/256: 5040 original-quad, 5040 mass-Taylor, 2520 precision pairs and 7140 pole/duplicate/scale checks, totaling 19740. Maximum reference error ≈2.77 × 10⁻¹⁶, precision difference ≈3.55 × 10⁻²⁹.], [Scoped],
  [LegacyNoOp N5], [PV/unit/h=0: 26 focused tests, 1680 complete quad slots and 840 strict 2⁻⁸⁰ pairs. Maximum reference error ≈1.09627 × 10⁻¹⁶; pair difference ≈2.65877 × 10⁻³². Legacy admits only a proved no-op snapping profile here.], [Scoped],
  [Checked supplied G], [Current default-Hybrid N6 rank-six unit/first-dot, h=0: 3360 original-quad slots and 1680 strict 2⁻⁸⁰ pairs pass. Maximum errors ≈9.11420 × 10⁻¹⁶ and 6.86405 × 10⁻³⁰. Exported acceptance diagnostics remain distinct from source-only inner proof guarantees.], [Scoped],
  [Remaining gates], [Cartesian generalized rank-six closure, D6/dimension shifts, broader positive powers, numerical Dual with imported quadrature budgets, accepted Legacy/rescue domains and remaining exceptional cases stay open.], [Open],
))

The N5 dotted pair uses the earlier memo CLI and its 119-source context.
Its two short preparation timeouts and three older unrun cases remain in the
archive. Passing first-dot arrays do not qualify N5 triple powers or later
G/MassTaylor sources. The Legacy pair likewise has a separate 151-source
context; its old memory abort, compile error and pre-launch refusal remain.
Neither pair establishes achieved working-precision bits or performance.

Mass differentiation is an explicit strategy with a nonzero differentiated-mass
domain. Default Cartesian/formal-G reduction retains direct recurrence, so
valid dimensionally regulated zero-mass dots are not silently removed from
its domain. The private checked-G hook still lacks a same-domain zero-differentiated-mass
fallback; the massive checks above do not close that gap. A current Cartesian
N6 first-dot Float128 run passes all 840 reference slots, but Float256 fails
its E6 convergence check at order128 and returns no parent array or precision
pair. D6/unit Float128 returns an array but fails all 210 finite reference
slots (maximum error ≈1.4213); its 420 IR and 126 lower-rank/84 rank-six UV
controls pass. Float256 remains unrun after that required gate fails.
The current guard on unsupported imported full-Dual quadrature remains active;
the proposed extension is not included in this publication.

#text(size: 8pt)[Sources: #raw("validation/checked-g-generalized-core-tests"),
#raw("validation/python-interface/facade-wheel-004"),
#raw("validation/low-n-generalized-rank6-memo-dot-pair"),
#raw("validation/legacy-noop-rank6"),
#raw("validation/checked-g-generalized-numeric"). Source binding is per cohort; no
unexecuted equivalence to later edits is asserted.]

#pagebreak()
= Public six-point rank-six mixed Duals
The public AffinePower facade now has complete fixed-family mixed-Dual results
for explicit PV and the unchanged Hybrid default. Each parent contains all
210 moments, four native Laurent slots and six ordered Taylor components for
the first squared mass and a simultaneous dilation. Sample lanes are separate
from Taylor components; every component is checked before contraction.

#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 5 { amber } else { rgb("#d9f0d3") },
  table.header([Public path], [Complete saved numerical checks], [Outcome]),
  [Explicit PV], [Float128/256: 30240 original-quad parent comparisons and 75600 parent/child precision pairs, including 60480 child pairs. Maximum errors ≈3.07559 × 10⁻¹⁵ and 5.47562 × 10⁻³⁰.], [Pass],
  [Default Hybrid], [The same full counts and numerical gates pass independently. Both captures explicitly record #raw("public_default") origin. Actual child methods, notes, batching and counters remain visible; requested Hybrid is not a pure-IBP claim.], [Pass],
  [Coupled contexts], [Per precision and method: 15 complete parent/child contexts, 12 actual N5 child rows, repeated valid inputs around an invalid-mu lane, empty reset and singleton recovery. Every child explicitly reports KnownNoQuadrature with complete error arrays.], [Pass],
  [Typed DoubleFloat], [106-bit metadata: 15120 quad, 15120 qualified Float128 controls and 5040 scale checks pass. Maximum normalized errors ≈3.07559 × 10⁻¹⁵, 1.96691 × 10⁻²⁵ and 8.03765 × 10⁻²⁶. Static/dynamic base outputs match exactly.], [Scoped],
  [Typed f64], [Three valid rows hit the real #raw("pv_node_12"), Taylor-component-two cancellation guard and return null arrays. Saved metadata passes, with zero numerical comparisons. The initial input-size helper refusal remains a separate pre-reduction failure.], [Guarded],
))

All cases use real massive unit powers, h=0 and requested accuracy53. The
reference/scale tolerance remains 10⁻¹⁰, precision/control pairs 2⁻⁸⁰.
The six ordered components are (0,0), (1,0), (0,1), (2,0), (1,1), (0,2).
Original unit/dot/triple arrays supply the mass/dilation responses; their
finite coefficients retain their scoped empirical admission. Agreement
between precision outputs is not a rigorous eighty-bit accuracy certificate.

The typed example repacks existing Symbolica static/dynamic Dual values into
the public component interface. It does not claim a new typed convenience
constructor, and it exports no hidden sensitivity context. The frozen observer
does export and validate full contexts. Zero quadrature arrays never establish
an origin by themselves: unsupported propagated origins remain failed.
Both earlier private ThinSix mixed128 allocation failures remain failed.

#text(size: 8pt)[Sources: #raw("validation/public-n6-affine-mixed-dual"),
#raw("validation/public-n6-hybrid-typed-dual"). No arbitrary-direction,
all-kinematic, rigorous-error, full-feature or speed claim follows.]

#pagebreak()
= Independent first and second mass finite differences
Five ordinary Float384 samples, requested accuracy160, provide an independent
check of the specified squared-mass derivatives. They form two three-point
central stencils sharing the center, at h=2⁻¹² and h=2⁻¹⁸. This is not a
fourth-order five-point stencil. Existing Symbolica Matrix.solve produces the
exact weights, including the second Taylor coefficient's factorial convention.

#report-table(table(
  columns: (1.2fr, 3.5fr, 0.8fr), inset: (x: 7pt, y: 15pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (3, 5) { amber } else { rgb("#d9f0d3") },
  table.header([Check], [Complete fixed-family evidence], [Outcome]),
  [Fine-step accuracy], [6720 comparison records include coarse diagnostics; all 3360 fine-step target gates pass 10⁻¹⁰ against qualified Float256 AD and original-quad dot/triple anchors. Maximum original-reference errors: first ≈4.63100 × 10⁻¹², second ≈5.76765 × 10⁻¹².], [Pass],
  [Resolved convergence], [420 transitions improve by at least factor16; the fixed nonpolynomial scalar finite first-mass witness resolves. Coarse errors reach roughly 2 × 10⁻⁸ and are not tested against the final fine-step accuracy threshold.], [Pass],
  [Explicit floor cases], [1260 transitions remain floor/unresolved. The floor is max(32 × AD256/quad gap, 32 × AD128/256 gap, 2⁻⁸⁰). It classifies convergence and never relaxes the final 10⁻¹⁰ accuracy requirement.], [Retained],
  [Actual ordinary data], [All 4200 parent complex slots, five parent rows, twenty N5 child rows and 25 full native contexts are observed. Input preservation, child origins, same-rank subsets and explicit KnownNoQuadrature status are checked.], [Pass],
  [Preparation history], [The first Symbolica weight construction fails on a 0⁰ matrix input and produces no weights. The reviewed repair uses literal one for the constant row, with the same Matrix solver and exact residual checks. The original failure remains unchanged.], [Retained],
))

The stdlib replay reproduces the complete original comparison, observer and
convergence report from immutable supplied bytes. It executes no solver,
compiler, native module or integral. The PV parent supplies the already
qualified AD/reference/source context; separate Hybrid and typed capsules are
unchanged. Native identities and process cleanup are historical metadata.

Finite differences and AD share integral software; the original IREGI anchors
retain their separate independent admission. This real massive N6/rank-six,
unit-power, h=0 PV family validates first/second derivatives in one mass
direction. It does not certify arbitrary higher/mixed directions, branches,
imported quadrature budgets or rigorous error bounds. No performance
measurement or complete-goal claim is added.

#text(size: 8pt)[Source: #raw("validation/public-n6-rank6-mass-fd").
All following sections retain earlier source checkpoints and their original
failures; later scoped successes above do not rewrite those outcomes.]

#pagebreak()
= Earlier standalone Python facade (wheel003)
A newly built standalone wheel uses one native host for the shared Symbolica,
Numerica, FeynKit and Spenso PyO3 classes. Its installed tests exercise the
public shared tensor facade, ordered Dual layouts and the explicit checked
scalar-product option. These frontend results qualify their captured source
context; the full generic rank-six mathematical gate remains open.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 5 { amber } else { rgb("#d9f0d3") },
  table.header([Executed check], [Current wheel003 evidence], [Outcome]),
  [Installed API], [64 cases: 25 upstream numeric precision/Float, 29 existing native-object interoperability/numerical cases and ten facade cases. Consumer typing has zero errors and warnings.], [Pass],
  [Actual marimo runtime], [Seven cases execute graph rendering, numerator preparation, batched contraction, dots, Duals, beyond-binary64 inputs and the existing benchmark explorer with the installed wheel.], [Pass],
  [Routing and guards], [Eligible six-point profiling and primal-only Dual workspaces select #raw("affine_power"). Checked ordinary, profiling and primal-only Dual calls reject inconsistent supplied products before scalar masters; empty/reset counters are checked.], [Scoped pass],
  [Source and saved replay], [#facade-wheel003.source_files selected source files. The saved-only replay passes #facade-replay003.checks checks over source/evidence bindings, test cases, consumer typing, process cleanup metadata and retained failures.], [Pass],
  [Domain limits], [This frontend cohort alone does not qualify complete checked rank-six or six-point mixed Dual arrays. Other platforms, Community-host loading and current browser exports remain unqualified. Existing N6 rank-one and N7 scalar cases are frontend regressions.], [Limited],
))

The first wheel build failed after external Nix Rust/Cargo paths disappeared.
The exact preceding 1.99 toolchain was restored and protected with project GC
roots; all 146 common frozen inputs agree across the failed and successful
builds. The first environment installation then hit an offline cache miss for
pinned marimo 0.24.0. Installing the same pins with downloads allowed succeeded.
Both original failed receipts remain failed.

The aggregate eight-case dispatch phase reached its fixed deadline after
seven exact pytest nodes had passed. Its final checked-Dual case was unfinished.
Only that case was completed with its full per-case budget; it passed without
changing numerical assertions or accuracy guards. The partial raw log and
failed aggregate receipt remain archived. The 64 API cases count each exact
successful node once.

The capsule records the wheel/native identities but contains no compiled
wheel, native binary, environment or cache. Its 147-file source selection is
not a full compiler/native/system closure. Portable replay starts no native
host, integral or benchmark. Historical wheel002 and wheel004 evidence below
retains its original contexts and measurements.

#text(size: 8pt)[Source: #raw("validation/python-interface/facade-wheel-003").
Linux CPython3.13 development build only; no new timing or performance gate.]

#pagebreak()
= Dotted rank-six tensors: N1 and N4
The first-propagator power-two and power-three families now have complete
TIRust rank-zero through six arrays at Float128 and Float256 for N1 and N4.
Every valid lane returns 210 Cartesian moments and four separated native
Laurent slots. The fixed massive spacelike inputs use h=0 and requested Hybrid.
This is actual dotted tensor evaluation, beyond the earlier reference anchors.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 5 { amber } else { rgb("#d9f0d3") },
  table.header([Check], [Saved complete evidence], [Outcome]),
  [Eight actual calls], [Four N1 and four N4 power-two/power-three calls. Each batch has base, exact-duplicate and scale-two valid lanes around an invalid-mu neighbor. All 20160 requested complex slots are present.], [Pass],
  [Reference and precision], [20160 original-quad comparisons at normalized tolerance 10⁻¹⁰ and 10080 Float128/256 pairs at 2⁻⁸⁰. Complete arrays are compared before any numerator contraction.], [Pass],
  [Batch and pole controls], [6720 exact duplicates, 6720 scale identities, 10080 massive IR zeros, 2046 UV zeros and 12240 parity zeros. Together with reference/precision checks: 68046 main gates and zero numerical failures.], [Pass],
  [Mass-Taylor consistency], [A separate ordinary-Dual path passes 10080 saved mass-Taylor identities per N, 20160 total. This is agreement between distinct TIRust paths; it is not an independent integral oracle.], [Scoped pass],
  [N5 completion gap], [The first-dot Float128 call reaches its fixed 810-second preparation limit under 32 GiB, with no array. Its Float256 partner and both power-three cases remain unrun.], [Open],
))

The initial N4 aggregate controller stopped at a deliberate barrier after
three completed cases. Only the previously unstarted fourth case was executed
with the corrected per-capture budget. Its original partial receipt remains
unchanged. The N5 timeout likewise remains failed, with input guards and
owned-process cleanup recorded; missing slots are never successful zeros.

The source-bound capsule retains all #low-N-generalized.counts.actual_selected_compiled_sources
selected compiled source bodies for its actual frozen CLI. Its source context
predates the later wheel003 CLI and does not claim equivalence to that binary.
The portable saved-only replay reconstructs the full original Fraction review
and separate mass-Taylor reports in a minimal copy without the external target
tree or a license. It starts no integral evaluator.

The original finite reference coefficients are empirically admitted. Pole and
parity identities do not independently certify every finite coefficient or
provide rigorous error bounds. These h=0 real-mass families do not qualify
shifted dimensions, arbitrary kinematics, complex masses, N5/N6 dotted closure,
or the complete rank-six objective. No timing measurement is added.

#text(size: 8pt)[Source: #raw("validation/low-n-generalized-rank6-current-cli").
Actual CLI #raw("f69615f9…"), coherent build002; historical failures retained.]

#pagebreak()
= Checked supplied products: six-point rank six
The opt-in checked Cartesian interface retains the supplied scalar products
and proves compatibility before evaluating child engines. Its newer packed
implementation now returns complete ordinary N6 rank-six arrays at Float128
and Float256, in addition to the earlier guard and rejection evidence.
The numerical scope is explicit PV, unit powers, h=0, Preserve and native
Cartesian components.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (4, 5) { amber } else { rgb("#d9f0d3") },
  table.header([Check], [Source-bound executed evidence], [Outcome]),
  [Complete rank-six output], [All 840 native complex slots at each working precision pass admitted original-quad comparisons at 10⁻¹⁰. Both maximum normalized reference errors are approximately 2.10018 × 10⁻¹⁶.], [Pass],
  [Working-precision pairs], [All 840 Float128/256 pairs pass 2⁻⁸⁰ normalized agreement, maximum 2.13539 × 10⁻³². Requested precision is 53 for both runs. This does not establish achieved 80/128/256-bit accuracy.], [Scoped pass],
  [Guard and batch lifecycle], [20 exact named normal tests plus an actual rank-one mixed-validity/full-array/contraction/recovery test. Its two valid rows use one packed child call with 12 child rows and one parent call; the invalid neighbor remains failed.], [Pass],
  [Earlier resource failures], [The old inline rank-six run has no completed array and a sampled last RSS near 145.7 GiB. A separate constructor test hits its fixed 60-second limit before immutable-variant reuse. Both failures remain archived.], [Retained],
  [Unqualified domains], [Generalized supplied-G powers/dimension shifts, numerical Dual derivatives, other conventions, default Hybrid and arbitrary kinematics remain outside this fixed numerical cohort.], [Open],
))

The compatible full-Taylor certificate is checked before child compilation.
Zero cofactors are masked before singular pinches are prepared. Child outputs
keep their supplied products and native UV/IR split; nonzero components and
poles are not projected away. The accepted native convention adapter is
covered by focused source/tests. The earlier normalization draft and its
positive source review are preserved together with their retraction; that
rejected packet was never compiled or numerically evaluated.

The source capsule reconstructs all #checked-G-unit.compiled_source_count
selected compiled-source hashes using 110 byte-identical wheel003 files and
35 own additions. Its checker imports no producer or native library and
recomputes all 1680 quad records, 840 pair records and 840 strict pair decisions
with exact Fraction arithmetic. Binary identities and historical cleanup are
saved attestations, not fresh binary or process probes.

The old inline timeout retains its original cleanup errors; a later additive
release records historical process-group absence. The RSS value is a sample,
not a measured peak or a performance result. No new timing ratio, speedup,
rigorous error bound or complete-feature claim is added.

#text(size: 8pt)[Source: #raw("validation/checked-supplied-g-unit-rank6").
Generalized and derivative runtime closure remain separate requirements.]

#pagebreak()
= Affine denominator descent: ordinary N6 rank six
The new dynamic positive-power engine uses a Symbolica-derived constant-one
denominator identity to descend to packed child topologies. It retains the
tensor rank, denominator powers, dimension and numerical component layout.
The ordinary N6 rank-six unit-power family now has complete Float128/256
arrays; its dotted and D6 resource gates remain open.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (4, 5) { amber } else { rgb("#d9f0d3") },
  table.header([Check], [Frozen actual engine evidence], [Outcome]),
  [Complete ordinary arrays], [All 840 complex slots per Float precision pass admitted original-quad comparisons at 10⁻¹⁰. Maximum normalized reference error is 2.10018 × 10⁻¹⁶. Separate controls check 840 IR zeros and 420 power-counted UV zeros across both captures.], [Pass],
  [Precision and batching], [840 Float128/256 pairs pass 2⁻⁸⁰, maximum approximately 1.80270 × 10⁻³¹. One affine profile has four DAG edges and four packed N5 rows in one child batch; 120 master rows use four families and the parent has no scalar masters.], [Scoped pass],
  [Actual source tests], [56 exact focused tests, a preparation-only guard and the core/CLI/public-example/PyO3 feature build pass. All 145 selected build-source copies are retained. PyO3 compile checking is separate from the installed wheel runtime cohort.], [Pass],
  [Generalized requests], [Both first-dot Float128 and D6 unit-power Float128 reach their fixed 810-second preparation deadlines without tensor arrays. Their Float256 partners remain unrun; cleanup and unchanged-source guards pass.], [Open],
  [Method scope], [The outer request is IBP; child execution is explicitly Passarino–Veltman completion. Raw method tags, preparation notes and diagnostics are retained. This is no pure-IBP or complete generalized-domain qualification.], [Limited],
))

The structure supports denominator descent without an imposed small point-count
limit. Its actual validation envelope is narrower: this fixed ordinary N6
family does not establish arbitrary-N numerical behavior. Resource budgets,
singular identity charts and exceptional kinematics still require explicit
diagnostics and qualification. The requested rank remains independent of
the number of denominators during descent.

The capsule retains three initial test-only Symbolica replacement-API compile
errors, the successful corrected build, the listing-only test failure and its
single module-path correction. No failed integral is relabeled successful.
Both generalized timeouts retain their original terminal receipts and empty
outputs; portable replay succeeds only when those failures remain intact.

Independent saved arithmetic checks all arrays and precision pairs. Native
products are represented by their recorded omitted identities; the checker
does not rehash absent executables or probe current processes. The original
quad reference has 113-bit storage with its original approximately 16-digit
constants and stopping criteria. Agreement at 2⁻⁸⁰ between TIRust outputs is
not a certified achieved-accuracy bound. No Dual, physical or timing claim is
added by this capsule.

#text(size: 8pt)[Source: #raw("validation/affine-power-generalized-current-core").
Complete generic rank-six positive-power/dimension-shift closure remains open.]

#pagebreak()
= Current CLI and physical graph integration
The current source-bound CLI passes fourteen tests. Complete rank-four tensor
and contracted physical checks pass for one top-loop $g g arrow H H$ box
and one $g g arrow H H H$ pentagon. Both graphs have seventy Cartesian moments,
with all four native Laurent slots retained before contraction. These are
fixed-graph spot checks, separate from the rank-six synthetic families.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 5 { amber } else { rgb("#d9f0d3") },
  table.header([Executed check], [Complete saved evidence], [Outcome]),
  [Current CLI], [14 actual tests pass from the wheel003 build context. The capsule binds the exact CLI/test products, inherited 147 build sources and twenty additional physical input/helper/controller files.], [Pass],
  [Box N4 / rank4], [Float128/256: 1680 original-IREGI tensor comparisons against IBP, PV and Hybrid; maximum normalized error 1.144 × 10⁻¹³. Largest MadLoop merged-coefficient error 4.766 × 10⁻¹⁵.], [Pass],
  [Pentagon N5 / rank4], [Float256/384: 1120 original-IREGI tensor comparisons against PV and Hybrid; maximum error 4.616 × 10⁻¹⁴. Largest MadLoop merged-coefficient error 9.291 × 10⁻¹⁴. Historical original-IBP disagreement is not retested here.], [Pass],
  [Exact precision pairs], [Each graph passes 280 tensor and four contracted Laurent checks, 568 total, using saved decimal strings and exact Fraction arithmetic at 2⁻⁸⁰. Maximum normalized L1 differences: box 7.952 × 10⁻³⁶; pentagon 6.145 × 10⁻⁵⁵.], [Scoped pass],
  [Retained pentagon failure], [Float128 / requested53 hits the conservative cancellation guard at #raw("pv_node_852"), with no tensor array or subsequent reference comparison. Prospective Float256/384 runs pass at unchanged requested accuracy.], [Failed128],
))

The current CLI requests Hybrid and executes PV in these successful cases.
The external comparison threshold remains dimensionally normalized 10⁻¹⁰,
using binary64 reference/comparison arithmetic. MadLoop uses isolated graph
groups two and five, fixed equal gluon colors, the retained helicity-one inputs
and graph factor −1/2. Its three Laurent coefficients combine UV and IR single
poles; complete original-IREGI tensor checks retain their separation. Bare
UV/IR poles and reduction-generated R1 are included; R2 and UV counterterms
are excluded. The cohort reuses the documented numerator/routing correspondence.

Precision-pair checks use $Q = 600$ for the box and $Q = 750$ for the pentagon,
tensor dimension $4 + r - 2 N$, and contracted dimensions zero and minus one.
The normalized L1 delta uses the larger of $Q^d$ and higher-precision coordinate
magnitudes. Pair agreement is not a certified eighty-bit accuracy bound.

The failed first controller preserves its unrun planned pentagon256 and
128/256 pair stages. The independent higher-precision receipts do not promote
its failed128 result. The capsule's saved replay checks all #current-physical.external_tensor_complex_checks
external tensor slots, #current-physical.madloop_merged_complex_checks MadLoop merged slots and the exact
pairs without evaluating an integral or starting a compiler. No new physical
numerator export, rank-six physical coverage or performance claim is added.

#text(size: 8pt)[Source: #raw("validation/current-cli-physical-integration").
The previous timing measurements and their failed cases remain unchanged.]

#pagebreak()
= Integration scope and remaining functional gates
This is an intermediate implementation checkpoint. The complete generic
rank-six objective remains open. The latest source integration adds per-call
reuse of repeated primal-amplitude bound expressions using the existing
Symbolica operations. Its focused tests qualify that narrow change; they
do not turn incomplete dotted or dimension-shifted requests into successes.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 15pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 2 { rgb("#d9f0d3") } else { amber },
  table.header([Area], [Executed scope or required closure], [Status]),
  [Primal-amplitude memo], [Six focused tests pass: alias-environment isolation, repeated conditional reuse, lazy inactive zero denominators, conditional path unions, nonlinear-master rejection and dimensional/Taylor-axis normalization.], [Focused pass],
  [Source/build bridge], [Both actual 119-source builds are retained. The saved checker recomputes equality of the entire production-module prefix between the builds; the later change is test-only. Core/CLI and the revised test executable compile successfully.], [Pass],
  [Positive powers / shifts], [N1/N4 first-dot/triple arrays pass; current N5 first-dot and N6 first-dot/D6 preparations remain failed in their frozen cohorts. Their unavailable arrays and unrun precision partners remain visible.], [Open],
  [Agreed numeric / API], [Public highpoint higher/mixed Duals, generalized checked supplied products, accepted Legacy runtime closure and remaining exceptional/method coverage still require actual numerical qualification.], [Open],
  [Completion and speed], [Commit the feature-full tested version with an updated report when the functional gates pass. Higher-rank optimization and low-rank COLLIER parity are later work; no new performance result is added here.], [Pending],
))

The initial memo stress fixture used forty nested conditional levels and
timed out before completing. It remains failed. A smaller representative
ten-level fixture explicitly records construction and cached traversal, with
21 unique amplitude recipes; all six focused tests pass. This replacement
does not establish that the original forty-level construction now succeeds
or supply a before/after runtime comparison.

The memo source follows the earlier complete tensor and physical capture
contexts. Source binding is per capsule: the focused build/test bridge does
not automatically confer their full numerical qualification on later edits.
The capsule claims no complete dotted array, current-CLI equivalence,
physical validation, achieved-precision bound or performance improvement.

Six new immutable cohorts are summarized before the historical sections.
All 395 preceding top-level result values and their original JSON spelling
are preserved, including unsuccessful cases and every historical timing.
Saved-only replay reconstructs the declared evidence and existing numerical
checks; it executes no compiler, native library, integral or benchmark.

#text(size: 8pt)[Latest narrow source evidence:
#raw("validation/primal-amplitude-memo-current-core"). Current completion scope:
#raw("docs/data/rank6-current-checkpoint-20261026.json").]

#pagebreak()
#let rank6-feature = data.rank6_feature_checkpoint.snapshot
= Rank-six feature validation: scoped progress
The completion target is generic cumulative tensor rank≤6, including positive
propagator powers, dimension shifts and the agreed numeric/API features. N≤6
is the current validation checkpoint; higher-point denominator descent needs
separate qualification. This checkpoint adds evidence from ten frozen source
cohorts. It does not establish their equivalence to later source changes or
close the complete mathematical acceptance gate. All prior result values and
timings remain unchanged; no performance measurements were added.

#report-table(table(
  columns: (1.2fr, 3.3fr, 0.9fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Evidence], [Complete saved checks], [Scope]),
  [N2 first/mixed jets], [One massive spacelike rank-seven family: 330 moments and all four native Laurent slots. Static/dynamic Float128/256 jets, ordinary f64/DoubleFloat controls, and an independent Float384 finite-difference route.], [Scoped pass],
  [Derivative identities], [#rank6-feature.N2_numeric.initial_original_comparisons original comparisons; #rank6-feature.N2_numeric.precision_pairs precision pairs; #rank6-feature.N2_numeric.mass_identities mass identities; #rank6-feature.N2_numeric.scale_identities scale identities; #rank6-feature.N2_numeric.finite_difference_components finite-difference components.], [Pass],
  [Beyond 53 bits], [Exact 1 + 2⁻⁵⁴ inputs: #rank6-feature.N2_numeric.beyond53_secant_slopes secant slopes and #rank6-feature.N2_numeric.beyond53_precision_pairs raw precision pairs. DoubleFloat/Float retain the distinction; binary64 rejects the unrepresentable input before planning.], [Pass],
  [Low-N dotted anchors], [Six original-quad N1/N4/N5 rank-six first-dot/triple arrays: #rank6-feature.low_N_reference.complete_complex_slots complex slots and #rank6-feature.low_N_reference.independently_replayed_records reference/control records.], [Anchors],
  [Remaining derivative scope], [Typed f64/DoubleFloat Dual carriers are not covered by the N2 cohort. Full public highpoint Dual and broader tensor derivative domains remain unqualified.], [Open],
))

The N2 source capsule records exact rational inputs and outputs, component
precision, signs, invalid lanes and the original failed build before its narrow
borrow correction. The finite-difference route uses Symbolica-generated jets,
points and weights. Initial comparison thresholds remain 10⁻¹⁰, precision/DD
controls 2⁻⁸⁰ and the final finite-difference target 2⁻⁷⁰. This one-family
historical rank-seven result supplies useful rank-six controls; it is not a
blanket qualification of all N≤6 families.

The low-N quad arrays are reference anchors. Storage at 113 bits / 33 decimal
digits does not establish 33-digit accuracy. Finite generalized coefficients
are not independently certified merely because a reference returned a stable
flag, and these six arrays alone do not qualify TIRust dotted reduction.

#text(size: 8pt)[Saved sources: #raw("validation/rank7-numeric-jets"),
#raw("validation/quad-original-iregi-low-n-rank6-dot-triple"). Their portable
checkers reconstruct declared source snapshots and replay saved arithmetic;
they execute no compiler or integral. Timings in their receipts are engineering
diagnostics.]

#pagebreak()
= Six-point runtime: Hybrid and precision rescue
At one real-mass N6 rank-six unit-power point, all 210 cumulative Cartesian
moments and four native Laurent slots are available. The newer Hybrid and
specialized precision-rescue cohorts close named ordinary runtime paths.
Earlier generic, dotted and dimension-shift failures remain visible in their
original frozen context.

#report-table(table(
  columns: (1.25fr, 3.35fr, 0.85fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Path], [Executed result], [Outcome]),
  [Hybrid primary], [Float128 / requested53: #rank6-feature.six_hybrid.reference_comparisons complex coefficients pass admitted original quad IBP; all #rank6-feature.six_hybrid.exact_PV_delegations equal explicit specialized PV exactly. Maximum normalized reference error 9.3341 × 10⁻¹⁷.], [Pass],
  [Hybrid orchestration], [#rank6-feature.six_hybrid.focused_tests focused tests: batched Unstable-only retries, coupled component/context scatter, lane isolation, counters, lazy shared plans and actual low-N IBP work. Invalid-mu and empty/reset lifecycle assertions pass in the N6 test.], [Pass],
  [Public f64 rescue], [Specialized SixPoint: output53, working128, verification256. Primary cancellation remains recorded; accepted output passes 840 quad comparisons and 630 separate zero-pole checks. Maximum reference error 1.8670 × 10⁻¹⁶.], [Scoped pass],
  [Earlier restricted CLI], [One explicit-PV DoubleFloat array passes 840 references. Five other attempts fail or remain incomplete: generic f64 unit, generic rescue, specialized f64, first-dot Float128 preparation and unit D6 Float128.], [1 / 6 pass],
))

Hybrid preserves the requested method and uses specialized PV as primary. On
this point its outer generic IBP fallback has *zero* rows, batches and
preparation attempts. Existing nested PV child retries remain visible. The
focused tests verify fallback mechanics, including real low-N IBP execution;
they do not prove a successful genuine N6 IBP fallback on this fixture.

The first N6 Hybrid invocation passed coefficient comparisons but failed an
incorrect observer assertion about nested flags. Its failure is retained. The
corrected native test passed; its controller then failed to parse a payload
sharing libtest’s test-name line. Additive saved-only extraction recovered the
unique complete raw payload without another integral evaluation. Both original
failed receipts remain failed. Independent replay checks the extracted arrays.

The precision-rescue CLI exports the final f64 array, not the internal
128/256 arrays. Its internal convergence acceptance is supported by the
executed shared Rust policy and stage counters; it is not independently
reconstructed from absent arrays. Converted scalar sensitivity context is
explicitly unavailable rather than replaced with stale data. Neither cohort
qualifies public highpoint Dual, dots, D6, complex/exceptional cases or speed.

#text(size: 8pt)[Saved sources: #raw("validation/six-point-hybrid-rank6"),
#raw("validation/public-six-point-precision-rescue-rank6"),
#raw("validation/n6-rank6-generalized-current-cli"). Historical process absence
is replayed as saved metadata; portable checks do not recreate PID history.]

#pagebreak()
= Thresholds, complex masses and retained collinear failure
The orthogonal CLI corpus is *partial*: 12 of 14 Float128/256 calls return
complete rank-six arrays. The two collinear zero-Gram N3 reductions fail with
nonfinite coefficients in both PV and IBP fallback. Their 1,680 unavailable
complex slots are not reported as successful zeros.

#report-table(table(
  columns: (1.2fr, 3.35fr, 0.9fr), inset: (x: 7pt, y: 14pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Domain], [Saved outcome], [Status]),
  [Threshold / complex], [Three equal-mass N2 threshold positions and lower-half-plane complex squared masses for N2/N3/N4, each at Float128/256. Unit powers; all 210 moments and four native slots.], [12 pass],
  [Independent checks], [#rank6-feature.orthogonal_partial.reference_comparisons references; #rank6-feature.orthogonal_partial.precision_pairs precision pairs; #rank6-feature.orthogonal_partial.exact_duplicate_checks duplicates; #rank6-feature.orthogonal_partial.separate_IR_checks IR and #rank6-feature.orthogonal_partial.convergent_UV_checks UV controls; #rank6-feature.orthogonal_partial.branch_checks scalar branch anchors. Total #rank6-feature.orthogonal_partial.total_numeric_checks.], [Pass],
  [Collinear oracle], [A separate order48 parameter reference admits 210 rank-six moments after #rank6-feature.collinear_anchor.refined_record_checks independent record checks. Maximum empirical convergence indicator 4.1716 × 10⁻¹⁴.], [Anchor only],
  [Earlier collinear TIRust], [Two complete reduction failures retained, no tensor arrays in this frozen CLI. The following page records a newer direct-core affine prepass on the same reference family.], [Failed],
  [Completion gates], [Generalized N6 arrays, highpoint Dual, checked supplied-G / accepted Legacy runtime closure, exceptional reductions and agreed physical/API coverage remain open.], [Pending],
))

Maximum normalized reference error among available orthogonal arrays is
2.629 × 10⁻¹⁵; the precision-pair maximum is 7.167 × 10⁻³⁶. Thresholds remain
10⁻¹⁰ and 2⁻⁸⁰. Below-threshold batches include duplicate valid samples around
an invalid-mu neighbor: valid arrays agree exactly and invalid output is null.
Requested Hybrid executes actual PV through the Generic engine in this frozen
cohort. Its successful rows use no fallback or precision rescue.

The original N4 complex quad result is rejected despite its stable flag and
zero finite scalar. N4 instead uses the independently audited order32 parameter
oracle. The separate collinear order48 anchor retains six failed coarse
convergence decisions and all 330 originally stored moments, while admitting
only the 210 rank-six moments. Both parameter-reference qualifications use
empirical convergence, not rigorous error enclosures.

The captured CLI predates later supplied-G, Legacy and Hybrid source changes.
It records requested precision, without per-component precision or numeric
before/after input witnesses. It therefore does not qualify signed-zero/input
preservation, threshold derivatives, Duals, broad exceptional coverage or later
source revisions. No benchmark timings or physical acceptance are added.

#text(size: 8pt)[Saved sources: #raw("validation/orthogonal-rank6-current-cli"),
#raw("validation/collinear-rank6-parameter-oracle"). Every cohort retains its
failures and declared source context. Full N≤6/rank≤6 acceptance remains open.]

#pagebreak()
#let numeric-rank6 = data.rank6_numeric_types_qualification.snapshot
#let collinear-core = data.rank6_affine_collinear_qualification.snapshot
= Rank-six numeric types and the collinear core fix
The newer collinear core uses a Symbolica-derived constant-denominator identity
before a singular parent Gram inverse or master instruction is evaluated.
It retains the original inputs and batches all accepted lanes through the
selected child recipe. Its numerical evidence is separate from the earlier
failed CLI cohort and the independent parameter oracle.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 11pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y in (3, 6) { amber } else { rgb("#d9f0d3") },
  table.header([Executed family], [Complete checks and limits], [Outcome]),
  [N1–N5 numeric], [#numeric-rank6.successful_capture_count complete captures from #numeric-rank6.capture_count actual attempts. Ordinary f64/DD controls and specified static/dynamic first/mixed Dual layouts; Float128/256 mixed components include N5.], [Scoped],
  [Saved component math], [#numeric-rank6.complete_complex_comparisons complex checks: 216720 quad, 60480 precision pairs, 55440 DD, 40320 f64 and 73920 scale identities.], [Pass],
  [Precision guards], [Unrescued N4 f64 mixed Dual and ordinary N5 f64 requested53 / DD requested96 remain guarded failures. The next page qualifies a separate public N4 rescue. No tolerance is relaxed.], [3 retained],
  [Collinear N3], [Float128/256: all 840 coefficients per precision against the independent anchor; 840 precision pairs. Selected first-dilation Dual: all 840 primals and 840 jets per precision; 1680 pairs.], [Pass¹],
  [Focused core], [16 affine, 10 component-sensitivity and 4 checked-facade tests pass. Geometry/Taylor rejection occurs before masters; batch isolation, metadata, demand masks and mixed products are checked.], [Scoped],
  [Symbolic stress], [Two fully symbolic general-N rank-four preparation tests exceed their fixed 90-second limits. Three initial build/controller failures remain preserved.], [Retained],
))

¹ Actual source-bound Rust assertions use the independent order48 anchor and
existing Symbolica dilation response, with normalized tolerance $10^(-10)$
and precision-pair threshold $2^(-80)$. Invalid-scale neighbors, equal duplicate
outputs, unchanged inputs and one grouped child batch are checked. Numerical
arrays were not exported: portable replay verifies the captured source, tests
and execution records; it cannot independently recompute absent raw arrays.
General Dual uncertainty, arbitrary directions and general-N reduction are
not qualified by this one collinear profile.

The separate numeric-type capsule retains one real massive spacelike
unit-power, h=0, PV family per N. DD/Float controls keep the $2^(-80)$ gate.
The N5 mixed Float precision-pair maximum is $5.424 times 10^(-31)$.
The saved Fraction checker imports no producer. Its author previously wrote
the four N2 prefix adapters; that overlap is disclosed rather than presented
as author-independent evidence. The other twenty captures have another
producer. Five saved mutation probes reject semantic corruptions.

Each cohort reconstructs its own captured sources. The current-core capsule
retains 349 compiled source bodies and the later formatting-only transition;
the numeric cohort uses its earlier frozen example. These are correctness
checks, with no new physical or timing qualification and no full-goal claim.

#text(size: 8pt)[Sources: #raw("validation/rank6-numeric-types"),
#raw("validation/affine-collinear-current-core"). Generalized high-point
tensors, public high-point mixed Duals and remaining API domains stay open.]

#pagebreak()
= Public box rank-six mixed Dual rescue
The public #raw("TensorRescueWorkspace::new_dual") now has an actual binary64
box check with all 210 moments, four Laurent slots and six ordered Taylor
components. It evaluates a static input and a dynamic three-lane batch with
two valid scales and an invalid-scale neighbor, then checks empty-state reset.
Higher-precision evaluation preserves the represented input exactly.

#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 15pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 5 { amber } else { rgb("#d9f0d3") },
  table.header([Check], [Executed evidence], [Outcome]),
  [Complete mixed output], [Three valid lanes each return all 840 × 6 complex values. Independent saved arithmetic checks 15120 quad responses, 15120 qualified Float128 controls and 5040 scale identities.], [35280 pass],
  [Reference accuracy], [Maximum normalized quad error 1.673 × 10⁻¹⁶; Float128-control error 9.572 × 10⁻¹⁷. Both retain the 10⁻¹⁰ gate.], [Pass],
  [Real fallback], [All three primary lanes hit the original recurrence cancellation guard in Taylor component 2. Working128 and verification256 report convergence; the invalid lane is not rescued.], [Pass],
  [Numeric integrity], [30240 real/imaginary coordinates are exactly binary64-representable with recorded 53-bit storage. Static/dynamic component ordering, actual counters and empty reset are checked.], [Pass],
  [Unexported context], [Internal 128/256 arrays and generic detailed stage reports are not exported. Accepted scalar-sensitivity context is explicitly unavailable.], [Limited],
))

The independent checker verifies returned values, status/counter associations
and the original primary failure. Internal higher-precision convergence is
observed through the executed policy and its status; the absent internal arrays
cannot be independently compared. Nearest rounding of the hidden verification
array likewise cannot be reconstructed from an external Float128 control.
The latter's 3326 exact-rounding mismatches are retained as diagnostics,
including values near roundoff zeros; normalized agreement is unchanged.

The first saved observer incorrectly required exact nearest rounding from that
external control. Its failure is preserved. The corrected observer retains
the numerical tolerance, exact binary64 representation, precision, metadata and
failure gates. The earlier example-build type error and its one-line harness
repair are also retained. Neither correction changes a reduction formula.

This is one massive spacelike, unit-power, h=0 PV family. It qualifies the
specified public box rescue, not high-point mixed Duals, arbitrary branches,
generalized tensors, rigorous error bounds, physical numerators or performance.
Its actual 106-source context and fresh minimal replay are retained separately
from the older ordinary binary64 failure and the later unfinished descent code.

#text(size: 8pt)[Source: #raw("validation/public-four-point-dual-rescue-rank6").
All prior unsuccessful requests remain unsuccessful in their original cohorts.]

#pagebreak()
= Shared Python facade and native operand precision
#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y <= 3 { rgb("#d9f0d3") } else { amber },
  table.header([Check], [Captured scope], [Outcome]),
  [Installed wheel002], [#facade-replay002.passed_installed_tests tests; consumer typing has #facade-replay002.typing_errors errors and #facade-replay002.typing_warnings warnings. Shared Symbolica, Numerica, FeynKit and Spenso classes use the public tensor facade.], [Pass¹],
  [Native operand precision], [Six new upstream regressions plus 19 existing Float/ComplexFloat tests. The installed reproducer retains the high-precision imaginary component when the real component has fewer bits.], [Pass¹],
  [Vendor reproduction], [Pinned upstream 9cdc2949 plus six committed patches reproduce all #native-operand-vendor.file_count retained files, totaling 9,601,877 bytes. Final observer exclusions leave the captured wheel arithmetic unchanged.], [Pass],
  [Earlier failures], [The trap wheel and two failed trait-observer stages remain archived. The final stage changes observer exclusions only; numerical assertions, tolerances and the installed wheel remain unchanged.], [Retained],
))

¹ One local Linux CPython3.13 development wheel. Its 60 tests comprise 25
upstream numeric tests, 29 existing interoperability/numerical tests, four
generic-facade tests, one six-point rank-one frontend test and one seven-point
scalar frontend test. They exercise explicit Dual layouts, profiling, engine
tags, packed batching, sparse contraction, invalid-lane isolation, empty/reset
state, snapshots and recovery. They do not constitute 60 independent physical
integral comparisons or qualify the complete high-point numerical domain.

The precision correction concerns native f64/complex operands in upstream
ComplexFloat arithmetic. A value with two real bits and 128 imaginary bits
retains the latter precision and agrees with the explicit 128-bit operand
control. The public #raw(".precision") property still reports the minimum
component precision. Arbitrary Python-integer coercion is not qualified here.
The source manifests and reproducer flags remain in their original spelling;
their captured native-operand scope is not widened.

The source archive reconstructs the 116 inputs owned at wheel-build time.
The Rust trait observer's #raw("float.rs") read was outside those original
guards; its later copy is retained as post-observation context. The successful
observer excludes Rust-only #raw("real_classify") and mutating
#raw("set_precision"); Python uses immutable #raw("with_precision") copies.
This is not a complete compiler/native/system closure or a new wheel build.
Other platforms, release performance, separate Community-host loading and
broader numerical frontend acceptance remain unqualified.

Saved-only replay:
#raw("python3 -I -S -B validation/python-interface/facade-wheel-002/check.py").
Exact six-patch reproduction is bound by
#raw("validation/python-interface/complex-native-precision-vendor-reproduction.json").

#pagebreak()
= Complex tensor oracle and generalized N6 references
#report-table(table(
  columns: (1.25fr, 3.4fr, 0.85fr), inset: (x: 7pt, y: 13pt), stroke: table-stroke,
  fill: (x, y) => if y == 0 { navy } else if y == 3 { amber } else { pale },
  table.header([Reference family], [Complete saved evidence], [Admission]),
  [Complex N2/N3/N4], [#complex-parameter-oracle.complete_grids fixed grids, #complex-parameter-oracle.point_rows point rows and #complex-parameter-oracle.complete_native_complex_slots complex Laurent slots. Orders 12/20/32 and both vertex maps are retained.], [Scoped],
  [Independent saved audit], [43,566 exact comparison records, 11,880 separate IR-zero checks and 15 negative controls pass. Includes 15,840 eligible N2/N3 quad comparisons and 27,720 order/map indicators.], [Pass¹],
  [Original complex N4], [Native stable flag is true, but the scalar finite coefficient is exactly zero despite a positive parameter-integral bound. The entire raw array remains diagnostic and enters no agreement gate.], [Rejected],
  [Generalized N6 / rank6], [#generalized-quad-six.complete_arrays original-quad arrays / #generalized-quad-six.complete_complex_slots complex slots: unit powers, first/last dot2, first power3, mixed dots2/2 and unit powers in D2/D6.], [Anchors²],
))

¹ The ordinary complex fixture uses mu=3, real spacelike shifts and squared
masses in the lower half-plane with Re(F)≥1. All 330 moments through rank7
and native UV1, IR1, IR2, finite coefficients are retained; the first 210
moments are the rank6 prefix. Symbolica prepares Gamma Laurent coefficients
and source derivatives independently of reduction. Native iπ² is included
once, and dimensional normalization uses exact powers of Q=3. The actual
kernel/transport and unchanged Python controls pass 15 and 17 tests.

Binary64 SciPy nodes and worker values are accumulated with longdouble real
and imaginary sums. Maximum N2/N3 reference discrepancy is
$7.008 times 10^(-13)$; the largest retained order/map difference is
$2.776 times 10^(-11)$. All gates retain the mixed $10^(-10)$ tolerance.
These are empirical convergence indicators, not rigorous error bounds or
arbitrary-precision integrated accuracy. The order32 identity-map N4 scalar
finite candidate is approximately −0.04470448539252041802 +
0.20873403285427626204i. Independent saved review admits this fixed empirical
reference; it does not itself qualify agreement with TIRust. The failed initial
E0599 build and bare-environment NumPy import attempt remain failed.

² The seven generalized reference arrays contain all 210 rank6 moments.
Independent replay checks 5,166 records: 2,940 massive IR zeros, 1,386
convergent UV checks and 840 exact ordinary-unit prefix matches. All 84
degree-six D6 UV coefficients remain present. These necessary identities and
controls do not independently prove every finite generalized coefficient or
qualify TIRust/Dual results. Original limited-digit constants and stopping
criteria remain despite quad storage. No validation duration becomes a
comparative timing measurement.

Saved-only replay:
#raw("python3 validation/complex-parameter-tensor-oracle/check.py")\
#raw("python3 validation/quad-original-iregi-generalized-n6-rank6/check.py").
The generic rank-six functional gate remains open. N≤6 is the present
validation checkpoint; it does not establish arbitrary-N denominator descent.
Broader complex, threshold, collinear and tensor-Dual cases need their own
durable comparisons. Supplied-G, Legacy and new Hybrid source work is not
qualified by this checkpoint. No new performance ratios are added.

#pagebreak()
= Rank-seven correctness checkpoint
#let six-rank7 = data.six_point_rank7_current_cargo_qualification.snapshot
#let arrays-n1-n5 = data.rank7_arrays_n1_n5_current_qualification.snapshot
#let arrays-six-life = data.rank7_arrays_six_lifecycle_qualification.snapshot
#let arrays-six-literal = data.rank7_arrays_six_literal_qualification.snapshot
#block(fill: pale, inset: 10pt, width: 100%)[
  *Current objective: generic rank-six tensors, with all agreed features
  including dots.* N≤6 is the current validation checkpoint, not a proof of
  arbitrary-N denominator descent. Retain earlier rank-seven checks and the
  seven-point challenge. Further optimization is deferred. The functional gate
  remains open; historical timings retain their measured contexts.
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
  [Functional objective], [Generic rank≤6: generalized tensors, numeric/Dual, supplied invariants and higher-N denominator descent still need qualification. Current tested checkpoint is N≤6; N7/rank7 is separate stress evidence.], [Open],
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
closing the present N≤6/rank-six checkpoint. Higher-N denominator descent
needs separate checked implementation and finite representative validation.

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
  = Historical wheel004 repair and executed gallery
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
  = Historical wheel004 C0/D0 mass-Taylor identities
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
