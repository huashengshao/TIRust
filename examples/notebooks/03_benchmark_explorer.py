# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="TIRust · benchmark explorer")


@app.cell
def _():
    import marimo as mo
    from _presentation import (
        COHORT_CHOICES,
        benchmark_grid,
        cards,
        hero,
        load_benchmark_cohort,
        load_reference_times,
        project_root,
        ratio_chart,
        theme,
    )
    return (
        COHORT_CHOICES, benchmark_grid, cards, hero,
        load_benchmark_cohort, load_reference_times, mo, project_root,
        ratio_chart, theme,
    )


@app.cell(hide_code=True)
def _(hero, mo, theme):
    mo.vstack([
        mo.Html(theme()),
        mo.Html(hero(
            "Performance, with the evidence attached",
            "Explore measured tensor-array kernels, their uncertainty, and the limits of each comparison. Every TIRust timing ratio uses TIRust in the numerator.",
            eyebrow="TIRust gallery · 03 / benchmarks",
        )),
        mo.md(r"""
        \[
        R=\frac{t_{\mathrm{TIRust}}}{t_{\mathrm{reference}}},
        \qquad R<1\;\Longrightarrow\;\text{TIRust is faster}.
        \]
        The colored cells use a logarithmic scale. Amber marks an inconclusive
        timing gate, even when the estimated ratio favors one implementation.
        """),
    ])
    return


@app.cell
def _(COHORT_CHOICES, mo):
    cohort_choice = mo.ui.dropdown(
        COHORT_CHOICES,
        value="Private selected TIRust / COLLIER · posthoc",
        allow_select_none=False, label="Measured comparison",
    )
    cohort_choice
    return (cohort_choice,)


@app.cell
def _(cohort_choice, load_benchmark_cohort, load_reference_times, project_root):
    checkout = project_root(__file__)
    benchmark_data, benchmark_rows = load_benchmark_cohort(checkout, cohort_choice.value)
    reference_data, reference_rows = load_reference_times(checkout)
    return benchmark_data, benchmark_rows, checkout, reference_data, reference_rows


@app.cell(hide_code=True)
def _(benchmark_data, benchmark_rows, cards, mo):
    mo.vstack([
        mo.Html(cards([
            ("Numerical comparisons", f"{sum(row.numerical_status == 'passed' for row in benchmark_rows)} / {len(benchmark_rows)} passed"),
            ("Timing protocol", benchmark_data["protocol"]),
            ("Source snapshot", benchmark_data["snapshot"]),
        ])),
        mo.callout(
            mo.md(benchmark_data["scope"]),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    topology_choice = mo.ui.dropdown(
        {"All topologies": None, "Bubble · N2": 2, "Triangle · N3": 3},
        value="All topologies", allow_select_none=False, label="Topology",
    )
    batch_choice = mo.ui.dropdown(
        {"All batch sizes": None, "Singleton latency": 1, "Batch 32 throughput": 32},
        value="All batch sizes", allow_select_none=False, label="Batch size",
    )
    mo.hstack([topology_choice, batch_choice], justify="start", gap=2)
    return batch_choice, topology_choice


@app.cell
def _(batch_choice, benchmark_rows, topology_choice):
    selected_rows = tuple(row for row in benchmark_rows
        if (topology_choice.value is None or row.propagators == topology_choice.value)
        and (batch_choice.value is None or row.batch == batch_choice.value))
    return (selected_rows,)


@app.cell(hide_code=True)
def _(benchmark_grid, mo, ratio_chart, selected_rows):
    mo.vstack([
        mo.Html(benchmark_grid(selected_rows)),
        mo.Html(ratio_chart(selected_rows)),
        mo.md("Times are **µs per point**; batch 32 is amortized throughput. Ratios are medians of paired block ratios, so they need not equal a quotient of the separately displayed medians. Preparation and subprocess transport are excluded."),
    ])
    return


@app.cell(hide_code=True)
def _(benchmark_data, mo):
    mo.accordion({
        "What is timed?": mo.md(benchmark_data["kernel_scope"]),
        "Cache policy and timing protocol": mo.md(benchmark_data["cache_policy"]),
        "Numerical validation": mo.md("Every requested Cartesian tensor component and every separated UV/IR/finite coefficient is compared before timing. The mixed dimensionally normalized tolerance is `1e-10`. An unstable or unsupported lane cannot supply successful zero coefficients."),
        "Analysis and qualification": mo.md(benchmark_data["analysis"]),
        "Reproducible source": mo.md(f"Committed evidence: `{benchmark_data['source']}`. The notebook verifies its SHA-256 before displaying the snapshot. Full environment, samples and uncertainty protocol remain in that file. The private cohort additionally verifies its original numerical qualification and separate posthoc qualification."),
    })
    return


@app.cell(hide_code=True)
def _(mo, reference_data, reference_rows):
    mo.vstack([
        mo.md("""
        ## The original reference codes

        Original IREGI and COLLIER were also measured together on the shared
        undotted bubble/triangle domain. This separate cohort helps establish
        the optimization target. Its absolute times are shown below.
        """),
        mo.ui.table(reference_rows, selection=None, show_column_summaries=False),
        mo.callout(
            mo.md("The direct **TIRust / COLLIER** choices above use their own same-session, four-engine cohort. These archival reference times remain separate; dividing values from different sessions would not be a paired comparison. Short-block families remain inconclusive, and dotted propagators are outside the COLLIER comparisons."),
            kind="info",
        ),
        mo.md(f"**Reference timing scope:** {reference_data['timing_boundary']}."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Follow the qualification work

    The [benchmark report](https://github.com/huashengshao/iregi-rust/blob/main/docs/benchmark.pdf)
    records physical graph checks, precision and Dual coverage, failed cases,
    batch scaling and private optimization experiments. A faster private kernel
    becomes a public performance result only after its complete reduction path
    passes the same numerical and timing gates.
    """)
    return


if __name__ == "__main__":
    app.run()
