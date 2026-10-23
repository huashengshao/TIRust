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
        KERNEL_REVISION,
        KERNEL_SOURCE,
        benchmark_grid,
        cards,
        hero,
        load_kernel_snapshot,
        load_reference_times,
        project_root,
        ratio_chart,
        theme,
    )
    return (
        KERNEL_REVISION, KERNEL_SOURCE, benchmark_grid, cards, hero,
        load_kernel_snapshot, load_reference_times, mo, project_root,
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
def _(load_kernel_snapshot, load_reference_times, project_root):
    checkout = project_root(__file__)
    kernel_data, kernel_rows = load_kernel_snapshot(checkout)
    reference_data, reference_rows = load_reference_times(checkout)
    return checkout, kernel_data, kernel_rows, reference_data, reference_rows


@app.cell(hide_code=True)
def _(KERNEL_REVISION, cards, kernel_data, kernel_rows, mo):
    mo.vstack([
        mo.Html(cards([
            ("Numerical comparisons", f"{sum(row.numerical_status == 'passed' for row in kernel_rows)} / {len(kernel_rows)} passed"),
            ("Timing protocol", f"{kernel_data['protocol']['repeats']} paired blocks · 99% CI"),
            ("Source snapshot", KERNEL_REVISION[:7]),
        ])),
        mo.callout(
            mo.md("**Historical snapshot.** These measurements identify the `6b24c03` tensor kernel. Later public APIs, preparation changes and private optimization candidates are not timed by this run. All eight numerical comparisons pass; seven performance regressions and one noisy result remain visible."),
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
def _(batch_choice, kernel_rows, topology_choice):
    selected_rows = tuple(row for row in kernel_rows
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
def _(KERNEL_SOURCE, kernel_data, mo):
    mo.accordion({
        "What is timed?": mo.md(kernel_data["protocol"]["kernel_scope"]),
        "Cache policy": mo.md(kernel_data["protocol"]["cache_policy"]),
        "Numerical validation": mo.md("Every requested Cartesian tensor component and every separated UV/IR/finite coefficient is compared before timing. The mixed dimensionally normalized tolerance is `1e-10`. An unstable or unsupported lane cannot supply successful zero coefficients."),
        "Reproducible source": mo.md(f"Committed evidence: `{KERNEL_SOURCE}`. The notebook verifies its SHA-256 before displaying the snapshot. Full environment, samples and uncertainty protocol remain in that file."),
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
            mo.md("A direct **TIRust / COLLIER** ratio is **pending**. Dividing times from these separate sessions would not be a paired measurement. The short-block family remains inconclusive, and dotted propagators are outside this COLLIER comparison."),
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
