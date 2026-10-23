# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="TIRust · a numerator becomes an integral")


@app.cell
def _():
    import marimo as mo
    from tirust import FeynmanDiagram, Model, PreparedGraph, ReductionPlan
    from _presentation import cards, hero, laurent_cards, project_root, theme
    return (
        FeynmanDiagram, Model, PreparedGraph, ReductionPlan,
        cards, hero, laurent_cards, mo, project_root, theme,
    )


@app.cell(hide_code=True)
def _(hero, mo, theme):
    mo.vstack([
        mo.Html(theme()),
        mo.Html(hero(
            "A numerator becomes an integral",
            "Start with a one-loop graph, keep its numerator symbolic, and reduce a batch of numerical points. The graph preparation and the integral reduction remain reusable steps.",
            eyebrow="TIRust gallery · 01 / first reduction",
        )),
        mo.md(r"""
        \[
        \mathcal I[N]=\mu^{2\epsilon}\int d^{4-2\epsilon}q\,
        \frac{N(q_0,q_1,q_2,q_3)}{\prod_i [(q+p_i)^2-m_i^2+i0]}.
        \]
        We use the Minkowski metric $(+,-,-,-)$ and four-dimensional numerator
        polynomials. Results retain the original IREGI normalization and the
        four separate **UV single, IR single, IR double, finite** coefficients.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    topology = mo.ui.dropdown(
        {"Bubble": "bubble", "Triangle": "triangle"},
        value="Bubble", allow_select_none=False, label="Graph",
    )
    sample_count = mo.ui.dropdown(
        {"One point": 1, "Three points": 3, "Thirty-two points": 32},
        value="Three points", allow_select_none=False, label="Batch",
    )
    numeric_type = mo.ui.dropdown(
        {"Double precision": "f64", "Double-double": "double-double", "128-bit Float": "float"},
        value="Double precision", allow_select_none=False, label="Arithmetic",
    )
    mo.hstack([topology, sample_count, numeric_type], justify="start", gap=2)
    return numeric_type, sample_count, topology


@app.cell
def _(project_root, topology):
    checkout = project_root(__file__)
    dot_source = (checkout / "examples" / "graphs" / f"{topology.value}.dot").read_text()
    return checkout, dot_source


@app.cell(hide_code=True)
def _(dot_source, mo):
    dot_editor = mo.ui.text_area(
        value=dot_source, rows=10, full_width=True, label="FeynKit DOT graph",
    ).form(submit_button_label="Prepare this graph")
    mo.accordion({"Inspect or edit the graph and its polynomial numerator": dot_editor})
    return (dot_editor,)


@app.cell
def _(FeynmanDiagram, Model, PreparedGraph, dot_editor, dot_source, mo,
      propagator_count):
    model = Model.phi3()
    diagram = FeynmanDiagram.from_dot(
        model, dot_source if dot_editor.value is None else dot_editor.value,
    )
    graph = PreparedGraph.from_diagram(diagram, source="supplied")
    mo.stop(
        graph.propagator_count != propagator_count,
        mo.callout(mo.md("This edited graph has a different number of propagators from the selected numerical example. Choose the matching bubble/triangle example or supply matching numerical inputs before reducing it."), kind="warn"),
    )
    numerator = graph.numerator
    return diagram, graph, model, numerator


@app.cell(hide_code=True)
def _(diagram, mo):
    mo.vstack([mo.md("## The graph"), diagram])
    return


@app.cell(hide_code=True)
def _(mo, numerator):
    mo.md("""
    ## Keep the numerator symbolic

    FeynKit imports the graph. Symbolica extracts the coefficients of the loop
    polynomial without introducing tensor multiplicities. This prepared
    numerator can be reused for every numerical batch below.
    """)
    return


@app.cell
def _(numerator):
    numerator.expression
    return


@app.cell(hide_code=True)
def _(cards, mo, numerator, sample_count):
    mo.Html(cards([
        ("Polynomial rank", numerator.rank),
        ("Nonzero monomials", len(numerator.terms)),
        ("Numerical points", sample_count.value),
    ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reduce the batch

    Use positive masses and spacelike shifts, away from physical thresholds.
    Every point has a slightly different shift. The denominator ordering is
    $(q+p_0)^2-m_0^2,(q+p_1)^2-m_1^2,\ldots$; the scale input is $\mu$, not
    $\mu^2$. Numerical masses are supplied here, independently of the model's
    symbolic mass label.
    """)
    return


@app.cell
def _(sample_count, topology):
    propagator_count = {"bubble": 2, "triangle": 3}[topology.value]
    shifts = []
    masses_squared = []
    for _lane in range(sample_count.value):
        _shifts = [["0", "0", "0", "0"],
                   ["0", f"{16 + _lane}/16", "0", "0"]]
        if propagator_count == 3:
            _shifts.append(["0", "1/4", "5/4", "0"])
        shifts.append(_shifts)
        masses_squared.append(["4", "9", "16"][:propagator_count])
    scales = ["3"] * sample_count.value
    return masses_squared, propagator_count, scales, shifts


@app.cell
def _(ReductionPlan, numeric_type, numerator, propagator_count):
    plan = ReductionPlan(
        propagators=propagator_count,
        rank=numerator.rank,
        algorithm="hybrid",
        convention="iregi",
    )
    precision_bits = {"f64": 53, "double-double": 106, "float": 128}[numeric_type.value]
    accuracy_bits = {"f64": 40, "double-double": 80, "float": 80}[numeric_type.value]
    workspace = plan.workspace(
        dtype=numeric_type.value, precision=precision_bits, accuracy_bits=accuracy_bits,
    )
    return accuracy_bits, plan, precision_bits, workspace


@app.cell
def _(masses_squared, plan, scales, shifts, workspace):
    packed = plan.pack(shifts=shifts, masses_squared=masses_squared, mu=scales)
    tensors = workspace.reduce_batch(packed)
    return packed, tensors


@app.cell
def _(numeric_type, numerator, precision_bits, sample_count, tensors, workspace):
    coefficient_rows = numerator.evaluate_batch(
        parameters=[], rows=[[] for _ in range(sample_count.value)],
        dtype=numeric_type.value, precision=precision_bits,
    )
    integral = workspace.contract(
        tensors, monomials=numerator.monomials, coefficients=coefficient_rows,
    )
    return coefficient_rows, integral


@app.cell(hide_code=True)
def _(integral, mo, tensors):
    lane_summary = [{
        "Point": _lane + 1,
        "Tensor status": tensors.statuses[_lane],
        "Contraction status": integral.statuses[_lane],
        "Fallback": tensors.fallback_used[_lane],
        "Diagnostic": "; ".join(tensors.diagnostics[_lane]),
    } for _lane in range(len(tensors.statuses))]
    mo.ui.table(lane_summary, selection=None, show_column_summaries=False)
    return (lane_summary,)


@app.cell(hide_code=True)
def _(mo, sample_count):
    point = mo.ui.dropdown(
        {f"Point {_index + 1}": _index for _index in range(sample_count.value)},
        value="Point 1", allow_select_none=False, label="Inspect a point",
    )
    point
    return (point,)


@app.cell(hide_code=True)
def _(integral, laurent_cards, mo, point):
    mo.stop(
        not integral.success[point.value],
        mo.callout(mo.md("This lane did not succeed. Inspect its diagnostic above; no coefficient values are exposed."), kind="warn"),
    )
    mo.vstack([
        mo.md("## The contracted integral"),
        mo.Html(laurent_cards(integral.lane(point.value), integral.laurent_names)),
    ])
    return


@app.cell(hide_code=True)
def _(mo, point, tensors):
    mo.stop(not tensors.success[point.value])
    _values = tensors.lane(point.value)
    _rows = [{
        "Monomial exponents (q₀,q₁,q₂,q₃)": str(_basis),
        **{_name: str(_coefficient) for _name, _coefficient in
           zip(tensors.laurent_names, _coefficients, strict=True)},
    } for _basis, _coefficients in zip(tensors.basis, _values, strict=True)]
    mo.accordion({"Inspect every tensor moment": mo.ui.table(
        _rows, selection=None, show_column_summaries=False,
    )})
    return


@app.cell(hide_code=True)
def _(accuracy_bits, mo, precision_bits, tensors):
    mo.accordion({
        "Accuracy and batch diagnostics": mo.vstack([
            mo.md(f"Working precision: **{precision_bits} bits**. Requested accuracy: **{accuracy_bits} bits**. These are distinct settings; additional working precision provides room for cancellation."),
            mo.ui.table([{"Counter": _name, "Value": str(_value)}
                         for _name, _value in tensors.stats.items()],
                        selection=None, show_column_summaries=False),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Next:** open `02_dots_and_derivatives.py` to raise a propagator power and
    verify the result using a mass derivative. The
    [benchmark report](https://github.com/huashengshao/iregi-rust/blob/main/docs/benchmark.pdf)
    documents independently validated domains and open cases.
    """)
    return


if __name__ == "__main__":
    app.run()
