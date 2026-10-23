# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="TIRust · dots, precision and derivatives")


@app.cell
def _():
    import marimo as mo
    from tirust import Dual, Float, ReductionPlan
    from _presentation import cards, hero, laurent_cards, theme
    return Dual, Float, ReductionPlan, cards, hero, laurent_cards, mo, theme


@app.cell(hide_code=True)
def _(hero, mo, theme):
    mo.vstack([
        mo.Html(theme()),
        mo.Html(hero(
            "Dots, precision and derivatives",
            "A raised propagator power is a first-class reduction input. Verify a dotted bubble through an analytic mass derivative carried by Dual numbers, and compare an independent finite difference.",
            eyebrow="TIRust gallery · 02 / generalized integrals",
        )),
        mo.md(r"""
        With $D_i=(q+p_i)^2-m_i^2+i0$,
        \[
        \frac{\partial}{\partial m_j^2}
        \int\!\frac{d^Dq\,N(q)}{\prod_i D_i^{\nu_i}}
        =\nu_j\int\!\frac{d^Dq\,N(q)}
        {D_j^{\nu_j+1}\prod_{i\ne j}D_i^{\nu_i}}.
        \]
        The sign is positive: differentiating $D_j^{-\nu_j}$ brings two minus
        signs. Here $N=1$ and $\nu=(1,1)$, so the mass derivative equals the
        dotted integral with powers $(2,1)$ or $(1,2)$.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    dotted_line = mo.ui.dropdown(
        {"First propagator · m₀² = 4": 0, "Second propagator · m₁² = 9": 1},
        value="First propagator · m₀² = 4", allow_select_none=False,
        label="Dotted propagator",
    )
    arithmetic = mo.ui.dropdown(
        {"Double precision": "f64", "Double-double": "double-double",
         "128-bit Float": "float128", "256-bit Float": "float256"},
        value="128-bit Float", allow_select_none=False, label="Arithmetic",
    )
    difference_step = mo.ui.dropdown(
        {"h = 2⁻¹⁰": 1024, "h = 2⁻¹⁶": 65536, "h = 2⁻²²": 4194304},
        value="h = 2⁻¹⁶", allow_select_none=False, label="Finite-difference step",
    )
    mo.hstack([dotted_line, arithmetic, difference_step], justify="start", gap=2)
    return arithmetic, difference_step, dotted_line


@app.cell
def _(arithmetic):
    dtype = "float" if arithmetic.value.startswith("float") else arithmetic.value
    precision_bits = {"f64": 53, "double-double": 106,
                      "float128": 128, "float256": 256}[arithmetic.value]
    accuracy_bits = {"f64": 40, "double-double": 80,
                     "float128": 80, "float256": 128}[arithmetic.value]
    return accuracy_bits, dtype, precision_bits


@app.cell
def _(ReductionPlan, accuracy_bits, dotted_line, dtype, precision_bits):
    powers = [1, 1]
    powers[dotted_line.value] = 2
    dotted_plan = ReductionPlan(propagators=2, rank=0, powers=powers)
    dotted_workspace = dotted_plan.workspace(
        dtype=dtype, precision=precision_bits, accuracy_bits=accuracy_bits,
    )
    ordinary_plan = ReductionPlan(propagators=2, rank=0)
    ordinary_workspace = ordinary_plan.workspace(
        dtype=dtype, precision=precision_bits, accuracy_bits=accuracy_bits,
    )
    return dotted_plan, dotted_workspace, ordinary_plan, ordinary_workspace, powers


@app.cell
def _(dotted_plan, dotted_workspace):
    reference_shifts = [["0", "0", "0", "0"], ["0", "1", "0", "0"]]
    central_masses = ["4", "9"]
    dotted_result = dotted_workspace.reduce_batch(dotted_plan.pack(
        shifts=[reference_shifts], masses_squared=[central_masses], mu=["3"],
    ))
    return central_masses, dotted_result, reference_shifts


@app.cell(hide_code=True)
def _(dotted_result, laurent_cards, mo):
    mo.stop(not dotted_result.success[0], mo.callout(
        mo.md("The dotted evaluation did not succeed: " + "; ".join(dotted_result.diagnostics[0])),
        kind="warn",
    ))
    mo.vstack([
        mo.md("## Reduce the dotted integral directly"),
        mo.Html(laurent_cards(dotted_result.lane(0)[0], dotted_result.laurent_names)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Follow the mass through the calculation

    Seed $m_j^2\mapsto m_j^2+\delta$ with $\delta^2=0$.
    The coefficient of $\delta$ is the derivative on the selected analytic
    branch. Taylor components belong to each number; they are distinct from
    the independently evaluated sample lanes.
    """)
    return


@app.cell
def _(Dual, accuracy_bits, central_masses, dotted_line, dtype, ordinary_plan,
      precision_bits, reference_shifts):
    derivative_shape = [[0], [1]]
    dual_workspace = ordinary_plan.workspace(
        dtype=dtype, precision=precision_bits, accuracy_bits=accuracy_bits,
        dual_shape=derivative_shape,
    )
    seeded_masses = list(central_masses)
    seeded_masses[dotted_line.value] = Dual(
        shape=derivative_shape, values=[central_masses[dotted_line.value], "1"],
    )
    derivative_result = dual_workspace.reduce_batch(ordinary_plan.pack(
        shifts=[reference_shifts], masses_squared=[seeded_masses], mu=["3"],
    ))
    return derivative_result, derivative_shape, dual_workspace, seeded_masses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An independent finite difference

    Evaluate $m_j^2+h$, $m_j^2-h$ and the central mass in **one batch**:
    \[
    I'_{\mathrm{FD}}=\frac{I(m_j^2+h)-I(m_j^2-h)}{2h}.
    \]
    Its truncation error scales as $h^2$; a very small step eventually amplifies
    rounding. Adjust the step and precision to see this tradeoff. This display
    is an interactive check, separate from the fixed acceptance tolerances in
    the independent validation suite.
    """)
    return


@app.cell
def _(Float, central_masses, difference_step, dotted_line, ordinary_plan,
      ordinary_workspace, precision_bits, reference_shifts):
    finite_difference_masses = []
    _step = Float(1, precision=precision_bits) / difference_step.value
    for _offset in (1, -1, 0):
        _masses = list(central_masses)
        _masses[dotted_line.value] = (
            Float(central_masses[dotted_line.value], precision=precision_bits)
            + _offset * _step
        )
        finite_difference_masses.append(_masses)
    difference_result = ordinary_workspace.reduce_batch(ordinary_plan.pack(
        shifts=[reference_shifts] * 3,
        masses_squared=finite_difference_masses,
        mu=["3"] * 3,
    ))
    return difference_result, finite_difference_masses


@app.cell(hide_code=True)
def _(derivative_result, difference_result, difference_step, dotted_result, mo):
    mo.stop(
        not all(dotted_result.success + derivative_result.success + difference_result.success),
        mo.callout(mo.md("At least one lane did not succeed. Inspect its diagnostic; no failed-lane values are exposed."), kind="warn"),
    )
    _direct = dotted_result.lane(0)[0]
    _dual = derivative_result.lane(0)[0]
    _plus, _minus = difference_result.lane(0)[0], difference_result.lane(1)[0]
    derivative_comparison = []
    for _slot, _name in enumerate(dotted_result.laurent_names):
        _derivative = _dual[_slot].values[1]
        _difference = (_plus[_slot] - _minus[_slot]) * (difference_step.value // 2)
        _normalization = 1 + abs(_direct[_slot])
        derivative_comparison.append({
            "Coefficient": _name,
            "Dotted integral": str(_direct[_slot]),
            "Dual derivative": str(_derivative),
            "Finite difference": str(_difference),
            "Dual discrepancy / (1+|I|)": str(abs(_derivative - _direct[_slot]) / _normalization),
            "FD discrepancy / (1+|I|)": str(abs(_difference - _direct[_slot]) / _normalization),
        })
    mo.ui.table(derivative_comparison, selection=None, show_column_summaries=False)
    return (derivative_comparison,)


@app.cell(hide_code=True)
def _(Float, cards, mo):
    exact_mass_text = "4.00000000000000000000000082718061255302767487140869206996285356581211090087890625"
    precise_mass = Float(exact_mass_text, precision=128)
    mo.vstack([
        mo.md("""
        ## Keep the digits you supplied

        Supply long decimal strings or the shared Symbolica numeric types when
        the distinction lies beyond binary64. The two masses below differ by
        $2^{-80}$; no Python `float` is used to construct the 128-bit input.
        """),
        mo.Html(cards([
            ("128-bit mass² input", precise_mass),
            ("Binary64 representation", float(exact_mass_text)),
        ])),
    ])
    return exact_mass_text, precise_mass


@app.cell
def _(ReductionPlan, exact_mass_text, reference_shifts):
    precision_plan = ReductionPlan(propagators=2, rank=0)
    precision_workspace = precision_plan.workspace(
        dtype="float", precision=256, accuracy_bits=128,
    )
    precision_result = precision_workspace.reduce_batch(precision_plan.pack(
        shifts=[reference_shifts, reference_shifts],
        masses_squared=[["4", "9"], [exact_mass_text, "9"]], mu=["3", "3"],
    ))
    return precision_plan, precision_result, precision_workspace


@app.cell(hide_code=True)
def _(cards, mo, precision_result):
    mo.stop(not all(precision_result.success))
    _difference = precision_result.lane(1)[0][3] - precision_result.lane(0)[0][3]
    mo.Html(cards([("Change in the finite coefficient at 256 bits", _difference)]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## An invalid neighbor has its own diagnostic

    Include a point with zero renormalization scale beside two valid points.
    The failed lane is masked; the successful lanes remain independently usable.
    """)
    return


@app.cell
def _(central_masses, ordinary_plan, ordinary_workspace, reference_shifts):
    isolated_result = ordinary_workspace.reduce_batch(ordinary_plan.pack(
        shifts=[reference_shifts] * 3,
        masses_squared=[central_masses] * 3,
        mu=["3", "0", "3"],
    ))
    return (isolated_result,)


@app.cell(hide_code=True)
def _(cards, isolated_result, mo):
    _lanes = isolated_result.lanes()
    isolation_verified = (
        isolated_result.success == (True, False, True)
        and _lanes[1] is None and _lanes[0] == _lanes[2]
    )
    mo.vstack([mo.ui.table([{
        "Point": _index + 1,
        "Status": isolated_result.statuses[_index],
        "Values available": _lanes[_index] is not None,
        "Diagnostic": "; ".join(isolated_result.diagnostics[_index]),
    } for _index in range(3)], selection=None, show_column_summaries=False),
        mo.Html(cards([("Valid neighbors agree; invalid lane is masked", isolation_verified)])),
    ])
    return (isolation_verified,)


@app.cell(hide_code=True)
def _(accuracy_bits, derivative_result, difference_result, dtype, mo, precision_bits):
    mo.accordion({
        "Numerical context": mo.md(f"Arithmetic: **{dtype}**; working precision: **{precision_bits} bits**; requested accuracy: **{accuracy_bits} bits**. Finite-difference samples: **{len(difference_result.statuses)}**; Dual samples: **{len(derivative_result.statuses)}**. The selected points are massive and spacelike, away from threshold or branch boundaries."),
        "Supported domain": mo.md("Positive powers and dimension shifts are represented by the Rust reduction API. Dotted tadpoles, bubbles and triangles have independent reference and recurrence checks. This notebook demonstrates one massive bubble identity; it does not qualify every dotted high-point topology or exceptional kinematic point. COLLIER is compared on its shared undotted domain only."),
    })
    return


if __name__ == "__main__":
    app.run()
