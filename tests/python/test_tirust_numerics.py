"""Public-wheel numerical checks; all reduction is performed by Rust batches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tirust import (
    ComplexFloat,
    Dual,
    E,
    Float,
    LaneError,
    PreparedNumerator,
    ReductionError,
    ReductionPlan,
)


def bubble_batch(plan, masses=("4", "9", "16"), mu=None):
    return plan.pack(
        shifts=[[[0, 0, 0, 0], [0, "1/2", 0, 0]] for _ in masses],
        masses_squared=[[mass, "2"] for mass in masses],
        mu=["1"] * len(masses) if mu is None else mu,
    )


@pytest.mark.parametrize("dtype,bits,accuracy", [("f64", 53, 40), ("double-double", 106, 70), ("float", 128, 70)])
def test_batch_metadata_lane_failure_and_empty(dtype, bits, accuracy):
    plan = ReductionPlan(2, 2)
    workspace = plan.workspace(dtype=dtype, precision=bits, accuracy_bits=accuracy, profiling=True)
    result = workspace.reduce_batch(bubble_batch(plan, mu=["1", "0", "1"]))
    assert result.rows == 3
    assert result.success == (True, False, True)
    assert result.statuses[1] == "invalid_input"
    assert result.lanes()[1] is None
    with pytest.raises(LaneError, match="lane 1"):
        result.lane(1)
    with pytest.raises(IndexError):
        result.lane(3)
    assert result.diagnostics[1]
    assert result.basis == plan.basis
    assert result.laurent_names == ("uv_single", "ir_single", "ir_double", "finite")
    assert result.stats["input_rows"] == 3
    assert result.stats["argument_rows"] > 0
    assert result.stats["master_family_batches"] > 0
    assert result.stats["master_rows"] >= 2
    assert result.profile["total_seconds"] >= 0
    assert result.dtype == dtype and result.precision == bits
    if dtype != "f64":
        assert isinstance(result.lane(0)[0][3], ComplexFloat)
    empty = workspace.reduce_batch(plan.pack(shifts=[], masses_squared=[], mu=[]))
    assert empty.rows == 0 and empty.lanes() == () and empty.success == ()
    assert empty.stats["input_rows"] == 0


def test_reordered_chunked_batches_and_independent_workspaces():
    plan = ReductionPlan(2, 1)
    workspace = plan.workspace(accuracy_bits=40)
    masses = ["4", "5", "6", "7", "8"]
    result = workspace.reduce_batch(bubble_batch(plan, masses)).lanes()
    reversed_result = workspace.reduce_batch(bubble_batch(plan, masses[::-1])).lanes()
    chunked = tuple(
        lane
        for chunk in [masses[:2], masses[2:]]
        for lane in workspace.reduce_batch(bubble_batch(plan, chunk)).lanes()
    )
    assert reversed_result[::-1] == result
    assert chunked == result

    def independent(_):
        return plan.workspace(accuracy_bits=40).reduce_batch(bubble_batch(plan, masses)).lanes()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(independent, range(2))) == [result, result]


def test_dotted_propagator_matches_dual_mass_derivative_all_laurent_slots():
    shape = [[0], [1]]
    plan = ReductionPlan(2, 0)
    dual_workspace = plan.workspace(dtype="float", precision=256, accuracy_bits=100, dual_shape=shape)
    dual_batch = bubble_batch(plan, [Dual(shape=shape, values=["4", "1"])])
    derivative = dual_workspace.reduce_batch(dual_batch)
    dotted_plan = ReductionPlan(2, 0, powers=[2, 1])
    dotted = dotted_plan.workspace(dtype="float", precision=256, accuracy_bits=100).reduce_batch(
        bubble_batch(dotted_plan, ["4"])
    )
    assert derivative.success == dotted.success == (True,)
    for jet, expected in zip(derivative.lane(0)[0], dotted.lane(0)[0], strict=True):
        assert isinstance(jet, Dual)
        assert jet.shape == shape
        assert abs(jet.values[1] - expected) < Float("1e-28", precision=256)


def test_float_inputs_beyond_binary64_remain_distinct():
    plan = ReductionPlan(2, 0)
    workspace = plan.workspace(dtype="float", precision=256, accuracy_bits=120)
    first = Float("4", precision=256)
    second = first + Float.from_ratio(1, 2**80, precision=256)
    assert first != second
    result = workspace.reduce_batch(bubble_batch(plan, [first, second]))
    assert result.success == (True, True)
    assert result.lane(0)[0][3] != result.lane(1)[0][3]


def test_double_float_transport_keeps_representable_wide_gap_tail():
    variables = [E(f"q{i}") for i in range(4)]
    prepared = PreparedNumerator.from_expression(E("a*q0"), variables)
    wide = Float(1, precision=256) + Float.from_ratio(1, 2**200, precision=256)
    assert wide != Float(1, precision=256)
    coefficient = prepared.evaluate_batch([E("a")], [[wide]], dtype="double-double", precision=106)[0][0]
    assert isinstance(coefficient, ComplexFloat)
    assert coefficient.real == wide
    assert coefficient.imag == 0
    # Export storage may need more bits than the arithmetic's nominal 106.
    assert coefficient.real.precision >= 201
    repeated = prepared.evaluate_batch([E("a")], [[coefficient]], dtype="double-double", precision=106)[0][0]
    assert repeated.real.as_integer_ratio() == coefficient.real.as_integer_ratio()


@pytest.mark.parametrize("dtype,bits", [("f64", 53), ("double-double", 106), ("float", 128)])
def test_coefficient_batch_and_core_contraction(dtype, bits):
    variables = [E(f"iregi::q{i}") for i in range(4)]
    prepared = PreparedNumerator.from_expression(E("1+2*iregi::q0+3*iregi::q1^2"), variables)
    assert prepared.rank == 2
    assert all(isinstance(index, tuple) and len(index) == 4 for index in prepared.monomials)
    coefficients = prepared.evaluate_batch([], [[], [], []], dtype=dtype, precision=bits)
    assert len(coefficients) == 3 and all(len(row) == 3 for row in coefficients)
    plan = ReductionPlan(2, prepared.rank)
    workspace = plan.workspace(dtype=dtype, precision=bits, accuracy_bits=40)
    moments = workspace.reduce_batch(bubble_batch(plan))
    result = workspace.contract(moments, monomials=prepared.monomials, coefficients=coefficients)
    assert result.success == (True, True, True)
    assert result.laurent_names == moments.laurent_names
    basis = {index: position for position, index in enumerate(moments.basis)}
    for lane in range(3):
        for pole in range(4):
            expected = (
                moments.lane(lane)[basis[(0, 0, 0, 0)]][pole]
                + 2 * moments.lane(lane)[basis[(1, 0, 0, 0)]][pole]
                + 3 * moments.lane(lane)[basis[(0, 2, 0, 0)]][pole]
            )
            assert abs(result.lane(lane)[pole] - expected) < (1e-10 if dtype == "f64" else Float("1e-10", precision=bits))


def test_zero_numerator_preserves_nonzero_batch_length():
    prepared = PreparedNumerator.from_expression(E("0"), [E(f"q{i}") for i in range(4)])
    coefficients = prepared.evaluate_batch([], [[], [], []])
    assert coefficients == ((), (), ())
    plan = ReductionPlan(2, 0)
    workspace = plan.workspace(accuracy_bits=40)
    moments = workspace.reduce_batch(bubble_batch(plan))
    contracted = workspace.contract(moments, monomials=[], coefficients=coefficients)
    assert contracted.lanes() == ((0j, 0j, 0j, 0j),) * 3


def test_parameter_coefficient_duals_keep_mixed_components():
    variables = [E(f"q{i}") for i in range(4)]
    prepared = PreparedNumerator.from_expression(E("a*b*q0 + a^2*q1"), variables)
    shape = [[0, 0], [1, 0], [0, 1], [1, 1]]
    values = prepared.evaluate_batch(
        [E("a"), E("b")],
        [[Dual(shape, [2, 1, 0, 0]), Dual(shape, [3, 0, 1, 0])]],
        dtype="float", precision=128, dual_shape=shape,
    )
    by_monomial = dict(zip(prepared.monomials, values[0], strict=True))
    assert by_monomial[(1, 0, 0, 0)].values == (6, 3, 2, 1)
    assert by_monomial[(0, 1, 0, 0)].values == (4, 4, 0, 0)


@pytest.mark.parametrize("dtype,bits", [("f64", 53), ("double-double", 106), ("float", 128)])
def test_nonfinite_coefficient_isolated_to_its_contraction_lane(dtype, bits):
    variables = [E(f"q{i}") for i in range(4)]
    numerator = PreparedNumerator.from_expression(E("a"), variables)
    coefficients = numerator.evaluate_batch(
        [E("a")], [["1"], ["NaN"], ["2"]], dtype=dtype, precision=bits
    )
    assert len(coefficients) == 3
    plan = ReductionPlan(2, 0)
    workspace = plan.workspace(dtype=dtype, precision=bits, accuracy_bits=40)
    moments = workspace.reduce_batch(bubble_batch(plan))
    contracted = workspace.contract(moments, monomials=numerator.monomials, coefficients=coefficients)
    assert contracted.success == (True, False, True)
    assert contracted.lanes()[1] is None
    assert contracted.diagnostics[1]
    with pytest.raises(LaneError):
        contracted.lane(1)


def test_invalid_shapes_and_polynomial_inputs_are_explicit_errors():
    for shape in [[], [[1]], [[0], [2]], [[0], [0]], [[0], [1, 0]]]:
        with pytest.raises(ValueError):
            Dual(shape, [0] * len(shape))
    variables = [E(f"q{i}") for i in range(4)]
    for invalid in [E("1"), E("q0+q1"), E("q0^2"), E("q1")]:
        with pytest.raises(ValueError):
            PreparedNumerator.from_expression(E("q0+1"), [invalid, *variables[1:]])
    for exponent in ["65536", "2147483648", "4294967295", "4294967296", "18446744073709551616"]:
        with pytest.raises(ReductionError):
            PreparedNumerator.from_expression(E(f"q0^{exponent}"), variables)
    for expression in ["1/q0", "sqrt(q0)", "sin(q0)", "q0^a"]:
        with pytest.raises(ReductionError):
            PreparedNumerator.from_expression(E(expression), variables)


def test_input_layout_and_kinematic_shape_errors():
    plan = ReductionPlan(2, 0)
    assert plan.input_layout == {
        "shifts": (0, 8), "scalar_products": None, "masses_squared": (8, 10),
        "mu": 10, "metric": "+---", "scale_is_squared": False,
    }
    with pytest.raises(ValueError, match="batch length"):
        plan.pack(shifts=[], masses_squared=[], mu=[1])
    with pytest.raises(ValueError, match="four-component"):
        plan.pack(shifts=[[[0, 0], [0, 0]]], masses_squared=[[1, 2]], mu=[1])
    other = ReductionPlan(2, 0)
    with pytest.raises(ValueError, match="different ReductionPlan"):
        plan.workspace().reduce_batch(bubble_batch(other))
