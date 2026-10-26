"""Run a typechecker against this file in a fresh installed-wheel environment."""
from typing import assert_type

from tirust import (
    BatchResult,
    ComplexFloat,
    ContractionResult,
    Dual,
    E,
    Expression,
    FeynmanDiagram,
    Float,
    Model,
    PreparedGraph,
    PreparedNumerator,
    ReductionPlan,
    TensorExpression,
    Workspace,
)

model = Model.phi3()
assert_type(model, Model)
expression = E("a*q0^2+q1")
assert_type(expression, Expression)
tensor = TensorExpression(expression)
assert_type(tensor.to_expression(), Expression)
variables = [E(f"q{axis}") for axis in range(4)]
numerator = PreparedNumerator.from_tensor(tensor, variables)
assert_type(numerator, PreparedNumerator)
plan = ReductionPlan(2, 2, powers=[1, 1])
checked_plan = ReductionPlan(6, 1, invariants="scalar_products", check_scalar_products=True)
assert_type(checked_plan, ReductionPlan)
workspace = plan.workspace(dtype="float", precision=128, accuracy_bits=60)
assert_type(workspace, Workspace)
batch = plan.pack(
    shifts=[[[0, 0, 0, 0], [0, "1/2", 0, 0]]],
    masses_squared=[["4", "2"]],
    mu=["1"],
)
result = workspace.reduce_batch(batch)
assert_type(result, BatchResult)
assert_type(result.stats, dict[str, int])
assert_type(result.success, tuple[bool, ...])
assert_type(result.diagnostics, tuple[tuple[str, ...], ...])
detail = result.engine_execution
if detail["engine"] == "certified_tensor":
    assert_type(detail["certificate_execution"]["cartesian"], dict[str, int])
elif detail["engine"] == "six_point_hybrid":
    assert_type(detail["fallback_rows"], int)
elif detail["engine"] == "affine_power":
    assert_type(detail["child_methods"][0]["powers"], list[int])
coefficients = numerator.evaluate_batch([E("a")], [[Float("2", precision=128)]], dtype="float", precision=128)
contracted = workspace.contract(result, monomials=numerator.monomials, coefficients=coefficients)
assert_type(contracted, ContractionResult)
jet = Dual(shape=[[0], [1]], values=[Float("4", precision=128), 1])
assert_type(jet, Dual)
if result.success[0]:
    value = result.lane(0)[0][3]
    if isinstance(value, ComplexFloat):
        assert_type(abs(value), Float)
