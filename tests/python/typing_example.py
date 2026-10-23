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
coefficients = numerator.evaluate_batch([E("a")], [[Float("2", precision=128)]], dtype="float", precision=128)
contracted = workspace.contract(result, monomials=numerator.monomials, coefficients=coefficients)
assert_type(contracted, ContractionResult)
jet = Dual(shape=[[0], [1]], values=[Float("4", precision=128), 1])
assert_type(jet, Dual)
if result.success[0]:
    value = result.lane(0)[0][3]
    if isinstance(value, ComplexFloat):
        assert_type(abs(value), Float)
