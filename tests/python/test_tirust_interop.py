"""Exercise the installed native host and its actual upstream Python objects.

These tests deliberately share inputs through Rust/PyO3 objects. They do not
round-trip a graph, tensor, or expression through a serialization format.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import tirust
from tirust import (
    ComplexFloat,
    Dual,
    E,
    Expression,
    FeynmanDiagram,
    Float,
    Model,
    PreparedGraph,
    PreparedNumerator,
    ReductionError,
    ReductionPlan,
    TensorExpression,
    configure_data_directory,
    initialize,
)


BUBBLE_DOT = """
digraph interop_bubble {
    graph [num="3 + 2*iregi::q0 + iregi::q1*iregi::q2", overall_factor="1"];
    edge [particle="phi"];
    ext [style=invis];
    ext -> a;
    a -> b [lmb_id=0];
    a -> b;
    b -> ext;
}
"""


def test_packaged_owner_assets_and_actual_svg_rendering() -> None:
    data = Path(tirust.__file__).with_name("data")
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["feynkit_revision"] == "865afdc43fb4e6b950439d34b4ec2c38352873e7"
    assert len(manifest["assets"]) == 18
    for asset in manifest["assets"]:
        payload = (data / asset["name"]).read_bytes()
        assert len(payload) == asset["bytes"]
        assert hashlib.sha256(payload).hexdigest() == asset["sha256"]
    for license in manifest["licenses"]:
        assert hashlib.sha256((data / license["name"]).read_bytes()).hexdigest() == license["sha256"]
    # The actual upstream registry validates these against compiled constants.
    configure_data_directory(data, validate=True)
    diagram = FeynmanDiagram.from_dot(Model.phi3(), BUBBLE_DOT)
    rendered = diagram._repr_svg_()
    assert isinstance(rendered, str) and "<svg" in rendered


def _variables() -> list[Expression]:
    return [E(f"tirust_interop::q{index}") for index in range(4)]


def _same_expression(left: Expression, right: Expression) -> None:
    assert isinstance(left, Expression)
    assert isinstance(right, Expression)
    assert (left - right).expand() == E("0")


def test_tensor_preparation_reuses_upstream_expression_and_coefficients() -> None:
    q0, q1, q2, q3 = variables = _variables()
    parameter = E("tirust_interop::a")
    expression = (parameter + 1) * q0**2 + 3 * q1 * q2 - 5
    tensor = TensorExpression(expression)

    assert isinstance(tensor, Expression)
    assert isinstance(tensor.to_expression(), Expression)
    prepared = PreparedNumerator.from_tensor(tensor, variables)
    ordinary = PreparedNumerator.from_expression(expression, variables)

    _same_expression(tensor.to_expression(), expression)
    _same_expression(prepared.expression, ordinary.expression)
    _same_expression(prepared.expression, expression)
    assert prepared.variables == variables
    assert prepared.rank == 2
    terms = dict(prepared.terms)
    assert set(terms) == {(0, 0, 0, 0), (2, 0, 0, 0), (0, 1, 1, 0)}
    _same_expression(terms[(0, 0, 0, 0)], E("-5"))
    _same_expression(terms[(2, 0, 0, 0)], parameter + 1)
    _same_expression(terms[(0, 1, 1, 0)], E("3"))

    coefficients = prepared.evaluate_batch([parameter], [[2], [7], [-3]])
    assert isinstance(coefficients, tuple)
    assert len(coefficients) == 3
    for value, lane in zip((2, 7, -3), coefficients, strict=True):
        assert isinstance(lane, tuple)
        by_monomial = dict(zip(prepared.monomials, lane, strict=True))
        assert by_monomial == {
            (0, 0, 0, 0): complex(-5),
            (2, 0, 0, 0): complex(value + 1),
            (0, 1, 1, 0): complex(3),
        }
    # The unused Cartesian symbol remains a valid, unchanged shared Expression.
    assert q3 == variables[3]


def test_actual_tensor_component_selection_preserves_metric_sign_and_sparse_zero() -> None:
    variables = _variables()
    expression = variables[0] * E(
        "spenso::g(spenso::mink(4,tirust_interop::mu),spenso::mink(4,tirust_interop::nu))"
    )
    tensor = TensorExpression(expression)
    assert tensor.rank == 2
    with pytest.raises(ReductionError, match="select an explicit component"):
        PreparedNumerator.from_tensor(tensor, variables)
    temporal = PreparedNumerator.from_tensor(tensor, variables, component=[0, 0])
    spatial = PreparedNumerator.from_tensor(tensor, variables, component=[1, 1])
    off_diagonal = PreparedNumerator.from_tensor(tensor, variables, component=[0, 1])
    _same_expression(temporal.expression, variables[0])
    _same_expression(spatial.expression, -variables[0])
    assert off_diagonal.terms == []
    assert off_diagonal.evaluate_batch([], [[], [], []]) == ((), (), ())
    with pytest.raises(ReductionError, match="coordinates"):
        PreparedNumerator.from_tensor(tensor, variables, component=[0])
    with pytest.raises(ReductionError):
        PreparedNumerator.from_tensor(tensor, variables, component=[4, 0])
    _same_expression(tensor.to_expression(), expression)


def test_prepared_graph_retains_the_actual_diagram_and_does_not_mutate_it() -> None:
    model = Model.phi3()
    diagram = FeynmanDiagram.from_dot(model, BUBBLE_DOT)
    before = diagram.numerator_expression().to_expression()
    prepared = PreparedGraph.from_diagram(diagram)

    assert prepared.diagram is diagram
    assert isinstance(prepared.diagram, FeynmanDiagram)
    assert isinstance(prepared.diagram.numerator_expression(), TensorExpression)
    assert prepared.name == "interop_bubble"
    assert prepared.propagator_count == 2
    assert len(prepared.propagators) == 2
    assert len(prepared.external_edges) == 2
    assert all(isinstance(item["mass_squared"], Expression) for item in prepared.propagators)
    assert all(abs(item["loop_coefficient"]) == 1 for item in prepared.propagators)
    _same_expression(diagram.numerator_expression().to_expression(), before)
    _same_expression(prepared.numerator.expression, E("3 + 2*iregi::q0 + iregi::q1*iregi::q2"))

    imported = PreparedGraph.from_dot(model, BUBBLE_DOT)
    assert isinstance(imported.diagram, FeynmanDiagram)
    assert imported.propagator_count == prepared.propagator_count
    assert imported.external_edges == prepared.external_edges
    _same_expression(imported.numerator.expression, prepared.numerator.expression)


def test_physical_preparation_matches_explicit_upstream_feynman_rules() -> None:
    model = Model.phi3()
    diagram = FeynmanDiagram.from_dot(
        model, BUBBLE_DOT.replace('num="3 + 2*iregi::q0 + iregi::q1*iregi::q2"', 'num="1"')
    )
    before = diagram.numerator_expression().to_expression()
    physical = PreparedGraph.from_diagram(diagram, source="feynman_rules")
    with_rules = diagram.apply_feynman_rules()
    independently_prepared = PreparedGraph.from_diagram(with_rules, source="supplied")

    assert physical.diagram is diagram
    assert independently_prepared.diagram is with_rules
    assert isinstance(with_rules.numerator_expression(), TensorExpression)
    assert physical.propagator_count == independently_prepared.propagator_count == 2
    assert physical.numerator.rank == independently_prepared.numerator.rank == 0
    _same_expression(physical.numerator.expression, independently_prepared.numerator.expression)
    _same_expression(diagram.numerator_expression().to_expression(), before)


def test_standalone_aliases_are_owner_classes_with_owner_metadata() -> None:
    native = tirust._tirust
    assert Expression is native.Expression
    assert Float is native.Float
    assert ComplexFloat is native.ComplexFloat
    assert Model is native._hepkit.Model
    assert FeynmanDiagram is native._hepkit.FeynmanDiagram
    assert TensorExpression is native._tensor.TensorExpression
    assert issubclass(TensorExpression, Expression)
    assert Expression.__module__ == "symbolica.core"
    assert Model.__module__ == FeynmanDiagram.__module__ == "symbolica.community.hepkit"
    assert TensorExpression.__module__ == "symbolica.community.tensor"
    # Numerica explicitly sets these two classes to their registering host.
    assert Float.__module__ == ComplexFloat.__module__ == native.__name__
    assert set(tirust.__all__) == set(native.__all__)
    for name in tirust.__all__:
        assert getattr(tirust, name) is getattr(native, name)


def test_repeated_initialization_keeps_classes_and_existing_objects() -> None:
    native = tirust._tirust
    classes = (Expression, Float, ComplexFloat, Model, FeynmanDiagram, TensorExpression)
    expression = E("tirust_interop::persistent + 1")
    tensor = TensorExpression(expression)
    for _ in range(4):
        initialize()
        native.initialize()
    assert classes == (
        tirust.Expression, tirust.Float, tirust.ComplexFloat,
        tirust.Model, tirust.FeynmanDiagram, tirust.TensorExpression,
    )
    _same_expression(tensor.to_expression(), expression)
    assert isinstance(TensorExpression(expression), TensorExpression)


def test_installed_stub_surface_and_signatures_match_native_api() -> None:
    """Read packaged stubs without importing their private type-only namespace."""
    stub = Path(tirust.__file__).with_name("_tirust.pyi")
    assert stub.is_file(), "the wheel must include its reviewed public stub"
    assert Path(tirust.__file__).with_name("py.typed").is_file()
    tree = ast.parse(stub.read_text())
    checked = 0
    for definition in tree.body:
        if not isinstance(definition, ast.ClassDef):
            continue
        cls = getattr(tirust, definition.name)
        for method in definition.body:
            if not isinstance(method, ast.FunctionDef) or method.name.startswith("__"):
                continue
            assert hasattr(cls, method.name), f"missing {definition.name}.{method.name}"
            if any(isinstance(item, ast.Name) and item.id == "property" for item in method.decorator_list):
                continue
            signature = inspect.signature(getattr(cls, method.name))
            positional = [*method.args.posonlyargs, *method.args.args]
            expected_names = [item.arg for item in positional if item.arg != "cls"]
            expected_names.extend(item.arg for item in method.args.kwonlyargs)
            assert list(signature.parameters) == expected_names, (definition.name, method.name, signature)
            positional_defaults = dict(zip(
                [item.arg for item in positional][-len(method.args.defaults):] if method.args.defaults else [],
                method.args.defaults,
                strict=True,
            ))
            keyword_defaults = {
                argument.arg: value for argument, value in zip(method.args.kwonlyargs, method.args.kw_defaults, strict=True)
                if value is not None
            }
            for name, default in (positional_defaults | keyword_defaults).items():
                assert signature.parameters[name].default == ast.literal_eval(default)
            checked += 1
    assert checked >= 12, "the check must inspect the functional binding surface"
    assert inspect.signature(initialize).parameters == {}


def test_container_shapes_match_typed_public_contract() -> None:
    numerator = PreparedNumerator.from_expression(E("2"), _variables())
    assert isinstance(numerator.monomials, list)
    assert all(isinstance(monomial, tuple) and len(monomial) == 4 for monomial in numerator.monomials)
    assert all(isinstance(term, tuple) and isinstance(term[0], tuple) for term in numerator.terms)
    shape = [[0], [1]]
    dual = Dual(shape, [3, 2])
    assert isinstance(dual.values, tuple)
    assert dual.values == (3, 2)
    shape[1][0] = 9
    copied_shape = dual.shape
    copied_shape[1][0] = 7
    assert dual.shape == [[0], [1]]
    coefficients = numerator.evaluate_batch([], [[], [], []])
    assert coefficients == ((2 + 0j,), (2 + 0j,), (2 + 0j,))

    plan = ReductionPlan(1, 0)
    batch = plan.pack(shifts=[[[0, 0, 0, 0]]], masses_squared=[[2]], mu=[1])
    workspace = plan.workspace()
    result = workspace.reduce_batch(batch)
    assert result.statuses == ("success",)
    assert result.success == (True,)
    assert isinstance(result.diagnostics, tuple)
    assert all(isinstance(notes, tuple) for notes in result.diagnostics)
    assert result.laurent_names == ("uv_single", "ir_single", "ir_double", "finite")
    assert isinstance(result.lane(0), tuple)
    assert isinstance(result.lane(0)[0], tuple)
    assert len(result.lane(0)[0]) == 4
    assert result.stats == result.execution_stats
    contracted = workspace.contract(result, monomials=numerator.monomials, coefficients=[coefficients[0]])
    assert contracted.statuses == ("success",)
    assert contracted.laurent_names == result.laurent_names
    assert contracted.lane(0) == tuple(2 * coefficient for coefficient in result.lane(0)[0])


def test_native_operations_work_with_canonical_symbolica_imports_blocked() -> None:
    """A fresh process prevents an earlier test from warming Python aliases."""
    program = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        assert not any(name == "symbolica" or name.startswith("symbolica.") for name in sys.modules)

        class BlockCanonicalSymbolica(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "symbolica" or fullname.startswith("symbolica."):
                    raise ImportError("canonical Symbolica import is forbidden in this test: " + fullname)
                return None

        sys.meta_path.insert(0, BlockCanonicalSymbolica())
        import tirust as t

        variables = [t.E("blocked_interop::q" + str(index)) for index in range(4)]
        expression = variables[0]**2 + 2*variables[1] - 3
        tensor = t.TensorExpression(expression)
        prepared = t.PreparedNumerator.from_tensor(tensor, variables)
        assert (prepared.expression - expression).expand() == t.E("0")
        assert prepared.evaluate_batch([], [[], [], []])
        model = t.Model.phi3()
        diagram = t.FeynmanDiagram.from_dot(model, {BUBBLE_DOT!r})
        graph = t.PreparedGraph.from_diagram(diagram)
        assert graph.diagram is diagram
        assert isinstance(graph.diagram.numerator_expression(), t.TensorExpression)
        t.initialize()
        t.initialize()
        plan = t.ReductionPlan(1, 0)
        batch = plan.pack(shifts=[[[0,0,0,0]], [[0,0,0,0]]], masses_squared=[[2], [3]], mu=[1,1])
        result = plan.workspace().reduce_batch(batch)
        assert result.statuses == ("success", "success"), result.diagnostics
        assert result.stats["input_rows"] == 2
        assert all(result.lane(index) for index in range(2))
        assert not any(name == "symbolica" or name.startswith("symbolica.") for name in sys.modules)
        assert not any(name.startswith("tirust._types") for name in sys.modules)
        print("TIRust native interoperability without canonical aliases: PASS")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60, check=False
    )
    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    assert "without canonical aliases: PASS" in completed.stdout


@pytest.mark.parametrize("source", ["unknown", "SUPPLIED"])
def test_unknown_graph_preparation_source_is_an_explicit_error(source: str) -> None:
    diagram = FeynmanDiagram.from_dot(Model.phi3(), BUBBLE_DOT)
    with pytest.raises(ValueError, match="source must be"):
        PreparedGraph.from_diagram(diagram, source=source)
