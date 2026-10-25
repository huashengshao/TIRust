"""Installed-wheel facade checks using the existing retained positive-mass fixtures.

These check Python routing, state and transport. They are not new reference or
performance qualification of the Rust high-point algorithms.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tirust import ComplexFloat, Dual, Float, LaneError, ReductionPlan


ROOT = Path(__file__).resolve().parents[2]


def _fixture(n):
    source = (
        "validation/public-highpoint/fixture-conditioned_masses.json"
        if n == 6
        else "validation/highpoint-seven-point-current-cargo/fixture.json"
    )
    sample = json.loads((ROOT / source).read_text())["samples"][0]
    propagators = sample["propagators"][:n]
    return {
        "shifts": [item["shift"] for item in propagators],
        "masses_squared": [tuple(item["mass_squared"]) for item in propagators],
        "mu": sample["mu"],
    }


def _batch(plan, sample, scales):
    return plan.pack(
        shifts=[sample["shifts"] for _ in scales],
        masses_squared=[sample["masses_squared"] for _ in scales],
        mu=scales,
    )


def test_generic_tag_profile_and_result_snapshot_are_owned():
    plan = ReductionPlan(1, 0)
    workspace = plan.workspace(profiling=True, accuracy_bits=40)
    assert workspace.engine == "generic"
    assert any("profiling=true" in note for note in workspace.preparation_notes)
    batch = plan.pack(shifts=[[[0, 0, 0, 0]]], masses_squared=[[2]], mu=[3])
    result = workspace.reduce_batch(batch)
    detail = result.engine_execution
    assert detail["engine"] == "generic"
    assert detail["execution"] == result.stats == result.execution_stats
    assert set(detail["certificate_execution"]) == {
        "input_rows", "evaluated_rows", "passes", "reduction_batches"
    }
    assert result.success == (True,)
    assert result.profile["total_seconds"] >= 0
    # Getters return owned dictionaries and later calls cannot overwrite results.
    detail["execution"]["master_rows"] = -1
    assert result.engine_execution["execution"]["master_rows"] >= 1
    empty = workspace.reduce_batch([])
    assert empty.engine_execution["execution"]["input_rows"] == 0
    assert result.engine_execution["execution"]["input_rows"] == 1


@pytest.mark.parametrize("shape", [[[0]], [[0, 0], [1, 0], [0, 1], [1, 1]]])
def test_explicit_dual_shape_uses_generic_and_preserves_components(shape):
    plan = ReductionPlan(1, 0)
    workspace = plan.workspace(
        dtype="float", precision=128, accuracy_bits=53, dual_shape=shape
    )
    assert workspace.engine == "generic"
    assert workspace.dual_shape == shape
    assert any("explicit Dual layout" in note for note in workspace.preparation_notes)
    result = workspace.reduce_batch(
        plan.pack(shifts=[[[0, 0, 0, 0]]], masses_squared=[[2]], mu=[3])
    )
    assert result.success == (True,)
    assert result.engine_execution["engine"] == "generic"
    for value in result.lane(0)[0]:
        assert isinstance(value, Dual)
        assert value.shape == shape
        assert len(value.values) == len(shape)
        assert all(component == 0 for component in value.values[1:])


def test_highpoint_profiling_and_zero_order_dual_keep_generic_capabilities():
    plan = ReductionPlan(6, 0, algorithm="pv")
    for options in ({"profiling": True}, {"dual_shape": [[0]]}):
        workspace = plan.workspace(dtype="float", precision=128, accuracy_bits=53, **options)
        assert workspace.engine == "generic"
        assert workspace.preparation_notes
        empty = workspace.reduce_batch([])
        assert empty.engine_execution["engine"] == "generic"
        assert empty.stats["input_rows"] == 0
        assert (empty.profile is not None) == options.get("profiling", False)


@pytest.mark.parametrize("n,rank,kind,subsets", [(6, 1, "six_point", 6), (7, 0, "seven_point", 21)])
def test_eligible_highpoint_batch_contraction_failure_and_recovery(n, rank, kind, subsets):
    plan = ReductionPlan(n, rank, algorithm="pv")
    workspace = plan.workspace(dtype="float", precision=128, accuracy_bits=53)
    assert workspace.engine == kind
    assert workspace.preparation_notes
    sample = _fixture(n)
    result = workspace.reduce_batch(_batch(plan, sample, [sample["mu"], "0", sample["mu"]]))
    assert result.success == (True, False, True)
    assert result.statuses[1] == "invalid_input"
    assert result.lanes()[1] is None
    assert result.diagnostics[1]
    with pytest.raises(LaneError):
        result.lane(1)
    assert result.lane(0) == result.lane(2)
    assert result.basis == plan.basis
    assert isinstance(result.lane(0)[0][3], ComplexFloat)
    assert result.profile is None
    detail = result.engine_execution
    assert detail["engine"] == kind
    assert detail["parent_input_rows"] == result.stats["input_rows"] == 3
    assert detail["child_batch_calls"] == detail["parent_batch_calls"] == 1
    assert detail["packed_child_rows"] == detail["child_execution"]["input_rows"] == 2 * subsets
    assert detail["packed_parent_rows"] == detail["parent_execution"]["input_rows"] == 2
    for key, value in result.stats.items():
        if key != "input_rows":
            assert value == detail["child_execution"][key] + detail["parent_execution"][key]
    assert set(detail["child_certificate_execution"]) == set(detail["parent_certificate_execution"])

    monomials = [(0, 0, 0, 0)] if rank == 0 else [(0, 0, 0, 0), (1, 0, 0, 0)]
    coefficients = [["2"] if rank == 0 else ["2", "3"]] * 3
    contracted = workspace.contract(result, monomials=monomials, coefficients=coefficients)
    assert contracted.success == result.success
    assert contracted.lanes()[1] is None
    positions = [result.basis.index(monomial) for monomial in monomials]
    for pole, actual in enumerate(contracted.lane(0)):
        expected = sum(
            Float(coefficient, precision=128) * result.lane(0)[position][pole]
            for coefficient, position in zip(coefficients[0], positions, strict=True)
        )
        assert abs(actual - expected) < Float("1e-28", precision=128)

    # Empty and entirely invalid calls must clear nested execution counters.
    for later in (workspace.reduce_batch([]), workspace.reduce_batch(_batch(plan, sample, ["0"]))):
        assert later.engine_execution["engine"] == kind
        assert later.engine_execution["child_batch_calls"] == 0
        assert later.engine_execution["parent_batch_calls"] == 0
        assert later.stats["master_rows"] == 0
    # The earlier captured detail remains unchanged across workspace reuse.
    assert result.engine_execution == detail
    recovered = workspace.reduce_batch(_batch(plan, sample, [sample["mu"]]))
    assert recovered.success == (True,)
    assert recovered.lane(0) == result.lane(0)

