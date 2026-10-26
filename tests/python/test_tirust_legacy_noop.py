"""Installed-wheel regression for conservative Legacy no-op admission.

Run only against the next fresh wheel containing LegacyNoOp. This checks public
routing/state/transport on one scalar fixture, not rank-six numerical coverage.
Requested PV and Legacy enum retention also has separate core constructor tests;
Python currently exposes no direct getter for those requested settings.
"""
from copy import deepcopy

import pytest
from tirust import LaneError, ReductionPlan


def test_legacy_noop_admits_unchanged_lanes_and_never_executes_rejected_neighbor():
    # The same exact fixture and changed neighbor as the core no-op guard test.
    good = [["0", "0", "0", "0"], ["0", "1", "0", "0"],
            ["0", "0", "1", "0"], ["0", "0", "0", "1"],
            ["1/2", "0", "0", "0"]]
    changed = deepcopy(good)
    changed[4][0] = "16777217/16777216"  # 1 + 2^-24: would change a selected snap.
    masses = ["1", "2", "3", "4", "2"]
    plan = ReductionPlan(5, 0, algorithm="pv", input_policy="legacy_on_shell",
                         on_shell_threshold="1/1000000")
    workspace = plan.workspace(dtype="float", precision=128, accuracy_bits=53,
                               profiling=True)
    assert workspace.engine == "legacy_noop"
    assert plan.propagators == 5 and plan.rank == 0 and plan.powers == [1] * 5
    assert any("LegacyNoOp" in note for note in plan.preparation_notes)

    def packed(shifts):
        # All caller data remain ordinary immutable-valued Python containers.
        data = {"shifts": deepcopy(shifts),
                "masses_squared": [masses.copy() for _ in shifts],
                "mu": ["3"] * len(shifts)}
        saved = deepcopy(data)
        batch = plan.pack(**data)
        return batch, data, saved

    # The first rejected-only call must not even prepare its Preserve child.
    rejected_input, caller, saved = packed([changed])
    rejected = workspace.reduce_batch(rejected_input)
    assert caller == saved
    assert rejected.statuses == ("invalid_input",) and rejected.lanes() == (None,)
    with pytest.raises(LaneError):
        rejected.lane(0)
    detail = rejected.engine_execution
    assert detail["engine"] == "legacy_noop"
    assert detail["input_rows"] == detail["rejected_rows"] == 1
    assert detail["accepted_rows"] == 0 and detail["query_count"] > 0
    assert detail["child"] is None and detail["child_preparation_attempts"] == 0
    assert detail["child_preparation_error"] is None
    assert rejected.stats["master_rows"] == rejected.stats["reduction_rows"] == 0
    assert any("no Preserve child was executed" in note for note in rejected.diagnostics[0])

    batch, caller, saved = packed([good, changed, good])
    result = workspace.reduce_batch(batch)
    assert caller == saved
    assert result.success == (True, False, True)
    assert result.statuses[1] == "invalid_input" and result.lanes()[1] is None
    assert result.lane(0) == result.lane(2)
    assert len(result.lane(0)) == 1 and len(result.lane(0)[0]) == 4
    assert result.lane(0)[0][3] != 0
    detail = result.engine_execution
    assert detail["engine"] == "legacy_noop"
    assert detail["input_rows"] == result.stats["input_rows"] == 3
    assert detail["accepted_rows"] == 2 and detail["rejected_rows"] == 1
    assert detail["child_preparation_attempts"] == 1
    assert detail["child_preparation_error"] is None
    assert detail["certificate_execution"]["input_rows"] == 3
    assert detail["certificate_execution"]["evaluated_rows"] > 0
    child = detail["child"]
    assert child is not None and child["engine"] == "generic"
    assert child["execution"]["input_rows"] == 2  # Rejected neighbor never entered.
    for name, value in result.stats.items():
        if name != "input_rows":
            assert value == child["execution"][name]
    assert any("Requested LegacyOnShell retained" in note for note in result.diagnostics[0])
    assert result.profile is not None and result.profile["total_seconds"] >= 0

    # Compare against unchanged-input execution with the same requested PV.
    # This is an interface equivalence check, not an independent integral oracle.
    preserve = ReductionPlan(5, 0, algorithm="pv", input_policy="preserve")
    reference = preserve.workspace(dtype="float", precision=128, accuracy_bits=53)
    assert reference.engine == "generic"
    reference_result = reference.reduce_batch(preserve.pack(
        shifts=[deepcopy(good)], masses_squared=[masses.copy()], mu=["3"]))
    assert reference_result.success == (True,)
    assert result.lane(0) == reference_result.lane(0)

    # Empty/rejected reuse clears captured work even after a child exists.
    for later in (workspace.reduce_batch([]), workspace.reduce_batch(rejected_input)):
        current = later.engine_execution
        assert current["engine"] == "legacy_noop"
        assert current["accepted_rows"] == 0 and current["child"] is None
        assert current["child_preparation_attempts"] == 0
        assert later.stats["master_rows"] == later.stats["reduction_rows"] == 0
        assert current["certificate_execution"]["input_rows"] == later.rows
    assert result.engine_execution == detail  # Owned snapshot survives reuse.
    recovered_input, caller, saved = packed([good])
    recovered = workspace.reduce_batch(recovered_input)
    assert caller == saved and recovered.success == (True,)
    assert recovered.lane(0) == result.lane(0)
    recovered_detail = recovered.engine_execution
    assert recovered_detail["accepted_rows"] == 1 and recovered_detail["rejected_rows"] == 0
    assert recovered_detail["child_preparation_attempts"] == 0  # Existing child reused.
    assert recovered_detail["child"]["engine"] == "generic"
    assert recovered_detail["child"]["execution"]["input_rows"] == 1
