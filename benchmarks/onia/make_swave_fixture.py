#!/usr/bin/env python3
"""Fixture of crates/iregi-rust/tests/onia.rs's S-wave amplitude test.

Each case is one S-wave amplitude at one point and polarization labels
(swave.py's key, e.g. 01 for 1S0 or 1000 for 3S1): every distinct loop
integral (lane) with its contracted coefficients sum over diagrams of tree
factor x numerator coefficient, keyed by TIRust basis index; the reference
amplitude and the Born (from swave.py's output); the number of gluons and the
single-pole prediction (log part and constant, swave.pole_targets).

usage: make_swave_fixture.py FIXTURE_JSON --case NUMERATORS OUTPUT POINT LABELS [--case ...]
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402
import swave  # noqa: E402


def text(x):
    return repr(float(x))


def case(diagrams, outputs, k, state, key):
    lanes = {}
    for diagram in diagrams:
        point_entry = diagram["points"][k]
        for item in point_entry["polarizations"]:
            labels, entry = item[:-1], item[-1]
            if "".join(map(str, labels)) != key or all(abs(c[0]) + abs(c[1]) == 0 for _, c in entry["terms"]):
                continue
            shifts, masses, props, tree = swave.integral_of(diagram, point_entry, entry)
            rank = max(sum(e) for e, _ in entry["terms"])
            lane = lanes.setdefault((props, rank), {"n": len(shifts), "rank": rank, "shifts": shifts,
                                                     "masses_squared": masses, "coefficients": {}})
            index = {m: i for i, m in enumerate(run.basis(rank))}
            for exponents, (re, im) in entry["terms"]:
                m = index[tuple(exponents)]
                lane["coefficients"][m] = lane["coefficients"].get(m, 0j) + complex(re, im) * tree
    result = next(r for r in outputs["points"] if r["index"] == point_entry["index"])
    pol = result["pols"][key]
    assert result["summary"]["reference"], "point without a reference"
    n, log_part, constant = swave.pole_targets(point_entry, state)
    return {
        "state": state,
        "point": point_entry["index"],
        "labels": key,
        "gluons": n,
        "m2": point_entry["m2"],
        "mu": text(swave.MU),
        "born": [text(x) for x in pol["born"]],
        "born_mass_derivative": [text(x) for x in pol["born_mass_derivative"]],
        "reference": [[text(x) for x in z] for z in pol["reference"]],
        "single_pole_log": [text(log_part.real), text(log_part.imag)],
        "single_pole_target": [text(constant.real), text(constant.imag)],
        "lanes": [{"n": v["n"], "rank": v["rank"],
                   "shifts": [[text(x) for x in row] for row in v["shifts"]],
                   "masses_squared": [text(x) for x in v["masses_squared"]],
                   "coefficients": [[m, text(c.real), text(c.imag)] for m, c in sorted(v["coefficients"].items())]}
                  for v in lanes.values()],
    }


def main():
    fixture = Path(sys.argv[1])
    args = sys.argv[2:]
    cases = []
    while args:
        assert args[0] == "--case", args
        numerators, output, index, key = Path(args[1]), Path(args[2]), int(args[3]), args[4]
        args = args[5:]
        data = json.loads(numerators.read_text())
        diagrams = [d for d in data["diagrams"] if d["loops"] == 1]
        outputs = json.loads(output.read_text())
        assert outputs["state"] == data["state"]
        position = {d["index"]: i for i, d in enumerate(diagrams[0]["points"])}
        cases.append(case(diagrams, outputs, position[index], data["state"], key))
    fixture.write_text(json.dumps({"generator": "benchmarks/onia/make_swave_fixture.py", "cases": cases}) + "\n")
    print(fixture, len(cases), "cases,", sum(len(c["lanes"]) for c in cases), "lanes")


if __name__ == "__main__":
    main()
