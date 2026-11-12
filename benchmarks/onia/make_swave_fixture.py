#!/usr/bin/env python3
"""Fixture of crates/iregi-rust/tests/onia.rs's S-wave amplitude test.

For each colour state and chosen point, the 1S0 amplitude at the gluon
polarizations (e1 of gluon 1, e2 of gluon 2): every distinct loop integral
(lane) with its contracted coefficients sum over diagrams of tree factor x
numerator coefficient, keyed by TIRust basis index; the reference amplitude
and the Born (from swave.py's output), and the predicted single-pole ratio.

usage: make_swave_fixture.py POINTS_JSON SINGLET_NUMERATORS SINGLET_OUTPUT
       OCTET_NUMERATORS OCTET_OUTPUT FIXTURE_JSON POINT [POINT...]
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


def case(diagrams, outputs, k, state, s_invariant):
    lanes = {}
    for diagram in diagrams:
        point_entry = diagram["points"][k]
        for l1, l2, entry in point_entry["polarizations"]:
            if (l1, l2) != (0, 1) or all(abs(c[0]) + abs(c[1]) == 0 for _, c in entry["terms"]):
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
    pol = result["pols"]["01"]
    assert result["summary"]["reference"], "point without a reference"
    target = (1 - swave.CF / swave.CA, math.pi) if state == "singlet" else (1.5 - swave.CF / swave.CA, math.pi / 2)
    return {
        "state": state,
        "point": point_entry["index"],
        "s": s_invariant,
        "m2": point_entry["m2"],
        "mu": text(swave.MU),
        "born": [text(x) for x in pol["born"]],
        "born_mass_derivative": [text(x) for x in pol["born_mass_derivative"]],
        "reference": [[text(x) for x in z] for z in pol["reference"]],
        "single_pole_target": [text(x) for x in target],
        "lanes": [{"n": v["n"], "rank": v["rank"],
                   "shifts": [[text(x) for x in row] for row in v["shifts"]],
                   "masses_squared": [text(x) for x in v["masses_squared"]],
                   "coefficients": [[m, text(c.real), text(c.imag)] for m, c in sorted(v["coefficients"].items())]}
                  for v in lanes.values()],
    }


def main():
    points_json, singlet_num, singlet_out, octet_num, octet_out, fixture = map(Path, sys.argv[1:7])
    chosen = [int(x) for x in sys.argv[7:]]
    s_of = {p["index"]: p["s"] for p in next(p for p in json.loads(points_json.read_text())["processes"]
                                             if p["id"] == "gg_QQ")["points"]}
    cases = []
    for state, num, out in (("singlet", singlet_num, singlet_out), ("octet", octet_num, octet_out)):
        data = json.loads(num.read_text())
        assert data["state"] == state
        diagrams = [d for d in data["diagrams"] if d["loops"] == 1]
        outputs = json.loads(out.read_text())
        position = {d["index"]: i for i, d in enumerate(diagrams[0]["points"])}
        for index in chosen:
            cases.append(case(diagrams, outputs, position[index], state, s_of[index]))
    fixture.write_text(json.dumps({"generator": "benchmarks/onia/make_swave_fixture.py", "cases": cases}) + "\n")
    print(fixture, len(cases), "cases,", sum(len(c["lanes"]) for c in cases), "lanes")


if __name__ == "__main__":
    main()
