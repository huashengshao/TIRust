"""Fixture for dotted integrals with a collinear pair (two massless lines
whose invariant vanishes), and for a pentagon with a doubled soft line.

Writes crates/iregi-rust/tests/fixtures/onia-collinear.json: scalar
integrals (rank 0) of a generic triangle, box and pentagon sharing the pair
(lines 0 and 1, separated by k = (3, 0, 0, 3)), with one line doubled, and
("soft") of a pentagon whose massless line 1, doubled, joins two on-shell
legs of the m^2 = 4 lines 0 and 2. From original IREGI's quad scalar API
with powers (the build run.py uses, ONIA_QUAD), in IREGI's native
convention, mu = 3. A pentagon with its pair member or its soft line doubled
needs the (D - 4) terms of its reduction.
"""
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
SHIFTS = [[0, 0, 0, 0], [3, 0, 0, 3], [5, 1, 0, 2], [4, -1, 2, 1], [1, 0.5, -1, 2]]
MASSES = [0, 0, 4, 1, 2]
CASES = [(3, 0), (3, 1), (4, 0), (4, 1), (5, 2), (5, 0), (5, 1)]
SOFT_SHIFTS = [[0, 0, 0, 0], [2, 0, 0, 0], [4.5, 1.5, 0, 0], [1, 0.5, -1, 2], [-1, 0.25, 0.75, 0.5]]
SOFT_MASSES = [4, 0, 4, 2, 1]
SOFT_CASES = [(5, 1)]


def evaluate(shifts, masses, cases, tag):
    text = ""
    for k, (n, dotted) in enumerate(cases):
        text += f"EVAL {tag}{k} scalar 0 {n} 0 0 3 1 1e-12\n"
        for i in range(n):
            power = 2 if i == dotted else 1
            text += " ".join(repr(float(x)) for x in shifts[i]) + f" {float(masses[i])!r} 0.0 {power}\n"
    run = subprocess.run([str(QUAD)], input=text + "QUIT\n", capture_output=True, text=True, timeout=600, check=True)
    results = {}
    for line in run.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            results[r["id"]] = r
    out = []
    for k, (n, dotted) in enumerate(cases):
        r = results[f"{tag}{k}"]
        assert r["stable"], f"quad IREGI reports case {tag}{k} unstable"
        out.append({"n": n, "dotted": dotted, "reference": [[a.strip(), b.strip()] for a, b in r["moments"][0]]})
    return out


def main():
    cases = evaluate(SHIFTS, MASSES, CASES, "c")
    soft = evaluate(SOFT_SHIFTS, SOFT_MASSES, SOFT_CASES, "s")
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-collinear.json"
    fixture = {
        "shifts": SHIFTS,
        "masses_squared": MASSES,
        "mu": 3,
        "cases": cases,
        "soft": {"shifts": SOFT_SHIFTS, "masses_squared": SOFT_MASSES, "cases": soft},
    }
    out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} + {len(soft)} cases)")


if __name__ == "__main__":
    main()
