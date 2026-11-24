"""Fixture for dotted integrals with a collinear pair (two massless lines
whose invariant vanishes).

Writes crates/iregi-rust/tests/fixtures/onia-collinear.json: scalar
integrals (rank 0) of a generic triangle, box and pentagon sharing the pair
(lines 0 and 1, separated by k = (3, 0, 0, 3)), with one line doubled, from
original IREGI's quad scalar API with powers (the build run.py uses,
ONIA_QUAD), in IREGI's native convention, mu = 3. The pentagon with a pair
member doubled has no entry: the direct engine must refuse it.
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
CASES = [(3, 0), (3, 1), (4, 0), (4, 1), (5, 2)]


def main():
    text = ""
    for k, (n, dotted) in enumerate(CASES):
        text += f"EVAL c{k} scalar 0 {n} 0 0 3 1 1e-12\n"
        for i in range(n):
            power = 2 if i == dotted else 1
            text += " ".join(repr(float(x)) for x in SHIFTS[i]) + f" {float(MASSES[i])!r} 0.0 {power}\n"
    run = subprocess.run([str(QUAD)], input=text + "QUIT\n", capture_output=True, text=True, timeout=600, check=True)
    results = {}
    for line in run.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            results[r["id"]] = r
    cases = []
    for k, (n, dotted) in enumerate(CASES):
        r = results[f"c{k}"]
        assert r["stable"], f"quad IREGI reports case {k} unstable"
        cases.append({"n": n, "dotted": dotted, "reference": [[a.strip(), b.strip()] for a, b in r["moments"][0]]})
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-collinear.json"
    out.write_text(json.dumps({"shifts": SHIFTS, "masses_squared": MASSES, "mu": 3, "cases": cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
