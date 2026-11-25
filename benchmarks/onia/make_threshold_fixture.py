"""Fixture for dotted threshold bubbles (equal masses at p^2 = 4 m^2).

Writes crates/iregi-rust/tests/fixtures/onia-threshold-bubbles.json: scalar
integrals (rank 0) of the bubble at threshold with raised powers, and of a
triangle containing it with each line doubled, from original IREGI's quad
scalar API with powers (ONIA_QUAD, as run.py), in IREGI's native
convention, mu = 3. Dimensional regularization makes the doubled threshold
bubble finite: its 4-dimensional mass derivative diverges like 1/beta.
"""
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
SHIFTS = [[0, 0, 0, 0], [4, 0, 0, 0], [3, 1, -1, 2]]
MASSES = [4, 4, 1]
CASES = [[2, 1], [1, 2], [3, 1], [1, 3], [2, 1, 1], [1, 2, 1], [1, 1, 2]]


def main():
    text = ""
    for k, powers in enumerate(CASES):
        n = len(powers)
        text += f"EVAL c{k} scalar 0 {n} 0 0 3 1 1e-12\n"
        for i in range(n):
            text += " ".join(repr(float(x)) for x in SHIFTS[i]) + f" {float(MASSES[i])!r} 0.0 {powers[i]}\n"
    run = subprocess.run([str(QUAD)], input=text + "QUIT\n", capture_output=True, text=True, timeout=600, check=True)
    results = {json.loads(l[7:])["id"]: json.loads(l[7:]) for l in run.stdout.splitlines() if l.startswith("RESULT ")}
    cases = []
    for k, powers in enumerate(CASES):
        r = results[f"c{k}"]
        assert r["stable"], f"quad IREGI reports case {k} unstable"
        cases.append({"powers": powers, "reference": [[a.strip(), b.strip()] for a, b in r["moments"][0]]})
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-threshold-bubbles.json"
    out.write_text(json.dumps({"shifts": SHIFTS, "masses_squared": MASSES, "mu": 3, "cases": cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
