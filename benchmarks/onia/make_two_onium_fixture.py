"""Fixture for two-onium pentagons whose Coulomb partial fractions reach a soft
triangle.

Writes crates/iregi-rust/tests/fixtures/onia-two-onium-pentagons.json: two
binary64 lanes of onia_loop's e+ e- > J/psi eta_c trace at v = 0 (MadLoop's
supplied invariants, masses and mu^2, as captured by tools/tirust_capture),
with original IREGI's arrays in quadruple precision (IBP mode, which reports
them stable; ML5 components in IREGI's native convention) from the quad
reference build that run.py uses (ONIA_QUAD).

Each pentagon has two Coulomb triples (heavy, gluon, heavy with
D_Q + D_Qbar = 2 D_g) sharing a heavy line. Their partial fractions double a
gluon that is on shell against both heavy lines: a soft line, which empties
its row of the triangle's Cayley matrix.
"""
import json
import os
import pathlib
import subprocess
from decimal import Decimal

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
H = "0x1.2000000000000p+1"  # m_c^2 = 2.25
Z = "0x0.0p+0"
A, B, C, E = "0x1.528f5c28f5c29p+1", "0x1.c652e3355a3fcp+0", "0x1.1eb19e7f90b00p+0", "0x1.2b8bf12d2f8cap-1"
S1, S2, S3 = "0x1.bfbedfa43fe5dp+4", "0x1.d1bedfa43fe5dp+5", "0x1.bfbedfa43fe5dp+6"
S1b = "0x1.bfbedfa43fe5fp+4"
LANES = [
    {
        "label": "masses m, 0, m, 0, m",
        "shifts": [[Z, Z, Z, Z], ["-" + A, B, "-" + C, "-" + E], ["-0x1.528f5c28f5c29p+2", Z, Z, Z],
                   ["-0x1.fbd70a3d70a3ep+2", B, "-" + C, "-" + E], ["-0x1.528f5c28f5c29p+3", Z, Z, Z]],
        "invariants": [Z, H, S1, S2, S3, H, Z, H, S1b, S2, S1, H, Z, H, S1, S2, S1b, H, Z, H, S3, S2, S1, H, Z],
        "masses": [H, Z, H, Z, H],
    },
    {
        "label": "masses m, 0, m, m, 0",
        "shifts": [[Z, Z, Z, Z], ["-" + A, B, "-" + C, "-" + E], ["-0x1.528f5c28f5c29p+2", Z, Z, Z],
                   ["0x1.528f5c28f5c29p+2", Z, Z, Z], [A, B, "-" + C, "-" + E]],
        "invariants": [Z, H, S1, S1, H, H, Z, H, S2, S1, S1, H, Z, S3, S2, S1, S2, S3, Z, H, H, S1, S2, H, Z],
        "masses": [H, Z, H, H, Z],
    },
]
MU2 = "0x1.2000000000000p+3"
RANK = 3


def value(h):
    return float.fromhex(h)


def exact(x):
    return format(Decimal(x), ".40e")


def main():
    text = ""
    for k, lane in enumerate(LANES):
        n = len(lane["masses"])
        text += f"EVAL c{k} tensor 0 {n} {RANK} 0 {exact(Decimal(value(MU2)).sqrt())} 1 1e-12\n"
        for shift, mass in zip(lane["shifts"], lane["masses"]):
            text += " ".join(exact(value(x)) for x in shift) + f" {exact(value(mass))} 0.0 1\n"
    run = subprocess.run([str(QUAD)], input=text + "QUIT\n", capture_output=True, text=True, timeout=3600, check=True)
    results = {}
    for line in run.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            results[r["id"]] = r
    cases = []
    for k, lane in enumerate(LANES):
        r = results[f"c{k}"]
        assert r["stable"], f"quad IREGI reports lane {k} unstable"
        cases.append(
            {
                "label": lane["label"],
                "rank": RANK,
                "shifts": [[repr(value(x)) for x in shift] for shift in lane["shifts"]],
                "invariants": [repr(value(x)) for x in lane["invariants"]],
                "masses_squared": [repr(value(x)) for x in lane["masses"]],
                "mu_squared": repr(value(MU2)),
                "reference": [[[a.strip(), b.strip()] for a, b in monomial] for monomial in r["moments"]],
            }
        )
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-two-onium-pentagons.json"
    out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
