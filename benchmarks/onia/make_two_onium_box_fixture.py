"""Fixture for two-onium boxes whose reduction reaches a doubled threshold
bubble.

Writes crates/iregi-rust/tests/fixtures/onia-two-onium-boxes.json: two
binary64 lanes of onia_loop's e+ e- > J/psi eta_c trace at v = 0 (MadLoop's
supplied invariants, masses and mu^2, as captured by tools/tirust_capture),
with original IREGI's arrays in quadruple precision (IBP mode, which reports
them stable; ML5 components in IREGI's native convention) from the quad
reference build that run.py uses (ONIA_QUAD).

Each box has a Coulomb triple (heavy, gluon, heavy with D_Q + D_Qbar = 2 D_g)
and two heavy pairs at p^2 = 4 m^2. The partial fraction doubles the gluon,
and the doubled triangles reduce to threshold bubbles with raised powers.

The binary64 shifts put those pairs off threshold by their rounding, about
3e-17 relative, and an integral with a threshold pair depends on
beta = sqrt(1 - 4 m^2 / p^2): IREGI, which takes its invariants from the
shifts, moves by up to 9e-9 against the supplied invariants, which are on
threshold. The reference therefore takes the shifts in quadruple precision
with the heavy pair's spatial momentum rescaled to put both pairs exactly on
threshold. Its other invariants then agree with the supplied ones to their
rounding.
"""
import json
import os
import pathlib
import subprocess
from decimal import Decimal, getcontext

getcontext().prec = 80
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
H = "0x1.2000000000000p+1"  # m_c^2 = 2.25
T = "0x1.2000000000000p+3"  # 4 m_c^2 = 9
Z = "0x0.0p+0"
E1, X, Y, W = "0x1.528f5c28f5c29p+1", "0x1.c652e3355a3fcp+0", "0x1.1eb19e7f90b00p+0", "0x1.2b8bf12d2f8cap-1"
E2, X2, Y2, W2 = "0x1.528f5c28f5c29p+2", "0x1.c652e3355a3fcp+1", "0x1.1eb19e7f90b00p+1", "0x1.2b8bf12d2f8cap+0"
S1, S3 = "0x1.d1bedfa43fe5dp+5", "0x1.bfbedfa43fe5dp+6"
LANES = [
    {
        "label": "masses m, 0, m, m; threshold pairs 02, 23",
        "shifts": [[Z, Z, Z, Z], ["-" + E1, "-" + X, Y, W], ["-" + E2, "-" + X2, Y2, W2], ["-0x1.528f5c28f5c29p+3", Z, Z, Z]],
        "invariants": [Z, H, T, S3, H, Z, H, S1, T, H, Z, T, S3, S1, T, Z],
        "masses": [H, Z, H, H],
    },
    {
        "label": "masses m, 0, m, m; threshold pairs 02, 03",
        "shifts": [[Z, Z, Z, Z], ["-" + E1, "-" + X, Y, W], ["-" + E2, "-" + X2, Y2, W2], [E2, "-" + X2, Y2, W2]],
        "invariants": [Z, H, T, T, H, Z, H, S1, T, H, Z, S3, T, S1, S3, Z],
        "masses": [H, Z, H, H],
    },
]
MU2 = "0x1.2000000000000p+3"
RANK = 3


def value(h):
    return float.fromhex(h)


def exact(x):
    return format(Decimal(x), ".45e")


def on_threshold(shifts):
    """Quadruple-precision shifts with p_2^2 = 4 m^2 exactly: line 2's spatial
    momentum rescaled, line 1 at half of it (the Coulomb triple), and line 3
    keeping its relation to line 2 (at rest, or p_2 with the energy flipped)."""
    p = [[Decimal(value(x)) for x in q] for q in shifts]
    energy, spatial = p[2][0], p[2][1:]
    scale = ((energy * energy - Decimal(value(T))) / sum(x * x for x in spatial)).sqrt()
    p[2] = [energy] + [x * scale for x in spatial]
    p[1] = [x / 2 for x in p[2]]
    if any(x != 0 for x in p[3][1:]):
        p[3] = [-energy] + p[2][1:]
    return p


def main():
    text = ""
    for k, lane in enumerate(LANES):
        n = len(lane["masses"])
        text += f"EVAL c{k} tensor 0 {n} {RANK} 0 {exact(Decimal(value(MU2)).sqrt())} 1 1e-12\n"
        for shift, mass in zip(on_threshold(lane["shifts"]), lane["masses"]):
            text += " ".join(exact(x) for x in shift) + f" {exact(value(mass))} 0.0 1\n"
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
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-two-onium-boxes.json"
    out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
