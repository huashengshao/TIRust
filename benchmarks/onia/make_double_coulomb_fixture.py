"""Fixture for two-onium pentagons with two Coulomb triples.

Writes crates/iregi-rust/tests/fixtures/onia-double-coulomb.json: the two
binary64 lanes of onia_loop's e+ e- > J/psi eta_c trace at v = 0 that hold
both onia's Coulomb triples (heavy, gluon, heavy with D_Q + D_Qbar = 2 D_g;
MadLoop's loop groups 41 and 42). They have rank 3, and MadLoop's supplied
invariants, masses and mu^2, as captured by tools/tirust_capture. The
reference is original IREGI's arrays in quadruple precision (IBP mode, which
reports them stable; ML5 components in IREGI's native convention) from the
quad reference build that run.py uses (ONIA_QUAD).

The two triples share a heavy line, which the partial fractions raise twice.
IREGI takes its invariants from the shifts, and an integral with a threshold
pair depends on beta = sqrt(1 - 4 m^2 / p^2), so the reference takes the
shifts in quadruple precision with both onia exactly on threshold:
- A = -P_psi keeps its energy, and its spatial momentum is rescaled to
  A^2 = 4 m^2;
- the other onium momentum (B = -P_eta, or C = P_eta with A's spatial
  momentum) is rebuilt from the same parts;
- each shift is the half-integer combination of A and B (or C) that the
  binary64 one is.

The other invariants then agree with the supplied ones to their rounding.
Those shifts and their squared differences are stored too
(`threshold_shifts`, `threshold_invariants`), so TIRust can be evaluated on
exactly IREGI's inputs.
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
E1, X1, Y1, W1 = "0x1.528f5c28f5c29p+1", "0x1.c652e3355a3fcp+0", "0x1.1eb19e7f90b00p+0", "0x1.2b8bf12d2f8cap-1"
E2, X2, Y2, W2 = "0x1.528f5c28f5c29p+2", "0x1.c652e3355a3fcp+1", "0x1.1eb19e7f90b00p+1", "0x1.2b8bf12d2f8cap+0"
LANES = [
    {
        "label": "masses m, 0, m, 0, m; triples 012, 234",
        "shifts": [
            [Z, Z, Z, Z],
            ["-" + E1, X1, "-" + Y1, "-" + W1],
            ["-" + E2, X2, "-" + Y2, "-" + W2],
            ["-0x1.fbd70a3d70a3ep+2", X1, "-" + Y1, "-" + W1],
            ["-0x1.528f5c28f5c29p+3", Z, Z, Z],
        ],
        "invariants": [
            Z, H, T, "0x1.d1bedfa43fe5dp+5", "0x1.bfbedfa43fe5dp+6",
            H, Z, H, "0x1.bfbedfa43fe5fp+4", "0x1.d1bedfa43fe5dp+5",
            T, H, Z, H, T,
            "0x1.d1bedfa43fe5dp+5", "0x1.bfbedfa43fe5fp+4", H, Z, H,
            "0x1.bfbedfa43fe5dp+6", "0x1.d1bedfa43fe5dp+5", T, H, Z,
        ],
        "masses": [H, Z, H, Z, H],
        # Each shift as a A + b B.
        "combination": [(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1)],
        "other": "B",
    },
    {
        "label": "masses m, 0, m, m, 0; triples 012, 340",
        "shifts": [
            [Z, Z, Z, Z],
            ["-" + E1, X1, "-" + Y1, "-" + W1],
            ["-" + E2, X2, "-" + Y2, "-" + W2],
            [E2, X2, "-" + Y2, "-" + W2],
            [E1, X1, "-" + Y1, "-" + W1],
        ],
        "invariants": [
            Z, H, T, T, H,
            H, Z, H, "0x1.d1bedfa43fe5dp+5", "0x1.bfbedfa43fe5dp+4",
            T, H, Z, "0x1.bfbedfa43fe5dp+6", "0x1.d1bedfa43fe5dp+5",
            T, "0x1.d1bedfa43fe5dp+5", "0x1.bfbedfa43fe5dp+6", Z, H,
            H, "0x1.bfbedfa43fe5dp+4", "0x1.d1bedfa43fe5dp+5", H, Z,
        ],
        "masses": [H, Z, H, H, Z],
        # Each shift as a A + c C.
        "combination": [(0, 0), (0.5, 0), (1, 0), (0, 1), (0, 0.5)],
        "other": "C",
    },
]
MU2 = "0x1.2000000000000p+3"
RANK = 3


def value(h):
    return float.fromhex(h)


def exact(x):
    return format(Decimal(x), ".45e")


def on_threshold(lane):
    """Quadruple-precision shifts with both onia exactly on threshold."""
    a = [Decimal(value(x)) for x in lane["shifts"][2]]
    energy, spatial = a[0], a[1:]
    scale = ((energy * energy - Decimal(value(T))) / sum(x * x for x in spatial)).sqrt()
    a = [energy] + [x * scale for x in spatial]
    other = [energy] + [-x for x in a[1:]] if lane["other"] == "B" else [-energy] + a[1:]
    shifts = []
    for (ca, cb), binary64 in zip(lane["combination"], lane["shifts"]):
        q = [Decimal(ca) * x + Decimal(cb) * y for x, y in zip(a, other)]
        assert all(abs(float(x) - value(b)) <= 1e-13 * (1 + abs(value(b))) for x, b in zip(q, binary64)), (q, binary64)
        shifts.append(q)
    return shifts


def main():
    text = ""
    for k, lane in enumerate(LANES):
        n = len(lane["masses"])
        text += f"EVAL c{k} tensor 0 {n} {RANK} 0 {exact(Decimal(value(MU2)).sqrt())} 1 1e-12\n"
        for shift, mass in zip(on_threshold(lane), lane["masses"]):
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
        threshold = on_threshold(lane)
        n = len(threshold)
        # The squared differences of the quad shifts, as IREGI takes them,
        # column-major like the supplied invariants.
        square = lambda q: q[0] * q[0] - q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
        threshold_invariants = [
            exact(square([x - y for x, y in zip(threshold[i], threshold[j])])) for j in range(n) for i in range(n)
        ]
        cases.append(
            {
                "label": lane["label"],
                "threshold_shifts": [[exact(x) for x in shift] for shift in threshold],
                "threshold_invariants": threshold_invariants,
                "rank": RANK,
                "shifts": [[repr(value(x)) for x in shift] for shift in lane["shifts"]],
                "invariants": [repr(value(x)) for x in lane["invariants"]],
                "masses_squared": [repr(value(x)) for x in lane["masses"]],
                "mu_squared": repr(value(MU2)),
                "reference": [[[a.strip(), b.strip()] for a, b in monomial] for monomial in r["moments"]],
            }
        )
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-double-coulomb.json"
    out.write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
