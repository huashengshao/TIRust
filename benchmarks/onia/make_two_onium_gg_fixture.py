"""Fixture for the unit-power integrals of g g > chi_c0 chi_c0 that need a
heavy raised line or a flat hexagon.

Writes crates/iregi-rust/tests/fixtures/onia-two-onium-gg.json from
onia_loop's records of the calls TIRust refused at sqrt(s) = 20 GeV, point 0
(docs/benchmarkfolder/research/two_pwave/data/refused_gg_chic0chic0_d5442e6.txt,
onia_loop e9387b298b; MadLoop's binary64 inputs: supplied invariants, masses
and mu^2). Its cases are 15 of the 87 unit-power calls, rank 4, covering
every pattern of linearly dependent subsets among them:
- boxes of a closed c, b or t loop, offsets 0, p1, (p1 - p2)/2, (p1 + p2)/2:
  D_0 + D_1 = D_2 + D_3 with equal masses and no massless member;
- pentagons with that set of four heavy lines and one gluon;
- hexagons. With p_Q = p_Qbar for both onia in a 2 -> 2 process the external
  points span three dimensions, so no six-point coordinate system exists;
  their dependent subsets are Coulomb triples, the four heavy lines above,
  and affinely dependent subsets with a constant.

The reference is quad original IREGI's array (IBP mode, which reports every
case stable; ML5 components in IREGI's native convention) from the quad
reference build that run.py uses (ONIA_QUAD), on exact kinematics. Every
shift of these calls is a p1 + b p2 + c p_c, with a, b, c quarters, of the
gluons p1 = (E, 0, 0, E), p2 = (E, 0, 0, -E) and one onium's constituent
p_c = (E/2, |P| u/2), |P|^2 = E^2 - 4 m_c^2 (the other's is
(p1 + p2)/2 - p_c), with E and m_c^2 the records' binary64 values taken as
exact and u the records' unit direction. The fixture stores a, b, c; the
reference evaluates the shifts in 200-digit decimals, so its gluons are
lightlike, its quarks on shell and its points flat to that precision, and
the test rebuilds the same kinematics at its working precision. MadLoop's
binary64 shifts and supplied invariants (also stored) agree with them to
5e-15; on some hexagons that moves the integral by up to 8e-9.

    make_two_onium_gg_fixture.py RECORDS
"""
import json
import os
import pathlib
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext

getcontext().prec = 200

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
# (call, label): one call per pattern of minimal linearly dependent subsets
# (sum c_i D_i = 0) and affinely dependent subsets with a constant
# (sum c_i D_i = K != 0); masses in line order, M heavy, 0 massless.
CASES = [
    (95, "box MMMM, closed c loop: {0,1,2,3} dependent"),
    (96, "box MMMM, closed b loop: {0,1,2,3} dependent"),
    (97, "box MMMM, closed t loop: {0,1,2,3} dependent"),
    (67, "pentagon MM0MM: {0,1,3,4} dependent"),
    (68, "pentagon MM0MM, the other onium: {0,1,3,4} dependent"),
    (1, "hexagon 000M0M: Coulomb triple {3,4,5}; constants {0,2,4}, {0,2,3,5}"),
    (2, "hexagon 000M0M: {0,2,3,5} dependent; constants {2,3,4}, {0,4,5}"),
    (25, "hexagon 00M00M: {0,1,3,4} dependent; constants {0,1,2,5}, {2,3,4,5}"),
    (29, "hexagon 00M00M: all six dependent; constants {1,2,3}, {0,4,5}"),
    (48, "hexagon 00MM0M: Coulomb triple {3,4,5}; constants {0,1,2,3,4}, {0,1,2,5}"),
    (51, "hexagon MM00M0: Coulomb triple {0,4,5}; constants {1,2,3,4}, {0,1,2,3,5}"),
    (65, "hexagon MM0M00: Coulomb triple {1,2,3}; constants {0,1,2,4,5}, {0,3,4,5}"),
    (71, "hexagon MM0MM0: Coulomb triples {1,2,3}, {0,4,5}; no constant"),
    (72, "hexagon MMM0M0: {2,3,4}, {0,2,3,5}, {0,4,5} dependent; no constant"),
    (77, "hexagon MM0MM0: four heavy lines {0,1,3,4} dependent; constants {0,1,2,5}, {2,3,4,5}"),
]


def struct_f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def records(path):
    """onia_loop's call records, numbered from 1 in file order: CALL pass n
    rank ... powers, MU2, M (masses), I (invariants, row by row), S
    (shifts), RES, END."""
    calls, call = {}, None
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "CALL":
            call = {"call": len(calls) + 1, "pass": int(f[1]), "n": int(f[2]), "rank": int(f[3]), "powers": [int(x) for x in f[6:]],
                    "masses": [], "invariants": [], "shifts": []}
        elif f[0] == "MU2":
            call["mu2"] = struct_f64(int(f[1], 16))
        elif f[0] == "M":
            call["masses"].append(struct_f64(int(f[2], 16)))
            assert int(f[3], 16) == 0, "complex mass"
        elif f[0] == "I":
            call["invariants"].append([struct_f64(int(x, 16)) for x in f[2:]])
        elif f[0] == "S":
            call["shifts"].append([struct_f64(int(x, 16)) for x in f[2:]])
        elif f[0] == "END":
            calls[call["call"]] = call
    return calls


def square(p):
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]


def fit(shift, basis):
    """a, b, c (quarters, as integers 4a, 4b, 4c) of a shift in the basis."""
    rows = [[basis[k][a] for k in range(3)] for a in range(4)]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    atb = [sum(r[i] * y for r, y in zip(rows, shift)) for i in range(3)]

    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det(ata)
    out = []
    for k in range(3):
        m = [row[:] for row in ata]
        for i in range(3):
            m[i][k] = atb[i]
        out.append(round(4 * det(m) / d))
    approx = [sum(out[k] * basis[k][a] for k in range(3)) / 4 for a in range(4)]
    scale = max(1.0, max(abs(x) for x in shift))
    assert max(abs(x - y) for x, y in zip(approx, shift)) <= 1e-13 * scale, (shift, out)
    return out


def main():
    calls = records(sys.argv[1])
    pentagon = calls[67]
    # E, m_c^2 and u from the records: the pentagon's k_1 - k_2 is p_c.
    p_c = [x - y for x, y in zip(pentagon["shifts"][1], pentagon["shifts"][2])]
    energy, m2 = pentagon["shifts"][1][0], pentagon["masses"][0]
    assert abs(p_c[0] - energy / 2) < 1e-14 * energy and abs(square(p_c) - m2) < 1e-12
    E, M2 = Decimal(energy), Decimal(m2)
    norm = sum(Decimal(x) ** 2 for x in p_c[1:]).sqrt()
    P = (E * E - 4 * M2).sqrt()
    exact_basis = [[E, 0, 0, E], [E, 0, 0, -E], [E / 2] + [P / 2 * Decimal(x) / norm for x in p_c[1:]]]
    exact_basis = [[Decimal(x) for x in v] for v in exact_basis]
    float_basis = [[float(x) for x in v] for v in exact_basis]

    def fmt(x):
        return format(x, ".170e")

    cases = []
    for number, label in CASES:
        c = calls[number]
        assert all(p == 1 for p in c["powers"]) and c["rank"] == 4
        coefficients = [fit(s, float_basis) for s in c["shifts"]]
        shifts = [[sum(Decimal(k) * exact_basis[j][a] for j, k in enumerate(q)) / 4 for a in range(4)]
                  for q in coefficients]
        mu = Decimal(c["mu2"]).sqrt()
        text = f"EVAL c{number} tensor 0 {c['n']} {c['rank']} 0 {fmt(mu)} 1 1e-12\n"
        for s, m in zip(shifts, c["masses"]):
            text += " ".join(fmt(x) for x in s) + f" {fmt(Decimal(m))} 0.0 1\n"
        cases.append((c, label, coefficients, text))

    def run(case):
        out = subprocess.run([str(QUAD)], input=case[3] + "QUIT\n", capture_output=True, text=True, timeout=3600,
                             check=True)
        return next(json.loads(line[7:]) for line in out.stdout.splitlines() if line.startswith("RESULT "))

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, cases))
    fixture = []
    for (c, label, coefficients, _), r in zip(cases, results):
        assert r["stable"], f"quad IREGI reports call {c['call']} unstable"
        fixture.append(
            {
                "call": c["call"],
                "label": label,
                "rank": c["rank"],
                "shifts": [[repr(x) for x in s] for s in c["shifts"]],
                "invariants": [repr(x) for row in c["invariants"] for x in row],
                "masses_squared": [repr(x) for x in c["masses"]],
                "mu_squared": repr(c["mu2"]),
                "coefficients": coefficients,
                "reference": [[[repr(float(a)), repr(float(b))] for a, b in monomial] for monomial in r["moments"]],
            }
        )
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-two-onium-gg.json"
    basis = {"energy": repr(energy), "constituent_mass_squared": repr(m2), "direction": [repr(x) for x in p_c[1:]]}
    lines = ",\n".join(json.dumps(case) for case in fixture)
    out.write_text('{"basis": ' + json.dumps(basis) + ',\n "cases": [\n' + lines + "\n]}\n")
    print(f"wrote {out} ({len(fixture)} cases)")


if __name__ == "__main__":
    main()
