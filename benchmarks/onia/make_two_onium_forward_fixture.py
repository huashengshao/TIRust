"""Fixture for g g > chi_c0 chi_c0 at a forward angle: flat pentagons whose
dotted gluon sits in a collinear pair.

Writes crates/iregi-rust/tests/fixtures/onia-two-onium-forward.json from
onia_loop's records of the raised-power calls TIRust 08cca00 refused
(docs/benchmarkfolder/research/two_pwave/data/refused_gg_chic0chic0_08cca00.txt,
onia_loop ffa86c9ec7; MadLoop's binary64 inputs: shifts, supplied invariants,
masses and mu^2). The points are seed 1 of its qp_check.py / fixtures.py:
point 1 (cos theta = -0.985844, 9.7 degrees from the beam) at 20, 50 and
200 GeV and point 3 (cos theta = -0.603568) at 20 GeV, rank 5 and 6, one or
two doubled lines. Each case's pentagon (or a pentagon of its hexagon) has a
massless doubled line in a collinear pair, and with p_Q = p_Qbar for both
onia in a 2 -> 2 process its external points span three dimensions.

Every shift is a p1 + b p2 + c p_c with a, b, c multiples of 1/4 of the
gluons p1 = (E, 0, 0, E), p2 = (E, 0, 0, -E) and one constituent
p_c = (E/2, |P| u/2), |P|^2 = E^2 - 4 m_c^2, with E = sqrt(mu^2)/2 and m_c^2
the record's binary64 values taken as exact and u the direction of the
record's constituent with energy +E/2. The fixture stores E, m_c^2, u and
the integers 4a, 4b, 4c. The reference is the scalar integral (the rank-0
component) in IREGI's native convention, on those kinematics in 200-digit
decimals: exact partial fractions of the linearly dependent subsets
(make_two_dotted_fixture.reduce) down to integrals without one, each from
quad original IREGI's scalar API with powers (ONIA_QUAD). IREGI's direct
value is kept as `iregi_direct`; on some hexagons at 200 GeV it is off by up
to O(1), for example by 3e-2 on call 134.

    make_two_onium_forward_fixture.py RECORDS
"""
import json
import math
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext

from make_two_dotted_fixture import reduce
from make_two_onium_gg_fixture import QUAD, ROOT, fit, records, square

getcontext().prec = 200

# Calls (numbered from 1 in the file), each refused by 08cca00.
CASES = [
    (1, "20 GeV, point 1: pentagon MM000, gluon 2 doubled, rank 5"),
    (4, "20 GeV, point 1: hexagon MM00M0, gluon 2 doubled, rank 5"),
    (60, "20 GeV, point 3: hexagon M00M0M, gluons 2 and 3 doubled, rank 6"),
    (62, "50 GeV, point 1: pentagon MM000, gluon 2 doubled, rank 5"),
    (95, "50 GeV, point 1: hexagon M0M00M, lines 1 and 4 doubled, rank 6"),
    (119, "200 GeV, point 1: pentagon MM000, gluons 2 and 4 doubled, rank 6"),
    (134, "200 GeV, point 1: hexagon MM0M00, gluons 2 and 4 doubled, rank 6"),
    (164, "200 GeV, point 1: hexagon M00M0M, gluons 2 and 3 doubled, rank 6"),
]


def kinematics(call):
    """E, m_c^2, u (binary64) and the exact shifts, masses^2 and invariants."""
    energy = math.sqrt(call["mu2"]) / 2
    m2 = max(call["masses"])
    momentum = math.sqrt(energy * energy - 4 * m2)
    constituent = next(
        s
        for s in call["shifts"]
        if abs(s[0] - energy / 2) <= 1e-12 * energy
        and any(s[1:3])
        and abs(math.sqrt(sum(x * x for x in s[1:])) - momentum / 2) <= 1e-12 * energy
    )
    direction = constituent[1:]
    E, M2 = Decimal(energy), Decimal(m2)
    P = (E * E - 4 * M2).sqrt()
    norm = sum(Decimal(x) ** 2 for x in direction).sqrt()
    exact_basis = [[E, 0, 0, E], [E, 0, 0, -E], [E / 2] + [P / 2 * Decimal(x) / norm for x in direction]]
    exact_basis = [[Decimal(x) for x in v] for v in exact_basis]
    float_basis = [[float(x) for x in v] for v in exact_basis]
    coefficients = [fit(s, float_basis) for s in call["shifts"]]
    shifts = [[sum(Decimal(k) * exact_basis[j][a] for j, k in enumerate(q)) / 4 for a in range(4)] for q in coefficients]
    masses = [Decimal(m) for m in call["masses"]]
    n = len(shifts)
    invariants = []
    for i in range(n):
        row = []
        for j in range(n):
            value = square([a - b for a, b in zip(shifts[i], shifts[j])])
            supplied = call["invariants"][i][j]
            assert abs(float(value) - supplied) <= 1e-12 * max(1.0, abs(supplied)), (call["call"], i, j)
            # MadLoop's lightlike and on-shell entries are exact.
            for exact in [Decimal(0)] + masses:
                if supplied == float(exact) and abs(value - exact) <= Decimal("1e-150") * max(1, abs(exact)):
                    value = exact
            row.append(value)
        invariants.append(row)
    return (energy, m2, direction), coefficients, shifts, masses, invariants


def main():
    calls = records(sys.argv[1])

    def fmt(x):
        return format(x, ".170e")

    cases = []
    for number, label in CASES:
        call = calls[number]
        basis, coefficients, shifts, masses, invariants = kinematics(call)
        lines = tuple(range(len(masses)))
        powers = tuple(call["powers"])
        combo = reduce(lines, powers, shifts, masses, {})
        leaves = list(combo) + [(lines, powers)]
        mu = Decimal(call["mu2"]).sqrt()
        text = ""
        for k, (ls, pw) in enumerate(leaves):
            text += f"EVAL r{k} scalar 0 {len(ls)} 0 0 {fmt(mu)} 1 1e-12\n"
            for line, power in zip(ls, pw):
                text += " ".join(fmt(x) for x in shifts[line]) + f" {fmt(masses[line])} 0.0 {power}\n"
        cases.append((call, label, basis, coefficients, combo, leaves, text))

    def run(case):
        out = subprocess.run([str(QUAD)], input=case[6] + "QUIT\n", capture_output=True, text=True, timeout=3600,
                             check=True)
        values = {}
        for line in out.stdout.splitlines():
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                values[case[5][int(r["id"][1:])]] = [(Decimal(a.strip()), Decimal(b.strip())) for a, b in r["moments"][0]]
        return values

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, cases))
    fixture = []
    for (call, label, basis, coefficients, combo, leaves, _), values in zip(cases, results):
        total = [[Decimal(0), Decimal(0)] for _ in range(4)]
        for leaf, weight in combo.items():
            for s in range(4):
                total[s][0] += weight * values[leaf][s][0]
                total[s][1] += weight * values[leaf][s][1]
        direct = values[leaves[-1]]
        scale = max(abs(complex(float(a), float(b))) for a, b in total)
        gap = max(abs(complex(float(a - x), float(b - y))) for (a, b), (x, y) in zip(total, direct)) / scale
        print(f"call {call['call']}: {len(combo)} regular integrals; IREGI direct {gap:.1e} from them")
        energy, m2, direction = basis
        fixture.append(
            {
                "call": call["call"],
                "label": label,
                "rank": call["rank"],
                "powers": call["powers"],
                "basis": {"energy": repr(energy), "constituent_mass_squared": repr(m2),
                          "direction": [repr(x) for x in direction]},
                "coefficients": coefficients,
                "shifts": [[repr(x) for x in s] for s in call["shifts"]],
                "invariants": [repr(x) for row in call["invariants"] for x in row],
                "masses_squared": [repr(x) for x in call["masses"]],
                "mu_squared": repr(call["mu2"]),
                "regular_integrals": len(combo),
                "reference": [[format(a, ".36e"), format(b, ".36e")] for a, b in total],
                "iregi_direct": [[format(a, ".36e"), format(b, ".36e")] for a, b in direct],
            }
        )
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-two-onium-forward.json"
    lines = ",\n".join(json.dumps(case) for case in fixture)
    out.write_text('{"cases": [\n' + lines + "\n]}\n")
    print(f"wrote {out} ({len(fixture)} cases)")


if __name__ == "__main__":
    main()
