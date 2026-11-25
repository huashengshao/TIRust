"""Fixture for integrals with two doubled lines, and with a doubled line
outside a Coulomb triple: two P-wave bound states.

Writes crates/iregi-rust/tests/fixtures/onia-two-dotted.json from
onia_loop's TIRust fixtures of its two-P-wave outputs
(docs/benchmarkfolder/research/two_pwave/data, onia_loop 809886b9f0). Its
cases are MadLoop's binary64 lanes at v = 0 (supplied invariants, masses
and mu^2, the primary pass), one per (propagators, rank, powers, massive
lines) among the calls TIRust refused before: two doubled lines, or a
status refusal.

The reference is the scalar integral (the rank-0 component of the array),
in IREGI's native convention:
- Exact partial-fraction identities of the lane's linearly dependent
  subsets (affine, sum w_i D_i = 1, and linear, sum c_i D_i = 0, found in
  60-digit decimals; a constant within 1e-12 of its terms counts as zero,
  as TIRust decides at the inputs' rounding), applied recursively down to
  integrals without such a subset. On a lane without one, that is the lane
  itself.
- Those integrals from original IREGI's quad scalar API with powers (the
  reference build run.py uses, ONIA_QUAD). IREGI's direct value of the lane
  is kept as `iregi_direct`: it is wrong on a box whose four lines are
  linearly dependent, with a line at power 3 or two lines doubled
  (benchmarks/onia/mre/iregi_dependent_box_powers.py). This build's general
  tensor API with powers stops on these lanes ("i > 12, please take long
  type (KIND=LINT) integer"), so the tensor components have no reference
  here.
- IREGI takes its invariants from the shifts, which put the heavy pairs at
  p^2 = 4 m^2 off threshold by their rounding (up to 2e-15 relative); with
  two doubled lines in a Coulomb triple that moves IREGI's value by orders
  of magnitude. The shifts are therefore taken in decimal, with each such
  pair's relative momentum rescaled about its midpoint (where the Coulomb
  gluon sits), or about the heavy line two pairs share, to put it exactly
  on threshold. TIRust keeps MadLoop's supplied invariants, which agree with
  those of the rescaled shifts to their rounding.

    make_two_dotted_fixture.py FIXTURES_DIR
"""
import gzip
import itertools
import json
import os
import pathlib
import subprocess
import sys
from decimal import Decimal, getcontext

getcontext().prec = 60

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUAD = pathlib.Path(
    os.environ.get("ONIA_QUAD", ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad")
)
FILES = [("e+ e- > chic0 hc", "fixtures_ee_chic0_hc.json.gz"), ("g g > chic0 chic0", "fixtures_gg_chic0_chic0_refused.json.gz")]


def value(h):
    return float.fromhex(h)


def exact(x):
    return format(Decimal(x), ".40e")


def square(p):
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]


def on_threshold(shifts, masses):
    """Decimal shifts with every equal-mass heavy pair within 1e-12 of
    p^2 = 4 m^2 exactly on threshold: its relative momentum rescaled about
    its midpoint, or about the line it shares with another such pair (two
    Coulomb triples sharing a heavy line), with the massless lines at its
    midpoint (Coulomb gluons) kept there. Returns the shifts and the pairs."""
    p = [[Decimal(x) for x in q] for q in shifts]
    n = len(p)
    pairs = []
    for i in range(n):
        for k in range(i + 1, n):
            m2 = Decimal(masses[i])
            if m2 != 0 and masses[k] == masses[i]:
                K = [a - b for a, b in zip(p[i], p[k])]
                if abs(square(K) - 4 * m2) <= Decimal("1e-12") * 4 * m2:
                    pairs.append((i, k))
    tolerance = Decimal("1e-12") * max(abs(x) for q in p for x in q)
    for i, k in pairs:
        m2 = Decimal(masses[i])
        gluons = [
            g
            for g in range(n)
            if masses[g] == 0 and all(abs((a + b) / 2 - c) <= tolerance for a, b, c in zip(p[i], p[k], p[g]))
        ]
        shared = [line for line in (i, k) if sum(line in pair for pair in pairs) > 1]
        assert len(shared) <= 1, "a pair whose both lines are shared"
        K = [a - b for a, b in zip(p[i], p[k])]
        scale = (4 * m2 / square(K)).sqrt()
        if shared:
            anchor, other = (i, k) if shared[0] == i else (k, i)
            p[other] = [a + scale * (b - a) for a, b in zip(p[anchor], p[other])]
        else:
            middle = [(a + b) / 2 for a, b in zip(p[i], p[k])]
            p[i] = [c + scale * x / 2 for c, x in zip(middle, K)]
            p[k] = [c - scale * x / 2 for c, x in zip(middle, K)]
        for g in gluons:
            p[g] = [(a + b) / 2 for a, b in zip(p[i], p[k])]
    for i, k in pairs:
        assert abs(square([a - b for a, b in zip(p[i], p[k])]) - 4 * Decimal(masses[i])) < Decimal("1e-50")
    return p, pairs


TOL = Decimal("1e-40")

def null_vector(rows, ncols):
    """A null vector of the matrix (list of rows) if its rank is ncols - 1, else None."""
    m = [list(r) for r in rows]
    scale = max((abs(x) for r in m for x in r), default=Decimal(1)) or Decimal(1)
    pivots, r = [], 0
    for c in range(ncols):
        best = max(range(r, len(m)), key=lambda i: abs(m[i][c]), default=None)
        if best is None or abs(m[best][c]) <= TOL * scale:
            continue
        m[r], m[best] = m[best], m[r]
        piv = m[r][c]
        m[r] = [x / piv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c); r += 1
        if r == len(m):
            break
    free = [c for c in range(ncols) if c not in pivots]
    if len(free) != 1:
        return None
    v = [Decimal(0)] * ncols
    v[free[0]] = Decimal(1)
    for row, c in zip(m, pivots):
        v[c] = -row[free[0]]
    return v

def relation(lines, p, m2):
    """A minimal linearly dependent subset of `lines`: ('affine', R, w) with
    sum w_i D_i = 1, or ('linear', R, c) with sum c_i D_i = 0; None if none."""
    for size in range(2, min(len(lines), 6) + 1):
        for R in itertools.combinations(lines, size):
            rows = [[Decimal(1)] * size] + [[p[i][mu] for i in R] for mu in range(4)]
            c = null_vector(rows, size)
            if c is None or any(abs(x) <= TOL for x in c):
                continue
            kappa = sum(ci * (square(p[i]) - m2[i]) for ci, i in zip(c, R))
            size_scale = sum(abs(ci) * (abs(square(p[i])) + abs(m2[i])) for ci, i in zip(c, R))
            # Linear to the inputs' rounding (binary64 shifts), as TIRust decides:
            # the reference is the v = 0 value.
            if abs(kappa) <= Decimal("1e-12") * size_scale:
                return ("linear", R, c)
            return ("affine", R, [ci / kappa for ci in c])
    return None

def reduce(lines, powers, p, m2, memo):
    """Linear combination {(lines, powers): weight} of regular integrals."""
    key = (lines, powers)
    if key in memo:
        return memo[key]
    keep = [k for k, pw in enumerate(powers) if pw > 0]
    if len(keep) < len(lines):
        out = reduce(tuple(lines[k] for k in keep), tuple(powers[k] for k in keep), p, m2, memo)
        memo[key] = out
        return out
    rel = relation(list(lines), p, m2)
    out = {}
    def add(terms, weight):
        for k, v in terms.items():
            out[k] = out.get(k, Decimal(0)) + weight * v
    if rel is None:
        out = {key: Decimal(1)}
    elif rel[0] == "affine":
        _, R, w = rel
        for wi, i in zip(w, R):
            k = lines.index(i)
            add(reduce(lines, powers[:k] + (powers[k] - 1,) + powers[k + 1:], p, m2, memo), wi)
    else:
        _, R, c = rel
        # Raise the member of largest power (then the first): the others' powers drop.
        j = max(R, key=lambda i: (powers[lines.index(i)], -lines.index(i)))
        cj = c[R.index(j)]
        kj = lines.index(j)
        for ci, i in zip(c, R):
            if i == j:
                continue
            k = lines.index(i)
            new = list(powers); new[k] -= 1; new[kj] += 1
            add(reduce(lines, tuple(new), p, m2, memo), -ci / cj)
    memo[key] = out
    return out


def lanes(directory):
    seen = set()
    for process, name in FILES:
        with gzip.open(pathlib.Path(directory) / name, "rt") as f:
            data = json.load(f)
        for record in data["records"]:
            if record["ctmode"] != 1:
                continue
            for call in record["calls"]:
                if call["supported"] and call.get("status", -1) == 0:
                    continue
                masses = [value(m[0]) for m in call["masses2"]]
                assert all(value(m[1]) == 0.0 for m in call["masses2"])
                pattern = "".join("0" if m == 0.0 else "M" for m in masses)
                key = (call["n"], call["rank"], tuple(call["powers"]), pattern)
                if key in seen:
                    continue
                seen.add(key)
                shifts = [[value(x) for x in shift] for shift in call["shifts"]]
                label = f"{process}, N{call['n']} rank {call['rank']} powers {call['powers']} masses {pattern}"
                yield {
                    "label": label,
                    "powers": call["powers"],
                    "rank": call["rank"],
                    "shifts": shifts,
                    "invariants": [value(x) for row in call["invariants"] for x in row],
                    "masses": masses,
                    "mu2": value(call["mu2"]),
                }


def main():
    cases = list(lanes(sys.argv[1]))
    plans, requests = [], []
    for lane in cases:
        shifts, moved = on_threshold(lane["shifts"], lane["masses"])
        lane["threshold_pairs"] = moved
        m2 = [Decimal(x) for x in lane["masses"]]
        lines = tuple(range(len(m2)))
        combo = reduce(lines, tuple(lane["powers"]), shifts, m2, {})
        plans.append((shifts, m2, combo))
        for leaf in list(combo) + [(lines, tuple(lane["powers"]))]:
            requests.append((len(plans) - 1, leaf))
    text = ""
    for k, (c, (lines, powers)) in enumerate(requests):
        shifts, m2, _ = plans[c]
        mu = exact(Decimal(cases[c]["mu2"]).sqrt())
        text += f"EVAL r{k} scalar 0 {len(lines)} 0 0 {mu} 1 1e-12\n"
        for i, power in zip(lines, powers):
            text += " ".join(exact(x) for x in shifts[i]) + f" {exact(m2[i])} 0.0 {power}\n"
    run = subprocess.run([str(QUAD)], input=text + "QUIT\n", capture_output=True, text=True, timeout=7200, check=True)
    values = {}
    for line in run.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            values[requests[int(r["id"][1:])]] = [(Decimal(a.strip()), Decimal(b.strip())) for a, b in r["moments"][0]]
    out_cases = []
    largest = 0.0
    for c, lane in enumerate(cases):
        shifts, m2, combo = plans[c]
        total = [[Decimal(0), Decimal(0)] for _ in range(4)]
        for leaf, weight in combo.items():
            for s in range(4):
                total[s][0] += weight * values[(c, leaf)][s][0]
                total[s][1] += weight * values[(c, leaf)][s][1]
        direct = values[(c, (tuple(range(len(m2))), tuple(lane["powers"])))]
        scale = max(abs(complex(float(a), float(b))) for a, b in total)
        gap = max(abs(complex(float(a - x), float(b - y))) for (a, b), (x, y) in zip(total, direct)) / scale
        largest = max(largest, gap)
        if gap > 1e-12:
            print(f"{lane['label']}: IREGI direct {gap:.1e} from the reduced reference ({len(combo)} regular integrals)")
        out_cases.append(
            {
                "label": lane["label"],
                "powers": lane["powers"],
                "rank": lane["rank"],
                "threshold_pairs": lane["threshold_pairs"],
                "regular_integrals": len(combo),
                "shifts": [[repr(x) for x in shift] for shift in lane["shifts"]],
                "invariants": [repr(x) for x in lane["invariants"]],
                "masses_squared": [repr(x) for x in lane["masses"]],
                "mu_squared": repr(lane["mu2"]),
                "reference": [[[format(a, ".36e"), format(b, ".36e")] for a, b in total]],
                "iregi_direct": [[[format(a, ".36e"), format(b, ".36e")] for a, b in direct]],
            }
        )
    out = ROOT / "crates/iregi-rust/tests/fixtures/onia-two-dotted.json"
    out.write_text(json.dumps({"cases": out_cases}, indent=1) + "\n")
    print(f"wrote {out} ({len(out_cases)} cases; IREGI direct at most {largest:.1e} from the reduced references)")


if __name__ == "__main__":
    main()
