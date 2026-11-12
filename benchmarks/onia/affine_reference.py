#!/usr/bin/env python3
"""Independent references for the degenerate quarkonium integrals by exact
denominator identities.

With p_c = p_c~ the shifts of many integrals are affinely dependent: there
are c_i, not all zero, with sum_i c_i = 0 and sum_i c_i q_i = 0. Then
sum_i c_i D_i = K with the constant K = sum_i c_i (q_i^2 - m_i^2), exactly,
for D_i = (l + q_i)^2 - m_i^2. If K != 0,

    1 / (D_0 ... D_{N-1}) = sum_i (c_i / K) / (D_0 ... D_{N-1} without D_i),

so the tensor integral (same numerator, same loop momentum, same shifts) is
the sum of its pinches with these exact rational coefficients. Applied
recursively, this writes every integral whose shifts are degenerate, but with
K != 0 at each step, as a combination of integrals whose shifts are affinely
independent (non-degenerate Gram determinants). Those are evaluated by the
TIRust direct engine at 256 bits and accepted only when quad original IREGI
(IBP mode, or PV) agrees to E/A <= 1e-9, or, where
both quad modes disagree, when COLLIER's scalar master (IR-finite C0/D0)
agrees with the 256-bit scalar moment and contradicts quad IREGI's (both quad
modes share IREGI's bundled OneLOop 3.4, whose three-mass D0 is wrong on some
boxes). The reference is the exact combination of the 256-bit digits, summed
in 60-digit decimal arithmetic.

If a step meets K = 0 (the propagators are then linearly dependent: the
Coulomb-exchange integrals, 2 D_gluon = D_c + D_c~), the identity becomes a
partial fraction that raises a power (a doubled gluon). Those dotted integrals
are tried with TIRust and quad IREGI's general API, but on these kinematics
both refuse them (TIRust: dotted massless propagators at lightlike or
on-shell legs; quad IREGI: its 32-bit factorials overflow), so the lane gets
no reference here; coulomb_check.py covers these lanes. Non-degenerate
integrals need no identity.

The coefficients depend only on the exact lattice inputs (Fractions of the
binary64 values), not on either code.

usage: affine_reference.py POINTS_JSON OUTPUT_DIR [--workers N] [--only PROCESS] [--max-points N]
writes OUTPUT_DIR/affine-PROCESS.jsonl: {"topology", "index", "status", "terms", "moments"}.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

getcontext().prec = 60
CONFIRMATION = 1e-9


def nullspace(rows, n):
    """A rational basis of {c : rows c = 0} (Gauss-Jordan)."""
    m = [list(r) for r in rows]
    pivots, r = [], 0
    for col in range(n):
        p = next((i for i in range(r, len(m)) if m[i][col] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = 1 / m[r][col]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][col] != 0:
                f = m[i][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(col)
        r += 1
        if r == len(m):
            break
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -m[i][f]
        basis.append(v)
    return basis


def square(q):
    return q[0] * q[0] - q[1] * q[1] - q[2] * q[2] - q[3] * q[3]


def decompose(props, depth=0):
    """props: tuple of (q (4 Fractions), m2 Fraction, power). Returns
    {props_sub: coefficient} of integrals with affinely independent shifts,
    or None if the recursion does not close (depth > 12)."""
    n = len(props)
    if n <= 1:
        return {props: Fraction(1)}
    rows = [[Fraction(1)] * n] + [[p[0][mu] for p in props] for mu in range(4)]
    null = nullspace(rows, n)
    if not null:
        return {props: Fraction(1)}
    if depth > 12:
        return None
    c = null[0]
    k = sum(ci * (square(q) - m2) for ci, (q, m2, _) in zip(c, props))

    def step(lower, raise_=None):
        entries = [list(p) for p in props]
        entries[lower][2] -= 1
        if raise_ is not None:
            entries[raise_][2] += 1
        return tuple(tuple(e) for e in entries if e[2] > 0)

    terms = []
    if k != 0:
        # sum_i c_i D_i = K: lower one power of each D_i, coefficient c_i / K.
        terms = [(ci / k, step(i)) for i, ci in enumerate(c) if ci != 0]
    else:
        # sum_i c_i D_i = 0 (linearly dependent propagators, e.g. the Coulomb
        # gluon 2 D_g = D_c + D_c~): D_j = -sum_{i != j} (c_i / c_j) D_i, so
        # 1 = -sum_{i != j} (c_i / c_j) D_i / D_j: lower D_i, raise D_j. The
        # raised propagator is a massless one where possible.
        j = max((i for i, ci in enumerate(c) if ci != 0), key=lambda i: (props[i][1] == 0, -props[i][2]))
        terms = [(-ci / c[j], step(i, j)) for i, ci in enumerate(c) if ci != 0 and i != j]
    out = defaultdict(Fraction)
    for coefficient, sub_props in terms:
        sub = decompose(sub_props, depth + 1)
        if sub is None:
            return None
        for key, value in sub.items():
            out[key] += coefficient * value
    return {key: v for key, v in out.items() if v != 0}


def exact_props(topology, point):
    momenta = {leg: [Fraction(float(x)) for x in v] for leg, v in point["momenta"].items()}
    m2 = Fraction(float(point["m2"]))
    q = [(Fraction(0),) * 4]
    for group in topology["groups"][:-1]:
        q.append(tuple(a + sum(momenta[leg][c] for leg in group) for c, a in enumerate(q[-1])))
    return tuple((qi, m2 if m == "H" else Fraction(0), 1) for qi, m in zip(q, topology["masses"]))


def canonical(props):
    return tuple(sorted(props))


def tirust_powers(lanes, n, rank, powers, mu):
    """TIRust onia_driver on lanes with the given propagator powers."""
    import os, subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        source, sink = Path(tmp) / "input.json", Path(tmp) / "out.jsonl"
        chunk = {"id": f"sub-N{n}", "n": n, "rank": rank, "mu": repr(mu),
                 "lanes": [{"shifts": [[repr(x) for x in sh] for sh in lane["shifts"]],
                            "masses_squared": [repr(m) for m in lane["masses_squared"]]} for lane in lanes]}
        if any(p != 1 for p in powers):
            chunk["powers"] = list(powers)
        source.write_text(json.dumps({"integrals": [chunk]}))
        out = subprocess.run([str(run.TIRUST), str(source), str(sink)], capture_output=True, text=True,
                             timeout=7200, env=dict(os.environ, TIRUST_ONIA_SYMBOLIC256="0"))
        if out.returncode != 0 or not sink.exists():
            return None, out.stderr[-2000:]
        return [json.loads(line) for line in sink.read_text().splitlines()], None


def translate(moments, a, rank):
    """Moments with shifts q_i from the moments M' of shifts q_i - a:
    l = l' - a, so M_alpha = sum_{beta <= alpha} C(alpha, beta) (-a)^(alpha - beta) M'_beta."""
    from math import comb
    monomials = run.basis(rank)
    index = {m: i for i, m in enumerate(monomials)}
    neg = [Decimal(-x.numerator) / Decimal(x.denominator) for x in a]
    out = []
    for alpha in monomials:
        slots = [[Decimal(0), Decimal(0)] for _ in range(4)]
        for beta in monomials:
            if any(b > al for b, al in zip(beta, alpha)):
                continue
            c = Decimal(1)
            for al, b, x in zip(alpha, beta, neg):
                if al > b:
                    c *= comb(al, b) * x ** (al - b)
            if c == 0:
                continue
            for slot in range(4):
                re, im = moments[index[beta]][slot]
                slots[slot][0] += c * Decimal(str(re).strip())
                slots[slot][1] += c * Decimal(str(im).strip())
        out.append([[str(x) for x in pair] for pair in slots])
    return out


def quad_general(key, rank, mu, mode):
    """Quad IREGI's general API (powers) on one integral: the first row must
    have a zero shift, so put one first or translate the loop momentum."""
    rows = list(key)
    zero = next((i for i, (q, _, _) in enumerate(rows) if all(x == 0 for x in q)), None)
    a = (Fraction(0),) * 4
    if zero is not None:
        rows.insert(0, rows.pop(zero))
    else:
        a = rows[0][0]
        rows = [(tuple(x - y for x, y in zip(q, a)), m2, nu) for q, m2, nu in rows]
    n = len(rows)
    text = f"EVAL g general {mode} {n} {rank} 0 {run.exact(mu)} 1 1e-12\n" + "".join(
        " ".join(run.exact(float(x)) for x in q) + f" {run.exact(float(m2))} 0.0 {nu}\n" for q, m2, nu in rows)
    res, _ = run.results([str(run.QUAD)], text + "QUIT\n")
    moments = res[0].get("moments") if res else None
    if moments is None:
        return None
    return translate(moments, a, rank) if zero is None else moments


def evaluate(group, mu):
    """group: list of (key, rank), all with the same (n, rank, powers); key is
    a canonical tuple of (q, m2, power). Returns {(key, rank): (status,
    moments strings or None, check)}."""
    n, rank = len(group[0][0]), group[0][1]
    powers = tuple(nu for _, _, nu in group[0][0])
    dotted = any(nu != 1 for nu in powers)
    lanes = [{"shifts": [[float(x) for x in q] for q, _, _ in key], "masses_squared": [float(m2) for _, m2, _ in key]}
             for key, _ in group]
    for lane, (key, _) in zip(lanes, group):
        assert all(Fraction(f) == x for row, (q, _, _) in zip(lane["shifts"], key) for f, x in zip(row, q))
    topology = {"id": f"sub-N{n}", "n": n, "rank": rank}
    t, error = tirust_powers(lanes, n, rank, powers, mu)
    out = {}
    for k, (key, _) in enumerate(group):
        d256 = t[k]["direct256"]["moments"] if t else None
        checks = {}
        for mode, name in ((0, "quad IBP"), (1, "quad PV")):
            if dotted:
                q = quad_general(key, rank, mu, mode)
            else:
                q = run.quad([lanes[k]], topology, mu, mode)[0]
            checks[name] = run.exact_error(d256, q, rank, n, mu)
            if checks[name] is not None and checks[name] <= CONFIRMATION:
                out[(key, rank)] = (f"confirmed by {name}" + (" (dotted)" if dotted else ""), d256, checks[name])
                break
        else:
            # Both quad modes disagree: COLLIER's scalar master arbitrates
            # (IR-finite scalars only; both quad modes share IREGI's bundled
            # OneLOop 3.4, whose D0 is wrong on some boxes, README).
            binary = None
            if not dotted:
                q = run.quad([lanes[k]], topology, mu, 0)[0]
                binary = run.scalar_arbitration(lanes[k], d256, q)
                if binary and binary["collier_vs_direct256"] <= CONFIRMATION and binary["collier_vs_quad"] > 1e-6:
                    out[(key, rank)] = ("direct256; quad IREGI's scalar master contradicted by COLLIER", d256,
                                        binary["collier_vs_direct256"])
            if (key, rank) not in out:
                out[(key, rank)] = ("unconfirmed", None,
                                    {"direct256_status": (t[k]["direct256"]["status"] if t else error)[:300],
                                     "powers": powers, **checks, "collier": binary})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.points.read_text())
    mu = float(data["mu"])
    for process in data["processes"]:
        if args.only and process["id"] != args.only:
            continue
        points = process["points"][: args.max_points]
        plans, needed = [], {}
        for topology in process["topologies"]:
            for point in points:
                props = exact_props(topology, point)
                rows = [[Fraction(1)] * len(props)] + [[p[0][mu_] for p in props] for mu_ in range(4)]
                if not nullspace(rows, len(props)):
                    continue                      # non-degenerate: no identity needed
                terms = decompose(props)
                plans.append((topology, point, terms))
                if terms is None:
                    continue
                for key in terms:
                    ck = canonical(key)
                    needed[(ck, topology["rank"])] = True
        print(f"{process['id']}: {len(plans)} degenerate lanes, {len(needed)} distinct sub-integrals",
              file=sys.stderr, flush=True)
        groups = defaultdict(list)
        for ck, rank in needed:
            groups[(len(ck), rank, tuple(nu for _, _, nu in ck))].append((ck, rank))
        values = {}
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = []
            for (n, rank, _), items in groups.items():
                for start in range(0, len(items), args.chunk):
                    jobs.append(pool.submit(evaluate, items[start:start + args.chunk], mu))
            for done, future in enumerate(as_completed(jobs), 1):
                for key, value in future.result().items():
                    values[key] = value
                if done % 20 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} sub-integral chunks", file=sys.stderr, flush=True)
        with (args.output / f"affine-{process['id']}.jsonl").open("w") as sink:
            for topology, point, terms in plans:
                record = {"topology": topology["id"], "index": point["index"], "frame": point["frame"]}
                if terms is None:
                    record.update(status="identity recursion did not close", terms=None, moments=None)
                    sink.write(json.dumps(record) + "\n")
                    continue
                size = len(run.basis(topology["rank"]))
                total = [[[Decimal(0), Decimal(0)] for _ in range(4)] for _ in range(size)]
                status, worst, hows = "ok", 0.0, set()
                record["terms"] = [{"n": len(key), "powers": [nu for _, _, nu in canonical(key)], "coefficient": str(c)}
                                   for key, c in terms.items()]
                for key, coefficient in terms.items():
                    how, moments, check = values[(canonical(key), topology["rank"])]
                    if moments is None:
                        status = f"sub-integral N{len(key)} unconfirmed: {check}"
                        break
                    worst = max(worst, check)
                    hows.add(how)
                    cr = Decimal(coefficient.numerator) / Decimal(coefficient.denominator)
                    for m in range(size):
                        for slot in range(4):
                            total[m][slot][0] += cr * Decimal(moments[m][slot][0])
                            total[m][slot][1] += cr * Decimal(moments[m][slot][1])
                record["status"] = status
                record["largest_sub_check"] = worst
                record["sub_confirmations"] = sorted(hows)
                record["moments"] = ([[[str(x) for x in pair] for pair in slots] for slots in total]
                                     if status == "ok" else None)
                sink.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
