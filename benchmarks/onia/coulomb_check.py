#!/usr/bin/env python3
"""Consistency check of original IREGI on the Coulomb-exchange integrals.

For these integrals (a gluon propagator between the c and c~ legs, p_c =
p_c~) the propagators are linearly dependent, 2 D_g = D_c + D_c~, so

    1 / (D_c D_g D_c~ ...) = (1/2) [1 / (D_c D_g^2 ...) + 1 / (D_g^2 D_c~ ...)]

(affine_reference.decompose applies this, and the constant-one identity to
what remains, exactly). TIRust refuses these integrals (typed), and its direct
engine also refuses the doubled-gluon integrals the identity produces
(lightlike or on-shell dotted massless propagators); quad IREGI's general API
overflows its 32-bit factorials on them. So the only available check is
internal to IREGI: its tensor result (IBP and PV modes, which use its
hand-coded Coulomb C0 at v = 0) against the exact combination of the dotted
integrals from its general API (binary64, with the factorial repair), a
different code path. The scalar Coulomb C0 itself is also compared with the
analytic v = 0 result, C0 = -(m^2)^(-1-eps) Gamma(1+eps) / (2 eps (1 + 2 eps)).

usage: coulomb_check.py POINTS_JSON OUTPUT_JSON [--max-points N]
"""
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import affine_reference as ar  # noqa: E402
import run  # noqa: E402


def iregi_general(key, rank, mu):
    """Binary64 IREGI general API (powers), first row with a zero shift; the
    moments are translated back when no shift is zero."""
    rows = list(key)
    zero = next((i for i, (q, _, _) in enumerate(rows) if all(x == 0 for x in q)), None)
    a = (Fraction(0),) * 4
    if zero is not None:
        rows.insert(0, rows.pop(zero))
    else:
        a = rows[0][0]
        rows = [(tuple(x - y for x, y in zip(q, a)), m2, nu) for q, m2, nu in rows]
    n = len(rows)
    head = f"BENCH s general 0 {n} {rank} 0 {mu!r} 1 0 1 1\n"
    text = head + "".join(" ".join(repr(float(x)) for x in q) + f" ({float(m2)!r},0.0) {nu}\n" for q, m2, nu in rows)
    res, _ = run.results([str(run.IREGI)], text + "QUIT\n")
    lane = res[0]["lanes"][0] if res else None
    if not lane or lane.get("status") != "success":
        return None
    moments = lane["moments"]
    return ar.translate(moments, a, rank) if zero is None else moments


def check(process_id, topology, point, mu):
    n, rank = topology["n"], topology["rank"]
    props = ar.exact_props(topology, point)
    rows = [[Fraction(1)] * n] + [[p[0][c] for p in props] for c in range(4)]
    null = ar.nullspace(rows, n)
    if not null:
        return None
    c = null[0]
    if sum(ci * (ar.square(q) - m2) for ci, (q, m2, _) in zip(c, props)) != 0:
        return None                                     # not linearly dependent: affine_reference.py
    terms = ar.decompose(props)
    record = {"process": process_id, "topology": topology["id"], "index": point["index"], "frame": point["frame"],
              "terms": None if terms is None else [[nu for _, _, nu in ar.canonical(k)] for k in terms]}
    total = None
    if terms is not None:
        total = [[[0.0, 0.0] for _ in range(4)] for _ in run.basis(rank)]
        for key, coefficient in terms.items():
            sub = iregi_general(ar.canonical(key), rank, mu)
            if sub is None:
                total = None
                record["failed_term"] = [nu for _, _, nu in ar.canonical(key)]
                break
            for m, slots in enumerate(sub):
                for s, (re, im) in enumerate(slots):
                    total[m][s][0] += float(coefficient) * float(re)
                    total[m][s][1] += float(coefficient) * float(im)
    lanes = run.lanes_of(None, topology, [point])
    direct = {}
    for mode, name in ((0, "iregi-ibp"), (1, "iregi-pv")):
        lane, _ = run.iregi(lanes, topology, mu, mode)
        direct[name] = lane[0]["moments"] if lane and lane[0] and lane[0].get("status") == "success" else None
    direct["quad-ibp"] = run.quad(lanes, topology, mu, 0)[0]
    direct["quad-pv"] = run.quad(lanes, topology, mu, 1)[0]
    record["vs_dotted_combination"] = {name: run.array_error(v, total, rank, n, mu)
                                       if v is not None and total is not None else None
                                       for name, v in direct.items()}
    if n == 3 and direct["iregi-ibp"] is not None:
        # Scalar C0 at v = 0 (IREGI native: i pi^2 times OneLOop's
        # normalization, finite part shifted by (gamma_E + ln pi) times the pole).
        m2 = float(point["m2"])
        pole = -1 / (2 * m2)
        finite = 1 / m2 - math.log(mu * mu / m2) / (2 * m2)
        eulogpi = 0.5772156649015329 + math.log(math.pi)
        native = (finite - eulogpi * pole) * math.pi ** 2
        got = direct["iregi-ibp"][0]
        record["scalar_c0_vs_analytic"] = {
            "ir_pole": abs(float(got[1][1]) - pole * math.pi ** 2) / abs(pole * math.pi ** 2),
            "finite": abs(float(got[3][1]) - native) / abs(native)}
    return record


def main():
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser()
    parser.add_argument("points", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    data = json.loads(args.points.read_text())
    mu = float(data["mu"])
    jobs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for process in data["processes"]:
            for topology in process["topologies"]:
                for point in process["points"][: args.max_points]:
                    jobs.append(pool.submit(check, process["id"], topology, point, mu))
        out = [r for r in (job.result() for job in jobs) if r is not None]
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} Coulomb lanes", file=sys.stderr)


if __name__ == "__main__":
    main()
