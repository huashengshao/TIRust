#!/usr/bin/env python3
"""S-wave quarkonium one-loop integrals: topologies and exact binary64 points.

Processes (all momenta incoming in the loop, p_c = p_c~ = P/2, P^2 = 4 m_c^2):
  gg_QQ   g(p1) g(p2) > c(P/2) c~(P/2)               (2 -> 1 kinematics)
  gg_QQg  g(p1) g(p2) > c(P/2) c~(P/2) g(k)          (2 -> 2 kinematics)

Topologies. Every cyclic ordering of the external legs (mirror images and
rotations removed), every grouping of consecutive legs into N >= 2 loop
vertices (pinches: a group is a tree attached to the loop), and every
propagator mass pattern that QCD vertices allow: a vertex that attaches an odd
number of heavy quarks joins a heavy-quark propagator (mass m_c) and a gluon
(massless), any other vertex joins two propagators of the same kind (both
heavy, or both massless: gluon, ghost or light quark). Since p_c = p_c~, the
c and c~ legs are interchangeable and duplicates are removed. The rank is the
maximal one of a renormalizable gauge theory, rank = N.

Points. With p_c = p_c~ the external momenta span only two (gg_QQ) or three
(gg_QQg) dimensions, so every Gram determinant that involves both heavy-quark
legs vanishes, and some modified Cayley determinants too (the Coulomb-exchange
triangle and box). To give every code exactly these kinematics, all momenta
are integer vectors on a lattice, scaled by a power of two:
  - massless momenta are integer lightlike vectors (rational points of the
    sphere: (a^2+b^2+c^2, 2ac, 2bc, a^2+b^2-c^2) times an integer);
  - P is their exact sum, p_c = p_c~ = P/2, m_c^2 = P^2/4 exactly;
  - every shift component (and every difference of two shifts) is below
    2^24 lattice units, so every invariant and scalar product of the shifts
    is exact in binary64, however a code forms it from the components.
make_points.py checks the last point for every point and topology.
Frames: "cm" has the beams along +-z (gg_QQ: boosted along z by a random
rapidity, the first point at rest; gg_QQg: the gluon in a random direction),
"generic" rotates and boosts a physical configuration at random before it is
put on the lattice. The charm mass is near 1.5 GeV but differs per point (the
lattice fixes P^2); mu = 3 GeV.

usage: make_points.py OUTPUT_JSON [--points-qq N] [--points-qqg N] [--seed S]
"""
import argparse
import itertools
import json
import math
import random
from fractions import Fraction

MU = 3.0
MC = 1.5
SEED = 20261112
HEAVY = {"Q", "Qb"}
PROCESSES = {
    "gg_QQ": {"label": "g g > c c~, S-wave (p_c = p_c~)", "legs": ["g1", "g2", "Q", "Qb"]},
    "gg_QQg": {"label": "g g > c c~ g, S-wave (p_c = p_c~)", "legs": ["g1", "g2", "Q", "Qb", "g3"]},
}
# |component| bound of every shift and of every difference of two shifts:
# products stay below 2^48, so any sum of up to 32 of them is exact.
LIMIT = 2 ** 24


# Topologies ------------------------------------------------------------------

def cyclic_orders(legs):
    first, rest = legs[0], legs[1:]
    seen = set()
    for perm in itertools.permutations(rest):
        if tuple(reversed(perm)) in seen:
            continue
        seen.add(perm)
        yield (first,) + perm


def groupings(order, n):
    """Groups of consecutive legs of the cyclic order, n groups."""
    size = len(order)
    for cuts in itertools.combinations(range(size), n):
        bounds = list(cuts) + [cuts[0] + size]
        yield [tuple(order[i % size] for i in range(a + 1, b + 1)) for a, b in zip(bounds, bounds[1:])]


def odd(group):
    return sum(leg in HEAVY for leg in group) % 2 == 1


def mass_patterns(groups):
    """Propagator i lies between groups i-1 and i (shift q_i = sum_{j<i} K_j)."""
    out = []
    for first in "HL":
        masses = [first]
        for group in groups[:-1]:
            masses.append(masses[-1] if not odd(group) else ("L" if masses[-1] == "H" else "H"))
        closing = masses[-1] if not odd(groups[-1]) else ("L" if masses[-1] == "H" else "H")
        if closing == masses[0]:
            out.append(masses)
    return out


def label(group):
    return "+".join(sorted("Q" if leg in HEAVY else leg for leg in group))


def canonical(groups, masses):
    """Key invariant under rotations and reflection of the loop, with c = c~."""
    n = len(groups)
    cyc = []
    for i in range(n):
        cyc += [masses[i], label(groups[i])]
    keys = []
    for seq in (cyc, cyc[::-1]):
        for shift in range(2 * n):
            rotated = seq[shift:] + seq[:shift]
            if rotated[0] in ("H", "L"):
                keys.append(tuple(rotated))
    return min(keys)


def topologies(legs):
    found = {}
    for n in range(len(legs), 1, -1):
        for order in cyclic_orders(legs):
            for groups in groupings(order, n):
                for masses in mass_patterns(groups):
                    key = canonical(groups, masses)
                    if key not in found:
                        found[key] = (groups, masses)
    out = []
    for groups, masses in found.values():
        n = len(groups)
        heavy = sum(m == "H" for m in masses)
        name = "N{}-{}-{}".format(n, "".join(masses), "|".join(label(g) for g in groups))
        out.append({"id": name, "n": n, "rank": n, "groups": [list(g) for g in groups], "masses": masses,
                    "heavy_propagators": heavy})
    out.sort(key=lambda t: (-t["n"], t["id"]))
    return out


# Lattice kinematics ----------------------------------------------------------

def lattice_lightlike(direction, energy, c, rng):
    """An integer lightlike vector with roughly the given direction and energy."""
    nx, ny, nz = direction
    if nz <= 0:
        u, v, sign = nx / (1 - nz), ny / (1 - nz), 1
    else:
        u, v, sign = nx / (1 + nz), ny / (1 + nz), -1
    a, b = round(u * c), round(v * c)
    base = (a * a + b * b + c * c, 2 * a * c, 2 * b * c, sign * (a * a + b * b - c * c))
    scale = max(1, round(energy / base[0]))
    return tuple(scale * x for x in base)


def dot(p, q):
    return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]


def add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def neg(p):
    return tuple(-x for x in p)


def random_direction(rng):
    c = 2 * rng.random() - 1
    phi = 2 * math.pi * rng.random()
    s = math.sqrt(1 - c * c)
    return (s * math.cos(phi), s * math.sin(phi), c)


def rotate_boost(p, rotation, rapidity):
    """Apply a rotation (3x3) then a boost along z."""
    e, x, y, z = p
    x, y, z = (sum(rotation[i][j] * w for j, w in enumerate((x, y, z))) for i in range(3))
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    return (ch * e + sh * z, x, y, sh * e + ch * z)


def random_rotation(rng):
    # Uniform random rotation from a random unit quaternion.
    q = [rng.gauss(0, 1) for _ in range(4)]
    n = math.sqrt(sum(x * x for x in q))
    w, x, y, z = (v / n for v in q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def to_lattice(p, c, rng):
    e = p[0]
    d = (p[1] / e, p[2] / e, p[3] / e)
    norm = math.sqrt(sum(x * x for x in d))
    return lattice_lightlike(tuple(x / norm for x in d), e, c, rng)


def point_qq(frame, index, rng):
    """gg_QQ: integer p1, p2 (lightlike); returns the incoming momenta."""
    unit = 3 * 2 ** 15                                                  # m_c = 1.5 GeV at 2^16 per GeV
    if frame == "cm":
        y = 0.0 if index == 0 else rng.uniform(-3, 3)
        a, b = round(unit * math.exp(y)), round(unit * math.exp(-y))
        return {"g1": (a, 0, 0, a), "g2": (b, 0, 0, -b)}
    # generic: a c.m. configuration rotated and boosted, put on the lattice.
    rotation, rapidity = random_rotation(rng), rng.uniform(-2, 2)
    p1 = rotate_boost((unit, 0, 0, unit), rotation, rapidity)
    p2 = rotate_boost((unit, 0, 0, -unit), rotation, rapidity)
    return {"g1": to_lattice(p1, 2 ** 6, rng), "g2": to_lattice(p2, 2 ** 6, rng)}


def point_qqg(frame, index, rng):
    """gg_QQg: integer p1, p2, k (lightlike); P = p1 + p2 - k."""
    while True:
        roots = math.exp(rng.uniform(math.log(6.0), math.log(60.0)))   # sqrt(s) / GeV
        m2 = (2 * MC) ** 2
        s = roots * roots
        k0 = (s - m2) / (2 * roots)
        direction = random_direction(rng)
        if k0 * math.hypot(direction[0], direction[1]) < 1.0:            # gluon p_T > 1 GeV
            continue
        scale = 2 ** 15                                                 # lattice units per GeV
        if frame == "cm":
            k = lattice_lightlike(direction, k0 * scale, 2 ** 7, rng)
            # Beams along z: p1 = (a,0,0,a), p2 = (b,0,0,-b). Fix b, solve P^2 for a.
            b = round(roots * scale / 2)
            target = m2 * scale * scale
            denom = 4 * b - 2 * (k[0] - k[3])
            if denom <= 0:
                continue
            a = round((target + 2 * b * (k[0] + k[3])) / denom)
            if a <= 0:
                continue
            moms = {"g1": (a, 0, 0, a), "g2": (b, 0, 0, -b), "g3": k}
        else:
            rotation, rapidity = random_rotation(rng), rng.uniform(-1.5, 1.5)
            h = roots * scale / 2
            kk = tuple(k0 * scale * x for x in (1.0,) + direction)
            moms = {"g1": to_lattice(rotate_boost((h, 0, 0, h), rotation, rapidity), 2 ** 6, rng),
                    "g2": to_lattice(rotate_boost((h, 0, 0, -h), rotation, rapidity), 2 ** 6, rng),
                    "g3": to_lattice(rotate_boost(kk, rotation, rapidity), 2 ** 6, rng)}
        p1, p2, k = moms["g1"], moms["g2"], moms["g3"]
        big = add(p1, p2, neg(k))
        s_lat = 2 * dot(p1, p2)
        if big[0] <= 0 or dot(big, big) <= 0:
            continue
        ratio = dot(big, big) / s_lat
        if not 0.002 < ratio < 0.5:
            continue
        # Gluon p_T > 1 GeV in the c.m. frame of p1 + p2 is ensured above for
        # "cm"; for "generic" check the invariant p_T^2 = 2 (p1.k)(p2.k)/(p1.p2).
        pt2 = 2 * dot(p1, k) * dot(p2, k) / dot(p1, p2) / (scale * scale)
        if pt2 < 1.0:
            continue
        return moms


def incoming(legs, moms):
    """All-incoming momenta of the legs: doubled so that P/2 is an integer."""
    gluons = {name: tuple(2 * x for x in moms[name]) for name in moms}
    p_total = add(gluons["g1"], gluons["g2"], *(neg(gluons[g]) for g in gluons if g not in ("g1", "g2")))
    p_q = tuple(x // 2 for x in p_total)
    assert add(p_q, p_q) == p_total
    out = {}
    for leg in legs:
        if leg in ("g1", "g2"):
            out[leg] = gluons[leg]
        elif leg in HEAVY:
            out[leg] = neg(p_q)
        else:
            out[leg] = neg(gluons[leg])
    return out, p_q


def shifts(topology, momenta):
    q = [(0, 0, 0, 0)]
    for group in topology["groups"][:-1]:
        q.append(add(q[-1], *(momenta[leg] for leg in group)))
    return q


def exact_check(q_list, power):
    """Every |component| of shifts and of their differences below LIMIT, and
    every scalar product exact in binary64 once scaled by 2^power."""
    scale = Fraction(2) ** power
    for qi in q_list:
        for qj in q_list:
            d = tuple(a - b for a, b in zip(qi, qj))
            if max(abs(x) for x in d + qi) >= LIMIT:
                return False
            for u, v in ((d, d), (qi, qj)):
                exact = Fraction(dot(u, v)) * scale * scale
                fu, fv = [float(Fraction(x) * scale) for x in u], [float(Fraction(x) * scale) for x in v]
                approx = fu[0] * fv[0] - fu[1] * fv[1] - fu[2] * fv[2] - fu[3] * fv[3]
                if Fraction(approx) != exact:
                    return False
    return True


def make_process(name, count, rng, tops):
    legs = PROCESSES[name]["legs"]
    generator = point_qq if name == "gg_QQ" else point_qqg
    points = []
    for index in range(count):
        frame = "cm" if index < count // 2 else "generic"
        while True:
            moms = generator(frame, index, rng)
            momenta, p_q = incoming(legs, moms)
            m2_lat = dot(p_q, p_q)
            # Physical units: lattice * 2^power GeV with m_c near 1.5 GeV.
            power = -round(math.log2(math.sqrt(m2_lat) / MC))
            if all(exact_check(shifts(t, momenta), power) for t in tops):
                break
        scale = Fraction(2) ** power
        gev = {leg: [float(Fraction(x) * scale) for x in p] for leg, p in momenta.items()}
        m2 = float(Fraction(m2_lat) * scale * scale)
        assert Fraction(m2) == Fraction(m2_lat) * scale * scale
        s = float(Fraction(2 * dot(momenta["g1"], momenta["g2"])) * scale * scale)
        points.append({"frame": frame, "index": index, "momenta": {k: [repr(x) for x in v] for k, v in gev.items()},
                       "m2": repr(m2), "s": repr(s), "lattice_power": power})
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--points-qq", type=int, default=100)
    parser.add_argument("--points-qqg", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = {"seed": args.seed, "mu": repr(MU), "processes": []}
    for name, count in (("gg_QQ", args.points_qq), ("gg_QQg", args.points_qqg)):
        tops = topologies(PROCESSES[name]["legs"])
        points = make_process(name, count, rng, tops)
        out["processes"].append({"id": name, "label": PROCESSES[name]["label"], "legs": PROCESSES[name]["legs"],
                                 "topologies": tops, "points": points})
        counts = {}
        for t in tops:
            counts[t["n"]] = counts.get(t["n"], 0) + 1
        print(f"{name}: {len(tops)} topologies {dict(sorted(counts.items(), reverse=True))}, {len(points)} points")
    with open(args.output, "w") as sink:
        json.dump(out, sink, indent=None)
        sink.write("\n")


if __name__ == "__main__":
    main()
