#!/usr/bin/env python3
"""Physical S-wave amplitude of g g > c c~[1S0, colour singlet or octet] at
one loop: TIRust against original IREGI, with physics checks.

Input: crates/iregi-prepare/examples/onia_swave.rs's output (every diagram's
projected numerator coefficients per point and gluon polarization pair,
its loop and tree propagator routings). For every point and diagram:
- the loop integral: propagator i is s_i q + sum_k c_ik P_k (P the physical
  external momenta in the routing slots), so its shift is s_i sum_k c_ik P_k
  (exact in the lattice), its squared mass as prepared;
- tree factor: prod over tree propagators of 1/(p^2 - m^2);
- tensor arrays from TIRust (binary64 default), original IREGI (binary64
  IBP) and an independent reference: the exact denominator identities of
  affine_reference.py when the shifts are degenerate (dotted pieces
  confirmed by binary64 IREGI's general API), else TIRust at 256 bits
  confirmed by quad IREGI;
- amplitude A = sum_diagrams tree * sum_alpha c_alpha M_alpha, in the native
  slots [UV, IR1, IR2, finite] (bare, R1 only).
Physics checks on each amplitude X (reference, TIRust, IREGI; native
normalization: X sums integral_d^dq N / D, so the amplitude is X / (2 pi)^4 at
the poles, at G = 1, alpha_s = 1/(4 pi)), against the Born A0 written out here
from the same Feynman rules (vertex i G gamma^mu T^a, quark propagator
i (pslash + m)/(p^2 - m^2), the same projector):
- double pole: X_IR2 = -6 pi^2 A0 (Catani's -C_A/eps^2 in units of alpha_s/2pi);
- single pole: IR1/IR2 + gamma_E + ln(pi) - ln(mu^2/s) = 1 - C_F/C_A + i pi
  (singlet) or 3/2 - C_F/C_A + i pi/2 (octet). These are Catani (singlet) and
  Becher-Neubert with the pair as one heavy octet of mass 2m (octet), minus the
  IR parts of the external on-shell wave-function factors that the bare
  amplitude leaves out (gluons -(5/6 C_A - 2/3 T_F n_l)/eps, c and c~
  -C_F/eps): n_l drops out;
- UV pole: X_UV = pi^2 [(2 C_A + C_F) A0 + 3 C_F m dA0/dm], minus the MS-bar
  coupling, on-shell field and on-shell mass counterterms (dA0/dm: internal
  propagator masses only);
- Ward identity: with eps_1 -> k_1/k_1^0 (or eps_2), all poles vanish; the
  finite part need not (R2 is left out), and is reported. For 1S0 at
  p_c = p_c~ this is trivial: every Levi-Civita structure needs two
  independent momenta besides the polarizations, and g g > c c~ at v = 0 has
  one (the 00 and 11 linear polarizations vanish for the same reason).

usage: swave.py NUMERATORS_JSON POINTS_JSON OUTPUT_JSON [--workers N]
"""
import argparse
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import affine_reference as ar  # noqa: E402
import run  # noqa: E402

MU = 3.0
CA, CF = 3.0, 4.0 / 3.0
EULER_PI = 0.5772156649015329 + math.log(math.pi)

# Weyl gamma matrices, as spenso-hep-lib's gamma_data_weyl; gamma5 =
# i gamma0 gamma1 gamma2 gamma3 = diag(-1, -1, 1, 1) (gamma5_weyl_data).
GAMMA = [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
         [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]],
         [[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]],
         [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]]
GAMMA5 = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
UNIT = [[float(i == j) for j in range(4)] for i in range(4)]


def mat(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def lin(*pairs):
    return [[sum(c * m[i][j] for c, m in pairs) for j in range(4)] for i in range(4)]


def slash(p):
    return lin(*[(p[0], GAMMA[0])] + [(-p[mu], GAMMA[mu]) for mu in (1, 2, 3)])


def mdot(p, q):
    return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]


_check = mat(mat(GAMMA[0], GAMMA[1]), mat(GAMMA[2], GAMMA[3]))
assert all(abs(1j * _check[i][j] - GAMMA5[i][j]) < 1e-15 for i in range(4) for j in range(4))


COLOUR = {"singlet": 4 / math.sqrt(24), "octet": math.sqrt(3 / 20) * 10 / 3, "3S1-singlet": (10 / 3) / math.sqrt(40)}


def born(point_entry, labels, state, mass_derivative=False):
    """Tree amplitude A0 (G = 1), or m dA0/dm (internal masses only), projected.

    All gluons on the open c line, summed over their orders: from c to c~,
    ubar V(e_s1) S(p_c - k_s1) V(e_s2) ... V(e_sn) v, with V(e) = i eslash,
    S(q) = i (qslash + m)/(q^2 - m^2), k the incoming gluon momenta (an
    outgoing gluon: minus its momentum). Every order has the same colour
    factor (COLOUR): singlet Tr(T^a T^a)/sqrt(24); octet sqrt(3/20) x 10/3;
    3S1 singlet sum Tr(T^a T^b T^c) d^abc/(sqrt(3) sqrt(40/3)) = (10/3)/sqrt(40).
    Diagrams with a triple-gluon vertex vanish: 1S0 by Tr[gamma^mu Pi0] = 0
    (and, singlet, by colour); 3S1 singlet by f^abc against d^abc.
    """
    momenta = point_entry["physical_momenta"]
    pc, pcb = momenta[2], momenta[3]
    m = math.sqrt(float(point_entry["m2"]))
    vectors = point_entry["polarization_vectors"]
    incoming = [momenta[0], momenta[1]] + ([[-x for x in momenta[4]]] if len(vectors) == 3 else [])
    gluons = [(incoming[g], vectors[g][labels[g]]) for g in range(len(vectors))]
    spin = GAMMA5 if len(vectors) == 2 else slash(point_entry["psi_polarizations"][labels[3]])
    projector = mat(mat(lin((1, slash(pcb)), (-m, UNIT)), spin), lin((1, slash(pc)), (m, UNIT)))
    projector = lin((1 / math.sqrt(8 * m ** 3), projector))
    def propagator(q, derivative):
        d = mdot(q, q) - m * m
        if derivative:  # m d/dm of i (qslash + m)/(q^2 - m^2)
            return lin((1j * m / d, UNIT), (2j * m * m / d ** 2, lin((1, slash(q)), (m, UNIT))))
        return lin((1j / d, lin((1, slash(q)), (m, UNIT))))
    total = 0j
    for order in itertools.permutations(range(len(gluons))):
        q, momenta_out = list(pc), []
        for g in order[:-1]:
            q = [q[mu] - gluons[g][0][mu] for mu in range(4)]
            momenta_out.append(q)
        for derived in (range(len(momenta_out)) if mass_derivative else [None]):
            x = UNIT
            for position, g in enumerate(order):
                x = mat(x, lin((1j, slash(gluons[g][1]))))
                if position < len(momenta_out):
                    x = mat(x, propagator(momenta_out[position], position == derived))
            x = mat(x, projector)
            total += sum(x[i][i] for i in range(4))
    return COLOUR[state] * total


def exact_mass(value, m2):
    """0 or the lattice m^2. The prepared masses are MC^2 with MC = fl(sqrt(m^2)),
    which can miss m^2 by an ulp; TIRust's on-shell detection is exact, so an
    ulp-off-shell soft triangle gives a regulated ln(delta) in place of its IR
    pole (README: a 1-ulp mass mismatch at point 5 broke the octet poles)."""
    if value == 0:
        return Fraction(0)
    assert abs(value - float(m2)) <= 1e-14 * float(m2), (value, m2)
    return m2


def integral_of(diagram, point_entry, pol_entry):
    m2 = Fraction(float(point_entry["m2"]))
    momenta = [[Fraction(x) for x in p] for p in point_entry["physical_momenta"]]
    slot_leg = {}
    for slot, edge in enumerate(diagram["external_edges"]):
        leg = next(l for l in diagram["external_legs"] if l["edge"] == edge)
        slot_leg[slot] = leg["index"]
    def routed(prop):
        return [sum(Fraction(c) * momenta[slot_leg[k]][mu] for k, c in enumerate(prop["external"])) for mu in range(4)]
    shifts, props = [], []
    for prop, mass in zip(diagram["loop_propagators"], pol_entry["masses_squared"]):
        k = routed(prop)
        s = prop["loop"]
        shift = tuple(s * x for x in k)
        shifts.append([float(x) for x in shift])
        assert all(Fraction(float(x)) == x for x in shift)
        props.append((shift, exact_mass(mass, m2), 1))
    tree = complex(1.0)
    for prop, mass in zip(diagram["tree_propagators"], pol_entry["tree_masses_squared"]):
        p = routed(prop)
        p2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
        tree /= float(p2 - exact_mass(mass, m2))
    return shifts, [float(m) for _, m, _ in props], tuple(props), tree


def reference_arrays(props, rank):
    """Independent reference of one integral (strings) or None."""
    n = len(props)
    rows = [[Fraction(1)] * n] + [[p[0][mu] for p in props] for mu in range(4)]
    lane = {"shifts": [[float(x) for x in q] for q, _, _ in props], "masses_squared": [float(m) for _, m, _ in props]}
    topology = {"id": "swave", "n": n, "rank": rank}
    if ar.nullspace(rows, n):
        terms = ar.decompose(props)
        if terms is None:
            return None, "identity did not close"
        groups = {}
        for key in terms:
            ck = ar.canonical(key)
            groups.setdefault((len(ck), tuple(nu for _, _, nu in ck)), []).append((ck, rank))
        values = {}
        for items in groups.values():
            values.update(ar.evaluate(items, MU))
        size = len(run.basis(rank))
        from decimal import Decimal
        total = [[[Decimal(0), Decimal(0)] for _ in range(4)] for _ in range(size)]
        for key, c in terms.items():
            how, moments, check = values[(ar.canonical(key), rank)]
            if moments is None:
                return None, f"sub-integral unconfirmed: {str(check)[:120]}"
            cr = Decimal(c.numerator) / Decimal(c.denominator)
            for m in range(size):
                for slot in range(4):
                    total[m][slot][0] += cr * Decimal(moments[m][slot][0])
                    total[m][slot][1] += cr * Decimal(moments[m][slot][1])
        return [[[str(x) for x in pair] for pair in slots] for slots in total], "exact identity"
    t, _ = run.tirust([lane], topology, MU, False)
    d256 = t[0]["direct256"]["moments"] if t else None
    for mode in (0, 1):
        q = run.quad([lane], topology, MU, mode)[0]
        check = run.exact_error(d256, q, rank, n, MU)
        if check is not None and check <= 1e-9:
            return d256, f"direct256 confirmed by quad mode {mode}"
    arbiter = run.scalar_arbitration(lane, d256, run.quad([lane], topology, MU, 0)[0]) if d256 else None
    if arbiter and arbiter["collier_vs_direct256"] <= 1e-9 and arbiter["collier_vs_quad"] > 1e-6:
        return d256, "direct256 arbitrated by COLLIER"
    return None, "unconfirmed"


def point_job(diagrams, k, state):
    out = {"index": None, "pols": {}}
    cache = {}
    for diagram in diagrams:
        point_entry = diagram["points"][k]
        out["index"] = point_entry["index"]
        for item in point_entry["polarizations"]:
            labels, entry = item[:-1], item[-1]
            if not entry["terms"] or all(abs(c[0]) + abs(c[1]) == 0 for _, c in entry["terms"]):
                continue
            shifts, masses, props, tree = integral_of(diagram, point_entry, entry)
            rank = max(sum(e) for e, _ in entry["terms"])
            key = (props, rank)
            if key not in cache:
                lane = {"shifts": shifts, "masses_squared": masses}
                topology = {"id": diagram["name"], "n": len(shifts), "rank": rank}
                t, _ = run.tirust([lane], topology, MU, False)
                ib, _ = run.iregi([lane], topology, MU, 0)
                ref, how = reference_arrays(props, rank)
                cache[key] = {"tirust": t[0]["default"]["moments"] if t and t[0]["default"]["status"] == "success" else None,
                              "iregi": ib[0]["moments"] if ib and ib[0] and ib[0].get("status") == "success" else None,
                              "reference": ref, "how": how}
            arrays = cache[key]
            index = {m: i for i, m in enumerate(run.basis(rank))}
            pol = out["pols"].setdefault("".join(map(str, labels)), {"tirust": [0j] * 4, "iregi": [0j] * 4, "reference": [0j] * 4,
                                                        "missing": set(), "iregi_subset": [[0j] * 4 for _ in range(3)],
                                                        "iregi_subset_missing": set(), "lanes_without_iregi": 0})
            # The IREGI subset: the integrals on which IREGI returns an array,
            # summed for all three codes (IREGI's score where it has no full
            # amplitude, e.g. its unstable Coulomb pentagons in g g > c c~ g).
            subset = arrays["iregi"] is not None
            pol["lanes_without_iregi"] += not subset
            for which, name in enumerate(("tirust", "iregi", "reference")):
                moments = arrays[name]
                if moments is None:
                    pol["missing"].add(name)
                    if subset:
                        pol["iregi_subset_missing"].add(name)
                    continue
                for exponents, (re, im) in entry["terms"]:
                    m = moments[index[tuple(exponents)]]
                    c = complex(re, im) * tree
                    for slot in range(4):
                        z = c * complex(float(m[slot][0]), float(m[slot][1]))
                        pol[name][slot] += z
                        if subset:
                            pol["iregi_subset"][which][slot] += z
    point_entry = diagrams[0]["points"][k]
    for key, pol in out["pols"].items():
        pol["missing"] = sorted(pol["missing"])
        pol["iregi_subset_missing"] = sorted(pol["iregi_subset_missing"])
        pol["iregi_subset"] = [[[z.real, z.imag] for z in row] for row in pol["iregi_subset"]]
        for name in ("tirust", "iregi", "reference"):
            pol[name] = [[z.real, z.imag] for z in pol[name]]
        labels = [int(c) for c in key]
        a0 = born(point_entry, labels, state)
        da0 = born(point_entry, labels, state, mass_derivative=True)
        pol["born"] = [a0.real, a0.imag]
        pol["born_mass_derivative"] = [da0.real, da0.imag]
    out["integrals"] = len(cache)
    out["references"] = sorted({v["how"] for v in cache.values()})
    return out


def pole_targets(point_entry, state):
    """(n_gluons, log part, constant) of the single-pole ratio IR1/IR2 + gamma_E + ln pi.

    Bare amplitude, colour-singlet heavy pair with n gluons (Catani minus the
    external on-shell field factors' IR parts): -(1/(n C_A)) sum_{i != j}
    T_i.T_j L_ij + 1 - 2 C_F/(n C_A), L_ij = ln(mu^2/(-s_ij - i0)). Two gluons:
    T_1.T_2 = -C_A, so ln(mu^2/s) + i pi + 1 - C_F/C_A; three: T_i.T_j = -C_A/2,
    so (1/3) sum_{i<j} L_ij + 1 - 2 C_F/(3 C_A). Octet 1S0 (Becher-Neubert,
    the pair as one heavy octet): ln(mu^2/s) + i pi/2 + 3/2 - C_F/C_A.
    """
    momenta = point_entry["physical_momenta"]
    n = len(point_entry["polarization_vectors"])
    legs = [(momenta[0], 1), (momenta[1], 1)] + ([(momenta[4], -1)] if n == 3 else [])
    def log(i, j):
        (p, a), (q, b) = legs[i], legs[j]
        sij = mdot([a * x + b * y for x, y in zip(p, q)], [a * x + b * y for x, y in zip(p, q)])
        return complex(math.log(MU * MU / abs(sij)), math.pi if sij > 0 else 0.0)
    if state == "octet":
        return n, log(0, 1) - 1j * math.pi / 2, complex(1.5 - CF / CA, 0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weight = 1.0 if n == 2 else 1 / 3
    return n, weight * sum(log(i, j) for i, j in pairs), complex(1 - 2 * CF / (n * CA), 0)


def physics(x, a0, da0, point_entry, state):
    """Pole checks of one amplitude X (4 native slots) against the Born:
    X_IR2 = -pi^2 n C_A A0; the single-pole ratio (pole_targets); X_UV =
    pi^2 [(n C_A + C_F) A0 + 3 C_F m dA0/dm] (n coupling, n/2 gluon-field,
    one heavy-field and the mass counterterms)."""
    n, log_part, constant = pole_targets(point_entry, state)
    uv, ir1, ir2 = x[0], x[1], x[2]
    out = {"double_pole": abs(ir2 / (-math.pi ** 2 * n * CA * a0) - 1)}
    ratio = ir1 / ir2 + EULER_PI - log_part if ir2 != 0 else math.nan
    out["single_pole"] = abs(ratio - constant)
    out["single_pole_ratio"] = [ratio.real, ratio.imag] if ir2 != 0 else None
    out["uv_pole"] = abs(uv / (math.pi ** 2 * ((n * CA + CF) * a0 + 3 * CF * da0)) - 1)
    return out


def checks(result, point_entry, state):
    """Scores and physics checks of one point (see the module docstring)."""
    n = len(point_entry["polarization_vectors"])
    gauge = lambda key: "2" in key[:n]
    physical = {k: v for k, v in result["pols"].items() if not gauge(k)}
    has_reference = all("reference" not in v["missing"] for v in result["pols"].values())
    basis = "reference" if has_reference else "tirust"
    scale = max(abs(complex(*z)) for v in physical.values() for z in v[basis]) or 1.0
    born_scale = max(abs(complex(*v["born"])) for v in physical.values()) or 1.0
    result["summary"] = {"scale": scale, "born_scale": born_scale, "reference": has_reference}
    for key, pol in result["pols"].items():
        ref = [complex(*z) for z in pol["reference"]]
        for name in ("tirust", "iregi"):
            val = [complex(*z) for z in pol[name]]
            ok = has_reference and name not in pol["missing"]
            pol[f"{name}_error"] = max(abs(x - y) for x, y in zip(val, ref)) / scale if ok else None
        if pol.get("lanes_without_iregi") and "reference" not in pol.get("iregi_subset_missing", []):
            rows = [[complex(*z) for z in row] for row in pol["iregi_subset"]]
            sub_scale = max(abs(z) for z in rows[2]) or 1.0
            pol["iregi_subset_error"] = {name: max(abs(x - y) for x, y in zip(rows[which], rows[2])) / sub_scale
                                         for which, name in ((0, "tirust"), (1, "iregi"))}
        a0, da0 = complex(*pol["born"]), complex(*pol["born_mass_derivative"])
        if gauge(key) or abs(a0) < 1e-12 * born_scale:
            pol["ward" if gauge(key) else "vanishing"] = {
                "born": abs(a0) / born_scale,
                **{name: [abs(complex(*z)) / scale for z in pol[name]] for name in ("reference", "tirust", "iregi")
                   if name not in pol["missing"]}}
            continue
        pol["physics"] = {name: physics([complex(*z) for z in pol[name]], a0, da0, point_entry, state)
                          for name in ("reference", "tirust", "iregi") if name not in pol["missing"]}
    # The same pole predictions scored against the point's scale: each slot's
    # largest deviation over the physical polarizations, divided by the
    # largest predicted value of that slot (a polarization with a small Born
    # inflates the per-polarization ratios above).
    n, log_part, constant = pole_targets(point_entry, state)
    predictions = {}
    for key, pol in physical.items():
        a0, da0 = complex(*pol["born"]), complex(*pol["born_mass_derivative"])
        ir2 = -math.pi ** 2 * n * CA * a0
        predictions[key] = (math.pi ** 2 * ((n * CA + CF) * a0 + 3 * CF * da0),
                            ir2 * (constant + log_part - EULER_PI), ir2)
    slot_scale = [max(abs(v[slot]) for v in predictions.values()) or 1.0 for slot in range(3)]
    result["summary"]["poles"] = {}
    for name in ("reference", "tirust", "iregi"):
        if any(name in physical[key]["missing"] for key in predictions):
            continue
        result["summary"]["poles"][name] = [
            max(abs(complex(*physical[key][name][slot]) - predictions[key][slot]) for key in predictions)
            / slot_scale[slot] for slot in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("numerators", type=Path)
    parser.add_argument("points", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reuse", action="store_true", help="recheck OUTPUT_JSON's amplitudes")
    args = parser.parse_args()
    data = json.loads(args.numerators.read_text())
    state = data.get("state", "singlet")
    diagrams = [d for d in data["diagrams"] if d["loops"] == 1]
    npoints = len(diagrams[0]["points"])
    entries = {e["index"]: e for e in diagrams[0]["points"]}
    if args.reuse:
        results = json.loads(args.output.read_text())["points"]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(point_job, [diagrams] * npoints, range(npoints), [state] * npoints))
        args.output.write_text(json.dumps({"state": state, "points": results}, indent=1) + "\n")
    for r in results:
        checks(r, entries[r["index"]], state)
    args.output.write_text(json.dumps({"state": state, "points": results}, indent=1) + "\n")
    worst = {}
    def keep(name, value):
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            worst[name] = max(worst.get(name, 0.0), value)
    fmt = lambda v: "-" if v is None else f"{v:.1e}"
    for r in results:
        for name, values in r["summary"]["poles"].items():
            for slot, value in zip(("UV", "IR1", "IR2"), values):
                keep(f"{name} {slot} pole (point scale)", value)
        print(r["index"], r["integrals"], "reference" if r["summary"]["reference"] else "NO REFERENCE",
              r["references"] if not r["summary"]["reference"] else "",
              "| poles (point scale) " + "  ".join(f"{k} " + "/".join(f"{v:.0e}" for v in vals)
                                                   for k, vals in r["summary"]["poles"].items()))
        for key, pol in sorted(r["pols"].items()):
            keep("tirust vs reference", pol["tirust_error"])
            keep("iregi vs reference", pol["iregi_error"])
            for name, value in pol.get("iregi_subset_error", {}).items():
                keep(f"{name} vs reference on IREGI's subset", value)
            line = f"   pol {key}: TIRust {fmt(pol['tirust_error'])}, IREGI {fmt(pol['iregi_error'])}"
            for name, c in pol.get("physics", {}).items():
                for check in ("double_pole", "single_pole", "uv_pole"):
                    keep(f"{name} {check}", c[check])
                line += (f" | {name}: IR2 {fmt(c['double_pole'])} IR1 {fmt(c['single_pole'])} UV {fmt(c['uv_pole'])}")
            for tag in ("ward", "vanishing"):
                if tag in pol:
                    keep(f"{tag} born", pol[tag]["born"])
                    for name in ("reference", "tirust", "iregi"):
                        if name in pol[tag]:
                            keep(f"{tag} {name}", max(pol[tag][name]))
                    line += f" | {tag}: born {fmt(pol[tag]['born'])} tirust {fmt(max(pol[tag].get('tirust', [0])))}"
            print(line)
    print("worst:", json.dumps(worst, indent=1))


if __name__ == "__main__":
    main()
