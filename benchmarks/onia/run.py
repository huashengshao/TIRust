#!/usr/bin/env python3
"""S-wave quarkonium one-loop integrals: TIRust against original IREGI.

For every point of points.json (make_points.py) and every topology of its
process, the complete tensor array (all Cartesian moments, all four Laurent
slots: UV pole, IR single pole, IR double pole, finite part) of:
  tirust           public TIRust binary64 default (direct engine; lanes it
                   refuses are rerun at higher precision automatically)
  tirust-binary64  TIRust with the precision rescue disabled (a refused lane
                   counts with its best-effort array, flagged)
  iregi-ibp/-pv    original IREGI, binary64, IBP and PV modes, IREGI's default
                   on-shell snapping (as in MadLoop)
  quad-ibp/-pv     original IREGI in quadruple precision (same snapping)
High-precision TIRust: the direct engine at 256 bits on the same binary64
inputs (direct256) and, with --symbolic256, the facade at 256 bits, which runs
the symbolic engine (symbolic256). The inputs are exact lattice points
(make_points.py), so every code sees exactly on-shell legs, p_c = p_c~ and the
exactly vanishing Gram determinants.

Reference per lane, in this order:
  0. with --affine DIR (affine_reference.py), for an integral whose shifts are
     affinely dependent: the exact denominator identity's combination of
     non-degenerate integrals, each confirmed by quad IREGI (independent of
     both codes' handling of the degenerate kinematics);
  1. direct256 confirmed by quad IBP or quad PV to E/A <= 1e-9;
  1b. direct256 where it disagrees with quad IREGI, the integral's scalar
     master is IR finite, and COLLIER's scalar (collier-scalar) agrees with
     direct256's scalar moment to 1e-9 and contradicts quad IREGI's by more
     than 1e-6 (both quad modes share IREGI's bundled OneLOop 3.4 scalars);
  2. symbolic256 confirmed by quad IBP or quad PV to E/A <= 1e-9;
  3. quad IBP = quad PV to E/A <= 1e-12, unless COLLIER contradicts quad's scalar;
  4. direct256 = symbolic256 to E/A <= 1e-12 (no quad confirmation);
otherwise no reference (counted). Every code is scored against it by E/A, the
whole-array maximum discrepancy with the moments rescaled by mu^(|alpha|+4-2N)
(as in the stability study), and TIRust is also compared with binary64 IREGI
directly (tirust_vs_iregi).

usage: run.py POINTS_JSON OUTPUT_DIR [--workers N] [--chunk N] [--only PROCESS[/TOPOLOGY]]
                                     [--no-quad] [--symbolic256] [--affine DIR]
Binaries (build.py): ONIA_TARGET (default target), ONIA_BIN (default
$ONIA_TARGET/onia/bin: tirust-onia, iregi-onia) and ONIA_QUAD (default
$ONIA_TARGET/quad-iregi-source-audit/build-v6-001/iregi-reference-quad).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = Path(os.environ.get("ONIA_TARGET", ROOT / "target"))
BIN = Path(os.environ.get("ONIA_BIN", TARGET / "onia/bin"))
TIRUST = BIN / "tirust-onia"
IREGI = BIN / "iregi-onia"
COLLIER_SCALAR = BIN / "collier-scalar"
QUAD = Path(os.environ.get("ONIA_QUAD", TARGET / "quad-iregi-source-audit/build-v6-001/iregi-reference-quad"))
CONFIRMATION = 1e-9
AGREEMENT = 1e-12
getcontext().prec = 60


def basis(rank):
    out = []
    for r in range(rank + 1):
        for t in range(r, -1, -1):
            for x in range(r - t, -1, -1):
                for y in range(r - t - x, -1, -1):
                    out.append((t, x, y, r - t - x - y))
    return out


def array_error(moments, reference, rank, n, mu):
    """E/A over all moments and Laurent slots (binary64); NaN if nonfinite."""
    if moments is None or reference is None:
        return None
    base = 4 - 2 * n
    error = amplitude = 0.0
    for m, monomial in enumerate(basis(rank)):
        weight = mu ** (sum(monomial) + base)
        for slot in range(4):
            a, b = moments[m][slot], reference[m][slot]
            if b is None or None in b:
                return None
            if a is None or None in a:
                return float("nan")
            a = (float(a[0]), float(a[1]))
            b = (float(b[0]), float(b[1]))
            if not all(math.isfinite(x) for x in a):
                return float("nan")
            error = max(error, math.hypot(a[0] - b[0], a[1] - b[1]) / weight)
            amplitude = max(amplitude, math.hypot(b[0], b[1]) / weight)
    return error / amplitude if amplitude else (0.0 if error == 0 else float("inf"))


def exact_error(moments, reference, rank, n, mu):
    """E/A in 60-digit decimal arithmetic on the printed digits."""
    if moments is None or reference is None:
        return None
    base = 4 - 2 * n
    scale = Decimal(repr(mu))
    error = amplitude = Decimal(0)
    for m, monomial in enumerate(basis(rank)):
        weight = scale ** (sum(monomial) + base)
        for slot in range(4):
            a, b = moments[m][slot], reference[m][slot]
            if a is None or b is None or None in a or None in b:
                return None
            a = [Decimal(str(x).strip()) for x in a]
            b = [Decimal(str(x).strip()) for x in b]
            if not all(x.is_finite() for x in a + b):
                return None
            error = max(error, ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2).sqrt() / weight)
            amplitude = max(amplitude, (b[0] ** 2 + b[1] ** 2).sqrt() / weight)
    return float(error / amplitude) if amplitude else (0.0 if error == 0 else float("inf"))


def exact(x):
    """The exact decimal expansion of a binary64 value (for the quad build)."""
    return format(Decimal(x), "f")


def collier_scalar(lane):
    """COLLIER's scalar C0 or D0 of a 3- or 4-point lane, as the IREGI-native
    finite part (i pi^2 times COLLIER's = OneLOop's normalization), or None
    (other N, IR-divergent, or no result)."""
    shifts = [[Fraction(x) for x in q] for q in lane["shifts"]]
    n = len(shifts)
    if n not in (3, 4):
        return None
    s = lambda i, j: float(sum((1 if mu == 0 else -1) * (shifts[i][mu] - shifts[j][mu]) ** 2 for mu in range(4)))
    if n == 3:
        args = ["C0", s(1, 0), s(2, 1), s(2, 0)]
    else:
        args = ["D0", s(1, 0), s(2, 1), s(3, 2), s(3, 0), s(2, 0), s(3, 1)]
    args += list(lane["masses_squared"])
    text = " ".join(a if isinstance(a, str) else repr(float(a)) for a in args) + "\nQUIT\n"
    out = subprocess.run([str(COLLIER_SCALAR)], input=text, capture_output=True, text=True, timeout=600)
    rows = [line.split()[1:] for line in out.stdout.splitlines() if line.startswith("SCALAR ")]
    if not rows:
        return None
    a, b, *poles = map(float, rows[0])
    if any(abs(x) > 0 for x in poles) or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return (-math.pi ** 2 * b, math.pi ** 2 * a)


def scalar_arbitration(lane, d256, quad_moments):
    """Relative distance of COLLIER's IR-finite scalar master from the scalar
    moment (finite slot) of the 256-bit TIRust array and of the quad IREGI
    array; None when COLLIER cannot arbitrate."""
    if d256 is None or quad_moments is None:
        return None
    for moments in (d256, quad_moments):
        if any(float(x) != 0 for slot in moments[0][:3] for x in slot):
            return None
    c = collier_scalar(lane)
    if c is None:
        return None
    norm = math.hypot(*c)
    rel = lambda m: math.hypot(float(m[0][3][0]) - c[0], float(m[0][3][1]) - c[1]) / norm
    return {"collier_vs_direct256": rel(d256), "collier_vs_quad": rel(quad_moments)}


def lanes_of(process, topology, points):
    """Shifts q_0 = 0, q_i = sum of the incoming momenta of groups 0..i-1, and
    the squared masses of every lane; exact by construction (checked)."""
    out = []
    for point in points:
        # Fraction(float(x)): the exact binary64 value of the shortest decimal.
        momenta = {leg: [Fraction(float(x)) for x in v] for leg, v in point["momenta"].items()}
        q = [[Fraction(0)] * 4]
        for group in topology["groups"][:-1]:
            q.append([a + sum(momenta[leg][c] for leg in group) for c, a in enumerate(q[-1])])
        shifts = [[float(x) for x in row] for row in q]
        assert all(Fraction(f) == x for row, frow in zip(q, shifts) for x, f in zip(row, frow))
        m2 = float(point["m2"])
        out.append({"shifts": shifts, "masses_squared": [m2 if m == "H" else 0.0 for m in topology["masses"]]})
    return out


def results(argv, text, timeout=7200, env=None):
    out = subprocess.run(argv, input=text, capture_output=True, text=True, timeout=timeout, cwd=ROOT, env=env)
    return [json.loads(line[7:]) for line in out.stdout.splitlines() if line.startswith("RESULT ")], out


def iregi(lanes, topology, mu, mode):
    n, rank = topology["n"], topology["rank"]
    head = f"BENCH s tensor {mode} {n} {rank} 0 {mu!r} {len(lanes)} 0 1 1\n"
    rows = "".join(" ".join(repr(x) for x in shift) + f" ({m!r},0.0) 1\n"
                   for lane in lanes for shift, m in zip(lane["shifts"], lane["masses_squared"]))
    res, out = results([str(IREGI)], head + rows + "QUIT\n")
    if not res:
        return [None] * len(lanes), (out.stderr or "")[-400:]
    return res[0]["lanes"], res[0]["total_ns"]


def quad(lanes, topology, mu, mode, flags=None):
    """Quad IREGI arrays (also of lanes it reports unstable, for use as a
    confirmer only); `flags`, if a list, receives each lane's stable flag."""
    n, rank = topology["n"], topology["rank"]
    text = ""
    for k, lane in enumerate(lanes):
        text += f"EVAL q{k} tensor {mode} {n} {rank} 0 {exact(mu)} 1 1e-12\n" + "".join(
            " ".join(exact(x) for x in shift) + f" {exact(m)} 0.0 1\n"
            for shift, m in zip(lane["shifts"], lane["masses_squared"]))
    res, _ = results([str(QUAD)], text + "QUIT\n")
    by_id = {r.get("id"): r for r in res}
    if flags is not None:
        flags.extend(by_id.get(f"q{k}", {}).get("stable") for k in range(len(lanes)))
    return [by_id.get(f"q{k}", {}).get("moments") for k in range(len(lanes))]


def tirust(lanes, topology, mu, symbolic256):
    with tempfile.TemporaryDirectory() as tmp:
        source, sink = Path(tmp) / "input.json", Path(tmp) / "out.jsonl"
        chunk = {"id": topology["id"], "n": topology["n"], "rank": topology["rank"], "mu": repr(mu),
                 "lanes": [{"shifts": [[repr(x) for x in s] for s in lane["shifts"]],
                            "masses_squared": [repr(m) for m in lane["masses_squared"]]} for lane in lanes]}
        source.write_text(json.dumps({"integrals": [chunk]}))
        env = dict(os.environ, TIRUST_ONIA_SYMBOLIC256="1" if symbolic256 else "0")
        run = subprocess.run([str(TIRUST), str(source), str(sink)], capture_output=True, text=True, timeout=7200,
                             env=env)
        if run.returncode != 0 or not sink.exists():
            return None, run.stderr[-2000:]
        return [json.loads(line) for line in sink.read_text().splitlines()], None


def lane_values(lane):
    if lane is None:
        return None, True
    if lane.get("status") == "success":
        return lane.get("moments"), False
    return lane.get("best_effort_moments"), True


def chunk_job(process_id, topology, start, points, mu, with_quad, symbolic256, affine=None):
    n, rank = topology["n"], topology["rank"]
    lanes = lanes_of(None, topology, points)
    t, t_error = tirust(lanes, topology, mu, symbolic256)
    codes, iregi_ns = {}, {}
    for mode, name in ((0, "iregi-ibp"), (1, "iregi-pv")):
        codes[name], iregi_ns[name] = iregi(lanes, topology, mu, mode)
    f_ibp, f_pv = [], []
    if with_quad:
        q_ibp, q_pv = quad(lanes, topology, mu, 0, f_ibp), quad(lanes, topology, mu, 1, f_pv)
    else:
        q_ibp = q_pv = [None] * len(lanes)
        f_ibp = f_pv = [None] * len(lanes)
    e = lambda a, b: array_error(a, b, rank, n, mu)
    x = lambda a, b: exact_error(a, b, rank, n, mu)
    records = []
    for k, point in enumerate(points):
        record = {"process": process_id, "topology": topology["id"], "n": n, "rank": rank,
                  "index": point["index"], "frame": point["frame"]}
        if t is None:
            record["tirust_error"] = t_error
            tk = {}
        else:
            tk = t[k]
        d256 = tk.get("direct256", {}).get("moments")
        s256 = tk.get("symbolic256", {}).get("moments")
        record["status"] = {
            "tirust": tk.get("default", {}).get("status"),
            "tirust-binary64": tk.get("binary64", {}).get("status"),
            "direct256": tk.get("direct256", {}).get("status"),
            "symbolic256": tk.get("symbolic256", {}).get("status"),
            "iregi-ibp": (codes["iregi-ibp"][k] or {}).get("status"),
            "iregi-pv": (codes["iregi-pv"][k] or {}).get("status"),
            "quad-ibp": None if q_ibp[k] is None else ("success" if f_ibp[k] else "unstable"),
            "quad-pv": None if q_pv[k] is None else ("success" if f_pv[k] else "unstable"),
        }
        aff = (affine or {}).get(point["index"])
        a_ref = aff["moments"] if aff else None
        if aff:
            record["affine_status"] = aff["status"]
        checks = {"direct256_affine": x(d256, a_ref), "quadIBP_affine": x(q_ibp[k], a_ref),
                  "quadPV_affine": x(q_pv[k], a_ref), "symbolic256_affine": x(s256, a_ref),
                  "direct256_quadIBP": x(d256, q_ibp[k]), "direct256_quadPV": x(d256, q_pv[k]),
                  "quadIBP_quadPV": x(q_pv[k], q_ibp[k]), "symbolic256_direct256": x(s256, d256),
                  "symbolic256_quadIBP": x(s256, q_ibp[k]), "symbolic256_quadPV": x(s256, q_pv[k])}
        record["reference_checks"] = checks
        ok = lambda v, tol: v is not None and math.isfinite(v) and v <= tol
        # COLLIER arbitrates (scalar master, IR-finite only) when TIRust's
        # 256-bit array and quad IREGI disagree: both quad modes share
        # IREGI's bundled OneLOop 3.4 scalars (README: its D0 is wrong on
        # some boxes with masses (m, m, 0, m)).
        arbiter = None
        if (d256 is not None and q_ibp[k] is not None and not ok(checks["direct256_quadIBP"], CONFIRMATION)
                and not ok(checks["direct256_quadPV"], CONFIRMATION)):
            arbiter = scalar_arbitration(lanes[k], d256, q_ibp[k])
            record["scalar_arbitration"] = arbiter
        quad_contradicted = bool(arbiter) and arbiter["collier_vs_quad"] > 1e-6
        reference = None
        if a_ref is not None:
            reference, record["reference"] = a_ref, "affine identity over confirmed non-degenerate integrals"
        elif d256 is not None and (ok(checks["direct256_quadIBP"], CONFIRMATION) or
                                 ok(checks["direct256_quadPV"], CONFIRMATION)):
            reference, record["reference"] = d256, "direct256 confirmed by quad IREGI"
        elif quad_contradicted and arbiter["collier_vs_direct256"] <= CONFIRMATION:
            reference, record["reference"] = d256, "direct256; quad IREGI's scalar master contradicted by COLLIER"
        elif s256 is not None and (ok(checks["symbolic256_quadIBP"], CONFIRMATION) or
                                   ok(checks["symbolic256_quadPV"], CONFIRMATION)):
            reference, record["reference"] = s256, "symbolic256 confirmed by quad IREGI"
        elif ok(checks["quadIBP_quadPV"], AGREEMENT) and not quad_contradicted:
            reference, record["reference"] = q_ibp[k], "quad IBP = quad PV"
        elif ok(checks["symbolic256_direct256"], AGREEMENT):
            reference, record["reference"] = d256, "direct256 = symbolic256 (no quad confirmation)"
        else:
            record["reference"] = None
        arrays, flagged = {}, {}
        if tk:
            default = tk["default"]
            arrays["tirust"] = default["moments"] if default["status"] == "success" else None
            flagged["tirust"] = bool(default.get("fallback_used"))
            record["tirust_diagnostics"] = default.get("diagnostics")
            record["tirust_estimated_error"] = default.get("estimated_error")
            record["tirust_ns_per_lane"] = tk.get("default_ns_per_lane")
            arrays["tirust-binary64"], flagged["tirust-binary64"] = lane_values(tk["binary64"])
            record["tirust_binary64_diagnostics"] = tk["binary64"].get("diagnostics")
        for name in ("iregi-ibp", "iregi-pv"):
            arrays[name], flagged[name] = lane_values(codes[name][k])
        record["iregi_ns_per_lane"] = {name: (v[0] / len(points) if isinstance(v, list) and v else None)
                                       for name, v in iregi_ns.items()}
        record["flagged"] = flagged
        if reference is not None:
            delta = {name: e(v, reference) if v is not None else None for name, v in arrays.items()}
            # Quad IREGI scored as a code only where it reports the lane stable.
            for name, q, stable in (("quad-ibp", q_ibp[k], f_ibp[k]), ("quad-pv", q_pv[k], f_pv[k])):
                delta[name] = e(q, reference) if q is not None and stable else None
            record["delta"] = delta
        record["tirust_vs_iregi"] = {name: e(arrays.get("tirust"), arrays[name])
                                     if arrays.get("tirust") is not None and arrays[name] is not None else None
                                     for name in ("iregi-ibp", "iregi-pv")}
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--only", default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--no-quad", action="store_true")
    parser.add_argument("--symbolic256", action="store_true")
    parser.add_argument("--affine", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.points.read_text())
    mu = float(data["mu"])
    jobs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for process in data["processes"]:
            points = process["points"][: args.max_points]
            affine = defaultdict(dict)
            if args.affine and (args.affine / f"affine-{process['id']}.jsonl").exists():
                for line in (args.affine / f"affine-{process['id']}.jsonl").read_text().splitlines():
                    r = json.loads(line)
                    affine[r["topology"]][r["index"]] = r
            for topology in process["topologies"]:
                name = f"{process['id']}/{topology['id']}"
                if args.only and not (name == args.only or process["id"] == args.only):
                    continue
                for start in range(0, len(points), args.chunk):
                    chunk = points[start:start + args.chunk]
                    future = pool.submit(chunk_job, process["id"], topology, start, chunk, mu, not args.no_quad,
                                         args.symbolic256,
                                         {p["index"]: affine[topology["id"]][p["index"]] for p in chunk
                                          if p["index"] in affine[topology["id"]]})
                    jobs[future] = (process["id"], name)
        sinks = {}
        done = 0
        for future in as_completed(jobs):
            process_id, name = jobs[future]
            if process_id not in sinks:
                sinks[process_id] = (args.output / f"deltas-{process_id}.jsonl").open("w")
            for record in future.result():
                sinks[process_id].write(json.dumps(record) + "\n")
            sinks[process_id].flush()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} chunks", file=sys.stderr, flush=True)
        for sink in sinks.values():
            sink.close()


if __name__ == "__main__":
    main()
