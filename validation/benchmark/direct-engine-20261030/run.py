#!/usr/bin/env python3
"""Same-session, same-core, interleaved timing of TIRust, original IREGI and COLLIER.

For every input row and batch size (1 and 32) each candidate gets a fresh
pinned process. A pilot (3 warm-ups + 1 call) fixes the per-block call count
for ~TARGET_NS; then ROUNDS rounds run one block per candidate in a seeded
random order (paired design). Arrays are validated outside the timers:
lane 0 against the archived reference tensors (independently admitted against
quadruple-precision original IREGI), other lanes against same-session TIRust
and IREGI cross-agreement. Failed, refused and STOP outcomes are kept.

usage: run.py --cpu CPU [--output DIR] [--rounds N] [--only ID-SUBSTRING]
Writes raw.jsonl (every block) and summary.json under the output directory.
"""
import argparse
import json
import math
import os
import queue
import random
import statistics
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BIN = ROOT / "target/direct-engine-20261030/bin"
TARGET_NS = 20_000_000
MAX_ITERATIONS = 1_000_000
MODES = {0: "IBP", 1: "PV", 2: "Hybrid"}
COLLIER_SUMMARY = ROOT / "validation/benchmark/opening-double-20261026/collier/summary.json"
# NinjaRS (alphal00p/ninjars) reference build, see the stability study's build.py.
NINJARS = Path(os.environ.get("NINJARS", ROOT / "target/stability-20261030/bin/ninjars"))


def ninja_index(e):
    """Position of q0^e0 q1^e1 q2^e2 q3^e3 in Ninja's dense tensor layout."""
    def ncoeff(r):
        return (r + 1) * (r + 2) * (r + 3) * (r + 4) // 24

    def homogeneous(e):
        e0, e1, e2, e3 = e
        r = e0 + e1 + e2 + e3
        if r == 0:
            return 0
        if e3 > 0:
            return 1 + r + r * (r + 1) // 2 + homogeneous((e0, e1, e2, e3 - 1))
        if e2 > 0:
            return 1 + r + homogeneous((e0, e1, e2 - 1, 0))
        if e1 > 0:
            return 1 + homogeneous((e0, e1 - 1, 0, 0))
        return 0
    r = sum(e)
    return (ncoeff(r - 1) if r else 0) + homogeneous(e)


def numerator(case, lane):
    """Deterministic dense numerator for (row, lane): (u + i v)/mu^|alpha|."""
    rng = random.Random(f"{case['id']}:{lane}:numerator")
    mu = float(Fraction(case["lanes"][0]["mu"]))
    return [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) / mu ** sum(e) for e in basis(case["rank"])]


def ninja_case(case, lanes):
    size = (case["rank"] + 1) * (case["rank"] + 2) * (case["rank"] + 3) * (case["rank"] + 4) // 24
    out = []
    for index, sample in enumerate(lanes):
        tensor = [None] * size
        for e, c in zip(basis(case["rank"]), numerator(case, index)):
            tensor[ninja_index(e)] = [c.real, c.imag]
        mu = float(Fraction(sample["mu"]))
        out.append({"id": f"lane{index}", "shifts": [[float(Fraction(x)) for x in s] for s in sample["shifts"]],
                    "masses_squared": [[float(Fraction(m[0])), float(Fraction(m[1]))] for m in sample["masses_squared"]],
                    "mu_squared": mu * mu, "tensor": tensor})
    return {"format": "ninjars-bench-case-v1", "name": case["id"], "n": case["n"], "rank": case["rank"], "lanes": out}


class NinjaEngine:
    """NinjaRS as a pseudo engine: each request runs `ninjars case-bench`
    (pinned, fresh process, one ~20 ms block per engine) on the request's
    lanes with both engines; the fast engine is used when every lane
    succeeds, else the generic one (the fast engine, like original Ninja,
    marks cuts in Euclidean planes Unstable). Times are per batch call
    (NinjaRS's per-point time times the batch size), as for the drivers."""

    def __init__(self, cpu):
        if not NINJARS.exists():
            raise RuntimeError(f"{NINJARS} missing")
        self.cpu, self.tail = cpu, []

    def request(self, payload):
        case, lanes = payload
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            source, sink = Path(tmp) / "case.json", Path(tmp) / "bench.json"
            source.write_text(json.dumps(ninja_case(case, lanes)))
            run = subprocess.run(["taskset", "-c", str(self.cpu), str(NINJARS), "case-bench", str(source),
                                  "--widths", str(len(lanes)), "--engines", "fast,generic", "--blocks", "1",
                                  "--min-block-ms", "20", "-o", str(sink)], cwd=ROOT, capture_output=True,
                                 text=True, timeout=3600)
            if not sink.exists():
                self.tail = run.stderr.splitlines()[-5:]
                return None
            data = json.loads(sink.read_text())["cases"][0]
        fast = data["per_point_timing"][0]
        generic = (data.get("generic_per_point_timing") or [None])[0]
        use, name = (fast, "fast") if fast.get("failed_lanes", 0) == 0 else (generic, "generic")
        if use is None:
            return None
        failed = use.get("failed_lanes", 0)
        return {"total_ns": [use["summary_ns_per_point"]["total"] * len(lanes)], "ninjars_engine": name,
                "fast_failed_lanes": fast.get("failed_lanes", 0), "stable": failed == 0,
                "lanes": [{"status": "success" if failed == 0 else "unknown"} for _ in lanes]}

    def close(self):
        pass


def ninja_lane0_error(case, lane, engine, reference):
    """NinjaRS's lane-0 value against the reference array contracted exactly
    with the same numerator, as max over [finite, single, double] of
    |I - I_ref| / max |I_ref| (single pole: UV + IR, native normalization)."""
    import tempfile
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    with tempfile.TemporaryDirectory() as tmp:
        source, sink = Path(tmp) / "case.json", Path(tmp) / "check.json"
        source.write_text(json.dumps(ninja_case(case, [lane])))
        subprocess.run([str(NINJARS), "case-check", str(source), "--engines", engine, "--precision", "f64",
                        "--no-fail", "-o", str(sink)], cwd=ROOT, capture_output=True, timeout=3600)
        if not sink.exists():
            return None
        runs = json.loads(sink.read_text())["cases"][0]["runs"]
    values = runs[0]["lanes"][0].get("values") if runs and runs[0].get("lanes") else None
    if values is None or runs[0]["lanes"][0].get("status") != "Success":
        return None
    c = [complex(z) for z in numerator(case, 0)]
    ref = []
    for slots in ((3,), (0, 1), (2,)):
        re = im = Fraction(0)
        for m, cm in enumerate(c):
            for slot in slots:
                a, b = (Fraction(x) for x in reference[4 * m + slot])
                re += Fraction(cm.real) * a - Fraction(cm.imag) * b
                im += Fraction(cm.real) * b + Fraction(cm.imag) * a
        ref.append((re, im))
    pi = Fraction(Decimal("3.141592653589793238462643383279502884197"))
    a = Fraction(Decimal("0.5772156649015328606065120900824024310422") + Decimal("3.141592653589793238462643383279502884197").ln())
    b = pi * pi / 12
    pair = lambda v: (Fraction(float(v[0])), Fraction(float(v[1])))
    v3, s1, v2 = (pair(values[k]) for k in ("eps0", "eps_m1", "eps_m2"))
    mix = lambda *terms: (sum(k * x[0] for k, x in terms), sum(k * x[1] for k, x in terms))
    native = [mix((1, v3), (-a, s1), (a * a / 2 - b, v2)), mix((1, s1), (-a, v2)), v2]
    native = [(-y * pi * pi, x * pi * pi) for x, y in native]
    error = max(abs(complex(float(x - p), float(y - q))) for (x, y), (p, q) in zip(native, ref))
    amplitude = max(abs(complex(float(p), float(q))) for p, q in ref)
    return error / amplitude if amplitude else None


class Engine:
    """A persistent driver process speaking the BENCH wire protocol.

    A reader thread moves stdout lines into a queue, so buffered reads never
    hide complete lines from the timeout logic.
    """

    def __init__(self, argv, cpu):
        self.process = subprocess.Popen(["taskset", "-c", str(cpu), *argv], cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.tail = []
        while True:
            line = self._line(120)
            if line is None:
                raise RuntimeError(f"{argv[0]} did not start")
            if line.startswith("READY"):
                break

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _line(self, timeout):
        try:
            return self.lines.get(timeout=max(timeout, 0))
        except queue.Empty:
            return None

    def request(self, text, timeout=1800):
        try:
            self.process.stdin.write(text)
            self.process.stdin.flush()
        except BrokenPipeError:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._line(deadline - time.monotonic())
            if line is None:
                return None
            if line.startswith("RESULT "):
                return json.loads(line[7:])
            self.tail = (self.tail + [line])[-5:]
        return None

    def close(self):
        try:
            self.process.stdin.write("QUIT\n")
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
            self.process.wait()


def decimal(value: str) -> str:
    return repr(float(Fraction(value)))


def rows(case, lanes, with_power):
    out = []
    for sample in lanes:
        for shift, mass, power in zip(sample["shifts"], sample["masses_squared"], case["powers"]):
            row = " ".join(decimal(x) for x in shift) + f" ({decimal(mass[0])},{decimal(mass[1])})"
            out.append(row + (f" {power}" if with_power else ""))
    return "\n".join(out) + "\n"


def tirust_wire(case, lanes, warmup, repeats, iterations):
    mu = decimal(lanes[0]["mu"])
    head = (f"BENCH {case['id']} {case['n']} {case['rank']} {case['dimension_shift']} {mu} {len(lanes)} "
            f"{warmup} {repeats} {iterations}\n")
    return head + rows(case, lanes, True)


def iregi_wire(case, lanes, mode, warmup, repeats, iterations):
    api = "tensor" if case["table"] == "ordinary" else "general"
    mu = decimal(lanes[0]["mu"])
    head = (f"BENCH {case['id']} {api} {mode} {case['n']} {case['rank']} {2 * case['dimension_shift']} {mu} "
            f"{len(lanes)} {warmup} {repeats} {iterations}\n")
    return head + rows(case, lanes, True)


def collier_wire(case, lanes, config, warmup, repeats, iterations):
    mu = decimal(lanes[0]["mu"])
    head = (f"BENCH {case['id']} {config['mode']} {case['n']} {case['rank']} {mu} {len(lanes)} {warmup} {repeats} "
            f"{iterations} {config['requested_accuracy']} {config['critical_accuracy']} "
            f"{config['direct_tensor_threshold']} {config['argument_permutation']}\n")
    return head + rows(case, lanes, False)


def basis(rank):
    out = []
    for r in range(rank + 1):
        for t in range(r, -1, -1):
            for x in range(r - t, -1, -1):
                for y in range(r - t - x, -1, -1):
                    out.append((t, x, y, r - t - x - y))
    return out


def reference_array(case):
    data = json.loads((ROOT / case["reference"]).read_text())["first_output"]
    values = data["lanes"][0]["values"]
    return [(float(Fraction(z[0]["exact_rational"])), float(Fraction(z[1]["exact_rational"]))) for z in values]


def array_error(case, moments, reference):
    """Dimensionally rescaled whole-array E/A over all moments and Laurent slots."""
    if moments is None:
        return None
    mu = float(Fraction(case["lanes"][0]["mu"]))
    base = 4 + 2 * case["dimension_shift"] - 2 * sum(case["powers"])
    error = amplitude = 0.0
    nonfinite = 0
    for m, monomial in enumerate(basis(case["rank"])):
        weight = mu ** (sum(monomial) + base)
        for slot in range(4):
            a = moments[m][slot]
            b = reference[4 * m + slot]
            if not all(math.isfinite(x) for x in a):
                nonfinite += 1
                continue
            error = max(error, math.hypot(a[0] - b[0], a[1] - b[1]) / weight)
            amplitude = max(amplitude, math.hypot(b[0], b[1]) / weight)
    if nonfinite:
        return float("nan")
    return error / amplitude if amplitude else (0.0 if error == 0 else float("inf"))


def lane_moments(result, lane):
    if result is None:
        return None
    entry = result["lanes"][lane]
    # best_effort: binary64-only TIRust's uncertified array of a refused lane.
    return entry.get("moments") if entry.get("status") in ("success", "best_effort") else None


def stats(values):
    if not values:
        return None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"mean": mean, "median": statistics.median(values), "standard_error": se,
            "min": min(values), "max": max(values), "blocks": len(values)}


def ratio_summary(numerator, denominator, rng):
    pairs = [a / b for a, b in zip(numerator, denominator) if b > 0]
    if not pairs:
        return None
    medians = []
    for _ in range(2000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        medians.append(statistics.median(sample))
    medians.sort()
    return {"median": statistics.median(pairs), "ci99": [medians[10], medians[-11]], "pairs": len(pairs)}


def cpu_busy(cpu, seconds=1.0):
    def snapshot():
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith(f"cpu{cpu} "):
                fields = [int(x) for x in line.split()[1:]]
                return sum(fields), fields[3] + fields[4]
    total0, idle0 = snapshot()
    time.sleep(seconds)
    total1, idle1 = snapshot()
    return round(1 - (idle1 - idle0) / max(total1 - total0, 1), 4)


def ninjars_eligible(case):
    """NinjaRS API: unit powers, D = 4 - 2 eps, numerator rank <= n + 2."""
    return case["table"] == "ordinary" and case["rank"] <= case["n"] + 2


def candidates(case, collier_configs):
    out = [{"id": "tirust", "engine": "TIRust", "argv": [str(BIN / "tirust-direct-timing")]},
           {"id": "tirust-contracted", "engine": "TIRust", "contract": True,
            "argv": [str(BIN / "tirust-direct-timing")]},
           # Binary64 only (precision rescue disabled) at the default accuracy
           # request 2^-34 and at 2^-20 (from capture 015).
           {"id": "tirust-binary64", "engine": "TIRust", "binary64": True, "mode": "binary64 only, 2^-34",
            "argv": [str(BIN / "tirust-direct-timing")]},
           {"id": "tirust-binary64-r20", "engine": "TIRust", "binary64": True, "accuracy": 39,
            "mode": "binary64 only, 2^-20", "argv": [str(BIN / "tirust-direct-timing")]}]
    if ninjars_eligible(case):
        out.append({"id": "ninjars", "engine": "NinjaRS", "argv": [str(NINJARS)]})
        # Binary64-only TIRust with the same contraction, against NinjaRS.
        out.append({"id": "tirust-binary64-contracted", "engine": "TIRust", "contract": True, "binary64": True,
                    "mode": "binary64 only, 2^-34", "argv": [str(BIN / "tirust-direct-timing")]})
        out.append({"id": "tirust-binary64-r20-contracted", "engine": "TIRust", "contract": True, "binary64": True,
                    "accuracy": 39, "mode": "binary64 only, 2^-20", "argv": [str(BIN / "tirust-direct-timing")]})
    modes = (0, 1, 2) if case["table"] == "ordinary" else (0,)
    for mode in modes:
        out.append({"id": f"iregi-{MODES[mode].lower()}", "engine": "IREGI", "mode": MODES[mode], "mode_code": mode,
                    "argv": [str(BIN / "iregi-general")]})
    if case["table"] == "ordinary":
        out.append({"id": "collier", "engine": "COLLIER",
                    "config": collier_configs.get(case["id"], collier_configs.get("ordinary-n6-r6")),
                    "argv": [str(BIN / "collier-batch")]})
    return out


def wire(candidate, case, lanes, warmup, repeats, iterations):
    if candidate["engine"] == "NinjaRS":
        return (case, lanes)
    if candidate["engine"] == "TIRust":
        text = tirust_wire(case, lanes, warmup, repeats, iterations)
        flags = (["contract"] if candidate.get("contract") else []) + (["binary64"] if candidate.get("binary64") else []) \
            + ([f"accuracy={candidate['accuracy']}"] if candidate.get("accuracy") else [])
        if flags:
            head, _, rest = text.partition("\n")
            text = head + " " + " ".join(flags) + "\n" + rest
        return text
    if candidate["engine"] == "IREGI":
        return iregi_wire(case, lanes, candidate["mode_code"], warmup, repeats, iterations)
    return collier_wire(case, lanes, candidate["config"], warmup, repeats, iterations)


def skips_iregi(case_id, batch, patterns):
    """Whether `--skip-iregi` leaves IREGI out of this row and batch: a pattern
    is `all`, a case id (both batches) or `case-id:batch`."""
    for pattern in patterns:
        name, _, size = pattern.partition(":")
        if (name == "all" or name == case_id) and (not size or int(size) == batch):
            return True
    return False


def measure(case, batch, cpu, rounds, rng, raw, collier_configs, skip_iregi=False):
    lanes = case["lanes"][:batch]
    reference = reference_array(case)
    runs = []
    for candidate in candidates(case, collier_configs):
        if skip_iregi and candidate["engine"] == "IREGI":
            continue
        entry = {key: candidate[key] for key in candidate if key != "argv"}
        try:
            engine = NinjaEngine(cpu) if candidate["engine"] == "NinjaRS" else Engine(candidate["argv"], cpu)
        except RuntimeError as error:
            entry.update(status="start_failure", detail=str(error))
            runs.append((entry, None))
            continue
        pilot = engine.request(wire(candidate, case, lanes, 3, 1, 1))
        if pilot is None:
            entry.update(status="no_result", detail=engine.tail)
            engine.close()
            runs.append((entry, None))
            continue
        per_call = max(pilot["total_ns"][0], 1.0)
        entry["iterations"] = max(1, min(MAX_ITERATIONS, math.ceil(TARGET_NS / per_call)))
        entry["pilot_ns"] = pilot["total_ns"][0]
        for key in ("plan_ns", "workspace_ns", "first_call_ns", "plan_engine", "workspace_engine"):
            if key in pilot:
                entry[key] = pilot[key]
        entry["blocks_ns"] = []
        entry["status"] = "measured"
        runs.append((entry, engine))
    active = [run for run in runs if run[1] is not None]
    results = {}
    for round_index in range(rounds):
        order = list(active)
        rng.shuffle(order)
        for entry, engine in order:
            if entry["status"] != "measured":
                continue
            result = engine.request(wire(next(c for c in candidates(case, collier_configs) if c["id"] == entry["id"]),
                                         case, lanes, 1, 1, entry["iterations"]))
            if result is None:
                entry.update(status="lost_during_rounds", detail=engine.tail)
                continue
            entry["blocks_ns"].append(result["total_ns"][0])
            raw.write(json.dumps({"case": case["id"], "batch": batch, "round": round_index, "candidate": entry["id"],
                                  "iterations": entry["iterations"], "ns_per_call": result["total_ns"][0]}) + "\n")
            results[entry["id"]] = result
    for entry, engine in active:
        engine.close()
    # Validation outside all timers.
    for entry, _ in runs:
        result = results.get(entry["id"])
        if result is None:
            entry.setdefault("status", "no_result")
            continue
        entry["stable_flag"] = result.get("stable")
        statuses = [lane.get("status") for lane in result["lanes"]]
        entry["lane_statuses"] = {s: statuses.count(s) for s in set(statuses)}
        if entry["engine"] == "NinjaRS":
            entry["ninjars_engine"] = result.get("ninjars_engine")
            entry["fast_failed_lanes"] = result.get("fast_failed_lanes")
            entry["lane0_contracted_error"] = ninja_lane0_error(case, lanes[0], result.get("ninjars_engine", "fast"),
                                                                reference)
        else:
            entry["lane0_relative_error"] = array_error(case, lane_moments(result, 0), reference)
        if "fallback_rows" in result:
            entry["fallback_rows"] = result["fallback_rows"]
        entry["timing_ns"] = stats(entry["blocks_ns"])
    # Cross agreement of lanes 1..: TIRust versus the IREGI IBP candidate.
    tirust = results.get("tirust")
    iregi_ibp = results.get("iregi-ibp")
    cross = None
    if batch > 1 and tirust is not None and iregi_ibp is not None:
        worst = 0.0
        for lane in range(1, batch):
            a, b = lane_moments(tirust, lane), lane_moments(iregi_ibp, lane)
            if a is None or b is None:
                worst = float("nan")
                break
            flat = [tuple(z) for moment in b for z in moment]
            worst = max(worst, array_error(case, a, flat))
        cross = worst
    return [entry for entry, _ in runs], cross


def selected(entries, engine):
    usable = [e for e in entries if e["engine"] == engine and e.get("timing_ns")
              and e.get("lane_statuses", {}).get("success", 0) == sum(e.get("lane_statuses", {}).values())
              and e.get("lane0_relative_error") is not None and e["lane0_relative_error"] <= 1e-3]
    if not usable:
        return None
    return min(usable, key=lambda e: e["timing_ns"]["mean"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "target/direct-engine-20261030/capture")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--batch-rounds", type=int, default=7)
    parser.add_argument("--only", default=None)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--build-record", default="build.json")
    parser.add_argument("--inputs", default="inputs.json", help="inputs-003.json for capture 003 (adds N7R7)")
    parser.add_argument("--skip-iregi", action="append", default=[], metavar="ROW[:BATCH]",
                        help="leave IREGI out (not rerun) for a case id, case-id:batch, or all; repeatable")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((HERE / args.inputs).read_text())
    collier_configs = {}
    for row in json.loads(COLLIER_SUMMARY.read_text())["rows"]:
        collier_configs[row["id"]] = next(a["configuration"] for a in row["all_candidate_attempts"]
                                          if a["id"] == row["selected_candidate"])
    rng = random.Random(args.seed)
    environment = {
        "cpu": args.cpu, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": Path("/proc/loadavg").read_text().split()[:3],
        "cpu_model": next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                          if l.startswith("model name")),
        "uname": " ".join(os.uname()), "target_ns_per_block": TARGET_NS,
        "rounds": {"batch1": args.rounds, "batch32": args.batch_rounds}, "seed": args.seed,
        "skip_iregi": args.skip_iregi,
        "build": json.loads((HERE / args.build_record).read_text()),
    }
    rows = []
    with open(args.output / "raw.jsonl", "w") as raw:
        for case in inputs["cases"]:
            if args.only and args.only not in case["id"]:
                continue
            row = {"id": case["id"], "table": case["table"], "label": case["label"], "n": case["n"],
                   "rank": case["rank"], "powers": case["powers"], "dimension_shift": case["dimension_shift"],
                   "moments": len(basis(case["rank"]))}
            for batch in inputs["batch_sizes"]:
                busy_before = cpu_busy(args.cpu, 0.5)
                started = time.monotonic()
                skip = skips_iregi(case["id"], batch, args.skip_iregi)
                entries, cross = measure(case, batch, args.cpu,
                                         args.rounds if batch == 1 else args.batch_rounds, rng, raw, collier_configs,
                                         skip_iregi=skip)
                key = f"batch{batch}"
                tirust = next(e for e in entries if e["id"] == "tirust")
                iregi = selected(entries, "IREGI")
                collier = next((e for e in entries if e["engine"] == "COLLIER"), None)
                ratios = {}
                ninjars = next((e for e in entries if e["engine"] == "NinjaRS"), None)
                contracted = next((e for e in entries if e["id"] == "tirust-contracted"), None)
                if ninjars is not None and ninjars.get("blocks_ns") and contracted and contracted.get("blocks_ns"):
                    ratios["tirust_contracted_over_ninjars"] = ratio_summary(contracted["blocks_ns"],
                                                                             ninjars["blocks_ns"], rng)
                for name, reference in (("tirust_over_iregi", iregi), ("tirust_over_collier", collier)):
                    if reference is not None and reference.get("blocks_ns") and tirust.get("blocks_ns"):
                        ratios[name] = ratio_summary(tirust["blocks_ns"], reference["blocks_ns"], rng)
                for variant, reference, name in (
                        ("tirust-binary64", collier, "collier"), ("tirust-binary64-r20", collier, "collier"),
                        ("tirust-binary64-contracted", ninjars, "ninjars"),
                        ("tirust-binary64-r20-contracted", ninjars, "ninjars")):
                    own = next((e for e in entries if e["id"] == variant), None)
                    if own is not None and own.get("blocks_ns") and reference is not None and reference.get("blocks_ns"):
                        ratios[variant.replace("-", "_") + "_over_" + name] = ratio_summary(
                            own["blocks_ns"], reference["blocks_ns"], rng)
                row[key] = {"candidates": entries, "selected_iregi": iregi["id"] if iregi else None,
                            "iregi_skipped": skip,
                            "tirust_iregi_cross_error_lanes": cross, "ratios": ratios,
                            "cpu_busy_before": busy_before, "wall_s": round(time.monotonic() - started, 2)}
                print(json.dumps({"id": case["id"], "batch": batch,
                                  "tirust_us": (tirust.get("timing_ns") or {}).get("median", float("nan")) / 1e3,
                                  "tirust_error": tirust.get("lane0_relative_error"),
                                  "iregi": iregi["id"] if iregi else None,
                                  "iregi_us": iregi["timing_ns"]["median"] / 1e3 if iregi else None,
                                  "collier_us": ((collier or {}).get("timing_ns") or {}).get("median", float("nan")) / 1e3
                                  if collier else None,
                                  "ratios": {k: v["median"] for k, v in ratios.items() if v}}), flush=True)
            rows.append(row)
    environment["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    environment["loadavg_end"] = Path("/proc/loadavg").read_text().split()[:3]
    (args.output / "summary.json").write_text(json.dumps({"environment": environment, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
