#!/usr/bin/env python3
"""Amplitude benchmark: TIRust sharing on/off against COLLIER cache on/off.

An amplitude is an N-gon of the binary64 stability study (top-loop gg -> gggg
hexagon, gg -> Hgg pentagon) with every pinch of at least two propagators, each
at rank = its number of propagators (renormalizable vertices, Feynman gauge),
on the study's first LANES phase-space points. One amplitude call evaluates
every integral at every point. Candidates, each a long-lived process pinned to
one CPU, in seeded random order per round, one block each:
  tirust-shared / tirust-separate: AmplitudeWorkspace, sharing on / off (the
    amplitude's common loop-momentum routing);
  collier-cache / collier-nocache: COLLIER 1.2.10 COLI, global cache on
    (InitCacheSystem once, InitEvent(1) per point) / off; each integral in its
    own zero-origin routing.
Every candidate first evaluates two warm-up calls (COLLIER's cache learns the
call sequence of its first events). Ratios are medians of same-round block
ratios with 99% bootstrap intervals. After the rounds: TIRust's bitwise
comparison of sharing off and on, and COLLIER's per (integral, point)
coefficient sums with and without its cache.

usage: run.py --cpu N --build-record build-NNN.json --output summary-NNN.json
Run from the repository root after `source scripts/env.sh`.
"""
import argparse
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BIN = ROOT / "target/amplitude-20261102/bin"
WORK = ROOT / "target/amplitude-20261102"
sys.path.insert(0, str(HERE.parent))
from run import cpu_busy, ratio_summary, stats  # noqa: E402  (direct-engine harness)


class Process:
    def __init__(self, argv, cpu):
        self.proc = subprocess.Popen(["taskset", "-c", str(cpu), *argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, cwd=ROOT)
        # COLLIER prints its banner on stdout while initializing.
        for _ in range(500):
            ready = self.proc.stdout.readline()
            if ready.startswith("READY") or not ready:
                break
        if not ready.startswith("READY"):
            raise RuntimeError(f"{argv[0]} did not start: {ready!r}")

    def request(self, line, prefix):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply.startswith(prefix + " "):
            raise RuntimeError(f"unexpected reply to {line!r}: {reply[:200]!r}")
        return json.loads(reply[len(prefix) + 1:])

    def close(self):
        self.proc.stdin.write("QUIT\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=60)


def amplitude_spec(integral, lanes):
    n = integral["n"]
    integrals = [{"members": list(members), "rank": len(members)}
                 for size in range(2, n + 1) for members in itertools.combinations(range(n), size)]
    points = integral["points"][:lanes]
    masses = integral["masses_squared"]
    tirust = {"integrals": integrals,
              "lanes": [{"shifts": point, "masses": masses, "mu": integral["mu"]} for point in points]}
    rows = [f"{len(integrals)} {len(points)} {n}"]
    rows += [" ".join(map(str, [len(i["members"]), i["rank"], *[m + 1 for m in i["members"]]])) for i in integrals]
    for point in points:
        rows += [" ".join(repr(float(x)) for x in [*point[t], masses[t]]) for t in range(n)]
        rows.append(repr(float(integral["mu"])))
    return tirust, "\n".join(rows) + "\n", len(integrals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--build-record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", default="target/stability-20261030/points-003.json")
    parser.add_argument("--amplitudes", default="hexagon-r6,pentagon-r5")
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    data = json.loads((ROOT / args.points).read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    environment = {"cpu": args.cpu, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "loadavg_start": Path("/proc/loadavg").read_text().split()[:3],
                   "cpu_model": next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                                     if l.startswith("model name")),
                   "rounds": args.rounds, "lanes": args.lanes, "seed": args.seed, "points": args.points,
                   "build": json.loads((HERE / args.build_record).read_text())}
    amplitudes = []
    for name in args.amplitudes.split(","):
        integral = next(i for i in data["integrals"] if i["id"] == name)
        tirust_spec, collier_spec, count = amplitude_spec(integral, args.lanes)
        (WORK / f"{name}.json").write_text(json.dumps(tirust_spec))
        (WORK / f"{name}.txt").write_text(collier_spec)
        tirust = Process([str(BIN / "tirust-amplitude-timing"), str(WORK / f"{name}.json")], args.cpu)
        colliers = {cache: Process([str(BIN / "collier-amplitude"), str(WORK / f"{name}.txt"), str(cache)], args.cpu)
                    for cache in (0, 1)}
        candidates = {
            "tirust-shared": (tirust, "BENCH 1 {w} {i}"), "tirust-separate": (tirust, "BENCH 0 {w} {i}"),
            "collier-cache": (colliers[1], "BENCH {w} {i}"), "collier-nocache": (colliers[0], "BENCH {w} {i}"),
        }
        # Pilot (two warm-up calls each) fixes the calls per block (>= 0.2 s).
        iterations = {}
        for key, (process, template) in candidates.items():
            pilot = process.request(template.format(w=2, i=1), "RESULT")
            iterations[key] = max(1, round(2e8 / pilot["ns_per_call"]))
        blocks = {key: [] for key in candidates}
        shared_stats = []
        busy_before = cpu_busy(args.cpu, 0.5)
        for _ in range(args.rounds):
            order = list(candidates)
            rng.shuffle(order)
            for key in order:
                process, template = candidates[key]
                result = process.request(template.format(w=0, i=iterations[key]), "RESULT")
                blocks[key].append(result["ns_per_call"])
                if key == "tirust-shared":
                    shared_stats.append({"reused": result["reused"], "computed": result["computed"]})
        tirust_final = tirust.request("FINAL", "FINAL")
        collier_final = {cache: colliers[cache].request("FINAL", "FINAL") for cache in (0, 1)}
        tirust.close()
        for process in colliers.values():
            process.close()
        # COLLIER with and without its cache: per (integral, point) sums.
        worst = 0.0
        for a, b in zip(collier_final[0]["sums"], collier_final[1]["sums"]):
            scale = a[2] if a[2] > 0 else 1.0
            worst = max(worst, abs(complex(a[0], a[1]) - complex(b[0], b[1])) / scale)
        ratios = {
            "tirust_shared_over_separate": ratio_summary(blocks["tirust-shared"], blocks["tirust-separate"], rng),
            "collier_cache_over_nocache": ratio_summary(blocks["collier-cache"], blocks["collier-nocache"], rng),
            "tirust_shared_over_collier_cache": ratio_summary(blocks["tirust-shared"], blocks["collier-cache"], rng),
            "tirust_separate_over_collier_nocache": ratio_summary(blocks["tirust-separate"], blocks["collier-nocache"], rng),
        }
        amplitudes.append({
            "id": name, "label": integral["label"], "n": integral["n"], "integrals": count, "lanes": args.lanes,
            "iterations_per_block": iterations, "cpu_busy_before": busy_before,
            "timing_ms_per_call": {key: {k: (v / 1e6 if isinstance(v, float) else v) for k, v in stats(values).items()}
                                   for key, values in blocks.items()},
            "blocks_ns": blocks, "ratios": ratios, "tirust_sharing": shared_stats[-1] if shared_stats else None,
            "tirust_final": tirust_final,
            "collier_final": {"cache_vs_nocache_max_relative_sum_difference": worst,
                              "accuracy_flagged": {str(c): collier_final[c]["accuracy_flagged"] for c in (0, 1)},
                              "error_flagged": {str(c): collier_final[c]["error_flagged"] for c in (0, 1)},
                              "calls": len(collier_final[0]["sums"])},
        })
        print(json.dumps({"id": name, "ms": {k: round(statistics.median(v) / 1e6, 2) for k, v in blocks.items()},
                          "ratios": {k: round(v["median"], 3) for k, v in ratios.items() if v},
                          "tirust_final": tirust_final, "collier_max_diff": worst}), flush=True)
    environment["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    environment["loadavg_end"] = Path("/proc/loadavg").read_text().split()[:3]
    build = environment.pop("build")
    summary = {"schema_version": 1, "campaign": "amplitude-20261102", "environment": environment,
               "binaries": {k: {"sha256": build[k]["sha256"]} for k in ("tirust", "collier")},
               "tirust_git_head": build["tirust"]["git_head"], "amplitudes": amplitudes}
    (HERE / args.output).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
