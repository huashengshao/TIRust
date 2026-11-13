#!/usr/bin/env python3
"""Quarkonium integral families as amplitudes: TIRust's AmplitudeWorkspace
against original IREGI.

For every family (families.py: a parent of the largest N and all its pinches,
in the parent's routing) and every chunk of points:
- TIRust evaluates all members in one AmplitudeWorkspace call
  (examples/onia_amplitude.rs), timed with sharing on and off; the arrays
  with sharing on and off must be bit-identical;
- every member goes through run.py's chunk_job, which gives its reference
  (the rules of run.py, with --affine and --coulomb inputs computed on the
  families), original IREGI's binary64 IBP and PV arrays, and the standalone
  TIRust array;
- the amplitude of a family at a point is A = sum over members of
  sum_alpha c_alpha M_alpha, with fixed random coefficients per family,
  member and point (run.numerator: (u + i v)/mu^|alpha|, u, v in [-1, 1]),
  in the native slots combined as [finite, single pole (UV + IR), double
  pole]. Its error is max_slot |A - A_ref| / max_slot |A_ref|
  (run.contracted_error), with A_ref summed in 60-digit arithmetic over the
  members' references; a family point without a reference for every member
  has no A_ref (counted).

Output in OUTPUT_DIR: amplitude-PROCESS.jsonl (one record per family and
point: errors of TIRust, IREGI IBP and PV, coverage, statuses, times per
point) and deltas-PROCESS.jsonl (run.py's per-member records, for
summarize.py).

usage: amplitude.py FAMILIES_JSON OUTPUT_DIR --affine DIR --coulomb FILE
                    [--workers N] [--chunk N] [--iterations N] [--only PROCESS]
Binaries as run.py, plus $ONIA_BIN/tirust-onia-amplitude.
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
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

AMPLITUDE_DRIVER = run.BIN / "tirust-onia-amplitude"


def tirust_family(parent, members, points, mu, iterations):
    lanes = run.lanes_of(None, {"groups": parent["groups"], "masses": parent["masses"]}, points)
    spec = {"integrals": [{"id": m["id"], "members": m["members"], "rank": m["rank"]} for m in members],
            "lanes": [{"shifts": [[repr(x) for x in s] for s in lane["shifts"]],
                       "masses_squared": [repr(x) for x in lane["masses_squared"]], "mu": repr(mu)}
                      for lane in lanes]}
    with tempfile.TemporaryDirectory() as tmp:
        source, sink = Path(tmp) / "spec.json", Path(tmp) / "out.json"
        source.write_text(json.dumps(spec))
        out = subprocess.run([str(AMPLITUDE_DRIVER), str(source), str(sink), str(iterations)], capture_output=True,
                             text=True, timeout=7200)
        if out.returncode != 0 or not sink.exists():
            return None, out.stderr[-2000:]
        return json.loads(sink.read_text()), None


def family_job(process_id, parent, members, start, points, mu, affine, coulomb, iterations):
    family_id = parent["id"]
    amplitude, error = tirust_family(parent, members, points, mu, iterations)
    member_records = []
    per_point = [defaultdict(list) for _ in points]
    for j, member in enumerate(members):
        records = run.chunk_job(process_id, member, start, points, mu, True, False,
                                {p["index"]: affine[member["id"]][p["index"]] for p in points
                                 if p["index"] in affine.get(member["id"], {})},
                                {p["index"]: coulomb[member["id"]][p["index"]] for p in points
                                 if p["index"] in coulomb.get(member["id"], {})},
                                return_arrays=True)
        for k, record in enumerate(records):
            arrays = record.pop("arrays")
            member_records.append(record)
            coefficients = run.numerator({"id": f"{family_id}:{member['id']}", "mu": mu, "rank": member["rank"]},
                                         points[k]["index"])
            slot = per_point[k]
            reference = arrays.get("reference")
            slot["reference"].append(None if reference is None else run.contract(reference, coefficients, True))
            lane = amplitude["integrals"][j]["lanes"][k] if amplitude else None
            tirust = lane["moments"] if lane and lane["status"] == "success" else None
            slot["tirust-amplitude"].append(None if tirust is None else run.contract(tirust, coefficients))
            slot["tirust-status"].append(lane["status"] if lane else error)
            for name in ("iregi-ibp", "iregi-pv"):
                values = arrays.get(name)
                slot[name].append(None if values is None else run.contract(values, coefficients))
            slot["iregi-ns"].append((record.get("iregi_ns_per_lane") or {}).get("iregi-ibp"))
            slot["member-delta-tirust-amplitude"].append(
                run.array_error(tirust, reference, member["rank"], member["n"], mu)
                if tirust is not None and reference is not None else None)
    out = []
    for k, point in enumerate(points):
        slot = per_point[k]
        record = {"process": process_id, "family": family_id, "index": point["index"], "frame": point["frame"],
                  "members": len(members),
                  "members_without_reference": sum(r is None for r in slot["reference"]),
                  "tirust_failed_members": sum(v is None for v in slot["tirust-amplitude"]),
                  "tirust_statuses": sorted({str(s)[:60] for s in slot["tirust-status"] if s != "success"}),
                  "largest_member_error_tirust_amplitude": max(
                      (v for v in slot["member-delta-tirust-amplitude"] if v is not None), default=None)}
        total_reference = None
        if all(r is not None for r in slot["reference"]):
            total_reference = [(sum(r[s][0] for r in slot["reference"]), sum(r[s][1] for r in slot["reference"]))
                               for s in range(3)]
            record["amplitude_reference"] = [[str(a), str(b)] for a, b in total_reference]
        errors = {}
        for name in ("tirust-amplitude", "iregi-ibp", "iregi-pv"):
            values = slot[name]
            if any(v is None for v in values) or total_reference is None:
                errors[name] = None
                continue
            total = [(math.fsum(v[s][0] for v in values), math.fsum(v[s][1] for v in values)) for s in range(3)]
            errors[name] = run.contracted_error(total, total_reference)
        record["amplitude_error"] = errors
        if amplitude:
            record["tirust_ns_per_point"] = {key: value / len(points) for key, value in amplitude["ns_per_call"].items()}
            record["tirust_bitwise_share_equal"] = amplitude["bitwise_equal"]
        iregi = slot["iregi-ns"]
        record["iregi_ibp_ns_per_point"] = math.fsum(iregi) if all(v is not None for v in iregi) else None
        out.append(record)
    return out, member_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("families", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--coulomb", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.families.read_text())
    mu = float(data["mu"])
    coulomb_all = defaultdict(dict)
    for r in json.loads(args.coulomb.read_text()):
        coulomb_all[(r["process"], r["topology"])][r["index"]] = r["vs_dotted_combination"].get("quad-ibp")
    jobs = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for process in data["processes"]:
            if args.only and process["id"] != args.only:
                continue
            by_id = {t["id"]: t for t in process["topologies"]}
            parents = {t["parent"]: t for t in process["topologies"] if len(t["members"]) == t["n"] and t["n"] ==
                       max(u["n"] for u in process["topologies"])}
            affine = defaultdict(dict)
            path = args.affine / f"affine-{process['id']}.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    affine[r["topology"]][r["index"]] = r
            coulomb = {t: coulomb_all[(process["id"], t)] for t in by_id if (process["id"], t) in coulomb_all}
            for family in process["families"]:
                members = [by_id[m] for m in family["members"]]
                parent = parents[family["parent"]]
                ids = set(family["members"])
                for start in range(0, len(process["points"]), args.chunk):
                    chunk = process["points"][start:start + args.chunk]
                    future = pool.submit(family_job, process["id"], parent, members, start, chunk, mu,
                                         {m: affine[m] for m in ids if m in affine},
                                         {m: coulomb[m] for m in ids if m in coulomb}, args.iterations)
                    jobs[future] = process["id"]
        sinks = {}
        for done, future in enumerate(as_completed(jobs), 1):
            process_id = jobs[future]
            if process_id not in sinks:
                sinks[process_id] = ((args.output / f"amplitude-{process_id}.jsonl").open("w"),
                                     (args.output / f"deltas-{process_id}.jsonl").open("w"))
            records, member_records = future.result()
            for record in records:
                sinks[process_id][0].write(json.dumps(record) + "\n")
            for record in member_records:
                sinks[process_id][1].write(json.dumps(record) + "\n")
            for sink in sinks[process_id]:
                sink.flush()
            if done % 10 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} family chunks", file=sys.stderr, flush=True)
        for pair in sinks.values():
            for sink in pair:
                sink.close()


if __name__ == "__main__":
    main()
