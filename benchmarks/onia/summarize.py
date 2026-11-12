#!/usr/bin/env python3
"""Summarize run.py's deltas-*.jsonl: per process and topology, for every
code, how many lanes reach E/A <= 1e-10 and <= 1e-6 against the reference,
how many return no array (status), and the largest E/A; TIRust against
binary64 IREGI directly; the reference sources.

usage: summarize.py OUTPUT_DIR   (writes OUTPUT_DIR/summary.json and summary.md)
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODES = ["tirust", "tirust-binary64", "iregi-ibp", "iregi-pv", "quad-ibp", "quad-pv"]
TIGHT, LOOSE = 1e-10, 1e-6


def stats(values):
    """values: E/A per lane (None: no array)."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {"lanes": len(values), "no_array": sum(v is None for v in values),
            "nonfinite": sum(v is not None and not math.isfinite(v) for v in values),
            "tight": sum(v <= TIGHT for v in finite), "loose": sum(v <= LOOSE for v in finite),
            "max": max(finite) if finite else None,
            "median": sorted(finite)[len(finite) // 2] if finite else None}


def median(values):
    values = sorted(v for v in values if v)
    return values[len(values) // 2] if values else None


def fmt(x):
    return "-" if x is None else f"{x:.1e}"


def main():
    out = Path(sys.argv[1])
    summary = {"processes": {}}
    md = []
    for path in sorted(out.glob("deltas-*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        process = records[0]["process"]
        by_top = defaultdict(list)
        for r in records:
            by_top[r["topology"]].append(r)
        tops = {}
        for top, rs in sorted(by_top.items(), key=lambda kv: (-kv[1][0]["n"], kv[0])):
            with_ref = [r for r in rs if r.get("delta")]
            entry = {"n": rs[0]["n"], "rank": rs[0]["rank"], "lanes": len(rs), "with_reference": len(with_ref),
                     "reference_sources": dict(Counter(r["reference"] for r in rs)),
                     "status": {c: dict(Counter(r["status"].get(c) for r in rs)) for c in
                                ["tirust", "tirust-binary64", "direct256", "iregi-ibp", "iregi-pv",
                                 "quad-ibp", "quad-pv"]},
                     "codes": {c: stats([r["delta"].get(c) for r in with_ref]) for c in CODES},
                     "tirust_vs_iregi": {c: stats([r["tirust_vs_iregi"].get(c) for r in rs])
                                         for c in ("iregi-ibp", "iregi-pv")},
                     "tirust_flagged": sum(bool(r["flagged"].get("tirust")) for r in rs),
                     "tirust_ns_per_lane": median([r.get("tirust_ns_per_lane") for r in rs]),
                     "iregi_ibp_ns_per_lane": median([r["iregi_ns_per_lane"].get("iregi-ibp") for r in rs])}
            tops[top] = entry
        summary["processes"][process] = tops
        md.append(f"### {process}\n")
        md.append("| topology | N | lanes | ref | TIRust ≤1e-10 | TIRust max | TIRust b64 max | IREGI IBP ≤1e-10 "
                  "| IREGI IBP max | IREGI PV ok/max | quad IBP max | TIRust vs IREGI IBP max |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for top, e in tops.items():
            c = e["codes"]
            pv = c["iregi-pv"]
            md.append(f"| `{top}` | {e['n']} | {e['lanes']} | {e['with_reference']} | "
                      f"{c['tirust']['tight']}/{e['with_reference']} | {fmt(c['tirust']['max'])} | "
                      f"{fmt(c['tirust-binary64']['max'])} | {c['iregi-ibp']['tight']}/{e['with_reference']} | "
                      f"{fmt(c['iregi-ibp']['max'])} | {pv['loose']}/{e['with_reference']} {fmt(pv['max'])} | "
                      f"{fmt(c['quad-ibp']['max'])} | {fmt(e['tirust_vs_iregi']['iregi-ibp']['max'])} |")
        md.append("")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
