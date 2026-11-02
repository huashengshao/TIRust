#!/usr/bin/env python3
"""Condense a direct-engine capture into the report snapshot (summary.json).

Reads only saved capture data (no numerical execution). Headline times are
medians of per-block means; ratios are medians of paired same-round block
ratios with 99% bootstrap intervals, as recorded by run.py. IREGI selects the
fastest mode whose complete finite lane-0 array is within 1e-3 of the
reference (the opening-table rule); every mode's outcome is retained.

usage: summarize.py CAPTURE_DIR OUTPUT_JSON [BUILD_RECORD]
(BUILD_RECORD adds the recorded TIRust source commit and provenance note.)
"""
import json
import math
import sys
from pathlib import Path


def us(stats, key="median"):
    return None if not stats else stats[key] / 1e3


def candidate(entry, batch):
    timing = entry.get("timing_ns")
    statuses = entry.get("lane_statuses") or {}
    # Binary64-only TIRust returns refused lanes' best (finite, uncertified)
    # arrays, like COLLIER's flagged lanes, which also count as complete.
    complete = bool(statuses) and statuses.get("success", 0) + statuses.get("best_effort", 0) == sum(statuses.values())
    # NinjaRS returns the integrated numerator, not the array: its error is
    # the contracted lane-0 value against the reference contracted exactly.
    error = entry.get("lane0_contracted_error") if entry["engine"] == "NinjaRS" else entry.get("lane0_relative_error")
    if error is not None and isinstance(error, float) and math.isnan(error):
        error = None
    record = {
        "id": entry["id"],
        "engine": entry["engine"],
        "mode": entry.get("mode") or (entry.get("config") or {}).get("id"),
        "status": entry.get("status"),
        "complete_finite": complete,
        "lane_statuses": statuses,
        "relative_error": error,
        "accepted": complete and error is not None and error <= 1e-3,
        "best_effort_lanes": statuses.get("best_effort", 0),
        "timing_us": None if not timing else {
            "median": us(timing), "mean": us(timing, "mean"),
            "standard_error": None if timing["standard_error"] is None else timing["standard_error"] / 1e3,
            "min": us(timing, "min"), "max": us(timing, "max"), "blocks": timing["blocks"],
        },
        "per_point_us": None if not timing else us(timing) / batch,
        "iterations_per_block": entry.get("iterations"),
    }
    for key in ("plan_ns", "workspace_ns", "first_call_ns", "plan_engine", "workspace_engine", "fallback_rows",
                "ninjars_engine", "fast_failed_lanes"):
        if key in entry:
            record[key] = entry[key]
    if entry.get("detail"):
        record["detail"] = entry["detail"]
    return record


def main():
    capture, output = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads((capture / "summary.json").read_text())
    rows = []
    for row in data["rows"]:
        out = {k: row[k] for k in ("id", "table", "label", "n", "rank", "powers", "dimension_shift", "moments")}
        for batch in (1, 32):
            block = row[f"batch{batch}"]
            candidates = [candidate(e, batch) for e in block["candidates"]]
            tirust = next(c for c in candidates if c["id"] == "tirust")
            iregi = next((c for c in candidates if c["id"] == block["selected_iregi"]), None)
            collier = next((c for c in candidates if c["engine"] == "COLLIER"), None)
            out[f"batch{batch}"] = {
                "tirust": tirust,
                "iregi": iregi,
                "iregi_modes": [c for c in candidates if c["engine"] == "IREGI"],
                # IREGI not rerun in this capture (run.py --skip-iregi).
                "iregi_skipped": block.get("iregi_skipped", False),
                "collier": collier,
                "tirust_contracted": next((c for c in candidates if c["id"] == "tirust-contracted"), None),
                "tirust_binary64": next((c for c in candidates if c["id"] == "tirust-binary64"), None),
                "tirust_binary64_r20": next((c for c in candidates if c["id"] == "tirust-binary64-r20"), None),
                "tirust_binary64_contracted": next((c for c in candidates if c["id"] == "tirust-binary64-contracted"), None),
                "tirust_binary64_r20_contracted": next(
                    (c for c in candidates if c["id"] == "tirust-binary64-r20-contracted"), None),
                "ninjars": next((c for c in candidates if c["engine"] == "NinjaRS"), None),
                "ratios": block["ratios"],
                "tirust_iregi_cross_error_lanes": block["tirust_iregi_cross_error_lanes"],
                "cpu_busy_before": block["cpu_busy_before"],
            }
        rows.append(out)
    environment = data["environment"]
    build = environment.pop("build")
    snapshot = {
        "schema_version": 1,
        "campaign": "direct-engine-20261030",
        "environment": environment,
        "binaries": {name: {"sha256": build[name]["sha256"]} for name in ("tirust", "iregi", "collier", "ninjars")
                     if name in build},
        "tirust_git_head": build["tirust"]["git_head"],
        "selection_rule": "IREGI: fastest mode with a complete finite lane-0 array within E/A<=1e-3 of the reference; COLLIER: archived opening-double selected public configuration; TIRust: public default facade (Hybrid, binary64).",
        "accuracy_metric": "E/A: dimensionally rescaled whole-array max discrepancy over all moments and four Laurent slots, versus the archived reference tensors admitted against quadruple-precision original IREGI.",
        "rows": rows,
    }
    if len(sys.argv) > 3:
        record = json.loads(Path(sys.argv[3]).read_text())["tirust"]
        snapshot["tirust_source_commit"] = record.get("source_commit", build["tirust"]["git_head"])
        if "source_note" in record:
            snapshot["tirust_source_note"] = record["source_note"]
    output.write_text(json.dumps(snapshot, indent=1) + "\n")


if __name__ == "__main__":
    main()
