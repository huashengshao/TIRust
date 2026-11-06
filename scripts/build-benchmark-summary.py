#!/usr/bin/env python3
"""Condense one set of benchmark captures into docs/benchmark-summary.json.

The slim report (docs/benchmark-summary.typ -> docs/benchmark.pdf) shows
mostly summary tables; the captures, raw deltas and logs stay outside the
repository (huashengshao/iregi-rust-evidence or release assets). This script
reads the local capture summaries and build records and keeps only what the
tables print, with the provenance needed to find the evidence again.

usage: build-benchmark-summary.py --timing SUMMARY --timing-build BUILD
           --amplitude SUMMARY --amplitude-build BUILD
           --stability SUMMARY --stability-build BUILD
           --evidence URL [--output docs/benchmark-summary.json]

Ratios are TIRust/reference as measured in the same capture (below one:
TIRust faster); none is reconstructed across captures.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dirty(build: dict) -> bool:
    """Timing and amplitude records carry source_tree_dirty; the stability
    record lists the changed paths instead."""
    if build.get("source_tree_dirty") is not None:
        return bool(build["source_tree_dirty"])
    return bool(build.get("source_changes") or build.get("tracked_changes"))


def ratio(entry: dict | None) -> dict | None:
    if not entry or entry.get("median") is None:
        return None
    return {"median": entry["median"], "ci99": entry.get("ci99"), "pairs": entry.get("pairs")}


def engine(batch: dict, key: str) -> dict:
    x = batch.get(key) or {}
    return {
        "status": x.get("status", "not measured"),
        "us": x.get("per_point_us"),
        "lanes": x.get("lane_statuses"),
        "error": x.get("relative_error"),
    }


def timing_rows(summary: dict) -> list[dict]:
    rows = []
    for row in summary["rows"]:
        out = {k: row[k] for k in ("id", "table", "label", "n", "rank", "powers", "dimension_shift")}
        for name in ("batch1", "batch32"):
            batch = row.get(name) or {}
            ratios = batch.get("ratios") or {}
            iregi = engine(batch, "iregi")
            if batch.get("iregi_skipped"):
                iregi["status"] = "skipped (too slow at this batch)"
            out[name] = {
                "tirust": engine(batch, "tirust"),
                "tirust_binary64": engine(batch, "tirust_binary64"),
                "tirust_contracted": engine(batch, "tirust_contracted"),
                "iregi": iregi,
                "iregi_mode": next((m.get("mode") for m in batch.get("iregi_modes") or [] if m.get("accepted")), None),
                "collier": engine(batch, "collier"),
                "ninjars": engine(batch, "ninjars"),
                "ratios": {
                    "tirust_over_iregi": ratio(ratios.get("tirust_over_iregi")),
                    "tirust_over_collier": ratio(ratios.get("tirust_over_collier")),
                    "tirust_contracted_over_ninjars": ratio(ratios.get("tirust_contracted_over_ninjars")),
                },
            }
        rows.append(out)
    return rows


def amplitude_rows(summary: dict) -> list[dict]:
    rows = []
    for amp in summary["amplitudes"]:
        t = amp["timing_ms_per_call"]
        rows.append({
            "id": amp["id"],
            "label": amp["label"],
            "n": amp["n"],
            "integrals": amp["integrals"],
            "lanes": amp["lanes"],
            "ms": {k: t[k]["median"] for k in ("tirust-shared", "tirust-separate", "collier-cache", "collier-nocache")},
            "ratios": {k: ratio(amp["ratios"].get(k)) for k in (
                "tirust_shared_over_collier_cache", "tirust_separate_over_collier_nocache",
                "tirust_shared_over_separate", "collier_cache_over_nocache")},
            "tirust_lanes": {"successful": amp["tirust_final"]["successful_integral_lanes"],
                             "best_effort": amp["tirust_final"]["best_effort_integral_lanes"],
                             "sharing_on_equals_off": amp["tirust_final"]["differing_integrals"] == 0,
                             "fingerprint": amp["tirust_final"]["fingerprint_on"]},
            "collier": amp.get("collier_final"),
        })
    return rows


def stability_rows(summary: dict) -> list[dict]:
    rows = []
    for integral in summary["integrals"]:
        codes = []
        for code in integral["codes"]:
            codes.append({k: code.get(k) for k in (
                "code", "label", "median", "p99", "max", "above_1e-3", "no_array", "flagged", "rescued_by_bits")})
        ninjars = next((c for c in integral.get("contracted_codes", []) if c["code"] == "ninjars"), None)
        if ninjars:
            codes.append({k: ninjars.get(k) for k in (
                "code", "label", "median", "p99", "max", "above_1e-3", "no_array", "flagged", "rescued_by_bits")}
                         | {"contracted": True})
        rows.append({k: integral[k] for k in ("id", "label", "n", "rank", "points", "evaluated", "with_reference")}
                    | {"codes": codes})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("timing", "timing-build", "amplitude", "amplitude-build", "stability", "stability-build"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--evidence", required=True, help="where the captures and full reports are archived")
    p.add_argument("--output", type=Path, default=ROOT / "docs/benchmark-summary.json")
    a = p.parse_args()
    timing, amplitude, stability = (json.loads(x.read_text()) for x in (a.timing, a.amplitude, a.stability))
    builds = {k: json.loads(x.read_text()) for k, x in
              (("timing", a.timing_build), ("amplitude", a.amplitude_build), ("stability", a.stability_build))}
    tirust_builds = {k: {f: b["tirust"].get(f) for f in ("sha256", "git_head", "rustflags", "profile", "cargo")}
                     | {"source_tree_dirty": dirty(b["tirust"])} for k, b in builds.items()}
    # Reference binaries: hash and compiler flags (their object lists stay in
    # the build records of the evidence).
    references = {k: {name: {f: b[name].get(f) for f in ("sha256", "flags")}
                      for name in ("iregi", "collier", "ninjars") if isinstance(b.get(name), dict)}
                  for k, b in builds.items()}
    out = {
        "schema_version": 1,
        "evidence": a.evidence,
        "ratio_direction": "TIRust/reference, measured in the same capture; below one: TIRust faster",
        "captures": {
            "timing": {"campaign": timing.get("campaign"), "environment": timing["environment"],
                       "summary_sha256": sha256(a.timing)},
            "amplitude": {"campaign": amplitude.get("campaign"), "environment": amplitude["environment"],
                          "summary_sha256": sha256(a.amplitude)},
            "stability": {"campaign": stability.get("campaign"), "points_sha256": stability.get("points_sha256"),
                          "accuracy_metric": stability.get("accuracy_metric"),
                          "summary_sha256": sha256(a.stability)},
        },
        "tirust_builds": tirust_builds,
        "reference_builds": references,
        "timing": {"accuracy_metric": timing.get("accuracy_metric"), "rows": timing_rows(timing)},
        "amplitudes": amplitude_rows(amplitude),
        "stability": stability_rows(stability),
    }
    a.output.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    print(f"wrote {a.output} ({a.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
