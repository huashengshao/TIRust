#!/usr/bin/env python3
"""Check committed report bindings and optional byte-preserved history.

This reads saved evidence only. It never imports the numerical implementation,
executes integrals, or interprets test durations as benchmark measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_members(text):
    """Keep the original spelling of each top-level JSON value for comparison."""
    decoder = json.JSONDecoder()
    position = text.index("{") + 1
    members = {}
    while True:
        while text[position].isspace():
            position += 1
        if text[position] == "}":
            if text[position + 1:].strip():
                raise ValueError("trailing report JSON content")
            return members
        key, position = decoder.raw_decode(text, position)
        if key in members:
            raise ValueError(f"duplicate report key: {key}")
        while text[position].isspace():
            position += 1
        if text[position] != ":":
            raise ValueError("missing JSON member separator")
        position += 1
        while text[position].isspace():
            position += 1
        start = position
        _, position = decoder.raw_decode(text, position)
        members[key] = text[start:position]
        while text[position].isspace():
            position += 1
        if text[position] == ",":
            position += 1
        elif text[position] != "}":
            raise ValueError("missing JSON value separator")


def check(root, history_ref):
    def project_path(name):
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"report path outside project: {name}")
        return path

    results_path = root / "docs/data/results.json"
    text = results_path.read_text()
    members = raw_members(text)
    results = json.loads(text)
    proof = json.loads((root / "docs/data/report-validation.json").read_text())
    for name, expected in proof["report_artifacts"].items():
        if sha256(project_path(name)) != expected:
            raise ValueError(f"report artifact changed: {name}")

    sources = snapshots = 0
    for name, record in results.items():
        if not isinstance(record, dict):
            continue
        source, expected = record.get("source"), record.get("sha256")
        if not isinstance(source, str) or not isinstance(expected, str):
            continue
        path = project_path(source)
        if sha256(path) != expected:
            raise ValueError(f"report evidence changed: {name}")
        sources += 1
        if "snapshot" in record and path.suffix == ".json":
            if json.loads(path.read_text()) == record["snapshot"]:
                snapshots += 1
            elif name != "symbolica_production_qualification":
                raise ValueError(f"report snapshot differs from source: {name}")

    ledger = proof["source_ledger"]
    if len(ledger) != proof["source_ledger_count"]:
        raise ValueError("report ledger count differs")
    for record in ledger:
        value = results.get(record["key"], {})
        if any(value.get(name) != record[name] for name in ("source", "sha256")):
            raise ValueError(f"ledger differs from report: {record['key']}")
        if sha256(project_path(record["source"])) != record["sha256"]:
            raise ValueError(f"report ledger source changed: {record['source']}")

    preserved = None
    if history_ref:
        previous = subprocess.check_output(
            ["git", "show", f"{history_ref}:docs/data/results.json"],
            cwd=root, text=True,
        )
        history = raw_members(previous)
        changed = [name for name, value in history.items() if members.get(name) != value]
        if changed:
            raise ValueError(f"historical report records changed: {changed}")
        preserved = len(history)

    return {
        "status": "passed",
        "bound_sources": sources,
        "exact_json_snapshots": snapshots,
        "ledger_entries": len(ledger),
        "historical_raw_records_preserved": preserved,
        "no_integral_or_benchmark_executed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserve-from", help="Git revision whose raw result values must survive")
    args = parser.parse_args()
    print(json.dumps(check(Path.cwd().resolve(), args.preserve_from)))
