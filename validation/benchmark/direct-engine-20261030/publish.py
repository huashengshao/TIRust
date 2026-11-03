#!/usr/bin/env python3
"""Append the direct-engine campaign to the report data and reseal the proof.

bind: append new top-level members to docs/data/results.json, preserving the
      raw spelling of every existing member (append-only, as the checker
      requires with --preserve-from).
seal: after rebuilding docs/benchmark.pdf and an independent byte-identical
      copy, refresh docs/data/report-validation.json: every listed report
      artifact is rehashed, the new records join the source ledger, and a new
      checkpoint entry keeps the previous proof hash.

usage: publish.py bind
       publish.py seal --independent-pdf PATH --page-count N --base-commit SHA
Run from the repository root.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "docs/data/results.json"
PROOF = ROOT / "docs/data/report-validation.json"
REPORT_DATE = "2026-10-30"
RECORDS = {
    "direct_engine_20261030": ("summary.json", True),
    "direct_engine_20261030_build": ("build.json", True),
    "direct_engine_20261030_inputs": ("inputs.json", False),
}
# Later captures append their own records (results.json is append-only).
LATER = {
    "002": {
        "direct_engine_20261030_capture002": ("summary-002.json", True),
        "direct_engine_20261030_build002": ("build-002.json", True),
    },
    "003": {
        "direct_engine_20261030_capture003": ("summary-003.json", True),
        "direct_engine_20261030_inputs003": ("inputs-003.json", False),
        "direct_engine_20261030_build003": ("build-003.json", True),
    },
    # Capture 004 reuses inputs-003.json (bound with capture 003).
    "004": {
        "direct_engine_20261030_capture004": ("summary-004.json", True),
        "direct_engine_20261030_build004": ("build-004.json", True),
    },
    # Capture 005 reuses inputs-003.json (bound with capture 003).
    "005": {
        "direct_engine_20261030_capture005": ("summary-005.json", True),
        "direct_engine_20261030_build005": ("build-005.json", True),
    },
    # Capture 006 reuses inputs-003.json (bound with capture 003).
    "006": {
        "direct_engine_20261030_capture006": ("summary-006.json", True),
        "direct_engine_20261030_build006": ("build-006.json", True),
    },
    # Capture 007 reuses inputs-003.json (bound with capture 003).
    "007": {
        "direct_engine_20261030_capture007": ("summary-007.json", True),
        "direct_engine_20261030_build007": ("build-007.json", True),
    },
    # Capture 008 reuses inputs-003.json (bound with capture 003).
    "008": {
        "direct_engine_20261030_capture008": ("summary-008.json", True),
        "direct_engine_20261030_build008": ("build-008.json", True),
    },
    # Capture 009 (32a79e2) was not published; capture 010 reuses inputs-003.json.
    "010": {
        "direct_engine_20261030_capture010": ("summary-010.json", True),
        "direct_engine_20261030_build010": ("build-010.json", True),
    },
    "011": {
        "direct_engine_20261030_capture011": ("summary-011.json", True),
        "direct_engine_20261030_build011": ("build-011.json", True),
    },
    "012": {
        "direct_engine_20261030_capture012": ("summary-012.json", True),
        "direct_engine_20261030_build012": ("build-012.json", True),
    },
    "013": {
        "direct_engine_20261030_capture013": ("summary-013.json", True),
        "direct_engine_20261030_build013": ("build-013.json", True),
    },
    "014": {
        "direct_engine_20261030_capture014": ("summary-014.json", True),
        "direct_engine_20261030_build014": ("build-014.json", True),
    },
    "015": {
        "direct_engine_20261030_capture015": ("summary-015.json", True),
        "direct_engine_20261030_build015": ("build-015.json", True),
    },
    "016": {
        "direct_engine_20261030_capture016": ("summary-016.json", True),
        "direct_engine_20261030_build016": ("build-016.json", True),
    },
    "017": {
        "direct_engine_20261030_capture017": ("summary-017.json", True),
        "direct_engine_20261030_build017": ("build-017.json", True),
    },
    "018": {
        "direct_engine_20261030_capture018": ("summary-018.json", True),
        "direct_engine_20261030_build018": ("build-018.json", True),
    },
    "019": {
        "direct_engine_20261030_capture019": ("summary-019.json", True),
        "direct_engine_20261030_build019": ("build-019.json", True),
    },
    "020": {
        "direct_engine_20261030_capture020": ("summary-020.json", True),
        "direct_engine_20261030_build020": ("build-020.json", True),
    },
    # Amplitude benchmark (amplitude/): TIRust sharing on/off, COLLIER cache on/off.
    "amp001": {
        "direct_engine_20261030_amplitude001": ("amplitude/summary-001.json", True),
        "direct_engine_20261030_amplitude_build001": ("amplitude/build-001.json", True),
    },
    "amp002": {
        "direct_engine_20261030_amplitude002": ("amplitude/summary-002.json", True),
        "direct_engine_20261030_amplitude_build002": ("amplitude/build-002.json", True),
    },
    "amp003": {
        "direct_engine_20261030_amplitude003": ("amplitude/summary-003.json", True),
        "direct_engine_20261030_amplitude_build003": ("amplitude/build-003.json", True),
    },
    "amp004": {
        "direct_engine_20261030_amplitude004": ("amplitude/summary-004.json", True),
        "direct_engine_20261030_amplitude_build004": ("amplitude/build-004.json", True),
    },
    "amp005": {
        "direct_engine_20261030_amplitude005": ("amplitude/summary-005.json", True),
        "direct_engine_20261030_amplitude_build005": ("amplitude/build-005.json", True),
    },
    "amp006": {
        "direct_engine_20261030_amplitude006": ("amplitude/summary-006.json", True),
        "direct_engine_20261030_amplitude_build006": ("amplitude/build-006.json", True),
    },
    "021": {
        "direct_engine_20261030_capture021": ("summary-021.json", True),
        "direct_engine_20261030_build021": ("build-021.json", True),
    },
    "amp007": {
        "direct_engine_20261030_amplitude007": ("amplitude/summary-007.json", True),
        "direct_engine_20261030_amplitude_build007": ("amplitude/build-007.json", True),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(capture=None) -> None:
    global RECORDS
    if capture:
        RECORDS = LATER[capture]
    text = RESULTS.read_text()
    existing = json.loads(text)
    additions = {}
    if "direct_engine_report_updated_on" not in existing and not capture:
        additions["direct_engine_report_updated_on"] = REPORT_DATE
    for key, (name, snapshot) in RECORDS.items():
        if key in existing:
            raise SystemExit(f"{key} already bound; results.json is append-only")
        path = HERE / name
        record = {"source": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        if snapshot:
            record["snapshot"] = json.loads(path.read_text())
        additions[key] = record
    body = text.rstrip()
    if not body.endswith("}"):
        raise SystemExit("unexpected results.json layout")
    body = body[:-1].rstrip()
    for key, value in additions.items():
        body += ",\n  " + json.dumps(key) + ": " + json.dumps(value, indent=None)
    RESULTS.write_text(body + "\n}\n")
    json.loads(RESULTS.read_text())
    print(f"bound {len(additions)} records")


def seal(independent: Path, pages: int, base_commit: str, capture=None) -> None:
    global RECORDS
    if capture:
        RECORDS = {**RECORDS, **LATER[capture]}
    proof = json.loads(PROOF.read_text())
    previous_sha = sha256(PROOF)
    pdf = ROOT / "docs/benchmark.pdf"
    if pdf.read_bytes() != independent.read_bytes():
        raise SystemExit("independent rebuild differs from docs/benchmark.pdf")
    results = json.loads(RESULTS.read_text())
    artifacts = proof["report_artifacts"]
    for name in list(artifacts):
        artifacts[name] = sha256(ROOT / name)
    for name in ("scripts/build-benchmark-report.sh",):
        artifacts[name] = sha256(ROOT / name)
    ledger_keys = {record["key"] for record in proof["source_ledger"]}
    new_keys = []
    for key in RECORDS:
        record = results[key]
        if key not in ledger_keys:
            proof["source_ledger"].append({"key": key, "source": record["source"], "sha256": record["sha256"]})
            new_keys.append(key)
    proof["source_ledger_count"] = len(proof["source_ledger"])
    stamp = int(datetime.datetime.combine(datetime.date.fromisoformat(REPORT_DATE), datetime.time(),
                                          datetime.timezone.utc).timestamp())
    proof["recorded_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    proof["status"] = "passed_direct_engine_same_session_report_rebuild"
    proof["scope"] = ("Direct numerical engine is the public binary64 default. New same-session interleaved "
                      "TIRust/original-IREGI/COLLIER tables (18 rows, batch 1 and 32) with full-array validation; "
                      "all previous records unchanged.")
    proof["creation_timestamp_utc"] = f"{REPORT_DATE}T00:00:00Z"
    proof["creation_timestamp_unix"] = stamp
    proof["independent_rebuild_command"] = (f"typst compile --root . --creation-timestamp {stamp} docs/benchmark.typ "
                                            f"{independent.relative_to(ROOT) if independent.is_relative_to(ROOT) else independent}")
    proof["byte_identical_rebuild"] = True
    proof["page_count"] = pages
    checkpoint = "direct_engine_checkpoint_20261030" + (f"_capture{capture}" if capture else "")
    proof[checkpoint] = {
        "base_commit": base_commit,
        "previous_proof_sha256": previous_sha,
        "new_result_records": new_keys + ([] if capture else ["direct_engine_report_updated_on"]),
        "campaign": str(HERE.relative_to(ROOT)),
        "note": ("Previous proof was stale for HANDOFF.md at the base commit; all listed artifacts are rehashed "
                 "here. Historical result members are preserved byte-for-byte."),
    }
    PROOF.write_text(json.dumps(proof, indent=1) + "\n")
    print(json.dumps({"ledger": proof["source_ledger_count"], "new": new_keys, "pages": pages}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bind", "seal"))
    parser.add_argument("--independent-pdf", type=Path)
    parser.add_argument("--page-count", type=int)
    parser.add_argument("--base-commit")
    parser.add_argument("--capture", default=None, help="later capture id, e.g. 002")
    args = parser.parse_args()
    if args.action == "bind":
        bind(args.capture)
    else:
        seal(args.independent_pdf.resolve(), args.page_count, args.base_commit, args.capture)


if __name__ == "__main__":
    main()
