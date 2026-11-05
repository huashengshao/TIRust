#!/usr/bin/env python3
"""Append the stability study (and the coplanar rescue evidence) to the
report data and reseal the report proof, as direct-engine-20261030/publish.py.

bind: append new top-level members to docs/data/results.json without touching
      the raw spelling of existing members (append-only).
seal: after rebuilding docs/benchmark.pdf and an independent byte-identical
      copy, rehash every listed report artifact, add the new records to the
      source ledger and record a checkpoint keeping the previous proof hash.

usage: publish.py bind [--capture 002]
       publish.py seal --independent-pdf PATH --page-count N --base-commit SHA [--capture 002]
Run from the repository root.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "docs/data/results.json"
PROOF = ROOT / "docs/data/report-validation.json"
REPORT_DATE = "2026-10-30"
RECORDS = {
    "stability_20261030": (HERE / "summary.json", True),
    "stability_20261030_build": (HERE / "build.json", True),
    "direct_engine_20261030_rescue_coplanar": (
        ROOT / "validation/benchmark/direct-engine-20261030/rescue-coplanar.jsonl", False),
}
# Later captures append their own records (results.json is append-only).
LATER = {
    "002": {
        "stability_20261030_capture002": (HERE / "summary-002.json", True),
        "stability_20261030_build002": (HERE / "build-002.json", True),
    },
    "003": {
        "stability_20261030_capture003": (HERE / "summary-003.json", True),
        "stability_20261030_build003": (HERE / "build-003.json", True),
    },
    "004": {
        "stability_20261030_capture004": (HERE / "summary-004.json", True),
        "stability_20261030_build004": (HERE / "build-004.json", True),
    },
    "005": {
        "stability_20261030_capture005": (HERE / "summary-005.json", True),
        "stability_20261030_build005": (HERE / "build-005.json", True),
    },
    "006": {
        "stability_20261030_capture006": (HERE / "summary-006.json", True),
        "stability_20261030_build006": (HERE / "build-006.json", True),
    },
    "007": {
        "stability_20261030_capture007": (HERE / "summary-007.json", True),
        "stability_20261030_build007": (HERE / "build-007.json", True),
    },    "008": {
        "stability_20261030_capture008": (HERE / "summary-008.json", True),
        "stability_20261030_build008": (HERE / "build-008.json", True),
    },    "009": {
        "stability_20261030_capture009": (HERE / "summary-009.json", True),
        "stability_20261030_build009": (HERE / "build-009.json", True),
    },
    "010": {
        "stability_20261030_capture010": (HERE / "summary-010.json", True),
        "stability_20261030_build010": (HERE / "build-010.json", True),
    },
    "011": {
        "stability_20261030_capture011": (HERE / "summary-011.json", True),
        "stability_20261030_build011": (HERE / "build-011.json", True),
    },
    "012": {
        "stability_20261030_capture012": (HERE / "summary-012.json", True),
        "stability_20261030_build012": (HERE / "build-012.json", True),
    },
    "013": {
        "stability_20261030_capture013": (HERE / "summary-013.json", True),
        "stability_20261030_build013": (HERE / "build-013.json", True),
    },
    # Capture 014 (32a79e2) was not published.
    "015": {
        "stability_20261030_capture015": (HERE / "summary-015.json", True),
        "stability_20261030_build015": (HERE / "build-015.json", True),
    },
    "016": {
        "stability_20261030_capture016": (HERE / "summary-016.json", True),
        "stability_20261030_build016": (HERE / "build-016.json", True),
    },
    "017": {
        "stability_20261030_capture017": (HERE / "summary-017.json", True),
        "stability_20261030_build017": (HERE / "build-017.json", True),
    },
    "018": {
        "stability_20261030_capture018": (HERE / "summary-018.json", True),
        "stability_20261030_build018": (HERE / "build-018.json", True),
    },
    "019": {
        "stability_20261030_capture019": (HERE / "summary-019.json", True),
        "stability_20261030_build019": (HERE / "build-019.json", True),
    },
    "020": {
        "stability_20261030_capture020": (HERE / "summary-020.json", True),
        "stability_20261030_build020": (HERE / "build-020.json", True),
    },
    "021": {
        "stability_20261030_capture021": (HERE / "summary-021.json", True),
        "stability_20261030_build021": (HERE / "build-021.json", True),
    },
    "022": {
        "stability_20261030_capture022": (HERE / "summary-022.json", True),
        "stability_20261030_build022": (HERE / "build-022.json", True),
    },
    "023": {
        "stability_20261030_capture023": (HERE / "summary-023.json", True),
        "stability_20261030_build023": (HERE / "build-023.json", True),
    },
    "024": {
        "stability_20261030_capture024": (HERE / "summary-024.json", True),
        "stability_20261030_build024": (HERE / "build-024.json", True),
    },
    "025": {
        "stability_20261030_capture025": (HERE / "summary-025.json", True),
        "stability_20261030_build025": (HERE / "build-025.json", True),
    },
    "026": {
        "stability_20261030_capture026": (HERE / "summary-026.json", True),
        "stability_20261030_build026": (HERE / "build-026.json", True),
    },
    "027": {
        "stability_20261030_capture027": (HERE / "summary-027.json", True),
        "stability_20261030_build027": (HERE / "build-027.json", True),
    },
    "028": {
        "stability_20261030_capture028": (HERE / "summary-028.json", True),
        "stability_20261030_build028": (HERE / "build-028.json", True),
    },
    "029": {
        "stability_20261030_capture029": (HERE / "summary-029.json", True),
        "stability_20261030_build029": (HERE / "build-029.json", True),
    },
    "030": {
        "stability_20261030_capture030": (HERE / "summary-030.json", True),
        "stability_20261030_build030": (HERE / "build-030.json", True),
    },
    "031": {
        "stability_20261030_capture031": (HERE / "summary-031.json", True),
        "stability_20261030_build031": (HERE / "build-031.json", True),
    },
    "032": {
        "stability_20261030_capture032": (HERE / "summary-032.json", True),
        "stability_20261030_build032": (HERE / "build-032.json", True),
    },
    "033": {
        "stability_20261030_capture033": (HERE / "summary-033.json", True),
        "stability_20261030_build033": (HERE / "build-033.json", True),
    },
    "034": {
        "stability_20261030_capture034": (HERE / "summary-034.json", True),
        "stability_20261030_build034": (HERE / "build-034.json", True),
    },
    "035": {
        "stability_20261030_capture035": (HERE / "summary-035.json", True),
        "stability_20261030_build035": (HERE / "build-035.json", True),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(capture=None) -> None:
    records = LATER[capture] if capture else RECORDS
    text = RESULTS.read_text()
    existing = json.loads(text)
    additions = {}
    for key, (path, snapshot) in records.items():
        if key in existing:
            raise SystemExit(f"{key} already bound; results.json is append-only")
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
    records = LATER[capture] if capture else RECORDS
    proof = json.loads(PROOF.read_text())
    previous_sha = sha256(PROOF)
    pdf = ROOT / "docs/benchmark.pdf"
    if pdf.read_bytes() != independent.read_bytes():
        raise SystemExit("independent rebuild differs from docs/benchmark.pdf")
    results = json.loads(RESULTS.read_text())
    for name in list(proof["report_artifacts"]):
        proof["report_artifacts"][name] = sha256(ROOT / name)
    ledger_keys = {record["key"] for record in proof["source_ledger"]}
    new_keys = []
    for key in records:
        record = results[key]
        if key not in ledger_keys:
            proof["source_ledger"].append({"key": key, "source": record["source"], "sha256": record["sha256"]})
            new_keys.append(key)
    proof["source_ledger_count"] = len(proof["source_ledger"])
    stamp = int(datetime.datetime.combine(datetime.date.fromisoformat(REPORT_DATE), datetime.time(),
                                          datetime.timezone.utc).timestamp())
    proof["recorded_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    proof["status"] = "passed_stability_study_report_rebuild"
    proof["scope"] = ("Binary64 stability study (TIRust, original IREGI, COLLIER against a confirmed "
                      "high-precision reference) and the direct-engine precision rescue; all previous records unchanged.")
    proof["creation_timestamp_utc"] = f"{REPORT_DATE}T00:00:00Z"
    proof["creation_timestamp_unix"] = stamp
    rel = independent.relative_to(ROOT) if independent.is_relative_to(ROOT) else independent
    proof["independent_rebuild_command"] = f"typst compile --root . --creation-timestamp {stamp} docs/benchmark.typ {rel}"
    proof["byte_identical_rebuild"] = True
    proof["page_count"] = pages
    proof["stability_checkpoint_20261030" + (f"_capture{capture}" if capture else "")] = {
        "base_commit": base_commit,
        "previous_proof_sha256": previous_sha,
        "new_result_records": new_keys,
        "campaign": str(HERE.relative_to(ROOT)),
        "note": "All listed artifacts rehashed; historical result members preserved byte-for-byte.",
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
