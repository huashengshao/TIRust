#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Derive a fixed UTC timestamp from the report
# date so identical source and data produce byte-identical PDF output.
source scripts/env.sh
report_creation_timestamp="$(python3 - <<'PY'
import datetime
import json
from pathlib import Path

report = json.loads(Path('docs/data/results.json').read_text())
date = datetime.date.fromisoformat(report.get('rank6_current_report_updated_on', report.get('report_updated_on', report['date'])))
print(int(datetime.datetime.combine(date, datetime.time(), datetime.timezone.utc).timestamp()))
PY
)"
typst compile --creation-timestamp "$report_creation_timestamp" docs/benchmark.typ docs/benchmark.pdf
