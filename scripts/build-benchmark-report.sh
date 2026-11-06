#!/usr/bin/env bash
set -euo pipefail

# Build the slim benchmark summary docs/benchmark.pdf from
# docs/benchmark-summary.typ and its data docs/benchmark-summary.json (written
# by scripts/build-benchmark-summary.py). Run from the repository root.
#
# The output is reproducible: the creation timestamp is fixed by the data
# (midnight UTC of the report date that the summary prints, i.e. the date the
# timing capture finished) and only Typst's embedded fonts are used, so
# identical source and data give a byte-identical PDF.
#
# --check builds into target/benchmark-report-check/ instead and fails unless
# the result equals the committed docs/benchmark.pdf byte for byte and stays
# within the 1 MiB blob-size policy (docs/PYAMPLICOL-COLLABORATION.md section 5).
source scripts/env.sh
report_creation_timestamp="$(python3 - <<'PY'
import datetime
import json
from pathlib import Path

summary = json.loads(Path('docs/benchmark-summary.json').read_text())
finished = summary['captures']['timing']['environment']['finished']
date = datetime.date.fromisoformat(finished[:10])
print(int(datetime.datetime.combine(date, datetime.time(), datetime.timezone.utc).timestamp()))
PY
)"
output=docs/benchmark.pdf
if [[ "${1:-}" == --check ]]; then
  mkdir -p target/benchmark-report-check
  output=target/benchmark-report-check/benchmark.pdf
elif [[ $# -gt 0 ]]; then
  echo 'usage: scripts/build-benchmark-report.sh [--check]' >&2
  exit 2
fi
typst compile --ignore-system-fonts --creation-timestamp "$report_creation_timestamp" \
  docs/benchmark-summary.typ "$output"
size=$(wc -c < "$output")
if (( size > 1048576 )) && ! grep -qxF docs/benchmark.pdf .blob-size-allowlist; then
  echo "$output is $size bytes, above the 1 MiB blob-size policy (.blob-size-allowlist)" >&2
  exit 1
fi
if [[ "$output" != docs/benchmark.pdf ]]; then
  cmp "$output" docs/benchmark.pdf
  echo "docs/benchmark.pdf is reproducible ($size bytes)"
fi
