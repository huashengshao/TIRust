#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f validation/reference/build.py ]]; then
  echo 'Run scripts/build-reference.sh from the iregi-rust repository root.' >&2
  exit 2
fi
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
  else
    PYTHON=/nix/store/0r6k8xa2kgqyp3r4v2w7yrb80ma2iawm-python3-3.13.12/bin/python3
  fi
fi
exec "$PYTHON" validation/reference/build.py "$@"
