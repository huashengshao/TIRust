#!/usr/bin/env python3
"""Copy upstream-owned stubs into a private, type-only standalone namespace.

This does not generate Python runtime packages, import Symbolica, run numerical
code, or edit source owners. Input files and every mechanical namespace rewrite
are recorded for review. Regenerate only against the project's actual pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolica-stub", type=Path, required=True)
    parser.add_argument("--feynkit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replacements = [
        ("symbolica.community.hepkit", "tirust._types.symbolica.community.hepkit"),
        ("symbolica.community.tensor", "tirust._types.symbolica.community.tensor"),
        ("symbolica.community.graph", "tirust._types.symbolica.community.graph"),
        ("symbolica.core", "tirust._types.symbolica.core"),
    ]
    sources = {
        "symbolica/core.pyi": args.symbolica_stub,
        "symbolica/community/hepkit/__init__.pyi": args.feynkit_root / "crates/feynkit-py/python/symbolica/community/hepkit/__init__.pyi",
        "symbolica/community/tensor/__init__.pyi": args.feynkit_root / "docs/api/python/spynso3.pyi",
        "symbolica/community/graph/__init__.pyi": args.feynkit_root / "crates/linnet-render-py/python/symbolica/community/graph/__init__.pyi",
    }
    manifest = {"purpose": "private type-only mirrors; no runtime shadow packages", "symbolica_base_revision": "9cdc2949", "feynkit_revision": "865afdc43fb4e6b950439d34b4ec2c38352873e7", "files": []}
    for relative, source in sources.items():
        original = source.read_bytes()
        text = original.decode()
        # Rewrite longer owner paths first; plain owner imports are handled last.
        for old, new in replacements:
            text = text.replace(old, new)
        text = text.replace("from symbolica import ", "from tirust._types.symbolica.core import ")
        text = text.replace("import symbolica\n", "import tirust._types.symbolica as symbolica\n")
        # FeynKit's pinned generated stub names an optional provider facade that
        # is not distributed in its owner repository or this standalone host.
        # Retain that accurate owner annotation but make its unavailable import
        # explicit; do not invent substitute provider classes.
        optional = "import tirust._types.symbolica.community.hepkit.sector_decomposition"
        text = text.replace(optional + "\n", optional + "  # type: ignore[import-not-found]  # optional provider, not part of this host\n")
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = ("# Type-only upstream mirror; provenance: ../owner-stubs.json.\n" + text).encode()
        target.write_bytes(payload)
        manifest["files"].append({"owner_source": str(source), "owner_sha256": hashlib.sha256(original).hexdigest(), "output": relative, "sha256": hashlib.sha256(payload).hexdigest()})
    for relative, text in {
        "__init__.pyi": '"""Private, type-only owner aliases; no runtime package exists."""\n',
        "symbolica/__init__.pyi": "from .core import *\n",
        "symbolica/community/__init__.pyi": "from . import graph as graph, hepkit as hepkit, tensor as tensor\n",
    }.items():
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (args.output.parent / "owner-stubs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
