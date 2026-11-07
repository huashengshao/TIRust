#!/usr/bin/env python3
"""Package the pinned owners' unchanged rendering assets for the standalone host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

FEYNKIT_REVISION = "9102f885722e856e30227ed46e685e5c465905f7"
# Owners whose data.json lists the rendering assets the standalone host bundles.
# spynso3 re-exports linnet-render-py's assets; no owner needs typst-assets fonts.
OWNERS = ["linnet-render-py"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feynkit-root", type=Path, required=True)
    parser.add_argument(
        "--symbolica-root", type=Path, required=True,
        help="checkout of the pinned Symbolica revision (the workspace [patch.crates-io] entry)",
    )
    parser.add_argument("--output", type=Path, default=Path("python/tirust/data"))
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.feynkit_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != FEYNKIT_REVISION:
        raise SystemExit("FeynKit assets must match the project's locked owner revision")
    entries = []
    manifests = []
    for owner in OWNERS:
        crate = args.feynkit_root / "crates" / owner
        manifest_path = crate / "data.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        if manifest["schema"] != 1:
            raise SystemExit("unsupported owner data schema")
        manifests.append({"owner": owner, "sha256": hashlib.sha256(original).hexdigest()})
        for entry in manifest["assets"]:
            name = Path(entry["name"])
            if name.is_absolute() or ".." in name.parts or not str(name).startswith("render/fonts/"):
                raise SystemExit("unexpected asset location in pinned owner manifest")
            if "package" in entry:
                raise SystemExit("owner asset from another package; extend this script before bundling it")
            source = crate / entry["path"]
            payload = source.read_bytes()
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            entries.append({"name": str(name), "owner": owner, "constant": entry["constant"], "owner_source": str(source), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    licenses = {
        "STIX-Two-OFL.txt": args.feynkit_root / "crates/linnet-render-py/fonts/STIX-Two-OFL.txt",
        "FeynKit-LICENSE": args.feynkit_root / "LICENSE",
        "Symbolica-LICENSE.md": args.symbolica_root / "License.md",
    }
    license_entries = []
    for name, source in licenses.items():
        target = args.output / "licenses" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        license_entries.append({"name": "licenses/" + name, "owner_source": str(source), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {
        "feynkit_revision": revision,
        "scope": "unchanged pinned rendering fonts; no compiled library or license credential",
        "owner_manifests": manifests,
        "assets": entries,
        "licenses": license_entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(entries)} unchanged assets, {sum(entry['bytes'] for entry in entries)} bytes")


if __name__ == "__main__":
    main()
