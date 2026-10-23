#!/usr/bin/env python3
"""Package the pinned owners' unchanged rendering assets for the standalone host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

FEYNKIT_REVISION = "865afdc43fb4e6b950439d34b4ec2c38352873e7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feynkit-root", type=Path, required=True)
    parser.add_argument("--typst-assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("python/tirust/data"))
    args = parser.parse_args()
    typst_package = tomllib.loads((args.typst_assets_root / "Cargo.toml").read_text())["package"]
    if (typst_package["name"], typst_package["version"]) != ("typst-assets", "0.15.1"):
        raise SystemExit("rendering fonts must come from the locked typst-assets 0.15.1")
    revision = subprocess.check_output(
        ["git", "-C", str(args.feynkit_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != FEYNKIT_REVISION:
        raise SystemExit("FeynKit assets must match the project's locked owner revision")
    entries = []
    manifests = []
    for owner in ["typst-renderer", "spynso3"]:
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
            source = (args.typst_assets_root if entry.get("package") == "typst-assets" else crate) / entry["path"]
            payload = source.read_bytes()
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            entries.append({"name": str(name), "owner": owner, "constant": entry["constant"], "owner_source": str(source), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    licenses = {
        "typst-assets-LICENSE": args.typst_assets_root / "LICENSE",
        "typst-fonts-NOTICE": args.feynkit_root / "examples/notebooks/symbolica-host/licenses/typst-fonts-NOTICE",
        "STIX-Two-OFL.txt": args.feynkit_root / "crates/spynso3/typst/STIX-Two-OFL.txt",
        "FeynKit-LICENSE": args.feynkit_root / "LICENSE",
        "Symbolica-LICENSE.md": Path("dependencies/symbolica/License.md"),
    }
    license_entries = []
    for name, source in licenses.items():
        target = args.output / "licenses" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        license_entries.append({"name": "licenses/" + name, "owner_source": str(source), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {
        "feynkit_revision": revision,
        "typst_assets_version": "0.15.1",
        "scope": "unchanged pinned rendering fonts; no compiled library or license credential",
        "owner_manifests": manifests,
        "assets": entries,
        "licenses": license_entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(entries)} unchanged assets, {sum(entry['bytes'] for entry in entries)} bytes")


if __name__ == "__main__":
    main()
