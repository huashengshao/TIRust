#!/usr/bin/env python3
"""Recreate the patched Symbolica source from its pinned revision."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != root:
        raise SystemExit(f"Run this script from {root}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Git checkout containing the pinned revision")
    parser.add_argument("--output", type=Path, default=Path("target/symbolica-scoped-cache"))
    parser.add_argument(
        "--vendor", action="store_true",
        help="Reproduce the retained production files and their provenance manifest",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise SystemExit("The output must be a new directory inside iregi-rust")
    series = root / "dependencies/symbolica-patches"
    provenance = json.loads((series / "provenance.json").read_text())
    revision = provenance["upstream_revision"]
    patches = provenance.get("patches", [{
        "path": provenance["patch"], "sha256": provenance["patch_sha256"]
    }])
    for item in patches:
        digest = hashlib.sha256((series / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit(f"Patch hash mismatch: {item['path']}")
    candidates = [args.source] if args.source else [
        *Path(".local/cargo-home/git/checkouts").glob("symbolica-*/9cdc294"),
        Path("../references/symbolica"),
    ]
    upstream = None
    for candidate in candidates:
        resolved = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", f"{revision}^{{commit}}"],
            text=True, capture_output=True,
        )
        if resolved.returncode == 0 and resolved.stdout.strip() == revision:
            upstream = candidate
            break
    if upstream is None:
        raise SystemExit("No checkout contains the pinned revision; specify one with --source")
    archive = subprocess.check_output(
        ["git", "-C", str(upstream), "archive", revision]
    )
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(output, filter="data")
    for item in patches:
        subprocess.run(
            ["git", "apply", f"--directory={output.relative_to(root)}", str(series / item["path"])],
            check=True,
        )
    expected = provenance.get("patched_files", {
        "src/evaluate/tree.rs": provenance["patched_file_sha256"]
    })
    for name, digest in expected.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Patched source hash mismatch: {name}")
    if args.vendor:
        vendor = root / "dependencies/symbolica"
        inventory = json.loads((vendor / "IREGI_PROVENANCE.json").read_text())
        if inventory["upstream_revision"] != revision or inventory["patches"] != patches:
            raise SystemExit("Vendor inventory does not match the pinned patch series")
        retained = {item["path"]: item for item in inventory["files"]}
        for path in list(output.rglob("*")):
            if path.is_file() and str(path.relative_to(output)) not in retained:
                path.unlink()
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        for name, item in retained.items():
            contents = (output / name).read_bytes()
            if len(contents) != item["bytes"] or hashlib.sha256(contents).hexdigest() != item["sha256"]:
                raise SystemExit(f"Vendored source hash mismatch: {name}")
        for name in ("IREGI_PROVENANCE.json", "IREGI_README.md"):
            shutil.copyfile(vendor / name, output / name)
    print(f"Prepared {output.relative_to(root)} from {revision} plus {len(patches)} patches")


if __name__ == "__main__":
    main()
