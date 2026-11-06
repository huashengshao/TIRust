#!/usr/bin/env python3
"""Build and freeze the two drivers of the amplitude benchmark.

TIRust: release example `amplitude_timing` (thin LTO, one codegen unit), the
public `AmplitudeWorkspace` with per-subset sharing on or off.
COLLIER: `collier_amplitude.f90` linked against the unmodified COLLIER 1.2.10
archive of the direct-engine campaign, with its global cache on or off.
Binaries go to target/amplitude-20261102/bin; their SHA-256, the toolchains
and TIRust's commit are recorded in the build record named by the argument.
Run from the repository root after `source scripts/env.sh`.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# TIRust's benchmark binaries are built with hardware FMA (Huasheng's decision,
# 6 Nov 2026; huashengshao/TIRust#5): outputs are bit-identical with and without it
# (the double-double arithmetic needs a correctly rounded fma either way, and Rust
# never contracts a*b+c), and its own target directory keeps the default build
# (tests, library, pyAmpliCol's baseline x86-64 builds) free of the flag. The
# reference codes keep their flags.
TIRUST_RUSTFLAGS = "-C target-feature=+fma"
TIRUST_TARGET = "target/fma-build"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BIN = ROOT / "target/amplitude-20261102/bin"
FFLAGS = ["-O3", "-std=legacy", "-fallow-argument-mismatch", "-Wtabs"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, **kwargs):
    started = time.monotonic()
    subprocess.run(argv, check=True, cwd=ROOT, **kwargs)
    return round(time.monotonic() - started, 3)


def main() -> None:
    record_name = sys.argv[1]
    BIN.mkdir(parents=True, exist_ok=True)
    record = {"host": platform.node(), "machine": platform.machine(), "steps": {}}
    record["toolchains"] = {
        "gfortran": subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
    }
    collier_source = ROOT / "target/reference/collier/build-attempt001/source"
    modules = ROOT / "target/amplitude-20261102/collier"
    modules.mkdir(parents=True, exist_ok=True)
    collier = BIN / "collier-amplitude"
    driver = HERE / "collier_amplitude.f90"
    record["steps"]["collier_link_s"] = run(["gfortran", *FFLAGS, "-J", str(modules), "-I", str(collier_source / "modules"),
                                             str(driver), str(collier_source / "libcollier.a"), "-o", str(collier)])
    record["collier"] = {"binary": str(collier.relative_to(ROOT)), "sha256": sha256(collier), "flags": FFLAGS,
                         "driver_sha256": sha256(driver), "archive_sha256": sha256(collier_source / "libcollier.a")}
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / TIRUST_TARGET), RUSTFLAGS=TIRUST_RUSTFLAGS)
    cargo = ["cargo", "build", "--release", "--offline", "-p", "iregi-rust", "--example", "amplitude_timing",
             "--features", "native-f64-abc"]
    record["steps"]["tirust_cargo_s"] = run(cargo, env=env)
    tirust = BIN / "tirust-amplitude-timing"
    shutil.copy2(ROOT / TIRUST_TARGET / "release/examples/amplitude_timing", tirust)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "crates", "Cargo.toml", "Cargo.lock"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    record["tirust"] = {"rustflags": TIRUST_RUSTFLAGS, "binary": str(tirust.relative_to(ROOT)), "sha256": sha256(tirust), "cargo": " ".join(cargo),
                        "profile": "release (thin LTO, codegen-units=1)", "git_head": head,
                        "source_tree_dirty": bool(dirty)}
    (HERE / record_name).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k]["sha256"] for k in ("tirust", "collier")}, indent=1))


if __name__ == "__main__":
    main()
