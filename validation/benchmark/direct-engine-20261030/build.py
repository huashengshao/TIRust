#!/usr/bin/env python3
"""Build and freeze the three timing drivers of the direct-engine campaign.

TIRust: release example `direct_timing` (thin LTO, one codegen unit).
IREGI: `drivers/iregi_general.f90` linked against the existing original-IREGI
objects (target/reference, with the documented factorial_pair repair).
COLLIER: `drivers/collier_batch.f90` linked against the unmodified COLLIER
1.2.10 archive used by the opening campaigns. All binaries are copied to
target/direct-engine-20261030/bin and their SHA-256 recorded in build.json.
Run from the repository root after `source scripts/env.sh`; an optional
argument names the build record (default build.json; later captures use
build-NNN.json so earlier bound records stay unchanged).
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
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
ROOT = HERE.parents[2]
OUT = ROOT / "target/direct-engine-20261030"
BIN = OUT / "bin"
FFLAGS = ["-O3", "-std=legacy", "-fallow-argument-mismatch"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, **kwargs):
    started = time.monotonic()
    subprocess.run(argv, check=True, cwd=ROOT, **kwargs)
    return round(time.monotonic() - started, 3)


def main() -> None:
    import sys
    record_name = sys.argv[1] if len(sys.argv) > 1 else "build.json"
    BIN.mkdir(parents=True, exist_ok=True)
    record = {"host": platform.node(), "machine": platform.machine(), "steps": {}}
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    record["toolchains"] = {"gfortran": gfortran, "rustc": rustc}

    reference = ROOT / "target/reference"
    objects = sorted(p for p in (reference / "objects").glob("*.o")
                     if p.name not in {"driver.o", "factorial_table.o", "funlib.o"})
    objects.append(reference / "patched/funlib.o")
    iregi_dir = OUT / "iregi"
    iregi_dir.mkdir(exist_ok=True)
    iregi = BIN / "iregi-general"
    record["steps"]["iregi_link_s"] = run(["gfortran", *FFLAGS, "-J", str(iregi_dir), "-I", str(reference / "objects"),
                                           str(HERE / "drivers/iregi_general.f90"), *map(str, objects), "-o", str(iregi)])
    record["iregi"] = {"binary": str(iregi.relative_to(ROOT)), "sha256": sha256(iregi), "flags": FFLAGS,
                       "driver_sha256": sha256(HERE / "drivers/iregi_general.f90"),
                       "objects": {str(p.relative_to(ROOT)): sha256(p) for p in objects}}

    collier_source = ROOT / "target/reference/collier/build-attempt001/source"
    collier_dir = OUT / "collier"
    collier_dir.mkdir(exist_ok=True)
    collier = BIN / "collier-batch"
    record["steps"]["collier_link_s"] = run(["gfortran", *FFLAGS, "-Wtabs", "-J", str(collier_dir), "-I",
                                             str(collier_source / "modules"), str(HERE / "drivers/collier_batch.f90"),
                                             str(collier_source / "libcollier.a"), "-o", str(collier)])
    record["collier"] = {"binary": str(collier.relative_to(ROOT)), "sha256": sha256(collier), "flags": FFLAGS + ["-Wtabs"],
                         "driver_sha256": sha256(HERE / "drivers/collier_batch.f90"),
                         "archive_sha256": sha256(collier_source / "libcollier.a")}

    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / TIRUST_TARGET), RUSTFLAGS=TIRUST_RUSTFLAGS)
    cargo = ["cargo", "build", "--release", "--offline", "-p", "iregi-rust", "--example", "direct_timing",
             "--features", "native-f64-abc"]
    record["steps"]["tirust_cargo_s"] = run(cargo, env=env)
    built = ROOT / TIRUST_TARGET / "release/examples/direct_timing"
    tirust = BIN / "tirust-direct-timing"
    shutil.copy2(built, tirust)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "crates", "Cargo.toml", "Cargo.lock"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    record["tirust"] = {"rustflags": TIRUST_RUSTFLAGS, "binary": str(tirust.relative_to(ROOT)), "sha256": sha256(tirust), "cargo": " ".join(cargo),
                        "profile": "release (thin LTO, codegen-units=1)", "git_head": head,
                        "source_tree_dirty": bool(dirty)}
    ninjars = ROOT / "target/stability-20261030/bin/ninjars"
    if ninjars.exists():
        record["ninjars"] = {"binary": str(ninjars.relative_to(ROOT)), "sha256": sha256(ninjars),
                             "provenance": "validation/benchmark/stability-20261030/build-003.json (ninjars)"}
    (HERE / record_name).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k]["sha256"] for k in ("tirust", "iregi", "collier")}, indent=1))


if __name__ == "__main__":
    main()
