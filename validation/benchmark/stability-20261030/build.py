#!/usr/bin/env python3
"""Build and freeze the binaries of the binary64 stability study.

TIRust: release example `stability_driver` (same features as the timing
driver of direct-engine-20261030). COLLIER: `drivers/collier_physical.f90`
linked against the unmodified COLLIER 1.2.10 archive; original IREGI
(binary64): `drivers/iregi_physical.f90` linked against the original-IREGI
objects of the timing campaigns (same repair). NinjaRS: the reference build under target/reference/ninjars (built
separately), copied and hashed. The quad reference is the quadruple-precision original IREGI build
target/quad-iregi-source-audit/build-v6-001. Binaries go to
target/stability-20261030/bin; SHA-256 sums, toolchains and the TIRust git
head are recorded in build.json (or the record named by the first argument,
e.g. build-002.json, so bound records stay unchanged). Run from the
repository root after `source scripts/env.sh`.
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
OUT = ROOT / "target/stability-20261030"
BIN = OUT / "bin"
FFLAGS = ["-O3", "-std=legacy", "-fallow-argument-mismatch"]
QUAD = ROOT / "target/quad-iregi-source-audit/build-v6-001/iregi-reference-quad"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, **kwargs):
    started = time.monotonic()
    subprocess.run(argv, check=True, cwd=ROOT, **kwargs)
    return round(time.monotonic() - started, 3)


def main() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    record = {"host": platform.node(), "machine": platform.machine(), "steps": {}}
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    record["toolchains"] = {"gfortran": gfortran, "rustc": rustc}

    collier_source = ROOT / "target/reference/collier/build-attempt001/source"
    collier_dir = OUT / "collier"
    collier_dir.mkdir(exist_ok=True)
    collier = BIN / "collier-physical"
    driver = HERE / "drivers/collier_physical.f90"
    record["steps"]["collier_link_s"] = run(["gfortran", *FFLAGS, "-Wtabs", "-J", str(collier_dir), "-I",
                                             str(collier_source / "modules"), str(driver),
                                             str(collier_source / "libcollier.a"), "-o", str(collier)])
    record["collier"] = {"binary": str(collier.relative_to(ROOT)), "sha256": sha256(collier),
                         "flags": FFLAGS + ["-Wtabs"], "driver_sha256": sha256(driver),
                         "archive_sha256": sha256(collier_source / "libcollier.a")}
    reference = ROOT / "target/reference"
    objects = sorted(p for p in (reference / "objects").glob("*.o")
                     if p.name not in {"driver.o", "factorial_table.o", "funlib.o"})
    objects.append(reference / "patched/funlib.o")
    iregi_dir = OUT / "iregi"
    iregi_dir.mkdir(exist_ok=True)
    iregi = BIN / "iregi-physical"
    iregi_driver = HERE / "drivers/iregi_physical.f90"
    record["steps"]["iregi_link_s"] = run(["gfortran", *FFLAGS, "-J", str(iregi_dir), "-I", str(reference / "objects"),
                                           str(iregi_driver), *map(str, objects), "-o", str(iregi)])
    record["iregi"] = {"binary": str(iregi.relative_to(ROOT)), "sha256": sha256(iregi), "flags": FFLAGS,
                       "driver_sha256": sha256(iregi_driver),
                       "objects": {str(p.relative_to(ROOT)): sha256(p) for p in objects}}
    record["quad_reference"] = {"binary": str(QUAD.relative_to(ROOT)), "sha256": sha256(QUAD),
                                "provenance": "target/quad-iregi-source-audit/build-v6-001/build.json"}

    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / TIRUST_TARGET), RUSTFLAGS=TIRUST_RUSTFLAGS)
    cargo = ["cargo", "build", "--release", "--offline", "-p", "iregi-rust", "--example", "stability_driver",
             "--features", "native-f64-abc"]
    record["steps"]["tirust_cargo_s"] = run(cargo, env=env)
    built = ROOT / TIRUST_TARGET / "release/examples/stability_driver"
    tirust = BIN / "tirust-stability"
    shutil.copy2(built, tirust)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.rstrip("\n")
    changed = [line[3:] for line in dirty.splitlines()]
    record["tirust"] = {"rustflags": TIRUST_RUSTFLAGS, "binary": str(tirust.relative_to(ROOT)), "sha256": sha256(tirust), "git_head": head,
                        "tracked_changes": changed,
                        "source_changes": [c for c in changed if c.startswith(("crates/", "dependencies/", "Cargo"))],
                        "cargo": " ".join(cargo)}
    # NinjaRS reference build (target/reference/ninjars, built separately with
    # `cargo build --profile perf --locked -p ninjars-cli`): copied and hashed.
    ninjars_source = ROOT / "target/reference/ninjars/source"
    ninjars_built = ROOT / "target/reference/ninjars/target/perf/ninjars"
    if ninjars_built.exists():
        ninjars = BIN / "ninjars"
        shutil.copy2(ninjars_built, ninjars)
        git = lambda *a: subprocess.run(["git", "-C", str(ninjars_source), *a], capture_output=True,
                                        text=True).stdout.strip()
        record["ninjars"] = {"binary": str(ninjars.relative_to(ROOT)), "sha256": sha256(ninjars),
                             "repository": "https://github.com/alphal00p/ninjars",
                             "git_head": git("rev-parse", "HEAD"), "upstream_base": git("rev-parse", "origin/main"),
                             "repairs": git("log", "--format=%h %s", "origin/main..HEAD").splitlines(),
                             "tracked_changes": [line[3:] for line in git("status", "--porcelain",
                                                                          "--untracked-files=no").splitlines()],
                             "cargo": "cargo build --profile perf --locked -p ninjars-cli"}
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "build.json"
    (HERE / name).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
