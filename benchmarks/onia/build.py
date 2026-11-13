#!/usr/bin/env python3
"""Build the binaries of benchmarks/onia and record their provenance.

TIRust: release examples `onia_driver` and `onia_amplitude` (features native-f64-abc, hardware FMA
as in the benchmark builds; outputs are bit-identical without it). Original
IREGI (binary64): drivers/iregi_onia.f90 linked against the original-IREGI
objects of the benchmark campaigns (target/reference/objects: MadGraph's
vendored IREGI, gfortran -O3, with the one repair of funlib.f90's
factorial_pair, which overflowed 32-bit factorials at high rank; see
README.md). COLLIER 1.2.10 (drivers/collier_scalar.f90, linked against the
unmodified archive of target/reference/collier) supplies scalar C0/D0 for
arbitration. The quadruple-precision IREGI build is used as is
(target/quad-iregi-source-audit/build-v6-001). Binaries go to
target/onia/bin, the record to the file named by the first argument
(default target/onia/build.json). Run from the repository root after
`source scripts/env.sh` (and the Symbolica licence).

Environment: ONIA_TARGET (default target, the directory that holds
reference/, quad-iregi-source-audit/ and receives onia/), ONIA_REFERENCE
(default $ONIA_TARGET/reference), CARGO_TARGET_DIR (default
$ONIA_TARGET/onia-build).
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = Path(os.environ.get("ONIA_TARGET", ROOT / "target"))
OUT = TARGET / "onia"
BIN = OUT / "bin"
FFLAGS = ["-O3", "-std=legacy", "-fallow-argument-mismatch"]
RUSTFLAGS = "-C target-feature=+fma"


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

    reference = Path(os.environ.get("ONIA_REFERENCE", TARGET / "reference"))
    objects = sorted(p for p in (reference / "objects").glob("*.o")
                     if p.name not in {"driver.o", "factorial_table.o", "funlib.o"})
    objects.append(reference / "patched/funlib.o")
    modules = OUT / "iregi-modules"
    modules.mkdir(exist_ok=True)
    iregi = BIN / "iregi-onia"
    driver = HERE / "drivers/iregi_onia.f90"
    record["steps"]["iregi_link_s"] = run(["gfortran", *FFLAGS, "-J", str(modules), "-I", str(reference / "objects"),
                                           str(driver), *map(str, objects), "-o", str(iregi)])
    record["iregi"] = {"binary": str(iregi), "sha256": sha256(iregi), "flags": FFLAGS,
                       "driver_sha256": sha256(driver),
                       "objects": {p.name if p.parent.name == "objects" else f"patched/{p.name}": sha256(p)
                                   for p in objects}}
    collier_source = Path(os.environ.get("ONIA_COLLIER", TARGET / "reference/collier/build-attempt001/source"))
    collier_modules = OUT / "collier-modules"
    collier_modules.mkdir(exist_ok=True)
    collier = BIN / "collier-scalar"
    collier_driver = HERE / "drivers/collier_scalar.f90"
    record["steps"]["collier_link_s"] = run(["gfortran", *FFLAGS, "-J", str(collier_modules), "-I",
                                             str(collier_source / "modules"), str(collier_driver),
                                             str(collier_source / "libcollier.a"), "-o", str(collier)])
    record["collier"] = {"binary": str(collier), "sha256": sha256(collier), "flags": FFLAGS,
                         "driver_sha256": sha256(collier_driver),
                         "archive_sha256": sha256(collier_source / "libcollier.a")}
    quad = TARGET / "quad-iregi-source-audit/build-v6-001/iregi-reference-quad"
    if quad.exists():
        record["quad_reference"] = {"binary": str(quad), "sha256": sha256(quad)}

    target = Path(os.environ.get("CARGO_TARGET_DIR", TARGET / "onia-build"))
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), RUSTFLAGS=RUSTFLAGS)
    cargo = ["cargo", "build", "--release", "--offline", "-p", "iregi-rust", "--example", "onia_driver",
             "--example", "onia_amplitude", "--features", "native-f64-abc"]
    record["steps"]["tirust_cargo_s"] = run(cargo, env=env)
    tirust = BIN / "tirust-onia"
    shutil.copy2(target / "release/examples/onia_driver", tirust)
    amplitude = BIN / "tirust-onia-amplitude"
    shutil.copy2(target / "release/examples/onia_amplitude", amplitude)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record["tirust_amplitude"] = {"binary": str(amplitude), "sha256": sha256(amplitude)}
    record["tirust"] = {"binary": str(tirust), "sha256": sha256(tirust), "rustflags": RUSTFLAGS,
                        "git_head": git("rev-parse", "HEAD"),
                        "tracked_changes": [l[3:] for l in git("status", "--porcelain", "--untracked-files=no").splitlines()],
                        "cargo": " ".join(cargo)}
    name = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "build.json"
    name.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
