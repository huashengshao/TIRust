# Development environment

Run commands from the `iregi-rust` checkout. The portable entrypoint is
`nix-shell`, using pinned nixpkgs revision
`5adc50831c259f87faeb09ddf930d43597197473` for the C/Fortran toolchain, Python and
Typst, and rust-overlay `1f69c7203034a4a974d7a4eac0f19b54af30804d` for Rust/Cargo
1.97.1 as pinned by `rust-toolchain.toml` (the toolchain shared with pyAmpliCol;
rustup reads the same file). The workspace's `rust-version` is 1.96.
The shell supplies Rust, C/C++ and Fortran compilers, GMP/MPFR/MPC, Python,
Typst, and reference-build utilities. Rust is production code; Fortran and
Python are validation tools only.

On a machine that already has these tools installed, use:

```sh
source scripts/env.sh
scripts/cargo.sh test
```

This tests the default members, which are every crate except the Python
extension `crates/iregi-python`. That extension does not build at the moment.
Its GammaLoop crates (gammaloop `865afdc4`) call Symbolica's Python
`evaluator_multiple` with 17 arguments, but the shared Symbolica revision
`20a9da94` takes 18 (E0061 in `spynso3`). So `--workspace` and the wheel build
fail until the GammaLoop pin moves to a revision built for that Symbolica. The
pyAmpliCol coordination record accepts this known break of a non-core crate
(DECISIONS 2026-11-06 ~19:45), and the pyAmpliCol maintainers own that pin
move, which they carry out with their combined Symbolica re-pin:

- pyAmpliCol's own GammaLoop pin `d1904bf2` is older than `865afdc4` and makes
  the same 17-argument call. The first GammaLoop revision with the 18-argument
  call is `3f0df2a8` (alphal00p/gammaloop#209).
- `iregi-python` passes FeynKit graph types into `iregi-prepare`, and the
  workspace patch of `linnest` applies to every GammaLoop revision, so the
  extension's GammaLoop crates cannot move alone: the whole workspace moves.
- At `3f0df2a8` the vendored `dependencies/linnest` (taken from `865afdc4`, with
  the typed-CBOR repair of `dependencies/linnest-patches`) no longer fits
  `feynkit-graph` and `linnet-render-py`, and upstream `linnest` still lacks
  that repair; `crates/typst-renderer` and the vendored Typst sources that the
  root `[patch.crates-io]` names are gone, so the font assets that this
  package takes from `typst-renderer` change. With the `linnest` patch and the
  Typst patches dropped and the `typst-renderer` assets removed, `cargo check
  --workspace --all-targets` passes at `3f0df2a8`.
- The move therefore re-vendors `linnest` from the new revision with the
  repair ported, and updates the packaged fonts and their manifest, the owner
  stubs, `scripts/vendor-python-{data,type-stubs}.py`, the revision checks in
  `tests/python` and the third-party notices in `LICENSE`.

`env.sh` respects tools already on `PATH`. If a tool is absent, it searches
installed Nix store packages without downloading or modifying system state.
This fallback prefers Rust/Cargo 1.97.1, GCC/GFortran 15.2.0, GNU Make 4.4.1,
and Typst 0.15.1 when available. The active compiler versions must be recorded
with benchmark measurements; the fallback is not a substitute for the pinned
Nix shell when reproducing results on another host.

Cargo target and intermediate build directories default to `target/` and
`target/cargo-build/`. Cargo's writable package cache defaults to
`.local/cargo-home/`; `XDG_CACHE_HOME` defaults to `.local/cache/`. Explicit
environment overrides are respected. No system-wide Cargo configuration,
Rust installation, shell profile, or reference checkout is modified.

The initial checkout was validated with Rust/Cargo 1.99.0, GCC/GFortran 15.2.0,
GNU M4 1.4.20, Python 3.13.12, and Typst 0.15.1. C and Fortran smoke programs
compiled and ran successfully. A C test linked GMP 6.3.0 and MPFR 4.2.2 and
computed a 192-bit square root. `shell.nix` evaluated successfully after fetching
its pinned public source. The Rust dependencies' own GMP/MPFR builds can instead
use their bundled sources; GNU M4 and Make are included for that route.

An existing package cache may be seeded without changing its source:

```sh
mkdir -p .local/cargo-home
cp -a --reflink=auto /path/to/existing/cargo-home/registry .local/cargo-home/
cp -a --reflink=auto /path/to/existing/cargo-home/git .local/cargo-home/
```

Do not copy a global `config.toml` blindly: this supplied environment redirects
Cargo intermediate builds outside the workspace. The local entrypoint exports
`CARGO_BUILD_BUILD_DIR` explicitly to prevent that redirection.

Repository-local author identity is `huashengshao <erdissshaw@gmail.com>`.
GitHub network connectivity and SSH authentication are separate prerequisites.
The read-only authentication check is `ssh -F /dev/null -T git@github.com`;
successful GitHub authentication returns exit status 1 with its greeting.
Never copy credentials into the repository or disable SSH host-key checks.
