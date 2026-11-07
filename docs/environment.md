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
extension `crates/iregi-python`. That extension and every GammaLoop crate
(`feynkit-*`, `spenso`, `idenso`, `spynso3`, `linnet-render-py`) are pinned to
gammaloop `9102f885`, the revision that the Symbolica Community build of the
shared Symbolica revision `b9eaa735` locks for the same Python modules; the
vendored `dependencies/linnest` is taken from that revision with its typed-CBOR
repair (`dependencies/linnest-patches`). `cargo check --workspace` covers the
extension; the wheel build is in `python/README.md`. The pyAmpliCol
maintainers own these pins.

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
