# Development environment

Run commands from the `iregi-rust` checkout. The portable entrypoint is
`nix-shell`, using the same pinned nixpkgs revision as the supplied
gammaloop/FeynKit checkout (`02f5696b0e6097e589076d886b317b83ff0437d7`).
The shell supplies Rust, C/C++ and Fortran compilers, GMP/MPFR/MPC, Python,
Typst, and reference-build utilities. Rust is production code; Fortran and
Python are validation tools only.

On a machine that already has these tools installed, use:

```sh
source scripts/env.sh
scripts/cargo.sh test --workspace
```

`env.sh` respects tools already on `PATH`. If a tool is absent, it searches
installed Nix store packages without downloading or modifying system state.
This fallback prefers Rust/Cargo 1.99.0, GCC/GFortran 15.2.0, GNU Make 4.4.1,
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
