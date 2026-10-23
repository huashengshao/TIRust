#!/usr/bin/env bash
# Source from any shell command: source scripts/env.sh
# Existing tool selections take priority; Nix store discovery is an offline fallback.

IREGI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export IREGI_ROOT

_iregi_require_tool() {
    local command_name="$1"
    shift
    command -v "$command_name" >/dev/null 2>&1 && return 0
    local pattern candidate
    for pattern in "$@"; do
        # Intentionally expand only the store-package glob, never arbitrary commands.
        for candidate in /nix/store/$pattern/bin/"$command_name"; do
            if [[ -x "$candidate" ]]; then
                PATH="$PATH:${candidate%/*}"
                export PATH
                return 0
            fi
        done
    done
    printf 'iregi-rust: missing %s; enter nix-shell or install it on PATH.\n' "$command_name" >&2
    return 1
}

_iregi_require_tool rustc '*-rustc-1.99.0' '*-rustc-1.*' || return 1
_iregi_require_tool cargo '*-cargo-1.99.0' '*-cargo-1.*' || return 1
_iregi_require_tool cc '*-gcc-wrapper-15.2.0' '*-gcc-wrapper-*' || return 1
_iregi_require_tool c++ '*-gcc-wrapper-15.2.0' '*-gcc-wrapper-*' || return 1
_iregi_require_tool ar '*-binutils-wrapper-2.44' '*-binutils-wrapper-*' || return 1
_iregi_require_tool make '*-gnumake-4.4.1' '*-gnumake-*' || return 1
_iregi_require_tool m4 '*-gnum4-1.4.*' '*-m4-*' || return 1
_iregi_require_tool pkg-config '*-pkg-config-wrapper-0.29.2' '*-pkg-config-*' || return 1
_iregi_require_tool python3 '*-python3-3.13.*' '*-python3-3.*' || return 1
# These are needed for reference validation and report production, respectively.
_iregi_require_tool gfortran '*-gfortran-wrapper-15.2.0' '*-gfortran-wrapper-*' || return 1
_iregi_require_tool typst '*-typst-0.15.1' '*-typst-*' || return 1

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$IREGI_ROOT/target}"
export CARGO_BUILD_BUILD_DIR="${CARGO_BUILD_BUILD_DIR:-$IREGI_ROOT/target/cargo-build}"
# Explicit user overrides are respected. The default keeps package-cache locks,
# registry updates and downloaded dependencies inside this checkout.
export CARGO_HOME="${CARGO_HOME:-$IREGI_ROOT/.local/cargo-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$IREGI_ROOT/.local/cache}"
# Keep Symbolica's supported per-user license lock in the shared project cache.
# This relocates the lock; it does not alter license checks or concurrency limits.
export SYMBOLICA_LOCK_DIR="${SYMBOLICA_LOCK_DIR:-$IREGI_ROOT/.local/symbolica-locks}"
mkdir -p -m 700 "$SYMBOLICA_LOCK_DIR"
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-1}"
export CC="${CC:-$(command -v cc)}"
export CXX="${CXX:-$(command -v c++)}"
export FC="${FC:-$(command -v gfortran)}"

unset -f _iregi_require_tool
