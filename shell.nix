# Pin supplies Rust/Cargo 1.99.0 and Typst 0.15.1. Override `pkgs` for a deliberate
# toolchain upgrade; keep benchmark environment metadata in sync.
{ pkgs ? import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/5adc50831c259f87faeb09ddf930d43597197473.tar.gz";
    sha256 = "sha256-Q/8KLB6jNfpaUN1HAedE/u/Et9feE/zYSsutBLKrCFk=";
  }) {} }:
pkgs.mkShell {
  packages = with pkgs; [
    rustc cargo rustfmt clippy gcc gfortran binutils gnumake gnum4 pkg-config
    gmp mpfr libmpc python3 typst git openssh
  ];
  shellHook = ''
    source scripts/env.sh
  '';
}
