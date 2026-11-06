# The nixpkgs pin supplies Typst 0.15.1 and the C/Fortran toolchain; Rust/Cargo come from
# rust-overlay at the version of rust-toolchain.toml (1.97.1, the toolchain shared with
# pyAmpliCol). Override `pkgs` for a deliberate toolchain upgrade (it must carry the
# rust-overlay overlay); keep benchmark environment metadata in sync.
{ pkgs ? import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/5adc50831c259f87faeb09ddf930d43597197473.tar.gz";
    sha256 = "sha256-Q/8KLB6jNfpaUN1HAedE/u/Et9feE/zYSsutBLKrCFk=";
  }) {
    overlays = [
      (import (builtins.fetchTarball {
        url = "https://github.com/oxalica/rust-overlay/archive/1f69c7203034a4a974d7a4eac0f19b54af30804d.tar.gz";
        sha256 = "sha256-bvaxxdk7mmMQ88AuXQGTUtk+PcUujHpgxqpryqTWvTU=";
      }))
    ];
  } }:
pkgs.mkShell {
  packages = with pkgs; [
    (rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
    gcc gfortran binutils gnumake gnum4 pkg-config
    gmp mpfr libmpc python3 typst git openssh
  ];
  shellHook = ''
    source scripts/env.sh
  '';
}
