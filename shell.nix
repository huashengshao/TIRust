# Pin matches the supplied gammaloop/FeynKit environment. Override `pkgs` for a
# deliberate toolchain upgrade; keep benchmark environment metadata in sync.
{ pkgs ? import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/02f5696b0e6097e589076d886b317b83ff0437d7.tar.gz";
    sha256 = "sha256-llGJbC0CcU8DfROr6mZjRJgMLQd/SKwfpzxJH/2lHO4=";
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
