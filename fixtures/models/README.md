`phi3.json` is a minimal real scalar model for supplied polynomial numerator tests.
It follows FeynKit's model JSON format and has one cubic interaction and one mass
parameter, `UFO::M`.

`sm.json` is copied without modification from
`crates/feynkit-model/tests/fixtures/sm.json` in gammaloop commit
`865afdc43fb4e6b950439d34b4ec2c38352873e7`. Its SHA-256 is
`95716e0deeabe7bce3638fdc69264022f9a25ddad63a97e6441bf4843cd1ec41`.
This pinned JSON model lets the production Rust preparation code apply Feynman
rules without importing Python. It uses `H` for the Higgs particle; MadGraph's
process syntax spells that particle `h`.
