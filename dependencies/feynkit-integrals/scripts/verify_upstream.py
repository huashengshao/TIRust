#!/usr/bin/env python3
"""Verify this source extraction without importing or running Symbolica."""

from hashlib import sha256
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
HASHES = {
    "upstream/integrals.rs": "9ff281c3d7cbfc058d78b0695a15c450230d07ed0399ca13d1553ac753a634b7",
    "upstream/partial_fraction.rs": "228f154bdfcdf8885f7dd139c6bd88122bc46087f53f3f9efc0f76f65207c5dc",
    "upstream/Cargo.toml": "79d20aaddf39916a161293cd78d69c3600c8b1b48ed7b66de2a1b2c9f9348873",
    "LICENSE": "00c7803634d51517a40dbe797e226e0b73c76a789de93db4279b606a3cdeba3d",
}


def require(condition: bool, description: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {description}")
    print(f"PASS: {description}")


def function(source: str, name: str) -> str:
    # The four pinned helper bodies contain no braces inside string literals.
    start = source.index(f"    fn {name}(")
    body = source.index("{", start)
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


for path, expected in HASHES.items():
    require(sha256((PACKAGE / path).read_bytes()).hexdigest() == expected, f"pristine hash {path}")

original = (PACKAGE / "upstream/partial_fraction.rs").read_text()
expected = original.split("#[cfg(test)]")[0].replace(
    "use super::{IntegralFamily, IntegralFamilyError};",
    "use super::{AffineFamily as IntegralFamily, IntegralFamilyError};",
)
require((PACKAGE / "src/partial_fraction.rs").read_text() == expected,
        "partial-fraction engine unchanged except imported type alias")

original = (PACKAGE / "upstream/integrals.rs").read_text()
helpers = (PACKAGE / "src/affine_helpers.rs").read_text()
for name in ["exact_polynomial", "affine_coefficients", "affine_system", "rank_of"]:
    require(function(helpers, name) == function(original, name), f"exact helper {name}")

library = (PACKAGE / "src/lib.rs").read_text()
for name in ["GaussianField", "FamilyCoefficient", "FamilyField", "CoefficientMatrix", "ExactPolynomial"]:
    declaration = next(line for line in original.splitlines() if line.startswith(f"type {name} ="))
    require(declaration in library.splitlines(), f"exact coefficient-field alias {name}")
