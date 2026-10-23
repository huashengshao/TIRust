# Upstream provenance and extraction audit

Source repository: https://github.com/alphal00p/gammaloop

Pinned revision: `865afdc43fb4e6b950439d34b4ec2c38352873e7`, matching the FeynKit
revision used by the embedding iregi-rust workspace. The source files were copied
from its supplied reference checkout. Pristine source copies are retained under
`upstream/` so the extraction can be audited without network access.

| Source file | SHA-256 |
| --- | --- |
| `crates/feynkit-graph/src/integrals.rs` | `9ff281c3d7cbfc058d78b0695a15c450230d07ed0399ca13d1553ac753a634b7` |
| `crates/feynkit-graph/src/integrals/partial_fraction.rs` | `228f154bdfcdf8885f7dd139c6bd88122bc46087f53f3f9efc0f76f65207c5dc` |
| `crates/feynkit-graph/Cargo.toml` | `79d20aaddf39916a161293cd78d69c3600c8b1b48ed7b66de2a1b2c9f9348873` |
| `LICENSE` | `00c7803634d51517a40dbe797e226e0b73c76a789de93db4279b606a3cdeba3d` |

The original root `LICENSE` is copied verbatim. The package retains the upstream
`feynkit-graph` manifest declaration `MIT OR Apache-2.0`; the pristine manifest is
included. No additional crate-local license or copyright text was present in the
source package. Symbolica retains its own license and runtime requirements.

## Extraction boundary

- `src/partial_fraction.rs` preserves upstream lines 1–176 byte for byte except
  the import aliases the new `AffineFamily` as `IntegralFamily`. The dependence
  search, ordering, power-transfer rules, exact solves, state budget, and error
  behavior are unchanged. Upstream graph-dependent tests are adapted separately.
- `src/affine_helpers.rs` preserves four upstream method bodies and signatures
  exactly: `exact_polynomial` (97–113), `affine_coefficients` (115–137),
  `affine_system` (164–197), and `rank_of` (321–337). Only their surrounding impl
  type is renamed. The file is included into the parent module so method privacy
  does not need to change.
- `src/lib.rs` preserves the exact coefficient-field aliases from upstream
  lines 26–30. It retains the four partial-fraction error cases and adds an
  explicit-variable validation error. The `scalar_products` field name is kept
  internally to avoid editing either algorithm or helper methods.
- The new constructor accepts explicit affine variables and denominators. It
  rejects empty, duplicate, or non-symbol variables and invokes unchanged
  `rank_of` before returning, preserving validation of every row and coefficient.
- The external-coefficient guard is upstream lines 302–319 with graph-only
  momentum/Spenso alias checks removed. Its recursive Symbolica `contains` check
  remains. Restricting variables to bare symbols prevents partially specified
  labeled-call variables from weakening hidden-dependence detection.
- Graph construction, kinematic assumptions, family completion, loop shifts,
  parametric integrals, mapping, and scalar-product rewriting are omitted.

`scripts/verify_upstream.py` checks all pristine hashes, the single permitted
engine import change, the four exact helper functions, and the field aliases.
It does not load Symbolica or perform algebra.

## Reuse checks and semantic limits

1. Symbolica source provides `AtomCore::apart_multivariate` and rational-polynomial
   `apart_multivariate`; these use factorization/Groebner reduction and return
   expressions rather than bounded decompositions retaining original propagator
   power vectors. Its matrix and exact polynomial APIs already provide all
   lower-level algebra needed by FeynKit, and are reused unchanged.
2. Symbolica usage/tests and FeynKit's own partial-fraction tests confirm that
   exact rational/Gaussian coefficient arithmetic avoids statistical zero tests
   and preserves tiny expanded coefficients and original row normalization.
3. The supplied FeynKit/gammaloop tree contains the matching bounded affine
   engine only in `feynkit-graph::IntegralFamily`; no smaller package exposes it.
   The whole graph crate requires model, kinematics, Spenso and Idenso packages.
   This extraction removes those dependencies without recreating its algorithm.

All integrated dependence must be expressed using the supplied variable symbols.
External opaque functions such as `sin(a)` are allowed; `sin(x)` for an integrated
variable is rejected. Exact polynomial conversion rejects `x²`, `1/x`, floating
numeric coefficients, infinities, and indeterminate values when they occur as
numeric coefficients. This wrapper introduces no conversion to binary64 and no
numeric rationalization. Denominators of generic external coefficients may
vanish after specialization; callers must construct a specialized family when
those relations change. Bare rational constants can be denominators, and a zero
denominator is an error only when its requested power is positive, as upstream.

The adapted tests retain upstream examples for mass shifts, coincident and
proportional propagators, homogeneous/inhomogeneous relations, signed powers,
fractional rows, Gaussian coefficients, and coefficients scaled by `10^±1000`.
Additional wrapper tests cover exact original-order coefficient mapping, invalid
variables, hidden/nonlinear dependence, zero-state budgets, and power overflow.

An independent read-only audit compared these files directly against the clean
pinned reference checkout and found no algorithmic changes. Runtime qualification
is tracked separately in `VALIDATION.md`.
