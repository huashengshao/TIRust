# High-point matrix preparation: existing-library options

This source audit proposes experiments, not measured optimizations. The reported
generic N=7 rank-zero preparation timeout (>120 seconds) and numeric inline
N=7 rank-seven timeout (>90 seconds) are different workloads. Neither identifies
a matrix bottleneck by itself. Use the new `IREGI_PV_PREPARATION_PROFILE` spans
to separate chart determinants, chart solves, Gram/Cayley geometry, scalar
recurrence construction, and final expression assembly before choosing a change.
No Symbolica runtime or production edit was performed for this note.

## Three source checks

Paths below are relative to the supplied `references/symbolica` or
`references/gammaloop_feynkit` checkout as indicated. These checks found existing
algebra facilities; no new elimination, polynomial, or factorization engine is
needed.

| Facility | Verified API/source | Existing usage/tests and practical limit |
| --- | --- | --- |
| Dense exact solve | Symbolica `lib/numerica/src/tensors/matrix.rs`: `Matrix::solve` (~1949), `solve_any` (~1990), `inv` (~1731) | `solve` accepts one RHS column and performs elimination on an augmented matrix. `inv` performs one elimination with an appended identity for matrices beyond its small hardcoded cases. The source has no public reusable dense LU, `solve_rhs`, `solve_subspace`, adjugate, or all-minors API. `solve_any` and inverse tests are in the same file. |
| Dense determinant | Same file: `Matrix::det` (~1256) | For size >=4 this already uses Bareiss fraction-free elimination, with library exact divisions. Reimplementing Bareiss is redundant. Calling `det` and then `solve` still performs two independent operations. |
| Fraction-free row reduction | Same file: `partial_row_reduce_fraction_free`, `back_substitution_fraction_free`, `solve_fraction_free` (~321–475) | The in-file `solve_fraction_free` test uses an integer system with an integral solution. The high-level method returns `ResultNotInDomain` when a rational-function solution is not in the chosen polynomial ring; it is not an automatic polynomial-matrix-to-rational-solution API. |
| Rational-polynomial matrices | Symbolica `AtomCore::system_to_matrix::<u16>` in `src/atom/core.rs` (~879) returns `Matrix<RationalPolynomialField<Z,u16>>`; `src/solve.rs` implements it | `examples/solve_linear_system.rs::solve_from_matrix` also converts entries with `to_rational_polynomial(&Q,&Z,Some(shared_map))`, constructs the field, and calls `Matrix::solve`. `src/solve.rs` has a matching solve-from-matrix test. Convert final coefficients with `to_expression()`. |
| Factored rational functions | Symbolica `src/domains/factorized_rational_polynomial.rs`: `FactorizedRationalPolynomialField::{new,new_from_poly}`, `Field` implementation (~1129); `AtomCore::try_to_factorized_rational_polynomial` (~1656) | `tests/rational_polynomial.rs` exercises conversion and arithmetic. This representation keeps denominator factors; conversion/factorization also has a cost. No high-point matrix benchmark establishes that it beats the ordinary rational-polynomial field. |
| Reusable sparse reduction state | Symbolica `lib/numerica/src/tensors/sparse.rs`: `SparseRowReducer`, `LuLMode::{Full,Pattern,None}`, `from_matrix`, `add_row`, `add_matrix`, `back_substitute`, `u`, `l`, `pivots` (~1527–1884) | The reducer is `Clone`, can reuse a common set of rows before adding further constraints, and has direct LU/RREF tests (~2717,2817). It does **not** expose a drop-in reusable `solve_rhs` or determinant-from-existing-factorization method. `SparseMatrix::{det,inv,solve}` each construct their own reducer. Tiny dense 5-by-5 charts may not benefit from sparse storage. |
| FeynKit exact projector solves | `crates/feynkit-tensor/src/orbit_projector.rs::solve_transposed` (~518), `weingarten.rs::solve_level` (~458), `reduction/external.rs::external_projector` (~38) | These use `Atom::system_to_matrix`, then the library `solve` or `inv`, then `to_expression`. This is a concrete existing alternative to solving directly over `AtomField` trees. |
| FeynKit integral families | `crates/feynkit-graph/src/integrals.rs`: `FamilyField`, `affine_system`, `scalar_product_rules`; `integrals/partial_fraction.rs` | `FamilyField` is a rational-polynomial field over Gaussian rationals. `partial_fraction(powers,max_states)` already handles constant/homogeneous denominator relations, caches relations by support, coalesces exponent states, and errors on budget exhaustion. In-file tests cover repeated powers and both relation types. This deserves reuse evaluation before adding a new partial-fraction engine. |

## Why the current field can be expensive

`pv.rs::field` selects exact zero tests, division cancellation, and a custom
normalizer that calls `together()`. In `src/domains/atom.rs`, that custom
normalizer is applied after every ring operation. `AtomField::div` first calls
`cancel()` when enabled, then normalizes; `div_assign` and `inv` can normalize
again. Thus one matrix elimination may repeatedly convert and simplify growing
expression trees. This is an identifiable source of work, not proof that it is
the observed timeout's dominant cost.

`AtomField` also implements `EuclideanDomain`, but its `gcd` currently returns
one. Merely replacing `solve` with `solve_fraction_free` on the same AtomField
does not gain polynomial-content removal and can increase intermediate products.
Disabling normalization blindly is unsafe: exact zero/pivot recognition would
then depend on expression shape. Statistical zero tests are not a replacement
for exact singular-chart decisions.

## Conditional experiments, in order

1. **If chart algebra dominates, use the established rational-polynomial matrix
   path.** Convert the entire coordinate/f row system once, sharing one variable
   map; form every chart by selecting entries from that representation. Keep the
   existing `Matrix::det`, `solve`, and lazy chart-selection semantics. Convert
   only final determinants and solution coefficients back to Atom aliases. This
   reduces repeated Atom-to-polynomial work while delegating all algebra to
   Symbolica. Benchmark conversion, determinant, solve, and output conversion
   separately. A prototype must retain exact zero detection and the N7
   alternative-chart/Dual regression.
2. **If the same shape is rebuilt often, cache exact matrix templates.** One
   generic 5-by-5 matrix with RHS `(0,0,0,0,1)` can be solved and its determinant
   prepared using the same library APIs. Instantiate its Atom expressions with
   Symbolica replacement for each chart, then share the resulting aliases.
   Keep all six N7 charts: a cached generic solution does not authorize dropping
   charts that become singular on individual lanes. Validate structural
   specializations and do not route known degenerate matrices through an
   unguarded generic inverse. This trades one potentially large template for
   repeated matrix work; measure it rather than assuming a win.
3. **If denominator-product growth dominates, compare the existing factorized
   rational-polynomial field.** It implements the traits required by the library
   matrix solve. Use a shared complete variable map and library conversion;
   account for factorization and memory overhead. It is a candidate, not the
   default conclusion of this audit.
4. **If repeated constrained nullspace systems dominate, consider sparse reducer
   reuse.** A reducer for common constraint rows can be cloned and extended by a
   unit-constraint row, matching the scalar exceptional-solve search. This is
   supported by `SparseRowReducer::add_row`. Replacing a chart column is a
   different operation: the reducer exposes no column-update factorization, so
   do not claim one LU solves all six charts. Adding a general library-level
   reusable solve/determinant API would be dependency work with provenance and
   tests, not permission to write a local elimination algorithm.
5. **If scalar recurrence or inline expansion dominates, changing matrix fields
   will not resolve the bottleneck.** Prefer the existing dependency-ordered DAG
   and shared recurrence/geometry caches. FeynKit's factor preservation is
   concrete at `reduction/external.rs` (~111) and the test
   `reduction_keeps_factored_scalar_and_foreign_tensor_spectators`; its orbit
   methods solve smaller systems rather than introducing a new matrix backend.

The six N7 charts share minors, but no public all-minors/adjugate preparation API
was found in the checked dense/sparse libraries. Do not introduce a handwritten
minor-expansion engine. Repeated calls to existing `Matrix::det` for a proposed
minor cache would still require a measured comparison against the much simpler
rational-polynomial solve and would need exact identity/chart tests.

## Qualification before adoption

Retain current completed/timeout artifacts unchanged. Compare both generic and
exact-numeric preparation independently, recording elapsed stage times, output
DAG size, master counts, and peak memory. Require exact denominator identities,
all alternative-chart selections, singular-input failures, ordinary and higher
Dual components, and the existing complete N6/N7 numerical adjudication where
available. The goal is a preparation optimization with identical accepted input
and diagnostic semantics; none of the proposals above has yet been measured.
