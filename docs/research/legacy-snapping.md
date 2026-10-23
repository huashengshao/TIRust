# Optional original-IREGI on-shell policy

`InputPolicy::Preserve` remains the default. `LegacyOnShell` reproduces the
original `makeonshell` selection rules at individual real `scalarprod` call
sites. It does not change the stored input momenta, squared masses, or scale.
The default legacy threshold is the exact rational `1/1000000`.

## Original source contract

The source paths below are relative to `references/madgraph/vendor/IREGI/src`.

1. `global.f90:40–43` defines the threshold and disabled snapping flags.
   `INITIREGI` (`IREGI4ML5_interface.f90:39–40`) sets the caller's threshold and
   enables `ONSHELL_IREGI`; `scalarprod` also snaps when the testing flag
   `check` is true (`kinematics.f90:23`). `cscalarprod` never snaps.
2. `IMLOOP` builds reference masses from the real parts of the input squared
   masses, in their original order (`IREGI4ML5_interface.f90:95–123`). A
   candidate `m` is eligible when `abs(m) > threshold`. It is discarded if an
   earlier retained reference `r` satisfies `abs(m-r)/m < threshold`. The first
   representative wins. This happens before CTMODE reverses the propagators.
   Pinched subtopologies continue to use that original reference list.
3. For a scalar product `x`, `makeonshell` (`kinematics.f90:33–57`) first tests
   `abs(x/first_reference) < threshold` and returns zero immediately. Otherwise
   it returns the first retained `r` satisfying `abs(x-r)/r < threshold`.
   With no retained references it instead tests `abs(x) < 1/1000`, independently
   of the configured threshold. If no test succeeds, `x` is unchanged. Every
   comparison is strict: equality takes the unsnapped/next branch.
4. Both relative reference comparisons have a **signed denominator**. This
   historical behavior is preserved explicitly: a negative candidate can be
   discarded against an unrelated positive reference, and an initial reference
   `-4` can snap an unrelated scalar product `123` to `-4`.

Complex masses still contribute only their real parts to the reference list.
Their imaginary parts remain in the reduction input. Original IMLOOP's separate
choice between its real and complex reduction routines uses
`abs(imag(m²))/max(abs(m²),1/100) > 1e-15`; this is not an instruction to erase
small imaginary parts in the Rust API.

## Scalar-product call sites and supplied invariants

Snapping is not bilinear. Consequently it cannot be implemented by snapping one
matrix and deriving every other invariant from that matrix:

| Original use | Source | Required Rust operation |
|---|---|---|
| Scalar-master invariants | `mis_warp.f90:60–84,106–130` | Snap each raw squared momentum difference |
| Cayley and IBP R/S matrices | `matrices.f90:22,44,140,201` | Snap each raw squared difference before subtracting masses |
| PV Y and X first row | `matrices.f90:288,299,362,373` | Snap each raw squared difference |
| PV Z / Gram cross-products | `matrices.f90:292,366` | Compute the actual coordinate dot product for that subtopology origin, snap it, then multiply by two |
| Additional scalar/tensor formulas | `si_reduce.f90:701`, `csi_reduce.f90:126`, `pave_reduce.f90:4256` and corresponding complex routines | Use the same individually snapped squared differences |

The isolated Fortran probe uses reference `4`, threshold `1e-6`,
`p=(1,0,0,0)`, and `q=(4.000001,1,0,0)`. Directly snapped `p·q` equals `4`,
whereas `(snap(p²)+snap(q²)-snap((p-q)²))/2` equals approximately `4.000001`.
The PV compiler therefore has a separate scalar-product transform applied to
each actual shifted coordinate cross-product, including pinched subtopologies.

**Supplied scalar products bypass snapping.** Original `CaleyMatrix2`,
`RSMatrices2`, `XYZMATRICES2`, `pave_f2`, and `I0C2` use supplied entries
algebraically without calling `makeonshell`. `SuppliedScalarProducts` therefore
leaves both supplied entries and derived invariants unchanged even when the
legacy policy is selected. The coordinate values continue to assemble tensor
components, as in the corresponding original tensor interface.

## Rust preparation, precision, and derivatives

`LegacySnapping` prepares a shared Symbolica alias graph from the original mass
ordering. `snap_scalar_product` accepts a raw coordinate-dot expression and
returns its selected-branch expression. Only Symbolica expressions perform
arithmetic; `IF`, guarded `ABS`, and `CONJ` implement branch selection and the
real-part projection. Identical raw queries share prepared nodes. Plans contain
no sampled values or numerical caches.

The threshold accepts an exact rational or decimal/scientific literal. Decimal
token handling separates digits and exponent and feeds Symbolica's `Integer`
parser plus exact powers of ten; it does not round through a floating-point
coefficient. Literal size is bounded to 4096 characters and exponent magnitude
65536. The Rust interface accepts a nonnegative threshold, including exact zero.
At zero, the historical no-reference absolute rule remains `abs(x) < 1e-3`;
relative comparisons with positive reference masses cannot pass. A positive
exact threshold that underflows in the selected numeric type is rejected.
Supplied-G inputs do not evaluate the unused threshold numerically.

One nonphysical historical combination is explicitly unsupported on active
matrix paths: exact zero threshold with a negative retained reference can map
`scalarprod(0,0)` to a negative mass. Original Cayley diagonals then become
nonzero, violating the zero-distance identities required by the current Rust
recurrences. A guarded compatibility probe rejects those lanes before masters
run; direct rank-zero tadpoles that never use these matrices remain valid.
This differs from the ordinary positive-threshold negative-mass quirks, which
the selected branches preserve. An isolated Fortran probe confirmed unchanged
direct A0 values and changed generalized power-two tadpoles (including doubled
UV residues) in this pathological combination.

Symbolica's Dualizer selects `IF` using the primal. A snapped-zero branch has a
zero jet. A snapped-mass branch carries the complete jet of `Re(m²)` of the
selected representative, including higher and mixed components. An unsnapped
branch retains the original scalar-product jet. Predicate derivatives are not
used. `ABS` is guarded at zero because the generic Dual rule otherwise contains
`abs(x)/x`.

Each active strict comparison also emits a diagnostic residual. An inactive
comparison emits one. A zero primal residual identifies a branch boundary;
nonfinite primal residuals indicate an unsafe branch classification at the
chosen precision. Boundary derivatives are not claimed to exist. The runtime
must inspect these primal diagnostics and expose the boundary/failure rather
than treating such derivatives as an unqualified successful result.
Each probe names its predicate alias, so the runtime can also guard it by the
outer branch that demands the snapped query. Unused queries and abandoned PV
preparation paths must not contribute diagnostic failures.

Coordinate-input legacy plans currently accept up to four propagators. Plans
with five or more explicitly return `Unsupported`: the geometric identities
used for higher-point completion require separate validation once snapping
has broken bilinearity. The supplied-G bypass retains its ordinary supported
domain. Coordinate rank-reduction shortcuts are disabled for the legacy policy.

## Reuse checks and independent validation

Three reuse checks preceded the small lexical decimal adapter:

1. Symbolica's `ParseSettings` and parser create floating coefficients for
   decimal literals; Numerica's `Integer` parser and exact `Rational`/`Atom`
   arithmetic supply the exact primitives. `Rational::try_from(Atom)` rejects
   floating coefficients. No exact decimal-to-rational parser was found.
2. Symbolica parser tests and numerical examples were inspected for an existing
   exact lexical decimal adapter. Existing exact examples use integer/rational
   literals. `Symbol::IF` has a nonzero predicate, so strict comparison uses
   the identity `abs(d)-d != 0` exactly when finite real `d < 0`, with its zero
   branch guarded before Dualization.
3. FeynKit/Spenso/Idenso numeric parsing and coefficient conversions were
   searched for equivalent facilities. Their existing Rational conversions
   likewise operate on already-exact atoms. No alternate algebra or Dual
   implementation is introduced here.

`validation/reference/snapping` contains a driver linked against the unchanged
original `kinematics.o` and `global.o`, plus durable snapshots and checks.
It covers strict boundaries, reference ordering/deduplication, negative masses,
massless absolute snapping, and the non-bilinear Gram example. The optional
reference harness now duplicates IMLOOP's reference-list and complex-mode
selection rules; this corrects the harness, not the original IREGI sources.
Previously captured `snap=false` fixtures are checked against an isolated
rebuild before accepting that harness correction.

The Rust module tests exercise matching branch outputs, branch diagnostics,
full selected Dual jets, and inputs distinguished only beyond binary64. Test
execution is recorded separately from source preparation; writing this note
does not imply that full reduction-level legacy qualification has passed.

The `reductions.json` fixture contains complete original-IREGI rank-two arrays
for bubbles, triangles, and boxes in all three modes, with translated
complex-mass variants. The triangle and box deliberately place a cross dot
inside a reference-mass snapping interval while the derived squared distances
remain elsewhere. These cases detect the incorrect practice of reconstructing
the snapped Gram matrix from snapped pair distances. Separate evidence records
the unsupported negative-reference/zero-threshold diagonal behavior.

The recorded primary binary64 baseline covers 30 cases: all 20 PV/hybrid cases
and both IBP bubbles agree at the unchanged `1e-10` tolerance. Eight IBP
triangle/box cases remain visible as failures: one exact pinched-bubble
pseudothreshold produces nonfinite coefficients, and seven are rejected by
the cancellation monitor and require verified higher precision. These statuses
do not establish wrong finite answers. Full inputs, reference arrays, source
hashes, and individual statuses are retained in `rust-validation.json`.
This baseline predates the separately developed analytic S-matrix recurrence.

After enabling that recurrence, all 30 cases were rerun. Primary binary64
coverage remains 22/30, while the exact pseudothreshold now reaches the
cancellation guard instead of producing nonfinite coefficients. All eight IBP
triangle/box cases then passed explicit 128-bit rescue with independent 256-bit
verification at the original requested 53-bit accuracy and unchanged `1e-10`
reference tolerance. The largest normalized reference discrepancy was
`2.93e-12`. Each precision evaluated the four triangle lanes and four box lanes
as batches, including their scalar-master families. All primary statuses and
complete output arrays are retained in `rust-validation-after-s.json` and
`rescue-validation-after-s.json`. This qualifies all 30 ordinary rank-two
fixtures using 22 primary successes and eight verified rescues; it does not
claim binary64-only parity or qualification of generalized legacy inputs.
The combined 30-case regression subsequently passed in 73.12 s and records
complete primary/rescue statuses and output arrays in
`rust-qualified-validation.json`. Stable primary lanes do not request higher
precision; the test checks that contract as well as every reference value.

Reproduce the isolated Fortran evidence from the repository root:

```sh
source scripts/env.sh
python validation/reference/snapping/check.py
python validation/reference/snapping/capture_reductions.py \
  --executable target/snapping-reference/oracle-build/iregi-reference
scripts/cargo.sh test -p iregi-rust snapping::tests -- --test-threads=1
```
