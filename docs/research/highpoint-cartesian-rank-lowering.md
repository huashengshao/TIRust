# Private Cartesian rank-lowering experiment

This experiment starts from production revision `06706e2`. Production remains
unchanged. It implements the independent integrand identity documented in
[the COLLIER audit](collier-1.2.10-audit.md), using Symbolica for every matrix and
expression operation. It does not reuse COLLIER source code.

The private policy applies only to ordinary preserved `FromShifts` inputs with
six or seven unit-power denominators in `D=4-2 epsilon`. It is independent of
the existing pentagon coordinate-identity policy. Direct `with_distances`
compilers keep their existing default, and public supplied-invariant inputs
retain their existing exclusion.

For each active topology, relative shifts and the existing `f_values` form the
five-row matrix `K`. Each chart uses one Symbolica matrix inverse and one matrix
product with four right-hand sides. Direction `mu` has the right-hand side
`(eta_mu e_mu/2, -p_origin^mu)`, with the current active origin. The resulting
weights satisfy `qbar^mu = sum_i w_i^mu D_i`. The common full-dimensional loop
square cancels, including its perpendicular component. Removing one occurrence
of a Cartesian numerator component therefore replaces a rank-r high-point
moment by rank-(r-1) pinched moments without combinatorial factors. All four
native Laurent slots pass through the same relation.

Rank-zero scalar requests retain the existing constant identity. The six N7
charts retain the existing deterministic lazy order. Structurally unavailable
Cartesian charts request the old constant-identity path; all-singular numerical
charts retain an active inverse denominator for existing runtime guards. This
experiment does not introduce a numerical global selector, conditioning
threshold, eager fallback DAG, or new singular expansion.

Reuse checks:

1. Symbolica/Numerica `matrix.rs` provides `Matrix::inv` and matrix multiplication;
   `Matrix::solve` accepts a single vector right-hand side. The implementation
   uses these operations directly instead of implementing a multi-RHS solver.
2. Existing `pv.rs::geometry` already uses Symbolica inversion, while the old
   denominator identity uses Symbolica vector solves. Existing Symbolica exact
   expression and evaluator facilities supply the witnesses.
3. FeynKit's external tensor projector and integral-family affine/partial-fraction
   code likewise delegate inversion and affine solves to Symbolica.

The first acceptance stage is exact integrand identities and singular controls,
followed by complete rank-one public outputs against an independently compiled
old implementation and untouched original IREGI. Full rank-seven arrays,
precision convergence, mixed Dual jets, exceptional lanes, and optimized timing
remain separate subsequent requirements. Earlier 180/300/600-second preparation
failures remain retained; this source experiment does not qualify them.

Sources, build manifests, and validation artifacts are under
`validation/highpoint-rank-lowering`. The paired private builds use the previously
frozen historical dependency closure, not a current optimized master build.
Their preparation timings are development diagnostics only.
