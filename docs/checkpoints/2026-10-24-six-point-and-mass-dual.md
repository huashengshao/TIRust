# Restricted six-point API and mass-Dual validation

This checkpoint adds the explicit Rust `SixPointReductionPlan` and
`SixPointWorkspace<T>` API. It packs all six five-point subsets across active
sample lanes into one child batch, then evaluates the parent through Symbolica.
The initial interface accepts ranks zero through three, unit powers, dimension
shift zero, preserved Cartesian shifts, PV and native raw tensor moments.
Ordinary reduction defaults are unchanged. See [the API guide](../six-point.md).

Both the isolated integration and the normal locked Cargo build pass the 34
focused tests. The current Cargo example passes every one of the 140 rank-three
Laurent slots against the retained original IREGI case at Float128 and Float256,
requesting 53 accuracy bits. The maximum dimensionally normalized original
discrepancy is `2.43e-12`, below `1e-10`; the maximum precision difference is
`1.48e-34`, below `2^-80`. These working precisions do not imply that every output
achieves 128 or 256 bits of accuracy.

The separate isolated qualification also checks 12 ordinary API cases exactly
and a rank-two batch lifecycle with 1,020 tensor slots and 68 contractions.
Those checks retain their original source and dependency context. They are not
reported as additional current-Cargo lifecycle runs. Numerical high-point
binary64, DoubleFloat, public Dual, higher-rank and broader kinematic coverage
remain open.

The previously built Python wheel004 independently passes two additional
massive spacelike cases, one triangle and one box. First, second and mixed
squared-mass Taylor coefficients agree with separately prepared raised-power
IBP reductions in all four Laurent slots: 28 evaluations, 96 identities and 96
precision comparisons. Both saved-data audits pass. This verifies the retained
dotted-propagator identities without implying additional physical, threshold or
sample-batch coverage. See [the exact inputs and audits](../../validation/c0d0-dual-identity/README.md).

A private seven-point preparation experiment now uses Symbolica's existing
rational-polynomial matrix domain. All 22 selected exact tests pass and the
generic rank-zero constructor completes within its fixed budget. No seven-point
workspace or numerical integral is qualified by that experiment. Earlier
preparation timeouts and the inconclusive scalar oracle remain visible.

The benchmark report adds these coverage results without changing historical
timings or ratios. Every timing ratio retains TIRust in the numerator. No new
performance gate or default backend adoption follows from this checkpoint;
IREGI/COLLIER parity and the full project acceptance criteria remain open.

Evidence:

- [Isolated public six-point integration](../../validation/highpoint-six-point-api/qualification.json)
- [Current Cargo integration](../../validation/highpoint-six-point-current-cargo/qualification.json)
- [C0/D0 mass-derivative qualification](../../validation/c0d0-dual-identity/qualification.json)
- [Private seven-point constructor](../../validation/highpoint-thin-n7-rational-domain/qualification.json)
- [Benchmark source](../benchmark.typ) and [PDF](../benchmark.pdf)
