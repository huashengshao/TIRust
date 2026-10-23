# Certified planar pentagon API

`CertifiedPlanarReductionPlan` and `CertifiedPlanarWorkspace<T>` provide a
separate PV interface for five propagators whose four-dimensional routing span
has rank three and admits a regular constant-one propagator identity. Supported
settings are rank zero through seven, unit powers, dimension shift zero,
`InvariantInput::FromShifts`, and `InputPolicy::Preserve`.

This interface is explicit. Ordinary `Workspace::reduce_batch` does not invoke it
automatically. A lower-dimensional external span or a homogeneous-only identity
without a regular constant-one chart remains unsupported by this interface.

```rust
use iregi_rust::{
    Algorithm, CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace,
    KinematicsBatch, ReductionSettings,
};
use symbolica::domains::float::Complex;

let mut settings = ReductionSettings::new(5, 2);
settings.algorithm = Algorithm::PassarinoVeltman;
let plan = CertifiedPlanarReductionPlan::new(settings)?;
let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0.0_f64)?;
// Each row contains20 Cartesian shift components,5 squared masses, then mu.
let values: Vec<Complex<f64>> = prepare_preserved_input_rows();
let input = KinematicsBatch { rows: values.len() / 26, components: 1, values };
let result = workspace.reduce_batch(&input)?;
for lane in 0..input.rows {
    match result.lane(lane) {
        Ok(moments) => consume_tensor_moments(moments),
        Err(status) => record_failed_lane(lane, status),
    }
}
# Ok::<(), iregi_rust::ReductionError>(())
```

The example uses application placeholders for input preparation and result
handling. Result basis, Laurent ordering, convention and tensor weights match
the ordinary semantic API. `new_dual` accepts Symbolica static or dynamic Dual
structures and retains the full ordered shape. Taylor components are separate
from batch rows. Real shift primals and the usual complex-mass branch convention
are validated before certificates; finite complex tangent components are allowed.

Every accepted lane must satisfy an exact represented-input polynomial
certificate for the routing determinant in **every requested Taylor component**.
The selected three-coordinate-plus-mass chart must have a certified nonzero
primal determinant. Symbolica constructs the determinant, solves the linear
identity and expands Dual arithmetic. Inputs are never projected or snapped.
Tangential jets can pass; a transverse mixed derivative is rejected even when
the primal remains planar. `set_certificate_precision_budget` controls adaptive
certificate evaluation independently of the numerical accuracy target.

Rows sharing a certified chart are evaluated together through the ordinary
instruction-major reducer, including all scalar-master and numerical stability
guards. Numerical guard rejection remains `Unstable`; an exact algebraic
certificate does not establish numerical accuracy. Invalid input and unresolved
certificates never expose initialized arrays as successful results. The preserved
binary64 spatial fixture remains an explicit expected-Unstable regression;
separate tetrahedral and higher-precision fixtures test successful reductions.

Structural plans are shared, while numerical scratch and diagnostics belong to
each workspace. `last_execution` reports submitted rows and actual child work.
`analytic_rows` counts numerical chart attempts, including those later rejected
by a guard; it excludes rows rejected before chart dispatch. Certificate passes
are reported separately by `last_certificate_execution`. Profiles close even on
errors and include child work already performed. `fallback_used` retains the
ordinary meaning of an alternate algorithm retry; this explicit chart interface
uses `analytic_used` to identify its chart attempts.

## Production qualification

The standalone API and the separate Symbolica `Atom::add_many` preparation
change are adopted. All seven changed Rust files are byte-identical to the
qualified clean candidate. The [promotion record](../../validation/planar-production/promotion.json)
binds that source to the following independent evidence:

- Nine focused API tests and 49 ordinary execution/PV regression tests pass.
- Twenty original-IREGI coefficient cases cover five topologies and four routing
  variants through rank two, with maximum normalized error `7.20e-13`.
- Ordinary rank-seven arrays agree with independent boxes at 128 and 256 bits;
  all 1,320 complex slots converge with maximum normalized difference `9.45e-32`.
- Two accepted rectangular mixed-Dual samples each have all 5,280 complex slots
  checked against independent Float256 boxes at requested 53-bit accuracy. The
  maximum normalized error is `2.37e-29`. A transverse mixed sample is rejected.

The original mixed test reached its 600-second bound after producing complete
primary arrays. Its timeout remains recorded. A separate comparator validated
those saved arrays in 124.94 seconds; it did not rerun or relabel the original
process. These are correctness diagnostics, not performance measurements.
Automatic ordinary PV/Hybrid fallback is not included in this adoption.

## Bounded qualification commands

The retained candidate was built outside the production tree. Its source and
commands remain available for reproducing the frozen qualification:

```bash
source scripts/env.sh
python3 target/planar-adoption/build_complete.py
source .local/symbolica-license.env
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The ignored ordinary rank-seven test writes all330×4 complex coefficients and
per-component precision. Run128 before256, preserving both output files:

```bash
IREGI_PLANAR_PRECISION=128 IREGI_PLANAR_OUTPUT="$PWD/target/planar-adoption/precision-128.json" \
  timeout 600 taskset --cpu-list 16,17 target/planar-adoption/core-tests \
  --exact certified_planar::tests::public_planar_rank7_precision_artifacts_and_convergence \
  --ignored --nocapture --test-threads=1
IREGI_PLANAR_PRECISION=256 IREGI_PLANAR_OUTPUT="$PWD/target/planar-adoption/precision-256.json" \
IREGI_PLANAR_PREVIOUS="$PWD/target/planar-adoption/precision-128.json" \
  timeout 600 taskset --cpu-list 16,17 target/planar-adoption/core-tests \
  --exact certified_planar::tests::public_planar_rank7_precision_artifacts_and_convergence \
  --ignored --nocapture --test-threads=1
```

The separate rectangular mixed test preserves the ordered shape
`[[0,0],[0,1],[1,0],[1,1]]`, two accepted tangent lanes and one rejected transverse
lane. It stores all5280 complex coefficients per accepted lane, actual output
metadata, exact input and coefficient jets, and the independent five-box result:

```bash
IREGI_PLANAR_MIXED_OUTPUT="$PWD/target/planar-adoption/mixed-rank7.json" \
  timeout 600 taskset --cpu-list 16,17 target/planar-adoption/core-tests \
  --exact certified_planar::tests::public_planar_rank7_rectangular_mixed_artifact \
  --ignored --nocapture --test-threads=1
```

For retained evidence use the accompanying run harness, which also hashes the
binary, build manifest, source and prior precision artifact, captures exit status,
and terminates the process group at the bound. Timing from these checks includes
preparation and is not a matched performance comparison. Exact-rational output
encoding retains represented values but does not preserve signed-zero signs.
