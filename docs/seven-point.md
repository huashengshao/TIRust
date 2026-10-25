# Explicit seven-point scalar reduction

`SevenPointReductionPlan` and `SevenPointWorkspace<T>` provide an explicitly
selected scalar path with the same plan/scratch ownership style as the
[six-point API](six-point.md). The accepted settings are seven propagators,
rank zero, unit powers, dimension shift zero, PV, preserved Cartesian shifts,
native IREGI normalization and raw components. Unsupported settings fail before
symbolic preparation.

```rust
use iregi_rust::{
    Algorithm, ReductionSettings, SevenPointReductionPlan, SevenPointWorkspace,
};
use symbolica::domains::float::Float;

let mut settings = ReductionSettings::new(7, 0);
settings.algorithm = Algorithm::PassarinoVeltman;
let plan = SevenPointReductionPlan::new(settings)?;
let mut workspace = SevenPointWorkspace::new(&plan, Float::new(128))?;
workspace.set_accuracy_precision(53)?;
// let result = workspace.reduce_batch(&kinematics)?;
```

Each `KinematicsBatch` row has one ordinary component and 36 complex arguments:
seven absolute four-component shifts, seven squared masses and mu (not mu²).
Construct inputs directly at the working precision. `input_layout()` and
`parameter_count()` describe the layout. The scalar moment contains all four
native Laurent slots `[UV single, IR single, IR double, finite]`. Check every
lane with `BatchResult::lane`; failed lanes do not expose partial arrays.

All 21 five-point subsets are mandatory. The workspace packs them across
active parent lanes into one child batch and uses Symbolica for the guarded
parent combination. Both parent and children explicitly use Symbolica scalar
masters, including when `native-f64-abc` is compiled. Missing child sensitivity
data, singular charts and unsupported error transport remain explicit
failures. The retained sensitivity model is an ordinary numerical estimate,
not a rigorous enclosure or derivative error certificate.

Use `last_execution()` and `last_diagnostics()` to inspect work and routes.
On the retained regular family, one family call evaluates `21 * active_lanes`
child rows and requests `105 * active_lanes` scalar-master rows. These are
requested rows, not unique integrals. Invalid lanes are excluded from the
active batch. Empty or malformed calls reset the previous call's records.
Independent workspaces own their numerical scratch. The plan's existing
`contraction_plan()` helper also supports prepared constant numerators.

The public probe accepts a caller-selected output file:

```sh
source scripts/env.sh
cargo run --locked --offline -p iregi-rust \
  --example seven_point_public_probe -- \
  validation/highpoint-seven-point-current-cargo/fixture.json \
  7 0 pv 128 53 1 /tmp/iregi-seven-point.json
```

Choose a fresh output path when retaining validation records. The probe writes
exact represented coefficients, input metadata, diagnostics and counters.
Its stage durations measure the validation lifecycle, not kernel performance.

The actual locked/offline default and selective Cargo builds pass 35 focused
exact tests. On one regular real-mass family, selective-build Float128/Float256
captures agree with stable original IREGI IBP and Hybrid results for all four
slots at the unchanged normalized `1e-10` gate; all four precision pairs pass
`2^-80`. Original PV is unstable on that point. Both the failed first 256-bit
attempt and the successful separately bounded follow-up are retained in
[the current-build capsule](../validation/highpoint-seven-point-current-cargo/README.md).

The current-build workspace lifecycle remains **incomplete**: its fixed time
limit stopped the nine-call schedule after six completed calls. Those records
check seven successful rows, three invalid-scale rows, an empty batch, reordered
and chunked samples, input preservation, 28 complex moment slots and 28 unit
contractions. Independent scratch, the later empty call and malformed-call
recovery are not qualified by that run. Earlier private lifecycle results retain
their separate source context. This partial checkpoint does not establish full
seven-point acceptance or the requested rank-seven coverage.

Generic typing alone does not qualify f64, DoubleFloat, complex masses or
arbitrary kinematics. There is no public Dual constructor. Higher rank, dotted
propagators, dimension shifts, other conventions and Hybrid/IBP selection are
outside this explicit interface. The ordinary generalized APIs retain their
separate supported domains; this restriction does not change them. No
seven-point performance-parity claim is made.
