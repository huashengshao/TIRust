# Explicit six-point reduction

`SixPointReductionPlan` and `SixPointWorkspace<T>` select the batched six-point
path explicitly. The public interface accepts rank zero through seven,
six unit propagator powers, dimension shift zero, preserved Cartesian shifts,
PV, native IREGI normalization and raw tensor components. Unsupported settings
return `ReductionError::Unsupported` before symbolic preparation. Ordinary
`ReductionPlan`, `Workspace`, Hybrid selection and CLI defaults are unchanged.

```rust
use iregi_rust::{Algorithm, ReductionSettings, SixPointReductionPlan, SixPointWorkspace};
use symbolica::domains::float::Float;

let mut settings = ReductionSettings::new(6, 3);
settings.algorithm = Algorithm::PassarinoVeltman;
let plan = SixPointReductionPlan::new(settings)?;
let mut workspace = SixPointWorkspace::new(&plan, Float::new(128))?;
workspace.set_accuracy_precision(53)?;
// result = workspace.reduce_batch(&kinematics)?;
```

Each `KinematicsBatch` row has one ordinary component and 31 complex arguments:
six four-component absolute shifts, six squared masses, then mu (not mu²).
Construct constants and inputs at the desired precision. Use `input_layout()`
and `parameter_count()` to inspect this layout. The result contains all
cumulative tensor moments in the ordinary IREGI order, with
`[UV single, IR single, IR double, finite]` at each moment. Check each lane via
`BatchResult::lane`; a failed lane does not expose partial or initialized-zero
coefficients.

The plan's `contraction_plan(&monomials)` returns the existing
`ContractionPlan`. Use `ContractionWorkspace` and `NumeratorBatch` to contract
prepared polynomial coefficients, retaining the same raw-component/native
convention. This helper does not expose the private imported child parameters
or apply normalization twice.

The evaluator packs all six five-point subsets across active parent samples
into one child batch. It retains each successful child's primal sensitivity
data before combining values through a Symbolica parent evaluator. Singular
charts, unresolved exact constraints, unavailable sensitivity data and
unsupported quadrature-error imports stay explicit failures. Child retries and
full-Taylor cofactor certificates are internal to this selected path. A
primal-zero coefficient alone never authorizes dropping a child.

`last_execution()` reports parent/child batch work and separate child/parent
certificate counters from the latest call, including work performed before a
failure. `last_diagnostics()` retains child route information. Empty and invalid
calls reset these records. The retained primal sensitivity model is the
ordinary reducer's estimate; it is not a rigorous error enclosure.

For the retained example, after sourcing the normal project environment:

```sh
cargo run --example six_point -p iregi-rust -- \
  validation/public-highpoint/fixture-conditioned_masses.json \
  6 3 pv 128 53 1 /tmp/iregi-six-point.json
```

The example writes every coefficient as an exact rational representation,
alongside settings, represented inputs, lane diagnostics and work counters.
The example writes to the caller's chosen destination; choose a fresh path
when preserving validation evidence. The example's stage durations
are diagnostic and do not establish performance parity.

Numerical evidence currently covers the retained base fixture at ranks two
and three using Float128/Float256 with requested 53; rank-two Float128 lifecycle
checks also cover a scaled fixture, mixed failed neighbors, reordered/chunked
samples, repeated calls and empty batches. Generic type support is broader
than this numerical evidence. Binary64, DoubleFloat, arbitrary complex masses
and genuine numerical Dual inputs need separate high-point qualification.
The public workspace deliberately has no Dual constructor yet and rejects
multi-component input batches.

The rank-seven implementation returns all 330 cumulative moments and 1,320
complex Laurent slots. On the retained fixture, the complete 128/256-bit arrays
agree at `2^-80` and match independently built, higher-precision original IREGI
IBP at the unchanged normalized `1e-10` gate, with maximum difference about
`1.27e-16`. Promoting original arithmetic to 113 binary digits identifies the
older 90 binary64-reference discrepancies as reference accuracy loss. Its
original formulas, constants and stopping criteria are retained, so stored
113-bit arithmetic does not imply 113-bit result accuracy.

The historical failed binary64 comparison remains unchanged in
`validation/highpoint-thin-n6-rank7-phase`. Higher-precision original PV is
unstable on this point; Hybrid reports stability but disagrees in 19 slots.
Neither mode qualifies an independent tensor array. The public guard expansion
does not change recurrence formulas, scalar masters, ordinary/Hybrid dispatch
or the separate Dual restriction. The actual locked/offline current-Cargo build,
two public domain tests and complete Float128/Float256 rank-seven captures pass
in [the current-build capsule](../validation/highpoint-six-point-rank7-current/README.md).
Literal rank-four/five/six plan regressions, rank-seven contractions and the full
batch lifecycle remain separate qualification work.

The isolated public-API qualification in `validation/highpoint-six-point-api`
records 34 targeted tests and 12 exact ordinary-default comparisons, followed
by all 140 rank-three Laurent slots at 128 and 256 bits. Both runs pass the
unchanged original-reference tolerance; their maximum normalized difference
is below `2^-80`. The separate rank-two lifecycle gate checks 1,020 moment
slots and 68 contraction slots across successful, rejected and empty calls.
Its exact source/dependency context remains distinct from later Cargo builds,
broader numerical domains and performance qualification.
