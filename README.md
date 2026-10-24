# iregi-rust

A Rust implementation of one-loop tensor reduction, using Symbolica for algebra
and batched numerical evaluation, and oneloopmaster for scalar integrals.

Development follows [PLAN.md](PLAN.md). The implementation is undergoing
independent validation; the full coverage and performance requirements are not
yet qualified. [docs/benchmark.pdf](docs/benchmark.pdf) records measured evidence,
known reference defects, and outstanding work without treating missing results
as successful cases.

## Workspace

| Crate | Purpose |
| --- | --- |
| `iregi-rust` | Immutable reduction plans, generic batch workspaces, tensor moments and numerator contraction |
| `iregi-masters` | Instruction-major oneloopmaster evaluation with separated UV/IR poles and Dual components |
| `iregi-prepare` | FeynKit DOT/model import, Feynman rules, Idenso/Spenso contractions and Symbolica polynomial extraction |
| `iregi-cli` | The standalone `iregi-rust` binary |
| `iregi-python` | Optional typed `tirust` bindings using the same Rust APIs and upstream shared Python classes |
| `feynkit-integrals` | Graph-independent affine partial fractions extracted from the pinned FeynKit source; planner integration is pending |

The reduction library does not depend on graph preparation, Python, MadGraph or
Fortran. Reference builds and generated physical-process code stay under
`target/`; the original reference checkouts are not modified.

## Python and notebooks

The optional [`tirust` Python interface](python/README.md) provides reusable
reduction plans, batched workspaces, dotted propagators, arbitrary precision and
Dual components. Numerator and graph preparation accept the upstream Symbolica,
FeynKit and Spenso types registered in the same native host. See the
[extension architecture](docs/research/python-interface.md) for source revisions
and the separate Community-host integration boundary.

The [marimo gallery](examples/notebooks/README.md) covers graph-to-integral
preparation, dots and derivatives, and an interactive benchmark explorer. Its
benchmark ratios always put TIRust in the numerator. Build and runtime
qualification, including retained failed attempts, is recorded separately in
[the benchmark report](docs/benchmark.pdf); the gallery does not imply that every
domain or platform is qualified.

## Build and run

Run commands from this repository. `shell.nix` pins the development environment;
`scripts/env.sh` also discovers an already installed compatible toolchain.

```sh
source scripts/env.sh
cargo build --release -p iregi-cli
cargo run --release -p iregi-cli -- run examples/graphs/bubble.dot \
  --model fixtures/models/phi3.json --input examples/inputs/bubble.json
cargo run --release -p iregi-cli -- run examples/graphs/triangle.dot \
  --model fixtures/models/phi3.json --input examples/inputs/triangle.json \
  --precision float --bits 256
```

Quote kinematic decimal inputs to preserve their precision. The comparison
fixture reader also preserves JSON number tokens exactly. See the
[CLI documentation](crates/iregi-cli/README.md) for batch JSON, complex squared
masses, physical numerator preparation, and reference comparisons. Dependencies
and exact revisions are pinned in `Cargo.toml` and `Cargo.lock`. Symbolica's
license requirements apply; its GMP/MPFR dependencies are permitted by the
project's implementation specification.

## Library interface

```rust
use iregi_rust::{KinematicsBatch, ReductionPlan, ReductionSettings, Workspace};
use symbolica::domains::float::Complex;

fn example() -> Result<(), Box<dyn std::error::Error>> {
let plan = ReductionPlan::new(ReductionSettings::new(2, 2))?;
let mut workspace = Workspace::new(&plan, 0.0_f64)?;
// Two shifts, two squared masses, then mu (not mu squared).
let row = [0., 0., 0., 0., 0., 1., 0., 0., 4., 9., 3.];
let batch = KinematicsBatch {
    rows: 1,
    components: 1,
    values: row.map(|value| Complex::new(value, 0.)).to_vec(),
};
let result = workspace.reduce_batch(&batch)?;
let moments = result.lane(0); // Failed lanes return a diagnostic, not zero data.
assert!(moments.is_ok());
Ok(())
}
```

Moments use increasing rank and IREGI's Cartesian monomial ordering, with
`[UV single, IR single, IR double, finite]` for each monomial. `Convention`
selects native IREGI, OneLOop, or MadLoop normalization. `TensorWeights` makes
multinomial weights explicit. `ContractionPlan` accepts ordinary monomial
coefficients and accounts for these choices.

Results retain their convention, tensor weights and complete Dual layout.
Contraction rejects incompatible metadata before evaluating coefficients.

`Workspace::new_dual` accepts Symbolica static or dynamic Dual layouts. Sample
lanes and Taylor components remain separate. `DoubleFloat` and arbitrary
precision `Float` use the same API, with constants converted directly at the
working precision. Independent workspaces own their numerical scratch storage.

The retained massive triangle and box cases also validate first, second and
mixed squared-mass derivatives against raised-power IBP reductions across all
four Laurent slots. See the [C0/D0 qualification](validation/c0d0-dual-identity/README.md)
for the tested inputs, precision settings and independent audits.

For `Float`, the workspace prototype sets constant precision; supplied values
retain their represented values and arithmetic precision. Accuracy requests
are never silently lowered. Checks use each contribution's tracked precision
and examine every returned Taylor coefficient. A conservative check on active
scalar-master arguments also protects finite parts that round to zero. It can
request `Unstable` even for an exact coincidence, an analytically unused scale,
or a tiny low-precision seed; precision rescue reevaluates the same represented
inputs. Supported constant scaleless branches are exempt only when every
non-scale Taylor coefficient is exactly zero. Uniform-precision arithmetic
reuses cached roundoff factors.

`InvariantInput::SuppliedScalarProducts` accepts the packed upper triangle of
`G_ij = p_i·p_j`, including the diagonal, between shifts and squared masses.
`ReductionPlan::input_layout()` describes the exact argument ranges. Supplied
invariants are authoritative; Cartesian shifts still specify tensor components.

Generalized PV and Hybrid plans select the reduction method for each scalar
term in the tensor expansion, as in original IREGI. With `e = sum(powers)-N`,
the existing canonical PV coefficients apply when the first power is one and
the term's dimension shift `h >= e`; other terms use IBP. This PV mapping is
qualified through four propagators. Generalized requests with more propagators
retain IBP and genuine shifted masters, including the six-dimensional pentagon,
and report that limitation in `preparation_notes()`. A mixed plan reports PV as
its primary method and records both methods' preparation counts. Hybrid retains
the independent IBP numerical fallback. The [qualification record](validation/generalized-pv/PROMOTION.md)
documents the tested cases and remaining coverage.

For generalized scalar integrals, `ScalarReductionPlan`,
`ScalarKinematicsBatch` and `ScalarWorkspace` accept only the packed scalar
products, squared masses and scale. Set rank zero in `ReductionSettings` and
choose the desired positive propagator powers and dimension shift. No momentum
coordinates are required. This interface uses the same batch executor and
supports the same ordinary and Dual numeric types.
These ordinary interfaces reject supplied-scalar-product inputs with six or more
propagators. Their high-point identities require four-dimensional Gram
constraints in every Taylor component. Coordinate-input topologies remain
available. The experimental `CertifiedScalarReductionPlan` and
`CertifiedScalarWorkspace` check those constraints before rank-zero IBP reduction
with six or seven propagators. Certificate tests pass, and the bounded six-point
scalar fixture converges at 128 and 256 bits against an independent parameter
integral, at requested binary64 accuracy. Its binary64 cancellation rejection
and earlier preparation timeouts remain recorded. This does not qualify all
six/seven-point requests or high-rank tensors; see the
[bounded qualification](validation/invariant-certificate/solve-vector/qualification-status.json).

`CertifiedPlanarReductionPlan` and `CertifiedPlanarWorkspace<T>` provide an
explicit PV path for planar five-point tensors through rank seven. They certify
the routing determinant in every requested Taylor component and require a
regular constant-one chart before grouping accepted samples for batch reduction.
Inputs remain unchanged; transverse jets and unresolved or unstable lanes retain
diagnostics. This interface supports unit powers, dimension shift zero and
Cartesian inputs. Ordinary reduction does not invoke it automatically. See the
[API and qualification](docs/research/certified-planar-api.md) for supported
settings, precision convergence and independent mixed-Dual comparisons.

`SixPointReductionPlan` and `SixPointWorkspace<T>` provide an explicit batched
six-point path through rank three, using preserved Cartesian shifts, unit
powers and native raw moments. Its retained Float128/Float256 fixtures pass
complete-array reference and convergence checks; failed neighbors, workspace
reuse and numerator contractions have separate lifecycle checks. The ordinary
reducer's defaults remain unchanged. See the [six-point API](docs/six-point.md)
for its input layout, example, numerical scope and remaining precision/Dual
qualification.

Derived scalar invariants are checked before selecting analytic branches or
evaluating masters. Exact represented-input polynomial checks prevent a rounded
zero from selecting a scaleless formula. Relative conditioning estimates also
detect severe cancellation in nonzero invariants, including Taylor components
and mixed input precisions. A rejected lane returns `Unstable` and can be retried
through precision rescue; the conditioning estimates are not certified error
bounds.

Plans cache exact prepared instructions separately for each ordered Dual shape.
Workspaces at different precisions share this symbolic preparation while mapping
constants directly into their own numeric domain and retaining separate scratch
buffers. Numerical sample results are never stored in that cache.

Ordinary scalar masters, reduction evaluators and numerator contractions reuse
temporary storage through Symbolica's existing root allocator. The instruction
and arithmetic order stay unchanged; Dual
preparation keeps its existing allocation path. The [paired qualification](validation/stack-reuse/production-factory/integration-qualification.json)
records master output agreement; the [core qualification](validation/core-stack-reuse/adoption/qualification.json)
records the later reduction/contraction integration. Controlled performance
gates are assessed separately.

`PrecisionRescueWorkspace` provides an optional accuracy check and recovery
path. Its default settings retry unstable samples together at 128 and 256 bits,
require every Laurent and Taylor component to converge, and only then round to
the caller's output type. `RescueTrigger::VerifyAllSuccessful` also checks results
that passed the ordinary stability estimates. Per-sample rescue diagnostics
distinguish these cases; convergence is not a certified error bound.

The generated Rust scalar-master kernels can be selected with
`--features iregi-masters/native-batch`. They use the same batched interfaces;
Dual evaluation continues through Symbolica's Dualizer. This backend remains
opt-in while end-to-end performance qualification is in progress.

`iregi_masters::certified::AccuracyMasterEvaluator` provides an explicit
accuracy-aware interface for finite D0 masters near the equal-mass vacuum.
It accepts exact rational error budgets and returns per-component enclosures
for accepted lanes. Orders two and four run across the eligible batch, including
supported Taylor components. Ordinary fallback requires an explicit policy and
is labelled uncertified. These bounds cover the represented master inputs;
upstream routing arithmetic requires its own enclosure. See the
[API contract](docs/research/certified-masters-api.md) for limits and validation.

## Validation and reporting

```sh
source scripts/env.sh
bash scripts/build-reference.sh
python3 validation/reference/capture.py --write-fixtures
python3 validation/reference/audit.py
cargo test --workspace -- --test-threads=1
bash scripts/build-benchmark-report.sh
```

Tests using Symbolica must respect the installed license's process and thread
limits. The cold concurrent plan-preparation test is ignored because it requires
a multithread-enabled Symbolica license; preparing a variant first allows the
separate concurrent cache-lookup test to use only Rust synchronization.
The [licensed concurrency qualification](validation/concurrent-workspaces/README.md)
records the explicit cold test and 360 compared calls from independent concurrent
workspaces. It contains no license key.
Independent validation procedures and conventions are documented in
[validation/reference](validation/reference/README.md),
[validation/physical](validation/physical/README.md), and
[validation/pv-high-rank](validation/pv-high-rank/README.md).
