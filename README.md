# iregi-rust

A Rust implementation of one-loop tensor reduction, using Symbolica for algebra
and batched numerical evaluation, and oneloopmaster for scalar integrals.

Development follows [PLAN.md](PLAN.md). The documented one- through six-point,
cumulative rank-six functional checkpoint is independently validated, including
dotted propagators and dimension shifts.
The [benchmark summary](docs/benchmark.pdf) shows the summary tables (built
from [docs/benchmark-summary.typ](docs/benchmark-summary.typ) by
`scripts/build-benchmark-report.sh`). The full benchmark report, which retains
measurements, failed attempts, reference defects and explicit qualification
limits, and its evidence are assets of the release
[tirust-bench-20261106](https://github.com/huashengshao/TIRust/releases/tag/tirust-bench-20261106);
the validation evidence is in the private backup named in [validation/README.md](validation/README.md).

**Direct numerical engine (binary64 default).** `TensorReductionPlan::new` now
selects a direct engine for unit-power or single-dotted, dimension-shifted
(up to four points), Cartesian-shift or supplied-invariant PV/Hybrid requests: the validated PV recurrences run
numerically per lane with oneloopmaster's direct scalar kernels, affinely dependent
sub-topologies use their constant-one denominator identity, a dotted propagator
is a Taylor jet in its squared mass, and `D=4+2h` uses
`C(lambda+h,kappa)(-2)^h`. Lanes with ill-conditioned systems fall back to the
symbolic engine, which is prepared only on demand (`new_without_direct` keeps the
old dispatch). In the [same-session benchmark](validation/README.md) (private backup: `validation/benchmark/direct-engine-20261030/README.md`)
TIRust is faster than original IREGI and COLLIER on all 18 opening inputs, singly
and in batches of 32: in the latest capture TIRust/IREGI 0.0008-0.36 and
TIRust/COLLIER 0.06-0.68, with full-array agreement against the archived
references (E/A <= 1.8e-11). Massless external legs (lightlike bubbles) are also
handled directly, cross-checked against original IREGI.

The full report opens with this comparison. The historical opening tables follow:
representative integral timings through six points and
rank six of the previous engine, followed by dotted and dimension-shifted requests outside COLLIER's
direct integral API. Large coloured cells separate TIRust time from nested
scalar-master evaluation. Exact inputs, raw calls, preparation costs, precision
and reproduction commands are in the
[binary64 opening timing suite](validation/README.md) (private backup: `validation/benchmark/opening-double-20261026/README.md`).
Both opening tables use double precision throughout. They show the fastest
measured configured path per input, with the achieved full-array relative
discrepancy alongside each total. The ordinary table includes COLLIER's total
time in the right-most column. The three ordinary N5/R5, N5/R6 and N6/R6 rows
request approximately `1e-6` accuracy from both tools. Internal warning flags
alone do not suppress timings; outputs worse than three relative digits or
nonfinite results remain explicit. Earlier mixed-precision captures and their
stricter admission decisions remain archived unchanged.
Two rank-six dotted binary64 requests hit the retained cancellation guard;
their cells explicitly show diagnostic timings with unavailable coefficients.
Their higher-precision functional validation remains separate evidence.

The [final checkpoint](docs/checkpoints/rank6-functional-completion-20261026.md)
records ordinary dotted/shifted tensors and the completed six-point, rank-six
second-mass-derivative Dual validation. For that witness, all 2,520 complex fields
per unique array at 128, 256 and 384 working bits agree with both independent
defining-integral
rules at the unchanged `1e-10` tolerance. The strict 256/384 comparison passes all
fields at `2^-80`; the earlier 128/256 failure in 628 fields remains archived.
Complete batch lifecycles, owning higher-component error contexts, invalid isolation,
empty reset and bit-identical recovery pass independent replay. See the
[handoff](HANDOFF.md) for evidence, reproduction and future work.

Tensor rank and propagator count remain separate. The API uses dynamically
sized denominator descent with no fixed point-count ceiling; this finite
qualification covers the documented N≤6/rank≤6 families. Arbitrary point counts,
every exceptional chart and every derivative neighborhood need their own
evidence. Historical rank-seven and higher-point checks remain in the report;
additional rank-seven work is deferred. Physical comparisons retain their
specified graph/rank-four scope, and the current Python smoke includes an N6
rank-one lifecycle case. Each family's validated domain is documented separately.

## Inputs, embedder mode and capabilities

- `InvariantInput::SuppliedInvariants`: the invariants `s_ij = (p_i - p_j)^2`
  (the S-matrix of the shifts, `i < j`, packed row-major after the shifts;
  `InputLayout::of`). Every scalar the reduction forms from a pair of shifts
  (Gram and Cayley systems, scalar-master arguments, the lightlike test) reads
  them exactly as given, so an on-shell invariant equal to a squared mass stays
  equal to it bit for bit and the masters take their on-shell branches in every
  precision; the shifts still give the tensor components and the vector
  geometry. The direct engine and the generic symbolic engine serve them at
  every point count (from six points on, the symbolic denominator identities
  take their vector rows from the shifts and their constant row from the
  supplied invariants; the specialised six- and seven-point plans take
  Cartesian shifts only). `tests/supplied_invariants.rs`:
  double-double equals 256-bit MPFR within 4e-32 on the IR-divergent
  (0, m_b, m_t) vertex triangle of b b~ > t t~ at a binary64-rounded point,
  binary64 within 5e-16, while the invariants of the rounded shifts lose its
  soft pole.
- `ScaleInput::MuSquared` (`ReductionSettings::scale_input`): the last
  argument is mu^2, used exactly; the direct engine reads it at every
  precision, the facade forms mu = sqrt(mu^2) once for the other engines.
- Embedder mode, `TensorWorkspace::new_embedded(plan, prototype,
  EmbedderOptions)`: the direct engine at the workspace's precision
  (binary64, double-double or MPFR) with its conditioning gates, then only what
  is configured: the a priori method choice for gate-refused lanes (on by
  default, unchecked and flagged per lane in `last_lane_routes`), partial
  double-double rescue and precision tiers (binary64, off by default). No
  loop-rotation test, no rescue-first routing, never the symbolic engine;
  settings outside the direct engine are refused (`tests/embedder_mode.rs`).
- The lightlike bubble rule is relative to the inputs' precision: |p^2| up
  to 2^-(p - 13) of the bubble's scale (2^-40 for binary64 inputs, also in the
  rescue tiers that lift them; `DirectWorkspace::set_input_precision`).
- `DirectReductionPlan::capabilities(&prototype)` (`DirectCapabilities`): the
  largest point count (64, the subset-mask width), rank (C(R+4,4) within the
  u32 monomial indices: 564), raised power (2^(64/N) - 1 IBP key fields,
  at most the settings' u16) and dimension shift (R + 2h within the u16
  coefficient ranks: `max_dimension_shift(rank)`), plus the algorithmic
  limits (one raised power, through six points; shifts through four points);
  `admits(&settings)` is the same check as `supports`. The bounds are the same
  for every carrier: a lane whose coefficients leave the carrier's exponent
  range (binary64 near h = 700) is refused per lane, typed. Rank and point
  count are otherwise limited by memory (a plan whose 2^N subsets cannot be
  allocated is refused typed) and the achieved precision
  (`tests/lifted_caps.rs`: rank 33 and 36 tadpoles against their closed form,
  raised powers 6 and 7 and h = 5 against independent references, 13- and
  17-point families evaluated and checked against a reordering of their
  propagators). The validated domain N <= 6, R <= 6 is documentation, not a
  limit. The generic symbolic engine has no preparation budget; the opt-in
  certified affine route serves three points only, which bound its chart
  enumeration, and the affine-power planner takes the caller's
  `AffinePowerBudget`.

## Tensor store, rotation copy and concurrency

- `TensorStore` (`DirectWorkspace::set_tensor_store`,
  `TensorWorkspace::set_tensor_store` for the facade and embedder mode): the
  per-subset results of `AmplitudeWorkspace` kept across calls until the caller
  clears them (`TensorStore::clear`, once per chunk of points), shared by any
  number of workspaces of different plans and threads. An embedder that
  reduces a parent family and its pinches in separate calls (one per (N, R)
  bucket) shares binary64 scalar masters, PV sub-integrals, Gram-free
  pentagons, partial rescues, small-Gram expansions and gathered moments, and
  double-double masters. Entries are pure functions of exact keys: results are
  bit-identical with and without the store, in any call order and thread
  count (unless `TensorStore::align_ranks` raised the rank floor, which moves
  injected boxes and triangles at their truncation level, or a masters
  provider returns other values than TIRust's kernels). Master keys hold
  invariants, masses and mu^2 only (translation-invariant: with
  `SuppliedInvariants` a box meets its copy in another frame);
  sub-integral and moment keys hold the shifts, so parent and pinches meet in
  an aligned loop-momentum frame supplied by the caller (common propagators
  with equal shifts, in the same relative order). `TensorStoreOptions`:
  `byte_budget` (none by default; stops inserting, values unchanged) and
  `masters`, a `MasterProvider` serving binary64 scalar masters from the
  embedder's own cache (oneloopmaster's argument order, OneLOop
  normalisation). Check passes never consult the provider: the rotation copy
  and the facade's loop-rotation test and cyclic-order checks reduce through
  `TensorStore::check_view` (TIRust's kernels at the store's rank floors, no
  entry read), since a provider keyed on a canonical propagator order would
  hand them the primary's boxes and triangles bit for bit and blind them to
  the masters' rounding; an embedder attaches `check_view()` to the
  workspaces of its own check passes. `TensorStore::align_ranks` raises the
  rank floor of injected boxes and triangles as `AmplitudeWorkspace` does
  (opt-in); `TensorStore::statistics` counts reuse per kind
  (`tests/tensor_store.rs`).
- `DirectWorkspace::reduce_batch_with_rotation(batch, rotation)`: the primary
  moments and, from the same call, those of the batch with its propagators
  relabelled cyclically (`DirectReductionPlan::rotated_batch`), the same
  integrals in the same frame through other PV systems: a stability check
  without a second call or a re-expanded numerator. The copy is a check pass:
  without a masters provider it is bit for bit a separate call on the rotated
  batch and the subsets that keep their order are computed once; with one it
  is computed in full by TIRust's kernels (`tests/rotation_copy.rs`).
- Concurrency: `DirectWorkspace` is `Send`, `DirectReductionPlan` and
  `TensorStore` are `Send + Sync`; one workspace per thread gives every lane
  bit for bit what one workspace gives on one thread, for any thread count and
  batch split, with or without a store shared by the threads (binary64 gates,
  method choice and partial rescue; double-double;
  `tests/direct_concurrency.rs`).
- `examples/store_sharing_timing.rs` times a top-loop gg -> gggg hexagon and its
  56 pinches reduced in one call per propagator count, with and without the
  store, against one `AmplitudeWorkspace` call, and the rotation copy against
  two calls. On 32 RAMBO points (binary64, one pinned core, 7 November 2026,
  medians of 9 rounds), the store across the five calls in the hexagon's
  frame was 1.23x (hexagon first) and 1.27x (bubbles first) faster than no
  store with the conditioning gates alone, 2.15x and 2.14x in embedder mode
  (the method choice for gate-refused lanes) and 1.13x and 1.14x on the
  facade, where one `AmplitudeWorkspace` call gave 1.14x. With every integral
  in its own frame (its last propagator at the origin) and supplied
  invariants: 1.30x, 1.95x and 1.16x. That frame is partially aligned:
  integrals with the same last propagator share their shifts bit for bit (the
  30 pinches containing the hexagon's last propagator are in the hexagon's
  own frame), so their sub-integrals and moments meet as well; between the
  others only the scalar masters meet. The rotation copy took 0.91x (gates)
  and 0.69x (method choice) of two calls' time.

## Workspace

| Crate | Purpose |
| --- | --- |
| `iregi-rust` | Immutable reduction plans, generic batch workspaces, tensor moments and numerator contraction |
| `iregi-masters` | Instruction-major oneloopmaster evaluation with separated UV/IR poles and Dual components |
| `iregi-prepare` | FeynKit DOT/model import, Feynman rules, Idenso/Spenso contractions and Symbolica polynomial extraction |
| `iregi-cli` | The standalone `iregi-rust` binary |
| `iregi-python` | Optional typed `tirust` bindings using the same Rust APIs and upstream shared Python classes |
| `feynkit-integrals` | Pinned FeynKit affine partial-fraction facilities; planner integration remains separate |

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
The [current restored-source wheel](validation/README.md) (private backup: `validation/python-interface/restored-zero-profile-wheel-refresh-smoke/README.md`)
passes ten installed API smoke cases, including precision/Dual transport,
checked rejection and a six-point rank-one workspace lifecycle. The earlier
[65 API cases, seven notebook runs and consumer typing](validation/README.md) (private backup: `validation/python-interface/facade-wheel-004/README.md`)
remain historical evidence. The standalone native host is qualified; loading
the extension in an external Symbolica Community host remains separate work.

The [marimo gallery](examples/notebooks/README.md) covers graph-to-integral
preparation, dots and derivatives, and an interactive benchmark explorer. Its
benchmark ratios always put TIRust in the numerator. Build and runtime
qualification, including retained failed attempts, is recorded separately in
the full benchmark report (release
[tirust-bench-20261106](https://github.com/huashengshao/TIRust/releases/tag/tirust-bench-20261106));
the gallery does not imply that every domain or platform is qualified.

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

The CLI uses `TensorReductionPlan` and `TensorWorkspace<T>` to select the reduction
engine and exposes its actual counters and per-lane diagnostics. Preserved
Cartesian positive-power inputs can use a denominator-descent DAG before the
low-point tensor planner. This separates point count from the rank-six target;
each accepted numerical chart still needs a fresh certificate. Explicit resource
budgets and the tested point counts remain part of the supported domain.
`TensorRescueWorkspace` keeps the selected ordinary engine when supported and
checks a second working precision before accepting a replacement result.

## Library interface

```rust
use iregi_rust::{KinematicsBatch, ReductionSettings, TensorReductionPlan, TensorWorkspace};
use symbolica::domains::float::Complex;

fn example() -> Result<(), Box<dyn std::error::Error>> {
let plan = TensorReductionPlan::new(ReductionSettings::new(2, 2))?;
let mut workspace = TensorWorkspace::new(&plan, 0.0_f64)?;
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

For binary64 workspaces `TensorReductionPlan::new` selects the direct numerical
engine whenever the request is in its domain (unit powers or one raised power up
to six points, `h>=0` up to four points, Cartesian shifts or supplied
invariants, `Preserve`, PV/Hybrid); its bounds come from its representation,
not from fixed caps, and `DirectReductionPlan::capabilities` reports them (see
"Inputs, embedder mode and capabilities" below);
`engine_kind()` reports `Direct`, and `engine_execution()` reports direct
successes, fallback rows and the fallback's own detail. Binary64 lanes are
checked by conditioning gates and, for four or more propagators with equal
powers, a loop-rotation test (the same integral with relabelled propagators,
tolerance the requested accuracy, 2^-34 by default; a lane within 4 bits of
the tolerance must also agree in a third cyclic order). Refused lanes where
COLLIER's a priori PV estimate 2^-52 w^r exceeds the request for some box or
triangle (five to seven propagators, rank <= 7) are first evaluated again with
those boxes and triangles off PV. Lanes the a priori gate refuses are
evaluated again below it, then with refused boxes and triangles
computed by the small-Gram expansions of Denner and Dittmaier (hep-ph/0509141:
small Gram determinant, and small Gram with small modified Cayley
determinant; pinches 8, then 12 ranks higher, still binary64) or the
alternative PV reduction, and kept when the rotated copies agree.
By default, boxes and triangles refused inside a lane are computed in
double-double, and lanes still refused are rerun by the direct engine at 106,
256 or 512 bits, then on the symbolic engine.
`TensorWorkspaceOptions::precision_rescue = PrecisionRescue::Disabled` (or
`TensorWorkspace::set_precision_rescue`; CLI `--no-precision-rescue`; Python
`workspace.set_precision_rescue(False)`) keeps every lane in binary64: refused
lanes are returned `Unstable`, with the reason in `last_diagnostics()`, and,
like COLLIER's results with their accuracy flag set, they keep the best binary64
array computed for them when one exists: `BatchResult::lane_with_estimate`
returns it uncertified with its estimated relative error (`estimated_errors`:
the E/A between the array and its loop-rotated copy, an estimate, not a bound;
also given for lanes that passed the rotation test). CLI: `estimated_error` and
`best_effort_moments` per sample; Python: `estimated_errors`, `best_effort` and
`lane(row, best_effort=True)`. A looser
accuracy request (`set_accuracy_precision(bits)`, tolerance 2^(19-bits)) lets
the binary64-only path keep more lanes. `TensorReductionPlan::new_without_direct`
selects the symbolic dispatch explicitly. `DoubleFloat`, `Float` and Dual
workspaces use the symbolic engine at their own precision. The lower-level
`DirectReductionPlan` / `DirectWorkspace` API is also public.

The integrals of an amplitude share boxes and triangles (a pinched integral's
are its parent's), and on physical kinematics most of the binary64 cost is
per box or triangle: refused ones computed in double-double, and the
method-choice retries' small-Gram expansions. `AmplitudeWorkspace` evaluates
several integrals at the same phase-space points (one plan and one batch per
integral, `reduce_batches`) and computes each such per-subset result once per
call, under an exact key (method settings, rank, the subset's propagators bit
for bit), so results are bit-identical with and without sharing.
`AmplitudeOptions::share_subsets` (default `true`) or `set_subset_sharing`
turns it off; `last_sharing()` counts reused and computed results. A top-loop
gg -> gggg hexagon with its 56 pinches on 32 physical points:
1.7-2.2x faster per call
(`crates/iregi-rust/examples/amplitude_sharing.rs`,
`crates/iregi-rust/tests/amplitude_sharing.rs`).

Moments use increasing rank and IREGI's Cartesian monomial ordering, with
`[UV single, IR single, IR double, finite]` for each monomial. `Convention`
selects native IREGI, OneLOop, or MadLoop normalization. `TensorWeights` makes
multinomial weights explicit. `ContractionPlan` accepts ordinary monomial
coefficients and accounts for these choices.

UV and IR poles stay separate. `TensorWorkspace::set_regulators` (CLI
`--mu-uv2/--mu-ir2/--delta-uv/--delta-ir1/--delta-ir2/--collier-normalization`,
Python `workspace.set_regulators(...)`) adds COLLIER's regulator parameters:
separate UV and IR scales instead of each lane's mu^2, numerical pole values
added to the finite slot, and optionally COLLIER's c(eps) normalization of the
IR double pole (with `Convention::OneLoop`). The default changes nothing. The
massless on-shell box reproduces COLLIER 1.2.10 to 1e-11 at two IR scales
(`crates/iregi-rust/tests/regulators.rs`).

Results retain their convention, tensor weights and complete Dual layout.
Contraction rejects incompatible metadata before evaluating coefficients.

`TensorWorkspace::new_dual` accepts Symbolica static or dynamic Dual layouts. Sample
lanes and Taylor components remain separate. `DoubleFloat` and arbitrary
precision `Float` use the same API, with constants converted directly at the
working precision. Independent workspaces own their numerical scratch storage.
The lower-level `ReductionPlan` and `Workspace<T>` APIs remain available when
the generic engine is explicitly required. Use the shared tensor facade for
automatic point-count reduction. Workspace capability and engine selection are
observable; API acceptance does not establish numerical qualification of every
combination of rank, powers, precision and kinematics.

The optional [`native-f64-abc` policy](docs/native-masters.md) selects generated
oneloopmaster A0/B0/C0 evaluators for ordinary binary64 workspaces through
`Workspace::new_with_master_policy`. Other numeric domains and scalar families
retain Symbolica under this policy. It does not change default constructors;
qualification and per-family performance remain explicit in the report.

The retained massive triangle and box cases also validate first, second and
mixed squared-mass derivatives against raised-power IBP reductions across all
four Laurent slots. See the [C0/D0 qualification](validation/README.md) (private backup: `validation/c0d0-dual-identity/README.md`)
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
invariants are authoritative for ordinary lower-point plans; Cartesian shifts
still specify tensor components. Checked high-point tensor routes additionally
require fresh full-Taylor compatibility with the four-dimensional routing.

Generalized PV and Hybrid plans select the reduction method for each scalar
term in the tensor expansion, as in original IREGI. With `e = sum(powers)-N`,
the existing canonical PV coefficients apply when the first power is one and
the term's dimension shift `h >= e`; other terms use IBP. This PV mapping is
qualified through four propagators. Generalized requests with more propagators
retain IBP and genuine shifted masters, including the six-dimensional pentagon,
and report that limitation in `preparation_notes()`. A mixed plan reports PV as
its primary method and records both methods' preparation counts. Hybrid retains
the independent IBP numerical fallback. The [qualification record](validation/README.md) (private backup: `validation/generalized-pv/PROMOTION.md`)
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
[bounded qualification](validation/README.md) (private backup: `validation/invariant-certificate/solve-vector/qualification-status.json`).

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
six-point path through rank seven, using preserved Cartesian shifts, unit
powers and native raw moments. Its retained Float128/Float256 fixtures pass
complete-array reference and convergence checks. The retained rank-seven
lifecycle also covers failed neighbors, reordered/chunked calls, scratch reuse
and sparse/dense numerator contractions. Concurrent cloned workspaces remain a
separate validation gate. The ordinary
reducer's defaults remain unchanged. See the [six-point API](docs/six-point.md)
for its input layout, example, numerical scope and remaining precision/Dual
qualification.

`SevenPointReductionPlan` and `SevenPointWorkspace<T>` provide a restricted
explicit scalar path through all 21 five-point subsets. Its current Cargo
Float128/Float256 singleton arrays pass reference and convergence checks on
one real-mass family; the complete workspace lifecycle remains incomplete.
Higher seven-point tensor ranks are still under validation and are not exposed
by this wrapper. See the [seven-point API](docs/seven-point.md) for the exact
settings and preserved failures. Generic rank-six functionality, including dots,
is the development priority. Tested point counts remain explicit; existing
six-point tensors do not establish arbitrary-point correctness. Historical
rank-seven cases remain challenge coverage.

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
preparation keeps its existing allocation path. The [paired qualification](validation/README.md) (private backup: `validation/stack-reuse/production-factory/integration-qualification.json`)
records master output agreement; the [core qualification](validation/README.md) (private backup: `validation/core-stack-reuse/adoption/qualification.json`)
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
cargo test -- --test-threads=1
bash scripts/build-benchmark-report.sh          # docs/benchmark.pdf from docs/benchmark-summary.typ
bash scripts/build-benchmark-report.sh --check  # byte-identical rebuild of the committed PDF
```

`scripts/build-benchmark-summary.py` writes `docs/benchmark-summary.json` from
one set of local captures. The PDF build fixes the creation timestamp from that
data (the report date) and uses only Typst's embedded fonts, so it is
reproducible. The reference builder and capture scripts that produced the kept
fixtures under `validation/` (formerly `validation/reference/{build,capture,audit}.py`
and `scripts/build-reference.sh`) are in the private backup named in [validation/README.md](validation/README.md).

`cargo test` runs the default members; `crates/iregi-python` (the Python
extension, gammaloop `9102f885`) builds with `cargo check --workspace` (see
[docs/environment.md](docs/environment.md)).

Tests using Symbolica must respect the installed license's process and thread
limits. The cold concurrent plan-preparation test is ignored because it requires
a multithread-enabled Symbolica license; preparing a variant first allows the
separate concurrent cache-lookup test to use only Rust synchronization.
The [licensed concurrency qualification](validation/README.md) (private backup: `validation/concurrent-workspaces/README.md`)
records the explicit cold test and 360 compared calls from independent concurrent
workspaces. It contains no license key.
Independent validation procedures and conventions are documented in
`validation/reference/README.md`, `validation/physical/README.md` and
`validation/pv-high-rank/README.md` in the [private backup](validation/README.md),
and in [docs/conventions.typ](docs/conventions.typ).

## Licence

TIRust's own code is under the BSD Zero Clause License ([LICENSE](LICENSE)),
including the scalar masters generated from oneloopmaster's expressions. The
third-party content listed in [LICENSE](LICENSE) (feynkit-integrals, linnest,
the bundled fonts and licence texts, the mirrored GammaLoop type stubs, copied
fixtures and reference captures)
keeps its own licence and notices. Dependencies fetched by Cargo, Symbolica in
particular, carry their own licences.
