# TIRust for Python

Status: a fresh local development wheel is qualified on Linux with CPython 3.13.
All 65 focused numerical, precision, native interoperability and facade cases
pass, together with seven actual marimo notebook runtime cases. Installed-wheel
consumer typing passes with zero errors or warnings.
The [qualification evidence](../validation/python-interface/facade-wheel-004/README.md)
records the exact wheel, 201 selected source files including all 158 coherent
build bodies, tested coverage and preserved
failed attempts. These are frontend checks on specified fixtures; full-array
integral validation is reported separately.
Release wheels, full Symbolica Community host integration, and other Python or
platform versions still require their own qualification.

TIRust provides reusable plans and genuine batched one-loop tensor reduction.
Its Python interface links the same Rust implementation and one Symbolica kernel
with the upstream graph, tensor and numerical types.

Build the standalone wheel from the repository root:

```bash
source scripts/env.sh
python -m maturin build --manifest-path crates/iregi-python/Cargo.toml --features standalone,native
```

Use the wheel's own shared types together:

```python
from tirust import ReductionPlan

plan = ReductionPlan(2, 1, powers=[1, 1])
workspace = plan.workspace(dtype="float", precision=128, accuracy_bits=53)
batch = plan.pack(
    shifts=[[[0, 0, 0, 0], [0, "1/2", 0, 0]]],
    masses_squared=[["1", "2"]],
    mu=["1"],
)
result = workspace.reduce_batch(batch)
print(result.statuses)
if result.success[0]:
    print(result.lane(0))
```

Inputs use the `+---` metric, time-first shifts, **squared** masses and the scale
**mu**, rather than mu-squared. Native Laurent output is ordered UV single, IR
single, IR double, finite. Tensor moments are ordered by increasing rank, then
decreasing time, x and y powers. `plan.basis` and `plan.input_layout` make both
layouts explicit. `dimension_shift=h` means `D=4+2h-2ε`.

`dtype="f64"`, `"double-double"` and `"float"` select 53-bit, 106-bit and arbitrary
precision arithmetic. Decimal/rational strings and Symbolica `Float` values
preserve digits beyond binary64. Existing Float inputs retain their represented
values and precision; increasing workspace precision cannot recover missing
input digits. Nonbinary64 results are upstream `ComplexFloat` objects. A
DoubleFloat result may require more than 106 storage bits to preserve an actual
wide-gap high/low pair; its arithmetic working precision remains 106.

When an upstream `ComplexFloat` has different real and imaginary precisions,
native Python numeric operands preserve the more precise component during
conversion. Existing `Float` operands retain their own precision, and the
complex `.precision` property still reports the smaller component precision.

`Dual(shape, values)` carries Symbolica Taylor components without introducing a
second arithmetic implementation. Construct a workspace with the exact
`dual_shape`; scalar arguments receive zero higher components. Sample lanes and
Taylor components remain separate throughout the Rust and scalar-master batches.

Each lane has independent status, diagnostics and fallback flags. `result.lane(i)`
raises `LaneError` for a failed sample, while `result.lanes()` returns `None` in
that position. Failed or unsupported results never become successful zeros.

The source bindings use the shared Rust `TensorReductionPlan` and
`TensorWorkspace` facade. It selects eligible ordinary PV high-point paths;
eligible ordinary high-point inputs use affine-power descent for explicit
Dual layouts and `profiling=True`. Lower-point inputs retain generic execution.
The requested
algorithm and compiled default scalar-master policy are preserved. Inspect
`workspace.engine` and `workspace.preparation_notes` for the actual workspace
selection. `result.engine_execution` captures its tagged Rust counters, including
separate child, parent and certificate work. The existing `result.stats` summary
counts logical input samples once. The qualified wheel exercises generic
profiling and explicit Dual layouts, plus ordinary six-point rank-one and
seven-point scalar dispatch on retained positive-mass fixtures. These frontend
checks do not establish all tensor, numeric-type or exceptional-kinematics
combinations. The project targets generic rank-six reduction, including positive
propagator powers and dimension shifts, with propagator-count reduction kept
separate. Six-point coverage is a concrete validation checkpoint; supported and
tested point counts remain explicit. Historical rank-seven cases remain
separate challenge coverage.

Dotted inputs may select the `mass_taylor` engine. Its tagged execution details
keep user components distinct from internal Taylor components and retain the
requested algorithm alongside the actual unit-integral procedure. The existing
dotted-bubble/mass-Dual identity passes in the current wheel.

For eligible high-point `input_policy="legacy_on_shell"` requests, the
`legacy_noop` engine proves that a conservative superset of snapping queries
leaves each admitted input unchanged before executing its Preserve child. A
changed neighbor is rejected before child execution. Nested metadata reports
the actual child and admitted/rejected rows. The current N5 scalar frontend
regression covers mixed-lane isolation, unchanged input containers, empty calls
and recovery; broader numerical coverage is reported separately.

The source API also offers
`ReductionPlan(6, rank, invariants="scalar_products", check_scalar_products=True)`
to check supplied products against the explicit Cartesian shifts before
reduction. Its `certified_tensor` engine keeps Cartesian and Gram certificate
counters separate, including with profiling or explicit Dual layouts. This
option is tested in the current wheel: incompatible inputs are rejected before
scalar-master evaluation for ordinary, profiling and primal-only Dual
workspaces, and empty calls reset the certificate counters. Rank-one numerical
evidence is separate. The original inline rank-six path exceeded its preparation
resource deadline; these frontend rejection checks do not establish complete
checked rank-six, generalized-integral or checked-rescue numerical coverage.

`PreparedNumerator.from_expression` accepts actual shared Symbolica Expressions;
`from_tensor` accepts actual shared Spenso TensorExpression objects. Preparation,
coefficient evaluation and contraction call Rust/Symbolica APIs directly.
`PreparedGraph.from_diagram` retains the supplied FeynKit diagram object and
prepares supplied or Feynman-rule numerators. `from_dot(model, dot)` imports DOT
using the actual shared native Model. See the marimo notebooks for complete
examples and honest benchmark coverage.

The standalone wheel's `Expression`, `Float`, `ComplexFloat`, `Model`,
`FeynmanDiagram` and `TensorExpression` are the real upstream classes registered
in the **same native host**. Do not mix them with objects from a separately built
installed Symbolica Community wheel. The bindings remain an `rlib` suitable for
future Community integration; integration into that complete host is a separate
qualification step.

The standalone package includes the unchanged rendering fonts required by the
same pinned FeynKit/Spenso owners, with hashes and licenses under `tirust/data`.
Its facade selects those package assets; a Community host keeps its own data
configuration. `configure_data_directory(path)` can explicitly select and
validate a replacement bundle against the upstream compiled asset hashes.
