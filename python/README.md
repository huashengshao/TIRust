# TIRust for Python

Status: a local development wheel is qualified on Linux with CPython 3.13.
All 29 focused numerical and native interoperability tests pass, including the
actual FeynKit SVG renderer; installed-wheel consumer typing also passes with
zero errors or warnings. The [qualification evidence](../validation/python-interface/checkpoint-wheel-004/README.md)
records the exact wheel, source archive, dependency pins and tested coverage.
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

`Dual(shape, values)` carries Symbolica Taylor components without introducing a
second arithmetic implementation. Construct a workspace with the exact
`dual_shape`; scalar arguments receive zero higher components. Sample lanes and
Taylor components remain separate throughout the Rust and scalar-master batches.

Each lane has independent status, diagnostics and fallback flags. `result.lane(i)`
raises `LaneError` for a failed sample, while `result.lanes()` returns `None` in
that position. Failed or unsupported results never become successful zeros.

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
