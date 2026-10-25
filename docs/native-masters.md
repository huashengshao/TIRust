# Optional generated binary64 scalar masters

The `native-f64-abc` feature lets a workspace select oneloopmaster's generated
Rust A0, B0 and C0 evaluators for ordinary `f64` inputs. The generated formulas
come from the existing Symbolica/oneloopmaster algebra and retain instruction
batching through the scalar-master calls.

Enable it for a Rust dependency:

```toml
iregi-rust = { path = "path/to/iregi-rust/crates/iregi-rust", features = ["native-f64-abc"] }
```

Then select the policy when creating scratch storage:

```rust
use iregi_rust::{
    MasterExecutionPolicy, ReductionPlan, ReductionSettings, Workspace,
};

fn example() -> Result<(), iregi_rust::ReductionError> {
    let plan = ReductionPlan::new(ReductionSettings::new(3, 1))?;
    let mut workspace = Workspace::new_with_master_policy(
        &plan,
        0.0_f64,
        MasterExecutionPolicy::NativeF64Abc,
    )?;
    // workspace.reduce_batch(&kinematics)? returns the complete tensor array.
    Ok(())
}
```

| Policy | Behavior |
| --- | --- |
| `Default` | Preserves the existing constructor's backend selection. |
| `Symbolica` | Explicitly selects the Symbolica batch evaluator. |
| `NativeF64Abc` | Uses generated A0/B0/C0 for ordinary `f64`; other families and numeric domains use Symbolica. Requires the feature. |

D0, the bubble derivative, DoubleFloat, arbitrary-precision Float and every
static or dynamic Dual layout use Symbolica under the selective policy. This
includes a Dual containing only its constant component. Selecting the policy
without the feature returns an error before fallback or evaluator preparation.

`Workspace::new_dual_with_master_policy` accepts the policy alongside the
validated Dual layout. Policy and ordered shape identify separate structural
cache entries. Numerical buffers, precision, sample values, accuracy settings
and diagnostics belong to each workspace. Eager and lazy reduction children
inherit the chosen policy. The explicit six-point interface selects Symbolica
for its parent and all five-point children.

The older `iregi-masters/native-batch` feature has a broader experimental scope.
It is a separate configuration; enabling the selective feature does not enable
that older constructor. `Default` retains its previous behavior if the broader
feature is independently selected.

Policy selection is currently exposed through the Rust API. The CLI and Python
interface use their existing default constructors.

## Validation and performance scope

The [constructor capsule](../validation/native-f64-sixpoint-constructors/README.md)
records twenty historical ordinary tests and seven new constructor tests.
The [whole-call capsule](../validation/native-f64-sixpoint-whole/README.md)
compares distinct baseline/candidate cores, complete tensor and master arrays,
repeated workspaces, precision/Dual controls and original IREGI on the retained
bubble/triangle corpus. Retained six-point reference, precision and lifecycle
regressions are qualified separately from native scalar execution.

The [benchmark report](benchmark.pdf) records the exact build context and
per-family timing outcomes. A generated evaluator is not a guarantee of
performance parity for every topology, rank or batch size. All displayed ratios
put TIRust in the numerator; correctness-run durations are not kernel timings.
