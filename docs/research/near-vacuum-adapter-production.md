# Minimal production promotion after the private v3 checks

This proposal promotes an explicit scalar-master facility only. It does not
enable a new default reduction path or certify production E6 quadrature.
The broader propagation obligations remain in
`near-vacuum-box-integration.md`.

## Files and API

1. Add `crates/iregi-masters/src/near_vacuum.rs`: the qualified Symbolica
   series, exact parameter integration, Dualizer and ball programs, with the
   v3 shape preflight. Keep it private to the crate. All algebra stays in the
   existing Symbolica/Numerica APIs; the oracle remains validation tooling.
   Remove the validation prototype's unconditional stderr stage tracing;
   ordinary library use should obtain diagnostics through the API, with any
   detailed preparation trace explicitly enabled.
2. Add `crates/iregi-masters/src/certified.rs`: the accuracy-aware wrapper,
   owned N2/N4 scratch, per-lane formula diagnostics and finite rectangles.
   V3 must retain ordinary fallback for unsupported series shapes and send
   only insufficient-tail/roundoff certificates from N2 to N4.
3. Update `crates/iregi-masters/src/lib.rs` to expose the new module and typed
   accuracy/failure errors. Leave `MasterEvaluator::evaluate_batch` and its
   ordinary backend selection unchanged. This adds public error variants;
   downstream exhaustive matches must handle them.
4. Add focused public-API tests in `crates/iregi-masters/tests`, with the
   independent long-form oracle checks retained under validation. Rebuild
   against the current locked workspace dependencies before promotion;
   retained private archives do not replace that check.

The private implementation's API is concrete:

```rust,ignore
let mut evaluator = AccuracyMasterEvaluator::new(existing_master);
let accuracy = MasterAccuracyContext {
    requested_bits,
    absolute_per_component: exact_absolute_budgets,
    relative: exact_relative_budget,
};
let result = evaluator.evaluate_batch_with_accuracy(
    &input,
    rows,
    &accuracy,
    CertificatePolicy::RequireFiniteCertificate,
)?;
```

The context has exact Rational budgets, a separate requested-bit field, and
one absolute budget per native Taylor component. It is supplied on every
call; precision-specific workspaces cache no accuracy decision or numerical
result. Sample rows remain distinct from derivative components. The wrapper
uses the existing generic scalar component type and the original prepared
static/dynamic Dual shape.
The explicit budgets define the actual guarantee. The `binary(bits, count)`
constructor derives both budgets as 2^-bits; a custom context's requested-bit
field is diagnostic and does not override its supplied budgets. Core callers
must derive dimensionally normalized budgets from their documented accuracy
policy rather than infer a guarantee from that field alone. In particular,
the current public core tolerance convention differs from this stricter
validation constructor.

`AccuracyMasterBatch::lane` hides values on failure. Its diagnostics identify
the chosen formula, attempted series orders, unsupported-series reasons and
an optional complete finite rectangle per component. An absent certificate is
represented by `None`. Each rectangle encloses the function of the exact
represented master inputs; upstream routing or shift errors require separate
input enclosures. The physical Laurent poles are exactly zero only for
the proved finite near-vacuum branch. The explicit alternative policy
`AllowOrdinaryUncertified` may retain an ordinary success, with formula
`OrdinaryUncertified` and no finite certificate. It never implies that the
requested error budget was met.
The fallback policy covers lane/domain failures and unsupported series
shapes. Shared symbolic-preparation or evaluator failures still return a
whole-call error; they are not silently converted into ordinary results.

## Acceptance evidence and remaining gates

The existing private checkpoint independently qualifies the candidate at
f64/request40, DoubleFloat/request80, Float128/request100 and
Float256/request128. The adapter verifies exact equality of all 96 accepted
mixed components and certificate rectangles across those four settings.
Its independent exact contour test certifies five primal D0s and their
ComplexBall contraction at working256/request128 against a separate direct
parameter-integral oracle. The v3 contract revision must retain those
accepted outputs while qualifying unsupported-shape fallback and avoiding
unused N4 preparation. Its own source and evidence must remain separate from
the frozen earlier checkpoint.

The private v3 revision now passes those checks in
`validation/near-vacuum-adapter-v3/qualification.json`, SHA256
`ccb126aef4271af4c56296b4ecd6b4db81c571953afbf990a6049fab9f088027`.
All 96 typed accepted components are unchanged, the degree-five fallback and
N2-only outside-domain contracts pass, and the five-box/contraction oracle
comparison passes against the exact v3 library. This retained-dependency
checkpoint does not replace the current-workspace dependency check below.

The unchanged private v3 sources also passed the selected checks against the
complete retained `6b24c03` release dependency closure. The current evidence
is in `validation/near-vacuum-adapter-current/qualification.json`: all 96
accepted typed components, the dispatch contracts and the five-box oracle
comparison are unchanged. An independent audit passed 4,845 checks and
verified four unchanged 870-file inventories, including 37 native-library
manifest entries. This closes the private dependency-compatibility repeat;
the promoted public module still needs the focused live-crate checks below.

Before merging this minimal promotion, run the public master API with the
current dependency build, preserving all selected accuracy thresholds and
failure cases. Check invalid physical lanes, unsupported valid Dual shapes,
strict versus ordinary policies, changed accuracy contexts, empty/reordered
batches, zero scale derivatives and independent workspaces. Require the full
rectangle's maximum distance to satisfy the component budget; merely placing
the returned number inside a rectangle is insufficient.

Bounded N2/N4 and degree/shape/resource limits are part of this initial API.
Unsupported requests remain explicit. Higher working precision does not
waive a missing series-order or derivative-tail proof. No general master
formula acquires a certificate from this wrapper.

The first subsequent experiment should extend the exact five-pinch probe to
ball-enclosed production contour mapping and shifted inputs. Only after that
passes should E6's quadrature node/weight intervals and core output error
propagation be integrated. Do not route default `reduce_batch` through this
new facility merely because the isolated finite-master tests pass.
