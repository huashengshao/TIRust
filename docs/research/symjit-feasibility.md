# SymJIT feasibility for batched reduction

Source audit, 2026-10-23, project checkpoint `80949dbc4711947971a702643f75d639ef217e82`.
No build, numerical experiment, timing experiment, dependency update, or production
change was performed for this note. Findings about defects below are source
findings, not reproduced failures.

## Decision

Defer SymJIT as a replacement for the production batch evaluator. The existing
APIs compile Rust-generated machine code and can evaluate four samples with SIMD,
but their scheduling is not the required instruction-major evaluation across the
active batch. Scalar batches run a complete scalar kernel for each sample; SIMD
batches run a complete kernel for each packet and rerun mixed-branch packets and
tails through scalar kernels. The current explicit SIMD wrapper also has an
unresolved multiple-output unpacking defect. These are architectural and
correctness obstacles before performance qualification.

Continue measuring the existing Symbolica batch interpreter and oneloopmaster's
instruction-major Rust emitter. The latter already preserves generic arithmetic,
active-lane lists, lazy control flow, and nested helper batching. A future JIT
extension would need the same scheduling contract; naming a scalar loop “batch”
or processing four samples together does not establish that contract.

## Sources and scheduling

`Cargo.lock:2532` pins **SymJIT 2.27.3**, registry checksum
`9d5297d2ef25e396f984747fd0e480aaabb08e40e424f9cd07811efc6b20c140`.
Its source was inspected in
`.local/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/symjit-2.27.3/`;
paths beginning `rust/` below are relative to that package. Symbolica's Cargo
requirement is `2.26`, which does not mean the resolved version is 2.26.

| Route | Actual scheduling | Constraint |
| --- | --- | --- |
| `JITCompiledEvaluator<Complex<f64>>::batch_evaluate` | Symbolica `evaluate/backend.rs:1304` calls SymJIT `Applet::evaluate_matrix`; `rust/applet.rs:124` loops over rows and calls the full scalar kernel | Outer sample loop; not a replacement for the required batch path |
| SymJIT automatic SIMD with scalar buffers | `rust/applet.rs:275` loops over SIMD packets; nonzero kernel status reruns each packet lane as a scalar; remaining tails are scalar | SIMD within packets, not instruction-major across the active batch |
| Explicit `Complex<wide::f64x4>` Symbolica wrapper | `evaluate/backend.rs:1566` invokes the complete evaluator for each packed group | Same scheduling limitation, plus wrapper defects below |
| SymJIT experimental lockstep | `rust/lockstep.rs:1` and `:68` use one stackful coroutine per point, switching between configured expression-node segments | Segment-major scalar execution; no dense node-indexed SoA or operation-wide active-lane guarantee |
| Existing oneloopmaster Rust batch emitter | `src/rust_codegen/batch.rs:1` documents shared helper lanes; `:176` chooses an active-lane loop inside the exported-instruction loop at `:181` | Matches the intended scheduling; generic domains and lazy control retained |

SymJIT 2.27.3 contains newer fixes than the warning in oneloopmaster
`src/evaluators.rs:66`: `rust/defuns.rs:311` includes a scalar-complex callback
SIMD/tail test, and `rust/amd/vector.rs:210` propagates nested kernel fallback.
Those earlier 2.26 warnings therefore cannot alone establish a defect in 2.27.3.
Nevertheless, oneloopmaster's current supported JIT settings explicitly disable
SIMD, threads, fastmath, and SIMD branch evaluation (`src/evaluators.rs:71`).

## Unresolved wrapper and branch issues

1. **Multiple outputs are unpacked incorrectly.** In Symbolica
   `src/evaluate/backend.rs:1573–1576`, each chunk of `4 * n_out` scalar complex
   outputs is paired with only one packed output vector, and its scalar index
   `j` indexes the four-element lane array. Two outputs and three rows attempt
   `as_array()[4]`; two outputs and one row instead copy two lanes of the first
   output, rather than one lane of each output. The real SIMD wrapper has the
   same missing transpose at `:1193`. This directly affects Laurent arrays and
   tensor arrays. Correct unpacking must retain both the output and lane axes.
2. **Explicit SIMD evaluation discards fallback status.** Symbolica's wrapper
   calls `Applet::evaluate`, whose `rust/applet.rs:72–76` invokes the SIMD function
   without inspecting its returned status. With `simd_branch=false`, divergent
   branches return early (`rust/amd/vector.rs:295–311`), and only the matrix
   dispatcher performs scalar fallback. The explicit packed wrapper bypasses
   that dispatcher. An absent SIMD kernel also produces no explicit error there.
3. **Turning on `simd_branch` changes lazy evaluation.** The compiler suppresses
   ordinary `goto` for this mode (`rust/compiler.rs:1144`) and joins branch values.
   A mixed packet can execute both arms, including expressions or callbacks
   invalid in inactive lanes. This is not the production lazy-lane guarantee;
   numerical masking alone would not prove that callbacks remain unevaluated.
4. **Zero and tail shapes need explicit handling.** The Symbolica SIMD wrappers
   divide by `batch_size` and use `chunks(n_params)` / `chunks_mut(n_out)` without
   an empty-batch or zero-parameter/output shortcut. Tail lanes are padded with
   zero (`evaluate/backend.rs:1517–1552`), which can select different branches or
   singular expressions. Oneloopmaster's scalar JIT adapter correctly returns
   before calling its backend for an empty batch (`src/evaluators.rs:184`).

## Numeric domains, branch prescriptions, and ownership

Symbolica's JIT entry point converts exact constants to binary64 at
`src/evaluate/backend.rs:494–507`. Supported implementations are `f64`,
`Complex<f64>`, and their `wide::f64x4` forms. Restrict any experiment to a concrete
binary64 specialization; never route `DoubleFloat`, MPFR `Float`, or generic Dual
values through these conversions. A future flattened binary64 Dual experiment
would still need separate derivative/sample axes and independent qualification.

The JIT does not itself encode IREGI's Feynman sheet. Retain oneloopmaster's exact
sheet expressions and registered primitives. Its JIT lowering replaces square
roots with Symbolica's stable complex root callback and branch predicates with
exact nonzero callbacks (`src/jit_primitives.rs:1–49`). This remains relevant:
SymJIT's packed-complex root computes an unscaled squared norm
(`rust/amd/complex.rs:324–336`), which can overflow or underflow for finite inputs
with representable roots. Complex log/power call `num_complex` implementations
(`rust/code.rs:655–668`); dilogarithms and other registered functions require the
existing Symbolica/OneLoop definitions. SIMD callback availability, signed zeros,
cut lips, subnormals, and inactive singular arms require tests, not assumptions.

SymJIT's compiler, assembler, and runtime are Rust; machine code is written to
executable memory (`rust/machine.rs:11–73`). No generated C++ compiler is needed.
The optional `libm` feature has C ABI complex-math bindings, while the default
path uses Rust implementations and normal system/runtime dependencies. This
does not add an independent algebra engine: translation consumes Symbolica IR.

`JITCompiledEvaluator` derives `Clone` and owns separate packing buffers
(`evaluate/backend.rs:851–861`). SymJIT `MachineCode` shares executable memory
through `Arc<Memory>` (`rust/machine.rs:11–25`), and `Config` shares callback tables
through `Arc<Defuns>` (`rust/config.rs:63–68`). This supports a structural-code
cache plus independent workspace buffers in principle. Use these owning APIs;
do not directly clone raw callback owners. Serialization is revision dependent
and recompilation settings must be retained. Oneloopmaster instead caches
Symbolica source IR and recompiles on load (`src/evaluators.rs:34–46`, `:203`).
Concurrent reuse still needs its own tests; none were run in this audit.

The FeynKit/gammaloop reuse check found `SymjitComplexEvaluatorGL` using a scalar
`evaluate` call in the reference `crates/gammalooprs/src/integrands/process/
evaluators.rs:255–270`. Its JIT construction at `:2766` disables compaction and
CSE and caps optimization at O2 because of its own branch/cache concerns. This
is useful prior art, but supplies no compliant instruction-major SIMD adapter.

## Measured optimization targets

The existing controlled `80949db` run's separate instrumented profile supports
different targets by family. Values are median ns/batch, not new measurements or
unbiased additive decompositions; the reduction stage includes stability checks.

| Family | Batch | Arguments | Masters | Reduction/guards | Profile total |
| --- | ---: | ---: | ---: | ---: | ---: |
| N2 rank 0 | 1 | 293.2 | 1068.1 | 581.5 | 2395.9 |
| N2 rank 0 | 32 | 7039.0 | 18261.6 | 15661.5 | 45461.6 |
| N2 rank 1 | 1 | 655.3 | 1567.7 | 4534.0 | 7348.4 |
| N2 rank 1 | 32 | 10166.2 | 26443.7 | 104539.2 | 147988.8 |
| N3 rank 0 | 1 | 912.6 | 27098.3 | 1140.5 | 30016.5 |
| N3 rank 0 | 32 | 25567.9 | 524688.4 | 27294.4 | 588479.8 |
| N3 rank 1 | 1 | 993.9 | 33630.5 | 7797.5 | 43763.7 |
| N3 rank 1 | 32 | 18007.9 | 393731.9 | 142300.9 | 562506.9 |

Source: `target/benchmark/baseline-80949db-licensed/results.json`, SHA256
`f31fe4034ba5fd5360d5327673c40d8bb604d4c72b8c9b5983958e303ffe179e`.
Masters account for approximately 70–90% of the N3 profile, while N2 rank 1
reduction/guards account for 62–71%. Prefer qualified native master improvements
for the former and evaluator/guard work for the latter; these shares do not
predict a speedup or waive the complete end-to-end timing gate.

## Minimal future isolated test, if this route is revisited

First reproduce and repair the wrapper independently of IREGI: two expressions
`[x + y, x * y]`, two complex parameters, and batch sizes `0, 1, 2, 3, 4, 5, 7, 32`,
with distinct real/imaginary values in every lane. Add a parameter-free constant
expression, reordered/chunked samples, and sentinel output buffers. Compare the
entire row-major array with Symbolica's existing batch evaluator. Test explicit
packed evaluation separately from scalar-buffer automatic SIMD.

Then test `IF(c, 1/x, 7)` and a nested shared-body version with mixed `c` values,
zero `x` only on inactive arms, and a counted callback which must not run on those
arms. Include imaginary-only conditions and signed-zero cut approaches. Record
SIMD availability and fallback counts; a correct scalar rerun is a correctness
result but still fails the requested batching architecture. Only after a real
instruction-major backend exists should A0/B0/C0/D0's six native outputs, master
deduplication, complete tensor arrays, concurrency, and controlled performance
be qualified. No such follow-up implementation is justified as a current
drop-in optimization by this source audit.
