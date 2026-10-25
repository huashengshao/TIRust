# iregi-rust implementation plan

## Goal and working rules

Deliver a feature-complete, independently validated Rust implementation of IREGI, including dotted propagators, dimension shifts, batching, numeric types, library, CLI and Python tools. The user accepts full validation through six propagators and rank six as the stopping envelope; retain the existing rank-seven evidence and explain the seven-point/rank-seven challenges. Do not expand beyond seven points or rank seven. Once the accepted version is tested, update the benchmark report, commit and push, mark the objective achieved and stop. Further performance optimization is deferred beyond completion; existing performance measurements remain part of the report.

### User priority and completion clarification — 25 October 2026

This clarification supersedes conflicting performance priorities in the original
milestones and acceptance text below. Preserve the mathematical validation
requirements and all failed or incomplete evidence.

- Prioritize complete, independently validated reduction through the accepted
  six-point/rank-six envelope. A
  restricted scalar high-point API or a converged array without an independent
  accuracy check does not complete the accepted tensor envelope.
- Use and improve whichever execution path is fastest. Separate interpreted
  versus native work is useful only when it helps correctness or the measured
  higher-rank bottleneck; it is not an independent development objective.
- Defer additional small-topology backend/performance cohorts. After correctness
  is established, stop after publishing the feature-complete tested version.
  Later optimization is separate work, chiefly at higher rank. Retain
  IREGI/COLLIER comparisons with TIRust in every timing-ratio numerator.
- Once the full implementation is working and validated, update the Typst
  benchmark report and machine-readable results, commit and push the validated
  result, mark the objective achieved, give the user the report link and a full
  summary, then stop as requested. Do not mark a narrower milestone complete as
  a substitute for the full implementation.

The user's subsequent clarification on the same date explicitly includes dot
support in the completion gate and removes performance optimization as a
precondition for finishing. Resource/preparation fixes needed to obtain complete
correct results remain part of implementation and validation. Do not continue
optional optimization after the feature-complete tested version has been pushed.

The next user clarification accepts a fully validated implementation through
six points and rank six, including the above features, as sufficient completion.
Seven-point/rank-seven support may remain a documented challenge, and work beyond
seven points or rank seven is out of scope. Preserve higher-rank results already
obtained; they do not replace missing feature validation within six-point/rank-six
coverage. This clarification supersedes the larger completion envelope below.

At implementation start:

- Assign this objective to the agent using `create_goal`, without a token budget.
- Write this complete plan to `/shared/external/iregi/references/PLAN.md` and maintain a committed copy in the project.
- Run all commands from `/shared/external/iregi/iregi-rust`. Keep implementation, dependency patches, generated reference builds, and benchmark artifacts there; the requested reference PLAN is the explicit exception.
- Configure repository-local author identity as `huashengshao <erdissshaw@gmail.com>`. Commit and push validated milestones to the existing `origin/main` in [huashengshao/iregi-rust](https://github.com/huashengshao/iregi-rust), without force-pushing.
- Keep research agents involved in algorithm optimization, numeric correctness, and independent code review. Milestones are checkpoints; completion requires the full validation and reporting criteria below.

## Architecture and public interfaces

**Separate reduction from input preparation.** Use workspace crates for the `iregi-rust` library, reusable graph/numerator preparation, the `iregi-rust` binary, and optional reference-validation tooling. The core library accepts numerical reduction inputs without requiring graph import, physical models, Python, or Fortran.

**Preserve IREGI semantics through idiomatic Rust APIs.**

- Provide immutable `ReductionPlan` objects, owned `Workspace<T>` scratch storage, and `reduce_batch` interfaces.
- Support momentum shifts, supplied scalar-product matrices, real/complex squared masses, renormalization scale, tensor rank, generalized positive propagator powers, and the original dimension-shift domain.
- Implement IBP, Passarino–Veltman, and hybrid PV→IBP modes. Hybrid is the default.
- Return all requested tensor moments in documented IREGI ordering. Provide a separate contraction helper for prepared numerator coefficients.
- Preserve native `[UV single, IR single, IR double, finite]` results and provide an explicit MadLoop-convention adapter.
- Expose per-sample stability, failure, and fallback diagnostics. Incomplete coefficients must never appear as successful zero results.
- Match the supplied reference’s initial validation envelope: up to seven propagators and rank seven, including generalized interfaces. Use dynamically sized structures rather than copying its fixed array limits.

**Propagate batching and precision throughout.**

- Use topology/rank recurrence graphs with dense node-indexed, structure-of-arrays workspaces. Evaluate each operation across active lanes before advancing.
- Batch scalar-master requests and exceptional-case handling; wrapping scalar reduction in an outer sample loop does not satisfy the requirement.
- Support `f64`, Symbolica `DoubleFloat`, arbitrary-precision `Float`, and Symbolica-compatible static/dynamic Dual types. Keep derivative components distinct from sample lanes.
- Reuse Symbolica’s Dualizer and oneloopmaster’s existing instruction-major Rust batch emitter. Extend oneloopmaster where coverage or interfaces are insufficient.
- Preserve precision when parsing inputs, constructing constants, normalizing results, and evaluating masters; prohibit intermediate `f64` conversions.
- Preserve inputs by default. Offer legacy on-shell snapping explicitly. Derivatives follow the selected analytic branch, with explicit diagnostics at nondifferentiable boundaries.
- Cache structural plans separately from numerical results; numerical caches must respect sample, scale, precision, and derivative context.

**Use existing algebra facilities.** Delegate polynomial manipulation, shifts, coefficient extraction, tensor contractions, simplification, and matrix algebra to Symbolica and its FeynKit/Spenso/Idenso ecosystem. Before adding any missing algebra operation, record three checks: Symbolica APIs/source, existing Symbolica usage/tests, and FeynKit/gammaloop equivalents.

## Implementation milestones

1. **Reproducible foundation and convention specification.** Establish the Rust/Nix development environment and verify GitHub connectivity before the first push. Pin dependencies and record source revisions. Start with oneloopmaster’s required Symbolica/Numerica/Graphica integration revision `9cdc2949`, testing compatibility with FeynKit before proceeding. Keep necessary dependency modifications inside the project with upstream provenance. Document metric, routing, tensor multiplicities, scale conventions, Laurent ordering, normalization, and branch prescriptions.

2. **Reference harness and scalar-master adapter.** Build isolated original-IREGI and OneLOop reference executables. Reproduce the suspected generalized-interface factorial initialization defect; apply only a narrow, documented repair to a generated reference copy if confirmed. Check repaired behavior against untouched working cases and independent identities. Implement batched oneloopmaster integration, including native UV/IR separation and scaleless integrals with cancelling poles.

3. **Bubble and triangle vertical slice.** Import FeynKit DOT graphs with explicit polynomial numerators, prepare coefficients using Symbolica, and compare complete tensor arrays and contracted results against IREGI. Include constants, odd-rank monomials, mixed components, the loop-momentum square, and sparse/dense polynomials. Exercise all numeric types and genuine batching immediately.

4. **Complete reduction coverage.** Implement both reduction methods, hybrid fallback, generalized powers/dimension shifts, routing translations, reversed propagators, complex masses, and exceptional kinematics. Derive the implementation from [Appendix C.1 and its references](https://arxiv.org/html/1405.0301v2#A3.SS1), preserving mathematical conventions while replacing global state and approximate cache matching.

5. **Physical numerator preparation and MadLoop validation.** Load pinned model JSON using FeynKit’s Rust model APIs; apply Feynman rules explicitly. Use FeynKit routing, Idenso contractions, Spenso component handling, and Symbolica polynomial extraction. Start with individual top-loop `gg→H` triangles and `gg→HH` boxes, followed by massless QCD graphs and a higher-point top-loop example. Keep UFO conversion and MadLoop execution in optional validation tooling.

6. **Optimization and final qualification.** Reuse HEPKit `tensor_reduce` symmetry-orbit methods, factor-preserving contractions, and `IntegralFamily` facilities where applicable. Benchmark optional numerator projection before enabling it; retain robust paths for singular external Gram matrices. Optimize shared recurrence nodes, invariant reuse, master deduplication, sparse contractions, and workspace allocation until correctness and performance gates pass.

Commit and push after each validated milestone. Update the benchmark report whenever coverage or measured performance changes.

## Validation and acceptance

**Independent agreement**

- Compare every Laurent coefficient and tensor component before comparing contracted numerators.
- Test normalization, odd-rank signs, coefficient ordering, multinomial weights, translated routing, reversed propagators, and scale dependence separately.
- Cover massive/massless and complex-mass cases, repeated propagators, scaleless integrals, thresholds, collinear limits, and small/zero Gram or Cayley determinants.
- Validate powers and dimension shifts using reference results plus independent mass-derivative and recurrence identities.
- For stable binary64 cases, use a mixed absolute/relative tolerance of `1e-10` on dimensionally normalized outputs. Investigate failures with higher precision; do not loosen thresholds merely to obtain agreement.
- Validate higher precision by convergence at increasing precision. Validate Dual components using analytic identities and independent high-precision finite differences away from singular boundaries, including C0 and D0.

**Batch and numeric integrity**

- Test empty/singleton batches, odd sizes, SIMD tails, reordered/chunked samples, mixed branch choices, and independent concurrent workspaces.
- Confirm an unstable lane cannot corrupt successful lanes.
- Include inputs distinguishable only beyond 53 bits and higher/mixed Dual components.
- Inspect execution traces or counters to establish batching through recurrence and master evaluation.

**Physical comparisons**

- Maintain explicit graph correspondence, routing, model parameters, helicity/color conventions, and graph factors.
- Compare independently prepared numerators at fixed loop momenta before integration.
- Export graph-isolated MadLoop coefficients and results before counterterm/process summation; disable loop grouping or resolve it explicitly.
- Exclude R2 contributions and UV counterterm diagrams. Retain bare UV/IR poles and reduction-generated R1.

**Performance**

- Gate binary64 results per topology/rank/mass benchmark family, measuring singleton latency and batch-32 throughput separately; additionally report batch scaling.
- Use identical inputs, requested outputs, cache policy, one pinned CPU core, comparable optimized builds, warm-up, repeated measurements, and uncertainty estimates.
- Measure preparation, numerator evaluation, reduction, scalar masters, and end-to-end evaluation separately. Exclude reference subprocess transport from kernel timing.
- Require parity or improvement within tightly measured timing uncertainty. Noisy measurements remain inconclusive.
- Report higher-precision and Dual performance separately. Failed and unsupported cases must remain visible and cannot improve aggregate scores through exclusion.

## Benchmark report and selected defaults

Generate `/shared/external/iregi/iregi-rust/docs/benchmark.pdf` from committed Typst source and machine-readable measurements. Match the [NinjaRS Typst reference](https://raw.githubusercontent.com/alphal00p/ninjars/main/docs/benchmark/benchmark.typ): A4 layout, serif body, navy headings, large comparison cells, green–white–red logarithmic timing ratios, and stacked component timings.

Include synthetic identical-input comparisons, physical MadLoop comparisons, precision/Dual coverage, stability failures, batch scaling, environment/compiler details, dependency revisions, and reproduction commands. Rebuild after each relevant milestone and identify stale measurements explicitly.

Confirmed defaults: Rust semantic API compatibility without a legacy ABI; Rust implementation with Symbolica’s GMP/MPFR dependencies allowed; external validation tools allowed; four-dimensional numerator polynomials with dimensionally regulated denominators; input preservation by default; top-loop physical examples first; performance assessed per benchmark family.

## COLLIER reference comparison (user-requested extension)

Use the pinned COLLIER 1.2.10 source supplied by the user as an additional optional reference. Study its published tensor-reduction methods for independently derived Symbolica implementations, retaining source provenance. Keep its build and execution isolated under the project, without changing the Rust-only core or copying its algebra routines.

Compare original IREGI and COLLIER on their shared undotted topology/rank domain, starting with massive bubbles and triangles at ranks zero and one, then extending to boxes and higher-point low-rank cases and representative higher supported ranks. Record each backend's actual limits; preserve failed and unsupported cases. The first eight bubble/triangle rows are a vertical slice, not qualification of the full shared domain.

Before timing, establish routing, normalization, scale, Cartesian ordering, UV/IR pole separation and complete tensor/Laurent agreement on identical inputs. Match requested outputs, optimized compiler settings and within-call cache policy; reset cross-event numerical caches. Measure singleton and batch-32 kernels separately with pinned-core paired blocks and uncertainty estimates, excluding subprocess transport. Report this comparison separately from Rust-versus-IREGI performance and update the Typst report only from qualified measurements.

At every intermediate milestone, update the Typst source and benchmark PDF, commit and push the validated work, and give the user a GitHub link to that pushed version of `docs/benchmark.pdf`. Include the measured IREGI–COLLIER comparison in the PDF, with unmeasured shared-domain families identified explicitly.

## TIRust naming, comparison direction and Python tools (user extension)

Use **TIRust** as the implementation name in benchmarks and user-facing Python tools. Keep the existing repository and Rust package names compatible. Every displayed implementation/reference timing ratio has TIRust in the numerator: `TIRust / IREGI` or `TIRust / COLLIER`; ratios below one mean TIRust is faster. Name internal comparisons explicitly, for example `TIRust (candidate) / TIRust (baseline)`. Preserve historical raw evidence unchanged. Reference-only IREGI/COLLIER studies retain their absolute timings and qualification outcomes, without presenting a reference/reference ratio as a TIRust measurement. Obtain new comparisons from identical inputs in the same controlled measurement cohort; never combine ratios or medians from different sessions.

Add a friendly, documented PyO3 interface and elegant reactive marimo notebooks as part of the active implementation goal. Follow the pinned [Symbolica Community extension guide](https://github.com/symbolica-dev/symbolica-community/blob/cad32c8f589c8a29531414f94b3fbe58020e0308/docs/writing-extensions.md) and study the supplied [HEPKit numerator notebook](https://hepkit.org/gallery/notebooks/four_loop_numerator.py) for presentation style.

Keep Python bindings separate from the Rust reduction and numerator-preparation libraries. Provide linkable `rlib` registration and an optional standalone native host, using actual upstream Symbolica, Numerica, FeynKit and Spenso Python types in one native library and one Symbolica kernel. Call Rust APIs directly; do not create lookalike shared types, import canonical Python modules to find constructors, or serialize objects merely to transfer them between components. Preserve precision, complex values, sample lanes and complete Dual shapes. Expose immutable plans, owned workspaces, batched reduction, generalized positive propagator powers and dimension shifts, prepared numerator/contraction tools, complete Laurent metadata and per-lane diagnostics with explicit failures.

Align PyO3 and shared dependencies, check the resolved dependency graph, generate accurate typed stubs, and package with maturin. Test the built standalone wheel in a fresh environment, shared Python class identities, invalid inputs, precision/Dual transport and operation without canonical Community imports. Keep compatibility with the documented Community registration model; publishing wheels and opening an upstream Community integration PR are separate actions, not required to expose a usable local interface.

Provide notebooks for DOT/numerator preparation and bubble/triangle reduction, dotted propagators and independent derivative identities, precision/Dual and stability diagnostics, and honest exploration of committed benchmark evidence. Use clear narrative, mathematical displays, reactive controls and readable results. Execute notebooks against the real built interface and validate headless execution/export. Notebook examples must respect the qualified coverage envelope and retain failed, unsupported, stale and noisy benchmark cases.
