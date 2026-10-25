# TIRust / iregi-rust: exhaustive session handoff

Prepared on **24 October 2026**, at the user's explicit request to wind down,
consolidate on `main`, update the report, preserve context and stop. This is a
restart document, not a claim that the original implementation goal is complete.

**Resumed on 25 October 2026 at the user's request.** The pause statements below
describe this handoff's historical checkpoint. Subsequent work is recorded in
[the resumed checkpoint](docs/checkpoints/resume-20261025.md); the full objective
is active and unchanged.

**Latest published checkpoint: `a69b754`, 25 October 2026.** The report has
39 pages. Complete retained massive PV arrays through rank seven pass at N1–N6;
the N6 batch/contraction lifecycle also passes. Retained massless N2–N4 arrays
pass 7,920 independent-reference comparisons with UV/IR poles kept separate.
The shared tensor facade and Symbolica's scoped branch-fact compiler change are
committed. See [the array checkpoint](docs/checkpoints/rank7-arrays-20261025.md)
and [the massless/dispatch checkpoint](docs/checkpoints/rank7-massless-dispatch-20261025.md).
The next checkpoint, [shared CLI and IBP](docs/checkpoints/rank7-cli-ibp-20261025.md),
updates the report to 41 pages. It adds passing shared CLI integration through
N6 rank three / N7 scalar, retained N2 rank-seven IBP, generalized original-IREGI
quad anchors and a preserved fresh N7 rank-seven allocation failure. Public N7
remains scalar-only; private ranks one/two pass. Fine instrumentation has located
distance-polynomial conversion over thousands of irrelevant imported parameters
as a concrete preparation bottleneck; its narrow Symbolica fix is being checked.
Broader methods, generalized tensors and numeric/Dual domains still need their
own qualification. Current work
continues in the worktree; inspect its files and receipts before restarting a
job. This document's later pause, page-count and rank-limit statements describe
the older wind-down and do not override this active checkpoint.

The user's latest priority is full independent validation through rank seven,
followed by optimization chiefly at higher rank. Low-rank COLLIER speed and
interpreter/native selection are not completion gates. The full objective is
unfinished and active; do not pause or mark it complete at a scoped milestone.

The first resumed implementation milestone adds the optional generated f64
A0/B0/C0 master policy. Its current locked-Cargo builds, selected tests and
complete triangle comparisons pass; the historical broad-backend metadata
timeout remains explicit. Read the resumed checkpoint before applying any
historical statements below about pending constructors or private composition.

The commit containing this file consolidates the final evidence and unfinished
source archives. The last preceding implementation commit is
`007ca9f7b1f3cd2005e0d4df1fcfb24f55aaecc7` (`Add restricted six-point reduction
and validate mass Dual identities`). The final consolidation commit necessarily
cannot contain its own hash; obtain it with `git log -1`. Check `origin/main`
before starting a later session.

## 1. Read this first

1. **The full goal remains unfinished and is active.** The user resumed after
   the historical wind-down. Do not mark it complete because a milestone,
   narrow test corpus or consolidation is complete.
2. Everything is in `/shared/external/iregi/iregi-rust`. Execute commands from
   that directory. The only intended write outside it is the requested
   `/shared/external/iregi/references/PLAN.md`, which matches committed `PLAN.md`.
3. Source `scripts/env.sh`. The private local license environment is
   `.local/symbolica-license.env`; source it silently for licensed execution.
   **The license value is deliberately absent from this document and Git.**
4. `main` contains the validated Rust/CLI/Python implementation and restricted
   public six-point API. The new seven-point and scheduler results are **private
   candidate evidence**, not default reducer changes. The optional native-master
   composition is also private and has outstanding regression gates.
5. Read [PLAN.md](PLAN.md), [README.md](README.md),
   [docs/benchmark.pdf](docs/benchmark.pdf), then the relevant qualification
   capsule. A cached binary's name or an old green receipt does not identify the
   current source or dependency closure.
6. The final report has **31 pages**, is generated from committed Typst and
   machine-readable results, and retains historical timings unchanged. No new
   timing experiment was performed during wind-down.
7. No implementation experiment was started during wind-down. The seven native
   composition constructor tests were authorized earlier but **never started**;
   their attempt directory does not exist. They remain a next-session task.
8. Preserve old failures, unstable reference results and inconclusive timings.
   Never replace a failed lane by zero-valued success or lower a tolerance to
   obtain agreement.

## 2. Objective, user preferences and completion requirements

The user requested a Rust port of original IREGI, with mathematical behavior
derived from Appendix C.1 of arXiv:1405.0301 and its references. Original sources
are under `../references/madgraph/vendor/IREGI`. The entire approved plan is in
`PLAN.md`; the following is a practical restatement, not a replacement scope.

- Pure Rust numerical library and a standalone CLI. Rust semantic compatibility
  with the original interface is required; a legacy ABI is not required.
- Immutable structural `ReductionPlan`s, owned generic `Workspace<T>` scratch,
  genuinely batched recurrence and scalar-master evaluation, documented complete
  tensor arrays and a separate numerator contraction interface.
- IBP, Passarino–Veltman and default Hybrid PV→IBP behavior; routing translations,
  reversed propagators, supplied invariant matrices, real/complex squared
  masses, scale, positive propagator powers and the original dimension-shift
  domain. Full initial acceptance extends to **seven propagators and rank
  seven**, including generalized interfaces. The present qualified envelope is
  narrower and must not silently become the final requirement.
- `f64`, Symbolica `DoubleFloat`, arbitrary-precision `Float`, and static/dynamic
  compatible Duals, including higher and mixed components. Sample lanes are not
  derivative components. No intermediate binary64 conversion is allowed in a
  higher-precision path.
- All algebra—polynomials, coefficient extraction, shifts, tensor contractions,
  simplification and matrix operations—must use Symbolica or existing
  FeynKit/Spenso/Idenso facilities. Before implementing a missing algebra
  operation, record three searches: Symbolica APIs/source, Symbolica usage/tests,
  and FeynKit/gammaloop equivalents. No home-grown symbolic engine or solver.
- Use oneloopmaster for scalar masters and extend it where required. Keep local
  dependency patches under the project with upstream provenance.
- Numerator preparation belongs in separate reusable modules and the CLI, not
  inside the reduction core. Import FeynKit DOT graphs, start with explicit
  bubble/triangle polynomials, then derive physical numerators through model APIs,
  Feynman rules, Idenso/Spenso and Symbolica.
- Compare every tensor component and Laurent slot before contractions. Stable
  binary64 acceptance uses mixed absolute/relative, dimensionally normalized
  `1e-10`; investigate failures at higher precision rather than loosening it.
- Higher precision requires convergence. Duals need analytic identities and
  independent high-precision finite differences away from boundaries, including
  C0 and D0. Threshold/collinear/Gram/Cayley failures must remain visible.
- Physical comparisons must isolate corresponding MadLoop graphs, match routing,
  helicity/color/model/factors and independently compare unintegrated numerators.
  Exclude R2 and UV counterterm diagrams; retain bare poles and reduction R1.
- Performance acceptance is **per family**, singleton latency and batch-32
  throughput separately, plus scaling. Identical inputs/outputs/cache policy,
  optimized comparable builds, one pinned core, warm-up, repeated measurements
  and uncertainty are required. Noisy or short blocks are inconclusive.
- Study HEPKit `tensor_reduce`/`IntegralFamily` ideas and COLLIER algorithms.
  COLLIER 1.2.10 is an additional isolated reference and benchmark on its shared
  undotted domain. Do not assume it accepts TIRust's dotted/generalized interface.
- The name **TIRust** is used in benchmarks/Python tools. Every displayed timing
  ratio involving this implementation has TIRust in the numerator:
  `TIRust / IREGI`, `TIRust / COLLIER`, `TIRust / MadLoop`, or explicitly
  `TIRust(candidate) / TIRust(baseline)`. Never mix sessions to manufacture ratios.
- Provide a user-friendly PyO3 interface following the pinned Symbolica Community
  extension guide and elegant marimo notebooks inspired by HEPKit. Upstream
  shared classes must live in one native host and one Symbolica kernel; no
  lookalike wrappers or serialization just to transfer shared objects.
- Keep a Typst report at `docs/benchmark.pdf`, source and machine-readable
  measurements committed. Follow the NinjaRS example: A4, serif text, navy
  headings, large colored cells, logarithmic green–white–red ratios and stacked
  timing components. Update and link the PDF on each pushed milestone.
- Commit and push validated intermediate milestones to the existing
  `huashengshao/iregi-rust` `origin/main`, no force pushes. Local author identity
  is `huashengshao <erdissshaw@gmail.com>`.
- Work autonomously on authorized tasks. Do not repeatedly ask for permission
  for routine reversible work. A paused task still requires a later user resume.

The active goal was created without a token budget. A later session should
inspect the existing goal and resume it as appropriate; do not create a competing
unfinished goal or narrow the objective. This session pauses it only because the
user expressly requested stopping after consolidation.

## 3. Repository and environment

### 3.1 Layout

| Path | Role |
| --- | --- |
| `crates/iregi-rust` | Generic numerical reduction plans/workspaces, tensor basis, guards, fallback, contraction, explicit certified/high-point APIs |
| `crates/iregi-masters` | Batched oneloopmaster adapter, pole separation, exact plans, derivative domains, shifted and near-vacuum facilities |
| `crates/iregi-prepare` | FeynKit model/DOT import, Feynman rules, Idenso/Spenso numerator preparation, Symbolica polynomial coefficients |
| `crates/iregi-cli` | Standalone binary; exact input parsing, coefficient evaluation and physical/reference tooling |
| `crates/iregi-python` | PyO3 registration `rlib` and optional standalone native host |
| `python` | Python facade, typing and packaging documentation |
| `examples/graphs`, `examples/inputs` | Synthetic and physical graph/input examples |
| `examples/notebooks` | Three marimo notebooks and display/helper tests |
| `dependencies/symbolica` | Coherent pinned Symbolica/Numerica/Graphica source with documented patches |
| `dependencies/oneloopmaster` | Local scalar-master source and extensions |
| `dependencies/feynkit-integrals` | Extracted pinned FeynKit affine-partial-fraction facilities; planner integration remains pending |
| `dependencies/linnest` | Pinned renderer owner with narrow typed-CBOR fix |
| `validation` | Committed raw data, source snapshots, protocols, tests, receipts and independent reviews |
| `validation/winddown` | New private source/evidence snapshots, explicit binary omissions and portable verifier |
| `docs/research` | Algorithm, guard, reference, optimization and API research notes |
| `docs/checkpoints` | Human-readable milestone descriptions |
| `docs/data/results.json` | Historical and current report records; each preserves its source context |
| `docs/data/report-validation.json` | Report artifact hashes, ledger, preservation and rendering evidence |
| `target` | Ignored generated builds, private candidates, reference executables and operational receipts |
| `.local` | Ignored tool environments, local Cargo/cache/browser data, license environment and installed wheels |

The core does not depend on graphs, Python, MadGraph, Fortran or physical model
loading. External tools are validation-only. Preserve that separation.

### 3.2 Toolchain and feature graph

Run from the project root:

```sh
source scripts/env.sh
git status --short
git log -3 --oneline
```

The qualified local native Cargo context used Rust/Cargo **1.99**, rustc
`b940084d7` / LLVM 21.1.8 and Cargo `5f94df478`. Nix pins the development
environment. `scripts/env.sh` also discovers a compatible installed toolchain.
Do not assume that the shell default or another user's shared build uses these
versions. Current workspace package `rust-version` is 1.99, edition 2024.

Normal development dependencies use `opt-level=2`; project crates use the
ordinary dev profile unless explicitly overridden. Release uses thin LTO and
one codegen unit; bench inherits release with debug information. Private direct
qualification builds often use opt0 core plus an older frozen opt2 dependency
closure. Those are correctness contexts, **not current optimized performance**.

The most recent non-Python Cargo build selected Symbolica features
`bincode,float-mpfr,integer-gmp,native_code_generation,serde,tracing_max_level_info`;
OneLoop selected `default,generated-evaluators,prebuilt,rust-codegen`. The Python
host has a substantially larger graph. Keep routine Rust builds separate from
the optional Python/rendering closure. Never substitute a Python-enabled cached
Symbolica archive for a native-only artifact on the strength of its filename.

Pinned major sources:

- Symbolica integration upstream revision
  `9cdc2949cdb153478c1eca0d40690dc0abfc616e`, matching the user's requested
  oneloopmaster integration start. See
  `dependencies/symbolica/IREGI_PROVENANCE.json` and
  `dependencies/symbolica-patches/provenance.json`.
- FeynKit/Spenso/Idenso/gammaloop revision
  `865afdc43fb4e6b950439d34b4ec2c38352873e7` in workspace dependencies.
- Symbolica Typst plugin revision
  `57b7455b37c30f4f87353c0073e49f5225e937ab`.
- Current root lock SHA256:
  `33865b33512eb2a5f7f9f7d54075baee1c8c7e90167ae8f5b87019966c381d80`.
  Recheck it after any dependency changes rather than relying on this document.
- `PLAN.md` and `../references/PLAN.md` both had SHA256
  `4bc599ab9157105f9ac532a7fcd9b74053a365f222f986a0651e5479500df60a`.

The existing production Symbolica patches are:

1. Journal/scope conditional cache bookkeeping.
2. Identify the vendored upstream version rather than inheriting outer Git text.
3. Limit Git watches to the version fallback in the build script.
4. Check polynomial exponent conversion.

The private uniform-branch scheduler candidate is **not** another applied
production patch. See its separate qualification below.

### 3.3 Credentials, author and push

The user supplied a Symbolica license and said it was not sensitive, but it is
still kept out of Git/log output. The local environment file is ignored. Use:

```sh
source .local/symbolica-license.env
```

Do not print that environment file or `env` wholesale. A fresh machine needs its
own authorized license setup; the repository intentionally cannot reconstruct
the literal key.

Git author is repository-local. The push command that worked was:

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main
```

Do not change global identity, force-push, invent a different remote or create a
PR merely to preserve this session. The user asked for direct milestone pushes.

## 4. Mathematical and API conventions

See `docs/conventions.typ`, README and CLI docs for the full specification.
Important traps:

- Metric is `(+---)`; Cartesian components are `(time,x,y,z)`.
- Denominators use canonical `q+p_i` routing and **squared masses**. A negative
  original loop orientation must be translated to that canonical shift.
- Input scale is **mu**, not mu-squared.
- Core/Python `dimension_shift=h` means `D=4+2h-2epsilon`. CLI JSON uses the full
  even offset and translates it. Do not confuse the two interfaces.
- Cumulative Cartesian tensor moments are ordered by increasing rank, then the
  documented decreasing component exponents. Use `basis()`; do not independently
  guess ordering or multinomial weights.
- Native result slots are `[UV single, IR single, IR double, finite]`.
  oneloopmaster's internal six-output observation order is different:
  `[finite,total_single,ir_double,uv_single,ir_single,derivative_discriminant]`.
- `Convention` adapters and `TensorWeights` metadata are explicit. MadLoop's loop
  variable/routing changes odd-rank signs and numerator coefficients. Compare
  complete metadata before contractions.
- Positive propagator powers and dimension shifts are actual mathematical
  inputs, not just CLI labels. Generalized PV follows the original termwise
  admissibility conditions; other terms use IBP. Strongest retained numerical
  mapping is N≤4.
- Supplied upper-triangle scalar products include the diagonal and remain
  authoritative. Ordinary supplied-invariant requests with N≥6 are rejected
  because four-dimensional Gram identities need full-Taylor certification.
- Inputs are preserved by default. Legacy on-shell snapping is an explicit
  policy, with its original unusual massless/threshold rules and separate
  qualification. Do not snap automatically to improve agreement.
- Scalar-less integrals may have cancelling UV/IR poles. Preserve their separate
  native slots, even if a total pole or finite integral vanishes.
- A failed/unsupported lane exposes a diagnostic and no successful partial array.
  Call `BatchResult::lane`; internal zero-initialized buffers are not evidence.
- A Float prototype controls constant precision. Supplied represented values
  keep their own precision; promoting them cannot restore digits already lost.
  Exact rational/decimal parsing must precede arithmetic at the desired precision.
- Dual shapes are ordered Taylor layouts. `None`, a primal-only Dual shape and
  different equal-size ordered shapes are distinct cache contexts. Higher/mixed
  components are not extra sample lanes.
- `set_accuracy_precision(53)` controls the stability target, not a proof of 53
  achieved bits. Working Float128/256 and convergence do not certify every real
  or imaginary component to that many bits.

Production `Workspace` executes instruction-major Symbolica evaluation and
batches scalar master families. The six-point path packs subsets across samples
before child evaluation; it is not an outer loop around scalar reductions.

Caches store structural preparation, not approximate numerical event matches.
Each workspace owns scratch. Do not reintroduce original IREGI's global state or
approximate cache promotion. See `docs/research/iregi-failed-cache-promotion.md`.

## 5. What is implemented on main

### 5.1 Reduction and numerical safeguards

Main has ordinary IBP/PV/Hybrid reduction, full tensor arrays, batched numeric
execution, separate contraction, exact parsing, precision rescue and explicit
lane statuses. It includes ordinary and generalized scalar interfaces, supplied
invariant handling, complex masses, native/OneLoop/MadLoop convention adapters,
and numerous exceptional/analytic paths. Dynamic structures replace copied
fixed Fortran array limits.

There are independent Float/DoubleFloat/Dual tests, including beyond-53-bit
inputs, mixed precision, empty/odd/reused batches, errors and cache isolation.
This is substantial implementation, but not proof of every N≤7/rank≤7/generalized
combination. Read the report's open ledger rather than inferring coverage from
generic signatures.

Numerical guard work includes contribution precision tracking, scalar-master
argument sensitivity, exact represented-input invariant checks before analytic
branch selection, conditioning estimates, precision rescue and route-specific
diagnostics. Roundoff estimates and retained sensitivity operands are not
rigorous interval/error enclosures. Avoid relabelling them as certificates of
complete numerical accuracy.

Public specialized interfaces include:

- `ScalarReductionPlan` / `ScalarWorkspace`, accepting packed invariants only.
- `CertifiedScalarReductionPlan` / workspace for explicit constrained high-point
  scalar domains. Retained six-point scalar precision checks exist, but do not
  qualify all six/seven-point tensors.
- `CertifiedPlanarReductionPlan` / `CertifiedPlanarWorkspace<T>` for explicit
  planar five-point PV tensors through rank seven, with routing/full-Taylor
  checks, regular chart selection and grouped lanes. Ordinary dispatch does not
  invoke this automatically.
- Finite-D0/shifted/near-vacuum facilities described in
  `docs/research/certified-masters-api.md`, `shifted-pentagon.md`, and associated
  validation capsules. Read exact domains before using these as general rescue.
- The new restricted public six-point API described next.

### 5.2 Restricted public six-point API

`SixPointReductionPlan` / `SixPointWorkspace<T>` are now on main. Read
`docs/six-point.md`, `crates/iregi-rust/src/six_point.rs`, `thin_n6.rs`,
`execution/planar_fallback.rs`, and the envelope/guard code in `types.rs`.

The public constructor accepts only:

- N=6, cumulative rank 0–3;
- six unit powers and dimension half-shift zero;
- `Algorithm::PassarinoVeltman`;
- Cartesian shifts / `InvariantInput::FromShifts`;
- `InputPolicy::Preserve`;
- native IREGI normalization and raw tensor component weights;
- one ordinary component. **There is no public Dual constructor.**

Each row has 31 complex arguments: 24 absolute shift components, six squared
masses and mu. It packs all six N5 subsets across active parent lanes into one
child batch. Successful children must provide actual-route primal sensitivity
context; the Symbolica parent combines their values and those operands.
Child-only planar retries and complete-Taylor zero-cofactor certificates cannot
drop a child solely because its primal coefficient is zero. Unavailable context,
singular charts or unsupported quadrature transport fail explicitly.

The public API also returns a normal contraction plan for prepared polynomial
coefficients. Output remains cumulative ordinary IREGI ordering; there is no
double normalization. Stats distinguish child/parent batch work and certificate
counters, and invalid/empty calls reset them.

Two distinct qualifications must remain distinct:

**Isolated integration:**
`validation/highpoint-six-point-api/qualification.json`, SHA256
`97ff1fc6866115d64d2b21c46e20ad25449e39392b06c7913e076e5b19abe428`.
34 focused tests; 12 ordinary paired cases with exact complete records (42
successful lanes, 18 matching rejection controls, 2,200 complex Taylor slots);
140 rank-three Laurent slots at each of Float128/256; rank-two lifecycle with
1,020 tensor slots and 68 contraction slots, 17 successful lanes, seven rejection
controls and two empty calls. Source patch SHA is
`b3e7b8aeeaa56a9d6fe5fd54b53d01b0adcd16fc5c47737ee50574c3ea6a4013`.
This immutable capsule retains `production_build_qualified=false` because that
was its actual source/build context. Do not edit it retrospectively.

**Actual current locked Cargo integration:**
`validation/highpoint-six-point-current-cargo/qualification.json`, SHA256
`d1174ee600964c18d9c9c1bfafe2ae2a8db06adc69c939447c4051cfa8725edb`.
Manifest `74c1c5dea61dfa33e90a546501ac6ed07e329663e196b47e94c6ee737d128010`.
Actual library test and example builds pass, then actual binary listing discovers
219 tests and selects the exact same 34. All 34 pass. Both complete rank-three
140-slot arrays pass: 280 original comparisons and 140 precision comparisons.
Maximum normalized original error is `2.4253553224e-12 < 1e-10`; maximum
128/256 difference is `1.4738132606e-34 < 2^-80`. All 105 pole slots are exactly
zero at each precision. Independent audit
`aff3b5ff5cafa3734503d09160b47b64cdf592a0c12f96dd079804e930bc563e`
rehashes 718 bindings twice and reconstructs the arithmetic independently.

The actual current test executable hash is
`750f19ae0437b79939eb5d8dfb3e5c601966cf795f437730e952fd58808f8869`;
example hash `563613ff31c79d7d0d7c2cf412d85c192a1d45803f7fa0c11e63a0cfca3ba1b9`.
The 674 runtime input bindings include recorded Cargo artifacts, source and exact
linked binaries. They are **not a complete native build-input closure**: GMP,
MPFR and MPC archives have post-run supplemental hashes only. This limit is
explicit in the capsule and report. No rerun of the isolated 12-case ordinary
corpus or rank-two lifecycle was falsely counted as another current-Cargo run.

Application matched 52 private candidate files. Later source differences were
module documentation only (`six_point.rs` `ea256a4e...` to `8e4341a8...`) and
`docs/six-point.md`; the API body remained identical. Current source reconstructs
from `4fe1a1e`, qualified V2 archive/patch and the small documentation overlay.

Binary64, DoubleFloat, arbitrary complex masses, generalized powers/shifts,
genuine numerical Duals and public ranks above three remain outside this N6
numerical qualification. Ordinary CLI defaults and Hybrid dispatch were not
changed to use this explicit path.

### 5.3 CLI and graph preparation

The CLI `prepare`, `reduce`, `run` and `compare` are implemented. See
`crates/iregi-cli/README.md` for exact JSON schema, moment normalization,
snapping/rescue semantics and physical examples. The graph/numerator preparation
is reusable Rust code outside the reduction library; coefficient evaluation is
separate from importing/applying Feynman rules.

Numerator polynomials, loop shifts, coefficients and contractions use existing
Symbolica/Spenso/Idenso facilities. Physical model JSON is pinned under fixtures.
Graph routing is canonical and parameter binding explicit; graph-derived
external kinematics are not magically inferred by `run`.

### 5.4 Physical MadLoop work

Retained examples cover top-loop `gg→H`, `gg→HH`, `gg→HHH`, and a massless up-quark
box. There are graph correspondence/routing/color factors, independent
numerator checks and graph-isolated MadLoop captures. See `validation/physical`,
the CLI docs and the physical sections of the report. MadLoop's improved phase
space can differ from the requested driver point: use captured schema-2
`evaluated_external_momenta` at the actual numerator seam.

The historical eight physical performance rows comprise six numerically measured
`ggh`, `gghh`, `gggg-up` singleton/batch rows with noisy/inconclusive timings and
two failed `gghhh` rows. Later high-precision evidence is separately recorded;
do not silently rewrite those historical failures or claim physical performance
parity. R2/UV counterterms are excluded as requested, bare poles/R1 retained.

## 6. Python interface and marimo gallery

The optional typed `tirust` PyO3 interface is implemented and locally qualified.
It links upstream Symbolica/Numerica/FeynKit/Spenso classes in one native host;
the binding crate also supports `rlib` registration. No canonical Community
module imports are used to discover constructors. Prepared expressions, tensors
and diagrams use the real upstream shared classes, not clones of their APIs.

The locally qualified development wheel004:

- Wheel SHA `5653ece749b25a8f1c4f9f4ba5538e2b888021b9d3a3ec96ea6dabd26cf570e0`.
- Native module SHA
  `cd4df512a134bb239678c9b9ba321a52f133f838f53f5886ada985f72d74df86`.
- Local wheel under `target/python-interface-build/wheels-004/` and installed
  environment `.local/tirust-wheel-004-env`.
- 29 actual tests pass: 17 numerical and 12 interoperability/rendering; zero
  skipped or failed. Installed consumer typing has zero errors/warnings.
- Canonical capsule `validation/python-interface/checkpoint-wheel-004`, source
  archive SHA `7e069179f2965266523b948635da11840e0a58974fc1965ea02641982a8a1ca6`.
- Qualified local Linux CPython 3.13 only. It is a development build, not a
  published release, other-platform guarantee or complete Community-host test.
- Wheel004 predates the final public N6 integration. Its new mass-Dual evidence
  is not evidence that its binary contains the newer six-point public API.

The native SVG failure was traced with GDB to feature-unified exact JSON:
`serde_json::Value` numeric coordinates became private-number CBOR maps. A narrow
local linnest patch changes two native CBOR sites to typed command tuples and
typed collision lines. It does not change layout geometry/math or relax the
decoder. Owner regression tests and the actual wheel renderer pass. Earlier
wheel003 failure/diagnosis and an initial malformed 1-versus-84 temperature test
fixture are retained. See `dependencies/linnest/IREGI_PROVENANCE.json` and
`dependencies/linnest-patches/0001-native-svg-typed-cbor.patch`.

Three marimo notebooks are real, executed examples:

1. Graph → numerator → integral.
2. Dots, derivatives, precision and failed-neighbor diagnostics.
3. Committed benchmark exploration with all qualification states visible.

The fixed gallery uses `.local/notebooks-wheel004-env`. Strict lint (three),
15 helper tests, native controls and three HTML exports pass, with 21/19/11
successful cells. Real Chromium desktop light and 390px mobile dark inspection
passes six views, 1,918 normal requests and no HTTP/page/console errors. The
wide chart scrolls inside its viewport. Static HTML has no live numerical
backend; do not describe it as a deployed reactive compute application.
Capsule: `validation/python-notebooks/native-wheel004`.

### 6.1 C0/D0 mass-Dual identities

The wheel004 public API now has additional direct core evidence:
`validation/c0d0-dual-identity/qualification.json`, SHA256
`ed51658ba9466624adc56fb5ced49cf50bb9d39f7a37920a78c9c78cc7316902`.

One massive spacelike scalar triangle and one box are evaluated at Float128 and
Float256, requested accuracy 53, with six ordered two-mass Taylor components
`[00,10,01,20,11,02]`. The base uses Hybrid; independently prepared raised-power
integrals use IBP. All four native Laurent slots are checked with factorial
normalization: **28 evaluations, 96 identities and 96 precision comparisons**.
All three poles are exactly zero and no fallback/analytic rescue occurs.
Maximum normalized identity errors are about `2.509e-36` (C0) and `3.712e-36`
(D0). Two independent saved-data audits perform 9,238 checks total.

The first C0 saved audit incorrectly required exactly zero real finite parts.
Its failed source/log/receipt remains. The frozen protocol already required
full complex values, so only audit arithmetic was corrected; no integral rerun,
input change or threshold relaxation occurred. D0 additionally checks rounding
of the non-dyadic Q=5 normalization against the exact physical scale.

Historical scalar-master finite differences are separate: 384-bit physical
triangle/box mixed-Dual cases pass their retained `2^-70` gates (C0 max about
`5.47e-28`, complete five-case D0 max `2.53e-30`). These were not all rerun on
wheel004. Do not replace their scope with these two identity points or vice versa.

## 7. Newly consolidated private seven-point numerical result

Capsule `validation/highpoint-thin-n7-rank0-rational-numeric`:

- Qualification SHA
  `0c95ae0b9e4b2c9bb06447b9a3fb603319f7d3b8734e30883a6f6d7fae93c0a0`.
- Manifest SHA
  `0fa4ffbe5b0306245c04850f09083b2b5e0b194d709c6bc97ddd7e5019cfb728`.
- Source archive SHA
  `5db9dfed550f2b7384991cce5f57308f4818454ce3e614746a9da7d4c418f80f`.
- Separate final byte review
  `validation/highpoint-thin-n7-rank0-rational-numeric-capsule-review.json`, SHA
  `ee84698b30a9985704f440b131a6fd29f57f05e7fc302dc2d945a880509730b5`.
- 44 payloads plus manifest, 57 archive members, including 40 source-bound core
  files; 38 copied origins and four parent manifests independently verified.

This is a private N7/rank0/unit-powers/h0/Cartesian/PV/native-components case,
one ordinary sample, independently prepared at Float128 and Float256 with
requested accuracy 53. There is no numerical substitution into symbolic plans.
The exact dyadic inputs round-trip at each precision. Geometry checks seven N6
systems and 21 N5 subsets; all 21 children are mandatory and evaluated in one
packed batch (105 scalar-master rows in the captured child execution).

The enabling change uses existing Symbolica
`RationalPolynomialField<IntegerRing,u16>` matrix determinant/inverse/product
facilities, via checked rational-polynomial conversion over the complete generic
variable map. No custom solver or polynomial engine was introduced. All-five-RHS
identities, denominator/pole conditions and original exact-complex AtomField
compatibility are preserved. Parent `validation/highpoint-thin-n7-rational-domain`
has 22 selected exact tests and constructor-only success. The old Symbolica
AtomField generic determinant was a preparation bottleneck.

Each new precision capture returns all four complex Laurent slots:

- 16 comparisons against retained **stable original IBP and Hybrid**, counting
  four slots × two modes × two precisions, pass unchanged normalized `1e-10`.
  Maximum is `4.7820439625e-12`.
- Four Float128/256 comparisons pass unchanged `2^-80`; maximum
  `9.1019999510e-34`.
- Three poles are exactly zero at both precisions.
- Original PV is unstable with zero outputs and remains visible, not counted as
  qualified agreement.
- The tiny real finite component has tracked precision **2**. Acceptance uses
  the full complex norm; this does not establish high relative accuracy for
  each real/imaginary component or achieved 128/256-bit output accuracy.
- Independent numerical audit
  `5e80602a6ffaa62504c6d89595b3cd2ace069f1b7b329fe8291180113e2e800b`
  reconstructs exact rational squared-inequality gates without importing the
  producer comparator, verifies 315 source inputs/118 closed dependency
  artifacts and current guards/grants/terminals.

Historical failures are intentionally not removed:

- First N7 capture exceeded its old 300s preparation budget and returned no Rust
  array (`validation/highpoint-thin-n7-rank0-attempt`).
- A constructor phase-attribution run exhausted a fixed 110s compute budget in
  later geometry (`validation/highpoint-thin-n7-phase-attribution`).
- Fixed scalar-oracle rules 8/12/16 across two mappings remain **inconclusive**
  at the unchanged uncertainty gate. This new result is stable-reference and
  precision agreement, **not an independent scalar-oracle qualification**.

Observed operational durations are not benchmark data. The latest captures
include complete plan/workspace/reduction and logging/provenance. The totals
including final guards were about **153.623s / 164.176s**, both within fixed
300s caps. `151.803s / 162.438s` are earlier pre-postguard elapsed fields, not
the final total. Do not mix them. Constructor-only 58.442s and later preparation
markers are diagnostic, not optimized latency.

The private core is not applied to main and no public N7 API was promoted.
Higher ranks, generalized powers/shifts, Duals, f64/DD, batches, exceptional
domains, physical graphs and performance still require work.

## 8. Newly consolidated private scheduler candidate

Read `validation/uniform-branch-dependent/README.md` and qualification JSON.
Qualification SHA is
`240c3a6190f34e782d336423321890c05e3f3e0172dfff3fb98229d0f4e1ea6e`.
Final copy audit is `validation/uniform-branch-dependent-copy-audit.json`, SHA
`8e2c0d6d91519dfc9f9f2ac9c55dcf2e6dbfd89872ff9ccd9a09bfe55b42baeb`.

The change optimizes uniform conditional branches across active batch lanes in
the Symbolica evaluator. Source review must retain the all-caller-lanes-active
condition, correct Label/Goto handling, PC pre-increment and inactive external
function behavior. Do not replace it with an intuitively similar unreviewed
scheduler. The existing production patch is unchanged; the candidate is private.

Evidence includes:

- Fifteen local exact scheduler tests over f64/DD/mixed Float/static and dynamic
  Duals, lazy/nested/shared conditionals and active traces. Both earlier failed
  fixture compile attempts are retained.
- Two uninstrumented Symbolica archives, baseline and candidate, then separate
  matching OneLoop→masters→core→probe chains. Eight dependent compiler steps
  pass; both link maps show exactly their own four fresh archives and one
  Symbolica archive. No stale matching-name archive was accepted.
- Identical read-only qualification capture hooks in both private cores.
- A fixed 60-request runtime, 30 responses per separate process, including eight
  N2/N3 rank0/1 batch1/32 families, own repeats, retained workspaces, complete
  controls and four forbidden timing/backend request fields.
- Exact complete cross-scheduler values, represented rationals, precision,
  signed zero, all requested Taylor/Laurent entries, six master outputs,
  diagnostics, fallback flags, masks, inputs and execution counters.
- Eight-family output counts: 1,584 tensor and 1,782 master complex values.
- Direct master controls: 67 cases/133 rows, 106 successes and 27 explicit
  error/null rows, 774 successful complex output components. Additional core
  controls and eight cache-interleaving steps are retained.
- Independent saved-data review
  `16dfb0c23dc50f8b7fdd18a9e9c1b52b9af9c33acb55a3e891db5dd9ea6d93f8`
  reconstructs all 120 raw events/60 requests without importing the producer
  admission code, verifies stdout correspondence and complete output equality.
- Strict admission fixes for Python bool/int ambiguity, contextual number schemas,
  known f64/DD precision and exact field sets are retained with the previous
  source packet. No numerical rerun was used to hide an output discrepancy.
- Numerical runtime was 213.900535s within fixed 220s work/240s overall limits;
  this is a lifecycle duration, **not a performance result**.

The capsule contains 72 payloads, about 24.7MB compressed, full link maps/raw
records, 347 source-context entries and 354 total archive members (roughly
391.6MB decompressed). It retains all 32 prior local history files, both failed
compiles, the old admission packet and a documentation-drift preflight failure.
Its flags correctly keep performance/full-domain/native-policy composition/live
application/adoption false.

## 9. Private work preserved but not finished

All source/evidence snapshots are committed under `validation/winddown`, with
member hashes and omitted compiled-product lists. The corresponding original
trees remain ignored under `target` on this machine. Inspect archives in a fresh
directory rather than overwriting immutable attempts.

### 9.1 Optional native master policy composed with six-point integration

Archive `native-f64-sixpoint-composition.tar.gz` preserves the full private
candidate, baseline, diffs, preps, prior drafts, source reviews, compiler/test
receipts and ordinary test logs. It has 338 noncompiled regular members;
rebuildable rlibs/rmeta/test executables are omitted explicitly in the manifest.

The reviewed combined patch SHA is
`70b5f94487f79d25c9bc831c0895fe2970ba8e222297ba0baf9f683d67101f9a`.
It changes 17 files: the older native API additions, required certified-planar
and ThinSix policy plumbing, and focused composition tests. The entire 30-file
masters crate matches the earlier qualified native candidate exactly.

Design:

- `WorkspacePolicy` remains byte-identical, including equality and constants.
  It controls private sidecars/retry and must not become a master policy.
- Separate `MasterExecutionPolicy` travels through constructors, exact cache
  keys, eager/lazy children, analytic specialization, planar retry and charts.
- Explicit `NativeF64Abc` requires its optional feature and selects generated
  A0/B0/C0 only for ordinary f64 without a Dual shape. dB0/D0/DD/Float/all Dual
  layouts remain Symbolica for that policy.
- Ordinary default constructors preserve old default behavior. The broad legacy
  native feature is distinct from this selective policy.
- Explicit SixPoint constructors force Symbolica for child and parent, ordinary
  and internal Dual paths, and expose no backend argument. This is intentional
  even when the broad legacy native feature is enabled.
- Recursive/sidecar stats and numeric recurrence bodies are unchanged by this
  composition; policy propagation and cache separation still need complete
  runtime regression before promotion.

Completed in this new combined source:

| Stage | Result | Primary receipt / independent review |
| --- | --- | --- |
| Source review | Pass | `reviews/independent-source-review.json`, hash `1fcbaacf...` |
| 8 metadata checks | Pass, 63.209s | receipt `29d1fe874e3644c2435acf5020357426d1be1a3052baae44a9a1dca92f73cdc8`; review `cc494cca...` |
| 4 code-generation steps | Pass, 103.432s | receipt `d7a6d201da1cc3237b15eebb92c170782a4f73a82832869a87fd6943f2e91e32`; review `ea8dc9ae...` |
| 20 ordinary exact tests | Pass, 163.876s | receipt `b714af2e9f493b0170a5636ccbedd86381644a1b006f229c47bed19e7efab98e`; review `ba76eebc10b0479c66b39dca7cf9edef974450dd8804eed081b2269b5180b623` |
| 7 constructor tests | **Unrun** | No `constructors-tests-001` directory or execution grant file was created |
| Whole-array / retained N6 regression | **Unrun** | Separate source/probe protocol still needed |
| Current Cargo feature forwarding | **Unrun** | Do after source integration and cache coordination |

The exact 27-test protocol is reviewed (`ef931b48...`), grouped as 20 ordinary
and seven constructor cases. Default tracks 12, selective 15. Fixed proposed
constructor limits were CPU22, 300s aggregate/eight-second reserve, 120s/test and
two 15s listings. Existing tests use owned exact binary copies and one exact
test per process. No constructors were started because the agents stopped before
wind-down, so do not invent a timeout or count authorization as execution.

Next-session sequence: inspect source/fixtures and current base; run or rebind
the seven constructor tests with a fresh supervised attempt, independent audit;
review/rebuild the 38-request whole-call capture protocol against both new chains;
repeat retained six-point reference/convergence/lifecycle gates; then consider
applying the combined patch, current Cargo feature checks and performance. Do
not apply the stale older eleven-file patch to today's N6 code.

### 9.2 Numeric-type / genuine Dual six-point pilot

Archive `n6-numeric-types-dual.tar.gz` has 81 files: private capture/helper,
exact source reconstruction, source/math reviews and bounded runner/comparator
drafts. **Nothing was compiled or evaluated.**

Source protocol proposal SHA
`7ee9f6d6017cd2ab154a32c7db3e8476c8519328862a98fac27566d396143855`;
source SHA `5acb0c7b6e7b9e522aca025fcf0320c5bf6024c3a1ec7f44b4c8598078bf12e4`;
review addendum `6f12a994693de7521f6809905161ecbdd80cc187021910f6ea4da1a406d68f80`;
independent dilation math review
`14da099580d7bd5e53f4fba3dcbf6221f365c017160189615b2ea3430d3fe7ae`.

It starts with the retained N6/rank0 primal point: ordinary f64/DD separately,
and Float128/256 six-component dynamic Duals. Two formal variables represent
**one physical dilation direction**, not arbitrary independent mass/invariant
derivatives. Shifts scale by lambda, squared masses by lambda², mu by lambda,
so the finite scalar N6 scales as lambda^-8. Symbolica Dualizer generates input
jets; separate ordinary symbolic differentiation generates expected coefficients.

Profiles are affine `lambda=1+a*x+b*y` and product
`lambda=(1+a*x)(1+b*y)`, `a=1+2^-60`, `b=3/2`. Mixed response coefficients differ
(72ab versus 64ab), and mixed squared-mass input coefficients differ (2ab versus
4ab), so the pilot is not merely repeating a constant seed. Retained jets require
at most 122 significant bits and are exactly represented at Float128/256.

Five calls/domain are intended to cover profiles, constants, repeat/reordered
lanes, invalid-mu neighbors and empty reset: ten successes, two explicit
InvalidInput lanes and one empty call; Float Dual outputs include 240 complete
Laurent/Taylor slots per precision. Matching failures must not qualify a type.
Each declared type depends only on the build so a f64 guard rejection cannot
silently suppress the other types.

Runner/common/build/prepare and `compare_saved.py` were being completed when work
stopped. `protocol-proposal.json` is a design source freeze, **not proof that a
final executable protocol and all fresh dependency bindings were reviewed**.
Inspect and finish that boundary first; prepare a new immutable protocol and
independent lifecycle/comparator review before any build. Proposed limits were
90s build/25s reserve, 180s/domain/20s reserve; they are not observations.

Crucially, imported child sensitivity/error context is **primal-only**. Dilation
agreement cannot establish requested-accuracy derivative uncertainty transport.
No outcome of this pilot alone authorizes a public N6 `new_dual` constructor.
Keep the public API deferred until the full derivative guard model is justified.

A source-only rank-seven translation assessment is also preserved:
`design/rank7-common-translation-assessment.md`, SHA
`472b28be3ef67c0a6a40f0a2dd4a86f4cdead93d7e86573cd356779738a2bb57`.
It predeclares common `t=(4,4,4,4)` and a complete 330-moment Symbolica transform
of `(k-t)^alpha`. Original IREGI recenters to the old recurrence inputs before
shifting, and its stable flag precedes the shift. Translation may alter output
conditioning but cannot repair the old highest-rank residual or erase 90 failed
slots. No such new fixture or reference execution occurred.

### 9.3 Optimized scheduler performance preparation

Archive `scheduler-optimized-followup.tar.gz` contains seven source/research files.
It is an unfinished build/performance proposal, with **no fresh optimized build
or timing execution**.

Established source census:

- The private scheduler **baseline** matches all 91 current production Symbolica
  source files, including polynomial patch0004 (`poly.rs` hash starts `0f246cec`).
- The private **candidate** differs from current only in `src/evaluate/batch.rs`.
  Watch confusing historical base/candidate filename labels; compare the actual
  patch and source hashes rather than inferring direction from a file name.
- Current OneLoop/masters/core match 108 frozen origin files. Qualification
  private cores add only three cfg observation hooks plus a capture module.
  Timing products must omit these hooks, while a separate captured product still
  verifies all master/value metadata.
- The **old optimized 6b24 closure predates polynomial patch0004** (`04d32...`
  versus current `0f246...`). Its kernel cannot be labelled current. Fresh matched
  Symbolica/OneLoop/masters/core chains are required for both variants.
- All 56 audited Numerica/Graphica source files are unchanged, and the 336 old
  lock packages remain in the current 568 with matching common checksums.
  However old optimized serde/bincode/serde_json features are broader (`alloc`,
  bincode `serde`, serde_json `float_roundtrip` in places) than current native
  Cargo. A reused optimized dependency context must be labelled explicitly;
  do not claim exact Cargo-feature graph identity.
- The contemplated minimal direct build has captured and uncaptured core/probe
  products sharing per-variant Symbolica/OneLoop/masters, about 14 products. Its
  final exact recipe, bounds, reuse proof and source review were unfinished.

A future gate must first qualify the freshly built optimized corpus, then use
the same exact eight families, requested outputs/cache/input policy and paired
IREGI/COLLIER measurements. Preserve all old rows. The one short old family is
**N2 rank0 batch1**, whose COLLIER block was 99,893,971ns versus 100ms required;
it is not “row eight.” Earlier informal “eighth family” meant one unqualified
family among eight. No historical verdict should be changed to conceal this.

### 9.4 Generalized high-point partial-fraction direction

A final research delegation was proposed but failed before doing any work due
the agent service limit. There is no implementation or hidden result to recover.
The useful next investigation remains:

- Inspect the actual extracted `dependencies/feynkit-integrals` affine partial
  fractions and existing FeynKit/HEPKit `IntegralFamily` facilities.
- Determine whether exact dimensionally regulated denominator identities can
  lower overcomplete N6/N7 families with repeated powers to supported subsets,
  including the sum-of-coefficients condition on q² and full Taylor geometric
  constraints.
- Reuse Symbolica matrix/polynomial operations; keep singular/no-solution lanes
  explicit and avoid fixture specialization.
- Record the three algebra reuse checks, concrete generic integration points and
  exact tests before a private adapter. Do not assume the existing extracted
  crate is already wired into `ReductionPlan`.

This could help close generalized coverage; it is not a substitute for validating
the original dimension-shift/rank envelope and exceptional domains.

## 10. Performance truth at pause

Performance parity is **not achieved**. The final report preserves all historical
raw source contexts, 40 historical kernel rows and 40 same-cohort ratio cells
(80 CI endpoints and five short/inconclusive cells). No new kernel measurement
was added by the final numerical/scheduler checks or wind-down.

The retained selected four-engine private prototype improves seven fully
measured families over its TIRust baseline. Against original IREGI it has one
gain and six regressions among those completed families; against COLLIER all
seven are regressions. N2/rank0/batch1 remains short/inconclusive. No default
backend adoption or global speedup is inferred.

Reference-only IREGI/COLLIER tables preserve absolute times and verdicts, not an
invented TIRust ratio. The first eight massive bubble/triangle rank0/1 rows are
a vertical slice, not the entire shared undotted domain. Higher-point COLLIER
identity/harness work exists separately; inspect its actual admitted engine
status and workspace outputs before adding timing claims.

Keep timing boundaries distinct: numerator/model preparation, coefficient
evaluation, reduction, masters and end-to-end graphs. Exclude subprocess transport
from kernel timing. Debug plan construction, codegen times and supervised capture
durations cannot be inserted into the benchmark comparison cells.

## 11. Known hard failures and interpretation traps

- Full N6 rank-seven original agreement remains unresolved. The retained
  128/256 arrays converge, but the unchanged original-reference acceptance
  gate fails 90 finite slots for one mode/corpus. Other original modes may be
  unstable or disagree differently. See
  `validation/highpoint-thin-n6-rank7-phase` and report sections. One independent
  `q0^6` component supports the TIRust value better than original, but it does
  **not** adjudicate every slot or the full imaginary part.
- N7 numerical success is a single private rank-zero point; previous timeouts
  and inconclusive scalar oracle remain real evidence.
- Primal-only retained error context does not certify high-point Dual jets.
- Native A/B/C is a selective ordinary-f64 candidate. Do not route DD/Float,
  dynamic/static Duals or D0/dB0 through it to gain performance.
- Structural source preparation and numerical caching are different. Never
  reuse a result across scale/precision/sample/derivative contexts.
- Exact source/ELF hashes prove the object used, not a complete native build-input
  history when certain native archives were only hashed after execution.
- Original reference factorial initialization was repaired only in a generated
  copy after reproducing the defect; untouched working cases and identities were
  cross-checked. Do not patch the reference checkout or silently compare against
  a different repaired convention. Read `validation/reference` and reference
  scripts before rebuilding.
- Existing supplied-invariant, legacy snapping, complex branch, rank/power and
  exceptional paths have specific domains. Generic Rust compilation is not
  numerical acceptance over all combinations.
- The PDF/data contains historical “pending” statements inside old snapshots.
  They describe their checkpoints. Later separate records supersede status
  without rewriting those raw snapshots.
- Blanket `git diff --check` can flag immutable captured compiler logs or patch
  context. Do not normalize evidence and break hashes. New production source
  had one preserved extra EOF blank line in `execution/planar_fallback.rs:210`
  from the exact tested V2 patch; it was intentionally not edited during its
  freeze. Future formatting is a separate source change requiring review.
- Do not kill unrelated shared-host processes. This machine runs other users'
  Kira/Rust work; a busy host does not imply a TIRust process is alive.

## 12. Reproduction and inspection commands

These are useful starting commands, not a demand to repeat every expensive
qualification before making progress. Inspect cached artifacts/protocols first,
run relevant meaningful checks and avoid concurrent Cargo builds.

### 12.1 Ordinary Rust and CLI

```sh
source scripts/env.sh
source .local/symbolica-license.env
cargo test --locked --offline -p iregi-rust --lib --no-run --jobs 8
cargo build --locked --offline -p iregi-rust --example six_point --jobs 8
cargo run -p iregi-cli -- run examples/graphs/bubble.dot \
  --model fixtures/models/phi3.json --input examples/inputs/bubble.json
cargo run -p iregi-cli -- run examples/graphs/triangle.dot \
  --model fixtures/models/phi3.json --input examples/inputs/triangle.json \
  --precision float --bits 256
```

The successful latest current test invocation was compile-only, then exact
selected groups using the emitted binary; it did not execute all 219 tests.
Use the capsule's `runtime/selection.json` for exact names. For numerical replay
choose fresh output/attempt paths, bind the current binary/dependency source and
serialize Symbolica execution; do not overwrite original evidence.

Example API execution:

```sh
cargo run --example six_point -p iregi-rust -- \
  validation/public-highpoint/fixture-conditioned_masses.json \
  6 3 pv 128 53 1 /tmp/iregi-six-point-new.json
```

Use a fresh path. The example captures exact rationals, input/settings, basis,
diagnostics and work counters. Its printed durations are not a benchmark gate.

### 12.2 References and physical validation

```sh
bash scripts/build-reference.sh
cargo run -p iregi-cli -- compare --fixtures validation/reference/fixtures.json
python3 validation/physical/compare_cli.py --cli target/debug/iregi-rust
```

Inspect tool arguments/path expectations before a fresh build; generated IREGI,
OneLOop, COLLIER and MadLoop code belongs under `target`, not reference sources.
Physical numerator/correspondence and reference hash evidence is committed even
when generated executables are not.

### 12.3 Python and gallery

```sh
source scripts/env.sh
python -m maturin build --manifest-path crates/iregi-python/Cargo.toml \
  --features standalone,native
```

Prefer the existing wheel004 environments for inspecting that exact historical
qualification. A newly built current wheel is a new context and needs its own
fresh installed-wheel tests, shared-class identity/rendering/stubs checks and
notebook replay. Read `python/README.md`, `docs/research/python-interface.md` and
`examples/notebooks/README.md` for test/export commands. Do not mix standalone
shared objects with a separately built Community wheel's classes.

### 12.4 Final report and preservation

```sh
bash scripts/build-benchmark-report.sh
python validation/winddown/check.py
```

Typst creation time is fixed from `results.json`'s date, so unchanged source/data
produce byte-identical PDFs. The final report's raw historical records through
`007ca9f` are preserved byte-for-byte; the new wind-down verifier checks all 279.
The older `validation/report-six-point/check.py` is a checkpoint verifier with
an explicit allowed-new-key set against `4fe1a1e`. It is not the latest report's
general verifier and can reject subsequent legitimate additions. Do not edit
old verification evidence to make its scope appear current.

For local PDF rendering, the known available Poppler executable prefix was
`/nix/store/80q9cnqf0wvs86qn57d2dlgh4y9jax97-poppler-utils-26.06.0/bin/`.
`pdfinfo` and `pdftoppm` were used; they are not necessarily on PATH. Final new
pages were rendered under `target/winddown-render`. Don't install unrelated
rendering dependencies just because `pdftoppm` is not on PATH.

### 12.5 Restore unfinished source snapshots

```sh
tar -tzf validation/winddown/native-f64-sixpoint-composition.tar.gz
mkdir -p target/restored-native-composition
tar -xzf validation/winddown/native-f64-sixpoint-composition.tar.gz \
  -C target/restored-native-composition
```

The archive root corresponds to its origin directory, not the repository root.
Original runner paths are absolute or rooted at their original `target/...`
layout. A fresh checkout must reconstruct that layout in a fresh tree or rebind
paths/source/tool hashes in a new reviewed protocol. Never restore over an
immutable attempt. Member hashes are in `validation/winddown/manifest.json`.

## 13. Local caches and operational resource management

Ignored local state is retained deliberately. “Clean repository” means no
uncommitted source/evidence or active owned work; it does not mean deleting
validated binaries and caches needed for efficient restart.

Useful caches on the present machine include:

- `target/cargo-build/debug/deps` for ordinary Cargo artifacts.
- `target/debug/examples/six_point` and the current Cargo capsule's frozen exact
  copies under `target/public-six-point-cargo/runtime/bin`.
- `target/e6-direct-n2/deps` (113 frozen files), `native` (three) and
  `target/core-fused-cse-census/supplemental-deps` (two) for older private direct
  rustc qualifications. Their search-directory inventories are closed and
  hashed. Adding an unbound file there can invalidate a frozen protocol.
- `target/public-highpoint/...` for private N6/N7 candidates and captures.
- `target/native-f64-opt-in/...` for historical selective-native evidence and
  source-identical master products, including a preserved failed broad metadata
  aggregate with later explicit supplemental completion/audits.
- `target/native-f64-sixpoint-port/...` for the current combined private source,
  two built test binaries and 20 ordinary exact test logs.
- `target/guard-performance-review/...` for baseline/candidate scheduler source,
  both compiled chains and complete runtime records.
- `.local/cargo-home`, `.local/cache`, `.local/symbolica-locks` and Python/browser
  environments. Do not commit these.

Avoid concurrent Cargo actors. Earlier this session `/root/python_interface` was
the sole Cargo actor; all its work is now terminal and its freeze released.
Separate direct rustc builds were allowed only with frozen source/closed extern
graphs, separate output directories and nonoverlapping CPUs. A later session
must acquire its own coordination rather than assuming these old grants apply.

Licensed numerical execution was serialized. Helpers own whole process groups,
persist terminals before risky hashes, reserve cleanup time, terminate/reap
descendants and independently verify absence. An observation timeout is not a
terminal result: poll the same session or inspect actual process state; do not
restart a numerical attempt because tool observation expired.

The widely reused exact helper SHA is
`c193cd7282d63b63a8f627612df684242a2a8941d1d0a35588514d0415ab8824`.
It exists in several capsules. Some reviewed N7 stages use their own preserved
supervisor with equivalent documented bounds; inspect its actual semantics, not
just a remembered helper label. Old grants/limits record operational provenance,
not a new user-facing permission mechanism.

All TIRust-owned numerical/compiler processes were terminal at wind-down.
Important releases include:

- Current Cargo's 11 groups: `6bb1044378984b84bf7e1c0c36043e117ea5dd492e16d893a8db1c1115a73fb1`.
- Scheduler groups 2107501/2107553:
  `00d875ba808e1a65305cbab63d1bb7a107765d69fcfc08a2d44611758335b713`.
- N7 driver/captures groups 1587026/2266835/2384082:
  `4f89904381550b1aec4bfc73cc3a9e3e1061e7c270fc1776017d617085b98890`;
  saved comparator was separately collected and its group absent.
- Native ordinary20's 22 groups:
  `5ec7de1d9d403b2b2c5127fbe73adc561c98c7dd627e2ff1ec808802f0a834a4`.

These PIDs are historical; don't kill a recycled PID in a later session. Inspect
current command/cwd/start time before taking action. Several agents stopped due
a service usage limit late in the session. Their partial files were preserved;
no running check should be inferred from an old agent status or lock file.

## 14. Ordered restart plan

The user may reprioritize, but the following order builds on the actual state:

1. Confirm user resume, goal status, clean current `main`/origin, environment,
   license availability and latest HANDOFF/report. Verify the new wind-down
   archive/report checks. Do not rerun all historical expensive tests by default.
2. Finish the combined native-policy seven constructor tests, independently audit
   complete logs/process cleanup, then create/review the 38-request whole-array
   and retained N6 reference/convergence/lifecycle regression protocol. Only after
   those pass consider production application and current Cargo feature tests.
3. Finish the matched optimized scheduler build recipe and source/reuse review.
   Rebuild current Symbolica/OneLoop/masters/core in both variants, captured
   qualification and uncaptured timing products. Complete numerical admission
   before timings. Measure the same eight families/reference outputs/cache
   policies with meaningful block durations and uncertainty. Keep all rows and
   TIRust numerators, with the original short family visible.
4. Finish and review the N6 typed/Dual pilot's final closed runtime/comparator
   protocol and fresh preflight. Qualify actual f64/DD/Float jets and lane
   lifecycle under independent declared stages. Diagnose guard failures rather
   than weakening them. Develop proper derivative uncertainty transport before
   any public N6 Dual exposure.
5. Turn the private N7 rank0 success into a complete, reviewable source integration
   proposal against current main. Expand rank, types, genuine batching, routing,
   exceptional and generalized coverage. Keep the original PV instability and
   old failed preparation/oracle records. Strengthen the independent scalar
   oracle with a predeclared fixed-rule/error protocol if needed.
6. Adjudicate the full N6 rank-seven discrepancies with independent full-array
   evidence. One supporting component and self-convergence are insufficient.
   A translated/rerouted point may add coverage but cannot erase the original
   failure. Preserve separate real/imaginary accuracy limits.
7. Investigate existing affine partial fractions/IntegralFamily for generalized
   N6/N7 powers/dimension shifts; wire existing FeynKit algebra only after the
   three reuse checks. Keep all original semantic domains in the acceptance map.
8. Requalify current physical graph numerators/results and broaden COLLIER's
   shared undotted topology/rank benchmarks. Close existing massless/threshold/
   Gram/Cayley/complex and batch/concurrency gaps with complete arrays.
9. Refresh current Python wheel and real marimo execution when public APIs/math
   change. Keep shared-host/other-platform/release boundaries explicit.
10. Iterate correctness/performance gates until the actual complete PLAN is true.
    Maintain report, meaningful tests and periodic commits/pushes. At each push
    provide the exact GitHub `docs/benchmark.pdf` link. Only then can the goal be
    marked complete with a requirement-by-requirement evidence audit.

## 15. Consolidation and cleanup policy used here

- No unqualified candidate was silently merged into default production code.
- Completed N7 and scheduler source/raw data/reviews are committed as capsules.
- All unfinished substantive source/research is archived with member hashes and
  a truthful state map. Compiled products are omitted explicitly, not mistaken
  for source loss; rebuilding them is possible from source/protocol provenance.
- Old failed, unexecuted and superseded drafts remain distinguishable. Source
  reviews and packaging checks are not relabelled as numerical tests.
- The working tree is committed and pushed on main. Ignored target/local caches
  are retained; no destructive `git clean -fdx` or reference mutation was used.
- Private license contents and compiled binaries are scanned out of staged
  payloads, recursively including compressed archives. No literal key is in
  this handoff.
- The final Typst/PDF/data are rebuilt and checked independently for deterministic
  bytes and preservation of 279 earlier raw result objects. New pages were
  visually inspected. Historical timing statistics are unchanged.
- This handoff and archived rolling notes preserve design choices, provenance,
  next actions and caveats. Canonical raw receipts take precedence over prose
  if a later audit finds a discrepancy.

The next session should be able to start from the committed tree and these
archives even if the present `target` caches disappear. Fresh artifact hashes,
attempt records, resource supervision and independent validation are still
required for newly built or changed source. The project is intentionally paused
with substantial progress and explicit unfinished acceptance gates.
