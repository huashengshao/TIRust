# COLLIER 1.2.10: source audit and proposed experiments

Audit date: 2026-10-23. This is a source study, with no COLLIER build, numerical
comparison, timing qualification, or production change. The source package is the
one supplied by the user through the [official download](https://collier.hepforge.org/downloads/?f=collier-1.2.10.tar.gz).
Its SHA-256 is `460624ea709211506b642343a7f6be41d3883ccb52598b14950f8e07c0262713`
(928,595 bytes). The accompanying `collier-1.2.10-manifest.json` binds the download
receipt, extracted files, and the project sources inspected here.

## Recommendation

The most useful new experiment is a **Cartesian rank-lowering recurrence for
six and seven denominators**, derived from the published Denner–Dittmaier
identities using Symbolica. COLLIER uses rank `r-1` pinched children for a rank `r`
high-point tensor. Our current high-point constant denominator identity keeps
rank `r` in every pinched child. This difference is directly relevant to the
unfinished rank-seven preparation runs.

COLLIER also supplies practical models for choosing a well-conditioned chart,
ranking exceptional expansions before running them, stopping when a propagated
error floor dominates, and retaining independent-method disagreements. These
are promising policies, not evidence that our acceptance criteria are met.
The current six-point run stops during preparation, so changing small-Gram
expansion formulas alone does not address its immediate bottleneck.

## Source and reuse boundaries

All COLLIER paths below are relative to
`target/reference/collier/source/collier-COLLIER-1.2.10/`. Line numbers refer to
the exact extracted bytes recorded in the manifest. Project paths are relative
to this repository, with separate hashes in the same manifest.

Extraction checked all 64 archive members for absolute paths and `..` traversal.
Only regular files and directories were extracted. Two convenience symlinks,
`manual/collier.pdf` and `manual/collier.tex`, were omitted; their ordinary target
files `collier_manual-1.2.10.{pdf,tex}` are present. No reference tree was changed.

`src/COLLIER.F90:1–8` and the other source headers identify GNU GPL version 3;
`COPYING:1–5` contains the GPLv3 text. This audit records ideas and published
identities. It does not copy Fortran routines into the Rust library or add a
COLLIER dependency. Any future direct source reuse must retain its provenance
and address the stated license before distribution. The project requirement
that all algebra use Symbolica/FeynKit remains in force: do not transplant
COLLIER's `chinv`, determinant, tensor-polynomial, or simplification routines.

The published references are:

- Denner, Dittmaier, Hofer, [COLLIER](https://arxiv.org/abs/1604.06792),
  Comput. Phys. Commun. **212** (2017) 220–238,
  DOI `10.1016/j.cpc.2016.10.013`. The package's updated manual contains the
  implementation description inspected here.
- Denner and Dittmaier, [Reduction schemes for one-loop tensor integrals](https://arxiv.org/abs/hep-ph/0509141),
  Nucl. Phys. B **734** (2006) 62–115,
  DOI `10.1016/j.nuclphysb.2005.11.007`: small-determinant expansions,
  five/six-point rank lowering, and the generalization after Eq. (7.10).
- Denner and Dittmaier, [Reduction of one-loop tensor 5-point integrals](https://arxiv.org/abs/hep-ph/0212259),
  Nucl. Phys. B **658** (2003) 175–202,
  DOI `10.1016/S0550-3213(03)00184-6`.
- Denner and Dittmaier, [Scalar one-loop 4-point integrals](https://arxiv.org/abs/1005.2076),
  Nucl. Phys. B **844** (2011) 199–242,
  DOI `10.1016/j.nuclphysb.2010.11.002`, for the scalar branch conventions.

The first three arXiv landing pages were independently checked through the web
tool. That tool could not open the download endpoint; the separately recorded
HTTPS download succeeded and supplied the inspected archive. This study does
not infer version 1.2.10's contents from a newer online manual.

## Ranked experiments

Costs below are qualitative engineering estimates, not elapsed-time promises.
Benefit remains unmeasured until the existing correctness and timing gates run.

| Priority | Experiment | Current bottleneck addressed | Implementation / validation cost |
| --- | --- | --- | --- |
| 1 | Direct Cartesian high-point rank lowering | N6/N7 rank-seven child-plan and stability-graph size | Medium / high: independent identity derivation, all tensor/Laurent components and Dual jets |
| 2 | Choose charts by coefficient amplification | Ill-conditioned otherwise nonzero charts; failed-lane rescue | Medium / high: full chart domain, lazy guards, mixed-lane routing and derivative checks |
| 3 | Use DAG-local error propagation to avoid expanded sensitivity trees | Master-activation/amplitude preparation growth | Medium to high / high: retain declared master-error assumptions and branch semantics |
| 4 | Adaptive C/D small-Gram and mixed Gram–Cayley expansion portfolio | Exceptional triangle/box coverage and expensive fallback work | High / very high: several regimes, remainder policy, complex masses and Taylor components |
| 5 | Precompute tensor symmetry and metric insertion schedules | Repeated tensor reconstruction and preparation | Low to medium / medium: signs, multinomial factors, routing and sparse contractions |
| 6 | Merge identical subrequests and allocate only needed ranks | Repeated work across requested outputs or related graphs | Medium / medium: much of the single-plan reuse already exists |

## Actionable high-point identity specification

This subsection is an independent integrand derivation, not a translation of
COLLIER's coefficient code. Adopt the project's metric
`eta = diag(1,-1,-1,-1)` and split `q_D = qbar + qperp`, with all shifts in
the four-dimensional `qbar` space. Write

```text
D_i = (q_D+p_i)^2 - m_i^2
    = q_D^2 + 2 sum_nu eta_nu p_i^nu qbar^nu + d_i,
d_i = p_i^2 - m_i^2.
```

For a Cartesian index `mu`, let Symbolica solve the exact six-row system

```text
sum_i w_i^mu = 0,
2 sum_i eta_nu p_i^nu w_i^mu = delta_nu,mu  (nu=0..3),
sum_i d_i w_i^mu = 0.
```

Substituting these equations into `sum_i w_i^mu D_i` gives `qbar^mu`.
The coefficient of the *entire* `q_D^2`, including `qperp^2`, is exactly zero.
For a numerator multi-index `alpha` with `alpha_mu>0`, multiplication before
integration therefore gives

```text
I_D(a; alpha) = sum_i w_i^mu I_D(a-e_i; alpha-e_mu).
```

Here `I_D(a;alpha)` denotes the original normalization and integration
dimension on **both** sides. With unit powers, every nonzero term has one fewer
denominator and one lower numerator rank. No four-dimensional integration
approximation has been made, so this identity alone loses no evanescent or R1
term. The scalar case `alpha=0` is separate. For raised positive powers the
identity is still algebraically true but `a_i-1` need not pinch a line. It is
also true at a common shifted dimension wherever the regulated integrals are
defined by continuation; this observation does not qualify the project's
generalized tensor-interface mapping or its shifted scalar masters. Restrict
the first production proposal to the already supported ordinary unit-power
Cartesian domain until those mappings receive independent tests.

### Reuse the existing relative-coordinate matrix

Choose origin `p_0`, let `k_i=p_i-p_0`, and
`f_i=k_i^2-m_i^2+m_0^2`. For a selected set of five non-origin lines, form the
same five-row matrix `K` used by the current constant denominator identity:
its first four rows are `k_i^nu`, and its last row is `f_i`. Solve

```text
K * w^mu = (eta_mu e_mu / 2, -p_0^mu),
w_0^mu = -sum_(i>0) w_i^mu,
w_i^mu = 0 for unselected lines.
```

Indeed `D_i-D_0 = 2 k_i·(qbar+p_0)+f_i`; the first four equations produce
`qbar^mu+p_0^mu` and the final equation subtracts `p_0^mu`. In particular,
the last right-hand-side component is **not zero on translated routing**.
The scalar constant identity uses the different right-hand side
`(0,0,0,0,1)`. Cache the shared matrix/chart geometry, not a solution with the
wrong right-hand side. Symbolica's `Matrix::solve` currently accepts one RHS
column (`dependencies/symbolica/lib/numerica/src/tensors/matrix.rs:1947–1955`);
do not assume an unimplemented multi-RHS solve. Benchmark existing `inv` plus
matrix multiplication versus separate `solve` calls before choosing the
preparation strategy.

Pure pinching requires this five-row chart to be nonsingular. Four spanning
momentum differences by themselves are insufficient: `f` can lie in their row
span. Conversely an exact solution need not divide by the external five-vector
Gram determinant, which vanishes identically in four dimensions. N6 has one
choice of five non-origin lines; N7 has six. The scalar and four numerator
directions may share the same selected chart.

If only a four-momentum basis is usable, the existing denominator-difference
identity instead gives `qbar^mu = A^mu + sum_i v_i^mu D_i`. Its `A^mu` term
requires a same-topology rank-one-lower moment. This is an exact possible
secondary recurrence, and should be distinguished from pure pinching. It can
cover some extended-Cayley-singular cases, but introduces an inverse external
four-vector Gram matrix and its own conditioning limits. Do not silently use
it when that Gram matrix is singular or claim that it resolves every small-Gram
case.

### Chart guards and an explicit amplification metric

For `FromShifts` with preserved inputs the coordinate identities hold in every
Taylor coefficient by construction. Require a finite, nonzero **primal** chart
determinant and the existing precision/conditioning checks; its inverse then
has a local Taylor expansion. A primal determinant that rounded to zero or
nonzero is not by itself an exact conditioning certificate. A merely nonzero
but tiny determinant is a reason for precision rescue or another chart, not a
reason to erase the guard.

For independently supplied Gram products, symmetry/zero distance diagonal or
Gram rank at most four alone does **not** prove compatibility with the supplied
Cartesian shifts. This recurrence must initially be disabled there, unless a
separate exact full-Taylor certificate proves every coordinate/invariant
identity actually used by its coefficients and scalar children. The existing
invariant-only rank-zero wrapper's covector coordinates do not authorize
arbitrary Cartesian moments. Preserve the original input-policy and on-shell
snapping restrictions; do not reconstruct a rounded geometry and claim an
identity for the original inputs.

For a common positive kinematic mass scale `Q`, a simple dimensionless chart
score is `max_mu sum_i Q*|w_i^mu|`. This uses the coefficient one-norm, while
COLLIER's observed subset selection uses a coefficient maximum. The sum is a
useful amplification diagnostic for equal normalized child error scales. A
more informative estimate is `sum_i |w_i^mu|*delta_child_i`, with explicitly
declared child-error assumptions. Neither score is a mathematical integral
error certificate. Complex norms, `Q`, target scales and scores must remain in
the lane's generic precision, with no `f64` conversion.

Choose charts from the primal, group lanes, and differentiate the selected
analytic identity through Symbolica's Dualizer. On an overlap the exact chart
identities agree; switching algebraic representations is not a physical branch
singularity. Check *all* Taylor components and retain inactive-branch laziness.
A chart with small predicted amplification can still fail final precision
checks. Every chart failed, an unresolved invariant certificate, or an exhausted
resource budget must leave an unsuccessful lane and the other lanes intact.

### Small first experiment and required witnesses

1. Prove all six equations and `sum_i w_i^mu D_i-qbar^mu=0` through Symbolica,
   for all four Cartesian directions. Use symbolic `qperp^2` as an independent
   indeterminate so its exact cancellation is explicitly tested.
2. Keep the existing conditioned N6/N7 fixture
   `validation/public-highpoint/fixture-conditioned_masses.json`, SHA-256
   `ec27f2c5774a18d3d84d2d733e634dae93e9bd4c74523d6f77e8784d13421ada`.
   Start with its N6 prefix and retained base/scaled/mass-deformed cases, then
   add translated and reversed versions of that same fixture. Translations
   exercise the nonzero final RHS component without substituting a new, easier
   benchmark. Its N6 origin is zero and non-origin shifts are
   `(0,1,0,0)`, `(0,0,1,0)`, `(0,0,0,1)`, `(1/2,0,0,0)`, `(0,1,1,0)`;
   the squared masses are `1,2,3,4,2,3`, and `mu=3`.
3. Compare all rank-one components first with the existing constant-identity
   path and untouched IREGI. Then retain every component through rank seven
   before comparing contractions, including zero and nonzero Laurent slots.
   Source counts and phases belong to the same unchanged fixture.
4. Include a singular first N7 chart with a regular alternative, near-singular
   charts at a sequence of precisions, rank-three/planar coordinates, and full
   four-span but singular extended `[k;f]`. Demonstrate explicit rejection or
   independently qualified fallback, never successful zero tensors.
5. Cover complex masses, scaled/translated/reversed routing, exact scaleless
   cases and massless poles. Check empty/singleton/odd/batch-32 groups, mixed
   chart choices, repeated/reordered/chunked calls, and a failed neighboring lane.
6. Validate `DoubleFloat` and increasing-precision `Float`; include full first,
   second and mixed static/dynamic Dual jets. Use analytic identities and
   independent high-precision differences away from physical singularities.
   Retain inputs that differ beyond 53 bits. Generalized-power and shifted-D
   identities are separate later witnesses if that domain is exposed.

The first experiment remains private and uses Symbolica's matrix, expression,
polynomial and evaluator operations. Record its implementation and independent
review before altering production dispatch. Per-family performance qualification
still requires matched complete outputs, singleton and batch throughput, and
the existing uncertainty gate.

An independent read-only review by the high-point coverage agent checked the
relative RHS, full `q_D^2` cancellation, same-D generalized-power algebra and
the supplied-invariant restriction, and found no equation/scope blocker. That
review is not a numerical validation of an implementation.

### 1. Lower rank while pinching high-point tensors

**Observed source.** `manual/collier_manual-1.2.10.tex:1269–1275` describes direct
N=5/6 reduction without inverse external Gram determinants and the N>=7
extension. Lines 1347–1362 explain why direct Cartesian tensors avoid a redundant
covariant coefficient representation. `src/tensors/TensorReduction.F90:1216–1250`
sets `rmax_m1 = max(rmax-1,0)` and requests six pinched child tensors. Lines
1400–1431 accumulate each nonzero-rank Cartesian tensor from the corresponding
rank-one-lower child component and a vector coefficient. The explicit N6
implementation has the same pattern at lines 617–643. The scalar coefficient
has its own construction at lines 1291–1329; it cannot be obtained by blindly
applying the rank-lowering expression at rank zero.

**Difference from this implementation.** `crates/iregi-rust/src/pv.rs:417–437`
uses the exact constant denominator identity for N>=6 and recursively asks for
the *same* Cartesian exponents on each child. N5 already has Cartesian rank
lowering (`five_point_moment`, line 459). The existing
`docs/research/cartesian-highpoint-identities.md` establishes the exact
denominator-difference conventions, so this proposal extends that derivation
rather than replacing it with a Fortran translation.

For all requested moments through rank seven, the Cartesian output count is
330. A direct N6 rank-lowering path would require children only through rank
six (210 components); a successive N7→N6→N5 path could reach N5 children through
rank five (126). These are combinatorial counts, **not speedup estimates**:
topology multiplicity, scalar paths, symbolic geometry and stability work also
matter. Our compiler already uses Cartesian high-point moments, so COLLIER's
covariant-versus-Cartesian count advantage must not be counted again.

**Proposed pilot.** Derive the coefficient relation at the integrand level from
Eq. (7.7) of the 2006 paper and exact denominator differences, preserving
four-dimensional numerator indices and dimensionally regulated denominators.
Let Symbolica construct/solve the matrix and prove the resulting identities.
Compare an isolated rank-one and then rank-seven compiler to the present exact
constant-identity path before modifying public dispatch. Keep the latter path
for singular geometries and rank zero. Measure alias count, active masters,
preparation phases and peak memory before a warmed runtime comparison.

This pilot initially covers unit powers and zero dimension shift. The observed
COLLIER entry points do not expose the full generalized-power/dimension-shift
interface required here, so no generalized-domain inference is justified.
COLLIER separately computes UV terms once rank reaches `2*N-4`
(`TensorReduction.F90:1346–1388`); this threshold is above rank seven for N6/N7,
but must remain explicit if coverage later increases. Qualification must still
include massless poles, complex masses, translated/reversed routing, every
Laurent slot, all mixed Dual components, and incompatible supplied invariants.

### 2. Rank charts by coefficient amplification

**Observed source.** `TensorReduction.F90:1089–1164` considers cyclic five-momentum
subsets for N>6. Within a subset it selects the largest absolute determinant
(lines 1117–1129), then forms the actual Cartesian recurrence coefficients and
selects the subset minimizing their largest magnitude (lines 1134–1163).
Some comments say “smallest determinant”; the executable comparisons use
`>` and choose the largest determinant. The final choice is therefore more
specific than determinant size alone. The source propagates errors from all
five selected non-origin children at lines 1430–1435, with an explicit 2023
correction adding the last two children to an older incomplete estimate.

**Difference here.** `pv.rs:1025–1098` retains all six N7 five-column charts and
chooses lazily by nonzero determinant; it does not select the least amplified
representation. Existing exact rank certificates are a useful foundation.

**Proposed pilot.** On lanes for which several charts are certified usable,
evaluate a dimensionally normalized amplification estimate from their actual
coefficients, group lanes by selected chart, and execute each group as a batch.
Keep chart metrics as diagnostics; determinant size or a finite primal inverse
alone does not certify a Taylor identity or full precision. A source with
physical dimension must be normalized before comparing amplification across
rescaled kinematics. Generating every full symbolic chart eagerly might worsen
preparation, so qualify a bounded small portfolio before enabling this policy.
Test a first chart that is singular, a nonzero badly conditioned chart, overlap
agreement, precision changes, reordered mixed batches, and full Dual jets.

### 3. Propagate declared error assumptions through the recurrence DAG

**Observed source.** The manual at lines 1222–1247 starts with scalar accuracy
assignments and propagates errors through reduction. Direct high-point
recurrences propagate child estimates weighted by coefficients rather than
expanding all master dependence (`TensorReduction.F90:1335–1341,1430–1435`).
For expansions, inherited triangle errors and estimated truncation errors are
kept separately (`src/COLI/reductionD.F90:5109–5131,5218–5219`).

**Difference here.** `stability.rs` already constructs bounded local-sum guards
and primal master-sensitivity aliases. Its comments explicitly call these
estimates, not certified master-error bounds. The next unfinished N6 run is
reported in `docs/checkpoints/2026-10-23-pause.md` as timing out during master
activation after primal amplitude construction. An unexpanded DAG error
schedule could avoid some of that preparation, but COLLIER's source does not
prove that such a schedule would cover our original guard assumptions.

Continue the already written `docs/research/dag-roundoff-guard.md` design:
use Symbolica's evaluator/Dualizer and existing ball arithmetic, preserve lazy
branches, and distinguish represented-master arithmetic error from assumed or
certified physical-master error. Do not replace error sums by COLLIER's maxima
and call the result a conservative bound. A zero radius around an inaccurate
scalar master does not certify the integral. Prototype alongside existing
guards with unchanged statuses; validate adversarial cancellation and branch
uncertainty before considering replacement.

### 4. Adaptive small-Gram / Gram–Cayley expansion portfolio

**Observed source.** The manual at lines 1213–1262 specifies ordinary PV first,
preliminary error estimates to rank alternatives, iteration until the target
or an accuracy plateau, trying other promising methods, and retaining the
best estimate if all miss the target. The concrete D scheduler estimates PV
at `reductionD.F90:914–967`, Gram expansion at lines 1203–1241, and mixed
Gram–Cayley at line 1296 onward. It has distinct `CalcDg`, `CalcDgm`, `CalcDgr`,
`CalcDgx`, `CalcDgy`, `CalcDgp`, `CalcDgpf` routines, not one universal small-Gram
series. The C module similarly contains multiple variants
(`src/COLI/reductionC.F90:4350,4980,5596,6335,7539,8008`).

`CalcDg` retains an explicit expansion-order axis and chooses stable adjugate
entries (`reductionD.F90:4720–4785`). It detects growing corrections at lines
5091–5104, estimates the next contribution using recent term ratios at lines
5153–5157, and stops increasing order when inherited triangle error dominates
at lines 5166–5173. `CopyDimp3`, lines 10037–10084, retains improved results by
rank. These are useful scheduling ideas; they are not a rigorous remainder
certificate. The preliminary estimates can also be revised when actual errors
exceed them (e.g. lines 2394–2428).

**Practical transfer.** Keep a structural expansion-order axis distinct from
both sample lanes and physical Dual components. Use Symbolica series and
coefficient operations; group active lanes by selected method/order and reuse
lower-point leaves as orders increase. Stop work when scalar uncertainty or
arithmetic cancellation dominates, with an explicit unresolved status. Existing
exact lightlike bubble branches and the restricted near-vacuum D0 certificate
are useful narrow examples, but do not constitute general C/D small-Gram
coverage. General C/D families require independent expansion/remainder analysis,
threshold/complex-mass tests and high-precision convergence before public use.

COLLIER's policy of returning a best estimate after missing its target must be
adapted to our API: a usable diagnostic estimate may accompany an unsuccessful
lane; it must not turn incomplete or insufficiently qualified results into
`Success`. The manual's default target is `1e-8` (lines 2436–2450), whereas our
binary64 acceptance target remains `1e-10` after dimensional normalization.

### 5. Tabulate tensor insertion schedules without introducing algebra

**Observed source.** `src/tensors/InitTensors.F90:73–115` enumerates symmetric
Cartesian components once. `SetAddGtab`, lines 170–209, tabulates metric-pair
insertions; `SetCFtab`, lines 219–247, tabulates metric signs and combinatorial
weights. `src/tensors/BuildTensors.F90:524–591` recursively builds momentum-only
structures for boxes and adds metric contributions through those tables. This
implements the manual's description at lines 1283–1296.

**Difference here.** `pv.rs:519–575` builds a generating polynomial and calls
Symbolica `coefficient_list` once per rank/Laurent slot. Symmetric tensor basis
enumeration is already shared. A reusable insertion/selection schedule could
reduce repeated structural work, but replacing coefficient extraction with a
new hand-written polynomial engine would violate the project rules. Check
Symbolica/FeynKit tensor symmetrization facilities and benchmark their schedules
against the current generating-polynomial path. Preserve all signs and
multinomial weights; compare every moment before sparse contractions. This is
secondary to the high-point preparation work because the latest timeout does
not isolate tensor reconstruction as its limiting phase.

### 6. Structural request reuse, with explicit numeric contexts

**Observed source.** The manual at lines 1535–1560 describes recurrence-path IDs
and learning equivalent calls across initial events. `src/Aux/cache.F90:1240–1255`
describes rank optimization and storing only multiply used requests. During
learning it checks an argument-sum prefilter followed by every exact argument
(`cache.F90:1347–1361`); the sum is not its only equality test. The optimized
event path (`cache.F90:1296–1334`) uses the learned pointer and call order,
without rechecking every numerical argument. Thus its fixed call sequence and
unchanged event settings are real correctness preconditions, not merely hints.

Our immutable plans, coefficient/moment maps, dense workspaces and grouped
master plans already implement much of the local reuse. Reuse the useful idea
of collecting maximum requested rank per structural subrequest before lowering
the batch. Do not introduce an event-order-dependent global numeric cache.
Any later multi-graph fusion must key structural identities independently of
sample values and keep scale, policy, precision, full Dual shape and derivative
context in any numeric reuse boundary. Test chunking, call reordering and
independent concurrent workspaces.

## Additional independent validation opportunity

The manual at lines 1431–1445 and 2415–2427 describes independent COLI and DD
branches and a comparison mode which includes their disagreement in its
reported estimate. `src/collier_tensors.F90:3840–3895` shows that comparison at
the full Cartesian tensor level. An optional isolated COLLIER executable could
provide a third external reference for stable exceptional C/D cases or N6
moments, alongside original IREGI and independent identities. It would remain
optional validation tooling; Rust masters continue to use oneloopmaster.

The supplied manual at lines 1694–1696 and `src/DDlib/DD_global.F90:117–126`
limit the DD branch to pentagon rank five and hexagon rank six. Do not advertise
that branch as an independent rank-seven N6 oracle. COLLIER's `DD` name also
does not mean Symbolica `DoubleFloat`; the inspected routines use `double
precision` / `double complex`. No arbitrary-precision, batched, or Dual
capability is inferred from that name.

Before numerical use, adapt and test metric/routing, scale, normalization and
UV/IR conventions. In particular the manual at lines 1175–1193 distinguishes
the convention change for the IR double-pole finite contribution. Compare all
four native Laurent coefficients and all symmetric Cartesian components;
do not validate only a sum of poles or a contracted numerator. Persist errors
and unresolved cases. A new external reference result would still not prove
our per-family performance parity.
