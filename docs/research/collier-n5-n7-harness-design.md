# COLLIER N5–N7 reference harness extension

Source design only, 2026-10-23. No new driver has been compiled or run for this
extension. The N4 fixed-input diagnostics are retained separately, with six of
eight COLI rows rejected by the unchanged requested-accuracy status gate.
Those failures must remain in later coverage and reports.

## Numerical and API envelope

Extend a new optional reference driver rather than modifying the frozen
N2/N3 timing producer or the N4 diagnostic binary. Use the generic public call

```fortran
call TNten_cll(tensor,tensor_uv,mom,invariants,masses(:,lane),n,rank,tensor_error)
```

Its exact signature is `src/collier_tensors.F90:5885–5891`: component vectors
`mom(0:3,n-1)`, invariant array of length `n*(n-1)/2`, squared masses with
zero-based upstream dummy indices, four-dimensional component tensors, and
absolute error estimates `tensor_error(0:rank)`. The existing actual mass array
has n entries, which conform to the dummy array for n>=2. Keep the explicit
Cartesian gather already used by the original driver; never assume upstream
flat storage has IREGI ordering. The upstream manual at lines 2172–2222 gives
the corresponding N-specific wrappers and Cartesian multiplicities.

Use the same optimized strict Fortran flags and immutable static library.
Only a new driver object/link is needed. `Init_cll(n,rank,'',.false.)` sets
dynamic topology/rank bounds; `src/collier_init.F90:150–156` records those
bounds and `:221–222` sizes expansion support. First cover n=5,6,7 at ranks
0,1,2, batches 1/32. Then add rank3 and representative ranks5 for N5,6 for N6,
and7 for N7, retaining explicit unsupported/rejected outcomes. Ordinary unit
powers and D=4−2epsilon remain the only denominator interface here. Neither
generalized powers nor shifted dimensions can be inferred from these calls.

COLI mode1 is the primary comparison branch. DD mode2 and cross-check mode3
are separate policies and must be reported separately. DD coefficient
`E_cll` rejects rank>5 and `F_cll` rejects rank>6
(`collier_coefs.F90:1907–1912`, `:2511–2516`); `G_cll` is unimplemented in DD
(`:3137–3142`). However, **this does not prove that generic N7 Cartesian
tensors are unsupported in mode2**. The default direct tensor reduction starts
at N=6, reducing N>=6 tensors recursively to N5 coefficient calls. The source
selects this path at `collier_tensors.F90:5985–5996`; its N5 boundary calls
coefficients at `TensorReduction.F90:791–797`, and its higher-N recursion
reduces both topology and requested rank at `:1216–1244`. Thus test the generic
mode2 N7 path and its native status before declaring its support. Explicitly
set and record `SetTenRed_cll(6)`; changing that setting defines another policy,
not an invisible optimization. The manual explains this default and its
performance rationale at lines 2540–2570.

An interface path alone does not establish kinematic coverage. In particular,
DD's massive scalar box implementation requires real Euler transformations.
The exact source is
`target/reference/collier/source/collier-COLLIER-1.2.10/src/DDlib/DD_4pt.F`,
SHA-256 `6ff001947c81af4156440b79b9f97d00c59bcbb5141ddf89f363a84cc41cb48b`.
Lines 4448–4498 set its fatal `-10` flag for nonpositive transformation
discriminants. Source inference: the beta(2) expression at lines 4476–4485 is
strictly negative for noncollinear purely spatial p1,p2, conditional on the
actual routing q1=p1², q2=(p2−p1)², q12=p2². The fixed N4 fixture has that
routing and noncollinearity, so this explains a sufficient unsupported
condition consistent with its observed DD rejection; no internal execution
branch trace was recorded. Higher-point
DD paths can inherit that limitation from a pinched box even if the root
kinematics have a nonzero time component. Retain and classify those failures;
never substitute another fixture to erase an unsuccessful fixed family.

Keep local-only recurrence caching, event reset, requested accuracy1e−12,
critical accuracy1e−10, UV/IR squared scales μ², finite checks and all native
status flags. An inaccurate or internally failed lane retains null accepted
moments. Labelled raw diagnostics never qualify it. Preserve error causes,
complete responses and source/input/binary guards even if a later comparison
or identity raises. No production Rust change is part of this harness work.

## Invariant order and routing

The manual defines invariant series at lines 2077–2105. For distance
d=1,...,floor(n/2), enumerate starts j=0,...,n−1 and append
`(p[(j+d) mod n]−p[j])²`; for the final distance n/2 of even n, enumerate
only j=0,...,n/2−1. This yields exactly n(n−1)/2 independent entries, starts
with the adjacent external squares, and reproduces the existing N2/N3/N4
packing. `TensorReduction.F90:1056–1073` independently reconstructs its pair
matrix from this ordering. This is index enumeration and numerical input
invariant evaluation, not a new symbolic algebra facility.

Before any higher-point qualification, compare the new TNten wrapper against
the unchanged Bten/Cten/Dten calls on every retained N2–N4 row. Compare raw
arrays and flags even where the old adapter rejects a lane. A wrapper change
must not erase the current N4 rejection. Add meaningful tests for even/odd
invariant ordering, complete Cartesian counts and invalid-domain rejection.

Keep zero-origin routing initially. Generic real four-vector shifts must now
be allowed because the retained conditioned high-point fixture contains a
nonzero time component. This is an explicit new domain, not a silent change
to the old all-spatial producer. Real timelike momenta alone do not establish
IR finiteness or absence of thresholds. For the initial conditioned positive
mass fixture, record an analytic positive Feynman-polynomial bound before
using the one-call IR-finite specialization. One sufficient bound is
`min(m_i²) − max(0,max_ij(p_i−p_j)²)/2 > 0`, since the pairwise Feynman weights
sum to at most1/2. Derive/check the needed scalar expressions with Symbolica;
do not generalize this check to complex masses. Keep delta-parameter pole
probes outside timing on every lane.

Use prefixes N5/N6/N7 of the existing
`validation/public-highpoint/fixture-conditioned_masses.json` as fixed
correctness witnesses. Preserve its base, scaled and mass-deformed rows and
exact rational input parsing. The retained N6/N7 IREGI captures already cover
those rows; generate any missing N5 capture from the same original reference.
For batch32 performance, preregister a distinct named family and its lane
construction; repeating or perturbing three existing rows must not be
misreported as32 independently validated kinematic branches. Input preparation
and master/reduction work remain distinct reporting boundaries.

## Qualification and later coverage

Run bounded correctness before requesting timing. Retain every Laurent slot
and every component, at the unchanged dimension-normalized1e−10 criterion.
Use Symbolica-prepared denominator-difference/routing identities and trace
identities with their explicit dimensional remainder. For massive N>=5 rank
<=3, the epsilon times six-dimensional scalar/vector remainder has no pole;
higher-rank tests need a fresh degree/evanescent analysis and must not reuse
the N4rank<=3 argument indiscriminately. Check UV/IR finiteness, scale changes,
permutations and the nonzero-time components independently.

Only rows that satisfy both the engine status gate and independent agreement
can enter a performance comparison. Other fixed rows stay visible with their
failure/unsupported status. A later timing producer must be frozen and
independently reviewed; use common warmed iteration counts, complete pre/post
checks, one pinned CPU, fixed balanced pairs and the established99% interval
and minimum-block gates. Do not time a failed N4 row by weakening its accuracy
gate or changing the fixture. Complex-mass and massless families follow as
separate correctness extensions, with full IR adapter witnesses before any
IR-divergent performance comparison.

All COLLIER source references above refer to version1.2.10, with archive,
manual and source hashes in `collier-1.2.10-manifest.json`. Upstream is GPLv3;
this optional reference use adds no Fortran or matrix algebra to the Rust
production implementation.

## Low-rank initialization diagnosis, 24 October 2026

The original design's `Init_cll(n,rank)` recommendation is unsafe for some
low-rank generic tensor calls in the pinned release. A separately bounded
diagnostic compiled only pristine `collier_tensors.F90` with bounds checking,
linked its object before the unchanged optimized library, and called the exact
retained N6/rank0 COLI singleton. It terminated with a bounds error at line5896:
`BinomTable(2,6)` accessed column6 when its upper bound was4. The checked compile
and link took12.109s and0.821s within60s; the diagnostic exited2 in0.0178s,
with zero result arrays and all source/input/module/library guards unchanged.
These are diagnostic durations, not kernel performance measurements.

Upstream `collier_init.F90:272` sizes the table through
`rmax+max(Nmax-2,4)`; `Aux/Combinatorics.F90:69` allocates exactly that bound.
The generic tensor entry point uses columnN in an automatic-array extent
before checking its arguments. This establishes a real out-of-bounds setup
defect. It does not establish which downstream unchecked write caused the
earlier heap-abort stack. The durable evidence is
`validation/collier-reference/shared/lowrank-bounds-001/qualification.json`.

A new reference-tooling candidate therefore reserves the minimum official
initialization capacity `max(rank,n-max(n-2,4))` for N>=5, leaving every
requested tensor rank, allocation, component count and reduction call unchanged.
N<=4 initialization and dedicated B/C/D APIs stay identical. The capacity is
recorded explicitly in responses; no upstream source patch or numerical
accuracy change is proposed. Qualification requires a new bounded capture,
the fixed interface regression gate, and an independent durable-package audit.

Two prior harness failures remain separate: attempt002 did not transport the
mixed-scale batch32 inputs, and attempt003 incorrectly required new interface
header metadata from the frozen legacy wrapper. All returned native arrays and
statuses are preserved, and the fixed gate blocked all higher-point requests
in attempt003. The fourth candidate checks each binary's actual header schema,
partitions unchanged input samples by exact scale, gathers lanes back to their
original order, and starts a fresh process for each engine/mode/scale request.
No timing or new qualified higher-point coverage follows from source readiness.
