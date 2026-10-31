# TIRust / iregi-rust: exhaustive session handoff

## Direct numerical engine: binary64 default (30 October 2026)

The user asked to optimize TIRust, which was slower than original IREGI and much
slower than COLLIER. The cause was architectural: per-call MPFR interval
certificates dominated low rank and Symbolica's interpreter of huge compiled
expressions dominated high rank (workspace construction also took 1-90 s). The
fix is a direct numerical engine, now the public binary64 default.

**Pushing:** plain `git push` has no stored credential on this host. `gh` is
logged in as huashengshao, so push with
`git -c credential.helper= -c credential.helper='!gh auth git-credential' push origin main`.

### What exists (crates/iregi-rust/src/direct*, dispatch/direct_hybrid.rs)

- `direct/plan.rs`: structural plan, no sampled values. Subsets of the N<=6
  propagators (bitmasks); PV subsets (<=4) get Denner-Dittmaier coefficient
  instructions mirroring the validated `pv.rs`; pentagons use the 4D Gram/Cayley
  moment recursion; hexagons the thin identity `l^mu = sum_i w_i^mu D_i`.
  Gathers turn PV coefficients into Cartesian moments: sparse momentum products,
  `g_lambda` blocks, a closed metric expansion (`MonomialTable::qexp`) and
  separable 1-D binomial translations along nonzero origin axes.
  `with_relations(settings, masks)` builds variants in which subsets containing
  an affinely dependent minimal subset reduce by `sum_i w_i D_i = 1` at the same
  rank. One raised power (<=5) gives Taylor jets of length `K = power`; `h<=3`
  (N<=4) reads `C(lambda+h,kappa)` scaled by `(-2)^h` in every gather.
- `direct/exec.rs`: per-lane executor generic over `NativeFloat`. Masters come
  from `oneloop::NativeEvaluator` (hand-written binary64 C0/D0 sectors) per lane,
  UV slots set exactly (A0 m^2, B0 1, C0/D0 0, IR1 = total - UV); a
  `DirectMasters::Policy` batched `MasterEvaluator` path also exists. Geometry
  solves small systems with the pivoted kernel in `direct/numeric.rs` (reuse
  checks are in its module doc). Ill-conditioned Gram systems trigger an affine
  relation search (complete pivoting on the coordinate matrix, residual and
  constant checks); success switches the lane to a cached plan variant, failure
  reports `Unstable`. Jets: geometry jets (m0^2, f, zf, pentagon `b` via
  `(R0^-1 E)^k b0`, hexagon weights, relation weights `w0(-t)^k`), master Taylor
  coefficients by a numeric port of hep-ph/0303184 Eq. (19) (`ibp()`), jet
  convolution in `run_program_jets`, per-order gathers. Output applies the UV
  power counting, IregiNative/OneLoop/MadLoop conventions and multinomial weights.
- `dispatch/direct_hybrid.rs` + `dispatch.rs`: `TensorReductionPlan::new` picks
  `OrdinaryPlan::Direct` when `DirectReductionPlan::supports` (unit or one raised
  power, `h` 0..3 with N<=4 for h>0, FromShifts, Preserve, PV/Hybrid, N<=6,
  rank<=8). Binary64 workspaces use the direct engine; `Unstable`/`MasterFailure`
  lanes rerun on the lazily prepared symbolic plan (`TensorReductionPlan::new_without_direct`,
  the old dispatch, also public), after the direct-engine precision rescue
  (item 3 below). Float/DoubleFloat/Dual workspaces and `PrecisionRescueWorkspace`
  use the symbolic plan. `TensorEngineKind::Direct`,
  `TensorEngineExecution::Direct(DirectExecutionStats)`.

### Validation

- `tests/direct_engine.rs`: all 18 opening rows against the archived reference
  tensors (E/A <= 2e-14 for N<=4 ordinary, <= 2.4e-11 N5/N6, <= 1.1e-13 dotted/h1),
  perturbed batches against the existing engine with an isolated invalid lane,
  and the coplanar-box denominator identity against an independent triangle sum.
- Full lib suite: 483 pass; `certified_scalar_batch_matches_coordinate_interface_and_isolates_failures`
  and `certified_scalar_rejects_an_incompatible_mixed_taylor_component` fail, and
  already failed at the pre-session commit a35690b (verified with the same build).
  All integration suites pass. Five dispatch tests that probe old-engine
  internals now construct `new_without_direct`.

### Same-session benchmark (validation/benchmark/direct-engine-20261030)

Interleaved, one pinned core (CPU 199), fresh process per candidate, 11/7 rounds,
paired ratios with 99% bootstrap CI. TIRust is faster than IREGI and COLLIER on
every row, batch 1 and 32. Batch-1 medians (µs): N1R0 0.27 (IREGI 0.77, COLLIER
0.39), N1R6 2.75 (23.1, 4.38), N2R6 4.94 (383, 8.42), N3R2 5.14 (24.9, 8.32),
N3R6 11.4 (2707, 51.5), N4R4 22.2 (529, 49.6), N4R6 32.0 (12494, 264), N5R5 64.7
(8687, 304), N5R6 87.3 (46937, 1059), N6R6 216 (199510; COLLIER nonfinite),
N4R6 dot2 79.0 (IREGI 14977), dot3 107 (18840). Report: new first section of
`docs/benchmark.pdf`; `publish.py` appended three results.json records and
resealed `docs/data/report-validation.json` (the previous proof was stale for
HANDOFF.md at a35690b).

Build cost: release (thin LTO, cgu=1) of iregi-rust takes ~6 min; use
`cargo build --profile devopt` (Cargo.toml, lto off, cgu 16, incremental; ~75 s)
for iteration and `examples/direct_check.rs` with the float fixture produced by
converting `opening-rank6-20261026` (see that example's doc comment). Use
`source scripts/env.sh` (stack size, toolchains) before running test binaries.

### Done after the first campaign (commits 394f1f3, 3f5aff2, c4a72e7)

- **Lightlike bubbles / massless external legs**: bubbles with `|p^2|` below
  `2^-40` of their scale become `lightlike` plan-variant subsets whose PV
  coefficients are closed forms (`lightlike_coefficients`, Feynman-parameter
  `C(lambda,k) = (-1)^k/(2^lambda lambda!) [I0/eps + H_lambda I0 - I1]`, series or
  recursion for `I1`, exact scaleless/equal/massless-line limits; `m1^2` taken
  directly, not `a + b`). PV Gram conditioning is also scale-relative now.
  Validated: symbolic engine (bubbles, ggH triangle), continuity at m^2=1e-20
  (the symbolic engine refuses one-massless bubbles), and original IREGI for the
  massless triangle, IR-divergent massless boxes, ggHH and mixed boxes
  (`validation/benchmark/direct-engine-20261030/lightlike-check.{py,json}`,
  <= 5e-14 vs IREGI IBP). Dotted propagators with lightlike bubbles still fall back.
- **Hand-written negative-Källén C0** in vendored oneloopmaster
  (`handwritten/triangle.rs::hooft_veltman`, PATCHES.md, `examples/c0_hv_check.rs`:
  80k points within 2e-12 of the generated code, both within 3e-12 of 128 bits;
  1.7 vs 3.2 µs).
- **IR-free two-slot kernels** (`ir_free`, `lfma_slots::<FULL>`) and a specialized
  Taylor-length-2 recurrence (`run_program_jet2`).
- Dev-build timings now (µs, CPU 199): N3R2 3.75, N4R6 23.6, N5R6 67.7,
  N6R6 170, dotted N4R6 42/103 (dot2/dot3). The report still shows the first
  campaign (capture001); a re-run is the next step.

### Next steps (in order of value)

1. Done: capture002 (TIRust 9979308, const-generic jet kernels K=2..5 included)
   is bound as `direct_engine_20261030_capture002` and shown in the report;
   capture001 stays bound. Batch-1 µs: N3R2 3.68, N4R6 23.3, N5R6 66.9, N6R6 166,
   dotted N4R6 42.0/66.3; TIRust/COLLIER 0.06-0.68, TIRust/IREGI 0.0008-0.36.
   For a later capture use `build.py build-NNN.json`, `run.py --build-record`,
   `summarize.py ... build-NNN.json`, extend `publish.py LATER`, then
   `publish.py bind/seal --capture NNN`.
2. Lightlike bubbles with jets (closed-form mass derivatives); currently those
   lanes fall back.
3. **Done: direct-engine precision rescue** (`dispatch/direct_hybrid.rs`).
   Lanes the binary64 direct engine refuses on conditioning (Gram, Cayley,
   six-point, jet Cayley) carry the refused estimate structurally
   (`DirectWorkspace::last_conditions`; exactly singular pivots report 0). The
   hybrid reruns them with the same direct plan at 106 bits (`DoubleFloat`),
   256 or 512 bits (`Float`), starting at the first tier whose acceptance
   threshold `2^-(bits-40)` admits the binary64 estimate with 8 bits margin
   (singular pivots start at 256) and climbing while the refusal is again a
   conditioning one. Inputs are lifted exactly; accepted lanes are rounded once
   and marked `fallback_used`. Only the remaining lanes (and master failures)
   go to the symbolic engine. `DirectExecutionStats` gained `precision_rescue`
   (`DirectRescueTier{bits, rows, successes}`) and `symbolic_rows`;
   `fallback_rows` now counts every lane rerun after the binary64 pass. CLI JSON,
   Python dict/stubs and `examples/direct_timing.rs` (per-lane diagnostics)
   expose them. Test: `near_degenerate_boxes_are_rescued_at_higher_precision`
   (106/256/512-bit tiers, E/A <= 1e-14 against a 512-bit direct evaluation).
   Evidence: `validation/benchmark/direct-engine-20261030/rescue-coplanar.{py,jsonl}`
   (release 877cf47) against quad IREGI: wherever quad IBP and PV agree the
   rescued arrays match them (rank 4: 2-3e-17 at delta 1/64, 1/256; within the
   quad disagreement at 1/1024, 1/4096), rank 6 matches quad IBP to 2-8e-17 at
   1/8, 1/64 (quad PV is wrong for rank > N; quad IBP degrades below 1/256).
   Binary64 IREGI IBP is wrong by up to 1e18 on these points while reporting
   success. At delta = 2^-24 IREGI treats the box as exactly coplanar (error
   O(delta) = 2e-7); TIRust evaluates the given kinematics.
   **N >= 5 loop-rotation test (877cf47).** The binary64 stability study
   (`validation/benchmark/stability-20261030`) showed the pentagon Gram
   recursion accepting physical gg->Hgg lanes with E/A up to 3e-4 (rounding
   ~ kappa^rank u, kappa = ||Z|| ||Z^-1|| ~ 60..6e5 for physical 2->3 points;
   the Hadamard-relative determinant does not track it). Now the pentagon gate
   uses 2^-7/kappa^rank (removes hopeless lanes) and the binary64 hybrid
   evaluates every N >= 5 lane twice (propagators rotated) and sends lanes whose
   arrays differ by more than 2^-30 to the rescue. Consequence: on physical
   pentagons/hexagons nearly every binary64 lane is rescued (correct, but
   ms per lane). **Next big item: a Gram-free pentagon reduction** (Denner-
   Dittmaier, hep-ph/0212259, as in COLLIER) so physical N=5/6 stays in binary64. Cost per rescued lane (dev build): ~8 ms rank 4 at 256
   bits, ~15 ms rank 6 at 512 bits (`examples/direct_tier_profile.rs`).
   The precision tiers were made affordable by a **native arbitrary-precision
   complex dilogarithm in oneloopmaster** (`src/native/dilog.rs`, PATCHES.md):
   Symbolica's generic `polylog(2,z)` cost 0.35-1 ms per call and made a 256-bit
   box 40 ms of masters; the port of Symbolica's binary64 Bernoulli-series kernel
   with exact tangent-number coefficients agrees with it to working precision
   (636 points, 106-512 bits) and costs 15-73 µs.
   Background (kept): binary64 IREGI IBP is wrong by 1.7e-4 at delta=1/64 and
   2.5e6 at delta=1/1024 on the coplanar box while reporting success
   (`quad-coplanar.{py,jsonl}`); TIRust used to refuse these lanes honestly.
   Long term: Gram expansions as in COLLIER would avoid the extended precision.
3b. **Binary64 stability study** (`validation/benchmark/stability-20261030`, report
   section "Binary64 stability"): cumulative E/A profiles as in Hirschi-Peraro
   arXiv:1604.01363 figs. 2-4 for top-loop gg->HH boxes (rank 4 and 6, 10k
   points each), a gg->Hgg pentagon (rank 5, 5k) and a gg->gggg hexagon (rank 6,
   2k) at sqrt(s)=1 TeV, pT>50 GeV, dR>0.5. Codes run as a user would:
   TIRust default and binary64-only, IREGI IBP/PV with its default on-shell
   snapping (the timing driver disables it), COLLIER COLI/DD with exact zero
   massless invariants (with rounded ~1e-10 invariants COLI returns NaN and DD
   wrong arrays on most pentagons/hexagons). Reference: TIRust direct engine at
   256 bits, confirmed per point by quad IREGI (<=1e-9; typically 6.3e-17,
   which is quad IREGI's binary64 pi in its normalization). Quad IREGI alone
   is unusable: quad PV fails on rank > N boxes, quad IBP on hexagons, and quad
   PV keeps only 1e-12..1e-10 on some hexagons (COLLIER sides with 256-bit).
   Pitfalls fixed on the way: serde_json's default float parser is not
   correctly rounded (pass decimal strings, parse with str::parse); the quad
   build reads decimals in quad, so give it exact expansions of binary64.
   Rescue cost on physical points (release, 877cf47): box r4 24 us, r6 49 us
   (no rescue); pentagon r5 6-9 ms (106 bits); hexagon r6 150-170 ms (256 bits).
   **Results (TIRust 7226962, report pages 4-5, bound as `stability_20261030`):**
   TIRust default returns an array on every point; worst E/A 9.6e-9 (box r4),
   3.4e-10 (box r6), 2.4e-10 (pentagon), 1e-16 (hexagon); binary64 alone keeps
   99.7% of r4 boxes, 96% of r6 boxes, 31/5000 pentagons, 0 hexagons. COLLIER
   COLI/DD: worst 6e-11..1.3e-9 everywhere. IREGI PV wrong on all r6 boxes,
   otherwise mostly fine (tails to 1e-1..3e3); IREGI IBP: 268/5000 pentagons
   and 1987/1994 hexagons worse than 1e-3. Excluded: 6/2000 hexagons (no
   confirmed reference). The first capture (877cf47; 7 pentagons where a
   nonfinite binary64 coefficient bypassed the rescue, fixed in 7226962)
   stays under target/stability-20261030/capture-877cf47.
   **Capture 002 (TIRust 3d3d152: Gram-free pentagon + native DD dilog),
   shown in the report, bound as `stability_20261030_capture002`:** binary64
   keeps 1069/5000 pentagons (was 31) with worst E/A 1.8e-9; pentagon rescues
   mostly at 106 bits (3636 vs 295 at 256; was 3038/1931), hexagons 1564 at
   106 / 430 at 256 (was 410/1584); boxes unchanged.
3c. **Done (first part): Gram-free pentagons.** `plan.rs` gives every
   unjetted pentagon whose box children are not degenerate a coefficient
   layout E(lambda, kappa) over its four momenta up to rank min(r, 5), built by
   Denner-Dittmaier hep-ph/0509141 Eqs. (5.9)-(5.10) symmetrized as (5.12)
   from the five boxes' PV coefficients (rank <= 5): only X^-1 of the modified
   Cayley matrix X = [[2m0^2, f^T], [f, Z]] (one pivoted 5x5 solve per lane,
   gated on its relative determinant) and Xtt_(kn)(0i)/det X = products of
   X^-1 entries. D(0) is rewritten in the pentagon basis by the paper's shift
   recursion; the rank-5 UV remainder (D_0000(0) + 1/48 in E_{0000k}) uses a
   unit finite value. A standard gather (4 momenta) produces the moments;
   ranks 6-8 continue with the Gram recursion (kappa^(r-5) instead of kappa^r).
   Opening N5/N6 rows unchanged within 1.5e-14; tests
   `gram_free_pentagon_is_stable_in_binary64` and the existing suites pass.
   Pilot (50 physical gg->Hgg pentagons): binary64 keeps 9 (was 1) at E/A
   1e-12..2e-10. Cost (release, CPU 199, one lane, includes the rotation
   test): N5R5 136 us, N5R6 200, N6R6 526 (877cf47: 117/162/390; capture002
   before the rotation test: 65/87/216). **The limit is now the boxes:** the
   remaining refusals are box PV Gram systems at coefficient rank 5 (and
   rotation-test failures from inaccurate box coefficients). Next: small-Gram
   box reduction (hep-ph/0509141 sects. 3.4-3.6, COLLIER's expansions) or the
   Cayley-based reduction of sect. 3.3 (needs numerically integrated
   D_{00..0}); then the same for the hexagon's sub-boxes.
   Native double-double dilog (0769821): the 106-bit rescue of a physical
   pentagon costs 4.8 ms (was 8.2); a hexagon handled at 106 bits ~21 ms.
3e. **31 October 2026 session (commits a5cd37c, 514e3cf, bc90c69; read first).**
   The small-Gram expansion (3d) was started and dropped: on refused physical
   pentagon boxes |Delta|/max|Ymodadj_0j| has median 0.24 (>1 for 10%), so
   sect. 3.4 converges too slowly. Root cause of the inaccurate binary64
   N=5/6 lanes instead: (i) oneloopmaster's generated B0 cancels like m^2/p^2
   for tiny nonzero p^2 (gluon legs p^2 ~ 1e-10 vs m_t^2: 5.8e-3 error), fixed
   by `dependencies/oneloopmaster/src/native/small_momentum.rs` + iregi-masters
   central repair (PATCHES.md "Small-momentum B0"); (ii) generated D0 NaN at
   such p^2: the direct engine now snaps squared distances within 8u of their
   own rounding to exact zero (`read_lane`); (iii) rotation-tested plans
   (N>=5) gate binary64 at 1e-15 (was 1e-12), rotation tolerance 2^-34.
   Probes: examples/subset_accuracy.rs (per-subset E/A, THRESH env),
   master_probe.rs, n7_probe.rs. 500 pentagons: binary64 keeps 457 (was ~21%).
   N=7 direct engine (thin identity over the best 6 of 7 members; unit powers);
   test `seven_point_rank_seven_heptagon_matches_high_precision`.
   Stability study: heptagon-r7 (500 pts, appended), COLLIER driver N,R<=7,
   NinjaRS + contracted random-numerator metric in run.py/summarize.py;
   capture 003 = target/stability-20261030/capture003 (build-003.json).
   NinjaRS reference build: target/reference/ninjars (source = upstream
   41a4c00 + branch serde-json-float-roundtrip 16bd20a; bin/ has both
   binaries). Upstream PR branches prepared in the scratchpad clone
   (oneloopmaster: native-arbitrary-precision-dilog, small-momentum-b0; PR
   texts pr-*.md); pre-existing upstream failures: oneloopmaster
   `euclidean_c0_preserves_primal_and_zero_seeds` (+4 C0 integration tests to
   recheck on main), NinjaRS `templates_do_not_depend_on_process_history`
   (libtest output parsing). MRE: validation/mre/collier-dd-small-invariants
   (DD wrong by 0.30 with p^2 ~ 1e-10). Pending: IREGI MREs (PV rank>N box
   r6 #0; IBP hexagon #0, pentagon #27), COLLIER-style regulators (stash
   "regulators WIP": src/regulators.rs, TensorWorkspace::set_regulators;
   verify the pi^2/6 Collier normalization against COLLIER's massless box
   D0: finite 0.50481+0.08492i, dIR1 0.13433-0.20944i, dIR2 -0.13333 at
   s=10, t=-3, mu^2=2), timing capture 003 with NinjaRS (case-bench) and an
   N7R7 row, report update.
   Status at 08:30Z: COLLIER-style regulators done (ce84d10, test against
   COLLIER's massless box). Stability capture 003 done and summarized
   (summary-003.json, b556032): pentagons 86% binary64, hexagons 24%,
   heptagons rescued (next optimization target); heptagon references by
   COLLIER COLI+DD (quad IREGI fails there). Timing capture 003 running
   (target/direct-engine-20261030/capture003, CPU 188, build-003.json at
   ba4aa16; NinjaRS via case-bench, generic engine on the Euclidean opening
   rows; N7R7 row q_6=(1/4,3/8,1/8,5/8)). Report code for both captures is in
   docs/benchmark.typ (uncommitted until sealed). Then: summarize timing
   (summarize.py capture003 summary-003.json build-003.json), publish.py
   bind --capture 003 in both dirs, typst compile --root . --creation-timestamp
   1793318400 (twice, cmp), publish.py seal ... --capture 003, check, push.
   NinjaRS PR https://github.com/alphal00p/ninjars/pull/11 (float_roundtrip);
   branch determinism-child-output (c636f94, worktree
   target/reference/ninjars/source-determinism) to test and open. oneloopmaster
   PRs NOT opened: dilog branch fails upstream test native_256_bits_matches_oneloop
   (results with ~10-bit precision at 256 bits under symbolica-dev f514a7f:
   argument precision from cancellation tracking?); B0 branch fails the same
   43-line set as dilog minus that test; rerun pristine main suite
   (scratchpad upstream/main-full.sh lacked env) to confirm pre-existing.
   Report sealed with both captures 003 (78 pages; stability section from p. 6,
   contracted/NinjaRS figure p. 7; timing tables incl. NinjaRS after the
   direct-engine section). Timing capture 003: TIRust faster than IREGI on all
   rows and than COLLIER on all ordinary rows; against NinjaRS (TIRust array +
   contraction) 1.5x slower on N1R0, ~1 on N2R1 (NinjaRS fast engine), 0.02-0.2
   on N3-N5 (NinjaRS generic engine: opening points have Euclidean cut planes),
   0.40 N6R6 and 0.57-0.76 N7R7 (NinjaRS lane-0 CutFailed there, no accepted
   value). TIRust N5R5/N6R6 slower than capture 002 (rotation test + Gram-free
   pentagon). Timing inputs for capture 003: inputs-003.json (inputs.json stays
   bound).
   Heptagon speed-up (next): on 30 physical heptagons every binary64 lane is
   refused by some box/triangle Gram gate (35 boxes + 35 triangles per lane,
   conditioning 1e-3..1e-4; even with gate 1e-15). Gate open: E/A 1e-13..1e-7
   (16/29 <= 1e-10). Plan A (preferred): partial rescue, i.e. evaluate only
   the refused subsets (their coefficient programs and masters) at 106 bits
   and feed the rounded coefficients into the binary64 program (thin
   identities and the Gram-free pentagon above them are stable). Plan B:
   looser gate for N=7 relying on the rotation test; first measure whether the
   rotation difference tracks the true error (it cannot see master-error
   amplification, as both evaluations share the masters).
   oneloopmaster dilog fix: Float::dilog lifts both argument parts to the
   working precision (bb6ee20; upstream branch d926e91); found by upstream's
   native_256_bits_matches_oneloop (C0 arguments with a 2-bit cancelled real
   part).
   Upstream PRs (opened 31 Oct): oneloopmaster #13 native dilog (branch
   native-arbitrary-precision-dilog, d926e91) and #14 small-momentum B0
   (small-momentum-b0, 956c1bb); NinjaRS #11 serde_json float_roundtrip and
   #12 determinism-test parsing. Upstream main 7b3f43c has 33 failing tests in
   this environment (cargo test --no-fail-fast --features generated-evaluators);
   both oneloopmaster branches add none.
   Heptagon rescue made cheaper (commit after 616d63c): the 106-bit tier uses
   20 bits of headroom (RESCUE_DD_HEADROOM) instead of 40, chosen first whenever
   the binary64 estimate allows it, and is checked a posteriori by a rotated
   106-bit evaluation (RESCUE_ROTATION_TOLERANCE 2^-46 on binary64-rounded
   arrays; failures escalate to 256). Evidence: the 106-bit engine without any
   gate agreed with 256 bits to <= 1.7e-23 on 30 heptagons. 40 study points
   each: heptagons 38 accepted at 106 (2 at 256), worst E/A 3e-24; hexagons and
   pentagons unchanged in accuracy. Heptagon #0: 121 ms instead of 644 ms. Next:
   profile the 106-bit heptagon (~60 ms per evaluation) and the partial rescue
   (plan A above).
   Partial rescue implemented (plan A): binary64 executors record boxes and
   triangles refused by their Gram gate (not the top subset, no jets/shifts),
   compute their coefficients with cached double-double executors of their own
   plans (DirectExecutor::inject_external, plan.coefficient_ranges/counts) and
   skip their instructions. 40 study points each: full rescues heptagon 40->6,
   hexagon 31->0, pentagon 3->0 (box r6 unchanged 3: top box), worst E/A
   3.4e-11..5.8e-11 (binary64 level, within the rotation tolerance); heptagon
   #0 20.6 ms (was 121 ms DD-first, 644 ms at 256 bits). Report/PDF not yet
   recaptured with this (next capture 004).
   Engine gaps found: exactly singular six-point thin systems and null-
   hyperplane boxes (singular Gram, affinely independent) have no route.
3f. **Binary64-only path and the rescue switch (31 October, after a43a524).**
   User requests: improve double-precision-only TIRust; add an option to
   disable the precision rescue. `PrecisionRescue::{Automatic, Disabled}` in
   `TensorWorkspaceOptions::precision_rescue` and
   `TensorWorkspace::set_precision_rescue` (CLI `--no-precision-rescue`,
   Python `set_precision_rescue(False)`): Disabled = no partial DD, no tiers;
   lanes with a conditioning estimate are returned Unstable ("precision rescue
   disabled" note); lanes without one still go to the binary64 symbolic
   engine. Checked binary64 retry (`direct_hybrid.rs`, both modes, N>=4 with
   equal powers): lanes refused by the a priori gate (or, N>=5, whose rotated
   copy was) are evaluated again with the gate at CHECKED_GATE = 1e-30 and kept
   when the rotated copy agrees within tolerance/4 (CHECKED_MARGIN_BITS = 2).
   The rotation tolerance is the accuracy request 2^(19-accuracy_bits)
   (default 2^-34). Probe (`examples/binary64_probe.rs`, all stability points;
   outputs in the session scratchpad): binary64-only kept box r4 9966 ->
   9976/10000, r6 9630 -> 9744, pentagon 4318 -> 4556/5000, hexagon 488 ->
   1519/2000, heptagon 0 -> 46/100; worst kept E/A unchanged (9.6e-9, 3.4e-10,
   2.7e-10, 1.5e-10, 2e-11). At the 2^-20 request: 9988, 9927, 4972, 1965,
   93/100, worst 2.5e-6. Default path: full rescues r6 370 -> 256, pentagon 38,
   hexagon 34, heptagon 13/100 (with partial rescue). Most remaining refusals
   are accurate to 1e-8 (open gate); genuinely lost precision: about 20 r4
   and 120 r6 top boxes (small Gram: |det Z|/max|Xadj_0j| median 3e-3 and
   2.7e-2, smallest on the worst lanes, so sect. 3.4 should converge fast
   there; COLLIER's CalcDg in target/reference/collier/source/.../reductionD.F90).
   Stability driver records `binary64` / `binary64_r20` (facade, rescue
   disabled); run.py/summarize.py use them from capture 005 (codes
   tirust-binary64, tirust-binary64-r20); capture 004's "binary64" curve is
   the old derivation (default without a full rescue, partial DD included).
3g. **Small-Gram box expansion implemented (31 October, after 8d7af86).**
   `direct/exec/gram.rs`: Denner-Dittmaier eqs. (recurGram1)/(recurGram2)
   (arXiv source read; COLLIER CalcDg as cross-check) in the engine's
   C(lambda, kappa) normalization with the (D-4) term kept (div_dimension,
   a = 2(lambda + |kappa|)); second adjugate = -eps eps Z (sign fixed by the
   paper's identities, unit test). Auxiliary 4-point executor per box rank:
   plan `with_top_child_boost(settings, boost)` (triangles boost ranks
   higher, top box at its own rank <= MAX_RANK), `gram_top` mode skips the
   box Gram system and the gathers, fills the box by passes t = 0..G (Delta
   terms from pass t-1; D0 = master), stops on growth, refuses if the last
   relative change > 2^-30. Parent executors with `enable_gram_expansion`
   mark gate-refused boxes (any position) and inject (like partial rescue).
   Hybrid: second binary64 retry after the checked one (gate as main, boost
   GRAM_BOOST = 8), kept by the rotation test at tolerance/4; stats
   expansion_rows/successes. Gate-refused stability boxes: converged 22/34
   (r4) and 239/370 (r6, saturates at boost 8), every converged lane <= 3.4e-11
   against 256 bits. Binary64-only kept (probe, scratchpad probe64c): r4
   9995, r6 9976, pentagon 4694, hexagon 1569 (of 2000), heptagon 46/100;
   default full rescues r4 5, r6 24. Coplanar test box: delta 1/64, 1/1024
   now binary64 (E/A 6e-14, 3e-14), 2^-24 still 512 bits.
   Published: capture 004 (timing + stability, a43a524; commit 50acff0) and
   stability capture 005 (binary64-only facade at 2^-34 and 2^-20, build
   50acff0 = code of ffb2716; commit 193415e), report pages 1-5 timing, 6-8
   stability. Capture 005 matches the probe numbers above.
3d. **Next: small-Gram box reduction** (hep-ph/0509141 sect. 3.4, Eqs.
   (recurGram1)/(recurGram2), arXiv source in the paper; COLLIER's scheme).
   For a box with momenta p1..p3: pick j maximizing |Xadj_0j| and (k,l)
   maximizing |Zadj_kl| per lane (fold the choice into geometry entries, the
   instructions stay structural). Level 0: all box coefficients up to rank
   R_max = r + K with every Delta^(3) term dropped; level t recomputes ranks
   <= R_max - t using level t-1 values in the Delta terms (separate value
   slots per level); the requested rank r comes from level K, accurate to
   O(Delta^(K+1)). Needs triangle coefficients up to R_max + 2 (S^(P+2)),
   S^P_{00...} = 2 C(0) + 2 m0^2 D, Shat = S without f D, the (D-4) terms via
   PostOp::DivideDimension. Select per lane by a plan variant (as relation /
   lightlike variants): boxes whose Gram gate fails switch to the expansion
   when |Xadj_0j| is not small; both small -> rescue as now. Validate box
   by box against the 256-bit engine on the stability points, then rerun
   the study (expect the pentagon/hexagon binary64 refusals to drop).
   Original plan (kept):  Gram-free 5- and 6-point reduction so physical N=5/6 lanes stay in
   binary64. Denner-Dittmaier hep-ph/0212259 Eq. (2.17)/(2.19)
   (E_P = -sum_i detY_i/detY D_P(i) + 2 sum_ij Zhat_ij/detY p_j^a Delta_aP(i)
   + UV/detY, P<=4) and, for any rank in dim. reg., hep-ph/0509141 sect. 5
   (rank P from 4-point rank P-1; sect. 6 for 6-point; sects. 3.4-3.6 are
   COLLIER's small-Gram expansions for boxes/triangles). Only det Y (5x5
   modified Cayley) is inverted. Needed in plan.rs: a pentagon subset kind
   whose instructions combine the box subsets' moments and their g-part
   (metric) coefficients; validate against the 256-bit direct engine and
   COLLIER on the stability points, then rerun the stability study.
4. Remaining cost centres: masters (log/atan2/dilog in oneloopmaster kernels), the
   recurrence loop (a real-factor fast path gave nothing measurable; the loop is
   load/latency bound, so try SoA L4 layout), gathers.
5. Batch-major (SIMD across lanes) execution if batch throughput matters more
   than singleton latency.

### Quota protocol (user instruction, also in Claude memory)

Check `mcp__ccd_session_mgmt__get_usage` about every 10 minutes. Below 5% of the
5-hour window left: pause subagents with handoffs, keep PIDs of long jobs, wake
after the reset. Below 3% of the weekly window: permanent stop with a complete
handoff here.

## Binary64 opening tables: 26 October 2026 follow-up

The current opening display supersedes the previous mixed-precision tables.
Both tables now use a separate all-binary64 campaign under
`validation/benchmark/opening-double-20261026/`. All 18 mathematical inputs
are unchanged from `opening-rank6-20261026`; every TIRust candidate uses Rust
`f64` (53 significant bits), with no higher-precision rescue. The three ordinary
N5R5/N5R6/N6R6 requests use accuracy parameter39: the existing 19-bit margin
corresponds to `2^-20 ≈ 9.54e-7`, not `2^-39`. All other TIRust rows retain53
(`2^-34` target). COLLIER requests `1e-6` for those three and `1e-10` otherwise;
every COLLIER configuration uses binary64, with critical accuracy `1e-3`.
COLLIER's DD label denotes the Denner–Dittmaier algorithm, not double-double.

The user explicitly prioritizes optimistic observed timings in these two
tables. Select the fastest mean within a fixed public-option portfolio that
returns a complete finite tensor with measured array-relative error <=`1e-3`.
Missing an older `1e-10` gate, a current `1e-6` request, or an internal accuracy
flag alone does not reject a timing. If no candidate produces a usable array,
retain the observed duration as diagnostic with the nonfinite/refusal status.
Do not present a quick refusal as a successful reduction. Do not change the
earlier raw records or historical strict admission decisions to implement this.

Accuracy is a true dimensionally rescaled whole-array max-norm discrepancy:
`E/A`, where `E=max_j |new_j-ref_j|/w_j`,
`A=max_j |ref_j|/w_j`, and
`w_j=mu^(degree_j+4+2h-2sum(powers))`. Every cumulative Cartesian moment and all
four complex Laurent slots participate. This avoids dividing by individual
symmetry-zero entries. The reference is the prior saved finite-precision tensor
corpus, independently admitted against original IREGI in all8,968 complex fields;
it is not an exact-integral oracle. The old componentwise mixed metric remains
in raw audits. If coefficients are nonfinite, any displayed finite-subset
discrepancy cannot certify the incomplete array. Withheld TIRust coefficients
remain null and have accuracy unavailable.

The new TIRust harness is `crates/iregi-rust/examples/opening_double.rs`.
Its38 predeclared candidates compare explicit NativeF64Abc and Symbolica master
policies for each row, plus existing thin Hybrid/PV for N6R6. The native policy
only specializes its qualified A/B/C families; other masters retain their
existing supported implementation. No scalar/reduction arithmetic, accuracy
guard or input convention is changed. The minimal core change exposes existing
thin six-point profiling after construction, preserving the selected engine.
Hybrid stage counters sum primary and actual fallback work once, with a single
outer wall timer; lazy fallback workspaces inherit the profiler setting. This
allows the measured split to include the actual thin engine rather than forcing
a different constructor path. Static independent source review checked all
three profiler files.

TIRust release binary SHA256 is
`6e07608750416c0a2efae25f04166c416569fd41239c84f8cc0a54ab86cb7c1f`.
The feature-enabled Cargo build took654.379s, exited0 with clean process-group
cleanup, and preserved all215 selected source hashes. This is selected build
context, not a claim of a complete hermetic closure. Raw build logs/receipts
are retained; no numerical binaries are committed.

The COLLIER wrapper was linked with strict `-O3`, no fast-math, against the same
unmodified COLLIER1.2.10 archive and28 module hashes used previously. New wrapper
SHA256 is `0c1cc996fdbbb2a21ad33eda9a84172f27639802cc35cbf53f0c049b009c76d0`.
Thirty-two fixed public configurations explore COLI/DD where supported,
argument permutation for high-point rows, and N6 direct-tensor thresholds6/7.
N5R6 DD is excluded because unsupported. Argument permutation disables local
cache use through the public API; all other candidates reset local memoization
with InitEvent for every root and disable event-result caches. Timing includes
the public call, invariants, masters, normalization and Cartesian gathering;
diagnostic/finite scans and pole probes are outside timing.

All32 COLLIER candidates completed timing. Capture took16.652s on CPU18,
exited0 and passed every binary/library/module/source guard. Its saved-data
independent Decimal100 replay passed20,991 checks. Eleven selected rows return
complete finite arrays within three relative digits. N5R5 is293.291µs with
error1.10084e-11; N5R6 is1051.821µs with error4.26744e-11. These are ordinary
timing rows, with no rejection based on native flags. N6R6 remains incomplete
under all eight tested configurations: the fastest observed diagnostic is
2819.232µs, with195 nonfinite complex fields and finite-subset discrepancy
about1.70509. Its time stays visible with a warning; this is not a usable result.
Other failed/partial configurations remain fully archived. No comparative speed
gate or hardware-isolation claim follows from selecting optimistic means.

Reproduction and publication scripts live with the new campaign. `run.py`
builds/freezes/supervises TIRust; `collier/run.py` takes an explicitly frozen
binary; independent checkers evaluate saved data only. `combine.py` joins the
retained `tirust-summary.json` with the audited COLLIER summary for the report.
`publish.py bind` appends new records while preserving all469 previous raw JSON
values exactly; `seal` keeps the entire previous78193d6 proof/history, checks
the manifest and distinct byte-identical PDF rebuild, and refreshes report
bindings. Finish all campaign files before binding its manifest. Root PDF review
belongs in `docs/data/`, outside the sealed campaign inventory. The prior
sections below are historical and retain their original precision and stricter
classifications; they do not describe the current opening display.

Final new TIRust capture takes1031.750s and all38 initial children exit0 with
all215 source/binary guards unchanged. The initial saved-data audit checks
32,512 complex fields across68 successful first/final arrays, plus eight honest
null refusal payloads from four candidate attempts. Sixteen of18 selected input
rows return complete finite tensors; maximum selected relative discrepancy is
2.45939e-11 (N5R6). N6R6 selects the measured Symbolica/AffinePower path at
419.120ms with error1.23023e-11. The actual thin Hybrid/PV profiles are retained
as alternatives; they were slower on this input. No numerical path is forced
solely by a native/interpreted distinction.

The two N4R6 generalized dot2/dot3 inputs hit an operand-dependent recurrence
cancellation estimate at request53. This is an internal admission estimate,
not independently measured output error. Failed lanes expose no coefficients.
Four additional public PV trials permute complete denominator triples by
`[1,0,2,3]`, leaving the loop variable, numerator and integral exactly unchanged.
They use the same frozen binary, working f64 and request53. Their separate
179.061s capture also exits0 with all guards intact, but all four still refuse.
This post-hoc supplement is explicit and independently checked, not retroactively
declared part of the fixed initial38 portfolio. The report selects the fastest
observed diagnostic for each of these two rows: about26.117ms and25.511ms,
respectively. They are labelled “Refused output · diagnostic timing” with
accuracy unavailable. Do not treat those durations as successful reductions or
substitute the old Float128 values in these binary64 cells. Higher-precision
functional qualification remains the separate earlier evidence.

`tirust-summary.json` and `qualification.json` retain the initial38 scope.
`supplemental/` owns its four raw trials, protocol, summary and independent
review. `summary.json` is the combined report view, and
`joined-qualification.json` binds both scopes plus the COLLIER review. All42
TIRust candidate attempts remain visible; no guard, requested tolerance or
mathematical recurrence was weakened to produce the opening timing cells.

## COLLIER opening-column follow-up: 26 October 2026

The next reporting request adds a right-most COLLIER column to the first
opening table. TIRust's prior18-row capture, all464 report records at91ce2a5,
and the generalized second table remain unchanged. No core/master/dependency
or Cargo implementation change was made.

`validation/benchmark/opening-collier-20261026/` owns the new external-reference
corpus. `driver.f90` derives from the frozen candidate004 wrapper, adding the
public Aten interface forN1 and per-block snapshots. It initializes supported
capacities Nmax=max(N,2) and Rmax=max(R,1,N-max(N-2,4)); requestedN/rank/output
remain exact. This avoids upstream zero-rank/N1 initialization defects through
public arguments, with no library patch. All12 ordinary rows use COLI mode1,
binary64, scalar masters included. N5R6 is not a DD-mode claim.

`build.json` links the already qualified, unmodified COLLIER1.2.10 static
archive4f1e6dc1… with strict -O3/no-fast-math gfortran15.2.0. All28module files
are pinned. Frozen wrapper binary SHA is
`bc0f57ba74c6af69042fc05047aa45a8e75daecbb2374025154d2d9c46441b5c`.
CPU18 capture is17.315s total, with3warmups, one single-call pilot, then11fixed
blocks targeting100ms (cap1e6calls/block) and pre/post UV/IR probes. Every root
resets local memoization via InitEvent; event caches are disabled. Timing
includes invariants, public tensor API, masters, native normalization/gather
and inherited status/finite scans. TIRust's post-call full scans are outside
its timer; the difference is explicit. No paired ratio or equal-precision
claim follows: TIRustN5/N6 useFloat128 and their observations stay frozen.

All12 native child processes exit0; the campaign intentionally exits1 to
expose3 retained numerical failures. The outer process group is gone and all
library/module/source/binary guards pass. `capture001/` archives169 raw files
(14,970,177bytes) byte-identically; original target paths are retained and the
portable checker maps them to the committed archive. No native jobs remain.

NineN1..N4 rows qualify, including all3864 distinct complex Laurent fields.
N5R5 has accuracyflag-1 and11finite mismatches (max≈3.88327e-10); N5R6 has
flag-1 and62mismatches (max≈1.50522e-9). N6R6 has flag-2,195nonfinite finite-slot
values at degrees3..6, one additional finite mismatch, and failed pole probes.
The producer comparison stops at the first null; the independent audit counts
all fields instead of relabeling missing comparisons aszero. Total6048fields
=5853finite+195nonfinite, with74finite mismatches. All216saved arrays and48wire
requests are checked;132timed blocks contain2,047,397calls.

The column shows accepted means/SE in orange and rejected rows in amber.
Subordinate observed rejected durations (0.597/2.097/3.293ms) are diagnostics,
not successful reduction benchmark times. No mode retuning, retry, tolerance
change or speed aggregate was introduced. These failures concern this exact
input family/precision/policy, not mathematical absence ofCOLLIER APIs.

`independent_check.py` audits saved JSON without evaluating integrals; its
default output is a target replay receipt, preserving the sealed review.
`summary.json`, `qualification.json` and `RESULTS.md` separate accepted and
rejected outcomes. See the suite README for frozen-build capture commands.
New report records are additive and retain all prior raw evidence and proof
history. Future benchmark work must preserve these rejected cases and source
scopes rather than quietly substituting another backend or precision.


## Opening benchmark tables: 26 October 2026 follow-up

The functional goal below remains complete. The user's subsequent reporting
request adds two opening tables to `docs/benchmark.typ` and the regenerated PDF:
12 ordinary N≤6/cumulative R≤6 examples, then six generalized requests outside
COLLIER 1.2.10's **direct public integral API**. No reduction algorithm, scalar
master, dependency, Cargo manifest or lockfile changed in this follow-up.

The new corpus is `validation/benchmark/opening-rank6-20261026/`:

- `inputs.json` and `make-inputs.py`: fixed 18-row selection, derived exactly
  from the conditioned massive Cartesian fixture. Shifts are the prefix of
  `(0,e_x,e_y,e_z,e_t/2,e_x+e_y)`, squared masses `(1,2,3,4,2,3)`, scale μ=3.
  All input strings are exactly represented dyadic rationals.
- Ordinary rows: N1R0/R6, N2R1/R6, N3R2/R3/R6, N4R4/R6, N5R5/R6 and N6R6.
  Hybrid; f64 through N4, Float128 for N5/N6, requested accuracy parameter53.
- Generalized rows: N1R6/N2R2/N3R4/N4R6 first-dot, N4R6 first propagator
  cubed, and N3R2/h1. All Float128/IBP/requested53. Positive powers and an
  independent dimensional request are not direct COLLIER arguments. Equivalent
  coincident-denominator or derivative representations were not benchmarked;
  do not claim these mathematical integrals are impossible in COLLIER.
- `crates/iregi-rust/examples/opening_timing.rs`: source-only benchmark driver
  through the public `TensorReductionPlan`/`TensorWorkspace` facade. It requests
  every cumulative Cartesian tensor moment and all four native Laurent slots.
- `run.py`: portable source guard, release build/freeze, pinned process-group
  supervision, non-overwriting attempts and raw-statistics aggregation.
  `source-files.json` names the selected committed context; it is explicitly
  not a hermetic native build-input closure.
- `build.json`: core source epoch d2762f5 with frozen example binary
  `0d3c5c37abfede1b10fb7ff99e55abc2a395bdfc9b0d4bac4ea3d966ed488c37`.
  The first release compilation took810.137s; two subsequent source-context
  refreshes reused the same executable, with the final Cargo artifact check
  taking0.316s. Build duration is not an integral timing. Preliminary build
  receipts/logs stay in `target/opening-tensor-timings-20261026/initial-build.*`.
- `cases/<id>/request.json`, `result.json`, `supervision.json`, `native.log`:
  all18 processes returned0 and measured complete finite successful tensors.
  First/final represented arrays and every individual timing observation are
  saved. There are227,451 timed calls; numerical repeat checks occur outside
  timers. Numeric equality does not certify signed-zero bit identity or
  unchanged internal routing flags.
- `summary.json`: block means, medians, min/max, standard deviation and SE.
  Low-point rows:3warmups,11blocks targeting100ms, maximum10000calls/block.
  N≥5:1warmup,5one-call blocks fixed in advance. Different working precisions
  are explicit and must not be treated as binary64 comparisons.
- `profiling-boundary-source-audit.json` and the independent fixture/timing
  boundary review explain the same-call split: TIRust-exclusive = outer call
  wall minus aggregate nested scalar-master wall. E6 includes its internal D0
  work; nested child profiles add stages once. Imported affine quadrature only
  propagates already-computed error estimates and belongs to exclusive time.
- Plan/workspace construction, the first numerical call/lazy specialization,
  warmup, full-array checks, result destruction and I/O are excluded from warm
  call time. Allocation/construction through result return remains included.
  Instrumentation stays enabled. Profiling selects AffinePower for the N6 row,
  rather than the unprofiled thin SixPoint path: no fastest/default-thin claim.
- CPU18, AMD EPYC9754, Rust1.99 release, one numerical thread. External host
  activity is uncontrolled. SE describes observed block variation, not a
  controlled-host confidence guarantee; these are absolute observations, with
  no new IREGI/COLLIER speed-parity gate or optimization adoption.

Representative same-call mean times: N4R6/f64 total16.827ms, of which masters
0.296ms; N6R6/Float128 total3.669s, masters0.347s; N4R6 first-dot/Float128
1.606s, masters0.0295s. Numerical source and precision must accompany these
figures. N6 cold preparation is31.339s plan +89.103s workspace +130.169s first
full numerical call; these costs cannot be relabeled as warm kernel time.

Independent validation uses `compare-reference.py --require-all`, admitting
only the pinned original IREGI IBP arrays with exact input/basis/Laurent mapping.
All8,968 complex fields pass the unchanged dimensionally normalized mixed
`1e-10` gate, maximum below2.714e-13. Original rejected PV/Hybrid history remains
in `reference-comparison.json`, SHA
`5d72855e1ad01690c534672d39c7a12e4e2bbbf043b4215a1b163375c95cae8e`.
The root replay is byte-identical; a closed72-file portable read callback also
reproduces the entire report without target reference files, a license, native
integrals or the producer's comparator. Working128 bits do not imply128 achieved
bits: Symbolica tracks cancellation-dependent operand precision.

See the suite README before recapturing: committed `cases/` already exist even
in a fresh clone. Preserve/move published observations in a separate output
copy. The saved-evidence comparator deliberately pins the original receipt
bindings; a new measurement campaign requires its own receipt bindings/review.
No numerical producers or builds remain active at publication.


## Final functional checkpoint: rank six, with optimization deferred

The user's revised numerical checkpoint is complete: documented rank-six
families through six points, including dotted propagators. The final response
records the pushed commit and objective closure. New sessions should start from
that commit and treat further performance optimization or rank-seven work as a
new task. No numerical producer or build remains active.

Tensor rank and point count are separate. The public tensor facade uses
dynamically sized denominator descent without a fixed mathematical upper point
count. The concrete independent validation checkpoint is the documented
one- through six-point families, through cumulative rank six. This does not
qualify arbitrary point counts, all Gram/Cayley charts, or the Cartesian product
of every type, power, shift and method. Four-dimensional numerator tensors have
210 cumulative Cartesian moments through rank six; each has the four native
Laurent slots `[UV single, IR single, IR double, finite]`.

### What is implemented

- A pure-Rust numerical reduction library with immutable plans and separate,
  owned generic batch workspaces. Graph preparation, Python, physical models,
  Fortran references and MadLoop are outside the reduction library.
- IBP, PV and default Hybrid dispatch; positive propagator powers, dimension
  shifts, real/complex squared masses, routing transformations, supplied
  scalar products, checked high-point compatibility certificates, normalization adapters,
  tensor multiplicities and separate numerator contraction.
- `f64`, Symbolica `DoubleFloat`, arbitrary-precision `Float` and Symbolica
  static/dynamic Dual layouts. Samples and Taylor components remain separate.
  Recurrence nodes and scalar-master instructions process active sample lanes;
  batching is not an outer loop around a scalar reducer.
- Symbolica/FeynKit/Spenso/Idenso algebra and oneloopmaster scalar evaluation.
  Reference dependencies and narrow local patches retain upstream provenance.
- An independent graph/numerator preparation crate and standalone CLI, plus
  typed PyO3 bindings and executable marimo examples. The current installed
  wheel smoke and historical broader Python qualification are separate source
  epochs; do not add their test counts together.
- Lane-local invalid, master-failure and stability diagnostics, preserved
  inputs by default, explicit Legacy snapping, and propagated full-component
  quadrature/roundoff ownership checks. Unsupported charts do not return
  successful zero tensors.

### Completed evidence already in the preceding milestone

The preceding precision-boundary milestone is
`da7e9bc73752eda8eac31ac13905ba770ada775d`. Its independently rebuilt and
reviewed 66-page report contains 454 raw records, preserving all 451 a7ccffb
records and all 58 historical proof fields exactly. Its PDF SHA-256 is
`f9b9ed9955e1d78f62ee332ccf46fac64fe8b8efe59b50842f7200a6f4f7181d`.
The final publication preserves these exact 454 records and nested history,
then adds the actually sealed final Dual continuation and its references.

The immutable precision-boundary parent is
`validation/e6-dual-rank6-precision-boundary`, manifest `cd656689…`.
It owns both complete 128/256 captures, their passing reference/context/lifecycle
checks, the strict 128/256 failure, independent saved arithmetic, diagnosis and
source/schema history. The final capsule inherits this parent once rather than
duplicating those captures or the DD/source payloads.

### Completed shifted rank-six Dual convergence

The original SAMPLE, N6/rank6/h1, second-mass-derivative witness now completes
all three actual calls at 128, 256 and 384 working bits on the same frozen 749b
artifact and selected 210-source build b069. Each unique parent array contains
2,520 complex fields: 210 cumulative Cartesian moments × four Laurent slots ×
three ordered Taylor components. All fields pass both independent DD rules at
unchanged `1e-10` dimensionally normalized mixed tolerance; the largest normalized
difference is 3.9743290309e-15.

The final strict SAME_SOURCE 256/384 pair passes all 2,520 fields at unchanged
2^-80, maximum normalized difference 3.8999626029e-60. Its complete joint report is
04f64a71f25acaf9b3391bc8d7c5481a557fd5d851b7944d3cb47cf127fed31d.
The older strict 128/256 comparison remains FAILED 628/2,520, maximum
2.9216229277e-21; exact failed report
2472628367171ba81589e18c949a3e2681489d76d2395693a7c23610b1be0b7a.
Never relabel or omit that failed precision pair. The successful increasing-
precision check does not establish a rigorous error enclosure or achieved bits.

Every precision completes the fixed odd batch `[valid, invalid_mu_zero, same_valid]`,
empty reset and singleton recovery. Each has three successful parent and six
specialized child owning contexts, positive propagated quadrature estimates in
all three Taylor components, exact valid duplicate/recovery equality, isolated
invalid null outputs/context refusal, and actual empty-result/counter reset.
The requested PV invocation selects the documented generalized IBP recipe:
0 canonical PV / 610 IBP requests, runtime fallback 0. This is not a new
numerical default-Hybrid h1 invocation; separate source-equivalence notes and
other historical Hybrid tests retain their own scope.

Actual 384 receipt 120f490a…, captureb848d205…, mechanismf3991137… and
single-precision numeric d16309e4… are joined in its final capsule. Root rehashed
all 262 receipt inputs and three outputs and independently observed the owned
group gone. Root terminal review is 69504307… . All native groups are terminal;
no numerical producer or build remains active. The 384 controller took
5402.417764s, its receipt worker step 5400.472286s and libtest 5392.21s. These are
separately scoped validation durations, not benchmark timings.

The unchanged closed independent arithmetic/context/lifecycle reviewer a6e79
and history/product wrapper 4ff7 passed the complete 256/384 packet, including
six 2,520-field comparisons and four 1,806-control raw-pole sets. Live guards 248+262
and three outputs per run were rehashed, identifying exact 749b. Independent
base report 4180b1f0… and history supplement cd17c2c1… reproduce in a fresh
13-file plain JSON/Python packet without native execution. Minimal proof 943a3fb7…
labels omitted live guard observations honestly; the old failed 247262 pair is
preserved. Raw UV/IR controls remain unprojected; 384's normalized residual
maximum is 1.7331466256e-106.

The requested accuracy parameter is 53; heuristic success thresholds `2^-34` and
E6 relative `2^-36` do not guarantee 80-bit consistency. Source analysis finds all 630
finite 128/256 differences consistent with combined saved heuristic estimates,
meaningful imaginary tracked precision 94→222 bits and cancellation amplitude
about 1.62e12. This is not a rigorous remainder bound, causal proof or achieved-bit
certificate. No core formula, input, branch, tolerance or build changed for 384.

The final immutable capsule is
`validation/e6-dual-full-rank6-mass-second`, manifest
`28c16b64347070b785ad25f5aba8b235afe8ba989e59d147672581b8549074ef`.
Its payload is `7b8362e2a04cf6a7f50c90f5df1a49fc6b621eeeeef7fbfe61d10e493c953656`.
It inherits the precision-boundary parent once, with no source, DD or128/256
body duplication. The direct root saved replay is
`docs/data/rank6-functional-completion-saved-replay-20261026.json`, SHA
`910b6c54ee9fe91265bc618c4294cca1ab3a5ca2859a4f320717a45d541a2fb1`.
Root's environment-cleared fresh42-file replay also passes; receipt24dab2cc… .
Root copied the exact reviewed six-file packet to its validation path without
recollection or manifest rewriting; sealing proof505b7b98… . These saved replays
execute the unchanged numerical and independent comparator functions on captured
data, not merely source-hash or summary checks. They do not reevaluate an integral.
Their `full_objective_met=false` flags describe collection-time scope; final
publication and objective closure are separate later steps.

The first final-capsule replay failed a source-guard assertion before any
mathematical replay: it looked for canonical source paths in a receipt that
correctly pins210 build-snapshot paths. Original packetf9fa88dc/checker2fbef734,
owner logs and root isolated42-file failuredf4f111a remain preserved in the
outer provenance. The only fix uses the exact declared build-specific
`current_source_aliases` mapping for each of210 source guards, with no canonical
fallback. Corrected checker785afa95 is independently audited in67ebcc04:
210 actual source bodies and receipt guards match;0 canonical/0 unmatched.
All262 input guards and numerical results remain unchanged. The corrected
root default and isolated replays pass. This checker failure, like the earlier
schema and scratch-contract failures, is not an eleventh failed native witness.

The final report has68 pages,457 archival records,409 bound sources and378
exact JSON snapshots. All454 preceding raw values and60 historical proof fields
remain exact. The current PDF SHA-256 is
`75759f1935ebf095f3645b94ce7eff0cf1438ecdc842d75e26520f83ecf809c8`.
Root independently rebuilt it byte for byte and viewed the current coverage,
completed gates, limits and historical precision-boundary pages. The report
retains IREGI/COLLIER measurements, TIRust/reference ratio direction and all
slower/noisy/unsupported cases; no new performance claim is made. Final source,
rebuild, visual and payload provenance lives in
`docs/data/rank6-functional-completion-root-review-20261026.json` and the adjacent
scan/replay records. The remaining action at publication is to push the final
commit, mark the existing revised objective complete and stop.

The ordinary N6/rank6 dotted and D2 capsule is
`validation/ordinary-gh-n6-generalized-requalification` (`185b9372…`). Each
family retains all 840 Laurent/component reference checks per precision array and 840 strict
cross-source precision pairs. The ordinary D6 capsule is
`validation/ordinary-gh-n6-d6-requalification` (`1386c4ce…`), including both
defining-integral rules and 5,332 independent saved checks. These precision
pairs compare current 256 with historical 128: label them CROSS_SOURCE.

The historical bypass and its 24 scoped passes are owned by
`validation/e6-dual-optional-zero-profile-core-tests` (`403dfee6…`). It did not
qualify the full rank-six/null-kinematics route. The current restoration is
`validation/e6-dual-restored-zero-profile-core-tests` (`1f1a41fd…`): all 24 focused
tests pass on actual opt3 product749b/buildb069/source210. Its regression preserves
two finite R0 lanes and one genuine transverse E6NonConvergent refusal with no
failed output/context. Prior23pass/failed-expectation b804 is retained. Use the
capsule's build-specific current_source_aliases; inherited canonical execution.rs
still denotes historicalb4d, not current4f673. Only three new source bodies are
owned, with older bodies borrowed rather than duplicated.

The current wheel is
`validation/python-interface/restored-zero-profile-wheel-refresh-smoke` (`c65b3e60…`):
205 selected bodies, 128 declared coherent core bodies, ten installed API
cases, and 2,709 saved checks. Its N6 frontend lifecycle smoke is rank one.
Fresh compiled native8978 is byte-identical to the historical safe pre-bypass
native, and differs from bypass-era2220; actual source/build guards are retained.
Historical 65 API cases, seven marimo runtime cases and consumer typing retain
their own wheel provenance; a complete separate Community-host loading path
is not qualified by the standalone shared-host tests.

The independent full mass-second reference is archived once in
`validation/e6-dual-rank6-mass-second-defining-integral` (`dd499b4a…`). Its
86,272 points, 337 batches and eleven windows produce two complete 5,880-field
arrays, each including all 2,520 parent tensor/Laurent/Taylor fields. Saved
stream and rule-pair reports reproduce exactly; the maximum normalized parent
rule difference is about `2.45e-15`. It uses nominal 106-bit DoubleFloat
arithmetic with the documented 53-bit quadrature nodes; this is neither an
achieved-bit nor a rigorous error-interval claim.

Four post-fix no-array attempts are archived in
`validation/e6-dual-postfix-resource-boundaries` (`78fb3bc3…`), borrowing
the earlier source context and independent reference rather than duplicating
them. Its saved replay reproduces all four failed mechanism audits and 976
nonbinary input guards. The four older failures remain in source capsule403.
The two-second/96-sample profile and the finite batch-memory model are
diagnostics; they do not establish the exact failure substage or whole-run
dominance. Raw perf data is deliberately outside Git.

Two later 96-GiB full-witness attempts are archived in
`validation/e6-dual-postfix-nonfinite-boundaries` (`792765c1…`). Their E6 masters
converged, but all four packed N5/h1/R6 tensor rows failed the nonfinite-output
guard; valid parents were Unstable and the invalid neighbor stayed isolated.
Both failed calls and all null output/context states remain lossless. No first
bad instruction was exported. These make ten unsuccessful full-witness attempts,
separate from compiler-boundary and focused-test failures.

The independent physical evidence includes four FeynKit/HELAS graph pairs at
three fixed loop momenta, documented graph factors and routing, and selected
graph-isolated MadLoop integration checks. The later CLI physical record has
14 CLI tests, 2,800 tensor checks, twelve merged MadLoop coefficients and 568
strict precision-pair checks for fixed rank-four box/pentagon examples. These
are not new rank-six physical checks. R2 and UV counterterm diagrams are
excluded; bare poles and reduction-generated R1 are retained.

### Important numerical and resource history

The ordinary complex-box coefficient repair preserves an exact correlated
combination before numerical rounding. It fixes a demonstrated cancellation
failure and is independently checked against defining integrals. The seeded
Dual expression remains on its separately qualified chart. Do not generalize
an ordinary-source repair into a derivative-domain claim.

The original IREGI D6 reference finite values failed an independent
positive-F/scalar-mean identity; all rejected values remain archived. The Rust
qualification instead uses two independent defining-integral rules with raw
UV contact terms retained. Never silently reinstate those original finite
coefficients as acceptance values.

The full-Dual SAMPLE has three exact null time-space separations. A second
12,318-alias preparation observed after the completed generic 12,441-alias
constructor was runtime exact-zero specialization, not proven Hybrid
constructor fallback. The temporary bypass avoided that preparation, but the
generic tensor recurrence retained removable lightlike-bubble divisions by
numerical B=0. Finite E6 masters therefore did not establish finite tensors.
The restored source removes only that 15-line bypass, reusing the existing
full-Taylor-certified structural zero chart and all admission/context guards.
The exact first bad instruction remains unavailable in the failed captures.
Actual requested PV selects generalized IBP (0PV/610IBP) for these N5/h1 children;
legitimate profile/analytic rows must be retained in mechanism evidence. Old
PV-only/zero-analytic expectations are obsolete, not numerical tolerances.

Preserve all earlier OOM/timeouts: Hybrid32 primary preparation; Hybrid64 and
PV128/PV256 optional-profile preparations; and the two post-fix 2,400-native
deadline failures. The additive 4,200-native attempts change only the time
allocation, retaining the same code, fixture, precision, requested accuracy
and 64 GiB address-space bound. A timeout with no arrays is not a numerical
pass. Both subsequent96GiB attempts ended in mathematical guard rejection.
The restored-source trial is separately compiled opt3, with compiler debug
assertions/overflow checks false and mandatory numerical guards unchanged.
The sequential Float128/256/384 protocols use separate 192-GiB/256-GiB/256-GiB
address-space budgets. Their actual terminal receipts and observations establish what ran;
the selected limits themselves do not establish resource sufficiency or accuracy.

### Limits and future work

Genuine E6 quadrature requires the documented positive-F or complex-width
domain. Seeded C0/D0 mass derivatives have conservative supported charts;
refusal of a zero-mass tangent is not a proof of mathematical nonanalyticity.
The retained massive positive-F R0 time-shift germ also reaches E6NonConvergent
at order128; this is a numerical refusal of an analytic germ, not a nonexistent
derivative. It is outside the accepted mass-only full-rank witness.
Checked high-point supplied-product routes require the full documented
Gram/Cartesian/Taylor certificates; lower-point supplied products retain
their authoritative-input semantics.
High-point Legacy snapping is accepted when certified to preserve the input;
geometry-changing snaps remain unsupported. Composite explicit-Dual precision
rescue has an explicit unsupported domain. Requested and actual methods can
differ in documented high-point completion; preserve those diagnostics.

Neither working precision nor a strict precision pair establishes achieved
accuracy. Successive quadrature-rule differences and propagated error
estimates are not rigorous intervals. Existing performance results include
slower, noisy, failed and unmeasured cases; functional completion does not
claim performance parity. Every implementation/reference timing ratio uses
TIRust in the numerator. Reference-only IREGI/COLLIER timings remain absolute.

If the user later requests more work, prioritize measured preparation and
evaluation costs, structural reuse and certified exceptional charts. Rank
seven increases expression/monitor size and resource pressure; arbitrary N
needs its own denominator-reduction and stability qualification. Do not start
those campaigns merely because the architecture can represent them.

### Reproduction and repository rules

All commands run from `/shared/external/iregi/iregi-rust`. Source the pinned
toolchain roots and `scripts/env.sh`; source the local Symbolica license file
silently when a native run needs it. Do not print or commit that environment
file. Original reference checkouts are read-only, except the requested
`../references/PLAN.md` copy. Keep generated builds and artifacts in `target/`.

Saved capsule replay needs only the standard Python library and does not
reevaluate any integral. Use each capsule's README and checker. For the final
Dual witness and report:

```sh
source .local/tirust-current-toolchain-roots/env.sh
source scripts/env.sh
python3 -I -S -B validation/e6-dual-full-rank6-mass-second/check.py
python3 scripts/check-benchmark-report.py --preserve-from da7e9bc
typst compile --root . --creation-timestamp 1792972800 docs/benchmark.typ \
  target/rank6-final-feature-report-20261026/rebuilt.pdf
cmp docs/benchmark.pdf target/rank6-final-feature-report-20261026/rebuilt.pdf
```

The final checkpoint and report contain the final capsule paths, actual
receipt hashes and exact arithmetic checks. Git contains source/evidence,
not native executables or the license. Configure local author identity as
`huashengshao <erdissshaw@gmail.com>` and push normally to `origin/main` without
force. All owned numerical process groups are terminal and cleaned up. The goal is
closed only after the actual final evidence/report checks and commit/push.

All sections below are historical context, superseded by the final checkpoint
when they describe an older pending state. Preserve their source epochs and
failed attempts; do not reinterpret them as new successful results.

---

## Current checkpoint: complete reference checks, convergence still open

The accepted finish line is the documented one- through six-point families,
through cumulative tensor rank six, including dotted propagators. Performance
parity and further rank-seven work are deferred. The goal remains open. An existing unbudgeted goal is already assigned;
its tool status may still show an earlier blocked state after the user's resume.
Progress is available. Do not create a second goal or infer completion from that
status. Finish
the remaining precision-convergence check, publish its evidence and updated
benchmark report, commit and push, mark the objective complete, and stop. Do not
start optimization automatically.

The ordinary six-point/rank-six dotted and dimension-shifted evidence is already
committed. The restored exact-zero route now also returns complete six-point,
rank-six, dimension-six second-mass-derivative Dual arrays at both 128 and 256
working bits. Every one of the 2,520 complex tensor/Laurent/Taylor fields agrees
with both independent defining-integral rules at the unchanged `1e-10`
dimensionally normalized mixed tolerance. Both runs pass invalid-lane isolation,
empty reset, duplicate equality, singleton recovery, and actual owning
full-component roundoff/quadrature context checks. Native groups exited cleanly.
These are numerical validation results, not performance benchmarks.

The stricter same-artifact 128/256 comparison **fails**: 628 of 2,520 fields
exceed `2^-80`, with maximum normalized difference `2.9216229277e-21`.
Keep this failure visible. Do not relabel it as passing, project small raw poles
to zero, or lower its threshold. The full joint report is
`target/e6-dual-restored-zero-profile-opt3-128-256-joint-comparison.json`, SHA
`2472628367171ba81589e18c949a3e2681489d76d2395693a7c23610b1be0b7a`.

Saved-data analysis is consistent with precision-dependent cancellation and E6
numerical noise at the requested accuracy parameter 53. At the worst field
(`qz^6`, second mass Taylor coefficient), the Float128 owning quadrature estimate
is about `9.99e-21`, larger than the observed `2.92e-21` difference. All 630 finite
field differences fit combined stored heuristic estimates. Meaningful tracked
imaginary precision rises from 94 to 222 bits; both runs use E6 order 64 and
2,400 box rows. Requested parameter 53 admits heuristic thresholds `2^-34`
(roundoff) and `2^-36` (E6 relative), and does not promise 80-bit cross-precision
agreement. None of these estimates is a rigorous interval or achieved-bit
certificate. No formula defect or fixed 53-bit ceiling has been demonstrated.

The immutable capsule is
`validation/e6-dual-rank6-precision-boundary`, manifest
`cd656689a9c02dd92c401a0ad9009be7e7f7e655d1f62d337b44fd66555777fa`,
payload `d0932d88b895e886e33d4e874ee5ca3c4df5da6b4e5d2397f6a13adaae4427fb`.
The [precision-boundary checkpoint](docs/checkpoints/rank6-dual-precision-boundary-20261026.md)
and [saved replay packet](docs/data/rank6-dual-precision-boundary-saved-replays-20261026.json)
record actual portable/source checks. Root independently replayed a fresh
36-file, environment-cleared copy without target/.local/license dependencies;
proof `073e5a55…`. A root-only attempt to relocate temporary reports into the
saved project namespace was correctly refused; its stderr and wrapper are
retained separately from the passing standard-contract replay. This is not a
native or numerical failure. No capsule source or threshold was changed.

The capsule and checkpoint identify actual receipt and source joins. The two previous 96-GiB
nonfinite captures, eight earlier resource/time failures, failed checking-script
assumptions and compiler/test failures remain separate historical evidence.
A failed precision comparison is not an additional failed native capture.

### Active numerical work

Only the Float384 convergence witness is active. Owner:
`/root/e6_precision_failure_research`. Directory:
`target/e6-dual-restored-zero-profile-opt3-384-256g`.

- Unified session: `27549`.
- Native PID/group/session: `1144571`; kernel start tick: `105132048`.
  Verify identity before any signal. The owner polls and collects this session.
- Protocol: `3e0f5544e1c1fca291c29838dc62137aa07e51e9ebf2869838e2919c338b9aae`.
- Root release: `1fcbf079008fde752d976e6a7d33514eabd843215367f4a549e3828e8169297e`.
- One pinned CPU30, 256-GiB address-space limit, 7,200 native seconds,
  7,380 total seconds and 120 seconds reserved for cleanup.
- Same frozen executable `749b2ae3489eb0096267421b120fa3586f2243c7eaf78fc9c02b0f9f30895906`,
  actual 210-source context, original SAMPLE, rank6/h1, mass-second shape
  `[[0,0],[1,0],[2,0]]`, requested accuracy parameter53 and three fixed calls.
- All 260 prelaunch bindings were checked independently. No source, Cargo,
  formula, branch, tolerance or build change. No other full producer or retry.
  Fresh available host memory was 504.9GiB; the cap is an engineering allowance,
  not a proof of sufficient resources.

The frozen observer explicitly supports working bits `[128,256,384]`.
The proposed 192-bit run was rejected to avoid introducing a new observer/build
epoch. The saved-analysis extension only admits the existing 384-bit mode;
its arithmetic comes from the unchanged `df350…` comparator. The next qualifying
convergence pair is explicitly **256/384** at the unchanged `2^-80` threshold.
The old 128/256 result remains failed even if the higher pair passes.

On the first actual384 array, compare every field against both DD rules and
perform a labeled preliminary 256/384 check. Full acceptance still requires all
three calls, exact duplicate/recovery equality, real owning contexts and nonzero
higher-component quadrature estimates, terminal cleanup and unchanged guards.
No current384 result may be inferred from128 or256.

### Source and evidence map

Current production restoration deletes the 15-line bypass of the existing
full-Taylor-certified exact-zero profile. It reuses existing IREGI/Symbolica
identities; no new algebra engine or numerical guard relaxation was added.
The old generic path retained removable lightlike-bubble divisions by numerical
`B=0`. The first bad instruction was not exported in those failed captures.

The full-Dual witness explicitly requests PV and selects the documented
610-term IBP child recipe (zero canonical PV requests), with no runtime Hybrid
fallback. Keep `default_Hybrid_qualified=false` local to this invocation.
Separately saved source-equivalence notes show the same successful h1 recipe for
PV and Hybrid; they do not fabricate an additional Hybrid numerical run.

Use `validation/e6-dual-restored-zero-profile-core-tests` manifest `1f1a41fd…`
for current buildb069/product749 and all24 focused passes. Resolve current source
through its build-specific `current_source_aliases`; inherited canonical
`execution.rs` is historical, not the restored body. The regression honestly
retains one analytic transverse germ refused as E6NonConvergent at order128.
This is a numerical limitation, not proof of mathematical nonanalyticity.

The original saved checker incorrectly assumed only valid parent rows and no
empty owning result. The observer preserves all original rows, including
invalid context refusal, and an actual empty result. Source-reviewed additive
checker `e0b6889d…` requires exact owning-result equality and all original failure
records. Preserve frozen `b2…`, failed audit `de3…`, and the corrected replay.
Only the later `dc690560…` bit-domain extension admits384; formula/tolerance
functions remain unchanged.

The independent DD reference is `validation/e6-dual-rank6-mass-second-defining-integral`
manifest `dd499b4a…`: 86,272 points, 337 batches, eleven windows, two complete
5,880-field streams including all2,520 parent fields. Saved stream reconstruction
and exact hi+lo witnesses are already qualified; do not rerun the integration.
Nominal106-bit arithmetic uses documented53-bit nodes and proves neither a
rigorous error bound nor106 achieved bits.

Full128 terminal receipt: `aa36ca07…`; full256: `f3ed1dc0…`.
Root rehashed all237/248 actual input guards and three outputs per run.
Root reviews: `bd3414a2…` and `3d3ac05a…`.
Independent complete128/256 arithmetic/lifecycle receipt: `7e4178aa…`, with closed
standard-library helper `a6e79be9…`; ten-file minimal replay `03b66c81…` confirms
all six2520-field comparisons and four1806-pole control sets. Expected exit1
preserves the failed strict gate. Native/compiler observations are historical
when their binary files are intentionally absent from a portable copy.

Current Python wheel capsule is `c65b3e60…`: ten installed smoke cases,205 selected
sources/128 coherent core bodies, N6 frontend smoke rank1. Historical65 API cases,
seven marimo runs and typing checks retain their own wheel provenance. The
standalone shared native host is qualified; external Community-host loading is
separate work. Physical comparisons remain documented rank4 graph examples,
with graph-isolated MadLoop, bare poles/R1 retained and R2/UV counterterms
excluded. No new rank6 physical or arbitrary-point-count qualification is claimed.

### Finishing and publishing

The precision-boundary capsule is an immutable parent for the final full capsule:
inherit its completed128/256 captures and failure records once, then own only384
and the new convergence proofs. Keep source/DD payloads deduplicated. Replay
actual saved numerical functions and independent validators, not merely their
source hashes. Run the final checker in a minimal copied packet without native
or license dependencies; never pretend omitted binaries were rehashed.

The report owner is `/root/current_rank6_report` (use `followup_task` when idle).
The capsule owner is its `low_n_generalized_capsule` child. Root owns final
README/HANDOFF, independent PDF rebuild/visual inspection, payload scan,
commit/push and objective completion. Preserve every prior raw record and nested
publication-proof object when rebasing final drafts from this milestone.

All commands run from `/shared/external/iregi/iregi-rust`. Source the pinned
`.local/tirust-current-toolchain-roots/env.sh` and `scripts/env.sh`; source the
local Symbolica license silently for native runs or private-literal scans.
Never print or commit the license. References are read-only except the explicitly
requested `../references/PLAN.md`. Keep generated work under the project.
Configure local author `huashengshao <erdissshaw@gmail.com>` and push normally to
origin/main, without force. Numerical source/inputs stay frozen while384 runs;
report publication paths have been checked disjoint from all260 bindings.

After actual384/both-DD/strict256384/lifecycle/capsule checks pass, update the
Typst report and handoff, independently rebuild and inspect the PDF, preserve
historical records, commit and push, give the immutable GitHub PDF link, mark the
existing goal complete, then stop. Performance, rank7 and additional charts are
future work, not automatic continuations.

Everything below is a historical session ledger. Older pending-state statements
are superseded by this checkpoint, while their source epochs and failures remain
part of the evidence.

---


## Live status after restoring the exact-zero route — 26 October 2026

Current source execution4f673f8c/testd4bc4ae0 has a successful optimized build
and ALL24 focused tests PASS. Actual buildb0695760d975d4c0f789400699861f3b4581f7fae5677d682748f2cf0e075da2,
artifact749b2ae3489eb0096267421b120fa3586f2243c7eaf78fc9c02b0f9f30895906,
focused65bc99d85aa08e9b07c704f818691212b4a8ec2d578184c06d7dec6e7089581f
(26.859s) at `target/e6-dual-restored-zero-profile-test-correction`. Root verified
actual opt3/test=true/source210. Compiler debug assertions and overflow checks
are false; mandatory numerical admission guards are unchanged. No source edits
while captures run.

One FULL Float128 witness is ACTIVE, ownerE6 agent, session15282CPU28:
`target/e6-dual-restored-zero-profile-opt3-128-192g`, protocol
ff2a9ef1f0e2b6219773a11bb1f01f4224a0886e6d959886ab3b340477598e7f.
All235bindings root+audit checked;192GiBAS/7200native/7380total120cleanup,
originalSAMPLE/masssecond/h1/rank6/requested53/all2520/fullQ/lifecycle guards
unchanged. No actual arrays yet. FIRST128 ONLY;256 HELD until actual128 review
and separately released protocol. Available host memory~337GiB was observed,
not a proof of required peak. No automatic retry or source edits.

Fresh Python wheel BUILD PASS118.851s/source205 current-context5dfe8459:
build86e16fbff76c819be2ee50fb9bac5ba493e3313bb64ef4b8dd8ac43c6ddac3df,
wheel03184baf6a36fa03fe66ddb1af0ba51e53bb2ffac6caea4ce29aaee50b5fc84b,
native8978f06fb3653502ce4de7a918a5c48faa6ad3b0ded4d2ca34ad4cf34d0c2de9.
Actual compiled/logged native differs from bypass-era2220aa0b and is byte-identical
to older pre-bypass8978, as expected from restoring production code. Install/
import/exactsame10smokes ALLPASSED; session96323 terminal, all7ownedgroupsgone.
Sealed capsule `validation/python-interface/restored-zero-profile-wheel-refresh-smoke`,
manifestc65b3e603706219f5562711c2d28f2190ebef46baa56fb879c80651c5cc329b3;
root2709savedchecks+independent17fileminimalPASS/proof3190e97c. No65/sevennotebook/
typing rerun; N6smoke rank1. No externalCommunityhost/fullrank6/perf promotion.
Cached wheel build used explicit target+target/cargo-build CPU20/21, unbound by
nativecapture; no code changes. All wheel phases and saved replays are terminal.

Source24 capsule `validation/e6-dual-restored-zero-profile-core-tests` is SEALED,
manifest1f1a41fdbc30ca1b3d5a18ce59db9f23fa0bb774950dba464aadfd05aabc5d04;
rootdefault+independent30fileminimalPASS/proofe4262f96. Four210-source contexts,
2161nonbinaryguards/56outputs/54groups replay, only3newsourcebodies owned; old
canonical sourcealiases stay historical. 792nonfinite capsule also sealed/
rootportablePASS. Intermediate report target separatefromfinalconditionals
has materialized truthful pending-fullDual publication after root cutoff.
The65-page PDF is a5a17216bf8f8d9cb25385e79bfc490799a710310c85bb3135c077f1253f8e14;
Typst a55083a376e149fc7fea1c7959e3b3c9e1538fdce6e6c2c187318f2fe9555232.
Root rebuilt byte-identically/viewed pages1–3. Data has451records/403bindings/
372JSONsnapshots, all446c112rawvalues and56earlierproof-fields exact. Three new
manifest records +saved-root-replay packet +checkpoint add archival records,
notnewtensorpasses. Bothfailed96capture files remain lossless in792payload;
per-attempt statuses/times/nullarrays/sourcepins are also in the checkpoint.
Root report proof d19352568c84a96b2d3546026f3d4e4a09146cb497d6da44f29ad43621964229
records rebuilt/visual state. Root owns finalpayloadreview/commitpush;
do not markgoalcomplete yet.
Restoration publication extends c112865; gitlog identifies the exact publication
commit. All446baseline records must remain byte-exact.

Latest active128 observation10:12:47UTC:995s elapsed, first exact-zero profile
preparing, stability/quadrature evaluators completed95.79/85.60s; AS79.8GiB /
RSS54.4GiB, identityclean/noerror/noarray. GenericN5constructor612.58s, numeric
mapping126.26s. These are engineering observations, not performance benchmarks.

## Earlier restoration and resource history

The ordinary six-point/rank-six dotted and dimension-shifted qualification is
committed and pushed. The goal remains open: both full quadrature-bearing,
second-mass-derivative Dual witnesses now finished with a numerical guard failure.
The96GiB128/256 attempts are TERMINAL, not active. Their E6 scalar masters
converged at order64, but all four packed N5/h1/rank6 tensor rows failed the
nonfinite-output guard; both valid parents were Unstable, invalidmu remained
isolated, and no failed tensor arrays or contexts were exposed. The independent
defining-integral oracle is complete; core comparisons are still impossible.

Float128 receipt d1e22aa773293ae57b2af35a3dedb2317ea04d0423cdacf556d7646bbbe6758e,
3978.668total/3967.944native; Float256 receipt
7c2b9078f2c0ca852764b303c6df8e17774ca343b3f205eab62056b8d9a1a3c1,
4427.056total/4419.194native. Bothrc101/no timeout, all immutable pins unchanged,
cleanupempty/groupgone; sessions44049/92862 terminal and CPU28/30free. Saved
diagnoses f05302a0 (128) andaa8e15fe (256) preserve exact inputs/seeds/statuses.
RequestedPV→actualIbp is EXPECTED generalizedN5h1 dispatch (610IBP/0PV), not a
runtime Hybrid fallback. Future mechanism checks must record this actual route.

Read-only independent source reviews localize a likely mechanism: skipping the
fullDual exact-zero specialization retains generic bubble descendants with
numerically zero B=1^T R^-1 1, so dimension recurrences can divide by zero. A
certified structural zero profile exposes the existing exact pinch shortcut.
The actual first bad node/component was not exported; do not claim it observed.
Restore/qualify the correct existing route with safe resources before new full
captures; no unchanged-source full optimized retry is planned. The old-source
opt3 build was explicitly cancelled before restoring source as recorded below.

Finish those actual gates, publish the final report and handoff, commit and push,
mark the accepted rank-six objective complete, then stop. Optimization and further
rank-seven work are deferred. All older pending-state prose below is historical
when it conflicts with this live status.

## Optimized test-build check after `c112865`

The resource/reference checkpoint is pushed as
`c112865e2dd3c6eab5dfe47cc6244f71e54b2d21`; the tree was clean after pushing.
Final target templates now preserve all446 records and the current publication
proof `9a0f26c9…`. Actual full-Dual arrays remain pending.

The recorded Cargo event proves current frozen4f430732 is an opt0 libtest
(debug assertions/overflow checks true). Root and finish-audit approved a NEW
compiler-profile-only build of the same210 canonical source bodies: ordinary
release opt3/testtrue/debugassertionsfalse/overflowchecksfalse. Numerical,
domain, precision and owned-error admission guards are unconditional source
checks; no formula or source changes are authorized by this build choice.

Initial build: `target/e6-dual-release-qualification`, source-freeze
`53d447e2754d10f5c58dd3ea31a171824eaf3cb6598418728902b99cdfb1039d`,
build.py4713eab7, CPU18/19, unifiedsession29328,900total/90cleanup/780cargo.
BOTH CARGO_TARGET_DIR and CARGO_BUILD_BUILD_DIR explicitly use
`target/optimized-build`; neither active96GiB producer binds those paths.
Old proposal7ce31c24/build1a3acc7d was corrected BEFORE launch and is preserved-v1.
E6researchagent owns the source/Cargo lease until build terminal; no edits.
Only if the build/event/freeze passes, root already authorized the SAME24
focused tests on CPU24 (600total/60cleanup, ordinary60s/newrealE6test240s).
No new full optimized capture, old producer cancellation or automatic retry is
authorized yet. Both96GiB debug producers continue unchanged on CPU28/30.

The initial780-second build ended at781.010s with an owned SIGTERM timeout,
no compiler errors, all210source guards unchanged, cleanupempty/groupgone and no
executable product. Receipt956f2005de48b528389bc1476e5b311233968f8077900ce8d94bbfd4148c94ea
is preserved. Root verified210sources+11inputs and released one additive bound-only
build under `target/e6-dual-release-qualification-2400`, source-freeze
67dd0587db9bfaaf2784c339048b99e45d83978665de7e5a9fecf83b5053c6ce,
same source/profile/isolated caches/CPU18,19/2jobs. New bounds2520total/90cleanup/
2400Cargo, then SAME focused24 only on actual successful artifact. No native
capture replacement or debug-job cancellation follows automatically. E6 agent
owned this build/source lease; existing debug96GiB pair was unchanged.

After both96GiB attempts returned the same mathematical guard failure, root
authorized restoration of the existing full-Taylor-certified exact-zero dispatch.
The old-skip optimized build was explicitly stopped through its owned controller
before edits: receipt3ddb5ffab22164f2351466e0deb790cf217fa26ddea149b42e1614612812b24a,
387.255s, SIGTERM, cleanupempty/groupgone and all210guards unchanged before edits.
E6 agent now owns only execution.rs/quadrature_context_tests.rs for removing the
skip and replacing its old implementation-specific regression with restored
profile/context assertions, including a null-primal/nonzero-tangent neighbor.
No new algebra or numerical guard changes. A fresh optimized artifact and focused
tests must pass before reviewed full captures; real regression cap may be600s /
focused1200total60cleanup because restoring specialization adds preparation.

Restoration is now applied/source-reviewed, not yet numerically qualified:
execution.rs4f673f8c89f46445ce62998eedd86729185c69f74c139d4dbdb172e2e716051b
removes only the15-line skip. Regression source1bfa6b75dfeb934d389fe746cbe5ff6713cc61115510d2e01597228780de9a43
requires actual full-Taylor profile lane plus generic nonnull and primal-null/
nonzero-tangent neighbors, complete arrays/metadata, owned contexts and anchors.
Root and independent finish audit cleared source. New210freeze
`target/e6-dual-restored-zero-profile-release-qualification/source-freeze.json`,
SHA2f16ec6572f4e0b58bc67843cdae80496b472da5f902b8fea3b4206cc2336e28,
all210sources+11inputs verified. Build ACTIVE session49946 CPU18/19, sameopt3/
isolatedcaches/2520-90-2400. Only on actual build pass run24 focused tests,
1200total/60cleanup/60ordinary/600restoredtest CPU24. Source/Cargo lease remains
with E6 owner. No full capture released. Future captures must correctly pin
requestedPV→actualIBP610 and legitimate profile/analytic rows rather than the
obsolete old actualPV/zeroanalytic/skip-note requirements; all2520array/DD/
strictpair/owningcontext/lifecycle gates are unchanged.

The restored opt3 build PASSED490.007s: receipt4dc4f4d12abb7c4dbef376e5e6e9f66792638824c73e8482e3f40cd1ab8fca24,
artifactbf65001714cc6504ec8cc076352dbcb4930eea5b4373600f75df4cf9145635f1;
root rehashed actual product/event (opt3/testtrue; compiler debug/overflow checks
false, mandatory numerical admission guards unchanged). Focused23passed,
but new regression initially FAILED17.81s, receiptb804684b44b9329e891f7220eb493135dee05bcd4406b0ff1f4ef5831f5ac73b:
the added third external-time-shift jet was refused by E6 NonConvergent/order128/
error6.24594e-9. This is a numerical refusal of an analytic positive-F germ, not
a mathematical nonanalyticity. First two ordinary scalar lanes succeeded. Preserve
the failed expectation and old1bfa source; no guard/tolerance was relaxed.

Root/audit approved TEST-ONLY correctiond4bc4ae0: retain firsttwo complete success
arrays/contexts/anchors; third must match ordinary NonConvergent diagnostic and
have Err output/context/envelope, analyticfalse/noextra zero profile. Production
execution4f673 unchanged. New210freeze
`target/e6-dual-restored-zero-profile-test-correction/source-freeze.json`,
SHA4aee3b7450b73ebff558b2aaa821f83226a64a039c0fbd8f5b2dd83306551fe5;
root verified210+15inputs/onlytestdiff. BUILD ACTIVE session23526CPU18/19,
2520-90-2400, then same24/1200-60/600realtest onactualpass. E6 owner holds lease.
Python refresh agent prepares same10installedsmokes in isolated cache onCPU20/21,
waiting matching actualcore receipt; no Python build/release yet.

New nonfinite capsule SEALED/rootdefault+independent26fileminimalPASS:
`validation/e6-dual-postfix-nonfinite-boundaries/manifest.json`,792765c1ca3b22cf20db81fd1a5d4eb6a25c4e3d37c9fc09bd1fef154b7d8ead.
Own103983B/26objects/311aliases, inherited78fb→403/DD/no source duplication;
526nonbinaryguards/2originalfailedaudits replay. Rootportableproof3bde0ce9.
Final conditional templates preserve all446c112 records and add exactlytwo
archived failedfirstcall snapshots (ten no-array attempts total, not ten zero-call
attempts); no productionreport change yet. Current capsule is uncommitted.

Final mechanism gate b2a3fbe923a7407476e85f348c23f90d830654928c554271b4e41bc08b478f7a
and firstcallwrappera1da5b65 independently reviewed: exactgood-parent×terminal
origins, all9emptyruntimecounterszero, trueAnalyticownroute/fullQcontexts, all2520/
DD/strictpair unchanged. Proposal skeleton208f973b is INCOMPLETE with noartifact/
focused hash and no release. First128 sequential192GiB/7200 is only a proposed
engineering bound; requires actual24/headroom/rootreview, sufficiency unproved.

Source-hashed algorithm-scope notes under finaltarget (md3b476497/json53f9c0fc)
explain why observedPV-only h1 capture exercises the same successful numerical
recipe as Hybrid: AffinePower→N5generalized0PV/610IBP→actualIbp, no Hybrid
numericfallback foractualIbp. Keep unrun defaultHybrid invocation flag false;
do not invent a new Hybrid capture. Existing separate Hybrid/h0 cells remain.

## Live continuation after pushed `dfcae98` — numerical memory boundary

Previous ordinary-closure publication is
`dfcae98b110fdd47647bb95dd82cf439db53c690`,63-pagePDF
`95fc3559…`,443rawrecords;395bindings/364JSON and root byte-identical rebuild/
visualreview passed. Narrowguard/source24+currentwheelccee and allordinary
N6R6dots/D2/D6 are committed/pushed. No further source change is planned.

The initial post-fix2400-native attempts are
TERMINAL TIMEOUT, not math failures:128receipte0f1f2cc(total2414.769/native2408.176),
2567b205d15(total2415.011/native2407.975). Both finished generic constructors
(1777.993/1786.698s, numericmapping427.52/424.34s), then nofirstcallreturned by
the fixed deadline. No allocation/kernel/assertionerror, all240bindingsunchanged,
SIGTERM−15/cleanupempty/groupgone. AS62.24GiB remained below64. Preserve these
new failures additively; do not mutate sourcecapsule403dfee6, which retains only
the four earlier memory failures. Partial failure audits/summaries are retained.

Root reviewed and explicitly released bound-only ADDITIVE4200-native/4380total/
120cleanup captures, same64GiB/accuracy53/originalSAMPLE/artifact4f430732/source210:
-128: `target/e6-dual-post-fix-full-sample-128-4200`, protocolfdf1b0f1,
  CPU28,unifiedsession34440.
-256: `target/e6-dual-post-fix-full-sample-256-4200`, protocol0984b306,
  CPU30,unifiedsession41298.
Both root-release.json verified248immutable inputs. `e6_precision_failure_research`
owns both autonomous lifecycles. Both generic child constructors completed
(128:1525.95s,256:1534.36s), with no second optional profile. Float256 then FAILED
during the first numerical batch: receipt `d74f5a15…`, native2912.259s /
total2920.615s, SIGABRT−6, GMP realloc16→40bytes, all250terminalbindings unchanged,
cleanupempty/groupgone, ZERO calls/arrays. Exact later numerical substage is
not exported. Float128 also FAILED at64GiB: receipt `cc0afb22…`, native3733.628s /
total3741.261s, SIGABRT−6, GMPallocate16bytes, all250terminalbindings unchanged,
cleanupempty/groupgone, ZERO calls/arrays. Both4200sessions are terminal.

Root checked actual host/cgroup headroom (MemAvailable~254–284GiB; actual user/
session memory.max andmemory.high=max). Root independently verified262bindings
and released ONE additive Float25696GiB/7200native/7380total/120cleanup packet:
`target/e6-dual-post-fix-full-sample-256-96g-7200`, protocol
`640c0afd61a9901f72a21931235468968811eba7cc7e826e38abd8a43338060a`, CPU30.
`root-release.json` written; launched CPU30/unifiedsession92862.
After128 actualfailure, root verified262bindings/released matching12896GiB packet:
`target/e6-dual-post-fix-full-sample-128-96g-7200`, protocol
`74f304c7fe9c3587fbfa089242298a328a7fce7a400ce7f82a664e81d5ce353a`,
CPU28/unifiedsession44049. BOTH96GiB jobs ACTIVE, owned byE6researchagent.
Latest08:31UTC:256elapsed1746s/firstnumericalbatch/RSS35.3GiB/AS62.22GiB,
genericchildconstructor1460.57s;128elapsed1120s/exactrecurrenceevalprep/
RSS30.3GiB/AS46.3GiB, clean identities/noerrors/noarrays. Same frozen4f430732 /source210/
originalfixture/requested53/allguards, no build/source/math change or automaticretry.
Preserve the numerical-stage failure additively; do not mutate source403 capsule.
Immediate each-precision full2520×bothDDrules comparisons must
follow actualarray+mechanismgates; allsix complete2520gates, exact input/odd-invalid/
empty/recovery/owningQ contexts and strictsame-source2^-80 remain required.
No accuracy/branch/domain/error guard or core source changed for the longer run.

New saved-only packaging is SEALED/PASS. IndependentDDanchor is
`validation/e6-dual-rank6-mass-second-defining-integral`, manifestdd499b4a,
3.01MB/36ownobjects/465aliases/210compiledsources208borrowed; defaultandminimal
11-file replay pass. Resource capsule
`validation/e6-dual-postfix-resource-boundaries`, manifest78fb3bc3,173130B/
49owntextobjects/296aliases, directlyborrows403source andanchor'seightoraclefiles;
fouractualfailedmechanismaudits/976nonbinaryguards reproduce, minimal22filePASS.
Neither capsule duplicates source/DD bodies or archivesactive96capturedata.
Root independentlyreplayedBOTHcapsules. Finalfullpasscapsule shouldinheritboth.

The resource checkpoint adds one report page:64-pagePDF5e011ac9,Typst6911e474,
446rawrecords/398bindings/367JSON; all443priorrawvalues and54historicalproof
fields exact. Rootrebuildbyteidentical/visualpages1–2/payloadscan19files85archive
members18.22MBPASS. Rootreview
`docs/data/rank6-dual-resource-boundaries-root-review-20261026.json`,bb18300a.
No newTIRustarray/integral/performancequalification; goalfalse. The independent
reference is newlyARCHIVED, notnewlyevaluated. Sourceandallmath unchanged.
Use gitlogforcurrentpublicationcommit; finalconditionaltemplates mustrebase
afterthisresourcecheckpointispushed, preservingall446records/nestedhistory.
Read-only memory model note `target/rank6-numeric-memory-layout-review-20261026.md`
SHA0fdce80b: batch scratch scales B*(root+allsharedfunctionslots), E6boxes5*R*order,
Floatslots initialized1bitthenfirsttouchprecisionexpandslimbs; boundedstorage,
NOTa96GiBceilingproofor exactfaultattribution. FixedD0requestwindows futureoption
onlyiflargerfrozencapturesalsofail; no patch/extraevaluationdone.

Owner prepared ONE bounded read-only2s perf sample at actual numerical stage
(~2000s), to avoid another blind budget change if needed. Existing perf:
`/nix/store/wizn21b9virxqcnm4n89b09pgqkaxfn3-perf-linux-7.2.5/bin/perf`,
SHAf97ebe85, kernel6.18.45/perf_event_paranoid2. PID/start/exe guard,49Hzusercpu-clock,
DWARF4096stack; noenvreads/sourceedits/install/escalation, no activejobsignal.
One actualFloat1282s attachment at numerical stage PASSED:96userCPUsamples,
0lost, nativeidentityunchanged; Float256unsampled. Initialofflineaddr2line
report hit25s cap; additive --no-inline offline report PASSED1.223s,06a27a0c,
under target/post-fix-e6-read-only-sampling/Float128-sample-001. Stack shows
SymbolicaBatchExpressionEvaluator andcomplexMPFRmultiply/clone/allocation plus
complexlog/polylognumericwork.4096Bytestacksstopatbatchevaluator: no whole-run
dominance or ordinaryD0-vs-E6D0 attribution. No native signal/change; allprofiler
groupscleaned. Rawperftrace f762da96 isTARGETONLY, omitfrompubliccapsule.

Finalconditional report templates rebased todfcae98/all443raw with exact current
publicationproofhistory preserved; current_rank6_report is idle until finalcaps
pass. READMEfinaldraft target/rank6-final-feature-report-20261026/README.final.draft.md
remains unapplied. Finalcheckpointname rank6-functional-completion-20261026.
The ready DDanchor(replay461859d5,2520rulepairmax2.45e-15) is unchanged. All leases
are free; only the two frozen native captures run. Goal remains OPEN; existing
goaltoolstatusBLOCKED is stale after userresume and cannot be manually setactive.
Do not markcompleteuntilactualremaininggates pass, reportupdated/committed/pushed,
then stop as requested. No rank7/performancegrid additions.

## Current milestone: ordinary rank-six closure and Dual route repair

This checkpoint extends pushed `fccffab` and publishes a63-page benchmark PDF,
SHA `95fc3559…`, with443total raw records and all437 prior values/timings and
nested history preserved. Root rebuilt it byte-identically, visually inspected
pages1–3, checked395bindings/364JSON snapshots and replayed the source24 capsule
as well as the ordinary/current-wheel capsules. The report records complete
ordinary six-point rank-six dots/D2/D6 checks, current wheel refresh, the narrow
optional-zero-profile route fix, its24 focused passes and retained failures.
It is an INTERMEDIATE milestone: the actual fullN6/R6 quadrature-Dual128/256
arrays, their ready independent-anchor comparison and final publication remain
open. No performance claim or arbitrary-point qualification is added.

After that report's cutoff, every ordinary six-point/rank-six followup is now
sealed and passing. The common dotted/D2 capsule is
`validation/ordinary-gh-n6-generalized-requalification`, manifest `185b9372…`,
with independent portable review `86f0b276…`: all840 reference coefficients per
precision and840 strict cross-source pairs per case. The positive-shift D6
capsule is `validation/ordinary-gh-n6-d6-requalification`, manifest `1386c4ce…`:
all840 coefficients against both defining-integral rules,840 strict cross-source
pairs (maximum1.77715e-26), and5332 independent saved gates. Historical failed
predecessors remain unchanged. Root replayed both capsules successfully.

The current wheel is also sealed: `validation/python-interface/final-rank6-wheel-smoke`,
manifest `5ef739c7…`, wheel `93cea129…`, native `8978f06f…`. Actual build270.195s,
204selected source bodies including coherent128, native owner/hash import and
exact10 installed cases allpass, seven owned process groups gone. The N6
frontend lifecycle smoke is rank1; mathematical rank6 evidence is separate.
Root saved replay passes2614bindings. No new notebook/typing/performance claim.

Only actual full-Dual imported-quadrature qualification remains numerical work.
The independent original-SAMPLE DD anchor is COMPLETE: receipt `ae85d933…`,
399.881s,86272points/337batches/11windows/both5880-field arrays. Independent
stream replay `461859d5…` verifies229bindings,229363 exactlimb witnesses,
11windows andvolume1/24. All2520 parent rule differences pass1e-10, max2.44997e-15
(`6cabc120…`). Raw UV/IR residues remain unprojected. This anchor is ready for
comparison; it has not yet qualified a successful full-Dual reducer array.

The original explicitPV128 attempt at64GiB FAILED: native1854.211s/total1861.935s,
malloc84226bytes, receipt `fb364156…`,193pins unchanged/groupgone/cleanupempty,
zero completed calls. IMPORTANT CORRECTION: the second12318-alias pass after
the generic12441-alias preparation is a runtime OPTIONAL EXACT-ZERO invariant
specialization, not established Hybrid fallback. SAMPLE has three null
(e_time−e_space)^2 separations. `execution.rs:1855–1914` groups the full-Taylor
zero distances and prepares a specialized IBP recipe while retaining the generic
one; savedlog673/674 show numericmapping/genericchild complete before this pass.
The earlier Hybrid64GiB attribution is therefore source-inferred and must be
corrected precisely rather than promoted as fact. All earlier failures remain.

OriginalPV256 CPU26/session89346/group2117393 continues unchanged to terminal
under protocoldbe8736c,64GiB/2580-120-2400. Do not mutate or duplicate it.
Root chose a minimal production resource-route repair after independent audit:
for genuineE6 quadrature Dual workspaces, retain the prepared generic path
instead of preparing optional zero-invariant profiles. Existing ordinary-retry
contract at1933–1957 establishes optionality. All exact-zero certificates,
master-domain/branch, roundoff, quadrature and coupled-context guards remain.
Unsupported generic charts can still fail honestly. No scalar algebra changes.
The proposed half-time fixture workaround is HELD and will not run.

`rank6_finish_audit` owns sourcelease for the minimal change and one concrete
small N5/D6/Dual regression; it preserves full-Taylor zero classification and
logs only actual skipped groups. On release, `e6_precision_failure_research`
owns new libtest build/freeze,22current focused tests +existingoptional-profile
retry +newregression, then originalSAMPLE R6PV128/256 concurrent captures with
same64GiB bounds and ready independent DD comparisons. Final wheel must then
be rebuilt and its exact10 smoke refreshed against this changed core. No broad
new grid or performance optimization is authorized/required.

Latest post-fix build state: source edits are COMPLETE in execution.rs and
execution/quadrature_context_tests.rs; source proof `e16e68ad…`. The production
change retains full-Taylor zero classification and logs actual skipped groups
before optional-profile cache lookup. Fresh libtest build/freeze passed42.626s,
receipt `05b6c4bc…`, artifact `4f430732…`,210frozen sources, under
`target/e6-dual-optional-zero-profile-qualification/build-001`. Focused-001 has
23PASS and one60s TIMEOUT (new real E6 regression), no assertion failure; all
source guards/cleanup pass. Root authorized one additive run of ONLY that test
onCPU24 with300total/30cleanup/240native; no source changes or23-test rerun.
The fresh post-fix wheel build also passed102.495s, receipt `023178c3…`,205source
context `acaf6c68…`, under `target/python-final-rank6-wheel-refresh`. All source/
Cargo leases are free. Wheel exact10 smoke usesCPU22; e6owner prepares new
originalSAMPLE128/256 packets from4f430732/05b6c4bc (same64GiB/53target/bounds).
Old explicitPV256 has now hit the same optional-profile64GiB malloc limit
(48244B/noarrays); its final immutable receipt/cleanup is being collected.

CURRENT RUNNING QUALIFICATION (supersedes older pending lines): all24 focused
post-fix tests now PASS. The one new real E6 regression completed146.707s under
the additive bound, receipt `984c361b…`; earlier23passes and60s timeout remain.
Fresh post-fix wheel exact10/ownership/hash also sealedPASS:
`validation/python-interface/final-rank6-wheel-refresh-smoke`, manifest
`ccee07d6…`,2674saved checks,205selected/128coherent-of-actual210, wheeldd84b465/
native2220aa0b. Root saved replay passes. No source/Cargo lease remains.

Fresh original-SAMPLE fullR6Q captures are RUNNING from artifact4f430732,
actual05b6build/source210/24gate, same64GiB/2580-120-2400/accuracy53/explicitPV:
128 CPU28/session17385, protocold5b71274;256 CPU30/session6697, protocol0cc087e5.
Both root-release.json files verified238immutable bindings. The saved mechanism
checker additionally requires actual skip diagnostics for BOTHN5terminals and
zero analytic-profile rows, plus all actual owningcontext/nonzerohigherQ gates.
The ready DD anchor is unchanged and six complete2520 numerical comparisons
remain required (fourcore/rules,rulepair,strictsame-source128/256).

Root requested another intermediate report/commit while these25min captures run.
`current_rank6_report/low_n_generalized_capsule` is sealing narrow source24
capsule `validation/e6-dual-optional-zero-profile-core-tests` using direct210map
`81db2d94…` (205borrowed,5ownDDexamplebodies47,527B) and sourceproofe16e68ad.
`current_rank6_report` prepares `rank6-ordinary-closure-and-dual-route-20261026`
intermediate PDF/data, goalfalse, preservingall437raw/nestedhistory, including
ordinary185b/D61386, wheel5ef/ccee and scopedsource24—notactiveR6Qarrays. Root
will independentlyreplay/rebuild/render/scan/commit+push andlinkPDF. Final
completion templates remain conditional; README draft not yet applied.

The HISTORICAL pre-fix source bridge `target/final-dual-current-source-bridge/comparison.json`
(`e37b277f…`) finds128 shared selected old/current sources and125 byte-identical.
All84 reduction and18 master bodies, entire pentagon6 and expressions helpers
are identical; the entire123899-byte box production prefix matches. The only
shared changes are test diagnostics and a dev-only sha2 dependency. No production
mathematical delta or rebuild is required within that declared source scope;
this is not a complete dependency-closure claim. This bridge applies to the
pre-skip wheel5ef and failed originalPV attempts; new post-fix library/wheel
contexts above require their own current-source equality binding.

Final report templates are under `target/rank6-final-feature-report-20261026`.
They preserve437raw records and all nested history. Report owner prepares target
sources only until the final Dual capsules pass. No production source changes,
new rank-seven grid or performance optimization is planned. After actual final
numerical gates: refresh README/HANDOFF/report, independent replay/rebuild/render,
commit+push, link the GitHubPDF, mark the current functional objective complete,
and stop as requested.

## Historical implementation interval after `6a826a0` (published in `fccffab`)

`6a826a0621229da0c8234a488a5a79f4af844dce` is pushed to `origin/main`.
Its 59-page report is byte-reproducible and preserves all 408 prior raw records.
The working tree has a subsequent, uncommitted full-component quadrature-budget
change: all 15 focused tests pass after two recorded test-fixture corrections.
Its actual D6 Dual qualification is held until the ordinary D6 and scalar-master
issues are resolved. The original reference violates a positive-parameter
rank-one/scalar bound. Two independent parameter rules now agree with all210
TIRust moments/all840 native Laurent slots, including20 nonzero UV residues
(saved replay `c113bfee…`). Source inspection identifies an unsafe genuine-D6-to-4D PV
metric-coefficient bridge as a leading explanation for the reference failure.
The scalar-only E6 trace reproduces the separate Cartesian first-dot Float256
failure and localizes lost precision to one box pinch. An ordinary-only,
factor-preserving OneLOop repair passes exact tests, an independent copied-box
value comparison and actual scalar E6 convergence; its
deliberately unchanged Dual program still needs actual qualification.

The private checked-G zero-mass dotted direct fallback is now applied and passes
ordinary mixed/lifecycle and independent reference/IR checks. An attempted jet
in another positive mass is explicitly rejected by the existing conservative
massless C0/D0 derivative chart; that is an unsupported chart, not proof that the
mathematical direction is nonanalytic. The scale-only Dual witness now passes
the full mixed/invalid/empty/recovery schedule and 68 saved exact-reference/
Laurent-scale checks (receipt `4753a710…`), alongside the 34 ordinary gates.
  Its compact durable capsule is sealed. The negative dimension-shift N6
rank-six Float128 case passes all 840 coefficients;
its repaired current Float256 partner now passes all840 reference coefficients and840 strict cross-source old128/new256 pairs (`c5cee76a…`, independent replay `a41c895d…`). The full goal remains open. Generic rank six
is the target, with no fixed point-count ceiling in the denominator-descent
architecture; N<=6 is the finite qualification checkpoint. Performance
optimization and new rank-seven work stay deferred.

Latest continuation receipts (all uncommitted after the published checkpoint):

- `validation/propagated-dual-joint-budget-core-tests`, manifest `c2254d90…`:
  all 15 focused source tests pass. Actual full-Dual imported quadrature is
  still a separate open gate.
- `validation/checked-g-zero-mass-fallback`, manifest `547c78cd…`: sealed
  ordinary and scale-only Dual evidence, 102 exact saved gates. The failed
  other-positive-mass derivative is retained as an unsupported scalar chart.
- `validation/affine-power-generalized-memo-dot-rank6`, manifest `82e48ff8…`:
  old CLI003 first-dot128 passes;256 returns no array after E6 failure.
- `validation/public-n6-rank6-d2-partial-evidence`, manifest `0b23b67e…`:
  old CLI003 D2/128 passes all840 coefficients;256 fails at the same N5 E6
  endpoint `[0,2,3,4,5]` as the first-dot case. Both256 pairs remain unrun.
- `validation/rank6-current-focused-core-tests`, manifest `61d24bf0…`,
  passes all22 focused normal tests in the current core build (`fff28b06…`).
  Its selected128 source bodies and actual emitted Cargo artifact are bound;
  it is not a new complete dependency closure or a numerical D6 witness.
- The ordinary OneLOop repair passes ten exact/existing regressions. Minimal
  copied-row receipt `dcb3feff…` reproduces the legacy8/6-bit output exactly,
  while the repaired output retains254/251 tracked bits. Independent20/32-rule
  value comparison passes at1e-10 (maximum3.55e-15), and the patched scalar E6
  converges at order64/600 D0 rows, estimated error7.39e-27. The sealed capsule
  is `validation/ordinary-complex-box-correlation`, manifest `aec68425…`;
  its separate independent-copy companion is `2f09b8e4…`. Current CLI7005b096
  and masters007c4e33 are frozen by build receipt `faa4ea0e…`. Current256
  first-dot parent remains active on CPU16. D2 current256 completed in1126.854s, all840 reference checks pass (max2.672e-17), all840 strict2^-80 cross-source precision pairs pass (max2.636e-27), and all630 pole controls are exactzero. Its source/date-scoped capsule is being prepared separately.
- The full210 D6 Float128 parameter oracle builds and passes its two exact tests
  (receipt `4fa0077d…`). Actual inventory is945 Gaussian instructions,210
  dynamic outputs and304640 complex slots for256 rows. Its numerical run
  stopped cleanly at the approved first-window cost gate:32768points/63.252s,
  receipt `02da4d7c…`; both final arrays are UNRUN for that attempt. The uniform
  Symbolica DoubleFloat specialization builds and passes both source tests,
  with identical kernels/map/J/slots. Its complete1,297,408-point run takes
  451.053s (`dc2b6d6f…`). Independent saved replay `c113bfee…` verifies all40
  windows/5068 ACKs/exact limb witnesses and all840 native slots per rule:
  maximum1.13558e-11 at order12 and6.67806e-15 at order16. This is numerical
  defining-integral evidence, not a rigorous enclosure or106 achieved bits.
  The durable capsule is sealed as `validation/affine-power-generalized-current-d6-rank6`, manifest `1b2c3b50…`. The current256 D6 partner is now active
  on CPU14 under `target/ordinary-gh-d6-parent-requalification`, protocol
  `686534c7…`, same2580/120/2400 bounds/noAS cap; both canonical rules and all840
  old128/new256 cross-source pairs are required. Original finite remains
  excluded from pass gates and retained diagnostically.
- The SAMPLE-geometry N6/rank6/h1 mass-second Dual witness uses five shifts
  `[0,e_time,e_x,e_y,e_z]` plus a duplicate origin with mass squared26,
  masses `[10,12,15,18,22,26]` andmu3. It needs two regular N5 children and
  avoids adding a new scalar Dual chart. Its32GiB attempt failed during
  reduction-evaluator preparation, after quadrature-evaluator preparation,
  before master evaluation: receipt `561167f1…`, requested allocation
  `10,565,729,856`bytes after about20GiB RSS. The additive64GiB attempt is
  explicitly released at `target/n6-sample-d6-mass-second-witness-64g`,
  protocol `44bfb5cc…`, CPU28, same2580/120/2400 time bounds and frozen core
  artifact `2d3c077b…`. It completed primary preparation, then failed during
  eager independent Hybrid IBP fallback preparation: receipt `fe45802d…`,
  native1774.309s/total1782.089s, malloc48672bytes, last AS62.43GiB/RSS42.01GiB.
  No tensor array or master result was returned; both Hybrid attempts remain
  resource-unqualified. A separately labeled explicit-PV witness now uses the
  identical successful primary calculation and existing public policy to omit
  the eager independent fallback. No source/guard/tolerance changes: protocol
  `0975c4f1…`, root-release.json, same64GiB/2580/120/2400, CPU28, session58827,
  process group1394216. Float256 and independent anchor execution remain held
  until actual128 review. Exact N5 Gaussian/partial-fraction anchor sources
  are prepared, with no numerical qualification yet. The final wheel owner
  has the exclusive Cargo lease for one current wheel build and exact10-case
  isolated smoke; other native jobs consume frozen artifacts only.

Do not promote a completed capture, tracked precision, source proof, cost-stop,
or memory failure into independent numerical agreement. No new rank-seven or
performance-parity campaign is required for the user's functional checkpoint.

## Intermediate ordinary-closure / Dual-route report after `fccffab`

The report checkpoint `docs/data/rank6-ordinary-closure-and-dual-route-20261026.json`
and its readable companion preserve all437 earlier raw records and timings.
The goal remains false/open. Root owns the top current continuation, README,
independent report rebuild/review and publication.

Sealed ordinary parent185b and D6current1386 close the documented ordinary
six-point/rank-six families with complete references and strict CROSS_SOURCE
pairs; no same-build convergence is claimed. Old5ef and current refreshccee
wheel smoke epochs are both retained. The refresh passes exact10 existing
cases with205 selected/128 declared core sources from the actual210 build;
its N6 case is rank one, not full rank-six mathematical qualification.

The source24 capsule preserves the original new-test60s timeout and additive
146.707s exact-test pass. The precise retained64GiB diagnosis is optional
exact-zero specialization after primary N5 completion in Hybrid64/PV128/PV256;
Hybrid32 failed earlier mapping. The generic-route resource fix preserves all
stability/domain guards and actual lane diagnostics. No old failure is erased.

Current full original-SAMPLE R6 quadrature-Dual128/256 runs and independent
core-anchor comparison remain pending; live captures and unsealed DD packets
are excluded from report bindings. Half-time replacement stays held/unrun.
Only those actual gates and coherent final publication remain completion work;
wider chart/N/Legacy/rescue and performance entries remain qualification limits.

## Report checkpoint after published `6a826a0`

The next report increment is prepared in `docs/data/rank6-generalized-followup-20261026.json`
and its companion under `docs/checkpoints/`. It preserves all 425 earlier raw
records/timings. Root owns the top continuation above, source staging, final
independent report review and publication. No new performance or fresh wheel
qualification is included; the full goal remains active.

Seven scoped capsule families now supplement the publication: current22 focused
core tests, the overlapping15-test joint-budget source history, zero-mass
checked-G rank zero ordinary/scale-only Dual 102 gates, old-CLI partial Cartesian-dot
and D2 outcomes, ordinary D0/E6 correlation repair, and independent complete D6
parameter-oracle adjudication. Current source tests do not qualify full numerical
Dual or unfinished parent precision pairs. The old dot/D2 Float256 convergence failures
remain failed; newer repaired runs remain separate until their own gates pass.

The defining-integral D6 oracle agrees with all 210 moments/840 Laurent slots at
both fixed rules (order12 max≈1.13558e-11; order16 max≈6.67806e-15). All840
rule differences pass1e-10. The original 210finite mismatches remain archived,
but positive-F bounds and the independent oracle reject those reference finite
values, rather than establish a Rust defining-integral failure. MPFR cost-stop,
finite mismatch and all pole/UV controls remain preserved. No D6 Float256 pair,
rigorous enclosure, actual full-Dual integral or performance claim follows.
Completion gates are the repaired parent precision pairs, a genuine quadrature-
bearing Dual with independent anchor, and final coherent current release.
Broader domains remain scoped limits rather than blanket new-grid requirements.

The ordinary-only box repair has 10 regressions, one independent20/32 D0 value
anchor and actual E6 convergence recovery at order 64/600 rows. Its old failed
trace remains saved, and the full seeded-Dual helper stays selected. Do not
infer independent E6 accuracy or parent-tensor qualification from convergence.
The report owner ran saved replay and document builds only.

## Current state, 26 October 2026 — supersedes older status below

### Latest continuation after published `6fb7d3f`

The next coherent feature checkpoint is prepared against `6fb7d3f`; root owns
its commit and push. The report adds three compact coverage pages while retaining
all 408 earlier raw result values and timings. The new checkpoint is
`docs/data/rank6-feature-boundary-20261026.json` with a readable companion under
`docs/checkpoints/`. **The full goal is not complete.** Each capsule qualifies
its own actual source context; working-tree similarity does not promote it.

The publication's source boundary is the central capsule
`validation/checked-g-generalized-core-tests`, manifest `ee4bc172…`: 158 canonical
bodies, nine additional source versions and four compile contexts. All 120
strict distinct normal tests pass as 95+25 across a test-only bridge. Six initial
compiler errors and the stale Legacy engine-tag observer failure remain failed.
The staged production prefix matches the frozen CLI003/wheel004 context; root
may place a future candidate in the worktree without including it in this commit.

New sealed positive and negative evidence:

- `validation/public-n6-affine-mixed-dual`, manifest `f025d7d4…`: public N6
  rank-six mixed-Dual PV Float128/256 passes 30,240 quad comparisons and 75,600
  parent/child precision pairs below `2^-80`, including 60,480 child pairs.
  Per precision there are 15 complete contexts and 12 actual N5 children,
  explicitly KnownNoQuadrature. Six Taylor components cover first squared mass
  and simultaneous dilation; requested accuracy remains53. The older private
  ThinSix allocation failures remain failed.
- `validation/public-n6-hybrid-typed-dual`, manifest `39d2f234…`: unchanged
  default Hybrid independently passes the same full counts. Typed DoubleFloat
  adds 15,120 quad, 15,120 Float128-control and 5,040 scale checks. Corrected
  f64 hits the real pv_node_12/Taylor2 cancellation guard, returns null arrays
  and has zero numeric comparisons. Its initial input-size refusal is a
  separate retained pre-reduction failure. This is repacking existing Symbolica
  Dual values, not a new typed convenience API or a hidden-context export.
- `validation/public-n6-rank6-mass-fd`, manifest `2cf715c8…`: five ordinary
  Float384 samples form two three-point central stencils sharing the center,
  at 2^-12 and 2^-18. All 3,360 fine-step accuracy gates pass1e-10; 6,720
  records include coarse diagnostics. There are 420 resolved transitions and
  1,260 explicit floor/unresolved transitions. The floor classifies convergence
  without relaxing accuracy. The original Symbolica0^0 weight-preparation
  failure and literal-one constant-row repair are retained.
- `validation/low-n-generalized-rank6-memo-dot-pair`, manifest `fe04df1f…`:
  actual N5 rank-six first-dot Float128/256 passes19,740 gates (5,040 quad,
  5,040 mass-Taylor, 2,520 precision, 7,140 pole/duplicate/scale). Its actual
  119-source memo context remains separate from later G/Mass sources. Two
  original short timeouts and three older unrun states are retained. No N5
  triple-power qualification follows.
- `validation/legacy-noop-rank6`, manifest `f3f4a469…`: 26 focused tests,
  N5 rank-six PV/unit/h0 Float128/256, 1,680 quad slots and840 strict pairs
  pass. Its separate151-source context and old OOM/compile/pre-launch failures
  remain explicit. This is neither full Legacy nor dots/h/Dual/N6/rescue closure.
- `validation/python-interface/facade-wheel-004`, manifest `44b4552b…`:65
  installed API cases, seven marimo runtime cases and clean consumer typing
  pass. All158 declared current core bodies match within201 selected files;
  portable replay passes1,575 checks. This is a selected source context, not
  the entire dependency closure. Historical wheel004 gallery/C0D0 evidence is
  a different older context and must not be conflated with this facade wheel.
- `validation/checked-g-generalized-numeric`, manifest `2a32b5aa…` with
  qualification `fec2dab0…`: current N6 rank-six default-Hybrid unit and
  first-dot h0 Float128/256 all pass3,360 original-quad slots and1,680 strict
  precision pairs. Worst reference error≈9.11420e-16; pair≈6.86405e-30.
  The supplied data remain authoritative. The complete auxiliary×user proof
  guard is source-enforced and covered by13 focused normal tests. The frozen
  CLI observes three dotted compound-acceptance messages per precision but
  does NOT export inner numeric proof counters. Do not invent those counts or
  infer public G Dual/full-domain qualification.

Important open outcomes at this cutoff:

- Current Cartesian N6 first-dot Float128 passes all840 reference slots, but
  Float256 fails E6NonConvergent at leaf30/order128/reportederror2.3794e-4,
  with no parent tensor array or precision pair. Root/owner retained receipt
  `3671d809…`; no retry or tolerance/limit change was authorized. A partial
  capsule is being prepared separately; the report records only the terminal
  observation and does not bind an unsealed target path as a pass.
- Current D6/unit Float128 reports native success and returns an array, but
  independent reference validation fails all210 finite slots (maximum
  normalized error≈1.4213). The420 IR,126 lower-rank UV and84 rank-six UV
  controls pass. Float256 is UNRUN because the required128 gate failed. Root
  retains receipt `c7df7491…`; owner is diagnosing, with no promotion or retry.
  Earlier sealed810-second preparation failure remains failed. Do not claim
  a D6 precision pair or84 nonzero UV coefficients:84 counts exact-degree-six
  tensor slots, whereas the reference has20 macroscopic all-even UV entries
  and64 tiny odd residuals.
- Default Cartesian/formal-G and ordinary Cartesian affine children keep
  direct recurrence. `new_mass_taylor` is explicit with a nonzero differentiated-
  mass domain. The private checked-G regular hook still lacks a same-requested-
  domain fallback when its mass differentiation would reject zero mass.
- Imported full-Dual propagated-quadrature remains guarded in the publication.
  A proposed extension is TARGET-only/future worktree context under separate
  review; root keeps it outside the current commit. Do not promote its source
  proposal or focused future tests into completed numerical coverage.
- Broader positive powers/dimension shifts, accepted Legacy/rescue, exceptional
  cases and remaining numeric/method/physical closure stay open. No new rank-seven
  or performance campaign is needed. Consistency below2^-80 is not a certificate
  of achieved working precision.

The following sections retain the prior session history and may describe jobs
that were pending at those earlier checkpoints. The latest status above takes
precedence. No report-owner native integral or benchmark run was added.

The unbudgeted goal is active. The user prioritizes feature-complete, validated
generic rank-six reduction, including dots; further performance optimization is
not a completion requirement. There is no mathematical point-count ceiling in
the new Cartesian denominator-descent facade. N≤6 remains the concrete numerical
checkpoint, not an arbitrary-N or all-singular-kinematics claim. Do not begin new
rank-seven stress campaigns. Finish the agreed features, update the report,
commit/push, then mark the goal complete and stop only when the remaining gates
actually pass.

Published `d4d2f9e` contains the previous 50-page report and saved validation.
The next report and coherent implementation milestone are being prepared. These
new sealed capsules contain the current positive and negative evidence:

- `validation/python-interface/facade-wheel-003`: 64 installed API cases, seven
  marimo runtime cases and consumer typing pass; 1,042 saved replay checks.
  The exact source context is `e873a0be…`. The external toolchain removal,
  offline package-cache miss and aggregate test timeout remain retained; only
  the unfinished test was completed separately.
- `validation/current-cli-physical-integration`: 14 current CLI tests, 2,800
  original-IREGI tensor comparisons, 12 MadLoop coefficient comparisons and
  568 exact `2^-80` precision-pair gates pass. Physical graphs are rank four.
  The pentagon at working128/requested53 is correctly rejected by its
  cancellation estimate; independent working256/384 recomputations pass.
- `validation/checked-supplied-g-unit-rank6`: actual checked N6 PV/unit/h0
  full arrays at Float128/256 pass 1,680 quad comparisons and all 840 strict
  precision pairs. This does not qualify generalized G or default Hybrid.
- `validation/affine-power-generalized-current-core`: ordinary N6 rank-six
  Cartesian AffinePower arrays at 128/256 pass 1,680 quad comparisons and 840
  strict precision pairs. Four N5 children execute as one packed batch.
  First-dot and D6 preparation each exceeded the original 810-second bound
  without producing arrays; their partners remain unrun. Child IBP requests
  may use the documented PV completion; do not call this pure IBP.
- `validation/low-n-generalized-rank6-current-cli`: explicit N1/N4 first-dot
  and triple-propagator arrays at 128/256 pass 68,046 main checks plus 20,160
  separate mass-Taylor identities. Direct N5 first-dot preparation also
  exceeded 810 seconds, without an array; three partners remain unrun.

The public AffinePower N6 mixed-Dual Float128 probe has now completed all three
calls and passed its independent saved/reference gate (15,120 parent Taylor
slots, maximum normalized error about `3.08e-15`). All 12 actual child rows have
validated `KnownNoQuadrature` contexts. Its Float256 partner has also passed: 30,240 external parent Taylor comparisons
and 75,600 parent/child precision pairs in total, maximum normalized pair
difference about `5.48e-30`, below `2^-80`. These newest Dual results are not
yet in the current sealed report. Default-Hybrid mixed128 is running; its
256 partner and public N6 typed f64/DoubleFloat coverage remain separate work. The old private mixed-Dual OOM
cases are not superseded retroactively.

Root added only a per-traversal memo to the existing Symbolica primal-amplitude
extraction in `stability.rs`; no formulas, branch choices or numerical budgets
changed. Build/CLI receipt `target/primal-amplitude-memo/build-001/receipt.json`
is `f242c55c…`; six focused tests pass at `tests-002/receipt.json`, `f9bee951…`.
An initial 40-level literal IF stress test exceeded 30 seconds; the retained
record has no internal phase attribution. The representative 10-level test
passes, along with context isolation, inactive-pole batching, nonlinear
rejection and existing branch/normalization checks. The CLI production prefix
is identical across this test-only adjustment. The original bridge had split
at the first helper-level `cfg(test)`; the additive
`test-only-bridge-addendum.json` correctly anchors the actual test module and
verifies the entire production prefix including the memo (`92296f91…`).
One actual N5 first-dot run on the changed CLI is now being checked by
`/root/rank6_completion_map`; inspect its owned receipt before doing any repeat.
The user imposed no 810-second or 32-GiB acceptance requirement. These are
local run guardrails; evidence-based larger preparation budgets can be
appropriate for the unoptimized functional milestone.

All generalized-G and auxiliary mass-Taylor implementation proposals remain
**target-only and unapplied** while this smaller change is investigated. The
main proposals are `target/checked-g-generalized-source`, root's
`target/checked-g-generalized-facade-proposal`, and
`target/lowpoint-dotted-mass-proposal`; root's frontend additions are under
`target/mass-taylor-frontend-proposal`. Coordinate their owners before composing
them. Mass-Taylor extraction must keep auxiliary directions distinct from user
components/sample lanes and transport amplitudes/quadrature errors through a
fresh guard at the dotted integral's mass dimension. A value-only projection is
insufficient. Never bypass the existing full-Dual joint quadrature guard.

The actual accepted no-op Legacy N5 rank-six call aborted at the 32-GiB bound
with a 17.4-GB allocation request before any coefficients; receipt
`target/legacy-n5-rank6-noop/128-001/receipt.json`, `a07f634d…`, and release
`01b2cb92…` retain it. Its 256-bit partner was not run. A target-only lightweight proof of
unchanged snapping before preparing a preserved-input engine is being designed;
geometry-changing highpoint snaps remain explicitly rejected by the current
contract. This is an unresolved functional/resource gate, not a successful
array validation.

All commands still run from `/shared/external/iregi/iregi-rust`. An external
Nix-store cleanup removed the previous compiler during a wheel build. Exact
Rust/Cargo1.99 and Python3.13.12 were restored and GC-rooted locally. Source
`.local/tirust-current-toolchain-roots/env.sh` **before** `scripts/env.sh` for
the restored pins; the Symbolica license environment remains in its ignored
local file and must not be printed or committed. Rustfmt1.99 is also rooted
there. Do not race Cargo or mutate another owner's live source snapshot;
numerical jobs use owned binaries and immutable copies, so documentation-only
commits and unrelated target proposals do not require stopping them.

Prepared on **24 October 2026**, at the user's explicit request to wind down,
consolidate on `main`, update the report, preserve context and stop. This is a
restart document, not a claim that the original implementation goal is complete.

**Resumed on 25 October 2026 at the user's request.** The pause statements below
describe this handoff's historical checkpoint. Subsequent work is recorded in
[the resumed checkpoint](docs/checkpoints/resume-20261025.md); the full objective
is active and unchanged.

**Current correctness milestone after `d9a2642`, 25 October 2026.** See [the complex/Python checkpoint](docs/checkpoints/complex-python-20261025.md). The corrected Python wheel passes sixty installed tests and typing; the independent complex parameter oracle and seven N6 generalized quad-reference arrays have portable, independently checked capsules. The oracle rejects a stable but incorrect original N4 complex zero. A fresh fourteen-case rank-six CLI cohort passes twelve complete cases and exposes a genuine collinear N3 nonfinite PV/IBP gap in two cases; the full functional objective remains open. N2 numeric packaging, scoped private N6 mass-Dual evidence, ordinary f64 rescue, remaining generalized tensors and supplied-G/Legacy/Hybrid work must not be mistaken for completed public coverage. Historical report timings remain unchanged. No blind rerun of the failed collinear or resource-limited cases is authorized; inspect the concrete source tasks and owned receipts in the checkpoint.

**Subsequent rank-six validation milestone after `2134a0d`.** The
[rank-six checkpoint](docs/checkpoints/rank6-feature-20261025.md) adds compact
portable evidence for numeric types, generalized reference anchors, actual
six-point Hybrid and precision rescue, threshold/complex cases and a newer
collinear triangle core fix. The N1–N5 rank-six numeric cohort has 21 successful
captures, three preserved precision guards and 446,880 saved complex checks.
The newer collinear core passes complete Float128/256 arrays and a selected
first-dilation Dual against the independent order48 anchor; those numerical
arrays were asserted in Rust, not exported for saved recomputation. It also
records 16 affine, ten component and four checked-facade test passes, alongside
two symbolic general-N preparation timeouts. Earlier CLI failures remain
unchanged. Each capsule reconstructs its own declared source context; this
validation milestone does not equate those contexts to unfinished later code.

Current source work is a dynamic constant-denominator/power DAG that descends
point count before the low-point planner, preserving tensor rank, positive
powers and dimension shift. A separate packed checked-G path addresses the
inline six-point rank-six preparation failure. Both require actual numerical
qualification before full acceptance. The older private mixed six-point Dual
path still fails a 32-GiB allocation bound after coefficient interning; public
Dual qualification will use the new descent route if it avoids that graph.
Do not restart the old memory or symbolic-stress failures without a concrete
code change. The Symbolica undo-stack storage correction remains a source
proposal, not an applied or tested dependency patch. CLI/PyO3 compile checks and
fourteen CLI tests passed before the new AffinePower enum integration; a fresh
coherent build and final installed wheel remain required. The goal is active,
with performance optimization deferred by the user.

**Earlier published checkpoint: `2ad3cbe`, 25 October 2026.** The report
at that revision has 41 pages. Complete retained massive PV arrays through rank seven pass at N1–N6;
the N6 batch/contraction lifecycle also passes. Retained massless N2–N4 arrays
pass 7,920 independent-reference comparisons with UV/IR poles kept separate.
The shared tensor facade and Symbolica's scoped branch-fact compiler change are
committed. See [the array checkpoint](docs/checkpoints/rank7-arrays-20261025.md)
and [the massless/dispatch checkpoint](docs/checkpoints/rank7-massless-dispatch-20261025.md).
The [shared CLI and IBP checkpoint](docs/checkpoints/rank7-cli-ibp-20261025.md)
adds passing shared CLI integration through
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

The [planar generalized checkpoint](docs/checkpoints/planar-generalized-20261025.md)
updates the report to 43 pages with independently validated planar dots/D6
scalars, additional method coverage, focused resource fixes and the instrumented
N7 failure. The full component-Taylor code remains private pending high-point
numerical qualification.

The latest objective is **all agreed features at generic tensor rank at most
six, including dots**. The 210 Cartesian moments do not depend on propagator
count. **N≤6 is the current concrete validation checkpoint, not a proof of
arbitrary-N denominator descent or singular-case handling.** Higher-N descent
requires checked implementation and finite representative validation; do not
simply remove inner-planner resource guards. Once the full objective is tested,
update the report, commit and push, mark it achieved and stop; optimization is
separate work. Retain existing higher-rank evidence and explain N7/rank7 stress
challenges without starting new stress cohorts. The objective remains active.
Generalized tensors, full-tensor Dual/numeric and supplied-G/legacy paths are
still being qualified; the corrected Python wheel has its scoped sixty-test
qualification. Consult owned receipts before restarting any job.

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
