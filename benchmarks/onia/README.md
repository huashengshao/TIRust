# S-wave quarkonium one-loop integrals: TIRust against original IREGI

This suite tests the one-loop tensor integrals of S-wave quarkonium production,
where the heavy quark and antiquark carry the same momentum, p_c = p_c~ = P/2
with P^2 = 4 m_c^2, in two processes:

| process | kinematics | topologies (N = 5 / 4 / 3 / 2) | points |
|---|---|---|---|
| `gg_QQ`: g g > c c~ | 2 -> 1, momenta span 2 dimensions | – / 4 / 8 / 8, 20 in all | 100 |
| `gg_QQg`: g g > c c~ g | 2 -> 2, momenta span 3 dimensions | 12 / 36 / 32 / 18, 98 in all | 200 |

With p_c = p_c~, every Gram determinant of an integral with both heavy-quark
legs (or with all external momenta of a pentagon) vanishes exactly. For the
Coulomb-exchange integrals (a gluon propagator between the c and c~ legs) the
modified Cayley determinant vanishes too: their propagators are linearly
dependent, 2 D_gluon = D_c + D_c~.

**Companion PR.** TIRust's Coulomb-exchange support (`direct/plan.rs`,
`direct/exec.rs`) and the regression test `crates/iregi-rust/tests/onia.rs`
with its fixtures are huashengshao/TIRust#33, to be merged before this
suite. The results below were measured with it; without it, main refuses
the Coulomb-exchange lanes (typed), and the scripts here still run. The
commits cited below (75b24ef, c65727b, 25e3f31) are on branch `hs/onia`, the
history of huashengshao/TIRust#32, kept for these references.

## Topologies

`make_points.py` enumerates every cyclic ordering of the external legs (up to
rotations and reflection). For each ordering it takes every grouping of
consecutive legs into N >= 2 loop vertices (a group is a tree attached to the
loop). For each grouping it takes every propagator mass pattern that QCD
vertices allow:
- a vertex that attaches an odd number of heavy quarks joins a heavy-quark
  propagator (H, mass m_c) and a massless one (L: gluon);
- every other vertex joins two propagators of the same kind (H H, or L L:
  gluon, ghost or light quark).

Integrals that coincide because c and c~ have the same momentum are counted
once. The rank is the maximal one of a renormalizable gauge theory, rank = N.
Topology names give N, the masses of propagators 0..N-1 and the groups:
`N4-HHLH-g2|Q|Q|g1` has the groups (g2), (c), (c~), (g1) and propagator i
between group i-1 and group i. Its propagator 2, between c and c~, is the
massless Coulomb gluon.

## Exact lattice points

Every code receives the same binary64 inputs, and they are exact:
- **Massless momenta.** Each is an integer lightlike vector, a rational point
  of the sphere, (a^2+b^2+c^2, 2ac, 2bc, a^2+b^2-c^2), times an integer.
- **Heavy-quark momenta.** P is the exact sum of the gluon momenta, so
  p_c = p_c~ = P/2 and m_c^2 = P^2/4 hold exactly.
- **Scale.** Every shift component, and every difference of two shifts, is
  below 2^24 lattice units. All the lattice units are scaled by one power of
  two.

So every invariant and scalar product of the shifts is exact in binary64,
however a code forms it. On-shell legs are exactly on shell, and the vanishing
Gram and Cayley determinants are exactly zero at the level of the invariants.
No code needs its on-shell snapping. `make_points.py` checks these conditions
for every point and topology, and `run.py` checks again that the shifts are
exact.

The two kinds of frame are:
- **`cm`** (first half of each process's points): the beams lie along +-z.
  - For `gg_QQ`, the system is boosted along z by a random rapidity in [-3, 3]; the first point is at rest.
  - For `gg_QQg`, the gluon goes in a random direction, with sqrt(s) log-uniform in [6, 60] GeV and gluon p_T > 1 GeV.
- **`generic`**: a physical configuration rotated and boosted at random, then put on the lattice.

The charm mass is near 1.5 GeV but differs per point, because the lattice
fixes P^2. Throughout, mu = 3 GeV.

## Codes and reference

`run.py` runs the following codes on every lane:

| name | what |
|---|---|
| `tirust` | TIRust public binary64 default (direct engine; refused lanes rerun at higher precision automatically) |
| `tirust-binary64` | TIRust with the precision rescue disabled; a refused lane is scored by its best-effort array and flagged |
| `iregi-ibp`, `iregi-pv` | original IREGI (MadGraph's vendored copy), binary64, IBP and PV modes, default on-shell snapping as in MadLoop |
| `quad-ibp`, `quad-pv` | original IREGI built in quadruple precision |
| `direct256` | the TIRust direct engine at 256 bits (accepting conditioning down to 2^-216) |
| `symbolic256` | (`--symbolic256`) the TIRust facade at 256 bits, which runs the symbolic engine |

The reference of a lane is the first of these that holds:
0. **Exact denominator identity** (`affine_reference.py`), for an integral
   whose shifts are affinely dependent. Write sum_i c_i = 0 and
   sum_i c_i q_i = 0. Then sum_i c_i D_i = K is a constant, and for K != 0
   the integral is exactly sum_i (c_i/K) times its pinches. This is applied
   recursively until the shifts are independent. Each non-degenerate
   integral is evaluated by TIRust at 256 bits and accepted only if quad
   IREGI (IBP or PV) or COLLIER confirms it (rule 1b). The coefficients are
   exact rationals of the lattice inputs. This reference is therefore
   independent of how either code handles the degenerate kinematics.
1. **direct256 confirmed by quad IREGI**: agreement with quad IBP or quad PV
   to E/A <= 1e-9.
1b. **direct256 arbitrated by COLLIER**, where direct256 and quad IREGI
   disagree on an integral whose scalar master is IR finite. COLLIER's scalar
   (`collier-scalar`) must agree with direct256's scalar moment to 1e-9 and
   contradict quad IREGI's by more than 1e-6. Both quad modes share IREGI's
   bundled OneLOop 3.4 scalars, so their agreement is not independent.
2. **symbolic256 confirmed by quad IREGI**, the same way.
3. **quad IBP and quad PV agreeing** to 1e-12, unless COLLIER contradicts
   their scalar.
4. **direct256 and symbolic256 agreeing** to 1e-12.

E/A is the whole-array maximum discrepancy (all moments, all four Laurent
slots) divided by the largest entry, with the moments rescaled by
mu^(|alpha|+4-2N), as in the stability study. TIRust is also compared with
binary64 IREGI directly. Quad IREGI is scored as a code only on lanes it
reports stable.

**Coulomb-exchange integrals** (K = 0: their propagators are linearly
dependent, 2 D_gluon = D_c + D_c~). With `--coulomb`, these come first:
- for the triangle, the closed form of `coulomb_closed_form.py`, at every
  rank;
- otherwise, quad IREGI's IBP array where `coulomb_check.py` found it equal to
  1e-12 to the exact partial-fraction combination of doubled-gluon integrals
  from IREGI's general API, a separate code path.

Next comes the exact identity of rule 0. It is carried through the doubled
powers, and its doubled non-degenerate integrals are confirmed by binary64
IREGI's general API, since quad IREGI's general API overflows on them.
`coulomb_check.py` also compares IREGI's scalar C0 with the analytic
v = 0 value.

## Running

From the repository root, after `source scripts/env.sh` and the Symbolica licence:

```bash
python3 benchmarks/onia/build.py
python3 benchmarks/onia/make_points.py target/onia/points.json
python3 benchmarks/onia/affine_reference.py target/onia/points.json target/onia/affine --workers 32
python3 benchmarks/onia/coulomb_check.py target/onia/points.json target/onia/coulomb.json --workers 32
python3 benchmarks/onia/run.py target/onia/points.json target/onia/run --workers 32 --symbolic256 --affine target/onia/affine --coulomb target/onia/coulomb.json
python3 benchmarks/onia/summarize.py target/onia/run
python3 benchmarks/onia/families.py target/onia/points.json target/onia/families.json
python3 benchmarks/onia/affine_reference.py target/onia/families.json target/onia/family-affine --workers 32
python3 benchmarks/onia/coulomb_check.py target/onia/families.json target/onia/family-coulomb.json --workers 32
python3 benchmarks/onia/amplitude.py target/onia/families.json target/onia/amplitude --affine target/onia/family-affine --coulomb target/onia/family-coulomb.json --workers 24
python3 benchmarks/onia/make_fixture.py target/onia/points.json target/onia/affine target/onia/coulomb.json crates/iregi-rust/tests/fixtures/onia-swave.json
```

The physical S-wave amplitudes:

```bash
cargo run -p iregi-prepare --example onia_diagrams -- fixtures/models/sm.json target/onia/diagrams-gg_QQ gg_QQ
for state in singlet octet; do
  cargo run -p iregi-prepare --example onia_swave -- fixtures/models/sm.json target/onia/diagrams-gg_QQ target/onia/points.json target/onia/swave-num-$state.json 100 $state
  python3 benchmarks/onia/swave.py target/onia/swave-num-$state.json target/onia/points.json target/onia/swave-$state.json --workers 16
done
cargo run --release -p iregi-prepare --example onia_diagrams -- fixtures/models/sm.json target/onia/diagrams-gg_QQg gg_QQg
cargo run --release -p iregi-prepare --example onia_swave -- fixtures/models/sm.json target/onia/diagrams-gg_QQg target/onia/points.json target/onia/swave-num-3s1.json 20 3S1-singlet
python3 benchmarks/onia/swave.py target/onia/swave-num-3s1.json target/onia/points.json target/onia/swave-3s1.json --workers 16
python3 benchmarks/onia/make_swave_fixture.py crates/iregi-rust/tests/fixtures/onia-swave-amplitude.json \
  --case target/onia/swave-num-singlet.json target/onia/swave-singlet.json 0 01 \
  --case target/onia/swave-num-singlet.json target/onia/swave-singlet.json 5 01 \
  --case target/onia/swave-num-octet.json target/onia/swave-octet.json 0 01 \
  --case target/onia/swave-num-octet.json target/onia/swave-octet.json 5 01 \
  --case target/onia/swave-num-3s1.json target/onia/swave-3s1.json 2 1000 \
  --case target/onia/swave-num-3S1-octet.json target/onia/swave-3S1-octet.json 2 10000
```

The octets, in four parallel point ranges per state (onia_swave's 7th
argument is the first point):

```bash
for state in 1S0-octet 3S1-octet; do
  for first in 0 5 10 15; do
    target/release/examples/onia_swave fixtures/models/sm.json target/onia/diagrams-gg_QQg target/onia/points.json target/onia/swave-num-$state-r$first.json 5 $state $first &
  done
done; wait
for state in 1S0-octet 3S1-octet; do
  python3 benchmarks/onia/merge_swave.py target/onia/swave-num-$state.json target/onia/swave-num-$state-r{0,5,10,15}.json
  python3 benchmarks/onia/swave.py target/onia/swave-num-$state.json target/onia/points.json target/onia/swave-$state.json --workers 20
done
```

The build has these prerequisites:
- The original-IREGI objects under `target/reference/objects`. These are
  MadGraph's vendored IREGI, built with gfortran -O3 by the benchmark
  campaigns' reference build.
- One repair, in `target/reference/patched/funlib.o`: `factorial_pair` no
  longer overflows 32-bit factorials. It computes the same binomial
  coefficient recursively.
- The quadruple-precision IREGI build
  `target/quad-iregi-source-audit/build-v6-001`.
- The COLLIER 1.2.10 archive `target/reference/collier/build-attempt001`, for
  the scalar arbiter.

`ONIA_TARGET` points the scripts at another `target` directory, for example
the main checkout's from a worktree.

## Coulomb-exchange support in TIRust

For the Coulomb-exchange integrals the propagators are linearly dependent,
2 D_g = D_c + D_c~, and both the Gram and the modified Cayley determinants
vanish. TIRust's direct engine (`crates/iregi-rust/src/direct/`) handles
them as follows:

1. **Detection.** The affine-relation solver (`exec.rs::affine_relation`)
   reports a vanishing identity constant K as a homogeneous relation when
   the propagators are linearly dependent to the working precision's
   rounding of the inputs (Tolerances, below). A small nonzero relative
   velocity stays refused (typed), as on main, because its integral is
   Coulomb singular, not the v = 0 value.
2. **Partial fraction.** The plan variant (`SubsetKind::Homogeneous`,
   `plan.rs`) reduces every subset that contains such a relation by
   1 = −Σ_{i≠j} (c_i/c_j) D_i/D_j, where j is the massless member, the gluon.
   Each moment becomes a sum of the first Taylor coefficients, in m_j², of
   the pinches without i: the pinches with the gluon doubled. The engine's
   existing jets (`master_jets`, `geometry_jets`, `run_program_jets`)
   compute those, and the variant outputs Taylor order 0.
3. **Closed-form bases.** The IBP recursion of the doubled masters ends in
   IR-singular bases whose Cayley matrices are singular, so they come from
   closed forms (`exec.rs::singular_ibp_base`, OneLOop normalization):
   - J(2,1) = −(1/ε_IR + ln(μ²/m²) − 2)/(2m²) and
     J(1,2) = (1/ε_IR + ln(μ²/m²))/(2m²), for the on-shell bubble
     p² = m² with a massless partner;
   - ∫ 1/(l² − m²)³ = −1/(2m²), for a lightlike bubble of equal masses;
   - ∫ 1/(l²)² = 1/ε_UV − 1/ε_IR, for the scaleless doubled massless tadpole.
4. **Lightlike bubbles under jets.** These are allowed when they do not
   contain the doubled propagator; their jets vanish.
5. **Order of the programs.** The partial fraction of a linearly dependent
   subset is an instruction of the moment program (`PostOp::TaylorShift`:
   Taylor order k of the moments from order k + 1 of the pinches), placed
   after its pinches and before every parent that reads it. Such a parent
   can be a relation subset (a planar pentagon whose constant relation
   leaves a Coulomb box) or a subset of six or more members. A partial
   fraction whose pinch needs another linearly dependent set's partial
   fraction would need a Taylor order that no partial fraction gives; the
   plan refuses it (`Unsupported`).
6. **Master families of a variant.** A variant can need scalar masters of
   a family its base plan has not: the rank-1 base pentagon has only C0
   and D0, its Coulomb variant also A0 and B0. The workspace creates that
   family's evaluator on first use.
7. **A singular Cayley matrix** of a pentagon starts the search for an exact
   relation, like a singular Gram matrix. The scalar pentagon has no box
   Gram check that would start it earlier.

Items 5 to 7 fix the review of huashengshao/TIRust#32. Before them, a
Coulomb lane after another lane in a `DirectWorkspace` batch read that lane's
values (E/A 0.09 at rank 2), the rank-1 Coulomb pentagon was wrong even alone
(E/A 0.66 in the finite slot), the scalar one was refused, and planar lanes
read their Coulomb pinches before they were written (O(1), also through the
facade). `tests/onia.rs` covers each case: mixed batches at ranks 0 to 3,
the planar pentagon, and the Coulomb triangle and box at every rank against
their full-rank arrays; reverting any of the three fixes fails it.

**Tolerances.** Three tests decide the Coulomb treatment. Each holds to the
working precision's rounding: u = 8 · 2^-p (p = 53 in binary64, 106 or more
in the facade's rescue tiers) relative to the size of the inputs it reads,
the tolerance that already snaps lightlike invariants to zero.
- **Linear dependence** (`exec.rs::affine_relation`), for the identity
  sum_j c_j D_j = K, f_j = d(origin, j) − m_j² + m_origin²:
  - the shifts' affine relation, component by component:
    |sum_i c_i q_ia| <= u sum_i |c_i| |q_ia|;
  - the constant: |K| <= u sum_j |c_j| s_j, where s_j is the size of
    d(origin, j)'s inputs, sum_a |Δ_a|² + 2 |Δ_a| (|q_ja| + |q_origin,a|),
    plus both masses.

  A lane whose K vanishes only relative to 2^-33 (main's window for
  refusing homogeneous relations) stays refused, as on main.
- **The on-shell bubble with a massless partner** (`singular_ibp_base`):
  |p² − m²| <= u (m² + the size of p²'s inputs).
- **The lightlike bubble**: equal masses, |m_a² − m_b²| <= u (|m_a²| +
  |m_b²|).

The closed-form bases (item 3) also serve main's dotted (raised-power)
plans, which reach them whenever an IBP base is IR singular. Triangles,
boxes and pentagons with a doubled propagator that main refuses ("singular
Cayley matrix", also through the facade) now succeed. #33's
`raised_powers_use_the_singular_base_closed_forms` checks them: bubbles
against the closed forms, and triangles and boxes against the mass
derivative of the undotted integral. Outside these bases TIRust's on-shell
condition stays exact (see the S-wave finding below).

**Limits.**
- At rank 0 a Coulomb triangle or box is only a scalar master, with no
  Gram check to detect the linear dependence. On exact inputs oneloopmaster
  returns a nonfinite master and the lane is refused (typed); the facade
  rescues the triangle and refuses the box.
- Rounded inputs. A lane that is linearly dependent only to binary64's
  rounding, such as a heavy mass one ulp above the on-shell value, or
  M = 9 (1 + 1e-15) in the triangle of the #33 review, is within u at
  p = 53. A binary64 `DirectWorkspace` cannot tell it from exact dependence
  and gives the v = 0 value. Through the facade, the triangle's
  loop-rotation test tells it apart, and the facade refuses it (at 106 bits
  and above it is far outside the rounding); the pentagon passes that test
  and keeps the v = 0 value. Lanes further from dependence (M = 9 (1 +
  delta) with |delta| >= 3e-14, or the third heavy-quark point moved off the
  line by 1e-14) are refused at every precision, directly and through the
  facade, or evaluated correctly by the facade's 256-bit tier. Exact inputs,
  as in this suite, are accepted on every path.

The result is the dimensional-regularization value at v = 0, with native
UV/IR separation. That is the value of IREGI's `ColumbCase`.

`coulomb_closed_form.py` gives the Coulomb triangle at every rank in closed
form, independent of both codes. The integral reduces to the one-momentum
on-shell family ∫ l^α / ((l²)² ((l+p)² − m²)), evaluated by Feynman
parameters. It agrees with IREGI's Coulomb triangles to 3e-15 and with
TIRust's to 5e-15.

## Results (12 November 2026, the engine of #32)

TIRust was built from `hs/onia` (25e3f31, #32's head), with main's
huashengshao/TIRust#31 merged (triangles get the loop-rotation test), using
this suite's drivers. Original IREGI is the binary64 build of the benchmark
campaigns. 21,585 of the 21,600 lanes have an independent reference. "> 1"
counts lanes with E/A above 1 (an array wrong at O(1) or worse), and "none"
counts lanes where IREGI returns no array. #33's final engine, with its
review fixes, gives bit-identical arrays on all 21,600 integral lanes and
the 2,800 family lanes.

| process | N | lanes | with reference | TIRust <= 1e-10 | TIRust max | IREGI IBP <= 1e-10 | IREGI IBP > 1 | IREGI none | IREGI IBP max |
|---|---|---|---|---|---|---|---|---|---|
| gg_QQ | 4 | 400 | 400 | 400 | 2.0e-13 | 400 | 0 | 0 | 3.6e-13 |
| gg_QQ | 3 | 800 | 800 | 800 | 3.4e-15 | 766 | 0 | 0 | 1.8e-08 |
| gg_QQ | 2 | 800 | 800 | 800 | 1.0e-15 | 800 | 0 | 0 | 7.0e-16 |
| gg_QQg | 5 | 2,400 | 2,385 | 2,384 | 9.4e-10 | 282 | 1,632 | 444 | 8.4e+72 |
| gg_QQg | 4 | 7,200 | 7,200 | 7,200 | 5.8e-11 | 5,477 | 836 | 0 | 3.9e+54 |
| gg_QQg | 3 | 6,400 | 6,400 | 6,400 | 5.3e-11 | 6,166 | 0 | 0 | 2.4e-05 |
| gg_QQg | 2 | 3,600 | 3,600 | 3,600 | 1.4e-15 | 3,600 | 0 | 0 | 1.0e-15 |

The references:

| source | gg_QQ | gg_QQg |
|---|---|---|
| exact identity (`affine_reference.py`) | 400 | 3,964 |
| direct256 confirmed by quad IREGI | 1,400 | 14,385 |
| direct256 arbitrated by COLLIER | 0 | 108 |
| Coulomb triangle closed form | 100 | 200 |
| quad IREGI = exact doubled-gluon combination (Coulomb) | 100 | 928 |

The full per-topology table is `summary.md` of a run (`summarize.py`).

**TIRust**
- The binary64 default stays within 1e-10 on 21,584 of the 21,585 lanes with
  a reference. The remaining lane is a pentagon (`N5-LHHHL-Q|g2|g3|Q|g1`,
  point 87) at 9.4e-10, with an estimated error of 2.9e-12.
- **Coulomb exchange.** All 1,600 lanes, in 2 topologies of gg_QQ and 7 of
  gg_QQg, now succeed in binary64, in the first pass (rotation test
  included), and at 256 bits.
  - On the 1,585 lanes with a reference, the largest E/A is 4.6e-15 for
    triangles, 1.6e-13 for boxes and 4.7e-11 for pentagons.
  - 15 Coulomb pentagon lanes have no independent reference: in their exact
    identity, one doubled box is not confirmed by binary64 IREGI to 1e-9.
  - Without Coulomb support (commit 75b24ef), these lanes were refused,
    typed. Before #31, every other lane gave exactly the same result with
    and without Coulomb support.
- **Small-Gram triangles.** Before #31 (commits 75b24ef to c65727b),
  17 gg_QQg triangle lanes reached between 1e-10 and 2.0e-8, in
  `N3-HLL-Q+g2|g3|Q+g1` and `N3-LHH-Q+g2|g3|Q+g1`.
  - These triangles have one lightlike leg and two nearly equal
    virtualities. Lanes with G/s^2 of about 7e-4 to 8e-4 were accepted by
    the binary64 gate and lost about 8 digits at rank 3, with no rotation
    test for N = 3.
  - With #31 they are rescued, and all triangles are within 5.3e-11.
- **Coulomb variant reused across lanes.** #31's rotation test of the
  triangles exposed a bug in the Coulomb variant: a variant reused from
  another lane (here, the rotated copy) raised the propagator at the old
  index, a heavy quark. Every Coulomb triangle then went to the 256-bit
  rescue: correct, but at 1.5 ms per lane. The lane now checks that the
  raised member is massless, and `tests/onia.rs` asserts that the binary64
  pass accepts the Coulomb lanes. Without the fix that test fails.

**Original IREGI** (`mre/IREGI_ONIA_MRE.md` has the reproducers)
- **gg_QQ.** IBP mode is right everywhere, within 1.8e-8.
- **PV mode** is wrong by 1e-3 to 1.5, flagged stable, on both zero-Gram
  boxes with massless propagators next to the heavy legs.
- **gg_QQg.** 2,468 lanes with a reference are off by more than 1, and IREGI
  flags all of them stable:
  - 1,632 pentagon lanes, with E/A up to 8.4e72, including the 3 Coulomb
    pentagons; it returns no array on 444 more;
  - 836 box lanes.
  Before Coulomb support (commit 75b24ef) the count was 2,225 of the 20,000
  non-Coulomb lanes: 1,504 pentagon lanes and 721 box lanes in 10
  topologies.
- **Bundled OneLOop 3.4.** Its three-mass D0 has a wrong real part on one box
  topology. There IREGI is wrong on 76 of 200 lanes in binary64 and its quad
  build on 108. COLLIER and TIRust agree with each other.
- **Coulomb integrals.** IREGI's hand-coded Coulomb C0 equals the analytic
  v = 0 value to 7e-16. Its IBP Coulomb triangles and boxes equal the exact
  partial-fraction combination of its own dotted integrals:
  - to 9e-14 for the gg_QQ box and all Coulomb triangles;
  - to 4e-13 in quad.
  - On the gg_QQg Coulomb boxes, binary64 IBP matches on only about half the
    lanes, being up to 4e4 off.
  - The Coulomb pentagons fail as the other pentagons do.

**Reference tools.**
- **Quad IREGI** is unreliable on these kinematics: it is off by more than 1 on
  30 pentagon lanes, and on 43 box lanes of the three-mass box above. That is
  why the reference uses the exact identity and COLLIER arbitration instead.
- **Quad IREGI's general API** overflows its 32-bit factorials on doubled
  propagators ("i > 12"): the `factorial_pair` repair is in the binary64
  objects only.
- **TIRust's symbolic engine at 256 bits** (symbolic256) refuses the
  pentagons and zero-Gram boxes.

**Timing (indicative only).** These times come from a correctness run. The
host was not pinned and carried a load of about 330. IREGI's times include
its first call, with no warm-up. The table gives the median time per lane:

| | TIRust default | IREGI IBP |
|---|---|---|
| gg_QQ boxes | 42 us | 1.7 ms |
| gg_QQ triangles | 6 us | 0.44 ms |
| gg_QQg pentagons | 0.82 ms | 15 ms |
| gg_QQg boxes | 62 us | 1.8 ms |
| gg_QQg triangles | 7 us | 0.56 ms |

## Integral families as amplitudes

A family is a parent topology of the largest N (the 4 boxes of gg_QQ and the
12 pentagons of gg_QQg, each with its mass pattern) together with all its
pinches down to bubbles: 11 and 26 integrals. Every member is taken in the
parent's loop-momentum routing (`families.py`), as an amplitude's integrals
are; this is the setting of the report's Table 9. The members are
translations of the integrals above, 4,400 and 62,400 member lanes.

**Method** (`amplitude.py`):
- **TIRust** evaluates all members of a family in one `AmplitudeWorkspace`
  call (`examples/onia_amplitude.rs`), timed with per-subset sharing on and
  off. The arrays must be bit-identical with sharing on and off.
- **References.** Every member gets the references above (run.py's rules,
  with `affine_reference.py` and `coulomb_check.py` run on the families),
  and original IREGI's binary64 arrays.
- **Amplitude.** The amplitude of a family at a point is
  A = Σ_members Σ_α c_α M_α. The coefficients are fixed random
  (u + i v)/μ^|α| per family, member and point. A is combined as [finite,
  single pole (UV + IR), double pole], and its error is
  max |A − A_ref| / max |A_ref|.

**Results** (with #31 merged):
- **Bit-identical sharing.** The sharing on/off check holds for all 2,800
  family points.
- **TIRust** returns every member on every point, the Coulomb ones included.
  - Its family amplitudes are within 1e-10 of the reference on all 400
    gg_QQ points and all 2,385 gg_QQg points with a reference. The largest
    error is 4.4e-12.
  - Before #31, 8 gg_QQg points reached up to 3.8e-10; each contained a
    small-Gram triangle member.
  - 15 gg_QQg family points have no reference: the 15 Coulomb pentagon lanes
    above.
- **Original IREGI** (IBP mode):
  - **gg_QQg:** its family amplitude is off by more than 1 on 1,648 of the
    1,941 family points where every member returns an array (up to 1.5e68),
    and some member returns no array on 444 points. For the Coulomb pentagon
    family `N5-HLHHH-Q|Q|g2|g3|g1` that happens on every point.
  - **gg_QQ:** it is within 6.7e-9.

**Timing (indicative only).** These times come from the correctness run,
with 16 to 24 workers on an unpinned host at load 300 or more. TIRust is
timed per call over 25 points: one warm-up, then three calls with sharing on
and three with it off. IREGI is the sum of its members' first calls. The
table gives medians per point; ratio is TIRust shared over IREGI. The
slowest TIRust families are those whose members need the precision rescue.

| process | family (parent) | points | with reference | TIRust <= 1e-10 | TIRust max | IREGI IBP <= 1e-10 | IREGI IBP > 1 | IREGI none | TIRust shared | TIRust separate | IREGI IBP | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gg_QQ | `N4-HHLH-g2\|Q\|Q\|g1` | 100 | 100 | 100 | 3.4e-15 | 70 | 0 | 0 | 91 us | 100 us | 8.4 ms | 0.011 |
| gg_QQ | `N4-HLLH-Q\|g2\|Q\|g1` | 100 | 100 | 100 | 3.6e-14 | 100 | 0 | 0 | 162 us | 157 us | 8.4 ms | 0.019 |
| gg_QQ | `N4-LHHL-Q\|g2\|Q\|g1` | 100 | 100 | 100 | 8.3e-15 | 100 | 0 | 0 | 152 us | 164 us | 8.1 ms | 0.019 |
| gg_QQ | `N4-LLHL-g2\|Q\|Q\|g1` | 100 | 100 | 100 | 4.6e-15 | 100 | 0 | 0 | 102 us | 107 us | 7.5 ms | 0.014 |
| gg_QQg | `N5-HHHLH-g2\|g3\|Q\|Q\|g1` | 200 | 189 | 189 | 2.0e-14 | 0 | 21 | 165 | 1.2 ms | 1.3 ms | 26.5 ms | 0.043 |
| gg_QQg | `N5-HHLHH-g2\|Q\|Q\|g3\|g1` | 200 | 196 | 196 | 1.4e-13 | 1 | 104 | 79 | 768 us | 889 us | 37.7 ms | 0.020 |
| gg_QQg | `N5-HHLLH-g2\|Q\|g3\|Q\|g1` | 200 | 200 | 200 | 1.1e-12 | 14 | 178 | 0 | 1.7 ms | 2.0 ms | 43.6 ms | 0.038 |
| gg_QQg | `N5-HLHHH-Q\|Q\|g2\|g3\|g1` | 200 | 200 | 200 | 2.4e-12 | 0 | 0 | 200 | 828 us | 869 us | - | - |
| gg_QQg | `N5-HLLHH-Q\|g2\|Q\|g3\|g1` | 200 | 200 | 200 | 1.8e-12 | 36 | 164 | 0 | 3.6 ms | 3.3 ms | 46.0 ms | 0.078 |
| gg_QQg | `N5-HLLLH-Q\|g2\|g3\|Q\|g1` | 200 | 200 | 200 | 1.4e-12 | 37 | 163 | 0 | 12.2 ms | 10.8 ms | 43.9 ms | 0.277 |
| gg_QQg | `N5-LHHHL-Q\|g2\|g3\|Q\|g1` | 200 | 200 | 200 | 2.0e-12 | 28 | 172 | 0 | 12.8 ms | 12.7 ms | 44.0 ms | 0.290 |
| gg_QQg | `N5-LHHLL-Q\|g2\|Q\|g3\|g1` | 200 | 200 | 200 | 2.7e-12 | 32 | 168 | 0 | 2.9 ms | 3.1 ms | 40.4 ms | 0.073 |
| gg_QQg | `N5-LHLLL-Q\|Q\|g2\|g3\|g1` | 200 | 200 | 200 | 4.4e-12 | 57 | 143 | 0 | 666 us | 679 us | 36.3 ms | 0.018 |
| gg_QQg | `N5-LLHHL-g2\|Q\|g3\|Q\|g1` | 200 | 200 | 200 | 1.3e-12 | 17 | 183 | 0 | 1.5 ms | 1.5 ms | 39.0 ms | 0.037 |
| gg_QQg | `N5-LLHLL-g2\|Q\|Q\|g3\|g1` | 200 | 200 | 200 | 3.2e-13 | 27 | 173 | 0 | 787 us | 833 us | 39.1 ms | 0.020 |
| gg_QQg | `N5-LLLHL-g2\|g3\|Q\|Q\|g1` | 200 | 200 | 200 | 8.4e-14 | 21 | 179 | 0 | 1.2 ms | 1.1 ms | 41.5 ms | 0.028 |

## Physical S-wave amplitudes: g g > c c~ [1S0]

This section evaluates the one-loop QCD amplitude of g g > c c~ with the pair
projected onto the 1S0 state at v = 0 (the η_c channel), in the colour singlet
1S0[1] and the octet 1S0[8]. The numerators come from Feynman rules. The
amplitude is bare and R1 only: no R2 rational terms and no counterterms, so
its finite part is not a physical prediction. Its poles are.

**Diagrams** (`crates/iregi-prepare/examples/onia_diagrams.rs`). FeynKit's
generator runs on `fixtures/models/sm.json` with these settings:
- QCD order 2 + 2 × loops and QED order 0;
- Feynman gauge with ghosts;
- b and t vetoed; u, d and s massless;
- self-energies on external legs, and tadpoles, vetoed.

That gives 3 tree and 39 one-loop diagrams. 32 of the one-loop diagrams have
a loop-independent internal propagator: an s-channel gluon, or the tree leg of
a vertex correction or of an internal self-energy. iregi-prepare's new
`prepare_dot_with_tree_propagators` lists these in
`PreparedGraph::tree_propagators`, and the caller multiplies by 1/(p² − m²).
`prepare_dot` itself still refuses such graphs.

**Projection** (`crates/iregi-prepare/examples/onia_swave.rs`). The DOT
`projector` replaces the open c line ū(p_c) … v(p_c~) by
Π₀ = (p̸_c~ − m) γ5 (p̸_c + m) / √(8m³), the Weyl γ5 = iγ⁰γ¹γ²γ³ of spenso.
The colour projector is:
- singlet: δ_ij/√3 · δ^ab/√8;
- octet: Σ_c √2 T^c_ji d^abc / √(40/3) = √(3/20) ({T^a,T^b}_ji − δ_ij δ^ab/3),
  by Fierz.

Each gluon gets two real linear polarizations, plus k/k⁰ for the Ward
identity. The coefficients are evaluated at 128 bits on the 100 lattice
points, with G = 1. 16 diagrams survive the singlet projection and 14 the
octet one. The other two are the non-abelian boxes with a triple-gluon
vertex, whose colour factor is 0 against d^abc; explicit Gell-Mann matrices
give the same 0 (singlet: −6i).

**Amplitude** (`swave.py`):
- **Integrals.** For each surviving diagram, the integral comes from FeynKit's
  routing, with the shifts exact on the lattice. Its masses are 0 or the
  lattice m².
- **Tree factors** are taken at the same exact momenta.
- **Tensor arrays** come from three sources:
  - TIRust (binary64 default);
  - original IREGI (binary64 IBP);
  - the reference: the exact denominator identity on degenerate integrals,
    else TIRust at 256 bits confirmed by quad IREGI or COLLIER, as above.
- **Contraction.** The arrays are contracted with the coefficients:
  X = Σ_diagrams tree · Σ_α c_α M_α. X uses the native slots, and the
  amplitude is X/(2π)⁴ at the poles.

**Physics checks.** These are independent of the integral references. Each of
the three amplitudes is checked against the Born A0, which swave.py writes
out from the same Feynman rules: vertex iγ^μT^a, propagator i(p̸+m)/(p²−m²),
and the same Π₀.
- **Double pole:** X_IR2 = −6π² A0. This is Catani's −C_A/ε² in units of
  α_s/2π.
- **Single pole:** IR1/IR2 + γ_E + ln π − ln(μ²/s) must equal 1 − C_F/C_A + iπ
  for the singlet and 3/2 − C_F/C_A + iπ/2 for the octet.
  - Singlet: Catani's two-gluon singlet.
  - Octet: Becher–Neubert with the pair as one heavy octet of mass 2m.
  - Both are corrected for the IR parts of the external on-shell field
    factors that the bare amplitude leaves out: −(5/6 C_A − 2/3 T_F n_l)/ε for
    the gluons and −C_F/ε for c c~. n_l cancels.
- **UV pole:** X_UV = π²[(2C_A + C_F) A0 + 3C_F m ∂A0/∂m]. This is minus the
  MS-bar coupling, on-shell field and on-shell mass counterterms; ∂/∂m acts
  on internal propagator masses only.
- **Ward identity:** ε₁ → k₁/k₁⁰ (or ε₂). The amplitude vanishes exactly, but
  trivially: every Levi-Civita structure needs two independent momenta
  besides the polarizations, and g g > c c~ at v = 0 has only one. The 00 and
  11 linear polarizations vanish for the same reason. Milestone 2, g g > c c~ g,
  will make this check non-trivial.

**Results** (100 lattice points per state). On a physical polarization pair
the Born is nonzero: 296 pairs in all, since 00 and 11 also contribute where
the gluons' linear polarizations are not aligned. Errors are
max |X − X_ref| / max |X_ref| over the four slots. The pole columns give each
check's worst deviation for the reference, TIRust and IREGI amplitudes.

| state | points (with reference) | polarization pairs | TIRust vs reference: max (pairs <= 1e-12) | IREGI vs reference: max (pairs <= 1e-12) | double pole: reference / TIRust / IREGI | single pole | UV pole |
|---|---|---|---|---|---|---|---|
| 1S0[1] | 100 (100) | 296 | 6.8e-12 (290) | 2.9e-12 (291) | 9.6e-13 / 4.0e-12 / 2.6e-12 | 3.7e-12 / 2.4e-11 / 1.6e-11 | 9.3e-15 / 9.3e-15 / 9.4e-15 |
| 1S0[8] | 100 (100) | 296 | 6.3e-12 (290) | 2.3e-12 (292) | 1.5e-12 / 1.9e-12 / 6.5e-13 | 2.4e-12 / 8.7e-12 / 2.4e-12 | 9.3e-15 / 9.3e-15 / 9.3e-15 |

- Every check holds on every point, in both states and for all three
  amplitudes.
- The worst errors are on the most boosted points, with gluon energy ratios
  of about 300 (points 15 and 24).
- The Ward entries vanish to 9e-14 of the amplitude scale.
- This 2 → 1 amplitude has at most boxes, so it does not reach the
  degenerate pentagons, nor the zero-Gram boxes on which IREGI fails in
  g g > c c~ g (above).
- R2 is left out, so the finite parts are not compared with the literature.

**Finding: the on-shell condition is exact in TIRust.** onia_swave binds
MC = fl(√m²), so the prepared MC² can miss the lattice m² by an ulp. It does
at point 5, by −2e-16 relative.
- The rank-2 soft triangles of diagrams 017, 018, 020 and 021 are then off
  their on-shell, IR-divergent configuration. They have one massive and two
  massless propagators, on an on-shell massive leg.
- TIRust correctly returns the regulated integral, with status success: IR2 =
  0, and ln δ in the IR1 and finite parts, up to 5e4.
- Original IREGI snaps to on-shell.
- The singlet hides this, because those four diagrams cancel by colour. The
  octet's poles were off by 50% (IR2) and by a factor of 73 (IR1).

swave.py therefore snaps the masses to the exact lattice value. An amplitude
code that squares a rounded mass will hit the same cliff. TIRust has no
on-shell tolerance outside the closed-form bases of dotted plans (the
working precision's rounding, "Coulomb-exchange support in TIRust" above),
which these soft triangles do not reach.

## Physical S-wave amplitudes: g g > c c~ [3S1, singlet] g

This section evaluates the J/ψ + g channel the same way: one-loop QCD, bare
and R1 only.

**Diagrams and projection.**
- FeynKit gives 16 tree and 470 one-loop diagrams (`onia_diagrams.rs ...
  gg_QQg`).
- The projector is Π₁ = (p̸_c~ − m) ε̸*_ψ (p̸_c + m) / √(8m³). The J/ψ
  polarization is bound as ϵbar on the c leg's edge, with three real vectors
  orthonormal and orthogonal to P.
- The colour projector is δ_ij/√3 · d^abc/√(40/3) for the three gluons; the
  f^abc part vanishes by C parity. d^abc comes from FeynKit's generators as
  2[Tr(T^aT^bT^c) + Tr(T^bT^aT^c)].
- The outgoing gluon's polarization is ϵbar on its edge.
- 147 one-loop diagrams survive the projection, 18 of them rank-4 pentagons.

**Preparation at numeric kinematics.** The fully symbolic numerator of a
pentagon expands to millions of terms in the momentum and polarization
components. onia_swave therefore takes iregi-prepare's unexpanded scalar
(`prepare_scalar_dot_with_tree_propagators`). For every point and
polarization it substitutes the numbers (128-bit floats) and only then
extracts the loop polynomial.
- Symbolically, one pentagon had not finished after 40 minutes (13 minutes
  with the pairwise conversion below). Numerically, all 470 diagrams on 3
  points, with 33 polarization sets each, take 5 minutes.
- On g g > c c~ this path reproduces the symbolic one to 3e-37.

**Checks.** The checks are those of the 1S0 section, with three gluons:
- X_IR2 = −9π² A0;
- the single-pole ratio equals (1/3) Σ_{i<j} ln(μ²/(−s_ij)) + 1 − 2C_F/(3C_A),
  from Catani with T_i·T_j = −C_A/2 for each gluon pair, minus the external
  field factors;
- X_UV = π²[(3C_A + C_F) A0 + 3C_F m ∂A0/∂m];
- the Born A0 sums the 3! orderings of the gluons on the c line; diagrams
  with a triple-gluon vertex vanish against d^abc.

The Ward identity is now non-trivial (ε_i → k_i/k_i⁰, the other gluons
physical):
- the IR poles vanish;
- the UV pole equals π² 3C_F m ∂A0/∂m on these polarizations, since the
  internal mass counterterm alone is not gauge invariant;
- the finite part does not vanish (R2 is left out).

**Results** (the first 20 gg_QQg lattice points):
- Each point has 84 distinct integrals, 24 physical polarization sets
  (2³ gluon × 3 J/ψ) and 9 Ward sets.
- Pole deviations are scored against the point's scale: each slot's largest
  deviation over the physical sets, divided by the largest prediction for that
  slot. A per-set ratio inflates where a set's Born is small.

| amplitude | points | vs reference | UV pole | IR1 pole | IR2 pole |
|---|---|---|---|---|---|
| reference | 6 | – | 2.0e-14 | 3.4e-14 | 2.6e-14 |
| TIRust | 20 | 2.2e-12 (6 points) | 5.5e-14 | 4.1e-11 | 9.6e-14 |
| original IREGI | 1 (point 10) | 8.5e-6 | 5.3e-14 | 1.6e-14 | 4.3e-15 |

- **References.** Only 6 points (2, 3, 10, 12, 14, 18) have a reference
  integral array. On the others, the exact identity produces dotted
  sub-integrals that nothing confirms: boxes with a doubled propagator, from
  the Coulomb pentagons. Quad IREGI returns no array for them, and COLLIER
  takes no dotted integrals. TIRust's amplitude still meets every pole
  prediction on all 20 points.
- **Original IREGI.**
  - At 19 of the 20 points, it flags the 6 Coulomb pentagons as unstable and
    returns no array. These are rank-4 pentagons with masses HHHHL: a gluon
    across the pair and the three external gluons on the c line. IREGI
    therefore has no 3S1 amplitude at those points.
  - On the integrals it does return, its partial amplitude is off by 8e43
    to 3e54. These are the degenerate pentagons that it flags as stable.
    TIRust agrees with the reference on the same subsets to 3.4e-11.
  - At point 10 it returns every array. Its amplitude there is within 8.5e-6
    of the reference: the poles are right, and the error is in the finite
    part.
- **Ward identity** (9 sets per point, deviations relative to the point's
  amplitude scale):
  - the IR poles vanish, to 3.5e-12 for TIRust and 5e-14 for the reference;
  - the UV pole equals the internal mass-counterterm structure
    π² 3C_F m ∂A0/∂m to 1.6e-14 (all three amplitudes);
  - the finite part stays at up to 7% of the scale, as expected without R2.

## Physical S-wave amplitudes: g g > c c~ [1S0, octet] g and [3S1, octet] g

These are the colour-octet channels of J/ψ (and η_c) + g production.

**Colour wavefunctions.**
- The octet pair and three gluons span several invariant colour tensors, and
  the amplitude is a vector in that space. So the colour is not projected.
- The pair gets √2 T^c_ji χ^c. Each gluon gets a colour wavefunction u^a.
- χ and the u are numeric unit 8-vectors, two random sets per point, bound
  like the polarizations: ϵbar(0|1, coad) for the incoming gluons, ϵ(4, coad)
  for the outgoing one, ϵ(3, coad) for χ.
- The spin projectors are Π₀ (1S0) and Π₁ (3S1), as above.

**Born from FeynKit.** The octet Born includes the triple- and four-gluon
vertex diagrams, which vanish for the singlets.
- In 3S1[8], g* → c c~[3S1, octet] is allowed, since Tr[γ^μ Π₁] ≠ 0.
- The Born is therefore taken from FeynKit's 16 tree diagrams, prepared by
  iregi-prepare's scalar path, which now accepts tree graphs.
- The projector's mass is a separate symbol, so m ∂A0/∂m over the internal
  masses is d/dMC of the tree scalars, plus the massive tree denominators.
- On g g > c c~ (singlet and octet), FeynKit's Born and its mass derivative
  equal swave.py's hand-written Born to 2.4e-16.

**Colour-space poles.** The IR poles follow Becher–Neubert with the pair as
one heavy octet P of mass 2m (as for 1S0[8] above), minus the external field
factors' IR parts. In units of α_s/4π, the single pole of the bare amplitude
is

S = 2 Σ_{i<j} T_i·T_j ln(μ²/(−s_ij)) + 2 Σ_i T_P·T_i ln(2mμ/(−s_Pi)) − 4C_A + 2C_F,

where i, j run over the gluons, s_ij = 2σ_ij p_i·p_j, and both logs carry
−i0. The double pole is diagonal: X_IR2 = −9π² A0.
- onia_swave evaluates the tree diagrams with T_i·T_j inserted on each pair
  of colour legs. A leg's wavefunction w becomes t^x w, with
  (t^x w)^b = Σ_a w^a (−i f^xab), summed over x.
- swave.py checks X_IR1 + (γ_E + ln π) X_IR2 = π² S A0.
- It also checks colour conservation: Σ_{j≠i} T_i·T_j A0 = −C_A A0 for every
  leg.
- The UV and Ward checks are those of the singlet.

**Results** (the first 20 gg_QQg lattice points, as for the singlet):
- Polarization sets per point: 1S0[8] has 16 physical sets (2³ gluon × 2
  colour) and 3 Ward sets; 3S1[8] has 48 physical sets (× 3 J/ψ) and 9 Ward
  sets.
- 1S0[8] has 161 to 164 distinct integrals per point, and 3S1[8] has 183 to
  201.
- Pole deviations are on the point scale, as for the singlet.

| state | amplitude | points | vs reference | UV pole | IR1 pole | IR2 pole |
|---|---|---|---|---|---|---|
| 1S0[8] | reference | 6 | – | 8.6e-15 | 1.3e-14 | 1.1e-14 |
| 1S0[8] | TIRust | 20 | 1.0e-13 (6 points) | 1.2e-14 | 2.4e-13 | 9.9e-14 |
| 1S0[8] | original IREGI | 1 (point 10) | 1.4e-7 | 7.7e-15 | 7.2e-15 | 4.0e-15 |
| 3S1[8] | reference | 6 | – | 8.3e-14 | 1.2e-14 | 1.1e-14 |
| 3S1[8] | TIRust | 20 | 1.6e-13 (6 points) | 1.0e-13 | 1.5e-13 | 2.9e-13 |
| 3S1[8] | original IREGI | 1 (point 10) | 2.9e-7 | 9.3e-15 | 9.7e-15 | 6.4e-15 |

- **Colour conservation** holds to 4.7e-14 (1S0[8]) and 1.1e-10 (3S1[8]),
  relative to each set's Born.
- **Poles.** The colour-space Becher–Neubert prediction holds on every
  point, for the reference and TIRust. This covers the T_P·T_i soft terms of
  the octet pair and its −C_A anomalous dimension.
- **References** exist at the same 6 points as for the singlet, for the same
  reason (dotted sub-integrals).
- **Original IREGI** is complete only at point 10. Elsewhere the Coulomb
  pentagons have no array, and its partial amplitudes are off by 5e38 to
  2.6e54 (TIRust on the same subsets: 1.3e-9).
- **Ward identity:**
  - the IR poles vanish (TIRust to 6e-14);
  - the UV pole equals the mass-counterterm structure to 1e-14;
  - the finite part reaches 0.7% (1S0[8]) and 11% (3S1[8]) of the scale,
    without R2.
- **Cost.** One process takes about 25 minutes for 5 points of 1S0[8] and
  about 70 minutes for 5 points of 3S1[8]. The tree diagrams' 48 colour
  insertions per set dominate. Point ranges run in parallel (onia_swave's
  FIRST_POINT, then `merge_swave.py`).

## Regression test

`crates/iregi-rust/tests/onia.rs` (huashengshao/TIRust#33) runs from
`tests/fixtures/onia-swave.json` (`make_fixture.py`):
- 12 degenerate lanes (boxes, a triangle and pentagons, both processes) must
  agree with the exact-identity reference to E/A <= 1e-9;
- 5 Coulomb lanes (a triangle, two boxes and two pentagons) must agree to
  1e-9 with original IREGI's consistency-checked arrays;
- the gg_QQ Coulomb box family (the box and its 10 pinches) evaluated in one
  `AmplitudeWorkspace` must succeed on every member, be bit-identical with
  sharing on and off, and give the box within 1e-9 of IREGI.

The Coulomb triangle's scalar moment must also equal the v = 0 value
C0 = −(m²)^(−1−ε) Γ(1+ε) / (2ε(1 + 2ε)), to 1e-12.

The S-wave amplitude test runs from `tests/fixtures/onia-swave-amplitude.json`
(`make_swave_fixture.py`, 518 KB). Its cases are:
- the 1S0 singlet and octet at points 0 (at rest) and 5 (boosted), at the
  polarizations (e1, e2);
- g g > c c~ [3S1, singlet] g at point 2, labels 1000: 84 integrals,
  pentagons included, among them the Coulomb pentagons on which IREGI returns
  no array;
- g g > c c~ [3S1, octet] g at point 2, labels 10000: 183 integrals. It is
  checked against the colour-space single-pole value stored in the fixture.

Each case holds its distinct loop integrals with contracted coefficients.
TIRust's amplitude must agree with the reference to 1e-9. Its poles must
match the Born predictions: double and UV poles to 1e-10, single to 1e-9.

The test file's other cases need no fixture:
- the #32 review's cases: mixed batches, the planar pentagon, and the
  Coulomb triangle and box at every rank;
- the #33 review's near-dependent lanes, M = 9 (1 + delta) and a point moved
  off the line, refused or equal to a 256-bit reference, directly and
  through the facade;
- dotted plans that reach the closed-form bases, against the closed forms or
  the mass derivative of the undotted integral.

## Files

| file | role |
|---|---|
| `make_points.py` | topologies and exact lattice points |
| `run.py` | runs every code; references; E/A per lane (`deltas-*.jsonl`) |
| `affine_reference.py` | exact denominator-identity references |
| `coulomb_check.py` | consistency of original IREGI on the Coulomb integrals |
| `coulomb_closed_form.py` | closed form of the Coulomb triangle at every rank |
| `summarize.py` | per-topology summary (`summary.json`, `summary.md`) |
| `make_fixture.py` | the regression test's fixture |
| `families.py` | integral families (parent and pinches, in the parent's routing) |
| `amplitude.py` | families as amplitudes: TIRust `AmplitudeWorkspace` against IREGI, contracted amplitudes, timing |
| `swave.py` | physical S-wave amplitudes (1S0 of g g > c c~; 3S1, 1S0[8] and 3S1[8] of g g > c c~ g): contraction, Born, pole and Ward checks |
| `make_swave_fixture.py` | the S-wave amplitude test's fixture |
| `merge_swave.py` | concatenates onia_swave outputs over point ranges |
| `crates/iregi-prepare/examples/onia_diagrams.rs` | FeynKit diagram generation (DOT files) |
| `crates/iregi-prepare/examples/onia_swave.rs` | S-wave projection and numerator coefficients per point and polarization |
| `build.py` | binaries and their provenance (`target/onia/build.json`) |
| `drivers/iregi_onia.f90` | original-IREGI driver |
| `drivers/collier_scalar.f90` | COLLIER scalar C0/D0, the arbiter |
| `mre/` | minimal reproducers of the original-IREGI defects |
| `crates/iregi-rust/examples/onia_driver.rs` | TIRust driver: per-lane masses and optional powers |
| `crates/iregi-rust/examples/onia_amplitude.rs` | TIRust family driver: `AmplitudeWorkspace`, per-member arrays, sharing on/off timing |
