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
   reports a vanishing identity constant K as a homogeneous relation, judged
   to the inputs' rounding with the same test that decides exact affine
   relations. A small nonzero relative velocity stays refused, because its
   integral is Coulomb singular.
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

The result is the dimensional-regularization value at v = 0, with native
UV/IR separation. That is the value of IREGI's `ColumbCase`.

`coulomb_closed_form.py` gives the Coulomb triangle at every rank in closed
form, independent of both codes. The integral reduces to the one-momentum
on-shell family ∫ l^α / ((l²)² ((l+p)² − m²)), evaluated by Feynman
parameters. It agrees with IREGI's Coulomb triangles to 3e-15 and with
TIRust's to 5e-15.

## Results (12 November 2026, branch onia)

TIRust was built from this branch with this suite's driver. Original IREGI is
the binary64 build of the benchmark campaigns. 21,585 of the 21,600 lanes
have an independent reference. "> 1" counts lanes with E/A above 1 (an array
wrong at O(1) or worse), and "none" counts lanes where IREGI returns no array.

| process | N | lanes | with reference | TIRust <= 1e-10 | TIRust max | IREGI IBP <= 1e-10 | IREGI IBP > 1 | IREGI none | IREGI IBP max |
|---|---|---|---|---|---|---|---|---|---|
| gg_QQ | 4 | 400 | 400 | 400 | 2.0e-13 | 400 | 0 | 0 | 3.6e-13 |
| gg_QQ | 3 | 800 | 800 | 800 | 3.4e-15 | 766 | 0 | 0 | 1.8e-8 |
| gg_QQ | 2 | 800 | 800 | 800 | 1.0e-15 | 800 | 0 | 0 | 7.0e-16 |
| gg_QQg | 5 | 2,400 | 2,385 | 2,384 | 9.4e-10 | 282 | 1,632 | 444 | 8.4e72 |
| gg_QQg | 4 | 7,200 | 7,200 | 7,200 | 5.8e-11 | 5,477 | 836 | 0 | 3.9e54 |
| gg_QQg | 3 | 6,400 | 6,400 | 6,383 | 2.0e-8 | 6,166 | 0 | 0 | 2.4e-5 |
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
- The binary64 default stays within 1e-10 on 21,567 of the 21,585 lanes with
  a reference, and within 2e-8 on all of them.
- **Coulomb exchange.** All 1,600 lanes, in 2 topologies of gg_QQ and 7 of
  gg_QQg, now succeed in binary64 and at 256 bits.
  - On the 1,585 lanes with a reference, the largest E/A is 4.6e-15 for
    triangles, 1.6e-13 for boxes and 4.7e-11 for pentagons.
  - 15 Coulomb pentagon lanes have no independent reference: in their exact
    identity, one doubled box is not confirmed by binary64 IREGI to 1e-9.
  - The other 20,000 lanes keep exactly the E/A they had before Coulomb
    support (commit 75b24ef).
- **Triangles near a small Gram determinant.** These are the remaining
  beyond-request lanes: 17 lanes between 1e-10 and 2.0e-8, in two triangle
  topologies of gg_QQg, `N3-HLL-Q+g2|g3|Q+g1` and `N3-LHH-Q+g2|g3|Q+g1`.
  - The triangles have one lightlike leg and two nearly equal virtualities.
  - Lanes with G/s^2 below about 1.4e-4 are refused by the binary64 gate and
    rescued, reaching E/A of 1e-22.
  - The lanes just above the gate, at G/s^2 of about 7e-4 to 8e-4, are
    accepted in binary64 and lose about 8 digits at rank 3.
  - Triangles get no loop-rotation test (it starts at four propagators), so
    these lanes carry no flag.
- One pentagon lane (`N5-LHHHL-Q|g2|g3|Q|g1`, point 87) reaches 9.4e-10, with
  an estimated error of 2.9e-12.

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

**Results.**
- **Bit-identical sharing.** The sharing on/off check holds for all 2,800
  family points.
- **TIRust** returns every member on every point, the Coulomb ones included.
  - Its family amplitudes are within 1e-10 of the reference on all 400
    gg_QQ points and on 2,377 of the 2,385 gg_QQg points with a reference.
    All are within 3.8e-10.
  - The 8 points beyond 1e-10 contain a small-Gram triangle member
    (E/A 1.5e-8 to 3.6e-8, see above), which huashengshao/TIRust#31
    addresses.
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
| gg_QQ | `N4-HHLH-g2|Q|Q|g1` | 100 | 100 | 100 | 3.4e-15 | 70 | 0 | 0 | 75 us | 78 us | 6.9 ms | 0.011 |
| gg_QQ | `N4-HLLH-Q|g2|Q|g1` | 100 | 100 | 100 | 3.6e-14 | 100 | 0 | 0 | 78 us | 85 us | 7.3 ms | 0.011 |
| gg_QQ | `N4-LHHL-Q|g2|Q|g1` | 100 | 100 | 100 | 8.3e-15 | 100 | 0 | 0 | 74 us | 79 us | 7.1 ms | 0.011 |
| gg_QQ | `N4-LLHL-g2|Q|Q|g1` | 100 | 100 | 100 | 4.6e-15 | 100 | 0 | 0 | 65 us | 72 us | 6.6 ms | 0.010 |
| gg_QQg | `N5-HHHLH-g2|g3|Q|Q|g1` | 200 | 189 | 189 | 2.0e-14 | 0 | 21 | 165 | 974 us | 1.1 ms | 31.3 ms | 0.031 |
| gg_QQg | `N5-HHLHH-g2|Q|Q|g3|g1` | 200 | 196 | 196 | 1.4e-13 | 1 | 104 | 79 | 898 us | 1.1 ms | 38.8 ms | 0.023 |
| gg_QQg | `N5-HHLLH-g2|Q|g3|Q|g1` | 200 | 200 | 200 | 1.7e-11 | 14 | 178 | 0 | 1.8 ms | 2.1 ms | 42.7 ms | 0.042 |
| gg_QQg | `N5-HLHHH-Q|Q|g2|g3|g1` | 200 | 200 | 199 | 2.1e-10 | 0 | 0 | 200 | 847 us | 953 us | - | - |
| gg_QQg | `N5-HLLHH-Q|g2|Q|g3|g1` | 200 | 200 | 199 | 1.0e-10 | 36 | 164 | 0 | 4.5 ms | 4.8 ms | 41.3 ms | 0.109 |
| gg_QQg | `N5-HLLLH-Q|g2|g3|Q|g1` | 200 | 200 | 198 | 1.9e-10 | 37 | 163 | 0 | 9.8 ms | 13.4 ms | 40.0 ms | 0.246 |
| gg_QQg | `N5-LHHHL-Q|g2|g3|Q|g1` | 200 | 200 | 199 | 3.8e-10 | 28 | 172 | 0 | 14.1 ms | 14.2 ms | 42.1 ms | 0.334 |
| gg_QQg | `N5-LHHLL-Q|g2|Q|g3|g1` | 200 | 200 | 198 | 1.8e-10 | 32 | 168 | 0 | 2.4 ms | 3.1 ms | 41.1 ms | 0.058 |
| gg_QQg | `N5-LHLLL-Q|Q|g2|g3|g1` | 200 | 200 | 199 | 1.5e-10 | 57 | 143 | 0 | 691 us | 806 us | 37.7 ms | 0.018 |
| gg_QQg | `N5-LLHHL-g2|Q|g3|Q|g1` | 200 | 200 | 200 | 5.2e-11 | 17 | 183 | 0 | 1.5 ms | 2.0 ms | 38.6 ms | 0.038 |
| gg_QQg | `N5-LLHLL-g2|Q|Q|g3|g1` | 200 | 200 | 200 | 3.2e-13 | 27 | 173 | 0 | 537 us | 728 us | 37.1 ms | 0.014 |
| gg_QQg | `N5-LLLHL-g2|g3|Q|Q|g1` | 200 | 200 | 200 | 8.4e-14 | 21 | 179 | 0 | 753 us | 793 us | 38.9 ms | 0.019 |

## Regression test

`crates/iregi-rust/tests/onia.rs` runs from `tests/fixtures/onia-swave.json`
(`make_fixture.py`):
- 12 degenerate lanes (boxes, a triangle and pentagons, both processes) must
  agree with the exact-identity reference to E/A <= 1e-9;
- 5 Coulomb lanes (a triangle, two boxes and two pentagons) must agree to
  1e-9 with original IREGI's consistency-checked arrays;
- the gg_QQ Coulomb box family (the box and its 10 pinches) evaluated in one
  `AmplitudeWorkspace` must succeed on every member, be bit-identical with
  sharing on and off, and give the box within 1e-9 of IREGI.

The Coulomb triangle's scalar moment must also equal the v = 0 value
C0 = −(m²)^(−1−ε) Γ(1+ε) / (2ε(1 + 2ε)), to 1e-12.

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
| `build.py` | binaries and their provenance (`target/onia/build.json`) |
| `drivers/iregi_onia.f90` | original-IREGI driver |
| `drivers/collier_scalar.f90` | COLLIER scalar C0/D0, the arbiter |
| `mre/` | minimal reproducers of the original-IREGI defects |
| `crates/iregi-rust/examples/onia_driver.rs` | TIRust driver: per-lane masses and optional powers |
| `crates/iregi-rust/examples/onia_amplitude.rs` | TIRust family driver: `AmplitudeWorkspace`, per-member arrays, sharing on/off timing |
