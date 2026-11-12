# Original IREGI on S-wave quarkonium kinematics: minimal reproducers

Original IREGI here is MadGraph's vendored copy (binary64, gfortran -O3).
The only repair is the 32-bit `factorial_pair` overflow in `funlib.f90`
(`target/reference/patched`). IREGI is not modified in any other way. Every
input below is exact in binary64: each massless leg is exactly lightlike,
p_c = p_c~, and every invariant is exact. On-shell snapping therefore plays no
role.

## Build

```bash
python3 benchmarks/onia/build.py
```

This builds `target/onia/bin/iregi-onia` (the driver `drivers/iregi_onia.f90`, a
copy of the stability study's IREGI driver) and `target/onia/bin/collier-scalar`.
The OneLOop probe links against IREGI's bundled OneLOop 3.4 object alone:

```bash
gfortran -O2 -I target/reference/objects benchmarks/onia/mre/olo_d0_probe.f90 target/reference/objects/avh_olo_foriregi.o -o target/onia/bin/olo-d0-probe
```

## 1. Pentagons of g g > c c~ g: arrays off by 1e62, flagged stable

`iregi_pentagon_mode0.in` (IBP) and `iregi_pentagon_mode1.in` (PV) hold one
rank-5 pentagon, `N5-LLLHL-g2|g3|Q|Q|g1`, at the first gg_QQg point. The beams
are along z. Its propagators are massless except the one between c and c~.

```bash
target/onia/bin/iregi-onia < benchmarks/onia/mre/iregi_pentagon_mode0.in
```

**Observed.**
- The lane reports `"stable":true`, and the scalar moment is right.
- Rank-5 moments are off by 63 orders of magnitude. For example, the finite
  part of the moment l_z^5 is 2.168e63 - 1.351e64 i, where the reference is
  -49.01 + 71.50 i.
- Against the reference, E/A is 1.6e62 (IBP) and 4.9e42 (PV).

**Reference.** This pentagon's external momenta span three dimensions, so its
Gram determinant vanishes exactly. `affine_reference.py` writes it exactly as
a sum of boxes (and their pinches) with rational coefficients, from the
identity sum_i c_i D_i = K. Each of those integrals is non-degenerate, and is
evaluated by TIRust at 256 bits and confirmed by quad IREGI to 1e-16. TIRust's
binary64 default agrees to 2.1e-14 here.

**Scope.** On the 200 gg_QQg points, 2,225 lanes are off by more than 1
(E/A > 1) and all of them are flagged stable:
- **Pentagons:** 1,504 of the 1,800 pentagon lanes, in all 9 non-Coulomb
  pentagon topologies, with E/A up to 8.4e72. The 3 Coulomb pentagons fail as
  well (item 4).
- **Boxes:** 721 of the 6,600 box lanes, in 10 of the 33 non-Coulomb box
  topologies, with E/A up to 3.9e54. Nine of these boxes have an exactly
  vanishing Gram determinant; the tenth is item 3.

**Likely origin.** IREGI's IBP and PV recursions divide by the vanishing Gram
determinant, or by a nearly vanishing pivot, without detecting it. The
`STABLE_IREGI` flag stays true.

## 2. PV mode on zero-Gram boxes of g g > c c~: wrong by O(1), flagged stable

`iregi_box-pv_mode1.in` holds the box `N4-LLHL-g2|Q|Q|g1` at the point where
c c~ is at rest. Its masses are (0, 0, m_c^2, 0), with m_c = 1.5 GeV, and its
shifts are exact small numbers.

**Observed.**
- IBP mode (`iregi_box-pv_mode0.in`) agrees with the reference to 2.5e-16.
- PV mode returns `"stable":true` but is wrong: the finite part of l_0^4 is
  18.73 + 29.24 i, where the reference is -2.277 i. Its E/A is 0.48.
- On all 100 points PV is wrong by 1.0e-3 to 1.5, on both boxes with
  gluons opposite (`N4-LHHL-Q|g2|Q|g1`) or adjacent (`N4-LLHL-g2|Q|Q|g1`).
- On the Coulomb-exchange box PV correctly reports unstable.

**Likely origin.** PV's Gram inversion is not guarded for an exactly
vanishing Gram determinant when the modified Cayley determinant does not
vanish.

## 3. Bundled OneLOop 3.4: wrong real part of an IR-finite D0

`olo_d0_box.txt` holds one scalar box, in OneLOop's argument order:
p1 p2 p3 p4 p12 p23 m1 m2 m3 m4 (squared masses).

```text
0  29.79339599609375  1.9231414794921875  0  -24.353713989257812  63.433074951171875  m2 m2 0 m2,  m2 = 1.9231414794921875
```

The masses are (m, m, 0, m). Leg p3^2 = m^2 is on shell at the massless
propagator, and p1^2 = p4^2 = 0.

```bash
target/onia/bin/olo-d0-probe < benchmarks/onia/mre/olo_d0_box.txt
```

| code | D0 (OneLOop normalization, mu = 3 GeV) |
|---|---|
| OneLOop 3.4 (bundled with IREGI) | -1.38730e-2 + 2.59363e-3 i |
| COLLIER 1.2.10 (`collier-scalar`) | -1.04346e-2 + 2.59363e-3 i |
| TIRust (oneloopmaster), binary64 and 256 bits | -1.04346e-2 + 2.59363e-3 i |

The imaginary parts agree. The real part differs by 33%, with no pole and no
warning.
- **Frequency.** On gg_QQg, the box `N4-HHLH-g2|Q+g3|Q|g1` contains this D0
  type. IREGI's arrays are then wrong (E/A > 1e-6, mostly O(1)) on 76 of 200
  points in binary64. IREGI's quad build is wrong on a different subset, 108
  points, because the precision moves the branch decision. TIRust stays within
  4.6e-12 on all 200.
- **Argument order.** OneLOop 3.4 gives the same wrong value for each argument
  order that IREGI's `D0C1` uses, so the defect is in OneLOop 3.4's D0, not in
  IREGI's dispatch.
- **Tensor arrays.** Where IREGI's D0 happens to be right, its whole rank-4
  array agrees with TIRust to 1e-12.

**Likely origin.** This is a three-mass box with real masses.
oneloopmaster's README (pinned revision 09b2c74; `tests/d0_real_mass_orderings.rs`)
records the same defect class: OneLOop's real-mass `boxf33` chart depends on
the labelling. It has a wrong real or imaginary part in some relabellings of
about a quarter of three-mass boxes, OneLOop 3.6 included. oneloopmaster
evaluates these boxes with a contour representation instead, so updating
IREGI's bundled OneLOop alone would not fix it.

## 4. The Coulomb-exchange integrals

TIRust refuses these integrals; IREGI evaluates them. IREGI's `C0C1` recognizes the Coulomb
triangle (m_1 = 0, p_1^2 = m_2^2, p_3^2 = m_3^2, p_2^2 = (m_2 + m_3)^2). It
returns the v = 0 value in dimensional regularization,
C0 = -(m^2)^(-1-eps) Gamma(1+eps) / (2 eps (1 + 2 eps)).

`coulomb_check.py` checks IREGI on all 1,600 Coulomb lanes:
- **Scalar C0.** It equals this formula to 7e-16, on all 300 Coulomb-triangle
  lanes.
- **Partial-fraction combination.** The exact combination of doubled-gluon
  integrals, built from 2 D_g = D_c + D_c~ and evaluated by IREGI's own
  general API (a separate code path), is compared with each tensor result:

  | IREGI result | agreement |
  |---|---|
  | binary64 IBP, Coulomb triangles of both processes and the gg_QQ box | 9e-14 |
  | quad IBP, triangles and boxes | 4e-13 |
  | binary64 IBP, gg_QQg Coulomb boxes | 87–109 of 200 lanes within 1e-10, up to 4e4 off |
  | Coulomb pentagons | fail as in item 1 (quad IBP within 1e-10 on 83–183 of 200) |

- **PV mode.** It reports these integrals unstable.

## Reporting

These are reproducers only. Following the project's rule for reference codes,
IREGI and its bundled OneLOop are not modified. Items 1 and 2 belong to IREGI's
reduction. Item 3 belongs to OneLOop's three-mass box chart, which IREGI bundles
(3.4 here); OneLOop 3.6 has it too, per oneloopmaster's README.
