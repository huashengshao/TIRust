# N4 trace and routing identities for the COLLIER reference extension

Source review, 2026-10-23. This specifies independent checks for the proposed
N4 ranks 0–3, batches 1/32 extension. No build, numerical run, timing, or
production change was performed for this review. The equations below are an
analytic specification; executable polynomial preparation and identity checks
must use the existing Symbolica CLI facilities.

## Conventions and scope

Write `D_i=(q_D+p_i)^2-m_i^2+i0`, `p_0=0`, `D=4-2epsilon`, and
`q_D=qbar+qperp`, with all shifts and returned numerator indices in the
four-dimensional space. The metric is `eta=diag(1,-1,-1,-1)`.
`D^alpha` below denotes an unweighted Cartesian box moment, **not** a
propagator power. `C_i^alpha` denotes the triangle obtained by removing line
`i`, while keeping its numerator in the original box variable `qbar`.
`C_i` without indices is its scalar moment.

The input masses are squared masses. These fixtures have positive real squared
masses and purely spatial real shifts. The Euclidean Feynman mass polynomial
is bounded below by the smallest squared mass, so there is no IR singularity
or threshold on this domain. Boxes of rank at most three are UV finite.

COLLIER defines its denominators and D-dimensional integral at manual
lines 930–943, decomposes ranks two and three at 1045–1049, and exposes
Cartesian indices 0–3 at 1961–1969. Its published conventions give the same
definitions and tensor decomposition. [COLLIER, Section 2](https://arxiv.org/html/1604.06792v2#S2).
The local `InitTensors.F90:234–237` supplies the spatial metric signs.
All COLLIER paths in this note refer to
`target/reference/collier/source/collier-COLLIER-1.2.10`.

## Rank-two and rank-three traces

The required coefficientwise identities, through the finite Laurent term, are

```text
D^{00} - D^{11} - D^{22} - D^{33} = C_0 + m_0^2 D^scalar,

D^{00nu} - D^{11nu} - D^{22nu} - D^{33nu}
    = C_0^nu + m_0^2 D^nu,                 nu=0,1,2,3.
```

These are not exact identities at all orders in epsilon for a four-dimensional
numerator. The exact difference between the four-dimensional trace and the
D-dimensional denominator-cancellation identity is `+2epsilon D_00` at rank
two, and `+2epsilon sum_i p_i^nu D_00i` at rank three. Here `D_00` and `D_00i`
are **covariant metric coefficients**, not Cartesian time components.
Contracting the metric part gives `4` versus `D` at rank two, and `6` versus
`D+2` at rank three; the momentum-only pieces agree.

These metric coefficients have no pole in the stated domain. Equivalently,
the defect involves epsilon times a six-dimensional scalar/vector box, which
is UV convergent and IR finite here. Consequently the defect contributes no
pole or finite coefficient. The published PV relations retain the explicit
`D-4` factors, and identify the metric coefficients as IR finite.
[Denner–Dittmaier, Eqs. (5.9)–(5.11) and Section 5.8](https://arxiv.org/html/hep-ph/0509141v2#S5.SS1).

There is therefore **no finite evanescent correction to add to these two
checks**. This conclusion must not be applied to box rank four or to
bubble/triangle traces. COLLIER explicitly starts UV tensors at rank `2N-4`
(`TensorReduction.F90:1346–1351`, `1374–1386`); its box UV metric construction
starts at rank four (`BuildTensors.F90:478–484`). Original IREGI likewise retains
UV-origin finite corrections in `pave_reduce.f90:1265–1267`.

## Pinch-zero routing

Removing line zero leaves shifts `[p_1,p_2,p_3]`. For a zero-origin triangle
call, set `q'=q+p_1` and use shifts `[0,p_2-p_1,p_3-p_1]` with squared masses
`[m_1^2,m_2^2,m_3^2]`. Then

```text
C_0^nu(original q) = C_local^nu(q') - p_1^nu C_local^scalar.
```

Thus the rank-three trace RHS is the contraction of the prepared local
numerator `q'_nu-p_1^nu`, plus `m_0^2 D^nu`. Use Symbolica for this polynomial
and for `(qbar_0^2-qbar_1^2-qbar_2^2-qbar_3^2)*qbar_nu` on the box. Do not
negate the whole odd-rank tensor. The original and COLLIER denominators both
use `q+p_i`; the Feynman-parameter shift is consequently `q=l-sum x_i p_i`.

For subsequent rank-two triangle checks the corresponding local numerator is
`(q'_nu-p_1^nu)*(q'_rho-p_1^rho)`, again prepared by Symbolica. Pinches `i>0`
retain the already-zero origin and need no numerator translation.

## Additional odd-sign and component checks

For each `i=1,2,3`, define `f_i=p_i^2-m_i^2+m_0^2`. The denominator difference
has no `qperp` term, and gives an exact same-D check:

```text
2 sum_mu eta_mu p_i^mu D^mu = C_i - C_0 - f_i D^scalar.
```

This independently tests the spatial sign, odd-rank convention, and pinch
ordering. Its rank-two/three extensions are

```text
2 sum_mu eta_mu p_i^mu D^{mu nu}
    = C_i^nu - C_0^nu - f_i D^nu,
2 sum_mu eta_mu p_i^mu D^{mu nu rho}
    = C_i^{nu rho} - C_0^{nu rho} - f_i D^{nu rho}.
```

The last check uses UV-divergent rank-two triangles whose UV terms cancel;
retain all four native slots during that cancellation. For the present purely
spatial shifts, every component with an odd number of time indices also
vanishes by time reflection. Spatial choices of `nu` make the rank-three trace
nontrivial.

## Laurent normalization and acceptance

Use native order `[UV single, IR single, IR double, finite]`, unweighted moments,
and the same scale `mu` on both sides. COLLIER setters receive `mu^2`. Within
this IR-finite envelope, conversion is
`U=i*pi^2*Uc`, `I=J=0`, `F=i*pi^2*(Fc-(EulerGamma+ln pi)*Uc)`.
There is no additional `(-1)^rank` factor. The existing
`validation/collier-reference/convention-review.md` records the full IR adapter
and source-level derivation; it must be used if the domain is expanded.

Apply the original `1e-10` mixed absolute/relative criterion separately to each
Laurent slot and each component. Normalize the rank-two trace and vector
denominator-difference identity by `Q^2`, and the rank-three trace by `Q`, for
one declared positive mass scale `Q`. Do not apply the homogeneous rank-three
box normalization separately to its lower-rank triangle term. Preserve raw
residuals, normalization, statuses, and failed cases.

Reviewed inputs: `validation/collier-reference/extensions/n4-inputs.json`,
SHA-256 `e248fd30e098bb4178c42da68d7915b18ecd843cf661fbf6c1c0285d85eaf595`.
Frozen build receipt: `validation/collier-reference/ready-initial8-v3/build.json`,
SHA-256 `654d6221d0169ba06d0f2ae7f9615737d6b67aae0f0b57539e74ad518902fc80`.
The manual/source hashes are already bound by
`docs/research/collier-1.2.10-manifest.json`. A second read-only source reviewer
independently confirmed the trace defect, routing sign, and UV threshold. These
reviews do not replace the pending complete-array numerical qualification.
