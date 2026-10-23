# Cartesian high-point identities in dimensional regularization

The PV compiler's high-point tensor identities do not discard an evanescent
loop-momentum component. Write `q=qbar+qperp`, where all external shifts and
the requested Cartesian numerator indices lie in the four-dimensional
`qbar` subspace. With `D_i=(q+p_i)^2-m_i^2`, choose origin `p_0` and define

```text
k_i = p_i-p_0
f_i = k_i^2-m_i^2+m_0^2
Z_ij = 2 k_i·k_j
D_i-D_0 = 2 k_i·(qbar+p_0)+f_i.
```

The common `qperp^2` term cancels in the denominator difference. When four
external differences form a complete basis of the Cartesian subspace,

```text
qbar^mu = -p_0^mu
          + sum(i,j) k_i^mu (Z^-1)_ij (D_j-D_0-f_j).
```

Multiplication by the remaining numerator monomial and by the denominator
product gives the five-point Cartesian recurrence used in `pv.rs`. It is an
identity before integration, including in `D=4-2epsilon`. The pinched tensor
integrals still carry their full dimension-dependent trace factors. This
argument does not assert that a scalar dimension-shifted pentagon can be
replaced by a four-dimensional scalar pentagon: that is a separate interface.

For six or more denominators, coefficients satisfying

```text
sum(c_i)=0
sum(c_i p_i^mu)=0
sum(c_i (p_i^2-m_i^2))=1
```

give the exact identity `sum(c_i D_i)=1`. In relative coordinates this is
the five-row system with rows `k_i^mu` and `f_i`, whose right-hand side is
`(0,0,0,0,1)`. Symbolica solves this system. The identity is dimension
independent and can lower one denominator power without expanding the
numerator.

A seven-point system has six columns after choosing the origin. Its six
five-column charts cover every system with full row rank. The symbolic
compiler retains them with lazy determinant selection so a singular initial
chart does not invalidate a regular sample. Overlaps give different valid
representations of the same integrand. Their complete integrated values and
their analytic Dual jets therefore agree; chart selection does not introduce
a physical branch boundary. Lower-rank systems and ill-conditioned charts
still require explicit diagnostics or their own analytic profiles.

These identities require coordinates consistent with the invariants used in
`f` and `Z`. In particular, they do not justify using zero-filled coordinates
for an invariant-only scalar API, or replacing independently snapped Gram
products by differences of separately snapped squared distances.
