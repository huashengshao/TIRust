# Analytic two-point recurrence at singular Cayley matrices

The legacy validation triangle with squared masses `4,9,16` and first squared
distance `1` contains a pinched bubble at the pseudothreshold
`s=(sqrt(9)-sqrt(4))²`. Its Cayley matrix is
`R=[[-8,-12],[-12,-18]]`: `det(R)=0`, while the augmented matrix
`S=[[0,1,1],[1,R00,R01],[1,R10,R11]]` has determinant `2`. Original IREGI
returns stable complete tensors. The initial Rust binary64 baseline rejected
the IBP lane because a generic symbolic inverse of R was evaluated at this
singularity. PV/hybrid values passed the same reference comparison.

The existing exceptional compiler follows the original
`si_reduce.f90:968–1070`. Its shortened Eq. (22) omits a term proportional to
`S⁻¹[0,0]` because that coefficient vanishes on `det(R)=0`. Differentiating
that shortened expression transversely would lose a nonzero derivative of the
coefficient. This is an implementation limitation of that expression, not a
physical nondifferentiability at the massive pseudothreshold.

An analytic alternative follows directly from Eqs. (10), (14), (18), and (19)
of [Duplančić–Nižić, hep-ph/0303184](https://arxiv.org/pdf/hep-ph/0303184), using
the existing normalization `J_h=π^(-h) I_(4+2h-2ε)`.

For a raised power, let `w=v-e_l` and `u_j=w_j J_h(w+e_j)`. The dimension
identity gives `sum(u_j)=-J_(h-1)(w)`. The unsummed IBP identities supply
`R u=T`, where

```text
T_i = sum_j w_j J_h(w+e_j-e_i) - (5+2h-|v|-2ε) J_h(w).
```

Consequently `S (0,u)=(-J_(h-1)(w),T)`, and row `l+1` of `S⁻¹` determines
the requested integral after division by `w_l`. This is exactly the existing
unsummed S-row formula; its validity extends over every invertible S and does
not require `det(R)=0` or a zero dimension divisor.

For shifted unit powers, retain the full dimension identity:

```text
J_h(v) = -[S⁻¹[0,0] J_(h-1)(v)
             + sum_i S⁻¹[i+1,0] J_(h-1)(v-e_i)]
           / (3+2h-|v|-2ε).
```

Retaining the first term makes this expression analytic across `det(R)=0`
and preserves transverse Dual derivatives. For two active propagators the
domain `h >= |v|-2` is closed under these recurrences. Raised-power terms
reduce the total power by one and decrease `h` by at most one. Unit-power
terms decrease positive `h` until the ordinary B0 master. Pinched terms use
the existing tadpole recurrence. This covers the bubble descendants arising
from ordinary four-dimensional tensor reduction.

The scalar compiler exposes this through
`ScalarCompiler::enable_gram_regular_bubbles`; core IBP plans now enable it
following the exact regression and ordinary legacy tensor checks below.
Symbolica constructs/inverts S,
performs every algebraic operation, and emits a lazy `IF(det(S), S-route,
existing-route)`. A separate S-geometry cache avoids marking R as singular in
the existing exceptional-geometry cache. Complete conditional results share
the ordinary recurrence cache and master symbols. Generalized requests
outside the terminating domain retain their existing behavior; this change
does not claim their exceptional transverse derivatives are qualified.

The exact symbolic regression passed on the integrated test build in 0.17 s,
with 48 Laurent comparisons across six requests. Its durable evidence is
[`s-recurrence-formal.json`](../../validation/reference/snapping/s-recurrence-formal.json).
It compares all four Laurent maps with generic R recurrences and independently
prepared structurally singular R maps. At the pseudothreshold those master
maps are not independent, so the latter check substitutes the exact analytic
A0/B0 values at `mu²=1`, including
`B0_finite=2+4 log(2)-6 log(3)` and vanishing IR coefficients. An initial
free-master comparison correctly exposed this invalid oracle assumption;
the generic nonsingular map comparison retains free masters.
The complete 30-case original-IREGI legacy tensor suite was then rerun. Its
22 primary binary64 successes were preserved. The exact pseudothreshold now
reaches the cancellation monitor rather than a nonfinite recurrence; all
eight IBP triangle/box rejections pass explicit 128-bit rescue with 256-bit
verification and unchanged requested 53-bit accuracy. Every tensor component
and Laurent slot agrees with the reference at the original `1e-10` tolerance,
with maximal normalized error `2.93e-12`. These results and primary failures
remain visible in `validation/reference/snapping/rust-validation-after-s.json`
and `rescue-validation-after-s.json`. Complete tensor Dual derivatives remain
a separate gate.

The integrated full-30 regression then passed with explicit unstable-only
rescue and retained primary diagnostics. A fresh run of all 31 original
reference fixtures also passed, preserving one verified generalized-triangle
rescue. The independent Gamma oracle was expanded to retain its 240 generic
symbolic requests and repeat all 240 at exact `s=-3` with the S route enabled.
All 1,920 Laurent comparisons passed. Binary identities, logs, and validation
durations are recorded in `validation/reference/post-s-regressions.json`;
these durations are not kernel performance measurements.

The scalar backend's new analytic B0/DB0 pseudothreshold chart has separately
passed an exact helper identity and 384-bit first, second, and mixed finite
differences at ordinary positive-real scales. The finite derivative in this
example is `dB0/ds = 5 log(3/2)-2`, obtained from the Feynman-parameter
quadratic `(x+2)^2`. Its current run is recorded in
`target/physical-bubble-pseudothreshold.log`; the final DB0 scaled error was
`7.79627e-35`. Genuine seeded thresholds remain explicitly rejected.

This does not yet qualify complete tensor derivatives through the S route.
Extreme common scales `2^±600` also remain a separate scalar-backend gate:
the initial chart run rejected them with `NonFiniteOutput` from intermediate
Dual inversion powers. Exact power-of-two lane normalization is under
development. No precision or scale robustness claim follows from the
ordinary-scale derivative check alone.
