= Algebra reuse audit
All symbolic algebra belongs to Symbolica and the FeynKit / Spenso / Idenso
ecosystem. Before introducing a missing operation, record three checks:
Symbolica APIs and implementation; Symbolica usage/examples/tests; and
FeynKit/gammaloop facilities and tests. An adapter or data-layout conversion
must not grow an independent polynomial, simplification or matrix engine.

== Existing facilities to reuse
- *Tensor identities and contractions:* Idenso contracts metrics and reduces
  supported gamma/color identities; Spenso carries tensor structure. Keep
  Lorentz dimension and any gamma-five prescription explicit.
- *Coefficient extraction:* FeynKit's tensor reducer uses Idenso contraction
  without global expansion, #raw("to_dots()") and the shared
  #raw("coefficient_list") collector. Preserve scalar factorization until
  polynomial extraction requires expansion.
- *Propagator and scalar-product algebra:* FeynKit's #raw("IntegralFamily")
  uses Symbolica for polynomial conversion, exact rank and linear solves,
  basis completion and mappings. Its Gaussian-rational conversion avoids
  statistical expression-field zero tests that may discard small coefficients.
- *Numerical preparation:* Compile reusable Symbolica expressions outside
  the sample loop. Evaluation and scalar-master types must preserve the
  selected precision and all Dual components without binary64 round trips.
- *Laurent coefficients:* IBP/PV factors and native normalization use cached
  Symbolica #raw("series") templates and #raw("coefficient") extraction.
  The three reuse checks are the pinned Symbolica
  #raw("src/atom/core.rs") and #raw("src/poly/series.rs") APIs; its
  #raw("src/derivative.rs") pole-series tests; and existing use in
  #raw("vakint/src/projection.rs") and OneLoopMaster's
  #raw("src/sheet_exact.rs"). No separate Laurent convolution engine is needed.
  Symbolica also evaluates the independent Gamma/series triangle oracle.

== HEPKit tensor-reduction optimizations
The local #raw("feynkit-tensor") reducer stores repeated-vector contractions
as symmetry orbits instead of factorially many labeled pairings. Its joint-color
dynamic program builds orbit intersections; alternating cycles build orbit Gram
matrices; Symbolica solves the smaller orbit system. A rank-20 isotropic
projection can have one orbit instead of $19!!$ labeled pairings.

The implementation caches canonical Weingarten coefficients and has a closed
isotropic path. General projectors are validated through rank 20; the special
all-equal isotropic route reaches rank 32. These are upstream limits, not a
statement of iregi-rust's current supported domain.

For denominators with external momenta, declare an independent external basis
and average only the transverse space. Odd total tensor ranks need not vanish.
Keep $D$ symbolic while constructing high-rank projectors; early integer
specialization can make the universal metric basis singular. External Gram
inverses can amplify small-Gram numerical errors, so projector preprocessing
requires a conditioning-aware comparison against direct recurrences.

This projector reduces tensor numerators, not complete scalar integral
families. Reuse it where applicable and benchmark the resulting plan; no
factorial-count argument alone establishes a numerical speedup. Retain
explicit output/pairing limits and report their failures honestly.

== References and provenance
The untouched generalized IREGI entry point stops during factorial-table
initialization: the configured powers/rank request factorial 16, while the
helper rejects arguments above 12. A narrow repair in a generated reference
copy replaces that table expression by its exact binomial recurrence. The
untouched/repaired twins agree exactly on 24 ordinary cases; all 80 table
entries match independent integer binomials; seven generalized fixtures check
normalization, unit powers, mass derivatives and a tadpole dimension shift.
The original #raw("STOP") returns exit code zero, so the harness also requires
an explicit result record. Evidence lives in
#raw("validation/reference/fixtures.json") and the report's JSON snapshot.

OneLoopMaster documents corrected analytic-continuation defects in its
upstream OneLOop reference, including complex-mass box cases. Preserve each
disagreement as a visible case and adjudicate it with independent analytic,
high-precision or parameter-integral evidence. Agreement must not mean
reproducing a known incorrect branch.

#set text(size: 8pt)
*Primary references:* #link("https://arxiv.org/abs/1405.0301")[Alwall et al., appendix C.1];
#link("https://hepkit.org/docs/hepkit/TensorReducer.html")[HEPKit TensorReducer];
#link("https://arxiv.org/abs/2408.05137")[Goode et al., tensor projectors];
#link("https://arxiv.org/abs/1701.04493")[Collins–Matsumoto, Weingarten recurrence];
#link("https://github.com/alphal00p/ninjars/blob/main/docs/benchmark/benchmark.typ")[NinjaRS report style].
Local source audit: FeynKit's #raw("feynkit-tensor/src") projector modules and
#raw("feynkit-graph/src/integrals.rs"); original IREGI's
#raw("IREGI4ML5_interface.f90"), #raw("ti_reduce.f90"), #raw("si_reduce.f90"),
#raw("mis_warp.f90") and #raw("funlib.f90").
