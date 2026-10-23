# Exact zero UV residues from power counting

For a Cartesian numerator monomial of degree `r`, total positive propagator
power `nu`, and dimension `D=4+2h-2epsilon`, the radial large-momentum behavior
is `R^(D+r-2nu-1)`. If `delta=4+2h+r-2nu` is negative, the large-radius integral
converges uniformly in a neighborhood of `epsilon=0`. It therefore produces
no UV Laurent pole. A one-loop integral has no nested loop UV subdivisions.
Finite-momentum singularities can still generate IR poles and are unaffected
by this argument. Momentum translations, complex masses, and normalization
factors analytic at epsilon zero do not change the conclusion.

This gives an exact final-output rule: set the UV slot to zero when
`delta<0`. It is not a numerical cutoff. Apply it only after the recurrence
has been compiled: UV terms of intermediate pinched or dimension-shifted
integrals can contribute finite rational terms and must remain available.
Do not change the IR or finite slots. The rule proves no converse when
`delta>=0`; angular symmetry can remove a pole in particular monomials, but
that requires a separate argument.

Useful regression families are:

- Rank-zero triangles: UV zero, retaining massless IR poles where present.
- Rank-four pentagons, and six- or seven-point moments through rank seven:
  UV zero by power counting.
- Generalized bubble powers `(2,1)` at `h=0`, rank zero: UV zero. At `h=1`,
  the superficial degree is zero, so the UV result must remain unchanged.
- Ordinary B0, including the scaleless bubble: degree zero, preserving the
  UV residue and the scaleless cancelling IR residue.
- Ordinary A0: degree two, preserving its squared-mass UV residue.

The canonical physical pentagon in
`validation/pv-high-rank/canonical-pentagon-adjudication.json` is an independent
motivation: original IREGI IBP returns a spurious UV residue for a rank-four
five-point moment. Power counting rules that residue out without using the
new reduction implementation or any numerical finite-integral oracle.

Validation status: the first combined regression run reached its 120-second
limit without identifying the active fixture; it is not a passing result.
The generalized-bubble and massless-pole checks now run separately from the
ignored high-point preparation test, with per-fixture progress messages. Their
updated runtime qualification is pending. The high-point test requires an
explicit time budget and reports preparation duration for each topology.
