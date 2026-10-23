# Scalar six- and seven-point reductions from supplied invariants

This is a design note for a remaining interface gap. The ordinary scalar wrapper
rejects all requests with six or more propagators. The direct
`ReductionPlan` interface also rejects supplied scalar products in this domain.
Its coordinate padding cannot be used to construct a denominator identity, and
IBP requires physical Gram constraints too. Coordinate inputs remain available.
A separate opt-in `CertifiedScalarReductionPlan` and workspace now implement
exact full-Taylor checking and chart-grouped IBP execution for regular rank-four
six/seven-point scalar inputs. Integration tests are pending; this is not yet a
qualified expansion of supported coverage.

Let `R_ij=(p_i-p_j)^2-m_i^2-m_j^2`. For six denominators whose external
differences span four dimensions, an invertible `R` gives
`b=R^-1 1`, `sum(b)=0`, and the exact identity `sum(b_i D_i)=1`.
For seven denominators, the same identity can use any suitable six-element
subset. Six subsets containing the chosen origin suffice when the complete
denominator vectors have full rank. Inverting a generic symbolic seven-point
`R` is invalid on the physical four-dimensional domain.

The existing scalar compiler already uses the six-point pinching formula for
an ordinary scalar request. It does not certify the dimensional constraints of
an arbitrary supplied invariant matrix. For example, let `G_0i=0`,
`G_ii=-1` for `i=1,...,5`, all remaining off-diagonal entries zero, and all
squared masses one. Then `R^-1 1=(1/3,-1/9,-1/9,-1/9,-1/9,-1/9)` and
`sum(b)=-2/9`. The dimension-shifted six-point remainder is nonzero, so using
the physical pinching formula would omit a finite contribution. These supplied
invariants describe five independent spacelike directions, outside the supported
four-dimensional external domain. Merely forwarding a rank-zero PV request into
IBP therefore does not establish a safe scalar interface.

## Required compatibility check

Construct the relative Gram matrix
`K_ij=(s_0i+s_0j-s_ij)/2`, with `s_ij=(p_i-p_j)^2`. For a nonsingular
four-by-four principal block `A`, every residual

```text
K_uv - K_uA A^-1 K_Av
```

must vanish for indices outside that block. There is one independent residual
for six propagators and three for seven. This checks the full rank constraint;
testing only a single six-point `sum(b)` is insufficient for a seven-point
matrix. All matrix operations and residual simplifications must use Symbolica.
Alternative principal blocks must be selected lazily, so a zero determinant
in an inactive block does not invalidate a regular sample. Lower-rank Gram
domains need their own profiles and must remain explicit failures until those
profiles are implemented.

An expected-zero residual is distinct from a cancellation warning. Evaluation
must certify every Taylor coefficient of the exact represented input. A small
residual cannot be accepted merely because it fits a roundoff estimate: that
would silently replace the supplied invariant domain. The prototype in
[exact-invariant-certificate.md](exact-invariant-certificate.md) uses Symbolica's
certified ball algebra, increasing precision until every residual is an exact
zero enclosure or one excludes zero. A ball merely containing zero remains
unresolved. Testing only the primal would accept Dual directions that leave the
rank-four invariant variety. No binary64 conversion enters higher-precision
checks.

## Integration qualification

1. Keep the first explicit invariant-only scalar path in IBP; its certified
   covectors only supply linear denominator relations. Cartesian moments remain
   outside this opt-in interface.
2. Prepare exact bordered-minor certificates separately from recurrence aliases.
3. Run the exact compatibility certificate before reduction, separate from
   denominator and cancellation monitors, including full Dual components.
4. Enable the scalar wrapper only after direct invariant inputs, alternative
   charts, incompatible matrices, incompatible Dual directions, and mixed
   valid/invalid batches pass independent tests.

The coordinate-based PV compiler now has a separate six-chart seven-point
identity implementation. Its exact and Dual regression uses an initial chart
that is singular at the selected sample while another is regular. That work
does not by itself close the invariant-only scalar interface gap.
