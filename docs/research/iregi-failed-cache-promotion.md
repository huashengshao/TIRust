# Original IREGI: failed cached values can retain a successful status

Source-only audit, 24 October 2026. No reference source was modified, no new
binary was built, and no diagnostic kernel was executed. This note identifies
a conditional source-level cache/status hazard. It does not establish that the
hazard caused any observed N6/N7 disagreement or determine the correct integral.

## Source identity

The supplied MadGraph checkout is commit
`6c1b4684b46ad26c4f0ebc57eb9963649f350142`. The following untouched source hashes
also appear in the original reference's committed
`validation/benchmark/baseline-6b24c03-licensed/reference-build.json`:

| IREGI source | SHA-256 |
| --- | --- |
| `src/si_reduce.f90` | `c9cc407de5db60d043acb81b40de8a16f2c19bd3de1bb92c9643c60667d5bfe6` |
| `src/ti_reduce.f90` | `17d9a0f5e14b84f46667f532a1463a5734cd46c885467cfb6230e862d2ceb49a` |
| `src/binary_tree.f90` | `c47405dc762e00045ca0a9ada8325c7c31bfa488e8b4fae7a30fb852677cbdbb` |

Paths are relative to `references/madgraph/vendor/IREGI` in the supplied source.
All execution copies and future instrumentation must stay under this project.

## Conditional control-flow finding

`scalar_integral_reduce2` in `si_reduce.f90` initializes its result to zero at
line47 and returns early when global stability is false at line48. When
recycling is enabled, it creates a candidate node with `item%stable=.TRUE.`
at line80 and inserts/searches it at line86. Cache hits load the stored value
and assign the global stability flag from the cached flag at lines88–90.
`binary_tree.f90:404–446` inserts the actual node pointer and, on a hit, copies
the stored value and stability flag into the lookup result.

At `si_reduce.f90:224–229`, a failed matrix inversion sets global
`STABLE_IREGI=.FALSE.` and caches the current zero result, but leaves that
node's stability flag at its initial true value. Neighboring inverse-failure
branches, for example lines179–185 and264–270, explicitly set the cached
stability false. Thus the line226 branch has a concrete status-write asymmetry.

There is a related propagation concern at lines216–220: the function calls
`pavefun_reduce2`, caches its returned value, and returns without copying the
callee's global failure status into its own cached node. Other recursive
return paths similarly assign a cached value after child calls without a
corresponding cached-status assignment. This is a source observation, not proof
that every such path encounters a failing child.

The hybrid wrapper `pave_opt_reduce2` calls PV at line1230; when PV fails, it
explicitly resets global stability to true at line1232 before the IBP fallback
at lines1233–1234. If a previously failed value with a stale true cached flag
is then reused, the cache-hit path can present that value as successful. The
full failing-node reuse sequence still needs runtime instrumentation.

## Relation to retained results

Shared COLLIER attempt004 preserves original-IREGI mode failures and substantial
IBP/Hybrid finite-term differences for the unchanged mass-deformed N6/rank2
case. No branch trace connects that observation to the cache hazard above.
Neither an originally successful status nor agreement with another original
mode is by itself a correctness proof on that exceptional input.

The original eight stable bubble/triangle benchmark inputs already satisfy
their independent numerical admission gates. This note does not invalidate
those measurements or claim that their runs took a particular internal branch.
A later trace should check those paths before making any branch-specific claim.
The untouched reference and all existing failed/accepted results remain intact.

## Proposed bounded diagnostic, not authorized execution

After the active benchmark quiet window, propose a separately reviewed generated
reference copy with diagnostic-only counters/logs at cache insertion, failure
assignment, fallback reset and cache hit. Bind the untouched and instrumented
source inventories and verify that the instrumentation changes no arithmetic,
branch conditions, cache comparator, ordering or numerical outputs. Log the
node identity, complete recurrence indices, topology, dimension shift and
native status transitions; retain stderr and resource outcomes.

Start with the single retained mass-deformed N6/rank2 lane and original Hybrid
mode, rather than rerunning unrelated accepted or failed families. A20s request
bound and separately reviewed driver/object build cap are appropriate proposal
limits; no build or run follows from this note. A trace must show a failed
node later retrieved with a successful flag before calling the hazard reproduced.

Any proposed generated-copy repair is a separate step. It requires comparison
against the untouched failing case, all relevant untouched working cases, and
an independent high-precision or analytic identity oracle. No repair is made
here. TIRust should continue to keep failed coefficients and cache entries
explicitly unsuccessful, so incomplete values cannot become successful zeros.
