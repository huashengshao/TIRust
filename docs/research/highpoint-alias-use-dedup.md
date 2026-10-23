# Exact alias-use deduplication qualification

The reverse-pass profile found 65,970,100 duplicate alias-use insertions in the
ordinary public N6 rank-seven plan. Most occurrences were already demanded by
the unconditional path `1`. Predicate unions and interning together took less
than 0.1 seconds; recursively visiting repeated variable occurrences dominated.

The isolated candidate uses Symbolica's `get_all_indeterminates(true)` for
expressions without `IF`, filters variable atoms, and records each alias once.
Conditional expressions retain the original recursive traversal and lazy
branch context. Once an alias has an unconditional use, later paths cannot
change its union; an unconditional insertion therefore replaces earlier paths.
Zero paths remain present, preserving reachability behavior.

Three checks established the existing API and its required semantics:

1. Symbolica `src/atom/core.rs:1865–1895` exposes
   `AtomCore::get_all_indeterminates`, `contains_symbol`, and the distinct
   `get_all_symbols` traversal; `src/id.rs:923–1030` implements discovery.
2. Direct witnesses exercise function arguments and names, power exponents,
   repeated occurrences, and rational-polynomial numeric coefficients. The
   latter must remain opaque as they were in the original collector;
   `get_all_symbols(false)` would traverse them and is unsuitable here.
3. FeynKit `crates/feynkit-model/src/model.rs:1593` uses symbol discovery for
   dependencies; Idenso `crates/idenso/src/lib.rs:596` uses it while retaining
   opaque contraction aliases. The implementation delegates discovery to
   Symbolica and retains IREGI's original branch bookkeeping; no new algebra
   is added.

The exact probe passes 17 normalized-use-map cases and three existing numeric
or activation tests: inactive singular branches, per-sample scalar-master
activation, and conservative master-dependent activation. The complete paired
N6 reverse pass also agrees for all 25,700 reachable aliases. It checks every
normalized lazy predicate, root identity, complete reachable map, all 15
predicate alias definitions, the predicate cache and remaining normalized
uses. Both algorithms execute in the same process on the same plan.

The retained original collector took 110.745951 seconds and the candidate took
10.352944 seconds in that paired development-profile diagnostic. Instrumented
reverse preparation took 121.316286 seconds, and the full diagnostic process
took 167.356997 seconds within its 240-second bound. These numbers do not
qualify a controlled performance gate.

The process intentionally returns a preparation error immediately after the
proof. Local checks, scalar masters and numerical reduction are not executed;
there is no successful N6 tensor array in this evidence. The previous public
180/300-second preparation timeouts remain retained. A separate candidate from
the current core will exercise the full stability suite and public reduction.

The independent audit approved the exact symbolic evidence. Reproduction is:

```sh
source scripts/env.sh
source .local/symbolica-license.env
python3 validation/public-highpoint/run_use_dedup_exact.py
python3 validation/public-highpoint/run_use_dedup.py fresh-name 6 --limit 240 --cpu 14
python3 validation/public-highpoint/qualify_use_dedup.py
```

Raw runners refuse to overwrite captures; the fixed qualification script binds
the committed capture names. Use a fresh evidence directory and adapt its
capture-name binding when reproducing the complete qualification.

`validation/public-highpoint/use-dedup-qualification.json` binds source, input,
build, binaries, logs, results, the exact production patch and baseline
master/Symbolica/OneLoop archives. The full build guards every `.rlib` and
`.so` in all dependency search directories before and after compilation.
The production patch has SHA-256
`746b05e8a2a9060640d1c0ae4d5f3ec350e91199d817382653efde81139a35a3`.
