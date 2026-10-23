# Public N6 reverse activation profile

The bounded N6 rank-seven run previously spent 118.047 seconds propagating
activation conditions before local checks. A separate private snapshot retains
the original algorithm and records substage timings and occurrence counters.
It deliberately stops after the reverse pass. No numerical reduction is run.

The resulting 152.995-second instrumented reverse pass spent 0.0505 seconds in
`union_paths`, 0.0132 seconds interning predicates, and 152.7915 seconds collecting
alias uses. It processed 26,114 incoming paths, with at most four per set and a
maximum 27-byte union. Only 15 predicate aliases were created. This evidence does
not support prioritizing suffix interning for this workload.

The traversal visited 270,405,574 nodes and 155,133,562 variable occurrences.
There were 65,995,224 alias occurrences and 65,970,100 duplicate insertions;
65,993,388 occurrences were demanded unconditionally by literal `1`. These
counts motivate deduplicating repeated references with Symbolica's existing
traversal API and avoiding redundant path storage after an unconditional use.
Any candidate must preserve the original treatment of numeric coefficients,
function names, power exponents, conditional contexts and lazy branch order.

The extra counters and set queries increase traversal cost, so these timings
are diagnostic rather than a measurement of unchanged production performance.
The run used CPU14, a private user-supplied Symbolica license, one Rayon thread,
and the frozen development-profile core/master archives. Its 240-second bound
was not reached; total elapsed time was 213.230 seconds including plan creation.

The public driver reports `workspace_failed` and exits with code 2 for the
intentional preparation stop. The raw runner retains that result; the separate
qualification identifies the explicit diagnostic stop and verifies that local
checks and numerical reduction were not entered. This is not a successful
full-array evaluation. Earlier 180/300-second public timeouts remain visible.

Source, build, input, binary and result bindings are recorded in
`validation/public-highpoint/reverse-profile-qualification.json`. The proposed
alias-use optimization remains a private, separately qualified experiment.
