# High-point diagnostics and Python tooling checkpoint

The implementation goal remains active. This checkpoint does not qualify the
complete seven-propagator/rank-seven domain or establish performance parity.

## Six-point rank-seven disagreement

The restricted private ordinary six-point candidate returns all 330 Cartesian
moments and 1,320 complex Laurent slots for the retained positive-F fixture.
Its 128-bit and separately captured 256-bit results converge within a maximum
normalized difference of `5.420929e-32`, below the unchanged `2^-80` check.
All 990 pole slots are exactly zero at each precision. Independent saved-data
review confirms the arrays, input identity, provenance and execution bounds.

The original-reference comparison remains failed. Original IBP, PV and Hybrid
disagree in 90, 151 and 109 finite slots respectively at `1e-10`; original PV
also reports an unstable result. The exact real-zero identity for this real,
positive-F fixture finds 88 original-IBP violations above tolerance. TIRust's
real residual decreases from approximately `2.31e-32` to `2.40e-70` as working
precision increases. This identifies reference real-part inaccuracies without
certifying every imaginary part or changing the failed acceptance decision.
The independent temporal-component check evaluates `q0^6` (moment 126) with
Symbolica-prepared coefficients and six fixed SciPy quadrature configurations.
TIRust's normalized discrepancy plus estimated uncertainty is `3.25e-15`;
original IBP and Hybrid exceed `1e-10`. The uncertainty is empirical rather
than interval certified, and this check covers only that component. Its retained
sources, initial adapter-test failure, corrected tests and all rule outputs are
in `validation/temporal-sixpoint-oracle/q0-sixth-001/`.

The previously qualified private rank-two/rank-three results retain their
restricted scopes. Public integration is being prepared as an explicitly
selected six-point API, initially limited to rank three, with child retry and
operand collection isolated from ordinary reduction defaults.

## Seven-point structural qualification

`validation/highpoint-thin-n7-exact/` preserves the private source, build and
17 selected test results with independent verification. One generic-constructor
test remains explicitly unrun. No seven-point numerical result follows from
this qualification. The retained base fixture has a singular required subset;
a separately named general-position fixture is being checked before its first
numerical capture.

## Reference and Python checks

The separate COLLIER N5 identity audit supplements its immutable audit-pending
capture record. All 5,397 residual records and 18,231 complex comparisons pass
the unchanged tolerance, with routing and pinch/denominator/trace checks kept
separate from physical or timing qualification.

The first standalone development wheel builds, installs in a fresh environment
and passes all 17 numerical tests, 11 of 12 interoperability tests, and installed
consumer typing with zero errors or warnings. The remaining interoperability
test exposes an upstream graph SVG-rendering failure. The exact wheel and source
hashes, previous build timeouts and failing test remain recorded in
`validation/python-interface/checkpoint-wheel-003/`; the compiled wheel itself
is not committed.

The benchmark notebook executes its six measured-cohort choices against frozen
evidence and exports real marimo output. Desktop/mobile browser checks and the
readable, scrollable uncertainty chart have separate source/execution records.
Against the real native wheel, notebook 02's dots, derivatives, precision and
lane-isolation checks pass. Notebook 01's graph-rendering cases retain the same
upstream failure. Historical compile-only evidence is not relabeled as this
installed runtime qualification, and the complete gallery is not yet qualified.

The private selective native-master API has passed default/selective compilation
and 17 focused runtime tests. Complete-array reference validation is a separate
gate. The legacy broad D0 metadata check hit its registered bound; subsequent
checks remain unrun. These checks introduce no new performance measurements,
and historical private-backend timings do not establish a production default.
