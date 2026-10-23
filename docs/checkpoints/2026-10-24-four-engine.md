# Four-engine and high-point validation checkpoint

The implementation goal remains active. This checkpoint records completed
validation and performance evidence; it does not select a new default backend
or establish the full seven-propagator/rank-seven acceptance envelope.

## Identical-input performance

`validation/selected-native-four-engine/v3-frozen/` preserves the one timing
cohort: eight N2/N3 rank0/1 families, singleton and batch32, 48 balanced quartets
per family. Every before/after numerical check and final control passed.
One COLLIER interval in the scalar-bubble singleton family was shorter than
the registered 100 ms minimum. That family's five ratios remain inconclusive.
No timing was repeated or extended.

The original analyzer conservatively disqualified all remaining families
because its aggregate status check also included that local short interval.
Its complete original result is retained unchanged. The separately reviewed
`validation/selected-native-four-engine/posthoc-revision2/` derives clean
termination and family completeness from the raw responses before applying
the same numerical checks, confidence intervals and width gates. This is an
explicit posthoc admission correction. All forty measured sample arrays and
statistics remain exactly equal to the frozen analysis.

Among the seven fully measured families, the selective native-master TIRust
candidate improves on its TIRust baseline in all seven. It improves on original
IREGI in one and regresses in six; it regresses against COLLIER in all seven.
These private historical-source measurements support further implementation
work, not a default-backend or performance-parity claim. Every displayed ratio
has TIRust in the numerator. Separate historical sessions are not combined.

## Bounded high-point and reference evidence

`validation/highpoint-thin-n6-mask/` contains the restricted private rank-two
six-point qualification. All 60 complex tensor/Laurent entries agree with the
original IBP reference at both working precisions; 128/256-bit convergence,
fourteen exact tests and the retained lifecycle checks pass. The earlier
unmasked failures remain separate evidence. General kinematics, generalized
requests and production adoption are outside this qualification.

The separate `validation/highpoint-thin-n6-generic/` package extends the private
candidate to rank three on the retained base fixture: all 140 complex slots
agree with untouched original IBP within the required normalized tolerance,
and 128/256-bit outputs converge within `2^-80`. Twenty-five exact tests pass,
including dynamic ranks and UV cancellation. This is restricted rank-three
numerical evidence, not numerical rank-seven or general-domain qualification.

`validation/collier-reference/shared/` retains the extended reference matrix,
native rejection flags, deadlines and IREGI mode disagreements. Attempt005
captures exactly the eight requests left unfinished by attempt004, without
repeating its 239 completed requests. The checked COLLIER initialization-capacity
repair changes the reference driver rather than upstream library arithmetic.
Independent Symbolica identity preparation and its subsequent numerical checks
have separate source and execution records; they cannot override native failures.

## Python status

`validation/python-interface/checkpoint-compile/` qualifies compilation against
the actual shared PyO3/Symbolica/FeynKit types and consumer source typing.
It explicitly does not qualify a wheel install, numerical runtime, executed
notebooks or integration into a separately built Community host. The first
native wheel and its runtime qualification proceed separately. The Rust core
retains its Python-free default workspace build.

The maintained Typst report includes these scopes and the failures as well as
the successful checks. Full ordinary and generalized high-point coverage,
precision/Dual qualification and per-family performance parity remain open.
