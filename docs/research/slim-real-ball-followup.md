# Follow-up after the longer slim RealBall experiment

Status: source-only analysis and a possible next design. No new candidate, build, timing, or production change is implemented here. The completed four-case run is frozen in `validation/real-ball-certificate/long-timing`; the independent read-only audit is `validation/real-ball-certificate/long-timing-review.json`.

The long protocol qualifies the real-mass singleton improvement, leaves the real-mass batch 32 case inconclusive, and records small regressions for both complex-mass controls. These outcomes prevent adoption under the familywise performance rule. The earlier shorter run is retained separately. Neither the favorable real medians nor averaging across cases can hide the regressions. The confidence intervals describe each run and do not eliminate between-run effects from uncontrolled host activity.

## What the source establishes

The frozen `execution.rs` prepares the raw-distance certificate using the complete `plan.parameters`. That ordered list includes momentum shifts or supplied scalar products, squared masses, and the renormalization scale. `DistanceZeroWorkspace::failures` converts and packs complete rows for the certificate. Raw-distance polynomials depend on the kinematic inputs, while mass and scale validation/precision have their own checks.

The slim certificate checks every argument and every Taylor imaginary component in an active row. Therefore the complex-mass ggH controls select ComplexBall even though their raw-distance polynomials depend only on real shifts. This is a conservative domain classification. It establishes an opportunity to avoid unnecessary complex arithmetic, not the cause of the measured approximately 0.6–0.7% regression.

On such all-complex classified calls, the source also creates a per-row classification vector, scans inputs up to the first complex component, scans active lanes to collect real slots, then calls the original complex evaluator through a helper. An empty real-slot collection need not allocate a heap buffer. Allocation, scanning, helper inlining and code layout are possible contributors, but the current measurements do not isolate them. No causal overhead percentage is assigned to any operation.

## Narrow structural projection proposal

A possible next experiment would prepare the distance certificate with only its ordered kinematic parameter range and pack only that range. The existing `InputLayout` and `precision_arguments` identify the shift or supplied-scalar-product range. Symbolica's existing polynomial validation would require that each raw-distance expression is a rational polynomial in precisely those supplied symbols; any unexpected dependency would reject preparation. This uses existing Symbolica algebra and explicit layout bookkeeping, not a new polynomial manipulation implementation.

The proof must be structural and independent of current floating values. The selected parameter map and full ordered Taylor shape become part of the immutable prepared certificate. Every Taylor component of every retained parameter remains subject to real/complex classification. Any permitted complex kinematic or derivative components continue to use ComplexBall; public input-domain restrictions stay unchanged. The generic invariant/certificate API must keep its existing full-input semantics unless an equally explicit projection is proved for its own expressions.

Input validation, mass/scale precision checks, derived nonzero conditioning, selected-zero masks, budgets, lane failure isolation and output inspection remain intact. A coordinate-only certificate does not authorize ignoring invalid masses or scales, changing branch choices, or accepting a zero-containing enclosure as exact zero. The existing reviewed polynomial-only/real-coefficient proof still guards RealBall mapping.

This proposal would need comparisons against the original certificate for true complex momentum/Taylor controls, mixed samples, unused complex masses/scales, supplied scalar products, exact-zero and tiny-nonzero cases, precision refinement, reorder/empty batches, cloned workspaces and rejected inputs. Performance would need every original case plus actual complex-dependent certificate controls; moving current complex-mass cases onto a real proof path does not qualify the remaining complex path.

## Interaction with per-call proof receipts

Receipt reuse is a separate optimization. Its frozen experiment cannot be combined numerically with these ratios: different baselines and timings require a new matched combined qualification.

Keep the existing complete schema/source binding: ordered parameters and exact polynomials/pairs, `InputLayout`, invariant-input mode, full ordered Dual shape, Preserve policy, working and requested precision, immutable source-batch identity, and original-to-compacted lane mapping. Add the exact ordered certificate projection/packing map, or retain a canonical schema after an explicit Symbolica equivalence proof. The initial implementation should require identical projected schemas for reuse.

A shared pair index or equal floating value is insufficient evidence. Retaining complete source-batch identity, including unused masses and scale, is conservative and avoids introducing a new numerical cache. A receipt may remove a repeated exact-zero proof only for the matching selected masks and context. The child still evaluates its own arguments and independently applies nonzero conditioning and all input checks. No reuse across rescue precisions, derivative shapes, policies or unmatched contexts follows from this proposal.

Before any implementation, agree the narrow schema and source scope, retain the existing evidence unchanged, and coordinate the scoped candidate experiment with the project owner. Any later timing requires its own coordinated quiet window.
