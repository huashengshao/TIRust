# Certified near-vacuum D0 integration proposal

Status: selected validation cases qualified; production integration is not yet
implemented. The immutable evidence is
`validation/near-vacuum-box/qualification.json`, SHA256
`4acbf48c5b443882a0f6dca6563e8e7e3cb9dd5e2c302fca0294d6a7769a88f4`.
The independent audit checked all 18 retained runs: 261 accepted primary
components, 35 separate overlap components and 11 ordinary-DNS primal checks.
Those counts include three final-source repetitions. Requested accuracy is
40 bits for f64, 80 for DoubleFloat, 100 for Float128 and 128 for Float256.
The DNS overlap is only qualified at 24 bits; it is not an accuracy default.

## First patch: explicit D0 accuracy and finite certificates

Move the qualified generic series plan and workspace into an isolated proposed
`iregi-masters::near_vacuum` module, retaining the existing Symbolica series,
exact polynomial integration, Dualizer and Numerica ball programs. Keep the
independent oracles in validation tooling. Begin with orders two and four,
the existing bounded shapes, and the existing resource caps. Unsupported
shapes must be diagnosed rather than silently dropping components.

Keep `MasterEvaluator::evaluate_batch` compatible. Add an explicitly
accuracy-aware entry point taking a per-call context; proposed names are
`evaluate_batch_with_accuracy` and `MasterAccuracyContext`. The context must
carry requested bits and an absolute budget for every ordered Taylor
component, together with the relative budget. These are exact rationals or
exact binary-scaled values, not intermediate f64 constants. A separate policy
states whether an uncertified ordinary result is permitted. The first E6
experiment requires certificates for every selected D0 contribution.

Extend `MasterBatch` with optional per-lane finite certificates and a formula
diagnostic. Ordinary DNS output has no certificate; absence must not be
represented by a zero error. A certificate contains the enclosing Cartesian
rectangle for each finite component and the worst squared distance from the
returned value to that rectangle. Keep the zero UV/IR coefficients explicit
for the proved finite positive-mass domain. Rejecting a certificate hides the
lane's numerical values exactly as other failures do. An uncertified
numerical result may only be returned under an explicit ordinary-result
policy, with its uncertified status retained.

Dispatch after existing physical input validation and before the constant
kinematics/DNS paths. Compact eligible rows, evaluate each series, enclosure
and tail instruction across those rows, then scatter successful rows. Try
N2, then N4 only for lanes whose full component budgets were not met. Group
the remaining ordinary rows into the existing batch. Certified series rows
do not execute the DNS derivative-discriminant test: their analytic
neighborhood has been proved by a different formula. The existing
discriminant and physical guards remain unchanged for ordinary DNS rows.

Preserve the original masses and invariants. The center M is auxiliary and
frozen, with zero derivatives; all physical mass and invariant jets remain
in the expansion. Keep the existing real-invariant, positive-real-mass and
lower-half-plane conditions, the rho bound, native Taylor ordering and
componentwise acceptance test. D0's finite coefficient in this domain has
zero scale derivatives; validate the original renormalization-scale input
and retain the established Laurent scale convention.

Do not prepare the series or convert ordinary-path inputs to exact rationals
unless a candidate lane is present. Cache structural programs separately
from precision-specific evaluator workspaces. Cache keys include formula
version, order, complete derivative shape, precision and backend as
applicable. Numerical results are not cached across contexts.

## Second patch: E6 propagates certified box errors

`Pentagon6Evaluator` already batches all active sample lanes, quadrature
nodes and five pinches into one D0 call. Pass its actual accuracy context to
that call. No inherited 24-bit setting is acceptable. Use caller budgets,
or stricter explicitly recorded inner budgets, and check the propagated
final error before success. A fixed guard-bit allowance alone cannot prove
the final requested accuracy when contraction coefficients amplify errors.

Add two distinct diagnostics: a rigorous finite-master/arithmetic enclosure
for the selected quadrature rule, and the existing successive-rule
quadrature difference estimate. Do not relabel the latter as a rigorous
integral enclosure. The final check must reserve budget for both, and state
which part is an estimate. Certified D0 values alone do not certify E6.
Retain enclosures for both successive rules: their difference is bounded by
the center difference plus both arithmetic radii, while the returned-rule
arithmetic radius is only the current radius. Keep these distinct quantities
when applying the existing consecutive-convergence criterion.

Evaluate the existing Symbolica geometry and contraction programs in
ComplexBall, including the contour shift, coefficients, node weights and
summation. Inject full D0 rectangles, including omitted-series tails. There
are two additional enclosure obligations before this can claim a rigorous
rule enclosure:

1. A D0 certificate currently encloses the integral at its exact represented
   input. E6 forms shifted masses using finite-precision additions. Treating
   those rounded masses as the original inputs loses this upstream error.
   Pass enclosing input balls into the series/majorant programs, or evaluate
   the fused Symbolica shift and D0 program on the original inputs. The
   majorant must cover the entire shifted-input rectangle while keeping M
   fixed. Never repair this by snapping the tiny invariants to zero.
2. `gauss_legendre` isolates exact Legendre roots, then discards each interval
   in favor of its midpoint. Retain the root intervals and evaluate the
   existing Symbolica weight formula on them, so node and weight errors are
   included. Evaluating midpoint weights exactly only certifies a different
   approximate rule. Contour-scale and coordinate conversions also need
   enclosures. The quadrature truncation beyond that exact rule remains a
   separate issue.

Reject interval divisions whose denominator cannot be proved nonzero. Keep
the numerical primal path for returned values, but accept only when its
distance to the enclosing rule rectangle satisfies the requested component
budget. Singular/small-Gram geometry may still require higher working
precision or a different existing Symbolica formula; the D0 branch must not
hide that remaining problem.

## Third patch: propagate bounds to requested reduction outputs

`Workspace::set_accuracy_precision` currently updates E6 tolerances and
existing child workspaces. Extend this propagation to ordinary master
contexts, E6's nested D0 contexts, clones and lazily prepared fallback
workspaces. Requested and working precision remain distinct.

`QuadratureStabilityPlan` already uses Symbolica affine projection and
Dualizer to propagate E6 component estimates through tensor coefficients.
Reuse that structure for certified D0 finite leaves as well as E6 rule
enclosures, but evaluate the relevant coefficients and Taylor products with
ball arithmetic. Coefficient rounding, reduction-input preparation and final
summation are part of the error; exact error radii multiplied by uncertified
floating-point coefficients are insufficient. Include the affine
master-independent term and the product of coefficient and master errors;
full interval evaluation of each coefficient-times-master expression includes
that cross term. Branch predicates must be fixed and certified from the
represented primal inputs; an ambiguous interval predicate cannot silently
select an affine branch. Keep rigorous enclosure
contributions separate from quadrature truncation estimates and the existing
roundoff diagnostics. Include every requested tensor moment and numerator
contraction, in its existing dimensional normalization. Ordinary uncertified
masters cannot silently acquire a certificate through this projection.

## Bounded qualification sequence

1. Freeze the isolated adapter sources and dependencies. Run the current
   independently qualified D0 fixtures through its public batch interface at
   their existing budgets, including interleaved ordinary/series/rejected
   rows, scale-only jets, row permutations and separate workspaces. Require
   exact correspondence between standalone candidate and adapter certificates.
   Preserve all expected N2 and impossible-accuracy rejections.
2. For one fixed E6 node on the exact tiny-routing fixture, compare the new
   shifted-input certificate with an independent direct parameter integral
   at the original exact shifted inputs. Check the contour geometry and
   every selected derivative component separately. This isolates the source
   of error before another full E6 execution. Initial bound: 60 seconds.
3. Evaluate one complete E6 quadrature rule with the ball propagation above.
   Record per-node rho, selected series order, box error, coefficient
   amplification and resulting rule rectangle. Compare against the retained
   exact E6 series oracle; an uncertain quadrature truncation must remain
   visible. Initial bound: 120 seconds at working256/requested128.
4. Only after that check, run the public E6 API at working128/requested128
   and working256/requested128, retaining both historical failures. Requested
   256-bit and higher/mixed jets need independently sufficient series order,
   reference radius and finite resource bounds; N4 is not assumed sufficient.
5. Add an independent parameter-quadrature check and broader finite-difference
   derivative checks before production promotion, then benchmark preparation,
   master kernels and end-to-end reduction separately. Existing diagnostic
   durations do not establish performance parity.

The first concrete implementation should be prepared in an isolated target
copy, with the committed production files unchanged until these checks and
review are complete. The user-authorized implementation goal remains broader
than this selected validation milestone.
