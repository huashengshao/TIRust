# Source-only assessment: batched high-point child dispatch

This is an unimplemented follow-up to the private Cartesian rank-lowering
experiment. It does not broaden that experiment's qualified domain. The first
rank-one candidate completed on the retained N6 fixture, while the old public
path rejected its tensor coefficients. A bounded rank-two probe is still needed
before diagnosing the higher-rank path. In particular, lowering N6 rank r to N5
rank r−1 still leaves pentagon tensor reduction to handle.

Globally enabling `PvCompiler::coordinate_identities` is unsuitable. It changes
pentagon routing independently of the new recurrence, and a primal planar
identity need not hold in transverse Dual directions. The existing certified
planar interface checks every requested Taylor component; its automatic retry
integration must retain those checks.

The smallest composition worth testing is a thin immutable high-point plan and
an owned workspace that batches child tensor requests:

1. Keep the existing Symbolica coordinate identities, four direction vectors,
   rank-zero constant identity, chart predicates, and parent tensor basis. Build
   the high-point recurrence DAG only down to N5 tensor leaves. Each leaf carries
   an ordered active-propagator subset and a requested Cartesian monomial.
2. Coalesce identical structural leaves and promote each N5 family to the maximum
   requested child rank. A single generic N5 plan can serve all ordered five-line
   subsets. Pack the active sample lanes for all subsets into one row-major
   `KinematicsBatch` (row, argument, Taylor component); Symbolica evaluates that
   batch instruction-major. Preserve absolute
   shifts so every child uses the same integration variable and tensor ordering.
3. Evaluate those children with the ordinary N5 workspace plus the independently
   qualified private automatic planar retry proposal, which is not yet promoted
   to production. It must retain ordinary precision guards,
   full-Taylor planar certificates, chart grouping, fallback flags, and child
   failure diagnostics. For N7, flatten the intermediate N6 recurrence into the
   same structural DAG before packing the final N5 requests; avoid independent
   repeated N6 workspaces that recompute overlapping N5 families. Keep child
   results in `IregiNative`/`Components`; apply the existing public convention
   conversion only after the complete parent tensor is assembled. Passing
   rank-dependent MadLoop signs or multinomial weights through a rank-lowering
   boundary without conversion would be incorrect.
4. Treat complete child tensor outputs as inputs to a thin parent Symbolica
   evaluator. Construct every parent moment and all four native Laurent slots
   with Atom operations. Use the existing Dualizer and instruction-major batch
   evaluator for products and sums, including mixed derivative components.
   The existing contraction module demonstrates this implementation pattern;
   no manual Taylor convolution or matrix/polynomial implementation is needed.

Demand must respect complete Taylor components. A coefficient whose primal is
zero may still multiply a child through its derivatives. Initially eliminate
only structurally zero weights, and preserve lazy chart branch contexts. A
failed demanded child must reject its dependent parent lane; unrelated lanes
remain independent. Never convert an unavailable child coefficient to a
successful zero. A maximum-rank child plan currently reports success for its
whole row, so an unused higher child moment may conservatively reject that row.
Do not read partial numerical values out of a failed child result.

Structural caches should key the child domain, rank, algorithm, input policy,
tensor convention, and complete derivative shape. Numerical workspaces remain
owned per scalar type and working precision. Changing requested precision must
propagate to every child and parent guard. Numerical inputs, scale, masses,
certificate outcomes, and receipt objects must not enter a shared structural
cache. Empty batches, repeated/reordered calls, and independent workspaces need
explicit tests.

The initial composition should **not transport parent raw-distance receipts
across the subset-packing boundary**. That changes both input layout and row
identity. Let each child certify its own packed batch; eligible analytic-profile
dispatch inside each ordinary child may reuse call-local receipts under the
existing source-pointer, domain, precision, and lane-map checks. Hybrid retries
and precision rescue retain their public no-receipt paths. The latest
automatic planar integration does not transfer receipts across the
`CertifiedPlanarWorkspace` boundary; it aggregates proof-work counters instead.
Cross-boundary proof reuse would require a separately reviewed subset-mapping
certificate and is an optimization, not a prerequisite.

There is one additional correctness obligation beyond child success: the parent
weighted sum can amplify child errors. The parent must run its own denominator
and cancellation checks, and its error estimate must account for the child
results' accuracy context. Existing guards explicitly provide estimates rather
than certified error bounds. Merely treating each successful child as an exact
master would silently strengthen that assumption. The first implementation
therefore needs an explicit conservative composition policy and a recorded
diagnostic/error envelope before it can report parent success. This is an open
design item, not an invitation to relax a threshold. `BatchResult` currently
exports values, metadata, and status without an achieved-error envelope;
`Float` precision is representation precision. The contraction workspace's
finite-output checks do not supply the missing parent accuracy policy.
This does not require a new public achieved-error API or a rigorous integral
certificate. The composition should preserve the same explicitly estimated
stability model used by ordinary `Workspace`: a private export of retained child
monitor/envelope data, combined with the parent cancellation guard and independent
precision validation, may be sufficient. Certified claims remain separate.

The current source offers a concrete private export: each `PrimalCheck` already
has a value root, an uncancelled master-amplitude root, and a dimensional degree.
Those numerical roots survive in the ordinary workspace output buffer or the
separate Dual primal-monitor buffer. An explicit output index should map them to
child tensor/Laurent slots; labels must not be parsed. Seed the parent's Symbolica
amplitude construction with the child's uncancelled amplitude, not with the
absolute value of its cancelled result.

That amplitude is deliberately primal-only. It is not a bound on compounded
Taylor errors, and differentiating its absolute values is not justified. Preserve
the child's existing component-local checks and the parent's actual Dualizer
contribution checks. Where numerical quadrature errors exist, transport those
per-component estimates separately; the existing quadrature module already uses
Symbolica to perform the positive Taylor-component convolution needed to propagate
such errors. Keep these errors distinct from primal ULP sensitivity.

The exported metadata must follow the **actual successful execution route**.
After analytic, Hybrid, or automatic-planar replacement, the original workspace's
monitor buffers may describe the rejected result rather than the returned child
values. Merge amplitude/error metadata with the successful child result using the
same verified output ordering, shape, precision and original-lane mapping, reset
it on every call, and reject a demanded result if required monitor data is missing.
This route-aware private export is additional implementation work; it is not an
existing public aggregate envelope.

Qualification should start with the bounded rank-two retained fixture and exact
sentinel dispatch tests, then cover full arrays and precision convergence,
planar/ordinary mixtures, transverse and tangent Dual jets, mixed failures,
reordered/chunked batches, and counters establishing batching through child
master evaluation. The current rank-one evidence and earlier preparation
timeouts remain separately recorded.

The reviewed automatic integration is archived in
`validation/automatic-planar/final/production-source.tar.gz`; its adoption patch
hash is `e23ce191d83994415f8f77e195ab31356a7d1155d353dd7f8ea424214b8226c3`.
Its runtime coverage currently includes automatic rank-one/rank-two fallback,
f64/Float128 at requested53, and ordered four/six-component Dual layouts. It does
not yet establish automatic rank-seven coverage. Any combined higher-rank
composition needs its own full-array qualification even though the standalone
certified planar API already has separate high-rank evidence.
