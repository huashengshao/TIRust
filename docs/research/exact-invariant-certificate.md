# Exact supplied-invariant certificate

The exact certificate prototype passed seven focused tests, preserved with source
snapshots in `validation/invariant-certificate`. Its opt-in reduction integration
is now implemented and undergoing qualification. Ordinary scalar and tensor
constructors retain the six/seven-point supplied-G guard. The separate
`CertifiedScalarReductionPlan` targets rank-zero IBP requests with a regular
rank-four primal Gram chart; it never exposes its unchecked internal chart plans.

## Input meaning

The certificate uses the exact represented value of every input coefficient.
It constructs the relative Gram matrix

`K_ij = G_ij - G_i0 - G_0j + G_00`, for `i,j > 0`.

This is invariant under translations of the common momentum origin. Scalar
integrals admit analytic continuation to complex external vectors, so the
certificate imposes no real-Minkowski inertia restriction. For a Cartesian
tensor interface, a later extension must additionally compare this relative
Gram matrix against the supplied relative coordinate shifts, coefficient by
coefficient. Rank-zero scalar inputs need no coordinate witness.

For each four-element principal chart `A`, Symbolica constructs `det(A)` and
the bordered minors

`E_uv = det([[A, K_Av], [K_uA, K_uv]])`.

If the primal determinant is nonzero, every coefficient of every `E_uv` must
vanish in the caller's Taylor algebra. This is equivalent to the full Schur
complement constraint in that chart, without dividing by its determinant.
There are five charts with one residual each for six propagators, and fifteen
charts with three residuals each for seven. Off-diagonal residuals are essential;
principal minors alone do not establish the seven-point rank constraint.

The prototype prepares rational polynomial instructions and Taylor coefficients
using Symbolica's Dualizer. Its existing batch evaluator requires `Real`, so
`Complex<Rational>` cannot execute through that interface: exact ring execution
currently exists only in the scalar evaluator. Instead, the prototype maps exact
inputs and coefficients to Symbolica's certified `ComplexBall` domain and runs
the polynomial instructions across all active lanes. Increasing precision either
certifies a nonzero residual or produces an exact zero enclosure, whose center
and radius are both zero in both complex components. Only the latter can pass.
A zero-containing ball with positive radius remains unresolved, never accepted.
All represented input coefficients are dyadic; the polynomial operations can
therefore become exact at sufficient precision. A bounded precision budget
leaves an explicit unresolved diagnostic when they do not.

The prototype never accepts a small nonzero residual as zero. Rounded scalar products
may therefore receive an incompatibility diagnostic even when they were intended
to approximate a physical rank-four matrix. Higher arithmetic precision cannot
restore information already lost from the supplied input. No input is projected
or modified.

`f64` is converted through an exact 53-bit Symbolica `Float`; `DoubleFloat` is
the exact rational sum of its two represented limbs; arbitrary `Float` uses
Symbolica's `try_to_rational`. These conversions introduce no binary64 detour
for higher-precision values. Derivative coefficients remain distinct from sample
lanes.

No invertible primal chart produces an explicit `NoRegularChart` diagnostic.
It establishes neither rank nor compatibility: both lower-rank matrices and
the incompatible rank-five matrix `-I_5+J_5/4` have all principal four-by-four
determinants zero. Lower-rank and rank-changing Taylor strata require separate
analytic profiles. The opt-in interface reports an unstable lane in this case.

## Reduction integration

In a certified chart, use Symbolica's matrix solve for
`c = A^-1 K_Aw`. Set `f_i=K_ii-m_i²+m_0²` and
`C=f_w-sum(c_a f_a)`. Then

`D_w - sum(c_a D_a) + (sum(c_a)-1) D_0 = C`.

A nonzero `C` gives an inhomogeneous pinching identity. A zero `C` gives a
homogeneous power-transfer relation. The existing scalar compiler's routing
relation engine already distinguishes these cases and supports generalized
powers and dimensions. Integration must reuse that engine and its termination
rules, rather than implement a second recurrence. A particularly small integration
can supply that existing engine with four linear coordinates `r_i=(K_ai)`, one
for each member of the certified chart, and `r_0=0`. These covector projections
are an invertible linear coordinate system on the external span. Their linear
dependencies are therefore the physical momentum dependencies; they are not
time/spatial components and must never be used to reconstruct squared distances.
The original Cayley matrix still supplies the denominator constants. The
prototype exposes this construction as `certified_relation_coordinates`.
Production lowering uses the equivalent `K_ai=(s_0a+s_0i-s_ai)/2`, retaining the
original supplied distances. The scalar compiler's private
`set_certified_linear_coordinates` setter explicitly records this contract.

`CertifiedScalarWorkspace` first checks exact represented inputs across the
batch, then groups accepted lanes by chart. Each group enters one ordinary core
`reduce_batch` call. Certificate plans are cached by ordered Dual shape and
reduction plans by chart; numeric scratch and adaptive certificate buffers belong
to each workspace. Invalid constraints, unresolved precision and absent regular
charts remain failed lanes. Both recurrence and certificate execution expose row
counters. Neither scalar preparations nor numeric results are cached by sample.

## Reusable zero-polynomial predicates

`ExactZeroConstraintPlan` reuses this same engine for analytic-branch predicates.
It supplies a constant unit chart and the caller's polynomial residuals, so only
the interpretation of the certificate changes. The outcome is `Zero`, a
certified `Nonzero` with constraint/component indices, or `Unresolved`. All
Taylor coefficients must vanish, and a positive-radius enclosure never proves
zero. This interface supports the independent planar-pentagon work; attaching
its result to branch dispatch still requires separate integration qualification.

Preparation uses Symbolica's `try_to_polynomial` over its rational field and
checks the resulting variable map against the unique declared symbols. It rejects
functions, non-polynomial powers and unknown symbols before compiling Symbolica's
reconstructed polynomial. This is necessary because a conditional expression
evaluated on an uncertain ball is not itself a proof of its exact branch.
The check reuses the same polynomial API as E6 quadrature construction;
Symbolica's `src/poly.rs` conversion source and `tests/rational_polynomial.rs`
were checked. FeynKit's `symbolic.rs::linear_terms`,
`IntegralFamily::parameter_terms` in `integrals/parametric.rs`, and its Python
energy-integration numerator extraction likewise validate the variable map after
conversion. Their coefficient domains allow general expressions, whereas this
certificate deliberately requires rational coefficients and declared symbols.

Numeric cancellation in a
valid chart remains the responsibility of stability diagnostics and precision
rescue; exact compatibility alone is not a conditioning certificate.

## Rounded-zero invariant preflight

An exact zero used to choose a scalar-master or analytic-profile branch must
refer to the represented input polynomial, rather than a rounded intermediate.
For example, the exactly represented binary64 vector
`q=(2^27, 2^27-2^-26, 2, 0)` has `q²=-2^-52`; ordinary binary64 squaring and
subtraction can return zero. A massless bubble would then acquire scaleless IR
poles even under input preservation.

The core prepares the original unsnapped squared-distance polynomials. Before
profile dispatch or scalar-master evaluation, it certifies every distance Taylor
component whose numeric evaluation returned zero. Exact nonzero or unresolved
components produce an unstable lane available to precision rescue. This does not
reinterpret an explicitly requested legacy snap as an exact identity: its raw
polynomial is retained separately from its effective snapped invariant.

`check_selected_batch` uses per-lane `(constraint, component)` selections, so a
genuinely zero primal may retain nonzero transverse derivatives. Empty selections
skip arithmetic; selected lanes run together through the same polynomial batch
evaluator, retaining their original masks during refinement. Gram-rank certificates
cannot use this selection API: they always check all minors and Taylor components.
`NativeFloat::exact_rational` supplies the exact represented values; custom numeric
domains without this operation fail explicitly only when such a proof is needed.

Rounded nonzero invariants also need conditioning checks. The represented vector
`q=(2^27, 2^27-1, 2^14, 2)` has `q²=-5`; direct binary64 evaluation yields `-4`.
For a massless bubble this changes the finite coefficient by `log(5/4)`, although
an exact-zero-only preflight cannot see the problem. The original recurrence and
moment monitors did not generally check scalar-master argument conditioning.

The new relative argument monitor uses Symbolica's rational polynomial
coefficient map to construct a polynomial with nonnegative coefficients. It
evaluates that polynomial through the same ordered Dualizer on the norm of every
original input Taylor coefficient. This positive convolution includes cancellation
inside individual monomial jets; taking the norm after evaluating a monomial jet
would miss such cancellation. It reuses the existing operation-count and precision
helpers, comparing the roundoff estimate against `2^(19-accuracy_bits)` times the
magnitude of each nonzero derived coefficient. True zeros defer to the exact
certificate. The monitor is a conservative conditioning estimate, not a certified
error bound, and can reject accurately computed but poorly conditioned inputs.
The relative roundoff model is also inapplicable to gradual underflow. Using
Symbolica's existing `real_classify`, the monitor rejects a nonzero subnormal
derived magnitude or positive envelope, and rejects any zero, subnormal or
nonfinite intermediate in its error-model comparison. For example, the represented
vector `(200000.75,100000.5,100000.5,100000.5)*2^-537` has exact square
`(10^10-3/16)*2^-1074`, whereas binary64 multiplication gives
`(10^10+1)*2^-1074`. Its relative error exceeds `1e-10`, but its estimated absolute
roundoff underflows to zero. Such a zero estimate cannot justify acceptance;
the lane requires precision rescue. The classification follows the numeric domain
and does not impose binary64 exponent limits on arbitrary-precision Float.

The instruction-major evaluator preserves each Float operand's precision;
workspace precision alone does not set arithmetic precision. Accordingly, the
argument monitor uses the minimum nonzero input-component precision, capped by
workspace and positive-envelope precision. It separately checks uncertainty from
reported output precision against the output magnitude. Multiplying that already
reduced output precision by the large input envelope would count cancellation
twice and incorrectly reject successful 128-bit rescue. A prototype256/input80
regression uses `q=(2^53,2^53-1,2^27,2)`, whose exact square is again `-5` while
80-bit products lose the final unit. Higher Taylor regressions retain distinct
80/160/256-bit input precisions. The older recurrence and primal sensitivity
monitors also use workspace precision; their mixed-precision assumptions require
a separate audit and are not qualified by this argument-monitor fix.
Its integration and regressions are pending qualification; earlier certificate
test results alone do not validate this additional path.

## Existing facilities checked

All checks use the project's pinned Symbolica/Numerica revision
`9cdc2949cdb153478c1eca0d40690dc0abfc616e`, rather than the newer convenience
checkout in `references/symbolica`.

1. Symbolica/Numerica APIs and source: `Matrix::det`, `inv`, `solve`, `solve_any`,
   `rank`, and `row_reduce` in `lib/numerica/src/tensors/matrix.rs`; exact rational
   preparation and certified-ball domains in `src/evaluate/domain.rs`; the
   `Real` bound in `src/evaluate/batch.rs` and exact scalar `evaluate_in_ring`
   in `src/evaluate/evaluator.rs`; directed interval
   arithmetic in `lib/numerica/src/domains/float/interval.rs`; exact float
   conversion in the multiprecision float implementation.
2. Existing usage/tests: Numerica's matrix determinant and solve tests; this
   project's PV denominator charts, E6 geometry preparation, and precision-rescue
   exact DoubleFloat promotion. Symbolica already owns both determinant
   construction and Taylor expansion.
3. FeynKit equivalents: `IntegralFamily::affine_system`, its partial-fraction
   dependence search, and orbit-projector matrix solves use the same Symbolica
   matrix facilities. No higher-level full-Taylor supplied-Gram compatibility
   certificate was found in these implementations.

The prototype adds chart enumeration, input conversion and diagnostics only.
It adds no determinant, elimination, polynomial, or Dual arithmetic algorithm.
