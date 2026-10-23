# Algebra reuse in scalar-master neighborhoods

The dependency pins are recorded in the workspace and in
`dependencies/oneloopmaster/PATCHES.md`. Every expression operation below uses
Symbolica. Numerical lane validation compares primal values; it does not
implement symbolic manipulation.

## Massive bubble pseudothreshold

The positive-mass pseudothreshold has vanishing Kallen discriminant but a regular
Feynman-parameter integral. The opposite discriminant-zero branch is a genuine
two-particle threshold. A separate Taylor chart is required to differentiate the
former without differentiating `sqrt(0)`.

Reuse was checked in three places before adding the chart:

1. Symbolica's `AtomCore::series`, `derivative`, and `replace_multiple` provide
   exact series and substitution. Its `transcendental::atanh` registration has
   an exact derivative; `poly/series.rs` supports rational and fractional series
   powers. The implementation expands `z*atanh(z)` with this existing engine,
   then substitutes `z=sqrt(lambda)/k`. Symbolica normalizes the even powers.
   A direct Puiseux expansion of the composed expression was tested and left
   indeterminate zero-limit terms, so it is not used.
2. Symbolica's transcendental derivative tests include `atanh`, and its series
   tests exercise fractional powers. Existing OneLOop `two_point.rs` already
   supplies the scalar coincident-root `dB0` limit, but supplies no transverse
   higher-order B0/DB0 chart. The current scalar branch and that threshold
   expression is retained at the genuine threshold. Lanes with constant
   kinematics are evaluated together by the existing ordinary master batch;
   their scale dependence is transported separately as described below. No
   transverse continuation is claimed at a genuine threshold.
3. FeynKit's installed FeynCalc comparison tests use the Symbolica `atanh` API;
   gammaloop's float wrappers forward it to the underlying numeric domain.
   Neither source provides this scalar bubble Taylor chart. Their facilities
   are reused through Symbolica rather than introducing an algebra engine.

The chart uses dimensionless ratios `s/m0²` and `m1²/m0²` before forming squared
quantities, including its discriminant guard. This avoids false zero
discriminants from a tiny common dimensional scale. OneLOop's registered
`__olo_sign_nonnegative` helper selects the real positive-mass chart. The
numerical adapter checks the same real-positive domain and keeps genuine
threshold derivatives explicitly unsupported.

The exact logarithmic formula is expanded only at a zero primal discriminant;
it is not a small-nonzero-discriminant numerical approximation. B0 uses the
prepared Dual depth, and DB0 differentiates a series one order deeper using
Symbolica. Physical finite differences, tiny/large scale tests, and constant
versus seeded true-threshold tests are used for qualification; recorded results
and pending limitations are kept in `tests/validation_evidence.json`.

## Frozen numerical binary normalization

Three reuse checks precede the narrow scaling helper:

1. Symbolica's generic `FloatLike`/`RealLike` interfaces provide logarithms,
   integer rounding, unsigned powers, and division, but no common binary
   exponent/rescaling operation across f64, DoubleFloat, and Float. Float's
   internal backend has exponent access; using it alone would break the
   generic API. The helper uses the existing generic numeric operations.
2. OneLOop's `native/primitives.rs` already rescales complex reciprocals,
   logarithms, and square roots. Those operations are reused unchanged, but
   they do not normalize a complete input Taylor jet before Dualizer's
   intermediate reciprocal powers.
3. FeynKit/gammaloop's numeric wrappers and wide-exponent tests use existing
   power operations for rescaling; they do not expose a shared scalar-master
   batch/jet normalization API.

The helper uses a logarithm only to choose an integer exponent. All actual
factors are exact powers of two made with Symbolica's numeric `pow` operation;
no f64 conversion, backend bit manipulation, or new algebra engine is used.
The same exponent scales every kinematic component, at that component's own precision.
Small factors are made by raising one-half to a positive power, avoiding an
overflowing positive power before inversion. Rescaling divides directly by
the factor and never forms a possibly nonrepresentable inverse factor. A
subnormal DoubleFloat regression showed that one large division can overflow
an internal quotient intermediate. Scaling is therefore split into exact
binary factors with exponents bounded by 256; the same Symbolica numeric
division handles each step. Tests retain the original failure and validate
the corrected route on subnormal factors, wide-exponent Float values, and
mixed component precisions.

`FloatLike::real_classify` identifies subnormals before exponent selection. An
exact pre-shift avoids a nonterminating DoubleFloat logarithm at those values.
The selected exponent is bounded by reversible numeric trials on every real
and imaginary kinematic component. Binary search chooses the largest safe
shift, so tiny seeds cannot silently lose represented bits. A separately
normalized scale is restored by the exact Laurent transport below, including
an integer multiple of `log(2)` for the difference of frozen exponents. This
keeps both scale and mass arguments moderate when their original ratio is
unrepresentable. DB0 is scale independent and retains its inverse kinematic
scale factor. The positive-mass chart also uses separate mass and scale logs.

## Constant kinematics and scale-only jets

The existing scalar expression defines kinematic boundary values, while
`(mu_squared / primal_mu_squared)^epsilon` defines their scale dependence.
Three reuse checks determine the implementation:

1. OneLOop's scalar A0/B0 logarithms and massless C0/D0 Laurent coefficients
   use this renormalization-scale convention. Its scalar batch evaluator is
   reused for the boundary value.
2. Symbolica's existing `series` and `coefficient` APIs expand the scale factor
   times the scalar Laurent series in a formal logarithm. Exact substitution
   inserts the additive logarithms and integer binary offset afterwards. A
   direct composed expansion was tested and produced nonfinite intermediates
   for a large offset; its diagnostic is retained. The implementation asks
   Symbolica for powers zero, minus one, and minus two; it does not implement
   Laurent multiplication or Taylor differentiation.
3. The same `OneLoopExpressions::dual_evaluator` and Symbolica batch evaluator
   used by the scalar adapter compile and evaluate the scale-transport program
   for the validated shape. FeynKit/gammaloop provide no additional scalar
   Laurent transport API requiring separate reuse.

All constant-kinematics lanes remain batched. UV residues are retained and the
IR single pole is reconstructed by Symbolica from total single pole minus UV.
The exact scale program has no kinematic square roots and therefore does not
differentiate a genuine threshold merely because the scale carries a jet.

## Dual helper inlining

Symbolica's public `FunctionRegistrationOptions::inlining` is used before
Dualizer. Its `Auto` currently has the same behavior as `Always`, so bounded
leaf bodies and pure forwarding aliases are selected explicitly. The pinned
Dual evaluator retains every outlined helper argument for each output
component; no public post-compilation signature-compaction API was found.
FeynKit/gammaloop already requests `Always` for Dual numerator functions in
`gammalooprs/src/integrands/process/evaluators.rs`. No custom optimizer or
instruction manipulation is implemented here.
