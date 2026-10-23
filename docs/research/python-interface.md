# TIRust Python interface and shared native types

The optional `iregi-python` crate links the Rust reduction and preparation APIs
with one Symbolica kernel. It uses the actual upstream Expression, Float,
ComplexFloat, FeynmanDiagram, Model and TensorExpression PyO3 classes. The
standalone `tirust` wheel registers all components in its one native library;
objects from a separately compiled installed Community wheel must not be mixed
with those classes. The bindings remain an rlib for future host integration.

The implementation follows the user's pinned [Community extension guide](https://github.com/symbolica-dev/symbolica-community/blob/cad32c8f589c8a29531414f94b3fbe58020e0308/docs/writing-extensions.md).
Its source requirements were compared with that exact manifest and lockfile.
That host uses different Symbolica/Numerica/Graphica and FeynKit revisions from
this numerical project. This implementation keeps the current project's
9cdc2949-based numeric owners and FeynKit 865afdc43fb4e6b950439d34b4ec2c38352873e7.
No complete Community host/wheel interoperability is claimed without an aligned
source build. PyO3 is pinned to 0.28.3, compatible with both owners' 0.28 APIs.

Numerical workspaces pass whole batches directly into Rust; Taylor components
stay inside each argument. Failed lanes raise LaneError or appear as None in
masked access. Float inputs preserve their represented values and own precision.
DoubleFloat export uses the existing exact high/low rational conversion, then
enough upstream Float storage bits to retain the entire represented pair,
including a low component far below nominal 106-bit accuracy. Reimport uses
Numerica's existing source-precision residual conversion. No arithmetic engine or
symbolic manipulation is implemented in Python or the bindings.

The existing owned numerator preparation helper for tensor component extraction
is exposed as `scalar_component_expression`; its Spenso/Idenso body is unchanged.
Expressions and coefficients cross component boundaries as native Atom values.
No serialized numerator or graph handoff is used.

Before exposing arbitrary Python numerator expressions, preparation was checked
for invalid-variable and exponent panics. The three reuse checks were:

1. Symbolica's public `try_to_polynomial`, `to_polynomial_in_vars_with_field`,
   PolyVariable conversion and fast-path exponent parser were read. The public
   fallible converter shares the same narrow exponent arithmetic; it does not
   by itself repair an unchecked u32-to-i32 cast.
2. Existing Symbolica polynomial tests and users already employ u32 for powers
   such as x^100000. Numerica owns exact rational and Float/DD conversion.
3. FeynKit `integrals/parametric.rs::parameter_terms` and Python energy
   integration use wide Symbolica polynomial extraction and checked coordinate
   conversion. No independent degree parser or polynomial walker is required.

Preparation now validates four distinct plain component symbols, extracts with
Symbolica's u32 exponents and checks narrowing into the public u16 interface.
The narrow vendored Symbolica patch 0004 changes only the fast-path acceptance
bound to the i32 domain actually used by its subsequent conversion. Larger
compact powers remain opaque coefficients and are rejected by the existing
residual-loop-dependence check. Compact monomial tests cover u16 overflow,
i32MAX+1, u32MAX, larger powers and ordinary accepted exponents. This does not
claim a resource bound for expanding an arbitrarily large multinomial sum.

Build, fresh-wheel, typing, actual graph/tensor interoperability and numerical
qualification are tracked separately; source inspection alone is not validation.
