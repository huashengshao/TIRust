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

The numerical bindings now delegate reduction selection and contraction
preparation to the shared Rust `TensorReductionPlan`/`TensorWorkspace` facade.
No Python-side topology list selects a backend. Workspace construction passes
profiling before engine selection, and every explicit Dual layout, including a
primal-only shape, uses the facade's Dual constructor. Eligible positive-power
high-point inputs retain the affine-power engine for profiling and Duals;
low-point inputs retain generic execution. The compiled
default scalar-master policy is retained. Python's existing API does not expose
precision rescue or a nondefault master-policy selector; these remain Rust
facade capabilities rather than silently added Python options.

`workspace.engine` reports the actual owned workspace path, and its
`preparation_notes` include capability selection notes. Each result owns a
tagged `engine_execution` snapshot copied from the public Rust counters.
Generic execution and its certificate work remain distinct. Six-/seven-point
execution retains packed child rows, parent rows, child/parent work and their
certificate counters. Existing `stats`/`execution_stats` use the facade's checked
summary, which counts logical parent input rows once; no packed or certificate
counter is invented or added twice. Numeric conversion, upstream shared types,
Dual transport and lane failure handling remain unchanged.

`tests/python/test_tirust_dispatch.py` checks the generic profiler, explicit
primal-only and mixed Dual layouts, high-point capability selection, real
packed batching, sparse contraction, invalid-lane isolation, empty/reset state,
immutable result diagnostics and recovery. Its high-point samples reuse the
retained six-point and seventh-time-quarter fixtures. These are installed-wheel
interface checks, not new high-point reference or performance qualification.
The current fresh standalone development wheel passes 65 installed API cases
on Linux with CPython 3.13: 29 numerical/interoperability cases, eleven facade
cases, six upstream ComplexFloat native-operand precision regressions and 19
upstream Float/ComplexFloat cases. Seven actual marimo runtime cases also pass.
Installed consumer typing reports zero errors and warnings, including the new
checked-input option and tagged affine execution types. The
[saved qualification](../../validation/python-interface/facade-wheel-004/README.md)
retains 201 selected source files, including all 158 coherent build bodies,
exact wheel/native hashes and every stage receipt. All seventeen current owned
process groups completed with empty cleanup errors. Source/runtime/compiler
closure limits are explicit. The new Legacy no-op scalar test verifies
rejected-only input never prepares a child, two accepted lanes execute together,
caller containers remain unchanged, and empty/rejected reuse and recovery keep
owned result snapshots. Requested enum forwarding is checked separately in the
core; Python has no direct requested-algorithm/policy getters.

The current dotted bubble identity exercises the MassTaylor route without
changing numerical assertions. The installed consumer checks its component
counters and the recursive optional Legacy child union using ordinary
assignability, strict tag/leaf assertions and recursive calls, without casts or
`Any`. Two earlier exact-type identity observers printed identical recursive
unions yet failed basedpyright's identity check; their source/diagnostics remain
preserved. The saved replay's initial assumption that `cfg(test)` immediately
preceded a module declaration was corrected to include the actual path
attribute. That record is a saved-observer correction, not a native test
failure.

The [historical wheel003](../../validation/python-interface/facade-wheel-003/README.md)
retains its aggregate dispatch deadline, external toolchain removal and offline
marimo cache miss unchanged. Current cold-constructor cases use separate
budgets; no successful case was repeated to repair an aggregate timeout. The
[earlier 60-case qualification](../../validation/python-interface/facade-wheel-002/README.md)
retains its historical trait-surface observer failures and narrow exclusions for
Rust-only `real_classify` and mutating `set_precision`.

An actual installed-wheel reproduction exposed precision loss when native
integer operands were converted at the smaller of a complex value's component
precisions. The upstream Numerica correction converts native operands at the
larger component precision in the two existing operand helpers. Existing
tracked Float/ComplexFloat operands and the minimum-precision property are
unchanged. The corrected wheel preserves all 128 imaginary bits in the original
asymmetric reproduction and passes all six dedicated regression cases. The
earlier wheel and its failing reproduction remain independently archived.

This is frontend qualification on the specified fixtures, rather than a new
full-array integral oracle or performance result. The current objective is
generic rank-six reduction, including positive propagator powers and dimension
shifts, with propagator-count reduction treated separately. Six-point coverage
is a concrete validation checkpoint, not proof of arbitrary-point support.
Historical rank-seven results remain separately reported challenge cases.
The existing seven-point scalar frontend regression was retained, without
adding positive-rank seven-point work.

The source API adds an explicit supplied-product check through
`ReductionPlan(6, rank, invariants="scalar_products", check_scalar_products=True)`.
It delegates to the Rust checked Cartesian facade and retains the supplied
products. Cartesian compatibility and a regular rank-four Gram certificate are
separate checks across all Taylor components. Profiling and explicit Dual
layouts retain these checks. The source tests verify rejection before any
master evaluation, tagged certificate counters, and empty-state reset. The
current wheel passes these constructor, rejection and empty-state checks for
ordinary, profiling and primal-only Dual workspaces. Ordinary inputs keep their
existing constructor guards. Checked six-point rank-one numerical evidence is
separate from these frontend tests; the original inline rank-six attempt
exceeded its preparation resource deadline. Neither generalized nor rank-six
checked numerical coverage is claimed from the constructor accepting those
settings. Checked precision rescue currently returns an explicit unsupported
error.
