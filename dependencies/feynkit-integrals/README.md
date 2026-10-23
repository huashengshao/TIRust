# FeynKit affine integral families

This package extracts FeynKit's existing affine propagator partial-fraction
engine for use without graph, model, kinematics, Spenso, or Idenso dependencies.
Its only dependencies are Symbolica and thiserror. The embedding workspace
provides the Symbolica version pin and numerical-backend features.

```rust
use feynkit_integrals::AffineFamily;
use symbolica::{atom::AtomCore, parse};

let x = parse!("example::x");
let mass = parse!("example::m2");
let family = AffineFamily::new(vec![x.clone()], vec![x.clone(), &x - mass])?;
let terms = family.partial_fraction(&[2, 1], 10_000)?;
# Ok::<(), feynkit_integrals::IntegralFamilyError>(())
```

Each returned `(coefficient, powers)` retains the input denominator order.
The positive-power denominators of every term are independent. Negative powers
remain numerator factors. A bounded-work error returns no partial result.

Variables must be a nonempty list of distinct bare Symbolica symbols. Use a
separate symbol for each independent affine variable: for a one-loop Cartesian
representation, a symbol for `q²` and four symbols for `q⁰,…,q³` make each
`(q+pᵢ)²−mᵢ²` affine. Here `q²` is an independent placeholder during this purely
algebraic denominator decomposition; it is not expanded into Cartesian squares.
The caller supplies all kinematic substitutions before constructing the family.

Polynomial conversion, exact Gaussian-rational coefficients, matrix rank and
linear solves remain Symbolica operations. Floating literals are rejected; use
exact rational constants or symbolic kinematic parameters. Functions of external
parameters are allowed, while hidden dependence on an affine variable such as
`sin(x)` is rejected. Generic symbolic coefficient denominators must be respected
when later specializing kinematics. No integration, contour selection, scaleless
term removal, or loop-momentum shift is performed.

Run the provenance check without loading Symbolica:

```sh
python3 dependencies/feynkit-integrals/scripts/verify_upstream.py
```

After adding the package to the embedding workspace, compile its tests with
`cargo test -p feynkit-integrals --no-run`. Running those tests uses Symbolica's
normal runtime and license checks. See [UPSTREAM.md](UPSTREAM.md) for exact
source boundaries and [VALIDATION.md](VALIDATION.md) for the recorded check status.
