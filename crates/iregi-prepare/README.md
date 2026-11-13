`iregi-prepare` converts FeynKit DOT graphs into reduction inputs. It has no
dependency on the tensor reducer or scalar-master implementation.

```rust,no_run
use iregi_prepare::{prepare_dot, PrepareOptions};

let model = std::fs::read_to_string("fixtures/models/phi3.json")?;
let dot = std::fs::read_to_string("examples/graphs/bubble.dot")?;
let graph = prepare_dot(&model, &dot, &PrepareOptions::default())?;
for term in &graph.numerator.terms {
    println!("{:?}: {}", term.exponents, term.coefficient);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `NumeratorSource::FeynmanRules` to derive the numerator from the model's
vertex and propagator rules. The model input is a pinned FeynKit JSON snapshot;
the production path does not import Python. `external_wavefunctions` includes
symbolic external spin wavefunctions. `component` selects any remaining open
color or Lorentz tensor axes in their declared logical order.

Model loading preserves JSON numeric tokens. A narrow compatibility adapter
converts integral decimal quantum numbers in `charge`, `y_charge` and
`y_charge_right` to the exact strings accepted by FeynKit. It uses Symbolica
numeric types, retains FeynKit's floating-point integer bound, and rejects
fractional numeric charges; use exact strings such as `"2/3"` for fractions.
Other model fields and numeric tokens pass through unchanged.

FeynKit parses and routes the graph. Idenso and the exact Spenso HEP library
contract finite components, including gamma matrices, the Minkowski metric and
SU(3) generators. Symbolica expands the result and extracts the coefficients
of `iregi::q0` through `iregi::q3`; statistical zero tests are disabled. A term's
exponent tuple carries ordinary polynomial powers without factorial weights.
Coefficients remain Symbolica atoms for later evaluation at the chosen numeric
type and precision.

Each `PropagatorRouting` describes `s*q + sum(c_i*p_i)`. `external_edges` maps
the `c_i` positions to external edge IDs; `external_legs` records their names,
ordering, incoming state and PDG codes. Before passing denominators to the
reducer, turn `s=-1` into the canonical `q + s*sum(c_i*p_i)` routing. The
numerator remains a polynomial in the selected `q`.

`prepare_dot` and `prepare_diagram` accept one-loop graphs whose internal
propagators all depend on the loop momentum. They report open tensor
components, nonpolynomial loop dependence and internal tree propagators as
explicit preparation errors. `prepare_dot_with_tree_propagators` and
`prepare_diagram_with_tree_propagators` accept loop-independent internal
propagators as well (an s-channel propagator, the tree leg of a vertex
correction), and list them in `tree_propagators`. Their Feynman-rule numerators
are part of the numerator, and the caller multiplies by `1/(p^2 - m^2)` for
each.

`prepare_scalar_dot_with_tree_propagators` stops one step earlier. It
returns the contracted, routed scalar numerator, unexpanded
(`PreparedScalar::scalar`), with the same routings; `into_graph` finishes the
preparation. It also accepts tree graphs: their scalar has no loop dependence,
and every internal propagator is a tree propagator. A Born amplitude then
comes from the same Feynman rules and projector as the loop graphs. A caller that evaluates many numeric configurations can
substitute its numbers into the scalar first and then call
`polynomial_from_expression`. A fully symbolic expansion can be far larger:
an S-wave-projected g g > c c~ g pentagon runs to millions of terms.
`polynomial_from_expression` converts sums of more than 4096 terms term by
term and adds the polynomials pairwise. Symbolica's direct conversion
accumulates each coefficient one term at a time, which is quadratic. The
result is the same.
