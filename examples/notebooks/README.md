# TIRust gallery

Three short, reactive notebooks walk from a symbolic graph to its numerical
integral, then explore dotted propagators and the measured performance evidence.

| Notebook | What to explore |
| --- | --- |
| `01_numerator_to_integral.py` | Edit a bubble or triangle DOT graph, inspect its Symbolica numerator, and reduce a batch with separated UV/IR/finite coefficients. |
| `02_dots_and_derivatives.py` | Raise a propagator power, compare a Dual mass derivative and a finite difference, change precision, and inspect an invalid neighbor lane. |
| `03_benchmark_explorer.py` | Select a measured cohort, topology and batch size; inspect paired TIRust/reference ratios, uncertainty and qualification. |

The first two notebooks use the real `tirust` PyO3 extension. The standalone
extension supplies its own shared Symbolica, graph and tensor classes in a
single native host. Use its supported installation environment; do not combine
objects from a separately installed Symbolica/Community native wheel.

Once the TIRust wheel and marimo 0.24.0 are installed in the same private Python
environment, open a notebook from this source checkout:

```bash
marimo edit examples/notebooks/01_numerator_to_integral.py
```

Run the app without the editor with `marimo run`, or export a static snapshot:

```bash
marimo export html examples/notebooks/01_numerator_to_integral.py -o target/gallery/first-reduction.html
```

The benchmark explorer only reads committed data and needs no native extension.
It verifies the SHA-256 of its frozen snapshots before displaying them. All
TIRust ratios put **TIRust in the numerator**; below one means TIRust is faster.
The direct four-engine choices use one frozen session and are explicitly
labelled as historical private prototypes with a posthoc analysis correction.
Every family remains visible, including the short-block bubble singleton.
The original conservative verdict, source-bound correction and unchanged
statistics remain in the evidence. These are separate from the selectable
`6b24c03` production-kernel snapshot and archival IREGI/COLLIER absolute times;
no ratio is reconstructed from unrelated sessions. Selecting the native-master
prototype does not imply production adoption or overall performance parity.

Graph import, polynomial preparation, contraction and reduction call the Rust
libraries. `_presentation.py` only formats output and reads benchmark evidence.
It does not implement symbolic algebra or numerical reduction.

The visual sequence is inspired by the
[HEPKit four-loop numerator notebook](https://hepkit.org/gallery/notebooks/four_loop_numerator.py):
direct displays of real graph and symbolic objects, concise mathematical
explanations and focused reactive controls. TIRust adds a restrained navy/serif
title, readable Laurent cards and accessible uncertainty charts.

The inspected local Linux wheel004 passes all 29 focused API tests, including
the real graph renderer. With that exact wheel, all four native gallery tests
pass: bubble binary64 and DoubleFloat, triangle Float, contraction, the dotted
bubble/Dual mass identity, precision distinctions and failed-lane isolation.
All three notebooks export successfully. Their static HTML has also been
checked in real Chromium at desktop and narrow dark-screen sizes, including
the actual graph, formulas, Laurent cards and readable benchmark charts.

These checks cover the committed examples in the local Python 3.13 environment.
They do not qualify a published release wheel, other platforms or loading the
extension into a separate Community native host. Static exports use marimo's
CDN frontend; changing numerical controls requires a running Python backend.
The earlier wheel003 rendering failure remains archived. Exact revisions,
commands, source guards and screenshots are recorded in
[`validation/python-notebooks/native-wheel004`](../../validation/python-notebooks/native-wheel004);
the mathematical and performance limits remain in the
[benchmark report](../../docs/benchmark.pdf).
