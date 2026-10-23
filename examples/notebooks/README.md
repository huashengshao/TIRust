# TIRust gallery

Three short, reactive notebooks walk from a symbolic graph to its numerical
integral, then explore dotted propagators and the measured performance evidence.

| Notebook | What to explore |
| --- | --- |
| `01_numerator_to_integral.py` | Edit a bubble or triangle DOT graph, inspect its Symbolica numerator, and reduce a batch with separated UV/IR/finite coefficients. |
| `02_dots_and_derivatives.py` | Raise a propagator power, compare a Dual mass derivative and a finite difference, change precision, and inspect an invalid neighbor lane. |
| `03_benchmark_explorer.py` | Select topology and batch size, inspect paired TIRust/reference ratios and uncertainty, and read the exact evidence and limitations. |

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
Original IREGI/COLLIER times remain a separate cohort. Unmeasured direct
TIRust/COLLIER ratios are not reconstructed from unrelated sessions.

Graph import, polynomial preparation, contraction and reduction call the Rust
libraries. `_presentation.py` only formats output and reads benchmark evidence.
It does not implement symbolic algebra or numerical reduction.

The visual sequence is inspired by the
[HEPKit four-loop numerator notebook](https://hepkit.org/gallery/notebooks/four_loop_numerator.py):
direct displays of real graph and symbolic objects, concise mathematical
explanations and focused reactive controls. TIRust adds a restrained navy/serif
title, readable Laurent cards and accessible uncertainty charts.

Native-wheel execution and headless exports are currently pending. Source and
API review are separate from successful runtime qualification; the source is
not presented as an already validated release. Exact reference revisions and
the latest executed checks are recorded in
[`validation/python-notebooks`](../../validation/python-notebooks).
