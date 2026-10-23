# Extraction validation

The source provenance check compares the four pristine files, the extracted
partial-fraction engine, four exact helpers, and five coefficient-field aliases.
All fourteen checks passed on the extraction. Rustfmt's check passed for the
wrapper and adapted tests; the copied engine is excluded from formatting to
preserve its upstream bytes (including its trailing blank line).
Run it from the iregi-rust repository with:

```sh
python3 dependencies/feynkit-integrals/scripts/verify_upstream.py
```

Runtime tests require the embedding workspace to wire in this path dependency
and use its pinned Symbolica build. They cover nine focused test cases in
`tests/affine.rs`; they have not yet been compiled or executed by the extracting
agent. No Symbolica runtime or license lease has been acquired for this task.
