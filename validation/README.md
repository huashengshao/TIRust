# validation/

This directory keeps only the fixtures that tests, the CLI default and the
examples read (listed in [scripts/keep-validation.txt](../scripts/keep-validation.txt)).
The validation evidence (captures, qualification records, reviews, reference
builders, payload archives, logs and the full benchmark data) lives outside the
repository:

PENDING-EVIDENCE-ARCHIVE: the URL and commit of the evidence archive from the reply on huashengshao/TIRust#15.

The full benchmark report and its evidence bundle of 6 November 2026 are assets of the
release [tirust-bench-20261106](https://github.com/huashengshao/TIRust/releases/tag/tirust-bench-20261106);
[docs/benchmark.pdf](../docs/benchmark.pdf) is the slim summary.
A path `validation/<path>` cited in the documentation or in source comments
that is not one of the files below refers to the same path in the evidence
archive. New captures written under `validation/` are git-ignored (slimness
policy, docs/PYAMPLICOL-COLLABORATION.md section 5).

| Fixture | Read by |
|---|---|
| `physical/ggh-numerator-reference.json` | `crates/iregi-prepare/tests/madloop_numerator.rs` |
| `physical/madloop/fixtures/{gghh,gghhh,gggg-up}-helicity-1.json` | `crates/iregi-prepare/tests/madloop_numerator.rs` |
| `physical/gghh-tensor-reference.json` | `crates/iregi-rust/tests/physical_box.rs` |
| `physical/gghhh-cli-input.json` | `crates/iregi-rust/examples/pv_physical_precision.rs` (default input) |
| `reference/fixtures.json` | `crates/iregi-rust/tests/oracle.rs`, unit tests in `plan.rs` and `rescue.rs`, `iregi-cli compare` (default) |
| `reference/snapping/reductions.json` | `crates/iregi-rust/tests/legacy_rescue.rs`, unit tests in `snapping.rs` |
| `pv-high-rank/lightlike-feynman-parameters.json` | unit tests in `crates/iregi-rust/src/pv.rs` |
| `highpoint-seven-point-current-cargo/fixture.json` | unit tests in `certified_tensor.rs`, `tests/python/test_tirust_dispatch.py` |
| `generalized-pv/reference{,-mixed,-complex}-capture.json` | `crates/iregi-rust/tests/generalized_pv.rs` |
| `public-highpoint/fixture-conditioned_masses.json` | `tests/python/test_tirust_dispatch.py` |
| `benchmark/opening-rank6-20261026/inputs.json` | `crates/iregi-rust/tests/direct_engine.rs` (reference values in `crates/iregi-rust/tests/fixtures/opening-rank6-20261026-references.json`) |
| `benchmark/baseline-6b24c03-licensed/results.json`, `collier-reference/timing-initial8/summary.json`, `selected-native-four-engine/{posthoc-revision2/{summary,qualification},v3-frozen/qualification}.json` | `examples/notebooks/_presentation.py` (sha256-bound), through `tests/python/test_gallery_{evidence,runtime}.py` |
