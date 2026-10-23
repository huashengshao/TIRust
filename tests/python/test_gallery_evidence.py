"""Focused checks for evidence direction, binding, precision display and escaping."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import median
import sys
import tempfile
import unittest

NOTEBOOKS = next(
    directory for candidate in Path(__file__).resolve().parents
    for directory in (candidate / "examples" / "notebooks", candidate / "notebooks")
    if (directory / "_presentation.py").is_file()
)
sys.path.insert(0, str(NOTEBOOKS))

from _presentation import (  # noqa: E402
    KERNEL_SOURCE,
    COHORT_CHOICES,
    JOINT_SOURCE,
    BenchmarkRow,
    benchmark_grid,
    cards,
    hero,
    laurent_cards,
    load_kernel_snapshot,
    load_benchmark_cohort,
    load_joint_snapshot,
    load_reference_times,
    project_root,
    ratio_chart,
)


class GalleryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = project_root(__file__)
        cls.kernel, cls.rows = load_kernel_snapshot(cls.root)

    def test_paired_ratio_has_tirust_in_numerator(self):
        for source, row in zip(self.kernel["rows"], self.rows, strict=True):
            with self.subTest(family=row.family, batch=row.batch):
                self.assertEqual(row.ratio, source["ratio"]["median"])
                self.assertEqual(row.ci99, tuple(source["ratio"]["ci99"]))
                self.assertIn("Rust / IREGI", source["ratio"]["definition"])
                self.assertGreater(row.ratio, 1)
                self.assertEqual(row.tirust_us,
                                 source["rust"]["total"]["median_ns"] / (1000 * row.batch))

    def test_all_failed_or_inconclusive_timing_gates_remain_visible(self):
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(Counter(row.timing_status for row in self.rows),
                         {"regression": 7, "inconclusive-noise": 1})
        self.assertTrue(all(row.numerical_status == "passed" for row in self.rows))
        markup = benchmark_grid(self.rows)
        self.assertIn("inconclusive-noise", markup)
        self.assertIn("TIRust / IREGI", markup)

    def test_evidence_hash_mismatch_is_not_silently_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / KERNEL_SOURCE
            source.parent.mkdir(parents=True)
            source.write_text('{"rows": []}\n')
            with self.assertRaisesRegex(ValueError, "Benchmark evidence changed"):
                load_kernel_snapshot(root)

    def test_original_reference_cohort_does_not_invent_tirust_ratios(self):
        source, rows = load_reference_times(self.root)
        self.assertEqual(len(rows), 8)
        self.assertEqual(source["counts"], {"inconclusive-short-block": 1, "improvement": 7})
        for row in rows:
            self.assertFalse(any("ratio" in key.lower() or "tirust" in key.lower() for key in row))

    def test_arbitrary_precision_display_never_calls_float(self):
        class Precise:
            def __str__(self):
                return "1.000000000000000000000000000000000001"

            def __float__(self):
                raise AssertionError("Lossy conversion is prohibited in result display")

        markup = cards([("Coefficient", Precise())])
        self.assertIn(str(Precise()), markup)

    def test_display_escapes_dynamic_values(self):
        attack = '<img src=x onerror="alert(1)">'
        for markup in (cards([(attack, attack)]), hero(attack, attack),
                       benchmark_grid([replace(self.rows[0], timing_status=attack)])):
            self.assertNotIn("<img", markup)
            self.assertIn("&lt;img", markup)

    def test_laurent_display_requires_all_four_slots(self):
        with self.assertRaises(ValueError):
            laurent_cards([1, 2, 3], ["uv", "ir", "finite"])
        markup = laurent_cards([0, 0, 0, "2.5"], ["uv_single", "ir_single", "ir_double", "finite"])
        self.assertIn("ir_double", markup)
        self.assertIn("2.5", markup)

    def test_unmeasured_ratio_stays_unmeasured(self):
        row = BenchmarkRow("pending", 5, 2, 32, None, None, None, None,
                           "unsupported", "unmeasured", None)
        self.assertIn("Unmeasured", benchmark_grid([row]))
        self.assertIn("No paired ratio", ratio_chart([row]))

    def test_uncertainty_chart_is_accessible(self):
        markup = ratio_chart(self.rows)
        self.assertIn('role="img"', markup)
        self.assertIn("TIRust divided by IREGI", markup)
        self.assertIn("99 percent confidence intervals", markup)
        self.assertEqual(markup.count("<circle"), len(self.rows))

    def test_same_cohort_ratios_match_all_forty_saved_results(self):
        expected_counts = {
            ("baselineTIRust", "iregi"): {"regression": 7, "inconclusive-short-block": 1},
            ("candidateTIRust", "iregi"): {"regression": 6, "improvement": 1, "inconclusive-short-block": 1},
            ("baselineTIRust", "collier"): {"regression": 7, "inconclusive-short-block": 1},
            ("candidateTIRust", "collier"): {"regression": 7, "inconclusive-short-block": 1},
            ("candidateTIRust", "baselineTIRust"): {"improvement": 7, "inconclusive-short-block": 1},
        }
        checked = 0
        for (numerator, denominator), counts in expected_counts.items():
            source, rows = load_joint_snapshot(self.root, numerator, denominator)
            self.assertEqual(Counter(row.timing_status for row in rows), counts)
            self.assertEqual(len(rows), 8)
            for saved, row in zip(source["rows"], rows, strict=True):
                exact = next(r for r in saved["ratios"]
                             if (r["numerator"], r["denominator"]) == (numerator, denominator))
                self.assertEqual(row.ratio, exact["statistics"]["median"])
                self.assertEqual(row.ci99, tuple(exact["statistics"]["ci99"]))
                self.assertEqual(row.timing_status, exact["gate"])
                self.assertEqual(row.tirust_us,
                                 median(saved["engine_samples_ns"][numerator]) / (1000 * row.batch))
                self.assertEqual(row.reference_us,
                                 median(saved["engine_samples_ns"][denominator]) / (1000 * row.batch))
                self.assertEqual(row.numerical_status, "passed")
                checked += 1
        self.assertEqual(checked, 40)

    def test_short_family_uses_row_gate_and_remains_amber(self):
        source, rows = load_joint_snapshot(self.root)
        self.assertEqual(source["rows"][0]["ratios"][0]["statistics"]["gate"], "correctness-failed")
        self.assertEqual(rows[0].timing_status, "inconclusive-short-block")
        markup = benchmark_grid(rows)
        self.assertIn("#fff0c7", markup)
        self.assertIn("TIRust (selected) / COLLIER", markup)
        self.assertIn("TIRust (selected) divided by COLLIER", ratio_chart(rows))

    def test_new_evidence_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / JOINT_SOURCE
            source.parent.mkdir(parents=True)
            source.write_text('{"rows": []}\n')
            with self.assertRaisesRegex(ValueError, "Benchmark evidence changed"):
                load_joint_snapshot(root)

    def test_only_measured_tirust_numerator_comparisons_are_allowed(self):
        for pair in [("collier", "iregi"), ("iregi", "candidateTIRust"),
                     ("candidateTIRust", "unmeasured")]:
            with self.assertRaisesRegex(ValueError, "TIRust in the numerator"):
                load_joint_snapshot(self.root, *pair)

    def test_historical_and_private_cohorts_are_separate_choices(self):
        self.assertEqual(len(COHORT_CHOICES), 6)
        for choice in COHORT_CHOICES.values():
            metadata, rows = load_benchmark_cohort(self.root, choice)
            self.assertEqual(len(rows), 8)
            self.assertIn("Historical", metadata["scope"])
            if choice == "historical":
                self.assertEqual(metadata["source"], KERNEL_SOURCE)
                self.assertEqual(rows, self.rows)
            else:
                self.assertEqual(metadata["source"], JOINT_SOURCE)
                self.assertIn("posthoc", metadata["analysis"])
        with self.assertRaisesRegex(ValueError, "separately measured"):
            load_benchmark_cohort(self.root, "cross-session")

    def test_mixed_comparison_labels_cannot_silently_share_a_chart(self):
        _, direct = load_joint_snapshot(self.root)
        for display in (benchmark_grid, ratio_chart):
            with self.assertRaisesRegex(ValueError, "separate tables"):
                display([self.rows[0], direct[0]])


if __name__ == "__main__":
    unittest.main()
