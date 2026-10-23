"""Wheel-backed notebook integration checks. Never substitute a numerical mock."""

import importlib.util
from pathlib import Path
import sys
import unittest

NOTEBOOKS = next(
    directory for candidate in Path(__file__).resolve().parents
    for directory in (candidate / "examples" / "notebooks", candidate / "notebooks")
    if (directory / "01_numerator_to_integral.py").is_file()
)
sys.path.insert(0, str(NOTEBOOKS))


def load_app(filename):
    path = NOTEBOOKS / filename
    spec = importlib.util.spec_from_file_location("gallery_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.app


class NativeGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An absent/broken native extension is a failure, never an automatic skip.
        import tirust
        cls.tirust = tirust

    def test_graph_batch_and_contraction_use_shared_types(self):
        _, definitions = load_app("01_numerator_to_integral.py").run()
        self.assertIsInstance(definitions["diagram"], self.tirust.FeynmanDiagram)
        self.assertIs(definitions["graph"].diagram, definitions["diagram"])
        graph_html = definitions["diagram"]._repr_html_()
        self.assertIn("<svg", graph_html)
        self.assertIn("Feynman diagram", graph_html)
        self.assertIsInstance(definitions["numerator"].expression, self.tirust.Expression)
        tensors = definitions["tensors"]
        contraction = definitions["integral"]
        self.assertEqual(tensors.success, (True, True, True))
        self.assertEqual(contraction.success, (True, True, True))
        self.assertEqual(len(tensors.laurent_names), 4)
        self.assertEqual(len(tensors.lane(0)), 15)
        self.assertEqual(len(contraction.lane(0)), 4)
        self.assertEqual(tensors.stats["input_rows"], 3)
        self.assertGreaterEqual(tensors.stats["master_rows"], 3)
        self.assertNotEqual(contraction.lane(0), contraction.lane(1))

    def test_alternate_graph_and_numeric_controls(self):
        import marimo as mo
        for graph, dtype in (("triangle", "float"), ("bubble", "double-double")):
            with self.subTest(graph=graph, dtype=dtype):
                _, definitions = load_app("01_numerator_to_integral.py").run(defs={
                    "topology": mo.ui.dropdown({graph: graph}, value=graph),
                    "sample_count": mo.ui.dropdown({"Three": 3}, value="Three"),
                    "numeric_type": mo.ui.dropdown({dtype: dtype}, value=dtype),
                })
                self.assertTrue(all(definitions["tensors"].success))
                self.assertTrue(all(definitions["integral"].success))
                value = definitions["integral"].lane(0)[3]
                self.assertIsInstance(value, self.tirust.ComplexFloat)

    def test_edited_topology_is_stopped_before_mismatched_reduction(self):
        import marimo as mo
        triangle = NOTEBOOKS / "01_numerator_to_integral.py"
        from _presentation import project_root
        dot = (project_root(triangle) / "examples" / "graphs" / "triangle.dot").read_text()
        _, definitions = load_app("01_numerator_to_integral.py").run(defs={
            "dot_editor": mo.ui.text_area(value=dot),
        })
        # The graph count guard stops the branch instead of integrating its
        # numerator with the bubble's two supplied denominators.
        self.assertNotIn("integral", definitions)

    def test_dots_dual_precision_and_invalid_neighbor(self):
        _, definitions = load_app("02_dots_and_derivatives.py").run()
        direct = definitions["dotted_result"]
        derivative = definitions["derivative_result"]
        differences = definitions["difference_result"]
        self.assertTrue(all(direct.success + derivative.success + differences.success))
        self.assertEqual(definitions["powers"], [2, 1])
        threshold = self.tirust.Float("1e-20", precision=128)
        for slot in range(4):
            value = direct.lane(0)[0][slot]
            dual = derivative.lane(0)[0][slot]
            self.assertIsInstance(dual, self.tirust.Dual)
            discrepancy = abs(dual.values[1] - value) / (1 + abs(value))
            self.assertLess(discrepancy, threshold)
        precise = definitions["precision_result"]
        self.assertTrue(all(precise.success))
        self.assertNotEqual(precise.lane(0)[0][3], precise.lane(1)[0][3])
        self.assertTrue(definitions["isolation_verified"])
        self.assertIsNone(definitions["isolated_result"].lanes()[1])
        with self.assertRaises(self.tirust.LaneError):
            definitions["isolated_result"].lane(1)


class BenchmarkGalleryTests(unittest.TestCase):
    def test_explorer_runs_without_numerical_extension(self):
        _, definitions = load_app("03_benchmark_explorer.py").run()
        self.assertEqual(len(definitions["selected_rows"]), 8)
        self.assertEqual(len(definitions["reference_rows"]), 8)
        self.assertTrue(all(row.ratio > 1 for row in definitions["selected_rows"]))

    def test_explorer_filters_without_changing_qualification(self):
        import marimo as mo
        _, definitions = load_app("03_benchmark_explorer.py").run(defs={
            "topology_choice": mo.ui.dropdown({"Triangle": 3}, value="Triangle"),
            "batch_choice": mo.ui.dropdown({"Singleton": 1}, value="Singleton"),
        })
        rows = definitions["selected_rows"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.propagators == 3 and row.batch == 1 for row in rows))
        self.assertEqual({row.timing_status for row in rows},
                         {"regression", "inconclusive-noise"})


if __name__ == "__main__":
    unittest.main()
