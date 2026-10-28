"""Small integration grid: accounting, shared-source semantics and provenance."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from activation_model import activation_traffic
from experiment import run_experiment
from model_layouts import MODELS, build_layout
from traffic_model import make_switch_sequence


SMALL_CONFIG = dict(
    n_inferences=8, dwell=2, batch_size=1, main_line_bytes=64,
    cache_kib=[0, 1024], scenarios=["alternating_extremes", "dwell_extremes"],
    preload_policies=["none", "all_regions"], sensitivity_line_bytes=[],
    sensitivity_cache_kib=64,
)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_experiment(dict(SMALL_CONFIG))
        cls.summaries = {row["trace_id"]: row for row in cls.results["summary"]}

    def test_six_models_nine_layouts_eighteen_primary_mode_views(self):
        result = self.results
        self.assertEqual(len(result["models"]), 6)
        self.assertEqual(len(result["storage"]), 9)
        self.assertEqual(sum(row["primary_variant"] for row in result["storage"]), 6)
        self.assertEqual(len(result["mode_payloads"]), 26)
        primary = [row for row in result["mode_payloads"] if row["primary_variant"]]
        self.assertEqual(len(primary), 18)
        self.assertEqual(len({(row["model_id"], row["mode"]) for row in primary}), 18)
        self.assertEqual(len(result["summary"]), 9 * 2 * 2 * 2)

    def test_initial_continuation_and_category_accounting(self):
        for row in self.results["summary"]:
            with self.subTest(row=row["trace_id"]):
                self.assertEqual(row["initial_load_bytes"] + row["continuation_bytes"],
                                 row["weight_total_bytes"])
                self.assertEqual(row["codes_read_bytes"] + row["scales_read_bytes"] + row["aux_read_bytes"],
                                 row["weight_total_bytes"])
                self.assertEqual(row["prefetch_bytes"] + row["inference_only_bytes"],
                                 row["weight_total_bytes"])
                self.assertEqual(row["weight_miss_lines"] * row["line_bytes"], row["weight_total_bytes"])
                self.assertGreaterEqual(row["continuation_bytes"], 0)
                if row["preload_policy"] == "none":
                    self.assertEqual(row["initial_load_bytes"], row["first_inference_bytes"])
                else:
                    self.assertEqual(row["initial_load_bytes"], row["prefetch_bytes"])

    def test_cumulative_final_equals_total_including_prefetch(self):
        grouped = {}
        for row in self.results["trace"]:
            grouped.setdefault(row["trace_id"], []).append(row)
        for trace_id, rows in grouped.items():
            with self.subTest(trace=trace_id):
                self.assertEqual(len(rows), SMALL_CONFIG["n_inferences"])
                self.assertEqual(rows[-1]["cumulative_weight_bytes"],
                                 self.summaries[trace_id]["weight_total_bytes"])
                self.assertEqual(rows[0]["cumulative_weight_bytes"],
                                 rows[0]["initial_prefetch_bytes"] + rows[0]["weight_read_bytes"])
                for previous, current in zip(rows, rows[1:]):
                    self.assertEqual(current["cumulative_weight_bytes"] - previous["cumulative_weight_bytes"],
                                     current["weight_read_bytes"])

    def test_no_cache_is_sum_of_independent_cold_mode_loads(self):
        cold = {(row["model_id"], row["variant"], row["mode"]): row["read_bytes"]
                for row in self.results["initial_loading"]}
        models = {model["id"]: model for model in MODELS}
        for row in self.results["summary"]:
            if row["cache_kib"] != 0:
                continue
            with self.subTest(trace=row["trace_id"]):
                seq = make_switch_sequence(models[row["model_id"]]["modes"], row["scenario"],
                                           n_inferences=8, dwell=2)
                expected = sum(cold[(row["model_id"], row["variant"], str(mode))] for mode in seq)
                if row["preload_policy"] == "all_regions":
                    expected += cold[(row["model_id"], row["variant"], "ALL")]
                self.assertEqual(row["weight_total_bytes"], expected)
                self.assertEqual(row["weight_hit_ratio"], 0.0)

    def test_fit_cache_prefetch_reads_once_then_zero_inference_reads(self):
        for row in self.results["summary"]:
            if row["cache_kib"] != 1024 or row["preload_policy"] != "all_regions":
                continue
            with self.subTest(trace=row["trace_id"]):
                self.assertTrue(row["all_modes_fit_cache"])
                self.assertEqual(row["prefetch_bytes"], row["cold_image_allocated_bytes"])
                self.assertEqual(row["weight_total_bytes"], row["prefetch_bytes"])
                self.assertEqual(row["inference_only_bytes"], 0)
                self.assertEqual(row["first_inference_bytes"], 0)

    def test_source_loading_and_selected_payload_are_not_conflated(self):
        for model_id in ("dual_mantissa", "triple_scale64_b32", "triple_scale64_b64"):
            selective = build_layout(model_id, "selective")
            full = build_layout(model_id, "full_source")
            with self.subTest(model=model_id):
                self.assertEqual(selective["storage"]["total_bytes"], full["storage"]["total_bytes"])
                high = str(max(selective["model"]["modes"]))
                self.assertEqual(selective["mode_payloads"][high]["total_bytes"],
                                 full["mode_payloads"][high]["total_bytes"])
                for mode in selective["model"]["modes"]:
                    self.assertLessEqual(selective["mode_payloads"][str(mode)]["total_bytes"],
                                         full["mode_payloads"][str(mode)]["total_bytes"])
                low = str(min(selective["model"]["modes"]))
                self.assertLess(selective["mode_payloads"][low]["codes"],
                                full["mode_payloads"][low]["codes"])
        # Same group size and mode: source selection carries precisely the same
        # useful code/scale/active-BN amounts as independent direct quantization.
        for direct_id, trunc_id in (("dual_direct", "dual_mantissa"),
                                    ("triple_direct", "triple_scale64_b32")):
            direct, trunc = build_layout(direct_id), build_layout(trunc_id, "selective")
            for mode in direct["model"]["modes"]:
                with self.subTest(model=direct_id, mode=mode):
                    self.assertEqual(direct["mode_payloads"][str(mode)],
                                     trunc["mode_payloads"][str(mode)])

    def test_activation_stream_independent_of_weight_cache_and_preload(self):
        a32 = activation_traffic(32, line_bytes=64, batch_size=1)
        for row in self.results["summary"]:
            with self.subTest(trace=row["trace_id"]):
                self.assertEqual(row["activation_read_bytes"] + row["activation_write_bytes"],
                                 row["activation_total_bytes"])
                if row["model_id"] != "github5":
                    self.assertEqual(row["activation_total_bytes"], 8 * a32["total_bytes"])
        by_model_scenario = {}
        for row in self.results["summary"]:
            key = row["model_id"], row["scenario"]
            by_model_scenario.setdefault(key, set()).add(row["activation_total_bytes"])
        self.assertTrue(all(len(values) == 1 for values in by_model_scenario.values()))

    def test_accuracy_is_historical_context_not_new_measurement(self):
        interpretation = self.results["interpretation"]
        self.assertTrue(interpretation["inference_only"])
        self.assertFalse(interpretation["training"])
        self.assertFalse(interpretation["hardware_measured"])
        self.assertFalse(interpretation["actual_tensor_decode"])
        self.assertNotIn("accuracy", self.results)
        context = json.loads((Path(__file__).parent / "data" / "accuracy_context.json").read_text(encoding="utf-8"))
        self.assertEqual({row["model_id"] for row in context}, {model["id"] for model in MODELS})
        self.assertEqual(sum(len(row["epoch100"]) for row in context), 18)
        for model in context:
            self.assertIn("no new accuracy", model["origin"].lower())
            self.assertTrue(model["run_id"])
            for row in model["epoch100"]:
                self.assertAlmostEqual(row["top1_percent"], 100 * row["correct1"] / row["n"], places=5)


def run_tests() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExperimentTests)
    result = unittest.TestResult()
    suite.run(result)
    failures = [{"test": str(test), "traceback": trace} for test, trace in result.failures + result.errors]
    return {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": failures}


if __name__ == "__main__":
    unittest.main()
