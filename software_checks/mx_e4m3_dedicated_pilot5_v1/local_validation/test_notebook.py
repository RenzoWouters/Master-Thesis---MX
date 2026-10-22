"""Offline analysis checks; fixtures are synthetic and live only in temporary folders."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import notebook_support as ns


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_fixture(folder, synthetic=True):
    folder.mkdir(parents=True, exist_ok=True)
    scores = [25, 40, 50, 50, 45]  # Tie: first strict maximum is epoch3.
    records, metrics, times, updates = [], [], [], []
    for epoch, test_score in enumerate(scores, 1):
        rows = {}
        for phase, score in (("train", 25 + 8 * epoch), ("test", test_score)):
            row = {"epoch": epoch, "phase": phase, "mode": "mx", "weight_format": ns.EXPECTED_FORMAT,
                   "activation_bits": 32, "n": 100, "correct1": score,
                   "top1_percent": float(score), "count_top1_percent": float(score),
                   "objective_type": "hard_cross_entropy", "objective_loss": 2 / epoch,
                   "lr": .001}
            metrics.append(row)
            rows[phase] = [row]
        time = {"epoch": epoch, "train_seconds": 45 + epoch, "test_seconds": 4,
                "checkpoint_seconds": 1, "other_seconds": 1, "total_epoch_seconds": 51 + epoch,
                "first_epoch_in_process": epoch == 1, "peak_gpu_allocated_bytes": (200 + epoch) * 1024 ** 2,
                "peak_gpu_reserved_bytes": 300 * 1024 ** 2}
        times.append(time)
        records.append({**rows, **time})
        updates.append({"epoch": epoch, "layer": "layers.3", "master_changed_values": 20,
                        "master_update_l2": .04, "master_dtype": "torch.float32",
                        "changed_element_codes_including_padding": 12, "changed_scale_codes": 2})
    config = {"weight_format": ns.EXPECTED_FORMAT, "activation_bits": 32, "master_dtype": "float32",
              "dataset": "svhn", "model": "svhnq", "expand": 2, "train_split": "train",
              "extra_used": False, "train_examples": 100, "test_examples": 100,
              "mx_block_size": 32, "mx_scale_format": "E8M0", "batch_size": 128,
              "seed": 20261009, "pilot_stop_epoch": 5, "planned_epochs": 100}
    report = {"status": "PASS", "synthetic_fixture_only": synthetic, "config": config,
              "warnings": [], "source_unchanged": True, "dataset_files_unchanged": True}
    selection = {"policy": "first strict maximum of active-mode official-test top1",
                 "selection_mode": ns.EXPECTED_FORMAT + "/A32", "selected_on_official_test": True,
                 "independent_test_estimate": False,
                 "best": {"epoch": 3, "file": "best_checkpoint.pt", "selection_score": 50.},
                 "last": {"epoch": 5, "file": "last_checkpoint.pt", "selection_score": 45.}}
    summary = {"completed_epochs": 5, "pilot_completed": True, "full_100_epoch_training_completed": False,
               "synthetic_fixture_only": synthetic, "gradient_checks_before_every_update": True,
               "fp32_masters_and_moments_verified": True, "optimizer_steps": 5}
    timing = {"available": True, "planned_epochs": 100, "steady_epochs": [2, 3, 4, 5],
              "median_epoch_seconds": 54.5, "estimated_100_epoch_seconds": 52 + 99 * 54.5,
              "observed_range_projection_seconds": [52 + 99 * 53, 52 + 99 * 56]}
    for filename, content in (("run_report.json", report), ("history.json", {"epochs": records}),
                              ("checkpoint_selection.json", selection), ("training_summary.json", summary),
                              ("timing_estimate.json", timing)):
        write_json(folder / filename, content)
    pd.DataFrame(metrics).to_csv(folder / "epoch_metrics.csv", index=False)
    pd.DataFrame(times).to_csv(folder / "epoch_times.csv", index=False)
    pd.DataFrame(updates).to_csv(folder / "weight_updates.csv", index=False)
    selected = [{"role": "best_active_test", **records[2]["test"][0]},
                {"role": "last_epoch", **records[4]["test"][0]}]
    pd.DataFrame(selected).to_csv(folder / "selected_metrics.csv", index=False)
    return folder


def raw_hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir() if path.is_file()}


class NotebookChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="e4m3_notebook_synthetic_")
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        plt.close("all")
        self.tmp.cleanup()

    def test_first_strict_max_and_no_invented_epoch100(self):
        run = ns.load_run(make_fixture(self.folder / "fixture"))
        self.assertEqual(run.observed_best_epoch, 3)
        self.assertEqual(run.last_epoch, 5)
        self.assertIsNone(run.metric(100))
        self.assertIn("SYNTHETISCHE CONTROLE", ns.status_markdown(run))
        self.assertFalse(any(item["Niveau"] == "FOUT" for item in run.issues), run.issues)
        fig = ns.plot_checkpoint_accuracy(run)
        self.assertEqual(len(fig.axes[1].patches), 0)
        self.assertTrue(any("GEEN EINDRESULTAAT" in text.get_text() for text in fig.axes[1].texts))

    def test_notebook_executes_with_empty_and_synthetic_inputs(self):
        notebook = json.loads((PACKAGE / "MX_E4M3_dedicated_pilot5_resultaten.ipynb").read_text(encoding="utf-8"))
        empty = self.folder / "empty"
        empty.mkdir()
        fixture = make_fixture(self.folder / "fixture")
        for directory in (empty, fixture):
            before = raw_hashes(directory)
            context = {"__name__": "__analysis_check__"}
            with redirect_stdout(io.StringIO()):
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    exec(compile("".join(cell["source"]), "notebook_cell", "exec"), context)
                    if "parameters" in cell.get("metadata", {}).get("tags", []):
                        context["RUN_DIR"] = directory
                        context["SAVE_FIGURES"] = False
            self.assertEqual(before, raw_hashes(directory))
            self.assertEqual(set(context["figures"]), {"checkpoint_accuracy", "learning_curves", "timing", "gpu_memory"})

    def test_figure_export_creates_new_folder_and_preserves_raw_files(self):
        directory = make_fixture(self.folder / "fixture")
        before = raw_hashes(directory)
        run = ns.load_run(directory)
        figures = {"checkpoint_accuracy": ns.plot_checkpoint_accuracy(run), "timing": ns.plot_timing(run)}
        first, second = ns.save_figures(figures, run), ns.save_figures(figures, run)
        self.assertNotEqual(first, second)
        self.assertTrue((first / "checkpoint_accuracy.png").is_file())
        self.assertEqual(before, raw_hashes(directory))
        manifest = json.loads((first / "analysis_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["synthetic_fixture_only"])
        self.assertFalse(manifest["training_started"])

    def test_selection_disagreement_and_failed_run_remain_visible(self):
        directory = make_fixture(self.folder / "fixture")
        selection_path = directory / "checkpoint_selection.json"
        selection = json.loads(selection_path.read_text())
        selection["best"]["epoch"] = 4
        write_json(selection_path, selection)
        report_path = directory / "run_report.json"
        report = json.loads(report_path.read_text())
        report.update(status="FAIL", error="synthetic example error")
        write_json(report_path, report)
        run = ns.load_run(directory)
        self.assertTrue(any("opgeslagen epoch 4, verwacht 3" in item["Melding"] for item in run.issues))
        self.assertTrue(any(item["Niveau"] == "RUNFOUT" for item in run.issues))
        self.assertIn("niet als PASS", ns.status_markdown(run))

    def test_missing_memory_is_not_zero_and_history_fallback_is_labelled(self):
        directory = make_fixture(self.folder / "fixture")
        (directory / "epoch_metrics.csv").unlink()
        run = ns.load_run(directory)
        self.assertEqual(run.last_epoch, 5)
        self.assertTrue(any("gebruiken history.json" in item["Melding"] for item in run.issues))
        run.times["peak_gpu_allocated_bytes"] = None
        run.times["peak_gpu_reserved_bytes"] = None
        fig = ns.plot_memory(run)
        self.assertEqual(len(fig.axes[0].lines), 0)
        self.assertTrue(any("niet als nul" in text.get_text() for text in fig.axes[0].texts))

    def test_wrong_precision_does_not_receive_e4m3_graph(self):
        directory = make_fixture(self.folder / "fixture")
        path = directory / "epoch_metrics.csv"
        metrics = pd.read_csv(path)
        metrics["weight_format"] = "MXFP6_E2M3"
        metrics.to_csv(path, index=False)
        run = ns.load_run(directory)
        self.assertFalse(run.metrics_safe)
        self.assertTrue(run.test.empty)
        self.assertTrue(any("verwacht uitsluitend" in item["Melding"] for item in run.issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
