"""Read-only notebook checks using GENERATED SYNTHETIC reports, not training.

The 100-row fixture and stopped fixture are fabricated software-test inputs.
They are always labelled synthetic, live outside the package, and are never
presented as SVHN results. No model/checkpoint/torch import is required.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="e4m3_ref100_mpl_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import notebook_support as ns


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir() if path.is_file()}


def make_fixture(folder, completed):
    """Create clearly synthetic inputs, including a best-score tie at 61/62."""
    folder.mkdir(parents=True)
    count = 100 if completed else 37
    metrics, records, times, updates = [], [], [], []
    for epoch in range(1, count + 1):
        score = 60 + 35 * epoch / 61 if epoch <= 61 else 95 - .5 * max(0, epoch - 62) / 38
        test_correct = round(score * 100)
        phase_rows = {}
        for phase, correct in (("train", min(9800, 6000 + 40 * epoch)), ("test", test_correct)):
            row = {"epoch": epoch, "phase": phase, "mode": "mx", "weight_format": ns.EXPECTED_FORMAT,
                   "activation_bits": 32, "n": 10000, "correct1": correct,
                   "top1_percent": correct / 100, "count_top1_percent": correct / 100,
                   "objective_type": "hard_cross_entropy", "objective_loss": 2 / (1 + epoch / 10),
                   "lr": .001}
            metrics.append(row)
            phase_rows[phase] = [row]
        duration = {"epoch": epoch, "train_seconds": 20 + epoch % 4,
                    "test_seconds": 5, "checkpoint_seconds": .1, "other_seconds": .2,
                    "total_epoch_seconds": 25.3 + epoch % 4,
                    "first_epoch_in_process": epoch == 1,
                    "peak_gpu_allocated_bytes": 200 * 1024 ** 2,
                    "peak_gpu_reserved_bytes": 300 * 1024 ** 2}
        times.append(duration)
        records.append({**phase_rows, **duration})
        updates.append({"epoch": epoch, "layer": "layers.3", "master_changed_values": 20,
                        "master_update_l2": .04, "master_dtype": "torch.float32",
                        "changed_element_codes_including_padding": 12, "changed_scale_codes": 2})
    best = max(records, key=lambda row: row["test"][0]["top1_percent"])
    last = records[-1]
    total = sum(row["total_epoch_seconds"] for row in times)
    config = {"weight_format": ns.EXPECTED_FORMAT, "activation_bits": 32, "master_dtype": "float32",
              "dataset": "SYNTHETIC_FIXTURE_NO_SVHN", "model": "svhnq", "expand": 2,
              "train_split": "synthetic", "extra_used": False, "train_examples": 10000,
              "test_examples": 10000, "mx_block_size": 32, "mx_scale_format": "E8M0",
              "batch_size": 128, "seed": 20261009, "reference_end_epoch": 100,
              "planned_epochs": 100}
    summary = {"completed_epochs": count, "full_100_epoch_training_completed": completed,
               "synthetic_fixture_only": True, "gradient_checks_before_every_update": True,
               "fp32_masters_and_moments_verified": True, "optimizer_steps": count,
               "measured_completed_epoch_seconds": total}
    report = {"status": "PASS" if completed else "STOPPED_AT_EPOCH_BOUNDARY",
              "synthetic_fixture_only": True, "config": config, "warnings": [],
              "source_unchanged": True, "dataset_files_unchanged": True,
              "execution_settings": {"cudnn_conv_fp32_precision": "tf32", "cudnn_benchmark": True,
                                     "deterministic_algorithms": False},
              "environment": {"device_name": "SYNTHETIC_REPORT_NO_DEVICE"}}
    selection = {"policy": "first strict maximum of active-mode official-test top1",
                 "selection_mode": ns.EXPECTED_FORMAT + "/A32", "selected_on_official_test": True,
                 "independent_test_estimate": False,
                 "best": {"epoch": best["epoch"], "file": "best_checkpoint.pt",
                          "selection_score": best["test"][0]["top1_percent"]},
                 "last": {"epoch": count, "file": "last_checkpoint.pt",
                          "selection_score": last["test"][0]["top1_percent"]}}
    timing = {"available": True, "planned_epochs": 100, "steady_epochs": list(range(2, count + 1)),
              "median_epoch_seconds": 27, "estimated_100_epoch_seconds": 9000,
              "observed_range_projection_seconds": [8500, 9500],
              "measured_completed_epoch_seconds": total,
              "measured_100_epoch_seconds": total if completed else None,
              "full_100_epoch_training_completed": completed}
    # The intentionally different projection tests that completed runs display
    # the recorded sum and never reuse an old projection as a measured total.
    for name, value in (("run_report.json", report), ("history.json", {"epochs": records}),
                        ("training_summary.json", summary), ("checkpoint_selection.json", selection),
                        ("timing_estimate.json", timing)):
        write_json(folder / name, value)
    pd.DataFrame(metrics).to_csv(folder / "epoch_metrics.csv", index=False)
    pd.DataFrame(times).to_csv(folder / "epoch_times.csv", index=False)
    pd.DataFrame(updates).to_csv(folder / "weight_updates.csv", index=False)
    selected = [{"role": "best_active_test", **best["test"][0]},
                {"role": "last_epoch", **last["test"][0]}]
    pd.DataFrame(selected).to_csv(folder / "selected_metrics.csv", index=False)
    return total


def execute_cells(notebook, directory):
    """Execute plain code cells; the notebook includes a display fallback."""
    context = {"__name__": "__synthetic_notebook_check__"}
    output = io.StringIO()
    modules_before = set(sys.modules)
    with redirect_stdout(output):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"notebook_cell_{index}", "exec"), context)
            if "parameters" in cell.get("metadata", {}).get("tags", []):
                context["RUN_DIR"] = directory
                context["SAVE_FIGURES"] = False
    assert not {"torch", "run_train", "mx_e4m3", "mx_adapter"}.intersection(set(sys.modules) - modules_before)
    return context, output.getvalue()


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    notebook_path = PACKAGE / "MX_E4M3_dedicated_reference100_resultaten.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert all(cell.get("execution_count") is None and not cell.get("outputs")
               for cell in notebook["cells"] if cell["cell_type"] == "code")
    results = {}
    for name, completed in (("synthetic_complete100", True), ("synthetic_stopped37", False)):
        folder = out / name / "input"
        total = make_fixture(folder, completed)
        before = raw_hashes(folder)
        context, output = execute_cells(notebook, folder)
        run, figures = context["run"], context["figures"]
        assert run.synthetic
        assert not any(item["Niveau"] in ("FOUT", "RUNFOUT") for item in run.issues), run.issues
        assert run.reference_complete == completed
        table = ns.checkpoint_table(run)
        protocol = ns.protocol_table(run)
        assert "tf32" in protocol["Opgeslagen configuratie"].astype(str).tolist()
        assert set(figures) == {"checkpoint_accuracy", "learning_curves", "timing", "gpu_memory"}
        assert all("SYNTHETISCHE CONTROLE" in figure._suptitle.get_text() for figure in figures.values())
        assert math.isclose(ns.measured_completed_seconds(run), total)
        if completed:
            assert run.observed_best_epoch == 61 and run.last_epoch == 100
            assert table.iloc[0]["Testaccuracy [%]"] == 94.5
            assert table.iloc[1]["Epoch"] == 61 and table.iloc[1]["Testaccuracy [%]"] == 95
            assert "geen echte SVHN-training" in ns.status_markdown(run)
            assert "Gemeten totaal; geen projectie" == figures["timing"].axes[1].get_title()
            assert "Gemeten duur over alle 100 epochs" in ns.timing_markdown(run)
            assert len(figures["checkpoint_accuracy"].axes[0].patches) == 2
        else:
            assert run.last_epoch == 37 and run.metric(100) is None
            assert pd.isna(table.iloc[0]["Testaccuracy [%]"])
            assert table.iloc[2]["Epoch"] == 37
            assert "Gecontroleerd gestopt" in ns.status_markdown(run)
            assert "Epoch100 niet beschikbaar als bevestigd eindresultaat" in ns.status_markdown(run)
            assert len(figures["checkpoint_accuracy"].axes[0].patches) == 1
        preview = out / name / "preview"
        preview.mkdir()
        for key, figure in figures.items():
            figure.savefig(preview / (key + ".png"), dpi=130, bbox_inches="tight")
        (out / name / "notebook_stdout.txt").write_text(output, encoding="utf-8")
        assert before == raw_hashes(folder), "Notebook modified input reports."
        # Explicit export creates fresh folders and preserves the raw inputs.
        first = ns.save_figures({"checkpoint_accuracy": figures["checkpoint_accuracy"]}, run)
        second = ns.save_figures({"checkpoint_accuracy": figures["checkpoint_accuracy"]}, run)
        assert first != second and before == raw_hashes(folder)
        manifest = json.loads((first / "analysis_manifest.json").read_text(encoding="utf-8"))
        assert manifest["synthetic_fixture_only"] and not manifest["training_started"]
        assert not manifest["checkpoint_loaded"]
        results[name] = {"status": "PASS", "synthetic_fixture_only": True,
                         "last_epoch_in_generated_fixture": run.last_epoch,
                         "raw_input_hashes_unchanged": True,
                         "notebook_code_cells_executed": True,
                         "model_checkpoint_or_training_imported": False,
                         "preview_directory": str(preview)}
        plt.close("all")
    report = {"status": "PASS", "scope": "GENERATED_SYNTHETIC_REPORTS_ONLY_NO_TRAINING",
              "empty_template_outputs_confirmed": True, "cases": results}
    write_json(out / "notebook_validation_report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.output.resolve())
