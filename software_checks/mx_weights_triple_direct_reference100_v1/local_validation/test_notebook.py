"""Notebook-only validation with GENERATED SYNTHETIC reports; no training.

These fixtures deliberately give each mode a different best epoch. They test
analysis and presentation, never SVHN accuracy or hardware performance. Output
stays in the caller-selected temporary directory; no executed notebook is saved.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="triple_direct_notebook_mpl_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import notebook_support as ns

BANNED_IMPORTS = {"torch", "run_train", "joint_adapter", "joint_support", "triple_adapter", "triple_support", "resume_core",
                  "svhn_support", "mx_adapter", "mx_format", "mx_e2m1", "mx_e4m3"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir() if path.is_file()}


def make_fixture(folder, count):
    """Fake reports: E4M3 best2 (tie3), E2M3 best4, E2M1 best5; later scores lower."""
    folder.mkdir(parents=True)
    metrics, records, times, updates = [], [], [], []
    correct_by_bit = {8: [6, 9, 9, 8, 8], 6: [7, 8, 9, 10, 8], 4: [5, 7, 8, 7, 9]}
    for bit in (4, 6, 8):
        correct_by_bit[bit].extend([7 if bit == 4 else 8] * max(0, count - 5))
    for epoch in range(1, count + 1):
        phase_rows = {}
        for phase in ("train", "test"):
            phase_rows[phase] = []
            for bit in (4, 6, 8):
                n = 20 if phase == "train" else 10
                correct = min(20, 14 + epoch) if phase == "train" else correct_by_bit[bit][epoch - 1]
                soft = phase == "train" and bit != 8
                row = {"epoch": epoch, "phase": phase, "mode": "joint", "weight_bits": bit,
                       "weight_format": ns.FORMATS[bit], "activation_bits": 32, "bn_bank_key": ns.BANKS[bit],
                       "n": n, "correct1": correct, "correct5": n, "top1_percent": 100 * correct / n,
                       "count_top1_percent": 100 * correct / n, "top5_percent": 100., "lr": .001,
                       "objective_type": "recursive_soft_cross_entropy" if soft else "hard_cross_entropy",
                       "teacher_weight_format": ns.FORMATS[8 if bit == 6 else 6] if soft else None,
                       "objective_loss": (1 + (8 - bit) / 10) / (1 + epoch) if phase == "train" else .2 + (10 - correct) / 10}
                metrics.append(row)
                phase_rows[phase].append(row)
        duration = {"epoch": epoch, "train_seconds": 9.5 + epoch, "test_seconds": 2.25,
                    "checkpoint_seconds": .25, "other_seconds": .5, "total_epoch_seconds": 12.5 + epoch,
                    "first_epoch_in_process": epoch == 1,
                    "peak_gpu_allocated_bytes": (200 + epoch) * 1024 ** 2,
                    "peak_gpu_reserved_bytes": 300 * 1024 ** 2}
        times.append(duration)
        epoch_updates = []
        for layer in ("layers.3", "layers.6", "layers.10", "layers.13", "layers.16", "layers.20"):
            for bit in (4, 6, 8):
                epoch_updates.append({"epoch": epoch, "mode": "joint", "weight_bits": bit, "layer": layer,
                                      "master_changed_values": 20, "master_update_l2": .04,
                                      "master_dtype": "torch.float32", "changed_element_codes_including_padding": bit,
                                      "changed_scale_codes": 2})
        updates.extend(epoch_updates)
        records.append({**duration, **phase_rows, "optimizer_steps": 2 * epoch,
                        "test_optimizer_steps": 0, "batchnorm_counters_correct": True,
                        "lr_used": .001, "lr_next": .001,
                        "gradient_checks": {"optimizer_steps_in_phase": 2, "finite_gradient_checks": 2,
                                            "detached_teacher_checks": 4, "forward_calls_E4M3": 2,
                                            "forward_calls_E2M3": 2, "forward_calls_E2M1": 2,
                                            "no_extra_FP32_teacher": True},
                        "adam_checks": {"all_active_adam_steps_correct": True, "moments_fp32_finite": True,
                                        "unused_bn_has_no_optimizer_state": True},
                        "weight_updates": epoch_updates})
    best, last = records[1], records[-1]
    total = sum(row["total_epoch_seconds"] for row in times)
    config = {"schema": "mx-weights-triple-direct-reference100-protocol-v1", "mode": "joint",
              "weight_bits": [4, 6, 8], "weight_formats": {str(k): v for k, v in ns.FORMATS.items()},
              "bn_bank_mapping": {str(k): str(v) for k, v in ns.BANKS.items()},
              "activation_bits": 32, "master_dtype": "float32", "distillation": True, "truncation": False,
              "extra_FP32_teacher_forward": False, "dataset": "GENERATED_SYNTHETIC_NO_SVHN",
              "model": "svhnq", "expand": 2, "train_split": "synthetic", "extra_used": False,
              "train_examples": 20, "test_examples": 10, "batch_size": 10, "seed": 20261009,
              "mx_block_size": 32, "mx_scale_format": "E8M0", "mx_scale_policy": "independent direct format-specific scales",
              "planned_epochs": 100, "scheduler_milestones": [50, 75, 90]}
    summary = {"completed_epochs": count,
               "full_100_epoch_training_completed": count == 100, "best_epoch_E4M3_A32": 2, "last_epoch": count,
               "optimizer_steps": 2 * count, "training_forward_passes": 6 * count,
               "gradient_checks_before_every_update": True, "fp32_masters_and_moments_verified": True,
               "measured_completed_epoch_seconds": total, "synthetic_fixture_only": True}
    selection = {"policy": "first strict maximum of E4M3/A32 official-test top1",
                 "selection_mode": ns.SELECTION_FORMAT + "/A32", "all_modes_from_same_selected_checkpoint": True,
                 "selected_on_official_test": True, "independent_test_estimate": False,
                 "same_checkpoint": False}
    for role, record in (("best", best), ("last", last)):
        selection[role] = {"epoch": record["epoch"], "file": role + "_checkpoint.pt",
                           "sha256": str(record["epoch"]) * 64,
                           "original_checkpoint": "SYNTHETIC_METADATA_ONLY_NO_CHECKPOINT_FILE",
                           "selection_score_E4M3": next(row["top1_percent"] for row in record["test"] if row["weight_bits"] == 8)}
    exports = {}
    for role, record in (("best_E4M3_test", best), ("last_epoch", last)):
        exports[role] = {str(bit): {"epoch": record["epoch"], "weight_bits": bit,
                                  "file": f"SYNTHETIC_METADATA_{bit}_{role}.pt", "bytes": 123,
                                  "decoded_weights_exact": True, "six_fp32_masters_absent": True,
                                  "full_test_forward_rerun": False, "truncation": False} for bit in (4, 6, 8)}
    report = {"schema": "mx-weights-triple-direct-reference100-report-v1",
              "status": "PASS" if count == 100 else "STOPPED_AT_EPOCH_BOUNDARY",
              "config": config, "synthetic_fixture_only": True, "warnings": [],
              "source_unchanged": True, "dataset_files_unchanged": True,
              "execution_settings": {"cudnn_conv_fp32_precision": "tf32", "cudnn_benchmark": True,
                                     "deterministic_algorithms": False}, "mx_exports": exports}
    timing = {"available": True, "planned_epochs": 100, "steady_epochs": list(range(2, count + 1)),
              "median_epoch_seconds": 16, "first_epoch_seconds": times[0]["total_epoch_seconds"],
              "estimated_100_epoch_seconds": 1597.5, "observed_range_projection_seconds": [1498.5, 1696.5],
              "range_is_confidence_interval": False, "measured_completed_epoch_seconds": total,
              "measured_100_epoch_seconds": total if count == 100 else None}
    for name, value in (("run_report.json", report), ("history.json", {"epochs": records}),
                        ("training_summary.json", summary), ("checkpoint_selection.json", selection),
                        ("timing_estimate.json", timing)):
        write_json(folder / name, value)
    pd.DataFrame(metrics).to_csv(folder / "epoch_metrics.csv", index=False)
    pd.DataFrame(times).to_csv(folder / "epoch_times.csv", index=False)
    pd.DataFrame(updates).to_csv(folder / "weight_updates.csv", index=False)
    selected = [{"role": role, **row} for role, record in (("best_E4M3_test", best), ("last_epoch", last)) for row in record["test"]]
    pd.DataFrame(selected).to_csv(folder / "selected_metrics.csv", index=False)
    shutil.copyfile(PACKAGE / "notebook_support.py", folder / "notebook_support.py")
    return total


def execute_cells(notebook, directory, save=False):
    """Execute code cells without model imports; never serialize an executed notebook."""
    context = {"__name__": "__synthetic_notebook_validation__"}
    output = io.StringIO()
    modules_before = set(sys.modules)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in BANNED_IMPORTS:
            raise AssertionError("Analysis imported forbidden model/training module: " + name)
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    try:
        with redirect_stdout(output):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                exec(compile("".join(cell["source"]), f"notebook_cell_{index}", "exec"), context)
                if "parameters" in cell.get("metadata", {}).get("tags", []):
                    context["RUN_DIR"], context["SAVE_FIGURES"] = directory, save
    finally:
        builtins.__import__ = original_import
    assert not BANNED_IMPORTS.intersection(set(sys.modules) - modules_before)
    return context, output.getvalue()


def negative_checks(out, source, complete_source):
    """Reject exactly the reporting mistakes that could change the interpretation."""
    checks = {}
    for case in ("mixed_mode_best", "missing_test_mode", "wrong_soft_loss_label", "false_epoch100_claim", "wrong_teacher_count",
                 "complete100_mixed_mode_best", "complete100_wrong_measured_duration"):
        folder = out / case
        folder.mkdir()
        fixture_source = complete_source if case.startswith("complete100_") else source
        for path in fixture_source.iterdir():
            if path.is_file():
                shutil.copyfile(path, folder / path.name)
        if case in ("mixed_mode_best", "complete100_mixed_mode_best"):
            data = pd.read_csv(folder / "selected_metrics.csv")
            # Preserve the low mode's actual epoch4 score, but incorrectly combine
            # it with E4M3's selected epoch2: a tempting per-mode-best mixture.
            metrics = pd.read_csv(folder / "epoch_metrics.csv")
            wrong = metrics.loc[(metrics.phase == "test") & (metrics.epoch == 4) & (metrics.weight_bits == 6)].iloc[0]
            mask = (data.role == "best_E4M3_test") & (data.weight_bits == 6)
            for key in metrics.columns:
                data.loc[mask, key] = wrong[key]
            data.to_csv(folder / "selected_metrics.csv", index=False)
        elif case == "missing_test_mode":
            path = folder / "epoch_metrics.csv"
            data = pd.read_csv(path)
            data.loc[~((data.phase == "test") & (data.epoch == 5) & (data.weight_bits == 4))].to_csv(path, index=False)
        elif case == "wrong_soft_loss_label":
            path = folder / "epoch_metrics.csv"
            data = pd.read_csv(path)
            data.loc[(data.phase == "train") & (data.weight_bits == 4), "objective_type"] = "hard_cross_entropy"
            data.to_csv(path, index=False)
        elif case == "false_epoch100_claim":
            path = folder / "training_summary.json"
            data = json.loads(path.read_text())
            data["full_100_epoch_training_completed"] = True
            write_json(path, data)
        elif case == "complete100_wrong_measured_duration":
            path = folder / "timing_estimate.json"
            data = json.loads(path.read_text())
            data["measured_100_epoch_seconds"] += 10
            write_json(path, data)
        else:
            path = folder / "history.json"
            data = json.loads(path.read_text())
            data["epochs"][0]["gradient_checks"]["detached_teacher_checks"] = 2
            write_json(path, data)
        before = raw_hashes(folder)
        run = ns.load_run(folder)
        errors = [row for row in run.issues if row["Niveau"] == "FOUT"]
        assert errors, case
        assert not run.reference_complete and raw_hashes(folder) == before
        table = ns.checkpoint_table(run)
        assert table.loc[table.Rol == "epoch100", "Testaccuracy [%]"].isna().all()
        if case in ("mixed_mode_best", "complete100_mixed_mode_best"):
            assert not run.selection_safe
            assert table.loc[table.Rol == "best_E4M3_test", "Testaccuracy [%]"].isna().all()
            assert table.loc[table.Rol == "last_epoch", "Testaccuracy [%]"].notna().all()
        if case in ("missing_test_mode", "wrong_soft_loss_label"):
            assert not run.metrics_safe and run.test.empty
        checks[case] = {"expected_rejection": True, "messages": [row["Melding"] for row in errors]}
    return checks


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    path = PACKAGE / "MX_triple_direct_reference100_resultaten.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    template_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(cell.get("execution_count") is None and not cell.get("outputs")
               for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "SAVE_FIGURES = False" in "".join(notebook["cells"][1]["source"])
    results = {}
    for name, count in (("synthetic_complete100", 100), ("synthetic_stopped5", 5)):
        folder = out / name / "input"
        measured = make_fixture(folder, count)
        before = raw_hashes(folder)
        context, stdout = execute_cells(notebook, folder)
        run, figures = context["run"], context["figures"]
        assert run.synthetic and run.reference_complete == (count == 100)
        assert not run.pilot_complete
        assert not any(row["Niveau"] in ("FOUT", "RUNFOUT") for row in run.issues), run.issues
        assert run.last_epoch == count and run.observed_best_epoch == 2
        assert len(run.metrics) == 6 * count
        assert math.isclose(ns.measured_completed_seconds(run), measured)
        assert "Gemeten som over" in ns.timing_markdown(run)
        assert "tf32" in ns.protocol_table(run)["Opgeslagen configuratie"].astype(str).tolist()
        table = ns.checkpoint_table(run)
        best = table.loc[table.Rol == "best_E4M3_test"]
        assert best.Epoch.eq(2).all() and best["Testaccuracy [%]"].tolist() == [90., 80., 70.]
        assert table.loc[table.Rol == "last_epoch", "Epoch"].eq(count).all()
        if count == 100:
            assert table.loc[table.Rol == "epoch100", "Testaccuracy [%]"].tolist() == [80., 80., 70.]
            assert "Alle honderd epochs" in ns.status_markdown(run)
            assert "Alle honderd epochs zijn aanwezig" in ns.timing_markdown(run)
            assert "nieuwe volledige run" not in ns.timing_markdown(run)
        else:
            assert table.loc[table.Rol == "epoch100", "Testaccuracy [%]"].isna().all()
            assert run.metric(100) is None and "Epoch100 niet gemeten" in ns.status_markdown(run)
            assert "nieuwe volledige run" in ns.timing_markdown(run)
        assert set(figures) == {"checkpoint_accuracy", "accuracy_curves", "objective_losses", "timing", "gpu_memory"}
        assert all("SYNTHETISCHE CONTROLE" in fig._suptitle.get_text() for fig in figures.values())
        assert [len(ax.patches) for ax in figures["checkpoint_accuracy"].axes] == ([3, 3] if count == 100 else [0, 3])
        assert "100" in figures["checkpoint_accuracy"].axes[0].get_title()
        assert [len(ax.lines) for ax in figures["accuracy_curves"].axes] == [3, 3]
        assert [len(ax.lines) for ax in figures["objective_losses"].axes] == [1, 1, 1, 3]
        assert "harde CE" in figures["objective_losses"].axes[0].get_title()
        assert "soft CE van E4M3" in figures["objective_losses"].axes[1].get_title()
        assert "soft CE van E2M3" in figures["objective_losses"].axes[2].get_title()
        for ax in figures["accuracy_curves"].axes:
            assert all(max(line.get_xdata()) == count for line in ax.lines)
        assert int(run.test.loc[run.test.weight_bits == 6].sort_values("top1_percent").iloc[-1].epoch) == 4
        assert int(run.test.loc[run.test.weight_bits == 4].sort_values("top1_percent").iloc[-1].epoch) == 5
        if count < 100:
            assert "Gecontroleerd gestopt" in ns.status_markdown(run)
        preview = out / name / "preview"
        preview.mkdir()
        for key, figure in figures.items():
            figure.savefig(preview / (key + ".png"), dpi=130, bbox_inches="tight")
        (out / name / "notebook_stdout.txt").write_text(stdout, encoding="utf-8")
        assert before == raw_hashes(folder) and not list(folder.glob("analysis_*"))
        first = ns.save_figures({"checkpoint_accuracy": figures["checkpoint_accuracy"]}, run)
        second = ns.save_figures({"checkpoint_accuracy": figures["checkpoint_accuracy"]}, run)
        assert first != second and before == raw_hashes(folder)
        manifest = json.loads((first / "analysis_manifest.json").read_text())
        assert manifest["synthetic_fixture_only"] and manifest["analysis_only"]
        assert not manifest["training_started"] and not manifest["checkpoint_loaded"]
        results[name] = {"status": "PASS", "synthetic_fixture_only": True,
                         "last_epoch_in_generated_fixture": count, "selected_shared_epoch": 2,
                         "raw_input_hashes_unchanged": True, "all_notebook_code_cells_executed": True,
                         "model_checkpoint_or_training_imported": False, "preview_directory": str(preview)}
        plt.close("all")
    negative = negative_checks(out, out / "synthetic_stopped5" / "input", out / "synthetic_complete100" / "input")
    # Exercise the runner's intended SAVE_FIGURES=True switch through the notebook.
    folder = out / "synthetic_complete100" / "input"
    before = raw_hashes(folder)
    context, _ = execute_cells(notebook, folder, save=True)
    assert context["analysis_dir"].is_dir() and before == raw_hashes(folder)
    assert len(list(context["analysis_dir"].glob("*.png"))) == 5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == template_hash
    assert not list(out.rglob("*.ipynb")), "No executed notebook may be generated."
    plt.close("all")
    report = {"status": "PASS", "scope": "GENERATED_SYNTHETIC_REPORTS_ONLY_NO_TRAINING",
              "empty_template_outputs_confirmed": True, "template_default_save_figures": False,
              "explicit_save_switch_tested": True, "no_executed_notebook_created": True,
              "tested_file_sha256": {str(file.relative_to(PACKAGE)): hashlib.sha256(file.read_bytes()).hexdigest()
                                     for file in (path, PACKAGE / "notebook_support.py", Path(__file__).resolve())},
              "cases": results, "expected_negative_cases": negative}
    write_json(out / "notebook_validation_report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.output.resolve())
