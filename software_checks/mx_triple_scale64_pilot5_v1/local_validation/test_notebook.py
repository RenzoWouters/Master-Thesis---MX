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
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="triple_scale64_notebook_mpl_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import notebook_support as ns

BANNED_IMPORTS = {"torch", "run_train", "triple_adapter", "triple_support", "resume_core",
                  "svhn_support", "mx_adapter", "mx_format", "mx_e2m1", "mx_e4m3",
                  "joint_adapter", "joint_support", "trunc_adapter", "trunc_support", "posthoc_truncation"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def raw_hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir() if path.is_file()}


def source_audit(batches, training):
    return {"source_triplets": batches, "high_encodings": 6 * batches,
            "exponent_derivations": 6 * batches, "mantissa_derivations": 6 * batches,
            "scale_offsets_plus6": 12 * batches, "mantissa_scale_copies_unchanged": 6 * batches,
            "scale_exponent_offset": 6, "scale_factor": 64, "source_weight_bits": 8,
            "training_order": [8, 6, 4] if training else [4, 6, 8],
            "source_cache_cleared": True, "cache_cleared_before_adam": True,
            "no_direct_lower_quantizer": True}


def make_fixture(folder, count, collapse=False):
    """Fake reports: E4M3 best2 (tie3), E2M3 best4, E2M1 best5."""
    folder.mkdir(parents=True)
    metrics, records, times, updates = [], [], [], []
    correct_by_bit = {8: [6, 9, 9, 8, 8], 6: [7, 8, 9, 10, 8], 4: [5, 7, 8, 7, 9]}
    if collapse:
        correct_by_bit[6] = [1] * 5
        correct_by_bit[4] = [1] * 5
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
        prediction_counts = {}
        for bit in (8, 6, 4):
            correct = correct_by_bit[bit][epoch - 1]
            predictions = [i if i < correct else (i + 1) % 10 for i in range(10)]
            if collapse and bit != 8:
                predictions = [2] * 10
            prediction_counts[str(bit)] = [predictions.count(i) for i in range(10)]
        records.append({**duration, **phase_rows, "optimizer_steps": 2 * epoch,
                        "test_prediction_counts": prediction_counts, "test_label_counts": [1] * 10,
                        "test_optimizer_steps": 0, "batchnorm_counters_correct": True,
                        "lr_used": .001, "lr_next": .001,
                        "gradient_checks": {"optimizer_steps_in_phase": 2, "finite_gradient_checks": 2,
                                            "detached_teacher_checks": 4, "forward_calls_E4M3": 2,
                                            "forward_calls_E2M3": 2, "forward_calls_E2M1": 2,
                                            "no_extra_FP32_teacher": True, "shared_source_checks": source_audit(2, True)},
                        "test_source_checks": source_audit(1, False),
                        "adam_checks": {"all_active_adam_steps_correct": True, "moments_fp32_finite": True,
                                        "unused_bn_has_no_optimizer_state": True},
                        "weight_updates": epoch_updates})
    best, last = records[1], records[-1]
    total = sum(row["total_epoch_seconds"] for row in times)
    config = {"schema": "mx-triple-scale64-pilot5-protocol-v1", "mode": "joint",
              "weight_bits": [4, 6, 8], "weight_formats": {str(k): v for k, v in ns.FORMATS.items()},
              "bn_bank_mapping": {str(k): str(v) for k, v in ns.BANKS.items()},
              "activation_bits": 32, "master_dtype": "float32", "distillation": True, "truncation": True,
              "exponent_truncation": True, "mantissa_truncation": True, "source_weight_bits": 8,
              "shared_scale_compensation": True, "learned_shared_scale": False,
              "scale_exponent_offset": 6, "scale_factor": 64,
              "direct_lower_quantization": False, "exponent_bias_compensation": False,
              "extra_FP32_teacher_forward": False, "dataset": "GENERATED_SYNTHETIC_NO_SVHN",
              "model": "svhnq", "expand": 2, "train_split": "synthetic", "extra_used": False,
              "train_examples": 20, "test_examples": 10, "batch_size": 10, "seed": 20261009,
              "mx_block_size": 32, "mx_scale_format": "E8M0", "mx_scale_policy": "keep E4 source scale; lower scales exact E8M0 source plus6, times64; refreshed each minibatch",
              "pilot_stop_epoch": 5, "planned_epochs": 100}
    summary = {"schema": "mx-triple-scale64-pilot5-completion-v1",
               "shared_scale_compensation": True, "learned_shared_scale": False,
               "scale_exponent_offset": 6, "scale_factor": 64,
               "all_lower_scale_bytes_equal_current_E4_source": False,
               "all_lower_scale_codes_equal_current_E4_source_plus6": True,
               "all_mantissa_scale_bytes_equal_E2M3": True,
               "one_E4_source_per_layer_per_minibatch_verified": True,
               "source_cache_cleared_before_every_optimizer_update": True,
               "completed_epochs": count, "pilot_completed": count == 5,
               "full_100_epoch_training_completed": False, "best_epoch_E4M3_A32": 2, "last_epoch": count,
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
                                  "full_test_forward_rerun": False, "truncation": bit != 8,
                                  "source_scale_bytes_copied": bit == 8, "master_weights_read_to_derive": False,
                                  "scale_exponent_offset": 0 if bit == 8 else 6,
                                  "scale_factor": 1 if bit == 8 else 64,
                                  "shared_scale_compensation": bit != 8, "learned_shared_scale": False,
                                  "is_source_export": bit == 8, "derived_from_saved_source": bit != 8,
                                  "saved_and_reloaded": True, "source_file_sha256": 'a' * 64,
                                  "sha256": 'a' * 64 if bit == 8 else str(bit) * 64} for bit in (4, 6, 8)}
    report = {"schema": "mx-triple-scale64-pilot5-report-v1",
              "status": "PASS" if count == 5 else "STOPPED_AT_EPOCH_BOUNDARY",
              "config": config, "synthetic_fixture_only": True, "warnings": [],
              "source_unchanged": True, "dataset_files_unchanged": True,
              "execution_settings": {"cudnn_conv_fp32_precision": "tf32", "cudnn_benchmark": True,
                                     "deterministic_algorithms": False}, "mx_exports": exports}
    timing = {"available": True, "planned_epochs": 100, "steady_epochs": list(range(2, count + 1)),
              "median_epoch_seconds": 16, "first_epoch_seconds": times[0]["total_epoch_seconds"],
              "estimated_100_epoch_seconds": 1597.5, "observed_range_projection_seconds": [1498.5, 1696.5],
              "range_is_confidence_interval": False, "measured_completed_epoch_seconds": total}
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


def negative_checks(out, source):
    """Reject exactly the reporting mistakes that could change the interpretation."""
    checks = {}
    for case in ("mixed_mode_best", "missing_test_mode", "wrong_soft_loss_label", "false_epoch100_claim", "wrong_teacher_count",
                 "wrong_direct_contract", "scale_compensation_disabled", "missing_predictions", "wrong_prediction_sum",
                 "impossible_constant_prediction_accuracy", "changed_test_labels", "export_source_scale_changed",
                 "wrong_scale_offset", "learned_scale", "shifted_E4_source", "wrong_source_hash",
                 "wrong_train_source_audit", "wrong_test_source_audit", "old_scale_summary") :
        folder = out / case
        folder.mkdir()
        for path in source.iterdir():
            if path.is_file():
                shutil.copyfile(path, folder / path.name)
        if case == "mixed_mode_best":
            data = pd.read_csv(folder / "selected_metrics.csv", float_precision="round_trip")
            # Preserve the low mode's actual epoch4 score, but incorrectly combine
            # it with E4M3's selected epoch2: a tempting per-mode-best mixture.
            metrics = pd.read_csv(folder / "epoch_metrics.csv", float_precision="round_trip")
            wrong = metrics.loc[(metrics.phase == "test") & (metrics.epoch == 4) & (metrics.weight_bits == 6)].iloc[0]
            mask = (data.role == "best_E4M3_test") & (data.weight_bits == 6)
            for key in metrics.columns:
                data.loc[mask, key] = wrong[key]
            data.to_csv(folder / "selected_metrics.csv", index=False)
        elif case == "missing_test_mode":
            path = folder / "epoch_metrics.csv"
            data = pd.read_csv(path, float_precision="round_trip")
            data.loc[~((data.phase == "test") & (data.epoch == 5) & (data.weight_bits == 4))].to_csv(path, index=False)
        elif case == "wrong_soft_loss_label":
            path = folder / "epoch_metrics.csv"
            data = pd.read_csv(path, float_precision="round_trip")
            data.loc[(data.phase == "train") & (data.weight_bits == 4), "objective_type"] = "hard_cross_entropy"
            data.to_csv(path, index=False)
        elif case == "false_epoch100_claim":
            path = folder / "training_summary.json"
            data = json.loads(path.read_text())
            data["full_100_epoch_training_completed"] = True
            write_json(path, data)
        elif case == 'wrong_teacher_count':
            path = folder / "history.json"
            data = json.loads(path.read_text())
            data["epochs"][0]["gradient_checks"]["detached_teacher_checks"] = 2
            write_json(path, data)
        elif case in ('wrong_direct_contract', 'scale_compensation_disabled'):
            path = folder / 'run_report.json'
            data = json.loads(path.read_text())
            data['config']['truncation' if case == 'wrong_direct_contract' else 'shared_scale_compensation'] = False
            write_json(path, data)
        elif case == 'export_source_scale_changed':
            path = folder / 'run_report.json'
            data = json.loads(path.read_text())
            data['mx_exports']['last_epoch']['6']['source_scale_bytes_copied'] = True
            write_json(path, data)
        elif case in ('wrong_scale_offset', 'learned_scale', 'shifted_E4_source', 'wrong_source_hash'):
            path = folder / 'run_report.json'
            data = json.loads(path.read_text())
            if case == 'wrong_scale_offset': data['config']['scale_exponent_offset'] = 5
            elif case == 'learned_scale': data['config']['learned_shared_scale'] = True
            elif case == 'shifted_E4_source': data['mx_exports']['last_epoch']['8']['scale_exponent_offset'] = 6
            else: data['mx_exports']['last_epoch']['4']['source_file_sha256'] = 'b' * 64
            write_json(path, data)
        elif case == 'old_scale_summary':
            path = folder / 'training_summary.json'
            data = json.loads(path.read_text()); data['all_lower_scale_bytes_equal_current_E4_source'] = True
            write_json(path, data)
        else:
            path = folder / 'history.json'
            data = json.loads(path.read_text())
            record = data['epochs'][0]
            if case == 'wrong_train_source_audit':
                record['gradient_checks']['shared_source_checks']['scale_offsets_plus6'] -= 1
            elif case == 'wrong_test_source_audit':
                record['test_source_checks']['mantissa_scale_copies_unchanged'] = 0
            elif case == 'missing_predictions':
                del record['test_prediction_counts']
            elif case == 'wrong_prediction_sum':
                record['test_prediction_counts']['6'][0] += 1
            elif case == 'impossible_constant_prediction_accuracy':
                record['test_prediction_counts']['6'] = [0, 0, 10, 0, 0, 0, 0, 0, 0, 0]
            else:
                record['test_label_counts'] = [2, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            write_json(path, data)
        before = raw_hashes(folder)
        run = ns.load_run(folder)
        errors = [row for row in run.issues if row["Niveau"] == "FOUT"]
        assert errors, case
        assert not run.reference_complete and raw_hashes(folder) == before
        table = ns.checkpoint_table(run)
        assert table.loc[table.Rol == "epoch100", "Testaccuracy [%]"].isna().all()
        if case == "mixed_mode_best":
            assert not run.selection_safe
            assert table.loc[table.Rol == "best_E4M3_test", "Testaccuracy [%]"].isna().all()
            assert table.loc[table.Rol == "last_epoch", "Testaccuracy [%]"].notna().all()
        if case in ("missing_test_mode", "wrong_soft_loss_label"):
            assert not run.metrics_safe and run.test.empty
        checks[case] = {"expected_rejection": True, "messages": [row["Melding"] for row in errors]}
    return checks


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    path = PACKAGE / "MX_triple_scale64_pilot5_resultaten.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    template_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(cell.get("execution_count") is None and not cell.get("outputs")
               for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "SAVE_FIGURES = False" in "".join(notebook["cells"][1]["source"])
    results = {}
    for name, count in (("synthetic_complete5", 5), ("synthetic_stopped3", 3), ("synthetic_constant_class5", 5)):
        folder = out / name / "input"
        collapse = name == "synthetic_constant_class5"
        measured = make_fixture(folder, count, collapse=collapse)
        before = raw_hashes(folder)
        context, stdout = execute_cells(notebook, folder)
        run, figures = context["run"], context["figures"]
        assert run.synthetic and not run.reference_complete
        assert run.pilot_complete == (count == 5)
        assert not any(row["Niveau"] in ("FOUT", "RUNFOUT") for row in run.issues), run.issues
        assert run.last_epoch == count and run.observed_best_epoch == 2
        assert len(run.metrics) == 6 * count
        assert math.isclose(ns.measured_completed_seconds(run), measured)
        assert "Gemeten som over" in ns.timing_markdown(run) and "nieuwe volledige run" in ns.timing_markdown(run)
        assert "tf32" in ns.protocol_table(run)["Opgeslagen configuratie"].astype(str).tolist()
        table = ns.checkpoint_table(run)
        best = table.loc[table.Rol == "best_E4M3_test"]
        assert best.Epoch.eq(2).all() and best["Testaccuracy [%]"].tolist() == ([90., 10., 10.] if collapse else [90., 80., 70.])
        assert table.loc[table.Rol == "last_epoch", "Epoch"].eq(count).all()
        assert table.loc[table.Rol == "epoch100", "Testaccuracy [%]"].isna().all()
        assert run.metric(100) is None and "Epoch100 niet gemeten" in ns.status_markdown(run)
        assert set(figures) == {"checkpoint_accuracy", "accuracy_curves", "objective_losses", "timing", "gpu_memory", "prediction_diversity"}
        assert all("SYNTHETISCHE CONTROLE" in fig._suptitle.get_text() for fig in figures.values())
        assert [len(ax.patches) for ax in figures["checkpoint_accuracy"].axes] == [3, 3]
        assert [len(ax.lines) for ax in figures["accuracy_curves"].axes] == [3, 3]
        assert [len(ax.lines) for ax in figures["objective_losses"].axes] == [1, 1, 1, 3]
        assert "harde CE" in figures["objective_losses"].axes[0].get_title()
        assert "soft CE van E4M3" in figures["objective_losses"].axes[1].get_title()
        assert "soft CE van E2M3" in figures["objective_losses"].axes[2].get_title()
        for ax in figures["accuracy_curves"].axes:
            assert all(max(line.get_xdata()) == count for line in ax.lines)
        if count == 5 and not collapse:
            assert int(run.test.loc[run.test.weight_bits == 6].sort_values("top1_percent").iloc[-1].epoch) == 4
            assert int(run.test.loc[run.test.weight_bits == 4].sort_values("top1_percent").iloc[-1].epoch) == 5
        elif count < 5:
            assert "Gecontroleerd gestopt" in ns.status_markdown(run)
        assert run.prediction_safe and len(run.predictions) == 3 * count
        if collapse:
            low = run.predictions.loc[run.predictions.weight_bits != 8]
            assert low['Constante voorspelling'].all() and low['Meest voorspelde cijfer'].eq(2).all()
            assert low['Aandeel dominant cijfer [%]'].eq(100).all()
            assert any(row['Niveau'] == 'UITKOMST' for row in run.issues)
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
    negative = negative_checks(out, out / "synthetic_complete5" / "input")
    # Regress the observed one-ULP pandas parser problem without widening gates.
    large = out / 'synthetic_large_CE_round_trip'
    large.mkdir()
    for original in (out / 'synthetic_complete5/input').iterdir():
        if original.is_file(): shutil.copyfile(original, large / original.name)
    exact_ce = 16073888499.481255
    path_metrics = large / 'epoch_metrics.csv'
    data = pd.read_csv(path_metrics, float_precision='round_trip')
    mask = (data.epoch == 1) & (data.phase == 'test') & (data.weight_bits == 4)
    data.loc[mask, 'objective_loss'] = exact_ce
    data.to_csv(path_metrics, index=False)
    history = json.loads((large / 'history.json').read_text())
    for row in history['epochs'][0]['test']:
        if row['weight_bits'] == 4: row['objective_loss'] = exact_ce
    write_json(large / 'history.json', history)
    before_large = raw_hashes(large)
    checked_large = ns.load_run(large)
    assert checked_large.metrics.loc[mask, 'objective_loss'].iloc[0] == exact_ce
    assert not any(x['Niveau'] == 'FOUT' for x in checked_large.issues), checked_large.issues
    assert raw_hashes(large) == before_large
    results['synthetic_large_CE_round_trip'] = {'status': 'PASS', 'exact_decimal_round_trip_preserved': True,
                                              'raw_input_hashes_unchanged': True}
    check_folder = out / 'synthetic_check_only'
    check_folder.mkdir()
    original_report = json.loads((out / 'synthetic_complete5/input/run_report.json').read_text())
    original_report.update(command='check', synthetic_checks_only=True, mx_exports={})
    write_json(check_folder / 'run_report.json', original_report)
    shutil.copyfile(PACKAGE / 'notebook_support.py', check_folder / 'notebook_support.py')
    before_check = raw_hashes(check_folder)
    check_context, _ = execute_cells(notebook, check_folder)
    checked = check_context['run']
    assert checked.test.empty and checked.predictions.empty
    assert not checked.pilot_complete and not checked.reference_complete
    assert ns.checkpoint_table(checked)['Testaccuracy [%]'].isna().all()
    assert before_check == raw_hashes(check_folder)
    results['synthetic_check_only'] = {'status': 'PASS', 'check_only_not_presented_as_training': True,
                                       'raw_input_hashes_unchanged': True}
    # Exercise the runner's intended SAVE_FIGURES=True switch through the notebook.
    folder = out / "synthetic_complete5" / "input"
    before = raw_hashes(folder)
    context, _ = execute_cells(notebook, folder, save=True)
    assert context["analysis_dir"].is_dir() and before == raw_hashes(folder)
    assert len(list(context["analysis_dir"].glob("*.png"))) == 6
    assert hashlib.sha256(path.read_bytes()).hexdigest() == template_hash
    assert not list(out.rglob("*.ipynb")), "No executed notebook may be generated."
    plt.close("all")
    report = {"status": "PASS", "scope": "GENERATED_SYNTHETIC_REPORTS_ONLY_NO_TRAINING",
              "empty_template_outputs_confirmed": True, "template_default_save_figures": False,
              "explicit_save_switch_tested": True, "no_executed_notebook_created": True,
              "all_ten_code_cells_executed": True, "all_six_figures_rendered": True,
              "constant_class_is_valid_research_outcome": True, "epoch100_never_claimed": True,
              "cases": results, "expected_negative_cases": negative,
              "tested_file_sha256": {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                                     for name in ['notebook_support.py',
                                                  'MX_triple_scale64_pilot5_resultaten.ipynb',
                                                  'local_validation/test_notebook.py']}}
    write_json(out / "notebook_validation_report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.output.resolve())
