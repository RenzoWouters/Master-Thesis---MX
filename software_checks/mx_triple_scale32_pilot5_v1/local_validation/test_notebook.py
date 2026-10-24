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
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="triple_scale32_pilot5_notebook_mpl_"))
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
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()
            and not any(part.startswith('analysis_') for part in path.relative_to(folder).parts)}


def source_audit(batches, training):
    return {"source_triplets": batches, "high_encodings": 6 * batches,
            "exponent_derivations": 6 * batches, "mantissa_derivations": 6 * batches,
            "scale_offsets_plus5": 12 * batches, "mantissa_scale_copies_unchanged": 6 * batches,
            "scale_exponent_offset": 5, "scale_factor": 32, "source_weight_bits": 8,
            "training_order": [8, 6, 4] if training else [4, 6, 8],
            "source_cache_cleared": True, "cache_cleared_before_adam": True,
            "no_direct_lower_quantizer": True}


def make_fixture(folder, count, collapse=False):
    """Synthetic pilot: E4 best2 (tie3), E2M3 best4, E2M1 best5."""
    folder.mkdir(parents=True)
    metrics, records, times, updates = [], [], [], []
    correct_by_bit = {8: [600, 900, 900, 800, 800], 6: [700, 800, 900, 1000, 800], 4: [500, 700, 800, 700, 900]}
    correct_by_bit[6][3] = 930
    if collapse:
        correct_by_bit[6] = [100] * 5
        correct_by_bit[4] = [100] * 5
    for epoch in range(1, count + 1):
        phase_rows = {}
        for phase in ("train", "test"):
            phase_rows[phase] = []
            for bit in (4, 6, 8):
                n = 20 if phase == "train" else 1000
                correct = min(20, 14 + epoch) if phase == "train" else correct_by_bit[bit][epoch - 1]
                soft = phase == "train" and bit != 8
                row = {"epoch": epoch, "phase": phase, "mode": "joint", "weight_bits": bit,
                       "weight_format": ns.FORMATS[bit], "activation_bits": 32, "bn_bank_key": ns.BANKS[bit],
                       "n": n, "correct1": correct, "correct5": n, "top1_percent": 100 * correct / n,
                       "count_top1_percent": 100 * correct / n, "top5_percent": 100.,
                       "lr": .001 * (.1 ** sum(epoch > k for k in (50, 75, 90))),
                       "objective_type": "recursive_soft_cross_entropy" if soft else "hard_cross_entropy",
                       "teacher_weight_format": ns.FORMATS[8 if bit == 6 else 6] if soft else None,
                       "objective_loss": (1 + (8 - bit) / 10) / (1 + epoch) if phase == "train" else .2 + (1000 - correct) / 1000}
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
            predictions = [i % 10 if i < correct else (i + 1) % 10 for i in range(1000)]
            if collapse and bit != 8:
                predictions = [2] * 1000
            prediction_counts[str(bit)] = [predictions.count(i) for i in range(10)]
        records.append({**duration, **phase_rows, "optimizer_steps": 2 * epoch,
                        "test_prediction_counts": prediction_counts, "test_label_counts": [100] * 10,
                        "test_optimizer_steps": 0, "batchnorm_counters_correct": True,
                        "lr_used": .001 * (.1 ** sum(epoch > k for k in (50, 75, 90))),
                        "lr_next": .001 * (.1 ** sum(epoch >= k for k in (50, 75, 90))),
                        "gradient_checks": {"optimizer_steps_in_phase": 2, "finite_gradient_checks": 2,
                                            "detached_teacher_checks": 4, "forward_calls_E4M3": 2,
                                            "forward_calls_E2M3": 2, "forward_calls_E2M1": 2,
                                            "no_extra_FP32_teacher": True, "shared_source_checks": source_audit(2, True)},
                        "test_source_checks": source_audit(100, False),
                        "adam_checks": {"all_active_adam_steps_correct": True, "moments_fp32_finite": True,
                                        "unused_bn_has_no_optimizer_state": True},
                        "weight_updates": epoch_updates})
    best_epoch = max(range(1, count + 1), key=lambda e: correct_by_bit[8][e - 1])
    best, last = records[best_epoch - 1], records[-1]
    total = sum(row["total_epoch_seconds"] for row in times)
    config = {"schema": "mx-triple-scale32-pilot5-protocol-v1", "mode": "joint",
              "weight_bits": [4, 6, 8], "weight_formats": {str(k): v for k, v in ns.FORMATS.items()},
              "bn_bank_mapping": {str(k): str(v) for k, v in ns.BANKS.items()},
              "activation_bits": 32, "master_dtype": "float32", "distillation": True, "truncation": True,
              "exponent_truncation": True, "mantissa_truncation": True, "source_weight_bits": 8,
              "shared_scale_compensation": True, "learned_shared_scale": False,
              "scale_exponent_offset": 5, "scale_factor": 32,
              "direct_lower_quantization": False, "exponent_bias_compensation": False,
              "extra_FP32_teacher_forward": False, "dataset": "GENERATED_SYNTHETIC_NO_SVHN",
              "model": "svhnq", "expand": 2, "train_split": "synthetic", "extra_used": False,
              "train_examples": 20, "test_examples": 1000, "batch_size": 10, "seed": 20261009,
              "mx_block_size": 32, "mx_scale_format": "E8M0", "mx_scale_policy": "keep E4 source scale; lower scales exact E8M0 source plus5, times32; refreshed each minibatch",
              "pilot_stop_epoch": 5, "planned_epochs": 100}
    summary = {"schema": "mx-triple-scale32-pilot5-completion-v1",
               "shared_scale_compensation": True, "learned_shared_scale": False,
               "scale_exponent_offset": 5, "scale_factor": 32,
               "all_lower_scale_bytes_equal_current_E4_source": False,
               "all_lower_scale_codes_equal_current_E4_source_plus5": True,
               "all_mantissa_scale_bytes_equal_E2M3": True,
               "one_E4_source_per_layer_per_minibatch_verified": True,
               "source_cache_cleared_before_every_optimizer_update": True,
               "pilot_stop_epoch": 5, "completed_epochs": count, "pilot_completed": count == 5,
               "full_100_epoch_training_completed": False, "best_epoch_E4M3_A32": best_epoch, "last_epoch": count,
               "optimizer_steps": 2 * count, "training_forward_passes": 6 * count,
               "gradient_checks_before_every_update": True, "fp32_masters_and_moments_verified": True,
               "measured_completed_epoch_seconds": total, "synthetic_fixture_only": True}
    selection = {"policy": "first strict maximum of E4M3/A32 official-test top1",
                 "selection_mode": ns.SELECTION_FORMAT + "/A32", "all_modes_from_same_selected_checkpoint": True,
                 "selected_on_official_test": True, "independent_test_estimate": False,
                 "same_checkpoint": False}
    for role, record in (("best", best), ("last", last)):
        selection[role] = {"epoch": record["epoch"], "file": role + "_checkpoint.pt",
                           "sha256": hashlib.sha256(str(record["epoch"]).encode()).hexdigest(),
                           "original_checkpoint": "SYNTHETIC_METADATA_ONLY_NO_CHECKPOINT_FILE",
                           "selection_score_E4M3": next(row["top1_percent"] for row in record["test"] if row["weight_bits"] == 8)}
    exports = {}
    for role, record in (("best_E4M3_test", best), ("last_epoch", last)):
        exports[role] = {str(bit): {"epoch": record["epoch"], "weight_bits": bit,
                                  "file": f"SYNTHETIC_METADATA_{bit}_{role}.pt", "bytes": 123,
                                  "decoded_weights_exact": True, "six_fp32_masters_absent": True,
                                  "full_test_forward_rerun": False, "truncation": bit != 8,
                                  "source_scale_bytes_copied": bit == 8, "master_weights_read_to_derive": False,
                                  "scale_exponent_offset": 0 if bit == 8 else 5,
                                  "scale_factor": 1 if bit == 8 else 32,
                                  "shared_scale_compensation": bit != 8, "learned_shared_scale": False,
                                  "is_source_export": bit == 8, "derived_from_saved_source": bit != 8,
                                  "saved_and_reloaded": True, "source_file_sha256": 'a' * 64,
                                  "sha256": 'a' * 64 if bit == 8 else str(bit) * 64} for bit in (4, 6, 8)}
    report = {"schema": "mx-triple-scale32-pilot5-report-v1",
              "status": "PASS" if count == 5 else "STOPPED_AT_EPOCH_BOUNDARY",
              "pilot_completed": count == 5,
              "config": config, "synthetic_fixture_only": True, "warnings": [],
              "source_unchanged": True, "dataset_files_unchanged": True,
              "execution_settings": {"cudnn_conv_fp32_precision": "tf32", "cudnn_benchmark": True,
                                     "deterministic_algorithms": False}, "mx_exports": exports}
    timing = {"available": True, "planned_epochs": 100, "steady_epochs": list(range(2, count + 1)),
              "median_epoch_seconds": 16, "first_epoch_seconds": times[0]["total_epoch_seconds"],
              "estimated_100_epoch_seconds": 1597.5, "observed_range_projection_seconds": [1498.5, 1696.5],
              "range_is_confidence_interval": False, "measured_completed_epoch_seconds": total,
              "pilot_completed": count == 5, "full_100_epoch_training_completed": False,
              "measured_100_epoch_seconds": None}
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
    shutil.copytree(PACKAGE / 'scale64_reference', folder / 'scale64_reference')
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


def negative_checks(out, source, stopped):
    """Reject exactly the reporting mistakes that could change the interpretation."""
    checks = {}
    for case in ("mixed_mode_best", "missing_test_mode", "wrong_soft_loss_label", "false_epoch100_claim", "wrong_teacher_count",
                 "wrong_direct_contract", "scale_compensation_disabled", "missing_predictions", "wrong_prediction_sum",
                 "impossible_constant_prediction_accuracy", "changed_test_labels", "export_source_scale_changed",
                 "wrong_scale_offset", "learned_scale", "shifted_E4_source", "wrong_source_hash",
                 "wrong_train_source_audit", "wrong_test_source_audit", "old_scale_summary") :
        folder = out / case
        folder.mkdir()
        case_source = stopped if case == "false_epoch100_claim" else source
        for path in case_source.iterdir():
            if path.is_file():
                shutil.copyfile(path, folder / path.name)
        if case == "mixed_mode_best":
            data = pd.read_csv(folder / "selected_metrics.csv", float_precision="round_trip")
            # Mix E2M3 epoch4 with E4M3 epoch2: explicitly reject cherry-picking.
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
            if case == 'wrong_scale_offset': data['config']['scale_exponent_offset'] = 6
            elif case == 'learned_scale': data['config']['learned_shared_scale'] = True
            elif case == 'shifted_E4_source': data['mx_exports']['last_epoch']['8']['scale_exponent_offset'] = 5
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
                record['gradient_checks']['shared_source_checks']['scale_offsets_plus5'] -= 1
            elif case == 'wrong_test_source_audit':
                record['test_source_checks']['mantissa_scale_copies_unchanged'] = 0
            elif case == 'missing_predictions':
                del record['test_prediction_counts']
            elif case == 'wrong_prediction_sum':
                record['test_prediction_counts']['6'][0] += 1
            elif case == 'impossible_constant_prediction_accuracy':
                record['test_prediction_counts']['6'] = [0, 0, 1000, 0, 0, 0, 0, 0, 0, 0]
            else:
                record['test_label_counts'] = [101, 99, 100, 100, 100, 100, 100, 100, 100, 100]
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
    notebook_path = PACKAGE / 'MX_triple_scale32_pilot5_resultaten.ipynb'
    notebook = json.loads(notebook_path.read_text())
    template_hash = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    assert all(cell.get('execution_count') is None and not cell.get('outputs')
               for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert sum(c['cell_type'] == 'code' for c in notebook['cells']) == 11
    assert notebook['metadata']['pilot_stop_epoch'] == 5
    assert notebook['metadata']['scale_exponent_offset'] == 5
    assert notebook['metadata']['scale_factor'] == 32
    results = {}
    expected_figures = {'checkpoint_accuracy', 'accuracy_curves', 'objective_losses', 'timing',
                        'gpu_memory', 'prediction_diversity', 'scale_comparison_curves',
                        'scale_comparison_checkpoints'}
    for name, count, collapse in (('synthetic_complete5', 5, False),
                                  ('synthetic_stopped3', 3, False),
                                  ('synthetic_constant_class5', 5, True)):
        folder = out / name / 'input'
        measured = make_fixture(folder, count, collapse)
        before = raw_hashes(folder)
        context, stdout = execute_cells(notebook, folder)
        run, figures, reference = context['run'], context['figures'], context['reference']
        assert run.synthetic and not run.reference_complete
        assert run.pilot_complete == (count == 5)
        assert not any(x['Niveau'] in ('FOUT', 'RUNFOUT') for x in run.issues), run.issues
        assert run.observed_best_epoch == 2 and run.last_epoch == count
        assert len(run.metrics) == 6 * count
        assert math.isclose(ns.measured_completed_seconds(run), measured)
        assert 'Gemeten som over' in ns.timing_markdown(run)
        table = ns.checkpoint_table(run)
        assert len(table) == 6 and set(table.Rol) == {'last_epoch', 'best_E4M3_test'}
        best = table.loc[table.Rol == 'best_E4M3_test']
        assert best.Epoch.eq(2).all()
        assert best['Testaccuracy [%]'].tolist() == ([90., 10., 10.] if collapse else [90., 80., 70.])
        assert set(figures) == expected_figures
        assert all('SYNTHETISCHE CONTROLE' in fig._suptitle.get_text() for fig in figures.values())
        assert [len(ax.patches) for ax in figures['checkpoint_accuracy'].axes] == [3, 3]
        assert [len(ax.lines) for ax in figures['accuracy_curves'].axes] == [3, 3]
        assert [len(ax.lines) for ax in figures['objective_losses'].axes] == [1, 1, 1, 3]
        assert 'soft CE van E4M3' in figures['objective_losses'].axes[1].get_title()
        assert 'soft CE van E2M3' in figures['objective_losses'].axes[2].get_title()
        assert all(max(line.get_xdata()) == count for ax in figures['accuracy_curves'].axes for line in ax.lines)
        assert reference.safe and not reference.compatible
        assert reference.best_epoch(5) == 5 and reference.test.epoch.max() == 5
        assert ns.comparison_table(run, reference).empty
        assert any('synthetische' in issue for issue in reference.issues)
        assert not any(ax.lines for ax in figures['scale_comparison_curves'].axes)
        if count < 5:
            assert 'Gecontroleerd gestopt' in ns.status_markdown(run)
        if collapse:
            low = run.predictions.loc[run.predictions.weight_bits != 8]
            assert low['Constante voorspelling'].all()
            assert any(x['Niveau'] == 'UITKOMST' for x in run.issues)
        preview = out / name / 'preview'
        preview.mkdir()
        for key, fig in figures.items():
            fig.savefig(preview / (key + '.png'), dpi=130, bbox_inches='tight')
        (out / name / 'notebook_stdout.txt').write_text(stdout)
        assert raw_hashes(folder) == before and not list(folder.glob('analysis_*'))
        results[name] = {'status': 'PASS', 'synthetic_fixture_only': True,
                         'completed_epochs': count, 'common_selected_E4_epoch': 2,
                         'all_11_code_cells_executed': True, 'all_8_figures_rendered': True,
                         'raw_inputs_unchanged': True, 'real_reference_comparison_blocked_for_synthetic_run': True}
        plt.close('all')

    full = out / 'synthetic_complete5/input'
    negative = negative_checks(out, full, out / 'synthetic_stopped3/input')
    # Regression: parser round-trips the exact serialized binary float, without
    # relaxing consistency checks when an unusual loss is numerically large.
    large = out / 'synthetic_large_CE_round_trip'
    shutil.copytree(full, large)
    exact_ce = 16073888499.481255
    metrics = pd.read_csv(large / 'epoch_metrics.csv', float_precision='round_trip')
    mask = (metrics.epoch == 1) & (metrics.phase == 'test') & (metrics.weight_bits == 4)
    metrics.loc[mask, 'objective_loss'] = exact_ce
    metrics.to_csv(large / 'epoch_metrics.csv', index=False)
    history = json.loads((large / 'history.json').read_text())
    for row in history['epochs'][0]['test']:
        if row['weight_bits'] == 4:
            row['objective_loss'] = exact_ce
    write_json(large / 'history.json', history)
    checked = ns.load_run(large)
    assert checked.metrics.loc[mask, 'objective_loss'].iloc[0] == exact_ce
    assert not any(x['Niveau'] == 'FOUT' for x in checked.issues), checked.issues
    results['large_CE_round_trip'] = {'status': 'PASS'}

    # Check-only folder never turns into a training result.
    check = out / 'synthetic_check_only'
    check.mkdir()
    report = json.loads((full / 'run_report.json').read_text())
    report.update(command='check', synthetic_checks_only=True, mx_exports={})
    write_json(check / 'run_report.json', report)
    shutil.copyfile(PACKAGE / 'notebook_support.py', check / 'notebook_support.py')
    shutil.copytree(PACKAGE / 'scale64_reference', check / 'scale64_reference')
    before = raw_hashes(check)
    context, _ = execute_cells(notebook, check)
    assert context['run'].test.empty and not context['run'].pilot_complete
    assert ns.checkpoint_table(context['run'])['Testaccuracy [%]'].isna().all()
    assert raw_hashes(check) == before
    results['check_only'] = {'status': 'PASS', 'no_training_metrics_invented': True}

    # Two deliberately synthetic experiments exercise comparison plots with
    # differing E4-best epochs, ties, and lower-mode peaks. No research claims.
    run = ns.load_run(full)
    baseline = run.metrics.copy()
    for phase in ('train', 'test'):
        values = [500, 600, 700, 950, 950] if phase == 'test' else [15, 16, 17, 18, 18]
        for epoch, correct in enumerate(values, 1):
            at = (baseline.epoch == epoch) & (baseline.phase == phase) & (baseline.weight_bits == 8)
            baseline.loc[at, 'correct1'] = correct
            baseline.loc[at, 'top1_percent'] = 100 * correct / (1000 if phase == 'test' else 20)
    ref = ns.Scale64Reference(out / 'synthetic_comparison', metrics=baseline,
                             safe=True, compatible=True, synthetic=True)
    table = ns.comparison_table(run, ref)
    best = table.loc[table.Controlepunt.str.startswith('Beste')]
    assert best['Epoch ×32'].eq(2).all() and best['Epoch ×64'].eq(4).all()
    assert best['×32 accuracy [%]'].tolist() == [90., 80., 70.]
    assert best['×64 accuracy [%]'].tolist() == [95., 93., 70.]
    assert best['Verschil ×32 − ×64 [pp]'].tolist() == [-5., -13., 0.]
    fixtures = {'curves': ns.plot_scale_comparison_curves(run, ref),
                'checkpoints': ns.plot_scale_comparison_checkpoints(run, ref)}
    assert [len(a.lines) for a in fixtures['curves'].axes] == [2, 2, 2]
    assert [len(a.patches) for a in fixtures['checkpoints'].axes] == [6, 6]
    preview = out / 'synthetic_comparison_preview'
    preview.mkdir()
    for key, fig in fixtures.items():
        assert 'SYNTHETISCHE CONTROLE' in fig._suptitle.get_text()
        fig.savefig(preview / (key + '.png'), dpi=140, bbox_inches='tight')
    short = ns.load_run(out / 'synthetic_stopped3/input')
    short_table = ns.comparison_table(short, ref)
    assert short_table.loc[short_table.Controlepunt == 'Epoch 5', '×32 accuracy [%]'].isna().all()
    short_best = short_table.loc[short_table.Controlepunt.str.startswith('Beste')]
    assert short_best['Epoch ×64'].eq(3).all()
    short_fig = ns.plot_scale_comparison_curves(short, ref)
    assert all(max(line.get_xdata()) == 3 for ax in short_fig.axes for line in ax.lines)
    # A future reference peak must not leak through the bounded comparison.
    future = baseline.loc[baseline.epoch == 5].copy()
    future['epoch'] = 100
    future['top1_percent'] = 100.
    ref.metrics = pd.concat([baseline, future], ignore_index=True)
    assert ref.best_epoch(5) == 4
    assert ns.comparison_table(run, ref).equals(table)
    results['synthetic_comparison'] = {'status': 'PASS', 'different_best_epochs_and_tie_checked': True,
                                      'common_checkpoint_per_precision_group': True,
                                      'future_epoch100_ignored': True, 'partial_range_bounded': True}

    # Validate the real pinned reference reader and its actual intended guard.
    control = ns.load_run(full)
    original = ns.load_scale64_reference(control)
    assert original.safe and original.best_epoch(5) == 5
    control.report = dict(original.report)
    control.report['config'] = dict(original.report['config'], scale_factor=32, scale_exponent_offset=5)
    control.summary = dict(control.summary, synthetic_fixture_only=False)
    compatible = ns.load_scale64_reference(control)
    assert compatible.safe and compatible.compatible, compatible.issues
    control.report['config']['batch_size'] = 1
    incompatible = ns.load_scale64_reference(control)
    assert incompatible.safe and not incompatible.compatible
    assert any('batch_size' in x for x in incompatible.issues)
    tampered = out / 'tampered_reference'
    shutil.copytree(PACKAGE / 'scale64_reference', tampered)
    with (tampered / 'epoch_metrics.csv').open('a') as handle:
        handle.write('\n')
    bad = ns.load_scale64_reference(run, tampered)
    assert not bad.safe and any('hash' in x for x in bad.issues)
    results['real_reference_guards'] = {'status': 'PASS', 'pinned_source_hash_checked': True,
                                       'first_five_only_best_epoch5_not_global42': True,
                                       'configuration_mismatch_blocks_comparison': True,
                                       'changed_reference_bytes_rejected': True}

    before = raw_hashes(full)
    context, _ = execute_cells(notebook, full, save=True)
    assert len(list(context['analysis_dir'].glob('*.png'))) == 8
    assert raw_hashes(full) == before
    manifest = json.loads((context['analysis_dir'] / 'analysis_manifest.json').read_text())
    assert manifest['analysis_only'] and not manifest['training_started'] and not manifest['checkpoint_loaded']
    assert 'run_report.json' in manifest['input_sha256']
    assert 'scale64_reference/run_report.json' in manifest['input_sha256']
    assert hashlib.sha256(notebook_path.read_bytes()).hexdigest() == template_hash
    assert not list(out.rglob('*.ipynb')), 'Never save an executed notebook.'
    plt.close('all')
    report = {'status': 'PASS', 'scope': 'SYNTHETIC_REPORTS_AND_READ_ONLY_PINNED_REFERENCE',
              'all_11_notebook_code_cells_executed': True, 'eight_figure_types_rendered': True,
              'blank_deliverable_notebook': True, 'no_executed_notebook_created': True,
              'no_training_or_model_imports': True, 'existing_data_unchanged': True,
              'cases': results, 'expected_negative_cases': negative,
              'tested_file_sha256': {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                                    for name in ['notebook_support.py', 'MX_triple_scale32_pilot5_resultaten.ipynb',
                                                 'local_validation/test_notebook.py']}}
    write_json(out / 'notebook_validation_report.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output.resolve())
