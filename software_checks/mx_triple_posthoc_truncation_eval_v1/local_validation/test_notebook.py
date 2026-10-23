"""Bounded notebook validation with generated reports, never model execution.

Fixtures deliberately include a lower-bit direct mode that outperforms E4M3,
a collapsed exponent route, and tied logits with a non-argmax-index prediction.
All output is labelled synthetic and kept outside the distribution package.
"""
from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mx_posthoc_notebook_mpl_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import notebook_support as ns

NOTEBOOK = "MX_triple_posthoc_truncatie_resultaten.ipynb"
LAYERS = ("layers.3", "layers.6", "layers.10", "layers.13", "layers.16", "layers.20")
PAIRS = (("direct8", "direct6"), ("direct8", "direct4"), ("direct6", "direct4"),
         ("direct4", "round64"), ("direct4", "mantissa64"), ("round64", "mantissa64"),
         ("direct6", "round86"), ("direct6", "exponent86"), ("round86", "exponent86"),
         ("direct4", "combined864"), ("mantissa64", "combined864"))
BANNED_IMPORTS = {"torch", "torchvision", "run_eval", "evaluation", "joint_adapter", "joint_support",
                  "truncation", "resume_core", "svhn_support", "mx_adapter", "mx_format", "mx_e2m1", "mx_e4m3"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_hashes(folder):
    return {p.name: digest(p) for p in folder.iterdir() if p.is_file()}


def make_fixture(folder):
    folder.mkdir(parents=True)
    report = {"schema": "mx-triple-posthoc-truncation-eval-report-v1", "status": "PASS", "command": "evaluate",
              "synthetic_fixture_only": True, "optimizer_steps": 0, "batchnorm_recalibration": False,
              "warnings": [], "limitations": ["SYNTHETIC REPORTS ONLY: no SVHN, no model or training."],
              "session_wall_seconds": 12.3, "roles": {}, "reference": {"checkpoint_selection": {}}}
    metrics, pairs, codes, weights, historical = [], [], [], [], []
    labels = np.arange(10, dtype=np.int64)
    indices = np.arange(10, dtype=np.int64)
    for role, epoch, ckpt in (("last_epoch", 100, "c" * 64), ("best_E4M3_test", 54, "b" * 64)):
        arrays = {"labels": labels, "indices": indices}
        correct = {"direct8": 8, "direct6": 9, "direct4": 9, "round64": 8,
                   "mantissa64": 7, "round86": 2, "exponent86": 1, "combined864": 1}
        predictions = {}
        for route in ns.PATHS:
            pred = labels.copy()
            wrong = np.arange(correct[route], len(labels))
            pred[wrong] = (pred[wrong] + 1) % 10
            logits = np.full((10, 10), -1, dtype=np.float32)
            logits[np.arange(10), pred] = 2
            if route == "exponent86":
                logits[:] = 0
                pred[:] = 8  # Legal topk tie; NumPy argmax chooses0 instead.
            ce = (np.log(np.exp(logits.astype(np.float64)).sum(axis=1)) - logits[np.arange(10), labels]).astype(np.float32)
            predictions[route] = pred
            arrays[route + "__predictions"] = pred
            arrays[route + "__logits"] = logits
            arrays[route + "__per_image_ce"] = ce
            ncorrect = int((pred == labels).sum())
            metrics.append({"role": role, "epoch": epoch, "path": route, "weight_bits": ns.BITS[route],
                            "activation_bits": 32, "bn_bank_key": ns.BANKS[ns.BITS[route]], "n": 10,
                            "correct1": ncorrect, "correct5": 10, "top1_percent": 10 * ncorrect, "top5_percent": 100.,
                            "hard_cross_entropy": float(ce.astype(np.float64).mean()), "optimizer_steps": 0,
                            "batchnorm_recalibrated": False})
            if route.startswith("direct"):
                historical.append({"role": role, "epoch": epoch, "path": route,
                                   "historical_top1_percent": 10 * ncorrect, "new_top1_percent": 10 * ncorrect,
                                   "difference_pp": 0., "synthetic_fixture_only": True})
        for ref, other in PAIRS:
            rp, op = predictions[ref], predictions[other]
            rc, oc = rp == labels, op == labels
            up, down = int((~rc & oc).sum()), int((rc & ~oc).sum())
            error = arrays[other + "__logits"].astype(np.float64) - arrays[ref + "__logits"].astype(np.float64)
            pairs.append({"role": role, "epoch": epoch, "reference_path": ref, "other_path": other,
                          "comparison": other + "_minus_" + ref, "n": 10,
                          "reference_correct1": int(rc.sum()), "other_correct1": int(oc.sum()),
                          "accuracy_difference_pp": 10 * (up - down), "prediction_changes": int((rp != op).sum()),
                          "wrong_to_correct": up, "correct_to_wrong": down,
                          "wrong_to_other_wrong": int((~rc & ~oc & (rp != op)).sum()),
                          "both_correct": int((rc & oc).sum()), "both_wrong": int((~rc & ~oc).sum()),
                          "net_correct_change": up - down, "mean_abs_logit_difference": float(np.abs(error).mean()),
                          "max_abs_logit_difference": float(np.abs(error).max())})
        for i, layer in enumerate(LAYERS):
            base = .01 / (i + 1)
            for route in ns.PATHS:
                ratio = {"round64": .98, "mantissa64": .90, "round86": .03, "exponent86": .02, "combined864": .015}.get(route, 1.)
                codes.append({"role": role, "epoch": epoch, "layer": layer, "path": route,
                              "source_bits": 8 if route in ("round86", "exponent86", "combined864", "direct8") else 6,
                              "target_bits": ns.BITS[route], "bn_bank_key": ns.BANKS[ns.BITS[route]],
                              "n_weights": 20, "blocks": 1, "padding_slots": 12, "mean_abs": base * ratio,
                              "rms": base * ratio * 1.2, "direct_baseline": ns.BASELINES[route],
                              "mean_abs_ratio_to_direct": ratio, "rms_ratio_to_direct": ratio,
                              "zero_count": 2 if route == "combined864" else 0, "local_subnormal_count": 1,
                              "scale_exp_min": -12 if route in ("direct8", "round86", "exponent86", "combined864") else -6,
                              "scale_exp_max": -12 if route in ("direct8", "round86", "exponent86", "combined864") else -6,
                              "scale_anchor_path": "direct8" if route in ("direct8", "round86", "exponent86", "combined864") else "direct6",
                              "scales_equal_source": True, "nonzero_dropped_exponent_suffix_count": 10 if route in ("exponent86", "combined864") else 0,
                              "nonzero_dropped_mantissa_suffix_count": 10 if route in ("mantissa64", "combined864") else 0})
            for route in ns.DERIVED:
                weights.append({"role": role, "epoch": epoch, "layer": layer, "reference_path": ns.BASELINES[route],
                                "other_path": route, "comparison": route + "_minus_" + ns.BASELINES[route],
                                "n_weights": 20, "code_differences": 10, "numeric_differences": 10,
                                "scale_code_differences": int(route in ("round86", "exponent86", "combined864")),
                                "mae": base * (.1 if route in ("mantissa64", "round64") else .98),
                                "rmse": base, "max_abs_difference": base * 2})
        npz = folder / (role + "_per_image_outputs.npz")
        np.savez_compressed(npz, **arrays)
        report["roles"][role] = {"epoch": epoch, "checkpoint_sha256": ckpt, "n_test": 10,
                                  "per_image_file": npz.name, "per_image_sha256": digest(npz),
                                  "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                                  "all_model_and_bn_states_unchanged": True, "all_models_no_grad": True,
                                  "six_FP32_master_parameters_absent_all_paths": True, "optimizer_steps": 0}
        report["reference"]["checkpoint_selection"]["last" if role == "last_epoch" else "best"] = {"epoch": epoch, "sha256": ckpt}
    write_json(folder / "run_report.json", report)
    for name, rows in (("metrics.csv", metrics), ("paired_metrics.csv", pairs), ("weight_pair_metrics.csv", weights),
                       ("layer_code_metrics.csv", codes), ("historical_reproduction.csv", historical)):
        pd.DataFrame(rows).to_csv(folder / name, index=False)
    shutil.copy2(ROOT / "notebook_support.py", folder / "notebook_support.py")
    shutil.copy2(ROOT / NOTEBOOK, folder / NOTEBOOK)


def execute_template(folder):
    notebook = json.loads((folder / NOTEBOOK).read_text(encoding="utf-8"))
    assert sum(c["cell_type"] == "code" for c in notebook["cells"]) == 9
    assert all(c.get("execution_count") is None and not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")
    originals = dict(sys.modules)
    stub = ModuleType("IPython")
    stub.version_info = (9, 0)
    stub.get_ipython = lambda: None
    display = ModuleType("IPython.display")
    display.display = lambda *args, **kwargs: None
    display.Markdown = lambda s: s
    stub.display = display
    sys.modules["IPython"] = stub
    sys.modules["IPython.display"] = display
    env = {}
    original_import = builtins.__import__
    def safe_import(name, *args, **kwargs):
        assert name.split('.')[0] not in BANNED_IMPORTS, "Forbidden notebook import: " + name
        return original_import(name, *args, **kwargs)
    try:
        with redirect_stdout(io.StringIO()):
            builtins.__import__ = safe_import
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                if "parameters" in cell.get("metadata", {}).get("tags", []):
                    env.update(Path=Path, RUN_DIR=folder, SAVE_FIGURES=False)
                else:
                    exec(compile(''.join(cell["source"]), f"notebook_cell_{i}", "exec"), env)
    finally:
        builtins.__import__ = original_import
        for name in ("IPython", "IPython.display"):
            if name in originals:
                sys.modules[name] = originals[name]
            else:
                sys.modules.pop(name, None)
    assert env["run"].confirmed
    assert set(env["figures"]) == {"checkpoint_accuracy", "accuracy_deltas", "paired_changes", "weight_differences", "weight_magnitudes"}
    assert all("SYNTHETISCHE CONTROLE" in fig._suptitle.get_text() for fig in env["figures"].values())
    return env


def main(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "synthetic_complete"
    make_fixture(fixture)
    before = raw_hashes(fixture)
    run = ns.load_run(fixture)
    assert run.confirmed, run.issues
    assert len(run.output_checks) == 2
    assert run.epoch("last_epoch") == 100 and run.epoch("best_E4M3_test") == 54
    direct = run.metrics.loc[run.metrics["role"] == "last_epoch"].set_index("path")
    assert direct.loc["direct4", "top1_percent"] > direct.loc["direct8", "top1_percent"]
    assert "torch" not in sys.modules
    env = execute_template(fixture)
    assert raw_hashes(fixture) == before
    saved = ns.save_figures(run, env["figures"])
    assert len(list(saved.glob('*.png'))) == len(list(saved.glob('*.pdf'))) == 5
    assert raw_hashes(fixture) == before
    assert not (saved / NOTEBOOK).exists()
    plt.close('all')
    negative = {}
    mutations = {
        "mixed_checkpoint_epoch": ("metrics.csv", lambda d: d.assign(epoch=np.where(d.index == 6, 55, d.epoch))),
        "missing_route": ("metrics.csv", lambda d: d.drop(index=7)),
        "wrong_target_BN": ("metrics.csv", lambda d: d.assign(bn_bank_key=np.where(d.path == 'exponent86', 8, d.bn_bank_key))),
        "changed_shared_scale": ("layer_code_metrics.csv", lambda d: d.assign(scales_equal_source=np.where(d.index == 5, False, d.scales_equal_source))),
        "pair_counts_inconsistent_with_predictions": ("paired_metrics.csv", lambda d: d.assign(wrong_to_correct=d.wrong_to_correct + 1, correct_to_wrong=d.correct_to_wrong + 1, prediction_changes=d.prediction_changes + 2)),
    }
    for name, (filename, change) in mutations.items():
        target = out / name
        shutil.copytree(fixture, target, ignore=shutil.ignore_patterns('analysis_*'))
        df = pd.read_csv(target / filename)
        change(df).to_csv(target / filename, index=False)
        bad = ns.load_run(target)
        assert not bad.confirmed and any(x['Niveau'] == 'FOUT' for x in bad.issues), (name, bad.issues)
        negative[name] = {'expected_rejection': True, 'messages': [x['Melding'] for x in bad.issues if x['Niveau'] == 'FOUT']}
    for name in ("NPZ_checksum_mismatch", "changed_original_checkpoint", "unconfirmed_real_provenance"):
        target = out / name
        shutil.copytree(fixture, target, ignore=shutil.ignore_patterns('analysis_*'))
        report = json.loads((target / 'run_report.json').read_text())
        if name == "NPZ_checksum_mismatch":
            report['roles']['last_epoch']['per_image_sha256'] = '0' * 64
        elif name == "changed_original_checkpoint":
            report['reference']['checkpoint_selection']['best']['sha256'] = '0' * 64
        else:
            report.pop('synthetic_fixture_only')
        write_json(target / 'run_report.json', report)
        bad = ns.load_run(target)
        assert not bad.confirmed and any(x['Niveau'] == 'FOUT' for x in bad.issues), (name, bad.issues)
        negative[name] = {'expected_rejection': True, 'messages': [x['Melding'] for x in bad.issues if x['Niveau'] == 'FOUT']}
    unverified = ns.load_run(fixture, verify_per_image=False)
    assert not unverified.confirmed
    check = out / 'check_only'
    check.mkdir()
    write_json(check / 'run_report.json', {'status':'PASS','command':'check','synthetic_checks_only':True})
    check_result = ns.load_run(check)
    assert not check_result.confirmed and check_result.metrics.empty
    report = {'status': 'PASS', 'scope': 'GENERATED_SYNTHETIC_REPORTS_ONLY_NO_MODEL_OR_TRAINING',
              'empty_template_outputs_confirmed': True, 'no_executed_notebook_created': True,
              'all_nine_code_cells_executed': True, 'all_five_figure_types_created': True,
              'both_fixed_checkpoints_all_eight_routes': True,
              'lower_precision_higher_accuracy_accepted': True,
              'tied_logits_compatible_with_torch_topk_prediction': True,
              'NPZ_hash_counts_CE_and_pairs_verified': True,
              'input_hashes_unchanged_after_execute_and_save': True,
              'explicit_save_new_directory_PNG_PDF': True,
              'per_image_verification_required_for_confirmation': True,
              'check_only_not_presented_as_evaluation': True,
              'expected_negative_cases': negative,
              'preview_directory': str(saved),
              'tested_file_sha256': {name: digest(ROOT / name) for name in ('notebook_support.py',NOTEBOOK,'local_validation/test_notebook.py')}}
    write_json(out / 'notebook_validation_report.json', report)
    print(json.dumps({'status':'PASS','report':str(out/'notebook_validation_report.json'),'preview_directory':str(saved)},indent=2))
    return report


if __name__ == '__main__':
    main(Path(sys.argv[1]))
