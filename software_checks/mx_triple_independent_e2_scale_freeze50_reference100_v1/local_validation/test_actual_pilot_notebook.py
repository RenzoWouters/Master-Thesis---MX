"""Exercise the new reader with unchanged real pilot values in a labelled QA copy.

Only the copy's report/configuration metadata is adapted to the new partial-run
contract. This is a software compatibility fixture, never a new training result.
The original feedback and all numerical metric/history/scale files stay intact.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    package, feedback, out = (p.resolve() for p in (args.package, args.feedback, args.out))
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "actual_pilot_values_compatibility_fixture"
    fixture.mkdir()
    originals = {p.name: digest(p) for p in feedback.iterdir()
                 if p.is_file() and p.suffix in (".json", ".csv")}
    for name in originals:
        shutil.copy2(feedback / name, fixture / name)

    report = json.loads((fixture / "run_report.json").read_text())
    config = report["config"]
    config.pop("pilot_stop_epoch", None)
    config.pop("pilot_new_epochs", None)
    config.update(branch_stop_epoch=100, branch_new_epochs=50, inherited_epochs=50)
    report.update(schema="mx-triple-independent-e2-scale-freeze50-reference100-report-v1",
                  status="RUNNING", package_revision="mx_triple_independent_e2_scale_freeze50_reference100_v1",
                  synthetic_fixture_only=True, branch_training_completed=False,
                  full_training_completed=False, full_100_epoch_training_completed=False)
    report.pop("pilot_completed", None)
    note = {
        "purpose": "Notebook compatibility check only; no new training was executed.",
        "source_feedback": str(feedback),
        "numerical_values": "Unmodified actual freeze-pilot values through epoch 55.",
        "metadata_only_adaptation": "New partial-run schema; explicitly marked synthetic software fixture.",
        "not_a_new_reference100_run": True,
    }
    report["compatibility_fixture"] = note
    (fixture / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = json.loads((fixture / "training_summary.json").read_text())
    summary.pop("pilot_completed", None)
    summary.pop("pilot_stop_epoch", None)
    summary.update(schema="mx-triple-independent-e2-scale-freeze50-reference100-completion-v1",
                   branch_stop_epoch=100, branch_new_epochs=50, synthetic_fixture_only=True,
                   branch_training_completed=False, full_training_completed=False,
                   full_100_epoch_training_completed=False)
    (fixture / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    timing = json.loads((fixture / "timing_estimate.json").read_text())
    timing.pop("pilot_completed", None)
    timing.update(branch_training_completed=False, full_training_completed=False,
                  full_100_epoch_training_completed=False, measured_50_epoch_branch_seconds=None,
                  combined_inherited_and_branch_epoch_seconds=timing["measured_completed_epoch_seconds"],
                  measured_completed_epoch_seconds=timing["measured_new_branch_epoch_seconds"])
    (fixture / "timing_estimate.json").write_text(json.dumps(timing, indent=2) + "\n")
    (fixture / "FIXTURE_PROVENANCE.json").write_text(json.dumps(note, indent=2) + "\n")

    spec = importlib.util.spec_from_file_location("actual_pilot_notebook_qa", package / "notebook_support.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run = module.load_run(fixture)
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name + ": " + str(run.issues))
        checks.append(name)

    check(run.synthetic, "compatibility_fixture_explicitly_not_new_research")
    check(run.metrics_safe and run.scales_safe and run.selection_safe and run.reference_safe,
          "real_pilot_values_pass_new_partial_reader")
    check(run.last_epoch == 55 and run.branch_epochs == 5 and not run.reference_complete,
          "five_available_new_epochs_do_not_claim_fifty")
    check("gedeeltelijke resultaten" in module.status_markdown(run), "partial_status_explicit")
    check("naar schatting" in module.timing_markdown(run), "remaining_time_explicitly_projected")
    check(module.checkpoint_table(run, "last").Epoch.tolist() == [55], "plot_uses_actual_epoch55")
    check(module.checkpoint_table(run, "best").Epoch.tolist() == [34], "global_best_retains_inherited_epoch34")
    check(module.checkpoint_table(run, "branch_best").Epoch.tolist() == [51], "branch_best_starts_after50")
    proof = module.freeze_evidence_table(run).set_index("Freezecontrole").Waarde
    check(proof["Groepen met gewijzigde continue φ sinds50"] == 0
          and proof["Groepen met andere absolute E2-exponent sinds50"] == 0,
          "actual_4080_scales_frozen")
    for name, function in (("partial55_accuracy", module.plot_checkpoints),
                           ("partial55_branch", module.plot_final_window),
                           ("partial55_freeze", module.plot_freeze_comparison),
                           ("partial55_timing", module.plot_timing)):
        figure = function(run)
        figure.canvas.draw()
        figure.savefig(out / (name + ".png"), dpi=100, bbox_inches="tight")
        module.plt.close(figure)
    checks.append("actual_partial_data_figures_render_with_fixture_label")
    immutable = ("history.json", "epoch_metrics.csv", "selected_metrics.csv", "scale_block_state.csv",
                 "scale_epoch_metrics.csv", "weight_updates.csv", "epoch_times.csv")
    check(all(digest(fixture / name) == originals[name] for name in immutable),
          "numerical_history_metrics_and_scales_not_changed")
    check(all(digest(feedback / name) == value for name, value in originals.items()),
          "original_feedback_left_unchanged")
    result = {"status": "PASS", "checks": checks, **note,
              "fixture_path": str(fixture), "new_training_executed": False}
    (out / "actual_pilot_notebook_validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
