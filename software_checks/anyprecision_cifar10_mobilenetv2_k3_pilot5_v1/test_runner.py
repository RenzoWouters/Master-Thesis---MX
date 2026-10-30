"""Scientific reporting and process-boundary checks; no GPU or dataset needed."""
from __future__ import annotations

from pathlib import Path
import tempfile
import sys
from zipfile import ZipFile

import support as s


def run_tests():
    from process_watchdog import run_bounded_process
    from run_train import classify_worker, select_approved_check
    tests = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        tests.append(name)
    with tempfile.TemporaryDirectory(prefix="mobilenet_runner_test_") as temp:
        root = Path(temp)
        out1 = s.new_run_directory(root, "check")
        out2 = s.new_run_directory(root, "check")
        check(out1 != out2 and out1.is_dir(), "results_directories_unique_previous_retained")
        fingerprint = {"sha256": "test_fingerprint"}
        good = {"status": "PASS", "command": "check", "experiment": s.CONFIG["experiment"],
                "package_fingerprint": fingerprint,
                "worker_result": {"status": "PASS", "eligible_check": True}}
        s.json_write(out1 / "run_report.json", good)
        s.json_write(out2 / "run_report.json", {**good, "package_fingerprint": {"sha256": "other"}})
        selected, _ = select_approved_check(root, fingerprint)
        check(selected.parent == out1, "only_matching_successful_check_selected")
        try:
            select_approved_check(root, {"sha256": "absent"})
            raise AssertionError("missing_check_should_fail")
        except RuntimeError:
            tests.append("missing_check_rejected")
        (out1 / "diagnostic_checkpoint.pt").write_bytes(b"not-real-weights")
        (out1 / "dataset.bin").write_bytes(b"not-real-data")
        s.json_write(out1 / "metrics.json", {"known": 1})
        bundle = s.make_feedback_bundle(out1)
        with ZipFile(bundle) as archive:
            names = archive.namelist()
            check("metrics.json" in names and "diagnostic_checkpoint.pt" not in names
                  and "dataset.bin" not in names, "feedback_excludes_weights_and_dataset")
        check(classify_worker({"timed_out": True, "cleanup_failed": False}, {"status": "PASS"})[0] == "TIMEOUT",
              "timeout_overrides_partial_worker_pass")
        check(classify_worker({"process_returncode": 0, "cleanup_failed": True}, {"status": "PASS"})[0] == "FAIL",
              "cleanup_failure_never_accepted")
        check(classify_worker({"process_returncode": 0, "cleanup_failed": False}, {"status": "PASS"}) == ("PASS", 0),
              "successful_worker_accepted")
        messages = []
        result = run_bounded_process([sys.executable, "-u", "-c", "print('bounded-child-ok',flush=True)"],
                                     timeout_seconds=10, heartbeat_seconds=2, log=messages.append,
                                     terminate_grace_seconds=1, label="selftest")
        check(result["process_returncode"] == 0 and not result["cleanup_failed"]
              and "bounded-child-ok" in messages, "native_process_tree_success_and_cleanup")
        result = run_bounded_process([sys.executable, "-u", "-c", "import time; time.sleep(30)"],
                                     timeout_seconds=.5, heartbeat_seconds=.2, log=messages.append,
                                     terminate_grace_seconds=1, label="selftest-timeout")
        check(result["timed_out"] and not result["cleanup_failed"], "native_process_timeout_stops_owned_child")
    tests.extend(extra_runner_checks())
    return {"status": "PASS", "tests_run": len(tests), "tests": tests,
            "scope": "CPU/filesystem/native process containment; no CUDA; no dataset"}


def extra_runner_checks():
    import contextlib
    import copy
    import io
    import math
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from train_worker import MetricAccumulator, checkpoint_selection, hydrate_resume_history, iterate_loader
    from run_train import build_parser, classify_worker
    from test_model_engine import _TinyPrecisionModel
    import engine
    names = []
    def record(okay, name):
        if not okay:
            raise AssertionError(name)
        names.append(name)
    metrics = lambda score, n: {str(b): {"loss": 2., "top1": score, "top5": max(50., score), "samples": n} for b in s.BITS}
    history = [{"epoch": ep, "lr": .001, "global_updates": ep * 391,
                "training": metrics(20, 50000), "validation": metrics(score, 10000),
                "checkpoint": {"path": "epoch" + str(ep)}}
               for ep, score in [(1, 70.), (2, 72.), (3, 72.)]]
    history[1]["validation"]["1"]["top1"] = 51.
    history[2]["validation"]["1"]["top1"] = 68.
    selection = checkpoint_selection(history)
    record(selection["last"]["epoch"] == 3 and selection["best_w32"]["epoch"] == 2
           and selection["best_w32"]["metrics"]["1"]["top1"] == 51.,
           "highest_w32_selects_one_shared_checkpoint_earliest_exact_tie")
    a = MetricAccumulator()
    for size, top1 in [(128, 100.), (80, 0.)]:
        a.add({"batch_size": size, "optimizer_steps": 1, "metrics": metrics(top1, size)}, size, True)
    result = a.finish()
    record(all(abs(v["top1"] - 100 * 128 / 208) < 1e-12 and v["samples"] == 208 for v in result.values()),
           "partial_final_batch_metrics_weighted_by_examples")
    record(classify_worker({"process_returncode": 0}, {"status": "PAUSED"}) == ("PAUSED", 0),
           "planned_pause_distinct_from_five_epoch_completion")
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            build_parser().parse_args(["run", "--stop-after-epoch", "400"])
            raise AssertionError("400-epoch option must be rejected")
        except SystemExit as exc:
            record(exc.code != 0, "public_pilot_cli_cannot_start_400_epochs")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "epoch_0003.pt"
        path.write_bytes(b"unit-test-only-fake-checkpoint")
        payload = {"epoch_completed": 3, "history": history}
        restored = hydrate_resume_history(payload, path)
        record(restored[-1]["timing"]["total_seconds"] is None,
               "missing_checkpoint_sidecar_preserves_unknown_timing")
        sidecar = copy.deepcopy(history[-1])
        sidecar["checkpoint"] = {"sha256": s.sha256(path), "path": str(path)}
        sidecar["timing"] = {"total_seconds": 123., "checkpoint_seconds": .3}
        s.json_write(path.with_suffix(".json"), sidecar)
        restored = hydrate_resume_history(payload, path)
        record(restored[-1]["timing"]["total_seconds"] == 123., "matching_sidecar_restores_actual_epoch_timing")
        sidecar["validation"]["32"]["top1"] += 1
        s.json_write(path.with_suffix(".json"), sidecar)
        try:
            hydrate_resume_history(payload, path)
            raise AssertionError("sidecar must not change accuracy")
        except RuntimeError:
            names.append("sidecar_cannot_replace_checkpoint_accuracy")
    # Exercise the actual iterator/engine path with one smaller final batch,
    # independent of the complete-dataset mocks in test_full_data.py.
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(177)
        loader = DataLoader(TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([1, 2, 3])), batch_size=2)
        model = _TinyPrecisionModel()
        optimizer = engine.create_optimizer(model)
        scores, rows, elapsed = iterate_loader(model, optimizer, loader, training=True,
                    epoch=1, device=torch.device("cpu"), log=lambda _: None)
        record([r["batch_size"] for r in rows] == [2, 1] and all(v["samples"] == 3 for v in scores.values())
               and all(int(v["step"]) == 2 for v in optimizer.state.values()) and elapsed > 0,
               "production_epoch_iterator_handles_tail_and_single_adam_per_batch")
        before = {k: v.clone() for k, v in model.state_dict().items()}
        scores, rows, _ = iterate_loader(model, optimizer, loader, training=False,
                    epoch=1, device=torch.device("cpu"), log=lambda _: None)
        record(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
               and all(v["samples"] == 3 for v in scores.values()),
               "production_validation_iterator_preserves_model_and_counts_all_examples")
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
    return names


if __name__ == "__main__":
    import json
    print(json.dumps(run_tests(), indent=2))
