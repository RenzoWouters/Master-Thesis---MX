"""Scientific reporting and process-boundary checks; no GPU or dataset needed."""
from __future__ import annotations

from pathlib import Path
import tempfile
import sys
from zipfile import ZipFile

import support as s


def run_tests():
    from process_watchdog import run_bounded_process
    from run_check import classify_worker, select_approved_check
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
    return {"status": "PASS", "tests_run": len(tests), "tests": tests,
            "scope": "CPU/filesystem/native process containment; no CUDA; no dataset"}


if __name__ == "__main__":
    import json
    print(json.dumps(run_tests(), indent=2))
