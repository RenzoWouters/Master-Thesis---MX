"""Short stdlib-only checks for bounded diagnostics and the GPU proof gate.

No CUDA, training data, real model steps, or long-running processes are used.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import check_watchdog


def _run_case(name, program, *, timeout=2., expected_code=0, timed_out=False):
    lines = []
    processes = []
    actual_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        child = actual_popen(*args, **kwargs)
        processes.append(child)
        return child

    with patch.object(check_watchdog.subprocess, "Popen", tracked_popen):
        result = check_watchdog.run_bounded_process(
            [sys.executable, "-u", "-c", program], timeout_seconds=timeout,
            heartbeat_seconds=.08, terminate_grace_seconds=.5,
            log=lines.append, label=name)
    assert result["timed_out"] is timed_out, (name, result)
    assert not result["cleanup_failed"], (name, result)
    assert len(processes) == 1 and processes[0].poll() is not None, (name, result)
    if not timed_out:
        assert result["process_returncode"] == expected_code, (name, result)
    else:
        assert result["elapsed_seconds"] >= timeout, (name, result)
        assert result["elapsed_seconds"] < timeout + 2., (name, result)
    return {"status": "PASS", "timed_out": timed_out,
            "elapsed_seconds": result["elapsed_seconds"],
            "returncode": result["process_returncode"],
            "child_terminated": processes[0].poll() is not None,
            "last_output": result["last_output"], "log_lines": len(lines)}


def _interrupt_cleanup():
    processes = []
    actual_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        child = actual_popen(*args, **kwargs)
        processes.append(child)
        return child

    def interrupting_log(message):
        if "READY_FOR_INTERRUPT" in message:
            raise KeyboardInterrupt("Synthetic parent interrupt")

    caught = False
    began = time.monotonic()
    with patch.object(check_watchdog.subprocess, "Popen", tracked_popen):
        try:
            check_watchdog.run_bounded_process(
                [sys.executable, "-u", "-c",
                 "import time; print('READY_FOR_INTERRUPT', flush=True); time.sleep(3)"],
                timeout_seconds=2., heartbeat_seconds=.1,
                terminate_grace_seconds=.5, log=interrupting_log)
        except KeyboardInterrupt:
            caught = True
    assert caught, "The parent interrupt must propagate."
    assert len(processes) == 1 and processes[0].poll() is not None
    return {"status": "PASS", "interrupt_propagated": caught,
            "child_terminated": True, "elapsed_seconds": time.monotonic() - began}


def _proof_gates():
    import run_train

    with tempfile.TemporaryDirectory(prefix="r50-proof-test-") as temporary:
        root = Path(temporary)
        report = {"package_signature": {"sha256": "test-signature"}}
        proof_file = root / "run_report.json"
        args = SimpleNamespace(check_report=proof_file)
        rejected = []
        examples = {
            "selected_timeout": {"serial_dp4": {"status": "TIMEOUT"}},
            "only_native_pass": {"native256": {"status": "PASS"}},
            "selected_skipped": {"serial_dp4": {"status": "SKIPPED"}},
        }
        for name, memory_tests in examples.items():
            proof = {"command": "check", "status": "PASS",
                     "package_signature": {"sha256": "test-signature"},
                     "memory_tests": memory_tests}
            proof_file.write_text(json.dumps(proof), encoding="utf-8")
            try:
                run_train.find_check(args, report)
            except RuntimeError:
                rejected.append(name)
        assert rejected == list(examples), rejected

        valid = {"command": "check", "status": "PASS",
                 "package_signature": {"sha256": "test-signature"},
                 "memory_tests": {"serial_dp4": {"status": "PASS"},
                                  "native256": {"status": "SKIPPED"}}}
        proof_file.write_text(json.dumps(valid), encoding="utf-8")
        found_path, found = run_train.find_check(args, report)
        assert found_path == proof_file.resolve() and found == valid

        # A recent failed check for this package must not silently fall back
        # to an older successful check. Unrelated packages do not decide it.
        older = root / "20200101_000000_000000_check"
        newer = root / "20200102_000000_000000_check"
        unrelated = root / "20200103_000000_000000_check"
        for directory in (older, newer, unrelated):
            directory.mkdir()
        (older / "run_report.json").write_text(json.dumps(valid), encoding="utf-8")
        newer_proof = dict(valid, status="BLOCKED_GPU_TIMEOUT",
                           memory_tests={"serial_dp4": {"status": "TIMEOUT"}})
        (newer / "run_report.json").write_text(json.dumps(newer_proof), encoding="utf-8")
        other_proof = dict(valid, package_signature={"sha256": "other-package"})
        (unrelated / "run_report.json").write_text(json.dumps(other_proof), encoding="utf-8")
        default_args = SimpleNamespace(check_report=None, output_root=root, project_root=root)
        rejected_latest = False
        try:
            run_train.find_check(default_args, report)
        except RuntimeError:
            rejected_latest = True
        assert rejected_latest, "The newest matching timeout must block an older PASS."

        # Explicitly pointing to an older valid report is still a conscious
        # supported choice; all status/signature requirements continue to hold.
        chosen_path, _ = run_train.find_check(
            SimpleNamespace(check_report=older / "run_report.json"), report)
        assert chosen_path == (older / "run_report.json").resolve()

        calls = []

        def timed_out_test(mode, *args, **kwargs):
            calls.append(mode)
            return {"mode": mode, "status": "TIMEOUT", "elapsed_seconds": 300.,
                    "error": "Synthetic timeout, no GPU used."}

        check_args = SimpleNamespace(include_native256=True, gpu_timeout_seconds=300.)
        blocked_report = {}
        with patch.object(run_train.c, "check_environment", return_value={}), \
                patch.object(run_train, "cpu_selftests", return_value={"status": "PASS"}), \
                patch.object(run_train, "execute_memory_test", side_effect=timed_out_test):
            run_train.command_check(check_args, root, blocked_report, lambda _: None, root)
        assert blocked_report["status"] == "BLOCKED_GPU_TIMEOUT", blocked_report
        assert calls == ["serial_dp4"], calls

    return {"status": "PASS", "rejected_proofs": rejected,
            "serial_pass_with_native_skipped_accepted": True,
            "newest_matching_timeout_blocks_older_pass": True,
            "explicit_valid_report_still_supported": True,
            "mandatory_timeout_blocks_run": True,
            "optional_native_not_started_after_mandatory_timeout": True}


def _parent_outcome_precedence():
    import run_train

    outcomes = {}
    cases = (
        ("late_pass", {"mode": "serial_dp4", "status": "PASS"}, True, False, 0, "TIMEOUT"),
        ("failed_cleanup", {"mode": "serial_dp4", "status": "PASS"}, True, True, None, "FAIL"),
        ("partial_progress", {"mode": "serial_dp4", "status": "RUNNING",
                              "last_progress": {"stage": "forward_start", "bit": 32}},
         True, False, -15, "TIMEOUT"),
        ("actual_oom", {"mode": "serial_dp4", "status": "OOM"}, False, False, 3, "OOM"),
        ("inconsistent_exit", {"mode": "serial_dp4", "status": "PASS"},
         False, False, 1, "REJECTED"),
    )
    with tempfile.TemporaryDirectory(prefix="r50-outcome-test-") as temporary:
        root = Path(temporary)
        for name, child_result, timed_out, cleanup_failed, code, expected in cases:
            case_root = root / name
            target = case_root / "memory_serial_dp4"
            target.mkdir(parents=True)
            filename = "progress.json" if name == "partial_progress" else "result.json"
            child_path = target / filename
            child_text = json.dumps(child_result)
            child_path.write_text(child_text, encoding="utf-8")
            watched = {"process_returncode": code, "timed_out": timed_out,
                       "elapsed_seconds": .3, "last_output": "Synthetic stage",
                       "pid": 1234567, "cleanup_failed": cleanup_failed,
                       "stdout_closed": True, "timeout_seconds": .3}
            args = SimpleNamespace(cpu_threads=1, gpu_timeout_seconds=.3)
            result = None
            with patch.object(run_train, "run_bounded_process", return_value=watched):
                try:
                    result = run_train.execute_memory_test(
                        "serial_dp4", root, case_root, args, lambda _: None)
                except RuntimeError:
                    assert expected == "REJECTED", name
                else:
                    assert expected != "REJECTED", name
                    assert result["status"] == expected, (name, result)
                    persisted = json.loads((target / "watchdog_result.json").read_text(encoding="utf-8"))
                    assert persisted == result, name
                    if name == "partial_progress":
                        assert result["last_progress"] == child_result["last_progress"]
            assert child_path.read_text(encoding="utf-8") == child_text, name
            outcomes[name] = expected
    return {"status": "PASS", "outcomes": outcomes,
            "child_reports_preserved": True, "parent_outcome_persisted": True}


def run_selftests():
    assert __debug__, "Run without -O; assertions are required."
    began = time.monotonic()
    cases = {}
    cases["successful_exit"] = _run_case("pass", "print('completed', flush=True)")
    cases["nonzero_exit"] = _run_case(
        "failure", "import sys; print('failed', flush=True); sys.exit(7)", expected_code=7)
    cases["quiet_timeout"] = _run_case(
        "quiet", "import time; time.sleep(3)", timeout=.3, timed_out=True)
    cases["continuous_output_timeout"] = _run_case(
        "chatty", "import time\nwhile True:\n print('tick', flush=True)\n time.sleep(.005)",
        timeout=.3, timed_out=True)
    cases["partial_line_timeout"] = _run_case(
        "partial", "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(3)",
        timeout=.3, timed_out=True)
    cases["eof_without_exit_timeout"] = _run_case(
        "eof", "import os,time; os.close(1); os.close(2); time.sleep(3)",
        timeout=.3, timed_out=True)
    cases["interrupt_cleanup"] = _interrupt_cleanup()
    cases["proof_gates"] = _proof_gates()
    cases["parent_outcome_precedence"] = _parent_outcome_precedence()
    return {"status": "PASS", "device": "none", "platform": sys.platform,
            "python": sys.version, "cases": cases,
            "elapsed_seconds": time.monotonic() - began,
            "scope": "Synthetic stdlib subprocesses and mocked status gates; no Windows/CUDA validation."}


if __name__ == "__main__":
    print(json.dumps(run_selftests(), indent=2))
