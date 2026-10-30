"""Focused delivery gates for the separate backend-only diagnostic.

No CUDA, ImageNet or neural-network updates are used by these tests.
Only temporary copies are modified when checking hash rejection.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _must_reject(call, label):
    try:
        call()
    except (RuntimeError, ValueError, FileNotFoundError):
        return label
    raise AssertionError("Expected rejection: " + label)


def run_selftests(reference_root=None):
    assert __debug__, "Run without Python -O; assertions are required."
    import run_check

    began = time.monotonic()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "reference_manifest.json").read_text(encoding="utf-8"))
    reference = Path(reference_root).resolve() if reference_root is not None else (
        root.parent / manifest["reference_package"])
    names = sorted(set(manifest["package_signature"]["files"]) |
                   {manifest["source_archive_relpath"]})
    original = {name: _digest(reference / name) for name in names}
    verified = run_check.verify_reference(reference)
    assert isinstance(verified, dict), "Reference verification must return metadata."

    rejected = []
    with tempfile.TemporaryDirectory(prefix="r50-backend-delivery-") as temporary:
        directory = Path(temporary)
        fixture = directory / manifest["reference_package"]
        for name in names:
            target = fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference / name, target)
        run_check.verify_reference(fixture)

        changed = fixture / "engine.py"
        before = changed.read_bytes()
        changed.write_bytes(before + b"\n# Deliberately modified test copy.\n")
        changed_digest = _digest(changed)
        rejected.append(_must_reject(lambda: run_check.verify_reference(fixture), "changed_engine"))
        assert _digest(changed) == changed_digest, "Verification must not repair or rewrite a file."
        changed.write_bytes(before)

        missing = fixture / "common.py"
        before = missing.read_bytes()
        missing.unlink()
        rejected.append(_must_reject(lambda: run_check.verify_reference(fixture), "missing_common"))
        assert not missing.exists(), "Verification must not recreate missing source files."
        missing.write_bytes(before)

        archive = fixture / manifest["source_archive_relpath"]
        before = archive.read_bytes()
        archive.write_bytes(before + b"invalid-hash-fixture")
        rejected.append(_must_reject(lambda: run_check.verify_reference(fixture), "changed_upstream_archive"))
        archive.write_bytes(before)
        run_check.verify_reference(fixture)

        # Even if a diagnostic report is copied into a check-like directory
        # and given the reference signature, its command prevents pilot use.
        reference_run = run_check.load_reference(reference)
        proof = {"command": "backend_diagnostic", "requested_command": "check",
                 "eligible_as_pilot_check": False, "status": "PASS",
                 "package_signature": manifest["package_signature"],
                 "memory_tests": {"serial_dp4": {"status": "PASS"}},
                 "config": {"cudnn_benchmark": False}}
        proof_path = directory / "run_report.json"
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
        args = SimpleNamespace(check_report=proof_path)
        destination = {"package_signature": manifest["package_signature"]}
        rejected.append(_must_reject(
            lambda: reference_run.find_check(args, destination), "diagnostic_cannot_unlock_pilot"))
        assert json.loads(proof_path.read_text(encoding="utf-8")) == proof

        # The parent decides whether a child actually completed within its
        # deadline. A late child PASS must never hide a timeout/cleanup error.
        import check_watchdog
        parent_outcomes = {}
        for name, child_status, timed_out, cleanup_failed, code, expected_status in (
            ("completed", "PASS", False, False, 0, "PASS"),
            ("late_pass", "PASS", True, False, 0, "TIMEOUT"),
            ("cleanup_failed", "PASS", True, True, None, "FAIL"),
            ("actual_oom", "OOM", False, False, 3, "OOM"),
        ):
            case = directory / name
            target = case / "backend_test"
            target.mkdir(parents=True)
            child_path = target / "result.json"
            child_text = json.dumps({"status": child_status})
            child_path.write_text(child_text, encoding="utf-8")
            watched = {"process_returncode": code, "timed_out": timed_out,
                       "elapsed_seconds": 300., "last_output": "Synthetic test",
                       "pid": 1234567, "cleanup_failed": cleanup_failed,
                       "stdout_closed": True, "timeout_seconds": 300.}
            args = SimpleNamespace(reference_root=reference, cpu_threads=4)
            with patch.object(check_watchdog, "run_bounded_process", return_value=watched):
                result = run_check.execute_gpu(args, case, directory, lambda _: None)
            assert result["status"] == expected_status, (name, result)
            assert result["eligible_as_pilot_check"] is False, name
            assert json.loads((target / "watchdog_result.json").read_text(encoding="utf-8")) == result
            assert child_path.read_text(encoding="utf-8") == child_text
            parent_outcomes[name] = expected_status

    rejected_commands = []
    for command in ("run", "resume", "train"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                run_check.parser().parse_args([command])
            except SystemExit as exc:
                assert exc.code == 2, (command, exc.code)
                rejected_commands.append(command)
            else:
                raise AssertionError("Training command unexpectedly accepted: " + command)
    for command in ("check", "selftest"):
        assert run_check.parser().parse_args([command]).command == command

    assert {name: _digest(reference / name) for name in names} == original
    config = json.loads((root / "diagnostic_config.json").read_text(encoding="utf-8"))
    old_config = json.loads((reference / "experiment_config.json").read_text(encoding="utf-8"))
    assert config["cudnn_benchmark"] is False and old_config["cudnn_benchmark"] is True
    assert config["global_batch"] == old_config["global_batch"] == 256
    assert config["bn_shards"] == [64] * old_config["reference_replicas"]
    assert config["precision_bits"] == old_config["bits"]
    for field in ("execution", "amp", "fp32_precision", "seed"):
        assert config[field] == old_config[field], field
    assert config["training_authorized"] is False and config["dataset_required"] is False
    return {"status": "PASS", "device": "none", "platform": sys.platform,
            "reference_files_unchanged": True, "checked_reference_files": len(names),
            "rejected_inputs": rejected, "rejected_training_commands": rejected_commands,
            "fixed_batch_and_precision_contract": True,
            "diagnostic_cannot_unlock_pilot": True,
            "parent_outcome_precedence": parent_outcomes,
            "elapsed_seconds": time.monotonic() - began,
            "scope": "Reference hashes, parser and pilot-report gates; no GPU or real training."}


if __name__ == "__main__":
    print(json.dumps(run_selftests(), indent=2))
