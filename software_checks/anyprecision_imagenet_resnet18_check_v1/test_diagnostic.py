"""Short CPU-only checks for the synthetic ResNet-18 diagnostic.

The process tests use tiny stdlib subprocesses. Worker tests use mocks and
therefore establish orchestration/persistence, never CUDA model equivalence.
Only temporary fixture copies are changed by source-integrity rejection tests.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class FakeOOM(RuntimeError):
    pass


class FakeCUDA:
    """Observable allocator mock; no torch/CUDA initialization occurs here."""

    OutOfMemoryError = FakeOOM

    def __init__(self, trace):
        self.trace = trace
        self.initialized = False
        self.allocated, self.reserved = 100, 200
        self.peak_allocated, self.peak_reserved = 100, 200

    def is_initialized(self):
        return self.initialized

    def synchronize(self):
        self.trace.append("sync")

    def mem_get_info(self):
        self.trace.append("memory_query")
        return 1000 - self.allocated, 1000

    def memory_allocated(self):
        self.trace.append("memory_query")
        return self.allocated

    def memory_reserved(self):
        self.trace.append("memory_query")
        return self.reserved

    def max_memory_allocated(self):
        self.trace.append("memory_query")
        return self.peak_allocated

    def max_memory_reserved(self):
        self.trace.append("memory_query")
        return self.peak_reserved

    def reset_peak_memory_stats(self):
        self.trace.append("peak_reset")
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def allocate(self, peak, after):
        self.peak_allocated = max(self.peak_allocated, peak)
        self.peak_reserved = max(self.peak_reserved, peak + 100)
        self.allocated, self.reserved = after, after + 100


class FileLog:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")

    def __call__(self, message):
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _must_reject(call, label):
    try:
        call()
    except (RuntimeError, ValueError, FileNotFoundError):
        return label
    raise AssertionError("Expected rejection: " + label)


def _process_case(name, program, *, timeout=2., expected_code=0, timed_out=False):
    import check_watchdog

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
    if timed_out:
        assert timeout <= result["elapsed_seconds"] < timeout + 2., (name, result)
    else:
        assert result["process_returncode"] == expected_code, (name, result)
    return {"status": "PASS", "timed_out": timed_out,
            "elapsed_seconds": result["elapsed_seconds"],
            "returncode": result["process_returncode"],
            "child_terminated": True, "log_lines": len(lines)}


def _interrupt_cleanup():
    import check_watchdog

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
    return {"status": "PASS", "interrupt_propagated": True,
            "child_terminated": True, "elapsed_seconds": time.monotonic() - began}


def _watchdog_tests():
    return {
        "successful_exit": _process_case("pass", "print('completed', flush=True)"),
        "nonzero_exit": _process_case(
            "failure", "import sys; print('failed', flush=True); sys.exit(7)", expected_code=7),
        "quiet_timeout": _process_case(
            "quiet", "import time; time.sleep(3)", timeout=.3, timed_out=True),
        "continuous_output_timeout": _process_case(
            "chatty", "import time\nwhile True:\n print('tick', flush=True)\n time.sleep(.005)",
            timeout=.3, timed_out=True),
        "partial_line_timeout": _process_case(
            "partial", "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(3)",
            timeout=.3, timed_out=True),
        "eof_without_exit_timeout": _process_case(
            "eof", "import os,time; os.close(1); os.close(2); time.sleep(3)",
            timeout=.3, timed_out=True),
        "interrupt_cleanup": _interrupt_cleanup(),
    }


def _cli_tests():
    import run_check

    rejected = []
    for command in ("run", "resume", "train"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                run_check.parser().parse_args([command])
            except SystemExit as exc:
                assert exc.code == 2, (command, exc.code)
                rejected.append(command)
            else:
                raise AssertionError("Training command unexpectedly accepted: " + command)
    for command in ("check", "selftest"):
        args = run_check.parser().parse_args([command])
        assert args.command == command
    return {"status": "PASS", "rejected_training_commands": rejected}


def _source_tests(package_root):
    import support as s

    root = Path(package_root).resolve()
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    pinned = sorted(list(manifest["inherited_files"]) +
                    [entry["archive_relpath"] for entry in manifest["archives"].values()])
    original = {name: _digest(root / name) for name in pinned}
    verified = s.verify_sources(root)
    assert isinstance(verified, dict), "Source verification must return provenance metadata."
    rejected = []
    with tempfile.TemporaryDirectory(prefix="r18-integrity-test-") as temporary:
        directory = Path(temporary)
        fixture = directory / "package"
        fixture.mkdir()
        for name in pinned + ["SOURCE_MANIFEST.json"]:
            destination = fixture / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        s.verify_sources(fixture)

        changed = fixture / "engine.py"
        before = changed.read_bytes()
        changed.write_bytes(before + b"\n# Intentional test-fixture mutation.\n")
        rejected_digest = _digest(changed)
        rejected.append(_must_reject(lambda: s.verify_sources(fixture), "changed_engine"))
        assert _digest(changed) == rejected_digest, "Verification must not rewrite sources."
        changed.write_bytes(before)

        for key, entry in manifest["archives"].items():
            archive = fixture / entry["archive_relpath"]
            before = archive.read_bytes()
            archive.write_bytes(before + b"deliberately-wrong-sha256")
            rejected.append(_must_reject(lambda: s.verify_sources(fixture), "changed_" + key))
            archive.write_bytes(before)
            archive.unlink()
            rejected.append(_must_reject(lambda: s.verify_sources(fixture), "missing_" + key))
            assert not archive.exists(), "Verification must not recreate missing archives."
            archive.write_bytes(before)
        s.verify_sources(fixture)

        output = directory / "result"
        output.mkdir()
        sources, provenance = s.prepare_sources(output)
        assert set(sources) == {"anyprecision", "truncquant"}
        assert isinstance(provenance, dict)
        source_root = output / "source_workcopy"
        assert s.verify_source_copy(source_root) == sources
        changed_copy = sources["anyprecision"] / "models" / "quan_ops.py"
        copy_before = changed_copy.read_bytes()
        changed_copy.write_bytes(copy_before + b"\n# Intentional extracted-source mutation.\n")
        rejected.append(_must_reject(
            lambda: s.verify_source_copy(source_root), "changed_extracted_quantizer"))
        changed_copy.write_bytes(copy_before)
        assert s.verify_source_copy(source_root) == sources
        assert not list(source_root.rglob("*.pyc")), "Uploaded bytecode must never be extracted."

    assert {name: _digest(root / name) for name in pinned} == original
    return {"status": "PASS", "rejected_inputs": rejected,
            "original_archives_and_inherited_code_unchanged": True,
            "verified_pinned_files": len(pinned), "extracted_copy_verified": True}


def _preflight_gating_test():
    import run_check
    import support as s

    with tempfile.TemporaryDirectory(prefix="r18-preflight-test-") as temporary:
        root = Path(temporary)
        with patch.object(s, "Log", FileLog), patch.object(s, "copy_context", lambda out: None), \
                patch.object(s, "verify_sources", side_effect=RuntimeError("mock source integrity failure")), \
                patch.object(s, "check_environment") as environment, \
                patch.object(run_check, "execute_gpu") as gpu:
            code = run_check.main(["check", "--project-root", str(root)])
        assert code == 1 and not environment.called and not gpu.called
        paths = list((root / "mx_runs" / "anyprecision_imagenet_resnet18_check").glob("*_check"))
        assert len(paths) == 1
        report = json.loads((paths[0] / "run_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "FAIL" and "gpu_check" not in report
        assert report["error"] == "mock source integrity failure"
        assert (paths[0] / "feedback_bundle.zip").is_file()
    return {"status": "PASS", "source_failure_before_environment_or_gpu": True,
            "failure_report_and_feedback_preserved": True}


def _output_isolation_test():
    import support as s

    with tempfile.TemporaryDirectory(prefix="r18-output-test-") as temporary:
        project = Path(temporary)
        old = project / "mx_runs" / "prior-experiment"
        old.mkdir(parents=True)
        marker = old / "existing-results.json"
        marker.write_text('{"accuracy": 95.0}', encoding="utf-8")
        before = marker.read_bytes()
        first = s.new_output(project, "check")
        second = s.new_output(project, "check")
        assert first != second and first.parent == second.parent
        assert first.parent.name == "anyprecision_imagenet_resnet18_check"
        assert first.is_dir() and second.is_dir() and marker.read_bytes() == before

        class ConstantTime:
            @classmethod
            def now(cls):
                return cls()

            def strftime(self, fmt):
                return "20000101_000000_000000"

        with patch.object(s, "datetime", ConstantTime):
            collision = s.new_output(project, "check")
            marker_collision = collision / "do-not-overwrite.txt"
            marker_collision.write_text("existing result", encoding="utf-8")
            try:
                s.new_output(project, "check")
            except FileExistsError:
                pass
            else:
                raise AssertionError("An existing output directory must never be reused.")
            assert marker_collision.read_text(encoding="utf-8") == "existing result"
        assert marker.read_bytes() == before
    return {"status": "PASS", "new_outputs_are_unique": True,
            "collision_rejected": True, "prior_results_unchanged": True}


def _parent_outcome_tests():
    import run_check

    outcomes = {}
    complete = {"status": "PASS", "step_seconds": [1., 1.],
                "optimizer_updates_completed": 3, "finite_losses_and_gradients": True,
                "memory_updates": [{"status": "PASS", "kind": kind, "index": index}
                                   for kind, index in (("warmup", 0), ("measured", 1), ("measured", 2))],
                "bn_update_checks": [{"status": "PASS"} for _ in range(3)]}
    cases = (
        ("completed", complete, False, False, 0, "PASS"),
        ("incomplete_pass", {"status": "PASS", "step_seconds": [1.],
                             "optimizer_updates_completed": 2}, False, False, 0, "REJECTED"),
        ("late_pass", {"status": "PASS"}, True, False, 0, "TIMEOUT"),
        ("failed_cleanup", {"status": "PASS"}, True, True, None, "FAIL"),
        ("partial_progress", {"status": "RUNNING", "step_seconds": [1.],
                              "last_progress": {"stage": "measured_update_start", "update": 2}},
         True, False, -15, "TIMEOUT"),
        ("actual_oom", {"status": "OOM"}, False, False, 3, "OOM"),
        ("inconsistent_exit", {"status": "PASS"}, False, False, 1, "REJECTED"),
    )
    with tempfile.TemporaryDirectory(prefix="r18-parent-test-") as temporary:
        root = Path(temporary)
        for name, child_result, timed_out, cleanup_failed, code, expected in cases:
            case_root = root / name
            target = case_root / "gpu_check"
            target.mkdir(parents=True)
            filename = "progress.json" if name == "partial_progress" else "result.json"
            child_path = target / filename
            child_text = json.dumps(child_result)
            child_path.write_text(child_text, encoding="utf-8")
            watched = {"process_returncode": code, "timed_out": timed_out,
                       "elapsed_seconds": 300., "last_output": "Synthetic test stage",
                       "pid": 1234567, "cleanup_failed": cleanup_failed,
                       "stdout_closed": True, "timeout_seconds": 300.}

            def bounded(command, **kwargs):
                assert kwargs["timeout_seconds"] == 300.
                assert kwargs["heartbeat_seconds"] == 15.
                assert "_worker" in command
                assert "native256" not in command
                assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
                return watched

            args = SimpleNamespace(cpu_threads=4)
            result = None
            with patch.object(run_check, "run_bounded_process", side_effect=bounded):
                try:
                    result = run_check.execute_gpu(args, case_root, root / "source_workcopy", lambda _: None)
                except RuntimeError:
                    assert expected == "REJECTED", name
                else:
                    assert expected != "REJECTED", name
                    assert result["status"] == expected, (name, result)
                    assert result["eligible_as_pilot_check"] is False
                    persisted = json.loads((target / "watchdog_result.json").read_text(encoding="utf-8"))
                    assert persisted == result
                    if name == "partial_progress":
                        assert result["step_seconds"] == [1.]
                        assert result["last_progress"] == child_result["last_progress"]
            assert child_path.read_text(encoding="utf-8") == child_text
            outcomes[name] = expected
    return {"status": "PASS", "outcomes": outcomes,
            "deadline_seconds": 300, "child_reports_preserved": True,
            "partial_step_preserved": True, "diagnostic_never_unlocks_pilot": True}


def _valid_metrics():
    return {"samples": 256, "optimizer_updates": 1, "mode": "serial_dp4",
            "chunk_sizes": [64] * 4, "forwards": 20, "replicas_used": 4,
            "by_bit": {str(bit): {"loss": 6.9, "samples": 256,
                                  "top1_correct": 0, "top5_correct": 1}
                       for bit in (1, 2, 4, 8, 32)}}


def _valid_bn_snapshot(update=0):
    counters = {"shared_" + str(index): 5 * update for index in range(4)}
    counters.update({f"branch_{layer}.bn_dict.{bit}": update
                     for layer in range(16) for bit in (1, 2, 4, 8, 32)})
    return counters


def _contract_tests():
    import gpu_worker
    import support as s

    s.verify_fixed_contract()
    expected = {"bits": [1, 2, 4, 8, 32], "global_batch": 256,
                "reference_replicas": 4, "bn_shards": [64] * 4,
                "execution": "serial_dp4", "optimizer": "SGD", "lr": .3,
                "momentum": .9, "weight_decay": 1e-5, "amp": False,
                "fp32_precision": "ieee", "cudnn_benchmark": True,
                "warmup_updates": 1, "measured_updates": 2,
                "gpu_timeout_seconds": 300, "dataset_required": False,
                "training_authorized": False, "eligible_as_pilot_check": False}
    for key, value in expected.items():
        assert s.CONFIG[key] == value, (key, s.CONFIG[key], value)
    with patch.dict(s.CONFIG, {"global_batch": 128}):
        _must_reject(s.verify_fixed_contract, "changed_global_batch")
    with patch.dict(s.CONFIG, {"lr": .5}):
        _must_reject(s.verify_fixed_contract, "changed_learning_rate")
    optimized = subprocess.run(
        [sys.executable, "-B", "-O", "-c",
         "import sys\nsys.path.insert(0, sys.argv[1])\nimport support\n"
         "try:\n support.verify_fixed_contract()\n"
         "except RuntimeError as exc:\n print('REJECTED: ' + str(exc))\n"
         "else:\n raise SystemExit(5)\n", str(s.ROOT)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10.)
    assert optimized.returncode == 0 and optimized.stdout.startswith("REJECTED: "), optimized
    rejected = []
    gpu_worker.check_metrics(_valid_metrics())
    for key, value in (("samples", 64), ("optimizer_updates", 5),
                       ("mode", "native256"), ("chunk_sizes", [32] * 8),
                       ("forwards", 5), ("replicas_used", 1)):
        bad = _valid_metrics()
        bad[key] = value
        rejected.append(_must_reject(lambda: gpu_worker.check_metrics(bad), "metrics_" + key))
    for name, mutate in (
            ("missing_precision", lambda m: m["by_bit"].pop("1")),
            ("nan_loss", lambda m: m["by_bit"]["2"].update(loss=float("nan"))),
            ("bad_topk", lambda m: m["by_bit"]["4"].update(top1_correct=2, top5_correct=1))):
        bad = _valid_metrics()
        mutate(bad)
        rejected.append(_must_reject(lambda: gpu_worker.check_metrics(bad), name))
    bn_report = gpu_worker.check_bn_updates(_valid_bn_snapshot(), _valid_bn_snapshot(1))
    assert bn_report["shared_bn_modules"] == 4 and bn_report["switchable_bn_bank_modules"] == 80
    bad = _valid_bn_snapshot(1)
    bad["branch_0.bn_dict.32"] = 4
    rejected.append(_must_reject(
        lambda: gpu_worker.check_bn_updates(_valid_bn_snapshot(), bad), "bn_kept_all_four_replicas"))
    bad = _valid_bn_snapshot(1)
    bad["shared_0"] = 1
    rejected.append(_must_reject(
        lambda: gpu_worker.check_bn_updates(_valid_bn_snapshot(), bad), "shared_bn_only_one_precision"))
    return {"status": "PASS", "fixed_recipe_checked": expected,
            "python_optimized_mode_rejected": True,
            "rejected_invalid_runtime_evidence": rejected,
            "bn_shared_and_switched_update_semantics": bn_report}


def _optimizer_wiring_test():
    """Exercise the actual helper's arguments without allocating a real model."""
    import gpu_worker

    calls = []
    parameters, optimizer = object(), object()
    model = SimpleNamespace(parameters=lambda: parameters)
    soft = SimpleNamespace()

    def model_to(device):
        calls.append(("model_to", device))
        return model

    def soft_to(device):
        calls.append(("soft_to", device))
        return soft

    model.to, soft.to = model_to, soft_to
    sources = {"anyprecision": Path("original-quantizers"), "truncquant": Path("topology")}
    module = ModuleType("resnet18_model")

    def build_model(original, topology, *, num_classes):
        assert original == sources["anyprecision"] and topology == sources["truncquant"]
        assert num_classes == 1000
        calls.append(("build_model", num_classes))
        return model

    def build_soft_loss(original):
        assert original == sources["anyprecision"]
        calls.append(("build_original_soft_loss",))
        return soft

    def sgd(params, **kwargs):
        assert params is parameters
        assert kwargs == {"lr": .3, "momentum": .9, "weight_decay": 1e-5}
        calls.append(("SGD", kwargs))
        return optimizer

    module.build_model, module.build_soft_loss = build_model, build_soft_loss
    torch = ModuleType("torch")
    torch.optim = SimpleNamespace(SGD=sgd)
    with patch.dict(sys.modules, {"torch": torch, "resnet18_model": module}):
        actual = gpu_worker.model_and_optimizer(sources, "cuda:0")
    assert actual == (model, optimizer, soft)
    assert ("model_to", "cuda:0") in calls and ("soft_to", "cuda:0") in calls
    return {"status": "PASS", "original_quantizers_and_soft_loss_selected": True,
            "classes": 1000, "optimizer": "SGD", "lr": .3, "momentum": .9, "weight_decay": 1e-5}


def _worker_case(failure=None):
    import gpu_worker
    import support as s

    trace = []
    cuda = FakeCUDA(trace)
    torch = ModuleType("torch")
    torch.cuda, torch.float32 = cuda, "mock-float32"
    x, y = object(), object()

    def randn(*shape, **kwargs):
        assert shape == (256, 3, 224, 224)
        assert kwargs == {"device": "cuda", "dtype": torch.float32}
        return x

    def randint(*args, **kwargs):
        assert args == (0, 1000, (256,)) and kwargs == {"device": "cuda"}
        return y

    torch.randn, torch.randint = randn, randint
    initial_config = copy.deepcopy(s.CONFIG)
    configure_calls = []
    sources = {"anyprecision": Path("mock-original"), "truncquant": Path("mock-topology")}

    def verify_source_copy(source_root):
        assert source_root == Path("mock-source-root")
        trace.append("verify_sources")
        if failure == "SOURCE":
            raise RuntimeError("mock changed source hash")
        return sources

    def configure(seed, threads):
        trace.append("configure_cuda")
        configure_calls.append((seed, threads))
        cuda.initialized = True
        return {"seed": seed, "threads": threads, "cudnn_benchmark": True,
                "AMP": False, "fp32_precision": "ieee",
                "device": {"name": "MOCK ONLY", "total_memory_bytes": 1000}}

    json_write = s.json_write

    def write_json(path, value):
        trace.append("persist_" + path.name)
        json_write(path, value)

    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 16)])
    optimizer, soft = object(), object()

    def make_model(actual_sources, device):
        assert actual_sources == sources and device == "cuda:0"
        return model, optimizer, soft

    model_module = ModuleType("resnet18_model")
    model_module.model_report = lambda m: {"mock_model": m is model}
    engine = ModuleType("engine")
    engine_calls = []

    def train_step(m, o, inputs, labels, loss, **kwargs):
        assert m is model and o is optimizer and inputs is x and labels is y and loss is soft
        assert kwargs["mode"] == "serial_dp4"
        engine_calls.append({"validate": kwargs["validate"], "progress": "progress" in kwargs})
        trace.append("train_warmup" if kwargs["validate"] else "train_measured")
        call = len(engine_calls)
        if call == 3 and failure in ("OOM", "FAIL"):
            raise FakeOOM("mock OOM") if failure == "OOM" else RuntimeError("mock update failure")
        if kwargs["validate"]:
            for stage in ("forward_start", "forward_done", "backward_start", "backward_done"):
                kwargs["progress"]({"stage": stage, "bit": 32, "shard": 1, "shards": 4})
        cuda.allocate((500, 600, 550)[call - 1], (150, 180, 180)[call - 1])
        return _valid_metrics()

    engine.train_step = train_step
    clocks = []

    def clock():
        trace.append("clock")
        clocks.append(len(clocks) + 1)
        return float(clocks[-1])

    def bn_snapshot(m):
        assert m is model
        trace.append("bn_query")
        return _valid_bn_snapshot(len(engine_calls))

    def finite_state(m, o):
        assert m is model and o is optimizer
        trace.append("final_finite_check")
        if failure == "FINITE":
            raise RuntimeError("mock nonfinite final state")

    with tempfile.TemporaryDirectory(prefix="r18-worker-test-") as temporary:
        out = Path(temporary) / "worker"
        args = SimpleNamespace(worker_out=out, source_root=Path("mock-source-root"), cpu_threads=4)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "torch": torch, "engine": engine, "resnet18_model": model_module}))
            for owner, name, replacement in (
                    (s, "verify_source_copy", verify_source_copy), (s, "configure", configure),
                    (s, "json_write", write_json), (s, "Log", FileLog),
                    (gpu_worker, "model_and_optimizer", make_model),
                    (gpu_worker, "bn_counter_snapshot", bn_snapshot),
                    (gpu_worker, "assert_finite_state", finite_state),
                    (gpu_worker.time, "perf_counter", clock)):
                stack.enter_context(patch.object(owner, name, replacement))
            code = gpu_worker.run_worker(args)
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert s.CONFIG == initial_config, "The diagnostic mutated the recipe."
        assert result["eligible_as_pilot_check"] is False
        assert result["exact_original_paper_reproduction_claim"] is False
        assert result["training_accuracy_reported"] is False
        assert result["optimizer"] == {"name": "SGD", "lr": .3, "momentum": .9, "weight_decay": 1e-5}
        assert result["precision_bits"] == [1, 2, 4, 8, 32]
        assert result["batchnorm_shard_sizes"] == [64] * 4
        assert not list(out.rglob("*.pt")) and not list(out.rglob("*.pth"))
        if failure == "SOURCE":
            assert code == 1 and result["status"] == "FAIL"
            assert not engine_calls and not configure_calls and not cuda.initialized
            assert "memory_query" not in trace
            assert result["step_seconds"] == [] and result["optimizer_updates_completed"] == 0
            return {"status": "PASS", "mock_worker_status": "FAIL", "source_failure_precedes_cuda": True}

        progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        assert configure_calls == [(20261009, 4)]
        assert result["runtime"]["cudnn_benchmark"] is True
        assert engine_calls == [{"validate": True, "progress": True},
                               {"validate": False, "progress": False},
                               {"validate": False, "progress": False}]
        assert result["peak_allocated_bytes"] == 600 and result["peak_reserved_bytes"] == 700
        assert result["memory_updates"][0]["peak_allocated_bytes"] == 500
        assert result["memory_updates"][1]["peak_allocated_bytes"] == 600
        if failure in ("OOM", "FAIL"):
            assert result["status"] == failure and code == (3 if failure == "OOM" else 1)
            assert result["step_seconds"] == [1.] and progress["step_seconds"] == [1.]
            assert result["optimizer_updates_completed"] == 2
            assert result["memory_updates"][-1]["completed"] is False
            assert result["last_progress"] == progress["last_progress"]
            assert result["last_progress"]["stage"] == "measured_update_start"
        else:
            assert result["step_seconds"] == [1., 1.] and progress["step_seconds"] == [1., 1.]
            assert result["optimizer_updates_completed"] == 3
            assert result["memory_updates"][2]["peak_allocated_bytes"] == 550
            if failure == "FINITE":
                assert result["status"] == "FAIL" and code == 1
                assert not result.get("finite_losses_and_gradients", False)
            else:
                assert result["status"] == "PASS" and code == 0
                assert result["model_forward_backward_pairs"] == 60
                assert result["logical_precision_passes"] == 15
                assert result["finite_losses_and_gradients"] is True
        timer_indices = [i for i, item in enumerate(trace) if item == "clock"]
        for start in range(2, len(timer_indices) - 1, 2):
            inside = trace[timer_indices[start] + 1:timer_indices[start + 1]]
            assert inside == ["train_measured", "sync"], inside
        first_measurement = trace.index("train_measured")
        assert "persist_progress.json" in trace[:first_measurement], "Warmup must be persisted before measurement."
        return {"status": "PASS", "mock_worker_status": result["status"],
                "recipe_unchanged": True, "timing_excludes_boundary_telemetry": True,
                "partial_times_preserved": True, "cumulative_peaks_preserved": True,
                "all_three_update_attempts": True,
                "final_state_check_executed": "final_finite_check" in trace,
                "invalid_final_state_rejected": failure == "FINITE"}


def run_tests(package_root=None):
    assert __debug__, "Run without -O; these tests require assertions."
    import support as s

    return {"status": "PASS", "device": "CPU / mocked CUDA",
            "scope": "Integrity, fixed recipe, process lifetime, status gates, timing and persistence; "
                     "these mocks make no GPU speed or model-equivalence claim.",
            "sources": _source_tests(package_root or s.ROOT),
            "preflight_gating": _preflight_gating_test(),
            "output_isolation": _output_isolation_test(),
            "cli": _cli_tests(), "contract": _contract_tests(),
            "optimizer_wiring": _optimizer_wiring_test(),
            "worker": {name: _worker_case(failure) for name, failure in (
                ("success", None), ("partial_failure", "FAIL"), ("partial_oom", "OOM"),
                ("invalid_final_state", "FINITE"), ("source_failure", "SOURCE"))},
            "parent_outcomes": _parent_outcome_tests(),
            "watchdog": _watchdog_tests()}


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
