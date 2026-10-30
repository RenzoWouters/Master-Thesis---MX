"""Bounded process-tree tests; no CUDA, model, or dataset is used.

Native process-tree tests run on the host OS. Windows API routing additionally
has isolated mocks; those mocks do not establish native Windows execution.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import process_watchdog as w


def _pid_live(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            status = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                raise ctypes.WinError(ctypes.get_last_error())
            return status.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    path = Path("/proc") / str(pid) / "stat"
    native_proc = False
    try:
        native_proc = int(Path("/proc/self/stat").read_text().split("(", 1)[0]) == os.getpid()
    except (OSError, ValueError):
        pass
    if native_proc:
        try:
            raw = path.read_text(encoding="utf-8")
            return raw[raw.rfind(")") + 2:].split()[0] not in ("Z", "X")
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _tree_program(after):
    return (
        "import os,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print('TREE_CHILD_PID='+str(child.pid), flush=True)\n" + after
    )


def _case(name, program, *, timed_out=False, code=0, interrupt=None, requires_descendant=False):
    lines, descendants, started = [], [], []
    actual_start = w._start_process
    timeout = (1.5 if os.name == "nt" else .4) if timed_out else 5.

    def start(command, env):
        result = actual_start(command, env)
        started.append(result[0])
        return result

    def logger(line):
        lines.append(line)
        if line.startswith("TREE_CHILD_PID="):
            descendants.append(int(line.partition("=")[2]))
            if interrupt == "keyboard":
                raise KeyboardInterrupt("Synthetic test interrupt")
            if interrupt == "logging":
                raise RuntimeError("Synthetic log failure")

    began = time.monotonic()
    caught = None
    result = None
    with patch.object(w, "_start_process", side_effect=start):
        try:
            result = w.run_bounded_process(
                [sys.executable, "-u", "-c", program], timeout_seconds=timeout,
                heartbeat_seconds=.1, terminate_grace_seconds=.5, log=logger, label=name)
        except (KeyboardInterrupt, RuntimeError) as exc:
            caught = exc
    if interrupt:
        expected = KeyboardInterrupt if interrupt == "keyboard" else RuntimeError
        assert isinstance(caught, expected), (name, caught)
        assert str(caught).startswith("Synthetic"), (name, caught)
    else:
        assert caught is None, (name, caught)
        assert result["timed_out"] is timed_out, (name, result)
        assert not result["cleanup_failed"], (name, result)
        assert result["process_tree_cleanup"]["verified"] is True
        if not timed_out:
            assert result["process_returncode"] == code, (name, result)
        else:
            assert timeout <= result["elapsed_seconds"] < timeout + 6., (name, result)
    assert len(started) == 1 and started[0].poll() is not None, (name, started)
    if requires_descendant:
        assert len(descendants) == 1, (name, lines)
    assert all(not _pid_live(pid) for pid in descendants), (name, descendants)
    assert time.monotonic() - began < timeout + 6.
    return {"status": "PASS", "timed_out": timed_out,
            "interruption": interrupt, "observed_descendants": len(descendants),
            "owned_processes_stopped": True,
            "seconds": time.monotonic() - began,
            "native_tree_control": "windows_job_object" if os.name == "nt" else "posix_process_group"}


class _FakePipe:
    def __init__(self, events):
        self.events, self.closed = events, False

    def write(self, value):
        self.events.append(("gate_write", value))

    def flush(self):
        self.events.append("gate_flush")

    def close(self):
        self.closed = True
        self.events.append("pipe_closed")


class _FakeProcess:
    pid = 345678
    _handle = 12345

    def __init__(self, events, returncode=None):
        self.events, self.returncode = events, returncode
        self.stdin, self.stdout = _FakePipe(events), _FakePipe(events)

    def poll(self):
        return self.returncode

    def kill(self):
        self.events.append("direct_kill")
        self.returncode = 1

    def wait(self, timeout):
        self.events.append(("direct_wait", timeout))
        if self.returncode is None:
            self.returncode = 1
        return self.returncode


class _FakeJob:
    def __init__(self, events, *, reject=False, active=3):
        self.events, self.reject, self.active = events, reject, active

    def assign(self, process):
        self.events.append(("assign", process.pid))
        if self.reject:
            raise OSError("Synthetic job assignment failure")

    def active_processes(self):
        self.events.append(("query_active", self.active))
        return self.active

    def terminate(self):
        self.events.append("terminate_job")
        self.active = 0

    def close(self):
        self.events.append("close_job")


def _windows_routing_tests():
    cases = {}
    events, popen_calls = [], []
    fake = _FakeProcess(events)
    job = _FakeJob(events)

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return fake

    with patch.object(w, "_IS_WINDOWS", True), patch.object(w, "_WindowsJob", return_value=job), \
            patch.object(w.subprocess, "Popen", side_effect=popen):
        process, tree = w._start_process(["python.exe", "run_pilot.py", "_worker"], {"TEST": "1"})
    assert process is fake and isinstance(tree, w._WindowsTree)
    assert events.index(("assign", fake.pid)) < events.index(("gate_write", "1"))
    command, options = popen_calls[0]
    assert command[:3] == [sys.executable, "-u", "-c"] and command[3] == w._LAUNCHER
    assert command[4:] == ["python.exe", "run_pilot.py", "_worker"]
    assert options["creationflags"] == 0x00000200
    assert options["env"] == {"TEST": "1"}
    assert options["stdin"] is subprocess.PIPE
    assert fake.stdin.closed
    cases["job_assignment_precedes_worker_gate"] = "PASS"

    calls = []

    def taskkill(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    with patch.object(w.subprocess, "run", side_effect=taskkill):
        detail = tree.stop(fake, .5, force=True)
    assert detail["verified"] and detail["job_active_processes_after_stop"] == 0
    assert calls[0][0][1:] == ["/PID", str(fake.pid), "/T", "/F"]
    assert calls[0][0][0].lower().endswith("taskkill.exe")
    assert calls[0][1]["timeout"] >= 1.
    assert "terminate_job" in events
    tree.close()
    assert "close_job" in events
    cases["timeout_targets_only_owned_PID_tree_and_job"] = "PASS"

    events = []
    fake = _FakeProcess(events, returncode=0)
    job = _FakeJob(events, active=2)
    tree = w._WindowsTree(job)
    with patch.object(w.subprocess, "run") as taskkill:
        detail = tree.stop(fake, .5)
    assert not taskkill.called and detail["verified"]
    assert "terminate_job" in events
    tree.close()
    cases["normal_launcher_exit_still_cleans_descendants"] = "PASS"

    events = []
    fake, job = _FakeProcess(events), _FakeJob(events, reject=True)
    with patch.object(w, "_IS_WINDOWS", True), patch.object(w, "_WindowsJob", return_value=job), \
            patch.object(w.subprocess, "Popen", return_value=fake):
        try:
            w._start_process(["python.exe", "worker.py"], None)
        except OSError as exc:
            assert str(exc) == "Synthetic job assignment failure"
        else:
            raise AssertionError("Failed job assignment must stop before the worker gate.")
    assert not any(isinstance(item, tuple) and item[0] == "gate_write" for item in events)
    assert "direct_kill" in events and "close_job" in events
    cases["failed_assignment_never_starts_worker"] = "PASS"
    return {"status": "PASS", "scope": "Mock routing only; native Windows verification occurs on the laptop.",
            "cases": cases}


def run_tests():
    if not __debug__:
        raise RuntimeError("Run watchdog tests without python -O.")
    began = time.monotonic()
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    native = {}
    try:
        native["successful_exit"] = _case("successful", "print('done', flush=True)")
        native["normal_exit_with_descendant"] = _case(
            "normal-descendant", _tree_program("sys.exit(0)\n"), requires_descendant=True)
        native["nonzero_exit_with_descendant"] = _case(
            "failed-descendant", _tree_program("sys.exit(7)\n"), code=7, requires_descendant=True)
        native["quiet_timeout_with_descendant"] = _case(
            "quiet-tree", _tree_program("time.sleep(30)\n"), timed_out=True, requires_descendant=True)
        native["continuous_output_timeout"] = _case(
            "continuous", "import time\nwhile True:\n print('tick', flush=True)\n time.sleep(.002)",
            timed_out=True)
        native["partial_line_timeout"] = _case(
            "partial", "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(30)",
            timed_out=True)
        native["eof_without_exit_timeout"] = _case(
            "eof", "import os,time; os.close(1); os.close(2); time.sleep(30)", timed_out=True)
        native["interrupt_with_descendant"] = _case(
            "interrupt-tree", _tree_program("time.sleep(30)\n"),
            interrupt="keyboard", requires_descendant=True)
        native["logging_error_with_descendant"] = _case(
            "logging-tree", _tree_program("time.sleep(30)\n"),
            interrupt="logging", requires_descendant=True)
        assert unrelated.poll() is None, "An unrelated process must not be terminated."
    finally:
        if unrelated.poll() is None:
            unrelated.kill()
        unrelated.wait(timeout=5.)
    return {"status": "PASS", "scope": "Process orchestration only; no CUDA or training.",
            "host": sys.platform, "native_tests": native,
            "unrelated_process_preserved": True,
            "windows_routing": _windows_routing_tests(), "seconds": time.monotonic() - began}


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
