"""Bound a real-data worker and its DataLoader descendants.

Windows uses a Job Object with kill-on-close. A gated launcher is assigned to
the job before it can start the requested worker, so there is no assignment
race with DataLoader creation. POSIX uses a separate session/process group.
Only the processes started by this call are stopped.

Windows API basis: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""
from __future__ import annotations

import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time


_IS_WINDOWS = os.name == "nt"
_LAUNCHER = (
    "import subprocess,sys\n"
    "if sys.stdin.buffer.read(1) != b'1': sys.exit(125)\n"
    "child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL)\n"
    "sys.exit(child.wait())\n"
)


class _WindowsJob:
    """Small ctypes binding; importing this module does not load Windows APIs."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in (
                "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
            _fields_ += [(name, wintypes.DWORD) for name in (
                "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                     ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.ctypes, self.kernel, self.Accounting = ctypes, kernel, Accounting
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def active_processes(self):
        info = self.Accounting()
        if not self.kernel.QueryInformationJobObject(
                self.handle, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return int(info.ActiveProcesses)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self.kernel.CloseHandle(handle):
                raise self.ctypes.WinError(self.ctypes.get_last_error())


def _wait_direct(process, grace):
    try:
        process.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
            return True
        except subprocess.TimeoutExpired:
            return False


def _posix_group_live(pgid):
    """Linux /proc distinguishes dead, unreaped descendants from live ones."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    # Some hosted runtimes translate PIDs while exposing a host /proc. Never
    # compare those unrelated numbers or mistake a host process for our child.
    native_proc = False
    try:
        with open("/proc/self/stat", encoding="utf-8") as stream:
            native_proc = int(stream.read().split("(", 1)[0]) == os.getpid()
    except (OSError, ValueError):
        pass
    if native_proc:
        observed_proc = False
        try:
            with os.scandir("/proc") as entries:
                for entry in entries:
                    if not entry.name.isdigit():
                        continue
                    try:
                        with open(os.path.join(entry.path, "stat"), encoding="utf-8") as stream:
                            raw = stream.read()
                        # comm may contain spaces or parentheses; fields follow its last ')'.
                        fields = raw[raw.rfind(")") + 2:].split()
                        state, group = fields[0], int(fields[2])
                        observed_proc = True
                        if group == pgid and state not in ("Z", "X"):
                            return True
                    except (OSError, ValueError, IndexError):
                        continue
            if observed_proc:
                return False
        except OSError:
            pass
    return True


class _PosixTree:
    kind = "posix_process_group"

    def __init__(self, process):
        self.pgid = process.pid

    def stop(self, process, grace, force=False):
        del force
        detail = {"method": self.kind, "group_id": self.pgid, "verified": False}
        errors = []
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not _posix_group_live(self.pgid):
                break
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                break
            except OSError as exc:
                errors.append(str(exc))
            deadline = time.monotonic() + grace
            while _posix_group_live(self.pgid) and time.monotonic() < deadline:
                process.poll()  # Reap the immediate child when possible.
                time.sleep(.01)
        direct_done = _wait_direct(process, grace)
        detail["verified"] = direct_done and not _posix_group_live(self.pgid)
        if errors:
            detail["errors"] = errors
        return detail

    def close(self):
        pass


class _WindowsTree:
    kind = "windows_job_object"

    def __init__(self, job):
        self.job = job

    def stop(self, process, grace, force=False):
        detail = {"method": self.kind, "verified": False,
                  "job_assigned_before_worker_start": True}
        errors = []
        # The job remains authoritative even if the launcher has already exited,
        # in which case taskkill /T alone cannot reliably discover descendants.
        if force and process.poll() is None:
            executable = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
            try:
                killed = subprocess.run([executable, "/PID", str(process.pid), "/T", "/F"],
                                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                                        timeout=max(1., grace), check=False)
                detail["taskkill_returncode"] = killed.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                detail["taskkill_error"] = str(exc)
        try:
            if self.job.active_processes():
                self.job.terminate()
            deadline = time.monotonic() + grace
            while self.job.active_processes() and time.monotonic() < deadline:
                time.sleep(.01)
            detail["job_active_processes_after_stop"] = self.job.active_processes()
            detail["verified"] = detail["job_active_processes_after_stop"] == 0
        except OSError as exc:
            errors.append(str(exc))
        detail["verified"] = _wait_direct(process, grace) and detail["verified"]
        if errors:
            detail["errors"] = errors
        return detail

    def close(self):
        self.job.close()


def _start_process(command, env):
    options = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
    if not _IS_WINDOWS:
        process = subprocess.Popen(command, start_new_session=True, **options)
        return process, _PosixTree(process)
    job = _WindowsJob()
    process = None
    try:
        # This launcher cannot create the worker until it has joined our job.
        process = subprocess.Popen([sys.executable, "-u", "-c", _LAUNCHER, *map(str, command)],
                                   stdin=subprocess.PIPE, creationflags=0x00000200, **options)
        job.assign(process)
        process.stdin.write("1")
        process.stdin.flush()
        process.stdin.close()
        return process, _WindowsTree(job)
    except BaseException:
        if process is not None:
            # If assignment failed the launcher is still blocked on the gate.
            try:
                process.kill()
            except ProcessLookupError:
                pass
            _wait_direct(process, 5.)
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        job.close()
        raise


def run_bounded_process(command, *, timeout_seconds, log, heartbeat_seconds=15.,
                        label="ImageNet-proef", env=None, terminate_grace_seconds=5.):
    """Stream output, enforce an absolute deadline, and always clean the tree.

    A heartbeat confirms only that the parent is waiting. On interruption or a
    logging exception, all owned descendants are stopped before re-raising.
    A normal worker exit also closes the job/group, preventing orphan loaders.
    """
    for value in (timeout_seconds, heartbeat_seconds, terminate_grace_seconds):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Watchdog intervals must be positive and finite.")
    started = time.monotonic()
    process, tree = _start_process(command, env)
    messages = queue.Queue(maxsize=4096)
    eof = object()
    dropped = 0

    def emit(message):
        nonlocal dropped
        try:
            messages.put_nowait(message)
        except queue.Full:
            try:
                messages.get_nowait()
                dropped += 1
            except queue.Empty:
                pass
            try:
                messages.put_nowait(message)
            except queue.Full:
                dropped += 1

    def read_output():
        try:
            for line in process.stdout:
                emit(line.rstrip("\r\n"))
        except (OSError, ValueError) as exc:
            emit("[stdout reader] " + str(exc))
        finally:
            emit(eof)

    reader = threading.Thread(target=read_output, daemon=True, name="realdata-probe-stdout")
    try:
        reader.start()
    except BaseException:
        try:
            tree.stop(process, terminate_grace_seconds, force=True)
        finally:
            tree.close()
            process.stdout.close()
        raise
    deadline = started + timeout_seconds
    next_heartbeat = started + heartbeat_seconds
    last_output = "Nog geen uitvoer van het proefproces."
    timed_out, stdout_closed = False, False
    cleanup = None

    def consume(message):
        nonlocal last_output, stdout_closed
        if message is eof:
            stdout_closed = True
        else:
            last_output = message
            log(message)

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                cleanup = tree.stop(process, terminate_grace_seconds, force=True)
                phrase = "procesboom gestopt" if cleanup["verified"] else "stop niet volledig bevestigd"
                log(f"{label}: TIMEOUT na {timeout_seconds:g} s; {phrase}. Dit bewijst geen geheugen-OOM.")
                break
            if process.poll() is not None:
                cleanup = tree.stop(process, terminate_grace_seconds)
                break
            if now >= next_heartbeat:
                log(f"{label}: wacht {now - started:.0f} s / limiet {timeout_seconds:g} s; "
                    f"laatste uitvoer: {last_output} (wachtmelding, geen bewijs van GPU-voortgang)")
                next_heartbeat = now + heartbeat_seconds
            try:
                consume(messages.get(timeout=min(.2, max(.001, deadline - now))))
            except queue.Empty:
                pass
        reader.join(timeout=.5)
        for _ in range(4097):
            try:
                consume(messages.get_nowait())
            except queue.Empty:
                break
    except BaseException:
        cleanup = tree.stop(process, terminate_grace_seconds, force=True)
        if not cleanup["verified"]:
            raise RuntimeError(f"De eigen procesboom van PID {process.pid} kon niet volledig worden gestopt. "
                               "Controleer dit PID en de bijbehorende dataloader-processen in Taakbeheer.")
        raise
    finally:
        tree.close()
        reader.join(timeout=.5)
        if not reader.is_alive():
            process.stdout.close()
    cleanup_failed = not bool(cleanup and cleanup.get("verified"))
    if cleanup_failed:
        log(f"Procesboom van PID {process.pid}: beëindiging niet volledig bevestigd.")
    return {"process_returncode": process.poll(), "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started, "last_output": last_output,
            "pid": process.pid, "cleanup_failed": cleanup_failed,
            "stdout_closed": stdout_closed, "timeout_seconds": timeout_seconds,
            "process_tree_control": tree.kind, "process_tree_cleanup": cleanup,
            "output_lines_dropped": dropped}
