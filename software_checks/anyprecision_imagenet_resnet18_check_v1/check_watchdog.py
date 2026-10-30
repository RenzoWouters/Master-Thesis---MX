"""Bound a GPU subprocess from a CPU-only parent, including silent CUDA stalls.

The reader thread never controls the deadline. This also works on Windows,
where select() on a subprocess stdout pipe is not supported.
"""
from __future__ import annotations

import math
import queue
import subprocess
import threading
import time


def _stop_process(process, grace_seconds):
    """Never leave a wait unbounded; return whether the process was stopped."""
    if process.poll() is not None:
        return True
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


def run_bounded_process(command, *, timeout_seconds, log, heartbeat_seconds=15.,
                        label="GPU-test", env=None, terminate_grace_seconds=5.):
    """Stream stdout, report liveness and stop after an absolute wall deadline.

    A heartbeat confirms only that the parent is waiting, not GPU progress.
    On interruption or a logging error, the child is terminated before rethrow.
    """
    for value in (timeout_seconds, heartbeat_seconds, terminate_grace_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Watchdog intervals must be positive and finite.")
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=env,
                               bufsize=1)
    messages = queue.Queue()
    eof = object()

    def read_output():
        try:
            for line in process.stdout:
                messages.put(line.rstrip("\r\n"))
        except (OSError, ValueError) as exc:
            messages.put("[stdout reader] " + str(exc))
        finally:
            messages.put(eof)

    reader = threading.Thread(target=read_output, daemon=True, name="gpu-check-stdout")
    reader.start()
    deadline = started + timeout_seconds
    next_heartbeat = started + heartbeat_seconds
    last_output = "Nog geen uitvoer van het testproces."
    timed_out = False
    cleanup_failed = False
    stdout_closed = False

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
            # Check time even if output is continuous or stdout has closed.
            if now >= deadline:
                timed_out = True
                cleanup_failed = not _stop_process(process, terminate_grace_seconds)
                stop_text = "stop niet bevestigd" if cleanup_failed else "testproces gestopt"
                log(f"{label}: TIMEOUT na {timeout_seconds:g} s; {stop_text}. "
                    "Dit bewijst geen geheugen-OOM.")
                break
            if process.poll() is not None:
                break
            if now >= next_heartbeat:
                log(f"{label}: wacht {now - started:.0f} s / limiet {timeout_seconds:g} s; "
                    f"laatste uitvoer: {last_output} "
                    "(wachtmelding, geen bewijs van GPU-voortgang)")
                next_heartbeat = now + heartbeat_seconds
            try:
                consume(messages.get(timeout=min(0.2, max(0.001, deadline - now))))
            except queue.Empty:
                pass
        reader.join(timeout=0.2)
        # A finite drain also bounds pathological output-producing children.
        for _ in range(10000):
            try:
                consume(messages.get_nowait())
            except queue.Empty:
                break
    except BaseException:
        stopped = _stop_process(process, terminate_grace_seconds)
        if not stopped:
            raise RuntimeError(f"Testproces PID {process.pid} kon niet worden gestopt; "
                               "controleer dit proces in Taakbeheer.")
        raise
    finally:
        if not reader.is_alive():
            process.stdout.close()
    if cleanup_failed:
        log(f"LET OP: testproces PID {process.pid} is mogelijk nog actief; controleer Taakbeheer.")
    return {"process_returncode": process.poll(), "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started, "last_output": last_output,
            "pid": process.pid, "cleanup_failed": cleanup_failed,
            "stdout_closed": stdout_closed, "timeout_seconds": timeout_seconds}
