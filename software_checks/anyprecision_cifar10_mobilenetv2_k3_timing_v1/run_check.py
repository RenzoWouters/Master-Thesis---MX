"""Windows-friendly, bounded MobileNetV2 feasibility check and real-data timer.

Only check/run/selftest are public commands. There is deliberately no full
training or resume command in this timing package.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import support as s
from process_watchdog import run_bounded_process


def verify_manifest():
    path = s.ROOT / "PACKAGE_MANIFEST.json"
    if not path.is_file():
        return {"status": "DEVELOPMENT", "manifest_available": False}
    entries = s.json_read(path)["files"]
    for relative, expected in entries.items():
        candidate = s.ROOT / relative
        s.require(candidate.is_file() and s.sha256(candidate) == expected,
                  "Pakketbestand ontbreekt of is gewijzigd: " + relative + ". Pak het originele ZIP-bestand opnieuw uit in een eigen map.")
    return {"status": "PASS", "verified_files": len(entries)}


def cpu_selftests(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = s.Logger(out / "cpu_selftests.log")
    report = {"status": "RUNNING", "cuda_tested": False, "dataset_downloaded": False}
    try:
        from test_model_engine import run_tests as model_tests
        from test_data_timing import run_tests as data_tests
        from test_runner import run_tests as runner_tests
        log("CPU: modelstructuur, upstream-quantisatie, vijf BN-banken en recursieve updates controleren.")
        report["model_engine"] = model_tests(progress=log)
        log("CPU: databehoud, CIFAR-selectie en tijdextrapolatie controleren.")
        report["data_timing"] = data_tests()
        log("CPU: rapportselectie, feedbackbundel en procesbegrenzing controleren.")
        report["runner"] = runner_tests()
        report["status"] = "PASS"
        log("CPU-zelftests PASS; geen CUDA- of datasettraining uitgevoerd.")
        returncode = 0
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log(report["traceback"])
        returncode = 1
    s.json_write(out / "cpu_selftests.json", report)
    return returncode


def select_approved_check(project, fingerprint, explicit=None):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        candidates = [candidate / "run_report.json" if candidate.is_dir() else candidate]
    else:
        parent = Path(project) / "mx_runs" / s.CONFIG["experiment"]
        candidates = sorted(parent.glob("*_check/run_report.json"), reverse=True)
    for path in candidates:
        try:
            report = s.json_read(path)
        except (OSError, ValueError):
            continue
        if (report.get("status") == "PASS" and report.get("command") == "check"
                and report.get("experiment") == s.CONFIG["experiment"]
                and report.get("package_fingerprint", {}).get("sha256") == fingerprint["sha256"]
                and report.get("worker_result", {}).get("status") == "PASS"
                and report["worker_result"].get("eligible_check") is True):
            return path, report
    raise RuntimeError("Geen geslaagde CUDA-check voor exact deze pakketversie gevonden. Voer eerst run_check.py check uit.")


def classify_worker(watchdog, worker):
    if watchdog.get("cleanup_failed"):
        return "FAIL", 1
    if watchdog.get("timed_out"):
        return "TIMEOUT", 124
    if watchdog.get("process_returncode") == 0 and worker.get("status") == "PASS":
        return "PASS", 0
    status = worker.get("status", "FAIL")
    return (status if status in ("OOM", "INTERRUPTED", "FAIL") else "FAIL"), 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Begrensde geheugen- en tijdproef; geen volledige training.")
    parser.add_argument("command", choices=("check", "run", "selftest", "_selftest_worker"))
    parser.add_argument("--project-root")
    parser.add_argument("--data-root", help="Map met cifar-10-batches-py, of die uitgepakte map zelf.")
    parser.add_argument("--download", action="store_true", help="Alleen bij run: sta officiële CIFAR-10-download toe als de dataset ontbreekt.")
    parser.add_argument("--approved-check", help="Optioneel specifiek geslaagd checkrapport of de resultatenmap ervan.")
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "_selftest_worker":
        if not args.out:
            parser.error("--out is required for the internal selftest worker")
        return cpu_selftests(args.out)
    if args.download and args.command != "run":
        parser.error("--download is alleen toegestaan bij run")
    project = s.project_root(args.project_root)
    out = s.new_run_directory(project, args.command)
    log = s.Logger(out / "run.log")
    fingerprint = s.package_fingerprint()
    report = {"schema_version": 1, "experiment": s.CONFIG["experiment"], "command": args.command,
              "status": "RUNNING", "run_directory": str(out), "project_root": str(project),
              "protocol": s.CONFIG, "package_fingerprint": fingerprint,
              "environment_parent": s.basic_environment(), "long_training_started": False,
              "existing_results_modified": False, "worker_result_file": "worker/worker_result.json"}
    s.json_write(out / "run_report.json", report)
    s.json_write(out / "experiment_config.json", s.CONFIG)
    log("RESULTATENMAP: " + str(out))
    log("MOBILENETV2 / CIFAR-10: K=3; batch128; W/A 32,8,4,2,1; FP32-masters; Adam; geen MX.")
    log("Gedocumenteerde reconstructie. Dit pakket voert alleen een begrensde haalbaarheidsproef uit.")
    began = time.monotonic()
    exit_code = 1
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        report["upstream_integrity"] = s.verify_upstream()
        report["package_integrity"] = verify_manifest()
        if args.command in ("check", "selftest"):
            checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=45, env=env)
            (out / "pip_check.txt").write_text(checked.stdout + checked.stderr, encoding="utf-8")
            report["pip_check"] = {"returncode": checked.returncode, "output": (checked.stdout + checked.stderr).strip()}
            log("pip check: " + report["pip_check"]["output"])
            s.require(checked.returncode == 0, "pip check meldt conflicterende pakketten.")
            cpu_command = [sys.executable, "-u", str(s.ROOT / "run_check.py"),
                           "_selftest_worker", "--out", str(out / "cpu")]
            report["cpu_watchdog"] = run_bounded_process(cpu_command, timeout_seconds=120,
                                                        label="CPU-zelftests", log=log, env=env)
            cpu_report = out / "cpu" / "cpu_selftests.json"
            report["checks"] = s.json_read(cpu_report) if cpu_report.is_file() else {}
            s.require(not report["cpu_watchdog"]["timed_out"] and
                      not report["cpu_watchdog"]["cleanup_failed"] and
                      report["cpu_watchdog"]["process_returncode"] == 0 and
                      report["checks"].get("status") == "PASS", "CPU-zelftests niet geslaagd; GPU-proef niet gestart.")
            if args.command == "selftest":
                report["status"] = "PASS"
                exit_code = 0
                return exit_code
        worker_command = [sys.executable, "-u", str(s.ROOT / "probe_worker.py"),
                          "--kind", "synthetic" if args.command == "check" else "real",
                          "--out", str(out / "worker")]
        if args.command == "run":
            approved_path, approved = select_approved_check(project, fingerprint, args.approved_check)
            report["approved_check"] = {"path": str(approved_path), "sha256": s.sha256(approved_path),
                                        "device": approved["worker_result"]["runtime"]["device"]}
            from data_support import find_data_root
            data_root = find_data_root(project, args.data_root, allow_missing=args.download)
            worker_command.extend(["--approved-report", str(approved_path), "--data-root", str(data_root)])
            if args.download:
                worker_command.append("--download")
            report["data_root"] = str(data_root)
            timeout = s.CONFIG["real_data_probe"]["timeout_seconds"]
            log("Korte realdataproef: 100 updates op 12.800 trainbeelden; daarna 1.000 evaluatiebeelden.")
        else:
            timeout = s.CONFIG["synthetic_check"]["timeout_seconds"]
            log("Synthetische CUDA-check: 1 opwarmupdate + 3 gemeten updates; elke update bevat alle vijf precisies.")
        s.json_write(out / "run_report.json", report)
        report["watchdog"] = run_bounded_process(worker_command, timeout_seconds=timeout,
                                                label="MobileNetV2-proef", log=log, env=env)
        worker_path = out / report["worker_result_file"]
        report["worker_result"] = s.json_read(worker_path) if worker_path.is_file() else {}
        report["status"], exit_code = classify_worker(report["watchdog"], report["worker_result"])
        if report["status"] != "PASS":
            report["error"] = report["worker_result"].get("error") or (
                "Harde tijdslimiet bereikt; dit is geen bewijs van een geheugen-OOM."
                if report["status"] == "TIMEOUT" else "De begrensde worker is niet geslaagd.")
            log("PROEF NIET GESLAAGD: " + report["error"])
    except KeyboardInterrupt:
        report.update(status="INTERRUPTED", error="Door gebruiker onderbroken; eigen childprocessen worden gestopt.")
        exit_code = 130
        log(report["error"])
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log("FAIL: " + str(exc))
        log(report["traceback"])
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        # Pick up partial progress even when the parent interrupted the worker.
        worker_path = out / report["worker_result_file"]
        if worker_path.is_file():
            try:
                report["worker_result"] = s.json_read(worker_path)
            except (OSError, ValueError):
                pass
        report["feedback_bundle"] = str(out / "feedback_bundle.zip")
        s.json_write(out / "run_report.json", report)
        log("EINDSTATUS: " + report["status"])
        log("Rapport: " + str(out / "run_report.json"))
        log("Deel feedback_bundle.zip uit deze map. Checkpoint en CIFAR-10 blijven lokaal.")
        s.make_feedback_bundle(out)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
