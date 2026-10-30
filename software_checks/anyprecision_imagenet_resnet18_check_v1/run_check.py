"""Short, synthetic ResNet-18 feasibility check. No dataset/training entry point."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import support as s
from check_watchdog import run_bounded_process


def parser():
    result = argparse.ArgumentParser(description="ResNet-18: korte geheugen- en snelheidscontrole, zonder ImageNet.")
    result.add_argument("command", choices=["check", "selftest", "_worker"], metavar="check|selftest")
    result.add_argument("--project-root", type=Path, default=s.default_project_root())
    result.add_argument("--output-root", type=Path, default=None)
    result.add_argument("--cpu-threads", type=int, choices=[4], default=4)
    result.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    return result


def validate_completed_gpu_result(result):
    """A zero exit code alone cannot establish a complete measurement."""
    s.require(result.get("optimizer_updates_completed") == 3,
              "GPU-PASS vereist één opwarmupdate en twee volledige gemeten updates.")
    steps = result.get("step_seconds", [])
    s.require(isinstance(steps, list) and len(steps) == 2 and
              all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                  math.isfinite(value) and value > 0 for value in steps),
              "GPU-PASS vereist twee positieve, eindige stapduren.")
    memory = result.get("memory_updates", [])
    s.require(isinstance(memory, list) and len(memory) == 3 and
              [item.get("kind") for item in memory] == ["warmup", "measured", "measured"] and
              all(item.get("status") == "PASS" for item in memory),
              "De drie voltooide geheugenmetingen ontbreken bij GPU-PASS.")
    bn_checks = result.get("bn_update_checks", [])
    s.require(isinstance(bn_checks, list) and len(bn_checks) == 3 and
              all(item.get("status") == "PASS" for item in bn_checks),
              "De drie geslaagde BatchNorm-controles ontbreken bij GPU-PASS.")
    s.require(result.get("finite_losses_and_gradients") is True,
              "De controle op eindige waarden ontbreekt bij GPU-PASS.")


def execute_gpu(args, out, sources_root, log):
    target = Path(out) / "gpu_check"
    command = [sys.executable, "-u", str(s.ROOT / "run_check.py"), "_worker",
               "--source-root", str(Path(sources_root).resolve()), "--worker-out", str(target),
               "--cpu-threads", str(args.cpu_threads)]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log("Start GPU-proces: uitsluitend ResNet-18, batch256 als vier BN64-groepen.")
    log("Harde GPU-tijdslimiet: 300 seconden inclusief opwarming en beide metingen.")
    watched = run_bounded_process(command, timeout_seconds=300., heartbeat_seconds=15.,
                                  label="ResNet-18-controle", log=log, env=environment)
    result = None
    for name in ("result.json", "progress.json"):
        path = target / name
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            break
    if result is None:
        result = {"status": "FAIL", "error": "Het GPU-proces schreef geen rapport."}
    result.update(watched)
    result["eligible_as_pilot_check"] = False
    if watched["cleanup_failed"]:
        result.update(status="FAIL", error="GPU-proces kon niet worden gestopt; controleer het gerapporteerde PID.")
    elif watched["timed_out"]:
        result.update(status="TIMEOUT", error="GPU-tijdslimiet bereikt; dit bewijst geen geheugen-OOM.",
                      peak_observation_scope="Only previously recorded observations; aborted-update final peak unknown.")
    else:
        s.require((result.get("status") == "PASS") == (watched["process_returncode"] == 0),
                  "Exitcode en GPU-rapport spreken elkaar tegen.")
        s.require(result.get("status") in ("PASS", "FAIL", "OOM"), "Onvolledig GPU-eindrapport.")
        if result["status"] == "PASS":
            validate_completed_gpu_result(result)
    target.mkdir(exist_ok=True)
    s.json_write(target / "watchdog_result.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "_worker":
        s.verify_fixed_contract()
        s.require(args.source_root is not None and args.worker_out is not None,
                  "Het interne GPU-proces vereist bron- en uitvoerpaden van de hoofdcontrole.")
        from gpu_worker import run_worker
        return run_worker(args)

    out = s.new_output(args.project_root, args.command, args.output_root)
    log = s.Log(out / "console.log")
    began = time.monotonic()
    report = {
        "schema_version": 1, "package_id": s.CONFIG["package_id"],
        "command": "resnet18_feasibility_check", "requested_command": args.command,
        "status": "RUNNING", "eligible_as_pilot_check": False,
        "started_utc": s.utc_now(), "output_directory": str(out),
        "config": s.CONFIG, "package_signature": s.package_signature(),
        "scope": "Synthetic feasibility only. No ImageNet data, epochs, saved model tensors or accuracy results.",
        "reconstruction": "TruncQuant R18 architecture with original Any-Precision quantizers/loss and documented recipe choices.",
        "baseline_comparison": "Prior R50 laptop checks are hardware context, not a controlled architecture-only comparison.",
    }
    log("RESULTATENMAP: " + str(out))
    log("RESNET-18-CONTROLE: batch256; vier groepen64; W/A 1,2,4,8,32; FP32; SGD lr=0.3; geen MX.")
    log("Gedocumenteerde reconstructie; geen claim van exacte reproductie van het oorspronkelijke ResNet-18-resultaat.")
    try:
        s.verify_fixed_contract()
        s.copy_context(out)
        s.verify_sources()
        report["environment"] = s.check_environment(log)
        sources, report["source"] = s.prepare_sources(out)
        s.configure(s.CONFIG["seed"], cuda=False, threads=args.cpu_threads)
        import engine
        import test_engine
        import test_resnet18
        import test_diagnostic
        report["selftests"] = {}
        log("CPU-controle: onafhankelijke vier-replica-oracle voor gradients, BN en SGD.")
        report["selftests"]["engine"] = test_engine.run_selftests()
        s.json_write(out / "run_report.json", report)
        log("CPU-controle: ResNet-18-architectuur en oorspronkelijke quantizers tegenover de ongewijzigde bron.")
        report["selftests"]["resnet18"] = test_resnet18.run_tests(
            sources["anyprecision"], sources["truncquant"], engine_module=engine)
        s.json_write(out / "run_report.json", report)
        log("CPU-controle: bronintegriteit, meetgrenzen en automatisch stoppen van het testproces.")
        report["selftests"]["diagnostic"] = test_diagnostic.run_tests()
        s.json_write(out / "run_report.json", report)
        gc.collect()
        log("CPU-zelftests geslaagd.")
        if args.command == "selftest":
            report["status"] = "PASS"
            log("Alleen CPU-zelftests uitgevoerd; geen GPU-meting.")
        else:
            report["gpu_check"] = execute_gpu(args, out, out / "source_workcopy", log)
            report["status"] = report["gpu_check"]["status"]
            baseline = json.loads((s.ROOT / "baseline_results.json").read_text(encoding="utf-8"))["records"][1]
            packages = ("torch", "torchvision", "numpy", "Pillow")
            report["comparison_context"] = {
                "baseline_run": baseline["run_name"], "baseline_model": "ResNet-50",
                "candidate_model": "ResNet-18 reconstructed", "baseline_lr": 0.5, "candidate_lr": 0.3,
                "same_package_versions": all(report["environment"]["packages"][name] ==
                                             baseline["environment"]["packages"][name] for name in packages),
                "same_gpu": report["gpu_check"].get("device") == baseline["runtime"]["device"],
                "note": "Non-concurrent observations; architecture and LR differ. Free memory is recorded separately."}
            log("Controle afgerond. Beoordeel geheugenruimte en snelheid voordat een ImageNet-pilot wordt voorbereid.")
    except KeyboardInterrupt:
        report.update(status="INTERRUPTED", error="Controle onderbroken; er loopt geen ImageNet-training.")
        log(report["error"])
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log("FAIL: " + str(exc))
        log(report["traceback"])
    finally:
        report["ended_utc"] = s.utc_now()
        report["wall_seconds"] = time.monotonic() - began
        s.json_write(out / "run_report.json", report)
        log("EINDSTATUS: " + report["status"])
        log("Deel feedback_bundle.zip uit deze map. Deze controle start geen ImageNet-training.")
        log.close()
        s.feedback_bundle(out)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
