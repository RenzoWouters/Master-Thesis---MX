"""One bounded, synthetic cuDNN-backend diagnostic; no ImageNet training CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "diagnostic_config.json").read_text(encoding="utf-8"))
NOTEBOOK = "Resultaten_backend_check.ipynb"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


class Log:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="\n")

    def __call__(self, text):
        text = str(text)
        print(text, flush=True)
        self.stream.write(text + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def parser():
    p = argparse.ArgumentParser(description="Korte ResNet-50-diagnose: cuDNN benchmark uit, batch256/BN64 behouden.")
    p.add_argument("command", choices=["check", "selftest", "_worker"])
    p.add_argument("--reference-root", type=Path,
                   default=ROOT.parent / "anyprecision_imagenet_resnet50_pilot2_v1_1",
                   help="Bestaande ongewijzigde v1_1-map, standaard naast deze map.")
    p.add_argument("--project-root", type=Path,
                   default=ROOT.parent.parent if ROOT.parent.name.lower() == "software_checks" else Path.cwd())
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    return p


def verify_reference(path):
    """Read-only verification before any code is imported from the sibling."""
    path = Path(path).resolve()
    require(path.is_dir(), "De bestaande v1_1-map ontbreekt: " + str(path) +
            ". Plaats beide uitgepakte mappen naast elkaar onder software_checks.")
    expected = json.loads((ROOT / "reference_manifest.json").read_text(encoding="utf-8"))
    for name, digest in expected["package_signature"]["files"].items():
        target = (path / name).resolve()
        require(target.is_relative_to(path) and target.is_file(), "Referentiebestand ontbreekt: " + name)
        require(sha256(target) == digest, "Referentiebestand wijkt af van de gecontroleerde v1_1: " + name)
    archive = path / expected["source_archive_relpath"]
    require(archive.is_file() and sha256(archive) == expected["source_archive_sha256"],
            "De oorspronkelijke auteursbron-ZIP ontbreekt of is gewijzigd.")
    return {"path": str(path), "package_id": expected["reference_package"],
            "package_signature": expected["package_signature"],
            "source_archive_sha256": expected["source_archive_sha256"], "modified_files": []}


def load_reference(path):
    checked = verify_reference(path)
    path = Path(checked["path"])
    for name in ("common", "run_train", "engine", "check_watchdog"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(path),
                    "Ander referentiepakket al geïmporteerd; start dit script in een nieuw Python-proces.")
    sys.path.insert(0, str(path))
    reference = importlib.import_module("run_train")
    require(reference.c.ROOT == path, "Onjuiste referentie-import.")
    reference.verify_fixed_contract()
    return reference


def package_signature():
    files = sorted(list(ROOT.glob("*.py")) + [ROOT / "diagnostic_config.json", ROOT / "reference_manifest.json"])
    digests = {p.name: sha256(p) for p in files}
    return {"files": digests,
            "sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}


def copy_context(out):
    for name in ("baseline_results.json", "diagnostic_config.json", "notebook_support.py", NOTEBOOK):
        source = ROOT / name
        require(source.is_file(), "Diagnosepakket onvolledig: " + name)
        shutil.copy2(source, out / name)
    notebook = json.loads((out / NOTEBOOK).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).replace('RUN_DIRECTORY = ""',
                                                   "RUN_DIRECTORY = " + json.dumps(str(out)))
            cell["source"] = source.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
    json_write(out / NOTEBOOK, notebook)


def make_feedback(out, reference_root=None):
    destination = out / "feedback_bundle.zip"
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path == destination or path.suffix in (".zip", ".pt", ".pth", ".pyc", ".tmp"):
                continue
            if path.stat().st_size <= 16 * 1024 * 1024:
                archive.write(path, path.relative_to(out).as_posix())
        for path in sorted(ROOT.glob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".md"):
                archive.write(path, "diagnostic_source/" + path.name)
        if reference_root:
            for path in sorted(Path(reference_root).glob("*.py")):
                archive.write(path, "reference_runner_source/" + path.name)


def execute_gpu(args, out, source, log):
    from check_watchdog import run_bounded_process
    target = out / "backend_test"
    command = [sys.executable, "-u", str(ROOT / "run_check.py"), "_worker",
               "--reference-root", str(args.reference_root.resolve()), "--source", str(source),
               "--worker-out", str(target), "--cpu-threads", str(args.cpu_threads)]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log("GPU-proef: alleen cuDNN benchmark=False; maximaal 300 seconden, inclusief opwarming en metingen.")
    watched = run_bounded_process(command, timeout_seconds=300., heartbeat_seconds=15.,
                                  log=log, label="backend benchmark=False", env=environment)
    result = None
    for name in ("result.json", "progress.json"):
        path = target / name
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            break
    if result is None:
        result = {"status": "FAIL", "error": "GPU-proces eindigde zonder rapport."}
    result.update(watched)
    result["eligible_as_pilot_check"] = False
    if watched["cleanup_failed"]:
        result.update(status="FAIL", error="GPU-proces kon niet worden gestopt; controleer Taakbeheer.")
    elif watched["timed_out"]:
        result.update(status="TIMEOUT", error="GPU-tijdslimiet bereikt; geen conclusie over OOM.")
    else:
        require((result.get("status") == "PASS") == (watched["process_returncode"] == 0),
                "Exitcode en GPU-rapport zijn niet consistent.")
    target.mkdir(exist_ok=True)
    json_write(target / "watchdog_result.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.cpu_threads == 4, "Deze vergelijking behoudt de vier CPU-threads van de eerdere controles.")
    fixed = {"cudnn_benchmark": False, "cudnn_deterministic": False,
             "global_batch": 256, "bn_shards": [64] * 4, "precision_bits": [1, 2, 4, 8, 32],
             "execution": "serial_dp4", "amp": False, "fp32_precision": "ieee", "seed": 20261009,
             "measured_updates": 2, "warmup_updates": 1, "gpu_timeout_seconds": 300,
             "training_authorized": False, "dataset_required": False}
    require(all(CONFIG.get(name) == value for name, value in fixed.items()),
            "De vaste diagnoseconfiguratie is gewijzigd.")
    if args.command == "_worker":
        from backend_worker import run_worker
        return run_worker(args, load_reference(args.reference_root))

    base = args.output_root or args.project_root / "mx_runs" / "anyprecision_resnet50_backend_check"
    base.mkdir(parents=True, exist_ok=True)
    out = (base / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.command)).resolve()
    out.mkdir(exist_ok=False)
    log = Log(out / "console.log")
    report = {"schema_version": 1, "package_id": CONFIG["package_id"],
              "command": "backend_diagnostic", "requested_command": args.command,
              "status": "RUNNING", "eligible_as_pilot_check": False,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "output_directory": str(out), "config": CONFIG, "package_signature": package_signature(),
              "scope": "Synthetic backend measurement only; no ImageNet data, epochs or accuracy.",
              "baseline_comparison": "Earlier user measurements; software/GPU/free memory must be considered."}
    started = time.monotonic()
    reference_root = None
    log("RESULTATENMAP: " + str(out))
    log("RESNET-50 BACKENDDIAGNOSE: batch256; vier groepen64; vijf precisies; FP32; cuDNN benchmark=False.")
    try:
        copy_context(out)
        report["reference"] = verify_reference(args.reference_root)
        reference = load_reference(args.reference_root)
        reference_root = args.reference_root.resolve()
        report["reference_config"] = reference.c.CONFIG
        report["environment"] = reference.c.check_environment(log)
        source, report["source"] = reference.c.prepare_source(out)
        report["selftests"] = reference.cpu_selftests(source, log)
        import test_backend_worker
        import test_delivery
        report["diagnostic_selftests"] = {"worker": test_backend_worker.run_selftests(),
                                          "delivery": test_delivery.run_selftests(args.reference_root)}
        json_write(out / "run_report.json", report)
        if args.command == "selftest":
            report["status"] = "PASS"
            log("CPU-zelftests geslaagd; geen GPU-meting en geen vrijgave voor een pilot.")
        else:
            report["backend_test"] = execute_gpu(args, out, source, log)
            report["status"] = report["backend_test"]["status"]
            baseline = json.loads((ROOT / "baseline_results.json").read_text(encoding="utf-8"))["records"][-1]
            packages = ("torch", "torchvision", "numpy", "Pillow")
            report["comparison_context"] = {
                "baseline_run": baseline["run_name"],
                "same_package_versions": all(report["environment"]["packages"][name] ==
                                             baseline["environment"]["packages"][name] for name in packages),
                "same_gpu": report["backend_test"].get("device") == baseline["runtime"]["device"],
                "note": "Earlier observations, not randomized concurrent trials; current free memory is recorded separately."}
            log("Diagnose afgerond. Beoordeel eerst snelheid/geheugen; dit rapport geeft de pilot niet vrij.")
    except KeyboardInterrupt:
        report.update(status="INTERRUPTED", error="Diagnose onderbroken; er loopt geen ImageNet-training.")
        log(report["error"])
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log("FAIL: " + str(exc))
        log(report["traceback"])
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.monotonic() - started
        json_write(out / "run_report.json", report)
        log("EINDSTATUS: " + report["status"])
        log("Deel feedback_bundle.zip uit deze map. Nog geen ImageNet-pilot starten.")
        log.close()
        make_feedback(out, reference_root)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
