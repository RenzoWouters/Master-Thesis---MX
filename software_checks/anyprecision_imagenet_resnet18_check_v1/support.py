"""Read-only provenance and runtime helpers for a bounded R18 feasibility check."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import subprocess
import sys
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
BITS = tuple(CONFIG["bits"])
NOTEBOOK = "ResNet18_geheugen_snelheid_check.ipynb"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


class Log:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8", newline="\n")

    def __call__(self, message):
        message = str(message)
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def verify_fixed_contract():
    require(__debug__, "Voer de controle uit zonder python -O: de onafhankelijke zelftests gebruiken assertions.")
    fixed = {
        "package_id": "anyprecision_imagenet_resnet18_check_v1",
        "command_kind": "resnet18_feasibility_check",
        "bits": [1, 2, 4, 8, 32], "global_batch": 256,
        "reference_replicas": 4, "bn_shards": [64] * 4, "execution": "serial_dp4",
        "optimizer": "SGD", "lr": 0.3, "momentum": 0.9, "weight_decay": 1e-5,
        "seed": 20261009, "amp": False, "fp32_precision": "ieee",
        "cudnn_benchmark": True, "cudnn_deterministic": False,
        "activation_checkpointing": False, "mx": False, "weight_truncation": False,
        "warmup_updates": 1, "measured_updates": 2, "gpu_timeout_seconds": 300,
        "cpu_threads": 4, "training_authorized": False, "dataset_required": False,
        "eligible_as_pilot_check": False, "exact_original_resnet18_reproduction_claim": False,
        "reference_epochs": 80, "lr_milestones": [45, 60, 70], "lr_gamma": 0.1,
        "reference_schedule_used_in_check": False,
    }
    require(all(CONFIG.get(name) == value for name, value in fixed.items()),
            "De vaste ResNet-18-diagnoseconfiguratie is gewijzigd.")


def _safe_relative(value):
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and ".." not in path.parts and
            "\\" not in value and ":" not in value, "Ongeldig pad in bronmanifest.")
    return Path(*path.parts)


def verify_sources(package_root=None):
    """Validate the bundled archives and inherited engine before importing them."""
    root = Path(package_root or ROOT).resolve()
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    require(set(manifest["archives"]) == {"anyprecision", "truncquant"},
            "De twee vereiste bronarchieven ontbreken in het manifest.")
    for name, item in manifest["archives"].items():
        path = root / _safe_relative(item["archive_relpath"])
        require(path.is_file(), "Bronarchief ontbreekt: " + name)
        require(sha256(path) == item["archive_sha256"], "Bronarchief gewijzigd: " + name)
    for name, item in manifest["inherited_files"].items():
        path = root / _safe_relative(name)
        require(path.is_file() and sha256(path) == item["sha256"],
                "De eerder gecontroleerde trainingslus/watchdog is gewijzigd: " + name)
    return manifest


def prepare_sources(out):
    manifest = verify_sources()
    base = Path(out) / "source_workcopy"
    base.mkdir(exist_ok=False)
    sources, reports = {}, {}
    for name, item in manifest["archives"].items():
        destination = base / name
        destination.mkdir()
        archive_path = ROOT / item["archive_relpath"]
        observed = {}
        with ZipFile(archive_path) as archive:
            for relative, digest in item["files_sha256"].items():
                target = destination / _safe_relative(relative)
                member = item["root"] + "/" + relative
                contents = archive.read(member)
                require(hashlib.sha256(contents).hexdigest() == digest,
                        "Broninhoud wijkt af: " + name + "/" + relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(contents)
                observed[relative] = digest
        sources[name] = destination.resolve()
        reports[name] = {"archive_sha256": item["archive_sha256"],
                         "files_sha256": observed, "modified_source_files": []}
    reports["integration"] = {
        "architecture": "TruncQuant/trunc_models/trunc_resnet_quan.py::ResNet(BasicBlock,[2,2,2,2])",
        "quantizers_and_soft_loss": "Original Any-Precision source, loaded under private module names",
        "source_files_modified": False,
        "exact_original_paper_R18_identity_established": False,
        "note": "The separate TruncQuant models/resnet18q route is not used.",
    }
    json_write(Path(out) / "source_manifest.json", reports)
    return sources, reports


def verify_source_copy(source_root):
    manifest = verify_sources()
    base = Path(source_root).resolve()
    sources = {}
    for name, item in manifest["archives"].items():
        destination = base / name
        require(destination.is_dir(), "Gecontroleerde bronwerkkopie ontbreekt: " + name)
        for relative, digest in item["files_sha256"].items():
            path = destination / _safe_relative(relative)
            require(path.is_file() and sha256(path) == digest,
                    "Bronwerkkopie gewijzigd: " + name + "/" + relative)
        sources[name] = destination
    return sources


def package_signature():
    paths = sorted(list(ROOT.glob("*.py")) + [ROOT / "experiment_config.json", ROOT / "SOURCE_MANIFEST.json"])
    files = {path.name: sha256(path) for path in paths}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def default_project_root():
    return ROOT.parent.parent if ROOT.parent.name.lower() == "software_checks" else Path.cwd()


def new_output(project_root, command, output_root=None):
    base = Path(output_root) if output_root else Path(project_root) / "mx_runs" / "anyprecision_imagenet_resnet18_check"
    base.mkdir(parents=True, exist_ok=True)
    out = base / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + command)
    out.mkdir(exist_ok=False)
    return out.resolve()


def check_environment(log):
    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "pandas", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    data = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}
    require(versions["torch"] is not None and versions["torch"].split("+")[0].startswith("2.10."),
            "Gebruik de bestaande .venv-train met PyTorch 2.10.x; er wordt niets geïnstalleerd.")
    require(versions["torchvision"] is not None and versions["torchvision"].split("+")[0].startswith("0.25."),
            "Dit pakket verwacht torchvision 0.25.x naast PyTorch 2.10.x.")
    log("Omgevingscontrole: pip check (maximaal 30 seconden).")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    data["pip_check"] = {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    log("pip check: " + data["pip_check"]["output"])
    require(result.returncode == 0, "pip check meldt een afhankelijkheidsconflict.")
    import torch
    import torchvision
    data.update(torch_import=str(torch.__version__), torchvision_import=str(torchvision.__version__),
                torch_cuda_build=torch.version.cuda)
    # No CUDA availability query or device allocation in this CPU parent.
    return data


def configure(seed, *, cuda=True, threads=4):
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = {"seed": seed, "threads": threads, "AMP": False, "torch_compile": False,
            "fp32_precision": "ieee", "cudnn_benchmark": True, "cudnn_deterministic": False,
            "activation_checkpointing": False, "bitexact_resume_guarantee": False}
    if cuda:
        require(torch.cuda.is_available(), "CUDA is niet beschikbaar; geen automatische CPU-training.")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        data["device"] = {"name": props.name, "index": 0, "total_memory_bytes": props.total_memory,
                          "compute_capability": [props.major, props.minor],
                          "visible_device_count": torch.cuda.device_count()}
    return data


def copy_context(out):
    for name in (NOTEBOOK, "notebook_support.py", "experiment_config.json", "baseline_results.json"):
        source = ROOT / name
        require(source.is_file(), "Het diagnosepakket is onvolledig: " + name)
        shutil.copy2(source, Path(out) / name)
    path = Path(out) / NOTEBOOK
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"]).replace('RUN_DIRECTORY = ""',
                                                  "RUN_DIRECTORY = " + json.dumps(str(Path(out).resolve())))
            cell["source"] = text.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
    json_write(path, notebook)


def feedback_bundle(out):
    out = Path(out)
    destination = out / "feedback_bundle.zip"
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path == destination or path.suffix.lower() in (".zip", ".pt", ".pth", ".tar", ".pyc", ".tmp"):
                continue
            if path.stat().st_size <= 16 * 1024 * 1024:
                archive.write(path, path.relative_to(out).as_posix())
        for path in sorted(ROOT.glob("*")):
            if path.is_file() and path.suffix.lower() in (".py", ".json", ".md"):
                archive.write(path, "runner_source/" + path.name)
    return destination
