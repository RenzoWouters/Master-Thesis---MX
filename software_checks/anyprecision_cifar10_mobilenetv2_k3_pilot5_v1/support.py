"""Small dependency-free parent helpers. No CUDA work occurs on import."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
BITS = tuple(CONFIG["bits"])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_fingerprint():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py" or relative in ("experiment_config.json", "SOURCE_MANIFEST.json"):
            files[relative] = sha256(path)
    encoded = json.dumps(files, sort_keys=True).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def verify_upstream():
    manifest = json_read(ROOT / "SOURCE_MANIFEST.json")
    for entry in manifest["files"]:
        require(sha256(ROOT / entry["packaged_path"]) == entry["sha256"],
                "De meegeleverde upstreambron is gewijzigd: " + entry["packaged_path"])
    return {"status": "PASS", "source_archive_sha256": manifest["archive_sha256"],
            "unchanged_files": [entry["packaged_path"] for entry in manifest["files"]]}


def project_root(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
    elif ROOT.parent.name.lower() == "software_checks":
        path = ROOT.parent.parent
    else:
        path = Path.cwd().resolve()
    require(path.is_dir(), "Projectmap bestaat niet: " + str(path))
    return path


def new_run_directory(project, command):
    parent = Path(project) / "mx_runs" / CONFIG["experiment"]
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = parent / (stamp + "_" + command)
    out.mkdir(exist_ok=False)
    return out


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message):
        message = str(message)
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def csv_write(path, rows):
    if not rows:
        return
    fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_feedback_bundle(out):
    """Explicit allowlists exclude weights, datasets and previous experiments."""
    out = Path(out)
    destination = out / "feedback_bundle.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in (".json", ".csv", ".log", ".txt"):
                continue
            if path.stat().st_size <= 20 * 1024 * 1024:
                archive.write(path, path.relative_to(out).as_posix())
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in (".py", ".json", ".md", ".ipynb"):
                archive.write(path, "runner_source/" + path.relative_to(ROOT).as_posix())
    return destination


def basic_environment():
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine()}


def configure_cuda():
    import numpy as np
    import torch
    import torchvision
    require(torch.cuda.is_available(), "CUDA is niet beschikbaar. Gebruik .venv-train met de CUDA-versie van PyTorch.")
    torch.set_num_threads(CONFIG["runtime"]["cpu_threads"])
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = CONFIG["runtime"]["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = CONFIG["runtime"]["cudnn_deterministic"]
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        precision_api = "fp32_precision=ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        precision_api = "allow_tf32=False"
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    # A bounded caching allocator helps surface pressure as an OOM rather than
    # willingly reserving almost the entire display GPU. External allocations
    # and WDDM residency are still outside this counter's scope.
    fraction = min(CONFIG["runtime"]["max_allocator_fraction"],
                   CONFIG["runtime"]["initial_free_memory_fraction"] * free / props.total_memory)
    require(fraction * props.total_memory >= 1024 ** 3,
            "Minder dan 1 GiB beschikbaar binnen het GPU-budget. Sluit andere GPU-taken en herhaal de check.")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {**basic_environment(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "precision_api": precision_api,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "amp": False,
            "cpu_threads": torch.get_num_threads(),
            "device": {"name": props.name, "total_memory_bytes": int(props.total_memory),
                       "initial_free_bytes": int(free), "cuda_reported_total_bytes": int(total),
                       "capability": list(torch.cuda.get_device_capability(0)),
                       "allocator_fraction": fraction,
                       "allocator_budget_bytes": int(fraction * props.total_memory)}}


def memory_snapshot(phase, step):
    import torch
    free, total = torch.cuda.mem_get_info()
    return {"phase": phase, "step": step,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_bytes": int(free), "total_bytes": int(total)}


def finite_state(model, optimizer):
    import torch
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, value in tensors:
        if value.is_floating_point():
            require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                    "Niet-eindige of niet-FP32 modelwaarde: " + name)
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                        "Niet-eindige of niet-FP32 Adam-state: " + name)
    return {"status": "PASS", "model_and_optimizer_finite": True, "master_dtype": "float32"}


def bn_snapshot(model):
    import torch
    return {name: {field: getattr(module, field).detach().cpu().clone()
                   for field in ("running_mean", "running_var", "num_batches_tracked")}
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}


def check_bn_updates(before, after, count):
    require(before.keys() == after.keys() and bool(before), "BatchNorm-structuur ontbreekt of is gewijzigd.")
    for name, values in before.items():
        difference = int(after[name]["num_batches_tracked"]) - int(values["num_batches_tracked"])
        require(difference == count, f"Verkeerd aantal BN-updates voor {name}: {difference}, verwacht {count}.")
    return {"status": "PASS", "banks": len(before), "updates_per_bank": count}


def check_bn_frozen(before, after):
    import torch
    require(before.keys() == after.keys() and bool(before), "BatchNorm-structuur is gewijzigd.")
    for name, values in before.items():
        for field, value in values.items():
            require(torch.equal(value, after[name][field]), "Evaluatie wijzigde BN: " + name + "/" + field)
    return {"status": "PASS", "all_buffers_unchanged": True, "banks": len(before)}
