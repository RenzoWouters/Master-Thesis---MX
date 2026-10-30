"""I/O, provenance and runtime helpers; never edits the user's existing sources."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
BITS = tuple(CONFIG["bits"])
NOTEBOOK = "AnyPrecision_ResNet50_ImageNet_pilot2_resultaten.ipynb"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    """Atomic update of a file in the new run's own directory."""
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(temp, path)


def package_signature():
    names = sorted([p.name for p in ROOT.glob("*.py")] +
                   ["experiment_config.json", "SOURCE_MANIFEST.json"])
    files = {name: sha256(ROOT / name) for name in names}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def default_project_root():
    if ROOT.parent.name.lower() == "software_checks":
        return ROOT.parent.parent
    return Path.cwd()


def new_output(project_root, command, output_root=None):
    base = Path(output_root) if output_root else (
        Path(project_root) / "mx_runs" / "anyprecision_imagenet_resnet50_pilot2")
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = base / (stamp + "_" + command)
    out.mkdir(exist_ok=False)
    return out.resolve()


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


def prepare_source(out):
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    archive = ROOT / manifest["archive_relpath"]
    require(sha256(archive) == manifest["archive_sha256"], "De bron-ZIP is gewijzigd.")
    destination = Path(out) / "source_workcopy"
    destination.mkdir(exist_ok=False)
    files = {}
    with ZipFile(archive) as z:
        for item in z.infolist():
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and ".." not in p.parts and
                    "\\" not in item.filename and ":" not in item.filename,
                    "Ongeldig pad in het bronarchief.")
            require(p.parts and p.parts[0] == "Any-Precision-DNNs-master",
                    "Onverwachte structuur van het bronarchief.")
            if item.is_dir():
                continue
            require(len(p.parts) > 1, "Ongeldig bronbestand.")
            relative = Path(*p.parts[1:])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(z.read(item))
            files[relative.as_posix()] = sha256(target)
    require(files == manifest["files_sha256"], "Bronbestanden wijken af van het manifest.")
    result = {"archive_sha256": manifest["archive_sha256"],
              "files_sha256": files, "changed_upstream_files": [],
              "used_modules": ["models", "models.losses", "optimizer"],
              "note": "Auteursmodel, quantizers en soft loss ongewijzigd. De nieuwe runner voert de gedocumenteerde trainingslus uit."}
    json_write(Path(out) / "source_manifest.json", result)
    return destination, result


def import_source(source):
    """Use a private module namespace; never import a different checkout's models."""
    source = Path(source).resolve()
    name = "ap_resnet50_upstream_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        name, source / "models" / "__init__.py",
        submodule_search_locations=[str(source / "models")])
    require(spec is not None and spec.loader is not None, "Kan auteursmodel niet importeren.")
    models = importlib.util.module_from_spec(spec)
    sys.modules[name] = models
    spec.loader.exec_module(models)
    losses = importlib.import_module(name + ".losses")
    opt_name = name + "_optimizer"
    opt_spec = importlib.util.spec_from_file_location(opt_name, source / "optimizer.py")
    optimizer = importlib.util.module_from_spec(opt_spec)
    sys.modules[opt_name] = optimizer
    opt_spec.loader.exec_module(optimizer)
    require(Path(models.__file__).resolve().is_relative_to(source), "Verkeerde modelimport.")
    return models.resnet50q, losses.CrossEntropyLossSoft, optimizer.get_optimizer_config


def environment_metadata():
    """Does not initialize CUDA in the parent of a memory-test subprocess."""
    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "pandas", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def check_environment(log, pip_check=True):
    data = environment_metadata()
    require(data["packages"]["torch"] is not None,
            "PyTorch ontbreekt. Gebruik de bestaande .venv-train\\Scripts\\python.exe.")
    require(data["packages"]["torch"].split("+")[0].startswith("2.10."),
            "Dit pakket is gecontroleerd voor PyTorch 2.10.x; geen automatische versie-upgrade.")
    require(data["packages"]["torchvision"] is not None and
            data["packages"]["torchvision"].split("+")[0].startswith("0.25."),
            "Dit pakket verwacht torchvision 0.25.x bij PyTorch 2.10.x.")
    if pip_check:
        result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=90)
        data["pip_check"] = {"returncode": result.returncode,
                             "output": (result.stdout + result.stderr).strip()}
        log("pip check: " + data["pip_check"]["output"])
        require(result.returncode == 0, "pip check meldt een afhankelijkheidsconflict.")
    import torch
    import torchvision
    data["torch_import"] = str(torch.__version__)
    data["torchvision_import"] = str(torchvision.__version__)
    data["torch_cuda_build"] = torch.version.cuda
    return data


def reseed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure(seed, *, cuda=True, threads=4):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Torch 2.10 API; do not mix with the legacy allow_tf32 flags.
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    reseed(seed)
    data = {"seed": seed, "threads": threads, "AMP": False,
            "torch_compile": False, "fp32_precision": "ieee",
            "cudnn_benchmark": True, "cudnn_deterministic": False,
            "bitexact_resume_guarantee": False}
    if cuda:
        require(torch.cuda.is_available(), "CUDA is niet beschikbaar; geen automatische CPU-training.")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        data["device"] = {"name": props.name, "index": 0,
                          "total_memory_bytes": props.total_memory,
                          "compute_capability": [props.major, props.minor],
                          "visible_device_count": torch.cuda.device_count()}
    return data


def rng_state(generator=None):
    import numpy as np
    import torch
    n = np.random.get_state()
    result = {"python": random.getstate(),
              "numpy": [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
              "torch_cpu": torch.get_rng_state(),
              "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    if generator is not None:
        result["loader_generator"] = generator.get_state()
    return result


def restore_rng(state, generator=None):
    import numpy as np
    import torch
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        require(torch.cuda.is_available(), "Een CUDA-checkpoint kan hier niet op CPU worden hervat.")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])
    if generator is not None:
        generator.set_state(state["loader_generator"].cpu())


def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    require(not path.exists(), "Een bestaand checkpoint wordt niet overschreven: " + str(path))
    temp = path.with_name(path.name + ".tmp")
    require(not temp.exists(), "Tijdelijk checkpoint bestaat al: " + str(temp))
    torch.save(payload, temp)
    os.replace(temp, path)


def copy_notebook(out):
    for name in (NOTEBOOK, "notebook_support.py", "experiment_config.json"):
        path = ROOT / name
        if path.is_file():
            shutil.copy2(path, Path(out) / name)
    notebook_path = Path(out) / NOTEBOOK
    if notebook_path.is_file():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                text = "".join(cell["source"])
                text = text.replace('RUN_DIRECTORY = ""',
                                    "RUN_DIRECTORY = " + json.dumps(str(Path(out).resolve())))
                cell["source"] = text.splitlines(keepends=True)
                cell["outputs"] = []
                cell["execution_count"] = None
        json_write(notebook_path, notebook)


def feedback_bundle(out):
    """Include reports, code and small logs; no data or model/optimizer tensors."""
    out = Path(out)
    target = out / "feedback_bundle.zip"
    require(not target.exists(), "Een feedbackbundel bestaat al in deze nieuwe resultatenmap.")
    with ZipFile(target, "x", compression=ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path == target:
                continue
            if "checkpoints" in path.relative_to(out).parts or path.suffix.lower() in (".pt", ".pth", ".tar", ".tmp", ".zip", ".pyc"):
                continue
            if path.stat().st_size > 16 * 1024 * 1024:
                continue
            z.write(path, path.relative_to(out).as_posix())
        for path in sorted(ROOT.glob("*.py")):
            z.write(path, "runner_source/" + path.name)
        for name in ("README.md", "SOURCE_MANIFEST.json", "experiment_config.json", "PACKAGE_MANIFEST.json", "PROTOCOL.md"):
            if (ROOT / name).is_file():
                z.write(ROOT / name, "runner_source/" + name)
    return target
