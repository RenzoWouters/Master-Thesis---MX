"""Complete epoch checkpoints for this five-epoch pilot.

The diagnostic timing checkpoint is deliberately incompatible.  Restoration
validates the protocol, model, Adam, RNG and data identity before modifying the
newly constructed model.  CUDA reproducibility is numerical, not bit-exact.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch


SCHEMA = "mobilenetv2_cifar10_complete_epoch_v1"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _plain(value):
    # Strips subclasses such as torch.torch_version.TorchVersion so that
    # weights_only=True never needs an allowlist of Python classes.
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprint(value):
    value = value.get("sha256") if isinstance(value, dict) else value
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "Ongeldige pakketfingerprint.")
    return value


def learning_rate_for_completed_epochs(completed, base_lr=0.001,
                                       milestones=(150, 250, 350), gamma=0.1):
    _require(isinstance(completed, int) and not isinstance(completed, bool)
             and completed >= 0, "Voltooide epochs moeten een niet-negatief geheel getal zijn.")
    _require(math.isfinite(base_lr) and base_lr > 0 and 0 < gamma < 1,
             "Ongeldig learning-rateschema.")
    _require(tuple(sorted(set(milestones))) == tuple(milestones)
             and all(isinstance(e, int) and e > 0 for e in milestones), "Ongeldige LR-mijlpalen.")
    return float(base_lr * gamma ** sum(completed >= e for e in milestones))


def apply_learning_rate(optimizer, completed, base_lr=0.001,
                        milestones=(150, 250, 350), gamma=0.1):
    lr = learning_rate_for_completed_epochs(completed, base_lr, milestones, gamma)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _schedule(protocol, completed):
    recipe = protocol["full_recipe_reference"]
    base = float(protocol["optimizer"]["lr"])
    milestones = tuple(recipe["lr_decay_epochs"])
    gamma = float(recipe["lr_decay_factor"])
    return {"completed_epochs": completed, "base_lr": base,
            "milestones": list(milestones), "gamma": gamma,
            "lr_for_next_epoch": learning_rate_for_completed_epochs(completed, base, milestones, gamma)}


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def capture_rng(train_generator, val_generator, *, device):
    np_state = np.random.get_state()
    device_type = torch.device(device).type
    return {"device_type": device_type, "python": random.getstate(),
            "numpy": [str(np_state[0]), np_state[1].tolist(), int(np_state[2]),
                      int(np_state[3]), float(np_state[4])],
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]
                if device_type == "cuda" else [],
            "train_generator": train_generator.get_state().clone(),
            "val_generator": val_generator.get_state().clone()}


def _numpy_state(value):
    return (value[0], np.asarray(value[1], dtype=np.uint32),
            int(value[2]), int(value[3]), float(value[4]))


def _validate_rng(state, device):
    kind = torch.device(device).type
    _require(state["device_type"] == kind, "Checkpoint en hervatting gebruiken een ander apparaattype.")
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(_numpy_state(state["numpy"]))
    for name in ("torch_cpu", "train_generator", "val_generator"):
        value = state[name]
        _require(torch.is_tensor(value) and value.device.type == "cpu"
                 and value.dtype == torch.uint8 and value.ndim == 1,
                 "Ongeldige RNG-status: " + name)
        torch.Generator(device="cpu").set_state(value)
    if kind == "cuda":
        _require(len(state["torch_cuda"]) == torch.cuda.device_count(),
                 "Ander aantal CUDA-apparaten dan in de checkpoint.")
        for index, value in enumerate(state["torch_cuda"]):
            _require(value.dtype == torch.uint8 and value.device.type == "cpu", "Ongeldige CUDA-RNG-status.")
            torch.Generator(device=f"cuda:{index}").set_state(value)
    else:
        _require(state["torch_cuda"] == [], "CPU-checkpoint bevat onverwachte CUDA-RNG-status.")


def restore_rng(state, train_generator, val_generator, *, device):
    _validate_rng(state, device)
    random.setstate(state["python"])
    np.random.set_state(_numpy_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])
    val_generator.set_state(state["val_generator"])


def _validate_payload(payload):
    _require(isinstance(payload, dict) and payload.get("checkpoint_schema") == SCHEMA,
             "Geen ondersteunde volledige-epochcheckpoint; de oude timingcheckpoint kan niet worden hervat.")
    _require(payload.get("partial_epoch") is False and payload.get("resume_supported") is True,
             "Deze checkpoint ondersteunt geen hervatting vanaf een volledige epoch.")
    p = payload["protocol"]
    _require(p["experiment"] == payload["experiment"] and _digest(p) == payload["protocol_sha256"],
             "Checkpointprotocol is inconsistent of gewijzigd.")
    _fingerprint(payload["package_fingerprint"])
    epoch, limit = payload["epoch_completed"], p["pilot"]["epochs"]
    _require(isinstance(epoch, int) and not isinstance(epoch, bool) and 1 <= epoch <= limit == 5,
             "Checkpoint ligt buiten de vijf-epochpilot.")
    batch, examples = int(p["batch_size"]), int(p["pilot"]["train_examples"])
    _require(batch > 0 and examples > 0 and p["pilot"].get("train_drop_last") is False,
             "Onverwacht databatchbeleid in checkpoint.")
    steps = (examples + batch - 1) // batch
    _require(payload["global_updates"] == epoch * steps,
             "Checkpoint heeft geen consistent aantal optimizerupdates per volledige epoch.")
    history = payload["history"]
    _require(isinstance(history, list) and len(history) == epoch
             and [row.get("epoch") for row in history] == list(range(1, epoch + 1)),
             "Checkpointgeschiedenis mist epochs of bevat dubbelen.")
    for row in history:
        if "global_updates" in row:
            _require(row["global_updates"] == row["epoch"] * steps,
                     "Optimizerupdates in geschiedenis komen niet overeen.")
    _require(isinstance(payload["best_epoch"], int) and 1 <= payload["best_epoch"] <= epoch,
             "Ongeldige beste epoch in checkpoint.")
    _require(math.isfinite(payload["best_top1"]) and 0 <= payload["best_top1"] <= 100,
             "Ongeldige beste accuracy in checkpoint.")
    _require(payload["lr_schedule_state"] == _schedule(p, epoch), "Learning-rateschema is inconsistent.")
    _require(payload.get("future_full_reference_must_restart_from_zero") is True,
             "Pilot mag niet als start van de latere referentietraining worden gebruikt.")
    return payload


def _validate_adam(state, model, optimizer, updates, protocol):
    _require(isinstance(optimizer, torch.optim.Adam), "Checkpoint vereist de vastgelegde Adamoptimizer.")
    groups = state["param_groups"]
    _require(len(groups) == len(optimizer.param_groups), "Adamparametergroepen zijn gewijzigd.")
    ids, parameters = [], []
    steps = (int(protocol["pilot"]["train_examples"]) + int(protocol["batch_size"]) - 1) // int(protocol["batch_size"])
    completed = updates // steps
    recipe = protocol["full_recipe_reference"]
    previous_lr = learning_rate_for_completed_epochs(completed - 1,
        float(protocol["optimizer"]["lr"]), tuple(recipe["lr_decay_epochs"]), float(recipe["lr_decay_factor"]))
    for saved, current in zip(groups, optimizer.param_groups):
        _require(len(saved["params"]) == len(current["params"]), "Adamparameteraantal is gewijzigd.")
        ids.extend(saved["params"])
        parameters.extend(current["params"])
        expected = protocol["optimizer"]
        for key in ("eps", "weight_decay"):
            _require(float(saved[key]) == float(expected[key]), "Adaminstelling verschilt: " + key)
        _require(tuple(saved["betas"]) == tuple(expected["betas"]), "Adambetas verschillen.")
        _require(not saved.get("amsgrad", False), "Onverwachte AMSGrad-status.")
        _require(not saved.get("maximize", False) and not saved.get("differentiable", False),
                 "Onverwacht Adamupdatebeleid.")
        _require(float(saved["lr"]) == previous_lr, "Adamlearningrate past niet bij de voltooide epoch.")
    _require(len(ids) == len(set(ids)) and set(ids) == set(state["state"]), "Onvolledige Adamstatus.")
    _require(len(parameters) == len(list(model.parameters())), "Adam omvat niet alle modelparameters.")
    for key, parameter in zip(ids, parameters):
        values = state["state"][key]
        _require(set(values) == {"step", "exp_avg", "exp_avg_sq"}, "Onverwachte Adamstatusvelden.")
        step = values["step"]
        _require(torch.is_tensor(step) and step.numel() == 1 and float(step) == updates,
                 "Adamstapteller komt niet overeen met de checkpointupdates.")
        for field in ("step", "exp_avg", "exp_avg_sq"):
            tensor = values[field]
            _require(torch.is_tensor(tensor) and tensor.dtype == torch.float32
                     and bool(torch.isfinite(tensor).all()), "Ongeldige FP32-Adamstatus: " + field)
        for field in ("exp_avg", "exp_avg_sq"):
            _require(values[field].shape == parameter.shape, "Vorm van Adammomenten klopt niet.")


def _validate_model_state(state, model):
    expected = model.state_dict()
    _require(state.keys() == expected.keys(), "Checkpointmodelstructuur is gewijzigd.")
    for name, value in state.items():
        _require(torch.is_tensor(value) and value.shape == expected[name].shape
                 and value.dtype == expected[name].dtype, "Checkpointtensor wijkt af: " + name)
        if value.is_floating_point():
            _require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                     "Checkpoint bevat niet-eindige of niet-FP32 modelstatus: " + name)


def _validate_runtime(saved, expected):
    keys = ("torch", "torchvision", "numpy", "cuda_build", "cudnn_version", "precision_api",
            "cudnn_benchmark", "cudnn_deterministic", "cpu_threads", "amp")
    for key in keys:
        _require(saved.get(key) == expected.get(key), "Runtime gewijzigd sinds checkpoint: " + key)
    for key in ("name", "total_memory_bytes", "capability"):
        _require(saved.get("device", {}).get(key) == expected.get("device", {}).get(key),
                 "GPU gewijzigd sinds checkpoint: " + key)


def save_checkpoint(path, model, optimizer, *, epoch_completed, global_updates, history,
                    train_generator, val_generator, protocol, package_fingerprint,
                    runtime, dataset_identity, best_epoch, best_top1):
    path = Path(path)
    _require(not path.exists(), "Bestaande checkpoint wordt niet overschreven: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    p = _plain(protocol)
    payload = {"checkpoint_schema": SCHEMA, "schema_version": 1,
        "experiment": p["experiment"], "protocol": p, "protocol_sha256": _digest(p),
        "package_fingerprint": _fingerprint(package_fingerprint),
        "partial_epoch": False, "resume_supported": True,
        "future_full_reference_must_restart_from_zero": True,
        "epoch_completed": epoch_completed, "global_updates": global_updates,
        "history": _plain(history), "best_epoch": best_epoch, "best_top1": float(best_top1),
        "lr_schedule_state": _schedule(p, epoch_completed),
        "runtime": _plain(runtime), "dataset_identity": _plain(dataset_identity),
        "model_state": _cpu_clone(model.state_dict()),
        "optimizer_state": _cpu_clone(optimizer.state_dict()),
        "rng_state": capture_rng(train_generator, val_generator, device=device)}
    _validate_payload(payload)
    _validate_model_state(payload["model_state"], model)
    _validate_adam(payload["optimizer_state"], model, optimizer, global_updates, p)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes a complete file atomically and
        # fails if the destination exists (also on Windows/NTFS).
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    seconds = time.perf_counter() - started
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "seconds": seconds, "sha256": _file_hash(path), "epoch_completed": epoch_completed,
            "partial_epoch": False, "resume_supported": epoch_completed < p["pilot"]["epochs"],
            "in_feedback_bundle": False}


def load_checkpoint(path, *, expected_experiment, expected_fingerprint):
    path = Path(path).expanduser().resolve()
    _require(path.is_file(), "Checkpoint ontbreekt: " + str(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_payload(payload)
    _require(payload["experiment"] == expected_experiment, "Checkpoint hoort bij een ander experiment.")
    _require(payload["package_fingerprint"] == _fingerprint(expected_fingerprint),
             "Checkpoint hoort bij een andere pakketversie.")
    return payload


def restore_checkpoint(payload, model, optimizer, train_generator, val_generator, *,
                       device, expected_runtime=None, expected_dataset_identity=None):
    _validate_payload(payload)
    _require(payload["epoch_completed"] < payload["protocol"]["pilot"]["epochs"],
             "Deze vijf-epochpilot is al voltooid; geen training voorbij epoch 5 toegestaan.")
    if expected_runtime is not None:
        _validate_runtime(payload["runtime"], _plain(expected_runtime))
    if expected_dataset_identity is not None:
        _require(payload["dataset_identity"] == _plain(expected_dataset_identity),
                 "Datasetidentiteit of databeleid wijkt af van de checkpoint.")
    _require(next(model.parameters()).device == torch.device(device)
             or (str(device) == "cuda" and next(model.parameters()).device.type == "cuda"),
             "Model staat niet op het gekozen hervatapparaat.")
    _validate_model_state(payload["model_state"], model)
    _validate_adam(payload["optimizer_state"], model, optimizer, payload["global_updates"], payload["protocol"])
    _validate_rng(payload["rng_state"], device)
    # No global RNG is touched until all structural/data/runtime checks pass.
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng(payload["rng_state"], train_generator, val_generator, device=device)
    return {"status": "PASS", "epoch_completed": payload["epoch_completed"],
            "next_epoch": payload["epoch_completed"] + 1, "global_updates": payload["global_updates"],
            "rng_restored": ["python", "numpy", "torch_cpu", "torch_cuda", "train_generator", "val_generator"],
            "model_and_bn_restored": True, "optimizer_restored": True,
            "bitexact_cuda_guaranteed": False}


def _fixture_protocol():
    # Deliberately separate from the production data protocol: four synthetic
    # examples make a tiny two-update epoch for this resume self-test only.
    import support
    p = _plain(support.CONFIG)
    p["experiment"] += "_resume_fixture"
    p["batch_size"] = 2
    p["pilot"]["train_examples"] = 4
    return p


def _fixture_epoch(model, optimizer, train_generator, val_generator, device):
    from engine import train_step, validation_step
    images = torch.linspace(-1, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
    labels = torch.tensor([0, 3, 6, 9])
    order = torch.randperm(4, generator=train_generator)
    for ids in order.reshape(2, 2):
        # Explicitly exercise every global RNG plus dropout's device RNG.
        scale = 0.95 + random.random() * 0.025 + float(np.random.random()) * 0.025
        inputs = images[ids] * scale + torch.rand(2, 3, 32, 32) * 0.005
        train_step(model, optimizer, inputs.to(device), labels[ids].to(device))
    val_order = torch.randperm(4, generator=val_generator)
    for ids in val_order.reshape(2, 2):
        validation_step(model, images[ids].to(device), labels[ids].to(device))


def _compare_nested(a, b, *, atol, rtol):
    result = {"exact": True, "outside": 0, "max_abs": 0.0}
    def visit(x, y):
        if torch.is_tensor(x):
            x, y = x.detach().cpu(), y.detach().cpu()
            exact = torch.equal(x, y)
            result["exact"] &= exact
            if x.is_floating_point():
                diff = float((x - y).abs().max()) if x.numel() else 0.0
                result["max_abs"] = max(result["max_abs"], diff)
                okay = bool(torch.allclose(x, y, atol=atol, rtol=rtol))
            else:
                okay = exact
            result["outside"] += int(not okay)
        elif isinstance(x, dict):
            _require(x.keys() == y.keys(), "Diagnosestructuur verschilt.")
            for key in x:
                visit(x[key], y[key])
        elif isinstance(x, (tuple, list)):
            _require(type(x) == type(y) and len(x) == len(y), "Diagnoselijst verschilt.")
            for left, right in zip(x, y):
                visit(left, right)
        else:
            result["exact"] &= x == y
            result["outside"] += int(x != y)
    visit(a, b)
    return result


def run_resume_diagnostic(device="cpu"):
    """Real MobileNet continuation; temporarily controlled cuDNN on CUDA.

    This does not download data, train an epoch of CIFAR-10, or authorize a
    bit-exact CUDA claim.  Parent training policy is restored in ``finally``.
    """
    from engine import create_optimizer
    from mobilenet_model import build_model
    device = torch.device(device)
    old_benchmark, old_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    train_generator, val_generator = torch.Generator().manual_seed(101), torch.Generator().manual_seed(102)
    outer_rng = capture_rng(train_generator, val_generator, device=device)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        random.seed(742); np.random.seed(743); torch.manual_seed(744)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(745)
        model = build_model().to(device)
        optimizer = create_optimizer(model)
        _fixture_epoch(model, optimizer, train_generator, val_generator, device)
        with tempfile.TemporaryDirectory(prefix="mobilenet_resume_") as directory:
            path = Path(directory) / "epoch001.pt"
            protocol = _fixture_protocol()
            save_checkpoint(path, model, optimizer, epoch_completed=1, global_updates=2,
                history=[{"epoch": 1, "global_updates": 2}], train_generator=train_generator,
                val_generator=val_generator, protocol=protocol, package_fingerprint="0" * 64,
                runtime={"device": {"name": str(device)}}, dataset_identity={"fixture": True},
                best_epoch=1, best_top1=0.0)
            _fixture_epoch(model, optimizer, train_generator, val_generator, device)
            reference_model, reference_optimizer = _cpu_clone(model.state_dict()), _cpu_clone(optimizer.state_dict())
            reference_rng = capture_rng(train_generator, val_generator, device=device)
            del optimizer, model
            other = build_model().to(device)
            other_optimizer = create_optimizer(other)
            train2, val2 = torch.Generator().manual_seed(11), torch.Generator().manual_seed(12)
            loaded = load_checkpoint(path, expected_experiment=protocol["experiment"], expected_fingerprint="0" * 64)
            restore_checkpoint(loaded, other, other_optimizer, train2, val2, device=device,
                               expected_dataset_identity={"fixture": True})
            _fixture_epoch(other, other_optimizer, train2, val2, device)
            atol, rtol = (0.0, 0.0) if device.type == "cpu" else (1e-6, 1e-4)
            m = _compare_nested(reference_model, other.state_dict(), atol=atol, rtol=rtol)
            o = _compare_nested(reference_optimizer, other_optimizer.state_dict(), atol=atol, rtol=rtol)
            rng = _compare_nested(reference_rng, capture_rng(train2, val2, device=device), atol=0, rtol=0)
            okay = m["outside"] == o["outside"] == rng["outside"] == 0
            return {"status": "PASS" if okay else "FAIL", "device": str(device),
                "model": "actual MobileNetV2 / five-precision engine", "rng_exact": rng["exact"],
                "model_exact": m["exact"], "optimizer_exact": o["exact"],
                "max_abs_difference": max(m["max_abs"], o["max_abs"]),
                "tensors_outside_tolerance": m["outside"] + o["outside"], "atol": atol, "rtol": rtol,
                "backend_control": "cuDNN benchmark=False, deterministic=True temporarily" if device.type == "cuda" else "CPU exact comparison",
                "bitexact_cuda_guaranteed": False, "real_dataset_training": False}
    finally:
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
        restore_rng(outer_rng, train_generator, val_generator, device=device)
        torch.set_num_threads(old_threads)
