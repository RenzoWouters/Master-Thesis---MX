"""A complete five-precision FP32 update, matching upstream recursive KD.

One minibatch is used for all five passes.  Gradients are summed, with immediate
backward after each pass and exactly one Adam update.  This is neither gradient
accumulation over microbatches nor separate optimizers per precision.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mobilenet_model import BITS, TRAIN_BITS, select_precision
from upstream.losses import CrossEntropyLossSoft


def create_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)


def _check_batch(inputs: Tensor, targets: Tensor) -> None:
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32) or inputs.shape[0] < 1:
        raise ValueError("Expected a nonempty NCHW CIFAR batch with shape [N,3,32,32].")
    if inputs.dtype != torch.float32:
        raise TypeError("Inputs must be FP32; this probe does not use AMP.")
    if targets.dtype != torch.long or targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
        raise TypeError("Targets must be an int64 vector, with one label per image.")
    if targets.device != inputs.device:
        raise TypeError("Inputs and targets must be on the same device.")
    if torch.is_autocast_enabled(inputs.device.type):
        raise RuntimeError("Autocast is outside the FP32 timing protocol.")


def _finite_tensors(tensors: list[Tensor], label: str) -> None:
    if tensors and not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
        raise FloatingPointError("Nonfinite " + label + ".")


def validate_state(model: nn.Module, optimizer=None, *, require_gradients: bool = False) -> None:
    params = list(model.parameters())
    if any(p.dtype != torch.float32 for p in params):
        raise TypeError("All master parameters must remain FP32.")
    buffers = [b for b in model.buffers() if b.is_floating_point()]
    if any(b.dtype != torch.float32 for b in buffers):
        raise TypeError("Floating model buffers must remain FP32.")
    _finite_tensors(params + buffers, "model state")
    if require_gradients:
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError("Missing accumulated gradients: " + ", ".join(missing))
        gradients = [p.grad for p in params if p.grad is not None]
        if any(g.dtype != torch.float32 for g in gradients):
            raise TypeError("Accumulated gradients must remain FP32.")
        _finite_tensors(gradients, "accumulated gradients")
    if optimizer is not None:
        # Adam's step scalars may be on the CPU while moments live on the GPU.
        states = [value for state in optimizer.state.values() for value in state.values()
                  if isinstance(value, Tensor) and value.is_floating_point()]
        if any(s.dtype != torch.float32 for s in states):
            raise TypeError("Adam's floating state must remain FP32.")
        for device in {s.device for s in states}:
            _finite_tensors([s for s in states if s.device == device], "Adam state")


def _metrics(output: Tensor, target: Tensor, loss_value: float) -> dict:
    with torch.no_grad():
        ranked = output.topk(min(5, output.shape[1]), dim=1).indices
        hits = ranked.eq(target[:, None])
        values = torch.stack((hits[:, 0].float().mean(), hits.any(dim=1).float().mean()))
        top1, top5 = (100.0 * values).cpu().tolist()
    return {"loss": float(loss_value), "top1": float(top1), "top5": float(top5)}


def train_step(model: nn.Module, optimizer, inputs: Tensor, targets: Tensor, *,
               validate: bool = False, progress: Callable[[str], None] | None = None) -> dict:
    _check_batch(inputs, targets)
    if validate:
        _finite_tensors([inputs], "input")
        validate_state(model, optimizer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    teacher = None
    soft_loss = CrossEntropyLossSoft()
    for index, bit in enumerate(TRAIN_BITS):
        select_precision(model, bit)
        if progress:
            progress(f"pass {index + 1}/5: W{bit}/A{bit} forward")
        output = model(inputs)
        loss = F.cross_entropy(output, targets) if teacher is None else soft_loss(output, teacher)
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite W{bit}/A{bit} loss before backward.")
        if progress:
            progress(f"pass {index + 1}/5: W{bit}/A{bit} backward")
        loss.backward()
        metrics[str(bit)] = _metrics(output.detach(), targets, value)
        # Every lower precision follows the immediately preceding precision.
        teacher = F.softmax(output.detach(), dim=1)
        del output, loss
    del teacher
    if validate:
        validate_state(model, require_gradients=True)
    if progress:
        progress("Adam: one update from the sum of all five precision gradients")
    optimizer.step()
    if validate:
        validate_state(model, optimizer, require_gradients=True)
    return {"batch_size": int(inputs.shape[0]), "precision_order": list(TRAIN_BITS),
            "optimizer_steps": 1, "metrics": metrics}


@torch.no_grad()
def validation_step(model: nn.Module, inputs: Tensor, targets: Tensor, *,
                    validate: bool = False, progress: Callable[[str], None] | None = None) -> dict:
    _check_batch(inputs, targets)
    if validate:
        _finite_tensors([inputs], "input")
        validate_state(model)
    model.eval()
    metrics = {}
    for bit in BITS:
        select_precision(model, bit)
        if progress:
            progress(f"validation W{bit}/A{bit}")
        output = model(inputs)
        value = float(F.cross_entropy(output, targets).item())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite W{bit}/A{bit} validation loss.")
        metrics[str(bit)] = _metrics(output, targets, value)
    return {"batch_size": int(inputs.shape[0]), "precision_order": list(BITS),
            "optimizer_steps": 0, "metrics": metrics}
