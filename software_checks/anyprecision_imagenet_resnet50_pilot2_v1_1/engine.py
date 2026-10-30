"""FP32 Any-Precision steps, including serial emulation of four-device DP.

The serial route preserves the original global minibatch and its four local
BatchNorm batches. Only replica zero's buffer updates survive, as in upstream
nn.DataParallel. It does not claim bit-exact agreement between CUDA devices.
The caller supplies the unchanged upstream model, soft loss and optimizer.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TRAIN_BITS = (32, 8, 4, 2, 1)
EVAL_BITS = (1, 2, 4, 8, 32)
SUPPORTED_MODES = ("native256", "serial_dp4")
SERIAL_BN_POLICY = "torch.chunk(4); local BN; retain only replica-zero buffers per precision"


def _select_precision(model: nn.Module, bit: int) -> None:
    # The upstream trainer sets these attributes throughout the module tree.
    model.apply(lambda module: setattr(module, "wbit", bit))
    model.apply(lambda module: setattr(module, "abit", bit))


def _check_batch(inputs: Tensor, targets: Tensor) -> int:
    if inputs.ndim != 4 or not 1 <= inputs.shape[0] <= 256:
        raise ValueError("Expected a nonempty NCHW minibatch with N <= 256.")
    if inputs.dtype != torch.float32:
        raise TypeError("This reference requires FP32 inputs; mixed precision is disabled.")
    if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
        raise ValueError("Targets must have one class index per input image.")
    if targets.dtype != torch.long or targets.device != inputs.device:
        raise TypeError("Targets must be int64 on the same device as the inputs.")
    if torch.is_autocast_enabled(inputs.device.type):
        raise RuntimeError("Autocast is not part of this FP32 reference protocol.")
    return int(inputs.shape[0])


def _validate_finite_model(model: nn.Module, inputs: Tensor) -> None:
    if not bool(torch.isfinite(inputs).all()):
        raise FloatingPointError("Nonfinite input values.")
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if not value.is_floating_point():
            continue
        if value.dtype != torch.float32:
            raise TypeError(f"Model tensor {name} is not FP32.")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite model tensor: {name}.")


def _validate_gradients(model: nn.Module) -> None:
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if parameter.grad.dtype != torch.float32:
            raise TypeError(f"Gradient for {name} is not FP32.")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Nonfinite gradient: {name}.")
    if missing:
        raise RuntimeError("Missing required gradients: " + ", ".join(missing))


def _validate_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, Tensor) and value.is_floating_point():
                if value.dtype != torch.float32:
                    raise TypeError(f"Optimizer state {name} is not FP32.")
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Nonfinite optimizer state: {name}.")


def _buffer_snapshot(buffers: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(buffer.detach().clone() for buffer in buffers)


def _restore_buffers(buffers: tuple[Tensor, ...], values: tuple[Tensor, ...]) -> None:
    # Called only after the preceding shard's backward has completed. BN
    # backward can reference these tensors, even when they do not need grads.
    with torch.no_grad():
        for destination, source in zip(buffers, values):
            destination.copy_(source)


def _correct_counts(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    # Use a common top-k result, like upstream utils.accuracy. argmax can use
    # different tie-breaking from topk(5), which matters for low-bit logits.
    predictions = logits.detach().topk(min(5, logits.shape[1]), dim=1, sorted=True).indices
    correct = predictions.eq(targets[:, None])
    return correct[:, 0].sum(), correct.any(dim=1).sum()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    soft_loss: nn.Module,
    *,
    mode: str = "serial_dp4",
    replicas: int = 4,
    validate: bool = False,
    progress: Callable[[dict[str, str | int | None]], None] | None = None,
) -> dict[str, Any]:
    """Run one global minibatch and exactly one optimizer update.

    In serial_dp4, losses are weighted by each shard's actual n/N. This also
    covers uneven/final minibatches, including fewer than four images. Losses
    are summed across precisions, without dividing by the number of modes.
    Only detached probabilities from the immediately higher precision teach
    each lower precision. No graph is retained across shards or precisions.

    ``validate`` enables expensive complete FP32/finiteness/gradient checks
    for preflight; normal training does not scan or clone parameter gradients.
    ``progress`` is optional warmup instrumentation. Events contain only scalar
    metadata, with one-based shard numbers. A caller may synchronize CUDA in
    its callback; the normal None path adds no logging or synchronization.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported execution mode: {mode!r}.")
    if mode == "serial_dp4" and replicas != 4:
        raise ValueError("serial_dp4 requires exactly four requested replicas.")
    samples = _check_batch(inputs, targets)
    if validate:
        if progress is not None:
            progress({"stage": "validation_before_start", "bit": None, "shard": None, "shards": None})
        _validate_finite_model(model, inputs)
        _validate_optimizer_state(optimizer)
        if progress is not None:
            progress({"stage": "validation_before_done", "bit": None, "shard": None, "shards": None})

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if mode == "serial_dp4":
        input_chunks = tuple(inputs.chunk(replicas, dim=0))
        target_chunks = tuple(targets.chunk(replicas, dim=0))
    else:
        input_chunks, target_chunks = (inputs,), (targets,)
    chunk_sizes = [int(chunk.shape[0]) for chunk in input_chunks]
    buffers = tuple(model.buffers())
    teachers: list[Tensor] | None = None
    by_bit: dict[str, dict[str, int | float]] = {}

    for bit in TRAIN_BITS:
        if progress is not None:
            progress({"stage": "precision_start", "bit": bit, "shard": None, "shards": len(input_chunks)})
        _select_precision(model, bit)
        initial_buffers = _buffer_snapshot(buffers) if mode == "serial_dp4" else ()
        replica_zero_buffers: tuple[Tensor, ...] | None = None
        next_teachers: list[Tensor] = []
        loss_sum = inputs.new_zeros(())
        top1_sum = torch.zeros((), device=inputs.device, dtype=torch.long)
        top5_sum = torch.zeros((), device=inputs.device, dtype=torch.long)

        for index, (chunk, labels) in enumerate(zip(input_chunks, target_chunks)):
            if index:
                _restore_buffers(buffers, initial_buffers)
            if progress is not None:
                progress({"stage": "forward_start", "bit": bit, "shard": index + 1, "shards": len(input_chunks)})
            logits = model(chunk)
            if progress is not None:
                progress({"stage": "forward_done", "bit": bit, "shard": index + 1, "shards": len(input_chunks)})
            if bit == TRAIN_BITS[0]:
                loss = F.cross_entropy(logits, labels)
            else:
                if teachers is None:
                    raise RuntimeError("Recursive teacher probabilities are missing.")
                loss = soft_loss(logits, teachers[index])
            if loss.ndim != 0:
                raise ValueError("The supplied soft loss must return one mean scalar.")
            if validate and not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss in W{bit}/A{bit}.")
            weight = chunk.shape[0] / samples
            if progress is not None:
                progress({"stage": "backward_start", "bit": bit, "shard": index + 1, "shards": len(input_chunks)})
            (loss * weight).backward()
            if progress is not None:
                progress({"stage": "backward_done", "bit": bit, "shard": index + 1, "shards": len(input_chunks)})

            # Snapshot AFTER backward; capture any in-place buffer updates
            # from the first replica and never retain another replica's state.
            if mode == "serial_dp4" and index == 0:
                replica_zero_buffers = _buffer_snapshot(buffers)
            if bit != TRAIN_BITS[-1]:
                next_teachers.append(F.softmax(logits.detach(), dim=1))
            loss_sum.add_(loss.detach() * weight)
            top1, top5 = _correct_counts(logits, labels)
            top1_sum.add_(top1)
            top5_sum.add_(top5)

        if mode == "serial_dp4":
            if replica_zero_buffers is None:
                raise RuntimeError("Replica-zero state was not recorded.")
            _restore_buffers(buffers, replica_zero_buffers)
        teachers = next_teachers
        by_bit[str(bit)] = {
            "loss": float(loss_sum.item()),
            "top1_correct": int(top1_sum.item()),
            "top5_correct": int(top5_sum.item()),
            "samples": samples,
        }
        if progress is not None:
            progress({"stage": "precision_done", "bit": bit, "shard": None, "shards": len(input_chunks)})

    if validate:
        if progress is not None:
            progress({"stage": "gradient_validation_start", "bit": None, "shard": None, "shards": None})
        _validate_gradients(model)
        if progress is not None:
            progress({"stage": "gradient_validation_done", "bit": None, "shard": None, "shards": None})
    if progress is not None:
        progress({"stage": "optimizer_start", "bit": None, "shard": None, "shards": None})
    optimizer.step()
    if progress is not None:
        progress({"stage": "optimizer_done", "bit": None, "shard": None, "shards": None})
    if validate:
        if progress is not None:
            progress({"stage": "validation_after_start", "bit": None, "shard": None, "shards": None})
        _validate_finite_model(model, inputs)
        _validate_optimizer_state(optimizer)
        if progress is not None:
            progress({"stage": "validation_after_done", "bit": None, "shard": None, "shards": None})
    return {
        "by_bit": by_bit,
        "samples": samples,
        "optimizer_updates": 1,
        "forwards": len(TRAIN_BITS) * len(input_chunks),
        "mode": mode,
        "chunk_sizes": chunk_sizes,
        "replicas_used": len(input_chunks),
        "bn_policy": SERIAL_BN_POLICY if mode == "serial_dp4" else "native minibatch BatchNorm",
        "validated": bool(validate),
    }


def eval_batch(model: nn.Module, inputs: Tensor, targets: Tensor) -> dict[str, Any]:
    """Evaluate all five modes with frozen BN and hard CE, without updates."""
    samples = _check_batch(inputs, targets)
    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    by_bit: dict[str, dict[str, int | float]] = {}
    try:
        with torch.no_grad():
            for bit in EVAL_BITS:
                _select_precision(model, bit)
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                top1, top5 = _correct_counts(logits, targets)
                by_bit[str(bit)] = {
                    "loss": float(loss.item()),
                    "top1_correct": int(top1.item()),
                    "top5_correct": int(top5.item()),
                    "samples": samples,
                }
    finally:
        for module, was_training in training_flags:
            module.training = was_training
    return {
        "by_bit": by_bit,
        "samples": samples,
        "optimizer_updates": 0,
        "forwards": len(EVAL_BITS),
    }
