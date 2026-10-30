"""Bounded CPU checks of checkpoint continuity, schedule and failure gates.

Only tiny synthetic tensors and scalar scheduler steps are used. No ImageNet
data, CUDA work or training experiment is started by these checks.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

import common
import engine
from test_engine import _RecordingSoftLoss, _TinyQuantNet


def _optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.035, momentum=0.9, weight_decay=1e-5)


def _random_batch(generator):
    python_draw, numpy_draw = random.random(), float(np.random.random())
    images = torch.randn(8, 3, 6, 6) + python_draw + numpy_draw
    labels = torch.randint(0, 7, (8,))
    order = torch.randperm(8, generator=generator)
    return images[order], labels[order], (python_draw, numpy_draw, order)


def _same_rng(left, right):
    assert left["python"] == right["python"] and left["numpy"] == right["numpy"]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert torch.equal(left["loader_generator"], right["loader_generator"])
    assert left["torch_cuda"] == right["torch_cuda"] == []


def _checkpoint_case():
    common.reseed(271828)
    direct = _TinyQuantNet()
    direct_opt = _optimizer(direct)
    direct_generator = torch.Generator().manual_seed(314159)
    x, y, _ = _random_batch(direct_generator)
    engine.train_step(direct, direct_opt, x, y, _RecordingSoftLoss(), validate=True)
    with tempfile.TemporaryDirectory(prefix="ap_resnet50_runner_test_") as directory:
        checkpoint = Path(directory) / "tiny_epoch_boundary.pt"
        common.save_checkpoint(checkpoint, {
            "model": direct.state_dict(), "optimizer": direct_opt.state_dict(),
            "rng": common.rng_state(direct_generator), "completed_steps": 1,
        })
        x2, y2, draws2 = _random_batch(direct_generator)
        engine.train_step(direct, direct_opt, x2, y2, _RecordingSoftLoss(), validate=True)
        direct_rng = common.rng_state(direct_generator)

        # Deliberately disturb every random stream and initialize a new model.
        common.reseed(999)
        resumed = _TinyQuantNet()
        resumed_opt = _optimizer(resumed)
        resumed_generator = torch.Generator().manual_seed(888)
        _random_batch(resumed_generator)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["completed_steps"] == 1
        resumed.load_state_dict(saved["model"], strict=True)
        resumed_opt.load_state_dict(saved["optimizer"])
        common.restore_rng(saved["rng"], resumed_generator)
        rx2, ry2, resumed_draws = _random_batch(resumed_generator)
        assert torch.equal(x2, rx2) and torch.equal(y2, ry2)
        assert draws2[:2] == resumed_draws[:2] and torch.equal(draws2[2], resumed_draws[2])
        engine.train_step(resumed, resumed_opt, rx2, ry2, _RecordingSoftLoss(), validate=True)
        _same_rng(direct_rng, common.rng_state(resumed_generator))
        for name, value in direct.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[name]), name
        for left, right in zip(direct.parameters(), resumed.parameters()):
            assert torch.equal(left.grad, right.grad)
            assert torch.equal(direct_opt.state[left]["momentum_buffer"],
                               resumed_opt.state[right]["momentum_buffer"])
    return {"status": "PASS", "training_steps_per_route": 2,
            "safe_weights_only_roundtrip": True,
            "parameters_BN_gradients_SGD_momentum_equal": True,
            "python_numpy_torch_loader_RNG_equal": True,
            "comparison": "Uninterrupted second step versus first-step checkpoint + restored RNG + second step"}


def _scheduler_case():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [45, 60, 70], gamma=0.1)
    milestones = {0: float(optimizer.param_groups[0]["lr"])}
    saved = None
    for completed in range(1, 81):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        expected = 0.5 if completed < 45 else 0.05 if completed < 60 else 0.005 if completed < 70 else 0.0005
        actual = float(optimizer.param_groups[0]["lr"])
        assert math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15), completed
        if completed in (44, 45, 59, 60, 69, 70, 80):
            milestones[completed] = actual
        if completed == 44:
            saved = copy.deepcopy({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
    resumed_parameter = torch.nn.Parameter(torch.ones(()))
    resumed_opt = torch.optim.SGD([resumed_parameter], lr=0.5, momentum=0.9)
    resumed_scheduler = torch.optim.lr_scheduler.MultiStepLR(resumed_opt, [45, 60, 70], gamma=0.1)
    resumed_opt.load_state_dict(saved["optimizer"])
    resumed_scheduler.load_state_dict(saved["scheduler"])
    resumed_parameter.grad = torch.zeros_like(resumed_parameter)
    resumed_opt.step()
    resumed_scheduler.step()
    assert resumed_scheduler.last_epoch == 45
    assert resumed_opt.param_groups[0]["lr"] == milestones[45]
    return {"status": "PASS", "scalar_scheduler_steps": 80,
            "model_training_epochs": 0, "lr_after_completed_steps": milestones,
            "restore_at_44_then_decay_at_45_equal": True}


def _expect_rejection(callback, label):
    try:
        callback()
    except RuntimeError:
        return label
    raise AssertionError(label + " was incorrectly accepted")


def _finite_gate_case():
    import run_train

    valid = {"samples": 4, "loss": 1.25, "top1_correct": 1, "top5_correct": 3}
    run_train.add_totals(run_train.fresh_totals(), {"by_bit": {"32": valid}})
    rejected = []
    for label, changes in [("nonfinite_loss", {"loss": float("nan")}),
                           ("invalid_accuracy_counts", {"top1_correct": 5})]:
        metric = {**valid, **changes}
        rejected.append(_expect_rejection(
            lambda: run_train.add_totals(run_train.fresh_totals(), {"by_bit": {"32": metric}}), label))
    model = _TinyQuantNet()
    optimizer = _optimizer(model)
    run_train.assert_finite_checkpoint_state(model, optimizer)
    with torch.no_grad():
        original = model.weight[0, 0, 0, 0].clone()
        model.weight[0, 0, 0, 0] = float("inf")
    rejected.append(_expect_rejection(
        lambda: run_train.assert_finite_checkpoint_state(model, optimizer), "nonfinite_model"))
    with torch.no_grad():
        model.weight[0, 0, 0, 0] = original
    optimizer.state[model.weight]["momentum_buffer"] = torch.full_like(model.weight, float("nan"))
    rejected.append(_expect_rejection(
        lambda: run_train.assert_finite_checkpoint_state(model, optimizer), "nonfinite_optimizer"))
    return {"status": "PASS", "valid_values_accepted": True, "rejected": rejected}


def run_selftests():
    started = time.perf_counter()
    assert not torch.cuda.is_initialized(), "Runner selftests require a CPU-only preflight context."
    initial_rng, initial_threads = common.rng_state(), torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        checkpoint = _checkpoint_case()
        scheduler = _scheduler_case()
        finite = _finite_gate_case()
    finally:
        common.restore_rng(initial_rng)
        torch.set_num_threads(initial_threads)
    return {"status": "PASS", "device": "cpu", "torch_version": str(torch.__version__),
            "synthetic_fixture_only": True, "checkpoint_continuity": checkpoint,
            "learning_rate_schedule": scheduler, "finite_gates": finite,
            "seconds": time.perf_counter() - started}


if __name__ == "__main__":
    import json
    print(json.dumps(run_selftests(), indent=2, ensure_ascii=False))
