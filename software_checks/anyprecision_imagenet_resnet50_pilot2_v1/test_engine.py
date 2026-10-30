"""Small CPU semantic checks; no dataset, download or training experiment.

The independent DP oracle creates explicit model replicas, gathers their
outputs and backpropagates a single global mean loss per precision. It does
not use the engine's sequential weighted-loss or buffer-restoration helpers.
"""
from __future__ import annotations

import copy
import json
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import engine


class _TinyQuantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 3, 3, 3) * 0.13)
        self.banks = nn.ModuleDict({str(bit): nn.BatchNorm2d(5) for bit in (1, 2, 4, 8, 32)})
        self.classifier = nn.Linear(5, 7)
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.long))
        self.wbit = self.abit = 32

    def forward(self, inputs: Tensor) -> Tensor:
        weight = torch.tanh(self.weight)
        if self.wbit != 32:
            levels = float(2 ** self.wbit - 1)
            rounded = torch.round((weight + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0
            weight = weight + (rounded - weight).detach()
        features = F.conv2d(inputs, weight, padding=1)
        features = F.relu(self.banks[str(self.abit)](features))
        if self.abit != 32:
            features = features.clamp(0.0, 1.0)
            levels = float(2 ** self.abit - 1)
            rounded = torch.round(features * levels) / levels
            features = features + (rounded - features).detach()
        if self.training:
            self.forward_counter.add_(1)
        return self.classifier(features.mean((2, 3)))


class _RecordingSoftLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[Tensor] = []

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if targets.requires_grad:
            raise AssertionError("Recursive teachers must be detached.")
        self.targets.append(targets.detach().clone())
        return -(targets * logits.log_softmax(dim=1)).sum(dim=1).mean()


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.035, momentum=0.9, weight_decay=1e-5)
        self.updates = 0

    def step(self, closure=None):
        self.updates += 1
        return super().step(closure)


def _data(samples: int) -> tuple[Tensor, Tensor]:
    inputs = torch.randn(samples, 3, 6, 6)
    inputs += torch.linspace(-1.7, 2.1, samples).reshape(-1, 1, 1, 1)
    labels = torch.arange(samples, dtype=torch.long).remainder(7)
    return inputs, labels


def _oracle_metrics(logits: Tensor, labels: Tensor, loss: Tensor) -> dict[str, int | float]:
    predictions = logits.detach().topk(5, dim=1).indices
    return {
        "loss": float(loss.detach()),
        "top1_correct": int((predictions[:, 0] == labels).sum()),
        "top5_correct": int((predictions == labels[:, None]).any(dim=1).sum()),
        "samples": labels.numel(),
    }


def _explicit_replicas_oracle(model, optimizer, inputs, labels, soft_loss, replicas=4):
    """Independent stand-in for scatter -> parallel replicas -> gather."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    teacher = None
    chunks = inputs.chunk(replicas)
    for bit in (32, 8, 4, 2, 1):
        copies = [copy.deepcopy(model) for _ in chunks]
        outputs = []
        for replica, shard in zip(copies, chunks):
            replica.zero_grad(set_to_none=True)
            for module in replica.modules():
                module.wbit = module.abit = bit
            outputs.append(replica(shard))
        gathered = torch.cat(outputs, dim=0)
        loss = F.cross_entropy(gathered, labels) if teacher is None else soft_loss(gathered, teacher)
        loss.backward()
        # DataParallel sums the replica gradients into the original module.
        for index, master in enumerate(model.parameters()):
            for replica in copies:
                gradient = list(replica.parameters())[index].grad
                if gradient is not None:
                    if master.grad is None:
                        master.grad = gradient.detach().clone()
                    else:
                        master.grad.add_(gradient)
        # Only the primary device's in-place buffer updates survive replication.
        with torch.no_grad():
            for destination, source in zip(model.buffers(), copies[0].buffers()):
                destination.copy_(source)
        teacher = gathered.detach().softmax(dim=1)
        metrics[str(bit)] = _oracle_metrics(gathered, labels, loss)
    optimizer.step()
    return metrics


def _native_oracle(model, optimizer, inputs, labels, soft_loss):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    teacher = None
    metrics = {}
    for bit in (32, 8, 4, 2, 1):
        for module in model.modules():
            module.wbit = module.abit = bit
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels) if teacher is None else soft_loss(logits, teacher)
        loss.backward()
        teacher = logits.detach().softmax(dim=1)
        metrics[str(bit)] = _oracle_metrics(logits, labels, loss)
    optimizer.step()
    return metrics


def _naive_accumulation(model, optimizer, inputs, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    chunks = tuple(zip(inputs.chunk(4), labels.chunk(4)))
    teachers = None
    for bit in (32, 8, 4, 2, 1):
        for module in model.modules():
            module.wbit = module.abit = bit
        next_teachers = []
        for index, (shard, target) in enumerate(chunks):
            logits = model(shard)
            loss = (F.cross_entropy(logits, target) if teachers is None else
                    -(teachers[index] * logits.log_softmax(dim=1)).sum(dim=1).mean())
            (loss * (len(shard) / len(inputs))).backward()
            next_teachers.append(logits.detach().softmax(dim=1))
        teachers = next_teachers
    optimizer.step()


def _compare_models(actual, expected) -> dict[str, float]:
    maximum = {"parameter_max_abs": 0.0, "gradient_max_abs": 0.0, "buffer_max_abs": 0.0}
    for (name, value), (other_name, other) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(value, other, rtol=5e-5, atol=2e-6, msg=name)
        maximum["parameter_max_abs"] = max(maximum["parameter_max_abs"], float((value - other).abs().max().detach()))
        assert value.grad is not None and other.grad is not None, name
        torch.testing.assert_close(value.grad, other.grad, rtol=5e-5, atol=2e-5, msg=name + ".grad")
        maximum["gradient_max_abs"] = max(maximum["gradient_max_abs"], float((value.grad - other.grad).abs().max()))
    for (name, value), (other_name, other) in zip(actual.named_buffers(), expected.named_buffers()):
        assert name == other_name
        torch.testing.assert_close(value, other, rtol=1e-5, atol=1e-6, msg=name)
        maximum["buffer_max_abs"] = max(maximum["buffer_max_abs"], float((value - other).abs().max()))
    return maximum


def _compare_metrics(actual, expected) -> None:
    assert set(actual) == set(expected)
    for bit in expected:
        for key in ("top1_correct", "top5_correct", "samples"):
            assert actual[bit][key] == expected[bit][key], (bit, key)
        assert math.isclose(actual[bit]["loss"], expected[bit]["loss"], rel_tol=2e-6, abs_tol=2e-6), bit


def _serial_case(samples: int) -> dict[str, Any]:
    torch.manual_seed(512 + samples)
    start = _TinyQuantNet()
    actual, expected = copy.deepcopy(start), copy.deepcopy(start)
    inputs, labels = _data(samples)
    actual_optimizer, expected_optimizer = _CountingSGD(actual.parameters()), _CountingSGD(expected.parameters())
    actual_soft, expected_soft = _RecordingSoftLoss(), _RecordingSoftLoss()
    result = engine.train_step(actual, actual_optimizer, inputs, labels, actual_soft, validate=True)
    oracle = _explicit_replicas_oracle(expected, expected_optimizer, inputs, labels, expected_soft)
    maxima = _compare_models(actual, expected)
    _compare_metrics(result["by_bit"], oracle)
    chunks = len(inputs.chunk(4))
    assert actual_optimizer.updates == expected_optimizer.updates == result["optimizer_updates"] == 1
    assert result["forwards"] == 5 * chunks
    assert result["samples"] == samples
    assert actual.forward_counter.item() == 5
    assert len(actual_soft.targets) == 4 * chunks and len(expected_soft.targets) == 4
    for index, expected_teacher in enumerate(expected_soft.targets):
        gathered_teacher = torch.cat(actual_soft.targets[index * chunks:(index + 1) * chunks], dim=0)
        torch.testing.assert_close(gathered_teacher, expected_teacher, rtol=1e-6, atol=1e-6)
    for bank in actual.banks.values():
        assert bank.num_batches_tracked.item() == 1
    for parameter, oracle_parameter in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(actual_optimizer.state[parameter]["momentum_buffer"],
                                   expected_optimizer.state[oracle_parameter]["momentum_buffer"],
                                   rtol=5e-5, atol=2e-5)
    return {"status": "PASS", "samples": samples, "chunk_sizes": result["chunk_sizes"],
            "one_update": True, "recursive_teacher_order": True, "replica_zero_buffers": True, **maxima}


def _native_case() -> dict[str, Any]:
    torch.manual_seed(890)
    start = _TinyQuantNet()
    actual, expected = copy.deepcopy(start), copy.deepcopy(start)
    inputs, labels = _data(8)
    result = engine.train_step(actual, _CountingSGD(actual.parameters()), inputs, labels,
                               _RecordingSoftLoss(), mode="native256", validate=True)
    oracle = _native_oracle(expected, _CountingSGD(expected.parameters()), inputs, labels, _RecordingSoftLoss())
    maxima = _compare_models(actual, expected)
    _compare_metrics(result["by_bit"], oracle)
    assert result["forwards"] == 5 and result["chunk_sizes"] == [8]
    return {"status": "PASS", **maxima}


def _bn_difference_case() -> dict[str, Any]:
    torch.manual_seed(1209)
    start = _TinyQuantNet()
    serial, native, naive = (copy.deepcopy(start) for _ in range(3))
    inputs, labels = _data(8)
    engine.train_step(serial, _CountingSGD(serial.parameters()), inputs, labels, _RecordingSoftLoss(), validate=True)
    engine.train_step(native, _CountingSGD(native.parameters()), inputs, labels, _RecordingSoftLoss(),
                      mode="native256", validate=True)
    _naive_accumulation(naive, _CountingSGD(naive.parameters()), inputs, labels)
    serial_bn, native_bn, naive_bn = (model.banks["32"] for model in (serial, native, naive))
    native_difference = float((serial_bn.running_var - native_bn.running_var).abs().max())
    naive_difference = float((serial_bn.running_mean - naive_bn.running_mean).abs().max())
    assert native_difference > 1e-5 and naive_difference > 1e-5
    assert serial_bn.num_batches_tracked.item() == native_bn.num_batches_tracked.item() == 1
    assert naive_bn.num_batches_tracked.item() == 4
    assert serial.forward_counter.item() == 5 and naive.forward_counter.item() == 20
    return {"status": "PASS", "native_batch_changes_BN": True,
            "naive_accumulation_changes_running_buffers": True,
            "native_running_var_max_abs_difference": native_difference,
            "naive_running_mean_max_abs_difference": naive_difference,
            "serial_BN_updates_per_mode": 1, "naive_BN_updates_per_mode": 4}


def _evaluation_case() -> dict[str, Any]:
    torch.manual_seed(712)
    model = _TinyQuantNet()
    inputs, labels = _data(5)
    engine.train_step(model, _CountingSGD(model.parameters()), inputs, labels, _RecordingSoftLoss())
    model.banks["2"].eval()
    old_flags = [module.training for module in model.modules()]
    old_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    old_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    expected = copy.deepcopy(model).eval()
    result = engine.eval_batch(model, inputs, labels)
    oracle = {}
    with torch.no_grad():
        for bit in (1, 2, 4, 8, 32):
            for module in expected.modules():
                module.wbit = module.abit = bit
            logits = expected(inputs)
            oracle[str(bit)] = _oracle_metrics(logits, labels, F.cross_entropy(logits, labels))
    _compare_metrics(result["by_bit"], oracle)
    assert result["optimizer_updates"] == 0 and result["forwards"] == 5
    assert old_flags == [module.training for module in model.modules()]
    for name, value in model.state_dict().items():
        assert torch.equal(value, old_state[name]), name
    for parameter, previous in zip(model.parameters(), old_gradients):
        assert torch.equal(parameter.grad, previous)
    return {"status": "PASS", "all_modes_hard_CE": True,
            "buffers_parameters_gradients_unchanged": True, "module_training_flags_restored": True}


def _validation_rejection_cases() -> dict[str, Any]:
    torch.manual_seed(90)
    model = _TinyQuantNet()
    optimizer = _CountingSGD(model.parameters())
    inputs, labels = _data(4)
    rejected = []
    for label, arguments in [
        ("unsupported_mode", {"mode": "other"}),
        ("wrong_replica_count", {"replicas": 2}),
    ]:
        try:
            engine.train_step(model, optimizer, inputs, labels, _RecordingSoftLoss(), **arguments)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label + " was accepted")
    try:
        engine.train_step(model, optimizer, inputs.double(), labels, _RecordingSoftLoss())
    except TypeError:
        rejected.append("non_FP32_inputs")
    else:
        raise AssertionError("Non-FP32 inputs were accepted")
    invalid = inputs.clone()
    invalid[0, 0, 0, 0] = float("nan")
    try:
        engine.train_step(model, optimizer, invalid, labels, _RecordingSoftLoss(), validate=True)
    except FloatingPointError:
        rejected.append("nonfinite_input")
    else:
        raise AssertionError("Nonfinite inputs were accepted")
    assert optimizer.updates == 0
    unused_model = _TinyQuantNet()
    unused_model.register_parameter("unused", nn.Parameter(torch.ones(())))
    unused_optimizer = _CountingSGD(unused_model.parameters())
    try:
        engine.train_step(unused_model, unused_optimizer, inputs, labels, _RecordingSoftLoss(), validate=True)
    except RuntimeError as error:
        assert "Missing required gradients" in str(error)
        rejected.append("missing_required_gradient")
    else:
        raise AssertionError("Missing gradients were accepted")
    assert unused_optimizer.updates == 0
    return {"status": "PASS", "rejected_before_optimizer_update": rejected}


def run_selftests() -> dict[str, Any]:
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            serial = [_serial_case(samples) for samples in (8, 13, 5, 1)]
            native = _native_case()
            difference = _bn_difference_case()
            evaluation = _evaluation_case()
            rejection = _validation_rejection_cases()
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "status": "PASS", "device": "cpu", "torch_version": torch.__version__,
        "synthetic_fixture_only": True, "real_dataset_loaded": False,
        "independent_oracle": "explicit deep-copied replicas, gathered outputs, one global mean loss per mode",
        "serial_cases": serial, "native_case": native, "BN_difference_case": difference,
        "evaluation_case": evaluation, "rejection_cases": rejection,
        "tolerances": {"parameter_atol": 2e-6, "gradient_atol": 2e-5,
                       "parameter_gradient_rtol": 5e-5, "buffer_atol": 1e-6, "buffer_rtol": 1e-5},
        "seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    print(json.dumps(run_selftests(), ensure_ascii=False, indent=2))
