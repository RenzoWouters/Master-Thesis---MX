"""CPU checks of the reconstructed architecture and full update semantics.

No dataset, download, training epoch or CUDA allocation is required.  The
structural oracle is torchvision's separately implemented MobileNetV2.  The
optimizer oracle sums all five independently written losses before backward,
so it does not reuse train_step or its soft-loss implementation.
"""
from __future__ import annotations

import copy
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from engine import create_optimizer, train_step, validate_state, validation_step
from mobilenet_model import (
    BITS, TRAIN_BITS, AnyPrecisionActivation, InvertedResidual,
    build_model, model_summary, select_precision,
)
from upstream.quan_ops import SwitchBatchNorm2d


def _require(condition, label):
    if not condition:
        raise AssertionError(label)


def _torchvision_oracle(model):
    """Adapt the official standard topology independently, then copy tensors.

    The only structural adaptations are documented separately: the stem stride
    and the spatial grouping.  Activated FP32 paths use released AP ReLU instead
    of ReLU6, and quantized-convolution weights use an independent tanh/E formula.
    """
    from torchvision.models import mobilenet_v2

    reference = mobilenet_v2(weights=None, num_classes=10, width_mult=1.0, dropout=0.2)
    target_convs = [(name, m) for name, m in reference.named_modules() if isinstance(m, nn.Conv2d)]
    source_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    _require(len(target_convs) == len(source_convs) == 52, "52 convolution sites in independent V2 topology")
    for index, ((name, old), source) in enumerate(zip(target_convs, source_convs)):
        group_count = old.in_channels // 8 if old.kernel_size == (3, 3) and old.groups == old.in_channels else old.groups
        stride = (1, 1) if index == 0 else old.stride
        replacement = nn.Conv2d(old.in_channels, old.out_channels, old.kernel_size,
                                stride=stride, padding=old.padding, dilation=old.dilation,
                                groups=group_count, bias=old.bias is not None)
        _require(tuple(replacement.weight.shape) == tuple(source.weight.shape),
                 f"Independent convolution shape: {name}")
        with torch.no_grad():
            weight = source.weight.detach()
            if hasattr(source, "quantize_fn"):
                transformed = torch.tanh(weight)
                weight = transformed / transformed.abs().max() * weight.abs().mean()
            replacement.weight.copy_(weight)
            if replacement.bias is not None:
                replacement.bias.copy_(source.bias)
        parent_name, leaf = name.rsplit(".", 1)
        setattr(reference.get_submodule(parent_name), leaf, replacement)
    source_bns = [m.bn_dict["32"] for m in model.modules() if isinstance(m, SwitchBatchNorm2d)]
    target_bns = [m for m in reference.modules() if isinstance(m, nn.BatchNorm2d)]
    _require(len(source_bns) == len(target_bns) == 52, "Independent BN-site count")
    for target, source in zip(target_bns, source_bns):
        target.load_state_dict(source.state_dict())
    for name, module in list(reference.named_modules()):
        if isinstance(module, nn.ReLU6):
            parent_name, leaf = name.rsplit(".", 1)
            setattr(reference.get_submodule(parent_name), leaf, nn.ReLU(inplace=False))
    reference.classifier[1].load_state_dict(model.classifier.state_dict())
    return reference


class _TinyPrecisionModel(nn.Module):
    """Nonlinear, stochastic toy with deliberately different teacher logits."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 10)
        self.register_buffer("offset", torch.linspace(-1.0, 1.0, 10))
        self.wbit = self.abit = 32

    def forward(self, x):
        x = x.mean(dim=(2, 3))
        x = self.head(self.dropout(torch.tanh(self.proj(x))))
        scale = {32: 0.5, 8: -0.7, 4: 1.1, 2: -1.3, 1: 0.3}[self.wbit]
        return x + scale * self.offset


def _check_update_oracle(inputs, targets):
    torch.manual_seed(1609)
    actual = _TinyPrecisionModel()
    expected = copy.deepcopy(actual)
    actual_opt, expected_opt = create_optimizer(actual), create_optimizer(expected)
    original_step = actual_opt.step
    calls = []

    def counted_step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)

    actual_opt.step = counted_step
    torch.manual_seed(321)
    actual_result = train_step(actual, actual_opt, inputs, targets, validate=True)
    actual_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(321)
    expected.train()
    expected_opt.zero_grad(set_to_none=True)
    losses = []
    teacher = None
    reference_losses = {}
    for bit in (32, 8, 4, 2, 1):
        # Independent switching and soft cross entropy (no engine helpers).
        for module in expected.modules():
            module.wbit = module.abit = bit
        output = expected(inputs)
        if teacher is None:
            loss = F.cross_entropy(output, targets)
        else:
            loss = -(teacher * F.log_softmax(output, dim=1)).sum(dim=1).mean()
        losses.append(loss)
        reference_losses[str(bit)] = float(loss.detach())
        teacher = F.softmax(output.detach(), dim=1)
    sum(losses).backward()
    expected_opt.step()
    _require(calls == [1], "Exactly one optimizer update per five passes")
    _require(torch.equal(actual_rng, torch.random.get_rng_state()), "Dropout RNG sequence agrees with oracle")
    for (name, a), (other, e) in zip(actual.named_parameters(), expected.named_parameters()):
        _require(name == other, "Oracle parameter ordering")
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-7, msg=lambda msg: f"Parameter {name}: {msg}")
        torch.testing.assert_close(a.grad, e.grad, rtol=2e-5, atol=2e-7,
                                   msg=lambda msg: f"Summed gradient {name}: {msg}")
        for field in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_opt.state[a][field], expected_opt.state[e][field],
                                       rtol=2e-5, atol=2e-8)
    for bit in ("32", "8", "4", "2", "1"):
        _require(abs(actual_result["metrics"][bit]["loss"] - reference_losses[bit]) < 2e-6,
                 "Per-precision recursive teacher loss")
    return {"optimizer_calls": len(calls), "rng_exact": True,
            "losses_match_independent_soft_ce": True}


def run_tests(progress=None):
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    tests = []

    def record(name, details=None):
        tests.append({"name": name, "passed": True, "details": details or {}})
        if progress:
            progress("CPU PASS: " + name)

    try:
        torch.manual_seed(20261030)
        model = build_model()
        summary = model_summary(model)
        inputs = torch.randn(2, 3, 32, 32)
        targets = torch.tensor([1, 7], dtype=torch.long)
        _require(len(model.blocks) == 17 and len(summary["spatial_groups"]) == 17,
                 "17 standard V2 bottlenecks")
        _require(summary["conv_count"] == 52 and summary["quantized_conv_count"] == 51,
                 "Only stem convolution is unquantized")
        _require(summary["switch_bn_count"] == 52 and summary["bn_bank_count"] == 260,
                 "All 52 BN sites have five banks, including stem")
        _require(model.stem[0].stride == (1, 1) and model.classifier.out_features == 10,
                 "CIFAR stem / classifier adaptations")
        for block in model.blocks:
            _require(block.spatial[0].groups == block.hidden_channels // 8,
                     "Spatial groups follow expanded input channels / 8")
            _require(len(block.project) == 2 and isinstance(block.project[1], SwitchBatchNorm2d),
                     "Linear projection has no appended activation")
        _require(not any(isinstance(m, AnyPrecisionActivation) for m in model.stem.modules()),
                 "Explicit FP32 stem activation")
        record("architecture_and_all_17_group_counts", summary)

        reference = _torchvision_oracle(model)
        select_precision(model, 32)
        model.train()
        reference.train()
        with torch.no_grad():
            torch.manual_seed(918)
            actual_output = model(inputs)
            torch.manual_seed(918)
            expected_output = reference(inputs)
        _require(float(actual_output.abs().max()) > 1e-4, "Nontrivial oracle logits")
        torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-6)
        record("32_bit_forward_matches_independent_torchvision_topology",
               {"max_abs_difference": float((actual_output - expected_output).abs().max())})
        del reference

        block = InvertedResidual(16, 16, 1, 6)
        select_precision(block, 32)
        block.eval()
        with torch.no_grad():
            block.project[1].bn_dict["32"].weight.zero_()
            block.project[1].bn_dict["32"].bias.fill_(-2.0)
            projected = block(torch.ones(2, 16, 4, 4))
        torch.testing.assert_close(projected, -torch.ones_like(projected), atol=0.0, rtol=0.0)
        record("linear_projection_and_residual_preserve_negative_outputs")
        del block

        banks = [m for m in model.modules() if isinstance(m, SwitchBatchNorm2d)]
        model.train()
        for bit in BITS:
            before = [[int(bank.bn_dict[str(b)].num_batches_tracked) for b in BITS] for bank in banks]
            select_precision(model, bit)
            with torch.no_grad():
                output = model(inputs)
            _require(tuple(output.shape) == (2, 10) and bool(torch.isfinite(output).all()),
                     f"Finite W{bit}/A{bit} logits")
            for bank, previous in zip(banks, before):
                for b, count in zip(BITS, previous):
                    _require(int(bank.bn_dict[str(b)].num_batches_tracked) == count + int(bit == b),
                             f"Only selected W{bit}/A{bit} BN bank advances")
        record("all_five_precisions_isolate_their_52_bn_banks")

        gradient_norms = {}
        for bit in BITS:
            model.zero_grad(set_to_none=True)
            select_precision(model, bit)
            F.cross_entropy(model(inputs), targets).backward()
            norms = {}
            for name, parameter in (("stem", model.stem[0].weight),
                                    ("first_grouped_master", model.blocks[0].spatial[0].weight),
                                    ("classifier", model.classifier.weight)):
                _require(parameter.grad is not None and parameter.grad.dtype == torch.float32,
                         f"W{bit} gradient reaches {name} FP32 master")
                _require(bool(torch.isfinite(parameter.grad).all()) and bool(parameter.grad.abs().sum() > 0),
                         f"Finite nonzero W{bit} gradient at {name}")
                norms[name] = float(parameter.grad.norm())
            gradient_norms[str(bit)] = norms
        record("each_precision_reaches_fp32_masters_with_nonzero_gradients", gradient_norms)

        record("recursive_teacher_detach_gradient_sum_and_single_adam_match_oracle",
               _check_update_oracle(inputs, targets))

        optimizer = create_optimizer(model)
        full_update = train_step(model, optimizer, inputs, targets, validate=True)
        _require(full_update["precision_order"] == list(TRAIN_BITS), "Descending five-precision update")
        _require(all(float(state["step"]) == 1 for state in optimizer.state.values()),
                 "One Adam state update on every parameter")
        validate_state(model, optimizer, require_gradients=True)
        record("complete_mobilenet_update_and_adam_state_remain_finite_fp32")

        bn_before = [int(bank.bn_dict[str(bit)].num_batches_tracked) for bank in banks for bit in BITS]
        snapshots = [p.detach().clone() for p in model.parameters()]
        validated = validation_step(model, inputs, targets, validate=True)
        bn_after = [int(bank.bn_dict[str(bit)].num_batches_tracked) for bank in banks for bit in BITS]
        _require(bn_before == bn_after, "Validation leaves BN counters unchanged")
        _require(all(torch.equal(old, new) for old, new in zip(snapshots, model.parameters())),
                 "Validation leaves master parameters unchanged")
        _require(validated["optimizer_steps"] == 0 and validated["precision_order"] == list(BITS),
                 "Validation evaluates all five precisions without updates")
        record("validation_does_not_update_weights_or_bn")

        try:
            train_step(model, optimizer, inputs.half(), targets)
        except TypeError:
            pass
        else:
            raise AssertionError("FP16 inputs must be rejected")
        record("rejects_mixed_precision_input")
        return {"status": "PASS", "device": "cpu", "cuda_tested": False,
                "elapsed_seconds": time.perf_counter() - started,
                "test_count": len(tests), "tests_run": len(tests),
                "tests": tests, "model_summary": summary}
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    print(json.dumps(run_tests(progress=print), indent=2))
