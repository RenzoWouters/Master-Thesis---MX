"""Small CPU correctness checks for the explicit ResNet-18 reconstruction.

The full topology is compared to the untouched TruncQuant quantization route.
An independent four-replica oracle gathers logits and backpropagates a global
loss, rather than reproducing the engine's serial weighted-loss calculation.
All inputs are synthetic and spatially small. No ImageNet accuracy is measured.
"""
from __future__ import annotations

import copy
import gc
import importlib
import math
from pathlib import Path
import shutil
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

import resnet18_model as m


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.3, momentum=0.9, weight_decay=1e-5)
        self.updates = 0

    def step(self, closure=None):
        self.updates += 1
        return super().step(closure)


def _select(model, bit):
    for module in model.modules():
        module.wbit = bit
        module.abit = bit


def _maximum(a, b):
    return float((a.detach().to(torch.float64) - b.detach().to(torch.float64)).abs().max())


def _compare_models(actual, expected, *, exact=False):
    parameters = dict(expected.named_parameters())
    buffers = dict(expected.named_buffers())
    assert list(dict(actual.named_parameters())) == list(parameters)
    assert list(dict(actual.named_buffers())) == list(buffers)
    atol_p, atol_g, rtol = (0.0, 0.0, 0.0) if exact else (3e-6, 1e-5, 1e-4)
    maxima = {"parameters": 0.0, "gradients": 0.0, "buffers": 0.0}
    for name, value in actual.named_parameters():
        other = parameters[name]
        torch.testing.assert_close(value, other, rtol=rtol, atol=atol_p, msg=name)
        maxima["parameters"] = max(maxima["parameters"], _maximum(value, other))
        assert (value.grad is None) == (other.grad is None), name
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, rtol=rtol, atol=atol_g,
                                       msg=name + ".grad")
            maxima["gradients"] = max(maxima["gradients"], _maximum(value.grad, other.grad))
    for name, value in actual.named_buffers():
        other = buffers[name]
        torch.testing.assert_close(value, other, rtol=0, atol=0, msg=name)
        maxima["buffers"] = max(maxima["buffers"], _maximum(value, other))
    return maxima


def _compare_optimizers(actual, expected, actual_model, expected_model, *, exact=False):
    rtol, atol = (0.0, 0.0) if exact else (1e-4, 1e-5)
    maximum = 0.0
    for a, b in zip(actual_model.parameters(), expected_model.parameters()):
        sa, sb = actual.state[a], expected.state[b]
        assert set(sa) == set(sb) == {"momentum_buffer"}
        torch.testing.assert_close(sa["momentum_buffer"], sb["momentum_buffer"], rtol=rtol, atol=atol)
        maximum = max(maximum, _maximum(sa["momentum_buffer"], sb["momentum_buffer"]))
    assert actual.updates == expected.updates == 1
    return maximum


def _assert_counts(model, *, switch_count, shared_count):
    banks, shared = {}, {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            count = int(module.num_batches_tracked)
            if ".bn_dict." in name:
                banks[name] = count
                assert count == switch_count, (name, count, switch_count)
            else:
                shared[name] = count
                assert count == shared_count, (name, count, shared_count)
    assert len(banks) == 80 and len(shared) == 4
    return {"precision_specific_banks": len(banks), "updates_per_bank": switch_count,
            "shared_BN_modules": len(shared), "updates_per_shared_BN": shared_count}


def _paired_models(any_source, trunc_source):
    before = torch.get_rng_state().clone()
    actual = m.build_model(any_source, trunc_source)
    after_actual = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    expected = m.build_reference_model(trunc_source)
    assert torch.equal(torch.get_rng_state(), after_actual), "Initialization consumes a different RNG stream."
    assert list(actual.state_dict()) == list(expected.state_dict())
    for name, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name]), name
    return actual, expected


def _topology_checks(model):
    report = m.model_report(model)
    assert report["stage_blocks"] == [2, 2, 2, 2]
    assert report["stage_channels"] == [64, 128, 256, 512]
    assert report["parameter_count"] == 11720232, report["parameter_count"]
    assert report["output_classes"] == 1000
    assert len(report["quantized_convolution_paths"]) == 16
    assert len(report["activation_quantizer_paths"]) == 16
    assert report["FP32_convolution_paths"] == ["conv1", "layer2.0.downsample.0",
                                               "layer3.0.downsample.0", "layer4.0.downsample.0"]
    assert report["shared_BN_paths"] == ["bn1", "layer2.0.downsample.1",
                                        "layer3.0.downsample.1", "layer4.0.downsample.1"]
    assert len(report["switchable_BN_paths"]) == 16
    assert model.conv1.kernel_size == (7, 7) and model.conv1.stride == (2, 2)
    assert model.maxpool.kernel_size == 3 and model.maxpool.stride == 2
    for module in model.modules():
        if hasattr(module, "quantize_fn"):
            assert type(module.quantize_fn).__module__.startswith("_r18_original_ops_")
            assert not hasattr(module.quantize_fn, "trunc")
        if hasattr(module, "bn_dict"):
            assert list(module.bn_dict) == [str(bit) for bit in m.BITS]
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return report


def _full_topology_case(any_source, trunc_source):
    torch.manual_seed(20031)
    actual, expected = _paired_models(any_source, trunc_source)
    architecture = _topology_checks(actual)
    actual.train()
    expected.train()
    actual_optimizer, expected_optimizer = CountingSGD(actual.parameters()), CountingSGD(expected.parameters())
    actual_optimizer.zero_grad(set_to_none=True)
    expected_optimizer.zero_grad(set_to_none=True)
    soft = m.build_soft_loss(any_source)
    inputs = torch.randn(2, 3, 32, 32)
    x = inputs.clone().requires_grad_(True)
    x_ref = inputs.clone().requires_grad_(True)
    labels = torch.tensor([17, 731], dtype=torch.long)
    teacher = teacher_ref = None
    cases = {}
    for bit in (32, 8, 4, 2, 1):
        _select(actual, bit)
        _select(expected, bit)
        x.grad = x_ref.grad = None
        output, output_ref = actual(x), expected(x_ref)
        assert output.shape == output_ref.shape == (2, 1000)
        torch.testing.assert_close(output, output_ref, rtol=0, atol=0)
        loss = F.cross_entropy(output, labels) if teacher is None else soft(output, teacher)
        loss_ref = F.cross_entropy(output_ref, labels) if teacher_ref is None else soft(output_ref, teacher_ref)
        loss.backward()
        loss_ref.backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
        maxima = _compare_models(actual, expected, exact=True)
        teacher, teacher_ref = output.detach().softmax(1), output_ref.detach().softmax(1)
        cases[str(bit)] = {"status": "PASS", "logits_max_abs": _maximum(output, output_ref),
                           "input_gradient_max_abs": _maximum(x.grad, x_ref.grad),
                           "loss_abs_difference": float((loss.detach() - loss_ref.detach()).abs()),
                           **maxima}
    assert all(p.grad is not None for p in actual.parameters())
    counts = _assert_counts(actual, switch_count=1, shared_count=5)
    actual_optimizer.step()
    expected_optimizer.step()
    final = _compare_models(actual, expected, exact=True)
    momentum = _compare_optimizers(actual_optimizer, expected_optimizer, actual, expected, exact=True)
    return {"status": "PASS", "input_shape": [2, 3, 32, 32], "architecture": architecture,
            "initialization_state_and_RNG_exact": True, "modes": cases,
            "all_parameter_gradients_present": True, "BN_counts": counts,
            "one_joint_SGD_update": True, "post_update_max_abs": final,
            "momentum_max_abs": momentum,
            "oracle": "Unmodified pinned TruncQuant resnet18q(trunc=False), not the unavailable original Any-Precision R18."}


def _explicit_replicas_oracle(model, optimizer, inputs, labels, soft):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    chunks = inputs.chunk(4)
    teacher = None
    result = {}
    for bit in (32, 8, 4, 2, 1):
        copies = [copy.deepcopy(model) for _ in chunks]
        for replica in copies:
            replica.zero_grad(set_to_none=True)
            _select(replica, bit)
        outputs = [replica(chunk) for replica, chunk in zip(copies, chunks)]
        gathered = torch.cat(outputs)
        loss = F.cross_entropy(gathered, labels) if teacher is None else soft(gathered, teacher)
        loss.backward()
        # DP sums each replica's gradients; only replica zero's buffers persist.
        replica_parameters = [dict(replica.named_parameters()) for replica in copies]
        for name, master in model.named_parameters():
            for replica_parameters_one in replica_parameters:
                gradient = replica_parameters_one[name].grad
                if gradient is not None:
                    if master.grad is None:
                        master.grad = gradient.detach().clone()
                    else:
                        master.grad.add_(gradient)
        with torch.no_grad():
            for destination, source in zip(model.buffers(), copies[0].buffers()):
                destination.copy_(source)
        teacher = gathered.detach().softmax(1)
        predictions = gathered.detach().topk(5, dim=1).indices
        correct = predictions.eq(labels[:, None])
        result[str(bit)] = {"loss": float(loss.detach()), "top1_correct": int(correct[:, 0].sum()),
                            "top5_correct": int(correct.any(dim=1).sum()), "samples": len(labels)}
        # Do not keep prior replicas or autograd storage alive into the next mode.
        del copies, replica_parameters, replica_parameters_one, replica, outputs, gathered, loss
    optimizer.step()
    return result


def _serial_model_case(any_source, trunc_source, engine):
    torch.manual_seed(21011)
    actual, expected = _paired_models(any_source, trunc_source)
    actual_optimizer, expected_optimizer = CountingSGD(actual.parameters()), CountingSGD(expected.parameters())
    inputs = torch.randn(15, 3, 32, 32)
    inputs += torch.linspace(-0.5, 0.7, 15).reshape(-1, 1, 1, 1)
    labels = torch.arange(15, dtype=torch.long) * 61
    before_rng = torch.get_rng_state().clone()
    measured = engine.train_step(actual, actual_optimizer, inputs, labels,
                                 m.build_soft_loss(any_source), mode="serial_dp4", validate=True)
    assert torch.equal(before_rng, torch.get_rng_state()), "Serial model update unexpectedly consumes RNG."
    expected_metrics = _explicit_replicas_oracle(expected, expected_optimizer, inputs, labels,
                                                m.build_soft_loss(any_source))
    assert torch.equal(before_rng, torch.get_rng_state()), "Reference update unexpectedly consumes RNG."
    maxima = _compare_models(actual, expected)
    momentum = _compare_optimizers(actual_optimizer, expected_optimizer, actual, expected)
    for bit, metrics in expected_metrics.items():
        current = measured["by_bit"][bit]
        for key in ("top1_correct", "top5_correct", "samples"):
            assert current[key] == metrics[key], (bit, key)
        assert math.isclose(current["loss"], metrics["loss"], rel_tol=3e-6, abs_tol=3e-6), bit
    assert measured["chunk_sizes"] == [4, 4, 4, 3]
    assert measured["forwards"] == 20 and measured["optimizer_updates"] == 1
    counts = _assert_counts(actual, switch_count=1, shared_count=5)
    return {"status": "PASS", "input_shape": [15, 3, 32, 32], "chunk_sizes": [4, 4, 4, 3],
            "oracle": "Explicit untouched TruncQuant model replicas; gathered global loss; summed replica gradients.",
            "one_SGD_update": True, "RNG_unchanged": True, "BN_counts": counts,
            "state_max_abs": maxima, "momentum_max_abs": momentum,
            "parameter_atol": 3e-6, "gradient_atol": 1e-5, "rtol": 1e-4,
            "BN_buffer_comparison": "exact"}


def _source_rejection_case(any_source, trunc_source):
    with tempfile.TemporaryDirectory(prefix="r18_source_test_") as temporary:
        root = Path(temporary)
        for kind, source in (("anyprecision", any_source), ("truncquant", trunc_source)):
            for name in m.SOURCE_HASHES[kind]:
                destination = root / kind / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(source) / name, destination)
        target = root / "anyprecision" / "models" / "quan_ops.py"
        target.write_bytes(target.read_bytes() + b"\n# changed test copy\n")
        try:
            m.build_model(root / "anyprecision", root / "truncquant")
        except RuntimeError as exc:
            assert "source differs" in str(exc)
        else:
            raise AssertionError("Changed quantizer source was accepted.")
    return {"status": "PASS", "changed_quantizer_source_rejected_before_import": True}


def run_tests(anyprecision_source, truncquant_source, engine_module=None):
    engine = engine_module or importlib.import_module("engine")
    started = time.monotonic()
    original_threads = torch.get_num_threads()
    original_rng = torch.get_rng_state().clone()
    hashes_before = {str(Path(source) / name): (Path(source) / name).read_bytes()
                     for kind, source in (("anyprecision", anyprecision_source),
                                          ("truncquant", truncquant_source))
                     for name in m.SOURCE_HASHES[kind]}
    try:
        torch.set_num_threads(4)
        with torch.random.fork_rng(devices=[]):
            full = _full_topology_case(anyprecision_source, truncquant_source)
            gc.collect()
            serial = _serial_model_case(anyprecision_source, truncquant_source, engine)
            rejection = _source_rejection_case(anyprecision_source, truncquant_source)
        assert torch.equal(original_rng, torch.get_rng_state())
        assert all(Path(path).read_bytes() == data for path, data in hashes_before.items())
        return {"status": "PASS", "device": "cpu", "torch_version": torch.__version__,
                "synthetic_inputs_only": True, "ImageNet_data_loaded": False,
                "full_model_equivalence": full, "serial_replica_equivalence": serial,
                "source_rejection": rejection, "source_files_unchanged": True,
                "caller_CPU_RNG_restored": True, "seconds": time.monotonic() - started,
                "limitation": "CPU semantic checks with small spatial inputs; no CUDA equivalence, GPU-memory estimate or ImageNet accuracy claim."}
    finally:
        torch.set_num_threads(original_threads)

