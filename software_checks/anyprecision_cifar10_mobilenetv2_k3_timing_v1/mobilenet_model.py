"""Documented CIFAR-10 MobileNetV2 reconstruction for a bounded timing probe.

This is not the missing authors' MobileNetV2 implementation.  The V2 stages,
linear projections, residual connections, initialization and dropout follow
torchvision's MobileNetV2 layout.  Explicit changes are the CIFAR stem stride,
K=3 grouped spatial convolutions and the vendored Any-Precision operators.

The mode's activation bit width applies to the placed activation quantizers.
The stem, linear projection / residual output and classifier boundary stay
floating point; an extra clamp after a linear bottleneck would change V2.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from upstream.quan_ops import (
    SwitchBatchNorm2d,
    activation_quantize_fn,
    conv2d_quantize_fn,
)


BITS = (1, 2, 4, 8, 32)
TRAIN_BITS = (32, 8, 4, 2, 1)
GROUP_K = 3
CHANNELS_PER_GROUP = 2 ** GROUP_K
STAGES = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
)


class AnyPrecisionActivation(nn.Module):
    """The released Any-Precision ReLU32 / clamp-and-round low-bit rule."""

    def __init__(self):
        super().__init__()
        self.abit = 32
        self.wbit = 32
        self.quantizer = activation_quantize_fn(list(BITS))

    def forward(self, x: Tensor) -> Tensor:
        if self.abit == 32:
            x = torch.relu(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        return self.quantizer(x)


def _bn(channels: int) -> SwitchBatchNorm2d:
    return SwitchBatchNorm2d(channels, list(BITS))


class InvertedResidual(nn.Module):
    """V2 expand -> grouped spatial -> linear project, with the usual skip."""

    def __init__(self, inp: int, out: int, stride: int, expansion: int):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("MobileNetV2 stride must be 1 or 2.")
        self.in_channels = inp
        self.out_channels = out
        self.stride = stride
        self.expansion = expansion
        self.hidden_channels = inp * expansion
        if self.hidden_channels % CHANNELS_PER_GROUP:
            raise ValueError("K=3 requires a spatial input-channel count divisible by 8.")
        self.groups = self.hidden_channels // CHANNELS_PER_GROUP
        self.use_res_connect = stride == 1 and inp == out
        Conv = conv2d_quantize_fn(list(BITS))
        if expansion != 1:
            self.expand = nn.Sequential(
                Conv(inp, self.hidden_channels, 1, bias=False),
                _bn(self.hidden_channels),
                AnyPrecisionActivation(),
            )
        else:
            self.expand = nn.Identity()
        self.spatial = nn.Sequential(
            Conv(self.hidden_channels, self.hidden_channels, 3, stride=stride,
                 padding=1, groups=self.groups, bias=False),
            _bn(self.hidden_channels),
            AnyPrecisionActivation(),
        )
        # Deliberately no activation / quantizer after the linear projection.
        self.project = nn.Sequential(
            Conv(self.hidden_channels, out, 1, bias=False),
            _bn(out),
        )

    def forward(self, x: Tensor) -> Tensor:
        result = self.project(self.spatial(self.expand(x)))
        return x + result if self.use_res_connect else result


class MobileNetV2AnyPrecision(nn.Module):
    def __init__(self):
        super().__init__()
        self.wbit = 32
        self.abit = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            _bn(32),
            nn.ReLU(inplace=False),
        )
        blocks = []
        inp = 32
        for expansion, out, count, stride in STAGES:
            for index in range(count):
                blocks.append(InvertedResidual(inp, out, stride if index == 0 else 1, expansion))
                inp = out
        self.blocks = nn.ModuleList(blocks)
        Conv = conv2d_quantize_fn(list(BITS))
        self.final = nn.Sequential(
            Conv(inp, 1280, 1, bias=False),
            _bn(1280),
            AnyPrecisionActivation(),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1280, 10)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(self.dropout(x))


def build_model() -> MobileNetV2AnyPrecision:
    return MobileNetV2AnyPrecision()


def select_precision(model: nn.Module, bit: int) -> None:
    if bit not in BITS:
        raise ValueError(f"Unsupported precision {bit}; expected one of {BITS}.")
    # Match the released trainer: also set nested quantizers and BN switches.
    model.apply(lambda module: setattr(module, "wbit", bit))
    model.apply(lambda module: setattr(module, "abit", bit))


def model_summary(model: MobileNetV2AnyPrecision) -> dict[str, Any]:
    bns = [m for m in model.modules() if isinstance(m, SwitchBatchNorm2d)]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    quantized = [m for m in convs if hasattr(m, "quantize_fn")]
    params = sum(p.numel() for p in model.parameters())
    master_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    float_buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers()
                             if b.is_floating_point())
    groups = []
    for index, block in enumerate(model.blocks):
        groups.append({
            "block": index, "input_channels": block.in_channels,
            "hidden_channels": block.hidden_channels,
            "output_channels": block.out_channels, "stride": block.stride,
            "expansion": block.expansion, "groups": block.groups,
            "channels_per_group": CHANNELS_PER_GROUP,
            "residual": block.use_res_connect,
        })
    return {
        "name": "MobileNetV2-CIFAR10-K3-reconstruction",
        "exact_paper_reproduction": False,
        "width_multiplier": 1.0, "input_shape": [3, 32, 32],
        "num_classes": 10, "stem_stride": 1, "group_k": GROUP_K,
        "channels_per_group": CHANNELS_PER_GROUP,
        "stages_t_c_n_s": [list(stage) for stage in STAGES],
        "dropout_probability": 0.2,
        "training_precisions": list(TRAIN_BITS),
        "evaluation_precisions": list(BITS),
        "parameter_count": params, "fp32_master_bytes": master_bytes,
        "floating_buffer_bytes": float_buffer_bytes,
        "parameter_gradient_adam_moments_bytes_estimate": 4 * master_bytes,
        "optimizer_memory_estimate_excludes": "activations, workspaces, caches, step tensors and buffers",
        "conv_count": len(convs), "quantized_conv_count": len(quantized),
        "grouped_spatial_conv_count": len(groups),
        "switch_bn_count": len(bns), "bn_bank_count": len(bns) * len(BITS),
        "activation_quantizer_count": sum(isinstance(m, AnyPrecisionActivation) for m in model.modules()),
        "all_master_parameters_fp32": all(p.dtype == torch.float32 for p in model.parameters()),
        "first_and_last_weight_layers": "FP32 stem Conv2d and FP32 final Linear",
        "weight_quantization": "Unchanged released Any-Precision tanh/E quantizer, including its non-identity 32-bit path",
        "activation_quantization": "At placed nonlinear internal activations: ReLU for 32, clamp[0,1] and round for 1/2/4/8",
        "activation_fp32_exceptions": [
            "Raw normalized input, stem convolution and stem ReLU are not activation-quantized.",
            "Linear projection BN outputs and residual additions have no extra activation or quantizer.",
            "Consequently the next expansion convolution can consume a floating-point linear/residual output.",
            "Pooling, dropout, classifier and softmax/loss remain floating-point.",
        ],
        "precision_label_scope": "A bits describe placed quantizers, not every intermediate tensor or every convolution input",
        "bn_policy": "Independent bank for each of 1/2/4/8/32 at every BN site, including stem; each selected once per full update",
        "spatial_groups": groups,
    }
