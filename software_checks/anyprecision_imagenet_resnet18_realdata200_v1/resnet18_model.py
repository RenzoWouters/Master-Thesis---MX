"""Explicit reconstruction: TruncQuant R18 topology + original Any-Precision ops.

The pinned upstream files are imported without editing them on disk. A private
copy of the TruncQuant architecture module receives the original Any-Precision
factories before construction. No truncation code is selected. The untouched
TruncQuant ``trunc=False`` implementation remains an independent CPU oracle.
This does not establish identity with the unavailable original R18 model.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch import nn


BITS = (1, 2, 4, 8, 32)
SOURCE_HASHES = {
    "anyprecision": {
        "models/quan_ops.py": "86fc8f68e3060f98913cb7f1b33f698f43bf8d2b6d85a3ad92c30f6e19a70f71",
        "models/losses.py": "70183b8a7e8ddd25eb86a5f3486abcce96a8407cd73851d0e5ea98772a2d70ba",
    },
    "truncquant": {
        "trunc_models/trunc_resnet_quan.py": "8c278e922dac649f80e041782ba56831845498561f6f344dc07aebb7886fd40d",
        "trunc_models/trunc_quan_ops.py": "7412992da1abac650c29545d9a51ba7b17bc06a0e31dd97b07d6e58142d6383a",
    },
}


def _checked_paths(root, source_name):
    root = Path(root).resolve()
    result = {}
    for name, expected in SOURCE_HASHES[source_name].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Pinned {source_name} source is missing: {name}.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned {source_name} source differs: {name}.")
        result[name] = path
    return result


def _load_file(name, path):
    """Load only a pinned leaf file; do not execute upstream package __init__."""
    if name in sys.modules:
        loaded = sys.modules[name]
        if Path(loaded.__file__).resolve() != path:
            raise RuntimeError("Private source-module name collision.")
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned source {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _namespace(prefix, roots):
    key = "\n".join(str(Path(root).resolve()) for root in roots)
    return prefix + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def load_original_ops(anyprecision_source):
    paths = _checked_paths(anyprecision_source, "anyprecision")
    name = _namespace("_r18_original_ops_", [anyprecision_source])
    return _load_file(name, paths["models/quan_ops.py"])


def build_soft_loss(anyprecision_source):
    paths = _checked_paths(anyprecision_source, "anyprecision")
    name = _namespace("_r18_original_loss_", [anyprecision_source])
    module = _load_file(name, paths["models/losses.py"])
    return module.CrossEntropyLossSoft()


def _load_architecture(truncquant_source, *, adapted, anyprecision_source=None):
    paths = _checked_paths(truncquant_source, "truncquant")
    roots = [truncquant_source] + ([anyprecision_source] if adapted else [])
    prefix = "_r18_adapted_" if adapted else "_r18_reference_"
    package_name = _namespace(prefix, roots)
    module_name = package_name + ".trunc_resnet_quan"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if Path(module.__file__).resolve() != paths["trunc_models/trunc_resnet_quan.py"]:
            raise RuntimeError("Private architecture-module name collision.")
        return module
    package = ModuleType(package_name)
    package.__path__ = [str(Path(truncquant_source).resolve() / "trunc_models")]
    package.__package__ = package_name
    sys.modules[package_name] = package
    _load_file(package_name + ".trunc_quan_ops", paths["trunc_models/trunc_quan_ops.py"])
    module = _load_file(module_name, paths["trunc_models/trunc_resnet_quan.py"])
    if adapted:
        original = load_original_ops(anyprecision_source)

        def original_conv_factory(bit_list, trunc=False):
            if trunc is not False:
                raise ValueError("The reconstruction does not enable weight truncation.")
            return original.conv2d_quantize_fn(bit_list)

        def original_activation_factory(bit_list, trunc=False):
            if trunc is not False:
                raise ValueError("The reconstruction does not enable activation truncation.")
            return original.activation_quantize_fn(bit_list)

        module.conv2d_quantize_fn = original_conv_factory
        module.activation_quantize_fn = original_activation_factory
        module.batchnorm_fn = original.batchnorm_fn
        module._r18_original_ops = original
    return module


def build_model(anyprecision_source, truncquant_source, num_classes=1000):
    """Build FP32 tensors; the caller selects the device and precision mode."""
    load_original_ops(anyprecision_source)  # Verify even when the module is cached.
    module = _load_architecture(truncquant_source, adapted=True,
                                anyprecision_source=anyprecision_source)
    model = module.resnet18q(list(BITS), num_classes=num_classes, trunc=False)
    model._r18_reconstruction = True
    return model


def build_reference_model(truncquant_source, num_classes=1000):
    """Unmodified TruncQuant quantization route, used only as a CPU oracle."""
    module = _load_architecture(truncquant_source, adapted=False)
    return module.resnet18q(list(BITS), num_classes=num_classes, trunc=False)


def model_report(model):
    conv_quant = []
    conv_fp32 = []
    bn_shared = []
    bn_switch = []
    activation_quantizers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            (conv_quant if hasattr(module, "quantize_fn") else conv_fp32).append(name)
        if hasattr(module, "bn_dict"):
            bn_switch.append(name)
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and ".bn_dict." not in name:
            bn_shared.append(name)
        if name.endswith(".quan"):
            activation_quantizers.append(name)
    return {
        "model": "resnet18q_reconstruction",
        "architecture_source": "TruncQuant/trunc_models/trunc_resnet_quan.py::ResNet(BasicBlock,[2,2,2,2])",
        "quantizer_source": "Original Any-Precision/models/quan_ops.py",
        "original_paper_R18_identity_verified": False,
        "stage_blocks": [len(getattr(model, f"layer{i}")) for i in range(1, 5)],
        "stage_channels": [getattr(model, f"layer{i}")[0].conv1.out_channels for i in range(1, 5)],
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "parameter_bytes": sum(value.numel() * value.element_size() for value in model.parameters()),
        "buffer_bytes": sum(value.numel() * value.element_size() for value in model.buffers()),
        "output_classes": model.fc.out_features,
        "precision_bits": list(BITS),
        "quantized_convolution_paths": conv_quant,
        "FP32_convolution_paths": conv_fp32,
        "FP32_classifier": "fc",
        "switchable_BN_paths": bn_switch,
        "switchable_BN_banks_per_module": len(BITS),
        "shared_BN_paths": bn_shared,
        "activation_quantizer_paths": activation_quantizers,
        "BN_semantics": "Branch BN banks are precision-specific; stem and projection BN are shared across precisions.",
        "FP32_weight_mode": "Original tanh/normalization/mean-absolute scaling retained in quantized convolutions.",
        "truncation": False,
        "MX": False,
    }
