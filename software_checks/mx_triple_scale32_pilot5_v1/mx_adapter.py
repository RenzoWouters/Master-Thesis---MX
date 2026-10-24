"""The research change: replace only six weight-quantizer modules in a copy.

A32/ReLU and BN32 remain selected throughout the first diagnostic experiment.
The original model files and the original reference checkpoint stay untouched.
"""
from __future__ import annotations
import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mx_format import MXTensor, encode_tensor, fake_quantize_mx, distortion, require

TARGET_LAYERS = ('layers.3', 'layers.6', 'layers.10', 'layers.13', 'layers.16', 'layers.20')

def select_a32(model: nn.Module) -> None:
    # Matches the reference's 32-bit mode, including Activate and SwitchBatchNorm.
    # DO NOT set abit=6: this is a weight FORMAT, not an activation bit setting.
    model.apply(lambda m: setattr(m, 'abit', 32))
    model.apply(lambda m: setattr(m, 'wbit', 32))

def effective_reference_weight(weight: Tensor) -> Tensor:
    """Original weight_quantize_fn W32 branch, including detached layer E.

    The all-zero guard defines a previously undefined 0/0 case. No epsilon is
    inserted into nonzero layers, and no raw parameter tensor is modified.
    """
    require(bool(torch.isfinite(weight).all()), 'Niet-eindige FP32-mastergewichten.')
    e = torch.mean(torch.abs(weight)).detach()
    bounded = torch.tanh(weight)
    maximum = torch.max(torch.abs(bounded))
    divisor = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    return (bounded / divisor) * e

class MXWeightQuantizer(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.abit = 32
        self.wbit = 32

    def forward(self, weight: Tensor) -> Tensor:
        require(self.abit == 32 and self.wbit == 32,
                "Stap 1 vereist A32 en W32-bronvoorbewerking; gebruik niet de oude vijf-bitlus.")
        effective = effective_reference_weight(weight)
        return fake_quantize_mx(effective) if self.enabled else effective

def adapt_copy(reference_model: nn.Module, enabled: bool = True) -> nn.Module:
    """Copy does not mutate reference, optimizer, checkpoint, activations or BN."""
    model = copy.deepcopy(reference_model)
    actual = tuple(n for n, m in model.named_modules()
                   if isinstance(m, nn.Conv2d) and hasattr(m, 'quantize_fn'))
    require(actual == TARGET_LAYERS, 'Niet het verwachte CNN-8-model: ' + repr(actual))
    for name in TARGET_LAYERS:
        model.get_submodule(name).quantize_fn = MXWeightQuantizer(enabled)
    select_a32(model)
    return model

class FrozenMXConv(nn.Module):
    """Emulation layer reconstructed from codes, WITHOUT an FP32 master parameter.

    It decodes once to a FP32 execution buffer. This buffer is NOT a compressed
    GPU kernel. Compact storage and the emulator's RAM footprint are distinct.
    """
    def __init__(self, original: nn.Conv2d, encoded: MXTensor):
        super().__init__()
        self.stride, self.padding = original.stride, original.padding
        self.dilation, self.groups = original.dilation, original.groups
        require(original.bias is None, 'De zes CNN-8-doellagen moeten bias=False hebben.')
        self.register_buffer('decoded_weight', encoded.decode())

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(inputs, self.decoded_weight, None, self.stride,
                        self.padding, self.dilation, self.groups)

def export_payload(reference_model: nn.Module, provenance: dict) -> tuple[dict, list[dict]]:
    """One source representation for quantized weights; keep FP32 exceptions.

    All five BN banks are retained as inherited auxiliary state for transparent
    compatibility. Only bank 32 is used. They count towards reported storage.
    """
    encoded, rows = {}, []
    for name in TARGET_LAYERS:
        layer = reference_model.get_submodule(name)
        effective = effective_reference_weight(layer.weight.detach())
        mx = encode_tensor(effective)
        encoded[name] = mx.payload()
        rows.append({'layer': name, **distortion(effective, mx)})
    remove = {name + '.weight' for name in TARGET_LAYERS}
    auxiliary = {k: v.detach().cpu().clone() for k, v in reference_model.state_dict().items() if k not in remove}
    require(not remove.intersection(auxiliary), 'FP-mastergewichten lekken in MX-export.')
    payload = {'schema': 'mx-weights-e2m3-source-v1', 'model': 'svhnq', 'expand': 2,
               'activation_policy': 'A32_original_ReLU', 'bn_policy': 'existing_bank32_no_recalibration',
               'weight_preprocess': 'original_W32_tanh_normalization_before_MX',
               'mx_layers': encoded, 'auxiliary_fp32_state': auxiliary,
               'provenance': provenance}
    return payload, rows

def load_exported_model(models_module, payload: dict, device: str | torch.device) -> nn.Module:
    require(payload.get('schema') == 'mx-weights-e2m3-source-v1', 'Onbekend exportschema.')
    require(payload.get('model') == 'svhnq' and payload.get('expand') == 2, 'Onbekend model.')
    require(tuple(payload['mx_layers']) == TARGET_LAYERS, 'MX-lagen ontbreken of volgorde wijkt af.')
    require(payload.get('activation_policy') == 'A32_original_ReLU', 'Onbekend activatiebeleid.')
    require(payload.get('weight_preprocess') == 'original_W32_tanh_normalization_before_MX', 'Onbekende voorbewerking.')
    require(payload.get('bn_policy') == 'existing_bank32_no_recalibration', 'Onbekend BN-beleid.')
    model = models_module.svhnq([1, 2, 4, 8, 32], expand=2)
    expected_aux = set(model.state_dict()) - {name + '.weight' for name in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected_aux, 'Auxiliaire staat mist of bevat mastergewichten.')
    for name in TARGET_LAYERS:
        parent_name, child_name = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        original = model.get_submodule(name)
        mx = MXTensor.from_payload(payload['mx_layers'][name])
        require(tuple(original.weight.shape) == mx.shape, 'Gewichtsvorm wijkt af.')
        setattr(parent, child_name, FrozenMXConv(original, mx))
    missing, unexpected = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {name + '.decoded_weight' for name in TARGET_LAYERS} and not unexpected,
            'Onvolledige reconstructie van opgeslagen MX-model.')
    select_a32(model)
    return model.to(device).eval().requires_grad_(False)
