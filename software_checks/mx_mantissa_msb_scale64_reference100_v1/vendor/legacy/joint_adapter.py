"""Three direct weight formats, fixed A32, one FP32 master per convolution.

E4M3 -> bank '8', E2M3 -> bank '32', E2M1 -> bank '4'. Bank keys are storage
identifiers, NOT activation bit widths. Every format independently quantizes
current G(W); no code truncation, decoded-format input or source cache exists.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
import mx_adapter as original_adapter
import mx_format as middle
import mx_e4m3 as high
import mx_e2m1 as low

TARGET_LAYERS = original_adapter.TARGET_LAYERS
BN_LAYERS = ('layers.4', 'layers.7', 'layers.11', 'layers.14', 'layers.17')
MODES = (4, 6, 8)  # Ascending evaluation order, as in the original forward.
BANK = {4: '4', 6: '32', 8: '8'}
FORMATS = {4: low.FORMAT, 6: middle.FORMAT, 8: high.FORMAT}
require = high.require
effective_reference_weight = original_adapter.effective_reference_weight


def codec(bit: int):
    require(type(bit) is int and bit in MODES, 'Verwacht gewichtsmodus 4, 6 of 8, niet activatiebits.')
    return {4: low, 6: middle, 8: high}[bit]


class JointMXQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_mode = 8
        self.abit = self.wbit = 32

    def forward(self, weight):
        require(self.abit == self.wbit == 32, 'A32 en W32-voorbewerking moeten behouden blijven.')
        # All three formats start from current G(W), independently and directly.
        return codec(self.weight_mode).fake_quantize_mx(effective_reference_weight(weight))


class WeightModeBatchNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.bn_dict = original.bn_dict
        self.bit_list = original.bit_list
        self.weight_mode = 8
        self.abit = self.wbit = 32
        require(set(self.bn_dict) == {'1', '2', '4', '8', '32'}, 'Onverwachte BN-structuur.')

    def forward(self, x):
        require(self.abit == self.wbit == 32, 'De legacy activatieselector mag niet wijzigen.')
        require(self.weight_mode in BANK, 'Onbekende BN-gewichtsmodus.')
        return self.bn_dict[BANK[self.weight_mode]](x)


def select_mode(model, bit):
    codec(bit)  # Validate before mutation.
    # Existing wbit/abit are deliberately NOT changed. This replaces the
    # three paired apply/setattr calls in the original forward function.
    for module in model.modules():
        if isinstance(module, (JointMXQuantizer, WeightModeBatchNorm)):
            module.weight_mode = bit
    model.weight_mode = bit


def adapt_copy(reference):
    model = copy.deepcopy(reference)
    names = tuple(n for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and hasattr(m, 'quantize_fn'))
    require(names == TARGET_LAYERS, 'Niet het afgesproken CNN-8-model.')
    bn_names = tuple(n for n, m in model.named_modules() if hasattr(m, 'bn_dict'))
    require(bn_names == BN_LAYERS, 'Onverwachte BatchNorm-lagen.')
    for name in TARGET_LAYERS:
        model.get_submodule(name).quantize_fn = JointMXQuantizer()
    for name in BN_LAYERS:
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, WeightModeBatchNorm(model.get_submodule(name)))
    original_adapter.select_a32(model)
    select_mode(model, 8)
    return model


def inactive_bn(name):
    return '.bn_dict.' in name and not any('.bn_dict.' + key + '.' in name for key in BANK.values())


def export_payload(model, bit, provenance):
    """Independent direct export of one format from the current shared master."""
    f = codec(bit)
    encoded, rows = {}, []
    for name in TARGET_LAYERS:
        effective = effective_reference_weight(model.get_submodule(name).weight.detach())
        mx = f.encode_tensor(effective)
        encoded[name] = mx.payload()
        rows.append({'weight_bits': bit, 'weight_format': FORMATS[bit], 'layer': name,
                     **f.distortion(effective, mx)})
    remove = {n + '.weight' for n in TARGET_LAYERS}
    auxiliary = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in remove}
    return {'schema': 'mx-triple-direct-single-mode-export-v1', 'model': 'svhnq', 'expand': 2,
            'weight_bits': bit, 'weight_format': FORMATS[bit], 'activation_bits': 32,
            'bn_bank_key': BANK[bit], 'weight_preprocess': 'original_G(W)',
            'truncation': False, 'mx_layers': encoded, 'auxiliary_fp32_state': auxiliary,
            'provenance': provenance}, rows


def load_exported_model(models, payload, device):
    require(payload.get('schema') == 'mx-triple-direct-single-mode-export-v1', 'Onbekende export.')
    bit = payload['weight_bits']; f = codec(bit)
    require(payload['model'] == 'svhnq' and payload['expand'] == 2 and payload['activation_bits'] == 32,
            'Afwijkend model-/activatiebeleid.')
    require(payload['weight_format'] == FORMATS[bit] and payload['bn_bank_key'] == BANK[bit]
            and payload['truncation'] is False and payload['weight_preprocess'] == 'original_G(W)',
            'Afwijkend formaat-/BN-beleid.')
    require(tuple(payload['mx_layers']) == TARGET_LAYERS, 'Onvolledige laagselectie.')
    model = adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected_aux = set(model.state_dict()) - {n + '.weight' for n in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected_aux, 'Ontbrekende staat of verborgen masters.')
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        # Decode on the requested execution device, as used for training/export.
        mx = f.MXTensor.from_payload(payload['mx_layers'][name], device)
        require(mx.shape == tuple(layer.weight.shape), 'Verkeerde tensorvorm.')
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, original_adapter.FrozenMXConv(layer, mx))
    missing, unexpected = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in TARGET_LAYERS} and not unexpected,
            'Onvolledige reconstructie.')
    select_mode(model, bit)
    return model.to(device).eval().requires_grad_(False)
