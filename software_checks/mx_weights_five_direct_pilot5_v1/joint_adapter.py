"""Five directly quantized MX weight formats; A32 and shared FP32 masters.

Format names identify quantizers and BN banks. No format is made by truncating
another. The five original BN banks are renamed, with every tensor preserved.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
import mx_adapter as original_adapter
import mx_e4m3
import mx_e5m2
import mx_format as mx_e2m3
import mx_e3m2
import mx_e2m1

MODES = ('E4M3', 'E5M2', 'E2M3', 'E3M2', 'E2M1')
EVAL_ORDER = ('E2M1', 'E2M3', 'E3M2', 'E4M3', 'E5M2')
BITS = {'E4M3': 8, 'E5M2': 8, 'E2M3': 6, 'E3M2': 6, 'E2M1': 4}
SOURCE_BANK = {'E4M3': '8', 'E5M2': '1', 'E2M3': '32', 'E3M2': '2', 'E2M1': '4'}
BANK = {name: name for name in MODES}
CODECS = dict(zip(MODES, (mx_e4m3, mx_e5m2, mx_e2m3, mx_e3m2, mx_e2m1)))
FORMATS = {name: module.FORMAT for name, module in CODECS.items()}
TARGET_LAYERS = original_adapter.TARGET_LAYERS
BN_LAYERS = ('layers.4', 'layers.7', 'layers.11', 'layers.14', 'layers.17')
require = mx_e4m3.require
effective_reference_weight = original_adapter.effective_reference_weight


def codec(name):
    require(isinstance(name, str) and name in CODECS,
            'Selecteer een formaatnaam: E4M3, E5M2, E2M3, E3M2 of E2M1.')
    return CODECS[name]


class JointMXQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_format = 'E4M3'
        self.abit = self.wbit = 32

    def forward(self, weight):
        require(self.abit == self.wbit == 32, 'A32/W32-bronvoorbewerking gewijzigd.')
        return codec(self.weight_format).fake_quantize_mx(effective_reference_weight(weight))


class FormatBatchNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        require(set(original.bn_dict) == set(SOURCE_BANK.values()), 'Onverwachte originele BN-banken.')
        self.bn_dict = nn.ModuleDict({name: original.bn_dict[old] for name, old in SOURCE_BANK.items()})
        self.weight_format = 'E4M3'
        self.abit = self.wbit = 32

    def forward(self, x):
        require(self.abit == self.wbit == 32, 'Activaties moeten A32 blijven.')
        codec(self.weight_format)
        return self.bn_dict[self.weight_format](x)


WeightModeBatchNorm = FormatBatchNorm


def select_mode(model, name):
    codec(name)
    for module in model.modules():
        if isinstance(module, (JointMXQuantizer, FormatBatchNorm)):
            module.weight_format = name
    model.weight_format = name


def renamed_state(state):
    """Expected new key names, without changing any tensors or their order."""
    reverse = {old: name for name, old in SOURCE_BANK.items()}
    result = {}
    for key, value in state.items():
        if '.bn_dict.' in key:
            prefix, suffix = key.split('.bn_dict.', 1)
            bank, tail = suffix.split('.', 1)
            key = prefix + '.bn_dict.' + reverse[bank] + '.' + tail
        result[key] = value
    return result


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
        setattr(model.get_submodule(parent), child, FormatBatchNorm(model.get_submodule(name)))
    original_adapter.select_a32(model)
    select_mode(model, 'E4M3')
    expected, actual = renamed_state(reference.state_dict()), model.state_dict()
    require(set(expected) == set(actual) and all(torch.equal(expected[k], actual[k]) for k in expected),
            'Adaptatie wijzigde de oorspronkelijke initialisatietensors.')
    return model


@torch.no_grad()
def export_payload(model, name, provenance):
    f = codec(name)
    encoded, rows = {}, []
    for layer in TARGET_LAYERS:
        effective = effective_reference_weight(model.get_submodule(layer).weight.detach())
        mx = f.encode_tensor(effective)
        encoded[layer] = mx.payload()
        rows.append({'weight_format': name, 'weight_bits': BITS[name], 'layer': layer,
                     **f.distortion(effective, mx)})
    excluded = {n + '.weight' for n in TARGET_LAYERS}
    auxiliary = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in excluded}
    return {'schema': 'mx-five-direct-single-mode-export-v1', 'model': 'svhnq', 'expand': 2,
            'weight_format': name, 'mx_format': FORMATS[name], 'weight_bits': BITS[name],
            'activation_bits': 32, 'bn_bank_key': name, 'weight_preprocess': 'original_G(W)',
            'truncation': False, 'mx_layers': encoded, 'auxiliary_fp32_state': auxiliary,
            'auxiliary_policy': 'all five named BN banks retained; no MX-layer masters',
            'provenance': provenance}, rows


def load_exported_model(models, payload, device):
    require(payload.get('schema') == 'mx-five-direct-single-mode-export-v1', 'Onbekend exportschema.')
    name = payload['weight_format']; f = codec(name)
    require(payload['model'] == 'svhnq' and payload['expand'] == 2 and payload['activation_bits'] == 32,
            'Afwijkend model-/activatiebeleid.')
    require(payload['mx_format'] == FORMATS[name] and payload['bn_bank_key'] == name
            and payload['weight_bits'] == BITS[name] and payload['truncation'] is False
            and payload['weight_preprocess'] == 'original_G(W)', 'Afwijkend formaat-/BN-beleid.')
    require(tuple(payload['mx_layers']) == TARGET_LAYERS, 'Onvolledige MX-laagselectie.')
    model = adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected_aux = set(model.state_dict()) - {n + '.weight' for n in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected_aux, 'Onvolledige auxiliaire staat of verborgen masters.')
    for name_layer in TARGET_LAYERS:
        layer = model.get_submodule(name_layer)
        mx = f.MXTensor.from_payload(payload['mx_layers'][name_layer], device)
        require(mx.shape == tuple(layer.weight.shape), 'Verkeerde gewichtsvorm.')
        parent, child = name_layer.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, original_adapter.FrozenMXConv(layer, mx))
    missing, unexpected = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in TARGET_LAYERS} and not unexpected,
            'Onvolledige exportreconstructie.')
    select_mode(model, name)
    return model.to(device).eval().requires_grad_(False)
