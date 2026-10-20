"""E2M3 -> E2M1 with unchanged E8M0 scales, from actual stored codes.

C uses ONLY a logical right shift of unsigned element codes; no FP32 master,
no scale selection, no rounding and no tanh transform. B is an explicitly
separate rounded-from-source control. A remains the original direct export.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor

import mx_format as high
import mx_e2m1 as low
import dual_adapter as adapter

require = high.require
DERIVED_SCHEMA = 'mx-e2m3-derived-e2m1-frozen-v1'
DIRECT_SCHEMA = 'mx-dual-direct-single-mode-export-v1'
METHODS = ('rounded_from_source', 'mantissa_truncation')
PATHS = ('H_source_E2M3', 'A_direct_E2M1', 'B_round_from_E2M3', 'C_truncate_E2M3')
BASE_KEYS = {'schema', 'model', 'expand', 'weight_bits', 'weight_format',
             'activation_bits', 'bn_bank_key', 'weight_preprocess', 'truncation',
             'mx_layers', 'auxiliary_fp32_state', 'provenance'}


def plain_metadata(value: Any) -> bool:
    """Reject tensors/hidden parameters in the provenance fields."""
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, tuple):
        return all(plain_metadata(v) for v in value)
    return type(value) is dict and all(type(k) is str and plain_metadata(v)
                                       for k, v in value.items())


def validate_direct_payload(payload: dict, bit: int) -> None:
    require(type(payload) is dict and set(payload) == BASE_KEYS, 'Onverwachte velden in directe export.')
    require(bit in (4, 6) and type(payload['weight_bits']) is int and payload['weight_bits'] == bit,
            'Verkeerde bronprecisie.')
    require(payload['schema'] == DIRECT_SCHEMA and payload['truncation'] is False,
            'Verwacht een direct afgeronde export uit de tweemodustraining.')
    require(payload['weight_format'] == adapter.FORMATS[bit] and payload['bn_bank_key'] == adapter.BANK[bit],
            'Formaat en BatchNorm-bank passen niet bij de modus.')
    require(payload['model'] == 'svhnq' and payload['expand'] == 2 and payload['activation_bits'] == 32,
            'Afwijkende architectuur of activatieprecisie.')
    require(payload['weight_preprocess'] == 'original_G(W)', 'Afwijkende gewichtsvoorbewerking.')
    require(tuple(payload['mx_layers']) == adapter.TARGET_LAYERS, 'Onvolledige of afwijkende MX-laagselectie.')
    require(plain_metadata(payload['provenance']), 'Provenance mag geen verborgen tensorstaat bevatten.')
    for name, value in payload['auxiliary_fp32_state'].items():
        require(name not in {n + '.weight' for n in adapter.TARGET_LAYERS}, 'Verborgen FP32-master in export.')
        require(isinstance(value, Tensor) and value.dtype in (torch.float32, torch.int64),
                'Onverwachte auxiliaire staat: ' + name)
        require(bool(torch.isfinite(value).all()), 'Niet-eindige auxiliaire staat: ' + name)
    for tensor_payload in payload['mx_layers'].values():
        adapter.codec(bit).MXTensor.from_payload(tensor_payload)


def truncate_codes(source: high.MXTensor) -> low.MXTensor:
    """[s ee mmm] -> [s ee m]. Keep sign/exponent/scale, including signed zero.

    uint8 stores one unpacked code; it is not interpreted as a signed integer.
    The result is NEW storage, never an in-place change of the high source.
    """
    source.validate()
    codes = (source.codes.to(torch.int64) >> 2).to(torch.uint8)
    result = low.MXTensor(codes, source.scale_codes.clone(), source.shape)
    result.validate()
    require(torch.equal((source.codes.to(torch.int64) >> 5), codes.to(torch.int64) >> 3),
            'Tekenbit veranderde tijdens truncatie.')
    require(torch.equal((source.codes.to(torch.int64) >> 3) & 3, (codes.to(torch.int64) >> 1) & 3),
            'Lokale exponent veranderde tijdens truncatie.')
    require(torch.equal(source.scale_codes, result.scale_codes), 'Gedeelde schaal veranderde.')
    return result


def round_source_codes(source: high.MXTensor) -> low.MXTensor:
    """B: nearest-even rounding of represented E2M3 elements, NOT new scales.

    Decode normalized *elements* in FP64. Do NOT call encode_tensor on decoded
    weights: that would choose a scale again and confound the comparison.
    """
    source.validate()
    codes = low.encode_elements(high.decode_elements(source.codes))
    result = low.MXTensor(codes, source.scale_codes.clone(), source.shape)
    result.validate()
    return result


def derive_payload(source: dict, method: str, source_sha256: str) -> dict:
    """Requires one HIGH export only; no direct-low export or checkpoint input."""
    validate_direct_payload(source, 6)
    require(method in METHODS, 'Onbekende afleiding.')
    require(len(source_sha256) == 64 and all(c in '0123456789abcdef' for c in source_sha256),
            'Ongeldige bron-SHA256.')
    transform = truncate_codes if method == 'mantissa_truncation' else round_source_codes
    derived = {name: transform(high.MXTensor.from_payload(value)).payload()
               for name, value in source['mx_layers'].items()}
    return {
        'schema': DERIVED_SCHEMA, 'model': 'svhnq', 'expand': 2,
        'weight_bits': 4, 'weight_format': low.FORMAT, 'activation_bits': 32,
        'bn_bank_key': '4', 'weight_preprocess': 'already_in_source; DO_NOT_reapply_G',
        'truncation': method == 'mantissa_truncation', 'mx_layers': derived,
        # Includes the trained LOW bank as well as HIGH/unused banks. Identical
        # to the checkpoint auxiliary state, not recalibrated or retrained.
        'auxiliary_fp32_state': copy.deepcopy(source['auxiliary_fp32_state']),
        'provenance': copy.deepcopy(source['provenance']),
        'derivation': {'method': method, 'source_format': high.FORMAT,
                       'source_file_sha256': source_sha256, 'dropped_mantissa_bits': 2,
                       'scale_policy': 'copy_source_E8M0_bytes_unchanged',
                       'requires_quantized_layer_FP32_masters': False,
                       'source_overwritten': False},
    }


def validate_derived(payload: dict) -> None:
    require(type(payload) is dict and set(payload) == BASE_KEYS | {'derivation'},
            'Onverwachte velden in afgeleide export.')
    require(payload['schema'] == DERIVED_SCHEMA, 'Geen afgeleide E2M1-export.')
    require(payload['model'] == 'svhnq' and payload['expand'] == 2 and payload['activation_bits'] == 32,
            'Afwijkende architectuur of activaties.')
    require(payload['weight_bits'] == 4 and payload['weight_format'] == low.FORMAT
            and payload['bn_bank_key'] == '4', 'Afwijkend formaat of verkeerde lage BN-bank.')
    require(payload['weight_preprocess'] == 'already_in_source; DO_NOT_reapply_G', 'Dubbele voorbewerking.')
    d = payload['derivation']
    require(type(d) is dict and set(d) == {'method', 'source_format', 'source_file_sha256',
                                         'dropped_mantissa_bits', 'scale_policy',
                                         'requires_quantized_layer_FP32_masters', 'source_overwritten'},
            'Onvolledige afleidingsmetadata.')
    require(d['method'] in METHODS and d['source_format'] == high.FORMAT
            and d['dropped_mantissa_bits'] == 2 and d['scale_policy'] == 'copy_source_E8M0_bytes_unchanged'
            and d['requires_quantized_layer_FP32_masters'] is False and d['source_overwritten'] is False,
            'Afleidingsbeleid wijkt af.')
    require(payload['truncation'] is (d['method'] == 'mantissa_truncation'), 'Onjuist truncatielabel.')
    require(plain_metadata(d) and plain_metadata(payload['provenance']), 'Verborgen staat in metadata.')
    require(tuple(payload['mx_layers']) == adapter.TARGET_LAYERS, 'MX-lagen ontbreken.')
    for value in payload['mx_layers'].values():
        low.MXTensor.from_payload(value)


def load_derived_model(models, payload: dict, device: str | torch.device):
    """Reconstruct without target-layer master parameters. FP32 decoded buffers
    are only emulator operands, NOT extra unquantized source weights.
    """
    validate_derived(payload)
    model = adapter.adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2))
    expected = set(model.state_dict()) - {n + '.weight' for n in adapter.TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected, 'Auxiliaire staat ontbreekt of bevat verborgen masters.')
    for name in adapter.TARGET_LAYERS:
        layer = model.get_submodule(name)
        encoded = low.MXTensor.from_payload(payload['mx_layers'][name])
        require(encoded.shape == tuple(layer.weight.shape), 'Verkeerde gewichtsvorm.')
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, adapter.original_adapter.FrozenMXConv(layer, encoded))
    missing, extra = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in adapter.TARGET_LAYERS} and not extra,
            'Onvolledige modelreconstructie.')
    adapter.select_mode(model, 4)
    model = model.to(device).eval().requires_grad_(False)
    require(not {n + '.weight' for n in adapter.TARGET_LAYERS}.intersection(dict(model.named_parameters())),
            'De afgeleide uitvoering mag geen zes FP32-masters bevatten.')
    return model


def tensor_exact(a: Tensor, b: Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
            and (not a.is_floating_point() or torch.equal(torch.signbit(a).cpu(), torch.signbit(b).cpu())))


def check_source_pair(h: dict, direct: dict) -> dict:
    validate_direct_payload(h, 6)
    validate_direct_payload(direct, 4)
    require(h['provenance'] == direct['provenance'], 'Exports horen niet bij hetzelfde checkpoint.')
    ah, al = h['auxiliary_fp32_state'], direct['auxiliary_fp32_state']
    require(set(ah) == set(al) and all(tensor_exact(ah[k], al[k]) for k in ah),
            'Directe paden hebben niet dezelfde auxiliaire toestand.')
    blocks = 0
    for name in adapter.TARGET_LAYERS:
        hh, ll = high.MXTensor.from_payload(h['mx_layers'][name]), low.MXTensor.from_payload(direct['mx_layers'][name])
        require(hh.shape == ll.shape, 'Groepsindeling verschilt.')
        require(torch.equal(hh.scale_codes, ll.scale_codes),
                'Schaalverschil in bronexports: stop in plaats van stilzwijgend schalen wijzigen.')
        blocks += hh.scale_codes.numel()
    return {'all_source_scales_equal': True, 'same_auxiliary_state': True,
            'same_checkpoint_provenance': True, 'blocks': blocks}


def weight_diagnostics(h: dict, direct: dict, rounded: dict, truncated: dict) -> tuple[list[dict], list[dict]]:
    """Code/numerical gaps; no errors relative to the absent FP32 master."""
    pair_rows, counts = [], []
    for name in adapter.TARGET_LAYERS:
        hs = high.MXTensor.from_payload(h['mx_layers'][name])
        tensors = {'A': low.MXTensor.from_payload(direct['mx_layers'][name]),
                   'B': low.MXTensor.from_payload(rounded['mx_layers'][name]),
                   'C': low.MXTensor.from_payload(truncated['mx_layers'][name])}
        cols = math.prod(hs.shape[1:])
        codes = {k: v.codes.reshape(hs.shape[0], -1)[:, :cols] for k, v in tensors.items()}
        hc = hs.codes.reshape(hs.shape[0], -1)[:, :cols]
        vals = {k: v.decode().to(torch.float64) for k, v in tensors.items()}
        hv = hs.decode().to(torch.float64)
        for ref, other in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
            err = vals[other] - vals[ref]
            pair_rows.append({'layer': name, 'comparison': other + '_minus_' + ref,
                              'n_weights': hv.numel(), 'code_differences': int((codes[other] != codes[ref]).sum()),
                              'numeric_differences': int((vals[other] != vals[ref]).sum()),
                              'mean_signed_difference': err.mean().item(), 'mae': err.abs().mean().item(),
                              'rmse': err.square().mean().sqrt().item(), 'max_abs_difference': err.abs().max().item(),
                              'reference_l2_norm': vals[ref].norm().item(), 'difference_l2_norm': err.norm().item()})
        # Verify invariants on codes, not by approximate floating point arithmetic.
        require(torch.equal(tensors['C'].codes, (hs.codes.long() >> 2).byte()), 'C is geen codebitshift.')
        require(all(torch.equal(hs.scale_codes, x.scale_codes) for x in tensors.values()), 'Schaalwijziging in evaluatie.')
        counts.append({'layer': name, 'n_weights': hv.numel(), 'blocks': hs.scale_codes.numel(),
                       'padding_slots': hs.codes.numel() - hv.numel(),
                       'high_local_subnormal_count': int((((hc & 24) == 0) & ((hc & 7) != 0)).sum()),
                       'nonzero_dropped_suffix_count': int(((hc & 3) != 0).sum()),
                       'C_new_zero_vs_high': int(((vals['C'] == 0) & (hv != 0)).sum()),
                       'A_zero_count': int((vals['A'] == 0).sum()), 'B_zero_count': int((vals['B'] == 0).sum()),
                       'C_zero_count': int((vals['C'] == 0).sum()),
                       'scale_exp_min': int(hs.scale_codes.min()) - 127, 'scale_exp_max': int(hs.scale_codes.max()) - 127,
                       'high_code_and_scale_bytes': hs.scale_codes.numel() * 25,
                       'low_code_and_scale_bytes': hs.scale_codes.numel() * 17,
                       'sign_exponent_scale_preserved_C': True})
    return pair_rows, counts
