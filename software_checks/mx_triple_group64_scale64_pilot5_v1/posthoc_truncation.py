"""Frozen, code-derived MX variants for the same triple-direct checkpoints.

No FP32 quantized-layer master is read. Bit truncation never calls an encoder,
applies G(W), changes a scale, corrects the exponent bias, or trains anything.
Rounded controls decode normalized elements and keep source E8M0 bytes too.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any

import torch
from group_config import packed_group_bytes
from torch import Tensor

import joint_adapter as adapter
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low

require = high.require
DIRECT_SCHEMA = 'mx-triple-direct-single-mode-export-v1'
DERIVED_SCHEMA = 'mx-triple-posthoc-derived-v1'
PATHS = ('direct8', 'direct6', 'direct4', 'round64', 'mantissa64',
         'round86', 'exponent86', 'combined864')
PATH_SPEC = {
    'direct8': dict(label='Direct E4M3', source_bits=8, target_bits=8, method='direct',
                    family='direct', direct_baseline='direct8', bn_bank_key='8'),
    'direct6': dict(label='Direct E2M3', source_bits=6, target_bits=6, method='direct',
                    family='direct', direct_baseline='direct6', bn_bank_key='32'),
    'direct4': dict(label='Direct E2M1', source_bits=4, target_bits=4, method='direct',
                    family='direct', direct_baseline='direct4', bn_bank_key='4'),
    'round64': dict(label='E2M3 naar E2M1: RNE, bronschaal', source_bits=6, target_bits=4,
                    method='rounded_from_source', family='mantissa', direct_baseline='direct4', bn_bank_key='4'),
    'mantissa64': dict(label='E2M3 naar E2M1: mantissatruncatie', source_bits=6, target_bits=4,
                    method='mantissa_truncation', family='mantissa', direct_baseline='direct4', bn_bank_key='4'),
    'round86': dict(label='E4M3 naar E2M3: RNE, bronschaal', source_bits=8, target_bits=6,
                    method='rounded_from_source', family='exponent', direct_baseline='direct6', bn_bank_key='32'),
    'exponent86': dict(label='E4M3 naar E2M3: hoogste exponentbits', source_bits=8, target_bits=6,
                    method='exponent_msb_truncation', family='exponent', direct_baseline='direct6', bn_bank_key='32'),
    'combined864': dict(label='E4M3 naar E2M1: exponent + mantissa', source_bits=8, target_bits=4,
                    method='exponent_msb_then_mantissa_truncation', family='combined',
                    direct_baseline='direct4', bn_bank_key='4'),
}
DERIVED_PATHS = PATHS[3:]
WEIGHT_COMPARISONS = (('direct4', 'round64'), ('direct4', 'mantissa64'),
                      ('round64', 'mantissa64'), ('direct6', 'round86'),
                      ('direct6', 'exponent86'), ('round86', 'exponent86'),
                      ('direct4', 'combined864'), ('mantissa64', 'combined864'))
BASE_KEYS = {'schema', 'model', 'expand', 'weight_bits', 'weight_format',
             'activation_bits', 'bn_bank_key', 'weight_preprocess', 'truncation',
             'mx_layers', 'auxiliary_fp32_state', 'provenance'}


def plain_metadata(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, tuple):
        return all(plain_metadata(v) for v in value)
    return type(value) is dict and all(type(k) is str and plain_metadata(v) for k, v in value.items())


def tensor_exact(a: Tensor, b: Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
            and (not a.is_floating_point() or torch.equal(torch.signbit(a).cpu(), torch.signbit(b).cpu())))


def _validate_common(payload: dict, bit: int) -> None:
    require(type(bit) is int and bit in adapter.MODES and type(payload['weight_bits']) is int
            and payload['weight_bits'] == bit, 'Verkeerde gewichtsmodus.')
    require(payload['model'] == 'svhnq' and payload['expand'] == 2 and payload['activation_bits'] == 32,
            'Afwijkende architectuur of activatieprecisie.')
    require(payload['weight_format'] == adapter.FORMATS[bit] and payload['bn_bank_key'] == adapter.BANK[bit],
            'Formaat en getrainde doel-BN-bank passen niet bij de modus.')
    require(type(payload['mx_layers']) is dict and tuple(payload['mx_layers']) == adapter.TARGET_LAYERS,
            'Onvolledige of afwijkende MX-laagselectie.')
    require(plain_metadata(payload['provenance']), 'Provenance bevat verborgen tensorstaat.')
    require(type(payload['auxiliary_fp32_state']) is dict, 'Ongeldige auxiliaire staat.')
    excluded = {name + '.weight' for name in adapter.TARGET_LAYERS}
    for name, value in payload['auxiliary_fp32_state'].items():
        require(type(name) is str and name not in excluded, 'Verborgen FP32-master in export.')
        require(isinstance(value, Tensor) and value.dtype in (torch.float32, torch.int64),
                'Onverwachte auxiliaire tensor: ' + str(name))
        require(bool(torch.isfinite(value).all()), 'Niet-eindige auxiliaire tensor: ' + name)
    for value in payload['mx_layers'].values():
        adapter.codec(bit).MXTensor.from_payload(value)


def validate_direct_payload(payload: dict, bit: int) -> None:
    require(type(payload) is dict and set(payload) == BASE_KEYS, 'Onverwachte velden in directe export.')
    require(payload['schema'] == DIRECT_SCHEMA and payload['truncation'] is False,
            'Verwacht een directe export uit de gezamenlijke drieprecisietraining.')
    require(payload['weight_preprocess'] == 'original_G(W)', 'Afwijkende gewichtsvoorbewerking.')
    _validate_common(payload, bit)


def mantissa_truncate(source: middle.MXTensor) -> low.MXTensor:
    """[s ee m2 m1 m0] -> [s ee m2]; logical shift of unsigned codewords."""
    require(isinstance(source, middle.MXTensor), 'Mantissatruncatie verwacht een E2M3-bron.')
    source.validate()
    codes = (source.codes.to(torch.int64) >> 2).to(torch.uint8)
    result = low.MXTensor(codes, source.scale_codes.clone(), source.shape)
    result.validate()
    require(torch.equal(source.codes.long() >> 5, codes.long() >> 3), 'Tekenbit gewijzigd.')
    require(torch.equal((source.codes.long() >> 3) & 3, (codes.long() >> 1) & 3), 'Exponent gewijzigd.')
    require(torch.equal(source.scale_codes, result.scale_codes), 'Bronschaal gewijzigd.')
    return result


def exponent_truncate(source: high.MXTensor) -> middle.MXTensor:
    """[s e3 e2 e1 e0 m2 m1 m0] -> [s e3 e2 m2 m1 m0].

    The two least-significant exponent-field bits are removed. The retained
    two-bit field is decoded as E2M3 (bias1); no bias or shared-scale correction.
    """
    require(isinstance(source, high.MXTensor), 'Exponenttruncatie verwacht een E4M3-bron.')
    source.validate()
    c = source.codes.to(torch.int64)
    sign, exponent, mantissa = c >> 7, (c >> 3) & 15, c & 7
    codes = ((sign << 5) | ((exponent >> 2) << 3) | mantissa).to(torch.uint8)
    result = middle.MXTensor(codes, source.scale_codes.clone(), source.shape)
    result.validate()
    require(torch.equal(c >> 7, codes.long() >> 5), 'Tekenbit gewijzigd.')
    require(torch.equal(c & 7, codes.long() & 7), 'Mantissa gewijzigd bij exponenttruncatie.')
    require(torch.equal((c >> 5) & 3, (codes.long() >> 3) & 3), 'Verkeerde exponentbits behouden.')
    require(torch.equal(source.scale_codes, result.scale_codes), 'Bronschaal gewijzigd.')
    return result


def round_source_codes(source, source_bits: int, target_bits: int):
    """Nearest-even element rounding with the ORIGINAL scale, including saturation.

    Calling encode_tensor(source.decode()) would reselect scales and is forbidden.
    The round86 control can saturate E4M3 elements above E2M3's local range.
    """
    require((source_bits, target_bits) in ((6, 4), (8, 6)), 'Onbekende afrondingsroute.')
    source_codec, target_codec = adapter.codec(source_bits), adapter.codec(target_bits)
    require(isinstance(source, source_codec.MXTensor), 'Onjuist bronformaat voor afrondingscontrole.')
    source.validate()
    codes = target_codec.encode_elements(source_codec.decode_elements(source.codes))
    result = target_codec.MXTensor(codes, source.scale_codes.clone(), source.shape)
    result.validate()
    require(torch.equal(source.scale_codes, result.scale_codes), 'Bronschaal gewijzigd.')
    return result


def transform_codes(source, path_id: str):
    require(path_id in DERIVED_PATHS, 'Geen afgeleide route: ' + str(path_id))
    if path_id == 'mantissa64':
        return mantissa_truncate(source)
    if path_id == 'exponent86':
        return exponent_truncate(source)
    if path_id == 'combined864':
        return mantissa_truncate(exponent_truncate(source))
    spec = PATH_SPEC[path_id]
    return round_source_codes(source, spec['source_bits'], spec['target_bits'])


def _derivation_metadata(path_id: str, source_sha256: str) -> dict:
    spec = PATH_SPEC[path_id]
    return {'path_id': path_id, 'method': spec['method'], 'source_bits': spec['source_bits'],
            'target_bits': spec['target_bits'], 'source_format': adapter.FORMATS[spec['source_bits']],
            'source_file_sha256': source_sha256,
            'dropped_exponent_bits': 2 if path_id in ('exponent86', 'combined864') else 0,
            'dropped_mantissa_bits': 2 if path_id in ('mantissa64', 'combined864') else 0,
            'rounding': 'nearest_ties_even' if path_id in ('round64', 'round86') else 'none',
            'scale_policy': 'copy_source_E8M0_bytes_unchanged', 'bias_compensation': False,
            'requires_quantized_layer_FP32_masters': False, 'source_overwritten': False}


def derive_payload(source: dict, path_id: str, source_digest: str) -> dict:
    require(path_id in DERIVED_PATHS, 'Onbekende afleidingsroute.')
    require(type(source_digest) is str and len(source_digest) == 64
            and all(c in '0123456789abcdef' for c in source_digest), 'Ongeldige bron-SHA256.')
    spec = PATH_SPEC[path_id]
    validate_direct_payload(source, spec['source_bits'])
    source_codec = adapter.codec(spec['source_bits'])
    layers = {name: transform_codes(source_codec.MXTensor.from_payload(value), path_id).payload()
              for name, value in source['mx_layers'].items()}
    result = {'schema': DERIVED_SCHEMA, 'model': 'svhnq', 'expand': 2,
              'weight_bits': spec['target_bits'], 'weight_format': adapter.FORMATS[spec['target_bits']],
              'activation_bits': 32, 'bn_bank_key': spec['bn_bank_key'],
              'weight_preprocess': 'already_in_source; DO_NOT_reapply_G',
              'truncation': path_id not in ('round64', 'round86'),
              'mx_layers': layers, 'auxiliary_fp32_state': copy.deepcopy(source['auxiliary_fp32_state']),
              'provenance': copy.deepcopy(source['provenance']),
              'derivation': _derivation_metadata(path_id, source_digest)}
    validate_derived(result)
    return result


def validate_derived(payload: dict) -> None:
    require(type(payload) is dict and set(payload) == BASE_KEYS | {'derivation'},
            'Onverwachte velden in afgeleide export.')
    require(payload['schema'] == DERIVED_SCHEMA, 'Geen afgeleide post-hoc-export.')
    require(payload['weight_preprocess'] == 'already_in_source; DO_NOT_reapply_G', 'Dubbele voorbewerking.')
    d = payload['derivation']
    require(type(d) is dict and plain_metadata(d), 'Onjuiste of verborgen afleidingsmetadata.')
    path_id = d.get('path_id')
    require(path_id in DERIVED_PATHS, 'Onbekende afleidingsroute.')
    digest = d.get('source_file_sha256')
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
            'Ongeldige bronhash.')
    require(d == _derivation_metadata(path_id, digest), 'Afleidingsbeleid wijkt af.')
    spec = PATH_SPEC[path_id]
    require(payload['truncation'] is (path_id not in ('round64', 'round86')), 'Onjuist truncatielabel.')
    _validate_common(payload, spec['target_bits'])


def load_derived_model(models, payload: dict, device):
    """Reconstruct directly from stored codes; freeze every tensor and target BN."""
    validate_derived(payload)
    bit = payload['weight_bits']
    model = adapter.adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected_aux = set(model.state_dict()) - {n + '.weight' for n in adapter.TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected_aux, 'Ontbrekende auxiliaire staat of verborgen master.')
    for name in adapter.TARGET_LAYERS:
        layer = model.get_submodule(name)
        encoded = adapter.codec(bit).MXTensor.from_payload(payload['mx_layers'][name], device)
        require(encoded.shape == tuple(layer.weight.shape), 'Verkeerde gewichtsvorm.')
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, adapter.original_adapter.FrozenMXConv(layer, encoded))
    missing, extra = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in adapter.TARGET_LAYERS} and not extra,
            'Onvolledige modelreconstructie.')
    adapter.select_mode(model, bit)
    model = model.to(device).eval().requires_grad_(False)
    require(not {n + '.weight' for n in adapter.TARGET_LAYERS}.intersection(dict(model.named_parameters())),
            'Afgeleide uitvoering bevat zes FP32-masters.')
    return model


def check_source_set(payloads_by_intbit: dict) -> dict:
    require(set(payloads_by_intbit) == {4, 6, 8}, 'Alle drie directe bronexports zijn vereist.')
    for bit, payload in payloads_by_intbit.items():
        validate_direct_payload(payload, bit)
    first = payloads_by_intbit[8]
    for bit in (6, 4):
        other = payloads_by_intbit[bit]
        require(first['provenance'] == other['provenance'], 'Exports horen niet bij hetzelfde checkpoint.')
        left, right = first['auxiliary_fp32_state'], other['auxiliary_fp32_state']
        require(set(left) == set(right) and all(tensor_exact(left[k], right[k]) for k in left),
                'Directe paden hebben niet dezelfde auxiliaire modelstaat.')
    blocks, delta_counts = 0, {}
    for name in adapter.TARGET_LAYERS:
        encoded = {bit: adapter.codec(bit).MXTensor.from_payload(payload['mx_layers'][name])
                   for bit, payload in payloads_by_intbit.items()}
        require(len({value.shape for value in encoded.values()}) == 1, 'Gewichtsvormen/groepen verschillen.')
        require(torch.equal(encoded[6].scale_codes, encoded[4].scale_codes),
                'Directe E2M3/E2M1-schalen verschillen ondanks dezelfde schaalregel.')
        # E4M3 has a different direct scale rule. Record its observed delta;
        # neither equality nor a constant -6 is imposed at zero/clamped blocks.
        delta = encoded[8].scale_codes.int() - encoded[6].scale_codes.int()
        for value, count in zip(*torch.unique(delta, return_counts=True)):
            key = str(int(value)); delta_counts[key] = delta_counts.get(key, 0) + int(count)
        blocks += encoded[8].scale_codes.numel()
    return {'same_checkpoint_provenance': True, 'same_auxiliary_state': True,
            'all_E2_source_scales_equal': True, 'E4_scale_equality_not_required': True,
            'blocks': blocks, 'E4_minus_E2_scale_exponent_counts': delta_counts}


def weight_diagnostics(direct_payloads: dict, derived_by_path: dict, device) -> tuple[list[dict], list[dict]]:
    """Same-layer numerical gaps and scale/bit diagnostics, excluding all padding.

    There is no error against the absent FP32 master. RNE86 saturation is counted
    against normalized SOURCE values with its unchanged source scale.
    """
    check_source_set(direct_payloads)
    require(set(derived_by_path) == set(DERIVED_PATHS), 'Afgeleide routes ontbreken.')
    for payload in derived_by_path.values():
        validate_derived(payload)
    payloads = {f'direct{bit}': value for bit, value in direct_payloads.items()}
    payloads.update(derived_by_path)
    pair_rows, counts = [], []
    for name in adapter.TARGET_LAYERS:
        encoded = {path: adapter.codec(PATH_SPEC[path]['target_bits']).MXTensor.from_payload(payloads[path]['mx_layers'][name], device)
                   for path in PATHS}
        shape = encoded['direct8'].shape
        cols = math.prod(shape[1:])
        codes = {path: value.codes.reshape(shape[0], -1)[:, :cols] for path, value in encoded.items()}
        values = {path: value.decode().to(torch.float64) for path, value in encoded.items()}
        for path in DERIVED_PATHS:
            spec = PATH_SPEC[path]; source_path = f"direct{spec['source_bits']}"
            expected = transform_codes(encoded[source_path], path)
            require(tensor_exact(expected.codes, encoded[path].codes)
                    and tensor_exact(expected.scale_codes, encoded[path].scale_codes),
                    'Afgeleide codes of bronschalen verschillen: ' + path)
            require(encoded[path].shape == encoded[source_path].shape, 'Afgeleide vorm verschilt.')
        for reference_path, other_path in WEIGHT_COMPARISONS:
            error = values[other_path] - values[reference_path]
            ref, other = encoded[reference_path], encoded[other_path]
            pair_rows.append({'layer': name, 'reference_path': reference_path, 'other_path': other_path,
                              'comparison': other_path + '_minus_' + reference_path,
                              'n_weights': error.numel(),
                              'code_differences': int((codes[other_path] != codes[reference_path]).sum()),
                              'scale_code_differences': int((other.scale_codes != ref.scale_codes).sum()),
                              'scale_bytes_equal': bool(torch.equal(other.scale_codes, ref.scale_codes)),
                              'numeric_differences': int((values[other_path] != values[reference_path]).sum()),
                              'mean_signed_difference': error.mean().item(), 'mae': error.abs().mean().item(),
                              'rmse': error.square().mean().sqrt().item(), 'max_abs_difference': error.abs().max().item(),
                              'reference_l2_norm': values[reference_path].norm().item(),
                              'difference_l2_norm': error.norm().item()})
        for path in PATHS:
            spec = PATH_SPEC[path]; bit = spec['target_bits']; source_path = f"direct{spec['source_bits']}"
            target, source, direct = encoded[path], encoded[source_path], encoded[spec['direct_baseline']]
            val, baseline = values[path], values[spec['direct_baseline']]
            mean_abs, rms = val.abs().mean().item(), val.square().mean().sqrt().item()
            baseline_abs, baseline_rms = baseline.abs().mean().item(), baseline.square().mean().sqrt().item()
            delta = target.scale_codes.int() - direct.scale_codes.int()
            delta_counts = {str(int(k)): int(v) for k, v in zip(*torch.unique(delta, return_counts=True))}
            source_codes = codes[source_path]
            exponent_suffix = int((((source_codes.long() >> 3) & 3) != 0).sum()) if path in ('exponent86', 'combined864') else None
            mantissa_suffix = int(((source_codes.long() & 3) != 0).sum()) if path in ('mantissa64', 'combined864') else None
            expmask, mantmask = {8: (120, 7), 6: (24, 7), 4: (6, 1)}[bit]
            source_normalized = adapter.codec(spec['source_bits']).decode_elements(source_codes)
            outside = int((source_normalized.abs() > (7.5 if bit == 6 else 6.0)).sum()) if path in ('round64', 'round86') else None
            counts.append({'layer': name, 'path': path, 'family': spec['family'],
                           'source_bits': spec['source_bits'], 'target_bits': bit,
                           'bn_bank_key': spec['bn_bank_key'], 'n_weights': val.numel(),
                           'blocks': target.scale_codes.numel(), 'padding_slots': target.codes.numel() - val.numel(),
                           'direct_baseline': spec['direct_baseline'], 'mean_abs': mean_abs, 'rms': rms,
                           'mean_abs_ratio_to_direct': mean_abs / baseline_abs if baseline_abs else (1.0 if mean_abs == 0 else None),
                           'rms_ratio_to_direct': rms / baseline_rms if baseline_rms else (1.0 if rms == 0 else None),
                           'zero_count': int((val == 0).sum()),
                           'new_zero_vs_source': int(((val == 0) & (values[source_path] != 0)).sum()),
                           'local_subnormal_count': int((((codes[path] & expmask) == 0) & ((codes[path] & mantmask) != 0)).sum()),
                           'scale_exp_min': int(target.scale_codes.min()) - 127,
                           'scale_exp_max': int(target.scale_codes.max()) - 127,
                           'scale_anchor_path': source_path, 'scales_equal_source': bool(torch.equal(target.scale_codes, source.scale_codes)),
                           'scale_delta_to_direct_counts': json.dumps(delta_counts, sort_keys=True),
                           'nonzero_dropped_exponent_suffix_count': exponent_suffix,
                           'nonzero_dropped_mantissa_suffix_count': mantissa_suffix,
                           'rounded_source_out_of_target_element_range': outside,
                           'code_and_scale_bytes': target.scale_codes.numel() * packed_group_bytes(bit)})
    return pair_rows, counts
