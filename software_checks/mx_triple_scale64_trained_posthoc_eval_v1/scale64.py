"""Frozen A/B/C routes from an archived E4M3 source, for the scale64-trained model.

A preserves source scale and exponent-MSB truncation. B changes only that scale
by exactly 2**6. C rounds normalized E4 elements/64 to E2M3 at that same changed
scale. Every 4-bit route then uses the unchanged mantissa shift of its 6-bit route.
No FP32 MX master, G(W), tensor encoder, learned scale or BN update is involved.
"""
from __future__ import annotations

import copy
import json
import math

import torch

import joint_adapter as adapter
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low
import posthoc_truncation as previous
import trunc_adapter as trained_truncation

require = high.require
tensor_exact = previous.tensor_exact
TRAININGS = ('scale64',)
SCALE_EXPONENT_OFFSET = 6
SCALE_FACTOR = 64
DERIVED_SCHEMA = 'mx-triple-scale64-trained-posthoc-derived-v1'
PATHS = ('source8', 'trained6', 'trained4', 'A6', 'A4', 'B6', 'B4', 'C6', 'C4')
DERIVED_PATHS = PATHS[3:]
PATH_SPEC = {
    'source8': dict(label='Opgeslagen E4M3-bron', source_bits=8, target_bits=8,
                    family='trained', direct_baseline='source8', bn_bank_key='8',
                    scale_exponent_offset=0, rounding='stored'),
    'trained6': dict(label='Opgeslagen getrainde E2M3-modus', source_bits=8, target_bits=6,
                     family='trained', direct_baseline='trained6', bn_bank_key='32',
                     scale_exponent_offset=6, rounding='stored'),
    'trained4': dict(label='Opgeslagen getrainde E2M1-modus', source_bits=8, target_bits=4,
                     family='trained', direct_baseline='trained4', bn_bank_key='4',
                     scale_exponent_offset=6, rounding='stored'),
}
for _family, _description in [('A', 'naieve exponenttruncatie, bronschaal'),
                               ('B', 'naieve exponenttruncatie, schaal x64'),
                               ('C', 'afronding vanuit E4M3, schaal x64')]:
    for _bit in (6, 4):
        PATH_SPEC[_family + str(_bit)] = dict(
            label=_family + ': ' + _description + ('; daarna mantissatruncatie' if _bit == 4 else ''),
            source_bits=8, target_bits=_bit, family=_family,
            direct_baseline='trained' + str(_bit), bn_bank_key=adapter.BANK[_bit],
            scale_exponent_offset=0 if _family == 'A' else 6,
            rounding='nearest_ties_even_E2M3' if _family == 'C' else 'none')
del _family, _description, _bit

COMPARISONS = tuple((f'trained{bit}', f'{family}{bit}')
                    for bit in (6, 4) for family in ('A', 'B', 'C')) + (
    ('A6', 'B6'), ('A4', 'B4'), ('B6', 'C6'), ('B4', 'C4'), ('A6', 'C6'), ('A4', 'C4'))
WEIGHT_COMPARISONS = tuple(('source8', key) for key in DERIVED_PATHS) + COMPARISONS


def _kind(training_kind: str) -> None:
    require(training_kind in TRAININGS and type(training_kind) is str, 'Onbekende bronstraining.')


def _digest(value) -> None:
    require(type(value) is str and len(value) == 64
            and set(value) <= set('0123456789abcdef'), 'Ongeldige bron-SHA256.')


def _validate_source8(payload: dict, training_kind: str) -> None:
    _kind(training_kind)
    trained_truncation.validate_source(payload)
    require(previous.plain_metadata(payload['provenance']), 'Verborgen of niet-eindige provenance.')


def check_source_set(payloads_by_intbit: dict, training_kind: str,
                     source_hashes: dict | None = None) -> dict:
    """Validate the B-trained exports and exact B derivation from their saved E4 source."""
    _kind(training_kind)
    require(set(payloads_by_intbit) == {8, 6, 4}, 'Alle drie opgeslagen modi zijn vereist.')
    _validate_source8(payloads_by_intbit[8], training_kind)
    if source_hashes is not None:
        require(set(source_hashes) == {8, 6, 4}, 'Onvolledige bronhashes.')
        for value in source_hashes.values(): _digest(value)
    for bit in (6, 4): trained_truncation.validate_derived(payloads_by_intbit[bit])
    first = payloads_by_intbit[8]
    blocks, delta_counts = 0, {}
    for bit in (6, 4):
        other = payloads_by_intbit[bit]
        require(previous.plain_metadata(other['provenance'])
                and first['provenance'] == other['provenance'], 'Exports hebben andere checkpointprovenance.')
        left, right = first['auxiliary_fp32_state'], other['auxiliary_fp32_state']
        require(set(left) == set(right) and all(tensor_exact(left[k], right[k]) for k in left),
                'Exports hebben verschillende auxiliaire/BN-staat.')
        origin = other['derivation']['source_sha256']
        if source_hashes is not None:
            require(origin == source_hashes[8], 'Afgeleide trainingsmodus verwijst naar andere E4-bronhash.')
        else:
            require(origin == payloads_by_intbit[6]['derivation']['source_sha256'],
                    'Afgeleide trainingsmodi hebben verschillende bronhashes.')
    for name in adapter.TARGET_LAYERS:
        mx = {bit: adapter.codec(bit).MXTensor.from_payload(payloads_by_intbit[bit]['mx_layers'][name])
              for bit in (8, 6, 4)}
        require(len({value.shape for value in mx.values()}) == 1, 'Bronvormen/blokgroepen verschillen.')
        for bit in (6, 4):
            expected = transform_codes(mx[8], 'B' + str(bit))
            require(tensor_exact(expected.codes, mx[bit].codes)
                    and tensor_exact(expected.scale_codes, mx[bit].scale_codes),
                    'Opgeslagen getrainde modus is geen exacte B-afleiding van haar E4-bron.')
            require(tensor_exact(expected.decode(), mx[bit].decode()),
                    'Opgeslagen B-inferentiegewichten verschillen van de bronafleiding.')
        delta = mx[8].scale_codes.int() - mx[6].scale_codes.int()
        for value, count in zip(*torch.unique(delta, return_counts=True)):
            key = str(int(value)); delta_counts[key] = delta_counts.get(key, 0) + int(count)
        blocks += mx[8].scale_codes.numel()
    require(delta_counts == {'-6': blocks}, 'Opgeslagen B-schalen zijn niet overal E4-schaal plus6.')
    return {'training_kind': training_kind, 'blocks': blocks,
            'same_checkpoint_provenance': True, 'same_auxiliary_state': True,
            'source_hashes_checked': source_hashes is not None,
            'E4_minus_trained_E2_scale_exponent_counts': delta_counts,
            'trained_lower_codes_exact_B_routes': True,
            'trained_lower_scale_codes_exact_B_routes': True}


def shifted_scale_codes(source: high.MXTensor) -> torch.Tensor:
    """Exact finite E8M0 offset, promoted before addition; never clip or wrap."""
    source.validate()
    shifted = source.scale_codes.to(torch.int32) + SCALE_EXPONENT_OFFSET
    require(bool((shifted <= 254).all()),
            'Schaal x64 valt buiten eindig E8M0-bereik; geen clipping of uint8-wrap toegestaan.')
    return shifted.to(torch.uint8)


def transform_codes(source: high.MXTensor, path_id: str):
    require(isinstance(source, high.MXTensor), 'Alle A/B/C-routes verwachten een E4M3-bron.')
    require(path_id in DERIVED_PATHS, 'Onbekende A/B/C-route.')
    source.validate()
    family, bit = path_id[0], PATH_SPEC[path_id]['target_bits']
    if family == 'A':
        middle_result = previous.exponent_truncate(source)
    elif family == 'B':
        naive = previous.exponent_truncate(source)
        middle_result = middle.MXTensor(naive.codes.clone(), shifted_scale_codes(source), source.shape)
    else:
        # Normalize analytically; do not decode FP32 global weights and do not
        # call encode_tensor, which would choose a new scale for each block.
        normalized = high.decode_elements(source.codes) / SCALE_FACTOR
        codes = middle.encode_elements(normalized)
        middle_result = middle.MXTensor(codes, shifted_scale_codes(source), source.shape)
    middle_result.validate()
    result = middle_result if bit == 6 else previous.mantissa_truncate(middle_result)
    result.validate()
    expected_scales = source.scale_codes if family == 'A' else shifted_scale_codes(source)
    require(tensor_exact(result.scale_codes, expected_scales), 'Schaalbeleid wijkt af.')
    source_sign = source.codes.to(torch.int64) >> 7
    require(torch.equal(source_sign, result.codes.to(torch.int64) >> (bit - 1)), 'Tekenbit gewijzigd.')
    return result


def _derivation_metadata(path_id, source_digest, training_kind):
    _kind(training_kind); _digest(source_digest)
    require(path_id in DERIVED_PATHS, 'Geen A/B/C-afleiding.')
    spec = PATH_SPEC[path_id]
    return {'path_id': path_id, 'training_kind': training_kind, 'family': spec['family'],
            'source_bits': 8, 'target_bits': spec['target_bits'],
            'source_file_sha256': source_digest,
            'scale_exponent_offset': spec['scale_exponent_offset'],
            'scale_factor': 1 if spec['family'] == 'A' else 64,
            'scale_policy': 'copy_source_E8M0_bytes' if spec['family'] == 'A'
                            else 'source_E8M0_exponent_plus6_no_clip_no_wrap',
            'exponent_mapping': 'highest_two_exponent_bits' if spec['family'] in ('A', 'B')
                                else 'RNE_E2M3_of_decoded_E4_normalized_div64',
            'rounding': spec['rounding'], 'dropped_exponent_bits': 2 if spec['family'] in ('A', 'B') else 0,
            'dropped_mantissa_bits': 2 if spec['target_bits'] == 4 else 0,
            'mantissa_policy': 'derived_E2M3_codes_shift_right2' if spec['target_bits'] == 4 else 'not_reduced',
            'requires_quantized_layer_FP32_masters': False, 'source_overwritten': False,
            'tensor_scale_reselection': False, 'learned_scale': False, 'batchnorm_recalibrated': False}


def derive_payload(source8: dict, path_id: str, source_digest: str, training_kind: str) -> dict:
    _validate_source8(source8, training_kind)
    meta = _derivation_metadata(path_id, source_digest, training_kind)
    bit = PATH_SPEC[path_id]['target_bits']
    layers = {}
    for name, value in source8['mx_layers'].items():
        converted = transform_codes(high.MXTensor.from_payload(value), path_id)
        converted.decode()  # Require actual inference weights to be finite FP32.
        layers[name] = converted.payload()
    result = {'schema': DERIVED_SCHEMA, 'model': 'svhnq', 'expand': 2,
              'weight_bits': bit, 'weight_format': adapter.FORMATS[bit],
              'activation_bits': 32, 'bn_bank_key': adapter.BANK[bit],
              'weight_preprocess': 'already_in_source; DO_NOT_reapply_G',
              'truncation': path_id != 'C6', 'mx_layers': layers,
              'auxiliary_fp32_state': copy.deepcopy(source8['auxiliary_fp32_state']),
              'provenance': copy.deepcopy(source8['provenance']), 'derivation': meta}
    validate_derived(result)
    return result


def validate_derived(payload: dict) -> None:
    require(type(payload) is dict and set(payload) == previous.BASE_KEYS | {'derivation'},
            'Onverwachte velden in schaal64-afleiding.')
    require(payload['schema'] == DERIVED_SCHEMA
            and payload['weight_preprocess'] == 'already_in_source; DO_NOT_reapply_G',
            'Onbekende afleiding of dubbele voorbewerking.')
    d = payload['derivation']
    require(type(d) is dict and previous.plain_metadata(d), 'Verborgen of ongeldige afleidingsmetadata.')
    require(d == _derivation_metadata(d.get('path_id'), d.get('source_file_sha256'), d.get('training_kind')),
            'Afleidingsbeleid wijkt af.')
    path = d['path_id']; bit = PATH_SPEC[path]['target_bits']
    require(payload['truncation'] is (path != 'C6'), 'Onjuist truncatielabel.')
    previous._validate_common(payload, bit)


def _load_frozen(models, payload, device):
    bit = payload['weight_bits']
    model = adapter.adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected = set(model.state_dict()) - {n + '.weight' for n in adapter.TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected, 'Ontbrekende staat of verborgen FP32-MX-master.')
    for name in adapter.TARGET_LAYERS:
        layer = model.get_submodule(name)
        mx = adapter.codec(bit).MXTensor.from_payload(payload['mx_layers'][name], device)
        require(mx.shape == tuple(layer.weight.shape), 'Onjuiste gewichtsvorm.')
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, adapter.original_adapter.FrozenMXConv(layer, mx))
    missing, extra = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in adapter.TARGET_LAYERS} and not extra,
            'Onvolledige reconstructie van opgeslagen staat.')
    adapter.select_mode(model, bit)
    model = model.to(device).eval().requires_grad_(False)
    require(not {n + '.weight' for n in adapter.TARGET_LAYERS}.intersection(dict(model.named_parameters())),
            'Een MX-laag bevat een FP32-master.')
    return model


def load_derived_model(models, payload, device):
    validate_derived(payload)
    return _load_frozen(models, payload, device)


def load_trained_model(models, payload, device, training_kind):
    _kind(training_kind)
    bit = payload['weight_bits']
    if bit == 8: trained_truncation.validate_source(payload)
    else: trained_truncation.validate_derived(payload)
    require(previous.plain_metadata(payload['provenance']), 'Ongeldige opgeslagen provenance.')
    return _load_frozen(models, payload, device)


def weight_diagnostics(payloads: dict, derived_by_path: dict, device) -> tuple[list, list]:
    require(set(derived_by_path) == set(DERIVED_PATHS), 'A/B/C-afleidingen ontbreken.')
    training_kind = derived_by_path['A6']['derivation']['training_kind']
    check_source_set(payloads, training_kind)
    for payload in derived_by_path.values(): validate_derived(payload)
    all_payloads = {'source8': payloads[8], 'trained6': payloads[6], 'trained4': payloads[4], **derived_by_path}
    pairs, counts = [], []
    for name in adapter.TARGET_LAYERS:
        mx = {path: adapter.codec(PATH_SPEC[path]['target_bits']).MXTensor.from_payload(all_payloads[path]['mx_layers'][name], device)
              for path in PATHS}
        shape = mx['source8'].shape; cols = math.prod(shape[1:])
        codes = {path: v.codes.reshape(shape[0], -1)[:, :cols] for path, v in mx.items()}
        values = {path: v.decode().to(torch.float64) for path, v in mx.items()}
        for path in DERIVED_PATHS:
            expected = transform_codes(mx['source8'], path)
            require(tensor_exact(expected.codes, mx[path].codes)
                    and tensor_exact(expected.scale_codes, mx[path].scale_codes),
                    'Opgeslagen A/B/C-codes of schalen verschillen van de bronafleiding: ' + path)
        for refpath, otherpath in WEIGHT_COMPARISONS:
            error = values[otherpath] - values[refpath]
            ref, other = mx[refpath], mx[otherpath]
            same_format = PATH_SPEC[refpath]['target_bits'] == PATH_SPEC[otherpath]['target_bits']
            pairs.append({'layer': name, 'reference_path': refpath, 'other_path': otherpath,
                'comparison': otherpath + '_minus_' + refpath, 'n_weights': error.numel(),
                'code_comparison_same_format': same_format,
                'code_differences': int((codes[otherpath] != codes[refpath]).sum()) if same_format else None,
                'scale_code_differences': int((other.scale_codes != ref.scale_codes).sum()),
                'scale_bytes_equal': bool(torch.equal(other.scale_codes, ref.scale_codes)),
                'numeric_differences': int((values[otherpath] != values[refpath]).sum()),
                'mean_signed_difference': error.mean().item(), 'mae': error.abs().mean().item(),
                'rmse': error.square().mean().sqrt().item(), 'max_abs_difference': error.abs().max().item(),
                'reference_l2_norm': values[refpath].norm().item(), 'difference_l2_norm': error.norm().item()})
        for path in PATHS:
            spec = PATH_SPEC[path]; bit = spec['target_bits']; target = mx[path]
            # Every route in this B-trained experiment descends from the same
            # saved E4 source. Stored lower modes also have source scale +6.
            sourcepath = 'source8'
            baseline = spec['direct_baseline']; val = values[path]; base = values[baseline]; src = values[sourcepath]
            mean_abs, rms = val.abs().mean().item(), val.square().mean().sqrt().item()
            base_abs, base_rms = base.abs().mean().item(), base.square().mean().sqrt().item()
            src_rms = src.square().mean().sqrt().item()
            def distribution(delta):
                return json.dumps({str(int(k)): int(v) for k, v in zip(*torch.unique(delta, return_counts=True))}, sort_keys=True)
            rawcodes = codes['source8']; exp_suffix = None; mant_suffix = None
            if path[:1] in ('A', 'B'): exp_suffix = int((((rawcodes.long() >> 3) & 3) != 0).sum())
            if path in ('A4', 'B4', 'C4'): mant_suffix = int(((codes[path[0] + '6'].long() & 3) != 0).sum())
            expmask, mantmask = {8: (120, 7), 6: (24, 7), 4: (6, 1)}[bit]
            outside = int((high.decode_elements(rawcodes).abs() / 64 > 7.5).sum()) if path == 'C6' else None
            counts.append({'layer': name, 'path': path, 'family': spec['family'],
                'source_bits': spec['source_bits'], 'target_bits': bit, 'bn_bank_key': spec['bn_bank_key'],
                'n_weights': val.numel(), 'blocks': target.scale_codes.numel(),
                'padding_slots': target.codes.numel() - val.numel(), 'direct_baseline': baseline,
                'mean_abs': mean_abs, 'rms': rms,
                'mean_abs_ratio_to_direct': mean_abs / base_abs if base_abs else (1. if mean_abs == 0 else None),
                'rms_ratio_to_direct': rms / base_rms if base_rms else (1. if rms == 0 else None),
                'rms_ratio_to_source': rms / src_rms if src_rms else (1. if rms == 0 else None),
                'zero_count': int((val == 0).sum()),
                'negative_zero_count': int(((val == 0) & torch.signbit(val)).sum()),
                'new_zero_vs_source': int(((val == 0) & (src != 0)).sum()),
                'local_subnormal_count': int((((codes[path] & expmask) == 0) & ((codes[path] & mantmask) != 0)).sum()),
                'scale_exp_min': int(target.scale_codes.min()) - 127,
                'scale_exp_max': int(target.scale_codes.max()) - 127,
                'scale_exponent_offset': spec['scale_exponent_offset'], 'scale_anchor_path': sourcepath,
                'scales_equal_source': bool(torch.equal(target.scale_codes, mx[sourcepath].scale_codes)),
                'source_scale_delta_counts': distribution(target.scale_codes.int() - mx[sourcepath].scale_codes.int()),
                'scale_delta_to_direct_counts': distribution(target.scale_codes.int() - mx[baseline].scale_codes.int()),
                'nonzero_dropped_exponent_suffix_count': exp_suffix,
                'nonzero_dropped_mantissa_suffix_count': mant_suffix,
                'rounded_source_out_of_target_element_range': outside,
                'code_and_scale_bytes': target.scale_codes.numel() * {8: 33, 6: 25, 4: 17}[bit]})
    return pairs, counts
