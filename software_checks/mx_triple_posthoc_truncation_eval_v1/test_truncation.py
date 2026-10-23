"""Bounded code and frozen-model checks; no real data, training or calibration."""
from __future__ import annotations

import copy
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import torch

import joint_adapter as adapter
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low
import resume_core as core
import svhn_support as s
import truncation as tr


def must_fail(call, label):
    try:
        call()
    except (ValueError, RuntimeError, AssertionError, KeyError, TypeError):
        return label
    raise AssertionError('Negatieve test werd niet geweigerd: ' + label)


def oracle_e2(code: int, mantissa_bits: int) -> Fraction:
    width = mantissa_bits + 2
    negative = code >= 2 ** width
    exponent = (code // 2 ** mantissa_bits) % 4
    fraction = Fraction(code % 2 ** mantissa_bits, 2 ** mantissa_bits)
    magnitude = fraction if exponent == 0 else (1 + fraction) * Fraction(2) ** (exponent - 1)
    return -magnitude if negative else magnitude


def oracle_e4(code: int) -> Fraction:
    if code in (127, 255):
        raise ValueError('NaN')
    exponent, fraction = (code // 8) % 16, Fraction(code % 8, 8)
    magnitude = fraction / 64 if exponent == 0 else (1 + fraction) * Fraction(2) ** (exponent - 7)
    return -magnitude if code >= 128 else magnitude


def nearest_code(value: Fraction, target_mantissa_bits: int, negative: bool) -> int:
    unsigned_count = 2 ** (target_mantissa_bits + 2)
    magnitudes = [oracle_e2(code, target_mantissa_bits) for code in range(unsigned_count)]
    unsigned = min(range(unsigned_count), key=lambda code: (abs(magnitudes[code] - abs(value)), code % 2, code))
    return unsigned + (unsigned_count if negative else 0)


def run_tests(modules, device, out: Path, log=print) -> dict:
    device = torch.device(device)
    folder = Path(out) / 'synthetic_tests'
    folder.mkdir(exist_ok=False)
    report = {'synthetic_fixture_only': True, 'SVHN_used': False, 'optimizer_steps': 0,
              'batchnorm_recalibration': False, 'device': str(device)}

    # Independent binary-string oracle for all254 FINITE E4M3 codes.
    finite_codes = [code for code in range(256) if code not in (127, 255)]
    hcodes = torch.tensor(finite_codes + [0, 0], dtype=torch.uint8, device=device).reshape(1, 8, 32)
    hscale = torch.full((1, 8), 127, dtype=torch.uint8, device=device)
    h = high.MXTensor(hcodes, hscale, (1, 254))
    original_h = h.payload()
    e, c, r = tr.exponent_truncate(h), tr.transform_codes(h, 'combined864'), tr.round_source_codes(h, 8, 6)
    expected_e, expected_c, expected_r = [], [], []
    for code in finite_codes:
        bits = f'{code:08b}'
        expected_e.append(int(bits[0:3] + bits[5:8], 2))
        expected_c.append(int(bits[0:3] + bits[5], 2))
        expected_r.append(nearest_code(oracle_e4(code), 3, code >= 128))
    assert e.codes.flatten()[:254].tolist() == expected_e
    assert c.codes.flatten()[:254].tolist() == expected_c
    assert r.codes.flatten()[:254].tolist() == expected_r
    assert middle.decode_elements(e.codes).flatten()[:254].tolist() == [float(oracle_e2(v, 3)) for v in expected_e]
    assert low.decode_elements(c.codes).flatten()[:254].tolist() == [float(oracle_e2(v, 1)) for v in expected_c]
    assert torch.equal(torch.signbit(high.decode_elements(h.codes)), torch.signbit(middle.decode_elements(e.codes)))
    assert torch.equal(torch.signbit(high.decode_elements(h.codes)), torch.signbit(low.decode_elements(c.codes)))
    assert not core.differences(original_h, h.payload())
    assert e.codes.data_ptr() != h.codes.data_ptr() and e.scale_codes.data_ptr() != h.scale_codes.data_ptr()
    assert torch.equal(e.scale_codes, hscale) and torch.equal(c.scale_codes, hscale)
    assert torch.equal(middle.unpack6(middle.pack6(e.codes)), e.codes)
    assert torch.equal(low.unpack4(low.pack4(c.codes)), c.codes)
    report['all_254_finite_E4_codes_MSB_exponent_and_combined_oracles'] = True
    report['all_254_finite_E4_codes_RNE_fixed_scale_oracle'] = True

    # Repeated mantissa route matches the prior independent64-code oracle.
    mcodes = torch.arange(64, dtype=torch.uint8, device=device).reshape(2, 1, 32)
    mscale = torch.full((2, 1), 127, dtype=torch.uint8, device=device)
    m = middle.MXTensor(mcodes, mscale, (2, 32))
    original_m = m.payload()
    mt, mr = tr.mantissa_truncate(m), tr.round_source_codes(m, 6, 4)
    expected_mt = [int(f'{code:06b}'[:4], 2) for code in range(64)]
    expected_mr = [nearest_code(oracle_e2(code, 3), 1, code >= 32) for code in range(64)]
    assert mt.codes.flatten().tolist() == expected_mt
    assert mr.codes.flatten().tolist() == expected_mr
    assert low.decode_elements(mt.codes).flatten().tolist() == [float(oracle_e2(v, 1)) for v in expected_mt]
    assert bool((low.decode_elements(mt.codes).abs() <= middle.decode_elements(m.codes).abs()).all())
    assert torch.equal(torch.signbit(middle.decode_elements(m.codes)), torch.signbit(low.decode_elements(mt.codes)))
    assert not core.differences(original_m, m.payload())
    report['all_64_E2M3_codes_mantissa_and_RNE_oracles'] = True
    report['signed_zero_subnormals_and_packed_roundtrip'] = True

    # All finite scale bytes are retained without invoking any scale selector.
    hwide = high.MXTensor(h.codes.expand(255, -1, -1).clone(),
                          torch.arange(255, dtype=torch.uint8, device=device).reshape(-1, 1).expand(-1, 8).clone(),
                          (255, 254))
    mwide = middle.MXTensor(torch.arange(64, dtype=torch.uint8, device=device).reshape(1, 2, 32).expand(255, -1, -1).clone(),
                            torch.arange(255, dtype=torch.uint8, device=device).reshape(-1, 1).expand(-1, 2).clone(),
                            (255, 64))
    for path, source in [('round64', mwide), ('mantissa64', mwide), ('round86', hwide),
                         ('exponent86', hwide), ('combined864', hwide)]:
        result = tr.transform_codes(source, path)
        assert torch.equal(result.scale_codes, source.scale_codes), path
    report['all_255_finite_E8M0_codes_preserved_all_five_derivations'] = True
    # This tests code retention, not an invalid claim that every arbitrary
    # finite element/scale combination decodes into finite FP32 at its maximum.

    for code, expected in [(1, Fraction(1, 8)), (8, Fraction(0)), (56, Fraction(1)),
                           (64, Fraction(2)), (96, Fraction(4)), (126, Fraction(7))]:
        index = finite_codes.index(code)
        assert oracle_e2(expected_e[index], 3) == expected
    assert oracle_e2(expected_e[finite_codes.index(1)], 3) / oracle_e4(1) == 64
    assert oracle_e2(expected_e[finite_codes.index(126)], 3) / oracle_e4(126) == Fraction(1, 64)
    assert expected_r[finite_codes.index(126)] == 31  # fixed-scale RNE saturates to7.5.
    report['naive_exponent_is_not_uniform_numeric_scaling'] = True
    report['round86_fixed_source_scale_saturation_explicit'] = True

    example = torch.tensor([[7., 1.375, -1.375, 1.3, 0.125, -0.125, 0., -0.]], device=device)
    source = middle.encode_tensor(example)
    rounded, truncated, direct = tr.round_source_codes(source, 6, 4), tr.mantissa_truncate(source), low.encode_tensor(example)
    assert torch.equal(source.scale_codes, direct.scale_codes)
    assert (direct.decode()[0, 1].item(), rounded.decode()[0, 1].item(), truncated.decode()[0, 1].item()) == (1.5, 1.5, 1.)
    assert (direct.decode()[0, 3].item(), rounded.decode()[0, 3].item(), truncated.decode()[0, 3].item()) == (1.5, 1., 1.)
    assert torch.signbit(truncated.decode()[0, 5]).item()
    report['previous_mantissa_examples_and_double_rounding_repeated'] = True
    for shape in [(2, 35), (2, 3, 3, 3), (3, 1)]:
        n = 1
        for d in shape: n *= d
        values = torch.linspace(-.31, .71, n, device=device).reshape(shape)
        for source_bits in (6, 8):
            source = adapter.codec(source_bits).encode_tensor(values)
            for path in tr.DERIVED_PATHS:
                if tr.PATH_SPEC[path]['source_bits'] != source_bits: continue
                result = tr.transform_codes(source, path)
                result.validate()
                assert result.decode().shape == values.shape
    for bit in (6, 8):
        source = adapter.codec(bit).encode_tensor(torch.zeros((2, 35), device=device))
        for path in tr.DERIVED_PATHS:
            if tr.PATH_SPEC[path]['source_bits'] == bit:
                assert bool((tr.transform_codes(source, path).codes == 0).all())
    report['multirow_padding_and_zero_blocks'] = True

    negatives = []
    for nan in (127, 255):
        invalid = high.MXTensor(torch.full((1, 1, 32), nan, dtype=torch.uint8, device=device),
                                torch.full((1, 1), 127, dtype=torch.uint8, device=device), (1, 32))
        negatives.append(must_fail(lambda: tr.exponent_truncate(invalid), 'E4_NaN_' + str(nan)))
    negatives.append(must_fail(lambda: tr.exponent_truncate(high.MXTensor(h.codes, torch.full_like(h.scale_codes, 255), h.shape)), 'E8M0_NaN_255'))
    negatives.append(must_fail(lambda: tr.exponent_truncate(high.MXTensor(h.codes.to(torch.int16), h.scale_codes, h.shape)), 'wrong_code_dtype'))
    badpad = high.MXTensor(h.codes.clone(), h.scale_codes.clone(), h.shape); badpad.codes[..., -1] = 1
    negatives.append(must_fail(lambda: tr.exponent_truncate(badpad), 'nonzero_padding'))
    negatives.append(must_fail(lambda: tr.exponent_truncate(m), 'wrong_source_type'))
    negatives.append(must_fail(lambda: tr.transform_codes(h, 'unknown'), 'unknown_path'))

    # Actual frozen model reconstruction from stored source codes. Distinct BN
    # fixture values make target-bank routing observable without calibration.
    torch.manual_seed(321)
    model = adapter.adapt_copy(modules.models.svhnq([1, 2, 4, 8, 32], expand=2).to(device)).eval().requires_grad_(False)
    with torch.no_grad():
        for name in adapter.BN_LAYERS:
            for bit in (8, 6, 4):
                bank = model.get_submodule(name).bn_dict[adapter.BANK[bit]]
                bank.running_mean.fill_(bit / 100.)
                bank.bias.fill_(bit / 200.)
    source_state = s.legacy.state_digest(model.state_dict())
    provenance = {'source_epoch': 0, 'checkpoint_sha256': '0' * 64, 'synthetic_fixture_only': True,
                  'training': 'NONE; fresh synthetic model with distinct BN fixtures',
                  'encoding_device': str(device)}
    direct_payloads, source_files, source_hashes = {}, {}, {}
    for bit in (8, 6, 4):
        payload, _ = adapter.export_payload(model, bit, provenance)
        path = folder / f'synthetic_direct{bit}.pt'
        info = core.atomic_save(path, payload)
        direct_payloads[bit] = torch.load(path, map_location='cpu', weights_only=True)
        source_files[bit], source_hashes[bit] = path, info['sha256']
    report['source_set_checks'] = tr.check_source_set(direct_payloads)
    derived, restored_models = {}, {}
    for path_id in tr.DERIVED_PATHS:
        spec = tr.PATH_SPEC[path_id]
        payload = tr.derive_payload(direct_payloads[spec['source_bits']], path_id, source_hashes[spec['source_bits']])
        filename = folder / f'synthetic_{path_id}.pt'
        info = core.atomic_save(filename, payload)
        loaded = torch.load(filename, map_location='cpu', weights_only=True)
        assert not core.differences(payload, loaded)
        assert core.file_sha256(filename) == info['sha256']
        derived[path_id] = loaded
        restored_models[path_id] = tr.load_derived_model(modules.models, loaded, device)

    def forbidden(*args, **kwargs):
        raise AssertionError('Verboden kwantisatie of G(W) in een bittruncatieroute')

    with ExitStack() as stack:
        for codec in (high, middle, low):
            stack.enter_context(patch.object(codec, 'encode_tensor', forbidden))
            stack.enter_context(patch.object(codec, 'encode_elements', forbidden))
        stack.enter_context(patch.object(adapter, 'effective_reference_weight', forbidden))
        for path_id in ('mantissa64', 'exponent86', 'combined864'):
            spec = tr.PATH_SPEC[path_id]
            payload = tr.derive_payload(direct_payloads[spec['source_bits']], path_id, source_hashes[spec['source_bits']])
            frozen = tr.load_derived_model(modules.models, payload, device)
            assert not {name + '.weight' for name in adapter.TARGET_LAYERS}.intersection(dict(frozen.named_parameters()))
            assert not core.differences(payload, derived[path_id])
    report['three_bit_routes_require_no_encoder_G_or_quantized_layer_master'] = True
    test_x = torch.randn((3, 3, 40, 40), device=device)
    with torch.no_grad():
        for bit in (8, 6, 4):
            path_id = f'direct{bit}'
            frozen = adapter.load_exported_model(modules.models, direct_payloads[bit], device)
            adapter.select_mode(model, bit)
            live_output, frozen_output = model(test_x), frozen(test_x)
            check = {'exact': bool(torch.equal(live_output, frozen_output)),
                     'max_abs_difference': float((live_output - frozen_output).abs().max()),
                     'allclose_rtol1e4_atol1e5': bool(torch.allclose(live_output, frozen_output, rtol=1e-4, atol=1e-5))}
            assert check['allclose_rtol1e4_atol1e5'], (path_id, check)
            report['live_vs_export_' + path_id] = check
            restored_models[path_id] = frozen
        for path_id in tr.PATHS:
            frozen = restored_models[path_id]
            state_before = s.legacy.state_digest(frozen.state_dict())
            output = frozen(test_x)
            assert output.shape == (3, 10) and bool(torch.isfinite(output).all())
            assert all(not module.training for module in frozen.modules())
            assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
            assert all(frozen.get_submodule(name).weight_mode == tr.PATH_SPEC[path_id]['target_bits']
                       for name in adapter.BN_LAYERS)
            assert s.legacy.state_digest(frozen.state_dict()) == state_before
    assert s.legacy.state_digest(model.state_dict()) == source_state
    report['all_eight_models_A32_frozen_target_BN_no_state_mutation'] = True
    weight_rows, count_rows = tr.weight_diagnostics(direct_payloads, derived, device)
    assert len(weight_rows) == 48 and len(count_rows) == 48
    assert all(row['scales_equal_source'] for row in count_rows)
    assert all(row['rms_ratio_to_direct'] == 1. for row in count_rows if row['path'].startswith('direct'))
    report['weight_diagnostics_48_pairs_48_route_layer_rows'] = True

    mutated = copy.deepcopy(direct_payloads)
    mutated[4]['mx_layers'][adapter.TARGET_LAYERS[0]]['scale_codes'][0, 0] += 1
    negatives.append(must_fail(lambda: tr.check_source_set(mutated), 'mismatched_E2_direct_scales'))
    wrongbank = copy.deepcopy(derived['exponent86']); wrongbank['bn_bank_key'] = '8'
    negatives.append(must_fail(lambda: tr.load_derived_model(modules.models, wrongbank, device), 'wrong_target_BN_bank'))
    hidden = copy.deepcopy(derived['mantissa64'])
    hidden['auxiliary_fp32_state'][adapter.TARGET_LAYERS[0] + '.weight'] = torch.ones(1)
    negatives.append(must_fail(lambda: tr.validate_derived(hidden), 'hidden_quantized_layer_master'))
    changed_policy = copy.deepcopy(derived['exponent86']); changed_policy['derivation']['bias_compensation'] = True
    negatives.append(must_fail(lambda: tr.validate_derived(changed_policy), 'bias_compensation_not_allowed'))
    wrongsource = copy.deepcopy(direct_payloads); wrongsource[6]['provenance']['source_epoch'] = 1
    negatives.append(must_fail(lambda: tr.check_source_set(wrongsource), 'mixed_checkpoint_provenance'))
    negatives.append(must_fail(lambda: tr.derive_payload(direct_payloads[8], 'exponent86', 'invalid'), 'invalid_source_digest'))
    negatives.append(must_fail(lambda: core.atomic_save(source_files[8], direct_payloads[8]), 'source_overwrite_refused'))
    assert all(core.file_sha256(source_files[bit]) == source_hashes[bit] for bit in (8, 6, 4))
    report['negative_tests'] = negatives
    report['source_exports_unchanged'] = True
    report['no_training_or_calibration'] = True
    report['status'] = 'PASS'
    s.write_json(folder / 'synthetic_test_results.json', report)
    s.write_csv(folder / 'weight_pair_metrics.csv', weight_rows)
    s.write_csv(folder / 'layer_code_metrics.csv', count_rows)
    log('CHECK PASS: 254 E4-codes, 64 E2M3-codes, vaste schalen, acht bevroren A32-routes en doel-BN.')
    return report
