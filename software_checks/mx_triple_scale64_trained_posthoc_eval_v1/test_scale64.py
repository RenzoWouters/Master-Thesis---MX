"""Independent finite-code and source-only model tests; no actual SVHN training."""
from __future__ import annotations

import copy
import json
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import torch

import joint_adapter as adapter
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low
import posthoc_truncation as previous
import trunc_adapter as training
import resume_core as core
import svhn_support as support
import scale64 as tr
import evaluation as ev


def must_fail(call, label):
    try:
        call()
    except (ValueError, RuntimeError, AssertionError, KeyError, TypeError):
        return label
    raise AssertionError('Negatieve controle werd niet geweigerd: ' + label)


def oracle_e4(code):
    if code in (127, 255): raise ValueError('E4 NaN')
    exponent, fraction = (code // 8) % 16, Fraction(code % 8, 8)
    magnitude = fraction / 64 if exponent == 0 else (1 + fraction) * Fraction(2) ** (exponent - 7)
    return -magnitude if code >= 128 else magnitude


def oracle_e2(code, mantissa_bits=3):
    width = mantissa_bits + 2
    exponent = (code // 2 ** mantissa_bits) % 4
    fraction = Fraction(code % 2 ** mantissa_bits, 2 ** mantissa_bits)
    magnitude = fraction if exponent == 0 else (1 + fraction) * Fraction(2) ** (exponent - 1)
    return -magnitude if code >= 2 ** width else magnitude


def nearest_e2_code(value, negative):
    code = min(range(32), key=lambda c: (abs(oracle_e2(c) - abs(value)), c % 2, c))
    return code + (32 if negative else 0)


def _oracle_luts(device):
    expected = {key: [] for key in tr.DERIVED_PATHS}
    for code in range(256):
        if code in (127, 255):
            for values in expected.values(): values.append(0)
            continue
        bits = f'{code:08b}'
        a6 = int(bits[:3] + bits[5:], 2)
        c6 = nearest_e2_code(oracle_e4(code) / 64, code >= 128)
        for family, value in [('A', a6), ('B', a6), ('C', c6)]:
            expected[family + '6'].append(value)
            expected[family + '4'].append(int(f'{value:06b}'[:4], 2))
    return {path: torch.tensor(values, dtype=torch.uint8, device=device) for path, values in expected.items()}


class _OracleQuantizer(torch.nn.Module):
    def __init__(self, weights):
        super().__init__(); self.register_buffer('values', weights.detach().clone())
        self.abit = self.wbit = 32

    def forward(self, ignored_weight):
        return self.values


def _oracle_model(source_model, source8, path, luts, device):
    """Independent lookup mapping, attached to the original quantized convolutions."""
    model = copy.deepcopy(source_model).eval().requires_grad_(False)
    bit = tr.PATH_SPEC[path]['target_bits']
    for name in adapter.TARGET_LAYERS:
        src = high.MXTensor.from_payload(source8['mx_layers'][name], device)
        target_codes = luts[path][src.codes.long()]
        scales = (src.scale_codes.int() + (0 if path[0] == 'A' else 6)).to(torch.uint8)
        mx = adapter.codec(bit).MXTensor(target_codes, scales, src.shape)
        model.get_submodule(name).quantize_fn = _OracleQuantizer(mx.decode())
    adapter.select_mode(model, bit)
    return model


def _tests(modules, device, folder, log):
    report = {'synthetic_fixture_only': True, 'SVHN_used': False, 'optimizer_steps': 0,
              'batchnorm_recalibration': False, 'device': str(device)}
    luts = _oracle_luts(device)
    finite = [c for c in range(256) if c not in (127, 255)]
    codes = torch.tensor(finite + [0, 0], dtype=torch.uint8, device=device).reshape(1, 8, 32)
    scales = torch.full((1, 8), 127, dtype=torch.uint8, device=device)
    source = high.MXTensor(codes, scales, (1, 254)); before = source.payload()
    transformed = {}
    for path in tr.DERIVED_PATHS:
        result = tr.transform_codes(source, path); transformed[path] = result
        expected = luts[path][codes.long()]
        assert torch.equal(result.codes, expected), path
        assert torch.equal(result.scale_codes.int(), scales.int() + (0 if path[0] == 'A' else 6))
        bit = tr.PATH_SPEC[path]['target_bits']
        assert adapter.codec(bit).decode_elements(result.codes).flatten()[:254].tolist() == [
            float(oracle_e2(int(c), 3 if bit == 6 else 1)) for c in expected.flatten()[:254]]
        assert torch.equal(torch.signbit(high.decode_elements(codes)), torch.signbit(adapter.codec(bit).decode_elements(result.codes)))
        assert result.codes.data_ptr() != source.codes.data_ptr() and result.scale_codes.data_ptr() != source.scale_codes.data_ptr()
        unpack = middle.unpack6(middle.pack6(result.codes)) if bit == 6 else low.unpack4(low.pack4(result.codes))
        assert torch.equal(result.codes, unpack)
    assert not core.differences(source.payload(), before)
    assert tr.tensor_exact(transformed['B6'].decode(), transformed['A6'].decode() * 64)
    assert tr.tensor_exact(transformed['B4'].decode(), transformed['A4'].decode() * 64)
    for family in ('A', 'B', 'C'):
        assert torch.equal(transformed[family + '4'].codes, transformed[family + '6'].codes >> 2)
    report['all_254_finite_codes_independent_bitstrings_and_fraction_RNE_oracle'] = True
    report['A_B_same_codes_B_exact64_numeric_scaling_and_C_mantissa_after_RNE'] = True
    report['signed_zero_padding_packed_roundtrip_and_source_nonmutation'] = True

    tie_count = 0
    for code in finite:
        x = abs(oracle_e4(code) / 64)
        distances = [abs(x - oracle_e2(c)) for c in range(32)]
        candidates = [c for c, distance in enumerate(distances) if distance == min(distances)]
        if len(candidates) == 2:
            tie_count += 1
            assert int(luts['C6'][code]) % 2 == 0
    assert tie_count > 0
    report['reachable_E4_div64_RNE_ties_checked'] = tie_count
    for values, expected in [([32., 64., 128., 240., 256.], [4., 4., 4., 7.5, 4.])]:
        src = high.encode_tensor(torch.tensor([values], dtype=torch.float32, device=device))
        assert bool((src.scale_codes == 127).all())
        assert tr.transform_codes(src, 'A6').decode()[0].tolist() == expected
        assert tr.transform_codes(src, 'B6').decode()[0].tolist() == [64 * v for v in expected]
        assert tr.transform_codes(src, 'C6').decode()[0].tolist() == values
    report['shared_example_32_64_128_240_256_and_nonuniform_information_loss'] = True

    # Code/scale boundary proof deliberately does not claim every arbitrary
    # extreme element/scale combination would fit finite FP32 inference.
    for family, maximum in [('A', 254), ('B', 248), ('C', 248)]:
        size = maximum + 1
        wide = high.MXTensor(codes.expand(size, -1, -1).clone(),
            torch.arange(size, dtype=torch.uint8, device=device)[:, None].expand(-1, 8).clone(), (size, 254))
        for bit in (6, 4):
            result = tr.transform_codes(wide, family + str(bit))
            assert torch.equal(result.codes, luts[family + str(bit)][wide.codes.long()])
            assert torch.equal(result.scale_codes.int(), wide.scale_codes.int() + (0 if family == 'A' else 6))
    report['all255_A_scale_codes_and_all249_exact_shiftable_BC_scales_checked'] = True
    negatives = []
    for scale in range(249, 255):
        mx = high.MXTensor(codes, torch.full_like(scales, scale), source.shape)
        for path in ('B6', 'B4', 'C6', 'C4'):
            negatives.append(must_fail(lambda mx=mx, path=path: tr.transform_codes(mx, path), f'scale{scale}_{path}_reject_no_clamp'))
    for nan in (127, 255):
        invalid = high.MXTensor(torch.full_like(codes, nan), scales, source.shape)
        negatives.append(must_fail(lambda invalid=invalid: tr.transform_codes(invalid, 'C6'), f'E4NaN{nan}'))
    negatives.append(must_fail(lambda: tr.transform_codes(high.MXTensor(codes, torch.full_like(scales, 255), source.shape), 'A6'), 'scaleNaN'))
    badpad = high.MXTensor(codes.clone(), scales.clone(), source.shape); badpad.codes[..., -1] = 1
    negatives.append(must_fail(lambda: tr.transform_codes(badpad, 'B6'), 'nonzero_padding'))
    negatives.append(must_fail(lambda: tr.transform_codes(high.MXTensor(codes.to(torch.int16), scales, source.shape), 'C6'), 'wrong_dtype'))
    negatives.append(must_fail(lambda: tr.transform_codes(transformed['A6'], 'C6'), 'wrong_source_format'))
    negatives.append(must_fail(lambda: tr.transform_codes(source, 'unknown'), 'unknown_path'))
    for shape in ((2, 35), (2, 3, 3, 3), (3, 1)):
        n = math_prod(shape)
        for values in (torch.linspace(-.3, .7, n, device=device).reshape(shape), torch.zeros(shape, device=device)):
            src = high.encode_tensor(values)
            for path in tr.DERIVED_PATHS:
                result = tr.transform_codes(src, path); result.validate()
                assert result.decode().shape == shape
                if not bool(values.any()): assert not bool(result.codes.any())
    report['scale_overflow_NaN_type_padding_and_path_guards'] = True
    report['multirow_convolution_padding_and_zero_blocks'] = True

    torch.manual_seed(921)
    source_model = adapter.adapt_copy(modules.models.svhnq([1, 2, 4, 8, 32], expand=2).to(device)).eval().requires_grad_(False)
    with torch.no_grad():
        for name in adapter.BN_LAYERS:
            for bit in (8, 6, 4):
                bank = source_model.get_submodule(name).bn_dict[adapter.BANK[bit]]
                bank.running_mean.fill_(bit / 100.)
                bank.running_var.fill_(1 + bit / 50.)
                bank.weight.fill_(1 + bit / 25.)
                bank.bias.fill_(bit / 200.)
    original_state = support.legacy.state_digest(source_model.state_dict())
    provenance = {'source_epoch': 0, 'checkpoint_sha256': '0' * 64, 'synthetic_fixture_only': True,
                  'training': 'NONE; fresh fixture with distinct BN values', 'encoding_device': str(device)}
    direct = {}; direct_hashes = {}
    for bit in (8, 6, 4):
        payload, _ = adapter.export_payload(source_model, bit, copy.deepcopy(provenance))
        path = folder / f'synthetic_direct_{bit}.pt'; saved = core.atomic_save(path, payload)
        direct[bit] = ev.read_export(path, saved['sha256']); direct_hashes[bit] = saved['sha256']
    trunc_model = training.adapt_copy(source_model).eval().requires_grad_(False)
    trunc_source, _ = training.source_payload(trunc_model, copy.deepcopy(provenance))
    path = folder / 'synthetic_trunc_source8.pt'; saved = core.atomic_save(path, trunc_source)
    trunc = {8: ev.read_export(path, saved['sha256'])}; trunc_hashes = {8: saved['sha256']}
    for bit in (6, 4):
        payload = training.derive_payload(trunc[8], bit, trunc_hashes[8])
        path = folder / f'synthetic_trunc_{bit}.pt'; saved = core.atomic_save(path, payload)
        trunc[bit] = ev.read_export(path, saved['sha256']); trunc_hashes[bit] = saved['sha256']
    assert not core.differences(direct[8]['mx_layers'], trunc[8]['mx_layers'])
    sets = {'scale64': (trunc, trunc_hashes)}
    report['source_sets'] = {kind: tr.check_source_set(payloads, kind, hashes) for kind, (payloads, hashes) in sets.items()}
    x = torch.randn((3, 3, 40, 40), device=device)
    trained_models = {bit: tr.load_trained_model(modules.models, trunc[bit], device, 'scale64') for bit in (8, 6, 4)}
    comparisons = []
    with torch.no_grad():
        for bit in (8, 6, 4):
            training.select_mode(trunc_model, bit)
            live, frozen = trunc_model(x), trained_models[bit](x)
            assert torch.allclose(live, frozen, rtol=1e-4, atol=1e-5), bit
            comparisons.append({'case': f'scale64_training_live_vs_saved{bit}', 'exact': torch.equal(live, frozen),
                                'max_abs_difference': float((live - frozen).abs().max())})
    adapter.select_mode(source_model, 8)

    def forbidden(*args, **kwargs): raise AssertionError('Forbidden master/training operation during frozen derivation/inference')
    original_train = torch.nn.Module.train
    def train_guard(self, mode=True):
        if mode: forbidden()
        return original_train(self, False)
    all_input_states = {kind: copy.deepcopy(payloads) for kind, (payloads, _) in sets.items()}
    all_derived = {}
    with ExitStack() as stack:
        for codec in (high, middle, low): stack.enter_context(patch.object(codec, 'encode_tensor', forbidden))
        for module in (adapter, adapter.original_adapter, training):
            stack.enter_context(patch.object(module, 'effective_reference_weight', forbidden))
        stack.enter_context(patch.object(torch.optim.Adam, '__init__', forbidden))
        stack.enter_context(patch.object(torch.Tensor, 'backward', forbidden))
        stack.enter_context(patch.object(torch.autograd, 'backward', forbidden))
        stack.enter_context(patch.object(torch.nn.Module, 'train', train_guard))
        # Derivation permits only the E2M3 element encoder in C. A/B are separately
        # checked under a trap on that encoder as well.
        stack.enter_context(patch.object(high, 'encode_elements', forbidden))
        stack.enter_context(patch.object(low, 'encode_elements', forbidden))
        for kind, (payloads, hashes) in sets.items():
            saved_payloads = {}
            for path_id in tr.DERIVED_PATHS:
                if path_id[0] in ('A', 'B'):
                    with patch.object(middle, 'encode_elements', forbidden):
                        payload = tr.derive_payload(payloads[8], path_id, hashes[8], kind)
                else:
                    payload = tr.derive_payload(payloads[8], path_id, hashes[8], kind)
                path = folder / f'{kind}_{path_id}.pt'; saved = core.atomic_save(path, payload)
                loaded = ev.read_export(path, saved['sha256']); assert not core.differences(payload, loaded)
                saved_payloads[path_id] = loaded
                assert not core.differences(payload['auxiliary_fp32_state'], payloads[8]['auxiliary_fp32_state'])
                with patch.object(middle, 'encode_elements', forbidden), torch.no_grad():
                    frozen = tr.load_derived_model(modules.models, loaded, device)
                    digest = ev.frozen_state_check(frozen, path_id)
                    oracle = _oracle_model(source_model, payloads[8], path_id, luts, device)
                    actual, expected = frozen(x), oracle(x)
                    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (kind, path_id)
                    assert ev.frozen_state_check(frozen, path_id) == digest
                    assert all(p.grad is None and not p.requires_grad for p in frozen.parameters())
                    comparisons.append({'case': kind + '/' + path_id + '_independent_lookup_model',
                        'exact': torch.equal(actual, expected), 'max_abs_difference': float((actual - expected).abs().max())})
                    if path_id in ('B6', 'B4'):
                        stored = tr.load_trained_model(modules.models, payloads[int(path_id[1])], device, kind)
                        repeated = stored(x)
                        assert torch.allclose(repeated, actual, rtol=1e-4, atol=1e-5)
                        if torch.device(device).type == 'cpu': assert torch.equal(repeated, actual)
            all_derived[kind] = saved_payloads
            weight_rows, code_rows = tr.weight_diagnostics(payloads, saved_payloads, device)
            assert len(weight_rows) == 108 and len(code_rows) == 54
            for row in code_rows:
                assert row['source_bits'] == 8 and row['scale_anchor_path'] == 'source8'
                offset = 0 if row['path'] in ('source8', 'A6', 'A4') else 6
                assert row['scale_exponent_offset'] == offset
                assert json.loads(row['source_scale_delta_counts']) == {str(offset): row['blocks']}
                assert row['scales_equal_source'] is (offset == 0)
                if row['path'] in ('trained6', 'trained4'):
                    assert json.loads(row['scale_delta_to_direct_counts']) == {'0': row['blocks']}
                    assert row['rms_ratio_to_direct'] == 1.
            for bit in (6, 4):
                for row in weight_rows:
                    if row['reference_path'] == 'A' + str(bit) and row['other_path'] == 'B' + str(bit):
                        assert row['code_differences'] == 0 and not row['scale_bytes_equal']
        with patch.object(middle, 'encode_elements', forbidden), torch.no_grad():
            for kind, (payloads, _) in sets.items():
                for bit in (8, 6, 4):
                    model = tr.load_trained_model(modules.models, payloads[bit], device, kind)
                    key = 'source8' if bit == 8 else 'trained' + str(bit)
                    digest = ev.frozen_state_check(model, key); model(x)
                    assert ev.frozen_state_check(model, key) == digest
    for kind, (payloads, _) in sets.items(): assert not core.differences(payloads, all_input_states[kind])
    assert support.legacy.state_digest(source_model.state_dict()) == original_state
    paired_folder = folder / 'paired_evaluator'; paired_folder.mkdir()
    paths, checks, weights, counts = ev.assemble_paths(modules.models, trunc, trunc_hashes,
        paired_folder, 'best_E4M3_test', device, 'scale64')
    xx = torch.cat([x, x[:2]], dim=0).cpu()
    yy = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xx, yy, torch.arange(5)),
                                         batch_size=3, shuffle=False)
    metrics, pairs, details = ev.evaluate_paths(paths, loader, 5, device, paired_folder,
                                               'best_E4M3_test', 0, log)
    assert len(metrics) == 9 and len(pairs) == 12 and len(weights) == 108 and len(counts) == 54
    for row in metrics:
        assert row['source_weight_bits'] == 8
        expected_offset = 0 if row['path'] in ('source8', 'A6', 'A4') else 6
        assert row['scale_exponent_offset'] == expected_offset
    report['all_route_source_metadata_relative_to_E4_stored_lower_offset6'] = True
    assert checks['B_same_frozen_state_as_trained']
    assert checks['trained_lower_codes_exact_B_routes'] and checks['trained_lower_scale_codes_exact_B_routes']
    for bit in ('6', '4'):
        r = details['B_same_run_reproduction'][bit]
        assert r['logits_allclose']
        if torch.device(device).type == 'cpu':
            assert r['logits_exact'] and r['ce_exact']
            assert r['predictions_equal'] and r['correct1_difference'] == 0
    report['paired_evaluator'] = {'synthetic_fixture_only': True, 'n_test': 5,
        'metrics': metrics, 'paired_metrics': pairs, 'checks': {**checks, **details},
        'weight_metric_rows': len(weights), 'code_metric_rows': len(counts)}
    report['model_comparisons'] = comparisons
    report['scale64_training_schemas_save_reload_source_only_independent_model_checks'] = True
    report['BN_banks_frozen_and_unchanged_all_paths_A32_no_masters_or_training'] = True
    report['A_B_no_element_encoder_C_only_E2_element_encoder_no_tensor_scale_reselection'] = True
    report['all_reconstructions_and_inference_no_encoder_no_G_no_optimizer_no_backward'] = True

    negatives.append(must_fail(lambda: tr.check_source_set(direct, 'scale64', direct_hashes), 'direct_exports_as_scale64'))
    negatives.append(must_fail(lambda: tr.check_source_set(trunc, 'direct', trunc_hashes), 'unsupported_direct_training_kind'))
    bad = copy.deepcopy(trunc); bad[6]['derivation']['source_sha256'] = '1' * 64
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'wrong_source_hash'))
    bad = copy.deepcopy(trunc)
    first = adapter.TARGET_LAYERS[0]; changed = middle.MXTensor.from_payload(bad[6]['mx_layers'][first])
    changed.codes.flatten()[0] ^= 1; bad[6]['mx_layers'][first] = changed.payload()
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'mutated_trained_derived_codes'))
    bad = copy.deepcopy(trunc); next(iter(bad[6]['auxiliary_fp32_state'].values())).view(-1)[0] += 1
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'different_auxiliary_state'))
    bad = copy.deepcopy(all_derived['scale64']['B6']); bad['derivation']['scale_exponent_offset'] = 5
    negatives.append(must_fail(lambda: tr.validate_derived(bad), 'mutated_scale_policy'))
    bad = copy.deepcopy(all_derived['scale64']['C4']); bad['bn_bank_key'] = '8'
    negatives.append(must_fail(lambda: tr.validate_derived(bad), 'wrong_target_BN'))
    bad = copy.deepcopy(all_derived['scale64']['C6']); bad['auxiliary_fp32_state'][first + '.weight'] = torch.zeros(1)
    negatives.append(must_fail(lambda: tr.validate_derived(bad), 'hidden_MX_master'))
    bad = copy.deepcopy(all_derived['scale64']['A6']); bad['provenance']['hidden'] = torch.zeros(1)
    negatives.append(must_fail(lambda: tr.validate_derived(bad), 'hidden_provenance_tensor'))
    bad = copy.deepcopy(trunc[8]); mx = high.MXTensor.from_payload(bad['mx_layers'][first])
    mx.codes.fill_(96); cols = math_prod(mx.shape[1:]); pad = mx.codes.shape[1] * 32 - cols
    if pad: mx.codes[:, -1, -pad:] = 0
    mx.scale_codes.fill_(248); bad['mx_layers'][first] = mx.payload()
    assert bool(torch.isfinite(mx.decode()).all())
    negatives.append(must_fail(lambda: tr.derive_payload(bad, 'B6', trunc_hashes[8], 'scale64'), 'finite_E8_scale_but_FP32_overflow_rejected'))
    bad = copy.deepcopy(trunc); bad[8]['schema'] = 'mx-triple-truncation-aware-source-v1'
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'naive_A_source_schema_rejected'))
    bad = copy.deepcopy(trunc); bad[6]['schema'] = 'mx-triple-truncation-aware-derived-v1'
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'naive_A_derived_schema_rejected'))
    bad = copy.deepcopy(trunc)
    changed = middle.MXTensor.from_payload(bad[6]['mx_layers'][first])
    changed.scale_codes.sub_(6); bad[6]['mx_layers'][first] = changed.payload()
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'A_scales_mislabeled_as_B_rejected'))
    bad = copy.deepcopy(trunc); bad[4]['derivation']['scale_factor'] = 4096
    negatives.append(must_fail(lambda: tr.check_source_set(bad, 'scale64', trunc_hashes), 'second_scale64_application_rejected'))
    negatives.append(must_fail(lambda: tr.check_source_set(trunc, 'truncation', trunc_hashes), 'unsupported_naive_training_kind'))
    report['negative_controls'] = negatives
    report['status'] = 'PASS'
    log('Schaal64-getraind-zelftest PASS: B-bronschema; A/B/C; exacte B-reconstructie; bevroren BN.')
    return report


def math_prod(values):
    result = 1
    for value in values: result *= value
    return result


def run_tests(modules, device, out: Path, log=print):
    device = torch.device(device)
    folder = Path(out) / 'synthetic_tests'; folder.mkdir(exist_ok=False)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    threads = torch.get_num_threads()
    if device.type == 'cpu': torch.set_num_threads(1)
    try:
        report = _tests(modules, device, folder, log)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        if torch.get_num_threads() != threads: torch.set_num_threads(threads)
    report['RNG_and_thread_count_restored'] = True
    return report
