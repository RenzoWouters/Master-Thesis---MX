"""Synthetic code/inference tests, no SVHN or training updates."""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import mx_format as high
import mx_e2m1 as low
import dual_adapter as adapter
import evaluation as ev
import truncation as tr
import resume_core as core
import svhn_support as s


def must_fail(call, label):
    try:
        call()
    except (ValueError, RuntimeError, AssertionError, KeyError, TypeError):
        return label
    raise AssertionError('Negatieve test werd niet geweigerd: ' + label)


def exact_scalar(code, mantissa):
    sign = -1 if code // (2 ** (2 + mantissa)) else 1
    exp = (code // 2 ** mantissa) % 4
    frac = Fraction(code % (2 ** mantissa), 2 ** mantissa)
    mag = frac if exp == 0 else Fraction(2) ** (exp - 1) * (1 + frac)
    return sign * mag


def run_tests(modules, device, out: Path, log=print) -> dict:
    device = torch.device(device)
    folder = out / 'synthetic_tests'; folder.mkdir(exist_ok=False)
    results = {'synthetic_fixture_only': True, 'SVHN_used': False, 'optimizer_steps': 0,
               'device': str(device)}
    codes = torch.arange(64, dtype=torch.uint8, device=device).reshape(2, 1, 32)
    scale = torch.full((2, 1), 127, dtype=torch.uint8, device=device)
    source = high.MXTensor(codes, scale, (2, 32))
    before = source.payload()
    c = tr.truncate_codes(source); b = tr.round_source_codes(source)
    positive = [exact_scalar(i, 1) for i in range(8)]
    c_expected, b_expected = [], []
    for i in range(64):
        mag = abs(exact_scalar(i, 3)); sign = 8 if i >= 32 else 0
        floor = max(j for j, value in enumerate(positive) if value <= mag)
        nearest = min(range(8), key=lambda j: (abs(positive[j] - mag), j % 2, j))
        c_expected.append(sign + floor); b_expected.append(sign + nearest)
    assert c.codes.flatten().tolist() == c_expected
    assert b.codes.flatten().tolist() == b_expected
    assert not core.differences(before, source.payload())
    assert torch.equal(torch.signbit(high.decode_elements(codes)), torch.signbit(low.decode_elements(c.codes)))
    assert bool((low.decode_elements(c.codes).abs() <= high.decode_elements(codes).abs()).all())
    assert torch.equal(low.unpack4(low.pack4(c.codes)), c.codes)
    results['all_64_source_codes_oracle_C_and_B'] = True
    results['sign_exponent_subnormal_signed_zero_packing_verified'] = True
    # Preserve every finite E8M0 encoding without calling a scale selector.
    for exp_code in range(255):
        scaled = high.MXTensor(codes, torch.full_like(scale, exp_code), (2, 32))
        derived = tr.truncate_codes(scaled)
        assert torch.equal(derived.scale_codes, scaled.scale_codes)
        if exp_code <= 252:  # All E2M3 codes remain finite in FP32 through exponent 125.
            assert torch.isfinite(derived.decode()).all()
    results['all_255_scale_codes_preserved'] = True
    x = torch.tensor([[7.0, 1.375, -1.375, 1.3, 0.125, -0.125, 0.0, -0.0]],
                     dtype=torch.float32, device=device)
    sx = high.encode_tensor(x)
    cx, bx, ax = tr.truncate_codes(sx), tr.round_source_codes(sx), low.encode_tensor(x)
    assert torch.equal(sx.scale_codes, ax.scale_codes)
    assert (ax.decode()[0, 1].item(), bx.decode()[0, 1].item(), cx.decode()[0, 1].item()) == (1.5, 1.5, 1.0)
    assert (ax.decode()[0, 3].item(), bx.decode()[0, 3].item(), cx.decode()[0, 3].item()) == (1.5, 1.0, 1.0)
    assert cx.decode()[0, 2].item() == -1.0 and torch.signbit(cx.decode()[0, 5]).item()
    # Multi-row nonmultiple-of-32 padding, zero blocks, tiny and normal weights.
    for shape in [(2, 35), (2, 3, 3, 3), (3, 1)]:
        values = torch.linspace(-0.31, 0.71, int(np.prod(shape)), device=device).reshape(shape)
        hi = high.encode_tensor(values)
        lo = tr.truncate_codes(hi)
        lo.validate(); assert lo.decode().shape == values.shape
    zeros = high.encode_tensor(torch.zeros((2, 35), device=device))
    assert bool((tr.truncate_codes(zeros).codes == 0).all())
    results['round_vs_truncate_and_double_rounding_examples'] = True
    results['multirow_padding_zero_blocks'] = True
    results['negative_tests'] = []
    badscale = high.MXTensor(codes, torch.full_like(scale, 255), (2, 32))
    results['negative_tests'].append(must_fail(lambda: tr.truncate_codes(badscale), 'NaN_scale_255'))
    badcode = high.MXTensor(torch.full_like(codes, 64), scale, (2, 32))
    results['negative_tests'].append(must_fail(lambda: tr.truncate_codes(badcode), 'code_64'))
    padding = high.encode_tensor(torch.ones((1, 1), device=device)); padding.codes[..., -1] = 1
    results['negative_tests'].append(must_fail(lambda: tr.truncate_codes(padding), 'nonzero_padding'))
    results['negative_tests'].append(must_fail(lambda: tr.truncate_codes(high.MXTensor(codes.to(torch.int8), scale, (2, 32))), 'signed_code_storage'))
    # The actual network from the archive, freshly initialized, NOT trained data.
    torch.manual_seed(321)
    model = adapter.adapt_copy(modules.models.svhnq([1, 2, 4, 8, 32], expand=2)).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        for name in adapter.BN_LAYERS:
            bank = model.get_submodule(name).bn_dict['4']
            bank.running_mean.fill_(0.05); bank.bias.fill_(0.03)
    digest_before = s.legacy.state_digest(model.state_dict())
    provenance = {'source_epoch': 0, 'checkpoint_sha256': '0' * 64, 'synthetic_fixture_only': True,
                  'training': 'NONE; fresh synthetic model with distinct low BN fixture'}
    hp, _ = adapter.export_payload(model, 6, provenance)
    ap, _ = adapter.export_payload(model, 4, provenance)
    high_file = folder / 'synthetic_high_only.pt'
    item = core.atomic_save(high_file, hp)
    loaded_high = ev.read_export(high_file, item['sha256'])
    # C must not round or fetch any master. Monkeypatch both full quantizers
    # and element encoders into traps, then derive and load the truncated file.
    def forbidden(*args, **kwargs):
        raise AssertionError('Verboden requantisatie in C-pad')
    with patch.object(high, 'encode_tensor', forbidden), patch.object(low, 'encode_tensor', forbidden), \
         patch.object(low, 'encode_elements', forbidden), patch.object(high, 'encode_elements', forbidden):
        cp = tr.derive_payload(loaded_high, 'mantissa_truncation', item['sha256'])
        cmodel = tr.load_derived_model(modules.models, cp, device)
    results['C_requires_no_quantizer_or_master'] = True
    paths, assembled, weight_rows, counts = ev.assemble_paths(modules.models, loaded_high, ap, item['sha256'],
                                                              folder, 'synthetic', device)
    test_x = torch.randn((3, 3, 40, 40), device=device)
    with torch.no_grad():
        for bit, key in [(6, tr.PATHS[0]), (4, tr.PATHS[1])]:
            adapter.select_mode(model, bit)
            ref_logits, restored_logits = model(test_x), paths[key](test_x)
            results['live_vs_export_' + str(bit)] = {
                'allclose_rtol1e4_atol1e5': bool(torch.allclose(ref_logits, restored_logits, rtol=1e-4, atol=1e-5)),
                'exact': bool(torch.equal(ref_logits, restored_logits)),
                'max_abs_difference': float((ref_logits - restored_logits).abs().max())}
            assert results['live_vs_export_' + str(bit)]['allclose_rtol1e4_atol1e5']
        assert torch.allclose(cmodel(test_x), paths[tr.PATHS[3]](test_x), rtol=1e-4, atol=1e-5)
    assert s.legacy.state_digest(model.state_dict()) == digest_before
    mutated = copy.deepcopy(ap)
    mutated['mx_layers'][adapter.TARGET_LAYERS[0]]['scale_codes'][0, 0] += 1
    results['negative_tests'].append(must_fail(lambda: tr.check_source_pair(hp, mutated), 'mismatched_source_scale'))
    wrongbank = copy.deepcopy(cp); wrongbank['bn_bank_key'] = '32'
    results['negative_tests'].append(must_fail(lambda: tr.load_derived_model(modules.models, wrongbank, device), 'wrong_BN_bank'))
    hidden = copy.deepcopy(cp); hidden['auxiliary_fp32_state'][adapter.TARGET_LAYERS[0]+'.weight'] = torch.ones(1)
    results['negative_tests'].append(must_fail(lambda: tr.load_derived_model(modules.models, hidden, device), 'hidden_master'))
    results['negative_tests'].append(must_fail(lambda: tr.derive_payload(hp, 'unknown', item['sha256']), 'unknown_method'))
    results['negative_tests'].append(must_fail(lambda: ev.read_export(high_file, 'f' * 64), 'wrong_checksum'))
    results['negative_tests'].append(must_fail(lambda: core.atomic_save(high_file, hp), 'overwrite_refused'))
    rp = {'predictions': np.array([0, 1, 2, 3]), 'logits': np.zeros((4, 10))}
    op = {'predictions': np.array([0, 2, 1, 5]), 'logits': np.zeros((4, 10))}
    pair = ev.paired_outputs(rp, op, np.array([0, 2, 2, 4]))
    assert pair['wrong_to_correct'] == pair['correct_to_wrong'] == pair['wrong_to_other_wrong'] == 1
    assert pair['net_correct_change'] == 0 and pair['prediction_changes'] == 3
    results['paired_correctness_accounting'] = True
    # Complete evaluator on seven synthetic examples, including a short batch.
    xx = torch.randn((7, 3, 40, 40)); yy = torch.arange(7, dtype=torch.int64)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xx, yy, torch.arange(7)), batch_size=3)
    metrics, pairs, checks = ev.evaluate_paths(paths, loader, 7, device, folder, 'synthetic', 0, log)
    assert len(metrics) == len(pairs) == 4 and checks['all_model_and_bn_states_unchanged']
    results['synthetic_full_evaluator'] = checks
    results['no_training_or_calibration'] = True
    results['all_negative_tests_rejected'] = True
    results['source_export_unchanged'] = core.file_sha256(high_file) == item['sha256']
    results['status'] = 'PASS'
    s.write_json(folder / 'synthetic_test_results.json', results)
    # Only for offline notebook validation, clearly tagged as synthetic.
    s.write_csv(folder / 'metrics.csv', metrics)
    s.write_csv(folder / 'paired_metrics.csv', pairs)
    s.write_csv(folder / 'weight_pair_metrics.csv', [{'role': 'synthetic', 'epoch': 0, **r} for r in weight_rows])
    s.write_csv(folder / 'layer_code_metrics.csv', [{'role': 'synthetic', 'epoch': 0, **r} for r in counts])
    return results
