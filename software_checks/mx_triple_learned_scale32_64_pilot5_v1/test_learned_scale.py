"""Focused synthetic checks for the learned binary E8M0 scale-offset route.

No SVHN data or research accuracy is used. The hard forward and the chosen
surrogate backward are tested separately; finite differences of a hard step
are deliberately not a test for an STE.
"""
from __future__ import annotations
import copy
import math
from contextlib import contextmanager
import torch
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low
import trunc_adapter as a
import svhn_support as s
import resume_core as core
from test_joint import backend_state


def rejected(fn):
    try:
        fn()
    except (AssertionError, ValueError, RuntimeError, TypeError, KeyError):
        return
    raise AssertionError('Expected rejection did not occur.')


def string_code(code, bit):
    word = format(code, '08b')
    return code if bit == 8 else int(word[:3] + (word[5:] if bit == 6 else word[5]), 2)


@contextmanager
def forbidden_encoders(include_high=False, include_preprocess=False):
    items = [(module, name, getattr(module, name)) for module in (middle, low)
             for name in ('encode_tensor', 'encode_elements', 'fake_quantize_mx')]
    if include_high:
        items += [(high, name, getattr(high, name)) for name in ('encode_tensor', 'encode_elements', 'fake_quantize_mx')]
    if include_preprocess:
        items.append((a, 'effective_reference_weight', a.effective_reference_weight))
    def fail(*args, **kwargs): raise AssertionError('Forbidden encoder/preprocess was called.')
    try:
        for module, name, _ in items: setattr(module, name, fail)
        yield
    finally:
        for module, name, old in items: setattr(module, name, old)


@contextmanager
def count_high_encodings():
    original = high.encode_tensor; calls = [0]
    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)
    try:
        high.encode_tensor = counted
        yield calls
    finally:
        high.encode_tensor = original


def oracle_block_gradient(q, grad):
    """Independent scalar float64 sum over REAL (unpadded) elements."""
    rows = q.shape[0]; cols = math.prod(q.shape[1:]); blocks = (cols + 31) // 32
    qq = q.detach().cpu().double().reshape(rows, cols).tolist()
    gg = grad.detach().cpu().double().reshape(rows, cols).tolist()
    result = [[math.log(2.0) * sum(qq[r][i] * gg[r][i]
              for i in range(b * 32, min((b + 1) * 32, cols)))
              for b in range(blocks)] for r in range(rows)]
    return torch.tensor(result, dtype=torch.float32, device=q.device)


def run_tests(modules, rt, source, log, checks=None):
    checks = {} if checks is None else checks
    before = core.capture_rng({}, rt.device); flags = backend_state()
    try:
        result = _run_tests(modules, rt, source, log, checks)
        checks['status'] = 'PASS'
        return result
    finally:
        core.restore_rng(before, {}, rt.device)
        s.require(backend_state() == flags, 'Zelftest heeft backendbeleid veranderd.')
        checks['RNG_and_backend_restored'] = True


def _run_tests(modules, rt, source, log, checks):
    device = rt.device
    finite = [i for i in range(256) if i not in (127, 255)]
    codes = torch.tensor(finite, dtype=torch.uint8, device=device).reshape(-1, 1, 1).expand(-1, 1, 32).clone()
    mx = high.MXTensor(codes, torch.full((254, 1), 127, dtype=torch.uint8, device=device), (254, 32))
    initial = (mx.codes.clone(), mx.scale_codes.clone())
    for offset in (5, 6):
        offsets = torch.full((254, 1), offset, dtype=torch.int32, device=device)
        with forbidden_encoders(): chain = a.code_chain(mx, offsets)
        for bit in (6, 4):
            expected = torch.tensor([string_code(i, bit) for i in finite], dtype=torch.uint8, device=device)
            assert torch.equal(chain[bit].codes[:, 0, 0], expected)
            assert torch.equal(chain[bit].scale_codes.to(torch.int32), mx.scale_codes.to(torch.int32) + offset)
            naive = a.exponent_truncate(mx)
            if bit == 4: naive = a.mantissa_truncate(naive)
            assert torch.equal(chain[bit].decode(), naive.decode() * (2 ** offset))
            assert torch.equal(torch.signbit(chain[bit].decode()), torch.signbit(naive.decode()))
        assert chain[8] is mx and torch.equal(chain[4].codes, chain[6].codes >> 2)
        assert torch.equal(chain[4].scale_codes, chain[6].scale_codes)
    assert torch.equal(mx.codes, initial[0]) and torch.equal(mx.scale_codes, initial[1])
    mixed = torch.tensor([[5 + i % 2] for i in range(254)], dtype=torch.int32, device=device)
    chain = a.code_chain(mx, mixed)
    assert torch.equal(chain[6].scale_codes.to(torch.int32), mx.scale_codes.to(torch.int32) + mixed)
    checks['all254_finite_codes_both_offsets_and_mixed_choices_match_independent_bitstrings'] = True
    checks['signed_zero_preserved_source_unchanged_and_mantissa_scale_shared'] = True

    for offset in (5, 6):
        n = 255 - offset
        valid = high.MXTensor(torch.zeros(n, 1, 32, dtype=torch.uint8, device=device),
                             torch.arange(n, dtype=torch.uint8, device=device).reshape(n, 1), (n, 32))
        offsets = torch.full((n, 1), offset, dtype=torch.int32, device=device)
        for value in a.code_chain(valid, offsets).values(): assert not bool(value.decode().any())
        for code in range(n, 255):
            zero = high.MXTensor(torch.zeros(1, 1, 32, dtype=torch.uint8, device=device),
                torch.tensor([[code]], dtype=torch.uint8, device=device), (1, 32))
            rejected(lambda zero=zero: a.code_chain(zero, offsets[:1]))
        huge = high.MXTensor(torch.full((1, 1, 32), 64, dtype=torch.uint8, device=device),
                torch.tensor([[254 - offset]], dtype=torch.uint8, device=device), (1, 32))
        assert bool(torch.isfinite(huge.decode()).all())
        for bit in (6, 4): rejected(lambda bit=bit: a.code_chain(huge, offsets[:1])[bit].decode())
    for code in (127, 255):
        bad = mx.codes.clone(); bad[0, 0, 0] = code
        rejected(lambda bad=bad: a.code_chain(high.MXTensor(bad, mx.scale_codes, mx.shape), mixed))
    rejected(lambda: a.code_chain(high.MXTensor(mx.codes, torch.full_like(mx.scale_codes, 255), mx.shape), mixed))
    for invalid in [torch.full_like(mixed, 4), torch.full_like(mixed, 7), mixed.float(), mixed[:1]]:
        rejected(lambda invalid=invalid: a.code_chain(mx, invalid))
    checks['all_valid_E8M0_source_scales_checked_and_overflow_rejected_even_zero_blocks'] = True
    checks['NaNs_invalid_offsets_and_finite_E8M0_but_overflowing_FP32_decode_rejected'] = True

    edge = torch.tensor([[5., 5.5, 6.]], dtype=torch.float32, device=device)
    edge[0, 0] = torch.nextafter(edge[0, 1], torch.tensor(5., device=device))
    assert torch.equal(a.hard_offsets(edge), torch.tensor([[5, 6, 6]], dtype=torch.int32, device=device))
    for value in (4.99, 6.01, float('nan'), float('inf')):
        rejected(lambda value=value: a.hard_offsets(torch.tensor([[value]], dtype=torch.float32, device=device)))
    rejected(lambda: a.hard_offsets(edge.double()))
    offsets = torch.tensor([[5, 6, 6, 5, 6, 5, 5, 6, 5], [6, 6, 5, 6, 5, 6, 6, 5, 6]], dtype=torch.int32, device=device)
    packed = a.pack_scale_choices(offsets)
    expected_bytes = [0, 0, 0]
    for j, value in enumerate(offsets.cpu().flatten().tolist()): expected_bytes[j // 8] |= (value - 5) << (j % 8)
    assert packed['packed_choices'].tolist() == expected_bytes
    assert torch.equal(a.unpack_scale_choices(packed, device), offsets)
    for key, value in [('schema', 'wrong'), ('num_blocks', 17), ('bit_order', 'big'), ('mapping', {'0': 6, '1': 5}), ('mapping', {'0': 5.0, '1': 6}),
                       ('packed_choices', packed['packed_choices'].float()), ('packed_choices', packed['packed_choices'][:-1])]:
        bad = copy.deepcopy(packed); bad[key] = value
        rejected(lambda bad=bad: a.unpack_scale_choices(bad))
    bad = copy.deepcopy(packed); bad['packed_choices'][-1] |= 128
    rejected(lambda: a.unpack_scale_choices(bad))
    bad = copy.deepcopy(packed); bad['theta'] = torch.ones(2, 9)
    rejected(lambda: a.unpack_scale_choices(bad))
    checks['threshold_tie_to6_and_exact_little_endian_one_bit_choices_with_padding_guards'] = True

    # One zero output row, two partial tail blocks, mixed choices, both signs.
    weight = torch.linspace(-2., 2., 66, device=device).reshape(2, 33)
    weight[1].zero_()
    theta = torch.nn.Parameter(torch.tensor([[5.25, 5.75], [5.1, 6.]], device=device))
    grad = torch.linspace(-.7, .9, 66, device=device).reshape_as(weight)
    chain = a.encode_chain(weight, a.hard_offsets(theta))
    expected_sum = torch.zeros_like(theta)
    for bit in (8, 6, 4):
        effective = weight.detach().clone().requires_grad_()
        theta.grad = None
        value = a._CachedCodeSTE.apply(effective, theta, chain[bit].codes, chain[bit].scale_codes, bit)
        assert torch.equal(value, chain[bit].decode())
        (value * grad).sum().backward()
        assert torch.equal(effective.grad, grad)
        if bit == 8: assert theta.grad is None
        else:
            oracle = oracle_block_gradient(value, grad)
            torch.testing.assert_close(theta.grad, oracle, rtol=2e-6, atol=2e-7)
            assert not bool(theta.grad[1].any())
            expected_sum += oracle
        assert bool((chain[bit].codes[:, -1, 1:] == 0).all())
    theta.grad = None
    for bit in (6, 4):
        effective = weight.detach().clone().requires_grad_()
        value = a._CachedCodeSTE.apply(effective, theta, chain[bit].codes, chain[bit].scale_codes, bit)
        (value * grad).sum().backward()
    torch.testing.assert_close(theta.grad, expected_sum, rtol=2e-6, atol=3e-7)
    checks['hard_forward_exact_weight_gradient_identity_scale_gradient_independent_real_element_oracle'] = True
    checks['E4_direct_theta_gradient_None_lower_gradients_add_zero_and_tail_blocks_correct'] = True

    # Actual G(W) derivatives are retained, separately from the added theta route.
    for bit in (8, 6, 4):
        master = torch.linspace(-2., 2., 66, device=device).reshape(2, 33).requires_grad_()
        reference = master.detach().clone().requires_grad_()
        quantizer = a.JointMXQuantizer(tuple(master.shape), device); quantizer.weight_mode = bit
        with torch.no_grad(): quantizer.theta[0, 0] = 5.25
        with forbidden_encoders(): value = quantizer(master)
        (value * grad).sum().backward()
        (a.effective_reference_weight(reference) * grad).sum().backward()
        assert torch.equal(master.grad, reference.grad)
    checks['original_G_derivatives_retained_no_scale_factor_in_master_gradient'] = True

    # Tiny isolated optimizer proof: a legitimate hard decision crosses 5.5.
    proxy = torch.nn.Parameter(torch.tensor([[5.5001]], device=device))
    effective = torch.ones(1, 32, device=device, requires_grad=True)
    small_source = high.encode_tensor(effective.detach())
    optimizer = torch.optim.Adam([proxy], lr=.001)
    before_offset = a.hard_offsets(proxy).clone()
    initial_chain = a.code_chain(small_source, before_offset)
    q = a._CachedCodeSTE.apply(effective, proxy, initial_chain[6].codes, initial_chain[6].scale_codes, 6)
    q.sum().backward(); optimizer.step()
    with torch.no_grad(): proxy.clamp_(5., 6.)
    after_offset = a.hard_offsets(proxy)
    assert int(before_offset.item()) == 6 and int(after_offset.item()) == 5
    assert torch.equal(a.code_chain(small_source, after_offset)[6].decode(), initial_chain[6].decode() * .5)
    assert optimizer.state[proxy]['exp_avg'].dtype == optimizer.state[proxy]['exp_avg_sq'].dtype == torch.float32
    checks['real_Adam_proxy_update_can_cross_hard_threshold_without_extra_forward_choice_search'] = True

    original = modules.models.svhnq(s.BITS, num_classes=10, expand=2).to(device)
    rng_before = torch.get_rng_state().clone()
    model = a.adapt_copy(original)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert s.legacy.state_digest(original.state_dict()) == s.legacy.state_digest(a.base_state_dict(model))
    assert sum(p.numel() for p in model.parameters()) == 137034
    assert sum(p.numel() for p in a.scale_parameters(model)) == 4080
    assert all(bool((p == 6).all()) for p in a.scale_parameters(model))
    assert a.validate_scale_parameters(model)['choice64_blocks'] == 4080
    with torch.no_grad():
        a.scale_parameters(model)[0].flatten()[0] = 4.8
        a.scale_parameters(model)[-1].flatten()[-1] = 6.2
    projected = a.project_scales_(model)
    assert projected['projected_values'] == 2
    assert a.validate_scale_parameters(model)['all_fp32_finite_in_range']
    checks['4080_FP32_parameters_added_without_random_draw_or_original_state_change'] = True
    checks['post_optimizer_projection_has_exact_counts_and_restores_valid_range'] = True

    # Three real model passes, with two detached teachers and one Adam update.
    x = torch.linspace(-1., 1., 2 * 3 * 40 * 40, device=device).reshape(2, 3, 40, 40)
    labels = torch.tensor([2, 7], dtype=torch.int64, device=device)
    model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    previous = None
    with forbidden_encoders(), count_high_encodings() as count:
        with a.shared_source_triplets(model) as audit:
            for bit in (8, 6, 4):
                a.select_mode(model, bit); output = model(x)
                if bit == 8:
                    loss = torch.nn.functional.cross_entropy(output, labels)
                else:
                    assert previous is not None and not previous.requires_grad
                    loss = modules.losses.CrossEntropyLossSoft()(output, previous)
                previous = torch.nn.functional.softmax(output.detach(), dim=1)
                loss.backward()
                if bit == 8: assert all(p.grad is None for p in a.scale_parameters(model))
    assert count[0] == 6 and audit['source_triplets'] == 1 and audit['scale_offsets_learned'] == 12
    assert audit['source_cache_cleared'] and a.cache_is_empty(model)
    assert all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
               for p in a.scale_parameters(model))
    theta_before = a.scale_snapshot(model)
    optimizer.step(); a.project_scales_(model)
    theta_after = a.scale_snapshot(model)
    assert any(not torch.equal(theta_before[n], theta_after[n]) for n in a.TARGET_LAYERS)
    assert all(int(optimizer.state[p]['step']) == 1 for p in a.scale_parameters(model))
    checks['three_real_model_passes_one_Adam_update_theta_learns_detached_teachers_cache_survives_backward'] = True

    # Mixed decisions are deliberately imposed to test inference storage, not accuracy.
    with torch.no_grad():
        for theta_value in a.scale_parameters(model):
            theta_value.flatten()[::2] = 5.25; theta_value.flatten()[1::2] = 5.75
    model.eval()
    state_before = s.legacy.state_digest(model.state_dict())
    with torch.no_grad(), forbidden_encoders(), count_high_encodings() as count:
        with a.shared_source_triplets(model) as audit:
            for bit in (4, 6, 8): a.select_mode(model, bit); model(x)
    assert count[0] == 6 and audit['source_cache_cleared']
    assert s.legacy.state_digest(model.state_dict()) == state_before
    checks['eval_shared_triplet_preserves_theta_weights_and_BN'] = True
    for failure in ('wrong_order', 'incomplete_cycle', 'theta_mutation', 'code_mutation'):
        probe = copy.deepcopy(model)
        def bad_cycle():
            with torch.no_grad(), a.shared_source_triplets(probe):
                a.select_mode(probe, 6 if failure == 'wrong_order' else 4); probe(x)
                if failure == 'incomplete_cycle': return
                if failure == 'theta_mutation': probe.get_submodule(a.TARGET_LAYERS[0]).quantize_fn.theta.add_(.01)
                if failure == 'code_mutation': probe.get_submodule(a.TARGET_LAYERS[0]).quantize_fn._chain[8].codes[0, 0, 0] ^= 1
                a.select_mode(probe, 6); probe(x)
        rejected(bad_cycle)
        assert a.cache_is_empty(probe)
    checks['wrong_order_incomplete_cycle_theta_or_code_mutation_rejected_and_cache_cleaned'] = True

    source_payload, rows = a.source_payload(model, {'synthetic_fixture_only': True, 'purpose': 'code and gradient self-check'})
    assert len(rows) == 18
    assert sum(v['packed_choices'].numel() for v in source_payload['scale_choices'].values()) == 510
    assert not (set(source_payload['auxiliary_fp32_state']) & a.theta_state_keys())
    assert not any(n + '.weight' in source_payload['auxiliary_fp32_state'] for n in a.TARGET_LAYERS)
    live_values = {}
    with torch.no_grad():
        for bit in (8, 6, 4): a.select_mode(model, bit); live_values[bit] = model(x)
    with torch.no_grad(), forbidden_encoders(include_high=True, include_preprocess=True):
        for bit in (8, 6, 4):
            payload = source_payload if bit == 8 else a.derive_payload(source_payload, bit, '0' * 64)
            frozen = a.load_exported_model(modules.models, payload, device)
            assert not any(n.endswith('.theta') for n, _ in frozen.named_parameters())
            out = frozen(x)
            torch.testing.assert_close(out, live_values[bit], rtol=1e-4, atol=1e-5)
            direct = a.load_exported_model(modules.models, source_payload, device, bit=bit)
            torch.testing.assert_close(direct(x), out, rtol=1e-4, atol=1e-5)
            for name in a.TARGET_LAYERS:
                expected = a.codec(bit).MXTensor.from_payload(payload['mx_layers'][name], device).decode()
                assert torch.equal(frozen.get_submodule(name).decoded_weight, expected)
    for mutation in ('theta', 'master', 'unknown_auxiliary', 'unknown_field', 'old_schema', 'wrong_choice_shape', 'nonfinite_provenance', 'invalid_integer_metadata'):
        bad = copy.deepcopy(source_payload)
        if mutation == 'theta': bad['auxiliary_fp32_state'][a.TARGET_LAYERS[0] + '.quantize_fn.theta'] = torch.ones(1)
        elif mutation == 'master': bad['auxiliary_fp32_state'][a.TARGET_LAYERS[0] + '.weight'] = torch.ones(1)
        elif mutation == 'unknown_auxiliary': bad['auxiliary_fp32_state']['secret_state'] = torch.ones(4080)
        elif mutation == 'unknown_field': bad['theta'] = torch.ones(1)
        elif mutation == 'nonfinite_provenance': bad['provenance']['unknown'] = float('nan')
        elif mutation == 'invalid_integer_metadata': bad['expand'] = 2.0
        elif mutation == 'old_schema': bad['schema'] = 'mx-triple-scale32-training-source-v1'
        else:
            name = a.TARGET_LAYERS[0]
            bad['scale_choices'][name]['shape'] = [1, bad['scale_choices'][name]['num_blocks']]
        rejected(lambda bad=bad: a.validate_source(bad))
    checks['510_packed_choice_bytes_no_master_or_theta_leakage_and_source_only_reconstruction'] = True
    checks['source_and_derived_all_modes_reconstruct_live_logits_without_encoders_or_G'] = True
    checks['malformed_choice_export_old_schema_and_hidden_training_state_rejected'] = True
    log('LEARNED SCALE CHECK PASS: harde x32/x64-forward, identity-mastergradient, expliciete blokgradient, cache, Adam en compacte bronexports.')
    return checks
