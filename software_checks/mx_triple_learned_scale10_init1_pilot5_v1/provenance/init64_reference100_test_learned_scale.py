"""Focused synthetic checks for the ten-choice E8M0 scale-offset route.

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
import trunc_support as training
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
    for offset in range(10):
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
    mixed = torch.tensor([[i % 10] for i in range(254)], dtype=torch.int32, device=device)
    chain = a.code_chain(mx, mixed)
    assert torch.equal(chain[6].scale_codes.to(torch.int32), mx.scale_codes.to(torch.int32) + mixed)
    checks['all254_finite_codes_all10_offsets_and_mixed_choices_match_independent_bitstrings'] = True
    checks['signed_zero_preserved_source_unchanged_and_mantissa_scale_shared'] = True

    for offset in range(10):
        n = 255 - offset
        valid = high.MXTensor(torch.zeros(n, 1, 32, dtype=torch.uint8, device=device),
                             torch.arange(n, dtype=torch.uint8, device=device).reshape(n, 1), (n, 32))
        offsets = torch.full((n, 1), offset, dtype=torch.int32, device=device)
        for value in a.code_chain(valid, offsets).values(): assert not bool(value.decode().any())
        for code in range(n, 255):
            zero = high.MXTensor(torch.zeros(1, 1, 32, dtype=torch.uint8, device=device),
                torch.tensor([[code]], dtype=torch.uint8, device=device), (1, 32))
            rejected(lambda zero=zero: a.code_chain(zero, offsets[:1]))
        if offset > 0:
            huge = high.MXTensor(torch.full((1, 1, 32), 64, dtype=torch.uint8, device=device),
                    torch.tensor([[254 - offset]], dtype=torch.uint8, device=device), (1, 32))
            assert bool(torch.isfinite(huge.decode()).all())
            for bit in (6, 4): rejected(lambda bit=bit: a.code_chain(huge, offsets[:1])[bit].decode())
    for code in (127, 255):
        bad = mx.codes.clone(); bad[0, 0, 0] = code
        rejected(lambda bad=bad: a.code_chain(high.MXTensor(bad, mx.scale_codes, mx.shape), mixed))
    rejected(lambda: a.code_chain(high.MXTensor(mx.codes, torch.full_like(mx.scale_codes, 255), mx.shape), mixed))
    for invalid in [torch.full_like(mixed, -1), torch.full_like(mixed, 10), mixed.float(), mixed[:1]]:
        rejected(lambda invalid=invalid: a.code_chain(mx, invalid))
    checks['all_valid_E8M0_source_scales_checked_and_overflow_rejected_even_zero_blocks'] = True
    checks['NaNs_invalid_offsets_and_finite_E8M0_but_overflowing_FP32_decode_rejected'] = True

    integers = torch.arange(10, dtype=torch.float32, device=device).reshape(2, 5)
    assert torch.equal(a.hard_offsets(integers), integers.to(torch.int32))
    ties = torch.arange(9, dtype=torch.float32, device=device) + .5
    below = torch.nextafter(ties, torch.full_like(ties, -float('inf')))
    above = torch.nextafter(ties, torch.full_like(ties, float('inf')))
    edge = torch.stack((below, ties, above), dim=1)
    expected_edge = torch.tensor([[i, i + 1, i + 1] for i in range(9)], dtype=torch.int32, device=device)
    assert torch.equal(a.hard_offsets(edge), expected_edge)
    legacy = torch.cat((torch.linspace(5., 6., 101, device=device), edge[5])).reshape(1, -1)
    old_binary = torch.where(legacy < 5.5, 5, 6).to(torch.int32)
    assert torch.equal(a.hard_offsets(legacy), old_binary)
    for value in (-.01, 9.01, float('nan'), float('inf')):
        rejected(lambda value=value: a.hard_offsets(torch.tensor([[value]], dtype=torch.float32, device=device)))
    rejected(lambda: a.hard_offsets(edge.double()))
    rejected(lambda: a.hard_offsets(torch.empty((1, 0), dtype=torch.float32, device=device)))
    rejected(lambda: a.hard_offsets(integers.flatten()))
    offsets = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3],
                            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 6]], dtype=torch.int32, device=device)
    packed = a.pack_scale_choices(offsets)
    expected_bytes = [0] * 11
    for j, value in enumerate(offsets.cpu().flatten().tolist()): expected_bytes[j // 2] |= value << (4 * (j % 2))
    assert packed['packed_choices'].tolist() == expected_bytes
    assert torch.equal(a.unpack_scale_choices(packed, device), offsets)
    odd = a.pack_scale_choices(offsets[:1])
    assert odd['packed_choices'].tolist() == [16, 50, 84, 118, 152, 3]
    assert torch.equal(a.unpack_scale_choices(odd, device), offsets[:1])
    wrong_type_mapping = {str(i): i for i in range(10)}; wrong_type_mapping['0'] = 0.0
    for key, value in [('schema', 'mx-binary-scale-choice-v1'), ('num_blocks', 17), ('bit_order', 'big'),
                       ('bits_per_choice', 3), ('bits_per_choice', 4.0),
                       ('mapping', {'0': 5, '1': 6}), ('mapping', wrong_type_mapping),
                       ('packed_choices', packed['packed_choices'].float()), ('packed_choices', packed['packed_choices'][:-1])]:
        bad = copy.deepcopy(packed); bad[key] = value
        rejected(lambda bad=bad: a.unpack_scale_choices(bad))
    bad = copy.deepcopy(odd); bad['packed_choices'][-1] |= 16
    rejected(lambda: a.unpack_scale_choices(bad))
    for invalid_code in range(10, 16):
        for nibble in (0, 1):
            bad = copy.deepcopy(packed)
            bad['packed_choices'][0] = invalid_code << (4 * nibble)
            rejected(lambda bad=bad: a.unpack_scale_choices(bad))
    bad = copy.deepcopy(packed); bad['theta'] = torch.ones(2, 9)
    rejected(lambda: a.unpack_scale_choices(bad))
    checks['all_half_ties_nextafter_neighbors_exact_half_up_and_legacy_binary_interval_preserved'] = True
    checks['all10_choices_exact_little_endian_four_bit_roundtrip_invalid_nibbles_and_odd_padding_rejected'] = True

    # One zero output row, two partial tail blocks, mixed choices, both signs.
    weight = torch.linspace(-2., 2., 66, device=device).reshape(2, 33)
    weight[1].zero_()
    theta = torch.nn.Parameter(torch.tensor([[.25, 3.75], [8.75, 9.]], device=device))
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

    # Every discrete choice, including both clamp endpoints, retains the same
    # analytic surrogate. This is not a finite difference of the hard step.
    for offset in range(10):
        proxy = torch.nn.Parameter(torch.full((1, 1), float(offset), device=device))
        effective = torch.linspace(-1., 1., 32, device=device).reshape(1, 32).requires_grad_()
        cotangent = torch.linspace(.1, .9, 32, device=device).reshape_as(effective)
        route = a.encode_chain(effective.detach(), a.hard_offsets(proxy))
        for bit in (6, 4):
            proxy.grad = effective.grad = None
            value = a._CachedCodeSTE.apply(effective, proxy, route[bit].codes, route[bit].scale_codes, bit)
            (value * cotangent).sum().backward()
            torch.testing.assert_close(proxy.grad, oracle_block_gradient(value, cotangent), rtol=2e-6, atol=2e-7)
            assert bool((proxy.grad != 0).all()) and torch.equal(effective.grad, cotangent)
    checks['surrogate_scale_gradient_all10_choices_including_unmasked_clamp_endpoints'] = True

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
    with torch.no_grad(): proxy.clamp_(0., 9.)
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
        a.scale_parameters(model)[0].flatten()[0] = -.2
        a.scale_parameters(model)[-1].flatten()[-1] = 9.2
    projected = a.project_scales_(model)
    assert projected['projected_values'] == 2
    assert a.validate_scale_parameters(model)['all_fp32_finite_in_range']
    checks['4080_FP32_parameters_added_without_random_draw_or_original_state_change'] = True
    checks['post_optimizer_projection_has_exact_counts_and_restores_valid_range'] = True

    # Synthetic transitions validate diagnostics without introducing exploration
    # into research training. The same group may switch twice within an epoch.
    audit_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in a.scale_parameters(audit_model): p.fill_(6.)
    diagnostic = training.ScaleEpochAudit(audit_model, 1, .01)
    name0 = a.TARGET_LAYERS[0]
    tracked = audit_model.get_submodule(name0).quantize_fn.theta
    steps = []
    for step in range(2):
        for p in a.scale_parameters(audit_model): p.grad = torch.ones_like(p)
        diagnostic.before_step()
        with torch.no_grad():
            if step == 0:
                tracked.flatten()[:13] = torch.tensor([0., .25, .5, 1.25, 2.25, 3.25, 4.25,
                                                      5.25, 6.25, 7.25, 8.25, 8.75, 9.], device=device)
            else:
                tracked.flatten()[0] = 9.; tracked.flatten()[12] = 0.
        audit_model._last_scale_projection = a.project_scales_(audit_model)
        diagnostic.after_step()
        # Independent Python scalar rounding, not the implementation histogram.
        steps.append([math.floor(float(v) + .5) for v in tracked.detach().cpu().flatten().tolist()])
    row = next(r for r in diagnostic.rows() if r['layer'] == name0)
    previous = [6] * tracked.numel()
    expected_flips = expected_distance = 0; switched = set(); max_jump = 0
    for current in steps:
        for i, (old, new) in enumerate(zip(previous, current)):
            jump = abs(new - old)
            expected_flips += int(jump != 0); expected_distance += jump; max_jump = max(max_jump, jump)
            if jump: switched.add(i)
        previous = current
    assert row['choice_flips_during_epoch'] == expected_flips
    assert row['unique_blocks_switched_during_epoch'] == len(switched)
    assert row['choice_jump_distance_during_epoch'] == expected_distance and row['max_choice_jump'] == max_jump == 9
    assert row['net_choice_changes'] == sum(v != 6 for v in steps[-1])
    for offset in range(10):
        count = steps[-1].count(offset)
        assert row[f'count_x{1 << offset}'] == count
        assert row[f'fraction_x{1 << offset}'] == count / tracked.numel()
    assert sum(row[f'count_x{1 << offset}'] for offset in range(10)) == tracked.numel()
    assert row['choice_min_blocks'] == row['choice_max_blocks'] == 2
    assert row['theta_at_min_blocks'] == row['theta_at_max_blocks'] == 1
    validation = a.validate_scale_parameters(audit_model)
    assert sum(validation['choice_counts'].values()) == sum(validation['factor_counts'].values()) == 4080
    assert validation['choice_min_blocks'] == validation['choice_max_blocks'] == 2
    assert validation['theta_at_min_blocks'] == validation['theta_at_max_blocks'] == 1
    checks['ten_bin_histograms_switch_counts_jumps_and_chosen_endpoints_distinguished_from_clamp'] = True

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
            theta_value.copy_((torch.arange(theta_value.numel(), device=device) % 10).reshape_as(theta_value))
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
    assert sum(v['packed_choices'].numel() for v in source_payload['scale_choices'].values()) == 2040
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
    checks['2040_packed_choice_bytes_no_master_or_theta_leakage_and_source_only_reconstruction'] = True
    checks['source_and_derived_all_modes_reconstruct_live_logits_without_encoders_or_G'] = True
    checks['malformed_choice_export_old_schema_and_hidden_training_state_rejected'] = True

    zero_source = copy.deepcopy(source_payload)
    for name, choice in zero_source['scale_choices'].items():
        zero_source['scale_choices'][name] = a.pack_scale_choices(torch.zeros(choice['shape'], dtype=torch.int32))
    with forbidden_encoders(include_high=True, include_preprocess=True):
        for bit in (6, 4):
            zero_derived = a.derive_payload(zero_source, bit, '0' * 64)
            assert zero_derived['derivation']['scales_unchanged'] is True
            assert zero_derived['derivation']['shared_scale_compensation'] is False
            for name in a.TARGET_LAYERS:
                src = high.MXTensor.from_payload(zero_source['mx_layers'][name])
                target = a.codec(bit).MXTensor.from_payload(zero_derived['mx_layers'][name])
                assert torch.equal(src.scale_codes, target.scale_codes)
    checks['all_x1_source_derivation_preserves_scale_and_reports_actual_no_compensation'] = True
    log('LEARNED SCALE10 CHECK PASS: tien harde keuzes x1..x512, identity-mastergradient, expliciete blokgradient, cache, Adam en 4bit-bronexports.')
    return checks
