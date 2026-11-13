"""Synthetic checks for independent absolute E2 E8M0 group scales.

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
    for exponent in (-127, -20, -1, 0, 5, 125):
        scales = torch.full((254, 1), exponent + 127, dtype=torch.uint8, device=device)
        with forbidden_encoders(): chain = a.code_chain(mx, scales)
        for bit in (6, 4):
            expected = torch.tensor([string_code(i, bit) for i in finite], dtype=torch.uint8, device=device)
            assert torch.equal(chain[bit].codes[:, 0, 0], expected)
            assert torch.equal(chain[bit].scale_codes, scales)
            # Independent element decoder in float64, then the specified FP32 execution.
            expected_q = torch.ldexp(a.codec(bit).decode_elements(chain[bit].codes),
                                      torch.full_like(chain[bit].codes, exponent, dtype=torch.int32)).float()
            actual = chain[bit].decode().reshape_as(expected_q)
            assert torch.equal(actual, expected_q)
            assert torch.equal(torch.signbit(actual), torch.signbit(expected_q))
        assert chain[8] is mx and torch.equal(chain[4].codes, chain[6].codes >> 2)
        assert torch.equal(chain[4].scale_codes, chain[6].scale_codes)
    assert torch.equal(mx.codes, initial[0]) and torch.equal(mx.scale_codes, initial[1])
    checks['all254_finite_source_codes_match_independent_bitstrings_at_absolute_scales'] = True
    checks['signed_zero_source_preserved_and_E2M1_shares_E2M3_scale'] = True

    # All finite E8M0 codes are permitted, independent of any source exponent.
    zero = high.MXTensor(torch.zeros(255, 1, 32, dtype=torch.uint8, device=device),
                         torch.arange(255, dtype=torch.uint8, device=device).reshape(255, 1), (255, 32))
    absolute = torch.arange(254, -1, -1, device=device).to(torch.uint8).reshape(255, 1)
    for value in a.code_chain(zero, absolute).values(): assert not bool(value.decode().any())
    for bit, safe_codes, unsafe_codes in ((6, (31, 16, 8), (24, 16)), (4, (7, 4, 2), (6, 4))):
        for e, c in zip((125, 126, 127), safe_codes):
            value = a.codec(bit).MXTensor(torch.full((1, 1, 32), c, dtype=torch.uint8, device=device),
                torch.tensor([[e+127]], dtype=torch.uint8, device=device), (1, 32))
            assert bool(torch.isfinite(value.decode()).all())
        for e, c in zip((126, 127), unsafe_codes):
            value = a.codec(bit).MXTensor(torch.full((1, 1, 32), c, dtype=torch.uint8, device=device),
                torch.tensor([[e+127]], dtype=torch.uint8, device=device), (1, 32))
            value.validate()  # Valid E8M0 and element fields, invalid FP32 PRODUCT.
            rejected(value.decode)
    for code in (127, 255):
        bad = mx.codes.clone(); bad[0, 0, 0] = code
        rejected(lambda bad=bad: a.code_chain(high.MXTensor(bad, mx.scale_codes, mx.shape), mx.scale_codes))
    rejected(lambda: a.code_chain(high.MXTensor(mx.codes, torch.full_like(mx.scale_codes, 255), mx.shape), mx.scale_codes))
    for invalid in (torch.full_like(mx.scale_codes, 255), mx.scale_codes.float(), mx.scale_codes.int(), mx.scale_codes[:1]):
        rejected(lambda invalid=invalid: a.code_chain(mx, invalid))
    checks['all255_finite_E8M0_codes_allowed_independently_of_source_scale'] = True
    checks['e126_e127_validity_depends_on_element_FP32_product_and_overflow_fails_fast'] = True
    checks['invalid_source_or_lower_codes_shapes_and_dtypes_rejected'] = True

    integers = torch.arange(-127, 128, dtype=torch.float32, device=device).reshape(1, -1)
    assert torch.equal(a.hard_exponents(integers), integers.to(torch.int32))
    assert torch.equal(a.scale_codes(integers).int(), integers.int()+127)
    ties = torch.arange(-127, 127, dtype=torch.float32, device=device) + .5
    below = torch.nextafter(ties, torch.full_like(ties, -float('inf')))
    above = torch.nextafter(ties, torch.full_like(ties, float('inf')))
    edge = torch.stack((below, ties, above), dim=1)
    expected = torch.tensor([[i, i+1, i+1] for i in range(-127, 127)], dtype=torch.int32, device=device)
    assert torch.equal(a.hard_exponents(edge), expected)
    for value in (-127.01, 127.01, float('nan'), float('inf')):
        rejected(lambda value=value: a.hard_exponents(torch.tensor([[value]], dtype=torch.float32, device=device)))
    for bad in (edge.double(), integers.flatten(), torch.empty((1,0), device=device)):
        rejected(lambda bad=bad: a.hard_exponents(bad))
    checks['all_positive_negative_half_ties_and_nextafter_neighbors_round_half_up_exactly'] = True

    # An unchanged phi does not follow a moving E4 source scale.
    effective = torch.linspace(.25, .5, 64, device=device).reshape(2,32)
    source1, source2 = high.encode_tensor(effective), high.encode_tensor(effective * 2)
    assert torch.equal(source1.codes, source2.codes)
    assert torch.equal(source2.scale_codes.int(), source1.scale_codes.int()+1)
    phi = torch.nn.Parameter((source1.scale_codes.int()-127+6).float())
    scales = a.scale_codes(phi)
    chain1, chain2 = a.code_chain(source1, scales), a.code_chain(source2, scales)
    assert torch.equal(chain2[8].decode(), chain1[8].decode()*2)
    for bit in (6,4):
        assert torch.equal(chain2[bit].scale_codes, scales)
        assert torch.equal(chain1[bit].decode(), chain2[bit].decode())
    chain3 = a.code_chain(source1, a.scale_codes(phi+1))
    for bit in (6,4):
        assert torch.equal(chain3[bit].codes, chain1[bit].codes)
        assert torch.equal(chain3[bit].decode(), chain1[bit].decode()*2)
    assert torch.equal(chain3[8].codes, chain1[8].codes) and torch.equal(chain3[8].scale_codes, chain1[8].scale_codes)
    checks['fixed_phi_does_not_follow_E4_scale_and_phi_plus1_doubles_only_lower_weights'] = True

    # Zero output row, partial tail blocks, mixed absolute scales and both signs.
    weight = torch.linspace(-2., 2., 66, device=device).reshape(2, 33); weight[1].zero_()
    phi = torch.nn.Parameter(torch.tensor([[-2.25, -1.75], [-4.75, -3.]], device=device))
    grad = torch.linspace(-.7, .9, 66, device=device).reshape_as(weight)
    chain = a.encode_chain(weight, a.scale_codes(phi)); expected_sum = torch.zeros_like(phi)
    for bit in (8,6,4):
        effective = weight.detach().clone().requires_grad_(); phi.grad = None
        value = a._CachedCodeSTE.apply(effective, phi, chain[bit].codes, chain[bit].scale_codes, bit)
        assert torch.equal(value, chain[bit].decode())
        (value * grad).sum().backward(); assert torch.equal(effective.grad, grad)
        if bit == 8: assert phi.grad is None
        else:
            oracle = oracle_block_gradient(value, grad)
            torch.testing.assert_close(phi.grad, oracle, rtol=2e-6, atol=2e-7)
            assert not bool(phi.grad[1].any()); expected_sum += oracle
        assert bool((chain[bit].codes[:,-1,1:] == 0).all())
    phi.grad = None
    for bit in (6,4):
        effective = weight.detach().clone().requires_grad_()
        value = a._CachedCodeSTE.apply(effective, phi, chain[bit].codes, chain[bit].scale_codes, bit)
        (value * grad).sum().backward()
    torch.testing.assert_close(phi.grad, expected_sum, rtol=2e-6, atol=3e-7)
    checks['analytic_surrogate_matches_real_element_oracle_and_lower_gradients_sum'] = True
    checks['master_identity_E4_phi_gradient_None_zero_groups_and_padding_correct'] = True

    # Endpoint gradients remain unmasked; chosen finite products avoid overflow.
    for exponent in (-127, -12, 0, 125, 126, 127):
        proxy = torch.nn.Parameter(torch.tensor([[float(exponent)]], device=device))
        effective = torch.ones(1,32,device=device,requires_grad=True)
        cotangent = torch.full_like(effective, math.ldexp(1., -exponent))
        for bit, code in ((6,8),(4,2)):
            proxy.grad = effective.grad = None
            value = a._CachedCodeSTE.apply(effective, proxy,
                torch.full((1,1,32),code,dtype=torch.uint8,device=device),a.scale_codes(proxy),bit)
            (value*cotangent).sum().backward()
            torch.testing.assert_close(proxy.grad, oracle_block_gradient(value,cotangent),rtol=2e-6,atol=2e-7)
            assert bool((proxy.grad != 0).all()) and torch.equal(effective.grad,cotangent)
    checks['absolute_scale_surrogate_retained_at_finite_E8M0_endpoints'] = True

    for bit in (8,6,4):
        master = torch.linspace(-2.,2.,66,device=device).reshape(2,33).requires_grad_()
        reference = master.detach().clone().requires_grad_()
        quantizer = a.JointMXQuantizer(master); quantizer.weight_mode = bit
        with torch.no_grad(): quantizer.phi[0,0].add_(.25)
        with forbidden_encoders(): value = quantizer(master)
        (value*grad).sum().backward(); (a.effective_reference_weight(reference)*grad).sum().backward()
        assert torch.equal(master.grad,reference.grad)
    checks['original_G_derivatives_retained_without_scale_multiplier_in_master_gradient'] = True

    # A real Adam update crosses a negative half-integer decision.
    proxy = torch.nn.Parameter(torch.tensor([[-2.4999]],device=device))
    effective = torch.ones(1,32,device=device,requires_grad=True)
    small_source=high.encode_tensor(effective.detach()); optimizer=torch.optim.Adam([proxy],lr=.001)
    initial_chain=a.code_chain(small_source,a.scale_codes(proxy))
    q=a._CachedCodeSTE.apply(effective,proxy,initial_chain[6].codes,initial_chain[6].scale_codes,6)
    q.sum().backward(); optimizer.step()
    assert int(a.hard_exponents(proxy)) == -3
    assert torch.equal(a.code_chain(small_source,a.scale_codes(proxy))[6].decode(),initial_chain[6].decode()*.5)
    checks['real_Adam_update_crosses_negative_exponent_threshold'] = True

    original=modules.models.svhnq(s.BITS,num_classes=10,expand=2).to(device)
    rng_before=core.capture_rng({},device); model=a.adapt_copy(original)
    assert not core.differences(rng_before,core.capture_rng({},device))
    assert s.legacy.state_digest(original.state_dict())==s.legacy.state_digest(a.base_state_dict(model))
    assert sum(p.numel() for p in model.parameters())==137034
    assert sum(p.numel() for p in a.scale_parameters(model))==4080
    for name in a.TARGET_LAYERS:
        layer=model.get_submodule(name); effective=a.effective_reference_weight(layer.weight.detach())
        src=high.encode_tensor(effective); phi=layer.quantize_fn.phi
        assert torch.equal(phi, (src.scale_codes.int()-127+6).float())
        chain=a.encode_chain(effective,a.scale_codes(phi))
        for bit in (6,4):
            naive=a.exponent_truncate(src)
            if bit==4: naive=a.mantissa_truncate(naive)
            assert torch.equal(chain[bit].decode(),naive.decode()*64)
    checks['4080_phi_parameters_initialized_once_at_exact_fixed64_without_RNG_or_base_state_changes'] = True

    boundary_model=copy.deepcopy(model)
    for bound, gradient, projected in ((-127.,1.,4080),(-127.,-1.,0),(127.,-1.,4080),(127.,1.,0)):
        with torch.no_grad():
            for p in a.scale_parameters(boundary_model): p.fill_(bound)
        opt=torch.optim.Adam(a.scale_parameters(boundary_model),lr=.001)
        for p in a.scale_parameters(boundary_model): p.grad=torch.full_like(p,gradient)
        opt.step(); projection=a.project_scales_(boundary_model)
        assert projection['projected_values']==projected
        assert a.validate_scale_parameters(boundary_model)['all_fp32_finite_in_range']
        if projected: assert all(bool((p==bound).all()) for p in a.scale_parameters(boundary_model))
        else: assert all(bool((p!=bound).all()) for p in a.scale_parameters(boundary_model))
    checks['Adam_projection_only_uses_E8M0_bounds_and_both_endpoints_can_move_inward'] = True

    audit_model=copy.deepcopy(model)
    with torch.no_grad():
        for p in a.scale_parameters(audit_model): p.fill_(-6.)
    diagnostic=training.ScaleEpochAudit(audit_model,1,.01)
    name0=a.TARGET_LAYERS[0]; tracked=audit_model.get_submodule(name0).quantize_fn.phi; steps=[]
    for step in range(2):
        for p in a.scale_parameters(audit_model): p.grad=torch.ones_like(p)
        diagnostic.before_step()
        with torch.no_grad():
            if step==0: tracked.flatten()[:9]=torch.tensor([-127.,-126.75,-6.5,-6.25,-.5,0.,.5,126.75,127.],device=device)
            else: tracked.flatten()[0]=127.; tracked.flatten()[8]=-127.
        audit_model._last_scale_projection=a.project_scales_(audit_model); diagnostic.after_step()
        steps.append([math.floor(float(v)+.5) for v in tracked.detach().cpu().flatten().tolist()])
    row=next(r for r in diagnostic.rows() if r['layer']==name0)
    previous=[-6]*tracked.numel(); flips=distance=max_jump=0; switched=set()
    for current in steps:
        for i,(old,new) in enumerate(zip(previous,current)):
            jump=abs(new-old); flips+=int(jump!=0); distance+=jump; max_jump=max(max_jump,jump)
            if jump: switched.add(i)
        previous=current
    assert row['exponent_flips_during_epoch']==flips
    assert row['unique_blocks_switched_during_epoch']==len(switched)
    assert row['exponent_jump_distance_during_epoch']==distance and row['max_exponent_jump']==max_jump==254
    assert row['net_exponent_changes']==sum(v!=-6 for v in steps[-1])
    for exponent in range(-127,128): assert row['e2_exponent_counts'][str(exponent)]==steps[-1].count(exponent)
    assert row['phi_at_min_blocks']==row['phi_at_max_blocks']==1
    assert row['e2_exponent_counts']['-127']==row['e2_exponent_counts']['127']==2
    actual_source=high.encode_tensor(a.effective_reference_weight(audit_model.get_submodule(name0).weight.detach())).scale_codes.cpu().int()-127
    expected_relative=a.hard_exponents(tracked.detach().cpu())-actual_source
    assert row['observed_relative_offset_counts']==a.observed_offset_histogram(expected_relative)
    assert sum(a.validate_scale_parameters(audit_model)['e2_exponent_counts'].values())==4080
    checks['absolute_exponent_histograms_flips_bounds_and_observed_relative_diagnostics_correct'] = True

    # Three real model passes, detached recursive teachers, one production Adam step.
    x=torch.linspace(-1.,1.,2*3*40*40,device=device).reshape(2,3,40,40)
    labels=torch.tensor([2,7],dtype=torch.int64,device=device)
    model.train(); optimizer=training.build_optimizer(modules,model,{'lr':.001,'scale_lr':.001,'weight_decay':0.})
    optimizer.zero_grad(set_to_none=True); previous=None
    with forbidden_encoders(),count_high_encodings() as count:
        with a.shared_source_triplets(model) as audit:
            for bit in (8,6,4):
                a.select_mode(model,bit); output=model(x)
                if bit==8: loss=torch.nn.functional.cross_entropy(output,labels)
                else:
                    assert previous is not None and not previous.requires_grad
                    loss=modules.losses.CrossEntropyLossSoft()(output,previous)
                previous=torch.nn.functional.softmax(output.detach(),dim=1); loss.backward()
                if bit==8: assert all(p.grad is None for p in a.scale_parameters(model))
    assert count[0]==6 and audit['source_triplets']==1 and audit['learned_absolute_scales_applied']==12
    assert audit['source_cache_cleared'] and a.cache_is_empty(model)
    assert all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in a.scale_parameters(model))
    phi_before=a.scale_snapshot(model); optimizer.step(); phi_after=a.scale_snapshot(model)
    assert any(not torch.equal(phi_before[n],phi_after[n]) for n in a.TARGET_LAYERS)
    assert all(int(optimizer.state[p]['step'])==1 for p in a.scale_parameters(model))
    checks['three_passes_one_production_Adam_update_phi_learns_and_cache_survives_backward'] = True

    # State loading restores absolute phi/Adam, not a fresh E4+6 initialization.
    restored=a.adapt_copy(original)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored_opt=training.build_optimizer(modules,restored,{'lr':.001,'scale_lr':.001,'weight_decay':0.})
    restored_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert not core.differences(model.state_dict(),restored.state_dict())
    assert not core.differences(optimizer.state_dict(),restored_opt.state_dict())
    restored_phi=a.scale_snapshot(restored)
    restored.eval()
    with torch.no_grad(): a.select_mode(restored,6); restored(x)
    assert not core.differences(restored_phi,a.scale_snapshot(restored))
    checks['state_restore_preserves_absolute_phi_Adam_and_forward_never_reinitializes'] = True

    # Mixed nearby absolute exponents exercise storage, without an accuracy claim.
    with torch.no_grad():
        for p in a.scale_parameters(model): p.add_(((torch.arange(p.numel(),device=device)%3)-1).reshape_as(p))
    model.eval(); state_before=s.legacy.state_digest(model.state_dict())
    with torch.no_grad(),forbidden_encoders(),count_high_encodings() as count:
        with a.shared_source_triplets(model) as audit:
            for bit in (4,6,8): a.select_mode(model,bit); model(x)
    assert count[0]==6 and audit['source_cache_cleared']
    assert s.legacy.state_digest(model.state_dict())==state_before
    checks['evaluation_preserves_phi_weights_and_BN'] = True
    for failure in ('wrong_order','incomplete_cycle','phi_mutation','code_mutation','scale_mutation'):
        probe=copy.deepcopy(model)
        def bad_cycle():
            with torch.no_grad(),a.shared_source_triplets(probe):
                a.select_mode(probe,6 if failure=='wrong_order' else 4); probe(x)
                if failure=='incomplete_cycle': return
                quantizer=probe.get_submodule(a.TARGET_LAYERS[0]).quantize_fn
                if failure=='phi_mutation': quantizer.phi.add_(.01)
                if failure=='code_mutation': quantizer._chain[8].codes[0,0,0]^=1
                if failure=='scale_mutation': quantizer._scales[0,0]^=1
                a.select_mode(probe,6); probe(x)
        rejected(bad_cycle); assert a.cache_is_empty(probe)
    checks['invalid_cycle_and_phi_code_or_scale_mutation_rejected_and_cache_cleaned'] = True

    payload,rows=a.source_payload(model,{'synthetic_fixture_only':True,'purpose':'code and gradient self-check'})
    assert len(rows)==18 and sum(v.numel() for v in payload['lower_scale_codes'].values())==4080
    assert all(v.dtype==torch.uint8 for v in payload['lower_scale_codes'].values())
    assert not (set(payload['auxiliary_fp32_state']) & a.phi_state_keys())
    assert not any(n+'.weight' in payload['auxiliary_fp32_state'] for n in a.TARGET_LAYERS)
    live={}
    with torch.no_grad():
        for bit in (8,6,4): a.select_mode(model,bit); live[bit]=model(x)
    with torch.no_grad(),forbidden_encoders(include_high=True,include_preprocess=True):
        for bit in (8,6,4):
            derived=payload if bit==8 else a.derive_payload(payload,bit,'0'*64)
            frozen=a.load_exported_model(modules.models,derived,device)
            assert not any(n.endswith('.phi') for n,_ in frozen.named_parameters())
            output=frozen(x); torch.testing.assert_close(output,live[bit],rtol=1e-4,atol=1e-5)
            direct=a.load_exported_model(modules.models,payload,device,bit=bit)
            torch.testing.assert_close(direct(x),output,rtol=1e-4,atol=1e-5)
            for name in a.TARGET_LAYERS:
                expected=a.codec(bit).MXTensor.from_payload(derived['mx_layers'][name],device).decode()
                assert torch.equal(frozen.get_submodule(name).decoded_weight,expected)
    for mutation in ('phi','theta','master','unknown_auxiliary','unknown_field','old_schema','wrong_scale_shape',
                     'wrong_scale_dtype','scale255','nonfinite_provenance','invalid_integer_metadata'):
        bad=copy.deepcopy(payload); name=a.TARGET_LAYERS[0]
        if mutation in ('phi','theta'): bad['auxiliary_fp32_state'][name+'.quantize_fn.'+mutation]=torch.ones(1)
        elif mutation=='master': bad['auxiliary_fp32_state'][name+'.weight']=torch.ones(1)
        elif mutation=='unknown_auxiliary': bad['auxiliary_fp32_state']['secret_state']=torch.ones(4080)
        elif mutation=='unknown_field': bad['phi']=torch.ones(1)
        elif mutation=='nonfinite_provenance': bad['provenance']['unknown']=float('nan')
        elif mutation=='invalid_integer_metadata': bad['expand']=2.0
        elif mutation=='old_schema': bad['schema']='mx-triple-learned-scale10-training-source-v1'
        elif mutation=='wrong_scale_shape': bad['lower_scale_codes'][name]=bad['lower_scale_codes'][name].reshape(1,-1)
        elif mutation=='wrong_scale_dtype': bad['lower_scale_codes'][name]=bad['lower_scale_codes'][name].float()
        elif mutation=='scale255': bad['lower_scale_codes'][name][0,0]=255
        rejected(lambda bad=bad:a.validate_source(bad))
    wrong=a.derive_payload(payload,6,'0'*64)
    wrong['lower_scale_codes'][a.TARGET_LAYERS[0]][0,0]^=1
    rejected(lambda:a.validate_derived(wrong))
    checks['4080_absolute_E2_scale_bytes_no_master_or_phi_leakage_and_source_only_reconstruction'] = True
    checks['all_modes_reconstruct_logits_without_encoders_or_G_and_reject_old_or_malformed_exports'] = True
    log('INDEPENDENT E2 SCALE CHECK PASS: absolute E8M0, eenmalige x64-init, onafhankelijke bronbeweging, expliciete blokgradient, cache, Adam en 8bit-schaalexports.')
    return checks
