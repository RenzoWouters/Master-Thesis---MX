"""Independent structural/gradient controls, on CPU offline or CUDA in check.

Finite training tests are not guarantees about learning accuracy. The baseline
low encoder is explicitly made unavailable for the actual shared-source test.
"""
from __future__ import annotations
import torch
from unittest.mock import patch
import trunc_adapter as a
import trunc_support as t
from test_mx import rejected
import svhn_support as s


def run_tests(modules, rt, source, log):
    d = rt.device
    checks = {}
    # All 64 source codes, independently decoded by sign/exponent/fraction.
    codes = torch.arange(64, device=d, dtype=torch.uint8).reshape(2,1,32)
    source_mx = a.high.MXTensor(codes, torch.full((2,1),127,device=d,dtype=torch.uint8), (2,32))
    old_codes, old_scales = codes.clone(), source_mx.scale_codes.clone()
    q = a.truncate_codes(source_mx)
    expected=[]
    for c in range(64):
        sign=-1. if c & 32 else 1.
        exponent=(c >> 3) & 3
        fraction=(c & 7)//4/2.
        expected.append(sign*(fraction if exponent==0 else (2.**(exponent-1))*(1+fraction)))
    oracle=torch.tensor(expected,device=d,dtype=torch.float32).reshape(2,32)
    assert torch.equal(q.decode(),oracle)
    assert torch.equal(torch.signbit(q.decode()),torch.signbit(oracle))
    assert torch.equal(source_mx.codes,old_codes) and torch.equal(source_mx.scale_codes,old_scales)
    assert torch.equal(q.codes, (source_mx.codes.long()>>2).byte())
    assert torch.equal(a.low.unpack4(a.low.pack4(q.codes)),q.codes)
    checks['all_64_codes_independent_oracle_sign_exponent_scale_signed_zero_pack']=True
    w=torch.zeros(1,33,device=d);w[0,:4]=torch.tensor([7.,1.375,-1.375,.375],device=d)
    h=a.high.encode_tensor(w);c=a.truncate_codes(h)
    assert c.decode()[0,:4].tolist()==[6.,1.,-1.,0.]
    assert float(a.low.encode_tensor(w).decode()[0,1])==1.5
    assert c.codes.shape==(1,2,32) and bool((c.codes[:,-1,1:]==0).all())
    for exponent in [-120,-40,0,40,120]:
        h=a.high.encode_tensor(torch.ldexp(w,torch.tensor(exponent,device=d)))
        c=a.truncate_codes(h)
        assert torch.equal(h.scale_codes,c.scale_codes)
    zeros=a.high.MXTensor(torch.zeros(255,1,32,device=d,dtype=torch.uint8),
                         torch.arange(255,device=d,dtype=torch.uint8).reshape(255,1),(255,32))
    assert not bool(a.truncate_codes(zeros).decode().any())
    checks['truncation_not_direct_rounding_padding_scale_range']=True

    # The low loss alone must reach G(W) and the FP32 master with the declared
    # identity surrogate, including signed, zero, subnormal and clipped cases.
    value=torch.linspace(-8,8,66,device=d).reshape(2,33).requires_grad_()
    c=a.truncate_codes(a.high.encode_tensor(value))
    coef=torch.linspace(-.9,.8,66,device=d).reshape(2,33)
    out=a._CachedCodeSTE.apply(value,c.codes,c.scale_codes,4)
    (out*coef).sum().backward()
    assert torch.equal(value.grad,coef)
    p=torch.linspace(-2,2,66,device=d).reshape(2,33).requires_grad_()
    r=p.detach().clone().requires_grad_()
    q=a.DualMXQuantizer();q.weight_mode=4
    (q(p)*coef).sum().backward()
    (a.effective_reference_weight(r)*coef).sum().backward()
    assert torch.equal(p.grad,r.grad)
    checks['low_loss_identity_STE_reaches_master_no_divide_by_four']=True

    model=t.build_model(modules,'joint',d)
    opt=modules.optimizer.get_optimizer_config(model,'adam',.001,0.)
    x=torch.linspace(-1,1,3*3*40*40,device=d).reshape(3,3,40,40)
    y=torch.tensor([1,4,6],device=d)
    captured={bit:{} for bit in (6,4)}
    hooks=[]
    for name in a.TARGET_LAYERS:
        q=model.get_submodule(name).quantize_fn
        def capture(module,inputs,output,name=name):
            h=module._source
            assert h is not None
            captured[module.weight_mode][name]=(id(h), h.codes.clone(),h.scale_codes.clone(),output.detach().clone())
        hooks.append(q.register_forward_hook(capture))
    model.train()
    original_high=a.high.encode_tensor
    try:
        with patch.object(a.low,'encode_tensor',side_effect=AssertionError('FORBIDDEN direct low encoder')), \
             patch.object(a.low,'encode_elements',side_effect=AssertionError('FORBIDDEN low re-round')), \
             patch.object(a.high,'encode_tensor',wraps=original_high) as high_calls:
            fn,_=t.load_dual_forward(source,modules.utils,log)
            fn([(x,y)],model,torch.nn.CrossEntropyLoss(),modules.losses.CrossEntropyLossSoft(),0,True,opt,None)
            assert high_calls.call_count==6, high_calls.call_count
            audit=model._last_source_audit
            assert audit['source_pairs']==1 and audit['source_cache_cleared']
        for n in a.TARGET_LAYERS:
            h,c=captured[6][n],captured[4][n]
            assert h[0]==c[0] and torch.equal(h[1],c[1]) and torch.equal(h[2],c[2])
            shape=tuple(model.get_submodule(n).weight.shape)
            expected=a.low.MXTensor((h[1].long()>>2).byte(),h[2],shape).decode()
            assert torch.equal(expected,c[3])
    finally:
        for hook in hooks:hook.remove()
    t.validate_gradients(model);t.validate_bn(model,1);t.validate_adam_state(model,opt,1)
    checks['exactly_six_high_encodings_one_pair_same_objects_no_low_encoder']=True
    # Evaluation uses the same source despite the original ascending order.
    model.eval()
    with patch.object(a.low,'encode_tensor',side_effect=AssertionError('FORBIDDEN')), \
         patch.object(a.high,'encode_tensor',wraps=original_high) as high_calls:
        with torch.no_grad(): fn([(x,y)],model,torch.nn.CrossEntropyLoss(),modules.losses.CrossEntropyLossSoft(),0,False)
        assert high_calls.call_count==6 and model._last_source_audit['source_pairs']==1
    checks['evaluation_4_then_6_also_shares_high_source']=True
    # Mutation between passes and stale/incomplete pairs must fail closed.
    model.train()
    def mutate_between():
        with a.shared_source_pairs(model):
            a.select_mode(model,6);model(x)
            with torch.no_grad():model.get_submodule(a.TARGET_LAYERS[0]).weight.add_(1e-4)
            a.select_mode(model,4);model(x)
    rejected(mutate_between)
    assert all(model.get_submodule(n).quantize_fn._source is None for n in a.TARGET_LAYERS)
    def incomplete():
        with a.shared_source_pairs(model):
            a.select_mode(model,6);model(x)
    rejected(incomplete)
    checks['mutation_incomplete_pair_rejected_cache_cleared_after_exception']=True

    # Neither serialized payload nor source-to-low path may carry target masters.
    h,_=a.source_payload(model,{'synthetic_fixture_only':True})
    old_keys=set(h['mx_layers'])
    with patch.object(a.high,'encode_tensor',side_effect=AssertionError('FORBIDDEN')), \
         patch.object(a.low,'encode_tensor',side_effect=AssertionError('FORBIDDEN')):
        payload=a.derive_payload(h,'1'*64)
        frozen=a.load_exported_model(modules.models,h,d,bit=4)
    assert set(h['mx_layers'])==old_keys
    assert not any(n+'.weight' in h['auxiliary_fp32_state'] for n in a.TARGET_LAYERS)
    assert not any(n+'.weight' in dict(frozen.named_parameters()) for n in a.TARGET_LAYERS)
    checks['source_derivation_without_master_or_encoder']=True
    return checks
