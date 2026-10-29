"""Small deterministic numerical and model tests; no dataset or optimizer."""
from __future__ import annotations
import io
import math
from pathlib import Path
import sys
import tempfile
import torch
import mx_format as f
import mx_adapter as a
from group_config import BLOCK_SIZE, padding_mask


def rejected(call) -> None:
    try:
        call()
    except (ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Ongeldige invoer werd niet afgewezen.')


def run_tests(modules, device: str) -> dict:
    torch.manual_seed(71203)
    checks = {}
    d = torch.device(device)
    codes = torch.arange(64, dtype=torch.uint8, device=d)
    values = f.decode_elements(codes)
    assert torch.equal(f.encode_elements(values), codes)
    assert torch.signbit(values[32]) and not torch.signbit(values[0])
    levels = [i/8 for i in range(8)] + [1+i/8 for i in range(8)] + [2+i/4 for i in range(8)] + [4+i/2 for i in range(8)]
    assert values[:32].cpu().tolist() == levels
    checks['all_64_element_codes_and_signed_zero'] = True

    # Independent scalar oracle: enumerate the real-value table, use even code
    # at an exact midpoint. Include both sides of every positive midpoint.
    samples = [-8.0, -0.0, 0.0, 8.0]
    for low, high in zip(levels[:-1], levels[1:]):
        m = (low+high)/2
        samples.extend([m, math.nextafter(m, -math.inf), math.nextafter(m, math.inf)])
        samples.extend([-m, math.nextafter(-m, -math.inf), math.nextafter(-m, math.inf)])
    samples.extend(torch.linspace(-9, 9, 4097, dtype=torch.float64).tolist())
    expected = []
    for v in samples:
        i = min(range(32), key=lambda c: (abs(abs(v)-levels[c]), c & 1))
        expected.append(i + (32 if math.copysign(1.0, v) < 0 else 0))
    actual = f.encode_elements(torch.tensor(samples, dtype=torch.float64, device=d)).cpu().tolist()
    assert actual == expected
    checks['scalar_oracle_samples'] = len(samples)
    checks['every_midpoint_both_sides_RNE'] = True

    w = torch.zeros(17, 9, dtype=torch.float32, device=d)
    w[0,:4] = torch.tensor([7.0, 1.3, -3.33, 0.0], device=d)
    w[8,0] = 128.0
    w[16,0] = -0.0
    mx = f.encode_tensor(w)
    assert mx.scale_codes[0].tolist() == [127,127]
    assert mx.scale_codes[1].tolist() == [132,127]
    assert mx.scale_codes[2].tolist() == [127,127]
    assert mx.decode()[0,:4].tolist() == [7.0,1.25,-3.25,0.0]
    assert torch.signbit(mx.decode()[16,0])
    w_changed = w.clone(); w_changed[8] *= 1024
    mx_changed = f.encode_tensor(w_changed)
    assert torch.equal(mx.codes[0],mx_changed.codes[0]) and torch.equal(mx.scale_codes[0],mx_changed.scale_codes[0])
    assert mx.codes.shape == (3,2,BLOCK_SIZE) and bool((mx.codes[padding_mask(tuple(w.shape), d)]==0).all())
    checks['tile_group_axis_both_padding_zero_blocks_independent_tiles'] = True

    for exponent in [-120,-40,-1,0,1,12,120]:
        v = torch.zeros(1,BLOCK_SIZE,device=d)
        v[0,0] = 2.0 ** exponent
        q = f.encode_tensor(v)
        assert int(q.scale_codes[0,0])-127 == exponent-2
        assert torch.equal(v,q.decode())
    tiny = torch.zeros(1,BLOCK_SIZE,device=d)
    tiny[0,0] = torch.finfo(torch.float32).tiny
    tiny[0,1] = torch.finfo(torch.float32).tiny/2
    z = f.encode_tensor(tiny)
    assert int(z.scale_codes[0,0]) == 0
    assert float(z.decode()[0,0]) == float(tiny[0,0]) and float(z.decode()[0,1]) == 0
    assert f.distortion(tiny,z)['fp32_subnormal_input_count'] == 1
    assert f.distortion(tiny,z)['scale_range_clamped_blocks'] == 1
    huge=torch.zeros(1,BLOCK_SIZE,device=d);huge[0,0]=torch.finfo(torch.float32).max
    assert bool(torch.isfinite(f.encode_tensor(huge).decode()).all())
    checks['scale_rule_powers_of_two_and_fp32_extremes'] = True

    every_scale = f.MXTensor(torch.zeros((255,1,BLOCK_SIZE),device=d,dtype=torch.uint8),
                             torch.arange(255,device=d,dtype=torch.uint8).reshape(255,1),(255*8,8))
    assert bool((every_scale.decode()==0).all())
    rejected(lambda: f.MXTensor(mx.codes, torch.full_like(mx.scale_codes,255), mx.shape).decode())
    rejected(lambda: f.encode_tensor(torch.tensor([[float('nan')]],device=d)))
    rejected(lambda: f.encode_tensor(torch.tensor([[float('inf')]],device=d)))
    rejected(lambda: f.encode_tensor(w.double()))
    rejected(lambda: f.decode_elements(torch.tensor([64],device=d,dtype=torch.uint8)))
    overflow = f.MXTensor(torch.full((1,1,BLOCK_SIZE),31,device=d,dtype=torch.uint8),
                          torch.full((1,1),254,device=d,dtype=torch.uint8),(8,8))
    rejected(overflow.decode)
    checks['all_255_finite_scales_and_invalid_inputs'] = True

    many_codes = torch.randint(0,64,(30,4,BLOCK_SIZE),device=d,dtype=torch.uint8)
    many_codes[0,:4,:] = codes.repeat((4 * BLOCK_SIZE + codes.numel() - 1) // codes.numel())[:4 * BLOCK_SIZE].reshape(4, BLOCK_SIZE)
    assert torch.equal(f.unpack6(f.pack6(many_codes)), many_codes)
    assert f.pack6(many_codes).shape == (30, 4, BLOCK_SIZE * 6 // 8)
    mx2 = f.MXTensor.from_payload(mx.payload(),d)
    assert torch.equal(mx.codes,mx2.codes) and torch.equal(mx.decode(),mx2.decode())
    checks['packed_6bit_codes_and_scale_roundtrip'] = True

    original = modules.models.svhnq([1,2,4,8,32], expand=2).to(d)
    a.select_a32(original);original.eval()
    saved = {k:v.detach().clone() for k,v in original.state_dict().items()}
    wrapped = a.adapt_copy(original,False).eval()
    cast = a.adapt_copy(original,True).eval()
    actual = tuple(n for n,m in original.named_modules() if isinstance(m,torch.nn.Conv2d) and hasattr(m,'quantize_fn'))
    assert actual == a.TARGET_LAYERS
    x = torch.linspace(-1,1,2*3*40*40,device=d).reshape(2,3,40,40)
    with torch.no_grad():
        y0,y1=original(x),wrapped(x)
    assert torch.allclose(y0,y1,rtol=1e-5,atol=1e-6)
    for n in a.TARGET_LAYERS:
        master = original.get_submodule(n).weight
        assert torch.equal(original.get_submodule(n).quantize_fn(master),a.effective_reference_weight(master))
    assert all(torch.equal(saved[k],v) for k,v in original.state_dict().items())
    checks['MX_off_matches_original_W32_A32'] = True
    checks['reference_weights_and_BN_unchanged'] = True
    guard = a.MXWeightQuantizer(True); guard.abit = 8
    rejected(lambda: guard(original.get_submodule(a.TARGET_LAYERS[0]).weight))
    checks['old_coupled_five_bit_loop_rejected'] = True

    w1 = original.get_submodule(a.TARGET_LAYERS[0]).weight.detach().clone().requires_grad_()
    w2 = w1.detach().clone().requires_grad_()
    coefficient = torch.linspace(-.5,.5,w1.numel(),device=d).reshape_as(w1)
    surrogate_value = a.MXWeightQuantizer(True)(w1)
    (surrogate_value*coefficient).sum().backward()
    (a.effective_reference_weight(w2)*coefficient).sum().backward()
    assert torch.equal(w1.grad,w2.grad) and bool(torch.isfinite(w1.grad).all())
    assert bool((w1.grad != 0).any())
    assert torch.equal(surrogate_value.detach(),f.encode_tensor(a.effective_reference_weight(w1)).decode())
    assert torch.equal(a.effective_reference_weight(torch.zeros_like(w1)),torch.zeros_like(w1))
    checks['declared_identity_STE_and_reference_preprocess_gradient'] = True
    checks['all_zero_layer_guard'] = True

    payload, layer_rows = a.export_payload(original, {'synthetic_only':True})
    stream = io.BytesIO(); torch.save(payload,stream);stream.seek(0)
    roundtrip = torch.load(stream,map_location='cpu',weights_only=True)
    frozen = a.load_exported_model(modules.models,roundtrip,d)
    with torch.no_grad():
        yc,yf=cast(x),frozen(x)
    assert torch.allclose(yc,yf,rtol=1e-5,atol=1e-6)
    assert all(name+'.weight' not in payload['auxiliary_fp32_state'] for name in a.TARGET_LAYERS)
    assert all(not list(frozen.get_submodule(n).parameters()) for n in a.TARGET_LAYERS)
    assert all(torch.equal(cast.get_submodule(n).quantize_fn(cast.get_submodule(n).weight),
                           frozen.get_submodule(n).decoded_weight) for n in a.TARGET_LAYERS)
    assert len(layer_rows)==6
    bad=dict(payload);bad['auxiliary_fp32_state']=dict(payload['auxiliary_fp32_state']);bad['auxiliary_fp32_state']['layers.3.weight']=w1
    rejected(lambda:a.load_exported_model(modules.models,bad,d))
    checks['source_only_export_reload_no_FP_master_for_six_layers'] = True
    checks['synthetic_optimizer_steps'] = 0
    checks['SVHN_data_read'] = False
    return checks

if __name__ == '__main__':
    import svhn_support as s
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    source,info=s.legacy.prepare_source(out)
    rt=s.runtime(20261009,'cpu',strict=False)
    torch.set_num_threads(2)
    modules=s.import_source(source)
    checks=run_tests(modules,'cpu')
    report={'status':'PASS','environment':rt.environment,'checks':checks,
            'scope':'synthetic CPU tests; no SVHN training; no real checkpoint accuracy; no CUDA test'}
    s.write_json(out/'test_report.json',report)
    print(__import__('json').dumps(report,indent=2))
