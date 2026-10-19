"""Small synthetic two-mode/codec checks. No SVHN data or research accuracies.

CPU Tensor.cuda substitution exists ONLY in local_validation/test_training.py.
On the laptop these checks use the actual production CUDA device.
"""
from __future__ import annotations
import ast
import copy
import math
import torch
import mx_format as high
import mx_e2m1 as low
import dual_adapter as a
import dual_support as t
import svhn_support as s
import resume_core as core
from test_mx import rejected


def run_tests(modules, rt, source, log):
    d = rt.device
    torch.manual_seed(71203)
    checks = {}
    codes = torch.arange(16, dtype=torch.uint8, device=d)
    levels = [0., .5, 1., 1.5, 2., 3., 4., 6.]  # OCP v1.0 table 5, p13.
    values = low.decode_elements(codes)
    assert values[:8].cpu().tolist() == levels
    assert torch.equal(low.encode_elements(values), codes)
    assert torch.signbit(values[8]) and not torch.signbit(values[0])
    samples = [-9., -0., 0., 9.]
    for x, y in zip(levels, levels[1:]):
        mid = (x + y) / 2
        for m in (mid, -mid):
            samples += [math.nextafter(m, -math.inf), m, math.nextafter(m, math.inf)]
    samples += torch.linspace(-9, 9, 4097, dtype=torch.float64).tolist()
    expected = [min(range(8), key=lambda c: (abs(abs(v)-levels[c]), c & 1)) +
                (8 if math.copysign(1., v) < 0 else 0) for v in samples]
    actual = low.encode_elements(torch.tensor(samples, device=d, dtype=torch.float64)).cpu().tolist()
    assert actual == expected
    checks['all_16_E2M1_codes_signed_zero_and_midpoint_oracle'] = len(samples)
    blocks = codes.repeat(4).reshape(2, 1, 32)
    assert low.pack4(blocks).shape == (2, 1, 16)
    assert torch.equal(low.unpack4(low.pack4(blocks)), blocks)
    w = torch.zeros(3, 33, dtype=torch.float32, device=d)
    w[0, :5] = torch.tensor([7., 1.375, -.25, -.5, -0.], device=d)
    w[1, 0] = 128.
    mh, ml = high.encode_tensor(w), low.encode_tensor(w)
    assert torch.equal(mh.scale_codes, ml.scale_codes)
    assert ml.codes.shape == (3, 2, 32) and bool((ml.codes[:, -1, 1:] == 0).all())
    assert ml.decode()[0, :4].tolist() == [6., 1.5, -0., -.5]
    assert torch.signbit(ml.decode()[0, 2]) and torch.signbit(ml.decode()[0, 4])
    # Explicit negative control: direct low quantization is NOT a code bitshift.
    assert float(low.decode_elements(mh.codes[0, 0, 1] >> 2)) == 1.
    assert float(ml.decode()[0, 1]) == 1.5
    restored = low.MXTensor.from_payload(ml.payload(), d)
    assert torch.equal(restored.codes, ml.codes) and torch.equal(restored.decode(), ml.decode())
    for exponent in [-120, -40, -1, 0, 1, 12, 120]:
        power = torch.zeros(1, 32, device=d); power[0, 0] = 2. ** exponent
        encoded = low.encode_tensor(power)
        assert int(encoded.scale_codes[0, 0]) - 127 == exponent - 2
        assert torch.equal(encoded.decode(), power)
    tiny = torch.zeros(1, 32, device=d)
    tiny[0, 0] = torch.finfo(torch.float32).tiny
    tiny[0, 1] = torch.finfo(torch.float32).tiny / 2
    tiny_mx = low.encode_tensor(tiny)
    assert tiny_mx.scale_codes.item() == 0 and tiny_mx.decode()[0, 1] == 0
    assert low.distortion(tiny, tiny_mx)['fp32_subnormal_input_count'] == 1
    assert low.distortion(tiny, tiny_mx)['scale_range_clamped_blocks'] == 1
    assert low.distortion(w, ml)['out_of_element_range_before_round'] == 1
    zeros = low.MXTensor(torch.zeros(255, 1, 32, dtype=torch.uint8, device=d),
                         torch.arange(255, dtype=torch.uint8, device=d).reshape(255, 1), (255, 32))
    assert not bool(zeros.decode().any())
    rejected(lambda: low.encode_tensor(torch.tensor([[float('nan')]], device=d)))
    rejected(lambda: low.encode_tensor(torch.tensor([[float('inf')]], device=d)))
    rejected(lambda: low.encode_tensor(w.double()))
    rejected(lambda: low.decode_elements(torch.tensor([16], dtype=torch.uint8, device=d)))
    rejected(lambda: low.MXTensor(ml.codes, torch.full_like(ml.scale_codes, 255), ml.shape).decode())
    bad_padding = ml.codes.clone(); bad_padding[0, -1, -1] = 1
    rejected(lambda: low.MXTensor(bad_padding, ml.scale_codes, ml.shape).decode())
    checks['E2M1_scales_padding_saturation_extremes_roundtrip_and_direct_not_truncated'] = True

    # Exact encoder forward, declared identity backward including G(W).
    master1 = torch.linspace(-2, 2, 66, device=d).reshape(2, 33).requires_grad_()
    master2 = master1.detach().clone().requires_grad_()
    coef = torch.linspace(-.7, .9, 66, device=d).reshape(2, 33)
    q = a.DualMXQuantizer(); q.weight_mode = 4
    value = q(master1)
    (value * coef).sum().backward()
    (a.effective_reference_weight(master2) * coef).sum().backward()
    assert torch.equal(master1.grad, master2.grad)
    assert torch.equal(value.detach(), low.encode_tensor(a.effective_reference_weight(master1)).decode())
    checks['E2M1_identity_STE_through_original_GW'] = True

    original, changed, _ = t.adapt_forward_text(source)
    normalized_original = ast.parse(original)
    normalized_changed = ast.parse(changed)
    # Delete ONLY selector statements from both trees; remaining AST must match.
    class RemoveSelectors(ast.NodeTransformer):
        def visit_Expr(self, node):
            v = node.value
            if isinstance(v, ast.Call):
                if isinstance(v.func, ast.Name) and v.func.id == 'select_mode': return None
                if isinstance(v.func, ast.Attribute) and v.func.attr == 'apply': return None
            return self.generic_visit(node)
    aa = RemoveSelectors().visit(normalized_original); bb = RemoveSelectors().visit(normalized_changed)
    assert ast.dump(aa, include_attributes=False) == ast.dump(bb, include_attributes=False)
    checks['original_forward_AST_identical_except_three_mode_selector_pairs'] = True

    torch.manual_seed(83411)
    model = t.build_model(modules, 'joint', d)
    initial = t.layer_snapshot(model)
    x = torch.linspace(-1, 1, 3*3*40*40, device=d).reshape(3, 3, 40, 40)
    labels = torch.tensor([1, 4, 9], device=d)
    reference = copy.deepcopy(model)
    opt = modules.optimizer.get_optimizer_config(model, 'adam', .001, 0.)
    reference_opt = modules.optimizer.get_optimizer_config(reference, 'adam', .001, 0.)
    rng = core.capture_rng({}, d)
    reference.train()
    criterion = torch.nn.CrossEntropyLoss()
    soft = modules.losses.CrossEntropyLossSoft()
    expected = []
    for _ in range(2):
        reference_opt.zero_grad()
        a.select_mode(reference, 6)
        teacher_logits = reference(x)
        hard_loss = criterion(teacher_logits, labels); hard_loss.backward()
        # Low bank is not trained by the teacher pass.
        assert all(p.grad is None for n, p in reference.named_parameters() if '.bn_dict.4.' in n)
        teacher = torch.softmax(teacher_logits.detach(), dim=1)
        a.select_mode(reference, 4)
        student_logits = reference(x)
        soft_loss = soft(student_logits, teacher); soft_loss.backward()
        t.validate_gradients(reference)
        reference_opt.step()
        expected.append((float(hard_loss.detach()), float(soft_loss.detach())))
    final_rng = core.capture_rng({}, d)
    core.restore_rng(rng, {}, d)
    forward, _ = t.load_dual_forward(source, modules.utils, log)
    model.train()
    forward([(x, labels), (x, labels)], model, criterion, soft, 0, True, opt, None)
    if d.type == 'cpu':
        assert not core.differences(reference.state_dict(), model.state_dict())
        assert not core.differences(reference_opt.state_dict(), opt.state_dict())
        assert not core.differences(final_rng, core.capture_rng({}, d))
    else:
        # cuDNN policy can be nondeterministic; do not demand bit equality.
        assert all(torch.allclose(v, model.state_dict()[n], rtol=1e-4, atol=1e-5)
                   for n, v in reference.state_dict().items())
    t.assert_policy(model); t.validate_bn(model, 2); t.validate_adam_state(model, opt, 2)
    updates = t.layer_updates(initial, t.layer_snapshot(model), 'joint', 0)
    assert all(row['master_changed_values'] > 0 for row in updates)
    checks['two_joint_updates_match_explicit_manual_distillation'] = True
    checks['two_BN_banks_once_per_batch_A32_unchanged_FP32_masters_and_Adam'] = True
    checks['synthetic_optimizer_steps_not_research_results'] = {'tested': 2, 'manual_reference': 2}

    # Verify separately reconstructed direct exports, not truncation.
    model.eval()
    for bit in (6, 4):
        payload, _ = a.export_payload(model, bit, {'synthetic_fixture_only': True})
        frozen = a.load_exported_model(modules.models, payload, d)
        a.select_mode(model, bit)
        with torch.no_grad():
            yf, ym = frozen(x), model(x)
        assert torch.allclose(yf, ym, rtol=1e-5, atol=1e-6)
        assert not any(n + '.weight' in dict(frozen.named_parameters()) for n in a.TARGET_LAYERS)
        for name in a.TARGET_LAYERS:
            layer = model.get_submodule(name)
            assert torch.equal(layer.quantize_fn(layer.weight), frozen.get_submodule(name).decoded_weight)
    checks['both_direct_mode_exports_reconstruct_without_six_FP32_masters'] = True
    checks['no_SVHN_data'] = True
    return checks
