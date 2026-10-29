"""Small synthetic three-mode direct-quantization and recursive-distillation checks. No SVHN data or research accuracies.

CPU Tensor.cuda substitution exists ONLY in local validation test harnesses.
On the laptop these checks use the actual production CUDA device.
"""
from __future__ import annotations
import ast
import copy
import math
import json
from pathlib import Path
import tempfile
from contextlib import contextmanager
import os
import torch
import mx_format as high
import mx_e4m3 as highest
import mx_e2m1 as low
import joint_adapter as a
import joint_support as t
import svhn_support as s
import resume_core as core
from group_config import BLOCK_SIZE, padding_mask
from test_mx import rejected


def run_tests(modules, rt, source, log, checks=None):
    d = rt.device
    torch.manual_seed(71203)
    if checks is None:
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
    blocks = codes.repeat((2 * BLOCK_SIZE + codes.numel() - 1) // codes.numel())[:2 * BLOCK_SIZE].reshape(2, 1, BLOCK_SIZE)
    assert low.pack4(blocks).shape == (2, 1, BLOCK_SIZE * 4 // 8)
    assert torch.equal(low.unpack4(low.pack4(blocks)), blocks)
    w = torch.zeros(17, 9, dtype=torch.float32, device=d)
    w[0, :5] = torch.tensor([7., 1.375, -.25, -.5, -0.], device=d)
    w[8, 0] = 128.
    mh, ml = high.encode_tensor(w), low.encode_tensor(w)
    assert torch.equal(mh.scale_codes, ml.scale_codes)
    assert ml.codes.shape == (3, 2, BLOCK_SIZE) and bool((ml.codes[padding_mask(tuple(w.shape), d)] == 0).all())
    assert ml.decode()[0, :4].tolist() == [6., 1.5, -0., -.5]
    assert torch.signbit(ml.decode()[0, 2]) and torch.signbit(ml.decode()[0, 4])
    # Explicit negative control: direct low quantization is NOT a code bitshift.
    assert float(low.decode_elements(mh.codes[0, 0, 1] >> 2)) == 1.
    assert float(ml.decode()[0, 1]) == 1.5
    restored = low.MXTensor.from_payload(ml.payload(), d)
    assert torch.equal(restored.codes, ml.codes) and torch.equal(restored.decode(), ml.decode())
    for exponent in [-120, -40, -1, 0, 1, 12, 120]:
        power = torch.zeros(1, BLOCK_SIZE, device=d); power[0, 0] = 2. ** exponent
        encoded = low.encode_tensor(power)
        assert int(encoded.scale_codes[0, 0]) - 127 == exponent - 2
        assert torch.equal(encoded.decode(), power)
    tiny = torch.zeros(8, 8, device=d)
    tiny[0, 0] = torch.finfo(torch.float32).tiny
    tiny[0, 1] = torch.finfo(torch.float32).tiny / 2
    tiny_mx = low.encode_tensor(tiny)
    assert tiny_mx.scale_codes.item() == 0 and tiny_mx.decode()[0, 1] == 0
    assert low.distortion(tiny, tiny_mx)['fp32_subnormal_input_count'] == 1
    assert low.distortion(tiny, tiny_mx)['scale_range_clamped_blocks'] == 1
    assert low.distortion(w, ml)['out_of_element_range_before_round'] == 1
    zeros = low.MXTensor(torch.zeros(255, 1, BLOCK_SIZE, dtype=torch.uint8, device=d),
                         torch.arange(255, dtype=torch.uint8, device=d).reshape(255, 1), (255 * 8, 8))
    assert not bool(zeros.decode().any())
    rejected(lambda: low.encode_tensor(torch.tensor([[float('nan')]], device=d)))
    rejected(lambda: low.encode_tensor(torch.tensor([[float('inf')]], device=d)))
    rejected(lambda: low.encode_tensor(w.double()))
    rejected(lambda: low.decode_elements(torch.tensor([16], dtype=torch.uint8, device=d)))
    rejected(lambda: low.MXTensor(ml.codes, torch.full_like(ml.scale_codes, 255), ml.shape).decode())
    bad_padding = ml.codes.clone(); bad_padding[0, -1, -1] = 1
    rejected(lambda: low.MXTensor(bad_padding, ml.scale_codes, ml.shape).decode())
    checks['E2M1_scales_padding_saturation_extremes_roundtrip_and_direct_not_truncated'] = True

    # An exact binary witness separates direct E2M3 from both proposed alternatives.
    # 47/256 rounds to 3/16 in E4M3, whose second E2M3 rounding gives 1/4;
    # direct E2M3 gives 1/8. The leading 7 fixes the independent block scales.
    witness = torch.zeros(8, 8, dtype=torch.float32, device=d)
    witness[0, :2] = torch.tensor([7., 47. / 256.], device=d)
    source8, direct6 = highest.encode_tensor(witness), high.encode_tensor(witness)
    from_decoded8 = high.encode_tensor(source8.decode())
    code8 = source8.codes.to(torch.int64)
    # Retain s, the TWO highest exponent bits and all three mantissa bits;
    # keep the source scale. This is a negative control, never a training route.
    kept = (((code8 & 0xe0) >> 2) | (code8 & 7)).to(torch.uint8)
    exponent_msb_truncation = high.MXTensor(kept, source8.scale_codes.clone(), source8.shape)
    assert float(direct6.decode()[0, 1]) == .125
    assert float(from_decoded8.decode()[0, 1]) == .25
    assert not torch.equal(direct6.decode(), from_decoded8.decode())
    assert not torch.equal(direct6.decode(), exponent_msb_truncation.decode())
    assert int(source8.scale_codes.item()) - 127 == -6
    assert int(direct6.scale_codes.item()) - 127 == 0
    checks['direct_quantization_negative_controls'] = {
        'direct_E2M1_vs_E2M3_mantissa_shift': {'direct': 1.5, 'shifted': 1.0},
        'source_for_E2M3_witness': witness[0, :2].cpu().tolist(),
        'direct_E2M3': direct6.decode()[0, :2].cpu().tolist(),
        'E2M3_from_decoded_E4M3': from_decoded8.decode()[0, :2].cpu().tolist(),
        'E2M3_exponent_MSB_keep_with_E4M3_scale': exponent_msb_truncation.decode()[0, :2].cpu().tolist(),
        'no_negative_control_used_in_training': True,
    }

    # Every adapter route equals direct encode/decode of the SAME G(W), and
    # identity-STE preserves its derivative. Repeating after a master change
    # also rules out a stale quantized source being reused across passes.
    checks['direct_forward_and_identity_STE_by_format'] = {}
    coef = torch.linspace(-.7, .9, 66, device=d).reshape(2, 33)
    for bit in a.MODES:
        master1 = torch.linspace(-2, 2, 66, device=d).reshape(2, 33).requires_grad_()
        master2 = master1.detach().clone().requires_grad_()
        untouched = master1.detach().clone()
        q = a.JointMXQuantizer(); q.weight_mode = bit
        value = q(master1)
        (value * coef).sum().backward()
        (a.effective_reference_weight(master2) * coef).sum().backward()
        assert torch.equal(master1.grad, master2.grad)
        assert torch.equal(value.detach(), a.codec(bit).encode_tensor(a.effective_reference_weight(master1)).decode())
        assert torch.equal(master1.detach(), untouched) and not q.state_dict()
        with torch.no_grad():
            # A sign flip preserves the layer's abs-mean and maxabs, while
            # necessarily changing this nonzero quantized weight in all modes.
            # A small increment can legitimately stay inside the same bin.
            master1[0, 5].neg_()
        changed_value = q(master1)
        assert torch.equal(changed_value.detach(), a.codec(bit).encode_tensor(a.effective_reference_weight(master1)).decode())
        assert not torch.equal(value.detach(), changed_value.detach())
        checks['direct_forward_and_identity_STE_by_format'][a.FORMATS[bit]] = True
    for invalid in (True, 2, 32, '8'):
        rejected(lambda invalid=invalid: a.codec(invalid))

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

    checks['comparison_policy'] = {
        'production_replay': 'diagnostic only for floating-point numerical differences',
        'deterministic_replay': 'mandatory; original rtol=1e-4, atol=1e-5, not widened',
        'CPU': 'exact equality required',
        'NaN_structure_RNG_BN_and_Adam_health': 'mandatory on both paths',
        'scope': 'synthetic checks only; run/resume protocol not changed'}
    checks['training_comparisons'] = {}
    before = backend_state()
    prod = checks['training_comparisons']['production_backend'] = {}
    model = training_pair(modules, rt, source, log, prod)
    prod['mandatory_health_checks_passed'] = True
    if d.type == 'cpu':
        require_pair_match(prod, exact=True)
    elif not prod['all_numerically_close']:
        log('DIAGNOSE: herhaling met productiebackend verschilt; zie training_comparisons. '
            'Dit is nog GEEN PASS. De deterministische vergelijking moet slagen.')
    strict = checks['training_comparisons']['deterministic_backend'] = {}
    with deterministic_validation(d):
        model = training_pair(modules, rt, source, log, strict)
        require_pair_match(strict, exact=(d.type == 'cpu'))
    checks['production_backend_restored'] = backend_state() == before
    s.require(checks['production_backend_restored'], 'Zelftest heeft backendbeleid veranderd.')
    checks['two_joint_updates_match_explicit_manual_distillation'] = True
    checks['manual_match_scope'] = 'deterministic synthetic replay; see per-tensor comparisons'
    checks['three_BN_banks_once_per_batch_A32_unchanged_FP32_masters_and_Adam'] = True
    checks['synthetic_optimizer_steps_not_research_results'] = {
        'production_pair': {'tested': 2, 'manual_reference': 2},
        'deterministic_pair': {'tested': 2, 'manual_reference': 2}}
    log('VERGELIJKING PASS: deterministische test; geen versoepelde toleranties. '
        'Oorspronkelijke backendinstellingen hersteld.')
    x = torch.linspace(-1, 1, 3*3*40*40, device=d).reshape(3, 3, 40, 40)

    # Verify separately reconstructed direct exports, not truncation.
    model.eval()
    for bit in (8, 6, 4):
        payload, _ = a.export_payload(model, bit, {'synthetic_fixture_only': True})
        frozen = a.load_exported_model(modules.models, payload, d)
        a.select_mode(model, bit)
        with torch.no_grad():
            yf, ym = frozen(x), model(x)
        export_check = tensor_comparison(yf, ym, 'export_logits_' + str(bit), 1e-5, 1e-6)
        checks.setdefault('export_logit_comparisons', []).append(export_check)
        s.require(export_check['close'], 'Exportlogits verschillen: ' + str(export_check))
        assert not any(n + '.weight' in dict(frozen.named_parameters()) for n in a.TARGET_LAYERS)
        for name in a.TARGET_LAYERS:
            layer = model.get_submodule(name)
            assert frozen.get_submodule(name).decoded_weight.device == d
            assert torch.equal(layer.quantize_fn(layer.weight), frozen.get_submodule(name).decoded_weight)
    checks['all_three_direct_exports_reconstruct_without_six_FP32_masters'] = True
    checks['no_SVHN_data'] = True
    return checks


def backend_state():
    return {'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


@contextmanager
def deterministic_validation(device):
    """Only test execution. Restore flags even when a comparison raises."""
    previous = backend_state()
    if device.type == 'cuda':
        s.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'),
                  'Start opnieuw met run_train.py check; deterministische cuBLAS-config ontbreekt.')
        torch.cuda.synchronize(device)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous['deterministic_algorithms'],
                                          warn_only=previous['deterministic_warn_only'])
        torch.backends.cudnn.deterministic = previous['cudnn_deterministic']
        torch.backends.cudnn.benchmark = previous['cudnn_benchmark']


def tensor_comparison(expected, actual, name, rtol=1e-4, atol=1e-5):
    result = {'name': name, 'rtol': rtol, 'atol': atol,
              'expected_shape': list(expected.shape), 'expected_dtype': str(expected.dtype)}
    result['structure_ok'] = (isinstance(actual, torch.Tensor)
                              and expected.dtype == actual.dtype and expected.shape == actual.shape)
    if not result['structure_ok']:
        return {**result, 'finite': False, 'exact': False, 'close': False}
    x, y = expected.detach().cpu(), actual.detach().cpu()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    result.update(finite=finite, exact=torch.equal(x, y))
    if not finite:
        return {**result, 'close': False}
    if x.is_floating_point() or x.is_complex():
        close_mask = torch.isclose(x, y, rtol=rtol, atol=atol, equal_nan=False)
    else:
        close_mask = x == y
    # Diagnostic arithmetic only; no changes to the model tensors.
    difference = (x.to(torch.float64) - y.to(torch.float64)).abs()
    result.update(close=bool(close_mask.all()),
                  different_elements=int((x != y).sum()),
                  outside_tolerance_elements=int((~close_mask).sum()),
                  max_abs_difference=float(difference.max()) if x.numel() else 0.0)
    return result


def compare_tree(expected, actual, path, rows):
    if isinstance(expected, torch.Tensor):
        rows.append(tensor_comparison(expected, actual, path))
    elif isinstance(expected, dict):
        s.require(isinstance(actual, dict) and set(expected) == set(actual), 'Structuurverschil: ' + path)
        for key in expected:
            compare_tree(expected[key], actual[key], path + '/' + str(key), rows)
    elif isinstance(expected, (tuple, list)):
        s.require(type(expected) is type(actual) and len(expected) == len(actual), 'Structuurverschil: ' + path)
        for i, (x, y) in enumerate(zip(expected, actual)):
            compare_tree(x, y, path + '/' + str(i), rows)
    else:
        s.require(type(expected) is type(actual) and expected == actual, 'Niet-tensorverschil: ' + path)


def require_pair_match(details, exact=False):
    bad = [row for row in details['tensor_comparisons'] if not row['exact' if exact else 'close']]
    if bad:
        worst = sorted(bad, key=lambda row: row.get('max_abs_difference', float('inf')), reverse=True)
        first = worst[0]
        raise RuntimeError('Manuele/Any-Precision-vergelijking mislukt: '
                           + str(len(bad)) + ' tensors; grootste afwijking: ' + str(first)
                           + '. Volledige diagnose staat in run_report.json. Geen SVHN-training starten.')
    s.require(details['rng_exact'], 'RNG-toestanden verschillen in vergelijking.')
    details['mandatory_match_passed'] = True


class AuditedSoftLoss(torch.nn.Module):
    """Test-only proof that each student receives its immediately preceding teacher."""
    def __init__(self, original, mode_order, logits):
        super().__init__()
        self.original, self.mode_order, self.logits = original, mode_order, logits
        self.targets, self.checks = [], []

    def forward(self, output, target):
        s.require(not target.requires_grad and target.grad_fn is None, 'Teacher moet detached zijn.')
        student, teacher = self.mode_order[-1], self.mode_order[-2]
        s.require({6: 8, 4: 6}.get(student) == teacher, 'Geen recursieve E4M3->E2M3->E2M1-teachervolgorde.')
        expected = torch.softmax(self.logits[-2].detach(), dim=1)
        s.require(torch.equal(target, expected), 'Student ontvangt niet de direct voorgaande detached teacher.')
        self.targets.append(target.detach().clone())
        self.checks.append({'teacher_weight_bits': teacher, 'teacher_weight_format': a.FORMATS[teacher],
                            'student_weight_bits': student, 'student_weight_format': a.FORMATS[student],
                            'detached': True, 'matches_preceding_teacher_exactly': True})
        return self.original(output, target)


def training_pair(modules, rt, source, log, details):
    """Independent manual three-pass loop vs the selector-adapted author loop."""
    d = rt.device
    details.update(backend=backend_state(), device=str(d), status='RUNNING')
    torch.manual_seed(83411)
    model = t.build_model(modules, 'joint', d)
    initial = t.layer_snapshot(model)
    x = torch.linspace(-1, 1, 3*3*40*40, device=d).reshape(3, 3, 40, 40)
    labels = torch.tensor([1, 4, 9], device=d)
    reference = copy.deepcopy(model)
    s.require(not core.differences(reference.state_dict(), model.state_dict()), 'Startgewichten verschillen.')
    opt = modules.optimizer.get_optimizer_config(model, 'adam', .001, 0.)
    reference_opt = modules.optimizer.get_optimizer_config(reference, 'adam', .001, 0.)
    reference_outputs, outputs, reference_gradients, gradients = [], [], [], []
    reference_modes, modes = [], []
    def output_hook(store, order):
        def hook(module, inputs, result):
            store.append(result.detach().clone())
            order.append(module.weight_mode)
        return hook
    def gradient_hook(net, store):
        def hook(optimizer, args, kwargs):
            t.validate_gradients(net)
            store.append({name: None if p.grad is None else p.grad.detach().cpu().clone()
                          for name, p in net.named_parameters()})
        return hook
    hooks = [reference.register_forward_hook(output_hook(reference_outputs, reference_modes)),
             model.register_forward_hook(output_hook(outputs, modes)),
             reference_opt.register_step_pre_hook(gradient_hook(reference, reference_gradients)),
             opt.register_step_pre_hook(gradient_hook(model, gradients))]
    temporary = tempfile.TemporaryDirectory(prefix='mx_joint3_check_')
    recorder = None
    try:
        rng = core.capture_rng({}, d)
        reference.train()
        criterion = torch.nn.CrossEntropyLoss()
        reference_soft = AuditedSoftLoss(modules.losses.CrossEntropyLossSoft(),
                                         reference_modes, reference_outputs)
        expected = []
        for _ in range(2):
            reference_opt.zero_grad()
            a.select_mode(reference, 8)
            high_logits = reference(x)
            hard_loss = criterion(high_logits, labels); hard_loss.backward()
            s.require(all(p.grad is None for n, p in reference.named_parameters()
                          if '.bn_dict.32.' in n or '.bn_dict.4.' in n),
                      'Hoogste teacherpass wijzigt gradients in student-BN.')
            high_teacher = torch.softmax(high_logits.detach(), dim=1)
            a.select_mode(reference, 6)
            middle_logits = reference(x)
            middle_loss = reference_soft(middle_logits, high_teacher); middle_loss.backward()
            s.require(all(p.grad is None for n, p in reference.named_parameters() if '.bn_dict.4.' in n),
                      'Middelste pass wijzigt gradients in laagste BN.')
            middle_teacher = torch.softmax(middle_logits.detach(), dim=1)
            a.select_mode(reference, 4)
            low_logits = reference(x)
            low_loss = reference_soft(low_logits, middle_teacher); low_loss.backward()
            t.validate_gradients(reference)
            reference_opt.step()
            expected.append((float(hard_loss.detach()), float(middle_loss.detach()), float(low_loss.detach())))
        final_rng = core.capture_rng({}, d)
        core.restore_rng(rng, {}, d)
        metric_file = Path(temporary.name) / 'training_batches.jsonl'
        recorder = t.JointMeterRecorder(modules.utils.AverageMeter, 1, 'train', .001, metric_file, 'joint')
        forward, _ = t.load_joint_forward(source, modules.utils, log, recorder.factory)
        model.train()
        actual_soft = AuditedSoftLoss(modules.losses.CrossEntropyLossSoft(), modes, outputs)
        actual = forward([(x, labels), (x, labels)], model, criterion, actual_soft, 0, True, opt, None)
        details['rng_exact'] = not core.differences(final_rng, core.capture_rng({}, d))
        metric_rows = recorder.rows(actual)
        assert len(recorder.meters) == 9 and recorder.batch == 2 and recorder.n == 6
        assert [row['weight_bits'] for row in metric_rows] == [4, 6, 8]
        assert [row['objective_type'] for row in metric_rows] == [
            'recursive_soft_cross_entropy', 'recursive_soft_cross_entropy', 'hard_cross_entropy']
        assert [row['teacher_weight_format'] for row in metric_rows] == [a.FORMATS[6], a.FORMATS[8], None]
        assert all(row['weight_format'] == a.FORMATS[row['weight_bits']] for row in metric_rows)
        recorder.close(); recorder = None
        batch_rows = [json.loads(line) for line in metric_file.read_text().splitlines()]
        assert len(batch_rows) == 2 and all(len(row['precisions']) == 3 for row in batch_rows)
        assert modes == reference_modes == [8, 6, 4, 8, 6, 4]
        assert len(gradients) == len(reference_gradients) == 2
        assert len(actual_soft.checks) == len(reference_soft.checks) == 4
        details['manual_losses_high_middle_low'] = expected
        details['upstream_mean_losses_low_middle_high'] = actual[0]
        details['recorded_training_metrics_low_to_high'] = metric_rows
        details['nine_meters_three_modes_two_batches_verified'] = True
        details['forward_mode_order'] = modes
        details['detached_teacher_source_checks'] = actual_soft.checks
        rows = details['tensor_comparisons'] = []
        compare_tree(reference_outputs, outputs, 'forward_logits', rows)
        compare_tree(reference_soft.targets, actual_soft.targets, 'detached_teacher_targets', rows)
        compare_tree(reference_gradients, gradients, 'pre_update_gradients', rows)
        compare_tree(reference.state_dict(), model.state_dict(), 'model_after_two_updates', rows)
        compare_tree(reference_opt.state_dict(), opt.state_dict(), 'adam_after_two_updates', rows)
        ref_losses = torch.tensor([sum(item[index] for item in expected) / 2 for index in (2, 1, 0)],
                                  dtype=torch.float64)
        rows.append(tensor_comparison(ref_losses, torch.tensor(actual[0], dtype=torch.float64), 'mean_losses'))
        details['all_exact'] = all(row['exact'] for row in rows)
        details['all_numerically_close'] = all(row['close'] for row in rows)
        details['outside_tolerance_tensors'] = [row['name'] for row in rows if not row['close']]
        s.require(all(row['structure_ok'] and row['finite'] for row in rows), 'Niet-eindige/ongeldige vergelijking.')
        s.require(details['rng_exact'], 'Handmatige lus en auteurslus verbruiken andere RNG-toestanden.')
        for net, optimizer in ((model, opt), (reference, reference_opt)):
            t.assert_policy(net); t.validate_bn(net, 2); t.validate_adam_state(net, optimizer, 2)
        updates = t.layer_updates(initial, t.layer_snapshot(model), 'joint', 0)
        s.require(len(updates) == 18 and all(row['master_changed_values'] > 0 for row in updates), 'Geen masterupdates.')
        s.require(all(row['weight_format'] == a.FORMATS[row['weight_bits']] for row in updates),
                  'Formaatlabel van update wijkt af.')
        details['mandatory_health_checks_passed'] = True
        details['status'] = 'COMPARISON_RECORDED'
        log('ZELFTEST ' + ('deterministisch' if torch.are_deterministic_algorithms_enabled() else 'productiebeleid')
            + ': ' + str(len(details['outside_tolerance_tensors'])) + ' tensors buiten tolerantie; '
            + 'RNG exact=' + str(details['rng_exact']))
    finally:
        if recorder is not None:
            recorder.close()
        temporary.cleanup()
        for hook in hooks:
            hook.remove()
    return model
