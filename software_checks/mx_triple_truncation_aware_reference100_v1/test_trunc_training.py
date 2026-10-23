"""Synthetic codec/cache/STE/three-pass training and source-only export checks.

No SVHN or research accuracy. CPU Tensor.cuda substitution lives only in the
local validation harness; laptop checks use the actual production device.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from contextlib import contextmanager
import torch
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low
import trunc_adapter as a
import trunc_support as t
import svhn_support as s
import resume_core as core
from test_mx import rejected
from test_joint import (backend_state, deterministic_validation, tensor_comparison,
                        compare_tree, require_pair_match, AuditedSoftLoss)


def bitstring_oracle(code, bit):
    word = format(code, '08b')
    return code if bit == 8 else int(word[:3] + (word[5:] if bit == 6 else word[5]), 2)


class _IndependentSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, effective, decoded): return decoded
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient): return gradient.clone(), None


class IndependentQuantizer(a.JointMXQuantizer):
    """Manual-loop oracle: fresh E4 encoding, independent string-table mapping."""
    def forward(self, weight):
        effective = a.effective_reference_weight(weight)
        source = high.encode_tensor(effective)
        bit = self.weight_mode
        lookup = torch.tensor([bitstring_oracle(i, bit) for i in range(256)],
                              dtype=torch.uint8, device=weight.device)
        codes = lookup[source.codes.long()]
        decoded = a.codec(bit).MXTensor(codes, source.scale_codes.clone(), source.shape).decode()
        return _IndependentSTE.apply(effective, decoded)


@contextmanager
def forbidden_lower_encoders():
    methods = [(module, name, getattr(module, name)) for module in (middle, low)
               for name in ('encode_tensor', 'encode_elements', 'fake_quantize_mx')]
    def fail(*args, **kwargs): raise AssertionError('Lower precision encoder was called.')
    try:
        for module, name, _ in methods: setattr(module, name, fail)
        yield
    finally:
        for module, name, original in methods: setattr(module, name, original)


@contextmanager
def count_high_encodings():
    original = high.encode_tensor
    count = [0]
    def wrapped(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)
    try:
        high.encode_tensor = wrapped
        yield count
    finally:
        high.encode_tensor = original


def run_tests(modules, rt, source, log, checks=None):
    checks = {} if checks is None else checks
    before = core.capture_rng({}, rt.device)
    flags = backend_state()
    try:
        return _run_tests(modules, rt, source, log, checks)
    finally:
        core.restore_rng(before, {}, rt.device)
        s.require(backend_state() == flags, 'Zelftest heeft backendbeleid veranderd.')
        checks['RNG_and_backend_restored'] = True


def _run_tests(modules, rt, source, log, checks):
    d = rt.device
    finite = [i for i in range(256) if i not in (127, 255)]
    codes = torch.tensor(finite, dtype=torch.uint8, device=d).reshape(-1, 1, 1).expand(-1, 1, 32).clone()
    source_mx = high.MXTensor(codes, torch.full((254, 1), 127, dtype=torch.uint8, device=d), (254, 32))
    with forbidden_lower_encoders():
        chain = a.code_chain(source_mx)
    for bit in (6, 4):
        expected = torch.tensor([bitstring_oracle(i, bit) for i in finite], dtype=torch.uint8, device=d)
        assert torch.equal(chain[bit].codes[:, 0, 0], expected)
        assert torch.equal(chain[bit].scale_codes, source_mx.scale_codes)
        assert torch.equal(a.codec(bit).MXTensor.from_payload(chain[bit].payload(), d).codes, chain[bit].codes)
    assert torch.signbit(chain[6].decode()[127, 0]) and torch.signbit(chain[4].decode()[127, 0])
    checks['all_254_finite_codes_independent_string_oracle_and_signed_zero'] = True
    scales = torch.arange(255, dtype=torch.uint8, device=d).reshape(255, 1)
    all_scales = high.MXTensor(torch.zeros(255, 1, 32, dtype=torch.uint8, device=d), scales, (255, 32))
    with forbidden_lower_encoders():
        for bit, mx in a.code_chain(all_scales).items():
            assert torch.equal(mx.scale_codes, scales)
            assert not bool(mx.decode().any())
    for invalid in (127, 255):
        bad = source_mx.codes.clone(); bad[0, 0, 0] = invalid
        rejected(lambda: a.code_chain(high.MXTensor(bad, source_mx.scale_codes, source_mx.shape)))
    rejected(lambda: a.code_chain(high.MXTensor(source_mx.codes, torch.full_like(source_mx.scale_codes, 255), source_mx.shape)))
    w = torch.zeros(2, 33, device=d); w[0, :5] = torch.tensor([7., 1.375, -.25, -.5, -0.], device=d)
    with forbidden_lower_encoders():
        chain = a.encode_chain(w)
    assert all(bool((x.codes[:, -1, 1:] == 0).all()) for x in chain.values())
    assert torch.equal(chain[4].codes, chain[6].codes >> 2)
    checks['all_255_scale_bytes_unchanged_padding_and_NaN_rejection'] = True

    coef = torch.linspace(-.7, .9, 66, device=d).reshape(2, 33)
    checks['exact_forward_and_identity_STE_by_mode'] = {}
    for bit in (8, 6, 4):
        master = torch.linspace(-2, 2, 66, device=d).reshape(2, 33).requires_grad_()
        independent = master.detach().clone().requires_grad_()
        q = a.JointMXQuantizer(); q.weight_mode = bit
        with forbidden_lower_encoders(): value = q(master)
        (value * coef).sum().backward()
        (a.effective_reference_weight(independent) * coef).sum().backward()
        assert torch.equal(master.grad, independent.grad)
        source_mx = high.encode_tensor(a.effective_reference_weight(master))
        lookup = torch.tensor([bitstring_oracle(i, bit) for i in range(256)], dtype=torch.uint8, device=d)
        oracle = a.codec(bit).MXTensor(lookup[source_mx.codes.long()], source_mx.scale_codes.clone(), source_mx.shape)
        assert torch.equal(value, oracle.decode())
        assert not q.state_dict()
        with torch.no_grad(): master[0, 5].add_(.25)
        changed_source = high.encode_tensor(a.effective_reference_weight(master))
        changed_oracle = a.codec(bit).MXTensor(lookup[changed_source.codes.long()], changed_source.scale_codes.clone(), changed_source.shape)
        assert torch.equal(q(master), changed_oracle.decode())
        checks['exact_forward_and_identity_STE_by_mode'][str(bit)] = True

    original, changed, _ = t.adapt_forward_text(source)
    class RemoveSelectors(ast.NodeTransformer):
        def visit_Expr(self, node):
            v = node.value
            if isinstance(v, ast.Call):
                if isinstance(v.func, ast.Name) and v.func.id == 'select_mode': return None
                if isinstance(v.func, ast.Attribute) and v.func.attr == 'apply': return None
            return self.generic_visit(node)
    assert ast.dump(RemoveSelectors().visit(ast.parse(original)), include_attributes=False) == ast.dump(
        RemoveSelectors().visit(ast.parse(changed)), include_attributes=False)
    checks['original_forward_AST_identical_except_three_selector_pairs'] = True
    checks['comparison_policy'] = {'production_backend': 'health mandatory; numerical replay diagnostic',
                                   'deterministic_backend': 'mandatory rtol1e-4 atol1e-5; CPU exact',
                                   'manual_reference': 'independent bitstring oracle, fresh E4 per pass',
                                   'source_cache': '6E4 encodings per minibatch; zero E2 encoder calls'}
    checks['training_comparisons'] = {}
    with forbidden_lower_encoders():
        prod = checks['training_comparisons']['production_backend'] = {}
        model = training_pair(modules, rt, source, log, prod)
        if d.type == 'cpu': require_pair_match(prod, exact=True)
        strict = checks['training_comparisons']['deterministic_backend'] = {}
        with deterministic_validation(d):
            model = training_pair(modules, rt, source, log, strict)
            require_pair_match(strict, exact=(d.type == 'cpu'))
    checks['two_threepass_updates_match_independent_manual_loop'] = True
    checks['three_BN_banks_update_FP32_masters_Adam_and_detached_teachers'] = True

    x = torch.linspace(-1, 1, 3 * 3 * 40 * 40, device=d).reshape(3, 3, 40, 40)
    model.eval()
    state = s.legacy.state_digest(model.state_dict())
    unchanged_snapshot = t.layer_snapshot(model)
    zero_updates = t.layer_updates(unchanged_snapshot, unchanged_snapshot, 'joint', 0)
    assert len(zero_updates) == 18 and all(r['master_changed_values'] == 0 for r in zero_updates)
    checks['zero_master_change_is_reported_not_a_failure'] = True
    with forbidden_lower_encoders(), count_high_encodings() as count:
        with a.shared_source_triplets(model) as audit:
            for bit in (4, 6, 8): a.select_mode(model, bit); model(x)
    assert count[0] == 6 and audit['source_triplets'] == 1 and audit['source_cache_cleared']
    assert s.legacy.state_digest(model.state_dict()) == state
    checks['eval_triplet_6_high_encodings_same_source_no_state_change'] = audit
    # Guards exercise real model hooks; every failure must clear its temporary cache.
    for failure in ('wrong_order', 'incomplete_cycle', 'master_mutation', 'code_mutation'):
        probe = copy.deepcopy(model).eval()
        def invalid_cycle():
            with a.shared_source_triplets(probe):
                a.select_mode(probe, 6 if failure == 'wrong_order' else 4)
                probe(x)
                if failure == 'incomplete_cycle': return
                if failure == 'master_mutation':
                    with torch.no_grad(): probe.get_submodule(a.TARGET_LAYERS[0]).weight.add_(.001)
                if failure == 'code_mutation':
                    probe.get_submodule(a.TARGET_LAYERS[0]).quantize_fn._chain[8].codes[0, 0, 0] ^= 1
                a.select_mode(probe, 6); probe(x)
                a.select_mode(probe, 8); probe(x)
        rejected(invalid_cycle)
        assert a.cache_is_empty(probe) and not probe._forward_pre_hooks and not probe._forward_hooks
    checks['order_incomplete_master_and_code_mutations_rejected_cache_cleanup'] = True

    with tempfile.TemporaryDirectory(prefix='mx_triple_trunc_source_check_') as temp:
        source_payload, rows = a.source_payload(model, {'synthetic_fixture_only': True})
        assert len(rows) == 18
        path = Path(temp) / 'source.pt'
        torch.save(source_payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix('.pt.sha256').write_text(digest, encoding='ascii')
        loaded_source = a.load_verified_payload(path)
        source_state = s.legacy.state_digest(loaded_source['auxiliary_fp32_state'])
        checks['export_logit_comparisons'] = []
        for bit in (8, 6, 4):
            a.select_mode(model, bit)
            with torch.no_grad(): expected = model(x)
            original_G, original_high = a.effective_reference_weight, high.encode_tensor
            def fail(*args, **kwargs): raise AssertionError('Master/encoder used during source-only reconstruction.')
            try:
                a.effective_reference_weight = fail; high.encode_tensor = fail
                with forbidden_lower_encoders():
                    payload = loaded_source if bit == 8 else a.derive_payload(loaded_source, bit, digest)
                    derived_path = Path(temp) / f'mode{bit}.pt'
                    torch.save(payload, derived_path)
                    derived_path.with_suffix('.pt.sha256').write_text(hashlib.sha256(derived_path.read_bytes()).hexdigest(), encoding='ascii')
                    reloaded = a.load_verified_payload(derived_path)
                    frozen = a.load_exported_model(modules.models, reloaded, d)
                    from_source = a.load_exported_model(modules.models, loaded_source, d, bit)
                    with torch.no_grad(): actual, via_source = frozen(x), from_source(x)
            finally:
                a.effective_reference_weight, high.encode_tensor = original_G, original_high
            comparison = tensor_comparison(expected, actual, f'export_mode{bit}', 1e-5, 1e-6)
            assert comparison['close']
            assert torch.equal(actual, via_source)
            assert not any(name + '.weight' in dict(frozen.named_parameters()) for name in a.TARGET_LAYERS)
            rejected(lambda: a.select_mode(frozen, 6 if bit == 8 else 8))
            checks['export_logit_comparisons'].append(comparison)
        assert source_state == s.legacy.state_digest(loaded_source['auxiliary_fp32_state'])
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
        invalid = copy.deepcopy(loaded_source); invalid['auxiliary_fp32_state'][a.TARGET_LAYERS[0] + '.weight'] = torch.zeros(1)
        rejected(lambda: a.validate_source(invalid))
        invalid = a.derive_payload(loaded_source, 6, digest); invalid['derivation']['bias_compensation'] = True
        rejected(lambda: a.validate_derived(invalid))
    checks['saved_reloaded_E4_source_and_both_derivations_master_free_encoder_free'] = True
    checks['no_SVHN_or_real_training'] = True
    checks['status'] = 'PASS'
    return checks


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
    for layer in a.TARGET_LAYERS:
        reference.get_submodule(layer).quantize_fn = IndependentQuantizer()
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
        with count_high_encodings() as encodings:
            actual = forward([(x, labels), (x, labels)], model, criterion, actual_soft, 0, True, opt, None)
        assert encodings[0] == 12
        assert a.cache_is_empty(model)
        details['high_encoder_calls'] = encodings[0]
        details['source_cache_audit'] = model._last_source_audit
        assert model._last_source_audit['source_triplets'] == 2
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
