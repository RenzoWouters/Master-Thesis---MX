"""Independent synthetic checks for five-format direct joint QAT.

No SVHN examples are opened and no research accuracy is measured.  The manual
reference below deliberately does not call the production training-step or
loss helpers.  It reconstructs the agreed objective from the five forwards.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import os

import torch
import torch.nn.functional as F

import joint_adapter as a
import joint_support as t
import svhn_support as s


EXPECTED_MODES = ('E4M3', 'E5M2', 'E2M3', 'E3M2', 'E2M1')
EXPECTED_EVAL = ('E2M1', 'E2M3', 'E3M2', 'E4M3', 'E5M2')
OLD_BANK = {'E4M3': '8', 'E5M2': '1', 'E2M3': '32', 'E3M2': '2', 'E2M1': '4'}
EXPECTED_BITS = {'E4M3': 8, 'E5M2': 8, 'E2M3': 6, 'E3M2': 6, 'E2M1': 4}


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError, TypeError, KeyError):
        return
    raise AssertionError('Ongeldige invoer werd ten onrechte aanvaard.')


def rng_state():
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_equal(left, right):
    return (torch.equal(left['cpu'], right['cpu'])
            and len(left['cuda']) == len(right['cuda'])
            and all(torch.equal(x, y) for x, y in zip(left['cuda'], right['cuda'])))


def backend_state():
    return {'benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


@contextmanager
def deterministic_validation(device):
    """Change only check-time execution and always restore the production policy."""
    before = backend_state()
    if device.type == 'cuda':
        s.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'),
                  'Deterministische zelftest vereist CUBLAS_WORKSPACE_CONFIG vóór CUDA-initialisatie.')
        torch.cuda.synchronize(device)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(before['algorithms'], warn_only=before['warn_only'])
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        torch.backends.cudnn.benchmark = before['benchmark']


def compare_tensors(expected, actual, label, device, rtol=None, atol=None):
    s.require(isinstance(actual, torch.Tensor), label + ': geen tensor.')
    s.require(expected.shape == actual.shape and expected.dtype == actual.dtype,
              label + ': vorm of dtype wijkt af.')
    s.require(bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()),
              label + ': niet-eindige tensor.')
    if rtol is None:
        rtol, atol = (0., 0.) if device.type == 'cpu' else (1e-4, 1e-5)
    # Integer counters must always match exactly.
    if not expected.is_floating_point():
        rtol = atol = 0.
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False,
                               msg=label)
    difference = (expected.detach().double() - actual.detach().double()).abs()
    return {'exact': bool(torch.equal(expected, actual)),
            'max_abs_difference': float(difference.max()) if difference.numel() else 0.,
            'rtol': rtol, 'atol': atol}


def compare_model(left, right, device):
    a_state, b_state = left.state_dict(), right.state_dict()
    s.require(list(a_state) == list(b_state), 'Modelstaatstructuren verschillen.')
    summary = {'checked_tensors': 0, 'all_exact': True, 'max_abs_difference': 0.}
    for name, tensor in a_state.items():
        result = compare_tensors(tensor, b_state[name], name, device)
        summary['checked_tensors'] += 1
        summary['all_exact'] &= result['exact']
        summary['max_abs_difference'] = max(summary['max_abs_difference'], result['max_abs_difference'])
    return summary


def compare_optimizer(left_model, left_opt, right_model, right_opt, device):
    left = dict(left_model.named_parameters())
    right = dict(right_model.named_parameters())
    s.require(list(left) == list(right), 'Parameterstructuren verschillen.')
    summary = {'checked_parameter_states': 0, 'all_exact': True, 'max_abs_difference': 0.}
    for name in left:
        ls, rs = left_opt.state[left[name]], right_opt.state[right[name]]
        s.require(set(ls) == set(rs) == {'step', 'exp_avg', 'exp_avg_sq'},
                  'Onvolledige of onverwachte Adamstaat voor ' + name)
        for key in ls:
            result = compare_tensors(ls[key], rs[key], name + '/' + key, device)
            summary['all_exact'] &= result['exact']
            summary['max_abs_difference'] = max(summary['max_abs_difference'], result['max_abs_difference'])
        summary['checked_parameter_states'] += 1
    return summary


def manual_step(model, optimizer, x, labels):
    """Independent explicit objective, with upstream soft-CE arithmetic."""
    optimizer.zero_grad()
    probabilities = {}
    trace = {}
    for name in EXPECTED_MODES:
        a.select_mode(model, name)
        logits = model(x)
        if name in ('E4M3', 'E5M2'):
            target = labels
            loss = F.cross_entropy(logits, labels)
        else:
            if name == 'E2M3':
                target = probabilities['E4M3']
            elif name == 'E3M2':
                target = probabilities['E5M2']
            else:
                target = .5 * probabilities['E2M3'] + .5 * probabilities['E3M2']
            s.require(not target.requires_grad and target.grad_fn is None,
                      'Handmatige teacher moet losgekoppeld zijn.')
            log_probability = F.log_softmax(logits, dim=1)
            loss = -torch.bmm(target.unsqueeze(1), log_probability.unsqueeze(2)).mean()
        trace[name] = {'logits': logits.detach().clone(), 'target': target.detach().clone(),
                       'loss': float(loss.detach())}
        loss.backward()
        probabilities[name] = F.softmax(logits.detach(), dim=1)
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    optimizer.step()
    return trace, gradients


def validate_initialization(modules, device):
    torch.manual_seed(20261009)
    original = modules.models.svhnq(s.BITS, num_classes=10, expand=2).to(device)
    expected = original.state_dict()
    after_original_rng = rng_state()
    torch.manual_seed(20261009)
    model = t.build_model(modules, 'joint', device)
    s.require(rng_equal(after_original_rng, rng_state()),
              'Aanpassing naar vijf formaten verbruikt extra initialisatie-RNG.')
    actual = model.state_dict()
    renamed = {}
    for name, tensor in expected.items():
        new_name = name
        if '.bn_dict.' in name:
            prefix, rest = name.split('.bn_dict.', 1)
            old, suffix = rest.split('.', 1)
            candidates = [f for f, bank in OLD_BANK.items() if bank == old]
            s.require(len(candidates) == 1, 'Onbekende oude BN-bank.')
            new_name = prefix + '.bn_dict.' + candidates[0] + '.' + suffix
        renamed[new_name] = tensor
    s.require(set(renamed) == set(actual), 'Formaatbenaming heeft extra/ontbrekende modelstaat.')
    for name, tensor in renamed.items():
        s.require(torch.equal(tensor, actual[name]), 'Initialisatie gewijzigd: ' + name)
    parameters = sum(p.numel() for p in model.parameters())
    s.require(parameters == 132954, 'Onverwacht aantal modelparameters.')
    master_ids = {name: model.get_submodule(name).weight.data_ptr() for name in a.TARGET_LAYERS}
    for fmt in EXPECTED_MODES:
        a.select_mode(model, fmt)
        t.assert_policy(model)
        s.require(master_ids == {name: model.get_submodule(name).weight.data_ptr()
                                 for name in a.TARGET_LAYERS},
                  'Formaatwissel vervangt de gedeelde mastergewichten.')
    norms = [m for m in model.modules() if hasattr(m, 'bn_dict')]
    s.require(len(norms) == 5 and all(set(m.bn_dict) == set(EXPECTED_MODES) for m in norms),
              'Verwacht vijf BN-lagen met vijf formaatspecifieke banken.')
    for norm in norms:
        s.require(len({norm.bn_dict[f].weight.data_ptr() for f in EXPECTED_MODES}) == 5,
                  'BN-banken delen onbedoeld trainbare parameters.')
    return {'all_original_values_preserved_after_BN_key_rename': True,
            'same_initialization_RNG': True, 'parameter_count': parameters,
            'BN_layers': 5, 'active_BN_banks_per_layer': 5,
            'one_FP32_master_per_MX_layer': True, 'MX_layers': list(a.TARGET_LAYERS)}


def validate_direct_and_ste(device):
    results = {}
    coef = torch.linspace(-.7, .9, 66, device=device).reshape(2, 33)
    for name in EXPECTED_MODES:
        master = torch.linspace(-2, 2, 66, device=device).reshape(2, 33).requires_grad_()
        reference = master.detach().clone().requires_grad_()
        unchanged = master.detach().clone()
        quantizer = a.JointMXQuantizer()
        quantizer.weight_format = name
        result = quantizer(master)
        expected = a.codec(name).encode_tensor(a.effective_reference_weight(master)).decode()
        s.require(torch.equal(result.detach(), expected), name + ': geen directe G(W)-kwantisatie.')
        (result * coef).sum().backward()
        (a.effective_reference_weight(reference) * coef).sum().backward()
        s.require(torch.equal(master.grad, reference.grad), name + ': identity-STE/G(W)-afgeleide gewijzigd.')
        s.require(torch.equal(master.detach(), unchanged), name + ': forward overschrijft mastergewichten.')
        with torch.no_grad():
            # Cross a sign boundary by a large margin: a small perturbation can
            # legitimately remain in the same low-precision quantization bin.
            master[0, 5] += 3.
        fresh = quantizer(master)
        expected_fresh = a.codec(name).encode_tensor(a.effective_reference_weight(master)).decode()
        s.require(torch.equal(fresh.detach(), expected_fresh), name + ': verouderde kwantisatiecache.')
        s.require(not torch.equal(result.detach(), fresh.detach()), name + ': masterwijziging is niet zichtbaar.')
        results[name] = {'direct_same_GW': True, 'identity_STE_exact': True,
                         'master_unchanged_by_forward': True, 'fresh_after_master_change': True}
    for invalid in (True, 8, 6, 4, 32, '8', 'unknown'):
        rejected(lambda invalid=invalid: a.codec(invalid))
    # Fixed binary witness: a lower format must not be obtained by quantizing
    # an already rounded higher-format result or shifting its mantissa bits.
    witness = torch.zeros(1, 32, device=device)
    witness[0, :3] = torch.tensor([7., 47. / 256., 1.375], device=device)
    direct6 = a.codec('E2M3').encode_tensor(witness)
    through8 = a.codec('E2M3').encode_tensor(a.codec('E4M3').encode_tensor(witness).decode())
    direct4 = a.codec('E2M1').encode_tensor(witness)
    shifted4 = a.codec('E2M1').decode_elements(direct6.codes >> 2)
    s.require(float(direct6.decode()[0, 1]) == .125 and float(through8.decode()[0, 1]) == .25,
              'Direct/dubbel afgerond negatief controlevoorbeeld mislukt.')
    s.require(float(direct4.decode()[0, 2]) == 1.5 and float(shifted4[0, 0, 2]) == 1.,
              'Direct/truncatie negatief controlevoorbeeld mislukt.')
    return {'formats': results, 'bitwidth_only_selectors_rejected': True,
            'direct_quantization_negative_controls': True}


def training_pair(modules, device):
    torch.manual_seed(71203)
    tested = t.build_model(modules, 'joint', device).train()
    manual = copy.deepcopy(tested).train()
    tested_opt = modules.optimizer.get_optimizer_config(tested, 'adam', .001, 0.)
    manual_opt = modules.optimizer.get_optimizer_config(manual, 'adam', .001, 0.)
    x = torch.linspace(-1, 1, 3 * 3 * 40 * 40, device=device).reshape(3, 3, 40, 40)
    labels = torch.tensor([2, 5, 9], dtype=torch.long, device=device)
    records = []
    last_trace = None
    for step in (1, 2):
        start_rng = rng_state()
        forward_order, pre_step_gradients, call_counts = [], {}, {'zero_grad': 0, 'step': 0}
        master_before = {name: tested.get_submodule(name).weight.detach().clone() for name in a.TARGET_LAYERS}
        original_zero_grad = tested_opt.zero_grad

        def checked_zero_grad(*args, **kwargs):
            call_counts['zero_grad'] += 1
            return original_zero_grad(*args, **kwargs)

        def before_forward(model, inputs):
            forward_order.append(model.weight_format)
            s.require(inputs[0].data_ptr() == x.data_ptr(), 'Niet dezelfde minibatch in alle forwards.')
            for name in a.TARGET_LAYERS:
                s.require(torch.equal(master_before[name], model.get_submodule(name).weight),
                          'Master gewijzigd vóór gezamenlijke optimizerupdate.')

        def before_step(optimizer, args, kwargs):
            call_counts['step'] += 1
            s.require(tuple(forward_order) == EXPECTED_MODES,
                      'Adamupdate vóór de vijf afgesproken forwards of met extra teacherpass.')
            for name, param in tested.named_parameters():
                s.require(param.grad is not None and bool(torch.isfinite(param.grad).all()),
                          'Ontbrekende/niet-eindige gradient: ' + name)
                pre_step_gradients[name] = param.grad.detach().clone()

        tested_opt.zero_grad = checked_zero_grad
        hooks = [tested.register_forward_pre_hook(before_forward),
                 tested_opt.register_step_pre_hook(before_step)]
        try:
            result = t.train_batch(tested, tested_opt, x, labels, collect_trace=True)
        finally:
            tested_opt.zero_grad = original_zero_grad
            for hook in hooks:
                hook.remove()
        tested_end_rng = rng_state()
        s.require(call_counts == {'zero_grad': 1, 'step': 1}, 'Niet exact één zero_grad en één optimizerupdate.')
        s.require(isinstance(result, dict) and len(result['rows']) == 5,
                  'Trainingsstap levert niet vijf modusmetrics.')
        s.require(tuple(row['weight_format'] for row in result['rows']) == EXPECTED_MODES,
                  'Metrics hebben onjuiste formaatnamen/volgorde.')
        actual_trace = result['trace']
        s.require(set(actual_trace) == set(EXPECTED_MODES), 'Onvolledige vijf-modustrace.')
        # These targets must be literal, equally weighted probabilities and
        # detached, not averaged logits or a second complete E2M1 loss.
        for name in ('E2M3', 'E3M2', 'E2M1'):
            target = actual_trace[name]['target']
            s.require(not target.requires_grad and target.grad_fn is None,
                      'Niet-losgekoppeld teacherdoel in trace.')
        expected_low = .5 * F.softmax(actual_trace['E2M3']['logits'], 1) + .5 * F.softmax(actual_trace['E3M2']['logits'], 1)
        torch.testing.assert_close(actual_trace['E2M1']['target'], expected_low, rtol=0., atol=0.)
        restore_rng(start_rng)
        expected_trace, manual_gradients = manual_step(manual, manual_opt, x, labels)
        s.require(rng_equal(tested_end_rng, rng_state()), 'Forwardaantal/dropout-RNG wijkt af van referentie.')
        loss_differences = {}
        for name in EXPECTED_MODES:
            compare_tensors(expected_trace[name]['logits'], actual_trace[name]['logits'], name + '/logits', device)
            compare_tensors(expected_trace[name]['target'], actual_trace[name]['target'], name + '/target', device)
            difference = abs(expected_trace[name]['loss'] - float(actual_trace[name]['loss']))
            tol = 0. if device.type == 'cpu' else 1e-5 + 1e-4 * abs(expected_trace[name]['loss'])
            s.require(math.isfinite(difference) and difference <= tol,
                      name + ': verlies verschilt van onafhankelijke referentie.')
            loss_differences[name] = difference
        for name, expected in manual_gradients.items():
            compare_tensors(expected, pre_step_gradients[name], name + '/summed_gradient', device)
        model_compare = compare_model(manual, tested, device)
        optimizer_compare = compare_optimizer(manual, manual_opt, tested, tested_opt, device)
        t.assert_policy(tested)
        bn = t.validate_bn(tested, step)
        adam = t.validate_adam_state(tested, tested_opt, step)
        records.append({'step': step, 'zero_grad_calls': 1, 'optimizer_steps': 1,
                        'forward_order': list(forward_order), 'teacher_targets_detached': True,
                        'probability_mixture_50_50_exact': True, 'dropout_RNG_matches': True,
                        'loss_absolute_differences': loss_differences,
                        'model_comparison': model_compare, 'optimizer_comparison': optimizer_compare,
                        'BN': bn, 'Adam': adam})
        last_trace = actual_trace
    return tested, x, labels, records


def validate_eval_and_exports(model, x, labels, modules, device):
    model.eval()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    order = []
    hook = model.register_forward_pre_hook(lambda module, inputs: order.append(module.weight_format))
    try:
        result = t.evaluate_batch(model, x, labels, collect_predictions=True)
    finally:
        hook.remove()
    s.require(tuple(order) == EXPECTED_EVAL, 'Verkeerde evaluatievolgorde of extra evaluatiepass.')
    s.require(isinstance(result, dict) and len(result['rows']) == 5,
              'Evaluatie moet vijf modussen rapporteren.')
    for name, tensor in model.state_dict().items():
        s.require(torch.equal(before[name], tensor), 'Evaluatie wijzigt modelstaat: ' + name)
    export_checks = {}
    with torch.no_grad():
        for fmt in EXPECTED_MODES:
            a.select_mode(model, fmt)
            expected = model(x)
            rows = [r for r in result['rows'] if r['weight_format'] == fmt]
            s.require(len(rows) == 1, 'Formaat ontbreekt of komt dubbel voor in evaluatie.')
            correct = expected.topk(5, dim=1).indices.eq(labels[:, None])
            s.require(rows[0]['correct1'] == int(correct[:, 0].sum()), 'Top1-telling wijkt af.')
            s.require(rows[0]['correct5'] == int(correct.any(dim=1).sum()), 'Top5-telling wijkt af.')
            expected_ce = float(F.cross_entropy(expected, labels))
            s.require(math.isclose(float(rows[0]['objective_loss']), expected_ce, rel_tol=1e-6, abs_tol=1e-7),
                      'Evaluatie gebruikt niet de harde label-CE.')
            payload, distortion_rows = a.export_payload(model, fmt, {'synthetic_fixture_only': True})
            s.require(payload['truncation'] is False, 'Directe export noemt truncatie.')
            auxiliary = payload['auxiliary_fp32_state']
            s.require(not any(name + '.weight' in auxiliary for name in a.TARGET_LAYERS),
                      'Compacte export bevat MX-mastergewichten.')
            frozen = a.load_exported_model(modules.models, payload, device)
            actual = frozen(x)
            comparison = compare_tensors(expected, actual, fmt + '/export_logits', device, rtol=1e-5, atol=1e-6)
            s.require(not any(name + '.weight' in dict(frozen.named_parameters()) for name in a.TARGET_LAYERS),
                      'Gereconstrueerde export bevat nog MX-masterparameters.')
            for name in a.TARGET_LAYERS:
                original = model.get_submodule(name)
                decoded = frozen.get_submodule(name).decoded_weight
                s.require(torch.equal(original.quantize_fn(original.weight), decoded),
                          fmt + ': exportgewicht wijkt af in ' + name)
                s.require(decoded.device == device, 'Export op verkeerd device.')
            export_checks[fmt] = {**comparison, 'MX_masters_absent': True,
                                  'decoded_weights_exact': True, 'MX_layers': len(distortion_rows)}
    return {'evaluation_preserves_BN_and_all_state': True, 'hard_label_CE_all_formats': True,
            'evaluation_order': list(EXPECTED_EVAL), 'exports': export_checks}


def run_tests(modules, rt, source, log, checks=None):
    """Run the synthetic contract checks on the actual selected device."""
    if checks is None:
        checks = {}
    initial_rng = rng_state()
    initial_backend = backend_state()
    try:
        s.require(tuple(a.MODES) == EXPECTED_MODES and tuple(a.EVAL_ORDER) == EXPECTED_EVAL,
                  'Verkeerde formaat- of evaluatievolgorde.')
        s.require(a.BITS == EXPECTED_BITS, 'Formaat/bitbreedtemapping wijkt af.')
        s.require(len({a.FORMATS[f] for f in EXPECTED_MODES}) == 5,
                  'De vijf modi hebben geen afzonderlijke formaatnamen.')
        checks['initialization'] = validate_initialization(modules, rt.device)
        checks['direct_quantization_and_STE'] = validate_direct_and_ste(rt.device)
        log('INTEGRATIE PASS: vijf formaatmodi, oorspronkelijke masters, A32 en directe G(W)-kwantisatie.')
        with deterministic_validation(rt.device):
            model, x, labels, records = training_pair(modules, rt.device)
            checks['independent_training_comparison'] = {
                'updates_per_model': 2, 'passes_per_update': 5,
                'CPU_exact': rt.device.type == 'cpu', 'CUDA_rtol': 1e-4, 'CUDA_atol': 1e-5,
                'records': records}
            checks['evaluation_and_exports'] = validate_eval_and_exports(model, x, labels, modules, rt.device)
        s.require(backend_state() == initial_backend, 'Zelftest heeft backendinstellingen gewijzigd.')
        checks.update(status='PASS', backend_policy_restored=True, no_SVHN_data=True,
                      synthetic_checks_only=True, no_research_accuracy_measured=True)
        log('TRAININGSSTAP PASS: twee gezamenlijke updates stemmen overeen met onafhankelijke referentie; vijf exports gecontroleerd.')
        return checks
    except BaseException as exc:
        checks.update(status='FAIL', error=repr(exc))
        raise
    finally:
        restore_rng(initial_rng)
