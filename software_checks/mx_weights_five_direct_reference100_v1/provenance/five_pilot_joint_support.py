"""Explicit five-format joint training and read-only research instrumentation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
import joint_adapter as a
import svhn_support as s

TEACHERS = {'E4M3': [], 'E5M2': [], 'E2M3': ['E4M3'],
            'E3M2': ['E5M2'], 'E2M1': ['E2M3', 'E3M2']}
LOSS_WEIGHTS = {name: 1.0 for name in a.MODES}


def weight_format(mode='joint'):
    s.require(mode == 'joint', 'Deze proef gebruikt een gezamenlijk vijfformaatmodel.')
    return '+'.join(a.FORMATS.values())


def build_model(modules, mode, device):
    weight_format(mode)
    original = modules.models.svhnq(s.BITS, num_classes=10, expand=2).to(device)
    model = a.adapt_copy(original)
    assert_policy(model)
    return model


def assert_policy(model, mode='joint'):
    weight_format(mode)
    quantizers = [m for m in model.modules() if isinstance(m, a.JointMXQuantizer)]
    norms = [m for m in model.modules() if isinstance(m, a.FormatBatchNorm)]
    s.require(len(quantizers) == 6 and len(norms) == 5, 'Verkeerde MX-/BN-laagselectie.')
    s.require(model.weight_format in a.MODES and all(m.weight_format == model.weight_format for m in quantizers + norms),
              'Formaatselectie inconsistent.')
    s.require(all(set(m.bn_dict) == set(a.MODES) for m in norms), 'BN-banken moeten formaatgebaseerd zijn.')
    s.require(all(getattr(m, 'abit', 32) == 32 and getattr(m, 'wbit', 32) == 32 for m in model.modules()),
              'Legacy activatieselector gewijzigd; A32 vereist.')
    s.require(all(p.dtype == torch.float32 for p in model.parameters()), 'Masterdtype gewijzigd.')


def soft_cross_entropy(logits, target):
    """Same batch-mean bmm arithmetic as upstream CrossEntropyLossSoft."""
    s.require(not target.requires_grad and target.grad_fn is None, 'Teacherdoel is niet losgekoppeld.')
    output = F.log_softmax(logits, dim=1)
    return -torch.bmm(target.view(target.size(0), 1, -1),
                      output.view(output.size(0), -1, 1)).mean()


def validate_gradients(model):
    gradients = []
    for name, p in model.named_parameters():
        s.require(p.grad is not None and p.grad.dtype == torch.float32, 'Ontbrekende/onjuiste gradient: ' + name)
        gradients.append(p.grad.detach())
    s.require(gradients and bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()),
              'Niet-eindige gradiënten.')
    norm = float(torch.stack([g.norm() for g in gradients]).norm())
    s.require(math.isfinite(norm), 'Niet-eindige gradientnorm.')
    return {'active_gradient_tensors': len(gradients), 'l2_norm': norm}


def validate_bn(model, steps):
    counts = {n: int(v) for n, v in model.state_dict().items() if n.endswith('num_batches_tracked')}
    s.require(len(counts) == 25 and all(v == steps for v in counts.values()), 'Onjuiste formaatgebaseerde BN-tellers.')
    return {'all_counters_correct': True, 'counts': counts, 'active_banks_per_layer': 5}


def validate_adam_state(model, optimizer, steps):
    for name, p in model.named_parameters():
        state = optimizer.state[p]
        s.require(int(state['step'].item()) == steps, 'Adam-teller wijkt af: ' + name)
        s.require(all(state[k].dtype == torch.float32 and bool(torch.isfinite(state[k]).all())
                      for k in ('exp_avg', 'exp_avg_sq')), 'Onjuiste Adam-momenten: ' + name)
    return {'all_active_adam_steps_correct': True, 'moments_fp32_finite': True,
            'active_parameter_tensors': len(list(model.parameters()))}


def _record(name, logits, labels, loss, train):
    with torch.no_grad():
        s.require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)), 'Niet-eindige logits/loss.')
        prediction = logits.detach().topk(5, dim=1).indices
        correct = prediction.eq(labels[:, None])
        n = int(labels.numel())
        c1, c5 = int(correct[:, 0].sum()), int(correct.any(dim=1).sum())
        return {'weight_format': name, 'weight_bits': a.BITS[name], 'activation_bits': 32,
                'n': n, 'correct1': c1, 'correct5': c5, 'top1_percent': 100.0 * c1 / n,
                'top5_percent': 100.0 * c5 / n, 'objective_loss': float(loss.detach()),
                'hard_ce': float(F.cross_entropy(logits.detach(), labels)),
                'objective_type': 'soft_cross_entropy' if train and TEACHERS[name] else 'hard_cross_entropy',
                'teacher_formats': '+'.join(TEACHERS[name]) if train else '',
                'loss_weight': LOSS_WEIGHTS[name]}, prediction[:, 0]


def train_batch(model, optimizer, x, y, collect_trace=False):
    """Five sequential forward/backward passes, followed by one Adam step."""
    assert_policy(model)
    model.train()
    optimizer.zero_grad()
    probabilities, rows, trace = {}, [], {}
    soft_checks = 0
    for name in a.MODES:
        a.select_mode(model, name)
        logits = model(x)
        if name in ('E4M3', 'E5M2'):
            target = y
            loss = F.cross_entropy(logits, y)
        else:
            target = ((probabilities['E2M3'] + probabilities['E3M2']) * 0.5
                      if name == 'E2M1' else probabilities[TEACHERS[name][0]])
            loss = soft_cross_entropy(logits, target)
            soft_checks += 1
        probabilities[name] = F.softmax(logits.detach(), dim=1)
        row, _ = _record(name, logits, y, loss, True)
        rows.append(row)
        if collect_trace:
            trace[name] = {'logits': logits.detach().clone(), 'target': target.detach().clone(),
                           'loss': float(loss.detach())}
        (LOSS_WEIGHTS[name] * loss).backward()
    gradients = validate_gradients(model)
    optimizer.step()
    return {'rows': rows, 'audit': {'mode_order': list(a.MODES), 'optimizer_steps': 1,
            'forwards': 5, 'backwards': 5, 'hard_label_losses': 2,
            'detached_teacher_target_checks': soft_checks, 'loss_weights': LOSS_WEIGHTS.copy(),
            'loss_reduction': 'sum', 'temperature': 1.0, 'gradient_l2': gradients['l2_norm'],
            'active_gradient_tensors': gradients['active_gradient_tensors']}, 'trace': trace}


@torch.no_grad()
def evaluate_batch(model, x, y, collect_predictions=False):
    assert_policy(model)
    model.eval()
    rows, predictions = [], {}
    for name in a.EVAL_ORDER:
        a.select_mode(model, name)
        logits = model(x)
        row, top1 = _record(name, logits, y, F.cross_entropy(logits, y), False)
        rows.append(row)
        if collect_predictions:
            predictions[name] = top1.detach().cpu()
    return {'rows': rows, 'predictions': predictions}


def phase(rt, model, optimizer, loader, epoch, phase_name, lr, out, log):
    s.require(phase_name in ('train', 'test'), 'Onbekende fase.')
    train = phase_name == 'train'
    sums = {f: {'n': 0, 'correct1': 0, 'correct5': 0, 'objective_loss': 0.0, 'hard_ce': 0.0} for f in a.MODES}
    audit = {'optimizer_steps_in_phase': 0, 'finite_gradient_checks': 0,
             'detached_teacher_target_checks': 0, 'gradient_l2_min': None, 'gradient_l2_max': None,
             'forward_calls': {f: 0 for f in a.MODES}, 'no_extra_FP32_teacher': True}
    predictions, targets = {f: [] for f in a.MODES}, []
    if rt.device.type == 'cuda': torch.cuda.synchronize(rt.device)
    start = time.perf_counter()
    with (out / f'{phase_name}_batches_epoch_{epoch:05d}.jsonl').open('x', encoding='utf-8') as file:
        for batch, (x, y) in enumerate(loader, 1):
            x, y = x.to(rt.device), y.to(rt.device, non_blocking=True)
            result = train_batch(model, optimizer, x, y) if train else evaluate_batch(model, x, y, True)
            for row in result['rows']:
                f = row['weight_format']; acc = sums[f]; n = row['n']
                for key in ('n', 'correct1', 'correct5'): acc[key] += row[key]
                for key in ('objective_loss', 'hard_ce'): acc[key] += row[key] * n
                audit['forward_calls'][f] += 1
            if train:
                b = result['audit']; norm = b['gradient_l2']
                s.require(b['mode_order'] == list(a.MODES) and b['optimizer_steps'] == 1, 'Onjuiste gezamenlijke stap.')
                audit['optimizer_steps_in_phase'] += b['optimizer_steps']
                audit['finite_gradient_checks'] += 1
                audit['detached_teacher_target_checks'] += b['detached_teacher_target_checks']
                audit['gradient_l2_min'] = norm if audit['gradient_l2_min'] is None else min(norm, audit['gradient_l2_min'])
                audit['gradient_l2_max'] = norm if audit['gradient_l2_max'] is None else max(norm, audit['gradient_l2_max'])
            else:
                targets.append(y.detach().cpu())
                for f in a.MODES: predictions[f].append(result['predictions'][f])
            file.write(json.dumps({'epoch': epoch, 'phase': phase_name, 'batch': batch, 'lr': lr,
                                   'precisions': result['rows']}, allow_nan=False) + '\n')
            if batch == 1 or batch % 20 == 0 or batch == len(loader):
                file.flush()
                log(f'Epoch {epoch}/5 {phase_name}: minibatch {batch}/{len(loader)}; vijf formaten.')
    if rt.device.type == 'cuda': torch.cuda.synchronize(rt.device)
    elapsed = time.perf_counter() - start
    rows = []
    for f in a.MODES:
        acc = sums[f]; n = acc['n']
        s.require(n > 0, 'Lege fase.')
        rows.append({'epoch': epoch, 'phase': phase_name, 'weight_format': f, 'weight_bits': a.BITS[f],
                     'activation_bits': 32, 'n': n, 'correct1': acc['correct1'], 'correct5': acc['correct5'],
                     'top1_percent': 100 * acc['correct1'] / n, 'top5_percent': 100 * acc['correct5'] / n,
                     'objective_loss': acc['objective_loss'] / n, 'hard_ce': acc['hard_ce'] / n,
                     'objective_type': 'soft_cross_entropy' if train and TEACHERS[f] else 'hard_cross_entropy',
                     'teacher_formats': '+'.join(TEACHERS[f]) if train else '', 'loss_weight': 1.0, 'lr': lr})
    expected = len(loader) if train else 0
    s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == expected
              and audit['detached_teacher_target_checks'] == 3 * expected
              and all(c == len(loader) for c in audit['forward_calls'].values()), 'Onvolledige fase-audit.')
    if not train:
        import numpy as np
        arrays = {f: torch.cat(predictions[f]).numpy().astype(np.int16) for f in a.MODES}
        arrays['labels'] = torch.cat(targets).numpy().astype(np.int16)
        arrays['sample_indices'] = np.arange(len(arrays['labels']), dtype=np.int32)
        for row in rows:
            s.require(int((arrays[row['weight_format']] == arrays['labels']).sum()) == row['correct1'],
                      'Voorspellingen en gerapporteerde accuracy verschillen.')
        with (out / f'test_predictions_epoch_{epoch:05d}.npz').open('xb') as file:
            np.savez_compressed(file, **arrays)
    return rows, elapsed, audit


@torch.no_grad()
def layer_snapshot(model):
    result = {}
    for layer in a.TARGET_LAYERS:
        w = model.get_submodule(layer).weight
        effective = a.effective_reference_weight(w)
        entry = {'master': w.detach().cpu().clone()}
        for f in a.MODES:
            mx = a.codec(f).encode_tensor(effective)
            entry[f] = {'codes': mx.codes.cpu(), 'scales': mx.scale_codes.cpu()}
        s.require(torch.equal(entry['E2M3']['scales'], entry['E2M1']['scales']), 'E2-schalen verschillen bij dezelfde bron.')
        result[layer] = entry
    return result


def layer_updates(before, after, epoch):
    rows = []
    for layer in a.TARGET_LAYERS:
        b, e = before[layer], after[layer]
        delta = e['master'] - b['master']
        for f in a.MODES:
            rows.append({'epoch': epoch, 'layer': layer, 'weight_format': f,
                         'master_l2_change': float(delta.norm()), 'master_max_abs_change': float(delta.abs().max()),
                         'changed_element_codes': int((b[f]['codes'] != e[f]['codes']).sum()),
                         'element_codes_including_padding': int(e[f]['codes'].numel()),
                         'changed_scale_codes': int((b[f]['scales'] != e[f]['scales']).sum()),
                         'scale_codes': int(e[f]['scales'].numel())})
    return rows
