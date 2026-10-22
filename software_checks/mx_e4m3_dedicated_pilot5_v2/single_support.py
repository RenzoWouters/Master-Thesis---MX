"""One-mode extension of the validated SVHN infrastructure.

The upstream forward AST and optimizer remain unchanged. Only the weight
codec changes from E2M3 to E4M3; the identity-STE policy is retained.
The internal upstream selector '32' selects A32 + BN32 + W32 preprocessing.
The actual weight format is controlled independently by MXWeightQuantizer.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import svhn_support as s
import mx_adapter as a
import mx_e4m3 as f


def weight_format(mode: str) -> str:
    s.require(mode in ('mx', 'control'), 'Onbekende modus.')
    return 'MXFP8_E4M3' if mode == 'mx' else 'W32_effective'


def inactive_bn(name: str) -> bool:
    return '.bn_dict.' in name and '.bn_dict.32.' not in name


def build_model(modules, mode: str, device):
    weight_format(mode)  # Reject an unknown mode before it could become a control.
    # Preserve topology and parameter initialization from the reference. Banks
    # 1/2/4/8 remain present but unused (no gradients or optimizer state).
    original = modules.models.svhnq(s.BITS, num_classes=10, expand=2).to(device)
    model = a.adapt_copy(original, enabled=mode == 'mx')
    s.require(s.legacy.state_digest(original.state_dict()) == s.legacy.state_digest(model.state_dict()),
              'Integratie heeft de initialisatie veranderd.')
    s.require(all(p.dtype == torch.float32 for p in model.parameters()), 'Masters zijn niet FP32.')
    return model


def assert_policy(model, mode: str):
    weight_format(mode)
    quantizers = [m for m in model.modules() if isinstance(m, a.MXWeightQuantizer)]
    s.require(len(quantizers) == 6 and all(q.enabled == (mode == 'mx') for q in quantizers),
              'Verkeerde MX-laagselectie of modus.')
    s.require(all(getattr(m, 'abit', 32) == 32 and getattr(m, 'wbit', 32) == 32
                  for m in model.modules()), 'De A32/BN32-selector is veranderd.')
    s.require(all(p.dtype == torch.float32 for p in model.parameters()), 'Masterdtype gewijzigd.')


def validate_gradients(model) -> dict:
    active = []
    for name, p in model.named_parameters():
        if inactive_bn(name):
            s.require(p.grad is None, 'Gradient in ongebruikte BN-bank: ' + name)
        else:
            s.require(p.grad is not None, 'Ontbrekende actieve gradient: ' + name)
            s.require(p.grad.dtype == torch.float32, 'Gradient niet FP32: ' + name)
            active.append(p.grad.detach())
    s.require(active and bool(torch.stack([torch.isfinite(g).all() for g in active]).all()),
              'Ontbrekende of niet-eindige gradients.')
    norm = float(torch.stack([g.norm() for g in active]).norm())
    s.require(math.isfinite(norm), 'Niet-eindige totale gradientnorm.')
    return {'active_gradient_tensors': len(active), 'l2_norm': norm}


def validate_bn(model, steps: int) -> dict:
    counts = {n: int(v) for n, v in model.state_dict().items() if n.endswith('num_batches_tracked')}
    s.require(len(counts) == 25, 'Verwacht vijf lagen met vijf BN-banken.')
    for name, value in counts.items():
        expected = 0 if inactive_bn(name) else steps
        s.require(value == expected, f'BN-teller {name}: {value}, verwacht {expected}.')
    return {'all_counters_correct': True, 'counts': counts,
            'note': 'BN32 leert een keer per minibatch; overige banken blijven ongebruikt.'}


def validate_adam_state(model, optimizer, steps: int) -> dict:
    count = 0
    for name, p in model.named_parameters():
        if inactive_bn(name):
            s.require(p not in optimizer.state, 'Ongebruikte BN-bank heeft optimizerstaat: ' + name)
            continue
        state = optimizer.state[p]
        s.require(int(state['step'].item()) == steps, 'Verkeerde Adam-stapteller: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            s.require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                      'Niet-eindige of niet-FP32 Adamstaat: ' + name + '/' + key)
        count += 1
    return {'active_parameter_tensors': count, 'all_active_adam_steps_correct': True,
            'moments_fp32_finite': True, 'unused_bn_has_no_optimizer_state': True}


class SingleMeterRecorder(s.MeterRecorder):
    """Same upstream AverageMeter arithmetic; just 3 meters instead of 15."""
    def __init__(self, base_class, epoch, phase, lr, path, mode):
        super().__init__(base_class, epoch, phase, lr, path)
        self.mode = mode
        self.correct1, self.correct5 = [0], [0]

    def updated(self, position, val, n):
        s.require(position not in self.pending, 'Dubbele batchmetric.')
        self.pending[position] = (val, n)
        if len(self.pending) != 3:
            return
        s.require(set(self.pending) == {0, 1, 2}, 'Verwacht precies een precisiemodus.')
        s.require(len({v[1] for v in self.pending.values()}) == 1, 'Batchgroottes verschillen.')
        self.batch += 1
        self.n += n
        loss, top1, top5 = [self.pending[i][0] for i in range(3)]
        c1, c5 = round(top1 * n / 100), round(top5 * n / 100)
        s.require(abs(top1 * n / 100 - c1) < .001 and abs(top5 * n / 100 - c5) < .001,
                  'Accuracy niet naar correcte aantallen te herleiden.')
        self.correct1[0] += c1
        self.correct5[0] += c5
        row = {'epoch': self.epoch, 'phase': self.phase, 'batch': self.batch,
               'mode': self.mode, 'weight_format': weight_format(self.mode), 'activation_bits': 32,
               'n': n, 'lr': self.lr, 'objective_loss': loss,
               'top1_percent': top1, 'top5_percent': top5, 'correct1': c1, 'correct5': c5}
        self.file.write(json.dumps(row, allow_nan=False) + '\n')
        if self.batch % 20 == 0:
            self.file.flush()
        self.pending.clear()

    def rows(self, values):
        s.require(not self.pending and len(self.meters) == 3 and self.batch > 0, 'Onvolledige metrics.')
        s.require(all(len(x) == 1 for x in values), 'Meer dan een precisiemodus uitgevoerd.')
        losses, top1, top5 = values
        return [{'epoch': self.epoch, 'phase': self.phase, 'mode': self.mode,
                 'weight_format': weight_format(self.mode), 'activation_bits': 32,
                 'upstream_selector': 32, 'n': self.n, 'objective_type': 'hard_cross_entropy',
                 'objective_loss': float(losses[0]), 'top1_percent': float(top1[0]),
                 'top5_percent': float(top5[0]), 'correct1': self.correct1[0],
                 'correct5': self.correct5[0], 'count_top1_percent': 100 * self.correct1[0] / self.n,
                 'lr': self.lr}]


def load_single_forward(source, utils, log, meter_class=None):
    forward, digest = s.load_forward(source, utils, log, meter_class)
    # Only the supplied argument changes; the function body is NOT rewritten.
    forward.__globals__['args'] = SimpleNamespace(bit_width_list='32', print_freq=20)
    return forward, digest


class ForbiddenDistillation:
    def __call__(self, *args, **kwargs):
        raise RuntimeError('Eenmodusproef mag geen distillatieloss gebruiken.')


def phase(rt, source, modules, model, optimizer, loader, epoch, phase_name, lr, out, log, mode):
    train = phase_name == 'train'
    assert_policy(model, mode)
    model.train(train)
    recorder = SingleMeterRecorder(modules.utils.AverageMeter, epoch, phase_name, lr,
                                   out / f'{phase_name}_batches_epoch_{epoch:05d}.jsonl', mode)
    forward, ast_hash = load_single_forward(source, modules.utils, log, recorder.factory)
    audit = {'optimizer_steps_in_phase': 0, 'finite_gradient_checks': 0,
             'gradient_l2_min': None, 'gradient_l2_max': None}
    def pre_step(opt, args, kwargs):
        r = validate_gradients(model)
        audit['finite_gradient_checks'] += 1
        audit['active_gradient_tensors'] = r['active_gradient_tensors']
        norm = r['l2_norm']
        audit['gradient_l2_min'] = norm if audit['gradient_l2_min'] is None else min(norm, audit['gradient_l2_min'])
        audit['gradient_l2_max'] = norm if audit['gradient_l2_max'] is None else max(norm, audit['gradient_l2_max'])
        # Returning None leaves optimizer arguments and gradient values unchanged.
    def post_step(opt, args, kwargs):
        audit['optimizer_steps_in_phase'] += 1
    handles = [optimizer.register_step_pre_hook(pre_step), optimizer.register_step_post_hook(post_step)] if train else []
    if rt.device.type == 'cuda':
        torch.cuda.synchronize(rt.device)
    t0 = time.perf_counter()
    try:
        values = forward(loader, model, torch.nn.CrossEntropyLoss().to(rt.device),
                         ForbiddenDistillation(), epoch - 1, training=train,
                         optimizer=optimizer if train else None, sum_writer=None)
        if rt.device.type == 'cuda':
            torch.cuda.synchronize(rt.device)
        elapsed = time.perf_counter() - t0
        rows = recorder.rows(values)
        expected = len(loader) if train else 0
        s.require(audit['optimizer_steps_in_phase'] == expected and audit['finite_gradient_checks'] == expected,
                  'Aantal gecontroleerde updates wijkt af.')
        assert_policy(model, mode)
    finally:
        for h in handles:
            h.remove()
        recorder.close()
    return rows, elapsed, ast_hash, audit


@torch.no_grad()
def layer_snapshot(model, mode):
    result = {}
    for name in a.TARGET_LAYERS:
        weight = model.get_submodule(name).weight
        row = {'master': weight.detach().cpu().clone()}
        if mode == 'mx':
            encoded = f.encode_tensor(a.effective_reference_weight(weight))
            row.update(codes=encoded.codes.cpu(), scales=encoded.scale_codes.cpu())
        result[name] = row
    result['_inactive_bn'] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if inactive_bn(n)}
    return result


@torch.no_grad()
def layer_updates(before, after, mode, epoch):
    s.require(not s.legacy.differences(before['_inactive_bn'], after['_inactive_bn']),
              'Ongebruikte BN-banken zijn gewijzigd.')
    rows = []
    for name in a.TARGET_LAYERS:
        b, e = before[name], after[name]
        difference = e['master'] - b['master']
        rows.append({'epoch': epoch, 'mode': mode, 'layer': name,
                     'master_changed_values': int((difference != 0).sum()),
                     'master_update_l2': float(difference.norm()),
                     'master_dtype': str(e['master'].dtype),
                     'changed_element_codes_including_padding': int((b['codes'] != e['codes']).sum()) if mode == 'mx' else None,
                     'changed_scale_codes': int((b['scales'] != e['scales']).sum()) if mode == 'mx' else None})
    s.require(any(r['master_changed_values'] > 0 for r in rows), 'Geen enkele MX-doellaag heeft masterupdates.')
    return rows


def best_record(records):
    s.require(bool(records), 'Geen epochs voor selectie.')
    best = records[0]
    for record in records:
        s.require(len(record['test']) == 1, 'Meer dan een modus in selectie.')
        if record['test'][0]['top1_percent'] > best['test'][0]['top1_percent']:
            best = record
    return best
