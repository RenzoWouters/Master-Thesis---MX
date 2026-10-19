"""Two-mode extension; same upstream loss/backprop/optimizer sequencing.

Only six legacy selector lines are replaced by three select_mode calls.
Model, data, loss definitions, Adam and scheduler come from the supplied source.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import torch
import svhn_support as s
import dual_adapter as a


def weight_format(mode='joint'):
    s.require(mode == 'joint', 'Deze proef heeft een gezamenlijk tweemodusmodel.')
    return 'MXFP6_E2M3+MXFP4_E2M1'


inactive_bn = a.inactive_bn


def build_model(modules, mode, device):
    weight_format(mode)
    original = modules.models.svhnq(s.BITS, num_classes=10, expand=2).to(device)
    model = a.adapt_copy(original)
    s.require(s.legacy.state_digest(original.state_dict()) == s.legacy.state_digest(model.state_dict()),
              'Integratie heeft de initialisatie veranderd.')
    assert_policy(model, mode)
    return model


def assert_policy(model, mode='joint'):
    weight_format(mode)
    quantizers = [m for m in model.modules() if isinstance(m, a.DualMXQuantizer)]
    norms = [m for m in model.modules() if isinstance(m, a.WeightModeBatchNorm)]
    s.require(len(quantizers) == 6 and len(norms) == 5, 'Verkeerde MX-/BN-laagselectie.')
    s.require(all(m.weight_mode == model.weight_mode for m in quantizers + norms), 'Modusselectie inconsistent.')
    s.require(model.weight_mode in a.MODES, 'Onbekende gewichtsmodus.')
    s.require(all(getattr(m, 'abit', 32) == 32 and getattr(m, 'wbit', 32) == 32 for m in model.modules()),
              'Legacy W/A-selector veranderd; activaties moeten A32 blijven.')
    s.require(all(p.dtype == torch.float32 for p in model.parameters()), 'Masterdtype gewijzigd.')


def validate_gradients(model):
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
    s.require(math.isfinite(norm), 'Niet-eindige gradientnorm.')
    return {'active_gradient_tensors': len(active), 'l2_norm': norm}


def validate_bn(model, steps):
    counts = {n: int(v) for n, v in model.state_dict().items() if n.endswith('num_batches_tracked')}
    s.require(len(counts) == 25, 'Verwacht vijf lagen met vijf bewaarde BN-banken.')
    for name, value in counts.items():
        s.require(value == (0 if inactive_bn(name) else steps), 'Verkeerde BN-teller: ' + name)
    return {'all_counters_correct': True, 'counts': counts,
            'note': 'E2M3-bank32 en E2M1-bank4 elk eenmaal per minibatch; overige banken ongebruikt.'}


def validate_adam_state(model, optimizer, steps):
    count = 0
    for name, p in model.named_parameters():
        if inactive_bn(name):
            s.require(p not in optimizer.state, 'Ongebruikte BN-bank heeft Adamstaat: ' + name)
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


class DualMeterRecorder(s.MeterRecorder):
    """Same upstream AverageMeter arithmetic; six meters, two modes."""
    def __init__(self, base_class, epoch, phase, lr, path, mode):
        super().__init__(base_class, epoch, phase, lr, path)
        self.mode = mode
        self.correct1, self.correct5 = [0, 0], [0, 0]

    def updated(self, position, val, n):
        s.require(position not in self.pending, 'Dubbele batchmetric.')
        self.pending[position] = (val, n)
        if len(self.pending) != 6:
            return
        s.require(set(self.pending) == set(range(6)), 'Verwacht zes meters.')
        s.require(len({v[1] for v in self.pending.values()}) == 1, 'Batchgroottes verschillen.')
        self.batch += 1; self.n += n
        records = []
        for i, bit in enumerate(a.MODES):
            loss, top1, top5 = [self.pending[i + offset][0] for offset in (0, 2, 4)]
            c1, c5 = round(top1 * n / 100), round(top5 * n / 100)
            s.require(abs(top1 * n / 100 - c1) < .001 and abs(top5 * n / 100 - c5) < .001,
                      'Accuracy niet naar correcte aantallen te herleiden.')
            self.correct1[i] += c1; self.correct5[i] += c5
            records.append({'weight_bits': bit, 'weight_format': a.FORMATS[bit], 'activation_bits': 32,
                            'objective_type': self.objective(bit), 'objective_loss': loss,
                            'top1_percent': top1, 'top5_percent': top5, 'correct1': c1, 'correct5': c5})
        self.file.write(json.dumps({'epoch': self.epoch, 'phase': self.phase, 'batch': self.batch,
                                    'n': n, 'lr': self.lr, 'precisions': records}, allow_nan=False) + '\n')
        if self.batch % 20 == 0:
            self.file.flush()
        self.pending.clear()

    def objective(self, bit):
        return 'recursive_soft_cross_entropy' if self.phase == 'train' and bit == 4 else 'hard_cross_entropy'

    def rows(self, values):
        s.require(not self.pending and len(self.meters) == 6 and self.batch > 0, 'Onvolledige metrics.')
        s.require(all(len(x) == 2 for x in values), 'Verwacht twee modi.')
        losses, top1, top5 = values
        return [{'epoch': self.epoch, 'phase': self.phase, 'mode': self.mode, 'weight_bits': bit,
                 'weight_format': a.FORMATS[bit], 'activation_bits': 32, 'bn_bank_key': a.BANK[bit],
                 'n': self.n, 'objective_type': self.objective(bit), 'objective_loss': float(losses[i]),
                 'top1_percent': float(top1[i]), 'top5_percent': float(top5[i]),
                 'correct1': self.correct1[i], 'correct5': self.correct5[i],
                 'count_top1_percent': 100 * self.correct1[i] / self.n, 'lr': self.lr}
                for i, bit in enumerate(a.MODES)]


def adapt_forward_text(source):
    """Exactly three selector-pair edits. No changes to losses, dropout or updates."""
    text = (Path(source) / 'train.py').read_text(encoding='utf-8')
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'forward']
    s.require(len(nodes) == 1, 'Precies een forwardfunctie verwacht.')
    original = ast.get_source_segment(text, nodes[0])
    changed = original
    for indent, value in [('                    ', 'bw'), ('            ', 'bit_width_list[-1]'),
                          ('                ', 'bw')]:
        old = indent + f"model.apply(lambda m: setattr(m, 'wbit', {value}))\n" + indent + f"model.apply(lambda m: setattr(m, 'abit', {value}))"
        new = indent + f'select_mode(model, {value})'
        s.require(changed.count(old) == 1, 'Bronselectie past niet exact; stop zonder gok.')
        changed = changed.replace(old, new)
    digest = hashlib.sha256(ast.dump(ast.parse(changed), include_attributes=False).encode()).hexdigest()
    return original, changed, digest


def load_dual_forward(source, utils, log, meter_class=None):
    _, changed, digest = adapt_forward_text(source)
    namespace = {'torch': torch, 'args': SimpleNamespace(bit_width_list='4,6', print_freq=20),
                 'AverageMeter': meter_class or utils.AverageMeter, 'accuracy': utils.accuracy,
                 'logging': s.UpstreamLog(log), 'select_mode': a.select_mode}
    exec(compile(changed, str(Path(source) / 'train.py') + ' [dual selectors]', 'exec'), namespace)
    return namespace['forward'], digest


class CheckedSoftLoss(torch.nn.Module):
    def __init__(self, original):
        super().__init__(); self.original = original; self.calls = 0

    def forward(self, output, target):
        s.require(not target.requires_grad and target.grad_fn is None, 'Teacher moet losgekoppeld zijn.')
        self.calls += 1
        return self.original(output, target)


def phase(rt, source, modules, model, optimizer, loader, epoch, phase_name, lr, out, log, mode='joint'):
    train = phase_name == 'train'
    assert_policy(model, mode); model.train(train)
    recorder = DualMeterRecorder(modules.utils.AverageMeter, epoch, phase_name, lr,
                                out / f'{phase_name}_batches_epoch_{epoch:05d}.jsonl', mode)
    forward, ast_hash = load_dual_forward(source, modules.utils, log, recorder.factory)
    criterion_soft = CheckedSoftLoss(modules.losses.CrossEntropyLossSoft())
    audit = {'optimizer_steps_in_phase': 0, 'finite_gradient_checks': 0,
             'gradient_l2_min': None, 'gradient_l2_max': None,
             'forward_calls_E2M3': 0, 'forward_calls_E2M1': 0}
    pending_modes = []
    def model_pre(module, inputs):
        pending_modes.append(module.weight_mode)
        audit['forward_calls_E2M3' if module.weight_mode == 6 else 'forward_calls_E2M1'] += 1
        if not train and len(pending_modes) == 2:
            s.require(pending_modes == [4, 6], 'Verkeerde evaluatievolgorde.')
            pending_modes.clear()
    def pre_step(opt, args, kwargs):
        s.require(pending_modes == [6, 4], 'Elke update vereist eerst E2M3, dan E2M1 op dezelfde minibatch.')
        pending_modes.clear()
        r = validate_gradients(model)
        audit['finite_gradient_checks'] += 1; audit['active_gradient_tensors'] = r['active_gradient_tensors']
        norm = r['l2_norm']
        audit['gradient_l2_min'] = norm if audit['gradient_l2_min'] is None else min(norm, audit['gradient_l2_min'])
        audit['gradient_l2_max'] = norm if audit['gradient_l2_max'] is None else max(norm, audit['gradient_l2_max'])
    def post_step(opt, args, kwargs):
        audit['optimizer_steps_in_phase'] += 1
    handles = [model.register_forward_pre_hook(model_pre)]
    if train:
        handles += [optimizer.register_step_pre_hook(pre_step), optimizer.register_step_post_hook(post_step)]
    if rt.device.type == 'cuda':
        torch.cuda.synchronize(rt.device)
    t0 = time.perf_counter()
    try:
        values = forward(loader, model, torch.nn.CrossEntropyLoss().to(rt.device), criterion_soft,
                         epoch - 1, training=train, optimizer=optimizer if train else None, sum_writer=None)
        if rt.device.type == 'cuda':
            torch.cuda.synchronize(rt.device)
        elapsed = time.perf_counter() - t0
        rows = recorder.rows(values)
        expected = len(loader) if train else 0
        s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == expected,
                  'Verkeerd aantal gecontroleerde updates.')
        s.require(not pending_modes and criterion_soft.calls == expected, 'Onvolledige distillatie/forward.')
        s.require(audit['forward_calls_E2M3'] == audit['forward_calls_E2M1'] == len(loader),
                  'Verkeerd aantal forwards per modus.')
        audit.update(detached_teacher_checks=criterion_soft.calls, no_extra_FP32_teacher=True)
        assert_policy(model, mode)
    finally:
        for h in handles: h.remove()
        recorder.close()
    return rows, elapsed, ast_hash, audit


@torch.no_grad()
def layer_snapshot(model, mode='joint'):
    result = {}
    for name in a.TARGET_LAYERS:
        weight = model.get_submodule(name).weight
        effective = a.effective_reference_weight(weight)
        row = {'master': weight.detach().cpu().clone()}
        for bit in a.MODES:
            encoded = a.codec(bit).encode_tensor(effective)
            row[str(bit)] = {'codes': encoded.codes.cpu(), 'scales': encoded.scale_codes.cpu()}
        s.require(torch.equal(row['4']['scales'], row['6']['scales']), 'E2M3/E2M1-schalen moeten gelijk zijn.')
        result[name] = row
    result['_inactive_bn'] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if inactive_bn(n)}
    return result


@torch.no_grad()
def layer_updates(before, after, mode, epoch):
    s.require(not s.legacy.differences(before['_inactive_bn'], after['_inactive_bn']), 'Ongebruikte BN gewijzigd.')
    rows = []
    for name in a.TARGET_LAYERS:
        b, e = before[name], after[name]
        difference = e['master'] - b['master']
        for bit in a.MODES:
            rows.append({'epoch': epoch, 'mode': mode, 'weight_bits': bit, 'layer': name,
                         'master_changed_values': int((difference != 0).sum()),
                         'master_update_l2': float(difference.norm()), 'master_dtype': str(e['master'].dtype),
                         'changed_element_codes_including_padding': int((b[str(bit)]['codes'] != e[str(bit)]['codes']).sum()),
                         'changed_scale_codes': int((b[str(bit)]['scales'] != e[str(bit)]['scales']).sum())})
    s.require(any(r['master_changed_values'] > 0 for r in rows), 'Geen gewijzigde mastergewichten.')
    return rows


def best_record(records):
    s.require(bool(records), 'Geen selectiegeschiedenis.')
    def score(record):
        s.require([row['weight_bits'] for row in record['test']] == list(a.MODES), 'Ontbrekende MX-testmodi.')
        return record['test'][-1]['top1_percent']
    best = records[0]
    for record in records:
        if score(record) > score(best): best = record
    return best
