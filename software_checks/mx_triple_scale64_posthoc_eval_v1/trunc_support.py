"""Three-mode E4-source exponent/mantissa truncation extension; same upstream loss/backprop/optimizer sequencing.

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
import trunc_adapter as a


def weight_format(mode='joint'):
    s.require(mode == 'joint', 'Deze proef heeft een gezamenlijk driemodusmodel.')
    return 'MXFP8_E4M3+MXFP6_E2M3+MXFP4_E2M1'


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
    quantizers = [m for m in model.modules() if isinstance(m, a.JointMXQuantizer)]
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
            'note': 'E4M3-bank8, E2M3-bank32 en E2M1-bank4 elk eenmaal per minibatch; banken1/2 ongebruikt.'}


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


class JointMeterRecorder(s.MeterRecorder):
    """Same upstream AverageMeter arithmetic; nine meters, three modes."""
    def __init__(self, base_class, epoch, phase, lr, path, mode):
        super().__init__(base_class, epoch, phase, lr, path)
        self.mode = mode
        self.correct1, self.correct5 = [0] * len(a.MODES), [0] * len(a.MODES)

    def updated(self, position, val, n):
        s.require(position not in self.pending, 'Dubbele batchmetric.')
        self.pending[position] = (val, n)
        if len(self.pending) != 3 * len(a.MODES):
            return
        s.require(set(self.pending) == set(range(3 * len(a.MODES))), 'Verwacht negen meters.')
        s.require(len({v[1] for v in self.pending.values()}) == 1, 'Batchgroottes verschillen.')
        self.batch += 1; self.n += n
        records = []
        for i, bit in enumerate(a.MODES):
            loss, top1, top5 = [self.pending[i + offset][0] for offset in (0, len(a.MODES), 2 * len(a.MODES))]
            c1, c5 = round(top1 * n / 100), round(top5 * n / 100)
            s.require(abs(top1 * n / 100 - c1) < .001 and abs(top5 * n / 100 - c5) < .001,
                      'Accuracy niet naar correcte aantallen te herleiden.')
            self.correct1[i] += c1; self.correct5[i] += c5
            records.append({'weight_bits': bit, 'weight_format': a.FORMATS[bit], 'activation_bits': 32,
                            'objective_type': self.objective(bit), 'teacher_weight_format': self.teacher(bit),
                            'objective_loss': loss,
                            'top1_percent': top1, 'top5_percent': top5, 'correct1': c1, 'correct5': c5})
        self.file.write(json.dumps({'epoch': self.epoch, 'phase': self.phase, 'batch': self.batch,
                                    'n': n, 'lr': self.lr, 'precisions': records}, allow_nan=False) + '\n')
        if self.batch % 20 == 0:
            self.file.flush()
        self.pending.clear()

    def objective(self, bit):
        return 'recursive_soft_cross_entropy' if self.phase == 'train' and bit != a.MODES[-1] else 'hard_cross_entropy'

    def teacher(self, bit):
        if self.phase != 'train' or bit == a.MODES[-1]:
            return None
        return a.FORMATS[a.MODES[a.MODES.index(bit) + 1]]

    def rows(self, values):
        s.require(not self.pending and len(self.meters) == 3 * len(a.MODES) and self.batch > 0, 'Onvolledige metrics.')
        s.require(all(len(x) == len(a.MODES) for x in values), 'Verwacht drie modi.')
        losses, top1, top5 = values
        return [{'epoch': self.epoch, 'phase': self.phase, 'mode': self.mode, 'weight_bits': bit,
                 'weight_format': a.FORMATS[bit], 'activation_bits': 32, 'bn_bank_key': a.BANK[bit],
                 'n': self.n, 'objective_type': self.objective(bit), 'teacher_weight_format': self.teacher(bit),
                 'objective_loss': float(losses[i]),
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


def load_joint_forward(source, utils, log, meter_class=None):
    _, changed, digest = adapt_forward_text(source)
    namespace = {'torch': torch, 'args': SimpleNamespace(bit_width_list='4,6,8', print_freq=20),
                 'AverageMeter': meter_class or utils.AverageMeter, 'accuracy': utils.accuracy,
                 'logging': s.UpstreamLog(log), 'select_mode': a.select_mode}
    exec(compile(changed, str(Path(source) / 'train.py') + ' [joint truncation selectors]', 'exec'), namespace)
    author_forward = namespace['forward']
    def shared_forward(loader, model, criterion, criterion_soft, epoch, training=True,
                       optimizer=None, sum_writer=None):
        model.train(training)
        with a.shared_source_triplets(model) as source_audit:
            result = author_forward(loader, model, criterion, criterion_soft, epoch,
                                    training=training, optimizer=optimizer, sum_writer=sum_writer)
        model._last_source_audit = source_audit
        return result
    return shared_forward, digest


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
    recorder = JointMeterRecorder(modules.utils.AverageMeter, epoch, phase_name, lr,
                                out / f'{phase_name}_batches_epoch_{epoch:05d}.jsonl', mode)
    forward, ast_hash = load_joint_forward(source, modules.utils, log, recorder.factory)
    criterion_soft = CheckedSoftLoss(modules.losses.CrossEntropyLossSoft())
    forward_names = {bit: 'forward_calls_' + a.FORMATS[bit].rsplit('_', 1)[1] for bit in a.MODES}
    audit = {'optimizer_steps_in_phase': 0, 'finite_gradient_checks': 0,
             'gradient_l2_min': None, 'gradient_l2_max': None,
             **{name: 0 for name in forward_names.values()}}
    pending_modes = []
    # Diagnostic counts only; the author topk(5) convention stays unchanged.
    predictions = {bit: torch.zeros(10, dtype=torch.int64, device=rt.device) for bit in a.MODES}
    labels = torch.zeros(10, dtype=torch.int64, device=rt.device)
    class CountLabelsOnce:
        def __init__(self, original): self.original = original
        def __len__(self): return len(self.original)
        def __iter__(self):
            for inputs, targets in self.original:
                if not train: labels.add_(torch.bincount(targets.detach().to(rt.device), minlength=10))
                yield inputs, targets
    counted_loader = CountLabelsOnce(loader)
    def model_post(module, inputs, output):
        if not train:
            predicted = output.detach().topk(5, 1, True, True)[1][:, 0]
            predictions[module.weight_mode].add_(torch.bincount(predicted, minlength=10))
    def model_pre(module, inputs):
        pending_modes.append(module.weight_mode)
        audit[forward_names[module.weight_mode]] += 1
        if not train and len(pending_modes) == len(a.MODES):
            s.require(pending_modes == list(a.MODES), 'Verkeerde evaluatievolgorde.')
            pending_modes.clear()
    def pre_step(opt, args, kwargs):
        s.require(pending_modes == [8, 6, 4], 'Elke update vereist E4M3, E2M3, E2M1 op dezelfde minibatch.')
        pending_modes.clear()
        s.require(a.cache_is_empty(model), 'Broncache niet leeg voor Adam-update.')
        r = validate_gradients(model)
        audit['finite_gradient_checks'] += 1; audit['active_gradient_tensors'] = r['active_gradient_tensors']
        norm = r['l2_norm']
        audit['gradient_l2_min'] = norm if audit['gradient_l2_min'] is None else min(norm, audit['gradient_l2_min'])
        audit['gradient_l2_max'] = norm if audit['gradient_l2_max'] is None else max(norm, audit['gradient_l2_max'])
    def post_step(opt, args, kwargs):
        audit['optimizer_steps_in_phase'] += 1
    handles = [model.register_forward_pre_hook(model_pre), model.register_forward_hook(model_post)]
    if train:
        handles += [optimizer.register_step_pre_hook(pre_step), optimizer.register_step_post_hook(post_step)]
    if rt.device.type == 'cuda':
        torch.cuda.synchronize(rt.device)
    t0 = time.perf_counter()
    try:
        values = forward(counted_loader, model, torch.nn.CrossEntropyLoss().to(rt.device), criterion_soft,
                         epoch - 1, training=train, optimizer=optimizer if train else None, sum_writer=None)
        if rt.device.type == 'cuda':
            torch.cuda.synchronize(rt.device)
        elapsed = time.perf_counter() - t0
        rows = recorder.rows(values)
        expected = len(loader) if train else 0
        s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == expected,
                  'Verkeerd aantal gecontroleerde updates.')
        s.require(not pending_modes and criterion_soft.calls == 2 * expected, 'Onvolledige recursieve distillatie/forward.')
        s.require(all(audit[name] == len(loader) for name in forward_names.values()),
                  'Verkeerd aantal forwards per modus.')
        source_audit = model._last_source_audit
        s.require(source_audit['source_triplets'] == len(loader), 'Verkeerd aantal broncycli.')
        s.require(source_audit['high_encodings'] == source_audit['exponent_derivations']
                  == source_audit['mantissa_derivations'] == 6 * len(loader),
                  'Verkeerd aantal bronencoderingen of bitafleidingen.')
        s.require(source_audit['scale_copies_unchanged'] == 12 * len(loader)
                  and source_audit['source_cache_cleared'] and a.cache_is_empty(model),
                  'Schaalbehoud of cache-opruiming mislukt.')
        source_audit['cache_cleared_before_adam'] = True
        audit.update(detached_teacher_checks=criterion_soft.calls, no_extra_FP32_teacher=True,
                     shared_source_checks=source_audit, cache_empty_before_all_optimizer_steps=True)
        if not train:
            audit['prediction_counts'] = {str(bit): values.cpu().tolist() for bit, values in predictions.items()}
            audit['label_counts'] = labels.cpu().tolist()
            s.require(all(sum(c) == recorder.n for c in audit['prediction_counts'].values())
                      and sum(audit['label_counts']) == recorder.n, 'Onvolledige label-/voorspellingstellingen.')
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
        chain = a.encode_chain(effective)
        for bit in a.MODES:
            encoded = chain[bit]
            row[str(bit)] = {'codes': encoded.codes.cpu(), 'scales': encoded.scale_codes.cpu()}
        s.require(torch.equal(row['8']['scales'], row['6']['scales'])
                  and torch.equal(row['8']['scales'], row['4']['scales']), 'Alle modi moeten de E4-bronschaal behouden.')
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
            rows.append({'epoch': epoch, 'mode': mode, 'weight_bits': bit, 'weight_format': a.FORMATS[bit], 'layer': name,
                         'master_changed_values': int((difference != 0).sum()),
                         'master_update_l2': float(difference.norm()), 'master_dtype': str(e['master'].dtype),
                         'changed_element_codes_including_padding': int((b[str(bit)]['codes'] != e[str(bit)]['codes']).sum()),
                         'changed_scale_codes': int((b[str(bit)]['scales'] != e[str(bit)]['scales']).sum())})
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
