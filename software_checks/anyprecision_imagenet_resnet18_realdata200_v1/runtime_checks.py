"""Checks shared by the bounded real-data pilot and its CPU fixtures."""
from __future__ import annotations

import math

import support as s


def model_and_optimizer(sources, device):
    import torch
    from resnet18_model import build_model, build_soft_loss
    model = build_model(sources['anyprecision'], sources['truncquant'], num_classes=1000).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=s.CONFIG['lr'],
                                momentum=s.CONFIG['momentum'], weight_decay=s.CONFIG['weight_decay'])
    return model, optimizer, build_soft_loss(sources['anyprecision']).to(device)


def check_metrics(metrics, *, training, expected_samples):
    s.require(metrics.get('samples') == expected_samples,
              'Onjuist aantal beelden in een trainings- of validatiebatch.')
    s.require(metrics.get('optimizer_updates') == (1 if training else 0),
              'Onjuist aantal optimizerupdates in een batch.')
    s.require(set(metrics.get('by_bit', {})) == {str(bit) for bit in s.BITS},
              'Niet alle vijf precisies zijn uitgevoerd.')
    if training:
        s.require(expected_samples == 256 and metrics.get('mode') == 'serial_dp4' and
                  metrics.get('chunk_sizes') == [64] * 4 and metrics.get('forwards') == 20 and
                  metrics.get('replicas_used') == 4,
                  'De globale batch256 met vier BN64-groepen is gewijzigd.')
    else:
        s.require(metrics.get('forwards') == 5 and 1 <= expected_samples <= 64,
                  'Validatie moet vijf precisies op maximaal 64 beelden per batch verwerken.')
    for value in metrics['by_bit'].values():
        s.require(value.get('samples') == expected_samples and math.isfinite(float(value['loss'])),
                  'Niet-eindige loss of inconsistente sampleteller.')
        s.require(0 <= value['top1_correct'] <= value['top5_correct'] <= expected_samples,
                  'Ongeldige top-k-tellingen.')


def fresh_totals():
    return {str(bit): {'loss_sum': 0., 'top1_correct': 0, 'top5_correct': 0, 'samples': 0}
            for bit in s.BITS}


def add_totals(totals, metrics):
    for bit, values in metrics['by_bit'].items():
        n = values['samples']
        totals[bit]['loss_sum'] += values['loss'] * n
        for field in ('top1_correct', 'top5_correct', 'samples'):
            totals[bit][field] += values[field]


def summarize_totals(totals):
    result = {}
    for bit, value in totals.items():
        n = value['samples']
        s.require(n > 0, 'Geen beelden in de metricsamenvatting.')
        result[bit] = {'loss': value['loss_sum'] / n, 'top1': 100 * value['top1_correct'] / n,
                       'top5': 100 * value['top5_correct'] / n, 'samples': n}
    return result


def bn_snapshot(model):
    """All BN buffers, so frozen evaluation is verified beyond the counters."""
    import torch
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            result[name] = {key: getattr(module, key).detach().cpu().clone()
                           for key in ('running_mean', 'running_var', 'num_batches_tracked')}
    return result


def check_bn_training(before, after, updates):
    s.require(before.keys() == after.keys() and before, 'BatchNorm-structuur is gewijzigd.')
    shared, switched = 0, 0
    for name, values in before.items():
        is_switched = '.bn_dict.' in name
        expected = updates if is_switched else 5 * updates
        observed = int(after[name]['num_batches_tracked']) - int(values['num_batches_tracked'])
        s.require(observed == expected, f'Onjuist aantal behouden BN-updates voor {name}: {observed}.')
        switched += int(is_switched)
        shared += int(not is_switched)
    s.require(shared == 4 and switched == 80, 'Onverwachte gedeelde/precisiespecifieke BN-indeling.')
    return {'status': 'PASS', 'optimizer_updates': updates, 'shared_bn_modules': shared,
            'precision_specific_bn_banks': switched, 'updates_per_shared_bn': 5 * updates,
            'updates_per_precision_bn_bank': updates}


def check_bn_unchanged(before, after):
    import torch
    s.require(before.keys() == after.keys() and before, 'BatchNorm-structuur veranderde bij validatie.')
    for name, values in before.items():
        for key, value in values.items():
            s.require(torch.equal(value, after[name][key]),
                      f'Validatie heeft BatchNorm bijgewerkt: {name}/{key}.')
    return {'status': 'PASS', 'all_BN_buffers_bitwise_unchanged': True,
            'modules': len(before), 'optimizer_updates_during_validation': 0}


def assert_finite_state(model, optimizer):
    import torch
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point():
            s.require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                      'Niet-eindige of niet-FP32 modelwaarde: ' + name)
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                s.require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                          'Niet-eindige of niet-FP32 SGD-state: ' + name)
