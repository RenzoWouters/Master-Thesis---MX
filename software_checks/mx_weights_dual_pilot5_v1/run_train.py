"""Five complete train-only epochs: joint E2M3 + E2M1, fixed A32, FP32 masters.

Upstream recursive distillation: hard CE at E2M3, detached teacher soft CE at
E2M1, summed gradients then ONE Adam update. No truncation or old checkpoint.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
import svhn_support as s
import dual_support as t
import dual_adapter as a

ROOT = Path(__file__).resolve().parent


def config(mode: str = 'joint') -> dict:
    s.require(mode == 'joint', 'Deze proef heeft exact twee gezamenlijke MX-modi.')
    return {
        'schema': 'mx-weights-dual-pilot5-protocol-v1',
        'purpose': 'five complete train-only epochs: joint weight-only MX quantization without truncation',
        'mode': mode, 'weight_format': t.weight_format(mode), 'weight_bits': [4, 6],
        'weight_formats': {'6': 'MXFP6_E2M3', '4': 'MXFP4_E2M1'},
        'weight_preprocess': 'original W32 G(W); detached mean(abs(W)), tanh/max normalization',
        'activation_bits': 32, 'activation_policy': 'original A32/ReLU in BOTH modes',
        'master_dtype': 'float32', 'gradient_dtype': 'float32', 'adam_moments_dtype': 'float32',
        'mx_block_size': 32, 'mx_scale_format': 'E8M0',
        'mx_group_axis': 'flatten Cin*Kh*Kw within each Cout, zero-pad tail',
        'mx_scale_policy': 'same power-3 frexp rule on current G(W) for both modes; independent direct encoding; scales equal for same W',
        'mx_rounding': 'nearest ties-to-even; finite FP32 source, scalar subnormal flush, local subnormals supported',
        'ste_policy': 'identity through each MX quantizer; retain original G(W) derivatives',
        'numeric_kernel': 'FP32 F.conv2d; codec FP64 intermediates; not native low-bit kernels',
        'truncation': False, 'distillation': True,
        'loss': 'E2M3 hard CE + E2M1 upstream soft CE with detached E2M3 teacher; T=1; no division by 2',
        'passes_per_training_batch': 2, 'optimizer_steps_per_training_batch': 1,
        'training_mode_order': [6, 4], 'test_mode_order': [4, 6],
        'internal_upstream_selectors': [4, 6], 'legacy_abits_and_wbits': 32,
        'model_bank_structure': s.BITS, 'bn_bank_mapping': {'6': '32', '4': '4'},
        'batchnorm_policy': 'two active weight-mode banks; legacy keys32/4 do NOT imply A32/A4; banks1/2/8 unused',
        'extra_FP32_teacher_forward': False,
        'dataset': 'svhn', 'train_split': 'train', 'test_split': 'test', 'extra_used': False,
        'train_examples': 73257, 'test_examples': 26032, 'validation_examples': 0,
        'initialization_policy': 'fresh seed20261009; same untrained state as prior model construction; no warmstart',
        'model': 'svhnq', 'expand': 2, 'parameters': 132954,
        'planned_epochs': 100, 'pilot_stop_epoch': 5,
        'batch_size': 128, 'optimizer': 'adam', 'lr': .001, 'weight_decay': 0.0,
        'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': .1,
        'workers': 0, 'pin_memory': True, 'drop_last': False,
        'train_shuffle': True, 'test_shuffle': False, 'seed': 20261009,
        'evaluation_policy': 'both modes on complete official test after every epoch; hard CE for both',
        'selection_policy': 'first strict maximum of E2M3/A32 official-test top1',
        'reporting_policy': 'both modes from same selected best checkpoint; also both at final epoch',
        'backend_policy': 'github: cudnn.benchmark=True; no deterministic override',
        'rng_policy': 'global streams as upstream; sequential dropout draws per mode; saved at epoch boundaries',
        'checkpoint_policy': 'immutable FP32 training state per epoch and selected best/last copies',
        'export_policy': 'independent direct E2M3 and E2M1 files per selected checkpoint; NOT one truncatable source',
        'exact_historical_paper_reproduction_proven': False,
    }


def dependencies(out: Path, log) -> None:
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    (out / 'pip_freeze.txt').write_text(freeze.stdout + freeze.stderr, encoding='utf-8')
    s.require(freeze.returncode == 0, 'pip freeze mislukt; zie pip_freeze.txt.')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
    (out / 'pip_check.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    s.require(check.returncode == 0, 'Afhankelijkheidsconflict; zie pip_check.txt. Geen automatische upgrades.')
    log('pip check: ' + check.stdout.strip())


def create_context(args, out: Path, report: dict, log, check_only: bool = False):
    package_hashes = s.verify_package()
    c = config(args.mode)
    source, source_info = s.legacy.prepare_source(out)
    report.update(config=c, source=source_info, package_sha256=package_hashes,
                  interpreter=sys.executable)
    s.write_json(out / 'training_change_manifest.json',
                 json.loads((ROOT / 'TRAINING_CHANGE_MANIFEST.json').read_text(encoding='utf-8')))
    dependencies(out, log)
    rt = s.runtime(c['seed'], args.device)
    modules = s.import_source(source)
    report.update(environment=rt.environment, execution_settings=rt.settings)
    log('GPU: ' + rt.environment['device_name'])
    log('Originele backend: cudnn.benchmark=True; geen bitexacte CUDA-hervatgarantie.')
    model = None
    if check_only:
        import test_mx
        import test_dual
        report['codec_tests'] = test_mx.run_tests(modules, str(rt.device))
        report['synthetic_training_checks'] = test_dual.run_tests(modules, rt, source, log)
        _, forward_hash = t.load_dual_forward(source, modules.utils, log)
        report.update(forward_ast_sha256=forward_hash, dataset_downloaded=False,
                      synthetic_checks_only=True)
        report['notebook_dependencies'] = {}
        for name in ['pandas', 'matplotlib', 'ipykernel']:
            try:
                report['notebook_dependencies'][name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                report['notebook_dependencies'][name] = 'MISSING: notebook only'
        log('CHECK PASS: E2M3/E2M1, A32, twee BN-banken, distillatie en gezamenlijke updates. Geen SVHN-training.')
        return None

    start_data = time.perf_counter()
    info = s.ensure_dataset(args.data_root.resolve(), modules.data.SVHNMIX, args.download, log)
    report['data_preparation_seconds_including_download_and_checksums'] = time.perf_counter() - start_data
    report['dataset_files'] = info
    modules.data.data_paths['svhn'] = str(args.data_root.resolve())
    start_load = time.perf_counter()
    train_tf = modules.data.get_transform('svhn', 'train')
    test_tf = modules.data.get_transform('svhn', 'val')
    s.require(c['train_split'] == 'train' and c['extra_used'] is False,
              'Deze runner is uitsluitend voor de volledige train-split, zonder extra.')
    train = modules.data.get_dataset('svhn', c['train_split'], train_tf, download=False)
    test = modules.data.get_dataset('svhn', 'val', test_tf, download=False)
    s.require(train.split == 'train' and test.split == 'test', 'Onverwachte datasetsplit.')
    s.require(set(info['files']) == {'train_32x32.mat', 'test_32x32.mat'},
              'Train-only mag alleen train en test laden of controleren.')
    selection = {'train': s.validate_dataset(train, c['train_examples'], 'SVHN train-only'),
                 'test': s.validate_dataset(test, c['test_examples'], 'SVHN test'),
                 'train_split_order': ['train'], 'extra_used': False,
                 'held_out_validation_examples': 0}
    report['dataset_load_seconds'] = time.perf_counter() - start_load
    report['dataset_selection'] = selection
    report['transforms'] = {'train': repr(train_tf), 'test': repr(test_tf)}
    s.write_json(out / 'dataset_manifest.json', {'files': info, 'selection': selection,
                                              'transforms': report['transforms']})
    log(f'DATA PASS: train={len(train)}, test={len(test)}; extra NIET gebruikt; '
        'invoer 40x40; volledige train-split; geen validatiesplit.')
    # No explicit generators: match the global RNG policy of the source loaders.
    torch = rt.torch
    loaders = {
        'train': s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(train),
                  batch_size=c['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=False)),
        'test': s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(test),
                  batch_size=c['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=False)),
    }
    log(f'MINIBATCHES per epoch: train={len(loaders["train"])}; test={len(loaders["test"])}; '
        f'batch={c["batch_size"]}, drop_last=False.')
    model = t.build_model(modules, c['mode'], rt.device)
    s.require(model.expand == c['expand'] and sum(p.numel() for p in model.parameters()) == c['parameters'],
              'Onverwacht model; verwacht svhnq expand=2.')
    optimizer = modules.optimizer.get_optimizer_config(model, 'adam', c['lr'], c['weight_decay'])
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    contract = {
        'config': c, 'environment': rt.environment, 'settings': rt.settings,
        'source_files': source_info['workcopy_files_sha256'], 'package_sha256': package_hashes,
        'data_files': info['files'], 'dataset_selection': selection, 'transforms': report['transforms'],
        'synthetic_fixture_only': False,
    }
    report['contract_sha256'] = rt.core.json_digest(contract)
    return SimpleNamespace(rt=rt, source=source, modules=modules, model=model, optimizer=optimizer,
                           scheduler=scheduler, loaders=loaders, contract=contract, c=c,
                           report=report, source_info=source_info)


def restore_timing_sidecar(rt, trace: dict) -> None:
    # The last checkpoint is written before its own write duration is known.
    # The immutable sidecar completes that timing without altering the checkpoint.
    if not trace['epochs']:
        return
    last = trace['epochs'][-1]
    path = Path(last['checkpoint_path']).parent.parent / 'epoch_summaries' / f"epoch_{last['epoch']:05d}.json"
    marker = path.with_suffix('.json.sha256')
    if path.is_file() and marker.is_file():
        s.require(rt.core.file_sha256(path) == marker.read_text().strip(), 'Epoch-sidecar checksum wijkt af.')
        full = json.loads(path.read_text(encoding='utf-8'))
        for key, value in last.items():
            s.require(value == full.get(key), 'Epoch-sidecar strookt niet met checkpoint: ' + key)
        trace['epochs'][-1] = full


def run_epochs(ctx, out: Path, log, stop_after: int, resume: Path | None = None,
               install_signal_handler: bool = True) -> dict:
    rt, c = ctx.rt, ctx.c
    torch, core, device = rt.torch, rt.core, rt.device
    model, optimizer, scheduler, loaders = ctx.model, ctx.optimizer, ctx.scheduler, ctx.loaders
    for directory in ['checkpoints', 'epoch_summaries', 'sample_order', 'batch_metrics']:
        (out / directory).mkdir(exist_ok=True)
    trace = {'initial_model_sha256': s.legacy.state_digest(model.state_dict()), 'epochs': []}
    done, steps = 0, 0
    expected_steps = len(loaders['train'])
    s.require(expected_steps == math.ceil(c['train_examples'] / c['batch_size']), 'Verkeerd aantal minibatches.')
    resume_hash = None
    if resume is not None:
        resume = resume.resolve()
        state = core.load_verified(resume)
        s.require(0 < state['completed_epoch'] < stop_after, 'Dit checkpoint laat geen nieuwe pilot-epochs toe.')
        ctx.report['restore_checks'] = core.restore_checkpoint(state, model, optimizer, scheduler, {},
                                                              device, ctx.contract)
        trace = copy.deepcopy(state['trace'])
        restore_timing_sidecar(rt, trace)
        done, steps = state['completed_epoch'], state['optimizer_steps']
        s.require(steps == done * expected_steps, 'Opgeslagen aantal stappen is onjuist.')
        s.require(len(trace['epochs']) == done, 'Geschiedenis van epochs is onvolledig.')
        resume_hash = core.file_sha256(resume)
        ctx.report.update(resumed_from=str(resume), resume_input_sha256=resume_hash)
        log(f'HERVAT na epoch {done}; geen initialisatie van het model of LR-schema opnieuw.')
    else:
        log('NIEUWE initialisatie. Geen ResNet- of proefcheckpoint ingeladen.')
    first_epoch = done + 1
    soft_stop = {'requested': False}
    old_handler = None
    def handler(signum, frame):
        if soft_stop['requested']:
            raise KeyboardInterrupt('Tweede Ctrl+C: stop direct; alleen volledige checkpoints zijn hervatbaar.')
        soft_stop['requested'] = True
        log('STOP gevraagd: huidige epoch wordt volledig afgewerkt en opgeslagen. Tweede Ctrl+C stopt direct.')
    if install_signal_handler:
        old_handler = signal.signal(signal.SIGINT, handler)
    try:
        for epoch in range(done + 1, stop_after + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            lr = float(optimizer.param_groups[0]['lr'])
            expected_lr = c['lr']
            for milestone in c['scheduler_milestones']:
                if milestone < epoch:
                    expected_lr *= c['scheduler_gamma']
            s.require(lr == expected_lr, f'LR-schema wijkt af in epoch {epoch}: {lr} vs {expected_lr}')
            log(f'EPOCH {epoch}/{stop_after}; MODE={c["mode"]}; LR={lr:g}; volledig schema={c["planned_epochs"]}')
            before_weights = t.layer_snapshot(model, c['mode'])
            train_rows, train_s, ast_hash, grad_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['train'], epoch, 'train', lr, out / 'batch_metrics', log, c['mode'])
            steps += expected_steps
            t.validate_gradients(model)
            update_rows = t.layer_updates(before_weights, t.layer_snapshot(model, c['mode']), c['mode'], epoch)
            adam_check = t.validate_adam_state(model, optimizer, steps)
            s.legacy.finite_state(model)
            state_before_eval = s.legacy.state_digest(model.state_dict())
            test_rows, test_s, test_ast_hash, test_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['test'], epoch, 'test', lr, out / 'batch_metrics', log, c['mode'])
            s.require(ast_hash == test_ast_hash, 'Train- en evaluatiefunctie zijn niet identiek.')
            s.require(s.legacy.state_digest(model.state_dict()) == state_before_eval,
                      'Evaluatie heeft modelgewichten of BatchNorm-statistieken gewijzigd.')
            coverage = {name: loader.validate(out / 'sample_order' / f'{name}_{epoch:05d}.npy')
                        for name, loader in loaders.items()}
            for rows, expected in [(train_rows, c['train_examples']), (test_rows, c['test_examples'])]:
                s.require(all(r['n'] == expected for r in rows), 'Metric bevat niet de volledige dataset.')
            bn_check = t.validate_bn(model, steps)
            scheduler.step()  # EXACT original position: after train and test.
            s.require(scheduler.last_epoch == epoch, 'Scheduler-epoch wijkt af.')
            checkpoint = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            record = {
                'epoch': epoch, 'train': train_rows, 'test': test_rows,
                'lr_used': lr, 'lr_next': float(optimizer.param_groups[0]['lr']),
                'optimizer_steps': steps, 'coverage': coverage,
                'batchnorm_counters_correct': bn_check['all_counters_correct'],
                'gradient_checks': grad_audit, 'adam_checks': adam_check, 'weight_updates': update_rows,
                'test_optimizer_steps': test_audit['optimizer_steps_in_phase'],
                'model_sha256': state_before_eval, 'checkpoint_path': str(checkpoint.resolve()),
                'forward_ast_sha256': ast_hash, 'train_seconds': train_s, 'test_seconds': test_s,
                'first_epoch_in_process': epoch == first_epoch,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            }
            trace['epochs'].append(record)
            best = t.best_record(trace['epochs'])
            record['is_best_at_this_epoch'] = best['epoch'] == epoch
            record['best_epoch_so_far'] = best['epoch']
            t_save = time.perf_counter()
            payload = core.capture_checkpoint(model, optimizer, scheduler, {}, device, epoch, steps,
                                              ctx.contract, trace, {})
            # Empty validation_logits is intentional: scores are in trace; there
            # is NO reason to copy all test logits into every training checkpoint.
            payload['best_prec1'] = best['test'][-1]['top1_percent']
            core.atomic_save(checkpoint, payload)
            record['checkpoint_seconds'] = time.perf_counter() - t_save
            record['total_epoch_seconds'] = time.perf_counter() - t0
            record['other_seconds'] = record['total_epoch_seconds'] - train_s - test_s - record['checkpoint_seconds']
            epoch_file = out / 'epoch_summaries' / f'epoch_{epoch:05d}.json'
            s.write_json(epoch_file, record)
            epoch_file.with_suffix('.json.sha256').write_text(core.file_sha256(epoch_file) + '\n', encoding='ascii')
            ctx.report['completed_epoch'] = epoch
            log(f'KLAAR epoch {epoch}: train={train_s:.1f}s; test={test_s:.1f}s; '
                f'totaal={record["total_epoch_seconds"]:.1f}s; beste epoch={best["epoch"]}')
            log('  TEST ' + ' | '.join(f'{r["weight_format"]}/A32: '
                                    f'{r["top1_percent"]:.4f}%' for r in test_rows))
            estimate = s.estimate_time(trace['epochs'])
            if estimate['available']:
                log(f'  SCHATTING 100 epochs: {estimate["estimated_100_epoch_seconds"] / 3600:.2f} uur '
                    '(zonder download/data-inladen; geen nauwkeurigheidsvoorspelling).')
            if soft_stop['requested'] or (out / 'STOP_AFTER_EPOCH').exists():
                log('Gecontroleerd gestopt op epochgrens; brondata en checkpoints blijven behouden.')
                break
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    if resume is not None:
        s.require(core.file_sha256(resume) == resume_hash, 'Invoercheckpoint is gewijzigd.')
    return trace


def export_results(ctx, out: Path, trace: dict, log):
    import torch
    records, c, core = trace['epochs'], ctx.c, ctx.rt.core
    s.require(records and [r['epoch'] for r in records] == list(range(1, records[-1]['epoch'] + 1)),
              'Onvolledige epochgeschiedenis.')
    expected_steps = math.ceil(c['train_examples'] / c['batch_size'])
    for record in records:
        epoch = record['epoch']
        s.require(record['optimizer_steps'] == expected_steps * epoch, 'Verkeerd aantal updates.')
        expected_lr = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone < epoch: expected_lr *= c['scheduler_gamma']
        s.require(record['lr_used'] == expected_lr, 'Onjuist LR-schema.')
        audit = record['gradient_checks']
        s.require(audit['finite_gradient_checks'] == audit['detached_teacher_checks'] == expected_steps,
                  'Onvolledige gradient-/teachercontroles.')
        s.require(audit['forward_calls_E2M3'] == audit['forward_calls_E2M1'] == expected_steps,
                  'Verkeerd aantal forwards.')
        for phase, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            s.require([row['weight_bits'] for row in record[phase]] == list(a.MODES), 'Onvolledige modi.')
            for row in record[phase]:
                bit = row['weight_bits']
                expected_loss = 'recursive_soft_cross_entropy' if phase == 'train' and bit == 4 else 'hard_cross_entropy'
                s.require(row['n'] == count and row['activation_bits'] == 32 and row['bn_bank_key'] == a.BANK[bit]
                          and row['weight_format'] == a.FORMATS[bit] and row['objective_type'] == expected_loss,
                          'Onjuiste modus, loss of datadekking.')
                s.require(abs(row['top1_percent'] - row['count_top1_percent']) < 1e-4,
                          'Accuracy strookt niet met correcte aantallen.')
    best, last = t.best_record(records), records[-1]
    selected, details, mx_exports = [], {}, {}
    for role, record in [('best_E2M3_test', best), ('last_epoch', last)]:
        path = Path(record['checkpoint_path'])
        state = core.load_verified(path)
        s.require(not core.differences(state['contract'], ctx.contract), 'Checkpointcontract wijkt af.')
        s.require(state['completed_epoch'] == record['epoch'], 'Verkeerde checkpointepoch.')
        s.require(s.legacy.state_digest(state['model']) == record['model_sha256'], 'Modelhash wijkt af.')
        s.require(not core.differences(state['trace']['epochs'][-1]['test'], record['test']), 'Testmetrics verschillen.')
        suffix = 'best' if role == 'best_E2M3_test' else 'last'
        filename = suffix + '_checkpoint.pt'
        target = out / filename
        with path.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        digest = core.file_sha256(target)
        s.require(digest == core.file_sha256(path), 'Checkpointkopie wijkt af.')
        target.with_suffix('.pt.sha256').write_text(digest + '\n', encoding='ascii')
        details[role] = {'epoch': record['epoch'], 'file': filename, 'sha256': digest,
                         'original_checkpoint': str(path), 'selection_score_E2M3': record['test'][-1]['top1_percent']}
        selected.extend({'role': role, **row} for row in record['test'])
        model = copy.deepcopy(ctx.model).cpu().eval()
        model.load_state_dict(state['model'], strict=True)
        exports_for_role = {}
        for bit in (6, 4):
            f = a.codec(bit)
            payload, layer_rows = a.export_payload(model, bit, {
                'training': 'joint E2M3->E2M1 distillation; A32; FP32 masters; no truncation',
                'source_epoch': record['epoch'], 'checkpoint_sha256': digest,
                'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']})
            mx_file = out / f'mx_direct_{a.FORMATS[bit]}_{suffix}.pt'
            mx_details = core.atomic_save(mx_file, payload)
            loaded = torch.load(mx_file, map_location='cpu', weights_only=True)
            for name in a.TARGET_LAYERS:
                restored = f.MXTensor.from_payload(loaded['mx_layers'][name]).decode()
                effective = a.effective_reference_weight(model.get_submodule(name).weight.detach())
                s.require(torch.equal(restored, f.encode_tensor(effective).decode()), 'MX-exportgewichten verschillen.')
                s.require(name + '.weight' not in loaded['auxiliary_fp32_state'], 'FP32-master in export.')
            exports_for_role[str(bit)] = {**mx_details, 'epoch': record['epoch'], 'weight_bits': bit,
                                         'decoded_weights_exact': True, 'six_fp32_masters_absent': True,
                                         'full_test_forward_rerun': False, 'truncation': False}
            s.write_csv(out / f'mx_export_layers_{bit}_{suffix}.csv', layer_rows)
        mx_exports[role] = exports_for_role
    s.write_csv(out / 'selected_metrics.csv', selected)
    s.write_csv(out / 'epoch_metrics.csv', [row for r in records for phase in ('train', 'test') for row in r[phase]])
    s.write_csv(out / 'weight_updates.csv', [row for r in records for row in r['weight_updates']])
    selection = {'policy': c['selection_policy'], 'best': details['best_E2M3_test'], 'last': details['last_epoch'],
                 'same_checkpoint': best['epoch'] == last['epoch'], 'selected_on_official_test': True,
                 'independent_test_estimate': False, 'selection_mode': 'MXFP6_E2M3/A32',
                 'both_modes_from_same_selected_checkpoint': True}
    s.write_json(out / 'checkpoint_selection.json', selection)
    s.write_json(out / 'history.json', trace)
    elapsed = sum(r['total_epoch_seconds'] for r in records)
    summary = {'completed_epochs': last['epoch'], 'pilot_completed': last['epoch'] == 5,
               'full_100_epoch_training_completed': False, 'optimizer_steps': last['optimizer_steps'],
               'training_forward_passes': last['optimizer_steps'] * 2,
               'train_examples': c['train_examples'], 'test_examples': c['test_examples'],
               'mode': c['mode'], 'distillation': True, 'truncation': False,
               'initial_model_sha256': trace['initial_model_sha256'],
               'measured_completed_epoch_seconds': elapsed,
               'gradient_checks_before_every_update': True, 'fp32_masters_and_moments_verified': True,
               'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
    s.write_json(out / 'training_summary.json', summary)
    estimate = s.estimate_time(records, c['planned_epochs'])
    estimate['weight_mode'] = c['weight_format']
    estimate['includes'] = ['two-mode training', 'two-mode test', 'epoch diagnostics', 'checkpoint write']
    estimate['measured_completed_epoch_seconds'] = elapsed
    s.write_json(out / 'timing_estimate.json', estimate)
    s.write_csv(out / 'epoch_times.csv', [
        {'epoch': r['epoch'], **{k: r.get(k) for k in ['train_seconds', 'test_seconds', 'checkpoint_seconds',
          'other_seconds', 'total_epoch_seconds', 'first_epoch_in_process',
          'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']}} for r in records])
    ctx.report.update(completed_epoch=last['epoch'], optimizer_steps=last['optimizer_steps'],
                      selected_best_epoch=best['epoch'], selected_last_epoch=last['epoch'],
                      initial_model_sha256=trace['initial_model_sha256'], checkpoint_selection=selection,
                      training_summary=summary, timing_estimate=estimate, mx_exports=mx_exports,
                      synthetic_fixture_only=ctx.contract['synthetic_fixture_only'])
    notebook = json.loads((ROOT / 'MX_dual_pilot5_resultaten.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n']
    s.write_json(out / 'MX_dual_pilot5_resultaten.ipynb', notebook)
    log(f'BEHOUDEN: beide modi uit beste E2M3-checkpoint {best["epoch"]}; laatste {last["epoch"]}.')


def verify_unchanged(ctx, log):
    for name, expected in ctx.source_info['workcopy_files_sha256'].items():
        s.require(ctx.rt.core.file_sha256(ctx.source / name) == expected, 'Bronbestand gewijzigd: ' + name)
    s.verify_package()
    for name, expected in ctx.report['dataset_files']['files'].items():
        actual = s.hashes(Path(ctx.report['dataset_files']['root']) / name)
        s.require(all(actual[key] == expected[key] for key in actual), 'Ruwe dataset gewijzigd: ' + name)
    ctx.report.update(source_unchanged=True, dataset_files_unchanged=True)
    log('Eindcontrole: bronbestanden en ruwe datasetbestanden ongewijzigd.')


def feedback_bundle(out):
    names = ['run_report.json', 'epoch_metrics.csv', 'selected_metrics.csv', 'weight_updates.csv',
             'checkpoint_selection.json', 'timing_estimate.json', 'epoch_times.csv', 'history.json',
             'dataset_manifest.json', 'source_manifest.json', 'compatibility.patch',
             'pip_freeze.txt', 'pip_check.txt', 'run.log', 'warnings.log',
             'training_change_manifest.json', 'training_summary.json',
             'mx_export_layers_6_best.csv', 'mx_export_layers_4_best.csv',
             'mx_export_layers_6_last.csv', 'mx_export_layers_4_last.csv']
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        for name in names:
            path = out / name
            if path.is_file():
                z.write(path, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume'])
    parser.set_defaults(mode='joint')
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_weights_dual_pilot5'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=5)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.stop_after_epoch != 5:
        parser.error('Dit pakket is een proef van vijf epochs; geen automatische honderd-epochtraining.')
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint; run begint altijd vanaf initialisatie.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.mode + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-weights-dual-pilot5-report-v1', 'status': 'RUNNING',
              'command': args.command, 'mode': args.mode, 'started_at': s.now(),
              'output_directory': str(out), 'argv': sys.argv, 'warnings': [],
              'limitations': ['train only; two MX weight modes; this is not a paper reproduction',
                              'FP32 update master, FP32 gradients/Adam; A32; no truncation',
                              'two forward/backwards: E2M3 hard CE, E2M1 detached soft CE, one Adam step',
                              'no same-protocol accuracy comparison with the older five-mode reference',
                              'official test used for checkpoint selection, not an independent test estimate',
                              'numerical weight-only MX emulation; no native MX kernel or energy claim',
                              'GPU bit-exact repeat/resume not guaranteed; fixed seed is not proof of determinism',
                              'five epochs test learning mechanics, not final convergence']}
    log = s.legacy.Log(out / 'run.log')
    started = time.perf_counter()
    exit_code = 1
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        entry = {'category': category.__name__, 'message': str(message), 'file': filename, 'line': lineno}
        report['warnings'].append(entry)
        log('WARNING ' + str(message))
        with (out / 'warnings.log').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.showwarning = warning_handler
            log('RESULTATENMAP: ' + str(out))
            log('TWEEMODUSPROEF: E2M3 -> E2M1; A32; FP32-masters; distillatie; geen truncatie.')
            ctx = create_context(args, out, report, log, check_only=args.command == 'check')
            if ctx is None:
                report['status'] = 'PASS'
            else:
                trace = run_epochs(ctx, out, log, 5, args.checkpoint)
                export_results(ctx, out, trace, log)
                verify_unchanged(ctx, log)
                report['status'] = 'PASS' if report['completed_epoch'] == 5 else 'STOPPED_AT_EPOCH_BOUNDARY'
            exit_code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report['error'], report['traceback'] = repr(exc), traceback.format_exc()
        log(report['status'] + ': ' + str(exc))
        log(report['traceback'])
        log('Geen bestaande data of eerdere resultaten gewijzigd. Deel de feedback_bundle.zip uit deze map.')
    finally:
        report['ended_at'] = s.now()
        report['session_wall_seconds'] = time.perf_counter() - started
        log('EINDSTATUS: ' + report['status'])
        log('Rapport: ' + str(out / 'run_report.json'))
        log.handle.close()
        s.write_json(out / 'run_report.json', report)
        feedback_bundle(out)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
