"""Five train-only epochs: group size16, fixed scale64 route, E4M3 source, exponent truncation + scale64, A32.

E4M3 is encoded once per layer/minibatch from current FP32 G(W).
E2M3 keeps the highest exponent bits; E2M1 then keeps the highest mantissa bit.
Both lower modes use the current E4M3 source scale exponent plus6 (times64).
Upstream recursive distillation: E4M3 hard CE; E2M3 learns from detached E4M3;
E2M1 learns from detached E2M3; summed gradients then ONE Adam update.
Fixed +6 scale offset, no learned scale or warm-start. Whole-path identity STE through G(W).
The unchanged 100-epoch LR schedule is configured, but this pilot stops at 5.
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
import trunc_support as t
import trunc_adapter as a
from group_config import BLOCK_SIZE
import group_costs

ROOT = Path(__file__).resolve().parent


def config(mode: str = 'joint') -> dict:
    s.require(mode == 'joint', 'Deze proef heeft exact drie gezamenlijke MX-modi uit één E4M3-bron.')
    return {
        'schema': 'mx-triple-group16-scale64-pilot5-protocol-v1',
        'purpose': 'five complete train-only epochs: group size16, fixed scale64 route, E4M3 source, naive exponent truncation plus fixed scale64 to E2M3, then mantissa truncation to E2M1',
        'mode': mode, 'weight_format': t.weight_format(mode), 'weight_bits': [4, 6, 8],
        'weight_formats': {'8': 'MXFP8_E4M3', '6': 'MXFP6_E2M3', '4': 'MXFP4_E2M1'},
        'codec_revision': {'8': 'e4m3-exact-binary-v2', '6': 'unchanged-E2M3-decode', '4': 'unchanged-E2M1-decode'},
        'weight_preprocess': 'original W32 G(W); detached mean(abs(W)), tanh/max normalization',
        'activation_bits': 32, 'activation_policy': 'original A32/ReLU in ALL THREE modes',
        'master_dtype': 'float32', 'gradient_dtype': 'float32', 'adam_moments_dtype': 'float32',
        'mx_block_size': BLOCK_SIZE, 'mx_scale_format': 'E8M0',
        'mx_group_size_research_variant': True, 'comparison_baseline_block_size': 32,
        'mx_group_axis': 'flatten Cin*Kh*Kw within each Cout, zero-pad tail',
        'mx_scale_policy': 'encode current E4M3 source scale floor(log2(amax))-8; clip[-127,127], zero block exponent0; both lower scale codes equal source code+6; integer promoted before addition; overflow above254 rejected without clipping/wrap, including zero blocks; source scales can change after optimizer steps',
        'mx_rounding': 'E4M3 source nearest ties-to-even; finite FP32 source, scalar subnormal flush; lower modes exact bit mappings without additional rounding',
        'ste_policy': 'identity wrt G(W) for E4M3 and complete scale64 E4-to-E2 code path; no extra factor64 in backward; retain original G(W) derivatives; no gradient through codes or scales',
        'numeric_kernel': 'FP32 F.conv2d; codec FP64 intermediates; not native low-bit kernels',
        'truncation': True, 'mantissa_truncation': True, 'exponent_truncation': True,
        'shared_scale_compensation': True, 'learned_shared_scale': False,
        'scale_exponent_offset': 6, 'scale_factor': 64,
        'quantization_source': 'one current E4M3 encoding per MX layer and minibatch, shared across all three modes',
        'source_weight_bits': 8,
        'exponent_mapping': 's e3 e2 e1 e0 m2 m1 m0 -> s e3 e2 m2 m1 m0; target bias1; no bias compensation',
        'mantissa_mapping': 'derived E2M3 uint8 codes >> 2; unchanged E2M3 scale equals original E4M3 scale times64',
        'source_cache_policy': 'detached integer source and derived codes; one triplet per minibatch; cleared before Adam',
        'direct_lower_quantization': False, 'exponent_bias_compensation': False,
        'distillation': True,
        'loss': 'E4M3 hard CE + E2M3 soft CE from detached E4M3 + E2M1 soft CE from detached E2M3; T=1; gradients summed, no division by 3',
        'passes_per_training_batch': 3, 'optimizer_steps_per_training_batch': 1,
        'training_mode_order': [8, 6, 4], 'test_mode_order': [4, 6, 8],
        'internal_upstream_selectors': [4, 6, 8], 'legacy_abits_and_wbits': 32,
        'model_bank_structure': s.BITS, 'bn_bank_mapping': {'8': '8', '6': '32', '4': '4'},
        'batchnorm_policy': 'normal training updates of three mode banks E4M3=8,E2M3=32,E2M1=4; parameters and running stats learn per mode; eval frozen; keys are storage IDs, all A32; banks1/2 unused',
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
        'evaluation_policy': 'all three modes on complete official test after every epoch; hard CE for each',
        'selection_policy': 'first strict maximum of E4M3/A32 official-test top1',
        'reporting_policy': 'all three modes from same selected E4M3-best checkpoint; also all at last pilot epoch',
        'backend_policy': 'github: cudnn.benchmark=True; no deterministic override',
        'rng_policy': 'global streams as upstream; sequential dropout draws per mode; saved at epoch boundaries',
        'checkpoint_policy': 'immutable FP32 training state per epoch and selected best/last copies',
        'export_policy': 'one E4M3 source with all BN auxiliary state per selected checkpoint; lower exports derived only from saved/reloaded source codes and source scale code+6; separate FP32 resume checkpoint',
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
    s.require(BLOCK_SIZE == 16 and a.SCALE_FACTOR == 64 and a.SCALE_EXPONENT_OFFSET == 6,
              'Dit pilotpakket vereist groepsgrootte16 en vaste schaalfactor64.')
    report['group_size_costs'] = group_costs.write_cost_report(out, active_group_size=BLOCK_SIZE)
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
        import test_e4m3
        import test_trunc_training
        import test_group_size
        report['codec_tests'] = {'E2M3': {}, 'E4M3': {}}
        try:
            report['codec_tests']['E2M3'] = test_mx.run_tests(modules, str(rt.device))
            test_e4m3.run_tests(str(rt.device), report=report['codec_tests']['E4M3'])
        except BaseException as exc:
            report['codec_tests'].update(status='FAIL', error=repr(exc))
            raise
        else:
            report['codec_tests']['status'] = 'PASS'
        finally:
            s.write_json(out / 'codec_report.json', report['codec_tests'])
        report['group_size_tests'] = test_group_size.run_tests(str(rt.device))
        s.write_json(out / 'group_size_check_report.json', report['group_size_tests'])
        report['synthetic_training_checks'] = {}
        try:
            test_trunc_training.run_tests(modules, rt, source, log, checks=report['synthetic_training_checks'])
        except BaseException as exc:
            report['synthetic_training_checks'].update(status='FAIL', error=repr(exc))
            raise
        else:
            report['synthetic_training_checks']['status'] = 'PASS'
        finally:
            s.write_json(out / 'trunc_training_check_report.json', report['synthetic_training_checks'])
        _, forward_hash = t.load_joint_forward(source, modules.utils, log)
        report.update(forward_ast_sha256=forward_hash, dataset_downloaded=False,
                      synthetic_checks_only=True)
        report['notebook_dependencies'] = {}
        for name in ['pandas', 'matplotlib', 'ipykernel']:
            try:
                report['notebook_dependencies'][name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                report['notebook_dependencies'][name] = 'MISSING: notebook only'
        log('CHECK PASS: fixed scale64 route; E4M3-bron, exponentbits, vaste schaaloffset+6, mantissatruncatie, identity STE, A32, drie BN-banken en recursieve updates. Geen SVHN-training.')
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
    observed_shapes = {name: tuple(model.get_submodule(name).weight.shape) for name in a.TARGET_LAYERS}
    geometry = group_costs.build_cost_report(layer_shapes=observed_shapes, active_group_size=BLOCK_SIZE)
    s.require(geometry == report['group_size_costs'], 'Werkelijke MX-laagvormen wijken af van de kostenanalyse.')
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
            if not any(row['master_changed_values'] > 0 for row in update_rows):
                log('DIAGNOSE: geen gewijzigde MX-masterwaarden in deze epoch; eindige gradients en Adamstappen worden afzonderlijk gecontroleerd.')
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
                'test_source_checks': test_audit['shared_source_checks'],
                'test_prediction_counts': test_audit['prediction_counts'],
                'test_label_counts': test_audit['label_counts'],
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
            for bit, counts in record['test_prediction_counts'].items():
                if max(counts) == sum(counts):
                    log(f'  DIAGNOSE {a.FORMATS[int(bit)]}: alle testbeelden krijgen klasse {counts.index(max(counts))}; meetresultaat, geen technische FAIL.')
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
    s.require(c['planned_epochs'] == 100 and c['pilot_stop_epoch'] == 5, 'Onjuist vijf-epochpilotcontract.')
    s.require(0 < records[-1]['epoch'] <= 5, 'Pilot moet uiterlijk na vijf epochs stoppen.')
    for record in records:
        epoch = record['epoch']
        s.require(record['optimizer_steps'] == expected_steps * epoch, 'Verkeerd aantal updates.')
        expected_lr = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone < epoch: expected_lr *= c['scheduler_gamma']
        s.require(record['lr_used'] == expected_lr, 'Onjuist LR-schema.')
        expected_next = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone <= epoch: expected_next *= c['scheduler_gamma']
        s.require(record['lr_next'] == expected_next, 'Onjuist LR voor de volgende epoch.')
        s.require(record['batchnorm_counters_correct'], 'BN-tellers niet correct.')
        s.require(record['test_optimizer_steps'] == 0, 'Evaluatie deed optimizerupdates.')
        s.require(record['adam_checks']['all_active_adam_steps_correct']
                  and record['adam_checks']['moments_fp32_finite']
                  and record['adam_checks']['unused_bn_has_no_optimizer_state'],
                  'Onjuiste Adamstaat.')
        audit = record['gradient_checks']
        s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == expected_steps,
                  'Onvolledige gradientcontroles.')
        s.require(audit['detached_teacher_checks'] == 2 * expected_steps,
                  'Er moeten twee losgekoppelde teachercontroles per minibatch zijn.')
        s.require(audit['forward_calls_E4M3'] == audit['forward_calls_E2M3']
                  == audit['forward_calls_E2M1'] == expected_steps, 'Verkeerd aantal forwards.')
        s.require(audit['cache_empty_before_all_optimizer_steps'],
                  'Broncache niet leeg voor optimizerupdate.')
        for phase_name, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            source_audit = (audit['shared_source_checks'] if phase_name == 'train'
                            else record['test_source_checks'])
            batches = math.ceil(count / c['batch_size'])
            s.require(source_audit['source_triplets'] == batches
                      and source_audit['high_encodings'] == source_audit['exponent_derivations']
                      == source_audit['mantissa_derivations'] == batches * len(a.TARGET_LAYERS)
                      and source_audit['scale_offsets_plus6'] == 2 * batches * len(a.TARGET_LAYERS)
                      and source_audit['mantissa_scale_copies_unchanged'] == batches * len(a.TARGET_LAYERS)
                      and source_audit['scale_exponent_offset'] == a.SCALE_EXPONENT_OFFSET and source_audit['scale_factor'] == a.SCALE_FACTOR
                      and source_audit['source_cache_cleared']
                      and source_audit['no_direct_lower_quantizer']
                      and source_audit['cache_cleared_before_adam'],
                      'Onvolledige gedeelde E4M3-broncontrole: ' + phase_name)
        s.require(set(record['test_prediction_counts']) == {'8', '6', '4'},
                  'Voorspellingsverdelingen ontbreken.')
        for counts in [record['test_label_counts'], *record['test_prediction_counts'].values()]:
            s.require(len(counts) == 10 and all(type(n) is int and n >= 0 for n in counts)
                      and sum(counts) == c['test_examples'], 'Onvolledige klasseverdeling.')
        for phase, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            s.require([row['weight_bits'] for row in record[phase]] == list(a.MODES), 'Onvolledige modi.')
            for row in record[phase]:
                bit = row['weight_bits']
                expected_loss = 'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy'
                expected_teacher = a.FORMATS[8 if bit == 6 else 6] if phase == 'train' and bit != 8 else None
                s.require(row['n'] == count and row['activation_bits'] == 32 and row['bn_bank_key'] == a.BANK[bit]
                          and row['weight_format'] == a.FORMATS[bit] and row['objective_type'] == expected_loss
                          and row['teacher_weight_format'] == expected_teacher,
                          'Onjuiste modus, loss of datadekking.')
                s.require(abs(row['top1_percent'] - row['count_top1_percent']) < 1e-4,
                          'Accuracy strookt niet met correcte aantallen.')
    best, last = t.best_record(records), records[-1]
    selected, details, mx_exports = [], {}, {}
    for role, record in [('best_E4M3_test', best), ('last_epoch', last)]:
        path = Path(record['checkpoint_path'])
        state = core.load_verified(path)
        s.require(not core.differences(state['contract'], ctx.contract), 'Checkpointcontract wijkt af.')
        s.require(state['completed_epoch'] == record['epoch'], 'Verkeerde checkpointepoch.')
        s.require(s.legacy.state_digest(state['model']) == record['model_sha256'], 'Modelhash wijkt af.')
        s.require(not core.differences(state['trace']['epochs'][-1]['test'], record['test']), 'Testmetrics verschillen.')
        suffix = 'best' if role == 'best_E4M3_test' else 'last'
        filename = suffix + '_checkpoint.pt'
        target = out / filename
        with path.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        digest = core.file_sha256(target)
        s.require(digest == core.file_sha256(path), 'Checkpointkopie wijkt af.')
        target.with_suffix('.pt.sha256').write_text(digest + '\n', encoding='ascii')
        details[role] = {'epoch': record['epoch'], 'file': filename, 'sha256': digest,
                         'original_checkpoint': str(path), 'selection_score_E4M3': record['test'][-1]['top1_percent']}
        selected.extend({'role': role, **row} for row in record['test'])
        # Encode and compare on the training device: G(W) can differ after moving to CPU.
        model = copy.deepcopy(ctx.model).to(ctx.rt.device).eval()
        model.load_state_dict(state['model'], strict=True)
        exports_for_role = {}
        provenance = {
            'training': 'joint fixed scale64 route: E4M3 source -> exponent-MSB E2M3 with scale64 -> mantissa E2M1; whole-path identity STE; A32',
            'source_epoch': record['epoch'], 'checkpoint_sha256': digest,
            'encoding_device': str(ctx.rt.device),
            'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
        source_payload, all_layer_rows = a.source_payload(model, provenance)
        source_file = out / f'mx_source_MXFP8_E4M3_{suffix}.pt'
        source_details = core.atomic_save(source_file, source_payload)
        # Derivation starts from the verified saved source. The compact source
        # has all BN/FP32 auxiliary state but no masters for the six MX layers.
        loaded_source = a.load_verified_payload(source_file)
        a.validate_source(loaded_source)
        source_digest = core.file_sha256(source_file)
        for bit in (8, 6, 4):
            f = a.codec(bit)
            if bit == 8:
                mx_file, mx_details, loaded = source_file, source_details, loaded_source
            else:
                payload = a.derive_payload(loaded_source, bit, source_digest)
                mx_file = out / f'mx_derived_{a.FORMATS[bit]}_{suffix}.pt'
                mx_details = core.atomic_save(mx_file, payload)
                loaded = a.load_verified_payload(mx_file)
            a.select_mode(model, bit)
            for name in a.TARGET_LAYERS:
                encoded = f.MXTensor.from_payload(loaded['mx_layers'][name], ctx.rt.device)
                restored = encoded.decode()
                layer = model.get_submodule(name)
                with torch.no_grad():
                    live = layer.quantize_fn(layer.weight.detach())
                s.require(torch.equal(restored, live), 'Export verschilt van dezelfde trainingsroute.')
                source_scale = loaded_source['mx_layers'][name]['scale_codes']
                expected_scale = source_scale.cpu().to(torch.int32) + (0 if bit == 8 else a.SCALE_EXPONENT_OFFSET)
                s.require(bool((expected_scale <= 254).all())
                          and torch.equal(encoded.scale_codes.cpu().to(torch.int32), expected_scale),
                          'Export heeft niet de voorgeschreven E4-bronschaal of exacte offset+6.')
                s.require(name + '.weight' not in loaded['auxiliary_fp32_state'], 'FP32-master in export.')
            exports_for_role[str(bit)] = {**mx_details, 'epoch': record['epoch'], 'weight_bits': bit,
                                         'decoded_weights_exact': True, 'six_fp32_masters_absent': True,
                                         'full_test_forward_rerun': False, 'truncation': bit != 8,
                                         'source_weight_bits': 8, 'is_source_export': bit == 8,
                                         'derived_from_saved_source': bit != 8,
                                         'source_file_sha256': source_digest,
                                         'source_scale_bytes_copied': bit == 8,
                                         'scale_exponent_offset': 0 if bit == 8 else a.SCALE_EXPONENT_OFFSET,
                                         'scale_factor': 1 if bit == 8 else a.SCALE_FACTOR,
                                         'shared_scale_compensation': bit != 8,
                                         'learned_shared_scale': False,
                                         'master_weights_read_to_derive': False,
                                         'master_weights_read_to_encode_source': bit == 8,
                                         'saved_and_reloaded': True,
                                         'encoding_and_comparison_device': str(ctx.rt.device),
                                         'export_validation_scope': 'exact weights versus live shared-source route on training device; no full test rerun'}
            layer_rows = [row for row in all_layer_rows if row['weight_bits'] == bit]
            s.require(len(layer_rows) == len(a.TARGET_LAYERS), 'Onvolledige exportdiagnostiek.')
            cost_audit = group_costs.validate_export_layers(layer_rows, bit, expected_group_size=BLOCK_SIZE)
            exports_for_role[str(bit)]['group_size_cost_audit'] = cost_audit
            exports_for_role[str(bit)]['mx_block_size'] = BLOCK_SIZE
            s.write_csv(out / f'mx_export_layers_{bit}_{suffix}.csv', layer_rows)
        s.require(core.file_sha256(source_file) == source_digest, 'Opgeslagen E4M3-bron gewijzigd tijdens afleiding.')
        mx_exports[role] = exports_for_role
    s.write_csv(out / 'selected_metrics.csv', selected)
    s.write_csv(out / 'epoch_metrics.csv', [row for r in records for phase in ('train', 'test') for row in r[phase]])
    s.write_csv(out / 'weight_updates.csv', [row for r in records for row in r['weight_updates']])
    selection = {'policy': c['selection_policy'], 'best': details['best_E4M3_test'], 'last': details['last_epoch'],
                 'same_checkpoint': best['epoch'] == last['epoch'], 'selected_on_official_test': True,
                 'independent_test_estimate': False, 'selection_mode': 'MXFP8_E4M3/A32',
                 'all_modes_from_same_selected_checkpoint': True}
    s.write_json(out / 'checkpoint_selection.json', selection)
    s.write_json(out / 'history.json', trace)
    prediction_rows = []
    for record in records:
        for bit in a.MODES:
            counts = record['test_prediction_counts'][str(bit)]
            prediction_rows.append({'epoch': record['epoch'], 'weight_bits': bit,
                'weight_format': a.FORMATS[bit], 'n': sum(counts),
                **{f'predicted_class_{k}': count for k, count in enumerate(counts)},
                'classes_predicted': sum(n > 0 for n in counts),
                'dominant_predicted_class': counts.index(max(counts)),
                'dominant_prediction_fraction': max(counts) / sum(counts)})
    s.write_csv(out / 'test_prediction_counts.csv', prediction_rows)
    elapsed = sum(r['total_epoch_seconds'] for r in records)
    pilot_completed = last['epoch'] == c['pilot_stop_epoch']
    full_completed = False  # This executable cannot perform the full 100-epoch experiment.
    summary = {'schema': 'mx-triple-group16-scale64-pilot5-completion-v1',
               'planned_epochs': c['planned_epochs'], 'pilot_stop_epoch': c['pilot_stop_epoch'],
               'completed_epochs': last['epoch'], 'pilot_completed': pilot_completed,
               'full_training_completed': full_completed,
               'full_100_epoch_training_completed': full_completed,
               'optimizer_steps': last['optimizer_steps'],
               'expected_optimizer_steps_at_pilot_completion': expected_steps * c['pilot_stop_epoch'],
               'expected_training_forward_passes_at_pilot_completion': expected_steps * c['pilot_stop_epoch'] * 3,
               'best_epoch_E4M3_A32': best['epoch'], 'last_epoch': last['epoch'],
               'lr_schedule_verified': True, 'all_epoch_modes_complete': True,
               'resumed': 'resumed_from' in ctx.report,
               'new_initialization_not_previous_experiment_warmstart': True,
               'training_forward_passes': last['optimizer_steps'] * 3,
               'train_examples': c['train_examples'], 'test_examples': c['test_examples'],
               'mode': c['mode'], 'weight_formats': c['weight_formats'], 'activation_bits': 32,
               'mx_block_size': BLOCK_SIZE, 'group_size_costs_file': 'group_size_costs.json',
               'distillation': True, 'truncation': True, 'source_weight_bits': 8,
               'exponent_truncation': True, 'mantissa_truncation': True,
               'shared_scale_compensation': True, 'learned_shared_scale': False,
               'scale_exponent_offset': 6, 'scale_factor': 64,
               'quantization_source': c['quantization_source'],
               'initial_model_sha256': trace['initial_model_sha256'],
               'measured_completed_epoch_seconds': elapsed,
               'gradient_checks_before_every_update': True, 'fp32_masters_and_moments_verified': True,
               'one_E4_source_per_layer_per_minibatch_verified': True,
               'all_lower_scale_bytes_equal_current_E4_source': False,
               'all_lower_scale_codes_equal_current_E4_source_plus6': True,
               'all_mantissa_scale_bytes_equal_E2M3': True,
               'source_cache_cleared_before_every_optimizer_update': True,
               'batchnorm_learned_during_training_eval_frozen': True,
               'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
    s.write_json(out / 'training_summary.json', summary)
    estimate = s.estimate_time(records, c['planned_epochs'])
    estimate['weight_mode'] = c['weight_format']
    estimate['includes'] = ['three-mode training', 'three-mode test', 'epoch diagnostics', 'checkpoint write']
    estimate['measured_completed_epoch_seconds'] = elapsed
    estimate['pilot_completed'] = pilot_completed
    estimate['full_training_completed'] = full_completed
    estimate['full_100_epoch_training_completed'] = False
    estimate['measured_100_epoch_seconds'] = None
    estimate['interpretation'] = ('Measured pilot time covers the completed epochs. The 100-epoch estimate is a '
                                  'projection for a fresh run under comparable conditions, not measured or remaining time.')
    s.write_json(out / 'timing_estimate.json', estimate)
    s.write_csv(out / 'epoch_times.csv', [
        {'epoch': r['epoch'], **{k: r.get(k) for k in ['train_seconds', 'test_seconds', 'checkpoint_seconds',
          'other_seconds', 'total_epoch_seconds', 'first_epoch_in_process',
          'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']}} for r in records])
    ctx.report.update(completed_epoch=last['epoch'], optimizer_steps=last['optimizer_steps'],
                      selected_best_epoch=best['epoch'], selected_last_epoch=last['epoch'],
                      initial_model_sha256=trace['initial_model_sha256'], checkpoint_selection=selection,
                      training_summary=summary, timing_estimate=estimate, mx_exports=mx_exports,
                      synthetic_fixture_only=ctx.contract['synthetic_fixture_only'],
                      full_training_completed=full_completed, pilot_completed=pilot_completed,
                      training_forward_passes=last['optimizer_steps'] * 3)
    notebook_name = 'MX_triple_group16_scale64_pilot5_resultaten.ipynb'
    notebook = json.loads((ROOT / notebook_name).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n',
                              'SAVE_FIGURES = False  # Zet op True om figuren in een nieuwe analysemap te bewaren.\n']
    s.write_json(out / notebook_name, notebook)
    for helper in ['notebook_support.py', 'group_costs.py']:
        with (ROOT / helper).open('rb') as src, (out / helper).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    # Archive unchanged real reference data beside the notebook; comparison uses epochs1-5 only.
    shutil.copytree(ROOT / 'scale64_reference', out / 'scale64_reference')
    log(f'BEHOUDEN: alle drie modi uit beste E4M3-checkpoint {best["epoch"]}; laatste {last["epoch"]}.')
    log(f'PILOT: {last["epoch"]}/5 epochs; {last["optimizer_steps"]} updates; '
        f'{last["optimizer_steps"] * 3} trainingspasses; gemeten epochtijd={elapsed / 60:.2f} min.')


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
             'test_prediction_counts.csv',
             'codec_report.json', 'trunc_training_check_report.json', 'group_size_check_report.json',
             'group_size_costs.json', 'group_size_costs_layers.csv', 'group_size_costs_summary.csv', 'group_costs.py',
             'MX_triple_group16_scale64_pilot5_resultaten.ipynb', 'notebook_support.py']
    names += [f'mx_export_layers_{bit}_{role}.csv' for bit in a.MODES for role in ('best', 'last')]
    if (out / 'scale64_reference').is_dir():
        names += [str(path.relative_to(out)).replace('\\', '/')
                  for path in sorted((out / 'scale64_reference').iterdir()) if path.is_file()]
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
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_triple_group16_scale64_pilot5'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=5)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.stop_after_epoch != 5:
        parser.error('Dit pilotpakket stopt vast na vijf epochs. Gebruik Ctrl+C of STOP_AFTER_EPOCH voor een eerdere hervatbare stop.')
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint; run begint altijd vanaf initialisatie.')
    # CHECK ONLY: configure cuBLAS before any CUDA context is initialized.
    # This Python process cannot modify the parent PowerShell environment.
    # The run/resume paths never set this variable.
    if args.command == 'check':
        import os
        previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if previous_workspace not in (':4096:8', ':16:8'):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.mode + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-triple-group16-scale64-pilot5-report-v1', 'status': 'RUNNING',
              'package_revision': 'mx_triple_group16_scale64_pilot5_v2',
              'command': args.command, 'mode': args.mode, 'started_at': s.now(),
              'output_directory': str(out), 'argv': sys.argv, 'warnings': [],
              'limitations': ['train only; shared E4M3 source and two code-derived MX modes; this is not a paper reproduction',
                              'FP32 update master/gradients/Adam; A32; exponent MSB and mantissa truncation, lower scales exactly current E4 source scale times64',
                              'three forward/backwards: E4M3 hard CE, E2M3/E2M1 recursive detached soft CE, one Adam step',
                              'BN banks learn during training; fixed scale exponent offset+6; no element-bias correction or learned scale',
                              'five-epoch pilot is not evidence about final100epoch convergence; low accuracy is a valid measured outcome',
                              'official test used for checkpoint selection, not an independent test estimate',
                              'numerical weight-only MX emulation with research group size16; no native MX kernel or energy claim',
                               'costs: packed six-layer payload including row padding; excludes FP32 auxiliary state and file/container overhead',
                              'GPU bit-exact repeat/resume not guaranteed; fixed seed is not proof of determinism',
                              'one seed, five train-only pilot epochs; no proof of general superiority or convergence']}
    if args.command == 'check':
        report['check_only_workspace'] = {
            'before': previous_workspace,
            'during_check_process': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'training_process_changed': False}
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
            log('GEZAMENLIJKE GROEP16-SCHAAL64-PILOT: 5 epochs; groepen van16; E4M3 -> E2M3 -> E2M1; lagere schalen exact E4-bronschaal x64; A32; FP32-masters.')
            ctx = create_context(args, out, report, log, check_only=args.command == 'check')
            if ctx is None:
                report['status'] = 'PASS'
            else:
                trace = run_epochs(ctx, out, log, args.stop_after_epoch, args.checkpoint)
                export_results(ctx, out, trace, log)
                verify_unchanged(ctx, log)
                report['status'] = 'PASS' if report['pilot_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY'
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
