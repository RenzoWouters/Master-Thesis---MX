"""A genuinely separate Python process for one segment of the resume check."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
import run_pilot as pilot
support = pilot.support
require = support.require


def setup_runtime(device_name: str, seed: int):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import numpy as np
    import resume_core as core
    pilot.torch = support.torch = torch
    pilot.np = support.np = np
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'Verwacht PyTorch 2.10.0.')
    require(importlib.metadata.version('torchvision').split('+')[0] == '0.25.0',
            'Verwacht torchvision 0.25.0.')
    device = torch.device('cuda:0' if device_name == 'cuda' else 'cpu')
    settings = support.configure(seed, device, 2)
    environment = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': str(torch.__version__), 'torchvision': importlib.metadata.version('torchvision'),
        'numpy': str(np.__version__), 'pillow': importlib.metadata.version('Pillow'),
        'device': str(device), 'cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
    }
    return torch, np, core, device, settings, environment


def prepare_data(data_root: Path, source: Path, config: dict, log):
    """Read-only use of already downloaded, checksum-verified official data."""
    dm = importlib.import_module('datasets.data')
    require(Path(dm.__file__).resolve().is_relative_to(source), 'Verkeerde datasetmodule.')
    info = pilot.ensure_dataset(data_root, dm.CIFAR10, False, log)
    dm.data_paths['cifar10'] = str(data_root)
    train_tf = dm.get_transform('cifar10', 'train')
    val_tf = dm.get_transform('cifar10', 'val')
    td = dm.get_dataset('cifar10', 'train', train_tf, download=False)
    vd = dm.get_dataset('cifar10', 'train', val_tf, download=False)
    require(td.train_labels == vd.train_labels, 'Bronlabels verschillen.')
    sp = pilot.make_split(td.train_labels, config['train_per_class'], config['val_per_class'])
    train_indices, val_indices = sp['pilot_train_indices'], sp['pilot_validation_indices']
    selection = {'source_split': sp['source_split'], 'split_seed': sp['seed'],
                 'train_indices': train_indices, 'validation_indices': val_indices,
                 'official_test_examples_used': 0}
    transforms = {'train': repr(train_tf), 'validation': repr(val_tf)}
    return (pilot.IndexedSubset(td, train_indices), pilot.IndexedSubset(vd, val_indices),
            selection, info, transforms)


def run_worker(args, report: dict, log) -> None:
    config = json.loads(args.config.read_text(encoding='utf-8'))
    require(config['purpose'] == 'technical_epoch_boundary_resume_check', 'Geen controleconfiguratie.')
    require(config['epochs'] == 3 and config['scheduler_milestones'] == [1, 2], 'Onverwacht testbudget.')
    require(1 <= args.stop_after_epoch <= config['epochs'], 'Ongeldige sessielimiet.')
    source = args.source.resolve()
    source_info = json.loads(args.source_manifest.read_text(encoding='utf-8'))
    for name, expected in source_info['workcopy_files_sha256'].items():
        require(support.sha256(source / name) == expected, 'Bronbestand gewijzigd: ' + name)
    torch, np, core, device, settings, environment = setup_runtime(args.device, config['seed'])
    report.update(config=config, environment=environment, execution_settings=settings,
                  pid=os.getpid(), parent_pid=os.getppid(), resumed_from=None)
    log('Apparaat: ' + environment['device_name'])
    make_model, SoftCE, get_optimizer, _ = support.import_source(source)
    td, vd, selection, data_info, transforms = prepare_data(args.data_root.resolve(), source, config, log)
    report['dataset_files'] = data_info
    report['selection'] = selection
    report['transforms'] = transforms
    support.json_write(args.out / 'selection.json', selection)
    code_hashes = {n: core.file_sha256(ROOT / n) for n in
                  ['resume_core.py', 'resume_worker.py', 'run_pilot.py', 'smoke_support.py']}
    contract = {'config': config, 'environment': environment, 'settings': settings,
                'source_files': source_info['workcopy_files_sha256'], 'code_hashes': code_hashes,
                'selection': selection, 'dataset_batch_files': data_info['batch_files'],
                'transforms': transforms,
                'synthetic_fixture_only': bool(data_info.get('synthetic_test_fixture_only', False))}
    report['synthetic_fixture_only'] = contract['synthetic_fixture_only']
    report['contract_sha256'] = core.json_digest(contract)
    # Restore occurs only AFTER model/optimizer/scheduler/loader construction.
    support.reseed(config['seed'])
    model = make_model(pilot.BITS).to(device)
    require(model.expand == 5 and sum(p.numel() for p in model.parameters()) == 6786330,
            'Onverwachte architectuur.')
    optimizer = get_optimizer(model, 'adam', config['lr'], 0.0)
    scheduler = importlib.import_module('optimizer').get_lr_scheduler(
        'adam', optimizer, config['scheduler_milestones'])
    soft_ce = SoftCE()
    generators = {'train': torch.Generator().manual_seed(config['seed'] + 11),
                  'validation': torch.Generator().manual_seed(config['seed'] + 12)}
    train_loader = torch.utils.data.DataLoader(td, batch_size=config['batch_size'], shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False, generator=generators['train'])
    val_loader = torch.utils.data.DataLoader(vd, batch_size=config['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, generator=generators['validation'])
    expected_steps = math.ceil(len(td) / config['batch_size'])
    require(expected_steps == 3 and len(td) == len(vd) == 20, 'Onverwachte omvang van de kleine controle.')
    steps, done = 0, 0
    trace: dict[str, Any] = {'initial_model_sha256': support.state_digest(model.state_dict()), 'epochs': []}
    if args.resume:
        state = core.load_verified(args.resume.resolve())
        require(0 < state['completed_epoch'] < args.stop_after_epoch, 'Geen nieuwe epochs te hervatten.')
        # Deliberately disturb all RNGs before restoring them, to avoid a false
        # pass caused by coincidentally reconstructing the same starting seeds.
        support.reseed(config['seed'] + 99999)
        for i, g in enumerate(generators.values()):
            g.manual_seed(config['seed'] + 88888 + i)
        report['restore_checks'] = core.restore_checkpoint(
            state, model, optimizer, scheduler, generators, device, contract)
        trace = state['trace']
        done, steps = int(state['completed_epoch']), int(state['optimizer_steps'])
        require(steps == done * expected_steps, 'Verkeerd aantal opgeslagen optimizerstappen.')
        require(len(trace['epochs']) == done, 'Onvolledige opgeslagen meetgeschiedenis.')
        report['resumed_from'] = str(args.resume.resolve())
        report['resume_input_sha256'] = core.file_sha256(args.resume.resolve())
        log(f'Checkpoint hersteld: epoch {done}, {steps} stappen; volgende epoch {done + 1}.')
        del state
    else:
        log('Nieuwe initialisatie; dit gebruikt GEEN eerder pilotcheckpoint.')
    report['first_epoch_this_process'] = done + 1
    report['epoch_times_diagnostic_only'] = []
    report['checkpoints'] = []
    with (args.out / 'batch_metrics.jsonl').open('x', encoding='utf-8') as bf:
        for epoch in range(done + 1, args.stop_after_epoch + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            lr = float(optimizer.param_groups[0]['lr'])
            expected_lr = config['lr']
            for milestone in config['scheduler_milestones']:
                if milestone < epoch:
                    expected_lr *= 0.1
            require(lr == expected_lr, f'Verkeerde learning rate in epoch {epoch}: {lr} vs {expected_lr}')
            rng_before = core.rng_probe(generators, device)
            acc = {bit: [] for bit in pilot.BITS}
            order, batch_records = [], []
            log(f'  mini-epoch {epoch}/3; learning rate={lr:.8g}')
            # Continuous RNG streams: NO per-epoch reseeding; saved states matter.
            for batch, (x, y, indices) in enumerate(train_loader, 1):
                order.extend(indices.tolist())
                x_hash = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
                y_hash = hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest()
                records = pilot.train_batch(model, optimizer, x.to(device), y.to(device), soft_ce)
                steps += 1
                rec = {'epoch': epoch, 'batch': batch, 'step': steps, 'lr': lr,
                       'original_indices': indices.tolist(), 'inputs_sha256': x_hash,
                       'labels_sha256': y_hash, 'precision_records': records}
                batch_records.append(rec)
                bf.write(json.dumps(rec, allow_nan=False) + '\n')
                bf.flush()
                for r in records:
                    acc[r['bits_w_and_a']].append(r)
                log(f'    batch {batch}/{len(train_loader)}: alle 5 precisies, gradients eindig, stap {steps}')
            require(sorted(order) == sorted(selection['train_indices']), 'Onjuiste gegevensvolgorde/dekking.')
            metrics_train = pilot.summarize(acc, epoch, 'train', lr)
            metrics_val, logits = pilot.evaluate(model, val_loader, device, epoch, lr)
            support.bn_check(model, steps)
            scheduler.step()
            require(scheduler.last_epoch == epoch, 'Learning-rateschema dubbel/verkeerd doorgeschoven.')
            epoch_trace = {'epoch': epoch, 'rng_before': rng_before, 'order': order,
                           'batches': batch_records, 'train': metrics_train, 'validation': metrics_val,
                           'lr_used': lr, 'lr_next': float(optimizer.param_groups[0]['lr']),
                           'optimizer_steps': steps, 'model_sha256': support.state_digest(model.state_dict())}
            trace['epochs'].append(epoch_trace)
            payload = core.capture_checkpoint(model, optimizer, scheduler, generators, device,
                       epoch, steps, contract, trace, logits)
            entry = core.atomic_save(args.out / f'epoch_{epoch:03d}.pt', payload)
            report['checkpoints'].append(entry)
            del payload
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            report['epoch_times_diagnostic_only'].append({'epoch': epoch, 'seconds': time.perf_counter() - t0})
            log(f'  Checkpoint epoch {epoch} opgeslagen; volgende learning rate={optimizer.param_groups[0]["lr"]:.8g}.')
    for name, expected in source_info['workcopy_files_sha256'].items():
        require(core.file_sha256(source / name) == expected, 'Broncode gewijzigd tijdens uitvoering.')
    if args.resume:
        require(core.file_sha256(args.resume.resolve()) == report['resume_input_sha256'],
                'Het geladen checkpoint is gewijzigd.')
    support.json_write(args.out / 'trace.json', trace)
    report['completed_epoch'] = args.stop_after_epoch
    report['optimizer_steps'] = steps
    report['source_unchanged'] = True
    report['all_precision_gradients_and_states_finite'] = True
    report['last_checkpoint'] = f'epoch_{args.stop_after_epoch:03d}.pt'
    report['next_epoch'] = args.stop_after_epoch + 1
    if device.type == 'cuda':
        report['peak_cuda_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024**2
    log('Dit Python-proces stopt nu normaal; geen handmatige onderbreking nodig.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--stop-after-epoch', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    log = support.Log(args.out / 'worker.log')
    report: dict[str, Any] = {'status': 'FAIL', 'purpose': 'technical_resume_check_worker',
        'started_at': datetime.now().astimezone().isoformat(), 'argv': sys.argv,
        'not_reference_training': True, 'mid_epoch_resume_tested': False}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            run_worker(args, report, log)
            report['status'] = 'PASS'
        except (Exception, KeyboardInterrupt) as exc:
            report['error'] = str(exc)
            report['traceback'] = traceback.format_exc()
            log('FOUT: ' + report['error'])
            log(report['traceback'])
        finally:
            report['warnings'] = [{'category': x.category.__name__, 'message': str(x.message),
                                   'filename': x.filename, 'line': x.lineno} for x in caught]
            report['seconds_diagnostic_only'] = time.perf_counter() - t0
            report['ended_at'] = datetime.now().astimezone().isoformat()
            support.json_write(args.out / 'worker_report.json', report)
            log(f'Workerstatus: {report["status"]}; waarschuwingen: {len(caught)}')
            log.handle.close()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
