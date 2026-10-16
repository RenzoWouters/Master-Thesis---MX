"""Five COMPLETE SVHN epochs using the supplied Any-Precision training function.

Production: CUDA, complete SVHN train ONLY, no extra, no MX, no early LR decay.
The 100-epoch protocol is fixed; this package stops no later than epoch 5.
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

ROOT = Path(__file__).resolve().parent


def config() -> dict:
    return {
        'schema': 'anyprecision-svhn-trainonly-pilot5-protocol-v1',
        'purpose': 'five_complete_train_only_epochs_explicit_variant_of_author_recipe',
        'dataset': 'svhn', 'train_split': 'train', 'test_split': 'test',
        'extra_used': False,
        'research_change': 'train_extra -> complete official train; no held-out subset',
        'initialization_policy': 'fresh run; never warm-start from train_extra',
        'train_examples': 73257, 'test_examples': 26032, 'validation_examples': 0,
        'model': 'svhnq', 'expand': 2, 'parameters': 132954,
        'bits': s.BITS, 'planned_epochs': 100, 'pilot_stop_epoch': 5,
        'batch_size': 128, 'optimizer': 'adam', 'lr': .001, 'weight_decay': 0.0,
        'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': .1,
        'workers': 0, 'pin_memory': True, 'drop_last': False,
        'train_shuffle': True, 'test_shuffle': False,
        'seed': 20261009,
        'evaluation_policy': 'all five precisions on official test after EVERY epoch',
        'selection_policy': 'first strict maximum of upstream W32/A32 test top1',
        'reporting_policy': 'same best checkpoint at all bits AND final epoch at all bits',
        'backend_policy': 'github: cudnn.benchmark=True; no deterministic override',
        'quantizer': 'original, FP master requantization; NO MX; NO stored-code truncation',
        'rng_policy': 'global streams as upstream; saved/restored at epoch boundaries',
        'checkpoint_policy': 'immutable per epoch, plus separate best and last copies',
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
    c = config()
    source, source_info = s.legacy.prepare_source(out)
    report.update(config=c, source=source_info, package_sha256=package_hashes,
                  interpreter=sys.executable)
    s.write_json(out / 'trainonly_change_manifest.json',
                 json.loads((ROOT / 'TRAINONLY_CHANGE_MANIFEST.json').read_text(encoding='utf-8')))
    dependencies(out, log)
    rt = s.runtime(c['seed'], args.device)
    modules = s.import_source(source)
    report.update(environment=rt.environment, execution_settings=rt.settings)
    log('GPU: ' + rt.environment['device_name'])
    log('Originele backend: cudnn.benchmark=True; geen bitexacte CUDA-hervatgarantie.')
    model = None
    if check_only:
        model = modules.models.svhnq(s.BITS).to(rt.device)
        s.require(model.expand == c['expand'], 'Onverwachte modelbreedte.')
        s.require(sum(p.numel() for p in model.parameters()) == c['parameters'], 'Parametertelling wijkt af.')
        model.eval()
        # A zero tensor does not advance RNG streams. This is not SVHN training.
        with rt.torch.no_grad():
            x = rt.torch.zeros(2, 3, 40, 40, device=rt.device)
            for bit in s.BITS:
                s.legacy.select_precision(model, bit)
                y = model(x)
                s.require(tuple(y.shape) == (2, 10) and bool(rt.torch.isfinite(y).all()),
                          'GPU-vormcontrole mislukt.')
        _, forward_hash = s.load_forward(source, modules.utils, log)
        report.update(checks={'model_factory_expand2': True, 'five_gpu_forward_shapes': True,
                              'source_hashes_verified': True, 'optimizer_steps': 0,
                              'dataset_downloaded': False}, forward_ast_sha256=forward_hash)
        optional = {}
        for name in ['pandas', 'matplotlib', 'ipykernel']:
            try:
                optional[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                optional[name] = 'MISSING (only required for notebook)'
        report['notebook_dependencies'] = optional
        log('CHECK PASS: model, GPU, SciPy, versies en broncontrole. Geen training uitgevoerd.')
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
    model = modules.models.svhnq(s.BITS, train.num_classes).to(rt.device)
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
            log(f'EPOCH {epoch}/{stop_after} (origineel schema: {c["planned_epochs"]}); LR={lr:g}')
            train_rows, train_s, ast_hash = s.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['train'], epoch, 'train', lr, out / 'batch_metrics', log)
            steps += expected_steps
            s.legacy.finite_gradients(model, require_all=True)
            s.legacy.finite_state(model)
            state_before_eval = s.legacy.state_digest(model.state_dict())
            test_rows, test_s, test_ast_hash = s.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['test'], epoch, 'test', lr, out / 'batch_metrics', log)
            s.require(ast_hash == test_ast_hash, 'Train- en evaluatiefunctie zijn niet identiek.')
            s.require(s.legacy.state_digest(model.state_dict()) == state_before_eval,
                      'Evaluatie heeft modelgewichten of BatchNorm-statistieken gewijzigd.')
            coverage = {name: loader.validate(out / 'sample_order' / f'{name}_{epoch:05d}.npy')
                        for name, loader in loaders.items()}
            for rows, expected in [(train_rows, c['train_examples']), (test_rows, c['test_examples'])]:
                s.require(all(r['n'] == expected for r in rows), 'Metric bevat niet de volledige dataset.')
            bn_check = s.legacy.bn_check(model, steps)
            scheduler.step()  # EXACT original position: after train and test.
            s.require(scheduler.last_epoch == epoch, 'Scheduler-epoch wijkt af.')
            checkpoint = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            record = {
                'epoch': epoch, 'train': train_rows, 'test': test_rows,
                'lr_used': lr, 'lr_next': float(optimizer.param_groups[0]['lr']),
                'optimizer_steps': steps, 'coverage': coverage,
                'batchnorm_counters_correct': bn_check['all_counters_correct'],
                'model_sha256': state_before_eval, 'checkpoint_path': str(checkpoint.resolve()),
                'forward_ast_sha256': ast_hash, 'train_seconds': train_s, 'test_seconds': test_s,
                'first_epoch_in_process': epoch == first_epoch,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            }
            trace['epochs'].append(record)
            best = s.best_record(trace['epochs'])
            record['is_best_at_this_epoch'] = best['epoch'] == epoch
            record['best_epoch_so_far'] = best['epoch']
            t_save = time.perf_counter()
            payload = core.capture_checkpoint(model, optimizer, scheduler, {}, device, epoch, steps,
                                              ctx.contract, trace, {})
            # Empty validation_logits is intentional: scores are in trace; there
            # is NO reason to copy 5x26032x10 logits into every training checkpoint.
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
            log('  TEST ' + ' | '.join(f'W{r["bits_w_and_a"]}/A{r["bits_w_and_a"]}: '
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


def export_results(ctx, out: Path, trace: dict, log) -> None:
    c, core = ctx.c, ctx.rt.core
    records = trace['epochs']
    s.require(bool(records), 'Geen volledig afgeronde epoch.')
    best, last = s.best_record(records), records[-1]
    all_rows = [r for ep in records for phase in ['train', 'test'] for r in ep[phase]]
    s.write_csv(out / 'epoch_metrics.csv', all_rows)
    selected, checkpoint_details = [], {}
    for role, record in [('best_W32_test', best), ('last_epoch', last)]:
        path = Path(record['checkpoint_path'])
        state = core.load_verified(path)
        s.require(state['completed_epoch'] == record['epoch'], 'Onjuiste checkpoint-epoch.')
        s.require(not core.differences(state['trace']['epochs'][-1]['test'], record['test']),
                  'Rapportage komt niet overeen met checkpointmetrics.')
        s.require(s.legacy.state_digest(state['model']) == record['model_sha256'],
                  'Checkpointgewichten komen niet overeen met geevalueerd model.')
        filename = 'best_checkpoint.pt' if role == 'best_W32_test' else 'last_checkpoint.pt'
        target = out / filename
        with path.open('rb') as f, target.open('xb') as g:
            shutil.copyfileobj(f, g)
        digest = core.file_sha256(target)
        s.require(digest == core.file_sha256(path), 'Checkpointkopie wijkt af.')
        target.with_suffix('.pt.sha256').write_text(digest + '\n', encoding='ascii')
        checkpoint_details[role] = {'epoch': record['epoch'], 'file': filename, 'sha256': digest,
                                    'original_checkpoint': str(path),
                                    'selection_score_W32_A32': record['test'][-1]['top1_percent']}
        selected.extend([{'role': role, **row} for row in record['test']])
    s.write_csv(out / 'selected_metrics.csv', selected)
    paper = json.loads((ROOT / 'paper_targets.json').read_text(encoding='utf-8'))
    comparison = []
    for r in selected:
        value = paper['accuracy_percent_by_bits'][str(r['bits_w_and_a'])]
        comparison.append({'role': r['role'], 'epoch': r['epoch'], 'bits_w_and_a': r['bits_w_and_a'],
                           'our_top1_percent': r['top1_percent'], 'paper_100epoch_top1_percent': value,
                           'difference_percentage_points': r['top1_percent'] - value,
                           'comparable_training_budget': False,
                           'same_training_split': False,
                           'our_train_split': c['train_split'],
                           'paper_train_split': 'train_extra',
                           'note': 'train-only pilot; paper used train+extra for 100 epochs; context only'})
    s.write_csv(out / 'paper_comparison.csv', comparison)
    s.write_json(out / 'paper_targets.json', paper)
    selection = {'policy': c['selection_policy'], 'best': checkpoint_details['best_W32_test'],
                 'last': checkpoint_details['last_epoch'], 'same_checkpoint': best['epoch'] == last['epoch'],
                 'selected_on_official_test': True, 'independent_test_estimate': False,
                 'completed_epochs': last['epoch'], 'paper_epochs': c['planned_epochs'],
                 'train_split': c['train_split'], 'extra_used': False}
    s.write_json(out / 'checkpoint_selection.json', selection)
    s.write_json(out / 'history.json', trace)
    estimate = s.estimate_time(records, c['planned_epochs'])
    s.write_json(out / 'timing_estimate.json', estimate)
    s.write_csv(out / 'epoch_times.csv', [
        {'epoch': r['epoch'], **{k: r.get(k) for k in
         ['train_seconds', 'test_seconds', 'checkpoint_seconds', 'other_seconds', 'total_epoch_seconds',
          'first_epoch_in_process', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']}}
        for r in records])
    ctx.report.update(completed_epoch=last['epoch'], optimizer_steps=last['optimizer_steps'],
                      selected_best_epoch=best['epoch'], selected_last_epoch=last['epoch'],
                      checkpoint_selection=selection, timing_estimate=estimate,
                      synthetic_fixture_only=ctx.contract['synthetic_fixture_only'])
    notebook = json.loads((ROOT / 'SVHN_trainonly_pilot5_resultaten.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n']
    s.write_json(out / 'SVHN_trainonly_pilot5_resultaten.ipynb', notebook)
    log(f'BEHOUDEN: beste checkpoint epoch {best["epoch"]}; laatste checkpoint epoch {last["epoch"]}.')


def verify_unchanged(ctx, log):
    for name, expected in ctx.source_info['workcopy_files_sha256'].items():
        s.require(ctx.rt.core.file_sha256(ctx.source / name) == expected, 'Bronbestand gewijzigd: ' + name)
    s.verify_package()
    # Raw input verification is outside epoch time estimates.
    for name, expected in ctx.report['dataset_files']['files'].items():
        actual = s.hashes(Path(ctx.report['dataset_files']['root']) / name)
        s.require(all(actual[key] == expected[key] for key in actual), 'Ruwe dataset gewijzigd: ' + name)
    ctx.report.update(source_unchanged=True, dataset_files_unchanged=True)
    log('Eindcontrole: bronbestanden en ruwe datasetbestanden ongewijzigd.')


def feedback_bundle(out: Path):
    # Small read-only export for sharing: no model/data/source/order arrays.
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        names = ['run_report.json', 'epoch_metrics.csv', 'selected_metrics.csv', 'paper_comparison.csv',
                 'checkpoint_selection.json', 'timing_estimate.json', 'epoch_times.csv',
                 'history.json', 'dataset_manifest.json', 'source_manifest.json', 'compatibility.patch',
                 'paper_targets.json', 'pip_freeze.txt', 'pip_check.txt', 'run.log', 'warnings.log',
                 'trainonly_change_manifest.json']
        for name in names:
            path = out / name
            if path.is_file():
                z.write(path, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume'])
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('pilot_runs/anyprecision_svhn_trainonly_pilot5'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=5,
                        help='Absolute pilotgrens 1..5; verandert NIET het LR-schema.')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if not 1 <= args.stop_after_epoch <= 5:
        parser.error('Dit pakket laat alleen de pilot tot en met epoch 5 toe, niet de volledige training.')
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint; run begint altijd vanaf nul.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'anyprecision-svhn-trainonly-pilot5-report-v1', 'status': 'RUNNING',
              'command': args.command, 'started_at': s.now(), 'output_directory': str(out),
              'argv': sys.argv, 'warnings': [],
              'limitations': ['train only, NO extra: deliberate dataset variant, NOT a paper replication',
                              'pilot: first five epochs; original LR milestones 50/75/90 retained',
                              'test-selected score is not independent of checkpoint selection',
                              'authors historical seed and environment not established',
                              'cuDNN benchmark=True: bit-exact CUDA replay is not promised',
                              'no MX, truncation or hardware energy measurement']}
    log = s.legacy.Log(out / 'run.log')
    t0 = time.perf_counter()
    exit_code = 1
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        entry = {'category': category.__name__, 'message': str(message), 'file': filename, 'line': lineno}
        report['warnings'].append(entry)
        log('WARNING ' + str(message))
        with (out / 'warnings.log').open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.showwarning = warning_handler
            log('RESULTATENMAP: ' + str(out))
            ctx = create_context(args, out, report, log, check_only=args.command == 'check')
            if ctx is not None:
                trace = run_epochs(ctx, out, log, args.stop_after_epoch, args.checkpoint)
                export_results(ctx, out, trace, log)
                verify_unchanged(ctx, log)
                report['status'] = 'PASS' if report['completed_epoch'] == args.stop_after_epoch else 'STOPPED_AT_EPOCH_BOUNDARY'
            else:
                report['status'] = 'PASS'
            exit_code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        log(report['status'] + ': ' + str(exc))
        log(report['traceback'])
        if isinstance(exc, ModuleNotFoundError) and exc.name and exc.name.startswith('scipy'):
            log('SciPy ontbreekt. Installeer alleen SciPy met --no-deps; zie README.md. Torch/NumPy niet wijzigen.')
        log('Niets wordt verwijderd of overschreven. Deel run_report.json of feedback_bundle.zip.')
    finally:
        report['ended_at'] = s.now()
        report['session_wall_seconds'] = time.perf_counter() - t0
        log('EINDSTATUS: ' + report['status'])
        log('Rapport: ' + str(out / 'run_report.json'))
        log.handle.close()
        s.write_json(out / 'run_report.json', report)
        feedback_bundle(out)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
