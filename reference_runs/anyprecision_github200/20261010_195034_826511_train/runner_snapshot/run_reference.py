"""GitHub-recipe Any-Precision reference (200 epochs). See PROTOCOL.md.

Train uses all 50,000 training images; official test data are never evaluated
unless the explicit 'evaluate' command is used. Pause limits are NOT changes
to the training schedule. Every session and checkpoint has a new path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
# No torch import before setup_runtime sets the CUDA workspace configuration.
import resume_worker as tested
pilot = tested.pilot
support = tested.support
require = support.require
FILES = ['run_reference.py', 'reference_config.json', 'resume_core.py',
         'resume_worker.py', 'run_pilot.py', 'smoke_support.py', 'SOURCE_MANIFEST.json',
         'upstream/Any-Precision-DNNs-master.zip', 'github_targets.json']


def load_config() -> dict[str, Any]:
    c = json.loads((ROOT / 'reference_config.json').read_text(encoding='utf-8'))
    require(c['planned_epochs'] == 200 and c['scheduler_milestones'] == [100, 150, 180],
            'Dit pakket verwacht het GitHub-coderecept: 200 epochs en LR100/150/180.')
    require(c['train_examples'] == 50000 and c['test_examples'] == 10000 and
            c['validation_examples'] == 0, 'Onverwachte datasetsplit.')
    require(c['batch_size'] == 128 and c['bits'] == pilot.BITS and c['expand'] == 5,
            'Onverwachte model/batch/precisieconfiguratie.')
    require(c['optimizer'] == 'adam' and c['lr'] == 0.001 and c['weight_decay'] == 0.0
            and c['scheduler_gamma'] == 0.1, 'Onverwachte optimizerinstellingen.')
    require(c['workers'] == 0 and c['pin_memory'] is False and c['drop_last'] is False,
            'Onverwachte DataLoader-instellingen.')
    require(c['seed'] == 20261009 and c['exact_paper_reproduction'] is False,
            'Configuratie gewijzigd; gebruik een nieuwe expliciete protocolversie.')
    return c


def verify_package() -> dict[str, str]:
    m = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for n, expected in m.items():
        require(support.sha256(ROOT / n) == expected,
                f'Pakketbestand gewijzigd: {n}. Verander dit pakket niet tijdens een run.')
    require(all(n in m for n in FILES), 'Onvolledig pakketmanifest.')
    return {n: m[n] for n in FILES}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('x', encoding='utf-8', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sync(torch: Any, device: Any) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def prepare_training_data(data_root: Path, source: Path, config: dict, log: Any):
    dm = importlib.import_module('datasets.data')
    require(Path(dm.__file__).resolve().is_relative_to(source), 'Verkeerde datasetmodule.')
    info = pilot.ensure_dataset(data_root, dm.CIFAR10, False, log)
    dm.data_paths['cifar10'] = str(data_root)
    tf = dm.get_transform('cifar10', 'train')
    data = dm.get_dataset('cifar10', 'train', tf, download=False)
    require(len(data) == config['train_examples'] and data.num_classes == 10,
            'Niet de volledige CIFAR-10-trainingsset.')
    require([data.train_labels.count(i) for i in range(10)] == [5000] * 10,
            'Onverwachte klasseverdeling.')
    # The test file is only checksummed here; not decoded or sent to the model.
    selection = {'split': 'official_train', 'n': len(data), 'indices': 'range(0,50000)',
                 'validation_examples': 0, 'official_test_examples_used': 0}
    return pilot.IndexedSubset(data, list(range(len(data)))), selection, info, repr(tf)


def make_context(args: Any, out: Path, report: dict, log: Any):
    code_hashes = verify_package()
    config = load_config()
    report['config'] = config
    source, source_info = pilot.prepare_source(out)
    torch, np, core, device, settings, environment = tested.setup_runtime(args.device, config['seed'])
    make_model, SoftCE, get_optimizer, _ = support.import_source(source)
    log('Apparaat: ' + environment['device_name'])
    data, selection, data_info, transform = prepare_training_data(
        args.data_root.resolve(), source, config, log)
    contract = {'config': config, 'environment': environment, 'settings': settings,
                'source_files': source_info['workcopy_files_sha256'], 'code_hashes': code_hashes,
                'selection': selection, 'dataset_batch_files': data_info['batch_files'],
                'train_transform': transform,
                'synthetic_fixture_only': bool(data_info.get('synthetic_test_fixture_only', False)),
                'validation_stage': 'skipped; validation_logits is empty; no holdout in this run'}
    report.update(source=source_info, environment=environment, execution_settings=settings,
                  selection=selection, dataset_files=data_info, train_transform=transform,
                  synthetic_fixture_only=contract['synthetic_fixture_only'],
                  contract_sha256=core.json_digest(contract))
    support.json_write(out / 'config.json', config)
    support.json_write(out / 'contract.json', contract)
    support.json_write(out / 'dataset_files.json', data_info)
    support.json_write(out / 'selection.json', selection)
    # Creation may consume random numbers. Checkpoint restore occurs after all creation.
    support.reseed(config['seed'])
    model = make_model(config['bits'], 10).to(device)
    require(model.expand == 5 and sum(p.numel() for p in model.parameters()) == 6786330,
            'Architectuur wijkt af.')
    optimizer = get_optimizer(model, 'adam', config['lr'], config['weight_decay'])
    scheduler = importlib.import_module('optimizer').get_lr_scheduler(
        'adam', optimizer, config['scheduler_milestones'])
    generators = {'train': torch.Generator().manual_seed(config['seed'] + 11)}
    loader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False, generator=generators['train'])
    return dict(torch=torch, np=np, core=core, device=device, config=config, contract=contract,
                source=source, source_info=source_info, model=model, optimizer=optimizer,
                scheduler=scheduler, generators=generators, loader=loader, soft_ce=SoftCE())


def check_sources(ctx: dict) -> None:
    for n, expected in ctx['source_info']['workcopy_files_sha256'].items():
        require(support.sha256(ctx['source'] / n) == expected, 'Bronbestand gewijzigd: ' + n)
    verify_package()


def add_checkpoint(ctx: dict, path: Path, epoch: int, steps: int, trace: dict) -> dict:
    core = ctx['core']
    # The unmodified core's legacy boundary string mentions validation. In this
    # contract the validation stage is explicitly SKIPPED, with empty logits.
    payload = core.capture_checkpoint(ctx['model'], ctx['optimizer'], ctx['scheduler'],
        ctx['generators'], ctx['device'], epoch, steps, ctx['contract'], trace, {})
    entry = core.atomic_save(path, payload)
    reloaded = core.load_verified(path)
    require(not core.differences(payload, reloaded), 'Checkpoint round-trip wijkt af.')
    # Includes finite Adam momentum/variance buffers, not only model weights.
    for state in reloaded['optimizer']['state'].values():
        for v in state.values():
            if isinstance(v, ctx['torch'].Tensor) and v.is_floating_point():
                require(bool(ctx['torch'].isfinite(v).all()), 'Niet-eindige optimizerbuffer.')
    entry['full_readback_identical'] = True
    return entry


def estimate_time(timings: list[dict], done: int, planned: int) -> dict:
    # Each invocation can have its own warm-up. Exclude its first completed epoch.
    chosen = timings[1:] if len(timings) > 1 else timings
    seconds = [r['total_epoch_seconds'] for r in chosen]
    if not seconds:
        return {'available': False}
    mean = statistics.mean(seconds)
    remaining = max(0, planned - done)
    return {'available': True, 'epochs_used': [r['epoch'] for r in chosen],
            'first_epoch_this_session_excluded': len(timings) > 1,
            'mean_epoch_seconds': mean, 'min_epoch_seconds': min(seconds),
            'max_epoch_seconds': max(seconds),
            'estimated_remaining_training_hours': remaining * mean / 3600,
            'projected_full_training_hours_at_this_rate': planned * mean / 3600,
            'planned_epochs': planned,
            'note': 'Wall-clock epoch + checks + checkpoint/readback; no final test evaluation, '
                    'session restarts, future thermal drift or external-machine prediction.'}


def train_session(args: Any, out: Path, report: dict, log: Any) -> None:
    ctx = make_context(args, out, report, log)
    torch, np, core, device = (ctx[n] for n in ['torch', 'np', 'core', 'device'])
    config, model, opt, sched, generators, loader = (ctx[n] for n in
        ['config', 'model', 'optimizer', 'scheduler', 'generators', 'loader'])
    steps_per_epoch = math.ceil(len(loader.dataset) / config['batch_size'])
    require(len(loader) == steps_per_epoch, 'Aantal batches wijkt af.')
    done, steps = 0, 0
    trace = {'initial_model_sha256': support.state_digest(model.state_dict()), 'epochs': []}
    if args.resume:
        path = args.resume.resolve()
        state = core.load_verified(path)
        report['restore_checks'] = core.restore_checkpoint(state, model, opt, sched,
            generators, device, ctx['contract'])
        done, steps = int(state['completed_epoch']), int(state['optimizer_steps'])
        trace = state['trace']
        require(len(trace['epochs']) == done and steps == done * steps_per_epoch,
                'Meetgeschiedenis/aantal stappen past niet bij checkpoint.')
        require(state['validation_logits'] == {}, 'Dit is geen train-only-referentiecheckpoint.')
        report['resumed_from'] = str(path)
        report['resume_input_sha256'] = core.file_sha256(path)
        del state
        log(f'Hervat vanaf epoch {done}; volgende epoch {done + 1}.')
    else:
        log('Nieuwe initialisatie; GEEN pilot- of controlecheckpoint gebruikt.')
    require(done < config['planned_epochs'], 'Het geplande aantal epochs is al afgerond.')
    stop_at = min(config['planned_epochs'], done + args.session_epochs)
    report.update(first_epoch_this_session=done + 1, stop_after_epoch_this_session=stop_at,
                  completed_epoch=done, optimizer_steps=steps, epoch_timings=[], checkpoints=[],
                  official_test_examples_used=0, validation_examples_used=0,
                  completed_this_session=0, stopped_by_request=False)
    support.json_write(out / 'session_plan.json', {
        'planned_epochs': config['planned_epochs'], 'first_epoch_this_session': done + 1,
        'pause_after_epoch': stop_at, 'train_examples': len(loader.dataset),
        'batch_size': config['batch_size'], 'steps_per_epoch': steps_per_epoch,
        'stop_request_file': str(out / 'STOP_AFTER_EPOCH.txt'),
        'resume_input': report.get('resumed_from'), 'protocol_unchanged_by_pause': True})
    log(f"Data: {len(loader.dataset)} train; geen validatie/test tijdens deze sessie.")
    log(f"Protocol: {config['planned_epochs']} epochs; batch {config['batch_size']}; "
        f"{steps_per_epoch} batches/epoch; automatische pauze na epoch {stop_at}.")
    log(f"Learning rate: {config['lr']}; factor {config['scheduler_gamma']} na epochs "
        + ', '.join(str(m) for m in config['scheduler_milestones']) + '.')
    log('Testbeleid: vooraf vastgelegd eindpunt, geen beste-checkpointselectie op de testset.')
    log('Stopverzoek mogelijk met request-stop; sluit de terminal niet tijdens opslaan.')
    initial_checkpoint = out / 'checkpoints' / 'epoch_00000.pt'
    if done == 0:
        entry = add_checkpoint(ctx, initial_checkpoint, 0, 0, trace)
        report['initial_checkpoint'] = entry
        report['last_checkpoint'] = str(initial_checkpoint)
        log('Initialisatiecheckpoint epoch 0 opgeslagen; ook een eerste-epochfout is herstelbaar.')
    else:
        report['last_checkpoint'] = str(args.resume.resolve())
    epoch_root = out / 'epochs'
    epoch_root.mkdir()
    # Each epoch has its own append-only batch log and immutable epoch summary.
    for epoch in range(done + 1, stop_at + 1):
        if (out / 'STOP_AFTER_EPOCH.txt').exists():
            report['stopped_by_request'] = True
            break
        require(shutil.disk_usage(out).free >= 1024**3, 'Minder dan 1 GiB vrij. Laatste checkpoint behouden.')
        sync(torch, device)
        t_epoch = time.perf_counter()
        lr = float(opt.param_groups[0]['lr'])
        expected_lr = config['lr']
        for m in config['scheduler_milestones']:
            if m < epoch:
                expected_lr *= config['scheduler_gamma']
        require(lr == expected_lr, 'Learning rate wijkt af van het geplande schema.')
        acc = {b: [] for b in pilot.BITS}
        order, zero_updates = [], 0
        input_hash, label_hash = hashlib.sha256(), hashlib.sha256()
        t_train = time.perf_counter()
        batch_path = epoch_root / f'epoch_{epoch:05d}_batches.jsonl'
        with batch_path.open('x', encoding='utf-8', newline='\n') as f:
            for batch, (x, y, indices) in enumerate(loader, 1):
                ids = indices.tolist()
                order.extend(ids)
                xb, yb = x.contiguous().numpy().tobytes(), y.contiguous().numpy().tobytes()
                input_hash.update(xb)
                label_hash.update(yb)
                records = pilot.train_batch(model, opt, x.to(device), y.to(device), ctx['soft_ce'])
                steps += 1
                delta = records[-1]['watched_weight_max_abs_update']
                zero_updates += int(delta == 0.0)
                record = {'epoch': epoch, 'batch': batch, 'optimizer_step': steps, 'lr': lr,
                          'original_indices': ids, 'input_sha256': hashlib.sha256(xb).hexdigest(),
                          'precision_metrics': records}
                f.write(json.dumps(record, allow_nan=False) + '\n')
                f.flush()
                for r in records:
                    acc[r['bits_w_and_a']].append(r)
                if batch == 1 or batch % 50 == 0 or batch == steps_per_epoch:
                    ce = next(r['hard_ce'] for r in records if r['bits_w_and_a'] == 32)
                    log(f'  epoch {epoch}/{config["planned_epochs"]}, batch {batch}/{steps_per_epoch}: '
                        f'W32/A32 CE={ce:.4f}; lr={lr:.6g}; gradients eindig')
        sync(torch, device)
        training_seconds = time.perf_counter() - t_train
        t_checks = time.perf_counter()
        require(sorted(order) == list(range(len(loader.dataset))), 'Sample ontbreekt of is dubbel gebruikt.')
        require(steps == epoch * steps_per_epoch, 'Optimizerstappenteller klopt niet.')
        metrics = pilot.summarize(acc, epoch, 'train', lr)
        support.bn_check(model, steps)
        support.finite_state(model)
        sched.step()
        require(sched.last_epoch == epoch, 'Scheduler is verkeerd doorgeschoven.')
        e = {'epoch': epoch, 'train': metrics, 'validation': None,
             'validation_reason': 'not_run_no_independent_validation_set', 'test': None,
             'lr_used': lr, 'lr_next': float(opt.param_groups[0]['lr']),
             'optimizer_steps': steps, 'model_sha256': support.state_digest(model.state_dict()),
             'order_sha256': hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest(),
             'augmented_inputs_sha256': input_hash.hexdigest(), 'labels_sha256': label_hash.hexdigest(),
             'zero_updates_of_watched_weight': zero_updates,
             'all_train_samples_used_once': True, 'finite_states': True, 'batchnorm_counters_correct': True}
        trace['epochs'].append(e)
        sync(torch, device)
        checks_seconds = time.perf_counter() - t_checks
        t_ckpt = time.perf_counter()
        ckpt = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
        entry = add_checkpoint(ctx, ckpt, epoch, steps, trace)
        sync(torch, device)
        checkpoint_seconds = time.perf_counter() - t_ckpt
        # Update durable progress immediately, even if writing a JSON summary fails next.
        report.update(completed_epoch=epoch, optimizer_steps=steps,
                      completed_this_session=epoch - report['first_epoch_this_session'] + 1,
                      last_checkpoint=str(ckpt))
        timing = {'epoch': epoch, 'training_seconds': training_seconds,
                  'checks_seconds': checks_seconds, 'validation_seconds': 0.0, 'test_seconds': 0.0,
                  'checkpoint_and_readback_seconds': checkpoint_seconds,
                  'total_epoch_seconds': time.perf_counter() - t_epoch}
        report['checkpoints'].append(dict(epoch=epoch, **entry))
        report['epoch_timings'].append(timing)
        support.json_write(epoch_root / f'epoch_{epoch:05d}_summary.json',
                           {'metrics': e, 'timing': timing, 'checkpoint': entry})
        text = ' | '.join(f'W{r["bits_w_and_a"]}/A{r["bits_w_and_a"]}={r["top1_percent"]:.2f}%'
                          for r in metrics)
        log('  TRAIN top-1 (geen testaccuracy): ' + text)
        log(f'  Epoch {epoch} veilig opgeslagen: train={training_seconds:.2f}s; '
            f'checkpoint+controle={checkpoint_seconds:.2f}s; totaal={timing["total_epoch_seconds"]:.2f}s.')
        est = estimate_time(report['epoch_timings'], epoch, config['planned_epochs'])
        log(f'  Voorlopige resterende training: {est["estimated_remaining_training_hours"]:.2f} uur '
            '(exclusief eindtest; schatting op dit apparaat).')
        if (out / 'STOP_AFTER_EPOCH.txt').exists():
            report['stopped_by_request'] = True
            break
    check_sources(ctx)
    if args.resume:
        require(core.file_sha256(args.resume.resolve()) == report['resume_input_sha256'],
                'Invoercheckpoint gewijzigd.')
    report['source_unchanged'] = True
    report['test_not_evaluated'] = True
    report['runtime_estimate'] = estimate_time(report['epoch_timings'], report['completed_epoch'],
                                             config['planned_epochs'])
    report['next_epoch'] = report['completed_epoch'] + 1
    report['status'] = ('TRAINING_COMPLETE_TEST_NOT_EVALUATED'
                        if report['completed_epoch'] == config['planned_epochs'] else 'PAUSED')
    report['checks_pass'] = True
    if device.type == 'cuda':
        report['cuda_peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024**2
        report['cuda_peak_reserved_MiB'] = torch.cuda.max_memory_reserved(device) / 1024**2
    write_csv(out / 'metrics.csv', [r for e in trace['epochs'] for r in e['train']])
    write_csv(out / 'timings.csv', report['epoch_timings'])
    support.json_write(out / 'training_history.json', trace)
    if report['completed_epoch']:
        nbytes = report['checkpoints'][-1]['bytes'] if report['checkpoints'] else 0
        report['checkpoint_size_MiB'] = nbytes / 1024**2
        report['storage_projection_all_checkpoints_GiB'] = nbytes * (config['planned_epochs'] + 1) / 1024**3
    log(f"Sessie afgerond; je mag nu de laptop afsluiten. De {config['planned_epochs']}-epochplanning is niet gewijzigd.")
    log('Laatste hervatbare checkpoint: ' + report['last_checkpoint'])


def github_comparison(rows: list[dict[str, Any]], epoch: int, synthetic: bool) -> dict[str, Any]:
    """Compare one checkpoint's complete test vector with the pinned README row.

    No independent maxima per bit width, no accuracy-based pass/fail criterion.
    """
    reference = json.loads((ROOT / 'github_targets.json').read_text(encoding='utf-8'))
    require(reference['readme_sha256'] == support.sha256(ROOT / 'source_notes/upstream_README.md'),
            'README en vergelijkingsreferentie komen niet overeen.')
    require(len(rows) == 5 and sorted(r['bits_w_and_a'] for r in rows) == pilot.BITS,
            'Vergelijking vereist exact vijf precisies van hetzelfde checkpoint.')
    actual = {r['bits_w_and_a']: r for r in rows}
    compared = []
    for bit, target in zip(reference['bits'], reference['top1_percent']):
        measured = actual[bit]
        require(measured['phase'] == 'official_test' and measured['epoch'] == epoch,
                'Verkeerde datasetfase of meerdere checkpoint-epochs in vergelijking.')
        require(synthetic or measured['n'] == 10000, 'Vergelijking vereist alle 10.000 testbeelden.')
        compared.append({'epoch': epoch, 'bits_w_and_a': bit, 'test_n': measured['n'],
                         'our_top1_percent': measured['top1_percent'],
                         'github_top1_percent': target,
                         'difference_percentage_points': measured['top1_percent'] - target})
    return {'reference': reference, 'checkpoint_epoch': epoch,
            'synthetic_fixture_only': synthetic, 'valid_research_result': not synthetic,
            'comparison_has_no_accuracy_pass_threshold': True,
            'declared_evaluation_policy': 'fixed_endpoint_not_best_on_test',
            'historical_checkpoint_selection_known': False, 'rows': compared}


def evaluate_endpoint(args: Any, out: Path, report: dict, log: Any) -> None:
    ctx = make_context(args, out, report, log)
    torch, core, device, c = (ctx[n] for n in ['torch', 'core', 'device', 'config'])
    path = args.checkpoint.resolve()
    state = core.load_verified(path)
    require(state['completed_epoch'] == args.endpoint_epoch and 0 < args.endpoint_epoch <= c['planned_epochs'],
            'Het checkpoint is niet het expliciet gekozen eindpunt.')
    require(args.endpoint_reason.strip() != '', 'Leg de vooraf gekozen eindpuntreden vast.')
    report['restore_checks'] = core.restore_checkpoint(state, ctx['model'], ctx['optimizer'],
        ctx['scheduler'], ctx['generators'], device, ctx['contract'])
    report['input_checkpoint_sha256'] = core.file_sha256(path)
    report.update(evaluated_checkpoint=str(path), endpoint_epoch=args.endpoint_epoch,
                  endpoint_reason=args.endpoint_reason, endpoint_selection_not_automatically_verified=True)
    support.json_write(out / 'endpoint_declaration.json', {
        'epoch': args.endpoint_epoch, 'reason': args.endpoint_reason, 'checkpoint': str(path),
        'note': 'User-declared endpoint; this program cannot prove absence of prior test-driven selection.'})
    dm = importlib.import_module('datasets.data')
    tf = dm.get_transform('cifar10', 'val')
    data = dm.get_dataset('cifar10', 'val', tf, download=False)
    require(len(data) == c['test_examples'], 'Niet de volledige officiele testset.')
    view = pilot.IndexedSubset(data, list(range(len(data))))
    gen = torch.Generator().manual_seed(c['seed'] + 12)
    loader = torch.utils.data.DataLoader(view, batch_size=c['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, generator=gen)
    sync(torch, device)
    t0 = time.perf_counter()
    rows, logits = pilot.evaluate(ctx['model'], loader, device, args.endpoint_epoch,
                                  float(ctx['optimizer'].param_groups[0]['lr']))
    sync(torch, device)
    for r in rows:
        r['phase'] = 'official_test'
    report.update(test_seconds=time.perf_counter() - t0, test_transform=repr(tf), test_metrics=rows,
                  official_test_examples_used=len(data), validation_examples_used=0,
                  model_unchanged_by_evaluation=True, status='PASS')
    write_csv(out / 'test_metrics.csv', rows)
    comparison = github_comparison(rows, args.endpoint_epoch, report['synthetic_fixture_only'])
    # This records the declared policy; the script cannot verify historic user choices.
    comparison['endpoint_selection_not_automatically_verified'] = True
    report['github_comparison'] = comparison
    write_csv(out / 'github_comparison.csv', comparison['rows'])
    support.json_write(out / 'github_comparison.json', comparison)
    with (out / 'test_logits.pt').open('xb') as f:
        torch.save(logits, f)
    require(core.file_sha256(path) == report['input_checkpoint_sha256'], 'Checkpoint gewijzigd.')
    check_sources(ctx)
    report['source_unchanged'] = True
    report['status'] = 'PASS'
    for r in rows:
        b = r['bits_w_and_a']
        log(f'OFFICIELE TEST: epoch {args.endpoint_epoch}, W{b}/A{b}: top-1={r["top1_percent"]:.2f}%')
    for r in comparison['rows']:
        log(f'  GitHub-vergelijking W{r["bits_w_and_a"]}/A{r["bits_w_and_a"]}: '
            f'eigen={r["our_top1_percent"]:.2f}%; README={r["github_top1_percent"]:.2f}%; '
            f'verschil={r["difference_percentage_points"]:+.2f} procentpunt.')
    log('Vergelijking is met de README-rij Resnet-20-Any (recursive), niet met een gegarandeerde target.')
    log('Dit is een afzonderlijke eindpuntevaluatie; er is niet verder getraind.')


def request_stop(args: Any) -> int:
    p = args.session_dir.resolve()
    require((p / 'session_plan.json').is_file(), 'Geen gestarte trainingssessie op dit pad.')
    require(not (p / 'reference_report.json').exists(), 'Deze sessie heeft al een eindrapport.')
    target = p / 'STOP_AFTER_EPOCH.txt'
    if not target.exists():
        with target.open('x', encoding='ascii') as f:
            f.write('Please stop after the current complete epoch and verified checkpoint.\n')
    print('Stopverzoek geplaatst. Wacht op STATUS: PAUSED in de trainingsterminal.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='Start of hervat; standaard sessie van vijf epochs.')
    evaluate = sub.add_parser('evaluate', help='Evalueer een expliciet gekozen checkpoint op de officiele testset.')
    for p in (train, evaluate):
        p.add_argument('--data-root', type=Path, required=True)
        p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        p.add_argument('--output-root', type=Path, default=Path('reference_runs/anyprecision_github200'))
    train.add_argument('--session-epochs', type=int, default=5)
    train.add_argument('--resume', type=Path)
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--endpoint-epoch', type=int, required=True)
    evaluate.add_argument('--endpoint-reason', required=True)
    stop = sub.add_parser('request-stop', help='Vraag een gecontroleerde stop na de huidige epoch.')
    stop.add_argument('--session-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'request-stop':
        try:
            return request_stop(args)
        except Exception as exc:
            print('FOUT: ' + str(exc), file=sys.stderr)
            return 1
    if args.command == 'train' and args.session_epochs < 1:
        parser.error('--session-epochs moet minstens 1 zijn.')
    out = args.output_root.resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.command)
    out.mkdir(parents=True, exist_ok=False)
    log = support.Log(out / 'run.log')
    report: dict[str, Any] = {'status': 'FAIL', 'purpose': 'github_code_recipe_anyprecision_reference',
        'command': args.command, 'argv': sys.argv, 'session_directory': str(out),
        'started_at': datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
        'limitations': ['Supplied GitHub implementation/recipe retained; see GITHUB_ALIGNMENT.md.',
                        'No MX or stored-code truncation; no hardware benefit measurement.',
                        'Only epoch-boundary resume; same environment/device required.',
                        'Training metrics are online metrics with augmentation, not holdout accuracy.',
                        'Runtime includes diagnostic finite-state checks, hashing and I/O.',
                        'Test results from an earlier checkpoint are not completed 200-epoch results.',
                        'Exact historical training recipe for the GitHub README scores is not established.']}
    log('ANY-PRECISION GITHUB-REFERENTIE - 200 epochs; geen MX')
    log('Sessie-/rapportmap: ' + str(out))
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            verify_package()
            snapshot = out / 'runner_snapshot'
            snapshot.mkdir()
            # Copy the complete manifest, so the archived runner is self-contained.
            manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
            for name in manifest:
                dst = snapshot / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dst)
            shutil.copy2(ROOT / 'PACKAGE_SHA256.json', snapshot / 'PACKAGE_SHA256.json')
            if args.command == 'train':
                train_session(args, out, report, log)
            else:
                evaluate_endpoint(args, out, report, log)
        except (Exception, KeyboardInterrupt) as exc:
            report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
            report['error'] = str(exc) or type(exc).__name__
            report['traceback'] = traceback.format_exc()
            report['recovery_note'] = 'Geen mid-epochcheckpoint. Hervat uitsluitend een eerder compleet .pt + .sha256 paar.'
            log('FOUT/ONDERBREKING: ' + report['error'])
            log(report['traceback'])
        finally:
            report['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
                                  'filename': w.filename, 'line': w.lineno} for w in caught]
            freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                                    capture_output=True, text=True, check=False)
            (out / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
            report['pip_freeze'] = {'returncode': freeze.returncode, 'stderr': freeze.stderr}
            report['elapsed_session_seconds'] = time.perf_counter() - start
            report['ended_at'] = datetime.now().astimezone().isoformat()
            support.json_write(out / 'reference_report.json', report)
            log(f'Waarschuwingen: {len(report["warnings"])}; zie reference_report.json.')
            log('STATUS: ' + report['status'])
            log('Rapportmap: ' + str(out))
            log('Geen downloads. Geen pakketinstallatie. Eerdere resultaten blijven ongewijzigd.')
            log.handle.close()
    return 0 if report['status'] in ['PASS', 'PAUSED', 'TRAINING_COMPLETE_TEST_NOT_EVALUATED'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
